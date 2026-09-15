"""M2 harness smoke test: single-process PPO with truncated BPTT for a small CNN+GRU agent on
the `flybrain.env` wrapper. Hyper-parameters follow PLAN.md section 3 "Algorithm" (T=32, GAE
0.95/0.99, clip 0.2, 2 epochs, Adam 3e-4 linear decay, entropy 0.003 -> 0.001 with a hard
floor, value coef 0.5, grad clip 0.5). Both terminated and truncated episodes cut the value
bootstrap (simplification, noted).

NO TRAINING RUNS without an explicit "start". The only sanctioned run is a <= 5 min CPU smoke:

    python -m flybrain.train.baseline_cnn_gru --scenario basic --envs 4 --rollout 16 \
        --updates 4 --time-limit 240 --run smoke

Its numbers prove the loop executes (loss finite, steps done); they are never results.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flybrain import OUT_DIR
from flybrain.env.doom import HEIGHT, N_ACTIONS, WIDTH
from flybrain.env.pool import EnvPool


@dataclass
class Config:
    scenario: str = "take_cover"
    envs: int = 64
    rollout: int = 32            # T decisions per segment (truncated-BPTT window)
    epochs: int = 2
    minibatches: int = 4         # env-wise splits; each keeps the full T sequence
    updates: int = 0             # 0 -> derive from total_steps
    total_steps: int = 5_000_000
    lr: float = 3e-4
    ent_start: float = 0.003
    ent_end: float = 0.001       # hard floor
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    seed: int = 0
    sticky_prob: float = 0.0
    time_limit: float = 300.0    # seconds; smoke cap
    threads: int = 4
    ckpt_every: int = 50
    run: str = "baseline"
    collapse_entropy: float = 0.15   # nats; degeneracy monitor
    collapse_updates: int = 5
    restart_on_collapse: bool = False
    reward_scale: float = 1.0     # multiply env rewards before GAE (take_cover returns are O(100))
    env_kwargs: dict = field(default_factory=dict)


class CnnGru(nn.Module):
    """3-conv encoder on the 120x160 grey frame -> GRU(hidden) -> 7 logits + value."""

    def __init__(self, n_actions: int = N_ACTIONS, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, 1, HEIGHT, WIDTH)).shape[1]
        self.fc = nn.Linear(n_flat, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        self.hidden = hidden
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.orthogonal_(self.v.weight, gain=1.0)

    def encode(self, gray: torch.Tensor) -> torch.Tensor:
        x = gray.float().div_(255.0).unsqueeze(1)  # [B,1,H,W]
        return F.relu(self.fc(self.conv(x)))

    def step(self, gray, h, mask):
        """One decision: gray uint8 [B,H,W], h [B,hidden], mask [B] (0 where a new episode starts)."""
        h = self.gru(self.encode(gray), h * mask.unsqueeze(1))
        return self.pi(h), self.v(h).squeeze(1), h

    def unroll(self, gray_seq, h0, masks):
        """gray_seq uint8 [T,B,H,W], h0 [B,hidden], masks [T,B] -> logits [T,B,A], values [T,B]."""
        T = gray_seq.shape[0]
        feats = self.encode(gray_seq.reshape(-1, HEIGHT, WIDTH)).reshape(T, -1, self.hidden)
        h, logits, values = h0, [], []
        for t in range(T):
            h = self.gru(feats[t], h * masks[t].unsqueeze(1))
            logits.append(self.pi(h))
            values.append(self.v(h).squeeze(1))
        return torch.stack(logits), torch.stack(values), h


def gae(rewards, values, dones, last_value, gamma, lam):
    """rewards/values/dones [T,N]; returns (advantages, returns) [T,N]."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        next_v = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.set_num_threads(cfg.threads)
        self.device = torch.device("cpu")
        self.pool = EnvPool(cfg.envs, cfg.scenario, sticky_prob=cfg.sticky_prob, rgb=False,
                            labels=False, **cfg.env_kwargs)
        self.model = CnnGru(hidden=cfg.hidden)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)
        self.n_updates = cfg.updates or max(1, cfg.total_steps // (cfg.envs * cfg.rollout))
        self.out = OUT_DIR / "baseline" / cfg.run
        self.out.mkdir(parents=True, exist_ok=True)
        self.log_f = open(self.out / "log.jsonl", "a")
        self.h = torch.zeros(cfg.envs, cfg.hidden)
        self.mask = torch.ones(cfg.envs)
        obs, _ = self.pool.reset()
        self.obs = torch.from_numpy(obs["gray"].copy())
        self.steps = 0
        self.low_entropy_streak = 0
        self.restarts = 0
        self.completed: list[dict] = []

    def log(self, rec: dict) -> None:
        self.log_f.write(json.dumps(rec) + "\n")
        self.log_f.flush()

    def rollout(self):
        cfg, N, T = self.cfg, self.cfg.envs, self.cfg.rollout
        obs_buf = torch.zeros(T, N, HEIGHT, WIDTH, dtype=torch.uint8)
        act_buf = torch.zeros(T, N, dtype=torch.long)
        logp_buf = torch.zeros(T, N)
        val_buf = torch.zeros(T, N)
        rew_buf = torch.zeros(T, N)
        done_buf = torch.zeros(T, N)
        mask_buf = torch.zeros(T, N)
        h0 = self.h.clone()
        self.completed = []
        with torch.no_grad():
            for t in range(T):
                obs_buf[t] = self.obs
                mask_buf[t] = self.mask
                logits, value, self.h = self.model.step(self.obs, self.h, self.mask)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                act_buf[t], logp_buf[t], val_buf[t] = a, dist.log_prob(a), value
                obs, r, term, trunc, infos = self.pool.step(a.numpy())
                done = np.logical_or(term, trunc)
                self.obs = torch.from_numpy(obs["gray"].copy())
                rew_buf[t] = torch.from_numpy(r) * cfg.reward_scale
                done_buf[t] = torch.from_numpy(done.astype(np.float32))
                self.mask = 1.0 - done_buf[t]
                self.completed += [i["final_info"]["episode"] for i in infos if "final_info" in i]
            _, last_value, _ = self.model.step(self.obs, self.h, self.mask)
        self.steps += N * T
        adv, ret = gae(rew_buf, val_buf, done_buf, last_value, cfg.gamma, cfg.lam)
        return dict(obs=obs_buf, act=act_buf, logp=logp_buf, val=val_buf, adv=adv, ret=ret,
                    mask=mask_buf, h0=h0)

    def update(self, batch, upd: int) -> dict[str, float]:
        cfg, N = self.cfg, self.cfg.envs
        frac = 1.0 - upd / max(1, self.n_updates)
        for g in self.opt.param_groups:
            g["lr"] = cfg.lr * frac
        ent_coef = max(cfg.ent_end, cfg.ent_start + (cfg.ent_end - cfg.ent_start) * (1 - frac))
        adv = batch["adv"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        stats = {k: [] for k in ("loss", "pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac")}
        mb = max(1, N // cfg.minibatches)
        for _ in range(cfg.epochs):
            perm = torch.randperm(N)
            for s in range(0, N, mb):
                idx = perm[s:s + mb]
                logits, values, _ = self.model.unroll(batch["obs"][:, idx], batch["h0"][idx], batch["mask"][:, idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(batch["act"][:, idx])
                ratio = torch.exp(logp - batch["logp"][:, idx])
                a = adv[:, idx]
                pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
                vl = F.mse_loss(values, batch["ret"][:, idx])
                ent = dist.entropy().mean()
                loss = pg + cfg.vf_coef * vl - ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    stats["loss"].append(loss.item())
                    stats["pg_loss"].append(pg.item())
                    stats["v_loss"].append(vl.item())
                    stats["entropy"].append(ent.item())
                    stats["approx_kl"].append((batch["logp"][:, idx] - logp).mean().item())
                    stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        out["lr"], out["ent_coef"] = cfg.lr * frac, ent_coef
        return out

    def degeneracy_check(self, batch, stats) -> dict[str, Any]:
        """Anti-degeneracy monitor (PLAN): action histogram + entropy floor + restart hook."""
        hist = torch.bincount(batch["act"].reshape(-1), minlength=N_ACTIONS).float()
        hist = (hist / hist.sum()).tolist()
        self.low_entropy_streak = self.low_entropy_streak + 1 if stats["entropy"] < self.cfg.collapse_entropy else 0
        collapsed = self.low_entropy_streak >= self.cfg.collapse_updates
        if collapsed and self.cfg.restart_on_collapse:
            self.model = CnnGru(hidden=self.cfg.hidden)
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, eps=1e-5)
            self.h.zero_()
            self.low_entropy_streak = 0
            self.restarts += 1
        return {"action_hist": hist, "max_action_frac": max(hist), "collapsed": collapsed,
                "restarts": self.restarts}

    def save(self, name: str = "ckpt.pt") -> Path:
        path = self.out / name
        torch.save({"model": self.model.state_dict(), "cfg": asdict(self.cfg), "steps": self.steps}, path)
        return path

    def train(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        last = {}
        upd = 0
        for upd in range(self.n_updates):
            batch = self.rollout()
            stats = self.update(batch, upd)
            stats.update(self.degeneracy_check(batch, stats))
            elapsed = time.perf_counter() - t0
            rec = {"update": upd + 1, "steps": self.steps, "elapsed_s": elapsed,
                   "steps_per_s": self.steps / elapsed, "episodes_done": len(self.completed)}
            if self.completed:  # logged for curves only; never a reported result of a smoke run
                rec["ep_reward_mean"] = float(np.mean([e["reward"] for e in self.completed]))
                rec["ep_len_mean"] = float(np.mean([e["survival_tics"] for e in self.completed]))
            rec.update(stats)
            self.log(rec)
            last = rec
            print(f"upd {upd + 1}/{self.n_updates} steps {self.steps} loss {stats['loss']:.4f} "
                  f"ent {stats['entropy']:.3f} kl {stats['approx_kl']:.4f} "
                  f"max_act {stats['max_action_frac']:.2f} {self.steps / elapsed:.0f} steps/s", flush=True)
            if (upd + 1) % self.cfg.ckpt_every == 0:
                self.save()
            if elapsed > self.cfg.time_limit:
                print(f"time limit {self.cfg.time_limit}s reached", flush=True)
                break
        self.save()
        self.pool.close()
        self.log_f.close()
        return {"updates_done": upd + 1 if self.n_updates else 0, "steps_done": self.steps,
                "last_loss": last.get("loss"), "last_entropy": last.get("entropy"),
                "loss_finite": bool(last and np.isfinite(last["loss"])),
                "elapsed_s": time.perf_counter() - t0, "out_dir": str(self.out)}


class ModelPolicy:
    """Plug a trained CnnGru into `flybrain.env.floors.run_episode`/`evaluate` (single process)."""

    name = "cnn_gru"

    def __init__(self, model: CnnGru, greedy: bool = False):
        self.model, self.greedy = model.eval(), greedy

    def reset(self, env, rng):
        self.h = torch.zeros(1, self.model.hidden)
        self.rng = rng

    def act(self, obs, info):
        with torch.no_grad():
            logits, _, self.h = self.model.step(torch.from_numpy(obs["gray"])[None], self.h, torch.ones(1))
            if self.greedy:
                return int(logits.argmax())
            p = torch.softmax(logits[0], 0).numpy().astype(np.float64)
            return int(self.rng.choice(N_ACTIONS, p=p / p.sum()))


def load_policy(path: str | os.PathLike, greedy: bool = False) -> ModelPolicy:
    ck = torch.load(path, map_location="cpu")
    m = CnnGru(hidden=ck["cfg"]["hidden"])
    m.load_state_dict(ck["model"])
    return ModelPolicy(m, greedy)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="CNN+GRU PPO baseline (code only; smoke runs <= 5 min CPU)")
    for f in Config.__dataclass_fields__.values():
        if f.name == "env_kwargs":
            continue
        if f.type == "bool":
            p.add_argument(f"--{f.name.replace('_', '-')}", action="store_true")
        else:
            p.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    p.add_argument("--eval-episodes", type=int, default=0, help="evaluate the saved policy on eval seeds")
    a = p.parse_args(argv)
    cfg = Config(**{k: v for k, v in vars(a).items() if k in Config.__dataclass_fields__})
    if cfg.time_limit > 300 and os.environ.get("FLYBRAIN_TRAINING_APPROVED") != "1":
        raise SystemExit("time_limit > 300 s is a training run: needs FLYBRAIN_TRAINING_APPROVED=1 (owner's 'start')")
    tr = Trainer(cfg)
    res = tr.train()
    print(json.dumps(res))
    if a.eval_episodes > 0:
        from flybrain.env.floors import POLICIES, evaluate
        POLICIES["cnn_gru"] = lambda: load_policy(tr.out / "ckpt.pt")
        r = evaluate("cnn_gru", cfg.scenario, a.eval_episodes)
        with open(tr.out / "eval.json", "w") as f:
            json.dump(r, f, indent=1)
        print("eval written to", tr.out / "eval.json")


if __name__ == "__main__":
    main()
