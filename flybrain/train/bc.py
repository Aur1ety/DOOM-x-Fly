"""Behaviour cloning of a teacher into the connectome agent (frozen front end -> ConnectomeCore -> linear head).

    record   play a trained CNN+GRU teacher (baseline_cnn_gru checkpoint) with an env pool and store
             frames + teacher logits per decision (optionally with epsilon-random actions for coverage).
    train    truncated-BPTT distillation: KL(teacher || student) over windows of T decisions with
             per-decision gradient checkpointing; the core's Dale signs are fixed, magnitudes / tau /
             bias train (per_edge) or the type-tied scalars (type_tied); the readout head is linear on
             the rates (and first differences) of a chosen population.
    play     closed-loop evaluation of a student checkpoint on held-out seeds, with --blind
             (black / mean-luminance / time-scrambled frames) for the reactivity check, and --video.

    python -m flybrain.train.bc record --teacher $FLYBRAIN_OUT/baseline/m2_take_cover_s0/ckpt.pt --decisions 60000 --out $FLYBRAIN_OUT/bc/teacher_tc.npz
    python -m flybrain.train.bc train --data $FLYBRAIN_OUT/bc/teacher_tc.npz --subgraph ... --run bc_tc_edge
    python -m flybrain.train.bc play --ckpt $FLYBRAIN_OUT/bc/bc_tc_edge/student.pt --episodes 50 [--blind black] [--video out.mp4]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from flybrain import OUT_DIR
from flybrain.env.doom import ACTIONS, EVAL_SEED_LO, N_ACTIONS
from flybrain.model.core import ConnectomeCore, CoreConfig, CoreState, load_subgraph

DEFAULT_GEOMETRY = OUT_DIR / "vision" / "geometry_v1.npz"
DEFAULT_NEURONS = OUT_DIR / "graph" / "neurons.parquet"


# ----------------------------------------------------------------------------- record (teacher)

def record(teacher_ckpt: Path, scenario: str, n_decisions: int, envs: int, seed_base: int, eps: float,
           out: Path, threads: int = 8) -> dict:
    """Run the CNN+GRU teacher in an env pool; store gray/rgb frames, teacher logits, taken action."""
    from flybrain.env.pool import EnvPool
    from flybrain.train.baseline_cnn_gru import load_policy

    torch.set_num_threads(threads)
    pol = load_policy(teacher_ckpt)
    model = pol.model
    pool = EnvPool(envs, scenario, rgb=True, labels=False)
    obs, _ = pool.reset()
    h = torch.zeros(envs, model.hidden); mask = torch.ones(envs)
    ep_id = np.arange(envs); next_ep = envs; step_in_ep = np.zeros(envs, np.int64)
    rng = np.random.default_rng(seed_base)
    G, R, L, A, E, S, RW = [], [], [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        while len(G) * envs < n_decisions:
            gray = torch.from_numpy(obs["gray"].copy())
            logits, _, h = model.step(gray, h, mask)
            a = torch.distributions.Categorical(logits=logits).sample().numpy()
            rnd = rng.random(envs) < eps
            a = np.where(rnd, rng.integers(0, N_ACTIONS, envs), a)
            obs2, r, term, trunc, infos = pool.step(a)
            done = np.logical_or(term, trunc)
            G.append(obs["gray"].copy()); R.append(obs["rgb"].reshape(envs, 60, 2, 80, 2, 3).mean((2, 4)).astype(np.uint8))
            L.append(logits.numpy().astype(np.float16)); A.append(a.astype(np.int8)); E.append(ep_id.copy()); S.append(step_in_ep.copy()); RW.append(r.astype(np.float32))
            step_in_ep += 1
            for i in np.flatnonzero(done):
                ep_id[i] = next_ep; next_ep += 1; step_in_ep[i] = 0
            mask = torch.from_numpy((~done).astype(np.float32))
            h = h * mask[:, None]
            obs = obs2
    pool.close()
    G, R, L, A, E, S, RW = (np.concatenate(x) for x in (G, R, L, A, E, S, RW))
    n = min(len(G), n_decisions)
    order = np.lexsort((S, E))                       # group by episode, in time order
    meta = {"scenario": scenario, "teacher": str(teacher_ckpt), "n_decisions": int(n), "n_episodes": int(len(np.unique(E))),
            "envs": envs, "eps": eps, "seed_base": seed_base, "record_s": time.time() - t0}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, gray=G[order], rgb=R[order], logits=L[order], action=A[order], episode=E[order].astype(np.int32),
             step=S[order].astype(np.int32), reward=RW[order], meta=json.dumps(meta))
    return meta


# ----------------------------------------------------------------------------- student

class Student(nn.Module):
    """Frozen FrontEnd -> ConnectomeCore (trainable) -> linear head on a population's rates (+ diffs)."""

    def __init__(self, sub: dict, geometry: Path, neurons: Path, param: str, readout: str, device: str,
                 K: int = 6, w0: float | None = None, op_point: str = "rest", seed: int = 0, head: str = "linear",
                 n_actions: int = N_ACTIONS):
        super().__init__()
        from flybrain.model.gain import match_gain
        from flybrain.vision.frontend import FrontEnd
        from flybrain.vision.geometry import Geometry

        self.device = device
        self.geom = Geometry.load(geometry)
        self.fe = FrontEnd(self.geom, K=K, device=device)
        self.core = ConnectomeCore(sub, CoreConfig(param=param, K=K), backend="spmm", device=device)
        body = np.asarray(sub["bodyId"]); is_cl = np.asarray(sub["is_clamped"], bool)
        self.register_buffer("perm", torch.as_tensor(self.fe.index.permutation_for(body[is_cl]), device=device))
        if w0 is None:
            gm = match_gain(self.core, g_star=0.95, op_point=op_point, seed=seed)
            self.w0 = float(gm["w0"])
        else:
            self.core.set_w0(w0); self.w0 = float(w0)
        self.readout = readout
        self.register_buffer("nodes", torch.as_tensor(self._node_set(sub, neurons, readout), device=device))
        d = 2 * int(self.nodes.numel())
        self.head_type = head
        if head == "mlp":
            self.head = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, n_actions)).to(device)
        else:
            self.head = nn.Linear(d, n_actions).to(device)
            nn.init.zeros_(self.head.bias); nn.init.normal_(self.head.weight, std=1.0 / np.sqrt(d))
        self.value = nn.Linear(d, 1).to(device)
        self.register_buffer("feat_mu", torch.zeros(d, device=device))
        self.register_buffer("feat_sd", torch.ones(d, device=device))
        self.K, self.n_actions = K, n_actions

    @staticmethod
    def _node_set(sub: dict, neurons: Path, readout: str) -> np.ndarray:
        if readout == "readout36":
            return np.flatnonzero(np.asarray(sub["is_output"], bool))
        import pandas as pd
        n = pd.read_parquet(neurons, columns=["bodyId", "superclass"])
        sc = dict(zip(n["bodyId"].to_numpy().tolist(), n["superclass"].astype(str).to_numpy().tolist()))
        body = np.asarray(sub["bodyId"]); dyn = ~np.asarray(sub["is_clamped"], bool)
        scl = np.array([sc.get(int(b), "") for b in body])
        want = {"dn": ("descending_neuron",), "vpn": ("visual_projection",), "dn_vpn": ("descending_neuron", "visual_projection")}[readout]
        return np.flatnonzero(dyn & np.isin(scl, want))

    # -- state ------------------------------------------------------------------------------
    def init_state(self, B: int):
        return {"fe": self.fe.init_state(B), "core": self.core.init_state(B),
                "prev": torch.zeros(B, self.nodes.numel(), device=self.device)}

    def reset_rows(self, state, mask: np.ndarray):
        if not mask.any():
            return state
        self.fe.reset(state["fe"], mask)
        m = torch.as_tensor(mask, device=self.device)
        init = self.core.init_state(len(mask))
        c = state["core"]
        state["core"] = CoreState(h=torch.where(m[None, :], init.h, c.h),
                                  x_clamped=torch.where(m[None, :], init.x_clamped, c.x_clamped),
                                  r_out_prev=torch.where(m[None, :], init.r_out_prev, c.r_out_prev))
        state["prev"] = torch.where(m[:, None], torch.zeros_like(state["prev"]), state["prev"])
        return state

    def detach_state(self, state):
        c = state["core"]
        state["core"] = CoreState(c.h.detach(), c.x_clamped.detach(), c.r_out_prev.detach())
        state["prev"] = state["prev"].detach()
        return state

    # -- one decision -----------------------------------------------------------------------
    def frontend(self, gray, rgb, state) -> torch.Tensor:
        with torch.no_grad():
            rates, state["fe"] = self.fe.process(gray, rgb, state["fe"])
        return rates.index_select(2, self.perm)                       # [K, B, n_clamped]

    def core_step(self, state, xc: torch.Tensor, ckpt: bool):
        c = state["core"]
        if ckpt and torch.is_grad_enabled():
            def f(h, x_clamped, r_prev, x):
                return tuple(self.core.step(CoreState(h, x_clamped, r_prev), x))
            state["core"] = CoreState(*checkpoint(f, c.h, c.x_clamped, c.r_out_prev, xc, use_reentrant=False))
        else:
            state["core"] = self.core.step(c, xc)
        r = self.core.rate(state["core"].h).index_select(0, self.nodes).T          # [B, n_nodes]
        feats = torch.cat([r, r - state["prev"]], 1)
        state["prev"] = r
        return (feats - self.feat_mu) / self.feat_sd

    def decide(self, gray, rgb, state, ckpt: bool = False) -> torch.Tensor:
        """Frames for one decision -> action logits [B, n_actions]."""
        xc = self.frontend(gray, rgb, state)
        feats = self.core_step(state, xc, ckpt)
        return self.head(feats)

    def panel(self, state) -> np.ndarray:
        return (state["prev"][0] / 1.5).clamp(0, 1).detach().cpu().numpy()      # typical rates ~0.7; r_max=5 is far off-scale

    @torch.no_grad()
    def fit_feature_stats(self, gray, rgb, episode, batch: int = 64, max_decisions: int = 12000) -> None:
        """Set feat_mu / feat_sd from the current core's features over a slice of the data (frozen pass)."""
        self.feat_mu.zero_(); self.feat_sd.fill_(1.0)
        n = min(max_decisions, len(episode))
        seqs, starts = _streams(episode[:n], batch)
        state = self.init_state(batch); acc, acc2, cnt = 0.0, 0.0, 0
        for t in range(max(len(s) for s in seqs)):
            idx = np.array([s[t] if t < len(s) else (s[-1] if len(s) else 0) for s in seqs]); live = torch.as_tensor([t < len(s) for s in seqs], device=self.device)
            state = self.reset_rows(state, np.array([bool(r[t]) if t < len(r) else False for r in starts]))
            f = self.core_step(state, self.frontend(gray[idx], rgb[idx], state), False)     # standardised with the neutral stats
            f = f[live]
            acc = acc + f.sum(0); acc2 = acc2 + (f * f).sum(0); cnt += int(live.sum())
        mu = acc / max(cnt, 1); var = acc2 / max(cnt, 1) - mu * mu
        self.feat_mu.copy_(mu); self.feat_sd.copy_((var.clamp_min(0) + 1e-6).sqrt())

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save(self, path: Path, extra: dict) -> None:
        torch.save({"core": self.core.state_dict(), "head": self.head.state_dict(), "value": self.value.state_dict(),
                    "w0": self.w0, "readout": self.readout, "param": self.core.cfg.param, "K": self.K, "head_type": self.head_type,
                    "feat_mu": self.feat_mu, "feat_sd": self.feat_sd, "n_actions": self.n_actions, **extra}, path)


def load_student(ckpt: Path, subgraph: Path, device: str, geometry=DEFAULT_GEOMETRY, neurons=DEFAULT_NEURONS) -> Student:
    ck = torch.load(ckpt, map_location=device)
    sub = load_subgraph(subgraph)
    st = Student(sub, geometry, neurons, ck["param"], ck["readout"], device, K=ck["K"], w0=ck["w0"], head=ck.get("head_type", "linear"),
                 n_actions=ck.get("n_actions", N_ACTIONS))
    st.core.load_state_dict(ck["core"]); st.head.load_state_dict(ck["head"]); st.value.load_state_dict(ck["value"])
    if "feat_mu" in ck:
        st.feat_mu.copy_(ck["feat_mu"].to(device)); st.feat_sd.copy_(ck["feat_sd"].to(device))
    return st


# ----------------------------------------------------------------------------- train (BC)

def _streams(episode: np.ndarray, batch: int):
    ids = np.unique(episode)
    seqs, starts = [[] for _ in range(batch)], [[] for _ in range(batch)]
    for i, e in enumerate(ids):
        idx = np.flatnonzero(episode == e); s = i % batch
        seqs[s].append(idx); starts[s].append(np.r_[True, np.zeros(len(idx) - 1, bool)])
    return ([np.concatenate(x) if x else np.zeros(0, np.int64) for x in seqs],
            [np.concatenate(x) if x else np.zeros(0, bool) for x in starts])


def _load_datasets(paths):
    """Concatenate BC datasets (gray, rgb, logits, episode) with disjoint episode ids."""
    G, R, L, E, off = [], [], [], [], 0
    for p in paths:
        z = np.load(p, allow_pickle=False)
        G.append(z["gray"]); R.append(z["rgb"]); L.append(z["logits"].astype(np.float32))
        e = z["episode"].astype(np.int64); E.append(e + off); off += int(e.max()) + 1
    return np.concatenate(G), np.concatenate(R), np.concatenate(L), np.concatenate(E)


def train(data: Path, subgraph: Path, run: str, param: str, readout: str, device: str, epochs: int, batch: int, T: int,
          lr: float, drift: float, seed: int, geometry=DEFAULT_GEOMETRY, neurons=DEFAULT_NEURONS, resume: Path | None = None,
          time_limit: float = 1e9, head: str = "linear", lr_head: float | None = None, reset_prob: float = 0.0,
          freeze_core: bool = False, balance: bool = False) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    gray, rgb, logits, episode = _load_datasets(data if isinstance(data, (list, tuple)) else [data])
    teacher_p = torch.as_tensor(np.exp(logits - logits.max(1, keepdims=True)), device=device)
    teacher_p = teacher_p / teacher_p.sum(1, keepdim=True)
    # optional class balancing: weight each decision by 1 / frequency of the teacher's argmax action
    cls = teacher_p.argmax(1); freq = torch.bincount(cls, minlength=teacher_p.shape[1]).float() / len(cls)
    row_w = (1.0 / freq.clamp_min(1e-3))[cls] if balance else torch.ones(len(cls), device=device)
    row_w = row_w / row_w.mean()
    sub = load_subgraph(subgraph)
    st = load_student(resume, subgraph, device, geometry, neurons) if resume else Student(sub, geometry, neurons, param, readout, device, seed=seed, head=head, n_actions=teacher_p.shape[1])
    out = OUT_DIR / "bc" / run; out.mkdir(parents=True, exist_ok=True)
    if not resume:
        st.fit_feature_stats(gray, rgb, episode, batch)
        print(f"feature stats: mu range [{float(st.feat_mu.min()):.3f}, {float(st.feat_mu.max()):.3f}] sd range [{float(st.feat_sd.min()):.4f}, {float(st.feat_sd.max()):.3f}]", flush=True)
    head_params = list(st.head.parameters()) + list(st.value.parameters())
    core_params = [p for p in st.core.parameters() if p.requires_grad]
    if freeze_core:
        for p in core_params: p.requires_grad_(False)
        core_params = []
    groups = ([{"params": core_params, "lr": lr}] if core_params else []) + [{"params": head_params, "lr": lr_head or 1e-2}]
    opt = torch.optim.Adam(groups, eps=1e-5)
    log_f = open(out / "log.jsonl", "a")
    # held-out episodes for the agreement metric
    ids = np.unique(episode); rng = np.random.default_rng(seed); rng.shuffle(ids)
    test_ids = set(ids[: max(1, len(ids) // 10)].tolist())
    is_test = np.isin(episode, list(test_ids))
    seqs, starts = _streams(episode[~is_test], batch)
    idx_map = np.flatnonzero(~is_test)
    seqs = [idx_map[s] for s in seqs]
    Tmax = max(len(s) for s in seqs)
    print(f"data {len(gray)} decisions, {len(ids)} episodes ({len(test_ids)} held out); streams {batch} x <= {Tmax}; "
          f"param {param} readout {readout} ({st.nodes.numel()} nodes) trainable {sum(p.numel() for p in st.trainable()):,} w0 {st.w0:.5f}", flush=True)
    t0 = time.time(); step = 0
    for ep in range(epochs):
        state = st.init_state(batch)
        for w0 in range(0, Tmax, T):
            state = st.detach_state(state)
            losses, agree, n_valid = [], 0.0, 0
            for t in range(w0, min(w0 + T, Tmax)):
                idx = np.array([s[t] if t < len(s) else (s[-1] if len(s) else 0) for s in seqs])
                live = np.array([t < len(s) for s in seqs])
                reset = np.array([bool(r[t]) if t < len(r) else False for r in starts])
                if reset_prob > 0:                       # random resets: the student cannot rely on long internal dynamics
                    reset = reset | (np.random.random(len(reset)) < reset_prob)
                state = st.reset_rows(state, reset)
                lg = st.decide(gray[idx], rgb[idx], state, ckpt=True)
                tp = teacher_p[idx]
                kl = (tp * (torch.log(tp + 1e-8) - F.log_softmax(lg, 1))).sum(1) * row_w[idx]
                lv = torch.as_tensor(live, device=device)
                losses.append((kl * lv).sum() / lv.sum().clamp_min(1))
                agree += float(((lg.argmax(1) == tp.argmax(1)) & lv).sum()); n_valid += int(lv.sum())
            loss = torch.stack(losses).mean()
            pen = st.core.drift_penalty() if (drift > 0 and param == "per_edge" and core_params) else torch.zeros((), device=device)
            opt.zero_grad(set_to_none=True)
            (loss + drift * pen).backward()
            gn = float(nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0))
            opt.step(); step += 1
            rec = {"epoch": ep, "window": w0 // T, "step": step, "kl": float(loss), "agree": agree / max(n_valid, 1),
                   "drift_pen": float(pen), "grad_norm": gn, "elapsed_s": time.time() - t0, "dale_ok": bool(st.core.dale_ok()) if step % 20 == 0 else None}
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            if step % 10 == 0:
                print(f"ep {ep} win {w0 // T} step {step} kl {float(loss):.4f} agree {rec['agree']:.3f} gn {gn:.2f} {rec['elapsed_s']:.0f}s", flush=True)
            if time.time() - t0 > time_limit:
                break
        # held-out agreement (teacher-forcing)
        ha = _heldout_agreement(st, gray, rgb, teacher_p, episode, np.flatnonzero(is_test), batch)
        print(f"== epoch {ep} done: held-out agreement {ha:.3f} ({time.time() - t0:.0f}s)", flush=True)
        log_f.write(json.dumps({"epoch_end": ep, "heldout_agree": ha}) + "\n"); log_f.flush()
        st.save(out / "student.pt", {"epoch": ep, "heldout_agree": ha, "subgraph": str(subgraph), "data": str(data)})
        if time.time() - t0 > time_limit:
            break
    log_f.close()
    return {"out": str(out), "epochs": ep + 1, "heldout_agree": ha, "elapsed_s": time.time() - t0}


@torch.no_grad()
def _heldout_agreement(st: Student, gray, rgb, teacher_p, episode, test_rows, batch) -> float:
    seqs, starts = _streams(episode[test_rows], batch)
    seqs = [test_rows[s] for s in seqs]
    state = st.init_state(batch); agree = 0.0; n = 0
    for t in range(max(len(s) for s in seqs)):
        idx = np.array([s[t] if t < len(s) else (s[-1] if len(s) else 0) for s in seqs]); live = np.array([t < len(s) for s in seqs])
        state = st.reset_rows(state, np.array([bool(r[t]) if t < len(r) else False for r in starts]))
        lg = st.decide(gray[idx], rgb[idx], state)
        agree += float(((lg.argmax(1) == teacher_p[idx].argmax(1)).cpu().numpy() & live).sum()); n += int(live.sum())
    return agree / max(n, 1)



# ----------------------------------------------------------------------------- dagger

def dagger_record(student_ckpt: Path, subgraph: Path, teacher_ckpt: Path, scenario: str, n_decisions: int, envs: int,
                  seed_base: int, beta: float, out: Path, device: str, greedy: bool = False, threads: int = 8) -> dict:
    """Roll out the STUDENT (teacher action with prob beta) and store frames + TEACHER logits at the visited states."""
    from flybrain.env.pool import EnvPool
    from flybrain.train.baseline_cnn_gru import load_policy

    torch.set_num_threads(threads)
    st = load_student(student_ckpt, subgraph, device)
    teacher = load_policy(teacher_ckpt).model
    pool = EnvPool(envs, scenario, rgb=True, labels=False)
    obs, _ = pool.reset()
    h = torch.zeros(envs, teacher.hidden); mask = torch.ones(envs)
    state = st.init_state(envs)
    ep_id = np.arange(envs); next_ep = envs; step_in_ep = np.zeros(envs, np.int64)
    rng = np.random.default_rng(seed_base)
    G, R, L, A, E, S, RW = [], [], [], [], [], [], []
    t0 = time.time(); ep_rewards = np.zeros(envs); finished = []
    with torch.no_grad():
        while len(G) * envs < n_decisions:
            gray = torch.from_numpy(obs["gray"].copy())
            t_logits, _, h = teacher.step(gray, h, mask)
            s_logits = st.decide(obs["gray"], obs["rgb"], state)
            if greedy:
                a_s = s_logits.argmax(1).cpu().numpy()
            else:
                a_s = torch.distributions.Categorical(logits=s_logits).sample().cpu().numpy()
            a_t = torch.distributions.Categorical(logits=t_logits).sample().numpy()
            a = np.where(rng.random(envs) < beta, a_t, a_s)
            obs2, r, term, trunc, infos = pool.step(a)
            done = np.logical_or(term, trunc)
            G.append(obs["gray"].copy()); R.append(obs["rgb"].reshape(envs, 60, 2, 80, 2, 3).mean((2, 4)).astype(np.uint8))
            L.append(t_logits.numpy().astype(np.float16)); A.append(a.astype(np.int8)); E.append(ep_id.copy()); S.append(step_in_ep.copy()); RW.append(r.astype(np.float32))
            step_in_ep += 1; ep_rewards += r
            for i in np.flatnonzero(done):
                finished.append(float(ep_rewards[i])); ep_rewards[i] = 0.0
                ep_id[i] = next_ep; next_ep += 1; step_in_ep[i] = 0
            state = st.reset_rows(state, done)
            mask = torch.from_numpy((~done).astype(np.float32)); h = h * mask[:, None]
            obs = obs2
    pool.close()
    G, R, L, A, E, S, RW = (np.concatenate(x) for x in (G, R, L, A, E, S, RW))
    order = np.lexsort((S, E))
    meta = {"scenario": scenario, "student": str(student_ckpt), "teacher": str(teacher_ckpt), "beta": beta, "greedy_student": greedy,
            "n_decisions": int(len(G)), "n_episodes": int(len(np.unique(E))), "student_episode_reward_mean": float(np.mean(finished)) if finished else None,
            "n_finished_episodes": len(finished), "record_s": time.time() - t0}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, gray=G[order], rgb=R[order], logits=L[order], action=A[order], episode=E[order].astype(np.int32),
             step=S[order].astype(np.int32), reward=RW[order], meta=json.dumps(meta))
    return meta


# ----------------------------------------------------------------------------- play (closed loop)

class StudentPolicy:
    """Closed-loop policy for flybrain.env.floors.run_episode / eval.record_video (single env)."""

    name = "connectome"

    def __init__(self, st: Student, greedy: bool = True, blind: str | None = None, seed: int = 0):
        self.st, self.greedy, self.blind = st, greedy, blind
        self.rng = np.random.default_rng(seed)

    def reset(self, env, rng):
        self.state = self.st.init_state(1); self.rng = rng; self.scramble = []

    def act(self, obs, info):
        gray, rgb = obs["gray"][None], obs["rgb"][None]
        if self.blind == "black":
            gray = np.zeros_like(gray); rgb = np.zeros_like(rgb)
        elif self.blind == "mean":
            gray = np.full_like(gray, int(gray.mean())); rgb = np.full_like(rgb, int(rgb.mean()))
        elif self.blind == "scramble":                    # frames from this episode, random order
            self.scramble.append((obs["gray"], obs["rgb"]))
            g, r = self.scramble[self.rng.integers(len(self.scramble))]
            gray, rgb = g[None], r[None]
        with torch.no_grad():
            lg = self.st.decide(gray, rgb, self.state)[0]
        if self.greedy:
            return int(lg.argmax())
        p = torch.softmax(lg, 0).cpu().numpy().astype(np.float64)
        return int(self.rng.choice(len(p), p=p / p.sum()))

    def panel(self):
        return self.st.panel(self.state)


def play(ckpt: Path, subgraph: Path, scenario: str, episodes: int, device: str, greedy: bool, blind: str | None,
         video: Path | None, seed_base: int = EVAL_SEED_LO) -> dict:
    from flybrain.env.doom import DoomEnv
    from flybrain.env.floors import run_episode, summarize

    st = load_student(ckpt, subgraph, device)
    pol = StudentPolicy(st, greedy=greedy, blind=blind)
    if video is not None:
        from flybrain.eval.record_video import record as rec_video
        res = rec_video(scenario, "connectome", episodes, video, policy=pol, seed_base=seed_base)
        eps = res["episodes"]
    else:
        env = DoomEnv(scenario, frameskip=4, all_frames=False, rgb=True, labels=True)
        eps = [run_episode(env, pol, seed_base + e) for e in range(episodes)]
        env.close()
    rew = np.array([e["reward"] for e in eps]); surv = np.array([e["survival_tics"] for e in eps]); kills = np.array([e["kills"] for e in eps])
    hist = np.sum([e["action_hist"] for e in eps], 0); hist = (hist / hist.sum()).round(3).tolist()
    out = {"ckpt": str(ckpt), "scenario": scenario, "episodes": episodes, "greedy": greedy, "blind": blind,
           "reward": summarize(rew), "survival_tics": summarize(surv), "kills": summarize(kills),
           "action_hist": dict(zip(ACTIONS, hist))}
    print(json.dumps({k: (v if not isinstance(v, dict) or "mean" not in v else {kk: v[kk] for kk in ("mean", "iqm", "iqm_ci95")}) for k, v in out.items()}, default=float))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("record")
    r.add_argument("--teacher", type=Path, required=True); r.add_argument("--scenario", default="take_cover")
    r.add_argument("--decisions", type=int, default=60_000); r.add_argument("--envs", type=int, default=32)
    r.add_argument("--seed-base", type=int, default=30_000); r.add_argument("--eps", type=float, default=0.1)
    r.add_argument("--out", type=Path, required=True)
    t = sp.add_parser("train")
    t.add_argument("--data", type=Path, nargs="+", required=True); t.add_argument("--subgraph", type=Path, required=True)
    t.add_argument("--run", required=True); t.add_argument("--param", default="per_edge", choices=["per_edge", "type_tied"])
    t.add_argument("--readout", default="dn", choices=["readout36", "dn", "dn_vpn"]); t.add_argument("--device", default="cuda:0")
    t.add_argument("--epochs", type=int, default=6); t.add_argument("--batch", type=int, default=64); t.add_argument("--T", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-4); t.add_argument("--drift", type=float, default=1e-4); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", type=Path, default=None); t.add_argument("--time-limit", type=float, default=1e9)
    t.add_argument("--head", default="linear", choices=["linear", "mlp"]); t.add_argument("--lr-head", type=float, default=None)
    t.add_argument("--reset-prob", type=float, default=0.0, help="per-decision probability of a random state reset during training")
    t.add_argument("--freeze-core", action="store_true", help="train the head only (frozen wiring)")
    t.add_argument("--balance", action="store_true", help="class-balance the loss by the teacher's argmax-action frequency")
    d = sp.add_parser("dagger")
    d.add_argument("--student", type=Path, required=True); d.add_argument("--subgraph", type=Path, required=True)
    d.add_argument("--teacher", type=Path, required=True); d.add_argument("--scenario", default="take_cover")
    d.add_argument("--decisions", type=int, default=30_000); d.add_argument("--envs", type=int, default=32)
    d.add_argument("--seed-base", type=int, default=40_000); d.add_argument("--beta", type=float, default=0.0)
    d.add_argument("--greedy", action="store_true"); d.add_argument("--device", default="cuda:0")
    d.add_argument("--out", type=Path, required=True)
    p = sp.add_parser("play")
    p.add_argument("--ckpt", type=Path, required=True); p.add_argument("--subgraph", type=Path, required=True)
    p.add_argument("--scenario", default="take_cover"); p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--sample", action="store_true", help="sample actions instead of greedy")
    p.add_argument("--blind", default=None, choices=[None, "black", "mean", "scramble"]); p.add_argument("--video", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "record":
        print(json.dumps(record(a.teacher, a.scenario, a.decisions, a.envs, a.seed_base, a.eps, a.out)))
    elif a.cmd == "dagger":
        print(json.dumps(dagger_record(a.student, a.subgraph, a.teacher, a.scenario, a.decisions, a.envs, a.seed_base, a.beta, a.out, a.device, a.greedy)))
    elif a.cmd == "train":
        print(json.dumps(train(a.data, a.subgraph, a.run, a.param, a.readout, a.device, a.epochs, a.batch, a.T, a.lr, a.drift, a.seed,
                               resume=a.resume, time_limit=a.time_limit, head=a.head, lr_head=a.lr_head, reset_prob=a.reset_prob,
                               freeze_core=a.freeze_core, balance=a.balance)))
    else:
        play(a.ckpt, a.subgraph, a.scenario, a.episodes, a.device, not a.sample, a.blind, a.video)


if __name__ == "__main__":
    main()
