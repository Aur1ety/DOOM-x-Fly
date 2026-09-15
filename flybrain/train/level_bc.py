"""Real Doom level (shareware E1M1 "Hangar", skill 1): imitation of the privileged navigator into the
frozen-wiring connectome agent, plus DAgger (the navigator labels the states the agent itself visits).

    record  navigator demonstrations; with prob eps a short burst of random movement is executed instead
            (the stored label is always the navigator's action): 120x160 grey, 60x80 RGB, label, episode id
    feats   teacher-forced rollout of the frozen eye model + connectome core over a recording; caches the
            readout population's rates + first differences per decision (exact: nothing upstream trains)
    fit     trains the readout head on cached features -> student.pt (readable by flybrain.train.bc.load_student)
    dagger  the agent drives (navigator with prob beta); stores frames + the navigator's action at visited states
    eval    closed-loop exit rate and route progress on held-out seeds (>= 10,000); --blind black|mean for the
            reactivity check; --policy random|forward|navigator for floors and ceiling
    video   640x480 game view + the map with the walked path + descending-neuron activity

    python -m flybrain.train.level_bc record --episodes 120 --workers 20 --out $FLYBRAIN_OUT/e1m1/demo_nav.npz
    python -m flybrain.train.level_bc feats --data $FLYBRAIN_OUT/e1m1/demo_nav.npz
    python -m flybrain.train.level_bc fit --data $FLYBRAIN_OUT/e1m1/demo_nav.npz --run e1m1_r0
    python -m flybrain.train.level_bc eval --ckpt $FLYBRAIN_OUT/e1m1/e1m1_r0/student.pt --episodes 30
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from flybrain import OUT_DIR
from flybrain.env.level import ACTIONS, N_ACTIONS, _down

SUBGRAPH = OUT_DIR / "graph" / "subgraph_v5.npz"
LEVEL_OUT = OUT_DIR / "e1m1"
LABEL_LOGIT = 5.0                      # stored "teacher logits": the navigator's action gets p ~ 0.95
EVAL_SEED_LO, DEMO_SEED_LO, DAGGER_SEED_LO = 10_000, 30_000, 50_000
INFO_KEYS = ("x", "y", "angle", "health", "kills", "distance", "progress")
MOVES = 6                              # actions 0..5 are movement / turning
NAV_MODES = ("navigate", "fight", "exit", "use_burst", "unstick", "stall_use", "lost")   # = level.Navigator.MODES
MASKED_MODES = ("unstick", "lost")     # labels not inferable from the frames: random wiggle picks / off the distance field


# ------------------------------------------------------------------------------------------ worker pool

def _worker(conn, env_kwargs: dict, seeds: list[int]) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
    from flybrain.env.level import LevelEnv, Navigator

    env = LevelEnv(**env_kwargs); nav = Navigator(); todo = list(seeds)

    def pack(obs, info):
        label = nav.act(obs, info)
        return obs["gray"], _down(obs["rgb"], 2), label, {k: float(info[k]) for k in INFO_KEYS if k in info}, NAV_MODES.index(nav.mode)

    def start():
        if not todo:
            return None
        s = todo.pop(0)
        obs, info = env.reset(s); nav.reset(env, np.random.default_rng(s))
        return pack(obs, info)

    try:
        while True:
            cmd, arg = conn.recv()
            if cmd == "reset":
                conn.send((start(), None))
            elif cmd == "step":
                nav.executed(int(arg))
                obs, r, term, trunc, info = env.step(int(arg))
                conn.send((start(), info["episode"]) if (term or trunc) else (pack(obs, info), None))
            else:
                break
    finally:
        env.close(); conn.close()


class LevelPool:
    """One LevelEnv + navigator per process; each worker plays its own seed list, then goes idle.

    After reset/step, `gray`, `rgb`, `label` (the navigator's action for the current state) and `info`
    hold the current decision's data for every env; `alive` marks envs that still have episodes to play.
    Create the pool before any CUDA initialisation (workers are forked).
    """

    def __init__(self, seeds: list[int], workers: int, **env_kwargs):
        ctx = mp.get_context("fork")
        self.n = min(workers, len(seeds))
        self.conns, self.procs = [], []
        for w in range(self.n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, env_kwargs, seeds[w::self.n]), daemon=True)
            p.start(); b.close()
            self.conns.append(a); self.procs.append(p)
        self.alive = np.zeros(self.n, bool)
        self.gray = np.zeros((self.n, 120, 160), np.uint8); self.rgb = np.zeros((self.n, 60, 80, 3), np.uint8)
        self.label = np.zeros(self.n, np.int64); self.info: list[dict] = [{} for _ in range(self.n)]
        self.mode = np.zeros(self.n, np.int8)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _take(self, i: int, pk) -> None:
        self.alive[i] = pk is not None
        if pk is not None:
            self.gray[i], self.rgb[i], self.label[i], self.info[i], self.mode[i] = pk

    def reset(self) -> None:
        for c in self.conns:
            c.send(("reset", None))
        for i, c in enumerate(self.conns):
            self._take(i, c.recv()[0])

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """Step the alive envs; returns (episode-ended mask, finished-episode stats). Ended envs auto-reset."""
        live = np.flatnonzero(self.alive)
        for i in live:
            self.conns[i].send(("step", int(actions[i])))
        done = np.zeros(self.n, bool); finals = []
        for i in live:
            pk, ep = self.conns[i].recv()
            if ep is not None:
                done[i] = True; finals.append(ep)
            self._take(i, pk)
        return done, finals

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()


def drive(pool: LevelPool, policy, store: bool, log_every: float = 60.0) -> tuple[dict | None, list[dict]]:
    """Run every env's seed list to the end. Stores frames at decision time + navigator labels if `store`."""
    pool.reset()
    ep = np.arange(pool.n); next_ep = pool.n; t_in = np.zeros(pool.n, np.int64)
    G, R, L, X, E, S, MO, DI, SA = [], [], [], [], [], [], [], [], []
    finals: list[dict] = []; t0 = t_log = time.time(); n_dec = 0
    while pool.alive.any():
        live = pool.alive.copy()
        acts = policy.act(pool)
        if store:
            G.append(pool.gray[live].copy()); R.append(pool.rgb[live].copy()); L.append(pool.label[live].astype(np.int8))
            X.append(acts[live].astype(np.int8)); E.append(ep[live].copy()); S.append(t_in[live].copy())
            MO.append(pool.mode[live].copy()); DI.append(np.array([pool.info[i].get("distance", np.inf) for i in np.flatnonzero(live)], np.float32))
            SA.append(np.asarray(getattr(policy, "last_student", acts))[live].astype(np.int8))
        done, fin = pool.step(acts)
        finals += fin; n_dec += int(live.sum()); t_in += 1
        for i in np.flatnonzero(done):
            ep[i] = next_ep; next_ep += 1; t_in[i] = 0
        policy.done(done)
        if time.time() - t_log > log_every:
            t_log = time.time()
            ex = sum(f["exited"] for f in finals)
            print(f"  {n_dec} decisions ({n_dec / (t_log - t0):.0f}/s), {len(finals)} episodes done, exited {ex}, {pool.alive.sum()} envs busy", flush=True)
    if not store:
        return None, finals
    data = {k: np.concatenate(v) for k, v in zip(("gray", "rgb", "label", "action", "episode", "step", "mode", "distance", "student_action"),
                                                  (G, R, L, X, E, S, MO, DI, SA))}
    order = np.lexsort((data["step"], data["episode"]))
    data = {k: v[order] for k, v in data.items()}
    return data, finals


def save_dataset(path: Path, data: dict, meta: dict) -> None:
    lg = np.zeros((len(data["label"]), N_ACTIONS), np.float16)
    lg[np.arange(len(lg)), data["label"].astype(np.int64)] = LABEL_LOGIT
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, gray=data["gray"], rgb=data["rgb"], logits=lg, label=data["label"], action=data["action"],
             episode=data["episode"].astype(np.int32), step=data["step"].astype(np.int32), mode=data["mode"],
             distance=data["distance"], student_action=data["student_action"], meta=json.dumps(meta))


def _train_mask(z) -> np.ndarray:
    """Rows whose navigator label is learnable from the frames (older recordings have no mode/distance: keep all)."""
    keep = np.ones(len(z["label"]), bool)
    if "mode" in z.files:
        keep &= ~np.isin(z["mode"], [NAV_MODES.index(m) for m in MASKED_MODES])
    if "distance" in z.files:
        keep &= np.isfinite(z["distance"])
    return keep


def _fingerprint(z) -> str:
    import hashlib
    return hashlib.sha1(z["episode"].tobytes() + z["step"].tobytes() + z["label"].tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------ policies

class NavPolicy:
    """The navigator's action, replaced with prob eps by a burst (1..burst decisions) of one random movement."""

    def __init__(self, n: int, eps: float, burst: int, seed: int):
        self.eps, self.burst, self.rng = eps, burst, np.random.default_rng(seed)
        self.left = np.zeros(n, np.int64); self.rand = np.zeros(n, np.int64)

    def act(self, pool):
        a = pool.label.copy()
        start = (self.left == 0) & (self.rng.random(pool.n) < self.eps)
        self.rand[start] = self.rng.integers(0, MOVES, int(start.sum()))
        self.left[start] = self.rng.integers(1, self.burst + 1, int(start.sum()))
        use = self.left > 0
        a[use] = self.rand[use]; self.left[use] -= 1
        return a

    def done(self, mask):
        self.left[mask] = 0


class FloorPolicy:
    def __init__(self, n: int, kind: str, seed: int):
        self.n, self.kind, self.rng = n, kind, np.random.default_rng(seed)

    def act(self, pool):
        if self.kind == "random":
            return self.rng.integers(0, N_ACTIONS, self.n)
        return np.zeros(self.n, np.int64)                      # forward

    def done(self, mask):
        pass


class AgentPolicy:
    """The connectome agent for a batch of envs; with prob beta the navigator's action is executed instead."""

    def __init__(self, st, n: int, beta: float = 0.0, greedy: bool = True, blind: str | None = None, seed: int = 0,
                 temperature: float = 1.0):
        import torch
        self.torch, self.st, self.n, self.beta, self.greedy, self.blind = torch, st, n, beta, greedy, blind
        self.temperature = temperature                 # sampling temperature on the head's logits (1 = as trained)
        self.rng = np.random.default_rng(seed); self.state = st.init_state(n)
        self.hist = np.zeros(N_ACTIONS, np.int64); self.buf: list[list] = [[] for _ in range(n)]

    def act(self, pool):
        gray, rgb = pool.gray, pool.rgb
        if self.blind == "black":
            gray, rgb = np.zeros_like(gray), np.zeros_like(rgb)
        elif self.blind == "mean":
            gray = np.broadcast_to(gray.mean((1, 2), keepdims=True), gray.shape).astype(np.uint8)
            rgb = np.broadcast_to(rgb.mean((1, 2, 3), keepdims=True), rgb.shape).astype(np.uint8)
        elif self.blind == "scramble":                 # this episode's own frames so far, in random order
            gray, rgb = gray.copy(), rgb.copy()
            for i in np.flatnonzero(pool.alive):
                self.buf[i].append((pool.gray[i].copy(), pool.rgb[i].copy()))
                g, r = self.buf[i][int(self.rng.integers(len(self.buf[i])))]
                gray[i], rgb[i] = g, r
        with self.torch.no_grad():
            lg = self.st.decide(gray, rgb, self.state)
            if self.greedy:
                a = lg.argmax(1).cpu().numpy()
            else:
                a = self.torch.distributions.Categorical(logits=lg / self.temperature).sample().cpu().numpy()
        np.add.at(self.hist, a[pool.alive], 1)
        self.last_student = a.copy()
        if self.beta > 0:
            a = np.where(self.rng.random(self.n) < self.beta, pool.label, a)
        return a

    def done(self, mask):
        self.state = self.st.reset_rows(self.state, mask)
        for i in np.flatnonzero(mask):
            self.buf[i] = []


class ReplayPolicy:
    """Open-loop floor: each episode replays the executed action sequence of a random recorded episode, blind."""

    def __init__(self, n: int, data: Path, seed: int):
        z = np.load(data)
        act, ep = z["action"].astype(np.int64), z["episode"]
        self.seqs = [act[ep == e] for e in np.unique(ep)]
        self.rng = np.random.default_rng(seed); self.t = np.zeros(n, np.int64)
        self.pick = self.rng.integers(len(self.seqs), size=n)

    def act(self, pool):
        a = np.array([self.seqs[k][t] if t < len(self.seqs[k]) else 0 for k, t in zip(self.pick, self.t)])
        self.t += 1
        return a

    def done(self, mask):
        self.t[mask] = 0; self.pick[mask] = self.rng.integers(len(self.seqs), size=int(mask.sum()))


def _guarded_device(device: str) -> str:
    """Share the card with the vLLM tenant: pin the index, cap our memory (flybrain.ops.gpu_guard)."""
    if not device.startswith("cuda"):
        return device
    if os.environ.get("FLYBRAIN_GPU_SHARED", "1") == "0":      # card not shared (vLLM stopped): no caps, use as given
        return device
    from flybrain.ops.gpu_guard import guard_gpu
    idx = device.split(":")[1] if ":" in device else "0"
    guard_gpu(max_own_gib=2.0, fraction_cap_gib=1.8, require_free_gib=2.2, visible_devices=idx)
    return "cuda:0"


def summarize_eps(finals: list[dict]) -> dict:
    from flybrain.env.floors import summarize
    ex = np.array([f["exited"] for f in finals], float)
    prog = np.array([f["best_progress"] for f in finals]); tics = np.array([f["tics"] for f in finals], float)
    hist = np.sum([f["action_hist"] for f in finals], 0); hist = hist / max(hist.sum(), 1)
    rng = np.random.default_rng(0); boot = ex[rng.integers(0, len(ex), (2000, len(ex)))].mean(1)
    return {"episodes": len(finals), "exit_rate": float(ex.mean()), "exit_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "progress": summarize(prog), "tics_exited_mean": float(tics[ex > 0].mean()) if ex.any() else None,
            "dead": int(sum(f["dead"] for f in finals)), "timeout": int(sum(f["timeout"] for f in finals)),
            "kills_mean": float(np.mean([f["kills"] for f in finals])), "action_hist": dict(zip(ACTIONS, np.round(hist, 3).tolist())),
            "exit_rate_wiggle0": _rate([f for f in finals if f.get("wiggle", -1) == 0]),
            "exit_rate_wiggle_gt0": _rate([f for f in finals if f.get("wiggle", -1) > 0])}


def _rate(fs: list[dict]):
    return float(np.mean([f["exited"] for f in fs])) if fs else None


def _env_kwargs(a) -> dict:
    return {"start_wiggle": a.wiggle, "timeout_s": a.timeout, "sticky_prob": a.sticky, "wiggle_moves": a.wiggle_moves}


# ------------------------------------------------------------------------------------------ commands

def cmd_record(a) -> None:
    seeds = list(range(a.seed_base, a.seed_base + a.episodes))
    t0 = time.time()
    with LevelPool(seeds, a.workers, **_env_kwargs(a)) as pool:
        data, finals = drive(pool, NavPolicy(pool.n, a.eps, a.burst, a.seed_base), store=True)
    meta = {"kind": "navigator", "episodes": len(finals), "decisions": int(len(data["label"])), "eps": a.eps, "burst": a.burst,
            "wiggle": a.wiggle, "seed_base": a.seed_base, "record_s": time.time() - t0, "stats": summarize_eps(finals),
            "label_hist": dict(zip(ACTIONS, np.round(np.bincount(data["label"], minlength=N_ACTIONS) / len(data["label"]), 3).tolist())),
            "mode_hist": dict(zip(NAV_MODES, np.round(np.bincount(data["mode"], minlength=len(NAV_MODES)) / len(data["mode"]), 3).tolist()))}
    save_dataset(a.out, data, meta)
    print(json.dumps(meta))


def _cache_path(data: Path, readout: str) -> Path:
    return data.with_suffix(f".{readout}.npy")


def cmd_feats(a) -> None:
    import torch
    from flybrain.model.core import load_subgraph
    from flybrain.train.bc import DEFAULT_GEOMETRY, DEFAULT_NEURONS, Student, _streams

    device = _guarded_device(a.device)
    for data in a.data:
        out = _cache_path(data, a.readout)
        z = np.load(data)
        fp = _fingerprint(z)
        if out.exists() and not a.force:
            mj = out.with_suffix(".json")
            if mj.exists() and json.loads(mj.read_text()).get("fp") == fp:
                print(f"{out} exists (matches {data.name})"); continue
            print(f"{out} is stale -> recomputing")
        gray, rgb, episode = z["gray"], z["rgb"], z["episode"]
        st = Student(load_subgraph(SUBGRAPH), DEFAULT_GEOMETRY, DEFAULT_NEURONS, a.param, a.readout, device, w0=a.w0, n_actions=N_ACTIONS)
        st.feat_mu.zero_(); st.feat_sd.fill_(1.0)
        seqs, starts = _streams(episode, a.batch)
        F = np.zeros((len(episode), 2 * st.nodes.numel()), np.float32)      # float32: DN rates vary by ~0.004 sd, float16 steps are 5e-4..2e-3
        state = st.init_state(a.batch); t0 = time.time()
        with torch.no_grad():
            for t in range(max(len(s) for s in seqs)):
                live = np.array([t < len(s) for s in seqs])
                idx = np.array([s[t] if t < len(s) else 0 for s in seqs])
                state = st.reset_rows(state, np.array([bool(r[t]) if t < len(r) else False for r in starts]))
                f = st.core_step(state, st.frontend(gray[idx], rgb[idx], state), False)
                F[idx[live]] = f[torch.as_tensor(live, device=device)].float().cpu().numpy()
                if t % 500 == 0:
                    print(f"  {data.name}: t {t} ({time.time() - t0:.0f}s)", flush=True)
        np.save(out, F)
        meta = {"data": str(data), "readout": a.readout, "param": a.param, "w0": st.w0, "n": int(len(F)), "d": int(F.shape[1]), "fp": fp, "s": time.time() - t0}
        out.with_suffix(".json").write_text(json.dumps(meta))
        print(json.dumps(meta), flush=True)


def cmd_fit(a) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    from flybrain.model.core import load_subgraph
    from flybrain.train.bc import DEFAULT_GEOMETRY, DEFAULT_NEURONS, Student

    device = _guarded_device(a.device)
    torch.manual_seed(a.seed)
    import math
    Xs, Ys, Es, Ks, Ps, off, w0s = [], [], [], [], [], 0, []
    for data in a.data:
        c = _cache_path(data, a.readout); cm = json.loads(c.with_suffix(".json").read_text())
        Xc = np.load(c, mmap_mode="r"); z = np.load(data)
        assert len(Xc) == len(z["label"]) == cm["n"], f"stale feature cache {c}: {len(Xc)} rows vs {len(z['label'])} labels"
        assert cm.get("fp") in (None, _fingerprint(z)), f"feature cache {c} was built from a different recording; rerun feats"
        Xs.append(Xc); w0s.append(float(cm["w0"])); Ys.append(z["label"].astype(np.int64)); Ks.append(_train_mask(z))
        e = z["episode"].astype(np.int64); Es.append(e + off); off += int(e.max()) + 1
        act = z["action"].astype(np.int64); same = np.r_[False, e[1:] == e[:-1]]
        Ps.append(np.where(same, np.r_[-1, act[:-1]], -1))           # previous executed action (copy-previous baseline)
    assert all(math.isclose(w, w0s[0], rel_tol=1e-6) for w in w0s), f"feature caches disagree on w0: {w0s}"
    X = np.concatenate(Xs); y = np.concatenate(Ys); ep = np.concatenate(Es); keep = np.concatenate(Ks); prev = np.concatenate(Ps)
    ids = np.unique(ep); rng = np.random.default_rng(a.seed); rng.shuffle(ids)
    test = np.isin(ep, ids[: max(1, len(ids) // 10)]); tr = np.flatnonzero(~test & keep); te = np.flatnonzero(test & keep)
    d = X.shape[1]
    s1 = np.zeros(d); s2 = np.zeros(d)
    for i in range(0, len(tr), 50_000):
        xb = X[tr[i:i + 50_000]].astype(np.float64); s1 += xb.sum(0); s2 += (xb * xb).sum(0)
    mu = s1 / len(tr); sd = np.sqrt(np.maximum(s2 / len(tr) - mu * mu, 0) + 1e-6)
    mu_t = torch.as_tensor(mu, dtype=torch.float32, device=device); sd_t = torch.as_tensor(sd, dtype=torch.float32, device=device)
    head = (nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, N_ACTIONS)) if a.head == "mlp" else nn.Linear(d, N_ACTIONS)).to(device)
    freq = np.bincount(y[tr], minlength=N_ACTIONS) / len(tr)
    cw = torch.as_tensor((1.0 / np.maximum(freq, 1e-3)) ** a.balance, dtype=torch.float32, device=device); cw = cw / (cw * torch.as_tensor(freq, dtype=torch.float32, device=device)).sum()
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    def batches(rows, bs):
        for i in range(0, len(rows), bs):
            r = np.sort(rows[i:i + bs])
            yield torch.as_tensor(X[r], device=device).float().sub_(mu_t).div_(sd_t), torch.as_tensor(y[r], device=device)

    best = (-1.0, None); t0 = time.time()
    print(f"fit: {len(tr)} train / {len(te)} held-out decisions ({int((~keep).sum())} masked), {len(ids)} episodes, d {d}, label freq {np.round(freq, 3).tolist()}", flush=True)
    for e in range(a.epochs):
        head.train(); perm = rng.permutation(tr); tot = 0.0
        for xb, yb in batches(perm, a.batch):
            if a.noise > 0:                               # robustness to small closed-loop feature deviations (z units)
                xb = xb + a.noise * torch.randn_like(xb)
            loss = Fn.cross_entropy(head(xb), yb, weight=cw, label_smoothing=a.smooth)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += float(loss) * len(yb)
        sched.step(); head.eval()
        with torch.no_grad():
            pred = np.concatenate([head(xb).argmax(1).cpu().numpy() for xb, _ in batches(te, 16_384)])
        agree = float((pred == y[te]).mean())
        per = {ACTIONS[k]: round(float((pred[y[te] == k] == k).mean()), 3) for k in range(N_ACTIONS) if (y[te] == k).any()}
        print(f"epoch {e} loss {tot / len(tr):.4f} held-out agreement {agree:.3f} recall {per} ({time.time() - t0:.0f}s)", flush=True)
        if agree > best[0]:
            best = (agree, {k: v.detach().clone() for k, v in head.state_dict().items()})
    st = Student(load_subgraph(SUBGRAPH), DEFAULT_GEOMETRY, DEFAULT_NEURONS, a.param, a.readout, device, w0=w0s[0], head=a.head, n_actions=N_ACTIONS)
    st.head.load_state_dict(best[1]); st.feat_mu.copy_(mu_t); st.feat_sd.copy_(sd_t)
    out = LEVEL_OUT / a.run; out.mkdir(parents=True, exist_ok=True)
    maj = float((y[te] == np.argmax(freq)).mean()); copy_prev = float((prev[te] == y[te]).mean())
    base = {"heldout_agree": best[0], "last_epoch_agree": agree, "majority_agree": maj, "copy_prev_agree": copy_prev}
    st.save(out / "student.pt", {**base, "data": [str(p) for p in a.data], "level": "E1M1"})
    res = {"run": a.run, "readout": a.readout, **base, "train_decisions": int(len(tr)), "masked": int((~keep).sum()), "noise": a.noise, "fit_s": time.time() - t0}
    (out / "fit.json").write_text(json.dumps(res)); print(json.dumps(res))


def _load_agent(ckpt: Path, device: str):
    from flybrain.train.bc import load_student
    return load_student(ckpt, SUBGRAPH, device)


def cmd_dagger(a) -> None:
    seeds = list(range(a.seed_base, a.seed_base + a.episodes))
    if not a.ckpt.is_file():
        raise SystemExit(f"missing checkpoint {a.ckpt}")
    t0 = time.time()
    with LevelPool(seeds, a.workers, **_env_kwargs(a)) as pool:
        device = _guarded_device(a.device)
        pol = AgentPolicy(_load_agent(a.ckpt, device), pool.n, beta=a.beta, greedy=not a.sample, seed=a.seed_base, temperature=a.temperature)
        data, finals = drive(pool, pol, store=True)
    stats = summarize_eps(finals)
    meta = {"kind": "dagger", "student": str(a.ckpt), "beta": a.beta, "episodes": len(finals), "decisions": int(len(data["label"])),
            "wiggle": a.wiggle, "seed_base": a.seed_base, "record_s": time.time() - t0, "student_stats": stats,
            "student_agree_with_navigator": float((data["label"] == data["student_action"]).mean()), "note": "stats are of the student+navigator(beta) mixture",
            "mode_hist": dict(zip(NAV_MODES, np.round(np.bincount(data["mode"], minlength=len(NAV_MODES)) / len(data["mode"]), 3).tolist()))}
    save_dataset(a.out, data, meta)
    print(json.dumps(meta))


def cmd_eval(a) -> None:
    seeds = list(range(a.seed_base, a.seed_base + a.episodes))
    if a.policy == "agent" and (a.ckpt is None or not a.ckpt.is_file()):
        raise SystemExit(f"missing checkpoint {a.ckpt}")
    if a.policy == "agent":
        import torch
        torch.manual_seed(a.seed_base)                   # reproducible action sampling
    t0 = time.time()
    with LevelPool(seeds, a.workers, **_env_kwargs(a)) as pool:
        if a.policy == "agent":
            device = _guarded_device(a.device)
            pol = AgentPolicy(_load_agent(a.ckpt, device), pool.n, greedy=not a.sample, blind=a.blind, seed=a.seed_base, temperature=a.temperature)
        elif a.policy == "navigator":
            pol = NavPolicy(pool.n, 0.0, 1, 0)
        elif a.policy == "replay":
            pol = ReplayPolicy(pool.n, a.replay_data, a.seed_base)
        else:
            pol = FloorPolicy(pool.n, a.policy, a.seed_base)
        _, finals = drive(pool, pol, store=False)
    res = {"policy": a.policy, "ckpt": str(a.ckpt) if a.ckpt else None, "blind": a.blind, "sample": a.sample, "wiggle": a.wiggle,
           "wiggle_moves": a.wiggle_moves, "sticky": a.sticky, "temperature": a.temperature, "seed_base": a.seed_base, "eval_s": time.time() - t0, **summarize_eps(finals),
           "per_episode": [{k: f.get(k) for k in ("seed", "exited", "dead", "timeout", "tics", "best_progress", "kills", "wiggle")} for f in finals]}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_episode"}))


# ------------------------------------------------------------------------------------------ video

class MapPanel:
    """Top-down E1M1 line map (walls, steps, doors, exit switch) with the walked path and the player."""

    def __init__(self, m, nav, width: int, height: int):
        from PIL import Image, ImageDraw
        self.Image, self.ImageDraw, self.W, self.H = Image, ImageDraw, width, height
        v = m.vertices.astype(float); lo, hi = v.min(0) - 48, v.max(0) + 48
        self.s = min((width - 16) / (hi[0] - lo[0]), (height - 48) / (hi[1] - lo[1]))
        self.lo, self.pad = lo, (8 + ((width - 16) - self.s * (hi[0] - lo[0])) / 2, 8)
        base = Image.new("RGB", (width, height), (12, 12, 16)); dr = ImageDraw.Draw(base)
        doors = set(m.door_lines())
        for i in range(len(m.lines)):
            x0, y0, x1, y1 = m.line_xy(i)
            one_sided = int(m.lines[i, 6]) < 0
            col, w = ((200, 200, 200), 1) if one_sided else ((90, 90, 100), 1)
            if i in doors:
                col, w = (90, 150, 255), 2
            if i == nav.exit_line:
                col, w = (40, 230, 90), 3
            dr.line([self.px(x0, y0), self.px(x1, y1)], fill=col, width=w)
        dr.text((8, height - 34), "E1M1 Hangar (Doom, 1993)", fill=(220, 220, 220))
        dr.text((8, height - 20), "green = exit switch", fill=(40, 230, 90))
        self.base = base

    def px(self, x: float, y: float) -> tuple[float, float]:
        return (self.pad[0] + (x - self.lo[0]) * self.s, self.H - 40 - (y - self.lo[1]) * self.s)

    def render(self, path: list[tuple[float, float]], x: float, y: float, angle: float) -> np.ndarray:
        import math
        img = self.base.copy(); dr = self.ImageDraw.Draw(img)
        if len(path) > 1:
            dr.line([self.px(*p) for p in path], fill=(255, 210, 40), width=2)
        cx, cy = self.px(x, y)
        dr.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 60, 60))
        hx, hy = self.px(x + 120 * math.cos(math.radians(angle)), y + 120 * math.sin(math.radians(angle)))
        dr.line([(cx, cy), (hx, hy)], fill=(255, 60, 60), width=2)
        return np.asarray(img)


def cmd_video(a) -> None:
    import imageio.v2 as iio
    import torch
    from PIL import Image, ImageDraw
    from flybrain.env.level import LevelEnv
    from flybrain.eval.record_video import _raster

    env = LevelEnv(keep_full=True, start_wiggle=a.wiggle, timeout_s=a.timeout, wiggle_moves=a.wiggle_moves)
    torch.manual_seed(a.seed_base)
    device = _guarded_device(a.device)
    st = _load_agent(a.ckpt, device)
    mp_panel = MapPanel(env.map, env.nav, 320, 480)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    vw = iio.get_writer(str(a.out), fps=35 // 4, codec="libx264", quality=8, macro_block_size=None)
    strip = np.zeros((480, 160, 3), np.uint8)
    stats = []
    for e in range(a.episodes):
        seed = a.seed_base + e
        obs, info = env.reset(seed); state = st.init_state(1); path = [(info["x"], info["y"])]
        done = False
        while not done:
            gray, rgb = obs["gray"][None], _down(obs["rgb"], 2)[None]
            if a.blind == "black":
                gray, rgb = np.zeros_like(gray), np.zeros_like(rgb)
            with torch.no_grad():
                lg = st.decide(gray, rgb, state)[0]
                act = int(torch.distributions.Categorical(logits=lg).sample()) if a.sample else int(lg.argmax())
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            if "x" in info:
                path.append((info["x"], info["y"]))
            game = Image.fromarray(obs["full"]); dr = ImageDraw.Draw(game)
            outcome = ("  EXIT REACHED" if info.get("exited") else "  DIED" if info.get("dead") else "  TIME UP") if done else ""
            txt = [f"fly connectome agent (MaleCNS wiring, frozen)  |  E1M1 skill 1  |  run {e + 1}/{a.episodes}",
                   f"route {100 * info.get('progress', 0.0):.0f}%   kills {int(info.get('kills', 0))}   health {int(info.get('health', 0))}   action: {ACTIONS[act]}{outcome}"]
            for j, t in enumerate(txt):
                dr.text((9, 9 + 14 * j), t, fill=(0, 0, 0)); dr.text((8, 8 + 14 * j), t, fill=(255, 255, 255))
            strip = np.roll(strip, -1, axis=1); strip[:, -1] = _raster(st.panel(state), 480, 1)[:, 0]
            mp_img = mp_panel.render(path, info.get("x", path[-1][0]), info.get("y", path[-1][1]), info.get("angle", 0.0))
            vw.append_data(np.concatenate([np.asarray(game), mp_img, strip], axis=1))
        stats.append(info["episode"])
        print(json.dumps({k: info["episode"][k] for k in ("seed", "exited", "dead", "timeout", "tics", "best_progress", "kills")}), flush=True)
    vw.close(); env.close()
    print(json.dumps({"out": str(a.out), **summarize_eps(stats)}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common_env(p, seed_base, episodes, timeout=180.0):
        p.add_argument("--episodes", type=int, default=episodes); p.add_argument("--seed-base", type=int, default=seed_base)
        p.add_argument("--workers", type=int, default=20); p.add_argument("--wiggle", type=int, default=6)
        p.add_argument("--timeout", type=float, default=timeout)
        p.add_argument("--sticky", type=float, default=0.0, help="prob. of repeating the previous action (stress test)")
        p.add_argument("--wiggle-moves", action="store_true", help="start wiggle also walks forward/backward (wider starts)")

    r = sp.add_parser("record"); common_env(r, DEMO_SEED_LO, 120)
    r.add_argument("--eps", type=float, default=0.05); r.add_argument("--burst", type=int, default=6)
    r.add_argument("--out", type=Path, required=True)
    f = sp.add_parser("feats")
    f.add_argument("--data", type=Path, nargs="+", required=True); f.add_argument("--readout", default="dn", choices=["readout36", "dn", "vpn", "dn_vpn"])
    f.add_argument("--param", default="per_edge"); f.add_argument("--w0", type=float, default=None)
    f.add_argument("--batch", type=int, default=64); f.add_argument("--device", default="cuda:0"); f.add_argument("--force", action="store_true")
    t = sp.add_parser("fit")
    t.add_argument("--data", type=Path, nargs="+", required=True); t.add_argument("--run", required=True)
    t.add_argument("--readout", default="dn", choices=["readout36", "dn", "vpn", "dn_vpn"]); t.add_argument("--param", default="per_edge")
    t.add_argument("--head", default="mlp", choices=["linear", "mlp"]); t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--lr", type=float, default=3e-3); t.add_argument("--wd", type=float, default=1e-4); t.add_argument("--batch", type=int, default=1024)
    t.add_argument("--smooth", type=float, default=0.05); t.add_argument("--balance", type=float, default=0.0, help="class weight = freq^-balance (0 = none, 0.5 = sqrt)")
    t.add_argument("--seed", type=int, default=0); t.add_argument("--device", default="cuda:0")
    t.add_argument("--noise", type=float, default=0.0, help="Gaussian noise (std, z units) added to standardised features while fitting")
    d = sp.add_parser("dagger"); common_env(d, DAGGER_SEED_LO, 40)
    d.add_argument("--ckpt", type=Path, required=True); d.add_argument("--beta", type=float, default=0.3)
    d.add_argument("--sample", action="store_true"); d.add_argument("--device", default="cuda:0"); d.add_argument("--out", type=Path, required=True)
    d.add_argument("--temperature", type=float, default=1.0)
    e = sp.add_parser("eval"); common_env(e, EVAL_SEED_LO, 30)
    e.add_argument("--policy", default="agent", choices=["agent", "navigator", "random", "forward", "replay"]); e.add_argument("--ckpt", type=Path, default=None)
    e.add_argument("--blind", default=None, choices=[None, "black", "mean", "scramble"]); e.add_argument("--sample", action="store_true")
    e.add_argument("--temperature", type=float, default=1.0, help="sampling temperature (with --sample)")
    e.add_argument("--replay-data", type=Path, default=LEVEL_OUT / "demo_nav.npz", help="recording whose action sequences --policy replay plays")
    e.add_argument("--device", default="cuda:0"); e.add_argument("--out", type=Path, default=None)
    v = sp.add_parser("video")
    v.add_argument("--ckpt", type=Path, required=True); v.add_argument("--out", type=Path, required=True)
    v.add_argument("--episodes", type=int, default=2); v.add_argument("--seed-base", type=int, default=EVAL_SEED_LO + 500)
    v.add_argument("--wiggle", type=int, default=6); v.add_argument("--timeout", type=float, default=180.0); v.add_argument("--wiggle-moves", action="store_true")
    v.add_argument("--blind", default=None, choices=[None, "black"]); v.add_argument("--device", default="cuda:0")
    v.add_argument("--sample", action="store_true", help="sample actions from the head (as in eval --sample)")
    a = ap.parse_args(argv)
    {"record": cmd_record, "feats": cmd_feats, "fit": cmd_fit, "dagger": cmd_dagger, "eval": cmd_eval, "video": cmd_video}[a.cmd](a)


if __name__ == "__main__":
    main()
