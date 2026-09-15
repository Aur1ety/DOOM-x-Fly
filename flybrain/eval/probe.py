"""Gate 0: can the frozen, gain-matched connectome's descending population decode the enemy?

Two stages, no training of any agent:

  record  play scripted policies in ViZDoom and store one frame per decision plus the labels-buffer
          targets (nearest enemy / projectile azimuth, distance, time-to-collision, visibility).
  run     frames -> frozen front end -> frozen ConnectomeCore at matched gain -> population rates
          at the last substep of every decision; ridge probes (held-out EPISODES) from
          descending / visual-projection / readout populations to the targets, for the real
          subgraph, every null graph given, and the front end alone (random projection, no core).
          Gate: held-out enemy-azimuth R^2 >= 0.50 from the DN population AND >= +0.10 over the
          mean of the N1 rewires.

    python -m flybrain.eval.probe record --scenario take_cover --decisions 40000 --out $FLYBRAIN_OUT/probe/take_cover.npz
    python -m flybrain.eval.probe run --frames $FLYBRAIN_OUT/probe/take_cover.npz \
        --subgraph $FLYBRAIN_OUT/graph/subgraph_v5.npz --nulls $FLYBRAIN_OUT/nulls_v5/null_N1_*.npz \
        --out $FLYBRAIN_OUT/probe/gate0_take_cover_v5.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from flybrain import OUT_DIR

LABEL_NAMES = ("enemy_visible", "enemy_azimuth", "enemy_distance", "enemy_ttc",
               "projectile_visible", "projectile_azimuth", "projectile_distance", "projectile_ttc")
FEATURE_SETS = ("readout", "dn", "vpn")
TARGETS = ("enemy_azimuth", "projectile_azimuth", "enemy_visible", "enemy_log_ttc")
AZ_MAX_DEG = 60.0            # azimuth targets restricted to (roughly) the 108 deg frustum
LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
FE_PROJ_DIM = 2048           # random projection of the 30.9k clamped rates (front-end-only control)


# ----------------------------------------------------------------------------- record

def record(scenario: str, policies: list[str], n_decisions: int, seed_base: int, out: Path,
           sticky: float = 0.0) -> dict:
    from flybrain.env.doom import DoomEnv
    from flybrain.env.floors import POLICIES

    env = DoomEnv(scenario, frameskip=4, all_frames=False, rgb=True, labels=True, sticky_prob=sticky)
    gray, rgb, lab, epi, step, act, rew, pol = [], [], [], [], [], [], [], []
    ep = 0
    t0 = time.time()
    while len(gray) < n_decisions:
        name = policies[ep % len(policies)]
        policy = POLICIES[name]()
        seed = seed_base + ep
        rng = np.random.default_rng(seed)
        obs, info = env.reset(seed=seed)
        policy.reset(env, rng)
        for k in range(10_000):
            a = policy.act(obs, info)
            obs, r, term, trunc, info = env.step(a)
            l = info["labels"]
            gray.append(obs["gray"])
            rgb.append(obs["rgb"].reshape(60, 2, 80, 2, 3).mean((1, 3)).astype(np.uint8))   # half res
            lab.append([float(l[n]) for n in LABEL_NAMES])
            epi.append(ep); step.append(k); act.append(a); rew.append(r); pol.append(policies.index(name))
            if term or trunc or len(gray) >= n_decisions:
                break
        ep += 1
    env.close()
    meta = {"scenario": scenario, "policies": policies, "seed_base": seed_base, "n_episodes": ep,
            "n_decisions": len(gray), "label_names": list(LABEL_NAMES), "frameskip": 4,
            "record_s": time.time() - t0}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, gray=np.stack(gray), rgb=np.stack(rgb), labels=np.asarray(lab, np.float32),
             episode=np.asarray(epi, np.int32), step=np.asarray(step, np.int32),
             action=np.asarray(act, np.int16), reward=np.asarray(rew, np.float32),
             policy=np.asarray(pol, np.int8), meta=json.dumps(meta))
    return meta


# ----------------------------------------------------------------------------- rollout

def _streams(episode: np.ndarray, batch: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Assign episodes round-robin to `batch` streams; each stream = concatenated decision indices
    plus a flag marking the first decision of every episode."""
    ids = np.unique(episode)
    seqs, starts = [[] for _ in range(batch)], [[] for _ in range(batch)]
    for i, e in enumerate(ids):
        idx = np.flatnonzero(episode == e)
        s = i % batch
        seqs[s].append(idx); starts[s].append(np.r_[True, np.zeros(len(idx) - 1, bool)])
    return ([np.concatenate(x) if x else np.zeros(0, np.int64) for x in seqs],
            [np.concatenate(x) if x else np.zeros(0, bool) for x in starts])


def _feature_sets(sub: dict, neurons_parquet: Path) -> dict[str, np.ndarray]:
    """Node indices (into the subgraph) of the populations we probe."""
    import pandas as pd
    n = pd.read_parquet(neurons_parquet, columns=["bodyId", "superclass"])
    sc = dict(zip(n["bodyId"].to_numpy().tolist(), n["superclass"].astype(str).to_numpy().tolist()))
    body = np.asarray(sub["bodyId"]); dyn = ~np.asarray(sub["is_clamped"], bool)
    scl = np.array([sc.get(int(b), "") for b in body])
    return {"readout": np.flatnonzero(np.asarray(sub["is_output"], bool)),
            "dn": np.flatnonzero(dyn & (scl == "descending_neuron")),
            "vpn": np.flatnonzero(dyn & (scl == "visual_projection"))}


@torch.no_grad()
def rollout(frames: dict, fe, core, perm: np.ndarray, feats: dict[str, np.ndarray], batch: int,
            device: str, fe_proj: torch.Tensor | None = None) -> dict[str, np.ndarray]:
    """Drive the front end (+ core) over all recorded decisions; return per-decision features [N, d]."""
    from flybrain.model.core import CoreState

    N = len(frames["episode"])
    seqs, starts = _streams(frames["episode"], batch)
    T = max(len(s) for s in seqs)
    out = {k: np.zeros((N, len(v)), np.float16) for k, v in feats.items()}
    if fe_proj is not None:
        out["frontend"] = np.zeros((N, fe_proj.shape[1]), np.float16)
    fe_state = fe.init_state(batch)
    if core is not None:
        st = core.init_state(batch); init = core.init_state(batch)
    perm_t = torch.as_tensor(perm, device=device)
    feat_idx = {k: torch.as_tensor(v, device=device) for k, v in feats.items()}
    last_idx = np.array([s[0] if len(s) else 0 for s in seqs])
    for t in range(T):
        idx = np.array([s[t] if t < len(s) else last_idx[i] for i, s in enumerate(seqs)])
        live = np.array([t < len(s) for s in seqs])
        last_idx = idx
        reset = np.array([bool(st_[t]) if t < len(st_) else False for st_ in starts])
        if reset.any():
            fe.reset(fe_state, reset)
            if core is not None:
                m = torch.as_tensor(reset, device=device)
                st = CoreState(h=torch.where(m[None, :], init.h, st.h),
                               x_clamped=torch.where(m[None, :], init.x_clamped, st.x_clamped),
                               r_out_prev=torch.where(m[None, :], init.r_out_prev, st.r_out_prev))
        rates, fe_state = fe.process(frames["gray"][idx], frames["rgb"][idx], fe_state)   # [K, B, n_units]
        xc = rates.index_select(2, perm_t)                                                # subgraph clamped order
        if core is not None:
            st = core.step(st, xc)
            r = core.rate(st.h)                                                           # [M, B]
            for k, fi in feat_idx.items():
                out[k][idx[live]] = r.index_select(0, fi).T[live].to(torch.float16).cpu().numpy()
        if fe_proj is not None:
            out["frontend"][idx[live]] = (xc[-1] @ fe_proj)[live].to(torch.float16).cpu().numpy()
    return out


# ----------------------------------------------------------------------------- probes

def _split_episodes(episode: np.ndarray, frac_test: float = 0.2, seed: int = 0):
    ids = np.unique(episode); rng = np.random.default_rng(seed); rng.shuffle(ids)
    n_test = max(1, int(round(frac_test * len(ids))))
    test_ids, train_ids = ids[:n_test], ids[n_test:]
    n_val = max(1, int(round(0.25 * len(train_ids))))
    return train_ids[n_val:], train_ids[:n_val], test_ids


def _targets(labels: np.ndarray) -> dict[str, np.ndarray]:
    L = {n: labels[:, i] for i, n in enumerate(LABEL_NAMES)}
    with np.errstate(invalid="ignore", divide="ignore"):
        log_ttc = np.log(np.clip(L["enemy_ttc"], 1.0, 1e4))
    return {"enemy_azimuth": np.where((L["enemy_visible"] > 0) & (np.abs(L["enemy_azimuth"]) <= AZ_MAX_DEG), L["enemy_azimuth"], np.nan),
            "projectile_azimuth": np.where((L["projectile_visible"] > 0) & (np.abs(L["projectile_azimuth"]) <= AZ_MAX_DEG), L["projectile_azimuth"], np.nan),
            "enemy_visible": L["enemy_visible"],
            "enemy_log_ttc": np.where((L["enemy_visible"] > 0) & np.isfinite(L["enemy_ttc"]), log_ttc, np.nan)}


def ridge_probe(X: np.ndarray, y: np.ndarray, episode: np.ndarray, seed: int = 0, device: str = "cpu") -> dict:
    """Ridge from X to y; lambda picked on validation episodes, R^2 reported on held-out episodes.

    The Gram matrices are formed once per split (float32, on `device`); the per-lambda solves are
    d x d in float64 on the CPU, so the cost is one pass over the data, not one per lambda.
    """
    ok = np.isfinite(y)
    fit_ids, val_ids, test_ids = _split_episodes(episode, seed=seed)
    fit = ok & np.isin(episode, fit_ids); val = ok & np.isin(episode, val_ids); test = ok & np.isin(episode, test_ids)
    if fit.sum() < 50 or val.sum() < 20 or test.sum() < 20:
        return {"r2": float("nan"), "n_fit": int(fit.sum()), "n_val": int(val.sum()), "n_test": int(test.sum())}
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device); yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    mu, sd = Xt[fit].mean(0), Xt[fit].std(0) + 1e-6
    Z = (Xt - mu) / sd
    ym = yt[fit].mean()

    def gram(rows):
        Zr = Z[rows]; yr = yt[rows] - ym
        return (Zr.T @ Zr).double().cpu(), (Zr.T @ yr).double().cpu()

    def r2(rows, w):
        wd = w.to(torch.float32).to(device)
        pred = Z[rows] @ wd + ym; yr = yt[rows]
        res = ((yr - pred) ** 2).sum(); tot = ((yr - yr.mean()) ** 2).sum()
        return float(1.0 - res / tot) if tot > 0 else float("nan")

    A_fit, b_fit = gram(fit)
    eye = torch.eye(A_fit.shape[0], dtype=A_fit.dtype)
    scores = {lam: r2(val, torch.linalg.solve(A_fit + lam * eye, b_fit)) for lam in LAMBDAS}
    best = max(scores, key=lambda l: (scores[l] if np.isfinite(scores[l]) else -1e9))
    A_all, b_all = gram(fit | val)
    w = torch.linalg.solve(A_all + best * eye, b_all)
    return {"r2": r2(test, w), "r2_val": scores[best], "lambda": best, "n_fit": int(fit.sum()),
            "n_val": int(val.sum()), "n_test": int(test.sum()), "d": int(X.shape[1])}


# ----------------------------------------------------------------------------- run

def run(frames_path: Path, subgraph: Path, nulls: list[Path], geometry: Path, neurons: Path, op_point: str,
        batch: int, device: str, out: Path, seed: int = 0) -> dict:
    from flybrain.model.core import ConnectomeCore, CoreConfig, load_subgraph
    from flybrain.model.gain import match_gain
    from flybrain.vision.frontend import FrontEnd
    from flybrain.vision.geometry import Geometry

    torch.set_num_threads(min(16, os.cpu_count() or 1))
    z = np.load(frames_path, allow_pickle=False)
    frames = {k: z[k] for k in ("gray", "rgb", "labels", "episode", "step", "action", "policy")}
    meta = json.loads(str(z["meta"]))
    targets = _targets(frames["labels"])
    print(f"frames: {meta['n_decisions']} decisions, {meta['n_episodes']} episodes, scenario {meta['scenario']}", flush=True)
    for k, v in targets.items():
        print(f"   target {k}: {int(np.isfinite(v).sum())} valid rows", flush=True)

    geom = Geometry.load(geometry)
    fe = FrontEnd(geom, K=6, device=device)
    sub = load_subgraph(subgraph)
    body = np.asarray(sub["bodyId"]); is_cl = np.asarray(sub["is_clamped"], bool)
    perm = fe.index.permutation_for(body[is_cl])
    feats = _feature_sets(sub, neurons)
    print("feature sets:", {k: len(v) for k, v in feats.items()}, flush=True)
    g = torch.Generator(device="cpu").manual_seed(seed)
    fe_proj = (torch.randn(int(is_cl.sum()), FE_PROJ_DIM, generator=g) / np.sqrt(is_cl.sum())).to(device)

    results = {"meta": meta, "op_point": op_point, "batch": batch, "feature_dims": {k: int(len(v)) for k, v in feats.items()},
               "graphs": {}}
    graphs = [("A", subgraph)] + [(f"N1_{Path(p).stem.split('_')[-1]}", Path(p)) for p in nulls]
    for name, path in graphs:
        t0 = time.time()
        sub_g = sub if name == "A" else load_subgraph(path)
        core = ConnectomeCore(sub_g, CoreConfig(param="type_tied"), backend="spmm", device=device)
        gm = match_gain(core, g_star=0.95, op_point=op_point, seed=seed)
        F = rollout(frames, fe, core, perm, feats, batch, device, fe_proj if name == "A" else None)
        res = {"gain": {k: gm[k] for k in ("w0", "gain", "n_evals", "mean_rate_dynamic", "frac_saturated", "frac_silent")},
               "probes": {}}
        for fs in list(feats) + (["frontend"] if name == "A" else []):
            res["probes"][fs] = {tg: ridge_probe(F[fs].astype(np.float32), targets[tg], frames["episode"], seed, device) for tg in TARGETS}
        res["seconds"] = time.time() - t0
        results["graphs"][name] = res
        print(f"{name}: w0={gm['w0']:.5f} gain={gm['gain']:.3f} sat={gm['frac_saturated']:.4f} | " +
              " ".join(f"{fs}/{tg}={res['probes'][fs][tg]['r2']:.3f}" for fs in res["probes"] for tg in ("enemy_azimuth", "projectile_azimuth"))
              + f" | {res['seconds']:.0f}s", flush=True)
        del core; torch.cuda.empty_cache() if device.startswith("cuda") else None

    # gate: DN population, enemy azimuth; projectile azimuth reported alongside for take_cover
    a = results["graphs"]["A"]["probes"]
    nulls_r2 = [results["graphs"][n]["probes"]["dn"]["enemy_azimuth"]["r2"] for n in results["graphs"] if n != "A"]
    gate = {"r2_A_dn": a["dn"]["enemy_azimuth"]["r2"], "r2_null_dn_mean": float(np.nanmean(nulls_r2)) if nulls_r2 else float("nan"),
            "r2_frontend_only": a["frontend"]["enemy_azimuth"]["r2"], "n_nulls": len(nulls_r2)}
    gate["pass_abs"] = bool(gate["r2_A_dn"] >= 0.50)
    gate["pass_vs_nulls"] = bool(np.isfinite(gate["r2_null_dn_mean"]) and gate["r2_A_dn"] - gate["r2_null_dn_mean"] >= 0.10)
    gate["GO"] = gate["pass_abs"] and gate["pass_vs_nulls"]
    results["gate"] = gate
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=float))
    print("GATE 0:", json.dumps(gate), flush=True)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("record")
    r.add_argument("--scenario", default="take_cover")
    r.add_argument("--policies", nargs="+", default=["random", "centroid"])
    r.add_argument("--decisions", type=int, default=40_000)
    r.add_argument("--seed-base", type=int, default=20_000, help="probe seeds: disjoint from train (0-999) and eval (>=10000) ranges used by floors")
    r.add_argument("--sticky", type=float, default=0.0)
    r.add_argument("--out", type=Path, required=True)
    u = sp.add_parser("run")
    u.add_argument("--frames", type=Path, required=True)
    u.add_argument("--subgraph", type=Path, required=True)
    u.add_argument("--nulls", nargs="*", default=[])
    u.add_argument("--geometry", type=Path, default=OUT_DIR / "vision" / "geometry_v1.npz")
    u.add_argument("--neurons", type=Path, default=OUT_DIR / "graph" / "neurons.parquet")
    u.add_argument("--op-point", default="rest", choices=["drive", "rest", "fixed_point"])
    u.add_argument("--batch", type=int, default=64)
    u.add_argument("--device", default="cuda:0")
    u.add_argument("--seed", type=int, default=0)
    u.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "record":
        print(json.dumps(record(a.scenario, a.policies, a.decisions, a.seed_base, a.out, a.sticky)))
    else:
        nulls = [p for pat in a.nulls for p in sorted(glob.glob(pat))]
        run(a.frames, a.subgraph, [Path(p) for p in nulls], a.geometry, a.neurons, a.op_point, a.batch, a.device, a.out, a.seed)


if __name__ == "__main__":
    main()
