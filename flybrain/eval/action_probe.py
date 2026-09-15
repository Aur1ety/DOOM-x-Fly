"""Diagnostic: from the FROZEN, gain-matched core, which population can linearly predict the teacher's action?

Uses the teacher recording (frames + logits) and probe.rollout to get per-decision rates of the
readout / DN / VPN populations and a random projection of the front end; then fits a softmax
head (Adam, GPU) per population with an episode split. Reports held-out agreement with the
teacher's argmax action vs the majority-class baseline, and the KL to the teacher.

    python -m flybrain.eval.action_probe --data $FLYBRAIN_OUT/bc/teacher_tc.npz --subgraph $FLYBRAIN_OUT/graph/subgraph_v5.npz
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from flybrain import OUT_DIR
from flybrain.eval.probe import FE_PROJ_DIM, _feature_sets, _split_episodes, rollout


def fit_softmax(X: np.ndarray, P: torch.Tensor, episode: np.ndarray, device: str, steps: int = 3000, lr: float = 1e-2, wd: float = 1e-4) -> dict:
    fit_ids, val_ids, test_ids = _split_episodes(episode)
    fit = np.isin(episode, fit_ids) | np.isin(episode, val_ids); test = np.isin(episode, test_ids)
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6
    Z = (X - mu) / sd
    W = torch.zeros(Z.shape[1], P.shape[1], device=device, requires_grad=True); b = torch.zeros(P.shape[1], device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    Zf, Pf = Z[fit], P[fit]
    for _ in range(steps):
        idx = torch.randint(0, Zf.shape[0], (4096,), device=device)
        logp = F.log_softmax(Zf[idx] @ W + b, 1)
        loss = -(Pf[idx] * logp).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        logits = Z[test] @ W + b
        agree = float((logits.argmax(1) == P[test].argmax(1)).float().mean())
        kl = float((P[test] * (torch.log(P[test] + 1e-8) - F.log_softmax(logits, 1))).sum(1).mean())
        maj = int(P[fit].argmax(1).mode().values)
        maj_agree = float((P[test].argmax(1) == maj).float().mean())
    return {"agree": agree, "majority_agree": maj_agree, "kl": kl, "n_test": int(test.sum()), "d": int(X.shape[1])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--subgraph", type=Path, required=True)
    ap.add_argument("--geometry", type=Path, default=OUT_DIR / "vision" / "geometry_v1.npz")
    ap.add_argument("--neurons", type=Path, default=OUT_DIR / "graph" / "neurons.parquet")
    ap.add_argument("--op-point", default="rest")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-decisions", type=int, default=40000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    from flybrain.model.core import ConnectomeCore, CoreConfig, load_subgraph
    from flybrain.model.gain import match_gain
    from flybrain.vision.frontend import FrontEnd
    from flybrain.vision.geometry import Geometry

    z = np.load(a.data, allow_pickle=False)
    n = min(a.max_decisions, len(z["episode"]))
    frames = {k: z[k][:n] for k in ("gray", "rgb", "episode")}
    lg = z["logits"][:n].astype(np.float32)
    P = torch.as_tensor(np.exp(lg - lg.max(1, keepdims=True)), device=a.device); P = P / P.sum(1, keepdim=True)
    geom = Geometry.load(a.geometry); fe = FrontEnd(geom, K=6, device=a.device)
    sub = load_subgraph(a.subgraph)
    body = np.asarray(sub["bodyId"]); is_cl = np.asarray(sub["is_clamped"], bool)
    perm = fe.index.permutation_for(body[is_cl])
    feats = _feature_sets(sub, a.neurons)
    core = ConnectomeCore(sub, CoreConfig(param="type_tied"), backend="spmm", device=a.device)
    gm = match_gain(core, g_star=0.95, op_point=a.op_point, seed=0)
    g = torch.Generator(device="cpu").manual_seed(0)
    fe_proj = (torch.randn(int(is_cl.sum()), FE_PROJ_DIM, generator=g) / np.sqrt(is_cl.sum())).to(a.device)
    t0 = time.time()
    Fs = rollout(frames, fe, core, perm, feats, a.batch, a.device, fe_proj)
    print(f"features in {time.time() - t0:.0f}s: " + ", ".join(f"{k}:{v.shape[1]}" for k, v in Fs.items()), flush=True)
    res = {"w0": gm["w0"], "n": n}
    for k, X in Fs.items():
        res[k] = fit_softmax(X.astype(np.float32), P, frames["episode"], a.device)
        print(f"{k:9s} d={res[k]['d']:5d}  held-out agreement {res[k]['agree']:.3f}  (majority {res[k]['majority_agree']:.3f})  KL {res[k]['kl']:.3f}", flush=True)
    # combined dn+vpn
    X = np.concatenate([Fs["dn"], Fs["vpn"]], 1).astype(np.float32)
    res["dn_vpn"] = fit_softmax(X, P, frames["episode"], a.device)
    print(f"{'dn_vpn':9s} d={res['dn_vpn']['d']:5d}  held-out agreement {res['dn_vpn']['agree']:.3f}  (majority {res['dn_vpn']['majority_agree']:.3f})  KL {res['dn_vpn']['kl']:.3f}", flush=True)
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
