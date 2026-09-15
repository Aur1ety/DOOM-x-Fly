"""Throughput / memory of ConnectomeCore on a REAL subgraph file (forward and forward+backward).

Reports brain-steps/s = batch * K * decisions / wall, with optional per-decision gradient
checkpointing (the K substeps are recomputed in backward).

    python -m flybrain.model.bench_real --subgraph $FLYBRAIN_OUT/graph/subgraph_v5.npz --batch 64 256
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from torch.utils.checkpoint import checkpoint

from flybrain.model.core import ConnectomeCore, CoreConfig, CoreState, load_subgraph


def _rollout(model: ConnectomeCore, xc: torch.Tensor, T: int, train: bool, ckpt: bool) -> None:
    st = model.init_state(xc.shape[1])
    with torch.set_grad_enabled(train):
        for _ in range(T):
            if train and ckpt:
                def f(h, x_clamped, r_prev, x):
                    return tuple(model.step(CoreState(h, x_clamped, r_prev), x))
                st = CoreState(*checkpoint(f, st.h, st.x_clamped, st.r_out_prev, xc, use_reentrant=False))
            else:
                st = model.step(st, xc)
        if train:
            model.readout_features(st).sum().backward()


def bench(path: str, params: list[str], batches: list[int], T: int, device: str, ckpt: bool, reps: int) -> list[dict]:
    sub = load_subgraph(path)
    rows = []
    for param in params:
        model = ConnectomeCore(sub, CoreConfig(param=param), backend="spmm", device=device)
        K = model.cfg.K
        print(f"# {param}: nodes {model.n_nodes:,} edges {model.n_edges:,} trainable {model.n_trainable():,}", flush=True)
        for B in batches:
            xc = torch.rand(K, B, model.n_clamped, device=device)
            for train in (False, True):
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                try:
                    _rollout(model, xc, T, train, ckpt)                       # warm-up
                    model.zero_grad(set_to_none=True)
                    if device.startswith("cuda"): torch.cuda.synchronize()
                    t0 = time.time()
                    for _ in range(reps):
                        _rollout(model, xc, T, train, ckpt); model.zero_grad(set_to_none=True)
                    if device.startswith("cuda"): torch.cuda.synchronize()
                    wall = time.time() - t0
                    row = dict(param=param, batch=B, T=T, K=K, train=train, checkpoint=ckpt,
                               brain_steps_per_s=B * K * T * reps / wall,
                               decisions_per_s=B * T * reps / wall,
                               peak_gib=(torch.cuda.max_memory_allocated() / 2**30) if device.startswith("cuda") else None)
                except torch.OutOfMemoryError:
                    row = dict(param=param, batch=B, T=T, K=K, train=train, checkpoint=ckpt, oom=True)
                    if device.startswith("cuda"): torch.cuda.empty_cache()
                rows.append(row)
                print(json.dumps(row), flush=True)
            del xc
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraph", required=True)
    ap.add_argument("--param", nargs="+", default=["type_tied", "per_edge"])
    ap.add_argument("--batch", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-checkpoint", action="store_true")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rows = bench(a.subgraph, a.param, a.batch, a.T, a.device, not a.no_checkpoint, a.reps)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
