"""Embed the learned mushroom-body memory inside the full recurrent brain (the Doom model).

The recurrent rate model cannot compute a sparse, odour-specific Kenyon-cell code (mb_sparse.py: it is flat,
cross-odour cosine 1.0 at every threshold). But real Kenyon cells are feedforward coincidence detectors, not
recurrently driven units, so that flatness is a property of the uniform rate model, not of the fly. The honest
embedding: compute the KC code the way the cell actually works (real PN->KC wiring + k-winners-take-all, the
same code the feedforward model uses), inject it at the KC layer of the full recurrent core, apply the learned
KC->MBON synapse changes to the core's OWN edges, and read the memory at the MBONs and descending neurons
THROUGH the full recurrent brain. The memory lives on the connectome's real synapses; only KC activity is
computed feedforward, which is disclosed.

    python -m flybrain.eval.mb_embed --device cuda:1 --punish PPL101 --out $FLYBRAIN_OUT/mb/embed.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain import OUT_DIR


@torch.no_grad()
def present_with_kc(core, kc_nodes: torch.Tensor, kc_rate: torch.Tensor, decisions: int, device: str) -> torch.Tensor:
    """Run the recurrent core in the dark while pinning the KC nodes to the feedforward code EVERY substep
    (the core does K substeps per decision; pinning only per decision lets the KCs drift). Return settled
    per-node rates [M]. kc_rate is the target rate for each KC node (length n_KC, 0 = silent)."""
    import dataclasses

    from flybrain.model.core import inv_softplus
    kc_rate = kc_rate.to(device)
    h_pin = inv_softplus(kc_rate.clamp_min(1e-6))
    h_pin[kc_rate <= 0] = -20.0
    orig = core.cfg
    core.cfg = dataclasses.replace(orig, K=1, decision_ms=orig.dt_ms)          # 1 substep/step, same dt
    try:
        xc = torch.zeros(1, core.n_clamped, device=device, dtype=core.dtype)
        state = core.init_state(1)
        state.h[kc_nodes, 0] = h_pin
        for _ in range(decisions * orig.K):
            state = core.step(state, xc)
            state.h[kc_nodes, 0] = h_pin
        return core.rates(state)[:, 0]
    finally:
        core.cfg = orig


def run(a) -> dict:
    device = a.device
    if device.startswith("cuda"):
        from flybrain.train.level_bc import _guarded_device
        device = _guarded_device(device)
    from flybrain.eval.mb_olfactory import MushroomBody
    from flybrain.model.core import ConnectomeCore, CoreConfig, load_subgraph
    from flybrain.model.gain import match_gain
    from flybrain.model.mb_plasticity import MushroomBodyPlasticity, node_types

    t0 = time.time()
    sub = load_subgraph(a.subgraph)
    core = ConnectomeCore(sub, CoreConfig(param="per_edge"), backend="spmm", device=device)
    for p in core.parameters():
        p.requires_grad_(False)
    match_gain(core, g_star=0.95, op_point="rest", seed=0)
    types = node_types(sub, a.neurons)
    plas = MushroomBodyPlasticity(core, types, lr=a.lr, w_max=a.w_max, device=device,
                                  punish_types=tuple(a.punish), reward_types=tuple(a.reward))
    # feedforward code generator on the SAME subgraph (indices align with the core's nodes)
    ff = MushroomBody(a.subgraph, a.neurons, a.sparsity, tuple(a.punish), tuple(a.reward), a.lr, a.w_max,
                      None, a.seed, binary=True)
    kc_nodes = torch.as_tensor(ff.kc, device=device, dtype=torch.long)         # KC node indices (subgraph order)
    mbon_nodes = plas.mbon_idx_t
    mbon_type = ff.mbon_type
    m11_local = np.flatnonzero(np.array([t == "MBON11" for t in mbon_type]))   # index into MBON list
    dn_local = np.flatnonzero(np.char.startswith(types.astype("U16"), "DN"))
    dn_nodes = torch.as_tensor(dn_local, device=device, dtype=torch.long)

    rng = np.random.default_rng(a.seed)
    odours = [rng.choice(len(ff.glom_names), size=a.glom_per_odour, replace=False) for _ in range(a.odours)]
    codes = [ff.kc_code(ff.odour(o)) for o in odours]                          # binary sparse codes over ff.kc

    def code_rate(code):
        r = torch.zeros(core.n_nodes, device=device)
        r[kc_nodes] = code.to(device) * a.kc_rate
        return r

    def mbon_of(node_rates):
        return node_rates.index_select(0, mbon_nodes)

    def measure(code):
        r = present_with_kc(core, kc_nodes, code * a.kc_rate, a.decisions, device)
        return mbon_of(r).cpu(), r.index_select(0, dn_nodes).cpu()

    # baseline (untrained core)
    plas.reset()
    base_mbon, base_dn = zip(*[measure(c) for c in codes])
    base_mbon = torch.stack(base_mbon); base_dn = torch.stack(base_dn)

    # calibrate lr so ONE pairing gives the target MBON11 drop inside the recurrent brain (bisection on lr)
    full0 = code_rate(codes[0])

    def paired_drop_at_lr(lr):
        plas.lr = lr; plas.reset()
        plas.reinforce(full0, "punish", a.strength)
        m, _ = measure(codes[0])
        b = float(base_mbon[0][m11_local].sum())
        return (1 - float(m[m11_local].sum()) / b) if b > 0 else 0.0

    HI = 8.0
    if a.lr_auto:
        lo, hi = 0.0, HI
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            (lo, hi) = (mid, hi) if paired_drop_at_lr(mid) < a.target_drop else (lo, mid)
        plas.lr = 0.5 * (lo + hi)
    calib_lr = plas.lr
    achieved = paired_drop_at_lr(calib_lr)                                     # actual paired drop at the reported lr
    plateau = paired_drop_at_lr(HI)                                            # max the recurrent readout can express (lr>=1 zeroes the synapse)
    calibrated_ok = bool(a.lr_auto and achieved >= a.target_drop - 0.02)
    lr_capped = bool(calib_lr >= HI - 1e-3)

    def drop(before, after, idx):
        b = float(before[idx].sum())
        return round(1 - float(after[idx].sum()) / b, 4) if b > 0 else None

    # train odour A, measure specificity at MBON11 and the DN change, inside the recurrent brain
    plas.lr = calib_lr; plas.reset()
    for _ in range(a.pairings):
        plas.reinforce(code_rate(codes[0]), "punish", a.strength)
    aft_mbon, aft_dn = zip(*[measure(c) for c in codes])
    aft_mbon = torch.stack(aft_mbon); aft_dn = torch.stack(aft_dn)

    dn_types = [str(types[i]) for i in dn_local]

    def rel(b, a_):
        bb = float(b.abs().sum()); return round(float((a_ - b).abs().sum() / bb), 4) if bb > 0 else None

    def dn_detail(before, after):
        d = (after - before).abs()
        named = {n: round(float(d[[i for i, t in enumerate(dn_types) if t == n]].sum()), 4) for n in ("DNa02", "DNa03") if n in dn_types}
        return {"rel_change_whole_brain": rel(before, after), "max_abs_rate_change": round(float(d.max()), 4),
                "n_DN_changed_gt_1pct_of_max": int((d > 0.01 * float(after.abs().max() + 1e-9)).sum()), "named_DN_abs_change": named}

    res = {"embedding": "feedforward KC code injected into the full recurrent core; learned synapses on the core's own KC->MBON edges",
           "circuit": {"n_nodes": int(core.n_nodes), "n_edges": int(core.n_edges), "n_KC": len(kc_nodes),
                       "n_MBON": len(mbon_type), "n_plastic_KC_MBON_edges": plas.n_plastic, "n_DN": len(dn_local),
                       "MBON11_cells": len(m11_local), "subgraph": str(a.subgraph)},
           "rule": {"lr": round(calib_lr, 4), "lr_calibrated_ok": calibrated_ok, "lr_capped_at_ceiling": lr_capped,
                    "target_drop": a.target_drop if a.lr_auto else None, "pairings": a.pairings,
                    "note": "paired_A_MBON11 magnitude is set by lr (calibrated), not by the wiring; the wiring-derived claims are "
                            "the unpaired specificity and the DN readout. Any lr>=~1 zeroes the MBON11 KC synapses, so the drop plateaus."},
           "kc_code": {"active_frac": round(float((codes[0] > 0).float().mean()), 4)},
           "WIRING_DERIVED_specificity": {
               "unpaired_MBON11_mean": round(float(np.mean([drop(base_mbon[i], aft_mbon[i], m11_local) for i in range(1, len(codes))])), 4),
               "unpaired_MBON11_each": [drop(base_mbon[i], aft_mbon[i], m11_local) for i in range(1, len(codes))],
               "paired_over_unpaired_ratio": None},
           "calibrated_paired_A_MBON11": {"value": drop(base_mbon[0], aft_mbon[0], m11_local),
                                          "max_achievable_drop_plateau": round(plateau, 4),
                                          "note": "calibrated/tuning-dependent, not wiring evidence"},
           "dn_readout_through_recurrent_brain": {
               "n_DN_driven": int((base_dn[0] != 0).sum()),
               "trained_A": dn_detail(base_dn[0], aft_dn[0]),
               "untouched_mean_rel_change": round(float(np.mean([rel(base_dn[i], aft_dn[i]) for i in range(1, len(codes))])), 4),
               "note": "even the max single-DN rate change is the honest test of whether the memory reaches steering; whole-brain ratio dilutes it"},
           "elapsed_s": round(time.time() - t0, 1)}
    up = res["WIRING_DERIVED_specificity"]["unpaired_MBON11_mean"]; pd_ = res["calibrated_paired_A_MBON11"]["value"]
    res["WIRING_DERIVED_specificity"]["paired_over_unpaired_ratio"] = round(pd_ / up, 2) if up else None
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraph", type=Path, default=OUT_DIR / "graph" / "subgraph_v5.npz")
    ap.add_argument("--neurons", type=Path, default=OUT_DIR / "graph" / "neurons.parquet")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--sparsity", type=float, default=0.05); ap.add_argument("--odours", type=int, default=6)
    ap.add_argument("--glom-per-odour", type=int, default=6); ap.add_argument("--pairings", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.9); ap.add_argument("--lr-auto", action="store_true")
    ap.add_argument("--target-drop", type=float, default=0.9); ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--w-max", type=float, default=2.0); ap.add_argument("--kc-rate", type=float, default=1.0)
    ap.add_argument("--decisions", type=int, default=30)
    ap.add_argument("--punish", nargs="+", default=["PPL101"]); ap.add_argument("--reward", nargs="+", default=["PAM"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = run(a)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
