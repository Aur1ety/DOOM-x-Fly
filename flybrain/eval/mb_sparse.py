"""Make-or-break check for olfactory learning in the model: when synthetic odours drive the projection
neurons, do the Kenyon cells produce a SPARSE, odour-specific code (as the real mushroom body does), or
do they smear like the visual pathway did?

Real fly: an odour lights up a specific combination of ~a few of the 59 glomeruli; the Kenyon cells then
fire sparsely (~5-10 % active), and the APL giant interneuron's feedback inhibition enforces that
sparseness. Odour-specific learning depends on it. If our rate model can't reproduce sparse KC coding,
odour learning would smear across odours the same way vision did, and that becomes the finding.

We drive the olfactory PNs directly (bypassing the antennal lobe we don't model): each odour = a random
subset of glomeruli held at a high rate, the rest silent. We read the KC code with the APL intact and
with APL ablated (held silent), to see whether APL is what sparsens.

    python -m flybrain.eval.mb_sparse --odours 8 --glom-per-odour 6 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from flybrain import OUT_DIR

PN_RE = re.compile(r"_(ad|l|il|v|vl|il2)?PN$")     # uniglomerular olfactory projection neurons


def _pops(subgraph: Path, neurons: Path):
    import pandas as pd

    from flybrain.model.core import load_subgraph
    sub = load_subgraph(subgraph); body = np.asarray(sub["bodyId"])
    n = pd.read_parquet(neurons, columns=["bodyId", "type"]).drop_duplicates("bodyId").set_index("bodyId")
    t = n["type"].reindex(body).fillna("").astype(str).to_list()
    pn = np.array([bool(PN_RE.search(x)) for x in t])
    kc = np.array([x.startswith("KC") for x in t])
    apl = np.array([x.startswith("APL") for x in t])
    glom = {}
    for i in np.flatnonzero(pn):
        glom.setdefault(t[i].split("_")[0], []).append(int(i))
    return np.flatnonzero(kc), np.flatnonzero(apl), glom


@torch.no_grad()
def present_odour(core, pn_active: np.ndarray, all_pn: np.ndarray, rate_hi: float, decisions: int,
                  apl_idx: np.ndarray | None, ablate_apl: bool, device: str) -> torch.Tensor:
    """Hold the odour's PNs at rate_hi and every other PN at 0, re-imposed EVERY substep (the core does K
    substeps per decision; clamping only per decision lets recurrent drive overwrite the odour for K-1 of
    every K substeps). Optionally silence APL. Return settled per-node rates [M]. Clamped visual input = dark."""
    import dataclasses

    from flybrain.model.core import inv_softplus
    h_hi = float(inv_softplus(torch.tensor(float(rate_hi))))
    all_pn_t = torch.as_tensor(all_pn, device=device, dtype=torch.long)
    act_t = torch.as_tensor(pn_active, device=device, dtype=torch.long)
    apl_t = torch.as_tensor(apl_idx, device=device, dtype=torch.long) if (apl_idx is not None and ablate_apl) else None

    def clamp(h):
        h[all_pn_t, 0] = -20.0                      # silence all PNs (rate ~ 0)
        h[act_t, 0] = h_hi                           # then raise the odour's PNs
        if apl_t is not None:
            h[apl_t, 0] = -20.0                      # ablate APL feedback inhibition
    orig = core.cfg
    core.cfg = dataclasses.replace(orig, K=1, decision_ms=orig.dt_ms)          # 1 substep/step, same dt
    try:
        xc = torch.zeros(1, core.n_clamped, device=device, dtype=core.dtype)
        state = core.init_state(1)
        clamp(state.h)
        for _ in range(decisions * orig.K):
            state = core.step(state, xc)
            clamp(state.h)
        return core.rates(state)[:, 0]
    finally:
        core.cfg = orig


def set_kc_threshold(core, kc_idx: np.ndarray, vrest: float) -> None:
    """Faithful physiology: real Kenyon cells sit near-silent at rest and fire only to strong coincident
    input (a high threshold), which is what makes their code sparse. Set the KC units' resting potential to
    `vrest` (negative -> softplus(vrest) ~ 0 at rest). Grounded in measured KC physiology, disclosed."""
    node_to_unit = torch.full((core.n_nodes,), -1, dtype=torch.long, device=core.v_rest.device)
    node_to_unit[core.dyn_idx] = core.unit_of_dyn
    units = node_to_unit[torch.as_tensor(kc_idx, device=core.v_rest.device, dtype=torch.long)]
    units = units[units >= 0]
    with torch.no_grad():
        core.v_rest[units] = float(vrest)


def run(a) -> dict:
    device = a.device
    if device.startswith("cuda"):
        from flybrain.train.level_bc import _guarded_device
        device = _guarded_device(device)
    from flybrain.model.core import ConnectomeCore, CoreConfig, load_subgraph
    from flybrain.model.gain import match_gain

    t0 = time.time()
    sub = load_subgraph(a.subgraph)
    core = ConnectomeCore(sub, CoreConfig(param="per_edge"), backend="spmm", device=device)
    for p in core.parameters():
        p.requires_grad_(False)
    gm = match_gain(core, g_star=0.95, op_point=a.op_point, seed=0)
    kc_idx, apl_idx, glom = _pops(a.subgraph, a.neurons)
    if a.kc_vrest is not None:
        set_kc_threshold(core, kc_idx, a.kc_vrest)
    all_pn = np.concatenate([np.array(v) for v in glom.values()])
    glom_names = sorted(glom)
    rng = np.random.default_rng(a.seed)
    odours = [rng.choice(len(glom_names), size=a.glom_per_odour, replace=False) for _ in range(a.odours)]

    kc_t = torch.as_tensor(kc_idx, device=device, dtype=torch.long)

    def code(ablate):
        rates = []
        for od in odours:
            active = np.concatenate([glom[glom_names[g]] for g in od])
            r = present_odour(core, active, all_pn, a.rate_hi, a.decisions, apl_idx, ablate, device)
            rates.append(r.index_select(0, kc_t))
        return torch.stack(rates)                    # [O, n_kc]

    def report(K, thr=0.1):
        frac = [round(float((K[i] > thr).float().mean()), 4) for i in range(len(odours))]
        off_mask = ~np.eye(len(odours), dtype=bool)
        x = K / (K.norm(dim=1, keepdim=True) + 1e-9)
        off = (x @ x.T).cpu().numpy()[off_mask]
        Kc = K - K.mean(dim=0, keepdim=True)                          # remove the per-KC common-mode pedestal
        xc = Kc / (Kc.norm(dim=1, keepdim=True) + 1e-9)
        offc = (xc @ xc.T).cpu().numpy()[off_mask]
        # decodability: can nearest-centroid tell the odours apart on the residual (chance = 1/O)?
        d = torch.cdist(Kc, Kc); d.fill_diagonal_(float("inf"))
        nn_correct = float((d.argmin(1) == torch.arange(len(odours), device=K.device)).float().mean())
        return {"kc_frac_active": frac, "kc_frac_active_mean": round(float(np.mean(frac)), 4),
                "kc_mean_rate": round(float(K.mean()), 4),
                "cross_odour_cos_mean": round(float(off.mean()), 4), "cross_odour_cos_max": round(float(off.max()), 4),
                "cross_odour_cos_centered_mean": round(float(offc.mean()), 4), "cross_odour_cos_centered_max": round(float(offc.max()), 4),
                "nearest_neighbour_odour_id_acc": round(nn_correct, 4), "chance": round(1.0 / len(odours), 4)}

    with_apl = report(code(ablate=False))
    without_apl = report(code(ablate=True))
    res = {"w0": gm["w0"], "n_KC": len(kc_idx), "n_glomeruli": len(glom_names), "n_PN": len(all_pn),
           "n_APL": len(apl_idx), "odours": a.odours, "glom_per_odour": a.glom_per_odour,
           "with_APL": with_apl, "APL_ablated": without_apl,
           "verdict_note": "sparse (kc_frac_active ~0.05-0.15) and distinct (cross_odour_cos low) with APL = olfactory learning viable; "
                           "APL ablation should raise frac_active if APL is sparsening; frac_active ~1 even with APL = rate model cannot code odours",
           "elapsed_s": round(time.time() - t0, 1)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraph", type=Path, default=OUT_DIR / "graph" / "subgraph_v5.npz")
    ap.add_argument("--neurons", type=Path, default=OUT_DIR / "graph" / "neurons.parquet")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--op-point", default="rest")
    ap.add_argument("--odours", type=int, default=8); ap.add_argument("--glom-per-odour", type=int, default=6)
    ap.add_argument("--rate-hi", type=float, default=2.0); ap.add_argument("--decisions", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--kc-vrest", type=float, default=None, help="set Kenyon-cell resting potential (negative = high threshold, sparse). None = unchanged")
    a = ap.parse_args(argv)
    res = run(a)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
