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

Decodability is measured properly: each odour is presented `reps` times with per-PN rate jitter, and a
leave-one-out nearest-centroid decoder has to name the odour of a held-out presentation from the other
presentations (chance 1/odours; a perfectly odour-specific code scores 1). One presentation per odour with
the self-match excluded is identically 0 for ANY code and proves nothing; that was the previous version.

    python -m flybrain.eval.mb_sparse --odours 6 --reps 5 --device cuda:0 --out $FLYBRAIN_OUT/mb/kcsparse2_none.json
    python -m flybrain.eval.mb_sparse --odours 6 --reps 5 --device cuda:0 --kc-vrest -5 --out $FLYBRAIN_OUT/mb/kcsparse2_-5.json
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


def loo_nearest_centroid(K: torch.Tensor, y: torch.Tensor) -> float:
    """Leave-one-out nearest-centroid decoding accuracy. For every sample the class centroids are rebuilt from all
    OTHER samples, and the sample is assigned to the nearest centroid. Chance = 1/n_classes; a perfectly
    class-specific code scores 1. Needs >= 2 samples per class (returns nan otherwise)."""
    y = y.to(K.device); classes = torch.unique(y)
    sums = torch.stack([K[y == c].sum(0) for c in classes]); counts = torch.stack([(y == c).sum() for c in classes]).float()
    if bool((counts < 2).any()):
        return float("nan")
    correct = 0
    for j in range(len(K)):
        ci = int((classes == y[j]).nonzero()[0, 0])
        s = sums.clone(); cnt = counts.clone(); s[ci] -= K[j]; cnt[ci] -= 1
        pred = classes[torch.cdist(K[j][None], s / cnt[:, None]).argmin()]
        correct += int(pred == y[j])
    return correct / len(K)


@torch.no_grad()
def present_odour(core, pn_active: np.ndarray, all_pn: np.ndarray, pn_rates: np.ndarray, decisions: int,
                  apl_idx: np.ndarray | None, ablate_apl: bool, device: str) -> torch.Tensor:
    """Hold the odour's PNs at their given rates (one per active PN) and every other PN at 0, re-imposed EVERY
    substep (the core does K substeps per decision; clamping only per decision lets recurrent drive overwrite
    the odour for K-1 of every K substeps). Optionally silence APL. Return settled per-node rates [M]. Clamped
    visual input = dark."""
    import dataclasses

    from flybrain.model.core import inv_softplus
    h_hi = inv_softplus(torch.as_tensor(np.asarray(pn_rates, dtype=np.float32), device=device))
    all_pn_t = torch.as_tensor(all_pn, device=device, dtype=torch.long)
    act_t = torch.as_tensor(pn_active, device=device, dtype=torch.long)
    apl_t = torch.as_tensor(apl_idx, device=device, dtype=torch.long) if (apl_idx is not None and ablate_apl) else None

    def clamp(h):
        h[all_pn_t, 0] = -20.0                      # silence all PNs (rate ~ 0)
        h[act_t, 0] = h_hi.to(h.dtype)               # then raise the odour's PNs
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
    O, R = len(odours), a.reps
    y = torch.arange(O).repeat_interleave(R)                                  # odour label per presentation

    def code(ablate):
        """[O*R, n_kc]: each odour presented R times with per-PN multiplicative rate jitter (lognormal, sigma = jitter),
        so a decoder can be tested on held-out presentations."""
        rates = []
        for od in odours:
            active = np.concatenate([glom[glom_names[g]] for g in od])
            for _ in range(R):
                pn_rates = a.rate_hi * np.exp(rng.normal(0.0, a.jitter, size=len(active)))
                r = present_odour(core, active, all_pn, pn_rates, a.decisions, apl_idx, ablate, device)
                rates.append(r.index_select(0, kc_t))
        return torch.stack(rates)

    def report(K, thr=0.1):
        frac = [round(float((K[i] > thr).float().mean()), 4) for i in range(len(K))]
        x = K / (K.norm(dim=1, keepdim=True) + 1e-9); C = (x @ x.T).cpu().numpy()
        same = (y[:, None] == y[None, :]).numpy(); eye = np.eye(len(K), dtype=bool)
        cross = C[~same]; within = C[same & ~eye]
        # decodability on held-out presentations (chance 1/O); plus how far apart the odour centroids sit relative
        # to the presentation-to-presentation scatter within an odour
        acc = loo_nearest_centroid(K, y)
        cents = torch.stack([K[y.to(K.device) == c].mean(0) for c in range(O)])
        between = torch.pdist(cents).mean() if O > 1 else torch.tensor(0.0)
        within_d = torch.stack([(K[y.to(K.device) == c] - cents[c]).norm(dim=1).mean() for c in range(O)]).mean()
        return {"kc_frac_active_mean": round(float(np.mean(frac)), 4), "kc_frac_active_min_max": [min(frac), max(frac)],
                "kc_mean_rate": round(float(K.mean()), 4),
                "cross_odour_cos_mean": round(float(cross.mean()), 4), "cross_odour_cos_max": round(float(cross.max()), 4),
                "within_odour_cos_mean": round(float(within.mean()), 4) if len(within) else None,
                "loo_nearest_centroid_acc": None if np.isnan(acc) else round(acc, 4), "chance": round(1.0 / O, 4),
                "centroid_separation_over_within_scatter": round(float(between / (within_d + 1e-9)), 4)}

    # decoder self-check on synthetic codes: a perfectly odour-specific code must decode at 1, shuffled labels near chance
    g = torch.Generator().manual_seed(0)
    synth = torch.repeat_interleave(torch.eye(O), R, dim=0) + 0.05 * torch.randn(O * R, O, generator=g)
    self_check = {"separable_code_acc": round(loo_nearest_centroid(synth, y), 4),
                  "shuffled_labels_acc": round(loo_nearest_centroid(synth, y[torch.randperm(O * R, generator=g)]), 4), "chance": round(1.0 / O, 4)}

    with_apl = report(code(ablate=False))
    without_apl = report(code(ablate=True))
    res = {"w0": gm["w0"], "n_KC": len(kc_idx), "n_glomeruli": len(glom_names), "n_PN": len(all_pn),
           "n_APL": len(apl_idx), "odours": a.odours, "glom_per_odour": a.glom_per_odour, "reps_per_odour": R, "pn_rate_jitter": a.jitter,
           "kc_vrest": a.kc_vrest, "decoder_self_check": self_check,
           "with_APL": with_apl, "APL_ablated": without_apl,
           "verdict_note": "sparse (kc_frac_active ~0.05-0.15) and decodable (loo_nearest_centroid_acc >> chance) with APL = olfactory "
                           "learning viable; APL ablation should raise frac_active if APL is sparsening; frac_active ~1 or ~0 at every "
                           "threshold with decoding at chance = the rate model cannot code odours. The decoder is scored on held-out "
                           "presentations; the earlier one-presentation nearest-neighbour test was identically 0 and is gone.",
           "elapsed_s": round(time.time() - t0, 1)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraph", type=Path, default=OUT_DIR / "graph" / "subgraph_v5.npz")
    ap.add_argument("--neurons", type=Path, default=OUT_DIR / "graph" / "neurons.parquet")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--op-point", default="rest")
    ap.add_argument("--odours", type=int, default=6); ap.add_argument("--glom-per-odour", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5, help="presentations per odour (held-out decoding needs >= 2)")
    ap.add_argument("--jitter", type=float, default=0.25, help="per-PN lognormal rate jitter (sigma) between presentations")
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
