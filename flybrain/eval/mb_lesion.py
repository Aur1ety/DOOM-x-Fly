"""Lesion the fly's own memory circuit in silico: knock out a cell type or pathway, then measure what breaks.

A simulated fly you can experiment on. Each lesion is applied to the connectome-built mushroom body, the same
aversive training is run (one calibrated pairing on the intact circuit, then the same rule strength kept fixed),
and the effect on the memory (drop at MBON-gamma1pedc) and the behaviour (avoidance of the trained odour) is
reported next to what real fly lesion studies find. The point is validation: if silencing a cell breaks the
model the way it breaks the animal, the model is a testbed; where it disagrees is a finding. Only the lesions
marked as independent wiring tests can disagree; the others are identities of the rule or the readout.

    python -m flybrain.eval.mb_lesion --wiring $FLYBRAIN_OUT/mb/mb_wiring.npz --out $FLYBRAIN_OUT/mb/lesion.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain import OUT_DIR
from flybrain.eval.mb_behaviour import build_valence
from flybrain.eval.mb_olfactory import MushroomBody, json_safe

# lesion -> (one-line what it does, the real-fly expectation, is the outcome a genuine wiring test?)
# Three rows are identities of the rule or the readout and cannot fail once the dopamine->MBON compartment map is
# taken as given: no dopamine means dW = 0; zeroing MBON11's valence removes the cell that carries ~97 % of the
# depression (that share IS the compartment map, a wiring result of its own); no KC->MBON synapses means nothing
# to depress. They are kept as sanity floors. Only the APL and KCg-m rows test the wiring here.
LESIONS = {
    "none": ("intact circuit (baseline)", "learns and avoids", True),
    "PPL101_silenced": ("silence the punishment dopamine neuron (PPL1-gamma1pedc)", "aversive memory abolished (Aso 2010, 2012); rule identity, cannot fail", False),
    "MBON11_silenced": ("silence the output cell MBON-gamma1pedc in the readout", "learned avoidance lost (Aso 2014, Perisse 2016); readout identity given the compartment map, cannot fail", False),
    "APL_disinhibited": ("remove APL feedback so the Kenyon code is dense, not sparse", "memory loses odour specificity (Lin 2014)", True),
    "KCg-m_silenced": ("silence the main gamma Kenyon cells", "short-term aversive memory impaired (Aso 2014)", True),
    "KC_to_MBON_cut": ("cut all Kenyon-cell -> MBON synapses", "no memory can be stored (mechanistic floor); identity, cannot fail", False),
}


def run(a) -> dict:
    t0 = time.time()
    wiring = a.wiring if (a.wiring and a.wiring.exists()) else None
    z = np.load(a.wiring, allow_pickle=False) if wiring else None
    kc_type = np.array([str(x) for x in z["kc_type"]]) if z is not None else None

    def build(sparsity=0.05):
        return MushroomBody(a.subgraph, a.neurons, sparsity=sparsity, punish=("PPL101",), reward=("PAM",),
                            binary=True, wiring=wiring, modality="olfactory")

    # calibrate the rule ONCE on the intact circuit; keep that strength fixed for every lesion
    intact = build()
    rng = np.random.default_rng(a.seed)
    odours = [rng.choice(len(intact.glom_names), size=6, replace=False) for _ in range(4)]
    m11 = intact.type_mask("MBON11")
    codes0 = [intact.kc_code(intact.odour(o)) for o in odours]
    lr = intact.calibrate_lr(codes0[0], intact.compartment_mask("punish"), a.target, "punish")
    intact_drive_m11 = float(intact.mbon_response(codes0[0])[m11].sum())        # untrained MBON11 drive to A, intact

    def one(lesion: str) -> dict:
        mb = build(sparsity=1.0 if lesion == "APL_disinhibited" else 0.05)
        mb.lr = lr
        val, _ = build_valence(mb, a.neurons)
        if lesion == "PPL101_silenced":
            mb.delta_punish = torch.zeros_like(mb.delta_punish)
        if lesion == "MBON11_silenced":
            val = val.clone(); val[m11] = 0.0
        if lesion == "KC_to_MBON_cut":
            mb.e_base = torch.zeros_like(mb.e_base)
        kc_mask = None; applied = True
        if lesion == "KCg-m_silenced":
            if kc_type is not None and bool(np.char.startswith(kc_type.astype(str), "KCg-m").any()):
                kc_mask = torch.as_tensor(np.char.startswith(kc_type.astype(str), "KCg-m"))   # these KCs cannot fire
            else:
                applied = False                                               # no KCg-m in this wiring/subgraph -> do not report intact-looking numbers

        def code(o):
            c = mb.kc_code(mb.odour(o))
            if kc_mask is not None:
                c = c.clone(); c[kc_mask] = 0.0
            return c
        codes = [code(o) for o in odours]
        before = torch.stack([mb.mbon_response(c) for c in codes])
        norm = float(before.abs().sum(1).mean()) or 1.0

        def score(R):
            return (R * val).sum(1) / norm

        def p_avoid(R, i=1, j=0):                                              # P(choose the other odour over A)
            s = score(R); return float(torch.sigmoid(torch.tensor(a.beta) * (s[i] - s[j])))
        s_avoid0 = p_avoid(before)
        mb.reset()
        for _ in range(a.pairings):
            mb.reinforce(codes[0], "punish")
        after = torch.stack([mb.mbon_response(c) for c in codes])

        def drop(i):
            b = float(before[i][m11].sum()); return round(1 - float(after[i][m11].sum()) / b, 4) if b > 0 else None
        what, expect, independent = LESIONS[lesion]
        drive = float(before[0][m11].sum())
        s_avoid1 = p_avoid(after)
        return {"lesion": lesion, "what": what, "real_fly": expect, "applied": applied, "independent_wiring_test": independent,
                "memory_paired_MBON11": drop(0), "memory_unpaired_MBON11": round(float(np.mean([drop(i) for i in range(1, 4) if drop(i) is not None])), 4) if drop(1) is not None else None,
                "mbon11_drive_to_A": round(drive, 2),
                # a dense code (APL lesion) RAISES the untrained drive, so 'loss' is not a memory measure there
                "drive_loss_vs_intact": (round(1 - drive / intact_drive_m11, 4) if (intact_drive_m11 > 0 and lesion != "APL_disinhibited") else None),
                "avoid_A_before": round(s_avoid0, 4), "avoid_A_after": round(s_avoid1, 4),
                "learned_shift": round(s_avoid1 - s_avoid0, 4)}                        # the behavioural readout: after minus before

    rows = [one(L) for L in LESIONS]
    ref = rows[0]["learned_shift"]
    for r in rows:
        r["learned_shift_frac_of_intact"] = round(r["learned_shift"] / ref, 4) if ref else None
    res = {"anchor": {"lr": round(lr, 4), "target_drop_MBON11": a.target, "pairings": a.pairings, "beta": a.beta},
           "lesions": rows,
           "note": "lr is calibrated once on the intact circuit and held fixed. The fractional memory_paired_MBON11 "
                   "drop is calibration-locked under the binary code (~target when any trained KC->MBON11 edge "
                   "survives, else None), so it CANNOT express graded impairment. Graded memory measure: "
                   "drive_loss_vs_intact (untrained MBON11 drive to A lost to the lesion; not applicable to "
                   "APL_disinhibited, where a dense code RAISES that drive). Behavioural readout: learned_shift = "
                   "avoid_A_after - avoid_A_before, and its fraction of the intact shift. avoid_A_before is "
                   "lesion-specific (the readout is normalised per lesioned circuit), so avoid_A_after alone must "
                   "NOT be compared with 0.5; learned_shift ~0 = no learned avoidance. Rows with "
                   "independent_wiring_test false are identities of the rule or readout given the dopamine->MBON "
                   "compartment map, and cannot fail.",
           "elapsed_s": round(time.time() - t0, 1)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraph", type=Path, default=OUT_DIR / "graph" / "subgraph_v5.npz")
    ap.add_argument("--neurons", type=Path, default=OUT_DIR / "graph" / "neurons.parquet")
    ap.add_argument("--wiring", type=Path, default=OUT_DIR / "mb" / "mb_wiring.npz")
    ap.add_argument("--pairings", type=int, default=1); ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--beta", type=float, default=8.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = json_safe(run(a)); text = json.dumps(res, indent=1, allow_nan=False)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
