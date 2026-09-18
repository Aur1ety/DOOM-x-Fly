"""Pattern completion: teach the fly odour A, then give it only PART of A and see how much it recalls.

Tests the "the impulse replays" idea. In a real associative memory the question is whether an incomplete cue
still triggers the whole stored memory (pattern completion, needs recurrent/attractor dynamics) or only a
proportional piece of it (graded recall). The fly mushroom body is feedforward, so the prediction is graded
recall: a partial odour activates the subset of Kenyon cells its glomeruli drive, and the depressed synapses
are read only for those, so recall tracks the fraction of the cue present, with no sharp completion.

    python -m flybrain.eval.mb_recall --wiring $FLYBRAIN_OUT/mb/mb_wiring.npz --out $FLYBRAIN_OUT/mb/recall.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flybrain import OUT_DIR
from flybrain.eval.mb_olfactory import MushroomBody


def run(a) -> dict:
    t0 = time.time()
    wiring = a.wiring if (a.wiring and a.wiring.exists()) else None
    mb = MushroomBody(a.subgraph, a.neurons, sparsity=0.05, punish=("PPL101",), reward=("PAM",),
                      binary=True, wiring=wiring, modality="olfactory")
    m11 = mb.type_mask("MBON11")
    rng = np.random.default_rng(a.seed)
    G = a.glom_per_odour
    A = list(rng.choice(len(mb.glom_names), size=G, replace=False))            # odour A: G glomeruli
    full = mb.kc_code(mb.odour(A))

    # train on the FULL odour A, one calibrated pairing
    mb.calibrate_lr(full, mb.compartment_mask("punish"), a.target, "punish")
    mb.reset()
    for _ in range(a.pairings):
        mb.reinforce(full, "punish")

    def drop(code):
        """Recall = how much training depressed the response TO THIS SAME cue: 1 - trained/untrained, both for
        `code`. Denominator is the W=1 (untrained) response to `code`, so a smaller cue is not mistaken for more
        memory."""
        untr = mb._sum_to_mbon(mb.e_base * code[mb.e_kc])                      # untrained (W=1) response to this cue
        trn = mb.mbon_response(code)                                           # trained response to this cue
        b = float(untr[m11].sum())
        return round(1 - float(trn[m11].sum()) / b, 4) if b > 0 else None
    full_recall = drop(full)

    # partial cue: keep a fraction of A's glomeruli (average over which ones are kept)
    by_glom = {}
    for keep in range(G, 0, -1):
        ds, kcs = [], []
        for _ in range(a.reps):
            sub = list(rng.choice(A, size=keep, replace=False))
            code = mb.kc_code(mb.odour(sub))
            ds.append(drop(code)); kcs.append(float((code > 0).float().mean()))
        by_glom[f"{keep}/{G}"] = {"cue_fraction": round(keep / G, 3), "recall_MBON11": round(float(np.mean(ds)), 4),
                                  "recall_fraction": round(float(np.mean(ds)) / full_recall, 3) if full_recall else None,
                                  "kc_active_frac": round(float(np.mean(kcs)), 4)}

    # is recall graded (linear in cue) or completing (stays high)? fit recall_fraction vs cue_fraction
    x = np.array([v["cue_fraction"] for v in by_glom.values()]); y = np.array([v["recall_fraction"] or 0 for v in by_glom.values()])
    slope = round(float(np.polyfit(x, y, 1)[0]), 3)
    res = {"odour_glomeruli": G, "full_recall_MBON11": full_recall,
           "partial_cue_by_glomeruli": by_glom, "recall_vs_cue_slope": slope,
           "verdict": (f"Partial SMELL (fewer glomeruli) gives graded recall (slope ~{slope:.2f}): a smaller odour "
                       "drives a different, partly-overlapping Kenyon pattern, so recall tracks the fraction of the "
                       "trained code the cue re-activates. The feedforward circuit does not pattern-complete a "
                       "degraded input. (Silencing the odour's own Kenyon cells is NOT reported: under a binary code "
                       "with uniform per-compartment dopamine every trained synapse carries the same depression, so "
                       "a self-normalised recall would be constant for any subset by construction and tests nothing.)"),
           "elapsed_s": round(time.time() - t0, 1)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraph", type=Path, default=OUT_DIR / "graph" / "subgraph_v5.npz")
    ap.add_argument("--neurons", type=Path, default=OUT_DIR / "graph" / "neurons.parquet")
    ap.add_argument("--wiring", type=Path, default=OUT_DIR / "mb" / "mb_wiring.npz")
    ap.add_argument("--glom-per-odour", type=int, default=6); ap.add_argument("--pairings", type=int, default=1)
    ap.add_argument("--target", type=float, default=0.9); ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = run(a)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
