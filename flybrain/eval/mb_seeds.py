"""Spread for the section 7 and 8 headline numbers: the full mb_olfactory battery over many seeds (fresh odour
draws each time), for each sense, reported as mean, SD, min and max. A single seed's numbers are one draw of
eight synthetic odours; a circuit property should survive the draw.

    python -m flybrain.eval.mb_seeds --wiring $FLYBRAIN_OUT/mb/mb_wiring.npz --seeds 10 --out-dir $FLYBRAIN_OUT/mb
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flybrain import OUT_DIR
from flybrain.eval import mb_olfactory

# (label, extractor) for the numbers the write-up quotes
METRICS = {
    "kc_active_frac": lambda r: r["kc_code"]["active_frac"][0],
    "kc_cross_odour_cos_mean": lambda r: r["kc_code"]["cross_odour_cos_mean"],
    "paired_A_MBON11": lambda r: r["one_memory"]["paired_A_MBON11"],
    "unpaired_MBON11_mean": lambda r: r["one_memory"]["unpaired_MBON11_mean"],
    "unpaired_MBON11_worst_odour": lambda r: max(r["one_memory"]["unpaired_MBON11_each"]),
    "share_of_depression_in_MBON11": lambda r: r["one_memory"]["share_of_depression_in_MBON11"],
    "reward_C_reward_compartments": lambda r: r["reward_memory"]["paired_C_reward_compartments"],
    "reward_C_at_MBON11": lambda r: r["reward_memory"]["paired_C_MBON11"],
    "coexist_A_punish_C_reward_A_MBON11": lambda r: r["coexistence"]["different_compartments_A_punish_C_reward"]["A_MBON11"],
    "coexist_A_punish_C_reward_C_compartments": lambda r: r["coexistence"]["different_compartments_A_punish_C_reward"]["C_reward_compartments"],
    "same_compartment_A_retained_fraction": lambda r: r["coexistence"]["same_compartment_blocked_A_then_B"]["A_retained_fraction"],
}


def summarise(vals: list[float]) -> dict:
    v = np.asarray([x for x in vals if x is not None], float)
    if not len(v):
        return {"n": 0}
    return {"mean": round(float(v.mean()), 4), "sd": round(float(v.std()), 4), "min": round(float(v.min()), 4),
            "max": round(float(v.max()), 4), "n": len(v)}


def run(a) -> dict:
    t0 = time.time()
    out = {"n_seeds": a.seeds, "seeds": list(range(a.seed, a.seed + a.seeds)), "per_modality": {}}
    for modality in a.modalities:
        runs = []
        for s in range(a.seed, a.seed + a.seeds):
            argv = ["--binary-code", "--wiring", str(a.wiring), "--modality", modality, "--seed", str(s)]
            runs.append(mb_olfactory.json_safe(mb_olfactory.run(mb_olfactory.build_parser().parse_args(argv))))
        summary = {k: summarise([f(r) for r in runs]) for k, f in METRICS.items()}
        gen = {}
        for key in runs[0]["generalisation"]:
            gen[key] = summarise([r["generalisation"][key]["drop_MBON11"] for r in runs])
        summary["generalisation_drop_MBON11"] = gen
        summary["circuit"] = {k: runs[0]["circuit"][k] for k in ("n_input", "n_channels", "n_KC", "n_drivable_KC", "kwta_k", "n_MBON")}
        out["per_modality"][modality] = summary
    out["note"] = ("each seed is a fresh draw of eight synthetic stimuli (six channels each); paired_A_MBON11 is calibrated "
                   "and should sit at the target in every seed; the wiring-dependent numbers are the unpaired drops "
                   "(mean AND worst single odour), the compartment share, the generalisation curve and the coexistence rows")
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiring", type=Path, default=OUT_DIR / "mb" / "mb_wiring.npz")
    ap.add_argument("--modalities", nargs="+", default=["olfactory", "visual"])
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    res = run(a)
    text = json.dumps(res, indent=1, allow_nan=False)
    if a.out_dir:
        a.out_dir.mkdir(parents=True, exist_ok=True); (a.out_dir / "mb_seeds.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
