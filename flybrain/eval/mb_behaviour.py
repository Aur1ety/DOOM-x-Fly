"""Brick 4/5: does the learned memory change what the fly does? Two readouts of the same learned mushroom
body, for either sense (olfactory or visual), built on the full-connectome wiring cache (mb_build.py).

1. Choice through the published valence map (the "organ"): Aso et al. 2014, activating GABAergic or
   cholinergic MBONs drives approach, glutamatergic MBONs drive avoidance. Net approach drive for a
   stimulus X is s(X) = (approach-MBON drive - avoidance-MBON drive) / baseline total. A two-arm choice is
   P(choose X over Y) = sigmoid(beta * (s(X) - s(Y))), beta the one motor gain (disclosed, swept). The
   performance index follows the reciprocal T-maze (Tully & Quinn): PI = 1/2[P(B)-P(A) | A punished] +
   1/2[P(A)-P(B) | B punished]. Ground truth: wild-type single-cycle PI 0.44-0.53.

2. Wiring readout: how much the learned change reaches the descending neurons through the connectome's own
   MBON -> DN direct and MBON -> one interneuron -> DN routes (both precomputed in the cache; a relay is an
   interneuron, never another MBON or a DN). Reported for the trained stimulus overall and at the steering DNs
   DNa02 / DNa03. The two routes are reported separately, never summed.

    python -m flybrain.eval.mb_behaviour --wiring $FLYBRAIN_OUT/mb/mb_wiring.npz --modality olfactory --binary-code
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain import OUT_DIR
from flybrain.eval.mb_olfactory import MushroomBody, json_safe

APPROACH_NT = {"gaba", "acetylcholine"}
AVOID_NT = {"glutamate"}


def build_valence(mb: MushroomBody, neurons: Path):
    """MBON transmitter -> valence, mapped by MBON type (works for any wiring source). +1 approach
    (GABA/ACh), -1 avoidance (glutamate), 0 unknown."""
    import pandas as pd
    n = pd.read_parquet(neurons, columns=["type", "consensus_nt"]).dropna(subset=["type"])
    n["type"] = n["type"].astype(str); n["consensus_nt"] = n["consensus_nt"].fillna("").astype(str)
    nt_of_type = n[n["consensus_nt"] != ""].groupby("type")["consensus_nt"].agg(lambda s: s.value_counts().index[0]).to_dict()

    def val(tp):
        nt = nt_of_type.get(tp, "")
        return 1.0 if nt in APPROACH_NT else (-1.0 if nt in AVOID_NT else 0.0)
    v = np.array([val(tp) for tp in mb.mbon_type])
    counts = {"approach": int((v > 0).sum()), "avoid": int((v < 0).sum()), "unknown": int((v == 0).sum())}
    return torch.as_tensor(v, dtype=torch.float32), counts


def load_dn_readout(wiring: Path):
    """MBON->DN direct and 2-hop matrices [DN, MBON] from the cache, plus DNa02/DNa03 row indices."""
    z = np.load(wiring, allow_pickle=False)
    W1 = torch.as_tensor(z["dn_mbon_direct"], dtype=torch.float32)
    W2 = torch.as_tensor(z["dn_mbon_2hop"], dtype=torch.float32)
    dn_type = [str(x) for x in z["dn_type"]]
    rows = {name: [i for i, t in enumerate(dn_type) if t == name] for name in ("DNa02", "DNa03")}
    return W1, W2, rows, len(dn_type)


def run(a) -> dict:
    t0 = time.time()
    lr0 = 0.9 if a.lr == "auto" else float(a.lr)
    mb = MushroomBody(a.subgraph, a.neurons, a.sparsity, tuple(a.punish), tuple(a.reward), lr0, a.w_max, a.shuffle,
                      a.seed, a.binary_code, a.recover_rate, wiring=a.wiring, modality=a.modality)
    val, counts = build_valence(mb, a.neurons)
    W1, W2, dn_rows, n_dn = load_dn_readout(a.wiring)
    rng = np.random.default_rng(a.seed)
    stimuli = [rng.choice(len(mb.glom_names), size=a.glom_per_odour, replace=False) for _ in range(a.odours)]
    codes = [mb.kc_code(mb.odour(o)) for o in stimuli]
    if a.lr == "auto":                                                                # one pairing = Hige's 90 % in the punished compartment
        mb.calibrate_lr(codes[0], mb.compartment_mask("punish"), a.target_drop, "punish", a.strength)

    def responses():
        return torch.stack([mb.mbon_response(c) for c in codes])                       # [O, n_mbon]

    base = responses()
    norm = float((base.abs().sum(1)).mean())

    def score(R):
        return (R * val).sum(1) / norm

    def p_choose(sx, sy, beta):
        return float(torch.sigmoid(beta * (sx - sy)))

    def train(idx, us, pairings):
        mb.reset()
        for _ in range(pairings):
            mb.reinforce(codes[idx], us, a.strength)

    res = {"modality": a.modality, "mbon_valence_counts": counts, "n_DN": n_dn, "binary_code": a.binary_code,
           "pairings": a.pairings, "shuffle": a.shuffle,
           "valence_table_source": "connectome consensus transmitter per MBON type (unanimous for all 37 types in MaleCNS); "
                                   "GABA / acetylcholine -> approach, glutamate -> avoid (Aso et al. 2014)",
           # the map this run trained under (changes under --shuffle dan_mbon): where punishment and reward land
           "punish_compartment_types": sorted({mb.mbon_type[i] for i in np.flatnonzero(mb.compartment_mask("punish").numpy())}),
           "reward_compartment_types": sorted({mb.mbon_type[i] for i in np.flatnonzero(mb.compartment_mask("reward").numpy())}),
           "wiring_build": getattr(mb, "wiring_build", None),
           "rule": {"lr": round(mb.lr, 4), "lr_calibrated": a.lr == "auto", "recover_rate": a.recover_rate},
           "baseline_scores": [round(float(x), 4) for x in score(base)[:4]]}

    # --- readout 1: T-maze PI, reciprocal design, swept over the motor gain beta
    def tmaze(cs):
        """Reciprocal T-maze on the pair (cs[0], cs[1]), normalised over the odour set cs. Per beta: the index, the
        untrained circuit's innate A-vs-B bias (the reciprocal UNTRAINED index is identically 0 by construction, so
        the bias is the informative baseline), and P(choose the punished odour). Also the beta-free learned shift
        of each punished odour's approach score, the quantity that is comparable across senses and odour pairs."""
        mb.reset()                                                              # the baseline must be the untrained circuit
        R0 = torch.stack([mb.mbon_response(c) for c in cs]); nrm = float(R0.abs().sum(1).mean())

        def sc(R):
            return (R * val).sum(1) / nrm

        def after(idx):
            mb.reset()
            for _ in range(a.pairings):
                mb.reinforce(cs[idx], "punish", a.strength)
            return sc(torch.stack([mb.mbon_response(c) for c in cs]))
        s0, sA, sB = sc(R0), after(0), after(1)
        by_beta = {}
        for beta in a.betas:
            half1 = p_choose(sA[1], sA[0], beta) - p_choose(sA[0], sA[1], beta)
            half2 = p_choose(sB[0], sB[1], beta) - p_choose(sB[1], sB[0], beta)
            by_beta[beta] = {"PI_trained": round(0.5 * (half1 + half2), 4),
                             "innate_bias_A_vs_B": round(p_choose(s0[0], s0[1], beta) - 0.5, 4),
                             "P_choose_punished_A_vs_B": round(p_choose(sA[0], sA[1], beta), 4)}
        shift = {"A": round(float(s0[0] - sA[0]), 4), "B": round(float(s0[1] - sB[1]), 4)}
        return by_beta, shift, s0

    pi, shift0, _ = tmaze(codes)
    res["tmaze_PI_by_beta"] = pi
    res["learned_score_shift_example_pair"] = shift0
    res["ground_truth_PI"] = "wild type, one training cycle: 0.44 (automated) - 0.53 (manual); Tully & Quinn design"

    # --- the same T-maze over many odour pairs. The reciprocal index of ONE pair is capped by that pair's innate
    # A-vs-B bias (after B is punished a strongly preferred B can still win), so a single pair is not a circuit
    # property. Fresh draws per seed; the calibrated lr is odour-independent for a binary code (lr = target/delta).
    per = []
    for s in range(a.seed, a.seed + a.pi_seeds):
        r = np.random.default_rng(s)
        st = [r.choice(len(mb.glom_names), size=a.glom_per_odour, replace=False) for _ in range(a.odours)]
        per.append(tmaze([mb.kc_code(mb.odour(o)) for o in st]))

    def ms(xs):
        v = np.asarray(xs, float); return [round(float(v.mean()), 4), round(float(v.std()), 4)]
    over = {}
    for beta in a.betas:
        pis = [p[0][beta]["PI_trained"] for p in per]; bias = [abs(p[0][beta]["innate_bias_A_vs_B"]) for p in per]
        over[beta] = {"PI_mean_sd": ms(pis), "PI_min_max": [round(min(pis), 4), round(max(pis), 4)],
                      "abs_innate_bias_mean_sd": ms(bias),
                      "corr_PI_vs_abs_innate_bias": round(float(np.corrcoef(pis, bias)[0, 1]), 3) if len(per) > 2 and np.std(bias) > 0 and np.std(pis) > 0 else None}
    shifts = [p[1]["A"] for p in per] + [p[1]["B"] for p in per]
    res["tmaze_PI_over_seeds"] = {"n_seeds": a.pi_seeds, "by_beta": over, "learned_score_shift_mean_sd": ms(shifts),
                                  "note": "PI_mean_sd is the reciprocal index over independently drawn odour pairs; a negative "
                                          "corr_PI_vs_abs_innate_bias means pairs the untrained circuit already prefers cap the "
                                          "index. learned_score_shift is the drop in the punished odour's approach score (beta-free), "
                                          "the size of what the memory writes, comparable across senses."}

    # --- multiple memories change behaviour: A punished, C rewarded; choices among A, C, D(untouched)
    mb.reset()
    for _ in range(a.pairings):
        mb.reinforce(codes[0], "punish", a.strength); mb.reinforce(codes[2], "reward", a.strength)
    s = score(responses()); s0 = score(base)
    beta = a.betas[len(a.betas) // 2]
    res["multi_memory_choices"] = {"beta": beta,
        "P(C_rewarded over A_punished)": {"before": round(p_choose(s0[2], s0[0], beta), 4), "after": round(p_choose(s[2], s[0], beta), 4)},
        "P(D_untouched over A_punished)": {"before": round(p_choose(s0[3], s0[0], beta), 4), "after": round(p_choose(s[3], s[0], beta), 4)},
        "P(C_rewarded over D_untouched)": {"before": round(p_choose(s0[2], s0[3], beta), 4), "after": round(p_choose(s[2], s[3], beta), 4)}}

    # --- readout 2: does the change reach the descending neurons through the wiring?
    train(0, "punish", a.pairings); R = responses()

    def dn_drive(W, r):
        return W @ r
    d0_1, d1_1 = dn_drive(W1, base[0]), dn_drive(W1, R[0])
    d0_2, d1_2 = dn_drive(W2, base[0]), dn_drive(W2, R[0])

    def rel(before, after):
        b = float(before.abs().sum()); return round(float((after - before).abs().sum() / b), 4) if b > 0 else None
    wiring = {"direct_MBON_DN": {"DNs_driven": int((d0_1 != 0).sum()), "rel_change_trained": rel(d0_1, d1_1)},
              "two_hop_MBON_X_DN": {"DNs_driven": int((d0_2 != 0).sum()), "rel_change_trained": rel(d0_2, d1_2),
                                    "rel_change_untouched": rel(dn_drive(W2, base[3]), dn_drive(W2, R[3]))}}
    for name, rows in dn_rows.items():
        if rows:
            wiring[name] = {"direct_rel_change": rel(d0_1[rows], d1_1[rows]), "two_hop_rel_change": rel(d0_2[rows], d1_2[rows])}
    res["wiring_readout"] = wiring
    res["elapsed_s"] = round(time.time() - t0, 1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraph", type=Path, default=OUT_DIR / "graph" / "subgraph_v5.npz")
    ap.add_argument("--neurons", type=Path, default=OUT_DIR / "graph" / "neurons.parquet")
    ap.add_argument("--wiring", type=Path, default=OUT_DIR / "mb" / "mb_wiring.npz")
    ap.add_argument("--modality", default="olfactory", choices=["olfactory", "visual"])
    ap.add_argument("--sparsity", type=float, default=0.05); ap.add_argument("--odours", type=int, default=8)
    ap.add_argument("--glom-per-odour", type=int, default=6); ap.add_argument("--pairings", type=int, default=1)
    ap.add_argument("--lr", default="auto"); ap.add_argument("--target-drop", type=float, default=0.9)
    ap.add_argument("--strength", type=float, default=1.0); ap.add_argument("--recover-rate", type=float, default=1.0)
    ap.add_argument("--w-max", type=float, default=2.0); ap.add_argument("--binary-code", action="store_true")
    ap.add_argument("--shuffle", default=None, choices=[None, "pn_kc", "kc_mbon", "dan_mbon"])
    ap.add_argument("--punish", nargs="+", default=["PPL101"]); ap.add_argument("--reward", nargs="+", default=["PAM"])
    ap.add_argument("--betas", type=float, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--pi-seeds", type=int, default=10, help="independent odour-pair draws for the T-maze spread")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = json_safe(run(a)); text = json.dumps(res, indent=1, allow_nan=False)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
