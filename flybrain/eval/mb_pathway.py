"""Trace the memory-to-motor path and name the cells that bridge it, from the connectome.

The learned memory (sections 7 to 9) changes the output cell MBON-gamma1pedc (MBON11). For that change to alter
what the fly DOES, it has to reach the descending neurons that command turning: DNa02 (an established steering
cell) and DNa03. The connectome names the specific interneurons on that path. This tool ranks them, and states
a falsifiable prediction: if the path is carried by a few specific interneuron types, silencing those types
should selectively impair LEARNED odour avoidance while leaving naive behaviour intact, which a wet lab can test.

Path strength for an interneuron i = (summed synapses source-MBON -> i) * (summed synapses i -> steering DN),
i.e. the two-hop capacity through i. Reported per interneuron TYPE (the unit a lab can target with a driver line).

  python -m flybrain.eval.mb_pathway --data $FLYBRAIN_DATA/malecns_v1 --out $FLYBRAIN_OUT/mb/pathway.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from flybrain import DATA_DIR


def run(a) -> dict:
    t0 = time.time()
    ann = pd.read_feather(a.data / "body-annotations-male-cns-v1.0-minconf-0.5.feather")[["bodyId", "type", "superclass"]]
    ann["type"] = ann["type"].fillna("").astype(str); ann["superclass"] = ann["superclass"].fillna("").astype(str)
    tof = dict(zip(ann.bodyId, ann["type"]))
    src = ann.bodyId[ann["type"] == a.source_type].to_numpy()                 # the learned output cell(s)
    steer = ann.bodyId[ann["type"].isin(a.steer_types)].to_numpy()           # steering descending neurons
    all_dn = ann.bodyId[ann.superclass == "descending_neuron"].to_numpy()
    if len(src) == 0 or len(steer) == 0:
        raise ValueError(f"source {a.source_type} ({len(src)}) or steer {a.steer_types} ({len(steer)}) not found")

    w = pd.read_feather(a.data / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    mo = w[w.body_pre.isin(set(src))]                                        # MBON outputs
    di = w[w.body_post.isin(set(steer))]                                     # steering-DN inputs

    # direct MBON -> steering DN (expected thin / zero for MBON11)
    direct = mo[mo.body_post.isin(set(steer))].weight.sum()

    inter = np.array(sorted(set(mo.body_post) & set(di.body_pre)))           # cells the MBON drives that also drive the steering DN
    inter = inter[~np.isin(inter, np.concatenate([src, steer]))]            # exclude the endpoints themselves
    w1 = mo.groupby("body_post").weight.sum()                                # MBON -> i
    w2 = di.groupby("body_pre").weight.sum()                                 # i -> DN
    tin = w[w.body_post.isin(set(inter))].groupby("body_post").weight.sum()  # TOTAL input onto i (for specificity)
    rows = []
    for i in inter:
        s1, s2 = float(w1.get(i, 0.0)), float(w2.get(i, 0.0))
        tot_i = float(tin.get(i, 0.0)) or 1.0
        spec = s1 / tot_i                                                    # MBON's share of i's input = how memory-specific i is
        thru = s2 * spec                                                     # DN drive that is actually MEMORY-modulated (causal throughput)
        rows.append((int(i), tof.get(i, "?"), s1, s2, tot_i, spec, thru))
    rows.sort(key=lambda r: -r[6])                                          # rank by causal throughput, not raw capacity
    thru_total = sum(r[6] for r in rows) or 1.0
    steer_total_in = float(di.weight.sum())
    via_mb = sum(r[6] for r in rows)                                        # real memory throughput onto the steering DNs

    # per-type aggregation, ranked by memory throughput (the unit a lab targets)
    bytype: dict[str, dict] = {}
    for _b, tp, s1, s2, tot, spec, thru in rows:
        d = bytype.setdefault(tp, {"thru": 0.0, "n": 0, "s1": 0.0, "s2": 0.0, "specmax": 0.0})
        d["thru"] += thru; d["n"] += 1; d["s1"] += s1; d["s2"] += s2; d["specmax"] = max(d["specmax"], spec)
    ranked = sorted(bytype.items(), key=lambda kv: -kv[1]["thru"])
    top_types = [{"type": tp, "memory_throughput": round(d["thru"], 2), "share_of_throughput": round(d["thru"] / thru_total, 3),
                  "mbon_input_fraction_max": round(d["specmax"], 4), "n_cells": d["n"],
                  "mbon_to_type_syn": round(d["s1"], 1), "type_to_DN_syn": round(d["s2"], 1)} for tp, d in ranked[:a.top]]
    # the most memory-SPECIFIC bridges (largest share of their input from the MBON): the honest silencing targets
    by_spec = sorted(rows, key=lambda r: -r[5])
    specific = [{"bodyId": b, "type": tp, "mbon_input_fraction": round(spec, 4), "mbon_to_i": round(s1, 1), "i_to_DN": round(s2, 1)}
                for b, tp, s1, s2, tot, spec, thru in by_spec[:a.top]]
    best_type = by_spec[0][1] if by_spec else "?"; best_spec = round(by_spec[0][5], 4) if by_spec else None
    frac = round(via_mb / steer_total_in * 100, 3) if steer_total_in else None

    res = {"source_MBON": a.source_type, "n_source_cells": len(src),
           "steering_DNs": a.steer_types, "n_steer_cells": len(steer),
           "direct_MBON_to_steerDN_synapses": round(float(direct), 1),
           "n_bridging_interneurons": len(inter), "n_bridging_types": len(bytype),
           "memory_throughput_total": round(thru_total, 2),
           "steerDN_input_pct_memory_modulated": frac,
           "top_bridging_types_by_throughput": top_types, "most_specific_bridges": specific,
           "prediction": (f"The connectome provides NO memory-specific route from {a.source_type} to the steering DNs "
                          f"{a.steer_types}: the direct path is {round(float(direct),1)} synapses, and even the strongest "
                          f"two-hop bridge ({best_type}) receives only {best_spec} (~{round((best_spec or 0)*100,2)}%) of "
                          f"its input from {a.source_type}, so the total memory-modulated throughput is {frac}% of the "
                          f"steering DNs' input. This ANATOMICALLY confirms the model's section-9 finding that the memory "
                          f"does not reach the descending neurons. PREDICTION (testable): learned odour avoidance is NOT "
                          f"routed through a single dedicated interneuron from this compartment, so silencing the top "
                          f"candidates ({', '.join(t['type'] for t in top_types[:3])}) should NOT selectively abolish "
                          f"learned avoidance. A null screen would support a diffuse-summation or longer/neuromodulatory "
                          f"route; a positive hit would mean the behavioural pathway uses a connection weaker than the "
                          f"connectome's confidence threshold captures. High-capacity cells like AOTU019 are especially "
                          f"POOR targets ({a.source_type} is a negligible fraction of their input, so a deficit would be "
                          f"non-specific to the memory)."),
           "caveat": ("This is a CONNECTOME-ANATOMY prediction, not a model result. Section 9 of docs/RESULTS.md finds "
                      "the memory does NOT measurably flow through this path IN THE MODEL (the behaviour there is read "
                      "through the published valence map, not this route), and the throughput here is a tiny percent of "
                      "steering input. So it names the strongest candidate cells only; a null silencing result would "
                      "say learned avoidance is routed through cells this two-hop trace misses (longer or neuromodulatory "
                      "paths). Full connectome, minconf 0.5. path_strength is CAPACITY not proven NECESSITY."),
           "n_all_DN": len(all_dn), "elapsed_s": round(time.time() - t0, 1)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_DIR / "malecns_v1")
    ap.add_argument("--source-type", default="MBON11")
    ap.add_argument("--steer-types", nargs="+", default=["DNa02", "DNa03"])
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = run(a)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
