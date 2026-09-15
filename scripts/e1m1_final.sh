#!/usr/bin/env bash
# Final E1M1 numbers for one frozen checkpoint + sampling temperature, on a seed block no earlier run touched
# (20000+), 100 episodes each: the agent, its blindfolds (black / mean / scramble), wide starts + sticky actions,
# greedy actions, and the floors (random, forward, open-loop replay of navigator runs) and ceiling (navigator)
# under the same settings. Two parallel streams (cuda:0 / cuda:1) when both cards are ours.
#   FLYBRAIN_GPU_SHARED=0 bash scripts/e1m1_final.sh <ckpt> <tag> <temperature>
set -euo pipefail
CKPT="${1:?checkpoint}"; TAG="${2:?tag}"; TEMP="${3:?temperature}"
D=/home/unnati/flybrain-doom/outputs/e1m1
OUT="$D/final_$TAG"; mkdir -p "$OUT"
M="python -m flybrain.train.level_bc eval --episodes 100 --workers 26 --seed-base 20000"
A="--ckpt $CKPT --sample --temperature $TEMP"      # the agent as evaluated everywhere; blindfolds use identical sampling
WIDE="--wiggle 20 --wiggle-moves --sticky 0.1"
echo "{\"ckpt\": \"$CKPT\", \"temperature\": $TEMP, \"seed_base\": 20000, \"episodes\": 100}" > "$OUT/config.json"

stream0() {
  $M --device cuda:0 $A --out "$OUT/agent.json"
  $M --device cuda:0 $A --blind scramble --out "$OUT/agent_scramble.json"
  $M --device cuda:0 $A --blind black --out "$OUT/agent_black.json"
  $M --device cuda:0 $A --blind mean --out "$OUT/agent_mean.json"
  $M --device cuda:0 --ckpt "$CKPT" --out "$OUT/agent_greedy.json"
}
stream1() {
  $M --device cuda:1 $A $WIDE --out "$OUT/agent_wide_sticky.json"
  for p in navigator replay forward random; do $M --device cuda:1 --policy $p --out "$OUT/floor_$p.json"; done
  $M --device cuda:1 --policy navigator $WIDE --out "$OUT/floor_navigator_wide_sticky.json"
  $M --device cuda:1 --policy replay $WIDE --out "$OUT/floor_replay_wide_sticky.json"
}
stream0 > "$OUT/stream0.log" 2>&1 & P0=$!
stream1 > "$OUT/stream1.log" 2>&1 & P1=$!
R0=0; R1=0
wait $P0 || R0=$?
wait $P1 || R1=$?
python - "$OUT" <<'PY'
import json, sys, glob, os
out = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(out, "*.json"))):
    if f.endswith("config.json") or f.endswith("summary.json"):
        continue
    r = json.load(open(f)); p = r["progress"]
    rows.append({"run": os.path.basename(f)[:-5], "exit_rate": r["exit_rate"], "exit_ci95": r["exit_ci95"], "progress_mean": p["mean"],
                 "progress_ci95": p["mean_ci95"], "progress_median": p["median"], "dead": r["dead"], "timeout": r["timeout"], "kills": r["kills_mean"]})
    print(f"{rows[-1]['run']:32s} exit {r['exit_rate']:.2f} {r['exit_ci95']}  progress {p['mean']:.3f} {[round(x, 3) for x in p['mean_ci95']]}  median {p['median']:.3f}  dead {r['dead']}  timeout {r['timeout']}")
json.dump(rows, open(os.path.join(out, "summary.json"), "w"), indent=1)
PY
echo "=== final $TAG done $(date +%T) streams rc $R0 $R1"
[ "$R0" = 0 ] && [ "$R1" = 0 ]
