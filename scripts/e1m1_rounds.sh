#!/usr/bin/env bash
# E1M1 imitation rounds on node1 (run through scripts/detach.sh from the code dir):
#   round 0: fit the readout on navigator demos; rounds 1..N: DAgger (agent drives, navigator labels,
#   beta decays), re-fit on everything, evaluate on held-out seeds (normal + blindfolded).
#   bash scripts/e1m1_rounds.sh <first_round> <last_round> [gpu]
set -euo pipefail
FIRST="${1:?first round}"; LAST="${2:?last round}"; GPU="${3:-0}"
D=/home/unnati/flybrain-doom/outputs/e1m1
M="python -m flybrain.train.level_bc"
DEV="cuda:$GPU"
READOUT="${READOUT:-dn_vpn}"          # population read by the head (dn = descending only; see docs/RESULTS.md)
DEMO="${DEMO:-$D/demo_nav_v2.npz}"
RUN="${RUN:-v2}"                      # run-name prefix: $D/${RUN}_r$k, DAgger data $D/${RUN}_dagger_r$k.npz
FULL="${FLYBRAIN_GPU_SHARED:-1}"      # 0 = both cards are ours: more envs, bigger batches, evals in parallel on cuda:0 / cuda:1
if [ "$FULL" = "0" ]; then DW=44; EW=28; FB=256; else DW=32; EW=32; FB=64; fi
BETAS=(0 0.5 0.3 0.15 0.0 0.0 0.0)   # rounds past the end use beta 0

for k in $(seq "$FIRST" "$LAST"); do
  echo "=== round $k $(date +%T)"
  DATA="$DEMO"
  for j in $(seq 1 "$k"); do DATA="$DATA $D/${RUN}_dagger_r$j.npz"; done
  if [ "$k" -gt 0 ] && [ ! -f "$D/${RUN}_dagger_r$k.npz" ]; then
    $M dagger --ckpt "$D/${RUN}_r$((k-1))/student.pt" --episodes 48 --workers $DW --beta "${BETAS[$k]:-0.0}" \
      --seed-base $((50000 + 1000 * k)) --sample --device "$DEV" --out "$D/${RUN}_dagger_r$k.npz"
  fi
  $M feats --data $DATA --readout "$READOUT" --batch $FB --device "$DEV"
  $M fit --data $DATA --readout "$READOUT" --run "${RUN}_r$k" --device "$DEV"
  # validation seeds 10000+ (used to follow rounds); the final numbers come from an untouched block (scripts/e1m1_final.sh)
  # the agent samples its actions (greedy argmax deadlocks against walls); blindfolds use the same sampling
  if [ "$FULL" = "0" ]; then
    $M eval --ckpt "$D/${RUN}_r$k/student.pt" --episodes 32 --workers 18 --device cuda:1 --seed-base 10000 --sample --out "$D/${RUN}_r$k/val_sample.json" & P1=$!
    $M eval --ckpt "$D/${RUN}_r$k/student.pt" --episodes 32 --workers 18 --device cuda:0 --seed-base 10000 --sample --blind scramble --out "$D/${RUN}_r$k/val_scramble.json" & P2=$!
    $M eval --ckpt "$D/${RUN}_r$k/student.pt" --episodes 32 --workers 18 --device cuda:0 --seed-base 10000 --out "$D/${RUN}_r$k/val_greedy.json" & P3=$!
    wait $P1; wait $P2; wait $P3
  else
    $M eval --ckpt "$D/${RUN}_r$k/student.pt" --episodes 32 --workers $EW --device "$DEV" --seed-base 10000 --sample --out "$D/${RUN}_r$k/val_sample.json"
    $M eval --ckpt "$D/${RUN}_r$k/student.pt" --episodes 32 --workers $EW --device "$DEV" --seed-base 10000 --sample --blind scramble --out "$D/${RUN}_r$k/val_scramble.json"
  fi
  python -c "import json; f=lambda n: json.load(open('$D/${RUN}_r$k/'+n)); r, s = f('val_sample.json'), f('val_scramble.json'); print('ROUND $k sample: exit', r['exit_rate'], 'progress', round(r['progress']['mean'], 3), 'median', round(r['progress']['median'], 3), '| scramble exit', s['exit_rate'], 'progress', round(s['progress']['mean'], 3), '| actions', r['action_hist'])"
done
echo "=== done $(date +%T)"
