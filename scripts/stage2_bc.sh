#!/usr/bin/env bash
# Stage 2 on the server (run inside the venv, e.g. via scripts/detach.sh node1 stage2 "bash scripts/stage2_bc.sh <teacher.pt> <tag>"):
#   teacher recording -> behaviour cloning -> closed-loop eval -> blindfold tests -> DAgger round -> retrain -> eval -> video
# Every step is idempotent-ish (skips recordings that already exist). Results land in $FLYBRAIN_OUT/bc/<tag>*.
set -euo pipefail
TEACHER="${1:?teacher checkpoint}"; TAG="${2:-tc}"; SCEN="${3:-take_cover}"; GPU="${CUDA_VISIBLE_DEVICES:-0}"
O=~/flybrain-doom/outputs; SUB=$O/graph/subgraph_v5.npz; BC=$O/bc
mkdir -p $BC $O/videos
export CUDA_VISIBLE_DEVICES=$GPU
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "1/7 teacher recording"
[ -f $BC/teacher_$TAG.npz ] || python -m flybrain.train.bc record --teacher "$TEACHER" --scenario $SCEN --decisions 60000 --envs 32 --eps 0.1 --out $BC/teacher_$TAG.npz
log "2/7 behaviour cloning (per_edge, DN readout)"
python -m flybrain.train.bc train --data $BC/teacher_$TAG.npz --subgraph $SUB --run bc_${TAG}_edge --param per_edge --readout dn --epochs 6 --batch 64 --T 32 --device cuda:0
log "3/7 closed-loop eval (greedy, 50 held-out episodes) + blindfold"
python -m flybrain.train.bc play --ckpt $BC/bc_${TAG}_edge/student.pt --subgraph $SUB --scenario $SCEN --episodes 50 --device cuda:0 | tee $BC/bc_${TAG}_edge/play.json
for b in black mean scramble; do python -m flybrain.train.bc play --ckpt $BC/bc_${TAG}_edge/student.pt --subgraph $SUB --scenario $SCEN --episodes 50 --device cuda:0 --blind $b | tee $BC/bc_${TAG}_edge/play_blind_$b.json; done
log "4/7 DAgger round: student plays, teacher labels"
python -m flybrain.train.bc dagger --student $BC/bc_${TAG}_edge/student.pt --subgraph $SUB --teacher "$TEACHER" --scenario $SCEN --decisions 40000 --envs 32 --beta 0.0 --device cuda:0 --out $BC/dagger1_$TAG.npz
log "5/7 retrain on aggregated data"
python -m flybrain.train.bc train --data $BC/teacher_$TAG.npz $BC/dagger1_$TAG.npz --subgraph $SUB --run bc_${TAG}_edge_d1 --param per_edge --readout dn --epochs 4 --batch 64 --T 32 --device cuda:0 --resume $BC/bc_${TAG}_edge/student.pt
log "6/7 eval + blindfold after DAgger"
python -m flybrain.train.bc play --ckpt $BC/bc_${TAG}_edge_d1/student.pt --subgraph $SUB --scenario $SCEN --episodes 50 --device cuda:0 | tee $BC/bc_${TAG}_edge_d1/play.json
for b in black mean scramble; do python -m flybrain.train.bc play --ckpt $BC/bc_${TAG}_edge_d1/student.pt --subgraph $SUB --scenario $SCEN --episodes 50 --device cuda:0 --blind $b | tee $BC/bc_${TAG}_edge_d1/play_blind_$b.json; done
log "7/7 video (3 episodes)"
python -m flybrain.train.bc play --ckpt $BC/bc_${TAG}_edge_d1/student.pt --subgraph $SUB --scenario $SCEN --episodes 3 --device cuda:0 --video $O/videos/connectome_${TAG}_d1.mp4
log "stage 2 done"
