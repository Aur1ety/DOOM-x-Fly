#!/usr/bin/env bash
# Run a command on the server inside the project venv.
#   scripts/remote.sh master "pytest tests/test_import.py -q"
#   scripts/remote.sh node1  "python -m flybrain.model.bench"           # GPU node
#   scripts/remote.sh node1  "pytest tests/test_spmm.py -q" dev/kernel   # in a scratch dir
set -euo pipefail
NODE="${1:?master|node1}"; CMD="${2:?command}"; DIR="${3:-code}"
REMOTE="cd ~/flybrain-doom/$DIR && source ~/flybrain-doom/.venv/bin/activate \
  && export FLYBRAIN_DATA=~/flybrain-doom/data FLYBRAIN_OUT=~/flybrain-doom/outputs \
     PYTHONPATH=~/flybrain-doom/$DIR OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && $CMD"
if [ "$NODE" = "master" ]; then
  ssh -o BatchMode=yes master "$REMOTE"
else
  ssh -o BatchMode=yes master "ssh -o BatchMode=yes $NODE $(printf '%q' "$REMOTE")"
fi 2>&1 | grep -v 'post-quantum\|store now\|openssh.com/pq' || true
