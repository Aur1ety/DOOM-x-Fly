#!/usr/bin/env bash
# Run a command on the server inside the project venv.
#   scripts/remote.sh login "pytest tests/test_import.py -q"
#   scripts/remote.sh gpu   "python -m flybrain.model.bench"           # GPU node
#   scripts/remote.sh gpu   "pytest tests/test_spmm.py -q" dev/kernel   # in a scratch dir
# Host names come from scripts/hosts.sh (FLYBRAIN_LOGIN_HOST / FLYBRAIN_GPU_HOST).
set -euo pipefail
NODE="${1:?login|gpu}"; CMD="${2:?command}"; DIR="${3:-code}"
. "$(dirname "${BASH_SOURCE[0]}")/hosts.sh"
NODE="$(resolve_node "$NODE")"
REMOTE="cd ~/flybrain-doom/$DIR && source ~/flybrain-doom/.venv/bin/activate \
  && export FLYBRAIN_DATA=~/flybrain-doom/data FLYBRAIN_OUT=~/flybrain-doom/outputs \
     PYTHONPATH=~/flybrain-doom/$DIR OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && $CMD"
if [ "$NODE" = "$LOGIN_HOST" ]; then
  ssh -o BatchMode=yes "$LOGIN_HOST" "$REMOTE"
else
  ssh -o BatchMode=yes "$LOGIN_HOST" "ssh -o BatchMode=yes $NODE $(printf '%q' "$REMOTE")"
fi 2>&1 | grep -v 'post-quantum\|store now\|openssh.com/pq' || true
