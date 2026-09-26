#!/usr/bin/env bash
# Launch a long-running command on the server DETACHED (nohup, log file), so a dropped SSH link
# cannot kill it. The command runs inside the project venv exactly like remote.sh.
#   scripts/detach.sh login gate0_tc "python -m flybrain.eval.probe run ... --out ..."
#   scripts/detach.sh gpu   bench    "CUDA_VISIBLE_DEVICES=0 python -m flybrain.model.bench_real ..."
# Logs: <login node>:~/flybrain-doom/outputs/logs/<name>.log ; done marker <name>.exit (exit code)
# Host names come from scripts/hosts.sh (FLYBRAIN_LOGIN_HOST / FLYBRAIN_GPU_HOST).
# Retries flaky SSH hops; exits non-zero (and prints DETACH FAILED) unless the job script was written and a PID came back.
set -uo pipefail
NODE="${1:?login|gpu}"; NAME="${2:?job name}"; CMD="${3:?command}"; DIR="${4:-code}"
. "$(dirname "${BASH_SOURCE[0]}")/hosts.sh"
NODE="$(resolve_node "$NODE")"
LOGDIR='~/flybrain-doom/outputs/logs'
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30"
# The command runs inside a brace group: a CMD that backgrounds several jobs with '&' keeps the venv and
# the exports for every job (without the group, '... && job1 & job2 &' would run job2 in a bare shell).
BODY="cd ~/flybrain-doom/$DIR && source ~/flybrain-doom/.venv/bin/activate \
  && export FLYBRAIN_DATA=~/flybrain-doom/data FLYBRAIN_OUT=~/flybrain-doom/outputs \
     PYTHONPATH=~/flybrain-doom/$DIR OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && { $CMD
}
echo \"exit \$?\" > $LOGDIR/$NAME.exit"
WANT=$(printf '%s\n' "$BODY" | md5sum | cut -d' ' -f1)
quiet() { grep -v 'post-quantum\|store now\|openssh.com/pq\|may need to be upgraded'; }

# 1. write the job script over NFS (visible from both nodes) and verify its checksum
ok=0
for attempt in 1 2 3 4 5; do
  GOT=$($SSH "$LOGIN_HOST" "mkdir -p $LOGDIR && cat > $LOGDIR/$NAME.sh && rm -f $LOGDIR/$NAME.exit && md5sum $LOGDIR/$NAME.sh" <<< "$BODY" 2>&1 | quiet | cut -d' ' -f1 | tail -1)
  if [ "$GOT" = "$WANT" ]; then ok=1; break; fi
  echo "write attempt $attempt failed ($GOT); retrying" >&2; sleep 10
done
[ "$ok" = 1 ] || { echo "DETACH FAILED: could not write $LOGDIR/$NAME.sh" >&2; exit 1; }

# 2. start it under nohup; require a numeric PID
START="nohup bash $LOGDIR/$NAME.sh > $LOGDIR/$NAME.log 2>&1 < /dev/null & echo \$!"
for attempt in 1 2 3 4 5; do
  if [ "$NODE" = "$LOGIN_HOST" ]; then
    PID=$($SSH "$LOGIN_HOST" "$START" 2>&1 | quiet | tail -1)
  else
    PID=$($SSH "$LOGIN_HOST" "$SSH $NODE $(printf '%q' "$START")" 2>&1 | quiet | tail -1)
  fi
  if [[ "$PID" =~ ^[0-9]+$ ]]; then
    echo "detached on $NODE, pid $PID; log: $LOGIN_HOST:$LOGDIR/$NAME.log ; exit marker: $LOGDIR/$NAME.exit"
    exit 0
  fi
  echo "start attempt $attempt failed ($PID); retrying" >&2; sleep 10
done
echo "DETACH FAILED: could not start $NAME on $NODE" >&2
exit 1
