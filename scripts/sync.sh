#!/usr/bin/env bash
# Push this folder's code to the server. Laptop is the source of truth; server only runs it.
#   scripts/sync.sh                      -> <login node>:~/flybrain-doom/code
#   SYNC_TARGET=dev/kernel scripts/sync.sh -> <login node>:~/flybrain-doom/dev/kernel
# The login node's ssh alias comes from scripts/hosts.sh (FLYBRAIN_LOGIN_HOST).
# Retries on SSH failures and exits non-zero if the push never succeeded (it used to report success regardless).
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${SYNC_TARGET:-code}"
. "$(dirname "${BASH_SOURCE[0]}")/hosts.sh"
for attempt in 1 2 3 4 5; do
  tar czf - -C "$HERE" \
    --exclude ./.git --exclude ./data --exclude ./outputs --exclude '__pycache__' \
    --exclude '*.pyc' --exclude '.pytest_cache' --exclude '*.npz' --exclude '*.parquet' --exclude ./videos --exclude ./scratch . \
    | ssh -o BatchMode=yes -o ConnectTimeout=30 "$LOGIN_HOST" "mkdir -p ~/flybrain-doom/$TARGET && tar xzf - -C ~/flybrain-doom/$TARGET" \
    2> >(grep -v 'post-quantum\|store now\|openssh.com/pq\|may need to be upgraded' >&2)
  if [ "${PIPESTATUS[1]}" -eq 0 ]; then
    echo "synced -> $LOGIN_HOST:~/flybrain-doom/$TARGET"
    exit 0
  fi
  echo "sync attempt $attempt failed; retrying" >&2
  sleep 10
done
echo "SYNC FAILED -> $LOGIN_HOST:~/flybrain-doom/$TARGET" >&2
exit 1
