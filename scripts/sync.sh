#!/usr/bin/env bash
# Push this folder's code to the server. Laptop is the source of truth; server only runs it.
#   scripts/sync.sh                      -> master:~/flybrain-doom/code
#   SYNC_TARGET=dev/kernel scripts/sync.sh -> master:~/flybrain-doom/dev/kernel
# Retries on SSH failures and exits non-zero if the push never succeeded (it used to report success regardless).
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${SYNC_TARGET:-code}"
for attempt in 1 2 3 4 5; do
  tar czf - -C "$HERE" \
    --exclude ./.git --exclude ./data --exclude ./outputs --exclude '__pycache__' \
    --exclude '*.pyc' --exclude '.pytest_cache' --exclude '*.npz' --exclude '*.parquet' --exclude ./videos --exclude ./scratch . \
    | ssh -o BatchMode=yes -o ConnectTimeout=30 master "mkdir -p ~/flybrain-doom/$TARGET && tar xzf - -C ~/flybrain-doom/$TARGET" \
    2> >(grep -v 'post-quantum\|store now\|openssh.com/pq\|may need to be upgraded' >&2)
  if [ "${PIPESTATUS[1]}" -eq 0 ]; then
    echo "synced -> master:~/flybrain-doom/$TARGET"
    exit 0
  fi
  echo "sync attempt $attempt failed; retrying" >&2
  sleep 10
done
echo "SYNC FAILED -> master:~/flybrain-doom/$TARGET" >&2
exit 1
