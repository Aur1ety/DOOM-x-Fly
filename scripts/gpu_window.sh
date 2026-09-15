#!/usr/bin/env bash
# GPU window on node1: stop the root-owned vLLM service (and its watchdog timer), run ONE
# command with the whole card, make sure our own GPU processes are gone, restart the service,
# wait until /metrics is healthy, re-arm the timer. The restore runs from an EXIT trap, so it
# happens even if the command fails or the window is interrupted (Ctrl-C, SIGTERM).
#
#   scripts/gpu_window.sh [--dry-run] [--gpu N] -- <command...>
#   GPU_WINDOW_DRY_RUN=1 scripts/gpu_window.sh -- python -m flybrain.model.bench
#
# Exit codes: the command's status; 2 = usage / window not obtained (command NOT run);
# 1 = the window was obtained but the restore failed (overrides a 0 from the command).
# Refuses cleanly if `sudo -n` is not whitelisted (flybrain.ops.qwen exits 3, nothing is run).
# Dry-run never signals any process and never runs sudo. Our GPU processes that already
# existed before the window opened (a teammate's guarded job under this same account) are
# spared with a warning unless GPU_WINDOW_KILL_ALL=1.
set -uo pipefail

DRY_RUN="${GPU_WINDOW_DRY_RUN:-0}"
GPU="${GPU_WINDOW_GPU:-0}"
KILL_ALL="${GPU_WINDOW_KILL_ALL:-0}"
PY="${PYTHON:-python}"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --gpu) GPU="${2:?--gpu needs a value}"; shift 2 ;;
    --) shift; break ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "gpu_window: unknown option '$1' (put the command after --)" >&2; exit 2 ;;
  esac
done
if [ $# -eq 0 ]; then
  echo "gpu_window: no command given. Usage: scripts/gpu_window.sh [--dry-run] [--gpu N] -- <command...>" >&2
  exit 2
fi

log() { printf '[gpu_window %s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
QWEN=("$PY" -m flybrain.ops.qwen)
if [ "$DRY_RUN" = 1 ]; then MODE=(--dry-run); else MODE=(--yes); fi
LEDGER=""
if [ -n "${FLYBRAIN_OUT:-}" ] && mkdir -p "$FLYBRAIN_OUT" 2>/dev/null; then LEDGER="$FLYBRAIN_OUT/gpu_window.log"; fi
ledger() { [ -n "$LEDGER" ] && printf '%s\t%s\t%s\n' "$(date '+%Y-%m-%dT%H:%M:%S')" "$1" "$2" >> "$LEDGER"; return 0; }

if [ "$DRY_RUN" != 1 ] && ! command -v nvidia-smi >/dev/null 2>&1; then
  log "nvidia-smi not found: this must run on node1"; exit 2
fi

CMD_PID=""
STATUS=0
RESTORED=0
RESTORE_FAILED=0
PRE_PIDS=""   # our GPU PIDs that were there before the window opened

own_pids() { "${QWEN[@]}" own-pids 2>/dev/null | tr '\n' ' '; }
is_pre() { case " $PRE_PIDS " in *" $1 "*) return 0 ;; esac; return 1; }

clear_own_gpu_procs() {
  # vLLM needs its memory free at startup (vllm#20305): our processes must be gone first.
  local tries=0 p pids killable spared sig
  while :; do
    pids="$(own_pids)"; killable=""; spared=""
    for p in $pids; do
      if [ "$KILL_ALL" != 1 ] && is_pre "$p"; then spared="$spared $p"; else killable="$killable $p"; fi
    done
    if [ -z "${killable// /}" ]; then
      if [ -n "${spared// /}" ]; then
        log "WARNING: leaving our pre-existing GPU processes alone:$spared (GPU_WINDOW_KILL_ALL=1 kills them); vLLM may fail to restart if they hold too much"
        ledger spared "$spared"
      fi
      return 0
    fi
    tries=$((tries + 1))
    if [ "$tries" -gt 20 ]; then log "WARNING: could not clear our GPU processes:$killable"; return 1; fi
    sig=TERM; [ "$tries" -gt 10 ] && sig=KILL
    if [ "$DRY_RUN" = 1 ]; then log "dry-run: would send $sig to our GPU processes:$killable"; return 0; fi
    log "our GPU processes still present:$killable (sending $sig)"
    # shellcheck disable=SC2086
    kill -s "$sig" $killable 2>/dev/null
    sleep 1
  done
}

restore() {
  # Runs once, from the EXIT trap. Never returns early: the service restart must happen.
  [ "$RESTORED" = 1 ] && return 0
  RESTORED=1
  if [ -n "$CMD_PID" ] && kill -0 "$CMD_PID" 2>/dev/null; then
    log "command still running (pid $CMD_PID): sending TERM"
    kill -TERM -- "-$CMD_PID" 2>/dev/null || kill -TERM "$CMD_PID" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$CMD_PID" 2>/dev/null || break; sleep 1; done
    if kill -0 "$CMD_PID" 2>/dev/null; then
      log "command did not exit: sending KILL"
      kill -KILL -- "-$CMD_PID" 2>/dev/null || kill -KILL "$CMD_PID" 2>/dev/null
    fi
  fi
  clear_own_gpu_procs || RESTORE_FAILED=1
  log "restoring service: ${QWEN[*]} start ${MODE[*]}"
  "${QWEN[@]}" start "${MODE[@]}"
  local rc=$?
  if [ "$rc" -eq 0 ] && [ "$DRY_RUN" = 1 ]; then
    log "dry-run: restore step done (nothing executed)"; ledger restored "dry-run"
  elif [ "$rc" -eq 0 ]; then
    log "service restored (healthy) and timer re-armed"; ledger restored "ok"
  else
    log "WARNING: service restart reported failure (rc=$rc): check 'systemctl status vllm-qwen.service' and tell the admin"
    ledger restored "FAILED"; RESTORE_FAILED=1
  fi
  if [ "$RESTORE_FAILED" = 1 ] && [ "$STATUS" -eq 0 ]; then exit 1; fi   # from an EXIT trap this sets the final status
  return 0
}

log "mode=$([ "$DRY_RUN" = 1 ] && echo dry-run || echo LIVE) gpu=$GPU command: $*"
"${QWEN[@]}" status || true
PRE_PIDS="$(own_pids)"
[ -n "${PRE_PIDS// /}" ] && log "our GPU processes already present before the window: $PRE_PIDS"

trap restore EXIT
trap 'log "interrupted"; STATUS=130; exit 130' INT TERM HUP

log "closing window: ${QWEN[*]} stop ${MODE[*]}"
"${QWEN[@]}" stop "${MODE[@]}" --gpu "$GPU"
rc=$?
if [ "$rc" -eq 3 ]; then
  RESTORED=1   # nothing was touched; nothing to restore
  log "window NOT obtained (sudo -n refused); command not run"; ledger refused "$*"
  exit 2
elif [ "$rc" -ne 0 ]; then
  log "window NOT obtained (stop failed rc=$rc); command not run, restoring"; ledger stop_failed "$*"
  STATUS=2
  exit 2
fi
ledger window_open "$*"

# setsid puts the command in its own process group so restore() can kill all of it.
if command -v setsid >/dev/null 2>&1; then setsid -w "$@" & else "$@" & fi
CMD_PID=$!
wait "$CMD_PID"
STATUS=$?
CMD_PID=""
log "command exited with status $STATUS"; ledger command_exit "$STATUS"
exit "$STATUS"
