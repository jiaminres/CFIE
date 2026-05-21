#!/usr/bin/env bash
set -u

if [ "$#" -lt 3 ]; then
  echo "usage: $0 RUN_NAME LOGDIR COMMAND [ARGS...]" >&2
  exit 2
fi

RUN_NAME="$1"
shift
LOGDIR="$1"
shift

mkdir -p "$LOGDIR"

MONITOR_CSV="$LOGDIR/${RUN_NAME}.nvidia_smi.csv"
STDOUT_LOG="$LOGDIR/${RUN_NAME}.stdout.log"
STDERR_LOG="$LOGDIR/${RUN_NAME}.stderr.log"
EXIT_CODE="$LOGDIR/${RUN_NAME}.exitcode"
COMMAND_TXT="$LOGDIR/${RUN_NAME}.command.txt"

printf '%q ' "$@" > "$COMMAND_TXT"
printf '\n' >> "$COMMAND_TXT"

(
  echo "timestamp,memory.used.MiB,memory.total.MiB,utilization.gpu.%,utilization.memory.%"
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory \
      --format=csv,noheader,nounits 2>/dev/null || true
    sleep "${CFIE_MONITOR_INTERVAL:-1}"
  done
) > "$MONITOR_CSV" &
MONITOR_PID=$!

cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

set +e
"$@" > "$STDOUT_LOG" 2> "$STDERR_LOG"
STATUS=$?
echo "$STATUS" > "$EXIT_CODE"
exit "$STATUS"
