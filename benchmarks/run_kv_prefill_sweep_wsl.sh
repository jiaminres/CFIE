#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/home/jiamin/projects/CFIE}"
PY="${PY:-$ROOT/.wsl-venv/bin/python}"
MODEL="${MODEL:-/home/jiamin/models/Qwen3.5-122B-A10B-GPTQ-Int4/snapshots/5b9f0050d3ec98b0c81a7716776533c5eacebb64}"
LOGDIR="${LOGDIR:-$ROOT/.bench_logs/20260521_kv_prefill_nograph}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TURNS="${TURNS:-2}"
WARM_PROMPT_MODE="${WARM_PROMPT_MODE:-rotated}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
GPU_UTIL="${GPU_UTIL:-0.88}"
GPU_SLOTS_PER_LAYER="${GPU_SLOTS_PER_LAYER:-16}"
PREFILL_BURST_SLOTS="${PREFILL_BURST_SLOTS:-256}"
CPU_COPY_BATCH="${CPU_COPY_BATCH:-8}"
CPU_STATIC_PINNED_GB="${CPU_STATIC_PINNED_GB:-44}"
MARLIN_INPUT_DTYPE="${MARLIN_INPUT_DTYPE:-fp8}"
SPEC_METHOD="${SPEC_METHOD:-mtp}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-1}"

mkdir -p "$LOGDIR"
cd "$ROOT" || exit 2

STATUS_TSV="$LOGDIR/sweep_status.tsv"
if [ ! -f "$STATUS_TSV" ]; then
  printf "run_name\tkv_bytes\tinput_tokens\tmax_model_len\texit_code\tstart_time\tend_time\n" > "$STATUS_TSV"
fi

run_case() {
  local run_name="$1"
  local kv_bytes="$2"
  local input_tokens="$3"
  local max_model_len="$4"
  local started
  local ended
  local status

  if [ -f "$LOGDIR/$run_name.json" ]; then
    echo "[skip] $run_name already has $LOGDIR/$run_name.json"
    return 0
  fi

  started="$(date --iso-8601=seconds)"
  echo "[start] $run_name kv=$kv_bytes input=$input_tokens max_model_len=$max_model_len at $started"

  CFIE_MONITOR_INTERVAL="${CFIE_MONITOR_INTERVAL:-1}" \
    "$ROOT/benchmarks/run_logged_wsl_command.sh" \
    "$run_name" \
    "$LOGDIR" \
    "$PY" \
    benchmarks/run_long_prefill_decode.py \
    --model "$MODEL" \
    --target-input-tokens "$input_tokens" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --turns "$TURNS" \
    --warm-prompt-mode "$WARM_PROMPT_MODE" \
    --max-model-len "$max_model_len" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --kv-cache-memory-bytes "$kv_bytes" \
    --gpu-slots-per-layer "$GPU_SLOTS_PER_LAYER" \
    --prefill-burst-slots "$PREFILL_BURST_SLOTS" \
    --prepare-cpu-copy-batch-size "$CPU_COPY_BATCH" \
    --cpu-static-pinned-gb "$CPU_STATIC_PINNED_GB" \
    --language-model-only \
    --skip-mm-profiling \
    --marlin-input-dtype "$MARLIN_INPUT_DTYPE" \
    --temperature 0 \
    --spec-method "$SPEC_METHOD" \
    --num-speculative-tokens "$NUM_SPEC_TOKENS" \
    --enforce-eager \
    --force-exit-after-result \
    --result-json "$LOGDIR/$run_name.json"

  status="$(cat "$LOGDIR/$run_name.exitcode" 2>/dev/null || echo 999)"
  ended="$(date --iso-8601=seconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$run_name" "$kv_bytes" "$input_tokens" "$max_model_len" "$status" "$started" "$ended" \
    >> "$STATUS_TSV"
  echo "[done] $run_name status=$status at $ended"
}

run_case kv1g_prefill5k_decode512 1073741824 5000 8192
run_case kv2g_prefill10k_decode512 2147483648 10000 16384
run_case kv4g_prefill20k_decode512 4294967296 20000 32768
run_case kv4g_prefill30k_decode512 4294967296 30000 32768
run_case kv6g_prefill40k_decode512 6442450944 40000 65536
run_case kv8g_prefill50k_decode512 8589934592 50000 65536
