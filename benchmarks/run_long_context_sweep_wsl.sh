#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/home/jiamin/projects/CFIE}"
PY="${PY:-$ROOT/.wsl-venv/bin/python}"
MODEL="${MODEL:-/home/jiamin/models/Qwen3.5-122B-A10B-GPTQ-Int4/snapshots/5b9f0050d3ec98b0c81a7716776533c5eacebb64}"
LOGDIR="${LOGDIR:-$ROOT/.bench_logs/20260521_long_context_sweep}"

# Format: max_model_len:kv_cache_bytes. Defaults are chosen from observed
# Qwen3.5 hybrid-KV estimates: 6 GiB reaches ~222K, 8 GiB reaches 256K.
CONTEXT_KV_CASES="${CONTEXT_KV_CASES:-184320:5368709120 222176:6442450944 262144:8589934592}"
MAX_BATCHED_TOKENS_LIST="${MAX_BATCHED_TOKENS_LIST:-4096 8192}"
PREFILL_TOKENS_LIST="${PREFILL_TOKENS_LIST:-20000 30000 40000 50000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TURNS="${TURNS:-2}"
WARM_PROMPT_MODE="${WARM_PROMPT_MODE:-rotated}"
GPU_UTIL="${GPU_UTIL:-0.88}"
GPU_SLOTS_PER_LAYER="${GPU_SLOTS_PER_LAYER:-16}"
PREFILL_BURST_SLOTS="${PREFILL_BURST_SLOTS:-256}"
CPU_COPY_BATCH="${CPU_COPY_BATCH:-8}"
CPU_STATIC_PINNED_GB="${CPU_STATIC_PINNED_GB:-44}"
MARLIN_INPUT_DTYPE="${MARLIN_INPUT_DTYPE:-fp8}"
SPEC_METHOD="${SPEC_METHOD:-none}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

mkdir -p "$LOGDIR"
cd "$ROOT" || exit 2

STATUS_TSV="$LOGDIR/sweep_status.tsv"
if [ ! -f "$STATUS_TSV" ]; then
  printf "run_name\tmax_model_len\tkv_bytes\tmax_num_batched_tokens\tprefill_tokens\tmax_new_tokens\tspec_method\texit_code\tstart_time\tend_time\n" > "$STATUS_TSV"
fi

run_case() {
  local max_model_len="$1"
  local kv_bytes="$2"
  local max_batched="$3"
  local run_name="ctx${max_model_len}_kv${kv_bytes}_mb${max_batched}_prefill_multi_new${MAX_NEW_TOKENS}_${SPEC_METHOD}"
  local started
  local ended
  local status

  if [ -f "$LOGDIR/$run_name.json" ] || [ -f "$LOGDIR/$run_name.exitcode" ]; then
    echo "[skip] $run_name already has result or exitcode in $LOGDIR"
    return 0
  fi

  started="$(date --iso-8601=seconds)"
  echo "[start] $run_name at $started"

  local cmd=(
    "$PY" benchmarks/run_long_prefill_decode.py
    --model "$MODEL"
    --target-input-tokens-list $PREFILL_TOKENS_LIST
    --max-new-tokens "$MAX_NEW_TOKENS"
    --turns "$TURNS"
    --warm-prompt-mode "$WARM_PROMPT_MODE"
    --isolate-prompt-lengths
    --max-model-len "$max_model_len"
    --max-num-batched-tokens "$max_batched"
    --gpu-memory-utilization "$GPU_UTIL"
    --kv-cache-memory-bytes "$kv_bytes"
    --gpu-slots-per-layer "$GPU_SLOTS_PER_LAYER"
    --prefill-burst-slots "$PREFILL_BURST_SLOTS"
    --prepare-cpu-copy-batch-size "$CPU_COPY_BATCH"
    --cpu-static-pinned-gb "$CPU_STATIC_PINNED_GB"
    --language-model-only
    --skip-mm-profiling
    --marlin-input-dtype "$MARLIN_INPUT_DTYPE"
    --temperature 0
    --spec-method "$SPEC_METHOD"
    --enforce-eager
    --force-exit-after-result
    --result-json "$LOGDIR/$run_name.json"
  )
  if [ "$SPEC_METHOD" = "mtp" ]; then
    cmd+=(--num-speculative-tokens "$NUM_SPEC_TOKENS")
  fi

  if [ "$ENFORCE_EAGER" = "0" ]; then
    local filtered=()
    local skip_next=0
    for arg in "${cmd[@]}"; do
      if [ "$skip_next" = "1" ]; then
        skip_next=0
        continue
      fi
      if [ "$arg" = "--enforce-eager" ]; then
        continue
      fi
      filtered+=("$arg")
    done
    cmd=("${filtered[@]}")
  fi

  CFIE_MONITOR_INTERVAL="${CFIE_MONITOR_INTERVAL:-1}" \
    "$ROOT/benchmarks/run_logged_wsl_command.sh" \
    "$run_name" \
    "$LOGDIR" \
    "${cmd[@]}"

  status="$(cat "$LOGDIR/$run_name.exitcode" 2>/dev/null || echo 999)"
  ended="$(date --iso-8601=seconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$run_name" "$max_model_len" "$kv_bytes" "$max_batched" \
    "$PREFILL_TOKENS_LIST" "$MAX_NEW_TOKENS" "$SPEC_METHOD" "$status" \
    "$started" "$ended" >> "$STATUS_TSV"
  echo "[done] $run_name status=$status at $ended"
}

for pair in $CONTEXT_KV_CASES; do
  max_model_len="${pair%%:*}"
  kv_bytes="${pair##*:}"
  for max_batched in $MAX_BATCHED_TOKENS_LIST; do
    run_case "$max_model_len" "$kv_bytes" "$max_batched"
  done
done

"$PY" benchmarks/summarize_long_context_sweep.py "$LOGDIR" \
  > "$LOGDIR/summary.md" || true
