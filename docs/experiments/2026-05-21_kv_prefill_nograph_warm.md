# 2026-05-21 KV Prefill No-Graph Warm TTFT

## Scope

- 122B WSL benchmark path, no CUDA graph.
- Standard decode/prefill config under test:
  - `gpu_slots_per_layer=16`
  - `prefill_burst_slots=256`
  - `prepare_cpu_copy_batch_size=8`
  - `cpu_static_pinned_gb=44`
  - `marlin_input_dtype=fp8`
  - `spec_method=mtp`
  - `num_speculative_tokens=1`
  - `enforce_eager`
- Goal for this entry:
  - fix misleading hybrid KV capacity logging;
  - record both cold first inference TTFT and warm next-conversation TTFT.

## Code Changes

- `benchmarks/run_long_prefill_decode.py`
  - Added `--turns` to run multiple sequential requests in the same engine.
  - Added `--warm-prompt-mode rotated|same`.
    - `rotated` keeps the prompt length but avoids exact full-prefix-cache reuse.
    - `same` measures repeated-prompt/prefix-cache behavior.
  - Result JSON now includes:
    - `cold_prefill_to_first_token_seconds`
    - `warm_prefill_to_first_token_seconds`
    - `warm_prefill_to_first_token_seconds_avg`
    - per-turn `request_results`
  - Added `--force-exit-after-result` for WSL cleanup hangs after engine shutdown.
  - Default benchmark text is ASCII and result JSON is ASCII-escaped for Windows tooling compatibility.
- `cfie/v1/core/kv_cache_utils.py`
  - Kept the original hybrid aggregate block capacity log, but labeled it as a lower-bound scheduler metric.
  - Added a separate estimated single-request max-context log.
  - The estimate now sums group-specific KV requirements instead of assuming every hybrid group scales like the first group.

## Important Interpretation

The old hybrid KV log:

```text
Hybrid GPU KV cache aggregate block capacity: N tokens
```

is not a single-request maximum context length for Qwen3.5-style hybrid models. It is a scheduler/block-pool lower-bound metric. The new log:

```text
Hybrid GPU KV cache estimated single-request max context: N tokens
```

is the number to use when asking "roughly how long can one request be with this KV allocation". It is still an estimate; activation peak, burst stage memory, MTP, graph memory, and runtime headroom can still OOM first.

## Completed Historical Runs

These were from the earlier one-turn no-graph sweep before the warm TTFT change:

| Run | KV | Prompt | Max Model Len | Result | TTFT | Decode TPS After First Token | Peak GPU |
|---|---:|---:|---:|---|---:|---:|---:|
| `kv1g_prefill5k_decode512` | 1 GiB | 5K | 8192 | success | 28.884s | 7.119 tok/s | ~31.18 GiB |
| `kv2g_prefill10k_decode512` | 2 GiB | 10K | 16384 | success | 20.898s | 6.823 tok/s | ~31.11 GiB |
| `kv4g_prefill20k_decode512` | 4 GiB | 20K | 32768 | success | 41.974s | 6.646 tok/s | ~30.68 GiB |
| `kv4g_prefill30k_decode512` | 4 GiB | 30K | 32768 | success | 39.161s | 5.870 tok/s | ~31.10 GiB |

## Warm TTFT Verification

Run directory:

```text
/home/jiamin/projects/CFIE/.bench_logs/20260521_kv_prefill_warm2_sync
```

Command shape:

```bash
CFIE_MONITOR_INTERVAL=1 ./benchmarks/run_logged_wsl_command.sh \
  kv1g_prefill5k_decode512_turns2_nograph \
  .bench_logs/20260521_kv_prefill_warm2_sync \
  ./.wsl-venv/bin/python benchmarks/run_long_prefill_decode.py \
  --model /home/jiamin/models/Qwen3.5-122B-A10B-GPTQ-Int4/snapshots/5b9f0050d3ec98b0c81a7716776533c5eacebb64 \
  --target-input-tokens 5000 \
  --max-new-tokens 512 \
  --turns 2 \
  --warm-prompt-mode rotated \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-memory-bytes 1073741824 \
  --gpu-slots-per-layer 16 \
  --prefill-burst-slots 256 \
  --prepare-cpu-copy-batch-size 8 \
  --cpu-static-pinned-gb 44 \
  --language-model-only \
  --skip-mm-profiling \
  --marlin-input-dtype fp8 \
  --temperature 0 \
  --spec-method mtp \
  --num-speculative-tokens 1 \
  --enforce-eager \
  --force-exit-after-result
```

Result:

| Turn | Prompt Mode | TTFT | Run Seconds | Decode TPS After First Token |
|---|---|---:|---:|---:|
| cold turn 1 | base | 20.073s | 91.169s | 7.188 tok/s |
| warm turn 2 | rotated | 4.478s | 76.063s | 7.138 tok/s |

GPU monitor:

- samples: 517
- peak memory: 30,683 MiB
- peak GPU util: 97%
- memory after shutdown: ~757 MiB

KV log from this run:

```text
Initial free memory 30.2 GiB, reserved 1.0 GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling.
Hybrid GPU KV cache aggregate block capacity: 8,384 tokens (computed as num_blocks / num_groups * min_block_size). This is a lower-bound scheduling metric, not the single-request maximum context length.
Hybrid GPU KV cache estimated single-request max context: 20,960 tokens (computed from each KV group spec, so Mamba/linear/state groups do not scale like full attention).
Approximate maximum concurrency at configured max_model_len=8,192: 1.46x
```

## Takeaways

- The 20s first-step latency is not representative of every later conversation turn in the same engine. With a second same-length but rotated prompt, TTFT dropped to 4.48s.
- Decode throughput stayed stable across the two turns, around 7.1 tok/s after the first output token.
- The corrected KV log now exposes why the old `8,384 tokens` line was misleading: it is lower-bound scheduler capacity, while the single-request estimate for 1 GiB KV in this run was 20,960 tokens.
- Future long-context sweeps should use `--turns 2` and report both cold and warm TTFT.
