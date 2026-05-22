# 2026-05-21 Prepare Fast Path Round 1

## Scope

- Target path: Qwen3.5-122B tiered MoE prepare hot path on Windows.
- Goal: remove avoidable Python work before/around `pinned -> H2D -> GPU scatter`.
- Standard run shape used for the smoke benchmark:
  - `gpu_slots_per_layer=16`
  - `prefill_burst_slots=256`
  - `prepare_cpu_copy_threads=8`
  - `cpu_static_pinned_gb=44`
  - `kv_cache_memory_bytes=4000000000`
  - `max_model_len=128000`
  - `max_num_batched_tokens=8192`
  - `marlin_input_dtype=fp8`
  - no CUDA graph

## Code Changes

- Renamed the runtime prepare copy control to `prepare_cpu_copy_threads`.
  - The old `prepare_cpu_copy_batch_size` is kept only as a deprecated CLI/API alias.
  - Thread count no longer controls H2D batching. Missing experts are written as one batch.
- Replaced router-score victim selection with a per-layer round-robin cursor.
  - Prepare now checks resident hits from the CPU mirror map and picks the next non-protected resident slots.
  - This avoids per-step router-score CPU tensor work and sorting.
- Moved CPU source lookup into a single plan-time resolution step.
  - Prepare no longer calls a materialize/repack path for missing experts.
  - If an expert is absent from CPU static mirror or is not runtime-ready, prepare raises.
- Removed Python sorting before batch stage write.
  - Stage prefix order follows missing expert order directly.
- Cached target field/device lookup and reused the GPU slot-id buffer.
  - Avoids repeated Python field discovery and repeated CUDA tensor allocation.
- Removed dead prepare helpers from `weight_offload.py`:
  - old materialize batch source helper
  - old sort helper
  - old router-score victim helper
  - old single-slot victim helper

## Timing Comparison

Timing runs use `CFIE_BENCH_TIMING=1`, so they include synchronizations and are only for bottleneck analysis.

| Layer Type | Before Total | After Total | Before Materialize | After Source Resolve | Before Write Overhead | After Write Overhead |
|---|---:|---:|---:|---:|---:|---:|
| pinned static direct H2D | 2.775 ms/layer | 2.370 ms/layer | 0.240 ms | 0.004 ms | 1.043 ms | 0.066 ms |
| pageable -> pinned stage | 4.873 ms/layer | 4.670 ms/layer | 0.256 ms | 0.005 ms | 0.853 ms | 0.099 ms |

Detailed after-run averages:

| Layer Type | plan | write | CPU pack | H2D | GPU scatter | install |
|---|---:|---:|---:|---:|---:|---:|
| pinned static direct H2D | 0.412 ms | 0.663 ms | 0.032 ms | 0.397 ms | 0.168 ms | 1.153 ms |
| pageable -> pinned stage | 0.499 ms | 3.205 ms | 2.587 ms | 0.279 ms | 0.240 ms | 0.806 ms |

Interpretation:

- The removed Python overhead is visible:
  - source resolution/materialize is near-zero now;
  - stage write overhead outside CPU pack/H2D/scatter is near-zero now.
- The next hot spot is `install`.
  - It still updates expert-map metadata with scalar per-expert operations.
  - A naive batched map update was tested and reverted because it reduced non-timing throughput.
  - The correct next step is a native prepare install/scatter op, not more Python-side tensor creation.

## Decode Smoke Result

Non-timing 64-token smoke:

| Run | Result |
|---|---:|
| total TPS | 6.01 tok/s |
| steady TPS after first token | 7.16 tok/s |
| first step | 1.85 s |
| generated tokens | 64 |

Result file:

```text
.bench_logs/20260521_prepare_fastpath_round1/decode64_notiming.json
```

After adding the fast GPU-stage field view, I re-ran the 64-token smoke:

| Run | Result |
|---|---:|
| total TPS | 5.43 tok/s |
| steady TPS after first token | 6.32 tok/s |
| first step | 1.82 s |
| generated tokens | 64 |

Result file:

```text
.bench_logs/20260521_prepare_fastpath_round3/decode64_after_stage_view.json
```

Environment note from this run:

```text
Skipping import of cpp extensions due to incompatible torch version.
Please upgrade to torch >= 2.11.0 (found 2.10.0+cu130).
```

Follow-up probe in the same Windows venv:

```text
torch 2.10.0+cu130
has cpu copy op True
has device copy op True
```

So the warning is real, but it is not direct evidence that the prepare copy helper ops are missing. Treat the round3 TPS drop as benchmark variability / MTP path sensitivity unless a timing run proves otherwise.

Startup notes from the same run:

- target weights: 34.36 s
- tiered expert cache attach: about 92 s, with fp8 Marlin-ready cache loaded from `.cfie_marlin_ready_cache/v1/fp8/layer_XXX.safetensors`
- MTP drafter weights: 19.75 s
- total model loading log: 22.12 GiB memory and 154.09 s
- manual KV cache: 3.73 GiB
- estimated single-request max context: 129,952 tokens
- pinned static covered layers 0..36; layer 37+ used pageable static mirror

## Failed Attempt Kept For Reference

I tested a Python-side batched GPU expert-map update after the first fast-path patch.

Result:

| Run | steady TPS |
|---|---:|
| first fast-path patch | 7.16 tok/s |
| batched map-update attempt | 6.50 tok/s |

The map-update attempt was reverted. It added synchronization/allocation cost without removing enough Python work.

## Next Work

- Implemented on 2026-05-22. See:

```text
docs/experiments/2026-05-22_native_scatter_install.md
```
