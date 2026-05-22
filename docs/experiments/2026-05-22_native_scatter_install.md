# 2026-05-22 Native Scatter And Install

## Scope

- Target: Qwen3.5-122B tiered MoE prepare hot path on Windows.
- Goal: move GPU-stage scatter plus `expert_map` install out of Python and into a native op.
- Standard run shape:
  - `gpu_slots_per_layer=16`
  - `prefill_burst_slots=256`
  - `prepare_cpu_copy_threads=32`
  - `cpu_static_pinned_gb=44`
  - `kv_cache_memory_bytes=4000000000`
  - `max_model_len=128000`
  - `max_num_batched_tokens=8192`
  - `marlin_input_dtype=fp8`
  - `spec_method=mtp`, `num_speculative_tokens=1`
  - no CUDA graph, eager mode

## Code Changes

- Added native Python wrappers in `cfie/_custom_ops.py` for:
  - `moe_batch_load_unquantized_runtime_and_install`
  - `moe_batch_load_gptq_runtime_and_install`
- Added native schemas and bindings in `csrc/torch_bindings.cpp`.
- Added install-capable entry points in `csrc/windows_triton_compat_ops.cpp`.
- Added `csrc/moe_runtime_scatter.cu`.
  - The fast path launches one CUDA kernel over `(expert, field)` blocks.
  - It copies GPU runtime stage field payloads into resident slots.
  - It updates `expert_map` in the same kernel.
  - Unsupported layouts fall back to the existing ATen `index_copy_` path.
- Updated `cfie/offload/weight_offload.py`.
  - The prepare path now requests native scatter+install when writing resident slots.
  - CPU-side `_slot_to_global` state is still updated after the native op.
  - Timing logs now include `native_scatter_install` and `installed_mappings`.

## Build And Validation

Build command:

```powershell
cmd /c "call C:\Users\13642\vs_buildtools\VC\Auxiliary\Build\vcvars64.bat >nul && ..\.venv\Scripts\cmake.exe --build .\cmake-build-release-visual-studio --target _C --config Release -j 4"
Copy-Item .\cmake-build-release-visual-studio\_C.pyd .\cfie\_C.pyd -Force
```

Validation:

- Direct CUDA smoke for GPTQ scatter+install: passed.
- Target unit tests: `18 passed, 16 warnings`.
- 122B decode smoke: passed, output text normal.

## Timing Comparison

Timing runs use `CFIE_BENCH_TIMING=1`; they include synchronization and are only for bottleneck analysis.

Comparison between the first C++ wrapper path and the fused CUDA scatter kernel:

| Path | Rows | Missing Avg | CPU Pack | H2D | GPU Scatter | Prepare Total |
|---|---:|---:|---:|---:|---:|---:|
| C++ wrapper, warm all | 384 | 8.24 | 0.613 ms | 0.391 ms | 1.302 ms | 2.999 ms |
| CUDA scatter, warm all | 384 | 8.24 | 0.551 ms | 0.350 ms | 0.186 ms | 2.648 ms |
| C++ wrapper, pinned warm | 296 | 8.59 | 0.035 ms | 0.423 ms | 1.329 ms | 2.445 ms |
| CUDA scatter, pinned warm | 296 | 8.59 | 0.029 ms | 0.378 ms | 0.168 ms | 2.211 ms |
| C++ wrapper, pageable warm | 88 | 7.07 | 2.558 ms | 0.281 ms | 1.211 ms | 4.860 ms |
| CUDA scatter, pageable warm | 88 | 7.07 | 2.306 ms | 0.255 ms | 0.246 ms | 4.120 ms |

Main result:

- `stage_gpu_scatter` is no longer the main fixed overhead.
- Warm scatter dropped from about `1.30 ms/layer` to `0.19 ms/layer`.
- Pageable layers are now mostly limited by pageable-to-pinned CPU copy.
- Pinned layers are now mostly limited by H2D, remaining scheduling overhead, and model compute.

Timing logs:

```text
.bench_logs/20260522_native_scatter_install_timing/decode16_timing.stderr.log
.bench_logs/20260522_cuda_scatter_install_timing/decode16_timing.stderr.log
```

## Plan Timing Correction

After the CUDA scatter work, the prepare timer was split again because the
older `plan` number included unrelated work after loading. The corrected run is:

```text
.bench_logs/20260522_plan_breakdown_timing_rerun/decode16_timing.stderr.log
.bench_logs/20260522_plan_breakdown_timing_rerun/decode16_timing.json
```

Warm-layer averages:

| Layer Group | Rows | Missing Avg | Request Unique | Plan Total | Resident Check | Victim Pick | Plan Zip | Stage Write | CPU Pack | H2D | GPU Scatter | Install | Prepare Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm all | 384 | 8.24 | 0.126 ms | 0.009 ms | 0.003 ms | 0.003 ms | 0.002 ms | 1.270 ms | 0.697 ms | 0.335 ms | 0.177 ms | 0.005 ms | 1.424 ms |
| pinned static | 296 | 8.59 | 0.128 ms | 0.008 ms | 0.003 ms | 0.003 ms | 0.002 ms | 0.601 ms | 0.027 ms | 0.361 ms | 0.158 ms | 0.004 ms | 0.755 ms |
| pageable static | 88 | 7.07 | 0.121 ms | 0.009 ms | 0.003 ms | 0.004 ms | 0.002 ms | 3.521 ms | 2.950 ms | 0.246 ms | 0.239 ms | 0.007 ms | 3.675 ms |

Important correction:

- The real resident/victim planning path is not the bottleneck; it is about
  `0.009 ms/layer`.
- The previous large `plan` value was caused by timer attribution around the
  load path, repeated request normalization, and timing/stat logging.
- A CUDA planning kernel is not justified yet. It would remove only a few
  microseconds per layer while adding launch and small-result transfer overhead.
- The next useful targets are `request_unique` and the pageable-to-pinned CPU
  pack path.

## Small Top-k Unique Fast Path

Decode usually passes only a small `topk_ids` tensor into prepare. The previous
path still launched `torch.unique` on GPU, then copied the unique result back to
CPU. A small-input fast path now copies the raw id tensor to CPU and performs
deduplication there; larger tensors keep the old GPU unique path.

Validation:

```text
.bench_logs/20260522_small_unique_timing/decode16_timing.stderr.log
.bench_logs/20260522_small_unique_timing/decode16_timing.json
```

Warm-layer averages after the change:

| Layer Group | Rows | Requested Avg | Missing Avg | Request Unique | Plan Total | Stage Write | CPU Pack | H2D | GPU Scatter | Install | Prepare Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm all | 528 | 13.45 | 8.57 | 0.059 ms | 0.009 ms | 1.477 ms | 0.804 ms | 0.377 ms | 0.203 ms | 0.006 ms | 1.550 ms |
| pinned static | 407 | 13.61 | 8.81 | 0.058 ms | 0.009 ms | 0.694 ms | 0.032 ms | 0.403 ms | 0.178 ms | 0.005 ms | 0.765 ms |
| pageable static | 121 | 12.93 | 7.79 | 0.062 ms | 0.010 ms | 4.113 ms | 3.402 ms | 0.291 ms | 0.289 ms | 0.009 ms | 4.191 ms |

This removes roughly `0.06-0.07 ms/layer` from the fixed request processing
cost in decode. It does not change the main conclusion: pageable-to-pinned CPU
pack is now the dominant prepare-side bottleneck for layers outside the pinned
static mirror.

Non-timing 512-token checks after this change:

| Run | Total TPS | Steady TPS | First Step | Tail 64 TPS | Notes |
|---|---:|---:|---:|---:|---|
| `20260522_small_unique_decode512` | 7.47 tok/s | 7.66 tok/s | 1.81 s | 8.20 tok/s | output normal; still emitted periodic prepare stats |
| `20260522_small_unique_decode512_nostats` | 7.12 tok/s | 7.31 tok/s | 1.99 s | 7.89 tok/s | output normal; prepare stats disabled by default |

Result files:

```text
.bench_logs/20260522_small_unique_decode512/decode512.json
.bench_logs/20260522_small_unique_decode512_nostats/decode512.json
```

## CPU Copy Threads Check

A synthetic pageable/static-mirror copy test showed that `32` native copy
threads can improve the 8-expert pageable-to-pinned case versus `8` threads.
The real 122B timing run confirms a smaller but measurable improvement:

```text
.bench_logs/20260522_cpu_copy_threads32_timing/decode16_timing.stderr.log
.bench_logs/20260522_cpu_copy_threads32_timing/decode16_timing.json
```

Warm-layer averages for `prepare_cpu_copy_threads=32`:

| Layer Group | Rows | Requested Avg | Missing Avg | Request Unique | Plan Total | Stage Write | CPU Pack | H2D | GPU Scatter | Install | Prepare Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| warm all | 528 | 13.45 | 8.57 | 0.059 ms | 0.010 ms | 1.416 ms | 0.732 ms | 0.377 ms | 0.213 ms | 0.006 ms | 1.489 ms |
| pinned static | 407 | 13.61 | 8.81 | 0.056 ms | 0.009 ms | 0.709 ms | 0.032 ms | 0.403 ms | 0.190 ms | 0.005 ms | 0.780 ms |
| pageable static | 121 | 12.93 | 7.79 | 0.066 ms | 0.011 ms | 3.792 ms | 3.086 ms | 0.291 ms | 0.292 ms | 0.008 ms | 3.877 ms |

Compared with the 8-thread timing run above, pageable CPU pack improved from
`3.402 ms/layer` to `3.086 ms/layer`. This is worth making the default, but it
does not change the larger conclusion: pageable memory copy bandwidth remains
the limiting prepare-side cost when a layer is not covered by pinned static
mirror.

## Decode Results

Non-timing 64-token smoke after the fused CUDA scatter kernel:

| Metric | Value |
|---|---:|
| total TPS | 5.93 tok/s |
| steady TPS after first token | 7.10 tok/s |
| first step | 1.92 s |
| generated tokens | 64 |

Result file:

```text
.bench_logs/20260522_cuda_scatter_install/decode64_cuda_scatter.json
```

Non-timing 512-token decode:

| Metric | Value |
|---|---:|
| total TPS | 6.55 tok/s |
| steady TPS after first token | 6.71 tok/s |
| first step | 2.06 s |
| tail 64 TPS | 6.89 tok/s |
| tail 128 TPS | 6.79 tok/s |
| tail 256 TPS | 6.79 tok/s |
| generated tokens | 512 |

Result file:

```text
.bench_logs/20260522_cuda_scatter_install_decode512/decode512_cuda_scatter.json
```

## Interpretation

- The native CUDA scatter+install direction is valid.
- The expected large scatter reduction happened, but end-to-end throughput did not double.
- Current next bottlenecks are:
  - pageable-to-pinned CPU copy for layers not covered by pinned static mirror;
  - H2D transfer itself;
  - MoE/GPU compute and speculative decode scheduling;
  - startup expert cache attach, which still dominates repeated benchmark iteration time.

The next optimization should not add more Python-side prepare reshaping. Further work should focus on:

- reducing pageable layer traffic or increasing useful pinned coverage;
- measuring CPU copy bandwidth under the new kernel path;
- avoiding per-run startup cost by keeping a warm OpenAI service for interactive tests;
- only then revisiting larger architectural changes such as deeper H2D/compute overlap.
