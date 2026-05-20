# Qwen3.5-122B WSL / VL / Long Context Optimization Plan

## Goal

目标是在 WSL 内部模型路径下，把 Qwen3.5-122B-A10B-GPTQ-Int4 的启动、长 prefill、decode 和 GUI agent/VL 场景跑通并压低端到端延迟。

当前优先级：

1. 先实现项目专用 Marlin-ready expert cache，避免每次启动都从原始 GPTQ expert 重新 GPU repack。
2. 修复或规避 PIECEWISE CUDA graph capture 异常。
3. 验证 Qwen3.5 VL 模块是否仍能正常处理图像/视频帧输入。
4. 验证常用显存配置下 256K context 的 KV cache 是否可规划。
5. 测试约 50K token prefill + decode 的真实耗时。
6. 增加可选的 MoE FP8 activation 路径，作为后续性能优化项。

## Baseline Facts

- WSL distro: `Ubuntu-22.04-D`
- WSL model path:
  `/home/jiamin/models/Qwen3.5-122B-A10B-GPTQ-Int4/snapshots/5b9f0050d3ec98b0c81a7716776533c5eacebb64`
- Previous smoke command used:
  - `--max-model-len 4096`
  - `--max-new-tokens 16`
  - `--max-num-batched-tokens 2112`
  - `--gpu-slots-per-layer 32`
  - `--cpu-static-pinned-gb 44`
- Previous WSL smoke showed:
  - HF shard loading: about 49 seconds
  - tiered CPU static mirror attach: about 360 seconds total
  - CPU static expert pool per layer: about 1192.50 MiB
  - pinned layers with `--cpu-static-pinned-gb 44`: layers 0-36 pinned, layers 37-47 pageable
  - FULL CUDA graph is now downgraded to PIECEWISE when MoE tiered cache is enabled
  - PIECEWISE split list includes `cfie::moe_forward` and `cfie::moe_forward_shared`
  - Current remaining graph issue is a PyTorch 2.10 AOT output descriptor assertion during compile, not the old stream-capture `torch.unique` failure.

## Task 1: Project Marlin-Ready Expert Cache

### Result So Far

- Implemented `CFIE_MARLIN_READY_CACHE` in `cfie/offload/weight_offload.py`.
- Default cache path:
  `<model_snapshot>/.cfie_marlin_ready_cache/v1/layer_XXX.safetensors`
- Disable with `CFIE_MARLIN_READY_CACHE=0`.
- Override path with `CFIE_MARLIN_READY_CACHE_DIR=/path/to/cache`.
- Built cache for 48 MoE layers.
- Cache size: 48 files, 55.90 GiB.
- First build run:
  - HF shard load: 42.25 s
  - model load + expert attach: 329.50 s
  - engine init after model load: 68.53 s
- Cache-hit run:
  - HF shard load: about 41.9 s
  - model load + expert attach: 114.81-117.64 s
  - engine init after model load: 18.81-19.58 s
- The second run logs `Loaded Marlin-ready expert cache` for all 48 layers, so the repeated GPTQ-to-Marlin repack path is skipped.

### Problem

Current initialization path still does this per layer:

```text
raw GPTQ safetensors expert
  -> CPU raw bundle
  -> GPU temporary tensors
  -> gptq_marlin_moe_repack
  -> marlin_moe_permute_scales
  -> CPU runtime-ready static mirror
```

This makes every benchmark startup pay the GPTQ-to-Marlin preprocessing cost again.

### Target Design

Create a project-owned cache format, separate from the original HuggingFace checkpoint:

```text
original HF GPTQ checkpoint
  -> one-time conversion
  -> CFIE Marlin-ready expert cache
  -> later startups load cache directly
  -> CPU static mirror is runtime-ready immediately
```

The cache must store runtime-ready expert-major bundles:

- `runtime.w13_qweight`
- `runtime.w2_qweight`
- `runtime.w13_scales`
- `runtime.w2_scales`
- `runtime.w13_qzeros`
- `runtime.w2_qzeros`
- optional `runtime.w13_g_idx`
- optional `runtime.w2_g_idx`
- optional `runtime.w13_g_idx_sort_indices`
- optional `runtime.w2_g_idx_sort_indices`

### Cache Key

Cache metadata must include enough information to reject stale caches:

- source model path or resolved snapshot path
- model type
- quantization method
- layer count
- experts per layer
- hidden/intermediate dimensions
- GPTQ bits, group size, desc_act
- Marlin input dtype mode
- CFIE cache format version
- source file mtimes/sizes or a source manifest hash

### Acceptance

- First run can build the cache.
- Second run must skip `gptq_marlin_moe_repack` for cached experts.
- CPU static mirror initialization time must drop materially.
- If cache metadata mismatches, the system must rebuild or ignore cache instead of silently loading bad tensors.

## Task 2: PIECEWISE CUDA Graph Exception

### Result So Far

- Fixed one local `PiecewiseBackend` crash: subgraphs without symbolic shape indices now use the first compiled runnable instead of indexing an empty `sym_shape_indices`.
- Re-tested experimental PIECEWISE compile with tiered MoE and hit a second shape boundary failure:
  `Shape: 2113 out of considered ranges: [(1, 2112)]`.
- Root cause: Qwen3.5/Mamba warmup can pass a symbolic size one token above `max_num_batched_tokens`; current compile range generation does not cover it.
- For stability, tiered MoE now disables torch.compile and CUDA graph by default because prepare/H2D/scatter remains capture-unsafe.
- Experimental override:
  `--allow-tiered-moe-compile`
- Verified non-eager CLI now starts with:
  - `compilation_config.mode = CompilationMode.NONE`
  - `cudagraph_mode = CUDAGraphMode.NONE`
  - log: `Disabling torch.compile and CUDA graph by default for stability`

### Current State

FULL graph is correctly disabled for MoE tiered cache because runtime prepare/H2D/scatter is not capture-safe.

MoE ops have been added as split boundaries:

- `cfie::moe_forward`
- `cfie::moe_forward_shared`

The previous stream-capture failure from `torch.unique(topk_ids)` inside MoE prepare is therefore addressed.

### Remaining Failure

The current failure is:

```text
AssertionError in torch._functorch._aot_autograd.utils.call_and_expect_output_descs
```

This occurs during WSL PIECEWISE AOT compile/profile, before real decode.

### Acceptance

- Either PIECEWISE compile/capture succeeds, or the project explicitly disables the incompatible compile subpath for this model while keeping runtime stable.
- The final test command must clearly report whether CUDA graph is active, partially active, or disabled.

## Task 3: Qwen3.5 VL Module Smoke Test

### Goal

Verify that text-only optimization has not broken VL/model multimodal paths.

### Result So Far

- Added `benchmarks/run_vl_smoke.py`.
- First run proved the processor produced `pixel_values` and `image_grid_thw`, but M-RoPE position setup resolved the inner text model and failed on multimodal input.
- Fixed `_init_mrope_positions` / `_init_xdrope_positions` to resolve the top-level model object instead of the predictor/language-model helper.
- Re-run succeeded with:
  - `--max-model-len 4096`
  - `--max-num-batched-tokens 2112`
  - `--kv-cache-memory-bytes 2147483648`
  - `--gpu-slots-per-layer 16`
  - `--max-new-tokens 4`
- Result:
  - model load: 116.55 s
  - engine init after load: 18.85 s
  - VL generation run: 83.42 s
  - generated text: `<think>\n\n</think>\n\n`

### Acceptance

- A minimal image input can pass through tokenizer/processor and engine.
- A short generated text response is produced.
- If VL fails due to missing processor/runtime support, document the exact failing module and required fix.

## Task 4: 256K Context KV Cache Test

### Goal

Determine whether 256K single-request context can be allocated under practical resident expert settings.

### Result So Far

All tests used manual KV reservation with `max_model_len=262144`, `max_num_batched_tokens=2112`, prefix caching enabled, language-model-only mode, and the Marlin-ready cache.

| GPU slots/layer | KV bytes | KV GiB | Reported KV token capacity | Decode smoke |
| --- | ---: | ---: | ---: | --- |
| 16 | 25,769,803,776 | 24 | 262,000 | success |
| 16 | 26,843,545,600 | 25 | 272,480 | success |
| 24 | 26,843,545,600 | 25 | 272,480 | success |
| 32 | 26,843,545,600 | 25 | 272,480 | success |

Notes:

- 25 GiB KV is enough to pass the 256K planning boundary in this single-request configuration.
- Increasing `gpu_slots_per_layer` increases resident expert memory. The 32-slot run succeeded but had much slower model attach in that run, so 24 is the current safer long-context default for further tests.
- The old startup OOM was from dummy/profile execution and graph/capture behavior, not from the 256K KV capacity itself once `kv_cache_memory_bytes` is set.

### Test Matrix

Start with:

- `gpu_slots_per_layer=16`
- `gpu_slots_per_layer=24`
- `gpu_slots_per_layer=32`
- `kv_cache_dtype=auto`
- then test `kv_cache_dtype=fp8` if available and correct

### Acceptance

- Report available KV cache memory.
- Report maximum supported context length or the failure reason.
- Distinguish OOM from scheduler/config limits.

## Task 5: 50K Token Prefill + Decode Benchmark

### Goal

Measure GUI-agent style long prefill:

```text
about 50K input tokens + short decode output
```

### Result So Far

Added `benchmarks/run_long_prefill_decode.py` and ran a 50K-token prompt with:

- `--target-input-tokens 50000`
- `--max-new-tokens 16`
- `--max-model-len 65536`
- `--max-num-batched-tokens 8192`
- `--kv-cache-memory-bytes 8589934592`
- `--gpu-slots-per-layer 24`
- `--cpu-static-pinned-gb 0`

The run did not complete within 30 minutes. It was not a clean engine crash; it timed out while still processing late layers.

The diagnosis logs are important:

- During long prefill, each layer/chunk often requested about 18-24 distinct experts, not just decode top-8.
- `stage_resident_hit` was often only about 20%-60%.
- `cpu_hits` and `evictions` reached tens of thousands per layer during the run.
- This means the current decode-oriented tiered MoE prepare path thrashes badly for large prefill chunks.

Conclusion:

The current resident/stage policy is acceptable to keep testing decode, but it is not yet a valid GUI-agent 50K prefill solution. Long prefill needs a separate strategy, for example larger/full expert residency during prefill, a prefill-burst expert pool sized to the chunk's distinct routed experts, or a dedicated prefill staging policy that avoids per-layer eviction churn.

### Metrics

- engine startup time
- prefill wall time
- first decode token latency
- decode tokens/s
- total end-to-end latency
- resident hit / CPU static hit / pageable staging count if logs are enabled

### Acceptance

- Run once with logs for diagnosis.
- Run once with nonessential logs disabled for throughput.
- Produce a JSON result and a short text summary.

## Task 6: Optional MoE FP8 Activation Path

### Goal

Add a CLI/config option to enable FP8 activation input for GPTQ Marlin MoE where supported.

### Result So Far

The project already has a lower-level opt-in path:

- CLI/config: `--marlin-input-dtype fp8`
- reader: `get_marlin_input_dtype(prefix)`
- GPTQ Marlin MoE receives `input_dtype=torch.float8_e4m3fn`
- load-time preprocessing calls `ops.marlin_int4_fp8_preprocess(...)`
- runtime activation quantization uses per-token FP8 scale metadata through `QuantFP8(False, GroupShape.PER_TOKEN)`

Added explicit benchmark switches:

- `benchmarks/run_decode_512.py --marlin-input-dtype auto|int8|fp8`
- `benchmarks/run_long_prefill_decode.py --marlin-input-dtype auto|int8|fp8`
- `benchmarks/run_vl_smoke.py --marlin-input-dtype auto|int8|fp8`

Default remains `auto`, which keeps the existing BF16/FP16 activation path.

Also updated tiered MoE cache so it no longer rejects GPTQ Marlin MoE when `input_dtype` is FP8/INT8. Cache metadata now records `marlin_input_dtype`, and non-default activation dtypes use separate cache subdirectories:

```text
.cfie_marlin_ready_cache/v1/layer_XXX.safetensors       # default A16 activation path
.cfie_marlin_ready_cache/v1/fp8/layer_XXX.safetensors   # FP8 activation path
```

This avoids overwriting the existing default 56 GiB cache when testing FP8.

FP8 smoke on RTX 5090 / SM120:

- command used `--marlin-input-dtype fp8`, `gpu_slots_per_layer=16`, `kv_cache_memory_bytes=2GiB`, `max_new_tokens=1`
- first run built the FP8 cache:
  - FP8 cache files: 48
  - FP8 cache size: 56 GiB
  - model load + FP8 cache build: 360.76 s
  - engine init after load: 21.76 s
  - first decode step: 12.97 s
- second run loaded the FP8 cache:
  - model load + cache attach: 117.99 s
  - engine init after load: 18.54 s
  - first decode step: 10.12 s

The smoke proves the FP8 activation path can initialize and generate one token under tiered MoE. It is not yet a throughput win based on the 1-token test; a 128/512 decode comparison is still needed because first-token overhead dominates this measurement.

### Expected Direction

Current code already has partial support signals:

- `GPTQMarlinMoEMethod.input_dtype`
- `get_marlin_input_dtype(prefix)`
- `ops.marlin_int4_fp8_preprocess(...)`
- FP8 activation utilities under fused MoE modules

The option should set MoE Marlin activation dtype to `torch.float8_e4m3fn` only when:

- hardware supports the path
- quant method is GPTQ Marlin MoE
- correctness smoke test passes

### Metadata

Each layer needs enough scale metadata to quantize activations safely:

- dynamic abs/amax scale for activation input
- chosen FP8 format
- fallback path on unsupported layer/backend

### Acceptance

- Default remains unchanged.
- New option is opt-in.
- Correctness smoke compares FP8 option against baseline text generation on a fixed seed.
- Benchmark reports speed and any output drift.

## Working Order

1. Implement and test Marlin-ready cache.
2. Re-run WSL 16-token startup smoke.
3. Fix/disable PIECEWISE graph issue.
4. Run VL smoke.
5. Run 256K KV planning tests.
6. Run 50K prefill + decode benchmark.
7. Add FP8 activation option and test separately.
