# 2026-05-20 WSL Graph MTP KV/Prefill Experiments

## Rules

- One dated file per experiment day. Append new runs here; do not overwrite older dated files.
- Record every run with: command intent, exact important CLI, OOM status, model/graph/KV memory evidence, peak `nvidia-smi` memory, prefill/first-step time, and decode token rate.
- Graph/MTP tests are experimental until output correctness and speed are both acceptable.

## Environment

- WSL repo: `/home/jiamin/projects/CFIE`
- Model: `/home/jiamin/models/Qwen3.5-122B-A10B-GPTQ-Int4/snapshots/5b9f0050d3ec98b0c81a7716776533c5eacebb64`
- GPU: RTX 5090 32GB
- Date/time zone: Asia/Shanghai
- Current pinned-memory guard:
  - WSL now probes real `torch.empty(..., pin_memory=True)` support.
  - If `--cpu-static-pinned-gb` or `--cpu-static-pinned-layers` is set while pinned memory is unavailable, planning fails immediately.
  - Pinned allocation failures no longer silently fall back to pageable storage.

## Standard CLI Baseline

Common graph/MTP options for this run series:

```bash
--gpu-slots-per-layer 16
--prefill-burst-slots 256
--prepare-cpu-copy-threads 8
--cpu-static-pinned-gb 44
--marlin-input-dtype fp8
--piecewise-cudagraph
--cudagraph-decode-capture-sizes 1 2
--cudagraph-prefill-capture-sizes
--spec-method mtp
--num-speculative-tokens 1
```

## Run Log

### Note: meaning of `--gpu-memory-utilization`

- Code path:
  - `split_gpu_memory_budget(total_memory, gpu_memory_utilization)`
  - `static_budget = ceil(total_gpu_memory * gpu_memory_utilization)`
  - `runtime_headroom = total_gpu_memory - static_budget`
- Meaning:
  - The ratio controls the static GPU memory budget, not the current observed GPU usage.
  - Static budget is the pool used for model weights, resident non-KV state, GPU MoE resident slots/stages, and automatically sized KV cache.
  - The remaining part is reserved as runtime headroom for prefill/decode peaks, temporary workspaces, CUDA graph capture overhead, and other dynamic allocations.
- Example on RTX 5090 log total 31.84 GiB:
  - `0.88` means static budget about 28.02 GiB and runtime headroom about 3.82 GiB.
  - `0.94` means static budget about 29.93 GiB and runtime headroom about 1.91 GiB.
  - `0.95` means static budget about 30.25 GiB and runtime headroom about 1.59 GiB.
- Startup check:
  - The code requires startup free memory to be at least the static budget.
  - That is why the 100K run with `0.95` failed early: free memory was 30.2 GiB, but required static budget was 30.25 GiB.
- Manual KV exception:
  - If `--kv-cache-memory-bytes` is set, automatic KV sizing ignores `gpu_memory_utilization`.
  - The initial static-budget/headroom split still exists for worker setup and safety checks.

### Code Change: runtime stage preallocation

- Change:
  - `SharedRuntimeExpertStagePool` now has an explicit `preallocate(...)` path.
  - Tiered MoE attach preallocates the shared CPU runtime stage and GPU H2D runtime stage during initialization.
  - For the standard `--prefill-burst-slots 256` configuration, runtime stage capacity is 256 expert-major slots.
- Reason:
  - Before this change, `prefill_burst_pool` itself was allocated during initialization, but the separate CPU/GPU runtime H2D stage could still allocate or grow lazily during the first prefill burst.
  - That meant KV/cache/graph could consume memory first, and the first real burst transfer could OOM in `_move_runtime_ready_bundles_to_device`.
  - Preallocation makes this memory visible before KV cache sizing/profiling and moves OOM to startup.

### Run 1: decode512 graph + MTP smoke

- Time: 2026-05-21 00:07 Asia/Shanghai
- Intent: standard decode speed check before long-context KV sweep.
- Status: success, no OOM.
- Key CLI:
  - `--max-model-len 4096`
  - `--max-num-batched-tokens 512`
  - `--kv-cache-memory-bytes 1073741824`
  - standard baseline options from above.
- Result JSON: `.bench_logs/20260520_graph_mtp_kv/decode512_graph_mtp1_slots16_burst256_fp8_20260520.json`
- Logs:
  - stderr: `.bench_logs/20260520_graph_mtp_kv/decode512_graph_mtp1_slots16_burst256_fp8_20260520.stderr.log`
  - nvidia-smi: `.bench_logs/20260520_graph_mtp_kv/decode512_graph_mtp1_slots16_burst256_fp8_20260520.nvidia_smi.csv`
- Decode:
  - total: 512 tokens in 124.287s, 4.119 tok/s
  - first step: 44.938s
  - steady: 511 tokens in 79.349s, 6.440 tok/s
  - tail 64: 6.670 tok/s
  - tail 128: 6.918 tok/s
  - tail 256: 6.994 tok/s
- Memory:
  - peak `nvidia-smi`: 30442 MiB
  - model loading total: 358.8s, GPU memory after model load 20.95 GiB
  - CUDA graph capture: 113.0s, graph memory 3.65 GiB
  - hybrid KV lower-bound capacity with fixed 1 GiB KV: 8384 tokens
  - static expert mirror: pinned 43.09 GiB across layers 0..36; pageable 12.81 GiB across layers 37..47
- Prepare stats from sampled logs:
  - average resident hit: 43.4%
  - average missing experts: 8.07/layer
  - layers 00-15: 23.6% hit, 11.78 missing/layer
  - layers 16-31: 49.3% hit, 6.96 missing/layer
  - layers 32-47: 57.4% hit, 5.46 missing/layer

### Comparison: graph cost versus no-graph decode

- Graph+MTP standard run above:
  - `gpu_slots_per_layer=16`, `prefill_burst_slots=256`, `cpu_static_pinned_gb=44`, `fp8`, decode graph sizes `[1, 2]`, prefill graph disabled.
  - total: 4.119 tok/s; steady: 6.440 tok/s; tail-256: 6.994 tok/s; first step: 44.938s.
  - graph capture: 113.0s and 3.65 GiB.
  - peak `nvidia-smi`: 30442 MiB.
- Closest no-graph WSL `slots16 + burst256 + pin44` decode512 run:
  - eager/no graph, 512 steps.
  - total: 4.180 tok/s; steady: 5.150 tok/s; first step: 23.258s.
  - model loading GPU memory: 16.25 GiB.
- Interpretation:
  - The exact same `slots16 + burst256 + fp8 + MTP=1` with only graph disabled has not been rerun as a 512-token strict A/B after the latest fixes.
  - Existing data does not justify graph as the default: graph consumes about 3.65 GiB and adds long capture time, while total decode TPS is not better than the closest no-graph run.
  - For long-context prefill/KV tests, keep graph disabled unless a strict same-config A/B proves a meaningful decode-only gain.

### Run 2: decode512 MTP without graph

- Time: 2026-05-21 01:01 Asia/Shanghai
- Intent: strict A/B for `gpu_slot=16 + burst=256 + fp8 + MTP=1 + pinned44`, with graph removed.
- Status: success, no OOM.
- Key CLI difference versus Run 1:
  - removed `--piecewise-cudagraph`
  - removed `--cudagraph-decode-capture-sizes 1 2`
  - added `--enforce-eager`
  - kept `--spec-method mtp --num-speculative-tokens 1`
- Result JSON: `.bench_logs/20260520_graph_mtp_kv/decode512_mtp1_slots16_burst256_fp8_nograph_pinned44_20260521.json`
- Logs:
  - stderr: `.bench_logs/20260520_graph_mtp_kv/decode512_mtp1_slots16_burst256_fp8_nograph_pinned44_20260521.stderr.log`
  - nvidia-smi: `.bench_logs/20260520_graph_mtp_kv/decode512_mtp1_slots16_burst256_fp8_nograph_pinned44_20260521.nvidia_smi.csv`
- Decode:
  - total: 512 tokens in 88.680s, 5.774 tok/s
  - first step: 14.039s
  - steady: 511 tokens in 74.641s, 6.846 tok/s
  - tail 64: 7.556 tok/s
  - tail 128: 7.662 tok/s
  - tail 256: 7.526 tok/s
  - engine steps: 288
- Memory:
  - peak `nvidia-smi`: 30036 MiB
  - model loading total: 348.0s, GPU memory after model load 22.12 GiB
  - no CUDA graph capture
  - hybrid KV lower-bound capacity with fixed 1 GiB KV: 8384 tokens
  - static expert mirror: pinned 43.09 GiB across layers 0..36; pageable 12.81 GiB across layers 37..47
- A/B conclusion against Run 1 graph+MTP:
  - total TPS: 5.774 vs 4.119, no-graph is 40.2% faster.
  - steady TPS: 6.846 vs 6.440, no-graph is 6.3% faster.
  - tail-256 TPS: 7.526 vs 6.994, no-graph is 7.6% faster.
  - first step: 14.039s vs 44.938s, no-graph is 30.9s faster.
  - observed peak `nvidia-smi`: 30036 MiB vs 30442 MiB. The old graph run also reported 3.65 GiB graph capture memory and 113.0s capture time.
  - Decision: graph is not worth enabling for this configuration.
