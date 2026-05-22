# 2026-05-21 Prepare Pinned vs Pageable Timing

## Scope

- Windows benchmark, Qwen3.5-122B-A10B-GPTQ-Int4.
- Goal: compare decode prepare timing for layers whose expert static mirror is already pinned against layers whose expert static mirror is pageable and must first be copied into the pinned runtime stage.
- Timing mode: `CFIE_BENCH_TIMING=1`. This intentionally synchronizes CUDA timing points, so use the TPS only as a synchronized diagnostic number, not as final serving throughput.

## Command Shape

```powershell
$env:CFIE_BENCH_TIMING = '1'
..\.venv\Scripts\python.exe benchmarks\run_decode_512.py `
  --model D:\models\Qwen3.5-122B-A10B-GPTQ-Int4\snapshots\5b9f0050d3ec98b0c81a7716776533c5eacebb64 `
  --prompt "请用一句话回答：CFIE是什么？" `
  --max-new-tokens 64 `
  --max-model-len 128000 `
  --max-num-seqs 1 `
  --max-num-batched-tokens 8192 `
  --kv-cache-memory-bytes 4000000000 `
  --gpu-slots-per-layer 16 `
  --prefill-burst-slots 256 `
  --prepare-cpu-copy-threads 8 `
  --cpu-static-pinned-gb 44 `
  --enable-prefix-caching `
  --language-model-only `
  --skip-mm-profiling `
  --marlin-input-dtype fp8 `
  --spec-method mtp `
  --num-speculative-tokens 1 `
  --temperature 0 `
  --enforce-eager
```

Logs:

- Raw stderr timing log: `.bench_logs\20260521_prepare_pinned_pageable\pinned44_decode64_timing.stderr.log`
- Parsed summary: `.bench_logs\20260521_prepare_pinned_pageable\pinned44_decode64_prepare_summary.json`

## Run Result

| Metric | Value |
|---|---:|
| Generated tokens | 64 |
| Engine steps | 34 |
| Total decode seconds | 12.265 s |
| Total TPS, timing-enabled | 5.218 tok/s |
| Steady TPS after first step, timing-enabled | 6.064 tok/s |
| First step seconds | 1.876 s |

## Prepare Breakdown

Steady rows skip step 1. With `cpu_static_pinned_gb=44`, layers 0..36 were served from pinned static mirror and layers 37..47 were served from pageable static mirror.

| Group | Layers | Samples | Avg staged experts | Avg prepare total | pageable -> pinned | H2D wait | GPU scatter | Stage write |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pinned static source | 0..36 | 1220 | 9.05 | 2.775 ms | 0.035 ms | 0.417 ms | 0.204 ms | 1.698 ms |
| pageable static source | 37..47 | 363 | 7.80 | 4.873 ms | 2.375 ms | 0.276 ms | 0.205 ms | 3.709 ms |

Interpretation:

- The direct pinned path is not dominated by H2D. H2D wait is about 0.42 ms/layer and scatter is about 0.20 ms/layer for this decode run.
- The pageable path adds about 2.37 ms/layer for pageable static mirror to pinned runtime stage copy. This is the main extra cost of the last 11 layers.
- Effective pageable-to-pinned bandwidth in this run was about 14.95 GiB/s, using staged H2D bytes as the comparable payload size.
- Effective H2D bandwidth from the timing fields was about 98.8 GiB/s for pinned-source rows and 128.5 GiB/s for pageable-source rows. These are diagnostic effective rates around the current staged payloads and timing boundaries.

## Example Layers

| Layer | Source | Avg staged experts | pageable -> pinned | H2D wait | GPU scatter | Prepare total |
|---:|---|---:|---:|---:|---:|---:|
| 0 | pinned | 13.48 | 0.050 ms | 0.666 ms | 0.278 ms | 3.824 ms |
| 36 | pinned | 7.24 | 0.028 ms | 0.305 ms | 0.175 ms | 2.140 ms |
| 37 | pageable | 7.48 | 2.328 ms | 0.295 ms | 0.193 ms | 4.676 ms |
| 47 | pageable | 9.85 | 3.003 ms | 0.373 ms | 0.258 ms | 6.212 ms |

## Hypothetical Pinned 44G -> 59G

Assumption: increasing the physical memory / stable pinned allocation budget lets all 48 MoE layers live in pinned static mirror. Under that assumption, layers 37..47 would avoid the pageable-to-pinned stage copy and keep roughly the same H2D and scatter costs.

Observed over steady decode:

- pageable-to-pinned total cost in layers 37..47: 862.0 ms over 63 steady tokens.
- estimated residual direct-pinned copy overhead for those same staged experts: 10.8 ms.
- estimated saved time: 851.2 ms over 63 steady tokens, or about 13.5 ms/token.
- synchronized steady TPS estimate: 6.064 tok/s -> 6.605 tok/s, about +8.9%.

Conclusion for this decode workload: moving from 44G pinned to full-MoE pinned is useful but not transformative. It mainly removes the pageable-to-pinned copy from the last 11 layers. The expected gain is roughly 9% in this synchronized 64-token decode diagnostic. It may be more valuable for long prefill/burst workloads that repeatedly touch many unique experts in layers 37..47, but for decode the cost-benefit is moderate rather than decisive.
