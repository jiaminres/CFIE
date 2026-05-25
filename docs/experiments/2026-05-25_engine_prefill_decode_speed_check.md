# 2026-05-25 Engine Prefill / Decode Speed Check

## Scope

This entry checks the current 122B engine speed against the historical target:

- prefill around 8192 tokens should be in the low single-digit seconds when warm/prefix-cached;
- decode should be around 8 tokens/s.

The tested model is:

```text
D:\models\Qwen3.5-122B-A10B-GPTQ-Int4\snapshots\5b9f0050d3ec98b0c81a7716776533c5eacebb64
```

## Standard Language-Only CLI Shape

```powershell
..\.venv\Scripts\python.exe benchmarks\run_long_prefill_decode.py `
  --model D:\models\Qwen3.5-122B-A10B-GPTQ-Int4\snapshots\5b9f0050d3ec98b0c81a7716776533c5eacebb64 `
  --target-input-tokens 2096 `
  --max-new-tokens 512 `
  --turns 2 `
  --warm-prompt-mode rotated `
  --max-model-len 128000 `
  --max-num-seqs 1 `
  --max-num-batched-tokens 2096 `
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
  --enforce-eager `
  --temperature 0
```

## Windows Results

Environment:

```text
python: C:\Users\13642\PycharmProjects\vllm\.venv\Scripts\python.exe
torch: 2.10.0+cu130
triton: not installed
```

| Run | Prompt | New Tokens | Warm Mode | Cold TTFT | Warm TTFT | Decode TPS After First Token | Result |
|---|---:|---:|---|---:|---:|---:|---|
| `.bench_logs\20260525_engine_offline_prefill8192_decode512_standard.json` | 8192 | 512 | rotated | 36.049s | 35.076s | 8.93 tok/s | decode OK, prefill slow |
| `.bench_logs\20260525_engine_offline_prefill8192_new1_sameprefix_standard.json` | 8192 | 1 | same | 38.252s | 18.464s | n/a | prefix cache helps, but still slow |
| `.bench_logs\20260525_engine_offline_prefill5k_new1_sameprefix_legacy8192.json` | 5000 | 1 | same | 24.903s | 13.667s | n/a | slower than historical WSL result |
| `.bench_logs\20260525_engine_api_prefill8192_decode512.json` | about 7681 API prompt tokens | 512 | n/a | 42.44s | n/a | 4.72 tok/s | API/VL service path slower |

The language-only offline decode path still reaches the decode target: about `8.9 tok/s`.

The original Windows prefill path did not reach the historical warm TTFT target.

## Windows GDN Fast Path Fix

Root cause found after adding layer-level timing:

- 2096-token prefill spent about `8.6s` inside 36 Qwen3.5 linear-attention GDN recurrent calls.
- The Windows precompiled fallback was still using the slow ATen token loop for the long prefill path.
- Two fast-path compatibility issues were fixed:
  - `g` from fused GDN gating is `float32`, while q/k/v are `bfloat16`; the CUDA recurrent fast path now accepts `g=float32`.
  - real scheduler `cu_seqlens/query_start_loc` can arrive as CUDA int32; the C++ op now converts it to CUDA int64 before testing the fast path.

Validation:

| Run | Prompt | Max Batched Tokens | Timing Mode | TTFT | Key Timing |
|---|---:|---:|---|---:|---|
| `.bench_logs\20260525_engine_windows_prefill2096_core_timing_after_gdn_fast.json` | 2096 | 2096 | on | 11.100s | recurrent `8.46s` total |
| `.bench_logs\20260525_engine_windows_prefill2096_after_cu_dtype_fix.json` | 2096 | 2096 | on | 3.917s | recurrent `1.16s` total |
| `.bench_logs\20260525_engine_windows_prefill_multiples_2096_after_gdn_fast_notiming.json` | 2096 | 2096 | off | 4.058s | output normal |
| `.bench_logs\20260525_engine_windows_prefill_multiples_2096_after_gdn_fast_notiming.json` | 4192 | 2096 | off | 6.610s | output normal |
| `.bench_logs\20260525_engine_windows_prefill_multiples_2096_after_gdn_fast_notiming.json` | 6288 | 2096 | off | 9.843s | output normal |
| `.bench_logs\20260525_engine_windows_prefill_multiples_2096_after_gdn_fast_notiming.json` | 8384 | 2096 | off | 12.971s | output normal |
| `.bench_logs\20260525_engine_windows_prefill2096_decode512_after_gdn_fast.json` | 2096 + 512 decode | 2096 | off | 4.208s | decode `13.26 tok/s`, output normal |

The 2096-token aligned Windows prefill target is now met. Longer prompts are still chunked by `max_num_batched_tokens=2096`, so TTFT scales with the number of chunks plus MoE burst cost.

## WSL Cross-Check

Environment:

```text
python: /home/jiamin/projects/CFIE/.wsl-venv/bin/python
torch: 2.10.0+cu130
triton: 3.6.0
```

Command used the same logical settings as above, with WSL paths and `--prepare-cpu-copy-batch-size 8` because the WSL copy of the benchmark still uses the older option name.

Result:

| Run | Prompt | New Tokens | Warm Mode | Cold TTFT | Warm TTFT | Result |
|---|---:|---:|---|---:|---:|---|
| `/home/jiamin/projects/CFIE/.bench_logs/20260525_engine_wsl_prefill8192_new1_sameprefix_standard.json` | 8192 | 1 | same | 22.158s | 1.933s | warm prefill target met |

## Interpretation

The CLI settings are not the primary cause of the Windows prefill slowdown:

- `effective_max_num_batched_tokens=8192`
- `effective_max_num_scheduled_tokens=8192`
- `enable_prefix_caching=True`
- `prefill_burst_slots=256`
- `gpu_slots_per_layer=16`
- `marlin_input_dtype=fp8`
- `MTP=1`

The major difference is runtime kernel availability. Windows logs show:

```text
Triton not installed or not compatible
Qwen3Next fused GDN gating is falling back to the PyTorch reference path because Triton runtime is unavailable
Mamba SSD / causal conv / chunk scan paths can fall back when Triton is unavailable
```

Qwen3.5 has many linear-attention / Mamba-style layers, so prefill depends heavily on these kernels. Decode can still meet the target because the decode path is much less dominated by long prefill Mamba/GDN kernels.

## Current Recommendation

For Windows GUI Agent tests, prefer `--max-num-batched-tokens 2096` while using the current C++/CUDA GDN recurrent fast path. This keeps each chunk aligned with the Qwen3.5 Mamba cache block behavior and avoids the old slow ATen recurrent fallback.

For Windows language-only decode checks, the current standard CLI remains acceptable and reaches about `8.9 tok/s`.

For OpenAI/VL service checks, benchmark separately. The API/VL path measured here is slower than the language-only engine benchmark and should not be used to judge the optimized decode baseline without separate investigation.
