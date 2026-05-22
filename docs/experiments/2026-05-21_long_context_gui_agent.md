# 2026-05-21 Long Context GUI Agent Sweep

## Scope

This file separates two benchmark scopes that should not be mixed:

- Completed baseline sweep: over-provisioned KV capacity with 20K/30K/40K/50K prefill.
- Next GUI Agent sweep: KV capacity rises with the actual video-frame prefill length.

The GUI Agent workload is different from a first-chat-only benchmark. Each new conversation turn can include a long video-frame prefill on top of an already large KV allocation, so the useful limit is the largest prompt that can still prefill and decode without hitting runtime activation, burst-stage, or H2D peak-memory OOM.

## Common Config Used So Far

- Model: `/home/jiamin/models/Qwen3.5-122B-A10B-GPTQ-Int4/snapshots/5b9f0050d3ec98b0c81a7716776533c5eacebb64`
- Runtime: WSL, no CUDA graph, eager mode.
- Speculation: disabled for this sweep (`--spec-method none`).
- `gpu_slots_per_layer=16`
- `prefill_burst_slots=256`
- `prepare_cpu_copy_threads=8`
- `cpu_static_pinned_gb=44`
- `marlin_input_dtype=fp8`
- `turns=2`
- `warm_prompt_mode=rotated`
- `isolate_prompt_lengths=true`
- Text check: generated text is non-empty and does not contain the Unicode replacement character.

Raw logs:

```text
/home/jiamin/projects/CFIE/.bench_logs/20260521_long_context_sweep_probe
```

## Completed Baseline Sweep

These runs answer: "With a large enough KV allocation already reserved, how do 4096 vs 8192 prefill chunks behave for 20K-50K prompts?"

They do not answer the final GUI Agent limit, because KV was over-provisioned relative to prompt length.

| Run | Max Len | KV | Max Batched | Prefill | Cold TTFT | Warm TTFT | Cold Decode TPS | Warm Decode TPS | Output OK | Peak GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| ctx184320_kv5g_mb4096 | 184320 | 5.00 GiB | 4096 | 20000 | 31.760s | 18.577s | 5.596 | 5.512 | yes | 29513 MiB |
| ctx184320_kv5g_mb4096 | 184320 | 5.00 GiB | 4096 | 30000 | 29.363s | 27.437s | 5.606 | 5.319 | yes | 29513 MiB |
| ctx184320_kv5g_mb4096 | 184320 | 5.00 GiB | 4096 | 40000 | 38.729s | 36.703s | 5.613 | 5.441 | yes | 29513 MiB |
| ctx184320_kv5g_mb4096 | 184320 | 5.00 GiB | 4096 | 50000 | 47.696s | 45.292s | 5.620 | 5.502 | yes | 29513 MiB |
| ctx184320_kv5g_mb8192 | 184320 | 5.00 GiB | 8192 | 20000 | 26.425s | 11.166s | 5.459 | 5.264 | yes | 31019 MiB |
| ctx184320_kv5g_mb8192 | 184320 | 5.00 GiB | 8192 | 30000 | 15.430s | 14.793s | 5.594 | 5.451 | yes | 31019 MiB |
| ctx184320_kv5g_mb8192 | 184320 | 5.00 GiB | 8192 | 40000 | 20.451s | 19.646s | 5.610 | 5.442 | yes | 31019 MiB |
| ctx184320_kv5g_mb8192 | 184320 | 5.00 GiB | 8192 | 50000 | 24.516s | 23.778s | 5.605 | 5.445 | yes | 31019 MiB |
| ctx222176_kv6g_mb4096 | 222176 | 6.00 GiB | 4096 | 20000 | 33.261s | 19.029s | 5.418 | 4.974 | yes | 30640 MiB |
| ctx222176_kv6g_mb4096 | 222176 | 6.00 GiB | 4096 | 30000 | 30.046s | 27.963s | 5.496 | 5.340 | yes | 30640 MiB |
| ctx222176_kv6g_mb4096 | 222176 | 6.00 GiB | 4096 | 40000 | 39.393s | 37.139s | 5.492 | 5.322 | yes | 30640 MiB |
| ctx222176_kv6g_mb4096 | 222176 | 6.00 GiB | 4096 | 50000 | 48.543s | 46.135s | 5.527 | 5.408 | yes | 30640 MiB |
| ctx222176_kv6g_mb8192 | 222176 | 6.00 GiB | 8192 | 20000 | 27.417s | 12.619s | 5.380 | 5.334 | yes | 31889 MiB |
| ctx222176_kv6g_mb8192 | 222176 | 6.00 GiB | 8192 | 30000 | 20.807s | 21.100s | 5.386 | 5.222 | yes | 31889 MiB |
| ctx222176_kv6g_mb8192 | 222176 | 6.00 GiB | 8192 | 40000 | 32.217s | 30.125s | 5.362 | 5.234 | yes | 31889 MiB |
| ctx222176_kv6g_mb8192 | 222176 | 6.00 GiB | 8192 | 50000 | 40.316s | 37.082s | 5.372 | 5.239 | yes | 31889 MiB |

## Completed Baseline Findings

- `max_num_batched_tokens=8192` materially improves prefill TTFT versus 4096 for 20K-50K prompts.
- Decode throughput is mostly insensitive to 4096 vs 8192 and stays around 5.2-5.6 tok/s in this no-MTP/no-graph setup.
- 222K + 6 GiB KV + 8192 is very close to the 32 GiB card limit: peak was 31889 MiB.
- All completed baseline rows produced non-empty text without replacement characters.

## GUI Agent Sweep Standard

Use this standard for the next phase:

- Keep `max_num_batched_tokens=8192`.
- Enable MTP with one speculative token: `--spec-method mtp --num-speculative-tokens 1`.
- Increase prompt lengths from 60K upward: 60K, 70K, 80K, ...
- Configure KV to approximately match the tested prompt length plus decode headroom, not a fixed 200K/256K budget.
- Stop at first OOM, then bisect or test smaller increments around the boundary.
- Record:
  - cold TTFT;
  - warm next-turn TTFT with rotated prompt;
  - decode TPS after first output token;
  - peak GPU memory;
  - output normality;
  - KV estimated single-request max context from startup log;
  - OOM phase if failed.

Initial screening will use shorter decode to find the memory boundary faster. Boundary candidates should then be rerun with 512 decode tokens.

Aborted run:

- `gui_prefill60k_kv2g_max65536_mb8192_new256_none` was started with `--spec-method none` before the MTP requirement was restated.
- It was terminated during startup and should not be used for decisions.

## GUI Agent MTP=1 Sweep

Run directory:

```text
/home/jiamin/projects/CFIE/.bench_logs/20260521_gui_agent_sync_kv_mtp1
```

Standard run shape:

```powershell
--max-num-batched-tokens 8192
--gpu-slots-per-layer 16
--prefill-burst-slots 256
--prepare-cpu-copy-threads 8
--cpu-static-pinned-gb 44
--marlin-input-dtype fp8
--spec-method mtp
--num-speculative-tokens 1
--enforce-eager
--turns 2
--warm-prompt-mode rotated
--max-new-tokens 256
```

Successful rows:

| Prefill | Max Len | KV | Cold TTFT | Warm TTFT | Cold Decode TPS | Warm Decode TPS | Peak GPU | Output OK |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 60K | 65536 | 2.50 GiB | 75.492s | 51.259s | 5.008 | 5.156 | 31900 MiB | yes |
| 70K | 73728 | 3.00 GiB | 93.922s | 34.307s | 5.213 | 5.519 | 31882 MiB | yes |
| 80K | 81920 | 3.00 GiB | 59.467s | 42.172s | 5.486 | 5.475 | 31893 MiB | yes |
| 90K | 98304 | 3.00 GiB | 60.918s | 43.265s | 5.414 | 5.463 | 31895 MiB | yes |
| 100K | 106496 | 3.50 GiB | 71.449s | 50.692s | 5.338 | 5.218 | 31907 MiB | yes |
| 110K | 114688 | 3.50 GiB | 82.710s | 60.764s | 5.085 | 5.087 | 31912 MiB | yes |
| 120K | 122880 | 4.00 GiB | 98.539s | 77.306s | 4.983 | 4.893 | 31821 MiB | yes |
| 130K | 131072 | 4.00 GiB | 155.763s | 136.700s | 2.501 | 2.450 | 31845 MiB | yes |
| 135K | 135256 | 4.00 GiB | 95.015s | 76.276s | 4.798 | 4.841 | 31899 MiB | yes |
| 138K | 138256 | 4.00 GiB | 113.596s | 84.694s | 4.689 | 4.776 | 31910 MiB | yes |

Failures and boundary probes:

| Run | Result | Root Cause |
|---|---|---|
| 60K / 2.00 GiB KV / max_len 65536 | failed at init | KV check required 2.13 GiB; 2.00 GiB only estimated max length 60784. |
| 90K / max_len 90112 | failed before engine | Prompt plus 256 decode tokens exceeded max_model_len. |
| 139K / 4.00 GiB KV / max_len 139256 | terminated manually | No first token after more than one hour with GPU at ~99%; treated as pathological/unusable for GUI Agent. |
| 140K / 4.00 GiB KV / max_len 140256 | failed during prefill | CUDA OOM inside tiered MoE burst stage while writing runtime-ready bundles after most of the prompt had been computed. Peak 31958 MiB. |
| 140K / 4.50 GiB KV / max_len 147456 | failed during prefill | CUDA OOM in `gdn_attention_core` / `chunk_gated_delta_rule`, tried to allocate another 100 MiB with no free VRAM. |

KV estimates observed in logs:

| KV | Estimated Single-Request Context |
|---:|---:|
| 2.50 GiB | 81744 tokens |
| 3.00 GiB | 100608 tokens |
| 3.50 GiB | 121568 tokens |
| 4.00 GiB | 140432 tokens |
| 4.50 GiB | 161392 tokens |

## Current Interpretation

- Under the requested standard config (`MTP=1`, `max_num_batched_tokens=8192`, `fp8`, `gpu_slots_per_layer=16`, `prefill_burst_slots=256`), 138K prefill plus 256 decode is the largest completed run so far.
- 139K is not usable in practice because it did not produce a first token after more than one hour.
- 140K is a hard failure in the current implementation: one run OOMed in the tiered MoE burst write path, and the larger-KV retry OOMed in the linear-attention/GDN path.
- The practical GUI Agent recommendation from this sweep is:
  - conservative: `max_model_len=135256`, `kv_cache_memory_bytes=4294967296`;
  - capacity-first: `max_model_len=138256`, `kv_cache_memory_bytes=4294967296`, with very little memory headroom.
- All successful rows passed the current text normality check: generated text was non-empty and did not contain replacement characters.
- Peak memory is consistently near the card limit. Any extra graph capture, larger decode, larger burst workspace, or additional visual encoder memory can invalidate the 138K setting.

## 128000 Auto-KV Recommendation Check

User-facing target:

- Use an exact 128000-token context tier: `max_model_len=128000`.
- Do not pass `kv_cache_memory_bytes` only after the auto-KV allocator is fixed.
- Until then, explicitly reserve the tight manual KV budget requested for this tier: `kv_cache_memory_bytes=4000000000`.

Validation run directory:

```text
/home/jiamin/projects/CFIE/.bench_logs/20260521_gui_agent_auto_kv_mtp1
```

Runs:

| Run | Result | Root Cause |
|---|---|---|
| 128000 / auto KV / `gpu_memory_utilization=0.88` | failed at init | Auto KV only exposed 2.02 GiB, but 128000 needs 3.69 GiB. |
| 128K / auto KV / `gpu_memory_utilization=0.95` | failed at init | Startup free memory was 30.2 GiB, while static budget at 0.95 required 30.25 GiB. |
| 128K / auto KV / `gpu_memory_utilization=0.94` | failed at init | Static budget 29.93 GiB plus profiled runtime peak 0.33 GiB exceeded startup free memory 30.2 GiB. |

Interpretation:

- The exact 128000 tier itself is reasonable and safer than the 135K/138K boundary results.
- Current auto-KV logic is too conservative for this tiered MoE setup. It computes KV from `static_memory_budget - steady_non_kv_memory`, then can only borrow runtime headroom up to the startup free-memory guard.
- Manual `kv_cache_memory_bytes` succeeds because it bypasses this profiling-derived auto-KV limit.
- The selected tight manual setting is `4000000000` bytes, not `4294967296`, because the 128000-token log requirement is about 3.69 GiB and the extra 0.29 GiB from 4 GiB is not useful.
- Do not use `4000000000` together with `max_model_len=131072` unless it is separately verified; prior logs estimated 131072 closer to 3.74 GiB and this setting may be too tight.
- Therefore, the current runnable recommendation is:

```bash
--max-model-len 128000
--kv-cache-memory-bytes 4000000000
```

- The desired future recommendation remains:

```bash
--max-model-len 128000
# no explicit kv-cache-memory-bytes
```

but that requires changing the auto-KV allocator for tiered MoE so it can reserve the required 128K KV budget without tripping the current static-budget guard.

## 128000 Manual-KV Startup Smoke

Run directory:

```text
/home/jiamin/projects/CFIE/.bench_logs/20260521_manual_kv_128000
```

Command shape:

```bash
--max-model-len 128000
--max-num-batched-tokens 8192
--kv-cache-memory-bytes 4000000000
--gpu-slots-per-layer 16
--prefill-burst-slots 256
--prepare-cpu-copy-threads 8
--cpu-static-pinned-gb 44
--marlin-input-dtype fp8
--spec-method mtp
--num-speculative-tokens 1
--enforce-eager
--target-input-tokens 1024
--max-new-tokens 64
```

Result:

| Run | Result | Startup / KV | TTFT | Decode TPS After First Token | Peak GPU | Output OK |
|---|---|---|---:|---:|---:|---|
| `manualkv_max128000_kv4000000000_input1024_new64_mtp1` | success, exit code 0 | `Initial free memory=30.2 GiB`, manual KV reserved `3.73 GiB`; estimated single-request max context `129,952` tokens with `3.69 GiB` KV tensors | 29.067s | 5.743 tok/s | 31840 MiB | yes |

Notes:

- This verifies that the tight manual KV setting is sufficient for exact `max_model_len=128000`.
- The run was a short-input startup/inference smoke test, not a 128000-token prefill throughput test.
- Generated text was non-empty and contained no replacement characters.

## Current Standard CLI

This is the current standard baseline for later GUI Agent and long-context tests. It intentionally uses exact `128000`, not `131072`.

Windows benchmark command shape for later non-graph tests:

```bash
..\.venv\Scripts\python.exe benchmarks\run_long_prefill_decode.py `
  --model D:\models\Qwen3.5-122B-A10B-GPTQ-Int4\snapshots\5b9f0050d3ec98b0c81a7716776533c5eacebb64 `
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
  --enforce-eager
```

OpenAI API server command shape for text-only Responses smoke tests:

```powershell
..\.venv\Scripts\python.exe -m cfie.entrypoints.openai.api_server `
  --model D:\models\Qwen3.5-122B-A10B-GPTQ-Int4\snapshots\5b9f0050d3ec98b0c81a7716776533c5eacebb64 `
  --host 127.0.0.1 `
  --port 8000 `
  --max-model-len 128000 `
  --max-num-seqs 1 `
  --max-num-batched-tokens 8192 `
  --kv-cache-memory-bytes 4000000000 `
  --gpu-slots-per-layer 16 `
  --prefill-burst-slots 256 `
  --prepare-cpu-copy-threads 8 `
  --cpu-static-pinned-gb 40 `
  --enable-prefix-caching `
  --language-model-only `
  --skip-mm-profiling `
  --marlin-input-dtype fp8 `
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' `
  --enforce-eager
```

Benchmark-specific fields such as `--target-input-tokens`, `--max-new-tokens`, `--turns`, and `--warm-prompt-mode` should be set per experiment.

Current rationale:

- `--kv-cache-memory-bytes 4000000000` reserves about 3.73 GiB, while the 128000-token log requirement is about 3.69 GiB.
- `--max-model-len 128000` passed startup and short inference, with estimated single-request max context `129,952` tokens.
- `--max-num-seqs 1` must be explicit for the GUI Agent single-session baseline. API server defaults can otherwise reserve/estimate KV for many concurrent sequences and make the MoE planner reject startup.
- `--enable-prefix-caching` is mandatory for GUI Agent tests and should be passed explicitly rather than relying on model defaults.
- For OpenAI/VL serving, prefer `--cpu-static-pinned-gb 40` until the runtime-stage pinned allocator is improved. `44` worked in language-only benchmark runs but failed one VL API request by leaving too little pinned headroom for the 256-slot runtime expert stage.
- CUDA graph remains disabled in the standard baseline because prior PIECE graph/MTP runs were not yet a stable speed win.
- The peak whole-GPU memory in the smoke run was `31840 MiB`; subtracting the idle baseline of about `922 MiB`, the engine added about `30918 MiB`.

## OpenAI Responses / GUI Agent API Status

- `/v1/responses` is already implemented and registered by the OpenAI-compatible server.
- Windows venv has the API-server dependencies needed for FastAPI/uvicorn startup.
- Fixed a Windows API-server startup bug: `cfie/entrypoints/openai/api_server.py` used `sys.platform` without importing `sys`.
- Standard Responses text input validates as `input: "..."`.
- Standard Responses image input validates with `input_image` parts:

```json
{
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Describe this GUI."},
        {"type": "input_image", "image_url": "data:image/png;base64,...", "detail": "auto"}
      ]
    }
  ]
}
```

- Multiple `input_image` parts also validate, so GUI video history should be represented as ordered screenshot frames in Responses.
- `video_url` does not validate through the current Responses request type from `openai-python`; do not use it as the standard GUI Agent path unless the protocol layer is intentionally extended.
- For VL/OpenAI Responses tests, remove `--language-model-only` and `--skip-mm-profiling`, and set an appropriate `--limit-mm-per-prompt`.

New helper scripts:

- `benchmarks/run_openai_responses_gui_smoke.py`: sends text, single GUI screenshot, and multi-frame GUI screenshot requests to `/v1/responses`.
- `benchmarks/estimate_qwen35_vl_frame_capacity.py`: estimates how many `input_image` frames fit into a 128000-token context for common GUI resolutions.

## Windows OpenAI Responses Smoke, VL Enabled

Run directory:

```text
.bench_logs/20260521_responses_gui/vl_server_8192_mtp_maxseq1_pinned40
```

Command differences from the 128K production baseline:

- `max_model_len=8192`
- `kv_cache_memory_bytes=1200000000`
- `cpu_static_pinned_gb=40`
- VL enabled, so no `--language-model-only` and no `--skip-mm-profiling`
- `--limit-mm-per-prompt '{"image":8,"video":0}'`

Startup observations:

| Item | Value |
|---|---:|
| Server ready wall time | 210.9s |
| Main model weight load | 35.93s |
| Expert cache attach, from post-weight load to runtime-stage ready | about 97s |
| MTP drafter weight load | 21.23s |
| Engine init / KV / warmup | 1.72s |
| Model loading memory log | 22.97 GiB |
| Manual KV reservation | 1.12 GiB |
| Estimated single-request context | 25,152 tokens |

The server log confirmed the important standard flags:

```text
enable_prefix_caching=True
max_num_seqs=1
gpu_slots_per_layer=16
prefill_burst_slots=256
cpu_static_pinned_gb=40.0
marlin_input_dtype='fp8'
speculative_config={'method': 'mtp', 'num_speculative_tokens': 1}
```

Responses smoke result:

| Case | Status | Input Tokens | Output Tokens | Seconds | Output OK |
|---|---:|---:|---:|---:|---|
| text | 200 | 21 | 96 | 21.13s | yes |
| one GUI image, 960x540 | 200 | 550 | 96 | 31.35s | yes |
| six GUI frames, 960x540 | 200 | 3,119 | 96 | 36.16s | yes |

The single-frame and multi-frame cases both identified the blue `Submit` button. The coordinate-like output was in the Qwen-VL normalized-coordinate style rather than direct screen pixels, so a GUI Agent caller should either request normalized coordinates explicitly or convert the model's coordinate convention before moving the cursor.

Previous failed attempt:

- `cpu_static_pinned_gb=44` started successfully but the first request failed with `Failed to allocate pinned runtime expert stage: capacity=256 bytes=1192.50 MiB`.
- Interpretation: static pinned expert mirror plus shared runtime CPU stage left too little pinned-memory headroom. Reducing the static pinned budget to `40` allowed the same Responses smoke to pass.

## 128K Multi-Frame Image Capacity Estimate

Run:

```powershell
..\.venv\Scripts\python.exe benchmarks\estimate_qwen35_vl_frame_capacity.py `
  --model D:\models\Qwen3.5-122B-A10B-GPTQ-Int4\snapshots\5b9f0050d3ec98b0c81a7716776533c5eacebb64 `
  --max-context-tokens 128000 `
  --reserve-output-tokens 1024 `
  --output-json .bench_logs\20260521_responses_gui\vl_frame_capacity_128k.json `
  --output-md .bench_logs\20260521_responses_gui\vl_frame_capacity_128k.md
```

Results are processor/tokenizer counts for standard Responses `input_image` frames:

| Resolution | Tokens / Frame | Estimated Max Frames | 1 Frame Tokens | 2 Frame Tokens |
|---|---:|---:|---:|---:|
| 384x216 | 86 | 1476 | 101 | 187 |
| 512x288 | 146 | 869 | 161 | 307 |
| 640x360 | 222 | 571 | 237 | 459 |
| 800x450 | 352 | 360 | 367 | 719 |
| 960x540 | 512 | 247 | 527 | 1039 |
| 1280x720 | 882 | 143 | 897 | 1779 |
| 1600x900 | 1402 | 90 | 1417 | 2819 |
| 1920x1080 | 2042 | 62 | 2057 | 4099 |
| 2560x1440 | 3602 | 35 | 3617 | 7219 |

Interpretation:

- 50 full-HD desktop frames at 1920x1080 are about `102K` image tokens before text and output reserve, so they fit under a 128K context budget in token accounting.
- 50 frames at 1280x720 are about `44K` image tokens and are much safer for GUI Agent history.
- 2560x1440 desktop frames are too expensive for a 50-frame history under 128K; use downsampling or crop regions of interest.
- These numbers are capacity estimates only; actual VL prefill runtime and peak memory still require model execution tests.
