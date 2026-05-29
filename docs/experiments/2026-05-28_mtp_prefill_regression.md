# 2026-05-28 MTP Prefill Regression Diagnosis

## Scope

GUI Agent live runs showed abnormal per-step latency for image requests. The first
suspect was visual input size or Responses protocol packaging, because recent
turns included 1920x1080 screenshots. This note records the diagnosis.

## Result

The slowdown was caused by the MTP speculative path during long prefill. It was
not caused by image preprocessing, base64 cleanup, Responses JSON handling, or
chat template rendering.

The fix is now in place:

- MTP drafter no longer expands to full 256-expert GPU residency when the target
  model is also loaded.
- MTP drafter no longer inherits the target model's 256-slot prefill burst pool.
- MTP drafter no longer inherits the target model's large pinned static CPU
  budget.
- The GPU runner skips drafter execution for prompt/chunked-prefill steps and
  only proposes draft tokens for decode-shaped batches.

## Evidence

Same Windows machine, same Qwen3.5-122B-A10B-GPTQ-Int4 model, same tiered-MoE
settings:

```text
gpu_slots_per_layer=16
prefill_burst_slots=256
prepare_cpu_copy_threads=32
cpu_static_pinned_gb=40-44
kv_cache_memory_bytes=4000000000
marlin_input_dtype=fp8
enable_prefix_caching=true
enforce_eager=true
```

### Offline Engine

| Run | MTP | Input Tokens | TTFT |
| --- | --- | ---: | ---: |
| `runs/diagnostics/20260528_inproc_prefill2096_no_mtp/result.json` | off | 2096 | 2.820s |
| `runs/diagnostics/20260528_inproc_prefill_timing/result.json` | on, 1 token | 2096 | 44.281s |
| `runs/diagnostics/20260528_multiproc_prefill_timing/result.json` | on, 1 token | 2096 | 44.583s |
| `runs/diagnostics/20260528_mtp_after_prefill_skip_fix/result.json` | on, fixed | 2096 | 3.326s |

The in-process and multiprocessing MTP results were both slow, so the main issue
was not Windows multiprocessing overhead. After the fix, MTP prefill latency is
back near the no-MTP baseline.

### Root Cause Detail

Before the fix, the actual MTP drafter plan logged:

```text
Loading speculative drafter ... gpu_slots/layer=256 prefill_burst_slots=0 cpu_slots/layer=0
Model loading took 22.12 GiB
CFIE_MTP_TIMING target_forward seconds=42.876673 tokens=2096 spec=True
CFIE_MTP_TIMING propose_draft seconds=1.733360 scheduled_tokens=2096
```

This showed that the drafter itself was not the 40s bottleneck. The full-resident
unquantized drafter pushed the target model into GPU memory pressure, making
target forward kernels extremely slow.

After the fix, the actual MTP drafter plan logs:

```text
Loading speculative drafter ... gpu_slots/layer=16 prefill_burst_slots=0 cpu_slots/layer=256
Model loading took 18.04 GiB
CFIE_MTP_TIMING target_forward seconds=3.295569 tokens=2096 spec=True
CFIE_MTP_TIMING propose_draft skipped=true scheduled_tokens=2096 max_scheduled_tokens=2096
```

The 64-token decode validation run also succeeded:

```text
runs/diagnostics/20260528_mtp_decode64_after_prefill_skip_fix/result.json
prefill_to_first_token_seconds=3.668s
decode_tokens_per_sec_after_first_step=7.048 tok/s
decode target forward avg=0.230s per 2-token speculative step
decode drafter propose avg=0.023s
```

### OpenAI / VL Service, No MTP

Service command used the same model and OpenAI/VL path, but removed
`--speculative-config`.

| Request | Input Tokens | Latency |
| --- | ---: | ---: |
| Text request around 2K tokens | 2096 | 5.406s |
| Single 1920x1080 image request | 2061 | 5.993s |

Earlier MTP-enabled API checks produced much larger values:

| Request | Input Tokens | Latency |
| --- | ---: | ---: |
| Text request around 2K tokens | about 2100 | 43-47s |
| 1920x1080 image request | about 2058 | about 90s |

Render-only checks were below 0.1s for image payloads, so protocol/template work
is not the bottleneck.

## Conclusion

MTP is now safe from the previous long-prefill regression because prompt chunks
skip drafter execution and the drafter no longer consumes full expert residency.
For GUI Agent and VL serving, MTP should still be treated as an optional decode
experiment rather than a default parameter until it shows a clear steady decode
throughput gain over no-MTP under the same live workload.

Recommended GUI Agent service defaults after this diagnosis:

```text
--max-num-batched-tokens 2096
--enable-prefix-caching
--marlin-input-dtype fp8
--enforce-eager
```

The user-facing interpretation is important: previous image inference was not
inherently this slow. The pathological latency came from the speculative MTP
prefill route.
