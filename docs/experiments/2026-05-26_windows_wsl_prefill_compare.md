# 2026-05-26 Windows / WSL Prefill Compare

## Scope

Compare current Windows and WSL prefill latency under the same logical 122B engine
settings. This run is text-only and graph-disabled; it is meant to answer whether
Windows and WSL have a large prefill-speed gap under the GUI Agent baseline.

## Shared Settings

- Model: Qwen3.5-122B-A10B-GPTQ-Int4, Marlin-ready cache path on each OS.
- `max_model_len=128000`
- `max_num_seqs=1`
- `max_num_batched_tokens=8192`
- `kv_cache_memory_bytes=4000000000`
- `gpu_slots_per_layer=16`
- `prefill_burst_slots=256`
- `prepare_cpu_copy_threads=32`
- `cpu_static_pinned_gb=40`
- `enable_prefix_caching=True`
- `language_model_only=True`
- `skip_mm_profiling=True`
- `marlin_input_dtype=fp8`
- `spec_method=mtp`
- `num_speculative_tokens=1`
- `enforce_eager=True`
- `max_new_tokens=1`
- `turns=2`
- `warm_prompt_mode=rotated`
- `isolate_prompt_lengths=True`

The warm turn uses a rotated prompt with the same token count, so it warms the
engine but avoids an exact full-prefix-cache hit.

## Result Files

- Windows: `.bench_logs/20260526_prefill_win_wsl_compare/windows_prefill_2096_8192_mb8192.json`
- WSL copy: `.bench_logs/20260526_prefill_win_wsl_compare/wsl_prefill_2096_8192_mb8192.json`
- WSL source: `/home/jiamin/projects/CFIE/.bench_logs/20260526_prefill_win_wsl_compare/wsl_prefill_2096_8192_mb8192.json`

## Results

| Prompt tokens | Windows cold TTFT | Windows warm TTFT | WSL cold TTFT | WSL warm TTFT | Windows / WSL warm | Windows warm tok/s | WSL warm tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2096 | 4.151s | 3.456s | 20.952s | 3.311s | 1.04x | 606.5 | 633.1 |
| 8192 | 9.635s | 9.480s | 6.572s | 5.754s | 1.65x | 864.2 | 1423.6 |

## Interpretation

- At the 2096-token low-latency target, Windows and WSL are effectively close:
  Windows warm TTFT is only about 4% slower than WSL.
- At the 8192-token chunk size, WSL is materially faster: Windows warm TTFT is
  about 1.65x WSL. This is a real gap, but not the earlier several-times
  slowdown seen before the Windows GDN fast path fixes.
- The WSL first 2096 run is not representative because it includes additional
  cold-start behavior inside the first engine request. The second 2096 run is
  the useful warm-engine prefill comparison.
- Both runs report `effective_max_num_batched_tokens=8192` and
  `effective_max_num_scheduled_tokens=8192`.
- This test does not include visual encoder/image-token preprocessing. GUI Agent
  turns with screenshots can still be slower if each step adds large new image
  inputs and breaks prefix-cache locality.

## Windows Bottleneck Check

Additional Windows-side checks were run to answer whether the 8192-token gap is
caused by a missing compiled operator.

### No-MTP A/B

Files:

- Windows no-MTP: `.bench_logs/20260526_prefill_win_wsl_compare/windows_prefill_8192_no_mtp.json`
- WSL no-MTP: `.bench_logs/20260526_prefill_win_wsl_compare/wsl_prefill_8192_no_mtp.json`

| Runtime | Spec decode | Cold TTFT | Warm TTFT |
|---|---|---:|---:|
| Windows | none | 9.342s | 8.563s |
| WSL | none | 21.364s | 5.489s |

Removing MTP improves Windows 8192 warm TTFT from 9.480s to 8.563s, so the MTP
drafter fallback is a real cost, but it is not the main Windows/WSL prefill gap.
The same no-MTP comparison still leaves Windows about 3.07s slower.

### Operator Availability

Both Windows and WSL report these precompiled custom ops as present:

- `_C.chunk_gated_delta_rule_precompiled`
- `_C.chunk_gated_delta_rule_fwd_h_precompiled`
- `_C.fused_recurrent_gated_delta_rule_packed_decode_precompiled`
- `_C.fused_gdn_gating_precompiled`

So Windows is not simply missing the GDN prefill/decode precompiled extension.
The major runtime difference is backend selection:

- Windows has no usable Triton runtime, so it logs:
  - `Triton is not available ... Falling back ... eager`
  - MTP drafter: `Using CUDA ATen backend for Unquantized MoE`
  - Mamba state block copy: `_C_cache_ops.swap_blocks` because Triton is unavailable
- WSL has Triton and logs:
  - MTP drafter: `Using TRITON backend for Unquantized MoE`

### Windows Timing Breakdown

Windows timing run:

- `.bench_logs/20260526_prefill_win_wsl_compare/windows_prefill_8192_no_mtp_timing.json`
- `.bench_logs/20260526_prefill_win_wsl_compare/windows_prefill_8192_no_mtp_timing.err.log`

The 8192-token prefill was internally split into two chunks: `6288 + 1904`.
Timing mode adds synchronization overhead, so the totals are for bottleneck
location only, not final TPS reporting.

| Component | Sum |
|---|---:|
| Qwen3.5 linear-attention outer total | 5.293s |
| GDN recurrent core inside that | 4.553s |
| MoE prefill burst total | 3.778s |
| MoE burst prepare | 3.497s |
| MoE burst apply/kernel | 0.278s |
| MoE prepare, layers 0..33 | 1.243s |
| MoE prepare, layers 34..47 | 2.254s |

Conclusion: the Windows 8192-token prefill gap is mainly a combination of:

1. GDN/linear-attention prefill recurrent work running without Triton.
2. MoE burst prepare on late pageable layers.
3. A smaller MTP-drafter backend penalty when MTP is enabled.

The precompiled operator symbols exist, but this does not mean the Windows
implementation is algorithmically equivalent to the Triton implementation. The
current Windows `chunk_gated_delta_rule_recurrent_cuda_fast` implementation
uses a token-serial recurrence inside each CUDA block. Triton uses the chunked
FLA/WY path with chunk-level matrix operations and parallel output recovery.
That is the core mismatch.

### GDN Microbench

Standalone random-tensor microbench, shape matching the Qwen3.5 prefill log:

- `q/k = [1, T, 16, 128]`
- `v/output = [1, T, 64, 128]`
- `state = [1, 64, 128, 128]`
- `cu_seqlens = [0, T]`
- `use_qk_l2norm_in_kernel=True`

Files:

- `.bench_logs/20260526_prefill_win_wsl_compare/gdn_microbench.py`
- `.bench_logs/20260526_prefill_win_wsl_compare/gdn_microbench_triton_allocator.py`

| T | Windows precompiled avg | WSL Triton avg after compile | Ratio |
|---:|---:|---:|---:|
| 1904 | 29.465 ms | 0.587 ms | 50.2x |
| 6288 | 96.941 ms | 1.621 ms | 59.8x |

WSL first warmup includes Triton compile/autotune and is intentionally excluded
from steady-state timing. After compilation, the gap is far too large to be
explained by "precompiled vs dynamic compilation"; it shows that the Windows
C++/CUDA implementation is not aligned with the Triton algorithm.

## Windows GDN Optimization Update

Two Windows CUDA fixes were added after the bottleneck check:

1. Corrected recurrent update order to match the Python reference:
   decay state first, then compute `delta_dot`, then apply the delta update.
2. Avoided repeated q/k L2 normalization and repeated q/k reads:
   - q/k are normalized once before the recurrent kernel when
     `use_qk_l2norm_in_kernel=True`;
   - the recurrent CUDA kernel now processes 8 value columns per block by
     default, instead of one value column per block.

Correctness smoke test against `_chunk_gated_delta_rule_ref`:

| Check | Max abs error |
|---|---:|
| Standard small shape | 0.00198 state / 0.00195 output |
| Non-8-divisible V shape | 0.00287 state / 0.00195 output |

Standalone Windows GDN microbench after the fixes:

| T | Before fixes | q/k normalize once | V_TILE=8 final | WSL Triton reference |
|---:|---:|---:|---:|---:|
| 1904 | 29.465 ms | 16.039 ms | 7.152 ms | 0.587 ms |
| 6288 | 96.941 ms | 53.426 ms | 23.218 ms | 1.621 ms |

`V_TILE=16` was tested and rejected because register pressure made it slower:

| T | V_TILE=16 |
|---:|---:|
| 1904 | 10.240 ms |
| 6288 | 33.819 ms |

122B end-to-end 8192-token no-MTP prefill after the fixes:

| Runtime | Variant | Cold TTFT | Warm TTFT |
|---|---|---:|---:|
| Windows | before GDN fixes | 9.342s | 8.563s |
| Windows | q/k normalize once | 7.848s | 7.223s |
| Windows | q/k normalize once + V_TILE=8, pinned40 | 6.416s | 5.776s |
| Windows | q/k normalize once + V_TILE=8, pinned44 | 6.149s | 5.503s |
| WSL | Triton, before Windows fixes | 21.364s | 5.489s |

The Windows 8192-token prefill gap is now small enough that GUI Agent latency is
no longer dominated by a broken Windows GDN prefill path. With the recommended
`cpu_static_pinned_gb=44` setting, Windows 8192-token warm prefill is effectively
at the same level as the earlier WSL Triton reference. Further gains require
either optimizing MoE burst prepare for pageable late layers, or porting the
full Triton chunked FLA/WY path to CUDA; the current V_TILE recurrent kernel is
still algorithmically more serial than Triton.
