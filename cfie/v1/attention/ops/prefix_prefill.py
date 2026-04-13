# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# The kernels in this file are adapted from LightLLM's context_attention_fwd:
# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py

import torch
import torch.nn.functional as F

from cfie import _custom_ops as ops
from cfie.logger import init_logger
from cfie.platforms import current_platform
from cfie.triton_utils import HAS_TRITON, tl, triton

# Static kernels parameters
BASE_BLOCK = 128 if current_platform.has_device_capability(80) else 64
NUM_WARPS = 4 if current_platform.is_rocm() else 8

# To check compatibility
IS_TURING = current_platform.get_device_capability() == (7, 5)
float8_info = torch.finfo(current_platform.fp8_dtype())
logger = init_logger(__name__)
FLOAT8_DTYPES = {
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
        current_platform.fp8_dtype(),
    )
    if dtype is not None
}


def _is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in FLOAT8_DTYPES


def _extract_scale_scalar(scale: torch.Tensor | None, *, device: torch.device) -> float:
    if scale is None:
        return 1.0
    return float(scale.to(device=device, dtype=torch.float32).reshape(-1)[0].item())


def _reshape_key_cache_for_reference(key_cache: torch.Tensor) -> torch.Tensor:
    return key_cache.permute(0, 1, 3, 2, 4).reshape(
        key_cache.shape[0],
        key_cache.shape[1],
        key_cache.shape[3],
        key_cache.shape[2] * key_cache.shape[4],
    )


def _reshape_value_cache_for_reference(value_cache: torch.Tensor) -> torch.Tensor:
    return value_cache.permute(0, 1, 3, 2).contiguous()


def _supports_precompiled_paged_kv_gather(
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    processed_b_loc: torch.Tensor,
) -> bool:
    return (
        hasattr(torch.ops, "_C_cache_ops")
        and hasattr(torch.ops._C_cache_ops, "gather_paged_kv_cache")
        and k_cache.is_cuda
        and v_cache.is_cuda
        and processed_b_loc.is_cuda
        and k_cache.ndim == 5
        and v_cache.ndim == 4
        and processed_b_loc.ndim == 2
        and k_cache.device == v_cache.device == processed_b_loc.device
    )


def _try_gather_paged_context_cache(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    processed_b_loc: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if not _supports_precompiled_paged_kv_gather(
        k_cache=k_cache,
        v_cache=v_cache,
        processed_b_loc=processed_b_loc,
    ):
        return None, None, None

    device = k_cache.device
    ctx_lens = (
        b_seq_len.to(device=device, dtype=torch.int32)
        - (b_start_loc[1:] - b_start_loc[:-1]).to(device=device, dtype=torch.int32)
    )
    if bool(torch.any(ctx_lens < 0).item()):
        return None, None, None

    cu_ctx_lens = torch.empty(
        (ctx_lens.numel() + 1,),
        dtype=torch.int32,
        device=device,
    )
    cu_ctx_lens[0] = 0
    cu_ctx_lens[1:] = torch.cumsum(ctx_lens, dim=0)
    total_ctx_tokens = int(cu_ctx_lens[-1].item())
    if total_ctx_tokens == 0:
        return (
            torch.empty(
                (0, k.shape[1], q.shape[-1]),
                dtype=k_cache.dtype,
                device=device,
            ),
            torch.empty(
                (0, v.shape[1], v.shape[-1]),
                dtype=v_cache.dtype,
                device=device,
            ),
            cu_ctx_lens,
        )

    gathered_key = torch.empty(
        (total_ctx_tokens, k.shape[1], q.shape[-1]),
        dtype=k_cache.dtype,
        device=device,
    )
    gathered_value = torch.empty(
        (total_ctx_tokens, v.shape[1], v.shape[-1]),
        dtype=v_cache.dtype,
        device=device,
    )
    try:
        ops.gather_paged_kv_cache(
            key_cache=k_cache,
            value_cache=v_cache,
            gathered_key=gathered_key,
            gathered_value=gathered_value,
            block_table=processed_b_loc.contiguous(),
            cu_seq_lens=cu_ctx_lens,
            batch_size=ctx_lens.numel(),
        )
    except (AttributeError, NotImplementedError, RuntimeError, ValueError) as exc:
        logger.warning_once(
            "Precompiled paged KV gather is unavailable for prefix prefill; "
            "falling back to Python cache materialization. Reason: %s",
            exc,
        )
        return None, None, None

    logger.info_once(
        "Using precompiled paged KV gather for prefix prefill cache "
        "materialization because Triton runtime is unavailable."
    )
    return gathered_key, gathered_value, cu_ctx_lens


def _supports_reference_sdpa_fastpath(
    *,
    sinks: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
) -> bool:
    return (
        sinks is None
        and alibi_slopes is None
        and hasattr(F, "scaled_dot_product_attention")
    )


def _supports_precompiled_prefix_prefill_attention(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    gathered_ctx_k: torch.Tensor | None,
    gathered_ctx_v: torch.Tensor | None,
    cu_ctx_lens: torch.Tensor | None,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    sinks: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
    fp8_out_scale: torch.Tensor | float | None,
) -> bool:
    if (
        not ops.has_precompiled_prefix_prefill_attention()
        or gathered_ctx_k is None
        or gathered_ctx_v is None
        or cu_ctx_lens is None
        or sinks is not None
        or alibi_slopes is not None
        or fp8_out_scale is not None
    ):
        return False

    if not (
        q.is_cuda
        and k.is_cuda
        and v.is_cuda
        and o.is_cuda
        and gathered_ctx_k.is_cuda
        and gathered_ctx_v.is_cuda
        and cu_ctx_lens.is_cuda
        and b_start_loc.is_cuda
        and b_seq_len.is_cuda
    ):
        return False

    return not (
        _is_fp8_dtype(q.dtype)
        or _is_fp8_dtype(k.dtype)
        or _is_fp8_dtype(v.dtype)
        or _is_fp8_dtype(gathered_ctx_k.dtype)
        or _is_fp8_dtype(gathered_ctx_v.dtype)
    )


def _try_precompiled_prefix_prefill_attention(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    gathered_ctx_k: torch.Tensor | None,
    gathered_ctx_v: torch.Tensor | None,
    cu_ctx_lens: torch.Tensor | None,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    sliding_window: int,
    sm_scale: float,
    skip_decode: bool,
    sinks: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
    fp8_out_scale: torch.Tensor | float | None,
) -> bool:
    if not _supports_precompiled_prefix_prefill_attention(
        q=q,
        k=k,
        v=v,
        o=o,
        gathered_ctx_k=gathered_ctx_k,
        gathered_ctx_v=gathered_ctx_v,
        cu_ctx_lens=cu_ctx_lens,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
        sinks=sinks,
        alibi_slopes=alibi_slopes,
        fp8_out_scale=fp8_out_scale,
    ):
        return False

    try:
        ops.prefix_prefill_attention_precompiled(
            output=o,
            q=q,
            k=k,
            v=v,
            gathered_ctx_k=gathered_ctx_k,
            gathered_ctx_v=gathered_ctx_v,
            cu_ctx_lens=cu_ctx_lens,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            skip_decode=skip_decode,
        )
    except (AttributeError, NotImplementedError, RuntimeError, ValueError) as exc:
        logger.warning_once(
            "Precompiled prefix prefill attention compute is unavailable; "
            "falling back to shared reference compute. Reason: %s",
            exc,
        )
        return False

    logger.info_once(
        "Using precompiled prefix prefill attention compute because Triton "
        "runtime is unavailable."
    )
    return True


def _reference_context_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    processed_b_loc: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
    sliding_window: int,
    sm_scale: float,
    skip_decode: bool,
    fp8_out_scale: torch.Tensor | float | None,
    sinks: torch.Tensor | None,
) -> None:
    logger.warning_once(
        "Prefix prefill attention is falling back to the PyTorch reference "
        "path because Triton runtime is unavailable."
    )

    start_locs_cpu = b_start_loc.to(device="cpu", dtype=torch.int64)
    seq_lens_cpu = b_seq_len.to(device="cpu", dtype=torch.int64)
    gathered_ctx_k, gathered_ctx_v, cu_ctx_lens = _try_gather_paged_context_cache(
        q=q,
        k=k,
        v=v,
        k_cache=k_cache,
        v_cache=v_cache,
        processed_b_loc=processed_b_loc,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
    )
    if cu_ctx_lens is None:
        key_cache_dense = _reshape_key_cache_for_reference(k_cache)
        value_cache_dense = _reshape_value_cache_for_reference(v_cache)
        block_size = key_cache_dense.shape[2]
        cu_ctx_lens_cpu = None
    else:
        key_cache_dense = None
        value_cache_dense = None
        block_size = 0
        cu_ctx_lens_cpu = cu_ctx_lens.to(device="cpu", dtype=torch.int64)
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    kv_group_num = num_q_heads // num_kv_heads

    if _try_precompiled_prefix_prefill_attention(
        q=q,
        k=k,
        v=v,
        o=o,
        gathered_ctx_k=gathered_ctx_k,
        gathered_ctx_v=gathered_ctx_v,
        cu_ctx_lens=cu_ctx_lens,
        b_start_loc=b_start_loc,
        b_seq_len=b_seq_len,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        skip_decode=skip_decode,
        sinks=sinks,
        alibi_slopes=alibi_slopes,
        fp8_out_scale=fp8_out_scale,
    ):
        return

    k_rescale = _extract_scale_scalar(k_scale, device=q.device)
    v_rescale = _extract_scale_scalar(v_scale, device=q.device)
    output_rescale = (
        _extract_scale_scalar(fp8_out_scale, device=q.device)
        if fp8_out_scale is not None
        else None
    )
    sink_values = (
        sinks.to(device=q.device, dtype=torch.float32).reshape(num_q_heads, 1, 1)
        if sinks is not None
        else None
    )
    alibi_values = (
        alibi_slopes.to(device=q.device, dtype=torch.float32).reshape(num_q_heads, 1, 1)
        if alibi_slopes is not None
        else None
    )

    for batch_idx in range(len(seq_lens_cpu)):
        seq_start = int(start_locs_cpu[batch_idx].item())
        seq_stop = int(start_locs_cpu[batch_idx + 1].item())
        query_len = seq_stop - seq_start
        seq_len = int(seq_lens_cpu[batch_idx].item())
        ctx_len = seq_len - query_len

        if query_len <= 0:
            continue
        if skip_decode and query_len == 1:
            continue

        if cu_ctx_lens_cpu is not None:
            ctx_start = int(cu_ctx_lens_cpu[batch_idx].item())
            ctx_stop = int(cu_ctx_lens_cpu[batch_idx + 1].item())
            ctx_k = gathered_ctx_k[ctx_start:ctx_stop]
            ctx_v = gathered_ctx_v[ctx_start:ctx_stop]
        else:
            num_ctx_blocks = (ctx_len + block_size - 1) // block_size
            if num_ctx_blocks > 0:
                block_ids = processed_b_loc[batch_idx, :num_ctx_blocks].to(
                    device=key_cache_dense.device,
                    dtype=torch.long,
                )
                ctx_k = key_cache_dense.index_select(0, block_ids).reshape(
                    num_ctx_blocks * block_size,
                    num_kv_heads,
                    -1,
                )[:ctx_len]
                ctx_v = value_cache_dense.index_select(0, block_ids).reshape(
                    num_ctx_blocks * block_size,
                    num_kv_heads,
                    -1,
                )[:ctx_len]
            else:
                ctx_k = k.new_empty((0, num_kv_heads, q.shape[-1]))
                ctx_v = v.new_empty((0, num_kv_heads, v.shape[-1]))

        if _is_fp8_dtype(ctx_k.dtype):
            ctx_k = ctx_k.to(torch.float32) * k_rescale
        else:
            ctx_k = ctx_k.to(torch.float32)
        if _is_fp8_dtype(ctx_v.dtype):
            ctx_v = ctx_v.to(torch.float32) * v_rescale
        else:
            ctx_v = ctx_v.to(torch.float32)

        q_seq = q[seq_start:seq_stop].to(torch.float32)
        k_seq = k[seq_start:seq_stop].to(torch.float32)
        v_seq = v[seq_start:seq_stop].to(torch.float32)

        all_k = torch.cat([ctx_k, k_seq], dim=0)
        all_v = torch.cat([ctx_v, v_seq], dim=0)

        q_heads = q_seq.permute(1, 0, 2).contiguous()
        k_heads = all_k.permute(1, 0, 2).contiguous()
        v_heads = all_v.permute(1, 0, 2).contiguous()

        if kv_group_num > 1:
            k_heads = k_heads.repeat_interleave(kv_group_num, dim=0)
            v_heads = v_heads.repeat_interleave(kv_group_num, dim=0)

        scores = torch.matmul(q_heads, k_heads.transpose(-1, -2)) * sm_scale

        query_positions = ctx_len + torch.arange(query_len, device=q.device)
        key_positions = torch.arange(ctx_len + query_len, device=q.device)
        mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        if sliding_window > 0:
            mask &= (query_positions.unsqueeze(1) - key_positions.unsqueeze(0)) < (
                sliding_window
            )

        if _supports_reference_sdpa_fastpath(
            sinks=sinks,
            alibi_slopes=alibi_slopes,
        ):
            out = F.scaled_dot_product_attention(
                q_heads.unsqueeze(0),
                k_heads.unsqueeze(0),
                v_heads.unsqueeze(0),
                attn_mask=mask.unsqueeze(0).unsqueeze(0),
                dropout_p=0.0,
                is_causal=False,
                scale=sm_scale,
            ).squeeze(0)
        else:
            scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))

            if alibi_values is not None:
                relative_positions = (
                    key_positions.unsqueeze(0) - query_positions.unsqueeze(1)
                ).to(torch.float32)
                scores = scores + alibi_values * relative_positions.unsqueeze(0)

            if sink_values is not None:
                scores = torch.cat(
                    [sink_values.expand(-1, query_len, 1), scores],
                    dim=-1,
                )
                sink_v = torch.zeros(
                    (num_q_heads, 1, v_heads.shape[-1]),
                    dtype=torch.float32,
                    device=q.device,
                )
                v_heads = torch.cat([sink_v, v_heads], dim=1)

            row_max = scores.amax(dim=-1, keepdim=True)
            row_max = torch.where(
                torch.isfinite(row_max), row_max, torch.zeros_like(row_max)
            )
            exp_scores = torch.exp(scores - row_max)
            denom = exp_scores.sum(dim=-1, keepdim=True)
            probs = torch.where(
                denom > 0,
                exp_scores / denom,
                torch.zeros_like(exp_scores),
            )
            out = torch.matmul(probs, v_heads)

        if output_rescale is not None:
            out = out * (1.0 / output_rescale)
            out = torch.clamp(out, float8_info.min, float8_info.max)

        o[seq_start:seq_stop].copy_(out.permute(1, 0, 2).to(dtype=o.dtype))


# Here's an example autotuner config for this kernel. This config does provide
# a performance improvement, but dramatically increases first call latency in
# triton 3.2. Because of this tradeoff, it's currently commented out.
# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, \
#                         "num_unroll_cache": 4, \
#                         "num_unroll_request": 1 } | \
#                         ({"kpack": 2, "waves_per_eu": 2} \
#                             if current_platform.is_rocm() else {}), \
#                         num_warps=4, \
#                         num_stages=1)
#     ],
#     key=["BLOCK_SIZE", "MAX_Q_LEN", "MAX_CTX_LEN"]
# )
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    K_cache,
    V_cache,
    sink_ptr,
    B_Loc,
    sm_scale,
    k_scale,
    v_scale,
    out_scale_inv,
    B_Start_Loc,
    B_Seqlen,
    x: tl.constexpr,
    Out,
    stride_b_loc_b,
    stride_b_loc_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_bl: tl.constexpr,
    stride_k_cache_x,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    num_queries_per_kv: tl.constexpr,
    IN_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_PADDED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PHYSICAL_BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    num_unroll_cache: tl.constexpr,
    num_unroll_request: tl.constexpr,
    SKIP_DECODE: tl.constexpr,
    USE_SINKS: tl.constexpr,
    USE_FP8: tl.constexpr,
    MAX_Q_LEN: tl.constexpr = 0,
    MAX_CTX_LEN: tl.constexpr = 0,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // num_queries_per_kv

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

    if SKIP_DECODE and cur_batch_query_len == 1:
        return

    # start position inside of the query
    # generally, N goes over kv, while M goes over query_len
    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    # [BLOCK_SIZE]; starts at 0
    offs_bs_n = tl.arange(0, BLOCK_SIZE)
    # [N]; starts at 0
    offs_n = tl.arange(0, BLOCK_N)
    # [D]; starts at 0
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
    # [M]; starts at current position in query
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # [M,D]
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )

    dim_mask = tl.where(tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(
        tl.int1
    )  # [D]

    q = tl.load(
        Q + off_q,
        mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len),
        other=0.0,
    )  # [M,D]

    # initialize pointer to m and l
    if not USE_SINKS:
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    else:
        m_i = tl.load(
            sink_ptr + tl.full([BLOCK_M], cur_head, dtype=tl.int64),
            mask=(offs_m < cur_batch_query_len),
            other=float("-inf"),
        ).to(dtype=tl.float32)
        l_i = tl.where(m_i > float("-inf"), 1.0, 0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)  # [M,D]

    # compute query against context (no causal mask here)
    for start_n in tl.range(
        0, cur_batch_ctx_len, BLOCK_SIZE, loop_unroll_factor=num_unroll_cache
    ):
        # Under a block size of 544 (Qwen/Qwen3-Next-80B-A3B-Thinking),
        # replace one physical block every 17 32-Tile blocks
        # Calculate the logical block index of each of the 32 tokens
        # in the current Tile (handling cross-block cases).
        token_indices = start_n + offs_bs_n
        bn_logical_indices = token_indices // PHYSICAL_BLOCK_SIZE

        # 2. Vectorized loading of physical block IDs from B_Loc
        bn = tl.load(
            B_Loc + cur_batch * stride_b_loc_b + bn_logical_indices * stride_b_loc_s
        ).to(tl.int64)

        # 3. Calculate the exact offset of
        # each token within its physical block.
        internal_offsets = token_indices % PHYSICAL_BLOCK_SIZE

        # Addressing of K (5D)
        off_k = (
            bn[None, :] * stride_k_cache_bs
            + cur_kv_head * stride_k_cache_h
            + (offs_d[:, None] // x) * stride_k_cache_d
            + internal_offsets[None, :] * stride_k_cache_bl
            + (offs_d[:, None] % x) * stride_k_cache_x
        )

        # Addressing of V (4D)
        off_v = (
            bn[:, None] * stride_v_cache_bs
            + cur_kv_head * stride_v_cache_h
            + offs_d[None, :] * stride_v_cache_d
            + internal_offsets[:, None] * stride_v_cache_bl
        )

        if (
            start_n + BLOCK_SIZE > cur_batch_ctx_len
            or BLOCK_DMODEL != BLOCK_DMODEL_PADDED
        ):
            k_load = tl.load(
                K_cache + off_k,
                mask=dim_mask[:, None]
                & ((start_n + offs_bs_n[None, :]) < cur_batch_ctx_len),
                other=0.0,
            )  # [D,N]
        else:
            k_load = tl.load(K_cache + off_k)

        if k_load.dtype.is_fp8():
            k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
        else:
            k = k_load

        # qk = tl.zeros([BLOCK_M, BLOCK_SIZE], dtype=tl.float32)  # [M,N]
        qk = sm_scale * tl.dot(q, k, input_precision=IN_PRECISION)
        qk = tl.where(
            (start_n + offs_bs_n[None, :]) < cur_batch_ctx_len, qk, float("-inf")
        )
        # qk *= sm_scale
        if SLIDING_WINDOW > 0:
            # (cur_batch_ctx_len + offs_m[:, None]) are the positions of
            # Q entries in sequence
            # (start_n + offs_bs_n[None, :]) are the positions of
            # KV entries in sequence
            # So the condition makes sure each entry in Q only attends
            # to KV entries not more than SLIDING_WINDOW away.
            #
            # We can't use -inf here, because the
            # sliding window may lead to the entire row being masked.
            # This then makes m_ij contain -inf, which causes NaNs in
            # exp().
            qk = tl.where(
                (cur_batch_ctx_len + offs_m[:, None]) - (start_n + offs_bs_n[None, :])
                < SLIDING_WINDOW,
                qk,
                float("-inf"),
            )

        # compute running maximum
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        acc = acc * alpha[:, None]

        # update acc
        if (
            start_n + BLOCK_SIZE > cur_batch_ctx_len
            or BLOCK_DMODEL != BLOCK_DMODEL_PADDED
        ):
            v_load = tl.load(
                V_cache + off_v,
                mask=dim_mask[None, :]
                & ((start_n + offs_bs_n[:, None]) < cur_batch_ctx_len),
                other=0.0,
            )  # [N,D]
        else:
            v_load = tl.load(V_cache + off_v)

        if v_load.dtype.is_fp8():
            v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
        else:
            v = v_load
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
        # # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    off_k = (
        offs_n[None, :] * stride_kbs
        + cur_kv_head * stride_kh
        + offs_d[:, None] * stride_kd
    )
    off_v = (
        offs_n[:, None] * stride_vbs
        + cur_kv_head * stride_vh
        + offs_d[None, :] * stride_vd
    )
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # block_mask is 0 when we're already past the current query length
    block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)

    # compute query against itself (with causal mask)
    for start_n in tl.range(
        0,
        block_mask * (start_m + 1) * BLOCK_M,
        BLOCK_N,
        loop_unroll_factor=num_unroll_request,
    ):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=dim_mask[:, None]
            & ((start_n + offs_n[None, :]) < cur_batch_query_len),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk *= sm_scale
        # apply causal mask
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        if SLIDING_WINDOW > 0:
            qk = tl.where(
                offs_m[:, None] - (start_n + offs_n[None, :]) < SLIDING_WINDOW,
                qk,
                float("-inf"),
            )

        # compute running maximum
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp(m_i - m_ij)
        # To prevent NaN from appearing in the first round
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        acc = acc * alpha[:, None]

        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=dim_mask[None, :]
            & ((start_n + offs_n[:, None]) < cur_batch_query_len),
            other=0.0,
        )
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
        # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / (l_i[:, None] + 1e-10)

    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
    tl.store(
        out_ptrs, acc, mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len)
    )
    return


@triton.jit
def _fwd_kernel_alibi(
    Q,
    K,
    V,
    K_cache,
    V_cache,
    B_Loc,
    sm_scale,
    k_scale,
    v_scale,
    B_Start_Loc,
    B_Seqlen,
    Alibi_slopes,
    block_size,
    x,
    Out,
    stride_b_loc_b,
    stride_b_loc_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_bl,
    stride_k_cache_x,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    num_queries_per_kv: int,
    IN_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,  # head size
    BLOCK_DMODEL_PADDED: tl.constexpr,  # head size padded to a power of 2
    BLOCK_N: tl.constexpr,
    SKIP_DECODE: tl.constexpr,
):
    # attn_bias[]
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // num_queries_per_kv

    # cur_batch_seq_len: the length of prompts
    # cur_batch_ctx_len: the length of prefix
    # cur_batch_in_all_start_index: the start id of the dim=0
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

    if SKIP_DECODE and cur_batch_query_len == 1:
        return

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )

    dim_mask = tl.where(tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(
        tl.int1
    )

    q = tl.load(
        Q + off_q,
        mask=dim_mask[None, :]
        & (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len),
        other=0.0,
    )

    # # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)

    alibi_slope = tl.load(Alibi_slopes + cur_head)
    alibi_start_q = tl.arange(0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
    alibi_start_k = 0
    for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        bn = tl.load(
            B_Loc
            + cur_batch * stride_b_loc_b
            + ((start_n + offs_n) // block_size) * stride_b_loc_s,
            mask=(start_n + offs_n) < cur_batch_ctx_len,
            other=0,
        ).to(tl.int64)
        off_k = (
            bn[None, :] * stride_k_cache_bs
            + cur_kv_head * stride_k_cache_h
            + (offs_d[:, None] // x) * stride_k_cache_d
            + ((start_n + offs_n[None, :]) % block_size) * stride_k_cache_bl
            + (offs_d[:, None] % x) * stride_k_cache_x
        )
        off_v = (
            bn[:, None] * stride_v_cache_bs
            + cur_kv_head * stride_v_cache_h
            + offs_d[None, :] * stride_v_cache_d
            + (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl
        )
        k_load = tl.load(
            K_cache + off_k,
            mask=dim_mask[:, None] & ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
            other=0.0,
        )  # [D,N]

        if k_load.dtype.is_fp8():
            k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
        else:
            k = k_load

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk = tl.where(
            (start_n + offs_n[None, :]) < cur_batch_ctx_len, qk, float("-inf")
        )
        qk *= sm_scale

        # load alibi
        alibi = (
            tl.arange(0, BLOCK_N)[None, :] + alibi_start_k - alibi_start_q[:, None]
        ) * alibi_slope
        alibi = tl.where(
            (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
            alibi,
            float("-inf"),
        )
        qk += alibi
        alibi_start_k += BLOCK_N

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i

        alpha = tl.math.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        # -- update output accumulator --
        # scale p
        # scale acc
        acc_scale = alpha
        # acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v_load = tl.load(
            V_cache + off_v,
            mask=dim_mask[None, :] & ((start_n + offs_n[:, None]) < cur_batch_ctx_len),
            other=0.0,
        )
        if v_load.dtype.is_fp8():
            v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
        else:
            v = v_load
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision="ieee")
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    off_k = (
        offs_n[None, :] * stride_kbs
        + cur_kv_head * stride_kh
        + offs_d[:, None] * stride_kd
    )
    off_v = (
        offs_n[:, None] * stride_vbs
        + cur_kv_head * stride_vh
        + offs_d[None, :] * stride_vd
    )
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    block_mask = tl.where(block_start_loc < cur_batch_seq_len - cur_batch_ctx_len, 1, 0)

    # init alibi
    alibi_slope = tl.load(Alibi_slopes + cur_head)
    alibi_start_q = tl.arange(0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
    alibi_start_k = cur_batch_ctx_len
    # # init debugger
    # offset_db_q = tl.arange(0, BLOCK_M) + block_start_loc
    # offset_db_k = tl.arange(0, BLOCK_N)
    # calc q[BLOCK_M, BLOCK_MODEL] mul k[prefix_len: , BLOCK_DMODEL]
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(
            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
            mask=dim_mask[:, None]
            & ((start_n + offs_n[None, :]) < cur_batch_seq_len - cur_batch_ctx_len),
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision="ieee")
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

        # load alibi
        alibi = (
            tl.arange(0, BLOCK_N)[None, :] + alibi_start_k - alibi_start_q[:, None]
        ) * alibi_slope
        alibi = tl.where(
            (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
            alibi,
            float("-inf"),
        )
        qk += alibi
        alibi_start_k += BLOCK_N

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.math.exp(qk - m_i_new[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i

        alpha = tl.math.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        # -- update output accumulator --
        # scale p
        # scale acc
        acc_scale = alpha
        # acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(
            v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
            mask=dim_mask[None, :]
            & ((start_n + offs_n[:, None]) < cur_batch_seq_len - cur_batch_ctx_len),
            other=0.0,
        )
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision="ieee")
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    acc = acc / l_i[:, None]

    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(
        out_ptrs,
        acc,
        mask=dim_mask[None, :]
        & (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len),
    )
    return


@torch.inference_mode()
def context_attention_fwd(
    q,
    k,
    v,
    o,
    kv_cache_dtype: str,
    k_cache,
    v_cache,
    b_loc,
    b_start_loc,
    b_seq_len,
    max_seq_len,
    max_input_len,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
    skip_decode=False,
    fp8_out_scale=None,
    sinks=None,
    is_block_table_ptr: bool = False,
):
    q_dtype_is_f32 = q.dtype is torch.float32

    # Turing does have tensor core for float32 multiplication
    # use ieee as fallback for triton kernels work. There is also
    # warning on vllm/config.py to inform users this fallback
    # implementation
    IN_PRECISION = "ieee" if IS_TURING and q_dtype_is_f32 else None

    # Conversion of FP8 Tensor from uint8 storage to
    # appropriate torch.dtype for interpretation by Triton
    if "fp8" in kv_cache_dtype:
        assert k_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]
        assert v_cache.dtype in [torch.uint8, current_platform.fp8_dtype()]

        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            target_dtype = current_platform.fp8_dtype()
        elif kv_cache_dtype == "fp8_e5m2":
            target_dtype = torch.float8_e5m2
        else:
            raise ValueError("Unsupported FP8 dtype:", kv_cache_dtype)

        k_cache = k_cache.view(target_dtype)
        v_cache = v_cache.view(target_dtype)

    if (
        k_cache.dtype == torch.uint8
        or v_cache.dtype == torch.uint8
        and kv_cache_dtype == "auto"
    ):
        raise ValueError(
            "kv_cache_dtype='auto' unsupported for\
            FP8 KV Cache prefill kernel"
        )

    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    # round up Lk to a power of 2 - this is required for Triton block size
    Lk_padded = triton.next_power_of_2(Lk)

    if sm_scale is None:
        sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    num_queries_per_kv = q.shape[1] // k.shape[1]

    assert batch + 1 == len(b_start_loc)

    # 0 means "disable"
    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    if is_block_table_ptr:
        kv_element_size = k_cache.element_size()
        block_byte_stride = k_cache.stride(0) * kv_element_size
        # The physical starting point of the obtained KV Cache Pool
        base_addr = k_cache.data_ptr()

        mask = b_loc > 0
        processed_b_loc = torch.where(
            mask, (b_loc - base_addr) // block_byte_stride, b_loc
        ).to(torch.int32)
    else:
        processed_b_loc = b_loc.to(torch.int32)

    if not HAS_TRITON:
        _reference_context_attention_fwd(
            q=q,
            k=k,
            v=v,
            o=o,
            processed_b_loc=processed_b_loc,
            b_start_loc=b_start_loc,
            b_seq_len=b_seq_len,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=k_scale,
            v_scale=v_scale,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            skip_decode=skip_decode,
            fp8_out_scale=fp8_out_scale,
            sinks=sinks,
        )
        return

    if alibi_slopes is not None:
        assert sinks is None, "Sinks arg is not supported with alibi"
        assert fp8_out_scale is None, "FP8 output not supported with alibi"
        # need to reduce num. blocks when using fp32
        # due to increased use of GPU shared memory
        # if q.dtype is torch.float32:
        BLOCK = BASE_BLOCK // 2 if q_dtype_is_f32 else BASE_BLOCK
        # batch, head,
        grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
        _fwd_kernel_alibi[grid](
            q,
            k,
            v,
            k_cache,
            v_cache,
            b_loc,
            sm_scale,
            k_scale,
            v_scale,
            b_start_loc,
            b_seq_len,
            alibi_slopes,
            v_cache.shape[3],
            k_cache.shape[4],
            o,
            b_loc.stride(0),
            b_loc.stride(1),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            k_cache.stride(4),  # [num_blocks, num_kv_heads, head_size/x, block_size, x]
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),  # [num_blocks, num_kv_heads, head_size, block_size]
            num_queries_per_kv=num_queries_per_kv,
            IN_PRECISION=IN_PRECISION,
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=Lk,
            BLOCK_DMODEL_PADDED=Lk_padded,
            BLOCK_N=BLOCK,
            SKIP_DECODE=skip_decode,
            num_warps=NUM_WARPS,
            num_stages=1,
        )
        return

    max_seq_len = 0 if max_seq_len is None else max_seq_len
    extra_kargs = {}
    if current_platform.is_rocm():
        extra_kargs = {}

    real_block_size = v_cache.shape[3]
    is_pow2 = real_block_size > 0 and (real_block_size & (real_block_size - 1) == 0)
    # For standard models involving powers of 2,
    # follow the original logic (Llama 128/64)
    # For non-standard models (Qwen3-next block_size 544), set to 32.
    if is_pow2:
        BLOCK_M = 128
        BLOCK_N = 64
    else:
        BLOCK_M = 32
        BLOCK_N = 32

    # TRITON_BLOCK_SIZE is kept at 32 to ensure
    # correct alignment logic when the kernel handles
    # non-standard sizes (such as 544).
    TRITON_BLOCK_SIZE = 32

    grid_fn = lambda META: (batch, head, triton.cdiv(max_input_len, META["BLOCK_M"]))
    _fwd_kernel[grid_fn](
        q,
        k,
        v,
        k_cache,
        v_cache,
        sinks,
        processed_b_loc,
        sm_scale,
        k_scale,
        v_scale,
        1.0 / fp8_out_scale if fp8_out_scale is not None else 1.0,
        b_start_loc,
        b_seq_len,
        k_cache.shape[4],
        o,
        processed_b_loc.stride(0),
        processed_b_loc.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        stride_k_cache_bs=k_cache.stride(0),
        stride_k_cache_h=k_cache.stride(1),
        stride_k_cache_d=k_cache.stride(2),
        stride_k_cache_bl=k_cache.stride(3),
        stride_k_cache_x=k_cache.stride(4),
        stride_v_cache_bs=v_cache.stride(0),
        stride_v_cache_h=v_cache.stride(1),
        stride_v_cache_d=v_cache.stride(2),
        stride_v_cache_bl=v_cache.stride(3),
        BLOCK_SIZE=TRITON_BLOCK_SIZE,
        PHYSICAL_BLOCK_SIZE=real_block_size,
        num_queries_per_kv=num_queries_per_kv,
        IN_PRECISION=IN_PRECISION,
        BLOCK_DMODEL=Lk,
        BLOCK_DMODEL_PADDED=Lk_padded,
        SLIDING_WINDOW=sliding_window,
        SKIP_DECODE=skip_decode,
        USE_FP8=fp8_out_scale is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_unroll_cache=4,
        num_unroll_request=1,
        num_warps=4,
        num_stages=1,
        USE_SINKS=sinks is not None,
        **extra_kargs,
    )
    return
