# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Literal

import torch

import cfie.envs as envs
from cfie.logger import init_logger
from cfie.platforms import current_platform
from cfie.scalar_type import ScalarType
from cfie.utils.flashinfer import (
    flashinfer_quant_nvfp4_8x4_sf_layout,
)
from cfie.utils.math_utils import cdiv

logger = init_logger(__name__)

current_platform.import_kernels()

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


# page attention ops
def paged_attention_v1(
        out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_kv_heads: int,
        scale: float,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        max_seq_len: int,
        alibi_slopes: torch.Tensor | None,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
) -> None:
    torch.ops._C.paged_attention_v1(
        out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
        alibi_slopes,
        kv_cache_dtype,
        k_scale,
        v_scale,
        tp_rank,
        blocksparse_local_blocks,
        blocksparse_vert_stride,
        blocksparse_block_size,
        blocksparse_head_sliding_step,
    )


def paged_attention_v2(
        out: torch.Tensor,
        exp_sum: torch.Tensor,
        max_logits: torch.Tensor,
        tmp_out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_kv_heads: int,
        scale: float,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        max_seq_len: int,
        alibi_slopes: torch.Tensor | None,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
) -> None:
    torch.ops._C.paged_attention_v2(
        out,
        exp_sum,
        max_logits,
        tmp_out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
        alibi_slopes,
        kv_cache_dtype,
        k_scale,
        v_scale,
        tp_rank,
        blocksparse_local_blocks,
        blocksparse_vert_stride,
        blocksparse_block_size,
        blocksparse_head_sliding_step,
    )


def paged_attention_rocm(
        out: torch.Tensor,
        exp_sum: torch.Tensor,
        max_logits: torch.Tensor,
        tmp_out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        num_kv_heads: int,
        scale: float,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor | None,
        block_size: int,
        max_seq_len: int,
        alibi_slopes: torch.Tensor | None,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        fp8_out_scale: torch.Tensor | None = None,
        mfma_type: str = "fp8" if envs.VLLM_ROCM_FP8_MFMA_PAGE_ATTN else "f16",
) -> None:
    torch.ops._rocm_C.paged_attention(
        out,
        exp_sum,
        max_logits,
        tmp_out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        query_start_loc,
        block_size,
        max_seq_len,
        alibi_slopes,
        kv_cache_dtype,
        k_scale,
        v_scale,
        fp8_out_scale,
        mfma_type,
    )


def mla_decode_kvcache_cpu(
        out: torch.Tensor,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        scale: float,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
) -> None:
    torch.ops._C.mla_decode_kvcache(out, query, kv_cache, scale, block_tables, seq_lens)


# merge attn states ops
def merge_attn_states(
        output: torch.Tensor,
        prefix_output: torch.Tensor,
        prefix_lse: torch.Tensor,
        suffix_output: torch.Tensor,
        suffix_lse: torch.Tensor,
        output_lse: torch.Tensor | None = None,
) -> None:
    torch.ops._C.merge_attn_states(
        output, output_lse, prefix_output, prefix_lse, suffix_output, suffix_lse
    )


def convert_vertical_slash_indexes(
        q_seqlens: torch.Tensor,  # [BATCH, ]
        kv_seqlens: torch.Tensor,  # [BATCH, ]
        vertical_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
        slash_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
        context_size: int,
        block_size_M: int,
        block_size_N: int,
        causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = slash_indexes.size(0)
    num_heads = slash_indexes.size(1)
    nnz_slash = slash_indexes.size(2)
    nnz_vertical = vertical_indexes.size(2)
    num_rows = (context_size + block_size_M - 1) // block_size_M

    block_count = torch.zeros(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    block_offset = torch.zeros(
        batch_size,
        num_heads,
        num_rows,
        nnz_slash,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )
    column_count = torch.zeros(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    column_index = torch.zeros(
        batch_size,
        num_heads,
        num_rows,
        nnz_vertical,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )

    torch.ops._C.convert_vertical_slash_indexes(
        block_count,
        block_offset,
        column_count,
        column_index,
        q_seqlens,
        kv_seqlens,
        vertical_indexes,
        slash_indexes,
        context_size,
        block_size_M,
        block_size_N,
        causal,
    )
    return block_count, block_offset, column_count, column_index


def convert_vertical_slash_indexes_mergehead(
        q_seqlens: torch.Tensor,  # [BATCH, ]
        kv_seqlens: torch.Tensor,  # [BATCH, ]
        vertical_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
        slash_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
        # [N_HEADS] : different head use different number of indices
        vertical_indices_count: torch.Tensor,
        slash_indices_count: torch.Tensor,
        context_size: int,
        block_size_M: int,
        block_size_N: int,
        causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = slash_indexes.size(0)
    num_heads = slash_indexes.size(1)
    nnz_slash = slash_indexes.size(2)
    nnz_vertical = vertical_indexes.size(2)
    num_rows = (context_size + block_size_M - 1) // block_size_M

    block_count = torch.empty(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    block_offset = torch.empty(
        batch_size,
        num_heads,
        num_rows,
        nnz_slash,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )
    column_count = torch.empty(
        batch_size, num_heads, num_rows, dtype=q_seqlens.dtype, device=q_seqlens.device
    )
    column_index = torch.empty(
        batch_size,
        num_heads,
        num_rows,
        nnz_vertical,
        dtype=q_seqlens.dtype,
        device=q_seqlens.device,
    )

    torch.ops._C.convert_vertical_slash_indexes_mergehead(
        block_count,
        block_offset,
        column_count,
        column_index,
        q_seqlens,
        kv_seqlens,
        vertical_indexes,
        slash_indexes,
        vertical_indices_count,
        slash_indices_count,
        context_size,
        block_size_M,
        block_size_N,
        causal,
    )
    return block_count, block_offset, column_count, column_index


# pos encoding ops
def rotary_embedding(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        head_size: int,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
) -> None:
    torch.ops._C.rotary_embedding(
        positions, query, key, head_size, cos_sin_cache, is_neox
    )


def has_mrope_rotary_embedding() -> bool:
    return hasattr(torch.ops._C, "mrope_rotary_embedding")


def mrope_rotary_embedding(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        head_size: int,
        rotary_dim: int,
        mrope_section: list[int],
        is_neox: bool,
        mrope_interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C.mrope_rotary_embedding(
        query,
        key,
        cos,
        sin,
        head_size,
        rotary_dim,
        mrope_section,
        is_neox,
        mrope_interleaved,
    )


# layer norm ops
def rms_norm(
        out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    torch.ops._C.rms_norm(out, input, weight, epsilon)


def fused_add_rms_norm(
        input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)


def fused_qk_norm_rope(
        qkv: torch.Tensor,
        num_heads_q: int,
        num_heads_k: int,
        num_heads_v: int,
        head_dim: int,
        eps: float,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        position_ids: torch.Tensor,
) -> None:
    torch.ops._C.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        position_ids,
    )


def has_gated_layer_norm() -> bool:
    return hasattr(torch.ops._C, "gated_layer_norm")


def gated_layer_norm(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        gate: torch.Tensor | None,
        epsilon: float,
        group_size: int,
        norm_before_gate: bool,
        is_rms_norm: bool,
        activation: str,
) -> torch.Tensor:
    return torch.ops._C.gated_layer_norm(
        input,
        weight,
        bias,
        gate,
        epsilon,
        group_size,
        norm_before_gate,
        is_rms_norm,
        activation,
    )


def has_precompiled_fused_sigmoid_gating_delta_rule_update() -> bool:
    return hasattr(
        torch.ops._C, "fused_sigmoid_gating_delta_rule_update_precompiled"
    )


def fused_sigmoid_gating_delta_rule_update_precompiled(
        A_log: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: float,
        threshold: float,
        scale: float,
        initial_state: torch.Tensor,
        inplace_final_state: bool,
        cu_seqlens: torch.Tensor | None,
        ssm_state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor | None,
        use_qk_l2norm_in_kernel: bool,
        is_kda: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C.fused_sigmoid_gating_delta_rule_update_precompiled(
        A_log,
        a,
        b,
        dt_bias,
        q,
        k,
        v,
        beta,
        threshold,
        scale,
        initial_state,
        inplace_final_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        use_qk_l2norm_in_kernel,
        is_kda,
    )


def has_precompiled_apply_rotary_emb() -> bool:
    return hasattr(torch.ops._C, "apply_rotary_emb_precompiled")


def apply_rotary_emb_precompiled(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_neox_style: bool,
        enable_fp32_compute: bool,
) -> torch.Tensor:
    return torch.ops._C.apply_rotary_emb_precompiled(
        x,
        cos,
        sin,
        is_neox_style,
        enable_fp32_compute,
    )


def has_precompiled_chunk_gated_delta_rule() -> bool:
    return hasattr(torch.ops._C, "chunk_gated_delta_rule_precompiled")


def chunk_gated_delta_rule_precompiled(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None,
        use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C.chunk_gated_delta_rule_precompiled(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    )


def has_precompiled_fused_recurrent_gated_delta_rule_packed_decode() -> bool:
    return hasattr(
        torch.ops._C, "fused_recurrent_gated_delta_rule_packed_decode_precompiled"
    )


def fused_recurrent_gated_delta_rule_packed_decode_precompiled(
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        out: torch.Tensor,
        ssm_state_indices: torch.Tensor,
        use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C.fused_recurrent_gated_delta_rule_packed_decode_precompiled(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        out,
        ssm_state_indices,
        use_qk_l2norm_in_kernel,
    )


def has_precompiled_l2norm() -> bool:
    return hasattr(torch.ops._C, "l2norm_precompiled")


def l2norm_precompiled(
        x: torch.Tensor,
        eps: float = 1e-6,
        output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.ops._C.l2norm_precompiled(
        x,
        eps,
        output_dtype,
    )


def has_precompiled_chunk_local_cumsum() -> bool:
    return hasattr(torch.ops._C, "chunk_local_cumsum_precompiled")


def chunk_local_cumsum_precompiled(
        g: torch.Tensor,
        chunk_size: int,
        reverse: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        head_first: bool = False,
        output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.ops._C.chunk_local_cumsum_precompiled(
        g,
        chunk_size,
        reverse,
        cu_seqlens,
        head_first,
        output_dtype,
    )


def has_precompiled_chunk_fwd_o() -> bool:
    return hasattr(torch.ops._C, "chunk_fwd_o_precompiled")


def chunk_fwd_o_precompiled(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
        g: torch.Tensor | None,
        scale: float,
        cu_seqlens: torch.Tensor | None,
        block_size: int,
) -> torch.Tensor:
    return torch.ops._C.chunk_fwd_o_precompiled(
        q,
        k,
        v,
        h,
        g,
        scale,
        cu_seqlens,
        block_size,
    )


def has_precompiled_chunk_scaled_dot_kkt_fwd() -> bool:
    return hasattr(torch.ops._C, "chunk_scaled_dot_kkt_fwd_precompiled")


def chunk_scaled_dot_kkt_fwd_precompiled(
        k: torch.Tensor,
        g: torch.Tensor | None,
        beta: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        chunk_size: int,
        output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.ops._C.chunk_scaled_dot_kkt_fwd_precompiled(
        k,
        g,
        beta,
        cu_seqlens,
        chunk_size,
        output_dtype,
    )


def has_precompiled_chunk_gated_delta_rule_fwd_h() -> bool:
    return hasattr(torch.ops._C, "chunk_gated_delta_rule_fwd_h_precompiled")


def chunk_gated_delta_rule_fwd_h_precompiled(
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        chunk_size: int = 64,
        save_new_value: bool = True,
        cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    return torch.ops._C.chunk_gated_delta_rule_fwd_h_precompiled(
        k,
        w,
        u,
        g,
        gk,
        initial_state,
        output_final_state,
        chunk_size,
        save_new_value,
        cu_seqlens,
    )


def has_precompiled_solve_tril() -> bool:
    return hasattr(torch.ops._C, "solve_tril_precompiled")


def solve_tril_precompiled(
        A: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.ops._C.solve_tril_precompiled(
        A,
        cu_seqlens,
        output_dtype,
    )


def has_precompiled_recompute_w_u_fwd() -> bool:
    return hasattr(torch.ops._C, "recompute_w_u_fwd_precompiled")


def recompute_w_u_fwd_precompiled(
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g_cumsum: torch.Tensor,
        A: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C.recompute_w_u_fwd_precompiled(
        k,
        v,
        beta,
        g_cumsum,
        A,
        cu_seqlens,
    )


def has_precompiled_fused_gdn_gating() -> bool:
    return hasattr(torch.ops._C, "fused_gdn_gating_precompiled")


def fused_gdn_gating_precompiled(
        A_log: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dt_bias: torch.Tensor,
        beta: float,
        threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C.fused_gdn_gating_precompiled(
        A_log,
        a,
        b,
        dt_bias,
        beta,
        threshold,
    )


def has_precompiled_causal_conv1d_fn() -> bool:
    return hasattr(torch.ops._C, "causal_conv1d_fn_precompiled")


def causal_conv1d_fn_precompiled(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor | None,
        has_initial_state: torch.Tensor | None,
        activation: str,
        pad_slot_id: int,
) -> torch.Tensor:
    return torch.ops._C.causal_conv1d_fn_precompiled(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation,
        pad_slot_id,
    )


def has_precompiled_causal_conv1d_update() -> bool:
    return hasattr(torch.ops._C, "causal_conv1d_update_precompiled")


def causal_conv1d_update_precompiled(
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str,
        conv_state_indices: torch.Tensor | None,
        num_accepted_tokens: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        pad_slot_id: int,
        block_idx_last_scheduled_token: torch.Tensor | None,
        initial_state_idx: torch.Tensor | None,
) -> torch.Tensor:
    return torch.ops._C.causal_conv1d_update_precompiled(
        x,
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        pad_slot_id,
        block_idx_last_scheduled_token,
        initial_state_idx,
    )


def has_precompiled_zero_kv_blocks() -> bool:
    return hasattr(torch.ops._C, "zero_kv_blocks_precompiled")


def zero_kv_blocks_precompiled(
        block_ids: torch.Tensor,
        kv_tensors: list[torch.Tensor],
        block_dims: list[int],
        ratios: list[int],
) -> None:
    torch.ops._C.zero_kv_blocks_precompiled(
        block_ids,
        kv_tensors,
        block_dims,
        ratios,
    )


def has_precompiled_moe_batch_load_unquantized_runtime() -> bool:
    return hasattr(torch.ops._C, "moe_batch_load_unquantized_runtime_precompiled")


def moe_batch_load_unquantized_runtime_precompiled(
        slot_ids: torch.Tensor,
        w13_src: torch.Tensor,
        w2_src: torch.Tensor,
        w13_dst: torch.Tensor,
        w2_dst: torch.Tensor,
) -> None:
    torch.ops._C.moe_batch_load_unquantized_runtime_precompiled(
        slot_ids,
        w13_src,
        w2_src,
        w13_dst,
        w2_dst,
    )


def has_precompiled_moe_batch_load_gptq_runtime() -> bool:
    return hasattr(torch.ops._C, "moe_batch_load_gptq_runtime_precompiled")


def moe_batch_load_gptq_runtime_precompiled(
        slot_ids: torch.Tensor,
        w13_qweight_src: torch.Tensor,
        w2_qweight_src: torch.Tensor,
        w13_scales_src: torch.Tensor,
        w2_scales_src: torch.Tensor,
        w13_qzeros_src: torch.Tensor,
        w2_qzeros_src: torch.Tensor,
        w13_qweight_dst: torch.Tensor,
        w2_qweight_dst: torch.Tensor,
        w13_scales_dst: torch.Tensor,
        w2_scales_dst: torch.Tensor,
        w13_qzeros_dst: torch.Tensor,
        w2_qzeros_dst: torch.Tensor,
        w13_g_idx_src: torch.Tensor | None = None,
        w2_g_idx_src: torch.Tensor | None = None,
        w13_g_idx_sort_indices_src: torch.Tensor | None = None,
        w2_g_idx_sort_indices_src: torch.Tensor | None = None,
        w13_g_idx_dst: torch.Tensor | None = None,
        w2_g_idx_dst: torch.Tensor | None = None,
        w13_g_idx_sort_indices_dst: torch.Tensor | None = None,
        w2_g_idx_sort_indices_dst: torch.Tensor | None = None,
) -> None:
    torch.ops._C.moe_batch_load_gptq_runtime_precompiled(
        slot_ids,
        w13_qweight_src,
        w2_qweight_src,
        w13_scales_src,
        w2_scales_src,
        w13_qzeros_src,
        w2_qzeros_src,
        w13_qweight_dst,
        w2_qweight_dst,
        w13_scales_dst,
        w2_scales_dst,
        w13_qzeros_dst,
        w2_qzeros_dst,
        w13_g_idx_src,
        w2_g_idx_src,
        w13_g_idx_sort_indices_src,
        w2_g_idx_sort_indices_src,
        w13_g_idx_dst,
        w2_g_idx_dst,
        w13_g_idx_sort_indices_dst,
        w2_g_idx_sort_indices_dst,
    )


def has_precompiled_moe_batched_mm() -> bool:
    return hasattr(torch.ops._C, "moe_batched_mm_precompiled")


def moe_batched_mm_precompiled(
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        expert_num_tokens: torch.Tensor,
        A_scale: torch.Tensor | None,
        B_scale: torch.Tensor | None,
        use_fp8_w8a8: bool,
        per_act_token_quant: bool,
) -> None:
    torch.ops._C.moe_batched_mm_precompiled(
        A,
        B,
        C,
        expert_num_tokens,
        A_scale,
        B_scale,
        use_fp8_w8a8,
        per_act_token_quant,
    )


def has_precompiled_count_expert_num_tokens() -> bool:
    return hasattr(torch.ops._C, "count_expert_num_tokens_precompiled")


def count_expert_num_tokens_precompiled(
        topk_ids: torch.Tensor,
        num_local_experts: int,
        expert_map: torch.Tensor | None,
) -> torch.Tensor:
    return torch.ops._C.count_expert_num_tokens_precompiled(
        topk_ids,
        num_local_experts,
        expert_map,
    )


def has_precompiled_zero_experts_compute_identity() -> bool:
    return hasattr(torch.ops._C, "zero_experts_compute_identity_precompiled")


def zero_experts_compute_identity_precompiled(
        expert_indices: torch.Tensor,
        expert_scales: torch.Tensor,
        num_experts: int,
        hidden_states: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C.zero_experts_compute_identity_precompiled(
        expert_indices,
        expert_scales,
        num_experts,
        hidden_states,
    )


def has_precompiled_expand_batch_to_tokens() -> bool:
    return hasattr(torch.ops._C, "expand_batch_to_tokens_precompiled")


def has_precompiled_correct_attn_out() -> bool:
    return hasattr(torch.ops._C, "correct_attn_out_precompiled")


def has_precompiled_dcp_lse_combine() -> bool:
    return hasattr(torch.ops._C, "dcp_lse_combine_precompiled")


def has_precompiled_prefill_attention() -> bool:
    return hasattr(torch.ops._C, "prefill_attention_precompiled")


def has_precompiled_prefix_prefill_attention() -> bool:
    return hasattr(torch.ops._C, "prefix_prefill_attention_precompiled")


def has_precompiled_pack_seq() -> bool:
    return hasattr(torch.ops._C, "pack_seq_precompiled")


def has_precompiled_unpack_seq() -> bool:
    return hasattr(torch.ops._C, "unpack_seq_precompiled")


def correct_attn_out_precompiled(
        out: torch.Tensor,
        lses: torch.Tensor,
        cp_rank: int,
        is_lse_base_on_e: bool = True,
) -> torch.Tensor:
    return torch.ops._C.correct_attn_out_precompiled(
        out,
        lses,
        cp_rank,
        is_lse_base_on_e,
    )


def dcp_lse_combine_precompiled(
        recv_output: torch.Tensor,
        recv_lse: torch.Tensor,
        return_lse: bool = False,
        is_lse_base_on_e: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C.dcp_lse_combine_precompiled(
        recv_output,
        recv_lse,
        return_lse,
        is_lse_base_on_e,
    )


def prefill_attention_precompiled(
        output: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b_start_loc: torch.Tensor,
        b_seq_len: torch.Tensor,
        is_causal: bool,
        softmax_scale: float,
        sliding_window_q: int,
        sliding_window_k: int,
) -> None:
    torch.ops._C.prefill_attention_precompiled(
        output,
        q,
        k,
        v,
        b_start_loc,
        b_seq_len,
        is_causal,
        softmax_scale,
        sliding_window_q,
        sliding_window_k,
    )


def prefix_prefill_attention_precompiled(
        output: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gathered_ctx_k: torch.Tensor,
        gathered_ctx_v: torch.Tensor,
        cu_ctx_lens: torch.Tensor,
        b_start_loc: torch.Tensor,
        b_seq_len: torch.Tensor,
        sm_scale: float,
        sliding_window: int,
        skip_decode: bool,
) -> None:
    torch.ops._C.prefix_prefill_attention_precompiled(
        output,
        q,
        k,
        v,
        gathered_ctx_k,
        gathered_ctx_v,
        cu_ctx_lens,
        b_start_loc,
        b_seq_len,
        sm_scale,
        sliding_window,
        skip_decode,
    )


def pack_seq_precompiled(
        x: torch.Tensor,
        lengths: torch.Tensor,
        pad_value: float = -float("inf"),
) -> torch.Tensor:
    return torch.ops._C.pack_seq_precompiled(
        x,
        lengths,
        pad_value,
    )


def unpack_seq_precompiled(
        packed_tensor: torch.Tensor,
        lengths: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C.unpack_seq_precompiled(
        packed_tensor,
        lengths,
    )


def expand_batch_to_tokens_precompiled(
        x: torch.Tensor,
        cu_num_tokens: torch.Tensor,
        replace_from: int = 0,
        replace_to: int = 0,
) -> torch.Tensor:
    return torch.ops._C.expand_batch_to_tokens_precompiled(
        x,
        cu_num_tokens,
        replace_from,
        replace_to,
    )


def has_precompiled_sample_recovered_tokens() -> bool:
    return hasattr(torch.ops._C, "sample_recovered_tokens_precompiled")


def sample_recovered_tokens_precompiled(
        cu_num_draft_tokens: torch.Tensor,
        draft_token_ids: torch.Tensor,
        draft_probs: torch.Tensor | None,
        target_probs: torch.Tensor,
        inv_q: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._C.sample_recovered_tokens_precompiled(
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,
    )


def has_precompiled_apply_top_k_top_p() -> bool:
    return hasattr(torch.ops._C, "apply_top_k_top_p_precompiled")


def apply_top_k_top_p_precompiled(
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        mask_value: float = -float("inf"),
) -> None:
    torch.ops._C.apply_top_k_top_p_precompiled(
        logits,
        k,
        p,
        mask_value,
    )


def has_precompiled_rejection_greedy_sample() -> bool:
    return hasattr(torch.ops._C, "rejection_greedy_sample_precompiled")


def rejection_greedy_sample_precompiled(
        output_token_ids: torch.Tensor,
        cu_num_draft_tokens: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_argmax: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        is_greedy: torch.Tensor | None,
        max_spec_len: int,
) -> None:
    torch.ops._C.rejection_greedy_sample_precompiled(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy,
        max_spec_len,
    )


def has_precompiled_rejection_random_sample() -> bool:
    return hasattr(torch.ops._C, "rejection_random_sample_precompiled")


def rejection_random_sample_precompiled(
        output_token_ids: torch.Tensor,
        cu_num_draft_tokens: torch.Tensor,
        draft_token_ids: torch.Tensor,
        draft_probs: torch.Tensor | None,
        target_probs: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        recovered_token_ids: torch.Tensor,
        uniform_probs: torch.Tensor,
        is_greedy: torch.Tensor | None,
        max_spec_len: int,
) -> None:
    torch.ops._C.rejection_random_sample_precompiled(
        output_token_ids,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        max_spec_len,
    )


def has_precompiled_input_batch_prepare_prefill_inputs() -> bool:
    return hasattr(torch.ops._C, "input_batch_prepare_prefill_inputs_precompiled")


def input_batch_prepare_prefill_inputs_precompiled(
        input_ids: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        all_token_ids: torch.Tensor,
        prefill_len: torch.Tensor,
        num_computed_tokens: torch.Tensor,
) -> None:
    torch.ops._C.input_batch_prepare_prefill_inputs_precompiled(
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_len,
        num_computed_tokens,
    )


def has_precompiled_input_batch_prepare_pos_seq_lens() -> bool:
    return hasattr(torch.ops._C, "input_batch_prepare_pos_seq_lens_precompiled")


def input_batch_prepare_pos_seq_lens_precompiled(
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        pos: torch.Tensor,
        seq_lens: torch.Tensor,
) -> None:
    torch.ops._C.input_batch_prepare_pos_seq_lens_precompiled(
        idx_mapping,
        query_start_loc,
        num_computed_tokens,
        pos,
        seq_lens,
    )


def has_precompiled_input_batch_combine_sampled_and_draft_tokens() -> bool:
    return hasattr(
        torch.ops._C, "input_batch_combine_sampled_and_draft_tokens_precompiled"
    )


def input_batch_combine_sampled_and_draft_tokens_precompiled(
        input_ids: torch.Tensor,
        idx_mapping: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        prefill_len: torch.Tensor,
        draft_tokens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        num_logits: int,
) -> torch.Tensor:
    return torch.ops._C.input_batch_combine_sampled_and_draft_tokens_precompiled(
        input_ids,
        idx_mapping,
        last_sampled_tokens,
        query_start_loc,
        seq_lens,
        prefill_len,
        draft_tokens,
        cu_num_logits,
        num_logits,
    )


def has_precompiled_input_batch_get_num_sampled_and_rejected() -> bool:
    return hasattr(
        torch.ops._C, "input_batch_get_num_sampled_and_rejected_precompiled"
    )


def input_batch_get_num_sampled_and_rejected_precompiled(
        num_sampled: torch.Tensor,
        seq_lens: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C.input_batch_get_num_sampled_and_rejected_precompiled(
        num_sampled,
        seq_lens,
        cu_num_logits,
        idx_mapping,
        prefill_len,
    )


def has_precompiled_input_batch_post_update() -> bool:
    return hasattr(torch.ops._C, "input_batch_post_update_precompiled")


def input_batch_post_update_precompiled(
        idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        last_sampled_tokens: torch.Tensor,
        output_bin_counts: torch.Tensor | None,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor,
        all_token_ids: torch.Tensor,
        total_len: torch.Tensor,
) -> None:
    torch.ops._C.input_batch_post_update_precompiled(
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        total_len,
    )


def has_precompiled_input_batch_post_update_pool() -> bool:
    return hasattr(torch.ops._C, "input_batch_post_update_pool_precompiled")


def input_batch_post_update_pool_precompiled(
        idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        query_start_loc: torch.Tensor,
) -> None:
    torch.ops._C.input_batch_post_update_pool_precompiled(
        idx_mapping,
        num_computed_tokens,
        query_start_loc,
    )


def has_precompiled_input_batch_expand_idx_mapping() -> bool:
    return hasattr(torch.ops._C, "input_batch_expand_idx_mapping_precompiled")


def input_batch_expand_idx_mapping_precompiled(
        idx_mapping: torch.Tensor,
        total_num_logits: int,
        cu_num_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C.input_batch_expand_idx_mapping_precompiled(
        idx_mapping,
        total_num_logits,
        cu_num_logits,
    )


def has_precompiled_eagle_step_update_slot_mapping_and_metadata() -> bool:
    return hasattr(
        torch.ops._C,
        "eagle_step_update_slot_mapping_and_metadata_precompiled",
    )


def eagle_step_update_slot_mapping_and_metadata_precompiled(
        positions_1d: torch.Tensor,
        block_table_tensor: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        max_model_len: int,
        out_clamped_positions: torch.Tensor,
        out_slot_mapping: torch.Tensor,
        input_batch_size: int,
) -> None:
    torch.ops._C.eagle_step_update_slot_mapping_and_metadata_precompiled(
        positions_1d,
        block_table_tensor,
        seq_lens,
        block_size,
        max_model_len,
        out_clamped_positions,
        out_slot_mapping,
        input_batch_size,
    )


def has_precompiled_eagle_prepare_inputs_padded() -> bool:
    return hasattr(torch.ops._C, "eagle_prepare_inputs_padded_precompiled")


def eagle_prepare_inputs_padded_precompiled(
        cu_num_draft_tokens: torch.Tensor,
        valid_sampled_tokens_count: torch.Tensor,
        query_start_loc_gpu: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        num_rejected_tokens_gpu: torch.Tensor,
) -> None:
    torch.ops._C.eagle_prepare_inputs_padded_precompiled(
        cu_num_draft_tokens,
        valid_sampled_tokens_count,
        query_start_loc_gpu,
        token_indices_to_sample,
        num_rejected_tokens_gpu,
    )


def has_precompiled_eagle_prepare_next_token_padded() -> bool:
    return hasattr(torch.ops._C, "eagle_prepare_next_token_padded_precompiled")


def eagle_prepare_next_token_padded_precompiled(
        sampled_token_ids: torch.Tensor,
        discard_request_mask: torch.Tensor,
        backup_next_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        valid_sampled_tokens_count: torch.Tensor,
        vocab_size: int,
) -> None:
    torch.ops._C.eagle_prepare_next_token_padded_precompiled(
        sampled_token_ids,
        discard_request_mask,
        backup_next_token_ids,
        next_token_ids,
        valid_sampled_tokens_count,
        vocab_size,
    )


def has_precompiled_copy_and_expand_eagle_inputs() -> bool:
    return hasattr(torch.ops._C, "copy_and_expand_eagle_inputs_precompiled")


def copy_and_expand_eagle_inputs_precompiled(
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        next_token_ids: torch.Tensor,
        out_input_ids: torch.Tensor,
        out_positions: torch.Tensor,
        out_is_rejected_token_mask: torch.Tensor,
        out_is_masked_token_mask: torch.Tensor,
        out_new_token_indices: torch.Tensor,
        out_hidden_state_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        query_end_loc: torch.Tensor,
        padding_token_id: int,
        parallel_drafting_token_id: int,
        total_input_tokens: int,
        num_padding_slots_per_request: int,
        shift_input_ids: bool,
) -> None:
    torch.ops._C.copy_and_expand_eagle_inputs_precompiled(
        target_token_ids,
        target_positions,
        next_token_ids,
        out_input_ids,
        out_positions,
        out_is_rejected_token_mask,
        out_is_masked_token_mask,
        out_new_token_indices,
        out_hidden_state_mapping,
        query_start_loc,
        query_end_loc,
        padding_token_id,
        parallel_drafting_token_id,
        total_input_tokens,
        num_padding_slots_per_request,
        shift_input_ids,
    )


def has_precompiled_prepare_eagle_inputs() -> bool:
    return hasattr(torch.ops._C, "prepare_eagle_inputs_precompiled")


def prepare_eagle_inputs_precompiled(
        last_token_indices: torch.Tensor,
        eagle_input_ids: torch.Tensor,
        eagle_positions: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        query_start_loc: torch.Tensor,
) -> None:
    torch.ops._C.prepare_eagle_inputs_precompiled(
        last_token_indices,
        eagle_input_ids,
        eagle_positions,
        target_input_ids,
        target_positions,
        idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
    )


def has_precompiled_prepare_eagle_decode() -> bool:
    return hasattr(torch.ops._C, "prepare_eagle_decode_precompiled")


def prepare_eagle_decode_precompiled(
        draft_tokens: torch.Tensor,
        output_hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        target_seq_lens: torch.Tensor,
        num_rejected: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        input_hidden_states: torch.Tensor,
        max_model_len: int,
        max_num_reqs: int,
) -> None:
    torch.ops._C.prepare_eagle_decode_precompiled(
        draft_tokens,
        output_hidden_states,
        last_token_indices,
        target_seq_lens,
        num_rejected,
        input_ids,
        positions,
        query_start_loc,
        seq_lens,
        input_hidden_states,
        max_model_len,
        max_num_reqs,
    )


def has_precompiled_update_eagle_inputs() -> bool:
    return hasattr(torch.ops._C, "update_eagle_inputs_precompiled")


def update_eagle_inputs_precompiled(
        draft_tokens: torch.Tensor,
        output_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        hidden_states: torch.Tensor,
        max_model_len: int,
) -> None:
    torch.ops._C.update_eagle_inputs_precompiled(
        draft_tokens,
        output_hidden_states,
        input_ids,
        positions,
        seq_lens,
        hidden_states,
        max_model_len,
    )


def apply_repetition_penalties_torch(
        logits: torch.Tensor,
        prompt_mask: torch.Tensor,
        output_mask: torch.Tensor,
        repetition_penalties: torch.Tensor,
) -> None:
    repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(
        1, logits.size(1)
    )
    # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
    penalties = torch.where(prompt_mask | output_mask, repetition_penalties, 1.0)
    # If logits are positive, divide by penalty, otherwise multiply by penalty.
    scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
    logits *= scaling


def apply_repetition_penalties_cuda(
        logits: torch.Tensor,
        prompt_mask: torch.Tensor,
        output_mask: torch.Tensor,
        repetition_penalties: torch.Tensor,
) -> None:
    torch.ops._C.apply_repetition_penalties_(
        logits, prompt_mask, output_mask, repetition_penalties
    )


def apply_repetition_penalties(
        logits: torch.Tensor,
        prompt_mask: torch.Tensor,
        output_mask: torch.Tensor,
        repetition_penalties: torch.Tensor,
) -> None:
    """Apply repetition penalties to logits in-place.

    Args:
        logits: The logits tensor of shape [num_seqs, vocab_size].
        prompt_mask: A boolean tensor indicating which tokens appear in the prompt.
        output_mask: A boolean tensor indicating which tokens appear in the output.
        repetition_penalties: The repetition penalties of shape (num_seqs, ).
    """
    if logits.is_cuda and logits.is_contiguous():
        apply_repetition_penalties_cuda(
            logits, prompt_mask, output_mask, repetition_penalties
        )
    else:
        apply_repetition_penalties_torch(
            logits, prompt_mask, output_mask, repetition_penalties
        )


# fused quant layer norm ops
def rms_norm_dynamic_per_token_quant(
        input: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
        quant_dtype: torch.dtype,
        scale_ub: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty(input.shape, dtype=quant_dtype, device=input.device)
    scales = torch.empty(
        (input.numel() // input.shape[-1], 1), device=input.device, dtype=torch.float32
    )

    torch.ops._C.rms_norm_dynamic_per_token_quant(
        output, input, weight, scales, epsilon, scale_ub, residual
    )
    return output, scales


# fused quant layer norm ops blocked
def rms_norm_per_block_quant(
        input: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_size: list[int],
        scale_ub: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        is_scale_transposed: bool = False,
        tma_alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(group_size) == 2
    output = torch.empty(input.shape, dtype=quant_dtype, device=input.device)
    if is_scale_transposed:
        if tma_alignment == 0:
            scales = torch.empty(
                (input.shape[-1] // group_size[1], input.numel() // input.shape[-1]),
                device=input.device,
                dtype=torch.float32,
            ).transpose(0, 1)
        else:
            m = input.shape[-2]
            sf_k = input.shape[-1] // group_size[1]
            tma_aligned_m = (m + tma_alignment - 1) // tma_alignment * tma_alignment
            shape = input.shape[:-2] + (m, sf_k)
            stride = (
                (1, tma_aligned_m)
                if input.dim() == 2
                else (tma_aligned_m * sf_k, 1, tma_aligned_m)
            )
            scales = torch.empty_strided(
                shape, stride, device=input.device, dtype=torch.float32
            )
    else:
        scales = torch.empty(
            (input.numel() // input.shape[-1], input.shape[-1] // group_size[1]),
            device=input.device,
            dtype=torch.float32,
        )

    assert tma_alignment in [0, 4], "Expected TMA alignment 0 or 4, but got " + str(
        tma_alignment
    )

    torch.ops._C.rms_norm_per_block_quant(
        output,
        input,
        weight,
        scales,
        epsilon,
        scale_ub,
        residual,
        group_size[1],
        is_scale_transposed,
    )
    return output, scales


# quantization ops
# awq
def awq_dequantize(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        split_k_iters: int,
        thx: int,
        thy: int,
) -> torch.Tensor:
    if envs.VLLM_USE_TRITON_AWQ:
        from cfie.model_executor.layers.quantization.awq_triton import (
            awq_dequantize_triton,
        )

        return awq_dequantize_triton(qweight, scales, zeros)
    return torch.ops._C.awq_dequantize(qweight, scales, zeros, split_k_iters, thx, thy)


if hasattr(torch.ops._C, "awq_dequantize"):
    @register_fake("_C::awq_dequantize")
    def _awq_dequantize_fake(
            qweight: torch.Tensor,
            scales: torch.Tensor,
            zeros: torch.Tensor,
            split_k_iters: torch.SymInt,
            thx: int,
            thy: int,
    ) -> torch.Tensor:
        in_c = qweight.size(0)
        qout_c = qweight.size(1)
        out_c = qout_c * 8
        return torch.empty((in_c, out_c), dtype=scales.dtype, device=scales.device)


def awq_gemm(
        input: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        split_k_iters: int,
) -> torch.Tensor:
    if envs.VLLM_USE_TRITON_AWQ:
        from cfie.model_executor.layers.quantization.awq_triton import awq_gemm_triton

        return awq_gemm_triton(input, qweight, scales, qzeros, split_k_iters)
    return torch.ops._C.awq_gemm(input, qweight, scales, qzeros, split_k_iters)


if hasattr(torch.ops._C, "awq_gemm"):
    @register_fake("_C::awq_gemm")
    def _awq_gemm_fake(
            input: torch.Tensor,
            qweight: torch.Tensor,
            scales: torch.Tensor,
            qzeros: torch.Tensor,
            split_k_iters: torch.SymInt,
    ) -> torch.Tensor:
        num_in_feats = input.size(0)
        return torch.empty(
            (split_k_iters, num_in_feats, qweight.size(1) * 8),
            dtype=input.dtype,
            device=input.device,
        ).sum(0)


# gptq
def gptq_gemm(
        a: torch.Tensor,
        b_q_weight: torch.Tensor,
        b_gptq_qzeros: torch.Tensor,
        b_gptq_scales: torch.Tensor,
        b_g_idx: torch.Tensor,
        use_exllama: bool,
        use_v2_format: bool,
        bit: int,
) -> torch.Tensor:
    return torch.ops._C.gptq_gemm(
        a,
        b_q_weight,
        b_gptq_qzeros,
        b_gptq_scales,
        b_g_idx,
        use_exllama,
        use_v2_format,
        bit,
    )


if hasattr(torch.ops._C, "gptq_gemm"):
    @register_fake("_C::gptq_gemm")
    def _gptq_gemm_fake(
            a: torch.Tensor,
            b_q_weight: torch.Tensor,
            b_gptq_qzeros: torch.Tensor,
            b_gptq_scales: torch.Tensor,
            b_g_idx: torch.Tensor,
            use_exllama: bool,
            use_v2_format: bool,
            bit: int,
    ) -> torch.Tensor:
        return torch.empty(
            (a.size(0), b_q_weight.size(1)), dtype=a.dtype, device=a.device
        )


def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor, bit: int) -> None:
    torch.ops._C.gptq_shuffle(q_weight, q_perm, bit)


if hasattr(torch.ops._C, "allspark_w8a16_gemm"):
    @register_fake("_C::allspark_w8a16_gemm")
    def _allspark_w8a16_gemm_fake(
            a: torch.Tensor,
            b_qweight: torch.Tensor,
            b_scales: torch.Tensor,
            b_qzeros: torch.Tensor | None,
            n: torch.SymInt,
            group_size: torch.SymInt,
            sm_count: torch.SymInt,
            sm_version: torch.SymInt,
            CUBLAS_M_THRESHOLD: torch.SymInt,
            has_zp: bool,
            n32k16_reorder: bool,
    ) -> torch.Tensor:
        m = a.size(0)
        return torch.empty((m, n), device=a.device, dtype=a.dtype)

if hasattr(torch.ops._C, "ggml_dequantize"):
    @register_fake("_C::ggml_dequantize")
    def _ggml_dequantize_fake(
            W: torch.Tensor,
            quant_type: int,
            m: torch.SymInt,
            n: torch.SymInt,
            dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return torch.empty((m, n), dtype=torch.float16, device=W.device)


    @register_fake("_C::ggml_mul_mat_vec_a8")
    def _ggml_mul_mat_vec_a8_fake(
            W: torch.Tensor,
            X: torch.Tensor,
            quant_type: int,
            row: torch.SymInt,
    ) -> torch.Tensor:
        return torch.empty((X.shape[0], row), dtype=X.dtype, device=W.device)


    @register_fake("_C::ggml_mul_mat_a8")
    def _ggml_mul_mat_a8_fake(
            W: torch.Tensor,
            X: torch.Tensor,
            quant_type: int,
            row: torch.SymInt,
    ) -> torch.Tensor:
        batch = X.size(0)
        return torch.empty((batch, row), dtype=X.dtype, device=W.device)


    @register_fake("_C::ggml_moe_a8")
    def _ggml_moe_a8_fake(
            X: torch.Tensor,
            W: torch.Tensor,
            sorted_token_ids: torch.Tensor,
            expert_ids: torch.Tensor,
            num_tokens_post_padded: torch.Tensor,
            quant_type: int,
            row: torch.SymInt,
            top_k: torch.SymInt,
            tokens: torch.SymInt,
    ) -> torch.Tensor:
        tokens = X.size(0)
        return torch.empty((tokens * top_k, row), dtype=torch.float16, device=W.device)

if hasattr(torch.ops._C, "ggml_moe_a8_vec"):
    @register_fake("_C::ggml_moe_a8_vec")
    def _ggml_moe_a8_vec_fake(
            X: torch.Tensor,
            W: torch.Tensor,
            topk_ids: torch.Tensor,
            top_k: int,
            quant_type: int,
            row: torch.SymInt,
            tokens: torch.SymInt,
    ) -> torch.Tensor:
        tokens = X.size(0)
        return torch.empty((tokens * top_k, row), dtype=X.dtype, device=W.device)


# cutlass
def cutlass_scaled_mm_supports_fp4(cuda_device_capability: int) -> bool:
    if not hasattr(torch.ops._C, "cutlass_scaled_mm_supports_fp4"):
        return False
    return torch.ops._C.cutlass_scaled_mm_supports_fp4(cuda_device_capability)


def cutlass_scaled_fp4_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        block_scale_a: torch.Tensor,
        block_scale_b: torch.Tensor,
        alpha: torch.Tensor,
        out_dtype: torch.dtype,
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    m, n = a.shape[0], b.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    torch.ops._C.cutlass_scaled_fp4_mm(out, a, b, block_scale_a, block_scale_b, alpha)
    return out


def cutlass_scaled_mm_supports_fp8(cuda_device_capability: int) -> bool:
    if not hasattr(torch.ops._C, "cutlass_scaled_mm_supports_fp8"):
        return False
    return torch.ops._C.cutlass_scaled_mm_supports_fp8(cuda_device_capability)


def cutlass_scaled_mm_supports_block_fp8(cuda_device_capability: int) -> bool:
    if not hasattr(torch.ops._C, "cutlass_scaled_mm_supports_block_fp8"):
        return False
    return torch.ops._C.cutlass_scaled_mm_supports_block_fp8(cuda_device_capability)


def cutlass_scaled_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    `cutlass_scaled_mm` implements a fused version of
        `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
    where scale_a * a and scale_b * b are implemented using numpy-style
    broadcasting.

    In order to support blockwise scaling like found in DeepSeek V3 we also
    support extended "group" broadcast rules. We extend the numpy-style
    broadcasting rules with the following rule:
        "if the extent of a dimension in the source shape is between 1 and
        corresponding extent in the target shape we repeat each element along
        that dimension  src_shape[dim] // target_shape[dim] times consecutively"
    example if we have:
          a = [[1, 2], and target_shape = (2, 4)
               [3, 4]]
    then we would expand a to:
          a = [[1, 1, 2, 2],
               [3, 3, 4, 4]]
    currently we only support the case:
        scale_a.shape * [1, 128] == a.shape
        scale_b.shape * [128, 128] == b.shape
    """
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.numel() == b.shape[1] and bias.dtype == out_dtype

    # Massage the input to be 2D
    target_shape = (*a.shape[:-1], b.shape[1])
    a = a.view(-1, a.shape[-1])

    cutlass_compatible_b = b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    if current_platform.is_rocm() or not cutlass_compatible_b:
        from cfie.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa
            triton_scaled_mm,
        )

        out = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    else:
        out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
        torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

    return out.view(*target_shape)


def cutlass_scaled_mm_azp(
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
        azp_adj: torch.Tensor,
        azp: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    :param azp_adj: In the per-tensor case, this should include the azp.
    Always per-channel.
    :param azp: Only set in the per-token case. Per-token if set.
    """
    assert b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.numel() == b.shape[1] and bias.dtype == out_dtype

    # Massage the input to be 2D
    target_shape = (*a.shape[:-1], b.shape[1])
    a = a.view(-1, a.shape[-1])
    assert azp is None or azp.numel() == a.shape[0]

    out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    torch.ops._C.cutlass_scaled_mm_azp(out, a, b, scale_a, scale_b, azp_adj, azp, bias)
    return out.view(*target_shape)


def cutlass_sparse_scaled_mm_supported(cuda_device_capability: int) -> bool:
    if not hasattr(torch.ops._C, "cutlass_sparse_scaled_mm_supported"):
        return False
    return torch.ops._C.cutlass_sparse_scaled_mm_supported(cuda_device_capability)


def cutlass_group_gemm_supported(cuda_device_capability: int) -> bool:
    if cuda_device_capability < 90 or cuda_device_capability >= 110:
        return False
    if not hasattr(torch.ops._C, "cutlass_group_gemm_supported"):
        return False
    try:
        return torch.ops._C.cutlass_group_gemm_supported(cuda_device_capability)
    except AttributeError:
        # Return False on non-CUDA platforms where it is not available
        return False


def cutlass_sparse_compress(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compresses a sparse matrix for use with Cutlass sparse operations.

    This function takes a dense tensor and compresses it into two components:
    non-zero elements and metadata. The compressed representation is compatible
    with Cutlass sparse kernels.

    Args:
        a (torch.Tensor):
            The input tensor to be compressed. Must have one of the following data types:
            - `torch.int8`
            - `torch.float8_e4m3fn`
            - `torch.bfloat16`
            - `torch.float16`

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            A tuple containing:
            - `a_nzs` (torch.Tensor): A tensor containing non-zero elements of `a`.
            - `a_meta` (torch.Tensor): A tensor containing metadata for the sparse representation.

    Raises:
        ValueError: If the compression operation fails.

    Notes:
        - The `a_meta` tensor has a data type of `torch.uint8`.
        - Each metadata element encodes the sparsity of 4 non-zero elements (i.e., `elemsPerMetaElem = 4`).
        - The shape of `a_nzs` is `(m, k // 2)`, where `m` and `k` are the dimensions of the input tensor.
        - The shape of `a_meta` is `(m, k // 2 // elemsPerMetaElem)`.
    """
    assert a.dtype in [torch.int8, torch.float8_e4m3fn, torch.bfloat16, torch.float16]
    assert a.is_contiguous()

    # a_meta.dtype: torch.uint8 so elemsPerMetaElem = 8b / 2b_per_nz = 4
    elemsPerMetaElem = 4
    assert a.shape[1] % (2 * elemsPerMetaElem) == 0

    return torch.ops._C.cutlass_sparse_compress(a)


def cutlass_scaled_sparse_mm(
        a: torch.Tensor,
        bt_nzs: torch.Tensor,
        bt_meta: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Performs a scaled sparse matrix multiplication using Cutlass.

    Steps:
    1. Create a dense matrix `a` of shape (m, k) on the CUDA device:
    `a = torch.randn((m, k), device='cuda')`.

    2. Create a dense matrix `b` of shape (k, n) on the CUDA device:
    `b = torch.randn((k, n), device='cuda')`.

    3. Prune matrix `b` to 2:4 sparsity along the specified dimension:
    `b = prune_to_2_4(b, dim=0)`.

    4. Compress the transposed sparse matrix `b.t()`:
    `bt_nzs, bt_meta = cutlass_sparse_compress(b.t())`.

    5. Perform sparse matrix multiplication using the compressed matrix,
    applying scaling factors for `a` and `b`, and the output data type:
    `out = cutlass_scaled_sparse_mm(a, bt_nzs, bt_meta, scale_a, scale_b, out_dtype)`.

    Returns:
    - The result of the scaled sparse matrix multiplication.
    """
    assert bt_nzs.shape[0] % 16 == 0 and bt_nzs.shape[1] % 16 == 0
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.shape[0] == bt_nzs.shape[0] and bias.dtype == out_dtype

    m = a.shape[0]
    n = bt_nzs.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    torch.ops._C.cutlass_scaled_sparse_mm(
        out, a, bt_nzs, bt_meta, scale_a, scale_b, bias
    )

    return out


def get_cutlass_moe_mm_data(
        topk_ids: torch.Tensor,
        expert_offsets: torch.Tensor,
        problem_sizes1: torch.Tensor,
        problem_sizes2: torch.Tensor,
        input_permutation: torch.Tensor,
        output_permutation: torch.Tensor,
        num_experts: int,
        n: int,
        k: int,
        blockscale_offsets: torch.Tensor | None = None,
):
    """
    Prepare data necessary to perform CUTLASS grouped matrix multiplications
    used in CUTLASS-based fused MoE.

    The function takes in topk_ids (token-expert mapping) and uses it to
    compute:
    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation after the input is sorted with
                      input_permutation. The number of tokens computed with
                      expert E is expert_offsets[E + 1] - expert_offsets[E]
    - problem_sizes1, problem_sizes2: MxNxK sizes of each expert's
                                      multiplication in two grouped MMs used in
                                      the fused MoE operation.
    - input_permutation: Permutation that must be used to shuffle the input
                         before executing the MMs.
    - output_permutation: Permutation that must be used to shuffle the output
                          after executing the MMs.
    - blockscale_offsets: Optional argument passed for fp4 moe. Indices that
                          mark at which block scale index each expert begins
                          its computation. The number of block scale rows
                          computed with expert E is blockscale_offsets[E + 1] -
                          blockscale_offsets[E]
    """
    return torch.ops._C.get_cutlass_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        num_experts,
        n,
        k,
        blockscale_offsets,
    )


def get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
        expert_first_token_offset: torch.Tensor,
        problem_sizes1: torch.Tensor,
        problem_sizes2: torch.Tensor,
        n: int,
        k: int,
        swap_ab: bool,
):
    """Compute per-expert (M, N, K) problem sizes from expert_first_token_offset"""
    return torch.ops._C.get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
        expert_first_token_offset,
        problem_sizes1,
        problem_sizes2,
        n,
        k,
        swap_ab,
    )


def shuffle_rows(input_tensor: torch.Tensor, dst2src_map: torch.Tensor):
    """
    Shuffle and expand the input tensor according to the dst2src_map and store the result in output_tensor.
    This is used in MoE to permute the input tensor before performing grouped matrix multiplications.
    """
    num_tokens_permuted = dst2src_map.shape[0]
    output_tensor = torch.empty(
        (num_tokens_permuted, input_tensor.shape[1]),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    torch.ops._moe_C.shuffle_rows(input_tensor, dst2src_map, output_tensor)
    return output_tensor


def get_cutlass_batched_moe_mm_data(
        expert_offsets: torch.Tensor,
        problem_sizes1: torch.Tensor,
        problem_sizes2: torch.Tensor,
        expert_num_tokens: torch.Tensor,
        num_local_experts: int,
        padded_m: int,
        n: int,
        k: int,
):
    """
    Prepare data necessary to perform CUTLASS grouped matrix multiplications
    used in CUTLASS-based fused MoE.

    The function takes in expert_num_tokens (token count per expert) and
    non_zero_expert_idxs (consecutive indices of experts with non-zero token
    counts) and uses them to compute:
    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation.
    - problem_sizes1, problem_sizes2: MxNxK sizes of each expert's
                                      multiplication in two grouped MMs used in
                                      the fused MoE operation.
    """
    return torch.ops._C.get_cutlass_batched_moe_mm_data(
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        expert_num_tokens,
        num_local_experts,
        padded_m,
        n,
        k,
    )


def cutlass_moe_mm(
        out_tensors: torch.Tensor,
        a_tensors: torch.Tensor,
        b_tensors: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        expert_offsets: torch.Tensor,
        problem_sizes: torch.Tensor,
        a_strides: torch.Tensor,
        b_strides: torch.Tensor,
        c_strides: torch.Tensor,
        per_act_token: bool,
        per_out_ch: bool,
):
    """
    A single grouped matrix multiplication used in CUTLASS-based fused MoE.
    The function executes fp8-quantized OUT = AB matrix multiplication.

    - expert_offsets: Indices that mark at which token index each expert begins
                      its computation. The number of tokens computed with
                      expert E is expert_offsets[E + 1] - expert_offsets[E]
    - problem_sizes: MxNxK sizes of each expert's multiplication in two grouped
                     MMs used in the fused MoE operation.
    - a/b/c_strides: The data strides passed to grouped matrix multiplication.
    """
    return torch.ops._C.cutlass_moe_mm(
        out_tensors,
        a_tensors,
        b_tensors,
        a_scales,
        b_scales,
        expert_offsets,
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        per_act_token,
        per_out_ch,
    )


def cutlass_fp4_moe_mm(
        out_tensors: torch.Tensor,
        a_tensors: torch.Tensor,
        b_tensors: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        alphas: torch.Tensor,
        problem_sizes: torch.Tensor,
        expert_offsets: torch.Tensor,
        sf_offsets: torch.Tensor,
):
    """
    An FP4 Blockscaled Group Gemm that takes in  a_tensors, b_tensors and runs
    the gemms for each combination based on the specified problem sizes.

    This is used as the MoE gemm during NVFP4 Quantized FusedMoE forward.
    - a/b_tensors: the NVFP4 a_ptrs and b_ptrs tensors which are quantized
                     input and expert weights.
    - a_/b_scales: The blockscales in FP8-E4M3 precision
    - expert_offsets/sf_offsets: Indices that mark at which token index
                    each expert begins its computation. The number of tokens
                    computed with expert E is expert_offsets[E + 1] -
                    expert_offsets[E] And the sf_size per expert is
                    sf_offset[E+1] - sf_offset[E]
    - problem_sizes: MxNxK sizes of each expert's multiplication in two grouped
                     MMs used in the fused MoE operation.
    """
    return torch.ops._C.cutlass_fp4_group_mm(
        out_tensors,
        a_tensors,
        b_tensors,
        a_scales,
        b_scales,
        alphas,
        problem_sizes,
        expert_offsets,
        sf_offsets,
    )


def mxfp8_experts_quant(
        input_tensor: torch.Tensor,
        problem_sizes: torch.Tensor,
        expert_offsets: torch.Tensor,
        blockscale_offsets: torch.Tensor,
        quant_output: torch.Tensor,
        scale_factor: torch.Tensor,
) -> None:
    torch.ops._C.mxfp8_experts_quant(
        input_tensor,
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
        quant_output,
        scale_factor,
    )


def cutlass_mxfp8_grouped_mm(
        a_tensors: torch.Tensor,
        b_tensors: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        out_tensors: torch.Tensor,
        problem_sizes: torch.Tensor,
        expert_offsets: torch.Tensor,
        blockscale_offsets: torch.Tensor,
) -> None:
    torch.ops._C.cutlass_mxfp8_grouped_mm(
        a_tensors,
        b_tensors,
        a_scales,
        b_scales,
        out_tensors,
        problem_sizes,
        expert_offsets,
        blockscale_offsets,
    )


if hasattr(torch.ops._C, "mxfp8_experts_quant"):
    @register_fake("_C::mxfp8_experts_quant")
    def _mxfp8_experts_quant_fake(
            input_tensor: torch.Tensor,
            problem_sizes: torch.Tensor,
            expert_offsets: torch.Tensor,
            blockscale_offsets: torch.Tensor,
            quant_output: torch.Tensor,
            scale_factor: torch.Tensor,
    ) -> None:
        return None

if hasattr(torch.ops._C, "cutlass_mxfp8_grouped_mm"):
    @register_fake("_C::cutlass_mxfp8_grouped_mm")
    def _cutlass_mxfp8_grouped_mm_fake(
            a_tensors: torch.Tensor,
            b_tensors: torch.Tensor,
            a_scales: torch.Tensor,
            b_scales: torch.Tensor,
            out_tensors: torch.Tensor,
            problem_sizes: torch.Tensor,
            expert_offsets: torch.Tensor,
            blockscale_offsets: torch.Tensor,
    ) -> None:
        return None


# gptq_marlin
def _get_torch_op(namespace: str, op_name: str):
    try:
        return getattr(getattr(torch.ops, namespace), op_name)
    except AttributeError:
        return None


_FLOAT8_DTYPES = {
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
    return dtype in _FLOAT8_DTYPES


def _extract_scale_scalar(scale: torch.Tensor | None, *, device: torch.device) -> float:
    if scale is None:
        return 1.0
    return float(scale.to(device=device, dtype=torch.float32).reshape(-1)[0].item())


def _prepare_kv_cache_values_for_store(
        tensor: torch.Tensor,
        *,
        cache_dtype: torch.dtype,
        kv_cache_dtype: str,
        scale: torch.Tensor | None,
) -> torch.Tensor:
    if kv_cache_dtype.startswith("fp8"):
        if cache_dtype == torch.uint8:
            raise RuntimeError(
                "Torch KV-cache fallback does not support uint8-backed FP8 "
                "cache tensors."
            )
        if not _is_fp8_dtype(tensor.dtype):
            scale_scalar = _extract_scale_scalar(scale, device=tensor.device)
            tensor = tensor if scale_scalar == 1.0 else tensor / scale_scalar
        return tensor.to(dtype=cache_dtype)
    return tensor.to(dtype=cache_dtype) if tensor.dtype != cache_dtype else tensor


def _reshape_and_cache_4d_torch_fallback(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor | None,
        v_scale: torch.Tensor | None,
) -> None:
    slots = slot_mapping.reshape(-1).to(dtype=torch.long)
    valid_mask = slots >= 0
    if not torch.any(valid_mask):
        return

    token_indices = valid_mask.nonzero(as_tuple=False).flatten()
    slots = slots[token_indices]
    block_size = key_cache.shape[1]
    block_indices = torch.div(slots, block_size, rounding_mode="floor")
    block_offsets = torch.remainder(slots, block_size)

    key_values = _prepare_kv_cache_values_for_store(
        key.index_select(0, token_indices),
        cache_dtype=key_cache.dtype,
        kv_cache_dtype=kv_cache_dtype,
        scale=k_scale,
    )
    value_values = _prepare_kv_cache_values_for_store(
        value.index_select(0, token_indices),
        cache_dtype=value_cache.dtype,
        kv_cache_dtype=kv_cache_dtype,
        scale=v_scale,
    )

    key_cache[block_indices, block_offsets] = key_values
    value_cache[block_indices, block_offsets] = value_values


def _reshape_and_cache_head_major_torch_fallback(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor | None,
        v_scale: torch.Tensor | None,
) -> None:
    slots = slot_mapping.reshape(-1).to(dtype=torch.long)
    valid_mask = slots >= 0
    if not torch.any(valid_mask):
        return

    token_indices = valid_mask.nonzero(as_tuple=False).flatten()
    slots = slots[token_indices]
    block_size = key_cache.shape[3]
    pack_x = key_cache.shape[4]
    block_indices = torch.div(slots, block_size, rounding_mode="floor")
    block_offsets = torch.remainder(slots, block_size)

    key_values = _prepare_kv_cache_values_for_store(
        key.index_select(0, token_indices),
        cache_dtype=key_cache.dtype,
        kv_cache_dtype=kv_cache_dtype,
        scale=k_scale,
    )
    value_values = _prepare_kv_cache_values_for_store(
        value.index_select(0, token_indices),
        cache_dtype=value_cache.dtype,
        kv_cache_dtype=kv_cache_dtype,
        scale=v_scale,
    )

    if key_values.shape[-1] % pack_x != 0:
        raise RuntimeError(
            "Torch KV-cache fallback requires head-major key cache width to be "
            f"divisible by pack size {pack_x}, got {key_values.shape[-1]}."
        )

    key_values = key_values.reshape(
        key_values.shape[0],
        key_values.shape[1],
        key_values.shape[2] // pack_x,
        pack_x,
    )

    key_cache[block_indices, :, :, block_offsets, :] = key_values
    value_cache[block_indices, :, :, block_offsets] = value_values


def _reshape_and_cache_torch_fallback(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor | None,
        v_scale: torch.Tensor | None,
        *,
        op_name: str,
) -> None:
    logger.warning_once(
        "Falling back to torch %s because `_C_cache_ops.%s` is unavailable. "
        "This keeps Windows inference startup unblocked but may reduce KV "
        "cache update throughput.",
        op_name,
        op_name,
    )
    if key_cache.ndim == 5:
        _reshape_and_cache_head_major_torch_fallback(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        return
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise RuntimeError(
            f"Torch fallback for {op_name} only supports 4D flash KV cache or "
            "5D head-major key cache layouts."
        )
    _reshape_and_cache_4d_torch_fallback(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def _reshape_and_cache_flash_diffkv_torch_fallback(
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor | None,
        v_scale: torch.Tensor | None,
) -> None:
    logger.warning_once(
        "Falling back to torch reshape_and_cache_flash_diffkv because "
        "`_C_cache_ops.reshape_and_cache_flash_diffkv` is unavailable. "
        "This keeps Windows inference startup unblocked but may reduce KV "
        "cache update throughput.",
    )
    if kv_cache.ndim != 4:
        raise RuntimeError(
            "Torch fallback for reshape_and_cache_flash_diffkv only supports "
            "4D flash KV cache layouts."
        )
    key_width = key.shape[-1]
    value_width = value.shape[-1]
    key_cache = kv_cache[..., :key_width]
    value_cache = kv_cache[..., key_width: key_width + value_width]
    _reshape_and_cache_4d_torch_fallback(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def _pack_repacked_marlin_values(values: torch.Tensor, num_bits: int) -> torch.Tensor:
    shifts = (
            torch.arange(values.shape[-1], device=values.device, dtype=torch.int64)
            * num_bits
    )
    shifts = shifts.view(*([1] * (values.ndim - 1)), -1)
    return ((values.to(torch.int64) << shifts).sum(dim=-1)).to(torch.int32)


def _unpack_row_packed_qweight(
        packed_qweight: torch.Tensor,
        num_bits: int,
        size_k: int,
        size_n: int,
) -> torch.Tensor:
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    packed = packed_qweight.to(torch.int64) & 0xFFFFFFFF
    shifts = (
            torch.arange(pack_factor, device=packed_qweight.device, dtype=torch.int64)
            * num_bits
    )
    unpacked = ((packed.unsqueeze(1) >> shifts.view(1, pack_factor, 1)) & mask)
    return unpacked.reshape((size_k, size_n)).to(torch.int32)


def _unpack_col_packed_qweight(
        packed_qweight: torch.Tensor,
        num_bits: int,
        size_k: int,
        size_n: int,
) -> torch.Tensor:
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    packed = packed_qweight.to(torch.int64) & 0xFFFFFFFF
    shifts = (
            torch.arange(pack_factor, device=packed_qweight.device, dtype=torch.int64)
            * num_bits
    )
    unpacked = ((packed.unsqueeze(-1) >> shifts.view(1, 1, pack_factor)) & mask)
    return unpacked.reshape((size_k, size_n)).to(torch.int32)


def _repack_logical_qweight_to_marlin(
        logical_qweight: torch.Tensor,
        num_bits: int,
        is_a_8bit: bool,
) -> torch.Tensor:
    pack_factor = 32 // num_bits
    size_k, size_n = logical_qweight.shape
    tile_k = 32 if is_a_8bit else 16
    tile_n = 32 if is_a_8bit else 64
    k_tiles = size_k // tile_k
    n_tiles = size_n // tile_n
    tile_size = tile_k * tile_n // pack_factor
    out_tiles = torch.empty(
        (k_tiles, n_tiles, tile_size),
        dtype=torch.int32,
        device=logical_qweight.device,
    )
    tile_chunk_size = 8

    for start in range(0, n_tiles, tile_chunk_size):
        stop = min(start + tile_chunk_size, n_tiles)
        tile_count = stop - start
        q_chunk = logical_qweight[:, start * tile_n: stop * tile_n].contiguous()

        if not is_a_8bit:
            q_chunk = q_chunk.reshape((k_tiles, 16, tile_count, 64))
            q_chunk = q_chunk.reshape((k_tiles, 2, 8, tile_count, 64))
            q_chunk = q_chunk.reshape((k_tiles, 2, 4, 2, tile_count, 64))
            q_chunk = q_chunk.reshape((k_tiles, 2, 4, 2, tile_count, 4, 16))
            q_chunk = q_chunk.reshape((k_tiles, 2, 4, 2, tile_count, 4, 2, 8))

            if num_bits == 4:
                packed_vals = torch.stack(
                    [
                        q_chunk[:, 0, :, 0, :, :, 0, :],
                        q_chunk[:, 1, :, 0, :, :, 0, :],
                        q_chunk[:, 0, :, 0, :, :, 1, :],
                        q_chunk[:, 1, :, 0, :, :, 1, :],
                        q_chunk[:, 0, :, 1, :, :, 0, :],
                        q_chunk[:, 1, :, 1, :, :, 0, :],
                        q_chunk[:, 0, :, 1, :, :, 1, :],
                        q_chunk[:, 1, :, 1, :, :, 1, :],
                    ],
                    dim=-1,
                ).permute(0, 2, 4, 1, 3, 5)
            else:
                packed_vals_0 = torch.stack(
                    [
                        q_chunk[:, 0, :, 0, :, :, 0, :],
                        q_chunk[:, 1, :, 0, :, :, 0, :],
                        q_chunk[:, 0, :, 1, :, :, 0, :],
                        q_chunk[:, 1, :, 1, :, :, 0, :],
                    ],
                    dim=-1,
                )
                packed_vals_1 = torch.stack(
                    [
                        q_chunk[:, 0, :, 0, :, :, 1, :],
                        q_chunk[:, 1, :, 0, :, :, 1, :],
                        q_chunk[:, 0, :, 1, :, :, 1, :],
                        q_chunk[:, 1, :, 1, :, :, 1, :],
                    ],
                    dim=-1,
                )
                packed_vals = torch.stack([packed_vals_0, packed_vals_1], dim=-2)
                packed_vals = packed_vals.permute(0, 2, 4, 1, 3, 5, 6)
        else:
            q_chunk = q_chunk.reshape((k_tiles, 32, tile_count, 32))
            q_chunk = q_chunk.reshape((k_tiles, 2, 16, tile_count, 32))
            q_chunk = q_chunk.reshape((k_tiles, 2, 4, 4, tile_count, 32))
            q_chunk = q_chunk.reshape((k_tiles, 2, 4, 4, tile_count, 2, 16))
            q_chunk = q_chunk.reshape((k_tiles, 2, 4, 4, tile_count, 2, 2, 8))

            if num_bits == 4:
                packed_vals = torch.stack(
                    [
                        q_chunk[:, 0, :, 0, :, :, :, :],
                        q_chunk[:, 1, :, 0, :, :, :, :],
                        q_chunk[:, 0, :, 1, :, :, :, :],
                        q_chunk[:, 1, :, 1, :, :, :, :],
                        q_chunk[:, 0, :, 2, :, :, :, :],
                        q_chunk[:, 1, :, 2, :, :, :, :],
                        q_chunk[:, 0, :, 3, :, :, :, :],
                        q_chunk[:, 1, :, 3, :, :, :, :],
                    ],
                    dim=-1,
                ).permute(0, 2, 5, 1, 3, 4, 6)
            else:
                packed_top = torch.stack(
                    [
                        q_chunk[:, 0, :, 0, :, :, :, :],
                        q_chunk[:, 0, :, 1, :, :, :, :],
                        q_chunk[:, 0, :, 2, :, :, :, :],
                        q_chunk[:, 0, :, 3, :, :, :, :],
                    ],
                    dim=-1,
                )
                packed_bottom = torch.stack(
                    [
                        q_chunk[:, 1, :, 0, :, :, :, :],
                        q_chunk[:, 1, :, 1, :, :, :, :],
                        q_chunk[:, 1, :, 2, :, :, :, :],
                        q_chunk[:, 1, :, 3, :, :, :, :],
                    ],
                    dim=-1,
                )
                packed_vals = torch.stack([packed_top, packed_bottom], dim=-2)
                packed_vals = packed_vals.permute(0, 2, 5, 1, 3, 4, 6, 7)

        out_tiles[:, start:stop, :] = _pack_repacked_marlin_values(
            packed_vals, num_bits
        ).reshape((k_tiles, tile_count, tile_size))

    return out_tiles.reshape((size_k // 16, size_n * 16 // pack_factor))


def _gptq_marlin_repack_torch_fallback(
        b_q_weight: torch.Tensor,
        perm: torch.Tensor,
        size_k: int,
        size_n: int,
        num_bits: int,
        is_a_8bit: bool = False,
) -> torch.Tensor:
    if perm.numel() and is_a_8bit:
        raise RuntimeError(
            "Torch fallback for gptq_marlin_repack does not support "
            "permuted K ordering with 8-bit activations."
        )

    logger.warning_once(
        "Falling back to torch gptq_marlin_repack because `_C."
        "gptq_marlin_repack` is unavailable. This keeps Windows startup "
        "unblocked but may slow checkpoint repacking."
    )
    logical_qweight = _unpack_row_packed_qweight(b_q_weight, num_bits, size_k, size_n)
    if perm.numel():
        logical_qweight = logical_qweight.index_select(
            0, perm.to(device=logical_qweight.device, dtype=torch.long)
        )
    return _repack_logical_qweight_to_marlin(
        logical_qweight, num_bits, is_a_8bit
    ).to(b_q_weight.dtype)


def _awq_marlin_repack_torch_fallback(
        b_q_weight: torch.Tensor,
        size_k: int,
        size_n: int,
        num_bits: int,
        is_a_8bit: bool = False,
) -> torch.Tensor:
    logger.warning_once(
        "Falling back to torch awq_marlin_repack because `_C."
        "awq_marlin_repack` is unavailable. This keeps Windows startup "
        "unblocked but may slow checkpoint repacking."
    )
    logical_qweight = _unpack_col_packed_qweight(b_q_weight, num_bits, size_k, size_n)
    undo_interleave = (
        [0, 4, 1, 5, 2, 6, 3, 7] if num_bits == 4 else [0, 2, 1, 3]
    )
    logical_qweight = logical_qweight.reshape((-1, len(undo_interleave)))
    logical_qweight = logical_qweight.index_select(
        1,
        torch.tensor(
            undo_interleave, device=logical_qweight.device, dtype=torch.long
        ),
    ).reshape((size_k, size_n))
    return _repack_logical_qweight_to_marlin(
        logical_qweight, num_bits, is_a_8bit
    ).to(b_q_weight.dtype)


def gptq_marlin_repack(
        b_q_weight: torch.Tensor,
        perm: torch.Tensor,
        size_k: int,
        size_n: int,
        num_bits: int,
        is_a_8bit: bool = False,
) -> torch.Tensor:
    # ------------------------------- 规范化输入尺寸与量化参数类型 -------------------------------
    # 将输入的 K 维大小显式转换为 Python int，避免后续底层调用时类型不一致。
    size_k = int(size_k)

    # 将输入的 N 维大小显式转换为 Python int，保证后续传给底层实现的参数类型稳定。
    size_n = int(size_n)

    # 将量化 bit 数显式转换为 Python int，便于统一传给底层实现。
    num_bits = int(num_bits)

    # ------------------------------- 优先尝试调用已注册的 C++ 扩展算子 -------------------------------
    # 从 _C 扩展模块中查找名为 gptq_marlin_repack 的 torch op。
    op = _get_torch_op("_C", "gptq_marlin_repack")

    # 当底层 C++ 扩展算子存在时，优先走高性能实现路径。
    if op is not None:
        # 调用已注册的 C++ 扩展算子完成 GPTQ Marlin 格式重排。
        return op(b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit)

    # ------------------------------- 在无扩展算子时退回到纯 Torch 实现 -------------------------------
    # 当当前环境没有可用的 C++ 扩展算子时，退回到 Torch fallback 实现。
    return _gptq_marlin_repack_torch_fallback(
        b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit
    )


if hasattr(torch.ops._C, "gptq_marlin_repack"):
    @register_fake("_C::gptq_marlin_repack")
    def _gptq_marlin_repack_fake(
            b_q_weight: torch.Tensor,
            perm: torch.Tensor,
            size_k: torch.SymInt,
            size_n: torch.SymInt,
            num_bits: int,
            is_a_8bit: bool = False,
    ) -> torch.Tensor:
        pack_factor = 32 // num_bits
        marlin_tile_size = 16
        return torch.empty(
            (size_k // marlin_tile_size, size_n * marlin_tile_size // pack_factor),
            dtype=b_q_weight.dtype,
            device=b_q_weight.device,
        )


# awq_marlin
def awq_marlin_repack(
        b_q_weight: torch.Tensor,
        size_k: int,
        size_n: int,
        num_bits: int,
        is_a_8bit: bool = False,
) -> torch.Tensor:
    size_k = int(size_k)
    size_n = int(size_n)
    num_bits = int(num_bits)
    op = _get_torch_op("_C", "awq_marlin_repack")
    if op is not None:
        return op(b_q_weight, size_k, size_n, num_bits, is_a_8bit)
    return _awq_marlin_repack_torch_fallback(
        b_q_weight, size_k, size_n, num_bits, is_a_8bit
    )


if hasattr(torch.ops._C, "awq_marlin_repack"):
    @register_fake("_C::awq_marlin_repack")
    def _awq_marlin_repack_fake(
            b_q_weight: torch.Tensor,
            size_k: torch.SymInt,
            size_n: torch.SymInt,
            num_bits: int,
            is_a_8bit: bool = False,
    ) -> torch.Tensor:
        pack_factor = 32 // num_bits
        marlin_tile_size = 16
        return torch.empty(
            (size_k // marlin_tile_size, size_n * marlin_tile_size // pack_factor),
            dtype=b_q_weight.dtype,
            device=b_q_weight.device,
        )


def gptq_marlin_moe_repack(
        b_q_weight: torch.Tensor,
        perm: torch.Tensor,
        size_k: int,
        size_n: int,
        num_bits: int,
        is_a_8bit: bool = False,
) -> torch.Tensor:
    # ------------------------------- 规范化输入尺寸参数并读取 expert 数量 -------------------------------
    # 将输入的 K 维大小显式转换为 Python int，避免后续参与张量形状计算时类型不一致。
    size_k = int(size_k)

    # 将输入的 N 维大小显式转换为 Python int，便于后续构造输出张量形状。
    size_n = int(size_n)

    # 将量化 bit 数显式转换为 Python int，确保后续形状计算与底层接口调用一致。
    num_bits = int(num_bits)

    # 读取当前批量量化权重中包含的 expert 数量。
    num_experts = b_q_weight.shape[0]

    # ------------------------------- 校验输入权重是否满足 Marlin repack 的块对齐要求 -------------------------------
    # Marlin repack 要求 size_k 必须能被 16 整除，否则无法按内核要求的 tile 方式重排。
    assert size_k % 16 == 0

    # ------------------------------- 预分配批量 repack 后的输出张量 -------------------------------
    """
    qweight原先按行压缩，按行取16个元素，则实际跨越 16/(32/num_bits) = (1 / 2) * num_bits = num_bits / 2 
    """
    # 为所有 experts 预先分配输出张量；输出形状中的第二维按 16 行一组压缩，
    # 第三维按 num_bits 对列方向进行展开，作为 Marlin kernel 直接消费的布局。
    output = torch.empty(
        # 相当于把行方向的16一组 沿着列方向展开
        (num_experts, size_k // 16, size_n * (num_bits // 2)),
        device=b_q_weight.device,
        dtype=b_q_weight.dtype,
    )

    # ------------------------------- 逐个 expert 调用单 expert repack 逻辑 -------------------------------
    # 遍历每个 expert，并对该 expert 的量化权重与对应排列索引执行单独的 Marlin repack。
    for e in range(num_experts):
        # 将第 e 个 expert 的量化权重按其对应的 perm 索引重排，并写入输出张量。
        output[e] = gptq_marlin_repack(
            b_q_weight[e],
            perm[e],
            size_k,
            size_n,
            num_bits,
            is_a_8bit,
        )

    # ------------------------------- 返回批量 expert 的 repack 结果 -------------------------------
    # 返回所有 experts 完成 Marlin 格式重排后的批量输出张量。
    return output


def awq_marlin_moe_repack(
        b_q_weight: torch.Tensor,
        perm: torch.Tensor,
        size_k: int,
        size_n: int,
        num_bits: int,
        is_a_8bit: bool = False,
) -> torch.Tensor:
    size_k = int(size_k)
    size_n = int(size_n)
    num_bits = int(num_bits)
    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = torch.empty(
        (num_experts, size_k // 16, size_n * (num_bits // 2)),
        device=b_q_weight.device,
        dtype=b_q_weight.dtype,
    )
    for e in range(num_experts):
        output[e] = awq_marlin_repack(
            b_q_weight[e], size_k, size_n, num_bits, is_a_8bit
        )
    return output


def marlin_int4_fp8_preprocess(
        qweight: torch.Tensor,
        qzeros_or_none: torch.Tensor | None = None,
        inplace: bool = False,
):
    return torch.ops._C.marlin_int4_fp8_preprocess(qweight, qzeros_or_none, inplace)


def marlin_gemm(
        a: torch.Tensor,
        c: torch.Tensor | None,
        b_q_weight: torch.Tensor,
        b_bias: torch.Tensor | None,
        b_scales: torch.Tensor,
        a_scales: torch.Tensor | None,
        global_scale: torch.Tensor | None,
        b_zeros: torch.Tensor | None,
        g_idx: torch.Tensor | None,
        perm: torch.Tensor | None,
        workspace: torch.Tensor,
        b_q_type: ScalarType,
        size_m: int,
        size_n: int,
        size_k: int,
        is_k_full: bool = True,
        use_atomic_add: bool = False,
        use_fp32_reduce: bool = False,
        is_zp_float: bool = False,
) -> torch.Tensor:
    return torch.ops._C.marlin_gemm(
        a,
        c,
        b_q_weight,
        b_bias,
        b_scales,
        a_scales,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        workspace,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )


if hasattr(torch.ops._C, "marlin_gemm"):

    @register_fake("_C::marlin_gemm")
    def _marlin_gemm_fake(
            a: torch.Tensor,
            c: torch.Tensor | None,
            b_q_weight: torch.Tensor,
            b_bias: torch.Tensor | None,
            b_scales: torch.Tensor,
            a_scales: torch.Tensor | None,
            global_scale: torch.Tensor | None,
            b_zeros: torch.Tensor | None,
            g_idx: torch.Tensor | None,
            perm: torch.Tensor | None,
            workspace: torch.Tensor,
            b_q_type_id: int,
            size_m: torch.SymInt,
            size_n: torch.SymInt,
            size_k: torch.SymInt,
            is_k_full: bool = True,
            use_atomic_add: bool = False,
            use_fp32_reduce: bool = False,
            is_zp_float: bool = False,
    ) -> torch.Tensor:
        dtype = a.dtype
        if dtype not in [torch.half, torch.bfloat16]:
            dtype = b_scales.dtype
        return torch.empty((size_m, size_n), device=a.device, dtype=dtype)


# machete
def machete_supported_schedules(
        a_type: torch.dtype,
        b_type: ScalarType,
        group_scales_type: torch.dtype | None,
        group_zeros_type: torch.dtype | None = None,
        channel_scales_type: torch.dtype | None = None,
        token_scales_type: torch.dtype | None = None,
        out_type: torch.dtype | None = None,
) -> list[str]:
    return torch.ops._C.machete_supported_schedules(
        a_type,
        b_type.id,
        group_scales_type,
        group_zeros_type,
        channel_scales_type,
        token_scales_type,
        out_type,
    )


def machete_mm(
        a: torch.Tensor,
        # b_q Should be the tensor returned by machete_prepack_B
        b_q: torch.Tensor,
        b_type: ScalarType,
        out_type: torch.dtype | None = None,
        b_group_scales: torch.Tensor | None = None,
        b_group_zeros: torch.Tensor | None = None,
        b_group_size: int | None = None,
        b_channel_scales: torch.Tensor | None = None,
        a_token_scales: torch.Tensor | None = None,
        schedule: str | None = None,
) -> torch.Tensor:
    return torch.ops._C.machete_mm(
        a,
        b_q,
        b_type.id,
        out_type,
        b_group_scales,
        b_group_zeros,
        b_group_size,
        b_channel_scales,
        a_token_scales,
        schedule,
    )


if hasattr(torch.ops._C, "machete_mm"):
    @register_fake("_C::machete_mm")
    def machete_mm_fake(
            a: torch.Tensor,
            # b_q Should be the tensor returned by machete_prepack_B
            b_q: torch.Tensor,
            b_type: ScalarType,
            out_type: torch.dtype | None = None,
            b_group_scales: torch.Tensor | None = None,
            b_group_zeros: torch.Tensor | None = None,
            b_group_size: int | None = None,
            b_channel_scales: torch.Tensor | None = None,
            a_token_scales: torch.Tensor | None = None,
            schedule: str | None = None,
    ) -> torch.Tensor:
        m = a.size(0)
        n = b_q.size(1)
        return torch.empty((m, n), device=a.device, dtype=a.dtype)


def machete_prepack_B(
        b_q_weight: torch.Tensor,
        a_type: torch.dtype,
        b_type: ScalarType,
        group_scales_type: torch.dtype | None,
) -> torch.Tensor:
    return torch.ops._C.machete_prepack_B(
        b_q_weight, a_type, b_type.id, group_scales_type
    )


if hasattr(torch.ops._C, "machete_prepack_B"):
    @register_fake("_C::machete_prepack_B")
    def machete_prepack_B_fake(
            b_q_weight: torch.Tensor,
            a_type: torch.dtype,
            b_type: ScalarType,
            group_scales_type: torch.dtype | None,
    ) -> torch.Tensor:
        return torch.empty_like(b_q_weight, memory_format=torch.contiguous_format)


# CUTLASS W4A8
def cutlass_w4a8_mm(
        a: torch.Tensor,
        # b_q Should be the tensor returned by cutlass_encode_and_reorder_int4b
        b_q: torch.Tensor,
        b_group_scales: torch.Tensor,
        b_group_size: int,
        b_channel_scales: torch.Tensor,
        a_token_scales: torch.Tensor,
        out_type: torch.dtype | None = None,
        maybe_schedule: str | None = None,
) -> torch.Tensor:
    return torch.ops._C.cutlass_w4a8_mm(
        a,
        b_q,
        b_group_scales,
        b_group_size,
        b_channel_scales,
        a_token_scales,
        out_type,
        maybe_schedule,
    )


if hasattr(torch.ops._C, "cutlass_w4a8_mm"):
    @register_fake("_C::cutlass_w4a8_mm")
    def cutlass_w4a8_mm_fake(
            a: torch.Tensor,
            # b_q Should be the tensor returned by cutlass_encode_and_reorder_int4b
            b_q: torch.Tensor,
            b_group_scales: torch.Tensor,
            b_group_size: int,
            b_channel_scales: torch.Tensor,
            a_token_scales: torch.Tensor,
            out_type: torch.dtype | None = None,
            maybe_schedule: str | None = None,
    ) -> torch.Tensor:
        m = a.size(0)
        n = b_q.size(1)
        out_dtype = out_type if out_type is not None else torch.bfloat16
        return torch.empty((m, n), device=a.device, dtype=out_dtype)


def cutlass_pack_scale_fp8(scales: torch.Tensor) -> torch.Tensor:
    return torch.ops._C.cutlass_pack_scale_fp8(scales)


if hasattr(torch.ops._C, "cutlass_pack_scale_fp8"):
    @register_fake("_C::cutlass_pack_scale_fp8")
    def cutlass_pack_scale_fp8_fake(scales: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(scales, memory_format=torch.contiguous_format)


def cutlass_encode_and_reorder_int4b(b: torch.Tensor) -> torch.Tensor:
    return torch.ops._C.cutlass_encode_and_reorder_int4b(b)


if hasattr(torch.ops._C, "cutlass_encode_and_reorder_int4b"):
    @register_fake("_C::cutlass_encode_and_reorder_int4b")
    def cutlass_encode_and_reorder_int4b_fake(b: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(b, memory_format=torch.contiguous_format)


def cutlass_w4a8_moe_mm(
        out_tensors: torch.Tensor,
        a_tensors: torch.Tensor,
        b_tensors: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        b_group_scales: torch.Tensor,
        b_group_size: int,
        expert_offsets: torch.Tensor,
        problem_sizes: torch.Tensor,
        a_strides: torch.Tensor,
        b_strides: torch.Tensor,
        c_strides: torch.Tensor,
        group_scale_strides: torch.Tensor,
        maybe_schedule: str | None = None,
):
    """
    Executes the CUTLASS-based fused-MoE grouped matrix multiplication for the
    W4A8 quantization scheme. Uses group-wise quantization (INT4 -> FP8)
    and both per-channel + per-token scaling in the epilogue.

    Args:
        out_tensors:
            Output buffer for all experts (updated in-place).
        a_tensors:
            FP8 (E4M3FN) activations for all experts.
        b_tensors:
            INT4-packed weight matrix for all experts, packed to INT32
        a_scales:
            Per-token FP8 activation scales, applied in the epilogue.
        b_scales:
            Per-channel FP8 weight scales for each expert, applied in the epilogue.
        b_group_scales:
            FP8 scale values for group-wise INT4 weight blocks.
        b_group_size:
            Number of elements grouped under each entry of b_group_scales.
        expert_offsets:
            Cumulative token offsets
        problem_sizes:
            Per-expert (M, N, K) GEMM sizes used by the grouped GEMM launcher.
        a/b/c/group_scale_strides:
            Strides describing the memory layout of the input tensors.
        maybe_schedule:
            Optional override to choose a specific kernel or epilogue schedule.

    Returns:
        out_tensors updated in-place with the dequantized INT4xFP8 grouped GEMM result.
    """
    return torch.ops._C.cutlass_w4a8_moe_mm(
        out_tensors,
        a_tensors,
        b_tensors,
        a_scales,
        b_scales,
        b_group_scales,
        b_group_size,
        expert_offsets,
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        group_scale_strides,
        maybe_schedule,
    )


def cutlass_encode_and_reorder_int4b_grouped(
        b_tensors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._C.cutlass_encode_and_reorder_int4b_grouped(b_tensors)


if hasattr(torch.ops._C, "cutlass_encode_and_reorder_int4b_grouped"):
    @register_fake("_C::cutlass_encode_and_reorder_int4b_grouped")
    def cutlass_encode_and_reorder_int4b_grouped_fake(b: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(b, memory_format=torch.contiguous_format)


def permute_cols(a: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return torch.ops._C.permute_cols(a, perm)


if hasattr(torch.ops._C, "permute_cols"):
    @register_fake("_C::permute_cols")
    def _permute_cols_fake(a: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(a)


# fp4
def scaled_fp4_quant(
        input: torch.Tensor,
        input_global_scale: torch.Tensor,
        is_sf_swizzled_layout: bool = True,
        backend: str = "none",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale.

    This function quantizes the last dimension of the given tensor `input`. For
    every 16 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is quantized using the `input_global_scale`
    and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).

    Args:
        input: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.
        use_8x4_sf_layout: Whether to use the 8x4 or 128x4 layout for the scaling

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
            two values are packed into a uint8 and float8_e4m3 scaling factors
            in the sizzled layout.
    """
    assert not current_platform.is_rocm()
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16
    device = input.device

    assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    assert input.dtype in (torch.float16, torch.bfloat16), (
        f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."
    )

    use_8x4_sf_layout = True if "trtllm" in backend and m <= 32 else False  # noqa: SIM210

    if use_8x4_sf_layout:
        output, output_scale = flashinfer_quant_nvfp4_8x4_sf_layout(
            input, input_global_scale
        )
    else:
        # Two fp4 values will be packed into an uint8.
        output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)
        if is_sf_swizzled_layout:
            # We use the rounded values to store the swizzled values. Due to the
            # requirement of the Tensor Core, the minimum tile is 128x4 for the scales.
            # So, we first pad the scales to multiples of 128 and 4. Then, the scales
            # (in float8_e4m3fn) are packed into an int32 for every 4 values. More:
            # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x
            round_up = lambda x, y: (x + y - 1) // y * y
            rounded_m = round_up(m, 128)
            scale_n = n // block_size
            rounded_n = round_up(scale_n, 4)
            output_scale = torch.empty(
                (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
            )
        else:
            output_scale = torch.empty((m, n // 16), device=device, dtype=torch.uint8)

        torch.ops._C.scaled_fp4_quant(
            output, input, output_scale, input_global_scale, is_sf_swizzled_layout
        )

    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale


def scaled_fp4_experts_quant(
        input_tensor: torch.Tensor,
        input_global_scale: torch.Tensor,
        expert_offsets: torch.Tensor,
        blockscale_offsets: torch.Tensor,
        topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to NVFP4 and return quantized tensor and scale, for
    packed MoE Inputs.
    Args:
        input_tensor: The input tensor to be quantized to NVFP4
        input_global_scale: A scalar scaling factor for the entire tensor.
        expert_offsets: The expert offsets tensor
        blockscale_offsets: The blockscale offsets tensor
    Outputs:
        output: The quantized tensor in NVFP4
        output_scales: The blockscale tensor in FP8-E4M3
    """
    assert not current_platform.is_rocm()
    assert input_tensor.ndim == 2, (
        f"input.ndim needs to be == 2, but got {input_tensor.ndim}."
    )

    # Control the maximum number of tokens per expert supported by the
    # NVFP4 MoE Expert Quantization. This is used to prevent the kernel
    # from running out of memory. This value can also be increased to support
    # larger models.
    MAX_TOKENS_PER_EXPERT = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE
    m_numtopk, k = input_tensor.shape

    assert m_numtopk <= MAX_TOKENS_PER_EXPERT * topk, (
        f"m_numtopk must be less than MAX_TOKENS_PER_EXPERT("
        f"{MAX_TOKENS_PER_EXPERT})"
        f" for cutlass_moe_fp4, observed m_numtopk = {m_numtopk}. Use"
        f" VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE to set this value."
    )
    scales_k = k // 16
    padded_k = (scales_k + (4 - 1)) // 4

    # output is uint8 and packed fp4 values
    output = torch.empty(
        m_numtopk, k // 2, device=input_tensor.device, dtype=torch.uint8
    )
    output_scales = torch.empty(
        MAX_TOKENS_PER_EXPERT * topk,
        padded_k,
        dtype=torch.int32,
        device=input_tensor.device,
    )
    torch.ops._C.scaled_fp4_experts_quant(
        output,
        output_scales,
        input_tensor,
        input_global_scale,
        expert_offsets,
        blockscale_offsets,
    )
    output_scales = output_scales.view(torch.float8_e4m3fn)
    return output, output_scales


def silu_and_mul_scaled_fp4_experts_quant(
        input_tensor: torch.Tensor,
        input_global_scale: torch.Tensor,
        expert_offsets: torch.Tensor,
        blockscale_offsets: torch.Tensor,
        topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused SiLU+Mul+NVFP4 quantization for MoE intermediate activations.

    Args:
        input_tensor: The input tensor with gate || up layout [m_topk, k*2]
        input_global_scale: A per-expert scaling factor [n_experts]
        expert_offsets: The expert offsets tensor [n_experts+1]
        blockscale_offsets: The blockscale offsets tensor [n_experts+1]
        topk: Number of top-k experts selected
    Outputs:
        output: The quantized tensor in NVFP4 [m_topk, k/2]
        output_scales: The blockscale tensor in FP8-E4M3
    """
    assert not current_platform.is_rocm()
    assert input_tensor.ndim == 2, (
        f"input.ndim needs to be == 2, but got {input_tensor.ndim}."
    )

    # Control the maximum number of tokens per expert supported by the
    # NVFP4 MoE Expert Quantization. This is used to prevent the kernel
    # from running out of memory. This value can also be increased to support
    # larger models.
    MAX_TOKENS_PER_EXPERT = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE
    m_numtopk, k_times_2 = input_tensor.shape
    assert k_times_2 % 2 == 0, "input width must be even (gate || up layout)"
    k = k_times_2 // 2

    assert m_numtopk <= MAX_TOKENS_PER_EXPERT * topk, (
        f"m_numtopk must be less than MAX_TOKENS_PER_EXPERT("
        f"{MAX_TOKENS_PER_EXPERT})"
        f" for cutlass_moe_fp4, observed m_numtopk = {m_numtopk}. Use"
        f" VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE to set this value."
    )
    scales_k = k // 16
    padded_k = (scales_k + (4 - 1)) // 4

    # output is uint8 and packed fp4 values
    output = torch.empty(
        m_numtopk, k // 2, device=input_tensor.device, dtype=torch.uint8
    )
    output_scales = torch.empty(
        MAX_TOKENS_PER_EXPERT * topk,
        padded_k,
        dtype=torch.int32,
        device=input_tensor.device,
    )
    torch.ops._C.silu_and_mul_scaled_fp4_experts_quant(
        output,
        output_scales,
        input_tensor,
        input_global_scale,
        expert_offsets,
        blockscale_offsets,
    )
    output_scales = output_scales.view(torch.float8_e4m3fn)
    return output, output_scales


# fp8
def scaled_fp8_quant(
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
        num_token_padding: int | None = None,
        scale_ub: torch.Tensor | None = None,
        use_per_token_if_dynamic: bool = False,
        output: torch.Tensor | None = None,
        group_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8 (must be 2D: [M, N])
        scale: Optional scaling factor for the FP8 quantization. Supports:
            - 0D or [1]: per-tensor scaling
            - 1D: requires explicit group_shape to disambiguate per-channel
              vs per-token (use (-1, 1) for per-channel, (1, -1) for per-token)
            - 2D [M/group_m, N/group_n]: group scaling (e.g. [M, N/128] for
              DeepSeek-style (1,128) groups, or [M/128, N/128] for (128,128))
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.
        group_shape: Optional tuple (group_m, group_n) specifying the group
            shape for static quantization. Use -1 for "full extent" (e.g.,
            (-1, -1) for per-tensor, (-1, 1) for per-channel, etc.)
            Required for 1D scales; optional for 2D scales.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    shape: tuple[int, int] | torch.Size = input.shape
    # For ROCm on MI300, the output fp8 dtype is torch.float_e3m3fnuz
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    if output is None:
        output = torch.empty(shape, device=input.device, dtype=out_dtype)
    else:
        assert num_token_padding is None, "padding not supported if output passed in"
        assert output.dtype == out_dtype

    if scale is None:
        if use_per_token_if_dynamic:
            scale = torch.empty((shape[0], 1), device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                output, input, scale, scale_ub
            )
        else:
            scale = torch.empty(1, device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        torch.ops._C.static_scaled_fp8_quant(output, input, scale, group_shape)

    return output, scale


# gptq allspark
def allspark_repack_weight(
        qweight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None = None,
        has_zp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rearrange qweight, scale, and zero_point(if asymmetric) to n32k16 format
    for Ampere W8A16 Fused Gemm kernel

    Args:
        qweight: uint8 weight tensor, original k x n format.
        scale: fp16/bf16 weight scale tensor, 1 x n format.
        zero_point: fp16/bf16 weight zero_point tensor, 1 x n format.
            Must be provided for asymmetric quantization.
        has_zp: if use symmetric quantization, has_zp = False.
            if use asymmetric quantization, has_zp = True.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] :
            rearranged weight, scale, and optionally zero_point.
    """
    K = qweight.shape[0]
    N = qweight.shape[1]
    N_32align = (N + 32 - 1) // 32 * 32

    qweight_reorder = torch.empty(
        (N_32align, K), device=qweight.device, dtype=qweight.dtype
    )
    scale_reorder = torch.empty((1, N_32align), device=scale.device, dtype=scale.dtype)
    zero_point_reorder = None
    if has_zp:
        assert zero_point is not None, (
            "zero_point must be provided for asymmetric quantization."
        )
        zero_point_reorder = torch.empty(
            (1, N_32align), device=zero_point.device, dtype=zero_point.dtype
        )

    torch.ops._C.rearrange_kn_weight_as_n32k16_order(
        qweight,
        scale,
        zero_point,
        has_zp,
        qweight_reorder,
        scale_reorder,
        zero_point_reorder,
        K,
        N,
        N_32align,
    )

    return qweight_reorder, scale_reorder, zero_point_reorder


def allspark_w8a16_gemm(
        a: torch.Tensor,
        b_qweight: torch.Tensor,
        b_scales: torch.Tensor,
        b_qzeros: torch.Tensor | None,
        n: int,
        group_size: int,
        sm_count: int,
        sm_version: int,
        CUBLAS_M_THRESHOLD: int,
        has_zp: bool,
        n32k16_reorder: bool,
) -> torch.Tensor:
    return torch.ops._C.allspark_w8a16_gemm(
        a,
        b_qweight,
        b_scales,
        b_qzeros,
        n,
        group_size,
        sm_count,
        sm_version,
        CUBLAS_M_THRESHOLD,
        has_zp,
        n32k16_reorder,
    )


# int8
def scaled_int8_quant(
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
        azp: torch.Tensor | None = None,
        symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] : Output int8 tensor, scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (azp is None), (
            "azp must only be provided for asymmetric quantization."
        )
        torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
        return output, scale, azp

    # dynamic-per-token quantization.
    input_scales = torch.empty(
        (input.numel() // input.shape[-1], 1), device=input.device, dtype=torch.float32
    )
    input_azp = None if symmetric else torch.empty_like(input_scales, dtype=torch.int32)
    torch.ops._C.dynamic_scaled_int8_quant(
        output, input.contiguous(), input_scales, input_azp
    )
    return output, input_scales, input_azp


# gguf
def ggml_dequantize(
        W: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None
) -> torch.Tensor:
    return torch.ops._C.ggml_dequantize(W, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_mul_mat_vec_a8(W, X, quant_type, row)


def ggml_mul_mat_a8(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_mul_mat_a8(W, X, quant_type, row)


def ggml_moe_a8(
        X: torch.Tensor,
        W: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        quant_type: int,
        row: int,
        top_k: int,
        tokens: int,
) -> torch.Tensor:
    return torch.ops._C.ggml_moe_a8(
        X,
        W,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        quant_type,
        row,
        top_k,
        tokens,
    )


def ggml_moe_a8_vec(
        X: torch.Tensor,
        W: torch.Tensor,
        topk_ids: torch.Tensor,
        top_k: int,
        quant_type: int,
        row: torch.SymInt,
        tokens: torch.SymInt,
) -> torch.Tensor:
    return torch.ops._C.ggml_moe_a8_vec(X, W, topk_ids, top_k, quant_type, row, tokens)


def ggml_moe_get_block_size(quant_type: int) -> int:
    return torch.ops._C.ggml_moe_get_block_size(quant_type)


# mamba
def selective_scan_fwd(
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D_: torch.Tensor | None,
        z_: torch.Tensor | None,
        delta_bias_: torch.Tensor | None,
        delta_softplus: bool,
        query_start_loc: torch.Tensor | None,
        cache_indices: torch.Tensor | None,
        has_initial_state: torch.Tensor | None,
        ssm_states: torch.Tensor,
        pad_slot_id: int,
        block_size: int = 1024,
        block_idx_first_scheduled_token: torch.Tensor | None = None,
        block_idx_last_scheduled_token: torch.Tensor | None = None,
        initial_state_idx: torch.Tensor | None = None,
        cu_chunk_seqlen: torch.Tensor | None = None,
        last_chunk_indices: torch.Tensor | None = None,
):
    torch.ops._C.selective_scan_fwd(
        u,
        delta,
        A,
        B,
        C,
        D_,
        z_,
        delta_bias_,
        delta_softplus,
        query_start_loc,
        cache_indices,
        has_initial_state,
        ssm_states,
        pad_slot_id,
        block_size,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
        cu_chunk_seqlen,
        last_chunk_indices,
    )


# ROCm skinny gemms
def LLMM1(a: torch.Tensor, b: torch.Tensor, rows_per_block: int) -> torch.Tensor:
    return torch.ops._rocm_C.LLMM1(a, b, rows_per_block)


def wvSplitK(
        a: torch.Tensor, b: torch.Tensor, cu_count: int, bias: torch.Tensor = None
) -> torch.Tensor:
    return torch.ops._rocm_C.wvSplitK(a, b, bias, cu_count)


def wvSplitKrc(
        a: torch.Tensor, b: torch.Tensor, cu_count: int, bias: torch.Tensor = None
) -> torch.Tensor:
    return torch.ops._rocm_C.wvSplitKrc(a, b, bias, cu_count)


def wvSplitKQ(
        a: torch.Tensor,
        b: torch.Tensor,
        out_dtype: torch.dtype,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        cu_count: int,
        bias: torch.Tensor = None,
) -> torch.Tensor:
    out = torch.empty((b.shape[0], a.shape[0]), dtype=out_dtype, device=b.device)
    torch.ops._rocm_C.wvSplitKQ(a, b, bias, out, scale_a, scale_b, cu_count)
    return out


# moe
def moe_sum(input: torch.Tensor, output: torch.Tensor):
    if not input.is_cuda or not output.is_cuda:
        output.copy_(input.sum(dim=1).to(output.dtype))
        return
    torch.ops._moe_C.moe_sum(input, output)


def moe_align_block_size(
        topk_ids: torch.Tensor,
        num_experts: int,
        block_size: int,
        sorted_token_ids: torch.Tensor,
        experts_ids: torch.Tensor,
        num_tokens_post_pad: torch.Tensor,
        expert_map: torch.Tensor | None = None,
) -> None:
    torch.ops._moe_C.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        expert_map,
    )


def batched_moe_align_block_size(
        max_tokens_per_batch: int,
        block_size: int,
        expert_num_tokens: torch.Tensor,
        sorted_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_pad: torch.Tensor,
) -> None:
    torch.ops._moe_C.batched_moe_align_block_size(
        max_tokens_per_batch,
        block_size,
        expert_num_tokens,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )


def moe_lora_align_block_size(
        topk_ids: torch.Tensor,
        token_lora_mapping: torch.Tensor,
        num_experts: int,
        block_size: int,
        max_loras: int,
        max_num_tokens_padded: int,
        max_num_m_blocks: int,
        sorted_token_ids: torch.Tensor,
        experts_ids: torch.Tensor,
        num_tokens_post_pad: torch.Tensor,
        adapter_enabled: torch.Tensor,
        lora_ids: torch.Tensor,
        expert_map: torch.Tensor | None = None,
) -> None:
    torch.ops._moe_C.moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        adapter_enabled,
        lora_ids,
        expert_map,
    )


def moe_wna16_gemm(
        input: torch.Tensor,
        output: torch.Tensor,
        b_qweight: torch.Tensor,
        b_scales: torch.Tensor,
        b_qzeros: torch.Tensor | None,
        topk_weights: torch.Tensor | None,
        sorted_token_ids: torch.Tensor,
        experts_ids: torch.Tensor,
        num_tokens_post_pad: torch.Tensor,
        top_k: int,
        BLOCK_SIZE_M: int,
        BLOCK_SIZE_N: int,
        BLOCK_SIZE_K: int,
        bit: int,
) -> torch.Tensor:
    if not current_platform.is_cuda():
        raise NotImplementedError(
            "The optimized moe_wna16_gemm kernel is only available on CUDA platforms"
        )
    torch.ops._moe_C.moe_wna16_gemm(
        input,
        output,
        b_qweight,
        b_scales,
        b_qzeros,
        topk_weights,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        top_k,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        bit,
    )


def router_gemm_bf16_fp32(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """bf16 x bf16 -> fp32 GEMM via cuBLAS. weight shape: (N, K)."""
    return torch.ops._moe_C.router_gemm_bf16_fp32(input, weight)


if hasattr(torch.ops, "_moe_C") and hasattr(torch.ops._moe_C, "router_gemm_bf16_fp32"):
    @register_fake("_moe_C::router_gemm_bf16_fp32")
    def router_gemm_bf16_fp32_fake(
            input: torch.Tensor,
            weight: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty(
            input.shape[0], weight.shape[0], dtype=torch.float32, device=input.device
        )


def dsv3_router_gemm(
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        output_dtype: torch.dtype,
) -> torch.Tensor:
    output = torch.empty(
        hidden_states.shape[0],
        router_weight.shape[0],
        device=hidden_states.device,
        dtype=output_dtype,
    )
    torch.ops._moe_C.dsv3_router_gemm(output, hidden_states, router_weight)
    return output


def topk_softmax(
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool = False,
        e_score_correction_bias: torch.Tensor | None = None,
) -> None:
    torch.ops._moe_C.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
    )


def topk_sigmoid(
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool = False,
        e_score_correction_bias: torch.Tensor | None = None,
) -> None:
    torch.ops._moe_C.topk_sigmoid(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
    )


def grouped_topk(
        scores: torch.Tensor,
        num_expert_group: int,
        topk_group: int,
        topk: int,
        renormalize: bool,
        routed_scaling_factor: float,
        bias: torch.Tensor,
        scoring_func: int = 0,
):
    """
    Perform grouped top-k routing for mixture of experts.

    Args:
        scores: Raw inputs (logits if scoring_func=1, scores if scoring_func=0)
        num_expert_group: Number of expert groups
        topk_group: Number of groups to select
        topk: Number of experts to select per token
        renormalize: Whether to renormalize the output weights
        routed_scaling_factor: Scaling factor for routing weights
        bias: Bias tensor (e_score_correction_bias). Always fused in kernel.
        scoring_func: 0=none (no activation), 1=sigmoid
    """
    if not current_platform.is_cuda():
        raise NotImplementedError(
            "The fused grouped_topk kernel is only available on CUDA platforms"
        )
    return torch.ops._moe_C.grouped_topk(
        scores,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )


def moe_wna16_marlin_gemm(
        input: torch.Tensor,
        output: torch.Tensor | None,
        b_qweight: torch.Tensor,
        b_bias: torch.Tensor | None,
        b_scales: torch.Tensor,
        a_scales: torch.Tensor | None,
        global_scale: torch.Tensor | None,
        b_qzeros: torch.Tensor | None,
        g_idx: torch.Tensor | None,
        perm: torch.Tensor | None,
        workspace: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_past_padded: torch.Tensor,
        topk_weights: torch.Tensor,
        moe_block_size: int,
        top_k: int,
        mul_topk_weights: bool,
        b_q_type: ScalarType,
        size_m: int,
        size_n: int,
        size_k: int,
        is_k_full: bool,
        use_atomic_add: bool,
        use_fp32_reduce: bool,
        is_zp_float: bool,
        thread_k: int = -1,
        thread_n: int = -1,
        blocks_per_sm: int = -1,
) -> torch.Tensor:
    return torch.ops._moe_C.moe_wna16_marlin_gemm(
        input,
        output,
        b_qweight,
        b_bias,
        b_scales,
        a_scales,
        global_scale,
        b_qzeros,
        g_idx,
        perm,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_past_padded,
        topk_weights,
        moe_block_size,
        top_k,
        mul_topk_weights,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
        thread_k,
        thread_n,
        blocks_per_sm,
    )


if hasattr(torch.ops, "_moe_C") and hasattr(torch.ops._moe_C, "marlin_gemm_moe"):
    @register_fake("_moe_C::marlin_gemm_moe")
    def marlin_gemm_moe_fake(
            a: torch.Tensor,
            b_q_weights: torch.Tensor,
            sorted_ids: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            b_scales: torch.Tensor,
            b_zero_points: torch.Tensor,
            g_idx: torch.Tensor,
            perm: torch.Tensor,
            workspace: torch.Tensor,
            b_q_type: ScalarType,
            size_m: torch.SymInt,
            size_n: torch.SymInt,
            size_k: torch.SymInt,
            is_k_full: bool,
            num_experts: int,
            topk: int,
            moe_block_size: int,
            replicate_input: bool,
            apply_weights: bool,
    ) -> torch.Tensor:
        return torch.empty((size_m, topk, size_n), dtype=a.dtype, device=a.device)


    @register_fake("_moe_C::moe_wna16_marlin_gemm")
    def moe_wna16_marlin_gemm_fake(
            input: torch.Tensor,
            output: torch.Tensor | None,
            b_qweight: torch.Tensor,
            b_bias: torch.Tensor | None,
            b_scales: torch.Tensor,
            a_scales: torch.Tensor | None,
            global_scale: torch.Tensor | None,
            b_qzeros: torch.Tensor | None,
            g_idx: torch.Tensor | None,
            perm: torch.Tensor | None,
            workspace: torch.Tensor,
            sorted_token_ids: torch.Tensor,
            expert_ids: torch.Tensor,
            num_tokens_past_padded: torch.Tensor,
            topk_weights: torch.Tensor,
            moe_block_size: int,
            top_k: int,
            mul_topk_weights: bool,
            b_q_type: ScalarType,
            size_m: int,
            size_n: int,
            size_k: int,
            is_k_full: bool,
            use_atomic_add: bool,
            use_fp32_reduce: bool,
            is_zp_float: bool,
    ):
        return torch.empty(
            (size_m * top_k, size_n), dtype=input.dtype, device=input.device
        )


def reshape_and_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
) -> None:
    op = _get_torch_op("_C_cache_ops", "reshape_and_cache")
    if op is not None:
        op(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        return
    _reshape_and_cache_torch_fallback(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
        op_name="reshape_and_cache",
    )


def reshape_and_cache_flash(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
) -> None:
    op = _get_torch_op("_C_cache_ops", "reshape_and_cache_flash")
    if op is not None:
        op(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        return
    _reshape_and_cache_torch_fallback(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
        op_name="reshape_and_cache_flash",
    )


def reshape_and_cache_flash_diffkv(
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
) -> None:
    op = _get_torch_op("_C_cache_ops", "reshape_and_cache_flash_diffkv")
    if op is not None:
        op(
            key,
            value,
            kv_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )
        return
    _reshape_and_cache_flash_diffkv_torch_fallback(
        key,
        value,
        kv_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


def concat_and_cache_mla(
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
    )


def concat_and_cache_mla_rope_fused(
        positions: torch.Tensor,
        q_pe: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        slot_mapping: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_cache_dtype: str,
        kv_cache_scale: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.concat_and_cache_mla_rope_fused(
        positions,
        q_pe,
        k_pe,
        kv_c,
        cos_sin_cache,
        is_neox,
        slot_mapping,
        kv_cache,
        kv_cache_dtype,
        kv_cache_scale,
    )


def swap_blocks(
        src: torch.Tensor,
        dst: torch.Tensor,
        block_size_in_bytes: int,
        block_mapping: torch.Tensor,
) -> None:
    """
    Copy specific blocks from one tensor to another.

    This method assumes each of the two input tensors is composed of
    consecutive contiguous blocks, of size block_size_in_bytes.
    i.e. the memory layout for each tensor is:
    [block0] [block1] ... [block N]

    block_mapping determines the subset of blocks to copy of the source tensor,
    and their matching destination block number on the destination tensor.
    block_mapping is expected to be a tensor of shape (num_blocks_to_copy, 2)
    where each block_mapping[i] represents a single copy operation, copying
    block #block_mapping[i][0] from the source tensor
    to block #block_mapping[i][1] on the destination tensor.
    block_mapping should have dtype int64.

    The source and the destination tensors can be either on cpu or gpu,
    but not both on cpu.
    the block mapping tensor must on cpu.
    """
    torch.ops._C_cache_ops.swap_blocks(src, dst, block_size_in_bytes, block_mapping)


def convert_fp8(
        output: torch.Tensor, input: torch.Tensor, scale: float = 1.0, kv_dtype: str = "fp8"
) -> None:
    torch.ops._C_cache_ops.convert_fp8(output, input, scale, kv_dtype)


def gather_and_maybe_dequant_cache(
        src_cache: torch.Tensor,
        dst: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        token_to_seq: torch.Tensor,
        num_tokens: int,
        kv_cache_dtype: str,
        scale: torch.Tensor,
        seq_starts: torch.Tensor | None = None,
) -> None:
    torch.ops._C_cache_ops.gather_and_maybe_dequant_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        kv_cache_dtype,
        scale,
        seq_starts,
    )


def cp_gather_cache(
        src_cache: torch.Tensor,
        dst: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        batch_size: int,
        seq_starts: torch.Tensor | None = None,
) -> None:
    torch.ops._C_cache_ops.cp_gather_cache(
        src_cache, dst, block_table, cu_seq_lens, batch_size, seq_starts
    )


def gather_paged_kv_cache(
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        gathered_key: torch.Tensor,
        gathered_value: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        batch_size: int,
        seq_starts: torch.Tensor | None = None,
) -> None:
    torch.ops._C_cache_ops.gather_paged_kv_cache(
        key_cache,
        value_cache,
        gathered_key,
        gathered_value,
        block_table,
        cu_seq_lens,
        batch_size,
        seq_starts,
    )


def cp_gather_and_upconvert_fp8_kv_cache(
        src_cache: torch.Tensor,
        dst: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        workspace_starts: torch.Tensor,
        batch_size: int,
) -> None:
    """Gather and upconvert FP8 KV cache to BF16 workspace.

    Args:
        src_cache: FP8 KV cache [num_blocks, block_size, 656]
        dst: BF16 output workspace [total_tokens, 576]
        block_table: Block indices [num_reqs, max_blocks]
        seq_lens: Sequence lengths [num_reqs]
        workspace_starts: Workspace start offsets [num_reqs]
        batch_size: Number of requests
    """
    torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache(
        src_cache, dst, block_table, seq_lens, workspace_starts, batch_size
    )


def concat_mla_q(
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        q_out: torch.Tensor,
) -> None:
    """Concatenate query nope and rope for MLA/DSA attention.

    Args:
        ql_nope: Query nope component [num_tokens, num_heads, nope_dim]
        q_pe: Query rope component [num_tokens, num_heads, rope_dim]
        q_out: Output tensor [num_tokens, num_heads, nope_dim + rope_dim]
    """
    torch.ops._C_cache_ops.concat_mla_q(ql_nope, q_pe, q_out)


def indexer_k_quant_and_cache(
        k: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        quant_block_size: int,
        kv_cache_dtype: str,
) -> None:
    torch.ops._C_cache_ops.indexer_k_quant_and_cache(
        k, kv_cache, slot_mapping, quant_block_size, kv_cache_dtype
    )


def cp_gather_indexer_k_quant_cache(
        kv_cache: torch.Tensor,
        dst_k: torch.Tensor,
        dst_scale: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
    )


def get_device_attribute(attribute: int, device: int) -> int:
    return torch.ops._C_cuda_utils.get_device_attribute(attribute, device)


def get_max_shared_memory_per_block_device_attribute(device: int) -> int:
    # ruff: noqa: E501
    return torch.ops._C_cuda_utils.get_max_shared_memory_per_block_device_attribute(
        device
    )


# custom ar
def init_custom_ar(
        ipc_tensors: list[torch.Tensor],
        rank_data: torch.Tensor,
        rank: int,
        fully_connected: bool,
) -> int:
    return torch.ops._C_custom_ar.init_custom_ar(
        ipc_tensors, rank_data, rank, fully_connected
    )


def all_reduce(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        reg_buffer: int,
        reg_buffer_sz_bytes: int,
) -> None:
    torch.ops._C_custom_ar.all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes)


def dispose(fa: int) -> None:
    torch.ops._C_custom_ar.dispose(fa)


def meta_size() -> int:
    return torch.ops._C_custom_ar.meta_size()


def register_buffer(fa: int, ipc_tensors: list[int]) -> None:
    return torch.ops._C_custom_ar.register_buffer(fa, ipc_tensors)


def get_graph_buffer_ipc_meta(fa: int) -> tuple[list[int], list[int]]:
    return torch.ops._C_custom_ar.get_graph_buffer_ipc_meta(fa)


def register_graph_buffers(
        fa: int, handles: list[list[int]], offsets: list[list[int]]
) -> None:
    torch.ops._C_custom_ar.register_graph_buffers(fa, handles, offsets)


def allocate_shared_buffer_and_handle(size: int) -> tuple[int, torch.Tensor]:
    return torch.ops._C_custom_ar.allocate_shared_buffer_and_handle(size)


def open_mem_handle(mem_handle: torch.Tensor):
    return torch.ops._C_custom_ar.open_mem_handle(mem_handle)


def free_shared_buffer(ptr: int) -> None:
    torch.ops._C_custom_ar.free_shared_buffer(ptr)


# quick all reduce
def init_custom_qr(rank: int, world_size: int, qr_max_size: int | None = None) -> int:
    return torch.ops._C_custom_ar.init_custom_qr(rank, world_size, qr_max_size)


def qr_destroy(fa: int) -> None:
    torch.ops._C_custom_ar.qr_destroy(fa)


def qr_all_reduce(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        quant_level: int,
        cast_bf2half: bool = False,
) -> None:
    torch.ops._C_custom_ar.qr_all_reduce(fa, inp, out, quant_level, cast_bf2half)


def qr_get_handle(fa: int) -> torch.Tensor:
    return torch.ops._C_custom_ar.qr_get_handle(fa)


def qr_open_handles(fa: int, handles: list[torch.Tensor]) -> None:
    return torch.ops._C_custom_ar.qr_open_handles(fa, handles)


def qr_max_size() -> int:
    return torch.ops._C_custom_ar.qr_max_size()


def get_flash_mla_metadata(
        cache_seqlens: torch.Tensor,
        num_heads_per_head_k: int,
        num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        cache_seqlens: (batch_size), dtype torch.int32.
        num_heads_per_head_k: Equals to seq_len_q * num_heads_q // num_heads_k.
        num_heads_k: num_heads_k.

    Return:
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), dtype torch.int32.
        num_splits: (batch_size + 1), dtype torch.int32.
    """
    return torch.ops._C.get_flash_mla_metadata(
        cache_seqlens, num_heads_per_head_k, num_heads_k
    )


def flash_mla_with_kvcache(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        head_dim_v: int,
        tile_scheduler_metadata: torch.Tensor,
        num_splits: torch.Tensor,
        softmax_scale: float | None = None,
        causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Arguments:
        q: (batch_size, seq_len_q, num_heads_q, head_dim).
        k_cache: (num_blocks, page_block_size, num_heads_k, head_dim).
        block_table: (batch_size, max_num_blocks_per_seq), torch.int32.
        cache_seqlens: (batch_size), torch.int32.
        head_dim_v: Head_dim of v.
        tile_scheduler_metadata: (num_sm_parts, TileSchedulerMetaDataSize), torch.int32, return by get_mla_metadata.
        num_splits: (batch_size + 1), torch.int32, return by get_mla_metadata.
        softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim).
        causal: bool. Whether to apply causal attention mask.

    Return:
        out: (batch_size, seq_len_q, num_heads_q, head_dim_v).
        softmax_lse: (batch_size, num_heads_q, seq_len_q), torch.float32.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = torch.ops._C.flash_mla_fwd_kvcache(
        q,
        k_cache,
        None,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
    )
    return out, softmax_lse


def sm100_cutlass_mla_decode(
        out: torch.Tensor,
        lse: torch.Tensor,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        workspace: torch.Tensor,
        scale: float,
        num_kv_splits: int,
) -> torch.Tensor:
    torch.ops._C.sm100_cutlass_mla_decode(
        out,
        lse,
        q_nope,
        q_pe,
        kv_c_and_k_pe_cache,
        seq_lens,
        page_table,
        workspace,
        scale,
        num_kv_splits,
    )
    return out


def sm100_cutlass_mla_get_workspace_size(
        max_seq_len: int, num_batches: int, sm_count: int, num_kv_splits: int
) -> int:
    return torch.ops._C.sm100_cutlass_mla_get_workspace_size(
        max_seq_len, num_batches, sm_count, num_kv_splits
    )


def dsv3_fused_a_gemm(
        output: torch.Tensor,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
) -> None:
    """DeepSeek V3 fused A GEMM (SM 9.0+, bf16 only, 1-16 tokens).

    Computes output = mat_a @ mat_b.T where:
      mat_a: [num_tokens, 7168] row-major bf16 (hidden states)
      mat_b: [7168, 2112] column-major bf16 (weight transposed)
      output: [num_tokens, 2112] row-major bf16

    Optimized for the DeepSeek V2/V3 QKV A-projection at small batch sizes.
    Requires SM 9.0+ (Hopper).
    """
    torch.ops._C.dsv3_fused_a_gemm(output, mat_a, mat_b)


if hasattr(torch.ops._C, "weight_packed_linear"):
    @register_fake("_C::weight_packed_linear")
    def weight_packed_linear_fake(
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            bias: torch.Tensor | None,
            is_vnni: bool,
    ) -> torch.Tensor:
        return torch.empty(
            (mat1.size(0), mat2.size(0)), dtype=mat1.dtype, device=mat2.device
        )

if hasattr(torch.ops._C, "fused_experts_cpu"):
    @register_fake("_C::fused_experts_cpu")
    def fused_experts_cpu_fake(
            hidden_states: torch.Tensor,
            w1: torch.Tensor,
            w2: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            inplace: bool,
            use_int8_w8a8: bool,
            use_fp8_w8a16: bool,
            w1_scale: torch.Tensor | None,
            w2_scale: torch.Tensor | None,
            block_size: list[int] | None,
            a1_scale: torch.Tensor | None,
            a2_scale: torch.Tensor | None,
            is_vnni: bool,
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states)

if hasattr(torch.ops._C, "int8_scaled_mm_with_quant"):
    @register_fake("_C::int8_scaled_mm_with_quant")
    def int8_scaled_mm_with_quant_fake(
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            scales2: torch.Tensor,
            bias: torch.Tensor | None,
            out_dtype: torch.dtype,
            is_vnni: bool,
    ) -> torch.Tensor:
        M = mat1.size(0)
        N = mat2.size(0)
        return torch.empty((M, N), dtype=out_dtype)


class CPUDNNLGEMMHandler:
    def __init__(self) -> None:
        self.handler_tensor: torch.Tensor | None = None
        self.n = -1
        self.k = -1

    def __del__(self):
        if self.handler_tensor is not None:
            torch.ops._C.release_dnnl_matmul_handler(self.handler_tensor.item())


_supports_onednn = bool(hasattr(torch.ops._C, "create_onednn_mm_handler"))


def is_onednn_acl_supported():
    return torch.ops._C.is_onednn_acl_supported()


def create_onednn_mm(
        weight: torch.Tensor,  # [K, N]
        primitive_cache_size: int = 128,
) -> CPUDNNLGEMMHandler:
    handler = CPUDNNLGEMMHandler()
    handler.k, handler.n = weight.size()
    # store the handler pointer in a tensor it doesn't get inlined
    handler.handler_tensor = torch.tensor(
        torch.ops._C.create_onednn_mm_handler(weight, primitive_cache_size),
        dtype=torch.int64,
    )
    return handler


def onednn_mm(
        dnnl_handler: CPUDNNLGEMMHandler,
        x: torch.Tensor,
        bias: torch.Tensor | None,
) -> torch.Tensor:
    output = torch.empty((*x.shape[0:-1], dnnl_handler.n), dtype=x.dtype)
    torch.ops._C.onednn_mm(
        output, x.reshape(-1, dnnl_handler.k), bias, dnnl_handler.handler_tensor
    )

    return output


def create_onednn_scaled_mm(
        weight: torch.Tensor,  # [K, N]
        weight_scales: torch.Tensor,
        output_type: torch.dtype,
        dynamic_quant: bool,
        use_azp: bool,
        primitive_cache_size: int = 128,
) -> CPUDNNLGEMMHandler:
    handler = CPUDNNLGEMMHandler()
    handler.k, handler.n = weight.size()
    # store the handler pointer in a tensor so it doesn't get inlined
    handler.handler_tensor = torch.tensor(
        torch.ops._C.create_onednn_scaled_mm_handler(
            weight,
            weight_scales,
            output_type,
            dynamic_quant,
            use_azp,
            primitive_cache_size,
        ),
        dtype=torch.int64,
    )
    return handler


def onednn_scaled_int8_quant(
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
        azp: torch.Tensor | None = None,
        symmetric: bool = True,
):
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] : Output int8 tensor, scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    token_num = input.numel() // input.shape[-1]
    input = input.view((token_num, input.shape[-1]))
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (azp is None), (
            "azp must only be provided for asymmetric quantization."
        )
        torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
        return output, scale, azp

    # dynamic-per-token quantization.
    input_scales = torch.empty((token_num, 1), device=input.device, dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(input_scales, dtype=torch.int32)
    torch.ops._C.dynamic_scaled_int8_quant(output, input, input_scales, input_azp)
    return output, input_scales, input_azp


def onednn_scaled_mm(
        dnnl_handler: CPUDNNLGEMMHandler,
        x: torch.Tensor,
        output: torch.Tensor,
        input_scale: torch.Tensor | None,
        input_zp: torch.Tensor | None,
        input_zp_adj: torch.Tensor | None,
        bias: torch.Tensor | None,
) -> torch.Tensor:
    torch.ops._C.onednn_scaled_mm(
        output,
        x,
        input_scale,
        input_zp,
        input_zp_adj,
        bias,
        dnnl_handler.handler_tensor,
    )

    return output


def cpu_attn_get_scheduler_metadata(
        num_reqs: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_lens: torch.Tensor,
        dtype: torch.dtype,
        query_start_loc: torch.Tensor,
        causal: bool,
        sliding_window_size: int,
        isa: str,
        enable_kv_split: bool,
) -> torch.Tensor:
    scheduler_metadata = torch.ops._C.get_scheduler_metadata(
        num_reqs,
        num_heads,
        num_kv_heads,
        head_dim,
        seq_lens,
        dtype,
        query_start_loc,
        causal,
        sliding_window_size,
        isa,
        enable_kv_split,
    )
    return scheduler_metadata


def cpu_attn_reshape_and_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        isa: str,
) -> None:
    torch.ops._C.cpu_attn_reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        isa,
    )


def cpu_attention_with_kv_cache(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        scale: float,
        causal: bool,
        alibi_slopes: torch.Tensor | None,
        sliding_window: tuple[int, int],
        block_table: torch.Tensor,
        softcap: float,
        scheduler_metadata: torch.Tensor,
        s_aux: torch.Tensor | None,
) -> None:
    torch.ops._C.cpu_attention_with_kv_cache(
        query,
        key_cache,
        value_cache,
        output,
        query_start_loc,
        seq_lens,
        scale,
        causal,
        alibi_slopes,
        sliding_window[0],
        sliding_window[1],
        block_table,
        softcap,
        scheduler_metadata,
        s_aux,
    )


def cpu_gemm_wna16(
        input: torch.Tensor,
        q_weight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor | None,
        g_idx: torch.Tensor | None,
        bias: torch.Tensor | None,
        pack_factor: int,
        isa_hint: str,
) -> torch.Tensor:
    output = torch.empty((input.size(0), scales.size(1)), dtype=input.dtype)
    torch.ops._C.cpu_gemm_wna16(
        input,
        q_weight,
        output,
        scales,
        zeros,
        g_idx,
        bias,
        pack_factor,
        isa_hint,
    )
    return output


def cpu_prepack_moe_weight(
        weight: torch.Tensor,
        isa: str,
) -> torch.Tensor:
    output = torch.empty_like(weight)
    torch.ops._C.prepack_moe_weight(weight, output, isa)
    return output


def cpu_fused_moe(
        input: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_bias: torch.Tensor | None,
        w2_bias: torch.Tensor | None,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        act: str,
        isa: str,
        skip_weighted: bool = False,
) -> torch.Tensor:
    output = torch.empty_like(input)
    torch.ops._C.cpu_fused_moe(
        output,
        input,
        w13,
        w2,
        w13_bias,
        w2_bias,
        topk_weights,
        topk_ids,
        skip_weighted,
        act,
        isa,
    )
    return output


if hasattr(torch.ops._qutlass_C, "matmul_mxf4_bf16_tn"):
    @register_fake("_qutlass_C::matmul_mxf4_bf16_tn")
    def _fake_matmul_mxf4_bf16_tn(
            a: torch.Tensor,
            b: torch.Tensor,
            a_sf: torch.Tensor,
            b_sf: torch.Tensor,
            alpha: torch.Tensor,
    ):
        return a.new_empty(*a.shape[:-1], b.shape[0], dtype=torch.bfloat16)


def matmul_mxf4_bf16_tn(
        a: torch.Tensor,
        b: torch.Tensor,
        a_sf: torch.Tensor,
        b_sf: torch.Tensor,
        alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._qutlass_C.matmul_mxf4_bf16_tn(a, b, a_sf, b_sf, alpha)


if hasattr(torch.ops._qutlass_C, "matmul_ada_mxf4_bf16_tn"):
    @register_fake("_qutlass_C::matmul_ada_mxf4_bf16_tn")
    def _fake_matmul_ada_mxf4_bf16_tn(
            a: torch.Tensor,
            b: torch.Tensor,
            a_sf: torch.Tensor,
            b_sf: torch.Tensor,
            alpha: torch.Tensor,
    ):
        return a.new_empty(*a.shape[:-1], b.shape[0], dtype=torch.bfloat16)


def matmul_ada_mxf4_bf16_tn(
        a: torch.Tensor,
        b: torch.Tensor,
        a_sf: torch.Tensor,
        b_sf: torch.Tensor,
        alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops._qutlass_C.matmul_ada_mxf4_bf16_tn(a, b, a_sf, b_sf, alpha)


if hasattr(torch.ops._qutlass_C, "fusedQuantizeMxQuest"):
    @register_fake("_qutlass_C::fusedQuantizeMxQuest")
    def _fake_fused_quantize_mx_quest(
            a: torch.Tensor, b: torch.Tensor, xh_e2m1: torch.Tensor, xh_e8m0: torch.Tensor
    ):
        return xh_e2m1, xh_e8m0

if hasattr(torch.ops._qutlass_C, "fusedQuantizeMxAbsMax"):
    @register_fake("_qutlass_C::fusedQuantizeMxAbsMax")
    def _fake_fused_quantize_mx_absmax(
            a: torch.Tensor, b: torch.Tensor, xh_e2m1: torch.Tensor, xh_e8m0: torch.Tensor
    ):
        return xh_e2m1, xh_e8m0


def fusedQuantizeMx(
        a: torch.Tensor, b: torch.Tensor, *, method: Literal["quest", "abs_max"] = "quest"
) -> tuple[torch.Tensor, torch.Tensor]:
    if a.dim() == 0:
        raise ValueError("`a` must have at least 1 dimension.")
    if a.size(-1) % 32 != 0:
        raise ValueError(f"last dim of `a` must be divisible by 32, got {a.size(-1)}.")
    if b.device != a.device:
        raise ValueError("`a` and `b` must be on the same device.")

    xh_e2m1 = torch.empty(
        *a.shape[:-1], a.size(-1) // 2, dtype=torch.uint8, device=a.device
    )

    rows, cols = a.numel() // a.size(-1), a.size(-1) // 32
    n_row_blocks = cdiv(rows, 128)
    n_col_blocks = cdiv(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    xh_e8m0 = torch.empty(
        padded_rows, padded_cols, dtype=torch.float8_e8m0fnu, device=a.device
    )

    if not hasattr(torch.ops, "_qutlass_C"):
        raise RuntimeError(
            "The `_qutlass_C` extension is not loaded. "
            "Make sure your custom op library is imported before calling fusedQuantizeMx."
        )

    if method == "quest":
        return torch.ops._qutlass_C.fusedQuantizeMxQuest(a, b, xh_e2m1, xh_e8m0)
    elif method == "abs_max":
        return torch.ops._qutlass_C.fusedQuantizeMxAbsMax(a, b, xh_e2m1, xh_e8m0)
    else:
        raise ValueError(f"invalid method {method!r}, must be 'quest' or 'abs_max'")


if hasattr(torch.ops._qutlass_C, "fusedQuantizeNv"):
    @register_fake("_qutlass_C::fusedQuantizeNv")
    def _fake_fused_quantize_nv(
            a: torch.Tensor,
            b: torch.Tensor,
            xh_e2m1: torch.Tensor,
            xh_e4m3: torch.Tensor,
            global_scale: torch.Tensor,
    ):
        return xh_e2m1, xh_e4m3


def fusedQuantizeNv(
        a: torch.Tensor, b: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xh_e2m1 = torch.empty(
        *a.shape[:-1], a.size(-1) // 2, dtype=torch.uint8, device=a.device
    )

    rows, cols = a.numel() // a.size(-1), a.size(-1) // 16
    n_row_blocks = cdiv(rows, 128)
    n_col_blocks = cdiv(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    xh_e4m3 = torch.empty(
        padded_rows, padded_cols, dtype=torch.float8_e4m3fn, device=a.device
    )

    return torch.ops._qutlass_C.fusedQuantizeNv(a, b, xh_e2m1, xh_e4m3, global_scale)


def hadacore_transform(x: torch.Tensor, inplace: bool = True) -> torch.Tensor:
    """
    Perform Hadamard transforms using [Hadacore](https://arxiv.org/abs/2412.08832)
    kernels. Note that these kernels exploit the recursive properties of
    Sylvester Hadamards, and therefore do not require transform weight data

    Note that sylvester hadamard transforms are also symmetric, which means that
    this function is also applies the (transpose <=> inverse) transform.

    :param x: value to be transformed inplace
    :param inplace: modify value in place
    :return: value after transformation
    """
    return torch.ops._C.hadacore_transform(x, inplace)


if hasattr(torch.ops._C, "hadacore_transform"):
    @register_fake("_C::hadacore_transform")
    def _hadacore_transform_fake(x: torch.Tensor, inplace: bool) -> torch.Tensor:
        return torch.empty_like(x) if not inplace else x
