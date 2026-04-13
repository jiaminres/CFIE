# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from cfie import _custom_ops as ops
from cfie.platforms import current_platform
from cfie.triton_utils import HAS_TRITON, tl, triton


# Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
# can be used to combine partial attention results (in the split-KV case)
def _supports_precompiled_merge_attn_states(*, output: torch.Tensor) -> bool:
    if not current_platform.is_cuda() or not output.is_cuda:
        return False
    if output.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False

    head_size = output.shape[2]
    if output.dtype == torch.float32:
        return head_size % 4 == 0
    return head_size % 8 == 0


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
) -> None:
    if not HAS_TRITON:
        if _supports_precompiled_merge_attn_states(output=output):
            ops.merge_attn_states(
                output,
                prefix_output,
                prefix_lse,
                suffix_output,
                suffix_lse,
                output_lse,
            )
            return

        p_lse = torch.where(
            prefix_lse == float("inf"),
            torch.full_like(prefix_lse, -float("inf")),
            prefix_lse,
        )
        s_lse = torch.where(
            suffix_lse == float("inf"),
            torch.full_like(suffix_lse, -float("inf")),
            suffix_lse,
        )
        max_lse = torch.maximum(p_lse, s_lse)
        valid = torch.isfinite(max_lse)

        p_shift = torch.where(valid, p_lse - max_lse, torch.full_like(p_lse, -float("inf")))
        s_shift = torch.where(valid, s_lse - max_lse, torch.full_like(s_lse, -float("inf")))
        p_se = torch.exp(p_shift)
        s_se = torch.exp(s_shift)
        out_se = p_se + s_se

        if output_lse is not None:
            out_lse = torch.where(
                valid,
                torch.log(out_se) + max_lse,
                torch.full_like(max_lse, -float("inf")),
            )
            output_lse.copy_(out_lse)

        p_scale = torch.where(out_se > 0, p_se / out_se, torch.zeros_like(out_se))
        s_scale = torch.where(out_se > 0, s_se / out_se, torch.zeros_like(out_se))
        output.copy_(
            prefix_output * p_scale.transpose(0, 1).unsqueeze(-1)
            + suffix_output * s_scale.transpose(0, 1).unsqueeze(-1)
        )
        return

    num_tokens = output.shape[0]
    num_query_heads = output.shape[1]
    head_size = output.shape[2]
    padded_head_size = triton.next_power_of_2(head_size)
    # We assume the output stride on num_head is not always as same as the
    # `suffix_output` and `prefix_output`, as them might be padded by the attention
    # backend.
    prefix_head_stride = prefix_output.stride(1)
    output_head_stride = output.stride(1)
    # TODO(woosuk): Use CUDA kernel instead of Triton to minimize CPU overhead.
    merge_attn_states_kernel[(num_tokens, num_query_heads)](
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        prefix_head_stride,
        output_head_stride,
        head_size,
        padded_head_size,
        output_lse is not None,
    )


@triton.jit
def merge_attn_states_kernel(
    output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    output_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse,  # [NUM_HEADS, NUM_TOKENS]
    prefix_head_stride,
    output_head_stride,
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    num_tokens = tl.num_programs(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    p_lse = tl.load(prefix_lse + head_idx * num_tokens + token_idx)
    s_lse = tl.load(suffix_lse + head_idx * num_tokens + token_idx)

    # FA2 and FA3 have different behavior for when the sum-exp is 0, this namely
    # arises with 0 len seqlens. FA3 returns -inf here while FA2 returns inf.
    # If we see an inf assume FA2 and convert inf to -inf for consistency
    # and correctness. Inf generally doesn't make sense in this context outside
    # of undefined-behavior/FA2-case, so I think this a safe assumption.
    p_lse = float("-inf") if p_lse == float("inf") else p_lse
    s_lse = float("-inf") if s_lse == float("inf") else s_lse

    max_lse = tl.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    # Will reuse precomputed Exp values for scale factor computation.
    p_se = tl.exp(p_lse)
    s_se = tl.exp(s_lse)
    out_se = p_se + s_se

    if OUTPUT_LSE:
        out_lse = tl.log(out_se) + max_lse
        tl.store(output_lse + head_idx * num_tokens + token_idx, out_lse)

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE
    p_out = tl.load(
        prefix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )
    s_out = tl.load(
        suffix_output
        + token_idx * num_heads * prefix_head_stride
        + head_idx * prefix_head_stride
        + head_arange,
        mask=head_mask,
    )

    # NOTE(woosuk): Be careful with the numerical stability.
    # We should compute the scale first, and then multiply it with the output.
    # Do not multiply the output with tl.exp(p_lse) or tl.exp(s_lse) directly.
    p_scale = p_se / out_se
    s_scale = s_se / out_se
    out = p_out * p_scale + s_out * s_scale
    tl.store(
        output
        + token_idx * num_heads * output_head_stride
        + head_idx * output_head_stride
        + head_arange,
        out,
        mask=head_mask,
    )
