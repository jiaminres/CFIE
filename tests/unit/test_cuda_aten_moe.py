# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch


def _make_quant_config(
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        quant_dtype=None,
        weight_quant_dtype=None,
        block_shape=None,
        per_act_token_quant=False,
        per_out_ch_quant=False,
        a1_scale=None,
        a2_scale=None,
        a1_gscale=None,
        a2_gscale=None,
        w1_scale=None,
        w2_scale=None,
        w1_zp=None,
        w2_zp=None,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        g1_alphas=None,
        g2_alphas=None,
    )


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def _reference_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states)
    for token_idx in range(hidden_states.size(0)):
        for route_idx in range(topk_ids.size(1)):
            expert_idx = int(topk_ids[token_idx, route_idx].item())
            gate_up = hidden_states[token_idx : token_idx + 1] @ w1[expert_idx].t()
            if w1_bias is not None:
                gate_up = gate_up + w1_bias[expert_idx]
            activated = _silu_and_mul(gate_up)
            expert_output = activated @ w2[expert_idx].t()
            if w2_bias is not None:
                expert_output = expert_output + w2_bias[expert_idx]
            output[token_idx] += (
                topk_weights[token_idx, route_idx].to(hidden_states.dtype)
                * expert_output.squeeze(0)
            )
    return output


def _skip_if_cuda_aten_moe_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    import cfie._custom_ops  # noqa: F401
    from cfie.model_executor.layers.fused_moe.moe_permute_unpermute import (
        moe_permute_unpermute_supported,
    )

    try:
        supported = moe_permute_unpermute_supported()
    except (AttributeError, RuntimeError):
        supported = False
    if not supported:
        pytest.skip("MoE permute/unpermute CUDA ops are not available")


@pytest.mark.parametrize("with_bias", [False, True])
def test_cuda_aten_moe_matches_reference(with_bias: bool) -> None:
    _skip_if_cuda_aten_moe_unavailable()

    from cfie.model_executor.layers.fused_moe.activation import MoEActivation
    from cfie.model_executor.layers.fused_moe.cuda_aten_moe import CudaAtenExperts

    torch.manual_seed(123)
    device = torch.device("cuda")
    num_tokens, hidden_dim, intermediate_dim, num_experts, topk = 7, 16, 24, 4, 2

    hidden_states = torch.randn(
        (num_tokens, hidden_dim), device=device, dtype=torch.float16
    )
    w1 = (
        torch.randn(
            (num_experts, 2 * intermediate_dim, hidden_dim),
            device=device,
            dtype=torch.float16,
        )
        / 4
    )
    w2 = (
        torch.randn(
            (num_experts, hidden_dim, intermediate_dim),
            device=device,
            dtype=torch.float16,
        )
        / 4
    )
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3], [2, 0]],
        device=device,
        dtype=torch.int64,
    )
    topk_weights = torch.softmax(
        torch.randn((num_tokens, topk), device=device), dim=-1
    )
    w1_bias = (
        torch.randn(
            (num_experts, 2 * intermediate_dim), device=device, dtype=torch.float16
        )
        / 8
        if with_bias
        else None
    )
    w2_bias = (
        torch.randn((num_experts, hidden_dim), device=device, dtype=torch.float16) / 8
        if with_bias
        else None
    )

    experts = CudaAtenExperts(SimpleNamespace(), _make_quant_config(w1_bias, w2_bias))
    workspace13_shape, workspace2_shape, output_shape = experts.workspace_shapes(
        num_tokens,
        2 * intermediate_dim,
        hidden_dim,
        topk,
        num_experts,
        num_experts,
        None,
        MoEActivation.SILU,
    )
    workspace13 = torch.empty(
        workspace13_shape, device=device, dtype=hidden_states.dtype
    )
    workspace2 = torch.empty(workspace2_shape, device=device, dtype=hidden_states.dtype)
    output = torch.empty(output_shape, device=device, dtype=hidden_states.dtype)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=num_experts,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    expected = _reference_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_bias,
        w2_bias,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)
