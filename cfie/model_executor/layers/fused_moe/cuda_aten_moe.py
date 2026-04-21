# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import cfie.model_executor.layers.fused_moe.modular_kernel as mk
from cfie.model_executor.layers.fused_moe.activation import MoEActivation
from cfie.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from cfie.model_executor.layers.fused_moe.moe_permute_unpermute import (
    moe_permute,
    moe_unpermute,
)
from cfie.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from cfie.model_executor.layers.quantization.utils.quant_utils import QuantKey
from cfie.platforms import current_platform


class CudaAtenExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (None, None)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU_NO_MUL,
            MoEActivation.RELU2_NO_MUL,
        )

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return (not moe_parallel_config.use_ep) and moe_parallel_config.dp_size == 1

    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        del global_num_experts, local_num_experts, expert_tokens_meta
        activation_dim = self.adjust_N_for_activation(N, activation)
        workspace13 = (M * topk, N)
        workspace2 = (M * topk, K)
        output = (M, K)
        return (workspace13, workspace2, output)

    @staticmethod
    def _linear_into(
        output: torch.Tensor,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        torch.mm(input_tensor, weight.transpose(0, 1), out=output)
        if bias is not None:
            output.add_(bias)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        del a1q_scale, a2_scale, expert_tokens_meta

        topk = topk_ids.size(1)
        local_num_experts = w1.size(0)
        hidden_dim = hidden_states.size(-1)
        intermediate_dim = w1.size(1)
        activation_dim = self.adjust_N_for_activation(intermediate_dim, activation)
        num_experts = global_num_experts if expert_map is None else expert_map.size(0)

        gate_up_workspace = workspace13
        activation_workspace = torch.empty(
            (hidden_states.size(0) * topk, activation_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        permuted_hidden_states = workspace2

        (
            permuted_hidden_states,
            _,
            expert_first_token_offset,
            inv_permuted_idx,
            _,
        ) = moe_permute(
            hidden_states,
            None,
            topk_ids,
            num_experts,
            local_num_experts,
            expert_map,
            permuted_hidden_states=permuted_hidden_states,
        )

        offsets_cpu = expert_first_token_offset.cpu()
        for expert_idx in range(local_num_experts):
            start = int(offsets_cpu[expert_idx].item())
            end = int(offsets_cpu[expert_idx + 1].item())
            if end <= start:
                continue

            expert_input = permuted_hidden_states[start:end, :hidden_dim]
            gate_up = gate_up_workspace[start:end, :intermediate_dim]
            activated = activation_workspace[start:end, :activation_dim]

            self._linear_into(
                gate_up,
                expert_input,
                w1[expert_idx],
                None if self.w1_bias is None else self.w1_bias[expert_idx],
            )
            self.activation(activation, activated, gate_up)
            self._linear_into(
                expert_input,
                activated,
                w2[expert_idx],
                None if self.w2_bias is None else self.w2_bias[expert_idx],
            )

        reduce_weights = (
            topk_weights
            if not apply_router_weight_on_input
            else torch.ones_like(topk_weights)
        )
        moe_unpermute(
            out=output,
            permuted_hidden_states=permuted_hidden_states,
            topk_weights=reduce_weights,
            inv_permuted_idx=inv_permuted_idx,
            expert_first_token_offset=expert_first_token_offset,
        )
