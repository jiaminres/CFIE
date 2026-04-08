# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import cfie.model_executor.layers.fused_moe.modular_kernel as mk
from cfie.distributed import get_ep_group
from cfie.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from cfie.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from cfie.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from cfie.utils.flashinfer import nvfp4_block_scale_interleave


def _quantize_and_setup_dispatch(
    a1: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    defer_input_quant: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    # 若后端支持在 MoE kernel 内部完成输入量化，这里就直接跳过前置量化。
    if defer_input_quant:
        a1q = a1
        a1q_scale = None
    else:
        input_sf = (
            quant_config.a1_gscale
            if quant_config.use_nvfp4_w4a4
            else quant_config.a1_scale
        )

        # swizzling 会把 scale 补齐到 128 的倍数，导致其 shape 与 hidden states
        # 不再一致，从而破坏 A2A kernel 的输入假设，因此延后到 A2A 之后再做。
        a1q, a1q_scale = a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            input_sf,
            quant_dtype=quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape,
            is_fp4_scale_swizzled=False,
        )

    # 若是静态量化或已经延后量化，就不需要把 scale 随 dispatch 一起收集。
    skip_gather_scales = a1q_scale is None or a1q_scale.ndim == 0
    scales = None if skip_gather_scales else [a1q_scale]

    return a1q, scales


def _unwrap_scale_and_prepare_for_moe(
    scales: list[torch.Tensor] | None,
    quant_config: FusedMoEQuantConfig,
) -> torch.Tensor:
    assert scales is not None and len(scales) == 1
    a1q_scale = scales[0]
    # 只有在 A2A 完成后，才按需要把 scale 转成 nvfp4 kernel 期望的 swizzled 形式。
    if quant_config.quant_dtype == "nvfp4" and quant_config.is_nvfp4_scale_swizzled:
        assert a1q_scale is not None
        if a1q_scale.element_size() == 1:
            a1q_scale = a1q_scale.view(torch.uint8)
        a1q_scale = nvfp4_block_scale_interleave(a1q_scale)

    return a1q_scale


class MoEPrepareAndFinalizeNaiveDPEPModular(mk.FusedMoEPrepareAndFinalizeModular):
    # DP/EP 场景下的 naive modular prepare/finalize 实现。
    # dispatch / combine 依赖 Torch 的 AR/RS 或 AR，并直接作用在 topk 权重与索引上。

    def __init__(
        self,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self._num_dispatchers = num_dispatchers

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def output_is_reduced(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        # 先量化输入，再把 topk 权重和 expert 索引分发到各 expert rank。

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            # shared experts overlap 打开时，这里不要做 inplace 修改。
            a1 = a1 * topk_weights.to(a1.dtype)

        a1q, scales = _quantize_and_setup_dispatch(a1, quant_config, defer_input_quant)

        res = get_ep_group().dispatch(
            a1q,
            topk_weights,
            topk_ids,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=scales,
        )

        if scales is None:
            a1q, topk_weights, topk_ids = res
            a1q_scale = None
        else:
            a1q, topk_weights, topk_ids, scales = res
            a1q_scale = _unwrap_scale_and_prepare_for_moe(scales, quant_config)

        return a1q, a1q_scale, None, topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        out = weight_and_reduce_impl.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        output.copy_(
            get_ep_group().combine(out, is_sequence_parallel=self.is_sequence_parallel)
        )


class MoEPrepareAndFinalizeNaiveDPEPMonolithic(mk.FusedMoEPrepareAndFinalizeMonolithic):
    # DP/EP 场景下的 naive monolithic prepare/finalize 实现。
    # dispatch / combine 依赖 Torch 的 AR/RS 或 AR，但这里分发的是 router logits。

    def __init__(
        self,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self._num_dispatchers = num_dispatchers

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def output_is_reduced(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        router_logits: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareMonolithicResultType:
        # 先量化输入，再把 router logits 分发到对应 expert rank。

        a1q, scales = _quantize_and_setup_dispatch(a1, quant_config, defer_input_quant)

        res = get_ep_group().dispatch_router_logits(
            a1q,
            router_logits,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=scales,
        )

        if scales is None:
            a1q, router_logits = res
            a1q_scale = None
        else:
            a1q, router_logits, scales = res
            a1q_scale = _unwrap_scale_and_prepare_for_moe(scales, quant_config)

        return a1q, a1q_scale, router_logits

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
    ) -> torch.Tensor:
        out = get_ep_group().combine(
            fused_expert_output, is_sequence_parallel=self.is_sequence_parallel
        )
        return out


def make_moe_prepare_and_finalize_naive_dp_ep(
    use_monolithic: bool,
    is_sequence_parallel: bool = False,
    num_dispatchers: int = 1,
) -> MoEPrepareAndFinalizeNaiveDPEPModular | MoEPrepareAndFinalizeNaiveDPEPMonolithic:
    # 按执行模式返回 naive DP/EP 的 prepare/finalize 实现。
    return (
        MoEPrepareAndFinalizeNaiveDPEPMonolithic(
            is_sequence_parallel=is_sequence_parallel,
            num_dispatchers=num_dispatchers,
        )
        if use_monolithic
        else MoEPrepareAndFinalizeNaiveDPEPModular(
            is_sequence_parallel=is_sequence_parallel,
            num_dispatchers=num_dispatchers,
        )
    )
