# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import Any

from cfie.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    activation_without_mul,
    apply_moe_activation,
)
from cfie.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    RoutingMethodType,
)
from cfie.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from cfie.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from cfie.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular,
)
from cfie.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from cfie.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from cfie.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
from cfie.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from cfie.model_executor.layers.fused_moe.zero_expert_fused_moe import (
    ZeroExpertFusedMoE,
)
from cfie.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    BatchedDeepGemmExperts,
)
from cfie.model_executor.layers.fused_moe.cutlass_moe import (
    CutlassBatchedExpertsFp8,
    CutlassExpertsFp8,
    CutlassExpertsW4A8Fp8,
    cutlass_moe_w4a8_fp8,
)
from cfie.model_executor.layers.fused_moe.deep_gemm_moe import DeepGemmExperts
from cfie.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedTritonExperts,
)
from cfie.model_executor.layers.fused_moe.fused_moe import (
    TritonExperts,
    TritonWNA16Experts,
    fused_experts,
    get_config_file_name,
)
from cfie.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    AiterExperts,
)
from cfie.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk,
)
from cfie.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopk,
)
from cfie.model_executor.layers.fused_moe.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts,
)
from cfie.model_executor.layers.fused_moe.xpu_fused_moe import (
    XPUExperts,
    XPUExpertsFp8,
)

_config: dict[str, Any] | None = None


@contextmanager
def override_config(config):
    global _config
    old_config = _config
    _config = config
    yield
    _config = old_config


def get_config() -> dict[str, Any] | None:
    return _config


__all__ = [
    "FusedMoE",
    "FusedMoERouter",
    "FusedMoEConfig",
    "FusedMoEMethodBase",
    "MoEActivation",
    "UnquantizedFusedMoEMethod",
    "FusedMoeWeightScaleSupported",
    "FusedMoEExpertsModular",
    "FusedMoEActivationFormat",
    "FusedMoEPrepareAndFinalizeModular",
    "GateLinear",
    "RoutingMethodType",
    "SharedFusedMoE",
    "ZeroExpertFusedMoE",
    "activation_without_mul",
    "apply_moe_activation",
    "override_config",
    "get_config",
]
__all__ += [
    "AiterExperts",
    "fused_topk",
    "fused_experts",
    "get_config_file_name",
    "GroupedTopk",
    "cutlass_moe_w4a8_fp8",
    "CutlassExpertsFp8",
    "CutlassBatchedExpertsFp8",
    "CutlassExpertsW4A8Fp8",
    "TritonExperts",
    "TritonWNA16Experts",
    "BatchedTritonExperts",
    "DeepGemmExperts",
    "BatchedDeepGemmExperts",
    "TritonOrDeepGemmExperts",
    "XPUExperts",
    "XPUExpertsFp8",
]
