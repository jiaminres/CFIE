# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any, Literal

from pydantic import Field, field_validator

from cfie.config.utils import config
from cfie.utils.hashing import safe_hash

# MoE kernel backend 的可选枚举。
MoEBackend = Literal[
    "auto",
    "triton",
    "deep_gemm",
    "cutlass",
    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_cutedsl",
    "marlin",
    "aiter",
]


@config
class KernelConfig:
    """Configuration for kernel selection and warmup behavior."""

    # 是否在 warmup 阶段执行 FlashInfer autotune。
    enable_flashinfer_autotune: bool = Field(default=None)
    """If True, run FlashInfer autotuning during kernel warmup."""

    # 选择 routed experts 计算所使用的 MoE kernel backend。
    moe_backend: MoEBackend = "auto"
    """Backend for MoE expert computation kernels. Available options:

    - "auto": Automatically select the best backend based on model and hardware\n
    - "triton": Use Triton-based fused MoE kernels\n
    - "deep_gemm": Use DeepGEMM kernels (FP8 block-quantized only)\n
    - "cutlass": Use vLLM CUTLASS kernels\n
    - "flashinfer_trtllm": Use FlashInfer with TRTLLM-GEN kernels\n
    - "flashinfer_cutlass": Use FlashInfer with CUTLASS kernels\n
    - "flashinfer_cutedsl": Use FlashInfer with CuteDSL kernels (FP4 only)\n
    - "marlin": Use Marlin kernels (weight-only quantization)\n
    - "aiter": Use AMD AITer kernels (ROCm only)"""

    @field_validator("moe_backend", mode="before")
    @classmethod
    def _normalize_moe_backend(cls, value: Any) -> Any:
        # 字符串形式统一转成小写并把短横线替换成下划线。
        if isinstance(value, str):
            return value.lower().replace("-", "_")
        # 非字符串值保持原样，交给 pydantic 后续处理。
        return value

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        # 当前 kernel config 不参与图结构哈希，只保留空 factors。
        factors: list[Any] = []
        # 基于空 factors 计算稳定哈希。
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        # 返回最终哈希字符串。
        return hash_str

    @field_validator("enable_flashinfer_autotune", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialization is delayed."""
        # 延迟初始化阶段允许字段暂时保持 None。
        if value is None:
            return value
        # 否则继续走 pydantic 原有校验逻辑。
        return handler(value)
