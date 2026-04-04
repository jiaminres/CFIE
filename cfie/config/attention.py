# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from pydantic import field_validator

from cfie.config.utils import config
from cfie.v1.attention.backends.registry import AttentionBackendEnum


@config
class AttentionConfig:
    """Configuration for attention mechanisms in vLLM."""

    # attention backend；None/auto 表示运行时自动选择。
    backend: AttentionBackendEnum | None = None
    """Attention backend to use. Use "auto" or None for automatic selection."""

    # 是否强制指定 flash-attention 版本。
    flash_attn_version: Literal[2, 3, 4] | None = None
    """Force cfie to use a specific flash-attention version (2, 3, or 4).
    Only valid when using the flash-attention backend."""

    # 是否把 prefill 和 decode attention 拆成独立 kernel。
    use_prefill_decode_attention: bool = False
    """Use separate prefill and decode kernels for attention instead of
    the unified triton kernel."""

    # CUDA graph decode 阶段允许的 flash-attention 最大 split 数。
    flash_attn_max_num_splits_for_cuda_graph: int = 32
    """Flash Attention max number splits for cuda graph decode."""

    # 是否启用 cudnn prefill 路径。
    use_cudnn_prefill: bool = False
    """Whether to use cudnn prefill."""

    # 是否启用 TRTLLM ragged DeepSeek prefill 路径。
    use_trtllm_ragged_deepseek_prefill: bool = False
    """Whether to use TRTLLM ragged deepseek prefill."""

    # 是否在 flashinfer 中启用 TRTLLM attention backend。
    use_trtllm_attention: bool | None = None
    """If set to True/False, use or don't use the TRTLLM attention backend
    in flashinfer. If None, auto-detect the attention backend in flashinfer."""

    # 是否禁用 flashinfer prefill。
    disable_flashinfer_prefill: bool = True
    """Whether to disable flashinfer prefill."""

    # 使用 fp8 kv 时，是否禁止把 Q 也量化成 fp8。
    disable_flashinfer_q_quantization: bool = False
    """If set, when using fp8 kv, do not quantize Q to fp8."""

    # 是否在 prefill attention 中量化 query。
    use_prefill_query_quantization: bool = False
    """If set, quantize query for attention in prefill."""

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        from cfie.config.utils import get_hash_factors, hash_factors

        # 当前 attention 配置没有额外排除项，全部交给 get_hash_factors 处理。
        ignored_factors: list[str] = []
        # 收集所有参与 attention 图形态的字段。
        factors = get_hash_factors(self, ignored_factors)
        # 返回这些字段的稳定哈希。
        return hash_factors(factors)

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the `backend` enum type from string.

        The special value "auto" is treated as None, which triggers
        automatic backend selection.
        """
        # 字符串输入需要先转成 AttentionBackendEnum 或 None。
        if isinstance(value, str):
            # auto 特判为 None，表示后续自动选择。
            if value.lower() == "auto":
                return None
            # 其他字符串按枚举名查找对应 backend。
            return AttentionBackendEnum[value.upper()]
        # 非字符串值保持原样。
        return value
