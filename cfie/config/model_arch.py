# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from cfie.logger import init_logger

logger = init_logger(__name__)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ModelArchitectureConfig:
    """
    Configuration for model architecture that required by vLLM runtime
    """

    # Hugging Face 暴露的 architecture 类名列表。
    architectures: list[str] | None
    """List of model architecture class names (e.g., ['LlamaForCausalLM']).
       It can be None upon calling `cfie_config.with_hf_config(config.text_config)`"""

    # 模型主类型标识。
    model_type: str
    """Model type identifier (e.g., 'llama', 'gpt_oss')."""

    # 文本子模型类型标识，多模态模型里可能与 model_type 不同。
    text_model_type: str | None
    """Text model type identifier (e.g., 'llama4_text')."""

    # 隐藏层维度。
    hidden_size: int
    """Hidden size of the model."""

    # 总层数。
    total_num_hidden_layers: int
    """Number of hidden layers in the model."""

    # 总注意力头数。
    total_num_attention_heads: int
    """Number of attention heads in the model."""

    # 单个头的维度。
    head_size: int
    """Head dimension of the model."""

    # 词表大小。
    vocab_size: int
    """Vocabulary size of the model."""

    # KV 头总数。
    total_num_kv_heads: int
    """Number of key value heads in the model."""

    # MoE 模型中的专家数；非 MoE 模型通常为 0。
    num_experts: int
    """Number of experts in the model."""

    # 量化配置原始字典。
    quantization_config: dict[str, Any] | None
    """Quantization configuration dictionary containing quantization parameters."""

    # 是否是 DeepSeek MLA 结构。
    is_deepseek_mla: bool
    """Whether the model is a DeepSeek MLA model."""

    # 从 HF 配置推导出的最大上下文长度以及对应 key 名。
    derived_max_model_len_and_key: tuple[float, str | None]
    """Derived maximum model length and key from the hf config."""
