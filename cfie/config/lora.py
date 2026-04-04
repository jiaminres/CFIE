# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from cfie.config.utils import config
from cfie.logger import init_logger
from cfie.utils.hashing import safe_hash

if TYPE_CHECKING:
    from cfie.config import ModelConfig
    from cfie.config.cache import CacheConfig
else:
    ModelConfig = Any
    CacheConfig = Any

logger = init_logger(__name__)

# LoRA 权重 dtype 的可选枚举。
LoRADType = Literal["auto", "float16", "bfloat16"]
# 支持的 LoRA rank 上限集合。
MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]
# 允许的额外词表大小集合。
LoRAExtraVocabSize = Literal[256, 512]


@config(config=ConfigDict(arbitrary_types_allowed=True))
class LoRAConfig:
    """Configuration for LoRA."""

    # 单个 LoRA adapter 允许的最大 rank。
    max_lora_rank: MaxLoRARanks = 16
    """Max LoRA rank."""
    # 一个 batch 内最多允许同时激活多少个 LoRA。
    max_loras: int = Field(default=1, ge=1)
    """Max number of LoRAs in a single batch."""
    # 是否把 LoRA 计算完全切到 tensor parallel 各 shard 上。
    fully_sharded_loras: bool = False
    """By default, only half of the LoRA computation is sharded with tensor
    parallelism. Enabling this will use the fully sharded layers. At high
    sequence length, max rank or tensor parallel size, this is likely faster.
    """
    # 最多在 CPU 内存里常驻多少个 LoRA；为空时回退到 max_loras。
    max_cpu_loras: int | None = None
    """Maximum number of LoRAs to store in CPU memory. Must be >= than
    `max_loras`."""
    # LoRA 自身的计算 dtype；auto 会跟随底模 dtype。
    lora_dtype: torch.dtype | LoRADType = "auto"
    """Data type for LoRA. If auto, will default to base model dtype."""
    # 多模态场景下，按模态名映射默认启用的 LoRA 路径。
    default_mm_loras: dict[str, str] | None = None
    """Dictionary mapping specific modalities to LoRA model paths; this field
    is only applicable to multimodal models and should be leveraged when a
    model always expects a LoRA to be active when a given modality is present.
    Note that currently, if a request provides multiple additional
    modalities, each of which have their own LoRA, we do NOT apply
    default_mm_loras because we currently only support one lora adapter
    per prompt. When run in offline mode, the lora IDs for n modalities
    will be automatically assigned to 1-n with the names of the modalities
    in alphabetic order."""
    # 是否为视觉 tower / connector 打开 LoRA。
    enable_tower_connector_lora: bool = False
    """If `True`, LoRA support for the tower (vision encoder) and connector 
    of multimodal models will be enabled. This is an experimental feature and 
    currently only supports some MM models such as the Qwen VL series. The default 
    is False."""
    # 是否按“当前激活的 LoRA 数量”专门化 kernel grid / cuda graph。
    specialize_active_lora: bool = False
    """Whether to construct lora kernel grid by the number of active LoRA adapters.
    When set to True, separate cuda graphs will be captured for different counts
    of active LoRAs (powers of 2 up to max_loras), which can improve performance
    for variable LoRA usage patterns at the cost of increased startup time and
    memory usage. Only takes effect when cudagraph_specialize_lora is True.
    """

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
        # LoRA 的图相关哈希只关心会改变执行形态的关键字段。
        factors: list[Any] = []
        # 最大 rank 会直接影响 LoRA 权重与 kernel 形状。
        factors.append(self.max_lora_rank)
        # 单批次最大 LoRA 数量也会影响调度与图捕获。
        factors.append(self.max_loras)
        # fully sharded 与否会改变 TP 下的计算路径。
        factors.append(self.fully_sharded_loras)
        # LoRA dtype 会改变算子 dtype。
        factors.append(self.lora_dtype)
        # 多模态 tower/connector LoRA 会改变额外模块是否参与 LoRA 路径。
        factors.append(self.enable_tower_connector_lora)

        # 返回这些因子的稳定哈希。
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @model_validator(mode="after")
    def _validate_lora_config(self) -> Self:
        # max_cpu_loras 未指定时，默认至少能容纳当前 batch 最大 LoRA 数。
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras
        # 若显式给出的 CPU 容量比 max_loras 还小，则配置无效。
        elif self.max_cpu_loras < self.max_loras:
            raise ValueError(
                f"max_cpu_loras ({self.max_cpu_loras}) must be >= "
                f"max_loras ({self.max_loras})."
            )

        # 返回当前对象，符合 pydantic model_validator 约定。
        return self

    def verify_with_model_config(self, model_config: ModelConfig):
        # auto/None 时直接继承底模 dtype。
        if self.lora_dtype in (None, "auto"):
            self.lora_dtype = model_config.dtype
        # 字符串形式的 dtype 转成 torch.dtype 对象。
        elif isinstance(self.lora_dtype, str):
            self.lora_dtype = getattr(torch, self.lora_dtype)
