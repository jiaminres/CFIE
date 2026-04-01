"""模型加载抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from cfie.config.schema import LoadConfig, ModelConfig


@dataclass(slots=True)
class LoaderStats:
    """加载统计信息。"""

    file_count: int
    total_bytes: int
    tier_mapping: str


class BaseModelLoader(ABC):
    """模型加载器抽象基类。"""

    @abstractmethod
    def download_model(self,
                       model_config: ModelConfig,
                       load_config: LoadConfig) -> None:
        """确保模型已可访问（本地目录或缓存目录）。"""

    @abstractmethod
    def load_weights(self,
                     model: Any,
                     model_config: ModelConfig,
                     load_config: LoadConfig) -> None:
        """把权重加载到模型对象。"""

    @abstractmethod
    def load_model(self,
                   model_config: ModelConfig,
                   load_config: LoadConfig) -> Any:
        """返回已初始化的模型对象。"""
