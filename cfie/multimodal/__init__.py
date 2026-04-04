# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .hasher import MultiModalHasher  # 多模态输入的哈希工具，用于缓存/去重/标识等用途
from .inputs import (
    BatchedTensorInputs,         # 批量化后的张量输入类型
    ModalityData,                # 单个模态的数据抽象类型
    MultiModalDataBuiltins,      # 多模态数据的内建类型定义
    MultiModalDataDict,          # 多模态数据字典类型，通常按模态名组织数据
    MultiModalKwargsItems,       # 多模态关键字参数项类型
    MultiModalPlaceholderDict,   # 多模态占位符字典，描述文本中多模态内容插入位置
    MultiModalUUIDDict,          # 多模态数据对应的 UUID 字典，用于唯一标识资源
    NestedTensors,               # 嵌套张量类型，用于表示复杂张量结构
)
from .registry import MultiModalRegistry  # 多模态注册表类

# 创建一个全局多模态注册表实例
# 这个注册表会在运行时根据“目标模型”找到对应的多模态处理逻辑
MULTIMODAL_REGISTRY = MultiModalRegistry()

"""
全局的 [`MultiModalRegistry`][cfie.multimodal.registry.MultiModalRegistry]
会被模型运行器（model runners）使用，
用于根据目标模型分发对应的数据处理流程。

也就是说：
- 不同模型支持的模态不同（如 text / image / video）
- 不同模型的预处理方式也不同
- 运行时会通过这个全局注册表，查到当前模型该使用哪套多模态处理器

更多信息：
    [mm_processing](../../../design/mm_processing.md)
"""

# 定义这个模块对外公开导出的符号
# 当外部使用 `from cfie.multimodal import *` 时，下面这些名字会被导出
__all__ = [
    "BatchedTensorInputs",       # 批量张量输入类型
    "ModalityData",              # 单模态数据类型
    "MultiModalDataBuiltins",    # 多模态内建数据类型
    "MultiModalDataDict",        # 多模态数据字典类型
    "MultiModalHasher",          # 多模态哈希工具
    "MultiModalKwargsItems",     # 多模态 kwargs 项类型
    "MultiModalPlaceholderDict", # 多模态占位符字典类型
    "MultiModalUUIDDict",        # 多模态 UUID 字典类型
    "NestedTensors",             # 嵌套张量类型
    "MULTIMODAL_REGISTRY",       # 全局多模态注册表实例
    "MultiModalRegistry",        # 多模态注册表类本身
]
