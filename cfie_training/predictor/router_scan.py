"""Helpers for discovering MoE router modules from teacher models."""

from __future__ import annotations

import re

import torch
import torch.nn as nn


def router_module_output_dim(module: nn.Module) -> int | None:
    # 优先读取标准线性层的输出维度字段。
    out_features = getattr(module, "out_features", None)
    if isinstance(out_features, int):
        return out_features
    # 兼容直接暴露 num_experts 的 router 模块。
    num_experts = getattr(module, "num_experts", None)
    if isinstance(num_experts, int):
        return num_experts
    # 最后回退到权重首维。
    weight = getattr(module, "weight", None)
    if torch.is_tensor(weight) and weight.ndim >= 2:
        return int(weight.shape[0])
    return None


def discover_router_modules_from_model(
    model: nn.Module,
    *,
    expected_num_experts: int,
) -> tuple[tuple[int, str, nn.Module], ...]:
    # ------------------------------- 扫描模型并发现可用 router gate 模块 -------------------------------
    # 返回结构为 (layer_index, module_name, module_obj) 元组列表，
    # 后续会按层序注册 forward hook 抓取 hidden_state 与 router logits。
    discovered: list[tuple[int, str, nn.Module]] = []
    # 从模块全名里抽取 layer 索引，兼容常见的 transformers 命名约定。
    layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

    # ------------------------------- 逐模块筛选候选 gate -------------------------------
    for name, module in model.named_modules():
        # 先按名称粗筛，只保留包含 ".gate" 的模块路径。
        if ".gate" not in name:
            continue
        # 无法解析层号的模块不参与捕获，避免混入非层级 gate。
        layer_match = layer_pattern.search(name)
        if layer_match is None:
            continue
        # 再按输出维精筛，确保候选模块的输出维等于专家数。
        output_dim = router_module_output_dim(module)
        if output_dim != expected_num_experts:
            continue
        # 记录当前候选模块，待最终按层号排序后统一返回。
        discovered.append((int(layer_match.group(1)), name, module))

    # ------------------------------- 校验扫描结果并按层号排序 -------------------------------
    # 若一个候选都没找到，直接报错，避免后续 forward-capture 进入空监督路径。
    if not discovered:
        raise ValueError(
            "could not discover any router gate modules from the teacher model"
        )
    # 保证返回结果按 layer_index 递增，便于后续按层序消费。
    discovered.sort(key=lambda item: item[0])
    # 返回不可变元组，避免调用方意外修改扫描结果。
    return tuple(discovered)
