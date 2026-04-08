# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utils for model executor."""

import copy
from typing import Any

import torch

from cfie.utils.torch_utils import is_torch_equal_or_newer


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: dict[str, Any] | None,
):
    """
    给权重张量挂载额外属性。

    这里不会覆盖张量上已经存在的同名属性。

    参数：
    - `weight`：目标权重张量
    - `weight_attrs`：要挂到权重张量上的属性字典
    """
    # 未提供属性字典时直接返回。
    if weight_attrs is None:
        return
    # 逐项把属性写入到目标权重张量上。
    for key, value in weight_attrs.items():
        # 禁止覆盖已有属性，避免把已有语义悄悄改掉。
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"

        # 延迟导入当前平台对象，避免模块初始化阶段引入额外平台依赖。
        from cfie.platforms import current_platform

        # TPU 上常见的加载写法是先对 param.data 做 narrow，再把真实权重 copy_ 进去。
        # 这条路径原本期望 narrow 视图和基张量共享同一份底层存储。
        # 但 TPU 可能会把视图写入延迟回传到基张量，导致中间多占一份内存。
        # 权重加载阶段这会放大峰值内存，严重时触发 OOM。
        # 因此这里对 `weight_loader` 做一层平台包装，让加载后显式同步基张量。
        if current_platform.use_sync_weight_loader() and key == "weight_loader":
            value = current_platform.make_synced_weight_loader(value)
        # 把最终属性写回到权重张量对象上。
        setattr(weight, key, value)


def replace_parameter(
    layer: torch.nn.Module, param_name: str, new_data: torch.Tensor | None
):
    """
    Replace a parameter of a layer while maintaining the ability to reload the weight.
    Called within implementations of the `process_weights_after_loading` method.

    This function should not be called on weights which are tied/shared

    Args:
        layer: Layer containing parameter to replace
        param_name: Name of parameter to replace
        new_data: New data of the new parameter, or None to set the parameter to None
    """
    # should not be used on a tied/shared param

    # If new_data is None, set the parameter to None
    if new_data is None:
        setattr(layer, param_name, None)
        return

    if isinstance(new_data, torch.nn.Parameter):
        new_data = new_data.data
    new_param = torch.nn.Parameter(new_data, requires_grad=False)

    old_param: torch.nn.Parameter | None = getattr(layer, param_name, None)
    if old_param is not None and hasattr(old_param, "weight_loader"):
        weight_loader = old_param.weight_loader
        set_weight_attrs(new_param, {"weight_loader": weight_loader})

    setattr(layer, param_name, new_param)


def get_packed_modules_mapping(model: torch.nn.Module) -> dict[str, list[str]]:
    parent_map = getattr(model, "packed_modules_mapping", None)
    parent_map = copy.deepcopy(parent_map) if parent_map is not None else {}

    # don't infer mapping if the model has defined it explicitly.
    if parent_map:
        return parent_map

    # We only check main components instead of whole model submodules
    for child in model.children():
        child_map = getattr(child, "packed_modules_mapping", None)
        child_map = copy.deepcopy(child_map) if child_map is not None else {}

        if any((k in parent_map and parent_map[k] != v) for k, v in child_map.items()):
            raise ValueError(
                f"Can't update {type(model).__name__}'s packed_modules_mapping "
                f"safely because of conflicts from {type(child).__name__}."
            )
        else:
            parent_map.update(child_map)
    return parent_map


def get_moe_expert_mapping(
    model: torch.nn.Module,
) -> list[tuple[str, str, int, str]]:
    if parent_map := getattr(model, "get_expert_mapping", None):
        return parent_map()
    else:
        # We only check main components instead of whole model submodules
        for child in model.children():
            child_map = getattr(child, "get_expert_mapping", None)
            if child_map is not None:
                return child_map()
        return []


def maybe_disable_graph_partition(current_backend: str) -> dict[str, bool]:
    if current_backend == "inductor" and is_torch_equal_or_newer("2.9.0.dev"):
        return {"graph_partition": False}
    else:
        return {}
