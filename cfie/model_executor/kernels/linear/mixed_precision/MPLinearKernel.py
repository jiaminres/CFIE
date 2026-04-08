# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch

from cfie.model_executor.layers.quantization.utils import replace_parameter
from cfie.scalar_type import ScalarType


@dataclass
class MPLinearLayerConfig:
    full_weight_shape: tuple[int, int]  # [in, out]
    partition_weight_shape: tuple[int, int]
    weight_type: ScalarType
    act_type: torch.dtype
    group_size: int
    zero_points: bool
    has_g_idx: bool
    out_type: torch.dtype | None = None


class MPLinearKernel(ABC):
    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        # 返回当前 kernel backend 支持的最低 GPU 计算能力
        #
        # 例如：
        # - MarlinLinearKernel 返回 75
        # - MacheteLinearKernel 返回 90
        #
        # choose_mp_linear_kernel(...) 会先用这个值做一轮粗筛
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        # 判断当前 kernel backend 能否实现给定的线性层配置 c
        #
        # c 里通常包含：
        # - full_weight_shape       : 全局权重 shape = [K_global, N_global]
        # - partition_weight_shape  : 本地权重 shape = [K_local, N_local]
        # - weight_type             : 权重量化类型
        # - act_type                : 激活 dtype
        # - group_size              : 分组量化大小
        # - zero_points             : 是否启用 zp
        # - has_g_idx               : 是否启用 desc_act / activation-order
        #
        # 返回：
        # - bool: 当前 kernel 能不能做
        # - str|None: 若不能做，失败原因是什么
        raise NotImplementedError

    def __init__(
        self,
        c: MPLinearLayerConfig,          # 当前 kernel 需要实现的线性层配置
        w_q_param_name: str,             # layer 上 qweight 参数的名字
        w_s_param_name: str,             # layer 上 scales 参数的名字
        w_zp_param_name: str | None = None,    # layer 上 zero-points 参数名（若有）
        w_gidx_param_name: str | None = None,  # layer 上 g_idx 参数名（若有）
    ) -> None:
        # 构造时再次断言：当前 kernel 必须能实现这个 config
        #
        # 注意：
        # can_implement(...) 返回的是 (bool, reason) 二元组
        # 这里直接 assert 它，依赖 Python 中非空 tuple 为真值；
        # 这种写法从“表达清晰度”上不算最好，但源码里就是这么写的
        #
        # 语义上它想表达的是：
        #   “既然已经选中了这个 kernel，那它必须可实现当前 config”
        assert self.can_implement(c)

        # 保存配置对象，后续 process/apply 都会用
        self.config = c

        # 保存 layer 上各个参数的属性名
        #
        # 这样 kernel backend 就不需要假设 layer 上参数一定叫固定名字，
        # 而是通过这些字符串去 getattr(...)
        self.w_q_name = w_q_param_name
        self.w_s_name = w_s_param_name

        # 若 config 说启用了 zero_points，则必须提供 zp 参数名
        if c.zero_points:
            assert w_zp_param_name is not None

        # 若 config 说启用了 g_idx / desc_act，则必须提供 g_idx 参数名
        if c.has_g_idx:
            assert w_gidx_param_name is not None

        # 保存可选参数名
        self.w_zp_name = w_zp_param_name
        self.w_gidx_name = w_gidx_param_name

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # checkpoint 权重加载完成后，对 layer 上的量化参数做后处理
        #
        # 常见操作包括：
        # - repack qweight
        # - permute scales
        # - 转换 zero-points
        # - 处理 g_idx / 排序索引
        # - 分配 workspace
        # - permute bias
        #
        # 不同 kernel backend 的具体做法不同，所以由子类实现
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,                     # 输入激活，shape 常见: [..., K_local]
        bias: torch.Tensor | None = None,   # 本地 bias，shape 常见: [N_local]
    ) -> torch.Tensor:
        # forward 时真正调用 kernel 计算输出
        #
        # 逻辑输入:
        #   x shape 常见为 [..., K_local]
        #
        # 逻辑输出:
        #   [..., N_local]
        #
        # 这里通常不会做 all-gather，也不会恢复全局输出，
        # 而只是当前 rank 的 local output
        raise NotImplementedError

    def _transform_param(
        self, layer: torch.nn.Module, name: str | None, fn: Callable
    ) -> None:
        # 这是一个通用辅助函数：
        # 用于把 layer 上某个参数拿出来，做一次变换，再替换回去
        #
        # 常用于：
        # - qweight repack
        # - scales permute
        # - zp 转换
        # - g_idx 重新整理
        #
        # 参数：
        # - layer: 当前线性层对象
        # - name : 目标参数名，例如 "qweight" / "scales" / "qzeros" / "g_idx"
        # - fn   : 变换函数，输入旧参数，输出变换后的参数对象/张量包装
        if name is not None and getattr(layer, name, None) is not None:
            # 取出原参数
            old_param = getattr(layer, name)

            # 执行变换函数
            new_param = fn(old_param)

            # 把参数替换成普通 torch.nn.Parameter
            #
            # 注释里写得很明确：
            # 这是为了兼容 TorchDynamo
            #
            # 也就是说：
            # - 前面加载/处理中，参数可能还是 BasevLLMParameter 这种自定义类
            # - 但运行期为了图捕获/编译兼容，最终更希望是普通 nn.Parameter
            replace_parameter(
                layer, name, torch.nn.Parameter(new_param.data, requires_grad=False)
            )

    def _get_weight_params(
        self, layer: torch.nn.Module
    ) -> tuple[
        torch.Tensor,       # w_q
        torch.Tensor,       # w_s
        torch.Tensor | None,  # w_zp
        torch.Tensor | None,  # w_gidx
    ]:
        # 从 layer 上按名字取出当前 kernel 需要的参数
        #
        # 返回顺序固定为：
        # - w_q    : qweight
        # - w_s    : scales
        # - w_zp   : zero-points（若无则 None）
        # - w_gidx : g_idx（若无则 None）
        #
        # 这里用了：
        #   getattr(layer, self.w_zp_name or "", None)
        # 这种写法的意思是：
        # - 若 self.w_zp_name 为 None
        # - 就去 getattr(layer, "", None)
        # - 最终自然返回 None
        #
        # 所以它能统一兼容“有/没有 zp”两种情况
        return (
            getattr(layer, self.w_q_name),
            getattr(layer, self.w_s_name),
            getattr(layer, self.w_zp_name or "", None),
            getattr(layer, self.w_gidx_name or "", None),
        )
