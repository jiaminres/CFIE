# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Conv Layer Class."""

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from cfie.model_executor.custom_op import CustomOp
from cfie.utils.torch_utils import is_torch_equal


class ConvLayerBase(CustomOp):
    """卷积层基类。"""

    # 卷积的维度数，由子类指定
    # 例如：
    # Conv1dLayer -> 1
    # Conv2dLayer -> 2
    # Conv3dLayer -> 3
    num_dim: int

    def __init__(
            self,
            in_channels: int,  # 输入通道数
            out_channels: int,  # 输出通道数
            kernel_size: int | tuple[int, ...],  # 卷积核大小，可传单个 int 或元组
            stride: int | tuple[int, ...] = 1,  # 步长
            padding: int | tuple[int, ...] | Literal["same", "valid"] = 0,  # padding 配置
            dilation: int | tuple[int, ...] = 1,  # 膨胀系数
            groups: int = 1,  # 分组卷积组数
            bias: bool = True,  # 是否使用偏置
            padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",  # padding 模式
            *,
            params_dtype: torch.dtype | None = None,  # 参数 dtype
    ) -> None:
        # 初始化 CustomOp / nn.Module 基类
        super().__init__()

        # 若未指定参数 dtype，则使用当前默认 dtype
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # 合法的 padding 字符串写法
        valid_padding_strings = {"same", "valid"}

        # 如果 padding 是字符串，但不是 "same"/"valid"，则报错
        if isinstance(padding, str) and padding not in valid_padding_strings:
            raise ValueError(
                f"Invalid padding string '{padding}'. "
                f"Expected one of {valid_padding_strings}."
            )

        # 如果 padding="same"，
        # 则自动按 kernel_size // 2 计算 padding
        # 这通常用于保持输出尺寸与输入尺寸一致（stride=1 时）
        if padding == "same":
            padding = (
                kernel_size // 2
                if isinstance(kernel_size, int)
                else tuple(k // 2 for k in kernel_size)
            )

        # 如果 padding="valid"，表示不补零，等价于 padding=0
        elif padding == "valid":
            padding = 0

        # 如果 kernel_size 传的是单个 int，
        # 就复制成 num_dim 维的元组
        # 例如 3D 卷积时，kernel_size=2 -> (2, 2, 2)
        kernel_size = (
            (kernel_size,) * self.num_dim
            if isinstance(kernel_size, int)
            else kernel_size
        )

        # stride 如果是单个 int，同样扩成 num_dim 维元组
        stride = (stride,) * self.num_dim if isinstance(stride, int) else stride

        # padding 如果是单个 int，同样扩成 num_dim 维元组
        padding = (padding,) * self.num_dim if isinstance(padding, int) else padding

        # dilation 如果是单个 int，同样扩成 num_dim 维元组
        dilation = (dilation,) * self.num_dim if isinstance(dilation, int) else dilation

        # 这句其实逻辑上有点“历史遗留味道”：
        # 因为上面若 padding == "same"，已经被转换成具体数字/元组了，
        # 所以下面这个判断几乎不会再命中字符串 "same"
        # 它表达的意思是：same padding 不支持 stride != 1 的情况
        if padding == "same" and any(s != 1 for s in stride):
            raise ValueError("padding='same' is not supported for strided convolutions")

        # 保存输入通道数
        self.in_channels = in_channels

        # 保存输出通道数
        self.out_channels = out_channels

        # 保存卷积核大小（统一为元组）
        self.kernel_size = kernel_size

        # 保存步长（统一为元组）
        self.stride = stride

        # 保存 padding（统一为元组）
        self.padding = padding

        # 保存 dilation（统一为元组）
        self.dilation = dilation

        # 保存 groups
        self.groups = groups

        # 保存 padding mode
        self.padding_mode = padding_mode

        # enable_linear 表示当前卷积是否可以退化/改写成 unfold + linear 的形式
        # 满足条件：
        # 1. kernel_size == stride，说明卷积块之间不重叠
        # 2. 没有 padding
        # 3. groups == 1，不是分组卷积
        #
        # 这类卷积常见于 patch embedding，非常适合改写成矩阵乘法
        self.enable_linear = (
                (self.kernel_size == self.stride)
                and not any(self.padding)
                and self.groups == 1
        )

        # 一个卷积窗口展平后的输入长度
        # = in_channels * kernel_size 各维乘积
        # 例如 Conv3d 中：
        # input_size = C * Kt * Kh * Kw
        self.input_size = in_channels * math.prod(self.kernel_size)

        # 卷积权重参数
        # 形状：
        # [out_channels, in_channels // groups, *kernel_size]
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                *kernel_size,
                dtype=params_dtype,
            ),
        )

        # 如果启用 bias，则创建偏置参数
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_channels, dtype=params_dtype))
        else:
            # 否则将 bias 注册为 None
            self.register_parameter("bias", None)

    def extra_repr(self) -> str:
        # 自定义打印模块时显示的附加信息
        s = f"in_channels={self.in_channels}, "
        s += f"out_channels={self.out_channels}, "
        s += f"kernel_size={self.kernel_size}, "
        s += f"stride={self.stride}, "
        s += f"padding={self.padding}, "
        s += f"bias={self.bias is not None}"
        return s


# --8<-- [start:conv2d]
@CustomOp.register("conv2d")
class Conv2dLayer(ConvLayerBase):
    """Conv layer with Conv2d."""

    # --8<-- [end:conv2d]

    num_dim = 2

    def _forward_mulmat(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        B, C, H, W = x.shape
        K1, K2 = self.kernel_size
        H, W = H // K1, W // K2
        x = x.unfold(2, K1, K1).unfold(3, K2, K2)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(-1, self.input_size)
        x = F.linear(
            x,
            self.weight.view(self.out_channels, self.input_size),
            self.bias,
        )
        x = x.view(B, H, W, self.out_channels).permute(0, 3, 1, 2)
        return x

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return x

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """Expected input shape: (batch_size, in_channels, height, width)"""
        assert x.dim() == 4
        if self.enable_linear:
            return self._forward_mulmat(x)
        else:
            return self._forward_conv(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        # By default, we use CUDNN's convolution ops with optimization.
        return self._forward_conv(x)


class CausalConv2dLayer(Conv2dLayer):
    """
    A causal version of nn.Conv2d where each location in the 2D matrix would
    have no access to locations on its right or down
    All arguments are the same as nn.Conv2d except padding which should be
    set as None
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",
            *,
            params_dtype: torch.dtype | None = None,
    ) -> None:
        if padding is not None:
            raise ValueError(
                "Argument padding should be set to None for CausalConv2dLayer."
            )
        self._left_padding: int = kernel_size - 1
        self._right_padding: int = stride - 1
        padding = 0

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            params_dtype=params_dtype,
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        x = F.pad(x, pad=(self._left_padding, self._right_padding, 0, 0))
        x = super().forward(x)
        return x


# --8<-- [start:conv3d]
@CustomOp.register("conv3d")
class Conv3dLayer(ConvLayerBase):
    """基于 Conv3d 的卷积层封装。"""

    # --8<-- [end:conv3d]

    # 卷积维度数，这里表示 3D 卷积
    num_dim = 3

    def _forward_mulmat(self, x: torch.Tensor) -> torch.Tensor:
        # 要求输入必须是 5 维张量
        # 形状: [B, C, T, H, W]
        assert x.dim() == 5

        # B: batch size
        # C: 输入通道数
        # T/H/W: 时间、高、宽
        B, C, T, H, W = x.shape

        # 三维卷积核大小
        K1, K2, K3 = self.kernel_size

        # 计算卷积后的输出时空尺寸
        # 这里假设 stride 与 kernel 对齐，按整块切分
        T, H, W = T // K1, H // K2, W // K3

        # unfold 相当于把输入按卷积核大小切成一个个局部块
        # 先在时间维切块，再在高维切块，再在宽维切块
        x = x.unfold(2, K1, K1).unfold(3, K2, K2).unfold(4, K3, K3)

        # 重新排列维度：
        # [B, C, T_out, H_out, W_out, K1, K2, K3]
        # -> [B, T_out, H_out, W_out, C, K1, K2, K3]
        # 再 reshape 成二维矩阵，每一行表示一个局部 3D patch
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, self.input_size)

        # 使用全连接等价地实现卷积：
        # 每个局部 patch 与卷积核展平后的权重做线性变换
        x = F.linear(
            x,
            self.weight.view(self.out_channels, self.input_size),
            self.bias,
        )

        # 再把线性结果恢复成 5 维卷积输出格式
        # [B, T_out, H_out, W_out, out_channels] -> [B, out_channels, T_out, H_out, W_out]
        x = x.view(B, T, H, W, self.out_channels).permute(0, 4, 1, 2, 3)

        return x

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        # 要求输入必须是 5 维张量
        assert x.dim() == 5

        # 直接调用 PyTorch 的 3D 卷积实现
        x = F.conv3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return x

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """期望输入形状: (batch_size, in_channels, time, height, width)"""

        # 如果启用了 linear 模式，则不用原生 conv3d，而走 unfold + linear 的等价实现
        #
        # 什么时候会满足 linear 条件？
        # 必须同时满足：
        # 1. kernel_size == stride
        #    表示卷积窗口按块滑动且彼此不重叠
        # 2. padding 全为 0
        #    表示输入边界不做补零
        # 3. groups == 1
        #    表示不是分组卷积
        #
        # 在这种情况下，每个输出位置都只对应输入中的一个独立局部块，
        # 因而可以先把局部块 unfold/展平，再用一次线性层(F.linear)完成与卷积等价的计算
        if self.enable_linear:
            return self._forward_mulmat(x)
        else:
            # 否则直接走普通 conv3d
            return self._forward_conv(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch 2.9.0 / 2.9.1 中，CUDNN 的 Conv3D 默认被禁用了，
        # 导致性能明显下降
        # 相关 issue:
        # https://github.com/cfie-project/cfie/issues/27406
        # https://github.com/pytorch/pytorch/issues/166122
        #
        # 因此在特定版本下，如果 enable_linear=True，
        # 优先走 unfold + linear 的实现，以规避性能问题
        if self.enable_linear and (is_torch_equal("2.9.0") or is_torch_equal("2.9.1")):
            return self._forward_mulmat(x)

        # 其他情况默认走普通 conv3d
        return self._forward_conv(x)
