# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch.nn.parameter import Parameter

from cfie.model_executor.custom_op import PluggableLayer
from cfie.model_executor.layers.linear import ReplicatedLinear
from cfie.platforms import current_platform


@PluggableLayer.register("gate_linear")
class GateLinear(ReplicatedLinear):
    # MoE gate 使用的线性层实现，内部按三档 GEMM 路径调度：
    # 1. DSV3 专用 kernel
    # 2. cuBLAS bf16->fp32
    # 3. 退回 ReplicatedLinear / F.linear
    # out_dtype 允许在构造后再设置，适合依赖后续量化配置才能确定输出 dtype 的场景。

    # DSV3 专用 kernel 当前支持的 expert 数与 hidden size。
    DSV3_SUPPORTED_NUM_EXPERTS = [256, 384]
    DSV3_SUPPORTED_HIDDEN_SIZES = [7168]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        prefix: str = "",
    ):
        is_hopper_or_blackwell = current_platform.is_device_capability(
            (9, 0)
        ) or current_platform.is_device_capability_family(100)
        can_use_specialized_kernels = (
            current_platform.is_cuda() and is_hopper_or_blackwell and not bias
        )

        # 若要求 fp32 计算、但当前又不能走专用 kernel，
        # 就直接把权重存成 fp32，让第三档回退路径也能原生用 fp32 计算。
        if force_fp32_compute and not can_use_specialized_kernels:
            params_dtype = torch.float32

        super().__init__(
            input_size,
            output_size,
            bias=bias,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=prefix,
        )
        self.out_dtype = out_dtype

        # DSV3 专用 kernel 的可用条件：平台支持且维度完全命中。
        self.allow_specialized_router_gemm = can_use_specialized_kernels
        self.allow_dsv3_router_gemm = (
            self.allow_specialized_router_gemm
            and output_size in self.DSV3_SUPPORTED_NUM_EXPERTS
            and input_size in self.DSV3_SUPPORTED_HIDDEN_SIZES
        )

        # cuBLAS bf16->fp32 路径的可用条件。
        self.allow_cublas_router_gemm = (
            self.allow_specialized_router_gemm
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )

    def set_out_dtype(self, out_dtype: torch.dtype) -> None:
        # 允许在初始化后再补设 router logits 的输出 dtype。
        # 常见于 gate 构造时还不知道 experts 最终量化方式的场景。
        if self.out_dtype is not None:
            raise ValueError("out_dtype has already been set")
        self.out_dtype = out_dtype

        if (
            not self.allow_cublas_router_gemm
            and self.allow_specialized_router_gemm
            and out_dtype == torch.float32
        ):
            self.allow_cublas_router_gemm = self.weight.dtype == torch.bfloat16

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        import cfie._custom_ops as ops

        # 第一档：优先走 DSV3 专用 kernel。
        if self.allow_dsv3_router_gemm and x.shape[0] <= 16:
            output = ops.dsv3_router_gemm(
                hidden_states=x,
                router_weight=self.weight,
                output_dtype=self.out_dtype,
            )
            return output, None

        # 第二档：走 cuBLAS bf16->fp32。
        if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:
            output = ops.router_gemm_bf16_fp32(x, self.weight)
            return output, None

        # 第三档：退回 ReplicatedLinear / F.linear 通用路径。
        if self.out_dtype is not None and x.dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)
        output, output_bias = super().forward(x)
        if self.out_dtype is not None and output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        return output, output_bias
