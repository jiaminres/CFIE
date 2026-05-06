from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cfie import _custom_ops as ops
from cfie.model_executor.layers.quantization.utils.quant_utils import gptq_pack
from cfie.op_validation.gptq_marlin_fp8 import (
    GPTQMarlinFP8Linear,
    load_opcheck_library,
)

# 当前验证依赖 CUDA 算子，纯 CPU 环境下直接跳过整个模块。
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def _dequantize_reference(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    group_size: int,
) -> torch.Tensor:
    # 分配解包后的 int4 权重矩阵 `unpacked: [K, N]`。
    unpacked = torch.empty((size_k, size_n), device=qweight.device, dtype=torch.int32)

    # 逐个 4-bit 槽位从 `qweight: [K / 8, N]` 里恢复出逻辑行。
    for offset in range(8):
        # 把当前槽位的 4-bit 无符号值写回 `unpacked[offset::8]: [K / 8, N]`。
        unpacked[offset::8] = ((qweight >> (offset * 4)) & 0xF).to(torch.int32)

    # 将无符号区间 `[0, 15]` 平移回 GPTQ 逻辑区间 `[-8, 7]`。
    unpacked = unpacked - 8

    # 分配最终的浮点参考权重矩阵 `dense: [K, N]`。
    dense = torch.empty((size_k, size_n), device=qweight.device, dtype=torch.float32)

    # 计算 group 量化下总共有多少个 scale 分组。
    num_groups = 1 if group_size == -1 else size_k // group_size

    # 计算每个 scale 分组覆盖多少行。
    rows_per_group = size_k if group_size == -1 else group_size

    # 逐组把 `unpacked` 乘对应 scale，恢复成浮点权重。
    for group_index in range(num_groups):
        # 计算当前 group 在 `dense: [K, N]` 中的起始行。
        start = group_index * rows_per_group

        # 计算当前 group 在 `dense: [K, N]` 中的结束行。
        end = min(start + rows_per_group, size_k)

        # 将 `unpacked[start:end]: [rows, N]` 乘 `scales[group_index]: [N]`。
        dense[start:end] = (
            unpacked[start:end].to(torch.float32)
            * scales[group_index].to(torch.float32)
        )

    # 返回参考浮点权重矩阵 `dense: [K, N]`。
    return dense


def test_gptq_marlin_fp8_validation_forward_and_backward() -> None:
    try:
        # 加载验证动态库，确保 `opcheck_C` 相关算子已经注册。
        library_path = load_opcheck_library()
    except FileNotFoundError as exc:
        # 若本地尚未构建验证目标，则将当前测试标记为跳过。
        pytest.skip(str(exc))

    # 确认本次测试确实拿到了一个有效的动态库路径。
    assert library_path

    # 固定测试设备为 CUDA。
    device = torch.device("cuda")

    # 固定逻辑输入维 `K`。
    size_k = 128

    # 固定逻辑输出维 `N`。
    size_n = 128

    # 固定 GPTQ group 大小。
    group_size = 32

    # 计算总 group 数 `num_groups = K / group_size`。
    num_groups = size_k // group_size

    # 固定 batch 大小 `B`。
    batch = 8

    # 构造逻辑 int4 权重 `logical_qweight: [K, N]`，值域为 `[-8, 7]`。
    logical_qweight = torch.randint(
        -8, 8, (size_k, size_n), device=device, dtype=torch.int32
    )

    # 将逻辑 int4 权重打包成 GPTQ 存储布局 `packed_qweight: [K / 8, N]`。
    packed_qweight = gptq_pack(
        (logical_qweight + 8).to(torch.int32),
        4,
        size_k,
        size_n,
    ).contiguous()

    # 构造 group scales `scales: [num_groups, N]`。
    scales = (
        torch.rand((num_groups, size_n), device=device, dtype=torch.float16) * 0.05
        + 0.01
    ).contiguous()

    # 通过纯 PyTorch 参考路径恢复浮点权重 `dense_ref: [K, N]`。
    dense_ref = _dequantize_reference(
        packed_qweight,
        scales,
        size_k=size_k,
        size_n=size_n,
        group_size=group_size,
    )

    # 调用验证算子恢复正向布局浮点权重 `dense_from_op: [K, N]`。
    dense_from_op = torch.ops.opcheck_C.gptq_marlin_fp8_dequantize(
        packed_qweight,
        scales,
        size_k,
        size_n,
        group_size,
        False,
    )

    # 调用验证算子恢复转置布局浮点权重 `dense_t_from_op: [N, K]`。
    dense_t_from_op = torch.ops.opcheck_C.gptq_marlin_fp8_dequantize(
        packed_qweight,
        scales,
        size_k,
        size_n,
        group_size,
        True,
    )

    # 将参考权重转成与算子输出一致的 dtype，避免 dtype 差异干扰比较。
    dense_ref_cast = dense_ref.to(dense_from_op.dtype)

    # 校验正向布局解量化结果与参考实现逐元素一致。
    torch.testing.assert_close(dense_from_op, dense_ref_cast, atol=0, rtol=0)

    # 校验转置布局解量化结果与参考实现转置后逐元素一致。
    torch.testing.assert_close(
        dense_t_from_op,
        dense_ref_cast.transpose(0, 1),
        atol=0,
        rtol=0,
    )

    # 基于原始 GPTQ 权重构造验证模块。
    module = GPTQMarlinFP8Linear.from_raw_gptq(
        packed_qweight,
        scales,
        size_k=size_k,
        size_n=size_n,
        group_size=group_size,
    )

    # 构造输入激活 `x: [B, K]`，并开启输入梯度跟踪。
    x = (
        torch.randn((batch, size_k), device=device, dtype=torch.float16) * 0.5
    ).requires_grad_(True)

    # 执行验证模块前向，得到 `output: [B, N]`。
    output = module(x)

    # 用参考浮点权重计算基准输出 `output_ref: [B, N]`。
    output_ref = x.detach().to(torch.float32) @ dense_ref

    # 校验验证模块前向结果与参考 GEMM 在容差范围内一致。
    torch.testing.assert_close(output.float(), output_ref, atol=2e-1, rtol=2e-1)

    # 构造上游梯度 `grad_output: [B, N]`。
    grad_output = torch.randn_like(output) * 0.5

    # 执行反向传播，触发验证专用 `dInput` 反向算子。
    output.backward(grad_output)

    # 用参考浮点权重计算输入梯度基准 `grad_input_ref: [B, K]`。
    grad_input_ref = grad_output.to(torch.float32) @ dense_ref.transpose(0, 1)

    # 校验验证算子回传的 `x.grad: [B, K]` 与参考结果在容差范围内一致。
    torch.testing.assert_close(
        x.grad.float(),
        grad_input_ref,
        atol=2e-1,
        rtol=2e-1,
    )
