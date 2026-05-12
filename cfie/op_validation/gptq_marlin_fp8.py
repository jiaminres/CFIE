from __future__ import annotations

"""
GPTQ-Marlin 与 FP8 激活验证封装。

该模块只服务于算子验证，不替代正式推理或训练路径。
它负责把原始 GPTQ int4 权重整理成 Marlin 可消费布局，并通过一个最小
的 `torch.nn.Module` 包装：

1. 复用现有 Marlin 前向算子完成 `int4 weight * FP8 activation` 计算。
2. 调用验证专用自定义算子计算输入梯度 `dInput`。
3. 明确将权重、scale、bias 视为冻结常量，不回传这些参数的梯度。
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from cfie import _custom_ops as ops
from cfie.model_executor.layers.quantization.utils.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    marlin_make_workspace_new,
    marlin_permute_bias,
    marlin_permute_scales,
    should_use_atomic_add_reduce,
)
from cfie.scalar_type import scalar_types

# ------------------------------- 搜索验证动态库的候选路径模式 -------------------------------

# 在 `.tmp` 构建目录中查找 Windows 动态库产物。
_LIBRARY_CANDIDATE_PATTERNS = (
    ".tmp/**/op_validation/opcheck_C*.pyd",
    ".tmp/**/op_validation/opcheck_C*.so",
    "build/**/op_validation/opcheck_C*.pyd",
    "build/**/op_validation/opcheck_C*.so",
)


# ------------------------------- 解析仓库根目录 -------------------------------

def _repo_root() -> Path:
    # 当前文件位于 `cfie/op_validation`，向上两级即仓库根目录。
    return Path(__file__).resolve().parents[2]


# ------------------------------- 解析单个库路径候选 -------------------------------

def _resolve_candidate_path(value: str) -> Path | None:
    # 先把字符串解释为文件系统路径对象。
    path = Path(value)

    # 若用户直接给到一个动态库文件，则直接返回其绝对路径。
    if path.is_file():
        return path.resolve()

    # 若用户给到的是目录，则在目录内查找最新的 `opcheck_C` 动态库。
    if path.is_dir():
        # 收集目录内的 `.pyd` 与 `.so` 候选文件。
        matches = sorted(
            list(path.glob("opcheck_C*.pyd")) + list(path.glob("opcheck_C*.so")),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )

        # 若目录下存在候选文件，则返回时间戳最新的一个。
        if matches:
            return matches[0].resolve()

    # 文件与目录两种路径解析都失败时，返回空值表示未命中。
    return None


# ------------------------------- 自动查找验证动态库 -------------------------------

def _find_opcheck_library() -> Path:
    # 优先尝试用户显式提供的动态库路径或输出目录路径。
    env_candidates = [
        os.environ.get("CFIE_OPCHECK_LIBRARY"),
        os.environ.get("CFIE_OP_VALIDATION_OUTPUT_DIR"),
    ]

    # 依次解析每个环境变量候选项。
    for candidate in env_candidates:
        # 跳过未设置的环境变量。
        if not candidate:
            continue

        # 尝试把当前环境变量解析成一个真实存在的动态库文件。
        resolved = _resolve_candidate_path(candidate)

        # 一旦成功解析，就立即返回该结果。
        if resolved is not None:
            return resolved

    # 准备收集仓库内默认构建目录中的所有候选动态库。
    matches: list[Path] = []

    # 获取仓库根目录，作为 glob 搜索的起点。
    root = _repo_root()

    # 按预设模式把所有候选动态库加入列表。
    for pattern in _LIBRARY_CANDIDATE_PATTERNS:
        matches.extend(root.glob(pattern))

    # 若整个仓库都未找到候选动态库，则直接报错提示用户构建验证目标。
    if not matches:
        raise FileNotFoundError(
            "Could not find opcheck_C validation library. "
            "Set CFIE_OPCHECK_LIBRARY or build with "
            "CFIE_BUILD_OP_VALIDATION_TARGETS=ON."
        )

    # 默认返回时间戳最新的动态库，降低多构建目录并存时的手工指定成本。
    return max(matches, key=lambda item: item.stat().st_mtime).resolve()


# ------------------------------- 加载验证动态库 -------------------------------

def load_opcheck_library(path: str | Path | None = None) -> Path:
    # 若 `torch.ops.opcheck_C.gptq_marlin_fp8_bwd_input` 已经注册且本次未显式
    # 指定其他库路径，则直接复用当前已加载库，避免重复 `load_library`。
    if hasattr(torch.ops, "opcheck_C") and hasattr(
        torch.ops.opcheck_C, "gptq_marlin_fp8_bwd_input"
    ):
        # 在未指定新路径时，直接返回一个占位路径说明“已加载”状态。
        if path is None:
            return Path("<already-loaded>")

    # 显式指定路径时直接解析该路径，否则自动搜索最近构建的验证动态库。
    library_path = Path(path).resolve() if path is not None else _find_opcheck_library()

    # 将验证动态库注册到当前 PyTorch 进程，供后续 `torch.ops.opcheck_C.*` 调用。
    torch.ops.load_library(str(library_path))

    # 返回实际加载的动态库绝对路径，便于上层记录与调试。
    return library_path


# ------------------------------- 构造空索引张量 -------------------------------

def _empty_index_tensor(device: torch.device) -> torch.Tensor:
    # 当前验证路径关闭 act-order，因此用空整型张量占位 `g_idx` 相关参数。
    return torch.empty(0, dtype=torch.int, device=device)


# ------------------------------- 保存预处理后的验证权重包 -------------------------------

@dataclass(slots=True)
class _PreparedValidationWeight:
    # Marlin 前向直接消费的 packed int4 权重。
    marlin_qweight: torch.Tensor

    # 已按 Marlin 布局重排的 group scales。
    marlin_scales: torch.Tensor

    # 反向 dInput 路径使用的 row-major group scales: [num_groups, N]。
    marlin_scales_bwd: torch.Tensor

    # Marlin kernel 所需的临时 workspace。
    workspace: torch.Tensor

    # 已按 Marlin 布局重排的 bias；若无 bias，则为 `None`。
    bias: torch.Tensor | None

    # act-order 关闭时的空 `g_idx` 占位张量。
    g_idx: torch.Tensor

    # act-order 关闭时的空 `g_idx_sort_indices` 占位张量。
    g_idx_sort_indices: torch.Tensor


# ------------------------------- 验证专用 Autograd 包装 -------------------------------

class _GPTQMarlinFP8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        marlin_qweight: torch.Tensor,
        marlin_scales: torch.Tensor,
        marlin_scales_bwd: torch.Tensor,
        workspace: torch.Tensor,
        bias: torch.Tensor | None,
        g_idx: torch.Tensor,
        g_idx_sort_indices: torch.Tensor,
        size_k: int,
        size_n: int,
        group_size: int,
        use_fp32_reduce: bool,
    ) -> torch.Tensor:
        # ------------------------------- 展平输入并量化为 FP8 激活 -------------------------------

        # 将输入 `x` 从 `[..., K]` 展平成二维矩阵 `x_2d: [M, K]` 以对齐 Marlin GEMM。
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()

        # 对 `x_2d: [M, K]` 做动态逐 token FP8 量化，得到：
        # - `x_q: [M, K]`，FP8 激活
        # - `a_scales: [M, 1]` 或等价逐 token scale 结构
        x_q, a_scales = ops.scaled_fp8_quant(
            x_2d,
            None,
            use_per_token_if_dynamic=True,
        )

        # ------------------------------- 调用 Marlin 前向 GEMM -------------------------------

        # 调用现有 Marlin 前向路径，计算：
        # - 输入激活：`x_q: [M, K]`
        # - 权重：`marlin_qweight`，其逻辑矩阵为 `[K, N]`
        # - 输出：`output: [M, N]`
        output = ops.marlin_gemm(
            x_q,
            None,
            marlin_qweight,
            bias,
            marlin_scales,
            a_scales,
            None,
            None,
            g_idx,
            g_idx_sort_indices,
            workspace,
            scalar_types.uint4b8,
            size_m=x_q.shape[0],
            size_n=size_n,
            size_k=size_k,
            is_k_full=True,
            use_atomic_add=should_use_atomic_add_reduce(
                m=x_2d.size(0),
                n=size_n,
                k=size_k,
                device=x.device,
                dtype=x.dtype,
            ),
            use_fp32_reduce=use_fp32_reduce,
            is_zp_float=False,
        )

        # ------------------------------- 保存反向所需状态 -------------------------------

        # 反向只复用同一份 packed qweight，并改用反向专用 scale 布局恢复 dInput。
        ctx.save_for_backward(marlin_qweight, marlin_scales_bwd)

        # 记录原始输入形状，以便把二维 `dInput: [M, K]` 恢复回 `[..., K]`。
        ctx.input_shape = x.shape

        # 记录原始输入 dtype，以便反向输出在最终返回前转回同一 dtype。
        ctx.input_dtype = x.dtype

        # 记录 group 大小，供反向算子决定如何解释 group scales。
        ctx.group_size = group_size

        # 将 `output: [M, N]` 恢复成 `[..., N]` 形状返回给上层调用者。
        return output.reshape(*x.shape[:-1], size_n)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # ------------------------------- 取回反向依赖的冻结权重状态 -------------------------------

        # 取回前向保存的 packed qweight 与反向专用 scales。
        marlin_qweight, marlin_scales_bwd = ctx.saved_tensors

        # ------------------------------- 展平上游梯度并量化为 FP8 -------------------------------

        # 将上游梯度 `grad_output` 从 `[..., N]` 展平成 `grad_output_2d: [M, N]`。
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        # 对 `grad_output_2d: [M, N]` 做动态逐 token FP8 量化，得到：
        # - `grad_output_fp8: [M, N]`
        # - `grad_output_scales: [M, 1]` 或等价逐 token scale 结构
        grad_output_fp8, grad_output_scales = ops.scaled_fp8_quant(
            grad_output_2d,
            None,
            use_per_token_if_dynamic=True,
        )

        # ------------------------------- 调用验证专用反向算子 -------------------------------

        # 计算 `grad_input: [M, K] = grad_output: [M, N] * weight^T: [N, K]`。
        grad_input = torch.ops.opcheck_C.gptq_marlin_fp8_bwd_input(
            grad_output_fp8,
            grad_output_scales,
            marlin_qweight,
            marlin_scales_bwd,
            ctx.input_shape[-1],
            grad_output_2d.shape[-1],
            ctx.group_size,
        )

        # ------------------------------- 恢复输入梯度形状并返回 -------------------------------

        # 将 `grad_input: [M, K]` 恢复成原始输入形状 `[..., K]`，并转回输入 dtype。
        grad_input = grad_input.reshape(ctx.input_shape).to(ctx.input_dtype)

        # 当前验证路径只回传输入 `x` 的梯度，其余参数统一视为冻结常量。
        return (
            grad_input,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ------------------------------- 验证专用线性层模块 -------------------------------

class GPTQMarlinFP8Linear(torch.nn.Module):
    def __init__(
        self,
        prepared: _PreparedValidationWeight,
        *,
        size_k: int,
        size_n: int,
        group_size: int,
        use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
    ) -> None:
        # ------------------------------- 记录逻辑矩阵元数据 -------------------------------

        # 先完成父类初始化，确保后续 buffer 注册生效。
        super().__init__()

        # 记录逻辑输入维 K，供前向与反向算子解释权重矩阵尺寸。
        self.size_k = size_k

        # 记录逻辑输出维 N，供前向 reshape 与 kernel 参数构造使用。
        self.size_n = size_n

        # 记录 GPTQ group 大小，供反向算子解释 scales 布局。
        self.group_size = group_size

        # 记录是否启用 FP32 reduce，以保持与 Marlin 前向数值策略一致。
        self.use_fp32_reduce = use_fp32_reduce

        # ------------------------------- 注册冻结权重相关 buffer -------------------------------

        # 注册 Marlin packed int4 权重，验证期间其值固定不参与梯度更新。
        self.register_buffer("marlin_qweight", prepared.marlin_qweight)

        # 注册已重排的 Marlin scales，验证期间其值固定不参与梯度更新。
        self.register_buffer("marlin_scales", prepared.marlin_scales)

        # 注册反向 dInput 专用 scales，保持 qweight 单份但允许 scale 使用反向布局。
        self.register_buffer("marlin_scales_bwd", prepared.marlin_scales_bwd)

        # 注册 kernel workspace，但不把它写入 state_dict。
        self.register_buffer("workspace", prepared.workspace, persistent=False)

        # 注册 act-order 关闭时的空 `g_idx` 占位张量，但不写入 state_dict。
        self.register_buffer("g_idx", prepared.g_idx, persistent=False)

        # 注册 act-order 关闭时的空 `g_idx_sort_indices` 占位张量，但不写入 state_dict。
        self.register_buffer(
            "g_idx_sort_indices",
            prepared.g_idx_sort_indices,
            persistent=False,
        )

        # 若当前权重不含 bias，则直接记录空值。
        if prepared.bias is None:
            self.bias = None
        else:
            # 若存在 bias，则把已重排 bias 注册为冻结 buffer。
            self.register_buffer("bias", prepared.bias)

    @classmethod
    def from_raw_gptq(
        cls,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        *,
        size_k: int,
        size_n: int,
        group_size: int,
        bias: torch.Tensor | None = None,
        use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
        library_path: str | Path | None = None,
    ) -> "GPTQMarlinFP8Linear":
        # ------------------------------- 确保验证动态库已加载 -------------------------------

        # 先加载验证动态库，保证 `opcheck_C.gptq_marlin_fp8_bwd_input` 已注册。
        load_opcheck_library(library_path)

        # ------------------------------- 校验输入张量与逻辑尺寸 -------------------------------

        # Marlin 验证路径要求原始 `qweight` 已位于 CUDA 设备上。
        if qweight.device.type != "cuda":
            raise ValueError("qweight must live on CUDA for Marlin validation")

        # `scales` 必须与 `qweight` 处于同一 CUDA 设备，避免后续重排时跨设备拷贝。
        if scales.device != qweight.device:
            raise ValueError("scales must be on the same device as qweight")

        # 原始 GPTQ int4 权重采用 `torch.int32` 作为 packed 存储类型。
        if qweight.dtype != torch.int32:
            raise ValueError("qweight must use torch.int32 storage")

        # group 模式下，逻辑输入维 K 必须能被 `group_size` 整除。
        if group_size != -1 and size_k % group_size != 0:
            raise ValueError("size_k must be divisible by group_size")

        # 计算期望的 group 数：全通道时为 1，否则为 `K / group_size`。
        expected_groups = 1 if group_size == -1 else size_k // group_size

        # 原始 GPTQ int4 打包权重的逻辑形状应为 `[K / 8, N]`。
        if qweight.shape != (size_k // 8, size_n):
            raise ValueError("qweight must have shape [size_k / 8, size_n]")

        # 原始 group scales 的逻辑形状应为 `[num_groups, N]`。
        if scales.shape != (expected_groups, size_n):
            raise ValueError("scales must have shape [num_groups, size_n]")

        # 若存在 bias，则它的逻辑形状应为 `[N]`。
        if bias is not None and bias.shape != (size_n,):
            raise ValueError("bias must have shape [size_n]")

        # ------------------------------- 构造 act-order 关闭时的空索引 -------------------------------

        # 当前验证路径不启用 act-order，因此直接复用空索引张量占位。
        empty = _empty_index_tensor(qweight.device)

        # ------------------------------- 预处理并重排原始 GPTQ 权重 -------------------------------

        # 先复制一份连续权重，避免原地预处理污染调用者传入的原始 `qweight`。
        marlin_qweight = qweight.contiguous().clone()

        # 对原始 packed int4 权重执行 Marlin FP8 激活路径所需的预处理。
        ops.marlin_int4_fp8_preprocess(marlin_qweight, None, True)

        # 将预处理后的权重 repack 成 Marlin kernel 直接消费的 packed 布局。
        marlin_qweight = ops.gptq_marlin_repack(
            marlin_qweight,
            empty,
            size_k,
            size_n,
            4,
            True,
        )

        # ------------------------------- 重排 scales 并对齐数值约定 -------------------------------

        # 将原始 `scales: [num_groups, N]` 重排为 Marlin 所需布局，并保持连续内存。
        marlin_scales = (
            marlin_permute_scales(
                scales.contiguous(),
                size_k,
                size_n,
                group_size,
                True,
            ).contiguous()
            # 按当前 Marlin FP8 激活路径的数值约定放大 scale。
            * 512
        )

        # 反向 dInput 的 reduction 维是 N，因此保留 row-major [num_groups, N]
        # 布局，让反向 kernel 沿 N 维连续读取 scale。
        marlin_scales_bwd = scales.contiguous()

        # ------------------------------- 打包验证所需的冻结权重状态 -------------------------------

        # 将所有前向/反向复用的中间结果打包成一个冻结权重结构体。
        prepared = _PreparedValidationWeight(
            marlin_qweight=marlin_qweight,
            marlin_scales=marlin_scales,
            marlin_scales_bwd=marlin_scales_bwd,
            workspace=marlin_make_workspace_new(qweight.device),
            bias=marlin_permute_bias(bias.contiguous()) if bias is not None else None,
            g_idx=empty,
            g_idx_sort_indices=empty,
        )

        # ------------------------------- 构造验证模块实例并返回 -------------------------------

        # 基于已准备好的冻结权重包构造一个可直接调用的验证模块。
        return cls(
            prepared,
            size_k=size_k,
            size_n=size_n,
            group_size=group_size,
            use_fp32_reduce=use_fp32_reduce,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 直接把输入 `x: [..., K]` 委托给验证专用 autograd.Function 执行。
        return _GPTQMarlinFP8LinearFn.apply(
            x,
            self.marlin_qweight,
            self.marlin_scales,
            self.marlin_scales_bwd,
            self.workspace,
            self.bias,
            self.g_idx,
            self.g_idx_sort_indices,
            self.size_k,
            self.size_n,
            self.group_size,
            self.use_fp32_reduce,
        )
