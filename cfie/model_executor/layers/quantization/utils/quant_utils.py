# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""量化测试与基准辅助工具。"""

# 导入可调用对象与只读映射类型注解。
from collections.abc import Callable, Mapping
# 导入轻量数据结构装饰器。
from dataclasses import dataclass
# 导入只读映射包装器。
from types import MappingProxyType
# 导入类型检查、静态类变量与具名元组支持。
from typing import TYPE_CHECKING, ClassVar, NamedTuple

# 导入 numpy 以便在 pack/unpack 辅助函数里做按位处理。
import numpy
# 导入 PyTorch 主库。
import torch
# 导入 FX dtype 缩写表。
from torch import fx

# 导入当前运行平台抽象。
from cfie.platforms import current_platform
# 导入量化标量类型描述。
from cfie.scalar_type import ScalarType, scalar_types

if TYPE_CHECKING:
    # 仅在类型检查阶段导入线性层基类，避免运行时循环依赖。
    from cfie.model_executor.layers.linear import LinearBase

# 记录当前平台默认使用的 FP8 dtype。
FP8_DTYPE = current_platform.fp8_dtype()
# 记录 FP4 packed 存储在 PyTorch 侧使用的承载 dtype。
FP4_DTYPE = torch.uint8
# 记录 MXFP scale 的存储 dtype。
MXFP_SCALE_DTYPE = torch.uint8


def get_fp8_min_max() -> tuple[float, float]:
    """返回当前平台 FP8 量化所使用的最小值与最大值。"""
    # FNUZ 平台不能直接使用 PyTorch 默认上限，否则动态量化精度会退化。
    if current_platform.is_fp8_fnuz():
        # 对 ROCm FNUZ 路径固定使用 `[-224, 224]` 作为裁剪区间。
        return -224.0, 224.0
    # 读取当前平台 FP8 dtype 的数值范围。
    finfo = torch.finfo(current_platform.fp8_dtype())
    # 返回当前 FP8 dtype 的最小值与最大值。
    return finfo.min, finfo.max


# 用基础具名元组承载 `GroupShape(row, col)` 的底层存储。
class _GroupShape(NamedTuple):
    # 记录 group 在行方向的覆盖范围。
    row: int
    # 记录 group 在列方向的覆盖范围。
    col: int


class GroupShape(_GroupShape):
    """描述量化 group 在矩阵上的覆盖形状。"""

    # 声明常用的整张量量化 group 形状别名。
    PER_TENSOR: ClassVar["GroupShape"]
    # 声明常用的逐 token 量化 group 形状别名。
    PER_TOKEN: ClassVar["GroupShape"]
    # 声明常用的逐通道量化 group 形状别名。
    PER_CHANNEL: ClassVar["GroupShape"]

    def is_per_tensor(self) -> bool:
        # `(-1, -1)` 表示覆盖整张矩阵。
        return self.row == -1 and self.col == -1

    def is_per_token(self) -> bool:
        # `(1, -1)` 表示每一行共享一个 scale。
        return self.row == 1 and self.col == -1

    def is_per_channel(self) -> bool:
        # `(-1, 1)` 表示每一列共享一个 scale。
        return self.row == -1 and self.col == 1

    def is_per_group(self) -> bool:
        # `(1, col>=1)` 表示逐行、按列分组量化。
        return self.row == 1 and self.col >= 1


# 初始化整张量量化别名 `(-1, -1)`。
GroupShape.PER_TENSOR = GroupShape(-1, -1)
# 初始化逐 token 量化别名 `(1, -1)`。
GroupShape.PER_TOKEN = GroupShape(1, -1)
# 初始化逐通道量化别名 `(-1, 1)`。
GroupShape.PER_CHANNEL = GroupShape(-1, 1)


@dataclass(frozen=True)
class ScaleDesc:
    """描述单层量化 scale 的 dtype、静动态属性与 group 形状。"""

    # 记录 scale 张量自身的数据类型。
    dtype: torch.dtype
    # 记录当前 scale 是否为静态常量。
    static: bool
    # 记录当前 scale 覆盖的 group 形状。
    group_shape: GroupShape

    def __str__(self):
        # 为常见 group 形状准备更易读的名称映射。
        d = {
            GroupShape.PER_TENSOR: "per_tensor",
            GroupShape.PER_TOKEN: "per_token",
            GroupShape.PER_CHANNEL: "per_channel",
        }
        # 若命中常见形状则返回别名，否则退回默认字符串形式。
        group_shape = d.get(self.group_shape, str(self.group_shape))
        # 拼出形如 `f32,static,per_tensor` 的描述字符串。
        return (
            f"{fx.graph.dtype_abbrs[self.dtype]},"
            f"{'static' if self.static else 'dynamic'},{group_shape}"
        )


@dataclass(frozen=True)
class QuantKey:
    """描述一种量化配置的量化 dtype、scale 结构与对称性。"""

    # 记录量化后的目标 dtype。
    dtype: torch.dtype
    # 记录主 scale 描述。
    scale: ScaleDesc
    # 记录可选的二级 scale 描述。
    scale2: ScaleDesc | None = None
    # 记录当前量化是否为对称量化。
    symmetric: bool = True

    def __str__(self):
        # 若存在二级 scale，则先拼出二级 scale 字符串。
        scale2_str = f"scale2({self.scale2})," if self.scale2 else ""
        # 拼出完整量化键字符串。
        return (
            f"QuantKey({fx.graph.dtype_abbrs[self.dtype]},"
            f"scale({self.scale}),{scale2_str}"
            f"{'a' if not self.symmetric else ''}symmetric)"
        )


# 定义 FP8 静态整张量量化的 scale 描述。
kStaticTensorScale = ScaleDesc(torch.float32, True, GroupShape.PER_TENSOR)
# 定义 FP8 静态整张量对称量化键。
kFp8StaticTensorSym = QuantKey(FP8_DTYPE, kStaticTensorScale, symmetric=True)

# 定义 FP8 动态整张量量化的 scale 描述。
kDynamicTensorScale = ScaleDesc(torch.float32, False, GroupShape.PER_TENSOR)
# 定义 FP8 动态整张量对称量化键。
kFp8DynamicTensorSym = QuantKey(FP8_DTYPE, kDynamicTensorScale, symmetric=True)

# 定义 FP8 静态逐 token 量化的 scale 描述。
kStaticTokenScale = ScaleDesc(torch.float32, True, GroupShape.PER_TOKEN)
# 定义 FP8 静态逐 token 对称量化键。
kFp8StaticTokenSym = QuantKey(FP8_DTYPE, kStaticTokenScale, symmetric=True)

# 定义 FP8 静态逐通道量化的 scale 描述。
kStaticChannelScale = ScaleDesc(torch.float32, True, GroupShape.PER_CHANNEL)
# 定义 FP8 静态逐通道对称量化键。
kFp8StaticChannelSym = QuantKey(FP8_DTYPE, kStaticChannelScale, symmetric=True)

# 定义 FP8 动态逐 token 量化的 scale 描述。
kDynamicTokenScale = ScaleDesc(torch.float32, False, GroupShape.PER_TOKEN)
# 定义 FP8 动态逐 token 对称量化键。
kFp8DynamicTokenSym = QuantKey(FP8_DTYPE, kDynamicTokenScale, symmetric=True)

# 定义 NVFP4 动态 group 量化的一阶 scale 描述。
kNvfp4DynamicGroupScale = ScaleDesc(FP8_DTYPE, False, GroupShape(1, 16))
# 定义 NVFP4 动态量化键，带二级静态整张量 scale。
kNvfp4Dynamic = QuantKey(
    FP4_DTYPE, scale=kNvfp4DynamicGroupScale, scale2=kStaticTensorScale
)

# 定义 NVFP4 静态 group 量化的一阶 scale 描述。
kNvfp4StaticGroupScale = ScaleDesc(FP8_DTYPE, True, GroupShape(1, 16))
# 定义 NVFP4 静态量化键，带二级静态整张量 scale。
kNvfp4Static = QuantKey(
    FP4_DTYPE, scale=kNvfp4StaticGroupScale, scale2=kStaticTensorScale
)

# 定义 FP8 动态 `1x128` 量化的 scale 描述。
kDynamic128Scale = ScaleDesc(torch.float32, False, GroupShape(1, 128))
# 定义 FP8 动态 `1x128` 对称量化键。
kFp8Dynamic128Sym = QuantKey(FP8_DTYPE, kDynamic128Scale, symmetric=True)

# 定义 FP8 静态 `128x128` block 量化的 scale 描述。
kStatic128BlockScale = ScaleDesc(torch.float32, True, GroupShape(128, 128))
# 定义 FP8 静态 `128x128` block 对称量化键。
kFp8Static128BlockSym = QuantKey(FP8_DTYPE, kStatic128BlockScale, symmetric=True)

# 定义 FP8 动态 `1x64` 量化的 scale 描述。
kDynamic64Scale = ScaleDesc(torch.float32, False, GroupShape(1, 64))
# 定义 FP8 动态 `1x64` 对称量化键。
kFp8Dynamic64Sym = QuantKey(FP8_DTYPE, kDynamic64Scale, symmetric=True)

# 当前仍沿用 `torch.dtype` 描述 MXFP scale，后续再统一成 scale_dtype。
kMxfp4DynamicGroupScale = ScaleDesc(MXFP_SCALE_DTYPE, False, GroupShape(1, 32))
# 定义 MXFP4 动态量化键。
kMxfp4Dynamic = QuantKey(FP4_DTYPE, scale=kMxfp4DynamicGroupScale, symmetric=True)

# 定义 MXFP8 动态 group 量化的 scale 描述。
kMxfp8DynamicGroupScale = ScaleDesc(MXFP_SCALE_DTYPE, False, GroupShape(1, 32))
# 定义 MXFP8 动态量化键。
kMxfp8Dynamic = QuantKey(FP8_DTYPE, scale=kMxfp8DynamicGroupScale, symmetric=True)

# 定义 MXFP4 静态 group 量化的 scale 描述。
kMxfp4StaticGroupScale = ScaleDesc(MXFP_SCALE_DTYPE, True, GroupShape(1, 32))
# 定义 MXFP4 静态量化键。
kMxfp4Static = QuantKey(FP4_DTYPE, scale=kMxfp4StaticGroupScale, symmetric=True)

def _normalize_quant_group_shape(x: torch.Tensor, group_shape: GroupShape):
    # 把 `-1` 的占位维度展开成 `x: [..., M, N]` 的完整尺寸。
    return (
        group_shape[0] if group_shape[0] > 0 else x.shape[-2],
        group_shape[1] if group_shape[1] > 0 else x.shape[-1],
    )


def group_broadcast(t, shape):
    # 逐维把 `t` 扩展到目标形状 `shape`。
    for i, s in enumerate(shape):
        # 若 `t` 在当前维不存在，则按广播规则把它视为尺寸 1。
        t_dim_size = t.shape[i] if i < t.ndim else 1
        # 仅在当前维既不匹配目标尺寸、也不是广播维时做显式重复。
        if t_dim_size != s and t_dim_size != 1:
            # 目标尺寸必须能整除当前尺寸，才能按 group 重复展开。
            assert s % t_dim_size == 0
            # 在当前维后插入重复维，再显式展开并压平成目标维。
            t = (
                t.unsqueeze(i + 1)
                .expand(*t.shape[: i + 1], s // t_dim_size, *t.shape[i + 1 :])
                .flatten(i, i + 1)
            )
    # 返回扩展后的张量。
    return t


def prep_scale_for_group_broadcast(
    scale: torch.Tensor,
    x: torch.Tensor,
    group_shape: GroupShape | None,
) -> torch.Tensor:
    """把输入 scale 调整成适合广播到 `x` 的形状。"""
    if scale.numel() == 1:
        # 整张量量化时保留 0 维标量，避免被后续逻辑误判成通道量化。
        return (
            scale
            if group_shape is not None and group_shape.is_per_tensor()
            else scale.reshape(1, 1)
        )
    if scale.ndim == 1:
        # 一维 scale 无法自行推断广播方向，因此必须显式提供 group_shape。
        assert group_shape is not None, (
            "group_shape must be provided to correctly broadcast 1D scale"
        )
        # 将 `-1` 归一化成 `x: [..., M, N]` 的真实尺寸。
        rows, cols = _normalize_quant_group_shape(x, group_shape)
        # 若 group 行覆盖整个输入行维，则把 scale 变成 `[1, N_groups]`。
        if rows == x.shape[-2]:
            scale = scale.unsqueeze(-2)
        # 若 group 列覆盖整个输入列维，则把 scale 变成 `[M_groups, 1]`。
        elif cols == x.shape[-1]:
            scale = scale.unsqueeze(-1)
        else:
            # 其余情况说明当前一维 scale 无法和目标张量对齐。
            raise ValueError(
                f"1D scale with shape {scale.shape} cannot be broadcast to x with shape"
                f" {x.shape}, group_shape={(rows, cols)}"
            )
    # 返回已整理好的 scale 张量。
    return scale


# 当前量化 helper 约定一个 scale 覆盖一个 `group_shape` 指定的元素块。
# 常见 group 形状包括：
# - `(-1, -1)`：整张量量化
# - `(1, -1)`：逐行量化
# - `(-1, 1)`：逐列量化
# - `(128, 128)`：`128x128` block 量化
# - `(1, 128)`：逐 token、按 128 列分组量化
def scaled_quantize(
    x: torch.Tensor,
    group_shape: GroupShape,
    quant_dtype: torch.dtype,
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按给定 group 形状把二维张量量化到浮点低精度格式。"""
    # 把 `group_shape` 中的 `-1` 归一化成 `x: [M, N]` 的真实尺寸。
    group_shape = _normalize_quant_group_shape(x, group_shape)
    # 当前实现仅支持目标量化 dtype 为浮点低精度类型。
    assert quant_dtype.is_floating_point, (
        "currently `scaled_quantize` only supports floating point dtypes "
        "but could be extended to support other dtypes"
    )

    # 读取目标量化 dtype 的最小值与最大值。
    finfo = torch.finfo(quant_dtype)

    # 若指定了计算 dtype，则先把输入 `x: [M, N]` 转到该 dtype。
    x_compute = x if compute_dtype is None else x.to(compute_dtype)

    # 当前实现约定输入必须是二维矩阵 `x: [M, N]`。
    assert x.ndim == 2
    # 输入矩阵的行列数必须能被 group 形状整除。
    assert x.shape[0] % group_shape[0] == 0 and x.shape[1] % group_shape[1] == 0
    # 计算行方向与列方向分别会被切成多少个 block。
    blk_m, blk_n = x.shape[0] // group_shape[0], x.shape[1] // group_shape[1]
    # 把 `x: [M, N]` 重排成 `x_blkd: [blk_m, group_m, blk_n, group_n]`。
    x_blkd = x_compute.reshape(blk_m, group_shape[0], blk_n, group_shape[1])

    # 把 group 维挪到最后，得到 `x_blkd_permd: [blk_m, blk_n, group_m, group_n]`。
    x_blkd_permd = x_blkd.permute(0, 2, 1, 3)
    # 将每个 block 拉平成 `x_blkd_permd: [blk_m, blk_n, group_m * group_n]`。
    x_blkd_permd = x_blkd_permd.flatten(start_dim=2)

    # 在每个 block 上同时求最小值和最大值。
    min_val, max_val = x_blkd_permd.aminmax(dim=-1)
    # 取 block 内绝对值最大项，并对极小值做下限截断。
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    # 读取当前平台 FP8 的正向最大值。
    _, fp8_max = get_fp8_min_max()
    # 计算每个 block 的量化 scale `scale: [blk_m, blk_n]`。
    scale = fp8_max / amax

    # 对 block 内元素乘 scale、裁剪到量化范围，再还原回 `x: [M, N]` 布局。
    x_scl_sat = (
        (x_blkd_permd * scale.unsqueeze(-1))
        .clamp(min=finfo.min, max=finfo.max)
        .reshape(blk_m, blk_n, group_shape[0], group_shape[1])
        .permute(0, 2, 1, 3)
        .reshape(x.shape)
    )

    # 返回量化后的张量 `x_q: [M, N]` 与反向恢复用的逆 scale。
    return x_scl_sat.to(quant_dtype).contiguous(), scale.float().reciprocal()


def scaled_dequantize(
    x_q: torch.Tensor,
    x_s: torch.Tensor,
    group_shape: GroupShape | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    # 先把 `x_s` 调整成适合广播到 `x_q` 的形状。
    x_s = prep_scale_for_group_broadcast(x_s, x_q, group_shape)
    # 若显式给出了 group_shape，则校验 scale 网格和量化张量网格相容。
    if group_shape is not None:
        # 校验 scale 的列维与 `x_q: [..., M, N]` 的 group 列数一致。
        assert x_s.shape[-1] == x_q.shape[-1] // group_shape[1]
        # 校验 scale 的行维与 `x_q: [..., M, N]` 的 group 行数一致。
        assert x_s.shape[-2] == x_q.shape[-2] // group_shape[0]
    # 把 scale 真正扩展成和 `x_q` 相同的形状。
    x_s = group_broadcast(x_s.to(torch.float32), x_q.shape)
    # 用浮点乘法恢复原始数值，并转到目标输出 dtype。
    return (x_q.to(torch.float32) * x_s).to(out_dtype)


def get_attribute_fallback(obj, attributes: list[str]):
    # 依次尝试读取候选属性名。
    for attr in attributes:
        # 一旦当前对象拥有该属性，就立即返回对应值。
        if hasattr(obj, attr):
            return getattr(obj, attr)
    # 若所有候选属性都不存在，则抛出带候选列表的错误。
    raise AttributeError(f"'{obj}' has no recognized attributes: {attributes}.")


def get_and_maybe_dequant_weights(
    layer: "LinearBase", out_dtype: torch.dtype = torch.float32
):
    """返回线性层的反量化权重，布局固定为 `[out, in]`。"""
    # 延迟导入线性层实现，避免模块初始化时的循环依赖。
    from cfie.model_executor.layers.linear import UnquantizedLinearMethod
    # 延迟导入 FP8 线性量化方法实现。
    from cfie.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    # 从常见权重属性名里取到当前层实际持有的权重张量。
    weight = get_attribute_fallback(layer, ["weight", "qweight", "weight_packed"])

    # 未量化层直接把基础权重转成目标 dtype 返回。
    if layer.quant_method is None or isinstance(
        layer.quant_method, UnquantizedLinearMethod
    ):
        return weight.to(out_dtype)

    # 简单 FP8 线性层可直接依赖保存的 weight scale 做反量化。
    if (
        isinstance(layer.quant_method, Fp8LinearMethod)
        and not layer.quant_method.use_marlin
        and not layer.quant_method.use_deep_gemm
    ):
        # 读取当前层保存的权重 scale 或其倒数版本。
        weight_scales = get_attribute_fallback(
            layer, ["weight_scale", "weight_scale_inv"]
        )
        # 用 `scaled_dequantize` 恢复浮点权重。
        dequant_weights = scaled_dequantize(
            weight,
            weight_scales,
            group_shape=layer.weight_block_size,
            out_dtype=out_dtype,
        )
        # 非 block quant 的 per-tensor 权重以 `[in, out]` 布局存储，需要转置。
        if not layer.quant_method.block_quant:
            dequant_weights = dequant_weights.T
        # 返回恢复后的 `[out, in]` 浮点权重。
        return dequant_weights

    # 兜底路径要求线性层暴露本地输入分片大小。
    assert hasattr(layer, "input_size_per_partition")
    # 构造单位矩阵 `eye: [in_local, in_local]` 作为“查询向量”。
    eye = torch.eye(
        layer.input_size_per_partition,
        dtype=out_dtype,
        device=weight.device,
    )
    # 通过量化层 `apply` 计算单位矩阵前向，间接恢复浮点权重。
    dequant_weights = layer.quant_method.apply(layer, eye, bias=None).to(out_dtype)
    # 将 `[in, out]` 结果转成统一约定的 `[out, in]` 布局返回。
    return dequant_weights.T


def pack_quantized_values_into_int32(
    w_q: torch.Tensor, wtype: ScalarType, packed_dim: int = 0
):
    # 先把待 pack 的维度移动到最后一维。
    perm = (*[i for i in range(len(w_q.shape)) if i != packed_dim], packed_dim)
    # 记录逆置换，便于最后恢复原始维度顺序。
    inv_perm = tuple(perm.index(i) for i in range(len(perm)))
    # 得到按目标维度排到最后的张量 `w_q_perm`。
    w_q_perm = w_q.permute(perm)

    # 计算一个 int32 能容纳多少个低比特元素。
    pack_factor = 32 // wtype.size_bits
    # 构造当前低比特元素的位掩码。
    mask = (1 << wtype.size_bits) - 1

    # 拷贝目标 shape 作为 packed 输出 shape 的基础。
    new_shape_perm = list(w_q_perm.shape)
    # 被 pack 的最后一维长度必须能整除 pack_factor。
    assert w_q_perm.shape[-1] % pack_factor == 0
    # 将最后一维缩小为原来的 `1 / pack_factor`。
    new_shape_perm[-1] //= pack_factor

    # 分配 packed 结果张量 `res`。
    res = torch.zeros(new_shape_perm, dtype=torch.int32, device=w_q.device)
    # 逐个槽位把原始低比特元素写入对应的 int32 bit 区间。
    for i in range(pack_factor):
        # 取出当前位置的低比特值、做掩码，并左移到目标 bit 槽位。
        res |= (w_q_perm[..., i::pack_factor] & mask) << wtype.size_bits * i

    # 恢复原始维度顺序后返回 packed 结果。
    return res.permute(inv_perm)


def unpack_quantized_values_into_int32(
    w_q: torch.Tensor, wtype: ScalarType, packed_dim: int = 0
):
    # 先把 packed 维度移动到最后一维。
    perm = (*[i for i in range(len(w_q.shape)) if i != packed_dim], packed_dim)
    # 记录逆置换，便于最后恢复原始维度顺序。
    inv_perm = tuple(perm.index(i) for i in range(len(perm)))
    # 得到按目标维度排到最后的张量 `w_q_perm`。
    w_q_perm = w_q.permute(perm)

    # 计算一个 int32 中打包了多少个低比特元素。
    pack_factor = 32 // wtype.size_bits
    # 构造当前低比特元素的位掩码。
    mask = (1 << wtype.size_bits) - 1

    # 拷贝 packed 输入 shape 作为解包 shape 的基础。
    new_shape_perm = list(w_q_perm.shape)
    # 将最后一维扩展回原来的 `pack_factor` 倍。
    new_shape_perm[-1] *= pack_factor

    # 分配解包后的 int32 结果张量 `res`。
    res = torch.zeros(new_shape_perm, dtype=torch.int32, device=w_q.device)
    # 逐个槽位从 int32 中解出对应低比特值。
    for i in range(pack_factor):
        # 提取位于第 `i` 个槽位的低比特元素。
        res[..., i::pack_factor] = (w_q_perm >> wtype.size_bits * i) & mask

    # 恢复原始维度顺序后返回解包结果。
    return res.permute(inv_perm)


def is_layer_skipped(
    prefix: str,
    ignored_layers: list[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
    *,
    skip_with_substr: bool = False,
) -> bool:
    def prefix_full_match(prefix: str, ignored_layers: list[str]) -> bool:
        # 仅当完整前缀命中忽略列表时才视为跳过。
        return prefix in ignored_layers

    def substr_match(prefix: str, ignored_layers: list[str]) -> bool:
        # 只要忽略项是当前前缀的子串，就视为跳过。
        return any(layer in prefix for layer in ignored_layers)

    # 根据匹配模式选择完整匹配或子串匹配函数。
    match_func = substr_match if skip_with_substr else prefix_full_match

    # 取当前前缀最后一段模块名，例如 `q_proj` 或 `gate_up_proj`。
    proj_name = prefix.split(".")[-1]

    # fused 层需要把融合名字拆回各 shard 名称逐一判断。
    if proj_name in fused_mapping:
        # 把当前 fused 前缀展开成各个 shard 的独立前缀。
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in fused_mapping[proj_name]
        ]

        # 初始化 fused 层的跳过结果占位。
        is_skipped = None
        # 逐个 shard 检查其是否命中忽略规则。
        for shard_prefix in shard_prefixes:
            # 计算当前 shard 的跳过结果。
            is_shard_skipped = match_func(shard_prefix, ignored_layers)

            # 第一个 shard 直接作为 fused 层的初始判定结果。
            if is_skipped is None:
                is_skipped = is_shard_skipped
            # 若后续 shard 判定和前面不一致，则说明 fused 层配置不一致。
            elif is_shard_skipped != is_skipped:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision."
                )
    # experts 路径在完整匹配模式下需要单独按 expert 前缀过滤。
    elif "experts" in prefix and not skip_with_substr:
        # 仅保留忽略列表里和 experts 相关的项。
        expert_ignore_layers = filter(
            lambda layer_name: "experts" in layer_name, ignored_layers
        )
        # 只要当前 expert 前缀命中任意 expert 忽略项，就返回跳过。
        return any(
            prefix in layer_name if not skip_with_substr else layer_name in prefix
            for layer_name in expert_ignore_layers
        )
    else:
        # 非 fused、非 experts 特例则直接按选定匹配函数判断。
        is_skipped = match_func(prefix, ignored_layers)

    # 前面的各条分支必须给出最终布尔结果。
    assert is_skipped is not None
    # 返回当前层是否应该跳过量化。
    return is_skipped


def get_pack_factor(num_bits):
    # 当前 bit 数必须能整除 32，才能完整 pack 进 int32。
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    # 返回一个 int32 中能容纳多少个该 bit 数元素。
    return 32 // num_bits


def permute_rows(
    q_w: torch.Tensor,
    w_ref: torch.Tensor,
    group_size: int,
    test_perm: torch.Tensor | None = None,
):
    # act-order 前后的量化权重与参考权重必须拥有相同 shape。
    assert q_w.shape == w_ref.shape

    # 保存原始设备，便于结果回迁。
    orig_device = q_w.device
    # 读取行数 `K`，供构造 group 索引使用。
    k_size, _ = q_w.shape

    # 分配每一行所属 group 的索引向量 `g_idx: [K]`。
    g_idx = torch.zeros((k_size,), dtype=torch.int32)
    # 按 `group_size` 为每一行写入所在的分组编号。
    for i in range(k_size):
        g_idx[i] = i // group_size

    # 若未给定测试置换，则随机生成一组 act-order 行置换。
    rand_perm = test_perm if test_perm is not None else torch.randperm(k_size)

    # 按置换重排行分组索引 `g_idx: [K]`。
    g_idx = g_idx[rand_perm].contiguous()
    # 按同一置换重排量化权重 `q_w: [K, N]`。
    q_w = q_w[rand_perm, :].contiguous()
    # 按同一置换重排参考权重 `w_ref: [K, N]`。
    w_ref = w_ref[rand_perm, :].contiguous()

    # 返回置换后的参考权重、量化权重、group 索引与置换表。
    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        rand_perm.to(device=orig_device),
    )


def quantize_weights(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int | None,
    zero_points: bool = False,
    ref_zero_points_after_scales: bool = False,
):
    # 当前实现主要验证整数型量化路径。
    assert quant_type.is_integer(), (
        "Floating point quantization may work but has not been tested"
    )
    # 使用 group zero points 时必须显式提供 group_size。
    assert not zero_points or group_size is not None, (
        "to have group zero points, group_size must be provided "
        "(-1 group_size is channelwise)"
    )

    # 保存原始设备，便于返回结果回迁。
    orig_device = w.device
    # 保存原始浮点 dtype，便于构造参考权重时还原。
    orig_type = w.dtype
    # 读取权重矩阵逻辑 shape `w: [K, N]`。
    size_k, size_n = w.shape

    # 量化入口要求原始权重必须是浮点张量。
    assert w.is_floating_point(), "w must be float"

    # `group_size = -1` 代表整列共用一个 group，即按整行维处理。
    if group_size == -1:
        group_size = size_k

    # group 量化时需要先把 `w: [K, N]` 重排成按 group 组织的视图。
    if group_size is not None and group_size < size_k:
        # 先变成 `[-1, group_size, N]`，把每组连续行切开。
        w = w.reshape((-1, group_size, size_n))
        # 再把 group 维放到最前，得到 `[group_size, num_groups, N]`。
        w = w.permute(1, 0, 2)
        # 最后拉平成 `[group_size, num_groups * N]`，便于逐列求 scale。
        w = w.reshape((group_size, -1))

    # 按列求每个 group 的最大值。
    max_val = torch.max(w, 0, keepdim=True).values
    # 按列求每个 group 的最小值。
    min_val = torch.min(w, 0, keepdim=True).values

    # 读取当前量化类型允许的最大整型值。
    max_q_val = quant_type.max()
    # 读取当前量化类型允许的最小整型值。
    min_q_val = quant_type.min()

    # 默认把 scale 初始化成无缩放的 1.0。
    w_s = torch.Tensor([1.0]).to(w.device)
    # 默认 zero point 为空，表示对称量化。
    maybe_w_zp = None
    # 仅在启用 group 量化时才真正计算 scale 与 zero point。
    if group_size is not None:
        # 非对称量化需要单独推导 zero point。
        if zero_points:
            # zero point 路径要求量化类型无符号且存在正区间。
            assert not quant_type.is_signed() and quant_type.max() > 0
            # 先按 `[max - min] / qmax` 计算 scale。
            w_s = (max_val - min_val).clamp(min=1e-5) / quant_type.max()
            # 再把最小值折算成 zero point，并裁剪到合法量化范围。
            maybe_w_zp = (
                torch.round(torch.abs(min_val / w_s)).clamp(min_q_val, max_q_val).int()
            )
        else:
            # 对称量化按正负绝对值中的较大者推导 scale。
            w_s = torch.max(
                abs(max_val / (max_q_val if max_q_val != 0 else torch.inf)),
                abs(min_val / (min_q_val if min_q_val != 0 else torch.inf)),
            )

    # 用 scale 把浮点权重量化到整数域，并在非对称量化时补上 zero point。
    w_q = torch.round(w / w_s).int() + (maybe_w_zp if zero_points else 0)
    # 把量化结果裁剪到当前量化类型允许的整数范围内。
    w_q = torch.clamp(w_q, min_q_val, max_q_val)

    # 部分 kernel 约定 zero point 在乘完 scale 后再扣除，因此单独构造参考值。
    if ref_zero_points_after_scales and maybe_w_zp is not None:
        # 先把量化值与 zero point 分别乘 scale，再做差得到参考权重。
        w_ref = w_q.to(orig_type) * w_s - maybe_w_zp.to(orig_type) * w_s
    else:
        # 常规路径直接先减 zero point，再乘 scale 恢复浮点权重。
        w_ref = (w_q - (maybe_w_zp if zero_points else 0)).to(orig_type) * w_s

    # 若量化类型带存储偏置，则把偏置写回量化整数张量。
    if quant_type.has_bias():
        w_q += quant_type.bias

    # group 量化路径需要把中间重排的张量恢复回原始 `[K, N]` 布局。
    if group_size is not None and group_size < size_k:

        def reshape_w(w):
            # 先恢复成 `[group_size, num_groups, N]`。
            w = w.reshape((group_size, -1, size_n))
            # 再把 group 维换回原来的行分块顺序。
            w = w.permute(1, 0, 2)
            # 最后恢复成逻辑权重矩阵 `[K, N]`。
            w = w.reshape((size_k, size_n)).contiguous()
            return w

        # 恢复量化权重 `w_q: [K, N]`。
        w_q = reshape_w(w_q)
        # 恢复参考浮点权重 `w_ref: [K, N]`。
        w_ref = reshape_w(w_ref)
        # 将 scale 恢复成 `[num_groups, N]` 布局。
        w_s = w_s.reshape((-1, size_n)).contiguous()

    # 若存在 zero point，则把它恢复成 `[num_groups, N]` 并迁回原设备。
    if maybe_w_zp is not None:
        maybe_w_zp = maybe_w_zp.reshape((-1, size_n)).contiguous()
        maybe_w_zp = maybe_w_zp.to(device=orig_device)

    # 返回参考权重、量化权重、scale 与可选 zero point。
    return (
        w_ref.to(device=orig_device),
        w_q.to(device=orig_device),
        w_s if group_size is not None else None,
        maybe_w_zp,
    )


# 声明当前 GPTQ 测试辅助函数支持的量化类型集合。
SUPPORTED_GPTQ_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint8b128]
# 声明当前 GPTQ 测试辅助函数支持的 group_size 集合。
SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]


def gptq_quantize_weights(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
    test_perm: torch.Tensor | None = None,
):
    # 读取权重矩阵逻辑 shape `w: [K, N]`。
    size_k, _ = w.shape

    # GPTQ 权重量化入口要求输入权重必须是浮点张量。
    assert w.is_floating_point(), "w must be float"
    # 当前仅支持预定义的 GPTQ 量化类型。
    assert quant_type in SUPPORTED_GPTQ_QUANT_TYPES, (
        f"Unsupported gptq type = {quant_type}"
    )
    # 当前仅支持预定义的 group size 集合，或整行大小 `size_k`。
    assert group_size in SUPPORTED_GROUP_SIZES + [size_k], (
        f"Unsupported groupsize = {group_size}"
    )

    # 先走通用量化逻辑得到参考权重、量化权重与 scale。
    w_ref, w_q, w_s, _ = quantize_weights(w, quant_type, group_size)

    # act-order 默认关闭时，`g_idx` 与 `rand_perm` 都保持空张量。
    g_idx = torch.empty(0, dtype=torch.int, device=w.device)
    # act-order 默认关闭时，随机置换表也保持空张量。
    rand_perm = torch.empty(0, dtype=torch.int, device=w.device)
    # 启用 act-order 时需要进一步对量化权重做行置换。
    if act_order:
        # act-order 必须在 `group_size < size_k` 的分组场景下才有意义。
        assert group_size < size_k, (
            "For act_order, groupsize = {} must be less than size_k = {}".format(
                group_size, size_k
            )
        )

        # 对参考权重、量化权重与 group 索引同步施加同一行置换。
        w_ref, w_q, g_idx, rand_perm = permute_rows(w_q, w_ref, group_size, test_perm)

    # 返回 GPTQ 参考权重、量化权重、scale、group 索引与置换表。
    return w_ref, w_q, w_s, g_idx, rand_perm


def sort_weights(q_w: torch.Tensor, g_idx: torch.Tensor):
    # 保存原始设备，便于结果回迁。
    orig_device = q_w.device

    # 按 `g_idx: [K]` 对行进行稳定排序，得到排序索引 `sort_indices: [K]`。
    sort_indices = torch.argsort(g_idx).to(dtype=torch.int32)

    # 用排序索引重排 group 索引。
    g_idx = g_idx[sort_indices].contiguous()
    # 用排序索引同步重排行权重 `q_w: [K, N]`。
    q_w = q_w[sort_indices, :].contiguous()

    # 返回排序后的权重、group 索引与排序索引。
    return (
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        sort_indices.to(device=orig_device),
    )


def pack_rows(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    # 输入量化矩阵必须已经是逻辑形状 `[K, N]`。
    assert q_w.shape == (size_k, size_n)

    # 计算一个 int32 中能打包多少个 `num_bits` 元素。
    pack_factor = get_pack_factor(num_bits)
    # 行方向打包要求 `K` 能被 pack_factor 整除。
    assert size_k % pack_factor == 0

    # 保存原始设备，便于 pack 完成后搬回。
    orig_device = q_w.device

    # 将权重搬到 CPU，并转成 `uint32` 以便做位运算。
    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    # 分配行打包结果 `q_res: [K / pack_factor, N]`。
    q_res = numpy.zeros((size_k // pack_factor, size_n), dtype=numpy.uint32)

    # 沿行方向逐槽位把 `pack_factor` 个元素压进一个 uint32。
    for i in range(pack_factor):
        # 将第 `i` 个槽位对应的元素左移到目标 bit 位置，并写入 packed 矩阵。
        q_res |= q_w[i::pack_factor, :] << num_bits * i

    # 将 packed 结果转回 PyTorch `int32` 并搬回原设备。
    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    # 返回按行打包后的量化矩阵。
    return q_res


def pack_cols(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    # 输入量化矩阵必须已经是逻辑形状 `[K, N]`。
    assert q_w.shape == (size_k, size_n)

    # 计算一个 int32 中能打包多少个 `num_bits` 元素。
    pack_factor = get_pack_factor(num_bits)

    # 列方向打包要求 `N` 能被 pack_factor 整除。
    assert size_n % pack_factor == 0

    # 保存原始设备，便于 pack 完成后搬回。
    orig_device = q_w.device

    # 将权重搬到 CPU，并转成 `uint32` 以便做位运算。
    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    # 分配列打包结果 `q_res: [K, N / pack_factor]`。
    q_res = numpy.zeros((size_k, size_n // pack_factor), dtype=numpy.uint32)

    # 沿列方向逐槽位把 `pack_factor` 个元素压进一个 uint32。
    for i in range(pack_factor):
        # 将第 `i` 个槽位对应的列元素左移到目标 bit 位置，并写入 packed 矩阵。
        q_res |= q_w[:, i::pack_factor] << num_bits * i

    # 将 packed 结果转回 PyTorch `int32`，并搬回原设备。
    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)

    # 保证 packed 结果是连续内存布局。
    q_res = q_res.contiguous()

    # 返回按列打包后的量化矩阵。
    return q_res


def unpack_cols(
    packed_q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    # 计算一个 int32 中实际打包了多少个 `num_bits` 元素。
    pack_factor = get_pack_factor(num_bits)

    # 逻辑列数 `N` 必须能被 pack_factor 整除。
    assert size_n % pack_factor == 0

    # packed 输入矩阵必须满足物理 shape `[K, N / pack_factor]`。
    assert packed_q_w.shape == (size_k, size_n // pack_factor), (
        "packed_q_w.shape = {} size_k = {}, size_n = {} pack_Factor = {}".format(
            packed_q_w.shape, size_k, size_n, pack_factor
        )
    )

    # 保存原始设备，便于解包结果回迁。
    orig_device = packed_q_w.device

    # 将 packed 权重搬到 CPU，并转成 `uint32` 以便做位运算。
    packed_q_w_cpu = packed_q_w.cpu().numpy().astype(numpy.uint32)

    # 分配解包结果 `q_res: [K, N]`。
    q_res = numpy.zeros((size_k, size_n), dtype=numpy.uint32)

    # 构造当前低比特元素的位掩码。
    mask = (1 << num_bits) - 1

    # 逐槽位把 packed 的列元素拆回逻辑列。
    for i in range(pack_factor):
        # 取出当前最低 `num_bits` 的槽位值。
        vals = packed_q_w_cpu & mask

        # 将 packed 值右移，为下一轮提取更高位槽位做准备。
        packed_q_w_cpu >>= num_bits

        # 把当前槽位值写回逻辑列 `i, i+pack_factor, ...`。
        q_res[:, i::pack_factor] = vals

    # 将解包结果转回 PyTorch `int32` 并搬回原设备。
    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)

    # 保证解包结果是连续内存布局。
    q_res = q_res.contiguous()

    # 返回按列解包后的量化矩阵。
    return q_res


def gptq_pack(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    # GPTQ 当前约定按行方向将低比特权重 pack 进 int32。
    return pack_rows(q_w, num_bits, size_k, size_n)


def awq_pack(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    # AWQ 打包入口要求输入矩阵已是逻辑形状 `[K, N]`。
    assert q_w.shape == (size_k, size_n)

    # 4bit AWQ 路径使用固定的 8 槽交错顺序。
    if num_bits == 4:
        interleave = numpy.array([0, 2, 4, 6, 1, 3, 5, 7])
    # 8bit AWQ 路径使用固定的 4 槽交错顺序。
    elif num_bits == 8:
        interleave = numpy.array([0, 2, 1, 3])
    else:
        # 其余 bit 宽当前不支持 AWQ pack。
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    # 先按交错分组把最后一维重排。
    q_w = q_w.reshape((-1, len(interleave)))[:, interleave].ravel()
    # 再恢复成逻辑矩阵形状 `[K, N]` 供后续列打包使用。
    q_w = q_w.reshape((-1, size_n)).contiguous()

    # AWQ 当前约定按列方向将交错后的低比特权重 pack 进 int32。
    return pack_cols(q_w, num_bits, size_k, size_n)


def convert_bf16_scales_to_fp8(
    quant_fp8: Callable, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """将 BF16 scale 转成 W4A8 kernel 需要的 `(fp8_scales, chan_scales)`。"""
    # scale 张量必须是连续内存，便于后续 view 成二维矩阵。
    assert scales.is_contiguous(), (
        f"scale tensor must be contiguous, got {scales.stride()=}"
    )
    # scale 张量必须位于 GPU 上，量化辅助函数也在 GPU 侧执行。
    assert scales.is_cuda, "scales must be on gpu"

    # 保存原始 scale 形状，便于后面恢复。
    orig_shape = scales.shape
    # 读取最后一维 group 数 `k_groups`。
    k_groups = orig_shape[-1]
    # 将 scale 拉平成二维矩阵 `flat_scales: [-1, k_groups]`。
    flat_scales = scales.view(-1, k_groups)

    # 调用外部量化回调生成 fp8 scales 与通道 scale。
    fp8_scales, chan_scales = quant_fp8(flat_scales)
    # 按当前 W4A8 约定把 fp8 scales 再缩放到最终 FP8 存储范围。
    fp8_scales = (fp8_scales.float() / 8.0).to(torch.float8_e4m3fn)
    # 与之对应地把通道 scale 放大回补偿系数。
    chan_scales *= 8.0

    # 将 fp8 scales 恢复成原始 scale 形状。
    fp8_scales = fp8_scales.view(orig_shape)
    # 将通道 scale 恢复成与原始前缀维一致的二维尾部布局。
    chan_scales = chan_scales.view(orig_shape[:-1], -1)

    # 返回供 W4A8 kernel 使用的两组 scale。
    return fp8_scales, chan_scales


def convert_packed_uint4b8_to_signed_int4_inplace(t: torch.Tensor) -> torch.Tensor:
    """把 packed `uint4b8` 原地转换成 packed `signed int4`。"""
    # 该转换 helper 当前只支持 GPU 张量。
    assert t.is_cuda, "tensor must be on gpu"
    # 输入 packed 权重必须使用 `int32` 作为承载类型。
    assert t.dtype == torch.int32, f"expected int32 packed weights but got {t.dtype}"

    # 一个 int32 中有 8 个 4-bit 槽位，需要逐个槽位处理。
    for i in range(8):
        # 计算当前槽位的 bit 起始偏移。
        shift = 4 * i
        # 提取当前 4-bit 槽位的无符号值 `nib: [0, 15]`。
        nib = (t >> shift) & 0xF
        # 先把当前槽位的旧 bit 清零。
        t &= ~(0xF << shift)
        # 再把 `[0, 15]` 平移成 `[-8, 7]` 后写回同一 4-bit 槽位。
        t |= ((nib - 8) & 0xF) << shift

    # 返回原地转换后的 packed 权重张量。
    return t
