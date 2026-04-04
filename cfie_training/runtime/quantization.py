"""Training-time GPTQ-style dynamic quantization helpers.

This module borrows two core engineering ideas from community GPTQ projects:

1. Group-wise int4 packing with per-group scales and zero-points.
2. Reconstructing a dequantized compute view on demand from packed storage.

The implementation here is intentionally lightweight and runtime-oriented so it
can slot into the representative CFIE training engine without depending on the
full external inference stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile

import torch

from cfie_training.config import TrainingProjectConfig

_PACK_DTYPE_MAP = {
    "int32": torch.int32,
}
_COMPUTE_VIEW_DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
}
_CUDA_TRANSFORM_CPU_FALLBACK_MIN_ELEMENTS = 1_000_000
_QUANTIZE_GROUP_CHUNK_SIZE = 4096
_RESIZE_INDEX_CACHE: dict[
    tuple[int, int, str],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def _ceil_div(lhs: int, rhs: int) -> int:
    # 返回 lhs / rhs 向上取整后的整数结果。
    return (lhs + rhs - 1) // rhs


def _default_nvme_mirror_root(config: TrainingProjectConfig) -> Path:
    # 为当前 profile 创建一个独立的 FP32 mirror 临时目录。
    return Path(
        tempfile.mkdtemp(
            prefix=f"cfie_training_fp32_master_{config.profile_name}_",
        )
    )


def runtime_gptq_enabled(config: TrainingProjectConfig) -> bool:
    # 只有运行时量化开启、方法为 gptq 且模型自身量化格式也是 gptq 时才启用。
    return (
        config.runtime_quantization.enabled
        and config.runtime_quantization.method == "gptq"
        and config.model_spec.quantization == "gptq"
    )


def runtime_device_weight_bytes_per_param(config: TrainingProjectConfig) -> float:
    # 未启用运行时 GPTQ 时，沿用常规设备权重字节数估算。
    if not runtime_gptq_enabled(config):
        return float(config.state_bytes.device_weight_bytes_per_param)
    # 启用 GPTQ 时，每参数设备权重字节数由 bits/8 决定。
    return config.runtime_quantization.bits / 8.0


def _resize_cache_key(
    *,
    input_size: int,
    output_size: int,
    device: torch.device,
) -> tuple[int, int, str]:
    # 用输入尺寸、输出尺寸和设备标识构造插值索引缓存键。
    return (
        input_size,
        output_size,
        f"{device.type}:{device.index}",
    )


def _linear_resize_indices_and_weights(
    *,
    input_size: int,
    output_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # 输入和输出尺寸都必须为正。
    if input_size < 1 or output_size < 1:
        raise ValueError("resize sizes must be >= 1")
    # 先构造当前尺寸组合对应的缓存键。
    key = _resize_cache_key(
        input_size=input_size,
        output_size=output_size,
        device=device,
    )
    # 命中缓存时直接复用。
    cached = _RESIZE_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    # 计算输出位置映射回输入坐标系后的连续位置。
    positions = (
        (torch.arange(output_size, device=device, dtype=torch.float32) + 0.5)
        * (float(input_size) / float(output_size))
        - 0.5
    )
    # 线性插值的下界索引。
    lower = torch.floor(positions).to(dtype=torch.int64)
    # 线性插值的上界索引。
    upper = lower + 1
    # 上界权重等于连续位置的小数部分。
    upper_weight = positions - lower.to(dtype=torch.float32)
    # 下界权重是互补部分。
    lower_weight = 1.0 - upper_weight
    # 把索引裁到合法输入范围内。
    lower = lower.clamp_(0, input_size - 1)
    upper = upper.clamp_(0, input_size - 1)
    # 把索引与权重一起写入缓存。
    cached = (lower, upper, lower_weight, upper_weight)
    _RESIZE_INDEX_CACHE[key] = cached
    # 返回当前尺寸组合的插值索引和权重。
    return cached


def _resize_linear_along_dim(
    values: torch.Tensor,
    *,
    dim: int,
    output_size: int,
) -> torch.Tensor:
    # 输出尺寸小于 1 时直接返回对应空张量。
    if output_size < 1:
        shape = list(values.shape)
        shape[dim] = output_size
        return torch.empty(*shape, dtype=values.dtype, device=values.device)
    # 读取被插值维度的原始长度。
    input_size = values.shape[dim]
    # 尺寸没变时无需插值。
    if input_size == output_size:
        return values
    # 先把目标维度移到最后，便于统一处理。
    moved = values.movedim(dim, -1)
    # 将前面各维展平成 batch，最后一维保留为待插值轴。
    flat = moved.reshape(-1, input_size)
    # 取出当前尺寸对应的线性插值索引与权重。
    lower_idx, upper_idx, lower_weight, upper_weight = (
        _linear_resize_indices_and_weights(
            input_size=input_size,
            output_size=output_size,
            device=values.device,
        )
    )
    # 采样每个输出位置对应的下界输入值。
    lower = flat.index_select(1, lower_idx)
    # 采样每个输出位置对应的上界输入值。
    upper = flat.index_select(1, upper_idx)
    # 若输入不是浮点，则插值时临时升到 float32。
    weight_dtype = flat.dtype if flat.is_floating_point() else torch.float32
    # 按线性权重混合上下界值。
    blended = (
        lower * lower_weight.to(dtype=weight_dtype).unsqueeze(0)
        + upper * upper_weight.to(dtype=weight_dtype).unsqueeze(0)
    )
    # 把插值结果还原回移动后张量的形状。
    resized = blended.reshape(*moved.shape[:-1], output_size)
    # 再把待插值维度移回原位置。
    return resized.movedim(-1, dim)


def resize_tensor(
    values: torch.Tensor,
    *,
    size: tuple[int, ...],
    mode: str,
) -> torch.Tensor:
    # linear 模式只接受一维目标尺寸。
    if mode == "linear":
        if len(size) != 1:
            raise ValueError("linear resize expects a 1D size tuple")
        return _resize_linear_along_dim(values, dim=0, output_size=size[0])
    # bilinear 模式按两个维度顺序做两次线性插值。
    if mode == "bilinear":
        if len(size) != 2:
            raise ValueError("bilinear resize expects a 2D size tuple")
        resized = _resize_linear_along_dim(values, dim=0, output_size=size[0])
        return _resize_linear_along_dim(resized, dim=1, output_size=size[1])
    # trilinear 模式按三个维度顺序做三次线性插值。
    if mode == "trilinear":
        if len(size) != 3:
            raise ValueError("trilinear resize expects a 3D size tuple")
        resized = _resize_linear_along_dim(values, dim=0, output_size=size[0])
        resized = _resize_linear_along_dim(resized, dim=1, output_size=size[1])
        return _resize_linear_along_dim(resized, dim=2, output_size=size[2])
    # 其他插值模式当前不支持。
    raise ValueError(f"unsupported resize mode: {mode}")


@dataclass(slots=True, frozen=True)
class PackedQuantizedTensor:
    qweight: torch.Tensor
    qzeros: torch.Tensor
    scales: torch.Tensor
    original_numel: int
    padded_numel: int
    group_size: int
    bits: int
    sym: bool
    pack_dtype_name: str = "int32"

    @property
    def pack_factor(self) -> int:
        # 一个 int32 word 内可打包的量化值个数由 bits 决定。
        return 32 // self.bits

    @property
    def group_count(self) -> int:
        # group 数等于 padded_numel / group_size 向上取整。
        return _ceil_div(self.padded_numel, self.group_size)

    @property
    def resident_bytes(self) -> int:
        # 常驻字节数等于 qweight、qzeros、scales 三部分之和。
        return (
            self.qweight.numel() * self.qweight.element_size()
            + self.qzeros.numel() * self.qzeros.element_size()
            + self.scales.numel() * self.scales.element_size()
        )

    def to_device(self, device: torch.device) -> "PackedQuantizedTensor":
        # 把打包权重及其伴生张量整体迁移到目标设备。
        return PackedQuantizedTensor(
            qweight=self.qweight.to(device=device),
            qzeros=self.qzeros.to(device=device),
            scales=self.scales.to(device=device),
            original_numel=self.original_numel,
            padded_numel=self.padded_numel,
            group_size=self.group_size,
            bits=self.bits,
            sym=self.sym,
            pack_dtype_name=self.pack_dtype_name,
        )


def _quantize_values(
    values: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    sym: bool,
    pack_dtype_name: str,
) -> PackedQuantizedTensor:
    # 先解析 pack dtype 与量化区间。
    pack_dtype = _PACK_DTYPE_MAP[pack_dtype_name]
    maxq = (1 << bits) - 1
    pack_factor = 32 // bits
    # group_size 必须能被 pack_factor 整除，才能整齐打包。
    if group_size % pack_factor != 0:
        raise ValueError("group_size must be divisible by 32 // bits")
    # 把输入拉平成 CPU float32 向量。
    flat = values.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    original_numel = int(flat.numel())
    # padded_numel 至少覆盖一个完整 group，并向 group_size 对齐。
    padded_numel = max(group_size, _ceil_div(original_numel, group_size) * group_size)
    # 计算总 group 数。
    group_count = padded_numel // group_size
    # 预分配每个 group 的 scale。
    scales = torch.empty(group_count, dtype=torch.float32, device="cpu")
    # 预分配每个 group 的 zero-point。
    zero_points = torch.empty(group_count, dtype=torch.int32, device="cpu")
    # 预分配最终打包后的 qweight。
    packed_weight = torch.empty(
        padded_numel // pack_factor,
        dtype=pack_dtype,
        device="cpu",
    )
    # 按 chunk 分批量化多个 group，控制峰值内存。
    chunk_group_count = max(1, min(group_count, _QUANTIZE_GROUP_CHUNK_SIZE))
    packed_write_offset = 0
    # 逐个 group chunk 做量化和打包。
    for group_start in range(0, group_count, chunk_group_count):
        # 当前这批 chunk 里实际包含的 group 数。
        current_group_count = min(chunk_group_count, group_count - group_start)
        # 计算这批 group 对应的原始值区间。
        value_start = group_start * group_size
        value_stop = value_start + current_group_count * group_size
        # 先构造零填充的 chunk 缓冲区。
        chunk = torch.zeros(
            current_group_count * group_size,
            dtype=torch.float32,
            device="cpu",
        )
        # 只拷贝原始向量中实际存在的部分，其余保持 0 作为 padding。
        source_stop = min(value_stop, original_numel)
        if source_stop > value_start:
            chunk[: source_stop - value_start].copy_(flat[value_start:source_stop])
        # 还原成 [group, group_size] 形式，便于逐组量化。
        grouped = chunk.view(current_group_count, group_size)
        # 对称量化路径。
        if sym:
            # 对称量化的 zero-point 固定在中点。
            chunk_zero_points = torch.full(
                (current_group_count,),
                1 << (bits - 1),
                dtype=torch.int32,
                device="cpu",
            )
            # 对称量化的 scale 由组内最大绝对值决定。
            denom = max((1 << (bits - 1)) - 1, 1)
            chunk_scales = grouped.abs().amax(dim=1).div(float(denom)).clamp_min_(1e-8)
            # 先缩放再四舍五入得到量化整数。
            quantized = torch.round(grouped / chunk_scales.unsqueeze(-1)).to(
                dtype=torch.int32
            )
            # 加上 zero-point 并裁剪到合法量化区间。
            quantized.add_(chunk_zero_points.unsqueeze(-1)).clamp_(0, maxq)
        else:
            # 非对称量化路径先算组内最小值和最大值。
            group_min = grouped.amin(dim=1)
            group_max = grouped.amax(dim=1)
            # scale 由动态范围除以 maxq 得到。
            chunk_scales = (
                (group_max - group_min)
                .div(float(max(maxq, 1)))
                .clamp_min_(1e-8)
            )
            # zero-point 由最小值反推，并裁到合法范围。
            chunk_zero_points = torch.round(-group_min / chunk_scales).to(
                dtype=torch.int32
            )
            chunk_zero_points.clamp_(0, maxq)
            # 按 scale 量化并平移到非负区间。
            quantized = torch.round(grouped / chunk_scales.unsqueeze(-1)).to(
                dtype=torch.int32
            )
            quantized.add_(chunk_zero_points.unsqueeze(-1)).clamp_(0, maxq)
        # 写回当前 chunk 对应的 scale。
        scales[group_start : group_start + current_group_count].copy_(chunk_scales)
        # 写回当前 chunk 对应的 zero-point。
        zero_points[group_start : group_start + current_group_count].copy_(
            chunk_zero_points
        )
        # 把量化后的值展平并打包成 int32 words。
        packed_chunk = _pack_values(
            quantized.reshape(-1),
            bits=bits,
            pack_dtype=pack_dtype,
        )
        # 把打包结果写到 packed_weight 的对应位置。
        packed_weight[
            packed_write_offset : packed_write_offset + packed_chunk.numel()
        ].copy_(packed_chunk)
        packed_write_offset += packed_chunk.numel()
    # zero-point 也需要补齐到 pack_factor 的整数倍才能打包。
    zero_pad_groups = max(
        pack_factor,
        _ceil_div(zero_points.numel(), pack_factor) * pack_factor,
    )
    # 构造带 padding 的 zero-point 缓冲区。
    padded_zeros = torch.zeros(zero_pad_groups, dtype=torch.int32, device="cpu")
    padded_zeros[: zero_points.numel()].copy_(zero_points)
    # 打包 zero-point。
    packed_zeros = _pack_values(
        padded_zeros,
        bits=bits,
        pack_dtype=pack_dtype,
    )
    # 返回完整的打包量化张量对象。
    return PackedQuantizedTensor(
        qweight=packed_weight.contiguous(),
        qzeros=packed_zeros.contiguous(),
        scales=scales.contiguous(),
        original_numel=original_numel,
        padded_numel=padded_numel,
        group_size=group_size,
        bits=bits,
        sym=sym,
        pack_dtype_name=pack_dtype_name,
    )


def _dequantize_values(
    packed: PackedQuantizedTensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # 先把打包 qweight 解包回逐元素整数值。
    unpacked_weight = _unpack_values(
        packed.qweight,
        bits=packed.bits,
        count=packed.padded_numel,
        device=device,
    )
    # 再把打包 qzeros 解包回逐 group 的 zero-point。
    zero_points = _unpack_values(
        packed.qzeros,
        bits=packed.bits,
        count=packed.group_count,
        device=device,
    )
    # 为每个元素构造其所属的 group 索引。
    g_idx = torch.div(
        torch.arange(packed.padded_numel, device=device, dtype=torch.int32),
        packed.group_size,
        rounding_mode="floor",
    )
    # 将 scales 搬到目标设备。
    scales = packed.scales.to(device=device, dtype=torch.float32)
    # 按 `(q - zero_point) * scale` 恢复浮点值。
    values = (
        unpacked_weight.to(dtype=torch.float32)
        - zero_points.index_select(0, g_idx).to(dtype=torch.float32)
    ) * scales.index_select(0, g_idx)
    # 裁掉 padding 部分并转换到目标 dtype。
    return values[: packed.original_numel].to(dtype=dtype)


def _dequantize_range(
    packed: PackedQuantizedTensor,
    *,
    start_offset: int,
    length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # 起始偏移必须非负。
    if start_offset < 0:
        raise ValueError("start_offset must be >= 0")
    # 读取长度必须非负。
    if length < 0:
        raise ValueError("length must be >= 0")
    # 空范围直接返回空张量。
    if length == 0:
        return torch.empty(0, dtype=dtype, device=device)
    # 起始偏移不能超出原始张量长度。
    if start_offset >= packed.original_numel:
        raise ValueError("start_offset is beyond packed tensor length")
    # 将 stop_offset 裁到 original_numel 范围内。
    stop_offset = min(start_offset + length, packed.original_numel)
    value_count = stop_offset - start_offset
    # 裁剪后仍为空时直接返回空张量。
    if value_count <= 0:
        return torch.empty(0, dtype=dtype, device=device)

    # 计算本次读取会覆盖到的 group 范围。
    group_start = start_offset // packed.group_size
    group_stop = _ceil_div(stop_offset, packed.group_size)
    # 只解包当前范围覆盖到的 qweight 子区间。
    unpacked_weight = _unpack_values_range(
        packed.qweight,
        bits=packed.bits,
        start=start_offset,
        count=value_count,
        device=device,
    )
    # 只解包当前范围覆盖到的 zero-point 子区间。
    zero_points = _unpack_values_range(
        packed.qzeros,
        bits=packed.bits,
        start=group_start,
        count=max(group_stop - group_start, 1),
        device=device,
    )
    # 构造当前范围内每个位置对应的原始绝对下标。
    positions = torch.arange(
        start_offset,
        stop_offset,
        dtype=torch.int64,
        device=device,
    )
    # 将绝对 group 索引转换为当前子区间内的局部索引。
    local_group_idx = torch.div(
        positions,
        packed.group_size,
        rounding_mode="floor",
    ) - group_start
    # 仅搬运当前范围涉及到的 scales 子向量。
    scales = packed.scales.narrow(0, group_start, max(group_stop - group_start, 1)).to(
        device=device,
        dtype=torch.float32,
    )
    # 恢复当前范围内的浮点值。
    values = (
        unpacked_weight.to(dtype=torch.float32)
        - zero_points.index_select(0, local_group_idx).to(dtype=torch.float32)
    ) * scales.index_select(0, local_group_idx)
    # 转成目标 dtype 后返回。
    return values.to(dtype=dtype)


def dequantize_packed_range(
    packed: PackedQuantizedTensor,
    *,
    start_offset: int,
    length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # 对外导出指定范围的打包权重解量化接口。
    return _dequantize_range(
        packed,
        start_offset=start_offset,
        length=length,
        device=device,
        dtype=dtype,
    )


def _transformed_packed_slice(
    packed: PackedQuantizedTensor,
    *,
    start_offset: int,
    raw_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    resize_shape: tuple[int, ...] | None = None,
    resize_mode: str | None = None,
    expert_index: int | None = None,
    transpose_last_two: bool = False,
    tanh_scale: float = 0.0,
    add_scalar: float = 0.0,
) -> torch.Tensor:
    # 默认在目标 device/dtype 上完成全部变换。
    transform_device = device
    transform_dtype = dtype
    # 大张量的 resize 在 CUDA 上成本高时，退回 CPU 做变换。
    if (
        device.type == "cuda"
        and resize_shape is not None
        and resize_mode is not None
        and math.prod(raw_shape) >= _CUDA_TRANSFORM_CPU_FALLBACK_MIN_ELEMENTS
    ):
        transform_device = torch.device("cpu")
        transform_dtype = torch.float32
    # 先把目标范围解量化并恢复成原始 shape。
    value = _dequantize_range(
        packed,
        start_offset=start_offset,
        length=math.prod(raw_shape),
        device=transform_device,
        dtype=transform_dtype,
    ).view(*raw_shape)
    # 如配置了 resize，且当前 shape 与目标 shape 不同，则执行插值。
    if (
        resize_shape is not None
        and resize_mode is not None
        and tuple(value.shape) != tuple(resize_shape)
    ):
        value = resize_tensor(
            value,
            size=tuple(resize_shape),
            mode=resize_mode,
        )
    # expert_index 仅适用于 3D 张量，表示取单个 expert 切片。
    if expert_index is not None:
        if value.ndim != 3:
            raise ValueError("expert_index requires a 3D packed slice")
        if expert_index < 0 or expert_index >= value.shape[0]:
            raise ValueError("expert_index is out of bounds for packed slice")
        value = value[expert_index]
    # 需要时交换最后两个维度。
    if transpose_last_two:
        value = value.transpose(-2, -1)
    # 需要时施加 tanh 缩放。
    if tanh_scale != 0.0:
        value = tanh_scale * torch.tanh(value)
    # 需要时加上常数偏置。
    if add_scalar != 0.0:
        value = value + add_scalar
    # 若中间在别的设备或 dtype 上变换，最后搬回目标 device/dtype。
    if value.device != device or value.dtype != dtype:
        value = value.to(device=device, dtype=dtype)
    # 返回完成所有变换后的切片张量。
    return value


def _pack_values(
    values: torch.Tensor,
    *,
    bits: int,
    pack_dtype: torch.dtype,
) -> torch.Tensor:
    # 一个 int32 word 内可打包的量化值个数。
    pack_factor = 32 // bits
    # 待打包值的长度必须是 pack_factor 的整数倍。
    if values.numel() % pack_factor != 0:
        raise ValueError("values length must be divisible by pack_factor")
    # 构造当前 bit 宽对应的位移量。
    shifts = torch.arange(
        0,
        32,
        bits,
        dtype=torch.int64,
        device=values.device,
    )
    # 重排成 [num_words, pack_factor] 形式。
    chunks = values.to(dtype=torch.int64).view(-1, pack_factor)
    # 逐列左移后求和，得到打包后的 int32 words。
    packed = torch.sum(chunks << shifts.unsqueeze(0), dim=1)
    # 转成目标 pack dtype 返回。
    return packed.to(dtype=pack_dtype)


def _unpack_values(
    packed: torch.Tensor,
    *,
    bits: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    # 构造当前 bit 宽对应的掩码。
    mask = (1 << bits) - 1
    # 构造每个量化值在 int32 word 中对应的位移量。
    shifts = torch.arange(
        0,
        32,
        bits,
        dtype=torch.int64,
        device=device,
    )
    # 把 packed words 搬到目标设备并扩一维，便于逐位右移。
    words = packed.to(device=device, dtype=torch.int64).unsqueeze(-1)
    # 右移后取掩码，恢复逐元素量化整数。
    unpacked = torch.bitwise_and(torch.bitwise_right_shift(words, shifts), mask)
    # 展平并裁到 count 个元素。
    return unpacked.reshape(-1)[:count].to(dtype=torch.int32)


def _unpack_values_range(
    packed: torch.Tensor,
    *,
    bits: int,
    start: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    # 起始偏移必须非负。
    if start < 0:
        raise ValueError("start must be >= 0")
    # 读取长度必须非负。
    if count < 0:
        raise ValueError("count must be >= 0")
    # 空范围直接返回空张量。
    if count == 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    # 计算一个 word 内可容纳的量化值数量。
    pack_factor = 32 // bits
    # 先定位需要读取的 word 范围。
    word_start = start // pack_factor
    word_stop = _ceil_div(start + count, pack_factor)
    # 起始 word 越界时说明当前起始偏移非法。
    if word_start >= packed.numel():
        raise ValueError("start is beyond packed tensor length")
    # 把结束 word 裁到实际长度内。
    word_stop = min(word_stop, packed.numel())
    # 计算当前局部解包结果中的起始偏移。
    local_start = start - word_start * pack_factor
    # 先解包覆盖当前范围的最小 word 区间。
    unpacked = _unpack_values(
        packed.narrow(0, word_start, max(word_stop - word_start, 1)),
        bits=bits,
        count=(word_stop - word_start) * pack_factor,
        device=device,
    )
    # 计算当前局部范围内最终能拿到的元素数。
    local_count = min(count, max(unpacked.numel() - local_start, 0))
    # 当前范围在局部解包结果中为空时返回空张量。
    if local_count <= 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    # 返回局部解包结果中的目标切片。
    return unpacked.narrow(0, local_start, local_count)


class GPTQTrainingQuantizer:
    def __init__(self, config: TrainingProjectConfig) -> None:
        # 保存运行时量化配置视图。
        self._cfg = config.runtime_quantization
        # 记录当前运行时是否启用了 GPTQ。
        self._enabled = runtime_gptq_enabled(config)
        # 解析 pack dtype。
        self._pack_dtype = _PACK_DTYPE_MAP[self._cfg.pack_dtype]
        # 解析计算视图 dtype。
        self._compute_view_dtype = _COMPUTE_VIEW_DTYPE_MAP[
            self._cfg.compute_view_dtype
        ]

    @property
    def enabled(self) -> bool:
        # 返回当前 quantizer 是否可用。
        return self._enabled

    @property
    def compute_view_dtype(self) -> torch.dtype:
        # 返回当前 dequantized compute view 默认使用的 dtype。
        return self._compute_view_dtype

    def quantize(self, values: torch.Tensor) -> PackedQuantizedTensor:
        # 未启用时禁止调用量化。
        if not self._enabled:
            raise RuntimeError("runtime GPTQ quantizer is not enabled")
        # 按运行时配置执行 GPTQ 风格打包量化。
        return _quantize_values(
            values,
            bits=self._cfg.bits,
            group_size=self._cfg.group_size,
            sym=self._cfg.sym,
            pack_dtype_name=self._cfg.pack_dtype,
        )

    def dequantize(
        self,
        packed: PackedQuantizedTensor,
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        # 未启用时禁止调用解量化。
        if not self._enabled:
            raise RuntimeError("runtime GPTQ quantizer is not enabled")
        # 未显式指定 dtype 时，使用 quantizer 的 compute view dtype。
        compute_dtype = self._compute_view_dtype if dtype is None else dtype
        # 把打包权重恢复成目标设备上的浮点计算视图。
        return _dequantize_values(
            packed,
            device=device,
            dtype=compute_dtype,
        )


class _QuantizedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bits: int,
        group_size: int,
        sym: bool,
        pack_dtype_name: str,
        compute_view_dtype_name: str,
    ) -> torch.Tensor:
        # 当前量化线性层只支持 2D 权重矩阵。
        if weight.ndim != 2:
            raise ValueError("quantized linear expects a 2D weight matrix")
        # 解析前向计算所使用的浮点 dtype。
        compute_dtype = _COMPUTE_VIEW_DTYPE_MAP[compute_view_dtype_name]
        # 先把实权重量化成 packed 结构。
        packed = _quantize_values(
            weight,
            bits=bits,
            group_size=group_size,
            sym=sym,
            pack_dtype_name=pack_dtype_name,
        )
        # 把打包权重迁移到输入所在设备。
        packed = packed.to_device(inputs.device)
        # 现场解量化出计算视图，并恢复成原始权重 shape。
        dequantized = _dequantize_values(
            packed,
            device=inputs.device,
            dtype=compute_dtype,
        ).view_as(weight)
        # 将输入展平成二维矩阵，便于执行 GEMM。
        flat_inputs = inputs.reshape(-1, inputs.shape[-1]).to(dtype=compute_dtype)
        # 执行线性变换。
        output = flat_inputs @ dequantized
        # 保存反向所需的输入和打包权重。
        ctx.save_for_backward(
            flat_inputs,
            packed.qweight,
            packed.qzeros,
            packed.scales,
        )
        # 记录反向阶段需要的各种元信息。
        ctx.input_shape = tuple(inputs.shape)
        ctx.weight_shape = tuple(weight.shape)
        ctx.bits = bits
        ctx.group_size = group_size
        ctx.sym = sym
        ctx.pack_dtype_name = pack_dtype_name
        ctx.compute_view_dtype_name = compute_view_dtype_name
        ctx.input_dtype = inputs.dtype
        ctx.weight_dtype = weight.dtype
        # 恢复前向输出 shape，并转回输入 dtype。
        return output.view(*inputs.shape[:-1], weight.shape[-1]).to(dtype=inputs.dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        # 取回前向保存的输入与 packed 权重。
        flat_inputs, qweight, qzeros, scales = ctx.saved_tensors
        # 根据保存的元信息重建 PackedQuantizedTensor。
        packed = PackedQuantizedTensor(
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            original_numel=math.prod(ctx.weight_shape),
            padded_numel=max(
                ctx.group_size,
                _ceil_div(
                    math.prod(ctx.weight_shape),
                    ctx.group_size,
                ) * ctx.group_size,
            ),
            group_size=ctx.group_size,
            bits=ctx.bits,
            sym=ctx.sym,
            pack_dtype_name=ctx.pack_dtype_name,
        )
        # 解析反向计算使用的浮点 dtype。
        compute_dtype = _COMPUTE_VIEW_DTYPE_MAP[ctx.compute_view_dtype_name]
        # 在 grad_output 所在设备上解量化出权重矩阵。
        dequantized = _dequantize_values(
            packed,
            device=grad_output.device,
            dtype=compute_dtype,
        ).view(ctx.weight_shape)
        # 将 grad_output 展平成二维矩阵。
        flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1]).to(
            dtype=compute_dtype
        )
        # 反向传播输入梯度。
        grad_input = (flat_grad_output @ dequantized.transpose(0, 1)).view(
            ctx.input_shape
        )
        # 反向传播权重梯度。
        grad_weight = flat_inputs.transpose(0, 1) @ flat_grad_output
        # 返回各输入项对应的梯度；超参数项没有梯度。
        return (
            grad_input.to(dtype=ctx.input_dtype),
            grad_weight.to(dtype=ctx.weight_dtype),
            None,
            None,
            None,
            None,
            None,
        )


class _PackedSliceLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        original_numel: int,
        padded_numel: int,
        group_size: int,
        bits: int,
        sym: bool,
        pack_dtype_name: str,
        start_offset: int,
        raw_rows: int,
        raw_cols: int,
        transpose_last_two: bool,
        tanh_scale: float,
        compute_view_dtype_name: str,
        raw_shape: tuple[int, ...] | None,
        resize_shape: tuple[int, ...] | None,
        resize_mode: str | None,
        expert_index: int | None,
        add_scalar: float,
    ) -> torch.Tensor:
        # 解析前向计算使用的浮点 dtype。
        compute_dtype = _COMPUTE_VIEW_DTYPE_MAP[compute_view_dtype_name]
        # 依据传入的 packed 组成部分重建 PackedQuantizedTensor。
        packed = PackedQuantizedTensor(
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            original_numel=original_numel,
            padded_numel=padded_numel,
            group_size=group_size,
            bits=bits,
            sym=sym,
            pack_dtype_name=pack_dtype_name,
        )
        # 原始 shape 显式给出时优先使用，否则退回 rows/cols 二维形状。
        source_shape = tuple(raw_shape) if raw_shape is not None else (raw_rows, raw_cols)
        # 从 packed 权重中解量化并按要求执行 resize / transpose / expert slice 等变换。
        slice_values = _transformed_packed_slice(
            packed,
            start_offset=start_offset,
            raw_shape=source_shape,
            device=inputs.device,
            dtype=compute_dtype,
            resize_shape=tuple(resize_shape) if resize_shape is not None else None,
            resize_mode=resize_mode,
            expert_index=expert_index,
            transpose_last_two=transpose_last_two,
            tanh_scale=tanh_scale,
            add_scalar=add_scalar,
        )
        # 线性层最终必须拿到二维权重矩阵。
        if slice_values.ndim != 2:
            raise ValueError("packed slice linear expects the transformed weight to be 2D")
        # 展平输入，便于执行 GEMM。
        flat_inputs = inputs.reshape(-1, inputs.shape[-1]).to(dtype=compute_dtype)
        # 执行线性变换。
        output = flat_inputs @ slice_values
        # 保存反向所需的输入与 packed 权重。
        ctx.save_for_backward(flat_inputs, qweight, qzeros, scales)
        # 记录反向所需的所有元信息。
        ctx.input_shape = tuple(inputs.shape)
        ctx.weight_shape = tuple(weight.shape)
        ctx.original_numel = original_numel
        ctx.padded_numel = padded_numel
        ctx.group_size = group_size
        ctx.bits = bits
        ctx.sym = sym
        ctx.pack_dtype_name = pack_dtype_name
        ctx.start_offset = start_offset
        ctx.raw_rows = raw_rows
        ctx.raw_cols = raw_cols
        ctx.raw_shape = source_shape
        ctx.resize_shape = None if resize_shape is None else tuple(resize_shape)
        ctx.resize_mode = resize_mode
        ctx.expert_index = expert_index
        ctx.transpose_last_two = transpose_last_two
        ctx.tanh_scale = tanh_scale
        ctx.add_scalar = add_scalar
        ctx.compute_view_dtype_name = compute_view_dtype_name
        ctx.input_dtype = inputs.dtype
        ctx.weight_dtype = weight.dtype
        # 恢复前向输出 shape，并转回输入 dtype。
        return output.view(*inputs.shape[:-1], weight.shape[-1]).to(dtype=inputs.dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        # 取回前向保存的输入和 packed 权重。
        flat_inputs, qweight, qzeros, scales = ctx.saved_tensors
        # 重建 PackedQuantizedTensor。
        packed = PackedQuantizedTensor(
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            original_numel=ctx.original_numel,
            padded_numel=ctx.padded_numel,
            group_size=ctx.group_size,
            bits=ctx.bits,
            sym=ctx.sym,
            pack_dtype_name=ctx.pack_dtype_name,
        )
        # 解析反向计算使用的浮点 dtype。
        compute_dtype = _COMPUTE_VIEW_DTYPE_MAP[ctx.compute_view_dtype_name]
        # 按前向同样的变换规则恢复权重矩阵。
        weight_matrix = _transformed_packed_slice(
            packed,
            start_offset=ctx.start_offset,
            raw_shape=ctx.raw_shape,
            device=grad_output.device,
            dtype=compute_dtype,
            resize_shape=ctx.resize_shape,
            resize_mode=ctx.resize_mode,
            expert_index=ctx.expert_index,
            transpose_last_two=ctx.transpose_last_two,
            tanh_scale=ctx.tanh_scale,
            add_scalar=ctx.add_scalar,
        )
        # 反向需要二维权重矩阵。
        if weight_matrix.ndim != 2:
            raise ValueError("packed slice linear expects a 2D transformed weight")
        # 展平 grad_output，便于执行 GEMM。
        flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1]).to(
            dtype=compute_dtype
        )
        # 计算输入梯度。
        grad_input = (flat_grad_output @ weight_matrix.transpose(0, 1)).view(
            ctx.input_shape
        )
        # 计算权重梯度。
        grad_weight = flat_inputs.transpose(0, 1) @ flat_grad_output
        # 仅 inputs 和 weight 有梯度，其余 packed 元信息没有梯度。
        return (
            grad_input.to(dtype=ctx.input_dtype),
            grad_weight.to(dtype=ctx.weight_dtype),
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


def quantized_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    sym: bool,
    pack_dtype_name: str = "int32",
    compute_view_dtype_name: str = "fp32",
) -> torch.Tensor:
    # 对外导出“先量化再线性”的 autograd 包装接口。
    return _QuantizedLinearFunction.apply(
        inputs,
        weight,
        bits,
        group_size,
        sym,
        pack_dtype_name,
        compute_view_dtype_name,
    )


def quantized_linear_from_packed_slice(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    *,
    packed: PackedQuantizedTensor,
    start_offset: int,
    raw_rows: int,
    raw_cols: int,
    transpose_last_two: bool = False,
    tanh_scale: float = 0.0,
    compute_view_dtype_name: str = "fp32",
    raw_shape: tuple[int, ...] | None = None,
    resize_shape: tuple[int, ...] | None = None,
    resize_mode: str | None = None,
    expert_index: int | None = None,
    add_scalar: float = 0.0,
) -> torch.Tensor:
    # 对外导出“从 packed slice 直接构造线性层”的 autograd 包装接口。
    return _PackedSliceLinearFunction.apply(
        inputs,
        weight,
        packed.qweight,
        packed.qzeros,
        packed.scales,
        packed.original_numel,
        packed.padded_numel,
        packed.group_size,
        packed.bits,
        packed.sym,
        packed.pack_dtype_name,
        start_offset,
        raw_rows,
        raw_cols,
        transpose_last_two,
        tanh_scale,
        compute_view_dtype_name,
        raw_shape,
        resize_shape,
        resize_mode,
        expert_index,
        add_scalar,
    )


class FP32MasterMirror:
    def __init__(self, config: TrainingProjectConfig) -> None:
        # 保存运行时量化配置视图。
        self._cfg = config.runtime_quantization
        # 只有 GPTQ 开启且显式要求持久化 FP32 master 时才启用镜像。
        self._enabled = runtime_gptq_enabled(config) and self._cfg.persist_fp32_to_nvme
        # 优先使用显式指定的 NVMe staging 目录。
        if self._cfg.nvme_staging_dir:
            base_root = Path(self._cfg.nvme_staging_dir).expanduser()
            self._root = base_root / self._cfg.session_id
        else:
            # 未指定时退回临时目录。
            self._root = _default_nvme_mirror_root(config).expanduser()

    @property
    def enabled(self) -> bool:
        # 返回当前 FP32 mirror 是否启用。
        return self._enabled

    @property
    def root(self) -> Path:
        # 返回 FP32 mirror 的根目录。
        return self._root

    def _path_for_group(self, group_id: str) -> Path:
        # 用 group_id 哈希生成稳定的镜像文件路径。
        digest = hashlib.sha1(group_id.encode("utf-8")).hexdigest()
        return self._root / f"{digest}.pt"

    def path_for_group(self, group_id: str) -> Path:
        # 对外暴露指定 group 的镜像文件路径。
        return self._path_for_group(group_id)

    def load(self, group_id: str) -> torch.Tensor | None:
        # 未启用 mirror 时直接返回 None。
        if not self._enabled:
            return None
        # 定位当前 group 的镜像文件。
        path = self._path_for_group(group_id)
        # 镜像文件不存在时返回 None。
        if not path.exists():
            return None
        # 从磁盘加载镜像 payload。
        payload = torch.load(path, map_location="cpu")
        values = payload.get("values")
        # payload 结构不符合预期时返回 None。
        if not isinstance(values, torch.Tensor):
            return None
        # 统一返回展平后的 CPU float32 张量。
        return values.to(dtype=torch.float32, device="cpu").reshape(-1).contiguous()

    def save(self, group_id: str, values: torch.Tensor) -> str | None:
        # 未启用 mirror 时不执行落盘。
        if not self._enabled:
            return None
        # 确保镜像根目录存在。
        self._root.mkdir(parents=True, exist_ok=True)
        # 生成当前 group 的目标镜像路径。
        path = self._path_for_group(group_id)
        # 先写到同目录临时文件，后续再原子替换。
        with tempfile.NamedTemporaryFile(
            dir=self._root,
            suffix=".pt.tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        try:
            # 把 group_id 和 FP32 values 写入临时文件。
            torch.save(
                {
                    "group_id": group_id,
                    "values": values.detach().to(dtype=torch.float32, device="cpu"),
                },
                temp_path,
            )
            # 原子替换为正式镜像文件。
            os.replace(temp_path, path)
        finally:
            # 清理可能残留的临时文件。
            if temp_path.exists():
                temp_path.unlink()
        # 返回最终镜像路径。
        return str(path)
