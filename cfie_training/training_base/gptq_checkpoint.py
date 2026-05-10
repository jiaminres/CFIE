"""GPTQ Checkpoint 解码——从 checkpoint tensor 恢复 pack_qweight/qzeros。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

GPTQ_BITS = 4
GPTQ_PACK_FACTOR_INT32 = 32 // GPTQ_BITS
GPTQ_DEFAULT_SYMMETRIC_ZERO = 8

GptqDecodedLayout = Literal["k_n", "n_k"]


def _require_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


@dataclass(frozen=True, slots=True)
class GptqInt4CheckpointTensors:
    qweight: Any
    scales: Any
    qzeros: Any | None = None
    g_idx: Any | None = None


@dataclass(frozen=True, slots=True)
class GptqInt4CheckpointDecoder:
    group_size: int = 128
    qzeros_are_biased: bool = True
    default_zero_point: int = GPTQ_DEFAULT_SYMMETRIC_ZERO
    decoded_layout: GptqDecodedLayout = "n_k"

    def __post_init__(self) -> None:
        _require_positive_int("group_size", self.group_size)
        if self.decoded_layout not in {"k_n", "n_k"}:
            raise ValueError("decoded_layout must be 'k_n' or 'n_k'")

    def decode(self, tensors: GptqInt4CheckpointTensors) -> torch.Tensor:
        qweight = _as_cpu_int_tensor(tensors.qweight, name="qweight")
        scales = _as_cpu_float32_tensor(tensors.scales, name="scales")
        g_idx = (
            _as_cpu_int_tensor(tensors.g_idx, name="g_idx").reshape(-1)
            if tensors.g_idx is not None
            else None
        )
        size_k = self._infer_size_k(qweight, scales, g_idx)
        size_n = self._infer_size_n(qweight)

        quantized = unpack_gptq_int4_qweight(qweight, size_k=size_k, size_n=size_n)
        group_ids = self._group_ids(size_k, g_idx)
        scale_matrix = self._normalize_scales(scales, group_ids, size_n)
        zero_matrix = self._zero_matrix(tensors.qzeros, group_ids, size_n)
        decoded = (quantized.to(torch.float32) - zero_matrix).mul(scale_matrix)
        if self.decoded_layout == "n_k":
            decoded = decoded.t().contiguous()
        return decoded.contiguous()

    def _infer_size_k(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor | None,
    ) -> int:
        if g_idx is not None:
            return int(g_idx.numel())
        packed_k = int(qweight.reshape(qweight.shape[0], -1).shape[0])
        inferred_from_qweight = packed_k * GPTQ_PACK_FACTOR_INT32
        if scales.ndim >= 1 and scales.shape[0] > 1:
            return min(inferred_from_qweight, int(scales.shape[0]) * self.group_size)
        return inferred_from_qweight

    @staticmethod
    def _infer_size_n(qweight: torch.Tensor) -> int:
        if qweight.ndim != 2:
            raise ValueError("qweight must have shape [packed_k, n]")
        return int(qweight.shape[1])

    def _group_ids(
        self,
        size_k: int,
        g_idx: torch.Tensor | None,
    ) -> torch.Tensor:
        if g_idx is None:
            return torch.arange(size_k, dtype=torch.int64) // self.group_size
        if g_idx.numel() != size_k:
            raise ValueError("g_idx length must match K")
        if (g_idx < 0).any():
            raise ValueError("g_idx must be non-negative")
        return g_idx.to(torch.int64)

    @staticmethod
    def _normalize_scales(
        scales: torch.Tensor,
        group_ids: torch.Tensor,
        size_n: int,
    ) -> torch.Tensor:
        if scales.ndim == 1:
            scales_2d = scales.reshape(1, -1)
        elif scales.ndim == 2:
            scales_2d = scales
        else:
            raise ValueError("scales must have shape [groups, n] or [n]")
        if scales_2d.shape[1] != size_n:
            raise ValueError("scales N dimension must match qweight N")
        if int(group_ids.max().item()) >= scales_2d.shape[0]:
            raise ValueError("g_idx references a scale group outside scales")
        return scales_2d[group_ids].contiguous()

    def _zero_matrix(
        self,
        qzeros: Any | None,
        group_ids: torch.Tensor,
        size_n: int,
    ) -> torch.Tensor:
        if qzeros is None:
            return torch.full(
                (group_ids.numel(), size_n),
                float(self.default_zero_point),
                dtype=torch.float32,
            )
        qzeros_tensor = _as_cpu_int_tensor(qzeros, name="qzeros")
        zeros = unpack_gptq_int4_qzeros(
            qzeros_tensor,
            num_groups=int(group_ids.max().item()) + 1,
            size_n=size_n,
        )
        if self.qzeros_are_biased:
            zeros = zeros + 1
        return zeros[group_ids].to(torch.float32).contiguous()


def unpack_gptq_int4_qweight(
    qweight: Any,
    *,
    size_k: int,
    size_n: int,
) -> torch.Tensor:
    _require_positive_int("size_k", size_k)
    _require_positive_int("size_n", size_n)
    qweight_tensor = _as_cpu_int_tensor(qweight, name="qweight")
    if qweight_tensor.ndim != 2:
        raise ValueError("qweight must have shape [packed_k, n]")
    if qweight_tensor.shape[1] != size_n:
        raise ValueError("qweight N dimension mismatch")
    required_packed_k = (size_k + GPTQ_PACK_FACTOR_INT32 - 1) // GPTQ_PACK_FACTOR_INT32
    if qweight_tensor.shape[0] < required_packed_k:
        raise ValueError("qweight packed K dimension is too small")

    unpacked = torch.empty(
        (required_packed_k * GPTQ_PACK_FACTOR_INT32, size_n),
        dtype=torch.uint8,
    )
    source = qweight_tensor.to(torch.int64)
    for pack_index in range(GPTQ_PACK_FACTOR_INT32):
        unpacked[pack_index::GPTQ_PACK_FACTOR_INT32] = (
            source.bitwise_right_shift(GPTQ_BITS * pack_index)
            .bitwise_and(0x0F)
            .to(torch.uint8)
        )
    return unpacked[:size_k].contiguous()


def unpack_gptq_int4_qzeros(
    qzeros: Any,
    *,
    num_groups: int,
    size_n: int,
) -> torch.Tensor:
    _require_positive_int("num_groups", num_groups)
    _require_positive_int("size_n", size_n)
    qzeros_tensor = _as_cpu_int_tensor(qzeros, name="qzeros")
    if qzeros_tensor.ndim != 2:
        raise ValueError("qzeros must have shape [groups, packed_n]")
    if qzeros_tensor.shape[0] < num_groups:
        raise ValueError("qzeros group dimension is too small")
    required_packed_n = (size_n + GPTQ_PACK_FACTOR_INT32 - 1) // GPTQ_PACK_FACTOR_INT32
    if qzeros_tensor.shape[1] < required_packed_n:
        raise ValueError("qzeros packed N dimension is too small")

    unpacked = torch.empty(
        (qzeros_tensor.shape[0], required_packed_n * GPTQ_PACK_FACTOR_INT32),
        dtype=torch.uint8,
    )
    source = qzeros_tensor.to(torch.int64)
    for pack_index in range(GPTQ_PACK_FACTOR_INT32):
        unpacked[:, pack_index::GPTQ_PACK_FACTOR_INT32] = (
            source.bitwise_right_shift(GPTQ_BITS * pack_index)
            .bitwise_and(0x0F)
            .to(torch.uint8)
        )
    return unpacked[:num_groups, :size_n].contiguous()


def pack_gptq_int4_qweight(values):
    values_tensor = _as_cpu_int_tensor(values, name="values")
    if values_tensor.ndim != 2:
        raise ValueError("values must have shape [k, n]")
    if (values_tensor < 0).any() or (values_tensor > 15).any():
        raise ValueError("values must be uint4 values in [0, 15]")
    size_k, size_n = int(values_tensor.shape[0]), int(values_tensor.shape[1])
    packed_k = (size_k + GPTQ_PACK_FACTOR_INT32 - 1) // GPTQ_PACK_FACTOR_INT32
    padded = torch.zeros(
        (packed_k * GPTQ_PACK_FACTOR_INT32, size_n),
        dtype=torch.int64,
    )
    padded[:size_k] = values_tensor.to(torch.int64)
    packed = torch.zeros((packed_k, size_n), dtype=torch.int32)
    for pack_index in range(GPTQ_PACK_FACTOR_INT32):
        packed.bitwise_or_(
            padded[pack_index::GPTQ_PACK_FACTOR_INT32].to(torch.int32)
            .bitwise_left_shift(GPTQ_BITS * pack_index)
        )
    return packed.contiguous()


def pack_gptq_int4_qzeros(values):
    values_tensor = _as_cpu_int_tensor(values, name="values")
    if values_tensor.ndim != 2:
        raise ValueError("values must have shape [groups, n]")
    if (values_tensor < 0).any() or (values_tensor > 15).any():
        raise ValueError("values must be uint4 values in [0, 15]")
    num_groups, size_n = int(values_tensor.shape[0]), int(values_tensor.shape[1])
    packed_n = (size_n + GPTQ_PACK_FACTOR_INT32 - 1) // GPTQ_PACK_FACTOR_INT32
    padded = torch.zeros(
        (num_groups, packed_n * GPTQ_PACK_FACTOR_INT32),
        dtype=torch.int64,
    )
    padded[:, :size_n] = values_tensor.to(torch.int64)
    packed = torch.zeros((num_groups, packed_n), dtype=torch.int32)
    for pack_index in range(GPTQ_PACK_FACTOR_INT32):
        packed.bitwise_or_(
            padded[:, pack_index::GPTQ_PACK_FACTOR_INT32].to(torch.int32)
            .bitwise_left_shift(GPTQ_BITS * pack_index)
        )
    return packed.contiguous()


def _as_cpu_int_tensor(value: Any, *, name: str) -> torch.Tensor:
    if not hasattr(value, "detach"):
        raise TypeError(f"{name} must be a torch.Tensor")
    tensor = value.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if not tensor.dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"{name} must be an integer tensor")
    return tensor.contiguous()


def _as_cpu_float32_tensor(value: Any, *, name: str) -> torch.Tensor:
    if not hasattr(value, "detach"):
        raise TypeError(f"{name} must be a torch.Tensor")
    tensor = value.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.to(dtype=torch.float32).contiguous()
