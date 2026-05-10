"""GPTQ 重量化——训练后将 FP32 master 重新量化为 Int4，写回 CPU GPTQ 缓存。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable, Mapping

import torch

from cfie_training.training_base.gptq_cache_store import GptqCacheStore
from cfie_training.training_base.gptq_checkpoint import (
    pack_gptq_int4_qweight,
    pack_gptq_int4_qzeros,
)
from cfie_training.training_base.gptq_marlin_bundle import (
    DecodedGptqMarlinBundle,
    decode_gptq_marlin_bundle,
    encode_gptq_marlin_bundle_sections,
)

INT4_SIGNED_MIN = -8
INT4_SIGNED_MAX = 7
DEFAULT_GPTQ_GROUP_SIZE = 128
LAYOUT_SCHEMA_VERSION = 1


def _require_non_empty_string(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


@dataclass(frozen=True, slots=True)
class SymmetricInt4GptqLayout:
    group_size: int = DEFAULT_GPTQ_GROUP_SIZE
    scale_dtype: str = "float32"

    def __post_init__(self) -> None:
        _require_positive_int("group_size", self.group_size)
        _require_non_empty_string("scale_dtype", self.scale_dtype)
        if self.scale_dtype != "float32":
            raise ValueError("only float32 scale dtype is supported")

    def num_groups(self, num_elements: int) -> int:
        _require_non_negative_int("num_elements", num_elements)
        if num_elements == 0:
            return 0
        return (num_elements + self.group_size - 1) // self.group_size

    def packed_weight_bytes(self, num_elements: int) -> int:
        _require_non_negative_int("num_elements", num_elements)
        return (num_elements + 1) // 2

    def payload_num_bytes(self, num_elements: int) -> int:
        return self.packed_weight_bytes(num_elements) + self.num_groups(
            num_elements
        ) * 4

    @property
    def layout_hash(self) -> str:
        payload = {
            "format": "symmetric_int4_placeholder",
            "group_size": self.group_size,
            "scale_dtype": self.scale_dtype,
            "schema_version": LAYOUT_SCHEMA_VERSION,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class SymmetricInt4GptqCodec:
    layout: SymmetricInt4GptqLayout = field(
        default_factory=SymmetricInt4GptqLayout
    )
    min_scale: float = 1e-12

    def __post_init__(self) -> None:
        if self.min_scale <= 0:
            raise ValueError("min_scale must be > 0")

    def payload_num_bytes(self, num_elements: int) -> int:
        return self.layout.payload_num_bytes(num_elements)

    @property
    def layout_hash(self) -> str:
        return self.layout.layout_hash

    def encode(self, tensor: Any) -> bytes:
        values = _as_cpu_float32_vector(tensor, name="tensor")
        if not torch.isfinite(values).all():
            raise ValueError("tensor must contain only finite values")
        num_elements = values.numel()
        if num_elements == 0:
            return b""

        quantized = torch.empty(num_elements, dtype=torch.uint8)
        scales = torch.empty(self.layout.num_groups(num_elements),
                             dtype=torch.float32)
        for group_index, start in enumerate(
            range(0, num_elements, self.layout.group_size)
        ):
            end = min(start + self.layout.group_size, num_elements)
            group = values[start:end]
            max_abs = float(group.abs().max().item())
            scale = 1.0 if max_abs == 0 else max(max_abs / INT4_SIGNED_MAX,
                                                  self.min_scale)
            scales[group_index] = scale
            signed = torch.round(group / scale).clamp(
                min=INT4_SIGNED_MIN,
                max=INT4_SIGNED_MAX,
            )
            quantized[start:end] = signed.to(torch.int16).add(8).to(torch.uint8)

        packed = _pack_uint4(quantized)
        return packed.numpy().tobytes() + scales.numpy().tobytes()

    def decode(
        self,
        payload: bytes | bytearray | memoryview,
        num_elements: int,
    ) -> torch.Tensor:
        _require_non_negative_int("num_elements", num_elements)
        payload_bytes = bytes(payload)
        expected_bytes = self.payload_num_bytes(num_elements)
        if len(payload_bytes) != expected_bytes:
            raise ValueError(
                f"GPTQ bundle expected {expected_bytes} bytes, "
                f"got {len(payload_bytes)}"
            )
        if num_elements == 0:
            return torch.empty(0, dtype=torch.float32)

        packed_size = self.layout.packed_weight_bytes(num_elements)
        packed = torch.frombuffer(
            bytearray(payload_bytes[:packed_size]),
            dtype=torch.uint8,
        )
        scales = torch.frombuffer(
            bytearray(payload_bytes[packed_size:]),
            dtype=torch.float32,
        ).clone()
        quantized = _unpack_uint4(packed, num_elements).to(torch.int16) - 8
        expanded_scales = scales.repeat_interleave(
            self.layout.group_size
        )[:num_elements]
        return quantized.to(torch.float32).mul(expanded_scales).contiguous()


@dataclass(slots=True)
class GptqCacheRequantizer:
    store: GptqCacheStore
    param_to_bundle: Mapping[str, str]
    codec: SymmetricInt4GptqCodec = field(
        default_factory=SymmetricInt4GptqCodec
    )
    require_mapping_for_touched: bool = False

    def requantize_touched(
        self,
        masters: Mapping[str, Any],
        touched_param_ids: Iterable[str],
    ) -> dict[str, bytes]:
        updates: dict[str, bytes] = {}
        for param_id in touched_param_ids:
            _require_non_empty_string("param_id", param_id)
            bundle_id = self.param_to_bundle.get(param_id)
            if bundle_id is None:
                if self.require_mapping_for_touched:
                    raise KeyError(f"missing GPTQ bundle mapping for {param_id!r}")
                continue
            if bundle_id in updates:
                raise ValueError(f"duplicate GPTQ cache bundle update {bundle_id!r}")
            try:
                master = masters[param_id]
            except KeyError as exc:
                raise KeyError(f"missing FP32 master for {param_id!r}") from exc
            payload = self.codec.encode(master)
            record = self.store.records[bundle_id]
            if record.num_bytes != len(payload):
                raise ValueError(
                    f"bundle {bundle_id} expected {record.num_bytes} bytes, "
                    f"requantized payload has {len(payload)} bytes"
                )
            if (
                record.quant_layout_hash
                and record.quant_layout_hash != self.codec.layout_hash
            ):
                raise ValueError(
                    f"bundle {bundle_id} quant layout mismatch: "
                    f"{record.quant_layout_hash} != {self.codec.layout_hash}"
                )
            updates[bundle_id] = payload
        return updates


@dataclass(slots=True)
class GptqMarlinBundleRequantizer:
    store: GptqCacheStore
    param_to_bundle: Mapping[str, str]
    min_scale: float = 1e-12
    require_mapping_for_touched: bool = False

    def __post_init__(self) -> None:
        if self.min_scale <= 0:
            raise ValueError("min_scale must be > 0")

    def requantize_touched(
        self,
        masters: Mapping[str, Any],
        touched_param_ids: Iterable[str],
    ) -> dict[str, bytes]:
        updates: dict[str, bytes] = {}
        for param_id in touched_param_ids:
            _require_non_empty_string("param_id", param_id)
            bundle_id = self.param_to_bundle.get(param_id)
            if bundle_id is None:
                if self.require_mapping_for_touched:
                    raise KeyError(f"missing GPTQ bundle mapping for {param_id!r}")
                continue
            if bundle_id in updates:
                raise ValueError(f"duplicate GPTQ cache bundle update {bundle_id!r}")
            try:
                master = masters[param_id]
            except KeyError as exc:
                raise KeyError(f"missing FP32 master for {param_id!r}") from exc
            record = self.store.records[bundle_id]
            existing = decode_gptq_marlin_bundle(self.store.read_bundle(bundle_id))
            payload = self._encode_from_existing_bundle(
                existing,
                master,
            )
            if len(payload) != record.num_bytes:
                raise ValueError(
                    f"bundle {bundle_id} expected {record.num_bytes} bytes, "
                    f"requantized payload has {len(payload)} bytes"
                )
            decoded = decode_gptq_marlin_bundle(payload)
            if (
                record.quant_layout_hash
                and record.quant_layout_hash != decoded.metadata.layout_hash
            ):
                raise ValueError(
                    f"bundle {bundle_id} quant layout mismatch: "
                    f"{record.quant_layout_hash} != {decoded.metadata.layout_hash}"
                )
            updates[bundle_id] = payload
        return updates

    def _encode_from_existing_bundle(
        self,
        existing: DecodedGptqMarlinBundle,
        master: Any,
    ) -> bytes:
        values = _as_cpu_float32_vector(master, name="master")
        prefixes = _section_prefixes(existing)
        sections: dict[str, torch.Tensor] = {}
        if prefixes:
            expected_numel = existing.metadata.size_k * existing.metadata.size_n
            if values.numel() != expected_numel:
                raise ValueError(
                    "FP32 master size does not match GPTQ bundle metadata"
                )
            cursor = 0
            for prefix in prefixes:
                size_n = _section_size_n(existing, prefix)
                part_numel = existing.metadata.size_k * size_n
                matrix = values[cursor:cursor + part_numel].reshape(
                    size_n,
                    existing.metadata.size_k,
                ).t().contiguous()
                cursor += part_numel
                sections.update(
                    self._quantize_section(
                        matrix,
                        existing=existing,
                        prefix=prefix,
                    )
                )
        else:
            expected_numel = existing.metadata.size_k * existing.metadata.size_n
            if values.numel() != expected_numel:
                raise ValueError(
                    "FP32 master size does not match GPTQ bundle metadata"
                )
            matrix = values.reshape(
                existing.metadata.size_n,
                existing.metadata.size_k,
            ).t().contiguous()
            sections.update(
                self._quantize_section(
                    matrix,
                    existing=existing,
                    prefix=None,
                )
            )
        return encode_gptq_marlin_bundle_sections(
            bundle_id=existing.metadata.bundle_id,
            group_size=existing.metadata.group_size,
            size_k=existing.metadata.size_k,
            size_n=existing.metadata.size_n,
            sections=sections,
            bits=existing.metadata.bits,
            act_order=existing.metadata.act_order,
        )

    def _quantize_section(
        self,
        matrix: torch.Tensor,
        *,
        existing: DecodedGptqMarlinBundle,
        prefix: str | None,
    ) -> dict[str, torch.Tensor]:
        size_k, size_n = int(matrix.shape[0]), int(matrix.shape[1])
        section_names = _section_names(prefix)
        group_ids = _existing_or_default_g_idx(
            existing,
            section_names["g_idx"],
            size_k=size_k,
            group_size=existing.metadata.group_size,
        )
        quantized, scales = _quantize_matrix_int4(
            matrix,
            group_ids=group_ids,
            min_scale=self.min_scale,
        )
        sections: dict[str, torch.Tensor] = {
            section_names["qweight"]: pack_gptq_int4_qweight(quantized),
            section_names["scales"]: scales.to(
                dtype=_existing_section_dtype(
                    existing,
                    section_names["scales"],
                    default=torch.float32,
                )
            ),
        }
        if section_names["qzeros"] in existing.tensors:
            num_groups = int(group_ids.max().item()) + 1
            qzeros = torch.full(
                (num_groups, size_n),
                7,
                dtype=torch.int32,
            )
            sections[section_names["qzeros"]] = pack_gptq_int4_qzeros(qzeros)
        if section_names["g_idx"] in existing.tensors:
            sections[section_names["g_idx"]] = existing.tensors[
                section_names["g_idx"]
            ].to(dtype=_existing_section_dtype(
                existing,
                section_names["g_idx"],
                default=torch.int32,
            ))
        if section_names["perm"] in existing.tensors:
            sections[section_names["perm"]] = existing.tensors[section_names["perm"]]
        return sections


def gptq_bundle_num_bytes(
    num_elements: int,
    *,
    group_size: int = DEFAULT_GPTQ_GROUP_SIZE,
) -> int:
    return SymmetricInt4GptqLayout(
        group_size=group_size
    ).payload_num_bytes(num_elements)


def gptq_layout_hash(
    *,
    group_size: int = DEFAULT_GPTQ_GROUP_SIZE,
) -> str:
    return SymmetricInt4GptqLayout(group_size=group_size).layout_hash


def _pack_uint4(values: torch.Tensor) -> torch.Tensor:
    if values.dtype != torch.uint8:
        raise TypeError("values must be torch.uint8")
    padded = values
    if values.numel() % 2:
        padded = torch.cat([values, torch.zeros(1, dtype=torch.uint8)])
    low = padded[0::2]
    high = padded[1::2]
    return low.bitwise_or(high.bitwise_left_shift(4)).contiguous()


def _unpack_uint4(packed: torch.Tensor, num_elements: int) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise TypeError("packed must be torch.uint8")
    _require_non_negative_int("num_elements", num_elements)
    low = packed.bitwise_and(0x0F)
    high = packed.bitwise_right_shift(4).bitwise_and(0x0F)
    unpacked = torch.empty(packed.numel() * 2, dtype=torch.uint8)
    unpacked[0::2] = low
    unpacked[1::2] = high
    return unpacked[:num_elements].contiguous()


def _quantize_matrix_int4(
    matrix: torch.Tensor,
    *,
    group_ids: torch.Tensor,
    min_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if matrix.ndim != 2:
        raise ValueError("matrix must have shape [K, N]")
    if group_ids.numel() != matrix.shape[0]:
        raise ValueError("group_ids length must match K")
    if (group_ids < 0).any():
        raise ValueError("group_ids must be non-negative")

    num_groups = int(group_ids.max().item()) + 1
    size_n = int(matrix.shape[1])
    scales = torch.empty((num_groups, size_n), dtype=torch.float32)
    quantized = torch.empty_like(matrix, dtype=torch.int32)
    for group_id in range(num_groups):
        rows = group_ids == group_id
        if not rows.any():
            scales[group_id] = 1.0
            continue
        group = matrix[rows].to(torch.float32)
        max_abs = group.abs().amax(dim=0)
        scale = torch.where(
            max_abs == 0,
            torch.ones_like(max_abs),
            torch.clamp(max_abs / INT4_SIGNED_MAX, min=min_scale),
        )
        scales[group_id] = scale
        signed = torch.round(group / scale).clamp(
            min=INT4_SIGNED_MIN,
            max=INT4_SIGNED_MAX,
        )
        quantized[rows] = signed.to(torch.int32).add(8)
    return quantized.contiguous(), scales.contiguous()


def _section_prefixes(bundle: DecodedGptqMarlinBundle) -> tuple[str, ...]:
    prefixes = {
        name.split(".", 1)[0]
        for name in bundle.tensors
        if "." in name and name.endswith(".qweight")
    }
    return tuple(sorted(prefixes))


def _section_size_n(bundle: DecodedGptqMarlinBundle, prefix: str) -> int:
    qweight = bundle.tensors[f"{prefix}.qweight"]
    if qweight.ndim != 2:
        raise ValueError(f"{prefix}.qweight must have shape [packed_k, N]")
    return int(qweight.shape[1])


def _section_names(prefix: str | None) -> dict[str, str]:
    if prefix is None:
        return {
            "qweight": "qweight",
            "scales": "scales",
            "qzeros": "qzeros",
            "g_idx": "g_idx",
            "perm": "perm",
        }
    return {
        "qweight": f"{prefix}.qweight",
        "scales": f"{prefix}.scales",
        "qzeros": f"{prefix}.qzeros",
        "g_idx": f"{prefix}.g_idx",
        "perm": f"{prefix}.perm",
    }


def _existing_or_default_g_idx(
    bundle: DecodedGptqMarlinBundle,
    section_name: str,
    *,
    size_k: int,
    group_size: int,
) -> torch.Tensor:
    existing = bundle.tensors.get(section_name)
    if existing is not None:
        g_idx = existing.reshape(-1).to(torch.int64)
        if g_idx.numel() != size_k:
            raise ValueError(f"{section_name} length must match K")
        return g_idx
    return torch.arange(size_k, dtype=torch.int64) // group_size


def _existing_section_dtype(
    bundle: DecodedGptqMarlinBundle,
    section_name: str,
    *,
    default: torch.dtype,
) -> torch.dtype:
    existing = bundle.tensors.get(section_name)
    return default if existing is None else existing.dtype


def _as_cpu_float32_vector(value: Any, *, name: str) -> torch.Tensor:
    if isinstance(value, (bytes, bytearray, memoryview)):
        tensor = torch.frombuffer(bytearray(value), dtype=torch.float32)
        return tensor.clone().contiguous()
    if not hasattr(value, "detach"):
        raise TypeError(f"{name} must be a torch.Tensor or FP32 bytes")
    tensor = value.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.to(dtype=torch.float32).reshape(-1).contiguous()
