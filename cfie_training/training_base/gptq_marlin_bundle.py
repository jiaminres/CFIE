"""GPTQ/Marlin 缓存束——编码/解码 Marlin-packed Int4 权重和元数据。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import struct
from typing import Any, Mapping

import torch

BUNDLE_MAGIC = b"CFIEGPTQ"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_HEADER_PREFIX = struct.Struct("<8sI")


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
class GptqMarlinBundleSection:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset_bytes: int
    num_bytes: int

    def __post_init__(self) -> None:
        _require_non_empty_string("name", self.name)
        _require_non_empty_string("dtype", self.dtype)
        if not self.shape:
            raise ValueError("shape must not be empty")
        for dim in self.shape:
            _require_positive_int("shape dim", dim)
        _require_non_negative_int("offset_bytes", self.offset_bytes)
        _require_non_negative_int("num_bytes", self.num_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "offset_bytes": self.offset_bytes,
            "num_bytes": self.num_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GptqMarlinBundleSection":
        return cls(
            name=str(payload["name"]),
            dtype=str(payload["dtype"]),
            shape=tuple(int(dim) for dim in payload["shape"]),
            offset_bytes=int(payload["offset_bytes"]),
            num_bytes=int(payload["num_bytes"]),
        )


@dataclass(frozen=True, slots=True)
class GptqMarlinBundleMetadata:
    bundle_id: str
    group_size: int
    size_k: int
    size_n: int
    bits: int = 4
    has_zero_points: bool = False
    has_g_idx: bool = False
    has_perm: bool = False
    act_order: bool = False
    sections: tuple[GptqMarlinBundleSection, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty_string("bundle_id", self.bundle_id)
        _require_positive_int("group_size", self.group_size)
        _require_positive_int("size_k", self.size_k)
        _require_positive_int("size_n", self.size_n)
        _require_positive_int("bits", self.bits)
        names = [section.name for section in self.sections]
        if len(names) != len(set(names)):
            raise ValueError("bundle sections must have unique names")

    @property
    def layout_hash(self) -> str:
        payload = {
            "act_order": self.act_order,
            "bits": self.bits,
            "format": "cfie_gptq_marlin_bundle",
            "group_size": self.group_size,
            "has_g_idx": self.has_g_idx,
            "has_perm": self.has_perm,
            "has_zero_points": self.has_zero_points,
            "schema_version": BUNDLE_SCHEMA_VERSION,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    @property
    def payload_num_bytes(self) -> int:
        data_bytes = max(
            (
                section.offset_bytes + section.num_bytes
                for section in self.sections
            ),
            default=0,
        )
        return BUNDLE_HEADER_PREFIX.size + len(self._header_bytes()) + data_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_id": self.bundle_id,
            "format": "cfie_gptq_marlin_bundle",
            "group_size": self.group_size,
            "size_k": self.size_k,
            "size_n": self.size_n,
            "bits": self.bits,
            "has_zero_points": self.has_zero_points,
            "has_g_idx": self.has_g_idx,
            "has_perm": self.has_perm,
            "act_order": self.act_order,
            "layout_hash": self.layout_hash,
            "sections": [section.to_dict() for section in self.sections],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GptqMarlinBundleMetadata":
        schema_version = int(payload.get("schema_version", 0))
        if schema_version != BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported GPTQ bundle schema {schema_version}")
        if payload.get("format") != "cfie_gptq_marlin_bundle":
            raise ValueError("unsupported GPTQ bundle format")
        return cls(
            bundle_id=str(payload["bundle_id"]),
            group_size=int(payload["group_size"]),
            size_k=int(payload["size_k"]),
            size_n=int(payload["size_n"]),
            bits=int(payload.get("bits", 4)),
            has_zero_points=bool(payload.get("has_zero_points", False)),
            has_g_idx=bool(payload.get("has_g_idx", False)),
            has_perm=bool(payload.get("has_perm", False)),
            act_order=bool(payload.get("act_order", False)),
            sections=tuple(
                GptqMarlinBundleSection.from_dict(section)
                for section in payload.get("sections", [])
            ),
        )

    def _header_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class GptqMarlinBundleTensors:
    qweight: Any
    scales: Any
    qzeros: Any | None = None
    g_idx: Any | None = None
    perm: Any | None = None


@dataclass(frozen=True, slots=True)
class DecodedGptqMarlinBundle:
    metadata: GptqMarlinBundleMetadata
    tensors: dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class GptqMarlinBundleCodec:
    bundle_id: str
    group_size: int
    size_k: int
    size_n: int
    bits: int = 4
    act_order: bool = False

    def __post_init__(self) -> None:
        _require_non_empty_string("bundle_id", self.bundle_id)
        _require_positive_int("group_size", self.group_size)
        _require_positive_int("size_k", self.size_k)
        _require_positive_int("size_n", self.size_n)
        _require_positive_int("bits", self.bits)

    def encode(self, tensors: GptqMarlinBundleTensors) -> bytes:
        return encode_gptq_marlin_bundle_sections(
            bundle_id=self.bundle_id,
            group_size=self.group_size,
            size_k=self.size_k,
            size_n=self.size_n,
            sections={
                "qweight": tensors.qweight,
                "scales": tensors.scales,
                "qzeros": tensors.qzeros,
                "g_idx": tensors.g_idx,
                "perm": tensors.perm,
            },
            bits=self.bits,
            act_order=self.act_order,
        )

    def decode(
        self,
        payload: bytes | bytearray | memoryview,
    ) -> DecodedGptqMarlinBundle:
        return decode_gptq_marlin_bundle(payload)

    @property
    def layout_hash(self) -> str:
        return GptqMarlinBundleMetadata(
            bundle_id=self.bundle_id,
            group_size=self.group_size,
            size_k=self.size_k,
            size_n=self.size_n,
            bits=self.bits,
            act_order=self.act_order,
        ).layout_hash


def decode_gptq_marlin_bundle(
    payload: bytes | bytearray | memoryview,
) -> DecodedGptqMarlinBundle:
    payload_bytes = bytes(payload)
    if len(payload_bytes) < BUNDLE_HEADER_PREFIX.size:
        raise ValueError("GPTQ bundle payload is too short")
    magic, header_size = BUNDLE_HEADER_PREFIX.unpack_from(payload_bytes)
    if magic != BUNDLE_MAGIC:
        raise ValueError("invalid GPTQ bundle magic")
    header_start = BUNDLE_HEADER_PREFIX.size
    header_end = header_start + header_size
    if len(payload_bytes) < header_end:
        raise ValueError("GPTQ bundle header is truncated")
    try:
        header = json.loads(payload_bytes[header_start:header_end].decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid GPTQ bundle header JSON") from exc
    metadata = GptqMarlinBundleMetadata.from_dict(header)
    tensors: dict[str, torch.Tensor] = {}
    data_start = header_end
    for section in metadata.sections:
        start = data_start + section.offset_bytes
        end = start + section.num_bytes
        if end > len(payload_bytes):
            raise ValueError(f"GPTQ bundle section {section.name!r} is truncated")
        dtype = _torch_dtype(section.dtype)
        expected_bytes = (
            _numel(section.shape)
            * torch.empty((), dtype=dtype).element_size()
        )
        if expected_bytes != section.num_bytes:
            raise ValueError(f"GPTQ bundle section {section.name!r} size mismatch")
        raw = bytearray(payload_bytes[start:end])
        tensors[section.name] = torch.frombuffer(raw, dtype=dtype).clone().reshape(
            section.shape
        )
    return DecodedGptqMarlinBundle(metadata=metadata, tensors=tensors)


def encode_gptq_marlin_bundle_sections(
    *,
    bundle_id: str,
    group_size: int,
    size_k: int,
    size_n: int,
    sections: Mapping[str, Any],
    bits: int = 4,
    act_order: bool = False,
) -> bytes:
    sections_payload: list[tuple[GptqMarlinBundleSection, bytes]] = []
    offset_bytes = 0
    for name, tensor_like in sections.items():
        _require_non_empty_string("section name", name)
        if tensor_like is None:
            continue
        tensor = _as_cpu_contiguous_tensor(tensor_like, name=name)
        payload = tensor.numpy().tobytes()
        section = GptqMarlinBundleSection(
            name=name,
            dtype=_dtype_name(tensor.dtype),
            shape=tuple(int(dim) for dim in tensor.shape),
            offset_bytes=offset_bytes,
            num_bytes=len(payload),
        )
        sections_payload.append((section, payload))
        offset_bytes += len(payload)

    metadata = GptqMarlinBundleMetadata(
        bundle_id=bundle_id,
        group_size=group_size,
        size_k=size_k,
        size_n=size_n,
        bits=bits,
        has_zero_points=any(
            section.name == "qzeros" or section.name.endswith(".qzeros")
            for section, _ in sections_payload
        ),
        has_g_idx=any(
            section.name == "g_idx" or section.name.endswith(".g_idx")
            for section, _ in sections_payload
        ),
        has_perm=any(
            section.name == "perm" or section.name.endswith(".perm")
            for section, _ in sections_payload
        ),
        act_order=act_order,
        sections=tuple(section for section, _ in sections_payload),
    )
    header = metadata._header_bytes()
    return (
        BUNDLE_HEADER_PREFIX.pack(BUNDLE_MAGIC, len(header))
        + header
        + b"".join(payload for _, payload in sections_payload)
    )


def gptq_marlin_bundle_layout_hash(
    *,
    group_size: int,
    size_k: int,
    size_n: int,
    bits: int = 4,
    act_order: bool = False,
    has_zero_points: bool = False,
    has_g_idx: bool = False,
    has_perm: bool = False,
    bundle_id: str = "layout",
) -> str:
    return GptqMarlinBundleMetadata(
        bundle_id=bundle_id,
        group_size=group_size,
        size_k=size_k,
        size_n=size_n,
        bits=bits,
        act_order=act_order,
        has_zero_points=has_zero_points,
        has_g_idx=has_g_idx,
        has_perm=has_perm,
    ).layout_hash


def _as_cpu_contiguous_tensor(value: Any, *, name: str) -> torch.Tensor:
    if not hasattr(value, "detach"):
        raise TypeError(f"{name} must be a torch.Tensor")
    tensor = value.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.contiguous()


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.int32:
        return "int32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.int16:
        return "int16"
    if dtype == torch.int64:
        return "int64"
    if dtype == torch.uint8:
        return "uint8"
    raise TypeError(f"unsupported GPTQ bundle tensor dtype {dtype}")


def _torch_dtype(name: str) -> torch.dtype:
    if name == "int32":
        return torch.int32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if name == "int16":
        return torch.int16
    if name == "int64":
        return torch.int64
    if name == "uint8":
        return torch.uint8
    raise ValueError(f"unsupported GPTQ bundle tensor dtype {name!r}")


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total
