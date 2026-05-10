"""GPTQ 专家缓存分片存储——NVMe 上管理冷专家的 Int4 权重和 scales（设计文档 Section 7.1）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MANIFEST_FILENAME = "gptq_cache_manifest.json"


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def _require_non_empty_string(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class GptqCacheRecord:
    """GPTQ 缓存束在 NVMe shard 文件中的位置记录。"""
    bundle_id: str               # bundle 唯一标识（如 "gptq_L0_E7"）
    shard_name: str              # shard 文件名
    offset_bytes: int            # 字节偏移
    num_bytes: int               # 字节长度
    quant_layout_hash: str = ""  # 量化布局哈希（group_size+scale_dtype 指纹）

    def __post_init__(self) -> None:
        _require_non_empty_string("bundle_id", self.bundle_id)
        _require_non_empty_string("shard_name", self.shard_name)
        if Path(self.shard_name).name != self.shard_name: raise ValueError("shard_name must be a plain file name")
        _require_non_negative_int("offset_bytes", self.offset_bytes)
        _require_positive_int("num_bytes", self.num_bytes)

    @property
    def end_bytes(self) -> int: return self.offset_bytes + self.num_bytes  # 末尾字节位置
    def to_dict(self) -> dict[str, Any]: return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GptqCacheRecord":
        return cls(
            bundle_id=str(data["bundle_id"]),
            shard_name=str(data["shard_name"]),
            offset_bytes=int(data["offset_bytes"]),
            num_bytes=int(data["num_bytes"]),
            quant_layout_hash=str(data.get("quant_layout_hash", "")),
        )


@dataclass(slots=True)
class GptqCacheStore:
    root: Path
    records: dict[str, GptqCacheRecord]
    generation: int = 0
    manifest_filename: str = DEFAULT_MANIFEST_FILENAME

    def __init__(
        self,
        root: str | Path,
        records: Mapping[str, GptqCacheRecord] | None = None,
        *,
        generation: int = 0,
        manifest_filename: str = DEFAULT_MANIFEST_FILENAME,
    ) -> None:
        self.root = Path(root)
        self.records = dict(records or {})
        self.generation = generation
        self.manifest_filename = manifest_filename
        _require_non_negative_int("generation", self.generation)

    @property
    def manifest_path(self) -> Path:
        return self.root / self.manifest_filename

    @classmethod
    def create(
        cls,
        root: str | Path,
        records: Mapping[str, GptqCacheRecord],
        *,
        generation: int = 0,
        manifest_filename: str = DEFAULT_MANIFEST_FILENAME,
    ) -> "GptqCacheStore":
        store = cls(
            root,
            records,
            generation=generation,
            manifest_filename=manifest_filename,
        )
        store.write_manifest()
        return store

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        manifest_filename: str = DEFAULT_MANIFEST_FILENAME,
    ) -> "GptqCacheStore":
        store_root = Path(root)
        manifest_path = store_root / manifest_filename
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid GPTQ cache manifest: {manifest_path}") from exc
        if not isinstance(manifest, Mapping):
            raise ValueError(f"GPTQ cache manifest must be an object: {manifest_path}")
        schema_version = int(manifest.get("schema_version", 0))
        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported GPTQ cache manifest schema {schema_version}")
        records_data = manifest.get("records", [])
        if not isinstance(records_data, list):
            raise ValueError("GPTQ cache manifest records must be a list")
        records = {
            record.bundle_id: record
            for record in (
                GptqCacheRecord.from_dict(item)
                for item in records_data
            )
        }
        return cls(
            store_root,
            records,
            generation=int(manifest.get("generation", 0)),
            manifest_filename=manifest_filename,
        )

    def write_manifest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generation": self.generation,
            "records": [
                self.records[bundle_id].to_dict()
                for bundle_id in sorted(self.records)
            ],
        }
        payload = json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        _atomic_write_text(self.manifest_path, payload + "\n")

    def read_bundle(self, bundle_id: str) -> bytes:
        record = self._record_for(bundle_id)
        shard_path = self._shard_path(record.shard_name)
        if not shard_path.exists():
            raise FileNotFoundError(f"missing GPTQ cache shard file: {shard_path}")
        with shard_path.open("rb") as stream:
            stream.seek(record.offset_bytes)
            payload = stream.read(record.num_bytes)
        if len(payload) != record.num_bytes:
            raise ValueError(
                f"bundle {bundle_id} expected {record.num_bytes} bytes, "
                f"got {len(payload)}"
            )
        return payload

    def flush_touched(
        self,
        updates: Mapping[str, Any],
        *,
        generation: int,
    ) -> int:
        _require_non_negative_int("generation", generation)
        if generation < self.generation:
            raise ValueError(
                f"generation must be >= current generation {self.generation}"
            )
        if not updates:
            self.generation = generation
            self.write_manifest()
            return 0

        normalized_updates = {
            bundle_id: self._payload_for(bundle_id, payload)
            for bundle_id, payload in updates.items()
        }
        updates_by_shard: dict[str, dict[GptqCacheRecord, bytes]] = {}
        for bundle_id, payload in normalized_updates.items():
            record = self._record_for(bundle_id)
            updates_by_shard.setdefault(record.shard_name, {})[record] = payload

        for shard_name, shard_updates in updates_by_shard.items():
            self._patch_shard_atomically(shard_name, shard_updates)

        self.generation = generation
        self.write_manifest()
        return len(normalized_updates)

    def shard_size_bytes(self, shard_name: str) -> int:
        size_bytes = 0
        for record in self.records.values():
            if record.shard_name == shard_name:
                size_bytes = max(size_bytes, record.end_bytes)
        if size_bytes == 0:
            raise KeyError(f"unknown shard {shard_name!r}")
        return size_bytes

    def _record_for(self, bundle_id: str) -> GptqCacheRecord:
        try:
            return self.records[bundle_id]
        except KeyError as exc:
            raise KeyError(f"unknown GPTQ cache bundle {bundle_id!r}") from exc

    def _payload_for(self, bundle_id: str, payload: Any) -> bytes:
        record = self._record_for(bundle_id)
        payload_bytes = _payload_to_bytes(payload)
        if len(payload_bytes) != record.num_bytes:
            raise ValueError(
                f"bundle {bundle_id} expected {record.num_bytes} bytes, "
                f"got {len(payload_bytes)}"
            )
        return payload_bytes

    def _shard_path(self, shard_name: str) -> Path:
        return self.root / shard_name

    def _patch_shard_atomically(
        self,
        shard_name: str,
        updates: Mapping[GptqCacheRecord, bytes],
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        shard_path = self._shard_path(shard_name)
        shard_size = self.shard_size_bytes(shard_name)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{shard_name}.",
            suffix=".tmp",
            dir=self.root,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w+b") as output_stream:
                if shard_path.exists():
                    with shard_path.open("rb") as input_stream:
                        shutil.copyfileobj(input_stream, output_stream)
                current_size = output_stream.tell()
                if current_size < shard_size:
                    output_stream.write(b"\0" * (shard_size - current_size))
                for record, payload in updates.items():
                    output_stream.seek(record.offset_bytes)
                    output_stream.write(payload)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            os.replace(tmp_path, shard_path)
            _fsync_directory_best_effort(self.root)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


def _payload_to_bytes(payload: Any) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, memoryview):
        return payload.tobytes()
    if hasattr(payload, "detach") and hasattr(payload, "dtype"):
        tensor = payload.detach()
        if hasattr(tensor, "is_cuda") and tensor.is_cuda:
            tensor = tensor.cpu()
        if hasattr(tensor, "is_contiguous") and not tensor.is_contiguous():
            tensor = tensor.contiguous()
        return tensor.numpy().tobytes()
    if hasattr(payload, "tobytes"):
        return payload.tobytes()
    raise TypeError(f"unsupported GPTQ cache payload type {type(payload)!r}")


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
        _fsync_directory_best_effort(path.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _fsync_directory_best_effort(directory: Path) -> None:
    if os.name == "nt":
        return
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        fd = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
