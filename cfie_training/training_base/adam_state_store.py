"""CPU Adam FP8 状态分片存储——训练窗口提交时持久化 m/v 到 NVMe（设计文档 Section 7.1）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MANIFEST_FILENAME = "adam_state_manifest.json"


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def _require_non_empty_string(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


# ──────────────── AdamStateShardRecord — Adam 状态分片记录 ────────────────

@dataclass(frozen=True, slots=True)
class AdamStateShardRecord:
    """Adam 状态在 NVMe shard 文件中的位置记录。

    每个 (param_id, component) 有独立的记录。component="m" 或 "v"。
    state_key = f"{param_id}:{component}" 用于索引。
    """
    param_id: str          # 参数 ID（如 "layers.0.experts.3.w13_weight"）
    component: str          # "m" 或 "v"
    shard_name: str         # shard 文件名（如 "adam_0000.bin"）
    offset_bytes: int       # 在 shard 文件中的起始字节偏移
    num_bytes: int          # 有效数据长度（字节）

    def __post_init__(self) -> None:
        _require_non_empty_string("param_id", self.param_id)
        _require_non_empty_string("component", self.component)
        _require_non_empty_string("shard_name", self.shard_name)
        if Path(self.shard_name).name != self.shard_name:
            raise ValueError("shard_name must be a plain file name")
        _require_non_negative_int("offset_bytes", self.offset_bytes)
        _require_positive_int("num_bytes", self.num_bytes)

    @property
    def key(self) -> str:
        return state_key(self.param_id, self.component)

    @property
    def end_bytes(self) -> int:
        return self.offset_bytes + self.num_bytes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdamStateShardRecord":
        return cls(
            param_id=str(data["param_id"]),
            component=str(data["component"]),
            shard_name=str(data["shard_name"]),
            offset_bytes=int(data["offset_bytes"]),
            num_bytes=int(data["num_bytes"]),
        )


@dataclass(slots=True)
class CpuAdamFp8StateStore:
    root: Path
    records: dict[str, AdamStateShardRecord]
    generation: int = 0
    manifest_filename: str = DEFAULT_MANIFEST_FILENAME

    def __init__(
        self,
        root: str | Path,
        records: Mapping[str, AdamStateShardRecord] | None = None,
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
        records: Mapping[str, AdamStateShardRecord],
        *,
        generation: int = 0,
        manifest_filename: str = DEFAULT_MANIFEST_FILENAME,
    ) -> "CpuAdamFp8StateStore":
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
    ) -> "CpuAdamFp8StateStore":
        store_root = Path(root)
        manifest_path = store_root / manifest_filename
        try:
            with manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Adam state manifest: {manifest_path}") from exc
        if not isinstance(manifest, Mapping):
            raise ValueError(f"Adam state manifest must be an object: {manifest_path}")
        schema_version = int(manifest.get("schema_version", 0))
        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported Adam state manifest schema {schema_version}")
        records_data = manifest.get("records", [])
        if not isinstance(records_data, list):
            raise ValueError("Adam state manifest records must be a list")
        records = {
            record.key: record
            for record in (
                AdamStateShardRecord.from_dict(item)
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
                self.records[key].to_dict()
                for key in sorted(self.records)
            ],
        }
        payload = json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        _atomic_write_text(self.manifest_path, payload + "\n")

    def read_state(self, param_id: str, component: str) -> bytes:
        record = self._record_for(param_id, component)
        shard_path = self._shard_path(record.shard_name)
        if not shard_path.exists():
            raise FileNotFoundError(f"missing Adam state shard file: {shard_path}")
        with shard_path.open("rb") as stream:
            stream.seek(record.offset_bytes)
            payload = stream.read(record.num_bytes)
        if len(payload) != record.num_bytes:
            raise ValueError(
                f"state {record.key} expected {record.num_bytes} bytes, "
                f"got {len(payload)}"
            )
        return payload

    def flush_touched(
        self,
        updates: Mapping[str, Mapping[str, Any]],
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

        normalized_updates: dict[str, bytes] = {}
        for param_id, component_updates in updates.items():
            for component, payload in component_updates.items():
                key = state_key(param_id, component)
                normalized_updates[key] = self._payload_for(
                    param_id,
                    component,
                    payload,
                )

        updates_by_shard: dict[str, dict[AdamStateShardRecord, bytes]] = {}
        for key, payload in normalized_updates.items():
            record = self._record_by_key(key)
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

    def _record_for(self, param_id: str, component: str) -> AdamStateShardRecord:
        return self._record_by_key(state_key(param_id, component))

    def _record_by_key(self, key: str) -> AdamStateShardRecord:
        try:
            return self.records[key]
        except KeyError as exc:
            raise KeyError(f"unknown Adam state {key!r}") from exc

    def _payload_for(self, param_id: str, component: str, payload: Any) -> bytes:
        record = self._record_for(param_id, component)
        payload_bytes = _payload_to_bytes(payload)
        if len(payload_bytes) != record.num_bytes:
            raise ValueError(
                f"state {record.key} expected {record.num_bytes} bytes, "
                f"got {len(payload_bytes)}"
            )
        return payload_bytes

    def _shard_path(self, shard_name: str) -> Path:
        return self.root / shard_name

    def _patch_shard_atomically(
        self,
        shard_name: str,
        updates: Mapping[AdamStateShardRecord, bytes],
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


def state_key(param_id: str, component: str) -> str:
    _require_non_empty_string("param_id", param_id)
    _require_non_empty_string("component", component)
    return f"{param_id}:{component}"


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
    raise TypeError(f"unsupported Adam state payload type {type(payload)!r}")


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
