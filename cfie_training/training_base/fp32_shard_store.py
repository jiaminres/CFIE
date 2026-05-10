"""FP32 主参数分片存储——NVMe 上按 shard 文件管理全部参数的 FP32 master（设计文档 Section 7.1）。

每个参数通过 ParamShardRecord 记录其所在的 shard 文件名、偏移量和元素数。
flush_touched 使用原子写（临时文件+rename）保证数据安全。
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
import json, os, shutil, tempfile
from pathlib import Path
from typing import Any, Mapping

FP32_BYTES = 4
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MANIFEST_FILENAME = "fp32_manifest.json"

def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0: raise ValueError(f"{name} must be >= 0")
def _require_positive_int(name: str, value: int) -> None:
    if value < 1: raise ValueError(f"{name} must be >= 1")
def _require_non_empty_string(name: str, value: str) -> None:
    if not value.strip(): raise ValueError(f"{name} must be a non-empty string")

# ──────── ParamShardRecord — FP32 参数在 NVMe shard 中的位置记录 ────────
@dataclass(frozen=True, slots=True)
class ParamShardRecord:
    """参数在 NVMe shard 文件中的位置索引。"""
    param_id: str           # 参数唯一标识（如 "layers.0.experts.3.w13_weight"）
    shard_name: str         # shard 文件名（纯文件名，不含路径）
    offset_elements: int    # 在 shard 中的起始元素偏移
    num_elements: int       # 元素数量

    def __post_init__(self) -> None:
        _require_non_empty_string("param_id", self.param_id)
        _require_non_empty_string("shard_name", self.shard_name)
        if Path(self.shard_name).name != self.shard_name:
            raise ValueError("shard_name must be a plain file name")
        _require_non_negative_int("offset_elements", self.offset_elements)
        _require_positive_int("num_elements", self.num_elements)

    @property
    def offset_bytes(self) -> int: return self.offset_elements * FP32_BYTES
    @property
    def num_bytes(self) -> int: return self.num_elements * FP32_BYTES
    @property
    def end_bytes(self) -> int: return self.offset_bytes + self.num_bytes
    def to_dict(self) -> dict[str, Any]: return asdict(self)
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParamShardRecord":
        return cls(param_id=str(data["param_id"]), shard_name=str(data["shard_name"]),
                   offset_elements=int(data["offset_elements"]), num_elements=int(data["num_elements"]))

# ──────── FP32ShardStore — NVMe FP32 主参数分片存储管理器 ────────
@dataclass(slots=True)
class FP32ShardStore:
    """NVMe 上的 FP32 主参数分片存储。"""
    root: Path
    records: dict[str, ParamShardRecord]   # {param_id: 记录}
    generation: int = 0
    manifest_filename: str = DEFAULT_MANIFEST_FILENAME

    def __init__(self, root: str | Path, records: Mapping[str, ParamShardRecord] | None = None,
                 *, generation: int = 0, manifest_filename: str = DEFAULT_MANIFEST_FILENAME) -> None:
        self.root = Path(root); self.records = dict(records or {})
        self.generation = generation; self.manifest_filename = manifest_filename
        _require_non_negative_int("generation", self.generation)

    @property
    def manifest_path(self) -> Path: return self.root / self.manifest_filename

    # ── create / load ──
    @classmethod
    def create(cls, root: str | Path, records: Mapping[str, ParamShardRecord],
               *, generation: int = 0, manifest_filename: str = DEFAULT_MANIFEST_FILENAME) -> "FP32ShardStore":
        store = cls(root, records, generation=generation, manifest_filename=manifest_filename)
        store.write_manifest()
        return store

    @classmethod
    def load(cls, root: str | Path, *, manifest_filename: str = DEFAULT_MANIFEST_FILENAME) -> "FP32ShardStore":
        store_root = Path(root); manifest_path = store_root / manifest_filename
        with manifest_path.open("r", encoding="utf-8") as stream: manifest = json.load(stream)
        if not isinstance(manifest, Mapping): raise ValueError(f"invalid FP32 manifest")
        schema_version = int(manifest.get("schema_version", 0))
        if schema_version != MANIFEST_SCHEMA_VERSION: raise ValueError(f"unsupported schema {schema_version}")
        records = {r.param_id: r for r in (ParamShardRecord.from_dict(d) for d in manifest.get("records", []))}
        return cls(store_root, records, generation=int(manifest.get("generation", 0)),
                    manifest_filename=manifest_filename)

    def write_manifest(self) -> None:
        """将 records 写为 manifest JSON 文件（原子操作）。"""
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = {"schema_version": MANIFEST_SCHEMA_VERSION, "generation": self.generation,
                    "records": [self.records[k].to_dict() for k in sorted(self.records)]}
        _atomic_write_text(self.manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    # ── 读参数 ──
    def read_param(self, param_id: str) -> bytes:
        """从 NVMe shard 文件读取单个 FP32 参数 → bytes。"""
        record = self._record_for(param_id)
        shard_path = self._shard_path(record.shard_name)
        if not shard_path.exists(): raise FileNotFoundError(f"missing FP32 shard: {shard_path}")
        with shard_path.open("rb") as stream:
            stream.seek(record.offset_bytes)
            payload = stream.read(record.num_bytes)
        if len(payload) != record.num_bytes: raise ValueError(f"expected {record.num_bytes} bytes, got {len(payload)}")
        return payload

    # ── 写参数（原子 patch）──
    def flush_touched(self, updates: Mapping[str, Any], *, generation: int) -> int:
        """将更新后的 FP32 参数原子写入 NVMe shard 文件。generation 单调递增。"""
        _require_non_negative_int("generation", generation)
        if generation < self.generation: raise ValueError(f"generation must be >= {self.generation}")
        if not updates: self.generation = generation; self.write_manifest(); return 0

        normalized = {pid: self._payload_for(pid, p) for pid, p in updates.items()}
        by_shard: dict[str, dict[ParamShardRecord, bytes]] = {}
        for pid, payload in normalized.items():
            record = self._record_for(pid)
            by_shard.setdefault(record.shard_name, {})[record] = payload
        for shard_name, shard_updates in by_shard.items():
            self._patch_shard_atomically(shard_name, shard_updates)
        self.generation = generation; self.write_manifest()
        return len(normalized)

    def shard_size_bytes(self, shard_name: str) -> int:
        sizes = [r.end_bytes for r in self.records.values() if r.shard_name == shard_name]
        if not sizes: raise KeyError(f"unknown shard {shard_name!r}")
        return max(sizes)

    # ── 内部 ──
    def _record_for(self, param_id: str) -> ParamShardRecord:
        try: return self.records[param_id]
        except KeyError as exc: raise KeyError(f"unknown FP32 param {param_id!r}") from exc
    def _shard_path(self, shard_name: str) -> Path: return self.root / shard_name
    def _payload_for(self, param_id: str, payload: Any) -> bytes:
        record = self._record_for(param_id); pb = _payload_to_bytes(payload)
        if len(pb) != record.num_bytes: raise ValueError(f"expected {record.num_bytes} bytes, got {len(pb)}")
        return pb
    def _patch_shard_atomically(self, shard_name: str, updates: Mapping[ParamShardRecord, bytes]) -> None:
        """原子 patch shard 文件（临时文件 + rename）。"""
        self.root.mkdir(parents=True, exist_ok=True)
        shard_path = self._shard_path(shard_name); shard_size = self.shard_size_bytes(shard_name)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{shard_name}.", suffix=".tmp", dir=self.root)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w+b") as out:
                if shard_path.exists():
                    with shard_path.open("rb") as inp: shutil.copyfileobj(inp, out)
                if out.tell() < shard_size: out.write(b"\0" * (shard_size - out.tell()))
                for record, payload in updates.items():
                    out.seek(record.offset_bytes); out.write(payload)
                out.flush(); os.fsync(out.fileno())
            os.replace(tmp_path, shard_path); _fsync_directory_best_effort(self.root)
        finally:
            if tmp_path.exists(): tmp_path.unlink()

# ── 工具函数 ──
def _payload_to_bytes(payload: Any) -> bytes:
    if isinstance(payload, bytes): return payload
    if isinstance(payload, (bytearray, memoryview)): return bytes(payload)
    if hasattr(payload, "detach") and hasattr(payload, "dtype"):
        t = payload.detach()
        if hasattr(t, "is_cuda") and t.is_cuda: t = t.cpu()
        if not t.is_contiguous(): t = t.contiguous()
        return t.numpy().tobytes()
    if hasattr(payload, "tobytes"): return payload.tobytes()
    raise TypeError(f"unsupported payload type {type(payload)!r}")

def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as s:
            s.write(payload); s.flush(); os.fsync(s.fileno())
        os.replace(tmp_path, path); _fsync_directory_best_effort(path.parent)
    finally:
        if tmp_path.exists(): tmp_path.unlink()

def _fsync_directory_best_effort(directory: Path) -> None:
    if os.name == "nt": return
    try: fd = os.open(directory, getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY)
    except OSError: return
    try: os.fsync(fd)
    finally: os.close(fd)
