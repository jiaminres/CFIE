"""训练进度原子持久化——临时文件+rename 保证 step/epoch/游标原子写入（设计文档 Section 13.1）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1
DEFAULT_PROGRESS_FILENAME = "progress_state.json"


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_string(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def digest_hot_set(hot_set: Iterable[str] | Mapping[str, Any] | None) -> str:
    if hot_set is None:
        return ""
    if isinstance(hot_set, Mapping):
        payload: Any = hot_set
    else:
        payload = sorted(str(item) for item in hot_set)
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass(slots=True)
class TrainingProgressState:
    """训练进度状态——记录恢复游标，不保存参数数据（设计文档 Section 13.1）。"""
    schema_version: int = SCHEMA_VERSION
    global_step: int = 0                          # 全局训练步数
    epoch: int = 0                                # 训练轮次
    dataset_cursor: str = ""                       # 数据集游标（恢复用）
    consumed_samples: int = 0
    consumed_tokens: int = 0
    round_id: int = 0                             # hot set round 编号
    hot_set_digest: str = ""                      # hot set 的 SHA256 摘要
    flush_generation: int = 0                     # 刷盘代数（单调递增）
    fp32_master_generation: int = 0
    optimizer_generation: int = 0
    gptq_cache_generation: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "TrainingProgressState":
        _require_non_negative_int("schema_version", self.schema_version)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported progress schema version {self.schema_version}"
            )
        _require_non_negative_int("global_step", self.global_step)
        _require_non_negative_int("epoch", self.epoch)
        _require_string("dataset_cursor", self.dataset_cursor)
        _require_non_negative_int("consumed_samples", self.consumed_samples)
        _require_non_negative_int("consumed_tokens", self.consumed_tokens)
        _require_non_negative_int("round_id", self.round_id)
        _require_string("hot_set_digest", self.hot_set_digest)
        _require_non_negative_int("flush_generation", self.flush_generation)
        _require_non_negative_int(
            "fp32_master_generation",
            self.fp32_master_generation,
        )
        _require_non_negative_int(
            "optimizer_generation",
            self.optimizer_generation,
        )
        _require_non_negative_int(
            "gptq_cache_generation",
            self.gptq_cache_generation,
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingProgressState":
        known_fields = cls.__dataclass_fields__
        values = {
            name: data[name]
            for name in known_fields
            if name in data
        }
        return cls(**values)

    @classmethod
    def initial(cls) -> "TrainingProgressState":
        return cls()

    def assert_generations(
        self,
        *,
        fp32_master_generation: int,
        optimizer_generation: int,
        gptq_cache_generation: int,
    ) -> None:
        expected = {
            "fp32_master_generation": self.fp32_master_generation,
            "optimizer_generation": self.optimizer_generation,
            "gptq_cache_generation": self.gptq_cache_generation,
        }
        actual = {
            "fp32_master_generation": fp32_master_generation,
            "optimizer_generation": optimizer_generation,
            "gptq_cache_generation": gptq_cache_generation,
        }
        mismatches = [
            f"{name}: expected {expected[name]}, got {actual[name]}"
            for name in expected
            if expected[name] != actual[name]
        ]
        if mismatches:
            raise ValueError("progress generation mismatch: " + "; ".join(mismatches))


@dataclass(slots=True)
# ────── ProgressStateWriter — 训练进度原子持久化（临时文件+rename）──────
class ProgressStateWriter:
    path: Path

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def in_dir(
        cls,
        directory: str | Path,
        filename: str = DEFAULT_PROGRESS_FILENAME,
    ) -> "ProgressStateWriter":
        return cls(Path(directory) / filename)

    def load_latest_or_init(self) -> TrainingProgressState:
        if not self.path.exists():
            return TrainingProgressState.initial()
        try:
            with self.path.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid progress state JSON: {self.path}") from exc
        if not isinstance(data, Mapping):
            raise ValueError(f"progress state must be a JSON object: {self.path}")
        return TrainingProgressState.from_dict(data)

    def write_state(self, state: TrainingProgressState) -> TrainingProgressState:
        state.validate()
        payload = json.dumps(
            state.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        self._atomic_write_text(payload + "\n")
        return state

    def write_after_flush(
        self,
        *,
        global_step: int,
        epoch: int,
        dataset_cursor: str,
        round_id: int,
        hot_set: Iterable[str] | Mapping[str, Any] | None = None,
        consumed_samples: int = 0,
        consumed_tokens: int = 0,
        flush_generation: int | None = None,
        fp32_master_generation: int | None = None,
        optimizer_generation: int | None = None,
        gptq_cache_generation: int | None = None,
    ) -> TrainingProgressState:
        generation = global_step if flush_generation is None else flush_generation
        state = TrainingProgressState(
            global_step=global_step,
            epoch=epoch,
            dataset_cursor=dataset_cursor,
            consumed_samples=consumed_samples,
            consumed_tokens=consumed_tokens,
            round_id=round_id,
            hot_set_digest=digest_hot_set(hot_set),
            flush_generation=generation,
            fp32_master_generation=(
                generation
                if fp32_master_generation is None
                else fp32_master_generation
            ),
            optimizer_generation=(
                generation
                if optimizer_generation is None
                else optimizer_generation
            ),
            gptq_cache_generation=(
                generation
                if gptq_cache_generation is None
                else gptq_cache_generation
            ),
        )
        return self.write_state(state)

    def _atomic_write_text(self, payload: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_path, self.path)
            _fsync_directory_best_effort(self.path.parent)
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
