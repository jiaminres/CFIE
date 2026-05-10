"""容量规划——根据 manifest 估算 FP32/Adam/GPTQ 三种存储的 NVMe 占用。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from cfie_training.training_base.adam_state_store import AdamStateShardRecord
from cfie_training.training_base.fp32_shard_store import ParamShardRecord
from cfie_training.training_base.gptq_cache_store import GptqCacheRecord
from cfie_training.training_base.manifest_builder import (
    ManifestShardConfig,
    Qwen35MoeManifestConfig,
    TrainingBaseManifest,
    TrainingBaseManifestBuilder,
    make_qwen35_moe_manifest_specs,
)

GIB = 1024**3


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True, slots=True)
class TrainingBaseCapacityReport:
    param_count: int
    trainable_param_count: int
    gptq_bundle_count: int
    fp32_shard_count: int
    adam_shard_count: int
    gptq_shard_count: int
    total_fp32_bytes: int
    total_adam_bytes: int
    total_gptq_bytes: int
    max_fp32_shard_bytes: int = 0
    max_adam_shard_bytes: int = 0
    max_gptq_shard_bytes: int = 0

    def __post_init__(self) -> None:
        _require_non_negative_int("param_count", self.param_count)
        _require_non_negative_int("trainable_param_count", self.trainable_param_count)
        _require_non_negative_int("gptq_bundle_count", self.gptq_bundle_count)
        _require_non_negative_int("fp32_shard_count", self.fp32_shard_count)
        _require_non_negative_int("adam_shard_count", self.adam_shard_count)
        _require_non_negative_int("gptq_shard_count", self.gptq_shard_count)
        _require_non_negative_int("total_fp32_bytes", self.total_fp32_bytes)
        _require_non_negative_int("total_adam_bytes", self.total_adam_bytes)
        _require_non_negative_int("total_gptq_bytes", self.total_gptq_bytes)
        _require_non_negative_int("max_fp32_shard_bytes", self.max_fp32_shard_bytes)
        _require_non_negative_int("max_adam_shard_bytes", self.max_adam_shard_bytes)
        _require_non_negative_int("max_gptq_shard_bytes", self.max_gptq_shard_bytes)
        if self.trainable_param_count > self.param_count:
            raise ValueError("trainable_param_count must be <= param_count")

    @property
    def total_persistent_bytes(self) -> int:
        return self.total_fp32_bytes + self.total_adam_bytes + self.total_gptq_bytes

    @property
    def total_persistent_gib(self) -> float:
        return bytes_to_gib(self.total_persistent_bytes)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "param_count": self.param_count,
            "trainable_param_count": self.trainable_param_count,
            "gptq_bundle_count": self.gptq_bundle_count,
            "fp32_shard_count": self.fp32_shard_count,
            "adam_shard_count": self.adam_shard_count,
            "gptq_shard_count": self.gptq_shard_count,
            "total_fp32_bytes": self.total_fp32_bytes,
            "total_adam_bytes": self.total_adam_bytes,
            "total_gptq_bytes": self.total_gptq_bytes,
            "total_persistent_bytes": self.total_persistent_bytes,
            "total_persistent_gib": self.total_persistent_gib,
            "max_fp32_shard_bytes": self.max_fp32_shard_bytes,
            "max_adam_shard_bytes": self.max_adam_shard_bytes,
            "max_gptq_shard_bytes": self.max_gptq_shard_bytes,
        }


def capacity_report_from_manifest(
    manifest: TrainingBaseManifest,
) -> TrainingBaseCapacityReport:
    trainable_param_ids = {
        record.param_id
        for record in manifest.adam_records.values()
    }
    return TrainingBaseCapacityReport(
        param_count=len(manifest.fp32_records),
        trainable_param_count=len(trainable_param_ids),
        gptq_bundle_count=len(manifest.gptq_records),
        fp32_shard_count=_unique_shard_count(manifest.fp32_records),
        adam_shard_count=_unique_shard_count(manifest.adam_records),
        gptq_shard_count=_unique_shard_count(manifest.gptq_records),
        total_fp32_bytes=manifest.total_fp32_bytes,
        total_adam_bytes=manifest.total_adam_bytes,
        total_gptq_bytes=manifest.total_gptq_bytes,
        max_fp32_shard_bytes=_max_shard_bytes(manifest.fp32_records),
        max_adam_shard_bytes=_max_shard_bytes(manifest.adam_records),
        max_gptq_shard_bytes=_max_shard_bytes(manifest.gptq_records),
    )


def estimate_qwen35_moe_capacity(
    qwen_config: Qwen35MoeManifestConfig,
    shard_config: ManifestShardConfig | None = None,
) -> TrainingBaseCapacityReport:
    manifest = TrainingBaseManifestBuilder(
        ManifestShardConfig() if shard_config is None else shard_config
    ).build(make_qwen35_moe_manifest_specs(qwen_config))
    return capacity_report_from_manifest(manifest)


def bytes_to_gib(num_bytes: int) -> float:
    # bytes → GiB
    _require_non_negative_int("num_bytes", num_bytes)
    return num_bytes / (1 << 30)


def _unique_shard_count(
    records: Mapping[str, ParamShardRecord | AdamStateShardRecord | GptqCacheRecord],
) -> int:
    return len({record.shard_name for record in records.values()})


def _max_shard_bytes(
    records: Mapping[str, ParamShardRecord | AdamStateShardRecord | GptqCacheRecord],
) -> int:
    max_by_shard: dict[str, int] = {}
    for record in records.values():
        max_by_shard[record.shard_name] = max(
            max_by_shard.get(record.shard_name, 0),
            record.end_bytes,
        )
    return max(max_by_shard.values(), default=0)
