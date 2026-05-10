"""Manifest 构建器——为 FP32/Adam/GPTQ 三种存储生成 ParamShardRecord 索引表。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from cfie_training.training_base.adam_state_store import (
    AdamStateShardRecord,
    CpuAdamFp8StateStore,
    state_key,
)
from cfie_training.training_base.adam_update import adam_state_num_bytes
from cfie_training.training_base.fp32_shard_store import (
    FP32_BYTES,
    FP32ShardStore,
    ParamShardRecord,
)
from cfie_training.training_base.gptq_cache_store import (
    GptqCacheRecord,
    GptqCacheStore,
)
from cfie_training.training_base.gptq_requant import (
    DEFAULT_GPTQ_GROUP_SIZE,
    gptq_bundle_num_bytes,
    gptq_layout_hash,
)


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
class TrainingParamManifestSpec:
    param_id: str
    num_elements: int
    trainable: bool = True
    gptq_bundle_id: str | None = None
    gptq_num_elements: int | None = None
    gptq_num_bytes: int | None = None
    quant_layout_hash: str = ""

    def __post_init__(self) -> None:
        _require_non_empty_string("param_id", self.param_id)
        _require_positive_int("num_elements", self.num_elements)
        if self.gptq_bundle_id is not None:
            _require_non_empty_string("gptq_bundle_id", self.gptq_bundle_id)
        if self.gptq_num_elements is not None:
            _require_positive_int("gptq_num_elements", self.gptq_num_elements)
        if self.gptq_num_bytes is not None:
            _require_positive_int("gptq_num_bytes", self.gptq_num_bytes)

    @property
    def resolved_gptq_num_elements(self) -> int:
        return (
            self.num_elements
            if self.gptq_num_elements is None
            else self.gptq_num_elements
        )

    @property
    def resolved_gptq_num_bytes(self) -> int | None:
        return self.gptq_num_bytes


@dataclass(frozen=True, slots=True)
class ManifestShardConfig:
    fp32_shard_bytes: int = 1 << 30
    adam_shard_bytes: int = 1 << 30
    gptq_shard_bytes: int = 1 << 30
    adam_block_size: int = 128
    gptq_group_size: int = DEFAULT_GPTQ_GROUP_SIZE
    fp32_shard_prefix: str = "fp32"
    adam_shard_prefix: str = "adam"
    gptq_shard_prefix: str = "gptq"
    adam_components: tuple[str, ...] = ("m", "v")

    def __post_init__(self) -> None:
        _require_positive_int("fp32_shard_bytes", self.fp32_shard_bytes)
        _require_positive_int("adam_shard_bytes", self.adam_shard_bytes)
        _require_positive_int("gptq_shard_bytes", self.gptq_shard_bytes)
        if self.fp32_shard_bytes % FP32_BYTES:
            raise ValueError("fp32_shard_bytes must be divisible by 4")
        _require_positive_int("adam_block_size", self.adam_block_size)
        _require_positive_int("gptq_group_size", self.gptq_group_size)
        _require_non_empty_string("fp32_shard_prefix", self.fp32_shard_prefix)
        _require_non_empty_string("adam_shard_prefix", self.adam_shard_prefix)
        _require_non_empty_string("gptq_shard_prefix", self.gptq_shard_prefix)
        if not self.adam_components:
            raise ValueError("adam_components must not be empty")
        for component in self.adam_components:
            _require_non_empty_string("adam component", component)


@dataclass(frozen=True, slots=True)
class TrainingBaseManifest:
    fp32_records: dict[str, ParamShardRecord]
    adam_records: dict[str, AdamStateShardRecord]
    gptq_records: dict[str, GptqCacheRecord]
    param_to_gptq_bundle: dict[str, str]

    def create_stores(
        self,
        root: str | Path,
        *,
        generation: int = 0,
    ) -> tuple[FP32ShardStore, CpuAdamFp8StateStore, GptqCacheStore]:
        root_path = Path(root)
        return (
            FP32ShardStore.create(
                root_path / "fp32",
                self.fp32_records,
                generation=generation,
            ),
            CpuAdamFp8StateStore.create(
                root_path / "adam",
                self.adam_records,
                generation=generation,
            ),
            GptqCacheStore.create(
                root_path / "gptq",
                self.gptq_records,
                generation=generation,
            ),
        )

    @property
    def total_fp32_bytes(self) -> int:
        return sum(record.num_bytes for record in self.fp32_records.values())

    @property
    def total_adam_bytes(self) -> int:
        return sum(record.num_bytes for record in self.adam_records.values())

    @property
    def total_gptq_bytes(self) -> int:
        return sum(record.num_bytes for record in self.gptq_records.values())


@dataclass(slots=True)
class TrainingBaseManifestBuilder:
    config: ManifestShardConfig = field(default_factory=ManifestShardConfig)

    def build(
        self,
        specs: Iterable[TrainingParamManifestSpec],
    ) -> TrainingBaseManifest:
        ordered_specs = tuple(specs)
        self._validate_unique_specs(ordered_specs)

        fp32_records = self._build_fp32_records(ordered_specs)
        adam_records = self._build_adam_records(ordered_specs)
        gptq_records, param_to_gptq_bundle = self._build_gptq_records(ordered_specs)
        return TrainingBaseManifest(
            fp32_records=fp32_records,
            adam_records=adam_records,
            gptq_records=gptq_records,
            param_to_gptq_bundle=param_to_gptq_bundle,
        )

    def _build_fp32_records(
        self,
        specs: tuple[TrainingParamManifestSpec, ...],
    ) -> dict[str, ParamShardRecord]:
        cursor = _ShardCursor(
            shard_bytes=self.config.fp32_shard_bytes,
            shard_prefix=self.config.fp32_shard_prefix,
            extension=".bin",
        )
        records: dict[str, ParamShardRecord] = {}
        for spec in specs:
            offset_bytes, shard_name = cursor.allocate(spec.num_elements * FP32_BYTES)
            records[spec.param_id] = ParamShardRecord(
                param_id=spec.param_id,
                shard_name=shard_name,
                offset_elements=offset_bytes // FP32_BYTES,
                num_elements=spec.num_elements,
            )
        return records

    def _build_adam_records(
        self,
        specs: tuple[TrainingParamManifestSpec, ...],
    ) -> dict[str, AdamStateShardRecord]:
        cursor = _ShardCursor(
            shard_bytes=self.config.adam_shard_bytes,
            shard_prefix=self.config.adam_shard_prefix,
            extension=".bin",
        )
        records: dict[str, AdamStateShardRecord] = {}
        for spec in specs:
            if not spec.trainable:
                continue
            num_bytes = adam_state_num_bytes(
                spec.num_elements,
                block_size=self.config.adam_block_size,
            )
            for component in self.config.adam_components:
                offset_bytes, shard_name = cursor.allocate(num_bytes)
                record = AdamStateShardRecord(
                    param_id=spec.param_id,
                    component=component,
                    shard_name=shard_name,
                    offset_bytes=offset_bytes,
                    num_bytes=num_bytes,
                )
                records[state_key(spec.param_id, component)] = record
        return records

    def _build_gptq_records(
        self,
        specs: tuple[TrainingParamManifestSpec, ...],
    ) -> tuple[dict[str, GptqCacheRecord], dict[str, str]]:
        cursor = _ShardCursor(
            shard_bytes=self.config.gptq_shard_bytes,
            shard_prefix=self.config.gptq_shard_prefix,
            extension=".bin",
        )
        records: dict[str, GptqCacheRecord] = {}
        param_to_bundle: dict[str, str] = {}
        default_layout_hash = gptq_layout_hash(
            group_size=self.config.gptq_group_size
        )
        for spec in specs:
            if spec.gptq_bundle_id is None:
                continue
            if spec.gptq_bundle_id in records:
                raise ValueError(f"duplicate GPTQ bundle {spec.gptq_bundle_id!r}")
            num_bytes = (
                spec.resolved_gptq_num_bytes
                if spec.resolved_gptq_num_bytes is not None
                else gptq_bundle_num_bytes(
                    spec.resolved_gptq_num_elements,
                    group_size=self.config.gptq_group_size,
                )
            )
            offset_bytes, shard_name = cursor.allocate(num_bytes)
            records[spec.gptq_bundle_id] = GptqCacheRecord(
                bundle_id=spec.gptq_bundle_id,
                shard_name=shard_name,
                offset_bytes=offset_bytes,
                num_bytes=num_bytes,
                quant_layout_hash=spec.quant_layout_hash or default_layout_hash,
            )
            param_to_bundle[spec.param_id] = spec.gptq_bundle_id
        return records, param_to_bundle

    @staticmethod
    def _validate_unique_specs(
        specs: tuple[TrainingParamManifestSpec, ...],
    ) -> None:
        seen: set[str] = set()
        for spec in specs:
            if spec.param_id in seen:
                raise ValueError(f"duplicate param_id {spec.param_id!r}")
            seen.add(spec.param_id)


@dataclass(frozen=True, slots=True)
class Qwen35MoeManifestConfig:
    num_layers: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    tp_size: int = 1
    layer_start: int = 0
    layer_prefix: str = "layers"
    trainable: bool = True
    include_gptq_cache: bool = True
    local_expert_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _require_positive_int("num_layers", self.num_layers)
        _require_positive_int("num_experts", self.num_experts)
        _require_positive_int("hidden_size", self.hidden_size)
        _require_positive_int("intermediate_size", self.intermediate_size)
        _require_positive_int("tp_size", self.tp_size)
        _require_non_negative_int("layer_start", self.layer_start)
        _require_non_empty_string("layer_prefix", self.layer_prefix)
        if self.intermediate_size % self.tp_size:
            raise ValueError("intermediate_size must be divisible by tp_size")
        if self.local_expert_ids is not None:
            for expert_id in self.local_expert_ids:
                _require_non_negative_int("expert_id", expert_id)
                if expert_id >= self.num_experts:
                    raise ValueError("local_expert_ids must be < num_experts")


def make_qwen35_moe_manifest_specs(
    config: Qwen35MoeManifestConfig,
) -> tuple[TrainingParamManifestSpec, ...]:
    expert_ids = (
        config.local_expert_ids
        if config.local_expert_ids is not None
        else tuple(range(config.num_experts))
    )
    intermediate_per_rank = config.intermediate_size // config.tp_size
    w13_elements = 2 * intermediate_per_rank * config.hidden_size
    w2_elements = config.hidden_size * intermediate_per_rank
    specs: list[TrainingParamManifestSpec] = []
    for layer_id in range(config.layer_start, config.layer_start + config.num_layers):
        for expert_id in expert_ids:
            for weight_name, num_elements in (
                ("w13_weight", w13_elements),
                ("w2_weight", w2_elements),
            ):
                param_id = (
                    f"{config.layer_prefix}.{layer_id}."
                    f"experts.{expert_id}.{weight_name}"
                )
                specs.append(
                    TrainingParamManifestSpec(
                        param_id=param_id,
                        num_elements=num_elements,
                        trainable=config.trainable,
                        gptq_bundle_id=(
                            param_id if config.include_gptq_cache else None
                        ),
                    )
                )
    return tuple(specs)


@dataclass(slots=True)
class _ShardCursor:
    shard_bytes: int
    shard_prefix: str
    extension: str
    shard_index: int = 0
    offset_bytes: int = 0

    def allocate(self, num_bytes: int) -> tuple[int, str]:
        _require_positive_int("num_bytes", num_bytes)
        if self.offset_bytes and self.offset_bytes + num_bytes > self.shard_bytes:
            self.shard_index += 1
            self.offset_bytes = 0
        shard_name = self._shard_name()
        offset_bytes = self.offset_bytes
        self.offset_bytes += num_bytes
        if self.offset_bytes > self.shard_bytes:
            self.shard_index += 1
            self.offset_bytes = 0
        return offset_bytes, shard_name

    def _shard_name(self) -> str:
        return f"{self.shard_prefix}_{self.shard_index:04d}{self.extension}"
