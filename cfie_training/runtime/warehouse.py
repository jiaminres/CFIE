"""Parameter shard warehouse for the CFIE training runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor

from cfie_training.config import TrainingProjectConfig
from cfie_training.runtime.memory import TrainingMemoryPlan
from cfie_training.runtime.types import (
    ParameterShardSnapshot,
    ParameterWarehouseSummary,
    ResidencyTransition,
)


@dataclass(slots=True)
class _ParameterShardRecord:
    group_id: str
    component: str
    residency_state: str
    logical_params: int
    committed_version: int = 0
    pending_version: int | None = None
    bucket_id: int | None = None
    expert_ids: tuple[int, ...] = ()
    last_touched_step: int = -1

    # 将参数分片记录转换为可持久化快照。
    def to_snapshot(self) -> ParameterShardSnapshot:
        return ParameterShardSnapshot(
            group_id=self.group_id,
            component=self.component,
            residency_state=self.residency_state,
            committed_version=self.committed_version,
            pending_version=self.pending_version,
            logical_params=self.logical_params,
            bucket_id=self.bucket_id,
            expert_ids=self.expert_ids,
            last_touched_step=self.last_touched_step,
        )


@dataclass(slots=True, frozen=True)
class WarehouseStepResult:
    touched_shards: tuple[ParameterShardSnapshot, ...]
    warehouse_summary: ParameterWarehouseSummary


@dataclass(slots=True)
class ParameterWarehouse:
    config: TrainingProjectConfig
    _records: dict[str, _ParameterShardRecord] = field(default_factory=dict)

    # 为驻留迁移事件生成稳定分片 id。
    def _stable_shard_id(self, transition: ResidencyTransition) -> str:
        if transition.component == "static_modules":
            return "static_modules"
        if transition.component == "bucket_non_routed":
            return f"bucket_non_routed:{transition.bucket_id}"
        if transition.component == "bucket_active_experts":
            expert_suffix = "-".join(str(expert_id) for expert_id in transition.expert_ids)
            return f"bucket_active_experts:{transition.bucket_id}:{expert_suffix}"
        if transition.component == "expert_window_prefetch":
            expert_suffix = "-".join(str(expert_id) for expert_id in transition.expert_ids)
            return f"expert_window_prefetch:{expert_suffix}"
        return transition.group_id

    # 估算某个 non-routed bucket 对应的逻辑参数量。
    def _logical_bucket_non_routed_params(
        self,
        bucket_id: int | None,
        memory_plan: TrainingMemoryPlan,
    ) -> int:
        if bucket_id is None:
            return memory_plan.params_per_bucket_non_routed
        if bucket_id < len(memory_plan.bucket_non_routed_params_by_bucket):
            return memory_plan.bucket_non_routed_params_by_bucket[bucket_id]
        base = floor(
            memory_plan.bucket_non_routed_params_total / max(memory_plan.bucket_count, 1)
        )
        remainder = (
            memory_plan.bucket_non_routed_params_total % max(memory_plan.bucket_count, 1)
        )
        return base + (1 if bucket_id < remainder else 0)

    # 估算某个 active expert bucket 对应的逻辑参数量。
    def _logical_bucket_active_expert_params(
        self,
        bucket_id: int | None,
        memory_plan: TrainingMemoryPlan,
    ) -> int:
        if bucket_id is None:
            return memory_plan.params_per_bucket_active_routed
        if bucket_id < len(memory_plan.bucket_active_routed_params_by_bucket):
            return memory_plan.bucket_active_routed_params_by_bucket[bucket_id]
        return memory_plan.params_per_bucket_active_routed

    # 根据迁移类型选择对应的逻辑参数规模。
    def _logical_params_for_transition(
        self,
        transition: ResidencyTransition,
        memory_plan: TrainingMemoryPlan,
    ) -> int:
        if transition.component == "static_modules":
            return memory_plan.static_params_total
        if transition.component == "bucket_non_routed":
            return self._logical_bucket_non_routed_params(
                transition.bucket_id,
                memory_plan,
            )
        if transition.component == "bucket_active_experts":
            return self._logical_bucket_active_expert_params(
                transition.bucket_id,
                memory_plan,
            )
        if transition.component == "expert_window_prefetch":
            return memory_plan.params_per_bucket_prefetched_routed
        return 0

    # 读取已有分片记录，必要时按当前迁移创建新记录。
    def _get_or_create_record(
        self,
        *,
        shard_id: str,
        transition: ResidencyTransition,
        memory_plan: TrainingMemoryPlan,
    ) -> _ParameterShardRecord:
        record = self._records.get(shard_id)
        if record is None:
            record = _ParameterShardRecord(
                group_id=shard_id,
                component=transition.component,
                residency_state=transition.from_state,
                logical_params=self._logical_params_for_transition(
                    transition,
                    memory_plan,
                ),
                bucket_id=transition.bucket_id,
                expert_ids=transition.expert_ids,
            )
            self._records[shard_id] = record
        return record

    # 汇总当前仓库内所有分片的驻留统计。
    def _build_summary(self) -> ParameterWarehouseSummary:
        counts = {
            "nvme_cold": 0,
            "cpu_staged": 0,
            "gpu_active": 0,
            "cpu_dirty": 0,
        }
        dirty_shards = 0
        for record in self._records.values():
            counts[record.residency_state] += 1
            if record.pending_version is not None:
                dirty_shards += 1
        return ParameterWarehouseSummary(
            total_shards=len(self._records),
            nvme_cold=counts["nvme_cold"],
            cpu_staged=counts["cpu_staged"],
            gpu_active=counts["gpu_active"],
            cpu_dirty=counts["cpu_dirty"],
            dirty_shards=dirty_shards,
        )

    # 应用单步驻留规划，并更新参数仓库中的分片状态。
    def apply_residency_plan(
        self,
        *,
        step_index: int,
        transitions: tuple[ResidencyTransition, ...],
        memory_plan: TrainingMemoryPlan,
    ) -> WarehouseStepResult:
        # -----------------
        # 逐条应用迁移，更新分片状态与版本号。
        touched_ids: list[str] = []
        for transition in transitions:
            shard_id = self._stable_shard_id(transition)
            record = self._get_or_create_record(
                shard_id=shard_id,
                transition=transition,
                memory_plan=memory_plan,
            )
            record.bucket_id = transition.bucket_id
            record.expert_ids = transition.expert_ids
            record.last_touched_step = step_index
            record.residency_state = transition.to_state
            if transition.to_state == "cpu_dirty":
                record.pending_version = record.committed_version + 1
            elif transition.to_state == "nvme_cold" and record.pending_version is not None:
                record.committed_version = record.pending_version
                record.pending_version = None
            touched_ids.append(shard_id)

        # -----------------
        # 返回本步实际触达的分片快照与最新汇总。
        unique_touched = tuple(dict.fromkeys(touched_ids))
        snapshots = tuple(
            self._records[group_id].to_snapshot() for group_id in unique_touched
        )
        return WarehouseStepResult(
            touched_shards=snapshots,
            warehouse_summary=self._build_summary(),
        )

    # 导出当前参数仓库快照。
    def snapshot(self) -> tuple[ParameterShardSnapshot, ...]:
        return tuple(
            record.to_snapshot()
            for _, record in sorted(self._records.items(), key=lambda item: item[0])
        )

    # 从外部快照恢复参数仓库状态。
    def load_snapshot(
        self,
        snapshots: tuple[ParameterShardSnapshot, ...],
    ) -> None:
        self._records = {
            snapshot.group_id: _ParameterShardRecord(
                group_id=snapshot.group_id,
                component=snapshot.component,
                residency_state=snapshot.residency_state,
                logical_params=snapshot.logical_params,
                committed_version=snapshot.committed_version,
                pending_version=snapshot.pending_version,
                bucket_id=snapshot.bucket_id,
                expert_ids=snapshot.expert_ids,
                last_touched_step=snapshot.last_touched_step,
            )
            for snapshot in snapshots
        }

    # 返回当前参数仓库汇总。
    def summary(self) -> ParameterWarehouseSummary:
        return self._build_summary()
