"""Parameter residency state machine for the CFIE training runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

from cfie_training.config import TrainingProjectConfig
from cfie_training.runtime.memory import TrainingMemoryPlan
from cfie_training.runtime.types import (
    LayerBucketPlan,
    ResidencyState,
    ResidencyTransition,
)


@dataclass(slots=True, frozen=True)
class ResidencyPlanResult:
    transitions: tuple[ResidencyTransition, ...]
    ending_states: dict[str, str]

    # 将驻留规划结果序列化为字典。
    def to_dict(self) -> dict[str, object]:
        return {
            "transitions": [transition.to_dict() for transition in self.transitions],
            "ending_states": self.ending_states,
        }


@dataclass(slots=True)
class _ResidencyStateStore:
    static_modules_state: ResidencyState = "nvme_cold"
    staged_expert_ids: tuple[int, ...] = ()


@dataclass(slots=True)
class ParameterResidencyController:
    config: TrainingProjectConfig
    _state: _ResidencyStateStore = field(default_factory=_ResidencyStateStore)

    # 返回静态模块当前驻留状态。
    @property
    def static_modules_state(self) -> ResidencyState:
        return self._state.static_modules_state

    # 返回当前缓存的 expert 预取窗口。
    @property
    def staged_expert_ids(self) -> tuple[int, ...]:
        return self._state.staged_expert_ids

    # 从外部快照恢复驻留控制器内部状态。
    def load_state(
        self,
        *,
        static_modules_state: ResidencyState,
        staged_expert_ids: tuple[int, ...],
    ) -> None:
        self._state = _ResidencyStateStore(
            static_modules_state=static_modules_state,
            staged_expert_ids=staged_expert_ids,
        )

    # 向迁移列表中追加一条标准化的驻留状态变更记录。
    def _transition(
        self,
        transitions: list[ResidencyTransition],
        *,
        group_id: str,
        component: str,
        from_state: ResidencyState,
        to_state: ResidencyState,
        trigger: str,
        bucket_id: int | None = None,
        expert_ids: tuple[int, ...] = (),
    ) -> None:
        transitions.append(
            ResidencyTransition(
                group_id=group_id,
                component=component,
                from_state=from_state,
                to_state=to_state,
                trigger=trigger,
                bucket_id=bucket_id,
                expert_ids=expert_ids,
            )
        )

    # 为单个训练 step 规划参数驻留迁移序列。
    def plan_step(
        self,
        *,
        step_index: int,
        layer_buckets: tuple[LayerBucketPlan, ...],
        active_expert_ids: tuple[int, ...],
        prefetched_expert_ids: tuple[int, ...],
        memory_plan: TrainingMemoryPlan,
        stage_static_modules: bool,
        update_state: bool,
    ) -> ResidencyPlanResult:
        # -----------------
        # 初始化本步的迁移列表与当前状态快照。
        transitions: list[ResidencyTransition] = []
        ending_states: dict[str, str] = {}

        static_modules_state = self._state.static_modules_state
        staged_expert_ids = self._state.staged_expert_ids

        # -----------------
        # 如需首次预热静态模块，则先把它们从冷存储拉到 CPU。
        if stage_static_modules and static_modules_state == "nvme_cold":
            self._transition(
                transitions,
                group_id="static_modules",
                component="static_modules",
                from_state="nvme_cold",
                to_state="cpu_staged",
                trigger="initial_stage",
            )
            static_modules_state = "cpu_staged"
        ending_states["static_modules"] = static_modules_state

        # -----------------
        # 根据 CPU hot 预算决定是否保留下一步的 expert 预取窗口。
        keep_prefetched_window = memory_plan.cpu_hot.within_budget
        prefetch_group_id = f"expert_window:{step_index + 1}"
        if prefetched_expert_ids and keep_prefetched_window:
            self._transition(
                transitions,
                group_id=prefetch_group_id,
                component="expert_window_prefetch",
                from_state="nvme_cold",
                to_state="cpu_staged",
                trigger="prefetch_next_step_window",
                expert_ids=prefetched_expert_ids,
            )
            ending_states[prefetch_group_id] = "cpu_staged"

        # -----------------
        # 判断当前 active window 是复用预取结果还是从冷层重新激活。
        active_source_state: ResidencyState = (
            "cpu_staged"
            if staged_expert_ids
            and set(active_expert_ids).issubset(staged_expert_ids)
            else "nvme_cold"
        )
        consumed_prefetch_group_id = f"expert_window:{step_index}"
        if active_source_state == "cpu_staged":
            self._transition(
                transitions,
                group_id=consumed_prefetch_group_id,
                component="expert_window_prefetch",
                from_state="cpu_staged",
                to_state="nvme_cold",
                trigger="release_consumed_prefetch_window",
                expert_ids=active_expert_ids,
            )
            ending_states[consumed_prefetch_group_id] = "nvme_cold"

        # -----------------
        # 为每个 bucket 依次规划 non-routed 与 active experts 的完整生命周期。
        for bucket in layer_buckets:
            non_routed_group = f"bucket_non_routed:{bucket.bucket_id}"
            self._transition(
                transitions,
                group_id=non_routed_group,
                component="bucket_non_routed",
                from_state="nvme_cold",
                to_state="cpu_staged",
                trigger="prefetch_bucket_non_routed",
                bucket_id=bucket.bucket_id,
            )
            self._transition(
                transitions,
                group_id=non_routed_group,
                component="bucket_non_routed",
                from_state="cpu_staged",
                to_state="gpu_active",
                trigger="activate_bucket_non_routed",
                bucket_id=bucket.bucket_id,
            )
            self._transition(
                transitions,
                group_id=non_routed_group,
                component="bucket_non_routed",
                from_state="gpu_active",
                to_state="cpu_dirty",
                trigger="bucket_backward_finished",
                bucket_id=bucket.bucket_id,
            )
            self._transition(
                transitions,
                group_id=non_routed_group,
                component="bucket_non_routed",
                from_state="cpu_dirty",
                to_state="nvme_cold",
                trigger="flush_updated_bucket_non_routed",
                bucket_id=bucket.bucket_id,
            )
            ending_states[non_routed_group] = "nvme_cold"

            experts_group = f"bucket_active_experts:{bucket.bucket_id}"
            self._transition(
                transitions,
                group_id=experts_group,
                component="bucket_active_experts",
                from_state=active_source_state,
                to_state="gpu_active",
                trigger="activate_bucket_expert_window",
                bucket_id=bucket.bucket_id,
                expert_ids=active_expert_ids,
            )
            self._transition(
                transitions,
                group_id=experts_group,
                component="bucket_active_experts",
                from_state="gpu_active",
                to_state="cpu_dirty",
                trigger="bucket_expert_backward_finished",
                bucket_id=bucket.bucket_id,
                expert_ids=active_expert_ids,
            )
            self._transition(
                transitions,
                group_id=experts_group,
                component="bucket_active_experts",
                from_state="cpu_dirty",
                to_state="nvme_cold",
                trigger="flush_updated_bucket_experts",
                bucket_id=bucket.bucket_id,
                expert_ids=active_expert_ids,
            )
            ending_states[experts_group] = "nvme_cold"

        # -----------------
        # 计算新的缓存窗口，并按需回写内部状态。
        next_staged_expert_ids = (
            prefetched_expert_ids if keep_prefetched_window else ()
        )
        if update_state:
            self._state = _ResidencyStateStore(
                static_modules_state=static_modules_state,
                staged_expert_ids=next_staged_expert_ids,
            )
        return ResidencyPlanResult(
            transitions=tuple(transitions),
            ending_states=ending_states,
        )
