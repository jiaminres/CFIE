
"""训练窗口运行时——管理窗口生命周期：begin→commit 的完整流程。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol

from cfie_training.training_base.progress_state import TrainingProgressState
from cfie_training.training_base.window_commit import TrainingWindowCommitter
from cfie_training.training_base.window_plan import (
    HotSetPlan,
    TrainableParamSpec,
    TrainingWindowPlanner,
)


class TrainingWindowHooks(Protocol):
    def prepare_window(
        self,
        plan: HotSetPlan,
        progress: TrainingProgressState,
    ) -> None:
        ...

    def drain_before_commit(
        self,
        plan: HotSetPlan,
        payload: "WindowCommitPayload",
    ) -> None:
        ...

    def after_commit(
        self,
        plan: HotSetPlan,
        state: TrainingProgressState,
    ) -> None:
        ...


@dataclass(slots=True)
class NoOpTrainingWindowHooks:
    def prepare_window(
        self,
        plan: HotSetPlan,
        progress: TrainingProgressState,
    ) -> None:
        return None

    def drain_before_commit(
        self,
        plan: HotSetPlan,
        payload: "WindowCommitPayload",
    ) -> None:
        return None

    def after_commit(
        self,
        plan: HotSetPlan,
        state: TrainingProgressState,
    ) -> None:
        return None


@dataclass(slots=True)
class LoggingTrainingWindowHooks:
    logger_name: str = "cfie.training_base.window"
    log_level: int = 20

    def prepare_window(
        self,
        plan: HotSetPlan,
        progress: TrainingProgressState,
    ) -> None:
        import logging

        logger = logging.getLogger(self.logger_name)
        logger.log(
            self.log_level,
            "Window %d: %d params, fp32=%d gpu=%d step=%d..%d",
            plan.window_index,
            len(plan.selected_params),
            plan.total_fp32_hot_bytes,
            plan.total_gpu_shadow_bytes,
            plan.start_step,
            plan.end_step_exclusive,
        )

    def drain_before_commit(
        self,
        plan: HotSetPlan,
        payload: "WindowCommitPayload",
    ) -> None:
        import logging

        logger = logging.getLogger(self.logger_name)
        logger.log(
            self.log_level,
            "Window %d drain: %d fp32 updates, %d adam updates, %d gptq updates",
            plan.window_index,
            len(payload.fp32_updates),
            len(payload.adam_updates or {}),
            len(payload.gptq_updates or {}),
        )

    def after_commit(
        self,
        plan: HotSetPlan,
        state: TrainingProgressState,
    ) -> None:
        import logging

        logger = logging.getLogger(self.logger_name)
        logger.log(
            self.log_level,
            "Window %d committed: step=%d epoch=%d generation=%d",
            plan.window_index,
            state.global_step,
            state.epoch,
            state.flush_generation,
        )


@dataclass(slots=True)
class ValidatingTrainingWindowHooks:
    strict: bool = True

    def prepare_window(
        self,
        plan: HotSetPlan,
        progress: TrainingProgressState,
    ) -> None:
        if not plan.selected_params:
            raise ValueError("Training window plan contains no params")
        if plan.start_step < progress.global_step:
            raise ValueError(
                f"Window start step {plan.start_step} is before "
                f"current progress step {progress.global_step}"
            )

    def drain_before_commit(
        self,
        plan: HotSetPlan,
        payload: "WindowCommitPayload",
    ) -> None:
        if not self.strict:
            return
        allowed = set(plan.param_ids)
        fp32_missing = allowed - set(payload.fp32_updates)
        if fp32_missing:
            raise ValueError(
                "FP32 updates missing for planned params: "
                + ", ".join(sorted(fp32_missing))
            )

    def after_commit(
        self,
        plan: HotSetPlan,
        state: TrainingProgressState,
    ) -> None:
        if state.global_step < plan.end_step_exclusive:
            raise ValueError(
                f"Progress global_step {state.global_step} is before "
                f"window end {plan.end_step_exclusive}"
            )


@dataclass(slots=True)
class CompositeTrainingWindowHooks:
    hooks: list[TrainingWindowHooks]

    def prepare_window(
        self,
        plan: HotSetPlan,
        progress: TrainingProgressState,
    ) -> None:
        for hook in self.hooks:
            hook.prepare_window(plan, progress)

    def drain_before_commit(
        self,
        plan: HotSetPlan,
        payload: "WindowCommitPayload",
    ) -> None:
        for hook in self.hooks:
            hook.drain_before_commit(plan, payload)

    def after_commit(
        self,
        plan: HotSetPlan,
        state: TrainingProgressState,
    ) -> None:
        for hook in self.hooks:
            hook.after_commit(plan, state)


@dataclass(frozen=True, slots=True)
class WindowCommitPayload:
    fp32_updates: Mapping[str, Any]
    global_step: int
    epoch: int
    dataset_cursor: str
    adam_updates: Mapping[str, Mapping[str, Any]] | None = None
    gptq_updates: Mapping[str, Any] | None = None
    touched_param_ids: Iterable[str] | None = None
    consumed_samples: int = 0
    consumed_tokens: int = 0
    flush_generation: int | None = None
    optimizer_generation: int | None = None
    gptq_cache_generation: int | None = None

    def touched_ids(self) -> tuple[str, ...]:
        if self.touched_param_ids is not None:
            return tuple(dict.fromkeys(str(item) for item in self.touched_param_ids))
        return tuple(dict.fromkeys(str(item) for item in self.fp32_updates))


@dataclass(slots=True)
# ────── TrainingWindowRuntime — 窗口生命周期管理：begin→commit ──────
class TrainingWindowRuntime:
    planner: TrainingWindowPlanner
    committer: TrainingWindowCommitter
    candidates: tuple[TrainableParamSpec, ...] = field(default_factory=tuple)
    hooks: TrainingWindowHooks = field(default_factory=NoOpTrainingWindowHooks)

    def begin_window(self, candidates: Iterable[TrainableParamSpec] | None = None) -> HotSetPlan:
        """开始新训练窗口：读取上次 progress → 规划 hot set → 返回 plan。"""
        progress = self.committer.load_committed_progress()      # 从 NVMe 读取上次提交的 progress state
        plan = self.planner.plan_window(                         # TrainingWindowPlanner 规划 hot set
            self._candidate_pool(candidates),                    # 候选参数列表
            global_step=progress.global_step,                    # 上次全局步数
        )
        self.hooks.prepare_window(plan, progress)                # hook: 窗口准备
        return plan

    def commit_window(self, plan: HotSetPlan, payload: WindowCommitPayload) -> TrainingProgressState:
        """提交窗口：校验 → hook → committer → hook → 返回新 progress state。"""
        self._validate_commit_payload(plan, payload)             # 校验 payload 与 plan 一致
        self.hooks.drain_before_commit(plan, payload)            # hook: 提交前
        state = self.committer.commit_window(                    # TrainingWindowCommitter 原子写入
            fp32_updates=payload.fp32_updates,                   # FP32 master → NVMe
            adam_updates=payload.adam_updates,                   # Adam FP8 → NVMe
            gptq_updates=payload.gptq_updates,                   # GPTQ Int4 → NVMe/CPU
            global_step=payload.global_step, epoch=payload.epoch,
            dataset_cursor=payload.dataset_cursor, round_id=plan.window_index,
            hot_set=plan.param_ids,
            consumed_samples=payload.consumed_samples, consumed_tokens=payload.consumed_tokens,
            flush_generation=payload.flush_generation,
            optimizer_generation=payload.optimizer_generation,
            gptq_cache_generation=payload.gptq_cache_generation,
        )
        self.hooks.after_commit(plan, state)                     # hook: 提交后
        return state

    def load_committed_progress(self) -> TrainingProgressState:
        return self.committer.load_committed_progress()

    def _candidate_pool(
        self,
        candidates: Iterable[TrainableParamSpec] | None,
    ) -> tuple[TrainableParamSpec, ...]:
        pool = tuple(self.candidates if candidates is None else candidates)
        if not pool:
            raise ValueError("training window candidates must not be empty")
        return pool

    @staticmethod
    def _validate_commit_payload(
        plan: HotSetPlan,
        payload: WindowCommitPayload,
    ) -> None:
        if payload.global_step <= plan.start_step:
            raise ValueError(
                "commit global_step must advance beyond the window start"
            )
        if payload.global_step > plan.end_step_exclusive:
            raise ValueError(
                "commit global_step must not exceed the window end"
            )

        allowed = set(plan.param_ids)
        fp32_outside = set(payload.fp32_updates) - allowed
        if fp32_outside:
            raise ValueError(
                "FP32 updates contain params outside the hot set: "
                + ", ".join(sorted(fp32_outside))
            )

        adam_outside = set(payload.adam_updates or {}) - allowed
        if adam_outside:
            raise ValueError(
                "Adam updates contain params outside the hot set: "
                + ", ".join(sorted(adam_outside))
            )

        touched_outside = set(payload.touched_ids()) - allowed
        if touched_outside:
            raise ValueError(
                "touched params contain ids outside the hot set: "
                + ", ".join(sorted(touched_outside))
            )
