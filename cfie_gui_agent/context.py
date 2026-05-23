from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

CONTEXT_USAGE_NORMAL = "normal"
CONTEXT_USAGE_WARNING = "warning"
CONTEXT_USAGE_COMPACT = "compact"
CONTEXT_USAGE_EMERGENCY = "emergency"

KEY_EVIDENCE_TAGS = frozenset(
    {
        "key_evidence",
        "failure",
        "rollback",
        "override",
        "interrupt",
        "task_completed",
        "first_arrival",
        "human_request",
    }
)


@dataclass(slots=True, frozen=True)
class VisionContextPolicy:
    mode: str = "balanced"
    resolution: str = "1080p"
    max_visual_frames: int = 43
    current_frame: int = 1
    recent_video_steps: int = 2
    frames_per_recent_step: int = 6
    mid_history_after_frames: int = 25
    key_evidence_frames: int = 5
    estimated_tokens_per_frame: int = 1570

    @classmethod
    def agility(
        cls,
        *,
        max_visual_frames: int = 43,
        resolution: str = "1080p",
        estimated_tokens_per_frame: int = 1570,
    ) -> "VisionContextPolicy":
        return cls(
            mode="agility",
            resolution=resolution,
            max_visual_frames=max_visual_frames,
            current_frame=1,
            recent_video_steps=0,
            frames_per_recent_step=0,
            mid_history_after_frames=max(0, max_visual_frames - 1),
            key_evidence_frames=0,
            estimated_tokens_per_frame=estimated_tokens_per_frame,
        )

    @property
    def recent_video_frame_budget(self) -> int:
        return self.recent_video_steps * self.frames_per_recent_step

    @property
    def configured_frame_budget(self) -> int:
        return (
            self.current_frame
            + self.recent_video_frame_budget
            + self.mid_history_after_frames
            + self.key_evidence_frames
        )

    @property
    def estimated_visual_tokens(self) -> int:
        return self.configured_frame_budget * self.estimated_tokens_per_frame

    def validate(self) -> None:
        if self.configured_frame_budget > self.max_visual_frames:
            raise ValueError(
                "configured visual frames exceed max_visual_frames: "
                f"{self.configured_frame_budget} > {self.max_visual_frames}"
            )
        if self.recent_video_steps < 0 or self.frames_per_recent_step < 0:
            raise ValueError("recent video policy values must be non-negative")
        if self.mid_history_after_frames < 0 or self.key_evidence_frames < 0:
            raise ValueError("history frame budgets must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "resolution": self.resolution,
            "max_visual_frames": self.max_visual_frames,
            "current_frame": self.current_frame,
            "recent_video_steps": self.recent_video_steps,
            "frames_per_recent_step": self.frames_per_recent_step,
            "mid_history_after_frames": self.mid_history_after_frames,
            "key_evidence_frames": self.key_evidence_frames,
            "estimated_tokens_per_frame": self.estimated_tokens_per_frame,
            "estimated_visual_tokens": self.estimated_visual_tokens,
        }

    def downgraded(self, usage_status: str) -> "VisionContextPolicy":
        if usage_status == CONTEXT_USAGE_NORMAL:
            return self
        if usage_status == CONTEXT_USAGE_WARNING:
            return replace(
                self,
                mid_history_after_frames=min(self.mid_history_after_frames, 15),
            )
        if usage_status == CONTEXT_USAGE_COMPACT:
            return replace(
                self,
                frames_per_recent_step=min(self.frames_per_recent_step, 4),
                mid_history_after_frames=min(self.mid_history_after_frames, 15),
                key_evidence_frames=min(self.key_evidence_frames, 3),
            )
        if usage_status == CONTEXT_USAGE_EMERGENCY:
            return replace(
                self,
                recent_video_steps=min(self.recent_video_steps, 1),
                frames_per_recent_step=min(self.frames_per_recent_step, 1),
                mid_history_after_frames=0,
                key_evidence_frames=min(self.key_evidence_frames, 2),
            )
        raise ValueError(f"unknown context usage status: {usage_status}")


@dataclass(slots=True, frozen=True)
class StepRecord:
    step_id: int
    action: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    before_ref: str | None = None
    after_ref: str | None = None
    video_refs: tuple[str, ...] = ()
    summary: str | None = None
    task_id: str | None = None
    subtask_id: str | None = None
    importance: str = "normal"
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_key_evidence(self) -> bool:
        return bool(KEY_EVIDENCE_TAGS.intersection(self.tags)) or (
            self.importance == "key"
        )

    def selected_video_refs(self, limit: int) -> tuple[str, ...]:
        if limit <= 0:
            return ()
        return self.video_refs[:limit]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "task_id": self.task_id,
            "subtask_id": self.subtask_id,
            "action": self.action,
            "result": self.result,
            "summary": self.summary,
            "before_ref": self.before_ref,
            "after_ref": self.after_ref,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class LongHistorySummary:
    completed: tuple[str, ...] = ()
    failed_attempts: tuple[str, ...] = ()
    do_not_repeat: tuple[str, ...] = ()
    known_targets: dict[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed": list(self.completed),
            "failed_attempts": list(self.failed_attempts),
            "do_not_repeat": list(self.do_not_repeat),
            "known_targets": self.known_targets,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(slots=True, frozen=True)
class CompactionApplication:
    steps: tuple[StepRecord, ...]
    long_history_summary: LongHistorySummary
    merged_step_ids: tuple[int, ...] = ()
    dropped_visual_step_ids: tuple[int, ...] = ()
    kept_visual_step_ids: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_summary_dict() for step in self.steps],
            "long_history_summary": self.long_history_summary.to_dict(),
            "merged_step_ids": list(self.merged_step_ids),
            "dropped_visual_step_ids": list(self.dropped_visual_step_ids),
            "kept_visual_step_ids": list(self.kept_visual_step_ids),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class PromptContextSelection:
    policy: VisionContextPolicy
    current_frame_ref: str | None
    recent_video_steps: tuple[StepRecord, ...]
    mid_history_after_frames: tuple[StepRecord, ...]
    key_evidence_frames: tuple[StepRecord, ...]
    long_history_summary: LongHistorySummary

    @property
    def selected_frame_count(self) -> int:
        current = 1 if self.current_frame_ref else 0
        recent = sum(
            len(step.selected_video_refs(self.policy.frames_per_recent_step))
            for step in self.recent_video_steps
        )
        mid = sum(1 for step in self.mid_history_after_frames if step.after_ref)
        evidence = sum(1 for step in self.key_evidence_frames if step.after_ref)
        return current + recent + mid + evidence

    def to_context_payload(self) -> dict[str, Any]:
        return {
            "vision_context_policy": self.policy.to_dict(),
            "selected_frame_count": self.selected_frame_count,
            "current_frame": self.current_frame_ref,
            "recent_video_history": [
                {
                    **step.to_summary_dict(),
                    "video_refs": list(
                        step.selected_video_refs(
                            self.policy.frames_per_recent_step
                        )
                    ),
                }
                for step in self.recent_video_steps
            ],
            "mid_history_after_frames": [
                step.to_summary_dict() for step in self.mid_history_after_frames
            ],
            "key_evidence_frames": [
                step.to_summary_dict() for step in self.key_evidence_frames
            ],
            "long_history_summary": self.long_history_summary.to_dict(),
        }


@dataclass(slots=True, frozen=True)
class ContextThresholds:
    warning: float = 0.70
    compact: float = 0.85
    emergency: float = 0.95

    def classify(self, usage_ratio: float) -> str:
        if usage_ratio >= self.emergency:
            return CONTEXT_USAGE_EMERGENCY
        if usage_ratio >= self.compact:
            return CONTEXT_USAGE_COMPACT
        if usage_ratio >= self.warning:
            return CONTEXT_USAGE_WARNING
        return CONTEXT_USAGE_NORMAL


class CompactionPlanError(ValueError):
    pass


@dataclass(slots=True)
class ContextManager:
    policy: VisionContextPolicy = field(default_factory=VisionContextPolicy)
    thresholds: ContextThresholds = field(default_factory=ContextThresholds)

    def __post_init__(self) -> None:
        self.policy.validate()

    def usage_status(self, usage_ratio: float) -> str:
        return self.thresholds.classify(usage_ratio)

    def policy_for_usage(self, usage_ratio: float | None) -> VisionContextPolicy:
        if usage_ratio is None:
            return self.policy
        return self.policy.downgraded(self.usage_status(usage_ratio))

    def select_prompt_context(
        self,
        steps: list[StepRecord] | tuple[StepRecord, ...],
        *,
        current_frame_ref: str | None = None,
        long_history_summary: LongHistorySummary | None = None,
        usage_ratio: float | None = None,
    ) -> PromptContextSelection:
        policy = self.policy_for_usage(usage_ratio)
        policy.validate()
        ordered = sorted(steps, key=lambda step: step.step_id)
        recent = tuple(ordered[-policy.recent_video_steps :])
        recent_ids = {step.step_id for step in recent}

        older = [step for step in ordered if step.step_id not in recent_ids]
        mid = tuple(
            step
            for step in reversed(older)
            if step.after_ref is not None
        )[: policy.mid_history_after_frames]
        mid = tuple(reversed(mid))
        selected_ids = recent_ids | {step.step_id for step in mid}

        evidence_candidates = [
            step
            for step in reversed(older)
            if step.step_id not in selected_ids
            and step.after_ref is not None
            and step.is_key_evidence
        ]
        evidence = tuple(
            reversed(evidence_candidates[: policy.key_evidence_frames])
        )

        selection = PromptContextSelection(
            policy=policy,
            current_frame_ref=current_frame_ref,
            recent_video_steps=recent,
            mid_history_after_frames=mid,
            key_evidence_frames=evidence,
            long_history_summary=long_history_summary or LongHistorySummary(),
        )
        if selection.selected_frame_count > policy.max_visual_frames:
            raise ValueError(
                "selected frames exceed visual budget: "
                f"{selection.selected_frame_count} > {policy.max_visual_frames}"
            )
        return selection

    def validate_compaction_plan(
        self,
        plan: dict[str, Any],
        *,
        available_step_ids: set[int],
        current_step_id: int | None = None,
    ) -> None:
        for item in plan.get("merge_steps", ()) or ():
            steps = item.get("steps", ())
            self._validate_step_ids(
                steps,
                available_step_ids=available_step_ids,
                current_step_id=current_step_id,
                allow_current=bool(item.get("allow_current_step", False)),
            )
            summary = str(item.get("summary", "")).strip()
            if len(summary) > 500:
                raise CompactionPlanError("compaction summary is too long")

        for key in ("keep_visual_steps", "drop_visual_steps"):
            for item in plan.get(key, ()) or ():
                steps = item.get("steps")
                if steps is None and "step" in item:
                    steps = [item["step"]]
                self._validate_step_ids(
                    steps or (),
                    available_step_ids=available_step_ids,
                    current_step_id=current_step_id,
                    allow_current=bool(item.get("allow_current_step", False)),
                )

    def apply_compaction_plan(
        self,
        steps: list[StepRecord] | tuple[StepRecord, ...],
        *,
        long_history_summary: LongHistorySummary | None = None,
        plan: dict[str, Any],
        current_step_id: int | None = None,
    ) -> CompactionApplication:
        available_step_ids = {step.step_id for step in steps}
        self.validate_compaction_plan(
            plan,
            available_step_ids=available_step_ids,
            current_step_id=current_step_id,
        )

        step_by_id = {step.step_id: step for step in steps}
        merged_step_ids: set[int] = set()
        dropped_visual_step_ids: set[int] = set()
        kept_visual_step_ids: set[int] = set()
        summary = long_history_summary or LongHistorySummary()

        completed = list(summary.completed)
        failed_attempts = list(summary.failed_attempts)
        do_not_repeat = list(summary.do_not_repeat)
        evidence_refs = list(summary.evidence_refs)
        known_targets = dict(summary.known_targets)
        compacted_records: list[dict[str, Any]] = []

        for item in plan.get("merge_steps", ()) or ():
            item_step_ids = tuple(int(step) for step in item.get("steps", ()))
            item_summary = str(item.get("summary", "")).strip()
            tags = tuple(str(tag) for tag in item.get("tags", ()) or ())
            keep_evidence = tuple(
                str(ref) for ref in item.get("keep_evidence", ()) or ()
            )
            if item.get("downgrade_to_text", True):
                merged_step_ids.update(item_step_ids)
            if item.get("drop_video", True):
                dropped_visual_step_ids.update(item_step_ids)
            if item_summary:
                if "failed_path" in tags or "do_not_repeat" in tags:
                    failed_attempts.append(item_summary)
                elif "completed" in tags or "task_completed" in tags:
                    completed.append(item_summary)
                else:
                    completed.append(item_summary)
                compacted_records.append(
                    {
                        "steps": list(item_step_ids),
                        "summary": item_summary,
                        "tags": list(tags),
                        "evidence_refs": list(keep_evidence),
                    }
                )
            for ref in keep_evidence:
                if ref not in evidence_refs:
                    evidence_refs.append(ref)

        for item in plan.get("drop_visual_steps", ()) or ():
            item_step_ids = _step_ids_from_plan_item(item)
            dropped_visual_step_ids.update(item_step_ids)

        for item in plan.get("keep_visual_steps", ()) or ():
            item_step_ids = _step_ids_from_plan_item(item)
            kept_visual_step_ids.update(item_step_ids)
            for step_id in item_step_ids:
                after_ref = step_by_id[step_id].after_ref
                if after_ref and after_ref not in evidence_refs:
                    evidence_refs.append(after_ref)

        for item in plan.get("do_not_repeat", ()) or ():
            text = str(item).strip()
            if text:
                do_not_repeat.append(text)

        for target_name, target_payload in (plan.get("known_targets", {}) or {}).items():
            known_targets[str(target_name)] = target_payload

        new_steps: list[StepRecord] = []
        for step in sorted(steps, key=lambda value: value.step_id):
            if step.step_id in merged_step_ids and step.step_id not in kept_visual_step_ids:
                after_ref = step.after_ref
                if after_ref and after_ref not in evidence_refs:
                    evidence_refs.append(after_ref)
                continue
            if step.step_id in dropped_visual_step_ids:
                step = replace(step, video_refs=())
            new_steps.append(step)

        new_summary = LongHistorySummary(
            completed=tuple(_dedupe(completed)),
            failed_attempts=tuple(_dedupe(failed_attempts)),
            do_not_repeat=tuple(_dedupe(do_not_repeat)),
            known_targets=known_targets,
            evidence_refs=tuple(_dedupe(evidence_refs)),
        )
        return CompactionApplication(
            steps=tuple(new_steps),
            long_history_summary=new_summary,
            merged_step_ids=tuple(sorted(merged_step_ids)),
            dropped_visual_step_ids=tuple(sorted(dropped_visual_step_ids)),
            kept_visual_step_ids=tuple(sorted(kept_visual_step_ids)),
            metadata={"compacted_records": compacted_records},
        )

    def _validate_step_ids(
        self,
        steps: Any,
        *,
        available_step_ids: set[int],
        current_step_id: int | None,
        allow_current: bool,
    ) -> None:
        try:
            step_ids = [int(step) for step in steps]
        except (TypeError, ValueError) as exc:
            raise CompactionPlanError("compaction step ids must be integers") from exc
        for step_id in step_ids:
            if step_id not in available_step_ids:
                raise CompactionPlanError(f"unknown compaction step id: {step_id}")
            if (
                current_step_id is not None
                and step_id == current_step_id
                and not allow_current
            ):
                raise CompactionPlanError("current active step cannot be compacted")


def _step_ids_from_plan_item(item: dict[str, Any]) -> tuple[int, ...]:
    steps = item.get("steps")
    if steps is None and "step" in item:
        steps = [item["step"]]
    return tuple(int(step) for step in steps or ())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
