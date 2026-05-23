from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class RewardEvent:
    type: str
    value: float
    step_id: int | None = None
    task_id: str | None = None
    subtask_id: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "value": self.value,
            "step_id": self.step_id,
            "task_id": self.task_id,
            "subtask_id": self.subtask_id,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class DiscountedCredit:
    step_id: int
    credit: float

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step_id, "credit": self.credit}


@dataclass(slots=True, frozen=True)
class SubtaskRewardAssignment:
    subtask_id: str
    terminal_step_id: int
    terminal_reward: float
    gamma: float
    credits: tuple[DiscountedCredit, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "terminal_step": self.terminal_step_id,
            "terminal_reward": self.terminal_reward,
            "gamma": self.gamma,
            "discounted_credit": [credit.to_dict() for credit in self.credits],
            "metadata": self.metadata,
        }


def assign_subtask_completion_credit(
    *,
    subtask_id: str,
    step_ids: tuple[int, ...] | list[int],
    terminal_reward: float,
    gamma: float = 0.9,
    metadata: dict[str, Any] | None = None,
) -> SubtaskRewardAssignment:
    if not step_ids:
        raise ValueError("step_ids must not be empty")
    if gamma < 0 or gamma > 1:
        raise ValueError("gamma must be in [0, 1]")

    ordered = tuple(int(step_id) for step_id in step_ids)
    terminal_step_id = ordered[-1]
    credits: list[DiscountedCredit] = []
    for distance, step_id in enumerate(reversed(ordered)):
        credits.append(
            DiscountedCredit(
                step_id=step_id,
                credit=terminal_reward * (gamma**distance),
            )
        )

    return SubtaskRewardAssignment(
        subtask_id=subtask_id,
        terminal_step_id=terminal_step_id,
        terminal_reward=terminal_reward,
        gamma=gamma,
        credits=tuple(credits),
        metadata=dict(metadata or {}),
    )


def transition_reward_event(
    *,
    transition_type: str,
    value: float,
    step_id: int | None = None,
    from_task_id: str | None = None,
    to_task_id: str | None = None,
    reason: str | None = None,
) -> RewardEvent:
    return RewardEvent(
        type=f"transition:{transition_type}",
        value=value,
        step_id=step_id,
        task_id=from_task_id,
        reason=reason,
        metadata={"from_task_id": from_task_id, "to_task_id": to_task_id},
    )
