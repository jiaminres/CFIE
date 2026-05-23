from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class WorkspaceProfile:
    profile_id: str
    name: str
    description: str = ""
    target_apps: tuple[str, ...] = ()
    business_rules: tuple[str, ...] = ()
    reference_image_refs: tuple[str, ...] = ()
    reference_video_refs: tuple[str, ...] = ()
    sop_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "description": self.description,
            "target_apps": list(self.target_apps),
            "business_rules": list(self.business_rules),
            "reference_image_refs": list(self.reference_image_refs),
            "reference_video_refs": list(self.reference_video_refs),
            "sop_refs": list(self.sop_refs),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class GuiAgentTaskSpec:
    task_id: str
    instruction: str
    profile: str = "manual"
    target_app: str | None = None
    expected_outcome: str | None = None
    workspace_profile: WorkspaceProfile | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_trace_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "profile": self.profile,
            "target_app": self.target_app,
            "expected_outcome": self.expected_outcome,
            "workspace_profile": (
                self.workspace_profile.to_dict()
                if self.workspace_profile is not None
                else None
            ),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class GuiAgentResult:
    task_id: str
    status: str
    score: float | None = None
    reason: str | None = None
    final_text: str | None = None
    steps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_trace_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "score": self.score,
            "reason": self.reason,
            "final_text": self.final_text,
            "steps": self.steps,
            "metadata": self.metadata,
        }
