from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class GuiTestCase:
    case_id: str
    instruction: str
    benchmark: str = "manual"
    target_app: str | None = None
    expected_outcome: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_trace_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "instruction": self.instruction,
            "benchmark": self.benchmark,
            "target_app": self.target_app,
            "expected_outcome": self.expected_outcome,
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class GuiTestResult:
    case_id: str
    status: str
    score: float | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_trace_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "score": self.score,
            "reason": self.reason,
            "metadata": self.metadata,
        }
