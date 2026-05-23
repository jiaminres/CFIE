from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cfie_gui_agent.context import ContextManager
from cfie_gui_agent.jobs import JobBoard, PerJobContextStore
from cfie_gui_agent.macros import ActionMacroRegistry
from cfie_gui_agent.policy import PolicyStore
from cfie_gui_agent.tools import ModelToolRegistry


@dataclass(slots=True, frozen=True)
class RuntimeContext:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.payload


@dataclass(slots=True)
class RuntimeContextBuilder:
    context_manager: ContextManager
    tool_registry: ModelToolRegistry
    policy_store: PolicyStore
    action_macros: ActionMacroRegistry | None = None

    def build(
        self,
        *,
        job_board: JobBoard,
        context_store: PerJobContextStore,
        active_job_id: str,
        current_frame_ref: str | None = None,
        usage_ratio: float | None = None,
    ) -> RuntimeContext:
        job = job_board.require_job(active_job_id)
        job_context = context_store.to_job_context_payload(
            active_job_id,
            manager=self.context_manager,
            current_frame_ref=current_frame_ref,
            usage_ratio=usage_ratio,
        )
        running = job.queues.running
        return RuntimeContext(
            payload={
                "job_board": job_board.to_global_context(),
                "active_job": job.to_summary(),
                "active_subtask": running.to_dict() if running is not None else None,
                "job_context": job_context,
                "prompt_context": job_context["prompt_context"],
                "policy": self.policy_store.to_context_payload(),
                "model_tools": list(self.tool_registry.allowed_tool_names),
                "action_macros": (
                    self.action_macros.to_context_payload()
                    if self.action_macros is not None
                    else {"macros": []}
                ),
            }
        )
