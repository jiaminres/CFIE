from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from cfie_client import ComputerLoop, find_computer_calls

from cfie_gui_agent.agent_tools import AgentToolCall, find_agent_tool_calls
from cfie_gui_agent.context import ContextManager, StepRecord
from cfie_gui_agent.human_loop import HumanLoopManager, InMemoryHumanChannel
from cfie_gui_agent.jobs import (
    JobBoard,
    JobState,
    PerJobContextStore,
    SubtaskState,
)
from cfie_gui_agent.policy import PolicyStore
from cfie_gui_agent.runtime_context import RuntimeContextBuilder
from cfie_gui_agent.specs import GuiAgentResult, GuiAgentTaskSpec
from cfie_gui_agent.tools import ModelToolRegistry
from cfie_gui_agent.trace import AgentTraceStore
from cfie_gui_agent.verifier import StepVerifier


class ResponseAgent(Protocol):
    def __call__(self, conversation: list[dict[str, Any]]) -> Any:
        ...


@dataclass(slots=True)
class GuiAgentRunner:
    computer_loop: ComputerLoop = field(default_factory=ComputerLoop)
    context_manager: ContextManager = field(default_factory=ContextManager)
    tool_registry: ModelToolRegistry = field(default_factory=ModelToolRegistry)
    step_verifier: StepVerifier = field(default_factory=StepVerifier)
    policy_store: PolicyStore = field(default_factory=PolicyStore)
    trace_store: AgentTraceStore = field(default_factory=AgentTraceStore)
    include_runtime_context: bool = True
    human_loop: HumanLoopManager = field(
        default_factory=lambda: HumanLoopManager(channel=InMemoryHumanChannel())
    )
    max_steps: int = 8

    def run_task(
        self,
        task: GuiAgentTaskSpec,
        agent: ResponseAgent | Callable[[list[dict[str, Any]]], Any],
    ) -> GuiAgentResult:
        job_board = self._build_initial_job_board(task)
        context_store = PerJobContextStore()
        active_job_id = job_board.active_job_id or task.task_id
        step_records: list[StepRecord] = []
        conversation, current_frame_ref = self._initial_conversation(task)
        if self.include_runtime_context:
            conversation.insert(
                0,
                self._runtime_context_message(
                    job_board=job_board,
                    context_store=context_store,
                    active_job_id=active_job_id,
                    current_frame_ref=current_frame_ref,
                ),
            )

        for step in range(1, self.max_steps + 1):
            if self.include_runtime_context:
                conversation[0] = self._runtime_context_message(
                    job_board=job_board,
                    context_store=context_store,
                    active_job_id=active_job_id,
                    current_frame_ref=current_frame_ref,
                )
            response = agent(conversation)
            computer_calls = find_computer_calls(response)
            agent_tool_calls = find_agent_tool_calls(response)

            if not computer_calls and not agent_tool_calls:
                running = job_board.jobs[active_job_id].queues.running
                if running is not None:
                    job_board.jobs[active_job_id].queues.move_running_to("completed")
                return GuiAgentResult(
                    task_id=task.task_id,
                    status="completed",
                    final_text=_extract_response_text(response),
                    steps=step,
                    metadata=self._build_result_metadata(
                        job_board=job_board,
                        context_store=context_store,
                        active_job_id=active_job_id,
                    ),
                )

            for call in computer_calls:
                self.tool_registry.validate_model_tool("computer_use")
                output = self.computer_loop.handle_call(call)
                conversation.append(output.to_openai_dict())
                running = job_board.jobs[active_job_id].queues.running
                record = StepRecord(
                    step_id=len(step_records) + 1,
                    task_id=active_job_id,
                    subtask_id=running.subtask_id if running is not None else None,
                    action=call.to_openai_dict(),
                    result="computer_call_output",
                    before_ref=current_frame_ref,
                    after_ref=output.output.image_url,
                    summary=f"Executed {len(call.actions)} computer action(s).",
                    tags=("computer_use",),
                )
                verification = self.step_verifier.verify(record)
                record.metadata["verification"] = verification.to_dict()
                if not verification.is_ok:
                    record = replace(
                        record,
                        tags=tuple(
                            dict.fromkeys((*record.tags, verification.status))
                        ),
                    )
                step_records.append(record)
                context_store.append_step(active_job_id, record)
                self.trace_store.record_step(record)
                current_frame_ref = output.output.image_url

            for call in agent_tool_calls:
                running_before = job_board.jobs[active_job_id].queues.running
                output = self._handle_agent_tool_call(call, job_board=job_board)
                running = job_board.jobs[active_job_id].queues.running
                subtask_id = (
                    output.get("subtask_id")
                    or (running_before.subtask_id if running_before is not None else None)
                    or (running.subtask_id if running is not None else None)
                )
                conversation.append(call.to_output_dict(output))
                record = StepRecord(
                    step_id=len(step_records) + 1,
                    task_id=active_job_id,
                    subtask_id=subtask_id,
                    action={
                        "type": "agent_tool",
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                    result=str(output.get("status", "tool_output")),
                    summary=f"Handled agent tool {call.name}.",
                    tags=("agent_tool", call.name),
                    metadata={"output": output},
                )
                step_records.append(record)
                context_store.append_step(active_job_id, record)
                self.trace_store.record_step(record)
                if call.name == "update_constraints":
                    self.trace_store.record_policy_update(output)

        if job_board.jobs[active_job_id].queues.running is not None:
            job_board.jobs[active_job_id].queues.move_running_to("failed")
        return GuiAgentResult(
            task_id=task.task_id,
            status="max_steps_exceeded",
            reason=f"Exceeded max_steps={self.max_steps}",
            steps=self.max_steps,
            metadata=self._build_result_metadata(
                job_board=job_board,
                context_store=context_store,
                active_job_id=active_job_id,
            )
        )

    def _handle_agent_tool_call(
        self,
        call: AgentToolCall,
        *,
        job_board: JobBoard,
    ) -> dict[str, Any]:
        self.tool_registry.validate_model_tool_call(call.name, call.arguments)
        active_job_id = job_board.active_job_id
        if active_job_id is None:
            return {"status": "rejected", "reason": "no active job"}
        job = job_board.require_job(active_job_id)

        if call.name == "finish_subtask":
            running = job.queues.running
            if running is None:
                return {"status": "ignored", "reason": "no running subtask"}
            completed = job.queues.move_running_to("completed")
            return {
                "status": "accepted",
                "subtask_id": completed.subtask_id,
                "reason": call.arguments.get("completion_reason")
                or call.arguments.get("reason"),
            }

        if call.name == "report_blocked":
            running = job.queues.running
            if running is None:
                return {"status": "ignored", "reason": "no running subtask"}
            blocked = job.queues.move_running_to("blocked")
            return {
                "status": "accepted",
                "subtask_id": blocked.subtask_id,
                "reason": call.arguments.get("blocked_reason")
                or call.arguments.get("reason"),
            }

        if call.name == "request_human_help":
            request = self.human_loop.request_help(
                question=str(call.arguments.get("question", "")).strip()
                or "Human help requested.",
                task_id=job.queues.running.subtask_id if job.queues.running else None,
                evidence_refs=tuple(call.arguments.get("evidence_refs", ()) or ()),
                risk_reason=call.arguments.get("risk_reason"),
                proposed_action=call.arguments.get("proposed_action"),
                allowed_reply_format=call.arguments.get("allowed_reply_format"),
                urgency=str(call.arguments.get("urgency", "normal")),
                metadata={
                    "job_id": active_job_id,
                    "tool_call_id": call.call_id,
                },
            )
            if job.queues.running is not None:
                job.queues.move_running_to(
                    "waiting_human",
                    human_request_id=request.request_id,
                )
            return {
                "status": "waiting_human",
                "request_id": request.request_id,
                "job_id": active_job_id,
            }

        if call.name == "ask_replan":
            return {
                "status": "proposal_recorded",
                "job_id": active_job_id,
                "suggested_transition": call.arguments.get("suggested_transition"),
                "reason": call.arguments.get("reason"),
            }

        if call.name == "update_constraints":
            update = self.policy_store.apply_update(
                summary=str(call.arguments.get("summary", "")).strip(),
                constraints=call.arguments.get("constraints", {}),
                reason=call.arguments.get("reason"),
                source="model",
                metadata={
                    "job_id": active_job_id,
                    "tool_call_id": call.call_id,
                },
            )
            return {
                "status": "accepted",
                "update_id": update.update_id,
                "rules_added": len(update.rules),
                "policy_update": update.to_dict(),
            }

        if call.name in {"read_image", "read_video_clip", "query_memory"}:
            return {
                "status": "deferred",
                "reason": "backend not connected in minimal runner",
                "tool": call.name,
            }

        return {"status": "ignored", "reason": f"tool not handled: {call.name}"}

    def _build_initial_job_board(self, task: GuiAgentTaskSpec) -> JobBoard:
        workspace_profile = (
            task.workspace_profile.to_dict()
            if task.workspace_profile is not None
            else None
        )
        target_apps = list(task.workspace_profile.target_apps) if (
            task.workspace_profile is not None
        ) else []
        if task.target_app and task.target_app not in target_apps:
            target_apps.insert(0, task.target_app)
        if not target_apps:
            target_apps = [task.target_app or task.task_id]
        target_app = task.target_app or target_apps[0]

        board = JobBoard()
        for app_name in target_apps:
            board.add_job(
                JobState(
                    job_id=f"job:{app_name}",
                    target_app=app_name,
                    goal=(
                        task.instruction
                        if app_name == target_app
                        else f"Handle workspace events for {app_name}."
                    ),
                    metadata={
                        "root_task_id": task.task_id,
                        "profile": task.profile,
                        "expected_outcome": task.expected_outcome,
                        "workspace_profile": workspace_profile,
                        **task.metadata,
                    },
                )
            )
        if len(target_apps) > 1:
            board.add_job(
                JobState(
                    job_id="job:monitor",
                    target_app="monitor",
                    goal="Monitor target APPs and create Subtasks under matching Jobs.",
                    priority=-100,
                    metadata={
                        "root_task_id": task.task_id,
                        "target_apps": target_apps,
                        "workspace_profile": workspace_profile,
                    },
                )
            )
        board.active_job_id = f"job:{target_app}"
        subtask = SubtaskState(
            subtask_id=f"subtask:{task.task_id}:root",
            job_id=f"job:{target_app}",
            goal=task.instruction,
            status="running",
            success_condition=task.expected_outcome,
            metadata={
                "root_task_id": task.task_id,
                "profile": task.profile,
                "target_app": task.target_app,
                "workspace_profile": workspace_profile,
                **task.metadata,
            },
        )
        board.add_subtask(subtask)
        return board

    def _initial_conversation(
        self,
        task: GuiAgentTaskSpec,
    ) -> tuple[list[dict[str, Any]], str | None]:
        screenshot = self.computer_loop.screen.screenshot()
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": task.instruction},
            {
                "type": "input_image",
                "image_url": screenshot.image_url,
                "detail": "auto",
            },
        ]
        return (
            [{"type": "message", "role": "user", "content": content}],
            screenshot.image_url,
        )

    def _runtime_context_message(
        self,
        *,
        job_board: JobBoard,
        context_store: PerJobContextStore,
        active_job_id: str,
        current_frame_ref: str | None = None,
    ) -> dict[str, Any]:
        runtime_context = RuntimeContextBuilder(
            context_manager=self.context_manager,
            tool_registry=self.tool_registry,
            policy_store=self.policy_store,
        ).build(
            job_board=job_board,
            context_store=context_store,
            active_job_id=active_job_id,
            current_frame_ref=current_frame_ref,
        ).to_dict()
        runtime_context_json = json.dumps(
            runtime_context,
            ensure_ascii=False,
            sort_keys=True,
        )
        return {
            "type": "message",
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Runtime context JSON. Follow this state instead of "
                        "inventing task queues.\n"
                        f"{runtime_context_json}"
                    ),
                }
            ],
        }

    def _build_result_metadata(
        self,
        *,
        job_board: JobBoard,
        context_store: PerJobContextStore,
        active_job_id: str,
    ) -> dict[str, Any]:
        step_records = context_store.get_steps(active_job_id)
        runtime_context = RuntimeContextBuilder(
            context_manager=self.context_manager,
            tool_registry=self.tool_registry,
            policy_store=self.policy_store,
        ).build(
            job_board=job_board,
            context_store=context_store,
            active_job_id=active_job_id,
        ).to_dict()
        return {
            "job_board": job_board.to_global_context(),
            "jobs": {
                job_id: job.to_dict()
                for job_id, job in job_board.jobs.items()
            },
            "active_job_id": active_job_id,
            "step_records": [step.to_summary_dict() for step in step_records],
            "runtime_context": runtime_context,
            "prompt_context": runtime_context["prompt_context"],
            "job_context": runtime_context["job_context"],
            "model_tools": runtime_context["model_tools"],
            "policy": runtime_context["policy"],
            "trace": self.trace_store.to_dict(),
        }


def _extract_response_text(response: Any) -> str | None:
    items = _read_field(response, "output", response)
    texts: list[str] = []
    for item in items or ():
        if _read_field(item, "type") != "message":
            continue
        for content in _read_field(item, "content", ()) or ():
            content_type = _read_field(content, "type")
            if content_type in {"output_text", "text"}:
                text = _read_field(content, "text", None)
                if text:
                    texts.append(str(text))
    return "\n".join(texts) if texts else None


def _read_field(value: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field_name, default)
    return getattr(value, field_name, default)
