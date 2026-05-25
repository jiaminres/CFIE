from __future__ import annotations

import json
import re
import base64
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

from PIL import Image

from cfie_client import ComputerAction, ComputerCall, ComputerLoop, find_computer_calls

from cfie_gui_agent.agent_tools import (
    AgentToolError,
    AgentToolCall,
    find_agent_tool_calls,
    find_computer_tool_calls,
    normalize_response_tool_calls,
)
from cfie_gui_agent.context import ContextManager, StepRecord, VisionContextPolicy
from cfie_gui_agent.human_loop import HumanLoopManager, InMemoryHumanChannel
from cfie_gui_agent.jobs import (
    JobBoard,
    JobState,
    PerJobContextStore,
    SubtaskState,
)
from cfie_gui_agent.macros import ActionMacroError, ActionMacroRegistry
from cfie_gui_agent.navigation import NavigationPlanError, NavigationPlanner, NavigationRequest
from cfie_gui_agent.policy import PolicyStore
from cfie_gui_agent.runtime_context import RuntimeContextBuilder
from cfie_gui_agent.specs import GuiAgentResult, GuiAgentTaskSpec
from cfie_gui_agent.tools import ModelToolRegistry, ToolRegistryError
from cfie_gui_agent.trace import AgentTraceStore
from cfie_gui_agent.verifier import StepVerifier, VERIFICATION_REPEATED_ACTION


class ResponseAgent(Protocol):
    def __call__(self, conversation: list[dict[str, Any]]) -> Any:
        ...


@dataclass(slots=True)
class GuiAgentRunner:
    computer_loop: ComputerLoop = field(default_factory=ComputerLoop)
    context_manager: ContextManager = field(
        default_factory=lambda: ContextManager(policy=VisionContextPolicy.agility())
    )
    tool_registry: ModelToolRegistry = field(default_factory=ModelToolRegistry)
    step_verifier: StepVerifier = field(default_factory=StepVerifier)
    policy_store: PolicyStore = field(default_factory=PolicyStore)
    trace_store: AgentTraceStore = field(default_factory=AgentTraceStore)
    action_macros: ActionMacroRegistry = field(default_factory=ActionMacroRegistry)
    navigation_planner: NavigationPlanner = field(default_factory=NavigationPlanner)
    include_runtime_context: bool = True
    image_detail: str | None = "low"
    auto_human_repeated_action_threshold: int = 3
    response_text_warning_chars: int = 800
    response_json_warning_chars: int = 12000
    response_latency_warning_seconds: float = 10.0
    refresh_runtime_context_each_step: bool = False
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
        model_response_metrics: list[dict[str, Any]] = []
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
            if self.include_runtime_context and self.refresh_runtime_context_each_step:
                conversation[0] = self._runtime_context_message(
                    job_board=job_board,
                    context_store=context_store,
                    active_job_id=active_job_id,
                    current_frame_ref=current_frame_ref,
                )
            response_started = perf_counter()
            response = normalize_response_tool_calls(agent(conversation))
            response_latency = perf_counter() - response_started
            response_metrics = _build_model_response_metrics(
                response,
                step=step,
                latency_seconds=response_latency,
                text_warning_chars=self.response_text_warning_chars,
                json_warning_chars=self.response_json_warning_chars,
                latency_warning_seconds=self.response_latency_warning_seconds,
                input_text_preview=_conversation_text_preview(conversation),
            )
            model_response_metrics.append(response_metrics)
            self.trace_store.record("model_response", response_metrics)
            try:
                computer_calls = (
                    *find_computer_calls(response),
                    *find_computer_tool_calls(response),
                )
                agent_tool_calls = find_agent_tool_calls(response)
            except AgentToolError as exc:
                self.trace_store.record(
                    "tool_call_parse_retry",
                    {
                        "step": step,
                        "reason": "invalid_tool_arguments",
                        "error": str(exc),
                    },
                )
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "The previous tool call could not be parsed "
                                    f"by the local harness: {exc}. Return one "
                                    "valid complete tool call now. Use "
                                    "keypress for keyboard actions."
                                ),
                            }
                        ],
                    }
                )
                continue
            human_help_override = _human_help_call_from_text(
                step=step,
                text=_extract_response_text(response) or "",
                current_frame_ref=current_frame_ref,
            )
            if human_help_override is not None and computer_calls:
                self.trace_store.record(
                    "tool_call_safety_override",
                    {
                        "step": step,
                        "reason": "human_help_intent_with_computer_action",
                        "preview": (_extract_response_text(response) or "")[:500],
                    },
                )
                computer_calls = ()
                agent_tool_calls = (human_help_override, *agent_tool_calls)

            if not computer_calls and not agent_tool_calls:
                final_text = _extract_response_text(response)
                if _looks_like_incomplete_tool_call(final_text):
                    self.trace_store.record(
                        "tool_call_parse_retry",
                        {
                            "step": step,
                            "reason": "incomplete_or_unclosed_tool_call",
                            "preview": (final_text or "")[:500],
                        },
                    )
                    conversation.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        "Your previous tool call was incomplete "
                                        "or truncated, so the harness could not "
                                        "execute it. Return exactly one complete "
                                        "tool call, or return a short final answer "
                                        "only if the task is genuinely finished."
                                    ),
                                }
                            ],
                        }
                    )
                    continue
                if _looks_like_unexecuted_tool_plan(
                    final_text,
                    self.tool_registry.allowed_tool_names,
                ):
                    self.trace_store.record(
                        "tool_call_parse_retry",
                        {
                            "step": step,
                            "reason": "prose_plan_without_tool_call",
                            "preview": (final_text or "")[:500],
                        },
                    )
                    conversation.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        "You wrote a plan or thinking text, but "
                                        "the task is still active and no tool "
                                        "call was executed. Return exactly one "
                                        "complete tool call now. Do not write "
                                        "Thinking Process, Plan, Analysis, or "
                                        "any explanatory prose before the tool "
                                        "call."
                                    ),
                                }
                            ],
                        }
                    )
                    continue
                _maybe_record_final_workflow_result(
                    final_text,
                    job=job_board.jobs[active_job_id],
                    trace_store=self.trace_store,
                )
                running = job_board.jobs[active_job_id].queues.running
                if running is not None:
                    job_board.jobs[active_job_id].queues.move_running_to("completed")
                return GuiAgentResult(
                    task_id=task.task_id,
                    status="completed",
                    final_text=final_text,
                    steps=step,
                    metadata=self._build_result_metadata(
                        job_board=job_board,
                        context_store=context_store,
                        active_job_id=active_job_id,
                        model_response_metrics=model_response_metrics,
                    ),
            )

            for call in computer_calls:
                self.tool_registry.validate_model_tool("computer_use")
                try:
                    output = self.computer_loop.handle_call(call)
                except Exception as exc:
                    before_ref = current_frame_ref
                    screenshot_ref = current_frame_ref
                    try:
                        screenshot = self.computer_loop.screen.screenshot()
                        screenshot_ref = screenshot.image_url
                        current_frame_ref = screenshot.image_url
                    except Exception:
                        screenshot = None
                    error_text = (
                        "computer_use was rejected by the local harness: "
                        f"{type(exc).__name__}: {exc}. Use coordinates inside "
                        f"the current screenshot ({_screen_bounds_text(self.computer_loop.screen)}), "
                        "or call request_human_help if "
                        "the UI cannot be operated safely."
                    )
                    conversation.append(
                        _screen_observation_message(
                            text=error_text,
                            image_url=(
                                screenshot.image_url
                                if screenshot is not None
                                else None
                            ),
                            image_detail=self.image_detail,
                        )
                    )
                    running = job_board.jobs[active_job_id].queues.running
                    record = StepRecord(
                        step_id=len(step_records) + 1,
                        task_id=active_job_id,
                        subtask_id=(
                            running.subtask_id if running is not None else None
                        ),
                        action=call.to_openai_dict(),
                        result="rejected",
                        before_ref=before_ref,
                        after_ref=screenshot_ref,
                        summary=error_text,
                        tags=("computer_use", "rejected"),
                        metadata={
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    step_records.append(record)
                    context_store.append_step(active_job_id, record)
                    self.trace_store.record_step(record)
                    continue
                conversation.append(output.to_openai_dict())
                conversation.append(_coordinate_bounds_message(self.computer_loop.screen))
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
                if self._should_auto_request_human_on_verification(verification):
                    request = self.human_loop.request_help(
                        question=(
                            "The same computer action repeated without useful "
                            "progress. Please inspect the current screen and "
                            "decide how the Agent should continue."
                        ),
                        task_id=running.subtask_id if running is not None else None,
                        evidence_refs=(current_frame_ref,),
                        risk_reason=verification.status,
                        proposed_action="Human should unblock the current UI state.",
                        urgency="normal",
                        metadata={
                            "job_id": active_job_id,
                            "step_id": record.step_id,
                            "verification": verification.to_dict(),
                        },
                    )
                    if job_board.jobs[active_job_id].queues.running is not None:
                        job_board.jobs[active_job_id].queues.move_running_to(
                            "waiting_human",
                            human_request_id=request.request_id,
                        )
                    return GuiAgentResult(
                        task_id=task.task_id,
                        status="waiting_human",
                        reason=(
                            "Harness requested human input after repeated "
                            "computer actions."
                        ),
                        steps=step,
                        metadata=self._build_result_metadata(
                            job_board=job_board,
                            context_store=context_store,
                            active_job_id=active_job_id,
                            model_response_metrics=model_response_metrics,
                        ),
                    )

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
                if (
                    call.name == "set_app_viewport"
                    and output.get("status") == "accepted"
                ):
                    try:
                        screenshot = self.computer_loop.screen.screenshot()
                    except Exception:
                        screenshot = None
                    if screenshot is not None:
                        current_frame_ref = screenshot.image_url
                        conversation.append(
                            _screen_observation_message(
                                text=(
                                    "Viewport updated. The next image is the "
                                    "current cropped APP screenshot. Use this "
                                    "image coordinate space for computer_use."
                                ),
                                image_url=screenshot.image_url,
                                image_detail=self.image_detail,
                            )
                        )
                if (
                    call.name == "submit_current_input"
                    and output.get("status") == "accepted"
                ):
                    screenshot_ref = str(output.get("screenshot_image_url") or "")
                    if screenshot_ref:
                        current_frame_ref = screenshot_ref
                        conversation.append(
                            _screen_observation_message(
                                text=(
                                    "Input submitted. The next image is the "
                                    "current APP screenshot after submission."
                                ),
                                image_url=screenshot_ref,
                                image_detail=self.image_detail,
                            )
                        )
                if (
                    call.name == "finish_subtask"
                    and output.get("status") == "accepted"
                ):
                    return GuiAgentResult(
                        task_id=task.task_id,
                        status="completed",
                        reason=str(output.get("reason", "") or ""),
                        steps=step,
                        metadata=self._build_result_metadata(
                            job_board=job_board,
                            context_store=context_store,
                            active_job_id=active_job_id,
                            model_response_metrics=model_response_metrics,
                        ),
                    )
                if (
                    call.name == "request_human_help"
                    and output.get("status") == "waiting_human"
                ):
                    return GuiAgentResult(
                        task_id=task.task_id,
                        status="waiting_human",
                        reason="Model requested human input.",
                        steps=step,
                        metadata=self._build_result_metadata(
                            job_board=job_board,
                            context_store=context_store,
                            active_job_id=active_job_id,
                            model_response_metrics=model_response_metrics,
                        ),
                    )
                if (
                    call.name == "report_blocked"
                    and output.get("status") == "accepted"
                ):
                    return GuiAgentResult(
                        task_id=task.task_id,
                        status="blocked",
                        reason=str(output.get("reason", "") or ""),
                        steps=step,
                        metadata=self._build_result_metadata(
                            job_board=job_board,
                            context_store=context_store,
                            active_job_id=active_job_id,
                            model_response_metrics=model_response_metrics,
                        ),
                    )

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
                model_response_metrics=model_response_metrics,
            )
        )

    def _handle_agent_tool_call(
        self,
        call: AgentToolCall,
        *,
        job_board: JobBoard,
    ) -> dict[str, Any]:
        arguments = _normalize_agent_tool_arguments(call.name, call.arguments)
        if "_parse_error" in arguments:
            return {
                "status": "rejected",
                "reason": "tool arguments were not valid JSON",
                "parse_error": arguments.get("_parse_error"),
                "raw_arguments": arguments.get("_raw_arguments"),
                "tool": call.name,
            }
        try:
            self.tool_registry.validate_model_tool_call(call.name, arguments)
        except ToolRegistryError as exc:
            return {
                "status": "rejected",
                "reason": str(exc),
                "tool": call.name,
            }
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
                "reason": arguments.get("completion_reason")
                or arguments.get("reason"),
            }

        if call.name == "report_blocked":
            running = job.queues.running
            if running is None:
                return {"status": "ignored", "reason": "no running subtask"}
            blocked = job.queues.move_running_to("blocked")
            return {
                "status": "accepted",
                "subtask_id": blocked.subtask_id,
                "reason": arguments.get("blocked_reason")
                or arguments.get("reason"),
            }

        if call.name == "request_human_help":
            request = self.human_loop.request_help(
                question=str(arguments.get("question", "")).strip()
                or "Human help requested.",
                task_id=job.queues.running.subtask_id if job.queues.running else None,
                evidence_refs=tuple(arguments.get("evidence_refs", ()) or ()),
                risk_reason=arguments.get("risk_reason"),
                proposed_action=arguments.get("proposed_action"),
                allowed_reply_format=arguments.get("allowed_reply_format"),
                urgency=str(arguments.get("urgency", "normal")),
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
                "suggested_transition": arguments.get("suggested_transition"),
                "reason": arguments.get("reason"),
            }

        if call.name == "update_constraints":
            update = self.policy_store.apply_update(
                summary=str(arguments.get("summary", "")).strip(),
                constraints=arguments.get("constraints", {}),
                reason=arguments.get("reason"),
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

        if call.name == "read_text_file":
            path = Path(str(arguments.get("path", ""))).expanduser()
            max_chars = int(arguments.get("max_chars", 20000))
            if not path.exists() or not path.is_file():
                return {
                    "status": "rejected",
                    "reason": f"file not found: {path}",
                    "tool": call.name,
                }
            text = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
            return {
                "status": "accepted",
                "path": str(path),
                "chars": len(text),
                "text": text,
            }

        if call.name == "append_trace_note":
            payload = {
                "job_id": active_job_id,
                "title": arguments.get("title"),
                "summary": arguments.get("summary", ""),
                "status": arguments.get("status", "recorded"),
                "artifact_refs": arguments.get("artifact_refs", []) or [],
                "metadata": arguments.get("metadata", {}) or {},
            }
            self.trace_store.record("operation", payload)
            return {"status": "accepted", "trace_event": payload}

        if call.name == "set_app_viewport":
            result = _set_screen_viewport(
                self.computer_loop.screen,
                x=int(arguments.get("x", 0)),
                y=int(arguments.get("y", 0)),
                width=int(arguments.get("width", 1)),
                height=int(arguments.get("height", 1)),
                coordinate_space=str(arguments.get("coordinate_space", "screenshot")),
            )
            payload = {
                "job_id": active_job_id,
                "reason": arguments.get("reason", ""),
                **result,
            }
            self.trace_store.record("viewport", payload)
            return payload

        if call.name == "record_workflow_result":
            item_id = arguments.get("item_id")
            item_defaults = _workflow_item_defaults(job, str(item_id or ""))
            payload = {
                "job_id": active_job_id,
                "item_id": item_id,
                "input_text": arguments.get("input_text")
                or item_defaults.get("input_text", ""),
                "expected_output": arguments.get("expected_output")
                or item_defaults.get("expected_output", ""),
                "output_text": arguments.get("output_text", ""),
                "status": arguments.get("status"),
                "latency_seconds": arguments.get("latency_seconds"),
                "artifact_refs": arguments.get("artifact_refs", []) or [],
                "reason": arguments.get("reason", ""),
            }
            self.trace_store.record("workflow_result", payload)
            return {"status": "accepted", "workflow_result": payload}

        if call.name == "read_image":
            return {
                "status": "accepted",
                "image_ref": arguments.get("image_ref"),
                "note": (
                    "The referenced screenshot has already been provided in "
                    "the visual conversation context. Continue by inspecting "
                    "that image directly."
                ),
                "tool": call.name,
            }

        if call.name in {"read_video_clip", "query_memory"}:
            return {
                "status": "deferred",
                "reason": "backend not connected in minimal runner",
                "tool": call.name,
            }

        if call.name == "run_action_macro":
            macro_name = str(arguments.get("macro_name", ""))
            repeat = int(arguments.get("repeat", 1))
            try:
                actions = self.action_macros.expand(macro_name, repeat=repeat)
            except ActionMacroError as exc:
                return {
                    "status": "rejected",
                    "reason": str(exc),
                    "macro_name": macro_name,
                }
            return {
                "status": "accepted",
                "macro_name": macro_name,
                "repeat": repeat,
                "expanded_actions": [
                    action.to_openai_dict() for action in actions
                ],
            }

        if call.name == "submit_current_input":
            method = str(arguments.get("method") or "auto")
            result = _submit_current_input(
                self.computer_loop,
                method=method,
                call_id=call.call_id,
            )
            return {
                **result,
                "reason": arguments.get("reason", ""),
            }

        if call.name == "navigate_to_target":
            try:
                request = NavigationRequest.from_arguments(arguments)
                plan = self.navigation_planner.plan(request)
            except (KeyError, TypeError, ValueError, NavigationPlanError) as exc:
                return {
                    "status": "rejected",
                    "reason": str(exc),
                    "tool": call.name,
                }
            return {
                "status": "planned",
                "navigation_plan": plan.to_dict(),
            }

        return {"status": "ignored", "reason": f"tool not handled: {call.name}"}

    def _should_auto_request_human_on_verification(self, verification: Any) -> bool:
        if self.auto_human_repeated_action_threshold <= 0:
            return False
        return verification.status == VERIFICATION_REPEATED_ACTION

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
        viewport = _screen_viewport_context(self.computer_loop.screen)
        viewport_note = ""
        if viewport:
            viewport_note = (
                "\n\nScreenshot viewport: "
                f"{json.dumps(viewport, ensure_ascii=False, sort_keys=True)}"
            )
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    f"{task.instruction}\n\n"
                    "当前截图坐标系："
                    f"width={screenshot.width}, height={screenshot.height}。"
                    "computer_use 的 x/y 必须使用这个截图坐标系，"
                    "不要使用物理屏幕坐标。"
                    "Qwen VL computer_use 坐标协议：必须在工具调用中写 "
                    'coordinate_space="qwen_normalized_1000"，并使用 0..1000 '
                    "归一化图像坐标；左上角是 (0,0)，右下角是 "
                    "(1000,1000)。不要使用物理屏幕坐标。"
                    "如果有效应用区域只占截图的一部分，先调用 "
                    "set_app_viewport 写入应用轮廓；后续截图和点击坐标"
                    "都会使用裁剪后的应用视口。"
                    f"{viewport_note}"
                ),
            },
            {
                "type": "input_image",
                "image_url": screenshot.image_url,
                "detail": self.image_detail,
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
        runtime_context = self._model_runtime_context_payload(
            job_board=job_board,
            context_store=context_store,
            active_job_id=active_job_id,
            current_frame_ref=current_frame_ref,
        )
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
                        "Runtime context JSON for this GUI Agent turn. Follow "
                        "this state instead of inventing task queues.\n"
                        "Efficiency policy: keep every response concise. Do not "
                        "describe screenshots or videos unless that description is "
                        "the actual task result. Prefer tool calls over prose. "
                        "Do not emit long thinking text. Do not write visible "
                        "'previous state' or 'next action' text before a tool call; "
                        "when an action is needed, emit the tool call only. "
                        "For record_workflow_result, "
                        "send only item_id, output_text, status, reason, and "
                        "artifact_refs unless other fields are truly needed. "
                        "If thinking is enabled, keep it to two short clauses: "
                        "previous state and next action. Keep each new turn's "
                        "extra text minimal so prefix cache can reuse the stable "
                        "conversation prefix. Historical screenshots already in "
                        "the conversation should stay stable and be reused by KV "
                        "cache; each new step should add only one operation-result "
                        "keyframe plus a small tool/result message. Use "
                        "set_app_viewport once the APP region is clear, so later "
                        "screenshots crop to the active application and each turn's "
                        "new prefill delta stays near the 2096/4192-token budget.\n"
                        f"{runtime_context_json}"
                    ),
                }
            ],
        }

    def _model_runtime_context_payload(
        self,
        *,
        job_board: JobBoard,
        context_store: PerJobContextStore,
        active_job_id: str,
        current_frame_ref: str | None,
    ) -> dict[str, Any]:
        job = job_board.require_job(active_job_id)
        running = job.queues.running
        recent_steps = context_store.get_steps(active_job_id)[-3:]
        return {
            "active_job_id": active_job_id,
            "active_app": job.target_app,
            "job_status": job.status,
            "queue_counts": job.queues.counts(),
            "active_subtask": (
                {
                    "subtask_id": running.subtask_id,
                    "goal": _short_text(running.goal, 600),
                    "status": running.status,
                    "success_condition": _short_text(
                        running.success_condition or "",
                        240,
                    ),
                }
                if running is not None
                else None
            ),
            "current_frame": current_frame_ref,
            "screen_viewport": _screen_viewport_context(self.computer_loop.screen),
            "context_budget": self.context_manager.policy.to_dict(),
            "recent_steps": [
                {
                    "step_id": step.step_id,
                    "summary": _short_text(step.summary or "", 160),
                    "result": step.result,
                    "tags": list(step.tags),
                    "after_ref": step.after_ref,
                }
                for step in recent_steps
            ],
            "policy_rules": [
                _short_text(rule.get("summary") or rule.get("text") or str(rule), 180)
                for rule in self.policy_store.to_context_payload().get("rules", [])[-5:]
            ],
            "model_tools": list(self.tool_registry.allowed_tool_names),
            "action_macros": (
                self.action_macros.to_context_payload()
                if self.action_macros is not None
                else {"macros": []}
            ),
        }

    def _build_result_metadata(
        self,
        *,
        job_board: JobBoard,
        context_store: PerJobContextStore,
        active_job_id: str,
        model_response_metrics: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        step_records = context_store.get_steps(active_job_id)
        runtime_context = RuntimeContextBuilder(
            context_manager=self.context_manager,
            tool_registry=self.tool_registry,
            policy_store=self.policy_store,
            action_macros=self.action_macros,
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
            "model_response_metrics": model_response_metrics or [],
            "human_requests": list(
                self.human_loop.list_requests(include_completed=True)
            ),
        }


def _normalize_agent_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(arguments)
    list_fields_by_tool = {
        "request_human_help": ("evidence_refs",),
        "finish_subtask": ("evidence_refs",),
        "report_blocked": ("evidence_refs",),
        "append_trace_note": ("artifact_refs",),
        "record_workflow_result": ("artifact_refs",),
    }
    for field_name in list_fields_by_tool.get(tool_name, ()):
        if field_name in normalized:
            normalized[field_name] = _as_string_list(normalized[field_name])
    int_fields_by_tool = {
        "set_app_viewport": ("x", "y", "width", "height"),
        "read_text_file": ("max_chars",),
        "run_action_macro": ("repeat",),
    }
    for field_name in int_fields_by_tool.get(tool_name, ()):
        if field_name in normalized:
            normalized[field_name] = _as_int_or_original(normalized[field_name])
    return normalized


def _as_int_or_original(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            number = float(stripped)
        except ValueError:
            return value
        if number.is_integer():
            return int(number)
    return value


def _as_string_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return [value]
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item not in (None, "")]
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    if isinstance(value, tuple):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _short_text(value: str, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _screen_viewport_context(screen: Any) -> dict[str, Any]:
    viewport_fn = getattr(screen, "viewport_context", None)
    if callable(viewport_fn):
        return dict(viewport_fn())
    size_fn = getattr(screen, "size", None)
    if callable(size_fn):
        width, height = size_fn()
        return {
            "screenshot_width": width,
            "screenshot_height": height,
            "crop_x": None,
            "crop_y": None,
            "crop_width": None,
            "crop_height": None,
        }
    return {}


def _screen_bounds_text(screen: Any) -> str:
    try:
        width, height = screen.size()
    except Exception:
        return "unknown size"
    return (
        f"width={width}, height={height}; "
        f"valid x is 0..{max(0, width - 1)}, "
        f"valid y is 0..{max(0, height - 1)}"
    )


def _coordinate_bounds_message(screen: Any) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": (
                    "Coordinate reminder for the next computer_use: "
                    f"{_screen_bounds_text(screen)}. For Qwen VL, set "
                    'coordinate_space="qwen_normalized_1000" and use 0..1000 '
                    "normalized image coordinates. Do not use physical screen "
                    "coordinates."
                ),
            }
        ],
    }


def _screen_observation_message(
    *,
    text: str,
    image_url: str | None,
    image_detail: str | None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
    if image_url:
        image_part: dict[str, Any] = {
            "type": "input_image",
            "image_url": image_url,
        }
        if image_detail is not None:
            image_part["detail"] = image_detail
        content.append(image_part)
    return {"type": "message", "role": "user", "content": content}


def _set_screen_viewport(
    screen: Any,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    coordinate_space: str,
) -> dict[str, Any]:
    if width <= 0 or height <= 0:
        return {
            "status": "rejected",
            "reason": "viewport width and height must be positive",
        }
    if coordinate_space not in {"screenshot", "physical"}:
        return {
            "status": "rejected",
            "reason": "coordinate_space must be screenshot or physical",
        }
    crop_box = (int(x), int(y), int(width), int(height))
    if coordinate_space == "screenshot":
        crop_box = _viewport_screenshot_to_physical(screen, crop_box)
    set_crop_box = getattr(screen, "set_crop_box", None)
    if callable(set_crop_box):
        set_crop_box(crop_box)
    elif hasattr(screen, "crop_box"):
        setattr(screen, "crop_box", crop_box)
    else:
        return {
            "status": "rejected",
            "reason": "screen capture backend does not support viewport cropping",
        }
    return {
        "status": "accepted",
        "tool": "set_app_viewport",
        "coordinate_space": coordinate_space,
        "crop_box": list(crop_box),
        "screen_viewport": _screen_viewport_context(screen),
    }


def _submit_current_input(
    computer_loop: ComputerLoop,
    *,
    method: str,
    call_id: str,
) -> dict[str, Any]:
    normalized_method = method if method in {"auto", "click_send_button", "enter"} else "auto"
    if normalized_method == "enter":
        return _execute_submit_call(
            computer_loop,
            call_id=call_id,
            actions=(ComputerAction(type="keypress", keys=("ENTER",)),),
            method="enter",
            detected=False,
        )

    try:
        screenshot = computer_loop.screen.screenshot()
        image = _decode_image_url_to_pil(screenshot.image_url)
        point = _detect_chat_send_point(image)
    except Exception as exc:
        point = None
        detect_error = f"{type(exc).__name__}: {exc}"
    else:
        detect_error = ""

    if point is not None:
        x, y = point
        model_x, model_y = _screenshot_point_to_loop_model_point(
            computer_loop,
            x=x,
            y=y,
        )
        return _execute_submit_call(
            computer_loop,
            call_id=call_id,
            actions=(
                ComputerAction(
                    type="click",
                    x=model_x,
                    y=model_y,
                    button="left",
                ),
            ),
            method="click_send_button",
            detected=True,
            screenshot_point=(x, y),
        )

    if normalized_method == "click_send_button":
        return {
            "status": "rejected",
            "tool": "submit_current_input",
            "method": normalized_method,
            "reason": f"send button was not detected: {detect_error}",
        }
    return _execute_submit_call(
        computer_loop,
        call_id=call_id,
        actions=(ComputerAction(type="keypress", keys=("ENTER",)),),
        method="enter",
        detected=False,
        detection_error=detect_error,
    )


def _execute_submit_call(
    computer_loop: ComputerLoop,
    *,
    call_id: str,
    actions: tuple[ComputerAction, ...],
    method: str,
    detected: bool,
    screenshot_point: tuple[int, int] | None = None,
    detection_error: str = "",
) -> dict[str, Any]:
    output = computer_loop.handle_call(
        ComputerCall(
            call_id=f"{call_id}:submit_current_input",
            actions=actions,
            coordinate_space=getattr(
                computer_loop,
                "model_coordinate_mode",
                "screenshot",
            ),
        )
    )
    payload: dict[str, Any] = {
        "status": "accepted",
        "tool": "submit_current_input",
        "method": method,
        "detected_send_button": detected,
        "screenshot_image_url": output.output.image_url,
    }
    if screenshot_point is not None:
        payload["screenshot_point"] = list(screenshot_point)
    if detection_error:
        payload["detection_error"] = detection_error
    return payload


def _screenshot_point_to_loop_model_point(
    computer_loop: ComputerLoop,
    *,
    x: int,
    y: int,
) -> tuple[int, int]:
    mode = getattr(computer_loop, "model_coordinate_mode", "screenshot")
    if mode != "qwen_normalized_1000":
        return (x, y)
    width, height = computer_loop.screen.size()
    if width <= 0 or height <= 0:
        return (x, y)
    return (
        max(0, min(1000, int(round(x * 1000 / width)))),
        max(0, min(1000, int(round(y * 1000 / height)))),
    )


def _decode_image_url_to_pil(image_url: str) -> Image.Image:
    parsed = urlparse(image_url)
    if parsed.scheme == "data":
        header, encoded = image_url.split(",", 1)
        if ";base64" not in header:
            raise ValueError("data image URL must be base64 encoded")
        return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
    if parsed.scheme == "file":
        path_text = unquote(parsed.path)
        if re.match(r"^/[A-Za-z]:/", path_text):
            path_text = path_text[1:]
        return Image.open(Path(path_text)).convert("RGB")
    raise ValueError(f"unsupported screenshot URL scheme: {parsed.scheme}")


def _detect_chat_send_point(image: Image.Image) -> tuple[int, int] | None:
    width, height = image.size
    if width <= 0 or height <= 0:
        return None
    components = _connected_components_for_pixels(
        image,
        x_range=(int(width * 0.20), int(width * 0.90)),
        y_range=(int(height * 0.55), height),
        predicate=_looks_like_input_border_pixel,
        min_pixels=10,
    )
    candidates: list[tuple[float, tuple[int, int]]] = []
    for area, x1, y1, x2, y2 in components:
        box_width = x2 - x1 + 1
        box_height = y2 - y1 + 1
        if box_width < 30 or box_height > 12:
            continue
        if y2 < int(height * 0.82):
            continue
        point = (max(0, x2 - 14), max(0, min(height - 1, y2 - 14)))
        score = x2 * 2 + y2 + min(area, 300) * 0.01
        candidates.append((score, point))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    icon_components = _connected_components_for_pixels(
        image,
        x_range=(int(width * 0.35), int(width * 0.95)),
        y_range=(int(height * 0.70), height),
        predicate=_looks_like_submit_icon_pixel,
        min_pixels=5,
    )
    icon_candidates: list[tuple[float, tuple[int, int]]] = []
    for area, x1, y1, x2, y2 in icon_components:
        box_width = x2 - x1 + 1
        box_height = y2 - y1 + 1
        if box_width <= 2 and box_height > 18:
            continue
        if box_height <= 2 and box_width > 30:
            continue
        if box_width > 80 or box_height > 80:
            continue
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        score = center[0] * 2 + center[1] + min(area, 200) * 0.02
        icon_candidates.append((score, center))
    if icon_candidates:
        return max(icon_candidates, key=lambda item: item[0])[1]
    return None


def _looks_like_input_border_pixel(rgb: tuple[int, int, int]) -> bool:
    r, g, b = rgb
    return b > 165 and g > 145 and r > 100 and b - r > 15 and b - g >= -10


def _looks_like_submit_icon_pixel(rgb: tuple[int, int, int]) -> bool:
    r, g, b = rgb
    if b > 120 and g > 80 and r < 180 and b - r > 20:
        return True
    return max(r, g, b) < 235 and max(r, g, b) - min(r, g, b) < 80


def _connected_components_for_pixels(
    image: Image.Image,
    *,
    x_range: tuple[int, int],
    y_range: tuple[int, int],
    predicate: Callable[[tuple[int, int, int]], bool],
    min_pixels: int,
) -> list[tuple[int, int, int, int, int]]:
    pixels = image.load()
    x_start, x_stop = x_range
    y_start, y_stop = y_range
    x_start = max(0, min(image.width, x_start))
    x_stop = max(x_start, min(image.width, x_stop))
    y_start = max(0, min(image.height, y_start))
    y_stop = max(y_start, min(image.height, y_stop))
    mask: set[tuple[int, int]] = set()
    for y in range(y_start, y_stop):
        for x in range(x_start, x_stop):
            if predicate(pixels[x, y]):
                mask.add((x, y))
    seen: set[tuple[int, int]] = set()
    components: list[tuple[int, int, int, int, int]] = []
    for point in tuple(mask):
        if point in seen:
            continue
        stack = [point]
        seen.add(point)
        xs: list[int] = []
        ys: list[int] = []
        while stack:
            x, y = stack.pop()
            xs.append(x)
            ys.append(y)
            for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if neighbor in mask and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if len(xs) >= min_pixels:
            components.append((len(xs), min(xs), min(ys), max(xs), max(ys)))
    return components


def _viewport_screenshot_to_physical(
    screen: Any,
    crop_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x, y, width, height = crop_box
    size_fn = getattr(screen, "size", None)
    physical_size_fn = getattr(screen, "physical_size", None)
    physical_origin_fn = getattr(screen, "physical_origin", None)
    if not callable(size_fn) or not callable(physical_size_fn):
        return crop_box
    logical_width, logical_height = size_fn()
    physical_width, physical_height = physical_size_fn()
    origin_x, origin_y = (
        physical_origin_fn()
        if callable(physical_origin_fn)
        else (0, 0)
    )
    if logical_width <= 0 or logical_height <= 0:
        return crop_box
    x = max(0, min(x, logical_width - 1))
    y = max(0, min(y, logical_height - 1))
    width = max(1, min(width, logical_width - x))
    height = max(1, min(height, logical_height - y))
    return (
        origin_x + int(round(x * physical_width / logical_width)),
        origin_y + int(round(y * physical_height / logical_height)),
        max(1, int(round(width * physical_width / logical_width))),
        max(1, int(round(height * physical_height / logical_height))),
    )


def _workflow_item_defaults(job: JobState, item_id: str) -> dict[str, str]:
    if not item_id:
        return {}
    input_path = job.metadata.get("input_path")
    if not input_path:
        return {}
    try:
        from cfie_gui_agent.workflow import load_workflow_items

        items = load_workflow_items(str(input_path))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    for item in items:
        if item.item_id == item_id:
            return {
                "input_text": item.input_text,
                "expected_output": item.expected_output,
            }
    return {}


def _maybe_record_final_workflow_result(
    final_text: str | None,
    *,
    job: JobState,
    trace_store: AgentTraceStore,
) -> dict[str, Any] | None:
    if not final_text:
        return None
    stripped = final_text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    if not {"item_id", "output_text", "status"}.issubset(value):
        return None
    defaults = _workflow_item_defaults(job, str(value.get("item_id") or ""))
    payload = {
        "job_id": job.job_id,
        "item_id": value.get("item_id"),
        "input_text": value.get("input_text") or defaults.get("input_text", ""),
        "expected_output": value.get("expected_output")
        or defaults.get("expected_output", ""),
        "output_text": value.get("output_text", ""),
        "status": value.get("status"),
        "latency_seconds": value.get("latency_seconds"),
        "artifact_refs": _as_string_list(value.get("artifact_refs", [])),
        "reason": value.get("reason", ""),
        "source": "final_message_json",
    }
    trace_store.record("workflow_result", payload)
    return payload


def _build_model_response_metrics(
    response: Any,
    *,
    step: int,
    latency_seconds: float,
    text_warning_chars: int,
    json_warning_chars: int,
    latency_warning_seconds: float,
    input_text_preview: str = "",
) -> dict[str, Any]:
    output_text = _extract_response_text(response) or ""
    reasoning_text = _extract_response_reasoning_text(response) or ""
    usage = _extract_response_usage(response)
    items = _read_field(response, "output", response) or ()
    function_call_count = 0
    lower_output_text = output_text.lower()
    text_tool_call_count = (
        lower_output_text.count("<tool_call")
        + lower_output_text.count("<tool_code")
    )
    message_count = 0
    tool_argument_chars = 0
    for item in items:
        item_type = _read_field(item, "type")
        if item_type == "message":
            message_count += 1
        if item_type in {"function_call", "tool_call", "computer_call"}:
            function_call_count += 1
            arguments = _read_field(item, "arguments", "")
            if not isinstance(arguments, str):
                try:
                    arguments = json.dumps(arguments, ensure_ascii=False)
                except TypeError:
                    arguments = str(arguments)
            tool_argument_chars += len(arguments)
    function_call_count += text_tool_call_count
    try:
        response_json = json.dumps(response, ensure_ascii=False, default=str)
        response_object = json.loads(response_json)
    except TypeError:
        response_json = str(response)
        response_object = {"repr": response_json[:4000]}
    warnings: list[str] = []
    if len(output_text) > text_warning_chars:
        warnings.append("long_output_text")
    if len(response_json) > json_warning_chars:
        warnings.append("large_response_json")
    if latency_seconds > latency_warning_seconds:
        warnings.append("slow_model_response")
    if "<think" in output_text.lower() or "</think>" in output_text.lower():
        warnings.append("thinking_text_visible")
    if reasoning_text and len(reasoning_text) > text_warning_chars:
        warnings.append("long_reasoning_text")
    return {
        "step": step,
        "latency_seconds": round(latency_seconds, 3),
        "response_json_chars": len(response_json),
        "response_object": response_object,
        "output_text_chars": len(output_text),
        "output_text": output_text,
        "output_text_preview": output_text[:240],
        "reasoning_text_chars": len(reasoning_text),
        "reasoning_text": reasoning_text,
        "reasoning_text_preview": reasoning_text[:240],
        "tool_argument_chars": tool_argument_chars,
        "text_tool_call_count": text_tool_call_count,
        "message_count": message_count,
        "function_call_count": function_call_count,
        "input_text_preview": input_text_preview,
        **usage,
        "warnings": warnings,
    }


def _extract_response_usage(response: Any) -> dict[str, int]:
    usage = _read_field(response, "usage", {}) or {}
    output_details = _read_field(usage, "output_tokens_details", {}) or {}
    completion_details = _read_field(usage, "completion_tokens_details", {}) or {}
    result: dict[str, int] = {}
    for key, aliases in {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
    }.items():
        for alias in aliases:
            value = _read_field(usage, alias, None)
            if isinstance(value, int):
                result[key] = value
                break
    reasoning_tokens = (
        _read_field(output_details, "reasoning_tokens", None)
        or _read_field(completion_details, "reasoning_tokens", None)
    )
    if isinstance(reasoning_tokens, int):
        result["reasoning_tokens"] = reasoning_tokens
    return result


def _conversation_text_preview(conversation: list[dict[str, Any]]) -> str:
    texts: list[str] = []
    for message in conversation[-6:]:
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "input_text":
                continue
            text = str(part.get("text") or "").strip()
            if text:
                texts.append(text)
    return _short_text("\n\n".join(texts), 1600)


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


def _looks_like_incomplete_tool_call(text: str | None) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if "<tool_call" in lower and "</tool_call>" not in lower:
        return True
    if "<function=" in lower and "</function>" not in lower:
        return True
    if "<parameter=" in lower and "</parameter>" not in lower:
        return True
    return False


def _looks_like_unexecuted_tool_plan(
    text: str | None,
    tool_names: tuple[str, ...],
) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    lower = stripped.lower()
    planning_markers = (
        "thinking process",
        "analysis:",
        "plan:",
        "**plan",
        "immediate action",
        "next action",
        "i need to",
        "i will",
        "需要调用",
        "下一步",
        "计划",
        "思考",
    )
    has_planning_marker = any(marker in lower for marker in planning_markers)
    mentioned_tools = [
        tool_name
        for tool_name in tool_names
        if tool_name and tool_name.lower() in lower
    ]
    if has_planning_marker and mentioned_tools:
        return True
    if not mentioned_tools:
        return False
    call_markers = (
        "call",
        "invoke",
        "use the",
        "use `",
        "调用",
        "使用",
        "工具",
    )
    return any(marker in lower for marker in call_markers)


def _human_help_call_from_text(
    *,
    step: int,
    text: str,
    current_frame_ref: str | None,
) -> AgentToolCall | None:
    stripped = (text or "").strip()
    if not stripped:
        return None
    lower = stripped.lower()
    mentions_human_help = (
        "request_human_help" in lower
        or "human help" in lower
        or "人工" in stripped
        and any(marker in stripped for marker in ("求助", "介入", "确认", "处理"))
    )
    blocked_state = any(
        marker in lower
        for marker in (
            "login",
            "captcha",
            "verify",
            "verification",
            "authorize",
            "authorization",
        )
    ) or any(
        marker in stripped
        for marker in (
            "登录",
            "验证码",
            "验证",
            "授权",
            "弹窗",
            "无法继续",
            "需要用户",
        )
    )
    if not mentions_human_help or not blocked_state:
        return None
    evidence_refs = []
    if current_frame_ref and not current_frame_ref.startswith("data:"):
        evidence_refs.append(current_frame_ref)
    return AgentToolCall(
        name="request_human_help",
        call_id=f"safety_human_help_{step}",
        arguments={
            "question": "当前界面需要人工处理，是否继续执行？",
            "risk_reason": _short_text(_strip_tool_markup(stripped), 300),
            "proposed_action": "请处理登录、验证码、授权或给出下一步操作约束。",
            "evidence_refs": evidence_refs,
            "urgency": "normal",
        },
    )


def _strip_tool_markup(text: str) -> str:
    text = re.sub(
        r"<tool_call>.*?</tool_call>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"<tool_code>.*?</tool_code>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return text.strip() or "模型请求人工介入。"


def _extract_response_reasoning_text(response: Any) -> str | None:
    items = _read_field(response, "output", response)
    texts: list[str] = []
    for item in items or ():
        if _read_field(item, "type") != "reasoning":
            continue
        for content in _read_field(item, "content", ()) or ():
            content_type = _read_field(content, "type")
            if content_type in {"reasoning_text", "text"}:
                text = _read_field(content, "text", None)
                if text:
                    texts.append(str(text))
    return "\n".join(texts) if texts else None


def _read_field(value: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(field_name, default)
    return getattr(value, field_name, default)
