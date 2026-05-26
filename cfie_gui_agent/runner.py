from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from cfie_client import ComputerCall, ComputerLoop, find_computer_calls

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
from cfie_gui_agent.verifier import (
    StepVerifier,
    VERIFICATION_NO_SCREEN_CHANGE,
    VERIFICATION_REPEATED_ACTION,
)


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
    max_repair_turns: int = 4
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

        max_model_turns = self.max_steps + max(0, self.max_repair_turns)
        for step in range(1, max_model_turns + 1):
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
            conflicting_tools = _mentioned_agent_tools_in_text(
                _extract_response_text(response) or "",
                self.tool_registry.allowed_tool_names,
            )
            if conflicting_tools and computer_calls and not agent_tool_calls:
                recovered_agent_calls = _recover_agent_tool_calls_from_text_intent(
                    step=step,
                    text=_extract_response_text(response) or "",
                    tool_names=conflicting_tools,
                    job=job_board.jobs[active_job_id],
                    trace_store=self.trace_store,
                )
                if recovered_agent_calls:
                    self.trace_store.record(
                        "tool_call_safety_override",
                        {
                            "step": step,
                            "reason": "recovered_agent_tool_intent",
                            "recovered_tools": [
                                call.name for call in recovered_agent_calls
                            ],
                            "preview": (_extract_response_text(response) or "")[:500],
                        },
                    )
                    computer_calls = ()
                    agent_tool_calls = (*recovered_agent_calls, *agent_tool_calls)
                else:
                    self.trace_store.record(
                        "tool_call_parse_retry",
                        {
                            "step": step,
                            "reason": "agent_tool_intent_with_computer_action",
                            "mentioned_tools": list(conflicting_tools),
                            "preview": (_extract_response_text(response) or "")[:500],
                        },
                    )
                    conversation.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        "Your text said the next step requires the "
                                        f"Agent tool {', '.join(conflicting_tools)}, "
                                        "but the returned structured call was "
                                        "computer_use. Do not use mouse clicks or "
                                        "keyboard actions as a substitute for Agent "
                                        "tools. Return exactly one function_call now. "
                                        "The function_call name must be one of: "
                                        f"{', '.join(conflicting_tools)}. Put all "
                                        "parameters in that tool's arguments. Do not "
                                        "return computer_use in this repair turn."
                                    ),
                                }
                            ],
                        }
                    )
                    continue

            computer_calls, agent_tool_calls = _select_single_tool_turn(
                step=step,
                computer_calls=computer_calls,
                agent_tool_calls=agent_tool_calls,
                trace_store=self.trace_store,
            )
            if _workflow_input_read_is_required(
                job_board.jobs[active_job_id],
                conversation=conversation,
            ) and computer_calls:
                input_path = job_board.jobs[active_job_id].metadata.get("input_path")
                self.trace_store.record(
                    "workflow_input_read_required",
                    {
                        "step": step,
                        "input_path": str(input_path or ""),
                        "blocked_computer_calls": len(computer_calls),
                        "blocked_agent_tools": [call.name for call in agent_tool_calls],
                    },
                )
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "当前任务配置了输入清单，但你还没有读取清单正文。"
                                    "在执行任何电脑操作或记录结果之前，必须先调用 "
                                    f"read_text_file 读取：{input_path}。"
                                    "只返回这一个工具调用，不要调用 computer_use。"
                                ),
                            }
                        ],
                    }
                )
                continue

            if not computer_calls and not agent_tool_calls:
                final_text = _extract_response_text(response)
                running = job_board.jobs[active_job_id].queues.running
                if running is not None and not (final_text or "").strip():
                    self.trace_store.record(
                        "tool_call_parse_retry",
                        {
                            "step": step,
                            "reason": "empty_response_without_tool_call",
                            "preview": "",
                        },
                    )
                    conversation.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        "The previous response did not contain "
                                        "any executable tool call or final text. "
                                        "The task is still active. Return exactly "
                                        "one complete tool call now. Do not spend "
                                        "the response budget on reasoning only."
                                    ),
                                }
                            ],
                        }
                    )
                    continue
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

            _append_assistant_tool_context(
                conversation,
                response=response,
                computer_calls=computer_calls,
                agent_tool_calls=agent_tool_calls,
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
                conversation.append(
                    _computer_call_output_context_item(output, call=call)
                )
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
                local_refinement = _local_click_refinement_message(
                    computer_loop=self.computer_loop,
                    call=call,
                    verification_status=verification.status,
                    image_detail=self.image_detail,
                )
                if local_refinement is not None:
                    conversation.append(local_refinement["message"])
                    record.metadata["local_refinement"] = local_refinement["metadata"]
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
            reason=(
                f"Exceeded max_steps={self.max_steps} "
                f"and max_repair_turns={self.max_repair_turns}"
            ),
            steps=len(step_records),
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
                "\n\n截图视口："
                f"{json.dumps(viewport, ensure_ascii=False, sort_keys=True)}"
            )
        viewport_guidance = (
            "如果有效应用区域只占截图的一部分，先调用 "
            "set_app_viewport 写入应用轮廓；后续截图和点击坐标"
            "都会使用裁剪后的应用视口。"
            if self.tool_registry.is_model_callable("set_app_viewport")
            else "应用视口由客户端和 harness 维护；不要调用或假设存在视口裁剪工具。"
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
                    f"{viewport_guidance}"
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
        viewport_rule = (
            "当 APP 区域清晰后，尽早调用 set_app_viewport，后续截图会裁剪到"
            "活动应用区域，使每轮新增 prefill 尽量接近 2096/4192 token 预算。"
            if self.tool_registry.is_model_callable("set_app_viewport")
            else "应用视口由客户端或 harness 负责裁剪；当前工具列表没有 "
            "set_app_viewport 时，不要在输出中计划或调用它。"
        )
        return {
            "type": "message",
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "运行规则：下面另一段 content 是本轮 GUI Agent 的运行状态 JSON，"
                        "你必须以该状态为准，不要自行编造任务队列或工具状态。\n"
                        "效率要求：回复保持简短。除非截图或视频描述本身就是任务结果，"
                        "否则不要复述画面内容。需要行动时优先调用工具，不要在工具调用前"
                        "输出可见的“上一步状态”或“下一步计划”。不要输出长篇思考；"
                        "如果启用思考，只保留两个短句：上一步状态和下一步动作。"
                        "调用 record_workflow_result 时，除非确有必要，只传 item_id、"
                        "output_text、status、reason、artifact_refs。每轮新增文本要尽量少，"
                        "让 prefix cache 复用稳定历史；历史截图已经在对话中，不要重复描述。"
                        "每一步只新增必要的操作结果关键帧和简短工具结果。"
                        f"{viewport_rule}"
                    ),
                },
                {
                    "type": "input_text",
                    "text": runtime_context_json,
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
            "current_frame": _model_context_ref(current_frame_ref),
            "screen_viewport": _screen_viewport_context(self.computer_loop.screen),
            "context_budget": self.context_manager.policy.to_dict(),
            "recent_steps": [
                {
                    "step_id": step.step_id,
                    "summary": _short_text(step.summary or "", 160),
                    "result": step.result,
                    "tags": list(step.tags),
                    "after_ref": _model_context_ref(step.after_ref),
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
                    "坐标提醒：下一次 computer_use 的当前截图范围是 "
                    f"{_screen_bounds_text(screen)}。Qwen VL 必须设置 "
                    'coordinate_space="qwen_normalized_1000"，并使用 0..1000 '
                    "归一化图像坐标；不要使用物理屏幕坐标。"
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


def _local_click_refinement_message(
    *,
    computer_loop: ComputerLoop,
    call: Any,
    verification_status: str,
    image_detail: str | None,
) -> dict[str, Any] | None:
    if verification_status not in {
        VERIFICATION_NO_SCREEN_CHANGE,
        VERIFICATION_REPEATED_ACTION,
    }:
        return None
    if not _is_click_only_call(call):
        return None
    screen = computer_loop.screen
    point = _last_click_logical_point(call, screen)
    if point is None:
        return None
    local_box = _local_refinement_logical_box(point, screen, radius=520)
    screenshot_region_around = getattr(screen, "screenshot_region_around", None)
    if not callable(screenshot_region_around):
        return None
    try:
        crop = screenshot_region_around(x=point[0], y=point[1], radius=520)
    except Exception:
        return None
    set_local_refinement_box = getattr(computer_loop, "set_local_refinement_box", None)
    if callable(set_local_refinement_box):
        set_local_refinement_box(local_box)
    text = (
        "局部定位兜底：上一次点击可能有轻微偏差。下一张图是以上次尝试点击点"
        f" x={point[0]}, y={point[1]} 为中心的高清局部截图，坐标仍来自当前"
        "截图坐标系。请先在这个局部图中重新确认目标位置，再决定下一次"
        " computer_use。若下一次鼠标操作基于局部图定位，必须设置 "
        'coordinate_space="local_refinement_1000"；harness 会把局部 0..1000 '
        "坐标转换回完整截图坐标。"
    )
    return {
        "message": _screen_observation_message(
            text=text,
            image_url=crop.image_url,
            image_detail=image_detail,
        ),
        "metadata": {
            "attempted_click": {"x": point[0], "y": point[1]},
            "local_refinement_box": list(local_box),
            "coordinate_space": "local_refinement_1000",
            "crop_width": crop.width,
            "crop_height": crop.height,
            "reason": verification_status,
        },
    }


def _is_click_only_call(call: Any) -> bool:
    actions = getattr(call, "actions", ())
    if not actions:
        return False
    return all(getattr(action, "type", "") in {"click", "double_click"} for action in actions)


def _last_click_logical_point(call: Any, screen: Any) -> tuple[int, int] | None:
    actions = list(getattr(call, "actions", ()) or ())
    for action in reversed(actions):
        if getattr(action, "type", "") not in {"click", "double_click", "move"}:
            continue
        x = getattr(action, "x", None)
        y = getattr(action, "y", None)
        if x is None or y is None:
            continue
        return _map_model_point_to_logical_screen(
            int(x),
            int(y),
            coordinate_space=str(getattr(call, "coordinate_space", "") or "screenshot"),
            screen=screen,
        )
    return None


def _map_model_point_to_logical_screen(
    x: int,
    y: int,
    *,
    coordinate_space: str,
    screen: Any,
) -> tuple[int, int]:
    try:
        width, height = screen.size()
    except Exception:
        return (x, y)
    if width <= 0 or height <= 0:
        return (x, y)
    if coordinate_space == "qwen_normalized_1000":
        return (
            max(0, min(width - 1, int(round(x * width / 1000)))),
            max(0, min(height - 1, int(round(y * height / 1000)))),
        )
    return (
        max(0, min(width - 1, x)),
        max(0, min(height - 1, y)),
    )


def _local_refinement_logical_box(
    point: tuple[int, int],
    screen: Any,
    *,
    radius: int,
) -> tuple[int, int, int, int]:
    try:
        width, height = screen.size()
    except Exception:
        width, height = (0, 0)
    if width <= 0 or height <= 0:
        x, y = point
        return (max(0, x - radius), max(0, y - radius), radius * 2, radius * 2)
    x, y = point
    left = max(0, x - radius)
    top = max(0, y - radius)
    right = min(width, x + radius)
    bottom = min(height, y + radius)
    return (left, top, max(1, right - left), max(1, bottom - top))


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
    item = _current_workflow_item(job, item_id=item_id)
    if item is None:
        return {}
    return {
        "input_text": item.input_text,
        "expected_output": item.expected_output,
    }


def _current_workflow_item(job: JobState, item_id: str = "") -> Any | None:
    input_path = job.metadata.get("input_path")
    if not input_path:
        return None
    try:
        from cfie_gui_agent.workflow import load_workflow_items

        items = load_workflow_items(str(input_path))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if item_id:
        for item in items:
            if item.item_id == item_id:
                return item
        return None
    for item in items:
        return item
    return None


def _next_unrecorded_workflow_item(
    job: JobState,
    *,
    trace_store: AgentTraceStore,
) -> Any | None:
    input_path = job.metadata.get("input_path")
    if not input_path:
        return None
    try:
        from cfie_gui_agent.workflow import load_workflow_items

        items = load_workflow_items(str(input_path))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    recorded = {
        str(event.payload.get("item_id") or "")
        for event in trace_store.events
        if event.kind == "workflow_result"
    }
    for item in items:
        if item.item_id not in recorded:
            return item
    return items[-1] if items else None


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


def _append_assistant_tool_context(
    conversation: list[dict[str, Any]],
    *,
    response: Any,
    computer_calls: tuple[ComputerCall, ...],
    agent_tool_calls: tuple[AgentToolCall, ...],
) -> None:
    if not computer_calls and not agent_tool_calls:
        return
    conversation.extend(
        _assistant_tool_context_items(
            response=response,
            computer_calls=computer_calls,
            agent_tool_calls=agent_tool_calls,
        )
    )


def _assistant_tool_context_items(
    *,
    response: Any,
    computer_calls: tuple[ComputerCall, ...],
    agent_tool_calls: tuple[AgentToolCall, ...],
) -> list[dict[str, Any]]:
    wanted_call_ids = {
        str(call.call_id)
        for call in (*computer_calls, *agent_tool_calls)
        if str(call.call_id)
    }
    output_items = tuple(_read_field(response, "output", response) or ())
    context_items: list[dict[str, Any]] = []
    included_call_ids: set[str] = set()

    for item in output_items:
        item_type = _read_field(item, "type")
        if item_type not in {"function_call", "tool_call", "computer_call"}:
            continue
        call_id = str(_read_field(item, "call_id", _read_field(item, "id", "")))
        if call_id not in wanted_call_ids:
            continue
        context_items.append(_plain_response_item(item))
        included_call_ids.add(call_id)

    for call in computer_calls:
        if call.call_id not in included_call_ids:
            context_items.append(_computer_call_as_function_call_item(call))
            included_call_ids.add(call.call_id)
    for call in agent_tool_calls:
        if call.call_id not in included_call_ids:
            context_items.append(_agent_call_as_function_call_item(call))
            included_call_ids.add(call.call_id)
    return context_items


def _select_single_tool_turn(
    *,
    step: int,
    computer_calls: tuple[ComputerCall, ...],
    agent_tool_calls: tuple[AgentToolCall, ...],
    trace_store: AgentTraceStore,
) -> tuple[tuple[ComputerCall, ...], tuple[AgentToolCall, ...]]:
    total_calls = len(computer_calls) + len(agent_tool_calls)
    if total_calls <= 1:
        return computer_calls, agent_tool_calls

    if agent_tool_calls:
        kept_computer_calls: tuple[ComputerCall, ...] = ()
        kept_agent_tool_calls = (agent_tool_calls[0],)
        kept = {
            "kind": "agent_tool",
            "name": kept_agent_tool_calls[0].name,
            "call_id": kept_agent_tool_calls[0].call_id,
        }
    else:
        kept_computer_calls = (computer_calls[0],)
        kept_agent_tool_calls = ()
        kept = {
            "kind": "computer_use",
            "name": "computer_use",
            "call_id": kept_computer_calls[0].call_id,
        }

    dropped = [
        {"kind": "computer_use", "name": "computer_use", "call_id": call.call_id}
        for call in computer_calls
        if call not in kept_computer_calls
    ]
    dropped.extend(
        {"kind": "agent_tool", "name": call.name, "call_id": call.call_id}
        for call in agent_tool_calls
        if call not in kept_agent_tool_calls
    )
    trace_store.record(
        "parallel_tool_call_pruned",
        {
            "step": step,
            "kept": kept,
            "dropped": dropped,
        },
    )
    return kept_computer_calls, kept_agent_tool_calls


def _workflow_input_read_is_required(
    job: JobState,
    *,
    conversation: list[dict[str, Any]],
) -> bool:
    if not job.metadata.get("input_path"):
        return False
    return not _workflow_input_has_been_read(conversation)


def _workflow_input_has_been_read(conversation: list[dict[str, Any]]) -> bool:
    read_call_ids: set[str] = set()
    for item in conversation:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"function_call", "tool_call"} and item.get("name") == "read_text_file":
            call_id = str(item.get("call_id") or item.get("id") or "")
            if call_id:
                read_call_ids.add(call_id)
        elif item_type == "function_call_output":
            call_id = str(item.get("call_id") or "")
            if call_id in read_call_ids:
                return True
    return False


def _plain_response_item(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        result = dict(item)
    elif hasattr(item, "model_dump"):
        result = item.model_dump(exclude_none=True)
    else:
        result = {
            "type": _read_field(item, "type"),
            "call_id": _read_field(item, "call_id", _read_field(item, "id", "")),
            "name": _read_field(item, "name", ""),
            "arguments": _read_field(item, "arguments", {}),
        }
    arguments = result.get("arguments")
    if arguments is not None and not isinstance(arguments, str):
        result["arguments"] = json.dumps(arguments, ensure_ascii=False)
    return result


def _computer_call_as_function_call_item(call: ComputerCall) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "actions": [action.to_openai_dict() for action in call.actions],
    }
    if call.coordinate_space is not None:
        arguments["coordinate_space"] = call.coordinate_space
    return {
        "type": "function_call",
        "name": "computer_use",
        "call_id": call.call_id,
        "arguments": json.dumps(arguments, ensure_ascii=False),
    }


def _computer_call_output_context_item(
    output: Any,
    *,
    call: ComputerCall,
) -> dict[str, Any]:
    item = output.to_openai_dict()
    output_payload = item.get("output")
    if isinstance(output_payload, dict):
        output_payload["summary"] = _computer_action_summary(call)
    return item


def _computer_action_summary(call: ComputerCall) -> str:
    parts: list[str] = []
    for action in call.actions:
        action_type = action.type
        if action_type in {"click", "double_click", "move"}:
            parts.append(f"{action_type}({action.x},{action.y})")
        elif action_type == "type":
            text = (action.text or "").replace("\n", " ")
            if len(text) > 80:
                text = text[:77] + "..."
            parts.append(f"type({text!r})")
        elif action_type == "keypress":
            parts.append("keypress(" + "+".join(action.keys) + ")")
        elif action_type == "wait":
            parts.append(f"wait({action.duration or 0:g}s)")
        elif action_type == "scroll":
            parts.append(f"scroll({action.scroll_x},{action.scroll_y})")
        elif action_type == "drag":
            parts.append(f"drag({len(action.path)} points)")
        else:
            parts.append(action_type)
    return "Executed actions: " + "; ".join(parts)


def _agent_call_as_function_call_item(call: AgentToolCall) -> dict[str, Any]:
    raw = _plain_response_item(call.raw) if call.raw else {}
    if raw.get("type") in {"function_call", "tool_call"}:
        return raw
    return {
        "type": "function_call",
        "name": call.name,
        "call_id": call.call_id,
        "arguments": json.dumps(call.arguments, ensure_ascii=False),
    }


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
        request_debug = _read_field(response, "_cfie_request_debug", None)
        response_for_json = _response_without_private_debug(response)
        response_json = json.dumps(
            response_for_json,
            ensure_ascii=False,
            default=str,
        )
        response_object = json.loads(response_json)
    except TypeError:
        request_debug = None
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
        "request_context": (
            request_debug.get("input", [])
            if isinstance(request_debug, dict)
            else []
        ),
        "request_payload_debug": request_debug if isinstance(request_debug, dict) else {},
        **usage,
        "warnings": warnings,
    }


def _response_without_private_debug(response: Any) -> Any:
    if isinstance(response, dict):
        return {
            key: value
            for key, value in response.items()
            if not str(key).startswith("_cfie_")
        }
    return response


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


def _model_context_ref(ref: str | None) -> str | None:
    if not ref:
        return ref
    if ref.startswith("data:image/"):
        return "<内联图片已从文本中省略；图片已作为 input_image 提供>"
    if ref.startswith("data:video/"):
        return "<inline video omitted from text; provided as input_video>"
    return ref


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


def _mentioned_agent_tools_in_text(
    text: str | None,
    tool_names: tuple[str, ...],
) -> tuple[str, ...]:
    stripped = (text or "").strip()
    if not stripped:
        return ()
    lower = stripped.lower()
    mentioned = [
        tool_name
        for tool_name in tool_names
        if tool_name
        and tool_name != "computer_use"
        and tool_name.lower() in lower
    ]
    if (
        "record_workflow_result" in tool_names
        and _has_workflow_result_evidence(stripped)
        and (
            re.search(r"(记录|保存).{0,20}(结果|答案|任务|状态)", stripped)
            or re.search(
                r"\b(record|save).{0,30}(result|answer|status)\b",
                lower,
            )
        )
    ):
        mentioned.append("record_workflow_result")
    if "finish_subtask" in tool_names and re.search(
        r"(结束|完成).{0,12}(任务|子任务|流程)",
        stripped,
    ):
        mentioned.append("finish_subtask")
    if not mentioned and not _looks_like_unexecuted_tool_plan(stripped, tool_names):
        return ()
    return tuple(dict.fromkeys(mentioned))


def _recover_agent_tool_calls_from_text_intent(
    *,
    step: int,
    text: str,
    tool_names: tuple[str, ...],
    job: JobState,
    trace_store: AgentTraceStore,
) -> tuple[AgentToolCall, ...]:
    """Recover obvious Agent tool calls when Qwen wraps intent as computer_use.

    This is intentionally conservative: it only reconstructs workflow bookkeeping
    tools from explicit text intent. UI actions still require a real computer_use
    call from the model.
    """

    requested = set(tool_names)
    calls: list[AgentToolCall] = []
    if "record_workflow_result" in requested:
        if not _has_workflow_result_evidence(text):
            return ()
        item = _next_unrecorded_workflow_item(job, trace_store=trace_store)
        item_id = str(getattr(item, "item_id", "") or "").strip()
        output_text = _extract_workflow_output_from_text(text)
        if item_id and output_text:
            status = "failed" if _looks_like_failed_workflow_result(text) else "passed"
            calls.append(
                AgentToolCall(
                    name="record_workflow_result",
                    call_id=f"call_recovered_record_{step}",
                    arguments={
                        "item_id": item_id,
                        "output_text": output_text,
                        "status": status,
                        "reason": _short_text(text, 320),
                    },
                )
            )
    if "finish_subtask" in requested:
        calls.append(
            AgentToolCall(
                name="finish_subtask",
                call_id=f"call_recovered_finish_{step}",
                arguments={"completion_reason": _short_text(text, 240)},
            )
        )
    return tuple(calls)


def _has_workflow_result_evidence(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if re.search(r"等待.{0,12}(输出|回答|结果).{0,12}(后|再)?记录", stripped):
        return False
    return bool(
        re.search(
            r"(已得到|已获取|已收到|得到|收到|显示|回复|回答|答案|失败|无法|不能)",
            stripped,
        )
        or re.search(
            r"\b(got|received|showed|shows|answered|answer|result|failed|unable|cannot)\b",
            stripped,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_failed_workflow_result(text: str) -> bool:
    return bool(
        re.search(
            r"(failed|失败|无法访问|不能访问|区域限制|不可用|无法观看|无法完成)",
            text or "",
            flags=re.IGNORECASE,
        )
    )


def _extract_workflow_output_from_text(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    if _looks_like_failed_workflow_result(stripped):
        return _short_text(stripped, 180)
    patterns = (
        r"答案[是为：:\s]*[“\"']([^”\"'，。；;\n]+)[”\"']?",
        r"\banswer\s*(?:is|=|:)\s*[“\"']?([^”\"'，。；;\n]+)[”\"']?",
        r"output_text[\"']?\s*[:=]\s*[\"']([^\"']+)[\"']",
        r"(\d+(?:\.\d+)?)\s*(?:studio albums?|albums?|张专辑|张录音室专辑|首|本|次)",
    )
    for pattern in patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1).strip()
        sentence_number = re.match(
            r"^(\d+(?:\.\d+)?)(?:\.\s+(?:next|then|record|call)\b|$)",
            value,
            flags=re.IGNORECASE,
        )
        if sentence_number:
            return sentence_number.group(1)
        number_match = re.fullmatch(r"(\d+(?:\.\d+)?)(?:\s+\S.*)?", value)
        if number_match and re.search(r"\d", value):
            return number_match.group(1)
        return _short_text(value, 120)
    return ""


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
