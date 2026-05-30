from __future__ import annotations

import json
import re
import subprocess
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from cfie_client import ComputerCall, ComputerLoop, find_computer_calls
from cfie.entrypoints.openai.reasoning_template import QWEN_REASONING_PREAMBLE_KWARG

from cfie_gui_agent.agent_tools import (
    AgentToolError,
    AgentToolCall,
    find_agent_tool_calls,
    find_computer_tool_calls,
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
    VERIFICATION_REPEATED_ACTION,
)


DEFAULT_AGENT_MAX_STEPS = 48


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
    local_click_refinement_radius: int = 120
    local_click_refinement_upscale: bool = True
    local_click_refinement_max_size: int = 1080
    local_click_refinement_resample: str = "nearest"
    local_click_refinement_image_format: str = "PNG"
    local_click_refinement_draw_center_marker: bool = False
    guard_untrusted_clicks: bool = False
    trusted_click_success_threshold: int = 2
    max_repair_turns: int = 4
    require_finish_tool_for_completion: bool = False
    recent_execution_image_frames: int = 4
    stop_requested: Callable[[], bool] | None = None
    human_loop: HumanLoopManager = field(
        default_factory=lambda: HumanLoopManager(channel=InMemoryHumanChannel())
    )
    max_steps: int = DEFAULT_AGENT_MAX_STEPS

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
        macro_proposal_reminder_sent = False
        pending_click_guard: dict[str, Any] | None = None
        trusted_click_counts: dict[str, int] = {}
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
            if self.stop_requested is not None and self.stop_requested():
                return GuiAgentResult(
                    task_id=task.task_id,
                    status="cancelled",
                    reason="User requested stop.",
                    steps=max(0, step - 1),
                    metadata=self._build_result_metadata(
                        job_board=job_board,
                        context_store=context_store,
                        active_job_id=active_job_id,
                        model_response_metrics=model_response_metrics,
                    ),
                )
            if self.include_runtime_context and self.refresh_runtime_context_each_step:
                conversation[0] = self._runtime_context_message(
                    job_board=job_board,
                    context_store=context_store,
                    active_job_id=active_job_id,
                    current_frame_ref=current_frame_ref,
                )
            prune_stats = _prune_execution_image_history(
                conversation,
                max_visual_frames=self.context_manager.policy.max_visual_frames,
                keep_recent=self.recent_execution_image_frames,
            )
            if prune_stats["pruned_now"]:
                self.trace_store.record("visual_history_pruned", prune_stats)
            response_started = perf_counter()
            response = agent(conversation)
            response_latency = perf_counter() - response_started
            if self.stop_requested is not None and self.stop_requested():
                return GuiAgentResult(
                    task_id=task.task_id,
                    status="cancelled",
                    reason="User requested stop.",
                    steps=max(0, step - 1),
                    metadata=self._build_result_metadata(
                        job_board=job_board,
                        context_store=context_store,
                        active_job_id=active_job_id,
                        model_response_metrics=model_response_metrics,
                    ),
                )
            response_metrics = _build_model_response_metrics(
                response,
                step=step,
                latency_seconds=response_latency,
                text_warning_chars=self.response_text_warning_chars,
                json_warning_chars=self.response_json_warning_chars,
                latency_warning_seconds=self.response_latency_warning_seconds,
                input_text_preview=_conversation_text_preview(conversation),
            )
            response_metrics["task_id"] = task.task_id
            app_id = str(task.metadata.get("app_id") or "").strip()
            if app_id:
                response_metrics["app_id"] = app_id
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
            response_text = _extract_response_text(response)
            if (
                computer_calls or _mentions_computer_use(response_text)
            ) and _should_escalate_computer_use_to_human(response_text):
                running = job_board.jobs[active_job_id].queues.running
                request = self.human_loop.request_help(
                    question=(
                        "当前界面可能涉及登录、授权、验证码或账号风险，"
                        "需要人工确认后再继续。"
                    ),
                    task_id=running.subtask_id if running is not None else None,
                    evidence_refs=(current_frame_ref,) if current_frame_ref else (),
                    risk_reason="auth_or_account_sensitive_computer_use",
                    proposed_action="请人工完成或确认登录/授权相关操作。",
                    urgency="normal",
                    metadata={"job_id": active_job_id, "step": step},
                )
                if job_board.jobs[active_job_id].queues.running is not None:
                    job_board.jobs[active_job_id].queues.move_running_to(
                        "waiting_human",
                        human_request_id=request.request_id,
                    )
                self.trace_store.record(
                    "tool_call_safety_override",
                    {
                        "step": step,
                        "reason": "auth_or_account_sensitive_computer_use",
                        "preview": (response_text or "")[:500],
                    },
                )
                return GuiAgentResult(
                    task_id=task.task_id,
                    status="waiting_human",
                    reason=(
                        "Harness converted a sensitive computer action into "
                        "a human-help request."
                    ),
                    steps=step,
                    metadata=self._build_result_metadata(
                        job_board=job_board,
                        context_store=context_store,
                        active_job_id=active_job_id,
                        model_response_metrics=model_response_metrics,
                    ),
                )
            ordered_tool_calls = _order_tool_turn(
                step=step,
                computer_calls=computer_calls,
                agent_tool_calls=agent_tool_calls,
                trace_store=self.trace_store,
            )
            computer_calls = tuple(
                call for kind, call in ordered_tool_calls if kind == "computer_use"
            )
            agent_tool_calls = tuple(
                call for kind, call in ordered_tool_calls if kind == "agent_tool"
            )
            if not computer_calls and not agent_tool_calls:
                final_text = response_text
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
                running = job_board.jobs[active_job_id].queues.running
                if running is not None and self.require_finish_tool_for_completion:
                    self.trace_store.record(
                        "tool_call_parse_retry",
                        {
                            "step": step,
                            "reason": "plain_text_without_tool_call_for_active_task",
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
                                        "The task is still active. Plain text "
                                        "does not complete this GUI Agent task. "
                                        "Return exactly one executable tool call "
                                        "now, or call finish_subtask only after "
                                        "the required results have actually been "
                                        "written."
                                    ),
                                }
                            ],
                        }
                    )
                    continue
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

            for call_kind, call in ordered_tool_calls:
                if call_kind == "computer_use":
                    assert isinstance(call, ComputerCall)
                    self.tool_registry.validate_model_tool("computer_use")
                    click_guard = _pre_click_guard_decision(
                        call=call,
                        computer_loop=self.computer_loop,
                        pending_guard=pending_click_guard,
                        trusted_click_counts=trusted_click_counts,
                        trusted_success_threshold=self.trusted_click_success_threshold,
                    )
                    if self.guard_untrusted_clicks and click_guard["status"] == "guard":
                        refinements = _pre_click_refinement_messages(
                            computer_loop=self.computer_loop,
                            call=call,
                            image_detail=self.image_detail,
                            radius=self.local_click_refinement_radius,
                            upscale=self.local_click_refinement_upscale,
                            max_size=self.local_click_refinement_max_size,
                            resample=self.local_click_refinement_resample,
                            image_format=self.local_click_refinement_image_format,
                            draw_center_marker=(
                                self.local_click_refinement_draw_center_marker
                            ),
                        )
                        if refinements:
                            pending_click_guard = {
                                "signatures": click_guard["signatures"],
                                "created_step": step,
                            }
                            guard_item = _pre_click_guard_context_item(
                                call=call,
                                refinements=refinements,
                                image_detail=self.image_detail,
                            )
                            conversation.append(guard_item)
                            running = job_board.jobs[active_job_id].queues.running
                            active_refinement = refinements[-1]["metadata"]
                            record = StepRecord(
                                step_id=len(step_records) + 1,
                                task_id=active_job_id,
                                subtask_id=(
                                    running.subtask_id if running is not None else None
                                ),
                                action=call.to_openai_dict(),
                                result="click_refinement_required",
                                before_ref=current_frame_ref,
                                after_ref=str(active_refinement.get("image_ref") or ""),
                                summary=(
                                    "Click was not executed; local refinement "
                                    "confirmation is required."
                                ),
                                tags=("computer_use", "click_refinement_required"),
                                metadata={
                                    "model_response_step": step,
                                    **_task_trace_metadata(task),
                                    "click_guard": click_guard,
                                    "local_refinements": [
                                        item["metadata"] for item in refinements
                                    ],
                                    "local_refinement": active_refinement,
                                },
                            )
                            step_records.append(record)
                            context_store.append_step(active_job_id, record)
                            self.trace_store.record_step(record)
                            continue
                    try:
                        executed_click_signatures = click_guard.get("signatures", ())
                        output = self.computer_loop.handle_call(call)
                        if click_guard.get("status") in {"confirmed", "trusted"}:
                            pending_click_guard = None
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
                                "model_response_step": step,
                                **_task_trace_metadata(task),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                        )
                        step_records.append(record)
                        context_store.append_step(active_job_id, record)
                        self.trace_store.record_step(record)
                        continue
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
                        metadata={
                            "model_response_step": step,
                            **_task_trace_metadata(task),
                        },
                    )
                    verification = self.step_verifier.verify(record)
                    record.metadata["verification"] = verification.to_dict()
                    if verification.is_ok:
                        for signature in executed_click_signatures:
                            trusted_click_counts[signature] = (
                                trusted_click_counts.get(signature, 0) + 1
                            )
                    local_refinements = []
                    if not self.guard_untrusted_clicks:
                        local_refinements = _local_click_refinement_messages(
                            computer_loop=self.computer_loop,
                            call=call,
                            verification_status=verification.status,
                            image_detail=self.image_detail,
                            radius=self.local_click_refinement_radius,
                            upscale=self.local_click_refinement_upscale,
                            max_size=self.local_click_refinement_max_size,
                            resample=self.local_click_refinement_resample,
                            image_format=self.local_click_refinement_image_format,
                            draw_center_marker=(
                                self.local_click_refinement_draw_center_marker
                            ),
                        )
                    if local_refinements:
                        record.metadata["local_refinements"] = [
                            item["metadata"] for item in local_refinements
                        ]
                        record.metadata["local_refinement"] = local_refinements[-1][
                            "metadata"
                        ]
                    conversation.append(
                        _computer_call_output_context_item(
                            output,
                            call=call,
                            local_refinements=local_refinements,
                        )
                    )
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
                    if (
                        not macro_proposal_reminder_sent
                        and _should_inject_macro_proposal_reminder(
                            step_records,
                            self.tool_registry,
                        )
                    ):
                        conversation.append(_macro_proposal_reminder_message())
                        self.trace_store.record(
                            "macro_proposal_reminder",
                            {
                                "step": step,
                                "reason": "repeated_successful_ui_workflow",
                                "step_count": len(step_records),
                            },
                        )
                        macro_proposal_reminder_sent = True
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
                    if len(step_records) >= self.max_steps:
                        return self._max_steps_result(
                            task=task,
                            job_board=job_board,
                            context_store=context_store,
                            active_job_id=active_job_id,
                            step_records=step_records,
                            model_response_metrics=model_response_metrics,
                        )
                    continue

                assert isinstance(call, AgentToolCall)
                running_before = job_board.jobs[active_job_id].queues.running
                output = self._handle_agent_tool_call(
                    call,
                    job_board=job_board,
                    context_store=context_store,
                    conversation=conversation,
                    current_frame_ref=current_frame_ref,
                    model_response_step=step,
                )
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
                    metadata={
                        "model_response_step": step,
                        **_task_trace_metadata(task),
                        "output": output,
                    },
                )
                step_records.append(record)
                context_store.append_step(active_job_id, record)
                self.trace_store.record_step(record)
                if (
                    not macro_proposal_reminder_sent
                    and _should_inject_macro_proposal_reminder(
                        step_records,
                        self.tool_registry,
                    )
                ):
                    conversation.append(_macro_proposal_reminder_message())
                    self.trace_store.record(
                        "macro_proposal_reminder",
                        {
                            "step": step,
                            "reason": "repeated_successful_ui_workflow",
                            "step_count": len(step_records),
                        },
                    )
                    macro_proposal_reminder_sent = True
                if call.name == "update_constraints":
                    self.trace_store.record_policy_update(output)
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
                if len(step_records) >= self.max_steps:
                    return self._max_steps_result(
                        task=task,
                        job_board=job_board,
                        context_store=context_store,
                        active_job_id=active_job_id,
                        step_records=step_records,
                        model_response_metrics=model_response_metrics,
                    )

        if job_board.jobs[active_job_id].queues.running is not None:
            job_board.jobs[active_job_id].queues.move_running_to("failed")
        return self._max_steps_result(
            task=task,
            job_board=job_board,
            context_store=context_store,
            active_job_id=active_job_id,
            step_records=step_records,
            model_response_metrics=model_response_metrics,
        )

    def _max_steps_result(
        self,
        *,
        task: GuiAgentTaskSpec,
        job_board: JobBoard,
        context_store: PerJobContextStore,
        active_job_id: str,
        step_records: list[StepRecord],
        model_response_metrics: list[dict[str, Any]],
    ) -> GuiAgentResult:
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
        context_store: PerJobContextStore,
        conversation: list[dict[str, Any]],
        current_frame_ref: str | None,
        model_response_step: int | None = None,
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
        if call.name != "append_trace_note":
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
            blocking = bool(arguments.get("blocking", True))
            context_snapshot = _human_resume_context_snapshot(
                job_board=job_board,
                context_store=context_store,
                active_job_id=active_job_id,
                conversation=conversation,
                current_frame_ref=current_frame_ref,
            )
            request = self.human_loop.request_help(
                question=str(arguments.get("question", "")).strip()
                or "Human help requested.",
                task_id=job.queues.running.subtask_id if job.queues.running else None,
                blocking=blocking,
                evidence_refs=tuple(arguments.get("evidence_refs", ()) or ()),
                risk_reason=arguments.get("risk_reason"),
                proposed_action=arguments.get("proposed_action"),
                allowed_reply_format=arguments.get("allowed_reply_format"),
                urgency=str(arguments.get("urgency", "normal")),
                metadata={
                    "job_id": active_job_id,
                    "tool_call_id": call.call_id,
                    "blocking": blocking,
                    "intervention_kind": "blocking" if blocking else "non_blocking",
                    "resume_context": context_snapshot,
                },
            )
            if blocking and job.queues.running is not None:
                job.queues.move_running_to(
                    "waiting_human",
                    human_request_id=request.request_id,
                )
            return {
                "status": "waiting_human" if blocking else "human_request_queued",
                "request_id": request.request_id,
                "job_id": active_job_id,
                "blocking": blocking,
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

        if call.name in {"write_text_file", "append_text_file"}:
            path = Path(str(arguments.get("path", ""))).expanduser()
            encoding = str(arguments.get("encoding") or "utf-8")
            text = str(arguments.get("text", ""))
            if bool(arguments.get("create_parent", True)):
                path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if call.name == "append_text_file" else "w"
            try:
                with path.open(mode, encoding=encoding, errors="replace") as handle:
                    handle.write(text)
            except OSError as exc:
                return {
                    "status": "rejected",
                    "reason": str(exc),
                    "path": str(path),
                    "tool": call.name,
                }
            return {
                "status": "accepted",
                "path": str(path),
                "chars": len(text),
                "mode": mode,
                "tool": call.name,
            }

        if call.name == "open_url":
            url = str(arguments.get("url", "")).strip()
            browser_path = str(arguments.get("browser_path") or "").strip()
            try:
                if browser_path:
                    command = [browser_path]
                    if bool(arguments.get("new_window", False)):
                        command.append("--new-window")
                    command.append(url)
                    process = subprocess.Popen(command)
                    return {
                        "status": "accepted",
                        "url": url,
                        "pid": process.pid,
                        "tool": call.name,
                    }
                webbrowser.open(url, new=1 if arguments.get("new_window") else 0)
            except Exception as exc:
                return {
                    "status": "rejected",
                    "reason": str(exc),
                    "url": url,
                    "tool": call.name,
                }
            return {"status": "accepted", "url": url, "tool": call.name}

        if call.name == "launch_app":
            executable_path = str(arguments.get("executable_path", "")).strip()
            args = arguments.get("args", []) or []
            if not isinstance(args, list):
                args = [str(args)]
            cwd = str(arguments.get("cwd") or "").strip() or None
            try:
                process = subprocess.Popen(
                    [executable_path, *(str(arg) for arg in args)],
                    cwd=cwd,
                )
            except Exception as exc:
                return {
                    "status": "rejected",
                    "reason": str(exc),
                    "tool": call.name,
                }
            return {
                "status": "accepted",
                "pid": process.pid,
                "tool": call.name,
            }

        if call.name == "run_shell":
            command = str(arguments.get("command", "")).strip()
            cwd = str(arguments.get("cwd") or "").strip() or None
            timeout_seconds = float(arguments.get("timeout_seconds") or 30.0)
            try:
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=max(0.1, timeout_seconds),
                )
            except subprocess.TimeoutExpired as exc:
                return {
                    "status": "rejected",
                    "reason": "timeout",
                    "stdout": _short_text(exc.stdout or "", 4000),
                    "stderr": _short_text(exc.stderr or "", 4000),
                    "tool": call.name,
                }
            except Exception as exc:
                return {
                    "status": "rejected",
                    "reason": str(exc),
                    "tool": call.name,
                }
            return {
                "status": "accepted" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "stdout": _short_text(completed.stdout, 8000),
                "stderr": _short_text(completed.stderr, 8000),
                "tool": call.name,
            }

        if call.name == "append_trace_note":
            metadata = arguments.get("metadata", {}) or {}
            if not isinstance(metadata, dict):
                metadata = {"value": metadata}
            if model_response_step is not None:
                metadata = {
                    **metadata,
                    "model_response_step": model_response_step,
                    "tool_call_id": call.call_id,
                }
            payload = {
                "job_id": active_job_id,
                "title": arguments.get("title"),
                "summary": arguments.get("summary", ""),
                "status": arguments.get("status", "recorded"),
                "artifact_refs": arguments.get("artifact_refs", []) or [],
                "metadata": metadata,
            }
            self.trace_store.record("operation", payload)
            return {"status": "accepted", "trace_event": payload}

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
            parameters = arguments.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {}
            try:
                actions = self.action_macros.expand(
                    macro_name,
                    repeat=repeat,
                    parameters=parameters,
                )
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
                "parameters": parameters,
                "coordinate_space": self.action_macros.require(macro_name).metadata.get(
                    "coordinate_space"
                ),
                "expanded_actions": [
                    action.to_openai_dict() for action in actions
                ],
            }

        if call.name == "propose_action_macro":
            proposal = _macro_proposal_payload(
                arguments,
                screen=self.computer_loop.screen,
            )
            context_snapshot = _human_resume_context_snapshot(
                job_board=job_board,
                context_store=context_store,
                active_job_id=active_job_id,
                conversation=conversation,
                current_frame_ref=current_frame_ref,
            )
            request = self.human_loop.request_help(
                question=(
                    "请确认是否批准这个操作宏："
                    f"{proposal.get('macro_name') or '未命名宏'}"
                ),
                task_id=job.queues.running.subtask_id if job.queues.running else None,
                blocking=False,
                evidence_refs=(current_frame_ref,) if current_frame_ref else (),
                risk_reason="model_proposed_action_macro",
                proposed_action=str(proposal.get("description") or ""),
                allowed_reply_format=(
                    "批准、拒绝，或提出修改建议；修改建议会恢复宏提案时的上下文后重新生成。"
                ),
                urgency="normal",
                metadata={
                    "job_id": active_job_id,
                    "tool_call_id": call.call_id,
                    "blocking": False,
                    "intervention_kind": "macro_approval",
                    "macro_proposal": proposal,
                    "resume_context": context_snapshot,
                },
            )
            return {
                "status": "macro_approval_requested",
                "request_id": request.request_id,
                "job_id": active_job_id,
                "blocking": False,
                "macro_proposal": proposal,
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
            "应用视口由客户端和 harness 维护；不要调用或假设存在视口裁剪工具。"
            "如需打开网页、启动程序、读取或写入本地文件，应调用对应通用工具。"
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
                    "如果一轮响应包含多个工具调用，每个工具调用都必须写 "
                    "index: 1, 2, 3... 表示执行顺序；如果一次 computer_use "
                    "包含多个动作，每个 action 也必须写 index。"
                    "向聊天框、搜索框或表单输入并提交文本时，优先使用 "
                    "computer_use 的 submit_text 动作；harness 会负责聚焦、"
                    "替换现有文本、输入并按 Enter，避免把提交拆成点击、输入、"
                    "点击发送按钮导致坐标误差。"
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
            "应用视口由客户端或 harness 负责裁剪；不要在输出中计划或调用视口裁剪工具。"
            "需要读写文件、打开 URL、启动应用或记录轨迹时，调用通用 Agent 工具。"
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
                        "输出可见的“上一步状态”或“下一步计划”。如果启用思考，"
                        "向聊天框、搜索框或表单输入并提交文本时优先使用 submit_text；"
                        "只有需要精确点选按钮、菜单或控件时才拆成鼠标点击。"
                        "把当前状态、下一步和风险写入 <think> 思考区；普通可见文本"
                        "只用于最终结果或极短说明，不能承载状态分析。不要输出长篇思考。"
                        "只要任务尚未完成，本轮必须以一个可执行工具调用结束；"
                        "不要只输出思考、空白文本或普通说明。"
                        "需要记录阶段性结果时，调用写文件或 shell 类通用工具。每轮新增文本要尽量少，"
                        "让 prefix cache 复用稳定历史；历史截图已经在对话中，不要重复描述。"
                        "每一步只新增必要的操作结果关键帧和简短工具结果。"
                        "如果当前截图已经满足当前子任务，立即调用 finish_subtask；不要为了保险再点击、输入或打开页面。"
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
                    "metadata": _subtask_runtime_metadata(running.metadata),
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
            "action_macros": (
                self.action_macros.to_context_payload()
                if self.action_macros is not None
                else {"macros": []}
            ),
            "macro_policy": (
                "如果当前任务出现稳定、重复的多步界面操作，例如连续提交多条同类输入，"
                "在同类流程连续成功两次后必须调用 propose_action_macro 非阻塞地建议一次操作宏，"
                "然后继续当前任务，不要等待人工批准。"
                "宏必须包含步骤目的、动作顺序、动态参数和可能的点击位置；批准前不要假设宏已经可用。"
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
    }
    for field_name in list_fields_by_tool.get(tool_name, ()):
        if field_name in normalized:
            normalized[field_name] = _as_string_list(normalized[field_name])
    int_fields_by_tool = {
        "read_text_file": ("max_chars",),
        "run_action_macro": ("repeat",),
    }
    for field_name in int_fields_by_tool.get(tool_name, ()):
        if field_name in normalized:
            normalized[field_name] = _as_int_or_original(normalized[field_name])
    if tool_name in {"write_text_file", "append_text_file"} and "text" in normalized:
        normalized["text"] = _normalize_text_tool_payload(
            tool_name=tool_name,
            path=normalized.get("path"),
            text=normalized["text"],
        )
    return normalized


def _normalize_text_tool_payload(
    *,
    tool_name: str,
    path: Any,
    text: Any,
) -> str:
    payload = _as_text_payload(text)
    if tool_name != "append_text_file":
        return payload
    if not str(path or "").lower().endswith(".jsonl"):
        return payload
    if payload.endswith("\n"):
        return payload
    return payload + "\n"


def _as_text_payload(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


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


def _subtask_runtime_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if not metadata:
        return {}
    result: dict[str, Any] = {}
    for key in ("source", "manager_reply", "human_request"):
        value = metadata.get(key)
        if value not in (None, "", {}, []):
            result[key] = value
    request = result.get("human_request")
    if isinstance(request, dict):
        request_metadata = request.get("metadata")
        if isinstance(request_metadata, dict) and "resume_context" in request_metadata:
            result["resume_context"] = request_metadata["resume_context"]
    return result


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


def _pre_click_guard_decision(
    *,
    call: ComputerCall,
    computer_loop: ComputerLoop,
    pending_guard: dict[str, Any] | None,
    trusted_click_counts: dict[str, int],
    trusted_success_threshold: int,
) -> dict[str, Any]:
    signatures = _click_guard_signatures(call, computer_loop.screen)
    if not signatures:
        return {"status": "none", "signatures": ()}
    if pending_guard is not None:
        pending_signatures = tuple(pending_guard.get("signatures") or ())
        if call.coordinate_space == "local_refinement_1000":
            return {
                "status": "confirmed",
                "reason": "local_refinement_correction",
                "signatures": signatures,
                "pending_signatures": pending_signatures,
            }
        if signatures == pending_signatures:
            return {
                "status": "confirmed",
                "reason": "same_click_repeated_after_local_preview",
                "signatures": signatures,
                "pending_signatures": pending_signatures,
            }
    threshold = max(1, int(trusted_success_threshold))
    if all(trusted_click_counts.get(signature, 0) >= threshold for signature in signatures):
        return {
            "status": "trusted",
            "reason": "repeated_successful_click_coordinate",
            "signatures": signatures,
        }
    return {
        "status": "guard",
        "reason": "new_or_untrusted_click_coordinate",
        "signatures": signatures,
    }


def _click_guard_signatures(
    call: ComputerCall,
    screen: Any,
) -> tuple[str, ...]:
    signatures: list[str] = []
    for item in _click_logical_points(call, screen):
        point = item["point"]
        signatures.append(f"{item['action_type']}:{point[0]}:{point[1]}")
    return tuple(signatures)


def _pre_click_refinement_messages(
    *,
    computer_loop: ComputerLoop,
    call: Any,
    image_detail: str | None,
    radius: int = 120,
    upscale: bool = True,
    max_size: int = 1080,
    resample: str = "nearest",
    image_format: str = "PNG",
    draw_center_marker: bool = False,
) -> list[dict[str, Any]]:
    screen = computer_loop.screen
    screenshot_region_around = getattr(screen, "screenshot_region_around", None)
    if not callable(screenshot_region_around):
        return []
    click_points = _click_logical_points(call, screen)
    if not click_points:
        return []

    refinements: list[dict[str, Any]] = []
    total_clicks = len(click_points)
    for local_index, item in enumerate(click_points, start=1):
        point = item["point"]
        intent = str(item.get("intent") or "").strip()
        local_box = _local_refinement_logical_box(point, screen, radius=radius)
        try:
            crop = _capture_local_refinement_crop(
                screenshot_region_around,
                x=point[0],
                y=point[1],
                radius=radius,
                upscale=upscale,
                max_size=max_size,
                resample=resample,
                image_format=image_format,
                draw_center_marker=draw_center_marker,
            )
        except Exception:
            continue
        is_last = local_index == total_clicks
        text = (
            "点击保护：本次 click 尚未执行。"
            "这是本次操作意图附近的局部高清截图。"
            "请不要围绕上一次拟点击中心做判断，也不要默认使用 x=500,y=500。"
            "请把这张局部图当作新的完整图，重新选择操作意图对应目标的中心点。"
        )
        if draw_center_marker:
            text += "图中如有品红色定位框，它只是辅助参考，不代表必须点击框中心。"
        else:
            text += "本轮不绘制中心定位框，避免诱导模型机械点击中心。"
        if intent:
            text += f"本次 click 的操作意图是：{intent}。"
        text += "本轮不提供机器推荐点；请只根据局部高清图重新判断目标中心。"
        if is_last:
            text += (
                "下一轮只看这张局部图，使用 "
                'coordinate_space="local_refinement_1000" 给出局部图 0..1000 坐标。'
                "不要复制任何完整截图坐标；局部图左上角是 x=0,y=0，右下角是 x=1000,y=1000。"
                "harness 只有在你确认或修正后才会真实执行点击并返回完整屏幕观察。"
            )
        else:
            text += "这是复合动作中的前序点击候选，当前动作整体也尚未执行。"
        metadata = {
            "click_index": local_index,
            "action_index": item["action_index"],
            "action_type": item["action_type"],
            "intent": intent or None,
            "proposed_click": {"x": point[0], "y": point[1]},
            "local_refinement_box": list(local_box),
            "coordinate_space": "local_refinement_1000",
            "image_ref": crop.image_url,
            "crop_width": crop.width,
            "crop_height": crop.height,
            "draw_cursor": False,
            "draw_center_marker": bool(draw_center_marker),
            "upscale": bool(upscale),
            "resample": str(resample),
            "image_format": str(image_format),
            "precision_mode": "pre_click_refinement",
            "executed": False,
            "is_active_local_refinement": is_last,
        }
        structured_data = {
            "mode": "pre_click_refinement",
            "executed": False,
            "click_index": local_index,
            "action_index": item["action_index"],
            "action_type": item["action_type"],
            "intent": intent or None,
            "next_coordinate_space": "local_refinement_1000",
            "local_image_coordinate_range": [0, 1000],
            "local_image_task": "choose_target_center_again",
            "draw_cursor": False,
            "draw_center_marker": bool(draw_center_marker),
            "upscale": bool(upscale),
            "resample": str(resample),
            "image_format": str(image_format),
            "do_not_copy_screen_coordinates": True,
        }
        output_payload = {
            "type": "computer_screenshot",
            "image_url": crop.image_url,
            "detail": _local_refinement_image_detail(image_detail),
            "summary": "Click requires local confirmation before execution.",
            "observation_text": text,
            "structured_data": structured_data,
            "click_index": local_index,
            "action_index": item["action_index"],
            "intent": intent or None,
            "coordinate_space": "local_refinement_1000",
            "draw_cursor": False,
            "draw_center_marker": bool(draw_center_marker),
            "upscale": bool(upscale),
            "resample": str(resample),
            "image_format": str(image_format),
            "precision_mode": "pre_click_refinement",
            "executed": False,
            "is_active_local_refinement": is_last,
        }
        refinements.append({"output": output_payload, "metadata": metadata})

    if refinements:
        active_box = refinements[-1]["metadata"]["local_refinement_box"]
        set_local_refinement_box = getattr(computer_loop, "set_local_refinement_box", None)
        if callable(set_local_refinement_box):
            set_local_refinement_box(tuple(int(value) for value in active_box))
    return refinements


def _pre_click_guard_context_item(
    *,
    call: ComputerCall,
    refinements: list[dict[str, Any]],
    image_detail: str | None,
) -> dict[str, Any]:
    active = refinements[-1]["output"]
    output_payload = {
        "type": "computer_screenshot",
        "image_url": active.get("image_url"),
        "detail": image_detail or active.get("detail") or "auto",
        "summary": "Click guard returned a local screenshot; no click was executed.",
        "observation_text": (
            "点击保护：computer_use 尚未执行。"
            "当前只返回拟点击点局部截图，确认无误后才执行真实点击。"
        ),
        "structured_data": active.get("structured_data"),
        "precision_mode": "pre_click_refinement",
        "executed": False,
        "coordinate_space": "local_refinement_1000",
    }
    if len(refinements) > 1:
        output_payload["local_refinements"] = [
            dict(item["output"])
            for item in refinements
            if isinstance(item, dict) and isinstance(item.get("output"), dict)
        ]
    return {
        "type": "computer_call_output",
        "call_id": call.call_id,
        "output": output_payload,
    }


def _local_click_refinement_messages(
    *,
    computer_loop: ComputerLoop,
    call: Any,
    verification_status: str,
    image_detail: str | None,
    radius: int = 120,
    upscale: bool = True,
    max_size: int = 1080,
    resample: str = "nearest",
    image_format: str = "PNG",
    draw_center_marker: bool = False,
) -> list[dict[str, Any]]:
    screen = computer_loop.screen
    screenshot_region_around = getattr(screen, "screenshot_region_around", None)
    if not callable(screenshot_region_around):
        return []
    click_points = _click_logical_points(call, screen)
    if not click_points:
        return []

    refinements: list[dict[str, Any]] = []
    total_clicks = len(click_points)
    for local_index, item in enumerate(click_points, start=1):
        point = item["point"]
        intent = str(item.get("intent") or "").strip()
        local_box = _local_refinement_logical_box(point, screen, radius=radius)
        try:
            crop = _capture_local_refinement_crop(
                screenshot_region_around,
                x=point[0],
                y=point[1],
                radius=radius,
                upscale=upscale,
                max_size=max_size,
                resample=resample,
                image_format=image_format,
                draw_center_marker=draw_center_marker,
            )
        except Exception:
            continue
        is_last = local_index == total_clicks
        text = (
            "点击局部截图：这是刚才 computer_use 中"
            f"第 {local_index}/{total_clicks} 个点击动作后的局部高清截图，"
            "用于重新查看操作意图附近的目标。"
            "请不要围绕上一次点击中心做判断，也不要默认使用 x=500,y=500。"
            "请把这张局部图当作新的完整图，重新选择操作意图对应目标的中心点。"
        )
        if draw_center_marker:
            text += "图中如有品红色定位框，它只是辅助参考，不代表必须点击框中心。"
        else:
            text += "本轮不绘制中心定位框，避免诱导模型机械点击中心。"
        if intent:
            text += f"本次 click 的操作意图是：{intent}。"
        text += "本轮不提供机器推荐点；请只根据局部高清图重新判断目标中心。"
        if is_last:
            text += (
                "如果下一步需要基于这张局部图修正点击，必须设置 "
                'coordinate_space="local_refinement_1000"；harness 会把局部 '
                "0..1000 坐标转换回完整截图坐标。请在思考中只判断目标中心在局部图中的位置，"
                "不要重复完整截图坐标。"
            )
        else:
            text += (
                "这张图用于诊断前序点击；如果要继续基于完整截图操作，"
                '继续使用 coordinate_space="qwen_normalized_1000"。'
            )
        metadata = {
            "click_index": local_index,
            "action_index": item["action_index"],
            "action_type": item["action_type"],
            "intent": intent or None,
            "attempted_click": {"x": point[0], "y": point[1]},
            "local_refinement_box": list(local_box),
            "coordinate_space": "local_refinement_1000",
            "image_ref": crop.image_url,
            "crop_width": crop.width,
            "crop_height": crop.height,
            "draw_cursor": False,
            "draw_center_marker": bool(draw_center_marker),
            "upscale": bool(upscale),
            "resample": str(resample),
            "image_format": str(image_format),
            "reason": verification_status,
            "is_active_local_refinement": is_last,
        }
        structured_data = {
            "mode": "post_click_refinement",
            "executed": True,
            "click_index": local_index,
            "action_index": item["action_index"],
            "action_type": item["action_type"],
            "intent": intent or None,
            "next_coordinate_space": "local_refinement_1000",
            "local_image_coordinate_range": [0, 1000],
            "local_image_task": "choose_target_center_again",
            "draw_cursor": False,
            "draw_center_marker": bool(draw_center_marker),
            "upscale": bool(upscale),
            "resample": str(resample),
            "image_format": str(image_format),
            "verification_status": verification_status,
            "do_not_copy_screen_coordinates": True,
        }
        output_payload = {
            "type": "computer_screenshot",
            "image_url": crop.image_url,
            "detail": _local_refinement_image_detail(image_detail),
            "summary": "Click-local refinement screenshot after execution.",
            "observation_text": text,
            "structured_data": structured_data,
            "click_index": local_index,
            "action_index": item["action_index"],
            "intent": intent or None,
            "coordinate_space": "local_refinement_1000",
            "draw_cursor": False,
            "draw_center_marker": bool(draw_center_marker),
            "upscale": bool(upscale),
            "resample": str(resample),
            "image_format": str(image_format),
            "is_active_local_refinement": is_last,
        }
        refinements.append(
            {
                "output": output_payload,
                "metadata": metadata,
            }
        )

    if refinements:
        active_box = refinements[-1]["metadata"]["local_refinement_box"]
        set_local_refinement_box = getattr(computer_loop, "set_local_refinement_box", None)
        if callable(set_local_refinement_box):
            set_local_refinement_box(tuple(int(value) for value in active_box))
    return refinements


def _capture_local_refinement_crop(
    screenshot_region_around: Any,
    *,
    x: int,
    y: int,
    radius: int,
    upscale: bool = True,
    max_size: int = 1080,
    resample: str = "nearest",
    image_format: str = "PNG",
    draw_center_marker: bool = False,
) -> Any:
    try:
        return screenshot_region_around(
            x=x,
            y=y,
            radius=radius,
            max_width=max(1, int(max_size)),
            max_height=max(1, int(max_size)),
            draw_cursor=False,
            draw_center_marker=draw_center_marker,
            upscale=upscale,
            resample=resample,
            image_format=image_format,
        )
    except TypeError:
        try:
            return screenshot_region_around(
                x=x,
                y=y,
                radius=radius,
                draw_cursor=False,
            )
        except TypeError:
            return screenshot_region_around(x=x, y=y, radius=radius)


def _local_refinement_image_detail(image_detail: str | None) -> str:
    if image_detail == "original":
        return "original"
    return "high"


def _click_logical_points(call: Any, screen: Any) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    actions = list(getattr(call, "actions", ()) or ())
    for action_index, action in enumerate(actions, start=1):
        action_type = getattr(action, "type", "")
        if action_type not in {"click", "double_click"}:
            continue
        x = getattr(action, "x", None)
        y = getattr(action, "y", None)
        if x is None or y is None:
            continue
        intent = str(getattr(action, "intent", "") or "").strip()
        raw = getattr(action, "raw", None)
        if not intent and isinstance(raw, dict):
            intent = str(
                raw.get("intent")
                or raw.get("operation_intent")
                or raw.get("purpose")
                or ""
            ).strip()
        points.append(
            {
                "action_index": action_index,
                "action_type": action_type,
                "intent": intent,
                "point": _map_model_point_to_logical_screen(
                    int(x),
                    int(y),
                    coordinate_space=str(
                        getattr(call, "coordinate_space", "") or "screenshot"
                    ),
                    screen=screen,
                ),
            }
        )
    return points


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
        if _read_field(item, "type") == "reasoning":
            context_items.append(_plain_response_item(item))

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


def _order_tool_turn(
    *,
    step: int,
    computer_calls: tuple[ComputerCall, ...],
    agent_tool_calls: tuple[AgentToolCall, ...],
    trace_store: AgentTraceStore,
) -> tuple[tuple[str, ComputerCall | AgentToolCall], ...]:
    total_calls = len(computer_calls) + len(agent_tool_calls)
    ordered_calls = _sort_tool_calls_across_kinds(
        (
            *((("computer_use", call) for call in computer_calls)),
            *((("agent_tool", call) for call in agent_tool_calls)),
        )
    )
    if total_calls <= 1:
        return ordered_calls

    trace_store.record(
        "parallel_tool_call_ordered",
        {
            "step": step,
            "tool_calls": [
                {
                    "kind": kind,
                    "name": "computer_use" if kind == "computer_use" else call.name,
                    "call_id": call.call_id,
                    "index": _tool_call_index(call),
                }
                for kind, call in ordered_calls
            ],
        },
    )
    return ordered_calls


def _should_inject_macro_proposal_reminder(
    step_records: list[StepRecord],
    tool_registry: ModelToolRegistry,
) -> bool:
    if not tool_registry.is_model_callable("propose_action_macro"):
        return False
    if any(_step_agent_tool_name(record) == "propose_action_macro" for record in step_records):
        return False
    submit_count = sum(
        1
        for record in step_records
        if record.result == "computer_call_output"
        and _step_contains_computer_action(record, "submit_text")
    )
    record_count = sum(
        1
        for record in step_records
        if record.result == "accepted"
        and _step_agent_tool_name(record)
        in {"append_text_file", "write_text_file", "run_shell"}
    )
    return submit_count >= 2 and record_count >= 2


def _macro_proposal_reminder_message() -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": (
                    "宏提醒：你已经连续成功完成至少两次同类 GUI 流程。"
                    "下一轮必须先调用 propose_action_macro 建议一个可复用宏；"
                    "宏应把输入文本、记录字段或其他变化项设计成动态参数。"
                    "宏步骤只能包含 computer_use 的键鼠/等待/输入动作，"
                    "只覆盖界面操作部分；读写文件、shell、记忆或轨迹记录"
                    "仍然作为宏外的普通工具调用完成，不要写进宏步骤。"
                    "这是非阻塞建议，不要等待批准，提交宏建议后继续当前任务。"
                ),
            }
        ],
    }


def _step_agent_tool_name(record: StepRecord) -> str:
    action = record.action if isinstance(record.action, dict) else {}
    if action.get("type") != "agent_tool":
        return ""
    return str(action.get("name") or "")


def _step_contains_computer_action(record: StepRecord, action_type: str) -> bool:
    action = record.action if isinstance(record.action, dict) else {}
    if action.get("type") != "computer_call":
        return False
    actions = action.get("actions")
    if not isinstance(actions, list):
        return False
    return any(
        isinstance(item, dict) and item.get("type") == action_type
        for item in actions
    )


def _sort_tool_calls_across_kinds(
    calls: tuple[tuple[str, ComputerCall | AgentToolCall], ...],
) -> tuple[tuple[str, ComputerCall | AgentToolCall], ...]:
    if len(calls) < 2:
        return calls
    if not all(_tool_call_index(call) is not None for _, call in calls):
        return calls
    return tuple(
        item
        for _, item in sorted(
            enumerate(calls),
            key=lambda item: (_tool_call_index(item[1][1]), item[0]),
        )
    )


def _tool_call_index(call: Any) -> int | None:
    value = getattr(call, "index", None)
    if value is None and isinstance(getattr(call, "arguments", None), dict):
        value = call.arguments.get("index")
    if value is None or isinstance(value, bool):
        return None
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if index > 0 else None


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
    if call.index is not None:
        arguments["index"] = call.index
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
    local_refinements: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    item = output.to_openai_dict()
    output_payload = item.get("output")
    if isinstance(output_payload, dict):
        output_payload["summary"] = _computer_action_summary(call)
        refinement_outputs = [
            dict(refinement["output"])
            for refinement in (local_refinements or ())
            if isinstance(refinement, dict) and isinstance(refinement.get("output"), dict)
        ]
        if refinement_outputs:
            output_payload["local_refinements"] = refinement_outputs
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
        elif action_type == "submit_text":
            text = (action.text or "").replace("\n", " ")
            if len(text) > 80:
                text = text[:77] + "..."
            target = (
                f"@({action.x},{action.y}) "
                if action.x is not None and action.y is not None
                else ""
            )
            parts.append(f"submit_text({target}{text!r})")
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


def _prune_execution_image_history(
    conversation: list[dict[str, Any]],
    *,
    max_visual_frames: int,
    keep_recent: int,
) -> dict[str, Any]:
    max_visual_frames = max(1, int(max_visual_frames or 1))
    keep_recent = max(1, min(int(keep_recent or 1), max_visual_frames))
    execution_indexes: list[int] = []
    active_image_indexes: list[int] = []
    for index, item in enumerate(conversation):
        if _is_pruned_execution_output(item):
            execution_indexes.append(index)
            continue
        if _is_execution_screenshot_output(item):
            execution_indexes.append(index)
            active_image_indexes.append(index)

    if len(execution_indexes) <= max_visual_frames:
        return {
            "total_execution_outputs": len(execution_indexes),
            "active_execution_images_before": len(active_image_indexes),
            "active_execution_images_after": len(active_image_indexes),
            "max_visual_frames": max_visual_frames,
            "keep_recent_execution_images": keep_recent,
            "pruned_now": 0,
        }

    keep_indexes = set(active_image_indexes[-keep_recent:])
    pruned_now = 0
    for index in active_image_indexes:
        if index in keep_indexes:
            continue
        conversation[index] = _execution_screenshot_to_text_output(conversation[index])
        pruned_now += 1

    return {
        "total_execution_outputs": len(execution_indexes),
        "active_execution_images_before": len(active_image_indexes),
        "active_execution_images_after": len(active_image_indexes) - pruned_now,
        "max_visual_frames": max_visual_frames,
        "keep_recent_execution_images": keep_recent,
        "pruned_now": pruned_now,
    }


def _is_execution_screenshot_output(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("type") != "computer_call_output":
        return False
    output = item.get("output")
    if not isinstance(output, dict):
        return False
    return output.get("type") == "computer_screenshot" and bool(output.get("image_url"))


def _is_pruned_execution_output(item: Any) -> bool:
    if not isinstance(item, dict) or item.get("type") != "function_call_output":
        return False
    output = item.get("output")
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return False
    return isinstance(output, dict) and bool(output.get("cfie_execution_image_pruned"))


def _execution_screenshot_to_text_output(item: dict[str, Any]) -> dict[str, Any]:
    output = item.get("output") if isinstance(item.get("output"), dict) else {}
    summary = ""
    if isinstance(output, dict):
        summary = str(output.get("summary") or "").strip()
    if not summary:
        summary = "computer_use completed; old execution screenshot omitted."
    payload = {
        "status": "computer_call_output_summary",
        "summary": summary,
        "screenshot_omitted": True,
        "cfie_execution_image_pruned": True,
        "reason": (
            "Older execution screenshots were replaced by text summaries so "
            "stable APP/task reference images and prefix cache remain stable."
        ),
    }
    return {
        "type": "function_call_output",
        "call_id": str(item.get("call_id") or ""),
        "output": json.dumps(payload, ensure_ascii=False),
    }


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
        response_for_json = _compact_response_for_trace(
            _response_without_private_debug(response)
        )
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
    reasoning_preamble = _extract_request_reasoning_preamble(request_debug)
    reasoning_generated_text = _strip_reasoning_preamble(
        reasoning_text,
        reasoning_preamble,
    )
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
        "reasoning_preamble": reasoning_preamble,
        "reasoning_generated_text_chars": len(reasoning_generated_text),
        "reasoning_generated_text": reasoning_generated_text,
        "reasoning_generated_text_preview": reasoning_generated_text[:240],
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


def _task_trace_metadata(task: GuiAgentTaskSpec) -> dict[str, str]:
    metadata: dict[str, str] = {"task_spec_id": task.task_id}
    app_id = str(task.metadata.get("app_id") or "").strip()
    if app_id:
        metadata["app_id"] = app_id
    return metadata


def _response_without_private_debug(response: Any) -> Any:
    if isinstance(response, dict):
        return {
            key: value
            for key, value in response.items()
            if not str(key).startswith("_cfie_")
        }
    return response


def _extract_request_reasoning_preamble(request_debug: Any) -> str:
    if not isinstance(request_debug, dict):
        return ""
    chat_kwargs = request_debug.get("chat_template_kwargs")
    if not isinstance(chat_kwargs, dict):
        return ""
    value = chat_kwargs.get(QWEN_REASONING_PREAMBLE_KWARG)
    return str(value).strip() if value else ""


def _strip_reasoning_preamble(reasoning_text: str, preamble: str) -> str:
    reasoning_text = str(reasoning_text or "").strip()
    preamble = str(preamble or "").strip()
    if not reasoning_text or not preamble:
        return reasoning_text
    if reasoning_text.startswith(preamble):
        generated = reasoning_text[len(preamble) :].strip()
        if generated and preamble.endswith("当前状态："):
            return f"当前状态：{generated}"
        return generated
    return reasoning_text


def _compact_response_for_trace(value: Any) -> Any:
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text == "tools" and isinstance(item, list):
                compacted[key_text] = [_compact_tool_schema_for_trace(tool) for tool in item]
                compacted["tool_schema_count"] = len(item)
                continue
            if key_text in {"input_messages", "prompt"} and item:
                compacted[key_text] = "<omitted; see request_context>"
                continue
            compacted[key_text] = _compact_response_for_trace(item)
        return compacted
    if isinstance(value, list):
        return [_compact_response_for_trace(item) for item in value]
    if isinstance(value, tuple):
        return [_compact_response_for_trace(item) for item in value]
    if isinstance(value, str):
        return _compact_media_text_for_trace(value)
    return value


def _compact_tool_schema_for_trace(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict):
        return {"repr": _short_text(str(tool), 120)}
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    name = str(tool.get("name") or function.get("name") or "").strip()
    tool_type = str(tool.get("type") or "function")
    result: dict[str, Any] = {"type": tool_type}
    if name:
        result["name"] = name
    return result


def _compact_media_text_for_trace(text: str) -> str:
    if text.startswith("data:image/"):
        header = text.split(",", 1)[0]
        return f"<image data omitted; {header}; chars={len(text)}>"
    if text.startswith("data:video/"):
        header = text.split(",", 1)[0]
        return f"<video data omitted; {header}; chars={len(text)}>"
    if "data:image/" not in text and "data:video/" not in text:
        return text
    return re.sub(
        r"data:(image|video)/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\r\n]+",
        lambda match: f"<{match.group(1)} data omitted; chars={len(match.group(0))}>",
        text,
    )


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


def _human_resume_context_snapshot(
    *,
    job_board: JobBoard,
    context_store: PerJobContextStore,
    active_job_id: str,
    conversation: list[dict[str, Any]],
    current_frame_ref: str | None,
) -> dict[str, Any]:
    job = job_board.require_job(active_job_id)
    running = job.queues.running
    recent_steps = context_store.get_steps(active_job_id)[-5:]
    return {
        "active_job_id": active_job_id,
        "active_app": job.target_app,
        "active_subtask": running.to_dict() if running is not None else None,
        "queue_counts": job.queues.counts(),
        "current_frame_ref": current_frame_ref,
        "current_frame_text_ref": _model_context_ref(current_frame_ref),
        "conversation_text_preview": _conversation_text_preview(conversation),
        "recent_steps": [step.to_summary_dict() for step in recent_steps],
        "restore_instruction": (
            "恢复该请求产生时的文字和截图上下文，把用户回复作为新的人工反馈处理；"
            "处理完该人工反馈后，再切回请求到来前的当前任务上下文。"
        ),
    }


def _macro_proposal_payload(
    arguments: dict[str, Any],
    *,
    screen: Any,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for raw_step in arguments.get("steps", ()) or ():
        if not isinstance(raw_step, dict):
            continue
        action = dict(raw_step.get("action") or {})
        step_payload = {
            "index": raw_step.get("index"),
            "purpose": str(raw_step.get("purpose") or "").strip(),
            "action": action,
        }
        preview = _macro_click_preview(action, screen=screen)
        if preview is not None:
            step_payload["click_preview"] = preview
        steps.append(step_payload)
    return {
        "macro_name": str(arguments.get("macro_name") or "").strip(),
        "description": str(arguments.get("description") or "").strip(),
        "reason": str(arguments.get("reason") or "").strip(),
        "scope": str(arguments.get("scope") or "current_app"),
        "dynamic_parameters": [
            str(item)
            for item in (arguments.get("dynamic_parameters") or ())
            if item not in (None, "")
        ],
        "steps": sorted(
            steps,
            key=lambda item: (
                int(item.get("index") or 10**9)
                if str(item.get("index") or "").isdigit()
                else 10**9
            ),
        ),
    }


def _macro_click_preview(action: dict[str, Any], *, screen: Any) -> dict[str, Any] | None:
    action_type = str(action.get("type") or action.get("action") or "").strip()
    if action_type not in {"click", "double_click", "move"}:
        return None
    try:
        x = int(action.get("x"))
        y = int(action.get("y"))
    except (TypeError, ValueError):
        return None
    coordinate_space = str(action.get("coordinate_space") or "qwen_normalized_1000")
    point = _map_model_point_to_logical_screen(
        x,
        y,
        coordinate_space=coordinate_space,
        screen=screen,
    )
    screenshot_region_around = getattr(screen, "screenshot_region_around", None)
    if not callable(screenshot_region_around):
        return {
            "center": {"x": point[0], "y": point[1]},
            "coordinate_space": coordinate_space,
            "available": False,
        }
    try:
        crop = screenshot_region_around(x=point[0], y=point[1], radius=80)
    except Exception:
        return {
            "center": {"x": point[0], "y": point[1]},
            "coordinate_space": coordinate_space,
            "available": False,
        }
    return {
        "center": {"x": point[0], "y": point[1]},
        "coordinate_space": coordinate_space,
        "available": True,
        "image_url": crop.image_url,
        "width": crop.width,
        "height": crop.height,
    }


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


def _should_escalate_computer_use_to_human(text: str | None) -> bool:
    lower = (text or "").lower()
    if not lower:
        return False
    human_markers = (
        "request_human_help",
        "human",
        "人工",
        "用户",
        "管理员",
    )
    sensitive_markers = (
        "login",
        "sign in",
        "auth",
        "authorize",
        "captcha",
        "password",
        "account",
        "登录",
        "授权",
        "验证码",
        "密码",
        "账号",
        "账户",
    )
    return (
        any(marker in lower for marker in human_markers)
        and any(marker in lower for marker in sensitive_markers)
    )


def _mentions_computer_use(text: str | None) -> bool:
    lower = (text or "").lower()
    return "computer_use" in lower or "computer call" in lower


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
