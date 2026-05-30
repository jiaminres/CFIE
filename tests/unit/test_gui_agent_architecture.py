from __future__ import annotations

import json

import pytest
from PIL import Image

from cfie_gui_agent import (
    ActionMacro,
    ActionMacroRegistry,
    ActionMacroStep,
    AgentTraceStore,
    ContextManager,
    find_agent_tool_calls,
    find_computer_tool_calls,
    HumanLoopManager,
    HumanReply,
    InMemoryHumanChannel,
    HUMAN_REQUEST_CLAIMED,
    HUMAN_REQUEST_RESOLVED,
    AgentScheduler,
    JobBoard,
    JobState,
    LongHistorySummary,
    ModelToolRegistry,
    MonitorController,
    MonitorEvent,
    NavigationPlanner,
    NavigationRequest,
    ObstaclePolygon,
    Point,
    PolicyStore,
    RuntimeContextBuilder,
    SubtaskState,
    QueuedTask,
    StepRecord,
    StepVerifier,
    TASK_STATUS_ACTIVE,
    TASK_STATUS_COMPLETED,
    TASK_STATUS_PAUSED,
    TASK_STATUS_SUPERSEDED,
    TASK_TYPE_INTERRUPT,
    TaskStack,
    TaskState,
    ToolRegistryError,
    VERIFICATION_OK,
    VERIFICATION_NO_SCREEN_CHANGE,
    VERIFICATION_REPEATED_ACTION,
    VisionContextPolicy,
    WorkspaceProfile,
    assign_subtask_completion_credit,
    model_tool_names_for_profile,
)
from cfie_gui_agent.context import CompactionPlanError


def test_task_stack_interrupt_resume_and_override():
    stack = TaskStack.with_root(TaskState(task_id="task_a", goal="clear dungeon"))

    transition = stack.push_interrupt(
        TaskState(task_id="task_b", goal="revive"),
        reason="death_dialog",
    )

    assert transition.from_task_id == "task_a"
    assert transition.to_task_id == "task_b"
    assert stack.tasks[0].status == TASK_STATUS_PAUSED
    assert stack.active_task_id == "task_b"
    assert stack.active_task.type == TASK_TYPE_INTERRUPT

    resume = stack.pop_resume(reason="revived")

    assert resume.from_task_id == "task_b"
    assert resume.to_task_id == "task_a"
    assert stack.tasks[1].status == TASK_STATUS_COMPLETED
    assert stack.active_task_id == "task_a"
    assert stack.tasks[0].status == TASK_STATUS_ACTIVE

    override = stack.override(
        TaskState(task_id="task_c", goal="higher priority quest"),
        reason="better_reward",
    )

    assert override.from_task_id == "task_a"
    assert override.to_task_id == "task_c"
    assert stack.tasks[0].status == TASK_STATUS_SUPERSEDED
    assert stack.active_task_id == "task_c"


def test_context_manager_uses_43_frame_default_policy():
    policy = VisionContextPolicy()
    assert policy.configured_frame_budget == 43

    steps = [
        StepRecord(
            step_id=index,
            after_ref=f"after_{index}.png",
            video_refs=tuple(f"{index}_{frame}.png" for frame in range(8)),
            tags=("key_evidence",) if index in {1, 2, 3, 4, 5, 6} else (),
        )
        for index in range(1, 36)
    ]
    manager = ContextManager(policy=policy)
    selection = manager.select_prompt_context(
        steps,
        current_frame_ref="current.png",
    )

    assert [step.step_id for step in selection.recent_video_steps] == [34, 35]
    assert len(selection.mid_history_after_frames) == 25
    assert len(selection.key_evidence_frames) == 5
    assert selection.selected_frame_count == 43

    payload = selection.to_context_payload()
    assert payload["current_frame"] == "current.png"
    assert payload["vision_context_policy"]["max_visual_frames"] == 43


def test_context_manager_downgrades_visual_budget_by_usage_ratio():
    steps = [
        StepRecord(
            step_id=index,
            after_ref=f"after_{index}.png",
            video_refs=tuple(f"{index}_{frame}.png" for frame in range(8)),
            tags=("key_evidence",) if index in {1, 2, 3, 4, 5} else (),
        )
        for index in range(1, 36)
    ]
    manager = ContextManager()

    warning = manager.select_prompt_context(
        steps,
        current_frame_ref="current.png",
        usage_ratio=0.75,
    )
    compact = manager.select_prompt_context(
        steps,
        current_frame_ref="current.png",
        usage_ratio=0.90,
    )
    emergency = manager.select_prompt_context(
        steps,
        current_frame_ref="current.png",
        usage_ratio=0.97,
    )

    assert warning.policy.mid_history_after_frames == 15
    assert compact.policy.frames_per_recent_step == 4
    assert compact.policy.key_evidence_frames == 3
    assert emergency.policy.recent_video_steps == 1
    assert emergency.policy.mid_history_after_frames == 0
    assert emergency.selected_frame_count < compact.selected_frame_count


def test_agility_context_policy_uses_after_frames_only():
    policy = VisionContextPolicy.agility(max_visual_frames=43)

    assert policy.mode == "agility"
    assert policy.recent_video_steps == 0
    assert policy.frames_per_recent_step == 0
    assert policy.mid_history_after_frames == 42
    assert policy.configured_frame_budget == 43


def test_keyframe_context_policy_uses_current_and_last_after_frame_only():
    policy = VisionContextPolicy.keyframe(max_visual_frames=2)

    assert policy.mode == "keyframe"
    assert policy.recent_video_steps == 0
    assert policy.frames_per_recent_step == 0
    assert policy.mid_history_after_frames == 1
    assert policy.configured_frame_budget == 2
    assert policy.estimated_visual_tokens <= 4192

    manager = ContextManager(policy=policy)
    selection = manager.select_prompt_context(
        [
            StepRecord(step_id=1, after_ref="after_1.png"),
            StepRecord(step_id=2, after_ref="after_2.png"),
        ],
        current_frame_ref="current.png",
    )

    assert selection.selected_frame_count == 2
    assert [step.step_id for step in selection.mid_history_after_frames] == [2]
    assert selection.recent_video_steps == ()


def test_context_manager_validates_compaction_plan():
    manager = ContextManager()
    manager.validate_compaction_plan(
        {
            "merge_steps": [
                {
                    "steps": [1, 2, 3],
                    "summary": "Repeated waits without progress.",
                }
            ],
            "keep_visual_steps": [{"step": 4, "reason": "target page"}],
        },
        available_step_ids={1, 2, 3, 4},
    )

    with pytest.raises(CompactionPlanError):
        manager.validate_compaction_plan(
            {"drop_visual_steps": [{"steps": [5]}]},
            available_step_ids={1, 2, 3, 4},
        )

    with pytest.raises(CompactionPlanError):
        manager.validate_compaction_plan(
            {"merge_steps": [{"steps": [4], "summary": "active"}]},
            available_step_ids={1, 2, 3, 4},
            current_step_id=4,
        )


def test_context_manager_applies_compaction_plan():
    manager = ContextManager()
    steps = [
        StepRecord(
            step_id=index,
            after_ref=f"after_{index}.png",
            video_refs=(f"{index}_0.png", f"{index}_1.png"),
        )
        for index in range(1, 6)
    ]

    result = manager.apply_compaction_plan(
        steps,
        long_history_summary=LongHistorySummary(),
        plan={
            "merge_steps": [
                {
                    "steps": [1, 2],
                    "summary": "Tried a stale path and it did not progress.",
                    "tags": ["failed_path", "do_not_repeat"],
                    "keep_evidence": ["after_2.png"],
                    "drop_video": True,
                    "downgrade_to_text": True,
                }
            ],
            "drop_visual_steps": [{"step": 3}],
            "do_not_repeat": ["Do not repeat the stale path."],
            "known_targets": {
                "chat_input": {
                    "description": "bottom input",
                    "evidence": "after_4.png",
                }
            },
        },
        current_step_id=5,
    )

    assert [step.step_id for step in result.steps] == [3, 4, 5]
    assert result.steps[0].video_refs == ()
    assert result.merged_step_ids == (1, 2)
    assert "Tried a stale path and it did not progress." in (
        result.long_history_summary.failed_attempts
    )
    assert "after_2.png" in result.long_history_summary.evidence_refs
    assert "Do not repeat the stale path." in result.long_history_summary.do_not_repeat
    assert result.long_history_summary.known_targets["chat_input"]["evidence"] == (
        "after_4.png"
    )


def test_step_verifier_detects_no_screen_change_and_repeated_action():
    verifier = StepVerifier(max_repeated_actions=3)
    action = {"type": "computer_call", "actions": [{"type": "click", "x": 1, "y": 2}]}

    no_change = verifier.verify(
        StepRecord(
            step_id=1,
            action=action,
            before_ref="same.png",
            after_ref="same.png",
        )
    )
    second = verifier.verify(
        StepRecord(
            step_id=2,
            action=action,
            before_ref="before.png",
            after_ref="after.png",
        )
    )
    third = verifier.verify(
        StepRecord(
            step_id=3,
            action=action,
            before_ref="before2.png",
            after_ref="after2.png",
        )
    )

    assert no_change.status == VERIFICATION_NO_SCREEN_CHANGE
    assert second.repeated_action_count == 2
    assert third.status == VERIFICATION_REPEATED_ACTION


def test_step_verifier_uses_visual_diff_for_file_screenshots(tmp_path):
    before = tmp_path / "before.jpg"
    after = tmp_path / "after.jpg"
    changed = tmp_path / "changed.jpg"
    Image.new("RGB", (64, 64), "white").save(before)
    Image.new("RGB", (64, 64), "white").save(after)
    Image.new("RGB", (64, 64), "black").save(changed)
    verifier = StepVerifier(max_repeated_actions=5)
    action = {"type": "computer_call", "actions": [{"type": "click", "x": 1, "y": 2}]}

    no_visual_change = verifier.verify(
        StepRecord(
            step_id=1,
            action=action,
            before_ref=before.as_uri(),
            after_ref=after.as_uri(),
        )
    )
    visual_change = verifier.verify(
        StepRecord(
            step_id=2,
            action=action,
            before_ref=after.as_uri(),
            after_ref=changed.as_uri(),
        )
    )

    assert no_visual_change.status == VERIFICATION_NO_SCREEN_CHANGE
    assert no_visual_change.screen_changed is False
    assert visual_change.status == VERIFICATION_OK
    assert visual_change.screen_changed is True


def test_step_verifier_detects_semantic_repeated_text_submit():
    verifier = StepVerifier(
        max_repeated_actions=3,
        max_repeated_semantic_actions=3,
    )

    def make_step(step_id: int, x: int, text: str) -> StepRecord:
        return StepRecord(
            step_id=step_id,
            task_id="task",
            action={
                "type": "computer_call",
                "actions": [
                    {"type": "click", "x": x, "y": 500},
                    {"type": "type", "text": text},
                    {"type": "keypress", "keys": ["enter"]},
                ],
            },
            result="computer_call_output",
            before_ref=f"before_{step_id}",
            after_ref=f"after_{step_id}",
        )

    first = verifier.verify(make_step(1, 240, "hello"))
    second = verifier.verify(make_step(2, 260, "hello"))
    third = verifier.verify(make_step(3, 280, "hello"))

    assert first.status == VERIFICATION_OK
    assert second.status == VERIFICATION_OK
    assert third.status == VERIFICATION_REPEATED_ACTION
    assert third.metadata["semantic_action_signature"].startswith(
        "computer_text_submit:"
    )
    assert third.metadata["semantic_repeated_action_count"] == 3


def test_step_verifier_does_not_treat_different_text_submits_as_repeated():
    verifier = StepVerifier(
        max_repeated_actions=3,
        max_repeated_semantic_actions=3,
    )

    def make_step(step_id: int, text: str) -> StepRecord:
        return StepRecord(
            step_id=step_id,
            task_id="task",
            action={
                "type": "computer_call",
                "actions": [
                    {"type": "click", "x": 500, "y": 900},
                    {"type": "keypress", "keys": ["ctrl", "a"]},
                    {"type": "type", "text": text},
                    {"type": "keypress", "keys": ["enter"]},
                ],
            },
            result="computer_call_output",
            before_ref=f"before_{step_id}",
            after_ref=f"after_{step_id}",
        )

    first = verifier.verify(make_step(1, "question one"))
    second = verifier.verify(make_step(2, "question two"))
    third = verifier.verify(make_step(3, "question three"))

    assert first.status == VERIFICATION_OK
    assert second.status == VERIFICATION_OK
    assert third.status == VERIFICATION_OK
    assert third.metadata["semantic_repeated_action_count"] == 1


def test_step_verifier_detects_semantic_repeated_click_only_probe():
    verifier = StepVerifier(
        max_repeated_actions=3,
        max_repeated_semantic_actions=3,
        max_repeated_click_only_actions=2,
    )

    def make_step(step_id: int, x: int, y: int) -> StepRecord:
        return StepRecord(
            step_id=step_id,
            task_id="task",
            action={
                "type": "computer_call",
                "actions": [{"type": "click", "x": x, "y": y}],
            },
            result="computer_call_output",
            before_ref=f"before_{step_id}",
            after_ref=f"after_{step_id}",
        )

    verifier.verify(
        StepRecord(
            step_id=0,
            task_id="task",
            action={
                "type": "computer_call",
                "actions": [
                    {"type": "click", "x": 450, "y": 500},
                    {"type": "type", "text": "hello"},
                    {"type": "keypress", "keys": ["enter"]},
                ],
            },
            result="computer_call_output",
            before_ref="before_0",
            after_ref="after_0",
        )
    )
    first = verifier.verify(make_step(1, 609, 421))
    second = verifier.verify(make_step(2, 692, 482))

    assert first.status == VERIFICATION_OK
    assert second.status == VERIFICATION_REPEATED_ACTION
    assert second.metadata["semantic_action_signature"] == "computer_click_only"
    assert second.metadata["semantic_repeated_action_count"] == 2


def test_step_verifier_does_not_join_clicks_across_other_actions():
    verifier = StepVerifier(
        max_repeated_actions=3,
        max_repeated_semantic_actions=3,
        max_repeated_click_only_actions=2,
    )

    def verify_step(step_id: int, actions: list[dict[str, object]]):
        return verifier.verify(
            StepRecord(
                step_id=step_id,
                task_id="task",
                action={"type": "computer_call", "actions": actions},
                result="computer_call_output",
                before_ref=f"before_{step_id}",
                after_ref=f"after_{step_id}",
            )
        )

    first = verify_step(1, [{"type": "click", "x": 500, "y": 900}])
    verify_step(2, [{"type": "type", "text": "question"}])
    second = verify_step(3, [{"type": "click", "x": 700, "y": 930}])

    assert first.status == VERIFICATION_OK
    assert second.status == VERIFICATION_OK
    assert second.metadata["semantic_action_signature"] == "computer_click_only"
    assert second.metadata["semantic_repeated_action_count"] == 1


def test_per_job_context_store_keeps_job_histories_separate():
    from cfie_gui_agent import PerJobContextStore

    store = PerJobContextStore()
    chrome_step = StepRecord(step_id=1, task_id="job_chrome", after_ref="chrome.png")
    wechat_step = StepRecord(step_id=1, task_id="job_wechat", after_ref="wechat.png")

    store.append_step("job_chrome", chrome_step)
    store.append_step("job_wechat", wechat_step)

    assert store.get_steps("job_chrome") == [chrome_step]
    assert store.get_steps("job_wechat") == [wechat_step]


def test_per_job_context_store_compacts_one_job_without_touching_another():
    from cfie_gui_agent import PerJobContextStore

    manager = ContextManager()
    store = PerJobContextStore()
    for index in range(1, 5):
        store.append_step(
            "job_chrome",
            StepRecord(step_id=index, task_id="job_chrome", after_ref=f"c{index}.png"),
        )
        store.append_step(
            "job_wechat",
            StepRecord(step_id=index, task_id="job_wechat", after_ref=f"w{index}.png"),
        )

    result = store.compact_job(
        "job_chrome",
        manager=manager,
        current_step_id=4,
        plan={
            "merge_steps": [
                {
                    "steps": [1, 2],
                    "summary": "Chrome setup completed.",
                    "tags": ["completed"],
                    "keep_evidence": ["c2.png"],
                }
            ]
        },
    )
    payload = store.to_job_context_payload("job_chrome", manager=manager)

    assert result.merged_step_ids == (1, 2)
    assert [step.step_id for step in store.get_steps("job_chrome")] == [3, 4]
    assert [step.step_id for step in store.get_steps("job_wechat")] == [1, 2, 3, 4]
    assert store.get_raw_steps("job_chrome")[0].after_ref == "c1.png"
    assert "Chrome setup completed." in store.get_summary("job_chrome").completed
    assert payload["active_step_count"] == 2
    assert payload["raw_step_count"] == 4


def test_runtime_context_builder_exposes_active_job_subtask_and_policy():
    from cfie_gui_agent import PerJobContextStore

    board = JobBoard()
    board.add_job(JobState(job_id="job_chrome", target_app="Chrome", goal="browse"))
    board.add_subtask(
        SubtaskState(
            subtask_id="open_page",
            job_id="job_chrome",
            goal="open target page",
            status="running",
        )
    )
    store = PerJobContextStore()
    store.append_step(
        "job_chrome",
        StepRecord(step_id=1, task_id="job_chrome", after_ref="after.png"),
    )
    policy = PolicyStore()
    policy.apply_update(
        summary="Do not close the browser.",
        constraints={"current_job": ["Do not close the browser."]},
    )

    context = RuntimeContextBuilder(
        context_manager=ContextManager(),
        tool_registry=ModelToolRegistry(),
        policy_store=policy,
    ).build(
        job_board=board,
        context_store=store,
        active_job_id="job_chrome",
    ).to_dict()

    assert context["active_job"]["target_app"] == "Chrome"
    assert context["active_subtask"]["subtask_id"] == "open_page"
    assert context["policy"]["rules"][0]["text"] == "Do not close the browser."
    assert context["prompt_context"]["current_frame"] == "after.png"


def test_action_macro_registry_expands_human_shortcut_sequence():
    registry = ActionMacroRegistry()
    registry.register(
        ActionMacro(
            name="select_all_then_b",
            steps=(
                ActionMacroStep.keypress("CTRL", "A"),
                ActionMacroStep.wait(0.02),
                ActionMacroStep.keypress("B"),
            ),
        )
    )

    actions = registry.expand("select_all_then_b")

    assert [action.type for action in actions] == ["keypress", "wait", "keypress"]
    assert actions[0].keys == ("CTRL", "A")
    assert registry.to_context_payload()["macros"][0]["name"] == "select_all_then_b"


def test_navigation_planner_adds_detour_for_obstacle():
    request = NavigationRequest(
        source=Point(0, 0),
        target=Point(100, 100),
        obstacles=(
            ObstaclePolygon(
                points=(
                    Point(40, 40),
                    Point(60, 40),
                    Point(60, 60),
                    Point(40, 60),
                )
            ),
        ),
    )

    plan = NavigationPlanner(obstacle_margin=10).plan(request)

    assert plan.status == "planned"
    assert len(plan.waypoints) == 3
    assert plan.waypoints[0] == Point(0, 0)
    assert plan.waypoints[-1] == Point(100, 100)


def test_workspace_profile_serializes_business_context():
    profile = WorkspaceProfile(
        profile_id="ecommerce_ops",
        name="E-commerce operations",
        description="Handle buyer messages and order exceptions.",
        target_apps=("QianNiu", "WeChat"),
        business_rules=("Ask manager before refunds.",),
        reference_image_refs=("artifact://home.png",),
        reference_video_refs=("artifact://flow.mp4",),
        sop_refs=("artifact://sop.md",),
    )

    payload = profile.to_dict()

    assert payload["target_apps"] == ["QianNiu", "WeChat"]
    assert payload["business_rules"] == ["Ask manager before refunds."]
    assert payload["sop_refs"] == ["artifact://sop.md"]


def test_model_tool_registry_blocks_internal_tools():
    registry = ModelToolRegistry()

    registry.validate_model_tool("computer_use")
    registry.validate_model_tool("request_human_help")

    with pytest.raises(ToolRegistryError):
        registry.validate_model_tool("extract_video_frames")

    with pytest.raises(ToolRegistryError):
        registry.validate_model_tool("check_human_reply")

    with pytest.raises(ToolRegistryError):
        registry.validate_model_tool("unknown_tool")


def test_model_tool_registry_validates_tool_arguments():
    registry = ModelToolRegistry()

    registry.validate_model_tool_call(
        "request_human_help",
        {
            "question": "Need approval before refund.",
            "urgency": "high",
            "evidence_refs": ["artifact://frame.png"],
        },
    )
    registry.validate_model_tool_call(
        "ask_replan",
        {
            "suggested_transition": "switch_job",
            "reason": "manager reply arrived",
            "target_job_id": "job_wechat",
        },
    )
    with pytest.raises(ToolRegistryError):
        registry.validate_model_tool_call(
            "request_human_help",
            {"urgency": "high"},
        )

    with pytest.raises(ToolRegistryError):
        registry.validate_model_tool_call(
            "append_trace_note",
            {
                "index": 2,
                "title": "result",
                "summary": "saved",
                "status": "passed",
            },
        )

    with pytest.raises(ToolRegistryError):
        registry.validate_model_tool_call(
            "request_human_help",
            {"question": "Need approval.", "urgency": "soon"},
        )

    with pytest.raises(ToolRegistryError):
        registry.validate_model_tool_call(
            "ask_replan",
            {
                "suggested_transition": "teleport",
                "reason": "bad plan",
            },
        )


def test_model_tool_registry_exports_non_empty_openai_schemas():
    tools = ModelToolRegistry().to_openai_tools()
    by_name = {tool["function"]["name"]: tool for tool in tools}

    request_schema = by_name["request_human_help"]["function"]["parameters"]
    assert request_schema["required"] == ["question"]
    assert "urgency" in request_schema["properties"]
    assert request_schema["properties"]["blocking"]["type"] == "boolean"
    assert "computer_use" in by_name
    computer_schema = by_name["computer_use"]["function"]["parameters"]
    assert computer_schema["required"] == ["coordinate_space", "actions"]
    assert (
        computer_schema["properties"]["coordinate_space"]["enum"]
        == ["qwen_normalized_1000", "local_refinement_1000", "screenshot"]
    )
    coordinate_description = computer_schema["properties"]["coordinate_space"][
        "description"
    ]
    assert "0..1000" in coordinate_description
    assert "not to the full screenshot" in coordinate_description
    assert "index" in computer_schema["properties"]
    action_schema = computer_schema["properties"]["actions"]["items"]
    assert "index" in action_schema["properties"]
    assert action_schema["properties"]["index"]["minimum"] == 1
    assert "0..1000" in action_schema["properties"]["x"]["description"]
    assert "0..1000" in action_schema["properties"]["y"]["description"]
    assert "submit_text" in action_schema["properties"]["type"]["enum"]
    assert "open_url" in by_name
    assert "write_text_file" in by_name
    assert "run_shell" in by_name
    open_url_schema = by_name["open_url"]["function"]["parameters"]
    assert open_url_schema["required"] == ["url"]
    macro_schema = by_name["propose_action_macro"]["function"]["parameters"]
    assert macro_schema["required"] == ["macro_name", "steps"]


def test_model_tool_registry_profiles_keep_core_small_and_generic():
    minimal_names = model_tool_names_for_profile("minimal")
    core_names = model_tool_names_for_profile("core")
    full_names = model_tool_names_for_profile("full")

    assert "computer_use" in minimal_names
    assert "finish_subtask" in minimal_names
    assert "open_url" in minimal_names
    assert "read_text_file" not in minimal_names
    assert "computer_use" in core_names
    assert "open_url" in core_names
    assert "run_shell" in core_names
    assert "propose_action_macro" in core_names
    assert "append_trace_note" not in core_names
    assert "append_trace_note" not in full_names
    assert "read_video_clip" not in core_names
    assert set(minimal_names) < set(core_names)
    assert set(core_names) < set(full_names)


def test_policy_store_converts_constraints_to_context_rules():
    store = PolicyStore()

    update = store.apply_update(
        summary="Avoid repeating failed refund path.",
        constraints={
            "current_job": [
                {
                    "text": "Do not click refund before manager approval.",
                    "severity": "high",
                    "evidence_refs": ["artifact://frame.png"],
                }
            ]
        },
        reason="user correction",
        source="user",
    )
    payload = store.to_context_payload()

    assert update.summary == "Avoid repeating failed refund path."
    assert update.rules[0].scope == "current_job"
    assert update.rules[0].severity == "high"
    assert payload["rules"][0]["text"] == "Do not click refund before manager approval."


def test_agent_trace_store_records_jsonl_events(tmp_path):
    trace_path = tmp_path / "agent_trace.jsonl"
    store = AgentTraceStore(path=trace_path)

    event = store.record_step(
        StepRecord(
            step_id=1,
            action={"type": "agent_tool", "name": "finish_subtask"},
            result="accepted",
        )
    )

    assert event.kind == "step"
    assert store.to_dict()["event_count"] == 1
    assert trace_path.read_text(encoding="utf-8").strip()


def test_agent_trace_store_compacts_loaded_model_response_events(tmp_path):
    trace_path = tmp_path / "agent_trace.jsonl"
    raw_event = {
        "time_unix_nano": 1,
        "kind": "model_response",
        "payload": {
            "response_object": {
                "id": "resp_1",
                "tools": [
                    {
                        "type": "function",
                        "name": "computer_use",
                        "parameters": {"type": "object", "properties": {"x": {}}},
                    }
                ],
                "prompt": "data:image/png;base64," + ("A" * 4096),
            },
            "request_context": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/jpeg;base64," + ("B" * 4096),
                        }
                    ],
                }
            ],
        },
    }
    trace_path.write_text(json.dumps(raw_event) + "\n", encoding="utf-8")

    store = AgentTraceStore()
    store.load_existing(trace_path)

    payload = store.events[0].payload
    assert payload["response_object"]["tools"] == [
        {"type": "function", "name": "computer_use"}
    ]
    assert payload["response_object"]["prompt"] == "<omitted; see request_context>"
    image_debug = payload["request_context"][0]["content"][0]["image_url"]
    assert image_debug["placeholder"] == "[图片]"
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "data:image" not in serialized
    assert "properties" not in serialized


def test_find_agent_tool_calls_parses_json_arguments_and_skips_computer_use():
    calls = find_agent_tool_calls(
        {
            "output": [
                {
                    "type": "function_call",
                    "name": "request_human_help",
                    "call_id": "call_1",
                    "arguments": '{"question": "Need approval."}',
                },
                {
                    "type": "function_call",
                    "name": "computer_use",
                    "call_id": "call_2",
                    "arguments": {"actions": []},
                },
            ]
        }
    )

    assert len(calls) == 1
    assert calls[0].name == "request_human_help"
    assert calls[0].arguments == {"question": "Need approval."}


def test_find_computer_tool_calls_adapts_function_tool_to_computer_call():
    calls = find_computer_tool_calls(
        {
            "output": [
                {
                    "type": "function_call",
                    "name": "computer_use",
                    "call_id": "call_2",
                    "arguments": {
                        "actions": (
                            '[{"action": "click", "coordinate": [10, 20]}, '
                            '{"action": "text", "content": "hello"}]'
                        )
                    },
                }
            ]
        }
    )

    assert len(calls) == 1
    assert calls[0].call_id == "call_2"
    assert [action.type for action in calls[0].actions] == ["click", "type"]
    assert calls[0].actions[0].x == 10
    assert calls[0].actions[1].text == "hello"


def test_find_computer_tool_calls_sorts_fully_indexed_actions():
    calls = find_computer_tool_calls(
        {
            "output": [
                {
                    "type": "function_call",
                    "name": "computer_use",
                    "call_id": "call_indexed",
                    "arguments": {
                        "actions": [
                            {"index": 2, "type": "type", "text": "hello"},
                            {"index": 1, "type": "click", "x": 10, "y": 20},
                        ]
                    },
                }
            ]
        }
    )

    assert [action.index for action in calls[0].actions] == [1, 2]
    assert [action.type for action in calls[0].actions] == ["click", "type"]


def test_find_computer_tool_calls_repairs_missing_top_level_comma():
    calls = find_computer_tool_calls(
        {
            "output": [
                {
                    "type": "function_call",
                    "name": "computer_use",
                    "call_id": "call_3",
                    "arguments": (
                        '{"actions": [{"type": "keypress", "keys": "ENTER"}] '
                        '"call_id": "call_3"}'
                    ),
                }
            ]
        }
    )

    assert calls[0].actions[0].type == "keypress"
    assert calls[0].actions[0].keys == ("ENTER",)


def test_agent_tools_ignore_bare_actions_json_text():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"actions": [{"type": "click", "x": 695, "y": 483}]}',
                    }
                ],
            }
        ]
    }

    assert find_agent_tool_calls(response) == ()
    assert find_computer_tool_calls(response) == ()


def test_agent_tools_ignore_model_native_text_tool_markup():
    response = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "<tool_call><function=computer_use>"
                            "<parameter=actions>"
                            '[{"type":"click","x":10,"y":20}]'
                            "</parameter></function></tool_call>"
                        ),
                    }
                ],
            }
        ]
    }

    assert find_agent_tool_calls(response) == ()
    assert find_computer_tool_calls(response) == ()


def test_agent_tools_ignore_nested_agent_actions_inside_computer_use():
    response = {
        "output": [
            {
                "type": "function_call",
                "name": "computer_use",
                "call_id": "call_nested",
                "arguments": {
                    "actions": [
                        {
                            "type": "call_tool",
                            "tool_name": "append_trace_note",
                            "parameters": {"title": "q1"},
                        }
                    ],
                },
            }
        ]
    }

    assert find_agent_tool_calls(response) == ()
    assert find_computer_tool_calls(response) == ()


def test_agent_tools_keep_real_computer_actions_from_mixed_computer_use():
    response = {
        "output": [
            {
                "type": "function_call",
                "name": "computer_use",
                "call_id": "call_mixed",
                "arguments": {
                    "actions": [
                        {"type": "click", "x": 10, "y": 20},
                        {"type": "open_url", "url": "https://example.com/"},
                    ],
                },
            }
        ]
    }

    agent_calls = find_agent_tool_calls(response)
    computer_calls = find_computer_tool_calls(response)

    assert agent_calls == ()
    assert len(computer_calls) == 1
    assert len(computer_calls[0].actions) == 1
    assert computer_calls[0].actions[0].type == "click"


def test_agent_tool_call_repairs_truncated_complete_object():
    calls = find_agent_tool_calls(
        {
            "output": [
                {
                    "type": "function_call",
                    "name": "request_human_help",
                    "call_id": "call_human",
                    "arguments": (
                        '{"question":"login blocks the task",'
                        '"urgency":"normal",'
                        '"risk_reason":"login modal blocks input"'
                    ),
                }
            ]
        }
    )

    assert len(calls) == 1
    assert calls[0].name == "request_human_help"
    assert calls[0].arguments["question"] == "login blocks the task"
    assert calls[0].arguments["risk_reason"] == "login modal blocks input"


def test_malformed_computer_use_arguments_are_reported_as_tool_error():
    response = {
        "output": [
            {
                "type": "function_call",
                "name": "computer_use",
                "call_id": "bad_computer",
                "arguments": '{"actions": [{"type": "click", "x": ',
            }
        ]
    }

    agent_calls = find_agent_tool_calls(response)
    computer_calls = find_computer_tool_calls(response)

    assert computer_calls == ()
    assert len(agent_calls) == 1
    assert agent_calls[0].name == "computer_use"
    assert "_parse_error" in agent_calls[0].arguments


def test_malformed_computer_use_actions_string_is_reported_as_tool_error():
    response = {
        "output": [
            {
                "type": "function_call",
                "name": "computer_use",
                "call_id": "bad_actions",
                "arguments": {"actions": '[{type: "click", "x": 10}]'},
            }
        ]
    }

    agent_calls = find_agent_tool_calls(response)
    computer_calls = find_computer_tool_calls(response)

    assert computer_calls == ()
    assert len(agent_calls) == 1
    assert agent_calls[0].name == "computer_use"
    assert "computer_use.actions" in agent_calls[0].arguments["_parse_error"]


def test_human_loop_reply_becomes_urgent_task():
    channel = InMemoryHumanChannel()
    manager = HumanLoopManager(channel=channel)

    request = manager.request_help(
        question="Seller asks for a human reply.",
        task_id="task_a",
        evidence_refs=("screen_1.png",),
        urgency="high",
    )

    assert channel.sent_requests == [request]
    assert request.request_id in manager.pending

    channel.push_reply(HumanReply(request_id=request.request_id, text="Reply: hello"))
    tasks = manager.poll()

    assert request.request_id not in manager.pending
    assert len(tasks) == 1
    assert tasks[0]["type"] == "manager_reply"
    assert tasks[0]["priority"] == "urgent"
    assert tasks[0]["reply"]["text"] == "Reply: hello"
    assert manager.pop_urgent_task() == tasks[0]


def test_human_loop_client_claim_blocks_channel_reply():
    channel = InMemoryHumanChannel()
    manager = HumanLoopManager(channel=channel)
    request = manager.request_help(question="Need human decision.")

    claimed = manager.claim_request(request.request_id, source="client")

    assert claimed.status == HUMAN_REQUEST_CLAIMED
    assert claimed.claimed_by == "client"

    channel.push_reply(HumanReply(request_id=request.request_id, text="channel reply"))
    assert manager.poll() == ()
    assert request.request_id in manager.pending

    task = manager.submit_reply(
        request_id=request.request_id,
        text="client reply",
        source="client",
    )

    assert task["reply"]["text"] == "client reply"
    assert manager.completed[request.request_id].status == HUMAN_REQUEST_RESOLVED
    assert request.request_id not in manager.pending


def test_job_board_manager_reply_promotes_waiting_subtask():
    board = JobBoard()
    board.add_job(JobState(job_id="job_wechat", target_app="WeChat", goal="chat"))
    board.add_subtask(
        SubtaskState(
            subtask_id="reply_buyer",
            job_id="job_wechat",
            goal="wait for manager wording",
            status="waiting_human",
            human_request_id="human_1",
        )
    )

    promoted = board.enqueue_manager_reply(
        {
            "request": {
                "request_id": "human_1",
                "metadata": {"job_id": "job_wechat"},
            },
            "reply": {"request_id": "human_1", "text": "Reply: approved"},
        }
    )
    selection = board.select_next(safe_to_interrupt=True)

    assert promoted.subtask_id == "reply_buyer"
    assert promoted.metadata["manager_reply"]["text"] == "Reply: approved"
    assert selection.subtask.subtask_id == "reply_buyer"
    assert board.jobs["job_wechat"].queues.counts()["waiting_human"] == 0


def test_monitor_controller_ingests_event_into_target_job():
    board = JobBoard()
    board.add_job(JobState(job_id="job_qianniu", target_app="QianNiu", goal="chat"))
    controller = MonitorController()

    result = controller.ingest_event(
        board,
        MonitorEvent(
            event_id="evt_1",
            target_job_id="job_qianniu",
            subtask_goal="Reply to new buyer message.",
            priority=80,
            evidence_refs=("artifact://notification.png",),
        ),
    )
    selection = board.select_next(safe_to_interrupt=True)

    assert result.accepted is True
    assert selection.subtask.subtask_id == "monitor:evt_1"
    assert selection.subtask.evidence_refs == ("artifact://notification.png",)


def test_monitor_controller_rejects_unknown_target_job():
    result = MonitorController().ingest_event(
        JobBoard(),
        MonitorEvent(
            event_id="evt_bad",
            target_job_id="job_missing",
            subtask_goal="Reply to message.",
        ),
    )

    assert result.accepted is False
    assert "unknown target job" in result.reason


def test_scheduler_waiting_human_does_not_block_active_work():
    scheduler = AgentScheduler()
    waiting = QueuedTask(task_id="seller_question", kind="waiting_human")
    other = QueuedTask(task_id="inspect_orders", kind="automation")

    scheduler.park_waiting_human(waiting, request_id="human_1")
    scheduler.enqueue_active(other)

    decision = scheduler.next_task(safe_to_interrupt=False)

    assert decision.task == other
    assert decision.queue == "active"
    assert "human_1" in scheduler.snapshot()["waiting_human"]


def test_scheduler_manager_reply_waits_for_safe_interruption_point():
    scheduler = AgentScheduler()
    scheduler.enqueue_manager_reply(
        {
            "request": {"request_id": "human_1"},
            "reply": {"text": "Reply to seller: hello"},
        }
    )

    unsafe = scheduler.next_task(safe_to_interrupt=False)

    assert unsafe.task is None
    assert unsafe.queue == "urgent"

    safe = scheduler.next_task(safe_to_interrupt=True)

    assert safe.task is not None
    assert safe.task.kind == "manager_reply"
    assert safe.queue == "urgent"


def test_job_board_selects_urgent_work_across_jobs_at_safe_point():
    board = JobBoard()
    board.add_job(JobState(job_id="job_chrome", target_app="Chrome", goal="web"))
    board.add_job(JobState(job_id="job_wechat", target_app="WeChat", goal="chat"))
    board.add_subtask(
        SubtaskState(
            subtask_id="reply_manager",
            job_id="job_wechat",
            goal="reply manager",
            priority=100,
        )
    )
    board.add_subtask(
        SubtaskState(
            subtask_id="continue_web",
            job_id="job_chrome",
            goal="continue web",
            priority=10,
        )
    )

    selection = board.select_next(safe_to_interrupt=True)

    assert selection.job.job_id == "job_wechat"
    assert selection.subtask.subtask_id == "reply_manager"
    assert board.active_job_id == "job_wechat"
    assert board.switch_history[-1].from_job_id == "job_chrome"
    assert board.switch_history[-1].to_job_id == "job_wechat"


def test_job_board_keeps_running_subtask_when_not_safe_to_interrupt():
    board = JobBoard()
    board.add_job(JobState(job_id="job_chrome", target_app="Chrome", goal="web"))
    board.add_job(JobState(job_id="job_wechat", target_app="WeChat", goal="chat"))
    board.add_subtask(
        SubtaskState(
            subtask_id="typing_reply",
            job_id="job_chrome",
            goal="type reply",
            status="running",
        )
    )
    board.add_subtask(
        SubtaskState(
            subtask_id="manager_reply",
            job_id="job_wechat",
            goal="manager reply",
            priority=100,
        )
    )

    selection = board.select_next(safe_to_interrupt=False)

    assert selection.job.job_id == "job_chrome"
    assert selection.subtask.subtask_id == "typing_reply"
    assert board.active_job_id == "job_chrome"


def test_subtask_completion_reward_discounting():
    assignment = assign_subtask_completion_credit(
        subtask_id="send_message",
        step_ids=[10, 11, 12],
        terminal_reward=2.0,
        gamma=0.9,
    )

    assert assignment.terminal_step_id == 12
    assert [credit.step_id for credit in assignment.credits] == [12, 11, 10]
    assert [round(credit.credit, 3) for credit in assignment.credits] == [
        2.0,
        1.8,
        1.62,
    ]
