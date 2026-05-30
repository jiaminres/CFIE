from __future__ import annotations

from cfie_gui_agent.desktop_client import (
    DIRECT_COMMAND_DO_NOT_REPLY,
    DesktopClientState,
    MacroConfig,
    ReferenceAsset,
    TargetAppConfig,
    build_initial_state,
    build_parser,
    normalize_window_title_pattern,
    parse_macro_sequence,
)
from cfie_gui_agent.desktop_client_ui import GuiAgentDesktopClient
from cfie_gui_agent.human_loop import HUMAN_REQUEST_PENDING
from cfie_gui_agent.jobs import JobState
from cfie_gui_agent.state_store import load_desktop_state, save_desktop_state
from cfie_gui_agent.trace import AgentTraceStore


def test_parse_macro_sequence_groups_combo_keys():
    assert parse_macro_sequence("CTRL+A, B, SHIFT + C") == (
        ("CTRL", "A"),
        ("B",),
        ("SHIFT", "C"),
    )


def test_desktop_state_starts_without_seed_data():
    state = DesktopClientState()

    assert state.target_apps == {}
    assert state.job_board.jobs == {}
    assert state.human_loop.list_requests(include_completed=True) == ()
    assert state.action_macros.macros == {}


def test_desktop_state_registers_macro_for_context_payload():
    state = DesktopClientState()
    state.register_macro(
        MacroConfig(
            name="quick_reply",
            description="Select all text and type a shortcut marker.",
            sequence="CTRL+A, B",
            scope="app",
            app_id="app_shop",
        )
    )

    payload = state.action_macros.to_context_payload()

    assert payload["macros"][0]["name"] == "quick_reply"
    assert payload["macros"][0]["metadata"]["sequence"] == "CTRL+A, B"
    assert payload["macros"][0]["metadata"]["scope"] == "app"
    assert payload["macros"][0]["metadata"]["app_id"] == "app_shop"
    assert state.macro_configs["quick_reply"].sequence == "CTRL+A, B"


def test_desktop_state_approves_model_macro_proposal_with_dynamic_parameters(tmp_path):
    state = DesktopClientState()
    state.add_target_app(
        TargetAppConfig(app_id="app_shop", app_name="Shop", job_id="job_shop")
    )
    state.job_board.add_job(JobState(job_id="job_shop", target_app="Shop", goal="Handle shop."))
    state.selected_app_id = "app_shop"
    request = state.human_loop.request_help(
        question="Approve macro?",
        blocking=False,
        metadata={
            "job_id": "job_shop",
            "intervention_kind": "macro_approval",
            "macro_proposal": {
                "macro_name": "submit_question",
                "description": "Submit one question.",
                "dynamic_parameters": ["question_text"],
                "steps": [
                    {
                        "index": 1,
                        "purpose": "Type dynamic text.",
                        "action": {
                            "type": "type",
                            "text": "{{question_text}}",
                        },
                    }
                ],
            },
        },
    )

    macro = state.approve_macro_request(request.request_id)
    actions = state.action_macros.expand(
        "submit_question",
        parameters={"question_text": "2 + 3"},
    )

    assert macro.metadata["app_id"] == "app_shop"
    assert actions[0].text == "2 + 3"

    state_path = tmp_path / "state.json"
    loaded = load_desktop_state(save_desktop_state(state, state_path))

    loaded_actions = loaded.action_macros.expand(
        "submit_question",
        parameters={"question_text": "7 + 8"},
    )
    assert loaded_actions[0].text == "7 + 8"


def test_desktop_state_persists_apps_macros_trace_and_human_requests(tmp_path):
    state_path = tmp_path / "runs" / "gui_agent" / "state.json"
    trace_path = tmp_path / "runs" / "gui_agent" / "traces" / "app_shop.jsonl"
    state = DesktopClientState(settings={"default_image_detail": "low"})
    state.add_target_app(
        TargetAppConfig(
            app_id="app_shop",
            app_name="Shop",
            job_id="job_shop",
            task_description="Handle seller messages.",
            reference_assets=(
                ReferenceAsset(
                    asset_id="image_map",
                    kind="image",
                    path="map.png",
                    title="Map",
                ),
            ),
            metadata={"trace_path": str(trace_path), "model": "qwen35-vl"},
        )
    )
    state.selected_app_id = "app_shop"
    state.register_macro(
        MacroConfig(
            name="clear_input",
            description="Clear the focused input.",
            sequence="CTRL+A, BACKSPACE",
        )
    )
    request = state.human_loop.request_help(
        question="Need a manager reply?",
        task_id="subtask_seller",
        metadata={"job_id": "job_shop"},
    )
    state.trace_store.path = trace_path
    state.record_operation_summary(
        app_id="app_shop",
        kind="agent_run",
        title="started",
        status="running",
    )

    saved_path = save_desktop_state(state, state_path)
    loaded = load_desktop_state(saved_path)

    assert loaded.selected_app_id == "app_shop"
    assert loaded.target_apps["app_shop"].app_name == "Shop"
    assert loaded.target_apps["app_shop"].reference_assets[0].asset_id == "image_map"
    assert loaded.macro_configs["clear_input"].sequence == "CTRL+A, BACKSPACE"
    assert "clear_input" in loaded.action_macros.macros
    assert loaded.settings["default_image_detail"] == "low"
    assert loaded.trace_store.path == trace_path
    assert [event.kind for event in loaded.trace_store.events] == ["operation"]

    restored_requests = loaded.human_loop.list_requests(include_completed=True)
    assert restored_requests[0]["status"] == HUMAN_REQUEST_PENDING
    assert restored_requests[0]["request"]["request_id"] == request.request_id


def test_desktop_state_removes_app_without_deleting_trace_file(tmp_path):
    trace_path = tmp_path / "app_shop.jsonl"
    trace_path.write_text("kept on disk\n", encoding="utf-8")
    state = DesktopClientState()
    state.add_target_app(
        TargetAppConfig(
            app_id="app_shop",
            app_name="Shop",
            job_id="job_shop",
            metadata={"trace_path": str(trace_path)},
        )
    )
    state.add_target_app(
        TargetAppConfig(app_id="app_other", app_name="Other", job_id="job_other")
    )
    state.selected_app_id = "app_shop"
    state.job_board.add_job(JobState(job_id="job_shop", target_app="Shop", goal="Shop"))
    state.register_macro(
        MacroConfig(
            name="shop_macro",
            description="Shop only",
            sequence="CTRL+A",
            app_id="app_shop",
        )
    )
    request = state.human_loop.request_help(
        question="Need help?",
        metadata={"app_id": "app_shop", "job_id": "job_shop"},
    )
    state.trace_store.path = trace_path
    state.record_operation_summary(
        app_id="app_shop",
        kind="agent_run",
        title="started",
        status="running",
    )

    removed = state.remove_target_app("app_shop")

    assert removed.app_name == "Shop"
    assert "app_shop" not in state.target_apps
    assert "job_shop" not in state.job_board.jobs
    assert "shop_macro" not in state.action_macros.macros
    assert request.request_id not in state.human_loop.pending
    assert state.trace_store.events == []
    assert state.selected_app_id == "app_other"
    assert trace_path.exists()


def test_desktop_state_loads_legacy_control_character_json(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        '{"version":1,"selected_app_id":"","settings":{"note":"line1\nline2"}}',
        encoding="utf-8",
    )

    loaded = load_desktop_state(state_path)

    assert loaded.settings["note"] == "line1\nline2"


def test_desktop_state_saves_manual_app_viewport():
    state = DesktopClientState()
    state.add_target_app(
        TargetAppConfig(app_id="app_shop", app_name="Shop", job_id="job_shop")
    )

    viewport = state.update_target_viewport(
        "app_shop",
        x=120,
        y=80,
        width=960,
        height=540,
    )

    assert viewport == {"x": 120, "y": 80, "width": 960, "height": 540}
    config = state.target_apps["app_shop"]
    assert config.metadata["manual_viewport"] == viewport
    assert config.metadata["viewport_source"] == "desktop_marker"

    state.clear_target_viewport("app_shop")

    assert "manual_viewport" not in state.target_apps["app_shop"].metadata
    assert "viewport_source" not in state.target_apps["app_shop"].metadata


def test_structured_human_reply_preserves_direct_command():
    state = DesktopClientState()
    state.add_target_app(
        TargetAppConfig(app_id="app_shop", app_name="Shop", job_id="job_shop")
    )
    state.job_board.add_job(JobState(job_id="job_shop", target_app="Shop", goal="Handle shop."))
    request = state.human_loop.request_help(
        question="Should I reply?",
        task_id="seller_question",
        metadata={"job_id": "job_shop"},
    )

    task = state.submit_structured_human_reply(
        request_id=request.request_id,
        manager_input="Do not answer this seller. Move to the next conversation.",
        decision_type="更改路径",
        direct_command=DIRECT_COMMAND_DO_NOT_REPLY,
        constraints="Do not mention AI.",
    )

    reply = task["reply"]
    structured = reply["metadata"]["structured_payload"]
    assert "不要回复当前对象" in reply["text"]
    assert structured["direct_command"] == DIRECT_COMMAND_DO_NOT_REPLY
    assert structured["decision_type"] == "更改路径"
    assert structured["constraints"] == ["Do not mention AI."]
    assert state.job_board.jobs["job_shop"].queues.urgent


def test_trace_store_loads_existing_jsonl(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        '{"time_unix_nano":1,"kind":"step","payload":{"step_id":1}}\n'
        '{"time_unix_nano":2,"kind":"operation","payload":{"title":"done"}}\n',
        encoding="utf-8",
    )
    store = AgentTraceStore()

    loaded = store.load_existing(trace_path)

    assert loaded == 2
    assert store.path == trace_path
    assert [event.kind for event in store.events] == ["step", "operation"]


def test_desktop_client_cli_builds_generic_session_and_loads_trace(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        '{"time_unix_nano":1,"kind":"operation","payload":{"title":"done"}}\n',
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--app-name",
            "Web App",
            "--task-description",
            "Open the target app and follow the configured task.",
            "--trace-path",
            str(trace_path),
            "--base-url",
            "http://127.0.0.1:8000",
            "--model",
            "qwen35-vl",
            "--image-detail",
            "low",
            "--max-steps",
            "3",
            "--screenshot-max-width",
            "1280",
            "--screenshot-max-height",
            "720",
            "--tool-profile",
            "full",
        ]
    )

    state = build_initial_state(args)

    assert len(state.target_apps) == 1
    config = next(iter(state.target_apps.values()))
    assert config.task_description.startswith("Open the target app")
    assert config.metadata["base_url"] == "http://127.0.0.1:8000"
    assert config.metadata["model"] == "qwen35-vl"
    assert config.metadata["image_detail"] == "low"
    assert config.metadata["max_steps"] == 3
    assert config.metadata["screenshot_max_width"] == 1280
    assert config.metadata["screenshot_max_height"] == 720
    assert config.metadata["tool_profile"] == "full"
    assert state.trace_store.path == trace_path
    assert [event.kind for event in state.trace_store.events] == ["operation"]


def test_normalize_window_title_pattern_drops_broken_placeholder_regex():
    assert normalize_window_title_pattern("??|Web App") == "Web App"


def test_completed_agent_run_hides_stale_running_card():
    class Event:
        def __init__(self, kind, payload):
            self.kind = kind
            self.payload = payload

    running = Event(
        "operation",
        {
            "app_id": "app_web",
            "kind": "agent_run",
            "status": "running",
            "payload": {
                "run_id": "client_run_20260525_010101",
            },
        },
    )
    finished = Event(
        "operation",
        {
            "app_id": "app_web",
            "kind": "agent_run",
            "status": "waiting_human",
            "payload": {
                "run_id": "client_run_20260525_010101",
            },
        },
    )

    events = GuiAgentDesktopClient._dedupe_agent_run_events([running, finished])

    assert events == [finished]


def test_desktop_status_marks_loaded_running_trace_as_paused():
    class Event:
        def __init__(self, kind, payload):
            self.kind = kind
            self.payload = payload

    class Flag:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    ui = object.__new__(GuiAgentDesktopClient)
    ui.agent_running = Flag(False)
    ui._events_for_selected_app = lambda _app_id: [
        Event(
            "operation",
            {
                "app_id": "app_web",
                "kind": "agent_run",
                "status": "running",
            },
        )
    ]

    assert GuiAgentDesktopClient._latest_agent_run_status(ui, "app_web") == "paused"
    assert GuiAgentDesktopClient._agent_status_label("paused") == "已暂停"


def test_desktop_status_uses_persisted_app_status_before_loaded_trace():
    class Event:
        def __init__(self, kind, payload):
            self.kind = kind
            self.payload = payload

    class Flag:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    state = DesktopClientState()
    state.add_target_app(
        TargetAppConfig(
            app_id="app_web",
            app_name="Web",
            job_id="job_web",
            metadata={"last_run_status": "completed"},
        )
    )
    ui = object.__new__(GuiAgentDesktopClient)
    ui.state = state
    ui.selected_app_id = Flag("app_web")
    ui.agent_running = Flag(False)
    ui._events_for_selected_app = lambda _app_id: [
        Event(
            "operation",
            {
                "app_id": "app_web",
                "kind": "agent_run",
                "status": "running",
            },
        )
    ]

    assert GuiAgentDesktopClient._latest_agent_run_status(ui, "app_web") == "completed"


def test_trace_detail_does_not_attach_model_context_to_operation():
    class Event:
        def __init__(self, kind, payload):
            self.kind = kind
            self.payload = payload

    class Selected:
        def get(self):
            return "app_web"

    model = Event("model_response", {"step": 1, "request_context": [1, 2]})
    operation = Event("operation", {"kind": "agent_run", "status": "waiting_human"})
    ui = object.__new__(GuiAgentDesktopClient)
    ui.selected_app_id = Selected()
    ui._events_for_selected_app = lambda _app_id: [model, operation]

    assert GuiAgentDesktopClient._model_response_event_for_trace_event(
        ui, operation
    ) is None


def test_trace_detail_attaches_operation_with_recorded_model_response_step():
    class Event:
        def __init__(self, kind, payload):
            self.kind = kind
            self.payload = payload

    class Selected:
        def get(self):
            return "app_web"

    model_1 = Event("model_response", {"step": 1, "request_context": ["first"]})
    model_2 = Event("model_response", {"step": 2, "request_context": ["second"]})
    operation = Event(
        "operation",
        {
            "kind": "append_trace_note",
            "metadata": {"model_response_step": 2},
        },
    )
    ui = object.__new__(GuiAgentDesktopClient)
    ui.selected_app_id = Selected()
    ui._events_for_selected_app = lambda _app_id: [model_1, model_2, operation]

    assert (
        GuiAgentDesktopClient._model_response_event_for_trace_event(ui, operation)
        is model_2
    )


def test_trace_detail_attaches_legacy_append_trace_note_operation_to_next_step():
    class Event:
        def __init__(self, kind, payload):
            self.kind = kind
            self.payload = payload

    class Selected:
        def get(self):
            return "app_web"

    model = Event("model_response", {"step": 3, "request_context": ["request"]})
    operation = Event(
        "operation",
        {
            "title": "note",
            "summary": "saved note",
        },
    )
    step = Event(
        "step",
        {
            "step_id": 8,
            "action": {"type": "agent_tool", "name": "append_trace_note"},
            "metadata": {"model_response_step": 3},
        },
    )
    ui = object.__new__(GuiAgentDesktopClient)
    ui.selected_app_id = Selected()
    ui._events_for_selected_app = lambda _app_id: [model, operation, step]

    assert (
        GuiAgentDesktopClient._model_response_event_for_trace_event(ui, operation)
        is model
    )


def test_trace_detail_maps_legacy_step_to_previous_model_response():
    class Event:
        def __init__(self, kind, payload):
            self.kind = kind
            self.payload = payload

    class Selected:
        def get(self):
            return "app_web"

    model_1 = Event("model_response", {"step": 1, "request_context": ["first"]})
    first_step = Event("step", {"step_id": 1, "metadata": {}})
    second_step_same_response = Event("step", {"step_id": 2, "metadata": {}})
    model_2 = Event("model_response", {"step": 2, "request_context": ["second"]})
    ui = object.__new__(GuiAgentDesktopClient)
    ui.selected_app_id = Selected()
    ui._events_for_selected_app = lambda _app_id: [
        model_1,
        first_step,
        second_step_same_response,
        model_2,
    ]

    assert (
        GuiAgentDesktopClient._model_response_event_for_trace_event(
            ui, second_step_same_response
        )
        is model_1
    )


def test_trace_detail_prefers_recorded_model_response_step():
    class Event:
        def __init__(self, kind, payload):
            self.kind = kind
            self.payload = payload

    class Selected:
        def get(self):
            return "app_web"

    model_1 = Event("model_response", {"step": 1, "request_context": ["first"]})
    model_2 = Event("model_response", {"step": 2, "request_context": ["second"]})
    step = Event("step", {"step_id": 99, "metadata": {"model_response_step": 2}})
    ui = object.__new__(GuiAgentDesktopClient)
    ui.selected_app_id = Selected()
    ui._events_for_selected_app = lambda _app_id: [model_1, model_2, step]

    assert (
        GuiAgentDesktopClient._model_response_event_for_trace_event(ui, step)
        is model_2
    )


def test_agent_completion_summary_clamps_reason_text(tmp_path):
    ui = object.__new__(GuiAgentDesktopClient)

    summary = GuiAgentDesktopClient._agent_completion_summary(
        ui,
        trace_path=tmp_path / "trace.jsonl",
        steps=2,
        reason="x" * 240,
    )

    assert "trace.jsonl" in summary
    assert "2 " in summary
    assert len(summary) < 220


def test_desktop_client_screen_capture_uses_scaled_viewport_metadata():
    ui = object.__new__(GuiAgentDesktopClient)

    screen = GuiAgentDesktopClient._screen_capture_from_metadata(
        ui,
        {
            "manual_viewport": {"x": 10, "y": 20, "width": 900, "height": 600},
            "screenshot_max_width": 640,
            "screenshot_max_height": 360,
        },
    )

    assert screen.crop_box == (10, 20, 900, 600)
    assert screen.max_width == 640
    assert screen.max_height == 360


def test_desktop_client_screen_capture_can_auto_crop_by_window_title(monkeypatch):
    class Window:
        crop_box = (100, 80, 1200, 760)

    monkeypatch.setattr(
        "cfie_gui_agent.desktop_client_ui.find_visible_window",
        lambda pattern: Window() if pattern == "Target App" else None,
    )
    ui = object.__new__(GuiAgentDesktopClient)

    screen = GuiAgentDesktopClient._screen_capture_from_metadata(
        ui,
        {
            "window_title_pattern": "Target App",
            "screenshot_max_width": 1280,
            "screenshot_max_height": 720,
        },
    )

    assert screen.crop_box == (100, 80, 1200, 760)


def test_desktop_client_screen_capture_ignores_wildcard_auto_crop(monkeypatch):
    monkeypatch.setattr(
        "cfie_gui_agent.desktop_client_ui.find_visible_window",
        lambda pattern: (_ for _ in ()).throw(AssertionError(pattern)),
    )
    ui = object.__new__(GuiAgentDesktopClient)

    screen = GuiAgentDesktopClient._screen_capture_from_metadata(
        ui,
        {"window_title_pattern": ".*"},
    )

    assert screen.crop_box is None


def test_desktop_timeline_strips_complete_and_truncated_tool_markup():
    pure_tool = (
        "<tool_call>\n"
        "<function=computer_use>\n"
        "<parameter=actions>[{\"type\":\"click\",\"x\":1,\"y\":2}]"
    )
    intent_plus_tool = (
        "Previous state: the input is ready.\n"
        "Next action: click send.\n"
        "<tool_call><function=computer_use>"
    )

    assert GuiAgentDesktopClient._strip_tool_markup(pure_tool) == ""
    assert GuiAgentDesktopClient._strip_tool_markup(intent_plus_tool) == (
        "Previous state: the input is ready.\nNext action: click send."
    )


def test_desktop_timeline_strips_orphan_reasoning_tags():
    assert GuiAgentDesktopClient._strip_reasoning_markup("</think>") == ""
    assert (
        GuiAgentDesktopClient._strip_tool_markup(
            "</think>\nPrevious state: ready.\n<tool_call><function=computer_use>"
        )
        == "Previous state: ready."
    )


def test_desktop_timeline_uses_friendly_generic_tool_names():
    ui = object.__new__(GuiAgentDesktopClient)

    assert GuiAgentDesktopClient._agent_tool_timeline_title(ui, "open_url") == "打开网页"
    assert GuiAgentDesktopClient._agent_tool_timeline_title(ui, "run_shell") == "执行命令"
    assert (
        GuiAgentDesktopClient._agent_tool_timeline_title(ui, "write_text_file")
        == "写入文件"
    )


def test_desktop_response_protocol_output_renders_reasoning_content():
    ui = object.__new__(GuiAgentDesktopClient)
    response_object = {
        "output": [
            {
                "type": "reasoning",
                "content": [
                    {"type": "reasoning_text", "text": "先确认输入框，再提交。"}
                ],
            }
        ]
    }

    text = GuiAgentDesktopClient._format_response_protocol_output(ui, response_object)

    assert "reasoning" in text
    assert "先确认输入框" in text
