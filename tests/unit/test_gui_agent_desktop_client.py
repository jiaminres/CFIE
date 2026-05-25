from __future__ import annotations

from cfie_gui_agent.desktop_client import (
    DIRECT_COMMAND_DO_NOT_REPLY,
    DesktopClientState,
    MacroConfig,
    TargetAppConfig,
    build_initial_state,
    build_parser,
    build_workflow_run_command,
    normalize_window_title_pattern,
    parse_macro_sequence,
)
from cfie_gui_agent.desktop_client_ui import GuiAgentDesktopClient
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


def test_trace_store_loads_existing_jsonl(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        '{"time_unix_nano":1,"kind":"step","payload":{"step_id":1}}\n'
        '{"time_unix_nano":2,"kind":"workflow_result","payload":{"item_id":"q1"}}\n',
        encoding="utf-8",
    )
    store = AgentTraceStore()

    loaded = store.load_existing(trace_path)

    assert loaded == 2
    assert store.path == trace_path
    assert [event.kind for event in store.events] == ["step", "workflow_result"]


def test_desktop_client_cli_builds_workflow_state_and_loads_trace(tmp_path):
    input_path = tmp_path / "items.jsonl"
    trace_path = tmp_path / "trace.jsonl"
    input_path.write_text(
        '{"item_id":"q1","input_text":"What is 2+2?","expected_output":"4"}\n',
        encoding="utf-8",
    )
    trace_path.write_text(
        '{"time_unix_nano":1,"kind":"workflow_result","payload":{"item_id":"q1"}}\n',
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        [
            "--app-name",
            "Web App",
            "--target-url",
            "https://example.com/",
            "--input-path",
            str(input_path),
            "--trace-path",
            str(trace_path),
            "--limit",
            "1",
        ]
    )

    state = build_initial_state(args)

    assert len(state.target_apps) == 1
    assert len(state.workflow_runs) == 1
    assert state.trace_store.path == trace_path
    assert [event.kind for event in state.trace_store.events] == ["workflow_result"]


def test_build_workflow_run_command_uses_manual_viewport():
    config = TargetAppConfig(
        app_id="app_web",
        app_name="Web",
        job_id="job_web",
        metadata={
            "target_url": "https://example.com/",
            "input_path": "items.jsonl",
            "trace_path": "runs/trace.jsonl",
            "expected_item_count": 2,
            "manual_viewport": {"x": 10, "y": 20, "width": 800, "height": 500},
        },
    )

    command = build_workflow_run_command(
        config,
        python_executable="python",
        result_json="runs/result.json",
    )

    assert command[:2] == ["python", "benchmarks/run_gui_agent_workflow_responses.py"]
    assert "--screenshot-crop" in command
    assert command[command.index("--screenshot-crop") + 1] == "10,20,800,500"
    assert "--focus-window-title-pattern" not in command
    assert command[command.index("--item-limit") + 1] == "2"
    assert command[command.index("--result-json") + 1] == "runs/result.json"


def test_normalize_window_title_pattern_drops_broken_placeholder_regex():
    assert normalize_window_title_pattern("??|Doubao") == "Doubao"


def test_build_workflow_run_command_sanitizes_window_title_pattern():
    config = TargetAppConfig(
        app_id="app_web",
        app_name="Web",
        job_id="job_web",
        metadata={
            "target_url": "https://example.com/",
            "input_path": "items.jsonl",
            "trace_path": "runs/trace.jsonl",
            "window_title_pattern": "??|Doubao",
        },
    )

    command = build_workflow_run_command(config, python_executable="python")

    assert command[command.index("--focus-window-title-pattern") + 1] == "Doubao"


def test_build_workflow_run_command_defaults_to_data_screenshot_urls():
    config = TargetAppConfig(
        app_id="app_web",
        app_name="Web",
        job_id="job_web",
        metadata={
            "target_url": "https://example.com/",
            "input_path": "items.jsonl",
            "trace_path": "runs/trace.jsonl",
            "window_title_pattern": "Edge",
        },
    )

    command = build_workflow_run_command(config, python_executable="python")

    assert command[command.index("--screenshot-url-mode") + 1] == "data"
    assert command[command.index("--screenshot-grid") + 1] == "off"


def test_completed_workflow_hides_stale_running_card():
    class Event:
        def __init__(self, kind, payload):
            self.kind = kind
            self.payload = payload

    running = Event(
        "operation",
        {
            "app_id": "app_web",
            "kind": "workflow",
            "status": "running",
            "payload": {
                "result_json": "runs/result.client_run_20260525_010101.json",
            },
        },
    )
    finished = Event(
        "operation",
        {
            "app_id": "app_web",
            "kind": "workflow",
            "status": "waiting_human",
            "payload": {
                "result_json": "runs/result.client_run_20260525_010101.json",
            },
        },
    )

    events = GuiAgentDesktopClient._dedupe_workflow_config_events([running, finished])

    assert events == [finished]


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
