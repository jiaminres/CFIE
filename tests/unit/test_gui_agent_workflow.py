from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cfie_client import ComputerLoop, ScreenshotResult
from cfie_client.executor import ComputerBackend
from cfie_gui_agent import (
    GuiAgentRunner,
    GuiAgentTaskSpec,
    ModelToolRegistry,
    build_workflow_target_config,
    load_workflow_items,
)
from cfie_gui_agent.desktop_client import DesktopClientState


@dataclass
class FakeBackend(ComputerBackend):
    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    def move(self, x: int, y: int) -> None:
        self.calls.append(("move", (x, y)))

    def click(self, x: int, y: int, button: str = "left") -> None:
        self.calls.append(("click", (x, y, button)))

    def double_click(self, x: int, y: int, button: str = "left") -> None:
        self.calls.append(("double_click", (x, y, button)))

    def drag(self, path: tuple[tuple[int, int], ...]) -> None:
        self.calls.append(("drag", (path,)))

    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self.calls.append(("scroll", (x, y, scroll_x, scroll_y)))

    def type_text(self, text: str) -> None:
        self.calls.append(("type_text", (text,)))

    def press_keys(self, keys: tuple[str, ...]) -> None:
        self.calls.append(("press_keys", (keys,)))

    def key_down(self, key: str) -> None:
        self.calls.append(("key_down", (key,)))

    def key_up(self, key: str) -> None:
        self.calls.append(("key_up", (key,)))

    def wait(self, seconds: float) -> None:
        self.calls.append(("wait", (seconds,)))


class FakeScreen:
    def screenshot(self) -> ScreenshotResult:
        return ScreenshotResult(
            image_url="data:image/png;base64,AAAA",
            width=800,
            height=600,
        )

    def size(self) -> tuple[int, int]:
        return (800, 600)


def test_load_workflow_items_accepts_question_answer_jsonl(tmp_path: Path):
    input_file = tmp_path / "items.jsonl"
    input_file.write_text(
        "\n".join(
            [
                '{"Task ID":"1","Question":"What is 2+2?","Final answer":"4"}',
                '{"task_id":"2","question":"Capital of France?","answer":"Paris"}',
            ]
        ),
        encoding="utf-8",
    )

    items = load_workflow_items(input_file)

    assert [item.item_id for item in items] == ["1", "2"]
    assert items[0].input_text == "What is 2+2?"
    assert items[0].expected_output == "4"
    assert items[1].input_text == "Capital of France?"


def test_build_workflow_target_config_records_browser_and_trace_metadata():
    config = build_workflow_target_config(
        app_name="Web App",
        target_url="https://example.com/",
        input_path="datasets/items.jsonl",
        trace_path="runs/workflow_trace.jsonl",
        item_count=3,
        window_title_pattern=".*",
    )

    assert config.app_id.startswith("app_")
    assert config.job_id.startswith("job_")
    assert "执行自动化任务流" in config.task_description
    assert config.metadata["process_name"] == "chrome.exe"
    assert config.metadata["browser_url_pattern"] == "https://example.com/"
    assert config.metadata["expected_item_count"] == 3
    assert config.metadata["trace_path"] == "runs/workflow_trace.jsonl"


def test_desktop_state_configures_workflow_and_records_operation(tmp_path: Path):
    input_file = tmp_path / "items.csv"
    input_file.write_text(
        "id,question,expected_answer\nq1,What is 1+1?,2\n",
        encoding="utf-8",
    )
    trace = tmp_path / "trace.jsonl"
    state = DesktopClientState()

    result = state.configure_workflow(
        app_name="Web App",
        target_url="https://example.com/",
        input_path=str(input_file),
        trace_path=str(trace),
    )

    assert result["item_count"] == 1
    assert result["app"]["app_name"] == "Web App"
    assert Path(result["manifest_path"]).exists()
    assert state.trace_store.path == trace
    assert state.trace_store.events[-1].kind == "operation"
    assert state.trace_store.events[-1].payload["kind"] == "workflow"
    assert trace.exists()


def test_model_tool_registry_allows_workflow_trace_tools():
    registry = ModelToolRegistry()

    registry.validate_model_tool_call(
        "append_trace_note",
        {"title": "Opened target page", "summary": "Chrome is ready."},
    )
    registry.validate_model_tool_call(
        "record_workflow_result",
        {
            "item_id": "q1",
            "output_text": "4",
            "status": "passed",
        },
    )
    registry.validate_model_tool_call(
        "read_text_file",
        {"path": "questions.jsonl", "max_chars": 1024},
    )
    registry.validate_model_tool_call(
        "set_app_viewport",
        {
            "x": 10,
            "y": 20,
            "width": 800,
            "height": 500,
            "coordinate_space": "screenshot",
        },
    )


def test_runner_handles_workflow_file_and_trace_tools(tmp_path: Path):
    text_file = tmp_path / "items.txt"
    text_file.write_text("input_text: What is 2+2?", encoding="utf-8")
    runner = GuiAgentRunner(
        computer_loop=ComputerLoop(backend=FakeBackend(), screen=FakeScreen()),
        max_steps=4,
    )
    task = GuiAgentTaskSpec(task_id="workflow", instruction="Run workflow.")

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        if len(conversation) == 2:
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "read_text_file",
                        "call_id": "read_1",
                        "arguments": {"path": str(text_file), "max_chars": 32},
                    }
                ]
            }
        if conversation[-1]["call_id"] == "read_1":
            assert conversation[-1]["output"]["status"] == "accepted"
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "record_workflow_result",
                        "call_id": "save_1",
                        "arguments": {
                            "item_id": "q1",
                            "input_text": "What is 2+2?",
                            "expected_output": "4",
                            "output_text": "4",
                            "status": "passed",
                        },
                    }
                ]
            }
        if conversation[-1]["call_id"] == "save_1":
            assert conversation[-1]["output"]["status"] == "accepted"
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Done"}],
                    }
                ]
            }
        raise AssertionError("unexpected conversation state")

    result = runner.run_task(task, agent)

    assert result.status == "completed"
    assert [event.kind for event in runner.trace_store.events] == [
        "model_response",
        "step",
        "model_response",
        "workflow_result",
        "step",
        "model_response",
    ]
    assert [
        event.kind
        for event in runner.trace_store.events
        if event.kind != "model_response"
    ] == ["step", "workflow_result", "step"]


def test_runner_records_final_message_json_as_workflow_result(tmp_path: Path):
    input_file = tmp_path / "items.jsonl"
    input_file.write_text(
        '{"item_id":"q1","input_text":"What is 2+2?","expected_output":"4"}',
        encoding="utf-8",
    )
    runner = GuiAgentRunner(
        computer_loop=ComputerLoop(backend=FakeBackend(), screen=FakeScreen()),
        max_steps=1,
    )
    task = GuiAgentTaskSpec(
        task_id="workflow_json_final",
        instruction="Run workflow.",
        metadata={"input_path": str(input_file)},
    )

    def agent(_conversation: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                '{"item_id":"q1","output_text":"4",'
                                '"status":"passed","reason":"correct"}'
                            ),
                        }
                    ],
                }
            ]
        }

    result = runner.run_task(task, agent)

    workflow_events = [
        event.payload
        for event in runner.trace_store.events
        if event.kind == "workflow_result"
    ]
    assert result.status == "completed"
    assert len(workflow_events) == 1
    assert workflow_events[0]["input_text"] == "What is 2+2?"
    assert workflow_events[0]["expected_output"] == "4"
    assert workflow_events[0]["source"] == "final_message_json"
