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
    assert "执行当前任务流" in config.task_description
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


def test_runner_retries_reasoning_only_response_for_active_task():
    runner = GuiAgentRunner(
        computer_loop=ComputerLoop(backend=FakeBackend(), screen=FakeScreen()),
        max_steps=1,
    )
    task = GuiAgentTaskSpec(
        task_id="reasoning_only_retry",
        instruction="Keep working until done.",
    )
    calls = 0

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "output": [
                    {
                        "type": "reasoning",
                        "content": [
                            {
                                "type": "reasoning_text",
                                "text": "I am thinking but not acting yet.",
                            }
                        ],
                    }
                ]
            }
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Done"}],
                }
            ]
        }

    result = runner.run_task(task, agent)

    retry_events = [
        event.payload
        for event in runner.trace_store.events
        if event.kind == "tool_call_parse_retry"
    ]
    assert result.status == "completed"
    assert calls == 2
    assert retry_events[-1]["reason"] == "empty_response_without_tool_call"


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


def test_runner_executes_only_one_tool_when_model_returns_parallel_calls(
    tmp_path: Path,
):
    text_file = tmp_path / "items.txt"
    text_file.write_text("input_text: What is 2+2?", encoding="utf-8")
    backend = FakeBackend()
    runner = GuiAgentRunner(
        computer_loop=ComputerLoop(backend=backend, screen=FakeScreen()),
        max_steps=3,
    )
    task = GuiAgentTaskSpec(task_id="workflow", instruction="Run workflow.")
    saw_read_output = False

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal saw_read_output
        if len(conversation) == 2:
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "computer_use",
                        "call_id": "click_1",
                        "arguments": {
                            "coordinate_space": "qwen_normalized_1000",
                            "actions": [{"type": "click", "x": 500, "y": 900}],
                        },
                    },
                    {
                        "type": "function_call",
                        "name": "read_text_file",
                        "call_id": "read_1",
                        "arguments": {"path": str(text_file), "max_chars": 32},
                    },
                ]
            }
        assert not backend.calls
        assert not any(
            item.get("call_id") == "click_1"
            for item in conversation
            if isinstance(item, dict)
        )
        assert conversation[-2]["type"] == "function_call"
        assert conversation[-2]["name"] == "read_text_file"
        assert conversation[-1]["type"] == "function_call_output"
        assert conversation[-1]["call_id"] == "read_1"
        saw_read_output = True
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Done"}],
                }
            ]
        }

    result = runner.run_task(task, agent)

    assert result.status == "completed"
    assert saw_read_output
    assert backend.calls == []
    assert any(
        event.kind == "parallel_tool_call_pruned"
        for event in runner.trace_store.events
    )


def test_runner_requires_workflow_input_read_before_computer_actions(
    tmp_path: Path,
):
    text_file = tmp_path / "items.txt"
    text_file.write_text("input_text: What is 2+2?", encoding="utf-8")
    backend = FakeBackend()
    runner = GuiAgentRunner(
        computer_loop=ComputerLoop(backend=backend, screen=FakeScreen()),
        max_steps=4,
    )
    task = GuiAgentTaskSpec(
        task_id="workflow",
        instruction="Run workflow.",
        metadata={"input_path": str(text_file)},
    )
    saw_guard = False

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal saw_guard
        if not saw_guard:
            if any(
                "必须先调用 read_text_file" in str(part.get("text", ""))
                for message in conversation
                if isinstance(message, dict)
                for part in message.get("content", [])
                if isinstance(part, dict)
            ):
                saw_guard = True
                assert backend.calls == []
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
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "computer_use",
                        "call_id": "click_1",
                        "arguments": {
                            "coordinate_space": "qwen_normalized_1000",
                            "actions": [{"type": "click", "x": 500, "y": 900}],
                        },
                    }
                ]
            }
        if conversation[-1]["type"] == "function_call_output":
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
    assert saw_guard
    assert backend.calls == []
    assert any(
        event.kind == "workflow_input_read_required"
        for event in runner.trace_store.events
    )


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

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        if (
            conversation
            and conversation[-1].get("type") == "function_call_output"
            and str(conversation[-1].get("call_id", "")).startswith(
                "call_recovered_record_"
            )
        ):
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Done"}],
                    }
                ]
            }
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


def test_runner_recovers_workflow_result_intent_from_text_with_wrong_tool(
    tmp_path: Path,
):
    input_file = tmp_path / "items.jsonl"
    input_file.write_text(
        '{"item_id":"q1","input_text":"How many albums?","expected_output":"4"}',
        encoding="utf-8",
    )
    runner = GuiAgentRunner(
        computer_loop=ComputerLoop(backend=FakeBackend(), screen=FakeScreen()),
        max_steps=2,
    )
    task = GuiAgentTaskSpec(
        task_id="workflow_recover",
        instruction="Run workflow.",
        metadata={"input_path": str(input_file)},
    )
    calls = 0

    def agent(_conversation: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    "上一步状态：已获取答案“4 studio albums”。"
                                    "下一步动作：调用 record_workflow_result 记录结果。"
                                ),
                            }
                        ],
                    },
                    {
                        "type": "function_call",
                        "name": "computer_use",
                        "call_id": "bad_click",
                        "arguments": {
                            "coordinate_space": "qwen_normalized_1000",
                            "actions": [{"type": "click", "x": 700, "y": 930}],
                        },
                    },
                ]
            }
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Done"}],
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
    assert workflow_events[0]["item_id"] == "q1"
    assert workflow_events[0]["output_text"] == "4"
    assert not [
        event
        for event in runner.trace_store.events
        if event.kind == "step" and "computer_use" in event.payload.get("tags", ())
    ]


def test_runner_recovers_workflow_result_from_natural_language_intent(
    tmp_path: Path,
):
    input_file = tmp_path / "items.jsonl"
    input_file.write_text(
        '{"item_id":"q1","input_text":"How many albums?","expected_output":"4"}',
        encoding="utf-8",
    )
    runner = GuiAgentRunner(
        computer_loop=ComputerLoop(backend=FakeBackend(), screen=FakeScreen()),
        max_steps=2,
    )
    task = GuiAgentTaskSpec(
        task_id="workflow_recover_natural_language",
        instruction="Run workflow.",
        metadata={"input_path": str(input_file)},
    )

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        if (
            conversation
            and conversation[-1].get("type") == "function_call_output"
            and str(conversation[-1].get("call_id", "")).startswith(
                "call_recovered_record_"
            )
        ):
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Done"}],
                    }
                ]
            }
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                "上一步状态：第一个问题已得到回答，显示发布了4张专辑。"
                                "下一步动作：记录第一个任务结果，然后开始处理第二个问题。"
                            ),
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "name": "computer_use",
                    "call_id": "bad_click",
                    "arguments": {
                        "coordinate_space": "qwen_normalized_1000",
                        "actions": [{"type": "click", "x": 500, "y": 925}],
                    },
                },
            ]
        }

    result = runner.run_task(task, agent)

    workflow_events = [
        event.payload
        for event in runner.trace_store.events
        if event.kind == "workflow_result"
    ]
    assert result.status == "completed"
    assert workflow_events[0]["item_id"] == "q1"
    assert workflow_events[0]["output_text"] == "4"


def test_runner_recovers_failed_workflow_status_for_next_unrecorded_item(
    tmp_path: Path,
):
    input_file = tmp_path / "items.jsonl"
    input_file.write_text(
        "\n".join(
            [
                '{"item_id":"q1","input_text":"first","expected_output":"1"}',
                '{"item_id":"q2","input_text":"video","expected_output":"2"}',
            ]
        ),
        encoding="utf-8",
    )
    runner = GuiAgentRunner(
        computer_loop=ComputerLoop(backend=FakeBackend(), screen=FakeScreen()),
        max_steps=3,
    )
    task = GuiAgentTaskSpec(
        task_id="workflow_recover_failed_status",
        instruction="Run workflow.",
        metadata={"input_path": str(input_file)},
    )
    calls = 0

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal calls
        if (
            conversation
            and conversation[-1].get("type") == "function_call_output"
            and str(conversation[-1].get("call_id", "")).startswith(
                "call_recovered_record_"
            )
        ):
            calls += 1
            if calls == 1:
                return {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": (
                                        "The second item failed because YouTube "
                                        "cannot be accessed in this region. "
                                        "Next action: record the second item "
                                        "failed status."
                                    ),
                                }
                            ],
                        },
                        {
                            "type": "function_call",
                            "name": "computer_use",
                            "call_id": "bad_click_2",
                            "arguments": {
                                "coordinate_space": "qwen_normalized_1000",
                                "actions": [{"type": "click", "x": 550, "y": 915}],
                            },
                        },
                    ]
                }
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Done"}],
                    }
                ]
            }
        return {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                "The first item has been answered; answer is 1. "
                                "Next action: record the first item result."
                            ),
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "name": "computer_use",
                    "call_id": "bad_click_1",
                    "arguments": {
                        "coordinate_space": "qwen_normalized_1000",
                        "actions": [{"type": "click", "x": 500, "y": 925}],
                    },
                },
            ]
        }

    result = runner.run_task(task, agent)

    workflow_events = [
        event.payload
        for event in runner.trace_store.events
        if event.kind == "workflow_result"
    ]
    assert result.status == "completed"
    assert [event["item_id"] for event in workflow_events] == ["q1", "q2"]
    assert workflow_events[0]["output_text"] == "1"
    assert workflow_events[0]["status"] == "passed"
    assert workflow_events[1]["status"] == "failed"
    assert "cannot be accessed" in workflow_events[1]["output_text"]
