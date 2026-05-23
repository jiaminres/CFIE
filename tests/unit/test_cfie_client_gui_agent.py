from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from cfie_client import ComputerLoop, Qwen35ComputerAdapter, ScreenshotResult
from cfie_client.executor import ComputerBackend
from cfie_client.protocol import ComputerAction, ComputerCall
from cfie_gui_agent import (
    GuiAgentRunner,
    GuiAgentTaskSpec,
    ToolRegistryError,
    WorkspaceProfile,
)


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


def test_protocol_accepts_single_action_dict_wait_and_key_chord():
    call = ComputerCall.from_openai(
        {
            "type": "computer_call",
            "call_id": "call_1",
            "actions": {"type": "wait", "seconds": 0.25},
        }
    )
    key_action = ComputerAction.from_openai(
        {"type": "keypress", "keys": "CTRL+L"}
    )

    assert call.actions[0].duration == 0.25
    assert key_action.keys == ("CTRL", "L")


def test_qwen_adapter_extracts_computer_call_from_json_text():
    adapter = Qwen35ComputerAdapter()
    call = adapter.to_computer_call_from_text(
        """
        ```json
        {
          "actions": [
            {"type": "click", "x": 12, "y": 34},
            {"type": "type", "text": "hello"}
          ]
        }
        ```
        """,
        default_call_id="call_from_text",
    )

    assert call.call_id == "call_from_text"
    assert [action.type for action in call.actions] == ["click", "type"]
    assert call.actions[0].x == 12
    assert call.actions[1].text == "hello"


def test_qwen_adapter_normalizes_action_and_coordinate_fields():
    adapter = Qwen35ComputerAdapter()
    call = adapter.to_computer_call_from_text(
        '{"actions":[{"action":"click","coordinate":[650,260]},'
        '{"action":"type","text":"hello cfie"}]}'
    )

    assert call.actions[0].type == "click"
    assert call.actions[0].x == 650
    assert call.actions[0].y == 260
    assert call.actions[1].type == "type"


def test_gui_agent_runner_executes_computer_call_until_final_message():
    backend = FakeBackend()
    loop = ComputerLoop(backend=backend, screen=FakeScreen())
    runner = GuiAgentRunner(computer_loop=loop, max_steps=3)
    task = GuiAgentTaskSpec(task_id="task_1", instruction="Open the browser.")

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        if len(conversation) == 2:
            assert conversation[0]["role"] == "developer"
            assert "Runtime context JSON" in conversation[0]["content"][0]["text"]
            first = conversation[1]
            assert first["role"] == "user"
            assert first["content"][1]["type"] == "input_image"
            return {
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "call_1",
                        "actions": [
                            {"type": "click", "x": 10, "y": 20},
                            {"type": "wait", "seconds": 0.1},
                        ],
                    }
                ]
            }

        assert conversation[-1]["type"] == "computer_call_output"
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Done."}],
                }
            ]
        }

    result = runner.run_task(task, agent)

    assert result.status == "completed"
    assert result.final_text == "Done."
    assert result.steps == 2
    assert result.metadata["active_job_id"] == "job:task_1"
    assert result.metadata["job_board"]["active_job_id"] == "job:task_1"
    assert (
        result.metadata["jobs"]["job:task_1"]["subtasks"]["completed"][0][
            "subtask_id"
        ]
        == "subtask:task_1:root"
    )
    assert result.metadata["prompt_context"]["selected_frame_count"] >= 1
    assert result.metadata["step_records"][0]["before_ref"].startswith("data:image")
    assert "verification" in result.metadata["step_records"][0]["metadata"]
    assert result.metadata["trace"]["event_count"] == 1
    assert "computer_use" in result.metadata["model_tools"]
    assert ("click", (10, 20, "left")) in backend.calls
    assert ("wait", (0.1,)) in backend.calls


def test_gui_agent_runner_builds_jobs_from_workspace_profile():
    loop = ComputerLoop(backend=FakeBackend(), screen=FakeScreen())
    runner = GuiAgentRunner(computer_loop=loop, max_steps=1)
    task = GuiAgentTaskSpec(
        task_id="multi_app",
        instruction="Handle workspaces.",
        workspace_profile=WorkspaceProfile(
            profile_id="ops",
            name="Operations",
            target_apps=("QianNiu", "WeChat"),
        ),
    )

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Ready."}],
                }
            ]
        }

    result = runner.run_task(task, agent)

    assert set(result.metadata["jobs"]) == {
        "job:QianNiu",
        "job:WeChat",
        "job:monitor",
    }
    assert result.metadata["active_job_id"] == "job:QianNiu"
    assert result.metadata["jobs"]["job:monitor"]["target_app"] == "monitor"


def test_gui_agent_runner_handles_finish_subtask_tool_call():
    loop = ComputerLoop(backend=FakeBackend(), screen=FakeScreen())
    runner = GuiAgentRunner(computer_loop=loop, max_steps=3)
    task = GuiAgentTaskSpec(task_id="task_finish", instruction="Finish cleanly.")

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        if len(conversation) == 2:
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "finish_subtask",
                        "call_id": "finish_1",
                        "arguments": {"completion_reason": "state verified"},
                    }
                ]
            }
        assert conversation[-1]["type"] == "function_call_output"
        assert conversation[-1]["output"]["status"] == "accepted"
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Finished."}],
                }
            ]
        }

    result = runner.run_task(task, agent)

    completed = result.metadata["jobs"]["job:task_finish"]["subtasks"]["completed"]
    assert result.status == "completed"
    assert completed[0]["subtask_id"] == "subtask:task_finish:root"
    assert result.metadata["step_records"][0]["action"]["name"] == "finish_subtask"


def test_gui_agent_runner_handles_human_help_tool_call():
    loop = ComputerLoop(backend=FakeBackend(), screen=FakeScreen())
    runner = GuiAgentRunner(computer_loop=loop, max_steps=3)
    task = GuiAgentTaskSpec(task_id="task_human", instruction="Ask human.")

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        if len(conversation) == 2:
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "request_human_help",
                        "call_id": "human_1",
                        "arguments": {
                            "question": "Should I refund this order?",
                            "urgency": "high",
                            "evidence_refs": ["screen.png"],
                        },
                    }
                ]
            }
        assert conversation[-1]["type"] == "function_call_output"
        assert conversation[-1]["output"]["status"] == "waiting_human"
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Waiting."}],
                }
            ]
        }

    result = runner.run_task(task, agent)

    waiting = result.metadata["jobs"]["job:task_human"]["subtasks"]["waiting_human"]
    request_id = next(iter(waiting))
    assert result.status == "completed"
    assert waiting[request_id]["human_request_id"] == request_id
    assert runner.human_loop.pending[request_id].question == "Should I refund this order?"


def test_gui_agent_runner_rejects_invalid_agent_tool_arguments():
    loop = ComputerLoop(backend=FakeBackend(), screen=FakeScreen())
    runner = GuiAgentRunner(computer_loop=loop, max_steps=1)
    task = GuiAgentTaskSpec(task_id="task_invalid", instruction="Ask badly.")

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "output": [
                {
                    "type": "function_call",
                    "name": "request_human_help",
                    "call_id": "human_bad",
                    "arguments": {"urgency": "high"},
                }
            ]
        }

    with pytest.raises(ToolRegistryError):
        runner.run_task(task, agent)


def test_gui_agent_runner_persists_update_constraints_tool_call():
    loop = ComputerLoop(backend=FakeBackend(), screen=FakeScreen())
    runner = GuiAgentRunner(computer_loop=loop, max_steps=3)
    task = GuiAgentTaskSpec(task_id="task_policy", instruction="Update policy.")

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        if len(conversation) == 2:
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "update_constraints",
                        "call_id": "policy_1",
                        "arguments": {
                            "summary": "Avoid risky action.",
                            "constraints": {
                                "current_job": [
                                    {
                                        "text": "Ask before refunding.",
                                        "severity": "high",
                                    }
                                ]
                            },
                            "reason": "user correction",
                        },
                    }
                ]
            }
        assert conversation[-1]["type"] == "function_call_output"
        assert conversation[-1]["output"]["status"] == "accepted"
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Policy updated."}],
                }
            ]
        }

    result = runner.run_task(task, agent)

    assert result.status == "completed"
    assert result.metadata["policy"]["rules"][0]["text"] == "Ask before refunding."
    assert result.metadata["policy"]["rules"][0]["severity"] == "high"
    assert result.metadata["trace"]["event_count"] == 2
