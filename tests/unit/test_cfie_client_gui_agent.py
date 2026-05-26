from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest
from PIL import Image

import cfie_client.screen as screen_mod
from cfie_client import (
    ComputerLoop,
    CoordinateScalingBackend,
    Qwen35ComputerAdapter,
    ScreenshotResult,
)
from cfie_client.executor import ComputerBackend
from cfie_client.protocol import ComputerAction, ComputerCall
from cfie_gui_agent import (
    ActionMacro,
    ActionMacroRegistry,
    ActionMacroStep,
    GuiAgentRunner,
    GuiAgentTaskSpec,
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


class FakeScaledScreen:
    def __init__(
        self,
        *,
        logical_size: tuple[int, int] = (400, 300),
        physical_size: tuple[int, int] = (800, 600),
        physical_origin: tuple[int, int] = (100, 50),
    ) -> None:
        self.logical_size = logical_size
        self._physical_size = physical_size
        self._physical_origin = physical_origin
        self.crop_box: tuple[int, int, int, int] | None = None

    def screenshot(self) -> ScreenshotResult:
        width, height = self.size()
        return ScreenshotResult(
            image_url="data:image/png;base64,BBBB",
            width=width,
            height=height,
        )

    def size(self) -> tuple[int, int]:
        physical_width, physical_height = self.physical_size()
        max_width, max_height = self.logical_size
        scale = min(max_width / physical_width, max_height / physical_height, 1.0)
        return (
            max(1, int(physical_width * scale)),
            max(1, int(physical_height * scale)),
        )

    def physical_size(self) -> tuple[int, int]:
        if self.crop_box is not None:
            return (self.crop_box[2], self.crop_box[3])
        return self._physical_size

    def physical_origin(self) -> tuple[int, int]:
        if self.crop_box is not None:
            return (self.crop_box[0], self.crop_box[1])
        return self._physical_origin

    def set_crop_box(self, crop_box: tuple[int, int, int, int] | None) -> None:
        self.crop_box = crop_box

    def viewport_context(self) -> dict[str, int | None]:
        physical_width, physical_height = self.physical_size()
        logical_width, logical_height = self.size()
        origin_x, origin_y = self.physical_origin()
        return {
            "crop_x": self.crop_box[0] if self.crop_box else None,
            "crop_y": self.crop_box[1] if self.crop_box else None,
            "crop_width": self.crop_box[2] if self.crop_box else None,
            "crop_height": self.crop_box[3] if self.crop_box else None,
            "physical_origin_x": origin_x,
            "physical_origin_y": origin_y,
            "physical_width": physical_width,
            "physical_height": physical_height,
            "screenshot_width": logical_width,
            "screenshot_height": logical_height,
        }


def test_protocol_accepts_single_action_dict_wait_and_key_chord():
    call = ComputerCall.from_openai(
        {
            "type": "computer_call",
            "call_id": "call_1",
            "coordinate_space": "qwen_normalized_1000",
            "actions": {"type": "wait", "seconds": 0.25},
        }
    )
    key_action = ComputerAction.from_openai(
        {"type": "keypress", "keys": "CTRL+L"}
    )
    key_alias_action = ComputerAction.from_openai(
        {"type": "key", "keys": ["enter"]}
    )
    key_press_alias_action = ComputerAction.from_openai(
        {"type": "key_press", "keys": ["Enter"]}
    )
    keys_alias_action = ComputerAction.from_openai(
        {"type": "keys", "keys": ["ctrl", "a"]}
    )

    assert call.actions[0].duration == 0.25
    assert call.coordinate_space == "qwen_normalized_1000"
    assert key_action.keys == ("CTRL", "L")
    assert key_alias_action.type == "keypress"
    assert key_alias_action.keys == ("enter",)
    assert key_press_alias_action.type == "keypress"
    assert key_press_alias_action.keys == ("Enter",)
    assert keys_alias_action.type == "keypress"
    assert keys_alias_action.keys == ("ctrl", "a")


def test_protocol_unquotes_coordinate_space_from_qwen_text_tool_output():
    call = ComputerCall.from_openai(
        {
            "type": "computer_call",
            "call_id": "call_quoted_space",
            "coordinate_space": '"qwen_normalized_1000"',
            "actions": {"type": "click", "x": 500, "y": 900},
        }
    )

    assert call.coordinate_space == "qwen_normalized_1000"


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


def test_coordinate_scaling_backend_maps_logical_to_physical_coordinates():
    backend = FakeBackend()
    scaled = CoordinateScalingBackend(
        backend,
        logical_size=(960, 540),
        physical_size=(1920, 1080),
        physical_offset=(100, 50),
    )

    scaled.click(480, 270)
    scaled.drag(((0, 0), (960, 540)))

    assert backend.calls[0] == ("click", (1060, 590, "left"))
    assert backend.calls[1] == ("drag", (((100, 50), (2020, 1130)),))


def test_computer_loop_maps_scaled_screenshot_coordinates_to_physical_screen():
    backend = FakeBackend()
    screen = FakeScaledScreen()
    loop = ComputerLoop(backend=backend, screen=screen)
    call = ComputerCall(
        call_id="scaled_call",
        actions=(
            ComputerAction(type="click", x=200, y=150, button="left"),
            ComputerAction(type="drag", path=((0, 0), (399, 299))),
        ),
    )

    loop.handle_call(call)

    assert backend.calls[0] == ("click", (500, 350, "left"))
    assert backend.calls[1] == ("drag", (((100, 50), (898, 648)),))


def test_computer_loop_can_map_qwen_normalized_coordinates_to_screenshot_pixels():
    backend = FakeBackend()
    screen = FakeScreen()
    loop = ComputerLoop(
        backend=backend,
        screen=screen,
        model_coordinate_mode="qwen_normalized_1000",
    )
    call = ComputerCall(
        call_id="qwen_grounding",
        actions=(ComputerAction(type="click", x=700, y=950, button="left"),),
    )

    loop.handle_call(call)

    assert backend.calls == [("click", (560, 570, "left"))]


def test_computer_call_coordinate_space_overrides_loop_default():
    backend = FakeBackend()
    screen = FakeScreen()
    loop = ComputerLoop(backend=backend, screen=screen)
    call = ComputerCall(
        call_id="qwen_grounding_override",
        coordinate_space="qwen_normalized_1000",
        actions=(ComputerAction(type="click", x=700, y=950, button="left"),),
    )

    loop.handle_call(call)

    assert backend.calls == [("click", (560, 570, "left"))]


def test_computer_loop_defaults_scroll_to_viewport_center():
    backend = FakeBackend()
    screen = FakeScreen()
    loop = ComputerLoop(backend=backend, screen=screen)
    call = ComputerCall.from_openai(
        {
            "type": "computer_call",
            "call_id": "scroll_center",
            "coordinate_space": "qwen_normalized_1000",
            "actions": [{"type": "scroll", "scroll_y": -300}],
        }
    )

    loop.handle_call(call)

    assert backend.calls == [("scroll", (400, 300, 0, -300))]


def test_computer_loop_defensively_maps_out_of_bounds_screenshot_coordinates():
    backend = FakeBackend()
    screen = FakeScreen()
    loop = ComputerLoop(backend=backend, screen=screen)
    call = ComputerCall(
        call_id="qwen_grounding_mislabeled",
        coordinate_space="screenshot",
        actions=(ComputerAction(type="click", x=500, y=950, button="left"),),
    )

    loop.handle_call(call)

    assert backend.calls == [("click", (400, 570, "left"))]


def test_computer_loop_auto_maps_qwen_normalized_coordinates_when_out_of_bounds():
    backend = FakeBackend()
    screen = FakeScaledScreen(logical_size=(960, 524), physical_size=(1920, 1048))
    loop = ComputerLoop(
        backend=backend,
        screen=screen,
        model_coordinate_mode="auto",
    )
    call = ComputerCall(
        call_id="qwen_grounding_auto",
        actions=(ComputerAction(type="click", x=700, y=950, button="left"),),
    )

    loop.handle_call(call)

    assert backend.calls == [("click", (1444, 1046, "left"))]


def test_computer_loop_maps_local_refinement_coordinates_to_screen_box():
    backend = FakeBackend()
    screen = FakeScreen()
    loop = ComputerLoop(backend=backend, screen=screen)
    loop.set_local_refinement_box((100, 200, 400, 300))
    call = ComputerCall(
        call_id="local_refinement",
        coordinate_space="local_refinement_1000",
        actions=(ComputerAction(type="click", x=500, y=500, button="left"),),
    )

    loop.handle_call(call)

    assert backend.calls == [("click", (300, 350, "left"))]


def test_scaled_screen_capture_can_emit_file_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        screen_mod.ImageGrab,
        "grab",
        lambda *args, **kwargs: Image.new("RGB", (80, 40), "white"),
    )
    screen = screen_mod.ScaledPillowScreenCapture(
        max_width=50,
        max_height=50,
        url_mode="file",
        output_dir=tmp_path,
        crop_box=(10, 5, 80, 40),
    )

    shot = screen.screenshot()
    parsed = urlparse(shot.image_url)
    path = Path(url2pathname(parsed.path))

    assert parsed.scheme == "file"
    assert path.exists()
    assert (shot.width, shot.height) == (50, 25)
    assert screen.physical_size() == (80, 40)
    assert screen.physical_origin() == (10, 5)


def test_scaled_screen_capture_can_overlay_coordinate_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        screen_mod.ImageGrab,
        "grab",
        lambda *args, **kwargs: Image.new("RGB", (160, 90), "white"),
    )
    screen = screen_mod.ScaledPillowScreenCapture(
        max_width=160,
        max_height=90,
        image_format="PNG",
        url_mode="file",
        output_dir=tmp_path,
        grid_overlay="coarse",
    )

    shot = screen.screenshot()
    parsed = urlparse(shot.image_url)
    path = Path(url2pathname(parsed.path))
    image = Image.open(path).convert("RGB")

    assert screen.viewport_context()["grid_overlay"] == "coarse"
    assert image.getpixel((50, 10)) != (255, 255, 255)
    assert image.getpixel((100, 10)) != (255, 255, 255)


def test_scaled_screen_capture_draws_cursor_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    background = (120, 120, 120)
    monkeypatch.setattr(
        screen_mod.ImageGrab,
        "grab",
        lambda *args, **kwargs: Image.new("RGB", (160, 90), background),
    )
    screen = screen_mod.ScaledPillowScreenCapture(
        max_width=160,
        max_height=90,
        image_format="PNG",
        url_mode="file",
        output_dir=tmp_path,
        cursor_position_provider=lambda: (20, 20),
    )

    shot = screen.screenshot()
    parsed = urlparse(shot.image_url)
    path = Path(url2pathname(parsed.path))
    image = Image.open(path).convert("RGB")

    assert image.getpixel((20, 20)) != background


def test_scaled_screen_capture_can_emit_local_region_around_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured_bboxes: list[tuple[int, int, int, int] | None] = []

    def fake_grab(*, bbox=None):
        captured_bboxes.append(bbox)
        if bbox is None:
            return Image.new("RGB", (200, 100), "white")
        left, top, right, bottom = bbox
        return Image.new("RGB", (right - left, bottom - top), "white")

    monkeypatch.setattr(screen_mod.ImageGrab, "grab", fake_grab)
    screen = screen_mod.ScaledPillowScreenCapture(
        max_width=100,
        max_height=50,
        image_format="PNG",
        url_mode="file",
        output_dir=tmp_path,
        crop_box=(0, 0, 200, 100),
        draw_cursor=False,
    )

    shot = screen.screenshot_region_around(x=50, y=25, radius=10)

    assert shot.width == 80
    assert shot.height == 80
    assert captured_bboxes == [(60, 10, 140, 90)]


def test_gui_agent_runner_executes_computer_call_until_final_message():
    backend = FakeBackend()
    loop = ComputerLoop(backend=backend, screen=FakeScreen())
    runner = GuiAgentRunner(computer_loop=loop, max_steps=3)
    task = GuiAgentTaskSpec(task_id="task_1", instruction="Open the browser.")

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        if len(conversation) == 2:
            assert conversation[0]["role"] == "developer"
            assert "运行规则" in conversation[0]["content"][0]["text"]
            assert "active_app" in conversation[0]["content"][1]["text"]
            first = conversation[1]
            assert first["role"] == "user"
            assert first["content"][1]["type"] == "input_image"
            assert first["content"][1]["detail"] == "low"
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

        assert conversation[-3]["type"] == "computer_call"
        assert conversation[-3]["call_id"] == "call_1"
        assert conversation[-2]["type"] == "computer_call_output"
        assert conversation[-2]["output"]["detail"] == "low"
        assert "click(10,20)" in conversation[-2]["output"]["summary"]
        assert conversation[-1]["role"] == "user"
        assert "坐标提醒" in conversation[-1]["content"][0]["text"]
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
    assert result.metadata["trace"]["event_count"] == 3
    assert len(result.metadata["model_response_metrics"]) == 2
    assert result.metadata["model_response_metrics"][0]["function_call_count"] == 1
    assert "computer_use" in result.metadata["model_tools"]
    assert ("click", (10, 20, "left")) in backend.calls
    assert ("wait", (0.1,)) in backend.calls


def test_gui_agent_runner_adds_local_crop_after_click_only_action():
    class LocalCropScreen(FakeScreen):
        def __init__(self) -> None:
            self.local_requests: list[tuple[int, int, int]] = []

        def screenshot_region_around(
            self,
            *,
            x: int,
            y: int,
            radius: int = 180,
            max_width: int = 720,
            max_height: int = 720,
        ) -> ScreenshotResult:
            self.local_requests.append((x, y, radius))
            return ScreenshotResult(
                image_url="data:image/png;base64,LOCAL",
                width=320,
                height=240,
            )

    backend = FakeBackend()
    screen = LocalCropScreen()
    loop = ComputerLoop(backend=backend, screen=screen)
    runner = GuiAgentRunner(computer_loop=loop, max_steps=2)
    task = GuiAgentTaskSpec(task_id="task_refine", instruction="Click target.")
    saw_local_refinement = False

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal saw_local_refinement
        if len(conversation) == 2:
            return {
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "call_refine",
                        "coordinate_space": "qwen_normalized_1000",
                        "actions": [{"type": "click", "x": 500, "y": 500}],
                    }
                ]
            }
        saw_local_refinement = any(
            any(
                isinstance(part, dict)
                and (
                    "局部定位兜底"
                    in str(part.get("text", ""))
                    or part.get("image_url") == "data:image/png;base64,LOCAL"
                )
                for part in message.get("content", [])
            )
            for message in conversation
            if isinstance(message, dict)
        )
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
    assert saw_local_refinement is True
    assert screen.local_requests == [(400, 300, 520)]
    local_refinement = result.metadata["step_records"][0]["metadata"][
        "local_refinement"
    ]
    assert local_refinement["coordinate_space"] == "local_refinement_1000"
    assert local_refinement["local_refinement_box"] == [0, 0, 800, 600]


def test_gui_agent_runner_does_not_complete_on_truncated_text_tool_call():
    loop = ComputerLoop(backend=FakeBackend(), screen=FakeScreen())
    runner = GuiAgentRunner(computer_loop=loop, max_steps=2)
    task = GuiAgentTaskSpec(task_id="task_truncated", instruction="Click once.")
    saw_retry = False

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal saw_retry
        for message in conversation:
            for part in message.get("content", []):
                if (
                    isinstance(part, dict)
                    and "previous tool call was incomplete" in str(part.get("text", ""))
                ):
                    saw_retry = True
        if saw_retry:
            return {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ]
            }
        return {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "<tool_call><function=computer_use>",
                        }
                    ],
                }
            ]
        }

    result = runner.run_task(task, agent)

    assert saw_retry
    assert result.status == "completed"
    assert result.final_text == "done"


def test_gui_agent_runner_converts_login_computer_action_to_human_help():
    backend = FakeBackend()
    runner = GuiAgentRunner(
        computer_loop=ComputerLoop(backend=backend, screen=FakeScreen()),
        max_steps=1,
    )
    task = GuiAgentTaskSpec(
        task_id="login_blocked",
        target_app="web",
        instruction="Use the web app.",
        expected_outcome="Complete the task.",
    )
    response = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": (
                            "当前页面出现登录弹窗，需要用户登录才能继续。"
                            "根据任务要求，遇到登录弹窗应立即调用 request_human_help。\n"
                            "<tool_call><function=computer_use>"
                            "<parameter=actions>"
                            '[{"type":"click","x":10,"y":20}]'
                            "</parameter>"
                            "</function></tool_call>"
                        ),
                    }
                ],
            }
        ]
    }

    result = runner.run_task(task, lambda _conversation: response)

    assert result.status == "waiting_human"
    assert backend.calls == []
    assert any(
        event.kind == "tool_call_safety_override"
        for event in runner.trace_store.events
    )


def test_gui_agent_runner_retries_prose_plan_without_tool_call():
    loop = ComputerLoop(backend=FakeBackend(), screen=FakeScreen())
    runner = GuiAgentRunner(computer_loop=loop, max_steps=3)
    task = GuiAgentTaskSpec(task_id="task_plan", instruction="Read file.")
    saw_retry = False

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal saw_retry
        for message in conversation:
            for part in message.get("content", []):
                if isinstance(part, dict) and "wrote a plan" in str(part.get("text", "")):
                    saw_retry = True
        if saw_retry:
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "finish_subtask",
                        "call_id": "finish_1",
                        "arguments": {"completion_reason": "retry accepted"},
                    }
                ]
            }
        return {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": (
                                "Thinking Process:\n"
                                "Plan: I need to call `read_text_file` first."
                            ),
                        }
                    ],
                }
            ]
        }

    result = runner.run_task(task, agent)

    assert saw_retry
    assert result.status == "completed"
    assert any(
        event.kind == "tool_call_parse_retry"
        and event.payload["reason"] == "prose_plan_without_tool_call"
        for event in runner.trace_store.events
    )


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
        assert conversation[-2]["type"] == "function_call"
        assert conversation[-2]["name"] == "finish_subtask"
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
                            "evidence_refs": "screen.png",
                        },
                    }
                ]
            }
        assert conversation[-2]["type"] == "function_call"
        assert conversation[-2]["name"] == "request_human_help"
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
    assert result.status == "waiting_human"
    assert result.steps == 1
    assert waiting[request_id]["human_request_id"] == request_id
    assert runner.human_loop.pending[request_id].question == "Should I refund this order?"
    assert runner.human_loop.pending[request_id].evidence_refs == ("screen.png",)


def test_gui_agent_runner_reports_malformed_tool_arguments_for_retry():
    loop = ComputerLoop(backend=FakeBackend(), screen=FakeScreen())
    runner = GuiAgentRunner(computer_loop=loop, max_steps=2)
    task = GuiAgentTaskSpec(task_id="task_bad_json", instruction="Record result.")

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        if len(conversation) == 2:
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "record_workflow_result",
                        "call_id": "result_bad",
                        "arguments": '{"item_id":"a","output_text":"unfinished',
                    }
                ]
            }
        assert conversation[-1]["type"] == "function_call_output"
        assert conversation[-1]["output"]["status"] == "rejected"
        assert "not valid JSON" in conversation[-1]["output"]["reason"]
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Retried later."}],
                }
            ]
        }

    result = runner.run_task(task, agent)

    assert result.status == "completed"
    assert result.final_text == "Retried later."
    assert result.metadata["step_records"][0]["result"] == "rejected"


def test_gui_agent_runner_requests_human_after_repeated_computer_actions():
    loop = ComputerLoop(backend=FakeBackend(), screen=FakeScreen())
    runner = GuiAgentRunner(computer_loop=loop, max_steps=5)
    task = GuiAgentTaskSpec(task_id="task_repeat", instruction="Click target.")

    def agent(_conversation: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "output": [
                {
                    "type": "computer_call",
                    "call_id": "repeat_click",
                    "actions": [{"type": "click", "x": 10, "y": 20}],
                }
            ]
        }

    result = runner.run_task(task, agent)

    waiting = result.metadata["jobs"]["job:task_repeat"]["subtasks"]["waiting_human"]
    assert result.status == "waiting_human"
    assert result.steps == 3
    assert waiting
    assert next(iter(runner.human_loop.pending.values())).risk_reason == "repeated_action"


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

    result = runner.run_task(task, agent)

    assert result.status == "max_steps_exceeded"
    assert result.metadata["step_records"][0]["result"] == "rejected"
    assert "request_human_help.question" in (
        result.metadata["step_records"][0]["metadata"]["output"]["reason"]
    )


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
    assert result.metadata["trace"]["event_count"] == 4
    assert len(result.metadata["model_response_metrics"]) == 2


def test_gui_agent_runner_handles_action_macro_tool_call():
    registry = ActionMacroRegistry()
    registry.register(
        ActionMacro(
            name="combo_asd",
            steps=(
                ActionMacroStep.keypress("A"),
                ActionMacroStep.keypress("S"),
                ActionMacroStep.keypress("D"),
            ),
        )
    )
    loop = ComputerLoop(backend=FakeBackend(), screen=FakeScreen())
    runner = GuiAgentRunner(
        computer_loop=loop,
        action_macros=registry,
        max_steps=3,
    )
    task = GuiAgentTaskSpec(task_id="task_macro", instruction="Use combo.")

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        if len(conversation) == 2:
            assert "combo_asd" in conversation[0]["content"][1]["text"]
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "run_action_macro",
                        "call_id": "macro_1",
                        "arguments": {"macro_name": "combo_asd"},
                    }
                ]
            }
        assert conversation[-1]["type"] == "function_call_output"
        assert conversation[-1]["output"]["status"] == "accepted"
        assert len(conversation[-1]["output"]["expanded_actions"]) == 3
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Macro planned."}],
                }
            ]
        }

    result = runner.run_task(task, agent)

    assert result.status == "completed"
    assert result.metadata["step_records"][0]["action"]["name"] == "run_action_macro"


def test_gui_agent_runner_handles_navigation_tool_call():
    loop = ComputerLoop(backend=FakeBackend(), screen=FakeScreen())
    runner = GuiAgentRunner(computer_loop=loop, max_steps=3)
    task = GuiAgentTaskSpec(task_id="task_nav", instruction="Move to target.")

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        if len(conversation) == 2:
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "navigate_to_target",
                        "call_id": "nav_1",
                        "arguments": {
                            "source": [0, 0],
                            "target": [100, 100],
                            "obstacles": [[[40, 40], [60, 40], [60, 60], [40, 60]]],
                            "target_label": "monster",
                        },
                    }
                ]
            }
        assert conversation[-1]["type"] == "function_call_output"
        assert conversation[-1]["output"]["status"] == "planned"
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Route planned."}],
                }
            ]
        }

    result = runner.run_task(task, agent)

    plan = result.metadata["step_records"][0]["metadata"]["output"]["navigation_plan"]
    assert result.status == "completed"
    assert plan["target_label"] == "monster"
    assert len(plan["waypoints"]) >= 2


def test_gui_agent_runner_accepts_app_viewport_crop_tool_call():
    screen = FakeScaledScreen(
        logical_size=(1000, 500),
        physical_size=(2000, 1000),
        physical_origin=(0, 0),
    )
    loop = ComputerLoop(backend=FakeBackend(), screen=screen)
    runner = GuiAgentRunner(computer_loop=loop, max_steps=3)
    task = GuiAgentTaskSpec(task_id="task_viewport", instruction="Crop the APP.")

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        if len(conversation) == 2:
            assert "set_app_viewport" in conversation[0]["content"][0]["text"]
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "set_app_viewport",
                        "call_id": "viewport_1",
                        "arguments": {
                            "x": "100",
                            "y": "50",
                            "width": "400",
                            "height": "200",
                            "coordinate_space": "screenshot",
                            "reason": "Only the browser area is useful.",
                        },
                    }
                ]
            }
        assert conversation[-2]["type"] == "function_call_output"
        assert conversation[-2]["output"]["status"] == "accepted"
        assert conversation[-1]["content"][1]["type"] == "input_image"
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Viewport set."}],
                }
            ]
        }

    result = runner.run_task(task, agent)

    assert result.status == "completed"
    assert screen.crop_box == (200, 100, 800, 400)
    viewport_record = result.metadata["step_records"][0]["metadata"]["output"]
    assert viewport_record["tool"] == "set_app_viewport"
    assert viewport_record["screen_viewport"]["crop_width"] == 800


def test_gui_agent_runner_keeps_stable_history_prefix_between_steps():
    backend = FakeBackend()
    loop = ComputerLoop(backend=backend, screen=FakeScreen())
    runner = GuiAgentRunner(computer_loop=loop, max_steps=4)
    task = GuiAgentTaskSpec(task_id="task_history", instruction="Click twice.")
    developer_texts: list[str] = []
    conversation_lengths: list[int] = []
    screenshot_output_counts: list[int] = []

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        developer_texts.append(conversation[0]["content"][0]["text"])
        conversation_lengths.append(len(conversation))
        screenshot_output_counts.append(
            sum(1 for item in conversation if item.get("type") == "computer_call_output")
        )
        if len(conversation) == 2:
            return {
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "click_1",
                        "actions": [{"type": "click", "x": 10, "y": 20}],
                    }
                ]
            }
        if len(conversation) == 5:
            assert conversation[1]["content"][1]["type"] == "input_image"
            assert conversation[2]["type"] == "computer_call"
            assert conversation[3]["type"] == "computer_call_output"
            assert "坐标提醒" in conversation[4]["content"][0]["text"]
            return {
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "click_2",
                        "actions": [{"type": "click", "x": 30, "y": 40}],
                    }
                ]
            }
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
    assert conversation_lengths == [2, 5, 8]
    assert screenshot_output_counts == [0, 1, 2]
    assert developer_texts[0] == developer_texts[1] == developer_texts[2]
    assert result.metadata["prompt_context"]["selected_frame_count"] >= 3


def test_gui_agent_runner_omits_inline_media_from_runtime_text_context():
    loop = ComputerLoop(backend=FakeBackend(), screen=FakeScreen())
    runner = GuiAgentRunner(computer_loop=loop, max_steps=1)
    captured: list[list[dict[str, Any]]] = []
    task = GuiAgentTaskSpec(task_id="task_inline_context", instruction="Inspect.")

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        captured.append(conversation)
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Done."}],
                }
            ]
        }

    runner.run_task(task, agent)

    runtime_text = "\n".join(
        str(part.get("text", ""))
        for part in captured[0][0]["content"]
        if isinstance(part, dict) and part.get("type") == "input_text"
    )
    image_part = captured[0][1]["content"][1]
    assert "data:image" not in runtime_text
    assert "图片已作为 input_image 提供" in runtime_text
    assert image_part["type"] == "input_image"
    assert image_part["image_url"].startswith("data:image")


def test_gui_agent_runner_clicks_inside_cropped_app_viewport():
    backend = FakeBackend()
    screen = FakeScaledScreen(
        logical_size=(1000, 500),
        physical_size=(2000, 1000),
        physical_origin=(0, 0),
    )
    loop = ComputerLoop(backend=backend, screen=screen)
    runner = GuiAgentRunner(computer_loop=loop, max_steps=4)
    task = GuiAgentTaskSpec(
        task_id="task_cropped_click",
        instruction="Crop to APP and click its center.",
    )

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        if len(conversation) == 2:
            return {
                "output": [
                    {
                        "type": "function_call",
                        "name": "set_app_viewport",
                        "call_id": "viewport_1",
                        "arguments": {
                            "x": 100,
                            "y": 50,
                            "width": 400,
                            "height": 200,
                            "coordinate_space": "screenshot",
                        },
                    }
                ]
            }
        if conversation[-2]["type"] == "function_call_output":
            assert conversation[-1]["content"][1]["type"] == "input_image"
            assert screen.size() == (800, 400)
            return {
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "cropped_click",
                        "actions": [{"type": "click", "x": 400, "y": 200}],
                    }
                ]
            }
        assert conversation[-2]["type"] == "computer_call_output"
        assert "坐标提醒" in conversation[-1]["content"][0]["text"]
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Clicked."}],
                }
            ]
        }

    result = runner.run_task(task, agent)

    assert result.status == "completed"
    assert screen.crop_box == (200, 100, 800, 400)
    assert ("click", (600, 300, "left")) in backend.calls


def test_gui_agent_runner_reports_rejected_computer_action_without_crashing():
    backend = FakeBackend()
    loop = ComputerLoop(backend=backend, screen=FakeScreen())
    runner = GuiAgentRunner(computer_loop=loop, max_steps=3)
    task = GuiAgentTaskSpec(task_id="task_bad_click", instruction="Click safely.")
    seen_rejection = False

    def agent(conversation: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal seen_rejection
        if len(conversation) == 2:
            return {
                "output": [
                    {
                        "type": "computer_call",
                        "call_id": "bad_click",
                        "actions": [{"type": "click", "x": 10, "y": 999}],
                    }
                ]
            }
        seen_rejection = (
            "computer_use was rejected" in conversation[-1]["content"][0]["text"]
        )
        assert conversation[-1]["content"][1]["type"] == "input_image"
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Recovered."}],
                }
            ]
        }

    result = runner.run_task(task, agent)

    assert result.status == "completed"
    assert seen_rejection
    assert result.metadata["step_records"][0]["result"] == "rejected"
    assert backend.calls == []
