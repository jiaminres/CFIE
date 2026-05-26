from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cfie_client.executor import ComputerBackend, ComputerExecutor
from cfie_client.protocol import (
    ComputerAction,
    ComputerCall,
    ComputerCallOutput,
    COORDINATE_SPACES,
    build_computer_call_output,
    find_computer_calls,
)
from cfie_client.safety import SafetyGate
from cfie_client.screen import PillowScreenCapture, ScreenCapture
from cfie_client.trace import TraceStore


class ComputerLoop:
    def __init__(
        self,
        *,
        executor: ComputerExecutor | None = None,
        backend: ComputerBackend | None = None,
        screen: ScreenCapture | None = None,
        safety: SafetyGate | None = None,
        trace_path: str | Path | None = None,
        trace_artifact_dir: str | Path | None = None,
        screenshot_detail: str | None = "low",
        map_screen_coordinates: bool = True,
        model_coordinate_mode: str = "screenshot",
    ) -> None:
        if executor is not None and backend is not None:
            raise ValueError("Pass either executor or backend, not both")
        self.executor = executor if executor is not None else ComputerExecutor(backend)
        self.screen = screen if screen is not None else PillowScreenCapture()
        self.safety = safety if safety is not None else SafetyGate()
        self.screenshot_detail = screenshot_detail
        self.map_screen_coordinates = map_screen_coordinates
        if model_coordinate_mode not in COORDINATE_SPACES:
            raise ValueError(
                "model_coordinate_mode must be screenshot, "
                "qwen_normalized_1000, local_refinement_1000, or auto"
            )
        self.model_coordinate_mode = model_coordinate_mode
        self.local_refinement_box: tuple[int, int, int, int] | None = None
        self.trace = TraceStore(
            Path(trace_path) if trace_path is not None else None,
            Path(trace_artifact_dir) if trace_artifact_dir is not None else None,
        )

    def set_local_refinement_box(
        self,
        box: tuple[int, int, int, int] | None,
    ) -> None:
        self.local_refinement_box = box

    def handle_call(self, call_like) -> ComputerCallOutput:
        call = (
            call_like
            if isinstance(call_like, ComputerCall)
            else ComputerCall.from_openai(call_like)
        )
        screen_size = self.screen.size()
        call = _normalize_model_coordinate_space(
            call,
            screen_size=screen_size,
            mode=self.model_coordinate_mode,
            local_refinement_box=self.local_refinement_box,
        )
        call = _default_scroll_coordinates(call, screen_size=screen_size)
        self.safety.check_call(call, screen_size=screen_size)
        self.trace.record_tool_call(call.to_openai_dict())
        executor_call = (
            _map_call_to_physical_screen(call, self.screen)
            if self.map_screen_coordinates
            else call
        )
        self.executor.execute_all(executor_call.actions)
        screenshot = self.screen.screenshot()
        output = build_computer_call_output(
            call_id=call.call_id,
            image_url=screenshot.image_url,
            detail=self.screenshot_detail,
            acknowledged_safety_checks=call.pending_safety_checks,
        )
        self.trace.record_tool_result(output.to_openai_dict())
        if call.coordinate_space == "screenshot":
            self.local_refinement_box = None
        return output

    def handle_response(self, response_or_items) -> tuple[ComputerCallOutput, ...]:
        return tuple(
            self.handle_call(call)
            for call in find_computer_calls(response_or_items)
        )


def _normalize_model_coordinate_space(
    call: ComputerCall,
    *,
    screen_size: tuple[int, int],
    mode: str,
    local_refinement_box: tuple[int, int, int, int] | None = None,
) -> ComputerCall:
    mode = call.coordinate_space or mode
    if mode == "local_refinement_1000":
        if local_refinement_box is None:
            return call
        return replace(
            call,
            coordinate_space="screenshot",
            actions=tuple(
                _map_action_from_local_refinement_1000(
                    action,
                    screen_size=screen_size,
                    local_box=local_refinement_box,
                )
                for action in call.actions
            ),
        )
    if (
        call.coordinate_space == "screenshot"
        and _looks_like_qwen_normalized_call(call, screen_size)
    ):
        mode = "qwen_normalized_1000"
    if mode == "screenshot":
        return call
    if mode == "auto" and not _looks_like_qwen_normalized_call(call, screen_size):
        return call
    return replace(
        call,
        coordinate_space="screenshot",
        actions=tuple(
            _map_action_from_qwen_normalized_1000(action, screen_size=screen_size)
            for action in call.actions
        ),
    )


def _default_scroll_coordinates(
    call: ComputerCall,
    *,
    screen_size: tuple[int, int],
) -> ComputerCall:
    width, height = screen_size
    if width <= 0 or height <= 0:
        return call
    center_x = max(0, width // 2)
    center_y = max(0, height // 2)
    changed = False
    actions: list[ComputerAction] = []
    for action in call.actions:
        if action.type == "scroll" and (action.x is None or action.y is None):
            changed = True
            actions.append(
                replace(
                    action,
                    x=center_x if action.x is None else action.x,
                    y=center_y if action.y is None else action.y,
                )
            )
        else:
            actions.append(action)
    if not changed:
        return call
    return replace(call, actions=tuple(actions))


def _looks_like_qwen_normalized_call(
    call: ComputerCall,
    screen_size: tuple[int, int],
) -> bool:
    width, height = screen_size
    if width <= 0 or height <= 0:
        return False
    for action in call.actions:
        for x, y in action.coordinate_points:
            if 0 <= x <= 1000 and 0 <= y <= 1000 and (x >= width or y >= height):
                return True
    return False


def _map_action_from_qwen_normalized_1000(
    action: ComputerAction,
    *,
    screen_size: tuple[int, int],
) -> ComputerAction:
    width, height = screen_size

    def map_xy(x: int, y: int) -> tuple[int, int]:
        return (
            max(0, min(width - 1, int(round(x * width / 1000)))),
            max(0, min(height - 1, int(round(y * height / 1000)))),
        )

    if action.path:
        return replace(action, path=tuple(map_xy(x, y) for x, y in action.path))
    if action.x is None or action.y is None:
        return action
    if action.type not in {"click", "double_click", "move", "scroll"}:
        return action
    x, y = map_xy(action.x, action.y)
    return replace(action, x=x, y=y)


def _map_action_from_local_refinement_1000(
    action: ComputerAction,
    *,
    screen_size: tuple[int, int],
    local_box: tuple[int, int, int, int],
) -> ComputerAction:
    screen_width, screen_height = screen_size
    box_x, box_y, box_width, box_height = local_box

    def map_xy(x: int, y: int) -> tuple[int, int]:
        return (
            max(
                0,
                min(
                    screen_width - 1,
                    box_x + int(round(int(x) * box_width / 1000)),
                ),
            ),
            max(
                0,
                min(
                    screen_height - 1,
                    box_y + int(round(int(y) * box_height / 1000)),
                ),
            ),
        )

    if action.path:
        return replace(action, path=tuple(map_xy(x, y) for x, y in action.path))
    if action.x is None or action.y is None:
        return action
    if action.type not in {"click", "double_click", "move", "scroll"}:
        return action
    x, y = map_xy(action.x, action.y)
    return replace(action, x=x, y=y)


def _map_call_to_physical_screen(
    call: ComputerCall,
    screen: ScreenCapture,
) -> ComputerCall:
    physical_size_fn = getattr(screen, "physical_size", None)
    physical_origin_fn = getattr(screen, "physical_origin", None)
    if not callable(physical_size_fn):
        return call
    logical_width, logical_height = screen.size()
    physical_width, physical_height = physical_size_fn()
    origin_x, origin_y = (
        physical_origin_fn()
        if callable(physical_origin_fn)
        else (0, 0)
    )
    if logical_width <= 0 or logical_height <= 0:
        return call
    if (
        logical_width == physical_width
        and logical_height == physical_height
        and origin_x == 0
        and origin_y == 0
    ):
        return call
    return replace(
        call,
        actions=tuple(
            _map_action_to_physical_screen(
                action,
                logical_size=(logical_width, logical_height),
                physical_size=(physical_width, physical_height),
                physical_origin=(origin_x, origin_y),
            )
            for action in call.actions
        ),
    )


def _map_action_to_physical_screen(
    action: ComputerAction,
    *,
    logical_size: tuple[int, int],
    physical_size: tuple[int, int],
    physical_origin: tuple[int, int],
) -> ComputerAction:
    def map_xy(x: int, y: int) -> tuple[int, int]:
        logical_width, logical_height = logical_size
        physical_width, physical_height = physical_size
        origin_x, origin_y = physical_origin
        return (
            int(round(x * physical_width / logical_width)) + origin_x,
            int(round(y * physical_height / logical_height)) + origin_y,
        )

    if action.path:
        return replace(
            action,
            path=tuple(map_xy(x, y) for x, y in action.path),
        )
    if action.x is None or action.y is None:
        return action
    if action.type not in {"click", "double_click", "move", "scroll"}:
        return action
    x, y = map_xy(action.x, action.y)
    return replace(action, x=x, y=y)
