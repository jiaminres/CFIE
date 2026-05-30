from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import Protocol

from cfie_client.protocol import ComputerAction


class ComputerBackend(Protocol):
    def move(self, x: int, y: int) -> None:
        ...

    def click(self, x: int, y: int, button: str = "left") -> None:
        ...

    def double_click(self, x: int, y: int, button: str = "left") -> None:
        ...

    def drag(self, path: tuple[tuple[int, int], ...]) -> None:
        ...

    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        ...

    def type_text(self, text: str) -> None:
        ...

    def press_keys(self, keys: tuple[str, ...]) -> None:
        ...

    def key_down(self, key: str) -> None:
        ...

    def key_up(self, key: str) -> None:
        ...

    def wait(self, seconds: float) -> None:
        ...


class UnsupportedComputerBackend:
    def _raise(self) -> None:
        raise RuntimeError(
            "No local computer backend is available for this platform. "
            "Pass a custom ComputerBackend to ComputerExecutor."
        )

    def move(self, x: int, y: int) -> None:
        self._raise()

    def click(self, x: int, y: int, button: str = "left") -> None:
        self._raise()

    def double_click(self, x: int, y: int, button: str = "left") -> None:
        self._raise()

    def drag(self, path: tuple[tuple[int, int], ...]) -> None:
        self._raise()

    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self._raise()

    def type_text(self, text: str) -> None:
        self._raise()

    def press_keys(self, keys: tuple[str, ...]) -> None:
        self._raise()

    def key_down(self, key: str) -> None:
        self._raise()

    def key_up(self, key: str) -> None:
        self._raise()

    def wait(self, seconds: float) -> None:
        time.sleep(seconds)


class CoordinateScalingBackend:
    def __init__(
        self,
        backend: ComputerBackend,
        *,
        logical_size: tuple[int, int] | Callable[[], tuple[int, int]],
        physical_size: tuple[int, int] | Callable[[], tuple[int, int]],
        physical_offset: tuple[int, int] | Callable[[], tuple[int, int]] = (0, 0),
    ) -> None:
        self.backend = backend
        self.logical_size = logical_size
        self.physical_size = physical_size
        self.physical_offset = physical_offset

    def move(self, x: int, y: int) -> None:
        self.backend.move(*self._scale_xy(x, y))

    def click(self, x: int, y: int, button: str = "left") -> None:
        self.backend.click(*self._scale_xy(x, y), button)

    def double_click(self, x: int, y: int, button: str = "left") -> None:
        self.backend.double_click(*self._scale_xy(x, y), button)

    def drag(self, path: tuple[tuple[int, int], ...]) -> None:
        self.backend.drag(tuple(self._scale_xy(x, y) for x, y in path))

    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self.backend.scroll(*self._scale_xy(x, y), scroll_x, scroll_y)

    def type_text(self, text: str) -> None:
        self.backend.type_text(text)

    def press_keys(self, keys: tuple[str, ...]) -> None:
        self.backend.press_keys(keys)

    def key_down(self, key: str) -> None:
        self.backend.key_down(key)

    def key_up(self, key: str) -> None:
        self.backend.key_up(key)

    def wait(self, seconds: float) -> None:
        self.backend.wait(seconds)

    def _scale_xy(self, x: int, y: int) -> tuple[int, int]:
        logical_width, logical_height = self._resolve_size(self.logical_size)
        physical_width, physical_height = self._resolve_size(self.physical_size)
        offset_x, offset_y = self._resolve_size(self.physical_offset)
        if logical_width <= 0 or logical_height <= 0:
            return (int(x) + offset_x, int(y) + offset_y)
        return (
            int(round(int(x) * physical_width / logical_width)) + offset_x,
            int(round(int(y) * physical_height / logical_height)) + offset_y,
        )

    @staticmethod
    def _resolve_size(
        value: tuple[int, int] | Callable[[], tuple[int, int]],
    ) -> tuple[int, int]:
        return value() if callable(value) else value


def create_default_backend() -> ComputerBackend:
    if sys.platform == "win32":
        from cfie_client.executor.windows import WindowsComputerBackend

        return WindowsComputerBackend()
    return UnsupportedComputerBackend()


class ComputerExecutor:
    def __init__(
        self,
        backend: ComputerBackend | None = None,
        *,
        action_delay_seconds: float = 0.05,
    ) -> None:
        self.backend = backend if backend is not None else create_default_backend()
        self.action_delay_seconds = max(0.0, action_delay_seconds)

    def execute_all(self, actions: tuple[ComputerAction, ...]) -> None:
        for action in actions:
            self.execute(action)
            if self.action_delay_seconds:
                time.sleep(self.action_delay_seconds)

    def execute(self, action: ComputerAction) -> None:
        if action.type == "click":
            self._with_modifiers(
                action.keys,
                lambda: self.backend.click(
                    int(action.x),
                    int(action.y),
                    action.button or "left",
                ),
            )
            return

        if action.type == "double_click":
            self._with_modifiers(
                action.keys,
                lambda: self.backend.double_click(
                    int(action.x),
                    int(action.y),
                    action.button or "left",
                ),
            )
            return

        if action.type == "drag":
            self._with_modifiers(action.keys, lambda: self.backend.drag(action.path))
            return

        if action.type == "move":
            self._with_modifiers(
                action.keys,
                lambda: self.backend.move(int(action.x), int(action.y)),
            )
            return

        if action.type == "scroll":
            self._with_modifiers(
                action.keys,
                lambda: self.backend.scroll(
                    int(action.x),
                    int(action.y),
                    action.scroll_x,
                    action.scroll_y,
                ),
            )
            return

        if action.type == "keypress":
            self.backend.press_keys(action.keys)
            return

        if action.type == "type":
            self.backend.type_text(action.text or "")
            return

        if action.type == "submit_text":
            if action.x is not None and action.y is not None:
                self._with_modifiers(
                    action.keys,
                    lambda: self.backend.click(
                        int(action.x),
                        int(action.y),
                        action.button or "left",
                    ),
                )
                self._sleep_between_subactions()
            self.backend.press_keys(("ctrl", "a"))
            self._sleep_between_subactions()
            self.backend.type_text(action.text or "")
            self._sleep_between_subactions()
            self.backend.press_keys(("enter",))
            return

        if action.type == "wait":
            self.backend.wait(action.duration if action.duration is not None else 1.0)
            return

        if action.type == "screenshot":
            return

        raise ValueError(f"Unsupported computer action: {action.type}")

    def _with_modifiers(self, keys: tuple[str, ...], callback) -> None:
        pressed: list[str] = []
        try:
            for key in keys:
                self.backend.key_down(key)
                pressed.append(key)
            callback()
        finally:
            for key in reversed(pressed):
                self.backend.key_up(key)

    def _sleep_between_subactions(self) -> None:
        if self.action_delay_seconds:
            time.sleep(self.action_delay_seconds)
