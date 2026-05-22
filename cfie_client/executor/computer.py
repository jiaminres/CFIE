from __future__ import annotations

import sys
import time
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


def create_default_backend() -> ComputerBackend:
    if sys.platform == "win32":
        from cfie_client.executor.windows import WindowsComputerBackend

        return WindowsComputerBackend()
    return UnsupportedComputerBackend()


class ComputerExecutor:
    def __init__(self, backend: ComputerBackend | None = None) -> None:
        self.backend = backend if backend is not None else create_default_backend()

    def execute_all(self, actions: tuple[ComputerAction, ...]) -> None:
        for action in actions:
            self.execute(action)

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

        if action.type == "wait":
            self.backend.wait(2.0)
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
