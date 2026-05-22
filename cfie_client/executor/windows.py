from __future__ import annotations

from cfie_client.executor import keyboard, mouse, scroll, wait


class WindowsComputerBackend:
    def move(self, x: int, y: int) -> None:
        mouse.move_to(x, y)

    def click(self, x: int, y: int, button: str = "left") -> None:
        mouse.click_at(x, y, button)

    def double_click(self, x: int, y: int, button: str = "left") -> None:
        mouse.double_click_at(x, y, button)

    def drag(self, path: tuple[tuple[int, int], ...]) -> None:
        mouse.drag_path(path)

    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        scroll.scroll_at(x, y, scroll_x, scroll_y)

    def type_text(self, text: str) -> None:
        keyboard.type_text(text)

    def press_keys(self, keys: tuple[str, ...]) -> None:
        keyboard.press_keys(keys)

    def key_down(self, key: str) -> None:
        keyboard.key_down(key)

    def key_up(self, key: str) -> None:
        keyboard.key_up(key)

    def wait(self, seconds: float) -> None:
        wait.wait_seconds(seconds)
