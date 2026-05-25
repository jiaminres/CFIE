from __future__ import annotations

import ctypes
import re
import sys
from dataclasses import dataclass
from ctypes import wintypes

from cfie_client.executor import keyboard, mouse, scroll, wait

DWMWA_CLOAKED = 14
VK_MENU = 0x12
KEYEVENTF_KEYUP = 0x0002


@dataclass(slots=True, frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    rect: tuple[int, int, int, int]

    @property
    def crop_box(self) -> tuple[int, int, int, int]:
        left, top, right, bottom = self.rect
        return (left, top, max(1, right - left), max(1, bottom - top))


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


def list_visible_windows() -> tuple[WindowInfo, ...]:
    if sys.platform != "win32":
        return ()
    user32 = ctypes.windll.user32
    windows: list[WindowInfo] = []

    enum_proc_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def enum_proc(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        if user32.IsIconic(hwnd):
            return True
        if _is_dwm_cloaked(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value.strip()
        if not title:
            return True
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        windows.append(
            WindowInfo(
                hwnd=int(hwnd),
                title=title,
                rect=(int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)),
            )
        )
        return True

    user32.EnumWindows(enum_proc_type(enum_proc), 0)
    return tuple(windows)


def find_visible_window(title_pattern: str) -> WindowInfo | None:
    try:
        pattern = re.compile(title_pattern, re.IGNORECASE)
    except re.error:
        return None
    for window in list_visible_windows():
        if pattern.search(window.title):
            return window
    return None


def focus_window(hwnd: int, *, maximize: bool = False) -> bool:
    if sys.platform != "win32":
        return False
    user32 = ctypes.windll.user32
    user32.ShowWindow(hwnd, 3 if maximize else 9)
    if user32.SetForegroundWindow(hwnd):
        return True
    user32.keybd_event(VK_MENU, 0, 0, 0)
    try:
        return bool(user32.SetForegroundWindow(hwnd))
    finally:
        user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)


def focus_window_by_title(
    title_pattern: str,
    *,
    maximize: bool = False,
) -> WindowInfo | None:
    window = find_visible_window(title_pattern)
    if window is None:
        return None
    focus_window(window.hwnd, maximize=maximize)
    refreshed = find_visible_window(re.escape(window.title))
    return refreshed or window


def _is_dwm_cloaked(hwnd: int) -> bool:
    try:
        dwmapi = ctypes.windll.dwmapi
    except AttributeError:
        return False
    cloaked = wintypes.DWORD()
    result = dwmapi.DwmGetWindowAttribute(
        hwnd,
        DWMWA_CLOAKED,
        ctypes.byref(cloaked),
        ctypes.sizeof(cloaked),
    )
    return result == 0 and bool(cloaked.value)
