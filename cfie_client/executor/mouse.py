from __future__ import annotations

import ctypes
import time

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_XDOWN = 0x0080
MOUSEEVENTF_XUP = 0x0100
XBUTTON1 = 0x0001
XBUTTON2 = 0x0002

_BUTTON_FLAGS = {
    "back": (MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, XBUTTON1),
    "forward": (MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, XBUTTON2),
    "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, 0),
    "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP, 0),
    "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, 0),
    "wheel": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP, 0),
}


def move_to(x: int, y: int) -> None:
    ctypes.windll.user32.SetCursorPos(int(x), int(y))


def click_at(x: int, y: int, button: str = "left") -> None:
    move_to(x, y)
    down, up, data = _BUTTON_FLAGS.get(button, _BUTTON_FLAGS["left"])
    ctypes.windll.user32.mouse_event(down, 0, 0, data, 0)
    ctypes.windll.user32.mouse_event(up, 0, 0, data, 0)


def double_click_at(x: int, y: int, button: str = "left") -> None:
    click_at(x, y, button)
    click_at(x, y, button)


def drag_path(path: tuple[tuple[int, int], ...]) -> None:
    if len(path) < 2:
        raise ValueError("drag path requires at least two points")
    start_x, start_y = path[0]
    move_to(start_x, start_y)
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    try:
        for x, y in path[1:]:
            move_to(x, y)
            time.sleep(0.01)
    finally:
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
