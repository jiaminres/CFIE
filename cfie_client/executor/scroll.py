from __future__ import annotations

import ctypes

from cfie_client.executor.mouse import move_to

MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x01000
WHEEL_DELTA = 120


def scroll_at(x: int, y: int, scroll_x: int, scroll_y: int) -> None:
    move_to(x, y)
    if scroll_y:
        ctypes.windll.user32.mouse_event(
            MOUSEEVENTF_WHEEL,
            0,
            0,
            int(-scroll_y / 100) * WHEEL_DELTA or (-WHEEL_DELTA if scroll_y > 0 else WHEEL_DELTA),
            0,
        )
    if scroll_x:
        ctypes.windll.user32.mouse_event(
            MOUSEEVENTF_HWHEEL,
            0,
            0,
            int(scroll_x / 100) * WHEEL_DELTA or (WHEEL_DELTA if scroll_x > 0 else -WHEEL_DELTA),
            0,
        )
