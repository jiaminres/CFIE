from __future__ import annotations

import base64
import ctypes
import sys
from dataclasses import dataclass
from io import BytesIO
from typing import Protocol

from PIL import ImageGrab


@dataclass(slots=True, frozen=True)
class ScreenshotResult:
    image_url: str
    width: int
    height: int


class ScreenCapture(Protocol):
    def screenshot(self) -> ScreenshotResult:
        ...

    def size(self) -> tuple[int, int]:
        ...


class PillowScreenCapture:
    def screenshot(self) -> ScreenshotResult:
        image = ImageGrab.grab()
        width, height = image.size
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return ScreenshotResult(
            image_url=f"data:image/png;base64,{encoded}",
            width=width,
            height=height,
        )

    def size(self) -> tuple[int, int]:
        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            return (
                int(user32.GetSystemMetrics(0)),
                int(user32.GetSystemMetrics(1)),
            )
        image = ImageGrab.grab()
        return image.size
