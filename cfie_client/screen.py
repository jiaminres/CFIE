from __future__ import annotations

import base64
import ctypes
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from PIL import ImageDraw, ImageGrab


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


class ScaledPillowScreenCapture:
    def __init__(
        self,
        *,
        max_width: int = 1280,
        max_height: int = 720,
        image_format: str = "JPEG",
        jpeg_quality: int = 85,
        url_mode: str = "data",
        output_dir: str | Path | None = None,
        filename_prefix: str = "screen",
        crop_box: tuple[int, int, int, int] | None = None,
        grid_overlay: str = "off",
    ) -> None:
        self.max_width = max(1, int(max_width))
        self.max_height = max(1, int(max_height))
        self.image_format = image_format.upper()
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self.url_mode = url_mode
        if self.url_mode not in {"data", "file"}:
            raise ValueError("url_mode must be 'data' or 'file'")
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.filename_prefix = filename_prefix
        self.crop_box = crop_box
        self.grid_overlay = grid_overlay
        if self.grid_overlay not in {"off", "coarse", "fine"}:
            raise ValueError("grid_overlay must be 'off', 'coarse', or 'fine'")

    def set_crop_box(self, crop_box: tuple[int, int, int, int] | None) -> None:
        self.crop_box = _normalize_crop_box(crop_box)

    def clear_crop_box(self) -> None:
        self.crop_box = None

    def viewport_context(self) -> dict[str, int | None]:
        origin_x, origin_y = self.physical_origin()
        physical_width, physical_height = self.physical_size()
        logical_width, logical_height = self.size()
        return {
            "crop_x": self.crop_box[0] if self.crop_box is not None else None,
            "crop_y": self.crop_box[1] if self.crop_box is not None else None,
            "crop_width": self.crop_box[2] if self.crop_box is not None else None,
            "crop_height": self.crop_box[3] if self.crop_box is not None else None,
            "physical_origin_x": origin_x,
            "physical_origin_y": origin_y,
            "physical_width": physical_width,
            "physical_height": physical_height,
            "screenshot_width": logical_width,
            "screenshot_height": logical_height,
            "grid_overlay": self.grid_overlay,
        }

    def screenshot(self) -> ScreenshotResult:
        image = ImageGrab.grab(bbox=self._grab_bbox())
        physical_width, physical_height = image.size
        width, height = self._scaled_size(physical_width, physical_height)
        if (width, height) != (physical_width, physical_height):
            image = image.resize((width, height))
        if self.grid_overlay != "off":
            image = _draw_grid_overlay(image, mode=self.grid_overlay)
        if self.image_format == "JPEG":
            image = image.convert("RGB")
            suffix = "jpg"
            mime = "image/jpeg"
        else:
            suffix = "png"
            mime = "image/png"
        if self.url_mode == "file":
            if self.output_dir is None:
                raise ValueError("output_dir is required when url_mode='file'")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"{self.filename_prefix}_{uuid4().hex}.{suffix}"
            if self.image_format == "JPEG":
                image.save(path, format="JPEG", quality=self.jpeg_quality, optimize=True)
            else:
                image.save(path, format="PNG", optimize=True)
            return ScreenshotResult(
                image_url=path.resolve().as_uri(),
                width=width,
                height=height,
            )
        buffer = BytesIO()
        if self.image_format == "JPEG":
            image.save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
        else:
            image.save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return ScreenshotResult(
            image_url=f"data:{mime};base64,{encoded}",
            width=width,
            height=height,
        )

    def size(self) -> tuple[int, int]:
        return self._scaled_size(*self.physical_size())

    def physical_size(self) -> tuple[int, int]:
        if self.crop_box is not None:
            return (self.crop_box[2], self.crop_box[3])
        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            return (
                int(user32.GetSystemMetrics(0)),
                int(user32.GetSystemMetrics(1)),
            )
        image = ImageGrab.grab()
        return image.size

    def physical_origin(self) -> tuple[int, int]:
        if self.crop_box is None:
            return (0, 0)
        return (self.crop_box[0], self.crop_box[1])

    def _scaled_size(self, width: int, height: int) -> tuple[int, int]:
        scale = min(self.max_width / width, self.max_height / height, 1.0)
        return (max(1, int(width * scale)), max(1, int(height * scale)))

    def _grab_bbox(self) -> tuple[int, int, int, int] | None:
        if self.crop_box is None:
            return None
        x, y, width, height = self.crop_box
        return (x, y, x + width, y + height)


def _normalize_crop_box(
    crop_box: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if crop_box is None:
        return None
    x, y, width, height = (int(value) for value in crop_box)
    if width <= 0 or height <= 0:
        raise ValueError("crop_box width and height must be positive")
    return (max(0, x), max(0, y), width, height)


def _draw_grid_overlay(image, *, mode: str):
    if mode == "off":
        return image
    result = image.convert("RGB")
    draw = ImageDraw.Draw(result)
    width, height = result.size
    minor_step = 50 if mode == "coarse" else 25
    major_step = 100 if mode == "coarse" else 50
    minor_color = "#f6f6f6"
    major_color = "#e8e8e8"
    label_color = "#9a9a9a"
    for x in range(0, width, minor_step):
        color = major_color if x % major_step == 0 else minor_color
        line_width = 1
        draw.line((x, 0, x, height), fill=color, width=line_width)
    for y in range(0, height, minor_step):
        color = major_color if y % major_step == 0 else minor_color
        line_width = 1
        draw.line((0, y, width, y), fill=color, width=line_width)
    for x in range(0, width, major_step):
        if x:
            draw.text((x + 3, 3), str(x), fill=label_color)
            draw.text((x + 3, max(3, height - 15)), str(x), fill=label_color)
    for y in range(0, height, major_step):
        if y:
            draw.text((3, y + 3), str(y), fill=label_color)
            draw.text((max(3, width - 36), y + 3), str(y), fill=label_color)
    return result
