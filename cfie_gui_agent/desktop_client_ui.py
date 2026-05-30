from __future__ import annotations

import base64
import ast
import json
import re
import threading
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import replace
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any
from urllib.parse import unquote, urlparse

from cfie_client import ComputerLoop
from cfie_client.executor.windows import find_visible_window
from cfie_client.screen import ScaledPillowScreenCapture

try:
    from PIL import Image, ImageDraw, ImageTk
except Exception:  # pragma: no cover - optional UI polish dependency
    Image = None
    ImageDraw = None
    ImageTk = None

from cfie_gui_agent.desktop_client import (
    DIRECT_COMMAND_LABELS,
    DIRECT_COMMAND_NONE,
    DEFAULT_TRACE_DIR,
    DEFAULT_RESPONSES_BASE_URL,
    DEFAULT_RESPONSES_MODEL,
    DesktopClientState,
    MacroConfig,
    ReferenceAsset,
    TargetAppConfig,
)
from cfie_gui_agent.state_store import save_desktop_state
from cfie_gui_agent.context import ContextManager, VisionContextPolicy
from cfie_gui_agent.jobs import JobState
from cfie_gui_agent.openai_responses import OpenAIResponsesAgent
from cfie_gui_agent.runner import DEFAULT_AGENT_MAX_STEPS, GuiAgentRunner
from cfie_gui_agent.specs import GuiAgentTaskSpec
from cfie_gui_agent.tools import ModelToolRegistry, model_tool_names_for_profile


def _agent_reasoning_effort(metadata: dict[str, Any]) -> str | None:
    mode = str(metadata.get("reasoning_mode") or "guided").strip().lower()
    effort = str(metadata.get("reasoning_effort") or "none").strip().lower()
    if mode == "default":
        return None
    if mode == "off":
        return "none"
    return effort or "none"


def _agent_chat_template_kwargs(metadata: dict[str, Any]) -> dict[str, Any]:
    mode = str(metadata.get("reasoning_mode") or "guided").strip().lower()
    effort = str(metadata.get("reasoning_effort") or "none").strip().lower()
    if mode == "default":
        return {"enable_thinking": True}
    if mode == "off":
        return {"enable_thinking": False}
    return {"enable_thinking": effort != "none"}


def draw_rounded_rect(
    canvas: tk.Canvas,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    radius: int,
    *,
    fill: str,
    outline: str = "",
    width: int = 1,
    tags: str | tuple[str, ...] = (),
) -> None:
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if radius == 0:
        canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            fill=fill,
            outline="",
            tags=tags,
        )
        if outline:
            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill="",
                outline=outline,
                width=width,
                tags=tags,
            )
        return
    canvas.create_rectangle(
        x1 + radius,
        y1,
        x2 - radius,
        y2,
        fill=fill,
        outline="",
        tags=tags,
    )
    canvas.create_rectangle(
        x1,
        y1 + radius,
        x2,
        y2 - radius,
        fill=fill,
        outline="",
        tags=tags,
    )
    canvas.create_arc(
        x1,
        y1,
        x1 + radius * 2,
        y1 + radius * 2,
        start=90,
        extent=90,
        fill=fill,
        outline="",
        tags=tags,
    )
    canvas.create_arc(
        x2 - radius * 2,
        y1,
        x2,
        y1 + radius * 2,
        start=0,
        extent=90,
        fill=fill,
        outline="",
        tags=tags,
    )
    canvas.create_arc(
        x2 - radius * 2,
        y2 - radius * 2,
        x2,
        y2,
        start=270,
        extent=90,
        fill=fill,
        outline="",
        tags=tags,
    )
    canvas.create_arc(
        x1,
        y2 - radius * 2,
        x1 + radius * 2,
        y2,
        start=180,
        extent=90,
        fill=fill,
        outline="",
        tags=tags,
    )
    if not outline or width <= 0:
        return
    canvas.create_line(x1 + radius, y1, x2 - radius, y1, fill=outline, width=width, tags=tags)
    canvas.create_line(x2, y1 + radius, x2, y2 - radius, fill=outline, width=width, tags=tags)
    canvas.create_line(x1 + radius, y2, x2 - radius, y2, fill=outline, width=width, tags=tags)
    canvas.create_line(x1, y1 + radius, x1, y2 - radius, fill=outline, width=width, tags=tags)
    canvas.create_arc(
        x1,
        y1,
        x1 + radius * 2,
        y1 + radius * 2,
        start=90,
        extent=90,
        style="arc",
        outline=outline,
        width=width,
        tags=tags,
    )
    canvas.create_arc(
        x2 - radius * 2,
        y1,
        x2,
        y1 + radius * 2,
        start=0,
        extent=90,
        style="arc",
        outline=outline,
        width=width,
        tags=tags,
    )
    canvas.create_arc(
        x2 - radius * 2,
        y2 - radius * 2,
        x2,
        y2,
        start=270,
        extent=90,
        style="arc",
        outline=outline,
        width=width,
        tags=tags,
    )
    canvas.create_arc(
        x1,
        y2 - radius * 2,
        x1 + radius * 2,
        y2,
        start=180,
        extent=90,
        style="arc",
        outline=outline,
        width=width,
        tags=tags,
    )


class CanvasButton(tk.Canvas):
    def __init__(
        self,
        master: tk.Misc,
        *,
        text: str = "",
        textvariable: tk.StringVar | None = None,
        command: Any | None = None,
        fill: str,
        hover_fill: str,
        foreground: str,
        outline: str = "",
        radius: int = 12,
        height: int = 36,
        width: int = 112,
        canvas_bg: str,
        font: tuple[str, int, str] = ("Microsoft YaHei UI", 9, "bold"),
    ) -> None:
        super().__init__(
            master,
            height=height,
            width=width,
            bg=canvas_bg,
            highlightthickness=0,
            borderwidth=0,
            cursor="hand2",
        )
        self._text = text
        self._textvariable = textvariable
        self._command = command
        self._fill = fill
        self._hover_fill = hover_fill
        self._foreground = foreground
        self._outline = outline or fill
        self._radius = radius
        self._height = height
        self._width = width
        self._font = font
        self._hovered = False
        self._background_image: Any | None = None
        if self._textvariable is not None:
            self._textvariable.trace_add("write", lambda *_args: self._redraw())
        self.bind("<Configure>", lambda _event: self._redraw())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self._redraw()

    def _button_text(self) -> str:
        if self._textvariable is not None:
            return self._textvariable.get()
        return self._text

    def raise_widget(self) -> None:
        tk.Misc.tkraise(self)

    def _on_enter(self, _event: tk.Event[Any]) -> None:
        self._hovered = True
        self._redraw()

    def _on_leave(self, _event: tk.Event[Any]) -> None:
        self._hovered = False
        self._redraw()

    def _on_click(self, _event: tk.Event[Any]) -> None:
        if self._command is not None:
            self._command()

    def _redraw(self) -> None:
        width = max(self.winfo_width(), self._width, 2)
        self.delete("all")
        fill = self._hover_fill if self._hovered else self._fill
        if Image is not None and ImageDraw is not None and ImageTk is not None:
            scale = 3
            image = Image.new("RGBA", (width * scale, self._height * scale), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle(
                (
                    scale,
                    scale,
                    (width - 1) * scale,
                    (self._height - 1) * scale,
                ),
                radius=self._radius * scale,
                fill=fill,
                outline=self._outline,
                width=scale,
            )
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            image = image.resize((width, self._height), resampling)
            self._background_image = ImageTk.PhotoImage(image)
            self.create_image(0, 0, image=self._background_image, anchor="nw")
        else:
            draw_rounded_rect(
                self,
                1,
                1,
                width - 1,
                self._height - 1,
                self._radius,
                fill=fill,
                outline=self._outline,
                width=1,
            )
        self.create_text(
            width // 2,
            self._height // 2,
            text=self._button_text(),
            anchor="center",
            fill=self._foreground,
            font=self._font,
        )


class CanvasChoice(tk.Canvas):
    def __init__(
        self,
        master: tk.Misc,
        *,
        textvariable: tk.StringVar,
        values: tuple[str, ...],
        fill: str,
        hover_fill: str,
        foreground: str,
        outline: str,
        canvas_bg: str,
        width: int = 170,
        height: int = 34,
        radius: int = 12,
    ) -> None:
        super().__init__(
            master,
            height=height,
            width=width,
            bg=canvas_bg,
            highlightthickness=0,
            borderwidth=0,
            cursor="hand2",
        )
        self._textvariable = textvariable
        self._values = values
        self._fill = fill
        self._hover_fill = hover_fill
        self._foreground = foreground
        self._outline = outline
        self._height = height
        self._radius = radius
        self._hovered = False
        self._enabled = True
        self._textvariable.trace_add("write", lambda *_args: self._redraw())
        self.bind("<Configure>", lambda _event: self._redraw())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._open_menu)
        self._redraw()

    def configure(self, cnf: Any | None = None, **kwargs: Any) -> Any:
        if cnf is None and "state" in kwargs:
            state = kwargs.pop("state")
            self._enabled = state != "disabled"
            super().configure(cursor="hand2" if self._enabled else "arrow")
            self._redraw()
            if not kwargs:
                return None
        return super().configure(cnf, **kwargs)

    config = configure

    def _on_enter(self, _event: tk.Event[Any]) -> None:
        self._hovered = True
        self._redraw()

    def _on_leave(self, _event: tk.Event[Any]) -> None:
        self._hovered = False
        self._redraw()

    def _open_menu(self, event: tk.Event[Any]) -> None:
        if not self._enabled:
            return
        menu = tk.Menu(self, tearoff=0)
        for value in self._values:
            menu.add_command(
                label=value,
                command=lambda selected=value: self._textvariable.set(selected),
            )
        menu.tk_popup(event.x_root, event.y_root)

    def _redraw(self) -> None:
        width = max(self.winfo_width(), 1)
        self.delete("all")
        fill = self._hover_fill if self._hovered and self._enabled else self._fill
        foreground = self._foreground if self._enabled else "#aaa39b"
        draw_rounded_rect(
            self,
            1,
            1,
            width - 1,
            self._height - 1,
            self._radius,
            fill=fill,
            outline=self._outline,
            width=1,
        )
        self.create_text(
            12,
            self._height // 2,
            text=self._textvariable.get(),
            anchor="w",
            fill=foreground,
            font=("Microsoft YaHei UI", 9),
        )
        self.create_text(
            width - 16,
            self._height // 2,
            text="⌄",
            anchor="center",
            fill=foreground,
            font=("Microsoft YaHei UI", 10, "bold"),
        )


class AutoHideScrollbar(tk.Canvas):
    def __init__(
        self,
        master: tk.Misc,
        *args: Any,
        orient: str = "vertical",
        command: Any | None = None,
        style: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            master,
            width=9,
            highlightthickness=0,
            borderwidth=0,
            bg=kwargs.pop("bg", "#ffffff"),
            cursor="hand2",
        )
        self._command = command
        self._first = 0.0
        self._last = 1.0
        self._drag_offset = 0
        self._dragging = False
        self._hovered = False
        self.bind("<Configure>", lambda _event: self._redraw())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)

    def set(self, first: Any, last: Any) -> None:
        try:
            first_float = float(first)
            last_float = float(last)
        except (TypeError, ValueError):
            first_float = 0.0
            last_float = 0.0
        self._first = max(0.0, min(1.0, first_float))
        self._last = max(self._first, min(1.0, last_float))
        if first_float <= 0.0 and last_float >= 1.0:
            self.grid_remove()
        else:
            self.grid()
        self._redraw()

    def _thumb_bounds(self) -> tuple[int, int, int, int]:
        height = max(1, self.winfo_height())
        visible = max(0.04, self._last - self._first)
        thumb_height = max(24, int(height * visible))
        y1 = int(height * self._first)
        y1 = max(2, min(height - thumb_height - 2, y1))
        y2 = min(height - 2, y1 + thumb_height)
        return (2, y1, 7, y2)

    def _redraw(self) -> None:
        self.delete("all")
        x1, y1, x2, y2 = self._thumb_bounds()
        fill = "#c7c3bf" if self._hovered or self._dragging else "#d4d1cd"
        draw_rounded_rect(
            self,
            x1,
            y1,
            x2,
            y2,
            4,
            fill=fill,
            outline=fill,
            width=0,
        )

    def _on_enter(self, _event: tk.Event[Any]) -> None:
        self._hovered = True
        self._redraw()

    def _on_leave(self, _event: tk.Event[Any]) -> None:
        self._hovered = False
        self._redraw()

    def _on_press(self, event: tk.Event[Any]) -> None:
        x1, y1, x2, y2 = self._thumb_bounds()
        if y1 <= event.y <= y2:
            self._dragging = True
            self._drag_offset = event.y - y1
        else:
            self._move_to_pointer(event.y)
        self._redraw()

    def _on_drag(self, event: tk.Event[Any]) -> None:
        if not self._dragging:
            return
        self._move_to_pointer(event.y - self._drag_offset)

    def _on_release(self, _event: tk.Event[Any]) -> None:
        self._dragging = False
        self._redraw()

    def _move_to_pointer(self, y: int) -> None:
        if self._command is None:
            return
        height = max(1, self.winfo_height())
        visible = max(0.04, self._last - self._first)
        thumb_height = max(24, int(height * visible))
        fraction = y / max(1, height - thumb_height)
        fraction = max(0.0, min(1.0 - visible, fraction))
        self._command("moveto", fraction)


class GuiAgentDesktopClient(tk.Tk):
    def __init__(
        self,
        state: DesktopClientState,
        *,
        state_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.state = state
        self.state_path = Path(state_path) if state_path is not None else None
        self.title("CFIE GUI Agent")
        self.minsize(1180, 720)
        self._configure_initial_window()

        self.colors = {
            "bg": "#ffffff",
            "sidebar": "#f7f4f1",
            "sidebar_hover": "#efebe7",
            "sidebar_selected": "#ebe7e2",
            "surface": "#ffffff",
            "surface_soft": "#f5f5f3",
            "surface_hover": "#eeeeec",
            "ink": "#202124",
            "muted": "#6f6f6f",
            "muted_2": "#a6a6a6",
            "line": "#e8e4df",
            "line_soft": "#eeeeea",
            "brand": "#111827",
            "brand_soft": "#f1f1ef",
            "accent": "#ff6b2b",
            "accent_soft": "#fff0e8",
            "success": "#0f9f7a",
        }

        initial_app_id = (
            state.selected_app_id
            if state.selected_app_id in state.target_apps
            else self._first_app_id()
        )
        state.selected_app_id = initial_app_id
        self.selected_app_id = tk.StringVar(value=initial_app_id)
        self.selected_request_id = tk.StringVar(value="")
        self.inspector_visible = tk.BooleanVar(value=False)
        self.direct_command_label = tk.StringVar(
            value=DIRECT_COMMAND_LABELS[DIRECT_COMMAND_NONE]
        )
        self.decision_type = tk.StringVar(value="人工回复")
        self.run_button_text = tk.StringVar(value="\u25b6")
        self.viewport_marker: ViewportMarkerWindow | None = None
        self.agent_running = tk.BooleanVar(value=False)
        self.agent_run_status = tk.StringVar(value="")
        self._agent_stop_event = threading.Event()
        self._timeline_images: list[Any] = []
        self._inspector_images: list[Any] = []
        self._selected_trace_event: Any | None = None
        self._inspector_width = 500
        self._inspector_min_width = 360
        self._inspector_max_width = 980
        self._inspector_resize_origin_x = 0
        self._inspector_resize_origin_width = self._inspector_width
        self._app_list_signature: tuple[Any, ...] | None = None
        self._message_signature: tuple[Any, ...] | None = None
        self._inspector_signature: tuple[Any, ...] | None = None
        self._focus_refresh_after_id: str | None = None

        self._setup_style()
        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<FocusIn>", self._on_focus_refresh, add="+")
        self.refresh_all(force=True)
        self.after(2000, self._periodic_refresh)

    def _configure_initial_window(self) -> None:
        screen_width = max(1, self.winfo_screenwidth())
        screen_height = max(1, self.winfo_screenheight())
        width = min(1440, max(1180, screen_width - 120))
        height = min(860, max(720, screen_height - 120))
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _setup_style(self) -> None:
        self.configure(bg=self.colors["bg"])
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.option_add("*Font", ("Microsoft YaHei UI", 10))
        self.option_add("*Menu.background", self.colors["surface"])
        self.option_add("*Menu.foreground", self.colors["ink"])
        self.option_add("*Menu.activeBackground", self.colors["surface_hover"])
        self.option_add("*Menu.activeForeground", self.colors["ink"])
        style.configure("Root.TFrame", background=self.colors["bg"])
        style.configure("Sidebar.TFrame", background=self.colors["sidebar"])
        style.configure("Surface.TFrame", background=self.colors["surface"])
        style.configure("Soft.TFrame", background=self.colors["surface_soft"])
        style.configure(
            "Title.TLabel",
            background=self.colors["surface"],
            foreground=self.colors["ink"],
            font=("Microsoft YaHei UI", 15, "bold"),
        )
        style.configure(
            "Section.TLabel",
            background=self.colors["surface"],
            foreground=self.colors["ink"],
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.configure(
            "Hint.TLabel",
            background=self.colors["surface"],
            foreground=self.colors["muted"],
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "SidebarTitle.TLabel",
            background=self.colors["sidebar"],
            foreground=self.colors["ink"],
            font=("Microsoft YaHei UI", 13, "bold"),
        )
        style.configure(
            "SidebarHint.TLabel",
            background=self.colors["sidebar"],
            foreground=self.colors["muted"],
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Primary.TButton",
            font=("Microsoft YaHei UI", 9, "bold"),
            foreground="#ffffff",
            background=self.colors["brand"],
            bordercolor=self.colors["brand"],
            lightcolor=self.colors["brand"],
            darkcolor=self.colors["brand"],
            relief="flat",
            borderwidth=0,
            focusthickness=0,
            padding=(14, 8),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#2b313c"), ("disabled", "#c9c3bc")],
            foreground=[("disabled", "#ffffff")],
        )
        style.configure(
            "TButton",
            font=("Microsoft YaHei UI", 9),
            foreground=self.colors["ink"],
            background=self.colors["surface"],
            bordercolor=self.colors["line"],
            lightcolor=self.colors["surface"],
            darkcolor=self.colors["surface"],
            relief="flat",
            borderwidth=1,
            focusthickness=0,
            padding=(12, 7),
        )
        style.map(
            "TButton",
            background=[("active", self.colors["surface_hover"])],
            bordercolor=[("active", "#d8d0c8")],
        )
        style.configure(
            "Icon.TButton",
            font=("Microsoft YaHei UI", 13, "bold"),
            foreground=self.colors["ink"],
            background=self.colors["sidebar"],
            bordercolor=self.colors["sidebar"],
            lightcolor=self.colors["sidebar"],
            darkcolor=self.colors["sidebar"],
            relief="flat",
            borderwidth=0,
            padding=(8, 3),
        )
        style.map("Icon.TButton", background=[("active", self.colors["sidebar_hover"])])
        style.configure(
            "TEntry",
            fieldbackground=self.colors["surface"],
            foreground=self.colors["ink"],
            bordercolor=self.colors["line"],
            lightcolor=self.colors["line"],
            darkcolor=self.colors["line"],
            insertcolor=self.colors["ink"],
            padding=(9, 7),
            borderwidth=1,
            relief="flat",
        )
        style.configure(
            "TCombobox",
            fieldbackground=self.colors["surface"],
            background=self.colors["surface"],
            foreground=self.colors["ink"],
            bordercolor=self.colors["line"],
            lightcolor=self.colors["line"],
            darkcolor=self.colors["line"],
            arrowsize=12,
            padding=(8, 6),
            borderwidth=1,
            relief="flat",
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", self.colors["surface"])],
            background=[("readonly", self.colors["surface"])],
        )
        style.configure(
            "Treeview",
            rowheight=30,
            font=("Microsoft YaHei UI", 9),
            background=self.colors["surface"],
            fieldbackground=self.colors["surface"],
            foreground=self.colors["ink"],
            bordercolor=self.colors["line_soft"],
            lightcolor=self.colors["surface"],
            darkcolor=self.colors["surface"],
            borderwidth=0,
            relief="flat",
        )
        style.map(
            "Treeview",
            background=[("selected", self.colors["sidebar_selected"])],
            foreground=[("selected", self.colors["ink"])],
        )
        style.configure(
            "Treeview.Heading",
            font=("Microsoft YaHei UI", 9, "bold"),
            background=self.colors["surface_soft"],
            foreground=self.colors["ink"],
            bordercolor=self.colors["line"],
            lightcolor=self.colors["surface_soft"],
            darkcolor=self.colors["surface_soft"],
            relief="flat",
        )
        style.layout(
            "Modern.Vertical.TScrollbar",
            [
                (
                    "Vertical.Scrollbar.trough",
                    {
                        "sticky": "ns",
                        "children": [
                            (
                                "Vertical.Scrollbar.thumb",
                                {"expand": "1", "sticky": "nswe"},
                            )
                        ],
                    },
                )
            ],
        )
        style.configure(
            "Modern.Vertical.TScrollbar",
            gripcount=0,
            width=3,
            background="#d2d0cc",
            troughcolor=self.colors["surface"],
            bordercolor=self.colors["surface"],
            lightcolor="#d2d0cc",
            darkcolor="#d2d0cc",
            arrowcolor=self.colors["surface"],
            relief="flat",
            borderwidth=0,
        )
        style.map(
            "Modern.Vertical.TScrollbar",
            background=[("active", "#c5c1bd")],
        )

    def _build_layout(self) -> None:
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, minsize=0)
        self.columnconfigure(3, minsize=0)
        self.rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_chat_area()
        self._build_inspector()
        if not self.inspector_visible.get():
            self.inspector_resize_handle.grid_remove()
            self.inspector.grid_remove()

    def _build_sidebar(self) -> None:
        sidebar = ttk.Frame(self, style="Sidebar.TFrame", padding=(12, 14))
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.rowconfigure(4, weight=1)

        top = ttk.Frame(sidebar, style="Sidebar.TFrame")
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        ttk.Label(top, text="CFIE", style="SidebarTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(
            top,
            text="+",
            width=3,
            style="Icon.TButton",
            command=self._show_create_menu,
        ).grid(
            row=0, column=1, sticky="e"
        )

        ttk.Label(
            sidebar,
            text="应用会话",
            style="SidebarHint.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(18, 6))
        search_holder = tk.Canvas(
            sidebar,
            height=38,
            bg=self.colors["sidebar"],
            highlightthickness=0,
            borderwidth=0,
        )
        search_holder.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.search_entry = tk.Entry(
            search_holder,
            relief="flat",
            borderwidth=0,
            bg=self.colors["surface"],
            fg=self.colors["muted"],
            insertbackground=self.colors["ink"],
            font=("Microsoft YaHei UI", 9),
        )
        self.search_window = search_holder.create_window(
            16,
            19,
            window=self.search_entry,
            anchor="w",
            height=24,
        )

        def redraw_search(event: tk.Event[Any]) -> None:
            search_holder.delete("search_bg")
            draw_rounded_rect(
                search_holder,
                1,
                1,
                event.width - 1,
                37,
                13,
                fill=self.colors["surface"],
                outline=self.colors["line"],
                tags="search_bg",
            )
            search_holder.tag_lower("search_bg")
            search_holder.itemconfigure(self.search_window, width=max(40, event.width - 32))

        search_holder.bind("<Configure>", redraw_search)
        self.search_entry.insert(0, "搜索 APP")
        self.search_entry.bind("<FocusIn>", self._clear_search_placeholder)

        self.app_canvas = tk.Canvas(
            sidebar,
            bg=self.colors["sidebar"],
            highlightthickness=0,
            borderwidth=0,
            width=260,
        )
        self.app_canvas.grid(row=4, column=0, sticky="nsew")
        self.app_list_frame = tk.Frame(self.app_canvas, bg=self.colors["sidebar"])
        self.app_canvas_window = self.app_canvas.create_window(
            (0, 0),
            window=self.app_list_frame,
            anchor="nw",
        )
        self.app_list_frame.bind(
            "<Configure>",
            lambda _event: self.app_canvas.configure(
                scrollregion=self.app_canvas.bbox("all")
            ),
        )
        self.app_canvas.bind(
            "<Configure>",
            lambda event: self.app_canvas.itemconfigure(
                self.app_canvas_window,
                width=event.width,
            ),
        )

        bottom = ttk.Frame(sidebar, style="Sidebar.TFrame")
        bottom.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        bottom.columnconfigure(0, weight=1)
        settings_button = CanvasButton(
            bottom,
            text="设置",
            command=self._open_settings,
            fill=self.colors["surface"],
            hover_fill=self.colors["surface_hover"],
            foreground=self.colors["ink"],
            outline=self.colors["line"],
            radius=12,
            height=36,
            width=230,
            canvas_bg=self.colors["sidebar"],
            font=("Microsoft YaHei UI", 9, "normal"),
        )
        settings_button._text = "\u2699  \u8bbe\u7f6e"
        settings_button._fill = self.colors["sidebar_hover"]
        settings_button._hover_fill = self.colors["surface_hover"]
        settings_button._outline = self.colors["sidebar_hover"]
        settings_button._height = 42
        settings_button.configure(height=42)
        settings_button.grid(row=0, column=0, sticky="ew")
        settings_button.after_idle(settings_button.raise_widget)
        self.status_text = tk.StringVar(value="生产模式")
        ttk.Label(
            bottom,
            textvariable=self.status_text,
            style="SidebarHint.TLabel",
            wraplength=230,
        ).grid(row=1, column=0, sticky="ew", pady=(8, 0))

    def _build_chat_area(self) -> None:
        self.chat_frame = ttk.Frame(self, style="Surface.TFrame")
        self.chat_frame.grid(row=0, column=1, sticky="nsew")
        self.chat_frame.columnconfigure(0, weight=1)
        self.chat_frame.rowconfigure(1, weight=1)

        header = ttk.Frame(self.chat_frame, style="Surface.TFrame", padding=(18, 12))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        self.app_title = ttk.Label(header, text="未选择 APP", style="Title.TLabel")
        self.app_title.grid(row=0, column=0, sticky="w")
        self.app_subtitle = ttk.Label(
            header,
            text="左侧新建或选择应用",
            style="Hint.TLabel",
        )
        self.app_subtitle.grid(row=1, column=0, sticky="w", pady=(3, 0))
        run_button = CanvasButton(
            header,
            textvariable=self.run_button_text,
            command=self._start_selected_agent_run,
            fill=self.colors["surface"],
            hover_fill=self.colors["surface_hover"],
            foreground=self.colors["ink"],
            outline=self.colors["line"],
            radius=14,
            height=36,
            width=44,
            canvas_bg=self.colors["surface"],
            font=("Microsoft YaHei UI", 12, "bold"),
        )
        run_button.grid(row=0, column=1, rowspan=2, sticky="e", padx=(8, 0))
        run_button.after_idle(run_button.raise_widget)

        inspector_button = CanvasButton(
            header,
            text="◨",
            command=self._toggle_inspector,
            fill=self.colors["surface"],
            hover_fill=self.colors["surface_hover"],
            foreground=self.colors["ink"],
            outline=self.colors["line"],
            radius=14,
            height=36,
            width=44,
            canvas_bg=self.colors["surface"],
            font=("Microsoft YaHei UI", 13, "normal"),
        )
        inspector_button.grid(row=0, column=2, rowspan=2, sticky="e", padx=(8, 0))
        inspector_button.after_idle(inspector_button.raise_widget)

        canvas_holder = ttk.Frame(self.chat_frame, style="Surface.TFrame")
        canvas_holder.grid(row=1, column=0, sticky="nsew")
        canvas_holder.columnconfigure(0, weight=1)
        canvas_holder.rowconfigure(0, weight=1)

        self.chat_canvas = tk.Canvas(
            canvas_holder,
            bg=self.colors["surface"],
            highlightthickness=0,
        )
        self.chat_canvas.grid(row=0, column=0, sticky="nsew")
        scroll = AutoHideScrollbar(
            canvas_holder,
            orient="vertical",
            command=self.chat_canvas.yview,
            style="Modern.Vertical.TScrollbar",
        )
        scroll.grid(row=0, column=1, sticky="ns")
        self.chat_canvas.configure(yscrollcommand=scroll.set)
        self.messages_frame = ttk.Frame(self.chat_canvas, style="Surface.TFrame")
        self.messages_window = self.chat_canvas.create_window(
            (0, 0),
            window=self.messages_frame,
            anchor="nw",
        )
        self.messages_frame.bind("<Configure>", self._on_messages_configure)
        self.chat_canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel_tree(canvas_holder, self.chat_canvas)

        composer = ttk.Frame(self.chat_frame, style="Surface.TFrame", padding=(18, 12))
        composer.grid(row=2, column=0, sticky="ew")
        composer.columnconfigure(0, weight=1)
        composer.columnconfigure(1, weight=0)
        self.composer_mode_text = tk.StringVar(value="")
        self.send_button_text = tk.StringVar(value="发送")
        toolbar = ttk.Frame(composer, style="Surface.TFrame")
        self.composer_toolbar = toolbar
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        toolbar.columnconfigure(0, weight=1)
        self.direct_command_combo = CanvasChoice(
            toolbar,
            textvariable=self.direct_command_label,
            values=tuple(DIRECT_COMMAND_LABELS.values()),
            fill=self.colors["surface"],
            hover_fill=self.colors["surface_hover"],
            foreground=self.colors["ink"],
            outline=self.colors["line"],
            canvas_bg=self.colors["surface"],
            width=178,
            height=34,
            radius=12,
        )
        self.direct_command_combo.grid(row=0, column=1, sticky="e")
        composer_text_shell = tk.Canvas(
            composer,
            height=104,
            bg=self.colors["surface"],
            highlightthickness=0,
            borderwidth=0,
        )
        composer_text_shell.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.composer_text = tk.Text(
            composer_text_shell,
            height=4,
            wrap="word",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            bg="#ffffff",
            fg=self.colors["ink"],
            insertbackground=self.colors["ink"],
            padx=12,
            pady=10,
            font=("Microsoft YaHei UI", 10),
        )
        composer_text_window = composer_text_shell.create_window(
            12,
            12,
            window=self.composer_text,
            anchor="nw",
        )
        send_button = CanvasButton(
            composer_text_shell,
            textvariable=self.send_button_text,
            command=self._submit_user_message,
            fill=self.colors["brand"],
            hover_fill="#2b313c",
            foreground="#ffffff",
            outline=self.colors["brand"],
            radius=14,
            height=42,
            width=82,
            canvas_bg=self.colors["surface"],
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        send_button_window = composer_text_shell.create_window(
            0,
            0,
            window=send_button,
            anchor="e",
        )

        def redraw_composer_text(event: tk.Event[Any]) -> None:
            composer_text_shell.delete("composer_bg")
            draw_rounded_rect(
                composer_text_shell,
                1,
                1,
                event.width - 1,
                103,
                16,
                fill="#ffffff",
                outline=self.colors["line"],
                tags="composer_bg",
            )
            composer_text_shell.tag_lower("composer_bg")
            composer_text_shell.itemconfigure(
                composer_text_window,
                width=max(260, event.width - 128),
                height=80,
            )
            composer_text_shell.coords(send_button_window, event.width - 12, 52)
            send_button.raise_widget()

        composer_text_shell.bind("<Configure>", redraw_composer_text)
        send_button.after_idle(send_button.raise_widget)

    def _build_inspector(self) -> None:
        self.inspector_resize_handle = tk.Canvas(
            self,
            width=6,
            bg=self.colors["bg"],
            highlightthickness=0,
            borderwidth=0,
            cursor="sb_h_double_arrow",
        )
        self.inspector_resize_handle.grid(row=0, column=2, sticky="ns")
        self.inspector_resize_handle.create_rectangle(
            2,
            0,
            4,
            5000,
            fill=self.colors["line_soft"],
            outline="",
            tags="line",
        )
        self.inspector_resize_handle.bind("<ButtonPress-1>", self._start_inspector_resize)
        self.inspector_resize_handle.bind("<B1-Motion>", self._drag_inspector_resize)
        self.inspector_resize_handle.bind(
            "<Enter>",
            lambda _event: self.inspector_resize_handle.itemconfigure(
                "line",
                fill="#d8d3cd",
            ),
        )
        self.inspector_resize_handle.bind(
            "<Leave>",
            lambda _event: self.inspector_resize_handle.itemconfigure(
                "line",
                fill=self.colors["line_soft"],
            ),
        )

        self.inspector = ttk.Frame(self, style="Surface.TFrame", padding=(14, 14))
        self.inspector.configure(width=self._inspector_width)
        self.inspector.grid(row=0, column=3, sticky="nsew")
        self.inspector.grid_propagate(False)
        self.inspector.columnconfigure(0, weight=1)
        self.inspector.rowconfigure(2, weight=1)

        header = ttk.Frame(self.inspector, style="Surface.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="详情", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            self.inspector,
            text="选中主窗口中的执行记录后，查看输入、输出、工具调用和截图。",
            style="Hint.TLabel",
            wraplength=self._detail_wraplength(90),
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))

        collapse_button = CanvasButton(
            header,
            text="›",
            command=self._toggle_inspector,
            fill=self.colors["surface"],
            hover_fill=self.colors["surface_hover"],
            foreground=self.colors["ink"],
            outline=self.colors["line"],
            radius=13,
            height=34,
            width=38,
            canvas_bg=self.colors["surface"],
            font=("Microsoft YaHei UI", 15, "normal"),
        )
        collapse_button.grid(row=0, column=1, sticky="e")
        collapse_button.after_idle(collapse_button.raise_widget)

        inspector_list_holder = ttk.Frame(self.inspector, style="Surface.TFrame")
        inspector_list_holder.grid(row=2, column=0, sticky="nsew")
        inspector_list_holder.columnconfigure(0, weight=1)
        inspector_list_holder.rowconfigure(0, weight=1)
        self.inspector_canvas = tk.Canvas(
            inspector_list_holder,
            bg=self.colors["surface"],
            highlightthickness=0,
            borderwidth=0,
        )
        self.inspector_canvas.grid(row=0, column=0, sticky="nsew")
        inspector_scroll = AutoHideScrollbar(
            inspector_list_holder,
            orient="vertical",
            command=self.inspector_canvas.yview,
            style="Modern.Vertical.TScrollbar",
        )
        inspector_scroll.grid(row=0, column=1, sticky="ns")
        self.inspector_canvas.configure(yscrollcommand=inspector_scroll.set)
        self.inspector_list_frame = tk.Frame(
            self.inspector_canvas,
            bg=self.colors["surface"],
        )
        self.inspector_canvas_window = self.inspector_canvas.create_window(
            (0, 0),
            window=self.inspector_list_frame,
            anchor="nw",
        )
        self.inspector_list_frame.bind(
            "<Configure>",
            lambda _event: self.inspector_canvas.configure(
                scrollregion=self.inspector_canvas.bbox("all")
            ),
        )
        self.inspector_canvas.bind(
            "<Configure>",
            lambda event: self.inspector_canvas.itemconfigure(
                self.inspector_canvas_window,
                width=max(1, event.width),
            ),
        )
        self._bind_mousewheel_tree(inspector_list_holder, self.inspector_canvas)

        inspector_detail_shell = tk.Canvas(
            self.inspector,
            height=154,
            bg=self.colors["surface"],
            highlightthickness=0,
            borderwidth=0,
        )
        self.inspector_detail_shell = inspector_detail_shell
        inspector_detail_shell.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.inspector_detail = tk.Text(
            inspector_detail_shell,
            height=12,
            wrap="word",
            bg=self.colors["surface_soft"],
            fg=self.colors["ink"],
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=12,
            pady=10,
            font=("Microsoft YaHei UI", 9),
        )
        inspector_detail_window = inspector_detail_shell.create_window(
            10,
            10,
            window=self.inspector_detail,
            anchor="nw",
        )

        def redraw_inspector_detail(event: tk.Event[Any]) -> None:
            inspector_detail_shell.delete("detail_bg")
            draw_rounded_rect(
                inspector_detail_shell,
                1,
                1,
                event.width - 1,
                153,
                16,
                fill=self.colors["surface_soft"],
                outline=self.colors["line_soft"],
                tags="detail_bg",
            )
            inspector_detail_shell.tag_lower("detail_bg")
            inspector_detail_shell.itemconfigure(
                inspector_detail_window,
                width=max(40, event.width - 20),
                height=132,
            )

        inspector_detail_shell.bind("<Configure>", redraw_inspector_detail)
        self.inspector_detail.configure(state="disabled")
        inspector_detail_shell.grid_remove()

    def _start_inspector_resize(self, event: tk.Event[Any]) -> None:
        self._inspector_resize_origin_x = int(event.x_root)
        self._inspector_resize_origin_width = self._inspector_width

    def _drag_inspector_resize(self, event: tk.Event[Any]) -> None:
        delta = self._inspector_resize_origin_x - int(event.x_root)
        self._set_inspector_width(self._inspector_resize_origin_width + delta)

    def _set_inspector_width(self, width: int) -> None:
        width = max(self._inspector_min_width, min(self._inspector_max_width, int(width)))
        if width == self._inspector_width:
            return
        self._inspector_width = width
        if self.inspector_visible.get():
            self.columnconfigure(3, minsize=width)
        self.inspector.configure(width=width)
        self.inspector_canvas.configure(scrollregion=self.inspector_canvas.bbox("all"))

    def _detail_wraplength(self, reserved: int = 80) -> int:
        canvas = getattr(self, "inspector_canvas", None)
        width = canvas.winfo_width() if canvas is not None else 0
        if width <= 1:
            width = self._inspector_width
        return max(240, width - reserved)

    def refresh_all(self, *, force: bool = False) -> None:
        self.refresh_apps(force=force)
        self.refresh_header()
        self.refresh_messages(force=force)
        self.refresh_inspector(force=force)
        self.refresh_composer()

    def refresh_apps(self, *, force: bool = False) -> None:
        current = self.selected_app_id.get()
        app_ids = list(self.state.target_apps)
        signature = (
            current,
            tuple(
                (
                    app_id,
                    config.app_name,
                    config.job_id,
                    self._app_status_text(app_id, config.job_id),
                    bool((config.metadata or {}).get("manual_viewport")),
                    len(config.reference_assets),
                    len(self._macros_for_app(app_id)),
                )
                for app_id, config in self.state.target_apps.items()
            ),
        )
        if not force and signature == self._app_list_signature:
            return
        self._app_list_signature = signature
        for child in self.app_list_frame.winfo_children():
            child.destroy()
        for app_id in app_ids:
            config = self.state.target_apps[app_id]
            subtitle = self._app_status_text(app_id, config.job_id)
            self._add_app_card(app_id=app_id, config=config, subtitle=subtitle)
        if current not in self.state.target_apps and app_ids:
            current = app_ids[0]
            self.selected_app_id.set(current)
            self.state.selected_app_id = current

    def refresh_header(self) -> None:
        config = self._selected_config()
        if config is None:
            self.app_title.configure(text="未选择 APP")
            self.app_subtitle.configure(text="左侧新建或选择应用")
            self.run_button_text.set("\u25b6")
            return
        refs = len(config.reference_assets)
        macros = len(self._macros_for_app(config.app_id))
        self.app_title.configure(text=config.app_name)
        parts = []
        if (config.metadata or {}).get("manual_viewport"):
            parts.append("视野已标定")
        if refs:
            parts.append(f"{refs} 个素材")
        if macros:
            parts.append(f"{macros} 个宏")
        if self.agent_running.get():
            parts.append("运行中")
            self.run_button_text.set("\u25a0")
        else:
            status_text = self._app_status_text(config.app_id, config.job_id)
            if status_text != "就绪":
                parts.append(status_text)
            self.run_button_text.set(
                "\u2298"
                if self._is_terminal_agent_status(
                    self._latest_agent_run_status(config.app_id)
                )
                else "\u25b6"
            )
        self.app_subtitle.configure(text=" · ".join(parts) if parts else "就绪")

    def refresh_messages(self, *, force: bool = False) -> None:
        config = self._selected_config()
        if config is None:
            signature = ("empty",)
            if not force and signature == self._message_signature:
                return
            self._message_signature = signature
            for child in self.messages_frame.winfo_children():
                child.destroy()
            self._timeline_images.clear()
            self._add_empty_message()
            return
        has_visible_content = False
        events = self._events_for_selected_app(config.app_id)
        self._ensure_pending_human_request_from_waiting_agent_run(config, events)
        human_items = [
            item
            for item in self.state.human_loop.list_requests(include_completed=True)
            if (item["request"].get("metadata") or {}).get("job_id") == config.job_id
        ]
        signature = (
            config.app_id,
            tuple(self._trace_event_signature(event) for event in events),
            tuple(self._human_item_signature(item) for item in human_items),
        )
        if not force and signature == self._message_signature:
            return
        self._message_signature = signature
        for child in self.messages_frame.winfo_children():
            child.destroy()
        self._timeline_images.clear()
        for event in events:
            if self._add_operation_card(event):
                has_visible_content = True
        for item in human_items:
            self._add_human_request_card(item)
            has_visible_content = True
        if not has_visible_content:
            self._add_empty_trace_message()
        self.after_idle(self._scroll_messages_to_bottom)

    def _ensure_pending_human_request_from_waiting_agent_run(
        self,
        config: TargetAppConfig,
        events: list[Any],
    ) -> None:
        latest_waiting: Any | None = None
        for event in reversed(events):
            payload = event.payload
            if (
                event.kind == "operation"
                and payload.get("kind") == "agent_run"
                and payload.get("status") == "waiting_human"
            ):
                latest_waiting = event
                break
        if latest_waiting is None:
            return
        key = ":".join(self._agent_run_operation_key(latest_waiting.payload))
        if not key.strip(":"):
            key = str(latest_waiting.payload.get("summary") or "waiting_human")
        for item in self.state.human_loop.list_requests(include_completed=True):
            request = item["request"]
            metadata = request.get("metadata") or {}
            if metadata.get("job_id") != config.job_id:
                continue
            if metadata.get("agent_waiting_key") == key:
                return
            if item["status"] != "resolved":
                return
        payload = latest_waiting.payload
        nested = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        reason = self._friendly_reason(
            str(nested.get("result_reason") or payload.get("summary") or "")
        )
        self.state.human_loop.request_help(
            question="当前任务需要人工处理后才能继续。",
            task_id=None,
            risk_reason=reason or "Agent 等待人工介入。",
            proposed_action="请查看上方执行记录，输入下一步处理意见或约束。",
            urgency="normal",
            metadata={
                "job_id": config.job_id,
                "app_id": config.app_id,
                "source": "agent_trace",
                "agent_waiting_key": key,
            },
        )

    def refresh_composer(self) -> None:
        pending = self._current_pending_request()
        if pending is None:
            self.composer_mode_text.set("")
            self.send_button_text.set("发送")
            self.direct_command_combo.configure(state="disabled")
            self.direct_command_label.set(DIRECT_COMMAND_LABELS[DIRECT_COMMAND_NONE])
            self.composer_toolbar.grid_remove()
            return
        self.composer_mode_text.set("")
        self.send_button_text.set("提交")
        self.composer_toolbar.grid()
        self.direct_command_combo.configure(state="readonly")

    def refresh_inspector(self, *, force: bool = False) -> None:
        signature = (
            self.selected_app_id.get(),
            self._trace_event_signature(self._selected_trace_event)
            if self._selected_trace_event is not None
            else None,
        )
        if not force and signature == self._inspector_signature:
            return
        self._inspector_signature = signature
        for child in self.inspector_list_frame.winfo_children():
            child.destroy()
        self._inspector_images.clear()
        if self._selected_trace_event is not None:
            self._render_trace_event_detail(self._selected_trace_event)
            return
        self.inspector_detail_shell.grid_remove()
        self._set_text(self.inspector_detail, "")

    def _periodic_refresh(self) -> None:
        self.refresh_apps()
        self.refresh_header()
        self.refresh_composer()
        if self.focus_displayof() is not None:
            self.refresh_messages()
            self.refresh_inspector()
        self.after(2000, self._periodic_refresh)

    def _on_focus_refresh(self, _event: tk.Event[Any]) -> None:
        if self.focus_displayof() is None:
            return
        if self._focus_refresh_after_id is not None:
            try:
                self.after_cancel(self._focus_refresh_after_id)
            except tk.TclError:
                pass
        self._focus_refresh_after_id = self.after(120, self._run_focus_refresh)

    def _run_focus_refresh(self) -> None:
        self._focus_refresh_after_id = None
        self.refresh_all()

    @staticmethod
    def _trace_event_signature(event: Any | None) -> tuple[Any, ...]:
        if event is None:
            return ("none",)
        payload = event.payload if isinstance(getattr(event, "payload", None), dict) else {}
        action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        image_refs = payload.get("image_refs") if isinstance(payload.get("image_refs"), list) else []
        return (
            getattr(event, "kind", ""),
            getattr(event, "time_unix_nano", 0),
            payload.get("step"),
            payload.get("step_id"),
            payload.get("kind"),
            payload.get("status"),
            payload.get("title"),
            payload.get("summary"),
            payload.get("app_id"),
            action.get("type"),
            action.get("name"),
            metadata.get("model_response_step"),
            payload.get("latency_seconds"),
            payload.get("input_tokens"),
            payload.get("output_tokens"),
            payload.get("reasoning_tokens"),
            payload.get("response_json_chars"),
            tuple(str(ref) for ref in image_refs),
        )

    @staticmethod
    def _human_item_signature(item: dict[str, Any]) -> tuple[Any, ...]:
        request = item.get("request") if isinstance(item.get("request"), dict) else {}
        metadata = request.get("metadata") if isinstance(request.get("metadata"), dict) else {}
        reply = item.get("reply") if isinstance(item.get("reply"), dict) else {}
        return (
            request.get("request_id"),
            item.get("status"),
            request.get("task_id"),
            request.get("question"),
            request.get("risk_reason"),
            request.get("proposed_action"),
            metadata.get("job_id"),
            metadata.get("app_id"),
            metadata.get("agent_waiting_key"),
            reply.get("text"),
            reply.get("decision_type"),
        )

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        self.state.selected_app_id = self.selected_app_id.get()
        try:
            save_desktop_state(self.state, self.state_path)
        except OSError as exc:
            self.status_text.set(f"状态保存失败：{exc}")

    def _on_close(self) -> None:
        self._save_state()
        self.destroy()

    def _load_selected_trace_if_available(self, app_id: str) -> None:
        state = getattr(self, "state", None)
        config = state.target_apps.get(app_id) if state is not None else None
        if config is None:
            return
        trace_path = str((config.metadata or {}).get("trace_path") or "").strip()
        if not trace_path:
            return
        try:
            self.state.trace_store.load_existing(trace_path)
        except OSError:
            self.state.trace_store.path = Path(trace_path)

    def _add_empty_message(self) -> None:
        holder = ttk.Frame(self.messages_frame, style="Surface.TFrame", padding=(0, 120))
        holder.grid(row=0, column=0, sticky="nsew")
        holder.columnconfigure(0, weight=1)
        ttk.Label(
            holder,
            text="从左侧新建 APP，开始配置自动化任务。",
            background=self.colors["surface"],
            foreground=self.colors["muted"],
            font=("Microsoft YaHei UI", 13, "bold"),
        ).grid(row=0, column=0)
        self._bind_mousewheel_tree(holder, self.chat_canvas)

    def _add_empty_trace_message(self) -> None:
        self._add_message(
            role="trace",
            title="",
            text="还没有执行记录。启动后，这里会显示模型的意图、电脑操作和关键截图。",
            align="left",
        )

    def _add_message(self, *, role: str, title: str, text: str, align: str) -> None:
        row = len(self.messages_frame.winfo_children())
        outer = ttk.Frame(self.messages_frame, style="Surface.TFrame", padding=(18, 8))
        outer.grid(row=row, column=0, sticky="ew")
        outer.columnconfigure(0, weight=1)
        canvas_width = min(820, max(620, self.chat_canvas.winfo_width() - 180))
        fill = self._role_color(role)
        outline = "#ebe5de" if role in {"system", "trace"} else fill
        bubble = tk.Canvas(
            outer,
            width=canvas_width,
            height=72,
            bg=self.colors["surface"],
            highlightthickness=0,
            borderwidth=0,
        )
        bubble.grid(
            row=0,
            column=0,
            sticky="e" if align == "right" else "w",
            padx=(96, 0) if align == "right" else (0, 96),
        )
        title_font = tkfont.Font(family="Microsoft YaHei UI", size=8, weight="bold")
        body_font = tkfont.Font(family="Microsoft YaHei UI", size=10)
        body_y = 34
        if title:
            bubble.create_text(
                18,
                13,
                text=title,
                anchor="w",
                justify="left",
                fill=self.colors["muted"],
                font=title_font,
            )
        else:
            body_y = 20
        body = bubble.create_text(
            18,
            body_y,
            text=text or "",
            anchor="nw",
            justify="left",
            width=canvas_width - 36,
            fill=self.colors["ink"],
            font=body_font,
        )
        bbox = bubble.bbox(body)
        height = max(66, (bbox[3] if bbox else 48) + 18)
        bubble.configure(height=height)
        draw_rounded_rect(
            bubble,
            2,
            2,
            canvas_width - 2,
            height - 2,
            16,
            fill=fill,
            outline=outline,
            width=1,
            tags="bubble_bg",
        )
        bubble.tag_lower("bubble_bg")
        self._bind_mousewheel_tree(outer, self.chat_canvas)

    def _add_operation_card(self, event: Any) -> bool:
        view = self._timeline_view_for_event(event)
        if view is None:
            return False
        title = view["title"]
        body = view["body"]
        detail = view.get("detail") or ""
        accent = view["accent"]
        image_refs = tuple(view.get("image_refs") or ())
        status = view.get("status") or ""
        row = len(self.messages_frame.winfo_children())
        outer = ttk.Frame(self.messages_frame, style="Surface.TFrame", padding=(18, 6))
        outer.grid(row=row, column=0, sticky="ew")
        outer.columnconfigure(0, weight=1)
        width = min(880, max(560, self.chat_canvas.winfo_width() - 150))
        shell = tk.Canvas(
            outer,
            width=width,
            height=96,
            bg=self.colors["surface"],
            highlightthickness=0,
            borderwidth=0,
            cursor="hand2",
        )
        shell.grid(row=0, column=0, sticky="w", padx=(0, 96))
        content = tk.Frame(shell, bg="#ffffff")
        content_window = shell.create_window(18, 16, window=content, anchor="nw")

        top = tk.Frame(content, bg="#ffffff")
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        tk.Label(
            top,
            text=title,
            bg="#ffffff",
            fg=self.colors["ink"],
            anchor="w",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w")
        if status:
            tk.Label(
                top,
                text=status,
                bg=accent[1],
                fg=accent[0],
                padx=10,
                pady=3,
                font=("Microsoft YaHei UI", 8, "bold"),
            ).grid(row=0, column=1, sticky="e")

        if body:
            tk.Label(
                content,
                text=body,
                bg="#ffffff",
                fg=self.colors["ink"],
                anchor="w",
                justify="left",
                wraplength=max(360, width - 84),
                font=("Microsoft YaHei UI", 10),
            ).grid(row=1, column=0, sticky="ew", pady=(8, 0))
        if detail:
            tk.Label(
                content,
                text=detail,
                bg="#ffffff",
                fg=self.colors["muted"],
                anchor="w",
                justify="left",
                wraplength=max(360, width - 84),
                font=("Microsoft YaHei UI", 9),
            ).grid(row=2, column=0, sticky="ew", pady=(5, 0))
        image_labels: list[tuple[tk.Label, str]] = []
        if image_refs:
            images = tk.Frame(content, bg="#ffffff")
            images.grid(row=3, column=0, sticky="w", pady=(10, 0))
            rendered = 0
            for ref in image_refs[:4]:
                photo = self._timeline_thumbnail(ref)
                if photo is None:
                    continue
                self._timeline_images.append(photo)
                label = tk.Label(
                    images,
                    image=photo,
                    bg="#ffffff",
                    highlightthickness=1,
                    highlightbackground=self.colors["line_soft"],
                    borderwidth=0,
                )
                label.grid(row=0, column=rendered, padx=(0, 8))
                image_labels.append((label, ref))
                rendered += 1

        def redraw(_event: tk.Event[Any] | None = None) -> None:
            try:
                if not shell.winfo_exists() or not content.winfo_exists():
                    return
                card_width = max(shell.winfo_width(), width)
                content_width = max(120, card_width - 36)
                shell.itemconfigure(content_window, width=content_width)
                content.update_idletasks()
                height = max(88, content.winfo_reqheight() + 32)
                shell.configure(height=height)
                shell.delete("bg")
            except tk.TclError:
                return
            draw_rounded_rect(
                shell,
                2,
                2,
                card_width - 2,
                height - 2,
                18,
                fill="#ffffff",
                outline=self.colors["line_soft"],
                width=1,
                tags="bg",
            )
            draw_rounded_rect(
                shell,
                2,
                2,
                7,
                height - 2,
                3,
                fill=accent[0],
                outline=accent[0],
                tags="bg",
            )
            shell.tag_lower("bg")

        content.bind("<Configure>", redraw)
        shell.bind("<Configure>", redraw)
        shell.bind("<Button-1>", lambda _event, value=event: self._select_trace_event(value))
        self._bind_click_tree(content, lambda value=event: self._select_trace_event(value))
        for label, ref in image_labels:
            label.configure(cursor="hand2")
            label.bind(
                "<Button-1>",
                lambda _event, value=ref: self._open_image_ref_window(value, "截图"),
            )
        self._bind_mousewheel_tree(outer, self.chat_canvas)
        redraw()
        return True

    def _add_human_request_card(self, item: dict[str, Any]) -> None:
        request = item["request"]
        is_resolved = item.get("status") == "resolved"
        row = len(self.messages_frame.winfo_children())
        outer = ttk.Frame(self.messages_frame, style="Surface.TFrame", padding=(18, 8))
        outer.grid(row=row, column=0, sticky="ew")
        outer.columnconfigure(0, weight=1)
        width = min(880, max(560, self.chat_canvas.winfo_width() - 150))
        shell = tk.Canvas(
            outer,
            width=width,
            height=190 if not is_resolved else 110,
            bg=self.colors["surface"],
            highlightthickness=0,
            borderwidth=0,
        )
        shell.grid(row=0, column=0, sticky="w", padx=(0, 96))
        content = tk.Frame(shell, bg="#fff8f1")
        content_window = shell.create_window(18, 16, window=content, anchor="nw")

        tk.Label(
            content,
            text="需要人工处理" if not is_resolved else "人工处理已提交",
            bg="#fff8f1",
            fg="#9a4f00",
            anchor="w",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w")
        body = self._human_request_user_text(item)
        tk.Label(
            content,
            text=body,
            bg="#fff8f1",
            fg=self.colors["ink"],
            anchor="w",
            justify="left",
            wraplength=max(360, width - 80),
            font=("Microsoft YaHei UI", 10),
        ).grid(row=1, column=0, sticky="ew", pady=(8, 0))

        next_row = 2
        macro_previews = self._human_macro_preview_refs(item)
        if macro_previews:
            previews = tk.Frame(content, bg="#fff8f1")
            previews.grid(row=next_row, column=0, sticky="w", pady=(10, 0))
            rendered = 0
            for label_text, ref in macro_previews[:6]:
                photo = self._timeline_thumbnail(ref)
                if photo is None:
                    continue
                self._timeline_images.append(photo)
                cell = tk.Frame(previews, bg="#fff8f1")
                cell.grid(row=0, column=rendered, sticky="nw", padx=(0, 10))
                image_label = tk.Label(
                    cell,
                    image=photo,
                    bg="#fff8f1",
                    highlightthickness=1,
                    highlightbackground=self.colors["line_soft"],
                    borderwidth=0,
                )
                image_label.grid(row=0, column=0)
                image_label.configure(cursor="hand2")
                image_label.bind(
                    "<Button-1>",
                    lambda _event, value=ref: self._open_image_ref_window(value, "宏点击预览"),
                )
                tk.Label(
                    cell,
                    text=label_text,
                    bg="#fff8f1",
                    fg=self.colors["muted"],
                    font=("Microsoft YaHei UI", 8),
                ).grid(row=1, column=0, pady=(4, 0))
                rendered += 1
            if rendered:
                next_row += 1

        if not is_resolved:
            if self._human_macro_proposal(item) is not None:
                approve_button = CanvasButton(
                    content,
                    text="批准并启用宏",
                    command=lambda value=item: self._approve_human_macro_request(value),
                    fill="#0f9f7a",
                    hover_fill="#087443",
                    foreground="#ffffff",
                    outline="#0f9f7a",
                    radius=13,
                    height=34,
                    width=118,
                    canvas_bg="#fff8f1",
                    font=("Microsoft YaHei UI", 9, "bold"),
                )
                approve_button.grid(row=next_row, column=0, sticky="w", pady=(10, 0))
                next_row += 1
            input_shell = tk.Canvas(
                content,
                height=72,
                bg="#fff8f1",
                highlightthickness=0,
                borderwidth=0,
            )
            input_shell.grid(row=next_row, column=0, sticky="ew", pady=(12, 0))
            reply_text = tk.Text(
                input_shell,
                height=3,
                wrap="word",
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
                bg="#ffffff",
                fg=self.colors["ink"],
                insertbackground=self.colors["ink"],
                padx=10,
                pady=8,
                font=("Microsoft YaHei UI", 10),
            )
            reply_window = input_shell.create_window(12, 10, window=reply_text, anchor="nw")
            submit = CanvasButton(
                input_shell,
                text="提交",
                command=lambda value=item, widget=reply_text: self._submit_human_card_reply(value, widget),
                fill=self.colors["brand"],
                hover_fill="#2b313c",
                foreground="#ffffff",
                outline=self.colors["brand"],
                radius=13,
                height=36,
                width=70,
                canvas_bg="#fff8f1",
                font=("Microsoft YaHei UI", 9, "bold"),
            )
            submit_window = input_shell.create_window(0, 0, window=submit, anchor="se")

            def redraw_input(event: tk.Event[Any]) -> None:
                input_shell.delete("input_bg")
                draw_rounded_rect(
                    input_shell,
                    1,
                    1,
                    event.width - 1,
                    71,
                    14,
                    fill="#ffffff",
                    outline=self.colors["line"],
                    tags="input_bg",
                )
                input_shell.tag_lower("input_bg")
                input_shell.itemconfigure(reply_window, width=max(180, event.width - 108), height=52)
                input_shell.coords(submit_window, event.width - 10, 62)
                submit.raise_widget()

            input_shell.bind("<Configure>", redraw_input)

        def redraw(_event: tk.Event[Any] | None = None) -> None:
            card_width = max(shell.winfo_width(), width)
            shell.itemconfigure(content_window, width=max(120, card_width - 36))
            content.update_idletasks()
            height = max(100, content.winfo_reqheight() + 32)
            shell.configure(height=height)
            shell.delete("bg")
            draw_rounded_rect(
                shell,
                2,
                2,
                card_width - 2,
                height - 2,
                18,
                fill="#fff8f1",
                outline="#f0d8c7",
                width=1,
                tags="bg",
            )
            shell.tag_lower("bg")

        content.bind("<Configure>", redraw)
        shell.bind("<Configure>", redraw)
        self._bind_mousewheel_tree(outer, self.chat_canvas)
        redraw()

    def _timeline_view_for_event(self, event: Any) -> dict[str, Any] | None:
        payload = event.payload
        if event.kind == "model_response":
            return None
        if event.kind == "step":
            action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
            result = str(payload.get("result") or payload.get("status") or "")
            model_event = self._model_response_event_for_trace_event(event)
            intent = (
                self._model_intent_text(model_event.payload)
                if model_event is not None
                else ""
            )
            if action.get("type") == "computer_call":
                actions = action.get("actions") if isinstance(action.get("actions"), list) else []
                title, body = self._computer_action_timeline_text(actions)
                if intent:
                    body = f"意图：{intent}\n\n{body}" if body else f"意图：{intent}"
                verification = payload.get("metadata", {}).get("verification", {})
                detail_parts = []
                if verification.get("screen_changed") is True:
                    detail_parts.append("界面已发生变化")
                elif verification.get("screen_changed") is False:
                    detail_parts.append("界面暂未检测到变化")
                if payload.get("before_ref") or payload.get("after_ref"):
                    detail_parts.append("下方为本步相关截图")
                return {
                    "title": title,
                    "body": body,
                    "detail": " · ".join(detail_parts),
                    "accent": ("#155eef", "#eef4ff"),
                    "status": "已执行" if result == "computer_call_output" else self._status_label_text(result),
                    "image_refs": self._model_request_image_refs_for_trace_event(event),
                }
            if action.get("type") == "agent_tool":
                name = str(action.get("name") or "")
                if result == "rejected":
                    output = payload.get("metadata", {}).get("output", {})
                    reason = ""
                    if isinstance(output, dict):
                        reason = str(output.get("parse_error") or output.get("reason") or "")
                    return {
                        "title": "动作未通过",
                        "body": "这一步没有执行，系统已把问题反馈给模型重新处理。",
                        "detail": self._friendly_reason(reason),
                        "accent": ("#b42318", "#fff0ed"),
                        "status": "已退回",
                        "image_refs": (),
                    }
                body = self._agent_tool_timeline_body(name, payload)
                if intent:
                    body = f"意图：{intent}\n\n{body}" if body else f"意图：{intent}"
                return {
                    "title": self._agent_tool_timeline_title(name),
                    "body": body,
                    "detail": "",
                    "accent": ("#087443", "#eaf7ef"),
                    "status": self._status_label_text(result),
                    "image_refs": (),
                }
            summary = str(payload.get("summary") or "").strip()
            if not summary:
                return None
            return {
                "title": "步骤",
                "body": summary,
                "detail": "",
                "accent": (self.colors["brand"], self.colors["surface_soft"]),
                "status": self._status_label_text(result),
                "image_refs": (),
            }
        if event.kind == "agent_result":
            item_id = str(payload.get("item_id") or "").strip()
            output = str(payload.get("output_text") or "").strip()
            status = self._status_label_text(str(payload.get("status") or ""))
            return {
                "title": f"结果 {item_id}".strip(),
                "body": output or status,
                "detail": str(payload.get("reason") or "").strip(),
                "accent": ("#087443", "#eaf7ef"),
                "status": status,
                "image_refs": tuple(payload.get("artifact_refs") or ()),
            }
        if event.kind == "operation":
            kind = str(payload.get("kind") or "")
            status = str(payload.get("status") or "")
            if kind == "agent_run" and status == "running":
                return {
                    "title": "开始执行",
                    "body": "Agent 已开始处理当前应用任务。",
                    "detail": "",
                    "accent": ("#155eef", "#eef4ff"),
                    "status": "运行中",
                    "image_refs": (),
                }
            if kind == "agent_run" and status == "waiting_human":
                nested = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
                reason = str(nested.get("result_reason") or payload.get("summary") or "")
                return {
                    "title": "等待人工处理",
                    "body": self._friendly_reason(reason) or "Agent 需要你确认下一步。",
                    "detail": "可在下方人工处理卡片中输入处理意见。",
                    "accent": ("#b45f06", "#fff6e5"),
                    "status": "待处理",
                    "image_refs": tuple(payload.get("artifact_refs") or ()),
                }
            title = str(payload.get("title") or "").strip()
            summary = str(payload.get("summary") or "").strip()
            if not title and not summary:
                return None
            return {
                "title": title or "记录",
                "body": summary,
                "detail": "",
                "accent": self._operation_accent(status=status, kind=kind),
                "status": self._status_label_text(status),
                "image_refs": tuple(payload.get("artifact_refs") or ()),
            }
        return None

    def _model_response_event_for_trace_event(self, event: Any) -> Any | None:
        if event.kind == "model_response":
            return event
        payload = event.payload if isinstance(event.payload, dict) else {}
        events = self._events_for_selected_app(self.selected_app_id.get())
        try:
            index = next(i for i, candidate in enumerate(events) if candidate is event)
        except StopIteration:
            return None
        if event.kind == "operation":
            metadata = (
                payload.get("metadata")
                if isinstance(payload.get("metadata"), dict)
                else {}
            )
            nested = (
                payload.get("payload")
                if isinstance(payload.get("payload"), dict)
                else {}
            )
            nested_metadata = (
                nested.get("metadata")
                if isinstance(nested.get("metadata"), dict)
                else {}
            )
            model_response_step = (
                metadata.get("model_response_step")
                or nested.get("model_response_step")
                or nested_metadata.get("model_response_step")
            )
            if model_response_step is None and index + 1 < len(events):
                next_event = events[index + 1]
                next_payload = (
                    next_event.payload if isinstance(next_event.payload, dict) else {}
                )
                next_action = (
                    next_payload.get("action")
                    if isinstance(next_payload.get("action"), dict)
                    else {}
                )
                next_metadata = (
                    next_payload.get("metadata")
                    if isinstance(next_payload.get("metadata"), dict)
                    else {}
                )
                if (
                    next_event.kind == "step"
                    and next_action.get("type") == "agent_tool"
                    and next_action.get("name") == "append_trace_note"
                ):
                    model_response_step = next_metadata.get("model_response_step")
            if model_response_step is not None:
                for candidate in reversed(events[:index]):
                    if candidate.kind != "model_response":
                        continue
                    if candidate.payload.get("step") == model_response_step:
                        return candidate
            return None
        if event.kind != "step":
            return None
        metadata = (
            payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        )
        model_response_step = metadata.get("model_response_step")
        if model_response_step is not None:
            for candidate in reversed(events[:index]):
                if candidate.kind != "model_response":
                    continue
                if candidate.payload.get("step") == model_response_step:
                    return candidate
            return None
        step_id = payload.get("step_id")
        if step_id is not None:
            for candidate in reversed(events[:index]):
                if candidate.kind != "model_response":
                    continue
                if candidate.payload.get("step") == step_id:
                    return candidate
        for candidate in reversed(events[:index]):
            if candidate.kind == "model_response":
                return candidate
        return None

    def _bind_click_tree(self, widget: tk.Widget, command: Any) -> None:
        widget.bind("<Button-1>", lambda _event: command())
        for child in widget.winfo_children():
            if isinstance(child, tk.Widget):
                self._bind_click_tree(child, command)

    def _bind_mousewheel_tree(self, widget: tk.Widget, canvas: tk.Canvas) -> None:
        if isinstance(widget, (tk.Text, tk.Entry, ttk.Entry, ttk.Combobox)):
            return
        widget.bind(
            "<MouseWheel>",
            lambda event, target=canvas: self._scroll_canvas_from_mousewheel(
                event,
                target,
            ),
            add="+",
        )
        widget.bind(
            "<Button-4>",
            lambda event, target=canvas: self._scroll_canvas_from_mousewheel(
                event,
                target,
            ),
            add="+",
        )
        widget.bind(
            "<Button-5>",
            lambda event, target=canvas: self._scroll_canvas_from_mousewheel(
                event,
                target,
            ),
            add="+",
        )
        for child in widget.winfo_children():
            if isinstance(child, tk.Widget):
                self._bind_mousewheel_tree(child, canvas)

    def _scroll_canvas_from_mousewheel(
        self,
        event: tk.Event[Any],
        canvas: tk.Canvas,
    ) -> str:
        if not canvas.winfo_exists():
            return "break"
        event_num = getattr(event, "num", None)
        if event_num == 4:
            units = -3
        elif event_num == 5:
            units = 3
        else:
            delta = int(getattr(event, "delta", 0) or 0)
            if delta == 0:
                return "break"
            units = -max(1, abs(delta) // 120) if delta > 0 else max(1, abs(delta) // 120)
        canvas.yview_scroll(units, "units")
        return "break"

    def _model_intent_text(self, payload: dict[str, Any]) -> str:
        reasoning = str(
            payload.get("reasoning_text") or payload.get("reasoning_text_preview") or ""
        ).strip()
        visible = str(
            payload.get("output_text") or payload.get("output_text_preview") or ""
        ).strip()
        text = self._reasoning_text_for_user(reasoning) or self._strip_tool_markup(visible)
        text = text.replace("Thinking Process:", "").replace("Plan:", "").strip()
        if not text:
            return ""
        return self._clamp_text(text, 220)

    @classmethod
    def _reasoning_text_for_user(cls, text: str) -> str:
        cleaned = cls._strip_reasoning_markup(text)
        if not cleaned:
            return ""
        lines: list[str] = []
        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lowered = line.lower()
            if lowered.startswith("current think mode:"):
                continue
            if line.startswith("当前思考模式：") or line.startswith("当前思考模式:"):
                continue
            if line.startswith("思考格式：") or line.startswith("思考格式:"):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    @staticmethod
    def _strip_tool_markup(text: str) -> str:
        if not text:
            return ""
        text = GuiAgentDesktopClient._strip_reasoning_markup(text)
        text = re.sub(
            r"<tool_call\b.*",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(
            r"<tool_code\b.*",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        cleaned = re.sub(
            r"<tool_call>.*?</tool_call>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        cleaned = re.sub(
            r"<tool_code>.*?</tool_code>",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return cleaned.strip()

    @staticmethod
    def _strip_reasoning_markup(text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(
            r"<think>\s*.*?</think>",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _computer_action_timeline_text(
        self,
        actions: list[dict[str, Any]],
    ) -> tuple[str, str]:
        if not actions:
            return "执行动作", "模型请求执行一步电脑操作。"
        types = {str(action.get("type") or "") for action in actions}
        if types <= {"click", "double_click", "move"}:
            title = "点击"
        elif "type" in types and ("click" in types or "move" in types):
            title = "定位并输入"
        elif "submit_text" in types:
            title = "提交文本"
        elif types == {"type"}:
            title = "输入文本"
        elif "keypress" in types:
            title = "按键"
        elif "scroll" in types:
            title = "滚动"
        elif "wait" in types:
            title = "等待"
        else:
            title = "执行动作"
        lines = [format_computer_action(action) for action in actions[:4]]
        if len(actions) > 4:
            lines.append(f"另有 {len(actions) - 4} 个动作")
        return title, "\n".join(lines)

    def _agent_tool_timeline_title(self, name: str) -> str:
        return {
            "read_text_file": "读取清单",
            "write_text_file": "写入文件",
            "append_text_file": "追加文件",
            "append_trace_note": "保存结果",
            "manage_window_focus": "记录视野",
            "open_url": "打开网页",
            "launch_app": "启动应用",
            "run_shell": "执行命令",
            "run_action_macro": "执行快捷操作",
            "propose_action_macro": "建议操作宏",
            "navigate_to_target": "移动到目标",
            "read_image": "查看图片",
            "read_video_clip": "查看视频",
            "request_human_help": "请求人工处理",
            "finish_subtask": "完成任务",
        }.get(name, "处理信息")

    def _agent_tool_timeline_body(self, name: str, payload: dict[str, Any]) -> str:
        output = payload.get("metadata", {}).get("output", {})
        if not isinstance(output, dict):
            output = {}
        if name == "read_text_file":
            chars = output.get("chars")
            path = Path(str(output.get("path") or "")).name
            if chars:
                return f"已读取文件 {chars} 字。{path}".strip()
            return f"已读取文件。{path}".strip()
        if name == "append_trace_note":
            result = output.get("agent_result") if isinstance(output.get("agent_result"), dict) else {}
            item_id = str(result.get("item_id") or "").strip()
            status = self._status_label_text(str(result.get("status") or output.get("status") or ""))
            return f"{item_id} {status}".strip() or "结果已写入轨迹。"
        if name in {"write_text_file", "append_text_file"}:
            chars = output.get("chars")
            path = Path(str(output.get("path") or "")).name
            verb = "写入" if name == "write_text_file" else "追加"
            if chars:
                return f"{verb}文件 {chars} 字。{path}".strip()
            return f"{verb}文件。{path}".strip()
        if name == "open_url":
            url = str(output.get("url") or "").strip()
            return f"已打开网页：{url}" if url else "已打开网页。"
        if name == "launch_app":
            pid = output.get("pid")
            return f"已启动应用，进程 {pid}。" if pid else "已启动应用。"
        if name == "run_shell":
            returncode = output.get("returncode")
            if returncode is not None:
                return f"命令已结束，退出码 {returncode}。"
            return "命令已执行。"
        if name == "propose_action_macro":
            proposal = output.get("macro_proposal")
            if not isinstance(proposal, dict):
                proposal = {}
            macro_name = str(proposal.get("macro_name") or "").strip()
            description = str(proposal.get("description") or "").strip()
            suffix = f"：{macro_name}" if macro_name else ""
            if description:
                return f"宏建议已提交，等待人工批准{suffix}。\n{description}"
            return f"宏建议已提交，等待人工批准{suffix}。"
        if name == "request_human_help":
            return "Agent 需要人工确认后再继续。"
        return self._step_summary(payload)

    def _step_image_refs(self, payload: dict[str, Any]) -> tuple[str, ...]:
        refs = []
        before = str(payload.get("before_ref") or "").strip()
        after = str(payload.get("after_ref") or "").strip()
        if before:
            refs.append(before)
        if after and after != before:
            refs.append(after)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        local_refinements = metadata.get("local_refinements")
        if isinstance(local_refinements, list):
            for item in local_refinements:
                if not isinstance(item, dict):
                    continue
                ref = str(item.get("image_ref") or item.get("image_url") or "").strip()
                if ref:
                    refs.append(ref)
        return tuple(refs)

    def _model_request_image_refs_for_trace_event(self, event: Any) -> tuple[str, ...]:
        return tuple(ref for _label, ref in self._model_request_image_ref_pairs(event))

    def _model_request_image_ref_pairs(self, event: Any) -> tuple[tuple[str, str], ...]:
        model_event = self._model_response_event_for_trace_event(event)
        model_payload = (
            model_event.payload
            if model_event is not None and isinstance(model_event.payload, dict)
            else {}
        )
        request_context = model_payload.get("request_context")
        latest_context = (
            self._latest_request_context_items(request_context)
            if isinstance(request_context, list)
            else []
        )
        refs: list[tuple[str, str]] = []
        seen: set[str] = set()

        def add(label: str, ref: str) -> None:
            ref = str(ref or "").strip()
            if not ref or ref in seen:
                return
            seen.add(ref)
            refs.append((label, ref))

        for item in latest_context:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type == "computer_call_output":
                call_id = str(item.get("call_id") or "").strip()
                for label, ref in self._computer_call_output_image_ref_pairs(call_id):
                    add(label, ref)
                continue
            if item_type != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            image_seen = 0
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "input_image":
                    continue
                ref = self._actual_image_ref_from_value(part.get("image_url"))
                if ref:
                    image_seen += 1
                    add(f"输入图片 {image_seen}", ref)

        if not refs and event.kind == "step":
            payload = event.payload if isinstance(event.payload, dict) else {}
            before = str(payload.get("before_ref") or "").strip()
            if before:
                add("输入截图", before)
        return tuple(refs)

    def _computer_call_output_image_ref_pairs(self, call_id: str) -> tuple[tuple[str, str], ...]:
        if not call_id:
            return ()
        for event in reversed(self._events_for_selected_app(self.selected_app_id.get())):
            if event.kind != "step":
                continue
            payload = event.payload if isinstance(event.payload, dict) else {}
            action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
            if action.get("type") != "computer_call" or action.get("call_id") != call_id:
                continue
            refs: list[tuple[str, str]] = []
            after = str(payload.get("after_ref") or "").strip()
            if after:
                refs.append(("上一轮工具执行后观察截图", after))
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            local_refinements = metadata.get("local_refinements")
            if isinstance(local_refinements, list):
                for index, item in enumerate(local_refinements, start=1):
                    if not isinstance(item, dict):
                        continue
                    ref = str(item.get("image_ref") or item.get("image_url") or "").strip()
                    if not ref:
                        continue
                    click_index = item.get("click_index") or index
                    refs.append((f"点击局部图 {click_index}", ref))
            return tuple(refs)
        return ()

    @staticmethod
    def _actual_image_ref_from_value(value: Any) -> str:
        if isinstance(value, str):
            if value.startswith("data:image") or value.startswith("file:"):
                return value
            return ""
        if isinstance(value, dict):
            for key in ("data_url", "url", "image_url", "path"):
                ref = str(value.get(key) or "").strip()
                if ref and ref != str(value.get("placeholder") or ""):
                    return ref
        return ""

    def _timeline_thumbnail(self, image_ref: str) -> Any | None:
        if Image is None or ImageTk is None:
            return None
        try:
            image = self._load_image_ref(image_ref)
        except Exception:
            return None
        if image is None:
            return None
        image = image.convert("RGB")
        image.thumbnail((142, 82))
        return ImageTk.PhotoImage(image)

    def _load_image_ref(self, image_ref: str) -> Any | None:
        if not image_ref:
            return None
        if image_ref.startswith("data:image/"):
            _, encoded = image_ref.split(",", 1)
            return Image.open(BytesIO(base64.b64decode(encoded)))
        if image_ref.startswith("file:"):
            parsed = urlparse(image_ref)
            path = Path(unquote(parsed.path))
            if sys.platform == "win32" and str(path).startswith("\\"):
                path = Path(str(path).lstrip("\\"))
            return Image.open(path)
        path = Path(image_ref)
        if path.exists():
            return Image.open(path)
        return None

    def _human_request_user_text(self, item: dict[str, Any]) -> str:
        request = item["request"]
        metadata = request.get("metadata") if isinstance(request.get("metadata"), dict) else {}
        proposal = metadata.get("macro_proposal") if isinstance(metadata, dict) else None
        blocking = bool(request.get("blocking", metadata.get("blocking", True)))
        lines = [str(request.get("question") or "当前任务需要你确认。")]
        lines.append("类型：阻塞人工介入" if blocking else "类型：非阻塞人工介入")
        if isinstance(proposal, dict):
            macro_name = str(proposal.get("macro_name") or "").strip()
            description = str(proposal.get("description") or "").strip()
            dynamic = proposal.get("dynamic_parameters") or []
            lines.append(f"建议宏：{macro_name}" if macro_name else "建议宏")
            if description:
                lines.append(f"用途：{description}")
            if dynamic:
                lines.append("动态参数：" + "、".join(str(item) for item in dynamic))
            for step in proposal.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                index = step.get("index")
                purpose = str(step.get("purpose") or "").strip()
                if purpose:
                    lines.append(f"{index}. {purpose}" if index else purpose)
        reason = self._friendly_reason(str(request.get("risk_reason") or ""))
        if reason:
            lines.append(f"原因：{reason}")
        proposed = str(request.get("proposed_action") or "").strip()
        if proposed:
            lines.append(f"建议：{self._friendly_reason(proposed)}")
        reply = item.get("reply")
        if reply:
            lines.append(f"你的回复：{reply.get('text') or ''}")
        return "\n".join(lines)

    def _human_macro_preview_refs(self, item: dict[str, Any]) -> tuple[tuple[str, str], ...]:
        proposal = self._human_macro_proposal(item)
        if not isinstance(proposal, dict):
            return ()
        refs: list[tuple[str, str]] = []
        for step in proposal.get("steps") or []:
            if not isinstance(step, dict):
                continue
            preview = step.get("click_preview")
            if not isinstance(preview, dict):
                continue
            ref = str(preview.get("image_url") or "").strip()
            if not ref:
                continue
            label = str(step.get("purpose") or f"步骤 {step.get('index') or ''}").strip()
            refs.append((label, ref))
        return tuple(refs)

    def _human_macro_proposal(self, item: dict[str, Any]) -> dict[str, Any] | None:
        request = item.get("request") if isinstance(item, dict) else {}
        metadata = request.get("metadata") if isinstance(request, dict) else {}
        proposal = metadata.get("macro_proposal") if isinstance(metadata, dict) else None
        if not isinstance(proposal, dict):
            return None
        return proposal

    def _approve_human_macro_request(self, item: dict[str, Any]) -> None:
        request_id = item["request"]["request_id"]
        try:
            macro = self.state.approve_macro_request(request_id)
            self.state.human_loop.claim_request(request_id, source="client")
            self.state.submit_structured_human_reply(
                request_id=request_id,
                manager_input=f"批准并启用操作宏：{macro.name}",
                decision_type="宏审批",
                direct_command=DIRECT_COMMAND_NONE,
            )
        except Exception as exc:
            messagebox.showerror("启用宏失败", str(exc))
            self.refresh_all()
            return
        self.status_text.set(f"已启用操作宏：{macro.name}")
        self.refresh_all()
        self._save_state()

    def _submit_human_card_reply(self, item: dict[str, Any], widget: tk.Text) -> None:
        text = widget.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("缺少输入", "请填写处理意见。")
            return
        request_id = item["request"]["request_id"]
        try:
            self.state.human_loop.claim_request(request_id, source="client")
            self.state.submit_structured_human_reply(
                request_id=request_id,
                manager_input=text,
                decision_type="人工回复",
                direct_command=DIRECT_COMMAND_NONE,
            )
        except Exception as exc:
            messagebox.showerror("提交失败", f"{exc}\n\n该请求可能已经被其他入口处理。")
            self.refresh_all()
            return
        self.status_text.set("人工处理已提交")
        self.refresh_all()

    @staticmethod
    def _friendly_reason(text: str) -> str:
        replacements = {
            "Harness requested human input after repeated computer actions.": "连续操作没有带来有效进展，需要你确认下一步。",
            "The same computer action repeated without useful progress.": "同一个操作反复执行但没有进展。",
            "Human should unblock the current UI state.": "请根据当前界面给出下一步处理意见。",
            "repeated_action": "重复操作",
        }
        result = text.strip()
        for source, target in replacements.items():
            result = result.replace(source, target)
        return result.strip()

    @staticmethod
    def _clamp_text(text: str, limit: int) -> str:
        text = " ".join(line.strip() for line in text.splitlines() if line.strip())
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def _operation_icon(self, kind: str) -> str:
        return {
            "model_response": "M",
            "step": "S",
            "computer_call": "C",
            "agent_tool": "T",
            "read_text_file": "R",
            "append_trace_note": "W",
            "computer_use": "C",
            "mouse": "↖",
            "keyboard": "⌨",
            "switch": "⇄",
            "observe": "◉",
            "model_intent": "◇",
            "harness_check": "✓",
            "verification": "✓",
            "human": "!",
            "failure": "!",
            "agent_run": "W",
            "app_configured": "W",
            "agent_result": "R",
            "viewport": "▣",
        }.get(kind, "T")

    def _operation_title(self, event_kind: str, payload: dict[str, Any]) -> str:
        if event_kind == "model_response":
            return "模型响应"
        if event_kind == "step":
            action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
            action_type = str(action.get("type") or "")
            if action_type == "computer_call":
                return "电脑操作"
            if action_type == "agent_tool":
                return self._tool_display_name(str(action.get("name") or "agent_tool"))
            return "执行步骤"
        if event_kind == "app_configured":
            return "应用已配置"
        if event_kind == "agent_result":
            item_id = str(payload.get("item_id") or "").strip()
            return f"记录结果：{item_id}" if item_id else "记录结果"
        if event_kind == "operation":
            title = str(payload.get("title") or "").strip()
            if title:
                return title
        return self._trace_kind_label(event_kind)

    def _operation_summary(self, event_kind: str, payload: dict[str, Any]) -> str:
        if event_kind == "model_response":
            parts = []
            latency = payload.get("latency_seconds")
            if isinstance(latency, (int, float)):
                parts.append(f"响应 {latency:.1f}s")
            tool_count = payload.get("function_call_count")
            if isinstance(tool_count, int):
                parts.append(f"工具 {tool_count} 次")
            text_chars = payload.get("output_text_chars")
            if isinstance(text_chars, int) and text_chars:
                parts.append(f"输出 {text_chars} 字")
            warnings = payload.get("warnings")
            if warnings:
                parts.append("关注：" + self._warning_summary(warnings))
            return " / ".join(parts) or "模型返回已记录"
        if event_kind == "step":
            return self._step_summary(payload)
        if event_kind == "app_configured":
            trace_name = Path(str(payload.get("trace_path") or "")).name
            return (
                f"{payload.get('app_name', '')} / "
                f"{payload.get('item_count', 0)} 条 / "
                f"轨迹 {trace_name}"
            )
        if event_kind == "agent_result":
            status = self._status_label_text(str(payload.get("status") or ""))
            output = str(payload.get("output_text") or "").strip()
            if output:
                return f"{status} / {output[:96]}"
            return status
        if event_kind == "operation":
            summary = str(payload.get("summary") or "").strip()
            if summary:
                return summary
        return short_payload(payload)

    def _step_summary(self, payload: dict[str, Any]) -> str:
        action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
        result = str(payload.get("result") or payload.get("status") or "")
        status = self._status_label_text(result) if result else ""
        if action.get("type") == "computer_call":
            actions = action.get("actions")
            count = len(actions) if isinstance(actions, list) else 0
            verification = payload.get("metadata", {}).get("verification", {})
            changed = verification.get("screen_changed")
            suffix = ""
            if changed is True:
                suffix = "，界面有变化"
            elif changed is False:
                suffix = "，界面未变化"
            return f"执行 {count} 个电脑动作{suffix}"
        if action.get("type") == "agent_tool":
            name = str(action.get("name") or "")
            if result == "rejected":
                output = payload.get("metadata", {}).get("output", {})
                reason = str(output.get("parse_error") or output.get("reason") or "")
                return f"工具参数被拒绝，已反馈给模型重试。{reason[:72]}"
            if name == "read_text_file":
                output = payload.get("metadata", {}).get("output", {})
                chars = output.get("chars")
                return f"读取文件 {chars} 字" if chars else "读取文件"
            if name == "append_trace_note":
                output = payload.get("metadata", {}).get("output", {})
                agent_result = output.get("agent_result", {})
                item_id = agent_result.get("item_id") or ""
                return f"保存记录结果 {item_id}".strip()
            return f"{self._tool_display_name(name)} / {status or '已处理'}"
        summary = str(payload.get("summary") or "").strip()
        return summary or status or short_payload(payload)

    @staticmethod
    def _tool_display_name(name: str) -> str:
        return {
            "computer_use": "电脑操作",
            "read_text_file": "读取文件",
            "read_image": "读取图片",
            "read_video_frames": "读取视频帧",
            "append_trace_note": "记录结果",
            "manage_window_focus": "设置视野",
            "request_human_help": "请求人工协助",
            "report_blocked": "报告阻塞",
            "finish_subtask": "结束子任务",
        }.get(name, name or "工具调用")

    @staticmethod
    def _warning_summary(warnings: Any) -> str:
        if not isinstance(warnings, list):
            return str(warnings)
        labels = {
            "long_output_text": "输出偏长",
            "large_response_json": "响应体偏大",
            "slow_model_response": "响应偏慢",
            "thinking_text_visible": "显式思考",
            "long_reasoning_text": "思考偏长",
        }
        return "、".join(labels.get(str(item), str(item)) for item in warnings[:3])

    def _operation_accent(self, *, status: str, kind: str) -> tuple[str, str]:
        if status in {"failed", "error", "rejected"} or kind == "failure":
            return ("#b42318", "#fff0ed")
        if status in {
            "waiting",
            "waiting_human",
            "pending",
            "macro_approval_requested",
        } or kind == "human":
            return ("#b45f06", "#fff6e5")
        if status in {"verified", "completed", "passed", "accepted"}:
            return ("#067647", "#eaf7ef")
        return (self.colors["brand"], self.colors["surface_soft"])

    def _status_label_text(self, status: str) -> str:
        return {
            "configured": "已配置",
            "running": "运行中",
            "recorded": "已记录",
            "planned": "已计划",
            "accepted": "已接受",
            "macro_approval_requested": "待批准",
            "rejected": "已退回",
            "executed": "已执行",
            "computer_call_output": "已执行",
            "verified": "已验证",
            "completed": "已完成",
            "passed": "通过",
            "failed": "失败",
            "error": "错误",
            "pending": "等待",
            "waiting": "等待",
            "waiting_human": "等人工",
        }.get(status, status)

    def _role_color(self, role: str) -> str:
        return {
            "human": self.colors["accent_soft"],
            "asset": "#eef8f2",
            "trace": "#f7f5f2",
            "system": "#f7f5f2",
        }.get(role, "#f7f5f2")

    @staticmethod
    def _task_brief(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return "尚未填写任务定义。"
        keep_prefixes = ("目标", "目标网址", "输入文件", "计划数量", "正确性优先")
        selected: list[str] = []
        for line in lines:
            if line.startswith(keep_prefixes):
                if line.startswith("输入文件"):
                    label, _, value = line.partition("：")
                    selected.append(f"{label}：{Path(value).name if value else value}")
                else:
                    selected.append(line)
            if len(selected) >= 4:
                break
        if not selected:
            selected = lines[:4]
        brief = "\n".join(selected)
        if len(brief) > 360:
            brief = brief[:357].rstrip() + "..."
        return brief

    def _add_app_card_legacy(
        self,
        *,
        app_id: str,
        config: TargetAppConfig,
        waiting: int,
    ) -> None:
        selected = app_id == self.selected_app_id.get()
        bg = "#ece7e2" if selected else self.colors["sidebar"]
        card = tk.Frame(
            self.app_list_frame,
            bg=bg,
            padx=10,
            pady=8,
            highlightthickness=0,
            borderwidth=0,
        )
        card.pack(fill="x", pady=2)
        title = tk.Label(
            card,
            text=config.app_name,
            bg=bg,
            fg=self.colors["ink"],
            anchor="w",
            font=("Microsoft YaHei UI", 10, "bold" if selected else "normal"),
        )
        title.grid(row=0, column=0, sticky="ew")
        subtitle_text = "待处理 " + str(waiting) if waiting else "就绪"
        subtitle = tk.Label(
            card,
            text=subtitle_text,
            bg=bg,
            fg=self.colors["accent"] if waiting else self.colors["muted"],
            anchor="w",
            font=("Microsoft YaHei UI", 8),
        )
        subtitle.grid(row=1, column=0, sticky="ew", pady=(3, 0))
        card.columnconfigure(0, weight=1)
        for widget in (card, title, subtitle):
            widget.bind("<Button-1>", lambda _event, value=app_id: self._select_app(value))
            widget.bind("<Button-3>", lambda event, value=app_id: self._show_app_menu(event, value))

    def _add_app_card(
        self,
        *,
        app_id: str,
        config: TargetAppConfig,
        subtitle: str,
    ) -> None:
        selected = app_id == self.selected_app_id.get()
        card = tk.Canvas(
            self.app_list_frame,
            height=62,
            bg=self.colors["sidebar"],
            highlightthickness=0,
            borderwidth=0,
        )
        card.pack(fill="x", pady=1)

        def redraw(_event: tk.Event[Any] | None = None) -> None:
            width = max(card.winfo_width(), 220)
            card.delete("all")
            fill = self.colors["sidebar_selected"] if selected else self.colors["sidebar"]
            outline = self.colors["line"] if selected else self.colors["sidebar"]
            draw_rounded_rect(
                card,
                4,
                3,
                width - 4,
                59,
                14,
                fill=fill,
                outline=outline,
                width=1,
            )
            if selected:
                draw_rounded_rect(
                    card,
                    9,
                    20,
                    13,
                    42,
                    2,
                    fill=self.colors["ink"],
                    outline=self.colors["ink"],
                )
            title_font = tkfont.Font(
                family="Microsoft YaHei UI",
                size=10,
                weight="bold" if selected else "normal",
            )
            hint_font = tkfont.Font(family="Microsoft YaHei UI", size=8)
            text_x = 22 if selected else 18
            card.create_text(
                text_x,
                22,
                text=config.app_name,
                anchor="w",
                fill=self.colors["ink"],
                font=title_font,
            )
            card.create_text(
                text_x,
                42,
                text=subtitle,
                anchor="w",
                fill=self.colors["accent"] if subtitle != "就绪" else self.colors["muted"],
                font=hint_font,
            )

        card.bind("<Configure>", redraw)
        card.bind("<Button-1>", lambda _event, value=app_id: self._select_app(value))
        card.bind("<Button-3>", lambda event, value=app_id: self._show_app_menu(event, value))
        redraw()

    def _add_inspector_card_legacy(
        self,
        *,
        item_id: str,
        kind: str,
        title: str,
        summary: str,
    ) -> None:
        card = tk.Frame(
            self.inspector_list_frame,
            bg=self.colors["surface_soft"],
            padx=10,
            pady=8,
            highlightthickness=1,
            highlightbackground="#edf1f7",
        )
        card.pack(fill="x", pady=(0, 8))
        top = tk.Frame(card, bg=self.colors["surface_soft"])
        top.pack(fill="x")
        tk.Label(
            top,
            text=kind,
            bg=self.colors["brand_soft"],
            fg=self.colors["brand"],
            font=("Microsoft YaHei UI", 8, "bold"),
            padx=6,
            pady=2,
        ).pack(side="left")
        tk.Label(
            top,
            text=title,
            bg=self.colors["surface_soft"],
            fg=self.colors["ink"],
            font=("Microsoft YaHei UI", 9, "bold"),
            padx=8,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        tk.Label(
            card,
            text=summary,
            bg=self.colors["surface_soft"],
            fg=self.colors["muted"],
            font=("Microsoft YaHei UI", 9),
            wraplength=290,
            justify="left",
            anchor="w",
        ).pack(fill="x", pady=(6, 0))
        for widget in (card, top):
            widget.bind("<Button-1>", lambda _event, value=item_id: self._select_inspector_item(value))
        for child in card.winfo_children():
            child.bind("<Button-1>", lambda _event, value=item_id: self._select_inspector_item(value))

    def _add_inspector_card(
        self,
        *,
        item_id: str,
        kind: str,
        title: str,
        summary: str,
    ) -> None:
        card = tk.Canvas(
            self.inspector_list_frame,
            height=76,
            bg=self.colors["surface"],
            highlightthickness=0,
            borderwidth=0,
        )
        card.pack(fill="x", pady=(0, 7))

        def redraw(_event: tk.Event[Any] | None = None) -> None:
            width = max(card.winfo_width(), 280)
            card.delete("all")
            draw_rounded_rect(
                card,
                2,
                2,
                width - 2,
                74,
                14,
                fill="#fbfbfa",
                outline=self.colors["line_soft"],
                width=1,
            )
            draw_rounded_rect(
                card,
                12,
                12,
                58,
                34,
                10,
                fill=self.colors["brand_soft"],
                outline=self.colors["brand_soft"],
            )
            card.create_text(
                35,
                23,
                text=kind,
                anchor="center",
                fill=self.colors["brand"],
                font=tkfont.Font(family="Microsoft YaHei UI", size=8, weight="bold"),
            )
            card.create_text(
                84,
                23,
                text=title,
                anchor="w",
                width=width - 100,
                fill=self.colors["ink"],
                font=tkfont.Font(family="Microsoft YaHei UI", size=9, weight="bold"),
            )
            card.create_text(
                14,
                45,
                text=summary,
                anchor="nw",
                width=width - 28,
                fill=self.colors["muted"],
                font=tkfont.Font(family="Microsoft YaHei UI", size=9),
            )

        card.bind("<Configure>", redraw)
        card.bind("<Button-1>", lambda _event, value=item_id: self._select_inspector_item(value))
        self._bind_mousewheel_tree(card, self.inspector_canvas)
        redraw()

    def _on_messages_configure(self, _event: tk.Event[Any]) -> None:
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event[Any]) -> None:
        self.chat_canvas.itemconfigure(self.messages_window, width=event.width)

    def _scroll_messages_to_bottom(self) -> None:
        self.chat_canvas.yview_moveto(1.0)

    def _select_app(self, app_id: str) -> None:
        self.selected_app_id.set(app_id)
        self.state.selected_app_id = app_id
        self._load_selected_trace_if_available(app_id)
        self._selected_trace_event = None
        self.refresh_apps()
        self.refresh_header()
        self.refresh_messages()
        self.refresh_inspector()
        self.refresh_composer()
        self._save_state()

    def _show_create_menu(self) -> None:
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="新建 APP", command=self._add_app_dialog)
        menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())

    def _show_app_menu(self, event: tk.Event[Any], app_id: str | None = None) -> None:
        if app_id is not None:
            self.selected_app_id.set(app_id)
            self.state.selected_app_id = app_id
            self._load_selected_trace_if_available(app_id)
            self.refresh_apps()
            self._save_state()
        menu = tk.Menu(self, tearoff=0)
        latest_status = self._latest_agent_run_status(self.selected_app_id.get())
        menu.add_command(
            label="开始执行",
            command=self._start_selected_agent_run,
            state=(
                "disabled"
                if self._is_terminal_agent_status(latest_status)
                else "normal"
            ),
        )
        menu.add_separator()
        menu.add_command(label="查看应用设定", command=self._show_selected_task_definition)
        menu.add_command(label="编辑应用设定", command=self._edit_selected_app)
        menu.add_separator()
        menu.add_command(label="标定 APP 视野", command=self._open_viewport_marker)
        menu.add_command(label="隐藏视野框", command=self._hide_viewport_marker)
        menu.add_command(label="复制视野坐标", command=self._copy_selected_viewport)
        menu.add_command(label="清除视野标定", command=self._clear_selected_viewport)
        menu.add_separator()
        menu.add_command(label="执行记录", command=self._show_selected_trace)
        menu.add_command(label="复制轨迹文件路径", command=self._copy_selected_trace_path)
        menu.add_command(label="打开轨迹文件目录", command=self._open_selected_trace_folder)
        menu.add_command(label="复制 APP ID", command=self._copy_selected_app_id)
        menu.add_separator()
        menu.add_command(label="删除会话", command=self._delete_selected_app)
        menu.add_separator()
        menu.add_command(label="打开设置", command=self._open_settings)
        menu.tk_popup(event.x_root, event.y_root)

    def _clear_search_placeholder(self, _event: tk.Event[Any]) -> None:
        if self.search_entry.get() == "搜索 APP":
            self.search_entry.delete(0, tk.END)

    def _add_app_dialog(self) -> None:
        dialog = AppConfigDialog(self, title="新建 APP")
        result = dialog.result
        if result is None:
            return
        app_name = result["app_name"].strip()
        job_id = result["job_id"].strip()
        if not app_name or not job_id:
            messagebox.showwarning("输入不完整", "APP 名称和内部 ID 都不能为空。")
            return
        app_id = unique_app_id(app_name, self.state.target_apps)
        config = TargetAppConfig(
            app_id=app_id,
            app_name=app_name,
            job_id=job_id,
            task_description=result["task_description"].strip(),
            reference_assets=tuple(result["assets"]),
        )
        self.state.add_target_app(config)
        if job_id not in self.state.job_board.jobs:
            self.state.job_board.add_job(
                JobState(job_id=job_id, target_app=app_name, goal=app_name)
            )
        self.selected_app_id.set(app_id)
        self.state.selected_app_id = app_id
        self.status_text.set(f"已添加：{app_name}")
        self.refresh_all()
        self._save_state()

    def _edit_selected_app(self) -> None:
        config = self._selected_config()
        if config is None:
            messagebox.showinfo("没有 APP", "请先新建或选择一个 APP。")
            return
        dialog = AppConfigDialog(self, title="应用设定", config=config)
        result = dialog.result
        if result is None:
            return
        updated = TargetAppConfig(
            app_id=config.app_id,
            app_name=result["app_name"].strip() or config.app_name,
            job_id=result["job_id"].strip() or config.job_id,
            task_description=result["task_description"].strip(),
            reference_assets=tuple(result["assets"]),
            metadata=config.metadata,
        )
        self.state.target_apps[config.app_id] = updated
        if updated.job_id not in self.state.job_board.jobs:
            self.state.job_board.add_job(
                JobState(
                    job_id=updated.job_id,
                    target_app=updated.app_name,
                    goal=updated.app_name,
                )
            )
        self.status_text.set("应用设定已更新")
        self.refresh_all()
        self._save_state()

    def _delete_selected_app(self) -> None:
        config = self._selected_config()
        if config is None:
            return
        if self.agent_running.get() and config.app_id == self.selected_app_id.get():
            messagebox.showinfo("正在运行", "当前会话正在执行，请先停止后再删除。")
            return
        if not messagebox.askyesno(
            "删除会话",
            f"删除“{config.app_name}”？\n\n这只会移除客户端里的会话配置和内存记录，不会删除磁盘上的轨迹文件。",
        ):
            return
        removed = self.state.remove_target_app(config.app_id)
        self.selected_app_id.set(self.state.selected_app_id)
        self._selected_trace_event = None
        self.status_text.set(f"已删除：{removed.app_name}")
        self.refresh_all()
        self._save_state()

    def _show_selected_task_definition(self) -> None:
        config = self._selected_config()
        if config is None:
            messagebox.showinfo("没有 APP", "请先新建或选择一个 APP。")
            return
        details = [
            f"APP：{config.app_name}",
            f"APP ID：{config.app_id}",
            f"JOB：{config.job_id}",
        ]
        viewport = (config.metadata or {}).get("manual_viewport")
        if isinstance(viewport, dict):
            details.append(
                "视野："
                f"x={viewport.get('x')}, y={viewport.get('y')}, "
                f"{viewport.get('width')}x{viewport.get('height')}"
            )
        details.extend(["", config.task_description or "尚未填写任务定义。"])
        if config.reference_assets:
            details.extend(["", "素材引用："])
            details.extend(
                f"- {asset.citation}  {asset.title or Path(asset.path).name}"
                for asset in config.reference_assets
            )
        self._show_inspector_detail("\n".join(details))
        if not self.inspector_visible.get():
            self._toggle_inspector()

    def _show_selected_trace(self) -> None:
        config = self._selected_config()
        if config is None:
            messagebox.showinfo("没有 APP", "请先新建或选择一个 APP。")
            return
        self.refresh_inspector()
        events = self._events_for_selected_app(config.app_id)
        if events:
            self._select_trace_event(events[-1])
        else:
            self._show_inspector_detail("还没有操作记录。")
            if not self.inspector_visible.get():
                self._toggle_inspector()

    def _copy_selected_app_id(self) -> None:
        config = self._selected_config()
        if config is None:
            return
        self.clipboard_clear()
        self.clipboard_append(config.app_id)
        self.status_text.set("已复制 APP ID")

    def _copy_selected_trace_path(self) -> None:
        path = self._trace_path_for_selected_app()
        if not path:
            self.status_text.set("当前 APP 未配置轨迹文件")
            return
        self.clipboard_clear()
        self.clipboard_append(path)
        self.status_text.set("已复制轨迹文件路径")

    def _open_selected_trace_folder(self) -> None:
        path = self._trace_path_for_selected_app()
        if not path:
            self.status_text.set("当前 APP 未配置轨迹文件")
            return
        folder = Path(path).expanduser().parent
        if not folder.exists():
            self.status_text.set("轨迹文件目录不存在")
            return
        try:
            import os

            os.startfile(folder)  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("打开失败", str(exc))

    def _open_viewport_marker(self) -> None:
        config = self._selected_config()
        if config is None:
            messagebox.showinfo("没有 APP", "请先新建或选择一个 APP。")
            return
        if self.viewport_marker is not None and self.viewport_marker.winfo_exists():
            self.viewport_marker.destroy()
        box = self._initial_viewport_box(config)
        self.viewport_marker = ViewportMarkerWindow(
            self,
            app_id=config.app_id,
            app_name=config.app_name,
            initial_box=box,
        )
        self.status_text.set("拖动视野框到目标 APP 区域，保存后隐藏。")

    def _hide_viewport_marker(self) -> None:
        if self.viewport_marker is not None and self.viewport_marker.winfo_exists():
            self.viewport_marker.destroy()
        self.viewport_marker = None
        self.status_text.set("视野框已隐藏")

    def _save_manual_viewport_from_marker(
        self,
        *,
        app_id: str,
        box: tuple[int, int, int, int],
    ) -> None:
        x, y, width, height = box
        try:
            viewport = self.state.update_target_viewport(
                app_id,
                x=x,
                y=y,
                width=width,
                height=height,
            )
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self.state.record_operation_summary(
            app_id=app_id,
            kind="viewport",
            title="APP 视野已标定",
            summary=(
                f"x={viewport['x']}, y={viewport['y']}, "
                f"{viewport['width']}x{viewport['height']}"
            ),
            status="configured",
            payload={"manual_viewport": viewport},
        )
        self._hide_viewport_marker()
        self.refresh_all()
        self._save_state()

    def _clear_selected_viewport(self) -> None:
        config = self._selected_config()
        if config is None:
            return
        self.state.clear_target_viewport(config.app_id)
        self.state.record_operation_summary(
            app_id=config.app_id,
            kind="viewport",
            title="APP 视野标定已清除",
            summary="后续将重新使用完整屏幕或运行时视野。",
            status="recorded",
        )
        self.status_text.set("已清除视野标定")
        self.refresh_all()
        self._save_state()

    def _copy_selected_viewport(self) -> None:
        config = self._selected_config()
        viewport = (config.metadata or {}).get("manual_viewport") if config else None
        if not isinstance(viewport, dict):
            self.status_text.set("当前 APP 尚未标定视野")
            return
        try:
            text = (
                f"{int(viewport['x'])},"
                f"{int(viewport['y'])},"
                f"{int(viewport['width'])},"
                f"{int(viewport['height'])}"
            )
        except (KeyError, TypeError, ValueError):
            self.status_text.set("当前 APP 的视野坐标无效")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_text.set("已复制视野坐标")

    def _initial_viewport_box(self, config: TargetAppConfig) -> tuple[int, int, int, int]:
        saved = (config.metadata or {}).get("manual_viewport")
        if isinstance(saved, dict):
            try:
                x = int(saved.get("x", 0))
                y = int(saved.get("y", 0))
                width = int(saved.get("width", 0))
                height = int(saved.get("height", 0))
                if width > 0 and height > 0:
                    return (x, y, width, height)
            except (TypeError, ValueError):
                pass
        screen_width = max(1, self.winfo_screenwidth())
        screen_height = max(1, self.winfo_screenheight())
        width = min(1100, max(640, int(screen_width * 0.62)))
        height = min(720, max(420, int(screen_height * 0.62)))
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)
        return (x, y, width, height)

    def _trace_path_for_selected_app(self) -> str:
        config = self._selected_config()
        if config is None:
            return ""
        return str((config.metadata or {}).get("trace_path") or "")

    def _open_settings(self) -> None:
        SettingsDialog(self, self.state, self.selected_app_id.get())
        self.refresh_all()
        self._save_state()

    def _open_session_template_dialog(self) -> None:
        self._add_app_dialog()

    def _toggle_inspector(self) -> None:
        if self.inspector_visible.get():
            self.inspector_resize_handle.grid_remove()
            self.inspector.grid_remove()
            self.columnconfigure(2, minsize=0)
            self.columnconfigure(3, minsize=0)
            self.inspector_visible.set(False)
        else:
            self.columnconfigure(2, minsize=6)
            self.columnconfigure(3, minsize=self._inspector_width)
            self.inspector_resize_handle.grid(row=0, column=2, sticky="ns")
            self.inspector.grid(row=0, column=3, sticky="nsew")
            self.inspector_visible.set(True)

    def _start_selected_agent_run(self) -> None:
        config = self._selected_config()
        if config is None:
            messagebox.showinfo("未选择 APP", "请先选择一个应用会话。")
            return
        if self.agent_running.get():
            self._agent_stop_event.set()
            self.agent_run_status.set("正在停止")
            self.status_text.set("已请求停止，当前模型响应结束后会停下")
            self.state.record_operation_summary(
                app_id=config.app_id,
                kind="agent_run",
                title="请求停止",
                summary="用户再次点击启动按钮，请求停止当前 Agent 执行。",
                status="stopping",
            )
            self.refresh_all()
            self._save_state()
            return
        latest_status = self._latest_agent_run_status(config.app_id)
        if self._is_terminal_agent_status(latest_status):
            self.status_text.set("当前会话已完成；如需继续，请先创建新会话。")
            return
        trace_path = self._ensure_trace_path(config)
        self._agent_stop_event.clear()
        self.agent_running.set(True)
        self.agent_run_status.set("运行中")
        self.state.trace_store.path = trace_path
        self._save_state()
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state.record_operation_summary(
            app_id=config.app_id,
            kind="agent_run",
            title="Agent 已启动",
            summary=f"轨迹写入 {trace_path.name}",
            status="running",
            payload={"run_id": run_id, "trace_path": str(trace_path)},
        )
        self.refresh_all()
        worker = threading.Thread(
            target=self._run_agent_session_worker,
            args=(config.app_id, trace_path, run_id),
            daemon=True,
        )
        worker.start()

    def _ensure_trace_path(self, config: TargetAppConfig) -> Path:
        metadata = dict(config.metadata or {})
        raw_path = str(metadata.get("trace_path") or "").strip()
        if raw_path:
            trace_path = Path(raw_path).expanduser()
        else:
            trace_path = (
                DEFAULT_TRACE_DIR
                / f"{config.app_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            )
            metadata["trace_path"] = str(trace_path)
            self.state.target_apps[config.app_id] = replace(config, metadata=metadata)
            self._save_state()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        return trace_path

    def _run_agent_session_worker(
        self,
        app_id: str,
        trace_path: Path,
        run_id: str,
    ) -> None:
        config = self.state.target_apps[app_id]
        metadata = dict(config.metadata or {})
        status = "failed"
        reason = ""
        steps = 0
        try:
            tool_registry = ModelToolRegistry(
                allowed_tool_names=model_tool_names_for_profile(
                    str(metadata.get("tool_profile") or "core")
                )
            )
            screen = self._screen_capture_from_metadata(metadata)
            runner = GuiAgentRunner(
                computer_loop=ComputerLoop(screen=screen),
                context_manager=ContextManager(
                    policy=VisionContextPolicy.agility(
                        max_visual_frames=max(
                            1, int(metadata.get("max_visual_frames") or 8)
                        )
                    )
                ),
                trace_store=self.state.trace_store,
                human_loop=self.state.human_loop,
                action_macros=self.state.action_macros,
                tool_registry=tool_registry,
                max_steps=int(metadata.get("max_steps") or DEFAULT_AGENT_MAX_STEPS),
                image_detail=str(metadata.get("image_detail") or "high"),
                require_finish_tool_for_completion=True,
                guard_untrusted_clicks=(
                    str(metadata.get("guard_untrusted_clicks", "true")).lower()
                    not in {"0", "false", "no", "off"}
                ),
                stop_requested=self._agent_stop_event.is_set,
            )
            task = GuiAgentTaskSpec(
                task_id=f"session:{config.app_id}:{run_id}",
                instruction=(config.task_description.strip() or f"操作 {config.app_name}"),
                target_app=config.app_name,
                metadata={"app_id": config.app_id, **metadata},
            )
            agent = OpenAIResponsesAgent(
                model=str(metadata.get("model") or DEFAULT_RESPONSES_MODEL),
                base_url=str(metadata.get("base_url") or DEFAULT_RESPONSES_BASE_URL),
                tool_registry=tool_registry,
                max_output_tokens=int(metadata.get("max_output_tokens") or 512),
                reasoning_effort=_agent_reasoning_effort(metadata),
                chat_template_kwargs=_agent_chat_template_kwargs(metadata),
            )
            result = runner.run_task(task, agent)
            status = result.status
            reason = result.reason or result.final_text or ""
            steps = result.steps
        except Exception as exc:
            reason = str(exc)
        self.after(
            0,
            lambda: self._finish_agent_run(
                app_id=app_id,
                trace_path=trace_path,
                run_id=run_id,
                status=status,
                reason=reason,
                steps=steps,
            ),
        )

    def _finish_agent_run(
        self,
        *,
        app_id: str,
        trace_path: Path,
        run_id: str,
        status: str,
        reason: str = "",
        steps: int = 0,
    ) -> None:
        self.agent_running.set(False)
        self._agent_stop_event.clear()
        self.agent_run_status.set(self._agent_status_label(status))
        config = self.state.target_apps.get(app_id)
        if config is not None:
            metadata = dict(config.metadata or {})
            metadata["last_run_status"] = status
            metadata["last_run_id"] = run_id
            metadata["last_trace_path"] = str(trace_path)
            self.state.target_apps[app_id] = replace(config, metadata=metadata)
        if trace_path.exists():
            self.state.trace_store.load_existing(trace_path)
        self.state.trace_store.path = trace_path
        self.state.record_operation_summary(
            app_id=app_id,
            kind="agent_run",
            title=self._agent_completion_title(status),
            summary=self._agent_completion_summary(
                trace_path=trace_path,
                steps=steps,
                reason=reason,
            ),
            status=status,
            payload={
                "run_id": run_id,
                "trace_path": str(trace_path),
                "result_status": status,
                "result_reason": reason,
                "result_steps": steps,
            },
        )
        self.refresh_all()
        self._save_state()

    def _screen_capture_from_metadata(self, metadata: dict[str, Any]) -> ScaledPillowScreenCapture:
        viewport = metadata.get("manual_viewport")
        crop_box = None
        if isinstance(viewport, dict):
            try:
                crop_box = (
                    int(viewport["x"]),
                    int(viewport["y"]),
                    int(viewport["width"]),
                    int(viewport["height"]),
                )
            except (KeyError, TypeError, ValueError):
                crop_box = None
        if crop_box is None:
            crop_box = self._auto_window_crop_box(metadata)
        return ScaledPillowScreenCapture(
            max_width=self._metadata_int(metadata, "screenshot_max_width", 1920),
            max_height=self._metadata_int(metadata, "screenshot_max_height", 1080),
            image_format=str(metadata.get("screenshot_format") or "JPEG"),
            jpeg_quality=self._metadata_int(metadata, "jpeg_quality", 85),
            crop_box=crop_box,
        )

    @staticmethod
    def _auto_window_crop_box(
        metadata: dict[str, Any],
    ) -> tuple[int, int, int, int] | None:
        pattern = str(metadata.get("window_title_pattern") or "").strip()
        if not pattern or pattern in {".*", "^.*$", ".+"}:
            return None
        try:
            window = find_visible_window(pattern)
        except Exception:
            return None
        if window is None:
            return None
        return window.crop_box

    @staticmethod
    def _metadata_int(
        metadata: dict[str, Any],
        key: str,
        default: int,
    ) -> int:
        try:
            value = int(metadata.get(key) or default)
        except (TypeError, ValueError):
            value = default
        return max(1, value)

    @staticmethod
    def _agent_status_label(status: str) -> str:
        return {
            "completed": "完成",
            "waiting_human": "等待人工",
            "paused": "已暂停",
            "blocked": "阻塞",
            "failed": "异常",
            "max_steps_exceeded": "达到步数上限",
        }.get(status, status or "完成")

    def _agent_completion_title(self, status: str) -> str:
        if status == "waiting_human":
            return "Agent 等待人工"
        if status == "blocked":
            return "Agent 已阻塞"
        if status == "max_steps_exceeded":
            return "Agent 达到步数上限"
        if status == "completed":
            return "Agent 执行完成"
        if status == "failed":
            return "Agent 执行异常"
        return f"Agent 状态：{self._agent_status_label(status)}"

    def _agent_completion_summary(
        self,
        *,
        trace_path: Path,
        steps: int,
        reason: str,
    ) -> str:
        parts = [f"轨迹 {trace_path.name}"]
        if steps:
            parts.append(f"{steps} 步")
        if reason:
            parts.append(_short_ui_text(reason, 160))
        return " / ".join(parts)
    def _submit_user_message(self) -> None:
        text = self.composer_text.get("1.0", tk.END).strip()
        pending = self._current_pending_request()
        if pending is not None:
            direct_command = self._selected_direct_command()
            if not text and direct_command == DIRECT_COMMAND_NONE:
                messagebox.showwarning("缺少输入", "请填写处理意见，或选择一个直接命令。")
                return
            request_id = pending["request"]["request_id"]
            try:
                self.state.human_loop.claim_request(request_id, source="client")
                self.state.submit_structured_human_reply(
                    request_id=request_id,
                    manager_input=text,
                    decision_type=self.decision_type.get(),
                    direct_command=direct_command,
                )
            except Exception as exc:
                messagebox.showerror("提交失败", f"{exc}\n\n该请求可能已被其他渠道处理。")
                self.refresh_all()
                return
            self.composer_text.delete("1.0", tk.END)
            self.direct_command_label.set(DIRECT_COMMAND_LABELS[DIRECT_COMMAND_NONE])
            self.status_text.set("人工处理已提交")
            self.refresh_all()
            self._save_state()
            return
        if not text:
            return
        self.state.trace_store.record(
            "manager_note",
            {
                "app_id": self.selected_app_id.get(),
                "text": text,
                "source": "desktop_client",
            },
        )
        self.composer_text.delete("1.0", tk.END)
        self.status_text.set("已记录管理者输入")
        self.refresh_messages()
        self.refresh_inspector()
        self.refresh_composer()
        self._save_state()

    def _on_inspector_select(self, _event: tk.Event[Any]) -> None:
        return

    def _select_inspector_item(self, selected: str) -> None:
        if not selected:
            return
        config = self._selected_config()
        if selected == "task_definition" and config is not None:
            self._show_inspector_detail(config.task_description)
            return
        if selected.startswith("asset:") and config is not None:
            asset_id = selected.split(":", 1)[1]
            for asset in config.reference_assets:
                if asset.asset_id == asset_id:
                    self._show_inspector_detail(format_asset(asset))
                    return
        if selected.startswith("human:"):
            index = int(selected.split(":", 1)[1])
            items = list(self.state.human_loop.list_requests(include_completed=True))
            if 0 <= index < len(items):
                self._show_inspector_detail(self._human_request_text(items[index]))
            return
        if selected.startswith("trace:"):
            index = int(selected.split(":", 1)[1])
            if config is not None:
                events = self._events_for_selected_app(config.app_id)[-30:]
                if 0 <= index < len(events):
                    self._show_inspector_detail(self._format_trace_detail(events[index]))

    def _select_trace_event(self, event: Any) -> None:
        self._selected_trace_event = event
        if not self.inspector_visible.get():
            self._toggle_inspector()
        self._inspector_signature = (
            self.selected_app_id.get(),
            self._trace_event_signature(event),
        )
        self._render_trace_event_detail(event)

    def _render_trace_event_detail(self, event: Any) -> None:
        for child in self.inspector_list_frame.winfo_children():
            child.destroy()
        self._inspector_images.clear()
        self.inspector_detail_shell.grid_remove()
        self._set_text(self.inspector_detail, "")

        payload = event.payload if isinstance(event.payload, dict) else {}
        model_event = self._model_response_event_for_trace_event(event)
        model_payload = (
            model_event.payload
            if model_event is not None and isinstance(model_event.payload, dict)
            else {}
        )
        title = self._operation_title(event.kind, payload)
        summary = self._operation_summary(event.kind, payload)
        overview = summary or ""
        self._add_detail_text_section(
            "记录概要",
            "\n".join(str(part) for part in (title, overview) if part),
        )

        if model_payload:
            request_step = model_payload.get("step")
            request_context_for_count = model_payload.get("request_context")
            context_count = (
                len(request_context_for_count)
                if isinstance(request_context_for_count, list)
                else 0
            )
            self._add_detail_text_section(
                "关联模型请求",
                (
                    f"当前记录关联第 {request_step} 次模型请求。\n"
                    f"该请求的完整上下文为 {context_count} 条 input item。"
                ),
            )

        request_context = model_payload.get("request_context")
        if isinstance(request_context, list) and request_context:
            latest_context = self._latest_request_context_items(request_context)
            if latest_context:
                self._add_detail_request_context_section(
                    "最新新增输入",
                    latest_context,
                    collapsed=False,
                    include_note=False,
                )
            self._add_detail_request_context_section(
                "完整请求上下文",
                request_context,
                collapsed=True,
                include_note=True,
            )
        elif event.kind == "step":
            self._add_detail_text_section(
                "最新新增输入",
                "当前轨迹没有记录完整请求上下文；下方截图为本轮操作前传给模型的界面证据。",
            )

        model_image_refs = self._model_request_image_ref_pairs(event)
        if model_image_refs:
            self._add_detail_image_section("本轮输入中的历史观察截图", model_image_refs)
        screenshot_refs = self._detail_screenshot_refs(event)
        if screenshot_refs:
            self._add_detail_image_section("本轮工具执行后的新观察截图", screenshot_refs)

        if model_payload:
            reasoning = str(
                model_payload.get("reasoning_text")
                or model_payload.get("reasoning_text_preview")
                or ""
            ).strip()
            reasoning_for_user = self._reasoning_text_for_user(reasoning)
            reasoning_preamble = str(model_payload.get("reasoning_preamble") or "").strip()
            reasoning_generated = str(
                model_payload.get("reasoning_generated_text")
                or model_payload.get("reasoning_generated_text_preview")
                or ""
            ).strip()
            output_text = str(
                model_payload.get("output_text")
                or model_payload.get("output_text_preview")
                or ""
            ).strip()
            visible_text = self._strip_tool_markup(output_text)
            if reasoning_for_user:
                self._add_detail_text_section("思考", reasoning_for_user)
            elif reasoning_preamble:
                self._add_detail_text_section(
                    "思考",
                    "本轮只注入了思考引导，模型没有继续生成额外思考内容。",
                )
            if reasoning_preamble:
                self._add_detail_text_section("思考引导", reasoning_preamble)
            if reasoning_generated and reasoning_generated != reasoning_for_user:
                self._add_detail_text_section("模型实际思考原文", reasoning_generated)
            response_object = model_payload.get("response_object")
            if response_object is not None:
                self._add_detail_json_section(
                    "Responses 返回对象",
                    response_object,
                    collapsed=True,
                )
            if visible_text and response_object is None:
                self._add_detail_text_section("模型文字输出", visible_text)
            elif response_object is None and output_text:
                self._add_detail_text_section("模型原始输出", output_text)

            stats = self._model_response_stats_text(model_payload)
            if stats:
                self._add_detail_text_section("响应统计", stats)

        tool_text = self._trace_tool_detail_text(event)
        if tool_text:
            self._add_detail_text_section("工具调用", tool_text)

    def _add_detail_text_section(self, title: str, text: str) -> None:
        if not text.strip():
            return
        card = tk.Frame(
            self.inspector_list_frame,
            bg="#fbfbfa",
            padx=12,
            pady=10,
            highlightthickness=1,
            highlightbackground=self.colors["line_soft"],
        )
        card.pack(fill="x", pady=(0, 10))
        tk.Label(
            card,
            text=title,
            bg="#fbfbfa",
            fg=self.colors["ink"],
            anchor="w",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(fill="x")
        tk.Label(
            card,
            text=text,
            bg="#fbfbfa",
            fg=self.colors["ink"],
            anchor="w",
            justify="left",
            wraplength=self._detail_wraplength(70),
            font=("Microsoft YaHei UI", 9),
        ).pack(fill="x", pady=(7, 0))
        self._bind_mousewheel_tree(card, self.inspector_canvas)

    def _add_detail_collapsible_section(
        self,
        title: str,
        build_body: Any,
        *,
        parent: tk.Widget | None = None,
        subtitle: str = "",
        collapsed: bool = True,
    ) -> None:
        parent = parent or self.inspector_list_frame
        card = tk.Frame(
            parent,
            bg="#fbfbfa",
            padx=12,
            pady=10,
            highlightthickness=1,
            highlightbackground=self.colors["line_soft"],
        )
        card.pack(fill="x", pady=(0, 10))
        header = tk.Frame(card, bg="#fbfbfa")
        header.pack(fill="x")
        text_holder = tk.Frame(header, bg="#fbfbfa")
        text_holder.pack(side="left", fill="x", expand=True)
        tk.Label(
            text_holder,
            text=title,
            bg="#fbfbfa",
            fg=self.colors["ink"],
            anchor="w",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(fill="x")
        if subtitle:
            subtitle_label = tk.Label(
                text_holder,
                text=subtitle,
                bg="#fbfbfa",
                fg=self.colors["muted"],
                anchor="w",
                justify="left",
                wraplength=self._detail_wraplength(120),
                font=("Microsoft YaHei UI", 8),
            )
            subtitle_label.pack(fill="x", pady=(3, 0))
            subtitle_label.bind(
                "<Configure>",
                lambda _event, label=subtitle_label: label.configure(
                    wraplength=self._detail_wraplength(120)
                ),
            )
        button_text = tk.StringVar(value="展开" if collapsed else "收起")
        body = tk.Frame(card, bg="#fbfbfa")
        built = {"value": False}

        def ensure_body() -> None:
            if built["value"]:
                return
            build_body(body)
            built["value"] = True
            self._bind_mousewheel_tree(card, self.inspector_canvas)
            self.inspector_canvas.configure(scrollregion=self.inspector_canvas.bbox("all"))

        def toggle() -> None:
            if body.winfo_manager():
                body.pack_forget()
                button_text.set("展开")
            else:
                ensure_body()
                body.pack(fill="x", pady=(10, 0))
                button_text.set("收起")
            self.inspector_canvas.configure(scrollregion=self.inspector_canvas.bbox("all"))

        tk.Button(
            header,
            textvariable=button_text,
            command=toggle,
            bg="#f3f2f0",
            fg=self.colors["ink"],
            activebackground="#ece9e5",
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            font=("Microsoft YaHei UI", 8),
        ).pack(side="right", padx=(8, 0))
        if not collapsed:
            ensure_body()
            body.pack(fill="x", pady=(10, 0))
        self._bind_mousewheel_tree(card, self.inspector_canvas)

    def _add_detail_json_section(
        self,
        title: str,
        value: Any,
        *,
        parent: tk.Widget | None = None,
        collapsed: bool = True,
    ) -> None:
        def build(body: tk.Frame) -> None:
            self._add_json_tree(body, value)

        self._add_detail_collapsible_section(
            title,
            build,
            parent=parent,
            subtitle=self._json_root_summary(value),
            collapsed=collapsed,
        )

    def _add_json_tree(self, parent: tk.Widget, value: Any) -> None:
        if isinstance(value, dict):
            if not value:
                self._add_nested_text_block(parent, "空对象", "{}")
                return
            for key, child in value.items():
                self._add_json_node(parent, str(key), child)
            return
        if isinstance(value, list):
            if not value:
                self._add_nested_text_block(parent, "空数组", "[]")
                return
            for index, child in enumerate(value):
                self._add_json_node(parent, f"[{index}]", child)
            return
        self._add_nested_text_block(parent, "值", self._format_json_scalar(value))

    def _add_json_node(self, parent: tk.Widget, label: str, value: Any) -> None:
        if isinstance(value, (dict, list)):
            self._add_detail_collapsible_section(
                label,
                lambda body, child=value: self._add_json_tree(body, child),
                parent=parent,
                subtitle=self._json_root_summary(value),
                collapsed=True,
            )
            return
        if isinstance(value, str):
            prose, parsed_json = self._split_json_tail(value)
            if parsed_json is not None:
                if prose:
                    self._add_nested_text_block(parent, f"{label} / 文本", prose)
                    json_label = f"{label} / JSON"
                else:
                    json_label = label
                self._add_detail_collapsible_section(
                    json_label,
                    lambda body, child=parsed_json: self._add_json_tree(body, child),
                    parent=parent,
                    subtitle=self._json_root_summary(parsed_json),
                    collapsed=True,
                )
                return
        parsed = self._try_parse_json_text(value)
        if parsed is not None:
            self._add_detail_collapsible_section(
                label,
                lambda body, child=parsed: self._add_json_tree(body, child),
                parent=parent,
                subtitle=self._json_root_summary(parsed),
                collapsed=True,
            )
            return
        self._add_nested_text_block(parent, label, self._format_json_scalar(value))

    @staticmethod
    def _json_root_summary(value: Any) -> str:
        if isinstance(value, dict):
            keys = list(value.keys())
            if not keys:
                return "JSON 对象，0 个字段。"
            head = "、".join(str(key) for key in keys[:4])
            suffix = "..." if len(keys) > 4 else ""
            return f"JSON 对象，{len(keys)} 个字段：{head}{suffix}"
        if isinstance(value, list):
            return f"JSON 数组，{len(value)} 项。"
        return "JSON 标量。"

    @staticmethod
    def _format_json_scalar(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

    def _add_json_text_widget(self, parent: tk.Widget, text: str) -> None:
        line_count = max(4, min(22, text.count("\n") + 1))
        shell = tk.Frame(parent, bg="#fbfbfa")
        shell.pack(fill="x")
        widget = tk.Text(
            shell,
            height=line_count,
            wrap="none",
            bg="#ffffff",
            fg=self.colors["ink"],
            relief="flat",
            padx=8,
            pady=8,
            font=("Consolas", 8),
            highlightthickness=1,
            highlightbackground=self.colors["line_soft"],
        )
        widget.insert("1.0", text)
        widget.configure(state="disabled")
        widget.pack(side="left", fill="x", expand=True)
        scroll = ttk.Scrollbar(shell, orient="vertical", command=widget.yview)
        scroll.pack(side="right", fill="y")
        widget.configure(yscrollcommand=scroll.set)
        self._bind_mousewheel_tree(shell, self.inspector_canvas)

    def _add_detail_request_context_section(
        self,
        title: str,
        request_context: list[Any],
        *,
        collapsed: bool = True,
        include_note: bool = True,
    ) -> None:
        def build(body: tk.Frame) -> None:
            if include_note:
                self._add_nested_text_block(
                    body,
                    "说明",
                    "这是本轮真正发送给 Responses API 的标准化 input。图片和视频只显示占位符，避免把 base64 当作文本渲染。",
                )
            for index, item in enumerate(request_context, start=1):
                if isinstance(item, dict):
                    self._add_request_context_item(body, index, item)
                else:
                    self._add_detail_json_section(
                        f"第 {index} 条",
                        item,
                        parent=body,
                        collapsed=True,
                    )

        self._add_detail_collapsible_section(
            title,
            build,
            subtitle=f"{len(request_context)} 条 input item；system、user、tool 分开显示。",
            collapsed=collapsed,
        )

    def _add_request_context_item(
        self,
        parent: tk.Widget,
        index: int,
        item: dict[str, Any],
    ) -> None:
        role = str(item.get("role") or self._request_item_role(item))
        item_type = str(item.get("type") or "item")
        title = f"第 {index} 条 / {self._role_label(role)} / {item_type}"
        subtitle = self._request_item_summary(item)

        def build(body: tk.Frame) -> None:
            if item_type == "message":
                content = item.get("content")
                if isinstance(content, list):
                    counts = self._request_content_type_counts(content)
                    seen: dict[str, int] = {}
                    for part in content:
                        part_type = (
                            str(part.get("type") or "")
                            if isinstance(part, dict)
                            else "part"
                        )
                        seen[part_type] = seen.get(part_type, 0) + 1
                        self._add_request_content_part(
                            body,
                            part,
                            counts=counts,
                            ordinal=seen[part_type],
                        )
                    return
                if isinstance(content, str):
                    self._add_text_or_json_block(body, "content", content)
                    return
            if item_type == "function_call_output":
                output = item.get("output")
                parsed = self._try_parse_json_text(output)
                if parsed is not None:
                    self._add_detail_json_section(
                        "工具结果 output",
                        parsed,
                        parent=body,
                        collapsed=True,
                    )
                else:
                    self._add_text_or_json_block(body, "工具结果 output", str(output or ""))
                return
            if item_type == "computer_call_output":
                output = item.get("output")
                if isinstance(output, dict):
                    self._add_computer_observation_detail(
                        body,
                        call_id=str(item.get("call_id") or "").strip(),
                        output=output,
                    )
                    return
            self._add_detail_json_section("原始 item", item, parent=body, collapsed=True)

        self._add_detail_collapsible_section(
            title,
            build,
            parent=parent,
            subtitle=subtitle,
            collapsed=True,
        )

    def _add_computer_observation_detail(
        self,
        parent: tk.Widget,
        *,
        call_id: str,
        output: dict[str, Any],
    ) -> None:
        if call_id:
            self._add_nested_text_block(parent, "工具调用", f"call_id={call_id}")

        observation_text = str(
            output.get("observation_text") or output.get("summary") or ""
        ).strip()
        if observation_text:
            self._add_nested_text_block(parent, "观察说明", observation_text)

        image_ref_index = 1
        image_ref_index = self._add_observation_image_or_placeholder(
            parent,
            title=f"观察截图（引用 {image_ref_index}）",
            value=output.get("image_url"),
            reference_label=f"引用 {image_ref_index}",
            next_index=image_ref_index,
        )

        structured_data = output.get("structured_data")
        if structured_data is not None:
            self._add_detail_json_section(
                "结构化观察数据",
                structured_data,
                parent=parent,
                collapsed=True,
            )

        local_refinements = output.get("local_refinements")
        if isinstance(local_refinements, list):
            for local in local_refinements:
                if not isinstance(local, dict):
                    continue
                click_index = local.get("click_index") or image_ref_index
                local_text = str(
                    local.get("observation_text") or local.get("summary") or ""
                ).strip()
                if local_text:
                    self._add_nested_text_block(
                        parent,
                        f"点击局部观察 {click_index}",
                        local_text,
                    )
                image_ref_index = self._add_observation_image_or_placeholder(
                    parent,
                    title=f"点击局部截图 {click_index}（引用 {image_ref_index}）",
                    value=local.get("image_url"),
                    reference_label=f"引用 {image_ref_index}",
                    next_index=image_ref_index,
                )
                local_structured_data = local.get("structured_data")
                if local_structured_data is not None:
                    self._add_detail_json_section(
                        f"点击局部结构化数据 {click_index}",
                        local_structured_data,
                        parent=parent,
                        collapsed=True,
                    )

    def _add_observation_image_or_placeholder(
        self,
        parent: tk.Widget,
        *,
        title: str,
        value: Any,
        reference_label: str,
        next_index: int,
    ) -> int:
        if not value:
            return next_index
        actual_ref = self._actual_image_ref_from_value(value)
        if actual_ref:
            self._add_inline_image_ref(parent, title, actual_ref)
        else:
            self._add_nested_text_block(
                parent,
                title,
                self._format_media_placeholder(
                    value,
                    reference_label=reference_label,
                ),
            )
        return next_index + 1

    def _add_request_content_part(
        self,
        parent: tk.Widget,
        part: Any,
        *,
        counts: dict[str, int] | None = None,
        ordinal: int = 1,
    ) -> None:
        if not isinstance(part, dict):
            self._add_text_or_json_block(parent, "内容", str(part))
            return
        counts = counts or {}
        part_type = str(part.get("type") or "part")
        if part_type == "input_text":
            label = self._content_part_label("文本内容", counts, part_type, ordinal)
            self._add_text_or_json_block(
                parent,
                label,
                str(part.get("text") or ""),
            )
            return
        if part_type in {"input_image", "input_video"}:
            key = "image_url" if part_type == "input_image" else "video_url"
            base_label = "图片内容" if part_type == "input_image" else "视频内容"
            reference_label = f"引用 {ordinal}"
            label = self._content_part_label(
                f"{base_label}（{reference_label}）",
                counts,
                part_type,
                ordinal,
            )
            self._add_nested_text_block(
                parent,
                label,
                self._format_media_placeholder(
                    part.get(key),
                    reference_label=reference_label,
                ),
            )
            actual_ref = self._actual_image_ref_from_value(part.get(key))
            if actual_ref:
                self._add_inline_image_ref(parent, reference_label, actual_ref)
            return
        self._add_detail_json_section(
            self._content_part_label(part_type, counts, part_type, ordinal),
            part,
            parent=parent,
            collapsed=True,
        )

    @staticmethod
    def _request_content_type_counts(content: list[Any]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for part in content:
            part_type = (
                str(part.get("type") or "part")
                if isinstance(part, dict)
                else "part"
            )
            counts[part_type] = counts.get(part_type, 0) + 1
        return counts

    @staticmethod
    def _content_part_label(
        base_label: str,
        counts: dict[str, int],
        part_type: str,
        ordinal: int,
    ) -> str:
        total = counts.get(part_type, 0)
        if total <= 1:
            return base_label
        return f"{base_label} {ordinal}/{total}"

    def _latest_request_context_items(
        self,
        request_context: list[Any],
    ) -> list[Any]:
        latest: list[Any] = []
        for item in reversed(request_context):
            if not isinstance(item, dict):
                if not latest:
                    latest.append(item)
                break
            item_type = str(item.get("type") or "")
            role = str(item.get("role") or self._request_item_role(item))
            if role in {"assistant", "system", "developer"}:
                if latest:
                    break
                continue
            if item_type in {"function_call", "tool_call", "computer_call"}:
                if latest:
                    break
                continue
            latest.append(item)
            if len(latest) >= 3:
                break
        return list(reversed(latest))

    def _add_text_or_json_block(
        self,
        parent: tk.Widget,
        title: str,
        text: str,
    ) -> None:
        prose, parsed_json = self._split_json_tail(text)
        if prose:
            self._add_nested_text_block(parent, title, prose)
        elif parsed_json is None:
            self._add_nested_text_block(parent, title, text)
        if parsed_json is not None:
            json_title = title if not prose else f"{title} / JSON"
            self._add_detail_json_section(
                json_title,
                parsed_json,
                parent=parent,
                collapsed=True,
            )

    def _add_nested_text_block(self, parent: tk.Widget, title: str, text: str) -> None:
        if not str(text).strip():
            return
        block = tk.Frame(parent, bg="#ffffff", padx=10, pady=8)
        block.pack(fill="x", pady=(0, 8))
        block.configure(highlightthickness=1, highlightbackground=self.colors["line_soft"])
        tk.Label(
            block,
            text=title,
            bg="#ffffff",
            fg=self.colors["muted"],
            anchor="w",
            font=("Microsoft YaHei UI", 8, "bold"),
        ).pack(fill="x")
        text_label = tk.Label(
            block,
            text=str(text),
            bg="#ffffff",
            fg=self.colors["ink"],
            anchor="w",
            justify="left",
            wraplength=self._detail_wraplength(110),
            font=("Microsoft YaHei UI", 8),
        )
        text_label.pack(fill="x", pady=(5, 0))
        text_label.bind(
            "<Configure>",
            lambda _event, label=text_label: label.configure(
                wraplength=self._detail_wraplength(110)
            ),
        )
        self._bind_mousewheel_tree(block, self.inspector_canvas)

    def _split_json_tail(self, text: str) -> tuple[str, Any | None]:
        raw = str(text or "").strip()
        if not raw:
            return "", None
        parsed = self._try_parse_json_text(raw)
        if parsed is not None:
            return "", parsed
        for index, char in enumerate(raw):
            if char not in "{[":
                continue
            prefix = raw[:index].strip()
            suffix = raw[index:].strip()
            parsed = self._try_parse_json_text(suffix)
            if parsed is not None:
                return prefix, parsed
        return raw, None

    @staticmethod
    def _try_parse_json_text(value: Any) -> Any | None:
        if not isinstance(value, str):
            return value if isinstance(value, (dict, list)) else None
        stripped = value.strip()
        if not stripped or stripped[0] not in "{[":
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError, TypeError):
                return None
            return parsed if isinstance(parsed, (dict, list)) else None

    @staticmethod
    def _request_item_role(item: dict[str, Any]) -> str:
        item_type = str(item.get("type") or "")
        if item_type in {"function_call_output", "computer_call_output"}:
            return "tool"
        if item_type in {"function_call", "tool_call", "computer_call"}:
            return "assistant"
        return item_type or "unknown"

    @staticmethod
    def _role_label(role: str) -> str:
        return {
            "system": "系统",
            "developer": "开发者",
            "user": "用户",
            "assistant": "模型",
            "tool": "工具",
            "function_call_output": "工具结果",
        }.get(role, role)

    def _request_item_summary(self, item: dict[str, Any]) -> str:
        item_type = str(item.get("type") or "")
        if item_type == "message":
            content = item.get("content")
            if isinstance(content, list):
                text_count = sum(
                    1
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "input_text"
                )
                image_count = sum(
                    1
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "input_image"
                )
                video_count = sum(
                    1
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "input_video"
                )
                return f"文本 {text_count} 段 / 图片 {image_count} 张 / 视频 {video_count} 段"
        if item_type in {"function_call_output", "computer_call_output"}:
            return f"call_id={item.get('call_id', '')}".strip()
        return ""

    @staticmethod
    def _format_media_placeholder(
        value: Any,
        *,
        reference_label: str | None = None,
    ) -> str:
        if isinstance(value, dict):
            parts = [str(value.get("placeholder") or "[媒体]")]
            if reference_label:
                parts.append(reference_label)
            for key, label in (
                ("source_type", "来源"),
                ("mime_type", "类型"),
                ("bytes", "字节"),
                ("path", "路径"),
                ("sha256", "sha256"),
                ("chars", "字符"),
            ):
                if value.get(key) not in (None, ""):
                    parts.append(f"{label}: {value.get(key)}")
            return "\n".join(parts)
        text = str(value or "[媒体]")
        if reference_label:
            return f"{reference_label}\n{text}"
        return text

    def _add_detail_image_section(self, title: str, image_refs: tuple[tuple[str, str], ...]) -> None:
        card = tk.Frame(
            self.inspector_list_frame,
            bg="#fbfbfa",
            padx=12,
            pady=10,
            highlightthickness=1,
            highlightbackground=self.colors["line_soft"],
        )
        card.pack(fill="x", pady=(0, 10))
        tk.Label(
            card,
            text=title,
            bg="#fbfbfa",
            fg=self.colors["ink"],
            anchor="w",
            font=("Microsoft YaHei UI", 10, "bold"),
        ).pack(fill="x")
        grid = tk.Frame(card, bg="#fbfbfa")
        grid.pack(fill="x", pady=(8, 0))
        column = 0
        for label_text, ref in image_refs:
            photo = self._timeline_thumbnail(ref)
            if photo is None:
                continue
            self._inspector_images.append(photo)
            holder = tk.Frame(grid, bg="#fbfbfa")
            holder.grid(row=0, column=column, sticky="w", padx=(0, 10), pady=(0, 8))
            thumb = tk.Label(
                holder,
                image=photo,
                bg="#ffffff",
                cursor="hand2",
                highlightthickness=1,
                highlightbackground=self.colors["line_soft"],
                borderwidth=0,
            )
            thumb.pack()
            thumb.bind(
                "<Button-1>",
                lambda _event, value=ref, title=label_text: self._open_image_ref_window(
                    value,
                    title,
                ),
            )
            tk.Label(
                holder,
                text=label_text,
                bg="#fbfbfa",
                fg=self.colors["muted"],
                font=("Microsoft YaHei UI", 8),
            ).pack(fill="x", pady=(4, 0))
            column += 1
        self._bind_mousewheel_tree(card, self.inspector_canvas)

    def _add_inline_image_ref(self, parent: tk.Widget, label_text: str, ref: str) -> None:
        photo = self._timeline_thumbnail(ref)
        if photo is None:
            return
        self._inspector_images.append(photo)
        holder = tk.Frame(parent, bg="#ffffff", padx=10, pady=8)
        holder.pack(fill="x", pady=(0, 8))
        holder.configure(highlightthickness=1, highlightbackground=self.colors["line_soft"])
        image_label = tk.Label(
            holder,
            image=photo,
            bg="#ffffff",
            highlightthickness=1,
            highlightbackground=self.colors["line_soft"],
            borderwidth=0,
            cursor="hand2",
        )
        image_label.pack(anchor="w")
        image_label.bind(
            "<Button-1>",
            lambda _event, value=ref, title=label_text: self._open_image_ref_window(
                value,
                title,
            ),
        )
        tk.Label(
            holder,
            text=label_text,
            bg="#ffffff",
            fg=self.colors["muted"],
            anchor="w",
            font=("Microsoft YaHei UI", 8),
        ).pack(fill="x", pady=(4, 0))
        self._bind_mousewheel_tree(holder, self.inspector_canvas)

    def _detail_screenshot_refs(self, event: Any) -> tuple[tuple[str, str], ...]:
        payload = event.payload if isinstance(event.payload, dict) else {}
        refs: list[tuple[str, str]] = []
        after = str(payload.get("after_ref") or "").strip()
        if after:
            refs.append(("本轮工具执行后观察截图", after))
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        local_refinements = metadata.get("local_refinements")
        if isinstance(local_refinements, list):
            for index, item in enumerate(local_refinements, start=1):
                if not isinstance(item, dict):
                    continue
                ref = str(item.get("image_ref") or item.get("image_url") or "").strip()
                if not ref:
                    continue
                click_index = item.get("click_index") or index
                refs.append((f"点击局部校验图 {click_index}", ref))
        for index, ref in enumerate(payload.get("artifact_refs") or ()):
            if isinstance(ref, str) and ref:
                refs.append((f"证据 {index + 1}", ref))
        return tuple(refs)

    def _format_response_protocol_output(self, response_object: Any) -> str:
        if not isinstance(response_object, dict):
            return ""
        output = response_object.get("output")
        if not isinstance(output, list):
            return ""
        sections: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type == "message":
                role = str(item.get("role") or "assistant")
                content_lines: list[str] = []
                for content in item.get("content") or []:
                    if not isinstance(content, dict):
                        continue
                    content_type = str(content.get("type") or "")
                    text = str(content.get("text") or "").strip()
                    if text:
                        content_lines.append(f"{content_type or 'text'}：{text}")
                if content_lines:
                    sections.append(f"message / {role}\n" + "\n".join(content_lines))
                continue
            if item_type in {"function_call", "tool_call"}:
                function = item.get("function")
                function_name = (
                    function.get("name")
                    if isinstance(function, dict)
                    else ""
                )
                name = str(item.get("name") or function_name or "")
                arguments = item.get("arguments")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False, default=str)
                sections.append(
                    "tool call"
                    + (f" / {name}" if name else "")
                    + "\n"
                    + str(arguments).strip()
                )
                continue
            if item_type == "reasoning":
                reasoning_lines: list[str] = []
                for content in item.get("content") or []:
                    if not isinstance(content, dict):
                        continue
                    text = str(content.get("text") or "").strip()
                    if text:
                        reasoning_lines.append(text)
                summary = item.get("summary")
                if reasoning_lines:
                    sections.append("reasoning\n" + "\n".join(reasoning_lines))
                elif summary:
                    sections.append(
                        "reasoning\n"
                        + json.dumps(summary, ensure_ascii=False, default=str)
                    )
        return "\n\n".join(section for section in sections if section.strip())

    def _model_response_stats_text(self, payload: dict[str, Any]) -> str:
        lines: list[str] = []
        latency = payload.get("latency_seconds")
        if isinstance(latency, (int, float)):
            lines.append(f"响应耗时：{latency:.3f}s")
        for source_key, label in (
            ("input_tokens", "输入 token"),
            ("output_tokens", "输出 token"),
            ("reasoning_tokens", "思考 token"),
            ("total_tokens", "总 token"),
            ("output_text_chars", "可见输出字符"),
            ("reasoning_text_chars", "思考字符"),
            ("reasoning_generated_text_chars", "模型实际思考字符"),
            ("tool_argument_chars", "工具参数字符"),
            ("response_json_chars", "响应体字符"),
        ):
            value = payload.get(source_key)
            if isinstance(value, (int, float)) and value:
                lines.append(f"{label}：{value}")
        warnings = payload.get("warnings")
        if warnings:
            lines.append(f"关注：{self._warning_summary(warnings)}")
        return "\n".join(lines)

    def _trace_tool_detail_text(self, event: Any) -> str:
        payload = event.payload if isinstance(event.payload, dict) else {}
        if event.kind != "step":
            return ""
        action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
        if action.get("type") == "computer_call":
            actions = action.get("actions") if isinstance(action.get("actions"), list) else []
            lines = ["工具：computer_use"]
            coordinate_space = action.get("coordinate_space")
            if coordinate_space:
                lines.append(f"坐标口径：{coordinate_space}")
            for item in actions:
                lines.append(f"- {format_computer_action(item)}")
            lines.append("")
            lines.append("原始参数：")
            lines.append(json.dumps(action, ensure_ascii=False, indent=2, default=str))
            return "\n".join(lines)
        if action.get("type") == "agent_tool":
            name = str(action.get("name") or "")
            lines = [f"工具：{self._tool_display_name(name)} ({name})"]
            arguments = action.get("arguments")
            if arguments is not None:
                lines.append("参数：")
                lines.append(json.dumps(arguments, ensure_ascii=False, indent=2, default=str))
            output = payload.get("metadata", {}).get("output", {})
            if output:
                lines.append("结果：")
                lines.append(json.dumps(output, ensure_ascii=False, indent=2, default=str))
            return "\n".join(lines)
        return ""

    def _open_image_ref_window(self, image_ref: str, title: str) -> None:
        if Image is None or ImageTk is None:
            messagebox.showinfo("无法预览", "当前环境没有可用的图片预览依赖。")
            return
        try:
            image = self._load_image_ref(image_ref)
        except Exception as exc:
            messagebox.showerror("图片加载失败", str(exc))
            return
        if image is None:
            messagebox.showerror("图片加载失败", "未找到这张截图。")
            return
        original = image.convert("RGB")
        max_width = min(1280, max(720, self.winfo_screenwidth() - 120))
        max_height = min(900, max(520, self.winfo_screenheight() - 120))
        canvas_width = max(520, max_width - 24)
        canvas_height = max(360, max_height - 70)
        fit_zoom = min(
            canvas_width / max(1, original.width),
            canvas_height / max(1, original.height),
            1.0,
        )
        fit_zoom = max(0.05, fit_zoom)
        zoom = {"value": fit_zoom}
        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

        window = tk.Toplevel(self)
        window.title(title)
        window.geometry(f"{max_width}x{max_height}")
        window.configure(bg="#ffffff")
        window.transient(self)
        toolbar = ttk.Frame(window, style="Surface.TFrame", padding=(10, 8))
        toolbar.pack(fill="x")
        zoom_text = tk.StringVar(value="")
        ttk.Label(
            toolbar,
            text=title,
            style="Section.TLabel",
        ).pack(side="left")
        ttk.Button(toolbar, text="－", width=3, command=lambda: set_zoom(zoom["value"] / 1.25)).pack(
            side="right",
            padx=(4, 0),
        )
        ttk.Button(toolbar, text="＋", width=3, command=lambda: set_zoom(zoom["value"] * 1.25)).pack(
            side="right",
            padx=(4, 0),
        )
        ttk.Button(toolbar, text="100%", width=6, command=lambda: set_zoom(1.0)).pack(
            side="right",
            padx=(4, 0),
        )
        ttk.Button(toolbar, text="适应", width=6, command=lambda: set_zoom(fit_zoom)).pack(
            side="right",
            padx=(4, 0),
        )
        ttk.Label(toolbar, textvariable=zoom_text, style="Hint.TLabel").pack(
            side="right",
            padx=(0, 8),
        )
        canvas = tk.Canvas(
            window,
            bg="#111827",
            highlightthickness=0,
            borderwidth=0,
            width=canvas_width,
            height=canvas_height,
        )
        canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        image_item = canvas.create_image(0, 0, anchor="nw")

        def render() -> None:
            width = max(1, int(original.width * zoom["value"]))
            height = max(1, int(original.height * zoom["value"]))
            resized = original.resize((width, height), resampling)
            photo = ImageTk.PhotoImage(resized)
            canvas.image = photo
            canvas.itemconfigure(image_item, image=photo)
            canvas.configure(scrollregion=(0, 0, width, height))
            zoom_text.set(f"{int(zoom['value'] * 100)}%")

        def set_zoom(value: float) -> None:
            zoom["value"] = max(0.05, min(8.0, float(value)))
            render()

        def on_wheel(event: tk.Event[Any]) -> str:
            event_num = getattr(event, "num", None)
            delta = int(getattr(event, "delta", 0) or 0)
            if delta and (int(getattr(event, "state", 0)) & 0x0004):
                set_zoom(zoom["value"] * (1.15 if delta > 0 else 1 / 1.15))
                return "break"
            if event_num == 4:
                canvas.yview_scroll(-3, "units")
            elif event_num == 5:
                canvas.yview_scroll(3, "units")
            elif delta:
                units = -max(1, abs(delta) // 120) if delta > 0 else max(1, abs(delta) // 120)
                canvas.yview_scroll(units, "units")
            return "break"

        canvas.bind("<MouseWheel>", on_wheel)
        canvas.bind("<Button-4>", on_wheel)
        canvas.bind("<Button-5>", on_wheel)
        canvas.bind("<ButtonPress-1>", lambda event: canvas.scan_mark(event.x, event.y))
        canvas.bind(
            "<B1-Motion>",
            lambda event: canvas.scan_dragto(event.x, event.y, gain=1),
        )
        window.bind("<Escape>", lambda _event: window.destroy())
        render()

    def _format_trace_detail(self, event: Any) -> str:
        payload = event.payload
        title = self._operation_title(event.kind, payload)
        summary = self._operation_summary(event.kind, payload)
        lines = [title]
        if summary:
            lines.append(summary)
        if event.kind == "model_response":
            latency = payload.get("latency_seconds")
            if isinstance(latency, (int, float)):
                lines.append(f"耗时：{latency:.3f}s")
            lines.append(f"可见文本：{payload.get('output_text_chars', 0)} 字")
            lines.append(f"工具参数：{payload.get('tool_argument_chars', 0)} 字")
            reasoning_chars = payload.get("reasoning_text_chars")
            if isinstance(reasoning_chars, int):
                lines.append(f"思考文本：{reasoning_chars} 字")
            preview = str(payload.get("output_text_preview") or "").strip()
            if preview:
                lines.append("")
                lines.append("模型文本输出：")
                lines.append(preview)
            reasoning_preview = str(payload.get("reasoning_text_preview") or "").strip()
            if reasoning_preview:
                lines.append("")
                lines.append("模型思考摘要：")
                lines.append(reasoning_preview)
            return "\n".join(lines)
        if event.kind == "agent_result":
            output_text = str(payload.get("output_text") or "").strip()
            if output_text:
                lines.append("")
                lines.append("实际输出：")
                lines.append(output_text)
            reason = str(payload.get("reason") or "").strip()
            if reason:
                lines.append(f"说明：{reason}")
            refs = payload.get("artifact_refs") or []
            if refs:
                lines.append("")
                lines.append("证据：")
                lines.extend(f"- {ref}" for ref in refs)
            return "\n".join(lines)
        if event.kind == "step":
            action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
            if action.get("type") == "computer_call":
                actions = action.get("actions") if isinstance(action.get("actions"), list) else []
                lines.append("")
                lines.append("电脑动作：")
                lines.extend(f"- {format_computer_action(action_item)}" for action_item in actions)
            elif action.get("type") == "agent_tool":
                name = str(action.get("name") or "")
                lines.append(f"工具：{self._tool_display_name(name)}")
                output = payload.get("metadata", {}).get("output", {})
                if isinstance(output, dict):
                    reason = str(output.get("parse_error") or output.get("reason") or "").strip()
                    if reason:
                        lines.append(f"处理结果：{reason}")
                    if name == "read_text_file" and output.get("path"):
                        lines.append(f"文件：{output.get('path')}")
            before = str(payload.get("before_ref") or "").strip()
            after = str(payload.get("after_ref") or "").strip()
            if before or after:
                lines.append("")
                lines.append("证据截图：")
                if before:
                    lines.append(f"- 操作前：{before}")
                if after:
                    lines.append(f"- 操作后：{after}")
            return "\n".join(lines)
        return "\n".join(lines)

    def _show_inspector_detail(self, text: str) -> None:
        self.inspector_detail_shell.grid()
        self._set_text(self.inspector_detail, text)

    def _events_for_selected_app(self, app_id: str) -> list[Any]:
        result: list[Any] = []
        for event in self.state.trace_store.events:
            if self._is_configuration_event(event):
                continue
            payload = event.payload
            event_app_id = payload.get("app_id")
            if event_app_id == app_id:
                result.append(event)
                continue
            if event.kind == "app_configured" and payload.get("app_id") == app_id:
                result.append(event)
                continue
            if event.kind in {"model_response", "step", "agent_result"} and not event_app_id:
                result.append(event)
                continue
            if event.kind == "operation" and not event_app_id:
                result.append(event)
        return self._dedupe_agent_run_events(result)

    @staticmethod
    def _is_configuration_event(event: Any) -> bool:
        payload = event.payload
        return event.kind == "app_configured" or (
            event.kind == "operation"
            and payload.get("kind") == "agent_run"
            and payload.get("status") == "configured"
        )

    @staticmethod
    def _dedupe_agent_run_events(events: list[Any]) -> list[Any]:
        terminal_agent_run_keys: set[tuple[str, str]] = set()
        for event in events:
            payload = event.payload
            if (
                event.kind == "operation"
                and payload.get("kind") == "agent_run"
                and payload.get("status") not in {"configured", "running"}
            ):
                key = GuiAgentDesktopClient._agent_run_operation_key(payload)
                if key[1]:
                    terminal_agent_run_keys.add(key)

        filtered_reversed: list[Any] = []
        seen: set[tuple[str, str]] = set()
        for event in reversed(events):
            payload = event.payload
            if (
                event.kind == "operation"
                and payload.get("kind") == "agent_run"
                and payload.get("status") == "running"
                and GuiAgentDesktopClient._agent_run_operation_key(payload)
                in terminal_agent_run_keys
            ):
                continue
            is_config = event.kind == "app_configured"
            is_config_operation = (
                event.kind == "operation"
                and payload.get("kind") == "agent_run"
                and payload.get("status") == "configured"
            )
            if is_config or is_config_operation:
                nested = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
                key = (
                    str(payload.get("app_id") or ""),
                    str(
                        payload.get("run_id")
                        or nested.get("run_id")
                        or payload.get("trace_path")
                        or payload.get("manifest_path")
                        or payload.get("summary")
                        or nested.get("manifest_path")
                        or payload.get("title")
                        or ""
                    ),
                )
                if key in seen:
                    continue
                seen.add(key)
            filtered_reversed.append(event)
        return list(reversed(filtered_reversed))

    @staticmethod
    def _agent_run_operation_key(payload: dict[str, Any]) -> tuple[str, str]:
        nested = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        key = (
            nested.get("result_json")
            or payload.get("result_json")
            or nested.get("run_id")
            or nested.get("log_path")
            or payload.get("summary")
            or payload.get("title")
            or ""
        )
        return (str(payload.get("app_id") or ""), str(key))

    def _current_pending_request(self) -> dict[str, Any] | None:
        config = self._selected_config()
        if config is None:
            return None
        for item in self.state.human_loop.list_requests():
            request = item["request"]
            metadata = request.get("metadata") or {}
            if metadata.get("job_id") != config.job_id:
                continue
            if item["status"] == "resolved":
                continue
            return item
        return None

    def _selected_direct_command(self) -> str:
        label = self.direct_command_label.get()
        for key, value in DIRECT_COMMAND_LABELS.items():
            if value == label:
                return key
        return DIRECT_COMMAND_NONE

    def _selected_config(self) -> TargetAppConfig | None:
        return self.state.target_apps.get(self.selected_app_id.get())

    def _first_app_id(self) -> str:
        try:
            return next(iter(self.state.target_apps))
        except StopIteration:
            return ""

    def _waiting_count_for_job(self, job_id: str) -> int:
        count = 0
        for item in self.state.human_loop.list_requests():
            metadata = item["request"].get("metadata") or {}
            if metadata.get("job_id") == job_id and item["status"] != "resolved":
                count += 1
        return count

    def _app_status_text(self, app_id: str, job_id: str) -> str:
        waiting = self._waiting_count_for_job(job_id)
        if waiting:
            return f"待处理 {waiting}"
        latest_status = self._latest_agent_run_status(app_id)
        if latest_status:
            return self._agent_status_label(latest_status)
        return "就绪"

    def _latest_agent_run_status(self, app_id: str) -> str:
        live_flag = self.__dict__.get("agent_running")
        selected = self.__dict__.get("selected_app_id")
        selected_app_id = selected.get() if selected is not None else ""
        if app_id == selected_app_id and live_flag is not None and live_flag.get():
            return "running"
        state = self.__dict__.get("state")
        config = state.target_apps.get(app_id) if state is not None else None
        metadata = config.metadata if config is not None else {}
        metadata_status = str((metadata or {}).get("last_run_status") or "")
        if metadata_status:
            return metadata_status
        for event in reversed(self._events_for_selected_app(app_id)):
            payload = event.payload
            if event.kind != "operation" or payload.get("kind") != "agent_run":
                continue
            status = str(payload.get("status") or "")
            if status and status != "configured":
                if status == "running":
                    return "paused"
                return status
        return ""

    @staticmethod
    def _is_terminal_agent_status(status: str) -> bool:
        return status in {
            "completed",
            "failed",
            "blocked",
            "max_steps_exceeded",
            "cancelled",
        }

    def _macros_for_app(self, app_id: str) -> list[Any]:
        return [
            macro
            for macro in self.state.action_macros.macros.values()
            if macro.metadata.get("scope") == "global"
            or macro.metadata.get("app_id") == app_id
        ]

    def _human_request_text(self, item: dict[str, Any]) -> str:
        request = item["request"]
        metadata = request.get("metadata") or {}
        lines = [
            f"状态：{self._status_label(item['status'])}",
            f"问题：{request.get('question') or ''}",
        ]
        if metadata.get("job_id"):
            lines.append(f"来源：{metadata['job_id']}")
        if request.get("risk_reason"):
            lines.append(f"风险：{request['risk_reason']}")
        if request.get("proposed_action"):
            lines.append(f"建议：{request['proposed_action']}")
        if item.get("reply"):
            lines.append(f"回复：{item['reply'].get('text') or ''}")
        return "\n".join(lines)

    @staticmethod
    def _status_label(status: str) -> str:
        return {
            "pending": "待处理",
            "claimed": "处理中",
            "resolved": "已处理",
            "cancelled": "已取消",
        }.get(status, status)

    @staticmethod
    def _trace_kind_label(kind: str) -> str:
        return {
            "step": "步骤",
            "manager_note": "输入",
            "policy_update": "规则",
            "result": "结果",
            "operation": "操作",
            "app_configured": "配置",
            "agent_result": "结果",
            "model_response": "响应",
            "computer_call": "电脑",
            "agent_tool": "工具",
            "computer_use": "电脑",
            "read_text_file": "文件",
            "append_trace_note": "记录",
            "viewport": "视野",
            "mouse": "鼠标",
            "keyboard": "键盘",
            "observe": "观察",
            "model_intent": "意图",
            "harness_check": "校验",
            "verification": "验证",
            "human": "人工",
            "failure": "异常",
        }.get(kind, kind)

    @staticmethod
    def _selected_tree_iid(tree: ttk.Treeview) -> str | None:
        selection = tree.selection()
        return str(selection[0]) if selection else None

    @staticmethod
    def _clear_tree(tree: ttk.Treeview) -> None:
        for item in tree.get_children():
            tree.delete(item)

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state="disabled")


class ViewportMarkerWindow(tk.Toplevel):
    def __init__(
        self,
        parent: GuiAgentDesktopClient,
        *,
        app_id: str,
        app_name: str,
        initial_box: tuple[int, int, int, int],
    ) -> None:
        super().__init__(parent)
        self.parent = parent
        self.app_id = app_id
        self.app_name = app_name
        self.min_width = 260
        self.min_height = 180
        self._transparent = "#ff00ff"
        self._action = ""
        self._start_pointer = (0, 0)
        self._start_geometry = initial_box
        self._button_bounds: dict[str, tuple[int, int, int, int]] = {}

        x, y, width, height = initial_box
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        try:
            self.attributes("-transparentcolor", self._transparent)
        except tk.TclError:
            self.attributes("-alpha", 0.88)
        self.geometry(f"{max(self.min_width, width)}x{max(self.min_height, height)}+{x}+{y}")

        self.canvas = tk.Canvas(
            self,
            bg=self._transparent,
            highlightthickness=0,
            borderwidth=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Motion>", self._on_motion)
        self.bind("<Escape>", lambda _event: self._close())
        self._redraw()

    def _current_box(self) -> tuple[int, int, int, int]:
        return (
            max(0, self.winfo_x()),
            max(0, self.winfo_y()),
            max(self.min_width, self.winfo_width()),
            max(self.min_height, self.winfo_height()),
        )

    def _redraw(self) -> None:
        width = max(self.canvas.winfo_width(), self.min_width)
        height = max(self.canvas.winfo_height(), self.min_height)
        self.canvas.delete("all")
        self._button_bounds = {}
        border = "#2563eb"
        fill = "#eff6ff"
        bar = "#111827"
        self.canvas.create_rectangle(
            2,
            2,
            width - 2,
            height - 2,
            outline=border,
            width=3,
        )
        self.canvas.create_rectangle(
            3,
            3,
            width - 3,
            35,
            fill=bar,
            outline=bar,
        )
        box = self._current_box()
        self.canvas.create_text(
            14,
            19,
            text=f"{self.app_name}  视野标定  {box[0]},{box[1]}  {box[2]}x{box[3]}",
            anchor="w",
            fill="#ffffff",
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        save_bounds = (width - 204, 7, width - 112, 31)
        hide_bounds = (width - 104, 7, width - 42, 31)
        close_bounds = (width - 34, 7, width - 10, 31)
        self._button_bounds = {
            "save": save_bounds,
            "hide": hide_bounds,
            "close": close_bounds,
        }
        self._draw_button(save_bounds, "保存隐藏", "#ffffff", "#111827")
        self._draw_button(hide_bounds, "隐藏", "#374151", "#ffffff")
        self._draw_button(close_bounds, "×", "#374151", "#ffffff")

        label = "把目标 APP 窗口拖到蓝框内；必要时拖动边框缩放。保存后该区域会作为当前 APP 的人工视野。"
        self.canvas.create_rectangle(
            14,
            height - 38,
            min(width - 14, 670),
            height - 12,
            fill=fill,
            outline=fill,
        )
        self.canvas.create_text(
            24,
            height - 25,
            text=label,
            anchor="w",
            fill="#1d4ed8",
            font=("Microsoft YaHei UI", 8),
        )
        for handle in (
            (width - 18, height - 18, width - 5, height - 5),
            (4, height - 18, 17, height - 5),
            (width - 18, 36, width - 5, 49),
        ):
            self.canvas.create_rectangle(*handle, fill=border, outline=border)

    def _draw_button(
        self,
        bounds: tuple[int, int, int, int],
        text: str,
        fill: str,
        foreground: str,
    ) -> None:
        x1, y1, x2, y2 = bounds
        draw_rounded_rect(
            self.canvas,
            x1,
            y1,
            x2,
            y2,
            8,
            fill=fill,
            outline=fill,
        )
        self.canvas.create_text(
            (x1 + x2) // 2,
            (y1 + y2) // 2,
            text=text,
            anchor="center",
            fill=foreground,
            font=("Microsoft YaHei UI", 8, "bold"),
        )

    def _on_press(self, event: tk.Event[Any]) -> None:
        button = self._button_at(event.x, event.y)
        if button == "save":
            self.parent._save_manual_viewport_from_marker(
                app_id=self.app_id,
                box=self._current_box(),
            )
            return
        if button in {"hide", "close"}:
            self._close()
            return
        self._action = self._hit_test(event.x, event.y)
        self._start_pointer = (event.x_root, event.y_root)
        self._start_geometry = self._current_box()

    def _on_drag(self, event: tk.Event[Any]) -> None:
        if not self._action:
            return
        dx = event.x_root - self._start_pointer[0]
        dy = event.y_root - self._start_pointer[1]
        x, y, width, height = self._start_geometry
        if self._action == "move":
            self.geometry(f"{width}x{height}+{max(0, x + dx)}+{max(0, y + dy)}")
            return
        left = x
        top = y
        right = x + width
        bottom = y + height
        if "e" in self._action:
            right += dx
        if "s" in self._action:
            bottom += dy
        if "w" in self._action:
            left += dx
        if "n" in self._action:
            top += dy
        if right - left < self.min_width:
            if "w" in self._action:
                left = right - self.min_width
            else:
                right = left + self.min_width
        if bottom - top < self.min_height:
            if "n" in self._action:
                top = bottom - self.min_height
            else:
                bottom = top + self.min_height
        self.geometry(
            f"{int(right - left)}x{int(bottom - top)}+{max(0, int(left))}+{max(0, int(top))}"
        )

    def _on_release(self, _event: tk.Event[Any]) -> None:
        self._action = ""
        self._redraw()

    def _on_motion(self, event: tk.Event[Any]) -> None:
        if self._action:
            return
        hit = self._hit_test(event.x, event.y)
        cursor = {
            "move": "fleur",
            "e": "sb_h_double_arrow",
            "w": "sb_h_double_arrow",
            "n": "sb_v_double_arrow",
            "s": "sb_v_double_arrow",
            "se": "size_nw_se",
            "nw": "size_nw_se",
            "ne": "size_ne_sw",
            "sw": "size_ne_sw",
        }.get(hit, "arrow")
        self.canvas.configure(cursor=cursor)

    def _button_at(self, x: int, y: int) -> str:
        for name, bounds in self._button_bounds.items():
            if _point_in_rect(x, y, bounds):
                return name
        return ""

    def _hit_test(self, x: int, y: int) -> str:
        width = max(self.winfo_width(), self.min_width)
        height = max(self.winfo_height(), self.min_height)
        margin = 12
        if y <= 36 and not self._button_at(x, y):
            return "move"
        west = x <= margin
        east = x >= width - margin
        north = y <= margin
        south = y >= height - margin
        if north and west:
            return "nw"
        if north and east:
            return "ne"
        if south and west:
            return "sw"
        if south and east:
            return "se"
        if west:
            return "w"
        if east:
            return "e"
        if north:
            return "n"
        if south:
            return "s"
        return ""

    def _close(self) -> None:
        if self.parent.viewport_marker is self:
            self.parent.viewport_marker = None
        self.destroy()


class AppConfigDialog(tk.Toplevel):
    def __init__(
        self,
        parent: GuiAgentDesktopClient,
        *,
        title: str,
        config: TargetAppConfig | None = None,
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.grab_set()
        self.geometry("760x620")
        self.minsize(680, 560)
        self.configure(bg=parent.colors["bg"])
        self.result: dict[str, Any] | None = None
        self.assets: list[ReferenceAsset] = list(config.reference_assets if config else ())

        frame = ttk.Frame(self, style="Surface.TFrame", padding=16)
        frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(5, weight=1)

        ttk.Label(frame, text=title, style="Title.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(frame, text="APP 名称", style="Hint.TLabel").grid(
            row=1, column=0, sticky="w", pady=(14, 4)
        )
        self.app_name = ttk.Entry(frame)
        self.app_name.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(14, 4))
        ttk.Label(frame, text="内部 ID", style="Hint.TLabel").grid(
            row=2, column=0, sticky="w", pady=4
        )
        self.job_id = ttk.Entry(frame)
        self.job_id.grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(frame, text="应用说明与任务规则", style="Section.TLabel").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(14, 4)
        )
        ttk.Label(
            frame,
            text="描述场景、规则、目标和约束。素材引用可直接插入到文字中。",
            style="Hint.TLabel",
        ).grid(row=4, column=0, columnspan=3, sticky="w")
        self.task_text = tk.Text(
            frame,
            height=14,
            wrap="word",
            bg="#ffffff",
            relief="solid",
            borderwidth=1,
            font=("Microsoft YaHei UI", 10),
        )
        self.task_text.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(8, 12))

        ttk.Label(frame, text="素材引用", style="Section.TLabel").grid(
            row=6, column=0, columnspan=3, sticky="w"
        )
        self.asset_tree = ttk.Treeview(
            frame,
            columns=("kind", "citation", "title"),
            show="headings",
            height=5,
        )
        self.asset_tree.heading("kind", text="类型")
        self.asset_tree.heading("citation", text="引用")
        self.asset_tree.heading("title", text="标题")
        self.asset_tree.column("kind", width=70, anchor="center")
        self.asset_tree.column("citation", width=190)
        self.asset_tree.column("title", width=320, stretch=True)
        self.asset_tree.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(8, 8))

        ttk.Button(frame, text="添加图片", command=lambda: self._add_asset("image")).grid(
            row=8, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(frame, text="添加视频", command=lambda: self._add_asset("video")).grid(
            row=8, column=1, sticky="ew", padx=6
        )
        ttk.Button(frame, text="插入引用", command=self._insert_selected_citation).grid(
            row=8, column=2, sticky="ew", padx=(6, 0)
        )

        buttons = ttk.Frame(frame, style="Surface.TFrame")
        buttons.grid(row=9, column=0, columnspan=3, sticky="e", pady=(16, 0))
        ttk.Button(buttons, text="取消", command=self.destroy).grid(row=0, column=0, padx=5)
        ttk.Button(
            buttons,
            text="保存",
            style="Primary.TButton",
            command=self._save,
        ).grid(row=0, column=1, padx=5)

        if config is not None:
            self.app_name.insert(0, config.app_name)
            self.job_id.insert(0, config.job_id)
            self.task_text.insert("1.0", config.task_description)
        self._refresh_assets()
        self.wait_window(self)

    def _add_asset(self, kind: str) -> None:
        filetypes = (
            [("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("All files", "*.*")]
            if kind == "image"
            else [("Video files", "*.mp4;*.mov;*.avi;*.mkv"), ("All files", "*.*")]
        )
        path = filedialog.askopenfilename(title=f"选择{kind}", filetypes=filetypes)
        if not path:
            return
        asset = ReferenceAsset(
            asset_id=f"{kind}_{Path(path).stem[:16]}_{len(self.assets) + 1}",
            kind=kind,
            path=path,
            title=Path(path).stem,
        )
        self.assets.append(asset)
        self._refresh_assets()

    def _insert_selected_citation(self) -> None:
        selection = self.asset_tree.selection()
        if not selection:
            return
        asset_id = str(selection[0])
        for asset in self.assets:
            if asset.asset_id == asset_id:
                self.task_text.insert(tk.INSERT, asset.citation)
                return

    def _refresh_assets(self) -> None:
        for item in self.asset_tree.get_children():
            self.asset_tree.delete(item)
        for asset in self.assets:
            self.asset_tree.insert(
                "",
                tk.END,
                iid=asset.asset_id,
                values=(asset.kind, asset.citation, asset.title),
            )

    def _save(self) -> None:
        self.result = {
            "app_name": self.app_name.get(),
            "job_id": self.job_id.get(),
            "task_description": self.task_text.get("1.0", tk.END),
            "assets": list(self.assets),
        }
        self.destroy()


class SettingsDialog(tk.Toplevel):
    def __init__(
        self,
        parent: GuiAgentDesktopClient,
        state: DesktopClientState,
        current_app_id: str,
    ) -> None:
        super().__init__(parent)
        self.title("设置")
        self.transient(parent)
        self.grab_set()
        self.geometry("760x560")
        self.minsize(700, 520)
        self.configure(bg=parent.colors["bg"])
        self.state = state
        self.current_app_id = current_app_id

        frame = ttk.Frame(self, style="Surface.TFrame", padding=16)
        frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(2, weight=1)

        ttk.Label(frame, text="设置", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            frame,
            text="全局规则和动作宏在这里维护。宏可应用于全部 APP 或当前 APP。",
            style="Hint.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 12))

        left = ttk.Frame(frame, style="Surface.TFrame")
        left.grid(row=2, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        ttk.Label(left, text="已注册宏", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.macro_tree = ttk.Treeview(
            left,
            columns=("status", "scope", "steps", "description"),
            show="headings",
        )
        self.macro_tree.heading("status", text="状态")
        self.macro_tree.heading("scope", text="范围")
        self.macro_tree.heading("steps", text="动作")
        self.macro_tree.heading("description", text="说明")
        self.macro_tree.column("status", width=80, anchor="center")
        self.macro_tree.column("scope", width=80, anchor="center")
        self.macro_tree.column("steps", width=140)
        self.macro_tree.column("description", width=260, stretch=True)
        self.macro_tree.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        right = ttk.Frame(frame, style="Surface.TFrame")
        right.grid(row=2, column=1, sticky="nsew")
        right.columnconfigure(1, weight=1)
        ttk.Label(right, text="新增动作宏", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(right, text="名称", style="Hint.TLabel").grid(row=1, column=0, sticky="w", pady=5)
        self.name_entry = ttk.Entry(right)
        self.name_entry.grid(row=1, column=1, sticky="ew", pady=5)
        ttk.Label(right, text="范围", style="Hint.TLabel").grid(row=2, column=0, sticky="w", pady=5)
        self.scope_var = tk.StringVar(value="全局")
        self.scope_combo = ttk.Combobox(
            right,
            textvariable=self.scope_var,
            values=("全局", "当前 APP"),
            state="readonly",
        )
        self.scope_combo.grid(row=2, column=1, sticky="ew", pady=5)
        ttk.Label(right, text="按键序列", style="Hint.TLabel").grid(row=3, column=0, sticky="w", pady=5)
        self.sequence_entry = ttk.Entry(right)
        self.sequence_entry.grid(row=3, column=1, sticky="ew", pady=5)
        ttk.Label(right, text="操作含义", style="Hint.TLabel").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(10, 4)
        )
        self.description_text = tk.Text(
            right,
            height=8,
            wrap="word",
            bg="#ffffff",
            relief="solid",
            borderwidth=1,
            font=("Microsoft YaHei UI", 10),
        )
        self.description_text.grid(row=5, column=0, columnspan=2, sticky="nsew")
        ttk.Button(
            right,
            text="保存宏",
            style="Primary.TButton",
            command=self._save_macro,
        ).grid(row=6, column=1, sticky="e", pady=(12, 0))

        ttk.Button(frame, text="关闭", command=self.destroy).grid(
            row=3, column=1, sticky="e", pady=(14, 0)
        )
        self._refresh_macros()
        self.wait_window(self)

    def _save_macro(self) -> None:
        name = self.name_entry.get().strip()
        sequence = self.sequence_entry.get().strip()
        description = self.description_text.get("1.0", tk.END).strip()
        if not name or not sequence or not description:
            messagebox.showwarning("输入不完整", "名称、按键序列和操作含义都需要填写。")
            return
        scope = "app" if self.scope_var.get() == "当前 APP" else "global"
        app_id = self.current_app_id if scope == "app" else None
        try:
            self.state.register_macro(
                MacroConfig(
                    name=name,
                    description=description,
                    sequence=sequence,
                    scope=scope,
                    app_id=app_id,
                )
            )
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self.name_entry.delete(0, tk.END)
        self.sequence_entry.delete(0, tk.END)
        self.description_text.delete("1.0", tk.END)
        self._refresh_macros()
        if self.master is not None and hasattr(self.master, "_save_state"):
            self.master._save_state()

    def _refresh_macros(self) -> None:
        for item in self.macro_tree.get_children():
            self.macro_tree.delete(item)
        for macro in self.state.action_macros.macros.values():
            scope = _macro_scope_label(macro.metadata, current_app_id=self.current_app_id)
            self.macro_tree.insert(
                "",
                tk.END,
                iid=macro.name,
                values=(
                    "已启用",
                    scope,
                    _macro_steps_summary(macro.to_dict()),
                    macro.description,
                ),
            )
        for item in self.state.human_loop.list_requests():
            request = item.get("request") if isinstance(item, dict) else {}
            metadata = request.get("metadata") if isinstance(request, dict) else {}
            proposal = metadata.get("macro_proposal") if isinstance(metadata, dict) else None
            if not isinstance(proposal, dict):
                continue
            job_id = str(metadata.get("job_id") or "")
            if self.current_app_id:
                app = self.state.target_apps.get(self.current_app_id)
                if app is not None and job_id and job_id != app.job_id:
                    continue
            request_id = str(request.get("request_id") or id(item))
            macro_name = str(proposal.get("macro_name") or "未命名宏")
            self.macro_tree.insert(
                "",
                tk.END,
                iid=f"pending:{request_id}",
                values=(
                    "待批准",
                    _macro_scope_label(proposal, current_app_id=self.current_app_id),
                    _macro_steps_summary(proposal),
                    f"{macro_name}：{proposal.get('description') or ''}".strip("："),
                ),
            )


def _macro_scope_label(data: dict[str, Any], *, current_app_id: str | None = None) -> str:
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    scope = str(data.get("scope") or metadata.get("scope") or "").strip()
    app_id = str(data.get("app_id") or metadata.get("app_id") or "").strip()
    if scope == "global":
        return "全局"
    if scope in {"app", "current_app"} or app_id:
        if current_app_id and app_id and app_id != current_app_id:
            return "其他 APP"
        return "当前 APP"
    return "当前 APP"


def _macro_steps_summary(data: dict[str, Any]) -> str:
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    sequence = str(data.get("sequence") or metadata.get("sequence") or "").strip()
    if sequence:
        return sequence
    steps = data.get("steps")
    if not isinstance(steps, list):
        return ""
    labels: list[str] = []
    for index, step in enumerate(steps[:4], start=1):
        if not isinstance(step, dict):
            continue
        action = step.get("action") if isinstance(step.get("action"), dict) else step
        action_type = str(action.get("type") or step.get("type") or "动作")
        purpose = str(step.get("purpose") or "").strip()
        label = purpose or _macro_action_label(action_type)
        labels.append(f"{index}. {label}")
    if len(steps) > 4:
        labels.append(f"另有 {len(steps) - 4} 步")
    return "；".join(labels)


def _macro_action_label(action_type: str) -> str:
    return {
        "click": "点击",
        "double_click": "双击",
        "submit_text": "输入并提交",
        "type": "输入文本",
        "wait": "等待",
        "keypress": "按键",
        "scroll": "滚动",
        "drag": "拖拽",
        "move": "移动鼠标",
        "screenshot": "截图",
        "computer": "电脑操作",
    }.get(action_type, action_type)


def format_payload(payload: dict[str, Any], *, indent: int = 0) -> str:
    lines: list[str] = []
    pad = "  " * indent
    for key, value in payload.items():
        label = str(key).replace("_", " ")
        if isinstance(value, dict):
            lines.append(f"{pad}{label}:")
            lines.append(format_payload(value, indent=indent + 1))
        elif isinstance(value, list):
            lines.append(f"{pad}{label}: {len(value)} 项")
            for item in value[:6]:
                if isinstance(item, dict):
                    lines.append(format_payload(item, indent=indent + 1))
                else:
                    lines.append(f"{pad}  - {item}")
        else:
            lines.append(f"{pad}{label}: {value}")
    return "\n".join(line for line in lines if line)


def short_payload(payload: dict[str, Any]) -> str:
    for key in ("summary", "text", "result", "status"):
        value = payload.get(key)
        if value:
            return str(value)[:160]
    if "action" in payload:
        return str(payload["action"])[:160]
    return format_payload(payload).splitlines()[0] if payload else ""


def _short_ui_text(text: str, limit: int = 160) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def format_computer_action(action: dict[str, Any]) -> str:
    action_type = str(action.get("type") or "")
    prefix = ""
    if action.get("index") is not None:
        prefix = f"{action.get('index')}. "
    intent = str(action.get("intent") or "").strip()
    intent_suffix = f"（{intent}）" if intent else ""
    if action_type == "click":
        button = action.get("button") or "left"
        return (
            f"{prefix}鼠标点击 ({action.get('x')}, {action.get('y')})"
            f" / {button}{intent_suffix}"
        )
    if action_type == "type":
        text = str(action.get("text") or "")
        preview = text[:80] + ("..." if len(text) > 80 else "")
        return f"{prefix}输入文本{intent_suffix}：{preview}"
    if action_type == "submit_text":
        text = str(action.get("text") or "")
        preview = text[:80] + ("..." if len(text) > 80 else "")
        target = (
            f" ({action.get('x')}, {action.get('y')})"
            if action.get("x") is not None and action.get("y") is not None
            else ""
        )
        return f"{prefix}提交文本{target}{intent_suffix}: {preview}"
    if action_type in {"keypress", "key"}:
        keys = action.get("keys") or action.get("key") or ""
        return f"{prefix}按键{intent_suffix}：{keys}"
    if action_type == "wait":
        return f"{prefix}等待 {action.get('seconds', '')} 秒"
    if action_type == "move":
        return f"{prefix}鼠标移动到 ({action.get('x')}, {action.get('y')})"
    if action_type == "scroll":
        return f"{prefix}滚动：dx={action.get('dx', 0)}, dy={action.get('dy', 0)}"
    return f"{prefix}{action_type or '电脑动作'}"


def format_asset(asset: ReferenceAsset) -> str:
    return "\n".join(
        [
            f"引用：{asset.citation}",
            f"类型：{asset.kind}",
            f"标题：{asset.title}",
            f"路径：{asset.path}",
            f"说明：{asset.description}",
        ]
    )


def _point_in_rect(x: int, y: int, bounds: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bounds
    return x1 <= x <= x2 and y1 <= y <= y2


def unique_app_id(app_name: str, existing: dict[str, TargetAppConfig]) -> str:
    base = "".join(ch.lower() if ch.isalnum() else "_" for ch in app_name)
    base = "_".join(part for part in base.split("_") if part) or "app"
    app_id = f"app_{base}"
    index = 2
    while app_id in existing:
        app_id = f"app_{base}_{index}"
        index += 1
    return app_id
