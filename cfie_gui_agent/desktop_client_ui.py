from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from cfie_gui_agent.desktop_client import (
    DIRECT_COMMAND_LABELS,
    DIRECT_COMMAND_NONE,
    DesktopClientState,
    MacroConfig,
    ReferenceAsset,
    TargetAppConfig,
)
from cfie_gui_agent.jobs import JobState


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
        self._font = font
        self._hovered = False
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
        width = max(self.winfo_width(), 1)
        self.delete("all")
        fill = self._hover_fill if self._hovered else self._fill
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


class GuiAgentDesktopClient(tk.Tk):
    def __init__(self, state: DesktopClientState) -> None:
        super().__init__()
        self.state = state
        self.title("CFIE GUI Agent")
        self.geometry("1440x860")
        self.minsize(1180, 720)

        self.colors = {
            "bg": "#fbfaf8",
            "sidebar": "#f5f1ed",
            "sidebar_hover": "#ebe6e1",
            "sidebar_selected": "#e8e2dc",
            "surface": "#ffffff",
            "surface_soft": "#f7f5f2",
            "surface_hover": "#f1efeb",
            "ink": "#252525",
            "muted": "#77736f",
            "muted_2": "#aaa39b",
            "line": "#e6e0d9",
            "line_soft": "#f0ebe5",
            "brand": "#111827",
            "brand_soft": "#f0eee9",
            "accent": "#ff6b2b",
            "accent_soft": "#fff0e8",
            "success": "#0f9f7a",
        }

        self.selected_app_id = tk.StringVar(value=self._first_app_id())
        self.selected_request_id = tk.StringVar(value="")
        self.inspector_visible = tk.BooleanVar(value=True)
        self.direct_command_label = tk.StringVar(
            value=DIRECT_COMMAND_LABELS[DIRECT_COMMAND_NONE]
        )
        self.decision_type = tk.StringVar(value="人工回复")

        self._setup_style()
        self._build_layout()
        self.refresh_all()
        self.after(2000, self._periodic_refresh)

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
            width=9,
            background="#d7d7d7",
            troughcolor=self.colors["surface"],
            bordercolor=self.colors["surface"],
            lightcolor="#d7d7d7",
            darkcolor="#d7d7d7",
            arrowcolor=self.colors["surface"],
            relief="flat",
            borderwidth=0,
        )
        style.map(
            "Modern.Vertical.TScrollbar",
            background=[("active", "#c8c8c8")],
        )

    def _build_layout(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_chat_area()
        self._build_inspector()

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
            command=self._add_app_dialog,
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
        ttk.Button(bottom, text="设置", command=self._open_settings).grid(
            row=0, column=0, sticky="ew"
        )
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
        ttk.Button(
            header,
            text="任务定义",
            command=self._edit_selected_app,
        ).grid(row=0, column=1, rowspan=2, sticky="e", padx=(8, 0))
        ttk.Button(
            header,
            text="检查器",
            command=self._toggle_inspector,
        ).grid(row=0, column=2, rowspan=2, sticky="e", padx=(8, 0))

        task_def_button = CanvasButton(
            header,
            text="任务定义",
            command=self._edit_selected_app,
            fill=self.colors["surface"],
            hover_fill=self.colors["surface_hover"],
            foreground=self.colors["ink"],
            outline=self.colors["line"],
            radius=13,
            height=36,
            width=104,
            canvas_bg=self.colors["surface"],
            font=("Microsoft YaHei UI", 9, "normal"),
        )
        task_def_button.grid(row=0, column=1, rowspan=2, sticky="e", padx=(8, 0))
        task_def_button.after_idle(task_def_button.raise_widget)
        inspector_button = CanvasButton(
            header,
            text="检查器",
            command=self._toggle_inspector,
            fill=self.colors["surface"],
            hover_fill=self.colors["surface_hover"],
            foreground=self.colors["ink"],
            outline=self.colors["line"],
            radius=13,
            height=36,
            width=92,
            canvas_bg=self.colors["surface"],
            font=("Microsoft YaHei UI", 9, "normal"),
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
        scroll = ttk.Scrollbar(
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

        composer = ttk.Frame(self.chat_frame, style="Surface.TFrame", padding=(18, 12))
        composer.grid(row=2, column=0, sticky="ew")
        composer.columnconfigure(0, weight=1)
        composer.columnconfigure(1, weight=0)
        self.composer_mode_text = tk.StringVar(value="向当前 APP 发送输入")
        self.send_button_text = tk.StringVar(value="发送给 Agent")
        toolbar = ttk.Frame(composer, style="Surface.TFrame")
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        toolbar.columnconfigure(0, weight=1)
        ttk.Label(
            toolbar,
            textvariable=self.composer_mode_text,
            style="Hint.TLabel",
        ).grid(row=0, column=0, sticky="w")
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
        composer_text_shell.grid(row=1, column=0, sticky="ew")
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
                width=max(40, event.width - 24),
                height=80,
            )

        composer_text_shell.bind("<Configure>", redraw_composer_text)
        ttk.Button(
            composer,
            textvariable=self.send_button_text,
            style="Primary.TButton",
            command=self._submit_user_message,
        ).grid(row=1, column=1, sticky="se", padx=(10, 0))
        send_button = CanvasButton(
            composer,
            textvariable=self.send_button_text,
            command=self._submit_user_message,
            fill=self.colors["brand"],
            hover_fill="#2b313c",
            foreground="#ffffff",
            outline=self.colors["brand"],
            radius=14,
            height=42,
            width=118,
            canvas_bg=self.colors["surface"],
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        send_button.grid(row=1, column=1, sticky="se", padx=(10, 0))
        send_button.after_idle(send_button.raise_widget)

    def _build_inspector(self) -> None:
        self.inspector = ttk.Frame(self, style="Surface.TFrame", padding=(14, 14))
        self.inspector.grid(row=0, column=2, sticky="nsew")
        self.inspector.columnconfigure(0, weight=1)
        self.inspector.rowconfigure(2, weight=1)

        header = ttk.Frame(self.inspector, style="Surface.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="检查器", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(header, text="收起", command=self._toggle_inspector).grid(
            row=0, column=1, sticky="e"
        )
        ttk.Label(
            self.inspector,
            text="模型与 harness 的交互历史、素材和人工请求。",
            style="Hint.TLabel",
            wraplength=330,
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))

        collapse_button = CanvasButton(
            header,
            text="收起",
            command=self._toggle_inspector,
            fill=self.colors["surface"],
            hover_fill=self.colors["surface_hover"],
            foreground=self.colors["ink"],
            outline=self.colors["line"],
            radius=13,
            height=34,
            width=76,
            canvas_bg=self.colors["surface"],
            font=("Microsoft YaHei UI", 9, "normal"),
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
        inspector_scroll = ttk.Scrollbar(
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
                width=event.width,
            ),
        )

        inspector_detail_shell = tk.Canvas(
            self.inspector,
            height=154,
            bg=self.colors["surface"],
            highlightthickness=0,
            borderwidth=0,
        )
        inspector_detail_shell.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.inspector_detail = tk.Text(
            inspector_detail_shell,
            height=12,
            wrap="word",
            bg="#fbfaf8",
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
                fill="#fbfaf8",
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

    def refresh_all(self) -> None:
        self.refresh_apps()
        self.refresh_header()
        self.refresh_messages()
        self.refresh_inspector()
        self.refresh_composer()

    def refresh_apps(self) -> None:
        current = self.selected_app_id.get()
        for child in self.app_list_frame.winfo_children():
            child.destroy()
        app_ids = list(self.state.target_apps)
        for app_id in app_ids:
            config = self.state.target_apps[app_id]
            waiting = self._waiting_count_for_job(config.job_id)
            self._add_app_card(app_id=app_id, config=config, waiting=waiting)
        if current not in self.state.target_apps and app_ids:
            current = app_ids[0]
            self.selected_app_id.set(current)

    def refresh_header(self) -> None:
        config = self._selected_config()
        if config is None:
            self.app_title.configure(text="未选择 APP")
            self.app_subtitle.configure(text="左侧新建或选择应用")
            return
        refs = len(config.reference_assets)
        macros = len(self._macros_for_app(config.app_id))
        self.app_title.configure(text=config.app_name)
        self.app_subtitle.configure(
            text=f"任务定义 · {refs} 个素材引用 · {macros} 个可用宏"
        )

    def refresh_messages(self) -> None:
        for child in self.messages_frame.winfo_children():
            child.destroy()
        config = self._selected_config()
        if config is None:
            self._add_empty_message()
            return
        self._add_message(
            role="system",
            title="任务定义",
            text=config.task_description.strip() or "尚未填写任务定义。",
            align="left",
        )
        if config.reference_assets:
            text = "\n".join(
                f"{asset.citation}  {asset.title or Path(asset.path).name}"
                for asset in config.reference_assets
            )
            self._add_message(
                role="asset",
                title="素材引用",
                text=text,
                align="left",
            )
        for item in self.state.human_loop.list_requests(include_completed=True):
            request = item["request"]
            metadata = request.get("metadata") or {}
            if metadata.get("job_id") != config.job_id:
                continue
            title = "人工介入" if item["status"] != "resolved" else "人工回复"
            text = self._human_request_text(item)
            self._add_message(role="human", title=title, text=text, align="right")
        for event in self.state.trace_store.events[-12:]:
            self._add_message(
                role="trace",
                title=self._trace_kind_label(event.kind),
                text=short_payload(event.payload),
                align="left",
            )
        self.after_idle(self._scroll_messages_to_bottom)

    def refresh_composer(self) -> None:
        pending = self._current_pending_request()
        if pending is None:
            self.composer_mode_text.set("向当前 APP 发送输入")
            self.send_button_text.set("发送给 Agent")
            self.direct_command_combo.configure(state="disabled")
            self.direct_command_label.set(DIRECT_COMMAND_LABELS[DIRECT_COMMAND_NONE])
            return
        request = pending["request"]
        self.composer_mode_text.set(
            f"正在处理：{str(request.get('question') or '')[:36]}"
        )
        self.send_button_text.set("提交处理")
        self.direct_command_combo.configure(state="readonly")

    def refresh_inspector(self) -> None:
        for child in self.inspector_list_frame.winfo_children():
            child.destroy()
        config = self._selected_config()
        if config is not None:
            self._add_inspector_card(
                item_id="task_definition",
                kind="任务",
                title="任务定义",
                summary=config.task_description[:80] or "未填写",
            )
            for asset in config.reference_assets:
                self._add_inspector_card(
                    item_id=f"asset:{asset.asset_id}",
                    kind=asset.kind,
                    title=asset.title or Path(asset.path).name,
                    summary=asset.citation,
                )
        for index, item in enumerate(
            self.state.human_loop.list_requests(include_completed=True)
        ):
            request = item["request"]
            self._add_inspector_card(
                item_id=f"human:{index}",
                kind=self._status_label(item["status"]),
                title="人工介入",
                summary=request.get("question") or "",
            )
        for index, event in enumerate(self.state.trace_store.events[-30:]):
            self._add_inspector_card(
                item_id=f"trace:{index}",
                kind=self._trace_kind_label(event.kind),
                title=self._trace_kind_label(event.kind),
                summary=short_payload(event.payload),
            )
        self._set_text(self.inspector_detail, "选择左侧记录查看详情。")

    def _periodic_refresh(self) -> None:
        self.refresh_apps()
        self.refresh_header()
        self.refresh_messages()
        self.refresh_inspector()
        self.refresh_composer()
        self.after(2000, self._periodic_refresh)

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

    def _add_message(self, *, role: str, title: str, text: str, align: str) -> None:
        row = len(self.messages_frame.winfo_children())
        outer = ttk.Frame(self.messages_frame, style="Surface.TFrame", padding=(18, 8))
        outer.grid(row=row, column=0, sticky="ew")
        outer.columnconfigure(0, weight=1)
        canvas_width = min(780, max(420, self.chat_canvas.winfo_width() - 180))
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
        bubble.create_text(
            18,
            13,
            text=title,
            anchor="w",
            justify="left",
            fill=self.colors["muted"],
            font=title_font,
        )
        body = bubble.create_text(
            18,
            34,
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

    def _role_color(self, role: str) -> str:
        return {
            "human": self.colors["accent_soft"],
            "asset": "#eef8f2",
            "trace": "#f7f5f2",
            "system": "#f7f5f2",
        }.get(role, "#f7f5f2")

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
        waiting: int,
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
            subtitle_text = "待处理 " + str(waiting) if waiting else "就绪"
            card.create_text(
                text_x,
                42,
                text=subtitle_text,
                anchor="w",
                fill=self.colors["accent"] if waiting else self.colors["muted"],
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
            height=82,
            bg=self.colors["surface"],
            highlightthickness=0,
            borderwidth=0,
        )
        card.pack(fill="x", pady=(0, 8))

        def redraw(_event: tk.Event[Any] | None = None) -> None:
            width = max(card.winfo_width(), 280)
            card.delete("all")
            draw_rounded_rect(
                card,
                2,
                2,
                width - 2,
                80,
                14,
                fill=self.colors["surface_soft"],
                outline=self.colors["line_soft"],
                width=1,
            )
            draw_rounded_rect(
                card,
                12,
                12,
                72,
                34,
                10,
                fill=self.colors["brand_soft"],
                outline=self.colors["brand_soft"],
            )
            card.create_text(
                42,
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
                fill=self.colors["ink"],
                font=tkfont.Font(family="Microsoft YaHei UI", size=9, weight="bold"),
            )
            card.create_text(
                14,
                48,
                text=summary,
                anchor="nw",
                width=width - 28,
                fill=self.colors["muted"],
                font=tkfont.Font(family="Microsoft YaHei UI", size=9),
            )

        card.bind("<Configure>", redraw)
        card.bind("<Button-1>", lambda _event, value=item_id: self._select_inspector_item(value))
        redraw()

    def _on_messages_configure(self, _event: tk.Event[Any]) -> None:
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event[Any]) -> None:
        self.chat_canvas.itemconfigure(self.messages_window, width=event.width)

    def _scroll_messages_to_bottom(self) -> None:
        self.chat_canvas.yview_moveto(1.0)

    def _select_app(self, app_id: str) -> None:
        self.selected_app_id.set(app_id)
        self.refresh_apps()
        self.refresh_header()
        self.refresh_messages()
        self.refresh_inspector()
        self.refresh_composer()

    def _show_app_menu(self, event: tk.Event[Any], app_id: str | None = None) -> None:
        if app_id is not None:
            self.selected_app_id.set(app_id)
            self.refresh_apps()
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="编辑任务定义", command=self._edit_selected_app)
        menu.add_command(label="复制 APP ID", command=self._copy_selected_app_id)
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
        self.status_text.set(f"已添加：{app_name}")
        self.refresh_all()

    def _edit_selected_app(self) -> None:
        config = self._selected_config()
        if config is None:
            messagebox.showinfo("没有 APP", "请先新建或选择一个 APP。")
            return
        dialog = AppConfigDialog(self, title="任务定义", config=config)
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
        self.status_text.set("任务定义已更新")
        self.refresh_all()

    def _copy_selected_app_id(self) -> None:
        config = self._selected_config()
        if config is None:
            return
        self.clipboard_clear()
        self.clipboard_append(config.app_id)
        self.status_text.set("已复制 APP ID")

    def _open_settings(self) -> None:
        SettingsDialog(self, self.state, self.selected_app_id.get())
        self.refresh_all()

    def _toggle_inspector(self) -> None:
        if self.inspector_visible.get():
            self.inspector.grid_remove()
            self.inspector_visible.set(False)
        else:
            self.inspector.grid(row=0, column=2, sticky="nsew")
            self.inspector_visible.set(True)

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

    def _on_inspector_select(self, _event: tk.Event[Any]) -> None:
        return

    def _select_inspector_item(self, selected: str) -> None:
        if not selected:
            return
        config = self._selected_config()
        if selected == "task_definition" and config is not None:
            self._set_text(self.inspector_detail, config.task_description)
            return
        if selected.startswith("asset:") and config is not None:
            asset_id = selected.split(":", 1)[1]
            for asset in config.reference_assets:
                if asset.asset_id == asset_id:
                    self._set_text(self.inspector_detail, format_asset(asset))
                    return
        if selected.startswith("human:"):
            index = int(selected.split(":", 1)[1])
            items = list(self.state.human_loop.list_requests(include_completed=True))
            if 0 <= index < len(items):
                self._set_text(self.inspector_detail, self._human_request_text(items[index]))
            return
        if selected.startswith("trace:"):
            index = int(selected.split(":", 1)[1])
            events = self.state.trace_store.events[-30:]
            if 0 <= index < len(events):
                self._set_text(self.inspector_detail, format_payload(events[index].payload))

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

        ttk.Label(frame, text="任务定义", style="Section.TLabel").grid(
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
            columns=("scope", "sequence", "description"),
            show="headings",
        )
        self.macro_tree.heading("scope", text="范围")
        self.macro_tree.heading("sequence", text="按键")
        self.macro_tree.heading("description", text="说明")
        self.macro_tree.column("scope", width=80, anchor="center")
        self.macro_tree.column("sequence", width=120)
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

    def _refresh_macros(self) -> None:
        for item in self.macro_tree.get_children():
            self.macro_tree.delete(item)
        for macro in self.state.action_macros.macros.values():
            scope = "全局" if macro.metadata.get("scope") == "global" else "当前 APP"
            self.macro_tree.insert(
                "",
                tk.END,
                iid=macro.name,
                values=(
                    scope,
                    macro.metadata.get("sequence", ""),
                    macro.description,
                ),
            )


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


def unique_app_id(app_name: str, existing: dict[str, TargetAppConfig]) -> str:
    base = "".join(ch.lower() if ch.isalnum() else "_" for ch in app_name)
    base = "_".join(part for part in base.split("_") if part) or "app"
    app_id = f"app_{base}"
    index = 2
    while app_id in existing:
        app_id = f"app_{base}_{index}"
        index += 1
    return app_id
