from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from cfie_gui_agent.desktop_client import (
    DIRECT_COMMAND_LABELS,
    DIRECT_COMMAND_NONE,
    DesktopClientState,
    MacroConfig,
    TargetAppConfig,
)


class GuiAgentDesktopClient(tk.Tk):
    def __init__(self, state: DesktopClientState) -> None:
        super().__init__()
        self.state = state
        self.title("CFIE GUI Agent")
        self.geometry("1280x820")
        self.minsize(1100, 720)

        self.selected_app_id = tk.StringVar(value=self._first_app_id())
        self.selected_request_id = tk.StringVar(value="")
        self.direct_command_label = tk.StringVar(
            value=DIRECT_COMMAND_LABELS[DIRECT_COMMAND_NONE]
        )
        self.decision_type = tk.StringVar(value="人工回复")

        self._setup_style()
        self._build_layout()
        self.refresh_all()
        self.after(2000, self._periodic_refresh)

    def _setup_style(self) -> None:
        self.configure(bg="#edf2f8")
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#edf2f8")
        style.configure("Surface.TFrame", background="#ffffff")
        style.configure("Sidebar.TFrame", background="#101827")
        style.configure(
            "Header.TLabel",
            background="#101827",
            foreground="#ffffff",
            font=("Microsoft YaHei UI", 16, "bold"),
        )
        style.configure(
            "SubHeader.TLabel",
            background="#101827",
            foreground="#a9b7cc",
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Section.TLabel",
            background="#ffffff",
            foreground="#172033",
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        style.configure(
            "Hint.TLabel",
            background="#ffffff",
            foreground="#607089",
            font=("Microsoft YaHei UI", 9),
        )
        style.configure("TButton", font=("Microsoft YaHei UI", 9), padding=(10, 6))
        style.configure(
            "Accent.TButton",
            font=("Microsoft YaHei UI", 9, "bold"),
            foreground="#ffffff",
            background="#2563eb",
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#1d4ed8"), ("pressed", "#1e40af")],
        )
        style.configure("Treeview", rowheight=26, font=("Microsoft YaHei UI", 9))
        style.configure(
            "Treeview.Heading",
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.configure("TNotebook", background="#edf2f8", borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            padding=(14, 8),
            font=("Microsoft YaHei UI", 10),
        )

    def _build_layout(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, style="Sidebar.TFrame", padding=(18, 14))
        header.grid(row=0, column=0, columnspan=2, sticky="nsew")
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text="CFIE GUI Agent 桌面客户端",
            style="Header.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="配置 APP 任务、宏、执行轨迹和人工介入",
            style="SubHeader.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(
            header,
            text="创建人工阻塞示例",
            style="Accent.TButton",
            command=self._create_demo_human_request,
        ).grid(row=0, column=1, rowspan=2, sticky="e", padx=(12, 0))

        sidebar = ttk.Frame(self, style="Sidebar.TFrame", padding=(14, 16))
        sidebar.grid(row=1, column=0, sticky="nsew")
        sidebar.rowconfigure(2, weight=1)
        ttk.Label(
            sidebar,
            text="Target APP",
            style="Header.TLabel",
            font=("Microsoft YaHei UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            sidebar,
            text="一个 APP 对应一个 JOB",
            style="SubHeader.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 10))

        self.app_listbox = tk.Listbox(
            sidebar,
            width=28,
            height=14,
            activestyle="none",
            bg="#172033",
            fg="#e8eef8",
            selectbackground="#2563eb",
            selectforeground="#ffffff",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#334155",
            font=("Microsoft YaHei UI", 10),
        )
        self.app_listbox.grid(row=2, column=0, sticky="nsew")
        self.app_listbox.bind("<<ListboxSelect>>", self._on_app_select)

        side_buttons = ttk.Frame(sidebar, style="Sidebar.TFrame")
        side_buttons.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        side_buttons.columnconfigure(0, weight=1)
        side_buttons.columnconfigure(1, weight=1)
        ttk.Button(side_buttons, text="新增 APP", command=self._add_app_dialog).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(side_buttons, text="刷新", command=self.refresh_all).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )

        self.status_text = tk.StringVar(value="就绪")
        ttk.Label(
            sidebar,
            textvariable=self.status_text,
            style="SubHeader.TLabel",
            wraplength=220,
        ).grid(row=4, column=0, sticky="ew", pady=(16, 0))

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=1, column=1, sticky="nsew", padx=12, pady=12)
        self._build_task_tab()
        self._build_human_tab()
        self._build_trace_tab()
        self._build_jobs_tab()
        self._build_macro_tab()

    def _build_task_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=14)
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(1, weight=1)
        self.notebook.add(tab, text="APP 与任务描述")

        form = ttk.Frame(tab, style="Surface.TFrame", padding=14)
        form.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        form.columnconfigure(1, weight=1)
        form.rowconfigure(5, weight=1)

        ttk.Label(form, text="当前 APP 配置", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(form, text="APP 名称", style="Hint.TLabel").grid(
            row=1, column=0, sticky="w", pady=(12, 0)
        )
        self.app_name_entry = ttk.Entry(form)
        self.app_name_entry.grid(row=1, column=1, sticky="ew", pady=(12, 0))
        ttk.Label(form, text="JOB ID", style="Hint.TLabel").grid(
            row=2, column=0, sticky="w", pady=(10, 0)
        )
        self.job_id_value = ttk.Label(form, text="", style="Hint.TLabel")
        self.job_id_value.grid(row=2, column=1, sticky="w", pady=(10, 0))

        ttk.Label(
            form,
            text="任务描述、场景规则与图片引用",
            style="Section.TLabel",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(16, 4))
        ttk.Label(
            form,
            text="图片/视频用右侧引用插入，例如 [image:map_main]。",
            style="Hint.TLabel",
        ).grid(row=4, column=0, columnspan=2, sticky="w")
        self.task_text = tk.Text(
            form,
            height=22,
            wrap="word",
            undo=True,
            bg="#fbfdff",
            relief="solid",
            borderwidth=1,
            font=("Microsoft YaHei UI", 10),
        )
        self.task_text.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(8, 8))
        ttk.Button(
            form,
            text="保存任务描述",
            style="Accent.TButton",
            command=self._save_task_description,
        ).grid(row=6, column=1, sticky="e")

        refs = ttk.Frame(tab, style="Surface.TFrame", padding=14)
        refs.grid(row=0, column=1, rowspan=2, sticky="nsew")
        refs.columnconfigure(0, weight=1)
        refs.rowconfigure(2, weight=1)
        ttk.Label(refs, text="参考图片 / 视频", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            refs,
            text="每个素材都有可插入到任务描述里的引用。",
            style="Hint.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 10))
        self.asset_tree = ttk.Treeview(
            refs,
            columns=("kind", "citation", "title"),
            show="headings",
            selectmode="browse",
        )
        self.asset_tree.heading("kind", text="类型")
        self.asset_tree.heading("citation", text="引用")
        self.asset_tree.heading("title", text="标题")
        self.asset_tree.column("kind", width=56, anchor="center")
        self.asset_tree.column("citation", width=130)
        self.asset_tree.column("title", width=160)
        self.asset_tree.grid(row=2, column=0, sticky="nsew")
        asset_buttons = ttk.Frame(refs, style="Surface.TFrame")
        asset_buttons.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        for idx in range(4):
            asset_buttons.columnconfigure(idx, weight=1)
        ttk.Button(asset_buttons, text="添加图片", command=lambda: self._add_asset("image")).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(asset_buttons, text="添加视频", command=lambda: self._add_asset("video")).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(asset_buttons, text="插入引用", command=self._insert_asset_citation).grid(
            row=0, column=2, sticky="ew", padx=4
        )
        ttk.Button(asset_buttons, text="复制引用", command=self._copy_asset_citation).grid(
            row=0, column=3, sticky="ew", padx=(4, 0)
        )

    def _build_human_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=14)
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(1, weight=1)
        self.notebook.add(tab, text="人工介入")

        left = ttk.Frame(tab, style="Surface.TFrame", padding=14)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)
        ttk.Label(left, text="阻塞请求", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            left,
            text="客户端、微信等 channel 共享同一份请求状态；认领后其他端不能回复。",
            style="Hint.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 10))
        self.human_tree = ttk.Treeview(
            left,
            columns=("status", "job", "task", "question"),
            show="headings",
            selectmode="browse",
        )
        for key, label, width in (
            ("status", "状态", 78),
            ("job", "JOB", 110),
            ("task", "子任务", 120),
            ("question", "问题", 360),
        ):
            self.human_tree.heading(key, text=label)
            self.human_tree.column(key, width=width, stretch=(key == "question"))
        self.human_tree.grid(row=2, column=0, sticky="nsew")
        self.human_tree.bind("<<TreeviewSelect>>", self._on_human_select)

        request_buttons = ttk.Frame(left, style="Surface.TFrame")
        request_buttons.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        for idx in range(4):
            request_buttons.columnconfigure(idx, weight=1)
        ttk.Button(request_buttons, text="认领", command=self._claim_request).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(request_buttons, text="释放", command=self._release_request).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(
            request_buttons,
            text="创建示例",
            command=self._create_demo_human_request,
        ).grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Button(request_buttons, text="刷新", command=self.refresh_human).grid(
            row=0, column=3, sticky="ew", padx=(4, 0)
        )

        right = ttk.Frame(tab, style="Surface.TFrame", padding=14)
        right.grid(row=0, column=1, rowspan=2, sticky="nsew")
        right.columnconfigure(1, weight=1)
        right.rowconfigure(7, weight=1)
        ttk.Label(right, text="人工解锁阻塞", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        self.request_detail = tk.Text(
            right,
            height=8,
            wrap="word",
            bg="#f8fafc",
            relief="solid",
            borderwidth=1,
            font=("Microsoft YaHei UI", 9),
        )
        self.request_detail.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 12))
        self.request_detail.configure(state="disabled")

        ttk.Label(right, text="处理类型", style="Hint.TLabel").grid(row=2, column=0, sticky="w")
        self.decision_combo = ttk.Combobox(
            right,
            textvariable=self.decision_type,
            values=("人工回复", "风险确认", "更改路径", "暂停/取消", "补充约束"),
            state="readonly",
        )
        self.decision_combo.grid(row=2, column=1, sticky="ew", pady=4)

        ttk.Label(right, text="直接命令", style="Hint.TLabel").grid(row=3, column=0, sticky="w")
        self.direct_command_combo = ttk.Combobox(
            right,
            textvariable=self.direct_command_label,
            values=tuple(DIRECT_COMMAND_LABELS.values()),
            state="readonly",
        )
        self.direct_command_combo.grid(row=3, column=1, sticky="ew", pady=4)

        ttk.Label(right, text="你的输入", style="Hint.TLabel").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        self.manager_input = tk.Text(
            right,
            height=7,
            wrap="word",
            bg="#fbfdff",
            relief="solid",
            borderwidth=1,
            font=("Microsoft YaHei UI", 10),
        )
        self.manager_input.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(4, 10))

        ttk.Label(right, text="新增约束 / 注意事项", style="Hint.TLabel").grid(
            row=6, column=0, columnspan=2, sticky="w"
        )
        self.constraints_input = tk.Text(
            right,
            height=6,
            wrap="word",
            bg="#fbfdff",
            relief="solid",
            borderwidth=1,
            font=("Microsoft YaHei UI", 10),
        )
        self.constraints_input.grid(row=7, column=0, columnspan=2, sticky="nsew", pady=(4, 10))

        ttk.Button(
            right,
            text="提交给 Agent",
            style="Accent.TButton",
            command=self._submit_human_reply,
        ).grid(row=8, column=1, sticky="e")

    def _build_trace_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=14)
        tab.columnconfigure(0, weight=2)
        tab.columnconfigure(1, weight=3)
        tab.rowconfigure(1, weight=1)
        self.notebook.add(tab, text="执行轨迹")

        left = ttk.Frame(tab, style="Surface.TFrame", padding=14)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)
        ttk.Label(left, text="模型执行轨迹", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            left,
            text="这里用于查看模型输入上下文、思考输出、工具调用与执行结果。",
            style="Hint.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 10))
        self.trace_tree = ttk.Treeview(
            left,
            columns=("kind", "summary"),
            show="headings",
            selectmode="browse",
        )
        self.trace_tree.heading("kind", text="类型")
        self.trace_tree.heading("summary", text="摘要")
        self.trace_tree.column("kind", width=90, anchor="center")
        self.trace_tree.column("summary", width=360)
        self.trace_tree.grid(row=2, column=0, sticky="nsew")
        self.trace_tree.bind("<<TreeviewSelect>>", self._on_trace_select)

        right = ttk.Frame(tab, style="Surface.TFrame", padding=14)
        right.grid(row=0, column=1, rowspan=2, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        ttk.Label(right, text="轨迹详情 / 模型上下文", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.trace_detail = tk.Text(
            right,
            wrap="word",
            bg="#0f172a",
            fg="#e5eefc",
            insertbackground="#ffffff",
            relief="flat",
            font=("Consolas", 10),
        )
        self.trace_detail.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

    def _build_jobs_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=14)
        tab.columnconfigure(0, weight=2)
        tab.columnconfigure(1, weight=3)
        tab.rowconfigure(1, weight=1)
        self.notebook.add(tab, text="JOB 队列")

        left = ttk.Frame(tab, style="Surface.TFrame", padding=14)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        ttk.Label(left, text="JOB 总览", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.job_tree = ttk.Treeview(
            left,
            columns=("app", "status", "running", "runnable", "waiting", "done"),
            show="headings",
            selectmode="browse",
        )
        for key, label, width in (
            ("app", "APP", 110),
            ("status", "状态", 70),
            ("running", "运行", 60),
            ("runnable", "可执行", 70),
            ("waiting", "等人工", 70),
            ("done", "完成", 60),
        ):
            self.job_tree.heading(key, text=label)
            self.job_tree.column(key, width=width, anchor="center")
        self.job_tree.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.job_tree.bind("<<TreeviewSelect>>", self._on_job_select)

        right = ttk.Frame(tab, style="Surface.TFrame", padding=14)
        right.grid(row=0, column=1, rowspan=2, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        ttk.Label(right, text="子任务队列", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.subtask_tree = ttk.Treeview(
            right,
            columns=("queue", "subtask", "goal", "priority"),
            show="headings",
        )
        for key, label, width in (
            ("queue", "队列", 100),
            ("subtask", "子任务", 140),
            ("goal", "目标", 380),
            ("priority", "优先级", 70),
        ):
            self.subtask_tree.heading(key, text=label)
            self.subtask_tree.column(key, width=width, stretch=(key == "goal"))
        self.subtask_tree.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

    def _build_macro_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=14)
        tab.columnconfigure(0, weight=2)
        tab.columnconfigure(1, weight=2)
        tab.rowconfigure(1, weight=1)
        self.notebook.add(tab, text="连续键控宏")

        left = ttk.Frame(tab, style="Surface.TFrame", padding=14)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        ttk.Label(left, text="已注册宏", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.macro_tree = ttk.Treeview(
            left,
            columns=("name", "sequence", "description"),
            show="headings",
        )
        self.macro_tree.heading("name", text="名称")
        self.macro_tree.heading("sequence", text="按键序列")
        self.macro_tree.heading("description", text="工具描述")
        self.macro_tree.column("name", width=130)
        self.macro_tree.column("sequence", width=150)
        self.macro_tree.column("description", width=360)
        self.macro_tree.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        right = ttk.Frame(tab, style="Surface.TFrame", padding=14)
        right.grid(row=0, column=1, rowspan=2, sticky="nsew")
        right.columnconfigure(1, weight=1)
        ttk.Label(right, text="新增宏", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            right,
            text="模型只看到宏名称和描述；客户端负责展开为连续按键。",
            style="Hint.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 12))

        ttk.Label(right, text="名称", style="Hint.TLabel").grid(row=2, column=0, sticky="w")
        self.macro_name = ttk.Entry(right)
        self.macro_name.grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Label(right, text="按键序列", style="Hint.TLabel").grid(row=3, column=0, sticky="w")
        self.macro_sequence = ttk.Entry(right)
        self.macro_sequence.insert(0, "CTRL+A, B")
        self.macro_sequence.grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Label(right, text="操作含义", style="Hint.TLabel").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        self.macro_description = tk.Text(
            right,
            height=8,
            wrap="word",
            bg="#fbfdff",
            relief="solid",
            borderwidth=1,
            font=("Microsoft YaHei UI", 10),
        )
        self.macro_description.grid(row=5, column=0, columnspan=2, sticky="nsew", pady=(4, 10))
        ttk.Button(
            right,
            text="注册宏",
            style="Accent.TButton",
            command=self._register_macro,
        ).grid(row=6, column=1, sticky="e")

    def refresh_all(self) -> None:
        self.refresh_apps()
        self.refresh_assets()
        self.refresh_human()
        self.refresh_jobs()
        self.refresh_macros()
        self.refresh_trace()

    def refresh_apps(self) -> None:
        current = self.selected_app_id.get()
        self.app_listbox.delete(0, tk.END)
        for app_id, config in self.state.target_apps.items():
            self.app_listbox.insert(tk.END, f"{config.app_name}  ({app_id})")
        app_ids = list(self.state.target_apps)
        if current not in self.state.target_apps and app_ids:
            current = app_ids[0]
            self.selected_app_id.set(current)
        if current in self.state.target_apps:
            idx = app_ids.index(current)
            self.app_listbox.selection_clear(0, tk.END)
            self.app_listbox.selection_set(idx)
            self.app_listbox.activate(idx)
            config = self.state.target_apps[current]
            self.app_name_entry.delete(0, tk.END)
            self.app_name_entry.insert(0, config.app_name)
            self.job_id_value.configure(text=config.job_id)
            self.task_text.delete("1.0", tk.END)
            self.task_text.insert("1.0", config.task_description)

    def refresh_assets(self) -> None:
        self._clear_tree(self.asset_tree)
        config = self._selected_config()
        if config is None:
            return
        for asset in config.reference_assets:
            self.asset_tree.insert(
                "",
                tk.END,
                iid=asset.asset_id,
                values=(asset.kind, asset.citation, asset.title),
            )

    def refresh_human(self) -> None:
        selected = self.selected_request_id.get()
        self._clear_tree(self.human_tree)
        for item in self.state.human_loop.list_requests(include_completed=True):
            request = item["request"]
            metadata = request.get("metadata") or {}
            request_id = request["request_id"]
            self.human_tree.insert(
                "",
                tk.END,
                iid=request_id,
                values=(
                    item["status"],
                    metadata.get("job_id") or "",
                    request.get("task_id") or "",
                    request.get("question") or "",
                ),
            )
        if selected and self.human_tree.exists(selected):
            self.human_tree.selection_set(selected)
            self.human_tree.focus(selected)
        self._show_selected_request()

    def refresh_jobs(self) -> None:
        selected = self._selected_tree_iid(self.job_tree)
        self._clear_tree(self.job_tree)
        for job_id, job in self.state.job_board.jobs.items():
            counts = job.queues.counts()
            self.job_tree.insert(
                "",
                tk.END,
                iid=job_id,
                values=(
                    job.target_app,
                    job.status,
                    counts["running"],
                    counts["runnable"] + counts["urgent"],
                    counts["waiting_human"],
                    counts["completed"],
                ),
            )
        if selected and self.job_tree.exists(selected):
            self.job_tree.selection_set(selected)
            self.job_tree.focus(selected)
        elif self.state.job_board.active_job_id and self.job_tree.exists(
            self.state.job_board.active_job_id
        ):
            self.job_tree.selection_set(self.state.job_board.active_job_id)
            self.job_tree.focus(self.state.job_board.active_job_id)
        self._refresh_subtasks()

    def refresh_macros(self) -> None:
        self._clear_tree(self.macro_tree)
        for macro in self.state.action_macros.macros.values():
            self.macro_tree.insert(
                "",
                tk.END,
                iid=macro.name,
                values=(
                    macro.name,
                    macro.metadata.get("sequence", ""),
                    macro.description,
                ),
            )

    def refresh_trace(self) -> None:
        selected = self._selected_tree_iid(self.trace_tree)
        self._clear_tree(self.trace_tree)
        for idx, event in enumerate(self.state.trace_store.events):
            payload = event.payload
            summary = (
                payload.get("summary")
                or payload.get("result")
                or payload.get("status")
                or event.kind
            )
            self.trace_tree.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(event.kind, str(summary)[:120]),
            )
        if selected and self.trace_tree.exists(selected):
            self.trace_tree.selection_set(selected)
            self.trace_tree.focus(selected)
        elif self.state.trace_store.events:
            self.trace_tree.selection_set(str(len(self.state.trace_store.events) - 1))
        self._show_selected_trace()

    def _periodic_refresh(self) -> None:
        self.refresh_human()
        self.refresh_jobs()
        self.after(2000, self._periodic_refresh)

    def _first_app_id(self) -> str:
        try:
            return next(iter(self.state.target_apps))
        except StopIteration:
            return ""

    def _selected_config(self) -> TargetAppConfig | None:
        return self.state.target_apps.get(self.selected_app_id.get())

    def _on_app_select(self, _event: tk.Event[Any]) -> None:
        selection = self.app_listbox.curselection()
        if not selection:
            return
        app_id = list(self.state.target_apps)[selection[0]]
        self.selected_app_id.set(app_id)
        self.refresh_apps()
        self.refresh_assets()

    def _on_human_select(self, _event: tk.Event[Any]) -> None:
        selected = self._selected_tree_iid(self.human_tree)
        if selected:
            self.selected_request_id.set(selected)
        self._show_selected_request()

    def _on_trace_select(self, _event: tk.Event[Any]) -> None:
        self._show_selected_trace()

    def _on_job_select(self, _event: tk.Event[Any]) -> None:
        self._refresh_subtasks()

    def _save_task_description(self) -> None:
        config = self._selected_config()
        if config is None:
            return
        app_name = self.app_name_entry.get().strip()
        description = self.task_text.get("1.0", tk.END).strip()
        self.state.target_apps[config.app_id] = TargetAppConfig(
            app_id=config.app_id,
            app_name=app_name or config.app_name,
            job_id=config.job_id,
            task_description=description,
            reference_assets=config.reference_assets,
            metadata=config.metadata,
        )
        self.status_text.set(f"已保存 {app_name or config.app_name} 的任务描述")
        self.refresh_apps()

    def _add_app_dialog(self) -> None:
        dialog = _SimpleInputDialog(
            self,
            title="新增 Target APP",
            fields=(
                ("APP 名称", "Chrome"),
                ("JOB ID", f"job_{len(self.state.target_apps) + 1}"),
                ("任务描述", "描述这个 APP 的业务场景、规则和约束。"),
            ),
        )
        values = dialog.result
        if values is None:
            return
        app_name = values["APP 名称"].strip()
        job_id = values["JOB ID"].strip()
        if not app_name or not job_id:
            messagebox.showwarning("输入不完整", "APP 名称和 JOB ID 都不能为空。")
            return
        app_id = f"app_{uuid_like(app_name)}_{len(self.state.target_apps) + 1}"
        config = TargetAppConfig(
            app_id=app_id,
            app_name=app_name,
            job_id=job_id,
            task_description=values["任务描述"].strip(),
        )
        self.state.add_target_app(config)
        if job_id not in self.state.job_board.jobs:
            self.state.job_board.add_job(
                JobState(job_id=job_id, target_app=app_name, goal=app_name)
            )
        self.selected_app_id.set(app_id)
        self.refresh_all()

    def _add_asset(self, kind: str) -> None:
        config = self._selected_config()
        if config is None:
            return
        filetypes = (
            [("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("All files", "*.*")]
            if kind == "image"
            else [("Video files", "*.mp4;*.mov;*.avi;*.mkv"), ("All files", "*.*")]
        )
        path = filedialog.askopenfilename(title=f"选择{kind}", filetypes=filetypes)
        if not path:
            return
        asset = self.state.add_reference_asset(
            app_id=config.app_id,
            kind=kind,
            path=path,
            title=Path(path).stem,
        )
        self.status_text.set(f"已添加素材 {asset.citation}")
        self.refresh_assets()

    def _insert_asset_citation(self) -> None:
        citation = self._selected_asset_citation()
        if not citation:
            return
        self.task_text.insert(tk.INSERT, citation)
        self.status_text.set(f"已插入引用 {citation}")

    def _copy_asset_citation(self) -> None:
        citation = self._selected_asset_citation()
        if not citation:
            return
        self.clipboard_clear()
        self.clipboard_append(citation)
        self.status_text.set(f"已复制引用 {citation}")

    def _selected_asset_citation(self) -> str | None:
        config = self._selected_config()
        selected = self._selected_tree_iid(self.asset_tree)
        if config is None or not selected:
            return None
        for asset in config.reference_assets:
            if asset.asset_id == selected:
                return asset.citation
        return None

    def _create_demo_human_request(self) -> None:
        request_id = self.state.create_demo_human_request()
        self.selected_request_id.set(request_id)
        self.refresh_human()
        self.notebook.select(1)
        self.status_text.set("已创建人工介入示例请求")

    def _claim_request(self) -> None:
        request_id = self._require_selected_request_id()
        if not request_id:
            return
        try:
            self.state.human_loop.claim_request(request_id, source="client")
        except Exception as exc:
            messagebox.showerror("认领失败", str(exc))
        self.refresh_human()

    def _release_request(self) -> None:
        request_id = self._require_selected_request_id()
        if not request_id:
            return
        try:
            self.state.human_loop.release_request(request_id, source="client")
        except Exception as exc:
            messagebox.showerror("释放失败", str(exc))
        self.refresh_human()

    def _submit_human_reply(self) -> None:
        request_id = self._require_selected_request_id()
        if not request_id:
            return
        manager_input = self.manager_input.get("1.0", tk.END).strip()
        constraints = self.constraints_input.get("1.0", tk.END).strip()
        direct_command = self._selected_direct_command()
        if not manager_input and direct_command == DIRECT_COMMAND_NONE:
            messagebox.showwarning("缺少输入", "请填写人工输入，或选择一个直接命令。")
            return
        try:
            self.state.human_loop.claim_request(request_id, source="client")
            self.state.submit_structured_human_reply(
                request_id=request_id,
                manager_input=manager_input,
                decision_type=self.decision_type.get(),
                direct_command=direct_command,
                constraints=constraints,
            )
        except Exception as exc:
            messagebox.showerror("提交失败", str(exc))
            return
        self.manager_input.delete("1.0", tk.END)
        self.constraints_input.delete("1.0", tk.END)
        self.status_text.set("已提交人工回复，阻塞请求进入 urgent manager reply")
        self.refresh_all()

    def _register_macro(self) -> None:
        name = self.macro_name.get().strip()
        sequence = self.macro_sequence.get().strip()
        description = self.macro_description.get("1.0", tk.END).strip()
        if not name or not sequence or not description:
            messagebox.showwarning("输入不完整", "名称、按键序列和操作含义都需要填写。")
            return
        try:
            self.state.register_macro(
                MacroConfig(name=name, description=description, sequence=sequence)
            )
        except Exception as exc:
            messagebox.showerror("注册失败", str(exc))
            return
        self.macro_name.delete(0, tk.END)
        self.macro_description.delete("1.0", tk.END)
        self.status_text.set(f"已注册宏 {name}")
        self.refresh_macros()

    def _show_selected_request(self) -> None:
        selected = self._selected_tree_iid(self.human_tree)
        detail = ""
        if selected:
            for item in self.state.human_loop.list_requests(include_completed=True):
                if item["request"]["request_id"] == selected:
                    detail = json.dumps(item, ensure_ascii=False, indent=2)
                    break
        self.request_detail.configure(state="normal")
        self.request_detail.delete("1.0", tk.END)
        self.request_detail.insert("1.0", detail)
        self.request_detail.configure(state="disabled")

    def _show_selected_trace(self) -> None:
        selected = self._selected_tree_iid(self.trace_tree)
        payload: dict[str, Any]
        if selected is not None and selected.isdigit():
            idx = int(selected)
            if 0 <= idx < len(self.state.trace_store.events):
                payload = self.state.trace_store.events[idx].to_dict()
            else:
                payload = self.state.to_dict()
        else:
            payload = self.state.to_dict()
        self.trace_detail.delete("1.0", tk.END)
        self.trace_detail.insert("1.0", json.dumps(payload, ensure_ascii=False, indent=2))

    def _refresh_subtasks(self) -> None:
        self._clear_tree(self.subtask_tree)
        job_id = self._selected_tree_iid(self.job_tree) or self.state.job_board.active_job_id
        if not job_id or job_id not in self.state.job_board.jobs:
            return
        queues = self.state.job_board.jobs[job_id].queues.to_dict()
        for queue_name, queue_value in queues.items():
            if queue_name == "running" and queue_value:
                self._insert_subtask_row(queue_name, queue_value)
            elif isinstance(queue_value, list):
                for subtask in queue_value:
                    self._insert_subtask_row(queue_name, subtask)
            elif isinstance(queue_value, dict):
                for subtask in queue_value.values():
                    self._insert_subtask_row(queue_name, subtask)

    def _insert_subtask_row(self, queue_name: str, subtask: dict[str, Any]) -> None:
        self.subtask_tree.insert(
            "",
            tk.END,
            values=(
                queue_name,
                subtask.get("subtask_id", ""),
                subtask.get("goal", ""),
                subtask.get("priority", ""),
            ),
        )

    def _selected_direct_command(self) -> str:
        label = self.direct_command_label.get()
        for key, value in DIRECT_COMMAND_LABELS.items():
            if value == label:
                return key
        return DIRECT_COMMAND_NONE

    def _require_selected_request_id(self) -> str | None:
        request_id = self._selected_tree_iid(self.human_tree) or self.selected_request_id.get()
        if not request_id:
            messagebox.showwarning("未选择请求", "请先选择一个人工阻塞请求。")
            return None
        return request_id

    @staticmethod
    def _selected_tree_iid(tree: ttk.Treeview) -> str | None:
        selection = tree.selection()
        return str(selection[0]) if selection else None

    @staticmethod
    def _clear_tree(tree: ttk.Treeview) -> None:
        for item in tree.get_children():
            tree.delete(item)


class _SimpleInputDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        *,
        title: str,
        fields: tuple[tuple[str, str], ...],
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.transient(parent)
        self.grab_set()
        self.resizable(False, False)
        self.result: dict[str, str] | None = None
        self.entries: dict[str, tk.Text | ttk.Entry] = {}
        frame = ttk.Frame(self, padding=14)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)
        for row, (label, default) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=5)
            if label == "任务描述":
                widget = tk.Text(frame, width=44, height=5, wrap="word")
                widget.insert("1.0", default)
            else:
                widget = ttk.Entry(frame, width=44)
                widget.insert(0, default)
            widget.grid(row=row, column=1, sticky="ew", pady=5)
            self.entries[label] = widget
        buttons = ttk.Frame(frame)
        buttons.grid(row=len(fields), column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="取消", command=self.destroy).grid(row=0, column=0, padx=4)
        ttk.Button(buttons, text="确定", command=self._ok).grid(row=0, column=1, padx=4)
        self.wait_window(self)

    def _ok(self) -> None:
        result: dict[str, str] = {}
        for label, widget in self.entries.items():
            if isinstance(widget, tk.Text):
                result[label] = widget.get("1.0", tk.END).strip()
            else:
                result[label] = widget.get().strip()
        self.result = result
        self.destroy()


def uuid_like(text: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized or "app"
