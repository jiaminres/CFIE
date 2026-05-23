from __future__ import annotations

import tkinter as tk
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


class GuiAgentDesktopClient(tk.Tk):
    def __init__(self, state: DesktopClientState) -> None:
        super().__init__()
        self.state = state
        self.title("CFIE GUI Agent")
        self.geometry("1320x840")
        self.minsize(1180, 760)

        self.colors = {
            "bg": "#f4f7fb",
            "surface": "#ffffff",
            "surface_soft": "#f8fbff",
            "ink": "#142033",
            "muted": "#64748b",
            "line": "#d9e3f1",
            "brand": "#2f6df6",
            "brand_dark": "#1e4fd1",
            "mint": "#1fbf9a",
            "amber": "#f59e0b",
            "purple": "#7c3aed",
            "sidebar": "#102033",
            "sidebar_soft": "#172a44",
        }

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
        self.configure(bg=self.colors["bg"])
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("Card.TFrame", background=self.colors["surface"])
        style.configure("Soft.TFrame", background=self.colors["surface_soft"])
        style.configure("Sidebar.TFrame", background=self.colors["sidebar"])
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
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background=self.colors["surface"],
            foreground=self.colors["ink"],
            font=("Microsoft YaHei UI", 9),
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
            foreground="#ffffff",
            font=("Microsoft YaHei UI", 15, "bold"),
        )
        style.configure(
            "SidebarHint.TLabel",
            background=self.colors["sidebar"],
            foreground="#a9b8ce",
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Metric.TLabel",
            background=self.colors["surface_soft"],
            foreground=self.colors["ink"],
            font=("Microsoft YaHei UI", 12, "bold"),
        )
        style.configure("TButton", font=("Microsoft YaHei UI", 9), padding=(12, 7))
        style.configure(
            "Primary.TButton",
            font=("Microsoft YaHei UI", 9, "bold"),
            foreground="#ffffff",
            background=self.colors["brand"],
            bordercolor=self.colors["brand"],
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.colors["brand_dark"])],
        )
        style.configure(
            "Ghost.TButton",
            font=("Microsoft YaHei UI", 9),
            foreground=self.colors["ink"],
            background="#eef4ff",
            bordercolor="#c8d8ff",
        )
        style.configure("Treeview", rowheight=28, font=("Microsoft YaHei UI", 9))
        style.configure(
            "Treeview.Heading",
            font=("Microsoft YaHei UI", 9, "bold"),
            background="#eef2f7",
        )

    def _build_layout(self) -> None:
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, minsize=360)
        self.rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_workspace()
        self._build_human_panel()

    def _build_sidebar(self) -> None:
        sidebar = ttk.Frame(self, style="Sidebar.TFrame", padding=(16, 18))
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.rowconfigure(3, weight=1)

        ttk.Label(sidebar, text="CFIE Agent", style="SidebarTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            sidebar,
            text="本地自动化工作台",
            style="SidebarHint.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 18))

        action_bar = ttk.Frame(sidebar, style="Sidebar.TFrame")
        action_bar.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        action_bar.columnconfigure(0, weight=1)
        action_bar.columnconfigure(1, weight=1)
        ttk.Button(
            action_bar,
            text="新建 APP",
            style="Primary.TButton",
            command=self._add_app_dialog,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(
            action_bar,
            text="设置",
            command=self._open_settings,
        ).grid(row=0, column=1, sticky="ew", padx=(5, 0))

        self.app_listbox = tk.Listbox(
            sidebar,
            width=30,
            activestyle="none",
            bg=self.colors["sidebar_soft"],
            fg="#eff6ff",
            selectbackground=self.colors["brand"],
            selectforeground="#ffffff",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#27415f",
            borderwidth=0,
            font=("Microsoft YaHei UI", 10),
        )
        self.app_listbox.grid(row=3, column=0, sticky="nsew")
        self.app_listbox.bind("<<ListboxSelect>>", self._on_app_select)

        self.status_text = tk.StringVar(value="生产模式：等待接入任务")
        ttk.Label(
            sidebar,
            textvariable=self.status_text,
            style="SidebarHint.TLabel",
            wraplength=220,
        ).grid(row=4, column=0, sticky="ew", pady=(16, 0))

    def _build_workspace(self) -> None:
        main = ttk.Frame(self, padding=(14, 14))
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)

        self.app_header = ttk.Frame(main, style="Card.TFrame", padding=16)
        self.app_header.grid(row=0, column=0, sticky="ew")
        self.app_header.columnconfigure(0, weight=1)
        self.app_title = ttk.Label(
            self.app_header,
            text="未选择 APP",
            style="Title.TLabel",
        )
        self.app_title.grid(row=0, column=0, sticky="w")
        self.app_subtitle = ttk.Label(
            self.app_header,
            text="从左侧新建或选择一个应用",
            style="Hint.TLabel",
        )
        self.app_subtitle.grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(
            self.app_header,
            text="编辑任务配置",
            style="Ghost.TButton",
            command=self._edit_selected_app,
        ).grid(row=0, column=1, rowspan=2, sticky="e")

        metrics = ttk.Frame(main, style="Card.TFrame", padding=14)
        metrics.grid(row=1, column=0, sticky="ew", pady=(12, 12))
        for index in range(4):
            metrics.columnconfigure(index, weight=1)
        self.metric_jobs = self._metric(metrics, 0, "JOB", "0")
        self.metric_waiting = self._metric(metrics, 1, "待人工", "0")
        self.metric_running = self._metric(metrics, 2, "运行中", "0")
        self.metric_macros = self._metric(metrics, 3, "可用宏", "0")

        body = ttk.Frame(main)
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        overview = ttk.Frame(body, style="Card.TFrame", padding=16)
        overview.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        overview.columnconfigure(0, weight=1)
        overview.rowconfigure(2, weight=1)
        ttk.Label(overview, text="任务说明", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            overview,
            text="文字描述、规则约束和图片/视频引用会共同组成模型上下文。",
            style="Hint.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 8))
        self.description_view = tk.Text(
            overview,
            wrap="word",
            bg=self.colors["surface_soft"],
            fg=self.colors["ink"],
            relief="flat",
            padx=12,
            pady=10,
            font=("Microsoft YaHei UI", 10),
        )
        self.description_view.grid(row=2, column=0, sticky="nsew")
        self.description_view.configure(state="disabled")

        refs = ttk.Frame(overview, style="Card.TFrame")
        refs.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        refs.columnconfigure(0, weight=1)
        ttk.Label(refs, text="素材引用", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.reference_list = tk.Listbox(
            refs,
            height=4,
            bg="#ffffff",
            fg=self.colors["ink"],
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            font=("Microsoft YaHei UI", 9),
        )
        self.reference_list.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        trace = ttk.Frame(body, style="Card.TFrame", padding=16)
        trace.grid(row=0, column=1, sticky="nsew")
        trace.columnconfigure(0, weight=1)
        trace.rowconfigure(2, weight=1)
        ttk.Label(trace, text="执行轨迹", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            trace,
            text="展示模型输入、思考摘要、工具调用和执行结果。",
            style="Hint.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 8))
        self.trace_tree = ttk.Treeview(
            trace,
            columns=("kind", "summary"),
            show="headings",
            selectmode="browse",
        )
        self.trace_tree.heading("kind", text="类型")
        self.trace_tree.heading("summary", text="摘要")
        self.trace_tree.column("kind", width=84, anchor="center")
        self.trace_tree.column("summary", width=280, stretch=True)
        self.trace_tree.grid(row=2, column=0, sticky="nsew")
        self.trace_tree.bind("<<TreeviewSelect>>", self._on_trace_select)
        self.trace_detail = tk.Text(
            trace,
            height=8,
            wrap="word",
            bg="#f8fafc",
            fg=self.colors["ink"],
            relief="flat",
            padx=10,
            pady=8,
            font=("Microsoft YaHei UI", 9),
        )
        self.trace_detail.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.trace_detail.configure(state="disabled")

        queues = ttk.Frame(main, style="Card.TFrame", padding=16)
        queues.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        queues.columnconfigure(0, weight=1)
        ttk.Label(queues, text="当前 JOB 队列", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.queue_tree = ttk.Treeview(
            queues,
            columns=("queue", "subtask", "goal"),
            show="headings",
            height=5,
        )
        self.queue_tree.heading("queue", text="队列")
        self.queue_tree.heading("subtask", text="子任务")
        self.queue_tree.heading("goal", text="目标")
        self.queue_tree.column("queue", width=110, anchor="center")
        self.queue_tree.column("subtask", width=150)
        self.queue_tree.column("goal", width=520, stretch=True)
        self.queue_tree.grid(row=1, column=0, sticky="ew", pady=(8, 0))

    def _build_human_panel(self) -> None:
        panel = ttk.Frame(self, padding=(0, 14, 14, 14))
        panel.grid(row=0, column=2, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(2, weight=1)

        card = ttk.Frame(panel, style="Card.TFrame", padding=16)
        card.grid(row=0, column=0, rowspan=3, sticky="nsew")
        card.columnconfigure(0, weight=1)
        card.rowconfigure(2, weight=1)

        ttk.Label(card, text="需要你处理", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            card,
            text="远程渠道和本地客户端会自动同步状态。你只需要处理业务问题。",
            style="Hint.TLabel",
            wraplength=320,
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))

        self.human_tree = ttk.Treeview(
            card,
            columns=("status", "job", "question"),
            show="headings",
            selectmode="browse",
            height=7,
        )
        self.human_tree.heading("status", text="状态")
        self.human_tree.heading("job", text="JOB")
        self.human_tree.heading("question", text="问题")
        self.human_tree.column("status", width=70, anchor="center")
        self.human_tree.column("job", width=90, anchor="center")
        self.human_tree.column("question", width=210, stretch=True)
        self.human_tree.grid(row=2, column=0, sticky="nsew")
        self.human_tree.bind("<<TreeviewSelect>>", self._on_human_select)

        ttk.Label(card, text="当前请求", style="Section.TLabel").grid(
            row=3, column=0, sticky="w", pady=(14, 4)
        )
        self.request_detail = tk.Text(
            card,
            height=8,
            wrap="word",
            bg=self.colors["surface_soft"],
            fg=self.colors["ink"],
            relief="flat",
            padx=10,
            pady=8,
            font=("Microsoft YaHei UI", 9),
        )
        self.request_detail.grid(row=4, column=0, sticky="ew")
        self.request_detail.configure(state="disabled")

        form = ttk.Frame(card, style="Card.TFrame")
        form.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="处理类型", style="Hint.TLabel").grid(
            row=0, column=0, sticky="w", pady=4
        )
        self.decision_combo = ttk.Combobox(
            form,
            textvariable=self.decision_type,
            values=("人工回复", "风险确认", "更改路径", "暂停/取消", "补充约束"),
            state="readonly",
        )
        self.decision_combo.grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(form, text="直接命令", style="Hint.TLabel").grid(
            row=1, column=0, sticky="w", pady=4
        )
        self.direct_command_combo = ttk.Combobox(
            form,
            textvariable=self.direct_command_label,
            values=tuple(DIRECT_COMMAND_LABELS.values()),
            state="readonly",
        )
        self.direct_command_combo.grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(card, text="你的输入", style="Hint.TLabel").grid(
            row=6, column=0, sticky="w", pady=(12, 4)
        )
        self.manager_input = tk.Text(
            card,
            height=7,
            wrap="word",
            bg="#ffffff",
            relief="solid",
            borderwidth=1,
            font=("Microsoft YaHei UI", 10),
        )
        self.manager_input.grid(row=7, column=0, sticky="ew")

        ttk.Label(card, text="新增约束 / 注意事项", style="Hint.TLabel").grid(
            row=8, column=0, sticky="w", pady=(12, 4)
        )
        self.constraints_input = tk.Text(
            card,
            height=5,
            wrap="word",
            bg="#ffffff",
            relief="solid",
            borderwidth=1,
            font=("Microsoft YaHei UI", 10),
        )
        self.constraints_input.grid(row=9, column=0, sticky="ew")
        ttk.Button(
            card,
            text="提交给 Agent",
            style="Primary.TButton",
            command=self._submit_human_reply,
        ).grid(row=10, column=0, sticky="ew", pady=(14, 0))

    def _metric(self, parent: ttk.Frame, column: int, title: str, value: str) -> tk.StringVar:
        holder = ttk.Frame(parent, style="Soft.TFrame", padding=(12, 10))
        holder.grid(row=0, column=column, sticky="ew", padx=5)
        ttk.Label(
            holder,
            text=title,
            background=self.colors["surface_soft"],
            foreground=self.colors["muted"],
            font=("Microsoft YaHei UI", 9),
        ).grid(row=0, column=0, sticky="w")
        var = tk.StringVar(value=value)
        ttk.Label(holder, textvariable=var, style="Metric.TLabel").grid(
            row=1, column=0, sticky="w", pady=(4, 0)
        )
        return var

    def refresh_all(self) -> None:
        self.refresh_apps()
        self.refresh_current_app()
        self.refresh_human()
        self.refresh_trace()
        self.refresh_queues()

    def refresh_apps(self) -> None:
        current = self.selected_app_id.get()
        self.app_listbox.delete(0, tk.END)
        app_ids = list(self.state.target_apps)
        for app_id in app_ids:
            config = self.state.target_apps[app_id]
            job = self.state.job_board.jobs.get(config.job_id)
            waiting = job.queues.counts()["waiting_human"] if job else 0
            suffix = f"  ·  待处理 {waiting}" if waiting else ""
            self.app_listbox.insert(tk.END, f"{config.app_name}{suffix}")
        if current not in self.state.target_apps and app_ids:
            current = app_ids[0]
            self.selected_app_id.set(current)
        if current in self.state.target_apps:
            index = app_ids.index(current)
            self.app_listbox.selection_clear(0, tk.END)
            self.app_listbox.selection_set(index)
            self.app_listbox.activate(index)

    def refresh_current_app(self) -> None:
        config = self._selected_config()
        if config is None:
            self.app_title.configure(text="未配置 APP")
            self.app_subtitle.configure(text="从左侧新建 APP，填入任务描述和素材引用")
            self._set_text(self.description_view, "当前没有 APP 配置。")
            self.reference_list.delete(0, tk.END)
            self.metric_jobs.set(str(len(self.state.job_board.jobs)))
            self.metric_waiting.set(str(self._total_waiting_requests()))
            self.metric_running.set("0")
            self.metric_macros.set(str(len(self.state.action_macros.macros)))
            return
        self.app_title.configure(text=config.app_name)
        self.app_subtitle.configure(text=f"JOB: {config.job_id}")
        description = config.task_description.strip() or "尚未填写任务描述。"
        self._set_text(self.description_view, description)
        self.reference_list.delete(0, tk.END)
        for asset in config.reference_assets:
            self.reference_list.insert(
                tk.END,
                f"{asset.citation}  {asset.title or Path(asset.path).name}",
            )
        job = self.state.job_board.jobs.get(config.job_id)
        counts = job.queues.counts() if job else {}
        self.metric_jobs.set(str(len(self.state.job_board.jobs)))
        self.metric_waiting.set(str(self._total_waiting_requests()))
        self.metric_running.set(str(counts.get("running", 0)))
        self.metric_macros.set(str(len(self._macros_for_app(config.app_id))))

    def refresh_human(self) -> None:
        selected = self.selected_request_id.get()
        self._clear_tree(self.human_tree)
        for item in self.state.human_loop.list_requests(include_completed=True):
            if item["status"] == "resolved":
                continue
            request = item["request"]
            metadata = request.get("metadata") or {}
            request_id = request["request_id"]
            self.human_tree.insert(
                "",
                tk.END,
                iid=request_id,
                values=(
                    self._status_label(item["status"]),
                    metadata.get("job_id") or "",
                    request.get("question") or "",
                ),
            )
        if selected and self.human_tree.exists(selected):
            self.human_tree.selection_set(selected)
            self.human_tree.focus(selected)
        self._show_selected_request()
        self.metric_waiting.set(str(self._total_waiting_requests()))

    def refresh_trace(self) -> None:
        selected = self._selected_tree_iid(self.trace_tree)
        self._clear_tree(self.trace_tree)
        for index, event in enumerate(self.state.trace_store.events):
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
                iid=str(index),
                values=(self._trace_kind_label(event.kind), str(summary)[:120]),
            )
        if selected and self.trace_tree.exists(selected):
            self.trace_tree.selection_set(selected)
            self.trace_tree.focus(selected)
        self._show_selected_trace()

    def refresh_queues(self) -> None:
        self._clear_tree(self.queue_tree)
        config = self._selected_config()
        if config is None:
            return
        job = self.state.job_board.jobs.get(config.job_id)
        if job is None:
            return
        queues = job.queues.to_dict()
        for queue_name, queue_value in queues.items():
            if queue_name == "running" and queue_value:
                self._insert_queue_row("运行中", queue_value)
            elif isinstance(queue_value, list):
                for subtask in queue_value:
                    self._insert_queue_row(self._queue_label(queue_name), subtask)
            elif isinstance(queue_value, dict):
                for subtask in queue_value.values():
                    self._insert_queue_row(self._queue_label(queue_name), subtask)

    def _periodic_refresh(self) -> None:
        self.refresh_apps()
        self.refresh_human()
        self.refresh_queues()
        self.after(2000, self._periodic_refresh)

    def _on_app_select(self, _event: tk.Event[Any]) -> None:
        selection = self.app_listbox.curselection()
        if not selection:
            return
        app_id = list(self.state.target_apps)[selection[0]]
        self.selected_app_id.set(app_id)
        self.refresh_current_app()
        self.refresh_queues()

    def _on_human_select(self, _event: tk.Event[Any]) -> None:
        selected = self._selected_tree_iid(self.human_tree)
        if selected:
            self.selected_request_id.set(selected)
        self._show_selected_request()

    def _on_trace_select(self, _event: tk.Event[Any]) -> None:
        self._show_selected_trace()

    def _add_app_dialog(self) -> None:
        dialog = AppConfigDialog(self, title="新建 APP")
        result = dialog.result
        if result is None:
            return
        app_name = result["app_name"].strip()
        job_id = result["job_id"].strip()
        if not app_name or not job_id:
            messagebox.showwarning("输入不完整", "APP 名称和 JOB ID 都不能为空。")
            return
        app_id = unique_app_id(app_name, self.state.target_apps)
        config = TargetAppConfig(
            app_id=app_id,
            app_name=app_name,
            job_id=job_id,
            task_description=result["task_description"].strip(),
        )
        self.state.add_target_app(config)
        if job_id not in self.state.job_board.jobs:
            self.state.job_board.add_job(
                JobState(job_id=job_id, target_app=app_name, goal=app_name)
            )
        self.selected_app_id.set(app_id)
        self.status_text.set(f"已添加 APP：{app_name}")
        self.refresh_all()

    def _edit_selected_app(self) -> None:
        config = self._selected_config()
        if config is None:
            messagebox.showinfo("没有 APP", "请先从左侧新建或选择一个 APP。")
            return
        dialog = AppConfigDialog(self, title="编辑任务配置", config=config)
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
        self.status_text.set(f"已更新 APP：{updated.app_name}")
        self.refresh_all()

    def _open_settings(self) -> None:
        SettingsDialog(self, self.state, self.selected_app_id.get())
        self.refresh_all()

    def _submit_human_reply(self) -> None:
        request_id = self._selected_tree_iid(self.human_tree) or self.selected_request_id.get()
        if not request_id:
            messagebox.showwarning("未选择请求", "请先选择一个需要处理的请求。")
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
            messagebox.showerror("提交失败", f"{exc}\n\n该请求可能已经被其他渠道处理。")
            self.refresh_human()
            return
        self.manager_input.delete("1.0", tk.END)
        self.constraints_input.delete("1.0", tk.END)
        self.status_text.set("人工处理已提交")
        self.refresh_all()

    def _show_selected_request(self) -> None:
        selected = self._selected_tree_iid(self.human_tree)
        text = "当前没有选中的人工请求。"
        if selected:
            for item in self.state.human_loop.list_requests(include_completed=True):
                request = item["request"]
                if request["request_id"] == selected:
                    metadata = request.get("metadata") or {}
                    lines = [
                        f"问题：{request.get('question') or ''}",
                        f"状态：{self._status_label(item['status'])}",
                        f"JOB：{metadata.get('job_id') or ''}",
                        f"子任务：{request.get('task_id') or ''}",
                    ]
                    if request.get("risk_reason"):
                        lines.append(f"风险：{request['risk_reason']}")
                    if request.get("proposed_action"):
                        lines.append(f"建议：{request['proposed_action']}")
                    if request.get("allowed_reply_format"):
                        lines.append(f"格式：{request['allowed_reply_format']}")
                    text = "\n".join(lines)
                    break
        self._set_text(self.request_detail, text)

    def _show_selected_trace(self) -> None:
        selected = self._selected_tree_iid(self.trace_tree)
        text = "暂无执行轨迹。"
        if selected is not None and selected.isdigit():
            index = int(selected)
            if 0 <= index < len(self.state.trace_store.events):
                event = self.state.trace_store.events[index]
                text = format_payload(event.payload)
        self._set_text(self.trace_detail, text)

    def _insert_queue_row(self, queue_name: str, subtask: dict[str, Any]) -> None:
        self.queue_tree.insert(
            "",
            tk.END,
            values=(
                queue_name,
                subtask.get("subtask_id", ""),
                subtask.get("goal", ""),
            ),
        )

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

    def _total_waiting_requests(self) -> int:
        return sum(
            1
            for item in self.state.human_loop.list_requests()
            if item["status"] != "resolved"
        )

    def _macros_for_app(self, app_id: str) -> list[Any]:
        return [
            macro
            for macro in self.state.action_macros.macros.values()
            if macro.metadata.get("scope") == "global"
            or macro.metadata.get("app_id") == app_id
        ]

    @staticmethod
    def _status_label(status: str) -> str:
        return {
            "pending": "待处理",
            "claimed": "处理中",
            "resolved": "已处理",
            "cancelled": "已取消",
        }.get(status, status)

    @staticmethod
    def _queue_label(queue_name: str) -> str:
        return {
            "urgent": "紧急",
            "runnable": "可执行",
            "waiting_human": "等人工",
            "blocked": "阻塞",
            "completed": "完成",
            "failed": "失败",
            "cancelled": "取消",
            "superseded": "替换",
        }.get(queue_name, queue_name)

    @staticmethod
    def _trace_kind_label(kind: str) -> str:
        return {
            "step": "步骤",
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

        frame = ttk.Frame(self, style="Card.TFrame", padding=16)
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
        ttk.Label(frame, text="JOB ID", style="Hint.TLabel").grid(
            row=2, column=0, sticky="w", pady=4
        )
        self.job_id = ttk.Entry(frame)
        self.job_id.grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(frame, text="任务描述", style="Section.TLabel").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(14, 4)
        )
        ttk.Label(
            frame,
            text="可在文字中引用下方素材，例如 [image:image_abcd1234]。",
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

        buttons = ttk.Frame(frame, style="Card.TFrame")
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

        frame = ttk.Frame(self, style="Card.TFrame", padding=16)
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

        left = ttk.Frame(frame, style="Card.TFrame")
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

        right = ttk.Frame(frame, style="Card.TFrame")
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


def unique_app_id(app_name: str, existing: dict[str, TargetAppConfig]) -> str:
    base = "".join(ch.lower() if ch.isalnum() else "_" for ch in app_name)
    base = "_".join(part for part in base.split("_") if part) or "app"
    app_id = f"app_{base}"
    index = 2
    while app_id in existing:
        app_id = f"app_{base}_{index}"
        index += 1
    return app_id
