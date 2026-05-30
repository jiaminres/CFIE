from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from cfie_gui_agent.human_loop import HumanLoopManager, InMemoryHumanChannel
from cfie_gui_agent.jobs import JobBoard, JobState
from cfie_gui_agent.macros import (
    ActionMacro,
    ActionMacroRegistry,
    ActionMacroStep,
    action_macro_from_proposal,
)
from cfie_gui_agent.trace import AgentTraceStore

DIRECT_COMMAND_NONE = "none"
DIRECT_COMMAND_CONTINUE = "continue"
DIRECT_COMMAND_DO_NOT_REPLY = "do_not_reply"
DIRECT_COMMAND_CHANGE_PATH = "change_path"
DIRECT_COMMAND_PAUSE_JOB = "pause_job"
DIRECT_COMMAND_CANCEL_SUBTASK = "cancel_subtask"
DIRECT_COMMAND_MARK_COMPLETE = "mark_complete"

DIRECT_COMMAND_LABELS = {
    DIRECT_COMMAND_NONE: "仅提交说明",
    DIRECT_COMMAND_CONTINUE: "继续执行",
    DIRECT_COMMAND_DO_NOT_REPLY: "不要回复当前对象",
    DIRECT_COMMAND_CHANGE_PATH: "更改操作路径",
    DIRECT_COMMAND_PAUSE_JOB: "暂停当前 JOB",
    DIRECT_COMMAND_CANCEL_SUBTASK: "取消当前子任务",
    DIRECT_COMMAND_MARK_COMPLETE: "标记任务已完成",
}

DEFAULT_RESPONSES_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_RESPONSES_MODEL = "qwen35-vl"
DEFAULT_CLIENT_STATE_PATH = Path("runs") / "gui_agent" / "state.json"
DEFAULT_TRACE_DIR = Path("runs") / "gui_agent" / "traces"


@dataclass(slots=True, frozen=True)
class ReferenceAsset:
    asset_id: str
    kind: str
    path: str
    title: str = ""
    description: str = ""

    @property
    def citation(self) -> str:
        return f"[{self.kind}:{self.asset_id}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "kind": self.kind,
            "path": self.path,
            "title": self.title,
            "description": self.description,
            "citation": self.citation,
        }


@dataclass(slots=True, frozen=True)
class TargetAppConfig:
    app_id: str
    app_name: str
    job_id: str
    task_description: str = ""
    reference_assets: tuple[ReferenceAsset, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_asset(self, asset: ReferenceAsset) -> "TargetAppConfig":
        return replace(self, reference_assets=(*self.reference_assets, asset))

    def with_description(self, text: str) -> "TargetAppConfig":
        return replace(self, task_description=text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "app_id": self.app_id,
            "app_name": self.app_name,
            "job_id": self.job_id,
            "task_description": self.task_description,
            "reference_assets": [
                asset.to_dict() for asset in self.reference_assets
            ],
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class MacroConfig:
    name: str
    description: str
    sequence: str
    scope: str = "global"
    app_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "sequence": self.sequence,
            "scope": self.scope,
            "app_id": self.app_id,
        }


@dataclass(slots=True)
class DesktopClientState:
    human_loop: HumanLoopManager = field(
        default_factory=lambda: HumanLoopManager(channel=InMemoryHumanChannel())
    )
    job_board: JobBoard = field(default_factory=JobBoard)
    trace_store: AgentTraceStore = field(default_factory=AgentTraceStore)
    action_macros: ActionMacroRegistry = field(default_factory=ActionMacroRegistry)
    target_apps: dict[str, TargetAppConfig] = field(default_factory=dict)
    selected_app_id: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    macro_configs: dict[str, MacroConfig] = field(default_factory=dict)

    def add_target_app(self, config: TargetAppConfig) -> None:
        if not config.app_id:
            raise ValueError("app_id is required")
        self.target_apps[config.app_id] = config
        if not self.selected_app_id:
            self.selected_app_id = config.app_id

    def remove_target_app(self, app_id: str) -> TargetAppConfig:
        config = self.target_apps.pop(app_id)
        self.job_board.jobs.pop(config.job_id, None)
        self.macro_configs = {
            name: macro_config
            for name, macro_config in self.macro_configs.items()
            if macro_config.app_id != app_id
        }
        self.action_macros.macros = {
            name: macro
            for name, macro in self.action_macros.macros.items()
            if macro.metadata.get("app_id") != app_id
        }
        for bucket in (
            self.human_loop.pending,
            self.human_loop.states,
            self.human_loop.completed,
        ):
            for request_id, item in list(bucket.items()):
                request = item.request if hasattr(item, "request") else item
                metadata = getattr(request, "metadata", {}) or {}
                if metadata.get("app_id") == app_id or metadata.get("job_id") == config.job_id:
                    bucket.pop(request_id, None)
        channel = self.human_loop.channel
        if hasattr(channel, "sent_requests"):
            channel.sent_requests = [
                request
                for request in channel.sent_requests
                if (request.metadata or {}).get("app_id") != app_id
                and (request.metadata or {}).get("job_id") != config.job_id
            ]
        self.trace_store.events = [
            event
            for event in self.trace_store.events
            if _trace_event_app_id(event.payload) != app_id
        ]
        if self.trace_store.path == Path(config.metadata.get("trace_path", "")):
            self.trace_store.path = None
        if self.selected_app_id == app_id:
            self.selected_app_id = next(iter(self.target_apps), "")
        return config

    def update_target_description(self, app_id: str, text: str) -> None:
        config = self.target_apps[app_id]
        self.target_apps[app_id] = config.with_description(text)

    def update_target_viewport(
        self,
        app_id: str,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> dict[str, int]:
        if width <= 0 or height <= 0:
            raise ValueError("viewport width and height must be positive")
        config = self.target_apps[app_id]
        viewport = {
            "x": max(0, int(x)),
            "y": max(0, int(y)),
            "width": int(width),
            "height": int(height),
        }
        metadata = dict(config.metadata)
        metadata["manual_viewport"] = viewport
        metadata["viewport_source"] = "desktop_marker"
        self.target_apps[app_id] = replace(config, metadata=metadata)
        return viewport

    def clear_target_viewport(self, app_id: str) -> None:
        config = self.target_apps[app_id]
        metadata = dict(config.metadata)
        metadata.pop("manual_viewport", None)
        metadata.pop("viewport_source", None)
        self.target_apps[app_id] = replace(config, metadata=metadata)

    def add_reference_asset(
        self,
        *,
        app_id: str,
        kind: str,
        path: str,
        title: str = "",
        description: str = "",
    ) -> ReferenceAsset:
        asset = ReferenceAsset(
            asset_id=f"{kind}_{uuid4().hex[:8]}",
            kind=kind,
            path=path,
            title=title or Path(path).stem,
            description=description,
        )
        self.target_apps[app_id] = self.target_apps[app_id].with_asset(asset)
        return asset

    def register_macro(self, macro_config: MacroConfig) -> ActionMacro:
        steps = tuple(
            ActionMacroStep.keypress(*keys)
            for keys in parse_macro_sequence(macro_config.sequence)
        )
        macro = ActionMacro(
            name=macro_config.name.strip(),
            description=macro_config.description.strip(),
            steps=steps,
            metadata={
                "sequence": macro_config.sequence,
                "scope": macro_config.scope,
                "app_id": macro_config.app_id,
            },
        )
        self.action_macros.register(macro)
        self.macro_configs[macro.name] = macro_config
        return macro

    def approve_macro_request(self, request_id: str) -> ActionMacro:
        state = self.human_loop.states.get(request_id)
        if state is None:
            state = self.human_loop.completed.get(request_id)
        if state is None:
            raise KeyError(f"unknown human request: {request_id}")
        proposal = state.request.metadata.get("macro_proposal")
        if not isinstance(proposal, dict):
            raise ValueError("human request does not contain a macro proposal")
        macro = action_macro_from_proposal(
            proposal,
            metadata={
                "request_id": request_id,
                "app_id": self.selected_app_id or None,
            },
        )
        self.action_macros.upsert(macro)
        return macro

    def configure_app_session(
        self,
        *,
        app_name: str,
        task_description: str = "",
        trace_path: str | None = None,
        process_name: str = "",
        executable_path: str = "",
        window_title_pattern: str = "",
        base_url: str = DEFAULT_RESPONSES_BASE_URL,
        model: str = DEFAULT_RESPONSES_MODEL,
        reasoning_mode: str = "guided",
        reasoning_effort: str = "none",
        max_output_tokens: int | None = None,
        tool_profile: str = "core",
        record_trace: bool = True,
    ) -> dict[str, Any]:
        app_name = app_name.strip()
        if not app_name:
            raise ValueError("app_name is required")
        app_id = unique_app_id(app_name, self.target_apps)
        job_id = f"job:{app_id}"
        metadata = {
            "trace_path": str(trace_path or ""),
            "process_name": process_name,
            "executable_path": executable_path,
            "window_title_pattern": normalize_window_title_pattern(window_title_pattern),
            "base_url": base_url,
            "model": model,
            "reasoning_mode": reasoning_mode,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            "tool_profile": tool_profile,
        }
        config = TargetAppConfig(
            app_id=app_id,
            app_name=app_name,
            job_id=job_id,
            task_description=task_description.strip(),
            metadata=metadata,
        )
        self.add_target_app(config)
        self.selected_app_id = app_id
        if job_id not in self.job_board.jobs:
            self.job_board.add_job(
                JobState(
                    job_id=job_id,
                    target_app=app_name,
                    goal=task_description.strip() or app_name,
                )
            )
        if trace_path:
            self.trace_store.path = Path(trace_path)
        if record_trace:
            self.trace_store.record(
                "app_configured",
                {
                    "app_id": app_id,
                    "app_name": app_name,
                    "job_id": job_id,
                    "task_description": task_description.strip(),
                    "trace_path": str(trace_path or ""),
                    "metadata": metadata,
                },
            )
        return {"app": config.to_dict(), "job_id": job_id}

    def record_operation_summary(
        self,
        *,
        app_id: str,
        kind: str,
        title: str,
        summary: str = "",
        status: str = "recorded",
        artifact_refs: tuple[str, ...] = (),
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event_payload = {
            "app_id": app_id,
            "kind": kind,
            "title": title,
            "summary": summary,
            "status": status,
            "artifact_refs": list(artifact_refs),
            "payload": payload or {},
        }
        self.trace_store.record("operation", event_payload)
        return event_payload

    def submit_structured_human_reply(
        self,
        *,
        request_id: str,
        manager_input: str,
        decision_type: str,
        direct_command: str = DIRECT_COMMAND_NONE,
        constraints: str = "",
    ) -> dict[str, Any]:
        if direct_command not in DIRECT_COMMAND_LABELS:
            raise ValueError(f"unknown direct command: {direct_command}")
        payload = {
            "decision_type": decision_type,
            "direct_command": direct_command,
            "direct_command_label": DIRECT_COMMAND_LABELS[direct_command],
            "manager_input": manager_input.strip(),
            "constraints": [
                line.strip()
                for line in constraints.splitlines()
                if line.strip()
            ],
        }
        text_parts = []
        if direct_command != DIRECT_COMMAND_NONE:
            text_parts.append(f"直接命令：{DIRECT_COMMAND_LABELS[direct_command]}")
        if manager_input.strip():
            text_parts.append(f"你的输入：{manager_input.strip()}")
        if constraints.strip():
            text_parts.append(f"新增约束：{constraints.strip()}")
        task = self.human_loop.submit_reply(
            request_id=request_id,
            text="\n".join(text_parts).strip() or DIRECT_COMMAND_LABELS[direct_command],
            source="client",
            metadata={
                "client": "desktop_client",
                "structured_payload": payload,
            },
        )
        try:
            self.job_board.enqueue_manager_reply(task)
        except Exception:
            pass
        return task

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_app_id": self.selected_app_id,
            "target_apps": {
                key: config.to_dict() for key, config in self.target_apps.items()
            },
            "jobs": {
                key: job.to_dict() for key, job in self.job_board.jobs.items()
            },
            "human_requests": list(
                self.human_loop.list_requests(include_completed=True)
            ),
            "macros": self.action_macros.to_context_payload(),
            "macro_configs": [
                config.to_dict() for config in self.macro_configs.values()
            ],
            "settings": self.settings,
            "trace": self.trace_store.to_dict(),
        }


def normalize_window_title_pattern(pattern: str) -> str:
    pattern = pattern.strip()
    if not pattern:
        return ""
    try:
        re.compile(pattern)
        return pattern
    except re.error:
        pass

    safe_parts: list[str] = []
    for part in pattern.split("|"):
        part = part.strip()
        if not part or set(part) <= {"?"}:
            continue
        try:
            re.compile(part)
            safe_parts.append(part)
        except re.error:
            safe_parts.append(re.escape(part))
    return "|".join(safe_parts)


def unique_app_id(app_name: str, existing: dict[str, TargetAppConfig]) -> str:
    base = re.sub(r"[^a-zA-Z0-9]+", "_", app_name.strip().lower()).strip("_")
    base = f"app_{base or 'app'}"
    candidate = base
    index = 2
    while candidate in existing:
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def parse_macro_sequence(sequence: str) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    for raw_step in sequence.split(","):
        raw_step = raw_step.strip()
        if not raw_step:
            continue
        keys = tuple(
            key.strip().upper()
            for key in raw_step.replace("+", " + ").split("+")
            if key.strip()
        )
        if keys:
            groups.append(keys)
    if not groups:
        raise ValueError("macro sequence is empty")
    return tuple(groups)


def _trace_event_app_id(payload: dict[str, Any]) -> str:
    app_id = str(payload.get("app_id") or "").strip()
    if app_id:
        return app_id
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        app_id = str(metadata.get("app_id") or "").strip()
        if app_id:
            return app_id
    nested = payload.get("payload")
    if isinstance(nested, dict):
        app_id = str(nested.get("app_id") or "").strip()
        if app_id:
            return app_id
    return ""


def run_desktop_client(
    *,
    state: DesktopClientState | None = None,
    auto_start: bool = False,
    state_path: str | Path | None = None,
) -> None:
    from cfie_gui_agent.desktop_client_ui import GuiAgentDesktopClient
    from cfie_gui_agent.state_store import load_desktop_state

    resolved_state_path = Path(state_path) if state_path else DEFAULT_CLIENT_STATE_PATH
    app_state = state if state is not None else load_desktop_state(resolved_state_path)
    app = GuiAgentDesktopClient(app_state, state_path=resolved_state_path)
    if auto_start:
        app.after(800, app._start_selected_agent_run)
    app.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the CFIE GUI Agent desktop client."
    )
    parser.add_argument("--app-name", default=None)
    parser.add_argument(
        "--state-path",
        default=None,
        help="Desktop client state file. Defaults to runs/gui_agent/state.json.",
    )
    parser.add_argument("--task-description", default="")
    parser.add_argument(
        "--task-description-file",
        default=None,
        help="Read the selected app task description from a UTF-8 text file.",
    )
    parser.add_argument("--trace-path", default=None)
    parser.add_argument("--process-name", default="chrome.exe")
    parser.add_argument(
        "--executable-path",
        default=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    )
    parser.add_argument("--window-title-pattern", default=".*")
    parser.add_argument("--base-url", default=DEFAULT_RESPONSES_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_RESPONSES_MODEL)
    parser.add_argument("--image-detail", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--screenshot-max-width", type=int, default=None)
    parser.add_argument("--screenshot-max-height", type=int, default=None)
    parser.add_argument(
        "--max-visual-frames",
        type=int,
        default=None,
        help="Maximum screenshots/video frames to keep in one model request.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "origin", "low", "medium", "high"),
        default="none",
    )
    parser.add_argument(
        "--reasoning-mode",
        choices=("guided", "default", "off"),
        default="guided",
        help=(
            "guided sends reasoning.effort and CFIE's Qwen think preamble; "
            "default enables Qwen thinking without the preamble; off disables "
            "thinking."
        ),
    )
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument(
        "--tool-profile",
        choices=("minimal", "core", "full"),
        default="core",
        help="Model-callable tool set. core is faster; full exposes every generic tool.",
    )
    parser.add_argument(
        "--load-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load existing trace events from --trace-path before opening the UI.",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start the selected Agent session after the client opens.",
    )
    return parser


def build_initial_state(args: argparse.Namespace) -> DesktopClientState:
    if args.state_path:
        from cfie_gui_agent.state_store import load_desktop_state

        state = load_desktop_state(args.state_path)
    else:
        state = DesktopClientState()
    has_session = any(
        getattr(args, name) is not None
        for name in ("app_name", "trace_path")
    )
    if not has_session:
        return state
    if not args.app_name:
        raise SystemExit("session launch requires --app-name")
    trace_exists = bool(args.trace_path) and Path(args.trace_path).exists()
    record_trace = not (args.load_trace and trace_exists)
    task_description = args.task_description
    if args.task_description_file:
        task_description = Path(args.task_description_file).read_text(
            encoding="utf-8"
        )

    state.configure_app_session(
        app_name=args.app_name,
        task_description=task_description,
        trace_path=args.trace_path,
        process_name=args.process_name,
        executable_path=args.executable_path,
        window_title_pattern=args.window_title_pattern,
        base_url=args.base_url,
        model=args.model,
        reasoning_mode=args.reasoning_mode,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        tool_profile=args.tool_profile,
        record_trace=record_trace,
    )
    config = state.target_apps[state.selected_app_id]
    metadata = dict(config.metadata or {})
    if args.image_detail:
        metadata["image_detail"] = args.image_detail
    if args.max_steps is not None:
        metadata["max_steps"] = args.max_steps
    if args.screenshot_max_width is not None:
        metadata["screenshot_max_width"] = args.screenshot_max_width
    if args.screenshot_max_height is not None:
        metadata["screenshot_max_height"] = args.screenshot_max_height
    if args.max_visual_frames is not None:
        metadata["max_visual_frames"] = args.max_visual_frames
    if metadata != config.metadata:
        state.target_apps[state.selected_app_id] = replace(config, metadata=metadata)
    if args.load_trace and trace_exists:
        state.trace_store.load_existing(args.trace_path)
    return state


def main() -> None:
    args = build_parser().parse_args()
    state = build_initial_state(args) if args.app_name or args.trace_path else None
    run_desktop_client(
        state=state,
        auto_start=args.auto_start,
        state_path=args.state_path,
    )


if __name__ == "__main__":
    main()
