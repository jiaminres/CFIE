from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from cfie_gui_agent.human_loop import HumanLoopManager, InMemoryHumanChannel
from cfie_gui_agent.jobs import JobBoard
from cfie_gui_agent.macros import ActionMacro, ActionMacroRegistry, ActionMacroStep
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
    workflow_runs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_target_app(self, config: TargetAppConfig) -> None:
        if not config.app_id:
            raise ValueError("app_id is required")
        self.target_apps[config.app_id] = config

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
        return macro

    def configure_workflow(
        self,
        *,
        app_name: str,
        target_url: str,
        input_path: str,
        trace_path: str,
        process_name: str = "chrome.exe",
        executable_path: str = r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        window_title_pattern: str = ".*",
        limit: int | None = None,
        reasoning_effort: str = "none",
        max_output_tokens: int | None = None,
        record_trace: bool = True,
    ) -> dict[str, Any]:
        from cfie_gui_agent.workflow import (
            WorkflowRun,
            build_workflow_target_config,
            load_workflow_items,
            make_operation_event,
            write_run_manifest,
        )

        items = load_workflow_items(input_path, limit=limit)
        if not items:
            raise ValueError("workflow input file does not contain usable rows")
        config = build_workflow_target_config(
            app_name=app_name,
            target_url=target_url,
            input_path=input_path,
            trace_path=trace_path,
            item_count=len(items),
            process_name=process_name,
            executable_path=executable_path,
            window_title_pattern=window_title_pattern,
        )
        config = replace(
            config,
            metadata={
                **config.metadata,
                "reasoning_effort": reasoning_effort,
                "max_output_tokens": max_output_tokens,
            },
        )
        self.add_target_app(config)
        if config.job_id not in self.job_board.jobs:
            from cfie_gui_agent.jobs import JobState

            self.job_board.add_job(
                JobState(
                    job_id=config.job_id,
                    target_app=config.app_name,
                    goal=f"Run workflow with {len(items)} items",
                )
            )
        run = WorkflowRun(
            run_id=f"workflow_{uuid4().hex[:8]}",
            app_id=config.app_id,
            job_id=config.job_id,
            app_name=config.app_name,
            target_url=target_url,
            input_path=input_path,
            trace_path=trace_path,
            item_count=len(items),
            metadata={
                "process_name": process_name,
                "executable_path": executable_path,
                "window_title_pattern": window_title_pattern,
                "first_item_id": items[0].item_id,
                "reasoning_effort": reasoning_effort,
                "max_output_tokens": max_output_tokens,
            },
        )
        manifest_path = write_run_manifest(run, items=items)
        self.workflow_runs[run.run_id] = {
            **run.to_dict(),
            "manifest_path": str(manifest_path),
        }
        self.trace_store.path = Path(trace_path)
        if record_trace:
            self.trace_store.record(
                "workflow_configured",
                {
                    **run.to_dict(),
                    "manifest_path": str(manifest_path),
                    "sample_items": [item.to_dict() for item in items[:3]],
                },
            )
            self.trace_store.record(
                "operation",
                make_operation_event(
                    app_id=config.app_id,
                    kind="workflow",
                    title="任务流已配置",
                    summary=(
                        f"{config.app_name} / {len(items)} items / "
                        f"trace: {trace_path}"
                    ),
                    status="configured",
                    payload={
                        "run_id": run.run_id,
                        "manifest_path": str(manifest_path),
                    },
                ),
            )
        return {
            "run": run.to_dict(),
            "app": config.to_dict(),
            "manifest_path": str(manifest_path),
            "item_count": len(items),
        }

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
        from cfie_gui_agent.workflow import make_operation_event

        event_payload = make_operation_event(
            app_id=app_id,
            kind=kind,
            title=title,
            summary=summary,
            status=status,
            artifact_refs=artifact_refs,
            payload=payload,
        )
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
        return self.human_loop.submit_reply(
            request_id=request_id,
            text="\n".join(text_parts).strip() or DIRECT_COMMAND_LABELS[direct_command],
            source="client",
            metadata={
                "client": "desktop_client",
                "structured_payload": payload,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
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
            "trace": self.trace_store.to_dict(),
            "workflow_runs": self.workflow_runs,
        }


def build_workflow_run_command(
    config: TargetAppConfig,
    *,
    python_executable: str,
    result_json: str | Path | None = None,
    base_url: str = DEFAULT_RESPONSES_BASE_URL,
    model: str = DEFAULT_RESPONSES_MODEL,
    max_steps: int = 8,
    max_output_tokens: int = 512,
    item_limit: int | None = None,
) -> list[str]:
    metadata = config.metadata or {}
    target_url = str(metadata.get("target_url") or metadata.get("browser_url_pattern") or "")
    input_path = str(metadata.get("input_path") or "")
    trace_path = str(metadata.get("trace_path") or "")
    if not target_url or not input_path or not trace_path:
        raise ValueError("当前 APP 缺少 target_url/input_path/trace_path，无法启动工作流。")

    trace_dir = Path(trace_path).expanduser().parent
    artifact_dir = str(metadata.get("artifact_dir") or trace_dir / "artifacts")
    executable_path = str(
        metadata.get("executable_path")
        or r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    )
    window_title_pattern = normalize_window_title_pattern(
        str(metadata.get("window_title_pattern") or "").strip()
    )
    resolved_limit = item_limit
    if resolved_limit is None:
        raw_limit = metadata.get("expected_item_count")
        if isinstance(raw_limit, int):
            resolved_limit = raw_limit
    effective_max_steps = max_steps
    if resolved_limit is not None:
        effective_max_steps = max(effective_max_steps, int(resolved_limit) * 8 + 4)

    command = [
        python_executable,
        "benchmarks/run_gui_agent_workflow_responses.py",
        "--base-url",
        base_url,
        "--model",
        model,
        "--app-name",
        config.app_name,
        "--target-url",
        target_url,
        "--input-path",
        input_path,
        "--trace-path",
        trace_path,
        "--artifact-dir",
        artifact_dir,
        "--chrome-path",
        executable_path,
        "--max-steps",
        str(effective_max_steps),
        "--max-output-tokens",
        str(max_output_tokens),
        "--tool-profile",
        "workflow",
        "--reasoning-effort",
        str(metadata.get("reasoning_effort") or "none"),
        "--screenshot-max-width",
            str(metadata.get("screenshot_max_width") or 1920),
        "--screenshot-max-height",
            str(metadata.get("screenshot_max_height") or 1080),
        "--screenshot-jpeg-quality",
        str(metadata.get("screenshot_jpeg_quality") or 90),
        "--screenshot-url-mode",
        str(metadata.get("screenshot_url_mode") or "file"),
        "--screenshot-grid",
        str(metadata.get("screenshot_grid") or "off"),
        "--image-detail",
        str(metadata.get("image_detail") or "high"),
    ]
    if resolved_limit is not None:
        command.extend(["--item-limit", str(max(1, int(resolved_limit)))])
    manual_viewport = metadata.get("manual_viewport")
    if isinstance(manual_viewport, dict):
        try:
            crop = (
                int(manual_viewport["x"]),
                int(manual_viewport["y"]),
                int(manual_viewport["width"]),
                int(manual_viewport["height"]),
            )
        except (KeyError, TypeError, ValueError):
            crop = None
        if crop is not None and crop[2] > 0 and crop[3] > 0:
            command.extend(["--screenshot-crop", ",".join(str(value) for value in crop)])
    elif window_title_pattern:
        command.extend(
            [
                "--focus-window-title-pattern",
                window_title_pattern,
                "--crop-focused-window",
                "--maximize-focused-window",
            ]
        )
    if result_json is not None:
        command.extend(["--result-json", str(result_json)])
    return command


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


def run_desktop_client(
    *,
    state: DesktopClientState | None = None,
    auto_start: bool = False,
) -> None:
    from cfie_gui_agent.desktop_client_ui import GuiAgentDesktopClient

    app_state = state if state is not None else DesktopClientState()
    app = GuiAgentDesktopClient(app_state)
    if auto_start:
        app.after(800, app._start_selected_workflow_run)
    app.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the CFIE GUI Agent desktop client."
    )
    parser.add_argument("--app-name", default=None)
    parser.add_argument("--target-url", default=None)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--trace-path", default=None)
    parser.add_argument("--process-name", default="chrome.exe")
    parser.add_argument(
        "--executable-path",
        default=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    )
    parser.add_argument("--window-title-pattern", default=".*")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="none",
    )
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument(
        "--load-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load existing trace events from --trace-path before opening the UI.",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start the selected configured workflow after the client opens.",
    )
    return parser


def build_initial_state(args: argparse.Namespace) -> DesktopClientState:
    state = DesktopClientState()
    has_workflow = any(
        getattr(args, name) is not None
        for name in ("app_name", "target_url", "input_path", "trace_path")
    )
    if not has_workflow:
        return state
    missing = [
        option
        for option, value in {
            "--app-name": args.app_name,
            "--target-url": args.target_url,
            "--input-path": args.input_path,
            "--trace-path": args.trace_path,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(
            "workflow launch requires "
            + ", ".join(missing)
        )
    trace_exists = Path(args.trace_path).exists()
    record_trace = not (args.load_trace and trace_exists)
    state.configure_workflow(
        app_name=args.app_name,
        target_url=args.target_url,
        input_path=args.input_path,
        trace_path=args.trace_path,
        process_name=args.process_name,
        executable_path=args.executable_path,
        window_title_pattern=args.window_title_pattern,
        limit=args.limit,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        record_trace=record_trace,
    )
    if args.load_trace and trace_exists:
        state.trace_store.load_existing(args.trace_path)
    return state


def main() -> None:
    args = build_parser().parse_args()
    run_desktop_client(state=build_initial_state(args), auto_start=args.auto_start)


if __name__ == "__main__":
    main()
