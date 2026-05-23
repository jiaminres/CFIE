from __future__ import annotations

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "sequence": self.sequence,
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

    def add_target_app(self, config: TargetAppConfig) -> None:
        if not config.app_id:
            raise ValueError("app_id is required")
        self.target_apps[config.app_id] = config

    def update_target_description(self, app_id: str, text: str) -> None:
        config = self.target_apps[app_id]
        self.target_apps[app_id] = config.with_description(text)

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
            metadata={"sequence": macro_config.sequence},
        )
        self.action_macros.register(macro)
        return macro

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
        }


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
) -> None:
    from cfie_gui_agent.desktop_client_ui import GuiAgentDesktopClient

    app_state = state if state is not None else DesktopClientState()
    app = GuiAgentDesktopClient(app_state)
    app.mainloop()


def main() -> None:
    run_desktop_client()


if __name__ == "__main__":
    main()
