from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import time_ns
from typing import Any
from uuid import uuid4

from cfie_gui_agent.desktop_client import TargetAppConfig


INPUT_KEYS = (
    "input_text",
    "input",
    "question",
    "Question",
    "prompt",
    "Prompt",
    "query",
    "Query",
    "task",
    "Task",
)
EXPECTED_OUTPUT_KEYS = (
    "expected_output",
    "expected_answer",
    "answer",
    "Answer",
    "final_answer",
    "Final answer",
    "Final Answer",
    "ground_truth",
    "Ground Truth",
)
ID_KEYS = ("id", "item_id", "task_id", "Task ID", "qid", "question_id")
ATTACHMENT_KEYS = ("file", "File", "attachment", "attachments", "image", "path")


@dataclass(slots=True, frozen=True)
class WorkflowInputItem:
    item_id: str
    input_text: str
    expected_output: str = ""
    source: str = ""
    attachment_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "input_text": self.input_text,
            "expected_output": self.expected_output,
            "source": self.source,
            "attachment_path": self.attachment_path,
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class WorkflowRun:
    run_id: str
    app_id: str
    job_id: str
    app_name: str
    target_url: str
    input_path: str
    trace_path: str
    item_count: int
    status: str = "configured"
    created_unix_nano: int = field(default_factory=time_ns)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "app_id": self.app_id,
            "job_id": self.job_id,
            "app_name": self.app_name,
            "target_url": self.target_url,
            "input_path": self.input_path,
            "trace_path": self.trace_path,
            "item_count": self.item_count,
            "status": self.status,
            "created_unix_nano": self.created_unix_nano,
            "metadata": self.metadata,
        }


def load_workflow_items(
    path: str | Path,
    *,
    limit: int | None = None,
) -> tuple[WorkflowInputItem, ...]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"workflow input file not found: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        rows = _read_jsonl(input_path)
    elif suffix == ".json":
        rows = _read_json(input_path)
    elif suffix == ".csv":
        rows = _read_csv(input_path)
    else:
        raise ValueError(
            "workflow input format must be .jsonl, .json, or .csv: "
            f"{input_path}"
        )
    items: list[WorkflowInputItem] = []
    for index, row in enumerate(rows, start=1):
        item = _row_to_item(row, index=index, source=str(input_path))
        if item.input_text:
            items.append(item)
        if limit is not None and len(items) >= limit:
            break
    return tuple(items)


def build_workflow_task_description(
    *,
    app_name: str,
    target_url: str,
    input_path: str,
    item_count: int,
    answer_timeout_seconds: int = 180,
) -> str:
    return "\n".join(
        [
            f"目标：在网页应用 {app_name} 中执行自动化任务流。",
            f"目标网址：{target_url}",
            f"输入清单：{input_path}",
            f"计划条目数：{item_count}",
            "",
            "执行规则：",
            "1. 打开目标网页应用，确认已经进入可输入任务内容的界面。",
            "2. 逐条读取输入清单中的 input_text/question 字段，并输入到网页应用。",
            "3. 每次输入后必须提交。网页聊天应用在完成 type 后，用 computer_use 点击可见发送按钮；如果应用明确支持 Enter 发送，也可以用 computer_use 发送 Enter。",
            "4. 如果输入框已经可见，并且已经知道 input_text，不要只点击输入框；先用 computer_use 完成 click/type，再用 computer_use 提交。",
            "5. 如果输入框里已经有待发送文本，不要再次输入同一段文字；下一步应提交或等待输出。不要连续重复同一个点击动作；如果两次点击后仍无法输入或提交，应调用 request_human_help。",
            "6. 将输入、期望输出、实际输出、耗时、截图/视频引用、完成状态写入轨迹文件。",
            "7. 如页面卡住、登录失效、输出超时、控件不可见或需要人工确认，调用 request_human_help。",
            "8. 所有条目处理完成并确认轨迹文件写入后，调用 finish_subtask 结束任务。",
            "",
            "正确性优先：不要跳过条目；无法判断输出完成时最多等待 "
            f"{answer_timeout_seconds} 秒，然后记录失败并请求人工或继续下一条。",
            "日志优先：每一步关键动作都要形成结构化 trace，不要只依赖自然语言描述。",
        ]
    )


def build_workflow_target_config(
    *,
    app_name: str,
    target_url: str,
    input_path: str,
    trace_path: str,
    item_count: int,
    process_name: str = "chrome.exe",
    executable_path: str = r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    window_title_pattern: str = ".*",
    app_id: str | None = None,
    job_id: str | None = None,
) -> TargetAppConfig:
    resolved_app_id = app_id or _slug_id("app", app_name)
    resolved_job_id = job_id or _slug_id("job", app_name)
    description = build_workflow_task_description_v2(
        app_name=app_name,
        target_url=target_url,
        input_path=input_path,
        item_count=item_count,
    )
    return TargetAppConfig(
        app_id=resolved_app_id,
        app_name=app_name,
        job_id=resolved_job_id,
        task_description=description,
        metadata={
            "template": "web_workflow",
            "process_name": process_name,
            "executable_path": executable_path,
            "launch_args": [target_url],
            "window_title_pattern": window_title_pattern,
            "browser_url_pattern": target_url,
            "input_path": input_path,
            "trace_path": trace_path,
            "expected_item_count": item_count,
            "max_unchecked_seconds": 300,
            "state_change_hints": [
                "网页标题变化、任务栏图标闪烁或加载动画长时间不结束，可能表示任务状态变化。",
                "输入框、发送按钮、输出区域是本任务的关键控件。",
                "输出完成后需要记录实际文本和执行证据。",
            ],
        },
    )


def build_workflow_task_description_v2(
    *,
    app_name: str,
    target_url: str,
    input_path: str,
    item_count: int,
    answer_timeout_seconds: int = 180,
) -> str:
    return "\n".join(
        [
            f"目标：在网页应用 {app_name} 中执行当前任务流。",
            f"目标网址：{target_url}",
            f"输入清单：{input_path}",
            f"计划条目数：{item_count}",
            "",
            "执行规则：",
            "1. 打开目标网页应用，确认已经进入可输入任务内容的界面。",
            "2. 使用通用文件读取工具读取输入清单，逐条处理 input_text/question 字段。",
            "3. 每次输入后必须提交，并等待网页应用输出稳定。",
            "4. 将题目、实际输出、状态、原因和证据写入轨迹文件。",
            "5. 页面卡住、登录失效、验证码、控件不可见或需要人工确认时，请求人工协助。",
            "6. 所有条目处理完成并确认轨迹写入后，结束当前子任务。",
            "",
            f"正确性优先：无法判断输出完成时最多等待 {answer_timeout_seconds} 秒，然后记录失败或请求人工。",
            "响应优先：模型只保留短思考，优先输出工具调用，不复述截图内容。",
        ]
    )


def make_operation_event(
    *,
    app_id: str,
    kind: str,
    title: str,
    summary: str = "",
    status: str = "recorded",
    icon: str | None = None,
    step_id: int | None = None,
    artifact_refs: tuple[str, ...] = (),
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "app_id": app_id,
        "kind": kind,
        "icon": icon or _default_icon(kind),
        "title": title,
        "summary": summary,
        "status": status,
        "step_id": step_id,
        "artifact_refs": list(artifact_refs),
        "payload": payload or {},
    }


def write_run_manifest(
    run: WorkflowRun,
    *,
    items: tuple[WorkflowInputItem, ...],
) -> Path:
    trace_path = Path(run.trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = trace_path.with_suffix(".manifest.json")
    manifest = {
        "run": run.to_dict(),
        "items": [item.to_dict() for item in items],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as source:
        for line in source:
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _read_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, dict):
        for key in ("data", "items", "examples", "questions", "rows"):
            maybe_rows = value.get(key)
            if isinstance(maybe_rows, list):
                return [row for row in maybe_rows if isinstance(row, dict)]
        return [value]
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    raise ValueError(f"JSON workflow input must contain object rows: {path}")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return [dict(row) for row in csv.DictReader(source)]


def _row_to_item(
    row: dict[str, Any],
    *,
    index: int,
    source: str,
) -> WorkflowInputItem:
    input_text = str(_first_value(row, INPUT_KEYS, "") or "").strip()
    expected_output = str(_first_value(row, EXPECTED_OUTPUT_KEYS, "") or "").strip()
    item_id = str(_first_value(row, ID_KEYS, "") or "").strip() or f"item_{index:04d}"
    attachment = _first_value(row, ATTACHMENT_KEYS, None)
    return WorkflowInputItem(
        item_id=item_id,
        input_text=input_text,
        expected_output=expected_output,
        source=source,
        attachment_path=str(attachment) if attachment else None,
        metadata={
            key: value
            for key, value in row.items()
            if key not in set((*INPUT_KEYS, *EXPECTED_OUTPUT_KEYS, *ID_KEYS))
        },
    )


def _first_value(
    row: dict[str, Any],
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value not in (None, ""):
            return value
    return default


def _slug_id(prefix: str, text: str) -> str:
    base = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    base = "_".join(part for part in base.split("_") if part)
    return f"{prefix}_{base or uuid4().hex[:8]}"


def _default_icon(kind: str) -> str:
    return {
        "mouse": "mouse",
        "keyboard": "keyboard",
        "switch": "switch",
        "observe": "eye",
        "model_intent": "brain",
        "harness_check": "check",
        "verification": "verify",
        "human": "human",
        "failure": "alert",
        "workflow": "workflow",
    }.get(kind, "trace")
