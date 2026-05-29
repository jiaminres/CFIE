from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from time import time_ns
from typing import Any

from cfie_gui_agent.context import StepRecord


@dataclass(slots=True, frozen=True)
class AgentTraceEvent:
    kind: str
    payload: dict[str, Any]
    time_unix_nano: int = field(default_factory=time_ns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_unix_nano": self.time_unix_nano,
            "kind": self.kind,
            "payload": self.payload,
        }


@dataclass(slots=True)
class AgentTraceStore:
    path: Path | None = None
    events: list[AgentTraceEvent] = field(default_factory=list)

    def load_existing(
        self,
        path: str | Path,
        *,
        append: bool = False,
    ) -> int:
        trace_path = Path(path)
        self.path = trace_path
        if not append:
            self.events.clear()
        if not trace_path.exists():
            return 0
        loaded = 0
        with trace_path.open("r", encoding="utf-8") as source:
            for line in source:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                kind = value.get("kind")
                payload = value.get("payload")
                if not isinstance(kind, str) or not isinstance(payload, dict):
                    continue
                payload = _compact_loaded_payload(kind, payload)
                self.events.append(
                    AgentTraceEvent(
                        kind=kind,
                        payload=payload,
                        time_unix_nano=int(value.get("time_unix_nano", 0) or 0),
                    )
                )
                loaded += 1
        return loaded

    def record(self, kind: str, payload: dict[str, Any]) -> AgentTraceEvent:
        payload = _compact_loaded_payload(kind, payload)
        event = AgentTraceEvent(kind=kind, payload=payload)
        self.events.append(event)
        self._write_event(event)
        return event

    def record_step(self, step: StepRecord) -> AgentTraceEvent:
        return self.record("step", step.to_summary_dict())

    def record_policy_update(self, payload: dict[str, Any]) -> AgentTraceEvent:
        return self.record("policy_update", payload)

    def record_result(self, payload: dict[str, Any]) -> AgentTraceEvent:
        return self.record("result", payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path) if self.path is not None else None,
            "event_count": len(self.events),
            "recent_events": [event.to_dict() for event in self.events[-20:]],
        }

    def _write_event(self, event: AgentTraceEvent) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


def _compact_loaded_payload(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if kind != "model_response":
        return payload
    compacted = dict(payload)
    response_object = compacted.get("response_object")
    if isinstance(response_object, dict):
        compacted["response_object"] = _compact_response_object(response_object)
    for key in ("request_context", "request_payload_debug"):
        if key in compacted:
            compacted[key] = _redact_inline_media(compacted[key])
    return compacted


def _compact_response_object(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text == "tools" and isinstance(item, list):
                result[key_text] = [_tool_summary(tool) for tool in item]
                result.setdefault("tool_schema_count", len(item))
                continue
            if key_text in {"input_messages", "prompt"} and item:
                result[key_text] = "<omitted; see request_context>"
                continue
            result[key_text] = _compact_response_object(item)
        return result
    if isinstance(value, list):
        return [_compact_response_object(item) for item in value]
    if isinstance(value, tuple):
        return [_compact_response_object(item) for item in value]
    if isinstance(value, str):
        return _redact_inline_media(value)
    return value


def _tool_summary(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict):
        return {"repr": str(tool)[:120]}
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    name = str(tool.get("name") or function.get("name") or "").strip()
    result = {"type": str(tool.get("type") or "function")}
    if name:
        result["name"] = name
    return result


def _redact_inline_media(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if str(key) in {"image_url", "video_url"}:
                result[str(key)] = _media_summary(item)
            else:
                result[str(key)] = _redact_inline_media(item)
        return result
    if isinstance(value, list):
        return [_redact_inline_media(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_inline_media(item) for item in value]
    if isinstance(value, str):
        if value.startswith("data:image/") or value.startswith("data:video/"):
            return _media_summary(value)
        if "data:image/" in value or "data:video/" in value:
            return re.sub(
                r"data:(image|video)/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\r\n]+",
                lambda match: f"<{match.group(1)} data omitted; chars={len(match.group(0))}>",
                value,
            )
    return value


def _media_summary(value: Any) -> dict[str, Any]:
    text = str(value or "")
    if text.startswith("data:"):
        header = text.split(",", 1)[0]
        mime_type = header.removeprefix("data:").split(";", 1)[0]
        kind = "video" if mime_type.startswith("video/") else "image"
        return {
            "placeholder": "[视频]" if kind == "video" else "[图片]",
            "source_type": "data_url",
            "mime_type": mime_type,
            "chars": len(text),
        }
    if isinstance(value, dict):
        return _redact_inline_media(value)
    return {"placeholder": "[媒体]", "source_type": "reference", "ref": text}
