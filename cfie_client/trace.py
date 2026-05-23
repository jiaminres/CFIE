from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import time_ns
from typing import Any

_DATA_URL_PREFIX = "data:"
_COMPUTER_TOOL_NAME = "computer"
_GENAI_EXECUTE_TOOL_OPERATION = "execute_tool"


@dataclass(slots=True)
class TraceStore:
    path: Path | None = None
    artifact_dir: Path | None = None
    persist_data_urls: bool = True

    def record_tool_call(self, payload: dict[str, Any]) -> None:
        call_id = str(payload.get("call_id", ""))
        self._record_otel_event(
            name="execute_tool computer",
            attributes={
                "gen_ai.operation.name": _GENAI_EXECUTE_TOOL_OPERATION,
                "gen_ai.tool.name": _COMPUTER_TOOL_NAME,
                "gen_ai.tool.type": "function",
                "gen_ai.tool.call.id": call_id,
                "cfie.client.phase": "call",
            },
            body=payload,
            body_attribute="gen_ai.tool.call.arguments",
        )

    def record_tool_result(self, payload: dict[str, Any]) -> None:
        call_id = str(payload.get("call_id", ""))
        self._record_otel_event(
            name="execute_tool computer",
            attributes={
                "gen_ai.operation.name": _GENAI_EXECUTE_TOOL_OPERATION,
                "gen_ai.tool.name": _COMPUTER_TOOL_NAME,
                "gen_ai.tool.type": "function",
                "gen_ai.tool.call.id": call_id,
                "cfie.client.phase": "result",
            },
            body=payload,
            body_attribute="gen_ai.tool.call.result",
        )

    def record(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "computer_call":
            self.record_tool_call(payload)
            return
        if kind == "computer_call_output":
            self.record_tool_result(payload)
            return
        self._record_otel_event(
            name=f"cfie.client.{kind}",
            attributes={
                "cfie.client.event.name": kind,
            },
            body=payload,
            body_attribute="cfie.client.payload",
        )

    def _record_otel_event(
        self,
        *,
        name: str,
        attributes: dict[str, Any],
        body: dict[str, Any],
        body_attribute: str,
    ) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        safe_body = self._sanitize_payload(body)
        safe_attributes = {
            key: self._sanitize_payload(value)
            for key, value in attributes.items()
            if value not in (None, "")
        }
        safe_attributes[body_attribute] = json.dumps(
            safe_body,
            ensure_ascii=False,
            sort_keys=True,
        )
        event = {
            "time_unix_nano": time_ns(),
            "name": name,
            "attributes": safe_attributes,
            "body": safe_body,
        }
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _sanitize_payload(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._sanitize_payload(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._sanitize_payload(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_payload(item) for item in value]
        if isinstance(value, str) and value.startswith(_DATA_URL_PREFIX):
            return self._persist_data_url(value)
        return value

    def _persist_data_url(self, value: str) -> dict[str, Any]:
        if "," not in value:
            return {"type": "data_url", "inline": value[:128], "truncated": True}

        header, encoded = value.split(",", 1)
        mime_type = header.removeprefix(_DATA_URL_PREFIX).split(";", 1)[0]
        if not self.persist_data_urls:
            return {
                "type": "data_url",
                "mime_type": mime_type,
                "inline": value[:128],
                "truncated": True,
            }

        try:
            data = base64.b64decode(encoded, validate=True)
        except ValueError:
            return {
                "type": "data_url",
                "mime_type": mime_type,
                "inline": value[:128],
                "truncated": True,
                "decode_error": True,
            }

        digest = hashlib.sha256(data).hexdigest()
        suffix = _suffix_for_mime_type(mime_type)
        artifact_dir = self._artifact_dir()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{digest}{suffix}"
        if not artifact_path.exists():
            artifact_path.write_bytes(data)

        return {
            "type": "data_url_artifact",
            "mime_type": mime_type,
            "sha256": digest,
            "bytes": len(data),
            "path": str(artifact_path),
        }

    def _artifact_dir(self) -> Path:
        if self.artifact_dir is not None:
            return self.artifact_dir
        if self.path is not None:
            return self.path.with_suffix("").parent / f"{self.path.stem}_artifacts"
        return Path("cfie_client_trace_artifacts")


def _suffix_for_mime_type(mime_type: str) -> str:
    if mime_type == "image/png":
        return ".png"
    if mime_type == "image/jpeg":
        return ".jpg"
    if mime_type == "image/webp":
        return ".webp"
    if mime_type == "video/mp4":
        return ".mp4"
    if mime_type == "video/webm":
        return ".webm"
    if mime_type == "video/jpeg":
        return ".videojpeg"
    return ".bin"
