from __future__ import annotations

import json
import hashlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from PIL import Image, ImageChops, ImageStat

from cfie_gui_agent.context import StepRecord

VERIFICATION_OK = "ok"
VERIFICATION_NO_SCREEN_CHANGE = "no_screen_change"
VERIFICATION_REPEATED_ACTION = "repeated_action"
VISUAL_CHANGE_THRESHOLD = 2.0


@dataclass(slots=True, frozen=True)
class StepVerification:
    step_id: int
    status: str
    screen_changed: bool | None = None
    repeated_action_count: int = 1
    action_signature: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ok(self) -> bool:
        return self.status == VERIFICATION_OK

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status,
            "screen_changed": self.screen_changed,
            "repeated_action_count": self.repeated_action_count,
            "action_signature": self.action_signature,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class StepVerifier:
    max_repeated_actions: int = 3
    max_repeated_semantic_actions: int = 3
    max_repeated_click_only_actions: int = 2
    recent_action_signatures: deque[str] = field(default_factory=deque)
    recent_semantic_signatures: deque[str] = field(default_factory=deque)

    def verify(self, step: StepRecord) -> StepVerification:
        signature = _action_signature(step.action)
        repeated = self._count_repeated(signature)
        self.recent_action_signatures.append(signature)
        while len(self.recent_action_signatures) > self.max_repeated_actions:
            self.recent_action_signatures.popleft()
        metadata: dict[str, Any] = {}
        semantic_signature = _semantic_action_signature(step.action)
        semantic_repeated = 1
        semantic_threshold = self.max_repeated_semantic_actions
        if semantic_signature:
            semantic_repeated = self._count_repeated_semantic(semantic_signature)
            if (
                semantic_signature == "computer_click_only"
                and any(
                    item.startswith("computer_text_submit:")
                    for item in self.recent_semantic_signatures
                )
            ):
                semantic_threshold = self.max_repeated_click_only_actions
            self.recent_semantic_signatures.append(semantic_signature)
            while len(self.recent_semantic_signatures) > self.max_repeated_semantic_actions:
                self.recent_semantic_signatures.popleft()
            metadata["semantic_action_signature"] = semantic_signature
            metadata["semantic_repeated_action_count"] = semantic_repeated
            repeated = max(repeated, semantic_repeated)
        elif step.action.get("type") == "computer_call":
            self.recent_semantic_signatures.append("computer_other")
            while len(self.recent_semantic_signatures) > self.max_repeated_semantic_actions:
                self.recent_semantic_signatures.popleft()

        screen_changed: bool | None = None
        if step.before_ref and step.after_ref:
            screen_changed, visual_difference = _screen_refs_changed(
                step.before_ref,
                step.after_ref,
            )
            if visual_difference is not None:
                metadata["visual_difference"] = visual_difference

        status = VERIFICATION_OK
        if screen_changed is False:
            status = VERIFICATION_NO_SCREEN_CHANGE
        if repeated >= self.max_repeated_actions:
            status = VERIFICATION_REPEATED_ACTION
        if semantic_signature and semantic_repeated >= semantic_threshold:
            status = VERIFICATION_REPEATED_ACTION

        return StepVerification(
            step_id=step.step_id,
            status=status,
            screen_changed=screen_changed,
            repeated_action_count=repeated,
            action_signature=signature,
            metadata=metadata,
        )

    def _count_repeated(self, signature: str) -> int:
        count = 1
        for previous in reversed(self.recent_action_signatures):
            if previous != signature:
                break
            count += 1
        return count

    def _count_repeated_semantic(self, signature: str) -> int:
        count = 1
        for previous in reversed(self.recent_semantic_signatures):
            if previous != signature:
                break
            count += 1
        return count


def _action_signature(action: dict[str, Any]) -> str:
    return json.dumps(
        _strip_nonsemantic_action_fields(action),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _strip_nonsemantic_action_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_nonsemantic_action_fields(item)
            for key, item in value.items()
            if key not in {"call_id", "id", "status", "pending_safety_checks"}
        }
    if isinstance(value, list):
        return [_strip_nonsemantic_action_fields(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_nonsemantic_action_fields(item) for item in value]
    return value


def _semantic_action_signature(action: dict[str, Any]) -> str | None:
    if action.get("type") != "computer_call":
        return None
    actions = action.get("actions")
    if not isinstance(actions, list):
        return None
    normalized_actions = [item for item in actions if isinstance(item, dict)]
    if normalized_actions and all(
        str(item.get("type") or "").lower() == "click"
        for item in normalized_actions
    ):
        return "computer_click_only"
    has_text = False
    submitted_texts: list[str] = []
    has_submit = False
    for item in normalized_actions:
        action_type = str(item.get("type") or "").lower()
        if action_type == "type" and str(item.get("text") or "").strip():
            has_text = True
            submitted_texts.append(str(item.get("text") or ""))
        if action_type in {"keypress", "key"}:
            keys = item.get("keys") or item.get("key") or ()
            if isinstance(keys, str):
                keys = (keys,)
            if any(str(key).lower() in {"enter", "return"} for key in keys):
                has_submit = True
    if has_text and has_submit:
        normalized_text = " ".join(" ".join(submitted_texts).split())
        digest = hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()[:12]
        return f"computer_text_submit:{digest}"
    return None


def _screen_refs_changed(
    before_ref: str,
    after_ref: str,
) -> tuple[bool, float | None]:
    if before_ref == after_ref:
        return False, 0.0
    difference = _visual_difference(before_ref, after_ref)
    if difference is None:
        return before_ref != after_ref, None
    return difference >= VISUAL_CHANGE_THRESHOLD, difference


def _visual_difference(before_ref: str, after_ref: str) -> float | None:
    before_path = _local_image_path(before_ref)
    after_path = _local_image_path(after_ref)
    if before_path is None or after_path is None:
        return None
    try:
        with Image.open(before_path) as before_image, Image.open(after_path) as after_image:
            before = before_image.convert("L").resize((64, 64))
            after = after_image.convert("L").resize((64, 64))
            diff = ImageChops.difference(before, after)
            return float(ImageStat.Stat(diff).mean[0])
    except Exception:
        return None


def _local_image_path(ref: str) -> Path | None:
    parsed = urlparse(ref)
    if parsed.scheme == "file":
        path = Path(url2pathname(unquote(parsed.path)))
        if parsed.netloc:
            path = Path(f"//{parsed.netloc}{unquote(parsed.path)}")
    elif not parsed.scheme:
        path = Path(ref)
    else:
        return None
    try:
        return path if path.exists() else None
    except OSError:
        return None
