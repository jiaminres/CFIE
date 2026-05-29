from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cfie_gui_agent.desktop_client import (
    DEFAULT_CLIENT_STATE_PATH,
    DEFAULT_TRACE_DIR,
    DesktopClientState,
    MacroConfig,
    ReferenceAsset,
    TargetAppConfig,
)
from cfie_gui_agent.human_loop import (
    HUMAN_REQUEST_RESOLVED,
    HumanReply,
    HumanRequest,
    HumanRequestState,
    InMemoryHumanChannel,
)
from cfie_gui_agent.jobs import JobState
from cfie_gui_agent.macros import ActionMacro, ActionMacroStep

STATE_SCHEMA_VERSION = 1


def load_desktop_state(path: str | Path = DEFAULT_CLIENT_STATE_PATH) -> DesktopClientState:
    state_path = Path(path)
    if not state_path.exists():
        return DesktopClientState(settings=_default_settings())
    try:
        text = state_path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = json.JSONDecoder(strict=False).decode(text)
    except (OSError, json.JSONDecodeError):
        return DesktopClientState(settings=_default_settings())
    if not isinstance(data, dict):
        return DesktopClientState(settings=_default_settings())

    state = DesktopClientState(settings=_settings_from_data(data))
    _load_apps(state, data)
    _load_macros(state, data)
    _load_human_requests(state, data)

    selected = str(data.get("selected_app_id") or "")
    if selected in state.target_apps:
        state.selected_app_id = selected
    elif state.target_apps:
        state.selected_app_id = next(iter(state.target_apps))
    _load_selected_trace(state)
    return state


def save_desktop_state(
    state: DesktopClientState,
    path: str | Path = DEFAULT_CLIENT_STATE_PATH,
) -> Path:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": STATE_SCHEMA_VERSION,
        "selected_app_id": state.selected_app_id,
        "settings": {**_default_settings(), **dict(state.settings or {})},
        "apps": {
            app_id: config.to_dict()
            for app_id, config in state.target_apps.items()
        },
        "macros": state.action_macros.to_context_payload()["macros"],
        "human_requests": list(
            state.human_loop.list_requests(include_completed=True)
        ),
        "trace": state.trace_store.to_dict(),
    }
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(state_path)
    return state_path


def _default_settings() -> dict[str, Any]:
    return {
        "default_trace_dir": str(DEFAULT_TRACE_DIR),
        "default_image_detail": "high",
        "auto_load_last_session": True,
    }


def _settings_from_data(data: dict[str, Any]) -> dict[str, Any]:
    settings = data.get("settings")
    if not isinstance(settings, dict):
        settings = {}
    return {**_default_settings(), **settings}


def _load_apps(state: DesktopClientState, data: dict[str, Any]) -> None:
    apps = data.get("apps")
    if apps is None:
        apps = data.get("target_apps")
    if not isinstance(apps, dict):
        return
    for app_id, raw_config in apps.items():
        if not isinstance(raw_config, dict):
            continue
        config = _target_app_from_dict(app_id, raw_config)
        if config is None:
            continue
        state.add_target_app(config)
        if config.job_id not in state.job_board.jobs:
            state.job_board.add_job(
                JobState(
                    job_id=config.job_id,
                    target_app=config.app_name,
                    goal=config.task_description or config.app_name,
                )
            )


def _target_app_from_dict(
    app_id_hint: str,
    raw: dict[str, Any],
) -> TargetAppConfig | None:
    app_id = str(raw.get("app_id") or app_id_hint or "").strip()
    app_name = str(raw.get("app_name") or "").strip()
    job_id = str(raw.get("job_id") or f"job:{app_id}").strip()
    if not app_id or not app_name:
        return None
    assets = tuple(
        asset
        for asset in (
            _reference_asset_from_dict(item)
            for item in raw.get("reference_assets") or ()
            if isinstance(item, dict)
        )
        if asset is not None
    )
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return TargetAppConfig(
        app_id=app_id,
        app_name=app_name,
        job_id=job_id,
        task_description=str(raw.get("task_description") or ""),
        reference_assets=assets,
        metadata=dict(metadata),
    )


def _reference_asset_from_dict(raw: dict[str, Any]) -> ReferenceAsset | None:
    asset_id = str(raw.get("asset_id") or "").strip()
    kind = str(raw.get("kind") or "").strip()
    path = str(raw.get("path") or "").strip()
    if not asset_id or not kind or not path:
        return None
    return ReferenceAsset(
        asset_id=asset_id,
        kind=kind,
        path=path,
        title=str(raw.get("title") or ""),
        description=str(raw.get("description") or ""),
    )


def _load_macros(state: DesktopClientState, data: dict[str, Any]) -> None:
    macros = data.get("macros")
    if isinstance(macros, dict):
        macros = macros.get("macros")
    if not isinstance(macros, list):
        return
    for raw_macro in macros:
        if not isinstance(raw_macro, dict):
            continue
        config = _macro_config_from_dict(raw_macro)
        try:
            if config is not None:
                state.register_macro(config)
                continue
            macro = _action_macro_from_dict(raw_macro)
            if macro is not None:
                state.action_macros.upsert(macro)
        except Exception:
            continue


def _macro_config_from_dict(raw: dict[str, Any]) -> MacroConfig | None:
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    sequence = str(raw.get("sequence") or metadata.get("sequence") or "").strip()
    name = str(raw.get("name") or "").strip()
    if not name or not sequence:
        return None
    return MacroConfig(
        name=name,
        description=str(raw.get("description") or ""),
        sequence=sequence,
        scope=str(raw.get("scope") or metadata.get("scope") or "global"),
        app_id=raw.get("app_id") or metadata.get("app_id"),
    )


def _action_macro_from_dict(raw: dict[str, Any]) -> ActionMacro | None:
    name = str(raw.get("name") or "").strip()
    if not name:
        return None
    steps: list[ActionMacroStep] = []
    for raw_step in raw.get("steps") or []:
        if not isinstance(raw_step, dict):
            continue
        step_type = str(raw_step.get("type") or "").strip()
        if step_type == "computer" and isinstance(raw_step.get("action"), dict):
            steps.append(
                ActionMacroStep.computer_action(
                    raw_step["action"],
                    purpose=str(raw_step.get("purpose") or ""),
                )
            )
        elif step_type == "keypress":
            keys = raw_step.get("keys") or []
            if isinstance(keys, list) and keys:
                steps.append(ActionMacroStep.keypress(*(str(key) for key in keys)))
        elif step_type == "wait":
            steps.append(ActionMacroStep.wait(float(raw_step.get("seconds") or 0)))
        elif step_type == "type":
            steps.append(ActionMacroStep.type_text(str(raw_step.get("text") or "")))
    if not steps:
        return None
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return ActionMacro(
        name=name,
        description=str(raw.get("description") or ""),
        steps=tuple(steps),
        max_repeat=int(raw.get("max_repeat") or 3),
        human_like=bool(raw.get("human_like", True)),
        metadata=dict(metadata),
    )


def _load_human_requests(state: DesktopClientState, data: dict[str, Any]) -> None:
    requests = data.get("human_requests")
    if not isinstance(requests, list):
        return
    channel = state.human_loop.channel
    for raw_state in requests:
        if not isinstance(raw_state, dict):
            continue
        request = _human_request_from_dict(raw_state.get("request"))
        if request is None:
            continue
        reply = _human_reply_from_dict(raw_state.get("reply"))
        status = str(raw_state.get("status") or "pending")
        restored = HumanRequestState(
            request=request,
            status=status,
            claimed_by=raw_state.get("claimed_by"),
            reply=reply,
            result_task=(
                raw_state.get("result_task")
                if isinstance(raw_state.get("result_task"), dict)
                else None
            ),
        )
        if status == HUMAN_REQUEST_RESOLVED:
            state.human_loop.completed[request.request_id] = restored
        else:
            state.human_loop.states[request.request_id] = restored
            state.human_loop.pending[request.request_id] = request
            if isinstance(channel, InMemoryHumanChannel):
                channel.sent_requests.append(request)


def _human_request_from_dict(raw: Any) -> HumanRequest | None:
    if not isinstance(raw, dict):
        return None
    request_id = str(raw.get("request_id") or "").strip()
    question = str(raw.get("question") or "").strip()
    if not request_id or not question:
        return None
    evidence = raw.get("evidence_refs")
    if not isinstance(evidence, list):
        evidence = []
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return HumanRequest(
        request_id=request_id,
        task_id=raw.get("task_id"),
        question=question,
        blocking=bool(raw.get("blocking", True)),
        evidence_refs=tuple(str(item) for item in evidence),
        risk_reason=raw.get("risk_reason"),
        proposed_action=raw.get("proposed_action"),
        allowed_reply_format=raw.get("allowed_reply_format"),
        urgency=str(raw.get("urgency") or "normal"),
        metadata=dict(metadata),
    )


def _human_reply_from_dict(raw: Any) -> HumanReply | None:
    if not isinstance(raw, dict):
        return None
    request_id = str(raw.get("request_id") or "").strip()
    text = str(raw.get("text") or "")
    if not request_id:
        return None
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return HumanReply(request_id=request_id, text=text, metadata=dict(metadata))


def _load_selected_trace(state: DesktopClientState) -> None:
    config = state.target_apps.get(state.selected_app_id)
    if config is None:
        return
    trace_path = str((config.metadata or {}).get("trace_path") or "").strip()
    if not trace_path:
        return
    state.trace_store.load_existing(trace_path)
