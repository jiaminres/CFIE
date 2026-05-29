from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MODEL_CALLABLE_TOOLS = (
    "computer_use",
    "read_image",
    "read_video_clip",
    "request_human_help",
    "update_constraints",
    "finish_subtask",
    "report_blocked",
    "ask_replan",
    "query_memory",
    "run_action_macro",
    "propose_action_macro",
    "navigate_to_target",
    "read_text_file",
    "write_text_file",
    "append_text_file",
    "open_url",
    "launch_app",
    "run_shell",
)

CORE_MODEL_CALLABLE_TOOLS = (
    "computer_use",
    "request_human_help",
    "finish_subtask",
    "report_blocked",
    "run_action_macro",
    "propose_action_macro",
    "read_text_file",
    "write_text_file",
    "append_text_file",
    "open_url",
    "launch_app",
    "run_shell",
)

LEGACY_AGENT_TOOL_NAMES = (
    "append_trace_note",
)

MINIMAL_MODEL_CALLABLE_TOOLS = (
    "computer_use",
    "request_human_help",
    "finish_subtask",
    "open_url",
    "run_action_macro",
)

HARNESS_INTERNAL_TOOLS = (
    "decode_video",
    "extract_video_frames",
    "resize_image",
    "run_ocr",
    "select_frame_budget",
    "execute_compaction",
    "validate_compaction_plan",
    "write_trace",
    "calculate_reward",
    "schedule_task_queue",
    "wechat_send",
    "wechat_poll",
    "validate_coordinates",
    "validate_safety_policy",
    "detect_screen_change",
    "detect_repeated_action",
    "manage_window_focus",
    "persist_artifact",
    "serialize_prompt_context",
    "retry_tool_execution",
    "check_human_reply",
    "run_shell_command",
)

MODEL_TOOL_PROFILES = {
    "minimal": MINIMAL_MODEL_CALLABLE_TOOLS,
    "core": CORE_MODEL_CALLABLE_TOOLS,
    "full": MODEL_CALLABLE_TOOLS,
}


class ToolRegistryError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class ModelToolSpec:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class ModelToolRegistry:
    allowed_tool_names: tuple[str, ...] = MODEL_CALLABLE_TOOLS
    internal_tool_names: tuple[str, ...] = HARNESS_INTERNAL_TOOLS
    specs: dict[str, ModelToolSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.specs:
            self.specs = {
                name: ModelToolSpec(
                    name=name,
                    description=_default_description(name),
                    parameters=_default_parameters(name),
                )
                for name in self.allowed_tool_names
            }

    def is_model_callable(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tool_names

    def is_internal(self, tool_name: str) -> bool:
        return tool_name in self.internal_tool_names

    def validate_model_tool(self, tool_name: str) -> None:
        if self.is_internal(tool_name):
            raise ToolRegistryError(
                f"{tool_name!r} is harness-internal and cannot be called by model"
            )
        if not self.is_model_callable(tool_name):
            raise ToolRegistryError(f"model tool is not allowlisted: {tool_name!r}")

    def validate_model_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        self.validate_model_tool(tool_name)
        spec = self.specs.get(tool_name)
        if spec is None:
            raise ToolRegistryError(f"missing tool spec: {tool_name!r}")
        _validate_object_schema(
            spec.parameters,
            dict(arguments or {}),
            path=tool_name,
        )

    def model_tools(self) -> tuple[ModelToolSpec, ...]:
        return tuple(self.specs[name] for name in self.allowed_tool_names)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [spec.to_openai_tool() for spec in self.model_tools()]


def model_tool_names_for_profile(profile: str | None) -> tuple[str, ...]:
    key = (profile or "core").strip().lower()
    if key not in MODEL_TOOL_PROFILES:
        raise ToolRegistryError(f"unknown model tool profile: {profile!r}")
    return MODEL_TOOL_PROFILES[key]


def _default_description(tool_name: str) -> str:
    descriptions = {
        "computer_use": (
            "Perform validated desktop computer actions. For Qwen VL grounding, "
            "set coordinate_space to qwen_normalized_1000 and express mouse "
            "coordinates on a 0..1000 image grid. Use local_refinement_1000 "
            "only after the harness provides a local refinement crop. When one "
            "model response returns multiple tool calls, give each tool call a "
            "positive integer index starting at 1. When one computer_use call "
            "contains multiple actions, give every action its own index too."
        ),
        "read_image": "Read a referenced image artifact into model context.",
        "read_video_clip": "Read a bounded video clip or selected frame set.",
        "request_human_help": "Ask a manager for human intervention.",
        "update_constraints": "Propose a structured policy update.",
        "finish_subtask": "Report that the current subtask appears complete.",
        "report_blocked": "Report that the current task is blocked.",
        "ask_replan": "Ask TaskManager to consider a task transition.",
        "query_memory": "Query workspace or business memory.",
        "run_action_macro": "Execute a registered low-latency action macro.",
        "propose_action_macro": (
            "Suggest a reusable operation macro for human approval. Use this "
            "for repeated multi-step UI actions that should later run with one "
            "macro call plus dynamic parameters."
        ),
        "navigate_to_target": (
            "Ask the harness to move a source element toward a target while "
            "avoiding model-identified obstacles."
        ),
        "read_text_file": "Read a bounded UTF-8 text file that the harness has allowed.",
        "write_text_file": "Write UTF-8 text to a local file for the current task.",
        "append_text_file": "Append UTF-8 text to a local file for the current task.",
        "open_url": "Open a URL in the local desktop browser.",
        "launch_app": "Launch a local desktop application.",
        "run_shell": "Run a bounded local shell command and return stdout/stderr.",
        "append_trace_note": "Append a structured note to the current trace.",
    }
    return descriptions.get(tool_name, tool_name)


def _default_parameters(tool_name: str) -> dict[str, Any]:
    schemas: dict[str, dict[str, Any]] = {
        "computer_use": _object_schema(
            {
                "coordinate_space": {
                    "type": "string",
                    "enum": [
                        "qwen_normalized_1000",
                        "local_refinement_1000",
                        "screenshot",
                    ],
                    "description": (
                        "Required. Use qwen_normalized_1000 for Qwen VL: "
                        "(0,0) is the current image top-left and "
                        "(1000,1000) is bottom-right. Use "
                        "local_refinement_1000 only when the harness has just "
                        "provided a local refinement crop. Use screenshot only "
                        "when actions are already in screenshot pixel coordinates."
                    ),
                },
                "actions": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "minimum": 1,
                                "description": (
                                    "Optional execution order. Required when "
                                    "returning multiple actions; use 1, 2, 3..."
                                ),
                            },
                            "type": {"type": "string"},
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                            "button": {"type": "string"},
                            "keys": {
                                "anyOf": [
                                    {"type": "string"},
                                    {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                ]
                            },
                            "text": {"type": "string"},
                            "path": {"type": "array"},
                            "scroll_x": {"type": "integer"},
                            "scroll_y": {"type": "integer"},
                            "seconds": {"type": "number"},
                        },
                        "required": ["type"],
                    },
                }
            },
            required=("coordinate_space", "actions"),
        ),
        "read_image": _object_schema(
            {
                "image_ref": {
                    "type": "string",
                    "description": "Artifact URI, local path, or image id.",
                },
                "detail": {
                    "type": "string",
                    "enum": ["low", "auto", "high", "original"],
                },
                "purpose": {"type": "string"},
            },
            required=("image_ref",),
        ),
        "read_video_clip": _object_schema(
            {
                "video_ref": {
                    "type": "string",
                    "description": "Artifact URI, local path, or video id.",
                },
                "start_seconds": {"type": "number"},
                "end_seconds": {"type": "number"},
                "frame_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "max_frames": {"type": "integer", "minimum": 1},
                "purpose": {"type": "string"},
            },
            required=("video_ref",),
        ),
        "request_human_help": _object_schema(
            {
                "question": {"type": "string", "minLength": 1},
                "urgency": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "critical"],
                },
                "evidence_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "risk_reason": {"type": "string"},
                "proposed_action": {"type": "string"},
                "allowed_reply_format": {"type": "string"},
                "blocking": {
                    "type": "boolean",
                    "description": (
                        "true when the current subtask cannot continue until "
                        "a human replies, such as captcha or account-risk "
                        "confirmation. false when other work can continue "
                        "while waiting, such as a buyer requesting manual "
                        "customer-service wording."
                    ),
                },
            },
            required=("question",),
        ),
        "update_constraints": _object_schema(
            {
                "summary": {"type": "string", "minLength": 1},
                "constraints": {"type": "object"},
                "reason": {"type": "string"},
            },
            required=("summary",),
        ),
        "finish_subtask": _object_schema(
            {
                "completion_reason": {"type": "string"},
                "evidence_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        ),
        "report_blocked": _object_schema(
            {
                "blocked_reason": {"type": "string", "minLength": 1},
                "evidence_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "retry_suggestion": {"type": "string"},
            },
            required=("blocked_reason",),
        ),
        "ask_replan": _object_schema(
            {
                "suggested_transition": {
                    "type": "string",
                    "enum": [
                        "continue",
                        "interrupt",
                        "override",
                        "rollback",
                        "switch_job",
                    ],
                },
                "reason": {"type": "string", "minLength": 1},
                "target_job_id": {"type": "string"},
                "target_subtask_goal": {"type": "string"},
            },
            required=("suggested_transition", "reason"),
        ),
        "query_memory": _object_schema(
            {
                "query": {"type": "string", "minLength": 1},
                "scope": {
                    "type": "string",
                    "enum": ["current_job", "all_jobs", "business_profile"],
                },
                "limit": {"type": "integer", "minimum": 1},
            },
            required=("query",),
        ),
        "run_action_macro": _object_schema(
            {
                "macro_name": {"type": "string", "minLength": 1},
                "repeat": {"type": "integer", "minimum": 1},
                "parameters": {"type": "object"},
                "objective": {"type": "string"},
                "stop_condition": {"type": "string"},
            },
            required=("macro_name",),
        ),
        "propose_action_macro": _object_schema(
            {
                "macro_name": {"type": "string", "minLength": 1},
                "description": {"type": "string"},
                "reason": {"type": "string"},
                "scope": {
                    "type": "string",
                    "enum": ["current_app", "global"],
                },
                "dynamic_parameters": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "items": _object_schema(
                        {
                            "index": {"type": "integer", "minimum": 1},
                            "purpose": {"type": "string", "minLength": 1},
                            "action": {"type": "object"},
                        },
                        required=("index", "purpose", "action"),
                        allow_index=False,
                    ),
                },
            },
            required=("macro_name", "steps"),
        ),
        "navigate_to_target": _object_schema(
            {
                "source": _point_schema(),
                "target": _point_schema(),
                "target_label": {"type": "string"},
                "obstacles": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": _point_schema(),
                    },
                },
                "arrive_radius": {"type": "integer", "minimum": 1},
                "max_seconds": {"type": "number", "minimum": 0.1},
                "objective": {"type": "string"},
            },
            required=("source", "target"),
        ),
        "read_text_file": _object_schema(
            {
                "path": {"type": "string", "minLength": 1},
                "max_chars": {"type": "integer", "minimum": 1},
                "purpose": {"type": "string"},
            },
            required=("path",),
        ),
        "write_text_file": _object_schema(
            {
                "path": {"type": "string", "minLength": 1},
                "text": {"type": "string"},
                "encoding": {"type": "string"},
                "create_parent": {"type": "boolean"},
                "purpose": {"type": "string"},
            },
            required=("path", "text"),
        ),
        "append_text_file": _object_schema(
            {
                "path": {"type": "string", "minLength": 1},
                "text": {"type": "string"},
                "encoding": {"type": "string"},
                "create_parent": {"type": "boolean"},
                "purpose": {"type": "string"},
            },
            required=("path", "text"),
        ),
        "open_url": _object_schema(
            {
                "url": {"type": "string", "minLength": 1},
                "browser_path": {"type": "string"},
                "new_window": {"type": "boolean"},
                "purpose": {"type": "string"},
            },
            required=("url",),
        ),
        "launch_app": _object_schema(
            {
                "executable_path": {"type": "string", "minLength": 1},
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "cwd": {"type": "string"},
                "purpose": {"type": "string"},
            },
            required=("executable_path",),
        ),
        "run_shell": _object_schema(
            {
                "command": {"type": "string", "minLength": 1},
                "cwd": {"type": "string"},
                "timeout_seconds": {"type": "number", "minimum": 0.1},
                "purpose": {"type": "string"},
            },
            required=("command",),
        ),
        "append_trace_note": _object_schema(
            {
                "title": {"type": "string", "minLength": 1},
                "summary": {"type": "string"},
                "status": {"type": "string"},
                "artifact_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "metadata": {"type": "object"},
            },
            required=("title",),
        ),
    }
    return schemas.get(tool_name, _object_schema({}))


def _point_schema() -> dict[str, Any]:
    return {
        "anyOf": [
            {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
            },
            _object_schema(
                {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                required=("x", "y"),
            ),
        ]
    }


def _object_schema(
    properties: dict[str, Any],
    *,
    required: tuple[str, ...] = (),
    allow_index: bool = True,
) -> dict[str, Any]:
    if allow_index and "index" not in properties:
        properties = {
            "index": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Optional order of this tool call within the current "
                    "model response. Use 1, 2, 3... when returning multiple "
                    "tool calls."
                ),
            },
            **properties,
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _validate_object_schema(
    schema: dict[str, Any],
    value: dict[str, Any],
    *,
    path: str,
) -> None:
    if schema.get("type") != "object":
        raise ToolRegistryError(f"{path} schema must be an object schema")
    properties = schema.get("properties", {}) or {}
    for required_key in schema.get("required", ()) or ():
        if required_key not in value:
            raise ToolRegistryError(f"{path}.{required_key} is required")
    if schema.get("additionalProperties") is False:
        for key in value:
            if key not in properties:
                raise ToolRegistryError(f"{path}.{key} is not a valid argument")
    for key, item in value.items():
        if key not in properties:
            continue
        _validate_schema_value(
            properties[key],
            item,
            path=f"{path}.{key}",
        )


def _validate_schema_value(schema: dict[str, Any], value: Any, *, path: str) -> None:
    if "anyOf" in schema:
        errors: list[Exception] = []
        for option in schema["anyOf"]:
            try:
                _validate_schema_value(option, value, path=path)
                return
            except ToolRegistryError as exc:
                errors.append(exc)
        raise ToolRegistryError(f"{path} does not match any allowed shape") from (
            errors[0] if errors else None
        )

    expected_type = schema.get("type")
    if expected_type is None:
        return
    if not _matches_type(expected_type, value):
        raise ToolRegistryError(f"{path} must be {expected_type}")
    if isinstance(value, str):
        min_length = schema.get("minLength")
        if min_length is not None and len(value) < int(min_length):
            raise ToolRegistryError(f"{path} is too short")
        enum_values = schema.get("enum")
        if enum_values is not None and value not in enum_values:
            raise ToolRegistryError(f"{path} must be one of {enum_values}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            raise ToolRegistryError(f"{path} must be >= {minimum}")
    if isinstance(value, list):
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < int(min_items):
            raise ToolRegistryError(f"{path} has too few items")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                _validate_schema_value(
                    item_schema,
                    item,
                    path=f"{path}[{index}]",
                )
    if isinstance(value, dict) and schema.get("properties"):
        _validate_object_schema(schema, value, path=path)


def _matches_type(expected_type: str, value: Any) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return True
