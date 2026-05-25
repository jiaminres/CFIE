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
    "submit_current_input",
    "navigate_to_target",
    "read_text_file",
    "append_trace_note",
    "set_app_viewport",
    "record_workflow_result",
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


def _default_description(tool_name: str) -> str:
    descriptions = {
        "computer_use": (
            "Perform validated desktop computer actions. For Qwen VL grounding, "
            "set coordinate_space to qwen_normalized_1000 and express mouse "
            "coordinates on a 0..1000 image grid."
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
        "submit_current_input": (
            "Submit the currently focused text input. Prefer this after typing "
            "into a chat-style input when a visible send button should be pressed."
        ),
        "navigate_to_target": (
            "Ask the harness to move a source element toward a target while "
            "avoiding model-identified obstacles."
        ),
        "read_text_file": "Read a bounded UTF-8 text file that the harness has allowed.",
        "append_trace_note": "Append a structured note to the current trace.",
        "set_app_viewport": (
            "Set the APP viewport crop box so later screenshots include only "
            "the useful application region."
        ),
        "record_workflow_result": (
            "Persist one workflow item result. Keep arguments short; when item_id "
            "is available, omit input_text and expected_output because the harness "
            "can recover them from the workflow input file."
        ),
    }
    return descriptions.get(tool_name, tool_name)


def _default_parameters(tool_name: str) -> dict[str, Any]:
    schemas: dict[str, dict[str, Any]] = {
        "computer_use": _object_schema(
            {
                "coordinate_space": {
                    "type": "string",
                    "enum": ["qwen_normalized_1000", "screenshot"],
                    "description": (
                        "Required. Use qwen_normalized_1000 for Qwen VL: "
                        "(0,0) is the current image top-left and "
                        "(1000,1000) is bottom-right. Use screenshot only "
                        "when actions are already in screenshot pixel coordinates."
                    ),
                },
                "actions": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
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
                "objective": {"type": "string"},
                "stop_condition": {"type": "string"},
            },
            required=("macro_name",),
        ),
        "submit_current_input": _object_schema(
            {
                "method": {
                    "type": "string",
                    "enum": ["auto", "click_send_button", "enter"],
                },
                "reason": {"type": "string"},
            },
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
        "set_app_viewport": _object_schema(
            {
                "x": {"type": "integer", "minimum": 0},
                "y": {"type": "integer", "minimum": 0},
                "width": {"type": "integer", "minimum": 1},
                "height": {"type": "integer", "minimum": 1},
                "coordinate_space": {
                    "type": "string",
                    "enum": ["screenshot", "physical"],
                    "description": (
                        "Use 'screenshot' when coordinates refer to the image "
                        "shown to the model; use 'physical' for full desktop pixels."
                    ),
                },
                "reason": {"type": "string"},
            },
            required=("x", "y", "width", "height"),
        ),
        "record_workflow_result": _object_schema(
            {
                "item_id": {"type": "string", "minLength": 1},
                "input_text": {
                    "type": "string",
                    "description": "Optional. Prefer omitting this to keep tool output short.",
                },
                "expected_output": {
                    "type": "string",
                    "description": "Optional. Prefer omitting this to keep tool output short.",
                },
                "output_text": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["passed", "failed", "uncertain", "skipped", "error"],
                },
                "latency_seconds": {"type": "number"},
                "artifact_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "reason": {"type": "string"},
            },
            required=("item_id", "output_text", "status"),
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
) -> dict[str, Any]:
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
