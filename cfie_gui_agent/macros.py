from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cfie_client.protocol import ComputerAction


class ActionMacroError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class ActionMacroStep:
    type: str
    keys: tuple[str, ...] = ()
    text: str | None = None
    seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def keypress(cls, *keys: str) -> "ActionMacroStep":
        if not keys:
            raise ActionMacroError("keypress macro step requires at least one key")
        return cls(type="keypress", keys=tuple(str(key) for key in keys))

    @classmethod
    def wait(cls, seconds: float) -> "ActionMacroStep":
        if seconds < 0:
            raise ActionMacroError("wait seconds must be non-negative")
        return cls(type="wait", seconds=float(seconds))

    @classmethod
    def type_text(cls, text: str) -> "ActionMacroStep":
        return cls(type="type", text=str(text))

    def to_computer_action(self) -> ComputerAction:
        if self.type == "keypress":
            return ComputerAction(type="keypress", keys=self.keys)
        if self.type == "wait":
            return ComputerAction(type="wait", duration=self.seconds)
        if self.type == "type":
            return ComputerAction(type="type", text=self.text or "")
        raise ActionMacroError(f"unsupported macro step type: {self.type}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "keys": list(self.keys),
            "text": self.text,
            "seconds": self.seconds,
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class ActionMacro:
    name: str
    steps: tuple[ActionMacroStep, ...]
    description: str = ""
    max_repeat: int = 3
    human_like: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ActionMacroError("macro name is required")
        if not self.steps:
            raise ActionMacroError("macro requires at least one step")
        if self.max_repeat < 1:
            raise ActionMacroError("macro max_repeat must be >= 1")

    def expand(self, *, repeat: int = 1) -> tuple[ComputerAction, ...]:
        if repeat < 1:
            raise ActionMacroError("macro repeat must be >= 1")
        if repeat > self.max_repeat:
            raise ActionMacroError(
                f"macro repeat exceeds max_repeat: {repeat} > {self.max_repeat}"
            )
        actions: list[ComputerAction] = []
        for _ in range(repeat):
            actions.extend(step.to_computer_action() for step in self.steps)
        return tuple(actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "max_repeat": self.max_repeat,
            "human_like": self.human_like,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ActionMacroRegistry:
    macros: dict[str, ActionMacro] = field(default_factory=dict)

    def register(self, macro: ActionMacro) -> None:
        if macro.name in self.macros:
            raise ActionMacroError(f"duplicate macro: {macro.name}")
        self.macros[macro.name] = macro

    def require(self, name: str) -> ActionMacro:
        try:
            return self.macros[name]
        except KeyError as exc:
            raise ActionMacroError(f"unknown macro: {name}") from exc

    def expand(self, name: str, *, repeat: int = 1) -> tuple[ComputerAction, ...]:
        return self.require(name).expand(repeat=repeat)

    def to_context_payload(self) -> dict[str, Any]:
        return {
            "macros": [macro.to_dict() for macro in self.macros.values()],
        }
