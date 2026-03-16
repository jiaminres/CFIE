"""Validation helpers for CFIE configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def validate_choice(name: str, value: str, choices: Iterable[str]) -> str:
    allowed = tuple(choices)
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return value


def validate_positive_int(name: str,
                          value: int,
                          *,
                          allow_zero: bool = False) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}, got {value}")
    return value


def validate_non_negative_float(name: str, value: float) -> float:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def validate_ratio(name: str, value: float) -> float:
    if not (0 < value <= 1):
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def validate_non_empty_string(name: str, value: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def normalize_path(path: str) -> str:
    return str(Path(path).expanduser())
