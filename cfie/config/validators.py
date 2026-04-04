"""Validation helpers for CFIE configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def validate_choice(name: str, value: str, choices: Iterable[str]) -> str:
    # 先把可迭代对象冻结成 tuple，便于复用和报错输出。
    allowed = tuple(choices)
    # 若值不在允许集合中，直接抛出带完整候选项的错误。
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    # 校验通过后原样返回。
    return value


def validate_positive_int(name: str,
                          value: int,
                          *,
                          allow_zero: bool = False) -> int:
    # 先校验传入值是否真的是 int。
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    # 根据 allow_zero 的语义分别限制“正数”或“非负数”。
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}, got {value}")
    return value


def validate_non_negative_float(name: str, value: float) -> float:
    # 浮点数只允许非负。
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def validate_ratio(name: str, value: float) -> float:
    # 比例类参数统一要求落在 (0, 1] 区间。
    if not (0 < value <= 1):
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def validate_non_empty_string(name: str, value: str) -> str:
    # 空字符串或纯空白字符串都视为非法。
    if not value or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def normalize_path(path: str) -> str:
    # 统一展开 `~` 等用户目录语法，返回规范化字符串路径。
    return str(Path(path).expanduser())
