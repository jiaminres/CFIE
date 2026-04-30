"""Minimal predictor runtime types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _require_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


@dataclass(slots=True, frozen=True)
class BatchShape:
    samples: int
    tokens_per_sample: int
    source_kind: str = "synthetic_shape"
    dataset_name: str | None = None
    sample_indices: tuple[int, ...] = ()
    loss_token_count: int = 0
    token_rows: tuple[tuple[int, ...], ...] = ()
    target_rows: tuple[tuple[int, ...], ...] = ()
    attention_mask_rows: tuple[tuple[int, ...], ...] = ()
    target_attention_mask_rows: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        _require_positive_int("samples", self.samples)
        _require_positive_int("tokens_per_sample", self.tokens_per_sample)
        _require_non_negative_int("loss_token_count", self.loss_token_count)
        if self.token_rows and len(self.token_rows) != self.samples:
            raise ValueError("token_rows length must match samples when provided")
        if self.target_rows and len(self.target_rows) != self.samples:
            raise ValueError("target_rows length must match samples when provided")
        if self.attention_mask_rows and len(self.attention_mask_rows) != self.samples:
            raise ValueError(
                "attention_mask_rows length must match samples when provided"
            )
        if (
            self.target_attention_mask_rows
            and len(self.target_attention_mask_rows) != self.samples
        ):
            raise ValueError(
                "target_attention_mask_rows length must match samples when provided"
            )
        for row in self.token_rows:
            if len(row) != self.tokens_per_sample:
                raise ValueError(
                    "each token_rows entry must match tokens_per_sample"
                )
        for row in self.target_rows:
            if len(row) != self.tokens_per_sample:
                raise ValueError(
                    "each target_rows entry must match tokens_per_sample"
                )
        for row in self.attention_mask_rows:
            if len(row) != self.tokens_per_sample:
                raise ValueError(
                    "each attention_mask_rows entry must match tokens_per_sample"
                )
            if any(value not in (0, 1, False, True) for value in row):
                raise ValueError("attention_mask_rows entries must be 0 or 1")
        for row in self.target_attention_mask_rows:
            if len(row) != self.tokens_per_sample:
                raise ValueError(
                    "each target_attention_mask_rows entry must match "
                    "tokens_per_sample"
                )
            if any(value not in (0, 1, False, True) for value in row):
                raise ValueError("target_attention_mask_rows entries must be 0 or 1")
        if self.attention_mask_rows and not self.token_rows:
            raise ValueError("attention_mask_rows requires token_rows")
        if self.target_attention_mask_rows and not self.target_rows:
            raise ValueError("target_attention_mask_rows requires target_rows")
        for row in self.attention_mask_rows:
            if tuple(row) != tuple(sorted(row, reverse=True)):
                raise ValueError("attention_mask_rows must use tail padding")
        for row in self.target_attention_mask_rows:
            if tuple(row) != tuple(sorted(row, reverse=True)):
                raise ValueError("target_attention_mask_rows must use tail padding")
        if self.sample_indices and len(self.sample_indices) != self.samples:
            raise ValueError("sample_indices length must match samples when provided")

    @property
    def total_tokens(self) -> int:
        return self.samples * self.tokens_per_sample

    @property
    def has_token_rows(self) -> bool:
        return bool(self.token_rows)

    @property
    def has_attention_mask_rows(self) -> bool:
        return bool(self.attention_mask_rows)

    @property
    def has_target_attention_mask_rows(self) -> bool:
        return bool(self.target_attention_mask_rows)

    @property
    def valid_token_count(self) -> int:
        if self.attention_mask_rows:
            return sum(int(value) for row in self.attention_mask_rows for value in row)
        return self.total_tokens

    @property
    def valid_loss_token_count(self) -> int:
        if self.target_attention_mask_rows:
            return sum(
                int(value)
                for row in self.target_attention_mask_rows
                for value in row
            )
        return self.loss_token_count or self.total_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "tokens_per_sample": self.tokens_per_sample,
            "total_tokens": self.total_tokens,
            "source_kind": self.source_kind,
            "dataset_name": self.dataset_name,
            "sample_indices": list(self.sample_indices),
            "loss_token_count": self.loss_token_count,
            "has_token_rows": self.has_token_rows,
            "has_attention_mask_rows": self.has_attention_mask_rows,
            "has_target_attention_mask_rows": self.has_target_attention_mask_rows,
            "valid_token_count": self.valid_token_count,
            "valid_loss_token_count": self.valid_loss_token_count,
        }


@dataclass(slots=True, frozen=True)
class BatchPlannerCheckpoint:
    planner_kind: str
    base_samples: int
    tokens_per_sample: int
    dataset_path: str | None = None
    tokenizer_path: str | None = None
    dataset_format: str = "auto"
    dataset_text_key: str = "text"

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_kind": self.planner_kind,
            "base_samples": self.base_samples,
            "tokens_per_sample": self.tokens_per_sample,
            "dataset_path": self.dataset_path,
            "tokenizer_path": self.tokenizer_path,
            "dataset_format": self.dataset_format,
            "dataset_text_key": self.dataset_text_key,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BatchPlannerCheckpoint":
        return cls(
            planner_kind=str(payload["planner_kind"]),
            base_samples=int(payload["base_samples"]),
            tokens_per_sample=int(payload["tokens_per_sample"]),
            dataset_path=payload.get("dataset_path"),
            tokenizer_path=payload.get("tokenizer_path"),
            dataset_format=str(payload.get("dataset_format", "auto")),
            dataset_text_key=str(payload.get("dataset_text_key", "text")),
        )
