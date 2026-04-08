"""Predictor online-expert state helpers for inference integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


def _normalize_expert_ids(expert_ids: Iterable[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for expert_id in expert_ids:
        parsed = int(expert_id)
        if parsed < 0 or parsed in seen:
            continue
        normalized.append(parsed)
        seen.add(parsed)
    return tuple(normalized)


@dataclass(slots=True, frozen=True)
class PredictorOnlineExpertState:
    source: str
    active_expert_ids: tuple[int, ...] = ()
    prefetched_expert_ids: tuple[int, ...] = ()
    hot_expert_ids: tuple[int, ...] = ()
    prefetch_priority_expert_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", str(self.source).strip())
        object.__setattr__(
            self,
            "active_expert_ids",
            _normalize_expert_ids(self.active_expert_ids),
        )
        object.__setattr__(
            self,
            "prefetched_expert_ids",
            _normalize_expert_ids(self.prefetched_expert_ids),
        )
        object.__setattr__(
            self,
            "hot_expert_ids",
            _normalize_expert_ids(self.hot_expert_ids),
        )
        object.__setattr__(
            self,
            "prefetch_priority_expert_ids",
            _normalize_expert_ids(self.prefetch_priority_expert_ids),
        )

    @property
    def cpu_hot_expert_ids(self) -> tuple[int, ...]:
        if self.hot_expert_ids:
            return self.hot_expert_ids
        return self.active_expert_ids

    @property
    def reachable_expert_ids(self) -> tuple[int, ...]:
        return _normalize_expert_ids(
            self.active_expert_ids
            + self.prefetched_expert_ids
            + self.hot_expert_ids
            + self.prefetch_priority_expert_ids
        )

    @staticmethod
    def _expert_id_stats(expert_ids: tuple[int, ...]) -> tuple[float, float, float, float, float]:
        if not expert_ids:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        expert_tensor = torch.tensor(expert_ids, dtype=torch.float32)
        return (
            float(expert_tensor.numel()),
            float(expert_tensor.mean().item()),
            float(expert_tensor.std(unbiased=False).item()),
            float(expert_tensor.min().item()),
            float(expert_tensor.max().item()),
        )

    def summary_stats(self) -> tuple[float, ...]:
        return (
            *self._expert_id_stats(self.active_expert_ids),
            *self._expert_id_stats(self.prefetched_expert_ids),
            *self._expert_id_stats(self.hot_expert_ids),
            *self._expert_id_stats(self.prefetch_priority_expert_ids),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "active_expert_ids": list(self.active_expert_ids),
            "prefetched_expert_ids": list(self.prefetched_expert_ids),
            "hot_expert_ids": list(self.hot_expert_ids),
            "prefetch_priority_expert_ids": list(
                self.prefetch_priority_expert_ids
            ),
            "cpu_hot_expert_ids": list(self.cpu_hot_expert_ids),
            "reachable_expert_ids": list(self.reachable_expert_ids),
        }

    @classmethod
    def from_logical_replica_count(
        cls,
        logical_replica_count: torch.Tensor,
        *,
        source: str = "eplb_logical_replica_count",
    ) -> "PredictorOnlineExpertState":
        online_mask = logical_replica_count.reshape(-1).to(dtype=torch.float32) > 0
        online_expert_ids = tuple(
            int(expert_id)
            for expert_id in torch.nonzero(online_mask, as_tuple=False).flatten().tolist()
        )
        return cls(
            source=source,
            active_expert_ids=online_expert_ids,
            hot_expert_ids=online_expert_ids,
            prefetch_priority_expert_ids=online_expert_ids,
        )
