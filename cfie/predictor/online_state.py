"""Predictor online-expert state helpers for inference integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch

DEFAULT_PREDICTOR_HOTNESS_DECAY = 0.35


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


@dataclass(slots=True)
class PredictorObservedRoutingTracker:
    hotness_decay: float = DEFAULT_PREDICTOR_HOTNESS_DECAY
    smoothed_scores_by_layer: dict[int, dict[int, float]] = field(
        default_factory=dict
    )
    current_counts_by_layer: dict[int, dict[int, float]] = field(
        default_factory=dict
    )

    def start_step(self) -> None:
        self.current_counts_by_layer.clear()

    def observe(
        self,
        *,
        layer_index: int,
        expert_ids: torch.Tensor,
    ) -> None:
        flattened = expert_ids.detach().reshape(-1).to(device="cpu", dtype=torch.int64)
        if flattened.numel() == 0:
            return
        unique_expert_ids, counts = torch.unique(flattened, sorted=False, return_counts=True)
        layer_counts = self.current_counts_by_layer.setdefault(int(layer_index), {})
        for expert_id, count in zip(unique_expert_ids.tolist(), counts.tolist()):
            parsed_expert_id = int(expert_id)
            if parsed_expert_id < 0:
                continue
            layer_counts[parsed_expert_id] = (
                float(layer_counts.get(parsed_expert_id, 0.0)) + float(count)
            )

    @staticmethod
    def _sort_scores(scores: dict[int, float]) -> tuple[int, ...]:
        return tuple(
            expert_id
            for expert_id, _ in sorted(
                scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if expert_id >= 0
        )

    def finalize_step(
        self,
        *,
        hot_budget: int,
        source: str = "observed_runtime_hotness",
    ) -> dict[int, PredictorOnlineExpertState]:
        normalized_hot_budget = max(int(hot_budget), 1)
        next_states: dict[int, PredictorOnlineExpertState] = {}
        next_smoothed_scores: dict[int, dict[int, float]] = {}
        all_layer_indices = set(self.smoothed_scores_by_layer) | set(
            self.current_counts_by_layer
        )
        for layer_index in sorted(all_layer_indices):
            previous_scores = self.smoothed_scores_by_layer.get(layer_index, {})
            current_counts = self.current_counts_by_layer.get(layer_index, {})
            merged_expert_ids = set(previous_scores) | set(current_counts)
            blended_scores = {
                expert_id: (
                    float(current_counts.get(expert_id, 0.0))
                    * (1.0 - self.hotness_decay)
                    + float(previous_scores.get(expert_id, 0.0)) * self.hotness_decay
                )
                for expert_id in merged_expert_ids
            }
            blended_scores = {
                expert_id: score
                for expert_id, score in blended_scores.items()
                if score > 0.0
            }
            if not blended_scores:
                continue
            next_smoothed_scores[layer_index] = blended_scores
            active_expert_ids = self._sort_scores(current_counts)
            hot_expert_ids = self._sort_scores(blended_scores)[:normalized_hot_budget]
            prefetched_expert_ids = tuple(
                expert_id
                for expert_id in hot_expert_ids
                if expert_id not in set(active_expert_ids)
            )
            next_states[layer_index] = PredictorOnlineExpertState(
                source=source,
                active_expert_ids=active_expert_ids,
                prefetched_expert_ids=prefetched_expert_ids,
                hot_expert_ids=hot_expert_ids,
                prefetch_priority_expert_ids=hot_expert_ids,
            )
        self.smoothed_scores_by_layer = next_smoothed_scores
        self.current_counts_by_layer.clear()
        return next_states
