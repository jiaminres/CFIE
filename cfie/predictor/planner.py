"""Predictor-driven candidate-pool planning helpers."""

from __future__ import annotations

import logging
import os
import time as _time
from dataclasses import dataclass
from typing import Any

import torch

from cfie.predictor.bundle import FutureExpertPredictor, PredictorRuntimeSchema

logger = logging.getLogger(__name__)


def _bench_timing_enabled() -> bool:
    return os.getenv("CFIE_BENCH_TIMING", "") == "1"


@dataclass(slots=True, frozen=True)
class CandidateLayerPlan:
    future_layer_index: int
    predicted_executed_expert_ids: tuple[int, ...]
    candidate_expert_ids: tuple[int, ...]
    candidate_scores: tuple[float, ...]

    @property
    def speculative_expert_ids(self) -> tuple[int, ...]:
        return self.candidate_expert_ids[len(self.predicted_executed_expert_ids) :]

    def to_dict(self) -> dict[str, Any]:
        return {
            "future_layer_index": self.future_layer_index,
            "predicted_executed_expert_ids": list(self.predicted_executed_expert_ids),
            "candidate_expert_ids": list(self.candidate_expert_ids),
            "candidate_scores": list(self.candidate_scores),
            "speculative_expert_ids": list(self.speculative_expert_ids),
        }


@dataclass(slots=True, frozen=True)
class PredictorCandidatePlan:
    profile_name: str
    selection_mode: str
    allow_candidate_mismatch: bool
    insertion_layer_index: int
    layer_plans: tuple[CandidateLayerPlan, ...]

    @property
    def speculative_experts_per_layer(self) -> int:
        if not self.layer_plans:
            return 0
        return len(self.layer_plans[0].speculative_expert_ids)

    @property
    def shared_gpu_candidate_slots(self) -> int:
        return sum(len(layer.speculative_expert_ids) for layer in self.layer_plans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "selection_mode": self.selection_mode,
            "allow_candidate_mismatch": self.allow_candidate_mismatch,
            "insertion_layer_index": self.insertion_layer_index,
            "speculative_experts_per_layer": self.speculative_experts_per_layer,
            "shared_gpu_candidate_slots": self.shared_gpu_candidate_slots,
            "layer_plans": [layer.to_dict() for layer in self.layer_plans],
        }


class PredictorCandidatePlanner:
    def __init__(
        self,
        *,
        schema: PredictorRuntimeSchema,
        model: FutureExpertPredictor,
    ) -> None:
        self.schema = schema.validate()
        self.model = model.eval()

    def _model_device(self) -> torch.device:
        first_param = next(self.model.parameters(), None)
        if first_param is None:
            return torch.device("cpu")
        return first_param.device

    def _model_dtype(self) -> torch.dtype:
        first_param = next(self.model.parameters(), None)
        if first_param is None:
            return torch.float32
        return first_param.dtype

    def _normalize_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if hidden_state.ndim == 1:
            hidden_state = hidden_state.unsqueeze(0)
        elif hidden_state.ndim >= 2:
            hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1])
        else:
            raise ValueError("hidden_state must have rank >= 1")
        if int(hidden_state.shape[-1]) != self.schema.input_summary_dim:
            raise ValueError(
                "hidden_state dimension mismatch: "
                f"expected {self.schema.input_summary_dim}, got {hidden_state.shape[-1]}"
            )
        if int(hidden_state.shape[0]) < 1:
            raise ValueError("hidden_state must contain at least one token row")
        return hidden_state.to(
            device=self._model_device(),
            dtype=self._model_dtype(),
        )

    def _resolve_future_layer_indices(
        self,
        *,
        insertion_layer_index: int,
        total_layers: int | None,
        future_layer_indices: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        if future_layer_indices is not None:
            resolved = tuple(int(index) for index in future_layer_indices)
            if len(resolved) != self.schema.window_layers:
                raise ValueError(
                    "future_layer_indices length must equal schema.window_layers"
                )
            return resolved
        if total_layers is None:
            raise ValueError(
                "total_layers is required when future_layer_indices is not provided"
            )
        if int(total_layers) < 1:
            raise ValueError("total_layers must be >= 1")
        return tuple(
            (int(insertion_layer_index) + offset + 1) % int(total_layers)
            for offset in range(self.schema.window_layers)
        )

    def plan_window(
        self,
        hidden_state: torch.Tensor,
        *,
        insertion_layer_index: int,
        total_layers: int | None = None,
        future_layer_indices: tuple[int, ...] | None = None,
    ) -> PredictorCandidatePlan:
        bench_timing = _bench_timing_enabled()
        plan_t0 = _time.perf_counter() if bench_timing else 0.0
        normalize_t0 = _time.perf_counter() if bench_timing else 0.0
        normalized_hidden = self._normalize_hidden_state(hidden_state)
        normalize_seconds = (
            _time.perf_counter() - normalize_t0 if bench_timing else 0.0
        )
        resolved_future_layers = self._resolve_future_layer_indices(
            insertion_layer_index=insertion_layer_index,
            total_layers=total_layers,
            future_layer_indices=future_layer_indices,
        )
        forward_t0 = _time.perf_counter() if bench_timing else 0.0
        with torch.no_grad():
            logits = self.model(normalized_hidden, int(insertion_layer_index))
        forward_seconds = (_time.perf_counter() - forward_t0) if bench_timing else 0.0
        topk_t0 = _time.perf_counter() if bench_timing else 0.0
        logits = logits.mean(dim=0).to(device="cpu", dtype=torch.float32)

        executed_k = min(
            int(self.schema.executed_experts_per_layer),
            int(self.schema.num_experts),
        )
        candidate_k = min(
            int(self.schema.candidate_experts_per_layer),
            int(self.schema.num_experts),
        )
        if candidate_k < executed_k:
            candidate_k = executed_k

        layer_plans: list[CandidateLayerPlan] = []
        for window_offset, future_layer_index in enumerate(resolved_future_layers):
            candidate_scores, candidate_ids = torch.topk(
                logits[window_offset],
                k=candidate_k,
                dim=-1,
            )
            executed_tuple = tuple(
                int(expert_id)
                for expert_id in candidate_ids[:executed_k].tolist()
            )
            candidate_tuple = tuple(int(expert_id) for expert_id in candidate_ids.tolist())
            layer_plans.append(
                CandidateLayerPlan(
                    future_layer_index=int(future_layer_index),
                    predicted_executed_expert_ids=executed_tuple,
                    candidate_expert_ids=candidate_tuple,
                    candidate_scores=tuple(float(score) for score in candidate_scores.tolist()),
                )
            )

        if bench_timing:
            logger.info(
                "CFIE_PREDICTOR_PLAN window_layers=%d stride_layers=%d insertion=%d "
                "executed_k=%d candidate_k=%d model_device=%s model_dtype=%s "
                "normalize=%.3fms forward=%.3fms postprocess=%.3fms total=%.3fms",
                self.schema.window_layers,
                self.schema.stride_layers,
                int(insertion_layer_index),
                executed_k,
                candidate_k,
                self._model_device(),
                self._model_dtype(),
                normalize_seconds * 1000.0,
                forward_seconds * 1000.0,
                (_time.perf_counter() - topk_t0) * 1000.0,
                (_time.perf_counter() - plan_t0) * 1000.0,
            )

        return PredictorCandidatePlan(
            profile_name=self.schema.profile_name,
            selection_mode=self.schema.selection_mode,
            allow_candidate_mismatch=self.schema.allow_candidate_mismatch,
            insertion_layer_index=int(insertion_layer_index),
            layer_plans=tuple(layer_plans),
        )
