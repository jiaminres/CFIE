"""Predictor-driven candidate-pool planning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cfie.predictor.bundle import FutureExpertPredictor, PredictorRuntimeSchema


@dataclass(slots=True, frozen=True)
class CandidateLayerPlan:
    future_layer_index: int
    predicted_executed_expert_ids: tuple[int, ...]
    candidate_expert_ids: tuple[int, ...]
    candidate_scores: tuple[float, ...]

    @property
    # 返回候选池里留给 speculative/shared slot 的专家子集。
    def speculative_expert_ids(self) -> tuple[int, ...]:
        # 预测执行预算之后的剩余候选即为 speculative 部分。
        return self.candidate_expert_ids[len(self.predicted_executed_expert_ids) :]

    # 将单层候选池规划导出为字典。
    def to_dict(self) -> dict[str, Any]:
        # 直接输出未来层、执行预算专家和完整候选池。
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
    # 返回当前窗口每层的 speculative expert 数。
    def speculative_experts_per_layer(self) -> int:
        # 没有层规划时退回 0。
        if not self.layer_plans:
            return 0
        # 默认所有层都共享同一档候选预算。
        return len(self.layer_plans[0].speculative_expert_ids)

    @property
    # 返回整个窗口总共需要的 shared GPU candidate slots。
    def shared_gpu_candidate_slots(self) -> int:
        # 把各层 speculative 部分相加即可得到总 slot 数。
        return sum(len(layer.speculative_expert_ids) for layer in self.layer_plans)

    # 将整个窗口候选池规划导出为字典。
    def to_dict(self) -> dict[str, Any]:
        # 直接输出窗口级元信息和各层候选池结果。
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
        # 先缓存并校验 runtime schema。
        self.schema = schema.validate()
        # 再缓存已经加载好权重的 predictor 模型。
        self.model = model.eval()

    def _model_device(self) -> torch.device:
        # 尝试从第一个参数上推断模型所在设备。
        first_param = next(self.model.parameters(), None)
        if first_param is None:
            return torch.device("cpu")
        # 返回模型参数当前设备。
        return first_param.device

    def _normalize_hidden_summary(self, hidden_summary: torch.Tensor) -> torch.Tensor:
        # hidden summary 规划当前只支持单条样本向量。
        if hidden_summary.ndim != 1:
            raise ValueError("hidden_summary must be a rank-1 tensor")
        # 输入维度必须与 runtime schema 对齐。
        if hidden_summary.numel() != self.schema.input_summary_dim:
            raise ValueError(
                "hidden_summary dimension mismatch: "
                f"expected {self.schema.input_summary_dim}, got {hidden_summary.numel()}"
            )
        # 统一转到模型设备和 float32，避免 dtype/device 分歧。
        return hidden_summary.to(device=self._model_device(), dtype=torch.float32)

    def _resolve_future_layer_indices(
        self,
        *,
        insertion_layer_index: int,
        total_layers: int | None,
        future_layer_indices: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        # 显式给 future layers 时，优先按显式值使用。
        if future_layer_indices is not None:
            resolved = tuple(int(index) for index in future_layer_indices)
            if len(resolved) != self.schema.window_layers:
                raise ValueError(
                    "future_layer_indices length must equal schema.window_layers"
                )
            return resolved

        # -----------------
        # 未显式给 future layers 时，要求给 total_layers 用于窗口内推导。
        if total_layers is None:
            raise ValueError(
                "total_layers is required when future_layer_indices is not provided"
            )
        if int(total_layers) < 1:
            raise ValueError("total_layers must be >= 1")
        # 按训练侧 trace 约定，从插入层后一层开始滚动未来窗口。
        return tuple(
            (int(insertion_layer_index) + offset + 1) % int(total_layers)
            for offset in range(self.schema.window_layers)
        )

    def plan_window(
        self,
        hidden_summary: torch.Tensor,
        *,
        insertion_layer_index: int,
        total_layers: int | None = None,
        future_layer_indices: tuple[int, ...] | None = None,
    ) -> PredictorCandidatePlan:
        # 先规范化 hidden summary，并推导 future layer 列表。
        normalized_hidden = self._normalize_hidden_summary(hidden_summary)
        resolved_future_layers = self._resolve_future_layer_indices(
            insertion_layer_index=insertion_layer_index,
            total_layers=total_layers,
            future_layer_indices=future_layer_indices,
        )

        # -----------------
        # 用 predictor 模型计算未来窗口每层的 expert logits。
        with torch.no_grad():
            logits = self.model(normalized_hidden.unsqueeze(0)).squeeze(0)
        logits = logits.to(device="cpu", dtype=torch.float32)

        # -----------------
        # 逐层取 top-k 候选池，并切出预测执行预算与 speculative 部分。
        layer_plans: list[CandidateLayerPlan] = []
        for window_offset, future_layer_index in enumerate(resolved_future_layers):
            scores, candidate_ids = torch.topk(
                logits[window_offset],
                k=self.schema.candidate_experts_per_layer,
                dim=-1,
            )
            candidate_tuple = tuple(int(expert_id) for expert_id in candidate_ids.tolist())
            executed_tuple = candidate_tuple[: self.schema.executed_experts_per_layer]
            layer_plans.append(
                CandidateLayerPlan(
                    future_layer_index=int(future_layer_index),
                    predicted_executed_expert_ids=executed_tuple,
                    candidate_expert_ids=candidate_tuple,
                    candidate_scores=tuple(float(score) for score in scores.tolist()),
                )
            )

        # 返回窗口级候选池规划结果。
        return PredictorCandidatePlan(
            profile_name=self.schema.profile_name,
            selection_mode=self.schema.selection_mode,
            allow_candidate_mismatch=self.schema.allow_candidate_mismatch,
            insertion_layer_index=int(insertion_layer_index),
            layer_plans=tuple(layer_plans),
        )
