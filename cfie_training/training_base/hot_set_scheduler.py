"""Hot Set 调度器——根据路由统计和预算选择每轮训练哪些 dense/MoE 参数（设计文档 Section 8）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from cfie_training.training_base.window_plan import TrainingWindowBudget, TrainableParamSpec


def _require_non_empty_string(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_positive_int(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def _require_non_negative_float(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True, slots=True)
class ExpertRoutingStats:
    layer_id: int
    expert_id: int
    activation_count: int = 0
    total_score: float = 0.0
    avg_grad_norm: float = 0.0
    avg_update_norm: float = 0.0
    rounds_since_trained: int = 0
    last_step_trained: int = -1
    token_count: int = 0

    def __post_init__(self) -> None:
        _require_non_negative_int("layer_id", self.layer_id)
        _require_non_negative_int("expert_id", self.expert_id)
        _require_non_negative_int("activation_count", self.activation_count)
        _require_non_negative_float("total_score", self.total_score)
        _require_non_negative_float("avg_grad_norm", self.avg_grad_norm)
        _require_non_negative_float("avg_update_norm", self.avg_update_norm)
        _require_non_negative_int("rounds_since_trained", self.rounds_since_trained)
        _require_non_negative_int("token_count", self.token_count)

    @property
    def avg_score(self) -> float:
        if self.activation_count == 0:
            return 0.0
        return self.total_score / self.activation_count

    @property
    def key(self) -> tuple[int, int]:
        return (self.layer_id, self.expert_id)


@dataclass(frozen=True, slots=True)
class RoutingWindowStats:
    expert_stats: Mapping[tuple[int, int], ExpertRoutingStats]
    total_activations: int = 0
    max_activated_experts_per_token: int = 8

    def __post_init__(self) -> None:
        _require_non_negative_int("total_activations", self.total_activations)
        _require_positive_int(
            "max_activated_experts_per_token",
            self.max_activated_experts_per_token,
        )
        for key, stats in self.expert_stats.items():
            if key != stats.key:
                raise ValueError(
                    f"expert_stats key {key} does not match stats key {stats.key}"
                )

    def router_entropy(self, layer_id: int) -> float:
        import math

        layer_experts = {
            key: stats
            for key, stats in self.expert_stats.items()
            if key[0] == layer_id
        }
        if not layer_experts:
            return 0.0
        total = sum(stats.activation_count for stats in layer_experts.values())
        if total == 0:
            return 0.0
        probs = [
            stats.activation_count / total for stats in layer_experts.values()
        ]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        max_entropy = math.log(len(layer_experts))
        return entropy / max_entropy if max_entropy > 0 else 0.0


@dataclass(frozen=True, slots=True)
class CoverageConstraint:
    min_low_frequency_ratio: float = 0.2
    max_rounds_untrained: int = 10
    collapse_boost_factor: float = 2.0
    collapse_entropy_threshold: float = 0.3

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_low_frequency_ratio <= 1.0:
            raise ValueError("min_low_frequency_ratio must be in [0, 1]")
        _require_positive_int("max_rounds_untrained", self.max_rounds_untrained)
        if self.collapse_boost_factor < 1.0:
            raise ValueError("collapse_boost_factor must be >= 1.0")
        if not 0.0 <= self.collapse_entropy_threshold <= 1.0:
            raise ValueError("collapse_entropy_threshold must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class HotSetSelection:
    dense_param_ids: tuple[str, ...]
    moe_experts: tuple[tuple[int, int], ...]
    moe_param_ids: tuple[str, ...]
    total_fp32_bytes: int
    total_gpu_shadow_bytes: int
    total_adam_bytes: int
    expected_grad_bytes: int
    activation_checkpoint_segment_hint: int
    budget_utilization_ratio: float
    low_freq_expert_count: int
    collapsed_layer_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_non_negative_int("total_fp32_bytes", self.total_fp32_bytes)
        _require_non_negative_int(
            "total_gpu_shadow_bytes",
            self.total_gpu_shadow_bytes,
        )
        _require_non_negative_int("total_adam_bytes", self.total_adam_bytes)
        _require_non_negative_int("expected_grad_bytes", self.expected_grad_bytes)
        _require_non_negative_int(
            "activation_checkpoint_segment_hint",
            self.activation_checkpoint_segment_hint,
        )
        _require_non_negative_float(
            "budget_utilization_ratio",
            self.budget_utilization_ratio,
        )
        _require_non_negative_int("low_freq_expert_count", self.low_freq_expert_count)

    @property
    def all_param_ids(self) -> tuple[str, ...]:
        return self.dense_param_ids + self.moe_param_ids

    @property
    def expert_count(self) -> int:
        return len(self.moe_experts)


@dataclass(slots=True)
class HotSetScheduler:
    coverage: CoverageConstraint = field(default_factory=CoverageConstraint)
    _frequency_counter: dict[tuple[int, int], int] = field(default_factory=dict)
    _round_id: int = 0

    def select_hot_set(
        self,
        candidates: Iterable[TrainableParamSpec],
        *,
        budget: TrainingWindowBudget,
        route_stats: RoutingWindowStats,
        expert_to_param_ids: Mapping[tuple[int, int], tuple[str, ...]],
        param_to_spec: Mapping[str, TrainableParamSpec] | None = None,
        predictor_candidates: tuple[tuple[int, int], ...] = (),
    ) -> HotSetSelection:
        spec_by_id = param_to_spec or {
            spec.param_id: spec for spec in candidates
        }
        dense_specs: list[TrainableParamSpec] = []
        moe_specs: list[TrainableParamSpec] = []
        spec_by_expert: dict[tuple[int, int], list[TrainableParamSpec]] = {}
        for spec in candidates:
            if spec.kind == "dense":
                dense_specs.append(spec)
                continue
            moe_specs.append(spec)
            expert_key = self._expert_key_from_param_id(spec.param_id)
            if expert_key is not None:
                spec_by_expert.setdefault(expert_key, []).append(spec)

        if not moe_specs and not dense_specs:
            return HotSetSelection(
                dense_param_ids=(),
                moe_experts=(),
                moe_param_ids=(),
                total_fp32_bytes=0,
                total_gpu_shadow_bytes=0,
                total_adam_bytes=0,
                expected_grad_bytes=0,
                activation_checkpoint_segment_hint=0,
                budget_utilization_ratio=0.0,
                low_freq_expert_count=0,
                collapsed_layer_ids=(),
            )

        scored_experts = self._score_experts(spec_by_expert, route_stats)

        # Predictor 加成：预测命中的专家给予最高优先级
        predictor_set = set(predictor_candidates)
        if predictor_set:
            scored_experts = [
                (key, score + 10.0 if key in predictor_set else score)
                for key, score in scored_experts
            ]
            # 按加分后的分数重排
            scored_experts.sort(key=lambda x: x[1], reverse=True)

        selected_experts, low_freq_count = self._select_with_coverage(
            scored_experts,
            spec_by_expert,
            route_stats,
        )

        selected_moe: list[TrainableParamSpec] = []
        for key in selected_experts:
            selected_moe.extend(spec_by_expert.get(key, []))

        selected_dense, selected_moe_filtered = self._apply_budget(
            dense_specs,
            selected_moe,
            budget,
        )

        total_fp32_bytes = 0
        total_gpu_bytes = 0
        total_adam_bytes = 0
        for spec in (*selected_dense, *selected_moe_filtered):
            total_fp32_bytes += spec.fp32_bytes
            total_gpu_bytes += spec.gpu_shadow_bytes
            total_adam_bytes += spec.adam_bytes

        collapsed_layers = self._detect_router_collapse(route_stats)
        segment_hint = self._estimate_segment_hint(selected_moe_filtered, budget)

        moe_expert_keys = tuple(
            dict.fromkeys(
                self._expert_key_from_param_id(spec.param_id)
                for spec in selected_moe_filtered
                if self._expert_key_from_param_id(spec.param_id) is not None
            )
        )

        max_fp32 = budget.max_fp32_hot_bytes or 1
        max_gpu = budget.max_gpu_shadow_bytes or 1
        utilization = max(
            total_fp32_bytes / max_fp32 if budget.max_fp32_hot_bytes > 0 else 0.0,
            total_gpu_bytes / max_gpu if budget.max_gpu_shadow_bytes > 0 else 0.0,
        )

        self._round_id += 1

        return HotSetSelection(
            dense_param_ids=tuple(spec.param_id for spec in selected_dense),
            moe_experts=moe_expert_keys,
            moe_param_ids=tuple(
                spec.param_id for spec in selected_moe_filtered
            ),
            total_fp32_bytes=total_fp32_bytes,
            total_gpu_shadow_bytes=total_gpu_bytes,
            total_adam_bytes=total_adam_bytes,
            expected_grad_bytes=total_adam_bytes,
            activation_checkpoint_segment_hint=segment_hint,
            budget_utilization_ratio=utilization,
            low_freq_expert_count=low_freq_count,
            collapsed_layer_ids=collapsed_layers,
        )

    def record_training_round(
        self,
        trained_experts: Iterable[tuple[int, int]],
    ) -> None:
        for key in self._frequency_counter:
            self._frequency_counter[key] += 1
        for key in trained_experts:
            self._frequency_counter[key] = 0

    def rounds_since_trained(self, key: tuple[int, int]) -> int:
        return self._frequency_counter.get(key, 0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "frequency_counter": {
                f"{layer_id}:{expert_id}": count
                for (layer_id, expert_id), count in self._frequency_counter.items()
            },
            "round_id": self._round_id,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._frequency_counter.clear()
        for key_str, count in state.get("frequency_counter", {}).items():
            layer_id, expert_id = key_str.split(":", 1)
            self._frequency_counter[(int(layer_id), int(expert_id))] = int(count)
        self._round_id = int(state.get("round_id", 0))

    def _score_experts(
        self,
        spec_by_expert: Mapping[tuple[int, int], list[TrainableParamSpec]],
        route_stats: RoutingWindowStats,
    ) -> list[tuple[tuple[int, int], float]]:
        scored: list[tuple[tuple[int, int], float]] = []
        for key in spec_by_expert:
            stats = route_stats.expert_stats.get(key)
            score = 0.0

            if stats is not None:
                score += stats.avg_score * 10.0
                score += stats.token_count * 0.5
                score += stats.avg_grad_norm * 0.1
                score += stats.avg_update_norm * 0.1

            rounds = self._frequency_counter.get(key, route_stats.total_activations)
            if rounds > self.coverage.max_rounds_untrained:
                score += float(rounds) * 0.5

            score += 1.0 / (rounds + 1) * 2.0

            collapsed = self._detect_router_collapse(route_stats)
            if key[0] in collapsed:
                score *= self.coverage.collapse_boost_factor

            scored.append((key, score))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored

    def _select_with_coverage(
        self,
        scored_experts: list[tuple[tuple[int, int], float]],
        spec_by_expert: Mapping[tuple[int, int], list[TrainableParamSpec]],
        route_stats: RoutingWindowStats,
    ) -> tuple[list[tuple[int, int]], int]:
        if not scored_experts:
            return [], 0

        min_low = max(
            1,
            int(len(scored_experts) * self.coverage.min_low_frequency_ratio),
        )

        low_freq_experts: list[tuple[int, int]] = []
        high_freq_experts: list[tuple[int, int]] = []

        for key, _score in scored_experts:
            stats = route_stats.expert_stats.get(key)
            rounds = self._frequency_counter.get(key, 10)
            is_low_freq = (
                stats is None
                or stats.activation_count == 0
                or rounds >= self.coverage.max_rounds_untrained
            )
            if is_low_freq:
                low_freq_experts.append(key)
            else:
                high_freq_experts.append(key)

        low_count = min(min_low, len(low_freq_experts))
        selected = list(low_freq_experts[:low_count])

        for key in high_freq_experts:
            if key not in selected:
                selected.append(key)

        for key in low_freq_experts[low_count:]:
            if key not in selected:
                selected.append(key)

        return selected, low_count

    def _apply_budget(
        self,
        dense_specs: list[TrainableParamSpec],
        moe_specs: list[TrainableParamSpec],
        budget: TrainingWindowBudget,
    ) -> tuple[list[TrainableParamSpec], list[TrainableParamSpec]]:
        selected_dense: list[TrainableParamSpec] = []
        remaining_fp32 = budget.max_fp32_hot_bytes or float("inf")
        remaining_gpu = budget.max_gpu_shadow_bytes or float("inf")
        remaining_adam = budget.max_adam_bytes or float("inf")

        for spec in dense_specs:
            if (
                (budget.max_fp32_hot_bytes == 0
                 or spec.fp32_bytes <= remaining_fp32)
                and (budget.max_gpu_shadow_bytes == 0
                     or spec.gpu_shadow_bytes <= remaining_gpu)
                and (budget.max_adam_bytes == 0
                     or spec.adam_bytes <= remaining_adam)
            ):
                selected_dense.append(spec)
                remaining_fp32 -= spec.fp32_bytes
                remaining_gpu -= spec.gpu_shadow_bytes
                remaining_adam -= spec.adam_bytes

        selected_moe: list[TrainableParamSpec] = []
        for spec in moe_specs:
            if (
                (budget.max_fp32_hot_bytes == 0
                 or spec.fp32_bytes <= remaining_fp32)
                and (budget.max_gpu_shadow_bytes == 0
                     or spec.gpu_shadow_bytes <= remaining_gpu)
                and (budget.max_adam_bytes == 0
                     or spec.adam_bytes <= remaining_adam)
            ):
                selected_moe.append(spec)
                remaining_fp32 -= spec.fp32_bytes
                remaining_gpu -= spec.gpu_shadow_bytes
                remaining_adam -= spec.adam_bytes

        return selected_dense, selected_moe

    @staticmethod
    def _detect_router_collapse(
        route_stats: RoutingWindowStats,
    ) -> tuple[int, ...]:
        layer_ids: set[int] = {key[0] for key in route_stats.expert_stats}
        collapsed: list[int] = []
        for layer_id in sorted(layer_ids):
            entropy = route_stats.router_entropy(layer_id)
            if entropy > 0 and entropy < 0.3:
                collapsed.append(layer_id)
        return tuple(collapsed)

    @staticmethod
    def _estimate_segment_hint(
        moe_specs: list[TrainableParamSpec],
        budget: TrainingWindowBudget,
    ) -> int:
        if not moe_specs:
            return 0
        total_params = sum(spec.gptq_requant_bytes for spec in moe_specs)
        if total_params <= 0:
            return len(moe_specs)
        bucket_size = budget.max_gpu_shadow_bytes or (1 << 30)
        if bucket_size <= 0:
            return 1
        return max(1, len(moe_specs) // max(1, total_params // bucket_size))

    @staticmethod
    def _expert_key_from_param_id(param_id: str) -> tuple[int, int] | None:
        parts = param_id.split(".")
        if len(parts) != 5 or parts[0] != "layers" or parts[2] != "experts":
            return None
        try:
            return (int(parts[1]), int(parts[3]))
        except ValueError:
            return None
