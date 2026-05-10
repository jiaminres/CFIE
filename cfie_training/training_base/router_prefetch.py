"""Router 驱动的 GPTQ cache 预取规划——predictor 预测 + router 命中 → 决定 prefetch 内容。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from cfie_training.training_base.resident_gptq_cache import ResidentGptqCache


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
class RoutedExpert:
    layer_id: int
    expert_id: int
    score: float = 1.0
    token_count: int = 1

    def __post_init__(self) -> None:
        _require_non_negative_int("layer_id", self.layer_id)
        _require_non_negative_int("expert_id", self.expert_id)
        _require_positive_int("token_count", self.token_count)

    @property
    def priority(self) -> float:
        return self.score * self.token_count


@dataclass(frozen=True, slots=True)
class ExpertBundleIds:
    w13_bundle_id: str
    w2_bundle_id: str

    def __post_init__(self) -> None:
        _require_non_empty_string("w13_bundle_id", self.w13_bundle_id)
        _require_non_empty_string("w2_bundle_id", self.w2_bundle_id)

    @property
    def bundle_ids(self) -> tuple[str, str]:
        return (self.w13_bundle_id, self.w2_bundle_id)


@dataclass(frozen=True, slots=True)
class RouterPrefetchPlan:
    prefetch_bundle_ids: tuple[str, ...]
    locked_bundle_ids: tuple[str, ...]
    skipped_experts: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class RouterPrefetchFailure:
    bundle_id: str
    reason: str

    def __post_init__(self) -> None:
        _require_non_empty_string("bundle_id", self.bundle_id)
        _require_non_empty_string("reason", self.reason)


@dataclass(frozen=True, slots=True)
class RouterPrefetchResult:
    plan: RouterPrefetchPlan
    loaded_bundle_ids: tuple[str, ...]
    ready_bundle_ids: tuple[str, ...]
    resident_bundle_ids: tuple[str, ...]
    locked_bundle_ids: tuple[str, ...]
    miss_rate: float
    failed_prefetches: tuple[RouterPrefetchFailure, ...] = ()

    @property
    def failed_bundle_ids(self) -> tuple[str, ...]:
        return tuple(failure.bundle_id for failure in self.failed_prefetches)

    @property
    def has_capacity_pressure(self) -> bool:
        return bool(self.failed_prefetches)


@dataclass(frozen=True, slots=True)
class RouterPrefetchDepthTuningConfig:
    min_prefetch_depth: int = 1
    max_prefetch_depth: int = 16
    increase_step: int = 2
    decrease_step: int = 2
    miss_rate_threshold: float = 0.05
    miss_rate_steps: int = 20
    capacity_pressure_threshold: float = 0.0
    capacity_pressure_steps: int = 1

    def __post_init__(self) -> None:
        _require_positive_int("min_prefetch_depth", self.min_prefetch_depth)
        _require_positive_int("max_prefetch_depth", self.max_prefetch_depth)
        if self.min_prefetch_depth > self.max_prefetch_depth:
            raise ValueError("min_prefetch_depth must be <= max_prefetch_depth")
        _require_positive_int("increase_step", self.increase_step)
        _require_positive_int("decrease_step", self.decrease_step)
        _require_non_negative_float("miss_rate_threshold", self.miss_rate_threshold)
        _require_positive_int("miss_rate_steps", self.miss_rate_steps)
        _require_non_negative_float(
            "capacity_pressure_threshold",
            self.capacity_pressure_threshold,
        )
        _require_positive_int(
            "capacity_pressure_steps",
            self.capacity_pressure_steps,
        )


@dataclass(frozen=True, slots=True)
class RouterPrefetchDepthDecision:
    old_depth: int
    new_depth: int
    reason: str
    miss_rate_streak: int = 0
    capacity_pressure_streak: int = 0

    def __post_init__(self) -> None:
        _require_positive_int("old_depth", self.old_depth)
        _require_positive_int("new_depth", self.new_depth)
        _require_non_empty_string("reason", self.reason)
        _require_non_negative_int("miss_rate_streak", self.miss_rate_streak)
        _require_non_negative_int(
            "capacity_pressure_streak",
            self.capacity_pressure_streak,
        )

    @property
    def changed(self) -> bool:
        return self.old_depth != self.new_depth


@dataclass(slots=True)
class RouterPrefetchDepthTuner:
    config: RouterPrefetchDepthTuningConfig = field(
        default_factory=RouterPrefetchDepthTuningConfig
    )
    _miss_rate_streak: int = 0
    _capacity_pressure_streak: int = 0

    def update(
        self,
        planner: "RouterGptqPrefetchPlanner",
        *,
        miss_rate: float,
        capacity_pressure_rate: float,
    ) -> RouterPrefetchDepthDecision:
        _require_non_negative_float("miss_rate", miss_rate)
        _require_non_negative_float(
            "capacity_pressure_rate",
            capacity_pressure_rate,
        )
        old_depth = planner.prefetch_depth

        if capacity_pressure_rate > self.config.capacity_pressure_threshold:
            self._capacity_pressure_streak += 1
        else:
            self._capacity_pressure_streak = 0

        if self._capacity_pressure_streak >= self.config.capacity_pressure_steps:
            new_depth = max(
                self.config.min_prefetch_depth,
                old_depth - self.config.decrease_step,
            )
            if new_depth < old_depth:
                planner.prefetch_depth = new_depth
                decision = self._decision(old_depth, new_depth, "capacity_pressure")
                self._reset_streaks()
                return decision
            return self._decision(old_depth, old_depth, "capacity_pressure_at_min")

        if self._capacity_pressure_streak == 0:
            if miss_rate > self.config.miss_rate_threshold:
                self._miss_rate_streak += 1
            else:
                self._miss_rate_streak = 0

        if self._miss_rate_streak >= self.config.miss_rate_steps:
            new_depth = min(
                self.config.max_prefetch_depth,
                old_depth + self.config.increase_step,
            )
            if new_depth > old_depth:
                planner.prefetch_depth = new_depth
                decision = self._decision(old_depth, new_depth, "cache_miss")
                self._reset_streaks()
                return decision
            return self._decision(old_depth, old_depth, "cache_miss_at_max")

        return self._decision(old_depth, old_depth, "stable")

    def _decision(
        self,
        old_depth: int,
        new_depth: int,
        reason: str,
    ) -> RouterPrefetchDepthDecision:
        return RouterPrefetchDepthDecision(
            old_depth=old_depth,
            new_depth=new_depth,
            reason=reason,
            miss_rate_streak=self._miss_rate_streak,
            capacity_pressure_streak=self._capacity_pressure_streak,
        )

    def _reset_streaks(self) -> None:
        self._miss_rate_streak = 0
        self._capacity_pressure_streak = 0


@dataclass(slots=True)
# ────── RouterGptqPrefetchPlanner — predictor 驱动的预取规划器 ──────
class RouterGptqPrefetchPlanner:
    layer_expert_to_bundles: Mapping[tuple[int, int], ExpertBundleIds]
    prefetch_depth: int = 16
    lock_current_experts: bool = True
    skip_hot_experts: bool = True
    _hot_experts: set[tuple[int, int]] = field(default_factory=set)

    def __post_init__(self) -> None:
        _require_positive_int("prefetch_depth", self.prefetch_depth)

    @classmethod
    def from_param_to_bundle(
        cls,
        param_to_bundle: Mapping[str, str],
        *,
        layer_prefix: str = "layers",
        prefetch_depth: int = 16,
        lock_current_experts: bool = True,
        skip_hot_experts: bool = True,
    ) -> "RouterGptqPrefetchPlanner":
        mapping: dict[tuple[int, int], dict[str, str]] = {}
        for param_id, bundle_id in param_to_bundle.items():
            parsed = _parse_expert_param_id(param_id, layer_prefix=layer_prefix)
            if parsed is None:
                continue
            layer_id, expert_id, weight_name = parsed
            if weight_name not in {"w13_weight", "w2_weight"}:
                continue
            mapping.setdefault((layer_id, expert_id), {})[weight_name] = bundle_id

        complete_mapping = {
            key: ExpertBundleIds(
                w13_bundle_id=parts["w13_weight"],
                w2_bundle_id=parts["w2_weight"],
            )
            for key, parts in mapping.items()
            if "w13_weight" in parts and "w2_weight" in parts
        }
        return cls(
            layer_expert_to_bundles=complete_mapping,
            prefetch_depth=prefetch_depth,
            lock_current_experts=lock_current_experts,
            skip_hot_experts=skip_hot_experts,
        )

    @property
    def hot_experts(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self._hot_experts))

    def set_hot_experts(self, experts: Iterable[tuple[int, int]]) -> None:
        normalized: set[tuple[int, int]] = set()
        for layer_id, expert_id in experts:
            _require_non_negative_int("layer_id", layer_id)
            _require_non_negative_int("expert_id", expert_id)
            normalized.add((layer_id, expert_id))
        self._hot_experts = normalized

    def plan(
        self,
        current_experts: Iterable[RoutedExpert],     # 当前 step router 选中的专家
        *,
        predicted_experts: Iterable[RoutedExpert] = (),  # predictor 预测的未来专家
    ) -> RouterPrefetchPlan:
        """生成预取计划：哪些 bundle 要 prefetch + 哪些要 lock。"""
        selected_experts: list[tuple[int, int]] = []     # 按优先级去重后的候选专家
        skipped: list[tuple[int, int]] = []              # 被跳过的专家（hot/无 bundle）

        # ── 按优先级合并 current + predicted ──
        current = self._ordered_experts(current_experts)    # 按 priority 降序
        predicted = self._ordered_experts(predicted_experts)
        for expert in (*current, *predicted):               # current 优先
            key = (expert.layer_id, expert.expert_id)
            if key in selected_experts: continue             # 去重
            if self.skip_hot_experts and key in self._hot_experts:  # hot 专家不占 cache
                skipped.append(key); continue
            if key not in self.layer_expert_to_bundles:      # 无映射
                skipped.append(key); continue
            selected_experts.append(key)                     # 加入候选

        # ── 收集 bundle_ids（受 prefetch_depth 上限限制）──
        bundle_ids: list[str] = []
        for key in selected_experts:
            for bundle_id in self.layer_expert_to_bundles[key].bundle_ids:  # w13 + w2
                if bundle_id not in bundle_ids: bundle_ids.append(bundle_id)
                if len(bundle_ids) >= self.prefetch_depth: break  # 达上限
            if len(bundle_ids) >= self.prefetch_depth: break

        # ── 当前激活的专家 lock（lock_current_experts=True 时）──
        locked_bundle_ids: list[str] = []
        if self.lock_current_experts:
            for expert in current:
                key = (expert.layer_id, expert.expert_id)
                if self.skip_hot_experts and key in self._hot_experts: continue
                bundles = self.layer_expert_to_bundles.get(key)
                if bundles is None: continue
                for bundle_id in bundles.bundle_ids:
                    if bundle_id in bundle_ids and bundle_id not in locked_bundle_ids:
                        locked_bundle_ids.append(bundle_id)

        return RouterPrefetchPlan(
            prefetch_bundle_ids=tuple(bundle_ids),           # 需要搬到 GPU cache 的 bundle
            locked_bundle_ids=tuple(locked_bundle_ids),      # 需要 lock 的当前激活专家
            skipped_experts=tuple(dict.fromkeys(skipped)),   # 跳过的
        )

    def execute(
        self,
        cache: ResidentGptqCache,                       # GPU resident cold expert cache
        current_experts: Iterable[RoutedExpert],        # 当前 step router 选中的专家
        *,
        predicted_experts: Iterable[RoutedExpert] = (), # predictor 预测的未来专家
        allow_partial: bool = False,                    # 允许部分失败（容量不足时跳过非 lock 项）
        lock: bool = True,                              # 是否 lock（TrainingLoop 传 False，改由显式 lock）
    ) -> RouterPrefetchResult:
        """执行预取：plan → prefetch → lock(可选) → wait_ready → 返回结果。"""
        # 1. 生成预取计划
        plan = self.plan(current_experts, predicted_experts=predicted_experts)
        loaded: list[str] = []
        failed: list[RouterPrefetchFailure] = []
        locked_bundle_ids = set(plan.locked_bundle_ids)

        # 2. 逐个 prefetch（异步 H2D，wait=False）
        for bundle_id in plan.prefetch_bundle_ids:
            was_resident = bundle_id in cache.resident_bundle_ids   # 是否已在 cache 中
            try:
                cache.prefetch(bundle_id)                            # 异步 H2D（不等待）
            except (RuntimeError, ValueError) as exc:
                if not allow_partial or bundle_id in locked_bundle_ids: raise  # 关键 bundle 不允许失败
                failed.append(RouterPrefetchFailure(bundle_id=bundle_id, reason=str(exc)))
                continue
            if not was_resident: loaded.append(bundle_id)           # 记录新加载的

        # 3. lock（可选）+ wait_ready
        if lock and plan.locked_bundle_ids: cache.lock(plan.locked_bundle_ids)
        ready = cache.wait_ready(plan.locked_bundle_ids)            # 等待 locked bundle 的 H2D 完成

        # 4. 返回结果
        return RouterPrefetchResult(
            plan=plan, loaded_bundle_ids=tuple(loaded), ready_bundle_ids=ready,
            resident_bundle_ids=cache.resident_bundle_ids, locked_bundle_ids=cache.locked_bundle_ids,
            miss_rate=cache.stats.miss_rate, failed_prefetches=tuple(failed),
        )

    @staticmethod
    def _ordered_experts(
        experts: Iterable[RoutedExpert],
    ) -> tuple[RoutedExpert, ...]:
        return tuple(
            sorted(
                experts,
                key=lambda item: (-item.priority, item.layer_id, item.expert_id),
            )
        )


def _parse_expert_param_id(
    param_id: str,
    *,
    layer_prefix: str,
) -> tuple[int, int, str] | None:
    prefix = f"{layer_prefix}."
    if not param_id.startswith(prefix):
        return None
    parts = param_id.split(".")
    if len(parts) != 5:
        return None
    if parts[0] != layer_prefix or parts[2] != "experts":
        return None
    try:
        layer_id = int(parts[1])
        expert_id = int(parts[3])
    except ValueError:
        return None
    return layer_id, expert_id, parts[4]
