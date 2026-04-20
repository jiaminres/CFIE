"""Planning primitives for the first CFIE training runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from cfie_training.config import TrainingProjectConfig
from cfie_training.runtime.source import LocalWeightManifest
from cfie_training.runtime.types import BatchShape, ExpertWindowPlan, LayerBucketPlan


@dataclass(slots=True)
class LayerBucketPlanner:
    config: TrainingProjectConfig

    # 根据 bucket 配置推导基础层跨度。
    def _base_bucket_span(self) -> int:
        # 读取模型天然的 full attention 周期。
        interval = self.config.model_spec.full_attention_interval
        # 读取当前 bucket 划分模式。
        unit = self.config.bucket_schedule.unit
        # 按层切分时，每个 bucket 只放 1 层。
        if unit == "layer":
            return 1
        # hybrid 模式下取半个注意力周期作为跨度。
        if unit == "hybrid":
            return max(1, interval // 2)
        # expert 模式下默认按完整注意力周期切分。
        return interval

    # 按模型层结构生成训练 bucket 划分方案。
    def build(self) -> tuple[LayerBucketPlan, ...]:
        # -----------------
        # 先根据注意力周期构造常规文本层 bucket。
        # 读取模型结构配置。
        model_spec = self.config.model_spec
        # 确保模型结构字段已经通过校验。
        model_spec.validate()
        # 用列表临时保存所有 bucket。
        buckets: list[LayerBucketPlan] = []
        # 计算当前模式下每个 bucket 的基础跨度。
        interval = self._base_bucket_span()
        # 读取模型原生注意力节奏模式。
        pattern = model_spec.attention_pattern
        for start in range(0, model_spec.num_hidden_layers, interval):
            # 生成当前 bucket 覆盖的层索引。
            layer_indices = tuple(
                range(start, min(start + interval, model_spec.num_hidden_layers))
            )
            # 为 bucket 内每一层映射对应的注意力类型。
            attention_types = tuple(
                pattern[layer_index % len(pattern)] for layer_index in layer_indices
            )
            # 追加当前文本层 bucket。
            buckets.append(
                LayerBucketPlan(
                    bucket_id=len(buckets),
                    layer_indices=layer_indices,
                    attention_types=attention_types,
                )
            )

        # -----------------
        # 如配置要求，为 MTP 层追加专用 bucket。
        if self.config.bucket_schedule.include_mtp_dedicated_bucket:
            # 读取 MTP 额外层数。
            mtp_layers = max(0, model_spec.mtp_num_hidden_layers)
            for mtp_index in range(mtp_layers):
                # 为每个 MTP 层追加一个独立 bucket。
                buckets.append(
                    LayerBucketPlan(
                        bucket_id=len(buckets),
                        layer_indices=(model_spec.num_hidden_layers + mtp_index,),
                        attention_types=("mtp",),
                    )
                )
        # 返回不可变的 bucket 规划结果。
        return tuple(buckets)


@dataclass(slots=True)
class ExpertRotationScheduler:
    config: TrainingProjectConfig
    _manifest: LocalWeightManifest = field(init=False, repr=False)
    _router_gate_cache: dict[int, torch.Tensor | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _window_active_cache: dict[int, tuple[int, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    # 初始化权重清单读取器。
    def __post_init__(self) -> None:
        # 为后续 router gate 读取创建本地权重清单对象。
        self._manifest = LocalWeightManifest(self.config)

    # 导出窗口缓存快照，便于 checkpoint 恢复。
    def window_cache_snapshot(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        # 按 rotation_index 排序后导出缓存内容。
        return tuple(
            (rotation_index, active_experts)
            for rotation_index, active_experts in sorted(
                self._window_active_cache.items(),
                key=lambda item: item[0],
            )
        )

    # 从外部快照恢复 active window 缓存。
    def load_window_cache(
        self,
        snapshot: tuple[tuple[int, tuple[int, ...]], ...],
    ) -> None:
        # 直接用 checkpoint 中的缓存内容覆盖当前缓存。
        self._window_active_cache = {
            rotation_index: active_experts
            for rotation_index, active_experts in snapshot
        }

    # 返回每步允许激活的 routed expert 数量。
    def _active_count(self) -> int:
        # 直接读取配置中的 active_experts_per_step。
        return self.config.expert_rotation.active_experts_per_step

    # 按 step / sample / token 维度计算当前轮换窗口索引。
    def _rotation_index(
        self,
        step_index: int,
        *,
        cumulative_samples_processed: int = 0,
        cumulative_tokens_processed: int = 0,
    ) -> int:
        # 如果配置按 token 窗口轮换，则优先按 token 数计算。
        if self.config.expert_rotation.rotate_every_tokens > 0:
            return (
                cumulative_tokens_processed
                // self.config.expert_rotation.rotate_every_tokens
            )
        # 否则如果配置按 sample 窗口轮换，就按 sample 数计算。
        if self.config.expert_rotation.rotate_every_samples > 0:
            return (
                cumulative_samples_processed
                // self.config.expert_rotation.rotate_every_samples
            )
        # 最后退回到默认的按 step 轮换。
        rotate_every_steps = max(1, self.config.expert_rotation.rotate_every_steps)
        return step_index // rotate_every_steps

    # 计算当前 step 对应的 active expert 窗口。
    def active_window(
        self,
        step_index: int,
        *,
        cumulative_samples_processed: int = 0,
        cumulative_tokens_processed: int = 0,
    ) -> tuple[int, ...]:
        # 读取模型总 expert 数。
        num_experts = self.config.model_spec.num_experts
        # 读取当前窗口可激活 expert 数。
        active_count = self._active_count()
        # 先根据轮换索引计算当前窗口起点。
        start = (
            self._rotation_index(
                step_index,
                cumulative_samples_processed=cumulative_samples_processed,
                cumulative_tokens_processed=cumulative_tokens_processed,
            )
            * active_count
        ) % num_experts
        # 再从起点开始顺序选出 active_count 个 expert。
        return tuple((start + offset) % num_experts for offset in range(active_count))

    # 计算基于未来窗口的预取 expert 集合。
    def prefetched_window(
        self,
        step_index: int,
        *,
        cumulative_samples_processed: int = 0,
        cumulative_tokens_processed: int = 0,
        step_samples: int = 0,
        step_tokens: int = 0,
    ) -> tuple[int, ...]:
        # 读取配置中的预取深度。
        prefetch_depth = self.config.bucket_schedule.prefetch_buckets
        # 没有预取深度时直接返回空集合。
        if prefetch_depth < 1:
            return ()
        # 用列表保存预取顺序，用集合去重。
        prefetched: list[int] = []
        seen: set[int] = set()
        for offset in range(1, prefetch_depth + 1):
            # 推导未来第 offset 个窗口之前累计处理的 sample 数。
            future_samples = cumulative_samples_processed + step_samples * offset
            # 推导未来第 offset 个窗口之前累计处理的 token 数。
            future_tokens = cumulative_tokens_processed + step_tokens * offset
            for expert_id in self.active_window(
                step_index + offset,
                cumulative_samples_processed=future_samples,
                cumulative_tokens_processed=future_tokens,
            ):
                # 只保留第一次出现的 expert，避免重复预取。
                if expert_id not in seen:
                    prefetched.append(expert_id)
                    seen.add(expert_id)
        # 返回按未来窗口顺序拼接后的预取 expert 集合。
        return tuple(prefetched)

    # 读取并缓存指定层的 router gate 权重。
    def _router_gate_tensor(self, layer_index: int) -> torch.Tensor | None:
        # 已缓存时直接返回，避免重复读盘。
        if layer_index in self._router_gate_cache:
            return self._router_gate_cache[layer_index]
        # 否则从本地 manifest 里读取该层的 router gate。
        tensor = self._manifest.load_router_gate_tensor(
            layer_index,
            dtype=torch.float32,
        )
        # 无论命中与否都写入缓存，避免重复判断。
        self._router_gate_cache[layer_index] = tensor
        return tensor

    # 从 batch token 中提取供 probe 使用的统计特征。
    def _batch_token_summary(
        self,
        batch: BatchShape,
    ) -> tuple[float, float, float, float]:
        # 显式 token 行存在时，按 mask 只统计真实 token，避免 padding 改写 probe 特征。
        if batch.has_token_rows:
            if batch.attention_mask_rows:
                flattened_tokens = [
                    token_id
                    for row, mask_row in zip(
                        batch.token_rows,
                        batch.attention_mask_rows,
                        strict=True,
                    )
                    for token_id, keep in zip(row, mask_row, strict=True)
                    if keep
                ]
            else:
                flattened_tokens = [
                    token_id for row in batch.token_rows for token_id in row
                ]
            if not flattened_tokens:
                flattened_tokens = [0]
            token_values = torch.tensor(
                flattened_tokens,
                dtype=torch.float32,
                device="cpu",
            )
        else:
            # 否则用 0..total_tokens-1 的顺序编号作为代理特征。
            token_values = torch.arange(
                0,
                batch.total_tokens,
                dtype=torch.float32,
                device="cpu",
            )
        # 返回均值、标准差、首 token 和尾 token 这四个统计量。
        return (
            float(token_values.mean().item()),
            float(token_values.std(unbiased=False).item()),
            float(token_values[0].item()),
            float(token_values[-1].item()),
        )

    # 构造某一层 router 的合成 probe 向量。
    def _router_probe(
        self,
        *,
        step_index: int,
        batch: BatchShape,
        layer_index: int,
        attention_type: str,
    ) -> torch.Tensor:
        # 构造 1..hidden_size 的特征索引向量。
        feature_ids = torch.arange(
            1,
            self.config.model_spec.hidden_size + 1,
            dtype=torch.float32,
            device="cpu",
        )
        # 从 batch token 中抽取统计特征。
        token_mean, token_std, token_first, token_last = self._batch_token_summary(
            batch
        )
        # 用 step 与 layer 编号构造相位偏移。
        phase = 0.07 * float(step_index + 1) + 0.11 * float(layer_index + 1)
        # full attention 与 linear attention 使用不同相位偏移。
        attention_phase = 0.19 if attention_type == "full_attention" else 0.05
        # 第一部分 probe 用 token 标准差调制正弦项。
        probe = torch.sin(
            feature_ids * (0.013 + token_std * 1e-7) + phase + attention_phase
        )
        # 第二部分 probe 叠加与 token 均值相关的余弦项。
        probe = probe + 0.35 * torch.cos(
            feature_ids * 0.007 + token_mean * 1e-4 + attention_phase
        )
        # 第三部分 probe 叠加与首尾 token 相关的正弦项。
        probe = probe + 0.20 * torch.sin(
            feature_ids * 0.017
            + token_first * 7e-5
            + token_last * 5e-5
            + phase * 0.5
        )
        # 最后用 tanh 把 probe 压回稳定范围。
        return torch.tanh(probe / 1.6)

    # 基于 router gate 与 probe 估算当前 batch 的 expert 热度分数。
    def _router_hotness_scores(
        self,
        *,
        step_index: int,
        batch: BatchShape,
        layer_buckets: tuple[LayerBucketPlan, ...],
    ) -> torch.Tensor | None:
        # -----------------
        # 先检查模型定义与本地权重清单是否可用。
        if not self.config.model_spec.is_defined():
            return None
        if not self._manifest.available:
            return None
        # 读取总 expert 数，作为 router 分数向量长度。
        num_experts = self.config.model_spec.num_experts
        # 初始化聚合分数向量。
        aggregated = torch.zeros(num_experts, dtype=torch.float32, device="cpu")
        # 记录累计层权重，后面用于归一化。
        total_weight = 0.0

        # -----------------
        # 对每层 router 计算 probe 分数，并按注意力类型加权聚合。
        for bucket in layer_buckets:
            for layer_index, attention_type in zip(
                bucket.layer_indices,
                bucket.attention_types,
            ):
                # 读取当前层 router gate。
                router_gate = self._router_gate_tensor(layer_index)
                # 缺少任一层 router gate 时直接放弃 hotness 路径。
                if router_gate is None:
                    return None
                # 为当前层构造 probe 向量。
                probe = self._router_probe(
                    step_index=step_index,
                    batch=batch,
                    layer_index=layer_index,
                    attention_type=attention_type,
                )
                # 兼容 gate 矩阵两种可能的维度朝向。
                if (
                    router_gate.shape[0] == num_experts
                    and router_gate.shape[1] == probe.numel()
                ):
                    raw_scores = router_gate @ probe
                elif (
                    router_gate.shape[1] == num_experts
                    and router_gate.shape[0] == probe.numel()
                ):
                    raw_scores = probe @ router_gate
                else:
                    # 形状对不上时无法继续计算热度。
                    return None
                # 对原始分数做标准化，避免个别层绝对值过大。
                normalized = (raw_scores - raw_scores.mean()) / raw_scores.std(
                    unbiased=False
                ).clamp_min(1e-6)
                # full attention 层略微提高权重。
                layer_weight = 1.15 if attention_type == "full_attention" else 1.0
                # 把该层 softmax 分布累加到全局热度向量。
                aggregated = aggregated + torch.softmax(normalized, dim=0) * layer_weight
                # 同时累计层权重。
                total_weight += layer_weight
        # 若没有任何有效层贡献，则返回 None。
        if total_weight <= 0:
            return None
        # 把累计热度除以总权重，得到平均热度。
        aggregated = aggregated / total_weight

        # -----------------
        # 用 round-robin 窗口加一个极小 tie-break，保证分数相同时稳定。
        round_robin = self.active_window(step_index)
        if round_robin:
            # 构造从大到小的极小扰动，用于稳定打破并列。
            tie_break = torch.linspace(
                1e-6,
                1e-7,
                steps=len(round_robin),
                dtype=torch.float32,
                device="cpu",
            )
            aggregated[list(round_robin)] += tie_break
        return aggregated

    # 将分数张量按从高到低排序为 expert id 序列。
    def _ranked_experts(self, scores: torch.Tensor) -> tuple[int, ...]:
        # 按分数从高到低返回 expert 下标。
        ranked = torch.argsort(scores, descending=True)
        return tuple(int(expert_id) for expert_id in ranked.tolist())

    # 计算本轮允许预取的 expert 总预算。
    def _prefetch_budget(self) -> int:
        # 预算等于预取 bucket 数乘每步 active expert 数。
        return max(0, self.config.bucket_schedule.prefetch_buckets) * self._active_count()

    # 将当前步分数与上一轮分数按衰减系数做平滑。
    def _blend_scores(
        self,
        *,
        base_scores: torch.Tensor,
        prior_scores: torch.Tensor | None,
        decay: float,
    ) -> torch.Tensor:
        # 没有上一步分数时，直接使用当前分数。
        if prior_scores is None:
            return base_scores
        # 否则按衰减系数把当前和上一轮分数做平滑混合。
        return base_scores * (1.0 - decay) + prior_scores * decay

    # 从排序后的 expert 序列里截取预算允许的预取集合。
    def _prefetch_from_ranked(
        self,
        *,
        ranked_experts: tuple[int, ...],
        budget: int,
        exclude_expert_ids: tuple[int, ...] = (),
    ) -> tuple[int, ...]:
        # 预算为零时直接返回空集合。
        if budget < 1:
            return ()
        # 先把排除集合转成 set，便于高频判断。
        exclude_set = set(exclude_expert_ids)
        # 用列表保留最终预取顺序。
        prefetched: list[int] = []
        for expert_id in ranked_experts:
            # active experts 等排除集合里的条目不再重复预取。
            if expert_id in exclude_set:
                continue
            # 收下当前 expert。
            prefetched.append(expert_id)
            # 达到预算后立刻停止。
            if len(prefetched) >= budget:
                break
        return tuple(prefetched)

    # 在 router 热度不可用时回退到 round-robin 窗口规划。
    def _fallback_window_plan(
        self,
        *,
        step_index: int,
        reason: str,
        batch: BatchShape | None = None,
        cumulative_samples_processed: int = 0,
        cumulative_tokens_processed: int = 0,
    ) -> ExpertWindowPlan:
        # -----------------
        # 先按基础轮换规则生成 active / prefetched expert 集合。
        # 无 batch 时本步 sample 数按 0 处理。
        step_samples = 0 if batch is None else batch.samples
        # 无 batch 时本步 token 数也按 0 处理。
        step_tokens = 0 if batch is None else batch.total_tokens
        # 先生成当前步 active expert 窗口。
        active_experts = self.active_window(
            step_index,
            cumulative_samples_processed=cumulative_samples_processed,
            cumulative_tokens_processed=cumulative_tokens_processed,
        )
        # 再生成基于未来窗口的 prefetched experts。
        prefetched_experts = self.prefetched_window(
            step_index,
            cumulative_samples_processed=cumulative_samples_processed,
            cumulative_tokens_processed=cumulative_tokens_processed,
            step_samples=step_samples,
            step_tokens=step_tokens,
        )

        # -----------------
        # 返回标准化的 fallback expert window 规划对象。
        return ExpertWindowPlan(
            # 标记当前使用的是 round_robin 策略。
            selection_strategy="round_robin",
            # 记录 fallback 原因，便于后续调试。
            router_score_source=reason,
            # 当前窗口激活的 experts。
            active_expert_ids=active_experts,
            # 当前窗口预取的 experts。
            prefetched_expert_ids=prefetched_experts,
            # hot 集合简单由 active + 部分 prefetched 拼接得到。
            hot_expert_ids=tuple(
                list(active_experts) + list(prefetched_experts[: self._prefetch_budget()])
            ),
            # fallback 路径下预取优先级就是预取结果本身。
            prefetch_priority_expert_ids=prefetched_experts,
        )

    # 为指定 step 规划 active / prefetch / hot expert 窗口。
    def plan_window(
        self,
        *,
        step_index: int,
        batch: BatchShape | None = None,
        layer_buckets: tuple[LayerBucketPlan, ...] | None = None,
        next_batch: BatchShape | None = None,
        cumulative_samples_processed: int = 0,
        cumulative_tokens_processed: int = 0,
    ) -> ExpertWindowPlan:
        # -----------------
        # 如果未启用 router_hotness，则直接走 round-robin 回退路径。
        if self.config.expert_rotation.selection_strategy != "router_hotness":
            return self._fallback_window_plan(
                step_index=step_index,
                reason="round_robin",
                batch=batch,
                cumulative_samples_processed=cumulative_samples_processed,
                cumulative_tokens_processed=cumulative_tokens_processed,
            )
        # 当前 batch 或 layer_buckets 缺失时，也只能回退。
        if batch is None or not layer_buckets:
            return self._fallback_window_plan(
                step_index=step_index,
                reason="fallback_missing_batch_or_layers",
                batch=batch,
                cumulative_samples_processed=cumulative_samples_processed,
                cumulative_tokens_processed=cumulative_tokens_processed,
            )

        # -----------------
        # 计算当前步 router 热度分数；如失败则回退。
        current_scores = self._router_hotness_scores(
            step_index=step_index,
            batch=batch,
            layer_buckets=layer_buckets,
        )
        if current_scores is None:
            return self._fallback_window_plan(
                step_index=step_index,
                reason="fallback_missing_router_manifest",
                batch=batch,
                cumulative_samples_processed=cumulative_samples_processed,
                cumulative_tokens_processed=cumulative_tokens_processed,
            )
        # 默认认为上一轮没有可用分数。
        previous_scores = None
        # 只有 step_index > 0 才尝试取上一轮分数。
        if step_index > 0:
            previous_scores = self._router_hotness_scores(
                step_index=step_index - 1,
                batch=batch,
                layer_buckets=layer_buckets,
            )

        # -----------------
        # 平滑当前分数，并按轮换窗口缓存决定 active experts。
        smoothed_current = self._blend_scores(
            base_scores=current_scores,
            prior_scores=previous_scores,
            decay=self.config.expert_rotation.cross_step_hotness_decay,
        )
        # 把平滑后的当前分数按从高到低排序。
        ranked_current = self._ranked_experts(smoothed_current)
        # 计算当前 step 对应的轮换窗口索引。
        rotation_index = self._rotation_index(
            step_index,
            cumulative_samples_processed=cumulative_samples_processed,
            cumulative_tokens_processed=cumulative_tokens_processed,
        )
        # 先尝试从窗口缓存里取 active experts。
        active_experts = self._window_active_cache.get(rotation_index)
        if active_experts is None:
            # 没缓存时，直接取当前热度最高的前 N 个 experts。
            active_experts = tuple(ranked_current[: self._active_count()])
            if (
                self.config.expert_rotation.rotate_every_steps > 1
                or self.config.expert_rotation.rotate_every_samples > 0
                or self.config.expert_rotation.rotate_every_tokens > 0
            ):
                # 只有存在跨步窗口时，才把 active experts 写回缓存。
                self._window_active_cache[rotation_index] = active_experts
        # 计算本轮预取预算。
        prefetch_budget = self._prefetch_budget()
        # 默认直接用当前平滑分数做预取排序。
        prefetch_scores = smoothed_current
        # 默认记录预取分数来源。
        prefetch_source = (
            "cross_step_router_hotness"
            if previous_scores is not None
            else "current_step_router_hotness"
        )

        # -----------------
        # 如已知下一步 batch，则把下一步热度混入预取排序。
        if next_batch is not None and prefetch_budget > 0:
            # 额外计算下一步 batch 的 router 热度。
            next_scores = self._router_hotness_scores(
                step_index=step_index + 1,
                batch=next_batch,
                layer_buckets=layer_buckets,
            )
            if next_scores is not None:
                # 读取“下一步热度”在预取排序中的权重。
                next_weight = self.config.expert_rotation.next_step_score_weight
                # 把下一步与当前步热度做加权融合。
                prefetch_scores = (
                    next_scores * next_weight
                    + smoothed_current * (1.0 - next_weight)
                )
                # 记录当前预取来源已切换到 blended next-step 模式。
                prefetch_source = "blended_next_step_router_hotness"
        # 对预取分数做从高到低排序。
        prefetch_priority = self._ranked_experts(prefetch_scores)
        # 先根据排序结果、预算和 active 集合筛出预取 experts。
        prefetched_experts = self._prefetch_from_ranked(
            ranked_experts=prefetch_priority,
            budget=prefetch_budget,
            exclude_expert_ids=active_experts,
        )

        # -----------------
        # 根据配置把部分 active experts 前缀并入预取集，提升窗口连续性。
        overlap_budget = min(
            self.config.expert_rotation.prefetch_active_overlap,
            prefetch_budget,
        )
        if overlap_budget > 0:
            # 取 active 前缀，强制让部分 active expert 出现在预取列表开头。
            overlap_prefix = tuple(active_experts[:overlap_budget])
            prefetched_experts = tuple(
                dict.fromkeys(overlap_prefix + prefetched_experts)
            )[:prefetch_budget]

        # -----------------
        # 若预取预算仍未填满，则用 round-robin 预取窗口补足。
        if len(prefetched_experts) < prefetch_budget:
            # 先按 round-robin 规则生成 fallback 预取序列。
            fallback_prefetch = self._prefetch_from_ranked(
                ranked_experts=self.prefetched_window(
                    step_index,
                    cumulative_samples_processed=cumulative_samples_processed,
                    cumulative_tokens_processed=cumulative_tokens_processed,
                    step_samples=batch.samples,
                    step_tokens=batch.total_tokens,
                ),
                budget=prefetch_budget - len(prefetched_experts),
                exclude_expert_ids=tuple(
                        dict.fromkeys(active_experts + prefetched_experts)
                    ),
            )
            # 把 fallback 预取并入当前预取结果，并再次按预算截断。
            prefetched_experts = tuple(
                dict.fromkeys(prefetched_experts + fallback_prefetch)
            )[:prefetch_budget]
        # hot_budget 决定需要保留多少个 top experts 用于热度可视化 / 调试。
        hot_budget = min(
            len(ranked_current),
            len(active_experts) + prefetch_budget,
        )

        # -----------------
        # 返回最终的 expert window 规划结果。
        return ExpertWindowPlan(
            # 标记当前策略为 router_hotness。
            selection_strategy="router_hotness",
            # 记录最终预取分数来源。
            router_score_source=prefetch_source,
            # 返回本步真正激活的 expert 集合。
            active_expert_ids=active_experts,
            # 返回本步预取的 expert 集合。
            prefetched_expert_ids=prefetched_experts,
            # 返回当前热度最高的一段 expert 排名。
            hot_expert_ids=tuple(ranked_current[:hot_budget]),
            # 返回用于预取排序的优先级前缀。
            prefetch_priority_expert_ids=tuple(prefetch_priority[:hot_budget]),
        )
