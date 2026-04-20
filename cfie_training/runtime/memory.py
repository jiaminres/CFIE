"""Memory-first planning for the CFIE training runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from cfie_training.config import TrainingProjectConfig
from cfie_training.runtime.planner import LayerBucketPlanner
from cfie_training.runtime.quantization import runtime_device_weight_bytes_per_param
from cfie_training.runtime.types import BatchShape

GiB = 1024**3


# 将字节数转换为 GiB 浮点值。
def _bytes_to_gib(value: int) -> float:
    # 直接用 1 GiB 的字节数做除法换算。
    return value / GiB


# 计算向上整除结果。
def _ceil_div(a: int, b: int) -> int:
    # 用经典整除上取整公式计算结果。
    return (a + b - 1) // b


# 将存储 dtype 名称映射为每元素字节数。
def _storage_dtype_bytes(dtype_name: str) -> int:
    # FP32 每个元素占 4 字节。
    if dtype_name in {"fp32"}:
        return 4
    # FP16 / BF16 每个元素占 2 字节。
    if dtype_name in {"fp16", "bf16"}:
        return 2
    # FP8 每个元素占 1 字节。
    if dtype_name in {"fp8_e4m3fn", "fp8_e5m2"}:
        return 1
    # 其他 dtype 当前不支持用于 CPU 状态存储估算。
    raise ValueError(f"unsupported CPU storage dtype: {dtype_name}")


@dataclass(slots=True, frozen=True)
class MemoryTierSummary:
    resident_bytes: int
    budget_bytes: int
    safety_margin_bytes: int

    # 返回扣除安全边界后的可用容量。
    @property
    def available_bytes(self) -> int:
        # 可用容量等于预算减去安全边界。
        return self.budget_bytes - self.safety_margin_bytes

    # 判断当前 tier 是否仍在预算内。
    @property
    def within_budget(self) -> bool:
        # 只要当前驻留字节数不超过可用容量就算预算内。
        return self.resident_bytes <= self.available_bytes

    # 将单个内存 tier 汇总序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 把字节值、GiB 值和预算判断统一导出。
        return {
            "resident_bytes": self.resident_bytes,
            "resident_gib": round(_bytes_to_gib(self.resident_bytes), 4),
            "budget_bytes": self.budget_bytes,
            "budget_gib": round(_bytes_to_gib(self.budget_bytes), 4),
            "safety_margin_bytes": self.safety_margin_bytes,
            "safety_margin_gib": round(_bytes_to_gib(self.safety_margin_bytes), 4),
            "available_bytes": self.available_bytes,
            "available_gib": round(_bytes_to_gib(self.available_bytes), 4),
            "within_budget": self.within_budget,
        }


@dataclass(slots=True, frozen=True)
class TrainingMemoryPlan:
    bucket_count: int
    mtp_bucket_count: int
    max_layers_per_bucket: int
    total_params: int
    routed_expert_params_total: int
    non_routed_params_total: int
    static_params_total: int
    bucket_non_routed_params_total: int
    params_per_routed_expert: int
    params_per_bucket_non_routed: int
    params_per_bucket_active_routed: int
    params_per_bucket_prefetched_routed: int
    bucket_non_routed_params_by_bucket: tuple[int, ...]
    bucket_active_routed_params_by_bucket: tuple[int, ...]
    activation_resident_bytes: int
    cpu_optimizer_state_storage_dtype: str
    cpu_optimizer_state_bytes_per_param: int
    host_gradient_buffer_storage_dtype: str
    host_gradient_buffer_bytes_per_param: int
    host_gradient_buffer_scope: str
    host_gradient_buffer_bytes: int
    full_model_gradient_buffer_bytes: int
    transport_staged_file_cache_bytes: int
    weight_stage_buffer_bytes: int
    transfer_staging_buffer_bytes: int
    transfer_overlap_enabled: bool
    gpu_hot: MemoryTierSummary
    cpu_hot: MemoryTierSummary
    nvme_cold: MemoryTierSummary

    # 判断 GPU / CPU / NVMe 三层预算是否同时满足。
    @property
    def all_tiers_within_budget(self) -> bool:
        # 只有三层都 within_budget 时才返回 True。
        return (
            self.gpu_hot.within_budget
            and self.cpu_hot.within_budget
            and self.nvme_cold.within_budget
        )

    # 将训练内存规划完整导出为字典。
    def to_dict(self) -> dict[str, Any]:
        # 把训练期所有核心参数规模和三层预算一起导出。
        return {
            "bucket_count": self.bucket_count,
            "mtp_bucket_count": self.mtp_bucket_count,
            "max_layers_per_bucket": self.max_layers_per_bucket,
            "total_params": self.total_params,
            "total_params_billion": round(self.total_params / 1_000_000_000, 4),
            "routed_expert_params_total": self.routed_expert_params_total,
            "non_routed_params_total": self.non_routed_params_total,
            "static_params_total": self.static_params_total,
            "bucket_non_routed_params_total": self.bucket_non_routed_params_total,
            "params_per_routed_expert": self.params_per_routed_expert,
            "params_per_bucket_non_routed": self.params_per_bucket_non_routed,
            "params_per_bucket_active_routed": self.params_per_bucket_active_routed,
            "params_per_bucket_prefetched_routed": self.params_per_bucket_prefetched_routed,
            "bucket_non_routed_params_by_bucket": list(
                self.bucket_non_routed_params_by_bucket
            ),
            "bucket_active_routed_params_by_bucket": list(
                self.bucket_active_routed_params_by_bucket
            ),
            "activation_resident_bytes": self.activation_resident_bytes,
            "activation_resident_gib": round(
                _bytes_to_gib(self.activation_resident_bytes),
                4,
            ),
            "cpu_optimizer_state_storage_dtype": self.cpu_optimizer_state_storage_dtype,
            "cpu_optimizer_state_bytes_per_param": self.cpu_optimizer_state_bytes_per_param,
            "host_gradient_buffer_storage_dtype": self.host_gradient_buffer_storage_dtype,
            "host_gradient_buffer_bytes_per_param": self.host_gradient_buffer_bytes_per_param,
            "host_gradient_buffer_scope": self.host_gradient_buffer_scope,
            "host_gradient_buffer_bytes": self.host_gradient_buffer_bytes,
            "host_gradient_buffer_gib": round(
                _bytes_to_gib(self.host_gradient_buffer_bytes),
                4,
            ),
            "full_model_gradient_buffer_bytes": self.full_model_gradient_buffer_bytes,
            "full_model_gradient_buffer_gib": round(
                _bytes_to_gib(self.full_model_gradient_buffer_bytes),
                4,
            ),
            "transport_staged_file_cache_bytes": self.transport_staged_file_cache_bytes,
            "transport_staged_file_cache_gib": round(
                _bytes_to_gib(self.transport_staged_file_cache_bytes),
                4,
            ),
            "weight_stage_buffer_bytes": self.weight_stage_buffer_bytes,
            "weight_stage_buffer_gib": round(
                _bytes_to_gib(self.weight_stage_buffer_bytes),
                4,
            ),
            "transfer_staging_buffer_bytes": self.transfer_staging_buffer_bytes,
            "transfer_staging_buffer_gib": round(
                _bytes_to_gib(self.transfer_staging_buffer_bytes),
                4,
            ),
            "transfer_overlap_enabled": self.transfer_overlap_enabled,
            "gpu_hot": self.gpu_hot.to_dict(),
            "cpu_hot": self.cpu_hot.to_dict(),
            "nvme_cold": self.nvme_cold.to_dict(),
            "all_tiers_within_budget": self.all_tiers_within_budget,
        }


@dataclass(slots=True, frozen=True)
class StartupParameterEstimate:
    """
    记录训练启动阶段的一组参数驻留估算结果。

    这个对象不是运行时真实显存快照，而是 memory planner 在正式启动前
    根据模型结构、batch 形状、专家活跃度与分层存储策略推导出的候选规划。
    调用方可以用它判断当前启动参数是否能放进 GPU hot tier 预算，
    也可以把估算结果导出到日志或诊断报告里辅助调参。
    """

    # 用户或配置侧给 GPU hot tier 预留的显存预算，单位是 GB。
    gpu_hot_budget_gb: float

    # 按预算折算后的 GPU hot tier 可用字节数，是后续 fits 判定的直接上限。
    gpu_hot_available_bytes: int

    # 单个训练 step 中预计会同时被访问或保持活跃的专家数量。
    active_experts_per_step: int

    # GPU hot tier 中允许同时驻留的最大 bucket 数，用于限制专家热缓存规模。
    max_live_buckets: int

    # 允许提前预取到热层级的 bucket 数，用于估算 overlap/prefetch 额外占用。
    prefetch_buckets: int

    # 当前估算绑定的 batch 形状，包含 micro-batch、序列长度等训练输入规模信息。
    batch: BatchShape

    # 当前候选规划是否能放进 GPU hot tier 预算内。
    fits_within_budget: bool

    # 规划后需要常驻或临时驻留在 GPU hot tier 的参数字节数。
    planned_gpu_hot_bytes: int

    # 规划后放入 CPU hot tier 的参数字节数，通常用于承接 GPU 放不下但仍需快速访问的部分。
    planned_cpu_hot_bytes: int

    # 规划后落到 NVMe cold tier 的参数字节数，通常代表最低频或冷启动可延迟加载的部分。
    planned_nvme_cold_bytes: int

    # 返回规划后 GPU hot 占用的 GiB 值。
    @property
    def planned_gpu_hot_gib(self) -> float:
        # 直接把 GPU hot 字节数转成 GiB。
        return _bytes_to_gib(self.planned_gpu_hot_bytes)

    # 返回 GPU hot tier 的实际可用 GiB。
    @property
    def gpu_hot_available_gib(self) -> float:
        # 直接把 GPU hot 可用字节数转成 GiB。
        return _bytes_to_gib(self.gpu_hot_available_bytes)

    # 计算 GPU hot tier 的填充比例。
    @property
    def gpu_fill_ratio(self) -> float:
        # 可用容量非正时，填充比例按 0 处理。
        if self.gpu_hot_available_bytes <= 0:
            return 0.0
        # 否则返回计划驻留字节数除以可用字节数。
        return self.planned_gpu_hot_bytes / self.gpu_hot_available_bytes

    # 将启动估算候选结果导出为字典。
    def to_dict(self) -> dict[str, Any]:
        # 把预算、候选参数和估算结果统一导出。
        return {
            "gpu_hot_budget_gb": self.gpu_hot_budget_gb,
            "gpu_hot_available_bytes": self.gpu_hot_available_bytes,
            "gpu_hot_available_gib": round(self.gpu_hot_available_gib, 4),
            "active_experts_per_step": self.active_experts_per_step,
            "max_live_buckets": self.max_live_buckets,
            "prefetch_buckets": self.prefetch_buckets,
            "batch": self.batch.to_dict(),
            "fits_within_budget": self.fits_within_budget,
            "planned_gpu_hot_bytes": self.planned_gpu_hot_bytes,
            "planned_gpu_hot_gib": round(self.planned_gpu_hot_gib, 4),
            "planned_cpu_hot_bytes": self.planned_cpu_hot_bytes,
            "planned_cpu_hot_gib": round(_bytes_to_gib(self.planned_cpu_hot_bytes), 4),
            "planned_nvme_cold_bytes": self.planned_nvme_cold_bytes,
            "planned_nvme_cold_gib": round(
                _bytes_to_gib(self.planned_nvme_cold_bytes),
                4,
            ),
            "gpu_fill_ratio": round(self.gpu_fill_ratio, 4),
        }


@dataclass(slots=True)
class TrainingMemoryPlanner:
    config: TrainingProjectConfig

    # 按权重将总量比例分配到每个 bucket。
    def _proportional_bucket_counts(
        self,
        *,
        total: int,
        weights: tuple[int, ...],
    ) -> tuple[int, ...]:
        # 没有 bucket 权重时直接返回空元组。
        if not weights:
            return ()
        # 总量非正时，每个 bucket 都分到 0。
        if total <= 0:
            return tuple(0 for _ in weights)
        # 先把所有负权重裁成 0 后求总权重。
        total_weight = sum(max(weight, 0) for weight in weights)
        # 总权重非正时无法分配，同样全部返回 0。
        if total_weight <= 0:
            return tuple(0 for _ in weights)
        # 先按比例算出浮点份额。
        raw_shares = [total * (weight / total_weight) for weight in weights]
        # 对每个份额先向下取整。
        counts = [int(math.floor(share)) for share in raw_shares]
        # 统计已经分配出去的总量。
        assigned = sum(counts)
        # 记录每个 bucket 的小数余量，供后续分配剩余份额。
        remainders = sorted(
            (
                (raw_shares[index] - counts[index], index)
                for index in range(len(weights))
            ),
            key=lambda item: (-item[0], item[1]),
        )
        # 把还没分完的份额按余量从大到小补给对应 bucket。
        for _, index in remainders[: total - assigned]:
            counts[index] += 1
        # 返回最终每个 bucket 的分配结果。
        return tuple(counts)

    # 基于训练配置与 batch 形状构造三层内存规划。
    def build(self, batch: BatchShape) -> TrainingMemoryPlan:
        # -----------------
        # 先校验配置，并抽取模型 / 预算 / 状态参数。
        # 读取当前训练配置对象。
        cfg = self.config
        # 先执行一次配置级校验。
        cfg.validate()
        # 取出模型结构配置。
        model = cfg.model_spec
        # 取出内存预算配置。
        budget = cfg.memory_budget
        # 取出各种状态字节配置。
        state = cfg.state_bytes
        # 基于配置构造 layer bucket 规划。
        bucket_plans = LayerBucketPlanner(cfg).build()
        # 统计 bucket 总数。
        bucket_count = len(bucket_plans)
        # 统计其中 MTP bucket 的数量。
        mtp_bucket_count = sum(
            1 for bucket in bucket_plans if "mtp" in bucket.attention_types
        )
        # 记录单个 bucket 覆盖的最大层数。
        max_layers_per_bucket = max(
            (len(bucket.layer_indices) for bucket in bucket_plans),
            default=1,
        )

        # -----------------
        # 计算 routed experts、non-routed 与静态模块的参数规模。
        # 每个 routed expert 的参数量按 gate/up/down 三部分估算。
        params_per_routed_expert = 3 * model.hidden_size * model.moe_intermediate_size
        # 全模型 routed expert 参数总量等于层数 * expert 数 * 单 expert 参数量。
        routed_expert_params_total = (
            model.num_hidden_layers
            * model.num_experts
            * params_per_routed_expert
        )

        # 总参数量取“配置给出的总量”和“按 expert 反推的总量”中的较大值。
        total_params = max(
            int(model.total_params_billion * 1_000_000_000),
            routed_expert_params_total,
        )
        # non-routed 参数量等于总参数减 routed expert 参数。
        non_routed_params_total = max(total_params - routed_expert_params_total, 0)
        # 静态模块参数量在一个保守下界和 non-routed 总量之间截断。
        static_params_total = min(
            max(
                non_routed_params_total // (bucket_count + 1),
                model.hidden_size * model.num_attention_heads,
            ),
            non_routed_params_total,
        )
        # 剩余的 non-routed 参数分配给各个 bucket。
        bucket_non_routed_params_total = max(
            non_routed_params_total - static_params_total,
            0,
        )
        # 用 bucket 层数作为 non-routed 参数的权重。
        bucket_weights = tuple(
            max(1, len(bucket.layer_indices)) for bucket in bucket_plans
        )
        # 按权重把 non-routed 参数总量分摊到每个 bucket。
        bucket_non_routed_params_by_bucket = self._proportional_bucket_counts(
            total=bucket_non_routed_params_total,
            weights=bucket_weights,
        )
        # 每个 bucket 的 active routed 参数量按层数 * active experts_per_step 估算。
        bucket_active_routed_params_by_bucket = tuple(
            len(bucket.layer_indices)
            * cfg.expert_rotation.active_experts_per_step
            * params_per_routed_expert
            for bucket in bucket_plans
        )
        # 取各 bucket 中最大的 non-routed 参数量作为峰值。
        params_per_bucket_non_routed = max(
            bucket_non_routed_params_by_bucket,
            default=0,
        )
        # 取各 bucket 中最大的 active routed 参数量作为峰值。
        params_per_bucket_active_routed = max(
            bucket_active_routed_params_by_bucket,
            default=0,
        )
        # 预取 routed 参数量等于未来若干 bucket 峰值之和。
        params_per_bucket_prefetched_routed = sum(
            sorted(bucket_active_routed_params_by_bucket, reverse=True)[
                : cfg.bucket_schedule.prefetch_buckets
            ]
        )

        # -----------------
        # 估算激活、梯度缓冲、传输缓冲和权重 staging 大小。
        # 激活驻留量按 token 数、hidden_size、元素字节数和 live bucket 数估算。
        activation_resident_bytes = math.ceil(
            batch.total_tokens
            * model.hidden_size
            * state.activation_bytes_per_element
            * state.activation_residency_multiplier
            * cfg.bucket_schedule.max_live_buckets
        )
        # 读取 CPU 优化器状态存储 dtype 的字节数。
        cpu_optimizer_state_bytes_per_param = _storage_dtype_bytes(
            cfg.optimizer.cpu_state_storage_dtype
        )
        # 读取 host 梯度缓冲区存储 dtype 的字节数。
        host_gradient_buffer_bytes_per_param = _storage_dtype_bytes(
            cfg.optimizer.gradient_buffer_storage_dtype
        )
        # current_bucket_only 模式下梯度缓冲区只按当前 bucket 峰值估算。
        if cfg.bucket_schedule.host_gradient_buffer_scope == "current_bucket_only":
            host_gradient_buffer_params = (
                params_per_bucket_non_routed + params_per_bucket_active_routed
            )
        else:
            # full_model 模式下按全模型参数量估算梯度缓冲区。
            host_gradient_buffer_params = total_params
        # 计算 host 梯度缓冲区总字节数。
        host_gradient_buffer_bytes = (
            host_gradient_buffer_params * host_gradient_buffer_bytes_per_param
        )
        # 额外记录 full-model 梯度缓冲区理论大小，便于对比。
        full_model_gradient_buffer_bytes = (
            total_params * host_gradient_buffer_bytes_per_param
        )
        # 读取设备侧每参数权重字节数。
        device_weight_bytes_per_param = runtime_device_weight_bytes_per_param(cfg)
        # 若启用窗口内常驻 + update 后 offload，则额外为“退休 routed window”预留热驻留空间。
        retired_window_hot_params = 0
        if (
            cfg.expert_rotation.retain_active_window_state_in_memory
            and cfg.optimizer.offload_state_after_update
        ):
            retired_window_hot_params = params_per_bucket_active_routed
        # 传输层 staged file cache 预算直接由 transport 配置给出。
        transport_staged_file_cache_bytes = int(
            cfg.transport.max_staged_file_cache_gb * GiB
        )
        # weight stage buffer 需要覆盖一个峰值 bucket 的参数量。
        weight_stage_buffer_bytes = math.ceil(
            (params_per_bucket_non_routed + params_per_bucket_active_routed)
            * device_weight_bytes_per_param
        )
        # 若允许重叠，传输 staging 需要同时容纳梯度和权重缓冲区。
        if cfg.execution.overlap_backward_and_update:
            transfer_staging_buffer_bytes = (
                host_gradient_buffer_bytes + weight_stage_buffer_bytes
            )
        else:
            # 不重叠时只需要取二者中的较大值。
            transfer_staging_buffer_bytes = max(
                host_gradient_buffer_bytes,
                weight_stage_buffer_bytes,
            )

        # -----------------
        # 聚合 GPU hot / CPU hot / NVMe cold 三层驻留字节数。
        # GPU hot 主要包含当前峰值 bucket 的权重、梯度和激活。
        gpu_hot_bytes = (
            math.ceil(
                (params_per_bucket_non_routed + params_per_bucket_active_routed)
                * device_weight_bytes_per_param
            )
            + (params_per_bucket_non_routed + params_per_bucket_active_routed)
            * state.gradient_bytes_per_param
            + activation_resident_bytes
        )
        # 运行时量化 + GPU 训练时，退休 routed window 的 GPU packed cache 也会继续占用热预算。
        if cfg.execution.compute_device == "gpu" and retired_window_hot_params > 0:
            gpu_hot_bytes += math.ceil(
                retired_window_hot_params * device_weight_bytes_per_param
            )

        # 静态模块在 CPU hot 上常驻 master + optimizer state。
        static_cpu_state_bytes = static_params_total * (
            state.master_weight_bytes_per_param
            + cpu_optimizer_state_bytes_per_param
        )
        # 退休 routed window 继续常驻时，也要把对应的 master + optimizer state 算进 CPU hot。
        retired_window_cpu_state_bytes = retired_window_hot_params * (
            state.master_weight_bytes_per_param
            + cpu_optimizer_state_bytes_per_param
        )
        # CPU hot 包含静态状态、当前/预取 bucket 状态、文件缓存和 staging 缓冲区。
        cpu_hot_bytes = (
            static_cpu_state_bytes
            + (
                (
                    params_per_bucket_non_routed
                    + params_per_bucket_active_routed
                    + params_per_bucket_prefetched_routed
                )
                * (
                    state.master_weight_bytes_per_param
                    + cpu_optimizer_state_bytes_per_param
                )
            )
            + retired_window_cpu_state_bytes
            + transport_staged_file_cache_bytes
            + transfer_staging_buffer_bytes
        )

        # canonical training state 表示全模型 master + optimizer state 理论总量。
        canonical_training_state_bytes = total_params * (
            state.master_weight_bytes_per_param
            + cpu_optimizer_state_bytes_per_param
        )
        # NVMe cold 保存“全量状态”减去当前 CPU hot 常驻部分后的剩余量。
        nvme_cold_bytes = max(
            canonical_training_state_bytes
            - (
                static_cpu_state_bytes
                + (
                    (
                        params_per_bucket_non_routed
                        + params_per_bucket_active_routed
                        + params_per_bucket_prefetched_routed
                    )
                    * (
                        state.master_weight_bytes_per_param
                        + cpu_optimizer_state_bytes_per_param
                    )
                )
                + retired_window_cpu_state_bytes
            ),
            0,
        )

        # -----------------
        # 构造 tier summary，并返回最终训练内存规划。
        # 组装 GPU hot 层汇总对象。
        gpu_hot = MemoryTierSummary(
            resident_bytes=int(gpu_hot_bytes),
            budget_bytes=int(budget.gpu_hot_budget_gb * GiB),
            safety_margin_bytes=int(budget.gpu_safety_margin_gb * GiB),
        )
        # 组装 CPU hot 层汇总对象。
        cpu_hot = MemoryTierSummary(
            resident_bytes=int(cpu_hot_bytes),
            budget_bytes=int(budget.cpu_hot_budget_gb * GiB),
            safety_margin_bytes=int(budget.cpu_safety_margin_gb * GiB),
        )
        # 组装 NVMe cold 层汇总对象。
        nvme_cold = MemoryTierSummary(
            resident_bytes=int(nvme_cold_bytes),
            budget_bytes=int(budget.nvme_cold_budget_gb * GiB),
            safety_margin_bytes=int(budget.nvme_safety_margin_gb * GiB),
        )
        # 把全部估算结果封装成 TrainingMemoryPlan 返回。
        return TrainingMemoryPlan(
            bucket_count=bucket_count,
            mtp_bucket_count=mtp_bucket_count,
            max_layers_per_bucket=max_layers_per_bucket,
            total_params=total_params,
            routed_expert_params_total=routed_expert_params_total,
            non_routed_params_total=non_routed_params_total,
            static_params_total=static_params_total,
            bucket_non_routed_params_total=bucket_non_routed_params_total,
            params_per_routed_expert=params_per_routed_expert,
            params_per_bucket_non_routed=params_per_bucket_non_routed,
            params_per_bucket_active_routed=params_per_bucket_active_routed,
            params_per_bucket_prefetched_routed=params_per_bucket_prefetched_routed,
            bucket_non_routed_params_by_bucket=bucket_non_routed_params_by_bucket,
            bucket_active_routed_params_by_bucket=bucket_active_routed_params_by_bucket,
            activation_resident_bytes=int(activation_resident_bytes),
            cpu_optimizer_state_storage_dtype=cfg.optimizer.cpu_state_storage_dtype,
            cpu_optimizer_state_bytes_per_param=cpu_optimizer_state_bytes_per_param,
            host_gradient_buffer_storage_dtype=cfg.optimizer.gradient_buffer_storage_dtype,
            host_gradient_buffer_bytes_per_param=host_gradient_buffer_bytes_per_param,
            host_gradient_buffer_scope=cfg.bucket_schedule.host_gradient_buffer_scope,
            host_gradient_buffer_bytes=int(host_gradient_buffer_bytes),
            full_model_gradient_buffer_bytes=int(full_model_gradient_buffer_bytes),
            transport_staged_file_cache_bytes=int(transport_staged_file_cache_bytes),
            weight_stage_buffer_bytes=int(weight_stage_buffer_bytes),
            transfer_staging_buffer_bytes=int(transfer_staging_buffer_bytes),
            transfer_overlap_enabled=cfg.execution.overlap_backward_and_update,
            gpu_hot=gpu_hot,
            cpu_hot=cpu_hot,
            nvme_cold=nvme_cold,
        )


@dataclass(slots=True)
class TrainingStartupEstimator:
    config: TrainingProjectConfig

    # 克隆一份配置，避免在候选试探过程中污染原配置。
    def _clone_config(self) -> TrainingProjectConfig:
        # 先序列化再反序列化，得到独立配置副本。
        return TrainingProjectConfig.from_dict(self.config.to_dict())

    # 在给定预算与并发候选下估算单个启动方案。
    def _estimate_candidate(
        self,
        *,
        batch: BatchShape,
        gpu_hot_budget_gb: float,
        active_experts_per_step: int,
        max_live_buckets: int,
        prefetch_buckets: int,
    ) -> StartupParameterEstimate:
        # 先克隆配置，避免修改原配置对象。
        candidate = self._clone_config()
        # 覆盖当前候选的 GPU hot 预算。
        candidate.memory_budget.gpu_hot_budget_gb = gpu_hot_budget_gb
        # 覆盖当前候选的 active expert 数。
        candidate.expert_rotation.active_experts_per_step = active_experts_per_step
        # 覆盖当前候选允许的最大 live bucket 数。
        candidate.bucket_schedule.max_live_buckets = max_live_buckets
        # 覆盖当前候选的预取 bucket 数。
        candidate.bucket_schedule.prefetch_buckets = prefetch_buckets
        # 对候选配置再次执行校验。
        candidate.validate()
        # 构造该候选下的完整内存规划。
        plan = TrainingMemoryPlanner(candidate).build(batch)
        # 返回候选启动参数与对应估算结果。
        return StartupParameterEstimate(
            gpu_hot_budget_gb=gpu_hot_budget_gb,
            gpu_hot_available_bytes=plan.gpu_hot.available_bytes,
            active_experts_per_step=active_experts_per_step,
            max_live_buckets=max_live_buckets,
            prefetch_buckets=prefetch_buckets,
            batch=batch,
            fits_within_budget=plan.all_tiers_within_budget,
            planned_gpu_hot_bytes=plan.gpu_hot.resident_bytes,
            planned_cpu_hot_bytes=plan.cpu_hot.resident_bytes,
            planned_nvme_cold_bytes=plan.nvme_cold.resident_bytes,
        )

    # 为满足预算的候选方案生成排序键。
    @staticmethod
    def _candidate_sort_key(
        estimate: StartupParameterEstimate,
    ) -> tuple[float, int, int, int]:
        # 预算内方案优先选择填充率更高、并行度更大的方案。
        return (
            estimate.gpu_fill_ratio,
            estimate.active_experts_per_step,
            estimate.max_live_buckets,
            estimate.prefetch_buckets,
        )

    # 为超预算候选方案生成排序键，用于选择溢出最轻的方案。
    @staticmethod
    def _overflow_sort_key(
        estimate: StartupParameterEstimate,
    ) -> tuple[float, float, int, int, int]:
        # 先计算超出 GPU hot 可用容量的字节数。
        overflow_bytes = max(
            0,
            estimate.planned_gpu_hot_bytes - estimate.gpu_hot_available_bytes,
        )
        # 再把超出的绝对量换成相对比例。
        overflow_ratio = (
            overflow_bytes / estimate.gpu_hot_available_bytes
            if estimate.gpu_hot_available_bytes > 0
            else float("inf")
        )
        # 超预算方案优先选择溢出比例更小的组合。
        return (
            -overflow_ratio,
            estimate.gpu_fill_ratio,
            estimate.active_experts_per_step,
            estimate.max_live_buckets,
            estimate.prefetch_buckets,
        )

    # 穷举候选集合并为每个 GPU 预算挑选最佳启动参数。
    def estimate(
        self,
        *,
        batch: BatchShape,
        gpu_hot_budget_candidates_gb: tuple[float, ...],
        active_expert_candidates: tuple[int, ...],
        max_live_bucket_candidates: tuple[int, ...],
        prefetch_bucket_candidates: tuple[int, ...],
    ) -> tuple[StartupParameterEstimate, ...]:
        # -----------------
        # 逐个 GPU 预算枚举候选参数，并拆分为可行 / 超预算两类。
        # 保存每个 GPU 预算最终选中的候选方案。
        estimates: list[StartupParameterEstimate] = []
        for gpu_hot_budget_gb in gpu_hot_budget_candidates_gb:
            # fit_candidates 保存预算内方案。
            fit_candidates: list[StartupParameterEstimate] = []
            # overflow_candidates 保存超预算方案。
            overflow_candidates: list[StartupParameterEstimate] = []
            for active_experts_per_step in active_expert_candidates:
                for max_live_buckets in max_live_bucket_candidates:
                    for prefetch_buckets in prefetch_bucket_candidates:
                        # 预取 bucket 数不能超过 live bucket 数。
                        if prefetch_buckets > max_live_buckets:
                            continue
                        # 估算当前候选组合的内存占用。
                        estimate = self._estimate_candidate(
                            batch=batch,
                            gpu_hot_budget_gb=gpu_hot_budget_gb,
                            active_experts_per_step=active_experts_per_step,
                            max_live_buckets=max_live_buckets,
                            prefetch_buckets=prefetch_buckets,
                        )
                        # 按是否预算内，把候选放到不同列表。
                        if estimate.fits_within_budget:
                            fit_candidates.append(estimate)
                        else:
                            overflow_candidates.append(estimate)

            # -----------------
            # 优先返回预算内最优方案，否则回退到溢出最轻方案。
            if fit_candidates:
                # 预算内候选按“越满越好”排序。
                fit_candidates.sort(
                    key=self._candidate_sort_key,
                    reverse=True,
                )
                # 选取当前预算下最优的预算内方案。
                estimates.append(fit_candidates[0])
                continue
            if overflow_candidates:
                # 没有预算内方案时，选溢出最轻的方案。
                overflow_candidates.sort(
                    key=self._overflow_sort_key,
                    reverse=True,
                )
                # 记录当前预算下最优的超预算候选。
                estimates.append(overflow_candidates[0])
        # 返回每个 GPU 预算对应选中的候选方案。
        return tuple(estimates)
