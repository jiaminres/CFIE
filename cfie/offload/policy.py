"""Automatic planning for tiered MoE expert caching."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from cfie.logger import init_logger
from cfie.transformers_utils.config import get_safetensors_params_metadata
from cfie.utils.mem_utils import split_gpu_memory_budget

logger = init_logger(__name__)

PLAN_KEY = "moe_tiered_cache"
TARGET_OCCUPIED_GPU_BYTES_KEY = "moe_tiered_cache_target_occupied_gpu_bytes"
MTP_RESERVE_MODE_KEY = "moe_tiered_cache_mtp_reserve_mode"
GiB = 1 << 30
DEFAULT_STAGE_BYTES = 1 * GiB
DEFAULT_DYNAMIC_RESERVE_BYTES = 3 * GiB
DEFAULT_CPU_MIN_FREE_BYTES = 12 * GiB
DEFAULT_CPU_MIN_FREE_FRACTION = 0.10
DEFAULT_CPU_SHARED_RESERVE_BYTES = 0
# Keep the CPU expert cache well below the host-memory headroom so large
# checkpoints do not turn the cache itself into the dominant resident set.
DEFAULT_CPU_CACHE_BUDGET_FRACTION = 0.50
# When the conservative baseline still leaves NVMe spill, allow the planner
# to consume more of the post-watermark headroom before giving up on CPU.
DEFAULT_CPU_CACHE_BOOST_FRACTION = 0.75
DEFAULT_MTP_BASE_GPU_SLOTS = 8

QWEN35_MOE_MODEL_TYPE = "qwen3_5_moe"
QWEN35_MTP_MODEL_TYPE = "qwen3_5_mtp"
QWEN35_MOE_MTP_ARCH = "Qwen3_5MoeMTP"

_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "U16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


@dataclass(slots=True)
class MoeTieredCachePlan:
    enabled: bool
    reason: str
    model_path: str = ""
    model_type: str = ""
    quantization: str = ""
    num_moe_layers: int = 0
    num_experts: int = 0
    top_k: int = 0
    expert_bytes_total: int = 0
    expert_bytes_per_layer: int = 0
    expert_bytes_per_expert: int = 0
    expert_bytes_per_slot_all_layers: int = 0
    dense_bytes: int = 0
    kv_bytes: int = 0
    linear_state_bytes: int = 0
    dynamic_bytes: int = 0
    resident_bytes: int = 0
    shared_gpu_reserve_bytes: int = 0
    gpu_budget_bytes: int = 0
    gpu_runtime_headroom_bytes: int = 0
    gpu_expert_budget_bytes: int = 0
    gpu_slots_per_layer: int = 0
    prefill_burst_slots: int = 0
    prefill_burst_bytes: int = 0
    cpu_budget_bytes: int = 0
    cpu_static_bytes: int = 0
    cpu_slots_per_layer: int = 0
    staging_bytes: int = DEFAULT_STAGE_BYTES
    nvme_expert_bytes: int = 0
    initial_gpu_experts: tuple[int, ...] = ()
    initial_cpu_experts: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        # 统一把 dataclass 计划对象转成可注入 additional_config 的普通字典。
        return asdict(self)


def get_moe_tiered_cache_plan(cfie_config: Any) -> dict[str, Any] | None:
    # 从总配置对象上读取 additional_config 容器。
    additional_config = getattr(cfie_config, "additional_config", None)
    # additional_config 不是字典时，说明当前配置里没有可读取 plan 的位置。
    if not isinstance(additional_config, dict):
        return None
    # 读取约定键名对应的 plan。
    plan = additional_config.get(PLAN_KEY)
    # 只有字典形态的 plan 才视为有效。
    return plan if isinstance(plan, dict) else None


def maybe_inject_moe_tiered_cache_plan(cfie_config: Any) -> None:
    # ----------------- plan 注入入口 -----------------
    # 这一步发生在 CfieConfig.__post_init__ 阶段。
    # 对 target 来说，它既负责产出 target 自身 plan，也可能触发一轮 reserve-only 的 MTP 递归预估。
    # 从总配置对象上读取 additional_config 容器。
    additional_config = getattr(cfie_config, "additional_config", None)
    # 若 additional_config 不是字典，则没有可写入 plan 的位置，直接返回。
    if not isinstance(additional_config, dict):
        return
    # 若当前配置里已经带有 plan，则不重复构建。
    if PLAN_KEY in additional_config:
        return
    # 构建当前配置视角下的 plan，并以字典形式注入到 additional_config 中。
    plan = build_moe_tiered_cache_plan(cfie_config)
    additional_config[PLAN_KEY] = plan.to_dict()
    # target 规划完成后，额外记录其已经占据的 GPU 预算。
    # 真正创建 draft/MTP 配置时，会把这个数字带过去，按“总预算 - target 已占”重新规划。
    _maybe_record_target_occupied_gpu_bytes(
        cfie_config=cfie_config,
        additional_config=additional_config,
        plan=plan,
    )


def build_moe_tiered_cache_plan(cfie_config: Any) -> MoeTieredCachePlan:
    # ------------------ 基础前置校验：模型类型、并行方式、量化后端 ------------------
    # 读取模型配置对象；后续几乎所有规划都依赖它。
    model_config = getattr(cfie_config, "model_config", None)
    # 只有 MoE 模型才需要构建专家分层缓存计划；否则直接返回 disabled plan。
    if model_config is None or not model_config.is_moe:
        return MoeTieredCachePlan(enabled=False, reason="not_moe_model")

    # 当前实现只支持 TP=1 的 tiered cache 规划；TP>1 直接禁用。
    if getattr(cfie_config.parallel_config, "tensor_parallel_size", 1) != 1:
        return MoeTieredCachePlan(enabled=False, reason="tensor_parallel_not_supported")

    # 当前实现还不支持 EPLB 场景，因此开启 EPLB 时也直接禁用。
    if getattr(cfie_config.parallel_config, "enable_eplb", False):
        return MoeTieredCachePlan(enabled=False, reason="eplb_not_supported")

    # 读取量化配置对象，用于判断当前权重格式是否适配该 planner。
    quant_config = getattr(cfie_config, "quant_config", None)
    # 优先通过 quant_config.get_name() 读取量化后端名；否则退回 model_config.quantization。
    quant_name = (
        quant_config.get_name()
        if quant_config is not None and hasattr(quant_config, "get_name")
        else str(getattr(model_config, "quantization", ""))
    )
    # 当前 planner 只支持 gptq_marlin；其他量化后端一律不启用。
    if quant_name != "gptq_marlin":
        return MoeTieredCachePlan(enabled=False, reason="quantization_not_supported")

    # desc_act=True 的 GPTQ 目前不在支持范围内。
    if bool(getattr(quant_config, "desc_act", False)):
        return MoeTieredCachePlan(enabled=False, reason="desc_act_not_supported")

    # 读取 Hugging Face 原始配置与文本子配置。
    hf_config = getattr(model_config, "hf_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    # 若缺失 HF 配置，无法识别模型结构和 MoE 维度，直接禁用。
    if hf_config is None or hf_text_config is None:
        return MoeTieredCachePlan(enabled=False, reason="missing_hf_config")

    # 记录原始 model_type，供最后回填到 plan 中与日志输出。
    model_type = getattr(hf_config, "model_type", "")
    # 解析当前模型属于 target 规划还是 MTP 规划模式。
    planning_mode = _resolve_qwen35_moe_planning_mode(hf_config)
    # 若不是当前 planner 支持的模型类型，则直接禁用。
    if planning_mode is None:
        return MoeTieredCachePlan(enabled=False, reason="model_type_not_supported")

    # mtp reserve 模式只用于 target 递归预估 draft 侧 GPU 预留，不代表最终 draft 计划。
    mtp_reserve_mode = _is_mtp_reserve_mode(cfie_config)

    # 真实 draft 计划会读取 target 已占据的 GPU 预算，再按剩余空间重新分配 resident slots。
    target_occupied_gpu_bytes = _get_target_occupied_gpu_bytes(cfie_config)

    # 读取模型本地目录路径；planner 只支持本地目录，不支持纯 repo id。
    model_path = getattr(model_config, "model", "")
    # 若模型路径不是本地目录，则无法基于本地 safetensors 元数据做规划。
    if not Path(model_path).is_dir():
        return MoeTieredCachePlan(enabled=False, reason="model_path_not_local")

    # ------------------ 读取 safetensors 元数据，并识别当前变体的专家层结构 ------------------
    # 扫描 safetensors 元数据，用于统计 dense / expert 权重大小。
    metadata = get_safetensors_params_metadata(model_path)
    # 若一个有效的 safetensors 元数据映射都拿不到，则无法规划。
    if not metadata:
        return MoeTieredCachePlan(enabled=False, reason="missing_safetensors_metadata")

    # 依据 planning_mode 过滤出当前变体真正参与规划的权重元数据。
    variant_metadata = _filter_metadata_for_planning(
        metadata=metadata,
        planning_mode=planning_mode,
    )
    # 若过滤后为空，说明缺少目标变体的关键张量。
    if not variant_metadata:
        return MoeTieredCachePlan(enabled=False, reason="missing_variant_tensors")

    # 从元数据名字里提取出所有专家层前缀，用于统计 MoE 层数。
    layer_prefixes = sorted(
        {
            prefix
            for name in variant_metadata
            if (prefix := _extract_expert_layer_prefix(name)) is not None
        }
    )
    # 若完全提取不到专家层前缀，则说明专家张量不存在或命名不匹配。
    if not layer_prefixes:
        return MoeTieredCachePlan(enabled=False, reason="missing_expert_tensors")

    # 计算模型中的 MoE 层数量。
    num_moe_layers = len(layer_prefixes)
    # 读取每层总专家数。
    num_experts = int(getattr(hf_text_config, "num_experts", 0))
    # 读取每个 token 路由到的专家数 top-k。
    top_k = int(getattr(hf_text_config, "num_experts_per_tok", 0))
    # 维度非法时无法继续规划。
    if num_experts <= 0 or top_k <= 0:
        return MoeTieredCachePlan(enabled=False, reason="invalid_moe_dimensions")

    # ------------------ 统计专家体积与 dense 体积，得到规划所需的核心规模参数 ------------------
    # 统计总权重大小、全部专家权重大小，以及“每层-每专家”的大小映射。
    (
        total_bytes,
        expert_bytes_total,
        layer_expert_bytes,
    ) = _collect_expert_bytes(variant_metadata)

    # 若统计不到每层专家大小，则说明关键专家张量不完整。
    if not layer_expert_bytes:
        return MoeTieredCachePlan(enabled=False, reason="missing_first_expert_bytes")

    # 把每一层“单个专家大小”折叠成该层的 worst-case expert 大小。
    layer_expert_sizes = {
        layer_prefix: max(expert_sizes.values(), default=0)
        for layer_prefix, expert_sizes in layer_expert_bytes.items()
    }

    # 只要有任一层的单专家大小为 0，就说明元数据不完整，直接禁用。
    if not all(expert_bytes > 0 for expert_bytes in layer_expert_sizes.values()):
        return MoeTieredCachePlan(enabled=False, reason="missing_first_expert_bytes")

    # 取所有层中最大的“单个专家大小”，作为 burst 相关预算的最坏情况。
    expert_bytes_per_expert = max(layer_expert_sizes.values())

    # 取所有层中最大的“整层全部专家大小”，方便记录总体规模信息。
    expert_bytes_per_layer = max(
        expert_bytes * num_experts for expert_bytes in layer_expert_sizes.values()
    )

    # 把各层单专家大小求和，得到“每层常驻 1 个专家跨全模型总共要多少字节”。
    expert_bytes_per_slot_all_layers = sum(layer_expert_sizes.values())

    # dense 权重大小等于总权重减去全部专家权重。
    dense_bytes = total_bytes - expert_bytes_total

    # ------------------ 先做 GPU 侧预算规划，决定常驻 experts 与 prefill burst 空间 ------------------

    # 估算本机在当前配置下，静态规划部分可用的 GPU 预算。
    gpu_budget_bytes = _get_gpu_budget_bytes(cfie_config)
    # 估算 ratio 外保留给运行时峰值的 GPU headroom。
    gpu_runtime_headroom_bytes = _get_gpu_runtime_headroom_bytes(cfie_config)

    # 若算出来没有 GPU 预算，通常意味着非 CUDA 设备，直接禁用。
    if gpu_budget_bytes <= 0:
        return MoeTieredCachePlan(enabled=False, reason="non_cuda_device")

    # 估算 KV cache 与线性状态缓存的预算。
    kv_bytes, linear_state_bytes = _estimate_hybrid_cache_bytes(cfie_config)

    # 估算运行时峰值需要的动态 headroom。
    # 这部分只做记录与日志，不再占用静态规划预算。
    dynamic_bytes = _estimate_dynamic_reserve_bytes(
        planning_mode=planning_mode,
        gpu_budget_bytes=gpu_budget_bytes,
    )
    if dynamic_bytes > gpu_runtime_headroom_bytes:
        logger.warning(
            "Estimated runtime peak exceeds configured runtime headroom: "
            "model=%s estimated_runtime_peak=%.2f GiB runtime_headroom=%.2f GiB. "
            "Actual runtime profiling may require lowering gpu_memory_utilization.",
            model_path,
            dynamic_bytes / GiB,
            gpu_runtime_headroom_bytes / GiB,
        )

    # resident_bytes 只表示“当前规划视角自身的静态非专家常驻内容”。
    resident_bytes = dense_bytes + kv_bytes + linear_state_bytes

    # target 侧会为 draft reserve 预留预算；真实 draft 侧则会反向扣减 target 已占预算。
    shared_gpu_reserve_bytes = _estimate_shared_gpu_reserve_bytes(
        cfie_config=cfie_config,
        planning_mode=planning_mode,
        mtp_reserve_mode=mtp_reserve_mode,
        target_occupied_gpu_bytes=target_occupied_gpu_bytes,
    )

    # 用总 GPU 预算减去 target 自身常驻量和共享/draft 预留量，得到理论上的专家显存预算。
    raw_gpu_expert_budget_bytes = max(
        0,
        gpu_budget_bytes - resident_bytes - shared_gpu_reserve_bytes,
    )

    # 进一步把专家显存预算换算成“每层最多能常驻多少个专家 slot”。
    raw_gpu_slots_per_layer = min(
        num_experts,
        raw_gpu_expert_budget_bytes // max(1, expert_bytes_per_slot_all_layers),
    )

    # 若连 top-k 个专家都无法常驻，则当前 GPU 预算不足以运行该方案，直接抛错终止启动。
    if raw_gpu_slots_per_layer < top_k:
        # top-k 常驻所需的最小专家显存预算。
        required_gpu_expert_budget_bytes = expert_bytes_per_slot_all_layers * top_k
        # 满足该最小专家预算时，对应的总 GPU 预算下限。
        required_total_gpu_budget_bytes = (
                resident_bytes
                + shared_gpu_reserve_bytes
                + required_gpu_expert_budget_bytes
        )
        # 直接抛出带详细预算拆分的错误，提示当前 GPU 分配无法满足该 MoE 模型。
        raise ValueError(
            "Insufficient GPU budget for MoE tiered cache: "
            f"model={model_path} model_type={model_type} quantization={quant_name} "
            f"top_k={top_k} raw_gpu_slots_per_layer={raw_gpu_slots_per_layer}. "
            "Current GPU memory allocation cannot satisfy the minimum MoE "
            "residency requirement for this model. "
            f"static_budget={gpu_budget_bytes / GiB:.2f} GiB "
            f"runtime_headroom={gpu_runtime_headroom_bytes / GiB:.2f} GiB "
            f"resident={resident_bytes / GiB:.2f} GiB "
            f"(dense={dense_bytes / GiB:.2f} GiB "
            f"kv={kv_bytes / GiB:.2f} GiB "
            f"linear_state={linear_state_bytes / GiB:.2f} GiB) "
            f"estimated_runtime_peak={dynamic_bytes / GiB:.2f} GiB "
            f"shared_gpu_reserve={shared_gpu_reserve_bytes / GiB:.2f} GiB "
            f"gpu_expert_budget={raw_gpu_expert_budget_bytes / GiB:.2f} GiB "
            f"required_expert_budget={required_gpu_expert_budget_bytes / GiB:.2f} GiB "
            f"required_total_gpu_budget={required_total_gpu_budget_bytes / GiB:.2f} GiB "
            f"expert_bytes_per_slot_all_layers={expert_bytes_per_slot_all_layers / GiB:.2f} GiB."
        )

    # reserve-only 的 mtp 临时计划必须保留 base_8 语义，不能在递归预估阶段直接扩成全量常驻。
    # 除此之外，target 和真正的 draft 计划只要预算足够，都可以直接走 full residency。
    if raw_gpu_expert_budget_bytes >= expert_bytes_total and not mtp_reserve_mode:
        return MoeTieredCachePlan(
            enabled=False,
            reason="full_gpu_residency_already_possible",
            model_path=model_path,
            model_type=model_type,
            quantization=quant_name,
            num_moe_layers=num_moe_layers,
            num_experts=num_experts,
            top_k=top_k,
            expert_bytes_total=expert_bytes_total,
            expert_bytes_per_layer=expert_bytes_per_layer,
            expert_bytes_per_expert=expert_bytes_per_expert,
            expert_bytes_per_slot_all_layers=expert_bytes_per_slot_all_layers,
            dense_bytes=dense_bytes,
            kv_bytes=kv_bytes,
            linear_state_bytes=linear_state_bytes,
            dynamic_bytes=dynamic_bytes,
            resident_bytes=resident_bytes,
            shared_gpu_reserve_bytes=shared_gpu_reserve_bytes,
            gpu_budget_bytes=gpu_budget_bytes,
            gpu_runtime_headroom_bytes=gpu_runtime_headroom_bytes,
            gpu_expert_budget_bytes=expert_bytes_total,
            gpu_slots_per_layer=num_experts,
        )

    # 不管 raw 预算多小，也至少要为 top-k 个常驻专家保留显存预算。
    min_gpu_expert_budget_bytes = expert_bytes_per_slot_all_layers * top_k

    # 真正参与后续规划的 GPU 专家预算取“raw 预算”和“top-k 最小预算”的较大者。
    gpu_expert_budget_bytes = max(
        min_gpu_expert_budget_bytes,
        raw_gpu_expert_budget_bytes,
    )

    # MTP reserve-only 与 standalone mtp 仍沿用固定 base_8 resident 逻辑。
    fixed_gpu_slots_per_layer = _resolve_fixed_gpu_slots_per_layer(
        planning_mode=planning_mode,
        top_k=top_k,
        num_experts=num_experts,
        mtp_reserve_mode=mtp_reserve_mode,
        target_occupied_gpu_bytes=target_occupied_gpu_bytes,
    )

    # 真正的 draft 计划如果拿到了 target 已占预算，就只按剩余 GPU 空间重算 resident slots，
    # 不再沿用 reserve 阶段的 base_8，也不额外启用 burst 池。
    if planning_mode == "mtp" and target_occupied_gpu_bytes > 0 and not mtp_reserve_mode:
        gpu_slots_per_layer = min(
            num_experts,
            raw_gpu_slots_per_layer,
        )
        gpu_slots_per_layer = max(top_k, gpu_slots_per_layer)
        prefill_burst_slots = 0
        prefill_burst_bytes = 0
        gpu_expert_budget_bytes = (
            gpu_slots_per_layer * expert_bytes_per_slot_all_layers
        )
    # 若当前模式要求固定 GPU slot 数，则不再额外规划 burst 池。
    elif fixed_gpu_slots_per_layer is not None:
        # 取“总专家数 / raw 能放下的数 / 固定目标值”三者最小值作为最终常驻 slot 数。
        gpu_slots_per_layer = min(
            num_experts,
            raw_gpu_slots_per_layer,
            fixed_gpu_slots_per_layer,
        )
        # 但仍需保证最终常驻数不少于 top-k。
        gpu_slots_per_layer = max(top_k, gpu_slots_per_layer)
        # 固定常驻模式下不启用 prefill burst。
        prefill_burst_slots = 0
        # 对应的 burst 临时池显存也为 0。
        prefill_burst_bytes = 0
        # GPU 专家预算收缩为“常驻 slot 数 * 每 slot 跨层总字节”。
        gpu_expert_budget_bytes = (
                gpu_slots_per_layer * expert_bytes_per_slot_all_layers
        )
    else:
        # 非固定模式下，先按“全量 prefill burst 池”预留预算，再用剩余预算计算常驻 resident slots。
        # 仅当这会挤掉 top-k 常驻下限时，才收缩 burst，保证 resident 至少能覆盖 top-k。
        gpu_slots_per_layer, prefill_burst_slots = _plan_prefill_burst_slots(
            num_experts=num_experts,
            top_k=top_k,
            total_gpu_budget_bytes=int(gpu_expert_budget_bytes),
            expert_bytes_per_slot_all_layers=expert_bytes_per_slot_all_layers,
            expert_bytes_per_expert=expert_bytes_per_expert,
        )

        # burst 池显存预算等于 burst slot 数乘以单专家最大字节数。
        prefill_burst_bytes = prefill_burst_slots * expert_bytes_per_expert

        # 最终 GPU 专家预算要把常驻 slot 和实际保留下来的 burst 临时池都算进去。
        gpu_expert_budget_bytes = (
                gpu_slots_per_layer * expert_bytes_per_slot_all_layers + prefill_burst_bytes
        )

    # ------------------ 再做 CPU 侧预算规划，决定 static experts、staging 与 NVMe spill ------------------

    # 计算在 GPU 常驻之后，还剩多少专家候选需要落到 CPU/NVMe。
    remaining_cpu_candidates = max(
        0,
        num_experts - gpu_slots_per_layer,
    )

    # 读取当前系统可用物理内存。
    available_system_bytes = _get_available_system_ram_bytes()

    # 读取当前系统总物理内存。
    total_system_bytes = _get_total_system_ram_bytes()

    # 估算主机侧最少需要保留的空闲内存水位。
    cpu_min_free_bytes = _estimate_cpu_min_free_bytes(
        cfie_config=cfie_config,
        total_system_bytes=total_system_bytes,
    )

    # 估算还需要额外预留给共享用途的 CPU 内存。
    cpu_shared_reserve_bytes = _estimate_cpu_shared_reserve_bytes(cfie_config)

    # 计算全部专家里，在 GPU 部分之外还剩多少专家字节需要找落点。
    remaining_expert_bytes = max(
        0,
        expert_bytes_total - gpu_expert_budget_bytes,
    )

    # 估算即使 CPU 不做静态缓存，至少也需要给 staging 预留多少内存。
    minimal_staging_bytes = _estimate_minimal_cpu_staging_bytes(
        cfie_config=cfie_config,
        available_system_bytes=available_system_bytes,
        remaining_expert_bytes=remaining_expert_bytes,
        expert_bytes_per_slot_all_layers=expert_bytes_per_slot_all_layers,
        remaining_cpu_candidates=remaining_cpu_candidates,
    )

    # 先把 CPU 预算上限初始化为“剩余专家总字节数”。
    cpu_budget_cap_bytes = remaining_expert_bytes

    # 若除了 staging 之外，CPU 还有希望静态常驻部分专家，则把预算上限改为“静态专家 + staging”。
    if remaining_expert_bytes > minimal_staging_bytes:
        # remaining_resident_expert_bytes 表示所有剩余 CPU 候选若全部静态缓存，需要多少字节。
        remaining_resident_expert_bytes = (
                remaining_cpu_candidates * expert_bytes_per_slot_all_layers
        )
        # CPU 真正有意义的预算上限，就是“静态部分 + staging”。
        cpu_budget_cap_bytes = remaining_resident_expert_bytes + minimal_staging_bytes

    # 用主机侧预算估算器得到一个保守 CPU 预算，再与预算上限取最小值。
    cpu_budget_bytes = min(
        _estimate_cpu_cache_budget_bytes(
            cfie_config=cfie_config,
            available_system_bytes=available_system_bytes,
            total_system_bytes=total_system_bytes,
        ),
        cpu_budget_cap_bytes,
    )

    # 若当前 CPU 预算连最基本的 staging 都装不下，则至少借出 minimal staging 预算。
    if cpu_budget_bytes < minimal_staging_bytes < remaining_expert_bytes:
        logger.info(
            "Borrowing minimal CPU staging budget below min-free watermark: "
            "model=%s staging=%.2f MiB available=%.2f GiB min_free=%.2f GiB",
            model_path,
            minimal_staging_bytes / (1 << 20),
            available_system_bytes / GiB,
            cpu_min_free_bytes / GiB,
        )
        # 强行把 CPU 预算抬到 minimal staging 水平，确保仍可运行分层加载。
        cpu_budget_bytes = minimal_staging_bytes

    # 若保守 CPU 预算只差不到 1 个跨层 expert slot 就能消除 NVMe spill，则尝试做一次温和提升。
    if (
            remaining_expert_bytes > minimal_staging_bytes
            and cpu_budget_bytes < cpu_budget_cap_bytes
            and cpu_budget_cap_bytes - cpu_budget_bytes <= expert_bytes_per_slot_all_layers
    ):
        # 这里用更激进的 boost fraction 重新估算一次 CPU 预算，但仍不允许超过理论预算上限。
        boosted_cpu_budget_bytes = min(
            _estimate_cpu_cache_budget_bytes(
                cfie_config=cfie_config,
                available_system_bytes=available_system_bytes,
                total_system_bytes=total_system_bytes,
                budget_fraction=DEFAULT_CPU_CACHE_BOOST_FRACTION,
            ),
            cpu_budget_cap_bytes,
        )

        # 只有在 boost 确实能继续抬高预算时才采用，避免无效覆盖。
        if boosted_cpu_budget_bytes > cpu_budget_bytes:
            logger.info(
                "Boosting CPU budget to avoid small NVMe spill: model=%s "
                "cpu_budget=%.2f GiB boosted=%.2f GiB cap=%.2f GiB",
                model_path,
                cpu_budget_bytes / GiB,
                boosted_cpu_budget_bytes / GiB,
                cpu_budget_cap_bytes / GiB,
            )
            cpu_budget_bytes = boosted_cpu_budget_bytes

    # 若还有 CPU 候选专家且 CPU 预算非零，则默认给 staging 留 2 个 slot 的空间。
    staging_slots_per_layer = (
        2 if remaining_cpu_candidates > 0 and cpu_budget_bytes > 0 else 0
    )

    # staging_bytes 等于“总 CPU 预算”和“2 个 slot staging 空间”中的较小值。
    staging_bytes = min(
        cpu_budget_bytes,
        staging_slots_per_layer * expert_bytes_per_slot_all_layers,
    )

    # 扣掉 staging 后，其余部分都视为 CPU 静态缓存预算。
    cpu_static_bytes = max(0, cpu_budget_bytes - staging_bytes)

    # 把静态缓存预算换算成“每层可静态缓存多少个 CPU 专家”。
    cpu_slots_per_layer = min(
        remaining_cpu_candidates,
        cpu_static_bytes // max(1, expert_bytes_per_slot_all_layers),
    )

    # 还装不进 GPU 和 CPU 的那部分专家字节，最终只能落到 NVMe。
    nvme_expert_bytes = max(
        0,
        expert_bytes_total
        - gpu_slots_per_layer * expert_bytes_per_slot_all_layers
        - cpu_slots_per_layer * expert_bytes_per_slot_all_layers,
    )

    # ------------------ 汇总最终 plan，并把自动规划结果打到启动日志 ------------------
    # 汇总前面所有中间结果，生成最终的 enabled plan 对象。
    plan = MoeTieredCachePlan(
        enabled=True,
        reason="enabled",
        # 记录模型本地路径，后续运行时会据此重新打开 safetensors expert store。
        model_path=model_path,
        # 记录原始模型类型，便于日志与运行时区分不同规划模式。
        model_type=model_type,
        # 记录量化后端名，便于运行时附加校验。
        quantization=quant_name,
        # 记录 MoE 层数。
        num_moe_layers=num_moe_layers,
        # 记录每层总专家数。
        num_experts=num_experts,
        # 记录每 token 路由到的专家数。
        top_k=top_k,
        # 记录专家总字节量。
        expert_bytes_total=expert_bytes_total,
        # 记录最坏层的整层专家字节量。
        expert_bytes_per_layer=expert_bytes_per_layer,
        # 记录最坏单专家字节量。
        expert_bytes_per_expert=expert_bytes_per_expert,
        # 记录“每层 1 个专家 slot 跨全模型总字节量”。
        expert_bytes_per_slot_all_layers=expert_bytes_per_slot_all_layers,
        # 记录 dense 权重字节量。
        dense_bytes=dense_bytes,
        # 记录 KV cache 预算字节量。
        kv_bytes=kv_bytes,
        # 记录线性状态缓存预算字节量。
        linear_state_bytes=linear_state_bytes,
        # 记录动态预留字节量。
        dynamic_bytes=dynamic_bytes,
        # 记录非专家常驻内容总字节量。
        resident_bytes=resident_bytes,
        # 记录 target 线路为 MTP/shared draft 额外让出的 GPU 预算。
        shared_gpu_reserve_bytes=shared_gpu_reserve_bytes,
        # 记录静态规划可用的总 GPU 预算。
        gpu_budget_bytes=gpu_budget_bytes,
        # 记录 ratio 外留给运行时峰值的 GPU 余量。
        gpu_runtime_headroom_bytes=gpu_runtime_headroom_bytes,
        # 记录实际分配给专家相关内容的 GPU 预算。
        gpu_expert_budget_bytes=gpu_expert_budget_bytes,
        # 记录每层最终 GPU 常驻 slot 数。
        gpu_slots_per_layer=int(gpu_slots_per_layer),
        # 记录 prefill burst 临时池的 slot 数。
        prefill_burst_slots=int(prefill_burst_slots),
        # 记录 prefill burst 临时池的字节预算。
        prefill_burst_bytes=int(prefill_burst_bytes),
        # 记录最终 CPU 总预算。
        cpu_budget_bytes=int(cpu_budget_bytes),
        # 记录 CPU 静态专家缓存预算。
        cpu_static_bytes=int(cpu_static_bytes),
        # 记录每层可静态缓存到 CPU 的专家 slot 数。
        cpu_slots_per_layer=int(cpu_slots_per_layer),
        # 记录 staging buffer 总预算。
        staging_bytes=int(staging_bytes),
        # 记录最终仍需落到 NVMe 的专家字节量。
        nvme_expert_bytes=int(nvme_expert_bytes),
        # 默认把前 gpu_slots_per_layer 个专家放进初始 GPU 常驻集合。
        initial_gpu_experts=tuple(range(int(gpu_slots_per_layer))),
        # 默认把紧随其后的 cpu_slots_per_layer 个专家放进初始 CPU 静态集合。
        initial_cpu_experts=tuple(
            range(
                int(gpu_slots_per_layer),
                int(gpu_slots_per_layer + cpu_slots_per_layer),
            )
        ),
    )
    # 打印最终自动规划结果，便于启动日志中快速核对 GPU/CPU/NVMe 分配。
    logger.info(
        "Auto-enabled MoE tiered cache: model=%s gpu_slots/layer=%d "
        "prefill_burst_slots=%d cpu_slots/layer=%d cpu_static=%.2f GiB "
        "cpu_stage=%.2f GiB "
        "cpu_total=%.2f GiB min_free=%.2f GiB shared_reserve=%.2f GiB "
        "static_budget=%.2f GiB runtime_headroom=%.2f GiB "
        "resident=%.2f GiB estimated_runtime_peak=%.2f GiB "
        "shared_gpu_reserve=%.2f GiB "
        "gpu_expert_budget=%.2f GiB nvme=%.2f GiB",
        model_path,
        plan.gpu_slots_per_layer,
        plan.prefill_burst_slots,
        plan.cpu_slots_per_layer,
        plan.cpu_static_bytes / GiB,
        plan.staging_bytes / GiB,
        plan.cpu_budget_bytes / GiB,
        cpu_min_free_bytes / GiB,
        cpu_shared_reserve_bytes / GiB,
        plan.gpu_budget_bytes / GiB,
        plan.gpu_runtime_headroom_bytes / GiB,
        plan.resident_bytes / GiB,
        plan.dynamic_bytes / GiB,
        plan.shared_gpu_reserve_bytes / GiB,
        plan.gpu_expert_budget_bytes / GiB,
        plan.nvme_expert_bytes / GiB,
    )
    # 返回最终规划对象。
    return plan


def _resolve_qwen35_moe_planning_mode(hf_config: Any) -> str | None:
    # 先读取原始 HF config 中声明的 model_type。
    model_type = str(getattr(hf_config, "model_type", ""))
    if model_type == QWEN35_MOE_MODEL_TYPE:
        # 命中主干 target 变体时，返回 target 规划模式。
        return "target"

    if model_type == QWEN35_MTP_MODEL_TYPE:
        # MTP 变体还要进一步确认 architectures 中确实带有对应架构名。
        architectures = tuple(
            str(arch) for arch in (getattr(hf_config, "architectures", None) or ())
        )
        if any(arch == QWEN35_MOE_MTP_ARCH for arch in architectures):
            # 命中草稿 MTP 变体时，返回 mtp 规划模式。
            return "mtp"

    # 其余 model_type 目前都不在 planner 支持范围内。
    return None


def _filter_metadata_for_planning(
        *,
        metadata: dict[str, Any],
        planning_mode: str,
) -> dict[str, Any]:
    # target 规划只关心主干参数，因此过滤掉 mtp.* 开头的草稿权重。
    if planning_mode == "target":
        return {
            name: info for name, info in metadata.items() if not name.startswith("mtp.")
        }
    # mtp 规划只关心草稿分支参数，因此只保留 mtp.* 权重。
    if planning_mode == "mtp":
        return {name: info for name, info in metadata.items() if name.startswith("mtp.")}
    # 未识别的规划模式直接返回空映射。
    return {}


def _resolve_fixed_gpu_slots_per_layer(
        *,
        planning_mode: str,
        top_k: int,
        num_experts: int,
        mtp_reserve_mode: bool,
        target_occupied_gpu_bytes: int,
) -> int | None:
    # 只有 MTP 路径要求固定常驻 slot 数；target 路径交给自动规划。
    if planning_mode != "mtp":
        return None
    # reserve-only 的临时 draft 预算必须固定在 base_8 语义上，供 target 预留共享显存。
    if mtp_reserve_mode:
        return min(num_experts, max(top_k, DEFAULT_MTP_BASE_GPU_SLOTS))
    # 真正的 draft 计划若已拿到 target 占用 hint，就改为按剩余预算动态重算 resident slots。
    if target_occupied_gpu_bytes > 0:
        return None
    # MTP 固定基线是 8，但绝不会小于 top-k，也不会超过总专家数。
    return min(num_experts, max(top_k, DEFAULT_MTP_BASE_GPU_SLOTS))


def _plan_prefill_burst_slots(
        *,
        num_experts: int,
        top_k: int,
        total_gpu_budget_bytes: int,
        expert_bytes_per_slot_all_layers: int,
        expert_bytes_per_expert: int,
) -> tuple[int, int]:
    # -----------------
    # 目标：先预留 prefill burst，再用剩余预算估算 resident slots。
    # -----------------
    # 若总专家数本来就不超过 top-k，则所有专家都可直接常驻 GPU，不需要 burst 池。
    if num_experts <= top_k:
        return min(num_experts, top_k), 0

    # 若预算或尺寸参数无效，则退化为“只保留 top-k 个常驻专家、不启用 burst”。
    if (
            total_gpu_budget_bytes <= 0
            or expert_bytes_per_slot_all_layers <= 0
            or expert_bytes_per_expert <= 0
    ):
        return min(num_experts, top_k), 0

    # -----------------
    # 第一步：预设 burst 目标容量。
    # -----------------
    # prefill burst 默认目标覆盖全量 experts；对真实 Qwen3.5 target 来说通常就是固定 256 slots。
    target_burst_slots = int(num_experts)

    # -----------------
    # 第二步：先整体扣掉 burst 池预算，再看 resident 还能剩多少。
    # -----------------
    # 先整体扣掉“全量 burst 池”预算，再用剩余预算估算常驻 resident slots。
    resident_slots = min(
        num_experts,
        max(
            0,
            total_gpu_budget_bytes - target_burst_slots * expert_bytes_per_expert,
        ) // max(1, expert_bytes_per_slot_all_layers),
    )

    # -----------------
    # 第三步：若 burst 挤掉了 top-k 常驻下限，就收缩 burst 保住 resident。
    # -----------------
    # 若全量 burst 会把 resident 压到 top-k 以下，则收缩 burst，只保留满足 top-k 后还能容纳的部分。
    if resident_slots < top_k:
        resident_slots = min(num_experts, top_k)
        burst_slots = min(
            num_experts,
            max(
                0,
                total_gpu_budget_bytes
                - resident_slots * expert_bytes_per_slot_all_layers,
            ) // max(1, expert_bytes_per_expert),
        )
    else:
        burst_slots = target_burst_slots

    # 返回“最终常驻 resident slot 数 + 最终 burst slot 数”。
    return int(resident_slots), int(burst_slots)


def _extract_expert_layer_prefix(param_name: str) -> str | None:
    # 约定专家参数名里一定会带 `.experts.` 这个分隔标记。
    marker = ".experts."
    # 如果参数名里根本没有专家标记，说明它不是 routed expert 参数，直接返回 None。
    if marker not in param_name:
        return None
    # 以第一次出现 `.experts.` 为界，把“层前缀”和后半段拆开。
    prefix, remainder = param_name.split(marker, 1)
    # 从后半段里取出紧跟在 `.experts.` 后面的那一段，期望它是专家编号。
    expert_id, _, _ = remainder.partition(".")
    # 如果这一段不是纯数字，说明命名不符合 `...experts.<id>...` 约定，返回 None。
    if not expert_id.isdigit():
        return None
    # 返回统一的专家层前缀，例如 `model.layers.0.mlp.experts`。
    return f"{prefix}{marker[:-1]}"


def _extract_expert_param_parts(param_name: str) -> tuple[str, int] | None:
    # 约定 routed expert 参数名里会包含 `.experts.` 这个分隔标记。
    marker = ".experts."
    # 如果参数名不包含该标记，说明它不是专家参数，直接返回 None。
    if marker not in param_name:
        return None
    # 以第一次出现 `.experts.` 为界，拆出层前缀和后半段内容。
    prefix, remainder = param_name.split(marker, 1)
    # 从后半段里取出第一个 `.` 之前的部分，期望它就是专家编号。
    expert_id, _, _ = remainder.partition(".")
    # 如果专家编号不是纯数字，说明命名不符合约定，直接返回 None。
    if not expert_id.isdigit():
        return None
    # 返回“专家层统一前缀 + 整数专家编号”，供后续按层、按专家聚合大小。
    return f"{prefix}{marker[:-1]}", int(expert_id)


def _collect_expert_bytes(
        metadata: dict[str, Any],
) -> tuple[int, int, dict[str, dict[int, int]]]:
    # 统计所有参数总字节量，不区分 dense / expert。
    total_bytes = 0
    # 只统计 routed expert 参数的总字节量。
    expert_bytes_total = 0
    # 按“专家层前缀 -> 专家 id -> 字节数”聚合每个专家的大小。
    layer_expert_bytes: dict[str, dict[int, int]] = {}

    # 遍历 metadata 中的每个参数名及其元信息。
    for name, info in metadata.items():
        # 先把当前参数条目的字节数算出来。
        nbytes = _metadata_nbytes(info)
        # 无论是否是专家参数，都要累计进全模型总字节量。
        total_bytes += nbytes

        # 尝试把参数名解析成“专家层前缀 + 专家编号”。
        expert_parts = _extract_expert_param_parts(name)
        # 不是 routed expert 参数时，跳过 expert 统计，仅保留 total_bytes 的累计结果。
        if expert_parts is None:
            continue

        # 解包出当前参数所属的专家层前缀和专家编号。
        layer_prefix, expert_id = expert_parts

        # routed expert 参数要累计进 expert 总字节量。
        expert_bytes_total += nbytes

        # 确保当前专家层前缀在聚合字典里已经有一层子字典。
        layer_expert_bytes.setdefault(layer_prefix, {})

        # 把当前参数字节数累加到对应“层前缀 + 专家 id”的总大小上。
        layer_expert_bytes[layer_prefix][expert_id] = (
                layer_expert_bytes[layer_prefix].get(expert_id, 0) + nbytes
        )

    # 返回“全参数总大小、全部专家总大小、按层按专家聚合后的大小映射”。
    return total_bytes, expert_bytes_total, layer_expert_bytes


def _metadata_nbytes(info: dict[str, Any]) -> int:
    # 优先读取 safetensors metadata 里直接给出的 data_offsets。
    data_offsets = info.get("data_offsets")

    # 若 data_offsets 是合法的 [start, end] 二元组，就直接用偏移差计算字节数。
    if data_offsets is not None and len(data_offsets) == 2:
        return int(data_offsets[1]) - int(data_offsets[0])

    # 否则退回到根据 shape 和 dtype 估算字节数。
    shape = info.get("shape", ())

    # 读取 dtype 名字，并转成字符串键用于查表。
    dtype = str(info.get("dtype", ""))

    # 根据 shape 计算参数元素总数。
    numel = math.prod(int(dim) for dim in shape)
    # 用元素个数乘以单元素字节数，得到该参数的估算总字节量。
    return numel * _DTYPE_BYTES.get(dtype, 0)


def _get_gpu_budget_bytes(cfie_config: Any) -> int:
    # 如果 CUDA 不可用，则返回 0
    if not torch.cuda.is_available():
        return 0

    total_memory = _get_total_gpu_memory_bytes()
    gpu_util = float(getattr(cfie_config.cache_config, "gpu_memory_utilization", 0.9))
    static_budget, _ = split_gpu_memory_budget(total_memory, gpu_util)
    return static_budget


def _get_gpu_runtime_headroom_bytes(cfie_config: Any) -> int:
    # 如果 CUDA 不可用，则返回 0
    if not torch.cuda.is_available():
        return 0

    total_memory = _get_total_gpu_memory_bytes()
    gpu_util = float(getattr(cfie_config.cache_config, "gpu_memory_utilization", 0.9))
    _, runtime_headroom = split_gpu_memory_budget(total_memory, gpu_util)
    return runtime_headroom


def _get_total_gpu_memory_bytes() -> int:
    # 获取当前使用的 GPU 设备索引。
    device_index = torch.cuda.current_device()
    # 获取当前 GPU 的总显存大小（单位：字节）。
    return int(torch.cuda.get_device_properties(device_index).total_memory)


def _estimate_dynamic_reserve_bytes(
        *,
        planning_mode: str,
        gpu_budget_bytes: int,
) -> int:
    # ----------------- GPU runtime headroom 预留 -----------------
    # target 视角需要给运行时额外开销留出安全余量：
    # - CUDA allocator / workspace 抖动
    # - GPTQ/Marlin 与 MoE 路径的运行时附加张量
    # - profile 阶段统计到的非 KV 峰值波动
    #
    # 若完全不预留，这类“理论上刚好塞满”的配置很容易在真正初始化 KV cache 时
    # 把 available_kv_cache_memory 顶成负数。
    #
    # 这份余量只在 target 规划视角扣一次，避免 target 与 actual draft 双重保守。
    if planning_mode != "target" or gpu_budget_bytes <= 0:
        return 0

    return max(
        DEFAULT_DYNAMIC_RESERVE_BYTES,
        int(gpu_budget_bytes * 0.10),
    )


def _get_total_system_ram_bytes() -> int:
    # 通过 sysconf 读取系统页大小与总页数，估算物理内存总量。
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        num_pages = os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        # 任一系统调用不可用时，退回 0，交由上层走保守路径。
        return 0
    return int(page_size) * int(num_pages)


def _get_available_system_ram_bytes() -> int:
    # Linux 上优先读取 /proc/meminfo 里的 MemAvailable 作为真实可用内存。
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        # /proc 不可访问时，后面退回到物理总内存。
        pass
    # 兜底返回总物理内存，意味着无法区分“已占用”和“可用”的差别。
    return _get_total_system_ram_bytes()


def _estimate_cpu_min_free_bytes(
        cfie_config: Any,
        total_system_bytes: int,
) -> int:
    # 读取 offload_config，优先尊重用户手工配置的主机最小空闲水位。
    offload_config = getattr(cfie_config, "offload_config", None)
    configured_min_free_gb = float(
        getattr(offload_config, "moe_cpu_min_free_gb", 0.0) or 0.0
    )
    if configured_min_free_gb > 0:
        # 显式配置存在时，直接按配置值返回。
        return int(configured_min_free_gb * GiB)

    if total_system_bytes <= 0:
        # 拿不到总物理内存时，退回默认保守值。
        return DEFAULT_CPU_MIN_FREE_BYTES

    # 否则在“绝对下限”和“按总内存比例估算”之间取较大者。
    return max(
        DEFAULT_CPU_MIN_FREE_BYTES,
        int(total_system_bytes * DEFAULT_CPU_MIN_FREE_FRACTION),
    )


def _estimate_cpu_shared_reserve_bytes(cfie_config: Any) -> int:
    # 读取统一 offload 配置；UVA 场景下，CPU offload 也会与 MoE cache 竞争主机内存。
    offload_config = getattr(cfie_config, "offload_config", None)
    uva_config = getattr(offload_config, "uva", None)
    configured_cpu_offload_gb = float(
        getattr(uva_config, "cpu_offload_gb", 0.0) or 0.0
    )
    configured_cpu_offload_bytes = int(configured_cpu_offload_gb * GiB)
    # 最终共享预留量 = 固定共享预留 + UVA 明确要求的 CPU offload 空间。
    return DEFAULT_CPU_SHARED_RESERVE_BYTES + configured_cpu_offload_bytes


def _estimate_shared_gpu_reserve_bytes(
        *,
        cfie_config: Any,
        planning_mode: str,
        mtp_reserve_mode: bool,
        target_occupied_gpu_bytes: int,
) -> int:
    # target 侧需要为 draft 预留共享显存；draft 侧则需要扣减 target 已占据的预算。
    if planning_mode == "target":
        return _estimate_target_shared_gpu_reserve_bytes(cfie_config=cfie_config)
    if planning_mode == "mtp" and not mtp_reserve_mode:
        return max(0, target_occupied_gpu_bytes)
    return 0


def _estimate_target_shared_gpu_reserve_bytes(
        *,
        cfie_config: Any,
) -> int:
    # ----------------- target 侧递归预估 draft reserve -----------------
    # 这里不是在真正加载 MTP 模型，而是在构建 target plan 时，先估算 draft/MTP 至少会吃掉多少 GPU。
    # 该预估必须保持 reserve-only 语义，不能直接等同于最终 actual draft plan。
    # 只有主干 target 规划才需要为同进程并存的 draft/MTP 模型让出 GPU 预算。
    speculative_config = getattr(cfie_config, "speculative_config", None)
    if speculative_config is None:
        return 0

    # 目前只在 Qwen3.5 的 MTP 路径下额外预留这部分 GPU 预算。
    if str(getattr(speculative_config, "method", "")) != "mtp":
        return 0

    # 没有草稿模型配置就无法构造独立的 draft 视角配置，也就无法做额外预算。
    if getattr(speculative_config, "draft_model_config", None) is None:
        return 0

    # 延迟导入，避免在模块导入期引入 spec-decode 方向的循环依赖。
    from cfie.v1.spec_decode.utils import create_vllm_config_for_draft_model

    # reserve 预估阶段会单独派生一份 draft CfieConfig。
    # 这份配置只负责回答“若按 base_8 估算，draft 至少会占多少 GPU”，不会被直接拿去跑模型。
    draft_cfie_config = create_vllm_config_for_draft_model(
        cfie_config,
        additional_config_updates={
            MTP_RESERVE_MODE_KEY: True,
            TARGET_OCCUPIED_GPU_BYTES_KEY: 0,
        },
    )

    # draft reserve 配置在 __post_init__ 时同样会尝试注入 plan；这里优先复用它。
    draft_plan = get_moe_tiered_cache_plan(draft_cfie_config)

    # 若 draft 配置里还没有 plan，则立即补算一次，保证拿到真实的 GPU 占用预算。
    if draft_plan is None:
        draft_plan = build_moe_tiered_cache_plan(draft_cfie_config).to_dict()

    # draft 自身的非专家常驻内容也会与 target 竞争同一张 GPU。
    draft_resident_bytes = int(draft_plan.get("resident_bytes", 0) or 0)

    # draft 为专家常驻/burst 预留的 GPU 预算同样要从 target 可用预算里扣掉。
    draft_gpu_expert_budget_bytes = int(
        draft_plan.get("gpu_expert_budget_bytes", 0) or 0
    )
    # 返回 target 必须让出的总 GPU 预算。
    return max(0, draft_resident_bytes + draft_gpu_expert_budget_bytes)


def _get_target_occupied_gpu_bytes(cfie_config: Any) -> int:
    # actual draft 计划会从 additional_config 里读取 target 已经占据的 GPU 预算。
    # 这个值由 target plan 构建完成后回写进去，供 draft 重新规划 resident slots。
    additional_config = getattr(cfie_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return 0
    return int(additional_config.get(TARGET_OCCUPIED_GPU_BYTES_KEY, 0) or 0)


def _is_mtp_reserve_mode(cfie_config: Any) -> bool:
    # reserve-only 标记只在 target 递归预估 draft budget 时存在。
    # 真正要加载 MTP 模型的 draft 配置不会带这个标记。
    additional_config = getattr(cfie_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return False
    return bool(additional_config.get(MTP_RESERVE_MODE_KEY, False))


def _get_plan_gpu_occupied_bytes(plan: MoeTieredCachePlan) -> int:
    # 当前计划真正占据的 GPU 预算 = 非专家常驻内容 + 专家相关 GPU 预算。
    # 这个数字不会区分 target/draft；写回 additional_config 时由调用方决定是否采用。
    return max(0, int(plan.resident_bytes) + int(plan.gpu_expert_budget_bytes))


def _maybe_record_target_occupied_gpu_bytes(
        *,
        cfie_config: Any,
        additional_config: dict[str, Any],
        plan: MoeTieredCachePlan,
) -> None:
    # 只有 target 计划才需要把“已占 GPU 预算”回写给后续真正创建的 draft 配置。
    # reserve-only 的临时 draft 配置不会走这条分支。
    model_config = getattr(cfie_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if _resolve_qwen35_moe_planning_mode(hf_config) != "target":
        return
    additional_config[TARGET_OCCUPIED_GPU_BYTES_KEY] = _get_plan_gpu_occupied_bytes(
        plan
    )


def _estimate_cpu_cache_budget_bytes(
        cfie_config: Any,
        available_system_bytes: int,
        total_system_bytes: int,
        budget_fraction: float = DEFAULT_CPU_CACHE_BUDGET_FRACTION,
) -> int:
    # 没有可用内存时，不给 CPU expert cache 分配预算。
    if available_system_bytes <= 0:
        return 0

    # 先读取用户可能配置的硬上限。
    offload_config = getattr(cfie_config, "offload_config", None)
    hard_cap_gb = float(getattr(offload_config, "moe_cpu_budget_gb", 0.0) or 0.0)
    hard_cap_bytes = int(hard_cap_gb * GiB) if hard_cap_gb > 0 else 0

    # 扣掉最小空闲水位和共享预留量后，得到 MoE cache 可消费的预算窗口。
    min_free_bytes = _estimate_cpu_min_free_bytes(
        cfie_config=cfie_config,
        total_system_bytes=total_system_bytes,
    )
    shared_reserve_bytes = _estimate_cpu_shared_reserve_bytes(cfie_config)
    budget_window_bytes = max(
        0,
        available_system_bytes - min_free_bytes - shared_reserve_bytes,
    )
    # budget_fraction 会被夹在 [0, 1] 区间，避免异常值放大或变负。
    budget_fraction = min(1.0, max(0.0, float(budget_fraction)))
    budget_bytes = int(budget_window_bytes * budget_fraction)
    if hard_cap_bytes > 0:
        # 若用户还配置了硬上限，则再做一次裁剪。
        budget_bytes = min(budget_bytes, hard_cap_bytes)
    # 最终返回非负预算。
    return max(0, budget_bytes)


def _estimate_minimal_cpu_staging_bytes(
        *,
        cfie_config: Any,
        available_system_bytes: int,
        remaining_expert_bytes: int,
        expert_bytes_per_slot_all_layers: int,
        remaining_cpu_candidates: int,
) -> int:
    # 只要系统可用内存、剩余专家字节、单 slot 专家体积或 CPU 候选专家数任一无效，就不需要 staging。
    if (
            available_system_bytes <= 0
            or remaining_expert_bytes <= 0
            or expert_bytes_per_slot_all_layers <= 0
            or remaining_cpu_candidates <= 0
    ):
        # 返回 0，表示当前场景下无需为 CPU staging 单独预留最小预算。
        return 0

    # minimal staging 只保守地按“剩余专家总字节”和“2 个跨层 slot”中的较小值来估算。
    required_staging_bytes = min(
        remaining_expert_bytes,
        2 * expert_bytes_per_slot_all_layers,
    )

    # 返回 minimal staging 所需的保守预算字节数，供后续 planner 至少保证这一块能跑通。
    return required_staging_bytes


def _estimate_hybrid_cache_bytes(cfie_config: Any) -> tuple[int, int]:
    # 获取模型配置对象
    model_config = cfie_config.model_config

    # 获取 Hugging Face 原始配置，用于区分主干 target 与 MTP draft 规划视角。
    hf_config = model_config.hf_config

    # 获取 Hugging Face 文本模型配置
    hf_text_config = model_config.hf_text_config

    # 获取调度配置，不存在时默认为 None
    scheduler_config = getattr(cfie_config, "scheduler_config", None)

    # 获取最大序列数，至少为 1
    max_num_seqs = max(1, int(getattr(scheduler_config, "max_num_seqs", 1)))

    # 获取模型最大长度
    max_model_len = int(getattr(model_config, "max_model_len", 0))

    # 如果模型最大长度无效，则回退到配置中的最大位置编码长度
    if max_model_len <= 0:
        max_model_len = int(getattr(hf_text_config, "max_position_embeddings", 0))

    # 解析当前规划视角是主干 target 还是 MTP draft。
    planning_mode = _resolve_qwen35_moe_planning_mode(hf_config)

    # Qwen3.5 MTP 草稿层固定按 full_attention 构造，不读取主干 text_config.layer_types。
    # 因此这里必须单独按 mtp_num_hidden_layers 估算，否则会错误复用主干 hybrid 布局。
    if planning_mode == "mtp":
        # MTP 路径的 full-attention 层数直接取 mtp_num_hidden_layers。
        num_full_layers = int(getattr(hf_text_config, "mtp_num_hidden_layers", 0))
        # MTP 草稿层当前实现中不包含 linear_attention 层。
        num_linear_layers = 0
    else:
        # 获取各层类型列表；如果未配置，则默认所有主干层都是 full_attention。
        layer_types = list(
            getattr(hf_text_config, "layer_types", [])
            or ["full_attention"] * int(getattr(hf_text_config, "num_hidden_layers", 0))
        )

        # 统计主干中的 full_attention 层数。
        num_full_layers = sum(
            1 for layer_type in layer_types if layer_type == "full_attention"
        )

        # 统计主干中的 linear_attention 层数。
        num_linear_layers = sum(
            1 for layer_type in layer_types if layer_type == "linear_attention"
        )

    # 获取 KV cache 每个元素占用的字节数
    kv_bytes_per_elem = _resolve_kv_elem_size(cfie_config)

    # 估算 full_attention 对应的 KV cache 总字节数
    kv_bytes = (
            max_num_seqs
            * max_model_len
            * num_full_layers
            * int(getattr(hf_text_config, "num_key_value_heads", 0))
            * int(getattr(hf_text_config, "head_dim", 0))
            * 2
            * kv_bytes_per_elem
    )

    # 初始化 linear attention 状态缓存字节数
    linear_state_bytes = 0

    # 仅当存在 linear_attention 层时才估算其状态缓存
    if num_linear_layers > 0:
        # 获取卷积缓存中每个元素占用的字节数
        conv_bytes = _resolve_mamba_cache_elem_size(cfie_config)

        # 获取 SSM 状态中每个元素占用的字节数
        ssm_bytes = _resolve_mamba_ssm_elem_size(cfie_config)

        # 获取线性卷积核尺寸，默认值为 4
        conv_kernel = int(getattr(hf_text_config, "linear_conv_kernel_dim", 4))

        # 获取 speculative decoding 的 token 数，不存在时为 0
        num_spec = (
            int(cfie_config.speculative_config.num_speculative_tokens)
            if cfie_config.speculative_config is not None
            else 0
        )

        # 计算卷积状态的总维度
        conv_dim = (
                int(getattr(hf_text_config, "linear_key_head_dim", 0))
                * int(getattr(hf_text_config, "linear_num_key_heads", 0))
                * 2
                + int(getattr(hf_text_config, "linear_value_head_dim", 0))
                * int(getattr(hf_text_config, "linear_num_value_heads", 0))
        )

        # 计算时序状态元素总数 v_heads * v_head_dim * k_head_dim
        temporal_elems = (
                int(getattr(hf_text_config, "linear_num_value_heads", 0))
                * int(getattr(hf_text_config, "linear_value_head_dim", 0))
                * int(getattr(hf_text_config, "linear_key_head_dim", 0))
        )

        # 计算卷积缓存元素总数
        conv_elems = (conv_kernel - 1 + num_spec) * conv_dim

        # 估算 linear attention 状态缓存总字节数
        linear_state_bytes = max_num_seqs * num_linear_layers * (
                conv_elems * conv_bytes + temporal_elems * ssm_bytes
        )

    # 返回 KV cache 字节数和 linear attention 状态缓存字节数
    return int(kv_bytes), int(linear_state_bytes)


def _resolve_kv_elem_size(cfie_config: Any) -> int:
    # KV cache 使用 fp8 时单元素固定为 1 字节，否则退回模型 dtype 的元素大小。
    cache_dtype = str(getattr(cfie_config.cache_config, "cache_dtype", "auto"))
    if cache_dtype.startswith("fp8"):
        return 1
    return _model_dtype_elem_size(cfie_config.model_config.dtype)


def _resolve_mamba_cache_elem_size(cfie_config: Any) -> int:
    # 线性注意力 / mamba 的 cache dtype 支持显式覆盖，否则默认跟随模型 dtype。
    cache_dtype = str(getattr(cfie_config.cache_config, "mamba_cache_dtype", "auto"))
    if cache_dtype == "float32":
        return 4
    if cache_dtype == "float16":
        return 2
    return _model_dtype_elem_size(cfie_config.model_config.dtype)  # 默认


def _resolve_mamba_ssm_elem_size(cfie_config: Any) -> int:
    # SSM state 的 dtype 优先级高于普通 mamba cache，需要单独解析。
    cache_dtype = str(getattr(cfie_config.cache_config, "mamba_ssm_cache_dtype", "auto"))
    if cache_dtype == "float32":  # 默认
        return 4
    if cache_dtype == "float16":
        return 2
    hf_text_config = cfie_config.model_config.hf_text_config
    model_declared_dtype = str(getattr(hf_text_config, "mamba_ssm_dtype", "auto"))
    if model_declared_dtype == "float32":
        return 4
    if model_declared_dtype == "float16":
        return 2
    return _resolve_mamba_cache_elem_size(cfie_config)


def _model_dtype_elem_size(dtype: torch.dtype | str) -> int:
    # torch.dtype 直接通过临时张量读取元素大小。
    if isinstance(dtype, torch.dtype):
        return torch.empty((), dtype=dtype).element_size()
    # 字符串 dtype 走启发式匹配。
    dtype_str = str(dtype).lower()
    if "bfloat16" in dtype_str or "float16" in dtype_str or dtype_str == "half":
        return 2
    if "float32" in dtype_str or dtype_str == "float":
        return 4
    # 兜底按 2 字节处理，匹配当前主流 BF16/F16 推理默认值。
    return 2
