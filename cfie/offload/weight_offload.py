"""Tiered MoE expert cache controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from cfie import _custom_ops as ops
from cfie.logger import init_logger
from cfie.model_executor.layers.fused_moe.layer import FusedMoE
from cfie.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    convert_to_unquantized_kernel_format,
)
from cfie.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from cfie.model_executor.layers.quantization.gptq_marlin import GPTQMarlinMoEMethod
from cfie.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_moe_permute_scales,
)
from cfie.offload.cpu_backend import ExpertBundle, bundle_nbytes
from cfie.offload.nvme_backend import SafetensorExpertStore
from cfie.offload.policy import PLAN_KEY, get_moe_tiered_cache_plan
from cfie.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)

DEFAULT_PREFILL_BURST_MIN_TOKENS = 8
DEFAULT_PREFILL_BURST_TOKENS_PER_GPU_SLOT = 4


@dataclass(slots=True)
class _RawExpertWeights:
    # 量化专家在 CPU staging/raw buffer 中重新拼装后的 w13 压缩权重。
    w13_qweight: torch.Tensor
    # 量化专家在 CPU staging/raw buffer 中重新拼装后的 w2 压缩权重。
    w2_qweight: torch.Tensor
    # 量化专家在 CPU staging/raw buffer 中重新拼装后的 w13 scale。
    w13_scales: torch.Tensor
    # 量化专家在 CPU staging/raw buffer 中重新拼装后的 w2 scale。
    w2_scales: torch.Tensor
    # 量化专家在 CPU staging/raw buffer 中重新拼装后的 w13 qzeros。
    w13_qzeros: torch.Tensor
    # 量化专家在 CPU staging/raw buffer 中重新拼装后的 w2 qzeros。
    w2_qzeros: torch.Tensor
    # desc_act=True 时，gate/up 路径还需要保留排序前的 g_idx。
    w13_g_idx: torch.Tensor | None = None
    # desc_act=True 时，down_proj 路径还需要保留排序前的 g_idx。
    w2_g_idx: torch.Tensor | None = None


@dataclass(slots=True)
class _RawUnquantizedExpertWeights:
    # 非量化专家在 CPU staging/raw buffer 中重新拼装后的合并 w13 权重。
    w13_weight: torch.Tensor
    # 非量化专家在 CPU staging/raw buffer 中重新拼装后的 w2 权重。
    w2_weight: torch.Tensor


@dataclass(slots=True)
class _PrefillBurstExecutionStats:
    # 本次 burst 执行直接命中 resident GPU experts 的次数。
    resident_hits: int = 0
    # 本次 burst 执行命中 CPU static experts 的次数。
    cpu_hits: int = 0
    # 本次 burst 执行需要从 NVMe 拉取 experts 的次数。
    nvme_loads: int = 0


class _PrefillBurstExecutionLayer:
    """绑定到共享 prefill burst 临时张量上的轻量代理层。"""

    def __init__(
            self,
            base_layer: FusedMoE,
            target: Any,
            expert_map: torch.Tensor,
            num_slots: int,
    ) -> None:
        # ------------------------------- 保存共享 burst pool 对象并完成首次绑定 -------------------------------
        # 记录共享 burst pool 本体，后续执行时会直接从这里读取临时执行张量。
        self._target = target

        # 使用当前基础层、专家映射表和槽位数完成代理层的首次绑定。
        self.bind(base_layer, expert_map, num_slots)

    def bind(
            self,
            base_layer: FusedMoE,
            expert_map: torch.Tensor,
            num_slots: int,
    ) -> None:
        # ------------------------------- 绑定基础层与当前 burst expert 映射关系 -------------------------------
        # 保存真实的基础 FusedMoE 层，后续未显式覆盖的属性会统一透传给它。
        self._base_layer = base_layer

        # 使用当前 burst pool 的 expert 映射表覆盖原始层上的 resident expert 映射。
        self._expert_map = expert_map

        # 将当前代理层的本地专家数量改写为 burst pool 的临时槽位数。
        self.local_num_experts = num_slots

        # 保持全局专家数量与原始层一致，确保 router 和 top-k expert id 的语义不变。
        self.global_num_experts = base_layer.global_num_experts

        # ------------------------------- 在量化路径下把执行权重张量重定向到 burst pool -------------------------------
        # 当共享目标对象上存在量化权重张量时，说明当前工作在量化执行路径下。
        if hasattr(self._target, "w13_qweight"):
            # 将 w13 量化权重重定向到 burst pool 的临时张量。
            self.w13_qweight = self._target.w13_qweight

            # 将 w2 量化权重重定向到 burst pool 的临时张量。
            self.w2_qweight = self._target.w2_qweight

            # 将 w13 的 scale 张量重定向到 burst pool 的临时张量。
            self.w13_scales = self._target.w13_scales

            # 将 w2 的 scale 张量重定向到 burst pool 的临时张量。
            self.w2_scales = self._target.w2_scales

            # 将 w13 的 qzeros 张量重定向到 burst pool 的临时张量。
            self.w13_qzeros = self._target.w13_qzeros

            # 将 w2 的 qzeros 张量重定向到 burst pool 的临时张量。
            self.w2_qzeros = self._target.w2_qzeros

            # 将 w13 的 g_idx 张量重定向到 burst pool 的临时张量。
            self.w13_g_idx = self._target.w13_g_idx

            # 将 w2 的 g_idx 张量重定向到 burst pool 的临时张量。
            self.w2_g_idx = self._target.w2_g_idx

            # 将 w13 的 g_idx_sort_indices 张量重定向到 burst pool 的临时张量。
            self.w13_g_idx_sort_indices = self._target.w13_g_idx_sort_indices

            # 将 w2 的 g_idx_sort_indices 张量重定向到 burst pool 的临时张量。
            self.w2_g_idx_sort_indices = self._target.w2_g_idx_sort_indices

            # 某些通用执行路径会通过 w13_weight 访问权重，这里把它别名到量化权重张量上。
            self.w13_weight = self._target.w13_qweight

            # 某些通用执行路径会通过 w2_weight 访问权重，这里把它别名到量化权重张量上。
            self.w2_weight = self._target.w2_qweight
        else:
            # ------------------------------- 在非量化路径下把执行权重张量重定向到 burst pool -------------------------------
            # 将 w13 dense 权重重定向到 burst pool 的临时张量。
            self.w13_weight = self._target.w13_weight

            # 将 w2 dense 权重重定向到 burst pool 的临时张量。
            self.w2_weight = self._target.w2_weight

    @property
    def expert_map(self) -> torch.Tensor:
        # ------------------------------- 返回当前 burst pool 使用的临时 expert 映射表 -------------------------------
        # 运行时 kernel 读取的 expert_map 应当来自当前 burst pool 的临时映射表。
        return self._expert_map

    def __getattr__(self, name: str) -> Any:
        # ------------------------------- 将未显式覆盖的属性统一透传给原始 FusedMoE 层 -------------------------------
        # 当代理层自身没有该属性时，回退到原始基础层上继续查找。
        return getattr(self._base_layer, name)


class SharedPrefillBurstPool:
    """用于大 prefill MoE batch 的共享单层临时执行缓冲池。"""

    def __init__(self, template_layer: FusedMoE, num_slots: int) -> None:
        # ------------------------------- 校验 burst pool 槽位数并初始化基础元数据 -------------------------------
        # 共享 prefill burst pool 至少需要 1 个临时槽位，否则没有实际意义。
        if num_slots <= 0:
            raise ValueError("Shared prefill burst pool requires a positive slot count")

        # 记录当前 burst pool 可容纳的临时槽位数量。
        self.num_slots = int(num_slots)

        # 记录模板层对应的全局专家总数，后续复用同一套全局 expert id 语义。
        self.global_num_experts = template_layer.global_num_experts

        # 记录模板层名称，主要用于日志输出与调试定位。
        self.layer_name = template_layer.layer_name

        # 共享池同一时刻只允许一个 layer 或 request 使用，防止临时张量被并发覆盖。
        self._busy = False

        # ------------------------------- 按模板层的量化模式分配临时执行张量 -------------------------------
        # 当模板层采用 GPTQ Marlin 量化路径时，需要为量化权重、scale 与辅助索引都准备 burst 张量。
        if isinstance(template_layer.quant_method, GPTQMarlinMoEMethod):
            # 记录当前 burst pool 工作在 GPTQ Marlin 模式下。
            self._mode = "gptq_marlin"

            # 为 w13 量化权重分配按 slot 组织的临时张量。
            self.w13_qweight = torch.empty(
                (self.num_slots, *template_layer.w13_qweight.shape[1:]),
                dtype=template_layer.w13_qweight.dtype,
                device=template_layer.w13_qweight.device,
            )

            # 为 w2 量化权重分配按 slot 组织的临时张量。
            self.w2_qweight = torch.empty(
                (self.num_slots, *template_layer.w2_qweight.shape[1:]),
                dtype=template_layer.w2_qweight.dtype,
                device=template_layer.w2_qweight.device,
            )

            # 为 w13 的 scale 张量分配按 slot 组织的临时张量。
            self.w13_scales = torch.empty(
                (self.num_slots, *template_layer.w13_scales.shape[1:]),
                dtype=template_layer.w13_scales.dtype,
                device=template_layer.w13_scales.device,
            )

            # 为 w2 的 scale 张量分配按 slot 组织的临时张量。
            self.w2_scales = torch.empty(
                (self.num_slots, *template_layer.w2_scales.shape[1:]),
                dtype=template_layer.w2_scales.dtype,
                device=template_layer.w2_scales.device,
            )

            # 为 w13 的 qzeros 张量分配按 slot 组织的临时张量。
            self.w13_qzeros = torch.empty(
                (self.num_slots, *template_layer.w13_qzeros.shape[1:]),
                dtype=template_layer.w13_qzeros.dtype,
                device=template_layer.w13_qzeros.device,
            )

            # 为 w2 的 qzeros 张量分配按 slot 组织的临时张量。
            self.w2_qzeros = torch.empty(
                (self.num_slots, *template_layer.w2_qzeros.shape[1:]),
                dtype=template_layer.w2_qzeros.dtype,
                device=template_layer.w2_qzeros.device,
            )

            # 为 w13 的 g_idx 张量分配按 slot 组织的临时张量。
            self.w13_g_idx = torch.empty(
                (self.num_slots, *template_layer.w13_g_idx.shape[1:]),
                dtype=template_layer.w13_g_idx.dtype,
                device=template_layer.w13_g_idx.device,
            )

            # 为 w2 的 g_idx 张量分配按 slot 组织的临时张量。
            self.w2_g_idx = torch.empty(
                (self.num_slots, *template_layer.w2_g_idx.shape[1:]),
                dtype=template_layer.w2_g_idx.dtype,
                device=template_layer.w2_g_idx.device,
            )

            # 为 w13 的 g_idx_sort_indices 张量分配按 slot 组织的临时张量。
            self.w13_g_idx_sort_indices = torch.empty(
                (self.num_slots, *template_layer.w13_g_idx_sort_indices.shape[1:]),
                dtype=template_layer.w13_g_idx_sort_indices.dtype,
                device=template_layer.w13_g_idx_sort_indices.device,
            )

            # 为 w2 的 g_idx_sort_indices 张量分配按 slot 组织的临时张量。
            self.w2_g_idx_sort_indices = torch.empty(
                (self.num_slots, *template_layer.w2_g_idx_sort_indices.shape[1:]),
                dtype=template_layer.w2_g_idx_sort_indices.dtype,
                device=template_layer.w2_g_idx_sort_indices.device,
            )

            # 记录 expert 映射表应放置的设备，沿用量化权重所在设备。
            expert_map_device = template_layer.w13_qweight.device

        # 当模板层采用非量化路径时，只需要准备运行时直接消费的 w13 和 w2 权重张量。
        elif isinstance(template_layer.quant_method, UnquantizedFusedMoEMethod):
            # 记录当前 burst pool 工作在非量化模式下。
            self._mode = "unquantized"

            # 为 w13 权重分配按 slot 组织的临时张量。
            self.w13_weight = torch.empty(
                (self.num_slots, *template_layer.w13_weight.shape[1:]),
                dtype=template_layer.w13_weight.dtype,
                device=template_layer.w13_weight.device,
            )

            # 为 w2 权重分配按 slot 组织的临时张量。
            self.w2_weight = torch.empty(
                (self.num_slots, *template_layer.w2_weight.shape[1:]),
                dtype=template_layer.w2_weight.dtype,
                device=template_layer.w2_weight.device,
            )

            # 记录 expert 映射表应放置的设备，沿用非量化权重所在设备。
            expert_map_device = template_layer.w13_weight.device
        else:
            # 当前共享 burst pool 仅支持 GPTQ Marlin 和非量化两类 FusedMoE 路径。
            raise TypeError(
                "Shared prefill burst pool only supports GPTQMarlinMoEMethod "
                "and UnquantizedFusedMoEMethod"
            )

        # ------------------------------- 构造全局 expert 到 burst slot 的映射表与执行代理层 -------------------------------
        # 为当前共享 burst pool 准备一张“全局 expert id -> burst slot id”的映射表。
        self._expert_map = torch.full(
            (self.global_num_experts,),
            -1,
            dtype=torch.int32,
            device=expert_map_device,
        )

        # 基于模板层构造一个轻量级执行代理层，把 kernel 入口重定向到 burst pool 上。
        self._execution_layer = _PrefillBurstExecutionLayer(
            base_layer=template_layer,
            target=self,
            expert_map=self._expert_map,
            num_slots=self.num_slots,
        )

        # ------------------------------- 打印 burst pool 初始化完成日志 -------------------------------
        # 输出当前共享 burst pool 的层名、模式、槽位数与显存占用大小。
        logger.info(
            "Initialized shared prefill burst pool: layer=%s mode=%s slots=%d bytes=%.2f MiB",
            template_layer.layer_name,
            self._mode,
            self.num_slots,
            self.nbytes / (1 << 20),
        )

    @property
    def nbytes(self) -> int:
        # ------------------------------- 统计当前 burst pool 的总字节占用 -------------------------------
        # 当工作在 GPTQ Marlin 模式时，需要把所有临时量化张量都纳入统计。
        if self._mode == "gptq_marlin":
            tensors = (
                self.w13_qweight,
                self.w2_qweight,
                self.w13_scales,
                self.w2_scales,
                self.w13_qzeros,
                self.w2_qzeros,
                self.w13_g_idx,
                self.w2_g_idx,
                self.w13_g_idx_sort_indices,
                self.w2_g_idx_sort_indices,
            )
        else:
            # 当工作在非量化模式时，只统计 w13 和 w2 两块临时权重张量。
            tensors = (self.w13_weight, self.w2_weight)

        # 对所有临时张量逐个按“元素数 * 单元素字节数”求和，得到总占用字节数。
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def supports_layer(self, layer: FusedMoE) -> bool:
        # ------------------------------- 判断当前共享 burst pool 是否可被指定层复用 -------------------------------
        # 当目标层的全局专家总数与当前 burst pool 不一致时，说明 top-k expert id 语义不同，不能复用。
        if layer.global_num_experts != self.global_num_experts:
            return False

        # 当工作在 GPTQ Marlin 模式时，需要额外校验所有核心量化张量的形状兼容性。
        if self._mode == "gptq_marlin":
            return (
                    isinstance(layer.quant_method, GPTQMarlinMoEMethod)
                    and layer.w13_qweight.shape[1:] == self.w13_qweight.shape[1:]
                    and layer.w2_qweight.shape[1:] == self.w2_qweight.shape[1:]
                    and layer.w13_scales.shape[1:] == self.w13_scales.shape[1:]
                    and layer.w2_scales.shape[1:] == self.w2_scales.shape[1:]
                    and layer.w13_qzeros.shape[1:] == self.w13_qzeros.shape[1:]
                    and layer.w2_qzeros.shape[1:] == self.w2_qzeros.shape[1:]
            )

        # 当工作在非量化模式时，仅需校验 w13 和 w2 权重张量的形状兼容性。
        return (
                isinstance(layer.quant_method, UnquantizedFusedMoEMethod)
                and layer.w13_weight.shape[1:] == self.w13_weight.shape[1:]
                and layer.w2_weight.shape[1:] == self.w2_weight.shape[1:]
        )

    def execute(
            self,
            controller: LayerTieredExpertCacheController,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # ------------------------------- 校验共享 burst pool 当前未被占用 -------------------------------
        # 共享 burst pool 被设计为串行使用，若当前已被占用则直接报错，防止临时张量被并发覆盖。
        if self._busy:
            raise RuntimeError(
                f"{controller.layer_key}: shared prefill burst pool is already in use"
            )

        # ------------------------------- 提取本次请求涉及的唯一 expert 集合并校验槽位容量 -------------------------------
        # 从当前 top-k expert id 中提取去重后的唯一 expert 集合。
        requested = controller._unique_experts(topk_ids)

        # 当本次请求的唯一 expert 数超过 burst pool 槽位容量时，无法走 burst 执行路径。
        if len(requested) > self.num_slots:
            raise RuntimeError(
                f"{controller.layer_key}: requested {len(requested)} experts but only "
                f"{self.num_slots} burst slots are available"
            )

        # ------------------------------- 装填 burst pool 并执行当前 prefill 请求 -------------------------------
        # 标记当前共享池进入占用状态。
        self._busy = True
        try:
            # 每次执行前先清空上一轮残留的“全局 expert -> burst slot”映射。
            self._expert_map.fill_(-1)

            # 按当前请求的 expert 集合，把 resident、CPU 或 NVMe 中的数据填充到 burst pool。
            stats = controller._populate_prefill_burst_pool(self, requested)

            # 依次登记当前请求中每个全局 expert 对应的 burst slot 编号。
            for slot, expert_id in enumerate(requested):
                self._expert_map[expert_id] = slot

            # 将轻量代理执行层重新绑定到当前控制器对应的真实 layer 上。
            self._execution_layer.bind(
                base_layer=controller.layer,
                expert_map=self._expert_map,
                num_slots=self.num_slots,
            )

            # 打印当前 burst 执行的数据来源命中情况，包括 resident 命中、CPU 命中与 NVMe 加载次数。
            logger.info(
                "Tiered MoE prefill burst execution: layer=%s unique=%d slots=%d "
                "resident_hits=%d cpu_hits=%d nvme_loads=%d",
                controller.layer_key,
                len(requested),
                self.num_slots,
                stats.resident_hits,
                stats.cpu_hits,
                stats.nvme_loads,
            )

            # 通过当前控制器的量化方法，把执行入口直接重定向到 burst proxy layer 上完成实际计算。
            return controller.quant_method.apply(
                layer=self._execution_layer,
                x=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
            )
        finally:
            # ------------------------------- 释放共享 burst pool 的占用标记 -------------------------------
            # 无论本次执行成功还是失败，都必须清除 busy 标记，允许后续请求继续复用该共享池。
            self._busy = False


class LayerTieredExpertCacheController:
    """Per-layer GPU/CPU tiered cache controller for routed experts."""

    def __init__(
            self,
            layer: FusedMoE,
            plan: dict[str, Any],
            expert_store: SafetensorExpertStore,
            prefill_burst_pool: SharedPrefillBurstPool | None = None,
    ) -> None:
        # ------------------------------- 校验当前层是否已建立全局 expert 到本地 slot 的映射 -------------------------------
        # Tiered MoE cache 依赖 FusedMoE 预先构造好的全局 expert 到本地 slot 映射表。
        if layer._expert_map is None:
            raise ValueError("Tiered MoE cache requires a global-to-local expert map")

        # ------------------------------- 保存控制器绑定的核心对象与基础元数据 -------------------------------
        # 保存当前控制器绑定的 FusedMoE 层对象。
        self.layer = layer

        # 保存启动期规划器生成的计划字典。
        self.plan = plan

        # 保存专家存储对象，后续冷专家权重都通过它按需读取。
        self.expert_store = expert_store

        # 保存共享 prefill burst pool；若存在，则可用于大 prefill batch 的临时执行路径。
        self.prefill_burst_pool = prefill_burst_pool

        # 保存当前层的层级主键，供本地存储查询与日志输出使用。
        self.layer_key = layer.layer_name

        # 保存当前层对应的量化方法对象，后续用于决定权重写回与执行路径。
        self.quant_method = layer.quant_method

        # resident GPU slot 数等于当前 FusedMoE 层在 CFIE 路径下实际创建的本地 expert 数。
        self.num_slots = layer.local_num_experts

        # 初始化逻辑时间步计数器，后续用于近似 LRU 驱逐时间戳。
        self._step = 0

        # 初始化每个全局 expert 的访问计数器。
        self._access_count = [0] * layer.global_num_experts

        # 初始化每个全局 expert 最近一次命中的逻辑时间记录。
        self._last_used_step = [0] * layer.global_num_experts

        # 初始化 resident GPU slot 到全局 expert 的反向映射表。
        self._slot_to_global = [-1] * self.num_slots

        # ------------------------------- 初始化运行期统计指标 -------------------------------
        # 记录控制器累计执行过的专家加载次数。
        self._total_loads = 0

        # 记录命中 CPU 常驻缓存的次数。
        self._cpu_hits = 0

        # 记录实际从冷存储加载专家的次数。
        self._nvme_loads = 0

        # 记录发生 expert 驱逐的次数。
        self._evictions = 0

        # 当前控制器使用的权重模式标识，稍后会根据 quant_method 进行赋值。
        self._mode: str

        # ------------------------------- 初始化 CPU 侧缓存与缓冲区状态 -------------------------------
        # 若平台支持 pinned memory，则优先使用 pinned CPU memory 以加快 H2D 传输。
        self._use_pinned_cpu = is_pin_memory_available()

        # 保存已物化到 CPU 常驻内存中的静态 expert bundle。
        self._cpu_static_bundles: dict[int, ExpertBundle] = {}

        # 保留旧版计划兼容字段；当前主线一般不再依赖单独的 staging bundle。
        self._cpu_stage_bundle: ExpertBundle | None = None

        # 保存计划中指定的 CPU 常驻 expert 集合。
        self._cpu_static_experts: frozenset[int] = frozenset()

        # 统计当前控制器持有的全部 CPU buffer 总字节数。
        self._cpu_buffer_bytes = 0

        # 量化路径下用于在 CPU 侧暂存并重组原始专家权重的 raw buffer。
        self._cpu_quantized_raw_buffer: _RawExpertWeights | None = None

        # 非量化路径下用于在 CPU 侧暂存并重组原始专家权重的 raw buffer。
        self._cpu_unquantized_raw_buffer: _RawUnquantizedExpertWeights | None = None

        # ------------------------------- 计算 prefill burst 的最小 token 触发阈值 -------------------------------
        # 优先读取计划中显式配置的 burst 最小 token 阈值。
        configured_burst_min_tokens = int(self.plan.get("prefill_burst_min_tokens", 0))

        # 当计划未显式配置时，按默认经验公式计算 burst 最小 token 阈值。
        self._prefill_burst_min_tokens = (
            configured_burst_min_tokens
            if configured_burst_min_tokens > 0
            else max(
                DEFAULT_PREFILL_BURST_MIN_TOKENS,
                self.num_slots * DEFAULT_PREFILL_BURST_TOKENS_PER_GPU_SLOT,
            )
        )

        # ------------------------------- 根据量化方法识别当前层的权重模式并校验约束 -------------------------------
        # 当当前层采用 GPTQ Marlin 量化方法时，进入量化 expert 动态加载路径。
        if isinstance(layer.quant_method, GPTQMarlinMoEMethod):
            # 当前实现假设输入激活仍为标准 bf16 或 fp16，不支持额外 input_dtype 改写。
            if layer.quant_method.input_dtype is not None:
                raise ValueError(
                    "Tiered MoE cache currently expects standard bf16/fp16 activations"
                )

            # 记录当前控制器工作在 GPTQ Marlin 模式下。
            self._mode = "gptq_marlin"

            # 记录 GPTQ 是否启用了 desc_act 配置，后续执行路径可按该标志决定处理分支。
            self._gptq_desc_act = bool(layer.quant_method.quant_config.desc_act)

            # 记录当前层量化权重所在的目标 GPU 设备。
            self.device = layer.w13_qweight.device

            # 保存量化权重的 pack_factor 参数。
            self.pack_factor = layer.quant_method.quant_config.pack_factor

            # 保存量化类型对应的 bit 数。
            self.num_bits = layer.quant_method.quant_config.quant_type.size_bits

            # 保存量化 group size 参数。
            self.group_size = layer.quant_method.quant_config.group_size

            # 标记当前路径不是 8bit 激活路径。
            self.is_a_8bit = False

        # 当当前层采用非量化 FusedMoE 方法时，进入非量化动态加载路径。
        elif isinstance(layer.quant_method, UnquantizedFusedMoEMethod):
            # 当前非量化动态加载路径不支持带 bias 的 MoE。
            if layer.quant_method.moe.has_bias:
                raise ValueError(
                    "Tiered MoE cache currently does not support biased unquantized MoE"
                )

            # 当前非量化路径支持 Triton、CUDA ATen 与 PyTorch 回退 backend。
            if layer.quant_method.unquantized_backend not in (
                    UnquantizedMoeBackend.TRITON,
                    UnquantizedMoeBackend.CUDA_ATEN,
                    UnquantizedMoeBackend.TORCH,
            ):
                raise TypeError(
                    "Tiered MoE cache currently only supports TRITON, "
                    "CUDA_ATEN, and TORCH fallback unquantized MoE"
                )

            # 记录当前控制器工作在非量化模式下。
            self._mode = "unquantized"

            # 非量化路径下不涉及 GPTQ desc_act，显式记录为 False。
            self._gptq_desc_act = False

            # 记录当前层非量化权重所在的目标 GPU 设备。
            self.device = layer.w13_weight.device
        else:
            # 当前控制器仅支持 GPTQ Marlin 和非量化 FusedMoE 两类路径。
            raise TypeError(
                "Tiered MoE cache currently supports GPTQMarlinMoEMethod "
                "and UnquantizedFusedMoEMethod only"
            )

        # ------------------------------- 根据 layer._expert_map 反推 resident GPU slot 的初始专家布局 -------------------------------
        # 遍历所有全局 expert，通过 layer._expert_map 反推出当前 resident GPU slot 中实际驻留的是哪个 expert。
        for global_expert in range(layer.global_num_experts):
            # 读取当前全局 expert 映射到的本地 slot 编号。
            slot = int(layer._expert_map[global_expert].item())

            # 当 slot 编号落在当前 resident slot 范围内时，记录其反向映射关系。
            if 0 <= slot < self.num_slots:
                # 本地slot --> 全局slot
                self._slot_to_global[slot] = global_expert

        # ------------------------------- 初始化 CPU 常驻 expert 池与原始权重缓冲区 -------------------------------
        # 根据当前计划与权重模式初始化 CPU 侧固定 expert 池和原始权重缓冲区。
        self._init_cpu_fixed_pools()

    @property
    def slot_to_global(self) -> tuple[int, ...]:
        # 对外暴露只读视图，避免调用方直接改写内部 resident 映射。
        return tuple(self._slot_to_global)

    @property
    def prefill_burst_capacity(self) -> int:
        # 未配置 burst pool 时，容量视为 0。
        if self.prefill_burst_pool is None:
            return 0
        # 否则返回共享 burst pool 可临时容纳的 experts 数。
        return self.prefill_burst_pool.num_slots

    @property
    def prefill_burst_min_tokens(self) -> int:
        # 无 burst pool 时，该阈值没有意义。
        if self.prefill_burst_pool is None:
            return 0
        # 返回 burst path 触发所需的最小 token 数。
        return self._prefill_burst_min_tokens

    def can_run_prefill_burst(self, num_unique_experts: int, num_tokens: int) -> bool:
        # 只有 burst pool 存在、容量足够、token 数够大且层形状兼容时，才能走 burst。
        return (
                self.prefill_burst_pool is not None
                and num_unique_experts <= self.prefill_burst_pool.num_slots
                and num_tokens >= self.prefill_burst_min_tokens
                and self.prefill_burst_pool.supports_layer(self.layer)
        )

    def run_prefill_burst(
            self,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 没有共享 burst pool 时，说明该层不支持/未启用该执行路径。
        if self.prefill_burst_pool is None:
            raise RuntimeError(f"{self.layer_key}: prefill burst pool is not available")
        # 真正执行逻辑下沉给共享 burst pool。
        return self.prefill_burst_pool.execute(
            controller=self,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts_input=shared_experts_input,
        )

    def prepare(self, topk_ids: torch.Tensor) -> None:
        # ----------------- 解析本次请求的唯一 experts 集合 -----------------
        requested = self._unique_experts(topk_ids)
        # 没有 routed experts 时无需做任何事。
        if not requested:
            return
        # 若本次请求的 unique experts 已超过 resident slot 容量，则说明外层应改走 chunking/burst。
        if len(requested) > self.num_slots:
            raise RuntimeError(
                f"{self.layer_key}: requested {len(requested)} experts but only "
                f"{self.num_slots} GPU slots are available"
            )

        # ----------------- 更新访问统计，供驱逐策略使用 -----------------
        self._step += 1
        for expert_id in requested:
            self._access_count[expert_id] += 1
            self._last_used_step[expert_id] = self._step

        # 当前请求里的 experts 不能在同一轮 prepare 中被驱逐。
        protected = set(requested)
        # 先把本轮缺失 experts 聚合成一个批量加载计划。
        # 这样 Python 主流程不再把“选 slot / 取 CPU bundle / 写 GPU slot”散成多段，
        # 后续可直接把该批量入口替换成真正的 C++/CUDA H2D 加载算子。
        load_plan: list[tuple[int, int]] = []
        for expert_id in requested:
            # 已经常驻在 GPU resident slots 中的 expert 直接跳过。
            if self._is_resident(expert_id):
                continue
            # 为缺失 expert 选一个可替换 slot。
            slot = self._choose_victim_slot(protected)
            load_plan.append((expert_id, slot))
            protected.add(expert_id)

        # 再统一执行本轮专家换入。
        if load_plan:
            self._load_experts_into_slots(load_plan)

    def _unique_experts(self, topk_ids: torch.Tensor) -> list[int]:
        # 空输入直接返回空列表。
        if topk_ids.numel() == 0:
            return []
        # 先做 unique，再回 CPU 转成 Python int 列表，方便后续控制流使用。
        unique_ids = torch.unique(topk_ids.detach()).to(device="cpu")
        return [int(expert_id) for expert_id in unique_ids.tolist()]

    def _is_resident(self, expert_id: int) -> bool:
        # expert_map 中 slot>=0 表示该全局 expert 当前正驻留在某个 GPU slot 中。
        return int(self.layer._expert_map[expert_id].item()) >= 0

    def _choose_victim_slot(self, protected: set[int]) -> int:
        # ----------------- 优先找空闲 slot -----------------
        for slot, global_expert in enumerate(self._slot_to_global):
            if global_expert < 0:
                return slot

        # ----------------- 否则从未受保护的 resident experts 中挑驱逐对象 -----------------
        candidates = [
            slot
            for slot, global_expert in enumerate(self._slot_to_global)
            if global_expert not in protected
        ]
        # 所有 slot 都被本轮请求保护时，说明外层 chunking 逻辑没把批次切到位。
        if not candidates:
            raise RuntimeError(
                f"{self.layer_key}: no evictable slot available for requested experts"
            )

        # 选驱逐分数最小的 slot；若分数相同则按 slot 编号打破平局。
        return min(
            candidates,
            key=lambda slot: (self._eviction_score(self._slot_to_global[slot]), slot),
        )

    def _eviction_score(self, expert_id: int) -> tuple[int, int, int]:
        # 若 expert 不在 CPU static 集合里，重新加载代价更高，因此附加 keep_bonus 尽量少驱逐。
        in_cpu_static = expert_id in self._cpu_static_experts
        keep_bonus = 4096 if not in_cpu_static else 0
        # 排序键依次是“访问次数 + 保留加成”“最近使用时间”“expert id”。
        return (
            self._access_count[expert_id] + keep_bonus,
            self._last_used_step[expert_id],
            expert_id,
        )

    def _materialize_cpu_static_bundle(
            self,
            expert_id: int,
            *,
            eager: bool = False,
    ) -> ExpertBundle | None:
        # ------------------------------- 仅为计划指定的 CPU 常驻 expert 物化静态 bundle -------------------------------
        # 当当前 expert 不属于 CPU 常驻 expert 集合时，直接返回 None。
        if expert_id not in self._cpu_static_experts:
            return None

        # ------------------------------- 优先复用已经物化完成的 CPU 静态 bundle -------------------------------
        # 先从 CPU 静态缓存字典中查找当前 expert 对应的 bundle。
        bundle = self._cpu_static_bundles.get(expert_id)

        # 当当前 expert 的 CPU 静态 bundle 已经存在时，直接复用该 bundle。
        if bundle is not None:
            return bundle

        # ------------------------------- 首次从本地 expert store 物化 CPU 静态 bundle -------------------------------
        # 为当前 expert 分配一份源 bundle，用于承接从本地存储中读取出的权重张量。
        bundle = self._allocate_source_bundle()

        # 当 GPTQ 未启用 desc_act 时，g_idx 可由运行时重建，因此这里允许跳过该后缀字段。
        skip_suffixes = ("g_idx",) if not getattr(self, "_gptq_desc_act", False) else ()

        # 将当前 expert 的权重从本地 expert store 拷贝到新分配的 CPU bundle 中。
        self.expert_store.copy_expert_into(
            self.layer_key,
            expert_id,
            bundle.tensors,
            skip_suffixes=skip_suffixes,
        )

        # ------------------------------- 一次性预处理成 runtime-ready CPU static bundle -------------------------------
        # CPU static mirror 的核心目标是“专家参数在 CPU 中已经是推理 runtime 格式”。
        # 因此 w13/w2 合并、GPTQ Marlin repack、scale permute 这类预处理应在物化时完成一次，
        # 后续 cache miss 只负责把 runtime-ready bundle 直接加载到 GPU resident slot。
        bundle = self._preprocess_cpu_static_bundle(bundle)

        # ------------------------------- 将新物化的 bundle 注册到 CPU 静态缓存并更新占用统计 -------------------------------
        # 将当前 expert 对应的 CPU 静态 bundle 写入缓存字典。
        self._cpu_static_bundles[expert_id] = bundle

        # 将当前 bundle 的字节占用累计到 CPU buffer 总大小中。
        self._cpu_buffer_bytes += bundle.nbytes

        # 统计当前已经完成物化的 CPU 静态 expert 数量。
        materialized = len(self._cpu_static_bundles)

        # ------------------------------- 在关键物化节点输出日志 -------------------------------
        # 仅在前几个 expert 或每累计 32 个静态 expert 时输出日志，避免日志过于频繁。
        if materialized <= 8 or materialized % 32 == 0:
            logger.info(
                "Materialized CPU static expert bundle: layer=%s expert=%d "
                "mode=%s materialized=%d/%d cpu_bytes=%.2f MiB",
                self.layer_key,
                expert_id,
                "eager" if eager else "lazy",
                materialized,
                len(self._cpu_static_experts),
                self._cpu_buffer_bytes / (1 << 20),
            )

        # ------------------------------- 返回当前物化完成的 CPU 静态 bundle -------------------------------
        # 返回当前 expert 对应的 CPU 静态 bundle。
        return bundle

    def _preprocess_cpu_static_bundle(self, bundle: ExpertBundle) -> ExpertBundle:
        # 已经是 runtime-ready 的 bundle 不重复处理。
        if bundle.runtime_ready:
            return bundle

        # 量化路径下，CPU static mirror 应缓存 Marlin MoE runtime 格式：
        # gate/up 先合并为 w13，w13/w2 qweight 做 repack，scale 做 permute。
        if getattr(self, "_mode", "") == "gptq_marlin":
            return self._preprocess_quantized_static_bundle(bundle)

        # 非量化路径下，CPU static mirror 应缓存执行层直接消费的 w13/w2。
        if getattr(self, "_mode", "") == "unquantized":
            return self._preprocess_unquantized_static_bundle(bundle)

        # 单测或兼容旧路径中可能通过 __new__ 构造半初始化对象；此时保持原样。
        return bundle

    def _preprocess_quantized_static_bundle(self, bundle: ExpertBundle) -> ExpertBundle:
        # ------------------------------- 将量化 CPU 静态 bundle 预处理为 GPTQ Marlin runtime 格式 -------------------------------
        # 先将 checkpoint 形态的 gate、up、down 张量拼装成单 expert 原始 w13 与 w2 结构。
        raw = self._assemble_raw_weights(bundle)

        # 将拼装后的原始权重移动到目标 GPU 设备，以便后续在设备侧执行 repack 与 permute。
        raw_gpu = self._move_raw_weights_to_device(raw)

        # ------------------------------- 按 desc_act 配置准备 g_idx 及其排序索引 -------------------------------
        # 当启用了 GPTQ desc_act 时，需要显式构造排序后的 g_idx 与对应排序索引。
        if self._gptq_desc_act:
            if raw_gpu.w13_g_idx is None or raw_gpu.w2_g_idx is None:
                raise RuntimeError(
                    f"{self.layer_key}: desc_act=True requires g_idx tensors "
                    "during CPU static preprocessing"
                )

            # 计算 w13 的 g_idx 排序索引。
            w13_g_idx_sort_indices = torch.argsort(raw_gpu.w13_g_idx, dim=-1).to(
                torch.int32
            )

            # 计算 w2 的 g_idx 排序索引。
            w2_g_idx_sort_indices = torch.argsort(raw_gpu.w2_g_idx, dim=-1).to(
                torch.int32
            )

            # 根据排序索引生成排序后的 w13 g_idx。
            w13_sorted_g_idx = torch.gather(
                raw_gpu.w13_g_idx, -1, w13_g_idx_sort_indices
            )

            # 根据排序索引生成排序后的 w2 g_idx。
            w2_sorted_g_idx = torch.gather(
                raw_gpu.w2_g_idx, -1, w2_g_idx_sort_indices
            )
        else:
            # 当未启用 desc_act 时，g_idx 与排序索引都使用空占位张量或 None。
            w13_g_idx_sort_indices = self._empty_perm(raw_gpu.w13_qweight.device)
            w2_g_idx_sort_indices = self._empty_perm(raw_gpu.w2_qweight.device)
            w13_sorted_g_idx = None
            w2_sorted_g_idx = None

        # ------------------------------- 对量化权重执行 Marlin runtime 预处理 -------------------------------
        # 将 w13 qweight repack 为 Marlin MoE kernel 可直接消费的布局。
        repacked_w13 = ops.gptq_marlin_moe_repack(
            raw_gpu.w13_qweight,
            w13_g_idx_sort_indices,
            raw_gpu.w13_qweight.shape[1] * self.pack_factor,
            raw_gpu.w13_qweight.shape[2],
            self.num_bits,
            is_a_8bit=self.is_a_8bit,
        )

        # 将 w2 qweight repack 为 Marlin MoE kernel 可直接消费的布局。
        repacked_w2 = ops.gptq_marlin_moe_repack(
            raw_gpu.w2_qweight,
            w2_g_idx_sort_indices,
            raw_gpu.w2_qweight.shape[1] * self.pack_factor,
            raw_gpu.w2_qweight.shape[2],
            self.num_bits,
            is_a_8bit=self.is_a_8bit,
        )

        # 对 w13 scales 按 Marlin MoE 需要的访存布局执行 permute。
        permuted_w13_scales = marlin_moe_permute_scales(
            s=raw_gpu.w13_scales,
            size_k=self.layer.intermediate_size_per_partition,
            size_n=raw_gpu.w13_scales.shape[2],
            group_size=self.group_size,
            is_a_8bit=self.is_a_8bit,
        )

        # 对 w2 scales 按 Marlin MoE 需要的访存布局执行 permute。
        permuted_w2_scales = marlin_moe_permute_scales(
            s=raw_gpu.w2_scales,
            size_k=raw_gpu.w2_scales.shape[1]
                   * (self.group_size if self.group_size != -1 else self.pack_factor),
            size_n=raw_gpu.w2_scales.shape[2],
            group_size=self.group_size,
            is_a_8bit=self.is_a_8bit,
        )

        # ------------------------------- 构造 runtime-ready 的 CPU 静态张量字典 -------------------------------
        # 将设备侧预处理后的运行时张量转回 CPU 静态镜像格式。
        tensors = {
            "runtime.w13_qweight": self._to_cpu_static_tensor(repacked_w13[0]),
            "runtime.w2_qweight": self._to_cpu_static_tensor(repacked_w2[0]),
            "runtime.w13_scales": self._to_cpu_static_tensor(permuted_w13_scales[0]),
            "runtime.w2_scales": self._to_cpu_static_tensor(permuted_w2_scales[0]),
            "runtime.w13_qzeros": self._to_cpu_static_tensor(raw_gpu.w13_qzeros[0]),
            "runtime.w2_qzeros": self._to_cpu_static_tensor(raw_gpu.w2_qzeros[0]),
        }

        # 当 desc_act 路径下同时具备排序后的 g_idx 及其排序索引时，一并写入 runtime-ready 张量字典。
        if (
                w13_sorted_g_idx is not None
                and w2_sorted_g_idx is not None
                and self._gptq_desc_act
        ):
            tensors.update(
                {
                    "runtime.w13_g_idx": self._to_cpu_static_tensor(
                        w13_sorted_g_idx[0]
                    ),
                    "runtime.w2_g_idx": self._to_cpu_static_tensor(
                        w2_sorted_g_idx[0]
                    ),
                    "runtime.w13_g_idx_sort_indices": self._to_cpu_static_tensor(
                        w13_g_idx_sort_indices[0]
                    ),
                    "runtime.w2_g_idx_sort_indices": self._to_cpu_static_tensor(
                        w2_g_idx_sort_indices[0]
                    ),
                }
            )

        # 返回预处理完成的 runtime-ready ExpertBundle。
        return ExpertBundle(
            tensors=tensors,
            nbytes=bundle_nbytes(tensors),
            pinned=all(tensor.is_pinned() for tensor in tensors.values()),
            runtime_ready=True,
        )

    def _preprocess_unquantized_static_bundle(
            self, bundle: ExpertBundle
    ) -> ExpertBundle:
        raw = self._assemble_unquantized_weights(bundle)
        tensors = {
            "runtime.w13_weight": self._to_cpu_static_tensor(raw.w13_weight[0]),
            "runtime.w2_weight": self._to_cpu_static_tensor(raw.w2_weight[0]),
        }
        return ExpertBundle(
            tensors=tensors,
            nbytes=bundle_nbytes(tensors),
            pinned=all(tensor.is_pinned() for tensor in tensors.values()),
            runtime_ready=True,
        )

    def _to_cpu_static_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        # ------------------------------- 先将输入张量转换为 CPU 上的连续张量 -------------------------------
        # 将输入张量从原设备分离出来，并转换为位于 CPU 上的 contiguous 张量。
        cpu_tensor = tensor.detach().to(device="cpu").contiguous()

        # ------------------------------- 在无需 pin 或已经是 pinned memory 时直接返回 -------------------------------
        # 当当前控制器未启用 pinned CPU memory，或者该张量本身已经处于 pinned 状态时，直接返回。
        if not self._use_pinned_cpu or cpu_tensor.is_pinned():
            return cpu_tensor

        # ------------------------------- 在允许时尝试将 CPU 张量转换为 pinned memory -------------------------------
        try:
            # 将当前 CPU 张量转换为 pinned memory 版本，以加快后续 H2D 传输。
            return cpu_tensor.pin_memory()
        except RuntimeError as exc:
            # 当 pin_memory 失败时，记录一次告警并退回到普通 pageable CPU 张量。
            logger.warning_once(
                "Failed to pin runtime-ready CPU expert bundle tensor; "
                "falling back to pageable CPU memory. error=%s",
                exc,
            )
            return cpu_tensor

    def _get_source_bundle(self, expert_id: int) -> tuple[ExpertBundle, str]:
        # 先尝试命中 CPU static experts。
        bundle = self._materialize_cpu_static_bundle(expert_id)
        source = "cpu_static"
        if bundle is None:
            raise RuntimeError(
                f"{self.layer_key}: expert {expert_id} is not present in the CPU "
                "static expert pool. Current MoE tiered cache mainline requires "
                "initial_cpu_experts to cover every expert."
            )
        # 返回 bundle 以及来源标签，供统计与日志使用。
        return bundle, source

    def _load_expert_into_slot(self, expert_id: int, slot: int) -> None:
        # 单 expert 入口保留给旧调用点；内部收敛到批量加载入口。
        self._load_experts_into_slots([(expert_id, slot)])

    def _load_experts_into_slots(self, assignments: list[tuple[int, int]]) -> None:
        # 当前主线里，动态换入专家只能来自 CPU static mirror。
        bundles_and_sources: list[tuple[int, int, ExpertBundle, str]] = []
        for expert_id, slot in assignments:
            bundle, source = self._get_source_bundle(expert_id)
            bundles_and_sources.append((expert_id, slot, bundle, source))
            if source == "nvme_stage":
                self._nvme_loads += 1
            else:
                self._cpu_hits += 1

        # 批量写入入口：当前 Python 版逐 slot copy，但调用语义已经变成“本轮一次批量加载”。
        # 下一步可以在这里下沉到 C++/CUDA：一次接收 N 个 CPU runtime-ready experts，
        # 一次调度完成 N 个 resident GPU slots 的 H2D 覆盖。
        self._write_expert_bundles(bundles_and_sources, self.layer)

        for expert_id, slot, _bundle, source in bundles_and_sources:
            # 覆盖 resident 映射表；旧 expert 若存在则在这里被驱逐。
            self._install_mapping(expert_id, slot)
            # 增加累计加载计数。
            self._total_loads += 1
            # 仅在前几次或每 100 次动态换入时打日志。
            if self._total_loads <= 8 or self._total_loads % 100 == 0:
                logger.info(
                    "Tiered MoE cache event: layer=%s load=%d batch=%d source=%s "
                    "expert=%d slot=%d cpu_hits=%d nvme_loads=%d evictions=%d",
                    self.layer_key,
                    self._total_loads,
                    len(assignments),
                    source,
                    expert_id,
                    slot,
                    self._cpu_hits,
                    self._nvme_loads,
                    self._evictions,
                )

    def _populate_prefill_burst_pool(
            self,
            pool: SharedPrefillBurstPool,
            requested: list[int],
    ) -> _PrefillBurstExecutionStats:
        # 统计本轮 burst pool 中不同来源的命中情况。
        stats = _PrefillBurstExecutionStats()
        # 逐个 requested expert 填满 burst pool 的临时 slots。
        for slot, expert_id in enumerate(requested):
            # 已经 resident 的 expert 直接从主 GPU slots 拷到 burst pool。
            if self._is_resident(expert_id):
                self._copy_resident_expert_to_target(expert_id, slot, pool)
                stats.resident_hits += 1
                continue
            # 非 resident 的 expert 则先从 CPU static 或 NVMe staging 获取来源 bundle。
            bundle, source = self._get_source_bundle(expert_id)
            # 再把来源 bundle 写入 burst pool 的临时 slot。
            self._write_expert_bundle(slot, bundle, pool)
            if source == "nvme_stage":
                stats.nvme_loads += 1
            else:
                stats.cpu_hits += 1
        return stats

    def _write_expert_bundles(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
            target: Any,
    ) -> None:
        # 当前实现先提供批量加载语义边界：同一轮 prepare 中缺失的 experts 聚合到这里统一处理。
        # 若 bundle 已经是 runtime-ready，运行时只做 CPU -> GPU slot 拷贝，不再重复 w13/w2 合并、
        # GPTQ Marlin repack 或 scale permute。
        if self._try_write_runtime_ready_bundles_batch(
                bundles_and_sources, target
        ):
            return

        for _expert_id, slot, bundle, _source in bundles_and_sources:
            self._write_expert_bundle(slot, bundle, target)

    def _try_write_runtime_ready_bundles_batch(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
            target: Any,
    ) -> bool:
        if not bundles_and_sources:
            return False

        bundles = [bundle for _expert_id, _slot, bundle, _source in bundles_and_sources]
        if not all(bundle.runtime_ready for bundle in bundles):
            return False

        if self._mode == "unquantized":
            if not ops.has_precompiled_moe_batch_load_unquantized_runtime():
                return False
            self._write_unquantized_runtime_ready_bundles_batch(
                bundles_and_sources, target
            )
            return True

        if self._mode == "gptq_marlin":
            if not ops.has_precompiled_moe_batch_load_gptq_runtime():
                return False
            self._write_gptq_runtime_ready_bundles_batch(
                bundles_and_sources, target
            )
            return True

        return False

    def _stack_runtime_ready_cpu_field(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
            field_name: str,
    ) -> torch.Tensor:
        field_tensors = [
            bundle.tensors[field_name]
            for _expert_id, _slot, bundle, _source in bundles_and_sources
        ]
        first = field_tensors[0]
        batch_shape = (len(field_tensors), *first.shape)
        use_pinned_batch = all(
            tensor.device.type == "cpu" and tensor.is_pinned()
            for tensor in field_tensors
        )

        batch_kwargs: dict[str, Any] = {"device": "cpu", "dtype": first.dtype}
        if use_pinned_batch:
            batch_kwargs["pin_memory"] = True
        try:
            batch = torch.empty(batch_shape, **batch_kwargs)
        except RuntimeError:
            batch_kwargs.pop("pin_memory", None)
            batch = torch.empty(batch_shape, **batch_kwargs)

        for batch_idx, tensor in enumerate(field_tensors):
            cpu_tensor = (
                tensor
                if tensor.device.type == "cpu"
                else tensor.to(device="cpu", non_blocking=use_pinned_batch)
            )
            batch[batch_idx].copy_(cpu_tensor, non_blocking=use_pinned_batch)
        return batch

    def _maybe_stack_runtime_ready_cpu_field(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
            field_name: str,
    ) -> torch.Tensor | None:
        if any(
                field_name not in bundle.tensors
                for _expert_id, _slot, bundle, _source in bundles_and_sources
        ):
            return None
        return self._stack_runtime_ready_cpu_field(
            bundles_and_sources, field_name
        )

    def _write_unquantized_runtime_ready_bundles_batch(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
            target: Any,
    ) -> None:
        slot_ids = torch.tensor(
            [slot for _expert_id, slot, _bundle, _source in bundles_and_sources],
            dtype=torch.int64,
        )
        w13_batch = self._stack_runtime_ready_cpu_field(
            bundles_and_sources, "runtime.w13_weight"
        )
        w2_batch = self._stack_runtime_ready_cpu_field(
            bundles_and_sources, "runtime.w2_weight"
        )
        ops.moe_batch_load_unquantized_runtime_precompiled(
            slot_ids,
            w13_batch,
            w2_batch,
            target.w13_weight,
            target.w2_weight,
        )

    def _write_gptq_runtime_ready_bundles_batch(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
            target: Any,
    ) -> None:
        slot_ids = torch.tensor(
            [slot for _expert_id, slot, _bundle, _source in bundles_and_sources],
            dtype=torch.int64,
        )
        w13_g_idx_batch = None
        w2_g_idx_batch = None
        w13_g_idx_sort_indices_batch = None
        w2_g_idx_sort_indices_batch = None
        target_w13_g_idx = None
        target_w2_g_idx = None
        target_w13_g_idx_sort_indices = None
        target_w2_g_idx_sort_indices = None
        if hasattr(target, "w13_g_idx"):
            w13_g_idx_batch = self._maybe_stack_runtime_ready_cpu_field(
                bundles_and_sources, "runtime.w13_g_idx"
            )
            w2_g_idx_batch = self._maybe_stack_runtime_ready_cpu_field(
                bundles_and_sources, "runtime.w2_g_idx"
            )
            w13_g_idx_sort_indices_batch = self._maybe_stack_runtime_ready_cpu_field(
                bundles_and_sources, "runtime.w13_g_idx_sort_indices"
            )
            w2_g_idx_sort_indices_batch = self._maybe_stack_runtime_ready_cpu_field(
                bundles_and_sources, "runtime.w2_g_idx_sort_indices"
            )
            if (
                    w13_g_idx_batch is not None
                    and w2_g_idx_batch is not None
                    and w13_g_idx_sort_indices_batch is not None
                    and w2_g_idx_sort_indices_batch is not None
            ):
                target_w13_g_idx = target.w13_g_idx
                target_w2_g_idx = target.w2_g_idx
                target_w13_g_idx_sort_indices = target.w13_g_idx_sort_indices
                target_w2_g_idx_sort_indices = target.w2_g_idx_sort_indices

        ops.moe_batch_load_gptq_runtime_precompiled(
            slot_ids,
            self._stack_runtime_ready_cpu_field(
                bundles_and_sources, "runtime.w13_qweight"
            ),
            self._stack_runtime_ready_cpu_field(
                bundles_and_sources, "runtime.w2_qweight"
            ),
            self._stack_runtime_ready_cpu_field(
                bundles_and_sources, "runtime.w13_scales"
            ),
            self._stack_runtime_ready_cpu_field(
                bundles_and_sources, "runtime.w2_scales"
            ),
            self._stack_runtime_ready_cpu_field(
                bundles_and_sources, "runtime.w13_qzeros"
            ),
            self._stack_runtime_ready_cpu_field(
                bundles_and_sources, "runtime.w2_qzeros"
            ),
            target.w13_qweight,
            target.w2_qweight,
            target.w13_scales,
            target.w2_scales,
            target.w13_qzeros,
            target.w2_qzeros,
            w13_g_idx_batch,
            w2_g_idx_batch,
            w13_g_idx_sort_indices_batch,
            w2_g_idx_sort_indices_batch,
            target_w13_g_idx,
            target_w2_g_idx,
            target_w13_g_idx_sort_indices,
            target_w2_g_idx_sort_indices,
        )

    def _copy_resident_expert_to_target(
            self,
            expert_id: int,
            slot: int,
            target: Any,
    ) -> None:
        # 通过主层的 expert_map 找到该 resident expert 当前位于哪个 GPU slot。
        src_slot = int(self.layer._expert_map[expert_id].item())
        # 若找不到 resident slot，说明调用方状态不一致。
        if src_slot < 0:
            raise RuntimeError(
                f"{self.layer_key}: expert {expert_id} is not resident in GPU slots"
            )
        with torch.no_grad():
            # 量化路径下把所有 runtime 相关张量完整拷贝到目标对象。
            if self._mode == "gptq_marlin":
                target.w13_qweight[slot].copy_(self.layer.w13_qweight[src_slot])
                target.w2_qweight[slot].copy_(self.layer.w2_qweight[src_slot])
                target.w13_scales[slot].copy_(self.layer.w13_scales[src_slot])
                target.w2_scales[slot].copy_(self.layer.w2_scales[src_slot])
                target.w13_qzeros[slot].copy_(self.layer.w13_qzeros[src_slot])
                target.w2_qzeros[slot].copy_(self.layer.w2_qzeros[src_slot])
                if hasattr(target, "w13_g_idx"):
                    target.w13_g_idx[slot].copy_(self.layer.w13_g_idx[src_slot])
                    target.w2_g_idx[slot].copy_(self.layer.w2_g_idx[src_slot])
                    target.w13_g_idx_sort_indices[slot].copy_(
                        self.layer.w13_g_idx_sort_indices[src_slot]
                    )
                    target.w2_g_idx_sort_indices[slot].copy_(
                        self.layer.w2_g_idx_sort_indices[src_slot]
                    )
                return
            # 非量化路径只需要拷 w13/w2 两块 runtime 权重。
            target.w13_weight[slot].copy_(self.layer.w13_weight[src_slot])
            target.w2_weight[slot].copy_(self.layer.w2_weight[src_slot])

    def _write_expert_bundle(self, slot: int, bundle: ExpertBundle, target: Any) -> None:
        if bundle.runtime_ready:
            self._write_runtime_ready_bundle(slot, bundle, target)
            return

        # ----------------- 量化路径：bundle -> CPU raw -> GPU raw -> Marlin runtime 格式 -----------------
        if self._mode == "gptq_marlin":
            # 先把 gate/up/down 三份量化张量重组回单 expert 的原始结构。
            raw = self._assemble_raw_weights(bundle)
            # 再把 raw 权重异步搬到目标 GPU 设备。
            raw_gpu = self._move_raw_weights_to_device(raw)
            if self._gptq_desc_act:
                if raw_gpu.w13_g_idx is None or raw_gpu.w2_g_idx is None:
                    raise RuntimeError(
                        f"{self.layer_key}: desc_act=True requires g_idx tensors "
                        "during dynamic expert load"
                    )
                w13_g_idx_sort_indices = torch.argsort(
                    raw_gpu.w13_g_idx, dim=-1
                ).to(torch.int32)
                w2_g_idx_sort_indices = torch.argsort(
                    raw_gpu.w2_g_idx, dim=-1
                ).to(torch.int32)
                w13_sorted_g_idx = torch.gather(
                    raw_gpu.w13_g_idx, -1, w13_g_idx_sort_indices
                )
                w2_sorted_g_idx = torch.gather(
                    raw_gpu.w2_g_idx, -1, w2_g_idx_sort_indices
                )
            else:
                w13_g_idx_sort_indices = self._empty_perm(raw_gpu.w13_qweight.device)
                w2_g_idx_sort_indices = self._empty_perm(raw_gpu.w2_qweight.device)
                w13_sorted_g_idx = None
                w2_sorted_g_idx = None
            # qweight 需要重新 repack 成 Marlin MoE kernel 可直接消费的布局。
            repacked_w13 = ops.gptq_marlin_moe_repack(
                raw_gpu.w13_qweight,
                w13_g_idx_sort_indices,
                raw_gpu.w13_qweight.shape[1] * self.pack_factor,
                raw_gpu.w13_qweight.shape[2],
                self.num_bits,
                is_a_8bit=self.is_a_8bit,
            )
            # w2 也要独立做一遍 repack。
            repacked_w2 = ops.gptq_marlin_moe_repack(
                raw_gpu.w2_qweight,
                w2_g_idx_sort_indices,
                raw_gpu.w2_qweight.shape[1] * self.pack_factor,
                raw_gpu.w2_qweight.shape[2],
                self.num_bits,
                is_a_8bit=self.is_a_8bit,
            )
            # scale 还需要按照 Marlin 的访存布局做一次 permute。
            permuted_w13_scales = marlin_moe_permute_scales(
                s=raw_gpu.w13_scales,
                size_k=self.layer.intermediate_size_per_partition,
                size_n=raw_gpu.w13_scales.shape[2],
                group_size=self.group_size,
                is_a_8bit=self.is_a_8bit,
            )
            # w2 scale 的 size_k 定义与 w13 不同，这里按下投影维度单独计算。
            permuted_w2_scales = marlin_moe_permute_scales(
                s=raw_gpu.w2_scales,
                size_k=raw_gpu.w2_scales.shape[1]
                       * (self.group_size if self.group_size != -1 else self.pack_factor),
                size_n=raw_gpu.w2_scales.shape[2],
                group_size=self.group_size,
                is_a_8bit=self.is_a_8bit,
            )
            # 最后把 repack/permute 后的结果写进目标 GPU slot。
            self._write_quantized_target_slot(
                target=target,
                slot=slot,
                repacked_w13=repacked_w13[0],
                repacked_w2=repacked_w2[0],
                permuted_w13_scales=permuted_w13_scales[0],
                permuted_w2_scales=permuted_w2_scales[0],
                raw_gpu=raw_gpu,
                w13_g_idx=w13_sorted_g_idx[0] if w13_sorted_g_idx is not None else None,
                w2_g_idx=w2_sorted_g_idx[0] if w2_sorted_g_idx is not None else None,
                w13_g_idx_sort_indices=(
                    w13_g_idx_sort_indices[0]
                    if self._gptq_desc_act
                    else None
                ),
                w2_g_idx_sort_indices=(
                    w2_g_idx_sort_indices[0]
                    if self._gptq_desc_act
                    else None
                ),
            )
            return

        # ----------------- 非量化路径：bundle -> CPU raw -> GPU raw -> runtime kernel 格式 -----------------
        # 先把 gate/up/down 三份原始权重拼成运行时需要的 w13/w2 结构。
        raw = self._assemble_unquantized_weights(bundle)
        # 再把拼好的原始权重搬到 GPU。
        raw_gpu = self._move_unquantized_weights_to_device(raw)
        # 不同 backend 可能需要额外格式转换，这里统一走 helper。
        runtime_w13, runtime_w2 = convert_to_unquantized_kernel_format(
            self.quant_method.unquantized_backend,
            layer=self.layer,
            w13_weight=raw_gpu.w13_weight,
            w2_weight=raw_gpu.w2_weight,
        )
        # 把 runtime 格式的权重写进目标 slot。
        self._write_unquantized_target_slot(
            target=target,
            slot=slot,
            runtime_w13=runtime_w13[0],
            runtime_w2=runtime_w2[0],
        )

    def _write_runtime_ready_bundle(
            self,
            slot: int,
            bundle: ExpertBundle,
            target: Any,
    ) -> None:
        tensors = bundle.tensors
        with torch.no_grad():
            if self._mode == "gptq_marlin":
                target.w13_qweight[slot].copy_(
                    tensors["runtime.w13_qweight"], non_blocking=bundle.pinned
                )
                target.w2_qweight[slot].copy_(
                    tensors["runtime.w2_qweight"], non_blocking=bundle.pinned
                )
                target.w13_scales[slot].copy_(
                    tensors["runtime.w13_scales"], non_blocking=bundle.pinned
                )
                target.w2_scales[slot].copy_(
                    tensors["runtime.w2_scales"], non_blocking=bundle.pinned
                )
                target.w13_qzeros[slot].copy_(
                    tensors["runtime.w13_qzeros"], non_blocking=bundle.pinned
                )
                target.w2_qzeros[slot].copy_(
                    tensors["runtime.w2_qzeros"], non_blocking=bundle.pinned
                )
                if hasattr(target, "w13_g_idx") and "runtime.w13_g_idx" in tensors:
                    target.w13_g_idx[slot].copy_(
                        tensors["runtime.w13_g_idx"], non_blocking=bundle.pinned
                    )
                    target.w2_g_idx[slot].copy_(
                        tensors["runtime.w2_g_idx"], non_blocking=bundle.pinned
                    )
                    target.w13_g_idx_sort_indices[slot].copy_(
                        tensors["runtime.w13_g_idx_sort_indices"],
                        non_blocking=bundle.pinned,
                    )
                    target.w2_g_idx_sort_indices[slot].copy_(
                        tensors["runtime.w2_g_idx_sort_indices"],
                        non_blocking=bundle.pinned,
                    )
                return

            target.w13_weight[slot].copy_(
                tensors["runtime.w13_weight"], non_blocking=bundle.pinned
            )
            target.w2_weight[slot].copy_(
                tensors["runtime.w2_weight"], non_blocking=bundle.pinned
            )

    def _write_quantized_target_slot(
            self,
            *,
            target: Any,
            slot: int,
            repacked_w13: torch.Tensor,
            repacked_w2: torch.Tensor,
            permuted_w13_scales: torch.Tensor,
            permuted_w2_scales: torch.Tensor,
            raw_gpu: _RawExpertWeights,
            w13_g_idx: torch.Tensor | None = None,
            w2_g_idx: torch.Tensor | None = None,
            w13_g_idx_sort_indices: torch.Tensor | None = None,
            w2_g_idx_sort_indices: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            # 依次覆盖目标 slot 中的 runtime 量化权重与 scale。
            target.w13_qweight[slot].copy_(repacked_w13)
            target.w2_qweight[slot].copy_(repacked_w2)
            target.w13_scales[slot].copy_(permuted_w13_scales)
            target.w2_scales[slot].copy_(permuted_w2_scales)
            # qzeros 不参与 repack/permute，直接沿用 raw 格式写入即可。
            target.w13_qzeros[slot].copy_(raw_gpu.w13_qzeros[0])
            target.w2_qzeros[slot].copy_(raw_gpu.w2_qzeros[0])
            if (
                    w13_g_idx is not None
                    and w2_g_idx is not None
                    and w13_g_idx_sort_indices is not None
                    and w2_g_idx_sort_indices is not None
                    and hasattr(target, "w13_g_idx")
            ):
                target.w13_g_idx[slot].copy_(w13_g_idx)
                target.w2_g_idx[slot].copy_(w2_g_idx)
                target.w13_g_idx_sort_indices[slot].copy_(w13_g_idx_sort_indices)
                target.w2_g_idx_sort_indices[slot].copy_(w2_g_idx_sort_indices)

    def _write_unquantized_target_slot(
            self,
            *,
            target: Any,
            slot: int,
            runtime_w13: torch.Tensor,
            runtime_w2: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            # 非量化路径直接覆盖目标 slot 对应的 w13/w2 runtime 权重。
            target.w13_weight[slot].copy_(runtime_w13)
            target.w2_weight[slot].copy_(runtime_w2)

    def _assemble_raw_weights(self, bundle: ExpertBundle) -> _RawExpertWeights:
        # ------------------------------- 读取当前 bundle 的张量字典并预计算 w13 合并布局参数 -------------------------------
        # 取出当前 expert bundle 中保存的张量字典，后续按字段名逐个拼装。
        tensors = bundle.tensors

        # 计算合并后 w13 的总列数，对应 gate_proj 与 up_proj 两半拼接后的总宽度。
        w13_cols = 2 * self.layer.intermediate_size_per_partition

        # 计算 w13 前半部分的列数，对应单个 gate_proj 或 up_proj 的宽度。
        w13_half_cols = self.layer.intermediate_size_per_partition

        # 计算合并后 w13_qzeros 的总列数，其列数按 pack_factor 压缩。
        w13_qzeros_cols = w13_cols // self.pack_factor

        # 计算 w13 前半部分 qzeros 的列数，其列数同样按 pack_factor 压缩。
        w13_half_qzeros_cols = w13_half_cols // self.pack_factor

        # ------------------------------- 校验量化路径所需的 CPU raw buffer 是否已经分配 -------------------------------
        # 当量化路径下的 CPU raw buffer 尚未初始化时，当前 expert 不能继续执行动态加载拼装。
        if self._cpu_quantized_raw_buffer is None:
            raise RuntimeError(f"{self.layer_key}: quantized CPU raw buffer is missing")

        # 取出可重复复用的量化 CPU raw buffer，避免每次动态加载都重新分配。
        raw = self._cpu_quantized_raw_buffer

        # ------------------------------- 初始化字段完整性检查状态 -------------------------------
        # 用于记录 gate_proj、up_proj、down_proj 的关键字段是否已经全部写入 raw buffer。
        seen_fields: set[tuple[str, str]] = set()

        # 记录 gate/up 合并路径是否已经看到 w13 对应的 g_idx 字段。
        seen_w13_g_idx = False

        # 记录 down 路径是否已经看到 w2 对应的 g_idx 字段。
        seen_w2_g_idx = False

        # ------------------------------- 逐张量将 gate、up、down 权重拼装到 raw buffer 中 -------------------------------
        # 遍历当前 bundle 中的全部张量条目，按字段名将其写入对应的 raw buffer 区域。
        for relative_name, tensor in tensors.items():
            # 去掉 slot 前缀，只保留投影名与字段名部分。
            _, suffix = relative_name.split(".", 1)

            # 将剩余部分拆成投影名和字段名两段。
            proj_name, field_name = suffix.split(".", 1)

            # 为保证后续 H2D 路径一致，先确保当前源张量位于 CPU 上。
            cpu_tensor = tensor if tensor.device.type == "cpu" else tensor.to(device="cpu")

            # 当当前张量属于 gate_proj 时，将其写入 w13 的前半部分。
            if proj_name == "gate_proj":
                self._copy_merged_half(
                    field_name,
                    cpu_tensor,
                    raw,
                    offset=0,
                    qzeros_offset=0,
                )
                seen_fields.add((proj_name, field_name))

            # 当当前张量属于 up_proj 时，将其写入 w13 的后半部分。
            elif proj_name == "up_proj":
                self._copy_merged_half(
                    field_name,
                    cpu_tensor,
                    raw,
                    offset=w13_half_cols,
                    qzeros_offset=w13_half_qzeros_cols,
                )
                seen_fields.add((proj_name, field_name))

            # 当当前张量属于 down_proj 时，将其直接写入 w2 对应字段。
            elif proj_name == "down_proj":
                self._copy_direct(field_name, cpu_tensor, raw)
                seen_fields.add((proj_name, field_name))

            # ------------------------------- 记录 g_idx 字段的命中情况 -------------------------------
            # 当当前字段名为 g_idx 时，按投影类型分别记录 w13 或 w2 路径是否已看到对应字段。
            if field_name == "g_idx":
                if proj_name in ("gate_proj", "up_proj"):
                    seen_w13_g_idx = True
                elif proj_name == "down_proj":
                    seen_w2_g_idx = True

        # ------------------------------- 校验动态加载所需的关键字段是否齐全 -------------------------------
        # 定义量化动态加载路径要求必须具备的基础字段集合。
        required_fields = {
            ("gate_proj", "qweight"),
            ("gate_proj", "scales"),
            ("gate_proj", "qzeros"),
            ("up_proj", "qweight"),
            ("up_proj", "scales"),
            ("up_proj", "qzeros"),
            ("down_proj", "qweight"),
            ("down_proj", "scales"),
            ("down_proj", "qzeros"),
        }

        # 计算当前 bundle 中缺失的关键字段集合。
        missing_fields = required_fields - seen_fields

        # 当存在缺失字段时，直接报错并终止当前动态加载。
        if missing_fields:
            raise KeyError(
                f"{self.layer_key}: missing expert tensors for dynamic load: "
                f"{sorted(missing_fields)}"
            )

        # 当当前路径启用了 desc_act，但缺少 w13 或 w2 路径所需的 g_idx 字段时，直接报错。
        if self._gptq_desc_act and (not seen_w13_g_idx or not seen_w2_g_idx):
            raise KeyError(
                f"{self.layer_key}: missing expert g_idx tensors for dynamic load"
            )

        # ------------------------------- 返回当前已拼装完成的量化 raw buffer -------------------------------
        # 返回已经完成 gate、up、down 合并写入的量化 CPU raw buffer。
        return raw

    def _copy_merged_half(
            self,
            field_name: str,
            tensor: torch.Tensor,
            raw: _RawExpertWeights,
            *,
            offset: int,
            qzeros_offset: int,
    ) -> None:
        # ------------------------------- 将 gate 或 up 分支字段写入合并后的 w13 对应半区 -------------------------------
        # 当当前字段为 qweight 时，将该张量按列偏移写入 raw.w13_qweight 的对应半区。
        if field_name == "qweight":
            raw.w13_qweight[0, :, offset: offset + tensor.shape[-1]].copy_(tensor)

        # 当当前字段为 scales 时，将该张量按列偏移写入 raw.w13_scales 的对应半区。
        elif field_name == "scales":
            raw.w13_scales[0, :, offset: offset + tensor.shape[-1]].copy_(tensor)

        # ------------------------------- 将压缩列数的 qzeros 写入合并后的 w13 对应半区 -------------------------------
        # 当当前字段为 qzeros 时，按 qzeros 专用列偏移写入 raw.w13_qzeros 的对应半区。
        elif field_name == "qzeros":
            raw.w13_qzeros[
                0, :, qzeros_offset: qzeros_offset + tensor.shape[-1]
            ].copy_(tensor)

        # ------------------------------- 将 g_idx 写入 w13 的原始索引缓冲区 -------------------------------
        # 当当前字段为 g_idx 时，需要先确认 w13_g_idx 原始缓冲区已经分配完成。
        elif field_name == "g_idx":
            if raw.w13_g_idx is None:
                raise RuntimeError(f"{self.layer_key}: w13_g_idx raw buffer is missing")

            # 将当前 g_idx 张量整块写入 raw.w13_g_idx。
            raw.w13_g_idx[0].copy_(tensor)

    def _copy_direct(
            self, field_name: str, tensor: torch.Tensor, raw: _RawExpertWeights
    ) -> None:
        # down_proj 不需要合并偏移，直接整块写入 w2 对应字段。
        if field_name == "qweight":
            raw.w2_qweight[0].copy_(tensor)
        elif field_name == "scales":
            raw.w2_scales[0].copy_(tensor)
        elif field_name == "qzeros":
            raw.w2_qzeros[0].copy_(tensor)
        elif field_name == "g_idx":
            if raw.w2_g_idx is None:
                raise RuntimeError(f"{self.layer_key}: w2_g_idx raw buffer is missing")
            raw.w2_g_idx[0].copy_(tensor)

    def _move_raw_weights_to_device(self, raw: _RawExpertWeights) -> _RawExpertWeights:
        # ------------------------------- 将量化原始权重缓冲区整体搬运到目标 GPU 设备 -------------------------------
        # 构造并返回一份新的原始权重对象，其中各字段都已从 CPU 异步搬运到当前控制器绑定的目标 GPU 设备。
        return _RawExpertWeights(
            # 将 w13 量化权重异步搬运到目标 GPU。
            w13_qweight=raw.w13_qweight.to(device=self.device, non_blocking=True),

            # 将 w2 量化权重异步搬运到目标 GPU。
            w2_qweight=raw.w2_qweight.to(device=self.device, non_blocking=True),

            # 将 w13 的 scale 张量异步搬运到目标 GPU。
            w13_scales=raw.w13_scales.to(device=self.device, non_blocking=True),

            # 将 w2 的 scale 张量异步搬运到目标 GPU。
            w2_scales=raw.w2_scales.to(device=self.device, non_blocking=True),

            # 将 w13 的 qzeros 张量异步搬运到目标 GPU。
            w13_qzeros=raw.w13_qzeros.to(device=self.device, non_blocking=True),

            # 将 w2 的 qzeros 张量异步搬运到目标 GPU。
            w2_qzeros=raw.w2_qzeros.to(device=self.device, non_blocking=True),

            # 当 w13_g_idx 存在时，将其异步搬运到目标 GPU；否则保持为 None。
            w13_g_idx=(
                raw.w13_g_idx.to(device=self.device, non_blocking=True)
                if raw.w13_g_idx is not None
                else None
            ),

            # 当 w2_g_idx 存在时，将其异步搬运到目标 GPU；否则保持为 None。
            w2_g_idx=(
                raw.w2_g_idx.to(device=self.device, non_blocking=True)
                if raw.w2_g_idx is not None
                else None
            ),
        )

    def _assemble_unquantized_weights(
            self, bundle: ExpertBundle
    ) -> _RawUnquantizedExpertWeights:
        # ----------------- 计算非量化 w13 的拼接尺寸 -----------------
        tensors = bundle.tensors
        is_act_and_mul = bool(self.layer.moe_config.is_act_and_mul)
        w13_up_dim = (
            2 * self.layer.intermediate_size_per_partition
            if is_act_and_mul
            else self.layer.intermediate_size_per_partition
        )
        half_dim = self.layer.intermediate_size_per_partition
        # 非量化路径同样要求预分配 CPU raw buffer。
        if self._cpu_unquantized_raw_buffer is None:
            raise RuntimeError(
                f"{self.layer_key}: unquantized CPU raw buffer is missing"
            )
        raw = self._cpu_unquantized_raw_buffer
        seen_fields: set[tuple[str, str]] = set()

        # ----------------- 把 gate/up/down 权重拼成 runtime 所需的 w13/w2 -----------------
        for relative_name, tensor in tensors.items():
            _, suffix = relative_name.split(".", 1)
            proj_name, field_name = suffix.split(".", 1)
            # 非量化路径只消费原始 weight 字段。
            if field_name != "weight":
                continue
            cpu_tensor = tensor if tensor.device.type == "cpu" else tensor.to(device="cpu")
            # gate_proj 写入 w13 前半部分。
            if proj_name == "gate_proj":
                raw.w13_weight[0, :half_dim].copy_(cpu_tensor)
                seen_fields.add((proj_name, field_name))
            # up_proj 写入 w13 后半部分。
            elif proj_name == "up_proj":
                raw.w13_weight[0, half_dim: half_dim + cpu_tensor.shape[0]].copy_(
                    cpu_tensor
                )
                seen_fields.add((proj_name, field_name))
            # down_proj 直接写入 w2。
            elif proj_name == "down_proj":
                raw.w2_weight[0].copy_(cpu_tensor)
                seen_fields.add((proj_name, field_name))

        # 缺任意一块权重都不能完成动态加载。
        required_fields = {
            ("gate_proj", "weight"),
            ("up_proj", "weight"),
            ("down_proj", "weight"),
        }
        missing_fields = required_fields - seen_fields
        if missing_fields:
            raise KeyError(
                f"{self.layer_key}: missing expert tensors for dynamic load: "
                f"{sorted(missing_fields)}"
            )

        return raw

    def _move_unquantized_weights_to_device(
            self, raw: _RawUnquantizedExpertWeights
    ) -> _RawUnquantizedExpertWeights:
        # 把非量化 raw buffer 异步搬到目标 GPU。
        return _RawUnquantizedExpertWeights(
            w13_weight=raw.w13_weight.to(device=self.device, non_blocking=True),
            w2_weight=raw.w2_weight.to(device=self.device, non_blocking=True),
        )

    def _empty_perm(self, device: torch.device) -> torch.Tensor:
        # 当前动态加载路径不复用预计算 perm，传一个空占位张量给 Marlin repack 接口。
        return torch.empty((1, 0), dtype=torch.int32, device=device)

    def _init_cpu_fixed_pools(self) -> None:
        # ------------------------------- 根据计划解析 CPU 常驻 expert 集合 -------------------------------
        # 从计划中的 initial_cpu_experts 读取需要在 CPU 侧常驻的 expert 编号，并过滤掉越界的 expert。
        cpu_static_experts = tuple(
            int(expert_id)
            for expert_id in self.plan.get("initial_cpu_experts", ())
            if int(expert_id) < self.layer.global_num_experts
        )

        # 将 CPU 常驻 expert 集合冻结为 frozenset，便于后续快速判断与避免误修改。
        self._cpu_static_experts = frozenset(cpu_static_experts)

        # ------------------------------- 初始化 CPU 侧 staging bundle 状态 -------------------------------
        # 当前主线路径不再单独维护 staging bundle，因此这里显式置为 None。
        self._cpu_stage_bundle = None

        # ------------------------------- 按当前权重模式初始化 CPU 侧原始权重缓冲区 -------------------------------
        # 当当前控制器工作在 GPTQ Marlin 模式下时，分配量化原始权重缓冲区。
        if self._mode == "gptq_marlin":
            # 分配 CPU 侧量化原始权重缓冲区。
            self._cpu_quantized_raw_buffer = self._allocate_quantized_raw_buffer()

            # 将量化原始权重缓冲区占用的字节数累计到 CPU buffer 总大小中。
            self._cpu_buffer_bytes += self._raw_quantized_nbytes(
                self._cpu_quantized_raw_buffer
            )
        else:
            # 当当前控制器工作在非量化模式下时，分配非量化原始权重缓冲区。
            self._cpu_unquantized_raw_buffer = self._allocate_unquantized_raw_buffer()

            # 将非量化原始权重缓冲区占用的字节数累计到 CPU buffer 总大小中。
            self._cpu_buffer_bytes += self._raw_unquantized_nbytes(
                self._cpu_unquantized_raw_buffer
            )

        # ------------------------------- 按计划将 CPU 常驻 expert 预热到内存中 -------------------------------
        # 遍历计划中指定的 CPU 常驻 expert，并将其对应的静态 bundle 预先物化到 CPU 内存中。
        for expert_id in cpu_static_experts:
            # 以 eager 模式物化当前 expert 的 CPU 静态 bundle。
            self._materialize_cpu_static_bundle(expert_id, eager=True)

        # ------------------------------- 在持有 CPU 缓存或缓冲区时输出初始化日志 -------------------------------
        # 当当前控制器确实持有 CPU 静态 bundle 或 staging bundle 时，输出初始化日志。
        if self._cpu_static_bundles or self._cpu_stage_bundle is not None:
            logger.info(
                "Initialized fixed CPU expert pool: layer=%s static=%d/%d staging=%s "
                "cpu_bytes=%.2f MiB burst_min_tokens=%d",
                self.layer_key,
                len(self._cpu_static_bundles),
                len(self._cpu_static_experts),
                False,
                self._cpu_buffer_bytes / (1 << 20),
                self.prefill_burst_min_tokens,
            )

    def _allocate_source_bundle(self) -> ExpertBundle:
        # ------------------------------- 为 checkpoint 形态 expert 分配临时 source bundle 容器 -------------------------------
        # 这个 bundle 只用于承接 safetensors 中的 gate/up/down 原始字段，随后会被一次性预处理成
        # runtime-ready CPU static bundle。长期驻留的 CPU static mirror 不再保存 checkpoint 形态，
        # 而是保存可直接写入推理 resident slot 的 runtime 格式。
        #
        # 因此这里保持普通 CPU 内存即可；真正希望加速 H2D 的是预处理后的 runtime-ready bundle。
        cpu_tensor_args = {"device": "cpu"}

        # 预声明 source bundle 内部保存的张量字典。
        tensors: dict[str, torch.Tensor]

        # ------------------------------- 在量化路径下分配 gate/up/down 的量化权重张量 -------------------------------
        # 当当前控制器工作在 GPTQ Marlin 模式下时，为 gate、up、down 三路分别分配 qweight、scales 和 qzeros 张量。
        if self._mode == "gptq_marlin":
            tensors = {
                # 为 gate 投影的量化权重分配 CPU 张量。
                "slot.gate_proj.qweight": torch.empty(
                    (
                        self.layer.hidden_size // self.pack_factor,
                        self.layer.intermediate_size_per_partition,
                    ),
                    dtype=torch.int32,
                    **cpu_tensor_args,
                ),

                # 为 gate 投影的 scale 张量分配 CPU 张量。
                "slot.gate_proj.scales": torch.empty(
                    (self.layer.num_groups_w13, self.layer.intermediate_size_per_partition),
                    dtype=self.layer.w13_scales.dtype,
                    **cpu_tensor_args,
                ),

                # 为 gate 投影的 qzeros 张量分配 CPU 张量。
                "slot.gate_proj.qzeros": torch.empty(
                    (
                        self.layer.num_groups_w13,
                        self.layer.intermediate_size_per_partition // self.pack_factor,
                    ),
                    dtype=self.layer.w13_qzeros.dtype,
                    **cpu_tensor_args,
                ),

                # 为 up 投影的量化权重分配 CPU 张量。
                "slot.up_proj.qweight": torch.empty(
                    (
                        self.layer.hidden_size // self.pack_factor,
                        self.layer.intermediate_size_per_partition,
                    ),
                    dtype=torch.int32,
                    **cpu_tensor_args,
                ),

                # 为 up 投影的 scale 张量分配 CPU 张量。
                "slot.up_proj.scales": torch.empty(
                    (
                        self.layer.num_groups_w13,
                        self.layer.intermediate_size_per_partition
                    ),
                    dtype=self.layer.w13_scales.dtype,
                    **cpu_tensor_args,
                ),

                # 为 up 投影的 qzeros 张量分配 CPU 张量。
                "slot.up_proj.qzeros": torch.empty(
                    (
                        self.layer.num_groups_w13,
                        self.layer.intermediate_size_per_partition // self.pack_factor,
                    ),
                    dtype=self.layer.w13_qzeros.dtype,
                    **cpu_tensor_args,
                ),

                # 为 down 投影的量化权重分配 CPU 张量。
                "slot.down_proj.qweight": torch.empty(
                    (
                        self.layer.intermediate_size_per_partition // self.pack_factor,
                        self.layer.hidden_size,
                    ),
                    dtype=torch.int32,
                    **cpu_tensor_args,
                ),

                # 为 down 投影的 scale 张量分配 CPU 张量。
                "slot.down_proj.scales": torch.empty(
                    (self.layer.num_groups_w2, self.layer.hidden_size),
                    dtype=self.layer.w2_scales.dtype,
                    **cpu_tensor_args,
                ),

                # 为 down 投影的 qzeros 张量分配 CPU 张量。
                "slot.down_proj.qzeros": torch.empty(
                    (self.layer.num_groups_w2, self.layer.hidden_size // self.pack_factor),
                    dtype=self.layer.w2_qzeros.dtype,
                    **cpu_tensor_args,
                ),
            }
            if self._gptq_desc_act:
                tensors.update(
                    {
                        "slot.gate_proj.g_idx": torch.empty(
                            (self.layer.hidden_size,),
                            dtype=self.layer.w13_g_idx.dtype,
                            **cpu_tensor_args,
                        ),
                        "slot.up_proj.g_idx": torch.empty(
                            (self.layer.hidden_size,),
                            dtype=self.layer.w13_g_idx.dtype,
                            **cpu_tensor_args,
                        ),
                        "slot.down_proj.g_idx": torch.empty(
                            (self.layer.intermediate_size_per_partition,),
                            dtype=self.layer.w2_g_idx.dtype,
                            **cpu_tensor_args,
                        ),
                    }
                )
        else:
            # ------------------------------- 在非量化路径下分配 gate/up/down 的原始权重张量 -------------------------------
            # 当当前控制器工作在非量化模式下时，只需要为 gate、up、down 三路分配原始 weight 张量。
            tensors = {
                # 为 gate 投影的原始权重分配 CPU 张量。
                "slot.gate_proj.weight": torch.empty(
                    (
                        self.layer.intermediate_size_per_partition,
                        self.layer.hidden_size
                    ),
                    dtype=self.layer.w13_weight.dtype,
                    **cpu_tensor_args,
                ),

                # 为 up 投影的原始权重分配 CPU 张量。
                "slot.up_proj.weight": torch.empty(
                    (
                        self.layer.intermediate_size_per_partition,
                        self.layer.hidden_size
                    ),
                    dtype=self.layer.w13_weight.dtype,
                    **cpu_tensor_args,
                ),

                # 为 down 投影的原始权重分配 CPU 张量。
                "slot.down_proj.weight": torch.empty(
                    (
                        self.layer.hidden_size,
                        self.layer.intermediate_size_per_partition
                    ),
                    dtype=self.layer.w2_weight.dtype,
                    **cpu_tensor_args,
                ),
            }

        # ------------------------------- 将张量字典封装为统一的 ExpertBundle 返回 -------------------------------
        # 基于当前张量字典构造统一的 ExpertBundle，供 CPU static 与动态加载路径复用。
        return ExpertBundle(
            tensors=tensors,
            nbytes=bundle_nbytes(tensors),
            pinned=False,
        )

    def _allocate_quantized_raw_buffer(self) -> _RawExpertWeights:
        # ------------------------------- 为量化路径分配单 expert 级 CPU 原始权重缓冲区 -------------------------------
        # 构造 CPU 侧张量分配参数；当平台支持时启用 pinned memory 以加速后续 H2D 传输。
        cpu_tensor_args = {"device": "cpu", "pin_memory": self._use_pinned_cpu}

        # 返回面向量化路径的单 expert 原始权重缓冲区对象。
        return _RawExpertWeights(
            # 为 gate/up 合并后的 w13 量化权重分配 CPU 缓冲区。
            w13_qweight=torch.empty(
                (
                    1,
                    self.layer.hidden_size // self.pack_factor,
                    2 * self.layer.intermediate_size_per_partition,
                ),
                dtype=torch.int32,
                **cpu_tensor_args,
            ),
            # 为 down 路径的 w2 量化权重分配 CPU 缓冲区。
            w2_qweight=torch.empty(
                (
                    1,
                    self.layer.intermediate_size_per_partition // self.pack_factor,
                    self.layer.hidden_size,
                ),
                dtype=torch.int32,
                **cpu_tensor_args,
            ),
            # 为 gate/up 合并后的 w13 scale 张量分配 CPU 缓冲区。
            w13_scales=torch.empty(
                (
                    1,
                    self.layer.num_groups_w13,
                    2 * self.layer.intermediate_size_per_partition
                ),
                dtype=self.layer.w13_scales.dtype,
                **cpu_tensor_args,
            ),
            # 为 down 路径的 w2 scale 张量分配 CPU 缓冲区。
            w2_scales=torch.empty(
                (
                    1,
                    self.layer.num_groups_w2,
                    self.layer.hidden_size
                ),
                dtype=self.layer.w2_scales.dtype,
                **cpu_tensor_args,
            ),
            # 为 gate/up 合并后的 w13 qzeros 张量分配 CPU 缓冲区。
            w13_qzeros=torch.empty(
                (
                    1,
                    self.layer.num_groups_w13,
                    (2 * self.layer.intermediate_size_per_partition) // self.pack_factor,
                ),
                dtype=self.layer.w13_qzeros.dtype,
                **cpu_tensor_args,
            ),
            # 为 down 路径的 w2 qzeros 张量分配 CPU 缓冲区。
            w2_qzeros=torch.empty(
                (
                    1,
                    self.layer.num_groups_w2,
                    self.layer.hidden_size // self.pack_factor
                ),
                dtype=self.layer.w2_qzeros.dtype,
                **cpu_tensor_args,
            ),
            w13_g_idx=(
                torch.empty(
                    (1, self.layer.hidden_size),
                    dtype=self.layer.w13_g_idx.dtype,
                    **cpu_tensor_args,
                )
                if self._gptq_desc_act
                else None
            ),
            w2_g_idx=(
                torch.empty(
                    (1, self.layer.intermediate_size_per_partition),
                    dtype=self.layer.w2_g_idx.dtype,
                    **cpu_tensor_args,
                )
                if self._gptq_desc_act
                else None
            ),
        )

    def _allocate_unquantized_raw_buffer(self) -> _RawUnquantizedExpertWeights:
        # ------------------------------- 为非量化路径分配单 expert 级 CPU 原始权重缓冲区 -------------------------------
        # 构造 CPU 侧张量分配参数；当平台支持时启用 pinned memory 以加速后续 H2D 传输。
        cpu_tensor_args = {"device": "cpu", "pin_memory": self._use_pinned_cpu}

        # 判断当前 MoE 路径是否采用 act-and-mul 形式，从而决定 w13 的上游输出维度。
        is_act_and_mul = bool(self.layer.moe_config.is_act_and_mul)

        # 根据是否启用 act-and-mul，计算 w13 合并权重的输出维度。
        w13_up_dim = (
            2 * self.layer.intermediate_size_per_partition
            if is_act_and_mul
            else self.layer.intermediate_size_per_partition
        )

        # 返回面向非量化路径的单 expert 原始权重缓冲区对象。
        return _RawUnquantizedExpertWeights(
            # 为 gate/up 合并后的 w13 权重分配 CPU 缓冲区。
            w13_weight=torch.empty(
                (
                    1,
                    w13_up_dim,
                    self.layer.hidden_size
                ),
                dtype=self.layer.w13_weight.dtype,
                **cpu_tensor_args,
            ),
            # 为 down 路径的 w2 权重分配 CPU 缓冲区。
            w2_weight=torch.empty(
                (
                    1,
                    self.layer.hidden_size,
                    self.layer.intermediate_size_per_partition,
                ),
                dtype=self.layer.w2_weight.dtype,
                **cpu_tensor_args,
            ),
        )

    def _install_mapping(self, expert_id: int, slot: int) -> None:
        # 记录该 slot 之前驻留的是哪个 expert。
        previous_global = self._slot_to_global[slot]
        with torch.no_grad():
            # 若该 slot 原本已有 expert，则先把旧 expert 从 expert_map 中标记为未驻留。
            if previous_global >= 0:
                self.layer._expert_map[previous_global] = -1
                self._evictions += 1
            # 再把新 expert 挂到该 slot 上。
            self.layer._expert_map[expert_id] = slot
        # 更新反向索引表，表示这个 slot 现在装着新 expert。
        self._slot_to_global[slot] = expert_id

    @staticmethod
    def _raw_quantized_nbytes(raw: _RawExpertWeights) -> int:
        # ------------------------------- 统计量化原始权重缓冲区的总字节数 -------------------------------
        # 对量化原始权重缓冲区中的各个张量按“元素数乘以单元素字节数”求和，得到总字节占用。
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                raw.w13_qweight,
                raw.w2_qweight,
                raw.w13_scales,
                raw.w2_scales,
                raw.w13_qzeros,
                raw.w2_qzeros,
                raw.w13_g_idx,
                raw.w2_g_idx,
            )
            if tensor is not None
        )

    @staticmethod
    def _raw_unquantized_nbytes(raw: _RawUnquantizedExpertWeights) -> int:
        # 统计非量化 raw buffer 总字节数。
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (raw.w13_weight, raw.w2_weight)
        )


def maybe_enable_tiered_moe_cache(model: nn.Module, cfie_config: Any) -> None:
    # ------------------------------- 读取 tiered MoE cache 计划并判断是否启用 -------------------------------
    # 从配置对象中读取启动期 planner 注入的 tiered MoE cache 计划。
    plan = get_moe_tiered_cache_plan(cfie_config)

    # 当计划不存在，或计划中显式标记为未启用时，直接返回，不挂载任何控制器。
    if not plan or not bool(plan.get("enabled", False)):
        return

    # ------------------------------- 打印 tiered MoE cache 挂载前的总体计划信息 -------------------------------
    # 输出当前 tiered MoE cache 计划的关键摘要信息，便于确认模型类型、GPU 槽位、burst 槽位和 CPU 槽位等参数。
    logger.info(
        "Preparing tiered MoE expert cache attachment: model_type=%s "
        "gpu_slots/layer=%d prefill_burst_slots=%d cpu_slots/layer=%d model=%s",
        plan.get("model_type", ""),
        int(plan.get("gpu_slots_per_layer", 0)),
        int(plan.get("prefill_burst_slots", 0)),
        int(plan.get("cpu_slots_per_layer", 0)),
        plan.get("model_path", ""),
    )

    # ------------------------------- 初始化冷数据读取存储与全局统计状态 -------------------------------
    # 基于计划中的模型路径创建 safetensors 专家存储，用于后续按层、按专家读取冷数据。
    expert_store = SafetensorExpertStore(plan["model_path"])

    # 读取计划中配置的共享 prefill burst pool 槽位数。
    prefill_burst_slots = int(plan.get("prefill_burst_slots", 0))

    # 记录最终真正挂载 tiered cache controller 的层数。
    enabled_layers = 0

    # 记录模型中被标记为应该启用 tiered cache 的 FusedMoE 层数。
    marked_layers = 0

    # 尽量在同一模型内复用一个共享 prefill burst pool，以减少额外显存占用。
    shared_prefill_burst_pool: SharedPrefillBurstPool | None = None

    # ------------------------------- 遍历模型模块并为目标 FusedMoE 层挂载控制器 -------------------------------
    # 遍历模型中的全部子模块，逐个查找需要启用 tiered cache 的 FusedMoE 层。
    for module in model.modules():
        # 仅处理 FusedMoE 类型的层。
        if not isinstance(module, FusedMoE):
            continue

        # 仅处理前面初始化阶段已被标记启用 CFIE tiered cache 的层。
        if not getattr(module, "_cfie_tiered_cache_enabled", False):
            continue

        # 统计当前模型中被标记需要启用 tiered cache 的层数。
        marked_layers += 1

        # 当前层默认不绑定任何 prefill burst pool。
        layer_prefill_burst_pool: SharedPrefillBurstPool | None = None

        # ------------------------------- 按需为当前层分配或复用共享 prefill burst pool -------------------------------
        # 当计划中启用了 prefill burst pool 时，优先尝试为当前层复用全局共享池。
        if prefill_burst_slots > 0:
            # 当全局共享 burst pool 尚未创建时，基于当前层构造一份模板化共享池。
            if shared_prefill_burst_pool is None:
                shared_prefill_burst_pool = SharedPrefillBurstPool(
                    template_layer=module,
                    num_slots=prefill_burst_slots,
                )

            # 当当前层与共享 burst pool 的形状兼容时，直接复用同一个共享池。
            if shared_prefill_burst_pool.supports_layer(module):
                layer_prefill_burst_pool = shared_prefill_burst_pool
            else:
                # 当当前层与共享 burst pool 不兼容时，保留告警，并让该层仅走 resident prepare 路径。
                logger.warning(
                    "Skipping shared prefill burst pool on incompatible MoE layer: %s",
                    module.layer_name,
                )

        # ------------------------------- 为当前 FusedMoE 层挂载 tiered cache controller -------------------------------
        # 基于当前层、全局计划、专家存储和可选的 prefill burst pool 构造分层专家缓存控制器。
        module._cfie_tiered_cache_controller = LayerTieredExpertCacheController(
            layer=module,
            plan=plan,
            expert_store=expert_store,
            prefill_burst_pool=layer_prefill_burst_pool,
        )

        # 统计当前已成功挂载控制器的层数。
        enabled_layers += 1

    # ------------------------------- 校验目标层数与实际挂载层数是否一致 -------------------------------
    # 当存在被标记的目标层，但实际成功挂载的层数与标记层数不一致时，直接失败并报错。
    if marked_layers and enabled_layers != marked_layers:
        raise RuntimeError(
            f"Tiered MoE cache expected {marked_layers} layers, but attached only "
            f"{enabled_layers}"
        )

    # ------------------------------- 在全部挂载成功后打印最终启用日志 -------------------------------
    # 当至少有一层成功启用了 tiered MoE cache 时，打印最终启用结果。
    if enabled_layers > 0:
        logger.info(
            "Enabled tiered MoE expert cache on %d layers (plan=%s)",
            enabled_layers,
            PLAN_KEY,
        )
