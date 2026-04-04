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
    """Lightweight layer proxy bound to the shared prefill burst tensors."""

    def __init__(
        self,
        base_layer: FusedMoE,
        target: Any,
        expert_map: torch.Tensor,
        num_slots: int,
    ) -> None:
        # target 是共享 burst pool 本身，里面持有真正的临时执行张量。
        self._target = target
        # 初始化时立即把 proxy 绑定到当前 layer 和专家映射上。
        self.bind(base_layer, expert_map, num_slots)

    def bind(
        self,
        base_layer: FusedMoE,
        expert_map: torch.Tensor,
        num_slots: int,
    ) -> None:
        # 保存真实的基础层，用于把其它属性透传回原始 FusedMoE。
        self._base_layer = base_layer
        # 用 burst pool 自己的 expert_map 覆盖原层的 resident expert_map。
        self._expert_map = expert_map
        # local_num_experts 改成 burst pool 的临时 slot 数。
        self.local_num_experts = num_slots
        # global_num_experts 保持和原始层一致，方便 router/topk_ids 继续工作。
        self.global_num_experts = base_layer.global_num_experts
        # 量化路径下，把执行时读取的权重张量全部改绑到 burst pool 的临时张量。
        if hasattr(self._target, "w13_qweight"):
            self.w13_qweight = self._target.w13_qweight
            self.w2_qweight = self._target.w2_qweight
            self.w13_scales = self._target.w13_scales
            self.w2_scales = self._target.w2_scales
            self.w13_qzeros = self._target.w13_qzeros
            self.w2_qzeros = self._target.w2_qzeros
            self.w13_g_idx = self._target.w13_g_idx
            self.w2_g_idx = self._target.w2_g_idx
            self.w13_g_idx_sort_indices = self._target.w13_g_idx_sort_indices
            self.w2_g_idx_sort_indices = self._target.w2_g_idx_sort_indices
            # 某些通用路径通过 w13_weight/w2_weight 访问权重，这里也别名到量化张量上。
            self.w13_weight = self._target.w13_qweight
            self.w2_weight = self._target.w2_qweight
        else:
            # 非量化路径则直接改绑到 burst pool 的 dense 权重张量。
            self.w13_weight = self._target.w13_weight
            self.w2_weight = self._target.w2_weight

    @property
    def expert_map(self) -> torch.Tensor:
        # 运行时执行 kernel 时读取的是 burst pool 的临时专家映射。
        return self._expert_map

    def __getattr__(self, name: str) -> Any:
        # 其余配置/超参统一回退到原始 FusedMoE 层上。
        return getattr(self._base_layer, name)


class SharedPrefillBurstPool:
    """Shared single-layer execution buffer for large prefill MoE batches."""

    def __init__(self, template_layer: FusedMoE, num_slots: int) -> None:
        # burst pool 至少要有 1 个临时 slot，否则没有意义。
        if num_slots <= 0:
            raise ValueError("Shared prefill burst pool requires a positive slot count")

        # 记录 burst pool 的临时 slot 容量。
        self.num_slots = int(num_slots)
        # 全局专家数要与模板层一致，便于沿用相同的 topk expert id。
        self.global_num_experts = template_layer.global_num_experts
        # 记录模板层名，仅用于日志输出。
        self.layer_name = template_layer.layer_name
        # 共享池一次只允许一个 layer/request 使用，避免被并发覆盖。
        self._busy = False

        # ----------------- 按量化模式分配临时执行张量 -----------------
        # GPTQ Marlin 路径需要为所有量化权重与辅助索引都分配一份 burst 张量。
        if isinstance(template_layer.quant_method, GPTQMarlinMoEMethod):
            self._mode = "gptq_marlin"
            self.w13_qweight = torch.empty(
                (self.num_slots, *template_layer.w13_qweight.shape[1:]),
                dtype=template_layer.w13_qweight.dtype,
                device=template_layer.w13_qweight.device,
            )
            self.w2_qweight = torch.empty(
                (self.num_slots, *template_layer.w2_qweight.shape[1:]),
                dtype=template_layer.w2_qweight.dtype,
                device=template_layer.w2_qweight.device,
            )
            self.w13_scales = torch.empty(
                (self.num_slots, *template_layer.w13_scales.shape[1:]),
                dtype=template_layer.w13_scales.dtype,
                device=template_layer.w13_scales.device,
            )
            self.w2_scales = torch.empty(
                (self.num_slots, *template_layer.w2_scales.shape[1:]),
                dtype=template_layer.w2_scales.dtype,
                device=template_layer.w2_scales.device,
            )
            self.w13_qzeros = torch.empty(
                (self.num_slots, *template_layer.w13_qzeros.shape[1:]),
                dtype=template_layer.w13_qzeros.dtype,
                device=template_layer.w13_qzeros.device,
            )
            self.w2_qzeros = torch.empty(
                (self.num_slots, *template_layer.w2_qzeros.shape[1:]),
                dtype=template_layer.w2_qzeros.dtype,
                device=template_layer.w2_qzeros.device,
            )
            self.w13_g_idx = torch.empty(
                (self.num_slots, *template_layer.w13_g_idx.shape[1:]),
                dtype=template_layer.w13_g_idx.dtype,
                device=template_layer.w13_g_idx.device,
            )
            self.w2_g_idx = torch.empty(
                (self.num_slots, *template_layer.w2_g_idx.shape[1:]),
                dtype=template_layer.w2_g_idx.dtype,
                device=template_layer.w2_g_idx.device,
            )
            self.w13_g_idx_sort_indices = torch.empty(
                (self.num_slots, *template_layer.w13_g_idx_sort_indices.shape[1:]),
                dtype=template_layer.w13_g_idx_sort_indices.dtype,
                device=template_layer.w13_g_idx_sort_indices.device,
            )
            self.w2_g_idx_sort_indices = torch.empty(
                (self.num_slots, *template_layer.w2_g_idx_sort_indices.shape[1:]),
                dtype=template_layer.w2_g_idx_sort_indices.dtype,
                device=template_layer.w2_g_idx_sort_indices.device,
            )
            expert_map_device = template_layer.w13_qweight.device
        # 非量化路径只需要准备 runtime 直接消费的 w13/w2 权重。
        elif isinstance(template_layer.quant_method, UnquantizedFusedMoEMethod):
            self._mode = "unquantized"
            self.w13_weight = torch.empty(
                (self.num_slots, *template_layer.w13_weight.shape[1:]),
                dtype=template_layer.w13_weight.dtype,
                device=template_layer.w13_weight.device,
            )
            self.w2_weight = torch.empty(
                (self.num_slots, *template_layer.w2_weight.shape[1:]),
                dtype=template_layer.w2_weight.dtype,
                device=template_layer.w2_weight.device,
            )
            expert_map_device = template_layer.w13_weight.device
        else:
            raise TypeError(
                "Shared prefill burst pool only supports GPTQMarlinMoEMethod "
                "and UnquantizedFusedMoEMethod"
            )

        # 为 burst pool 单独准备一张“全局 expert -> burst slot”的映射表。
        self._expert_map = torch.full(
            (self.global_num_experts,),
            -1,
            dtype=torch.int32,
            device=expert_map_device,
        )
        # 再构造一个轻量代理层，把 kernel 入口重定向到 burst pool 上。
        self._execution_layer = _PrefillBurstExecutionLayer(
            base_layer=template_layer,
            target=self,
            expert_map=self._expert_map,
            num_slots=self.num_slots,
        )
        # 记录 burst pool 初始化完成后的占用情况。
        logger.info(
            "Initialized shared prefill burst pool: layer=%s mode=%s slots=%d bytes=%.2f MiB",
            template_layer.layer_name,
            self._mode,
            self.num_slots,
            self.nbytes / (1 << 20),
        )

    @property
    def nbytes(self) -> int:
        # 量化路径需要把所有临时量化张量都纳入统计。
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
            # 非量化路径只统计 w13/w2 两块权重张量。
            tensors = (self.w13_weight, self.w2_weight)
        # 逐张量按元素数 * 单元素字节数求和。
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def supports_layer(self, layer: FusedMoE) -> bool:
        # 专家总数不一致时，topk expert id 语义不同，不能复用同一个 burst pool。
        if layer.global_num_experts != self.global_num_experts:
            return False
        # 量化路径下，还要确保所有核心张量形状都兼容。
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
        # 非量化路径则校验 w13/w2 形状即可。
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
        # 共享 burst pool 设计成串行使用，防止两个请求同时覆盖临时张量。
        if self._busy:
            raise RuntimeError(
                f"{controller.layer_key}: shared prefill burst pool is already in use"
            )

        # 先把本次请求真正涉及到的唯一 expert 集合提取出来。
        requested = controller._unique_experts(topk_ids)
        # 若本次唯一 experts 超过 burst pool 容量，则不能走 burst 路径。
        if len(requested) > self.num_slots:
            raise RuntimeError(
                f"{controller.layer_key}: requested {len(requested)} experts but only "
                f"{self.num_slots} burst slots are available"
            )

        # ----------------- 装填 burst pool 并临时执行 -----------------
        self._busy = True
        try:
            # 每次执行前先清空上一轮残留的 burst expert 映射。
            self._expert_map.fill_(-1)
            # 按 requested experts 把 resident/CPU/NVMe 来源的数据写入 burst pool。
            stats = controller._populate_prefill_burst_pool(self, requested)
            # 依次登记“全局 expert -> burst slot”关系，供 quant_method.apply 使用。
            for slot, expert_id in enumerate(requested):
                self._expert_map[expert_id] = slot
            # 把轻量执行层重新绑定到当前控制器对应的 layer 上。
            self._execution_layer.bind(
                base_layer=controller.layer,
                expert_map=self._expert_map,
                num_slots=self.num_slots,
            )
            # 记录 burst 执行的数据来源命中情况。
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
            # 真正执行时直接把 burst proxy layer 交给量化方法。
            return controller.quant_method.apply(
                layer=self._execution_layer,
                x=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
            )
        finally:
            # 无论成功还是失败，都要释放共享池占用标记。
            self._busy = False


class LayerTieredExpertCacheController:
    """Per-layer GPU/CPU/NVMe cache controller for routed experts."""

    def __init__(
        self,
        layer: FusedMoE,
        plan: dict[str, Any],
        expert_store: SafetensorExpertStore,
        prefill_burst_pool: SharedPrefillBurstPool | None = None,
    ) -> None:
        # CFIE tiered cache 依赖 FusedMoE 预先建立好的全局 expert -> 本地 slot 映射表。
        if layer._expert_map is None:
            raise ValueError("Tiered MoE cache requires a global-to-local expert map")

        # ----------------- 保存控制器核心依赖 -----------------
        # 记录当前控制器绑定的 FusedMoE 层。
        self.layer = layer
        # 保存启动期规划器生成的 plan 字典。
        self.plan = plan
        # expert_store 负责从 safetensors/NVMe 读取冷专家权重。
        self.expert_store = expert_store
        # 共享 burst pool 用于 prefill 大批量时的临时执行路径。
        self.prefill_burst_pool = prefill_burst_pool
        # layer_key 作为 expert_store 与日志的层级主键。
        self.layer_key = layer.layer_name
        # quant_method 决定写回 GPU slot 时需要的转换逻辑。
        self.quant_method = layer.quant_method
        # resident GPU slot 数等于当前 FusedMoE 在 CFIE 路径下实际创建的 local experts 数。
        self.num_slots = layer.local_num_experts
        # step 用于给近似 LRU 驱逐打时间戳。
        self._step = 0
        # access_count 统计每个 expert 被请求的次数。
        self._access_count = [0] * layer.global_num_experts
        # last_used_step 记录每个 expert 最近一次命中的逻辑时间。
        self._last_used_step = [0] * layer.global_num_experts
        # slot_to_global 是 resident GPU slots 的反向索引表。
        self._slot_to_global = [-1] * self.num_slots
        # 以下指标仅用于日志和调试统计。
        self._total_loads = 0
        self._cpu_hits = 0
        self._nvme_loads = 0
        self._evictions = 0
        # _mode 标识当前层走量化专家还是非量化专家加载链路。
        self._mode: str
        # 若平台允许，则优先用 pinned CPU memory 作为 staging/static cache，以加快 H2D。
        self._use_pinned_cpu = is_pin_memory_available()
        # _cpu_static_bundles 保存已物化到 CPU 内存中的静态专家 bundle。
        self._cpu_static_bundles: dict[int, ExpertBundle] = {}
        # _cpu_stage_bundle 是一块可复用的 staging 区，用于从 NVMe 读入冷专家。
        self._cpu_stage_bundle: ExpertBundle | None = None
        # _cpu_static_experts 记录 plan 指定的 CPU 常驻专家集合。
        self._cpu_static_experts: frozenset[int] = frozenset()
        # 统计 controller 持有的所有 CPU buffer 大小，便于日志观察。
        self._cpu_buffer_bytes = 0
        # 量化路径在 CPU 侧还需要一块 raw buffer，用于把 gate/up/down 重组回 kernel 需要的格式。
        self._cpu_quantized_raw_buffer: _RawExpertWeights | None = None
        # 非量化路径也需要一块 raw buffer，用于把 gate/up/down 合成 runtime w13/w2。
        self._cpu_unquantized_raw_buffer: _RawUnquantizedExpertWeights | None = None
        # 允许通过 plan 显式覆盖 burst 的最小 token 阈值。
        configured_burst_min_tokens = int(self.plan.get("prefill_burst_min_tokens", 0))
        # 否则按“至少 8 个 token，且每个 resident slot 对应 4 个 token”的经验值计算。
        self._prefill_burst_min_tokens = (
            configured_burst_min_tokens
            if configured_burst_min_tokens > 0
            else max(
                DEFAULT_PREFILL_BURST_MIN_TOKENS,
                self.num_slots * DEFAULT_PREFILL_BURST_TOKENS_PER_GPU_SLOT,
            )
        )

        # ----------------- 识别当前层的权重模式并校验约束 -----------------
        if isinstance(layer.quant_method, GPTQMarlinMoEMethod):
            # desc_act=True 会改变 runtime 行为，当前动态加载链路未适配。
            if layer.quant_method.quant_config.desc_act:
                raise ValueError(
                    "Tiered MoE cache does not support GPTQ desc_act=True"
                )
            # 当前实现假设激活输入仍是标准 bf16/fp16，不支持额外 input_dtype 改写。
            if layer.quant_method.input_dtype is not None:
                raise ValueError(
                    "Tiered MoE cache currently expects standard bf16/fp16 activations"
                )
            # GPTQ Marlin 路径下，后续需要 repack/permute，所以记录相关量化超参。
            self._mode = "gptq_marlin"
            self.device = layer.w13_qweight.device
            self.pack_factor = layer.quant_method.quant_config.pack_factor
            self.num_bits = layer.quant_method.quant_config.quant_type.size_bits
            self.group_size = layer.quant_method.quant_config.group_size
            self.is_a_8bit = False
        elif isinstance(layer.quant_method, UnquantizedFusedMoEMethod):
            # 当前非量化动态加载路径未实现 bias 分支。
            if layer.quant_method.moe.has_bias:
                raise ValueError(
                    "Tiered MoE cache currently does not support biased unquantized MoE"
                )
            # 非量化路径目前只支持 Triton backend。
            if layer.quant_method.unquantized_backend != UnquantizedMoeBackend.TRITON:
                raise TypeError(
                    "Tiered MoE cache currently only supports TRITON unquantized MoE"
                )
            # 记录非量化模式与 GPU 目标设备。
            self._mode = "unquantized"
            self.device = layer.w13_weight.device
        else:
            raise TypeError(
                "Tiered MoE cache currently supports GPTQMarlinMoEMethod "
                "and UnquantizedFusedMoEMethod only"
            )

        # 根据 layer._expert_map 反推当前 resident GPU slots 里分别装着哪个全局 expert。
        for global_expert in range(layer.global_num_experts):
            slot = int(layer._expert_map[global_expert].item())
            if 0 <= slot < self.num_slots:
                self._slot_to_global[slot] = global_expert

        # 初始化 CPU static experts、staging bundle 与 raw buffer。
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
        # 逐个检查本次需要的 experts，缺谁补谁。
        for expert_id in requested:
            # 已经常驻在 GPU resident slots 中的 expert 直接跳过。
            if self._is_resident(expert_id):
                continue
            # 为缺失 expert 选一个可替换 slot。
            slot = self._choose_victim_slot(protected)
            # 再把 expert 从 CPU/NVMe 写入该 GPU slot。
            self._load_expert_into_slot(expert_id, slot)

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
        # 非 CPU static expert 没有物化资格，直接返回 None。
        if expert_id not in self._cpu_static_experts:
            return None

        # 已经物化过的 CPU bundle 直接复用。
        bundle = self._cpu_static_bundles.get(expert_id)
        if bundle is not None:
            return bundle

        # ----------------- 首次把 static expert 从 expert store 拉到 CPU 内存 -----------------
        bundle = self._allocate_source_bundle()
        self.expert_store.copy_expert_into(
            self.layer_key,
            expert_id,
            bundle.tensors,
            skip_suffixes=("g_idx",),
        )
        # 放入静态缓存字典，并累计 CPU 占用。
        self._cpu_static_bundles[expert_id] = bundle
        self._cpu_buffer_bytes += bundle.nbytes
        materialized = len(self._cpu_static_bundles)
        # 仅在前几次或每 32 个静态专家时打日志，避免刷屏。
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
        return bundle

    def _get_source_bundle(self, expert_id: int) -> tuple[ExpertBundle, str]:
        # 先尝试命中 CPU static experts。
        bundle = self._materialize_cpu_static_bundle(expert_id)
        source = "cpu_static"
        # static 未命中时，退化成“从 NVMe 读到 staging bundle”。
        if bundle is None:
            if self._cpu_stage_bundle is None:
                raise RuntimeError(
                    f"{self.layer_key}: no preallocated CPU staging buffer available "
                    f"for expert {expert_id}"
                )
            # 这里不会长期保存该 expert，只是把它临时读到 staging 区。
            self.expert_store.copy_expert_into(
                self.layer_key,
                expert_id,
                self._cpu_stage_bundle.tensors,
                skip_suffixes=("g_idx",),
            )
            bundle = self._cpu_stage_bundle
            source = "nvme_stage"
        # 返回 bundle 以及来源标签，供统计与日志使用。
        return bundle, source

    def _load_expert_into_slot(self, expert_id: int, slot: int) -> None:
        # 先确定该 expert 当前是从 CPU static 命中，还是走 NVMe staging 读取。
        bundle, source = self._get_source_bundle(expert_id)
        # 统计来源命中。
        if source == "nvme_stage":
            self._nvme_loads += 1
        else:
            self._cpu_hits += 1

        # 把来源 bundle 按当前量化模式转换后写进目标 GPU resident slot。
        self._write_expert_bundle(slot, bundle, self.layer)
        # 覆盖 resident 映射表；旧 expert 若存在则在这里被驱逐。
        self._install_mapping(expert_id, slot)
        # 增加累计加载计数。
        self._total_loads += 1
        # 仅在前几次或每 100 次动态换入时打日志。
        if self._total_loads <= 8 or self._total_loads % 100 == 0:
            logger.info(
                "Tiered MoE cache event: layer=%s load=%d source=%s expert=%d slot=%d "
                "cpu_hits=%d nvme_loads=%d evictions=%d",
                self.layer_key,
                self._total_loads,
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
        # ----------------- 量化路径：bundle -> CPU raw -> GPU raw -> Marlin runtime 格式 -----------------
        if self._mode == "gptq_marlin":
            # 先把 gate/up/down 三份量化张量重组回单 expert 的原始结构。
            raw = self._assemble_raw_weights(bundle)
            # 再把 raw 权重异步搬到目标 GPU 设备。
            raw_gpu = self._move_raw_weights_to_device(raw)
            # qweight 需要重新 repack 成 Marlin MoE kernel 可直接消费的布局。
            repacked_w13 = ops.gptq_marlin_moe_repack(
                raw_gpu.w13_qweight,
                self._empty_perm(raw_gpu.w13_qweight.device),
                raw_gpu.w13_qweight.shape[1] * self.pack_factor,
                raw_gpu.w13_qweight.shape[2],
                self.num_bits,
                is_a_8bit=self.is_a_8bit,
            )
            # w2 也要独立做一遍 repack。
            repacked_w2 = ops.gptq_marlin_moe_repack(
                raw_gpu.w2_qweight,
                self._empty_perm(raw_gpu.w2_qweight.device),
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
        # ----------------- 预计算 w13 合并布局的列偏移 -----------------
        tensors = bundle.tensors
        w13_cols = 2 * self.layer.intermediate_size_per_partition
        w13_qzeros_cols = w13_cols // self.pack_factor
        w13_half_cols = self.layer.intermediate_size_per_partition
        w13_half_qzeros_cols = w13_half_cols // self.pack_factor
        # 量化路径必须有预分配好的 CPU raw buffer。
        if self._cpu_quantized_raw_buffer is None:
            raise RuntimeError(f"{self.layer_key}: quantized CPU raw buffer is missing")
        # raw buffer 会被重复复用，避免每次动态加载都重新分配。
        raw = self._cpu_quantized_raw_buffer
        # seen_fields 用于确保 gate/up/down 的关键张量都被正确拼齐。
        seen_fields: set[tuple[str, str]] = set()

        # ----------------- 逐张量把 bundle 中的 gate/up/down 权重拼进 raw buffer -----------------
        for relative_name, tensor in tensors.items():
            _, suffix = relative_name.split(".", 1)
            proj_name, field_name = suffix.split(".", 1)
            # 为了后续 H2D，一律先保证源张量位于 CPU。
            cpu_tensor = tensor if tensor.device.type == "cpu" else tensor.to(device="cpu")
            # gate_proj 填到 w13 的前半段。
            if proj_name == "gate_proj":
                self._copy_merged_half(
                    field_name,
                    cpu_tensor,
                    raw,
                    offset=0,
                    qzeros_offset=0,
                )
                seen_fields.add((proj_name, field_name))
            # up_proj 填到 w13 的后半段。
            elif proj_name == "up_proj":
                self._copy_merged_half(
                    field_name,
                    cpu_tensor,
                    raw,
                    offset=w13_half_cols,
                    qzeros_offset=w13_half_qzeros_cols,
                )
                seen_fields.add((proj_name, field_name))
            # down_proj 直接对应 w2。
            elif proj_name == "down_proj":
                self._copy_direct(field_name, cpu_tensor, raw)
                seen_fields.add((proj_name, field_name))

        # ----------------- 校验动态加载所需字段是否齐全 -----------------
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
        missing_fields = required_fields - seen_fields
        if missing_fields:
            raise KeyError(
                f"{self.layer_key}: missing expert tensors for dynamic load: "
                f"{sorted(missing_fields)}"
            )

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
        # qweight/scales 按列偏移写入 w13 的对应半边。
        if field_name == "qweight":
            raw.w13_qweight[0, :, offset : offset + tensor.shape[-1]].copy_(tensor)
        elif field_name == "scales":
            raw.w13_scales[0, :, offset : offset + tensor.shape[-1]].copy_(tensor)
        # qzeros 的列数按 pack_factor 压缩，因此使用单独的 qzeros_offset。
        elif field_name == "qzeros":
            raw.w13_qzeros[0, :, qzeros_offset : qzeros_offset + tensor.shape[-1]].copy_(
                tensor
            )

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

    def _move_raw_weights_to_device(self, raw: _RawExpertWeights) -> _RawExpertWeights:
        # 把 CPU raw buffer 异步搬到目标 GPU，后续 repack/permute 在设备侧完成。
        return _RawExpertWeights(
            w13_qweight=raw.w13_qweight.to(device=self.device, non_blocking=True),
            w2_qweight=raw.w2_qweight.to(device=self.device, non_blocking=True),
            w13_scales=raw.w13_scales.to(device=self.device, non_blocking=True),
            w2_scales=raw.w2_scales.to(device=self.device, non_blocking=True),
            w13_qzeros=raw.w13_qzeros.to(device=self.device, non_blocking=True),
            w2_qzeros=raw.w2_qzeros.to(device=self.device, non_blocking=True),
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
                raw.w13_weight[0, half_dim : half_dim + cpu_tensor.shape[0]].copy_(
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
        # ----------------- 根据 plan 确认 CPU static experts 集合 -----------------
        cpu_static_experts = tuple(
            int(expert_id)
            for expert_id in self.plan.get("initial_cpu_experts", ())
            if int(expert_id) < self.layer.global_num_experts
        )
        self._cpu_static_experts = frozenset(cpu_static_experts)

        # ----------------- 先初始化 staging bundle -----------------
        if int(self.plan.get("staging_bytes", 0)) > 0:
            self._cpu_stage_bundle = self._allocate_source_bundle()
            self._cpu_buffer_bytes += self._cpu_stage_bundle.nbytes

        # ----------------- 再初始化 raw buffer -----------------
        if self._mode == "gptq_marlin":
            self._cpu_quantized_raw_buffer = self._allocate_quantized_raw_buffer()
            self._cpu_buffer_bytes += self._raw_quantized_nbytes(
                self._cpu_quantized_raw_buffer
            )
        else:
            self._cpu_unquantized_raw_buffer = self._allocate_unquantized_raw_buffer()
            self._cpu_buffer_bytes += self._raw_unquantized_nbytes(
                self._cpu_unquantized_raw_buffer
            )

        # ----------------- 最后按计划把 CPU static experts 预热到内存 -----------------
        for expert_id in cpu_static_experts:
            self._materialize_cpu_static_bundle(expert_id, eager=True)

        # 若控制器确实持有 CPU 侧缓存/缓冲区，则打印初始化日志。
        if self._cpu_static_bundles or self._cpu_stage_bundle is not None:
            logger.info(
                "Initialized fixed CPU expert pool: layer=%s static=%d/%d staging=%s "
                "cpu_bytes=%.2f MiB burst_min_tokens=%d",
                self.layer_key,
                len(self._cpu_static_bundles),
                len(self._cpu_static_experts),
                self._cpu_stage_bundle is not None,
                self._cpu_buffer_bytes / (1 << 20),
                self.prefill_burst_min_tokens,
            )

    def _allocate_source_bundle(self) -> ExpertBundle:
        # CPU source bundle 既可作为 static expert 存储，也可作为 NVMe staging 区复用。
        cpu_tensor_args = {"device": "cpu", "pin_memory": self._use_pinned_cpu}
        tensors: dict[str, torch.Tensor]
        # 量化路径需要为 gate/up/down 的 qweight/scales/qzeros 分别预分配张量。
        if self._mode == "gptq_marlin":
            tensors = {
                "slot.gate_proj.qweight": torch.empty(
                    (
                        self.layer.hidden_size // self.pack_factor,
                        self.layer.intermediate_size_per_partition,
                    ),
                    dtype=torch.int32,
                    **cpu_tensor_args,
                ),
                "slot.gate_proj.scales": torch.empty(
                    (self.layer.num_groups_w13, self.layer.intermediate_size_per_partition),
                    dtype=self.layer.w13_scales.dtype,
                    **cpu_tensor_args,
                ),
                "slot.gate_proj.qzeros": torch.empty(
                    (
                        self.layer.num_groups_w13,
                        self.layer.intermediate_size_per_partition // self.pack_factor,
                    ),
                    dtype=self.layer.w13_qzeros.dtype,
                    **cpu_tensor_args,
                ),
                "slot.up_proj.qweight": torch.empty(
                    (
                        self.layer.hidden_size // self.pack_factor,
                        self.layer.intermediate_size_per_partition,
                    ),
                    dtype=torch.int32,
                    **cpu_tensor_args,
                ),
                "slot.up_proj.scales": torch.empty(
                    (self.layer.num_groups_w13, self.layer.intermediate_size_per_partition),
                    dtype=self.layer.w13_scales.dtype,
                    **cpu_tensor_args,
                ),
                "slot.up_proj.qzeros": torch.empty(
                    (
                        self.layer.num_groups_w13,
                        self.layer.intermediate_size_per_partition // self.pack_factor,
                    ),
                    dtype=self.layer.w13_qzeros.dtype,
                    **cpu_tensor_args,
                ),
                "slot.down_proj.qweight": torch.empty(
                    (
                        self.layer.intermediate_size_per_partition // self.pack_factor,
                        self.layer.hidden_size,
                    ),
                    dtype=torch.int32,
                    **cpu_tensor_args,
                ),
                "slot.down_proj.scales": torch.empty(
                    (self.layer.num_groups_w2, self.layer.hidden_size),
                    dtype=self.layer.w2_scales.dtype,
                    **cpu_tensor_args,
                ),
                "slot.down_proj.qzeros": torch.empty(
                    (self.layer.num_groups_w2, self.layer.hidden_size // self.pack_factor),
                    dtype=self.layer.w2_qzeros.dtype,
                    **cpu_tensor_args,
                ),
            }
        else:
            # 非量化路径则只需要原始 weight 张量。
            tensors = {
                "slot.gate_proj.weight": torch.empty(
                    (self.layer.intermediate_size_per_partition, self.layer.hidden_size),
                    dtype=self.layer.w13_weight.dtype,
                    **cpu_tensor_args,
                ),
                "slot.up_proj.weight": torch.empty(
                    (self.layer.intermediate_size_per_partition, self.layer.hidden_size),
                    dtype=self.layer.w13_weight.dtype,
                    **cpu_tensor_args,
                ),
                "slot.down_proj.weight": torch.empty(
                    (self.layer.hidden_size, self.layer.intermediate_size_per_partition),
                    dtype=self.layer.w2_weight.dtype,
                    **cpu_tensor_args,
                ),
            }
        # 返回一个统一的 ExpertBundle，供 static/staging 路径复用。
        return ExpertBundle(
            tensors=tensors,
            nbytes=bundle_nbytes(tensors),
            pinned=self._use_pinned_cpu,
        )

    def _allocate_quantized_raw_buffer(self) -> _RawExpertWeights:
        # 量化路径 raw buffer 的形状直接对齐单 expert 在运行时需要的合并格式。
        cpu_tensor_args = {"device": "cpu", "pin_memory": self._use_pinned_cpu}
        return _RawExpertWeights(
            w13_qweight=torch.empty(
                (
                    1,
                    self.layer.hidden_size // self.pack_factor,
                    2 * self.layer.intermediate_size_per_partition,
                ),
                dtype=torch.int32,
                **cpu_tensor_args,
            ),
            w2_qweight=torch.empty(
                (
                    1,
                    self.layer.intermediate_size_per_partition // self.pack_factor,
                    self.layer.hidden_size,
                ),
                dtype=torch.int32,
                **cpu_tensor_args,
            ),
            w13_scales=torch.empty(
                (1, self.layer.num_groups_w13, 2 * self.layer.intermediate_size_per_partition),
                dtype=self.layer.w13_scales.dtype,
                **cpu_tensor_args,
            ),
            w2_scales=torch.empty(
                (1, self.layer.num_groups_w2, self.layer.hidden_size),
                dtype=self.layer.w2_scales.dtype,
                **cpu_tensor_args,
            ),
            w13_qzeros=torch.empty(
                (
                    1,
                    self.layer.num_groups_w13,
                    (2 * self.layer.intermediate_size_per_partition) // self.pack_factor,
                ),
                dtype=self.layer.w13_qzeros.dtype,
                **cpu_tensor_args,
            ),
            w2_qzeros=torch.empty(
                (1, self.layer.num_groups_w2, self.layer.hidden_size // self.pack_factor),
                dtype=self.layer.w2_qzeros.dtype,
                **cpu_tensor_args,
            ),
        )

    def _allocate_unquantized_raw_buffer(self) -> _RawUnquantizedExpertWeights:
        # 非量化 raw buffer 同样只为“单个 expert 的合并形式”分配 1 个 batch 维。
        cpu_tensor_args = {"device": "cpu", "pin_memory": self._use_pinned_cpu}
        is_act_and_mul = bool(self.layer.moe_config.is_act_and_mul)
        w13_up_dim = (
            2 * self.layer.intermediate_size_per_partition
            if is_act_and_mul
            else self.layer.intermediate_size_per_partition
        )
        return _RawUnquantizedExpertWeights(
            w13_weight=torch.empty(
                (1, w13_up_dim, self.layer.hidden_size),
                dtype=self.layer.w13_weight.dtype,
                **cpu_tensor_args,
            ),
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
        # 统计量化 raw buffer 总字节数。
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                raw.w13_qweight,
                raw.w2_qweight,
                raw.w13_scales,
                raw.w2_scales,
                raw.w13_qzeros,
                raw.w2_qzeros,
            )
        )

    @staticmethod
    def _raw_unquantized_nbytes(raw: _RawUnquantizedExpertWeights) -> int:
        # 统计非量化 raw buffer 总字节数。
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (raw.w13_weight, raw.w2_weight)
        )


def maybe_enable_tiered_moe_cache(model: nn.Module, cfie_config: Any) -> None:
    # 从总配置里读取启动期 planner 注入的 tiered cache plan。
    plan = get_moe_tiered_cache_plan(cfie_config)
    # plan 不存在或 planner 判断 disabled 时，直接不挂载任何控制器。
    if not plan or not bool(plan.get("enabled", False)):
        return

    # 打印准备挂载 tiered cache 的总览信息。
    logger.info(
        "Preparing tiered MoE expert cache attachment: model_type=%s "
        "gpu_slots/layer=%d prefill_burst_slots=%d cpu_slots/layer=%d model=%s",
        plan.get("model_type", ""),
        int(plan.get("gpu_slots_per_layer", 0)),
        int(plan.get("prefill_burst_slots", 0)),
        int(plan.get("cpu_slots_per_layer", 0)),
        plan.get("model_path", ""),
    )
    # SafetensorExpertStore 负责后续所有按层/按 expert 的冷数据读取。
    expert_store = SafetensorExpertStore(plan["model_path"])
    # 读取计划中的共享 burst pool 容量。
    prefill_burst_slots = int(plan.get("prefill_burst_slots", 0))
    # enabled_layers 统计最终真正挂上控制器的层数。
    enabled_layers = 0
    # marked_layers 统计模型里被标记为“应该启用 tiered cache”的 FusedMoE 层数。
    marked_layers = 0
    # 同一模型里尽量复用一个共享 burst pool，减少额外显存占用。
    shared_prefill_burst_pool: SharedPrefillBurstPool | None = None

    # ----------------- 遍历模型模块，为每个目标 FusedMoE 层挂控制器 -----------------
    for module in model.modules():
        # 只处理 FusedMoE 层。
        if not isinstance(module, FusedMoE):
            continue
        # 只处理前面在 layer 初始化时被标记启用 CFIE tiered cache 的层。
        if not getattr(module, "_cfie_tiered_cache_enabled", False):
            continue
        marked_layers += 1
        layer_prefill_burst_pool: SharedPrefillBurstPool | None = None
        # 若 plan 启用了 burst pool，则优先复用全局共享池。
        if prefill_burst_slots > 0:
            if shared_prefill_burst_pool is None:
                shared_prefill_burst_pool = SharedPrefillBurstPool(
                    template_layer=module,
                    num_slots=prefill_burst_slots,
                )
            # 当前层形状与共享池兼容时，直接挂同一个 burst pool。
            if shared_prefill_burst_pool.supports_layer(module):
                layer_prefill_burst_pool = shared_prefill_burst_pool
            else:
                # 不兼容时保留警告并让该层只走 resident prepare 路径。
                logger.warning(
                    "Skipping shared prefill burst pool on incompatible MoE layer: %s",
                    module.layer_name,
                )
        # 为当前层挂上真正的 per-layer tiered cache controller。
        module._cfie_tiered_cache_controller = LayerTieredExpertCacheController(
            layer=module,
            plan=plan,
            expert_store=expert_store,
            prefill_burst_pool=layer_prefill_burst_pool,
        )
        enabled_layers += 1

    # 若理论应启用的层数和实际挂载数不一致，直接 fail fast。
    if marked_layers and enabled_layers != marked_layers:
        raise RuntimeError(
            f"Tiered MoE cache expected {marked_layers} layers, but attached only "
            f"{enabled_layers}"
        )

    # 所有目标层都成功挂载后，打印最终启用日志。
    if enabled_layers > 0:
        logger.info(
            "Enabled tiered MoE expert cache on %d layers (plan=%s)",
            enabled_layers,
            PLAN_KEY,
        )
