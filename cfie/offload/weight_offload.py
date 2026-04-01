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
    w13_qweight: torch.Tensor
    w2_qweight: torch.Tensor
    w13_scales: torch.Tensor
    w2_scales: torch.Tensor
    w13_qzeros: torch.Tensor
    w2_qzeros: torch.Tensor


@dataclass(slots=True)
class _RawUnquantizedExpertWeights:
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor


@dataclass(slots=True)
class _PrefillBurstExecutionStats:
    resident_hits: int = 0
    cpu_hits: int = 0
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
        self._target = target
        self.bind(base_layer, expert_map, num_slots)

    def bind(
        self,
        base_layer: FusedMoE,
        expert_map: torch.Tensor,
        num_slots: int,
    ) -> None:
        self._base_layer = base_layer
        self._expert_map = expert_map
        self.local_num_experts = num_slots
        self.global_num_experts = base_layer.global_num_experts
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
            self.w13_weight = self._target.w13_qweight
            self.w2_weight = self._target.w2_qweight
        else:
            self.w13_weight = self._target.w13_weight
            self.w2_weight = self._target.w2_weight

    @property
    def expert_map(self) -> torch.Tensor:
        return self._expert_map

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_layer, name)


class SharedPrefillBurstPool:
    """Shared single-layer execution buffer for large prefill MoE batches."""

    def __init__(self, template_layer: FusedMoE, num_slots: int) -> None:
        if num_slots <= 0:
            raise ValueError("Shared prefill burst pool requires a positive slot count")

        self.num_slots = int(num_slots)
        self.global_num_experts = template_layer.global_num_experts
        self.layer_name = template_layer.layer_name
        self._busy = False

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

        self._expert_map = torch.full(
            (self.global_num_experts,),
            -1,
            dtype=torch.int32,
            device=expert_map_device,
        )
        self._execution_layer = _PrefillBurstExecutionLayer(
            base_layer=template_layer,
            target=self,
            expert_map=self._expert_map,
            num_slots=self.num_slots,
        )
        logger.info(
            "Initialized shared prefill burst pool: layer=%s mode=%s slots=%d bytes=%.2f MiB",
            template_layer.layer_name,
            self._mode,
            self.num_slots,
            self.nbytes / (1 << 20),
        )

    @property
    def nbytes(self) -> int:
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
            tensors = (self.w13_weight, self.w2_weight)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def supports_layer(self, layer: FusedMoE) -> bool:
        if layer.global_num_experts != self.global_num_experts:
            return False
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
        if self._busy:
            raise RuntimeError(
                f"{controller.layer_key}: shared prefill burst pool is already in use"
            )

        requested = controller._unique_experts(topk_ids)
        if len(requested) > self.num_slots:
            raise RuntimeError(
                f"{controller.layer_key}: requested {len(requested)} experts but only "
                f"{self.num_slots} burst slots are available"
            )

        self._busy = True
        try:
            self._expert_map.fill_(-1)
            stats = controller._populate_prefill_burst_pool(self, requested)
            for slot, expert_id in enumerate(requested):
                self._expert_map[expert_id] = slot
            self._execution_layer.bind(
                base_layer=controller.layer,
                expert_map=self._expert_map,
                num_slots=self.num_slots,
            )
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
            return controller.quant_method.apply(
                layer=self._execution_layer,
                x=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
            )
        finally:
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
        if layer._expert_map is None:
            raise ValueError("Tiered MoE cache requires a global-to-local expert map")

        self.layer = layer
        self.plan = plan
        self.expert_store = expert_store
        self.prefill_burst_pool = prefill_burst_pool
        self.layer_key = layer.layer_name
        self.quant_method = layer.quant_method
        self.num_slots = layer.local_num_experts
        self._step = 0
        self._access_count = [0] * layer.global_num_experts
        self._last_used_step = [0] * layer.global_num_experts
        self._slot_to_global = [-1] * self.num_slots
        self._total_loads = 0
        self._cpu_hits = 0
        self._nvme_loads = 0
        self._evictions = 0
        self._mode: str
        self._use_pinned_cpu = is_pin_memory_available()
        self._cpu_static_bundles: dict[int, ExpertBundle] = {}
        self._cpu_stage_bundle: ExpertBundle | None = None
        self._cpu_static_experts: frozenset[int] = frozenset()
        self._cpu_buffer_bytes = 0
        self._cpu_quantized_raw_buffer: _RawExpertWeights | None = None
        self._cpu_unquantized_raw_buffer: _RawUnquantizedExpertWeights | None = None
        configured_burst_min_tokens = int(self.plan.get("prefill_burst_min_tokens", 0))
        self._prefill_burst_min_tokens = (
            configured_burst_min_tokens
            if configured_burst_min_tokens > 0
            else max(
                DEFAULT_PREFILL_BURST_MIN_TOKENS,
                self.num_slots * DEFAULT_PREFILL_BURST_TOKENS_PER_GPU_SLOT,
            )
        )

        if isinstance(layer.quant_method, GPTQMarlinMoEMethod):
            if layer.quant_method.quant_config.desc_act:
                raise ValueError(
                    "Tiered MoE cache does not support GPTQ desc_act=True"
                )
            if layer.quant_method.input_dtype is not None:
                raise ValueError(
                    "Tiered MoE cache currently expects standard bf16/fp16 activations"
                )
            self._mode = "gptq_marlin"
            self.device = layer.w13_qweight.device
            self.pack_factor = layer.quant_method.quant_config.pack_factor
            self.num_bits = layer.quant_method.quant_config.quant_type.size_bits
            self.group_size = layer.quant_method.quant_config.group_size
            self.is_a_8bit = False
        elif isinstance(layer.quant_method, UnquantizedFusedMoEMethod):
            if layer.quant_method.moe.has_bias:
                raise ValueError(
                    "Tiered MoE cache currently does not support biased unquantized MoE"
                )
            if layer.quant_method.unquantized_backend != UnquantizedMoeBackend.TRITON:
                raise TypeError(
                    "Tiered MoE cache currently only supports TRITON unquantized MoE"
                )
            self._mode = "unquantized"
            self.device = layer.w13_weight.device
        else:
            raise TypeError(
                "Tiered MoE cache currently supports GPTQMarlinMoEMethod "
                "and UnquantizedFusedMoEMethod only"
            )

        for global_expert in range(layer.global_num_experts):
            slot = int(layer._expert_map[global_expert].item())
            if 0 <= slot < self.num_slots:
                self._slot_to_global[slot] = global_expert

        self._init_cpu_fixed_pools()

    @property
    def slot_to_global(self) -> tuple[int, ...]:
        return tuple(self._slot_to_global)

    @property
    def prefill_burst_capacity(self) -> int:
        if self.prefill_burst_pool is None:
            return 0
        return self.prefill_burst_pool.num_slots

    @property
    def prefill_burst_min_tokens(self) -> int:
        if self.prefill_burst_pool is None:
            return 0
        return self._prefill_burst_min_tokens

    def can_run_prefill_burst(self, num_unique_experts: int, num_tokens: int) -> bool:
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
        if self.prefill_burst_pool is None:
            raise RuntimeError(f"{self.layer_key}: prefill burst pool is not available")
        return self.prefill_burst_pool.execute(
            controller=self,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts_input=shared_experts_input,
        )

    def prepare(self, topk_ids: torch.Tensor) -> None:
        requested = self._unique_experts(topk_ids)
        if not requested:
            return
        if len(requested) > self.num_slots:
            raise RuntimeError(
                f"{self.layer_key}: requested {len(requested)} experts but only "
                f"{self.num_slots} GPU slots are available"
            )

        self._step += 1
        for expert_id in requested:
            self._access_count[expert_id] += 1
            self._last_used_step[expert_id] = self._step

        protected = set(requested)
        for expert_id in requested:
            if self._is_resident(expert_id):
                continue
            slot = self._choose_victim_slot(protected)
            self._load_expert_into_slot(expert_id, slot)

    def _unique_experts(self, topk_ids: torch.Tensor) -> list[int]:
        if topk_ids.numel() == 0:
            return []
        unique_ids = torch.unique(topk_ids.detach()).to(device="cpu")
        return [int(expert_id) for expert_id in unique_ids.tolist()]

    def _is_resident(self, expert_id: int) -> bool:
        return int(self.layer._expert_map[expert_id].item()) >= 0

    def _choose_victim_slot(self, protected: set[int]) -> int:
        for slot, global_expert in enumerate(self._slot_to_global):
            if global_expert < 0:
                return slot

        candidates = [
            slot
            for slot, global_expert in enumerate(self._slot_to_global)
            if global_expert not in protected
        ]
        if not candidates:
            raise RuntimeError(
                f"{self.layer_key}: no evictable slot available for requested experts"
            )

        return min(
            candidates,
            key=lambda slot: (self._eviction_score(self._slot_to_global[slot]), slot),
        )

    def _eviction_score(self, expert_id: int) -> tuple[int, int, int]:
        in_cpu_static = expert_id in self._cpu_static_experts
        keep_bonus = 4096 if not in_cpu_static else 0
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
        if expert_id not in self._cpu_static_experts:
            return None

        bundle = self._cpu_static_bundles.get(expert_id)
        if bundle is not None:
            return bundle

        bundle = self._allocate_source_bundle()
        self.expert_store.copy_expert_into(
            self.layer_key,
            expert_id,
            bundle.tensors,
            skip_suffixes=("g_idx",),
        )
        self._cpu_static_bundles[expert_id] = bundle
        self._cpu_buffer_bytes += bundle.nbytes
        materialized = len(self._cpu_static_bundles)
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
        bundle = self._materialize_cpu_static_bundle(expert_id)
        source = "cpu_static"
        if bundle is None:
            if self._cpu_stage_bundle is None:
                raise RuntimeError(
                    f"{self.layer_key}: no preallocated CPU staging buffer available "
                    f"for expert {expert_id}"
                )
            self.expert_store.copy_expert_into(
                self.layer_key,
                expert_id,
                self._cpu_stage_bundle.tensors,
                skip_suffixes=("g_idx",),
            )
            bundle = self._cpu_stage_bundle
            source = "nvme_stage"
        return bundle, source

    def _load_expert_into_slot(self, expert_id: int, slot: int) -> None:
        bundle, source = self._get_source_bundle(expert_id)
        if source == "nvme_stage":
            self._nvme_loads += 1
        else:
            self._cpu_hits += 1

        self._write_expert_bundle(slot, bundle, self.layer)
        self._install_mapping(expert_id, slot)
        self._total_loads += 1
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
        stats = _PrefillBurstExecutionStats()
        for slot, expert_id in enumerate(requested):
            if self._is_resident(expert_id):
                self._copy_resident_expert_to_target(expert_id, slot, pool)
                stats.resident_hits += 1
                continue
            bundle, source = self._get_source_bundle(expert_id)
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
        src_slot = int(self.layer._expert_map[expert_id].item())
        if src_slot < 0:
            raise RuntimeError(
                f"{self.layer_key}: expert {expert_id} is not resident in GPU slots"
            )
        with torch.no_grad():
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
            target.w13_weight[slot].copy_(self.layer.w13_weight[src_slot])
            target.w2_weight[slot].copy_(self.layer.w2_weight[src_slot])

    def _write_expert_bundle(self, slot: int, bundle: ExpertBundle, target: Any) -> None:
        if self._mode == "gptq_marlin":
            raw = self._assemble_raw_weights(bundle)
            raw_gpu = self._move_raw_weights_to_device(raw)
            repacked_w13 = ops.gptq_marlin_moe_repack(
                raw_gpu.w13_qweight,
                self._empty_perm(raw_gpu.w13_qweight.device),
                raw_gpu.w13_qweight.shape[1] * self.pack_factor,
                raw_gpu.w13_qweight.shape[2],
                self.num_bits,
                is_a_8bit=self.is_a_8bit,
            )
            repacked_w2 = ops.gptq_marlin_moe_repack(
                raw_gpu.w2_qweight,
                self._empty_perm(raw_gpu.w2_qweight.device),
                raw_gpu.w2_qweight.shape[1] * self.pack_factor,
                raw_gpu.w2_qweight.shape[2],
                self.num_bits,
                is_a_8bit=self.is_a_8bit,
            )
            permuted_w13_scales = marlin_moe_permute_scales(
                s=raw_gpu.w13_scales,
                size_k=self.layer.intermediate_size_per_partition,
                size_n=raw_gpu.w13_scales.shape[2],
                group_size=self.group_size,
                is_a_8bit=self.is_a_8bit,
            )
            permuted_w2_scales = marlin_moe_permute_scales(
                s=raw_gpu.w2_scales,
                size_k=raw_gpu.w2_scales.shape[1]
                * (self.group_size if self.group_size != -1 else self.pack_factor),
                size_n=raw_gpu.w2_scales.shape[2],
                group_size=self.group_size,
                is_a_8bit=self.is_a_8bit,
            )
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

        raw = self._assemble_unquantized_weights(bundle)
        raw_gpu = self._move_unquantized_weights_to_device(raw)
        runtime_w13, runtime_w2 = convert_to_unquantized_kernel_format(
            self.quant_method.unquantized_backend,
            layer=self.layer,
            w13_weight=raw_gpu.w13_weight,
            w2_weight=raw_gpu.w2_weight,
        )
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
            target.w13_qweight[slot].copy_(repacked_w13)
            target.w2_qweight[slot].copy_(repacked_w2)
            target.w13_scales[slot].copy_(permuted_w13_scales)
            target.w2_scales[slot].copy_(permuted_w2_scales)
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
            target.w13_weight[slot].copy_(runtime_w13)
            target.w2_weight[slot].copy_(runtime_w2)

    def _assemble_raw_weights(self, bundle: ExpertBundle) -> _RawExpertWeights:
        tensors = bundle.tensors
        w13_cols = 2 * self.layer.intermediate_size_per_partition
        w13_qzeros_cols = w13_cols // self.pack_factor
        w13_half_cols = self.layer.intermediate_size_per_partition
        w13_half_qzeros_cols = w13_half_cols // self.pack_factor
        if self._cpu_quantized_raw_buffer is None:
            raise RuntimeError(f"{self.layer_key}: quantized CPU raw buffer is missing")
        raw = self._cpu_quantized_raw_buffer
        seen_fields: set[tuple[str, str]] = set()

        for relative_name, tensor in tensors.items():
            _, suffix = relative_name.split(".", 1)
            proj_name, field_name = suffix.split(".", 1)
            cpu_tensor = tensor if tensor.device.type == "cpu" else tensor.to(device="cpu")
            if proj_name == "gate_proj":
                self._copy_merged_half(
                    field_name,
                    cpu_tensor,
                    raw,
                    offset=0,
                    qzeros_offset=0,
                )
                seen_fields.add((proj_name, field_name))
            elif proj_name == "up_proj":
                self._copy_merged_half(
                    field_name,
                    cpu_tensor,
                    raw,
                    offset=w13_half_cols,
                    qzeros_offset=w13_half_qzeros_cols,
                )
                seen_fields.add((proj_name, field_name))
            elif proj_name == "down_proj":
                self._copy_direct(field_name, cpu_tensor, raw)
                seen_fields.add((proj_name, field_name))

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
        if field_name == "qweight":
            raw.w13_qweight[0, :, offset : offset + tensor.shape[-1]].copy_(tensor)
        elif field_name == "scales":
            raw.w13_scales[0, :, offset : offset + tensor.shape[-1]].copy_(tensor)
        elif field_name == "qzeros":
            raw.w13_qzeros[0, :, qzeros_offset : qzeros_offset + tensor.shape[-1]].copy_(
                tensor
            )

    def _copy_direct(
        self, field_name: str, tensor: torch.Tensor, raw: _RawExpertWeights
    ) -> None:
        if field_name == "qweight":
            raw.w2_qweight[0].copy_(tensor)
        elif field_name == "scales":
            raw.w2_scales[0].copy_(tensor)
        elif field_name == "qzeros":
            raw.w2_qzeros[0].copy_(tensor)

    def _move_raw_weights_to_device(self, raw: _RawExpertWeights) -> _RawExpertWeights:
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
        tensors = bundle.tensors
        is_act_and_mul = bool(self.layer.moe_config.is_act_and_mul)
        w13_up_dim = (
            2 * self.layer.intermediate_size_per_partition
            if is_act_and_mul
            else self.layer.intermediate_size_per_partition
        )
        half_dim = self.layer.intermediate_size_per_partition
        if self._cpu_unquantized_raw_buffer is None:
            raise RuntimeError(
                f"{self.layer_key}: unquantized CPU raw buffer is missing"
            )
        raw = self._cpu_unquantized_raw_buffer
        seen_fields: set[tuple[str, str]] = set()

        for relative_name, tensor in tensors.items():
            _, suffix = relative_name.split(".", 1)
            proj_name, field_name = suffix.split(".", 1)
            if field_name != "weight":
                continue
            cpu_tensor = tensor if tensor.device.type == "cpu" else tensor.to(device="cpu")
            if proj_name == "gate_proj":
                raw.w13_weight[0, :half_dim].copy_(cpu_tensor)
                seen_fields.add((proj_name, field_name))
            elif proj_name == "up_proj":
                raw.w13_weight[0, half_dim : half_dim + cpu_tensor.shape[0]].copy_(
                    cpu_tensor
                )
                seen_fields.add((proj_name, field_name))
            elif proj_name == "down_proj":
                raw.w2_weight[0].copy_(cpu_tensor)
                seen_fields.add((proj_name, field_name))

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
        return _RawUnquantizedExpertWeights(
            w13_weight=raw.w13_weight.to(device=self.device, non_blocking=True),
            w2_weight=raw.w2_weight.to(device=self.device, non_blocking=True),
        )

    def _empty_perm(self, device: torch.device) -> torch.Tensor:
        return torch.empty((1, 0), dtype=torch.int32, device=device)

    def _init_cpu_fixed_pools(self) -> None:
        cpu_static_experts = tuple(
            int(expert_id)
            for expert_id in self.plan.get("initial_cpu_experts", ())
            if int(expert_id) < self.layer.global_num_experts
        )
        self._cpu_static_experts = frozenset(cpu_static_experts)

        if int(self.plan.get("staging_bytes", 0)) > 0:
            self._cpu_stage_bundle = self._allocate_source_bundle()
            self._cpu_buffer_bytes += self._cpu_stage_bundle.nbytes

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

        for expert_id in cpu_static_experts:
            self._materialize_cpu_static_bundle(expert_id, eager=True)

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
        cpu_tensor_args = {"device": "cpu", "pin_memory": self._use_pinned_cpu}
        tensors: dict[str, torch.Tensor]
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
        return ExpertBundle(
            tensors=tensors,
            nbytes=bundle_nbytes(tensors),
            pinned=self._use_pinned_cpu,
        )

    def _allocate_quantized_raw_buffer(self) -> _RawExpertWeights:
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
        previous_global = self._slot_to_global[slot]
        with torch.no_grad():
            if previous_global >= 0:
                self.layer._expert_map[previous_global] = -1
                self._evictions += 1
            self.layer._expert_map[expert_id] = slot
        self._slot_to_global[slot] = expert_id

    @staticmethod
    def _raw_quantized_nbytes(raw: _RawExpertWeights) -> int:
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
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (raw.w13_weight, raw.w2_weight)
        )


def maybe_enable_tiered_moe_cache(model: nn.Module, cfie_config: Any) -> None:
    plan = get_moe_tiered_cache_plan(cfie_config)
    if not plan or not bool(plan.get("enabled", False)):
        return

    logger.info(
        "Preparing tiered MoE expert cache attachment: model_type=%s "
        "gpu_slots/layer=%d prefill_burst_slots=%d cpu_slots/layer=%d model=%s",
        plan.get("model_type", ""),
        int(plan.get("gpu_slots_per_layer", 0)),
        int(plan.get("prefill_burst_slots", 0)),
        int(plan.get("cpu_slots_per_layer", 0)),
        plan.get("model_path", ""),
    )
    expert_store = SafetensorExpertStore(plan["model_path"])
    prefill_burst_slots = int(plan.get("prefill_burst_slots", 0))
    enabled_layers = 0
    marked_layers = 0
    shared_prefill_burst_pool: SharedPrefillBurstPool | None = None

    for module in model.modules():
        if not isinstance(module, FusedMoE):
            continue
        if not getattr(module, "_cfie_tiered_cache_enabled", False):
            continue
        marked_layers += 1
        layer_prefill_burst_pool: SharedPrefillBurstPool | None = None
        if prefill_burst_slots > 0:
            if shared_prefill_burst_pool is None:
                shared_prefill_burst_pool = SharedPrefillBurstPool(
                    template_layer=module,
                    num_slots=prefill_burst_slots,
                )
            if shared_prefill_burst_pool.supports_layer(module):
                layer_prefill_burst_pool = shared_prefill_burst_pool
            else:
                logger.warning(
                    "Skipping shared prefill burst pool on incompatible MoE layer: %s",
                    module.layer_name,
                )
        module._cfie_tiered_cache_controller = LayerTieredExpertCacheController(
            layer=module,
            plan=plan,
            expert_store=expert_store,
            prefill_burst_pool=layer_prefill_burst_pool,
        )
        enabled_layers += 1

    if marked_layers and enabled_layers != marked_layers:
        raise RuntimeError(
            f"Tiered MoE cache expected {marked_layers} layers, but attached only "
            f"{enabled_layers}"
        )

    if enabled_layers > 0:
        logger.info(
            "Enabled tiered MoE expert cache on %d layers (plan=%s)",
            enabled_layers,
            PLAN_KEY,
        )
