"""Bounded representative tensor executor for the CFIE training runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import weakref

import torch
import torch.nn.functional as F

from cfie_training.config import TrainingProjectConfig
from cfie_training.runtime.quantization import (
    PackedQuantizedTensor,
    dequantize_packed_range,
    quantized_linear,
    quantized_linear_from_packed_slice,
    resize_tensor,
    runtime_gptq_enabled,
)
from cfie_training.runtime.store import ParameterShardStore
from cfie_training.runtime.types import (
    BatchShape,
    LayerBucketPlan,
    ParameterShardSnapshot,
    ParameterSourceSlice,
    RepresentativeBucketRecord,
    RepresentativeExecutionSummary,
)


@dataclass(slots=True, frozen=True)
class GradientPayload:
    group_id: str
    logical_params: int
    gradient: torch.Tensor
    start_offset: int = 0


@dataclass(slots=True, frozen=True)
class RepresentativeExecutionResult:
    gradients: tuple[GradientPayload, ...]
    execution_summary: RepresentativeExecutionSummary


@dataclass(slots=True, frozen=True)
class RepresentativeBucketExecutionResult:
    gradients: tuple[GradientPayload, ...]
    bucket_record: RepresentativeBucketRecord


@dataclass(slots=True, frozen=True)
class _RepresentativeBucketGraphShape:
    sequence_length: int
    hidden_dim: int
    expert_hidden_dim: int
    active_expert_count: int
    topk: int
    attention_head_count: int
    kv_head_count: int
    head_dim: int


@dataclass(slots=True, frozen=True)
class _PackedWeightSliceSpec:
    packed: PackedQuantizedTensor
    start_offset: int
    raw_shape: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class _PackedWeightSliceBinding:
    slices: tuple[_PackedWeightSliceSpec, ...]
    resize_shape: tuple[int, ...] | None = None
    resize_mode: str | None = None
    expert_index: int | None = None
    transpose_last_two: bool = False
    tanh_scale: float = 0.0
    add_scalar: float = 0.0


@dataclass(slots=True)
class RepresentativeBucketExecutor:
    config: TrainingProjectConfig
    _compute_device: torch.device = field(init=False, repr=False)
    _quantized_execution: bool = field(init=False, repr=False)
    _quantization_bits: int = field(init=False, repr=False)
    _quantization_group_size: int = field(init=False, repr=False)
    _quantization_sym: bool = field(init=False, repr=False)
    _quantization_pack_dtype: str = field(init=False, repr=False)
    _quantization_compute_view_dtype: str = field(init=False, repr=False)
    _flat_packed_views: dict[
        int,
        tuple[weakref.ReferenceType[torch.Tensor], PackedQuantizedTensor],
    ] = field(
        init=False,
        repr=False,
    )
    _packed_weight_bindings: dict[
        int,
        tuple[weakref.ReferenceType[torch.Tensor], _PackedWeightSliceBinding],
    ] = field(
        init=False,
        repr=False,
    )

    def _configure_deterministic_cuda_runtime(self) -> None:
        # 仅在 GPU + 显式要求 deterministic CUDA 时配置确定性运行时。
        if (
            self.config.execution.compute_device != "gpu"
            or not self.config.execution.deterministic_cuda_execution
        ):
            return
        # 固定 CUBLAS workspace，减少算子非确定性。
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # 禁用 CUDA matmul 的 TF32。
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            # 若存在 cuDNN，则同时禁用 TF32 与 benchmark。
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        # 打开 PyTorch 层面的确定性算法开关。
        torch.use_deterministic_algorithms(True, warn_only=True)

    def __post_init__(self) -> None:
        # GPU 模式下先检查当前 torch 运行时是否具备 CUDA 能力。
        if self.config.execution.compute_device == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "execution.compute_device=gpu requires a CUDA-capable torch runtime"
                )
            # 再按配置启用 deterministic CUDA 约束。
            self._configure_deterministic_cuda_runtime()
            self._compute_device = torch.device("cuda")
        else:
            # CPU 模式下直接固定到 cpu 设备。
            self._compute_device = torch.device("cpu")
        # 仅在 CUDA + runtime GPTQ 开启时走量化执行路径。
        self._quantized_execution = (
            self._compute_device.type == "cuda" and runtime_gptq_enabled(self.config)
        )
        # 缓存量化相关超参数，后续执行路径直接复用。
        self._quantization_bits = self.config.runtime_quantization.bits
        self._quantization_group_size = self.config.runtime_quantization.group_size
        self._quantization_sym = self.config.runtime_quantization.sym
        self._quantization_pack_dtype = self.config.runtime_quantization.pack_dtype
        self._quantization_compute_view_dtype = (
            self.config.runtime_quantization.compute_view_dtype
        )
        # 初始化“flat tensor -> packed view”缓存。
        self._flat_packed_views = {}
        # 初始化“weight tensor -> packed slice binding”缓存。
        self._packed_weight_bindings = {}

    @property
    def compute_device(self) -> torch.device:
        # 返回当前执行器实际运行的 compute device。
        return self._compute_device

    def _full_bucket_logical_cuda_enabled(self) -> bool:
        # 仅在 CUDA + logical 物化 + full_bucket 模式下返回 True。
        return (
            self._compute_device.type == "cuda"
            and self.config.execution.trainable_shard_materialization == "logical"
            and self.config.execution.logical_cuda_execution_mode == "full_bucket"
        )

    def _stabilize_tensor(
        self,
        tensor: torch.Tensor,
        *,
        max_abs: float = 32.0,
    ) -> torch.Tensor:
        # 统一把张量转成 float32，并裁掉 NaN/Inf 与过大值。
        stabilized = torch.nan_to_num(
            tensor.to(dtype=torch.float32),
            nan=0.0,
            posinf=max_abs,
            neginf=-max_abs,
        )
        # 再把数值限制到 [-max_abs, max_abs]。
        return stabilized.clamp(min=-max_abs, max=max_abs)

    def _stabilize_bound(
        self,
        tensor: torch.Tensor,
        *,
        max_abs: float = 32.0,
    ) -> torch.Tensor:
        # 先做普通数值稳定化。
        stabilized = self._stabilize_tensor(tensor, max_abs=max_abs)
        # 若当前张量没有绑定 packed slice，则直接返回。
        binding = self._weight_binding_for_tensor(tensor)
        if binding is None:
            return stabilized
        # 若存在 binding，则把稳定化后的张量重新绑定回原 packed slice 语义。
        return self._bind_weight_slice(stabilized, binding=binding)

    def _bounded_sequence_length(self, batch: BatchShape) -> int:
        # 有 mask 时优先使用真实 token 数；否则沿用形状 token 数作为估算规模。
        effective_tokens = max(1, batch.valid_token_count)

        # full_bucket logical CUDA 模式下，优先用真实 token 规模，但受 micro-batch 和位置编码上限约束。
        if self._full_bucket_logical_cuda_enabled():
            return max(
                2,
                min(
                    effective_tokens,
                    self.config.execution.max_tokens_per_micro_batch,
                    self.config.model_spec.max_position_embeddings,
                ),
            )
        # 轻量代表性模式下，把序列长度压到更小范围以控制计算规模。
        return max(2, min(8, max(batch.samples * 2, min(effective_tokens, 8))))

    def _reset_quantized_bindings(self) -> None:
        # 每次 bucket 执行前都清空上一轮残留的 packed 绑定缓存。
        self._flat_packed_views = {}
        self._packed_weight_bindings = {}

    def _bind_flat_packed_view(
        self,
        tensor: torch.Tensor,
        packed: PackedQuantizedTensor | None,
    ) -> None:
        # 只有 packed 视图存在时才建立绑定。
        if packed is None:
            return
        # 用张量 id 记录 tensor 到 packed view 的弱引用绑定。
        self._flat_packed_views[id(tensor)] = (weakref.ref(tensor), packed)

    def _bind_weight_slice(
        self,
        tensor: torch.Tensor,
        *,
        binding: _PackedWeightSliceBinding | None,
    ) -> torch.Tensor:
        # 有 binding 时登记 tensor 到 packed slice 语义的映射。
        if binding is not None:
            self._packed_weight_bindings[id(tensor)] = (weakref.ref(tensor), binding)
        # 返回原张量，便于在表达式里内联使用。
        return tensor

    def _packed_view_for_tensor(
        self,
        tensor: torch.Tensor,
    ) -> PackedQuantizedTensor | None:
        # 先按张量 id 查 flat packed view 缓存。
        entry = self._flat_packed_views.get(id(tensor))
        if entry is None:
            return None
        # 解出弱引用和对应的 packed 视图。
        ref, packed = entry
        target = ref()
        # 弱引用仍指向当前张量时，说明绑定有效。
        if target is tensor:
            return packed
        # 目标已被回收时，顺手清理失效缓存。
        if target is None:
            del self._flat_packed_views[id(tensor)]
        return None

    def _weight_binding_for_tensor(
        self,
        tensor: torch.Tensor,
    ) -> _PackedWeightSliceBinding | None:
        # 先按张量 id 查 packed slice binding 缓存。
        entry = self._packed_weight_bindings.get(id(tensor))
        if entry is None:
            return None
        # 解出弱引用和绑定描述。
        ref, binding = entry
        target = ref()
        # 弱引用仍指向当前张量时，说明绑定有效。
        if target is tensor:
            return binding
        # 目标已被回收时，顺手清理失效缓存。
        if target is None:
            del self._packed_weight_bindings[id(tensor)]
        return None

    def _flat_packed_source(
        self,
        tensor: torch.Tensor,
    ) -> tuple[PackedQuantizedTensor, int] | None:
        # 优先尝试命中整块 flat packed view。
        packed = self._packed_view_for_tensor(tensor)
        if packed is not None:
            return packed, 0
        # 否则再尝试从更细粒度的 weight binding 反推 flat source。
        binding = self._weight_binding_for_tensor(tensor)
        if (
            binding is None
            or len(binding.slices) != 1
            or binding.resize_shape is not None
            or binding.resize_mode is not None
            or binding.expert_index is not None
            or binding.transpose_last_two
            or binding.tanh_scale != 0.0
            or binding.add_scalar != 0.0
        ):
            return None
        # 单切片、一维、无额外变换时，当前 binding 可退化成 flat packed source。
        entry = binding.slices[0]
        if len(entry.raw_shape) != 1:
            return None
        return entry.packed, entry.start_offset

    def _transpose_last_two_bound(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        # 先执行普通的后两维转置。
        transposed = tensor.transpose(-2, -1)
        # 再查当前张量是否带有 packed slice binding。
        binding = self._weight_binding_for_tensor(tensor)
        if binding is None:
            return transposed
        # 若存在 binding，则返回转置语义更新后的绑定张量。
        return self._bind_weight_slice(
            transposed,
            binding=_PackedWeightSliceBinding(
                slices=binding.slices,
                resize_shape=binding.resize_shape,
                resize_mode=binding.resize_mode,
                expert_index=binding.expert_index,
                transpose_last_two=not binding.transpose_last_two,
                tanh_scale=binding.tanh_scale,
                add_scalar=binding.add_scalar,
            ),
        )

    def _tanh_scale_bound(
        self,
        tensor: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        # 先执行 tanh 缩放变换。
        transformed = scale * torch.tanh(tensor)
        # 再同步更新 packed slice binding 中的变换元信息。
        binding = self._weight_binding_for_tensor(tensor)
        if binding is None:
            return transformed
        return self._bind_weight_slice(
            transformed,
            binding=_PackedWeightSliceBinding(
                slices=binding.slices,
                resize_shape=binding.resize_shape,
                resize_mode=binding.resize_mode,
                expert_index=binding.expert_index,
                transpose_last_two=binding.transpose_last_two,
                tanh_scale=scale,
                add_scalar=binding.add_scalar,
            ),
        )

    def _add_scalar_bound(
        self,
        tensor: torch.Tensor,
        *,
        value: float,
    ) -> torch.Tensor:
        # 先对张量施加常数偏置。
        shifted = tensor + value
        # 再同步更新 packed slice binding 中的 add_scalar 元信息。
        binding = self._weight_binding_for_tensor(tensor)
        if binding is None:
            return shifted
        return self._bind_weight_slice(
            shifted,
            binding=_PackedWeightSliceBinding(
                slices=binding.slices,
                resize_shape=binding.resize_shape,
                resize_mode=binding.resize_mode,
                expert_index=binding.expert_index,
                transpose_last_two=binding.transpose_last_two,
                tanh_scale=binding.tanh_scale,
                add_scalar=binding.add_scalar + value,
            ),
        )

    def _expert_binding_for_index(
        self,
        tensor: torch.Tensor,
        expert_index: int,
    ) -> _PackedWeightSliceBinding | None:
        # 当前张量若没有 binding 或没有 slice，则无法构造 expert binding。
        binding = self._weight_binding_for_tensor(tensor)
        if binding is None or not binding.slices:
            return None
        # 读取第一段切片的原始 shape，用于判断是否是 expert 张量。
        first_shape = binding.slices[0].raw_shape
        if len(first_shape) != 3:
            return None
        # 若 binding 已经带 3D resize_shape，则直接在 resize 后的 expert 维上索引。
        if binding.resize_shape is not None and len(binding.resize_shape) == 3:
            experts = binding.resize_shape[0]
            if expert_index < 0 or expert_index >= experts:
                return None
            return _PackedWeightSliceBinding(
                slices=binding.slices,
                resize_shape=binding.resize_shape,
                resize_mode=binding.resize_mode,
                expert_index=expert_index,
                transpose_last_two=binding.transpose_last_two,
                tanh_scale=binding.tanh_scale,
                add_scalar=binding.add_scalar,
            )
        # 否则按原始三维 expert 张量切出单个 expert 的二维切片。
        experts, rows, cols = first_shape
        if expert_index < 0 or expert_index >= experts:
            return None
        # 每个 expert 占据的连续参数个数等于 rows * cols。
        per_expert = rows * cols
        return _PackedWeightSliceBinding(
            slices=tuple(
                _PackedWeightSliceSpec(
                    packed=entry.packed,
                    start_offset=entry.start_offset + expert_index * per_expert,
                    raw_shape=(rows, cols),
                )
                for entry in binding.slices
            ),
            resize_shape=None,
            resize_mode=None,
            expert_index=None,
            transpose_last_two=binding.transpose_last_two,
            tanh_scale=binding.tanh_scale,
            add_scalar=binding.add_scalar,
        )

    def _reshape_binding(
        self,
        binding: _PackedWeightSliceBinding,
        *,
        raw_shape: tuple[int, ...],
    ) -> _PackedWeightSliceBinding:
        # 仅在单 slice 情况下允许直接覆写 raw_shape；多 slice 时保留各自原 shape。
        updated_slices = tuple(
            _PackedWeightSliceSpec(
                packed=entry.packed,
                start_offset=entry.start_offset,
                raw_shape=raw_shape if len(binding.slices) == 1 else entry.raw_shape,
            )
            for entry in binding.slices
        )
        # 返回带新 raw_shape 的 binding 副本。
        return _PackedWeightSliceBinding(
            slices=updated_slices,
            resize_shape=binding.resize_shape,
            resize_mode=binding.resize_mode,
            expert_index=binding.expert_index,
            transpose_last_two=binding.transpose_last_two,
            tanh_scale=binding.tanh_scale,
            add_scalar=binding.add_scalar,
        )

    def _materialize_bound_weight(
        self,
        *,
        weight: torch.Tensor,
        binding: _PackedWeightSliceBinding,
    ) -> torch.Tensor | None:
        # 用列表收集每个 packed slice 还原后的权重块。
        slices = []
        # 逐个切片解量化并应用 binding 中记录的变换。
        for entry in binding.slices:
            value = dequantize_packed_range(
                entry.packed,
                start_offset=entry.start_offset,
                length=math.prod(entry.raw_shape),
                device=weight.device,
                dtype=weight.dtype,
            ).view(*entry.raw_shape)
            # 按需做 resize。
            if (
                binding.resize_shape is not None
                and binding.resize_mode is not None
                and tuple(value.shape) != binding.resize_shape
            ):
                value = resize_tensor(
                    value,
                    size=tuple(binding.resize_shape),
                    mode=binding.resize_mode,
                )
            # 按需抽取单个 expert。
            if binding.expert_index is not None:
                if value.ndim != 3:
                    return None
                if binding.expert_index < 0 or binding.expert_index >= value.shape[0]:
                    return None
                value = value[binding.expert_index]
            # 按需做后两维转置。
            if binding.transpose_last_two:
                value = value.transpose(-2, -1)
            # 按需做 tanh 缩放。
            if binding.tanh_scale != 0.0:
                value = binding.tanh_scale * torch.tanh(value)
            # 按需加常数偏置。
            if binding.add_scalar != 0.0:
                value = value + binding.add_scalar
            slices.append(value)
        # 没有任何有效切片时返回 None。
        if not slices:
            return None
        # 多 slice 时对各 slice 取平均，得到单个有效权重矩阵。
        combined = slices[0] if len(slices) == 1 else torch.stack(slices, dim=0).mean(dim=0)
        # 还原后的权重形状必须和目标 weight 一致。
        if tuple(combined.shape) != tuple(weight.shape):
            return None
        # 用 straight-through 形式把解量化权重绑定回原计算图。
        return combined + (weight - weight.detach())

    def _resolve_bound_value(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        # 非量化执行路径直接返回原张量。
        if not self._quantized_execution:
            return tensor
        # 当前张量没有 binding 时无需额外处理。
        binding = self._weight_binding_for_tensor(tensor)
        if binding is None:
            return tensor
        # 尝试把 binding 还原成真实有效的权重张量。
        effective = self._materialize_bound_weight(
            weight=tensor,
            binding=binding,
        )
        # 还原失败时退回原张量。
        if effective is None:
            return tensor
        # 还原成功后重新绑定语义信息，供后续链路复用。
        return self._bind_weight_slice(effective, binding=binding)

    def _linear_projection(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        # 非量化执行路径直接做普通矩阵乘。
        if not self._quantized_execution:
            return inputs @ weight
        # 先查看当前 weight 是否带有 packed slice binding。
        binding = self._weight_binding_for_tensor(weight)
        if binding is not None:
            transformed_rank = None
            # 优先从 resize_shape 推断变换后的张量阶数。
            if binding.resize_shape is not None:
                transformed_rank = len(binding.resize_shape)
            elif binding.slices:
                # 否则从首个 raw_shape 推断。
                transformed_rank = len(binding.slices[0].raw_shape)
            # 满足条件时，可直接走 packed slice linear 快路径。
            direct_linear = (
                bool(binding.slices)
                and transformed_rank in {2, 3}
                and not (
                    transformed_rank == 3 and binding.expert_index is None
                )
                and all(len(entry.raw_shape) in {2, 3} for entry in binding.slices)
            )
            if direct_linear:
                # 逐个 slice 直接从 packed 权重做线性投影。
                outputs = [
                    quantized_linear_from_packed_slice(
                        inputs,
                        weight,
                        packed=entry.packed,
                        start_offset=entry.start_offset,
                        raw_rows=entry.raw_shape[0],
                        raw_cols=(
                            entry.raw_shape[1]
                            if len(entry.raw_shape) > 1
                            else math.prod(entry.raw_shape)
                        ),
                        transpose_last_two=binding.transpose_last_two,
                        tanh_scale=binding.tanh_scale,
                        compute_view_dtype_name=(
                            self._quantization_compute_view_dtype
                        ),
                        raw_shape=entry.raw_shape,
                        resize_shape=binding.resize_shape,
                        resize_mode=binding.resize_mode,
                        expert_index=binding.expert_index,
                        add_scalar=binding.add_scalar,
                    )
                    for entry in binding.slices
                ]
                # 单 slice 直接返回；多 slice 时取平均。
                if len(outputs) == 1:
                    return outputs[0]
                return torch.stack(outputs, dim=0).mean(dim=0)
            # 不能走 packed slice 快路径时，先尝试把 binding 物化成普通权重。
            effective_weight = self._materialize_bound_weight(
                weight=weight,
                binding=binding,
            )
            if effective_weight is not None:
                return inputs @ effective_weight
        # 最后退回“先量化整块 weight 再线性”的通用路径。
        return quantized_linear(
            inputs,
            weight,
            bits=self._quantization_bits,
            group_size=self._quantization_group_size,
            sym=self._quantization_sym,
            pack_dtype_name=self._quantization_pack_dtype,
            compute_view_dtype_name=self._quantization_compute_view_dtype,
        )

    def _expert_projection(
        self,
        token_hidden: torch.Tensor,
        expert_weight: torch.Tensor,
    ) -> torch.Tensor:
        # expert 投影要求输入权重是 [expert, in, out] 三维张量。
        if expert_weight.ndim != 3:
            raise ValueError("expert projection expects a 3D expert weight tensor")
        # 逐个 expert 执行线性投影，并带上该 expert 的 binding 语义。
        outputs = [
            self._linear_projection(
                token_hidden.unsqueeze(0),
                self._bind_weight_slice(
                    expert_weight[index],
                    binding=self._expert_binding_for_index(expert_weight, index),
                ),
            ).squeeze(0)
            for index in range(expert_weight.shape[0])
        ]
        # 按 expert 维重新堆叠成三维输出。
        return torch.stack(outputs, dim=0)

    def _plan_graph_shape(
        self,
        *,
        non_routed_params: int,
        routed_params: int,
        active_expert_count: int,
        batch: BatchShape,
    ) -> _RepresentativeBucketGraphShape:
        # 至少保留 1 个 active expert，避免后续 shape 退化成非法值。
        expert_count = max(1, active_expert_count)
        # full_bucket logical CUDA 模式下，优先尝试直接使用模型真实 hidden / expert hidden 尺度。
        if self._full_bucket_logical_cuda_enabled():
            hidden_dim = max(1, self.config.model_spec.hidden_size)
            expert_hidden_dim = max(1, self.config.model_spec.moe_intermediate_size)
            # 估算一层非专家参数最少需要的预算。
            non_routed_needed = (
                2 * hidden_dim
                + 4 * hidden_dim * hidden_dim
                + hidden_dim * expert_count
                + 2 * hidden_dim * expert_hidden_dim
            )
            # 估算一层 routed expert 参数最少需要的预算。
            routed_needed = expert_count * 2 * hidden_dim * expert_hidden_dim
            # 当前预算足够时，直接返回真实尺度的图形状。
            if non_routed_needed <= non_routed_params and routed_needed <= routed_params:
                attention_head_count, kv_head_count, head_dim = self._plan_head_shape(
                    hidden_dim
                )
                return _RepresentativeBucketGraphShape(
                    sequence_length=self._bounded_sequence_length(batch),
                    hidden_dim=hidden_dim,
                    expert_hidden_dim=expert_hidden_dim,
                    active_expert_count=expert_count,
                    topk=min(
                        max(1, self.config.model_spec.num_experts_per_tok),
                        expert_count,
                    ),
                    attention_head_count=attention_head_count,
                    kv_head_count=kv_head_count,
                    head_dim=head_dim,
                )
        # 否则退回小图搜索，寻找预算内能容纳的最大 hidden / expert hidden 组合。
        best: tuple[int, int] | None = None
        for hidden_dim in range(8, 0, -1):
            for expert_hidden_dim in range(4, 0, -1):
                # 重新估算当前候选尺度下的非专家预算需求。
                non_routed_needed = (
                    2 * hidden_dim
                    + 4 * hidden_dim * hidden_dim
                    + hidden_dim * expert_count
                    + 2 * hidden_dim * expert_hidden_dim
                )
                # 重新估算当前候选尺度下的 routed expert 预算需求。
                routed_needed = expert_count * 2 * hidden_dim * expert_hidden_dim
                # 第一组满足预算的候选就是当前最优解。
                if non_routed_needed <= non_routed_params and routed_needed <= routed_params:
                    best = (hidden_dim, expert_hidden_dim)
                    break
            if best is not None:
                break
        # 所有候选都放不下时，退到最小 1x1 图。
        if best is None:
            best = (1, 1)
        # 基于最终 hidden_dim 规划 attention 头数和 head_dim。
        attention_head_count, kv_head_count, head_dim = self._plan_head_shape(best[0])
        # 轻量模式下 topk 也做额外上限压缩，避免执行成本过高。
        topk = min(
            max(1, self.config.model_spec.num_experts_per_tok),
            expert_count,
            2,
        )
        # 返回最终选定的代表性图形状。
        return _RepresentativeBucketGraphShape(
            sequence_length=self._bounded_sequence_length(batch),
            hidden_dim=best[0],
            expert_hidden_dim=best[1],
            active_expert_count=expert_count,
            topk=topk,
            attention_head_count=attention_head_count,
            kv_head_count=kv_head_count,
            head_dim=head_dim,
        )

    def _required_layer_param_sizes(
        self,
        graph_shape: _RepresentativeBucketGraphShape,
    ) -> tuple[int, int]:
        # 非专家参数需求由 norm、attention、router/shared expert 等模块共同组成。
        non_routed_needed = (
            2 * graph_shape.hidden_dim
            + 4 * graph_shape.hidden_dim * graph_shape.hidden_dim
            + graph_shape.hidden_dim * graph_shape.active_expert_count
            + 2 * graph_shape.hidden_dim * graph_shape.expert_hidden_dim
        )
        # routed expert 参数需求等于 active experts 的 gate/up/down 相关张量总量。
        routed_needed = (
            2
            * graph_shape.active_expert_count
            * graph_shape.hidden_dim
            * graph_shape.expert_hidden_dim
        )
        # 返回“非专家需求 + 专家需求”二元组。
        return non_routed_needed, routed_needed

    def _plan_head_shape(
        self,
        hidden_dim: int,
    ) -> tuple[int, int, int]:
        # attention 头数不能超过 hidden_dim，也不能超过模型配置上限。
        max_heads = min(hidden_dim, self.config.model_spec.num_attention_heads)
        # 轻量模式下进一步压缩 attention 头数上限。
        if not self._full_bucket_logical_cuda_enabled():
            max_heads = min(max_heads, 4)
        # 优先选择能整除 hidden_dim 的最大头数。
        attention_head_count = 1
        for candidate in range(max_heads, 0, -1):
            if hidden_dim % candidate == 0:
                attention_head_count = candidate
                break
        # KV 头数不能超过 attention 头数，也不能超过模型配置上限。
        max_kv = min(attention_head_count, self.config.model_spec.num_key_value_heads)
        # 优先选择能整除 attention 头数的最大 KV 头数。
        kv_head_count = 1
        for candidate in range(max_kv, 0, -1):
            if attention_head_count % candidate == 0:
                kv_head_count = candidate
                break
        # head_dim 由 hidden_dim / attention_head_count 得到。
        return (
            attention_head_count,
            kv_head_count,
            hidden_dim // attention_head_count,
        )

    def _take_vector(
        self,
        tensor: torch.Tensor,
        cursor: int,
        size: int,
    ) -> tuple[torch.Tensor, int]:
        # 从当前 cursor 开始截取 size 个元素。
        chunk = tensor.narrow(0, cursor, size)
        # 把截取结果稳定化并视作一维向量。
        vector = self._stabilize_tensor(chunk.view(size))
        # 若源张量带有 flat packed source，则把切出的向量重新绑定到对应 packed slice。
        flat_source = self._flat_packed_source(tensor)
        if flat_source is None:
            return vector, cursor + size
        packed, base_offset = flat_source
        return (
            self._bind_weight_slice(
                vector,
                binding=_PackedWeightSliceBinding(
                    slices=(
                        _PackedWeightSliceSpec(
                            packed=packed,
                            start_offset=base_offset + cursor,
                            raw_shape=(size,),
                        ),
                    ),
                ),
            ),
            # 返回更新后的 cursor。
            cursor + size,
        )

    def _take_matrix(
        self,
        tensor: torch.Tensor,
        cursor: int,
        rows: int,
        cols: int,
    ) -> tuple[torch.Tensor, int]:
        # 当前矩阵块的总元素数等于 rows * cols。
        size = rows * cols
        # 从当前 cursor 开始截取矩阵块对应的一维区间。
        chunk = tensor.narrow(0, cursor, size)
        # 稳定化后重排成 [rows, cols]。
        matrix = self._stabilize_tensor(chunk.view(rows, cols))
        # 若源张量带有 flat packed source，则把矩阵块重新绑定到对应 packed slice。
        flat_source = self._flat_packed_source(tensor)
        return (
            self._bind_weight_slice(
                matrix,
                binding=(
                    None
                    if flat_source is None
                    else _PackedWeightSliceBinding(
                        slices=(
                            _PackedWeightSliceSpec(
                                packed=flat_source[0],
                                start_offset=flat_source[1] + cursor,
                                raw_shape=(rows, cols),
                            ),
                        ),
                    )
                ),
            ),
            # 返回更新后的 cursor。
            cursor + size,
        )

    def _build_hidden_states(
        self,
        *,
        step_index: int,
        bucket: LayerBucketPlan,
        batch: BatchShape,
        hidden_dim: int,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        # 构造序列位置索引。
        positions = torch.arange(sequence_length, dtype=torch.float32, device=device)
        # 构造隐藏维特征索引。
        features = torch.arange(1, hidden_dim + 1, dtype=torch.float32, device=device)
        # 用 batch token 规模调制基础频率。
        token_ratio = batch.total_tokens / max(self.config.model_spec.hidden_size, 1)
        # bucket 和 step 共同决定相位偏移。
        phase = 0.17 * float(bucket.bucket_id + 1) + 0.09 * float(step_index + 1)
        # 基础项使用 sin 生成具有时序结构的 hidden。
        base = torch.sin(
            positions.unsqueeze(-1) * 0.31 + features.unsqueeze(0) * 0.23 + phase
        )
        # 调制项使用 cos 引入另一组频率模式。
        modulation = 0.5 * torch.cos(
            positions.unsqueeze(-1) * 0.13
            + features.unsqueeze(0) * (0.07 + token_ratio * 0.001)
        )
        # 基础 hidden 由 base 与 modulation 叠加而成。
        hidden_states = base + modulation
        # 若 batch 自带显式 token 行，则把真实 token id 注入 hidden 模式。
        if batch.has_token_rows:
            token_values = self._sequence_token_values(
                batch.token_rows,
                mask_rows=batch.attention_mask_rows,
                sequence_length=sequence_length,
                device=device,
            )
            # token_phase 把离散 token id 映射到连续相位。
            token_phase = 0.0013 * token_values.unsqueeze(-1)
            # 第一项 token 调制。
            hidden_states = hidden_states + 0.35 * torch.sin(
                token_phase + features.unsqueeze(0) * 0.19 + phase * 0.7
            )
            # 第二项 token 调制。
            hidden_states = hidden_states + 0.15 * torch.cos(
                token_phase * 0.5
                + positions.unsqueeze(-1) * 0.21
                + features.unsqueeze(0) * 0.11
            )
        # 最后统一缩放，避免初始 hidden 过大。
        return hidden_states / 1.8

    def _sequence_token_values(
        self,
        rows: tuple[tuple[int, ...], ...],
        *,
        mask_rows: tuple[tuple[int, ...], ...] = (),
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        # ------------------------------- 展平真实 token 序列 -------------------------------
        # 有 mask 时只保留真实 token，尾部 padding 不参与代表性特征构造。
        if mask_rows:
            flattened = [
                token_id
                for row, mask_row in zip(rows, mask_rows, strict=True)
                for token_id, keep in zip(row, mask_row, strict=True)
                if keep
            ]
        else:
            # 没有 mask 时保持旧语义：所有 token_rows 位置都视为有效。
            flattened = [token_id for row in rows for token_id in row]
        # 没有显式 token 时，退回简单的位置序列。
        if not flattened:
            return torch.arange(sequence_length, dtype=torch.float32, device=device)
        # token 序列不足时，用最后一个真实 token 延展到目标长度，避免复制整段上下文制造周期模式。
        if len(flattened) < sequence_length:
            flattened = flattened + [flattened[-1]] * (sequence_length - len(flattened))
        else:
            # token 序列过长时只保留前 sequence_length 个。
            flattened = flattened[:sequence_length]
        # 转成 device 上的 float 张量。
        return torch.tensor(flattened, dtype=torch.float32, device=device)

    def _build_target_signal(
        self,
        *,
        step_index: int,
        bucket: LayerBucketPlan,
        batch: BatchShape,
        hidden_dim: int,
        sequence_length: int,
        fallback: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        # 没有 target_rows 时，退回对 fallback 做一步时间平移。
        if not batch.target_rows:
            return torch.roll(fallback, shifts=-1, dims=0)
        # 构造位置和特征索引。
        positions = torch.arange(sequence_length, dtype=torch.float32, device=device)
        features = torch.arange(1, hidden_dim + 1, dtype=torch.float32, device=device)
        # 把目标 token 行映射成一维 token 序列。
        target_values = self._sequence_token_values(
            batch.target_rows,
            mask_rows=batch.target_attention_mask_rows,
            sequence_length=sequence_length,
            device=device,
        )
        # bucket 与 step 共同决定目标相位偏移。
        phase = 0.13 * float(bucket.bucket_id + 1) + 0.05 * float(step_index + 1)
        # 将 token id 映射为目标相位。
        target_phase = 0.0017 * target_values.unsqueeze(-1)
        # 第一组 target 信号分量。
        target_signal = torch.sin(
            target_phase + features.unsqueeze(0) * 0.17 + phase
        )
        # 第二组 target 信号分量。
        target_signal = target_signal + 0.4 * torch.cos(
            positions.unsqueeze(-1) * 0.27
            + target_phase * 0.6
            + features.unsqueeze(0) * 0.09
        )
        # 统一缩放到更稳定的数值范围。
        return target_signal / 1.4

    def _mtp_target(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 取下一 token 的 hidden 作为主目标。
        next_token = torch.roll(hidden_states, shifts=-1, dims=0)
        # 再取下下 token 的 hidden 作为 draft 目标。
        draft_token = torch.roll(hidden_states, shifts=-2, dims=0)
        # 用 0.7 / 0.3 加权融合后做稳定化。
        return self._stabilize_tensor(0.7 * next_token + 0.3 * draft_token)

    def build_predictor_hidden_summary(
        self,
        *,
        step_index: int,
        batch: BatchShape,
        bucket: LayerBucketPlan,
        insertion_layer_index: int,
        active_expert_ids: tuple[int, ...],
        summary_dim: int,
    ) -> torch.Tensor:
        # summary 维度必须为正。
        if summary_dim < 1:
            raise ValueError("summary_dim must be >= 1")
        # predictor hidden summary 统一在 CPU 上构造。
        summary_device = torch.device("cpu")
        # 捕获用 hidden 维度控制在一个较小但不太小的范围内。
        capture_hidden_dim = max(8, min(max(summary_dim, 8), 64))
        sequence_length = self._bounded_sequence_length(batch)
        # 先构造一份代表性的 hidden states。
        hidden_states = self._build_hidden_states(
            step_index=step_index,
            bucket=bucket,
            batch=batch,
            hidden_dim=capture_hidden_dim,
            sequence_length=sequence_length,
            device=summary_device,
        )
        # 再取出序列 token 值。
        token_values = self._sequence_token_values(
            batch.token_rows,
            mask_rows=batch.attention_mask_rows,
            sequence_length=sequence_length,
            device=summary_device,
        )
        # active expert id 集合统一转成张量，便于统计。
        active_expert_tensor = torch.tensor(
            active_expert_ids if active_expert_ids else (0,),
            dtype=torch.float32,
            device=summary_device,
        )
        # 记录当前 bucket 的起始绝对层号。
        bucket_start = bucket.layer_indices[0]
        # 计算 insertion layer 相对 bucket 起点的偏移。
        layer_offset = insertion_layer_index - bucket_start
        # 抽取 hidden、bucket、expert、token 多类统计量并拼成大向量。
        pooled_features = torch.cat(
            (
                hidden_states.mean(dim=0),
                hidden_states.std(dim=0, unbiased=False),
                hidden_states[0],
                hidden_states[-1],
                torch.tensor(
                    (
                        float(batch.samples),
                        float(batch.tokens_per_sample),
                        float(batch.total_tokens),
                        float(bucket.bucket_id),
                        float(insertion_layer_index),
                        float(layer_offset),
                        float(bucket.contains_full_attention),
                        float(
                            sum(
                                1
                                for attention_type in bucket.attention_types
                                if attention_type == "full_attention"
                            )
                        ),
                        float(
                            sum(
                                1
                                for attention_type in bucket.attention_types
                                if attention_type == "linear_attention"
                            )
                        ),
                        float(active_expert_tensor.mean().item()),
                        float(active_expert_tensor.std(unbiased=False).item()),
                        float(active_expert_tensor.min().item()),
                        float(active_expert_tensor.max().item()),
                        float(token_values.mean().item()),
                        float(token_values.std(unbiased=False).item()),
                        float(token_values[0].item()),
                        float(token_values[-1].item()),
                    ),
                    dtype=torch.float32,
                    device=summary_device,
                ),
            ),
            dim=0,
        )
        # 将拼好的统计向量 resize 到 summary_dim。
        summary = self._resize_vector(pooled_features, size=summary_dim)
        # 再构造一组 summary 特征索引，用于注入轻量位置调制。
        summary_feature_ids = torch.arange(
            1,
            summary_dim + 1,
            dtype=torch.float32,
            device=summary_device,
        )
        # 用 sin 调制对 summary 做轻量平滑扰动。
        summary = 0.85 * summary + 0.15 * torch.sin(
            summary_feature_ids * 0.17
            + 0.09 * float(step_index + 1)
            + 0.05 * float(insertion_layer_index + 1)
        )
        # 最后用 tanh 压缩动态范围并固定返回到 CPU。
        return torch.tanh(summary / 2.5).to(device="cpu")

    def _full_attention_context(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        graph_shape: _RepresentativeBucketGraphShape,
    ) -> torch.Tensor:
        # 先把 q 拆成多头表示。
        q_heads = self._split_heads(q, graph_shape=graph_shape)
        # k/v 需要按 KV 头分组扩展到 attention 头数。
        k_heads = self._expand_grouped_kv_heads(k, graph_shape=graph_shape)
        v_heads = self._expand_grouped_kv_heads(v, graph_shape=graph_shape)
        # 标准 attention scale 因子。
        scale = 1.0 / math.sqrt(max(graph_shape.head_dim, 1))
        # 计算并稳定化注意力分数。
        scores = self._stabilize_tensor(
            torch.einsum("thd,shd->ths", q_heads, k_heads) * scale,
            max_abs=20.0,
        )
        # 构造 causal mask，保证只能看见当前位置及其之前。
        causal_mask = torch.triu(
            torch.ones(
                q_heads.shape[0],
                q_heads.shape[0],
                dtype=torch.bool,
                device=q_heads.device,
            ),
            diagonal=1,
        )
        # 用一个极小值屏蔽未来位置。
        scores = scores.masked_fill(causal_mask.unsqueeze(1), -1e9)
        # softmax 后与 v 做加权求和得到上下文。
        context = torch.einsum(
            "ths,shd->thd",
            F.softmax(scores, dim=-1),
            v_heads,
        )
        # 合并多头并做稳定化。
        return self._stabilize_tensor(self._merge_heads(context))

    def _linear_attention_context(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        graph_shape: _RepresentativeBucketGraphShape,
    ) -> torch.Tensor:
        # 先把 q/k/v 统一转成稳定化后的多头表示。
        q_heads = self._stabilize_tensor(self._split_heads(q, graph_shape=graph_shape))
        k_heads = self._stabilize_tensor(
            self._expand_grouped_kv_heads(k, graph_shape=graph_shape)
        )
        v_heads = self._stabilize_tensor(
            self._expand_grouped_kv_heads(v, graph_shape=graph_shape)
        )
        # 使用 elu+1 构造正值特征映射。
        q_phi = F.elu(q_heads) + 1.0
        k_phi = F.elu(k_heads) + 1.0
        # 前缀累积 K*V 和 K，形成线性 attention 的前缀状态。
        kv_prefix = torch.cumsum(
            k_phi.unsqueeze(-1) * v_heads.unsqueeze(-2),
            dim=0,
        )
        k_prefix = torch.cumsum(k_phi, dim=0)
        outputs = []
        # 逐 token 读取前缀状态并恢复当前 token 的上下文。
        for index in range(q_heads.shape[0]):
            numerator = torch.einsum("hd,hde->he", q_phi[index], kv_prefix[index])
            denominator = torch.sum(q_phi[index] * k_prefix[index], dim=-1).clamp_min(
                1e-6
            )
            outputs.append(
                self._stabilize_tensor(numerator / denominator.unsqueeze(-1))
            )
        # 合并多头并做稳定化。
        return self._stabilize_tensor(
            self._merge_heads(torch.stack(outputs, dim=0))
        )

    def _split_heads(
        self,
        tensor: torch.Tensor,
        *,
        graph_shape: _RepresentativeBucketGraphShape,
    ) -> torch.Tensor:
        # 将最后一维按 [attention_head_count, head_dim] 拆成多头表示。
        return tensor.view(
            tensor.shape[0],
            graph_shape.attention_head_count,
            graph_shape.head_dim,
        )

    def _merge_heads(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        # 将 [seq, heads, head_dim] 合并回 [seq, hidden_dim]。
        return tensor.reshape(tensor.shape[0], -1)

    def _expand_grouped_kv_heads(
        self,
        tensor: torch.Tensor,
        *,
        graph_shape: _RepresentativeBucketGraphShape,
    ) -> torch.Tensor:
        # 先把输入拆成 attention 头表示。
        attention_heads = self._split_heads(tensor, graph_shape=graph_shape)
        # KV 头数与 attention 头数一致时无需额外扩展。
        if graph_shape.kv_head_count == graph_shape.attention_head_count:
            return attention_heads
        # 计算每个 KV 头需要覆盖多少个 attention 头。
        group_size = graph_shape.attention_head_count // graph_shape.kv_head_count
        # 先按 group 聚合成真实 KV 头。
        grouped = attention_heads.view(
            tensor.shape[0],
            graph_shape.kv_head_count,
            group_size,
            graph_shape.head_dim,
        ).mean(dim=2)
        # 再重复展开回 attention 头数量。
        return grouped.repeat_interleave(group_size, dim=1)

    def _rms_norm_scaled(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        # 先做 layer_norm 近似的归一化，再施加逐通道缩放。
        return self._stabilize_tensor(
            F.layer_norm(tensor, (tensor.shape[-1],)) * scale.unsqueeze(0)
        )

    def _covered_roles(
        self,
        *entry_groups: tuple[ParameterSourceSlice, ...],
    ) -> tuple[str, ...]:
        # 汇总多组 source entry 实际覆盖到的语义角色。
        return tuple(
            sorted(
                {
                    entry.semantic_role
                    for entries in entry_groups
                    for entry in entries
                    if entry.length > 0
                }
            )
        )

    def _semantic_dot_gate(
        self,
        hidden_states: torch.Tensor,
        gate_vector: torch.Tensor,
    ) -> torch.Tensor:
        # 用 hidden 与 gate 向量的点积构造逐 token 的门控权重。
        return torch.sigmoid(
            torch.sum(hidden_states * gate_vector.unsqueeze(0), dim=-1, keepdim=True)
            / math.sqrt(max(hidden_states.shape[-1], 1))
        )

    def _shared_expert_forward(
        self,
        *,
        hidden_states: torch.Tensor,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        expert_gate: torch.Tensor,
    ) -> torch.Tensor:
        # 先构造 shared expert 的逐 token gate。
        shared_gate = self._semantic_dot_gate(hidden_states, expert_gate)
        # gate_proj 走 SiLU 激活。
        gated = F.silu(self._linear_projection(hidden_states, gate_proj))
        # up_proj 走线性映射。
        up = self._linear_projection(hidden_states, up_proj)
        # 共享专家输出 = down_proj(gate * up) * shared_gate。
        return self._stabilize_tensor(
            self._linear_projection(gated * up, down_proj) * shared_gate
        )

    def _moe_forward_gated(
        self,
        *,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        expert_gate: torch.Tensor,
        expert_up: torch.Tensor,
        expert_down: torch.Tensor,
        expert_ids: tuple[int, ...],
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 先对 router logits 做稳定化，避免 topk/softmax 溢出。
        router_logits = self._stabilize_tensor(router_logits, max_abs=20.0)
        # 用 expert id 构造一个轻量缩放因子，模拟不同 expert 的差异性。
        expert_id_scale = 1.0 + (
            torch.tensor(expert_ids, dtype=torch.float32, device=hidden_states.device)
            / max(float(self.config.model_spec.num_experts), 1.0)
        )
        # 对每个 token 选择 top-k experts。
        topk_values, topk_indices = torch.topk(router_logits, k=topk, dim=-1)
        # 将 top-k logits 归一化成混合权重。
        topk_weights = F.softmax(topk_values, dim=-1)
        mixed_outputs = []
        # 逐 token 执行专家前向并按 top-k 权重混合。
        for token_index in range(hidden_states.shape[0]):
            selected_indices = topk_indices[token_index]
            # 取出当前 token 命中的 gate/up/down 专家权重。
            selected_gate = expert_gate.index_select(0, selected_indices)
            selected_up = expert_up.index_select(0, selected_indices)
            selected_down = expert_down.index_select(0, selected_indices)
            selected_scale = expert_id_scale.index_select(0, selected_indices)
            token_hidden = hidden_states[token_index]
            # 对每个命中的专家分别做 gate/up 投影。
            gate_hidden = self._expert_projection(token_hidden, selected_gate)
            up_hidden = self._expert_projection(token_hidden, selected_up)
            # 专家隐状态 = SiLU(gate) * up，并乘 expert id 缩放因子。
            expert_hidden = F.silu(gate_hidden) * up_hidden
            expert_hidden = expert_hidden * selected_scale.unsqueeze(-1)
            # 对每个命中的专家分别做 down_proj。
            expert_output = torch.stack(
                [
                    self._linear_projection(
                        expert_hidden[index].unsqueeze(0),
                        selected_down[index],
                    ).squeeze(0)
                    for index in range(selected_down.shape[0])
                ],
                dim=0,
            )
            # 用 top-k routing 权重加权混合专家输出。
            mixed_outputs.append(
                self._stabilize_tensor(
                    (expert_output * topk_weights[token_index].unsqueeze(-1)).sum(dim=0)
                )
            )
        # 计算平均 routing 分布，供负载均衡损失使用。
        routing_distribution = self._stabilize_tensor(
            F.softmax(router_logits, dim=-1).mean(dim=0)
        )
        # 构造均匀分布作为平衡目标。
        uniform = torch.full_like(
            routing_distribution,
            1.0 / max(routing_distribution.numel(), 1),
        )
        # 用 MSE 近似负载均衡损失。
        routing_balance_loss = torch.nan_to_num(
            F.mse_loss(routing_distribution, uniform),
            nan=0.0,
            posinf=1e3,
            neginf=1e3,
        )
        # 返回混合后的 token 输出和 routing 平衡损失。
        return self._stabilize_tensor(torch.stack(mixed_outputs, dim=0)), routing_balance_loss

    def _moe_forward(
        self,
        *,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        expert_up: torch.Tensor,
        expert_down: torch.Tensor,
        expert_ids: tuple[int, ...],
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 先对 router logits 做稳定化。
        router_logits = self._stabilize_tensor(router_logits, max_abs=20.0)
        # 用 expert id 构造一个轻量缩放因子，模拟不同 expert 的差异性。
        expert_id_scale = 1.0 + (
            torch.tensor(expert_ids, dtype=torch.float32, device=hidden_states.device)
            / max(float(self.config.model_spec.num_experts), 1.0)
        )
        # 对每个 token 选择 top-k experts。
        topk_values, topk_indices = torch.topk(router_logits, k=topk, dim=-1)
        # 将 top-k logits 归一化成混合权重。
        topk_weights = F.softmax(topk_values, dim=-1)
        mixed_outputs = []
        # 逐 token 执行专家前向并按 top-k 权重混合。
        for token_index in range(hidden_states.shape[0]):
            selected_indices = topk_indices[token_index]
            # 取出当前 token 命中的 up/down 专家权重。
            selected_up = expert_up.index_select(0, selected_indices)
            selected_down = expert_down.index_select(0, selected_indices)
            selected_scale = expert_id_scale.index_select(0, selected_indices)
            token_hidden = hidden_states[token_index]
            # 对每个命中的专家分别做 up_proj。
            expert_hidden = self._expert_projection(token_hidden, selected_up)
            # 专家隐状态走 SiLU，并乘 expert id 缩放因子。
            expert_hidden = F.silu(expert_hidden) * selected_scale.unsqueeze(-1)
            # 对每个命中的专家分别做 down_proj。
            expert_output = torch.stack(
                [
                    self._linear_projection(
                        expert_hidden[index].unsqueeze(0),
                        selected_down[index],
                    ).squeeze(0)
                    for index in range(selected_down.shape[0])
                ],
                dim=0,
            )
            # 用 top-k routing 权重加权混合专家输出。
            mixed_outputs.append(
                self._stabilize_tensor(
                    (expert_output * topk_weights[token_index].unsqueeze(-1)).sum(dim=0)
                )
            )
        # 计算平均 routing 分布，供负载均衡损失使用。
        routing_distribution = self._stabilize_tensor(
            F.softmax(router_logits, dim=-1).mean(dim=0)
        )
        # 构造均匀分布作为平衡目标。
        uniform = torch.full_like(
            routing_distribution,
            1.0 / max(routing_distribution.numel(), 1),
        )
        # 用 MSE 近似负载均衡损失。
        routing_balance_loss = torch.nan_to_num(
            F.mse_loss(routing_distribution, uniform),
            nan=0.0,
            posinf=1e3,
            neginf=1e3,
        )
        # 返回混合后的 token 输出和 routing 平衡损失。
        return self._stabilize_tensor(torch.stack(mixed_outputs, dim=0)), routing_balance_loss

    def _layer_param_slice(
        self,
        tensor: torch.Tensor,
        layer_index: int,
        layer_count: int,
    ) -> torch.Tensor:
        # 先定位当前层在线性参数中的起止区间。
        start, stop = self._layer_param_span(
            tensor,
            layer_index=layer_index,
            layer_count=layer_count,
        )
        # 截取当前层对应的一段参数，并做数值稳定化。
        sliced = self._stabilize_tensor(
            tensor.narrow(0, start, max(stop - start, 1))
        )
        # 若源张量对应 flat packed source，则把当前层切片重新绑定回 packed 语义。
        flat_source = self._flat_packed_source(tensor)
        if flat_source is None:
            return sliced
        packed, base_offset = flat_source
        return self._bind_weight_slice(
            sliced,
            binding=_PackedWeightSliceBinding(
                slices=(
                    _PackedWeightSliceSpec(
                        packed=packed,
                        start_offset=base_offset + start,
                        raw_shape=(sliced.numel(),),
                    ),
                ),
            ),
        )

    def _layer_param_span(
        self,
        tensor: torch.Tensor,
        *,
        layer_index: int,
        layer_count: int,
    ) -> tuple[int, int]:
        # 把整块参数按层数做等分。
        base = tensor.numel() // max(layer_count, 1)
        # 当前层起始偏移等于 layer_index * base。
        start = layer_index * base
        # 最后一层吃掉全部剩余，其余层长度固定为 base。
        stop = tensor.numel() if layer_index == layer_count - 1 else start + base
        # 返回当前层参数区间。
        return start, stop

    def _detached_bound_parameter(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        # 先断开计算图、clone 一份并重新开启梯度。
        detached = tensor.detach().clone().requires_grad_(True)
        # 若原张量带有 flat packed view，则把绑定复制过来。
        packed = self._packed_view_for_tensor(tensor)
        if packed is not None:
            self._bind_flat_packed_view(detached, packed)
        # 若原张量带有 packed slice binding，也一并复制过来。
        binding = self._weight_binding_for_tensor(tensor)
        if binding is not None:
            detached = self._bind_weight_slice(detached, binding=binding)
        # 返回带绑定语义的 detached 参数。
        return detached

    def _compact_layer_parameter(
        self,
        tensor: torch.Tensor,
        *,
        start_offset: int,
        compact_numel: int,
        packed: PackedQuantizedTensor | None,
    ) -> torch.Tensor:
        # 先把层参数前 compact_numel 个元素截出来，作为紧凑视图。
        compact = self._stabilize_tensor(
            tensor.narrow(0, 0, min(compact_numel, tensor.numel())).to(
                device=self._compute_device,
                dtype=torch.float32,
            )
        )
        # 紧凑视图长度必须与预期完全一致。
        if compact.numel() != compact_numel:
            raise ValueError(
                f"compact layer parameter expected {compact_numel} values, "
                f"got {compact.numel()}"
            )
        # 断开后重新开启梯度，供局部梯度路径单独求导。
        compact = compact.detach().clone().requires_grad_(True)
        # 没有 packed 视图时直接返回普通紧凑张量。
        if packed is None:
            return compact
        # 否则把紧凑张量绑定到原 packed 权重的对应切片。
        return self._bind_weight_slice(
            compact,
            binding=_PackedWeightSliceBinding(
                slices=(
                    _PackedWeightSliceSpec(
                        packed=packed,
                        start_offset=start_offset,
                        raw_shape=(compact_numel,),
                    ),
                ),
            ),
        )

    def _semantic_entries_for_layer(
        self,
        source_layout: tuple[ParameterSourceSlice, ...],
        *,
        layer_index: int,
    ) -> tuple[ParameterSourceSlice, ...]:
        # 只保留当前绝对层号且 length>0 的来源条目。
        return tuple(
            entry
            for entry in source_layout
            if entry.layer_index == layer_index and entry.length > 0
        )

    def _layout_budget_for_layer(
        self,
        source_layout: tuple[ParameterSourceSlice, ...],
        *,
        layer_index: int,
    ) -> int:
        # 当前层布局预算等于该层所有 source entry 的长度和。
        return sum(
            entry.length
            for entry in source_layout
            if entry.layer_index == layer_index
        )

    def _role_values(
        self,
        tensor: torch.Tensor,
        *,
        layer_entries: tuple[ParameterSourceSlice, ...],
        roles: tuple[str, ...],
    ) -> torch.Tensor:
        # 用列表收集命中指定 roles 的线性切片。
        chunks = []
        for entry in layer_entries:
            # 过滤掉不属于目标 roles 的 entry。
            if roles and entry.semantic_role not in roles:
                continue
            # entry 起点越界时跳过。
            if entry.start_offset >= tensor.numel():
                continue
            # 实际可读取长度不能超出张量末尾。
            length = min(entry.length, tensor.numel() - entry.start_offset)
            if length <= 0:
                continue
            # 把当前 entry 的线性切片展平后加入结果。
            chunks.append(tensor.narrow(0, entry.start_offset, length).reshape(-1))
        # 指定 roles 下没有任何命中时，退回到“不限 roles”的兜底路径。
        if not chunks and roles:
            return self._role_values(
                tensor,
                layer_entries=layer_entries,
                roles=(),
            )
        # 命中多个 chunk 时把它们拼接起来。
        if chunks:
            return torch.cat(chunks, dim=0)
        # 完全没有可用 chunk 时，至少返回 1 个元素，避免后续 shape 退化。
        return tensor[:1].reshape(-1)

    def _entry_block(
        self,
        tensor: torch.Tensor,
        *,
        entry: ParameterSourceSlice,
    ) -> torch.Tensor | None:
        # entry 起点越界时直接返回 None。
        if entry.start_offset >= tensor.numel():
            return None
        # 实际可读取长度不能超出张量末尾。
        length = min(entry.length, tensor.numel() - entry.start_offset)
        if length <= 0:
            return None
        # 先按 entry 线性区间切出原始块。
        raw = tensor.narrow(0, entry.start_offset, length)
        # 查看源张量是否带有 flat packed view。
        packed = self._packed_view_for_tensor(tensor)
        # 若 entry 自带 slice_shape，则优先尝试恢复成对应形状的块。
        if entry.slice_shape:
            block_elements = math.prod(entry.slice_shape)
            if block_elements > 0:
                trimmed = raw[: min(block_elements, raw.numel())]
                if trimmed.numel() == block_elements:
                    return self._bind_weight_slice(
                        trimmed.view(*entry.slice_shape),
                        binding=(
                            None
                            if packed is None
                            else _PackedWeightSliceBinding(
                                slices=(
                                    _PackedWeightSliceSpec(
                                        packed=packed,
                                        start_offset=entry.start_offset,
                                        raw_shape=entry.slice_shape,
                                    ),
                                ),
                            )
                        ),
                    )
        # 无法恢复成 slice_shape 时，退回线性 block。
        return self._bind_weight_slice(
            raw.reshape(-1),
            binding=(
                None
                if packed is None
                else _PackedWeightSliceBinding(
                    slices=(
                        _PackedWeightSliceSpec(
                            packed=packed,
                            start_offset=entry.start_offset,
                            raw_shape=(length,),
                        ),
                    ),
                )
            ),
        )

    def _role_blocks(
        self,
        tensor: torch.Tensor,
        *,
        layer_entries: tuple[ParameterSourceSlice, ...],
        roles: tuple[str, ...],
    ) -> tuple[torch.Tensor, ...]:
        # 用列表收集命中指定 roles 的 entry block。
        blocks = []
        for entry in layer_entries:
            # 过滤掉不属于目标 roles 的 entry。
            if roles and entry.semantic_role not in roles:
                continue
            block = self._entry_block(tensor, entry=entry)
            if block is not None:
                # 当前 block 非空时，做稳定化并加入结果。
                blocks.append(self._stabilize_bound(block))
        # 指定 roles 下没有任何命中时，退回到“不限 roles”的兜底路径。
        if not blocks and roles:
            return self._role_blocks(
                tensor,
                layer_entries=layer_entries,
                roles=(),
            )
        # 返回命中的 block 元组。
        return tuple(blocks)

    def _expand_values(
        self,
        values: torch.Tensor,
        *,
        size: int,
    ) -> torch.Tensor:
        # 目标长度非正时直接返回空向量。
        if size <= 0:
            return torch.empty(0, dtype=torch.float32, device=values.device)
        # 原向量为空时，用单个 1 作为兜底种子。
        if values.numel() <= 0:
            values = torch.ones(1, dtype=torch.float32, device=values.device)
        # 计算至少需要重复多少次才能覆盖目标长度。
        repeats = max(1, math.ceil(size / values.numel()))
        # 重复并裁剪到目标长度。
        return values.repeat(repeats)[:size].reshape(-1)

    def _resize_vector(
        self,
        values: torch.Tensor,
        *,
        size: int,
    ) -> torch.Tensor:
        # 目标长度非正时直接返回空向量。
        if size <= 0:
            return torch.empty(0, dtype=torch.float32, device=values.device)
        # 先稳定化，并保留可能存在的 packed binding。
        flat = self._stabilize_bound(values)
        binding = self._weight_binding_for_tensor(flat)
        # 非一维输入时先展平成一维，再同步更新 binding shape。
        if flat.ndim != 1:
            flat = flat.reshape(-1)
            if binding is not None:
                flat = self._bind_weight_slice(
                    flat,
                    binding=self._reshape_binding(
                        binding,
                        raw_shape=(flat.numel(),),
                    ),
                )
                binding = self._weight_binding_for_tensor(flat)
        # 长度已匹配时直接返回。
        if flat.numel() == size:
            return flat
        # 否则走一维线性插值 resize。
        result = self._stabilize_tensor(
            resize_tensor(flat, size=(size,), mode="linear")
        )
        # 没有 binding 时直接返回普通结果。
        if binding is None:
            return result
        # 有 binding 时，把 resize 语义写回 binding 元信息。
        return self._bind_weight_slice(
            result,
            binding=_PackedWeightSliceBinding(
                slices=self._reshape_binding(
                    binding,
                    raw_shape=(flat.numel(),),
                ).slices,
                resize_shape=(size,),
                resize_mode="linear",
                expert_index=binding.expert_index,
                transpose_last_two=binding.transpose_last_two,
                tanh_scale=binding.tanh_scale,
                add_scalar=binding.add_scalar,
            ),
        )

    def _resize_matrix(
        self,
        matrix: torch.Tensor,
        *,
        size: tuple[int, int],
    ) -> torch.Tensor:
        # 目标行列任一非正时直接返回空矩阵。
        rows, cols = size
        if rows <= 0 or cols <= 0:
            return torch.empty(rows, cols, dtype=torch.float32, device=matrix.device)
        # 先稳定化，并保留可能存在的 packed binding。
        source = self._stabilize_bound(matrix)
        binding = self._weight_binding_for_tensor(source)
        # 一维输入时先扩成 [1, n] 矩阵。
        if source.ndim == 1:
            source = source.view(1, -1)
            if binding is not None:
                source = self._bind_weight_slice(
                    source,
                    binding=self._reshape_binding(
                        binding,
                        raw_shape=tuple(source.shape),
                    ),
                )
        # 更高维输入时，退回“前面展平、最后一维保留”的二维矩阵。
        elif source.ndim > 2:
            source = source.reshape(-1, source.shape[-1])
        # 形状已匹配时直接返回。
        if tuple(source.shape) == size:
            return source
        # 否则走二维 bilinear resize。
        result = self._stabilize_tensor(
            resize_tensor(source, size=size, mode="bilinear")
        )
        # 没有 binding 时直接返回普通结果。
        if binding is None:
            return result
        # 有 binding 时，把 resize 语义写回 binding 元信息。
        return self._bind_weight_slice(
            result,
            binding=_PackedWeightSliceBinding(
                slices=self._reshape_binding(
                    binding,
                    raw_shape=tuple(source.shape),
                ).slices,
                resize_shape=size,
                resize_mode="bilinear",
                expert_index=binding.expert_index,
                transpose_last_two=binding.transpose_last_two,
                tanh_scale=binding.tanh_scale,
                add_scalar=binding.add_scalar,
            ),
        )

    def _resize_expert_tensor(
        self,
        tensor: torch.Tensor,
        *,
        size: tuple[int, int, int],
    ) -> torch.Tensor:
        # 目标 expert/rows/cols 任一非正时直接返回空三维张量。
        experts, rows, cols = size
        if experts <= 0 or rows <= 0 or cols <= 0:
            return torch.empty(
                experts,
                rows,
                cols,
                dtype=torch.float32,
                device=tensor.device,
            )
        # 先稳定化，并保留可能存在的 packed binding。
        source = self._stabilize_bound(tensor)
        binding = self._weight_binding_for_tensor(source)
        # 一维输入时先扩成 [1, 1, n]。
        if source.ndim == 1:
            source = source.view(1, 1, -1)
            if binding is not None:
                source = self._bind_weight_slice(
                    source,
                    binding=self._reshape_binding(
                        binding,
                        raw_shape=tuple(source.shape),
                    ),
                )
        # 二维输入时先扩成 [1, rows, cols]。
        elif source.ndim == 2:
            source = source.unsqueeze(0)
            if binding is not None:
                source = self._bind_weight_slice(
                    source,
                    binding=self._reshape_binding(
                        binding,
                        raw_shape=tuple(source.shape),
                    ),
                )
        # 形状已匹配时直接返回。
        if tuple(source.shape) == size:
            return source
        # 否则走三维 trilinear resize。
        result = self._stabilize_tensor(
            resize_tensor(source, size=size, mode="trilinear")
        )
        # 没有 binding 时直接返回普通结果。
        if binding is None:
            return result
        # 有 binding 时，把 resize 语义写回 binding 元信息。
        return self._bind_weight_slice(
            result,
            binding=_PackedWeightSliceBinding(
                slices=self._reshape_binding(
                    binding,
                    raw_shape=tuple(source.shape),
                ).slices,
                resize_shape=size,
                resize_mode="trilinear",
                expert_index=binding.expert_index,
                transpose_last_two=binding.transpose_last_two,
                tanh_scale=binding.tanh_scale,
                add_scalar=binding.add_scalar,
            ),
        )

    def _semantic_scale_vector(
        self,
        tensor: torch.Tensor,
        *,
        layer_entries: tuple[ParameterSourceSlice, ...],
        roles: tuple[str, ...],
        size: int,
    ) -> torch.Tensor:
        # 先收集当前层、当前 roles 下可用的 block。
        role_blocks = self._role_blocks(
            tensor,
            layer_entries=layer_entries,
            roles=roles,
        )
        # 命中 block 时优先走“逐 block resize + 平均”路径。
        if role_blocks:
            vectors = [
                self._resize_vector(block, size=size)
                for block in role_blocks
            ]
            # 多个 block 时对齐后取平均，得到当前 scale 向量。
            raw = torch.stack(vectors, dim=0).mean(dim=0)
            # 尝试把多个 vector 的 binding 合并回一个统一 binding。
            bindings = [self._weight_binding_for_tensor(vector) for vector in vectors]
            if all(binding is not None for binding in bindings):
                first = bindings[0]
                assert first is not None
                if all(
                    binding is not None
                    and binding.resize_shape == first.resize_shape
                    and binding.resize_mode == first.resize_mode
                    and binding.expert_index == first.expert_index
                    and binding.transpose_last_two == first.transpose_last_two
                    and binding.tanh_scale == first.tanh_scale
                    and binding.add_scalar == first.add_scalar
                    for binding in bindings
                ):
                    raw = self._bind_weight_slice(
                        raw,
                        binding=_PackedWeightSliceBinding(
                            slices=tuple(
                                entry
                                for binding in bindings
                                if binding is not None
                                for entry in binding.slices
                            ),
                            resize_shape=first.resize_shape,
                            resize_mode=first.resize_mode,
                            expert_index=first.expert_index,
                            transpose_last_two=first.transpose_last_two,
                            tanh_scale=first.tanh_scale,
                            add_scalar=first.add_scalar,
                        ),
                    )
        else:
            # 没有 block 时退回线性值展开路径。
            raw = self._expand_values(
                self._role_values(tensor, layer_entries=layer_entries, roles=roles),
                size=size,
            )
        # scale 向量最后统一经过 tanh+偏置，并解析可能存在的 binding。
        return self._resolve_bound_value(
            self._add_scalar_bound(
                self._tanh_scale_bound(raw, scale=0.05),
                value=1.0,
            )
        )

    def _semantic_vector(
        self,
        tensor: torch.Tensor,
        *,
        layer_entries: tuple[ParameterSourceSlice, ...],
        roles: tuple[str, ...],
        size: int,
    ) -> torch.Tensor:
        # 先收集当前层、当前 roles 下可用的 block。
        role_blocks = self._role_blocks(
            tensor,
            layer_entries=layer_entries,
            roles=roles,
        )
        # 命中 block 时优先走“逐 block resize + 平均”路径。
        if role_blocks:
            vectors = [
                self._resize_vector(block, size=size)
                for block in role_blocks
            ]
            # 多个 block 时对齐后取平均，得到当前语义向量。
            raw = torch.stack(vectors, dim=0).mean(dim=0)
            # 尝试把多个 vector 的 binding 合并回一个统一 binding。
            bindings = [self._weight_binding_for_tensor(vector) for vector in vectors]
            if all(binding is not None for binding in bindings):
                first = bindings[0]
                assert first is not None
                if all(
                    binding is not None
                    and binding.resize_shape == first.resize_shape
                    and binding.resize_mode == first.resize_mode
                    and binding.expert_index == first.expert_index
                    and binding.transpose_last_two == first.transpose_last_two
                    and binding.tanh_scale == first.tanh_scale
                    and binding.add_scalar == first.add_scalar
                    for binding in bindings
                ):
                    raw = self._bind_weight_slice(
                        raw,
                        binding=_PackedWeightSliceBinding(
                            slices=tuple(
                                entry
                                for binding in bindings
                                if binding is not None
                                for entry in binding.slices
                            ),
                            resize_shape=first.resize_shape,
                            resize_mode=first.resize_mode,
                            expert_index=first.expert_index,
                            transpose_last_two=first.transpose_last_two,
                            tanh_scale=first.tanh_scale,
                            add_scalar=first.add_scalar,
                        ),
                    )
        else:
            # 没有 block 时退回线性值展开路径。
            raw = self._expand_values(
                self._role_values(tensor, layer_entries=layer_entries, roles=roles),
                size=size,
            )
        # 普通语义向量只做 tanh 缩放，不额外加偏置。
        return self._resolve_bound_value(self._tanh_scale_bound(raw, scale=0.05))

    def _semantic_matrix(
        self,
        tensor: torch.Tensor,
        *,
        layer_entries: tuple[ParameterSourceSlice, ...],
        roles: tuple[str, ...],
        rows: int,
        cols: int,
        transpose_source: bool = True,
    ) -> torch.Tensor:
        # 先收集当前层、当前 roles 下可用的 block。
        role_blocks = self._role_blocks(
            tensor,
            layer_entries=layer_entries,
            roles=roles,
        )
        # 命中 block 时优先走“逐 block resize + 平均”路径。
        if role_blocks:
            matrices = []
            # 某些源矩阵需要先按转置前的 shape 做 resize。
            pre_transpose_shape = (cols, rows) if transpose_source else (rows, cols)
            for block in role_blocks:
                # 先把 block resize 到目标矩阵大小。
                resized = self._resize_matrix(block, size=pre_transpose_shape)
                if transpose_source:
                    # 需要时再把最后两维转置到真正使用的方向。
                    resized = self._transpose_last_two_bound(resized)
                matrices.append(resized)
            # 单 block 直接复用；多 block 时对齐后取平均。
            if len(matrices) == 1:
                raw = matrices[0]
            else:
                raw = torch.stack(matrices, dim=0).mean(dim=0)
                # 尝试把多个 matrix 的 binding 合并回一个统一 binding。
                bindings = [
                    self._weight_binding_for_tensor(matrix) for matrix in matrices
                ]
                if all(binding is not None for binding in bindings):
                    first = bindings[0]
                    assert first is not None
                    if all(
                        binding is not None
                        and binding.resize_shape == first.resize_shape
                        and binding.resize_mode == first.resize_mode
                        and binding.expert_index == first.expert_index
                        and binding.transpose_last_two == first.transpose_last_two
                        and binding.tanh_scale == first.tanh_scale
                        and binding.add_scalar == first.add_scalar
                        for binding in bindings
                    ):
                        raw = self._bind_weight_slice(
                            raw,
                            binding=_PackedWeightSliceBinding(
                                slices=tuple(
                                    entry
                                    for binding in bindings
                                    if binding is not None
                                    for entry in binding.slices
                                ),
                                resize_shape=first.resize_shape,
                                resize_mode=first.resize_mode,
                                expert_index=first.expert_index,
                                transpose_last_two=first.transpose_last_two,
                                tanh_scale=first.tanh_scale,
                                add_scalar=first.add_scalar,
                            ),
                        )
        else:
            # 没有 block 时退回线性值展开并 reshape 成矩阵。
            raw = self._expand_values(
                self._role_values(tensor, layer_entries=layer_entries, roles=roles),
                size=rows * cols,
            ).view(rows, cols)
        # 最终矩阵统一经过轻量 tanh 缩放。
        return self._tanh_scale_bound(raw, scale=0.05)

    def _semantic_expert_tensor(
        self,
        tensor: torch.Tensor,
        *,
        layer_entries: tuple[ParameterSourceSlice, ...],
        roles: tuple[str, ...],
        experts: int,
        rows: int,
        cols: int,
        transpose_last_two: bool = True,
    ) -> torch.Tensor:
        # 先收集当前层、当前 roles 下可用的 block。
        role_blocks = self._role_blocks(
            tensor,
            layer_entries=layer_entries,
            roles=roles,
        )
        # 命中 block 时优先走“逐 block resize + 平均”路径。
        if role_blocks:
            tensors = []
            # 某些源 expert 张量需要先按转置前的 shape 做 resize。
            pre_transpose_shape = (
                (experts, cols, rows) if transpose_last_two else (experts, rows, cols)
            )
            for block in role_blocks:
                # 先把 block resize 到目标 expert 张量大小。
                resized = self._resize_expert_tensor(
                    block,
                    size=pre_transpose_shape,
                )
                if transpose_last_two:
                    # 需要时再把最后两维转置到真正使用的方向。
                    resized = self._transpose_last_two_bound(resized)
                tensors.append(resized)
            # 单 block 直接复用；多 block 时对齐后取平均。
            if len(tensors) == 1:
                raw = tensors[0]
            else:
                raw = torch.stack(tensors, dim=0).mean(dim=0)
                # 尝试把多个 tensor block 的 binding 合并回一个统一 binding。
                bindings = [
                    self._weight_binding_for_tensor(tensor_block)
                    for tensor_block in tensors
                ]
                if all(binding is not None for binding in bindings):
                    first = bindings[0]
                    assert first is not None
                    if all(
                        binding is not None
                        and binding.resize_shape == first.resize_shape
                        and binding.resize_mode == first.resize_mode
                        and binding.expert_index == first.expert_index
                        and binding.transpose_last_two == first.transpose_last_two
                        and binding.tanh_scale == first.tanh_scale
                        and binding.add_scalar == first.add_scalar
                        for binding in bindings
                    ):
                        raw = self._bind_weight_slice(
                            raw,
                            binding=_PackedWeightSliceBinding(
                                slices=tuple(
                                    entry
                                    for binding in bindings
                                    if binding is not None
                                    for entry in binding.slices
                                ),
                                resize_shape=first.resize_shape,
                                resize_mode=first.resize_mode,
                                expert_index=first.expert_index,
                                transpose_last_two=first.transpose_last_two,
                                tanh_scale=first.tanh_scale,
                                add_scalar=first.add_scalar,
                            ),
                        )
        else:
            # 没有 block 时退回线性值展开并 reshape 成 expert 张量。
            raw = self._expand_values(
                self._role_values(tensor, layer_entries=layer_entries, roles=roles),
                size=experts * rows * cols,
            ).view(experts, rows, cols)
        # 最终 expert 张量统一经过轻量 tanh 缩放。
        return self._tanh_scale_bound(raw, scale=0.05)

    def _build_semantic_non_routed_layer_param(
        self,
        *,
        tensor: torch.Tensor,
        layer_entries: tuple[ParameterSourceSlice, ...],
        graph_shape: _RepresentativeBucketGraphShape,
        attention_type: str,
    ) -> torch.Tensor:
        # full_attention / mtp 路径和 linear_attention 路径使用不同的语义角色集合。
        if attention_type in {"full_attention", "mtp"}:
            norm_roles = (
                "input_layernorm",
                "self_attn_q_norm",
                "self_attn_k_norm",
            )
            q_roles = ("self_attn_q_proj",)
            k_roles = ("self_attn_k_proj",)
            v_roles = ("self_attn_v_proj",)
            o_roles = ("self_attn_o_proj",)
            residual_roles = (
                "post_attention_layernorm",
                "shared_expert_gate",
            )
        else:
            norm_roles = (
                "input_layernorm",
                "linear_attn_norm",
            )
            q_roles = ("linear_attn_in_proj_qkv",)
            k_roles = (
                "linear_attn_in_proj_a",
                "linear_attn_in_proj_qkv",
            )
            v_roles = (
                "linear_attn_in_proj_b",
                "linear_attn_in_proj_qkv",
            )
            o_roles = ("linear_attn_out_proj",)
            residual_roles = (
                "post_attention_layernorm",
                "linear_attn_dt_bias",
                "linear_attn_A_log",
                "linear_attn_in_proj_z",
                "linear_attn_conv1d",
                "shared_expert_gate",
            )
        # 按固定顺序重建一层非专家参数的所有语义部件。
        pieces = (
            self._semantic_scale_vector(
                tensor,
                layer_entries=layer_entries,
                roles=norm_roles,
                size=graph_shape.hidden_dim,
            ),
            self._semantic_matrix(
                tensor,
                layer_entries=layer_entries,
                roles=q_roles,
                rows=graph_shape.hidden_dim,
                cols=graph_shape.hidden_dim,
            ),
            self._semantic_matrix(
                tensor,
                layer_entries=layer_entries,
                roles=k_roles,
                rows=graph_shape.hidden_dim,
                cols=graph_shape.hidden_dim,
            ),
            self._semantic_matrix(
                tensor,
                layer_entries=layer_entries,
                roles=v_roles,
                rows=graph_shape.hidden_dim,
                cols=graph_shape.hidden_dim,
            ),
            self._semantic_matrix(
                tensor,
                layer_entries=layer_entries,
                roles=o_roles,
                rows=graph_shape.hidden_dim,
                cols=graph_shape.hidden_dim,
            ),
            self._semantic_matrix(
                tensor,
                layer_entries=layer_entries,
                roles=("mlp_router_gate",),
                rows=graph_shape.hidden_dim,
                cols=graph_shape.active_expert_count,
            ),
            self._semantic_matrix(
                tensor,
                layer_entries=layer_entries,
                roles=(
                    "shared_expert_up_proj",
                    "shared_expert_gate_proj",
                ),
                rows=graph_shape.hidden_dim,
                cols=graph_shape.expert_hidden_dim,
            ),
            self._semantic_matrix(
                tensor,
                layer_entries=layer_entries,
                roles=("shared_expert_down_proj",),
                rows=graph_shape.expert_hidden_dim,
                cols=graph_shape.hidden_dim,
            ),
            self._semantic_vector(
                tensor,
                layer_entries=layer_entries,
                roles=residual_roles,
                size=graph_shape.hidden_dim,
            ),
        )
        # 最终把所有语义部件展平后拼成单层非专家参数向量。
        return torch.cat([piece.reshape(-1) for piece in pieces], dim=0)

    def _build_semantic_routed_layer_param(
        self,
        *,
        tensor: torch.Tensor,
        layer_entries: tuple[ParameterSourceSlice, ...],
        graph_shape: _RepresentativeBucketGraphShape,
    ) -> torch.Tensor:
        # 专家参数只需重建 gate/up 与 down 两类核心部件。
        pieces = (
            self._semantic_matrix(
                tensor,
                layer_entries=layer_entries,
                roles=("expert_gate_up_proj",),
                rows=graph_shape.active_expert_count * graph_shape.hidden_dim,
                cols=graph_shape.expert_hidden_dim,
            ),
            self._semantic_matrix(
                tensor,
                layer_entries=layer_entries,
                roles=("expert_down_proj",),
                rows=graph_shape.active_expert_count * graph_shape.expert_hidden_dim,
                cols=graph_shape.hidden_dim,
            ),
        )
        # 最终把所有语义部件展平后拼成单层 routed 参数向量。
        return torch.cat([piece.reshape(-1) for piece in pieces], dim=0)

    def _execute_semantic_layer(
        self,
        *,
        hidden_states: torch.Tensor,
        non_routed_param: torch.Tensor,
        routed_param: torch.Tensor,
        graph_shape: _RepresentativeBucketGraphShape,
        attention_type: str,
        layer_index: int,
        bucket_id: int,
        expert_ids: tuple[int, ...],
        non_routed_entries: tuple[ParameterSourceSlice, ...],
        routed_entries: tuple[ParameterSourceSlice, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        # -----------------
        # 先从语义布局中提取本层共用的 norm、router、shared expert 参数。
        input_norm = self._semantic_scale_vector(
            non_routed_param,
            layer_entries=non_routed_entries,
            roles=("input_layernorm",),
            size=graph_shape.hidden_dim,
        )
        router_weight = self._semantic_matrix(
            non_routed_param,
            layer_entries=non_routed_entries,
            roles=("mlp_router_gate",),
            rows=graph_shape.hidden_dim,
            cols=graph_shape.active_expert_count,
        )
        shared_gate_proj = self._semantic_matrix(
            non_routed_param,
            layer_entries=non_routed_entries,
            roles=("shared_expert_gate_proj",),
            rows=graph_shape.hidden_dim,
            cols=graph_shape.expert_hidden_dim,
        )
        shared_up_proj = self._semantic_matrix(
            non_routed_param,
            layer_entries=non_routed_entries,
            roles=("shared_expert_up_proj",),
            rows=graph_shape.hidden_dim,
            cols=graph_shape.expert_hidden_dim,
        )
        shared_down_proj = self._semantic_matrix(
            non_routed_param,
            layer_entries=non_routed_entries,
            roles=("shared_expert_down_proj",),
            rows=graph_shape.expert_hidden_dim,
            cols=graph_shape.hidden_dim,
        )
        shared_expert_gate = self._semantic_vector(
            non_routed_param,
            layer_entries=non_routed_entries,
            roles=("shared_expert_gate",),
            size=graph_shape.hidden_dim,
        )
        residual_bias = self._semantic_vector(
            non_routed_param,
            layer_entries=non_routed_entries,
            roles=("post_attention_layernorm",),
            size=graph_shape.hidden_dim,
        )

        # 先对输入 hidden 做归一化。
        normalized = self._rms_norm_scaled(hidden_states, input_norm)
        # 再计算本层 router logits。
        router_logits = self._stabilize_tensor(
            self._linear_projection(normalized, router_weight),
            max_abs=20.0,
        )

        # -----------------
        # 按 attention_type 选择 full attention 或 linear attention 路径。
        if attention_type == "full_attention":
            # 提取 full attention 路径需要的 q/k/v/o 权重和 q/k norm。
            q_weight = self._semantic_matrix(
                non_routed_param,
                layer_entries=non_routed_entries,
                roles=("self_attn_q_proj",),
                rows=graph_shape.hidden_dim,
                cols=graph_shape.hidden_dim,
            )
            k_weight = self._semantic_matrix(
                non_routed_param,
                layer_entries=non_routed_entries,
                roles=("self_attn_k_proj",),
                rows=graph_shape.hidden_dim,
                cols=graph_shape.hidden_dim,
            )
            v_weight = self._semantic_matrix(
                non_routed_param,
                layer_entries=non_routed_entries,
                roles=("self_attn_v_proj",),
                rows=graph_shape.hidden_dim,
                cols=graph_shape.hidden_dim,
            )
            o_weight = self._semantic_matrix(
                non_routed_param,
                layer_entries=non_routed_entries,
                roles=("self_attn_o_proj",),
                rows=graph_shape.hidden_dim,
                cols=graph_shape.hidden_dim,
            )
            q_norm = self._semantic_scale_vector(
                non_routed_param,
                layer_entries=non_routed_entries,
                roles=("self_attn_q_norm",),
                size=graph_shape.hidden_dim,
            )
            k_norm = self._semantic_scale_vector(
                non_routed_param,
                layer_entries=non_routed_entries,
                roles=("self_attn_k_norm",),
                size=graph_shape.hidden_dim,
            )
            # 先构造 q/k/v。
            q_base = self._linear_projection(normalized, q_weight)
            q = self._rms_norm_scaled(q_base, q_norm)
            k = self._rms_norm_scaled(
                self._linear_projection(normalized, k_weight),
                k_norm,
            )
            v = self._stabilize_tensor(self._linear_projection(normalized, v_weight))
            # 执行 full attention 上下文计算。
            context = self._full_attention_context(q, k, v, graph_shape)
            # 用 q_weight 的转置再构造一个 attention gate。
            attn_gate = torch.sigmoid(
                self._linear_projection(
                    normalized,
                    self._transpose_last_two_bound(q_weight),
                )
            )
            # 输出投影前先用 gate 调制上下文。
            attention_out = self._stabilize_tensor(
                self._linear_projection(context * attn_gate, o_weight)
            )
        else:
            # 提取 linear attention 路径所需的各类投影权重和辅助向量。
            q_weight = self._semantic_matrix(
                non_routed_param,
                layer_entries=non_routed_entries,
                roles=("linear_attn_in_proj_qkv",),
                rows=graph_shape.hidden_dim,
                cols=graph_shape.hidden_dim,
            )
            k_weight = self._semantic_matrix(
                non_routed_param,
                layer_entries=non_routed_entries,
                roles=("linear_attn_in_proj_a", "linear_attn_in_proj_qkv"),
                rows=graph_shape.hidden_dim,
                cols=graph_shape.hidden_dim,
            )
            v_weight = self._semantic_matrix(
                non_routed_param,
                layer_entries=non_routed_entries,
                roles=("linear_attn_in_proj_b", "linear_attn_in_proj_qkv"),
                rows=graph_shape.hidden_dim,
                cols=graph_shape.hidden_dim,
            )
            z_weight = self._semantic_matrix(
                non_routed_param,
                layer_entries=non_routed_entries,
                roles=("linear_attn_in_proj_z", "linear_attn_in_proj_qkv"),
                rows=graph_shape.hidden_dim,
                cols=graph_shape.hidden_dim,
            )
            a_weight = self._semantic_matrix(
                non_routed_param,
                layer_entries=non_routed_entries,
                roles=("linear_attn_in_proj_a",),
                rows=graph_shape.hidden_dim,
                cols=graph_shape.hidden_dim,
            )
            conv_weight = self._semantic_matrix(
                non_routed_param,
                layer_entries=non_routed_entries,
                roles=("linear_attn_conv1d",),
                rows=graph_shape.hidden_dim,
                cols=graph_shape.hidden_dim,
            )
            o_weight = self._semantic_matrix(
                non_routed_param,
                layer_entries=non_routed_entries,
                roles=("linear_attn_out_proj",),
                rows=graph_shape.hidden_dim,
                cols=graph_shape.hidden_dim,
            )
            linear_norm = self._semantic_scale_vector(
                non_routed_param,
                layer_entries=non_routed_entries,
                roles=("linear_attn_norm",),
                size=graph_shape.hidden_dim,
            )
            dt_bias = self._semantic_vector(
                non_routed_param,
                layer_entries=non_routed_entries,
                roles=("linear_attn_dt_bias",),
                size=graph_shape.hidden_dim,
            )
            a_log = self._semantic_vector(
                non_routed_param,
                layer_entries=non_routed_entries,
                roles=("linear_attn_A_log",),
                size=graph_shape.hidden_dim,
            )
            # 先构造 q/k/v。
            q = self._stabilize_tensor(self._linear_projection(normalized, q_weight))
            k = self._stabilize_tensor(self._linear_projection(normalized, k_weight))
            v = self._stabilize_tensor(self._linear_projection(normalized, v_weight))
            # z 和 delta 用于混合线性 attention 与局部状态。
            z = torch.sigmoid(
                self._linear_projection(normalized, z_weight) + dt_bias.unsqueeze(0)
            )
            delta = torch.sigmoid(
                self._linear_projection(normalized, a_weight) + dt_bias.unsqueeze(0)
            )
            # forget 由 A_log 经过 softplus/exp 转成衰减系数。
            forget = torch.exp(-F.softplus(a_log)).unsqueeze(0)
            # 执行 linear attention 上下文计算。
            linear_context = self._linear_attention_context(q, k, v, graph_shape)
            # 局部状态由前一 token hidden 和 conv1d 路径混合得到。
            local_state = 0.5 * (
                torch.roll(normalized, shifts=1, dims=0)
                + self._linear_projection(normalized, conv_weight)
            )
            # 用 z/delta/forget 混合上下文、局部状态和残留输入。
            mixed = z * linear_context + (1.0 - z) * (
                delta * local_state + forget * normalized
            )
            # 再做归一化和输出投影。
            context = self._rms_norm_scaled(mixed, linear_norm)
            attention_out = self._stabilize_tensor(
                self._linear_projection(context, o_weight)
            )

        # -----------------
        # 继续执行 shared expert 和 routed MoE 路径。
        shared_out = self._shared_expert_forward(
            hidden_states=normalized,
            gate_proj=shared_gate_proj,
            up_proj=shared_up_proj,
            down_proj=shared_down_proj,
            expert_gate=shared_expert_gate,
        )
        # 取出 gate+up 的拼接专家张量，并沿最后一维拆成 gate/up 两块。
        expert_gate_up = self._semantic_expert_tensor(
            routed_param,
            layer_entries=routed_entries,
            roles=("expert_gate_up_proj",),
            experts=graph_shape.active_expert_count,
            rows=graph_shape.hidden_dim,
            cols=graph_shape.expert_hidden_dim * 2,
        )
        expert_gate, expert_up = torch.chunk(expert_gate_up, 2, dim=-1)
        # 取出 expert down 张量。
        expert_down = self._semantic_expert_tensor(
            routed_param,
            layer_entries=routed_entries,
            roles=("expert_down_proj",),
            experts=graph_shape.active_expert_count,
            rows=graph_shape.expert_hidden_dim,
            cols=graph_shape.hidden_dim,
        )
        # 执行 routed MoE 混合，并得到 routing balance loss。
        moe_out, routing_balance_loss = self._moe_forward_gated(
            hidden_states=normalized,
            router_logits=router_logits,
            expert_gate=expert_gate,
            expert_up=expert_up,
            expert_down=expert_down,
            expert_ids=expert_ids,
            topk=graph_shape.topk,
        )
        # residual_bias 会随 bucket/layer 位置施加一个轻量放大系数。
        layer_bias = residual_bias.unsqueeze(0) * (
            1.0 + 0.03 * float(bucket_id + layer_index + 1)
        )
        # 下一层 hidden 由残差输入、attention、shared expert、MoE 和 layer_bias 共同组成。
        next_hidden = self._stabilize_tensor(
            hidden_states + attention_out + shared_out + moe_out + layer_bias
        )
        # 先构造一个基础 shifted target。
        shifted_hidden = torch.roll(hidden_states, shifts=-1, dims=0)
        # attention_type 不同，对 target 的构造方式也不同。
        if attention_type == "full_attention":
            target = shifted_hidden + 0.05 * torch.flip(hidden_states, dims=(0,))
        elif attention_type == "mtp":
            target = self._mtp_target(hidden_states)
        else:
            # linear attention 路径额外注入 prefix hidden 的平滑趋势。
            prefix_hidden = torch.cumsum(hidden_states, dim=0)
            prefix_scale = torch.arange(
                1,
                graph_shape.sequence_length + 1,
                dtype=torch.float32,
                device=hidden_states.device,
            ).unsqueeze(-1)
            target = shifted_hidden + 0.03 * (prefix_hidden / prefix_scale)
        # 对 attention/shared/MoE 输出分别加一个轻量正则。
        semantic_regularizer = (
            0.01 * attention_out.square().mean()
            + 0.01 * shared_out.square().mean()
            + 0.01 * moe_out.square().mean()
        )
        # 主损失由 hidden 重建损失、routing 平衡损失、正则项和参数 L2 共同组成。
        layer_loss = (
            F.mse_loss(next_hidden, target)
            + 0.05 * routing_balance_loss
            + semantic_regularizer
            + 0.01 * non_routed_param.square().mean()
            + 0.01 * routed_param.square().mean()
        )
        # MTP 路径额外鼓励 next_hidden 与 shifted_hidden 的变化更平滑。
        if attention_type == "mtp":
            layer_loss = layer_loss + 0.02 * (
                next_hidden - shifted_hidden
            ).abs().mean()
        # 最后裁掉 NaN/Inf，保证 loss 稳定。
        layer_loss = torch.nan_to_num(
            layer_loss,
            nan=0.0,
            posinf=1e3,
            neginf=1e3,
        )
        # 用本层主要中间张量估算激活峰值字节数。
        peak_bytes = int(
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    hidden_states,
                    normalized,
                    router_logits,
                    attention_out,
                    shared_out,
                    moe_out,
                    next_hidden,
                    target,
                )
            )
        )
        return next_hidden, layer_loss, peak_bytes

    def _execute_layer(
        self,
        *,
        hidden_states: torch.Tensor,
        non_routed_param: torch.Tensor,
        routed_param: torch.Tensor,
        graph_shape: _RepresentativeBucketGraphShape,
        attention_type: str,
        layer_index: int,
        bucket_id: int,
        expert_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        # -----------------
        # 先按固定布局从非专家参数里切出 norm / attention / shared expert 部件。
        # cursor 负责在线性参数向量中推进当前读取位置。
        cursor = 0
        # 读取 input norm 缩放向量。
        norm_scale, cursor = self._take_vector(
            non_routed_param,
            cursor,
            graph_shape.hidden_dim,
        )
        # 读取 q 投影矩阵。
        q_weight, cursor = self._take_matrix(
            non_routed_param,
            cursor,
            graph_shape.hidden_dim,
            graph_shape.hidden_dim,
        )
        # 读取 k 投影矩阵。
        k_weight, cursor = self._take_matrix(
            non_routed_param,
            cursor,
            graph_shape.hidden_dim,
            graph_shape.hidden_dim,
        )
        # 读取 v 投影矩阵。
        v_weight, cursor = self._take_matrix(
            non_routed_param,
            cursor,
            graph_shape.hidden_dim,
            graph_shape.hidden_dim,
        )
        # 读取 attention 输出投影矩阵。
        o_weight, cursor = self._take_matrix(
            non_routed_param,
            cursor,
            graph_shape.hidden_dim,
            graph_shape.hidden_dim,
        )
        # 读取 router gate 权重。
        gate_weight, cursor = self._take_matrix(
            non_routed_param,
            cursor,
            graph_shape.hidden_dim,
            graph_shape.active_expert_count,
        )
        # 读取 shared expert 的 up 投影。
        shared_up, cursor = self._take_matrix(
            non_routed_param,
            cursor,
            graph_shape.hidden_dim,
            graph_shape.expert_hidden_dim,
        )
        # 读取 shared expert 的 down 投影。
        shared_down, cursor = self._take_matrix(
            non_routed_param,
            cursor,
            graph_shape.expert_hidden_dim,
            graph_shape.hidden_dim,
        )
        # 读取层末残差偏置向量。
        residual_bias, cursor = self._take_vector(
            non_routed_param,
            cursor,
            graph_shape.hidden_dim,
        )

        # -----------------
        # 再按固定布局从 routed 参数里切出 expert up/down 张量。
        # routed_cursor 单独追踪专家参数线性读取位置。
        routed_cursor = 0
        # 读取所有 active experts 的 up 投影块。
        expert_up, routed_cursor = self._take_matrix(
            routed_param,
            routed_cursor,
            graph_shape.active_expert_count * graph_shape.hidden_dim,
            graph_shape.expert_hidden_dim,
        )
        # 将线性专家 up 参数恢复成 [expert, hidden, expert_hidden]。
        expert_up = expert_up.view(
            graph_shape.active_expert_count,
            graph_shape.hidden_dim,
            graph_shape.expert_hidden_dim,
        )
        # 读取所有 active experts 的 down 投影块。
        expert_down, routed_cursor = self._take_matrix(
            routed_param,
            routed_cursor,
            graph_shape.active_expert_count * graph_shape.expert_hidden_dim,
            graph_shape.hidden_dim,
        )
        # 将线性专家 down 参数恢复成 [expert, expert_hidden, hidden]。
        expert_down = expert_down.view(
            graph_shape.active_expert_count,
            graph_shape.expert_hidden_dim,
            graph_shape.hidden_dim,
        )

        # -----------------
        # 执行 attention 前的归一化与 q/k/v 投影。
        # 先做 layer_norm 近似归一化。
        normalized = self._stabilize_tensor(
            F.layer_norm(hidden_states, (graph_shape.hidden_dim,))
        )
        # 再乘以输入缩放向量。
        normalized = self._stabilize_tensor(normalized * norm_scale.unsqueeze(0))
        # 计算 q 表示。
        q = self._stabilize_tensor(self._linear_projection(normalized, q_weight))
        # 计算 k 表示。
        k = self._stabilize_tensor(self._linear_projection(normalized, k_weight))
        # 计算 v 表示。
        v = self._stabilize_tensor(self._linear_projection(normalized, v_weight))
        # attention_type 决定当前层走 full attention 还是 linear attention。
        if attention_type in {"full_attention", "mtp"}:
            # full_attention / mtp 共用标准 attention 上下文。
            context = self._full_attention_context(q, k, v, graph_shape)
        else:
            # 其余路径走线性 attention 上下文。
            context = self._linear_attention_context(q, k, v, graph_shape)
        # attention 上下文经过输出投影得到 attention_out。
        attention_out = self._stabilize_tensor(
            self._linear_projection(context, o_weight)
        )
        # router logits 用于后续 routed MoE 的 top-k 选择。
        router_logits = self._stabilize_tensor(
            self._linear_projection(normalized, gate_weight),
            max_abs=20.0,
        )

        # -----------------
        # 执行 shared expert 与 routed expert 前向。
        # shared expert 先做 up 投影并经过 SiLU。
        shared_hidden = self._stabilize_tensor(
            F.silu(self._linear_projection(normalized, shared_up))
        )
        # 再做 down 投影得到 shared expert 输出。
        shared_out = self._stabilize_tensor(
            self._linear_projection(shared_hidden, shared_down)
        )
        # routed MoE 使用 router logits 对 active experts 做 top-k 混合。
        moe_out, routing_balance_loss = self._moe_forward(
            hidden_states=normalized,
            router_logits=router_logits,
            expert_up=expert_up,
            expert_down=expert_down,
            expert_ids=expert_ids,
            topk=graph_shape.topk,
        )
        # 残差偏置会随 bucket/layer 位置施加轻量放大。
        layer_bias = residual_bias.unsqueeze(0) * (
            1.0 + 0.03 * float(bucket_id + layer_index + 1)
        )
        # 下一层 hidden 由输入残差、attention、shared 和 MoE 共同组成。
        next_hidden = self._stabilize_tensor(
            hidden_states + attention_out + shared_out + moe_out + layer_bias
        )

        # -----------------
        # 构造监督目标，并计算当前层代表性损失。
        # 默认目标是下一 token hidden。
        shifted_hidden = torch.roll(hidden_states, shifts=-1, dims=0)
        # full attention / mtp / linear attention 的 target 细节各不相同。
        if attention_type == "full_attention":
            # full attention 额外引入反向序列的轻量扰动。
            target = shifted_hidden + 0.05 * torch.flip(hidden_states, dims=(0,))
        elif attention_type == "mtp":
            # mtp 使用专门的多 token 预测目标。
            target = self._mtp_target(hidden_states)
        else:
            # linear attention 额外注入 prefix 趋势项。
            prefix_hidden = torch.cumsum(hidden_states, dim=0)
            prefix_scale = torch.arange(
                1,
                graph_shape.sequence_length + 1,
                dtype=torch.float32,
                device=hidden_states.device,
            ).unsqueeze(-1)
            target = shifted_hidden + 0.03 * (prefix_hidden / prefix_scale)
        # 主损失由隐藏态重建、routing 平衡和参数正则组成。
        layer_loss = (
            F.mse_loss(next_hidden, target)
            + 0.05 * routing_balance_loss
            + 0.0005 * non_routed_param.square().mean()
            + 0.001 * routed_param.square().mean()
        )
        # mtp 再增加一项相邻目标平滑约束。
        if attention_type == "mtp":
            layer_loss = layer_loss + 0.02 * (
                next_hidden - shifted_hidden
            ).abs().mean()
        # 最后把 NaN/Inf 裁掉，保证 loss 稳定。
        layer_loss = torch.nan_to_num(
            layer_loss,
            nan=0.0,
            posinf=1e3,
            neginf=1e3,
        )

        # -----------------
        # 用主要中间张量估算本层激活峰值字节数。
        peak_bytes = int(
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    hidden_states,
                    normalized,
                    q,
                    k,
                    v,
                    context,
                    attention_out,
                    router_logits,
                    shared_hidden,
                    shared_out,
                    moe_out,
                    next_hidden,
                    target,
                )
            )
        )
        # 返回下一层 hidden、当前层损失和峰值字节数。
        return next_hidden, layer_loss, peak_bytes

    def select_bucket_shards(
        self,
        *,
        bucket_id: int,
        parameter_shards: tuple[ParameterShardSnapshot, ...],
    ) -> tuple[ParameterShardSnapshot, ParameterShardSnapshot]:
        # 先定位当前 bucket 对应的非专家参数分片。
        non_routed = next(
            shard
            for shard in parameter_shards
            if shard.component == "bucket_non_routed" and shard.bucket_id == bucket_id
        )
        # 再定位当前 bucket 对应的 active expert 参数分片。
        routed = next(
            shard
            for shard in parameter_shards
            if shard.component == "bucket_active_experts" and shard.bucket_id == bucket_id
        )
        # 返回“非专家分片 + 专家分片”二元组。
        return non_routed, routed

    def execute_bucket(
        self,
        *,
        step_index: int,
        batch: BatchShape,
        bucket: LayerBucketPlan,
        parameter_shards: tuple[ParameterShardSnapshot, ...],
        parameter_store: ParameterShardStore,
    ) -> RepresentativeBucketExecutionResult:
        # -----------------
        # 先准备当前 bucket 的参数视图和量化绑定上下文。
        self._reset_quantized_bindings()
        # 解析当前 bucket 对应的两类参数分片。
        non_routed_shard, routed_shard = self.select_bucket_shards(
            bucket_id=bucket.bucket_id,
            parameter_shards=parameter_shards,
        )
        # logical 物化模式下，梯度会按层局部切片回传。
        force_layer_local_gradients = (
            self.config.execution.trainable_shard_materialization == "logical"
        )
        # full_bucket logical CUDA 模式允许直接用整块 device 参数视图。
        full_bucket_logical_cuda = (
            force_layer_local_gradients and self._full_bucket_logical_cuda_enabled()
        )
        # 按当前模式获取非专家 / 专家参数视图。
        if force_layer_local_gradients:
            if full_bucket_logical_cuda:
                # full_bucket logical CUDA 直接读 device 参数视图。
                non_routed_param = parameter_store.parameter_view(
                    non_routed_shard,
                    step_index=step_index,
                    device=self._compute_device,
                )
                routed_param = parameter_store.parameter_view(
                    routed_shard,
                    step_index=step_index,
                    device=self._compute_device,
                )
            else:
                # 其余 logical 路径取可写 CPU 参数张量。
                non_routed_param = parameter_store.mutable_parameter(
                    non_routed_shard,
                    step_index=step_index,
                )
                routed_param = parameter_store.mutable_parameter(
                    routed_shard,
                    step_index=step_index,
                )
        else:
            # 非 logical 路径统一读普通参数视图。
            non_routed_param = parameter_store.parameter_view(
                non_routed_shard,
                step_index=step_index,
                device=self._compute_device,
            )
            routed_param = parameter_store.parameter_view(
                routed_shard,
                step_index=step_index,
                device=self._compute_device,
            )
        # 预留 packed 量化视图引用。
        non_routed_packed: PackedQuantizedTensor | None = None
        routed_packed: PackedQuantizedTensor | None = None
        # 量化执行路径下，尝试从参数存储获取量化视图并绑定。
        if self._quantized_execution:
            non_routed_packed = parameter_store.quantized_parameter_view(
                non_routed_shard,
                step_index=step_index,
                device=self._compute_device,
            )
            routed_packed = parameter_store.quantized_parameter_view(
                routed_shard,
                step_index=step_index,
                device=self._compute_device,
            )
            if not force_layer_local_gradients or full_bucket_logical_cuda:
                # 仅在整块参数视图场景下绑定 flat packed view。
                self._bind_flat_packed_view(
                    non_routed_param,
                    non_routed_packed,
                )
                self._bind_flat_packed_view(
                    routed_param,
                    routed_packed,
                )
        # 整块参数视图路径下，需要对参数张量打开 autograd。
        if not force_layer_local_gradients or full_bucket_logical_cuda:
            non_routed_param = non_routed_param.requires_grad_(True)
            routed_param = routed_param.requires_grad_(True)

        # -----------------
        # 决定当前 bucket 是否可走语义布局路径，并据此规划图形状。
        layer_count = max(1, len(bucket.attention_types))
        non_routed_layout = parameter_store.source_layout(non_routed_shard)
        routed_layout = parameter_store.source_layout(routed_shard)
        # 只有非 logical 路径且所有层都有足够布局预算时，才启用语义布局执行。
        semantic_layout_available = not force_layer_local_gradients and all(
            self._layout_budget_for_layer(non_routed_layout, layer_index=layer_index) > 0
            and self._layout_budget_for_layer(routed_layout, layer_index=layer_index) > 0
            for layer_index in bucket.layer_indices
        )
        if semantic_layout_available:
            # 语义布局路径按所有层的最小可用预算来确定图规模。
            non_routed_budget = min(
                self._layout_budget_for_layer(non_routed_layout, layer_index=layer_index)
                for layer_index in bucket.layer_indices
            )
            routed_budget = min(
                self._layout_budget_for_layer(routed_layout, layer_index=layer_index)
                for layer_index in bucket.layer_indices
            )
        else:
            # 无语义布局时，按整块参数平均切给每层。
            non_routed_budget = max(1, non_routed_param.numel() // layer_count)
            routed_budget = max(1, routed_param.numel() // layer_count)
        # 根据预算和 active expert 数规划当前 bucket 的代表性图形状。
        graph_shape = self._plan_graph_shape(
            non_routed_params=non_routed_budget,
            routed_params=routed_budget,
            active_expert_count=len(routed_shard.expert_ids),
            batch=batch,
        )
        # 构造当前 bucket 的初始 hidden states。
        current_hidden = self._build_hidden_states(
            step_index=step_index,
            bucket=bucket,
            batch=batch,
            hidden_dim=graph_shape.hidden_dim,
            sequence_length=graph_shape.sequence_length,
            device=self._compute_device,
        )
        # 保存初始 hidden，后面构建监督目标时会回用。
        initial_hidden = current_hidden
        layer_losses = []
        layer_peak_bytes = [
            int(current_hidden.numel() * current_hidden.element_size())
        ]
        gradients: list[GradientPayload] = []
        covered_roles: set[str] = set()
        # -----------------
        # 逐层执行当前 bucket，并收集 loss / peak bytes / gradients。
        for layer_index, attention_type in enumerate(bucket.attention_types):
            if semantic_layout_available:
                # 语义布局路径按绝对层号提取当前层的语义条目。
                absolute_layer_index = bucket.layer_indices[layer_index]
                layer_non_routed_entries = self._semantic_entries_for_layer(
                    non_routed_layout,
                    layer_index=absolute_layer_index,
                )
                layer_routed_entries = self._semantic_entries_for_layer(
                    routed_layout,
                    layer_index=absolute_layer_index,
                )
                # 记录当前层实际覆盖到的语义角色。
                covered_roles.update(
                    self._covered_roles(
                        layer_non_routed_entries,
                        layer_routed_entries,
                    )
                )
                # 执行当前层的语义布局路径。
                current_hidden, layer_loss, peak_bytes = self._execute_semantic_layer(
                    hidden_states=current_hidden,
                    non_routed_param=non_routed_param,
                    routed_param=routed_param,
                    graph_shape=graph_shape,
                    attention_type=attention_type,
                    layer_index=layer_index,
                    bucket_id=bucket.bucket_id,
                    expert_ids=routed_shard.expert_ids,
                    non_routed_entries=layer_non_routed_entries,
                    routed_entries=layer_routed_entries,
                )
            else:
                # 非语义布局路径先估算当前层所需的非专家 / 专家参数大小。
                layer_non_routed_needed, layer_routed_needed = (
                    self._required_layer_param_sizes(graph_shape)
                )
                # 计算当前层在整块参数中的起始偏移。
                non_routed_start, _ = self._layer_param_span(
                    non_routed_param,
                    layer_index=layer_index,
                    layer_count=layer_count,
                )
                routed_start, _ = self._layer_param_span(
                    routed_param,
                    layer_index=layer_index,
                    layer_count=layer_count,
                )
                # 切出当前层对应的参数视图。
                layer_non_routed = self._layer_param_slice(
                    non_routed_param,
                    layer_index,
                    layer_count,
                )
                layer_routed = self._layer_param_slice(
                    routed_param,
                    layer_index,
                    layer_count,
                )
                # logical 局部梯度路径下，还要把整层参数压缩成实际所需的紧凑视图。
                if force_layer_local_gradients and not full_bucket_logical_cuda:
                    layer_non_routed = self._compact_layer_parameter(
                        layer_non_routed,
                        start_offset=non_routed_start,
                        compact_numel=layer_non_routed_needed,
                        packed=non_routed_packed,
                    )
                    layer_routed = self._compact_layer_parameter(
                        layer_routed,
                        start_offset=routed_start,
                        compact_numel=layer_routed_needed,
                        packed=routed_packed,
                    )
                # 执行当前层的普通代表性路径。
                current_hidden, layer_loss, peak_bytes = self._execute_layer(
                    hidden_states=current_hidden,
                    non_routed_param=layer_non_routed,
                    routed_param=layer_routed,
                    graph_shape=graph_shape,
                    attention_type=attention_type,
                    layer_index=layer_index,
                    bucket_id=bucket.bucket_id,
                    expert_ids=routed_shard.expert_ids,
                )
                if force_layer_local_gradients:
                    # logical 路径下，为每层单独构造可监督的有效 loss。
                    effective_layer_loss = layer_loss
                    if layer_index == len(bucket.attention_types) - 1:
                        # 最后一层额外加一项轻量监督目标，稳定局部梯度。
                        supervision_target = self._build_target_signal(
                            step_index=step_index,
                            bucket=bucket,
                            batch=batch,
                            hidden_dim=graph_shape.hidden_dim,
                            sequence_length=graph_shape.sequence_length,
                            fallback=initial_hidden,
                            device=self._compute_device,
                        )
                        effective_layer_loss = effective_layer_loss + 0.01 * F.mse_loss(
                            current_hidden,
                            supervision_target,
                        )
                    # 只对当前层紧凑参数视图求梯度。
                    layer_non_routed_grad, layer_routed_grad = torch.autograd.grad(
                        effective_layer_loss,
                        (layer_non_routed, layer_routed),
                    )
                    # 把当前层局部梯度封装成 GradientPayload。
                    gradients.extend(
                        (
                            GradientPayload(
                                group_id=non_routed_shard.group_id,
                                logical_params=layer_non_routed_grad.numel(),
                                gradient=self._stabilize_tensor(
                                    layer_non_routed_grad.detach(),
                                    max_abs=1e3,
                                ),
                                start_offset=non_routed_start,
                            ),
                            GradientPayload(
                                group_id=routed_shard.group_id,
                                logical_params=layer_routed_grad.numel(),
                                gradient=self._stabilize_tensor(
                                    layer_routed_grad.detach(),
                                    max_abs=1e3,
                                ),
                                start_offset=routed_start,
                            ),
                        )
                    )
                    # 断开当前 hidden，避免跨层累积计算图。
                    current_hidden = current_hidden.detach()
                    layer_loss = effective_layer_loss.detach()
            # 记录当前层的 loss 和激活峰值。
            layer_losses.append(layer_loss)
            layer_peak_bytes.append(peak_bytes)
        # 聚合当前 bucket 的平均层 loss。
        loss = torch.stack(layer_losses).mean()
        if force_layer_local_gradients:
            # logical 局部梯度路径下，按 group 聚合梯度范数。
            group_norm_squares = {
                non_routed_shard.group_id: 0.0,
                routed_shard.group_id: 0.0,
            }
            for payload in gradients:
                norm = float(torch.linalg.vector_norm(payload.gradient).item())
                group_norm_squares[payload.group_id] = (
                    group_norm_squares.get(payload.group_id, 0.0) + norm * norm
                )
            non_routed_norm = math.sqrt(
                group_norm_squares.get(non_routed_shard.group_id, 0.0)
            )
            routed_norm = math.sqrt(
                group_norm_squares.get(routed_shard.group_id, 0.0)
            )
        else:
            # 非 logical 路径下，直接对整块参数做一次总 loss 反向。
            supervision_target = self._build_target_signal(
                step_index=step_index,
                bucket=bucket,
                batch=batch,
                hidden_dim=graph_shape.hidden_dim,
                sequence_length=graph_shape.sequence_length,
                fallback=initial_hidden,
                device=self._compute_device,
            )
            loss = loss + 0.01 * F.mse_loss(current_hidden, supervision_target)
            # 对整块非专家 / 专家参数求梯度。
            non_routed_grad, routed_grad = torch.autograd.grad(
                loss,
                (non_routed_param, routed_param),
            )
            # 对梯度做数值稳定化。
            non_routed_grad = self._stabilize_tensor(
                non_routed_grad.detach(),
                max_abs=1e3,
            )
            routed_grad = self._stabilize_tensor(
                routed_grad.detach(),
                max_abs=1e3,
            )
            # 计算整块非专家 / 专家参数的梯度范数。
            non_routed_norm = float(torch.linalg.vector_norm(non_routed_grad).item())
            routed_norm = float(torch.linalg.vector_norm(routed_grad).item())
            # 封装成两条整块 GradientPayload。
            gradients = [
                GradientPayload(
                    group_id=non_routed_shard.group_id,
                    logical_params=non_routed_shard.logical_params,
                    gradient=non_routed_grad,
                ),
                GradientPayload(
                    group_id=routed_shard.group_id,
                    logical_params=routed_shard.logical_params,
                    gradient=routed_grad,
                ),
            ]
        # 取各层峰值中的最大值作为当前 bucket 激活峰值。
        bucket_peak_bytes = max(layer_peak_bytes)
        # 将最终 loss 裁掉 NaN/Inf 后转成 Python float。
        loss_value = float(
            torch.nan_to_num(
                loss.detach(),
                nan=0.0,
                posinf=1e3,
                neginf=-1e3,
            ).item()
        )
        # 返回当前 bucket 的梯度 payload 和聚合执行记录。
        return RepresentativeBucketExecutionResult(
            gradients=tuple(gradients),
            bucket_record=RepresentativeBucketRecord(
                bucket_id=bucket.bucket_id,
                attention_types=bucket.attention_types,
                contains_full_attention=bucket.contains_full_attention,
                active_expert_ids=routed_shard.expert_ids,
                semantic_layout_used=semantic_layout_available,
                semantic_roles=tuple(sorted(covered_roles)),
                execution_mode=(
                    "structured_qwen35_bucket"
                    if semantic_layout_available
                    else "synthetic_representative_bucket"
                ),
                loss_value=loss_value,
                non_routed_gradient_l2_norm=non_routed_norm,
                expert_gradient_l2_norm=routed_norm,
                peak_activation_bytes=bucket_peak_bytes,
            ),
        )

    def execute_step(
        self,
        *,
        step_index: int,
        batch: BatchShape,
        layer_buckets: tuple[LayerBucketPlan, ...],
        parameter_shards: tuple[ParameterShardSnapshot, ...],
        parameter_store: ParameterShardStore,
    ) -> RepresentativeExecutionResult:
        # -----------------
        # 初始化 step 级聚合指标容器。
        # 用列表累积每个 bucket 的执行记录。
        bucket_records: list[RepresentativeBucketRecord] = []
        # total_loss 汇总所有 bucket 的 loss。
        total_loss = 0.0
        # max_gradient_norm 记录当前 step 的最大梯度范数。
        max_gradient_norm = 0.0
        # peak_activation_bytes 记录当前 step 的最大激活峰值。
        peak_activation_bytes = 0
        # gradient_payload_count 统计本 step 共生成多少条梯度 payload。
        gradient_payload_count = 0

        # 逐个 bucket 执行，并聚合 step 级指标。
        for bucket in layer_buckets:
            # 执行单个 bucket 的代表性前向/反向。
            bucket_result = self.execute_bucket(
                step_index=step_index,
                batch=batch,
                bucket=bucket,
                parameter_shards=parameter_shards,
                parameter_store=parameter_store,
            )
            # 累加梯度 payload 数。
            gradient_payload_count += len(bucket_result.gradients)
            # 记录 bucket 执行结果。
            bucket_records.append(bucket_result.bucket_record)
            # 累加总 loss。
            total_loss += bucket_result.bucket_record.loss_value
            # 更新 step 级最大梯度范数。
            max_gradient_norm = max(
                max_gradient_norm,
                bucket_result.bucket_record.non_routed_gradient_l2_norm,
                bucket_result.bucket_record.expert_gradient_l2_norm,
            )
            # 更新 step 级激活峰值。
            peak_activation_bytes = max(
                peak_activation_bytes,
                bucket_result.bucket_record.peak_activation_bytes,
            )

        # -----------------
        # 组装 step 级执行摘要并返回。
        # 返回 step 级聚合执行摘要。
        return RepresentativeExecutionResult(
            gradients=(),
            execution_summary=RepresentativeExecutionSummary(
                executed_buckets=len(bucket_records),
                gradient_shards=gradient_payload_count,
                total_loss=total_loss,
                max_gradient_l2_norm=max_gradient_norm,
                peak_activation_bytes=peak_activation_bytes,
                peak_host_gradient_buffer_bytes=0,
                gradient_buffer_storage_dtype=(
                    self.config.optimizer.gradient_buffer_storage_dtype
                ),
                bucket_records=tuple(bucket_records),
            ),
        )
