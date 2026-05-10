"""激活检查点策略——按 gradient bucket 容量划分 segment，控制中间激活保留/重算（设计文档 Section 11）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from cfie_training.training_base.window_plan import TrainableParamSpec


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
class CheckpointSegment:
    layer_ids: tuple[int, ...]
    grad_bytes: int
    activation_peak_bytes: int
    total_peak_bytes: int
    trainable_param_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.layer_ids:
            raise ValueError("layer_ids must not be empty")
        _require_non_negative_int("grad_bytes", self.grad_bytes)
        _require_non_negative_int("activation_peak_bytes", self.activation_peak_bytes)
        _require_non_negative_int("total_peak_bytes", self.total_peak_bytes)
        _require_non_negative_int("total_peak_bytes", self.total_peak_bytes)


@dataclass(frozen=True, slots=True)
class SegmentGradEstimator:
    grad_dtype_bytes: int = 2

    def __post_init__(self) -> None:
        _require_positive_int("grad_dtype_bytes", self.grad_dtype_bytes)

    def estimate_grad_bytes(
        self,
        params: Iterable[TrainableParamSpec],
    ) -> int:
        total = 0
        for spec in params:
            total += spec.fp32_bytes
        return total


@dataclass(frozen=True, slots=True)
class ActivationPeakEstimator:
    max_tokens: int
    hidden_size: int
    num_attention_heads: int = 0
    moe_topk: int = 8
    activation_dtype_bytes: int = 2

    def __post_init__(self) -> None:
        _require_positive_int("max_tokens", self.max_tokens)
        _require_positive_int("hidden_size", self.hidden_size)
        _require_non_negative_int("num_attention_heads", self.num_attention_heads)
        _require_positive_int("moe_topk", self.moe_topk)
        _require_positive_int("activation_dtype_bytes", self.activation_dtype_bytes)

    def estimate_peak_bytes(self) -> int:
        token_bytes = (
            self.max_tokens * self.hidden_size * self.activation_dtype_bytes
        )
        attention_bytes = (
            self.max_tokens
            * self.hidden_size
            * self.activation_dtype_bytes
            * 3
        )
        moe_bytes = (
            self.max_tokens
            * self.hidden_size
            * self.activation_dtype_bytes
            * self.moe_topk
        )
        return token_bytes + attention_bytes + moe_bytes


@dataclass(slots=True)
class ActivationCheckpointPlanner:
    grad_estimator: SegmentGradEstimator = field(
        default_factory=SegmentGradEstimator
    )
    peak_estimator: ActivationPeakEstimator | None = None

    def plan_segments(
        self,
        param_specs: Iterable[TrainableParamSpec],
        *,
        bucket_size_bytes: int,
        num_buckets: int,
        vram_budget_bytes: int,
        static_vram_bytes: int = 0,
    ) -> tuple[CheckpointSegment, ...]:
        _require_positive_int("bucket_size_bytes", bucket_size_bytes)
        _require_positive_int("num_buckets", num_buckets)
        _require_positive_int("vram_budget_bytes", vram_budget_bytes)
        _require_non_negative_int("static_vram_bytes", static_vram_bytes)

        specs = tuple(param_specs)
        if not specs:
            return ()

        if self.peak_estimator is None:
            act_peak = 0
        else:
            act_peak = self.peak_estimator.estimate_peak_bytes()

        min_segment_bytes = int(bucket_size_bytes * 0.8)
        max_segment_bytes = (num_buckets - 1) * bucket_size_bytes

        segments: list[CheckpointSegment] = []
        current_specs: list[TrainableParamSpec] = []
        current_grad_bytes = 0
        current_layer_ids: list[int] = []

        for spec in specs:
            spec_grad = self.grad_estimator.estimate_grad_bytes([spec])

            if spec_grad > max_segment_bytes:
                sub_specs = self._split_large_param(spec, bucket_size_bytes)
                for sub in sub_specs:
                    sub_grad = self.grad_estimator.estimate_grad_bytes([sub])
                    if (
                        current_grad_bytes + sub_grad > max_segment_bytes
                        and current_specs
                    ):
                        segments.append(
                            self._make_segment(
                                current_specs,
                                current_layer_ids,
                                current_grad_bytes,
                                act_peak,
                                static_vram_bytes,
                                vram_budget_bytes,
                            )
                        )
                        current_specs = []
                        current_layer_ids = []
                        current_grad_bytes = 0
                    current_specs.append(sub)
                    current_grad_bytes += sub_grad

            elif (
                current_grad_bytes + spec_grad > max_segment_bytes
                and current_specs
                and current_grad_bytes >= min_segment_bytes
            ):
                segments.append(
                    self._make_segment(
                        current_specs,
                        current_layer_ids,
                        current_grad_bytes,
                        act_peak,
                        static_vram_bytes,
                        vram_budget_bytes,
                    )
                )
                current_specs = [spec]
                current_layer_ids = [self._layer_id_from_spec(spec)]
                current_grad_bytes = spec_grad
            else:
                current_specs.append(spec)
                layer_id = self._layer_id_from_spec(spec)
                if layer_id not in current_layer_ids:
                    current_layer_ids.append(layer_id)
                current_grad_bytes += spec_grad

        if current_specs:
            segments.append(
                self._make_segment(
                    current_specs,
                    current_layer_ids,
                    current_grad_bytes,
                    act_peak,
                    static_vram_bytes,
                    vram_budget_bytes,
                )
            )

        return tuple(segments)

    def estimate_total_workspace_bytes(
        self,
        segments: Iterable[CheckpointSegment],
    ) -> int:
        return max(
            (seg.activation_peak_bytes for seg in segments),
            default=0,
        )

    @staticmethod
    def _split_large_param(
        spec: TrainableParamSpec,
        bucket_size_bytes: int,
    ) -> tuple[TrainableParamSpec, ...]:
        grad_bytes_total = spec.fp32_bytes
        num_chunks = max(2, (grad_bytes_total + bucket_size_bytes - 1) // bucket_size_bytes)
        chunk_bytes = (grad_bytes_total + num_chunks - 1) // num_chunks
        layer_id = ActivationCheckpointPlanner._layer_id_from_spec(spec)
        parts: list[TrainableParamSpec] = []
        for i in range(num_chunks):
            part_bytes = min(chunk_bytes, grad_bytes_total - i * chunk_bytes)
            if part_bytes <= 0:
                break
            parts.append(
                TrainableParamSpec(
                    param_id=f"layers.{layer_id}.experts.0.chunk{i}",
                    kind=spec.kind,
                    fp32_bytes=part_bytes,
                    gpu_shadow_bytes=spec.gpu_shadow_bytes // num_chunks,
                    adam_bytes=spec.adam_bytes // num_chunks,
                    priority=spec.priority,
                )
            )
        return tuple(parts)

    @staticmethod
    def _layer_id_from_spec(spec: TrainableParamSpec) -> int:
        parts = spec.param_id.split(".")
        if len(parts) >= 2 and parts[0] == "layers":
            try:
                return int(parts[1])
            except ValueError:
                pass
        return 0

    @staticmethod
    def _make_segment(
        current_specs: list[TrainableParamSpec],
        layer_ids: list[int],
        grad_bytes: int,
        act_peak: int,
        static_vram_bytes: int,
        vram_budget_bytes: int,
    ) -> CheckpointSegment:
        total_peak = static_vram_bytes + grad_bytes + act_peak
        if total_peak > vram_budget_bytes * 0.95 and vram_budget_bytes > 0:
            raise ValueError(
                f"Segment total peak {total_peak} exceeds "
                f"95% of vram budget {vram_budget_bytes}"
            )
        if not layer_ids:
            layer_ids = [0]
        return CheckpointSegment(
            layer_ids=tuple(layer_ids),
            grad_bytes=grad_bytes,
            activation_peak_bytes=act_peak,
            total_peak_bytes=total_peak,
            trainable_param_ids=tuple(
                spec.param_id for spec in current_specs
            ),
        )


@dataclass(slots=True)
class ActivationCheckpointPolicy:
    """训练前向中的激活检查点策略 context manager。

    由 ActivationCheckpointPlanner.plan_segments() 生成的 segments 驱动，
    控制前向过程中哪些层/segment 的中间激活需要保留、哪些可以释放后重算。

    Usage:
        planner = ActivationCheckpointPlanner(...)
        segments = planner.plan_segments(params, ...)
        with ActivationCheckpointPolicy(segments) as policy:
            for layer_id in range(num_layers):
                seg_idx = policy.segment_for_layer(layer_id)
                if policy.is_first_layer_in_segment(layer_id):
                    policy.save_boundary_input(seg_idx, hidden_states)
                hidden_states = model.forward_layer(layer_id, hidden_states)
    """

    segments: tuple[CheckpointSegment, ...] = ()
    _boundary_inputs: dict[int, Any] = field(default_factory=dict)
    _layer_to_segment: dict[int, int] = field(default_factory=dict)
    _first_layers: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._build_index()

    def _build_index(self) -> None:
        self._layer_to_segment.clear()
        self._first_layers.clear()
        for seg_idx, seg in enumerate(self.segments):
            if seg.layer_ids:
                self._first_layers.add(seg.layer_ids[0])
            for layer_id in seg.layer_ids:
                self._layer_to_segment[layer_id] = seg_idx

    def __enter__(self) -> "ActivationCheckpointPolicy":
        self._boundary_inputs.clear()
        return self

    def __exit__(self, *args: Any) -> bool:
        self._boundary_inputs.clear()
        return False

    def segment_for_layer(self, layer_id: int) -> int:
        return self._layer_to_segment.get(layer_id, -1)

    def is_first_layer_in_segment(self, layer_id: int) -> bool:
        return layer_id in self._first_layers

    def save_boundary_input(
        self, segment_idx: int, hidden_state: Any
    ) -> None:
        """保存 segment 边界的 hidden state，用于反向时重算该 segment。"""
        self._boundary_inputs[segment_idx] = (
            hidden_state.detach().clone()
            if hasattr(hidden_state, "detach")
            else hidden_state
        )

    def get_boundary_input(self, segment_idx: int) -> Any:
        """获取之前保存的 segment 输入。"""
        if segment_idx not in self._boundary_inputs:
            raise KeyError(
                f"segment {segment_idx} 没有保存的边界输入"
            )
        return self._boundary_inputs[segment_idx]

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    @property
    def saved_segment_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._boundary_inputs))
