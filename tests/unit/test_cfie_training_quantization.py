"""Unit tests for training-time GPTQ quantization and FP32 mirror lifecycle."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from cfie_training.profiles import build_profile_config
from cfie_training.runtime import quantization as quantization_module
from cfie_training.runtime.executor import (
    RepresentativeBucketExecutor,
    _PackedWeightSliceBinding,
    _PackedWeightSliceSpec,
)
from cfie_training.runtime.quantization import (
    GPTQTrainingQuantizer,
    dequantize_packed_range,
    quantized_linear,
    quantized_linear_from_packed_slice,
)
from cfie_training.runtime.store import ParameterShardStore
from cfie_training.runtime.types import BatchShape, ParameterShardSnapshot


def _bucket_non_routed_shard() -> ParameterShardSnapshot:
    return ParameterShardSnapshot(
        group_id="bucket_non_routed:0",
        component="bucket_non_routed",
        residency_state="nvme_cold",
        committed_version=0,
        pending_version=None,
        logical_params=1024,
        bucket_id=0,
        last_touched_step=-1,
    )


def test_gptq_training_quantizer_roundtrips_vector() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    quantizer = GPTQTrainingQuantizer(cfg)
    source = torch.linspace(-1.0, 1.0, steps=128, dtype=torch.float32)

    packed = quantizer.quantize(source)
    restored = quantizer.dequantize(
        packed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert packed.original_numel == 128
    assert packed.group_count == 1
    assert packed.resident_bytes > 0
    assert restored.shape == source.shape
    assert torch.max(torch.abs(restored - source)).item() <= 0.15


def test_gptq_training_quantizer_roundtrips_vector_across_multiple_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quantization_module, "_QUANTIZE_GROUP_CHUNK_SIZE", 2)
    cfg = build_profile_config("qwen35-35b-a3b")
    quantizer = GPTQTrainingQuantizer(cfg)
    source = torch.linspace(-2.0, 2.0, steps=1024, dtype=torch.float32)

    packed = quantizer.quantize(source)
    restored = quantizer.dequantize(
        packed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert packed.original_numel == 1024
    assert packed.group_count == 8
    assert restored.shape == source.shape
    assert torch.max(torch.abs(restored - source)).item() <= 0.2


def test_quantized_linear_matches_packed_forward_and_backward() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    quantizer = GPTQTrainingQuantizer(cfg)
    inputs = torch.tensor(
        [
            [0.5, -0.75, 1.25],
            [1.0, 0.25, -0.5],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    weight = torch.tensor(
        [
            [0.25, -0.75],
            [1.0, 0.5],
            [-0.5, 0.25],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    output = quantized_linear(
        inputs,
        weight,
        bits=cfg.runtime_quantization.bits,
        group_size=cfg.runtime_quantization.group_size,
        sym=cfg.runtime_quantization.sym,
        pack_dtype_name=cfg.runtime_quantization.pack_dtype,
        compute_view_dtype_name=cfg.runtime_quantization.compute_view_dtype,
    )
    output.sum().backward()

    packed = quantizer.quantize(weight.detach())
    restored = quantizer.dequantize(
        packed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).view_as(weight)
    expected_output = inputs.detach() @ restored
    expected_grad_output = torch.ones_like(output)
    expected_grad_input = expected_grad_output @ restored.transpose(0, 1)
    expected_grad_weight = inputs.detach().transpose(0, 1) @ expected_grad_output

    assert torch.allclose(output.detach(), expected_output, atol=1e-5)
    assert inputs.grad is not None
    assert weight.grad is not None
    assert torch.allclose(inputs.grad, expected_grad_input, atol=1e-5)
    assert torch.allclose(weight.grad, expected_grad_weight, atol=1e-5)


def test_quantized_linear_from_packed_slice_matches_expected_forward_and_backward() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    quantizer = GPTQTrainingQuantizer(cfg)
    flat_source = torch.linspace(-1.0, 1.0, steps=64, dtype=torch.float32)
    packed = quantizer.quantize(flat_source)
    start_offset = 8
    raw_rows = 3
    raw_cols = 4
    weight = (
        0.05
        * torch.tanh(
            flat_source[start_offset : start_offset + raw_rows * raw_cols]
            .view(raw_rows, raw_cols)
            .transpose(0, 1)
        )
    ).detach().requires_grad_(True)
    inputs = torch.tensor(
        [
            [0.25, -0.5, 0.75, 1.0],
            [1.0, 0.5, -0.25, 0.125],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    output = quantized_linear_from_packed_slice(
        inputs,
        weight,
        packed=packed,
        start_offset=start_offset,
        raw_rows=raw_rows,
        raw_cols=raw_cols,
        transpose_last_two=True,
        tanh_scale=0.05,
        compute_view_dtype_name=cfg.runtime_quantization.compute_view_dtype,
    )
    output.sum().backward()

    restored = quantizer.dequantize(
        packed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected_weight = 0.05 * torch.tanh(
        restored[start_offset : start_offset + raw_rows * raw_cols]
        .view(raw_rows, raw_cols)
        .transpose(0, 1)
    )
    expected_output = inputs.detach() @ expected_weight
    expected_grad_input = torch.ones_like(output) @ expected_weight.transpose(0, 1)
    expected_grad_weight = inputs.detach().transpose(0, 1) @ torch.ones_like(output)

    assert torch.allclose(output.detach(), expected_output, atol=1e-5)
    assert inputs.grad is not None
    assert weight.grad is not None
    assert torch.allclose(inputs.grad, expected_grad_input, atol=1e-5)
    assert torch.allclose(weight.grad, expected_grad_weight, atol=1e-5)


def test_dequantize_packed_range_reads_only_requested_slice() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    quantizer = GPTQTrainingQuantizer(cfg)
    source = torch.linspace(-2.0, 2.0, steps=512, dtype=torch.float32)
    packed = quantizer.quantize(source)
    expected = quantizer.dequantize(
        packed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )[120:184]

    with patch.object(
        quantization_module,
        "_dequantize_values",
        side_effect=AssertionError("range path should not materialize full tensor"),
    ):
        restored = dequantize_packed_range(
            packed,
            start_offset=120,
            length=64,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    assert torch.allclose(restored, expected, atol=1e-5)


def test_executor_materializes_resize_aware_bound_weight_from_packed_slice() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    executor = RepresentativeBucketExecutor(cfg)
    quantizer = GPTQTrainingQuantizer(cfg)
    flat_source = torch.linspace(-1.0, 1.0, steps=64, dtype=torch.float32)
    packed = quantizer.quantize(flat_source)
    binding = _PackedWeightSliceBinding(
        slices=(
            _PackedWeightSliceSpec(
                packed=packed,
                start_offset=8,
                raw_shape=(1, 12),
            ),
        ),
        resize_shape=(3, 4),
        resize_mode="bilinear",
        transpose_last_two=True,
        tanh_scale=0.05,
    )
    weight = torch.zeros(4, 3, dtype=torch.float32, requires_grad=True)
    inputs = torch.tensor(
        [
            [0.25, -0.5, 0.75, 1.0],
            [1.0, 0.5, -0.25, 0.125],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    effective = executor._materialize_bound_weight(weight=weight, binding=binding)
    assert effective is not None
    output = inputs @ effective
    output.sum().backward()

    restored = quantizer.dequantize(
        packed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected = restored[8:20].view(1, 12)
    expected = F.interpolate(
        expected.unsqueeze(0).unsqueeze(0),
        size=(3, 4),
        mode="bilinear",
        align_corners=False,
    ).view(3, 4)
    expected = 0.05 * torch.tanh(expected.transpose(0, 1))

    assert torch.allclose(effective.detach(), expected, atol=1e-5)


def test_executor_uses_actual_model_dimensions_in_full_bucket_logical_cuda_mode() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for full-bucket logical CUDA mode")

    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.execution.compute_device = "gpu"
    cfg.execution.trainable_shard_materialization = "logical"
    cfg.execution.logical_cuda_execution_mode = "full_bucket"
    cfg.validate()
    executor = RepresentativeBucketExecutor(cfg)

    graph_shape = executor._plan_graph_shape(
        non_routed_params=10**9,
        routed_params=10**9,
        active_expert_count=cfg.expert_rotation.active_experts_per_step,
        batch=BatchShape(samples=1, tokens_per_sample=32),
    )

    assert graph_shape.hidden_dim == cfg.model_spec.hidden_size
    assert graph_shape.expert_hidden_dim == cfg.model_spec.moe_intermediate_size
    assert graph_shape.attention_head_count == cfg.model_spec.num_attention_heads
    assert graph_shape.kv_head_count == cfg.model_spec.num_key_value_heads
    assert graph_shape.topk == cfg.model_spec.num_experts_per_tok


def test_executor_linear_projection_uses_direct_packed_slice_path() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    executor = RepresentativeBucketExecutor(cfg)
    executor._quantized_execution = True
    quantizer = GPTQTrainingQuantizer(cfg)
    flat_source = torch.linspace(-1.0, 1.0, steps=64, dtype=torch.float32)
    packed = quantizer.quantize(flat_source)
    weight = executor._bind_weight_slice(
        torch.zeros(4, 3, dtype=torch.float32, requires_grad=True),
        binding=_PackedWeightSliceBinding(
            slices=(
                _PackedWeightSliceSpec(
                    packed=packed,
                    start_offset=8,
                    raw_shape=(3, 4),
                ),
            ),
            transpose_last_two=True,
            tanh_scale=0.05,
        ),
    )
    inputs = torch.tensor(
        [
            [0.25, -0.5, 0.75, 1.0],
            [1.0, 0.5, -0.25, 0.125],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    def _unexpected_materialize(
        self,
        *,
        weight: torch.Tensor,
        binding: _PackedWeightSliceBinding,
    ) -> torch.Tensor | None:
        raise AssertionError("direct packed slice path should bypass materialization")

    with patch.object(
        RepresentativeBucketExecutor,
        "_materialize_bound_weight",
        _unexpected_materialize,
    ):
        output = executor._linear_projection(inputs, weight)
        output.sum().backward()

    restored = quantizer.dequantize(
        packed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected_weight = 0.05 * torch.tanh(
        restored[8:20].view(3, 4).transpose(0, 1)
    )
    expected_output = inputs.detach() @ expected_weight
    expected_grad_input = torch.ones_like(output) @ expected_weight.transpose(0, 1)
    expected_grad_weight = inputs.detach().transpose(0, 1) @ torch.ones_like(output)

    assert torch.allclose(output.detach(), expected_output, atol=1e-5)
    assert inputs.grad is not None
    assert weight.grad is not None
    assert torch.allclose(inputs.grad, expected_grad_input, atol=1e-5)
    assert torch.allclose(weight.grad, expected_grad_weight, atol=1e-5)


def test_executor_resolves_resize_aware_bound_scale_vector_from_packed_slice() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    executor = RepresentativeBucketExecutor(cfg)
    quantizer = GPTQTrainingQuantizer(cfg)
    flat_source = torch.linspace(-1.0, 1.0, steps=64, dtype=torch.float32)
    packed = quantizer.quantize(flat_source)
    source = executor._bind_weight_slice(
        torch.zeros(12, dtype=torch.float32, requires_grad=True),
        binding=_PackedWeightSliceBinding(
            slices=(
                _PackedWeightSliceSpec(
                    packed=packed,
                    start_offset=8,
                    raw_shape=(12,),
                ),
            ),
        ),
    )

    resized = executor._resize_vector(source, size=6)
    transformed = executor._add_scalar_bound(
        executor._tanh_scale_bound(resized, scale=0.05),
        value=1.0,
    )
    binding = executor._weight_binding_for_tensor(transformed)
    assert binding is not None
    resolved = executor._materialize_bound_weight(
        weight=transformed,
        binding=binding,
    )
    assert resolved is not None
    resolved.sum().backward()

    restored = quantizer.dequantize(
        packed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected = F.interpolate(
        restored[8:20].view(1, 1, -1),
        size=6,
        mode="linear",
        align_corners=False,
    ).view(6)
    expected = 1.0 + 0.05 * torch.tanh(expected)

    assert torch.allclose(resolved.detach(), expected, atol=1e-5)
    assert source.grad is not None


def test_executor_reuses_packed_binding_for_trilinear_expert_resize() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    executor = RepresentativeBucketExecutor(cfg)
    quantizer = GPTQTrainingQuantizer(cfg)
    flat_source = torch.linspace(-1.0, 1.0, steps=64, dtype=torch.float32)
    packed = quantizer.quantize(flat_source)
    source = executor._bind_weight_slice(
        torch.zeros(1, 2, 6, dtype=torch.float32),
        binding=_PackedWeightSliceBinding(
            slices=(
                _PackedWeightSliceSpec(
                    packed=packed,
                    start_offset=8,
                    raw_shape=(1, 2, 6),
                ),
            ),
        ),
    )
    resized = executor._resize_expert_tensor(source, size=(2, 3, 4))
    transformed = executor._tanh_scale_bound(
        executor._transpose_last_two_bound(resized),
        scale=0.05,
    )
    expert_binding = executor._expert_binding_for_index(transformed, 1)
    assert expert_binding is not None
    weight = executor._bind_weight_slice(
        torch.zeros(4, 3, dtype=torch.float32, requires_grad=True),
        binding=expert_binding,
    )
    inputs = torch.tensor(
        [
            [0.25, -0.5, 0.75, 1.0],
            [1.0, 0.5, -0.25, 0.125],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    effective = executor._materialize_bound_weight(
        weight=weight,
        binding=expert_binding,
    )
    assert effective is not None
    output = inputs @ effective
    output.sum().backward()

    restored = quantizer.dequantize(
        packed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected_weight = restored[8:20].view(1, 2, 6)
    expected_weight = F.interpolate(
        expected_weight.unsqueeze(0).unsqueeze(0),
        size=(2, 3, 4),
        mode="trilinear",
        align_corners=False,
    ).view(2, 3, 4)[1]
    expected_weight = 0.05 * torch.tanh(expected_weight.transpose(0, 1))
    expected_output = inputs.detach() @ expected_weight
    expected_grad_input = torch.ones_like(output) @ expected_weight.transpose(0, 1)

    assert torch.allclose(effective.detach(), expected_weight, atol=1e-5)
    assert torch.allclose(output.detach(), expected_output, atol=1e-5)
    assert inputs.grad is not None
    assert weight.grad is not None
    assert torch.allclose(inputs.grad, expected_grad_input, atol=1e-5)
    assert torch.allclose(
        weight.grad,
        inputs.detach().transpose(0, 1) @ torch.ones_like(output),
        atol=1e-5,
    )


def test_executor_linear_projection_uses_direct_packed_slice_for_trilinear_expert_resize() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    executor = RepresentativeBucketExecutor(cfg)
    executor._quantized_execution = True
    quantizer = GPTQTrainingQuantizer(cfg)
    flat_source = torch.linspace(-1.0, 1.0, steps=64, dtype=torch.float32)
    packed = quantizer.quantize(flat_source)
    source = executor._bind_weight_slice(
        torch.zeros(1, 2, 6, dtype=torch.float32),
        binding=_PackedWeightSliceBinding(
            slices=(
                _PackedWeightSliceSpec(
                    packed=packed,
                    start_offset=8,
                    raw_shape=(1, 2, 6),
                ),
            ),
        ),
    )
    resized = executor._resize_expert_tensor(source, size=(2, 3, 4))
    transformed = executor._tanh_scale_bound(
        executor._transpose_last_two_bound(resized),
        scale=0.05,
    )
    expert_binding = executor._expert_binding_for_index(transformed, 1)
    assert expert_binding is not None
    weight = executor._bind_weight_slice(
        torch.zeros(4, 3, dtype=torch.float32, requires_grad=True),
        binding=expert_binding,
    )
    inputs = torch.tensor(
        [
            [0.25, -0.5, 0.75, 1.0],
            [1.0, 0.5, -0.25, 0.125],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    def _unexpected_materialize(
        self,
        *,
        weight: torch.Tensor,
        binding: _PackedWeightSliceBinding,
    ) -> torch.Tensor | None:
        raise AssertionError("trilinear expert path should bypass materialization")

    with patch.object(
        RepresentativeBucketExecutor,
        "_materialize_bound_weight",
        _unexpected_materialize,
    ):
        output = executor._linear_projection(inputs, weight)
        output.sum().backward()

    restored = quantizer.dequantize(
        packed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected_weight = restored[8:20].view(1, 2, 6)
    expected_weight = F.interpolate(
        expected_weight.unsqueeze(0).unsqueeze(0),
        size=(2, 3, 4),
        mode="trilinear",
        align_corners=False,
    ).view(2, 3, 4)[1]
    expected_weight = 0.05 * torch.tanh(expected_weight.transpose(0, 1))
    expected_output = inputs.detach() @ expected_weight
    expected_grad_input = torch.ones_like(output) @ expected_weight.transpose(0, 1)
    expected_grad_weight = inputs.detach().transpose(0, 1) @ torch.ones_like(output)

    assert torch.allclose(output.detach(), expected_output, atol=1e-5)
    assert inputs.grad is not None
    assert weight.grad is not None
    assert torch.allclose(inputs.grad, expected_grad_input, atol=1e-5)
    assert torch.allclose(weight.grad, expected_grad_weight, atol=1e-5)


def test_executor_linear_projection_uses_direct_packed_slice_for_composite_binding() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    executor = RepresentativeBucketExecutor(cfg)
    executor._quantized_execution = True
    quantizer = GPTQTrainingQuantizer(cfg)
    flat_source = torch.linspace(-1.0, 1.0, steps=128, dtype=torch.float32)
    packed = quantizer.quantize(flat_source)
    weight = executor._bind_weight_slice(
        torch.zeros(4, 3, dtype=torch.float32, requires_grad=True),
        binding=_PackedWeightSliceBinding(
            slices=(
                _PackedWeightSliceSpec(
                    packed=packed,
                    start_offset=8,
                    raw_shape=(3, 4),
                ),
                _PackedWeightSliceSpec(
                    packed=packed,
                    start_offset=32,
                    raw_shape=(3, 4),
                ),
            ),
            transpose_last_two=True,
            tanh_scale=0.05,
        ),
    )
    inputs = torch.tensor(
        [
            [0.25, -0.5, 0.75, 1.0],
            [1.0, 0.5, -0.25, 0.125],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    def _unexpected_materialize(
        self,
        *,
        weight: torch.Tensor,
        binding: _PackedWeightSliceBinding,
    ) -> torch.Tensor | None:
        raise AssertionError("composite packed path should bypass materialization")

    with patch.object(
        RepresentativeBucketExecutor,
        "_materialize_bound_weight",
        _unexpected_materialize,
    ):
        output = executor._linear_projection(inputs, weight)
        output.sum().backward()

    restored = quantizer.dequantize(
        packed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    first = 0.05 * torch.tanh(restored[8:20].view(3, 4).transpose(0, 1))
    second = 0.05 * torch.tanh(restored[32:44].view(3, 4).transpose(0, 1))
    expected_weight = torch.stack((first, second), dim=0).mean(dim=0)
    expected_output = inputs.detach() @ expected_weight
    expected_grad_input = torch.ones_like(output) @ expected_weight.transpose(0, 1)
    expected_grad_weight = inputs.detach().transpose(0, 1) @ torch.ones_like(output)

    assert torch.allclose(output.detach(), expected_output, atol=1e-5)
    assert inputs.grad is not None
    assert weight.grad is not None
    assert torch.allclose(inputs.grad, expected_grad_input, atol=1e-5)
    assert torch.allclose(weight.grad, expected_grad_weight, atol=1e-5)


def test_executor_preserves_packed_binding_through_layer_param_slice() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    executor = RepresentativeBucketExecutor(cfg)
    quantizer = GPTQTrainingQuantizer(cfg)
    flat_source = torch.linspace(-1.0, 1.0, steps=64, dtype=torch.float32)
    packed = quantizer.quantize(flat_source)
    full_tensor = torch.zeros(24, dtype=torch.float32)
    executor._bind_flat_packed_view(full_tensor, packed)

    sliced = executor._layer_param_slice(full_tensor, layer_index=1, layer_count=2)
    matrix, _ = executor._take_matrix(sliced, 0, 4, 3)
    binding = executor._weight_binding_for_tensor(matrix)

    assert binding is not None
    effective = executor._materialize_bound_weight(weight=matrix, binding=binding)
    assert effective is not None

    restored = quantizer.dequantize(
        packed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected = restored[12:24].view(4, 3)

    assert torch.allclose(effective.detach(), expected, atol=1e-5)


def test_parameter_store_persists_fp32_master_to_nvme_and_reloads_from_mirror(
    tmp_path: Path,
) -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.runtime_quantization.nvme_staging_dir = str(tmp_path)
    cfg.validate()
    shard = _bucket_non_routed_shard()

    store = ParameterShardStore(cfg)
    baseline = store.parameter_view(shard, step_index=0).clone()
    mutable = store.mutable_parameter(shard, step_index=0)
    mutable.add_(0.25)
    store.finalize_update(
        shard,
        step_index=0,
        offload_after_update=True,
    )
    summary = store.summary()

    assert summary.quantized_shards == 1
    assert summary.quantized_bytes > 0
    assert summary.cumulative_quantize_ops == 1
    assert summary.cumulative_nvme_sync_ops == 1
    assert store._states[shard.group_id].parameter_buffer == ()

    reloaded_store = ParameterShardStore(cfg)
    reloaded = reloaded_store.parameter_view(shard, step_index=1)
    source_summary = reloaded_store.source_summary((shard,))
    load_summary = reloaded_store.step_load_summary()

    assert torch.allclose(reloaded, baseline + 0.25, atol=1e-6)
    assert source_summary.nvme_fp32_mirror_shards == 1
    assert load_summary.nvme_fp32_mirror_loads == 1
    assert not load_summary.direct_manifest_loads


def test_parameter_store_isolates_nvme_mirror_by_session_id(
    tmp_path: Path,
) -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.runtime_quantization.nvme_staging_dir = str(tmp_path)
    cfg.validate()
    shard = _bucket_non_routed_shard()

    store = ParameterShardStore(cfg)
    baseline = store.parameter_view(shard, step_index=0).clone()
    mutable = store.mutable_parameter(shard, step_index=0)
    mutable.add_(0.25)
    store.finalize_update(
        shard,
        step_index=0,
        offload_after_update=True,
    )

    fresh_cfg = build_profile_config("qwen35-35b-a3b")
    fresh_cfg.runtime_quantization.nvme_staging_dir = str(tmp_path)
    fresh_cfg.validate()
    assert (
        fresh_cfg.runtime_quantization.session_id
        != cfg.runtime_quantization.session_id
    )

    fresh_store = ParameterShardStore(fresh_cfg)
    reloaded = fresh_store.parameter_view(shard, step_index=0)
    source_summary = fresh_store.source_summary((shard,))
    load_summary = fresh_store.step_load_summary()

    assert torch.allclose(reloaded, baseline, atol=1e-6)
    assert not torch.allclose(reloaded, baseline + 0.25, atol=1e-6)
    assert source_summary.manifest_backed_shards == 1
    assert load_summary.direct_manifest_loads == 1
    assert not load_summary.nvme_fp32_mirror_loads


def test_parameter_store_load_snapshot_requantizes_from_nvme_mirror(
    tmp_path: Path,
) -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.runtime_quantization.nvme_staging_dir = str(tmp_path)
    cfg.validate()
    shard = _bucket_non_routed_shard()

    store = ParameterShardStore(cfg)
    baseline = store.parameter_view(shard, step_index=0).clone()
    mutable = store.mutable_parameter(shard, step_index=0)
    mutable.add_(0.25)
    store.finalize_update(
        shard,
        step_index=0,
        offload_after_update=True,
    )
    snapshot = store.snapshot()

    reloaded_store = ParameterShardStore(cfg)
    reloaded_store.load_snapshot(snapshot)
    packed = reloaded_store.quantized_parameter_view(
        shard,
        step_index=1,
        device="cpu",
    )

    assert packed is not None
    assert packed.original_numel == 128
    restored = reloaded_store.parameter_view(shard, step_index=1)
    assert torch.allclose(restored, baseline + 0.25, atol=1e-6)


def test_parameter_store_stages_gpu_quantized_cache_when_cuda_available(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU quantized cache staging")

    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.runtime_quantization.nvme_staging_dir = str(tmp_path)
    cfg.execution.compute_device = "gpu"
    cfg.validate()
    shard = _bucket_non_routed_shard()
    store = ParameterShardStore(cfg)

    view = store.parameter_view(
        shard,
        step_index=0,
        device="cuda",
    )
    summary = store.summary()

    assert view.device.type == "cuda"
    assert summary.gpu_quantized_shards == 1
    assert summary.gpu_quantized_bytes > 0
    assert summary.quantized_shards == 1


def test_parameter_store_stage_compute_views_skips_full_gpu_dequantize_in_logical_mode(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU quantized cache staging")

    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.runtime_quantization.nvme_staging_dir = str(tmp_path)
    cfg.execution.compute_device = "gpu"
    cfg.execution.trainable_shard_materialization = "logical"
    cfg.execution.logical_cuda_execution_mode = "compact_layer"
    cfg.model_source.use_local_weight_manifest = False
    cfg.validate()
    shard = _bucket_non_routed_shard()
    store = ParameterShardStore(cfg)

    with patch.object(
        ParameterShardStore,
        "_materialize_quantized_view",
        side_effect=AssertionError(
            "logical-mode stage_compute_views should not pre-stage quantized views"
        ),
    ):
        store.stage_compute_views(
            step_index=0,
            parameter_shards=(shard,),
            device="cuda",
        )

    summary = store.summary()
    assert summary.gpu_quantized_shards == 0
    assert summary.quantized_shards == 0
    assert summary.gpu_cached_shards == 0


def test_parameter_store_stage_compute_views_materializes_full_gpu_view_in_full_bucket_mode(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU quantized cache staging")

    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.runtime_quantization.nvme_staging_dir = str(tmp_path)
    cfg.execution.compute_device = "gpu"
    cfg.execution.trainable_shard_materialization = "logical"
    cfg.execution.logical_cuda_execution_mode = "full_bucket"
    cfg.model_source.use_local_weight_manifest = False
    cfg.validate()
    shard = _bucket_non_routed_shard()
    store = ParameterShardStore(cfg)

    store.stage_compute_views(
        step_index=0,
        parameter_shards=(shard,),
        device="cuda",
    )

    summary = store.summary()
    assert summary.gpu_cached_shards == 1
    assert summary.gpu_cached_bytes > 0
    assert summary.gpu_quantized_shards == 1


def test_parameter_store_finalize_update_retains_gpu_quantized_cache_for_window_resident_shard(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU quantized cache staging")

    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.runtime_quantization.nvme_staging_dir = str(tmp_path)
    cfg.execution.compute_device = "gpu"
    cfg.execution.trainable_shard_materialization = "logical"
    cfg.execution.logical_cuda_execution_mode = "full_bucket"
    cfg.model_source.use_local_weight_manifest = False
    cfg.validate()
    shard = _bucket_non_routed_shard()
    store = ParameterShardStore(cfg)

    store.stage_compute_views(
        step_index=0,
        parameter_shards=(shard,),
        device="cuda",
    )
    mutable = store.mutable_parameter(shard, step_index=0)
    mutable.add_(0.25)
    store.finalize_update(
        shard,
        step_index=0,
        offload_after_update=False,
        sync_fp32_to_nvme=False,
        retain_gpu_quantized_cache=True,
    )

    summary = store.summary()
    cpu_view = store.parameter_view(shard, step_index=1)
    gpu_view = store.parameter_view(shard, step_index=1, device="cuda").cpu()

    assert summary.gpu_cached_shards == 0
    assert summary.gpu_quantized_shards == 1
    assert summary.quantized_shards == 1
    assert torch.allclose(gpu_view, cpu_view, atol=2e-2)


def test_parameter_store_exposes_quantized_parameter_view(tmp_path: Path) -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.runtime_quantization.nvme_staging_dir = str(tmp_path)
    cfg.validate()
    shard = _bucket_non_routed_shard()
    store = ParameterShardStore(cfg)

    store.parameter_view(shard, step_index=0)
    packed = store.quantized_parameter_view(shard, step_index=0, device="cpu")

    assert packed is not None
    assert packed.original_numel == 128
    assert packed.resident_bytes > 0


def test_parameter_store_skips_eager_requantize_on_logical_offload(
    tmp_path: Path,
) -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.execution.trainable_shard_materialization = "logical"
    cfg.runtime_quantization.nvme_staging_dir = str(tmp_path)
    cfg.validate()
    shard = _bucket_non_routed_shard()
    store = ParameterShardStore(cfg)

    mutable = store.mutable_parameter(shard, step_index=0)
    mutable.add_(0.25)
    store.finalize_update(
        shard,
        step_index=0,
        offload_after_update=True,
    )
    summary = store.summary()
    state = store._states[shard.group_id]

    assert summary.cumulative_nvme_sync_ops == 1
    assert summary.cumulative_quantize_ops == 0
    assert summary.quantized_shards == 0
    assert state.resident_tier == "nvme_cold"
    assert state.parameter_tensor is None
    assert state.quantized_parameter is None
