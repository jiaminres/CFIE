"""Unit tests for the CFIE CPU optimizer runtime."""

from __future__ import annotations

import torch

from cfie_training.profiles import build_profile_config
from cfie_training.runtime.executor import GradientPayload
from cfie_training.runtime.optimizer import (
    CPUOptimizerRuntime,
    _HostGradientBufferManager,
    _OptimizerShardState,
)
from cfie_training.runtime.types import ParameterShardSnapshot


def test_host_gradient_buffer_manager_skips_empty_payloads() -> None:
    manager = _HostGradientBufferManager(
        storage_dtype_name="fp8_e4m3fn",
        scope="current_bucket_only",
    )

    staged = manager.stage(
        (
            GradientPayload(
                group_id="empty",
                logical_params=0,
                gradient=torch.empty(0, dtype=torch.float32),
            ),
            GradientPayload(
                group_id="non_empty",
                logical_params=4,
                gradient=torch.ones(4, dtype=torch.float32),
            ),
        )
    )

    assert "empty" not in staged
    assert "non_empty" in staged
    assert len(staged["non_empty"]) == 1
    assert staged["non_empty"][0].gradient.shape == (4,)
    assert manager.last_bucket_staged_bytes == 4
    assert manager.peak_bucket_staged_bytes == 4


def test_optimizer_stabilizes_non_finite_params_and_gradients() -> None:
    runtime = CPUOptimizerRuntime(build_profile_config("qwen35-35b-a3b"))
    state = _OptimizerShardState(
        group_id="bucket_active_experts:test",
        component="bucket_active_experts",
        logical_params=4,
        representative_params=4,
        exp_avg_buffer=(float("nan"), float("inf"), -float("inf"), 0.0),
        exp_avg_sq_buffer=(float("nan"), float("inf"), 1.0, 0.0),
    )
    params = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.5],
        dtype=torch.float32,
        device="cpu",
    )
    gradient = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.25],
        dtype=torch.float32,
        device="cpu",
    )

    grad_norm, param_norm_before, param_norm_after = runtime._apply_adamw_update(
        state=state,
        params=params,
        gradient=gradient,
    )

    assert grad_norm >= 0.0
    assert param_norm_before >= 0.0
    assert param_norm_after >= 0.0
    assert torch.isfinite(params).all()
    exp_avg, exp_avg_sq = runtime._materialize_cpu_state(state)
    assert torch.isfinite(exp_avg).all()
    assert torch.isfinite(exp_avg_sq).all()


def test_optimizer_uses_logical_param_count_when_materialization_mode_is_logical() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.model_source.use_local_weight_manifest = False
    cfg.execution.trainable_shard_materialization = "logical"
    runtime = CPUOptimizerRuntime(cfg)
    shard = ParameterShardSnapshot(
        group_id="bucket_active_experts:test",
        component="bucket_active_experts",
        residency_state="nvme_cold",
        committed_version=1,
        pending_version=None,
        logical_params=256,
        bucket_id=0,
        expert_ids=(0, 1),
        last_touched_step=0,
    )

    state = runtime._get_or_create_state(shard)
    params = torch.zeros(256, dtype=torch.float32, device="cpu")
    gradient = torch.full((256,), 0.01, dtype=torch.float32, device="cpu")

    assert state.representative_params == 256
    grad_norm, param_norm_before, param_norm_after = runtime._apply_adamw_update(
        state=state,
        params=params,
        gradient=gradient,
    )

    assert grad_norm > 0.0
    assert param_norm_before == 0.0
    assert param_norm_after > 0.0
    assert params.numel() == 256


def test_optimizer_uses_tensor_backed_zero_state_for_large_logical_shards() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.model_source.use_local_weight_manifest = False
    cfg.execution.trainable_shard_materialization = "logical"
    cfg.optimizer.offload_state_after_update = False
    runtime = CPUOptimizerRuntime(cfg)
    shard = ParameterShardSnapshot(
        group_id="bucket_active_experts:test-large",
        component="bucket_active_experts",
        residency_state="nvme_cold",
        committed_version=1,
        pending_version=None,
        logical_params=1024,
        bucket_id=0,
        expert_ids=(0, 1),
        last_touched_step=0,
    )

    state = runtime._get_or_create_state(shard)

    assert state.representative_params > 128
    assert isinstance(state.exp_avg_buffer, torch.Tensor)
    assert isinstance(state.exp_avg_sq_buffer, torch.Tensor)


def test_optimizer_offloads_large_logical_state_to_nvme_mirror(
    tmp_path,
) -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.model_source.use_local_weight_manifest = False
    cfg.execution.trainable_shard_materialization = "logical"
    cfg.runtime_quantization.nvme_staging_dir = str(tmp_path)
    runtime = CPUOptimizerRuntime(cfg)
    shard = ParameterShardSnapshot(
        group_id="bucket_active_experts:test-nvme",
        component="bucket_active_experts",
        residency_state="nvme_cold",
        committed_version=1,
        pending_version=None,
        logical_params=1024,
        bucket_id=0,
        expert_ids=(0, 1),
        last_touched_step=0,
    )

    state = runtime._get_or_create_state(shard)
    params = torch.zeros(1024, dtype=torch.float32, device="cpu")

    assert state.exp_avg_buffer == ()
    assert state.exp_avg_sq_buffer == ()

    runtime._apply_adamw_update(
        state=state,
        params=params,
        gradient=torch.full((1024,), 0.1, dtype=torch.float32),
    )
    runtime._offload_cpu_state(state)

    assert state.exp_avg_tensor is None
    assert state.exp_avg_sq_tensor is None
    assert state.exp_avg_buffer == ()
    assert state.exp_avg_sq_buffer == ()
    assert runtime._state_mirror._path_for_group(shard.group_id).exists()

    exp_avg, exp_avg_sq = runtime._materialize_cpu_state(state)

    assert exp_avg.shape == (1024,)
    assert exp_avg_sq.shape == (1024,)
    assert torch.count_nonzero(exp_avg).item() > 0
    assert torch.isfinite(exp_avg_sq).all()


def test_optimizer_applies_slice_gradients_without_dense_full_gradient() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.model_source.use_local_weight_manifest = False
    cfg.execution.trainable_shard_materialization = "logical"
    runtime = CPUOptimizerRuntime(cfg)
    shard = ParameterShardSnapshot(
        group_id="bucket_non_routed:test-slice",
        component="bucket_non_routed",
        residency_state="nvme_cold",
        committed_version=1,
        pending_version=None,
        logical_params=256,
        bucket_id=0,
        last_touched_step=0,
    )
    state = runtime._get_or_create_state(shard)
    params = torch.zeros(256, dtype=torch.float32, device="cpu")

    grad_norm, param_norm_before, param_norm_after = runtime._apply_gradient_payloads(
        state=state,
        params=params,
        gradient_payloads=(
            GradientPayload(
                group_id=shard.group_id,
                logical_params=4,
                gradient=torch.full((4,), 0.1, dtype=torch.float32),
                start_offset=8,
            ),
            GradientPayload(
                group_id=shard.group_id,
                logical_params=4,
                gradient=torch.full((4,), -0.2, dtype=torch.float32),
                start_offset=20,
            ),
        ),
    )

    assert grad_norm > 0.0
    assert param_norm_before == 0.0
    assert param_norm_after > 0.0
    assert torch.count_nonzero(params[:8]).item() == 0
    assert torch.count_nonzero(params[12:20]).item() == 0
    assert torch.count_nonzero(params[24:]).item() == 0
    assert state.exp_avg_tensor is None
    assert state.exp_avg_sq_tensor is None
    assert (8, 4) in state.sparse_exp_avg
    assert (20, 4) in state.sparse_exp_avg
    assert (8, 4) in state.sparse_exp_avg_sq
    assert (20, 4) in state.sparse_exp_avg_sq
    assert torch.count_nonzero(state.sparse_exp_avg[(8, 4)]).item() > 0
    assert torch.count_nonzero(state.sparse_exp_avg_sq[(20, 4)]).item() > 0


def test_optimizer_offloads_sparse_logical_state_to_nvme_mirror(tmp_path) -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.model_source.use_local_weight_manifest = False
    cfg.execution.trainable_shard_materialization = "logical"
    cfg.runtime_quantization.nvme_staging_dir = str(tmp_path)
    runtime = CPUOptimizerRuntime(cfg)
    shard = ParameterShardSnapshot(
        group_id="bucket_non_routed:test-sparse-nvme",
        component="bucket_non_routed",
        residency_state="nvme_cold",
        committed_version=1,
        pending_version=None,
        logical_params=256,
        bucket_id=0,
        last_touched_step=0,
    )
    state = runtime._get_or_create_state(shard)
    params = torch.zeros(256, dtype=torch.float32, device="cpu")

    runtime._apply_gradient_payloads(
        state=state,
        params=params,
        gradient_payloads=(
            GradientPayload(
                group_id=shard.group_id,
                logical_params=8,
                gradient=torch.full((8,), 0.1, dtype=torch.float32),
                start_offset=16,
            ),
        ),
    )
    expected_exp_avg = state.sparse_exp_avg[(16, 8)].clone()
    expected_exp_avg_sq = state.sparse_exp_avg_sq[(16, 8)].clone()

    runtime._offload_cpu_state(state)

    assert state.sparse_exp_avg == {}
    assert state.sparse_exp_avg_sq == {}

    reloaded_exp_avg, reloaded_exp_avg_sq = runtime._sparse_state_views(
        state,
        start_offset=16,
        size=8,
    )

    assert torch.allclose(reloaded_exp_avg, expected_exp_avg, atol=5e-4, rtol=1e-3)
    assert torch.allclose(
        reloaded_exp_avg_sq,
        expected_exp_avg_sq,
        atol=5e-4,
        rtol=1e-3,
    )
