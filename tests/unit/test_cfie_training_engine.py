"""Unit tests for the first-version CFIE training engine."""

from __future__ import annotations

import os
from pathlib import Path
import math

import pytest
import torch

from cfie_training.config import TrainingProjectConfig
from cfie_training.predictor.trainer import EngineRouterTeacherModelBackend
from cfie_training.profiles import build_profile_config
from cfie_training.runtime.engine import FirstVersionTrainingEngine
from cfie_training.runtime.executor import RepresentativeBucketExecutor
from cfie_training.runtime.planner import ExpertRotationScheduler, LayerBucketPlanner
from cfie_training.runtime.quantization import runtime_device_weight_bytes_per_param
from cfie_training.runtime.store import ParameterShardStore
from cfie_training.runtime.types import BatchShape, ExpertWindowPlan


def _expected_window(
    cfg,
    *,
    step_index: int,
    batch: BatchShape,
    next_batch: BatchShape | None = None,
):
    scheduler = ExpertRotationScheduler(cfg)
    return scheduler.plan_window(
        step_index=step_index,
        batch=batch,
        layer_buckets=LayerBucketPlanner(cfg).build(),
        next_batch=batch if next_batch is None else next_batch,
    )


def _expected_peak_weight_stage_bytes(engine: FirstVersionTrainingEngine) -> int:
    parameter_shards = engine._warehouse.snapshot()
    bytes_per_param = runtime_device_weight_bytes_per_param(engine.config)
    return max(
        math.ceil(
            sum(
                shard.logical_params * bytes_per_param
                for shard in engine._executor.select_bucket_shards(
                    bucket_id=bucket.bucket_id,
                    parameter_shards=parameter_shards,
                )
            )
        )
        for bucket in engine.layer_buckets
    )


def _expected_peak_gradient_stage_bytes(engine: FirstVersionTrainingEngine) -> int:
    gradient_dtype = engine.config.optimizer.gradient_buffer_storage_dtype
    bytes_per_param = 4 if gradient_dtype == "fp32" else 2 if gradient_dtype in {
        "fp16",
        "bf16",
    } else 1
    return max(
        sum(
            shard.logical_params * bytes_per_param
            for shard in engine._executor.select_bucket_shards(
                bucket_id=bucket.bucket_id,
                parameter_shards=engine._warehouse.snapshot(),
            )
        )
        for bucket in engine.layer_buckets
    )


def test_layer_bucket_planner_builds_qwen35_windows() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")

    buckets = LayerBucketPlanner(cfg).build()

    assert len(buckets) == 10
    assert buckets[0].layer_indices == (0, 1, 2, 3)
    assert buckets[0].attention_types == (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    )
    assert buckets[-1].layer_indices == (36, 37, 38, 39)


def test_layer_bucket_planner_builds_hybrid_windows() -> None:
    cfg = TrainingProjectConfig.from_dict({
        "profile_name": "hybrid-window-smoke",
        "model_spec": {
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "text_model_type": "qwen3_5_moe_text",
            "hidden_size": 256,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "num_experts": 16,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 128,
            "full_attention_interval": 4,
            "max_position_embeddings": 4096,
            "mtp_num_hidden_layers": 1,
            "attention_pattern": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "total_params_billion": 0.2,
        },
        "bucket_schedule": {
            "unit": "hybrid",
        },
    })

    buckets = LayerBucketPlanner(cfg).build()

    assert [bucket.layer_indices for bucket in buckets] == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    ]
    assert [bucket.attention_types for bucket in buckets] == [
        ("linear_attention", "linear_attention"),
        ("linear_attention", "full_attention"),
        ("linear_attention", "linear_attention"),
        ("linear_attention", "full_attention"),
    ]


def test_layer_bucket_planner_can_append_mtp_bucket() -> None:
    cfg = TrainingProjectConfig.from_dict({
        "profile_name": "mtp-bucket-smoke",
        "model_spec": {
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "text_model_type": "qwen3_5_moe_text",
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "num_experts": 16,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 128,
            "full_attention_interval": 4,
            "max_position_embeddings": 4096,
            "mtp_num_hidden_layers": 1,
            "attention_pattern": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "total_params_billion": 0.2,
        },
        "bucket_schedule": {
            "unit": "expert",
            "include_mtp_dedicated_bucket": True,
        },
    })

    buckets = LayerBucketPlanner(cfg).build()

    assert len(buckets) == 2
    assert buckets[0].layer_indices == (0, 1, 2, 3)
    assert buckets[1].layer_indices == (4,)
    assert buckets[1].attention_types == ("mtp",)


def test_expert_rotation_scheduler_wraps_after_full_cycle() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    scheduler = ExpertRotationScheduler(cfg)

    assert scheduler.active_window(0) == (0, 1, 2, 3, 4, 5, 6, 7)
    assert scheduler.active_window(1) == (8, 9, 10, 11, 12, 13, 14, 15)
    assert scheduler.active_window(32) == (0, 1, 2, 3, 4, 5, 6, 7)
    assert scheduler.prefetched_window(0) == (8, 9, 10, 11, 12, 13, 14, 15)


def test_expert_rotation_scheduler_supports_sample_and_token_budgets() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.expert_rotation.rotate_every_steps = 1
    cfg.expert_rotation.rotate_every_samples = 4
    scheduler = ExpertRotationScheduler(cfg)

    assert scheduler.active_window(
        0,
        cumulative_samples_processed=0,
    ) == (0, 1, 2, 3, 4, 5, 6, 7)
    assert scheduler.active_window(
        1,
        cumulative_samples_processed=2,
    ) == (0, 1, 2, 3, 4, 5, 6, 7)
    assert scheduler.active_window(
        2,
        cumulative_samples_processed=4,
    ) == (8, 9, 10, 11, 12, 13, 14, 15)

    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.expert_rotation.rotate_every_steps = 1
    cfg.expert_rotation.rotate_every_tokens = 1024
    scheduler = ExpertRotationScheduler(cfg)

    assert scheduler.active_window(
        0,
        cumulative_tokens_processed=0,
    ) == (0, 1, 2, 3, 4, 5, 6, 7)
    assert scheduler.active_window(
        1,
        cumulative_tokens_processed=512,
    ) == (0, 1, 2, 3, 4, 5, 6, 7)
    assert scheduler.active_window(
        2,
        cumulative_tokens_processed=1024,
    ) == (8, 9, 10, 11, 12, 13, 14, 15)


def test_predictor_teacher_backend_uses_dedicated_teacher_engine_config() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    backend = EngineRouterTeacherModelBackend.create(cfg)

    assert backend._resolve_engine_offload_backend() == "auto"
    assert backend._resolve_cpu_offload_gb() == 0.0
    assert backend._resolve_moe_cpu_budget_gb() == 0.0
    assert backend._resolve_moe_cpu_min_free_gb() == 0.0
    assert backend._resolve_gpu_memory_utilization() == pytest.approx(0.6)
    assert backend._resolve_engine_additional_config() == {}

    cfg.resource_policy.weight_offload_backend = "cpu"
    cfg.memory_budget.gpu_hot_budget_gb = 1.0
    cfg.memory_budget.cpu_hot_budget_gb = 1.0
    backend = EngineRouterTeacherModelBackend.create(cfg)

    assert backend._resolve_engine_offload_backend() == "auto"
    assert backend._resolve_cpu_offload_gb() == 0.0
    assert backend._resolve_moe_cpu_budget_gb() == 0.0
    assert backend._resolve_moe_cpu_min_free_gb() == 0.0
    assert backend._resolve_gpu_memory_utilization() == pytest.approx(0.6)


def test_predictor_teacher_backend_allows_profile_gpu_util_override() -> None:
    cfg = build_profile_config("qwen35-122b-a10b")
    backend = EngineRouterTeacherModelBackend.create(cfg)

    assert backend._resolve_gpu_memory_utilization() == pytest.approx(0.55)


def test_predictor_teacher_backend_allows_explicit_teacher_engine_overrides() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.teacher_engine.gpu_memory_utilization = 0.7
    cfg.teacher_engine.offload_backend = "uva"
    cfg.teacher_engine.cpu_offload_gb = 12.0
    cfg.teacher_engine.moe_cpu_budget_gb = 24.0
    cfg.teacher_engine.moe_cpu_min_free_gb = 6.0
    backend = EngineRouterTeacherModelBackend.create(cfg)

    assert backend._resolve_gpu_memory_utilization() == pytest.approx(0.7)
    assert backend._resolve_engine_offload_backend() == "uva"
    assert backend._resolve_cpu_offload_gb() == pytest.approx(12.0)
    assert backend._resolve_moe_cpu_budget_gb() == pytest.approx(24.0)
    assert backend._resolve_moe_cpu_min_free_gb() == pytest.approx(6.0)


def test_expert_rotation_scheduler_blends_next_step_prefetch_with_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = TrainingProjectConfig.from_dict({
        "profile_name": "router-hotness-smoke",
        "model_spec": {
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "text_model_type": "qwen3_5_moe_text",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "num_experts": 6,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64,
            "full_attention_interval": 4,
            "max_position_embeddings": 2048,
            "mtp_num_hidden_layers": 1,
            "attention_pattern": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "total_params_billion": 0.05,
        },
        "expert_rotation": {
            "selection_strategy": "router_hotness",
            "active_experts_per_step": 2,
            "next_step_score_weight": 1.0,
            "prefetch_active_overlap": 1,
        },
        "bucket_schedule": {
            "prefetch_buckets": 1,
        },
    })
    scheduler = ExpertRotationScheduler(cfg)

    score_map = {
        0: torch.tensor([0.1, 0.2, 0.9, 0.8, 0.4, 0.3], dtype=torch.float32),
        1: torch.tensor([0.95, 0.85, 0.2, 0.1, 0.7, 0.6], dtype=torch.float32),
    }

    def _fake_router_hotness_scores(self, *, step_index, batch, layer_buckets):
        return score_map[step_index]

    monkeypatch.setattr(
        ExpertRotationScheduler,
        "_router_hotness_scores",
        _fake_router_hotness_scores,
    )

    plan = scheduler.plan_window(
        step_index=0,
        batch=BatchShape(samples=1, tokens_per_sample=16),
        layer_buckets=LayerBucketPlanner(cfg).build(),
        next_batch=BatchShape(samples=1, tokens_per_sample=16),
    )

    assert plan.selection_strategy == "router_hotness"
    assert plan.router_score_source == "blended_next_step_router_hotness"
    assert plan.active_expert_ids == (2, 3)
    assert plan.prefetched_expert_ids == (2, 0)
    assert plan.prefetch_priority_expert_ids[:4] == (0, 1, 4, 5)


def test_expert_rotation_scheduler_keeps_router_selected_window_stable_within_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = TrainingProjectConfig.from_dict({
        "profile_name": "router-window-retain-smoke",
        "model_spec": {
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "text_model_type": "qwen3_5_moe_text",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "num_experts": 6,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64,
            "full_attention_interval": 4,
            "max_position_embeddings": 2048,
            "mtp_num_hidden_layers": 1,
            "attention_pattern": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "total_params_billion": 0.05,
        },
        "expert_rotation": {
            "selection_strategy": "router_hotness",
            "active_experts_per_step": 2,
            "rotate_every_steps": 2,
            "next_step_score_weight": 0.0,
        },
        "bucket_schedule": {
            "prefetch_buckets": 1,
        },
    })
    scheduler = ExpertRotationScheduler(cfg)

    score_map = {
        0: torch.tensor([0.1, 0.2, 0.9, 0.8, 0.4, 0.3], dtype=torch.float32),
        1: torch.tensor([0.95, 0.85, 0.2, 0.1, 0.7, 0.6], dtype=torch.float32),
        2: torch.tensor([0.92, 0.87, 0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
        3: torch.tensor([0.90, 0.86, 0.2, 0.1, 0.3, 0.4], dtype=torch.float32),
    }

    def _fake_router_hotness_scores(self, *, step_index, batch, layer_buckets):
        return score_map[step_index]

    monkeypatch.setattr(
        ExpertRotationScheduler,
        "_router_hotness_scores",
        _fake_router_hotness_scores,
    )
    layer_buckets = LayerBucketPlanner(cfg).build()
    batch = BatchShape(samples=1, tokens_per_sample=16)

    first_plan = scheduler.plan_window(
        step_index=0,
        batch=batch,
        layer_buckets=layer_buckets,
        next_batch=batch,
    )
    second_plan = scheduler.plan_window(
        step_index=1,
        batch=batch,
        layer_buckets=layer_buckets,
        next_batch=batch,
    )
    third_plan = scheduler.plan_window(
        step_index=2,
        batch=batch,
        layer_buckets=layer_buckets,
        next_batch=batch,
    )

    assert first_plan.active_expert_ids == (2, 3)
    assert second_plan.active_expert_ids == first_plan.active_expert_ids
    assert third_plan.active_expert_ids == (0, 1)


def test_first_version_engine_simulates_step_actions() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    engine = FirstVersionTrainingEngine(cfg)
    batch = BatchShape(samples=2, tokens_per_sample=512)
    expected_window = _expected_window(
        cfg,
        step_index=0,
        batch=batch,
    )

    step = engine.run_step(batch)

    assert step.step_index == 0
    assert step.static_modules_staged is True
    assert step.active_expert_ids == expected_window.active_expert_ids
    assert step.prefetched_expert_ids == expected_window.prefetched_expert_ids
    assert step.expert_window_plan == expected_window
    assert len(step.layer_buckets) == 10
    assert [micro_batch.samples for micro_batch in step.scheduled_micro_batches] == [1, 1]
    assert len(step.bucket_stream_traces) == 10
    assert len(step.stream_operations) == 40
    assert step.actions[0].name == "stage_static_modules"
    assert step.actions[1].name == "prefetch_routed_experts"
    assert step.actions[1].metadata["compute_device"] == cfg.execution.compute_device
    assert step.actions[1].metadata["selection_strategy"] == "router_hotness"
    assert step.actions[1].metadata["router_score_source"] == (
        expected_window.router_score_source
    )
    assert step.actions[-1].name == "rotate_expert_window"
    assert sum(action.name == "forward_bucket" for action in step.actions) == 10
    assert sum(action.name == "backward_bucket" for action in step.actions) == 10
    assert (
        sum(action.name == "cpu_update_release_bucket" for action in step.actions) == 10
    )
    assert len(step.residency_transitions) == 72
    first_expert_activation = next(
        transition
        for transition in step.residency_transitions
        if transition.group_id == "bucket_active_experts:0"
        and transition.to_state == "gpu_active"
    )
    assert first_expert_activation.from_state == "nvme_cold"
    assert step.residency_ending_states["static_modules"] == "cpu_staged"
    assert step.residency_ending_states["expert_window:1"] == "cpu_staged"
    assert step.warehouse_summary is not None
    assert step.warehouse_summary.total_shards == 22
    assert step.warehouse_summary.cpu_staged == 2
    assert step.parameter_store_summary is not None
    assert step.parameter_store_summary.tracked_shards == 20
    assert step.parameter_store_summary.nvme_cold_shards == 20
    assert step.parameter_store_summary.gpu_cached_shards == 0
    assert step.parameter_store_summary.gpu_cached_bytes == 0
    assert step.parameter_store_summary.manifest_backed_shards == 20
    assert step.parameter_store_summary.synthetic_seeded_shards == 0
    assert 0 < step.parameter_store_summary.transport_backed_shards < 20
    assert step.parameter_store_summary.source_file_count > 0
    assert step.parameter_store_summary.source_tensor_count > 0
    assert step.parameter_store_summary.cumulative_gpu_stage_ops == 0
    assert step.parameter_store_summary.cumulative_gpu_release_ops == 0
    assert step.parameter_source_summary is not None
    assert step.parameter_source_summary.touched_shards == 20
    assert step.parameter_source_summary.manifest_backed_shards == 20
    assert step.parameter_source_summary.synthetic_seeded_shards == 0
    assert 0 < step.parameter_source_summary.transport_backed_shards < 20
    assert step.parameter_source_summary.file_count > 0
    assert step.parameter_source_summary.tensor_count > 0
    first_non_routed_source = next(
        record
        for record in step.parameter_source_summary.shard_sources
        if record.group_id == "bucket_non_routed:0"
    )
    assert first_non_routed_source.layer_indices == (0, 1, 2, 3)
    assert "linear_attn_in_proj_qkv" in first_non_routed_source.semantic_roles
    assert "self_attn_q_proj" in first_non_routed_source.semantic_roles
    assert step.parameter_prefetch_summary is not None
    assert step.parameter_prefetch_summary.touched_shards == 20
    assert step.parameter_prefetch_summary.transport_cache_prefetches > 0
    assert step.parameter_prefetch_summary.direct_manifest_prefetches > 0
    assert step.parameter_prefetch_summary.buffer_reuses == 0
    assert step.parameter_prefetch_summary.cpu_hot_reuses == 0
    assert step.parameter_load_summary is not None
    assert step.parameter_load_summary.touched_shards == 20
    assert step.parameter_load_summary.transport_cache_loads == 0
    assert step.parameter_load_summary.direct_manifest_loads == 0
    assert step.parameter_load_summary.buffer_reuses == 0
    assert step.parameter_load_summary.cpu_hot_reuses == 20
    assert step.transport_summary is not None
    assert step.transport_summary.manifest_available is True
    assert step.transport_summary.matched_shards == 22
    assert step.transport_summary.unmatched_shards == 0
    assert step.transport_summary.file_count > 5
    assert step.transport_summary.tensor_count > 0
    assert step.transport_summary.estimated_stage_bytes > 0
    assert step.transport_execution_summary is not None
    assert step.transport_execution_summary.manifest_available is True
    assert (
        step.transport_execution_summary.requested_file_count
        == step.transport_summary.file_count
    )
    assert step.transport_execution_summary.staged_file_count == 2
    assert step.transport_execution_summary.cache_miss_shards == 22
    assert step.transport_execution_summary.cache_file_count > 0
    assert (
        step.transport_execution_summary.cache_resident_bytes
        <= step.transport_execution_summary.max_cache_bytes
    )
    assert (
        step.transport_execution_summary.weight_stage_buffer_bytes
        == _expected_peak_weight_stage_bytes(engine)
    )
    assert (
        step.transport_execution_summary.gradient_stage_buffer_bytes
        == _expected_peak_gradient_stage_bytes(engine)
    )
    assert step.transport_execution_summary.h2d_transfer_bytes > 0
    assert step.transport_execution_summary.d2h_transfer_bytes > 0
    assert step.transport_execution_summary.overlap_eligible_bytes >= (
        step.transport_execution_summary.h2d_transfer_bytes
        + step.transport_execution_summary.d2h_transfer_bytes
    )
    assert step.transport_execution_summary.released_buffer_count > 0
    assert step.transport_execution_summary.active_buffer_count == 0
    assert step.transport_execution_summary.pooled_buffer_count == 40
    assert step.execution_summary is not None
    assert step.execution_summary.executed_buckets == 10
    assert step.execution_summary.gradient_shards == 40
    assert step.execution_summary.total_loss > 0
    assert step.execution_summary.max_gradient_l2_norm > 0
    assert step.execution_summary.peak_host_gradient_buffer_bytes > 0
    assert step.execution_summary.gradient_buffer_storage_dtype == "fp8_e4m3fn"
    assert step.stream_overlap_summary is not None
    assert step.stream_overlap_summary.micro_batch_count == 2
    assert step.stream_overlap_summary.compute_operation_count == 20
    assert step.stream_overlap_summary.transfer_operation_count == 20
    assert step.stream_overlap_summary.estimated_step_makespan_us > 0
    assert 0.0 < step.stream_overlap_summary.overlap_ratio < 1.0
    assert len(step.optimizer_updates) == 40
    assert step.optimizer_summary is not None
    assert step.optimizer_summary.tracked_shards == 20
    assert step.optimizer_summary.nvme_cold_shards == 20
    assert step.optimizer_summary.state_storage_dtype == "fp8_e4m3fn"
    assert step.optimizer_summary.gradient_buffer_storage_dtype == "fp8_e4m3fn"
    assert step.optimizer_summary.gradient_buffer_scope == "current_bucket_only"
    assert step.optimizer_summary.last_bucket_staged_gradient_bytes > 0
    first_bucket_trace = step.bucket_stream_traces[0]
    assert first_bucket_trace.bucket_id == 0
    assert first_bucket_trace.cpu_hot_shards_before_prefetch == 0
    assert first_bucket_trace.lookahead_prefetched_bucket_ids == (1,)
    assert first_bucket_trace.prefetch_summary.touched_shards == 2
    assert first_bucket_trace.load_summary.touched_shards == 2
    assert first_bucket_trace.load_summary.cpu_hot_reuses == 2
    assert first_bucket_trace.micro_batch_count == 2
    assert first_bucket_trace.bucket_record.semantic_layout_used is True
    assert first_bucket_trace.bucket_record.execution_mode == "structured_qwen35_bucket"
    assert "linear_attn_in_proj_qkv" in first_bucket_trace.bucket_record.semantic_roles
    assert "self_attn_q_proj" in first_bucket_trace.bucket_record.semantic_roles
    assert "expert_gate_up_proj" in first_bucket_trace.bucket_record.semantic_roles
    assert "mlp_router_gate" in first_bucket_trace.bucket_record.semantic_roles
    assert "shared_expert_up_proj" in first_bucket_trace.bucket_record.semantic_roles
    assert first_bucket_trace.optimizer_update_count == 4
    assert first_bucket_trace.gradient_release_count == 4
    assert first_bucket_trace.gradients_released_immediately is True
    assert first_bucket_trace.host_gradient_buffer_bytes > 0
    assert first_bucket_trace.host_gradient_buffer_storage_dtype == "fp8_e4m3fn"
    assert first_bucket_trace.offloaded_shards_after_update == 1
    assert first_bucket_trace.cpu_hot_shards_after_update == 1
    second_bucket_trace = step.bucket_stream_traces[1]
    assert second_bucket_trace.cpu_hot_shards_before_prefetch == 2
    assert second_bucket_trace.lookahead_prefetched_bucket_ids == (2,)
    assert second_bucket_trace.load_summary.cpu_hot_reuses == 2
    last_bucket_trace = step.bucket_stream_traces[-1]
    assert last_bucket_trace.lookahead_prefetched_bucket_ids == ()
    active_suffix = "-".join(str(expert_id) for expert_id in step.active_expert_ids)
    first_expert_shard = next(
        shard
        for shard in step.parameter_shards
        if shard.group_id == f"bucket_active_experts:0:{active_suffix}"
    )
    assert first_expert_shard.residency_state == "nvme_cold"
    assert first_expert_shard.committed_version == 1
    static_shard = next(
        shard for shard in step.parameter_shards if shard.group_id == "static_modules"
    )
    assert static_shard.logical_params == engine.build_memory_plan(step.batch).static_params_total
    assert static_shard.residency_state == "cpu_staged"
    first_optimizer_update = next(
        update
        for update in step.optimizer_updates
        if update.group_id == "bucket_non_routed:0"
    )
    assert first_optimizer_update.algorithm == "adamw"
    assert first_optimizer_update.target_version == 1
    assert first_optimizer_update.offloaded_after_update is True
    assert first_optimizer_update.representative_params == 128
    assert first_optimizer_update.shard_update_count == 1
    assert first_optimizer_update.gradient_l2_norm > 0
    first_stream_op = step.stream_operations[0]
    assert first_stream_op.stream_name == cfg.execution.compute_stream_name
    assert first_stream_op.operation == "bucket_compute"
    assert first_stream_op.micro_batch_id == 0
    assert first_stream_op.bucket_id == 0
    assert first_stream_op.duration_us > 0
    first_update_op = step.stream_operations[1]
    assert first_update_op.stream_name == cfg.execution.transfer_stream_name
    assert first_update_op.operation == "bucket_update_release"
    assert first_update_op.update_group_count == 4
    first_forward_action = next(
        action for action in step.actions if action.name == "forward_bucket"
    )
    assert first_forward_action.metadata["compute_device"] == cfg.execution.compute_device


def test_engine_sanitizes_nonfinite_parameter_buffers() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    engine = FirstVersionTrainingEngine(cfg)
    batch = BatchShape(samples=2, tokens_per_sample=128)

    engine.run_step(batch)

    for state in engine._parameter_store._states.values():
        state.parameter_buffer = tuple(
            float("nan") for _ in range(max(state.representative_params, 1))
        )
        state.parameter_tensor = torch.full(
            (max(state.representative_params, 1),),
            float("nan"),
            dtype=torch.float32,
        )

    step = engine.run_step(batch)

    assert step.execution_summary is not None
    assert math.isfinite(step.execution_summary.total_loss)
    assert math.isfinite(step.execution_summary.max_gradient_l2_norm)
    assert step.stream_overlap_summary is not None
    assert step.stream_overlap_summary.estimated_step_makespan_us > 0
    assert step.optimizer_updates
    assert all(
        math.isfinite(record.non_routed_gradient_l2_norm)
        and math.isfinite(record.expert_gradient_l2_norm)
        for record in step.execution_summary.bucket_records
    )
    assert all(
        math.isfinite(update.gradient_l2_norm) for update in step.optimizer_updates
    )


def test_engine_respects_requested_compute_device() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.execution.compute_device = "gpu"

    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA-capable torch runtime"):
            FirstVersionTrainingEngine(cfg)
        return

    engine = FirstVersionTrainingEngine(cfg)
    step = engine.run_step(BatchShape(samples=1, tokens_per_sample=32))

    assert engine._executor._compute_device.type == "cuda"
    assert step.parameter_store_summary is not None
    assert step.parameter_store_summary.cpu_hot_shards == 1
    assert step.parameter_store_summary.gpu_cached_shards == 1
    assert step.parameter_store_summary.gpu_cached_bytes > 0
    assert step.parameter_store_summary.quantized_shards > 0
    assert step.parameter_store_summary.quantized_bytes > 0
    assert step.parameter_store_summary.gpu_quantized_shards == 1
    assert step.parameter_store_summary.gpu_quantized_bytes > 0
    assert step.parameter_store_summary.cumulative_gpu_stage_ops == 21
    assert step.parameter_store_summary.cumulative_gpu_release_ops == 20
    assert step.parameter_store_summary.cumulative_quantize_ops > 0
    assert step.parameter_store_summary.cumulative_nvme_sync_ops > 0
    assert step.execution_summary is not None
    assert math.isfinite(step.execution_summary.total_loss)
    assert step.execution_summary.total_loss > 0


def test_executor_enables_deterministic_cuda_runtime_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.execution.compute_device = "gpu"
    cfg.execution.deterministic_cuda_execution = True

    calls: list[tuple[bool, bool]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda enabled, warn_only=False: calls.append((enabled, warn_only)),
    )
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    if hasattr(torch.backends, "cudnn"):
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
        monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
        monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)

    executor = RepresentativeBucketExecutor(cfg)

    assert executor.compute_device.type == "cuda"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    if hasattr(torch.backends, "cudnn"):
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
    assert calls == [(True, True)]


def test_engine_retains_gpu_quantized_window_cache_across_same_rotation(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU cache retention")

    cfg = TrainingProjectConfig.from_dict({
        "profile_name": "gpu-window-cache-smoke",
        "model_spec": {
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "text_model_type": "qwen3_5_moe_text",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64,
            "full_attention_interval": 4,
            "max_position_embeddings": 2048,
            "mtp_num_hidden_layers": 1,
            "attention_pattern": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "quantization": "gptq",
            "quant_bits": 4,
            "quant_group_size": 128,
            "quant_sym": True,
            "total_params_billion": 0.1,
        },
        "model_source": {
            "use_local_weight_manifest": False,
        },
        "expert_rotation": {
            "active_experts_per_step": 2,
            "rotate_every_steps": 1,
            "rotate_every_samples": 2,
            "retain_active_window_state_in_memory": True,
        },
        "execution": {
            "compute_device": "gpu",
            "optimizer_device": "cpu",
            "gradient_device": "cpu",
            "trainable_shard_materialization": "logical",
            "logical_cuda_execution_mode": "full_bucket",
        },
        "optimizer": {
            "offload_state_after_update": True,
        },
        "runtime_quantization": {
            "enabled": True,
            "persist_fp32_to_nvme": True,
            "nvme_staging_dir": str(tmp_path),
        },
    })

    engine = FirstVersionTrainingEngine(cfg)
    batch = BatchShape(samples=1, tokens_per_sample=16)

    step0 = engine.run_step(batch, next_batch=batch)
    live_summary = engine._parameter_store.summary()
    step1 = engine.run_step(batch)

    assert step0.active_expert_ids == step1.active_expert_ids
    assert live_summary.gpu_quantized_shards > 0
    assert live_summary.gpu_quantized_shards > live_summary.gpu_cached_shards
    assert step1.parameter_load_summary.cpu_hot_reuses > 0
    assert step1.parameter_load_summary.nvme_fp32_mirror_loads == 0


def test_engine_simulation_includes_resource_plan() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    engine = FirstVersionTrainingEngine(cfg)
    batch = BatchShape(samples=2, tokens_per_sample=128)
    expected_first_window = _expected_window(
        cfg,
        step_index=0,
        batch=batch,
        next_batch=batch,
    )
    expected_second_window = _expected_window(
        cfg,
        step_index=1,
        batch=batch,
        next_batch=batch,
    )

    run_trace = engine.simulate(
        steps=2,
        batch=batch,
    )

    assert run_trace.resource_plan["all_tiers_within_budget"] is True
    assert run_trace.resource_plan["cpu_optimizer_state_storage_dtype"] == "fp8_e4m3fn"
    assert run_trace.resource_plan["host_gradient_buffer_scope"] == "current_bucket_only"
    assert run_trace.resource_plan["gpu_hot"]["resident_gib"] > 0
    assert run_trace.resource_plan["nvme_cold"]["resident_gib"] > 100
    assert run_trace.steps[1].warehouse_summary is not None
    assert run_trace.steps[1].warehouse_summary.cpu_staged == 3
    assert run_trace.steps[1].warehouse_summary.dirty_shards == 0
    assert run_trace.steps[1].parameter_store_summary is not None
    assert run_trace.steps[1].parameter_store_summary.tracked_shards == 30
    assert run_trace.steps[1].parameter_store_summary.gpu_cached_shards == 0
    assert run_trace.steps[1].parameter_store_summary.gpu_cached_bytes == 0
    assert run_trace.steps[1].parameter_store_summary.manifest_backed_shards == 30
    assert 0 < run_trace.steps[1].parameter_store_summary.transport_backed_shards < 30
    assert run_trace.steps[1].parameter_store_summary.cumulative_gpu_stage_ops == 0
    assert run_trace.steps[1].parameter_store_summary.cumulative_gpu_release_ops == 0
    assert run_trace.steps[1].parameter_source_summary is not None
    assert run_trace.steps[1].parameter_source_summary.touched_shards == 20
    assert run_trace.steps[1].parameter_source_summary.manifest_backed_shards == 20
    assert 0 < run_trace.steps[1].parameter_source_summary.transport_backed_shards < 20
    assert run_trace.steps[1].parameter_prefetch_summary is not None
    assert run_trace.steps[1].parameter_prefetch_summary.touched_shards == 20
    assert run_trace.steps[1].parameter_prefetch_summary.transport_cache_prefetches > 0
    assert run_trace.steps[1].parameter_prefetch_summary.direct_manifest_prefetches > 0
    assert (
        run_trace.steps[1].parameter_prefetch_summary.buffer_reuses
        + run_trace.steps[1].parameter_prefetch_summary.nvme_fp32_mirror_prefetches
        > 0
    )
    assert run_trace.steps[1].parameter_prefetch_summary.cpu_hot_reuses == 0
    assert run_trace.steps[1].parameter_load_summary is not None
    assert run_trace.steps[1].parameter_load_summary.touched_shards == 20
    assert run_trace.steps[1].parameter_load_summary.transport_cache_loads == 0
    assert run_trace.steps[1].parameter_load_summary.direct_manifest_loads == 0
    assert run_trace.steps[1].parameter_load_summary.buffer_reuses == 0
    assert run_trace.steps[1].parameter_load_summary.cpu_hot_reuses == 20
    assert run_trace.steps[1].transport_summary is not None
    assert run_trace.steps[1].transport_summary.manifest_available is True
    assert run_trace.steps[1].transport_summary.matched_shards == 21
    assert run_trace.steps[1].transport_summary.file_count > 5
    assert run_trace.steps[1].transport_execution_summary is not None
    assert (
        run_trace.steps[1].transport_execution_summary.requested_file_count
        == run_trace.steps[1].transport_summary.file_count
    )
    assert run_trace.steps[1].transport_execution_summary.reused_file_count == 2
    assert run_trace.steps[1].transport_execution_summary.cache_hit_shards > 0
    assert run_trace.steps[1].transport_execution_summary.cache_file_count > 0
    assert (
        run_trace.steps[1].transport_execution_summary.cache_resident_bytes
        <= run_trace.steps[1].transport_execution_summary.max_cache_bytes
    )
    assert run_trace.steps[1].transport_execution_summary.weight_stage_buffer_bytes > 0
    assert run_trace.steps[1].transport_execution_summary.gradient_stage_buffer_bytes > 0
    assert run_trace.steps[1].transport_execution_summary.h2d_transfer_bytes > 0
    assert run_trace.steps[1].transport_execution_summary.d2h_transfer_bytes > 0
    assert run_trace.steps[1].transport_execution_summary.released_buffer_count > 0
    assert run_trace.steps[1].execution_summary is not None
    assert run_trace.steps[1].execution_summary.executed_buckets == 10
    assert run_trace.steps[1].execution_summary.gradient_shards == 20
    assert run_trace.steps[1].execution_summary.peak_host_gradient_buffer_bytes > 0
    assert run_trace.steps[1].stream_overlap_summary is not None
    assert run_trace.steps[1].stream_overlap_summary.micro_batch_count == 1
    assert run_trace.steps[1].optimizer_summary is not None
    assert run_trace.steps[1].optimizer_summary.tracked_shards == 30
    assert run_trace.steps[1].optimizer_summary.cumulative_updates_applied == 40
    assert run_trace.steps[1].optimizer_summary.gradient_buffer_scope == "current_bucket_only"
    assert len(run_trace.steps[1].optimizer_updates) == 20
    assert len(run_trace.steps[1].bucket_stream_traces) == 10
    assert all(
        trace.optimizer_update_count == 2
        for trace in run_trace.steps[1].bucket_stream_traces
    )
    assert all(
        trace.offloaded_shards_after_update == 1
        for trace in run_trace.steps[1].bucket_stream_traces
    )
    assert all(
        trace.cpu_hot_shards_after_update == 1
        for trace in run_trace.steps[1].bucket_stream_traces
    )
    assert run_trace.steps[1].bucket_stream_traces[0].lookahead_prefetched_bucket_ids == (1,)
    assert run_trace.steps[1].bucket_stream_traces[1].cpu_hot_shards_before_prefetch == 2
    second_step = run_trace.steps[1]
    assert run_trace.steps[0].expert_window_plan == expected_first_window
    assert second_step.expert_window_plan == expected_second_window
    assert second_step.active_expert_ids == expected_second_window.active_expert_ids
    assert second_step.prefetched_expert_ids == expected_second_window.prefetched_expert_ids
    second_expert_activation = next(
        transition
        for transition in second_step.residency_transitions
        if transition.group_id == "bucket_active_experts:0"
        and transition.to_state == "gpu_active"
    )
    assert second_expert_activation.from_state == "nvme_cold"
    next_prefetch = next(
        transition
        for transition in second_step.residency_transitions
        if transition.group_id == "expert_window:2"
        and transition.trigger == "prefetch_next_step_window"
    )
    assert next_prefetch.to_state == "cpu_staged"
    second_active_suffix = "-".join(
        str(expert_id) for expert_id in second_step.active_expert_ids
    )
    second_expert_shard = next(
        shard
        for shard in second_step.parameter_shards
        if shard.group_id == f"bucket_active_experts:0:{second_active_suffix}"
    )
    assert second_expert_shard.committed_version == 1
    assert second_expert_shard.last_touched_step == 1


def test_engine_plans_micro_batches_for_large_sample_batch() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    engine = FirstVersionTrainingEngine(cfg)

    step = engine.plan_step(
        step_index=0,
        batch=BatchShape(samples=5, tokens_per_sample=128),
    )

    assert [micro_batch.samples for micro_batch in step.scheduled_micro_batches] == [2, 2, 1]
    assert step.stream_overlap_summary is not None
    assert step.stream_overlap_summary.micro_batch_count == 3
    assert step.stream_overlap_summary.scheduled_samples == 5
    assert step.stream_overlap_summary.compute_operation_count == 30
    assert step.stream_overlap_summary.transfer_operation_count == 30
    assert all(trace.micro_batch_count == 3 for trace in step.bucket_stream_traces)
    assert all(trace.optimizer_update_count == 6 for trace in step.bucket_stream_traces)
    assert all(trace.gradient_release_count == 6 for trace in step.bucket_stream_traces)
    assert len(step.stream_operations) == 60
    assert step.stream_operations[-1].micro_batch_id == 2
    assert step.stream_operations[-1].bucket_id == 9


def test_engine_adapts_micro_batches_to_token_budget() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.execution.sample_parallelism = 4
    cfg.execution.max_tokens_per_micro_batch = 256
    engine = FirstVersionTrainingEngine(cfg)

    step = engine.plan_step(
        step_index=0,
        batch=BatchShape(samples=5, tokens_per_sample=192),
    )

    assert [micro_batch.samples for micro_batch in step.scheduled_micro_batches] == [
        1,
        1,
        1,
        1,
        1,
    ]
    assert step.stream_overlap_summary is not None
    assert step.stream_overlap_summary.micro_batch_count == 5
    assert len(step.stream_operations) == 100


def test_engine_snapshot_round_trip_preserves_runtime_progress() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    batch = BatchShape(samples=2, tokens_per_sample=128)

    baseline_engine = FirstVersionTrainingEngine(cfg)
    baseline_engine.run_step(batch)
    expected_second_step = baseline_engine.run_step(batch)

    resumed_engine = FirstVersionTrainingEngine(cfg)
    resumed_engine.run_step(batch)
    snapshot = resumed_engine.snapshot_state()
    assert snapshot.parameter_store_shards[0].representative_params == 128
    assert len(snapshot.parameter_store_shards[0].parameter_values) == 128
    assert len(snapshot.parameter_store_shards[0].source_layout) > 0
    assert len(snapshot.transport_cached_files) > 0
    assert len(snapshot.transport_buffers) > 0

    restored_engine = FirstVersionTrainingEngine(cfg)
    restored_engine.load_state(snapshot)
    restored_second_step = restored_engine.run_step(batch)

    assert restored_second_step.step_index == expected_second_step.step_index
    assert restored_second_step.active_expert_ids == expected_second_step.active_expert_ids
    assert restored_second_step.prefetched_expert_ids == expected_second_step.prefetched_expert_ids
    assert restored_second_step.residency_ending_states == expected_second_step.residency_ending_states
    assert restored_second_step.warehouse_summary == expected_second_step.warehouse_summary
    assert restored_second_step.parameter_store_summary == expected_second_step.parameter_store_summary
    assert restored_second_step.parameter_source_summary == expected_second_step.parameter_source_summary
    assert restored_second_step.parameter_prefetch_summary == expected_second_step.parameter_prefetch_summary
    assert restored_second_step.parameter_load_summary == expected_second_step.parameter_load_summary
    assert restored_second_step.transport_summary == expected_second_step.transport_summary
    assert (
        restored_second_step.transport_execution_summary
        == expected_second_step.transport_execution_summary
    )
    assert restored_second_step.execution_summary == expected_second_step.execution_summary
    assert restored_second_step.optimizer_summary == expected_second_step.optimizer_summary
    assert restored_second_step.optimizer_updates == expected_second_step.optimizer_updates


def test_engine_snapshot_round_trip_preserves_sample_window_progress() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    cfg.expert_rotation.rotate_every_samples = 4
    batch = BatchShape(samples=2, tokens_per_sample=128)

    baseline_engine = FirstVersionTrainingEngine(cfg)
    baseline_engine.run_step(batch)
    expected_second_step = baseline_engine.run_step(batch)

    resumed_engine = FirstVersionTrainingEngine(cfg)
    resumed_engine.run_step(batch)
    snapshot = resumed_engine.snapshot_state()

    restored_engine = FirstVersionTrainingEngine(cfg)
    restored_engine.load_state(snapshot)
    restored_second_step = restored_engine.run_step(batch)

    assert snapshot.cumulative_samples_processed == 2
    assert snapshot.cumulative_tokens_processed == 256
    assert restored_second_step.active_expert_ids == expected_second_step.active_expert_ids
    assert (
        restored_second_step.prefetched_expert_ids
        == expected_second_step.prefetched_expert_ids
    )


def test_engine_retains_active_window_state_when_next_batch_is_known(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    engine = FirstVersionTrainingEngine(cfg)
    batch = BatchShape(samples=2, tokens_per_sample=128)
    active_expert_ids = tuple(range(cfg.expert_rotation.active_experts_per_step))
    retained_window = ExpertWindowPlan(
        selection_strategy="router_hotness",
        router_score_source="mock_same_window",
        active_expert_ids=active_expert_ids,
        prefetched_expert_ids=active_expert_ids[
            : cfg.expert_rotation.prefetch_active_overlap
        ],
    )

    def _same_window(
        self,
        *,
        step_index,
        batch,
        layer_buckets,
        next_batch,
        cumulative_samples_processed=0,
        cumulative_tokens_processed=0,
    ):
        return retained_window

    monkeypatch.setattr(ExpertRotationScheduler, "plan_window", _same_window)

    first_step = engine.run_step(batch, next_batch=batch)

    assert first_step.parameter_store_summary is not None
    assert first_step.optimizer_summary is not None
    assert first_step.parameter_store_summary.cpu_hot_shards > 0
    assert first_step.parameter_store_summary.cpu_hot_resident_bytes > 0
    assert first_step.optimizer_summary.cpu_hot_shards > 0
    assert any(
        trace.cpu_hot_shards_after_update == 1
        and trace.offloaded_shards_after_update == 1
        for trace in first_step.bucket_stream_traces
    )


def test_engine_retains_recent_boundary_window_when_next_batch_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    engine = FirstVersionTrainingEngine(cfg)
    batch = BatchShape(samples=2, tokens_per_sample=128)
    first_active_expert_ids = tuple(range(cfg.expert_rotation.active_experts_per_step))
    second_active_expert_ids = tuple(
        range(
            cfg.expert_rotation.active_experts_per_step,
            cfg.expert_rotation.active_experts_per_step * 2,
        )
    )

    def _rotating_window(
        self,
        *,
        step_index,
        batch,
        layer_buckets,
        next_batch,
        cumulative_samples_processed=0,
        cumulative_tokens_processed=0,
    ):
        active_expert_ids = (
            first_active_expert_ids if step_index % 2 == 0 else second_active_expert_ids
        )
        return ExpertWindowPlan(
            selection_strategy="router_hotness",
            router_score_source="mock_boundary_rotation",
            active_expert_ids=active_expert_ids,
            prefetched_expert_ids=active_expert_ids[
                : cfg.expert_rotation.prefetch_active_overlap
            ],
        )

    monkeypatch.setattr(ExpertRotationScheduler, "plan_window", _rotating_window)

    first_step = engine.run_step(batch, next_batch=batch)

    assert first_step.parameter_store_summary is not None
    assert first_step.optimizer_summary is not None
    assert first_step.parameter_store_summary.cpu_hot_shards == len(engine.layer_buckets)
    assert first_step.parameter_store_summary.cpu_hot_resident_bytes > 0
    assert first_step.optimizer_summary.cpu_hot_shards == len(engine.layer_buckets)
    assert len(engine._state.retired_window_group_ids) == len(engine.layer_buckets)
    assert all(
        group_id.startswith("bucket_active_experts:")
        for group_id in engine._state.retired_window_group_ids
    )


def test_engine_accumulates_retired_windows_until_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    engine = FirstVersionTrainingEngine(cfg)
    batch = BatchShape(samples=2, tokens_per_sample=128)
    first_active_expert_ids = tuple(range(cfg.expert_rotation.active_experts_per_step))
    second_active_expert_ids = tuple(
        range(
            cfg.expert_rotation.active_experts_per_step,
            cfg.expert_rotation.active_experts_per_step * 2,
        )
    )

    def _rotating_window(
        self,
        *,
        step_index,
        batch,
        layer_buckets,
        next_batch,
        cumulative_samples_processed=0,
        cumulative_tokens_processed=0,
    ):
        active_expert_ids = (
            first_active_expert_ids if step_index % 2 == 0 else second_active_expert_ids
        )
        return ExpertWindowPlan(
            selection_strategy="router_hotness",
            router_score_source="mock_accumulating_rotation",
            active_expert_ids=active_expert_ids,
            prefetched_expert_ids=active_expert_ids[
                : cfg.expert_rotation.prefetch_active_overlap
            ],
        )

    monkeypatch.setattr(ExpertRotationScheduler, "plan_window", _rotating_window)

    engine.run_step(batch, next_batch=batch)
    second_step = engine.run_step(batch, next_batch=batch)

    assert second_step.parameter_store_summary is not None
    assert second_step.optimizer_summary is not None
    assert second_step.parameter_store_summary.cpu_hot_shards == (
        len(engine.layer_buckets) * 2
    )
    assert second_step.optimizer_summary.cpu_hot_shards == (
        len(engine.layer_buckets) * 2
    )
    assert len(engine._state.retired_window_group_ids) == len(engine.layer_buckets) * 2


def test_engine_reclaims_retained_window_under_cpu_hot_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    engine = FirstVersionTrainingEngine(cfg)
    batch = BatchShape(samples=2, tokens_per_sample=128)
    active_expert_ids = tuple(range(cfg.expert_rotation.active_experts_per_step))
    retained_window = ExpertWindowPlan(
        selection_strategy="router_hotness",
        router_score_source="mock_same_window",
        active_expert_ids=active_expert_ids,
        prefetched_expert_ids=active_expert_ids[
            : cfg.expert_rotation.prefetch_active_overlap
        ],
    )

    def _same_window(
        self,
        *,
        step_index,
        batch,
        layer_buckets,
        next_batch,
        cumulative_samples_processed=0,
        cumulative_tokens_processed=0,
    ):
        return retained_window

    monkeypatch.setattr(ExpertRotationScheduler, "plan_window", _same_window)
    monkeypatch.setattr(
        ParameterShardStore,
        "cpu_hot_resident_bytes",
        lambda self: 1 << 60,
    )

    first_step = engine.run_step(batch, next_batch=batch)

    assert first_step.parameter_store_summary is not None
    assert first_step.optimizer_summary is not None
    assert first_step.parameter_store_summary.cpu_hot_shards == 0
    assert first_step.parameter_store_summary.cpu_hot_resident_bytes == 0
    assert first_step.optimizer_summary.cpu_hot_shards == 0


def test_engine_rejects_invalid_simulation_length() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")
    engine = FirstVersionTrainingEngine(cfg)

    with pytest.raises(ValueError, match="steps"):
        engine.simulate(steps=0, batch=BatchShape(samples=1, tokens_per_sample=16))
