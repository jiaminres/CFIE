"""Unit tests for the CFIE training memory planner."""

from __future__ import annotations

import math

from cfie_training.config import TrainingProjectConfig
from cfie_training.profiles import build_profile_config
from cfie_training.runtime.memory import TrainingMemoryPlanner, TrainingStartupEstimator
from cfie_training.runtime.quantization import runtime_device_weight_bytes_per_param
from cfie_training.runtime.types import BatchShape


def test_training_memory_plan_tracks_three_tiers() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")

    plan = TrainingMemoryPlanner(cfg).build(BatchShape(samples=2, tokens_per_sample=256))

    assert plan.bucket_count == 10
    assert plan.mtp_bucket_count == 0
    assert plan.max_layers_per_bucket == 4
    assert plan.total_params == 35_000_000_000
    assert plan.params_per_routed_expert == 3 * 2048 * 512
    assert plan.static_params_total > 0
    assert (
        plan.static_params_total + plan.bucket_non_routed_params_total
        == plan.non_routed_params_total
    )
    assert len(plan.bucket_non_routed_params_by_bucket) == plan.bucket_count
    assert sum(plan.bucket_non_routed_params_by_bucket) == (
        plan.bucket_non_routed_params_total
    )
    assert max(plan.bucket_non_routed_params_by_bucket) - min(
        plan.bucket_non_routed_params_by_bucket
    ) <= 1
    assert plan.params_per_bucket_active_routed == 4 * 8 * (3 * 2048 * 512)
    assert len(plan.bucket_active_routed_params_by_bucket) == plan.bucket_count
    assert plan.bucket_active_routed_params_by_bucket == (
        plan.params_per_bucket_active_routed,
    ) * plan.bucket_count
    assert plan.cpu_optimizer_state_storage_dtype == "fp8_e4m3fn"
    assert plan.host_gradient_buffer_storage_dtype == "fp8_e4m3fn"
    assert plan.host_gradient_buffer_scope == "current_bucket_only"
    assert plan.host_gradient_buffer_bytes_per_param == 1
    assert plan.host_gradient_buffer_bytes == (
        plan.params_per_bucket_non_routed + plan.params_per_bucket_active_routed
    )
    assert plan.full_model_gradient_buffer_bytes > plan.host_gradient_buffer_bytes
    assert plan.transport_staged_file_cache_bytes == int(
        cfg.transport.max_staged_file_cache_gb * 1024**3
    )
    assert plan.weight_stage_buffer_bytes == math.ceil(
        (
            plan.params_per_bucket_non_routed + plan.params_per_bucket_active_routed
        ) * runtime_device_weight_bytes_per_param(cfg)
    )
    assert plan.transfer_overlap_enabled is True
    assert plan.transfer_staging_buffer_bytes == (
        plan.host_gradient_buffer_bytes + plan.weight_stage_buffer_bytes
    )
    assert plan.cpu_hot.resident_bytes > (
        plan.static_params_total
        * (
            cfg.state_bytes.master_weight_bytes_per_param
            + plan.cpu_optimizer_state_bytes_per_param
        )
    )
    assert plan.cpu_hot.resident_bytes > (
        plan.transport_staged_file_cache_bytes
        + plan.transfer_staging_buffer_bytes
    )
    assert plan.gpu_hot.resident_bytes < plan.cpu_hot.resident_bytes
    assert plan.nvme_cold.resident_bytes > plan.cpu_hot.resident_bytes
    assert plan.gpu_hot.within_budget is True
    assert plan.cpu_hot.within_budget is True
    assert plan.nvme_cold.within_budget is True
    assert plan.all_tiers_within_budget is True


def test_training_memory_plan_tracks_hybrid_and_mtp_buckets() -> None:
    cfg = TrainingProjectConfig.from_dict({
        "profile_name": "hybrid-mtp-memory",
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
        "expert_rotation": {
            "active_experts_per_step": 4,
        },
        "bucket_schedule": {
            "unit": "hybrid",
            "include_mtp_dedicated_bucket": True,
            "prefetch_buckets": 2,
        },
    })

    plan = TrainingMemoryPlanner(cfg).build(BatchShape(samples=1, tokens_per_sample=64))

    assert plan.bucket_count == 5
    assert plan.mtp_bucket_count == 1
    assert plan.max_layers_per_bucket == 2
    assert len(plan.bucket_non_routed_params_by_bucket) == 5
    assert len(plan.bucket_active_routed_params_by_bucket) == 5
    assert sum(plan.bucket_non_routed_params_by_bucket) == (
        plan.bucket_non_routed_params_total
    )
    assert plan.bucket_active_routed_params_by_bucket[-1] == (
        1 * cfg.expert_rotation.active_experts_per_step * plan.params_per_routed_expert
    )
    assert plan.params_per_bucket_active_routed == (
        2 * cfg.expert_rotation.active_experts_per_step * plan.params_per_routed_expert
    )
    assert plan.params_per_bucket_prefetched_routed == (
        2 * plan.params_per_bucket_active_routed
    )


def test_training_startup_estimator_prefers_fuller_gpu_fit_for_larger_budget() -> None:
    cfg = TrainingProjectConfig.from_dict({
        "profile_name": "startup-estimate-smoke",
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
            "quantization": "gptq",
            "quant_bits": 4,
            "quant_group_size": 128,
            "quant_sym": True,
            "total_params_billion": 0.2,
        },
        "expert_rotation": {
            "active_experts_per_step": 4,
        },
        "bucket_schedule": {
            "max_live_buckets": 1,
            "prefetch_buckets": 1,
        },
        "memory_budget": {
            "gpu_hot_budget_gb": 2.0,
            "cpu_hot_budget_gb": 32.0,
            "nvme_cold_budget_gb": 128.0,
            "gpu_safety_margin_gb": 0.5,
            "cpu_safety_margin_gb": 1.0,
            "nvme_safety_margin_gb": 1.0,
        },
    })
    batch = BatchShape(samples=1, tokens_per_sample=128)

    estimates = TrainingStartupEstimator(cfg).estimate(
        batch=batch,
        gpu_hot_budget_candidates_gb=(2.0, 4.0),
        active_expert_candidates=(4, 8, 12),
        max_live_bucket_candidates=(1, 2),
        prefetch_bucket_candidates=(0, 1),
    )

    assert len(estimates) == 2
    smaller_budget, larger_budget = estimates
    assert smaller_budget.gpu_hot_budget_gb == 2.0
    assert larger_budget.gpu_hot_budget_gb == 4.0
    assert smaller_budget.fits_within_budget is True
    assert larger_budget.fits_within_budget is True
    assert larger_budget.planned_gpu_hot_bytes >= smaller_budget.planned_gpu_hot_bytes
    assert larger_budget.active_experts_per_step >= smaller_budget.active_experts_per_step
    assert 0.0 < smaller_budget.gpu_fill_ratio <= 1.0
    assert 0.0 < larger_budget.gpu_fill_ratio <= 1.0


def test_training_memory_plan_accounts_for_retired_window_hot_state() -> None:
    retained_cfg = TrainingProjectConfig.from_dict({
        "profile_name": "memory-retired-window-retained",
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
            "quantization": "gptq",
            "quant_bits": 4,
            "quant_group_size": 128,
            "quant_sym": True,
            "total_params_billion": 0.2,
        },
        "expert_rotation": {
            "active_experts_per_step": 4,
            "retain_active_window_state_in_memory": True,
        },
        "execution": {
            "compute_device": "gpu",
        },
        "optimizer": {
            "offload_state_after_update": True,
        },
        "runtime_quantization": {
            "enabled": True,
            "method": "gptq",
            "bits": 4,
        },
    })
    non_retained_cfg = TrainingProjectConfig.from_dict({
        **retained_cfg.to_dict(),
        "profile_name": "memory-retired-window-disabled",
        "expert_rotation": {
            **retained_cfg.to_dict()["expert_rotation"],
            "retain_active_window_state_in_memory": False,
        },
    })
    batch = BatchShape(samples=1, tokens_per_sample=128)

    retained_plan = TrainingMemoryPlanner(retained_cfg).build(batch)
    non_retained_plan = TrainingMemoryPlanner(non_retained_cfg).build(batch)

    assert retained_plan.cpu_hot.resident_bytes > non_retained_plan.cpu_hot.resident_bytes
    assert retained_plan.gpu_hot.resident_bytes > non_retained_plan.gpu_hot.resident_bytes
    assert retained_plan.nvme_cold.resident_bytes < non_retained_plan.nvme_cold.resident_bytes
