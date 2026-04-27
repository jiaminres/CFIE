"""Unit tests for dedicated CFIE training profiles."""

from __future__ import annotations

from cfie_training.profiles import SUPPORTED_TRAINING_PROFILES, build_profile_config


def _normalized_path(path: str) -> str:
    return path.replace("\\", "/")


def test_supported_training_profiles_include_qwen35_122b() -> None:
    assert "qwen35-35b-a3b" in SUPPORTED_TRAINING_PROFILES
    assert "qwen35-122b-a10b" in SUPPORTED_TRAINING_PROFILES
    assert "generic" in SUPPORTED_TRAINING_PROFILES


def test_qwen35_35b_profile_uses_confirmed_local_model_geometry() -> None:
    cfg = build_profile_config("qwen35-35b-a3b")

    assert cfg.profile_name == "qwen35-35b-a3b"
    assert cfg.model_spec.architecture == "Qwen3_5MoeForConditionalGeneration"
    assert cfg.model_spec.hidden_size == 2048
    assert cfg.model_spec.num_hidden_layers == 40
    assert cfg.model_spec.num_attention_heads == 16
    assert cfg.model_spec.num_key_value_heads == 2
    assert cfg.model_spec.num_experts == 256
    assert cfg.model_spec.num_experts_per_tok == 8
    assert cfg.model_spec.moe_intermediate_size == 512
    assert cfg.model_spec.shared_expert_intermediate_size == 512
    assert cfg.model_spec.total_params_billion == 35.0
    assert _normalized_path(cfg.model_source.model_path).endswith(
        "models--Qwen--Qwen3.5-35B-A3B-GPTQ-Int4/snapshots/"
        "33f4e5e615e1f29a7b218906555ea6fe2d09c741"
    )
    assert cfg.bucket_schedule.unit == "expert"
    assert cfg.bucket_schedule.host_gradient_buffer_scope == "current_bucket_only"
    assert cfg.bucket_schedule.include_mtp_dedicated_bucket is False
    assert cfg.expert_rotation.selection_strategy == "router_hotness"
    assert cfg.expert_rotation.cross_step_hotness_decay == 0.35
    assert cfg.expert_rotation.next_step_score_weight == 0.7
    assert cfg.expert_rotation.prefetch_active_overlap == 2
    assert cfg.execution.compute_device == "cpu"
    assert cfg.execution.activation_policy == "recompute"
    assert cfg.execution.max_tokens_per_micro_batch == 256
    assert cfg.optimizer.algorithm == "adamw"
    assert cfg.optimizer.cpu_state_storage_dtype == "fp8_e4m3fn"
    assert cfg.optimizer.gradient_buffer_storage_dtype == "fp8_e4m3fn"
    assert cfg.optimizer.offload_state_after_update is True
    assert cfg.transport.max_staged_file_cache_gb == 8.0
    assert cfg.predictor_routing.window_layers == 8
    assert cfg.predictor_routing.shared_gpu_candidate_slots == 256
    assert cfg.predictor_routing.candidate_experts_per_layer == 40
    assert cfg.predictor_routing.executed_experts_per_layer == 8
    assert cfg.predictor_trainer.input_summary_dim == 64
    assert cfg.predictor_trainer.hidden_dim == 128
    assert cfg.predictor_trainer.examples_per_step == 4
    assert cfg.memory_budget.gpu_hot_budget_gb == 6.0


def test_qwen35_122b_profile_uses_confirmed_local_model_geometry() -> None:
    cfg = build_profile_config("qwen35-122b-a10b")

    assert cfg.profile_name == "qwen35-122b-a10b"
    assert cfg.model_spec.architecture == "Qwen3_5MoeForConditionalGeneration"
    assert cfg.model_spec.hidden_size == 3072
    assert cfg.model_spec.num_hidden_layers == 48
    assert cfg.model_spec.num_attention_heads == 32
    assert cfg.model_spec.num_key_value_heads == 2
    assert cfg.model_spec.num_experts == 256
    assert cfg.model_spec.num_experts_per_tok == 8
    assert cfg.model_spec.moe_intermediate_size == 1024
    assert cfg.model_spec.shared_expert_intermediate_size == 1024
    assert cfg.model_spec.total_params_billion == 122.0
    assert _normalized_path(cfg.model_source.model_path).endswith(
        "models--Qwen--Qwen3.5-122B-A10B-GPTQ-Int4/snapshots/"
        "5b9f0050d3ec98b0c81a7716776533c5eacebb64"
    )
    assert cfg.bucket_schedule.unit == "expert"
    assert cfg.bucket_schedule.host_gradient_buffer_scope == "current_bucket_only"
    assert cfg.expert_rotation.selection_strategy == "router_hotness"
    assert cfg.execution.compute_device == "cpu"
    assert cfg.execution.activation_policy == "recompute"
    assert cfg.transport.max_staged_file_cache_gb == 16.0
    assert cfg.predictor_routing.window_layers == 8
    assert cfg.predictor_routing.stride_layers == 4
    assert cfg.predictor_routing.candidate_experts_per_layer == 40
    assert cfg.predictor_routing.executed_experts_per_layer == 8
    assert cfg.predictor_trainer.model_architecture == "factorized"
    assert cfg.predictor_trainer.hidden_dim == 512
    assert cfg.predictor_trainer.model_depth == 4
    assert cfg.predictor_trainer.batch_size == 4096
    assert cfg.predictor_trainer.epochs == 10
    assert cfg.memory_budget.gpu_hot_budget_gb == 10.0
