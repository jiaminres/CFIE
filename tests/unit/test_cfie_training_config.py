"""Unit tests for the standalone CFIE training package config."""

from __future__ import annotations

import pytest

from cfie_training.config import TrainingProjectConfig


def test_training_project_defaults_are_resource_first() -> None:
    cfg = TrainingProjectConfig()

    assert cfg.package_name == "cfie_training"
    assert cfg.profile_name == "generic"
    assert cfg.model_spec.architecture == ""
    assert cfg.model_source.index_filename == "model.safetensors.index.json"
    assert cfg.expert_rotation.selection_strategy == "round_robin"
    assert cfg.expert_rotation.rotate_every_samples == 0
    assert cfg.expert_rotation.rotate_every_tokens == 0
    assert cfg.expert_rotation.retain_active_window_state_in_memory is True
    assert cfg.expert_rotation.cross_step_hotness_decay == 0.35
    assert cfg.expert_rotation.next_step_score_weight == 0.7
    assert cfg.expert_rotation.prefetch_active_overlap == 2
    assert cfg.bucket_schedule.release_gradients_immediately is True
    assert cfg.bucket_schedule.host_gradient_buffer_scope == "current_bucket_only"
    assert cfg.bucket_schedule.include_mtp_dedicated_bucket is False
    assert cfg.execution.compute_device == "cpu"
    assert cfg.execution.optimizer_device == "cpu"
    assert cfg.execution.trainable_shard_materialization == "representative"
    assert cfg.execution.activation_policy == "minimal_cache"
    assert cfg.execution.max_tokens_per_micro_batch == 256
    assert cfg.optimizer.algorithm == "adamw"
    assert cfg.optimizer.cpu_state_storage_dtype == "fp8_e4m3fn"
    assert cfg.optimizer.gradient_buffer_storage_dtype == "fp8_e4m3fn"
    assert cfg.optimizer.offload_state_after_update is True
    assert cfg.resource_policy.weight_offload_backend == "cpu+nvme"
    assert cfg.transport.max_staged_file_cache_gb == 2.0
    assert cfg.transport.reuse_staged_files_across_steps is True
    assert cfg.runtime_quantization.enabled is False
    assert cfg.runtime_quantization.session_id
    assert cfg.predictor_routing.enabled is True
    assert cfg.predictor_routing.window_layers == 8
    assert cfg.predictor_routing.shared_gpu_candidate_slots == 256
    assert cfg.predictor_routing.speculative_experts_per_layer == 32
    assert cfg.predictor_routing.candidate_experts_per_layer == 40
    assert cfg.predictor_routing.executed_experts_per_layer == 8
    assert cfg.predictor_routing.selection_mode == "masked_candidate_topk"
    assert cfg.predictor_routing.online_expert_source == "cpu_hot_only"
    assert cfg.predictor_trainer.input_summary_dim == 64
    assert cfg.predictor_trainer.hidden_dim == 128
    assert cfg.predictor_trainer.model_architecture == "mlp"
    assert cfg.predictor_trainer.model_depth == 2
    assert cfg.predictor_trainer.model_dropout == 0.0
    assert cfg.predictor_trainer.model_num_heads == 8
    assert cfg.predictor_trainer.model_memory_tokens == 8
    assert cfg.predictor_trainer.model_ffn_multiplier == 4
    assert cfg.predictor_trainer.batch_size == 8
    assert cfg.predictor_trainer.epochs == 4
    assert cfg.predictor_trainer.examples_per_step == 4


def test_training_project_from_dict_accepts_nested_overrides() -> None:
    cfg = TrainingProjectConfig.from_dict({
        "profile_name": "custom",
        "model_spec": {
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "text_model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "full_attention_interval": 4,
            "max_position_embeddings": 262144,
            "mtp_num_hidden_layers": 1,
            "attention_pattern": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        },
        "model_source": {
            "model_path": "/tmp/model",
        },
        "expert_rotation": {
            "active_experts_per_step": 8,
            "rotate_every_samples": 128,
            "retain_active_window_state_in_memory": False,
            "cross_step_hotness_decay": 0.2,
            "next_step_score_weight": 0.85,
            "prefetch_active_overlap": 3,
        },
        "bucket_schedule": {
            "host_gradient_buffer_scope": "full_model",
            "include_mtp_dedicated_bucket": True,
        },
        "execution": {
            "compute_device": "gpu",
            "sample_parallelism": 3,
            "max_tokens_per_micro_batch": 384,
        },
        "optimizer": {
            "learning_rate": 2e-5,
            "cpu_state_storage_dtype": "bf16",
        },
        "transport": {
            "max_staged_file_cache_gb": 1.5,
        },
        "predictor_routing": {
            "window_layers": 8,
            "stride_layers": 4,
            "shared_gpu_candidate_slots": 128,
            "executed_experts_per_layer": 8,
            "candidate_experts_per_layer": 24,
            "selection_mode": "shadow_exact",
        },
        "predictor_trainer": {
            "input_summary_dim": 32,
            "hidden_dim": 64,
            "model_architecture": "query_transformer",
            "model_depth": 3,
            "model_dropout": 0.1,
            "model_num_heads": 8,
            "model_memory_tokens": 4,
            "model_ffn_multiplier": 3,
            "batch_size": 4,
            "epochs": 2,
            "examples_per_step": 3,
        },
    })

    assert cfg.profile_name == "custom"
    assert cfg.model_spec.hidden_size == 2048
    assert cfg.model_source.model_path == "/tmp/model"
    assert cfg.expert_rotation.active_experts_per_step == 8
    assert cfg.expert_rotation.rotate_every_samples == 128
    assert cfg.expert_rotation.rotate_every_tokens == 0
    assert cfg.expert_rotation.retain_active_window_state_in_memory is False
    assert cfg.expert_rotation.selection_strategy == "round_robin"
    assert cfg.expert_rotation.cross_step_hotness_decay == 0.2
    assert cfg.expert_rotation.next_step_score_weight == 0.85
    assert cfg.expert_rotation.prefetch_active_overlap == 3
    assert cfg.bucket_schedule.host_gradient_buffer_scope == "full_model"
    assert cfg.bucket_schedule.include_mtp_dedicated_bucket is True
    assert cfg.execution.compute_device == "gpu"
    assert cfg.execution.trainable_shard_materialization == "representative"
    assert cfg.execution.sample_parallelism == 3
    assert cfg.execution.max_tokens_per_micro_batch == 384
    assert cfg.optimizer.learning_rate == 2e-5
    assert cfg.optimizer.cpu_state_storage_dtype == "bf16"
    assert cfg.transport.max_staged_file_cache_gb == 1.5
    assert cfg.runtime_quantization.enabled is False
    assert cfg.predictor_routing.stride_layers == 4
    assert cfg.predictor_routing.shared_gpu_candidate_slots == 128
    assert cfg.predictor_routing.speculative_experts_per_layer == 16
    assert cfg.predictor_routing.candidate_experts_per_layer == 24
    assert cfg.predictor_routing.selection_mode == "shadow_exact"
    assert cfg.predictor_trainer.input_summary_dim == 32
    assert cfg.predictor_trainer.hidden_dim == 64
    assert cfg.predictor_trainer.model_architecture == "query_transformer"
    assert cfg.predictor_trainer.model_depth == 3
    assert cfg.predictor_trainer.model_dropout == 0.1
    assert cfg.predictor_trainer.model_num_heads == 8
    assert cfg.predictor_trainer.model_memory_tokens == 4
    assert cfg.predictor_trainer.model_ffn_multiplier == 3
    assert cfg.predictor_trainer.batch_size == 4
    assert cfg.predictor_trainer.epochs == 2
    assert cfg.predictor_trainer.examples_per_step == 3


def test_training_project_from_dict_ignores_legacy_model_targets() -> None:
    cfg = TrainingProjectConfig.from_dict({
        "profile_name": "custom",
        "model_targets": {
            "development_model": "Qwen3.5-35B-A3B-GPTQ",
            "target_model": "Qwen3.5-122B-class-MoE",
        },
    })

    assert cfg.profile_name == "custom"
    assert "model_targets" not in cfg.to_dict()


def test_invalid_expert_rotation_raises_value_error() -> None:
    with pytest.raises(ValueError, match="active_experts_per_step"):
        TrainingProjectConfig.from_dict({
            "expert_rotation": {
                "active_experts_per_step": 0,
            },
        })


def test_invalid_prefetch_active_overlap_raises_value_error() -> None:
    with pytest.raises(ValueError, match="prefetch_active_overlap"):
        TrainingProjectConfig.from_dict({
            "expert_rotation": {
                "active_experts_per_step": 4,
                "prefetch_active_overlap": 5,
            },
        })


def test_invalid_dual_rotation_budgets_raise_value_error() -> None:
    with pytest.raises(
        ValueError,
        match="rotate_every_samples and rotate_every_tokens",
    ):
        TrainingProjectConfig.from_dict({
            "expert_rotation": {
                "rotate_every_samples": 64,
                "rotate_every_tokens": 4096,
            },
        })


def test_invalid_next_step_score_weight_raises_value_error() -> None:
    with pytest.raises(ValueError, match="next_step_score_weight"):
        TrainingProjectConfig.from_dict({
            "expert_rotation": {
                "next_step_score_weight": 1.5,
            },
        })


def test_invalid_predictor_candidate_budget_raises_value_error() -> None:
    with pytest.raises(ValueError, match="candidate_experts_per_layer"):
        TrainingProjectConfig.from_dict({
            "predictor_routing": {
                "window_layers": 8,
                "shared_gpu_candidate_slots": 256,
                "executed_experts_per_layer": 8,
                "candidate_experts_per_layer": 39,
            },
        })


def test_invalid_compute_device_raises_value_error() -> None:
    with pytest.raises(ValueError, match="compute_device"):
        TrainingProjectConfig.from_dict({
            "execution": {
                "compute_device": "tpu",
            },
        })


def test_invalid_trainable_shard_materialization_raises_value_error() -> None:
    with pytest.raises(ValueError, match="trainable_shard_materialization"):
        TrainingProjectConfig.from_dict({
            "execution": {
                "trainable_shard_materialization": "full",
            },
        })


def test_trainable_shard_materialization_accepts_logical_mode() -> None:
    cfg = TrainingProjectConfig.from_dict({
        "execution": {
            "trainable_shard_materialization": "logical",
        },
    })

    assert cfg.execution.trainable_shard_materialization == "logical"


def test_logical_cuda_execution_mode_accepts_full_bucket() -> None:
    cfg = TrainingProjectConfig.from_dict({
        "execution": {
            "logical_cuda_execution_mode": "full_bucket",
        },
    })

    assert cfg.execution.logical_cuda_execution_mode == "full_bucket"


def test_invalid_logical_cuda_execution_mode_raises_value_error() -> None:
    with pytest.raises(ValueError, match="logical_cuda_execution_mode"):
        TrainingProjectConfig.from_dict({
            "execution": {
                "logical_cuda_execution_mode": "full_model",
            },
        })


def test_gptq_model_spec_auto_enables_runtime_quantization() -> None:
    cfg = TrainingProjectConfig.from_dict({
        "model_spec": {
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "text_model_type": "qwen3_5_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "full_attention_interval": 4,
            "max_position_embeddings": 262144,
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
        },
    })

    assert cfg.runtime_quantization.enabled is True
    assert cfg.runtime_quantization.bits == 4
    assert cfg.runtime_quantization.group_size == 128
    assert cfg.runtime_quantization.sym is True


def test_invalid_predictor_trainer_batch_size_raises_value_error() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        TrainingProjectConfig.from_dict({
            "predictor_trainer": {
                "batch_size": 0,
            },
        })
