"""Dedicated training profile for Qwen3.5-35B-A3B."""

from __future__ import annotations

from cfie_training.config import (
    BucketScheduleConfig,
    ExecutionConfig,
    ExpertRotationConfig,
    MemoryBudgetConfig,
    ModelSpecConfig,
    ModelSourceConfig,
    ModelTargets,
    OptimizerConfig,
    PredictorTrainerConfig,
    ResourcePolicyConfig,
    RuntimeQuantizationConfig,
    StateBytesConfig,
    TransportConfig,
    TrainingProjectConfig,
)

QWEN35_35B_A3B_PROFILE = "qwen35-35b-a3b"


# 构造 Qwen3.5-35B-A3B 的默认训练基座配置。
def build_qwen35_35b_a3b_config() -> TrainingProjectConfig:
    return TrainingProjectConfig(
        profile_name=QWEN35_35B_A3B_PROFILE,
        model_targets=ModelTargets(
            development_model="Qwen3.5-35B-A3B",
            target_model="Qwen3.5-122B-class-MoE",
            family="qwen3.5_moe",
            stage="development",
        ),
        model_spec=ModelSpecConfig(
            architecture="Qwen3_5MoeForConditionalGeneration",
            text_model_type="qwen3_5_moe_text",
            hidden_size=2048,
            num_hidden_layers=40,
            num_attention_heads=16,
            num_key_value_heads=2,
            num_experts=256,
            num_experts_per_tok=8,
            moe_intermediate_size=512,
            shared_expert_intermediate_size=512,
            full_attention_interval=4,
            max_position_embeddings=262144,
            mtp_num_hidden_layers=1,
            attention_pattern=(
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ),
            quantization="gptq",
            quant_bits=4,
            quant_group_size=128,
            quant_sym=True,
            quant_dynamic_exclusions=(
                ".*attn.*",
                ".*shared_expert.*",
                ".*mtp.*",
                ".*visual.*",
            ),
            total_params_billion=35.0,
        ),
        model_source=ModelSourceConfig(
            model_path=(
                "/home/gaojiamin/.cache/huggingface/hub/"
                "models--Qwen--Qwen3.5-35B-A3B/snapshots/"
                "ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
            ),
            index_filename="model.safetensors.index.json",
            use_local_weight_manifest=True,
        ),
        expert_rotation=ExpertRotationConfig(
            enabled=True,
            active_experts_per_step=8,
            rotate_every_steps=1,
            train_shared_expert_every_step=True,
            selection_strategy="router_hotness",
        ),
        bucket_schedule=BucketScheduleConfig(
            unit="expert",
            max_live_buckets=1,
            prefetch_buckets=1,
            host_gradient_buffer_scope="current_bucket_only",
            release_gradients_immediately=True,
            update_immediately_after_backward=True,
        ),
        execution=ExecutionConfig(
            optimizer_device="cpu",
            gradient_device="cpu",
            activation_policy="recompute",
            overlap_backward_and_update=True,
            compute_stream_name="backward_compute",
            transfer_stream_name="cpu_update_release",
            sample_parallelism=2,
        ),
        optimizer=OptimizerConfig(
            algorithm="adamw",
            learning_rate=1e-5,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            weight_decay=0.0,
            cpu_state_storage_dtype="fp8_e4m3fn",
            gradient_buffer_storage_dtype="fp8_e4m3fn",
            offload_state_after_update=True,
        ),
        resource_policy=ResourcePolicyConfig(
            gpu_is_scarcest_resource=True,
            prioritize_memory_over_throughput=True,
            allow_cpu_participation=True,
            weight_offload_backend="cpu+nvme",
        ),
        transport=TransportConfig(
            max_staged_file_cache_gb=8.0,
            reuse_staged_files_across_steps=True,
            eviction_policy="lru",
        ),
        runtime_quantization=RuntimeQuantizationConfig(
            enabled=True,
            method="gptq",
            bits=4,
            group_size=128,
            sym=True,
            compute_view_dtype="fp32",
            persist_fp32_to_nvme=True,
        ),
        predictor_trainer=PredictorTrainerConfig(
            input_summary_dim=64,
            hidden_dim=128,
            batch_size=8,
            epochs=4,
            learning_rate=1e-3,
            weight_decay=1e-4,
            examples_per_step=4,
            synthetic_trace_noise_scale=0.05,
            seed=0,
        ),
        memory_budget=MemoryBudgetConfig(
            gpu_hot_budget_gb=6.0,
            cpu_hot_budget_gb=24.0,
            nvme_cold_budget_gb=512.0,
            gpu_safety_margin_gb=1.0,
            cpu_safety_margin_gb=2.0,
            nvme_safety_margin_gb=16.0,
        ),
        state_bytes=StateBytesConfig(
            device_weight_bytes_per_param=2,
            gradient_bytes_per_param=2,
            master_weight_bytes_per_param=4,
            optimizer_state_bytes_per_param=8,
            activation_bytes_per_element=2,
            activation_residency_multiplier=2.0,
        ),
        notes=(
            "Dedicated training profile for Qwen3.5-35B-A3B.",
            "Model geometry is confirmed from the local HuggingFace config cache.",
            "Treat the 256 routed experts as a rotating pool; never keep them all trainable on device.",
            "Use per-bucket backward/update/release with CPU-side optimizer math as the default path.",
            "Compress CPU-side AdamW state and bucket gradient buffers with FP8 storage where the training path remains stable.",
            "Reserve host gradient memory only for the current bucket-sized ingress/update window.",
            "Align layer buckets with the Qwen3.5 4-layer attention cadence: 3 linear-attention layers plus 1 full-attention layer.",
        ),
    )
