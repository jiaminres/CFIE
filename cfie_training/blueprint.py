"""Blueprint helpers for the standalone CFIE training project."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from cfie_training.config import TrainingProjectConfig


@dataclass(slots=True)
class TrainingPhase:
    name: str
    owner: str
    goal: str
    actions: tuple[str, ...]

    # 将训练阶段对象序列化为字典，便于输出到 JSON / 文本蓝图。
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "owner": self.owner,
            "goal": self.goal,
            "actions": list(self.actions),
        }


@dataclass(slots=True)
class TrainingBlueprint:
    package_name: str
    profile_name: str
    summary: str
    invariants: tuple[str, ...]
    phases: tuple[TrainingPhase, ...]
    metadata: dict[str, Any]

    # 将完整训练蓝图序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        return {
            "package_name": self.package_name,
            "profile_name": self.profile_name,
            "summary": self.summary,
            "invariants": list(self.invariants),
            "phases": [phase.to_dict() for phase in self.phases],
            "metadata": self.metadata,
        }

    # 将训练蓝图导出为稳定排序的 JSON 文本。
    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    # 将训练蓝图渲染为便于阅读的纯文本摘要。
    def render_text(self) -> str:
        # -----------------
        # 先写入蓝图头部与全局约束。
        lines = [
            f"Package: {self.package_name}",
            f"Profile: {self.profile_name}",
            self.summary,
            "",
            "Invariants:",
        ]
        lines.extend(f"- {item}" for item in self.invariants)

        # -----------------
        # 如存在额外元数据，则追加到单独小节。
        if self.metadata:
            lines.append("")
            lines.append("Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"- {key}: {value}")

        # -----------------
        # 最后追加阶段列表与每阶段动作。
        lines.append("")
        lines.append("Phases:")
        for phase in self.phases:
            lines.append(f"- {phase.name} [{phase.owner}]: {phase.goal}")
            lines.extend(f"  * {action}" for action in phase.actions)
        return "\n".join(lines)


# 按训练配置生成整包训练蓝图。
def build_training_blueprint(config: TrainingProjectConfig) -> TrainingBlueprint:
    # -----------------
    # 先校验配置，并判断是否命中 Qwen3.5 专用 profile。
    config.validate()
    model_spec = config.model_spec
    qwen35_profile = config.profile_name == "qwen35-35b-a3b" and model_spec.is_defined()

    # -----------------
    # 对已知 Qwen3.5 profile 生成更细化的专用蓝图。
    if qwen35_profile:
        bucket_window_count = (
            model_spec.num_hidden_layers // model_spec.full_attention_interval
        )
        summary = (
            "Qwen3.5-35B-A3B resource-first training blueprint. "
            f"Development model: {config.model_targets.development_model}. "
            f"Design target: {config.model_targets.target_model}."
        )
        invariants = (
            (
                f"Schedule the {model_spec.num_hidden_layers} text layers as "
                f"{bucket_window_count} windows of {model_spec.full_attention_interval} layers."
            ),
            (
                "Each layer window should follow the native Qwen3.5 cadence: "
                + " -> ".join(model_spec.attention_pattern)
            ),
            (
                f"Treat the {model_spec.num_experts} routed experts as a rotating pool "
                f"and keep at most {config.expert_rotation.active_experts_per_step} "
                "experts trainable in the active step window."
            ),
            (
                "Finish backward, optimizer update, and gradient release at the bucket boundary "
                "instead of waiting for a full-model optimizer step."
            ),
            (
                f"Keep CPU-side optimizer/update work as default and use "
                f"{config.execution.transfer_stream_name} to overlap update/release with "
                f"{config.execution.compute_stream_name}."
            ),
            (
                "Treat CPU-side AdamW master/state/gradient staging as a memory-compression target "
                "and explicitly evaluate FP8-class "
                f"({config.optimizer.cpu_state_storage_dtype}) compressed buffers whenever "
                "numerical stability permits."
            ),
            (
                "Because gradients refresh host-side master parameters bucket by bucket, host gradient "
                "residency should be bounded to the current bucket-sized ingress/update buffer instead "
                "of a full-model gradient allocation."
            ),
            (
                f"Preserve the shared expert outside routed-expert rotation; its intermediate size is "
                f"{model_spec.shared_expert_intermediate_size}."
            ),
        )
        phases = (
            TrainingPhase(
                name="stage_static_modules",
                owner="host_io",
                goal="Keep embeddings, router, norms, and shared expert state available while routed experts rotate.",
                actions=(
                    "pin static modules in CPU memory or the smallest safe GPU residency set",
                    "avoid pulling the full routed-expert pool onto the device",
                    "treat the MTP branch as an optional independent training slice",
                ),
            ),
            TrainingPhase(
                name="prefetch_routed_expert_window",
                owner="host_io",
                goal="Load only the routed experts selected for the current training step.",
                actions=(
                    f"choose up to {config.expert_rotation.active_experts_per_step} active experts from {model_spec.num_experts}",
                    f"prefetch exactly {config.bucket_schedule.prefetch_buckets} future expert bucket(s)",
                    f"stage expert weights via {config.resource_policy.weight_offload_backend}",
                ),
            ),
            TrainingPhase(
                name="forward_qwen35_bucket",
                owner=config.execution.compute_stream_name,
                goal="Run one Qwen3.5 bucket with the native attention cadence and minimal activation residency.",
                actions=(
                    f"process {model_spec.full_attention_interval} layers per bucket",
                    "follow the 3x linear-attention + 1x full-attention rhythm",
                    "use recompute rather than persistent activation caches",
                ),
            ),
            TrainingPhase(
                name="streaming_backward_update",
                owner=config.execution.compute_stream_name,
                goal="Complete backward for the bucket and hand gradients off immediately.",
                actions=(
                    "recompute the current bucket as needed during backward",
                    "emit gradients bucket-by-bucket instead of holding them for the whole model",
                    (
                        "flush gradients into a host-side buffer sized according to "
                        f"{config.bucket_schedule.host_gradient_buffer_scope}"
                    ),
                    "hand gradients to the CPU update path as soon as the bucket finishes",
                ),
            ),
            TrainingPhase(
                name="cpu_update_release",
                owner=config.execution.transfer_stream_name,
                goal="Update parameters on CPU and free GPU state immediately after each bucket.",
                actions=(
                    f"run optimizer math on {config.execution.optimizer_device}",
                    (
                        "evaluate "
                        f"{config.optimizer.cpu_state_storage_dtype}-compressed CPU master/state "
                        f"buffers and {config.optimizer.gradient_buffer_storage_dtype}-compressed "
                        "gradient staging where numerically safe"
                    ),
                    (
                        "keep host gradient buffering capped by "
                        f"{config.bucket_schedule.host_gradient_buffer_scope}"
                    ),
                    f"release gradients immediately after update: {config.bucket_schedule.release_gradients_immediately}",
                    "evict stale expert shards before the next bucket becomes active",
                ),
            ),
            TrainingPhase(
                name="rotate_expert_window",
                owner="scheduler",
                goal="Advance the routed-expert training window for the next step.",
                actions=(
                    f"rotate every {config.expert_rotation.rotate_every_steps} step(s)",
                    "keep shared expert handling separate from routed-expert rotation",
                    f"balance sample parallelism at {config.execution.sample_parallelism} to avoid GPU idle time",
                ),
            ),
        )
        metadata = {
            "development_model": config.model_targets.development_model,
            "target_model": config.model_targets.target_model,
            "architecture": model_spec.architecture,
            "text_model_type": model_spec.text_model_type,
            "hidden_size": model_spec.hidden_size,
            "num_hidden_layers": model_spec.num_hidden_layers,
            "num_experts": model_spec.num_experts,
            "num_experts_per_tok": model_spec.num_experts_per_tok,
            "moe_intermediate_size": model_spec.moe_intermediate_size,
            "shared_expert_intermediate_size": model_spec.shared_expert_intermediate_size,
            "full_attention_interval": model_spec.full_attention_interval,
            "attention_pattern": " -> ".join(model_spec.attention_pattern),
            "quantization_reference": (
                f"{model_spec.quantization}:{model_spec.quant_bits}bit/"
                f"group{model_spec.quant_group_size}"
            ),
            "optimizer_algorithm": config.optimizer.algorithm,
            "optimizer_learning_rate": config.optimizer.learning_rate,
            "cpu_optimizer_state_storage_dtype": (
                config.optimizer.cpu_state_storage_dtype
            ),
            "gradient_buffer_storage_dtype": (
                config.optimizer.gradient_buffer_storage_dtype
            ),
            "host_gradient_buffer_scope": (
                config.bucket_schedule.host_gradient_buffer_scope
            ),
        }
        return TrainingBlueprint(
            package_name=config.package_name,
            profile_name=config.profile_name,
            summary=summary,
            invariants=invariants,
            phases=phases,
            metadata=metadata,
        )

    # -----------------
    # 其余 profile 走通用训练蓝图。
    invariants = (
        "Only a subset of experts should be trainable per step.",
        "Each bucket should finish backward, update, and gradient release before the next bucket stays resident.",
        f"Optimizer and low-density update work should default to {config.execution.optimizer_device}.",
        (
            "CPU-side optimizer master/state/gradient staging should be considered for FP8 compression "
            "when the numerical path remains stable."
        ),
        (
            "Host gradient buffering should stay bounded to the current bucket-sized ingress/update window, "
            "not the full model."
        ),
        f"Activation handling should prefer {config.execution.activation_policy}.",
        "A dedicated transfer/update stream should overlap with the compute stream whenever possible.",
    )
    phases = (
        TrainingPhase(
            name="prefetch_active_bucket",
            owner="host_io",
            goal="Prepare the next active bucket and expert slice with minimal residency.",
            actions=(
                f"stage weights via {config.resource_policy.weight_offload_backend}",
                f"prefetch up to {config.bucket_schedule.prefetch_buckets} future bucket(s)",
                "materialize only the experts selected for the current training step",
            ),
        ),
        TrainingPhase(
            name="forward_bucket",
            owner=config.execution.compute_stream_name,
            goal="Run forward for the current bucket with minimal activation residency.",
            actions=(
                "recompute-friendly forward path",
                "avoid long-lived intermediate activations",
                "keep inactive experts off the device",
            ),
        ),
        TrainingPhase(
            name="recompute_backward_bucket",
            owner=config.execution.compute_stream_name,
            goal="Run backward in bucket order and emit gradients as soon as they are available.",
            actions=(
                "recompute missing activations for the bucket",
                "finish backward for one bucket before keeping the next resident",
                "flush gradients into a host-side buffer sized for the current bucket only",
                "hand gradients to the update path immediately",
            ),
        ),
        TrainingPhase(
            name="cpu_update_and_release",
            owner=config.execution.transfer_stream_name,
            goal="Update parameters and free gradients without waiting for the full model backward to finish.",
            actions=(
                f"ship gradients to {config.execution.gradient_device}",
                f"run optimizer math on {config.execution.optimizer_device}",
                "evaluate FP8-compressed CPU master/state/gradient staging where numerically safe",
                "release gradients and stale parameter shards immediately after update",
            ),
        ),
        TrainingPhase(
            name="rotate_experts",
            owner="scheduler",
            goal="Advance the expert window for the next training step.",
            actions=(
                (
                    "rotate the active expert set every "
                    f"{config.expert_rotation.rotate_every_steps} step(s)"
                ),
                (
                    "limit concurrent trainable experts to "
                    f"{config.expert_rotation.active_experts_per_step}"
                ),
                "preserve shared-expert handling separately from routed-expert rotation",
            ),
        ),
    )
    summary = (
        "Resource-first training scaffold for CFIE. "
        f"Development model: {config.model_targets.development_model}. "
        f"Design target: {config.model_targets.target_model}."
    )
    return TrainingBlueprint(
        package_name=config.package_name,
        profile_name=config.profile_name,
        summary=summary,
        invariants=invariants,
        phases=phases,
        metadata={},
    )
