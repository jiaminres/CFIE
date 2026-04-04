"""Blueprint helpers for the predictor-routed MoE design."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from cfie_training.blueprint import TrainingPhase
from cfie_training.config import TrainingProjectConfig


@dataclass(slots=True)
class PredictorBlueprint:
    package_name: str
    profile_name: str
    summary: str
    invariants: tuple[str, ...]
    phases: tuple[TrainingPhase, ...]
    metadata: dict[str, Any]

    # 将 predictor 蓝图序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        return {
            "package_name": self.package_name,
            "profile_name": self.profile_name,
            "summary": self.summary,
            "invariants": list(self.invariants),
            "phases": [phase.to_dict() for phase in self.phases],
            "metadata": self.metadata,
        }

    # 将 predictor 蓝图导出为稳定排序的 JSON 文本。
    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    # 将 predictor 蓝图渲染为纯文本说明。
    def render_text(self) -> str:
        # -----------------
        # 先输出蓝图头部与全局约束。
        lines = [
            f"Package: {self.package_name}",
            f"Profile: {self.profile_name}",
            self.summary,
            "",
            "Invariants:",
        ]
        lines.extend(f"- {item}" for item in self.invariants)

        # -----------------
        # 如配置中带有元数据，则单独输出。
        if self.metadata:
            lines.append("")
            lines.append("Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"- {key}: {value}")

        # -----------------
        # 最后列出 predictor 生命周期各阶段。
        lines.append("")
        lines.append("Phases:")
        for phase in self.phases:
            lines.append(f"- {phase.name} [{phase.owner}]: {phase.goal}")
            lines.extend(f"  * {action}" for action in phase.actions)
        return "\n".join(lines)


# 按训练配置生成 predictor-routed MoE 蓝图。
def build_predictor_blueprint(config: TrainingProjectConfig) -> PredictorBlueprint:
    # -----------------
    # 先校验配置，并抽取 predictor 关键参数。
    config.validate()
    predictor = config.predictor_routing
    speculative_experts = predictor.speculative_experts_per_layer

    # -----------------
    # 组织面向文档输出的摘要、约束、阶段与元数据。
    summary = (
        "Predictor-routed MoE blueprint for CFIE. "
        "Assume routed experts are already CPU-resident and keep NVMe off the "
        "steady-state online path. Each predictor window stages future expert "
        "candidates on the CPU->GPU path, then executes the real router inside the "
        "candidate pool."
    )
    invariants = (
        "Keep online expert transport on the CPU->GPU path; treat NVMe as a cold-start or offline preparation tier only.",
        (
            f"Predict {predictor.window_layers} future MoE layer(s) per window "
            f"with stride {predictor.stride_layers}."
        ),
        (
            f"Split {predictor.shared_gpu_candidate_slots} shared GPU candidate slots "
            f"evenly across the active window for {speculative_experts} speculative "
            "experts per layer."
        ),
        (
            f"Keep {predictor.candidate_experts_per_layer} candidate experts per layer "
            f"and execute top {predictor.executed_experts_per_layer} experts inside "
            "the candidate pool."
        ),
        (
            f"Train predictor quality against teacher {predictor.teacher_metric} "
            "rather than exact future top-k identity."
        ),
        (
            "Treat decode as the first target path; large-token prefill can require "
            "a separate batch-aware candidate budget."
        ),
    )
    phases = (
        TrainingPhase(
            name="capture_teacher_router_traces",
            owner="training_data",
            goal="Collect hidden-state summaries plus future-layer teacher top-k traces.",
            actions=(
                "record predictor inputs at each insertion point",
                (
                    f"store future router targets for the next {predictor.window_layers} "
                    "MoE layers"
                ),
                "measure teacher top-k coverage against the candidate budget",
            ),
        ),
        TrainingPhase(
            name="train_predictor_ranker",
            owner="cfie_training",
            goal="Optimize a future-expert ranker that maximizes candidate-pool recall.",
            actions=(
                (
                    f"emit one expert-score vector per future layer over "
                    f"{config.model_spec.num_experts or 'N'} experts"
                ),
                (
                    f"optimize teacher {predictor.teacher_metric} with a candidate "
                    f"budget of {predictor.candidate_experts_per_layer}"
                ),
                "export the predictor checkpoint and runtime schema separately from inference weights",
            ),
        ),
        TrainingPhase(
            name="cpu_to_gpu_candidate_staging",
            owner="online_transfer",
            goal="Stage candidate experts for the active predictor window from CPU to GPU.",
            actions=(
                (
                    f"reserve {predictor.shared_gpu_candidate_slots} shared GPU slots "
                    "for predictor-managed candidates"
                ),
                (
                    f"use an even split of {speculative_experts} speculative experts "
                    "per future layer in the current design"
                ),
                (
                    f"copy candidates only from {predictor.online_expert_source} "
                    "during steady-state online execution"
                ),
            ),
        ),
        TrainingPhase(
            name="masked_candidate_routing",
            owner="online_compute",
            goal="Run the real router inside the predictor-provided candidate pool.",
            actions=(
                (
                    f"construct a candidate pool of {predictor.candidate_experts_per_layer} "
                    "experts per future layer"
                ),
                (
                    f"select top {predictor.executed_experts_per_layer} experts "
                    "inside the candidate pool during execution"
                ),
                (
                    f"allow predictor mismatch with executed experts: "
                    f"{predictor.allow_candidate_mismatch}"
                ),
            ),
        ),
        TrainingPhase(
            name="shadow_eval_and_rollout",
            owner="validation",
            goal="Compare shadow-exact and masked-routing behavior before broad rollout.",
            actions=(
                "measure candidate-pool recall against the full teacher router",
                "measure decode throughput and GPU copy overlap under the staged window",
                "gate broader rollout on quality and stall reduction metrics",
            ),
        ),
    )
    metadata = {
        "development_model": config.model_targets.development_model,
        "target_model": config.model_targets.target_model,
        "num_experts": config.model_spec.num_experts,
        "num_experts_per_tok": config.model_spec.num_experts_per_tok,
        "predictor_window_layers": predictor.window_layers,
        "predictor_stride_layers": predictor.stride_layers,
        "shared_gpu_candidate_slots": predictor.shared_gpu_candidate_slots,
        "speculative_experts_per_layer": speculative_experts,
        "candidate_experts_per_layer": predictor.candidate_experts_per_layer,
        "executed_experts_per_layer": predictor.executed_experts_per_layer,
        "selection_mode": predictor.selection_mode,
        "online_expert_source": predictor.online_expert_source,
        "teacher_metric": predictor.teacher_metric,
    }

    # -----------------
    # 返回最终 predictor 蓝图对象。
    return PredictorBlueprint(
        package_name=config.package_name,
        profile_name=config.profile_name,
        summary=summary,
        invariants=invariants,
        phases=phases,
        metadata=metadata,
    )
