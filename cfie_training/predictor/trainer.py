"""Bounded predictor trainer for candidate-routed MoE experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from cfie_training.config import TrainingProjectConfig
from cfie_training.runtime.data import TokenizedDatasetBatchPlanner
from cfie_training.runtime.executor import RepresentativeBucketExecutor
from cfie_training.runtime.planner import ExpertRotationScheduler, LayerBucketPlanner
from cfie_training.runtime.session import SyntheticBatchPlanner
from cfie_training.runtime.source import LocalWeightManifest
from cfie_training.runtime.types import BatchShape


class PredictorBatchPlanner(Protocol):
    # 返回指定 step 使用的 batch 形状。
    def batch_for_step(self, step_index: int) -> BatchShape:
        ...


@dataclass(slots=True, frozen=True)
class PredictorTraceExample:
    example_index: int
    step_index: int
    insertion_layer_index: int
    future_layer_indices: tuple[int, ...]
    hidden_summary: tuple[float, ...]
    future_teacher_topk_ids: tuple[tuple[int, ...], ...]

    # 将单条 predictor trace 样本序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出样本编号、插入层、hidden summary 和未来 teacher top-k 标签。
        return {
            "example_index": self.example_index,
            "step_index": self.step_index,
            "insertion_layer_index": self.insertion_layer_index,
            "future_layer_indices": list(self.future_layer_indices),
            "hidden_summary": list(self.hidden_summary),
            "future_teacher_topk_ids": [
                list(expert_ids) for expert_ids in self.future_teacher_topk_ids
            ],
        }

    @classmethod
    # 从字典恢复单条 predictor trace 样本。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorTraceExample":
        # 逐字段恢复样本索引、未来层索引、hidden summary 和 top-k 标签。
        return cls(
            example_index=int(payload["example_index"]),
            step_index=int(payload["step_index"]),
            insertion_layer_index=int(payload["insertion_layer_index"]),
            future_layer_indices=tuple(
                int(layer_index) for layer_index in payload["future_layer_indices"]
            ),
            hidden_summary=tuple(
                float(value) for value in payload["hidden_summary"]
            ),
            future_teacher_topk_ids=tuple(
                tuple(int(expert_id) for expert_id in expert_ids)
                for expert_ids in payload["future_teacher_topk_ids"]
            ),
        )


@dataclass(slots=True, frozen=True)
class PredictorTraceDataset:
    profile_name: str
    teacher_source: str
    summary_source: str
    example_count: int
    window_layers: int
    candidate_experts_per_layer: int
    executed_experts_per_layer: int
    examples: tuple[PredictorTraceExample, ...]

    # 将 predictor trace 数据集序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出数据集来源、窗口配置和所有样本。
        return {
            "profile_name": self.profile_name,
            "teacher_source": self.teacher_source,
            "summary_source": self.summary_source,
            "example_count": self.example_count,
            "window_layers": self.window_layers,
            "candidate_experts_per_layer": self.candidate_experts_per_layer,
            "executed_experts_per_layer": self.executed_experts_per_layer,
            "examples": [example.to_dict() for example in self.examples],
        }

    # 将 predictor trace 数据集导出为 JSON。
    def to_json(self, *, indent: int = 2) -> str:
        # 使用稳定排序键和指定缩进导出 JSON 文本。
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    # 将 predictor trace 数据集写入 JSON 文件。
    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        # 直接把 JSON 文本写到目标路径。
        Path(path).write_text(self.to_json(indent=indent), encoding="utf-8")

    @classmethod
    # 从字典恢复 predictor trace 数据集。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorTraceDataset":
        # 先逐条恢复 examples 列表。
        examples = tuple(
            PredictorTraceExample.from_dict(example)
            for example in payload.get("examples", [])
        )
        # 再恢复数据集元信息和 examples 元组。
        return cls(
            profile_name=str(payload["profile_name"]),
            teacher_source=str(payload.get("teacher_source", "synthetic")),
            summary_source=str(
                payload.get("summary_source", "synthetic_formula")
            ),
            example_count=int(payload.get("example_count", len(examples))),
            window_layers=int(payload["window_layers"]),
            candidate_experts_per_layer=int(payload["candidate_experts_per_layer"]),
            executed_experts_per_layer=int(payload["executed_experts_per_layer"]),
            examples=examples,
        )

    @classmethod
    # 从 JSON 文件恢复 predictor trace 数据集。
    def from_json_file(cls, path: str | Path) -> "PredictorTraceDataset":
        # 读取并解析 JSON 文本。
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        # 顶层必须解析成对象。
        if not isinstance(payload, dict):
            raise ValueError("predictor trace dataset JSON must decode to an object")
        # 继续按字典格式恢复数据集。
        return cls.from_dict(payload)


@dataclass(slots=True, frozen=True)
class PredictorEpochSummary:
    epoch_index: int
    mean_loss: float
    recall_at_candidate_budget: float
    recall_at_executed_budget: float

    # 将单个 epoch 汇总序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出 loss 和两档 recall 指标。
        return {
            "epoch_index": self.epoch_index,
            "mean_loss": self.mean_loss,
            "recall_at_candidate_budget": self.recall_at_candidate_budget,
            "recall_at_executed_budget": self.recall_at_executed_budget,
        }

    @classmethod
    # 从字典恢复单个 epoch 汇总。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorEpochSummary":
        # 逐字段恢复 epoch 编号、loss 与两档 recall。
        return cls(
            epoch_index=int(payload["epoch_index"]),
            mean_loss=float(payload["mean_loss"]),
            recall_at_candidate_budget=float(
                payload["recall_at_candidate_budget"]
            ),
            recall_at_executed_budget=float(
                payload["recall_at_executed_budget"]
            ),
        )


@dataclass(slots=True, frozen=True)
class PredictorTrainingRunTrace:
    profile_name: str
    teacher_source: str
    summary_source: str
    example_count: int
    epochs: int
    candidate_experts_per_layer: int
    executed_experts_per_layer: int
    epoch_summaries: tuple[PredictorEpochSummary, ...]

    @property
    # 返回最后一个 epoch 的平均损失。
    def final_mean_loss(self) -> float:
        # 没有 epoch 时退回 0。
        return self.epoch_summaries[-1].mean_loss if self.epoch_summaries else 0.0

    @property
    # 返回最后一个 epoch 的 candidate budget recall。
    def final_recall_at_candidate_budget(self) -> float:
        # 没有 epoch 时退回 0。
        if not self.epoch_summaries:
            return 0.0
        # 否则取最后一个 epoch 的 recall。
        return self.epoch_summaries[-1].recall_at_candidate_budget

    @property
    # 返回最后一个 epoch 的 executed budget recall。
    def final_recall_at_executed_budget(self) -> float:
        # 没有 epoch 时退回 0。
        if not self.epoch_summaries:
            return 0.0
        # 否则取最后一个 epoch 的 recall。
        return self.epoch_summaries[-1].recall_at_executed_budget

    # 将整个 predictor 训练 run 序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出训练来源、epoch 数、最终指标和逐 epoch 汇总。
        return {
            "profile_name": self.profile_name,
            "teacher_source": self.teacher_source,
            "summary_source": self.summary_source,
            "example_count": self.example_count,
            "epochs": self.epochs,
            "candidate_experts_per_layer": self.candidate_experts_per_layer,
            "executed_experts_per_layer": self.executed_experts_per_layer,
            "final_mean_loss": self.final_mean_loss,
            "final_recall_at_candidate_budget": (
                self.final_recall_at_candidate_budget
            ),
            "final_recall_at_executed_budget": (
                self.final_recall_at_executed_budget
            ),
            "epoch_summaries": [
                summary.to_dict() for summary in self.epoch_summaries
            ],
        }

    # 将 predictor 训练 run 导出为 JSON。
    def to_json(self, *, indent: int = 2) -> str:
        # 使用稳定排序键和指定缩进导出 JSON 文本。
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    # 从字典恢复 predictor 训练 run。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorTrainingRunTrace":
        # 先恢复逐 epoch 汇总列表。
        epoch_summaries = tuple(
            PredictorEpochSummary.from_dict(summary)
            for summary in payload.get("epoch_summaries", [])
        )
        # 再恢复训练来源、预算和总 epoch 数。
        return cls(
            profile_name=str(payload["profile_name"]),
            teacher_source=str(payload["teacher_source"]),
            summary_source=str(payload["summary_source"]),
            example_count=int(payload["example_count"]),
            epochs=int(payload["epochs"]),
            candidate_experts_per_layer=int(
                payload["candidate_experts_per_layer"]
            ),
            executed_experts_per_layer=int(
                payload["executed_experts_per_layer"]
            ),
            epoch_summaries=epoch_summaries,
        )


@dataclass(slots=True, frozen=True)
class PredictorCheckpointMetadata:
    checkpoint_kind: str
    profile_name: str
    teacher_source: str
    summary_source: str
    input_summary_dim: int
    hidden_dim: int
    window_layers: int
    stride_layers: int
    num_experts: int
    candidate_experts_per_layer: int
    executed_experts_per_layer: int
    selection_mode: str
    online_expert_source: str
    allow_candidate_mismatch: bool
    example_count: int
    epochs: int
    final_mean_loss: float
    final_recall_at_candidate_budget: float
    final_recall_at_executed_budget: float

    # 将 predictor checkpoint 元信息序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出 checkpoint 对运行时兼容性和最终指标有影响的全部字段。
        return {
            "checkpoint_kind": self.checkpoint_kind,
            "profile_name": self.profile_name,
            "teacher_source": self.teacher_source,
            "summary_source": self.summary_source,
            "input_summary_dim": self.input_summary_dim,
            "hidden_dim": self.hidden_dim,
            "window_layers": self.window_layers,
            "stride_layers": self.stride_layers,
            "num_experts": self.num_experts,
            "candidate_experts_per_layer": self.candidate_experts_per_layer,
            "executed_experts_per_layer": self.executed_experts_per_layer,
            "selection_mode": self.selection_mode,
            "online_expert_source": self.online_expert_source,
            "allow_candidate_mismatch": self.allow_candidate_mismatch,
            "example_count": self.example_count,
            "epochs": self.epochs,
            "final_mean_loss": self.final_mean_loss,
            "final_recall_at_candidate_budget": (
                self.final_recall_at_candidate_budget
            ),
            "final_recall_at_executed_budget": (
                self.final_recall_at_executed_budget
            ),
        }

    @classmethod
    # 从字典恢复 predictor checkpoint 元信息。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorCheckpointMetadata":
        # 逐字段恢复 checkpoint 兼容性约束和最终指标。
        return cls(
            checkpoint_kind=str(
                payload.get("checkpoint_kind", "cfie_predictor_checkpoint")
            ),
            profile_name=str(payload["profile_name"]),
            teacher_source=str(payload["teacher_source"]),
            summary_source=str(payload["summary_source"]),
            input_summary_dim=int(payload["input_summary_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            window_layers=int(payload["window_layers"]),
            stride_layers=int(payload["stride_layers"]),
            num_experts=int(payload["num_experts"]),
            candidate_experts_per_layer=int(
                payload["candidate_experts_per_layer"]
            ),
            executed_experts_per_layer=int(
                payload["executed_experts_per_layer"]
            ),
            selection_mode=str(payload["selection_mode"]),
            online_expert_source=str(payload["online_expert_source"]),
            allow_candidate_mismatch=bool(
                payload.get("allow_candidate_mismatch", True)
            ),
            example_count=int(payload["example_count"]),
            epochs=int(payload["epochs"]),
            final_mean_loss=float(payload["final_mean_loss"]),
            final_recall_at_candidate_budget=float(
                payload["final_recall_at_candidate_budget"]
            ),
            final_recall_at_executed_budget=float(
                payload.get("final_recall_at_executed_budget", 0.0)
            ),
        )


@dataclass(slots=True, frozen=True)
class PredictorRuntimeSchema:
    schema_kind: str
    profile_name: str
    teacher_source: str
    summary_source: str
    input_summary_dim: int
    predictor_hidden_dim: int
    window_layers: int
    stride_layers: int
    num_experts: int
    candidate_experts_per_layer: int
    executed_experts_per_layer: int
    selection_mode: str
    online_expert_source: str
    allow_candidate_mismatch: bool

    # 将 predictor runtime schema 序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出运行时推理侧需要的 predictor 结构与路由约束。
        return {
            "schema_kind": self.schema_kind,
            "profile_name": self.profile_name,
            "teacher_source": self.teacher_source,
            "summary_source": self.summary_source,
            "input_summary_dim": self.input_summary_dim,
            "predictor_hidden_dim": self.predictor_hidden_dim,
            "window_layers": self.window_layers,
            "stride_layers": self.stride_layers,
            "num_experts": self.num_experts,
            "candidate_experts_per_layer": self.candidate_experts_per_layer,
            "executed_experts_per_layer": self.executed_experts_per_layer,
            "selection_mode": self.selection_mode,
            "online_expert_source": self.online_expert_source,
            "allow_candidate_mismatch": self.allow_candidate_mismatch,
        }

    # 将 predictor runtime schema 导出为 JSON。
    def to_json(self, *, indent: int = 2) -> str:
        # 使用稳定排序键和指定缩进导出 JSON 文本。
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    # 将 predictor runtime schema 写入 JSON 文件。
    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        # 直接把 JSON 文本写到目标路径。
        Path(path).write_text(self.to_json(indent=indent), encoding="utf-8")

    @classmethod
    # 基于 checkpoint 元信息构造运行时 schema。
    def from_checkpoint_metadata(
        cls,
        metadata: PredictorCheckpointMetadata,
    ) -> "PredictorRuntimeSchema":
        # 把 checkpoint 中的兼容性字段映射到 runtime schema。
        return cls(
            schema_kind="cfie_predictor_runtime_schema",
            profile_name=metadata.profile_name,
            teacher_source=metadata.teacher_source,
            summary_source=metadata.summary_source,
            input_summary_dim=metadata.input_summary_dim,
            predictor_hidden_dim=metadata.hidden_dim,
            window_layers=metadata.window_layers,
            stride_layers=metadata.stride_layers,
            num_experts=metadata.num_experts,
            candidate_experts_per_layer=metadata.candidate_experts_per_layer,
            executed_experts_per_layer=metadata.executed_experts_per_layer,
            selection_mode=metadata.selection_mode,
            online_expert_source=metadata.online_expert_source,
            allow_candidate_mismatch=metadata.allow_candidate_mismatch,
        )

    @classmethod
    # 从字典恢复 predictor runtime schema。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorRuntimeSchema":
        # 逐字段恢复运行时 schema 内容。
        return cls(
            schema_kind=str(
                payload.get("schema_kind", "cfie_predictor_runtime_schema")
            ),
            profile_name=str(payload["profile_name"]),
            teacher_source=str(payload["teacher_source"]),
            summary_source=str(payload["summary_source"]),
            input_summary_dim=int(payload["input_summary_dim"]),
            predictor_hidden_dim=int(payload["predictor_hidden_dim"]),
            window_layers=int(payload["window_layers"]),
            stride_layers=int(payload["stride_layers"]),
            num_experts=int(payload["num_experts"]),
            candidate_experts_per_layer=int(
                payload["candidate_experts_per_layer"]
            ),
            executed_experts_per_layer=int(
                payload["executed_experts_per_layer"]
            ),
            selection_mode=str(payload["selection_mode"]),
            online_expert_source=str(payload["online_expert_source"]),
            allow_candidate_mismatch=bool(
                payload.get("allow_candidate_mismatch", True)
            ),
        )


@dataclass(slots=True, frozen=True)
class PredictorEvaluationTrace:
    profile_name: str
    teacher_source: str
    summary_source: str
    example_count: int
    candidate_experts_per_layer: int
    executed_experts_per_layer: int
    mean_loss: float
    recall_at_candidate_budget: float
    recall_at_executed_budget: float
    checkpoint_metadata: PredictorCheckpointMetadata | None = None

    # 将 predictor 评估结果序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出评估来源、loss、recall，以及可选的 checkpoint 元信息。
        return {
            "profile_name": self.profile_name,
            "teacher_source": self.teacher_source,
            "summary_source": self.summary_source,
            "example_count": self.example_count,
            "candidate_experts_per_layer": self.candidate_experts_per_layer,
            "executed_experts_per_layer": self.executed_experts_per_layer,
            "mean_loss": self.mean_loss,
            "recall_at_candidate_budget": self.recall_at_candidate_budget,
            "recall_at_executed_budget": self.recall_at_executed_budget,
            "checkpoint_metadata": (
                None
                if self.checkpoint_metadata is None
                else self.checkpoint_metadata.to_dict()
            ),
        }

    # 将 predictor 评估结果导出为 JSON。
    def to_json(self, *, indent: int = 2) -> str:
        # 使用稳定排序键和指定缩进导出 JSON 文本。
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(slots=True, frozen=True)
class PredictorMetricsSummary:
    metrics_kind: str
    profile_name: str
    teacher_source: str
    summary_source: str
    example_count: int
    epochs: int
    final_mean_loss: float
    final_recall_at_candidate_budget: float
    final_recall_at_executed_budget: float

    # 将 predictor 指标摘要序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出最终 loss 与 recall 指标摘要。
        return {
            "metrics_kind": self.metrics_kind,
            "profile_name": self.profile_name,
            "teacher_source": self.teacher_source,
            "summary_source": self.summary_source,
            "example_count": self.example_count,
            "epochs": self.epochs,
            "final_mean_loss": self.final_mean_loss,
            "final_recall_at_candidate_budget": (
                self.final_recall_at_candidate_budget
            ),
            "final_recall_at_executed_budget": (
                self.final_recall_at_executed_budget
            ),
        }

    # 将 predictor 指标摘要导出为 JSON。
    def to_json(self, *, indent: int = 2) -> str:
        # 使用稳定排序键和指定缩进导出 JSON 文本。
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    # 将 predictor 指标摘要写入 JSON 文件。
    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        # 直接把 JSON 文本写到目标路径。
        Path(path).write_text(self.to_json(indent=indent), encoding="utf-8")

    @classmethod
    # 根据 checkpoint 元信息构造指标摘要。
    def from_checkpoint_metadata(
        cls,
        metadata: PredictorCheckpointMetadata,
    ) -> "PredictorMetricsSummary":
        # 直接把 checkpoint 中的最终指标映射成 metrics summary。
        return cls(
            metrics_kind="cfie_predictor_metrics_summary",
            profile_name=metadata.profile_name,
            teacher_source=metadata.teacher_source,
            summary_source=metadata.summary_source,
            example_count=metadata.example_count,
            epochs=metadata.epochs,
            final_mean_loss=metadata.final_mean_loss,
            final_recall_at_candidate_budget=(
                metadata.final_recall_at_candidate_budget
            ),
            final_recall_at_executed_budget=(
                metadata.final_recall_at_executed_budget
            ),
        )


@dataclass(slots=True, frozen=True)
class PredictorDeploymentManifest:
    bundle_kind: str
    profile_name: str
    teacher_source: str
    summary_source: str
    source_checkpoint: str
    weights_kind: str
    weights_format: str
    weights_file: str
    schema_kind: str
    schema_file: str
    metrics_kind: str
    metrics_file: str

    # 将 predictor 部署清单序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出 bundle 中各个文件及其语义类型。
        return {
            "bundle_kind": self.bundle_kind,
            "profile_name": self.profile_name,
            "teacher_source": self.teacher_source,
            "summary_source": self.summary_source,
            "source_checkpoint": self.source_checkpoint,
            "weights_kind": self.weights_kind,
            "weights_format": self.weights_format,
            "weights_file": self.weights_file,
            "schema_kind": self.schema_kind,
            "schema_file": self.schema_file,
            "metrics_kind": self.metrics_kind,
            "metrics_file": self.metrics_file,
        }

    # 将 predictor 部署清单导出为 JSON。
    def to_json(self, *, indent: int = 2) -> str:
        # 使用稳定排序键和指定缩进导出 JSON 文本。
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    # 将 predictor 部署清单写入 JSON 文件。
    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        # 直接把 JSON 文本写到目标路径。
        Path(path).write_text(self.to_json(indent=indent), encoding="utf-8")


class FutureExpertPredictor(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        window_layers: int,
        num_experts: int,
    ) -> None:
        # 先初始化 PyTorch 模块基类。
        super().__init__()
        # 缓存输出窗口层数。
        self.window_layers = window_layers
        # 缓存每层 expert 数。
        self.num_experts = num_experts
        # 构造一个轻量 MLP，用于把 hidden summary 映射成未来 expert logits。
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, window_layers * num_experts),
        )

    # 前向返回 [batch, window_layers, num_experts] logits。
    def forward(self, hidden_summary: torch.Tensor) -> torch.Tensor:
        # 先经由 MLP 得到展平的 logits。
        logits = self.net(hidden_summary)
        # 再恢复成“未来层窗口 x expert”二维结构。
        return logits.view(-1, self.window_layers, self.num_experts)


class SyntheticTeacherTraceBuilder:
    teacher_source = "synthetic"
    summary_source = "synthetic_formula"

    def __init__(self, config: TrainingProjectConfig) -> None:
        # 缓存配置，并先执行一次配置校验。
        self.config = config
        self.config.validate()
        # predictor trace 生成必须建立在完整 model_spec 之上。
        if not self.config.model_spec.is_defined():
            raise ValueError(
                "predictor trace generation requires a defined model_spec"
            )
        # 读取 predictor trace 构造所需的核心超参数。
        self._summary_dim = self.config.predictor_trainer.input_summary_dim
        self._num_experts = self.config.model_spec.num_experts
        self._num_layers = self.config.model_spec.num_hidden_layers
        self._window_layers = self.config.predictor_routing.window_layers
        self._executed_experts = self.config.predictor_routing.executed_experts_per_layer
        self._stride_layers = self.config.predictor_routing.stride_layers
        self._noise_scale = self.config.predictor_trainer.synthetic_trace_noise_scale
        self._seed = self.config.predictor_trainer.seed
        # 预构造 feature 索引，用于后续合成 hidden summary。
        self._feature_ids = torch.arange(
            1, self._summary_dim + 1, dtype=torch.float32, device="cpu"
        )
        # 预构造 expert basis 和 layer basis，避免重复计算。
        self._expert_basis = self._build_expert_basis()
        self._layer_basis = self._build_layer_basis()

    def _build_expert_basis(self) -> torch.Tensor:
        # 构造 expert 维索引。
        expert_ids = torch.arange(
            1, self._num_experts + 1, dtype=torch.float32, device="cpu"
        ).unsqueeze(1)
        # 构造 feature 维索引。
        feature_ids = self._feature_ids.unsqueeze(0)
        # 用 sin/cos 组合构造 expert basis。
        basis = torch.sin(expert_ids * 0.017 + feature_ids * 0.071)
        basis = basis + 0.5 * torch.cos(expert_ids * 0.031 + feature_ids * 0.043)
        # 统一缩放到稳定范围。
        return basis / 1.5

    def _build_layer_basis(self) -> torch.Tensor:
        # 构造层号索引。
        layer_ids = torch.arange(
            1, self._num_layers + 1, dtype=torch.float32, device="cpu"
        ).unsqueeze(1)
        # 复用 feature 维索引。
        feature_ids = self._feature_ids.unsqueeze(0)
        # 用 sin/cos 组合构造 layer basis。
        basis = torch.cos(layer_ids * 0.13 + feature_ids * 0.051)
        basis = basis + 0.35 * torch.sin(layer_ids * 0.07 + feature_ids * 0.029)
        # 统一缩放到稳定范围。
        return basis / 1.35

    def _build_hidden_summary(
        self,
        *,
        step_index: int,
        example_offset: int,
        insertion_layer_index: int,
    ) -> torch.Tensor:
        # step 和 example offset 共同决定当前样本的基础相位。
        phase = 0.19 * float(step_index + 1) + 0.11 * float(example_offset + 1)
        # 先生成基础 summary 波形。
        summary = torch.sin(self._feature_ids * 0.097 + phase)
        # 再叠加 insertion layer 驱动的调制项。
        summary = summary + 0.45 * torch.cos(
            self._feature_ids * 0.053 + 0.09 * float(insertion_layer_index + 1)
        )
        # 若开启噪声，则再注入一项可复现的轻量噪声。
        if self._noise_scale > 0:
            noise_phase = torch.sin(
                self._feature_ids * 0.211
                + 0.23 * float(step_index + example_offset + self._seed + 1)
            )
            summary = summary + self._noise_scale * noise_phase
        # 统一缩放后返回 summary。
        return summary / 1.6

    def _teacher_scores(
        self,
        *,
        hidden_summary: torch.Tensor,
        future_layer_index: int,
    ) -> torch.Tensor:
        # 取出未来层对应的 layer basis。
        layer_basis = self._layer_basis[future_layer_index]
        # 用 layer basis 对 hidden summary 做轻量调制。
        mixed = hidden_summary * (1.0 + 0.25 * layer_basis)
        # 与 expert basis 做乘法得到每个 expert 的分数。
        scores = self._expert_basis @ mixed
        # 叠加一项与 expert id / layer id 相关的扰动，拉开分数差异。
        expert_ids = torch.arange(
            0, self._num_experts, dtype=torch.float32, device="cpu"
        )
        scores = scores + 0.3 * torch.sin(
            expert_ids * 0.019 + 0.07 * float(future_layer_index + 1)
        )
        # 返回最终 teacher 分数。
        return scores

    def build_examples(
        self,
        *,
        steps: int,
        examples_per_step: int | None = None,
        batch_planner: PredictorBatchPlanner | None = None,
    ) -> tuple[PredictorTraceExample, ...]:
        # 训练步数至少为 1。
        if steps < 1:
            raise ValueError("steps must be >= 1")
        # 未显式给 examples_per_step 时，退回配置中的默认值。
        examples_per_step = (
            self.config.predictor_trainer.examples_per_step
            if examples_per_step is None
            else int(examples_per_step)
        )
        # 每步样本数至少为 1。
        if examples_per_step < 1:
            raise ValueError("examples_per_step must be >= 1")

        # 用列表累积全部 trace 样本。
        examples: list[PredictorTraceExample] = []
        for step_index in range(steps):
            for example_offset in range(examples_per_step):
                # 当前插入层按 stride 在全层范围内循环推进。
                insertion_layer_index = (
                    step_index * self._stride_layers
                    + example_offset * self._stride_layers
                ) % self._num_layers
                # 先生成该插入层的 hidden summary。
                hidden_summary = self._build_hidden_summary(
                    step_index=step_index,
                    example_offset=example_offset,
                    insertion_layer_index=insertion_layer_index,
                )
                # 计算未来窗口内的层索引列表。
                future_layer_indices = tuple(
                    (insertion_layer_index + offset + 1) % self._num_layers
                    for offset in range(self._window_layers)
                )
                # 用列表累积每个未来层的 teacher top-k experts。
                future_teacher_topk_ids = []
                for future_layer_index in future_layer_indices:
                    # 先计算当前未来层的 teacher scores。
                    teacher_scores = self._teacher_scores(
                        hidden_summary=hidden_summary,
                        future_layer_index=future_layer_index,
                    )
                    # 再取 executed_experts 预算下的 top-k experts。
                    teacher_topk = torch.topk(
                        teacher_scores,
                        k=self._executed_experts,
                        dim=0,
                    ).indices
                    # 转成 Python 元组，便于后续 JSON 序列化。
                    future_teacher_topk_ids.append(
                        tuple(int(expert_id) for expert_id in teacher_topk.tolist())
                    )
                # 将当前样本写入 examples 列表。
                examples.append(
                    PredictorTraceExample(
                        example_index=len(examples),
                        step_index=step_index,
                        insertion_layer_index=insertion_layer_index,
                        future_layer_indices=future_layer_indices,
                        hidden_summary=tuple(float(value) for value in hidden_summary.tolist()),
                        future_teacher_topk_ids=tuple(future_teacher_topk_ids),
                    )
                )
        # 返回不可变样本元组。
        return tuple(examples)


class ManifestRouterTeacherTraceBuilder(SyntheticTeacherTraceBuilder):
    teacher_source = "manifest_router"

    def __init__(
        self,
        config: TrainingProjectConfig,
        manifest: LocalWeightManifest,
    ) -> None:
        # 先初始化父类中的通用 synthetic trace 构造逻辑。
        super().__init__(config)
        # 缓存本地权重 manifest。
        self._manifest = manifest
        # 缓存模型 hidden_size，供 router gate 投影使用。
        self._hidden_size = self.config.model_spec.hidden_size
        # 预构造 hidden_summary -> hidden_state 的投影 basis。
        self._projection_basis = self._build_projection_basis()
        # 为 router gate 张量建立按层缓存。
        self._router_gate_cache: dict[int, torch.Tensor] = {}

    def _build_projection_basis(self) -> torch.Tensor:
        # feature 维作为输入轴。
        feature_ids = self._feature_ids.unsqueeze(1)
        # hidden 维作为输出轴。
        hidden_ids = torch.arange(
            1,
            self._hidden_size + 1,
            dtype=torch.float32,
            device="cpu",
        ).unsqueeze(0)
        # 用 sin/cos 组合构造 summary->hidden 的投影 basis。
        basis = torch.sin(feature_ids * 0.031 + hidden_ids * 0.017)
        basis = basis + 0.5 * torch.cos(feature_ids * 0.047 + hidden_ids * 0.023)
        # 统一缩放到稳定范围。
        return basis / 1.5

    def _project_hidden_state(self, hidden_summary: torch.Tensor) -> torch.Tensor:
        # 先把 hidden summary 投影到模型 hidden 维。
        projected = hidden_summary @ self._projection_basis
        # 再用 tanh 压缩动态范围。
        return torch.tanh(projected / max(self._summary_dim, 1) ** 0.5)

    def _router_gate_tensor(self, layer_index: int) -> torch.Tensor:
        # 先查按层缓存，避免重复从 manifest 读取。
        cached = self._router_gate_cache.get(layer_index)
        if cached is not None:
            return cached
        # 从 manifest 加载当前层的 router gate 权重。
        tensor = self._manifest.load_router_gate_tensor(
            layer_index,
            dtype=torch.float32,
        )
        # 缺失权重时直接报错。
        if tensor is None:
            raise ValueError(
                "missing router gate tensor for "
                f"model.language_model.layers.{layer_index}.mlp.gate.weight"
            )
        # router gate 必须是二维矩阵。
        if tensor.ndim != 2:
            raise ValueError(
                "router gate tensor must be rank-2, got "
                f"shape={tuple(tensor.shape)} for layer {layer_index}"
            )
        # 两个轴里必须有一个轴对应 expert 维。
        if self._num_experts not in tensor.shape:
            raise ValueError(
                "router gate tensor must expose the expert axis, got "
                f"shape={tuple(tensor.shape)} for layer {layer_index}"
            )
        # 缓存并返回当前层的 router gate 张量。
        self._router_gate_cache[layer_index] = tensor
        return tensor

    def _teacher_scores(
        self,
        *,
        hidden_summary: torch.Tensor,
        future_layer_index: int,
    ) -> torch.Tensor:
        # 先把 hidden summary 投影到近似 hidden state。
        hidden_state = self._project_hidden_state(hidden_summary)
        # 读取未来层对应的 router gate 张量。
        router_gate = self._router_gate_tensor(future_layer_index)
        if (
            router_gate.shape[0] == self._num_experts
            and router_gate.shape[1] == hidden_state.numel()
        ):
            # [expert, hidden] 形状时，直接做矩阵乘。
            scores = router_gate @ hidden_state
        elif (
            router_gate.shape[1] == self._num_experts
            and router_gate.shape[0] == hidden_state.numel()
        ):
            # [hidden, expert] 形状时，改走右乘路径。
            scores = hidden_state @ router_gate
        else:
            # 两种常见形状都不匹配时，视为 manifest 权重不兼容。
            raise ValueError(
                "router gate tensor shape does not match predictor hidden state: "
                f"shape={tuple(router_gate.shape)} hidden={hidden_state.numel()}"
            )
        # 再加一项极小扰动，避免完全同分。
        expert_ids = torch.arange(
            0,
            self._num_experts,
            dtype=torch.float32,
            device="cpu",
        )
        return scores + 1e-4 * torch.sin(
            expert_ids * 0.019 + 0.07 * float(future_layer_index + 1)
        )


class ManifestRouterRuntimeTraceBuilder(ManifestRouterTeacherTraceBuilder):
    summary_source = "representative_runtime"

    def __init__(
        self,
        config: TrainingProjectConfig,
        manifest: LocalWeightManifest,
        executor: RepresentativeBucketExecutor,
        layer_buckets,
        rotation: ExpertRotationScheduler,
    ) -> None:
        # 先初始化 manifest router teacher trace 构造基类。
        super().__init__(config, manifest)
        # 缓存代表性执行器、layer buckets 和 expert rotation 调度器。
        self._executor = executor
        self._layer_buckets = tuple(layer_buckets)
        self._rotation = rotation
        # 预构造可选 insertion layer 列表。
        self._insertion_layer_indices = tuple(
            range(0, self._num_layers, self._stride_layers)
        )
        # 建立“插入层 -> bucket”反向索引。
        self._bucket_for_insertion_layer = {
            layer_index: bucket
            for bucket in self._layer_buckets
            for layer_index in bucket.layer_indices
        }

    def _selected_insertion_layers(
        self,
        *,
        step_index: int,
        examples_per_step: int | None,
    ) -> tuple[int, ...]:
        # 没有 insertion layer 候选时直接返回空元组。
        if not self._insertion_layer_indices:
            return ()
        # 未显式限制样本数时，直接返回全部 insertion layer。
        if examples_per_step is None:
            return self._insertion_layer_indices
        # 显式给定样本数时，先转成整数。
        example_count = int(examples_per_step)
        # 每步样本数至少为 1。
        if example_count < 1:
            raise ValueError("examples_per_step must be >= 1")
        # 每步样本数不能超过 insertion layer 总数。
        example_count = min(example_count, len(self._insertion_layer_indices))
        # 按 step 驱动起点滑动，实现轮转采样。
        start = (step_index * example_count) % len(self._insertion_layer_indices)
        # 返回本 step 选中的 insertion layer 元组。
        return tuple(
            self._insertion_layer_indices[
                (start + example_offset) % len(self._insertion_layer_indices)
            ]
            for example_offset in range(example_count)
        )

    def build_examples(
        self,
        *,
        steps: int,
        examples_per_step: int | None = None,
        batch_planner: PredictorBatchPlanner | None = None,
    ) -> tuple[PredictorTraceExample, ...]:
        # 训练步数至少为 1。
        if steps < 1:
            raise ValueError("steps must be >= 1")
        # 运行时代表性 trace 必须依赖 batch planner。
        if batch_planner is None:
            raise ValueError(
                "runtime-backed predictor traces require a batch planner"
            )
        # 用列表累积全部 trace 样本。
        examples: list[PredictorTraceExample] = []
        for step_index in range(steps):
            # 先取当前 step 的 batch。
            batch = batch_planner.batch_for_step(step_index)
            # 再根据当前 batch 规划 active expert window。
            active_expert_ids = self._rotation.plan_window(
                step_index=step_index,
                batch=batch,
                layer_buckets=self._layer_buckets,
                next_batch=batch,
            ).active_expert_ids
            for insertion_layer_index in self._selected_insertion_layers(
                step_index=step_index,
                examples_per_step=examples_per_step,
            ):
                # 通过 insertion layer 找到所属 bucket。
                bucket = self._bucket_for_insertion_layer[insertion_layer_index]
                # 用代表性执行器生成 runtime-backed hidden summary。
                hidden_summary = self._executor.build_predictor_hidden_summary(
                    step_index=step_index,
                    batch=batch,
                    bucket=bucket,
                    insertion_layer_index=insertion_layer_index,
                    active_expert_ids=active_expert_ids,
                    summary_dim=self._summary_dim,
                )
                # 计算未来窗口内的层索引列表。
                future_layer_indices = tuple(
                    (insertion_layer_index + offset + 1) % self._num_layers
                    for offset in range(self._window_layers)
                )
                # 用列表累积每个未来层的 teacher top-k experts。
                future_teacher_topk_ids = []
                for future_layer_index in future_layer_indices:
                    # 先计算当前未来层的 teacher scores。
                    teacher_scores = self._teacher_scores(
                        hidden_summary=hidden_summary,
                        future_layer_index=future_layer_index,
                    )
                    # 再取 executed_experts 预算下的 top-k experts。
                    teacher_topk = torch.topk(
                        teacher_scores,
                        k=self._executed_experts,
                        dim=0,
                    ).indices
                    # 转成 Python 元组，便于后续 JSON 序列化。
                    future_teacher_topk_ids.append(
                        tuple(int(expert_id) for expert_id in teacher_topk.tolist())
                    )
                # 将当前样本写入 examples 列表。
                examples.append(
                    PredictorTraceExample(
                        example_index=len(examples),
                        step_index=step_index,
                        insertion_layer_index=insertion_layer_index,
                        future_layer_indices=future_layer_indices,
                        hidden_summary=tuple(
                            float(value) for value in hidden_summary.tolist()
                        ),
                        future_teacher_topk_ids=tuple(future_teacher_topk_ids),
                    )
                )
        # 返回不可变样本元组。
        return tuple(examples)


class PredictorTrainer:
    def __init__(self, config: TrainingProjectConfig) -> None:
        # 先校验并缓存训练配置。
        self.config = config.validate()
        # predictor 训练必须建立在完整 model_spec 之上。
        if not self.config.model_spec.is_defined():
            raise ValueError("predictor training requires a defined model_spec")
        # 初始化 manifest、bucket planner、expert rotation 和代表性执行器。
        self._manifest = LocalWeightManifest(self.config)
        self._layer_buckets = LayerBucketPlanner(self.config).build()
        self._rotation = ExpertRotationScheduler(self.config)
        self._executor = RepresentativeBucketExecutor(self.config)
        # 根据本地条件选择合适的 teacher trace 构造器。
        self._trace_builder = self._build_trace_builder()

    @property
    # 返回当前训练器使用的 teacher source。
    def teacher_source(self) -> str:
        return self._trace_builder.teacher_source

    @property
    # 返回当前训练器使用的 summary source。
    def summary_source(self) -> str:
        return self._trace_builder.summary_source

    def _build_trace_builder(
        self,
    ) -> (
        SyntheticTeacherTraceBuilder
        | ManifestRouterTeacherTraceBuilder
        | ManifestRouterRuntimeTraceBuilder
    ):
        # manifest 可用且所有层都能找到 router gate 时，优先走 runtime-backed trace。
        if self._manifest.available and all(
            self._manifest.router_gate_ref(layer_index) is not None
            for layer_index in range(self.config.model_spec.num_hidden_layers)
        ):
            return ManifestRouterRuntimeTraceBuilder(
                self.config,
                self._manifest,
                self._executor,
                self._layer_buckets,
                self._rotation,
            )
        # 否则退回纯 synthetic trace 构造器。
        return SyntheticTeacherTraceBuilder(self.config)

    def _build_batch_planner(
        self,
        *,
        samples: int,
        tokens_per_sample: int,
        dataset_path: str | None = None,
        tokenizer_path: str | None = None,
        dataset_format: str = "auto",
        dataset_text_key: str = "text",
    ) -> PredictorBatchPlanner:
        # 提供数据集路径时，构造 tokenizer-backed batch planner。
        if dataset_path is not None:
            return TokenizedDatasetBatchPlanner(
                config=self.config,
                dataset_path=dataset_path,
                base_samples=samples,
                tokens_per_sample=tokens_per_sample,
                tokenizer_path=tokenizer_path,
                dataset_format=dataset_format,
                dataset_text_key=dataset_text_key,
            )
        # 否则退回 synthetic batch planner。
        return SyntheticBatchPlanner(
            base_samples=samples,
            base_tokens_per_sample=tokens_per_sample,
        )

    def build_model(self) -> FutureExpertPredictor:
        # 读取 predictor trainer 与 routing 配置。
        trainer_cfg = self.config.predictor_trainer
        routing_cfg = self.config.predictor_routing
        # 根据配置构造 predictor 模型。
        return FutureExpertPredictor(
            input_dim=trainer_cfg.input_summary_dim,
            hidden_dim=trainer_cfg.hidden_dim,
            window_layers=routing_cfg.window_layers,
            num_experts=self.config.model_spec.num_experts,
        )

    def build_runtime_schema(
        self,
        *,
        teacher_source: str | None = None,
        summary_source: str | None = None,
    ) -> PredictorRuntimeSchema:
        # 读取 routing 与 trainer 配置。
        routing_cfg = self.config.predictor_routing
        trainer_cfg = self.config.predictor_trainer
        # 组装当前配置对应的 runtime schema。
        return PredictorRuntimeSchema(
            schema_kind="cfie_predictor_runtime_schema",
            profile_name=self.config.profile_name,
            teacher_source=(
                self.teacher_source if teacher_source is None else teacher_source
            ),
            summary_source=(
                self.summary_source if summary_source is None else summary_source
            ),
            input_summary_dim=trainer_cfg.input_summary_dim,
            predictor_hidden_dim=trainer_cfg.hidden_dim,
            window_layers=routing_cfg.window_layers,
            stride_layers=routing_cfg.stride_layers,
            num_experts=self.config.model_spec.num_experts,
            candidate_experts_per_layer=routing_cfg.candidate_experts_per_layer,
            executed_experts_per_layer=routing_cfg.executed_experts_per_layer,
            selection_mode=routing_cfg.selection_mode,
            online_expert_source=routing_cfg.online_expert_source,
            allow_candidate_mismatch=routing_cfg.allow_candidate_mismatch,
        )

    @staticmethod
    # 从 checkpoint 文件读取原始载荷。
    def _read_checkpoint_payload(path: str | Path) -> dict[str, Any]:
        # 使用 torch.load 在 CPU 上读取 checkpoint。
        payload = torch.load(Path(path), map_location="cpu")
        # 顶层必须解码成字典。
        if not isinstance(payload, dict):
            raise ValueError("predictor checkpoint must decode to a dictionary")
        # 返回原始 payload。
        return payload

    @classmethod
    # 只读取 predictor checkpoint 的 metadata 部分。
    def read_checkpoint_metadata(
        cls,
        path: str | Path,
    ) -> PredictorCheckpointMetadata:
        # 先读取 checkpoint 原始载荷。
        payload = cls._read_checkpoint_payload(path)
        # 取出 metadata 部分。
        metadata_payload = payload.get("metadata")
        # metadata 必须存在且为字典。
        if not isinstance(metadata_payload, dict):
            raise ValueError("predictor checkpoint is missing metadata")
        # 继续恢复为 PredictorCheckpointMetadata。
        return PredictorCheckpointMetadata.from_dict(metadata_payload)

    @classmethod
    # 只读取 predictor checkpoint 的 run_trace 部分。
    def read_checkpoint_run_trace(
        cls,
        path: str | Path,
    ) -> PredictorTrainingRunTrace | None:
        # 先读取 checkpoint 原始载荷。
        payload = cls._read_checkpoint_payload(path)
        # 取出可选的 run_trace 部分。
        run_trace_payload = payload.get("run_trace")
        # 旧 checkpoint 没有 run_trace 时直接返回空。
        if run_trace_payload is None:
            return None
        # run_trace 若存在，则必须是字典。
        if not isinstance(run_trace_payload, dict):
            raise ValueError("predictor checkpoint run_trace must be a dictionary")
        # 继续恢复为 PredictorTrainingRunTrace。
        return PredictorTrainingRunTrace.from_dict(run_trace_payload)

    def _checkpoint_metadata(
        self,
        run_trace: PredictorTrainingRunTrace,
    ) -> PredictorCheckpointMetadata:
        # 读取 routing 与 trainer 配置。
        routing_cfg = self.config.predictor_routing
        trainer_cfg = self.config.predictor_trainer
        # 根据 run_trace 和当前配置组装 checkpoint 元信息。
        return PredictorCheckpointMetadata(
            checkpoint_kind="cfie_predictor_checkpoint",
            profile_name=run_trace.profile_name,
            teacher_source=run_trace.teacher_source,
            summary_source=run_trace.summary_source,
            input_summary_dim=trainer_cfg.input_summary_dim,
            hidden_dim=trainer_cfg.hidden_dim,
            window_layers=routing_cfg.window_layers,
            stride_layers=routing_cfg.stride_layers,
            num_experts=self.config.model_spec.num_experts,
            candidate_experts_per_layer=routing_cfg.candidate_experts_per_layer,
            executed_experts_per_layer=routing_cfg.executed_experts_per_layer,
            selection_mode=routing_cfg.selection_mode,
            online_expert_source=routing_cfg.online_expert_source,
            allow_candidate_mismatch=routing_cfg.allow_candidate_mismatch,
            example_count=run_trace.example_count,
            epochs=run_trace.epochs,
            final_mean_loss=run_trace.final_mean_loss,
            final_recall_at_candidate_budget=(
                run_trace.final_recall_at_candidate_budget
            ),
            final_recall_at_executed_budget=(
                run_trace.final_recall_at_executed_budget
            ),
        )

    def _validate_checkpoint_metadata(
        self,
        metadata: PredictorCheckpointMetadata,
    ) -> None:
        # 读取当前 trainer 与 routing 配置。
        trainer_cfg = self.config.predictor_trainer
        routing_cfg = self.config.predictor_routing
        # 用列表累积所有不兼容字段。
        mismatches = []
        if metadata.input_summary_dim != trainer_cfg.input_summary_dim:
            mismatches.append("input_summary_dim")
        if metadata.hidden_dim != trainer_cfg.hidden_dim:
            mismatches.append("hidden_dim")
        if metadata.window_layers != routing_cfg.window_layers:
            mismatches.append("window_layers")
        if metadata.stride_layers != routing_cfg.stride_layers:
            mismatches.append("stride_layers")
        if metadata.num_experts != self.config.model_spec.num_experts:
            mismatches.append("num_experts")
        if (
            metadata.candidate_experts_per_layer
            != routing_cfg.candidate_experts_per_layer
        ):
            mismatches.append("candidate_experts_per_layer")
        if (
            metadata.executed_experts_per_layer
            != routing_cfg.executed_experts_per_layer
        ):
            mismatches.append("executed_experts_per_layer")
        # 只要存在任一关键字段不匹配，就拒绝加载 checkpoint。
        if mismatches:
            raise ValueError(
                "predictor checkpoint is incompatible with current config: "
                + ", ".join(mismatches)
            )

    def _validate_resume_run_trace(
        self,
        run_trace: PredictorTrainingRunTrace,
    ) -> None:
        # 先校验 profile/source 与当前训练器一致。
        mismatches = []
        if run_trace.profile_name != self.config.profile_name:
            mismatches.append("profile_name")
        if run_trace.candidate_experts_per_layer != (
            self.config.predictor_routing.candidate_experts_per_layer
        ):
            mismatches.append("candidate_experts_per_layer")
        if run_trace.executed_experts_per_layer != (
            self.config.predictor_routing.executed_experts_per_layer
        ):
            mismatches.append("executed_experts_per_layer")
        # 若 run_trace 自身与当前配置不兼容，则拒绝作为续训起点。
        if mismatches:
            raise ValueError(
                "predictor checkpoint run_trace is incompatible with current config: "
                + ", ".join(mismatches)
            )

    def _validate_resume_dataset(
        self,
        dataset: PredictorTraceDataset,
        run_trace: PredictorTrainingRunTrace,
    ) -> None:
        # 续训时要求数据集来源与历史 run_trace 一致。
        mismatches = []
        if dataset.profile_name != run_trace.profile_name:
            mismatches.append("profile_name")
        if dataset.teacher_source != run_trace.teacher_source:
            mismatches.append("teacher_source")
        if dataset.summary_source != run_trace.summary_source:
            mismatches.append("summary_source")
        if dataset.example_count != run_trace.example_count:
            mismatches.append("example_count")
        if dataset.candidate_experts_per_layer != (
            run_trace.candidate_experts_per_layer
        ):
            mismatches.append("candidate_experts_per_layer")
        if dataset.executed_experts_per_layer != (
            run_trace.executed_experts_per_layer
        ):
            mismatches.append("executed_experts_per_layer")
        # 数据集与续训起点不一致时直接报错。
        if mismatches:
            raise ValueError(
                "predictor resume dataset is incompatible with checkpoint run_trace: "
                + ", ".join(mismatches)
            )

    def save_checkpoint(
        self,
        *,
        model: FutureExpertPredictor,
        run_trace: PredictorTrainingRunTrace,
        path: str | Path,
        optimizer_state_dict: dict[str, Any] | None = None,
    ) -> PredictorCheckpointMetadata:
        # 规范化 checkpoint 路径。
        checkpoint_path = Path(path)
        # 预先创建父目录。
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # 基于 run_trace 生成 metadata。
        metadata = self._checkpoint_metadata(run_trace)
        # 以 torch.save 写入 metadata 和 model_state_dict。
        payload = {
            "checkpoint_kind": metadata.checkpoint_kind,
            "metadata": metadata.to_dict(),
            "model_state_dict": model.state_dict(),
            "run_trace": run_trace.to_dict(),
        }
        # 若调用方提供了优化器状态，则一并写入 checkpoint。
        if optimizer_state_dict is not None:
            payload["optimizer_state_dict"] = optimizer_state_dict
        torch.save(payload, checkpoint_path)
        # 返回写入的 metadata。
        return metadata

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> tuple[FutureExpertPredictor, PredictorCheckpointMetadata]:
        # 先读取 checkpoint 原始载荷。
        payload = self._read_checkpoint_payload(path)
        # 提取模型参数字典。
        state_dict = payload.get("model_state_dict")
        # checkpoint 必须包含 model_state_dict。
        if not isinstance(state_dict, dict):
            raise ValueError("predictor checkpoint is missing model_state_dict")
        # 再读取并校验 metadata。
        metadata = self.read_checkpoint_metadata(path)
        self._validate_checkpoint_metadata(metadata)
        # 基于当前配置重建模型实例。
        model = self.build_model()
        # 加载 checkpoint 权重。
        model.load_state_dict(state_dict)
        # 返回模型和 metadata。
        return model, metadata

    def load_training_checkpoint(
        self,
        path: str | Path,
    ) -> tuple[
        FutureExpertPredictor,
        PredictorCheckpointMetadata,
        PredictorTrainingRunTrace | None,
        dict[str, Any] | None,
    ]:
        # 先读取 checkpoint 原始载荷。
        payload = self._read_checkpoint_payload(path)
        # 复用现有模型权重与 metadata 加载逻辑。
        model, metadata = self.load_checkpoint(path)
        # 再读取可选的 run_trace。
        run_trace = self.read_checkpoint_run_trace(path)
        if run_trace is not None:
            # 若存在 run_trace，则校验它与当前配置兼容。
            self._validate_resume_run_trace(run_trace)
        # 读取可选的 optimizer 状态。
        optimizer_state_dict = payload.get("optimizer_state_dict")
        if optimizer_state_dict is not None and not isinstance(
            optimizer_state_dict,
            dict,
        ):
            raise ValueError(
                "predictor checkpoint optimizer_state_dict must be a dictionary"
            )
        # 返回完整训练续训所需的全部状态。
        return model, metadata, run_trace, optimizer_state_dict

    @classmethod
    def export_checkpoint_bundle(
        cls,
        *,
        checkpoint_path: str | Path,
        output_dir: str | Path,
    ) -> PredictorDeploymentManifest:
        # 先读取 checkpoint 原始载荷。
        payload = cls._read_checkpoint_payload(checkpoint_path)
        # 提取模型参数字典。
        state_dict = payload.get("model_state_dict")
        # checkpoint 必须包含 model_state_dict。
        if not isinstance(state_dict, dict):
            raise ValueError("predictor checkpoint is missing model_state_dict")

        # 再恢复 checkpoint metadata、runtime schema 和指标摘要。
        metadata = cls.read_checkpoint_metadata(checkpoint_path)
        schema = PredictorRuntimeSchema.from_checkpoint_metadata(metadata)
        metrics = PredictorMetricsSummary.from_checkpoint_metadata(metadata)

        # 规范化 bundle 输出目录并预先创建。
        bundle_dir = Path(output_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # 约定 bundle 内各文件名。
        weights_path = bundle_dir / "predictor_weights.pt"
        schema_path = bundle_dir / "predictor_schema.json"
        metrics_path = bundle_dir / "predictor_metrics.json"
        manifest_path = bundle_dir / "predictor_bundle.json"

        # 写出权重文件。
        torch.save(
            {
                "weights_kind": "cfie_predictor_weights",
                "profile_name": metadata.profile_name,
                "model_state_dict": state_dict,
            },
            weights_path,
        )
        # 写出 schema 和 metrics 文件。
        schema.write_json(schema_path)
        metrics.write_json(metrics_path)

        # 组装 bundle manifest。
        manifest = PredictorDeploymentManifest(
            bundle_kind="cfie_predictor_deployment_bundle",
            profile_name=metadata.profile_name,
            teacher_source=metadata.teacher_source,
            summary_source=metadata.summary_source,
            source_checkpoint=Path(checkpoint_path).name,
            weights_kind="cfie_predictor_weights",
            weights_format="torch_state_dict",
            weights_file=weights_path.name,
            schema_kind=schema.schema_kind,
            schema_file=schema_path.name,
            metrics_kind=metrics.metrics_kind,
            metrics_file=metrics_path.name,
        )
        # 写出 manifest 并返回。
        manifest.write_json(manifest_path)
        return manifest

    def build_trace_dataset(
        self,
        *,
        steps: int,
        examples_per_step: int | None = None,
        samples: int = 2,
        tokens_per_sample: int = 256,
        dataset_path: str | None = None,
        tokenizer_path: str | None = None,
        dataset_format: str = "auto",
        dataset_text_key: str = "text",
    ) -> PredictorTraceDataset:
        # 先构造 batch planner。
        batch_planner = self._build_batch_planner(
            samples=samples,
            tokens_per_sample=tokens_per_sample,
            dataset_path=dataset_path,
            tokenizer_path=tokenizer_path,
            dataset_format=dataset_format,
            dataset_text_key=dataset_text_key,
        )
        # 再让 trace builder 生成样本。
        examples = self._trace_builder.build_examples(
            steps=steps,
            examples_per_step=examples_per_step,
            batch_planner=batch_planner,
        )
        # 组装 predictor trace 数据集对象。
        return PredictorTraceDataset(
            profile_name=self.config.profile_name,
            teacher_source=self.teacher_source,
            summary_source=self.summary_source,
            example_count=len(examples),
            window_layers=self.config.predictor_routing.window_layers,
            candidate_experts_per_layer=(
                self.config.predictor_routing.candidate_experts_per_layer
            ),
            executed_experts_per_layer=(
                self.config.predictor_routing.executed_experts_per_layer
            ),
            examples=examples,
        )

    def _dataset_tensors(
        self,
        dataset: PredictorTraceDataset,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 先把 hidden_summary 堆成特征张量。
        features = torch.tensor(
            [example.hidden_summary for example in dataset.examples],
            dtype=torch.float32,
            device="cpu",
        )
        # 再按 [example, future_layer, expert] 形状创建多热标签张量。
        targets = torch.zeros(
            (
                dataset.example_count,
                dataset.window_layers,
                self.config.model_spec.num_experts,
            ),
            dtype=torch.float32,
            device="cpu",
        )
        for example_index, example in enumerate(dataset.examples):
            for future_index, teacher_topk_ids in enumerate(
                example.future_teacher_topk_ids
            ):
                # teacher top-k expert 位置置为 1，形成多标签监督。
                targets[example_index, future_index, list(teacher_topk_ids)] = 1.0
        # 返回特征和标签张量。
        return features, targets

    def _mean_recall_at_budget(
        self,
        *,
        logits: torch.Tensor,
        targets: torch.Tensor,
        budget: int,
    ) -> float:
        # 先按预算取每个样本/未来层的 top-k experts。
        topk = torch.topk(logits, k=budget, dim=-1).indices
        # 取出 top-k 位置上的命中标签。
        matches = torch.gather(targets, dim=-1, index=topk)
        # 分母等于 teacher 正样本数，至少夹到 1。
        denom = targets.sum(dim=-1).clamp_min(1.0)
        # recall 等于 top-k 命中数除以 teacher 正样本数。
        recall = matches.sum(dim=-1) / denom
        # 返回全数据集平均 recall。
        return float(recall.mean().item())

    def _validate_dataset_compatibility(
        self,
        dataset: PredictorTraceDataset,
    ) -> None:
        # 数据集至少要有 1 条样本。
        if dataset.example_count < 1:
            raise ValueError("predictor trace dataset must contain at least 1 example")
        # 取第一条样本检查 hidden summary 维度。
        first_example = dataset.examples[0]
        if (
            len(first_example.hidden_summary)
            != self.config.predictor_trainer.input_summary_dim
        ):
            raise ValueError(
                "predictor trace dataset hidden summary size does not match "
                "predictor_trainer.input_summary_dim"
            )
        if dataset.window_layers != self.config.predictor_routing.window_layers:
            raise ValueError(
                "predictor trace dataset window_layers does not match "
                "predictor_routing.window_layers"
            )

    def evaluate_dataset(
        self,
        dataset: PredictorTraceDataset,
        *,
        model: FutureExpertPredictor | None = None,
        checkpoint_metadata: PredictorCheckpointMetadata | None = None,
    ) -> PredictorEvaluationTrace:
        # 先校验数据集兼容性。
        self._validate_dataset_compatibility(dataset)
        # 读取 routing 配置。
        routing_cfg = self.config.predictor_routing
        # 未显式给模型时，按当前配置重建模型。
        model = self.build_model() if model is None else model
        # 准备特征和标签张量。
        features, targets = self._dataset_tensors(dataset)
        with torch.no_grad():
            # 评估前切到 eval 模式。
            model.eval()
            # 前向得到 logits。
            logits = model(features)
            # 计算二元交叉熵损失。
            mean_loss = float(
                F.binary_cross_entropy_with_logits(logits, targets).item()
            )
            # 计算 candidate budget recall。
            candidate_recall = self._mean_recall_at_budget(
                logits=logits,
                targets=targets,
                budget=routing_cfg.candidate_experts_per_layer,
            )
            # 计算 executed budget recall。
            executed_recall = self._mean_recall_at_budget(
                logits=logits,
                targets=targets,
                budget=routing_cfg.executed_experts_per_layer,
            )
        # 组装评估结果对象。
        return PredictorEvaluationTrace(
            profile_name=self.config.profile_name,
            teacher_source=dataset.teacher_source,
            summary_source=dataset.summary_source,
            example_count=dataset.example_count,
            candidate_experts_per_layer=routing_cfg.candidate_experts_per_layer,
            executed_experts_per_layer=routing_cfg.executed_experts_per_layer,
            mean_loss=mean_loss,
            recall_at_candidate_budget=candidate_recall,
            recall_at_executed_budget=executed_recall,
            checkpoint_metadata=checkpoint_metadata,
        )

    def evaluate_checkpoint(
        self,
        *,
        checkpoint_path: str | Path,
        dataset: PredictorTraceDataset,
    ) -> PredictorEvaluationTrace:
        # 先加载 checkpoint 对应的模型和元信息。
        model, metadata = self.load_checkpoint(checkpoint_path)
        # 再对目标数据集执行评估。
        return self.evaluate_dataset(
            dataset,
            model=model,
            checkpoint_metadata=metadata,
        )

    def train_dataset(
        self,
        dataset: PredictorTraceDataset,
        *,
        epochs: int | None = None,
    ) -> PredictorTrainingRunTrace:
        # 复用 fit_dataset 训练，并只返回 run_trace。
        _, run_trace, _ = self.fit_dataset(
            dataset,
            epochs=epochs,
        )
        return run_trace

    def fit_dataset(
        self,
        dataset: PredictorTraceDataset,
        *,
        epochs: int | None = None,
        model: FutureExpertPredictor | None = None,
        optimizer_state_dict: dict[str, Any] | None = None,
        initial_run_trace: PredictorTrainingRunTrace | None = None,
    ) -> tuple[
        FutureExpertPredictor,
        PredictorTrainingRunTrace,
        dict[str, Any],
    ]:
        # 先校验数据集兼容性。
        self._validate_dataset_compatibility(dataset)
        # 若提供了历史 run_trace，则校验其与当前数据集可续接。
        if initial_run_trace is not None:
            self._validate_resume_dataset(dataset, initial_run_trace)
        # 读取 trainer 与 routing 配置。
        trainer_cfg = self.config.predictor_trainer
        routing_cfg = self.config.predictor_routing
        # 未显式给 epochs 时，退回配置中的默认值。
        epochs = trainer_cfg.epochs if epochs is None else int(epochs)
        # epoch 数至少为 1。
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        # 准备训练特征和标签张量。
        features, targets = self._dataset_tensors(dataset)
        # 固定随机种子，保证初始化和训练顺序可复现。
        torch.manual_seed(trainer_cfg.seed)
        # 未显式给模型时，按当前配置重建模型。
        model = self.build_model() if model is None else model
        # 使用 AdamW 作为优化器。
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=trainer_cfg.learning_rate,
            weight_decay=trainer_cfg.weight_decay,
        )
        # 若 checkpoint 里带有优化器状态，则在这里恢复。
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        # batch_size 不能超过样本总数。
        batch_size = min(trainer_cfg.batch_size, dataset.example_count)
        # 用列表累积逐 epoch 汇总，并接上历史 run_trace。
        epoch_summaries: list[PredictorEpochSummary] = (
            []
            if initial_run_trace is None
            else list(initial_run_trace.epoch_summaries)
        )
        # 续训时从历史已完成 epoch 数继续编号。
        completed_epochs = 0 if initial_run_trace is None else initial_run_trace.epochs

        for epoch_offset in range(epochs):
            # 当前 epoch 编号需要接在历史 run_trace 之后。
            epoch_index = completed_epochs + epoch_offset
            # 每个 epoch 用独立 generator 构造确定性的打乱顺序。
            generator = torch.Generator(device="cpu")
            generator.manual_seed(trainer_cfg.seed + epoch_index)
            order = torch.randperm(dataset.example_count, generator=generator)
            # 用于累计当前 epoch 的平均损失。
            total_loss = 0.0
            total_examples = 0
            for start in range(0, dataset.example_count, batch_size):
                # 取当前 mini-batch 的样本索引。
                batch_indices = order[start : start + batch_size]
                # 选出当前 batch 的特征和标签。
                batch_features = features.index_select(0, batch_indices)
                batch_targets = targets.index_select(0, batch_indices)
                # 前向得到 logits 并计算 BCE loss。
                logits = model(batch_features)
                loss = F.binary_cross_entropy_with_logits(logits, batch_targets)
                # 标准训练三步：清梯度、反向、更新。
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                # 以样本数加权累计当前 batch 的损失。
                batch_count = int(batch_features.shape[0])
                total_loss += float(loss.item()) * batch_count
                total_examples += batch_count

            with torch.no_grad():
                # 每个 epoch 结束后在全量数据上做一次评估。
                model.eval()
                all_logits = model(features)
                # 计算 candidate budget recall。
                candidate_recall = self._mean_recall_at_budget(
                    logits=all_logits,
                    targets=targets,
                    budget=routing_cfg.candidate_experts_per_layer,
                )
                # 计算 executed budget recall。
                executed_recall = self._mean_recall_at_budget(
                    logits=all_logits,
                    targets=targets,
                    budget=routing_cfg.executed_experts_per_layer,
                )
                # 评估完成后切回 train 模式。
                model.train()
            # 记录当前 epoch 的汇总结果。
            epoch_summaries.append(
                PredictorEpochSummary(
                    epoch_index=epoch_index,
                    mean_loss=total_loss / max(total_examples, 1),
                    recall_at_candidate_budget=candidate_recall,
                    recall_at_executed_budget=executed_recall,
                )
            )

        # 返回训练后的模型和完整训练 run_trace。
        return (
            model,
            PredictorTrainingRunTrace(
                profile_name=self.config.profile_name,
                teacher_source=dataset.teacher_source,
                summary_source=dataset.summary_source,
                example_count=dataset.example_count,
                epochs=completed_epochs + epochs,
                candidate_experts_per_layer=(
                    routing_cfg.candidate_experts_per_layer
                ),
                executed_experts_per_layer=(
                    routing_cfg.executed_experts_per_layer
                ),
                epoch_summaries=tuple(epoch_summaries),
            ),
            optimizer.state_dict(),
        )

    def train_synthetic(
        self,
        *,
        steps: int,
        examples_per_step: int | None = None,
        epochs: int | None = None,
    ) -> PredictorTrainingRunTrace:
        # 兼容旧调用名，直接复用通用 train()。
        return self.train(
            steps=steps,
            examples_per_step=examples_per_step,
            epochs=epochs,
        )

    def train(
        self,
        *,
        steps: int,
        examples_per_step: int | None = None,
        epochs: int | None = None,
        samples: int = 2,
        tokens_per_sample: int = 256,
        dataset_path: str | None = None,
        tokenizer_path: str | None = None,
        dataset_format: str = "auto",
        dataset_text_key: str = "text",
    ) -> PredictorTrainingRunTrace:
        # 先构造 predictor trace 数据集。
        dataset = self.build_trace_dataset(
            steps=steps,
            examples_per_step=examples_per_step,
            samples=samples,
            tokens_per_sample=tokens_per_sample,
            dataset_path=dataset_path,
            tokenizer_path=tokenizer_path,
            dataset_format=dataset_format,
            dataset_text_key=dataset_text_key,
        )
        # 再对该数据集执行训练。
        return self.train_dataset(
            dataset,
            epochs=epochs,
        )
