"""Bounded predictor trainer for candidate-routed MoE experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from cfie_training.config import TrainingProjectConfig
from cfie_training.runtime.data import TokenizedDatasetBatchPlanner
from cfie_training.runtime.types import BatchShape


class PredictorBatchPlanner(Protocol):
    # 返回指定 step 使用的 batch 形状。
    def batch_for_step(self, step_index: int) -> BatchShape:
        ...


@dataclass(slots=True, frozen=True)
class CapturedForwardBatch:
    layer_hidden_states: tuple[torch.Tensor, ...]
    layer_router_logits: tuple[torch.Tensor, ...]


class PredictorTeacherModelBackend(Protocol):
    teacher_source: str
    summary_source: str

    def capture_batch(self, batch: BatchShape) -> CapturedForwardBatch:
        ...


class TransformersRouterTeacherModelBackend:
    teacher_source = "forward_router"
    summary_source = "forward_hidden_state"

    def __init__(self, config: TrainingProjectConfig) -> None:
        from transformers import AutoModelForCausalLM

        self._config = config
        self._model_path = str(config.model_source.model_path).strip()
        if not self._model_path:
            raise ValueError(
                "model_source.model_path is required for forward-capture traces"
            )
        self._device = self._resolve_device()
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": "auto",
        }
        if self._device.type == "cuda":
            model_kwargs["device_map"] = "auto"
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            **model_kwargs,
        )
        if self._device.type != "cuda":
            self._model.to(self._device)
        self._model.eval()
        self._captured_hidden_states: dict[int, torch.Tensor] = {}
        self._captured_router_logits: dict[int, torch.Tensor] = {}
        self._capture_enabled = False
        self._hook_handles = [
            module.register_forward_hook(self._build_capture_hook(layer_index))
            for layer_index, _, module in self._discover_router_modules()
        ]

    def _resolve_device(self) -> torch.device:
        if (
            self._config.execution.compute_device == "gpu"
            and torch.cuda.is_available()
        ):
            return torch.device("cuda")
        return torch.device("cpu")

    def _discover_router_modules(
        self,
    ) -> tuple[tuple[int, str, nn.Module], ...]:
        router_modules: dict[int, tuple[str, nn.Module]] = {}
        layer_pattern = re.compile(r"(?:^|\\.)layers\\.(\\d+)(?:\\.|$)")
        for name, module in self._model.named_modules():
            if ".gate" not in name:
                continue
            layer_match = layer_pattern.search(name)
            if layer_match is None:
                continue
            output_dim = self._module_output_dim(module)
            if output_dim != self._config.model_spec.num_experts:
                continue
            layer_index = int(layer_match.group(1))
            router_modules[layer_index] = (name, module)
        if not router_modules:
            raise ValueError(
                "could not discover any router gate modules from the teacher model"
            )
        return tuple(
            (layer_index, *router_modules[layer_index])
            for layer_index in sorted(router_modules)
        )

    @staticmethod
    def _module_output_dim(module: nn.Module) -> int | None:
        out_features = getattr(module, "out_features", None)
        if isinstance(out_features, int):
            return out_features
        weight = getattr(module, "weight", None)
        if torch.is_tensor(weight) and weight.ndim >= 2:
            return int(weight.shape[0])
        return None

    @staticmethod
    def _flatten_token_rows(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        if tensor.ndim >= 2:
            return tensor.reshape(-1, tensor.shape[-1])
        raise ValueError("captured router tensors must have rank >= 1")

    def _build_capture_hook(self, layer_index: int):
        def hook(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            del module
            if not self._capture_enabled:
                return
            if not inputs:
                raise ValueError(
                    f"router module for layer {layer_index} did not receive inputs"
                )
            hidden_states = self._flatten_token_rows(inputs[0]).detach()
            router_logits = output[0] if isinstance(output, tuple) else output
            router_logits = self._flatten_token_rows(router_logits).detach()
            self._captured_hidden_states[layer_index] = hidden_states.to(
                device="cpu",
                dtype=torch.float32,
            )
            self._captured_router_logits[layer_index] = router_logits.to(
                device="cpu",
                dtype=torch.float32,
            )

        return hook

    def capture_batch(self, batch: BatchShape) -> CapturedForwardBatch:
        if not batch.has_token_rows:
            raise ValueError(
                "forward-capture traces require BatchShape.token_rows to be populated"
            )
        input_ids = torch.tensor(
            batch.token_rows,
            dtype=torch.long,
            device=self._device,
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        self._captured_hidden_states.clear()
        self._captured_router_logits.clear()
        self._capture_enabled = True
        try:
            with torch.no_grad():
                self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
        finally:
            self._capture_enabled = False
        num_layers = self._config.model_spec.num_hidden_layers
        missing_layers = [
            layer_index
            for layer_index in range(num_layers)
            if layer_index not in self._captured_hidden_states
            or layer_index not in self._captured_router_logits
        ]
        if missing_layers:
            raise ValueError(
                "teacher model forward capture missed router layers: "
                + ", ".join(str(layer_index) for layer_index in missing_layers[:8])
            )
        return CapturedForwardBatch(
            layer_hidden_states=tuple(
                self._captured_hidden_states[layer_index]
                for layer_index in range(num_layers)
            ),
            layer_router_logits=tuple(
                self._captured_router_logits[layer_index]
                for layer_index in range(num_layers)
            ),
        )


@dataclass(slots=True, frozen=True)
class PredictorTraceExample:
    example_index: int
    step_index: int
    token_index: int
    insertion_layer_index: int
    future_layer_indices: tuple[int, ...]
    hidden_state: tuple[float, ...]
    future_teacher_topk_ids: tuple[tuple[int, ...], ...]

    # 将单条 predictor trace 样本序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出样本编号、插入层、hidden_state 和未来 teacher top-k 标签。
        return {
            "example_index": self.example_index,
            "step_index": self.step_index,
            "token_index": self.token_index,
            "insertion_layer_index": self.insertion_layer_index,
            "future_layer_indices": list(self.future_layer_indices),
            "hidden_state": list(self.hidden_state),
            "future_teacher_topk_ids": [
                list(expert_ids) for expert_ids in self.future_teacher_topk_ids
            ],
        }

    @classmethod
    # 从字典恢复单条 predictor trace 样本。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorTraceExample":
        # 逐字段恢复样本索引、未来层索引、hidden_state 和 top-k 标签；
        # 同时兼容旧版数据里的 hidden_summary 字段。
        return cls(
            example_index=int(payload["example_index"]),
            step_index=int(payload["step_index"]),
            token_index=int(payload.get("token_index", 0)),
            insertion_layer_index=int(payload["insertion_layer_index"]),
            future_layer_indices=tuple(
                int(layer_index) for layer_index in payload["future_layer_indices"]
            ),
            hidden_state=tuple(
                float(value)
                for value in payload.get(
                    "hidden_state",
                    payload.get("hidden_summary", ()),
                )
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
            teacher_source=str(payload.get("teacher_source", "forward_router")),
            summary_source=str(
                payload.get("summary_source", "forward_hidden_state")
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
        super().__init__()
        self.window_layers = window_layers
        self.num_experts = num_experts
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_proj = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, window_layers * num_experts),
        )

    @staticmethod
    def _normalize_hidden_state(hidden_state: torch.Tensor) -> torch.Tensor:
        if hidden_state.ndim == 1:
            return hidden_state.unsqueeze(0)
        if hidden_state.ndim != 2:
            raise ValueError("hidden_state must be rank-1 or rank-2")
        return hidden_state

    @staticmethod
    def _normalize_layer_index(
        layer_index: int | torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(layer_index, int):
            return torch.full(
                (batch_size,),
                int(layer_index),
                dtype=torch.float32,
                device=device,
            )
        if layer_index.ndim == 0:
            return torch.full(
                (batch_size,),
                int(layer_index.item()),
                dtype=torch.float32,
                device=device,
            )
        if layer_index.ndim != 1 or int(layer_index.shape[0]) != batch_size:
            raise ValueError("layer_index must be scalar or match batch size")
        return layer_index.to(device=device, dtype=torch.float32)

    @staticmethod
    def _layer_features(layer_index: torch.Tensor) -> torch.Tensor:
        layer_index = layer_index.unsqueeze(-1)
        return torch.cat(
            (
                layer_index,
                layer_index.square(),
                torch.sin(layer_index * 0.1),
                torch.cos(layer_index * 0.1),
                torch.sin(layer_index * 0.01),
                torch.cos(layer_index * 0.01),
            ),
            dim=-1,
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        layer_index: int | torch.Tensor,
    ) -> torch.Tensor:
        hidden_state = self._normalize_hidden_state(hidden_state)
        layer_index = self._normalize_layer_index(
            layer_index,
            batch_size=int(hidden_state.shape[0]),
            device=hidden_state.device,
        )
        fused_hidden = self.input_proj(hidden_state) + self.layer_proj(
            self._layer_features(layer_index)
        )
        logits = self.net(fused_hidden)
        return logits.view(-1, self.window_layers, self.num_experts)


HiddenStateFutureExpertPredictor = FutureExpertPredictor


class PredictorTraceBuilderBase:
    teacher_source = "forward_router"
    summary_source = "forward_hidden_state"

    def __init__(self, config: TrainingProjectConfig) -> None:
        self.config = config.validate()
        if not self.config.model_spec.is_defined():
            raise ValueError(
                "predictor trace generation requires a defined model_spec"
            )
        self._hidden_dim = self.config.model_spec.hidden_size
        self._num_layers = self.config.model_spec.num_hidden_layers
        self._window_layers = self.config.predictor_routing.window_layers
        self._executed_experts = (
            self.config.predictor_routing.executed_experts_per_layer
        )
        self._stride_layers = self.config.predictor_routing.stride_layers
        self._insertion_layer_indices = tuple(
            range(
                0,
                max(self._num_layers - self._window_layers, 0),
                self._stride_layers,
            )
        )

    def _resolve_examples_per_step(
        self,
        examples_per_step: int | None,
    ) -> int:
        resolved = (
            self.config.predictor_trainer.examples_per_step
            if examples_per_step is None
            else int(examples_per_step)
        )
        if resolved < 1:
            raise ValueError("examples_per_step must be >= 1")
        return resolved

    def _selected_example_specs(
        self,
        *,
        step_index: int,
        examples_per_step: int,
        token_count: int,
    ) -> tuple[tuple[int, int, int], ...]:
        if not self._insertion_layer_indices:
            raise ValueError(
                "predictor trace generation requires at least one eligible insertion layer"
            )
        token_count = max(int(token_count), 1)
        total_slots = len(self._insertion_layer_indices) * token_count
        example_count = min(examples_per_step, total_slots)
        start = (step_index * example_count) % total_slots
        specs: list[tuple[int, int, int]] = []
        for example_offset in range(example_count):
            flat_index = (start + example_offset) % total_slots
            token_index = flat_index // len(self._insertion_layer_indices)
            insertion_index = flat_index % len(self._insertion_layer_indices)
            specs.append(
                (
                    example_offset,
                    self._insertion_layer_indices[insertion_index],
                    token_index,
                )
            )
        return tuple(specs)

    def _future_layer_indices(
        self,
        insertion_layer_index: int,
    ) -> tuple[int, ...]:
        return tuple(
            insertion_layer_index + offset + 1
            for offset in range(self._window_layers)
        )


class ForwardCaptureTraceBuilder(PredictorTraceBuilderBase):
    def __init__(
        self,
        config: TrainingProjectConfig,
        teacher_model_backend: PredictorTeacherModelBackend,
    ) -> None:
        super().__init__(config)
        self.teacher_source = teacher_model_backend.teacher_source
        self.summary_source = teacher_model_backend.summary_source
        self._teacher_model_backend = teacher_model_backend

    def build_examples(
        self,
        *,
        steps: int,
        examples_per_step: int | None = None,
        batch_planner: PredictorBatchPlanner | None = None,
    ) -> tuple[PredictorTraceExample, ...]:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if batch_planner is None:
            raise ValueError(
                "forward-capture predictor traces require a batch planner"
            )
        resolved_examples_per_step = self._resolve_examples_per_step(
            examples_per_step
        )
        examples: list[PredictorTraceExample] = []
        for step_index in range(steps):
            batch = batch_planner.batch_for_step(step_index)
            captured = self._teacher_model_backend.capture_batch(batch)
            if len(captured.layer_hidden_states) != self._num_layers:
                raise ValueError(
                    "captured hidden-state layer count does not match model_spec.num_hidden_layers"
                )
            if len(captured.layer_router_logits) != self._num_layers:
                raise ValueError(
                    "captured router-logit layer count does not match model_spec.num_hidden_layers"
                )
            token_count = int(captured.layer_hidden_states[0].shape[0])
            for _, insertion_layer_index, token_index in self._selected_example_specs(
                step_index=step_index,
                examples_per_step=resolved_examples_per_step,
                token_count=token_count,
            ):
                hidden_state = captured.layer_hidden_states[insertion_layer_index][
                    token_index
                ]
                if hidden_state.numel() != self._hidden_dim:
                    raise ValueError(
                        "captured hidden_state size does not match model_spec.hidden_size"
                    )
                future_layer_indices = self._future_layer_indices(
                    insertion_layer_index
                )
                future_teacher_topk_ids = []
                for future_layer_index in future_layer_indices:
                    teacher_topk = torch.topk(
                        captured.layer_router_logits[future_layer_index][token_index],
                        k=self._executed_experts,
                        dim=0,
                    ).indices
                    future_teacher_topk_ids.append(
                        tuple(int(expert_id) for expert_id in teacher_topk.tolist())
                    )
                examples.append(
                    PredictorTraceExample(
                        example_index=len(examples),
                        step_index=step_index,
                        token_index=token_index,
                        insertion_layer_index=insertion_layer_index,
                        future_layer_indices=future_layer_indices,
                        hidden_state=tuple(
                            float(value) for value in hidden_state.tolist()
                        ),
                        future_teacher_topk_ids=tuple(future_teacher_topk_ids),
                    )
                )
        return tuple(examples)


class PredictorTrainer:
    def __init__(
        self,
        config: TrainingProjectConfig,
        *,
        teacher_model_backend: PredictorTeacherModelBackend | None = None,
    ) -> None:
        # -------------------- 校验并缓存训练配置 --------------------
        # 训练器构造完成后会立即依赖 model_spec、routing 与 trainer 超参，
        # 因此这里先做一次全量配置校验，避免带着不完整配置进入后续流程。
        self.config = config.validate()

        # predictor 训练需要明确知道层数、expert 数和 hidden 维度；
        # 若 model_spec 未定义，就无法构造样本与模型结构。
        if not self.config.model_spec.is_defined():
            raise ValueError("predictor training requires a defined model_spec")

        self._teacher_model_backend = teacher_model_backend
        self._trace_builder: ForwardCaptureTraceBuilder | None = None

    @property
    # 返回当前训练器使用的 teacher source。
    def teacher_source(self) -> str:
        return self._resolve_trace_builder().teacher_source

    @property
    # 返回当前训练器使用的 summary source。
    def summary_source(self) -> str:
        return self._resolve_trace_builder().summary_source

    def _resolve_teacher_model_backend(
        self,
    ) -> PredictorTeacherModelBackend:
        if self._teacher_model_backend is None:
            self._teacher_model_backend = TransformersRouterTeacherModelBackend(
                self.config
            )
        return self._teacher_model_backend

    def _resolve_trace_builder(self) -> ForwardCaptureTraceBuilder:
        if self._trace_builder is None:
            self._trace_builder = ForwardCaptureTraceBuilder(
                self.config,
                self._resolve_teacher_model_backend(),
            )
        return self._trace_builder

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
        if dataset_path is None:
            raise ValueError(
                "predictor trace generation requires a dataset-backed batch planner; "
                "pass dataset_path/--dataset"
            )
        return TokenizedDatasetBatchPlanner(
            config=self.config,
            dataset_path=dataset_path,
            base_samples=samples,
            tokens_per_sample=tokens_per_sample,
            tokenizer_path=tokenizer_path,
            dataset_format=dataset_format,
            dataset_text_key=dataset_text_key,
        )

    def build_model(self) -> FutureExpertPredictor:
        # -------------------- 读取决定 predictor 形状的关键配置 --------------------
        # trainer 配置决定隐藏层宽度；
        # 输入维度当前直接取 model hidden_size。
        trainer_cfg = self.config.predictor_trainer
        # routing 配置决定未来窗口层数，也就决定输出张量的第二维大小。
        routing_cfg = self.config.predictor_routing

        # model_spec 里的 num_experts 决定每个未来层要预测多少个 expert logit。
        # 这里三类配置共同决定 predictor 的完整张量形状。
        return FutureExpertPredictor(
            input_dim=self.config.model_spec.hidden_size,
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
            input_summary_dim=self.config.model_spec.hidden_size,
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
            input_summary_dim=self.config.model_spec.hidden_size,
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
        if metadata.input_summary_dim != self.config.model_spec.hidden_size:
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
        # -------------------- 构造样本批规划器 --------------------
        # batch planner 负责给每个训练 step 产出带真实 token rows 的抽象 batch 形状；
        # forward-capture trace 会消费这些 token rows 直接运行 teacher forward。
        batch_planner = self._build_batch_planner(
            samples=samples,
            tokens_per_sample=tokens_per_sample,
            dataset_path=dataset_path,
            tokenizer_path=tokenizer_path,
            dataset_format=dataset_format,
            dataset_text_key=dataset_text_key,
        )

        # -------------------- 生成 predictor 监督样本 --------------------
        # trace builder 会产出每条样本的 hidden_state，
        # 以及对应 future window 的 teacher top-k experts。
        trace_builder = self._resolve_trace_builder()
        examples = trace_builder.build_examples(
            steps=steps,
            examples_per_step=examples_per_step,
            batch_planner=batch_planner,
        )

        # -------------------- 打包为数据集对象 --------------------
        # 除了样本本身，还把 teacher/source、窗口长度和预算元信息一起固化下来，
        # 便于后续训练、评估和 checkpoint 兼容性校验。
        return PredictorTraceDataset(
            profile_name=self.config.profile_name,
            teacher_source=trace_builder.teacher_source,
            summary_source=trace_builder.summary_source,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # -------------------- 物化特征张量 --------------------
        # 每条样本里的 hidden_state 都是定长 tuple；
        # 这里把它们堆成 [example_count, hidden_size] 的 CPU 浮点张量。
        features = torch.tensor(
            [example.hidden_state for example in dataset.examples],
            dtype=torch.float32,
            device="cpu",
        )
        layer_indices = torch.tensor(
            [example.insertion_layer_index for example in dataset.examples],
            dtype=torch.float32,
            device="cpu",
        )

        # -------------------- 物化多标签目标张量 --------------------
        # predictor 要同时预测“未来多层 x 每层多个 expert”，
        # 因此目标不是单类别标签，而是 [example, future_layer, expert] 的多热张量。
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
                # teacher 选中的 top-k experts 位置置为 1；
                # 这样后续 BCEWithLogitsLoss 就能把它当作多标签监督来训练。
                targets[example_index, future_index, list(teacher_topk_ids)] = 1.0

        # 返回训练和评估都会复用的特征/标签张量对。
        return features, layer_indices, targets

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
        # 取第一条样本检查 hidden_state 维度。
        first_example = dataset.examples[0]
        if (
            len(first_example.hidden_state)
            != self.config.model_spec.hidden_size
        ):
            raise ValueError(
                "predictor trace dataset hidden_state size does not match "
                "model_spec.hidden_size"
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
        # -------------------- 校验输入并准备评估对象 --------------------
        # 评估前先确认数据集维度与当前 trainer 配置兼容，避免 silent mismatch。
        self._validate_dataset_compatibility(dataset)
        # routing 配置里的候选预算和执行预算会直接决定 recall 指标的取值口径。
        routing_cfg = self.config.predictor_routing
        # 若调用方未传入模型，就按当前配置构造一个新 predictor 实例。
        model = self.build_model() if model is None else model
        # 把 trace 数据集转换成可直接前向的张量形式。
        features, layer_indices, targets = self._dataset_tensors(dataset)

        # -------------------- 关闭梯度并执行前向评估 --------------------
        with torch.no_grad():
            # 评估阶段不需要 dropout/batchnorm 等训练时行为，因此切到 eval 模式。
            model.eval()
            # 前向得到每个 future layer、每个 expert 的原始 logits。
            logits = model(features, layer_indices)
            # 标签是多热而不是单类别，因此这里使用 BCEWithLogitsLoss 而不是 softmax CE。
            mean_loss = float(
                F.binary_cross_entropy_with_logits(logits, targets).item()
            )
            # 按候选预算统计 recall，用来度量“给推理侧预热多少 experts”时的覆盖能力。
            candidate_recall = self._mean_recall_at_budget(
                logits=logits,
                targets=targets,
                budget=routing_cfg.candidate_experts_per_layer,
            )
            # 按执行预算统计 recall，用来度量“最终真正会执行的 experts”覆盖得如何。
            executed_recall = self._mean_recall_at_budget(
                logits=logits,
                targets=targets,
                budget=routing_cfg.executed_experts_per_layer,
            )

        # -------------------- 打包评估结果 --------------------
        # 把本次评估的 loss、recall 和可选 checkpoint 元信息固化为 trace，供日志和导出使用。
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
        # train_dataset 是 fit_dataset 的轻包装：
        # 调用方若只关心训练轨迹而不关心模型对象和优化器状态，就走这里。
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
        # -------------------- 校验训练输入并解析超参数 --------------------
        # 训练前先做数据集维度校验，避免把不匹配的 summary 或窗口长度送进模型。
        self._validate_dataset_compatibility(dataset)
        # 若当前是续训，还要确认历史 run_trace 与新数据集来源一致，避免接错 checkpoint。
        if initial_run_trace is not None:
            self._validate_resume_dataset(dataset, initial_run_trace)
        # trainer 配置负责优化超参，routing 配置负责 recall 预算口径。
        trainer_cfg = self.config.predictor_trainer
        routing_cfg = self.config.predictor_routing
        # 若调用方未显式给 epochs，就沿用配置默认值。
        epochs = trainer_cfg.epochs if epochs is None else int(epochs)
        # 训练轮数至少为 1；0 轮训练没有意义，也无法产出新的 run_trace。
        if epochs < 1:
            raise ValueError("epochs must be >= 1")

        # -------------------- 物化张量并初始化训练对象 --------------------
        # 先把 trace 数据集物化成 CPU 张量，后续训练循环直接复用，避免重复构造。
        features, layer_indices, targets = self._dataset_tensors(dataset)
        # 固定随机种子，保证模型初始化和 epoch 内打乱顺序可复现。
        torch.manual_seed(trainer_cfg.seed)
        # 若调用方未传入模型，就按当前 trainer 配置构造一个新 predictor。
        model = self.build_model() if model is None else model
        # 训练目前使用 AdamW；这里只优化 predictor 自身参数，不涉及主模型参数。
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=trainer_cfg.learning_rate,
            weight_decay=trainer_cfg.weight_decay,
        )
        # 若本次是从 checkpoint 续训，就把优化器状态一并恢复，保持动量连续。
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        # batch_size 不能超过样本总数，避免最后构造空 batch。
        batch_size = min(trainer_cfg.batch_size, dataset.example_count)
        # epoch_summaries 既承载本轮训练结果，也在续训场景下承接历史轨迹。
        epoch_summaries: list[PredictorEpochSummary] = (
            []
            if initial_run_trace is None
            else list(initial_run_trace.epoch_summaries)
        )
        # 续训时，新的 epoch 编号要接在历史已完成 epoch 之后。
        completed_epochs = 0 if initial_run_trace is None else initial_run_trace.epochs

        # -------------------- 执行逐 epoch 的训练循环 --------------------
        for epoch_offset in range(epochs):
            # 当前 epoch 的逻辑编号需要考虑续训场景下的历史偏移量。
            epoch_index = completed_epochs + epoch_offset
            # 每个 epoch 使用独立 generator 打乱顺序；
            # 这样既保证可复现，又能让不同 epoch 的样本顺序不同。
            generator = torch.Generator(device="cpu")
            generator.manual_seed(trainer_cfg.seed + epoch_index)
            order = torch.randperm(dataset.example_count, generator=generator)
            # 累积当前 epoch 的样本加权损失，后面再除以总样本数得到 mean_loss。
            total_loss = 0.0
            total_examples = 0

            # -------------------- 执行逐 mini-batch 的前向、反向与更新 --------------------
            for start in range(0, dataset.example_count, batch_size):
                # 根据打乱后的顺序切出当前 mini-batch 的样本索引。
                batch_indices = order[start : start + batch_size]
                # 从全量张量中抽取当前 batch 的特征和标签。
                batch_features = features.index_select(0, batch_indices)
                batch_layer_indices = layer_indices.index_select(0, batch_indices)
                batch_targets = targets.index_select(0, batch_indices)
                # 前向得到每个 future layer / expert 的 logits。
                logits = model(batch_features, batch_layer_indices)
                # 目标是多标签预测，因此这里按多热标签计算 BCEWithLogitsLoss。
                loss = F.binary_cross_entropy_with_logits(logits, batch_targets)
                # 标准训练三步：清梯度、反向传播、参数更新。
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                # 用样本数加权累计 batch loss，避免最后一个不满 batch 时均值失真。
                batch_count = int(batch_features.shape[0])
                total_loss += float(loss.item()) * batch_count
                total_examples += batch_count

            # -------------------- 在全量数据上计算 epoch 级指标 --------------------
            with torch.no_grad():
                # 训练完一个 epoch 后，在全量数据上做一次统一评估，记录可比较的指标。
                model.eval()
                all_logits = model(features, layer_indices)
                # candidate recall 衡量“给推理侧准备的候选 experts”覆盖了多少 teacher experts。
                candidate_recall = self._mean_recall_at_budget(
                    logits=all_logits,
                    targets=targets,
                    budget=routing_cfg.candidate_experts_per_layer,
                )
                # executed recall 衡量“最终真正允许执行的 experts”能覆盖多少 teacher experts。
                executed_recall = self._mean_recall_at_budget(
                    logits=all_logits,
                    targets=targets,
                    budget=routing_cfg.executed_experts_per_layer,
                )
                # 评估完成后切回 train 模式，保证下一个 epoch 继续按训练模式运行。
                model.train()

            # 把本轮 epoch 的损失和 recall 汇总成一条稳定记录，供日志、导出与续训使用。
            epoch_summaries.append(
                PredictorEpochSummary(
                    epoch_index=epoch_index,
                    mean_loss=total_loss / max(total_examples, 1),
                    recall_at_candidate_budget=candidate_recall,
                    recall_at_executed_budget=executed_recall,
                )
            )

        # -------------------- 返回训练产物 --------------------
        # fit_dataset 会同时返回：
        # 1) 训练后的 predictor 模型
        # 2) 完整训练轨迹 run_trace
        # 3) 优化器状态，便于后续继续续训
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
        # -------------------- 先构造训练数据集 --------------------
        # train() 是最上层入口：它先根据 step 数和数据源参数生成 predictor trace 数据集。
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

        # -------------------- 再委托给数据集训练主链 --------------------
        # 数据集构造和参数更新解耦后，调用方既可以走 train() 一步到位，
        # 也可以先单独生成 dataset，再调用 train_dataset()/fit_dataset()。
        return self.train_dataset(
            dataset,
            epochs=epochs,
        )
