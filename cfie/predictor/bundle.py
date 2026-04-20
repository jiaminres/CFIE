"""Predictor deployment bundle loading for CFIE inference runtime."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


_BUNDLE_KIND = "cfie_predictor_deployment_bundle"
_SCHEMA_KIND = "cfie_predictor_runtime_schema"
_METRICS_KIND = "cfie_predictor_metrics_summary"
_WEIGHTS_KIND = "cfie_predictor_weights"
_WEIGHTS_FORMAT = "torch_state_dict"
_DEFAULT_MANIFEST_FILE = "predictor_bundle.json"


def _require_non_empty_string(name: str, value: Any) -> str:
    # 统一校验字符串字段，避免运行时再处理空值。
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    # 返回去首尾空白后的稳定值。
    return value.strip()


def _require_int(name: str, value: Any, *, allow_zero: bool = False) -> int:
    # 先把输入统一收敛成整数。
    parsed = int(value)
    # 再校验是否满足正整数约束。
    lower_bound = 0 if allow_zero else 1
    if parsed < lower_bound:
        comparator = ">=" if allow_zero else ">"
        raise ValueError(f"{name} must be {comparator} 0")
    # 返回已经校验过的整数值。
    return parsed


def _require_float(name: str, value: Any) -> float:
    # 将数值字段统一转成浮点，便于后续做区间校验。
    return float(value)


def _require_ratio(name: str, value: Any) -> float:
    # 先解析成浮点值。
    parsed = _require_float(name, value)
    # recall 这类指标必须落在 [0, 1]。
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    # 返回已校验值。
    return parsed


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    # 文件必须存在，否则 bundle 不完整。
    if not path.is_file():
        raise FileNotFoundError(f"{description} file does not exist: {path}")
    # 读取并解析 JSON 文本。
    payload = json.loads(path.read_text(encoding="utf-8"))
    # 顶层必须是对象，不能是数组或标量。
    if not isinstance(payload, dict):
        raise ValueError(f"{description} JSON must decode to an object")
    # 返回原始对象字典。
    return payload


def _resolve_manifest_path(bundle_path: str | Path) -> Path:
    # 允许直接传 bundle 目录或 manifest 文件。
    resolved = Path(bundle_path)
    if resolved.is_dir():
        # 目录输入时，默认寻找约定文件名的 manifest。
        resolved = resolved / _DEFAULT_MANIFEST_FILE
    # 返回最终 manifest 路径。
    return resolved


@dataclass(slots=True, frozen=True)
class PredictorRuntimeSchema:
    schema_kind: str
    profile_name: str
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

    # 校验 runtime schema 的结构与候选池约束。
    def validate(self) -> "PredictorRuntimeSchema":
        # schema kind 必须与训练侧导出契约一致。
        if self.schema_kind != _SCHEMA_KIND:
            raise ValueError(
                f"schema_kind must be {_SCHEMA_KIND!r}, got {self.schema_kind!r}"
            )
        # 关键来源字段必须非空，便于后续做跨文件一致性校验。
        _require_non_empty_string("profile_name", self.profile_name)
        _require_non_empty_string("selection_mode", self.selection_mode)
        _require_non_empty_string("online_expert_source", self.online_expert_source)
        # 维度与窗口字段都必须为正。
        _require_int("input_summary_dim", self.input_summary_dim)
        _require_int("predictor_hidden_dim", self.predictor_hidden_dim)
        _require_int("window_layers", self.window_layers)
        _require_int("stride_layers", self.stride_layers)
        _require_int("num_experts", self.num_experts)
        _require_int(
            "candidate_experts_per_layer",
            self.candidate_experts_per_layer,
        )
        _require_int(
            "executed_experts_per_layer",
            self.executed_experts_per_layer,
        )
        # 执行预算不能超过候选预算，也不能超过模型 expert 总数。
        if self.candidate_experts_per_layer < self.executed_experts_per_layer:
            raise ValueError(
                "candidate_experts_per_layer must be >= executed_experts_per_layer"
            )
        if self.candidate_experts_per_layer > self.num_experts:
            raise ValueError(
                "candidate_experts_per_layer must be <= num_experts"
            )
        # 返回自身，便于链式使用。
        return self

    # 将 runtime schema 转成字典，供测试与调试输出使用。
    def to_dict(self) -> dict[str, Any]:
        # 直接返回稳定字段映射。
        return {
            "schema_kind": self.schema_kind,
            "profile_name": self.profile_name,
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

    @classmethod
    # 从字典恢复 runtime schema，并立即做约束校验。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorRuntimeSchema":
        # 逐字段完成类型归一化。
        schema = cls(
            schema_kind=str(payload.get("schema_kind", _SCHEMA_KIND)),
            profile_name=str(payload["profile_name"]),
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
        # 返回已通过校验的 schema。
        return schema.validate()

    @classmethod
    # 从 JSON 文件恢复 runtime schema。
    def from_json_file(cls, path: str | Path) -> "PredictorRuntimeSchema":
        # 先读取 JSON 对象。
        payload = _read_json_object(Path(path), description="predictor schema")
        # 再按字典格式恢复。
        return cls.from_dict(payload)


@dataclass(slots=True, frozen=True)
class PredictorMetricsSummary:
    metrics_kind: str
    profile_name: str
    example_count: int
    epochs: int
    final_mean_loss: float
    final_recall_at_candidate_budget: float
    final_recall_at_executed_budget: float

    # 校验 metrics 摘要的基础结构和指标区间。
    def validate(self) -> "PredictorMetricsSummary":
        # metrics kind 必须与训练侧导出格式一致。
        if self.metrics_kind != _METRICS_KIND:
            raise ValueError(
                f"metrics_kind must be {_METRICS_KIND!r}, got {self.metrics_kind!r}"
            )
        # 关键来源字段必须可用于跨文件对齐。
        _require_non_empty_string("profile_name", self.profile_name)
        # 样本数与 epoch 数允许为 0，但不能为负。
        _require_int("example_count", self.example_count, allow_zero=True)
        _require_int("epochs", self.epochs, allow_zero=True)
        # loss 只要求能解析成浮点。
        _require_float("final_mean_loss", self.final_mean_loss)
        # recall 指标必须落在标准比例区间。
        _require_ratio(
            "final_recall_at_candidate_budget",
            self.final_recall_at_candidate_budget,
        )
        _require_ratio(
            "final_recall_at_executed_budget",
            self.final_recall_at_executed_budget,
        )
        # 返回自身，便于链式使用。
        return self

    # 将 metrics 摘要转成字典。
    def to_dict(self) -> dict[str, Any]:
        # 直接返回稳定字段映射。
        return {
            "metrics_kind": self.metrics_kind,
            "profile_name": self.profile_name,
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
    # 从字典恢复 metrics 摘要。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorMetricsSummary":
        # 逐字段完成类型归一化。
        summary = cls(
            metrics_kind=str(payload.get("metrics_kind", _METRICS_KIND)),
            profile_name=str(payload["profile_name"]),
            example_count=int(payload["example_count"]),
            epochs=int(payload["epochs"]),
            final_mean_loss=float(payload["final_mean_loss"]),
            final_recall_at_candidate_budget=float(
                payload["final_recall_at_candidate_budget"]
            ),
            final_recall_at_executed_budget=float(
                payload["final_recall_at_executed_budget"]
            ),
        )
        # 返回已通过校验的 metrics 对象。
        return summary.validate()

    @classmethod
    # 从 JSON 文件恢复 metrics 摘要。
    def from_json_file(cls, path: str | Path) -> "PredictorMetricsSummary":
        # 先读取 JSON 对象。
        payload = _read_json_object(Path(path), description="predictor metrics")
        # 再按字典格式恢复。
        return cls.from_dict(payload)


@dataclass(slots=True, frozen=True)
class PredictorDeploymentManifest:
    bundle_kind: str
    profile_name: str
    source_checkpoint: str
    weights_kind: str
    weights_format: str
    weights_file: str
    schema_kind: str
    schema_file: str
    metrics_kind: str
    metrics_file: str

    # 校验 manifest 中的 bundle 类型与文件引用。
    def validate(self) -> "PredictorDeploymentManifest":
        # bundle kind 必须与训练侧导出契约一致。
        if self.bundle_kind != _BUNDLE_KIND:
            raise ValueError(
                f"bundle_kind must be {_BUNDLE_KIND!r}, got {self.bundle_kind!r}"
            )
        # 顶层来源字段必须非空。
        _require_non_empty_string("profile_name", self.profile_name)
        _require_non_empty_string("source_checkpoint", self.source_checkpoint)
        # 每个子文件及其语义标签都必须完整。
        if self.weights_kind != _WEIGHTS_KIND:
            raise ValueError(
                f"weights_kind must be {_WEIGHTS_KIND!r}, got {self.weights_kind!r}"
            )
        if self.weights_format != _WEIGHTS_FORMAT:
            raise ValueError(
                f"weights_format must be {_WEIGHTS_FORMAT!r}, got {self.weights_format!r}"
            )
        if self.schema_kind != _SCHEMA_KIND:
            raise ValueError(
                f"schema_kind must be {_SCHEMA_KIND!r}, got {self.schema_kind!r}"
            )
        if self.metrics_kind != _METRICS_KIND:
            raise ValueError(
                f"metrics_kind must be {_METRICS_KIND!r}, got {self.metrics_kind!r}"
            )
        _require_non_empty_string("weights_file", self.weights_file)
        _require_non_empty_string("schema_file", self.schema_file)
        _require_non_empty_string("metrics_file", self.metrics_file)
        # 返回自身，便于链式使用。
        return self

    # 将 manifest 转成字典。
    def to_dict(self) -> dict[str, Any]:
        # 直接返回稳定字段映射。
        return {
            "bundle_kind": self.bundle_kind,
            "profile_name": self.profile_name,
            "source_checkpoint": self.source_checkpoint,
            "weights_kind": self.weights_kind,
            "weights_format": self.weights_format,
            "weights_file": self.weights_file,
            "schema_kind": self.schema_kind,
            "schema_file": self.schema_file,
            "metrics_kind": self.metrics_kind,
            "metrics_file": self.metrics_file,
        }

    @classmethod
    # 从字典恢复 manifest。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorDeploymentManifest":
        # 逐字段完成类型归一化。
        manifest = cls(
            bundle_kind=str(payload.get("bundle_kind", _BUNDLE_KIND)),
            profile_name=str(payload["profile_name"]),
            source_checkpoint=str(payload["source_checkpoint"]),
            weights_kind=str(payload["weights_kind"]),
            weights_format=str(payload["weights_format"]),
            weights_file=str(payload["weights_file"]),
            schema_kind=str(payload["schema_kind"]),
            schema_file=str(payload["schema_file"]),
            metrics_kind=str(payload["metrics_kind"]),
            metrics_file=str(payload["metrics_file"]),
        )
        # 返回已通过校验的 manifest。
        return manifest.validate()

    @classmethod
    # 从 JSON 文件恢复 manifest。
    def from_json_file(cls, path: str | Path) -> "PredictorDeploymentManifest":
        # 先读取 JSON 对象。
        payload = _read_json_object(Path(path), description="predictor manifest")
        # 再按字典格式恢复。
        return cls.from_dict(payload)


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
        reference_param = next(self.parameters(), None)
        target_device = (
            hidden_state.device if reference_param is None else reference_param.device
        )
        target_dtype = (
            hidden_state.dtype if reference_param is None else reference_param.dtype
        )
        hidden_state = self._normalize_hidden_state(hidden_state).to(
            device=target_device,
            dtype=target_dtype,
        )
        layer_index = self._normalize_layer_index(
            layer_index,
            batch_size=int(hidden_state.shape[0]),
            device=target_device,
        )
        layer_index = layer_index.to(dtype=target_dtype)
        fused_hidden = self.input_proj(hidden_state) + self.layer_proj(
            self._layer_features(layer_index)
        )
        logits = self.net(fused_hidden)
        return logits.view(-1, self.window_layers, self.num_experts)


HiddenStateFutureExpertPredictor = FutureExpertPredictor


@dataclass(slots=True, frozen=True)
class LoadedPredictorBundle:
    bundle_dir: Path
    manifest_path: Path
    manifest: PredictorDeploymentManifest
    schema: PredictorRuntimeSchema
    metrics: PredictorMetricsSummary
    weights_path: Path
    state_dict: dict[str, torch.Tensor]

    # 根据 runtime schema 重建 predictor 模块并加载权重。
    def build_model(self, *, device: str | torch.device = "cpu") -> FutureExpertPredictor:
        model_dtype = next(iter(self.state_dict.values())).dtype
        # 先按 schema 恢复与训练侧同构的模型。
        model = FutureExpertPredictor(
            input_dim=self.schema.input_summary_dim,
            hidden_dim=self.schema.predictor_hidden_dim,
            window_layers=self.schema.window_layers,
            num_experts=self.schema.num_experts,
        )
        model.to(dtype=model_dtype)
        # 再把导出的 state_dict 严格加载回模型。
        model.load_state_dict(self.state_dict, strict=True)
        # 最后切到目标设备并设为 eval 模式。
        model.to(device=device, dtype=model_dtype)
        model.eval()
        return model


def _load_weights_payload(
    weights_path: Path,
    *,
    expected_weights_kind: str,
    expected_profile_name: str,
    map_location: str | torch.device,
) -> dict[str, torch.Tensor]:
    # 权重文件必须存在，否则 bundle 不完整。
    if not weights_path.is_file():
        raise FileNotFoundError(f"predictor weights file does not exist: {weights_path}")
    # 在指定设备映射上读取导出的权重载荷。
    payload = torch.load(weights_path, map_location=map_location)
    # 顶层必须是字典。
    if not isinstance(payload, dict):
        raise ValueError("predictor weights payload must decode to a dictionary")

    # -----------------
    # 先校验顶层元信息。
    weights_kind = str(payload.get("weights_kind", ""))
    if weights_kind != expected_weights_kind:
        raise ValueError(
            f"weights_kind mismatch: expected {expected_weights_kind!r}, got {weights_kind!r}"
        )
    profile_name = _require_non_empty_string(
        "weights profile_name",
        payload.get("profile_name", ""),
    )
    if profile_name != expected_profile_name:
        raise ValueError(
            "predictor bundle profile_name mismatch between manifest and weights payload"
        )

    # -----------------
    # 再校验 state_dict 内容。
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("predictor weights payload is missing model_state_dict")
    normalized_state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        # 参数名必须是非空字符串。
        normalized_name = _require_non_empty_string("state_dict key", name)
        # 参数值必须是张量。
        if not torch.is_tensor(tensor):
            raise ValueError(
                f"predictor state_dict entry {normalized_name!r} must be a tensor"
            )
        normalized_state_dict[normalized_name] = tensor
    # 不能出现空的 state_dict。
    if not normalized_state_dict:
        raise ValueError("predictor model_state_dict must not be empty")
    # 返回已校验的 state_dict。
    return normalized_state_dict


def load_predictor_bundle(
    bundle_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedPredictorBundle:
    # 先把输入解析成实际 manifest 路径。
    manifest_path = _resolve_manifest_path(bundle_path)
    # 读取并校验 manifest。
    manifest = PredictorDeploymentManifest.from_json_file(manifest_path)
    # 以 manifest 所在目录作为 bundle 根目录。
    bundle_dir = manifest_path.parent.resolve()

    # -----------------
    # 读取 schema 与 metrics，并检查 kind/来源是否一致。
    schema = PredictorRuntimeSchema.from_json_file(bundle_dir / manifest.schema_file)
    metrics = PredictorMetricsSummary.from_json_file(bundle_dir / manifest.metrics_file)
    if schema.schema_kind != manifest.schema_kind:
        raise ValueError("predictor bundle schema_kind mismatch")
    if metrics.metrics_kind != manifest.metrics_kind:
        raise ValueError("predictor bundle metrics_kind mismatch")

    # -----------------
    # 校验 profile/source 是否跨文件一致。
    if schema.profile_name != manifest.profile_name:
        raise ValueError("predictor bundle profile_name mismatch between manifest and schema")
    if metrics.profile_name != manifest.profile_name:
        raise ValueError("predictor bundle profile_name mismatch between manifest and metrics")

    # -----------------
    # 最后读取权重载荷并返回完整 bundle。
    weights_path = (bundle_dir / manifest.weights_file).resolve()
    state_dict = _load_weights_payload(
        weights_path,
        expected_weights_kind=manifest.weights_kind,
        expected_profile_name=manifest.profile_name,
        map_location=map_location,
    )
    return LoadedPredictorBundle(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        schema=schema,
        metrics=metrics,
        weights_path=weights_path,
        state_dict=state_dict,
    )


def load_predictor_model(
    bundle_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    device: str | torch.device = "cpu",
) -> tuple[FutureExpertPredictor, LoadedPredictorBundle]:
    # 先读取并校验完整 bundle。
    bundle = load_predictor_bundle(bundle_path, map_location=map_location)
    # 再基于 bundle 恢复 predictor 模块。
    model = bundle.build_model(device=device)
    # 把模型和 bundle 一起返回，便于后续 candidate planning 接入。
    return model, bundle
