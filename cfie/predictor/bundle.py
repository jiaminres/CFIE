"""Predictor runtime loading helpers for CFIE inference."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
import torch

from cfie_training.predictor.architectures import (
    FutureExpertPredictor,
    build_predictor_model,
)
from cfie_training.predictor.models import (
    PredictorCheckpointMetadata,
    PredictorTrainingRunTrace,
)

_BUNDLE_KIND = "cfie_predictor_deployment_bundle"
_SCHEMA_KIND = "cfie_predictor_runtime_schema"
_METRICS_KIND = "cfie_predictor_metrics_summary"
_WEIGHTS_KIND = "cfie_predictor_weights"
_WEIGHTS_FORMAT = "torch_state_dict"
_DEFAULT_MANIFEST_FILE = "predictor_bundle.json"


def _require_non_empty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_int(name: str, value: Any, *, allow_zero: bool = False) -> int:
    parsed = int(value)
    lower_bound = 0 if allow_zero else 1
    if parsed < lower_bound:
        comparator = ">=" if allow_zero else ">"
        raise ValueError(f"{name} must be {comparator} 0")
    return parsed


def _require_float(name: str, value: Any) -> float:
    return float(value)


def _require_ratio(name: str, value: Any) -> float:
    parsed = _require_float(name, value)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return parsed


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} JSON must decode to an object")
    return payload


def _resolve_manifest_path(bundle_path: str | Path) -> Path:
    resolved = Path(bundle_path)
    if resolved.is_dir():
        resolved = resolved / _DEFAULT_MANIFEST_FILE
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
    model_descriptor: dict[str, Any] = field(default_factory=dict)
    min_insertion_layer_index: int = 0

    def validate(self) -> "PredictorRuntimeSchema":
        if self.schema_kind != _SCHEMA_KIND:
            raise ValueError(
                f"schema_kind must be {_SCHEMA_KIND!r}, got {self.schema_kind!r}"
            )
        _require_non_empty_string("profile_name", self.profile_name)
        _require_non_empty_string("selection_mode", self.selection_mode)
        _require_non_empty_string("online_expert_source", self.online_expert_source)
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
        _require_int(
            "min_insertion_layer_index",
            self.min_insertion_layer_index,
            allow_zero=True,
        )
        if not isinstance(self.model_descriptor, dict):
            raise ValueError("model_descriptor must be a dictionary")
        if self.candidate_experts_per_layer < self.executed_experts_per_layer:
            raise ValueError(
                "candidate_experts_per_layer must be >= executed_experts_per_layer"
            )
        if self.candidate_experts_per_layer > self.num_experts:
            raise ValueError(
                "candidate_experts_per_layer must be <= num_experts"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
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
            "model_descriptor": dict(self.model_descriptor),
            "min_insertion_layer_index": self.min_insertion_layer_index,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorRuntimeSchema":
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
            model_descriptor=dict(payload.get("model_descriptor", {})),
            min_insertion_layer_index=int(
                payload.get("min_insertion_layer_index", 0)
            ),
        )
        return schema.validate()

    @classmethod
    def from_json_file(cls, path: str | Path) -> "PredictorRuntimeSchema":
        payload = _read_json_object(Path(path), description="predictor schema")
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

    def validate(self) -> "PredictorMetricsSummary":
        if self.metrics_kind != _METRICS_KIND:
            raise ValueError(
                f"metrics_kind must be {_METRICS_KIND!r}, got {self.metrics_kind!r}"
            )
        _require_non_empty_string("profile_name", self.profile_name)
        _require_int("example_count", self.example_count, allow_zero=True)
        _require_int("epochs", self.epochs, allow_zero=True)
        _require_float("final_mean_loss", self.final_mean_loss)
        _require_ratio(
            "final_recall_at_candidate_budget",
            self.final_recall_at_candidate_budget,
        )
        _require_ratio(
            "final_recall_at_executed_budget",
            self.final_recall_at_executed_budget,
        )
        return self

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorMetricsSummary":
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
        return summary.validate()

    @classmethod
    def from_json_file(cls, path: str | Path) -> "PredictorMetricsSummary":
        payload = _read_json_object(Path(path), description="predictor metrics")
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

    def validate(self) -> "PredictorDeploymentManifest":
        if self.bundle_kind != _BUNDLE_KIND:
            raise ValueError(
                f"bundle_kind must be {_BUNDLE_KIND!r}, got {self.bundle_kind!r}"
            )
        _require_non_empty_string("profile_name", self.profile_name)
        _require_non_empty_string("source_checkpoint", self.source_checkpoint)
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
        return self

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorDeploymentManifest":
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
        return manifest.validate()

    @classmethod
    def from_json_file(cls, path: str | Path) -> "PredictorDeploymentManifest":
        payload = _read_json_object(Path(path), description="predictor manifest")
        return cls.from_dict(payload)


HiddenStateFutureExpertPredictor = FutureExpertPredictor


@dataclass(slots=True, frozen=True)
class LoadedPredictorBundle:
    bundle_dir: Path
    manifest_path: Path | None
    manifest: PredictorDeploymentManifest | None
    schema: PredictorRuntimeSchema
    metrics: PredictorMetricsSummary
    weights_path: Path
    state_dict: dict[str, torch.Tensor]
    checkpoint_metadata: PredictorCheckpointMetadata | None = None

    def build_model(
        self,
        *,
        device: str | torch.device = "cpu",
        base_model_path: str | Path | None = None,
        num_layers: int | None = None,
    ) -> FutureExpertPredictor:
        model_dtype = next(iter(self.state_dict.values())).dtype
        model_descriptor = dict(self.schema.model_descriptor)
        if not model_descriptor and self.checkpoint_metadata is not None:
            model_descriptor = dict(self.checkpoint_metadata.model_descriptor)

        model_architecture = str(model_descriptor.get("architecture", "mlp"))
        model_hidden_dim = int(
            model_descriptor.get("hidden_dim", self.schema.predictor_hidden_dim)
        )
        model_depth = int(model_descriptor.get("depth", 2))
        model_dropout = float(model_descriptor.get("dropout", 0.0))
        model_num_heads = int(model_descriptor.get("num_heads", 1))
        model_memory_tokens = int(model_descriptor.get("memory_tokens", 0))
        model_ffn_multiplier = int(model_descriptor.get("ffn_multiplier", 4))
        resolved_num_layers = int(
            num_layers
            if num_layers is not None
            else model_descriptor.get("num_layers", 0) or 0
        )

        frozen_router_weights = None
        if model_architecture == "frozen_router_delta":
            if base_model_path is None:
                raise ValueError(
                    "base_model_path is required for frozen_router_delta runtime loading"
                )
            if resolved_num_layers < 1:
                raise ValueError(
                    "num_layers is required for frozen_router_delta runtime loading"
                )
            frozen_router_weights = _load_router_gate_weights(
                base_model_path,
                num_layers=resolved_num_layers,
                dtype=torch.float32,
            )

        model = build_predictor_model(
            input_dim=self.schema.input_summary_dim,
            hidden_dim=model_hidden_dim,
            window_layers=self.schema.window_layers,
            num_experts=self.schema.num_experts,
            model_architecture=model_architecture,
            model_depth=model_depth,
            model_dropout=model_dropout,
            model_num_heads=model_num_heads,
            model_memory_tokens=model_memory_tokens,
            model_ffn_multiplier=model_ffn_multiplier,
            num_layers=resolved_num_layers if resolved_num_layers > 0 else None,
            frozen_router_weights=frozen_router_weights,
        )
        model.to(dtype=model_dtype)
        model.load_state_dict(self.state_dict, strict=True)
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
    if not weights_path.is_file():
        raise FileNotFoundError(f"predictor weights file does not exist: {weights_path}")
    payload = torch.load(weights_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("predictor weights payload must decode to a dictionary")

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

    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("predictor weights payload is missing model_state_dict")
    normalized_state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        normalized_name = _require_non_empty_string("state_dict key", name)
        if not torch.is_tensor(tensor):
            raise ValueError(
                f"predictor state_dict entry {normalized_name!r} must be a tensor"
            )
        normalized_state_dict[normalized_name] = tensor
    if not normalized_state_dict:
        raise ValueError("predictor model_state_dict must not be empty")
    return normalized_state_dict


def _build_metrics_summary_from_checkpoint(
    metadata: PredictorCheckpointMetadata,
    run_trace: PredictorTrainingRunTrace | None,
) -> PredictorMetricsSummary:
    if run_trace is not None:
        return PredictorMetricsSummary(
            metrics_kind=_METRICS_KIND,
            profile_name=run_trace.profile_name,
            example_count=run_trace.example_count,
            epochs=run_trace.epochs,
            final_mean_loss=run_trace.final_mean_loss,
            final_recall_at_candidate_budget=(
                run_trace.final_recall_at_candidate_budget
            ),
            final_recall_at_executed_budget=(
                run_trace.final_recall_at_executed_budget
            ),
        ).validate()
    return PredictorMetricsSummary(
        metrics_kind=_METRICS_KIND,
        profile_name=metadata.profile_name,
        example_count=metadata.example_count,
        epochs=metadata.epochs,
        final_mean_loss=metadata.final_mean_loss,
        final_recall_at_candidate_budget=(
            metadata.final_recall_at_candidate_budget
        ),
        final_recall_at_executed_budget=(
            metadata.final_recall_at_executed_budget
        ),
    ).validate()


def _build_runtime_schema_from_checkpoint(
    metadata: PredictorCheckpointMetadata,
) -> PredictorRuntimeSchema:
    return PredictorRuntimeSchema(
        schema_kind=_SCHEMA_KIND,
        profile_name=metadata.profile_name,
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
        model_descriptor=dict(metadata.model_descriptor),
        min_insertion_layer_index=metadata.min_insertion_layer_index,
    ).validate()


def _read_checkpoint_payload(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("predictor checkpoint must decode to a dictionary")
    return payload


def _load_predictor_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedPredictorBundle:
    resolved_path = Path(checkpoint_path).resolve()
    payload = _read_checkpoint_payload(resolved_path, map_location=map_location)

    metadata_payload = payload.get("metadata")
    if not isinstance(metadata_payload, dict):
        raise ValueError("predictor checkpoint is missing metadata")
    metadata = PredictorCheckpointMetadata.from_dict(metadata_payload)

    run_trace_payload = payload.get("run_trace")
    run_trace = None
    if run_trace_payload is not None:
        if not isinstance(run_trace_payload, dict):
            raise ValueError("predictor checkpoint run_trace must be a dictionary")
        run_trace = PredictorTrainingRunTrace.from_dict(run_trace_payload)

    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("predictor checkpoint is missing model_state_dict")
    normalized_state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        normalized_name = _require_non_empty_string("state_dict key", name)
        if not torch.is_tensor(tensor):
            raise ValueError(
                f"predictor state_dict entry {normalized_name!r} must be a tensor"
            )
        normalized_state_dict[normalized_name] = tensor
    if not normalized_state_dict:
        raise ValueError("predictor model_state_dict must not be empty")

    return LoadedPredictorBundle(
        bundle_dir=resolved_path.parent,
        manifest_path=None,
        manifest=None,
        schema=_build_runtime_schema_from_checkpoint(metadata),
        metrics=_build_metrics_summary_from_checkpoint(metadata, run_trace),
        weights_path=resolved_path,
        state_dict=normalized_state_dict,
        checkpoint_metadata=metadata,
    )


def _read_safetensors_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return {}
    payload = _read_json_object(index_path, description="model safetensors index")
    weight_map = payload.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise ValueError("model safetensors index weight_map must be an object")
    return {str(name): str(file_name) for name, file_name in weight_map.items()}


def _candidate_router_gate_tensor_names(layer_index: int) -> tuple[str, ...]:
    return (
        f"model.language_model.layers.{layer_index}.mlp.gate.weight",
        f"model.layers.{layer_index}.mlp.gate.weight",
    )


def _load_router_gate_weights(
    base_model_path: str | Path,
    *,
    num_layers: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    model_dir = Path(base_model_path).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"base model path does not exist: {model_dir}")

    weight_map = _read_safetensors_weight_map(model_dir)
    router_weights: list[torch.Tensor | None] = [None] * int(num_layers)

    if weight_map:
        requests_by_file: dict[str, list[tuple[int, str]]] = {}
        for layer_index in range(int(num_layers)):
            tensor_name = next(
                (
                    candidate_name
                    for candidate_name in _candidate_router_gate_tensor_names(layer_index)
                    if candidate_name in weight_map
                ),
                None,
            )
            if tensor_name is None:
                raise ValueError(
                    f"missing router gate weight for layer {layer_index} under {model_dir}"
                )
            requests_by_file.setdefault(weight_map[tensor_name], []).append(
                (layer_index, tensor_name)
            )

        for file_name, requests in requests_by_file.items():
            file_path = model_dir / file_name
            if not file_path.is_file():
                raise FileNotFoundError(
                    f"router gate shard file does not exist: {file_path}"
                )
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                for layer_index, tensor_name in requests:
                    router_weights[layer_index] = handle.get_tensor(tensor_name).to(
                        dtype=dtype
                    ).contiguous()
    else:
        safetensor_files = sorted(model_dir.glob("*.safetensors"))
        if len(safetensor_files) != 1:
            raise FileNotFoundError(
                "router gate tensor lookup requires model.safetensors.index.json "
                "or a single .safetensors file"
            )
        with safe_open(safetensor_files[0], framework="pt", device="cpu") as handle:
            handle_keys = set(handle.keys())
            for layer_index in range(int(num_layers)):
                tensor_name = next(
                    (
                        candidate_name
                        for candidate_name in _candidate_router_gate_tensor_names(
                            layer_index
                        )
                        if candidate_name in handle_keys
                    ),
                    None,
                )
                if tensor_name is None:
                    raise ValueError(
                        f"missing router gate weight for layer {layer_index} under "
                        f"{safetensor_files[0]}"
                    )
                router_weights[layer_index] = handle.get_tensor(tensor_name).to(
                    dtype=dtype
                ).contiguous()

    if any(router_weight is None for router_weight in router_weights):
        missing_layers = [
            layer_index
            for layer_index, router_weight in enumerate(router_weights)
            if router_weight is None
        ]
        raise ValueError(
            f"failed to load router gate weights for layers: {missing_layers}"
        )

    return torch.stack(
        [router_weight for router_weight in router_weights if router_weight is not None],
        dim=0,
    )


def load_predictor_bundle(
    bundle_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedPredictorBundle:
    manifest_path = _resolve_manifest_path(bundle_path)
    manifest = PredictorDeploymentManifest.from_json_file(manifest_path)
    bundle_dir = manifest_path.parent.resolve()

    schema = PredictorRuntimeSchema.from_json_file(bundle_dir / manifest.schema_file)
    metrics = PredictorMetricsSummary.from_json_file(bundle_dir / manifest.metrics_file)
    if schema.schema_kind != manifest.schema_kind:
        raise ValueError("predictor bundle schema_kind mismatch")
    if metrics.metrics_kind != manifest.metrics_kind:
        raise ValueError("predictor bundle metrics_kind mismatch")
    if schema.profile_name != manifest.profile_name:
        raise ValueError("predictor bundle profile_name mismatch between manifest and schema")
    if metrics.profile_name != manifest.profile_name:
        raise ValueError("predictor bundle profile_name mismatch between manifest and metrics")

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
        checkpoint_metadata=None,
    )


def load_predictor_model(
    bundle_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    device: str | torch.device = "cpu",
    base_model_path: str | Path | None = None,
    num_layers: int | None = None,
) -> tuple[FutureExpertPredictor, LoadedPredictorBundle]:
    resolved_path = Path(bundle_path)
    if resolved_path.is_dir() or resolved_path.suffix.lower() == ".json":
        bundle = load_predictor_bundle(bundle_path, map_location=map_location)
    else:
        bundle = _load_predictor_checkpoint(bundle_path, map_location=map_location)
    model = bundle.build_model(
        device=device,
        base_model_path=base_model_path,
        num_layers=num_layers,
    )
    return model, bundle
