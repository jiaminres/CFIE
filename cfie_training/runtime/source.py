"""Local safetensors manifest helpers used by predictor training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from safetensors import safe_open
import torch

from cfie_training.config import TrainingProjectConfig


@dataclass(slots=True, frozen=True)
class WeightTensorRef:
    tensor_name: str
    file_name: str


class LocalWeightManifest:
    def __init__(self, config: TrainingProjectConfig) -> None:
        self.config = config
        self._model_path = Path(config.model_source.model_path).expanduser().resolve()
        self._index_path = self._model_path / config.model_source.index_filename
        self._weight_map: dict[str, str] = {}
        if (
            config.model_source.use_local_weight_manifest
            and self._model_path.is_dir()
            and self._index_path.is_file()
        ):
            payload = json.loads(self._index_path.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map", {})
            if isinstance(weight_map, dict):
                self._weight_map = {
                    str(key): str(value) for key, value in weight_map.items()
                }

    @property
    def available(self) -> bool:
        return bool(self._weight_map) or self._single_safetensors_path() is not None

    @property
    def model_path(self) -> str:
        return str(self._model_path)

    def resolve_file_path(self, file_name: str) -> Path:
        return self._model_path / file_name

    def tensor_ref(self, tensor_name: str) -> WeightTensorRef | None:
        file_name = self._weight_map.get(tensor_name)
        if file_name is not None:
            return WeightTensorRef(tensor_name=tensor_name, file_name=file_name)
        single_file = self._single_safetensors_path()
        if single_file is None:
            return None
        with safe_open(single_file, framework="pt", device="cpu") as handle:
            if tensor_name not in handle.keys():
                return None
        return WeightTensorRef(tensor_name=tensor_name, file_name=single_file.name)

    def load_tensor(
        self,
        tensor_name: str,
        *,
        dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor | None:
        tensor_ref = self.tensor_ref(tensor_name)
        if tensor_ref is None:
            return None
        file_path = self.resolve_file_path(tensor_ref.file_name)
        if not file_path.is_file():
            return None
        try:
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(tensor_name)
        except (KeyError, RuntimeError, ValueError):
            return None
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor.contiguous()

    def router_gate_ref(self, layer_index: int) -> WeightTensorRef | None:
        for tensor_name in self._candidate_router_gate_tensor_names(layer_index):
            tensor_ref = self.tensor_ref(tensor_name)
            if tensor_ref is not None:
                return tensor_ref
        return None

    def load_router_gate_tensor(
        self,
        layer_index: int,
        *,
        dtype: torch.dtype | None = torch.float32,
    ) -> torch.Tensor | None:
        tensor_ref = self.router_gate_ref(layer_index)
        if tensor_ref is None:
            return None
        return self.load_tensor(tensor_ref.tensor_name, dtype=dtype)

    def _candidate_router_gate_tensor_names(
        self,
        layer_index: int,
    ) -> tuple[str, ...]:
        if layer_index < 0:
            return ()
        if layer_index < self.config.model_spec.num_hidden_layers:
            return (
                f"model.language_model.layers.{layer_index}.mlp.gate.weight",
                f"model.layers.{layer_index}.mlp.gate.weight",
                f"language_model.model.layers.{layer_index}.mlp.gate.weight",
            )
        mtp_index = layer_index - self.config.model_spec.num_hidden_layers
        if mtp_index >= self.config.model_spec.mtp_num_hidden_layers:
            return ()
        return (
            f"mtp.layers.{mtp_index}.mlp.gate.weight",
            f"model.mtp.layers.{mtp_index}.mlp.gate.weight",
        )

    def _single_safetensors_path(self) -> Path | None:
        if not self._model_path.is_dir():
            return None
        safetensors_files = sorted(self._model_path.glob("*.safetensors"))
        if len(safetensors_files) != 1:
            return None
        return safetensors_files[0]
