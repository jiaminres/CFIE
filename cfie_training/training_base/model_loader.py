"""Qwen3.5-122B 真实模型导入——从 packed checkpoint 解析 expert 权重，构建 FP32 stores。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

from cfie_training.training_base.adam_state_store import CpuAdamFp8StateStore
from cfie_training.training_base.fp32_shard_store import FP32ShardStore, ParamShardRecord
from cfie_training.training_base.gptq_cache_store import GptqCacheStore
from cfie_training.training_base.manifest_builder import (
    ManifestShardConfig,
    TrainingBaseManifest,
    TrainingBaseManifestBuilder,
    TrainingParamManifestSpec,
)
from cfie_training.training_base.progress_state import ProgressStateWriter

Qwen35_122B_CONFIG = {
    "num_layers": 48,
    "num_experts": 256,
    "hidden_size": 3072,
    "intermediate_size": 1024,
    "num_experts_per_tok": 8,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "vocab_size": 248320,
    "dtype": "bfloat16",
}


def _require_non_empty_string(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(slots=True)
class Qwen35RealImporter:
    checkpoint_dir: str | Path
    num_layers: int = 48
    num_experts: int = 256
    hidden_size: int = 3072
    intermediate_size: int = 1024
    _shard_index: Path | None = None

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)

    def iter_expert_weights(
        self,
        *,
        layers: tuple[int, ...] | None = None,
        experts: tuple[int, ...] | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        from safetensors import safe_open

        target_layers = set(layers) if layers else set(range(self.num_layers))
        target_experts = set(experts) if experts else set(range(self.num_experts))

        shard_files = sorted(
            p for p in self.checkpoint_dir.glob("model*.safetensors")
            if p.suffix == ".safetensors"
        )

        for shard_path in shard_files:
            with safe_open(str(shard_path), framework="pt") as f:
                for key in f.keys():
                    parsed = self._parse_packed_expert_key(key)
                    if parsed is None:
                        continue
                    layer_id, weight_type = parsed
                    if layer_id not in target_layers:
                        continue

                    tensor = f.get_tensor(key)
                    expert_dim = tensor.shape[0]

                    for expert_id in range(expert_dim):
                        if expert_id not in target_experts:
                            continue
                        expert_tensor = tensor[expert_id].clone().to(torch.float32)

                        if weight_type == "gate_up_proj":
                            k = self.intermediate_size * 2
                            gate = expert_tensor[:self.intermediate_size, :]
                            up = expert_tensor[self.intermediate_size:, :]
                            fused = torch.cat([gate.reshape(-1), up.reshape(-1)])
                            yield (
                                f"layers.{layer_id}.experts.{expert_id}.w13_weight",
                                fused,
                            )
                        elif weight_type == "down_proj":
                            yield (
                                f"layers.{layer_id}.experts.{expert_id}.w2_weight",
                                expert_tensor.reshape(-1),
                            )

    def build_manifest_specs(
        self,
        *,
        layers: tuple[int, ...] | None = None,
        experts: tuple[int, ...] | None = None,
    ) -> tuple[TrainingParamManifestSpec, ...]:
        target_layers = layers or tuple(range(self.num_layers))
        target_experts = experts or tuple(range(self.num_experts))

        specs: list[TrainingParamManifestSpec] = []
        for layer_id in target_layers:
            for expert_id in target_experts:
                w13_elements = 2 * self.intermediate_size * self.hidden_size
                w2_elements = self.hidden_size * self.intermediate_size
                specs.append(
                    TrainingParamManifestSpec(
                        param_id=f"layers.{layer_id}.experts.{expert_id}.w13_weight",
                        num_elements=w13_elements,
                        trainable=True,
                    )
                )
                specs.append(
                    TrainingParamManifestSpec(
                        param_id=f"layers.{layer_id}.experts.{expert_id}.w2_weight",
                        num_elements=w2_elements,
                        trainable=True,
                    )
                )
        return tuple(specs)

    def import_to_stores(
        self,
        root: str | Path,
        *,
        manifest_config: ManifestShardConfig | None = None,
        layers: tuple[int, ...] | None = None,
        experts: tuple[int, ...] | None = None,
    ) -> tuple[FP32ShardStore, CpuAdamFp8StateStore, GptqCacheStore, TrainingBaseManifest, ProgressStateWriter]:
        root_path = Path(root)
        cfg = manifest_config or ManifestShardConfig()

        specs = self.build_manifest_specs(layers=layers, experts=experts)
        manifest = TrainingBaseManifestBuilder(cfg).build(specs)
        fp32_store, adam_store, gptq_store = manifest.create_stores(
            root_path, generation=0,
        )

        updates: dict[str, torch.Tensor] = {}
        for param_id, tensor in self.iter_expert_weights(
            layers=layers, experts=experts,
        ):
            updates[param_id] = tensor

        fp32_store.flush_touched(updates, generation=0)

        progress_writer = ProgressStateWriter.in_dir(root_path / "state")
        progress_writer.write_after_flush(
            global_step=0, epoch=0, dataset_cursor="",
            round_id=0, hot_set=tuple(spec.param_id for spec in specs),
            fp32_master_generation=0, optimizer_generation=0,
            gptq_cache_generation=0,
        )

        return fp32_store, adam_store, gptq_store, manifest, progress_writer

    @staticmethod
    def _parse_packed_expert_key(name: str) -> tuple[int, str] | None:
        prefix = "model.language_model.layers."
        if not name.startswith(prefix):
            return None

        rest = name[len(prefix):]
        parts = rest.split(".", 3)
        if len(parts) < 4:
            return None

        try:
            layer_id = int(parts[0])
        except ValueError:
            return None

        if parts[1] != "mlp" or parts[2] != "experts":
            return None

        weight_type = parts[3]
        if weight_type in ("gate_up_proj", "down_proj"):
            return layer_id, weight_type

        return None


@dataclass(slots=True)
class Qwen35TrainingAdapter:
    fp32_store: FP32ShardStore
    adam_store: CpuAdamFp8StateStore
    import_config: dict = field(default_factory=lambda: dict(Qwen35_122B_CONFIG))

    def build_layer_weights(
        self,
        layer_id: int,
        *,
        expert_ids: tuple[int, ...],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> dict[str, torch.Tensor]:
        weights: dict[str, torch.Tensor] = {}
        hidden = self.import_config["hidden_size"]
        intermediate = self.import_config["intermediate_size"]

        for expert_id in expert_ids:
            w13_key = f"layers.{layer_id}.experts.{expert_id}.w13_weight"
            w2_key = f"layers.{layer_id}.experts.{expert_id}.w2_weight"

            if w13_key in self.fp32_store.records:
                w13_data = self.fp32_store.read_param(w13_key)
                w13 = torch.frombuffer(bytearray(w13_data), dtype=torch.float32)
                w13 = w13.reshape(2 * intermediate, hidden).to(device=device, dtype=dtype)
                weights[w13_key] = w13

            if w2_key in self.fp32_store.records:
                w2_data = self.fp32_store.read_param(w2_key)
                w2 = torch.frombuffer(bytearray(w2_data), dtype=torch.float32)
                w2 = w2.reshape(hidden, intermediate).to(device=device, dtype=dtype)
                weights[w2_key] = w2

        return weights

    def estimate_memory_bytes(self, param_ids: tuple[str, ...]) -> int:
        total = 0
        for pid in param_ids:
            record = self.fp32_store.records.get(pid)
            if record is not None:
                total += record.num_bytes
        return total


def build_quick_smoke_import(checkpoint_dir, output_dir, *, max_layers=2, max_experts=4, shard_gib=1.0):
    # 快速 smoke 导入：仅导入少量层/专家用于单元测试
    gb = int(shard_gib * (1 << 30))
    importer = Qwen35RealImporter(checkpoint_dir=checkpoint_dir, num_layers=max_layers, num_experts=max_experts)
    return importer.import_to_stores(output_dir, manifest_config=ManifestShardConfig(
        fp32_shard_bytes=gb, adam_shard_bytes=gb, gptq_shard_bytes=gb,
    ), layers=tuple(range(max_layers)), experts=tuple(range(max_experts)))