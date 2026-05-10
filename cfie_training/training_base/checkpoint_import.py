"""Checkpoint 导入——从 Qwen3.5 safetensors 解析 packed expert 格式并写入 FP32 stores。"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch

from cfie_training.training_base.adam_state_store import CpuAdamFp8StateStore
from cfie_training.training_base.fp32_shard_store import FP32ShardStore
from cfie_training.training_base.gptq_checkpoint import (
    GptqDecodedLayout,
    GPTQ_PACK_FACTOR_INT32,
    GptqInt4CheckpointDecoder,
    GptqInt4CheckpointTensors,
)
from cfie_training.training_base.gptq_cache_store import GptqCacheStore
from cfie_training.training_base.gptq_marlin_bundle import (
    decode_gptq_marlin_bundle,
    encode_gptq_marlin_bundle_sections,
)
from cfie_training.training_base.manifest_builder import (
    ManifestShardConfig,
    TrainingBaseManifest,
    TrainingBaseManifestBuilder,
    TrainingParamManifestSpec,
)


def _require_non_empty_string(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True, slots=True)
class ImportedCheckpointParam:
    param_id: str
    tensor: torch.Tensor
    source_keys: tuple[str, ...]
    gptq_bundle_id: str | None = None
    trainable: bool = True
    gptq_payload: bytes | None = None
    gptq_quant_layout_hash: str = ""

    def __post_init__(self) -> None:
        _require_non_empty_string("param_id", self.param_id)
        if not self.source_keys:
            raise ValueError("source_keys must not be empty")
        for key in self.source_keys:
            _require_non_empty_string("source key", key)
        if self.gptq_bundle_id is not None:
            _require_non_empty_string("gptq_bundle_id", self.gptq_bundle_id)
        if self.gptq_payload is not None and self.gptq_bundle_id is None:
            raise ValueError("gptq_payload requires gptq_bundle_id")

    @property
    def num_elements(self) -> int:
        return self.tensor.numel()

    def to_manifest_spec(self) -> TrainingParamManifestSpec:
        return TrainingParamManifestSpec(
            param_id=self.param_id,
            num_elements=self.num_elements,
            trainable=self.trainable,
            gptq_bundle_id=self.gptq_bundle_id,
            gptq_num_bytes=(
                None if self.gptq_payload is None else len(self.gptq_payload)
            ),
            quant_layout_hash=self.gptq_quant_layout_hash,
        )


@dataclass(frozen=True, slots=True)
class CheckpointImportPlan:
    imported_params: tuple[ImportedCheckpointParam, ...]
    skipped_keys: tuple[str, ...]

    @property
    def specs(self) -> tuple[TrainingParamManifestSpec, ...]:
        return tuple(param.to_manifest_spec() for param in self.imported_params)

    @property
    def fp32_updates(self) -> dict[str, torch.Tensor]:
        return {
            param.param_id: param.tensor.reshape(-1).contiguous()
            for param in self.imported_params
        }

    @property
    def param_to_source_keys(self) -> dict[str, tuple[str, ...]]:
        return {
            param.param_id: param.source_keys
            for param in self.imported_params
        }

    @property
    def gptq_cache_updates(self) -> dict[str, bytes]:
        return {
            param.gptq_bundle_id: param.gptq_payload
            for param in self.imported_params
            if param.gptq_bundle_id is not None and param.gptq_payload is not None
        }


@dataclass(frozen=True, slots=True)
class CheckpointStoreInitResult:
    import_plan: CheckpointImportPlan
    manifest: TrainingBaseManifest
    fp32_store: FP32ShardStore
    adam_store: CpuAdamFp8StateStore
    gptq_store: GptqCacheStore


@dataclass(frozen=True, slots=True)
class Qwen35MoeCheckpointImportConfig:
    checkpoint_layer_prefix: str = "layers"
    checkpoint_mlp_name: str = "mlp"
    checkpoint_experts_name: str = "experts"
    internal_layer_prefix: str = "layers"
    gate_proj_name: str = "gate_proj"
    up_proj_name: str = "up_proj"
    down_proj_name: str = "down_proj"
    weight_name: str = "weight"
    qweight_name: str = "qweight"
    scales_name: str = "scales"
    qzeros_name: str = "qzeros"
    g_idx_name: str = "g_idx"
    gptq_group_size: int = 128
    gptq_decoded_layout: GptqDecodedLayout = "n_k"
    known_root_prefixes: tuple[str, ...] = ("model.", "")
    include_gptq_cache: bool = True
    trainable: bool = True
    layer_start: int = 0
    layer_end_exclusive: int | None = None
    local_expert_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(
            "checkpoint_layer_prefix",
            self.checkpoint_layer_prefix,
        )
        _require_non_empty_string("checkpoint_mlp_name", self.checkpoint_mlp_name)
        _require_non_empty_string(
            "checkpoint_experts_name",
            self.checkpoint_experts_name,
        )
        _require_non_empty_string("internal_layer_prefix", self.internal_layer_prefix)
        _require_non_empty_string("gate_proj_name", self.gate_proj_name)
        _require_non_empty_string("up_proj_name", self.up_proj_name)
        _require_non_empty_string("down_proj_name", self.down_proj_name)
        _require_non_empty_string("weight_name", self.weight_name)
        _require_non_empty_string("qweight_name", self.qweight_name)
        _require_non_empty_string("scales_name", self.scales_name)
        _require_non_empty_string("qzeros_name", self.qzeros_name)
        _require_non_empty_string("g_idx_name", self.g_idx_name)
        if self.gptq_group_size < 1:
            raise ValueError("gptq_group_size must be >= 1")
        if self.gptq_decoded_layout not in {"k_n", "n_k"}:
            raise ValueError("gptq_decoded_layout must be 'k_n' or 'n_k'")
        _require_non_negative_int("layer_start", self.layer_start)
        if self.layer_end_exclusive is not None:
            if self.layer_end_exclusive <= self.layer_start:
                raise ValueError("layer_end_exclusive must be > layer_start")
        if self.local_expert_ids is not None:
            for expert_id in self.local_expert_ids:
                _require_non_negative_int("expert_id", expert_id)


@dataclass(slots=True)
class Qwen35MoeCheckpointImporter:
    config: Qwen35MoeCheckpointImportConfig = field(
        default_factory=Qwen35MoeCheckpointImportConfig
    )
    _pending_w13: dict[tuple[int, int], dict[str, torch.Tensor]] = field(
        default_factory=dict
    )
    _pending_w13_keys: dict[tuple[int, int], dict[str, tuple[str, ...]]] = field(
        default_factory=dict
    )
    _pending_w13_gptq: dict[tuple[int, int], dict[str, dict[str, torch.Tensor]]] = (
        field(default_factory=dict)
    )
    _pending_gptq: dict[tuple[int, int, str], dict[str, torch.Tensor]] = field(
        default_factory=dict
    )
    _pending_gptq_keys: dict[tuple[int, int, str], dict[str, str]] = field(
        default_factory=dict
    )
    _imported: list[ImportedCheckpointParam] = field(default_factory=list)
    _skipped_keys: list[str] = field(default_factory=list)

    def import_weights(
        self,
        weights: Iterable[tuple[str, Any]] | Mapping[str, Any],
    ) -> CheckpointImportPlan:
        items = weights.items() if isinstance(weights, Mapping) else weights
        for name, tensor in items:
            self.consume(name, tensor)
        return self.finalize()

    def consume(
        self,
        name: str,
        tensor: Any,
    ) -> tuple[ImportedCheckpointParam, ...]:
        parsed = self._parse_key(name)
        if parsed is None:
            return self._consume_gptq_component(name, tensor)

        layer_id, expert_id, projection = parsed
        if not self._is_selected(layer_id, expert_id):
            self._skipped_keys.append(name)
            return ()

        tensor_fp32 = _as_cpu_float32_tensor(tensor, name=name)
        before = len(self._imported)
        if projection == self.config.down_proj_name:
            self._imported.append(
                ImportedCheckpointParam(
                    param_id=self._internal_param_id(
                        layer_id,
                        expert_id,
                        "w2_weight",
                    ),
                    tensor=tensor_fp32.reshape(-1).contiguous(),
                    source_keys=(name,),
                    gptq_bundle_id=self._maybe_gptq_bundle_id(
                        layer_id,
                        expert_id,
                        "w2_weight",
                    ),
                    trainable=self.config.trainable,
                )
            )
        else:
            self._consume_w13_part(
                layer_id=layer_id,
                expert_id=expert_id,
                projection=projection,
                name=name,
                tensor=tensor_fp32,
            )
        return tuple(self._imported[before:])

    def matches_key(self, name: str) -> bool:
        parsed = self._parse_key(name)
        if parsed is not None:
            layer_id, expert_id, _ = parsed
            return self._is_selected(layer_id, expert_id)

        parsed_gptq = self._parse_gptq_key(name)
        if parsed_gptq is None:
            return False
        layer_id, expert_id, _, _ = parsed_gptq
        return self._is_selected(layer_id, expert_id)

    def finalize(self) -> CheckpointImportPlan:
        self._finalize_pending_gptq()
        if self._pending_w13:
            missing = []
            for layer_expert, parts in sorted(self._pending_w13.items()):
                expected = {self.config.gate_proj_name, self.config.up_proj_name}
                missing_parts = sorted(expected - set(parts))
                if missing_parts:
                    missing.append(f"{layer_expert}: {','.join(missing_parts)}")
            if missing:
                raise ValueError(
                    "incomplete w13 checkpoint pairs: " + "; ".join(missing)
                )

        imported = tuple(sorted(self._imported, key=lambda item: item.param_id))
        return CheckpointImportPlan(
            imported_params=imported,
            skipped_keys=tuple(self._skipped_keys),
        )

    def _consume_w13_part(
        self,
        *,
        layer_id: int,
        expert_id: int,
        projection: str,
        name: str,
        tensor: torch.Tensor,
    ) -> None:
        key = (layer_id, expert_id)
        parts = self._pending_w13.setdefault(key, {})
        source_keys = self._pending_w13_keys.setdefault(key, {})
        if projection in parts:
            raise ValueError(f"duplicate checkpoint tensor for {name!r}")
        parts[projection] = tensor
        source_keys[projection] = (name,)
        if (
            self.config.gate_proj_name not in parts
            or self.config.up_proj_name not in parts
        ):
            return

        gate = parts[self.config.gate_proj_name]
        up = parts[self.config.up_proj_name]
        if gate.shape != up.shape:
            raise ValueError(
                "gate_proj and up_proj must have identical shapes for w13 fusion"
            )

        fused = torch.cat(
            [gate.reshape(-1), up.reshape(-1)],
            dim=0,
        ).contiguous()
        self._imported.append(
            ImportedCheckpointParam(
                param_id=self._internal_param_id(
                    layer_id,
                    expert_id,
                    "w13_weight",
                ),
                tensor=fused,
                source_keys=(
                    *source_keys[self.config.gate_proj_name],
                    *source_keys[self.config.up_proj_name],
                ),
                gptq_bundle_id=self._maybe_gptq_bundle_id(
                    layer_id,
                    expert_id,
                    "w13_weight",
                ),
                trainable=self.config.trainable,
            )
        )
        del self._pending_w13[key]
        del self._pending_w13_keys[key]

    def _consume_gptq_component(
        self,
        name: str,
        tensor: Any,
    ) -> tuple[ImportedCheckpointParam, ...]:
        parsed = self._parse_gptq_key(name)
        if parsed is None:
            self._skipped_keys.append(name)
            return ()

        layer_id, expert_id, projection, component = parsed
        if not self._is_selected(layer_id, expert_id):
            self._skipped_keys.append(name)
            return ()

        key = (layer_id, expert_id, projection)
        components = self._pending_gptq.setdefault(key, {})
        source_keys = self._pending_gptq_keys.setdefault(key, {})
        if component in components:
            raise ValueError(f"duplicate GPTQ checkpoint tensor for {name!r}")
        components[component] = tensor.detach().cpu().contiguous()
        source_keys[component] = name
        return ()

    def _finalize_pending_gptq(self) -> None:
        if not self._pending_gptq:
            return
        decoder = GptqInt4CheckpointDecoder(
            group_size=self.config.gptq_group_size,
            decoded_layout=self.config.gptq_decoded_layout,
        )
        for key, components in sorted(self._pending_gptq.items()):
            layer_id, expert_id, projection = key
            source_keys = self._pending_gptq_keys[key]
            if self.config.qweight_name not in components:
                raise ValueError(f"missing GPTQ qweight for {key}")
            if self.config.scales_name not in components:
                raise ValueError(f"missing GPTQ scales for {key}")

            decoded = decoder.decode(
                GptqInt4CheckpointTensors(
                    qweight=components[self.config.qweight_name],
                    scales=components[self.config.scales_name],
                    qzeros=components.get(self.config.qzeros_name),
                    g_idx=components.get(self.config.g_idx_name),
                )
            )
            component_source_keys = tuple(
                source_keys[component]
                for component in (
                    self.config.qweight_name,
                    self.config.scales_name,
                    self.config.qzeros_name,
                    self.config.g_idx_name,
                )
                if component in source_keys
            )
            if projection == self.config.down_proj_name:
                gptq_payload, gptq_layout_hash = self._build_projection_gptq_bundle(
                    layer_id=layer_id,
                    expert_id=expert_id,
                    components=components,
                    weight_name="w2_weight",
                )
                self._imported.append(
                    ImportedCheckpointParam(
                        param_id=self._internal_param_id(
                            layer_id,
                            expert_id,
                            "w2_weight",
                        ),
                        tensor=decoded.reshape(-1).contiguous(),
                        source_keys=component_source_keys,
                        gptq_bundle_id=self._maybe_gptq_bundle_id(
                            layer_id,
                            expert_id,
                            "w2_weight",
                        ),
                        trainable=self.config.trainable,
                        gptq_payload=gptq_payload,
                        gptq_quant_layout_hash=gptq_layout_hash,
                    )
                )
            else:
                self._consume_decoded_w13_part(
                    layer_id=layer_id,
                    expert_id=expert_id,
                    projection=projection,
                    tensor=decoded,
                    source_keys=component_source_keys,
                    gptq_components=components,
                )
        self._pending_gptq.clear()
        self._pending_gptq_keys.clear()

    def _consume_decoded_w13_part(
        self,
        *,
        layer_id: int,
        expert_id: int,
        projection: str,
        tensor: torch.Tensor,
        source_keys: tuple[str, ...],
        gptq_components: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        key = (layer_id, expert_id)
        parts = self._pending_w13.setdefault(key, {})
        pending_source_keys = self._pending_w13_keys.setdefault(key, {})
        pending_gptq = self._pending_w13_gptq.setdefault(key, {})
        if projection in parts:
            raise ValueError(f"duplicate decoded tensor for {key} {projection}")
        parts[projection] = tensor
        pending_source_keys[projection] = source_keys
        if gptq_components is not None:
            pending_gptq[projection] = dict(gptq_components)
        if (
            self.config.gate_proj_name not in parts
            or self.config.up_proj_name not in parts
        ):
            return

        gate = parts[self.config.gate_proj_name]
        up = parts[self.config.up_proj_name]
        if gate.shape != up.shape:
            raise ValueError(
                "gate_proj and up_proj must have identical shapes for w13 fusion"
            )
        gptq_payload = None
        gptq_layout_hash = ""
        if (
            self.config.gate_proj_name in pending_gptq
            and self.config.up_proj_name in pending_gptq
        ):
            gptq_payload, gptq_layout_hash = self._build_w13_gptq_bundle(
                layer_id=layer_id,
                expert_id=expert_id,
                gate_components=pending_gptq[self.config.gate_proj_name],
                up_components=pending_gptq[self.config.up_proj_name],
            )
        self._imported.append(
            ImportedCheckpointParam(
                param_id=self._internal_param_id(
                    layer_id,
                    expert_id,
                    "w13_weight",
                ),
                tensor=torch.cat(
                    [gate.reshape(-1), up.reshape(-1)],
                    dim=0,
                ).contiguous(),
                source_keys=(
                    *pending_source_keys[self.config.gate_proj_name],
                    *pending_source_keys[self.config.up_proj_name],
                ),
                gptq_bundle_id=self._maybe_gptq_bundle_id(
                    layer_id,
                    expert_id,
                    "w13_weight",
                ),
                trainable=self.config.trainable,
                gptq_payload=gptq_payload,
                gptq_quant_layout_hash=gptq_layout_hash,
            )
        )
        del self._pending_w13[key]
        del self._pending_w13_keys[key]
        self._pending_w13_gptq.pop(key, None)

    def _build_projection_gptq_bundle(
        self,
        *,
        layer_id: int,
        expert_id: int,
        components: Mapping[str, torch.Tensor],
        weight_name: str,
    ) -> tuple[bytes, str]:
        size_k, size_n = _infer_gptq_raw_shape(
            qweight=components[self.config.qweight_name],
            g_idx=components.get(self.config.g_idx_name),
        )
        payload = encode_gptq_marlin_bundle_sections(
            bundle_id=self._internal_param_id(layer_id, expert_id, weight_name),
            group_size=self.config.gptq_group_size,
            size_k=size_k,
            size_n=size_n,
            sections=self._bundle_sections_from_components(components),
            act_order=self.config.g_idx_name in components,
        )
        layout_hash = decode_gptq_marlin_bundle(payload).metadata.layout_hash
        return payload, layout_hash

    def _build_w13_gptq_bundle(
        self,
        *,
        layer_id: int,
        expert_id: int,
        gate_components: Mapping[str, torch.Tensor],
        up_components: Mapping[str, torch.Tensor],
    ) -> tuple[bytes, str]:
        gate_size_k, gate_size_n = _infer_gptq_raw_shape(
            qweight=gate_components[self.config.qweight_name],
            g_idx=gate_components.get(self.config.g_idx_name),
        )
        up_size_k, up_size_n = _infer_gptq_raw_shape(
            qweight=up_components[self.config.qweight_name],
            g_idx=up_components.get(self.config.g_idx_name),
        )
        if gate_size_k != up_size_k:
            raise ValueError("gate_proj and up_proj GPTQ K dimensions must match")
        sections = {
            **self._bundle_sections_from_components(gate_components, prefix="w1"),
            **self._bundle_sections_from_components(up_components, prefix="w3"),
        }
        payload = encode_gptq_marlin_bundle_sections(
            bundle_id=self._internal_param_id(layer_id, expert_id, "w13_weight"),
            group_size=self.config.gptq_group_size,
            size_k=gate_size_k,
            size_n=gate_size_n + up_size_n,
            sections=sections,
            act_order=(
                self.config.g_idx_name in gate_components
                or self.config.g_idx_name in up_components
            ),
        )
        layout_hash = decode_gptq_marlin_bundle(payload).metadata.layout_hash
        return payload, layout_hash

    def _bundle_sections_from_components(
        self,
        components: Mapping[str, torch.Tensor],
        *,
        prefix: str | None = None,
    ) -> dict[str, torch.Tensor | None]:
        names = {
            self.config.qweight_name: "qweight",
            self.config.scales_name: "scales",
            self.config.qzeros_name: "qzeros",
            self.config.g_idx_name: "g_idx",
        }
        sections: dict[str, torch.Tensor | None] = {}
        for component_name, section_name in names.items():
            if component_name not in components:
                continue
            full_name = section_name if prefix is None else f"{prefix}.{section_name}"
            sections[full_name] = components[component_name]
        return sections

    def _parse_key(self, name: str) -> tuple[int, int, str] | None:
        stripped = self._strip_known_root_prefix(name)
        pattern = (
            rf"^{re.escape(self.config.checkpoint_layer_prefix)}\."
            rf"(?P<layer>\d+)\."
            rf"{re.escape(self.config.checkpoint_mlp_name)}\."
            rf"{re.escape(self.config.checkpoint_experts_name)}\."
            rf"(?P<expert>\d+)\."
            rf"(?P<projection>{self._projection_pattern()})\."
            rf"{re.escape(self.config.weight_name)}$"
        )
        match = re.match(pattern, stripped)
        if match is None:
            return None
        return (
            int(match.group("layer")),
            int(match.group("expert")),
            match.group("projection"),
        )

    def _parse_gptq_key(self, name: str) -> tuple[int, int, str, str] | None:
        stripped = self._strip_known_root_prefix(name)
        pattern = (
            rf"^{re.escape(self.config.checkpoint_layer_prefix)}\."
            rf"(?P<layer>\d+)\."
            rf"{re.escape(self.config.checkpoint_mlp_name)}\."
            rf"{re.escape(self.config.checkpoint_experts_name)}\."
            rf"(?P<expert>\d+)\."
            rf"(?P<projection>{self._projection_pattern()})\."
            rf"(?P<component>{self._gptq_component_pattern()})$"
        )
        match = re.match(pattern, stripped)
        if match is None:
            return None
        return (
            int(match.group("layer")),
            int(match.group("expert")),
            match.group("projection"),
            match.group("component"),
        )

    def _projection_pattern(self) -> str:
        return "|".join(
            re.escape(name)
            for name in (
                self.config.gate_proj_name,
                self.config.up_proj_name,
                self.config.down_proj_name,
            )
        )

    def _gptq_component_pattern(self) -> str:
        return "|".join(
            re.escape(name)
            for name in (
                self.config.qweight_name,
                self.config.scales_name,
                self.config.qzeros_name,
                self.config.g_idx_name,
            )
        )

    def _strip_known_root_prefix(self, name: str) -> str:
        for prefix in self.config.known_root_prefixes:
            if prefix and name.startswith(prefix):
                return name[len(prefix):]
        return name

    def _is_selected(self, layer_id: int, expert_id: int) -> bool:
        if layer_id < self.config.layer_start:
            return False
        if (
            self.config.layer_end_exclusive is not None
            and layer_id >= self.config.layer_end_exclusive
        ):
            return False
        return (
            self.config.local_expert_ids is None
            or expert_id in self.config.local_expert_ids
        )

    def _internal_param_id(
        self,
        layer_id: int,
        expert_id: int,
        weight_name: str,
    ) -> str:
        return (
            f"{self.config.internal_layer_prefix}.{layer_id}."
            f"experts.{expert_id}.{weight_name}"
        )

    def _maybe_gptq_bundle_id(
        self,
        layer_id: int,
        expert_id: int,
        weight_name: str,
    ) -> str | None:
        if not self.config.include_gptq_cache:
            return None
        return self._internal_param_id(layer_id, expert_id, weight_name)


def import_qwen35_moe_checkpoint(
    weights: Iterable[tuple[str, Any]] | Mapping[str, Any],
    *,
    config: Qwen35MoeCheckpointImportConfig | None = None,
) -> CheckpointImportPlan:
    importer = Qwen35MoeCheckpointImporter(
        Qwen35MoeCheckpointImportConfig() if config is None else config
    )
    return importer.import_weights(weights)


def qwen35_moe_checkpoint_key_filter(
    config: Qwen35MoeCheckpointImportConfig | None = None,
) -> Callable[[str], bool]:
    importer = Qwen35MoeCheckpointImporter(
        Qwen35MoeCheckpointImportConfig() if config is None else config
    )
    return importer.matches_key


def initialize_fp32_store_from_import_plan(
    *,
    plan: CheckpointImportPlan,
    root: str | Path,
    manifest_config: ManifestShardConfig | None = None,
    generation: int = 0,
) -> CheckpointStoreInitResult:
    manifest = TrainingBaseManifestBuilder(
        ManifestShardConfig() if manifest_config is None else manifest_config
    ).build(plan.specs)
    fp32_store, adam_store, gptq_store = manifest.create_stores(
        root,
        generation=generation,
    )
    fp32_store.flush_touched(plan.fp32_updates, generation=generation)
    if plan.gptq_cache_updates:
        gptq_store.flush_touched(plan.gptq_cache_updates, generation=generation)
    return CheckpointStoreInitResult(
        import_plan=plan,
        manifest=manifest,
        fp32_store=fp32_store,
        adam_store=adam_store,
        gptq_store=gptq_store,
    )


def import_qwen35_moe_checkpoint_to_fp32_store(
    weights: Iterable[tuple[str, Any]] | Mapping[str, Any],
    *,
    root: str | Path,
    import_config: Qwen35MoeCheckpointImportConfig | None = None,
    manifest_config: ManifestShardConfig | None = None,
    generation: int = 0,
) -> CheckpointStoreInitResult:
    plan = import_qwen35_moe_checkpoint(weights, config=import_config)
    return initialize_fp32_store_from_import_plan(
        plan=plan,
        root=root,
        manifest_config=manifest_config,
        generation=generation,
    )


def _as_cpu_float32_tensor(value: Any, *, name: str) -> torch.Tensor:
    if not hasattr(value, "detach"):
        raise TypeError(f"{name} must be a torch.Tensor")
    tensor = value.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.to(dtype=torch.float32).contiguous()


def _infer_gptq_raw_shape(
    *,
    qweight: torch.Tensor,
    g_idx: torch.Tensor | None,
) -> tuple[int, int]:
    if qweight.ndim != 2:
        raise ValueError("GPTQ qweight must have shape [packed_k, n]")
    size_k = (
        int(g_idx.reshape(-1).numel())
        if g_idx is not None
        else int(qweight.shape[0]) * GPTQ_PACK_FACTOR_INT32
    )
    return size_k, int(qweight.shape[1])
