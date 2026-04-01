"""Local safetensors-backed expert source for tiered MoE caching."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from cfie.logger import init_logger

logger = init_logger(__name__)

_EXPERT_KEY_RE = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<expert>\d+)\.(?P<suffix>.+)$"
)


@dataclass(frozen=True, slots=True)
class TensorRef:
    file_name: str
    full_key: str
    relative_key: str


class SafetensorExpertStore:
    """Fetch a single expert from local safetensors shards on demand."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self._expert_refs = self._build_index()

    def has_expert(self, layer_name: str, expert_id: int) -> bool:
        return self._resolve_refs(layer_name, expert_id) is not None

    def load_expert(
        self,
        layer_name: str,
        expert_id: int,
        *,
        skip_suffixes: tuple[str, ...] = (),
    ) -> dict[str, torch.Tensor]:
        refs = self._resolve_refs(layer_name, expert_id)
        if refs is None:
            raise KeyError(f"Expert {layer_name}.{expert_id} not found in local store")

        refs_by_file: dict[str, list[TensorRef]] = defaultdict(list)
        for ref in refs:
            if skip_suffixes and ref.relative_key.endswith(skip_suffixes):
                continue
            refs_by_file[ref.file_name].append(ref)

        tensors: dict[str, torch.Tensor] = {}
        for file_name, refs_for_file in refs_by_file.items():
            file_path = self.model_dir / file_name
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                for ref in refs_for_file:
                    tensors[ref.relative_key] = handle.get_tensor(ref.full_key).contiguous()
        return tensors

    def copy_expert_into(
        self,
        layer_name: str,
        expert_id: int,
        dst_tensors: dict[str, torch.Tensor],
        *,
        skip_suffixes: tuple[str, ...] = (),
    ) -> None:
        refs = self._resolve_refs(layer_name, expert_id)
        if refs is None:
            raise KeyError(f"Expert {layer_name}.{expert_id} not found in local store")

        suffix_to_name = {
            relative_name.split(".", 1)[1]: relative_name for relative_name in dst_tensors
        }
        refs_by_file: dict[str, list[TensorRef]] = defaultdict(list)
        for ref in refs:
            if skip_suffixes and ref.relative_key.endswith(skip_suffixes):
                continue
            refs_by_file[ref.file_name].append(ref)

        for file_name, refs_for_file in refs_by_file.items():
            file_path = self.model_dir / file_name
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                for ref in refs_for_file:
                    suffix = ref.relative_key.split(".", 1)[1]
                    dst_name = suffix_to_name.get(suffix)
                    if dst_name is None:
                        continue
                    dst_tensors[dst_name].copy_(handle.get_tensor(ref.full_key))

    def _build_index(self) -> dict[tuple[str, int], tuple[TensorRef, ...]]:
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
        else:
            weight_map = {}
            for shard_path in sorted(self.model_dir.glob("*.safetensors")):
                with safe_open(shard_path, framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        weight_map[key] = shard_path.name

        expert_refs: dict[tuple[str, int], list[TensorRef]] = defaultdict(list)
        for full_key, file_name in weight_map.items():
            match = _EXPERT_KEY_RE.match(full_key)
            if match is None:
                continue
            layer_name = match.group("prefix")
            expert_id = int(match.group("expert"))
            suffix = match.group("suffix")
            expert_refs[(layer_name, expert_id)].append(
                TensorRef(
                    file_name=file_name,
                    full_key=full_key,
                    relative_key=f"{expert_id}.{suffix}",
                )
            )

        logger.info(
            "Indexed %d local expert entries from %s",
            len(expert_refs),
            self.model_dir,
        )
        return {key: tuple(value) for key, value in expert_refs.items()}

    def _resolve_refs(
        self, layer_name: str, expert_id: int
    ) -> tuple[TensorRef, ...] | None:
        for candidate in _iter_layer_name_aliases(layer_name):
            refs = self._expert_refs.get((candidate, expert_id))
            if refs is not None:
                return refs
        return None


def _iter_layer_name_aliases(layer_name: str) -> tuple[str, ...]:
    aliases: list[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in aliases:
            aliases.append(candidate)

    add(layer_name)
    if layer_name.startswith("model."):
        add(layer_name.removeprefix("model."))
    else:
        add(f"model.{layer_name}")

    if layer_name.startswith("language_model.model."):
        suffix = layer_name.removeprefix("language_model.model.")
        add(f"model.language_model.{suffix}")
        add(f"model.{suffix}")
    if layer_name.startswith("model.language_model."):
        suffix = layer_name.removeprefix("model.language_model.")
        add(f"language_model.model.{suffix}")
        add(f"language_model.{suffix}")
    if layer_name.startswith("mtp."):
        add(f"model.{layer_name}")
    if layer_name.startswith("model.mtp."):
        add(layer_name.removeprefix("model."))

    return tuple(aliases)
