"""Checkpoint 张量加载——按 key filter 迭代 safetensors 中的张量。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import torch

CheckpointKeyFilter = Callable[[str], bool]

HF_INDEX_FILENAMES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


@dataclass(slots=True)
class CheckpointTensorLoadStats:
    index_file_count: int = 0
    index_key_count: int = 0
    indexed_shard_count: int = 0
    selected_key_count: int = 0
    selected_shard_count: int = 0
    filtered_key_count: int = 0
    opened_file_count: int = 0
    yielded_tensor_count: int = 0
    yielded_tensor_elements: int = 0
    yielded_tensor_bytes: int = 0

    def __post_init__(self) -> None:
        _require_non_negative_int("index_file_count", self.index_file_count)
        _require_non_negative_int("index_key_count", self.index_key_count)
        _require_non_negative_int("indexed_shard_count", self.indexed_shard_count)
        _require_non_negative_int("selected_key_count", self.selected_key_count)
        _require_non_negative_int("selected_shard_count", self.selected_shard_count)
        _require_non_negative_int("filtered_key_count", self.filtered_key_count)
        _require_non_negative_int("opened_file_count", self.opened_file_count)
        _require_non_negative_int("yielded_tensor_count", self.yielded_tensor_count)
        _require_non_negative_int(
            "yielded_tensor_elements",
            self.yielded_tensor_elements,
        )
        _require_non_negative_int("yielded_tensor_bytes", self.yielded_tensor_bytes)

    def record_tensor(self, tensor: Any) -> None:
        self.yielded_tensor_count += 1
        self.yielded_tensor_elements += _tensor_numel(tensor)
        self.yielded_tensor_bytes += _tensor_nbytes(tensor)

    def to_dict(self) -> dict[str, int]:
        return {
            "index_file_count": self.index_file_count,
            "index_key_count": self.index_key_count,
            "indexed_shard_count": self.indexed_shard_count,
            "selected_key_count": self.selected_key_count,
            "selected_shard_count": self.selected_shard_count,
            "filtered_key_count": self.filtered_key_count,
            "opened_file_count": self.opened_file_count,
            "yielded_tensor_count": self.yielded_tensor_count,
            "yielded_tensor_elements": self.yielded_tensor_elements,
            "yielded_tensor_bytes": self.yielded_tensor_bytes,
        }


def iter_checkpoint_tensors(
    checkpoint: str | Path,
    *,
    key_filter: CheckpointKeyFilter | None = None,
    stats: CheckpointTensorLoadStats | None = None,
) -> Iterator[tuple[str, Any]]:
    # 迭代 safetensors/PyTorch checkpoint 中的张量
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        yield from _iter_checkpoint_dir(checkpoint_path, key_filter, stats)
        return
    yield from _iter_checkpoint_file(checkpoint_path, key_filter, stats)


def _iter_checkpoint_dir(
    directory: Path,
    key_filter: CheckpointKeyFilter | None,
    stats: CheckpointTensorLoadStats | None,
) -> Iterator[tuple[str, Any]]:
    if not directory.exists():
        raise FileNotFoundError(f"checkpoint directory not found: {directory}")
    index_path = _find_checkpoint_index(directory)
    if index_path is not None:
        yield from _iter_checkpoint_index(directory, index_path, key_filter, stats)
        return

    files = tuple(
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and _is_supported_checkpoint_file(path)
    )
    if not files:
        raise FileNotFoundError(
            f"no supported checkpoint files found under {directory}"
        )
    for path in files:
        yield from _iter_checkpoint_file(path, key_filter, stats)


def _iter_checkpoint_index(
    directory: Path,
    index_path: Path,
    key_filter: CheckpointKeyFilter | None,
    stats: CheckpointTensorLoadStats | None,
) -> Iterator[tuple[str, Any]]:
    weight_map = _load_weight_map(index_path)
    original_key_count = len(weight_map)
    if stats is not None:
        stats.index_file_count += 1
        stats.index_key_count += original_key_count
        stats.indexed_shard_count += len(set(weight_map.values()))
    if key_filter is not None:
        weight_map = {
            name: shard_name
            for name, shard_name in weight_map.items()
            if key_filter(name)
        }
    if stats is not None:
        stats.selected_key_count += len(weight_map)
        stats.filtered_key_count += original_key_count - len(weight_map)
    if not weight_map:
        return
    names_by_shard = _group_names_by_shard(weight_map)
    if stats is not None:
        stats.selected_shard_count += len(names_by_shard)
    for shard_name in sorted(names_by_shard):
        shard_path = directory / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(
                f"checkpoint shard {shard_name!r} from {index_path} not found"
            )
        requested_names = tuple(sorted(names_by_shard[shard_name]))
        yield from _iter_checkpoint_file_keys(shard_path, requested_names, stats)


def _iter_checkpoint_file(
    path: Path,
    key_filter: CheckpointKeyFilter | None,
    stats: CheckpointTensorLoadStats | None,
) -> Iterator[tuple[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        yield from _iter_safetensors(path, key_filter, stats)
        return
    if suffix in {".bin", ".pt", ".pth"}:
        yield from _iter_torch_checkpoint(path, key_filter, stats)
        return
    raise ValueError(f"unsupported checkpoint file extension: {path}")


def _iter_checkpoint_file_keys(
    path: Path,
    names: tuple[str, ...],
    stats: CheckpointTensorLoadStats | None,
) -> Iterator[tuple[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        yield from _iter_safetensors_keys(path, names, stats)
        return
    if suffix in {".bin", ".pt", ".pth"}:
        yield from _iter_torch_checkpoint_keys(path, names, stats)
        return
    raise ValueError(f"unsupported checkpoint file extension: {path}")


def _iter_torch_checkpoint(
    path: Path,
    key_filter: CheckpointKeyFilter | None,
    stats: CheckpointTensorLoadStats | None,
) -> Iterator[tuple[str, Any]]:
    if stats is not None:
        stats.opened_file_count += 1
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(payload, path)
    for name in sorted(state_dict):
        if key_filter is not None and not key_filter(name):
            if stats is not None:
                stats.filtered_key_count += 1
            continue
        tensor = state_dict[name]
        if stats is not None:
            stats.selected_key_count += 1
            stats.record_tensor(tensor)
        yield name, tensor


def _iter_torch_checkpoint_keys(
    path: Path,
    names: tuple[str, ...],
    stats: CheckpointTensorLoadStats | None,
) -> Iterator[tuple[str, Any]]:
    if stats is not None:
        stats.opened_file_count += 1
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(payload, path)
    for name in names:
        if name not in state_dict:
            raise KeyError(f"checkpoint shard {path} does not contain {name!r}")
        tensor = state_dict[name]
        if stats is not None:
            stats.record_tensor(tensor)
        yield name, tensor


def _iter_safetensors(
    path: Path,
    key_filter: CheckpointKeyFilter | None,
    stats: CheckpointTensorLoadStats | None,
) -> Iterator[tuple[str, Any]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "reading .safetensors checkpoints requires safetensors"
        ) from exc

    with safe_open(path, framework="pt", device="cpu") as handle:
        if stats is not None:
            stats.opened_file_count += 1
        for name in sorted(handle.keys()):
            if key_filter is not None and not key_filter(name):
                if stats is not None:
                    stats.filtered_key_count += 1
                continue
            tensor = handle.get_tensor(name)
            if stats is not None:
                stats.selected_key_count += 1
                stats.record_tensor(tensor)
            yield name, tensor


def _iter_safetensors_keys(
    path: Path,
    names: tuple[str, ...],
    stats: CheckpointTensorLoadStats | None,
) -> Iterator[tuple[str, Any]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "reading .safetensors checkpoints requires safetensors"
        ) from exc

    with safe_open(path, framework="pt", device="cpu") as handle:
        if stats is not None:
            stats.opened_file_count += 1
        available = set(handle.keys())
        for name in names:
            if name not in available:
                raise KeyError(f"checkpoint shard {path} does not contain {name!r}")
            tensor = handle.get_tensor(name)
            if stats is not None:
                stats.record_tensor(tensor)
            yield name, tensor


def _find_checkpoint_index(directory: Path) -> Path | None:
    for filename in HF_INDEX_FILENAMES:
        path = directory / filename
        if path.exists():
            return path
    indexes = tuple(sorted(directory.glob("*.index.json")))
    if not indexes:
        return None
    return indexes[0]


def _load_weight_map(index_path: Path) -> dict[str, str]:
    try:
        with index_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid checkpoint index JSON: {index_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint index must be an object: {index_path}")
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError(f"checkpoint index has no non-empty weight_map: {index_path}")
    normalized: dict[str, str] = {}
    for name, shard_name in weight_map.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"checkpoint index has invalid weight name: {index_path}")
        if not isinstance(shard_name, str) or not shard_name:
            raise ValueError(f"checkpoint index has invalid shard name: {index_path}")
        shard_path = Path(shard_name)
        if shard_path.is_absolute() or ".." in shard_path.parts:
            raise ValueError(
                f"checkpoint index shard path must be relative: {shard_name!r}"
            )
        normalized[name] = shard_name
    return normalized


def _group_names_by_shard(weight_map: Mapping[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name, shard_name in weight_map.items():
        grouped.setdefault(shard_name, []).append(name)
    return grouped


def _extract_state_dict(payload: Any, path: Path) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        if _looks_like_state_dict(payload):
            return payload
        for key in ("state_dict", "model", "module"):
            nested = payload.get(key)
            if isinstance(nested, Mapping) and _looks_like_state_dict(nested):
                return nested
    raise ValueError(f"checkpoint does not contain a tensor state dict: {path}")


def _looks_like_state_dict(payload: Mapping[str, Any]) -> bool:
    if not payload:
        return False
    return all(isinstance(key, str) for key in payload)


def _is_supported_checkpoint_file(path: Path) -> bool:
    return path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"}


def _tensor_numel(tensor: Any) -> int:
    numel = getattr(tensor, "numel", None)
    if numel is None:
        return 0
    return int(numel())


def _tensor_nbytes(tensor: Any) -> int:
    element_size = getattr(tensor, "element_size", None)
    if element_size is None:
        return 0
    return _tensor_numel(tensor) * int(element_size())
