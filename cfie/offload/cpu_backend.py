"""CPU pinned-memory cache for MoE expert bundles."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

import torch

from cfie.logger import init_logger
from cfie.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)


@dataclass(slots=True)
class ExpertBundle:
    tensors: dict[str, torch.Tensor]
    nbytes: int
    pinned: bool = False
    runtime_ready: bool = False
    storage: torch.Tensor | None = None
    storage_offset_bytes: int = 0


@dataclass(slots=True, frozen=True)
class PackedExpertTensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


def bundle_nbytes(tensors: dict[str, torch.Tensor]) -> int:
    return int(sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()))


def _normalize_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    cpu_tensor = tensor.detach()
    if cpu_tensor.device.type != "cpu":
        cpu_tensor = cpu_tensor.to(device="cpu")
    return cpu_tensor.contiguous()


def allocate_packed_cpu_tensor_views(
    specs: Sequence[PackedExpertTensorSpec],
    *,
    pin_memory: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ordered_specs = tuple(specs)
    total_bytes = int(
        sum(
            int(torch.Size(spec.shape).numel())
            * torch.empty((), dtype=spec.dtype).element_size()
            for spec in ordered_specs
        )
    )
    storage = torch.empty(
        total_bytes,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=bool(pin_memory),
    )
    views: dict[str, torch.Tensor] = {}
    offset = 0
    for spec in ordered_specs:
        numel = int(torch.Size(spec.shape).numel())
        itemsize = torch.empty((), dtype=spec.dtype).element_size()
        num_bytes = numel * itemsize
        view = storage[offset: offset + num_bytes].view(spec.dtype).view(spec.shape)
        views[spec.name] = view
        offset += num_bytes
    return storage, views


def pack_cpu_tensor_dict(
    tensors: dict[str, torch.Tensor],
    *,
    pin_memory: bool = False,
    runtime_ready: bool = False,
) -> ExpertBundle:
    ordered_items = list(tensors.items())
    storage, packed_views = allocate_packed_cpu_tensor_views(
        [
            PackedExpertTensorSpec(
                name=name,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
            )
            for name, tensor in ordered_items
        ],
        pin_memory=pin_memory,
    )
    for name, tensor in ordered_items:
        packed_views[name].copy_(_normalize_cpu_tensor(tensor))
    return ExpertBundle(
        tensors=packed_views,
        nbytes=bundle_nbytes(packed_views),
        pinned=bool(storage.is_pinned()),
        runtime_ready=runtime_ready,
        storage=storage,
        storage_offset_bytes=0,
    )


def pack_batched_cpu_tensor_dicts_by_expert(
    batched_tensors: dict[str, torch.Tensor],
    *,
    pin_memory: bool = False,
    runtime_ready: bool = False,
) -> list[ExpertBundle]:
    if not batched_tensors:
        return []

    ordered_items = list(batched_tensors.items())
    batch_size = int(ordered_items[0][1].shape[0])
    for name, tensor in ordered_items:
        if int(tensor.shape[0]) != batch_size:
            raise ValueError(
                "batched expert tensors must share the same batch dimension, "
                f"but {name!r} has batch={tensor.shape[0]} vs expected {batch_size}"
            )

    per_expert_specs = [
        PackedExpertTensorSpec(
            name=name,
            shape=tuple(tensor.shape[1:]),
            dtype=tensor.dtype,
        )
        for name, tensor in ordered_items
    ]
    per_expert_bytes = int(
        sum(
            int(torch.Size(spec.shape).numel())
            * torch.empty((), dtype=spec.dtype).element_size()
            for spec in per_expert_specs
        )
    )
    storage = torch.empty(
        batch_size * per_expert_bytes,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=bool(pin_memory),
    )

    bundles: list[ExpertBundle] = []
    for expert_index in range(batch_size):
        expert_slice = storage[
            expert_index * per_expert_bytes: (expert_index + 1) * per_expert_bytes
        ]
        views: dict[str, torch.Tensor] = {}
        offset = 0
        for name, batched_tensor in ordered_items:
            source = _normalize_cpu_tensor(batched_tensor[expert_index])
            num_bytes = int(source.numel() * source.element_size())
            view = expert_slice[offset: offset + num_bytes].view(source.dtype).view(
                source.shape
            )
            view.copy_(source)
            views[name] = view
            offset += num_bytes
        bundles.append(
            ExpertBundle(
                tensors=views,
                nbytes=per_expert_bytes,
                pinned=bool(storage.is_pinned()),
                runtime_ready=runtime_ready,
                storage=storage,
                storage_offset_bytes=expert_index * per_expert_bytes,
            )
        )
    return bundles


def pack_cpu_bundles_by_expert(
    bundles: Sequence[ExpertBundle],
    *,
    pin_memory: bool = False,
    runtime_ready: bool | None = None,
) -> list[ExpertBundle]:
    if not bundles:
        return []

    first_bundle = bundles[0]
    ordered_names = list(first_bundle.tensors.keys())
    per_expert_specs = [
        PackedExpertTensorSpec(
            name=name,
            shape=tuple(first_bundle.tensors[name].shape),
            dtype=first_bundle.tensors[name].dtype,
        )
        for name in ordered_names
    ]
    per_expert_bytes = int(
        sum(
            int(torch.Size(spec.shape).numel())
            * torch.empty((), dtype=spec.dtype).element_size()
            for spec in per_expert_specs
        )
    )
    storage = torch.empty(
        len(bundles) * per_expert_bytes,
        dtype=torch.uint8,
        device="cpu",
        pin_memory=bool(pin_memory),
    )

    packed_bundles: list[ExpertBundle] = []
    resolved_runtime_ready = (
        all(bundle.runtime_ready for bundle in bundles)
        if runtime_ready is None
        else bool(runtime_ready)
    )
    for expert_index, bundle in enumerate(bundles):
        if list(bundle.tensors.keys()) != ordered_names:
            raise ValueError("all expert bundles must share the same tensor fields")
        expert_slice = storage[
            expert_index * per_expert_bytes: (expert_index + 1) * per_expert_bytes
        ]
        views: dict[str, torch.Tensor] = {}
        offset = 0
        for spec in per_expert_specs:
            source = _normalize_cpu_tensor(bundle.tensors[spec.name])
            if source.dtype != spec.dtype or tuple(source.shape) != spec.shape:
                raise ValueError(
                    "all expert bundles must share the same tensor shapes and dtypes"
                )
            num_bytes = int(source.numel() * source.element_size())
            view = expert_slice[offset: offset + num_bytes].view(source.dtype).view(
                source.shape
            )
            view.copy_(source)
            views[spec.name] = view
            offset += num_bytes
        packed_bundles.append(
            ExpertBundle(
                tensors=views,
                nbytes=per_expert_bytes,
                pinned=bool(storage.is_pinned()),
                runtime_ready=resolved_runtime_ready,
                storage=storage,
                storage_offset_bytes=expert_index * per_expert_bytes,
            )
        )
    return packed_bundles


class PinnedExpertCache:
    """Simple LRU cache for cold experts kept in CPU memory."""

    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = max(0, int(capacity_bytes))
        self.current_bytes = 0
        self._entries: OrderedDict[tuple[str, int], ExpertBundle] = OrderedDict()

    def contains(self, key: tuple[str, int]) -> bool:
        return key in self._entries

    def get(self, key: tuple[str, int]) -> ExpertBundle | None:
        bundle = self._entries.get(key)
        if bundle is None:
            return None
        self._entries.move_to_end(key)
        return bundle

    def put(self, key: tuple[str, int], tensors: dict[str, torch.Tensor]) -> ExpertBundle:
        bundle = pack_cpu_tensor_dict(
            tensors,
            pin_memory=is_pin_memory_available(),
        )
        if bundle.nbytes > self.capacity_bytes:
            return bundle

        existing = self._entries.pop(key, None)
        if existing is not None:
            self.current_bytes -= existing.nbytes

        while self._entries and self.current_bytes + bundle.nbytes > self.capacity_bytes:
            _, evicted = self._entries.popitem(last=False)
            self.current_bytes -= evicted.nbytes

        self._entries[key] = bundle
        self.current_bytes += bundle.nbytes
        return bundle

    def _maybe_pin_bundle(
        self, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if not is_pin_memory_available():
            return tensors

        return pack_cpu_tensor_dict(
            tensors,
            pin_memory=True,
        ).tensors
