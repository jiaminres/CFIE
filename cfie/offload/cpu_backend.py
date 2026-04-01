"""CPU pinned-memory cache for MoE expert bundles."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch

from cfie.logger import init_logger
from cfie.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)


@dataclass(slots=True)
class ExpertBundle:
    tensors: dict[str, torch.Tensor]
    nbytes: int
    pinned: bool = False


def bundle_nbytes(tensors: dict[str, torch.Tensor]) -> int:
    return int(sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()))


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
        bundle = ExpertBundle(
            tensors=self._maybe_pin_bundle(tensors),
            nbytes=bundle_nbytes(tensors),
            pinned=is_pin_memory_available(),
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

        pinned_tensors: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            cpu_tensor = tensor.contiguous()
            if cpu_tensor.device.type != "cpu":
                cpu_tensor = cpu_tensor.to(device="cpu")
            if not cpu_tensor.is_pinned():
                cpu_tensor = cpu_tensor.pin_memory()
            pinned_tensors[name] = cpu_tensor
        return pinned_tensors
