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
    # bundle 内每个张量按相对名字索引，通常对应一个完整 expert 的若干参数。
    tensors: dict[str, torch.Tensor]
    # 记录整个 bundle 的总字节数，便于 cache 做容量控制。
    nbytes: int
    # 标记 bundle 是否已经放在 pinned CPU memory 上。
    pinned: bool = False


def bundle_nbytes(tensors: dict[str, torch.Tensor]) -> int:
    # 按所有张量的元素数 * 单元素字节数求和，得到 bundle 总大小。
    return int(sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()))


class PinnedExpertCache:
    """Simple LRU cache for cold experts kept in CPU memory."""

    def __init__(self, capacity_bytes: int):
        # 记录 CPU cache 的最大容量；负值统一收敛到 0。
        self.capacity_bytes = max(0, int(capacity_bytes))
        # current_bytes 跟踪当前 cache 中已占用的总字节数。
        self.current_bytes = 0
        # OrderedDict 以插入/访问顺序维护 LRU 队列，键为 (layer_name, expert_id)。
        self._entries: OrderedDict[tuple[str, int], ExpertBundle] = OrderedDict()

    def contains(self, key: tuple[str, int]) -> bool:
        # 仅检查 key 是否已经缓存在 CPU 中。
        return key in self._entries

    def get(self, key: tuple[str, int]) -> ExpertBundle | None:
        # 读取 bundle；未命中时返回 None。
        bundle = self._entries.get(key)
        if bundle is None:
            return None
        # 命中后把该项移动到队尾，表示最近使用过。
        self._entries.move_to_end(key)
        return bundle

    def put(self, key: tuple[str, int], tensors: dict[str, torch.Tensor]) -> ExpertBundle:
        # 插入前先尝试把 bundle 规范化到 CPU/pinned memory 上。
        bundle = ExpertBundle(
            tensors=self._maybe_pin_bundle(tensors),
            nbytes=bundle_nbytes(tensors),
            pinned=is_pin_memory_available(),
        )
        # 单个 bundle 本身就超过容量时，不进入 LRU，只把结果原样返回给调用方。
        if bundle.nbytes > self.capacity_bytes:
            return bundle

        # 若 key 已存在，先弹出旧值，避免重复计算 current_bytes。
        existing = self._entries.pop(key, None)
        if existing is not None:
            self.current_bytes -= existing.nbytes

        # 按 LRU 顺序持续驱逐最旧项，直到能容纳新 bundle。
        while self._entries and self.current_bytes + bundle.nbytes > self.capacity_bytes:
            _, evicted = self._entries.popitem(last=False)
            self.current_bytes -= evicted.nbytes

        # 把新 bundle 放到队尾，并更新总占用。
        self._entries[key] = bundle
        self.current_bytes += bundle.nbytes
        return bundle

    def _maybe_pin_bundle(
        self, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # 平台不支持 pinned memory 时，直接返回原始 tensors。
        if not is_pin_memory_available():
            return tensors

        # 否则逐张量确保其位于 CPU、是连续内存，并尽量 pin 住。
        pinned_tensors: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            cpu_tensor = tensor.contiguous()
            if cpu_tensor.device.type != "cpu":
                cpu_tensor = cpu_tensor.to(device="cpu")
            if not cpu_tensor.is_pinned():
                cpu_tensor = cpu_tensor.pin_memory()
            pinned_tensors[name] = cpu_tensor
        return pinned_tensors
