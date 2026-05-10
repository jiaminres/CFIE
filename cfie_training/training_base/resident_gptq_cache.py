"""GPU Resident GPTQ 冷专家缓存——有限容量，LRU 淘汰，支持 lock/unlock/prefetch（设计文档 Section 9.4）。

提供两种角色：
- Prefill 阶段: 逐层流式加载，用完即弃
- Decode 训练阶段: predictor 预取 + router 命中后缓存复用
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol
from cfie_training.training_base.gptq_cache_store import GptqCacheStore

def _require_non_empty_string(name: str, value: str) -> None:
    if not value.strip(): raise ValueError(f"{name} must be a non-empty string")
def _require_non_negative_int(name: str, value: int) -> None:
    if value < 0: raise ValueError(f"{name} must be >= 0")
def _require_positive_int(name: str, value: int) -> None:
    if value < 1: raise ValueError(f"{name} must be >= 1")

# ──────── 后端接口 ────────
class ResidentGptqPayloadBackend(Protocol):
    def load(self, bundle_id: str, payload: bytes) -> Any: ...
    def synchronize(self, bundle_id: str, payload: Any) -> None: ...
    def release(self, bundle_id: str, payload: Any) -> None: ...
    def num_bytes(self, payload: Any) -> int: ...

# ──────── Bytes 后端（CPU 内存）────────
@dataclass(frozen=True, slots=True)
class BytesResidentGptqBackend:
    def load(self, bundle_id: str, payload: bytes) -> bytes: return payload
    def release(self, bundle_id: str, payload: Any) -> None: pass
    def synchronize(self, bundle_id: str, payload: bytes) -> None: pass
    def num_bytes(self, payload: bytes) -> int: return len(payload)

# ──────── Tensor 后端（GPU，支持 CUDA stream 异步 H2D）────────
@dataclass(slots=True)
class TorchTensorResidentGptqBackend:
    device: Any = "cpu"; pin_memory: bool = False; non_blocking: bool = True
    use_cuda_stream: bool = True
    _torch_device: Any = field(init=False, repr=False)
    _upload_stream: Any = field(init=False, repr=False, default=None)
    _ready_events: dict[int, Any] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        torch = _import_torch(); self._torch_device = torch.device(self.device)
        if self.use_cuda_stream and self._torch_device.type == "cuda" and torch.cuda.is_available():
            with torch.cuda.device(self._torch_device): self._upload_stream = torch.cuda.Stream(device=self._torch_device)

    def load(self, bundle_id: str, payload: bytes) -> Any:
        """从 CPU bytes → GPU tensor（独立 CUDA stream，非阻塞 H2D）。"""
        torch = _import_torch(); host = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        if self._torch_device.type == "cpu": return host.clone() if not self.pin_memory else self._copy_to_pinned_cpu(host)
        upload = self._copy_to_pinned_cpu(host) if self.pin_memory else host
        if self._upload_stream is None: return upload.to(self._torch_device, non_blocking=self.non_blocking)
        with torch.cuda.stream(self._upload_stream):
            resident = upload.to(self._torch_device, non_blocking=self.non_blocking)
            ready_event = torch.cuda.Event(); ready_event.record(self._upload_stream)
        self._ready_events[id(resident)] = ready_event
        return resident

    def release(self, bundle_id: str, payload: Any) -> None: self._ready_events.pop(id(payload), None)

    def synchronize(self, bundle_id: str, payload: Any) -> None:
        """等待异步 H2D 完成（主 stream 上 wait_event）。"""
        ready_event = self._ready_events.get(id(payload))
        if ready_event is None: return
        torch = _import_torch()
        if payload.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.current_stream(payload.device).wait_event(ready_event)

    def num_bytes(self, payload: Any) -> int: return int(payload.numel() * payload.element_size())
    def _copy_to_pinned_cpu(self, host_tensor: Any) -> Any:
        torch = _import_torch()
        if not torch.cuda.is_available(): return host_tensor.clone()
        pinned = torch.empty(host_tensor.numel(), dtype=torch.uint8, device="cpu", pin_memory=True)
        pinned.copy_(host_tensor, non_blocking=False); return pinned

# ──────── Cache 数据结构 ────────
@dataclass(frozen=True, slots=True)
class ResidentGptqCacheEntry:
    bundle_id: str; payload: Any; num_bytes: int; last_access_tick: int; locked: bool = False
    def __post_init__(self) -> None:
        _require_non_empty_string("bundle_id", self.bundle_id)
        _require_positive_int("num_bytes", self.num_bytes)
        _require_non_negative_int("last_access_tick", self.last_access_tick)

@dataclass(frozen=True, slots=True)
class ResidentGptqCacheStats:
    hits: int = 0; misses: int = 0; evictions: int = 0; loads: int = 0
    @property
    def requests(self) -> int: return self.hits + self.misses
    @property
    def miss_rate(self) -> float: return self.misses / self.requests if self.requests else 0.0

# ──────── ResidentGptqCache ────────
@dataclass(slots=True)
class ResidentGptqCache:
    """GPU 冷专家缓存——有限容量 + LRU + lock。"""
    store: GptqCacheStore; capacity_bytes: int
    backend: ResidentGptqPayloadBackend = field(default_factory=BytesResidentGptqBackend)
    _entries: dict[str, ResidentGptqCacheEntry] = field(default_factory=dict)
    _used_bytes: int = 0; _tick: int = 0; _hits: int = 0; _misses: int = 0
    _evictions: int = 0; _loads: int = 0

    def __post_init__(self) -> None: _require_positive_int("capacity_bytes", self.capacity_bytes)

    @property
    def used_bytes(self) -> int: return self._used_bytes
    @property
    def resident_bundle_ids(self) -> tuple[str, ...]: return tuple(sorted(self._entries))
    @property
    def locked_bundle_ids(self) -> tuple[str, ...]: return tuple(sorted(k for k, e in self._entries.items() if e.locked))
    @property
    def stats(self) -> ResidentGptqCacheStats: return ResidentGptqCacheStats(hits=self._hits, misses=self._misses, evictions=self._evictions, loads=self._loads)

    # ── get: 从 cache 获取，未命中则 prefetch+wait ──
    def get(self, bundle_id: str) -> Any:
        """从 GPU cache 获取 bundle（命中刷新 LRU tick，未命中触发 prefetch+wait）。"""
        _require_non_empty_string("bundle_id", bundle_id); self._tick += 1
        entry = self._entries.get(bundle_id)
        if entry is not None: self._hits += 1; return self._refresh_and_sync(entry).payload
        self._misses += 1; return self.prefetch(bundle_id, wait=True)

    # ── put: 直接存入 GPU cache ──
    def put(self, bundle_id: str, payload: Any) -> None:
        """将已解码的专家权重直接存入 GPU cache。用于 router 命中后 CPU decode 结果缓存。"""
        _require_non_empty_string("bundle_id", bundle_id); self._tick += 1
        existing = self._entries.get(bundle_id)
        if existing is not None:
            self._entries[bundle_id] = ResidentGptqCacheEntry(bundle_id=bundle_id, payload=existing.payload,
                num_bytes=existing.num_bytes, last_access_tick=self._tick, locked=existing.locked)
            return
        # 计算 bytes（支持 tuple/list 形式的多个 tensor）
        if isinstance(payload, (tuple, list)): pb = sum(t.numel() * t.element_size() for t in payload if hasattr(t, "numel"))
        else: pb = self.backend.num_bytes(payload)
        self._evict_until_fits(pb)
        self._entries[bundle_id] = ResidentGptqCacheEntry(bundle_id=bundle_id, payload=payload, num_bytes=pb, last_access_tick=self._tick, locked=False)
        self._used_bytes += pb; self._loads += 1

    # ── prefetch: 预取（异步 H2D）──
    def prefetch(self, bundle_id: str, *, locked: bool = False, wait: bool = False) -> Any:
        """预取 bundle 到 GPU cache。wait=False 时不等待 H2D 完成（异步预取）。"""
        _require_non_empty_string("bundle_id", bundle_id); self._tick += 1
        existing = self._entries.get(bundle_id)
        if existing is not None:
            self._entries[bundle_id] = ResidentGptqCacheEntry(bundle_id=bundle_id, payload=existing.payload,
                num_bytes=existing.num_bytes, last_access_tick=self._tick, locked=existing.locked or locked)
            if wait: self._synchronize_entry(self._entries[bundle_id])
            return existing.payload
        record = self.store.records[bundle_id]
        if record.num_bytes > self.capacity_bytes: raise ValueError(f"bundle {bundle_id!r} has {record.num_bytes} bytes, exceeds cache capacity {self.capacity_bytes}")
        self._evict_until_fits(record.num_bytes)
        raw = self.store.read_bundle(bundle_id); payload = self.backend.load(bundle_id, raw)
        entry = ResidentGptqCacheEntry(bundle_id=bundle_id, payload=payload, num_bytes=record.num_bytes, last_access_tick=self._tick, locked=locked)
        self._entries[bundle_id] = entry; self._used_bytes += entry.num_bytes; self._loads += 1
        if wait: self._synchronize_entry(entry)
        return payload

    # ── prefetch_many ──
    def prefetch_many(self, bundle_ids: Iterable[str], *, locked: bool = False, wait: bool = False) -> tuple[str, ...]:
        """批量预取，返回新加载的 bundle_id 列表。"""
        loaded: list[str] = []
        for bid in bundle_ids:
            was = bid in self._entries
            self.prefetch(bid, locked=locked, wait=wait)
            if not was: loaded.append(bid)
        return tuple(loaded)

    # ── lock / unlock ──
    def lock(self, bundle_ids: Iterable[str]) -> None:
        """Lock bundle（防止淘汰，用于当前 step 激活的冷专家）。先 prefetch 再 lock。"""
        for bid in bundle_ids: self.prefetch(bid, locked=True)

    def unlock(self, bundle_ids: Iterable[str]) -> None:
        """Unlock bundle（backward 完成后释放，允许后续淘汰）。"""
        for bid in bundle_ids:
            entry = self._entries.get(bid)
            if entry is not None: self._entries[bid] = ResidentGptqCacheEntry(
                bundle_id=bid, payload=entry.payload, num_bytes=entry.num_bytes, last_access_tick=entry.last_access_tick, locked=False)

    def wait_ready(self, bundle_ids: Iterable[str] | None = None) -> tuple[str, ...]:
        """等待指定 bundle 的异步 H2D 完成。返回实际存在的 bundle ID 列表。"""
        if bundle_ids is None: targets = tuple(sorted(self._entries))
        else: targets = tuple(bundle_ids)
        ready: list[str] = []
        for bid in targets:
            entry = self._entries.get(bid)
            if entry is not None: self._synchronize_entry(entry); ready.append(bid)
        return tuple(ready)

    # ── 淘汰与清理 ──
    def invalidate(self, bundle_ids: Iterable[str]) -> tuple[str, ...]:
        removed = []
        for bid in tuple(bundle_ids):
            entry = self._entries.pop(bid, None)
            if entry is not None: self._release_entry(entry); removed.append(bid)
        return tuple(removed)

    def clear_unlocked(self) -> tuple[str, ...]:
        return self.invalidate(bid for bid, e in tuple(self._entries.items()) if not e.locked)

    def reset_stats(self) -> None: self._hits = 0; self._misses = 0; self._evictions = 0; self._loads = 0

    # ── 内部 ──
    def _refresh_and_sync(self, entry: ResidentGptqCacheEntry) -> ResidentGptqCacheEntry:
        updated = ResidentGptqCacheEntry(bundle_id=entry.bundle_id, payload=entry.payload, num_bytes=entry.num_bytes, last_access_tick=self._tick, locked=entry.locked)
        self._entries[entry.bundle_id] = updated; self._synchronize_entry(updated); return updated

    def _synchronize_entry(self, entry: ResidentGptqCacheEntry) -> None:
        if hasattr(self.backend, "synchronize"): self.backend.synchronize(entry.bundle_id, entry.payload)

    def _evict_until_fits(self, incoming_bytes: int) -> None:
        while self._used_bytes + incoming_bytes > self.capacity_bytes:
            victim_id = min((e for e in self._entries.values() if not e.locked), key=lambda e: e.last_access_tick, default=None)
            if victim_id is None: raise RuntimeError("cannot evict enough unlocked bundles")
            self._release_entry(self._entries.pop(victim_id.bundle_id)); self._evictions += 1

    def _release_entry(self, entry: ResidentGptqCacheEntry) -> None:
        self.backend.release(entry.bundle_id, entry.payload); self._used_bytes -= entry.num_bytes

def _import_torch() -> Any:
    try: import torch; return torch
    except ImportError: raise RuntimeError("TorchTensorResidentGptqBackend requires PyTorch") from None
