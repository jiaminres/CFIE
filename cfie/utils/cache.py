# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import UserDict
from collections.abc import Callable, Hashable, Iterator, KeysView, Mapping
from types import MappingProxyType
from typing import NamedTuple, TypeVar, cast, overload

import cachetools

# cache/dict 的 key 类型，要求必须可哈希。
_K = TypeVar("_K", bound=Hashable)
# cache 中存储的 value 类型。
_V = TypeVar("_V")
# 用于 `get(..., default=...)` / `pop(..., default=...)` 的辅助类型变量。
_T = TypeVar("_T")


class _Sentinel: ...


ALL_PINNED_SENTINEL = _Sentinel()


class _MappingOrderCacheView(UserDict[_K, _V]):
    def __init__(self, data: Mapping[_K, _V], ordered_keys: Mapping[_K, None]):
        super().__init__(data)
        self.ordered_keys = ordered_keys

    def __iter__(self) -> Iterator[_K]:
        return iter(self.ordered_keys)

    def keys(self) -> KeysView[_K]:
        return KeysView(self.ordered_keys)


class CacheInfo(NamedTuple):
    hits: int
    total: int

    @property
    def hit_ratio(self) -> float:
        if self.total == 0:
            return 0

        return self.hits / self.total

    def __sub__(self, other: "CacheInfo"):
        return CacheInfo(
            hits=self.hits - other.hits,
            total=self.total - other.total,
        )


class LRUCache(cachetools.LRUCache[_K, _V]):
    """带统计信息与 pin 语义扩展的 LRU 缓存。"""

    def __init__(
            self, capacity: float,
            getsizeof: Callable[[_V], float] | None = None
    ):
        super().__init__(capacity, getsizeof)

        # 被 pin 的 key 不会在普通 LRU 淘汰路径里被移除。
        self.pinned_items = set[_K]()

        # 记录累计命中/查询次数，供上层统计 cache 命中率。
        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)

    def __getitem__(self, key: _K, *, update_info: bool = True) -> _V:
        value = super().__getitem__(key)

        if update_info:
            self._hits += 1
            self._total += 1

        return value

    def __delitem__(self, key: _K) -> None:
        run_on_remove = key in self
        value = self.__getitem__(key, update_info=False)  # type: ignore[call-arg]
        super().__delitem__(key)
        if key in self.pinned_items:
            # TODO: 后续可补一个告警，提示这里删除的是 pinned 条目。
            self._unpin(key)
        if run_on_remove:
            self._on_remove(key, value)

    @property
    def cache(self) -> Mapping[_K, _V]:
        """按当前 LRU 顺序返回内部 cache 字典的只读视图。"""
        return _MappingOrderCacheView(
            self._Cache__data,  # type: ignore
            self.order,
        )

    @property
    def order(self) -> Mapping[_K, None]:
        """返回内部顺序字典的只读视图。"""
        return MappingProxyType(self._LRUCache__order)  # type: ignore

    @property
    def capacity(self) -> float:
        return self.maxsize

    @property
    def usage(self) -> float:
        if self.maxsize == 0:
            return 0

        return self.currsize / self.maxsize

    def stat(self, *, delta: bool = False) -> CacheInfo:
        """
        获取当前 cache 的累计命中数与查询数。

        如果 `delta=True`，则返回自上一次同样传入 `delta=True`
        以来的增量统计。
        """
        info = CacheInfo(hits=self._hits, total=self._total)

        if delta:
            info_delta = info - self._last_info
            self._last_info = info
            info = info_delta

        return info

    def touch(self, key: _K) -> None:
        try:
            self._LRUCache__order.move_to_end(key)  # type: ignore
        except KeyError:
            self._LRUCache__order[key] = None  # type: ignore

    @overload
    def get(self, key: _K, /) -> _V | None:
        ...

    @overload
    def get(self, key: _K, /, default: _V | _T) -> _V | _T:
        ...

    def get(self, key: _K, /, default: _V | _T | None = None) -> _V | _T | None:
        value: _V | _T | None
        if key in self:
            value = self.__getitem__(key, update_info=False)  # type: ignore[call-arg]

            self._hits += 1
        else:
            value = default

        self._total += 1
        return value

    @overload
    def pop(self, key: _K) -> _V:
        ...

    @overload
    def pop(self, key: _K, default: _V | _T) -> _V | _T:
        ...

    def pop(self, key: _K, default: _V | _T | None = None) -> _V | _T | None:
        value: _V | _T | None
        if key not in self:
            return default

        value = self.__getitem__(key, update_info=False)  # type: ignore[call-arg]
        self.__delitem__(key)
        return value

    def put(self, key: _K, value: _V) -> None:
        self.__setitem__(key, value)

    def pin(self, key: _K) -> None:
        """
        将某个 key 固定在 cache 中，使其不会按普通 LRU 顺序被淘汰。
        """
        if key not in self:
            raise ValueError(f"Cannot pin key: {key} not in cache.")
        self.pinned_items.add(key)

    def _unpin(self, key: _K) -> None:
        """
        取消某个 key 的固定状态，使其重新允许按 LRU 顺序被淘汰。
        """
        self.pinned_items.remove(key)

    def _on_remove(self, key: _K, value: _V | None) -> None:
        pass

    def remove_oldest(self, *, remove_pinned: bool = False) -> None:
        if len(self) == 0:
            return

        self.popitem(remove_pinned=remove_pinned)

    def _remove_old_if_needed(self) -> None:
        while self.currsize > self.capacity:
            self.remove_oldest()

    def popitem(self, remove_pinned: bool = False):
        """移除并返回当前最久未使用的 `(key, value)` 条目。"""
        if not remove_pinned:
            # 只移除当前 cache 中最老且未被 pin 的条目。
            lru_key = next(
                (key for key in self.order if key not in self.pinned_items),
                ALL_PINNED_SENTINEL,
            )
            if lru_key is ALL_PINNED_SENTINEL:
                raise RuntimeError(
                    "All items are pinned, cannot remove oldest from the cache."
                )
        else:
            lru_key = next(iter(self.order))
        value = self.pop(cast(_K, lru_key))
        return (lru_key, value)

    def clear(self) -> None:
        while len(self) > 0:
            self.remove_oldest(remove_pinned=True)

        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)
