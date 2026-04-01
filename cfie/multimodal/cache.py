# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import operator
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from multiprocessing.synchronize import Lock as LockType
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar, cast

import torch
from typing_extensions import override

import cfie.envs as envs
from cfie.distributed.device_communicators.shm_object_storage import (
    MsgpackSerde,
    SingleWriterShmObjectStorage,
    SingleWriterShmRingBuffer,
)
from cfie.logger import init_logger
from cfie.utils.cache import CacheInfo, LRUCache
from cfie.utils.jsontree import json_count_leaves, json_map_leaves, json_reduce_leaves
from cfie.utils.mem_constants import GiB_bytes, MiB_bytes
from cfie.utils.mem_utils import format_gib

from .inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    NestedTensors,
)

if TYPE_CHECKING:
    from cfie.config import ModelConfig, CfieConfig

    from .processing.processor import ResolvedPromptUpdate

logger = init_logger(__name__)


class MultiModalProcessorCacheItem:
    """
    The data to store inside `MultiModalProcessorOnlyCache`.

    Args:
        item: The processed tensor data corresponding to a multi-modal item.
        prompt_updates: The prompt updates corresponding to `item`.
    """

    def __init__(
        self,
        item: MultiModalKwargsItem,
        prompt_updates: Sequence["ResolvedPromptUpdate"],
    ) -> None:
        super().__init__()

        self.item = item
        self.prompt_updates = prompt_updates


class MultiModalProcessorCacheItemMetadata:
    """
    The metadata to store inside `MultiModalProcessorSenderCache`.

    Args:
        item: The processed tensor data corresponding to a multi-modal item.
            Since P1 already stores the tensor data, we only store its size
            metadata in P0 to reduce memory usage. The size metadata is still
            needed to keep the same cache eviction policy as P0.
        prompt_updates: The prompt updates corresponding to `item`.
            This needs to stay on P0 because for some models, they are
            dependent on the processed tensor data (cached on P1).
    """

    def __init__(
        self,
        item: MultiModalKwargsItem,
        prompt_updates: Sequence["ResolvedPromptUpdate"],
    ) -> None:
        super().__init__()

        self.item_size = MultiModalCache.get_item_size(item)
        self.prompt_updates = prompt_updates


MultiModalCacheValue: TypeAlias = (
    MultiModalProcessorCacheItem
    | MultiModalProcessorCacheItemMetadata
    | MultiModalKwargsItems
    | MultiModalKwargsItem
    | Mapping[str, NestedTensors]
)

_V = TypeVar("_V", bound=MultiModalCacheValue)


class MultiModalCache:
    """多模态缓存工具类，负责估算缓存项大小并创建对应的 LRU cache。"""

    @classmethod
    def get_leaf_size(cls, leaf: object) -> int:
        if isinstance(leaf, MultiModalProcessorCacheItem):
            return cls.get_leaf_size(leaf.item)
        if isinstance(leaf, MultiModalProcessorCacheItemMetadata):
            return leaf.item_size

        # 这些类型虽然内部表现得像映射/容器，但本身并不是 dict 的子类。
        if isinstance(
            leaf,
            (MultiModalKwargsItems, MultiModalKwargsItem, MultiModalFieldElem),
        ):
            return cls.get_item_size(leaf.data)  # type: ignore

        # `sys.getsizeof` 不能正确反映 tensor 实际占用的数据字节数。
        if isinstance(leaf, torch.Tensor):
            return leaf.nbytes

        return sys.getsizeof(leaf)

    @classmethod
    def get_item_size(
        cls,
        value: MultiModalCacheValue,
        *,
        debug: bool = False,
    ) -> int:
        size = json_reduce_leaves(
            operator.add, json_map_leaves(cls.get_leaf_size, value)
        )

        if debug:
            leaf_count = json_count_leaves(value)
            logger.debug(
                "Calculated size of %s to be %s GiB (%d leaves)",
                type(value),
                format_gib(size),
                leaf_count,
            )

        return size

    @classmethod
    def get_item_complexity(cls, value: MultiModalCacheValue) -> int:
        """
        获取一个多模态缓存值中叶子节点的数量。

        这个指标可以作为结构复杂度的近似度量，用来辅助调试缓存性能
        或分析数据形态。

        Args:
            value: 要分析的多模态缓存值。

        Returns:
            该嵌套结构中的叶子节点数量。
        """
        return json_count_leaves(value)

    @classmethod
    def get_lru_cache(
        cls,
        capacity_gb: float,
        value_type: type[_V],
        *,
        debug: bool = False,
    ) -> LRUCache[str, _V]:
        # `capacity_gb` 是整个 LRU cache 的容量上限，单位为 GiB。
        # `value_type` 表示这个 LRU 逻辑上要存放的 value 类型，主要用于
        # 调用点表达语义与类型推导；当前实现不会在运行时直接使用它做分支。
        return LRUCache(
            GiB_bytes * capacity_gb,
            getsizeof=lambda x: cls.get_item_size(x, debug=debug),
        )


_I = TypeVar("_I", contravariant=True)
_O = TypeVar("_O", covariant=True)


class BaseMultiModalCache(ABC, Generic[_I, _O]):
    """
    Abstract base class to read/write multi-modal items from cache.

    The idea of multi-modal caching is based on having a client and server
    where the client executes in the frontend process (=P0) and
    the server in the core process (=P1). The data flow is as follows:

    ```
                  is_cached() x N    get_and_update()
    P0: From API -----------------> -----------------> To P1

                 get_and_update()
    P1: From P0 -----------------> To model
    ```

    `is_cached()` can be called any number of times in P0. However,
    `get_and_update()` must be called in P0 and P1 one after another
    so that their cache eviction order remains the same.

    This ensures that the keys in P0 and P1 caches are mirrored,
    allowing us to determine whether a key is cached in P1 by looking
    up the P0 cache, without having to communicate with P1.
    """

    @abstractmethod
    def get_and_update_item(
        self,
        mm_item: _I,
        mm_hash: str,
    ) -> _O:
        """
        Possibly update a multi-modal item based on whether it is
        in the underlying cache.

        This update is done out-of-place and updates the cache eviction order.

        Args:
            mm_item: The multi-modal item to update.
            mm_hash: The hash of `mm_item`.

        Returns:
            The update multi-modal item.
        """
        raise NotImplementedError

    def get_and_update(
        self,
        mm_items: Sequence[_I],
        mm_hashes: list[str],
    ) -> list[_O]:
        """
        Possibly update a sequence of multi-modal items based on whether they
        are in the underlying cache.

        This update is done out-of-place and updates the cache eviction order.

        Args:
            mm_items: The multi-modal items to update.
            mm_hashes: The hash of each item in `mm_items`.

        Returns:
            A new list of updated multi-modal items.
        """
        assert len(mm_items) == len(mm_hashes)

        return [
            self.get_and_update_item(mm_item, mm_hash)
            for mm_item, mm_hash in zip(mm_items, mm_hashes)
        ]

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear the underlying cache."""
        raise NotImplementedError


MultiModalProcessorCacheInItem: TypeAlias = (
    tuple[MultiModalKwargsItem, Sequence["ResolvedPromptUpdate"]] | None
)


MultiModalProcessorCacheOutItem: TypeAlias = tuple[
    MultiModalKwargsItem | None, Sequence["ResolvedPromptUpdate"]
]

# P0 侧多模态 processor cache 的抽象接口。
class BaseMultiModalProcessorCache(
    BaseMultiModalCache[MultiModalProcessorCacheInItem, MultiModalProcessorCacheOutItem]
):
    """P0 侧缓存需要实现的接口。"""

    @abstractmethod
    def is_cached_item(self, mm_hash: str) -> bool:
        """
        检查某个多模态条目是否已经存在于底层缓存中。

        这个操作**不会**更新缓存的淘汰顺序。

        Args:
            mm_hash: 要检查的多模态条目的哈希值。

        Returns:
            若该条目已缓存则返回 `True`，否则返回 `False`。
        """
        raise NotImplementedError

    def is_cached(self, mm_hashes: list[str]) -> list[bool]:
        """
        检查一组多模态条目是否已经存在于底层缓存中。

        这个操作**不会**更新缓存的淘汰顺序。

        Args:
            mm_hashes: 每个待检查条目的哈希值。

        Returns:
            对每个条目分别返回是否已缓存的布尔值。
        """
        return [self.is_cached_item(mm_hash) for mm_hash in mm_hashes]

    def close(self) -> None:
        """如有需要，关闭底层缓存资源。"""
        pass

    @abstractmethod
    def touch_sender_cache_item(self, mm_hash: str) -> None:
        """
        更新某个多模态条目的缓存淘汰顺序。

        这个操作用于在不修改缓存值的前提下，刷新该条目的最近使用时间。

        Args:
            mm_hash: 多模态条目的哈希值。
        """
        raise NotImplementedError

    @abstractmethod
    def make_stats(self, *, delta: bool = False) -> CacheInfo:
        """
        获取多模态缓存统计信息（必要时返回增量统计并重置基线）。

        Returns:
            当前的多模态缓存统计结果。
        """
        raise NotImplementedError


class MultiModalProcessorOnlyCache(BaseMultiModalProcessorCache):
    """
    The cache which is used on P0 when IPC caching is disabled.

    How to update each item:

    - If the item is in the cache, replace the input with the cached item.
    - If the item is not in the cache, store that item (which includes
      tensor data and metadata) into the cache, and return the input.
    """

    def __init__(self, model_config: "ModelConfig") -> None:
        super().__init__()

        mm_config = model_config.get_multimodal_config()

        self._cache = MultiModalCache.get_lru_cache(
            mm_config.mm_processor_cache_gb,
            MultiModalProcessorCacheItem,
        )

    @override
    def is_cached_item(self, mm_hash: str) -> bool:
        return mm_hash in self._cache

    @override
    def get_and_update_item(
        self,
        mm_item: MultiModalProcessorCacheInItem,
        mm_hash: str,
    ) -> MultiModalProcessorCacheOutItem:
        if (cached_item := self._cache.get(mm_hash)) is not None:
            return cached_item.item, cached_item.prompt_updates

        assert mm_item is not None, f"Expected a cached item for {mm_hash=}"

        self._cache[mm_hash] = MultiModalProcessorCacheItem(*mm_item)

        return mm_item

    @override
    def touch_sender_cache_item(self, mm_hash: str) -> None:
        self._cache.touch(mm_hash)

    @override
    def clear_cache(self) -> None:
        self._cache.clear()

    @override
    def make_stats(self, *, delta: bool = False) -> CacheInfo:
        return self._cache.stat(delta=delta)

# IPC 可用时，P0 侧使用的 sender 元数据缓存。
class MultiModalProcessorSenderCache(BaseMultiModalProcessorCache):
    """
    当启用 IPC 缓存时，P0 侧使用的缓存实现。

    每个条目的更新方式如下：

    - 如果条目已经在缓存中，就清空输入，避免不必要的 IPC 传输。

    - 如果条目还不在缓存中，就只保存该条目的元数据，
      这样可以让它与 P1 侧缓存保持相同的淘汰策略，然后返回原始输入。
      由于这里只保存元数据，因此可以避免在 P0 内存里额外保留实际数据本体。
    """

    def __init__(self, model_config: "ModelConfig") -> None:
        super().__init__()

        mm_config = model_config.get_multimodal_config()

        # sender 侧只缓存元数据，不缓存真正的多模态张量/kwargs 数据。
        self._cache = MultiModalCache.get_lru_cache(
            # 这个 LRU cache 的总容量上限，单位是 GiB。
            mm_config.mm_processor_cache_gb,
            # 这个 cache 里每个 value 的逻辑类型：
            # sender 侧存的是 `MultiModalProcessorCacheItemMetadata`，
            # 即“多模态结果的元数据 + prompt_updates”，而不是实际张量数据。
            MultiModalProcessorCacheItemMetadata,
        )

    @override
    def is_cached_item(self, mm_hash: str) -> bool:
        return mm_hash in self._cache

    @override
    def get_and_update_item(
        self,
        mm_item: MultiModalProcessorCacheInItem,
        mm_hash: str,
    ) -> MultiModalProcessorCacheOutItem:
        if (cached_item := self._cache.get(mm_hash)) is not None:
            # 命中时返回空输入，只把 prompt_updates 交给下游，避免重复 IPC。
            return None, cached_item.prompt_updates

        assert mm_item is not None, f"Expected a cached item for {mm_hash=}"

        # 未命中时仅登记元数据，驱逐顺序与对端 receiver cache 保持一致。
        self._cache[mm_hash] = MultiModalProcessorCacheItemMetadata(*mm_item)

        return mm_item

    @override
    def touch_sender_cache_item(self, mm_hash: str) -> None:
        self._cache.touch(mm_hash)

    @override
    def clear_cache(self) -> None:
        self._cache.clear()

    @override
    def make_stats(self, *, delta: bool = False) -> CacheInfo:
        return self._cache.stat(delta=delta)


class ShmObjectStoreSenderCache(BaseMultiModalProcessorCache):
    """
    The cache which is used on P0 when IPC caching is enabled.

    How to update each item:

    - If the item is already in the cache, clear the input to avoid
      unnecessary IPC.

    - If the item is not in the cache, store the data in shared memory.
    """

    def __init__(self, cfie_config: "CfieConfig") -> None:
        super().__init__()

        self.world_size = cfie_config.parallel_config.world_size
        mm_config = cfie_config.model_config.get_multimodal_config()

        ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=int(mm_config.mm_processor_cache_gb * GiB_bytes),
            name=envs.VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME,
            create=True,  # sender is the writer
        )
        self._shm_cache = SingleWriterShmObjectStorage(
            max_object_size=mm_config.mm_shm_cache_max_object_size_mb * MiB_bytes,
            n_readers=self.world_size,
            ring_buffer=ring_buffer,
            serde_class=MsgpackSerde,
        )
        # cache prompt_updates for P0 only
        self._p0_cache: dict[str, Sequence[ResolvedPromptUpdate]] = {}

        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)

    def _stat(self, *, delta: bool = False) -> CacheInfo:
        info = CacheInfo(hits=self._hits, total=self._total)

        if delta:
            info_delta = info - self._last_info
            self._last_info = info
            info = info_delta

        return info

    @override
    def is_cached_item(self, mm_hash: str) -> bool:
        return self._shm_cache.is_cached(mm_hash)

    @override
    def get_and_update_item(
        self,
        mm_item: MultiModalProcessorCacheInItem,
        mm_hash: str,
    ) -> MultiModalProcessorCacheOutItem:
        if self._shm_cache.is_cached(mm_hash):
            self._hits += 1
            self._total += 1

            address, monotonic_id = self._shm_cache.get_cached(mm_hash)
            prompt_updates = self._p0_cache[mm_hash]
            return self.address_as_item(address, monotonic_id), prompt_updates

        assert mm_item is not None, f"Expected a cached item for {mm_hash=}"
        item, prompt_updates = mm_item

        self._total += 1

        try:
            address, monotonic_id = self._shm_cache.put(mm_hash, item)
            # Try to remove dangling items if p0 cache is too large.
            if len(self._p0_cache) >= 2 * len(self._shm_cache.key_index):
                self.remove_dangling_items()

            self._p0_cache[mm_hash] = prompt_updates
            return self.address_as_item(address, monotonic_id), prompt_updates
        except (ValueError, MemoryError) as e:
            # put may fail if the object is too large or
            # the cache is full.
            # In this case we log the error and keep the original mm_input.
            logger.debug("Failed to cache mm_input with hash %s: %s", mm_hash, e)
            return mm_item

    @override
    def touch_sender_cache_item(self, mm_hash: str) -> None:
        """Touch the item in shared memory cache to prevent eviction.
        Increments writer_flag on sender side."""
        self._shm_cache.touch(mm_hash)

    @override
    def clear_cache(self) -> None:
        self._shm_cache.clear()
        self._p0_cache.clear()

        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)

    @override
    def make_stats(self, *, delta: bool = False) -> CacheInfo:
        return self._stat(delta=delta)

    @override
    def close(self) -> None:
        self._shm_cache.close()

    def remove_dangling_items(self) -> None:
        """Remove items that are no longer in the shared memory cache."""
        cached_hashes = self._shm_cache.key_index.keys()
        dangling_hashes = set(self._p0_cache.keys()) - cached_hashes
        for mm_hash in dangling_hashes:
            del self._p0_cache[mm_hash]

    def address_as_item(
        self,
        address: int,
        monotonic_id: int,
    ) -> MultiModalKwargsItem:
        addr_elem = MultiModalFieldElem(
            data=address,
            field=MultiModalBatchedField(),
        )
        id_elem = MultiModalFieldElem(
            data=monotonic_id,
            field=MultiModalBatchedField(),
        )

        return MultiModalKwargsItem({"address": addr_elem, "monotonic_id": id_elem})


class BaseMultiModalReceiverCache(
    BaseMultiModalCache[MultiModalKwargsItem | None, MultiModalKwargsItem]
):
    """The required interface for caches on P1."""

    def get_and_update_features(
        self,
        mm_features: list["MultiModalFeatureSpec"],
    ) -> list["MultiModalFeatureSpec"]:
        """
        Update multimodal features with cached encoder outputs.
        Touch all identifier at first before update to avoid
        item in updated list evict during update.

        Uses mm_hash for cache key to share across LoRAs (falls back to
        identifier for backward compatibility).
        """
        for feature in mm_features:
            cache_key = feature.mm_hash or feature.identifier
            self.touch_receiver_cache_item(cache_key, feature.data)

        for feature in mm_features:
            cache_key = feature.mm_hash or feature.identifier
            feature.data = self.get_and_update_item(feature.data, cache_key)
        return mm_features

    @abstractmethod
    def touch_receiver_cache_item(
        self,
        mm_hash: str,
        mm_item: MultiModalKwargsItem | None = None,
    ) -> None:
        """
        Update the cache eviction order for a multi-modal item.

        This is used to touch the item in the cache without changing
        its value.

        Args:
            mm_hash: The hash of the multi-modal item.
            mm_item: The multi-modal item itself. This is optional and
                may not be needed by some cache implementations.
        """
        raise NotImplementedError


class MultiModalReceiverCache(BaseMultiModalReceiverCache):
    """
    The cache which is used on P1 when IPC caching is enabled.

    How to update each item:

    - If the item is in the cache, replace the input with the cached item.
    - If the item is not in the cache, store that item (which includes tensor
      data) into the cache, and return the input.
    """

    def __init__(self, model_config: "ModelConfig") -> None:
        super().__init__()

        mm_config = model_config.get_multimodal_config()

        self._cache = MultiModalCache.get_lru_cache(
            mm_config.mm_processor_cache_gb,
            MultiModalKwargsItem,
        )

    @override
    def get_and_update_item(
        self,
        mm_item: MultiModalKwargsItem | None,
        mm_hash: str,
    ) -> MultiModalKwargsItem:
        if (cached_item := self._cache.get(mm_hash)) is not None:
            return cached_item

        assert mm_item is not None, f"Expected a cached item for {mm_hash=}"

        self._cache[mm_hash] = mm_item
        return mm_item

    @override
    def touch_receiver_cache_item(
        self,
        mm_hash: str,
        mm_item: MultiModalKwargsItem | None = None,
    ) -> None:
        self._cache.touch(mm_hash)

    @override
    def clear_cache(self) -> None:
        self._cache.clear()


class ShmObjectStoreReceiverCache(BaseMultiModalReceiverCache):
    """
    The cache which is used on P1 Worker Process when IPC caching is enabled.

    How to update each item:

    - If the item has an address, replace the input with the cached item.
    - If not, return the input.
    """

    def __init__(
        self,
        cfie_config: "CfieConfig",
        shared_worker_lock: LockType,
    ) -> None:
        super().__init__()

        self.world_size = cfie_config.parallel_config.world_size
        mm_config = cfie_config.model_config.get_multimodal_config()

        ring_buffer = SingleWriterShmRingBuffer(
            data_buffer_size=int(mm_config.mm_processor_cache_gb * GiB_bytes),
            name=envs.VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME,
            create=False,  # Server is a reader
        )
        self._shm_cache = SingleWriterShmObjectStorage(
            max_object_size=mm_config.mm_shm_cache_max_object_size_mb * MiB_bytes,
            n_readers=self.world_size,
            ring_buffer=ring_buffer,
            serde_class=MsgpackSerde,
            reader_lock=shared_worker_lock,
        )

    @override
    def get_and_update_item(
        self,
        mm_item: MultiModalKwargsItem | None,
        mm_hash: str,
    ) -> MultiModalKwargsItem:
        assert mm_item is not None, f"Expected an address item for {mm_hash=}"
        if "address" in mm_item:
            address = cast(int, mm_item["address"].data)
            monotonic_id = cast(int, mm_item["monotonic_id"].data)
            return self._shm_cache.get(address, monotonic_id)

        return mm_item

    @override
    def touch_receiver_cache_item(
        self,
        mm_hash: str,
        mm_item: MultiModalKwargsItem | None = None,
    ) -> None:
        """Touch the item in shared memory cache to prevent eviction.
        Increments reader_count on receiver side."""
        assert mm_item is not None
        if "address" in mm_item:
            address = cast(int, mm_item["address"].data)
            monotonic_id = cast(int, mm_item["monotonic_id"].data)
            self._shm_cache.touch(mm_hash, address=address, monotonic_id=monotonic_id)

    @override
    def clear_cache(self) -> None:
        self._shm_cache.clear()
