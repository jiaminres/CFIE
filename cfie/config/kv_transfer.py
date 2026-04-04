# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import uuid
from dataclasses import field
from typing import Any, Literal, get_args

from cfie.config.utils import config
from cfie.utils.hashing import safe_hash

# KV 发送端角色枚举。
KVProducer = Literal["kv_producer", "kv_both"]
# KV 接收端角色枚举。
KVConsumer = Literal["kv_consumer", "kv_both"]
# KV 角色总枚举，允许生产/消费/双向。
KVRole = Literal[KVProducer, KVConsumer]


def kv_buffer_device_default_factory() -> str:
    # 延迟导入当前平台对象，按平台默认设备类型初始化 buffer device。
    from cfie.platforms import current_platform

    # 直接复用当前平台的 device_type 作为 KV buffer 默认设备。
    return current_platform.device_type


@config
class KVTransferConfig:
    """Configuration for distributed KV cache transfer."""

    # KV connector 名称；为空表示未启用跨实例 KV 传输。
    kv_connector: str | None = None
    """The KV connector for vLLM to transmit KV caches between vLLM instances.
    """

    # 当前引擎在 KV 传输平面中的唯一 ID。
    engine_id: str | None = None
    """The engine id for KV transfers."""

    # KV connector 用来缓存 KV block 的设备类型。
    kv_buffer_device: str = field(default_factory=kv_buffer_device_default_factory)
    """The device used by kv connector to buffer the KV cache. Choices are
    'cuda', 'cpu' and 'xpu'."""

    # KV buffer 的容量预算，单位字节。
    kv_buffer_size: float = 1e9
    """The buffer size for TorchDistributedConnector. Measured in number of
    bytes. Recommended value: 1e9 (about 1GB)."""

    # 当前实例在 KV 拓扑中的角色。
    kv_role: KVRole | None = None
    """Whether this vLLM instance produces, consumes KV cache, or both. Choices
    are 'kv_producer', 'kv_consumer', and 'kv_both'."""

    # 当前实例在 KV 传输组中的 rank。
    kv_rank: int | None = None
    """The rank of this vLLM instance in the KV cache transfer. Typical value:
    0 for prefill instance, 1 for decode instance.
    Currently only 1P1D is supported."""

    # KV 传输并行度；某些 connector 会要求固定为 2。
    kv_parallel_size: int = 1
    """The number of parallel instances for KV cache transfer. For
    P2pNcclConnector, this should be 2."""

    # 建立 KV 传输连接时使用的目标 IP。
    kv_ip: str = "127.0.0.1"
    """The KV connector ip, used to build distributed connection."""

    # 建立 KV 传输连接时使用的端口。
    kv_port: int = 14579
    """The KV connector port, used to build distributed connection."""

    # connector 的附加配置字典。
    kv_connector_extra_config: dict[str, Any] = field(default_factory=dict)
    """any extra config that the connector may need."""

    # 动态加载 connector 时使用的 Python 模块路径。
    kv_connector_module_path: str | None = None
    """The Python module path to dynamically load the KV connector from.
    Only supported in V1."""

    # 是否启用本地 KV 布局置换实验开关。
    enable_permute_local_kv: bool = False
    """Experiment feature flag to enable HND to NHD KV Transfer"""

    # KV 加载失败后的处理策略。
    kv_load_failure_policy: Literal["recompute", "fail"] = "fail"
    """Policy for handling KV cache load failures.
    'recompute': reschedule the request to recompute failed blocks
    'fail': immediately fail the request with an error finish reason (default)"""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        # KV transfer 配置不改变模型图结构，因此 factors 为空。
        factors: list[Any] = []
        # 用空 factors 生成稳定哈希。
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        # 返回最终哈希。
        return hash_str

    def __post_init__(self) -> None:
        # 若调用方未提供 engine_id，则自动生成 UUID。
        if self.engine_id is None:
            self.engine_id = str(uuid.uuid4())

        # kv_role 若给出，则必须属于合法角色集合。
        if self.kv_role is not None and self.kv_role not in get_args(KVRole):
            raise ValueError(
                f"Unsupported kv_role: {self.kv_role}. "
                f"Supported roles are {get_args(KVRole)}"
            )

        # 只要启用了 connector，就必须同时声明实例角色。
        if self.kv_connector is not None and self.kv_role is None:
            raise ValueError(
                "Please specify kv_role when kv_connector "
                f"is set, supported roles are {get_args(KVRole)}"
            )

    @property
    def is_kv_transfer_instance(self) -> bool:
        # 只要 connector 存在且角色合法，就视为 KV transfer 实例。
        return self.kv_connector is not None and self.kv_role in get_args(KVRole)

    @property
    def is_kv_producer(self) -> bool:
        # 生产者包含 kv_producer 与 kv_both 两类角色。
        return self.kv_connector is not None and self.kv_role in get_args(KVProducer)

    @property
    def is_kv_consumer(self) -> bool:
        # 消费者包含 kv_consumer 与 kv_both 两类角色。
        return self.kv_connector is not None and self.kv_role in get_args(KVConsumer)

    def get_from_extra_config(self, key, default) -> Any:
        # 从 connector 附加配置字典里按 key 取值，不存在时返回默认值。
        return self.kv_connector_extra_config.get(key, default)
