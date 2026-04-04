# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import uuid
from dataclasses import field
from typing import Any, Literal, get_args

from cfie.config.utils import config

# EC 发送端角色枚举。
ECProducer = Literal["ec_producer", "ec_both"]
# EC 接收端角色枚举。
ECConsumer = Literal["ec_consumer", "ec_both"]
# EC 角色总枚举。
ECRole = Literal[ECProducer, ECConsumer]


@config
class ECTransferConfig:
    """Configuration for distributed EC cache transfer."""

    # EC connector 名称；为空表示未启用跨实例 EC 传输。
    ec_connector: str | None = None
    """The EC connector for vLLM to transmit EC caches between vLLM instances.
    """

    # 当前引擎在 EC 传输平面中的唯一 ID。
    engine_id: str | None = None
    """The engine id for EC transfers."""

    # EC cache buffer 使用的设备；当前仅支持 cuda。
    ec_buffer_device: str | None = "cuda"
    """The device used by ec connector to buffer the EC cache.
    Currently only support 'cuda'."""

    # EC buffer 的容量预算，单位字节。
    ec_buffer_size: float = 1e9
    """The buffer size for TorchDistributedConnector. Measured in number of
    bytes. Recommended value: 1e9 (about 1GB)."""

    # 当前实例在 EC 拓扑中的角色。
    ec_role: ECRole | None = None
    """Whether this vLLM instance produces, consumes EC cache, or both. Choices
    are 'ec_producer', 'ec_consumer', 'ec_both'."""

    # 当前实例在 EC 传输组中的 rank。
    ec_rank: int | None = None
    """The rank of this vLLM instance in the EC cache transfer. Typical value:
    0 for encoder, 1 for pd instance.
    Currently only 1P1D is supported."""

    # EC 传输并行度。
    ec_parallel_size: int = 1
    """The number of parallel instances for EC cache transfer. For
    PyNcclConnector, this should be 2."""

    # 建立 EC 传输连接时使用的 IP。
    ec_ip: str = "127.0.0.1"
    """The EC connector ip, used to build distributed connection."""

    # 建立 EC 传输连接时使用的端口。
    ec_port: int = 14579
    """The EC connector port, used to build distributed connection."""

    # connector 私有附加配置字典。
    ec_connector_extra_config: dict[str, Any] = field(default_factory=dict)
    """any extra config that the connector may need."""

    # 动态加载 connector 时使用的模块路径。
    ec_connector_module_path: str | None = None
    """The Python module path to dynamically load the EC connector from.
    Only supported in V1."""

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
        # EC transfer 配置同样不参与模型图哈希。
        factors: list[Any] = []
        # 基于空 factors 生成稳定哈希。
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        # 返回最终哈希字符串。
        return hash_str

    def __post_init__(self) -> None:
        # 若未提供 engine_id，则自动生成 UUID。
        if self.engine_id is None:
            self.engine_id = str(uuid.uuid4())

        # ec_role 若给出，则必须属于合法角色枚举。
        if self.ec_role is not None and self.ec_role not in get_args(ECRole):
            raise ValueError(
                f"Unsupported ec_role: {self.ec_role}. "
                f"Supported roles are {get_args(ECRole)}"
            )

        # 启用了 EC connector 时必须显式给出实例角色。
        if self.ec_connector is not None and self.ec_role is None:
            raise ValueError(
                "Please specify ec_role when ec_connector "
                f"is set, supported roles are {get_args(ECRole)}"
            )

    @property
    def is_ec_transfer_instance(self) -> bool:
        # connector 存在且角色合法时，说明当前实例参与 EC transfer。
        return self.ec_connector is not None and self.ec_role in get_args(ECRole)

    @property
    def is_ec_producer(self) -> bool:
        # 生产者包含 ec_producer 与 ec_both。
        return self.ec_connector is not None and self.ec_role in get_args(ECProducer)

    @property
    def is_ec_consumer(self) -> bool:
        # 消费者包含 ec_consumer 与 ec_both。
        return self.ec_connector is not None and self.ec_role in get_args(ECConsumer)

    def get_from_extra_config(self, key, default) -> Any:
        # 从附加配置中取值，不存在时返回默认值。
        return self.ec_connector_extra_config.get(key, default)
