# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import Literal

from pydantic import Field

from cfie.config.utils import config


@config
class KVEventsConfig:
    """Configuration for KV event publishing."""

    # 是否启用 KV cache 事件发布功能。
    enable_kv_cache_events: bool = False
    """If True, enable KV cache events for tracking block storage and removal.
    Events can be published externally by zmq using the event publisher config.
    """

    # 事件发布后端；为空时会在 __post_init__ 里按开关自动补默认值。
    publisher: Literal["null", "zmq"] = Field(default=None)
    """The publisher to use for publishing kv events. Can be "null", "zmq".
    """

    # ZMQ 发布端点。
    endpoint: str = "tcp://*:5557"
    """The zmq endpoint to use for publishing kv events.
    """

    # 若配置了 replay，则该端点可回放最近若干步的缓存事件。
    replay_endpoint: str | None = None
    """The zmq endpoint to use for replaying kv events.
    """

    # replay 模式最多保留最近多少步事件。
    buffer_steps: int = 10_000
    """The number of steps to cache for replay endpoint. Will only save
    events from the last N steps for the replay endpoint.
    """

    # ZMQ 高水位，超过后消费者跟不上时会开始丢事件。
    hwm: int = 100_000
    """The zmq high water mark for the event publisher. After queueing N events,
    events will start dropping if the consumer is not keeping up.
    """

    # 内部待发布队列的最大长度。
    max_queue_size: int = 100_000
    """The maximum number of events to queue while waiting for publishing.
    """

    # 可选 topic 前缀，供消费者按主题订阅。
    topic: str = ""
    """The topic to use for the event publisher. Consumers can subscribe to
    this topic to receive events.
    """

    def __post_init__(self):
        # 若用户未显式指定 publisher，则按总开关自动推断默认值。
        if self.publisher is None:
            self.publisher = "zmq" if self.enable_kv_cache_events else "null"
