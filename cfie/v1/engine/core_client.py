# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
import multiprocessing
import queue
import sys
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Thread
from typing import Any, TypeAlias, TypeVar

import msgspec.msgpack
import zmq
import zmq.asyncio

from cfie.config import CfieConfig
from cfie.envs import VLLM_ENGINE_READY_TIMEOUT_S
from cfie.logger import init_logger
from cfie.lora.request import LoRARequest
from cfie.tasks import SupportedTask
from cfie.tracing import instrument
from cfie.utils.async_utils import in_loop
from cfie.utils.network_utils import (
    close_sockets,
    get_open_zmq_inproc_path,
    make_zmq_socket,
)
from cfie.v1.engine import (
    EEP_NOTIFICATION_CALL_ID,
    EEPNotificationType,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    PauseMode,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
)
from cfie.v1.engine.coordinator import DPCoordinator
from cfie.v1.engine.core import EngineCore, EngineCoreProc
from cfie.v1.engine.exceptions import EngineDeadError
from cfie.v1.engine.utils import (
    CoreEngineActorManager,
    CoreEngineProcManager,
    get_engine_zmq_addresses,
    launch_core_engines,
)
from cfie.v1.executor import Executor
from cfie.v1.pool.late_interaction import get_late_interaction_engine_index
from cfie.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr

logger = init_logger(__name__)

AnyFuture: TypeAlias = asyncio.Future[Any] | Future[Any]

_R = TypeVar("_R")  # Return type for collective_rpc

EngineIdentity = bytes


# EngineCore 通信客户端抽象基类。
class EngineCoreClient(ABC):
    """
    EngineCoreClient：用于抽象与 EngineCore 之间的通信客户端。
    其子类分别负责在 asyncio / multiprocessing 等不同模式下，
    处理向 EngineCore 推送请求与拉取结果的具体方式。

    子类包括：
    * InprocClient：进程内 EngineCore（用于 V0 风格的 LLMEngine）
    * SyncMPClient：基于 ZMQ + 后台进程的 EngineCore（用于 LLM）
    * AsyncMPClient：基于 ZMQ + asyncio + 后台进程的 EngineCore（用于 AsyncLLM）
    """

    @staticmethod
    # 按 multiprocessing/async 模式选择具体的 engine core client 实现。
    def make_client(
            multiprocess_mode: bool,
            asyncio_mode: bool,
            cfie_config: CfieConfig,
            executor_class: type[Executor],
            log_stats: bool,
    ) -> "EngineCoreClient":
        # TODO: support this for debugging purposes.
        # asyncio 但非多进程的组合当前未实现。
        if asyncio_mode and not multiprocess_mode:
            raise NotImplementedError(
                "Running EngineCore in asyncio without multiprocessing "
                "is not currently supported."
            )

        # asyncio + 多进程时创建异步 MP client。
        if multiprocess_mode and asyncio_mode:
            return EngineCoreClient.make_async_mp_client(
                cfie_config, executor_class, log_stats
            )

        # 同步 + 多进程时创建同步 MP client。
        if multiprocess_mode and not asyncio_mode:  # 默认
            return SyncMPClient(cfie_config, executor_class, log_stats)

        # 默认直接在当前进程内创建 Inproc client。
        return InprocClient(cfie_config, executor_class, log_stats)

    @staticmethod
    @instrument(span_name="Overall Loading")
    # 创建异步多进程 client，并在 DP 场景下选择对应的负载均衡实现。
    def make_async_mp_client(
            cfie_config: CfieConfig,
            executor_class: type[Executor],
            log_stats: bool,
            client_addresses: dict[str, str] | None = None,
            client_count: int = 1,
            client_index: int = 0,
    ) -> "AsyncMPClient":
        # 取出并行配置，决定 DP 下的 client 形态。
        parallel_config = cfie_config.parallel_config
        # 先整理异步 MP client 共同构造参数。
        client_args = (
            cfie_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )
        if parallel_config.data_parallel_size > 1:
            if parallel_config.data_parallel_external_lb:
                # External load balancer - client per DP rank.
                # 外部负载均衡时为每个 DP rank 建一个 client。
                return DPAsyncMPClient(*client_args)
            # Internal load balancer - client balances to all DP ranks.
            # 内部负载均衡时创建能在 DP ranks 间分发的 client。
            return DPLBAsyncMPClient(*client_args)
        # 非 DP 场景直接创建普通异步 MP client。
        return AsyncMPClient(*client_args)

    @abstractmethod
    # 关闭 client 并释放底层通信与执行资源。
    def shutdown(self, timeout: float | None = None) -> None:
        # 由具体子类实现实际的关闭流程。
        ...

    # 同步拉取一次 EngineCore 输出。
    def get_output(self) -> EngineCoreOutputs:
        # 基类不提供默认实现，要求子类自行定义同步取输出逻辑。
        raise NotImplementedError

    # 查询底层 engine 支持的任务类型。
    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        # 基类不提供默认实现，要求子类自行查询任务能力。
        raise NotImplementedError

    # 向 engine 提交一条新请求。
    def add_request(self, request: EngineCoreRequest) -> None:
        # 基类不提供默认实现，要求子类自行发送请求。
        raise NotImplementedError

    # 开始或结束 profiling，并可指定输出前缀。
    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        # 基类不提供默认实现，要求子类自行转发 profiling 控制。
        raise NotImplementedError

    # 重置多模态处理相关缓存。
    def reset_mm_cache(self) -> None:
        # 基类不提供默认实现，要求子类自行实现多模态缓存重置。
        raise NotImplementedError

    # 重置 prefix cache，并可选清理运行中请求或 connector 状态。
    def reset_prefix_cache(
            self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        # 基类不提供默认实现，要求子类自行实现 prefix cache 重置。
        raise NotImplementedError

    # 重置 encoder cache。
    def reset_encoder_cache(self) -> None:
        # 基类不提供默认实现，要求子类自行实现 encoder cache 重置。
        raise NotImplementedError

    # 让 engine 进入休眠/暂停状态。
    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        # 基类不提供默认实现，要求子类自行实现休眠控制。
        raise NotImplementedError

    # 唤醒已休眠的 engine。
    def wake_up(self, tags: list[str] | None = None) -> None:
        # 基类不提供默认实现，要求子类自行实现唤醒控制。
        raise NotImplementedError

    # 查询 engine 当前是否处于休眠状态。
    def is_sleeping(self) -> bool:
        # 基类不提供默认实现，要求子类自行查询休眠状态。
        raise NotImplementedError

    # 执行一次 dummy batch。
    def execute_dummy_batch(self) -> None:
        # 基类不提供默认实现，要求子类自行触发 dummy batch。
        raise NotImplementedError

    # 异步执行一次 dummy batch。
    async def execute_dummy_batch_async(self) -> None:
        # 基类不提供默认实现，要求子类自行实现异步 dummy batch。
        raise NotImplementedError

    # 中止一组指定 request_id 对应的请求。
    def abort_requests(self, request_ids: list[str]) -> None:
        # 基类不提供默认实现，要求子类自行实现请求中止。
        raise NotImplementedError

    # 动态加载一个 LoRA。
    def add_lora(self, lora_request: LoRARequest) -> bool:
        # 基类不提供默认实现，要求子类自行实现 LoRA 加载。
        raise NotImplementedError

    # 卸载指定 LoRA。
    def remove_lora(self, lora_id: int) -> bool:
        # 基类不提供默认实现，要求子类自行实现 LoRA 卸载。
        raise NotImplementedError

    # 列出当前已加载的 LoRA 集合。
    def list_loras(self) -> set[int]:
        # 基类不提供默认实现，要求子类自行返回已加载 LoRA 集合。
        raise NotImplementedError

    # 固定指定 LoRA，避免其被回收。
    def pin_lora(self, lora_id: int) -> bool:
        # 基类不提供默认实现，要求子类自行实现 LoRA 固定。
        raise NotImplementedError

    # 保存当前分片状态到磁盘。
    def save_sharded_state(
            self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        # 基类不提供默认实现，要求子类自行实现状态保存。
        raise NotImplementedError

    # 对底层 engine 集体执行一个 RPC 方法。
    def collective_rpc(
            self,
            method: str | Callable[..., _R],
            timeout: float | None = None,
            args: tuple = (),
            kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        # 基类不提供默认实现，要求子类自行实现 collective RPC。
        raise NotImplementedError

    # 查询数据并行各 engine 是否整体处于运行状态。
    def dp_engines_running(self) -> bool:
        """Returns True if data parallel engines are collectively in a
        running state."""
        # 基类不提供默认实现，要求子类自行维护并返回该状态。
        raise NotImplementedError

    # 异步调整 elastic EP 的数据并行规模。
    async def scale_elastic_ep(self, new_data_parallel_size: int) -> None:
        # 基类不提供默认实现，要求子类自行实现 elastic EP 扩缩容。
        raise NotImplementedError

    # 异步拉取一次 EngineCore 输出。
    async def get_output_async(self) -> EngineCoreOutputs:
        # 基类不提供默认实现，要求子类自行定义异步取输出逻辑。
        raise NotImplementedError

    # 异步查询底层 engine 支持的任务类型。
    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
        # 基类不提供默认实现，要求子类自行查询异步任务能力。
        raise NotImplementedError

    # 异步向 engine 提交一条新请求。
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # 基类不提供默认实现，要求子类自行异步发送请求。
        raise NotImplementedError

    # 异步开始或结束 profiling，并可指定输出前缀。
    async def profile_async(
            self, is_start: bool = True, profile_prefix: str | None = None
    ) -> None:
        # 基类不提供默认实现，要求子类自行异步转发 profiling 控制。
        raise NotImplementedError

    # 异步重置多模态处理相关缓存。
    async def reset_mm_cache_async(self) -> None:
        # 基类不提供默认实现，要求子类自行异步重置多模态缓存。
        raise NotImplementedError

    # 异步重置 prefix cache，并可选清理运行中请求或 connector 状态。
    async def reset_prefix_cache_async(
            self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        # 基类不提供默认实现，要求子类自行异步重置 prefix cache。
        raise NotImplementedError

    # 异步重置 encoder cache。
    async def reset_encoder_cache_async(self) -> None:
        # 基类不提供默认实现，要求子类自行异步重置 encoder cache。
        raise NotImplementedError

    # 异步让 engine 进入休眠/暂停状态。
    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:
        # 基类不提供默认实现，要求子类自行异步实现休眠控制。
        raise NotImplementedError

    # 异步唤醒已休眠的 engine。
    async def wake_up_async(self, tags: list[str] | None = None) -> None:
        # 基类不提供默认实现，要求子类自行异步实现唤醒控制。
        raise NotImplementedError

    # 异步查询 engine 当前是否处于休眠状态。
    async def is_sleeping_async(self) -> bool:
        # 基类不提供默认实现，要求子类自行异步查询休眠状态。
        raise NotImplementedError

    # 异步中止一组指定 request_id 对应的请求。
    async def abort_requests_async(self, request_ids: list[str]) -> None:
        # 基类不提供默认实现，要求子类自行异步实现请求中止。
        raise NotImplementedError

    # 异步动态加载一个 LoRA。
    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        # 基类不提供默认实现，要求子类自行异步实现 LoRA 加载。
        raise NotImplementedError

    # 异步卸载指定 LoRA。
    async def remove_lora_async(self, lora_id: int) -> bool:
        # 基类不提供默认实现，要求子类自行异步实现 LoRA 卸载。
        raise NotImplementedError

    # 异步列出当前已加载的 LoRA 集合。
    async def list_loras_async(self) -> set[int]:
        # 基类不提供默认实现，要求子类自行异步返回已加载 LoRA 集合。
        raise NotImplementedError

    # 异步固定指定 LoRA，避免其被回收。
    async def pin_lora_async(self, lora_id: int) -> bool:
        # 基类不提供默认实现，要求子类自行异步实现 LoRA 固定。
        raise NotImplementedError

    # 异步保存当前分片状态到磁盘。
    async def save_sharded_state_async(
            self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        # 基类不提供默认实现，要求子类自行异步实现状态保存。
        raise NotImplementedError

    # 异步对底层 engine 集体执行一个 RPC 方法。
    async def collective_rpc_async(
            self,
            method: str | Callable[..., _R],
            timeout: float | None = None,
            args: tuple = (),
            kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        # 基类不提供默认实现，要求子类自行异步实现 collective RPC。
        raise NotImplementedError


# 进程内 EngineCore 的同步直连 client。
class InprocClient(EngineCoreClient):
    """
    InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """

    # 直接在当前进程内创建 `EngineCore`，用于非多进程模式。
    def __init__(self, *args, **kwargs):
        # 当前进程内直接构造 EngineCore 实例。
        self.engine_core = EngineCore(*args, **kwargs)

    # 同步执行一次 engine core step 并返回输出。
    def get_output(self) -> EngineCoreOutputs:
        # 调用 engine core 的 step_fn 执行一轮调度与推理。
        outputs, model_executed = self.engine_core.step_fn()
        # 处理该轮 step 之后的收尾逻辑。
        self.engine_core.post_step(model_executed=model_executed)
        # 取 client_index=0 的输出；若为空则返回空结果对象。
        return outputs and outputs.get(0) or EngineCoreOutputs()

    # 直接读取本地 engine core 暴露的任务能力。
    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        # 直接转发到底层 EngineCore。
        return self.engine_core.get_supported_tasks()

    # 把请求直接送进本地 engine core。
    def add_request(self, request: EngineCoreRequest) -> None:
        # 先把前端请求预处理成调度器内部 Request。
        req, request_wave = self.engine_core.preprocess_add_request(request)
        # 再把请求送入本地 EngineCore。
        self.engine_core.add_request(req, request_wave)

    # 在本地 engine core 上中止请求。
    def abort_requests(self, request_ids: list[str]) -> None:
        # 仅在存在 request_id 时才触发中止，避免空调用。
        if len(request_ids) > 0:
            # 把中止请求直接下发给本地 EngineCore。
            self.engine_core.abort_requests(request_ids)

    # 关闭本地 engine core。
    def shutdown(self, timeout: float | None = None) -> None:
        # 直接关闭本地 EngineCore 持有的资源。
        self.engine_core.shutdown()

    # 转发 profile 控制到底层本地 EngineCore。
    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        # 直接转发 profiling 开关与前缀。
        self.engine_core.profile(is_start, profile_prefix)

    # 重置本地 EngineCore 的多模态缓存。
    def reset_mm_cache(self) -> None:
        # 直接清空本地多模态缓存。
        self.engine_core.reset_mm_cache()

    # 重置本地 EngineCore 的 prefix cache。
    def reset_prefix_cache(
            self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        # 直接返回本地 prefix cache 重置结果。
        return self.engine_core.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    # 重置本地 EngineCore 的 encoder cache。
    def reset_encoder_cache(self) -> None:
        # 直接清空本地 encoder cache。
        self.engine_core.reset_encoder_cache()

    # 让本地 EngineCore 进入休眠/暂停状态。
    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        # 进程内模式不支持 wait 型暂停。
        if mode == "wait":
            raise ValueError("'wait' pause mode is not supported in inproc-engine mode")
        # 调用本地 EngineCore 执行休眠逻辑。
        result = self.engine_core.sleep(level, mode)
        # inproc 路径这里不应返回额外结果。
        assert result is None

    # 唤醒本地 EngineCore。
    def wake_up(self, tags: list[str] | None = None) -> None:
        # 直接唤醒本地 EngineCore。
        self.engine_core.wake_up(tags)

    # 查询本地 EngineCore 是否在休眠。
    def is_sleeping(self) -> bool:
        # 直接返回本地 EngineCore 的休眠状态。
        return self.engine_core.is_sleeping()

    # 在本地 EngineCore 上执行一次 dummy batch。
    def execute_dummy_batch(self) -> None:
        # 直接触发本地 dummy batch 执行。
        self.engine_core.execute_dummy_batch()

    # 在本地 EngineCore 上加载 LoRA。
    def add_lora(self, lora_request: LoRARequest) -> bool:
        # 直接返回本地 LoRA 加载结果。
        return self.engine_core.add_lora(lora_request)

    # 在本地 EngineCore 上卸载 LoRA。
    def remove_lora(self, lora_id: int) -> bool:
        # 直接返回本地 LoRA 卸载结果。
        return self.engine_core.remove_lora(lora_id)

    # 列出本地 EngineCore 中已加载的 LoRA。
    def list_loras(self) -> set[int]:
        # 直接返回本地 LoRA 集合。
        return self.engine_core.list_loras()

    # 固定本地 EngineCore 中的指定 LoRA。
    def pin_lora(self, lora_id: int) -> bool:
        # 直接返回本地 LoRA 固定结果。
        return self.engine_core.pin_lora(lora_id)

    # 保存本地 EngineCore 的分片状态。
    def save_sharded_state(
            self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        # 直接让本地 EngineCore 执行状态保存。
        self.engine_core.save_sharded_state(path, pattern, max_size)

    # 在本地 EngineCore 上执行 collective RPC。
    def collective_rpc(
            self,
            method: str | Callable[..., _R],
            timeout: float | None = None,
            args: tuple = (),
            kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        # 直接把 collective RPC 转发给本地 EngineCore。
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    # 进程内模式下不存在独立 DP engines 运行态。
    def dp_engines_running(self) -> bool:
        # 进程内模式固定返回 False。
        return False


@dataclass
class BackgroundResources:
    """作为终结器使用，用于在关闭时安全清理资源，
    避免反向持有 client 对象而形成循环引用。"""

    # ZMQ 上下文对象。
    ctx: zmq.Context

    # 若为 CoreEngineProcManager，则负责管理本地 engine；
    # 若为 CoreEngineActorManager，则负责管理全部 engine。
    engine_manager: CoreEngineProcManager | CoreEngineActorManager | None = None

    # 分布式并行协调器。
    coordinator: DPCoordinator | None = None

    # 输出 socket，可为同步或异步类型。
    output_socket: zmq.Socket | zmq.asyncio.Socket | None = None

    # 输入 socket，可为同步或异步类型。
    input_socket: zmq.Socket | zmq.asyncio.Socket | None = None

    # 首个请求发送 socket，仅异步场景使用。
    first_req_send_socket: zmq.asyncio.Socket | None = None

    # 首个请求接收 socket，仅异步场景使用。
    first_req_rcv_socket: zmq.asyncio.Socket | None = None

    # 统计信息更新 socket，仅异步场景使用。
    stats_update_socket: zmq.asyncio.Socket | None = None

    # 处理输出队列的后台任务。
    output_queue_task: asyncio.Task | None = None

    # 处理统计更新的后台任务。
    stats_update_task: asyncio.Task | None = None

    # 同步模式下用于通知关闭的 inproc 路径。
    shutdown_path: str | None = None

    # 当任一 engine 死亡时置为 True，便于输出处理线程在不持有 client 引用时也能感知状态。
    engine_dead: bool = False

    # 清理后台 socket、任务与 engine manager 等资源。
    def __call__(self):
        """清理后台资源。"""

        # 先标记 engine 已死亡，阻止后续继续通信。
        self.engine_dead = True

        # 如果当前持有 engine manager，则先关闭其管理的全部 engine。
        if self.engine_manager is not None:
            self.engine_manager.shutdown()

        # 如果当前持有 coordinator，也一并关闭。
        if self.coordinator is not None:
            self.coordinator.shutdown()

        # 异步 socket 需要走事件循环内或跨线程的关闭路径。
        if isinstance(self.output_socket, zmq.asyncio.Socket):

            # 异步场景。
            # 尽量复用输出任务所属的事件循环来关闭资源。
            loop = self.output_queue_task._loop if self.output_queue_task else None

            # 汇总所有需要关闭的 socket。
            sockets = (
                self.output_socket,
                self.input_socket,
                self.first_req_send_socket,
                self.first_req_rcv_socket,
                self.stats_update_socket,
            )

            # 汇总所有需要取消的后台任务。
            tasks = (self.output_queue_task, self.stats_update_task)

            # 定义统一的关闭 socket 与取消任务逻辑。
            def close_sockets_and_tasks():
                # 先统一关闭所有 socket。
                close_sockets(sockets)

                # 再取消仍在运行中的后台任务。
                for task in tasks:
                    if task is not None and not task.done():
                        with contextlib.suppress(Exception):
                            task.cancel()

            # 如果还能拿到事件循环，则尽量在对应 loop 中调度关闭逻辑。
            if loop is not None:
                # 如果当前线程就在该 loop 所在线程中，则直接执行关闭。
                if in_loop(loop):
                    close_sockets_and_tasks()

                # 否则通过线程安全方式把关闭逻辑投递给对应事件循环。
                elif not loop.is_closed():
                    loop.call_soon_threadsafe(close_sockets_and_tasks)

            else:
                # 事件循环已关闭，尝试直接清理本地资源。
                del tasks

                # 删除局部关闭函数引用，避免无意义保留。
                del close_sockets_and_tasks

                # 直接关闭所有 socket。
                close_sockets(sockets)

                # 删除任务引用，帮助资源释放。
                del self.output_queue_task

                # 删除统计任务引用，帮助资源释放。
                del self.stats_update_task

        else:
            # 同步场景。
            # 如果不先显式关闭 socket，直接终止 ZMQ context 可能会卡住。
            close_sockets((self.output_socket, self.input_socket))

            # 若配置了 shutdown 通道，则通知同步输出线程自行退出。
            if self.shutdown_path is not None:
                # 必须保证同步输出 socket 在其所属线程中被干净关闭。
                with self.ctx.socket(zmq.PAIR) as shutdown_sender:
                    # 连接到输出线程监听的 inproc 关闭通道。
                    shutdown_sender.connect(self.shutdown_path)

                    # 发送关闭信号。
                    shutdown_sender.send(b"")

    # 检查收到的 frame 是否表示 EngineCore 已死亡。
    def validate_alive(self, frames: Sequence[zmq.Frame]):
        # 若只收到一个 frame，且其内容为 ENGINE_CORE_DEAD，则说明 EngineCore 已死亡。
        if len(frames) == 1 and (frames[0].buffer == EngineCoreProc.ENGINE_CORE_DEAD):
            # 记录死亡状态，供后续调用快速失败。
            self.engine_dead = True

            # 抛出统一的 EngineDeadError 异常。
            raise EngineDeadError()


@dataclass
class ElasticScalingCache:
    existing_core_engines: list[EngineIdentity]
    num_new_core_engines: int
    pending_notifications: dict[EEPNotificationType, set[int]]


def allocate_stateless_group_ports(parallel_config, new_data_parallel_size: int):
    """
    Allocate stateless group ports for elastic EP.
    """
    from cfie.utils.network_utils import get_open_ports_list

    assert parallel_config.enable_elastic_ep, "Elastic EP must be enabled"
    world_size = parallel_config.world_size
    new_world_size_across_dp = world_size * new_data_parallel_size
    num_world_groups = 1
    num_dp_groups = max(1, new_world_size_across_dp // new_data_parallel_size)
    num_ep_groups = max(
        1,
        new_world_size_across_dp
        // (new_data_parallel_size * parallel_config.tensor_parallel_size),
    )
    num_eplb_groups = num_ep_groups
    total_ports_needed = (
                                 num_world_groups + num_dp_groups + num_ep_groups + num_eplb_groups
                         ) * 3 + 5
    all_ports = get_open_ports_list(total_ports_needed)
    new_data_parallel_master_port_list = all_ports[-5:]
    all_ports = all_ports[:-5]
    new_stateless_world_group_port_list = [
        all_ports[i: i + 3] for i in range(0, num_world_groups * 3, 3)
    ]
    start_idx = num_world_groups * 3
    new_stateless_dp_group_port_list = [
        all_ports[i: i + 3] for i in range(start_idx, start_idx + num_dp_groups * 3, 3)
    ]
    start_idx += num_dp_groups * 3
    new_stateless_ep_group_port_list = [
        all_ports[i: i + 3] for i in range(start_idx, start_idx + num_ep_groups * 3, 3)
    ]
    start_idx += num_ep_groups * 3
    new_stateless_eplb_group_port_list = [
        all_ports[i: i + 3]
        for i in range(start_idx, start_idx + num_eplb_groups * 3, 3)
    ]

    parallel_config._stateless_world_group_port_list = (
        new_stateless_world_group_port_list
    )
    parallel_config._stateless_dp_group_port_list = new_stateless_dp_group_port_list
    parallel_config._stateless_ep_group_port_list = new_stateless_ep_group_port_list
    parallel_config._stateless_eplb_group_port_list = new_stateless_eplb_group_port_list
    parallel_config.data_parallel_master_port = new_data_parallel_master_port_list.pop()
    parallel_config._data_parallel_master_port_list = new_data_parallel_master_port_list


# 多进程 EngineCore client 的公共基类。
class MPClient(EngineCoreClient):
    """多进程 `EngineCore` 的基础客户端。

    该类负责把前端 `LLMEngine` 与后台 `EngineCore` 进程串起来：

    - 前端通过 `input_socket` 把 `EngineCoreRequest` 送到后台
    - 后台通过 `output_socket` 把 `EngineCoreOutputs` 回传给前端
    - 若当前 client 处于自托管模式，还会在初始化阶段负责拉起
      `EngineCoreProc` / DP coordinator，并完成启动握手

    该基类只负责多进程通信、资源托管与启动握手，不直接区分同步或异步 API：

    - `SyncMPClient` 用于 `LLMEngine.step()` 这类同步调用链
    - `AsyncMPClient` 用于 `AsyncLLM` 这类 asyncio 调用链
    """

    # ------------------------------- 初始化多进程 EngineCore 客户端 -------------------------------
    def __init__(
            self,
            asyncio_mode: bool,
            cfie_config: CfieConfig,
            executor_class: type[Executor],
            log_stats: bool,
            # 若上层已经把 engine 地址准备好，则通过该字典把地址注入进来。
            # 传入后表示当前 client 只负责“连接已有 engine”，
            # 不再负责分配地址、拉起进程或管理底层 engine 生命周期。
            client_addresses: dict[str, str] | None = None,
    ):
        # ------------------------------- 固化配置并准备通信编解码器 -------------------------------
        # 先保存顶层配置对象，后续并行规模、地址规划和启动逻辑都会频繁读取它。
        self.cfie_config = cfie_config

        # 前端发给后台的请求统一走 msgpack 编码，保证普通 Python 对象能稳定过进程边界。
        self.encoder = MsgpackEncoder()

        # 后台返回的 `EngineCoreOutputs` 也统一按固定 schema 解码，避免手写拆包逻辑。
        self.decoder = MsgpackDecoder(EngineCoreOutputs)

        # 先创建一个真实的同步 ZMQ context；
        # 即使上层走 asyncio，也是在它外面包一层异步 facade，而不是维护两套底层资源。
        sync_ctx = zmq.Context(io_threads=2)

        # 根据当前 client 形态决定暴露同步 context 还是 asyncio context，
        # 这样后续 socket 创建代码可以共用同一套初始化主线。
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx

        # 把 context、socket、线程、engine manager 等资源统一托管到清理器里，
        # 便于初始化失败、显式 shutdown 或 GC 析构时走同一条回收路径。
        self.resources = BackgroundResources(ctx=sync_ctx)

        # 给当前对象注册终结器，避免调用方忘记 `shutdown()` 时遗留后台进程或 socket。
        self._finalizer = weakref.finalize(self, self.resources)

        # 初始化期间任何一步失败都必须回滚已创建资源，因此这里先记录成功标记。
        success = False
        try:
            # ------------------------------- 初始化运行状态与并行配置 -------------------------------
            # 该标记反映“当前 client 负责管理的 engine 是否仍在活跃处理请求”，
            # 后续 DP client 会用它决定是否继续等待远端输出。
            self.engines_running = False

            # 取出并行配置，后面地址规划、DP rank 归属和握手范围都依赖它。
            parallel_config = cfie_config.parallel_config

            # Elastic EP 允许新 engine 接管旧 identity，
            # 因而这里需要给 ROUTER 打开 handover，避免扩缩容时旧连接卡死。
            enable_input_socket_handover = parallel_config.enable_elastic_ep

            # 控制面统计地址不承载正常请求/响应，只用于订阅 DP 协调信息：
            # - waiting / running 计数
            # - wave 状态
            # - elastic EP 控制通知
            # 单 engine 或不需要控制面同步的路径下，它可以保持为空。
            self.stats_update_address: str | None = None

            # ------------------------------- 根据模式建立前后端通信通道 -------------------------------
            # 若上层传入了 `client_addresses`，说明当前 client 只做“外部连接”：
            # 地址由 launcher / API server 预先分配，engine 也已由外部托管，
            # 当前 client 只负责把本地 socket 接到既有 engine 上。
            if client_addresses:
                # 输入地址对应前端写入请求的 ROUTER 通道。
                input_address = client_addresses["input_address"]

                # 输出地址对应后台推送结果的 PULL 通道。
                output_address = client_addresses["output_address"]

                # 若外部同时提供控制面统计地址，也一并保留下来供 DP 逻辑使用。
                self.stats_update_address = client_addresses.get("stats_update_address")

                # 当前 client 在前端侧持有 ROUTER 入口，所有请求都从这里进入后台 engine。
                self.input_socket = self.resources.input_socket = make_zmq_socket(
                    self.ctx,
                    input_address,
                    zmq.ROUTER,
                    bind=True,
                    router_handover=enable_input_socket_handover,
                )

                # 输出侧使用 PULL，从后台异步收集 `EngineCoreOutputs`。
                self.resources.output_socket = make_zmq_socket(
                    self.ctx, output_address, zmq.PULL
                )
            else:  # 默认
                # 未传入外部地址时，当前 client 进入“自托管模式”：
                # 它自己负责规划地址、拉起 engine / coordinator，并在退出时回收这些资源。
                addresses = get_engine_zmq_addresses(cfie_config)

                # 先把前端请求入口绑到本地 ROUTER socket 上，供后续 engine 连接回来。
                self.input_socket = self.resources.input_socket = make_zmq_socket(
                    self.ctx,
                    addresses.inputs[0],
                    zmq.ROUTER,
                    bind=True,
                    router_handover=enable_input_socket_handover,
                )

                # 再创建结果回收通道，后续后台 engine 会把输出推到这里。
                self.resources.output_socket = make_zmq_socket(
                    self.ctx,
                    addresses.outputs[0],
                    zmq.PULL
                )

                # 在握手上下文里拉起后台 engine manager / coordinator；
                # 上下文退出前会保证需要的后台进程已经进入可通信状态。
                with launch_core_engines(
                        cfie_config, executor_class, log_stats, addresses
                ) as (engine_manager, coordinator, addresses):
                    # coordinator 只在 DP 协调场景下存在，后续 shutdown 与监控都要用到它。
                    self.resources.coordinator = coordinator

                    # engine manager 负责托管本地拉起的 engine core 进程集合。
                    self.resources.engine_manager = engine_manager

                # 启动完成后，把最终协商出的控制面统计地址暴露给前端使用。
                self.stats_update_address = addresses.frontend_stats_publish_address

                # 若 coordinator 已启动，则要求它自己报告的统计地址与前端视角保持一致，
                # 避免后续 DP client 订阅到错误通道。
                if coordinator is not None:
                    assert self.stats_update_address == (
                        coordinator.get_stats_publish_address()
                    )

            # ------------------------------- 计算当前 client 负责管理的 engine 范围 -------------------------------
            # DP 总规模决定全局一共有多少条 engine 主线。
            dp_size = parallel_config.data_parallel_size
            # `data_parallel_index` 表示当前 client 管理的起始 DP rank。
            dp_rank = parallel_config.data_parallel_index
            # `data_parallel_size_local` 表示当前节点本地会托管多少个 engine core。
            dp_local_size = parallel_config.data_parallel_size_local
            # offline 模式下，当前 client 只面向一个显式指定的本地 DP rank。
            offline_mode = parallel_config.data_parallel_rank_local is not None

            # 在纯 internal LB 下，当前 client 可能同时感知本地和远端 engines；
            # 在 hybrid / external LB 下，它通常只负责自己本地的 engines。
            num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size

            # 生成当前 client 逻辑上需要对接的 engine rank 列表；
            # 后续 ready 等待、utility 调用和输出状态聚合都按这组 rank 进行。
            self.engine_ranks_managed = (
                [dp_rank] if offline_mode else list(range(dp_rank, dp_rank + num_ranks))
            )

            # 本地 engine 数不能超过当前 client 的管理范围，否则说明 DP 配置自相矛盾。
            assert parallel_config.data_parallel_size_local <= len(
                self.engine_ranks_managed
            )

            # 把每个 engine rank 编码成 ROUTER socket 使用的 identity，
            # 后续所有点对点消息都会靠这个 identity 路由到正确的 engine。
            self.core_engines: list[EngineIdentity] = [
                rank.to_bytes(2, "little") for rank in self.engine_ranks_managed
            ]

            # ------------------------------- 等待所有目标 engine 完成就绪 -------------------------------
            # 初始化一个“尚未 ready 的 engine identity”集合；
            # 只有全部 ready 后，前端才能安全发送正式请求。
            identities = set(self.core_engines)

            # 这里显式构造同步 shadow socket，是为了在初始化阶段用阻塞式等待 ready，
            # 避免把 engine 尚未启动完成的复杂性泄露给后续正常收发逻辑。
            sync_input_socket = zmq.Socket.shadow(self.input_socket)

            # 循环直到当前 client 负责管理的所有 engine 都发来 ready 为止。
            while identities:
                # 启动超时通常意味着大模型权重加载慢、后台进程卡死或握手链路异常；
                # 这里直接抛出明确错误，避免前端静默挂住。
                if not sync_input_socket.poll(
                        timeout=VLLM_ENGINE_READY_TIMEOUT_S * 1000  # convert to ms
                ):
                    raise TimeoutError(
                        f"Timed out waiting for engine core processes to "
                        f"start. This is often caused by slow weight loading "
                        f"for large models. Waited "
                        f"{VLLM_ENGINE_READY_TIMEOUT_S}s (configured by "
                        f"VLLM_ENGINE_READY_TIMEOUT_S). To increase the "
                        f"timeout, set the environment variable: "
                        f"VLLM_ENGINE_READY_TIMEOUT_S=<seconds>"
                    )
                # 读取一条 ready 消息，并根据 identity 把对应 engine 从等待集合里移除。
                identity, _ = sync_input_socket.recv_multipart()
                identities.remove(identity)

            # 默认用第一条 engine identity 作为普通请求的默认目标；
            # DP / LB 场景下，后续子类仍可根据策略动态切换目标。
            self.core_engine: EngineIdentity = self.core_engines[0]

            # utility RPC 需要按 call_id 回填返回值，因此这里准备一个 future 映射表。
            self.utility_results: dict[int, AnyFuture] = {}

            # 某些请求对象里可能夹带 torch tensor backing buffer；
            # 在 ZMQ 真正完成发送前必须保留这些对象引用，避免底层 buffer 被 Python 提前回收。
            self.pending_messages = deque[tuple[zmq.MessageTracker, Any]]()

            # 启动后台监控逻辑，尽早发现 engine core 意外退出并把异常传播回前端。
            self.start_engine_core_monitor()

            # 走到这里说明 socket、后台 engine 和 ready 握手都已经完成。
            success = True
        finally:
            # 任何中途失败都要立即回收已创建资源，避免遗留半初始化的后台进程与 socket。
            if not success:
                self._finalizer()

    # 关闭多进程 client、后台线程和 engine core 进程。
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown engine manager under timeout and clean up resources."""
        # 只有终结器还活着时，才执行一次真正的关闭。
        if self._finalizer.detach() is not None:
            # 若当前 client 直接持有 engine manager，则先按 timeout 关闭它。
            if self.resources.engine_manager is not None:
                self.resources.engine_manager.shutdown(timeout=timeout)
            # 再统一清理 socket、task 和 coordinator 等资源。
            self.resources()

    # 将内部异常包装成更明确的 EngineDeadError。
    def _format_exception(self, e: Exception) -> Exception:
        """If errored, use EngineDeadError so root cause is clear."""
        # 若已知 engine 已死亡，则优先返回统一的 EngineDeadError。
        return (
            EngineDeadError(suppress_context=True) if self.resources.engine_dead else e
        )

    # 在发消息前检查 engine core 是否仍然存活。
    def ensure_alive(self):
        # 一旦后台资源标记 engine 已死，就立即失败，避免继续发送请求。
        if self.resources.engine_dead:
            raise EngineDeadError()

    # 记录一个仍被 ZMQ 持有底层 buffer 的待释放消息。
    def add_pending_message(self, tracker: zmq.MessageTracker, msg: Any):
        # 只有 tracker 尚未完成时，才需要保留消息引用。
        if not tracker.done:
            self.pending_messages.appendleft((tracker, msg))

    # 释放已经被 ZMQ 发送完毕的待释放消息引用。
    def free_pending_messages(self):
        # 从队尾开始清理那些 tracker 已完成的旧消息。
        while self.pending_messages and self.pending_messages[-1][0].done:
            self.pending_messages.pop()

    # 返回当前数据并行 engine 的整体运行状态。
    def dp_engines_running(self) -> bool:
        # 直接返回本地维护的 DP 运行状态位。
        return self.engines_running

    # 启动后台监控线程，及时感知 engine core 进程异常退出。
    def start_engine_core_monitor(self):
        """Start a monitor thread for engine core processes."""
        # 读取当前 client 持有的 engine manager。
        engine_manager = self.resources.engine_manager
        # 若没有本地进程可监控，则直接返回。
        if (
                engine_manager is None
                or not hasattr(engine_manager, "processes")
                or not engine_manager.processes
        ):
            # No engine processes to monitor
            return

        # 取出所有 engine 子进程对象。
        engine_processes = engine_manager.processes
        # 创建对当前 client 的弱引用，避免线程持有强引用阻止回收。
        self_ref = weakref.ref(self)

        # Monitor engine core process liveness. If any die unexpectedly,
        # logs an error, shuts down the client and invokes the failure
        # callback to inform the engine.
        def monitor_engine_cores():
            # 收集每个子进程的 sentinel，用于阻塞等待死亡事件。
            sentinels = [proc.sentinel for proc in engine_processes]
            # 等待任一 engine 进程退出。
            died = multiprocessing.connection.wait(sentinels)
            # 取回 client 弱引用对应的真实对象。
            _self = self_ref()
            # 若 client 已被回收、已关闭或已知死亡，则无需继续处理。
            if not _self or not _self._finalizer.alive or _self.resources.engine_dead:
                return
            # 标记当前 client 感知到底层 engine 已死亡。
            _self.resources.engine_dead = True
            # 根据 sentinel 找到具体是哪一个进程死亡。
            proc_name = next(
                proc.name for proc in engine_processes if proc.sentinel == died[0]
            )
            # 记录明确的错误日志。
            logger.error(
                "Engine core proc %s died unexpectedly, shutting down client.",
                proc_name,
            )
            # 向输出队列注入 EngineDeadError，唤醒正在等待输出的调用方。
            _self.outputs_queue.put_nowait(EngineDeadError())
            # 主动触发 client 关闭流程。
            _self.shutdown()
            # Note: For MPClient, we don't have a failure callback mechanism
            # like MultiprocExecutor, but we set engine_dead flag which will
            # cause subsequent operations to raise EngineDeadError

        # 启动后台守护线程监控 engine 子进程生死。
        Thread(
            target=monitor_engine_cores, daemon=True, name="MPClientEngineMonitor"
        ).start()


def _process_utility_output(
        output: UtilityOutput, utility_results: dict[int, AnyFuture]
):
    """Set the result from a utility method in the waiting future."""
    future = utility_results.pop(output.call_id)
    failure_message = output.failure_message
    try:
        if failure_message is not None:
            future.set_exception(Exception(failure_message))
        else:
            assert output.result is not None
            future.set_result(output.result.result)
    except asyncio.InvalidStateError:
        # This can happen if the future is cancelled due to the
        # original calling task being cancelled.
        if failure_message is not None:
            logger.error(
                "Cancelled call to utility method failed with error: %s",
                failure_message,
            )


# 面向同步 LLM 调用路径的多进程 client。
class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""

    @instrument(span_name="SyncMPClient init")
    # 初始化同步多进程 client，建立 socket、后台线程并等待 engine core 就绪。
    def __init__(
            self,
            cfie_config: CfieConfig,
            executor_class: type[Executor],
            log_stats: bool
    ):
        # 先初始化多进程公共通信基础设施。
        super().__init__(
            asyncio_mode=False,
            cfie_config=cfie_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        # 标记当前是否处于数据并行场景。
        self.is_dp = self.cfie_config.parallel_config.data_parallel_size > 1
        # 创建同步输出队列，供后台线程与前台阻塞读取协作。
        self.outputs_queue = queue.Queue[EngineCoreOutputs | Exception]()

        # Ensure that the outputs socket processing thread does not have
        # a ref to the client which prevents gc.
        # 先提取 context，供后台线程内创建 shutdown socket 使用。
        ctx = self.ctx
        # 取出当前的输出 socket 引用。
        out_socket = self.resources.output_socket
        # 取出响应解码器。
        decoder = self.decoder
        # 取出 utility future 映射，便于线程内回填返回值。
        utility_results = self.utility_results
        # 取出输出队列引用，供线程投递普通结果。
        outputs_queue = self.outputs_queue

        # 为同步输出线程创建一个独立的 inproc 关闭通道。
        shutdown_path = get_open_zmq_inproc_path()
        # 复用资源清理器对象，便于线程中访问共享状态。
        resources = self.resources
        # 把关闭通道路径登记到资源清理器，供 shutdown 时发送退出信号。
        resources.shutdown_path = shutdown_path

        def process_outputs_socket():
            assert isinstance(out_socket, zmq.Socket)
            shutdown_socket = ctx.socket(zmq.PAIR)
            try:
                shutdown_socket.bind(shutdown_path)
                poller = zmq.Poller()
                poller.register(shutdown_socket, zmq.POLLIN)
                poller.register(out_socket, zmq.POLLIN)
                while True:
                    socks = poller.poll()
                    if not socks:
                        continue
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        # shutdown signal, exit thread.
                        break

                    frames = out_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        _process_utility_output(outputs.utility_output, utility_results)
                    else:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            finally:
                # Close sockets.
                shutdown_socket.close(linger=0)
                out_socket.close(linger=0)

        # Process outputs from engine in separate thread.
        self.output_queue_thread = Thread(
            target=process_outputs_socket,
            name="EngineCoreOutputQueueThread",
            daemon=True,
        )
        self.output_queue_thread.start()

        # The thread takes on responsibility for closing the socket.
        self.resources.output_socket = None

    # 阻塞式接收一次 engine core 输出。
    def get_output(self) -> EngineCoreOutputs:
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        # 阻塞等待后台线程推送一条输出或异常。
        outputs = self.outputs_queue.get()

        # 若后台线程传来异常，则在前台重新抛出。
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        # 如果本轮输出标记 wave 已完成，则更新本地运行状态。
        if outputs.wave_complete is not None:
            self.engines_running = False
        # 返回普通的 EngineCoreOutputs。
        return outputs

    # 发送一条普通请求或控制请求到 engine core 进程。
    def _send_input(self, request_type: EngineCoreRequestType, request: Any):
        # 发送前先确认 engine 仍然存活。
        self.ensure_alive()
        # 释放已完成发送的旧消息引用。
        self.free_pending_messages()
        # (Identity, RequestType, SerializedRequest)
        # 把请求类型和序列化后的请求体组装成 multipart 消息。
        msg = (self.core_engine, request_type.value, *self.encoder.encode(request))

        # 若没有额外 tensor buffer，可直接无跟踪发送。
        if len(msg) <= 3:
            # No auxiliary buffers => no tensor backing buffers in request.
            self.input_socket.send_multipart(msg, copy=False)
            return

        # 否则启用 tracker 发送，以便延迟释放底层对象引用。
        tracker = self.input_socket.send_multipart(msg, copy=False, track=True)
        # 记录这条尚未发送完成的消息对象。
        self.add_pending_message(tracker, request)

    # 发送 utility RPC 并同步等待其返回。
    def call_utility(self, method: str, *args) -> Any:
        # 生成一个 utility 调用唯一 ID。
        call_id = uuid.uuid1().int >> 64
        # 创建同步 future，用于等待 utility 结果。
        future: Future[Any] = Future()
        # 把 future 放入映射表，等待输出线程回填。
        self.utility_results[call_id] = future
        # 把 utility 调用包装后发送给远端 engine core。
        self._send_input(EngineCoreRequestType.UTILITY, (0, call_id, method, args))

        # 阻塞等待远端 utility 调用完成。
        return future.result()

    # 从远端 engine core 读取支持的任务类型。
    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        # 通过 utility RPC 获取支持任务列表。
        return self.call_utility("get_supported_tasks")

    # 把请求发送给远端 engine core。
    def add_request(self, request: EngineCoreRequest) -> None:
        # DP 场景下发送请求意味着 engine 进入活跃运行状态。
        if self.is_dp:
            self.engines_running = True
        # 把普通 ADD 请求发往远端 engine core。
        self._send_input(EngineCoreRequestType.ADD, request)

    # 向远端 engine core 发送 abort 请求。
    def abort_requests(self, request_ids: list[str]) -> None:
        # 仅在有 request_id 且 engine 未死时才真正发送中止请求。
        if request_ids and not self.resources.engine_dead:
            self._send_input(EngineCoreRequestType.ABORT, request_ids)

    # 同步切换远端 engine core 的 profiling 状态。
    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        # 通过 utility RPC 控制远端 profiling。
        self.call_utility("profile", is_start, profile_prefix)

    # 同步重置远端多模态缓存。
    def reset_mm_cache(self) -> None:
        # 通过 utility RPC 清空远端多模态缓存。
        self.call_utility("reset_mm_cache")

    # 同步重置远端 prefix cache。
    def reset_prefix_cache(
            self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        # 通过 utility RPC 返回远端 prefix cache 重置结果。
        return self.call_utility(
            "reset_prefix_cache", reset_running_requests, reset_connector
        )

    # 同步重置远端 encoder cache。
    def reset_encoder_cache(self) -> None:
        # 通过 utility RPC 清空远端 encoder cache。
        self.call_utility("reset_encoder_cache")

    # 同步向远端加载一个 LoRA。
    def add_lora(self, lora_request: LoRARequest) -> bool:
        # 通过 utility RPC 返回远端 LoRA 加载结果。
        return self.call_utility("add_lora", lora_request)

    # 同步从远端卸载一个 LoRA。
    def remove_lora(self, lora_id: int) -> bool:
        # 通过 utility RPC 返回远端 LoRA 卸载结果。
        return self.call_utility("remove_lora", lora_id)

    # 同步查询远端已加载 LoRA 列表。
    def list_loras(self) -> set[int]:
        # 通过 utility RPC 返回远端 LoRA 列表。
        return self.call_utility("list_loras")

    # 同步固定远端指定 LoRA。
    def pin_lora(self, lora_id: int) -> bool:
        # 通过 utility RPC 返回远端 LoRA 固定结果。
        return self.call_utility("pin_lora", lora_id)

    # 同步让远端 engine core 进入休眠。
    def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None:
        # 通过 utility RPC 控制远端休眠。
        self.call_utility("sleep", level, mode)

    # 同步唤醒远端 engine core。
    def wake_up(self, tags: list[str] | None = None) -> None:
        # 通过 utility RPC 控制远端唤醒。
        self.call_utility("wake_up", tags)

    # 同步查询远端 engine core 是否休眠。
    def is_sleeping(self) -> bool:
        # 通过 utility RPC 查询远端休眠状态。
        return self.call_utility("is_sleeping")

    # 同步触发远端执行一次 dummy batch。
    def execute_dummy_batch(self) -> None:
        # 通过 utility RPC 触发远端 dummy batch。
        self.call_utility("execute_dummy_batch")

    # 同步在远端执行 collective RPC。
    def collective_rpc(
            self,
            method: str | Callable[..., _R],
            timeout: float | None = None,
            args: tuple = (),
            kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        # 通过 utility RPC 请求远端执行 collective 调用并返回结果列表。
        return self.call_utility("collective_rpc", method, timeout, args, kwargs)

    # 同步要求远端保存分片状态。
    def save_sharded_state(
            self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        # 通过 utility RPC 请求远端保存状态分片。
        self.call_utility("save_sharded_state", path, pattern, max_size)


# 面向 asyncio/AsyncLLM 调用路径的多进程 client。
class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    @instrument(span_name="AsyncMPClient init")
    # 初始化异步多进程 client，并在可用时提前启动输出处理任务。
    def __init__(
            self,
            cfie_config: CfieConfig,
            executor_class: type[Executor],
            log_stats: bool,
            client_addresses: dict[str, str] | None = None,
            client_count: int = 1,
            client_index: int = 0,
    ):
        # 先初始化异步多进程公共通信基础设施。
        super().__init__(
            asyncio_mode=True,
            cfie_config=cfie_config,
            executor_class=executor_class,
            log_stats=log_stats,
            client_addresses=client_addresses,
        )

        # 保存当前异步 client 总数。
        self.client_count = client_count
        # 保存当前 client 在 client 集合中的索引。
        self.client_index = client_index
        # 创建异步输出队列，供后台 task 与调用方协作。
        self.outputs_queue = asyncio.Queue[EngineCoreOutputs | Exception]()
        try:
            # If we are running in an asyncio event loop, start the queue task.
            # Otherwise, it will be started lazily. If it is not started here,
            # we could miss EXECUTOR_FAILED messages from engine core if they
            # occur prior to any requests being sent.
            # 尝试获取当前运行中的事件循环。
            asyncio.get_running_loop()
            # 如果当前就在事件循环中，则立即启动输出处理 task。
            self._ensure_output_queue_task()
        except RuntimeError:
            # 若当前不在事件循环中，则延后到首次异步调用时再启动。
            pass

    # 确保异步输出处理 task 已启动。
    def _ensure_output_queue_task(self):
        # 取出统一资源对象，便于访问 socket 与 task 引用。
        resources = self.resources
        # 如果输出处理 task 已存在，则无需重复创建。
        if resources.output_queue_task is not None:
            return

        # Perform IO in separate task to parallelize as much as possible.
        # Avoid task having direct reference back to the client.
        # 取出响应解码器。
        decoder = self.decoder
        # 取出 utility future 映射。
        utility_results = self.utility_results
        # 取出输出队列引用。
        outputs_queue = self.outputs_queue
        # 若子类定义了输出后处理钩子，则取出该处理器。
        output_handler: (
                Callable[[AsyncMPClient, EngineCoreOutputs], Awaitable[None]] | None
        ) = getattr(self.__class__, "process_engine_outputs", None)
        # 只有需要回调到 client 时，才创建弱引用避免循环引用。
        _self_ref = weakref.ref(self) if output_handler else None
        # 取出当前输出 socket。
        output_socket = resources.output_socket
        # 异步输出 task 要求输出 socket 一定已经创建。
        assert output_socket is not None

        # 若子类定义了 EEP 通知处理钩子，则取出该回调。
        notification_callback_handler: (
                Callable[[AsyncMPClient, Sequence[Any]], Any] | None
        ) = getattr(self.__class__, "eep_process_engine_core_notification", None)

        async def process_outputs_socket():
            try:
                while True:
                    frames = await output_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    if outputs.utility_output:
                        if (
                                outputs.utility_output.call_id == EEP_NOTIFICATION_CALL_ID
                                and notification_callback_handler is not None
                        ):
                            assert _self_ref is not None
                            _self = _self_ref()
                            if not _self:
                                return
                            if outputs.utility_output.result is None:
                                continue
                            notification_data = outputs.utility_output.result.result
                            assert isinstance(notification_data, Sequence)
                            assert len(notification_data) == 2
                            asyncio.create_task(
                                notification_callback_handler(_self, notification_data)
                            )
                        else:
                            _process_utility_output(
                                outputs.utility_output, utility_results
                            )
                        continue

                    if output_handler is not None:
                        assert _self_ref is not None
                        _self = _self_ref()
                        if not _self:
                            # Client has been garbage collected, abort.
                            return
                        await output_handler(_self, outputs)

                    if outputs.outputs or outputs.scheduler_stats:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            except asyncio.CancelledError:
                outputs_queue.put_nowait(EngineDeadError())

        resources.output_queue_task = asyncio.create_task(
            process_outputs_socket(), name="EngineCoreOutputQueueTask"
        )

    # 异步接收一次 engine core 输出。
    async def get_output_async(self) -> EngineCoreOutputs:
        # 取输出前先确保后台输出 task 已启动。
        self._ensure_output_queue_task()
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        # 异步输出路径要求输出队列必须已经准备好。
        assert self.outputs_queue is not None
        # 等待后台 task 投递一条输出或异常。
        outputs = await self.outputs_queue.get()
        # 若后台 task 投递的是异常，则在前台重新抛出。
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        # 返回普通的 EngineCoreOutputs。
        return outputs

    # 组装并发送一条异步输入消息。
    def _send_input(
            self,
            request_type: EngineCoreRequestType,
            request: Any,
            engine: EngineIdentity | None = None,
    ) -> Awaitable[Any]:
        # 未显式指定目标 engine 时，默认发给主 engine。
        if engine is None:
            engine = self.core_engine

        # 把请求类型与序列化后的请求体组装成消息内容。
        message = (request_type.value, *self.encoder.encode(request))
        # 交给通用消息发送函数处理。
        return self._send_input_message(message, engine, request)

    # 发送已编码好的消息，并在需要时保留底层对象引用。
    def _send_input_message(
            self, message: tuple[bytestr, ...], engine: EngineIdentity, objects: Any
    ) -> Awaitable[Any]:
        """
        objects is a reference to retain until zmq is finished with the
        buffers, in case they were extracted from tensors in the request.
        """
        # 发送前先确认 engine 仍然存活。
        self.ensure_alive()
        # 释放已经发送完成的旧消息引用。
        self.free_pending_messages()

        # 在编码消息前面补上目标 engine identity。
        msg = (engine,) + message
        # 若没有附带对象或辅助 buffer，则可直接异步发送。
        if not objects or len(msg) <= 3:
            # No auxiliary buffers => no tensor backing buffers in request.
            return self.input_socket.send_multipart(msg, copy=False)

        # 若存在需要保留的对象，则启用 tracker 发送。
        future: asyncio.Future[zmq.MessageTracker]
        future = self.input_socket.send_multipart(msg, copy=False, track=True)

        def add_pending(f: asyncio.Future[zmq.MessageTracker]):
            with contextlib.suppress(BaseException):
                self.add_pending_message(f.result(), objects)

        future.add_done_callback(add_pending)
        return future

    # 异步发送一个 utility RPC 到默认 engine。
    async def call_utility_async(self, method: str, *args) -> Any:
        # 默认把 utility 请求发送到主 engine。
        return await self._call_utility_async(method, *args, engine=self.core_engine)

    # 异步发送一个 utility RPC 到指定 engine。
    async def _call_utility_async(
            self, method: str, *args, engine: EngineIdentity
    ) -> Any:
        # 生成当前 utility 调用的唯一 ID。
        call_id = uuid.uuid1().int >> 64
        # 为本次 utility 调用创建一个异步 future。
        future = asyncio.get_running_loop().create_future()
        # 登记 future，等待输出处理 task 在结果返回时回填。
        self.utility_results[call_id] = future
        # 把 utility 调用序列化成请求消息。
        message = (
            EngineCoreRequestType.UTILITY.value,
            *self.encoder.encode((self.client_index, call_id, method, args)),
        )
        # 把 utility 消息发往目标 engine。
        await self._send_input_message(message, engine, args)
        # 确保输出处理 task 正在运行，以便消费返回结果。
        self._ensure_output_queue_task()
        # 等待 utility 返回并把结果交给调用方。
        return await future

    # 异步查询底层 engine 支持的任务类型。
    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
        # 通过 utility RPC 获取远端支持任务列表。
        return await self.call_utility_async("get_supported_tasks")

    # 异步向远端 engine core 提交请求。
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        # 先把当前 client 索引写入请求，供后端区分来源。
        request.client_index = self.client_index
        # 异步发送普通 ADD 请求。
        await self._send_input(EngineCoreRequestType.ADD, request)
        # 发送请求后确保输出处理 task 已启动。
        self._ensure_output_queue_task()

    # 异步向远端发送 abort 请求。
    async def abort_requests_async(self, request_ids: list[str]) -> None:
        # 仅在有 request_id 且 engine 未死时才真正发送中止请求。
        if request_ids and not self.resources.engine_dead:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)

    # 异步暂停远端调度器。
    async def pause_scheduler_async(
            self, mode: PauseMode = "abort", clear_cache: bool = True
    ) -> None:
        # 通过 utility RPC 暂停远端调度器。
        await self.call_utility_async("pause_scheduler", mode, clear_cache)

    # 异步恢复远端调度器。
    async def resume_scheduler_async(self) -> None:
        # 通过 utility RPC 恢复远端调度器。
        await self.call_utility_async("resume_scheduler")

    # 异步查询远端调度器是否已暂停。
    async def is_scheduler_paused_async(self) -> bool:
        # 通过 utility RPC 查询暂停状态。
        return await self.call_utility_async("is_scheduler_paused")

    # 异步切换远端 profiling 状态。
    async def profile_async(
            self, is_start: bool = True, profile_prefix: str | None = None
    ) -> None:
        # 通过 utility RPC 控制远端 profiling。
        await self.call_utility_async("profile", is_start, profile_prefix)

    # 异步重置远端多模态缓存。
    async def reset_mm_cache_async(self) -> None:
        # 通过 utility RPC 清空远端多模态缓存。
        await self.call_utility_async("reset_mm_cache")

    # 异步重置远端 prefix cache。
    async def reset_prefix_cache_async(
            self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        # 通过 utility RPC 返回远端 prefix cache 重置结果。
        return await self.call_utility_async(
            "reset_prefix_cache", reset_running_requests, reset_connector
        )

    # 异步重置远端 encoder cache。
    async def reset_encoder_cache_async(self) -> None:
        # 通过 utility RPC 清空远端 encoder cache。
        await self.call_utility_async("reset_encoder_cache")

    # 异步让远端 engine core 进入休眠。
    async def sleep_async(self, level: int = 1, mode: PauseMode = "abort") -> None:
        # 通过 utility RPC 控制远端休眠。
        await self.call_utility_async("sleep", level, mode)

    # 异步唤醒远端 engine core。
    async def wake_up_async(self, tags: list[str] | None = None) -> None:
        # 通过 utility RPC 控制远端唤醒。
        await self.call_utility_async("wake_up", tags)

    # 异步查询远端 engine core 是否休眠。
    async def is_sleeping_async(self) -> bool:
        # 通过 utility RPC 查询远端休眠状态。
        return await self.call_utility_async("is_sleeping")

    # 异步触发远端执行一次 dummy batch。
    async def execute_dummy_batch_async(self) -> None:
        # 通过 utility RPC 触发远端 dummy batch。
        await self.call_utility_async("execute_dummy_batch")

    # 异步向远端加载一个 LoRA。
    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        # 通过 utility RPC 返回远端 LoRA 加载结果。
        return await self.call_utility_async("add_lora", lora_request)

    # 异步从远端卸载一个 LoRA。
    async def remove_lora_async(self, lora_id: int) -> bool:
        # 通过 utility RPC 返回远端 LoRA 卸载结果。
        return await self.call_utility_async("remove_lora", lora_id)

    # 异步查询远端已加载 LoRA 列表。
    async def list_loras_async(self) -> set[int]:
        # 通过 utility RPC 返回远端 LoRA 列表。
        return await self.call_utility_async("list_loras")

    # 异步固定远端指定 LoRA。
    async def pin_lora_async(self, lora_id: int) -> bool:
        # 通过 utility RPC 返回远端 LoRA 固定结果。
        return await self.call_utility_async("pin_lora", lora_id)

    # 异步要求远端保存分片状态。
    async def save_sharded_state_async(
            self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        # 通过 utility RPC 请求远端保存状态分片。
        await self.call_utility_async("save_sharded_state", path, pattern, max_size)

    # 异步在远端执行 collective RPC。
    async def collective_rpc_async(
            self,
            method: str | Callable[..., _R],
            timeout: float | None = None,
            args: tuple = (),
            kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        # 通过 utility RPC 请求远端执行 collective 调用并返回结果列表。
        return await self.call_utility_async(
            "collective_rpc", method, timeout, args, kwargs
        )


class DPAsyncMPClient(AsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Assumes external load-balancing by default."""

    def __init__(
            self,
            cfie_config: CfieConfig,
            executor_class: type[Executor],
            log_stats: bool,
            client_addresses: dict[str, str] | None = None,
            client_count: int = 1,
            client_index: int = 0,
    ):
        self.current_wave = 0

        super().__init__(
            cfie_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )

        # List of [waiting, running] pair per engine.
        # Used only by DPLBAsyncMPClient subclass.
        self.lb_engines: list[list[int]] = [[0, 0] for _ in self.core_engines]

        self.eep_scaling_cache: ElasticScalingCache | None = None

        self.first_req_sock_addr = get_open_zmq_inproc_path()
        self.first_req_send_socket = self.resources.first_req_send_socket = (
            make_zmq_socket(self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=True)
        )
        try:
            # If we are running in an asyncio event loop, start the stats task.
            # Otherwise, it will be started lazily.
            asyncio.get_running_loop()
            self._ensure_stats_update_task()
        except RuntimeError:
            pass

    def _ensure_stats_update_task(self):
        resources = self.resources
        if resources.stats_update_task is not None:
            return

        assert self.stats_update_address is not None
        stats_addr: str = self.stats_update_address
        assert len(self.engine_ranks_managed) > 0

        async def run_engine_stats_update_task():
            with (
                make_zmq_socket(self.ctx, stats_addr, zmq.XSUB, linger=0) as socket,
                make_zmq_socket(
                    self.ctx, self.first_req_sock_addr, zmq.PAIR, bind=False, linger=0
                ) as first_req_rcv_socket,
            ):
                assert isinstance(socket, zmq.asyncio.Socket)
                assert isinstance(first_req_rcv_socket, zmq.asyncio.Socket)
                self.resources.stats_update_socket = socket
                self.resources.first_req_rcv_socket = first_req_rcv_socket
                # Send subscription message.
                await socket.send(b"\x01")

                poller = zmq.asyncio.Poller()
                poller.register(socket, zmq.POLLIN)
                poller.register(first_req_rcv_socket, zmq.POLLIN)

                while True:
                    events = await poller.poll()
                    if (
                            not self.engines_running
                            and len(events) == 2
                            or (events[0][0] == first_req_rcv_socket)
                    ):
                        # Check if this is a regular request notification or
                        # scale up notification
                        buf = first_req_rcv_socket.recv(flags=zmq.NOBLOCK).result()

                        decoded = msgspec.msgpack.decode(buf)
                        if (
                                isinstance(decoded, (list, tuple))
                                and len(decoded) == 2
                                and decoded[0] == "SCALE_ELASTIC_EP"
                        ):
                            # Extract new engine count from the decoded message
                            new_engine_count = decoded[1]
                            # Update engine_ranks_managed and count_slice
                            parallel_config = self.cfie_config.parallel_config
                            dp_size = parallel_config.data_parallel_size
                            dp_rank = parallel_config.data_parallel_rank
                            assert dp_rank == 0
                            assert dp_size == new_engine_count
                            assert not (
                                    parallel_config.data_parallel_hybrid_lb
                                    or parallel_config.data_parallel_external_lb
                            )
                            num_ranks = dp_size
                            self.engine_ranks_managed = list(
                                range(dp_rank, dp_rank + num_ranks)
                            )
                            if len(self.lb_engines) < new_engine_count:
                                self.lb_engines = self.lb_engines + [
                                    [0, 0]
                                    for _ in range(
                                        new_engine_count - len(self.lb_engines)
                                    )
                                ]
                            else:
                                self.lb_engines = self.lb_engines[:new_engine_count]
                            # Send scale up notification to coordinator
                            scale_msg = msgspec.msgpack.encode(
                                ("SCALE_ELASTIC_EP", new_engine_count)
                            )
                            await socket.send(scale_msg)
                            continue

                        # we're sending a request while the engines are
                        # paused, so that it can wake the others up
                        # (to run dummy EP loop).
                        assert decoded[0] == "FIRST_REQ"
                        target_eng_index = decoded[1]
                        self.engines_running = True
                        msg = msgspec.msgpack.encode(
                            (target_eng_index, self.current_wave)
                        )
                        await socket.send(msg)

                    buf = None
                    while True:
                        # Drain all stats events (we only care about latest).
                        future: asyncio.Future[bytes] = socket.recv(flags=zmq.NOBLOCK)
                        if isinstance(future.exception(), zmq.Again):
                            break
                        buf = future.result()
                    if buf is None:
                        continue

                    # Update local load-balancing state.
                    counts, wave, running = msgspec.msgpack.decode(buf)
                    self.current_wave = wave
                    self.engines_running = running
                    if counts is not None:
                        # Running and waiting counts are global from the
                        # Coordinator including all EngineCores. Slice to get
                        # just the cores managed by this client.
                        ranks = self.engine_ranks_managed
                        count_slice = slice(ranks[0], ranks[-1] + 1)
                        sliced_counts = counts[count_slice]
                        self.lb_engines = sliced_counts
                        logger.debug(
                            "Received counts: %s (%s)", sliced_counts, count_slice
                        )

        resources.stats_update_task = asyncio.create_task(
            run_engine_stats_update_task()
        )

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        self._ensure_stats_update_task()

        request.current_wave = self.current_wave
        request.client_index = self.client_index

        chosen_engine = self.get_core_engine_for_request(request)
        to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
        if not self.engines_running:
            # Notify coordinator that we're sending a request
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))
            await self.first_req_send_socket.send(req_msg)

        await to_await

        self._ensure_output_queue_task()

    def get_core_engine_for_request(self, request: EngineCoreRequest):
        return self.core_engine


class DPLBAsyncMPClient(DPAsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Load-balances between multiple engine processes."""

    def __init__(
            self,
            cfie_config: CfieConfig,
            executor_class: type[Executor],
            log_stats: bool,
            client_addresses: dict[str, str] | None = None,
            client_count: int = 1,
            client_index: int = 0,
    ):
        self.client_count = client_count

        # To route aborts to the correct engine.
        self.reqs_in_flight: dict[str, EngineIdentity] = {}

        super().__init__(
            cfie_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )

        assert len(self.core_engines) > 1

        self.eng_start_index = (
                                       len(self.core_engines) * self.client_index
                               ) // client_count

    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        # Engines are in rank order.
        if (eng_index := request.data_parallel_rank) is None and (
                eng_index := get_late_interaction_engine_index(
                    request.pooling_params, len(self.core_engines)
                )
        ) is None:
            current_counts = self.lb_engines
            # TODO use P2C alg for larger DP sizes
            num_engines = len(current_counts)
            min_score = sys.maxsize
            eng_index = 0
            for i in range(num_engines):
                # Start from client_index to help with balancing when engines
                # are empty.
                idx = (self.eng_start_index + i) % num_engines
                waiting, running = current_counts[idx]
                score = waiting * 4 + running
                if score < min_score:
                    min_score = score
                    eng_index = idx
            # Increment local waiting count for better balancing between stats
            # updates from the coordinator (which happen every 100ms).
            current_counts[eng_index][0] += self.client_count

        chosen_engine = self.core_engines[eng_index]
        # Record which engine is chosen for this request, to handle aborts.
        self.reqs_in_flight[request.request_id] = chosen_engine
        return chosen_engine

    async def call_utility_async(self, method: str, *args) -> Any:
        # Only the result from the first engine is returned.
        return (
            await asyncio.gather(
                *[
                    self._call_utility_async(method, *args, engine=engine)
                    for engine in self.core_engines
                ]
            )
        )[0]

    @staticmethod
    async def process_engine_outputs(
            self: "DPLBAsyncMPClient", outputs: EngineCoreOutputs
    ):
        if outputs.finished_requests and self.reqs_in_flight:
            for req_id in outputs.finished_requests:
                self.reqs_in_flight.pop(req_id, None)

    @staticmethod
    async def eep_process_engine_core_notification(
            self: "DPLBAsyncMPClient", notification_data: tuple[str, int]
    ):
        cache = self.eep_scaling_cache
        notification_type_str, dp_rank = notification_data
        try:
            notification_type = EEPNotificationType(notification_type_str)
        except ValueError as e:
            raise ValueError(
                f"Unknown EEP notification type: {notification_type_str}"
            ) from e

        if notification_type == EEPNotificationType.RECONFIGURE_FINISHED:
            from cfie.v1.engine import UtilityResult

            # NOTE(yongji): process a dummy UtilityOutput to resolve the future
            # awaited in _eep_wait_for_setup_switch_complete(), signaling that
            # all engine cores have completed reconfiguration.
            dummy_output = UtilityOutput(
                call_id=EEP_NOTIFICATION_CALL_ID, result=UtilityResult(None)
            )
            _process_utility_output(dummy_output, self.utility_results)
            return
        assert cache is not None
        if notification_type not in cache.pending_notifications:
            cache.pending_notifications[notification_type] = set()
        if dp_rank in cache.pending_notifications[notification_type]:
            raise ValueError(
                f"Duplicate notification {notification_type} from dp_rank {dp_rank}"
            )
        cache.pending_notifications[notification_type].add(dp_rank)
        if len(cache.pending_notifications[notification_type]) >= abs(
                cache.num_new_core_engines
        ):
            if notification_type == EEPNotificationType.SHUTDOWN_COMPLETE:
                assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
                assert cache.num_new_core_engines < 0
                old_dp_size = len(cache.existing_core_engines)
                new_dp_size = old_dp_size + cache.num_new_core_engines
                self.resources.engine_manager.scale_down_elastic_ep(
                    old_dp_size, new_dp_size
                )
            else:
                await asyncio.gather(
                    *[
                        self._call_utility_async(
                            "eep_handle_engine_core_notification",
                            notification_type,
                            engine=engine,
                        )
                        for engine in cache.existing_core_engines
                    ]
                )
            cache.pending_notifications[notification_type] = set()
            if notification_type in [
                EEPNotificationType.SHUTDOWN_COMPLETE,
                EEPNotificationType.NEW_CORE_ENGINES_WEIGHTS_INIT_READY,
            ]:
                self.eep_scaling_cache = None

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if not request_ids or self.resources.engine_dead:
            return

        if len(request_ids) == 1:
            # Fast-path common case.
            if engine := self.reqs_in_flight.get(request_ids[0]):
                await self._abort_requests(request_ids, engine)
            return

        by_engine = defaultdict[EngineIdentity, list[str]](list)
        for req_id in request_ids:
            if engine := self.reqs_in_flight.get(req_id):
                by_engine[engine].append(req_id)
        for engine, req_ids in by_engine.items():
            await self._abort_requests(req_ids, engine)

    async def _abort_requests(
            self, request_ids: list[str], engine: EngineIdentity
    ) -> None:
        await self._send_input(EngineCoreRequestType.ABORT, request_ids, engine)

    async def scale_elastic_ep(self, new_data_parallel_size: int) -> None:
        """Scale elastic EP data parallel size"""
        cur_data_parallel_size = len(self.core_engines)

        assert new_data_parallel_size != cur_data_parallel_size, (
            f"new_data_parallel_size {new_data_parallel_size} must be "
            f"different from cur_data_parallel_size {cur_data_parallel_size}"
        )

        assert self.cfie_config.parallel_config.data_parallel_backend == "ray", (
            "Only ray DP backend supports scaling elastic EP"
        )

        scale_up = new_data_parallel_size > cur_data_parallel_size

        if scale_up:
            await self._scale_up_elastic_ep(
                cur_data_parallel_size, new_data_parallel_size
            )
        else:
            await self._scale_down_elastic_ep(
                cur_data_parallel_size, new_data_parallel_size
            )

    async def _eep_wait_for_setup_switch_complete(self) -> None:
        """
        Wait for core engines to switch to the new setup.

        In eep_process_engine_core_notification(), a dummy UtilityOutput with
        EEP_NOTIFICATION_CALL_ID will be set when RECONFIGURE_FINISHED
        notification is received from engine 0. We create a future with
        that call_id and wait for it to be resolved.
        """
        future = asyncio.get_running_loop().create_future()
        self.utility_results[EEP_NOTIFICATION_CALL_ID] = future
        self._ensure_output_queue_task()
        await future

    async def _scale_up_elastic_ep(
            self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None:
        """Scale up the data parallel size by creating new engine cores
        and reconfiguring existing ones."""
        cur_data_parallel_size = len(self.core_engines)

        self.eep_scaling_cache = ElasticScalingCache(
            existing_core_engines=self.core_engines.copy(),
            num_new_core_engines=new_data_parallel_size - cur_data_parallel_size,
            pending_notifications=dict(),
        )

        parallel_config = self.cfie_config.parallel_config
        allocate_stateless_group_ports(parallel_config, new_data_parallel_size)

        # Phase 1: Send reconfig messages to existing engines
        reconfig_futures = []
        for engine in self.core_engines:
            reconfig_request = ReconfigureDistributedRequest(
                new_data_parallel_size=new_data_parallel_size,
                new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_master_ip=parallel_config.data_parallel_master_ip,
                new_data_parallel_master_port=parallel_config.data_parallel_master_port,
                new_data_parallel_master_port_list=parallel_config._data_parallel_master_port_list,
                new_stateless_world_group_port_list=parallel_config._stateless_world_group_port_list,
                new_stateless_dp_group_port_list=parallel_config._stateless_dp_group_port_list,
                new_stateless_ep_group_port_list=parallel_config._stateless_ep_group_port_list,
                new_stateless_eplb_group_port_list=parallel_config._stateless_eplb_group_port_list,
            )
            coro = self._call_utility_async(
                "reinitialize_distributed", reconfig_request, engine=engine
            )
            reconfig_futures.append(asyncio.create_task(coro))

        # Phase 2: Create new engines
        assert isinstance(self.resources.engine_manager, CoreEngineActorManager)
        parallel_config.eplb_config.num_redundant_experts = 0
        start_new_worker_future = asyncio.to_thread(
            self.resources.engine_manager.scale_up_elastic_ep,
            self.cfie_config,
            new_data_parallel_size,
        )
        wait_future = self._eep_wait_for_setup_switch_complete()

        # Phase 3: Wait for new engines to be created
        # and reconfig messages to be received
        await asyncio.gather(start_new_worker_future, *reconfig_futures)
        logger.info("[Elastic EP] Successfully started new engines")

        # Create new CoreEngine objects for the new engines
        new_engine_identities = set()
        for i in range(cur_data_parallel_size, new_data_parallel_size):
            new_engine = i.to_bytes(2, "little")
            self.core_engines.append(new_engine)
            # NOTE(yongji): we don't update lb_engines here,
            # we let run_engine_stats_update_task to update it.
            new_engine_identities.add(new_engine)

        # Wait for ready messages from new engines on the input socket
        sync_input_socket = zmq.Socket.shadow(self.input_socket)
        while new_engine_identities:
            if not sync_input_socket.poll(
                    timeout=VLLM_ENGINE_READY_TIMEOUT_S * 1000  # convert to ms
            ):
                raise TimeoutError(
                    f"Timed out waiting for new engine core processes to "
                    f"start. Waited "
                    f"{VLLM_ENGINE_READY_TIMEOUT_S}s (configured by "
                    f"VLLM_ENGINE_READY_TIMEOUT_S). To increase the "
                    f"timeout, set the environment variable: "
                    f"VLLM_ENGINE_READY_TIMEOUT_S=<seconds>"
                )
            identity, _ = sync_input_socket.recv_multipart()
            new_engine_identities.discard(identity)

        # NOTE(yongji): Before we schedule any requests on the new workers,
        # we should wait for them to switch to the new setup.
        await wait_future
        # Update the parallel config
        self.cfie_config.parallel_config.data_parallel_size = new_data_parallel_size
        # Notify coordinator about scale up through existing
        # stats_update_task connection
        self._ensure_stats_update_task()
        scale_up_marker = msgspec.msgpack.encode(
            ("SCALE_ELASTIC_EP", new_data_parallel_size)
        )
        await self.first_req_send_socket.send(scale_up_marker)

        logger.info(
            "[Elastic EP] Scale up completed, new data parallel size: %s",
            new_data_parallel_size,
        )

    async def _scale_down_elastic_ep(
            self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None:
        """Scale down the data parallel size by shutting down and
        reconfiguring existing engine cores."""
        cur_data_parallel_size = len(self.core_engines)

        self.eep_scaling_cache = ElasticScalingCache(
            existing_core_engines=self.core_engines.copy(),
            num_new_core_engines=new_data_parallel_size - cur_data_parallel_size,
            pending_notifications=dict(),
        )

        parallel_config = self.cfie_config.parallel_config
        allocate_stateless_group_ports(parallel_config, new_data_parallel_size)

        reconfig_futures = []
        for cur_dp_rank, engine in enumerate(self.core_engines):
            reconfig_request = ReconfigureDistributedRequest(
                new_data_parallel_size=new_data_parallel_size,
                new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
                new_data_parallel_master_ip=parallel_config.data_parallel_master_ip,
                new_data_parallel_master_port=parallel_config.data_parallel_master_port,
                new_data_parallel_master_port_list=parallel_config._data_parallel_master_port_list,
                new_stateless_world_group_port_list=parallel_config._stateless_world_group_port_list,
                new_stateless_dp_group_port_list=parallel_config._stateless_dp_group_port_list,
                new_stateless_ep_group_port_list=parallel_config._stateless_ep_group_port_list,
                new_stateless_eplb_group_port_list=parallel_config._stateless_eplb_group_port_list,
            )
            if cur_dp_rank >= new_data_parallel_size:
                reconfig_request.new_data_parallel_rank = (
                    ReconfigureRankType.SHUTDOWN_CURRENT_RANK
                )
            coro = self._call_utility_async(
                "reinitialize_distributed", reconfig_request, engine=engine
            )
            reconfig_futures.append(asyncio.create_task(coro))

        # NOTE(yongji): Immediately stop sending requests to the removing engines.
        self.core_engines = self.core_engines[:new_data_parallel_size]
        self.lb_engines = self.lb_engines[:new_data_parallel_size]
        wait_future = self._eep_wait_for_setup_switch_complete()

        await asyncio.gather(*reconfig_futures)

        self.cfie_config.parallel_config.data_parallel_size = new_data_parallel_size
        self._ensure_stats_update_task()
        scale_down_marker = msgspec.msgpack.encode(
            ("SCALE_ELASTIC_EP", new_data_parallel_size)
        )
        await self.first_req_send_socket.send(scale_down_marker)

        # NOTE(yongji): Unlike scaling up,
        # here we don't actually need to wait for the setup switch to complete.
        # We may want to remove it in the future.
        await wait_future
        logger.info(
            "[Elastic EP] Scale down completed, new data parallel size: %s",
            new_data_parallel_size,
        )
