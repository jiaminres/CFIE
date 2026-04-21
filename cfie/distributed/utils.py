# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/utils.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import dataclasses
import os
import pickle
import socket
import sys
import time
import uuid
from collections import deque
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import torch
from torch.distributed import ProcessGroup, Store, TCPStore
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    _get_default_timeout,
    _unregister_process_group,
)
from torch.distributed.rendezvous import rendezvous

import cfie.envs as envs
from cfie.logger import init_logger
from cfie.utils.network_utils import get_tcp_uri
from cfie.utils.system_utils import suppress_stdout

logger = init_logger(__name__)

# We prefer to use os.sched_yield as it results in tighter polling loops,
# measured to be around 3e-7 seconds. However on earlier versions of Python
# os.sched_yield() does not release the GIL, so we fall back to time.sleep(0)
USE_SCHED_YIELD = (sys.version_info[:3] >= (3, 11, 1)) or (
        sys.version_info[:2] == (3, 10) and sys.version_info[2] >= 8
)


def sched_yield():
    if USE_SCHED_YIELD:
        os.sched_yield()
    else:
        time.sleep(0)


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> Sequence[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # NOTE: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def get_pp_indices(
        num_hidden_layers: int,
        pp_rank: int,
        pp_size: int
) -> tuple[int, int]:
    """
    尽量把 hidden layers 均匀分配到各个流水线并行分区中。

    若总层数不能整除分区数，则把多出来的层尽量平均分给非最后一个分区。
    最后一个分区通常还会带额外的 norm/output 负担，因此尽量不给它增加额外层。

    返回：
        (start_layer, end_layer)
        表示当前 pp_rank 负责区间 [start_layer, end_layer)
    """

    # 读取手动指定的分层方案环境变量
    partition_list_str = envs.VLLM_PP_LAYER_PARTITION

    # --------------------------------------------------
    # 情况 1：用户手动指定了每个分区的层数
    # --------------------------------------------------
    if partition_list_str is not None:
        try:
            # 例如 "4,4,5,3" -> [4, 4, 5, 3]
            partitions = [int(layer) for layer in partition_list_str.split(",")]
        except ValueError as err:
            raise ValueError(
                "Invalid partition string: {}".format(partition_list_str)
            ) from err

        # 分区数必须等于 pp_size
        if len(partitions) != pp_size:
            raise ValueError(f"{len(partitions)=} does not match {pp_size=}.")

        # 分区层数总和必须等于总层数
        if sum(partitions) != num_hidden_layers:
            raise ValueError(f"{sum(partitions)=} does not match {num_hidden_layers=}.")

    # --------------------------------------------------
    # 情况 2：自动切分
    # --------------------------------------------------
    else:
        # 每个分区先平均分到的层数
        layers_per_partition = num_hidden_layers // pp_size

        # 初始化分区表
        partitions = [layers_per_partition for _ in range(pp_size)]

        # 如果有剩余层，则把剩余层分配给部分分区
        if remaining_layers := num_hidden_layers % pp_size:
            # 从倒数第二个分区开始往前加
            # 不优先给最后一个分区，避免最后一个分区负担过重
            for i in range(2, remaining_layers + 2):
                partitions[-i] += 1

            logger.info(
                "Hidden layers were unevenly partitioned: [%s]. "
                "This can be manually overridden using the "
                "VLLM_PP_LAYER_PARTITION environment variable",
                ",".join(str(p) for p in partitions),
            )

    # 当前 rank 的起始层号 = 前面所有分区层数之和
    start_layer = sum(partitions[:pp_rank])

    # 当前 rank 的结束层号 = 起始层号 + 当前分区层数
    end_layer = start_layer + partitions[pp_rank]

    return (start_layer, end_layer)


@dataclasses.dataclass
class StatelessProcessGroup:
    """A dataclass to hold a metadata store, and the rank, world_size of the
    group. Only use it to communicate metadata between processes.
    For data-plane communication, create NCCL-related objects.
    """

    rank: int
    world_size: int
    store: torch._C._distributed_c10d.Store

    # stores a reference to the socket so that the file descriptor stays alive
    socket: socket.socket | None

    data_expiration_seconds: int = 3600  # 1 hour

    # dst rank -> counter
    send_dst_counter: dict[int, int] = dataclasses.field(default_factory=dict)
    # src rank -> counter
    recv_src_counter: dict[int, int] = dataclasses.field(default_factory=dict)
    broadcast_send_counter: int = 0
    broadcast_recv_src_counter: dict[int, int] = dataclasses.field(default_factory=dict)

    # A deque to store the data entries, with key and timestamp.
    entries: deque[tuple[str, float]] = dataclasses.field(default_factory=deque)

    def __post_init__(self):
        assert self.rank < self.world_size
        self.send_dst_counter = {i: 0 for i in range(self.world_size)}
        self.recv_src_counter = {i: 0 for i in range(self.world_size)}
        self.broadcast_recv_src_counter = {i: 0 for i in range(self.world_size)}

    def send_obj(self, obj: Any, dst: int):
        """Send an object to a destination rank."""
        self.expire_data()
        key = f"send_to/{dst}/{self.send_dst_counter[dst]}"
        self.store.set(key, pickle.dumps(obj))
        self.send_dst_counter[dst] += 1
        self.entries.append((key, time.time()))

    def expire_data(self):
        """Expire data that is older than `data_expiration_seconds` seconds."""
        while self.entries:
            # check the oldest entry
            key, timestamp = self.entries[0]
            if time.time() - timestamp > self.data_expiration_seconds:
                self.store.delete_key(key)
                self.entries.popleft()
            else:
                break

    def recv_obj(self, src: int) -> Any:
        """Receive an object from a source rank."""
        obj = pickle.loads(
            self.store.get(f"send_to/{self.rank}/{self.recv_src_counter[src]}")
        )
        self.recv_src_counter[src] += 1
        return obj

    def broadcast_obj(self, obj: Any | None, src: int) -> Any:
        """Broadcast an object from a source rank to all other ranks.
        It does not clean up after all ranks have received the object.
        Use it for limited times, e.g., for initialization.
        """
        if self.rank == src:
            self.expire_data()
            key = f"broadcast_from/{src}/{self.broadcast_send_counter}"
            self.store.set(key, pickle.dumps(obj))
            self.broadcast_send_counter += 1
            self.entries.append((key, time.time()))
            return obj
        else:
            key = f"broadcast_from/{src}/{self.broadcast_recv_src_counter[src]}"
            recv_obj = pickle.loads(self.store.get(key))
            self.broadcast_recv_src_counter[src] += 1
            return recv_obj

    def all_gather_obj(self, obj: Any) -> list[Any]:
        """All gather an object from all ranks."""
        gathered_objs = []
        for i in range(self.world_size):
            if i == self.rank:
                gathered_objs.append(obj)
                self.broadcast_obj(obj, src=self.rank)
            else:
                recv_obj = self.broadcast_obj(None, src=i)
                gathered_objs.append(recv_obj)
        return gathered_objs

    def broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        """Broadcast a tensor from source rank to all other ranks."""
        if self.rank == src:
            tensor_bytes = pickle.dumps(tensor)
            self.expire_data()
            key = f"broadcast_tensor/{src}/{self.broadcast_send_counter}"
            self.store.set(key, tensor_bytes)
            self.broadcast_send_counter += 1
            self.entries.append((key, time.time()))
            return tensor
        else:
            key = f"broadcast_tensor/{src}/{self.broadcast_recv_src_counter[src]}"
            tensor = pickle.loads(self.store.get(key))
            self.broadcast_recv_src_counter[src] += 1
            return tensor

    def send(self, tensor: torch.Tensor, dst: int):
        """Send a tensor to a destination rank."""
        self.expire_data()
        key = f"send_tensor/{dst}/{self.send_dst_counter[dst]}"
        self.store.set(key, pickle.dumps(tensor))
        self.send_dst_counter[dst] += 1
        self.entries.append((key, time.time()))

    def recv(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        """Receive a tensor from a source rank."""
        key = f"send_tensor/{self.rank}/{self.recv_src_counter[src]}"
        received = pickle.loads(self.store.get(key))
        self.recv_src_counter[src] += 1
        tensor.copy_(received)
        return tensor

    def all_reduce(
            self, tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM
    ) -> torch.Tensor:
        """All-reduce a tensor across all ranks."""
        tensors = self.all_gather_obj(tensor)
        result = tensors[0].clone()
        for t in tensors[1:]:
            if op == torch.distributed.ReduceOp.SUM:
                result.add_(t)
            elif op == torch.distributed.ReduceOp.PRODUCT:
                result.mul_(t)
            elif op == torch.distributed.ReduceOp.MAX:
                result = torch.maximum(result, t)
            elif op == torch.distributed.ReduceOp.MIN:
                result = torch.minimum(result, t)
        return result

    def barrier(self, timeout: float = 30.0):
        """A robust barrier to synchronize all ranks.


        Uses a multi-phase approach to ensure all processes reach the barrier
        before proceeding:

        1. Each process signals it has reached the barrier

        2. Each process signals that it has confirmed the arrival of all other
        ranks.

        3. Rank 0 waits for all other ranks to signal their departure to ensure
        that all ranks have departed the barrier first.

        Args:
            timeout: Maximum time in seconds to wait for each phase (in seconds)


        Raises:
            RuntimeError: If coordination fails or times out
        """
        # Generate a barrier ID that is globally unique
        try:
            if self.rank == 0:
                barrier_id = f"barrier_{uuid.uuid4()}"
                self.broadcast_obj(barrier_id, src=0)
            else:
                barrier_id = self.broadcast_obj(None, src=0)
        except Exception as e:
            raise RuntimeError("Failed to broadcast barrier_id") from e

        # Phase 1: Signal arrival at barrier
        # Wait for all processes to arrive
        # We need all ranks to confirm the arrival of all other ranks.
        # This is the key synchronization point.
        arrival_key = f"arrival_{barrier_id}_{self.rank}"
        try:
            self.store.set(arrival_key, b"1")
        except Exception as e:
            raise RuntimeError("Failed to signal barrier arrival") from e

        start_time = time.time()
        processes_arrived: set[int] = set()

        while len(processes_arrived) < self.world_size:
            # Check for timeout
            cur_time = time.time()
            if cur_time - start_time > timeout:
                raise RuntimeError(f"Barrier timed out after {timeout:.2f} seconds")

            # Check for each process
            for i in range(self.world_size):
                if i in processes_arrived:
                    continue

                key = f"arrival_{barrier_id}_{i}"
                try:
                    # Try to get the key - if it exists, we'll get a value
                    # If it doesn't exist, it will throw an exception
                    self.store.get(key)
                    processes_arrived.add(i)
                except KeyError:
                    # Key doesn't exist yet
                    pass
                except Exception as check_e:
                    logger.debug("Error checking key existence: %s", check_e)
                    sched_yield()

            # Short sleep to avoid tight polling
            if len(processes_arrived) < self.world_size:
                sched_yield()

        # Phase 2: Signal departure from barrier
        # We only care to block at this stage in rank 0, which runs the
        # server side of the TCPStore. We want to make sure that all
        # clients have departed the barrier before rank 0 in case the
        # next thing after the barrier is a shutdown, including tearing
        # down the TCPStore. Other ranks can exit the barrier immediately
        # after signaling their departure.
        departure_key = f"departure_{barrier_id}_{self.rank}"
        try:
            self.store.set(departure_key, b"1")
        except Exception as e:
            raise RuntimeError("Failed to signal barrier departure") from e

        if self.rank != 0:
            return

        # Make rank 0 wait for all processes to signal departure
        start_time = time.time()
        processes_departed: set[int] = set()

        while len(processes_departed) < self.world_size:
            # Check for timeout
            if time.time() - start_time > timeout:
                raise RuntimeError(
                    f"Barrier departure timed out after {timeout:.2f} seconds"
                )

            # Check for each process
            for i in range(self.world_size):
                if i in processes_departed:
                    continue

                key = f"departure_{barrier_id}_{i}"
                try:
                    # Try to get the key - if it exists, we'll get a value
                    # If it doesn't exist, it will throw an exception
                    self.store.get(key)
                    processes_departed.add(i)
                except KeyError:
                    # Key doesn't exist yet
                    pass
                except Exception as check_e:
                    logger.debug("Error checking key existence: %s", check_e)
                    sched_yield()

            # Short sleep to avoid tight polling
            if len(processes_departed) < self.world_size:
                sched_yield()

        # Clean up keys to avoid leaking memory in the store
        for i in range(self.world_size):
            try:
                self.store.delete_key(f"arrival_{barrier_id}_{i}")
            except Exception:
                logger.debug("Error deleting key: %s", f"arrival_{barrier_id}_{i}")

            try:
                self.store.delete_key(f"departure_{barrier_id}_{i}")
            except Exception:
                logger.debug("Error deleting key: %s", f"departure_{barrier_id}_{i}")

    @staticmethod
    def create(
            host: str,
            port: int,
            rank: int,
            world_size: int,
            data_expiration_seconds: int = 3600,
            store_timeout: int = 300,
    ) -> "StatelessProcessGroup":
        """
        创建一个独立于 torch.distributed 全局 WORLD 的轻量级 stateless 进程组。

        这个对象不负责高性能数据面 collective，而是依赖 TCPStore 做：
        - 对象广播
        - 元数据同步
        - 简单的 send / recv
        - barrier

        它的核心目标是：在不污染已有全局进程组状态的前提下，
        临时再拉起一条额外的控制面通信通道。
        """  # noqa
        # -------------------- 先确定谁来充当 TCPStore server --------------------
        # 约定 rank 0 负责监听端口并创建底层 TCPStore server。
        launch_server = rank == 0
        if launch_server:
            # 只监听指定 host，而不是监听 0.0.0.0，避免暴露到不必要的网卡。
            listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # 允许端口快速复用，减轻进程重启后的 TIME_WAIT 影响。
            listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # 绑定到调用方指定的 host:port。
            listen_socket.bind((host, port))
            # 开始监听连接请求。
            listen_socket.listen()
            # 取出监听 socket 的 fd，后面交给 TCPStore 复用。
            listen_fd = listen_socket.fileno()
        else:
            # 非 server rank 不需要本地监听 socket。
            listen_socket = None
            # 对应地也没有可传给 TCPStore 的 listen fd。
            listen_fd = None

        # -------------------- 基于 TCPStore 构造 stateless 元数据通道 --------------------
        store = TCPStore(
            # TCPStore server/client 连接的主机地址。
            host_name=host,
            # TCPStore 监听或连接的端口。
            port=port,
            # 这个 store 期望参与的总 rank 数。
            world_size=world_size,
            # 只有 rank 0 会真正拉起 server，其余 ranks 走 client 模式接入。
            is_master=launch_server,
            # store 层面的超时时间。
            timeout=timedelta(seconds=store_timeout),
            # 当前实现先禁用 libuv 路径，规避上游兼容性问题。
            use_libuv=False,  # for now: github.com/pytorch/pytorch/pull/150215
            # server 端把已经 bind 好的 socket fd 交给 TCPStore 使用。
            master_listen_fd=listen_fd,
        )

        # -------------------- 返回包装后的 StatelessProcessGroup --------------------
        return StatelessProcessGroup(
            # 当前 rank 在这个 stateless 组里的组内 rank。
            rank=rank,
            # 这个 stateless 组的 world size。
            world_size=world_size,
            # 底层共享的 TCPStore，用于控制面通信。
            store=store,
            # 保留 socket 引用，防止监听 fd 被 Python 提前回收。
            socket=listen_socket,
            # 过期数据保留时间，超时后旧 key 会被清理。
            data_expiration_seconds=data_expiration_seconds,
        )


def init_gloo_process_group(
        prefix_store: PrefixStore,
        group_rank: int,
        group_size: int,
        timeout: timedelta,
) -> ProcessGroup:
    """
    以 stateless 方式初始化一个基于 gloo 的 ProcessGroup。

    这里不走 `torch.distributed.init_process_group` 的全局注册路径，
    而是直接手工拼一个 ProcessGroup 实例，并把 Gloo backend 挂进去。
    """
    # 某些 torch 版本在创建 ProcessGroupGloo 时会打印噪声日志，这里统一屏蔽。
    with suppress_stdout():
        # -------------------- 先创建一个通用的 ProcessGroup 外壳 --------------------
        pg = ProcessGroup(
            # PrefixStore 用来给这个组的 rendezvous key 加命名前缀，避免与其它组冲突。
            prefix_store,
            # 当前 rank 在这个组内的组内 rank。
            group_rank,
            # 这个组的总成员数。
            group_size,
        )
        # 延迟导入 Gloo backend 实现类。
        from torch.distributed.distributed_c10d import ProcessGroupGloo

        # -------------------- 再创建实际的 Gloo backend --------------------
        backend_class = ProcessGroupGloo(
            # Gloo backend 复用同一份 PrefixStore 做 rendezvous。
            prefix_store, group_rank, group_size, timeout=timeout
        )
        # 声明这个 backend 的类型是 GLOO。
        backend_type = ProcessGroup.BackendType.GLOO
        # Gloo 是 CPU backend，因此设备类型固定填 cpu。
        device = torch.device("cpu")
        # 把这个 backend 注册为当前 ProcessGroup 的默认 backend。
        pg._set_default_backend(backend_type)
        # 初始化组内 sequence number，保证 collective 调用顺序一致。
        backend_class._set_sequence_number_for_group()

        # 把 Gloo backend 挂到这个 ProcessGroup 外壳上。
        pg._register_backend(device, backend_type, backend_class)
    # 返回已经组装完成的 ProcessGroup。
    return pg


def stateless_init_torch_distributed_process_group(
        host: str,
        port: int,
        rank: int,
        world_size: int,
        backend: str,
        group_name: str | None = None,
        return_store: bool = False,
) -> ProcessGroup | tuple[ProcessGroup, Store]:
    """
    在不污染 torch.distributed 全局状态的前提下，按给定 host/port/rank/world_size
    构造一个额外的 ProcessGroup。

    这个函数的目标不是替代全局 WORLD，而是给某些临时/额外拓扑
    另外拉起一条独立的 torch ProcessGroup 通道。

    注意：
    - 它适合 `all_reduce` 一类只依赖组内 rank 的 collective。
    - 某些依赖全局 rank 语义的操作（例如标准 `broadcast`）不适合直接依赖它，
      因此 stateless 体系另外配了一条 `TCPStore` 控制面通道。
    """
    # -------------------- 先把 host/port 转成 rendezvous URI 并解析 backend --------------------
    # 组初始化统一通过 tcp://host:port 形式的 rendezvous 地址进行。
    init_method = get_tcp_uri(host, port)
    # 把字符串 backend 规范化成 torch.distributed.Backend 对象。
    backend = Backend(backend)
    # 根据 backend 类型获取默认超时时间。
    timeout = _get_default_timeout(backend)

    # -------------------- 通过 rendezvous 拿到底层 store / rank / world_size --------------------
    store, rank, world_size = next(
        # rendezvous 会负责建立底层 store，并回传这个组真正使用的 rank/world_size。
        rendezvous(init_method, rank, world_size, timeout=timeout)
    )
    # 给返回的 store 也设置同样的超时时间。
    store.set_timeout(timeout)

    # 这里 group_rank/group_size 与 rendezvous 返回的 rank/world_size 一致，
    # 单独起名只是为了强调它们是“这个 stateless 组内部”的编号。
    group_rank = rank
    group_size = world_size

    # -------------------- 为当前组包一层 PrefixStore，隔离 store key 空间 --------------------
    # 避免同一个底层 store 被多个系统/多个组共享时互相覆盖 key。
    prefix_store = PrefixStore(init_method, store)

    # -------------------- 根据 backend 类型创建具体的 torch ProcessGroup --------------------
    if backend == "gloo":
        # CPU / gloo 路径走本文件内的手工 Gloo PG 初始化。
        pg = init_gloo_process_group(
            prefix_store=prefix_store,
            group_rank=group_rank,
            group_size=group_size,
            timeout=timeout,
        )
    else:
        # 设备 backend（例如 nccl）交给当前平台实现去创建。
        from cfie.platforms import current_platform

        pg = current_platform.stateless_init_device_torch_dist_pg(
            # 设备侧 backend 名称。
            backend=backend,
            # 带前缀的 rendezvous store。
            prefix_store=prefix_store,
            # 当前 rank 在组内的组内 rank。
            group_rank=group_rank,
            # 这个组的 world size。
            group_size=group_size,
            # backend 初始化超时配置。
            timeout=timeout,
        )

    # -------------------- 如有需要，把这个组注册进 torch 的命名组表 --------------------
    if group_name is not None:
        # 延迟导入 torch 的内部组注册函数。
        from torch._C._distributed_c10d import _register_process_group

        # 给这个 ProcessGroup 设置稳定名字，便于后续按名调试或销毁。
        pg._set_group_name(group_name)
        # 注册到 torch 内部命名组表。
        _register_process_group(group_name, pg)

    # -------------------- 按调用方要求返回 ProcessGroup 或 (ProcessGroup, Store) --------------------
    if return_store:
        # 某些调用方还需要直接访问底层 store，因此一并返回。
        return pg, store
    else:
        # 默认只返回构造好的 ProcessGroup。
        return pg


def stateless_destroy_torch_distributed_process_group(pg: ProcessGroup) -> None:
    """
    销毁由 `stateless_init_torch_distributed_process_group()` 创建的 ProcessGroup。
    """
    # 先关闭这个 ProcessGroup 对应的底层 backend / 通信资源。
    pg.shutdown()
    # 再从 torch 内部命名组注册表里移除它，避免悬挂引用。
    _unregister_process_group(pg.group_name)


def get_worker_rank_suffix(global_rank: int | None = None) -> str:
    """Generate a descriptive rank suffix for worker identification.

    Returns a string like 'dp0_pp0_tp0_dcp0_ep0_rank0' including all
    parallel dimensions: DP, PP, TP, DCP, EP.

    Args:
        global_rank: Optional global rank to append. If not provided,
                     only parallel dimension ranks are included.

    Returns:
        A string suffix identifying the worker's position in the
        distributed topology.
    """
    from cfie.distributed.parallel_state import (
        get_dcp_group,
        get_dp_group,
        get_ep_group,
        get_pp_group,
        get_tp_group,
    )

    try:
        dp_rank = get_dp_group().rank_in_group
        pp_rank = get_pp_group().rank_in_group
        tp_rank = get_tp_group().rank_in_group
        dcp_rank = get_dcp_group().rank_in_group
        ep_rank = get_ep_group().rank_in_group

        suffix = f"dp{dp_rank}_pp{pp_rank}_tp{tp_rank}_dcp{dcp_rank}_ep{ep_rank}"
        if global_rank is not None:
            suffix = f"{suffix}_rank{global_rank}"
        return suffix
    except Exception:
        # Fallback if parallel state not initialized
        if global_rank is not None:
            return f"rank{global_rank}"
        return ""
