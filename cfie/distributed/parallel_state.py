# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""vLLM distributed state.
It takes over the control of the distributed environment from PyTorch.
The typical workflow is:

- call `init_distributed_environment` to initialize the distributed environment.
- call `initialize_model_parallel` or `ensure_model_parallel_initialized` to
 initialize the model parallel groups.

- any code dealing with the distributed stuff

- call `destroy_model_parallel` to destroy the model parallel groups.
- call `destroy_distributed_environment` to destroy the distributed environment.

If you only need to use the distributed environment without model/pipeline
 parallelism, you can skip the model parallel initialization and destruction
 steps.
"""

import contextlib
import gc
import os
import pickle
import weakref
from collections import namedtuple
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Protocol
from unittest.mock import patch

import torch
import torch.distributed
import torch.distributed._functional_collectives as funcol
import torch.distributed._symmetric_memory
from torch.distributed import Backend, ProcessGroup

import cfie.envs as envs
from cfie.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from cfie.distributed.utils import StatelessProcessGroup
from cfie.logger import init_logger
from cfie.utils.import_utils import resolve_obj_by_qualname
from cfie.utils.network_utils import get_distributed_init_method
from cfie.utils.system_utils import suppress_stdout
from cfie.utils.torch_utils import (
    direct_register_custom_op,
)

if TYPE_CHECKING:
    from cfie.distributed.stateless_coordinator import StatelessGroupCoordinator


@dataclass
class GraphCaptureContext:
    stream: torch.cuda.Stream


TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])


class Handle(Protocol):
    """Minimal async work handle used by P2P send/recv methods."""

    def is_completed(self) -> bool: ...

    def wait(self) -> None: ...


def _split_tensor_dict(
        tensor_dict: dict[str, torch.Tensor | Any],
) -> tuple[list[tuple[str, Any]], list[torch.Tensor]]:
    """Split the tensor dictionary into two parts:
    1. A list of (key, value) pairs. If the value is a tensor, it is replaced
         by its metadata.
    2. A list of tensors.
    """
    metadata_list: list[tuple[str, Any]] = []
    tensor_list: list[torch.Tensor] = []
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            # Note: we cannot use `value.device` here,
            # because it contains not only the device type but also the device
            # index (e.g. "cuda:0"). We only need the device type.
            # receiving side will set the device index.
            device = value.device.type
            metadata_list.append(
                (key, TensorMetadata(device, value.dtype, value.size()))
            )
            tensor_list.append(value)
        else:
            metadata_list.append((key, value))
    return metadata_list, tensor_list


_group_name_counter: dict[str, int] = {}


def _get_unique_name(name: str) -> str:
    """
    为给定组名前缀生成一个当前进程内唯一的名字。

    示例：
        _get_unique_name("tp") -> "tp:0"
        _get_unique_name("tp") -> "tp:1"
    """
    # -------------------- 先确保这个前缀在计数表里有初始计数 --------------------
    # 第一次见到某个组名前缀时，从 0 开始编号。
    if name not in _group_name_counter:
        _group_name_counter[name] = 0
    # -------------------- 用“前缀:当前计数”拼出唯一名字 --------------------
    # 例如 name="tp" 且当前计数为 3 时，生成 "tp:3"。
    newname = f"{name}:{_group_name_counter[name]}"
    # -------------------- 自增计数，供下一次同名前缀调用使用 --------------------
    # 这样下一次再请求同一个前缀时，就会得到新的序号。
    _group_name_counter[name] += 1
    # 返回本次生成的唯一名字。
    return newname


_groups: dict[str, Callable[[], "GroupCoordinator | None"]] = {}


def _register_group(group: "GroupCoordinator") -> None:
    _groups[group.unique_name] = weakref.ref(group)


def all_reduce(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._all_reduce_out_place(tensor)


def all_reduce_fake(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    return torch.empty_like(tensor)


def reduce_scatter(
        tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._reduce_scatter_out_place(tensor, dim)


def reduce_scatter_fake(
        tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    new_shape = list(tensor.shape)
    new_shape[dim] = tensor.shape[dim] // world_size
    return torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)


def all_gather(
        tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._all_gather_out_place(tensor, dim)


def all_gather_fake(
        tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    new_shape = list(tensor.shape)
    new_shape[dim] = tensor.shape[dim] * world_size
    return torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)


def patched_fused_scaled_matmul_reduce_scatter_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        reduce_op: str,
        orig_scatter_dim: int,
        scatter_dim_after_maybe_reshape: int,
        group_name: str,
        output_shape: list[int],
        bias: torch.Tensor | None = None,
        result_scale: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        use_fast_accum: bool = False,
) -> torch.Tensor:
    # Copied from
    # https://github.com/pytorch/pytorch/blob/50c338c2da905062449e4d9ac807832d1b5cd90e/torch/distributed/_symmetric_memory/__init__.py#L1189
    if A_scale.numel() > 1:
        if A_scale.shape[:-1] != A.shape[:-1]:
            raise ValueError(
                "For row-wise scaling, the leading dims of A_scale "
                "must match the leading dims of A "
                f"(A shape: {A.shape}, A_scale shape: {A_scale.shape})"
            )
        A_scale = A_scale.flatten(0, -2).contiguous()
    elif A_scale.numel() != 1:
        raise ValueError(
            "Invalid A_scale shape "
            f"(A shape: {A.shape}, A_scale shape: {A_scale.shape})"
        )

    C = torch._scaled_mm(
        A.flatten(0, -2).contiguous(),
        B,
        A_scale,
        B_scale,
        bias,
        result_scale,
        out_dtype,
        use_fast_accum,
    )
    C = C.view(*output_shape[:-1], B.shape[1])
    res = funcol.reduce_scatter_tensor(
        C,
        reduce_op,
        orig_scatter_dim,  # need original scatter dim for 3D+ output tensor here
        group_name,
    )
    res = funcol.wait_tensor(res)
    return res


def patched_fused_scaled_matmul_reduce_scatter(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        reduce_op: str,
        orig_scatter_dim: int,
        scatter_dim_after_maybe_reshape: int,
        group_name: str,
        output_shape: list[int],
        bias: torch.Tensor | None = None,
        result_scale: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        use_fast_accum: bool = False,
) -> torch.Tensor:
    return torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
        A,
        B,
        A_scale,
        B_scale,
        reduce_op,
        orig_scatter_dim,
        scatter_dim_after_maybe_reshape,
        group_name,
        output_shape,
        bias,
        result_scale,
        out_dtype,
        use_fast_accum,
    )


direct_register_custom_op(
    op_name="all_reduce",
    op_func=all_reduce,
    fake_impl=all_reduce_fake,
)

direct_register_custom_op(
    op_name="reduce_scatter",
    op_func=reduce_scatter,
    fake_impl=reduce_scatter_fake,
)

direct_register_custom_op(
    op_name="all_gather",
    op_func=all_gather,
    fake_impl=all_gather_fake,
)

# TODO: Remove this once the pytorch fix
# (https://github.com/pytorch/pytorch/pull/165086) gets released,
# in either 2.9.1 or 2.10
direct_register_custom_op(
    op_name="patched_fused_scaled_matmul_reduce_scatter",
    op_func=patched_fused_scaled_matmul_reduce_scatter,
    fake_impl=patched_fused_scaled_matmul_reduce_scatter_fake,
)


class GroupCoordinator:
    """
    一组进程对应的 PyTorch ProcessGroup 封装。
    PyTorch ProcessGroup 绑定到单一通信后端，例如 NCCL、Gloo、MPI。
    GroupCoordinator 负责组内的统一通信编排。
    它同时管理 CPU 通信与设备侧通信。
    """

    # 当前进程的全局 rank。
    rank: int
    # 当前 group 内全部成员的全局 rank 列表。
    ranks: list[int]
    # 当前 group 的总规模。
    world_size: int
    # `local_rank` 用于节点内设备分配。
    # `rank_in_group` 表示当前进程在这个 group 内的顺序编号。
    # 例如一个跨两台机器、总规模为 4 的 group：
    # Process | Node | Rank | Local Rank | Rank in Group
    #   0     |   0  |  0   |     0      |       0
    #   1     |   0  |  1   |     1      |       1
    #   2     |   1  |  2   |     0      |       2
    #   3     |   1  |  3   |     1      |       3
    # 当前进程在本地节点内的 rank。
    local_rank: int
    # 当前进程在本 group 内的 rank。
    rank_in_group: int
    # 用于 CPU 通信的 process group。
    cpu_group: ProcessGroup
    # 用于设备侧通信的 process group。
    device_group: ProcessGroup
    # 设备通信器。
    # 仅在 `use_device_communicator=True` 时初始化。
    device_communicator: DeviceCommunicatorBase | None
    # 基于共享内存的消息广播器。
    mq_broadcaster: Any | None

    def __init__(
            self,
            group_ranks: list[list[int]],
            local_rank: int,
            torch_distributed_backend: str | Backend,
            use_device_communicator: bool,  # 是否启用设备通信器。
            use_message_queue_broadcaster: bool = False,
            group_name: str | None = None,
    ):
        # 若调用方未显式命名，则给当前 group 分配匿名名称。
        group_name = group_name or "anonymous"
        # 为当前 group 生成唯一名，避免不同 group 间命名冲突。
        self.unique_name = _get_unique_name(group_name)
        # 把当前 coordinator 注册到全局表，便于后续查找和释放。
        _register_group(self)
        # 记录当前 group 期望使用的后端，供单 rank 本地模式复用。
        self.backend = str(torch_distributed_backend)

        from cfie.platforms import current_platform

        # Windows 单卡本地模式下不依赖 torch.distributed，直接构造一个
        # 只服务当前进程的轻量 coordinator。
        if not torch.distributed.is_initialized():
            if len(group_ranks) != 1 or len(group_ranks[0]) != 1:
                raise RuntimeError(
                    "torch.distributed is not initialized, but group_ranks "
                    "does not describe a single-rank local group"
                )
            self.rank = int(group_ranks[0][0])
            self.local_rank = local_rank
            self.ranks = [self.rank]
            self.world_size = 1
            self.rank_in_group = 0
            self.cpu_group = None
            self.device_group = None
            if current_platform.is_cuda_alike():
                self.device = torch.device(f"cuda:{local_rank}")
            elif current_platform.is_xpu():
                self.device = torch.device(f"xpu:{local_rank}")
            elif current_platform.is_out_of_tree():
                self.device = torch.device(
                    f"{current_platform.device_name}:{local_rank}"
                )
            else:
                self.device = torch.device("cpu")
            self.use_device_communicator = False
            self.device_communicator = None
            self.mq_broadcaster = None
            self.use_custom_op_call = False
            self.use_cpu_custom_send_recv = False
            return

        # 读取当前进程在全局分布式环境中的 rank。
        self.rank = torch.distributed.get_rank()
        # 保存当前进程的本地 rank，后续用于设备绑定。
        self.local_rank = local_rank

        # 暂存当前进程所属的设备侧 / CPU 侧 process group。
        self_device_group = None
        self_cpu_group = None

        # 为传入的每个 rank 子集分别创建一组通信 group。
        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend
            )
            # 额外创建一个 `gloo` CPU group，供 CPU 侧直接协调通信。
            with suppress_stdout():
                cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            # 只有当前 rank 落在这组 ranks 内时，才把它记录为自身所属 group。
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self_device_group = device_group
                self_cpu_group = cpu_group

        # 走到这里必须已经为当前 rank 找到所属 group。
        assert self_cpu_group is not None
        assert self_device_group is not None

        # 固化当前进程所属的 CPU / 设备通信 group。
        self.cpu_group = self_cpu_group
        self.device_group = self_device_group

        # 根据当前平台类型选择本进程应绑定的逻辑设备。
        if current_platform.is_cuda_alike():
            self.device = torch.device(f"cuda:{local_rank}")
        elif current_platform.is_xpu():
            self.device = torch.device(f"xpu:{local_rank}")
        elif current_platform.is_out_of_tree():
            self.device = torch.device(f"{current_platform.device_name}:{local_rank}")
        else:
            self.device = torch.device("cpu")

        # 记录是否启用设备通信器。
        self.use_device_communicator = use_device_communicator
        # 默认先不创建设备通信器。
        self.device_communicator = None
        # 只有多卡 / 多进程场景下，设备通信器才有实际意义。
        if use_device_communicator and self.world_size > 1:
            device_comm_cls = resolve_obj_by_qualname(
                current_platform.get_device_communicator_cls()
            )
            # 根据平台提供的通信器实现创建设备通信封装。
            self.device_communicator = device_comm_cls(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
            )

        from cfie.distributed.device_communicators.shm_broadcast import MessageQueue

        # 默认先不创建共享内存消息广播器。
        self.mq_broadcaster: MessageQueue | None = None
        # 仅在显式开启且 group 规模大于 1 时初始化共享内存广播器。
        if use_message_queue_broadcaster and self.world_size > 1:
            self.mq_broadcaster = MessageQueue.create_from_process_group(
                self.cpu_group, 1 << 22, 6
            )

        # TPU 当前仍需走兼容分支，待上游补齐后可移除此特判。
        self.use_custom_op_call = (
                current_platform.is_tpu() or current_platform.use_custom_op_collectives()
        )

        # CPU 平台若提供共享内存管理器，则允许走自定义 send/recv 实现。
        self.use_cpu_custom_send_recv = current_platform.is_cpu() and hasattr(
            torch.ops._C, "init_shm_manager"
        )

    def create_mq_broadcaster(
            self, writer_rank=0, external_writer_handle=None, blocking=True
    ):
        from cfie.distributed.device_communicators.shm_broadcast import MessageQueue

        return MessageQueue.create_from_process_group(
            self.cpu_group,
            1 << 22,
            6,
            writer_rank=writer_rank,
            external_writer_handle=external_writer_handle,
            blocking=blocking,
        )

    def create_single_reader_mq_broadcasters(
            self, reader_rank_in_group=0, blocking=False
    ):
        from cfie.distributed.device_communicators.shm_broadcast import MessageQueue

        return MessageQueue.create_from_process_group_single_reader(
            self.cpu_group,
            1 << 22,
            6,
            reader_rank=self.ranks[reader_rank_in_group],
            blocking=blocking,
        )

    @property
    def first_rank(self):
        """Return the global rank of the first process in the group"""
        return self.ranks[0]

    @property
    def last_rank(self):
        """Return the global rank of the last process in the group"""
        return self.ranks[-1]

    @property
    def is_first_rank(self):
        """Return whether the caller is the first process in the group"""
        return self.rank == self.first_rank

    @property
    def is_last_rank(self):
        """Return whether the caller is the last process in the group"""
        return self.rank == self.last_rank

    @property
    def next_rank(self):
        """Return the global rank of the process that follows the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group + 1) % world_size]

    @property
    def prev_rank(self):
        """Return the global rank of the process that precedes the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group - 1) % world_size]

    @contextmanager
    def graph_capture(self, graph_capture_context: GraphCaptureContext | None = None):
        if graph_capture_context is None:
            stream = torch.cuda.Stream()
            graph_capture_context = GraphCaptureContext(stream)
        else:
            stream = graph_capture_context.stream

        # only cuda uses this function,
        # so we don't abstract it into the base class
        maybe_ca_context = nullcontext()
        from cfie.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator,
        )

        if self.device_communicator is not None:
            assert isinstance(self.device_communicator, CudaCommunicator)
            ca_comm = self.device_communicator.ca_comm
            if ca_comm is not None:
                maybe_ca_context = ca_comm.capture()  # type: ignore

        # ensure all initialization operations complete before attempting to
        # capture the graph on another stream
        curr_stream = torch.cuda.current_stream()
        if curr_stream != stream:
            stream.wait_stream(curr_stream)

        with torch.cuda.stream(stream), maybe_ca_context:
            yield graph_capture_context

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """
        User-facing all-reduce function before we actually call the
        all-reduce operation.

        We need this because Dynamo does not support passing an arbitrary
        object (`self` in this case) to a custom op. We need to pass the
         group name as a string, and then look up the group coordinator from
         the group name, dispatch the all-reduce operation to the group
         coordinator.

        In addition, PyTorch custom ops do not support mutation or returning
        a new tensor in the same op. So we always make the all-reduce operation
        out-of-place.
        """
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_

        if self.use_custom_op_call:
            return torch.ops.cfie.all_reduce(input_, group_name=self.unique_name)
        else:
            return self._all_reduce_out_place(input_)

    def _all_reduce_out_place(self, input_: torch.Tensor) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.all_reduce(input_)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )

        if self.use_custom_op_call:
            return torch.ops.cfie.all_gather(
                input_, dim, world_size, group_name=self.unique_name
            )
        else:
            return self._all_gather_out_place(input_, dim)

    def _all_gather_out_place(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.all_gather(input_, dim)

    def all_gatherv(
            self,
            input_: torch.Tensor | list[torch.Tensor],
            dim: int = 0,
            sizes: list[int] | None = None,
    ):
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.all_gatherv(input_, dim, sizes)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )

        if self.use_custom_op_call:
            return torch.ops.cfie.reduce_scatter(
                input_, dim, world_size, group_name=self.unique_name
            )
        else:
            return self._reduce_scatter_out_place(input_, dim)

    def reduce_scatterv(
            self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.reduce_scatterv(input_, dim, sizes)

    def _reduce_scatter_out_place(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.reduce_scatter(input_, dim)

    def gather(
            self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.gather(input_, dst, dim)

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        """Broadcast the input tensor.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        # Broadcast.
        torch.distributed.broadcast(
            input_, src=self.ranks[src], group=self.device_group
        )
        return input_

    def broadcast_object(self, obj: Any | None = None, src: int = 0):
        """Broadcast the input object.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj
        if self.mq_broadcaster is not None:
            assert src == 0, "Message queue broadcaster only supports src=0"
            return self.mq_broadcaster.broadcast_object(obj)
        if self.rank_in_group == src:
            torch.distributed.broadcast_object_list(
                [obj], src=self.ranks[src], group=self.cpu_group
            )
            return obj
        else:
            recv = [None]
            torch.distributed.broadcast_object_list(
                recv, src=self.ranks[src], group=self.cpu_group
            )
            return recv[0]

    def broadcast_object_list(
            self, obj_list: list[Any], src: int = 0, group: ProcessGroup | None = None
    ):
        """Broadcast the input object list.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj_list
        # Broadcast.
        torch.distributed.broadcast_object_list(
            obj_list, src=self.ranks[src], group=self.device_group
        )
        return obj_list

    def send_object(self, obj: Any, dst: int) -> None:
        """Send the input object list to the destination rank."""
        """NOTE: `dst` is the local rank of the destination rank."""

        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        assert dst != self.rank_in_group, (
            "Invalid destination rank. Destination rank is the same "
            "as the current rank."
        )

        # Serialize object to tensor and get the size as well
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)

        size_tensor = torch.tensor(
            [object_tensor.numel()], dtype=torch.long, device="cpu"
        )

        # Send object size

        torch.distributed.send(size_tensor, dst=self.ranks[dst], group=self.cpu_group)

        # Send object
        torch.distributed.send(object_tensor, dst=self.ranks[dst], group=self.cpu_group)

        return None

    def recv_object(self, src: int) -> Any:
        """Receive the input object list from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""

        assert src < self.world_size, f"Invalid src rank ({src})"

        assert src != self.rank_in_group, (
            "Invalid source rank. Source rank is the same as the current rank."
        )

        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")

        # Receive object size
        rank_size = torch.distributed.recv(
            size_tensor, src=self.ranks[src], group=self.cpu_group
        )

        # Tensor to receive serialized objects into.
        object_tensor = torch.empty(  # type: ignore[call-overload]
            size_tensor.item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device="cpu",
        )

        rank_object = torch.distributed.recv(
            object_tensor, src=self.ranks[src], group=self.cpu_group
        )

        assert rank_object == rank_size, (
            "Received object sender rank does not match the size sender rank."
        )

        obj = pickle.loads(object_tensor.numpy().tobytes())

        return obj

    def broadcast_tensor_dict(
            self,
            tensor_dict: dict[str, torch.Tensor | Any] | None = None,
            src: int = 0,
            group: ProcessGroup | None = None,
            metadata_group: ProcessGroup | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        """Broadcast the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict

        group = self.device_group
        metadata_group = self.cpu_group
        assert src < self.world_size, f"Invalid src rank ({src})"

        rank_in_group = self.rank_in_group
        if rank_in_group == src:
            metadata_list: list[tuple[Any, Any]] = []
            assert isinstance(tensor_dict, dict), (
                f"Expecting a dictionary, got {type(tensor_dict)}"
            )
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
            # `metadata_list` lives in CPU memory.
            # `broadcast_object_list` has serialization & deserialization,
            # all happening on CPU. Therefore, we can use the CPU group.
            self.broadcast_object(metadata_list, src=src)
            async_handles = []
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    handle = torch.distributed.broadcast(
                        tensor, src=self.ranks[src], group=metadata_group, async_op=True
                    )
                else:
                    # use group for GPU tensors
                    handle = torch.distributed.broadcast(
                        tensor, src=self.ranks[src], group=group, async_op=True
                    )
                async_handles.append(handle)
            for async_handle in async_handles:
                async_handle.wait()

        else:
            metadata_list = self.broadcast_object(None, src=src)
            tensor_dict = {}
            async_handles = []
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(
                        value.size, dtype=value.dtype, device=value.device
                    )
                    if tensor.numel() == 0:
                        # Skip broadcasting empty tensors.
                        tensor_dict[key] = tensor
                        continue
                    if tensor.is_cpu:
                        # use metadata_group for CPU tensors
                        handle = torch.distributed.broadcast(
                            tensor,
                            src=self.ranks[src],
                            group=metadata_group,
                            async_op=True,
                        )
                    else:
                        # use group for GPU tensors
                        handle = torch.distributed.broadcast(
                            tensor, src=self.ranks[src], group=group, async_op=True
                        )
                    async_handles.append(handle)
                    tensor_dict[key] = tensor
                else:
                    tensor_dict[key] = value
            for async_handle in async_handles:
                async_handle.wait()
        return tensor_dict

    def _should_use_all_gather(
            self,
            key: str,
            numel: int,
            all_gather_group: "GroupCoordinator | None",
            all_gather_tensors: dict[str, bool] | None,
    ) -> bool:
        if all_gather_group is None:
            return False
        use_all_gather = numel % all_gather_group.world_size == 0
        if all_gather_tensors is not None:
            use_all_gather = all_gather_tensors.get(key, use_all_gather)
        return use_all_gather

    def send_tensor_dict(
            self,
            tensor_dict: dict[str, torch.Tensor | Any],
            dst: int | None = None,
            all_gather_group: "GroupCoordinator | None" = None,
            all_gather_tensors: dict[str, bool] | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        """Send the input tensor dictionary.
        NOTE: `dst` is the local rank of the source rank.

        all_gather_group: The group for the all-gather operation. If provided,
            an optimization is enabled where each rank in the group sends a
            slice of a tensor and the receiver reconstructs it using an
            all-gather, which can improve performance. This is typically the
            tensor-parallel group.
        all_gather_tensors: A dictionary to specify which tensors should use
            the all-gather optimization, which is only effective when
            `all_gather_group` is provided. By default, this optimization is
            on for any tensor whose size is divisible by the
            `all_gather_group`'s world size. However, it should be disabled
            for tensors that are not fully replicated across the group (e.g.,
            the residual tensor when sequence parallelism is enabled). This
            dictionary allows overriding the default behavior on a per-tensor
            basis.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict
        handles = self.isend_tensor_dict(
            tensor_dict,
            dst=dst,
            all_gather_group=all_gather_group,
            all_gather_tensors=all_gather_tensors,
        )
        for handle in handles:
            handle.wait()
        return None

    def isend_tensor_dict(
            self,
            tensor_dict: dict[str, torch.Tensor | Any],
            dst: int | None = None,
            all_gather_group: "GroupCoordinator | None" = None,
            all_gather_tensors: dict[str, bool] | None = None,
    ) -> list[Handle]:
        if self.world_size <= 1:
            return []

        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        if self.use_cpu_custom_send_recv:
            if self.device_communicator is None:
                raise ValueError("No device communicator found")
            # custom device communicator path is synchronous
            self.device_communicator.send_tensor_dict(  # type: ignore
                tensor_dict, dst
            )
            return []

        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        group = self.device_group
        metadata_group = self.cpu_group

        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        self.send_object(metadata_list, dst=dst)

        tensor_keys = [k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)]
        assert len(tensor_keys) == len(tensor_list)

        handles: list[Handle] = []
        for key, tensor in zip(tensor_keys, tensor_list):
            if tensor.numel() == 0:
                continue

            if self._should_use_all_gather(
                    key, tensor.numel(), all_gather_group, all_gather_tensors
            ):
                tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

            comm_group = metadata_group if tensor.is_cpu else group
            handle = torch.distributed.isend(
                tensor, dst=self.ranks[dst], group=comm_group
            )
            if tensor.is_cuda:
                tensor.record_stream(torch.cuda.current_stream(tensor.device))
            handles.append(handle)

        return handles

    def recv_tensor_dict(
            self,
            src: int | None = None,
            all_gather_group: "GroupCoordinator | None" = None,
            all_gather_tensors: dict[str, bool] | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        """Recv the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.

        all_gather_group: The group for the all-gather operation. If provided,
            an optimization is enabled where each rank in the group sends a
            slice of a tensor and the receiver reconstructs it using an
            all-gather, which can improve performance. This is typically the
            tensor-parallel group.
        all_gather_tensors: A dictionary to specify which tensors should use
            the all-gather optimization, which is only effective when
            `all_gather_group` is provided. By default, this optimization is
            on for any tensor whose size is divisible by the
            `all_gather_group`'s world size. However, it should be disabled
            for tensors that are not fully replicated across the group (e.g.,
            the residual tensor when sequence parallelism is enabled). This
            dictionary allows overriding the default behavior on a per-tensor
            basis.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None
        tensor_dict, handles, postprocess = self.irecv_tensor_dict(
            src=src,
            all_gather_group=all_gather_group,
            all_gather_tensors=all_gather_tensors,
        )
        for handle in handles:
            handle.wait()
        for fn in postprocess:
            fn()
        return tensor_dict

    def irecv_tensor_dict(
            self,
            src: int | None = None,
            all_gather_group: "GroupCoordinator | None" = None,
            all_gather_tensors: dict[str, bool] | None = None,
    ) -> tuple[
        dict[str, torch.Tensor | Any] | None,
        list[Handle],
        list[Callable[[], None]],
    ]:
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None, [], []

        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        assert src < self.world_size, f"Invalid src rank ({src})"

        if self.use_cpu_custom_send_recv:
            if self.device_communicator is None:
                raise ValueError("No device communicator found")
            # custom device communicator path is synchronous
            sync_tensor_dict = self.device_communicator.recv_tensor_dict(  # type: ignore
                src
            )
            return sync_tensor_dict, [], []

        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        group = self.device_group
        metadata_group = self.cpu_group

        recv_metadata_list = self.recv_object(src=src)
        tensor_dict: dict[str, Any] = {}
        handles: list[Handle] = []
        postprocess: list[Callable[[], None]] = []

        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                full_tensor = torch.empty(
                    value.size, dtype=value.dtype, device=value.device
                )
                if full_tensor.numel() == 0:
                    tensor_dict[key] = full_tensor
                    continue

                if self._should_use_all_gather(
                        key, full_tensor.numel(), all_gather_group, all_gather_tensors
                ):
                    orig_shape = full_tensor.shape
                    slice_tensor = full_tensor.reshape(all_gather_size, -1)[
                        all_gather_rank
                    ]
                    comm_group = metadata_group if slice_tensor.is_cpu else group
                    handle = torch.distributed.irecv(
                        slice_tensor, src=self.ranks[src], group=comm_group
                    )
                    handles.append(handle)

                    def _postprocess(
                            key: str = key,
                            slice_tensor: torch.Tensor = slice_tensor,
                            orig_shape: tuple[int, ...] = tuple(orig_shape),
                            all_gather_group=all_gather_group,
                    ) -> None:
                        assert all_gather_group is not None
                        tensor_dict[key] = all_gather_group.all_gather(
                            slice_tensor, dim=0
                        ).reshape(orig_shape)

                    postprocess.append(_postprocess)
                    tensor_dict[key] = slice_tensor
                else:
                    comm_group = metadata_group if full_tensor.is_cpu else group
                    handle = torch.distributed.irecv(
                        full_tensor, src=self.ranks[src], group=comm_group
                    )
                    handles.append(handle)
                    tensor_dict[key] = full_tensor
            else:
                tensor_dict[key] = value

        return tensor_dict, handles, postprocess

    def barrier(self):
        """Barrier synchronization among the group.
        NOTE: don't use `device_group` here! `barrier` in NCCL is
        terrible because it is internally a broadcast operation with
        secretly created GPU tensors. It is easy to mess up the current
        device. Use the CPU group instead.
        """
        torch.distributed.barrier(group=self.cpu_group)

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        """Sends a tensor to the destination rank in a blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        self.device_communicator.send(tensor, dst)

    def recv(
            self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.recv(size, dtype, src)

    def destroy(self):
        if (
                hasattr(self, "device_group")
                and self.device_group is not None
                and torch.distributed.is_initialized()
        ):
            torch.distributed.destroy_process_group(self.device_group)
            del self.device_group
        if (
                hasattr(self, "cpu_group")
                and self.cpu_group is not None
                and torch.distributed.is_initialized()
        ):
            torch.distributed.destroy_process_group(self.cpu_group)
            del self.cpu_group
        if self.device_communicator is not None:
            self.device_communicator.destroy()
        if self.mq_broadcaster is not None:
            self.mq_broadcaster = None

    def prepare_communication_buffer_for_model(self, model: torch.nn.Module):
        if self.device_communicator is not None:
            self.device_communicator.prepare_communication_buffer_for_model(model)

    def dispatch_router_logits(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            is_sequence_parallel: bool = False,
            extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
            tuple[torch.Tensor, torch.Tensor]
            | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        if self.device_communicator is not None:
            return self.device_communicator.dispatch_router_logits(
                hidden_states,
                router_logits,
                is_sequence_parallel,
                extra_tensors,
            )
        else:
            return hidden_states, router_logits

    def dispatch(
            self,
            hidden_states: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            is_sequence_parallel: bool = False,
            extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
            | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        if self.device_communicator is not None:
            return self.device_communicator.dispatch(
                hidden_states,
                topk_weights,
                topk_ids,
                is_sequence_parallel,
                extra_tensors,
            )
        else:
            return hidden_states, topk_weights, topk_ids

    def combine(
            self, hidden_states, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        if self.device_communicator is not None:
            return self.device_communicator.combine(hidden_states, is_sequence_parallel)
        else:
            return hidden_states


_WORLD: GroupCoordinator | None = None
_INNER_DP_WORLD: GroupCoordinator | None = None
_NODE_COUNT: int | None = None


def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


def get_inner_dp_world_group() -> GroupCoordinator:
    assert _INNER_DP_WORLD is not None, "inner dp world group is not initialized"
    return _INNER_DP_WORLD


def init_world_group(
        ranks: list[int], local_rank: int, backend: str
) -> GroupCoordinator:
    # 用一个只包含“全体 ranks”这一项的列表来描述 world 组。
    return GroupCoordinator(
        # world 组只有一个逻辑 group，成员就是传入的所有全局 rank。
        group_ranks=[ranks],
        # 记录当前进程在本机内的 local rank，供 device 选择等逻辑使用。
        local_rank=local_rank,
        # 指定底层 torch.distributed 使用的通信后端，例如 nccl / gloo。
        torch_distributed_backend=backend,
        # world 组只承担全局协调，不额外创建 device communicator。
        use_device_communicator=False,
        # 给这个协调器打上固定名字，便于日志与调试。
        group_name="world",
    )


def init_model_parallel_group(
        group_ranks: list[list[int]],
        local_rank: int,
        backend: str,
        use_message_queue_broadcaster: bool = False,
        group_name: str | None = None,
        use_device_communicator: bool = True,
) -> GroupCoordinator:
    # 根据给定的 rank 分组定义，创建一个通用的模型并行协调器。
    return GroupCoordinator(
        # 每个子列表就是一个并行组的成员 rank 列表。
        group_ranks=group_ranks,
        # 当前进程在本机内的 local rank，用于绑定本地 GPU 等操作。
        local_rank=local_rank,
        # 底层 torch.distributed 通信后端。
        torch_distributed_backend=backend,
        # 大多数模型并行组都需要额外的 device communicator 加速设备侧 collective。
        use_device_communicator=use_device_communicator,
        # 某些组（如 TP / DCP / inner_dp_world）还会额外启用消息队列广播器。
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        # 组名仅用于标识与调试，不影响分组语义。
        group_name=group_name,
    )


def _init_stateless_group(
        group_ranks: list[list[int]],
        group_name: str,
        group_ports: list[list[int]],
        host: str,
        backend: str,
        use_device_communicator: bool = True,
) -> "StatelessGroupCoordinator":
    """用给定参数创建一个 StatelessGroupCoordinator。"""
    # 延迟导入 stateless coordinator，避免模块初始化阶段形成循环依赖。
    from cfie.distributed.stateless_coordinator import StatelessGroupCoordinator

    # 复用当前已经建立好的 world 组，从中继承 local/global rank 信息。
    world = get_world_group()
    return StatelessGroupCoordinator(
        # 每个子列表定义一个 stateless 通信组的成员 rank。
        group_ranks=group_ranks,
        # stateless 组仍沿用当前 world 的 local rank 作为本地设备索引。
        local_rank=world.local_rank,
        # 指定底层 torch.distributed 通信后端。
        torch_distributed_backend=backend,
        # 是否为该组额外创建 device communicator。
        use_device_communicator=use_device_communicator,
        # 组名用于区分 dp / ep / eplb / world 等不同 stateless 组。
        group_name=group_name,
        # stateless 组统一通过给定 host 提供的 TCP store 建链。
        host=host,
        # 每个逻辑组分配独立端口，避免不同组之间互相冲突。
        group_ports=group_ports,
        # 当前进程在“全局 elastic 世界”中的 rank。
        global_rank=world.rank,
        # 整个 elastic 世界的总 rank 数。
        global_world_size=world.world_size,
    )


def _replace_active_groups(
        *,
        world: GroupCoordinator | None,
        dp: GroupCoordinator | None,
        ep: GroupCoordinator | None,
        eplb: GroupCoordinator | None,
        node_count: int | None,
) -> None:
    """Destroy the current DP/EP/WORLD/EPLB groups and replace them.

    Destruction is collective — all ranks in the old groups must call this
    function together.  Pass all-``None`` to tear down without replacement.
    """
    global _WORLD, _DP, _EP, _EPLB, _NODE_COUNT
    for group in (_DP, _EP, _WORLD, _EPLB):
        if group is not None:
            group.destroy()
    _WORLD = world
    _DP = dp
    _EP = ep
    _EPLB = eplb
    _NODE_COUNT = node_count


_TP: GroupCoordinator | None = None


def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


_DCP: GroupCoordinator | None = None


def get_dcp_group() -> GroupCoordinator:
    assert _DCP is not None, "decode context model parallel group is not initialized"
    return _DCP


# kept for backward compatibility
get_context_model_parallel_group = get_dcp_group

_PP: GroupCoordinator | None = None


def get_pp_group() -> GroupCoordinator:
    assert _PP is not None, "pipeline model parallel group is not initialized"
    return _PP


_DP: GroupCoordinator | None = None


def get_dp_group() -> GroupCoordinator:
    assert _DP is not None, "data parallel group is not initialized"
    return _DP


_EP: GroupCoordinator | None = None


def get_ep_group() -> GroupCoordinator:
    assert _EP is not None, (
        "expert parallel group is not initialized. "
        "EP group is only created for MoE models with num_experts > 0. "
        "This function should only be called for MoE models."
    )
    return _EP


_EPLB: GroupCoordinator | None = None


def get_eplb_group() -> GroupCoordinator:
    assert _EPLB is not None, (
        "EPLB group is not initialized. "
        "EPLB group is only created for MoE models when EPLB is enabled. "
        "Ensure parallel_config.enable_eplb is True."
    )
    return _EPLB


_PCP: GroupCoordinator | None = None


def get_pcp_group() -> GroupCoordinator:
    assert _PCP is not None, "prefill context parallel group is not initialized"
    return _PCP


@contextmanager
def graph_capture(device: torch.device):
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the CUDA graph. Its main purpose is to ensure that some
    operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current CUDA stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    context = GraphCaptureContext(torch.cuda.Stream(device=device))
    with get_tp_group().graph_capture(context), get_pp_group().graph_capture(context):
        yield context


logger = init_logger(__name__)

_ENABLE_CUSTOM_ALL_REDUCE = True


def set_custom_all_reduce(enable: bool):
    global _ENABLE_CUSTOM_ALL_REDUCE
    _ENABLE_CUSTOM_ALL_REDUCE = enable


def _init_elastic_ep_world(
        config,
        local_rank: int,
        backend: str,
        rank: int,
        world_size: int
) -> None:
    # 延迟导入 StatelessGroupCoordinator，避免顶层导入时就拉起相关依赖。
    from cfie.distributed.stateless_coordinator import StatelessGroupCoordinator

    # -------------------- 解析 elastic EP 全局 world 的基础参数 --------------------
    global _WORLD, _NODE_COUNT
    # elastic 模式下 world 组只允许初始化一次。
    assert _WORLD is None, "world group already initialized"

    # 取出并行配置，后续全局 rank/world size 都从这里推导。
    parallel_config = config.parallel_config

    # 当前进程的全局 rank = DP rank 偏移 + 当前 TP/PP/PCP 局部 rank。
    global_rank = parallel_config.data_parallel_rank * world_size + rank

    # 整个 elastic 世界横跨所有 DP shard，因此总大小要包含 DP 维度。
    global_world_size = parallel_config.world_size_across_dp

    # 构造完整的全局 rank 列表。
    all_ranks = list(range(global_world_size))

    # 先把每个 rank 单独包装成一个长度为 1 的组，作为默认占位分组。
    group_ranks = [all_ranks[i: i + 1] for i in range(global_world_size)]

    # 当前进程实际属于“全体 rank 的 world 组”，因此把 group_ranks 改成单个全量组。
    if global_rank in all_ranks:
        group_ranks = [all_ranks]

    # world stateless 组只需要一组端口即可。
    group_ports = [parallel_config.get_next_stateless_world_group_port()]

    # -------------------- 构造 elastic EP 的 stateless world 组 --------------------
    world = StatelessGroupCoordinator(
        # world 组成员就是全部全局 ranks。
        group_ranks=group_ranks,
        # 记录当前进程在本机内的 local rank。
        local_rank=local_rank,
        # 指定底层通信后端。
        torch_distributed_backend=backend,
        # world 组只承担协调作用，不额外建设备 communicator。
        use_device_communicator=False,
        # 命名为 world，和常规分布式路径保持语义一致。
        group_name="world",
        # elastic 路径统一通过 data_parallel_master_ip 建立 TCP store。
        host=parallel_config.data_parallel_master_ip,
        # world 组的端口列表。
        group_ports=group_ports,
        # 当前进程在 elastic 全局世界中的 rank。
        global_rank=global_rank,
        # elastic 全局世界的总 world size。
        global_world_size=global_world_size,
    )
    # 当前实现要求同一个 TP/PP shard 必须位于单节点内，否则 stateless DP/EP 组无法正确建链。
    assert parallel_config.nnodes_within_dp == 1, (
        "Elastic EP is not supported with multi-node TP/PP"
    )

    # 通过 world 的 TCP store 反推当前分布式环境总共有多少台节点。
    _NODE_COUNT = _node_count(world.tcp_store_group)

    # 把新建的 stateless world 记录为当前全局 world。
    _WORLD = world


def init_distributed_environment(
        world_size: int = -1,
        rank: int = -1,
        distributed_init_method: str = "env://",
        local_rank: int = -1,
        backend: str = "nccl",
        timeout: timedelta | None = None,
):
    # -------------------- 记录调用入口参数并读取当前配置 --------------------
    # 先打印初始化入口参数，便于排查 launch 参数与运行时配置是否一致。
    logger.debug(
        "world_size=%d rank=%d local_rank=%d distributed_init_method=%s backend=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
    )
    # 延迟导入当前线程/上下文绑定的 CfieConfig 读取函数。
    from cfie.config import get_current_cfie_config_or_none

    # 读取当前上下文中生效的 CFIE 配置；单进程/无配置路径下可能为 None。
    config = get_current_cfie_config_or_none()

    # 只有存在配置且 parallel_config.enable_elastic_ep=True 时，才进入 elastic EP 路径。
    enable_elastic_ep = config is not None and config.parallel_config.enable_elastic_ep

    # -------------------- 常规 DP 路径下修正全局 rank / world_size / init_method --------------------

    # 当使用内部 launcher 且存在多节点或 DP>1 时，需要把 rank/world 扩展到“跨 DP 的全局世界”。
    if (
            config is not None
            and config.parallel_config.distributed_executor_backend != "external_launcher"
            and (
            config.parallel_config.nnodes > 1
            or config.parallel_config.data_parallel_size > 1
    )
            and not enable_elastic_ep
    ):
        # 取出并行配置，后续多次访问时避免重复写长链属性。
        parallel_config = config.parallel_config
        # -------------------- 把当前 rank 从“DP 内局部 rank”修正成“跨 DP 全局 rank” --------------------

        # 全局 rank = 当前 DP 副本编号 * 每个 DP 副本内部的 world_size + 当前局部 rank。
        rank = parallel_config.data_parallel_rank * world_size + rank

        # 把 world_size 扩展到跨 DP 的全局 world 大小。
        world_size = parallel_config.world_size_across_dp

        # -------------------- 根据部署方式选择 torch.distributed 的 init_method --------------------
        # 多节点场景下复用主 master_addr/master_port。
        if parallel_config.nnodes > 1:
            ip = parallel_config.master_addr
            port = parallel_config.master_port
            # 拼出形如 tcp://host:port 的初始化地址。
            distributed_init_method = get_distributed_init_method(ip, port)
        else:
            # 单节点但 DP>1 时，使用 data_parallel_master_ip 和专门分配的 DP 端口。
            ip = parallel_config.data_parallel_master_ip
            port = parallel_config.get_next_dp_init_port()
            # 拼出 TCP 初始化地址，供不同 DP shard 共同接入。
            distributed_init_method = get_distributed_init_method(ip, port)
            # 记录修正后的 world_size/rank/init_method，便于核对 DP 初始化结果。
            logger.debug(
                "Adjusting world_size=%d rank=%d distributed_init_method=%s for DP",
                world_size,
                rank,
                distributed_init_method,
            )
    # -------------------- 需要时初始化 torch.distributed 默认 WORLD --------------------
    # 若当前进程尚未初始化 torch.distributed，则在这里创建默认 process group。
    local_single_rank_mode = (
        os.name == "nt"
        and world_size == 1
        and not torch.distributed.is_initialized()
    )
    if not torch.distributed.is_initialized() and not local_single_rank_mode:
        # 打印真正用于 init_process_group 的最终参数。
        logger.info(
            "world_size=%d rank=%d local_rank=%d distributed_init_method=%s backend=%s",
            world_size,
            rank,
            local_rank,
            distributed_init_method,
            backend,
        )
        # init_process_group 必须拿到明确的 init_method。
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment"
        )
        # 如果请求的 backend 当前后端不可用，则自动回退到 gloo。
        if not torch.distributed.is_backend_available(backend):
            logger.warning(
                "Distributed backend %s is not available; falling back to gloo.",
                backend,
            )
            # 既然要回退，就必须确保 gloo 可用。
            assert torch.distributed.is_gloo_available(), (
                "Fallback Gloo backend is not available."
            )
            # 把 backend 改成 gloo，下面统一用它初始化 WORLD。
            backend = "gloo"
        # -------------------- 创建 torch.distributed 默认 WORLD 组 --------------------
        # 这里创建的是底层 torch.distributed 默认 WORLD，后续 GroupCoordinator 会复用它。
        torch.distributed.init_process_group(
            # torch.distributed 的通信后端。
            backend=backend,
            # 进程组初始化地址，例如 env:// 或 tcp://ip:port。
            init_method=distributed_init_method,
            # 默认 WORLD 的总 rank 数。
            world_size=world_size,
            # 当前进程在默认 WORLD 中的 rank。
            rank=rank,
            # 初始化超时时间配置。
            timeout=timeout,
        )
    elif local_single_rank_mode:
        logger.info(
            "Skipping torch.distributed init_process_group for single-rank "
            "Windows local execution."
        )
        # elastic EP 还要额外检查当前 TP/PP shard 是否完全落在单节点内。
        if enable_elastic_ep:
            # 用 gloo 建一个 CPU 组，仅用于检测当前 torch WORLD 跨了多少个节点。
            tp_pp_cpu_group = torch.distributed.new_group(
                backend="gloo",
                timeout=timeout
            )
            # 若当前 TP/PP shard 跨节点，则现有 elastic EP 设计无法支持。
            if _node_count(tp_pp_cpu_group) > 1:
                # EPLB要求只有DP维度可以跨机
                raise RuntimeError(
                    "Elastic EP is not yet supported with multi-node TP/PP"
                )

    # -------------------- 补齐 local_rank --------------------
    # torch ProcessGroup 本身不保存 local_rank，因此这里需要自行推断。
    if local_rank == -1:
        # env:// launcher（例如 torchrun）通常会显式提供 LOCAL_RANK。
        # 否则退化为把当前全局 rank 视作 local rank，适用于单节点场景。
        local_rank = envs.LOCAL_RANK if distributed_init_method == "env://" else rank

    # -------------------- 初始化高层 world 组与 inner_dp_world 组 --------------------
    global _WORLD, _NODE_COUNT, _INNER_DP_WORLD
    # elastic EP 走单独的 stateless world 初始化逻辑，初始化完即可返回。
    if enable_elastic_ep:
        _init_elastic_ep_world(
            config,
            local_rank,
            backend,
            rank,
            world_size
        )
        return
    # 常规路径下，如果高层 world 组尚未初始化，则基于 torch WORLD 构造 GroupCoordinator。
    if _WORLD is None:
        # 高层 world 组的成员就是 torch.distributed 默认 WORLD 的全部 ranks。
        if torch.distributed.is_initialized():
            ranks = list(range(torch.distributed.get_world_size()))
        else:
            ranks = list(range(world_size))
        # 用全部 ranks 创建高层 world 协调器。
        _WORLD = init_world_group(ranks, local_rank, backend)
        # 多节点场景若配置中已经显式给出 nnodes，则直接复用，避免再次探测。
        if config is not None and config.parallel_config.nnodes > 1:
            _NODE_COUNT = config.parallel_config.nnodes
        elif local_single_rank_mode:
            _NODE_COUNT = 1
        else:
            # 否则通过 CPU group 的共享内存探测逻辑估算节点数。
            _NODE_COUNT = _node_count(_WORLD.cpu_group)
        # 打印最终识别出的节点数量。
        logger.debug("Detected %d nodes in the distributed environment", _NODE_COUNT)
    else:
        # 若 world 组已存在，则必须保证底层 torch WORLD 大小一致。
        assert _WORLD.world_size == torch.distributed.get_world_size(), (
            "world group already initialized with a different world size"
        )
    # -------------------- 如有需要，为“单个 DP 副本内部”额外构造 inner_dp_world --------------------
    # 当一个 DP 副本内部横跨多个节点时，需要单独建 inner_dp_world 组做局部协调。
    if config is not None and config.parallel_config.nnodes_within_dp > 1:
        # 这里依赖上面已经解析过的 parallel_config 变量；原有逻辑保持不变。
        if parallel_config.data_parallel_size > 1:
            # world_size_inner_dp 只包含单个 DP 副本内部的 TP/PP/PCP 世界大小。
            world_size_inner_dp = parallel_config.world_size
            # 第 dp_rank 个组的成员就是该 DP 副本对应的一段连续 rank。
            group_ranks = [
                [dp_rank * world_size_inner_dp + i for i in range(world_size_inner_dp)]
                for dp_rank in range(parallel_config.data_parallel_size)
            ]
            # 用这些分组创建 inner_dp_world，并开启 message queue broadcaster。
            _INNER_DP_WORLD = init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                use_message_queue_broadcaster=True,
                group_name="inner_dp_world",
                use_device_communicator=False,
            )
        else:
            # 若没有真正的 DP 维度，则 inner_dp_world 退化为全局 world。
            _INNER_DP_WORLD = _WORLD


def initialize_model_parallel(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        prefill_context_model_parallel_size: int = 1,
        decode_context_model_parallel_size: int | None = 1,
        backend: str | None = None,
) -> None:
    """
    初始化模型并行进程组。

    参数说明：
        tensor_model_parallel_size: 用于张量并行的 GPU 数量。
        pipeline_model_parallel_size: 用于流水线并行的 GPU 数量。
        backend: torch distributed 使用的通信后端名称。

    假设我们总共有 8 张 GPU，记作 g0 ... g7，
    其中用 2 张 GPU 做张量并行，用 4 张 GPU 做流水线并行。
    那么本函数会创建 4 个张量并行组和 2 个流水线并行组：
        4 个张量并行组：
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 个流水线并行组：
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    为了提高效率，调用方应尽量保证相邻 rank 位于同一台 DGX 机器上。
    例如，如果我们使用 2 台 DGX-1，总共 16 张 GPU，
    那么 rank 0 到 7 应位于第一台机器上，
    rank 8 到 15 应位于第二台机器上。
    """
    # -------------------- 校验分布式前置条件 --------------------
    # Get world size and rank. Ensure some consistencies.
    # 初始化模型并行组之前，torch.distributed 必须已经完成初始化。
    dist_initialized = torch.distributed.is_initialized()
    local_single_rank_mode = (
        not dist_initialized and _WORLD is not None and _WORLD.world_size == 1
    )
    assert dist_initialized or local_single_rank_mode

    from cfie.config import get_current_cfie_config

    # -------------------- 读取当前全局并行配置 --------------------
    # 获取当前生效的 CFIE 全局配置对象。
    config = get_current_cfie_config()
    # 读取模型内数据并行大小。
    data_parallel_size = config.parallel_config.data_parallel_size
    # 记录是否启用了 elastic expert parallel。
    enable_elastic_ep = config.parallel_config.enable_elastic_ep
    if enable_elastic_ep:
        # -------------------- elastic EP 路径下的 world/rank 解析 --------------------

        """
        world_size 和 rank 在两个分支都表示包括DP计算在内的全局大小
        """

        # elastic EP 下优先从 stateless world group 读取全局 world size。
        world_size = get_world_group().world_size
        # elastic EP 下优先从 stateless world group 读取当前全局 rank。
        rank = get_world_group().rank
        # 若外部未显式指定 backend，则 elastic EP 默认强制使用 nccl。
        backend = backend or "nccl"
        # 这里只计算 TP * PP * PCP 这一侧的局部 rank 空间大小，不包含 DP 维度。
        tp_pp_pcp_size = (
                tensor_model_parallel_size
                * pipeline_model_parallel_size
                * prefill_context_model_parallel_size
        )
        # 构造 elastic EP 本地 rank 布局，维度顺序是 PP x PCP x TP。
        local_all_ranks = torch.arange(tp_pp_pcp_size).reshape(
            pipeline_model_parallel_size,
            prefill_context_model_parallel_size,
            tensor_model_parallel_size,
        )
    else:
        # -------------------- 常规分布式路径下的 world/rank 解析 --------------------
        if dist_initialized:
            # 常规路径直接从 torch.distributed 读取 world size。
            world_size = torch.distributed.get_world_size()
            # 常规路径直接从 torch.distributed 读取 rank。
            rank = torch.distributed.get_rank()
            # 若外部未显式指定 backend，则复用 world group 当前 device_group 的 backend。
            backend = backend or torch.distributed.get_backend(
                get_world_group().device_group
            )
        else:
            world_group = get_world_group()
            world_size = world_group.world_size
            rank = world_group.rank
            backend = backend or getattr(world_group, "backend", "gloo")

    # -------------------- 构造全局 rank 布局张量 --------------------
    # all_ranks 是后续切分 TP/PP/DP/EP/PCP/DCP 各类进程组的基础 rank 网格。
    all_ranks = torch.arange(world_size).reshape(
        -1,
        data_parallel_size,
        pipeline_model_parallel_size,
        prefill_context_model_parallel_size,
        tensor_model_parallel_size,
    )  # noqa

    # -------------------- 构造 TP 组 --------------------
    # Build the tensor model-parallel groups.
    global _TP
    # TP 组只能初始化一次，重复初始化直接报错。
    assert _TP is None, "tensor model parallel group is already initialized"
    # 把 all_ranks 按 TP 维切成若干组，每组长度等于 tensor_model_parallel_size。
    group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
    # 把每个 rank 组从 tensor 转成 Python list，便于后续 group 创建接口消费。
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        # elastic EP 下改用局部 rank 布局来切 TP 组。
        group_ranks = local_all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
        # 同样转成 Python list。
        group_ranks = [x.tolist() for x in group_ranks]
    # message queue broadcaster is only used in tensor model parallel group
    # 初始化 TP 进程组，并额外开启 message queue broadcaster。
    _TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="tp",
    )

    # -------------------- 构造 DCP 组 --------------------
    global _DCP
    # DCP 组只能初始化一次。
    assert _DCP is None, "decode context model parallel group is already initialized"
    # 这里按 decode context parallel 维度把 rank 重新切组。
    group_ranks = all_ranks.reshape(-1, decode_context_model_parallel_size).unbind(0)
    # 转成 Python list 供 group 初始化。
    group_ranks = [x.tolist() for x in group_ranks]

    if enable_elastic_ep:
        # elastic EP 下使用局部 rank 布局切 DCP 组。
        group_ranks = local_all_ranks.reshape(
            -1, decode_context_model_parallel_size
        ).unbind(0)
        # 转成 Python list。
        group_ranks = [x.tolist() for x in group_ranks]

    # 初始化 DCP 进程组，并开启 message queue broadcaster。
    _DCP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="dcp",
    )

    # -------------------- 构造 PCP 组 --------------------
    global _PCP
    # PCP 组只能初始化一次。
    assert _PCP is None, "prefill context parallel group is already initialized"
    # 先把 PCP 维度转到最后，再按 prefill context parallel 大小切组。
    group_ranks = (
        all_ranks.transpose(3, 4)
        .reshape(-1, prefill_context_model_parallel_size)
        .unbind(0)
    )
    # 转成 Python list。
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        # elastic EP 下按局部 rank 布局切 PCP 组。
        group_ranks = (
            local_all_ranks.transpose(1, 2)
            .reshape(-1, prefill_context_model_parallel_size)
            .unbind(0)
        )
        # 转成 Python list。
        group_ranks = [x.tolist() for x in group_ranks]
    # 初始化 PCP 进程组。
    _PCP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="pcp"
    )

    # -------------------- 构造 PP 组 --------------------
    global _PP
    # PP 组只能初始化一次。
    assert _PP is None, "pipeline model parallel group is already initialized"
    # 把 PP 维度转到最后，再按 pipeline parallel 大小切组。
    group_ranks = (
        all_ranks.transpose(2, 4).reshape(-1, pipeline_model_parallel_size).unbind(0)
    )
    # 转成 Python list。
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        # elastic EP 下按局部 rank 布局切 PP 组。
        group_ranks = (
            local_all_ranks.transpose(0, 2)
            .reshape(-1, pipeline_model_parallel_size)
            .unbind(0)
        )
        # 转成 Python list。
        group_ranks = [x.tolist() for x in group_ranks]
    # 初始化 PP 进程组。
    _PP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="pp"
    )

    # -------------------- 构造 DP 组 --------------------
    global _DP
    # DP 组只能初始化一次。
    assert _DP is None, "data parallel group is already initialized"
    # 把 DP 维度转到最后，再按 data parallel 大小切组。
    group_ranks = all_ranks.transpose(1, 4).reshape(-1, data_parallel_size).unbind(0)
    # 转成 Python list。
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        # elastic EP 下 DP 组使用 stateless group，因此要先为每个组分配独立端口。
        parallel_config = config.parallel_config
        dp_ports = [
            parallel_config.get_next_stateless_dp_group_port() for _ in group_ranks
        ]
        # 初始化 stateless DP 组。
        _DP = _init_stateless_group(
            group_ranks,
            "dp",
            dp_ports,
            parallel_config.data_parallel_master_ip,
            backend,
        )
    else:
        # 常规路径直接初始化普通 DP 进程组。
        _DP = init_model_parallel_group(
            group_ranks, get_world_group().local_rank, backend, group_name="dp"
        )

    # -------------------- 构造 EP / EPLB 组 --------------------
    global _EP
    # EP 组只能初始化一次。
    assert _EP is None, "expert parallel group is already initialized"

    if config.model_config is None or config.model_config.is_moe:
        # 只有 MoE 模型才需要构造 EP 组。
        # 这里把 DP、PCP、TP 合并到同一组宽度里，得到 expert parallel 组。
        group_ranks = (
            all_ranks.transpose(1, 2)
            .reshape(
                -1,
                data_parallel_size
                * prefill_context_model_parallel_size
                * tensor_model_parallel_size,
            )
            .unbind(0)
        )
        # 转成 Python list。
        group_ranks = [x.tolist() for x in group_ranks]
        if enable_elastic_ep:
            # elastic EP 下为每个 EP 组分配独立 stateless 端口。
            parallel_config = config.parallel_config
            ep_ports = [
                parallel_config.get_next_stateless_ep_group_port() for _ in group_ranks
            ]
            # 初始化 stateless EP 组。
            _EP = _init_stateless_group(
                group_ranks,
                "ep",
                ep_ports,
                parallel_config.data_parallel_master_ip,
                backend,
            )
        else:
            # 常规路径直接初始化普通 EP 进程组。
            _EP = init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                group_name="ep"
            )

        # -------------------- 若启用 EPLB，则构造独立的 EPLB 通信组 --------------------

        global _EPLB
        # EPLB 组同样只能初始化一次。
        assert _EPLB is None, "EPLB group is already initialized"
        if (
                config is not None
                and config.parallel_config is not None
                and config.parallel_config.enable_eplb
        ):
            if enable_elastic_ep:
                # elastic EP 下为每个 EPLB 组分配 stateless 端口。
                eplb_ports = [
                    parallel_config.get_next_stateless_eplb_group_port()
                    for _ in group_ranks
                ]
                # 初始化 stateless EPLB 组。
                _EPLB = _init_stateless_group(
                    group_ranks,
                    "eplb",
                    eplb_ports,
                    parallel_config.data_parallel_master_ip,
                    backend,
                )
            else:
                # 常规路径直接初始化普通 EPLB 进程组。
                _EPLB = init_model_parallel_group(
                    group_ranks,
                    get_world_group().local_rank,
                    backend,
                    group_name="eplb",
                )
    # If no EP group needed, _EP remains None
    # If no EPLB group needed, _EPLB remains None

    # -------------------- 打印当前 rank 的并行角色 --------------------
    # 记录当前 rank 在各类并行组中的组内 rank，方便启动日志和问题排查。
    logger.info_once(
        "rank %s in world size %s is assigned as "
        "DP rank %s, PP rank %s, PCP rank %s, "
        "TP rank %s, EP rank %s, EPLB rank %s",
        rank,
        world_size,
        _DP.rank_in_group,
        _PP.rank_in_group,
        _PCP.rank_in_group,
        _TP.rank_in_group,
        _EP.rank_in_group if _EP is not None else "N/A",
        _EPLB.rank_in_group if _EPLB is not None else "N/A",
    )


def ensure_model_parallel_initialized(
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size: int,
        prefill_context_model_parallel_size: int = 1,
        decode_context_model_parallel_size: int | None = 1,
        backend: str | None = None,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    world_group = get_world_group()
    if hasattr(world_group, "backend"):
        backend = backend or world_group.backend
    else:
        backend = backend or torch.distributed.get_backend(world_group.device_group)
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            prefill_context_model_parallel_size,
            decode_context_model_parallel_size,
            backend,
        )
        return

    assert get_tensor_model_parallel_world_size() == tensor_model_parallel_size, (
        "tensor parallel group already initialized, but of unexpected size. "
        f"got: {get_tensor_model_parallel_world_size()=} vs. "
        f"wanted: {tensor_model_parallel_size=}"
    )
    pp_world_size = get_pp_group().world_size
    assert pp_world_size == pipeline_model_parallel_size, (
        "pipeline parallel group already initialized, but of unexpected size. "
        f"got: {pp_world_size=} vs. "
        f"wanted: {pipeline_model_parallel_size=}"
    )
    pcp_world_size = get_pcp_group().world_size
    assert pcp_world_size == prefill_context_model_parallel_size, (
        "prefill context parallel group already initialized, but of unexpected size: "
        f"{pcp_world_size=} vs. "
        f"{prefill_context_model_parallel_size=}"
    )


def prepare_communication_buffer_for_model(model: torch.nn.Module):
    """Prepare the communication buffer for the model.
    Traditional communication libraries like NCCL are almost
    model agnostic. However, emerging new communication libraries like
    MoE all2all (DeepEP) usually allocate the communication buffer
    based on the model shape for optimal performance.
    """
    if _TP is not None:
        _TP.prepare_communication_buffer_for_model(model)
    if _PCP is not None:
        _PCP.prepare_communication_buffer_for_model(model)
    if _PP is not None:
        _PP.prepare_communication_buffer_for_model(model)
    if _DP is not None:
        _DP.prepare_communication_buffer_for_model(model)
    if _EP is not None:
        _EP.prepare_communication_buffer_for_model(model)
    if _EPLB is not None:
        _EPLB.prepare_communication_buffer_for_model(model)


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return _TP is not None and _PP is not None


_TP_STATE_PATCHED = False


@contextmanager
def patch_tensor_parallel_group(tp_group: GroupCoordinator):
    """Patch the tp group temporarily until this function ends.

    This method is for draft workers of speculative decoding to run draft model
    with different tp degree from that of target model workers.

    Args:
        tp_group (GroupCoordinator): the tp group coordinator
    """
    global _TP_STATE_PATCHED
    assert not _TP_STATE_PATCHED, "Should not call when it's already patched"

    _TP_STATE_PATCHED = True
    old_tp_group = get_tp_group()
    global _TP
    _TP = tp_group
    try:
        yield
    finally:
        # restore the original state
        _TP_STATE_PATCHED = False
        _TP = old_tp_group


def get_tensor_model_parallel_world_size() -> int:
    """Return world size for the tensor model parallel group."""
    return get_tp_group().world_size


def get_tensor_model_parallel_rank() -> int:
    """Return my rank for the tensor model parallel group."""
    return get_tp_group().rank_in_group


def get_decode_context_model_parallel_world_size() -> int:
    """Return world size for the decode context model parallel group."""
    return get_dcp_group().world_size


def get_decode_context_model_parallel_rank() -> int:
    """Return my rank for the decode context model parallel group."""
    return get_dcp_group().rank_in_group


def get_node_count() -> int:
    """Return the total number of nodes in the distributed environment."""
    assert _NODE_COUNT is not None, "distributed environment is not initialized"
    return _NODE_COUNT


def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _TP

    if _TP:
        _TP.destroy()
    _TP = None

    global _DCP
    if _DCP:
        _DCP.destroy()
    _DCP = None

    global _PCP
    if _PCP:
        _PCP.destroy()
    _PCP = None

    global _PP
    if _PP:
        _PP.destroy()
    _PP = None

    global _DP
    if _DP:
        _DP.destroy()
    _DP = None

    global _EP
    if _EP:
        _EP.destroy()
    _EP = None

    global _EPLB
    if _EPLB:
        _EPLB.destroy()
    _EPLB = None


def destroy_distributed_environment():
    global _WORLD, _NODE_COUNT
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    _NODE_COUNT = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    # Reset environment variable cache
    envs.disable_envs_cache()
    # Ensure all objects are not frozen before cleanup
    gc.unfreeze()

    destroy_model_parallel()
    destroy_distributed_environment()
    if shutdown_ray:
        import ray  # Lazy import Ray

        ray.shutdown()
    gc.collect()
    from cfie.platforms import current_platform

    if not current_platform.is_cpu():
        torch.accelerator.empty_cache()
        try:
            torch._C._host_emptyCache()
        except AttributeError:
            logger.warning(
                "torch._C._host_emptyCache() only available in Pytorch >=2.5"
            )


def in_the_same_node_as(
        pg: ProcessGroup | StatelessProcessGroup, source_rank: int = 0
) -> list[bool]:
    """
    这是一个 collective 操作，用来判断进程组里每个 rank 是否与 source rank
    位于同一台节点上。

    其基本思路是：
    1. source rank 创建一段共享内存并写入魔术字节。
    2. 其它 rank 尝试按名称打开这段共享内存。
    3. 能成功打开并读到魔术字节的 rank，就被视为与 source rank 处于同一节点。
    """
    # -------------------- 先统一解析当前进程在给定组里的 rank / world_size / ranks --------------------
    if isinstance(pg, ProcessGroup):
        # 这个探测逻辑依赖 CPU 侧对象广播与共享内存，不应在 NCCL 组上执行。
        assert torch.distributed.get_backend(pg) != torch.distributed.Backend.NCCL, (
            "in_the_same_node_as should be tested with a non-NCCL group."
        )
        # 当前进程在这个 process group 内部的 rank。
        rank = torch.distributed.get_rank(group=pg)
        # 这个 process group 内部的 world size。
        world_size = torch.distributed.get_world_size(group=pg)

        # 取出这个 process group 对应的全局 rank 列表。
        ranks = torch.distributed.get_process_group_ranks(pg)
    else:
        # StatelessProcessGroup 直接暴露 rank/world_size。
        rank = pg.rank
        world_size = pg.world_size
        # stateless 组这里默认用连续编号表示其全局 rank 空间。
        ranks = list(range(world_size))

    # -------------------- 初始化“是否同节点”的本地结果张量 --------------------
    # 每个位置对应一个 rank；1 表示确认与 source rank 同节点，0 表示不是或尚未确认。
    is_in_the_same_node = torch.tensor(
        [0] * world_size, dtype=torch.int32, device="cpu"
    )

    # 魔术字节用于校验打开的共享内存确实来自 source rank。
    magic_message = b"magic_message"
    # 共享内存句柄先初始化为空，便于 finally 中统一关闭。
    shm = None

    # -------------------- 通过共享内存可见性判断是否同节点 --------------------
    try:
        # 共享内存可能因为异常或竞争而创建失败，这里显式忽略 OSError，尽量继续探测。
        with contextlib.suppress(OSError):
            if rank == source_rank:
                # source rank 负责创建共享内存段。
                shm = shared_memory.SharedMemory(create=True, size=128)
                # 创建后理论上一定要拿到可写 buffer。
                assert shm.buf is not None, "Buffer was not created"
                # 把魔术字节写进共享内存，供其它 rank 校验。
                shm.buf[: len(magic_message)] = magic_message
                if isinstance(pg, ProcessGroup):
                    # 普通 process group 用对象广播把共享内存名字发给其它 ranks。
                    torch.distributed.broadcast_object_list(
                        [shm.name], src=ranks[source_rank], group=pg
                    )
                else:
                    # stateless 组走自己的对象广播接口。
                    pg.broadcast_obj(shm.name, src=source_rank)
                # source rank 自己显然与自己同节点。
                is_in_the_same_node[rank] = 1
            else:
                # 非 source rank 先接收共享内存名字，再尝试打开这段共享内存。
                if isinstance(pg, ProcessGroup):
                    recv = [None]
                    torch.distributed.broadcast_object_list(
                        recv, src=ranks[source_rank], group=pg
                    )
                    # 取出 source rank 广播过来的共享内存名字。
                    name = recv[0]
                else:
                    # stateless 组直接返回广播结果。
                    name = pg.broadcast_obj(None, src=source_rank)
                # Python 的 resource_tracker 会错误追踪“不是本进程创建的共享内存”，
                # 这里临时屏蔽 register，规避已知问题。
                with patch(
                        "multiprocessing.resource_tracker.register",
                        lambda *args, **kwargs: None,
                ):
                    # 按收到的名字打开 source rank 创建的共享内存。
                    shm = shared_memory.SharedMemory(name=name)
                # 打开后必须能拿到有效 buffer。
                assert shm.buf is not None, "Buffer was not opened"
                # 只有读到正确魔术字节，才说明这段共享内存对当前进程可见，也即同节点。
                if shm.buf[: len(magic_message)] == magic_message:
                    is_in_the_same_node[rank] = 1
    except Exception as e:
        # 探测失败时只记录日志，不让节点探测错误中断主流程。
        logger.error("Error ignored in is_in_the_same_node: %s", e)
    finally:
        # 无论成功失败，只要打开过共享内存，都先关闭本地句柄。
        if shm:
            shm.close()

    # -------------------- 先同步，再由 source rank 清理共享内存 --------------------
    if isinstance(pg, ProcessGroup):
        # 普通 process group 使用 barrier 确保所有 rank 都完成了共享内存访问。
        torch.distributed.barrier(group=pg)
    else:
        # stateless 组走自己的 barrier 实现。
        pg.barrier()

    # 所有 rank 都离开 barrier 后，再由 source rank 删除共享内存对象。
    with contextlib.suppress(OSError):
        if rank == source_rank and shm:
            shm.unlink()

    # -------------------- 聚合同节点结果并返回布尔列表 --------------------
    if isinstance(pg, ProcessGroup):
        # 普通 process group 直接 all-reduce，把所有 rank 的 0/1 标记聚合起来。
        torch.distributed.all_reduce(is_in_the_same_node, group=pg)
        aggregated_data = is_in_the_same_node
    else:
        # stateless 组没有现成 all-reduce，这里通过逐 rank 广播来手工聚合。
        aggregated_data = torch.zeros_like(is_in_the_same_node)
        for i in range(world_size):
            # 依次让每个 rank 广播自己的本地结果，再加到总结果里。
            rank_data = pg.broadcast_obj(is_in_the_same_node, src=i)
            aggregated_data += rank_data

    # 每个位置恰好等于 1，表示该 rank 与 source rank 同节点。
    return [x == 1 for x in aggregated_data.tolist()]


def is_global_first_rank() -> bool:
    """
    Check if the current process is the first rank globally across all
    parallelism strategies (PP, TP, DP, EP, etc.).

    Unlike group-specific checks like `get_tensor_model_parallel_rank() == 0`
    or `get_pp_group().is_first_rank`, this function checks the global rank
    across all parallelism dimensions.

    Returns:
        bool: True if this is the global first rank (rank 0), False otherwise.
              Returns True if distributed is not initialized (single process).
    """
    try:
        # If world group is available, use it for the most accurate check
        global _WORLD
        if _WORLD is not None:
            return _WORLD.is_first_rank

        # If torch distributed is not initialized, assume single process
        if not torch.distributed.is_initialized():
            return True

        # Fallback to torch's global rank
        return torch.distributed.get_rank() == 0

    except Exception:
        # If anything goes wrong, assume this is the first rank
        return True


def is_local_first_rank() -> bool:
    """
    Check if the current process is the first local rank (rank 0 on its node).
    """
    try:
        # prefer the initialized world group if available
        global _WORLD
        if _WORLD is not None:
            return _WORLD.local_rank == 0

        if not torch.distributed.is_initialized():
            return True

        # fallback to environment-provided local rank if available
        # note: envs.LOCAL_RANK is set when using env:// launchers (e.g., torchrun)
        try:
            return int(envs.LOCAL_RANK) == 0  # type: ignore[arg-type]
        except Exception:
            return torch.distributed.get_rank() == 0
    except Exception:
        return True


def _node_count(pg: ProcessGroup | StatelessProcessGroup) -> int:
    """
    返回给定进程组涉及的节点总数。

    算法思路是：
    1. 逐个挑选一个尚未归类的 rank 作为“节点代表”。
    2. 调用 `in_the_same_node_as` 找出与它同节点的全部 rank。
    3. 把这批 rank 标记成同一个 node_id。
    4. 最终统计一共分出了多少个 node_id。
    """
    # -------------------- 先读取这个组的 world size --------------------
    if isinstance(pg, ProcessGroup):
        # 普通 process group 通过 torch.distributed 查询组内 world size。
        world_size = torch.distributed.get_world_size(group=pg)
    else:
        # StatelessProcessGroup 直接暴露 world_size。
        world_size = pg.world_size

    # 单 rank 组显然只涉及 1 台节点。
    if world_size == 1:
        return 1

    # -------------------- 准备 rank -> node_id 的归类数组 --------------------
    # 0 表示“尚未归类”；正整数表示已经被分配到某个 node_id。
    node_assignment = [0] * world_size  # rank -> node_id
    # 下一个可用的 node_id，从 1 开始递增。
    next_node_id = 0

    # -------------------- 逐个扫描 rank，把同节点的 ranks 归成一类 --------------------
    for current_rank in range(world_size):
        # 已经归过类的 rank 直接跳过，避免重复探测。
        if node_assignment[current_rank] != 0:
            continue

        # 为这个尚未归类的 rank 分配一个新的节点编号。
        next_node_id += 1
        # 当前 rank 先作为这个新节点编号的代表。
        node_assignment[current_rank] = next_node_id

        # 找出所有与 current_rank 处于同一节点的 ranks。
        same_node_flags = in_the_same_node_as(pg, current_rank)
        for other_rank, is_same_node in enumerate(same_node_flags):
            # 只有“确实同节点”且“还没被归类”的 rank，才归到当前 node_id。
            if is_same_node and node_assignment[other_rank] == 0:
                node_assignment[other_rank] = next_node_id

    # 最终用掉了多少个 node_id，就说明进程组覆盖了多少台节点。
    return next_node_id
