# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

import torch
from torch.distributed import Backend, ProcessGroup

from cfie.distributed.device_communicators.cuda_communicator import CudaCommunicator
from cfie.distributed.parallel_state import (
    GroupCoordinator,
    TensorMetadata,
    _get_unique_name,
    _register_group,
    _split_tensor_dict,
)
from cfie.distributed.utils import (
    StatelessProcessGroup,
    stateless_destroy_torch_distributed_process_group,
    stateless_init_torch_distributed_process_group,
)
from cfie.logger import init_logger
from cfie.utils.import_utils import resolve_obj_by_qualname

logger = init_logger(__name__)


class StatelessGroupCoordinator(GroupCoordinator):
    """
    `parallel_state.GroupCoordinator` 的 stateless 版本。

    它不会依附已有的 torch.distributed WORLD 去 `new_group()`，
    而是自己额外创建三类独立通道：
    - device_group: 设备侧高性能 collective
    - cpu_group: CPU 协调用 gloo 组
    - tcp_store_group: 基于 TCPStore 的对象/元数据控制面通道

    这样就能在不销毁现有 WORLD/子组的前提下，
    再额外拉起一套参与者集合不同的通信组。
    """

    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str | Backend,
        use_device_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: str | None = None,
        host: str = "127.0.0.1",
        group_ports: list[list[int]] | None = None,
        global_rank: int = 0,
        global_world_size: int = 1,
    ):
        # -------------------- 先规范组名并完成全局注册 --------------------
        # 若调用方未显式给组名，则退化成 anonymous。
        group_name = group_name or "anonymous"
        # 生成一个全局唯一的组名，避免和其它 coordinator 名称冲突。
        self.unique_name = _get_unique_name(group_name)
        # 把当前 coordinator 注册到全局组表，便于按名查找与清理。
        _register_group(self)

        # -------------------- 记录当前进程的全局 rank / local_rank --------------------
        # 这里的 rank 是 stateless 全局世界里的 rank，而不是组内 rank。
        self.rank = global_rank
        # local_rank 仍表示当前进程在本机内绑定的设备编号。
        self.local_rank = local_rank

        # 这三个变量会在“找到当前进程所属组”后被真正填充。
        self_device_group = None
        self_cpu_group = None
        self_tcp_store_group = None

        # 延迟导入平台对象，后面需要根据平台类型选择设备与 communicator。
        from cfie.platforms import current_platform

        # -------------------- 解析 backend 与端口参数 --------------------
        # 统一把 backend 转成字符串，后面传给 stateless PG 初始化函数。
        backend = str(torch_distributed_backend)
        # 记录这个 stateless 组最终使用的设备 backend。
        self.backend = backend
        # StatelessGroupCoordinator 必须显式拿到每个逻辑组对应的端口三元组。
        assert group_ports is not None, "group_ports is not provided"
        # -------------------- 遍历所有候选分组，找到当前 rank 所属的那个组 --------------------
        for idx, ranks in enumerate(group_ranks):
            if self.rank in ranks:
                # 记录当前组的全局 rank 列表。
                self.ranks = ranks
                # 当前组的 world size 就是成员个数。
                self.world_size = len(ranks)
                # 当前进程在这个组内的组内 rank。
                self.rank_in_group = ranks.index(self.rank)

                # 当前逻辑组对应的三元端口：[device_port, cpu_port, tcp_store_port]。
                ports = group_ports[idx]
                # 设备侧 ProcessGroup 使用的端口。
                device_port = ports[0]
                # CPU/gloo ProcessGroup 使用的端口。
                cpu_port = ports[1]
                # TCPStore 控制面通道使用的端口。
                tcp_store_port = ports[2]

                # -------------------- 创建设备侧 ProcessGroup --------------------
                device_group = stateless_init_torch_distributed_process_group(
                    # rendezvous host。
                    host=host,
                    # 设备组的 rendezvous 端口。
                    port=device_port,
                    # 当前进程在这个逻辑组内的组内 rank。
                    rank=self.rank_in_group,
                    # 这个逻辑组的 world size。
                    world_size=self.world_size,
                    # 设备侧 backend，通常是 nccl。
                    backend=backend,
                    # 给这个设备组注册稳定名字。
                    group_name=f"{self.unique_name}_device",
                )
                # -------------------- 创建 CPU/gloo ProcessGroup --------------------
                cpu_group = stateless_init_torch_distributed_process_group(
                    # rendezvous host。
                    host=host,
                    # CPU 组的 rendezvous 端口。
                    port=cpu_port,
                    # 仍使用同一个组内 rank。
                    rank=self.rank_in_group,
                    # CPU 组与设备组成员集合完全一致。
                    world_size=self.world_size,
                    # CPU 协调通道固定走 gloo。
                    backend="gloo",
                    # 给这个 CPU 组注册稳定名字。
                    group_name=f"{self.unique_name}_cpu",
                )
                # -------------------- 创建基于 TCPStore 的控制面通道 --------------------
                tcp_store_group = StatelessProcessGroup.create(
                    # TCPStore server/client 所在主机。
                    host=host,
                    # TCPStore 控制面端口。
                    port=tcp_store_port,
                    # 当前进程在这个逻辑组里的组内 rank。
                    rank=self.rank_in_group,
                    # 控制面通道的 world size 与前两者一致。
                    world_size=self.world_size,
                )

                # 暂存当前 rank 所属组真正对应的三类通信通道。
                self_device_group = device_group
                self_cpu_group = cpu_group
                self_tcp_store_group = tcp_store_group

        # 当前全局 rank 必须能在 group_ranks 里找到自己所属的组。
        assert self_cpu_group is not None
        assert self_device_group is not None
        assert self_tcp_store_group is not None

        # -------------------- 把找到的三类通道正式挂到 coordinator 实例上 --------------------
        self.cpu_group = self_cpu_group
        self.device_group = self_device_group
        self.tcp_store_group = self_tcp_store_group

        # -------------------- 根据当前平台解析本进程绑定的设备对象 --------------------
        if current_platform.is_cuda_alike():
            # CUDA-like 平台直接按 local_rank 选 cuda 设备。
            self.device = torch.device(f"cuda:{local_rank}")
        elif current_platform.is_xpu():
            # XPU 平台按 local_rank 选 xpu 设备。
            self.device = torch.device(f"xpu:{local_rank}")
        elif current_platform.is_out_of_tree():
            # 第三方平台使用平台注册的 device_name。
            self.device = torch.device(f"{current_platform.device_name}:{local_rank}")
        else:
            # 兜底退化成 CPU 设备。
            self.device = torch.device("cpu")

        # -------------------- 按需创建设备侧高性能 communicator --------------------
        # 记录调用方是否要求额外的 device communicator。
        self.use_device_communicator = use_device_communicator
        # 默认先不创建，后面按条件打开。
        self.device_communicator = None
        if use_device_communicator and self.world_size > 1:
            # 从平台对象解析真正要使用的 device communicator 类。
            device_comm_cls = resolve_obj_by_qualname(
                current_platform.get_device_communicator_cls()
            )
            # 当前 stateless 路径只支持 CudaCommunicator。
            assert device_comm_cls == CudaCommunicator
            # 构建设备侧 communicator，它会复用 device_group/cpu_group/tcp_store_group。
            self.device_communicator = CudaCommunicator(
                # CPU 协调通道。
                cpu_group=self.cpu_group,
                # 当前 rank 绑定的本地设备。
                device=self.device,
                # 设备侧 ProcessGroup。
                device_group=self.device_group,
                # 这个 communicator 对应的唯一名字。
                unique_name=self.unique_name,
                # 这个 stateless 组的全局 ranks 列表。
                global_ranks=self.ranks,
                # 整个 stateless 全局世界的 world size。
                global_world_size=global_world_size,
                # 控制面 TCPStore 通道，供设备 communicator 做额外元数据交换。
                tcp_store_group=self.tcp_store_group,
            )

        # stateless coordinator 当前不使用消息队列广播器，这里显式占位为 None。
        self.mq_broadcaster = None

        # -------------------- 记录若干能力开关 --------------------
        # CUDA-like 或 TPU 平台允许走自定义 op collectives。
        self.use_custom_op_call = (
            current_platform.is_cuda_alike() or current_platform.is_tpu()
        )
        # 当前 stateless 路径不启用 CPU custom send/recv。
        self.use_cpu_custom_send_recv = False

    def destroy(self):
        if self.device_communicator:
            self.device_communicator.destroy()
        if self.device_group:
            stateless_destroy_torch_distributed_process_group(self.device_group)
        if self.cpu_group:
            stateless_destroy_torch_distributed_process_group(self.cpu_group)

    def size(self) -> int:
        """Return the world size of this group."""
        return self.world_size

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        if self.world_size == 1:
            return input_

        if self.device_communicator and input_.is_cuda:
            return self.device_communicator.broadcast(input_, src)
        else:
            return self.tcp_store_group.broadcast(input_, src)

    def broadcast_object(self, obj=None, src: int = 0):
        if self.world_size == 1:
            return obj
        return self.tcp_store_group.broadcast_obj(obj, src)

    def broadcast_object_list(
        self, obj_list: list[Any], src: int = 0, group: ProcessGroup | None = None
    ):
        assert src < self.world_size

        if self.world_size == 1:
            return obj_list

        if self.rank_in_group == src:
            for obj in obj_list:
                self.tcp_store_group.broadcast_obj(obj, src)
        else:
            for i in range(len(obj_list)):
                obj_list[i] = self.tcp_store_group.broadcast_obj(None, src)

        return obj_list

    def broadcast_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any] | None = None,
        src: int = 0,
        group: ProcessGroup | None = None,
        metadata_group: ProcessGroup | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        if self.world_size == 1:
            return tensor_dict

        if self.rank_in_group == src:
            assert isinstance(tensor_dict, dict), (
                f"Expecting a dictionary, got {type(tensor_dict)}"
            )
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        else:
            metadata_list = None
            tensor_list = []

        recv_metadata_list: list[tuple[str, Any]] = self.tcp_store_group.broadcast_obj(
            metadata_list, src
        )

        if self.rank_in_group != src:
            tensor_dict = {}
            for key, value in recv_metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(
                        value.size, dtype=value.dtype, device=value.device
                    )
                    tensor_list.append(tensor)
                    tensor_dict[key] = tensor
                else:
                    tensor_dict[key] = value

        for tensor in tensor_list:
            if tensor.numel() == 0:
                continue
            if self.device_communicator and tensor.is_cuda:
                tensor.copy_(self.device_communicator.broadcast(tensor, src))
            else:
                tensor.copy_(self.tcp_store_group.broadcast(tensor, src))

        return tensor_dict

    def send_object(self, obj, dst: int) -> None:
        assert dst < self.world_size
        assert dst != self.rank_in_group
        self.tcp_store_group.send_obj(obj, dst)

    def recv_object(self, src: int):
        assert src < self.world_size
        assert src != self.rank_in_group
        return self.tcp_store_group.recv_obj(src)

    def send_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        if self.world_size == 1:
            return tensor_dict

        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert dst < self.world_size

        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        self.tcp_store_group.send_obj(metadata_list, dst)

        for tensor in tensor_list:
            if tensor.numel() == 0:
                continue
            if self.device_communicator and tensor.is_cuda:
                self.device_communicator.send(tensor, dst)
            else:
                self.tcp_store_group.send(tensor, dst)

        return None

    def recv_tensor_dict(
        self,
        src: int | None = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        if self.world_size == 1:
            return None

        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        assert src < self.world_size

        recv_metadata_list = self.tcp_store_group.recv_obj(src)
        tensor_dict = {}
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
                if tensor.numel() > 0:
                    if self.device_communicator and tensor.is_cuda:
                        tensor = self.device_communicator.recv(
                            tensor.size(), tensor.dtype, src
                        )
                    else:
                        tensor = self.tcp_store_group.recv(tensor, src)
                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        return tensor_dict

    def barrier(self):
        self.tcp_store_group.barrier()

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        if self.world_size == 1:
            return input_

        if self.device_communicator is None:
            raise ValueError("No device communicator found")

        if self.rank_in_group == dst:
            gathered_list = [torch.empty_like(input_) for _ in range(self.world_size)]
            gathered_list[self.rank_in_group] = input_
            for src_rank in range(self.world_size):
                if src_rank != self.rank_in_group:
                    gathered_list[src_rank] = self.device_communicator.recv(
                        input_.size(), input_.dtype, src_rank
                    )
            return torch.cat(gathered_list, dim=dim)
        else:
            self.device_communicator.send(input_, dst)
            return None
