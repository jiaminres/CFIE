# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Expert parallelism load balancer (EPLB) metrics and states.

# Glossary

- **Logical Expert**: An expert that is part of the model's logical structure.
  It holds a set of weights and is replicated across multiple physical
  experts.
- **Redundant Expert**: To achieve load balancing, for some popular logical
  experts, we create additional copies of the expert weights. During inference,
  each of these copies can be routed to by the same set of tokens.
- **Physical Expert**: An expert that is instantiated on a specific device.
  It is a replica of a logical expert and can be rearranged across devices.
  I.e., one logical expert may have multiple sets of weights initialized on
  different devices, and each of these sets is a physical expert.
- **Local Physical Expert**: A physical expert that is instantiated on the
  current device.

For example: DeepSeek-R1 has 256 logical experts, so each MoE layer
has 256 sets of linear layer weights in the model parameters. If we add 32
redundant experts, DeepSeek-R1 will have 256 + 32 = 288 physical experts in
total. And when deploying, we'll have 288 sets of linear layer weights for each
MoE layer. If we have 32 EP ranks, then each GPU will hold 288 / 32 = 9 local
physical experts.
"""

import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributed import ProcessGroup, all_reduce

from cfie.config import ModelConfig, ParallelConfig
from cfie.distributed.parallel_state import (
    get_ep_group,
    get_node_count,
    in_the_same_node_as,
)
from cfie.distributed.stateless_coordinator import StatelessGroupCoordinator
from cfie.distributed.utils import StatelessProcessGroup
from cfie.logger import init_logger
from cfie.model_executor.models.interfaces import MixtureOfExperts

from .async_worker import start_async_worker
from .policy import EPLB_POLICIES, AbstractEplbPolicy, DefaultEplbPolicy
from .rebalance_execute import (
    RecvMetadata,
    move_from_buffer,
    rearrange_expert_weights_inplace,
)

logger = init_logger(__name__)


@dataclass
class EplbStats:
    """
    Model stats used in EPLB rebalancing algorithm.
    """

    # 全局聚合后的 expert 负载滑动窗口。
    global_expert_load_window: torch.Tensor
    """
    Experts load window.
    Shape: (window_size, num_moe_layers, num_physical_experts)
    """
    # 当前总共有多少个 physical experts（含冗余副本）。
    num_replicas: int
    """
    Number of physical experts.
    """
    # expert group 数。
    num_groups: int
    """
    Number of expert groups.
    """
    # 节点数。
    num_nodes: int
    """
    Number of nodes.
    """
    # GPU 总数。
    num_gpus: int
    """
    Number of GPUs.
    """


@dataclass
class EplbModelState:
    """EPLB metrics."""

    # 每层的“physical expert -> logical expert” 映射表。
    physical_to_logical_map: torch.Tensor
    """
    Mapping from physical experts to logical experts.

    Shape: (num_moe_layers, num_physical_experts)

    # Example

    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the mapping could look like this:

    ```
    [[0, 1, 2, 3, 0, 1],
     [0, 2, 0, 1, 0, 3]]
    ```
    """
    # 每层的“logical expert -> 多个 physical experts” 稀疏映射表。
    logical_to_physical_map: torch.Tensor
    """
    Mapping from logical experts to physical experts.

    This is a sparse matrix, where -1 indicates no mapping.

    Shape: (num_moe_layers, num_logical_experts, num_redundant_experts + 1)

    # Example

    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the mapping could look like this:

    ```
    [[[0, 4, -1],
      [1, 5, -1],
      [2, -1, -1],
      [3, -1, -1]],
     [[0, 2, 4],
      [3, -1, -1],
      [1, -1, -1],
      [5, -1, -1]]]
    ```
    """
    # 每层每个 logical expert 当前拥有多少个 physical 副本。
    logical_replica_count: torch.Tensor
    """
    Number of replicas for each logical expert.
    This is exactly the non-`-1` count in the `logical_to_physical_map`.

    Shape: (num_moe_layers, num_logical_experts)

    # Example
    For a 2-layer MoE model with 6 physical experts and 4 logical experts on 3
    EP ranks, the count could look like this:

    ```
    [[2, 2, 1, 1],
     [3, 1, 1, 1]]
    """

    # 单次 forward 内观测到的 physical expert 负载。
    expert_load_pass: torch.Tensor
    """
    Expert load during this forward pass. 
    We use the token count each expert processes as the load.

    Shape: (num_moe_layers, num_physical_experts)
    """
    # 多次 forward 累积形成的负载滑动窗口。
    expert_load_window: torch.Tensor
    """
    A sliding window of expert load.

    Shape: (window_size, num_moe_layers, num_physical_experts)

    NOTE: The expert_load_view now records load for all physical experts
    rather than just local experts. This ensures consistent load statistics
    across different dispatch methods (naive all-to-all, DeepEP).
    The recorded load will be multiplied by dp_size when using naive all-to-all
    due to each DP rank contributing the same token set to the calculation.
    See:
    https://github.com/cfie-project/cfie/pull/22167#pullrequestreview-3086143856
    """
    # 模型名，用于日志区分主模型/草稿模型。
    model_name: str
    # 绑定的 MoE 模型对象。
    model: MixtureOfExperts
    # 暂存 expert 权重迁移数据的 buffer。
    expert_buffer: list[torch.Tensor]
    """
    The buffer to store the expert weights during transfer.
    """
    # 保护 expert_buffer 的线程锁。
    buffer_lock: threading.Lock
    """
    The lock to protect the expert buffer.
    """
    # async worker 填满 buffer 后记录的 CUDA event。
    buffer_ready_event: torch.cuda.Event | None
    """
    CUDA event recorded when the async worker finishes filling the buffer.
    The main thread waits on this before consuming the buffer.
    """
    # 主线程消费完 buffer 后记录的 CUDA event。
    buffer_consumed_event: torch.cuda.Event | None
    """
    CUDA event recorded after the main thread finishes consuming the buffer.
    The async worker waits on this before writing to the buffer again.
    """
    # 主线程完成 load all-reduce 后记录的 CUDA event。
    window_ready_event: torch.cuda.Event | None
    """
    CUDA event recorded after all-reduce and clone on the main thread.
    The async worker waits on this before accessing global_expert_load_window.
    """
    # 当前 EP rank 的 buffer 是否已经准备好，可被其他 rank 消费。
    ep_buffer_ready: int
    """
    The flag indicates whether the expert buffer is ready for transfer.
    0 or 1.
    """
    # async EPLB 当前准备传输的层号。
    layer_to_transfer: int
    """
    The layer index to transfer in async mode.
    """
    # 新映射是否已经计算完成。
    rebalanced: bool
    """
    The flag indicates whether the experts rebalance have been computed.
    """
    # async 路径下是否还需要继续轮询其他 rank 的 buffer ready 标志。
    pending_global_ready_check: bool
    """
    Whether the async EPLB needs to poll peers for buffer readiness.
    """
    # 针对当前模型的一次重排统计信息。
    eplb_stats: EplbStats | None
    """
    EPLB stats for the model.
    """
    # move_to_buffer / move_to_workspace 之间的中间标记：哪些 expert 未变化。
    is_unchanged: np.ndarray
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    The size is same as the num of physical experts in the current layer.
    """
    # move_to_buffer / move_to_workspace 之间的中间标记：哪些 expert 在本地收到。
    is_received_locally: np.ndarray
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    The size is same as the num of physical experts in the current layer.
    """
    # move_to_buffer / move_to_workspace 之间的接收元数据。
    recv_metadata: RecvMetadata
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    """
    # async worker 所使用的 CUDA 设备编号。
    cuda_device_index: int | None
    """
    CUDA device index for the async EPLB worker thread.
    """
    # 重排后新的 physical_to_logical_map，等待真正落盘到模型状态。
    new_physical_to_logical_map: torch.Tensor | None = None
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    the size is same as physical_to_logical_map
    """
    # 重排后新的 logical_to_physical_map。
    new_logical_to_physical_map: torch.Tensor | None = None
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    the size is same as logical_to_physical_map
    """
    # 重排后新的 logical_replica_count。
    new_logical_replica_count: torch.Tensor | None = None
    """
    intermediate variable between `move_to_buffer` and `move_to_workspace`.
    the size is same as logical_replica_count
    """


class EplbState:
    """
    EplbState of each expert parallel model. Key is the model config hash.
    """

    def __init__(self, parallel_config: ParallelConfig, device: torch.device):
        # 保存全局并行配置。
        self.parallel_config = parallel_config
        # 保存当前 rank 所在设备。
        self.device = device
        # 用 model_config hash -> EplbModelState 管理多个模型实例。
        self.model_states: dict[str, EplbModelState] = {}
        # 默认使用的 EPLB policy 类型。
        self.policy: type[AbstractEplbPolicy] = DefaultEplbPolicy
        """
        Selected EPLB algorithm class
        """
        # 当前写入滑动窗口的 step 游标。
        self.expert_load_window_step: int = 0
        """
        Current step in the sliding window.

        Different from `expert_rearrangement_step`, 
        each EP rank may have its own `expert_load_window_step`.
        """
        # 负载滑动窗口大小。
        self.expert_load_window_size: int = 0
        """
        Size of the expert load sliding window.
        This is a constant and is taken from the config.
        """
        # 距离上次重排已经过去多少 step。
        self.expert_rearrangement_step: int = 0
        """
        Steps after last rearrangement.
        Will trigger a rearrangement if it exceeds the threshold.

        NOTE: Keep in mind that all EP ranks need to have the same
        `expert_rearrangement_step` value to ensure synchronization.
        Otherwise, the rearrangement will hang at collective
        communication calls.
        """
        # 触发下一次重排所需的 step 间隔。
        self.expert_rearrangement_step_interval: int = 0
        """
        Interval for expert rearrangement steps.
        This is a constant and is taken from the config.
        """
        # 是否启用异步 EPLB。
        self.is_async: bool = False
        """
        The flag indicates whether the EPLB is running in async mode.
        """
        # 用于唤醒异步线程开始新一轮 rearrange。
        self.rearrange_event = threading.Event()
        """
        Event to signal when a new rearrangement is needed for the async thread.
        """
        # 后台异步 worker 线程。
        self.async_worker: threading.Thread | None = None
        """
        Background thread handling async transfers.
        """
        # 异步 worker 绑定的 CUDA device index。
        self.cuda_device_index: int | None = None
        """
        CUDA device index for the async EPLB worker thread.
        """
        # 当前真正参与映射的 physical experts 数量。
        self.num_valid_physical_experts: int = 0
        """
        Number of valid physical experts.
        This is the number of physical experts that are
        actually mapped to logical experts. In elastic EP,
        newly started EP ranks may not have physical experts
        mapped yet.
        """
        # CUDA 设备上记录 index，供 async worker 线程重新设置 device。
        if self.device.type == "cuda":
            self.cuda_device_index = self.device.index
            if self.cuda_device_index is None and torch.cuda.is_available():
                self.cuda_device_index = torch.accelerator.current_device_index()

    @staticmethod
    def build_initial_global_physical_to_logical_map(
        num_routed_experts: int,
        num_redundant_experts: int,
    ) -> Sequence[int]:
        """
        构建初始的全局 physical->logical 映射：
        [routed experts, redundant experts]
        """
        # routed experts 先按一一对应写入前缀映射。
        global_physical_to_logical_map = list(range(num_routed_experts))

        # 冗余 experts 追加到末尾，并按轮询映射到已有 logical experts。
        global_physical_to_logical_map += [
            i % num_routed_experts for i in range(num_redundant_experts)
        ]

        # 返回初始 global physical expert id -> logical expert id 映射。
        return global_physical_to_logical_map

    def validate_ep_configuration(self, new_model: MixtureOfExperts):
        """
        Validate that the expert parallel configuration of
        the new model is the same as the existing models.
        """
        # 只有已经注册过模型时，才需要和历史模型比较 EP/EPLB 拓扑是否一致。
        if len(self.model_states) > 0:
            model = next(iter(self.model_states.values())).model
            # routed experts / redundant experts / physical experts / logical experts / groups
            # 这些关键维度任意一个不同，都不能复用同一份 EPLB 状态。
            if (
                model.num_routed_experts != new_model.num_routed_experts
                or model.num_redundant_experts != new_model.num_redundant_experts
                or model.num_physical_experts != new_model.num_physical_experts
                or model.num_logical_experts != new_model.num_logical_experts
                or model.num_expert_groups != new_model.num_expert_groups
            ):
                raise RuntimeError(
                    "Model: {} "
                    "with config {} "
                    "{} {} {} {} "
                    "mismatch with new model {} "
                    "with config {} "
                    "{} {} {} {}".format(
                        type(model),
                        model.num_routed_experts,
                        model.num_redundant_experts,
                        model.num_physical_experts,
                        model.num_logical_experts,
                        model.num_expert_groups,
                        type(new_model),
                        new_model.num_routed_experts,
                        new_model.num_redundant_experts,
                        new_model.num_physical_experts,
                        new_model.num_logical_experts,
                        new_model.num_expert_groups,
                    )
                )

    def add_model(
        self,
        model: MixtureOfExperts,
        model_config: ModelConfig,
    ):
        """
        Build the initial EPLB state.
        """
        # ----------------- 先校验新模型是否与现有 EPLB 拓扑兼容 -----------------
        self.validate_ep_configuration(model)
        # 记录当前 EPLB 是否走异步模式。
        self.is_async = self.parallel_config.eplb_config.use_async

        # ----------------- 构造初始的 physical/logical expert 双向映射 -----------------
        physical_to_logical_map_list = (
            EplbState.build_initial_global_physical_to_logical_map(
                model.num_routed_experts,
                model.num_redundant_experts,
            )
        )
        physical_to_logical_map = torch.tensor(
            physical_to_logical_map_list,
            device=self.device,
        )
        # 当前实现把每个 logical expert 最多可挂的 physical 副本数上限写死为 1024。
        # Assuming 8 GPUs per node, this supports up to
        # (1023 + 1) / 8 = 128 nodes for now.
        # TODO(rui): make this configurable
        MAX_EXPERT_REDUNDANCY = 1023
        assert model.num_redundant_experts <= MAX_EXPERT_REDUNDANCY, (
            f"num_redundant_experts {model.num_redundant_experts} "
            f"must be less than or equal to {MAX_EXPERT_REDUNDANCY}"
        )
        max_slots_per_logical_expert = MAX_EXPERT_REDUNDANCY + 1
        # logical_to_physical_map 初始全填 -1，表示该槽位还没有副本。
        logical_to_physical_map = torch.full(
            (model.num_logical_experts, max_slots_per_logical_expert),
            -1,
            device=self.device,
        )
        # replica_count 跟踪每个 logical expert 已经登记了多少个副本。
        logical_replica_count = torch.zeros(
            (model.num_logical_experts,),
            device=self.device,
            dtype=torch.long,
        )

        # 遍历所有 physical experts，把它们回填进 logical_to_physical_map。
        for i in range(model.num_physical_experts):
            logical_idx = physical_to_logical_map[i]
            logical_to_physical_map[logical_idx, logical_replica_count[logical_idx]] = i
            logical_replica_count[logical_idx] += 1

        # ----------------- 将单层映射扩展成“每层各一份”的 MoE 层级映射 -----------------
        # Duplicate initial mapping for all layers
        physical_to_logical_map = (
            physical_to_logical_map.unsqueeze(0)
            .expand(
                model.num_moe_layers,
                -1,
            )
            .contiguous()
        )
        logical_to_physical_map = (
            logical_to_physical_map.unsqueeze(0)
            .expand(
                model.num_moe_layers,
                -1,
                -1,
            )
            .contiguous()
        )
        logical_replica_count = (
            logical_replica_count.unsqueeze(0)
            .expand(
                model.num_moe_layers,
                -1,
            )
            .contiguous()
        )

        # ----------------- 初始化本轮负载和滑动窗口 -----------------
        expert_load_pass = torch.zeros(
            (model.num_moe_layers, model.num_physical_experts),
            dtype=torch.int32,
            device=self.device,
        )
        # 窗口大小来自 eplb_config.window_size。
        self.expert_load_window_size = self.parallel_config.eplb_config.window_size
        expert_load_window = torch.zeros(
            (
                self.expert_load_window_size,
                model.num_moe_layers,
                model.num_physical_experts,
            ),
            dtype=torch.int32,
            device=self.device,
        )

        # 把初始重排步数设置到 step_interval 的 3/4，缩短首次重排等待时间。
        # Set the initial progress of rearrangement to 3/4
        eplb_step_interval = self.parallel_config.eplb_config.step_interval
        self.expert_rearrangement_step = max(
            0, eplb_step_interval - eplb_step_interval // 4
        )
        self.expert_rearrangement_step_interval = eplb_step_interval

        # 解析并记录当前使用的 EPLB policy 类型。
        policy_type = self.parallel_config.eplb_config.policy
        self.policy = EPLB_POLICIES[policy_type]
        logger.debug("Selected EPLB policy: %s", policy_type)

        # 把初始 EPLB 映射同步灌入模型内部，供 router/runtime 直接使用。
        model.set_eplb_state(
            expert_load_pass,
            logical_to_physical_map,
            logical_replica_count,
        )

        # expert_buffer 的形状与单层 expert_weights[0] 对齐，用于后续权重搬运。
        expert_buffer = [torch.empty_like(w) for w in model.expert_weights[0]]

        # 汇总所有初始状态，构造 EplbModelState。
        model_state = EplbModelState(
            physical_to_logical_map=physical_to_logical_map,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            expert_load_pass=expert_load_pass,
            expert_load_window=expert_load_window,
            model_name=model_config.model,
            model=model,
            expert_buffer=expert_buffer,
            buffer_lock=threading.Lock(),
            buffer_ready_event=None,
            buffer_consumed_event=None,
            window_ready_event=None,
            ep_buffer_ready=0,
            layer_to_transfer=0,
            rebalanced=False,
            pending_global_ready_check=False,
            eplb_stats=None,
            is_unchanged=np.array([]),
            is_received_locally=np.array([]),
            recv_metadata=RecvMetadata(
                recv_primary_mask=np.array([]),
                recv_count=0,
                recv_expert_ids=np.array([]),
                recv_dst_rows=np.array([]),
            ),
            cuda_device_index=self.cuda_device_index,
            new_physical_to_logical_map=None,
            new_logical_to_physical_map=None,
            new_logical_replica_count=None,
        )
        # 以 model_config hash 为键登记该模型的 EPLB 状态。
        self.model_states[model_config.compute_hash()] = model_state
        # 初始时所有 physical experts 都是有效的。
        self.num_valid_physical_experts = model.num_physical_experts

    def step(
        self,
        is_dummy: bool = False,
        is_profile: bool = False,
        log_stats: bool = False,
    ) -> None:
        """
        Step the EPLB state.

        Args:
            is_dummy (bool): If `True`, this is a dummy step and the load
                metrics recorded in this forward pass will not count.
                Defaults to `False`.
            is_profile (bool): If `True`, perform a dummy rearrangement
                with maximum communication cost. This is used in
                `profile_run` to reserve enough memory
                for the communication buffer.
            log_stats (bool): If `True`, log the expert load metrics.

        # Stats
            The metrics are all summed up across layers.
            - `avg_tokens`: The average load across ranks.
            - `max_tokens`: The maximum load across ranks.
            - `balancedness`: The ratio of average load to maximum load.
        """
        # 当前 EP device_group 用于后续 all-reduce 与 rank 判断。
        ep_group = get_ep_group().device_group
        # profile 模式下直接做一次最大开销的假重排，不走正常 step 逻辑。
        if is_profile:
            self.rearrange(is_profile=True)
            return

        # dummy step 不计入真实负载，因此把本轮统计直接清零。
        if is_dummy:
            # Do not record load metrics for dummy steps
            for eplb_model_state in self.model_states.values():
                eplb_model_state.expert_load_pass.zero_()

        # ----------------- 如有需要，先打印当前负载均衡统计 -----------------
        if (
            log_stats
            and self.expert_rearrangement_step
            % self.parallel_config.eplb_config.log_balancedness_interval
            == 0
        ):
            # Sync the expert load pass for each model (main and drafter).
            # expert_load_pass: (num_moe_layers, num_physical_experts)
            expert_load_pass_list = self._sync_load_pass()
            ep_group = get_ep_group().device_group
            for expert_load_pass, eplb_model_state in zip(
                expert_load_pass_list, self.model_states.values()
            ):
                # 先把 physical expert 维度 reshape 成 (layer, rank, local_experts)。
                # num_tokens_per_rank: (num_moe_layers, num_ranks)
                num_tokens_per_rank = (
                    expert_load_pass.reshape(
                        expert_load_pass.shape[0], ep_group.size(), -1
                    )
                    .sum(dim=-1)
                    .float()
                )

                # Compute balancedness ratio:
                # for each layer:
                #   (mean load across ranks) / (max load across ranks)
                avg_tokens_tensor = num_tokens_per_rank.mean(dim=0).sum(dim=0)
                max_tokens_tensor = num_tokens_per_rank.max(dim=0).values.sum(dim=0)

                # Just to make type checker happy
                tokens_tensors: list[float] = torch.stack(
                    [avg_tokens_tensor, max_tokens_tensor]
                ).tolist()
                avg_tokens, max_tokens = tokens_tensors
                # balancedness 越接近 1，说明各 rank token 负载越平均。
                balancedness = avg_tokens / max_tokens if max_tokens > 0 else 0.0

                if ep_group.rank() == 0:
                    logger.info(
                        "EPLB step: %d for model %s: avg_tokens=%.2f, "
                        "max_tokens=%d, balancedness=%.4f, "
                        "steps until the next rearrangement: %d",
                        self.expert_rearrangement_step,
                        eplb_model_state.model_name,
                        avg_tokens,
                        max_tokens,
                        balancedness,
                        self.expert_rearrangement_step_interval
                        - self.expert_rearrangement_step,
                    )

        # ----------------- 把本轮负载写入滑动窗口 -----------------
        # Update the expert load sliding window
        if not is_dummy:
            for eplb_model_state in self.model_states.values():
                # clone 一份 expert_load_pass，冻结当前 step 的观测结果。
                eplb_model_state.expert_load_window[self.expert_load_window_step] = (
                    eplb_model_state.expert_load_pass.clone()
                )
                # 写入窗口后立即清零，等待下一轮 forward 重新累计。
                eplb_model_state.expert_load_pass.zero_()

            self.expert_load_window_step += 1
            if self.expert_load_window_step >= self.expert_load_window_size:
                self.expert_load_window_step = 0

        # 无论是不是 dummy step，都要前进 rearrangement step，确保各 rank 节奏一致。
        # Step the expert rearrangement step
        # Note that even if this is a dummy step, we still increment the
        # rearrangement step and perform rearrangement to ensure all ranks are
        # performing collective communication.
        self.expert_rearrangement_step += 1

        # ----------------- async 模式下，先尝试把已经准备好的 buffer 落回 workspace -----------------
        if self.is_async:
            for eplb_model_state in self.model_states.values():
                all_ranks_buffer_ready = False
                if eplb_model_state.pending_global_ready_check:
                    all_ranks_buffer_ready = self._all_ranks_buffer_ready(
                        eplb_model_state
                    )
                if eplb_model_state.ep_buffer_ready and all_ranks_buffer_ready:
                    self.move_to_workspace(
                        model_state=eplb_model_state,
                        ep_group=ep_group,
                        is_profile=is_profile,
                    )

        # ----------------- 达到阈值后触发一轮真正的 rearrange -----------------
        if self.expert_rearrangement_step >= self.expert_rearrangement_step_interval:
            if self.is_async and any(
                eplb_model_state.rebalanced
                for eplb_model_state in self.model_states.values()
            ):
                # Still performing asynchronous rearrangement
                return
            self.expert_rearrangement_step = 0
            self.rearrange()

    def rearrange(
        self,
        is_profile: bool = False,
        rank_mapping: dict[int, int] | None = None,
    ) -> torch.Tensor | None:
        """
        Rearrange the experts according to the current load.

        Args:
            is_profile (bool): If `True`, perform a dummy rearrangement.
                This is used in `profile_run` to reserve enough memory,
                no memory movement will be performed. Default is False.
            rank_mapping (dict[int, int] | None): The rank mapping
                when scaling is done in EEP.
        """

        # 读取当前 EP device group 与本 rank。
        ep_group = get_ep_group().device_group
        ep_rank = ep_group.rank()

        # 仅主 rank 负责打印重排耗时日志。
        start_event = None
        end_event = None
        is_main_rank = ep_rank == 0
        if is_main_rank:
            if not self.is_async or is_profile:
                # 同步重排或 profile 模式下，主 rank 用 CUDA event 统计纯 GPU 耗时。
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            logger.info(
                "Rearranging experts %s %s...",
                "(async mode)" if self.is_async else "sync mode",
                "(profile)" if is_profile else "",
            )

        # ----------------- 先把 physical expert 负载映射回 logical expert 视角 -----------------
        # Map the physical expert load to global logical experts
        global_expert_load_windows = []
        for eplb_model_state in self.model_states.values():
            # elastic EP 场景下，只统计当前仍然有效的 physical experts。
            expert_load_window = eplb_model_state.expert_load_window[
                :, :, : self.num_valid_physical_experts
            ]
            # 构造一块 logical expert 视角的滑动窗口张量。
            logical_expert_load_window = torch.zeros(
                self.expert_load_window_size,
                eplb_model_state.model.num_moe_layers,
                eplb_model_state.model.num_logical_experts,
                dtype=eplb_model_state.expert_load_window.dtype,
                device=eplb_model_state.expert_load_window.device,
            )
            # scatter_add 按 physical_to_logical_map 把多个副本的 load 汇总到逻辑 expert 上。
            logical_expert_load_window.scatter_add_(
                dim=-1,
                index=eplb_model_state.physical_to_logical_map[
                    :, : self.num_valid_physical_experts
                ]
                .unsqueeze(0)
                .expand_as(expert_load_window)
                .long(),
                src=expert_load_window,
            )

            # 再沿时间窗口维度求和，得到“当前用于重排决策”的总 logical load。
            global_expert_load_window = logical_expert_load_window.sum(dim=0)
            global_expert_load_windows.append(global_expert_load_window)
        # 对所有 EP ranks 做 all-reduce，得到全局一致的 load 视图。
        # Perform all-reduce to get the expert load across all ranks for each model
        global_expert_load_windows = self._allreduce_list(global_expert_load_windows)

        # 暂时取任意一个模型状态读取通用拓扑参数。
        # TODO(bowen): Treat differently for prefill and decode nodes
        eplb_model_state = next(iter(self.model_states.values()))
        model = eplb_model_state.model
        num_replicas = model.num_physical_experts
        num_groups = model.num_expert_groups

        # ----------------- 若 elastic EP 正在 scale-down，则按 rank_mapping 重算节点/副本规模 -----------------
        if rank_mapping is not None and len(rank_mapping) == ep_group.size():
            # NOTE(yongji): scale down, we need to rebalance the experts on
            # remaining GPUs, transfer the experts while we haven't shutdown
            # the GPUs to be released.
            coordinator = get_ep_group()
            assert isinstance(coordinator, StatelessGroupCoordinator)
            tcp_store_group = coordinator.tcp_store_group
            num_nodes = _node_count_with_rank_mapping(tcp_store_group, rank_mapping)
            num_gpus = sum(new_rank != -1 for new_rank in rank_mapping.values())
            num_replicas = (
                num_replicas // ep_group.size() * num_gpus
            )  # handle num replicas change
        else:
            # 否则沿用当前 EP 组的真实节点数与 GPU 数。
            num_nodes = get_node_count()
            num_gpus = ep_group.size()

        # 若 GPU 数无法整除节点数，则放弃层级重排策略，退回单层次算法。
        if num_gpus % num_nodes != 0:
            num_nodes = 1
            logger.warning_once(
                f"num_gpus % num_nodes != 0, "
                "not using hierarchical rearrangement algorithm.\n"
                f"{num_gpus=}, {num_nodes=}"
            )

        # ----------------- 对每个模型分别计算新映射，并执行同步或异步重排 -----------------
        # Get new expert mappings
        for eplb_model_state, global_expert_load_window in zip(
            self.model_states.values(), global_expert_load_windows
        ):
            if not self.is_async or is_profile:
                # 同步模式下，直接在主线程算出新映射。
                # Get new expert mappings for the model
                (
                    new_physical_to_logical_map,
                    new_logical_to_physical_map,
                    new_logical_replica_count,
                ) = self.policy.rebalance_experts(
                    global_expert_load_window,
                    num_replicas,
                    num_groups,
                    num_nodes,
                    num_gpus,
                    eplb_model_state.physical_to_logical_map,
                )

                # 按旧映射 -> 新映射的差异，就地搬运各层 expert 权重。
                # Update expert weights
                rearrange_expert_weights_inplace(
                    eplb_model_state.physical_to_logical_map,
                    new_physical_to_logical_map,
                    eplb_model_state.model.expert_weights,
                    ep_group,
                    is_profile,
                    rank_mapping,
                )

                if not is_profile:
                    # physical expert 数变化时，整块替换映射张量，避免形状不匹配。
                    if (
                        eplb_model_state.physical_to_logical_map.shape[1]
                        != new_physical_to_logical_map.shape[1]
                    ):
                        eplb_model_state.physical_to_logical_map = (
                            new_physical_to_logical_map.to(
                                eplb_model_state.physical_to_logical_map.device
                            )
                        )
                    else:
                        eplb_model_state.physical_to_logical_map.copy_(
                            new_physical_to_logical_map
                        )
                    # logical_to_physical_map 可能比新结果预留了更多槽位，因此需要先 pad。
                    max_physical_slots = new_logical_to_physical_map.shape[-1]
                    assert (
                        max_physical_slots
                        <= eplb_model_state.logical_to_physical_map.shape[-1]
                    )
                    new_logical_to_physical_map = torch.nn.functional.pad(
                        new_logical_to_physical_map,
                        (
                            0,
                            eplb_model_state.logical_to_physical_map.shape[-1]
                            - max_physical_slots,
                        ),
                        value=-1,
                    )
                    eplb_model_state.logical_to_physical_map.copy_(
                        new_logical_to_physical_map
                    )
                    eplb_model_state.logical_replica_count.copy_(
                        new_logical_replica_count
                    )
                if is_main_rank:
                    # 主 rank 输出同步重排耗时。
                    assert start_event is not None
                    assert end_event is not None
                    end_event.record()
                    end_event.synchronize()
                    gpu_elapsed = start_event.elapsed_time(end_event) / 1000.0
                    logger.info(
                        "Rearranged experts %s in %.2f s.",
                        " (profile) " if is_profile else " ",
                        gpu_elapsed,
                    )
            else:
                # 异步模式下只先保存重排计划和统计信息，真正搬运交给后台线程。
                eplb_model_state.eplb_stats = EplbStats(
                    # We copy the tensor to snapshot the global_expert_load_window
                    # on the main thread so that async worker can access it safely
                    # while the main thread is running.
                    global_expert_load_window=global_expert_load_window.clone(),
                    num_replicas=num_replicas,
                    num_groups=num_groups,
                    num_nodes=num_nodes,
                    num_gpus=num_gpus,
                )
                # clone 之后再 record event，告诉异步线程这份 load snapshot 已经稳定可读。
                # Record event after clone to signal async worker
                # that load stats data is ready
                sync_event = torch.cuda.Event()
                sync_event.record()
                eplb_model_state.window_ready_event = sync_event

                # 标记后台线程可以从第 0 层开始分层传输了。
                eplb_model_state.rebalanced = True
                eplb_model_state.layer_to_transfer = 0
                eplb_model_state.pending_global_ready_check = True
        # 异步模式下，统一唤醒后台线程开始搬运。
        # Signal async thread to start transferring layers
        if self.is_async and (not is_profile):
            self.rearrange_event.set()
        return None

    def start_async_loop(
        self,
        rank_mapping: dict[int, int] | None = None,
        is_profile: bool = False,
    ):
        # 非 async 模式不需要启动后台线程。
        if not self.is_async:
            return
        # 只在第一次调用时真正拉起 worker，避免重复启动多个线程。
        if self.async_worker is None:
            self.async_worker = start_async_worker(
                self,
                is_profile=is_profile,
            )

    def _update_layer_mapping_from_new(
        self, model_state: EplbModelState, layer: int
    ) -> None:
        # 若后台线程尚未生成新映射，直接返回。
        if (
            model_state.new_physical_to_logical_map is None
            or model_state.new_logical_to_physical_map is None
            or model_state.new_logical_replica_count is None
        ):
            return

        # ----------------- 先更新当前层的 physical_to_logical_map -----------------
        target_device = model_state.physical_to_logical_map.device
        new_physical = model_state.new_physical_to_logical_map
        # If the number of physical experts has changed, then the new map needs to
        # be copied synchronously to avoid a race condition with the async worker
        if model_state.physical_to_logical_map.shape[1] != new_physical.shape[1]:
            model_state.physical_to_logical_map = new_physical.to(target_device)
        else:
            model_state.physical_to_logical_map[layer].copy_(
                new_physical[layer].to(target_device, non_blocking=True)
            )

        # ----------------- 再更新当前层的 logical_to_physical_map -----------------
        logical_device = model_state.logical_to_physical_map.device
        new_logical = model_state.new_logical_to_physical_map[layer].to(logical_device)
        max_slots = model_state.logical_to_physical_map.shape[-1]
        slot_delta = max_slots - new_logical.shape[-1]
        if slot_delta > 0:
            new_logical = torch.nn.functional.pad(
                new_logical, (0, slot_delta), value=-1
            )
        model_state.logical_to_physical_map[layer].copy_(new_logical)

        # ----------------- 最后更新当前层的副本计数 -----------------
        replica_device = model_state.logical_replica_count.device
        model_state.logical_replica_count[layer].copy_(
            model_state.new_logical_replica_count[layer].to(replica_device)
        )

    def _all_ranks_buffer_ready(self, model_state: EplbModelState) -> bool:
        # 优先看是否存在 CPU group；有的话在 CPU 上同步 ready 标志。
        parallel_state = get_ep_group()
        cpu_group = getattr(parallel_state, "cpu_group", None)
        if cpu_group is not None and cpu_group.size() > 1:
            flag = torch.tensor(
                (int(model_state.ep_buffer_ready),), dtype=torch.int32, device="cpu"
            )
            all_reduce(flag, group=cpu_group)
            return int(flag.item()) == cpu_group.size()

        # 单卡时，本 rank 的 ready 标志就是全局 ready。
        device_group = parallel_state.device_group
        if device_group.size() <= 1:
            return bool(model_state.ep_buffer_ready)

        # 否则退回 device group 上同步 ready 标志。
        device = getattr(
            parallel_state, "device", model_state.physical_to_logical_map.device
        )
        flag = torch.tensor(
            (int(model_state.ep_buffer_ready),), dtype=torch.int32, device=device
        )
        all_reduce(flag, group=device_group)
        return int(flag.item()) == device_group.size()

    def move_to_workspace(
        self,
        model_state: EplbModelState,
        ep_group: ProcessGroup,
        is_profile: bool = False,
    ):
        # ----------------- 主线程等待拿到 buffer_lock，再消费异步线程准备好的 buffer -----------------
        # We call move_to_workspace only when ep_buffer_ready is 1.
        # It means we only need to wait for the lock for a short time.
        max_retries = 6  # 1 minute max
        retries = 0
        while not model_state.buffer_lock.acquire(blocking=True, timeout=10.0):
            retries += 1
            if retries >= max_retries:
                raise RuntimeError(
                    f"Rank {ep_group.rank()}: buffer_lock timeout after "
                    "{max_retries * 10}s"
                )
            logger.warning(
                "Rank %d: EPLB buffer_lock acquire failed, retrying (%d/%d)",
                ep_group.rank(),
                retries,
                max_retries,
            )
        try:
            # 新 physical 映射必须已由后台线程准备好。
            assert model_state.new_physical_to_logical_map is not None
            device_index = model_state.cuda_device_index or self.cuda_device_index
            if model_state.buffer_ready_event is not None and device_index is not None:
                # 等后台线程真正写完 buffer 对应的 CUDA event。
                stream = torch.cuda.current_stream(device=device_index)
                stream.wait_event(model_state.buffer_ready_event)
                model_state.buffer_ready_event = None
            # 取出当前层真实权重区和中转 buffer。
            expert_weights = model_state.model.expert_weights[
                model_state.layer_to_transfer
            ]
            expert_weights_buffer = model_state.expert_buffer
            new_indices = model_state.new_physical_to_logical_map[
                model_state.layer_to_transfer
            ].numpy()
            # 根据后台线程准备好的 recv_metadata/new_indices，把 buffer 中数据搬回工作区。
            move_from_buffer(
                expert_weights=expert_weights,
                expert_weights_buffers=expert_weights_buffer,
                is_unchanged=model_state.is_unchanged,
                is_received_locally=model_state.is_received_locally,
                recv_metadata=model_state.recv_metadata,
                new_indices=new_indices,
                ep_rank=ep_group.rank(),
            )
            # 主线程消费完成后记录事件，让后台线程知道这块 buffer 可以再次覆写。
            # Record event after consuming buffer to signal async thread
            # that it's safe to overwrite the intermediate buffer
            consumed_event = torch.cuda.Event()
            consumed_event.record()
            model_state.buffer_consumed_event = consumed_event

            transferred_layer = model_state.layer_to_transfer
            self._update_layer_mapping_from_new(model_state, transferred_layer)
            # 当前层完成后推进到下一层。
            # After the main thread consumes, advance layer_to_transfer
            model_state.layer_to_transfer += 1
            model_state.ep_buffer_ready = 0
            logger.debug(
                "model %s successfully move_to_workspace layer %d",
                model_state.model_name,
                transferred_layer,
            )
            if model_state.layer_to_transfer >= model_state.model.num_moe_layers:
                # 所有层都处理完后，清理中间态并结束本轮异步重排。
                self.post_eplb(model_state, is_profile)
                model_state.rebalanced = False
                model_state.layer_to_transfer = 0
                model_state.pending_global_ready_check = False
                logger.info(
                    "finish async transfer for model %s rank %d layer %d",
                    model_state.model_name,
                    ep_group.rank(),
                    model_state.model.num_moe_layers,
                )

        finally:
            try:
                # 无论成功失败都释放锁，避免异步 EPLB 死锁。
                model_state.buffer_lock.release()
            except Exception as e:
                logger.error(
                    "Rank %d: buffer_lock release failed in move_to_workspace: %s",
                    ep_group.rank(),
                    str(e),
                )

    def post_eplb(self, model_state: EplbModelState, is_profile: bool = False) -> None:
        # 这里要求三份新映射中间态都已经存在。
        assert model_state.new_physical_to_logical_map is not None
        assert model_state.new_logical_to_physical_map is not None
        assert model_state.new_logical_replica_count is not None

        # 主线程完成消费后，把中间态引用全部清空，为下一轮 EPLB 做准备。
        model_state.new_physical_to_logical_map = None
        model_state.new_logical_to_physical_map = None
        model_state.new_logical_replica_count = None

    def _allreduce_list(self, tensor_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        All-reduce a list of tensors.
        """
        # 只有一个张量时，直接原地 all-reduce 即可。
        if len(tensor_list) == 1:
            all_reduce(tensor_list[0], group=get_ep_group().device_group)
            return tensor_list
        # 当前 helper 假设列表里所有张量都是二维，且列数一致。
        assert all(t.dim() == 2 for t in tensor_list), "All tensors must be 2D."
        assert all(t.shape[1] == tensor_list[0].shape[1] for t in tensor_list), (
            "All tensors must have the same shape[1]."
        )
        # 先沿行维拼接成一个大张量，减少多次 collective 调用。
        # Concatenate, all_reduce, then unpack to original shapes.
        # We assume all tensors are 2D and shape[1] (num_physical_experts)
        # is the same across all models.
        shapes = [t.shape for t in tensor_list]
        concat_tensor = torch.cat(tensor_list, dim=0)

        ep_group = get_ep_group().device_group
        all_reduce(concat_tensor, group=ep_group)

        # all-reduce 完成后再按原始 shape 切回列表形式。
        all_reduce_list = []
        offset = 0
        for shape in shapes:
            all_reduce_list.append(concat_tensor[offset : offset + shape[0], :])
            offset += shape[0]
        return all_reduce_list

    def _sync_load_pass(self) -> list[torch.Tensor]:
        """
        Sync the expert load pass across all ranks for log stats.
        Doesn't update the expert load pass in eplb_model_state.
        """
        # clone 一份各模型当前 expert_load_pass，避免直接改写原状态。
        load_pass_list = []
        for eplb_model_state in self.model_states.values():
            load_pass_list.append(eplb_model_state.expert_load_pass.clone())
        # 对每个模型的负载快照做 all-reduce。
        return self._allreduce_list(load_pass_list)

    @classmethod
    def from_mapping(
        cls,
        model: MixtureOfExperts,
        model_config: ModelConfig,
        device: torch.device,
        parallel_config: ParallelConfig,
        expanded_physical_to_logical: torch.Tensor,
        num_valid_physical_experts: int,
    ) -> "EplbState":
        # 先按普通初始化路径构造一个新的 EplbState。
        eplb_state = cls(
            parallel_config=parallel_config,
            device=device,
        )
        eplb_state.add_model(
            model=model,
            model_config=model_config,
        )
        # 覆盖当前有效 physical experts 数。
        eplb_state.num_valid_physical_experts = num_valid_physical_experts
        num_moe_layers = expanded_physical_to_logical.shape[0]
        num_physical_experts = expanded_physical_to_logical.shape[1]
        # 取出刚刚登记好的模型状态，并把 physical_to_logical_map 替换成传入映射。
        eplb_model_state = eplb_state.model_states[model_config.compute_hash()]
        eplb_model_state.physical_to_logical_map.copy_(expanded_physical_to_logical)

        # 根据新的 physical_to_logical_map 反推 logical_to_physical_map 与副本计数。
        logical_to_physical_map = torch.full(
            (
                num_moe_layers,
                model.num_logical_experts,
                eplb_model_state.logical_to_physical_map.shape[2],
            ),
            -1,
            dtype=torch.int64,
        )
        logical_replica_count = torch.zeros(
            (num_moe_layers, model.num_logical_experts),
            dtype=torch.int64,
        )
        expanded_physical_to_logical_numpy = expanded_physical_to_logical.cpu().numpy()
        for layer_idx in range(num_moe_layers):
            for phys_idx in range(num_physical_experts):
                logical_idx = expanded_physical_to_logical_numpy[layer_idx, phys_idx]
                if logical_idx >= 0:
                    # 把当前 physical expert 填进对应 logical expert 的下一个副本槽位。
                    replica_idx = logical_replica_count[layer_idx, logical_idx]
                    logical_to_physical_map[layer_idx, logical_idx, replica_idx] = (
                        phys_idx
                    )
                    logical_replica_count[layer_idx, logical_idx] += 1

        # 把 CPU 上重建出的映射搬回目标 device，并写入模型状态。
        logical_to_physical_map = logical_to_physical_map.to(device)
        logical_replica_count = logical_replica_count.to(device)
        eplb_model_state.logical_to_physical_map.copy_(logical_to_physical_map)
        eplb_model_state.logical_replica_count.copy_(logical_replica_count)
        return eplb_state


@dataclass
class EplbLayerState:
    """Runtime EPLB data stored in the MoE layer."""

    # 当前层在 forward 期间记录的 physical expert 负载视图。
    expert_load_view: torch.Tensor | None = None
    # 当前层的 logical -> physical 映射。
    logical_to_physical_map: torch.Tensor | None = None
    # 当前层每个 logical expert 的副本数。
    logical_replica_count: torch.Tensor | None = None


def _node_count_with_rank_mapping(
    pg: ProcessGroup | StatelessProcessGroup,
    rank_mapping: dict[int, int],
) -> int:
    # 统一拿到 world_size，兼容普通 ProcessGroup 与 StatelessProcessGroup。
    if isinstance(pg, ProcessGroup):
        world_size = torch.distributed.get_world_size(group=pg)
    else:
        world_size = pg.world_size

    # 单卡时节点数固定为 1。
    if world_size == 1:
        return 1

    # ----------------- 根据 rank_mapping 和“是否同机”关系，重建节点划分 -----------------
    # Build node assignment map
    node_assignment = [0] * world_size  # rank -> node_id
    next_node_id = 0

    for current_rank in range(world_size):
        # 已经归属某个节点的 rank 直接跳过。
        if node_assignment[current_rank] != 0:
            continue  # Already assigned to a node

        assert current_rank in rank_mapping
        # scale-down 中标记为 -1 的 rank 表示即将下线，不计入有效节点。
        if rank_mapping[current_rank] == -1:
            continue  # Pending shutdown

        # 给当前 rank 开一个新的 node_id。
        # Assign current rank to a new node
        next_node_id += 1
        node_assignment[current_rank] = next_node_id

        # 再把所有与 current_rank 同机的其他 ranks 一起归入该 node_id。
        # Find all ranks on the same node as current_rank
        same_node_flags = in_the_same_node_as(pg, current_rank)
        for other_rank, is_same_node in enumerate(same_node_flags):
            if is_same_node and node_assignment[other_rank] == 0:
                node_assignment[other_rank] = next_node_id

    # 返回有效节点总数。
    return next_node_id
