# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, overload

import torch
from pydantic import Field, field_validator, model_validator
from torch.distributed import ProcessGroup, ReduceOp, Store
from typing_extensions import Self

import cfie.envs as envs
from cfie.config.utils import config
from cfie.logger import init_logger
from cfie.model_executor.layers.batch_invariant import (
    cfie_is_batch_invariant,
)
from cfie.platforms import current_platform
from cfie.utils.network_utils import get_open_ports_list
from cfie.utils.torch_utils import cuda_device_count_stateless

if TYPE_CHECKING:
    from ray.runtime_env import RuntimeEnv
    from ray.util.placement_group import PlacementGroup

    from cfie.v1.executor import Executor
else:
    RuntimeEnv = Any
    PlacementGroup = Any
    Executor = Any

logger = init_logger(__name__)

# MoE 专家在 rank 间的放置策略。
ExpertPlacementStrategy = Literal["linear", "round_robin"]
# 分布式执行器后端类型。
DistributedExecutorBackend = Literal["ray", "mp", "uni", "external_launcher"]
# 数据并行自身可选的后端类型。
DataParallelBackend = Literal["ray", "mp"]
# 当前 EPLB 仅实现 default 策略。
EPLBPolicyOption = Literal["default"]
# DCP 当前支持的通信实现。
DCPCommBackend = Literal["ag_rs", "a2a"]
# MoE all-to-all 可选通信后端集合。
All2AllBackend = Literal[
    "naive",
    "pplx",
    "deepep_high_throughput",
    "deepep_low_latency",
    "mori",
    "nixl_ep",
    "allgather_reducescatter",
    "flashinfer_all2allv",
]


@config
class EPLBConfig:
    """Configuration for Expert Parallel Load Balancing (EP)."""

    # 统计 expert 负载时使用的滑动窗口长度。
    window_size: int = 1000
    """Window size for expert load recording."""
    # 两次重排 experts 之间间隔多少 step。
    step_interval: int = 3000
    """
    Interval for rearranging experts in expert parallelism.

    Note that if this is greater than the EPLB window size, only the metrics
    of the last `lb_window_size` steps will be used for rearranging experts.
    """

    # 为热门 logical expert 额外复制多少 physical expert 副本。
    num_redundant_experts: int = Field(default=0, ge=0)
    """Number of redundant experts to use for expert parallelism."""

    # 是否输出每一步的负载均衡度日志。
    log_balancedness: bool = False
    """
    Log the balancedness each step of expert parallelism.
    This is turned off by default since it will cause communication overhead.
    """
    # 负载均衡度日志的打印间隔。
    log_balancedness_interval: int = 1
    """
    Interval for logging the balancedness.
    """
    # 是否启用异步 EPLB worker 线程。
    use_async: bool = False
    """
    Whether to use non-blocking EPLB.
    """

    # EPLB 重排策略类型。
    policy: EPLBPolicyOption = "default"
    """The policy type for expert parallel load balancing (EPLB)."""

    @model_validator(mode="after")
    def _validate_eplb_config(self) -> Self:
        # async worker 目前只配套 default policy。
        if self.use_async and self.policy != "default":
            raise ValueError("Async EPLB is only supported with the default policy.")
        # 打开 balancedness 日志时，间隔必须是正数。
        if self.log_balancedness and self.log_balancedness_interval <= 0:
            raise ValueError("log_balancedness_interval must be greater than 0.")
        # 校验完成后返回自身，符合 pydantic after-validator 约定。
        return self


@config
class ParallelConfig:
    """Configuration for the distributed execution."""

    # ----------------- 基础并行维度配置 -----------------
    # PP 维度大小。
    pipeline_parallel_size: int = 1
    """Number of pipeline parallel groups."""
    # TP 维度大小。
    tensor_parallel_size: int = 1
    """Number of tensor parallel groups."""
    # 预填充阶段 PCP 维度大小。
    prefill_context_parallel_size: int = 1
    """Number of prefill context parallel groups."""
    # DP 维度大小。
    data_parallel_size: int = 1
    """Number of data parallel groups. MoE layers will be sharded according to
    the product of the tensor parallel size and data parallel size."""
    # 当前节点内本地 DP 份数。
    data_parallel_size_local: int = 1
    """Number of local data parallel groups."""
    # 当前进程所在 DP rank。
    data_parallel_rank: int = 0
    """Rank of the data parallel group."""
    # SPMD 模式下的本地 DP rank。
    data_parallel_rank_local: int | None = None
    """Local rank of the data parallel group,
    set only in SPMD mode."""
    # DP master IP。
    data_parallel_master_ip: str = "127.0.0.1"
    """IP of the data parallel master."""
    # DP RPC 通信端口。
    data_parallel_rpc_port: int = 29550
    """Port for data parallel messaging."""
    # DP torch.distributed master 端口。
    data_parallel_master_port: int = 29500
    """Port of the data parallel master."""
    # DP 运行后端。
    data_parallel_backend: DataParallelBackend = "mp"
    """Backend to use for data parallel, either "mp" or "ray"."""
    # 是否启用外部负载均衡场景下的 DP 模式。
    data_parallel_external_lb: bool = False
    """Whether to use "external" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. This is useful for a "one-pod-per-rank"
    wide-EP setup in Kubernetes. Set implicitly when --data-parallel-rank
    is provided explicitly to cfie serve."""
    # 是否启用混合负载均衡场景下的 DP 模式。
    data_parallel_hybrid_lb: bool = False
    """Whether to use "hybrid" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. Enables running an AsyncLLM
    and API server on a "per-node" basis where vLLM load balances
    between local data parallel ranks, but an external LB balances
    between vLLM nodes/replicas. Set explicitly in conjunction with
    --data-parallel-start-rank."""
    # 若上层已知模型是否为 MoE，可提前写入该标记。
    is_moe_model: bool | None = None
    """Whether the deployed model is MoE (if known)."""
    # 是否对 MoE 层启用 expert parallel，而不是把专家继续走 TP 切分。
    enable_expert_parallel: bool = False
    """Use expert parallelism instead of tensor parallelism for MoE layers."""
    # 是否启用 EPLB。
    enable_eplb: bool = False
    """Enable expert parallelism load balancing for MoE layers."""
    # EPLB 具体参数。
    eplb_config: EPLBConfig = Field(default_factory=EPLBConfig)
    """Expert parallelism configuration."""
    # expert 在 rank 间的初始放置策略。
    expert_placement_strategy: ExpertPlacementStrategy = "linear"
    """The expert placement strategy for MoE layers:\n
    - "linear": Experts are placed in a contiguous manner. For example, with 4
      experts and 2 ranks, rank 0 will have experts [0, 1] and rank 1 will have
      experts [2, 3].\n
    - "round_robin": Experts are placed in a round-robin manner. For example,
      with 4 experts and 2 ranks, rank 0 will have experts [0, 2] and rank 1
      will have experts [1, 3]. This strategy can help improve load balancing
      for grouped expert models with no redundant experts."""
    # MoE routed tokens 在 EP ranks 间交换时使用的 all-to-all 实现。
    all2all_backend: All2AllBackend = "allgather_reducescatter"
    """All2All backend for MoE expert parallel communication. Available options:

    - "naive": Naive all2all implementation using broadcasts\n
    - "allgather_reducescatter": All2all based on allgather and reducescatter\n
    - "deepep_high_throughput": Use deepep high-throughput kernels\n
    - "deepep_low_latency": Use deepep low-latency kernels\n
    - "mori": Use mori kernels\n
    - "nixl_ep": Use nixl-ep kernels\n
    - "flashinfer_all2allv": Use flashinfer alltoallv kernels for mnnvl"""

    # ----------------- 运行期/加载期附加配置 -----------------
    # 顺序加载大模型时，最多允许多少并行加载 worker。
    max_parallel_loading_workers: int | None = None
    """Maximum number of parallel loading workers when loading model
    sequentially in multiple batches. To avoid RAM OOM when using tensor
    parallel and large models."""

    # 是否禁用自定义 all-reduce kernel。
    disable_custom_all_reduce: bool = False
    """Disable the custom all-reduce kernel and fall back to NCCL."""

    # 是否启用 elastic EP。
    enable_elastic_ep: bool = False
    """Enable elastic expert parallelism with stateless NCCL groups for DP/EP."""

    # 是否启用 DBO。
    enable_dbo: bool = False
    """Enable dual batch overlap for the model executor."""
    # 手工指定的 ubatch 大小。
    ubatch_size: int = 0
    """Number of ubatch size."""

    # decode-only 场景下触发 DBO 的 token 阈值。
    dbo_decode_token_threshold: int = 32
    """The threshold for dual batch overlap for batches only containing decodes.
    If the number of tokens in the request is greater than this threshold,
    microbatching will be used. Otherwise, the request will be processed in a
    single batch."""
    # 含 prefill 场景下触发 DBO 的 token 阈值。
    dbo_prefill_token_threshold: int = 512  # TODO(lucas): tune
    """The threshold for dual batch overlap for batches that contain one or more
    prefills. If the number of tokens in the request is greater than this
    threshold, microbatching will be used. Otherwise, the request will be
    processed in a single batch."""

    # 是否强制 DP 同步改走 Gloo。
    disable_nccl_for_dp_synchronization: bool | None = Field(default=None)
    """Forces the dp synchronization logic in cfie/v1/worker/dp_utils.py 
    to use Gloo instead of NCCL for its all reduce.

    Defaults to True when async scheduling is enabled, False otherwise.
    """

    # 是否用 nsight profile Ray workers。
    ray_workers_use_nsight: bool = False
    """Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler."""

    # Ray runtime environment。
    ray_runtime_env: RuntimeEnv | None = None
    """Ray runtime environment to pass to distributed workers."""

    # Ray placement group。
    placement_group: PlacementGroup | None = None
    """ray distributed model workers placement group."""

    # 分布式执行器后端，既可以是字符串，也可以是 Executor 子类。
    distributed_executor_backend: (
            str | DistributedExecutorBackend | type[Executor] | None
    ) = None
    """Backend to use for distributed model workers, either "ray" or "mp"
    (multiprocessing). If the product of pipeline_parallel_size and tensor_parallel_size
    is less than or equal to the number of GPUs available, "mp" will be used to
    keep processing on a single host. Otherwise, an error will be raised. To use "mp"
    you must also set nnodes, and to use "ray" you must manually set
    distributed_executor_backend to "ray".

    Note that tpu only support Ray for distributed inference."""

    # 主 worker 类名。
    worker_cls: str = "auto"
    """The full name of the worker class to use. If "auto", the worker class
    will be determined based on the platform."""
    # speculative decoding worker 类名。
    sd_worker_cls: str = "auto"
    """The full name of the worker class to use for speculative decoding.
    If "auto", the worker class will be determined based on the platform."""
    # 注入到 worker 上的扩展类名。
    worker_extension_cls: str = ""
    """The full name of the worker extension class to use. The worker extension
    class is dynamically inherited by the worker class. This is used to inject
    new attributes and methods to the worker class for use in collective_rpc
    calls."""
    # 多机 mp 模式下的 master 地址。
    master_addr: str = "127.0.0.1"
    """distributed master address for multi-node distributed 
    inference when distributed_executor_backend is mp."""
    # 多机 mp 模式下的 master 端口。
    master_port: int = 29501
    """distributed master port for multi-node distributed 
    inference when distributed_executor_backend is mp."""
    # 当前节点的 node rank。
    node_rank: int = 0
    """distributed node rank for multi-node distributed 
    inference when distributed_executor_backend is mp."""
    # 总节点数。
    nnodes: int = 1
    """num of nodes for multi-node distributed
    inference when distributed_executor_backend is mp."""

    # 分布式操作超时时间。
    distributed_timeout_seconds: int | None = None
    """Timeout in seconds for distributed operations (e.g., init_process_group).
    If set, this value is passed to torch.distributed.init_process_group as the
    timeout parameter. If None, PyTorch's default timeout is used (600s for NCCL).
    Increase this for multi-node setups where model downloads may be slow."""

    # world_size 只统计 TP*PP*PCP，不含 DP。
    world_size: int = Field(init=False)
    """world_size is TPxPP, it affects the number of workers we create."""

    # 当前全局 rank。
    rank: int = 0
    """Global rank in distributed setup."""

    # 为多进程初始化 DP 组提前预留的一批端口。
    _data_parallel_master_port_list: list[int] = Field(default_factory=list)
    """List of open port auto-queried for data parallel messaging.
    Set to be private as it's not intended to be configured by users.
    """

    # elastic EP 下 stateless DP group 的端口集合。
    _stateless_dp_group_port_list: list[list[int]] = Field(default_factory=list)
    """List of open ports for stateless DP groups when enable_elastic_ep is True.
    Set to be private as it's not intended to be configured by users.
    It is a list of list[int], with each inner list contains a set of 3 ports
    to be used for setting up the stateless CPU/device/TCPStore groups
    in StatelessGroupCoordinator. The number of inner lists is equal to
    the number of DP groups, 
    i.e., len(self._stateless_dp_group_port_list) == world_size_across_dp // dp_size,
    and len(self._stateless_dp_group_port_list[i]) == 3 for all i.
    """

    # elastic EP 下 stateless EP group 的端口集合。
    _stateless_ep_group_port_list: list[list[int]] = Field(default_factory=list)
    """List of open ports for stateless EP groups when enable_elastic_ep is True.
    Set to be private as it's not intended to be configured by users.
    len(self._stateless_ep_group_port_list) == world_size_across_dp // ep_size,
    """

    # elastic EP 下 stateless EPLB group 的端口集合。
    _stateless_eplb_group_port_list: list[list[int]] = Field(default_factory=list)
    """List of open ports for stateless EPLB groups when enable_elastic_ep is True.
    Same topology as EP but separate NCCL communicator to avoid deadlocks.
    """

    # elastic EP 下 stateless world group 的端口集合。
    _stateless_world_group_port_list: list[list[int]] = Field(default_factory=list)
    """List of open ports for stateless world group when enable_elastic_ep is True.
    Set to be private as it's not intended to be configured by users.
    len(self._stateless_world_group_port_list) == 1,
    """

    # DCP 维度大小。
    decode_context_parallel_size: int = 1
    """Number of decode context parallel groups, because the world size does
    not change by dcp, it simply reuse the GPUs of TP group, and tp_size
    needs to be divisible by dcp_size."""

    # 兼容旧字段名的 DCP interleave 配置。
    dcp_kv_cache_interleave_size: int = 1
    """
    Interleave size of kv_cache storage while using DCP.
    dcp_kv_cache_interleave_size has been replaced by cp_kv_cache_interleave_size,
    and will be deprecated when PCP is fully supported.

    """
    # DCP 使用的通信后端。
    dcp_comm_backend: DCPCommBackend = "ag_rs"
    """Communication backend for Decode Context Parallel (DCP).
    - "ag_rs": AllGather + ReduceScatter (default, existing behavior)
    - "a2a": All-to-All exchange of partial outputs + LSE, then
      combine with Triton kernel. Reduces NCCL calls from 3 to 2
      per layer for MLA models.
    """

    # PCP/DCP 共享的 KV interleave 配置。
    cp_kv_cache_interleave_size: int = 1
    """Interleave size of kv_cache storage while using DCP or PCP.
    For `total_cp_rank = pcp_rank * dcp_world_size + dcp_rank`,
        and `total_cp_world_size = pcp_world_size * dcp_world_size`.
    store interleave_size tokens on total_cp_rank i,
    then store next interleave_size tokens on total_cp_rank i+1.
    Interleave_size=1: token-level alignment, where token `i` is stored on
        total_cp_rank `i % total_cp_world_size`.
    Interleave_size=block_size: block-level alignment, where tokens are
        first populated to the preceding ranks. Tokens are then stored
        in (rank i+1, block j) only after (rank i, block j) is fully occupied.
    Block_size should be greater than or equal to cp_kv_cache_interleave_size.
    Block_size should be divisible by cp_kv_cache_interleave_size.
    """

    # data_parallel_index 与 data_parallel_rank 数值相同，但语义上不参与 torch group 覆写。
    data_parallel_index: int = Field(init=False)
    """Equal to the data parallel rank but not used for torch process groups
    and not overridden for dense models."""

    # API scale-out 场景下的 API 进程总数。
    _api_process_count: int = Field(default=1, gt=0)
    """
    The number of API processes initialized.

    Note:
        This is an internal config that is only valid for and
        should only be set by API server scale-out.
    """

    # API scale-out 场景下当前 API 进程编号。
    _api_process_rank: int = Field(default=0, ge=-1)
    """
    The rank of this API process, or `-1` for engine core processes
    under API server scale-out.

    Note:
        This is an internal config that is only valid for and
        should only be set by API server scale-out.
    """

    @field_validator("disable_nccl_for_dp_synchronization", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialisation is delayed."""
        # 某些路径会延后再决定该字段；此时保留 None，不走常规验证。
        return None if value is None else handler(value)

    @model_validator(mode="after")
    def _validate_parallel_config(self) -> Self:
        # ----------------- 先做 API scale-out 参数合法性校验 -----------------
        if self._api_process_rank >= self._api_process_count:
            raise ValueError(
                "Invalid value of `_api_process_rank`. "
                f"Expected to be `-1` or `[0, {self._api_process_count})`, "
                f"but found: {self._api_process_rank}"
            )

        # pplx 已被移除；这里向后兼容并自动回退到当前默认后端。
        if self.all2all_backend == "pplx":
            logger.warning(
                "The 'pplx' all2all backend has been removed. "
                "Falling back to 'allgather_reducescatter'."
            )
            self.all2all_backend = "allgather_reducescatter"

        # 本地 DP 份数不能超过总 DP 份数。
        if self.data_parallel_size_local > self.data_parallel_size:
            raise ValueError(
                f"data_parallel_size_local ({self.data_parallel_size_local}) "
                f"must be <= data_parallel_size ({self.data_parallel_size})"
            )

        # 外部 LB 只在真正启用 DP 时有意义。
        if self.data_parallel_size <= 1 and self.data_parallel_external_lb:
            raise ValueError(
                "data_parallel_external_lb can only be set when data_parallel_size > 1"
            )

        # ----------------- 再校验 EP/EPLB 相关约束 -----------------
        if self.enable_eplb:
            # EPLB 目前只支持 CUDA-like 平台。
            if not current_platform.is_cuda_alike():
                raise ValueError(
                    "Expert parallelism load balancing is only supported on "
                    "CUDA devices or ROCm devices now."
                )
            # EPLB 必须依附在 expert parallel 之上。
            if not self.enable_expert_parallel:
                raise ValueError("enable_expert_parallel must be True to use EPLB.")
            # 至少要有 TP 或 DP 的一个维度大于 1，否则 EPLB 没有分布式拓扑可调。
            if self.tensor_parallel_size * self.data_parallel_size <= 1:
                raise ValueError(
                    "EPLB requires tensor_parallel_size or data_parallel_size "
                    f"to be greater than 1, but got "
                    f"TP={self.tensor_parallel_size},DP={self.data_parallel_size}."
                )
        else:
            # 未启用 EPLB 时，不允许残留 redundant experts 配置。
            if self.eplb_config.num_redundant_experts != 0:
                raise ValueError(
                    "num_redundant_experts is set to "
                    f"{self.eplb_config.num_redundant_experts} but EPLB is not "
                    "enabled. Either enable EPLB or unset "
                    "num_redundant_experts."
                )

        # ----------------- 最后校验 DCP 本身的拓扑约束 -----------------
        # Note(hc): In the current implementation of decode context
        # parallel(DCP), tp_size needs to be divisible by dcp_size,
        # because the world size does not change by dcp, it simply
        # reuses the GPUs of TP group, and split one TP group into
        # tp_size//dcp_size DCP groups.
        if self.tensor_parallel_size % self.decode_context_parallel_size != 0:
            raise ValueError(
                f"tp_size={self.tensor_parallel_size} must be divisible by"
                f"dcp_size={self.decode_context_parallel_size}."
            )

        # a2a 版 DCP 至少要有 2 个 decode context ranks。
        if self.dcp_comm_backend == "a2a" and self.decode_context_parallel_size <= 1:
            raise ValueError(
                "dcp_comm_backend='a2a' requires decode_context_parallel_size > 1."
            )

        # after-validator 约定返回 self。
        return self

    @property
    def world_size_across_dp(self) -> int:
        """world_size_across_dp is TPxPPxDP, it is the size of the world
        including data parallelism."""
        # 在 world_size 的基础上再乘进 DP 维度。
        return self.world_size * self.data_parallel_size

    @property
    def use_ubatching(self) -> bool:
        # 开启 DBO 或显式指定 ubatch_size>1 时，都视为启用 ubatching。
        return self.enable_dbo or self.ubatch_size > 1

    @property
    def num_ubatches(self) -> int:
        # DBO 固定拆成 2 个 ubatch；否则沿用用户指定值。
        return 2 if self.enable_dbo else self.ubatch_size

    @property
    def local_engines_only(self) -> bool:
        """
        是否只管理本地 EngineCore。

        在混合负载均衡和外部负载均衡场景下返回 True；
        在纯内部负载均衡场景下返回 False，此时 Client 会同时管理本地和远端的 EngineCore。
        """
        return self.data_parallel_external_lb or self.data_parallel_hybrid_lb

    def get_next_dp_init_port(self) -> int:
        """
        We might need to initialize process groups in multiple
        processes that is related to data parallelism,
        e.g. both in the worker and in the engine, which
        can live in different processes. To avoid port conflicts, we
        pop a new port from the prepared port list each time we need to
        initialize a new process group related to data parallelism.
        """
        # 优先消费预先申请好的端口，避免多个进程现场抢同一个空闲端口。
        if self._data_parallel_master_port_list:
            answer = self._data_parallell_master_port_list.pop()
        else:
            # 若预留端口耗尽，则退化成在当前 master_port 基础上自增分配。
            answer = self.data_paralel_master_port
            self.data_parallel_master_port += 1

        # 返回本次真正使用的 DP 初始化端口。
        return answer

    def allocate_elastic_ep_ports(self) -> None:
        """Allocate all ports for elastic EP (stateless groups + DP master).

        Must be called AFTER ray.init() so that ports claimed by Ray's
        idle worker pool are already in use and won't be returned by
        get_open_ports_list().
        """
        # 未启用 elastic EP 时，不需要预分配任何 stateless group 端口。
        if not self.enable_elastic_ep:
            return
        # 已经分配过一次时直接复用，避免重复覆盖。
        if self._stateless_world_group_port_list:
            return

        # world group 固定只有 1 组。
        num_world_groups = 1
        # DP 组大小就是 data_parallel_size。
        dp_size = self.data_parallel_size
        # 这里的 EP 组大小使用整个 DP 内 world 的总规模。
        ep_size = self.data_parallel_size * self.world_size_across_dp
        # DP 组个数等于总 world 除以单组 DP 大小。
        num_dp_groups = max(1, self.world_size_across_dp // dp_size)
        # EP 组个数按总 world 除以单组 EP 大小计算。
        num_ep_groups = max(1, self.world_size_across_dp // ep_size)
        # EPLB group 拓扑与 EP 相同，但 communicator 独立。
        num_eplb_groups = num_ep_groups
        # StatelessGroupCoordinator 每组都需要 3 个端口。
        total_stateless_ports = (
                                        num_world_groups + num_dp_groups + num_ep_groups + num_eplb_groups
                                ) * 3
        # 额外再为 DP master 预留 5 个端口。
        num_dp_master_ports = 5

        # 一次性申请所有端口，减少并发场景下端口冲突概率。
        all_ports = get_open_ports_list(total_stateless_ports + num_dp_master_ports)

        # 最后 5 个端口分配给 DP master 复用池。
        self._data_parallel_master_port_list = all_ports[-num_dp_master_ports:]
        self.data_parallel_master_port = self._data_parallel_master_port_list.pop()
        all_ports = all_ports[:-num_dp_master_ports]

        # 前 3 个端口给 stateless world group。
        self._stateless_world_group_port_list = [
            all_ports[i: i + 3] for i in range(0, num_world_groups * 3, 3)
        ]
        start_idx = num_world_groups * 3
        # 接下来切出每个 stateless DP group 的端口三元组。
        self._stateless_dp_group_port_list = [
            all_ports[i: i + 3]
            for i in range(start_idx, start_idx + num_dp_groups * 3, 3)
        ]
        start_idx += num_dp_groups * 3
        # 再切出 stateless EP groups。
        self._stateless_ep_group_port_list = [
            all_ports[i: i + 3]
            for i in range(start_idx, start_idx + num_ep_groups * 3, 3)
        ]
        start_idx += num_ep_groups * 3
        # 最后切出独立的 stateless EPLB groups。
        self._stateless_eplb_group_port_list = [
            all_ports[i: i + 3]
            for i in range(start_idx, start_idx + num_eplb_groups * 3, 3)
        ]

    def get_next_stateless_world_group_port(self) -> list[int]:
        # 取出下一组 stateless world group 端口。
        return self._stateless_world_group_port_list.pop()

    def get_next_stateless_dp_group_port(self) -> list[int]:
        # 取出下一组 stateless DP group 端口。
        return self._stateless_dp_group_port_list.pop()

    def get_next_stateless_ep_group_port(self) -> list[int]:
        # 取出下一组 stateless EP group 端口。
        return self._stateless_ep_group_port_list.pop()

    def get_next_stateless_eplb_group_port(self) -> list[int]:
        # 取出下一组 stateless EPLB group 端口。
        return self._stateless_eplb_group_port_list.pop()

    @overload
    def stateless_init_dp_group(
            self, return_store: Literal[False] = ...
    ) -> ProcessGroup:
        ...

    @overload
    def stateless_init_dp_group(
            self, return_store: Literal[True] = ...
    ) -> tuple[ProcessGroup, Store]:
        ...

    def stateless_init_dp_group(
            self, return_store: bool = False
    ) -> ProcessGroup | tuple[ProcessGroup, Store]:
        # ----------------- 初始化 stateless DP group，并对端口冲突做重试 -----------------
        # NOTE: In high-concurrency scenarios multiple processes
        # can pick the same (currently free) port through a race
        # condition when calling `get_open_port()`. When the first
        # process binds the port the others will subsequently fail
        # with `torch.distributed.DistNetworkError: EADDRINUSE`.
        # To make the initialization more robust we retry a few times
        # with a fresh port whenever this specific error is observed.
        from torch.distributed import DistNetworkError

        from cfie.distributed.utils import (
            stateless_init_torch_distributed_process_group,
        )

        # 最多重试 5 次，专门兜底 EADDRINUSE 这类端口竞争问题。
        max_retries = 5
        last_exc: Exception | None = None
        for _ in range(max_retries):
            try:
                # 这里固定用 gloo，因为 engine 进程未必持有可用 CUDA device。
                # use gloo since the engine process might not have cuda device
                return stateless_init_torch_distributed_process_group(
                    self.data_parallel_master_ip,
                    self.get_next_dp_init_port(),
                    self.data_parallel_rank,
                    self.data_parallel_size,
                    backend="gloo",
                    return_store=return_store,
                )
            except DistNetworkError as e:
                # 只有真正的端口占用冲突才重试，其他网络错误直接继续向上抛。
                # We only want to retry when the root cause is EADDRINUSE.
                if "EADDRINUSE" in str(e):
                    logger.warning("Address already in use. Retrying with a new port.")
                    last_exc = e
                    continue  # try again with a new port
                raise e

        # If we get here all retries have failed.
        assert last_exc is not None
        raise last_exc

    # The all_reduce at the end of attention (during o_proj) means that
    # inputs are replicated across each rank of the tensor parallel group.
    # If using expert-parallelism with DeepEP All2All ops, replicated
    # tokens results in useless duplicate computation and communication.
    #
    # In this case, ensure the input to the experts is sequence parallel
    # to avoid the excess work.
    #
    @property
    def use_sequence_parallel_moe(self) -> bool:
        # 只有支持 sequence-parallel 输入布局的 all2all backend 才需要这条路径。
        return (
                self.all2all_backend
                in (
                    "allgather_reducescatter",
                    "naive",
                    "deepep_high_throughput",
                    "deepep_low_latency",
                    "mori",
                    "nixl_ep",
                )
                and self.enable_expert_parallel
                and self.tensor_parallel_size > 1
                and self.data_parallel_size > 1
        )

    @property
    def node_rank_within_dp(self) -> int:
        # 把全局 node_rank 折叠到当前 DP 子拓扑内部。
        return self.node_rank % self.nnodes_within_dp

    @property
    def nnodes_within_dp(self) -> int:
        # 单机时 DP 子拓扑天然只包含 1 个节点。
        if self.nnodes == 1:
            return 1
        # 先算出一个 DP 组跨多少节点，再反推出每个 DP 子拓扑内部节点数。
        data_parallel_node_size = (
                self.data_parallel_size // self.data_parallel_size_local
        )
        return self.nnodes // data_parallel_node_size

    @property
    def local_world_size(self) -> int:
        # local_world_size 表示单个 DP 子拓扑内部，每个节点持有多少 worker。
        return self.world_size // self.nnodes_within_dp

    @staticmethod
    def has_unfinished_dp(dp_group: ProcessGroup, has_unfinished: bool) -> bool:
        # 先把布尔值映射成 CPU int32 tensor，便于走 all_reduce。
        tensor = torch.tensor([has_unfinished], dtype=torch.int32, device="cpu")
        # dp rank 0: has_unfinished_seqs=True
        # dp rank 1: has_unfinished_seqs=False
        # aggregated: has_unfinished_seqs=True
        # so this is an OR operation, i.e. MAX in integers
        # 用 MAX 聚合所有 DP ranks 的 unfinished 状态，相当于逻辑或。
        torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
        aggregated_has_unfinished = bool(tensor.item())
        # 返回整个 DP 组内是否仍有未完成序列。
        return aggregated_has_unfinished

    @staticmethod
    def sync_kv_cache_memory_size(dp_group: ProcessGroup, kv_cache_memory: int) -> int:
        # -1 代表“不设上限”，这里转成 int64 最大值，便于后面取 MIN。
        if kv_cache_memory == -1:
            kv_cache_memory = torch.iinfo(torch.int64).max
        tensor = torch.tensor([kv_cache_memory], dtype=torch.int64, device="cpu")
        # we cannot use broadcast for stateless dp group since it depends
        # on global rank
        # DP ranks 取最小值，确保所有副本都按最保守的 KV cache 容量对齐。
        torch.distributed.all_reduce(tensor, op=ReduceOp.MIN, group=dp_group)
        return tensor.item()

    def compute_hash(self):
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.

        This hash is also used for DP worker configuration validation
        to prevent hangs from mismatched collective communication patterns.
        """
        # 这些字段属于运行时拓扑、端口或拉起方式差异，不应影响计算图结构哈希。
        ignored_factors = {
            # Derived/runtime topology, networking, or launch details
            "data_parallel_rank",
            "data_parallel_rank_local",
            "data_parallel_size_local",
            "data_parallel_index",
            "data_parallel_backend",
            "data_parallel_external_lb",
            "data_parallel_hybrid_lb",
            "data_parallel_master_ip",
            "data_parallel_master_port",
            "_data_parallel_master_port_list",
            "data_parallel_rpc_port",
            "rank",
            "master_addr",
            "master_port",
            "node_rank",
            "nnodes",
            "max_parallel_loading_workers",
            "disable_custom_all_reduce",
            "ray_workers_use_nsight",
            "ray_runtime_env",
            "placement_group",
            "distributed_executor_backend",
            "worker_cls",
            "sd_worker_cls",
            "worker_extension_cls",
            "_api_process_count",
            "_api_process_rank",
        }

        from cfie.config.utils import get_hash_factors, hash_factors

        # 只提取真正影响计算图与 collective 模式的字段参与哈希。
        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    def __post_init__(self) -> None:
        # ----------------- 先根据 PP/TP/PCP 计算基础 world_size -----------------
        # Continue with the rest of the initialization
        self.world_size = (
                self.pipeline_parallel_size
                * self.tensor_parallel_size
                * self.prefill_context_parallel_size
        )

        # external launcher 会把 DP ranks 也直接铺进总 world 中。
        if self.distributed_executor_backend == "external_launcher":
            logger.info("Using external launcher for distributed inference.")
            self.world_size *= self.data_parallel_size

        # ----------------- 校验 elastic EP 与其它模式的兼容性 -----------------
        if self.enable_elastic_ep:
            if not self.enable_eplb:
                raise ValueError("Elastic EP is only supported with enable_eplb=True.")
            if self.pipeline_parallel_size > 1:
                raise ValueError(
                    "Elastic EP is not supported with pipeline parallelism "
                    f"(pipeline_parallel_size={self.pipeline_parallel_size})."
                )
            if self.data_parallel_external_lb or self.data_parallel_hybrid_lb:
                raise NotImplementedError(
                    "Elastic EP is not compatible with data_parallel_external_lb "
                    "or data_parallel_hybrid_lb. Elastic EP relies on a single API "
                    "server and core client to coordinate scale up/down."
                )

        # ----------------- 处理在线 DP 显式配置场景 -----------------
        if self.data_parallel_size > 1 or self.data_parallel_size_local == 0:
            # Data parallel was specified in the engine args.
            if self.distributed_executor_backend == "external_launcher":
                # external launcher 模式下，DP rank 由全局 RANK 自动折算。
                # For external launcher,
                # we need to set the data parallel rank automatically
                self.data_parallel_rank = int(os.environ["RANK"]) // (
                        self.world_size // self.data_parallel_size
                )
                logger.info(
                    "Set data_parallel_rank to %d automatically.",
                    self.data_parallel_rank,
                )
            if not self.enable_elastic_ep:
                # 非 elastic EP 场景下，提前为 DP 相关组初始化准备端口池。
                if not self._data_parallel_master_port_list:
                    self._data_parallel_master_port_list = get_open_ports_list(5)
                self.data_parallel_master_port = (
                    self._data_parallel_master_port_list.pop()
                )

            # 显式指定的 DP rank 必须落在合法范围内。
            if not (0 <= self.data_parallel_rank < self.data_parallel_size):
                raise ValueError(
                    f"data_parallel_rank ({self.data_parallel_rank})"
                    f" must be in the range [0, {self.data_parallel_size})"
                )
        else:
            # ----------------- 否则退回离线 SPMD 环境变量配置 -----------------
            # Otherwise fall back to env vars (e.g. for offline SPMD case).
            self.data_parallel_size = envs.VLLM_DP_SIZE
            self.data_parallel_rank = envs.VLLM_DP_RANK
            self.data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL
            self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
            self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT

            if self.data_parallel_size > 1 and self.is_moe_model is False:
                raise ValueError(
                    "Offline data parallel mode is not supported/useful"
                    " for dense models."
                )

        # data_parallel_index 默认与 data_parallel_rank 对齐。
        self.data_parallel_index = self.data_parallel_rank

        # external launcher 已经由外部统一管理多进程，因此关掉 v1 自己的 multiprocessing。
        if self.distributed_executor_backend == "external_launcher":
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("Disabling V1 multiprocessing for external launcher.")

        # ----------------- 若用户未显式指定后端，则按当前平台与拓扑自动推断 -----------------
        if self.distributed_executor_backend is None and self.world_size_across_dp > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.

            from cfie.v1.executor import ray_utils

            backend: DistributedExecutorBackend = "mp"
            ray_found = ray_utils.ray_is_available()
            # TPU + XLA SPMD 优先走 uni。
            if current_platform.is_tpu() and envs.VLLM_XLA_USE_SPMD:
                backend = "uni"
            # 多机 CUDA 目前默认仍走 mp。
            elif current_platform.is_cuda() and self.nnodes > 1:
                backend = "mp"
            elif (
                    current_platform.is_cuda()
                    and cuda_device_count_stateless() < self.world_size
            ):
                # 单机可见 GPU 不够时，直接 fail fast，提示用户改用 ray 或正确设置 nnodes。
                gpu_count = cuda_device_count_stateless()
                raise ValueError(
                    f"World size ({self.world_size}) is larger than the number of "
                    f"available GPUs ({gpu_count}) in this node. If this is "
                    "intentional and you are using:\n"
                    "- ray, set '--distributed-executor-backend ray'.\n"
                    "- multiprocessing, set '--nnodes' appropriately."
                )
            # 用户显式要求 DP 走 ray 时，整体执行器也切到 ray。
            elif self.data_parallel_backend == "ray":
                logger.info(
                    "Using ray distributed inference because "
                    "data_parallel_backend is ray"
                )
                backend = "ray"
            elif ray_found:
                # 若运行在 placement group 或 Ray 上下文里，也自动切到 ray。
                if self.placement_group:
                    backend = "ray"
                else:
                    from ray import is_initialized as ray_is_initialized

                    if ray_is_initialized():
                        from ray.util import get_current_placement_group

                        if get_current_placement_group():
                            backend = "ray"
            self.distributed_executor_backend = backend
            logger.debug("Defaulting to use %s for distributed inference", backend)

        # 单卡单进程场景下，未指定后端就退回 uni。
        if self.distributed_executor_backend is None and self.world_size == 1:
            self.distributed_executor_backend = "uni"

        # 该选项当前未真正接入，因此保留 warning。
        if self.max_parallel_loading_workers is not None:
            logger.warning(
                "max_parallel_loading_workers is currently "
                "not supported and will be ignored."
            )
        allowed_backends = ("mp", "uni", "external_launcher")
        # 多机场景下只允许这些显式后端。
        if (
                self.distributed_executor_backend not in allowed_backends
                and self.nnodes > 1
        ):
            raise ValueError(
                "nnodes > 1 can only be set when distributed executor "
                "backend is mp, uni or external_launcher."
            )

        # 某些 all2all backend 与 async EPLB 组合会死锁，这里主动降级成同步。
        if (
                self.all2all_backend in ("allgather_reducescatter", "naive")
                and self.eplb_config.use_async
        ):
            logger.warning(
                "Async EPLB causes hangs with the '%s' all2all backend. "
                "Forcing synchronous EPLB.",
                self.all2all_backend,
            )
            self.eplb_config.use_async = False

    @property
    def use_ray(self) -> bool:
        # 除了字符串 "ray"，自定义 Executor 也可能通过 uses_ray 声明依赖 Ray。
        return self.distributed_executor_backend == "ray" or (
                isinstance(self.distributed_executor_backend, type)
                and getattr(self.distributed_executor_backend, "uses_ray", False)
        )

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        # ----------------- 做最终后端类型与平台能力校验 -----------------
        # Lazy import to avoid circular import
        from cfie.v1.executor import Executor

        # batch-invariant 模式下，直接禁用 custom all-reduce。
        # Enable batch invariance settings if requested
        if cfie_is_batch_invariant():
            self.disable_custom_all_reduce = True

        # distributed_executor_backend 若不是字符串，就必须是 Executor 子类。
        if (
                self.distributed_executor_backend is not None
                and not isinstance(self.distributed_executor_backend, str)
                and not (
                isinstance(self.distributed_executor_backend, type)
                and issubclass(self.distributed_executor_backend, Executor)
        )
        ):
            raise ValueError(
                "Unrecognized distributed executor backend "
                f"{self.distributed_executor_backend}. Supported "
                "values are 'ray', 'mp' 'uni', 'external_launcher', "
                " custom Executor subclass or its import path."
            )
        if self.use_ray:
            # 既然最终决定走 Ray，就强制确认 Ray 依赖可用。
            from cfie.v1.executor import ray_utils

            ray_utils.assert_ray_available()

        # 当前平台不支持 custom all-reduce 时，自动禁用。
        if not current_platform.use_custom_allreduce():
            self.disable_custom_all_reduce = True
            logger.debug(
                "Disabled the custom all-reduce kernel because it is not "
                "supported on current platform."
            )
        # 多机场景下同样不启用 custom all-reduce。
        if self.nnodes > 1:
            self.disable_custom_all_reduce = True
            logger.debug(
                "Disabled the custom all-reduce since we are running on multi-node."
            )
        # nsight profiling 仅在 Ray worker 模式下可用。
        if self.ray_workers_use_nsight and not self.use_ray:
            raise ValueError(
                "Unable to use nsight profiling unless workers run with Ray."
            )

        # 返回最终校验完成的配置对象。
        return self
