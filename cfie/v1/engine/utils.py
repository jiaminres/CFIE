# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os
import threading
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing import Process, connection
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING
from unittest.mock import patch

import msgspec
import zmq

from cfie import envs
from cfie.config import CacheConfig, ParallelConfig, CfieConfig
from cfie.logger import init_logger
from cfie.platforms import current_platform
from cfie.ray.ray_env import get_env_vars_to_copy
from cfie.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from cfie.utils.system_utils import get_mp_context
from cfie.v1.engine.coordinator import DPCoordinator
from cfie.v1.executor import Executor
from cfie.v1.utils import get_engine_client_zmq_addr, shutdown

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

STARTUP_POLL_PERIOD_MS = 10000


class CoreEngineState(Enum):
    NEW = auto()
    CONNECTED = auto()
    READY = auto()


class CoreEngine:
    """One per data parallel rank, used to track state during handshaking."""

    def __init__(self, index: int = 0, local: bool = True):
        self.local = local
        self.identity = index.to_bytes(2, "little")

        self.state = CoreEngineState.NEW


@dataclass
class EngineZmqAddresses:
    # ZMQ input socket addresses for each front-end client (requests)
    inputs: list[str]
    # ZMQ output socket addresses for each front-end client (responses)
    outputs: list[str]
    # ZMQ input socket address of DP coordinator if applicable
    coordinator_input: str | None = None
    # ZMQ output socket address of DP coordinator if applicable
    coordinator_output: str | None = None
    # ZMQ socket for front-end to connect to DP coordinator.
    # Not used by engine, just relayed to front-end in handshake response.
    # Only required for external DP LB case.
    frontend_stats_publish_address: str | None = None


@dataclass
class EngineHandshakeMetadata:
    """Metadata sent to each engine process during startup handshake,
    including addresses of the front-end ZMQ queues that they should
    connect to.
    """

    addresses: EngineZmqAddresses
    parallel_config: dict[str, int | str | list[int]]


class CoreEngineProcManager:
    """托管本地 `EngineCore` 后台进程集合的轻量管理器。

    它只负责进程级生命周期管理，不参与调度或模型执行本身：

    - 按给定的 DP rank 范围拉起本地 `EngineCoreProc`
    - 在启动阶段跟踪这些子进程是否成功存活
    - 在失败或退出时统一执行 shutdown / join / 状态查询

    上层 `MPClient` / `launch_core_engines()` 会用它来持有
    “当前前端负责管理的那一批本地 engine 进程”。
    """

    # ------------------------------- 拉起并托管本地 EngineCore 进程 -------------------------------
    def __init__(
        self,
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        cfie_config: CfieConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
    ):
        # 统一取多进程上下文，确保后面创建出来的所有 engine 进程共享同一启动语义。
        context = get_mp_context()

        # 这批参数对当前 manager 管理的所有本地 engine 都一致，
        # 后面只会按具体 DP rank 再补上各自独有的 rank 信息。
        common_kwargs = {
            "cfie_config": cfie_config,
            "local_client": local_client,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
        }

        # 某些 DP 拓扑下，client 与 engine 的握手地址需要拆分；
        # 只有传入时才额外注入，避免默认路径携带无意义配置。
        if client_handshake_address:
            common_kwargs["client_handshake_address"] = client_handshake_address

        # 是否处于 DP 场景会影响子进程命名和部分设备控制策略。
        is_dp = cfie_config.parallel_config.data_parallel_size > 1

        from cfie.v1.engine.core import EngineCoreProc

        # `processes` 保存当前 manager 持有的所有后台 engine 进程句柄。
        self.processes: list[BaseProcess] = []
        # `local_dp_ranks` 与 `processes` 一一对应，后面启动时会用到。
        local_dp_ranks = []
        for index in range(local_engine_count):
            # `local_index` 是当前节点内部的局部 DP 序号。
            local_index = local_start_index + index
            # `global_index` 是全局 DP 视角下的 engine rank。
            global_index = start_index + index

            # ------------------------------- 逐个构造后台 EngineCoreProc -------------------------------
            # 这里先只创建 `Process` 对象，真正 `start()` 放到后面统一执行，
            # 便于在全部参数就绪后再进入启动阶段。
            local_dp_ranks.append(local_index)
            self.processes.append(
                context.Process(
                    target=EngineCoreProc.run_engine_core,
                    name=f"EngineCore_DP{global_index}" if is_dp else "EngineCore",
                    kwargs=common_kwargs
                    | {"dp_rank": global_index, "local_dp_rank": local_index},
                )
            )

        # 给 manager 自己也挂一个终结器，防止外层忘记显式 shutdown 时遗留子进程。
        self._finalizer = weakref.finalize(self, shutdown, self.processes)

        try:
            for proc, local_dp_rank in zip(self.processes, local_dp_ranks):
                # ------------------------------- 启动子进程并处理平台差异 -------------------------------
                # 在非 CUDA 平台、外部 launcher 或 Ray 路径下，
                # 设备可见性往往不能依赖 CUDA 默认语义，因此要临时注入设备控制环境变量。
                if is_dp and (
                    not current_platform.is_cuda_alike()
                    or cfie_config.parallel_config.use_ray
                ):
                    with set_device_control_env_var(cfie_config, local_dp_rank):
                        proc.start()
                else:
                    proc.start()
        finally:
            # 只要发现某些子进程已经异常结束，就说明本轮启动不完整；
            # 此时直接把整批进程回收掉，避免留下半活状态的 engine 集合。
            if self.finished_procs():
                self.shutdown()

    # ------------------------------- 按给定超时关闭整批 EngineCore 进程 -------------------------------
    def shutdown(self, timeout: float | None = None) -> None:
        # 只有在终结器尚未被消费时，才真正执行一次关闭，避免重复 shutdown。
        if self._finalizer.detach() is not None:
            shutdown(self.processes, timeout=timeout)

    # ------------------------------- 等待任意一个 EngineCore 子进程退出 -------------------------------
    def join_first(self):
        # 该方法常用于监控线程阻塞等待“是否有子进程先异常退出”。
        connection.wait(proc.sentinel for proc in self.processes)

    # ------------------------------- 导出全部子进程 sentinel -------------------------------
    def sentinels(self) -> list:
        # 上层 poller / 监控器会把这些 sentinel 与其他句柄一起监听。
        return [proc.sentinel for proc in self.processes]

    # ------------------------------- 统计已退出的 EngineCore 子进程 -------------------------------
    def finished_procs(self) -> dict[str, int]:
        # 只返回已经拿到 exitcode 的子进程；
        # 仍在运行的进程不会出现在结果里。
        return {
            proc.name: proc.exitcode
            for proc in self.processes
            if proc.exitcode is not None
        }


def _describe_startup_failure(
    events: list[tuple[object, int]],
    handshake_socket: zmq.Socket,
    proc_manager: CoreEngineProcManager | None,
    coord_process: Process | None,
) -> tuple[dict[str, int], list[str]]:
    triggered_handles = [event[0] for event in events if event[0] != handshake_socket]

    if proc_manager is not None:
        for proc in proc_manager.processes:
            if proc.sentinel in triggered_handles:
                proc.join(timeout=0)

    if coord_process is not None and coord_process.sentinel in triggered_handles:
        coord_process.join(timeout=0)

    finished = proc_manager.finished_procs() if proc_manager else {}
    if coord_process is not None and coord_process.exitcode is not None:
        finished[coord_process.name] = coord_process.exitcode

    triggered = []
    for handle in triggered_handles:
        proc_name = None
        exitcode = None

        if proc_manager is not None:
            proc = next(
                (candidate for candidate in proc_manager.processes
                 if candidate.sentinel == handle),
                None,
            )
            if proc is not None:
                proc_name = proc.name
                exitcode = proc.exitcode

        if proc_name is None and coord_process is not None \
                and coord_process.sentinel == handle:
            proc_name = coord_process.name
            exitcode = coord_process.exitcode

        if proc_name is None:
            triggered.append(f"unknown_handle={handle}")
        else:
            exitcode_text = "pending" if exitcode is None else str(exitcode)
            triggered.append(f"{proc_name}(exitcode={exitcode_text})")

    return finished, triggered


class SignalCallback:
    """Safely trigger a callback from signal handler context via a dedicated thread."""

    def __init__(self, callback: Callable[[], None]):
        self._callback = callback
        self._event = threading.Event()
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="signal-callback",
        )
        self._thread.start()

    def _run(self):
        self._event.wait()
        if not self._stopped:
            self._callback()

    def trigger(self):
        self._event.set()

    def stop(self):
        self._stopped = True
        self._event.set()


@contextlib.contextmanager
def set_device_control_env_var(
    cfie_config: CfieConfig, local_dp_rank: int
) -> Iterator[None]:
    """
    Temporarily set CUDA_VISIBLE_DEVICES or equivalent
    for engine subprocess.
    """
    world_size = cfie_config.parallel_config.world_size
    local_world_size = cfie_config.parallel_config.local_world_size
    evar = current_platform.device_control_env_var

    value = get_device_indices(evar, local_dp_rank, world_size, local_world_size)
    with patch.dict(os.environ, values=((evar, value),)):
        yield


def get_device_indices(
    device_control_env_var: str,
    local_dp_rank: int,
    world_size: int,
    local_world_size: int | None = None,
):
    """
    Returns a comma-separated string of device indices for the specified
    data parallel rank.

    For example, if world_size=2 and local_dp_rank=1, and there are 4 devices,
    this will select devices 2 and 3 for local_dp_rank=1.
    """
    if local_world_size is None:
        local_world_size = world_size
    try:
        value = ",".join(
            str(current_platform.device_id_to_physical_device_id(i))
            for i in range(
                local_dp_rank * world_size,
                local_dp_rank * world_size + local_world_size,
            )
        )
    except IndexError as e:
        raise Exception(
            f"Error setting {device_control_env_var}: "
            f"local range: [{local_dp_rank * world_size}, "
            f"{(local_dp_rank + 1) * world_size}) "
            "base value: "
            f'"{os.getenv(device_control_env_var)}"'
        ) from e
    return value


class CoreEngineActorManager:
    """
    Utility class to handle creation, readiness, and shutdown
    of core engine Ray actors used by the AsyncLLM and LLMEngine.

    Different from CoreEngineProcManager, this class manages
    core engines for both local and remote nodes.
    """

    def __init__(
        self,
        cfie_config: CfieConfig,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        placement_groups: list["PlacementGroup"] | None = None,
        local_dp_ranks: list[int] | None = None,
    ):
        import copy

        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        from cfie.v1.engine.core import DPMoEEngineCoreActor, EngineCoreActor

        dp_size = cfie_config.parallel_config.data_parallel_size
        actor_class = (
            DPMoEEngineCoreActor
            if dp_size > 1 and cfie_config.model_config.is_moe
            else EngineCoreActor
        )

        self.local_engine_actors: list[ray.ActorHandle] = []
        self.remote_engine_actors: list[ray.ActorHandle] = []

        env_vars_list = get_env_vars_to_copy(destination=actor_class.__name__)
        self.env_vars_dict = {
            name: os.environ[name] for name in env_vars_list if name in os.environ
        }
        runtime_env = RuntimeEnv(env_vars=self.env_vars_dict)

        self.addresses = addresses
        self.executor_class = executor_class
        self.log_stats = log_stats
        local_engine_count = cfie_config.parallel_config.data_parallel_size_local
        world_size = cfie_config.parallel_config.world_size

        if ray.is_initialized():
            logger.info("Ray is already initialized. Skipping Ray initialization.")
        else:
            ray.init()

        cfie_config.parallel_config.allocate_elastic_ep_ports()

        if placement_groups is not None:
            assert local_dp_ranks is not None, (
                "local_dp_ranks must be provided if placement_groups is provided"
            )
            assert len(placement_groups) == len(local_dp_ranks), (
                "placement_groups and local_dp_ranks must have the same length"
            )
            logger.info("Using provided placement groups")
            # TODO(rui): validate passed-in placement groups
            self.created_placement_groups = []
        else:
            placement_groups, local_dp_ranks = (
                CoreEngineActorManager.create_dp_placement_groups(cfie_config)
            )
            self.created_placement_groups = placement_groups
        assert len(placement_groups) == dp_size, (
            "Number of placement groups must match data parallel size"
        )

        self.placement_group_is_local = []
        refs = []
        for index, local_index, pg in zip(
            range(dp_size), local_dp_ranks, placement_groups
        ):
            dp_cfie_config = copy.deepcopy(cfie_config)
            dp_cfie_config.parallel_config.placement_group = pg
            local_client = index < local_engine_count

            if dp_size > 1 and dp_cfie_config.kv_transfer_config is not None:
                # modify the engine_id and append the local_dp_rank to it to ensure
                # that the kv_transfer_config is unique for each DP rank.
                dp_cfie_config.kv_transfer_config.engine_id = (
                    f"{dp_cfie_config.kv_transfer_config.engine_id}_dp{local_index}"
                )

            # Ray XPU known issue: dpctl initializes the GPU runtime early, so
            # setting device env vars in Ray actor's initialization method
            # will not affect device selection. See:
            # https://github.com/ray-project/ray/blob/master/python/ray/_private/accelerators/intel_gpu.py#L56 # noqa: E501
            if current_platform.is_xpu():
                device_evar = current_platform.device_control_env_var
                device_indices = get_device_indices(
                    device_evar, local_index, world_size
                )
                actor_env_vars = self.env_vars_dict.copy()
                actor_env_vars[device_evar] = device_indices
                runtime_env = RuntimeEnv(env_vars=actor_env_vars)

            actor = (
                ray.remote(actor_class)
                .options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=world_size,
                    ),
                    runtime_env=runtime_env,
                )
                .remote(
                    cfie_config=dp_cfie_config,
                    executor_class=executor_class,
                    log_stats=log_stats,
                    local_client=local_client,
                    addresses=addresses,
                    dp_rank=index,
                    local_dp_rank=local_index,
                )
            )
            if local_client:
                self.local_engine_actors.append(actor)
            else:
                self.remote_engine_actors.append(actor)
            self.placement_group_is_local.append(local_client)
            refs.append(actor.wait_for_init.remote())

        ray.get(refs)
        self.run_refs = []
        for actor in self.local_engine_actors + self.remote_engine_actors:
            self.run_refs.append(actor.run.remote())

    @staticmethod
    def create_dp_placement_groups(
        cfie_config: CfieConfig,
    ) -> tuple[list["PlacementGroup"], list[int]]:
        """
        Create placement groups for data parallel.
        """

        import ray
        from ray._private.state import available_resources_per_node

        logger.info("Creating placement groups for data parallel")
        dp_master_ip = cfie_config.parallel_config.data_parallel_master_ip
        dp_size = cfie_config.parallel_config.data_parallel_size
        dp_size_local = cfie_config.parallel_config.data_parallel_size_local

        available_resources = available_resources_per_node()
        world_size = cfie_config.parallel_config.world_size
        placement_groups: list[PlacementGroup] = []
        local_dp_ranks: list[int] = []

        dp_master_ip_key = f"node:{dp_master_ip}"
        nodes = sorted(
            available_resources.values(), key=lambda x: dp_master_ip_key not in x
        )
        assert len(nodes) > 0, "No nodes with resources found in Ray cluster."
        assert dp_master_ip_key in nodes[0], (
            f"The DP master node (ip: {dp_master_ip}) is missing or dead"
        )
        device_str = current_platform.ray_device_key
        n_node_devices: list[int] = [
            int(node_resources[device_str])
            for node_resources in nodes
            if device_str in node_resources
        ]
        assert n_node_devices, f"No {device_str} found in Ray cluster."
        max_device_per_node = max(n_node_devices)

        pack_strategy = envs.VLLM_RAY_DP_PACK_STRATEGY
        _supported_pack_strategies = ("strict", "fill", "span")
        if pack_strategy not in _supported_pack_strategies:
            raise ValueError(
                f"{envs.VLLM_RAY_DP_PACK_STRATEGY} is not supported. "
                "Make sure to set `VLLM_RAY_DP_PACK_STRATEGY` "
                f"to one of {_supported_pack_strategies}"
            )

        all2all_backend = cfie_config.parallel_config.all2all_backend
        if pack_strategy == "fill" and (
            all2all_backend == "deepep_high_throughput"
            or all2all_backend == "deepep_low_latency"
        ):
            raise ValueError(
                "DeepEP kernels require EP ranks [0,7] (same for [8,15], ...) "
                "to be on the same node, but VLLM_RAY_DP_PACK_STRATEGY=fill "
                "does not guarantee that. "
                "Please use VLLM_RAY_DP_PACK_STRATEGY=strict instead."
            )

        if pack_strategy in ("strict", "fill"):
            placement_strategy = "STRICT_PACK"
        else:
            placement_strategy = "PACK"
            assert world_size > max_device_per_node, (
                f"World size {world_size} is smaller than the "
                "maximum number of devices per node "
                f"{max_device_per_node}. Make sure to set "
                "`VLLM_RAY_DP_PACK_STRATEGY` to `strict` or `fill`"
            )

            # if we need multiple nodes per dp group, we require for now that
            # available nodes are homogeneous
            assert set(n_node_devices) == {max_device_per_node}, (
                f"Nodes are not homogeneous, {nodes}"
            )
            assert world_size % max_device_per_node == 0, (
                f"For multi-node data parallel groups, world_size ({world_size}) must "
                f"be a multiple of number of devices per node ({max_device_per_node})."
            )
            assert len(n_node_devices) * max_device_per_node >= world_size * dp_size, (
                f"Not enough total available nodes ({len(n_node_devices)}) "
                f"and devices per node ({max_device_per_node}) "
                f"to satisfy required world size {world_size} and data parallel size "
                f"{dp_size}"
            )
            assert dp_size_local == 1, (
                f"data-parallel-size-local {dp_size_local} should be set as the "
                "default (1) for VLLM_RAY_DP_PACK_STRATEGY=span. "
                "The actual data-parallel-size-local will be auto determined."
            )

        # bundles collected for a single DP rank from multiple nodes,
        # for "span" pack strategy
        collected_bundles = []
        for node_resources in nodes:
            node_ip_keys = [
                key
                for key in node_resources
                if key != "node:__internal_head__" and key.startswith("node:")
            ]
            assert len(node_ip_keys) == 1, (
                f"Zero or multiple node IP keys found in node resources: {node_ip_keys}"
            )
            node_ip_key = node_ip_keys[0]
            node_ip = node_ip_key.split(":")[1]

            n_device_on_node = int(node_resources.get(device_str, 0))
            if pack_strategy == "span" and n_device_on_node != 0:
                # Strictly speaking,
                # dp_size_available = n_device_on_node / world_size
                # and is a fraction, but we use 1 for easier processing
                dp_size_available = 1
            else:
                dp_size_available = n_device_on_node // world_size

            if node_ip == dp_master_ip:
                if dp_size_available < dp_size_local:
                    raise ValueError(
                        f"Not enough resources to allocate {dp_size_local} DP ranks "
                        f"on DP master node {dp_master_ip}, possible to fit "
                        f"{dp_size_available} DP ranks."
                    )
                dp_size_to_allocate = dp_size_local
            elif pack_strategy == "strict":
                if dp_size_available < dp_size_local:
                    logger.info(
                        "Skipping node %s as %s DP ranks could not fit, "
                        "possible to fit %s DP ranks",
                        node_ip,
                        dp_size_local,
                        dp_size_available,
                    )
                    continue
                dp_size_to_allocate = dp_size_local
            else:
                # for "pack_strategy" in "fill" and "span"
                # we always take everything that's available
                dp_size_to_allocate = dp_size_available

            for i in range(dp_size_to_allocate):
                device_bundle = [{device_str: 1.0, "node:" + node_ip: 0.001}]
                if pack_strategy == "span":
                    collected_bundles += device_bundle * n_device_on_node
                    assert len(collected_bundles) <= world_size, (
                        "collected_bundles should be <= world_size, "
                        f"but got {len(collected_bundles)=} and {world_size=}"
                    )

                    # we only create a placement group if we collected enough devices
                    if len(collected_bundles) < world_size:
                        continue

                    bundles = collected_bundles + [{"CPU": 1.0}]
                    collected_bundles = []
                else:
                    bundles = device_bundle * world_size + [{"CPU": 1.0}]

                pg = ray.util.placement_group(
                    name=f"dp_rank_{len(placement_groups)}",
                    strategy=placement_strategy,
                    bundles=bundles,
                )
                placement_groups.append(pg)
                local_dp_ranks.append(i)
                if len(placement_groups) == dp_size:
                    break

        if len(placement_groups) < dp_size:
            raise ValueError(
                f"Not enough resources to allocate {dp_size} "
                "placement groups, only created "
                f"{len(placement_groups)} placement groups. "
                "Available resources: "
                f"{available_resources}"
            )
        assert len(placement_groups) == dp_size, (
            f"Created {len(placement_groups)} DP placement groups, expected {dp_size}"
        )
        assert len(local_dp_ranks) == dp_size, (
            f"local_dp_ranks length {len(local_dp_ranks)} does not match "
            f"expected {dp_size}"
        )
        return placement_groups, local_dp_ranks

    @staticmethod
    def add_dp_placement_groups(
        old_cfie_config: CfieConfig, new_data_parallel_size: int
    ) -> tuple[list["PlacementGroup"], list[int]]:
        """
        Add placement groups for new data parallel size.
        """
        import ray
        from ray._private.state import (
            available_resources_per_node,
            total_resources_per_node,
        )
        from ray.util.state import list_nodes

        old_dp_size = old_cfie_config.parallel_config.data_parallel_size
        num_pg_to_create = new_data_parallel_size - old_dp_size

        if num_pg_to_create <= 0:
            return [], []

        dp_master_ip = old_cfie_config.parallel_config.data_parallel_master_ip
        world_size = old_cfie_config.parallel_config.world_size

        nodes = list_nodes()
        nodes = sorted(nodes, key=lambda node: node.node_ip != dp_master_ip)
        assert nodes[0].node_ip == dp_master_ip, "The first node must be the head node"
        assert len(nodes) == 1 or nodes[1].node_ip != dp_master_ip, (
            "There can only be one head node"
        )

        available_resources = available_resources_per_node()
        total_resources = total_resources_per_node()

        placement_groups = []
        local_dp_ranks = []
        num_pg_created = 0

        device_str = current_platform.ray_device_key
        for node in nodes:
            if num_pg_created >= num_pg_to_create:
                break

            node_ip = node.node_ip
            node_id = node.node_id
            if device_str not in available_resources[node_id]:
                continue
            available_gpus = int(available_resources[node_id][device_str])

            # Get total GPUs on this node from the node's resources
            # Ray stores node resources with node ID as key
            total_gpus = int(total_resources[node_id][device_str])

            # Calculate used GPUs and used engines on this node
            used_gpus = max(0, total_gpus - available_gpus)
            used_engines_on_node = used_gpus // world_size

            # Calculate how many new engines this node can accommodate
            available_engine_count = available_gpus // world_size

            # Create placement groups for new engines on this node
            for i in range(available_engine_count):
                if num_pg_created >= num_pg_to_create:
                    break

                rank = old_dp_size + num_pg_created

                # Create bundles with node constraint for master node
                if node_ip == dp_master_ip:
                    bundles = [
                        {device_str: 1.0, "node:" + dp_master_ip: 0.001}
                    ] * world_size + [{"CPU": 1.0}]
                else:
                    bundles = [{device_str: 1.0}] * world_size + [{"CPU": 1.0}]

                pg = ray.util.placement_group(
                    name=f"dp_rank_{rank}",
                    strategy="STRICT_PACK",
                    bundles=bundles,
                )
                placement_groups.append(pg)

                # Local rank starts from the number of engines already used
                # on this node
                local_rank = used_engines_on_node + i
                local_dp_ranks.append(local_rank)
                num_pg_created += 1

        return placement_groups, local_dp_ranks

    def scale_up_elastic_ep(
        self, cur_cfie_config: CfieConfig, new_data_parallel_size: int
    ) -> None:
        import copy

        import ray
        from ray.runtime_env import RuntimeEnv
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        from cfie.v1.engine.core import DPMoEEngineCoreActor, EngineCoreActor

        actor_class = (
            DPMoEEngineCoreActor
            if cur_cfie_config.model_config.is_moe
            else EngineCoreActor
        )

        cur_data_parallel_size = len(self.local_engine_actors) + len(
            self.remote_engine_actors
        )

        assert new_data_parallel_size > cur_data_parallel_size, (
            f"New data parallel size {new_data_parallel_size} must be greater "
            f"than current data parallel size {cur_data_parallel_size} "
            "for scale up"
        )

        placement_groups, local_dp_ranks = self.add_dp_placement_groups(
            cur_cfie_config, new_data_parallel_size
        )

        world_size = cur_cfie_config.parallel_config.world_size
        dp_master_ip = cur_cfie_config.parallel_config.data_parallel_master_ip
        new_local_engines = 0

        runtime_env = RuntimeEnv(
            env_vars=self.env_vars_dict | {"VLLM_ELASTIC_EP_SCALE_UP_LAUNCH": "1"}
        )
        for i, (pg, local_rank) in enumerate(zip(placement_groups, local_dp_ranks)):
            rank = cur_data_parallel_size + i
            dp_cfie_config = copy.deepcopy(cur_cfie_config)
            dp_cfie_config.parallel_config.data_parallel_size = new_data_parallel_size
            dp_cfie_config.parallel_config.placement_group = pg

            # Check if this placement group is on the head node
            local_client = any(
                bundle.get("node:" + dp_master_ip, 0) > 0 for bundle in pg.bundle_specs
            )

            if local_client:
                new_local_engines += 1
                # Update data_parallel_size_local
                dp_cfie_config.parallel_config.data_parallel_size_local = (
                    cur_cfie_config.parallel_config.data_parallel_size_local
                    + new_local_engines
                )

            actor = (
                ray.remote(actor_class)
                .options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=world_size,
                    ),
                    runtime_env=runtime_env,
                )
                .remote(
                    cfie_config=dp_cfie_config,
                    executor_class=self.executor_class,
                    log_stats=self.log_stats,
                    local_client=local_client,
                    addresses=self.addresses,
                    dp_rank=rank,
                    local_dp_rank=local_rank,
                )
            )

            if local_client:
                self.local_engine_actors.append(actor)
            else:
                self.remote_engine_actors.append(actor)
            self.created_placement_groups.append(pg)
            self.placement_group_is_local.append(local_client)

        ray.get(
            [
                actor.wait_for_init.remote()
                for actor in (
                    self.local_engine_actors[-new_local_engines:]
                    if new_local_engines > 0
                    else []
                )
                + self.remote_engine_actors[
                    -(len(placement_groups) - new_local_engines) :
                ]
            ]
        )

        actors = (
            self.local_engine_actors[-new_local_engines:]
            if new_local_engines > 0
            else []
        ) + self.remote_engine_actors[-(len(placement_groups) - new_local_engines) :]

        for actor in actors:
            self.run_refs.append(actor.run.remote())

        cur_cfie_config.parallel_config.data_parallel_size = new_data_parallel_size
        # Update old_cfie_config with new data_parallel_size_local if any new
        # local engines were added
        if new_local_engines > 0:
            cur_cfie_config.parallel_config.data_parallel_size_local += (
                new_local_engines
            )

    def scale_down_elastic_ep(
        self, cur_data_parallel_size: int, new_data_parallel_size: int
    ) -> None:
        import ray

        assert cur_data_parallel_size > new_data_parallel_size, (
            f"cur_data_parallel_size {cur_data_parallel_size} must be greater "
            f"than new_data_parallel_size {new_data_parallel_size} "
            "for scale down"
        )
        for _ in range(cur_data_parallel_size - new_data_parallel_size):
            pg = self.created_placement_groups.pop()
            is_local = self.placement_group_is_local.pop()
            if is_local:
                self.local_engine_actors.pop()
            else:
                self.remote_engine_actors.pop()
            ray.util.remove_placement_group(pg)

    def get_run_refs(self):
        return self.run_refs

    def shutdown(self, timeout: float | None = None) -> None:
        import ray

        for actor in self.local_engine_actors + self.remote_engine_actors:
            ray.kill(actor)
        for pg in self.created_placement_groups:
            ray.util.remove_placement_group(pg)


def get_engine_zmq_addresses(
    # 配置对象，包含并行与通信相关参数。
    cfie_config: CfieConfig,
    # API Server 的数量，默认分配 1 组地址。
    num_api_servers: int = 1,

) -> EngineZmqAddresses:

    # 为 engine 与 client 通信分配 ZMQ 地址。
    """为 engine 与 client 之间的通信分配 ZMQ 地址。"""

    # 取出并行配置，后续统一从中读取 DP 相关参数。
    parallel_config = cfie_config.parallel_config

    # 当前节点上的本地 engine 数量。
    local_engine_count = parallel_config.data_parallel_size_local

    # 当前节点对应的本地 DP 起始 rank。
    local_start_index = parallel_config.data_parallel_rank_local

    # 全局数据并行总规模。
    dp_size = parallel_config.data_parallel_size

    # 数据并行主节点 IP。
    host = parallel_config.data_parallel_master_ip

    # 是否仅使用本地 engine。
    local_engines_only = parallel_config.local_engines_only

    # 离线模式下，每个 DP rank 对应一个 LLM 实例，
    # 且每个 LLM 实例对应一个 core engine，
    # 参见 examples/offline_inference/data_parallel.py。
    offline_mode = local_start_index is not None

    # 当 client 只向本机同置的 engine 发送请求时，该标记为 True。
    # 满足以下任一条件即可视为仅访问本地 engine：
    # 1. 当前为离线模式；
    # 2. 配置中显式要求仅使用本地 engine；
    # 3. 本地 engine 数量已经等于全局 DP 总数。
    client_local_only = (
        offline_mode or local_engines_only or (local_engine_count == dp_size)
    )

    # 处理从单机扩展到多机的弹性 EP 场景。
    if parallel_config.enable_elastic_ep:
        client_local_only = False

    # 为输入与输出通道分别生成 num_api_servers 组 ZMQ 地址。
    return EngineZmqAddresses(

        # 输入通道地址列表。
        inputs=[
            get_engine_client_zmq_addr(client_local_only, host)
            for _ in range(num_api_servers)
        ],

        # 输出通道地址列表。
        outputs=[
            get_engine_client_zmq_addr(client_local_only, host)
            for _ in range(num_api_servers)
        ],
    )


@contextlib.contextmanager
def launch_core_engines(
    cfie_config: CfieConfig,
    executor_class: type[Executor],
    log_stats: bool,
    addresses: EngineZmqAddresses,
    num_api_servers: int = 1,
) -> Iterator[
    tuple[
        CoreEngineProcManager | CoreEngineActorManager | None,
        DPCoordinator | None,
        EngineZmqAddresses,
    ]
]:
    """按当前并行配置拉起本地 `EngineCore` 与可选的 DP coordinator。

    这是前端进入多进程模式时的总入口，负责把“当前 client 该托管哪些后台组件”
    统一决策出来。它处理的核心问题有三类：

    - 当前是否需要额外拉起 `DPCoordinator`
    - 当前是使用本地进程版 engine 还是 Ray actor 版 engine
    - 当前前端需要与哪些 DP rank 完成启动握手

    函数会先把需要的后台组件创建出来并 `yield` 给调用方，
    待调用方完成前端 socket 初始化后，再继续等待所有目标 engine 完成启动握手。
    """

    # ------------------------------- 读取当前 DP 启动所需的基础配置 -------------------------------
    parallel_config = cfie_config.parallel_config
    # DP 总规模决定全局一共有多少条 engine 主线。
    dp_size = parallel_config.data_parallel_size
    # 当前节点本地需要拉起多少个 engine。
    local_engine_count = parallel_config.data_parallel_size_local
    # offline 模式下，本地 engine 的起始局部 rank 由外部显式给出。
    local_start_index = parallel_config.data_parallel_rank_local
    # 当前前端视角下的起始 DP rank。
    dp_rank = parallel_config.data_parallel_rank
    # 握手地址可能需要走跨节点网络，因此先取出 master host。
    host = parallel_config.data_parallel_master_ip
    # 某些 LB 模式下，当前前端只管理本地 engines。
    local_engines_only = parallel_config.local_engines_only

    # offline 模式表示当前前端只和一个显式指定的本地 DP rank 打交道。
    offline_mode = local_start_index is not None

    # ------------------------------- 决定是否需要拉起 DP coordinator -------------------------------
    # DP coordinator 只在 online DP 场景下由 rank 0 拉起；
    # 它负责：
    # 1. internal / hybrid LB 下的队列统计汇总与发布
    # 2. MoE 模型下的 wave 协调
    run_coordinator = (
        cfie_config.needs_dp_coordinator and not offline_mode and dp_rank == 0
    )

    if run_coordinator:
        coordinator = DPCoordinator(
            parallel_config,
            enable_wave_coordination=cfie_config.model_config.is_moe,
        )

        addresses.coordinator_input, addresses.coordinator_output = (
            coordinator.get_engine_socket_addresses()
        )
        addresses.frontend_stats_publish_address = (
            coordinator.get_stats_publish_address()
        )

        logger.info("Started DP Coordinator process (PID: %d)", coordinator.proc.pid)
    else:
        coordinator = None

    # ------------------------------- 按 DP backend 选择 engine 托管方式 -------------------------------
    # Ray backend 不走本地 `EngineCoreProcManager`，而是交给 actor manager 统一托管。
    if parallel_config.data_parallel_backend == "ray":
        logger.info("Starting ray-based data parallel backend")

        engine_actor_manager = CoreEngineActorManager(
            cfie_config=cfie_config,
            addresses=addresses,
            executor_class=executor_class,
            log_stats=log_stats,
        )

        yield engine_actor_manager, coordinator, addresses
        return

    # ------------------------------- 推导当前前端需要等待握手的 engine 集合 -------------------------------
    if offline_mode:
        # offline 模式只允许当前前端对应一个本地 engine。
        assert local_engine_count == 1
        engines_to_handshake = [CoreEngine(index=dp_rank, local=True)]
    elif dp_rank == 0:
        # rank 0 持有 coordinator，因此无论 internal 还是 external DPLB，
        # 它都要感知并等待所有 core engines 完成握手。
        # 这也覆盖了“rank 0 本身不托管本地 engine、只做控制面”的 headless 情况。
        engines_to_handshake = [
            CoreEngine(index=i, local=(i < local_engine_count)) for i in range(dp_size)
        ]
    else:
        # 非 rank 0 前端只应等待自己实际托管的本地 cores；
        # 若这里仍要求它管理远端 cores，说明并行拓扑和 LB 模式不兼容。
        assert local_engines_only, (
            "Attempting to launch core_engines from dp_rank > 0, but "
            "found internal DPLB, which is incompatible."
        )
        engines_to_handshake = [
            CoreEngine(index=i, local=True)
            for i in range(dp_rank, dp_rank + local_engine_count)
        ]

    # ------------------------------- 规划 engine-client 握手地址 -------------------------------
    # `handshake_local_only` 表示本轮启动出来的 engines 是否只和同机前端握手。
    # external DP LB 下，非 rank 0 前端上的 engines 还需要和 rank 0 前端协作，
    # 因此不能把握手范围限制在 purely local。
    handshake_local_only = offline_mode or local_engine_count == dp_size

    # Elastic EP 会发生跨节点扩缩容，因此强制走可跨节点复用的握手地址。
    if parallel_config.enable_elastic_ep:
        handshake_local_only = False

    # 根据握手可见范围，得到 engine-client 之间真正使用的握手入口地址。
    handshake_address = get_engine_client_zmq_addr(
        handshake_local_only, host, parallel_config.data_parallel_rpc_port
    )

    # local_engines_only 且 dp_rank > 0 时，需要给本地 client 额外分配一个本地握手地址，
    # 这样本地 engines 和 rank 0 前端可以分层握手，避免地址冲突。
    if local_engines_only and dp_rank > 0:
        assert not handshake_local_only
        local_handshake_address = get_open_zmq_ipc_path()
        client_handshake_address = local_handshake_address
    else:
        local_handshake_address = handshake_address
        client_handshake_address = None

    # ------------------------------- 拉起本地 engine manager 并等待后续握手 -------------------------------
    # 这里先把握手 ROUTER 建起来，再启动本地 engines；
    # 这样子进程起来后可以立刻把 HELLO / READY 打回到已存在的握手通道上。
    with zmq_socket_ctx(
        local_handshake_address, zmq.ROUTER, bind=True
    ) as handshake_socket:
        if local_engine_count:
            local_engine_manager = CoreEngineProcManager(
                cfie_config=cfie_config,
                executor_class=executor_class,
                log_stats=log_stats,
                handshake_address=handshake_address,
                client_handshake_address=client_handshake_address,
                local_client=True,
                local_engine_count=local_engine_count,
                start_index=dp_rank,
                local_start_index=local_start_index or 0,
            )
        else:
            local_engine_manager = None

        # 先把刚创建好的 manager / coordinator / addresses 暴露给调用方，
        # 让前端有机会完成 socket 连接与资源登记。
        yield local_engine_manager, coordinator, addresses

        # 当前端准备完成后，再统一等待目标 engines 完成正式启动握手。
        wait_for_engine_startup(
            handshake_socket,
            addresses,
            engines_to_handshake,
            parallel_config,
            dp_size > 1 and cfie_config.model_config.is_moe,
            cfie_config.cache_config,
            local_engine_manager,
            coordinator.proc if coordinator else None,
        )


def wait_for_engine_startup(
    handshake_socket: zmq.Socket,
    addresses: EngineZmqAddresses,
    core_engines: list[CoreEngine],
    parallel_config: ParallelConfig,
    coordinated_dp: bool,
    cache_config: CacheConfig,
    proc_manager: CoreEngineProcManager | None,
    coord_process: Process | None,
):
    # 等待所有 engine core 完成启动握手。
    # 这里的握手分两阶段：
    # 1) engine 先发送 HELLO，前端回传初始化元数据
    # 2) engine 完成自身初始化后再发送 READY
    # 只有所有目标 engine 都走完这两个阶段，启动才算成功。

    # ----------------- 初始化待等待状态 -----------------
    # 本地 engine 数量由 local DP size 决定。
    local_count = parallel_config.data_parallel_size_local
    # 远端 engine 数量等于总待握手 engine 数减去本地 engine 数。
    remote_count = len(core_engines) - local_count
    # conn_pending 表示尚未收到 HELLO 的 [local, remote] engine 数。
    # start_pending 表示已收到 HELLO 但尚未收到 READY 的 [local, remote] engine 数。
    conn_pending, start_pending = [local_count, remote_count], [0, 0]
    # 统一使用 poller 同时监听握手 socket 与子进程 sentinel。
    poller = zmq.Poller()
    poller.register(handshake_socket, zmq.POLLIN)
    process_sentinels = []

    # remote engine 是否必须以 headless 方式启动，取决于 DP 负载均衡模式。
    remote_should_be_headless = (
        not parallel_config.data_parallel_hybrid_lb
        and not parallel_config.data_parallel_external_lb
    )

    # 注册本地 engine 子进程 sentinel；若子进程提前退出，poller 会立刻感知。
    if proc_manager is not None:
        for sentinel in proc_manager.sentinels():
            process_sentinels.append(sentinel)
            # Windows 的进程 sentinel 不是 ZMQ socket；
            # 不能注册进 zmq.Poller，否则会在 poll 阶段触发 “not a socket”。
            if os.name != "nt":
                poller.register(sentinel, zmq.POLLIN)
    # 若存在 coordinator，也把其 sentinel 一并纳入监控。
    if coord_process is not None:
        process_sentinels.append(coord_process.sentinel)
        if os.name != "nt":
            poller.register(coord_process.sentinel, zmq.POLLIN)

    def _poll_windows_process_sentinels() -> list[tuple[object, int]]:
        # Windows 下用 multiprocessing.connection.wait 单独检查进程退出；
        # 这样既保留子进程提前失败检测，又避免把非 socket 句柄交给 ZMQ。
        if os.name != "nt" or not process_sentinels:
            return []
        ready_sentinels = connection.wait(process_sentinels, timeout=0)
        return [(sentinel, zmq.POLLIN) for sentinel in ready_sentinels]

    # ----------------- 轮询等待握手完成 -----------------
    while any(conn_pending) or any(start_pending):
        events = _poll_windows_process_sentinels()
        if not events:
            events = poller.poll(STARTUP_POLL_PERIOD_MS)
            events.extend(_poll_windows_process_sentinels())
        if not events:
            # 长时间未收到事件时输出当前卡在哪个阶段，便于判断是没连上还是没 ready。
            if any(conn_pending):
                logger.debug(
                    "Waiting for %d local, %d remote core engine proc(s) to connect.",
                    *conn_pending,
                )
            if any(start_pending):
                logger.debug(
                    "Waiting for %d local, %d remote core engine proc(s) to start.",
                    *start_pending,
                )
            continue
        if len(events) > 1 or events[0][0] != handshake_socket:
            # 只要 poller 返回的不是纯握手 socket 事件，就说明至少有一个被监控进程
            # 在握手完成前提前退出，需要立刻终止启动并汇总失败信息。
            finished, triggered = _describe_startup_failure(
                events,
                handshake_socket,
                proc_manager,
                coord_process,
            )
            failed_summary: dict[str, int] | list[str] = (
                finished if finished else triggered
            )
            raise RuntimeError(
                "Engine core initialization failed. "
                "See root cause above. "
                f"Failed core proc(s): {failed_summary}. "
                f"Startup state: conn_pending={conn_pending}, "
                f"start_pending={start_pending}"
            )

        # ----------------- 处理一条握手消息 -----------------
        # 从握手 socket 读取 engine 发来的 HELLO / READY 消息。
        eng_identity, ready_msg_bytes = handshake_socket.recv_multipart()
        # identity 直接编码的是 DP rank。
        eng_index = int.from_bytes(eng_identity, "little")
        # 找到这条消息对应的 engine 启动状态对象。
        engine = next((e for e in core_engines if e.identity == eng_identity), None)
        if engine is None:
            raise RuntimeError(
                f"Message from engine with unexpected data parallel rank: {eng_index}"
            )
        # 解码 msgpack 握手载荷，并提取本轮需要校验的关键字段。
        msg = msgspec.msgpack.decode(ready_msg_bytes)
        status, local, headless = msg["status"], msg["local"], msg["headless"]
        # 收到的 local/remote 属性必须和预期 engine 身份一致。
        if local != engine.local:
            raise RuntimeError(
                f"{status} message from "
                f"{'local' if local else 'remote'} "
                f"engine {eng_index}, expected it to be "
                f"{'local' if engine.local else 'remote'}"
            )

        # remote engine 的 headless 语义必须和当前 DP 拓扑匹配。
        if not local and headless != remote_should_be_headless:
            if headless:
                raise RuntimeError(
                    f"Remote engine {eng_index} must not use "
                    f"--headless in external or hybrid dp lb "
                    f"mode"
                )
            else:
                raise RuntimeError(
                    f"Remote engine {eng_index} must use "
                    f"--headless unless in external or hybrid "
                    f"dp lb mode"
                )

        # ----------------- HELLO 分支 -----------------
        if status == "HELLO" and engine.state == CoreEngineState.NEW:
            # HELLO 表示 engine 已连上前端握手 socket，但尚未拿到初始化元数据。
            # 此时前端把地址、DP 参数等启动元数据回传给对应 engine。
            init_message = msgspec.msgpack.encode(
                EngineHandshakeMetadata(
                    addresses=addresses,
                    parallel_config={
                        k: getattr(parallel_config, k)
                        for k in (
                            "data_parallel_master_ip",
                            "data_parallel_master_port",
                            "_data_parallel_master_port_list",
                            "data_parallel_size",
                        )
                    }
                    if coordinated_dp
                    else {},
                )
            )
            # 把初始化信息定向发回当前 engine。
            handshake_socket.send_multipart((eng_identity, init_message), copy=False)
            # 该 engine 已从“等待连接”转入“等待启动完成”。
            conn_pending[0 if local else 1] -= 1
            start_pending[0 if local else 1] += 1
            # 更新本地状态机，后续只接受 READY。
            engine.state = CoreEngineState.CONNECTED

        # ----------------- READY 分支 -----------------
        elif status == "READY" and engine.state == CoreEngineState.CONNECTED:
            # READY 表示该 engine 已完成模型/worker 初始化，可把初始化结果回灌前端。
            # KV cache block 数在 DP 下需要聚合所有 engine 的上报值。
            num_gpu_blocks = cache_config.num_gpu_blocks or 0
            num_gpu_blocks += msg["num_gpu_blocks"]
            cache_config.num_gpu_blocks = num_gpu_blocks

            # external DP LB 模式下，前端需要从某个 engine 的 READY 消息里接收
            # coordinator 的 stats 发布地址，并缓存下来供后续本地前端连接。
            if addresses.frontend_stats_publish_address is None:
                addresses.frontend_stats_publish_address = msg.get("dp_stats_address")

            # MoE + coordinated DP 场景下，要求所有 worker 的关键并行配置完全一致。
            # 这里使用 hash 做一次快速一致性校验，避免 collective 参数不匹配。
            if coordinated_dp:
                worker_config_hash = msg.get("parallel_config_hash")
                expected_hash = parallel_config.compute_hash()
                if worker_config_hash != expected_hash:
                    raise RuntimeError(
                        f"Configuration mismatch detected for engine "
                        f"{eng_index}. All DP workers must have identical "
                        f"configurations for parameters that affect collective "
                        f"communication (e.g., enable_eplb, "
                        f"eplb_config.log_balancedness). "
                        f"Worker hash: {worker_config_hash}, "
                        f"Expected hash: {expected_hash}. "
                        f"Please ensure all workers are started with the same "
                        f"command-line arguments."
                    )

            # READY 处理完成后，该 engine 不再处于待启动集合中。
            start_pending[0 if local else 1] -= 1
            engine.state = CoreEngineState.READY
        else:
            # 任意不符合状态机约束的消息顺序都视为启动协议错误。
            raise RuntimeError(
                f"Unexpected {status} message for "
                f"{'local' if local else 'remote'} engine "
                f"{eng_index} in {engine.state} state."
            )

        # 记录每个 engine 的握手推进情况，便于排查卡在哪个 rank / 阶段。
        logger.debug(
            "%s from %s core engine process %s.",
            status,
            "local" if local else "remote",
            eng_index,
        )
