# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""

import gc
import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from datetime import timedelta
from types import NoneType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

import cfie.envs as envs
from cfie.config import CUDAGraphMode, CfieConfig, set_current_cfie_config
from cfie.config.compilation import CompilationMode
from cfie.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)
from cfie.distributed.ec_transfer import ensure_ec_transfer_initialized
from cfie.distributed.eplb.eplb_utils import override_envs_for_eplb
from cfie.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from cfie.distributed.parallel_state import (
    Handle,
    get_pp_group,
    get_tp_group,
)
from cfie.distributed.weight_transfer import WeightTransferEngineFactory
from cfie.logger import init_logger
from cfie.lora.request import LoRARequest
from cfie.model_executor.warmup.kernel_warmup import kernel_warmup
from cfie.platforms import current_platform
from cfie.profiler.wrapper import CudaProfilerWrapper, TorchProfilerWrapper
from cfie.sequence import IntermediateTensors
from cfie.tasks import SupportedTask
from cfie.tracing import instrument
from cfie.utils.mem_constants import GiB_bytes
from cfie.utils.mem_utils import (
    MemorySnapshot,
    format_gib,
    memory_profiling,
    split_gpu_memory_budget,
)
from cfie.utils.torch_utils import set_random_seed
from cfie.v1.core.sched.output import GrammarOutput, SchedulerOutput
from cfie.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from cfie.v1.outputs import (
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from cfie.v1.utils import compute_iteration_details, report_usage_stats
from cfie.v1.worker.utils import is_residual_scattered_for_sp
from cfie.v1.worker.worker_base import WorkerBase
from cfie.v1.worker.workspace import init_workspace_manager

from ...model_executor.model_loader import TensorizerLoader
from .gpu.warmup import warmup_kernels
from .utils import get_static_memory_budget

logger = init_logger(__name__)

if TYPE_CHECKING:
    from cfie.model_executor.model_loader.tensorizer import TensorizerConfig
    from cfie.v1.worker.gpu_model_runner import GPUModelRunner


class AsyncIntermediateTensors(IntermediateTensors):
    """IntermediateTensors with lazy comm synchronization"""

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        comm_handles: list[Handle] | None = None,
        comm_postprocess: list[Callable[[], None]] | None = None,
    ) -> None:
        super().__init__(tensors)
        self._comm_handles = comm_handles
        self._comm_postprocess = comm_postprocess
        self._comm_waited = False

    def wait_for_comm(self) -> None:
        if self._comm_waited:
            return
        if self._comm_handles:
            for handle in self._comm_handles:
                handle.wait()
        if self._comm_postprocess:
            for fn in self._comm_postprocess:
                fn()
        self._comm_waited = True

    def __getattribute__(self, name: str):
        # ensure `.tensors` is ready before use
        if name == "tensors" and not object.__getattribute__(self, "_comm_waited"):
            object.__getattribute__(self, "wait_for_comm")()
        return object.__getattribute__(self, name)


class Worker(WorkerBase):
    def __init__(
        self,
        cfie_config: CfieConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            cfie_config=cfie_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        # configure float32 matmul precision according to vLLM env.
        precision = envs.VLLM_FLOAT32_MATMUL_PRECISION
        torch.set_float32_matmul_precision(precision)

        from cfie.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor

        self.elastic_ep_executor = ElasticEPScalingExecutor(self)

        # Buffers saved before sleep
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

        # Weight transfer engine (initialized on-demand)
        self.weight_transfer_engine = (
            WeightTransferEngineFactory.create_engine(
                self.cfie_config.weight_transfer_config,
                self.cfie_config.parallel_config,
            )
            if self.cfie_config.weight_transfer_config is not None
            else None
        )

        # Torch/CUDA profiler. Enabled and configured through profiler_config.
        # Profiler wrapper is created lazily in profile() when start is called,
        # so we have all the information needed for proper trace naming.
        self.profiler: Any | None = None
        self.profiler_config = cfie_config.profiler_config

        # Only validate profiler config is valid, don't instantiate yet
        if self.profiler_config.profiler not in ("torch", "cuda", None):
            raise ValueError(f"Unknown profiler type: {self.profiler_config.profiler}")

        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
        # pending non-blocking PP send work from the previous iteration
        self._pp_send_work: list[Handle] = []

    def sleep(self, level: int = 1) -> None:
        from cfie.device_allocator.cumem import CuMemAllocator

        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone() for name, buffer in model.named_buffers()
            }

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %s GiB memory, %s GiB memory is still in use.",
            format_gib(freed_bytes),
            format_gib(used_bytes),
        )

    def wake_up(self, tags: list[str] | None = None) -> None:
        from cfie.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

        # If the KV cache has just been woken up,
        # the internal state of cache_engine must be reset,
        # especially the FP8 scaling factor.
        if (
            (tags is None or "kv_cache" in tags)
            and self.cache_config.cache_dtype.startswith("fp8")
            and hasattr(self.model_runner, "init_fp8_kv_scales")
        ):
            self.model_runner.init_fp8_kv_scales()

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if not self.cfie_config.model_config.enable_sleep_mode:
            return nullcontext()

        from cfie.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        if tag == "weights":
            assert allocator.get_current_usage() == 0, (
                "Sleep mode can only be used for one instance per process."
            )
        return allocator.use_memory_pool(tag=tag)

    @instrument(span_name="Init device")
    def init_device(self):
        # ------------------------------- 校验设备类型并进入 CUDA 设备初始化路径 -------------------------------
        # 当前 worker 仅支持在 CUDA 设备上完成初始化；若设备类型不是 CUDA，则直接报错。
        if self.device_config.device_type == "cuda":
            # ------------------------------- 清理会干扰图构建的环境变量并读取并行配置 -------------------------------
            # 移除由 Ray 注入的 NCCL_ASYNC_ERROR_HANDLING，避免其影响后续图构建流程。
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

            # 读取当前 worker 的并行配置对象，后续会用于本地 rank 修正与设备合法性检查。
            parallel_config = self.parallel_config

            # ------------------------------- 在单节点本地 DP 场景下修正 local_rank -------------------------------
            # 当不是 ray 或 external_launcher 驱动，且 DP 后端也不是 ray，同时每个 DP 仅位于单节点内时，
            # 需要把 DP rank 对本地设备编号的偏移量合并进 local_rank。
            if (
                    parallel_config.distributed_executor_backend
                    not in ("ray", "external_launcher")
                    and parallel_config.data_parallel_backend != "ray"
                    and parallel_config.nnodes_within_dp == 1
            ):
                # 优先读取本地 DP rank；若未提供，则退回到全局 DP index。
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    # 使用全局 DP index 作为本地 DP rank 的退化值。
                    dp_local_rank = self.parallel_config.data_parallel_index

                # 计算当前 TP 与 PP 组合后的局部并行世界大小。
                tp_pp_world_size = (
                        self.parallel_config.pipeline_parallel_size
                        * self.parallel_config.tensor_parallel_size
                )

                # ------------------------------- 将 DP rank 的设备偏移量并入 local_rank -------------------------------
                # 按 DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK 的方式修正本地设备编号。
                self.local_rank += dp_local_rank * tp_pp_world_size

                # 校验修正后的 local_rank 没有越过当前可见设备数量上界。
                assert self.local_rank < torch.accelerator.device_count(), (
                    f"DP adjusted local rank {self.local_rank} is out of bounds. "
                )

                # 读取当前进程可见的 CUDA 设备数量；若 CUDA 不可用则记为 0。
                visible_device_count = (
                    torch.accelerator.device_count() if torch.cuda.is_available() else 0
                )

                # 校验当前 local_world_size 不超过可见设备总数。
                assert self.parallel_config.local_world_size <= visible_device_count, (
                    f"local_world_size ({self.parallel_config.local_world_size}) must "
                    f"be less than or equal to the number of visible devices "
                    f"({visible_device_count})."
                )

            # ------------------------------- 绑定当前 worker 对应的 CUDA 设备 -------------------------------
            # 根据修正后的 local_rank 构造当前 worker 绑定的 CUDA 设备对象。
            self.device = torch.device(f"cuda:{self.local_rank}")

            # 将当前线程的默认设备切换到该 CUDA 设备。
            torch.accelerator.set_device_index(self.device)

            # 校验当前平台是否支持模型配置要求的数据类型。
            current_platform.check_if_supports_dtype(self.model_config.dtype)

            # ------------------------------- 在采集显存快照前初始化分布式环境 -------------------------------
            # 必须先初始化分布式环境，使 NCCL 相关缓冲区先分配完成，再去测量后续可用显存。
            init_worker_distributed_environment(
                self.cfie_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

            # ------------------------------- 记录 model runner 版本并设置随机种子 -------------------------------
            # 当启用了 V2 model runner 时，仅在本地打印一次日志提示。
            if self.use_v2_model_runner:
                logger.info_once("Using V2 Model Runner", scope="local")

            # 设置模型随机种子，保证后续初始化与 profiling 过程可复现。
            set_random_seed(self.model_config.seed)

            # ------------------------------- 在 NCCL 初始化完成后采集显存快照并拆分预算 -------------------------------
            # 在采集显存快照前先执行垃圾回收，尽量减少无关对象残留。
            gc.collect()

            # 清空 PyTorch 缓存分配器中的可释放缓存。
            torch.accelerator.empty_cache()

            # 采集当前设备的初始化显存快照。
            self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)

            # 基于初始化显存快照与 cache 配置计算静态显存预算。
            self.static_memory_budget = get_static_memory_budget(
                init_snapshot, self.cache_config
            )

            # 按 gpu_memory_utilization 把总显存拆成静态预算与运行时余量，并保存运行时余量部分。
            _, self.runtime_memory_headroom = split_gpu_memory_budget(
                init_snapshot.total_memory,
                self.cache_config.gpu_memory_utilization,
            )

            # 输出初始化显存快照调试信息。
            logger.debug("worker init memory snapshot: %r", self.init_snapshot)

            # 输出静态显存预算调试信息。
            logger.debug(
                "worker static memory budget: %sGiB",
                format_gib(self.static_memory_budget),
            )
        else:
            # 当前设备类型不受支持时，直接抛出异常。
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # ------------------------------- 初始化 workspace manager -------------------------------
        # 当启用 DBO 时使用 2 个 ubatch，否则使用 1 个 ubatch。
        num_ubatches = 2 if self.cfie_config.parallel_config.enable_dbo else 1

        # 基于当前设备与 ubatch 数量初始化 workspace manager。
        init_workspace_manager(self.device, num_ubatches)

        # ------------------------------- 构造模型运行器实例 -------------------------------
        # 当启用 V2 model runner 时，导入并构造 V2 版本的 GPUModelRunner。
        if self.use_v2_model_runner:
            from cfie.v1.worker.gpu.model_runner import (
                GPUModelRunner as GPUModelRunnerV2,
            )

            # 构造 V2 版本 model runner，并显式标注类型忽略以规避当前临时类型问题。
            self.model_runner: GPUModelRunner = GPUModelRunnerV2(  # type: ignore
                self.cfie_config, self.device
            )
        else:
            # 否则导入并构造 V1 版本的 GPUModelRunner。
            from cfie.v1.worker.gpu_model_runner import (
                GPUModelRunner as GPUModelRunnerV1,
            )

            # 构造 V1 版本 model runner。
            self.model_runner = GPUModelRunnerV1(self.cfie_config, self.device)

        # ------------------------------- 在 rank 0 上报使用统计信息 -------------------------------
        # 仅由 rank 0 收集并上报当前配置相关的使用统计信息。
        if self.rank == 0:
            report_usage_stats(self.cfie_config)

    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    # 调用 model runner 真正加载主模型与可选 drafter 模型。
    def load_model(self) -> None:
        # Elastic EP 扩容启动时用 dummy 权重先把结构拉起来。
        dummy_weights = os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1"
        if dummy_weights:
            # 从 executor 接收新的专家映射关系。
            (
                expanded_physical_to_logical,
                num_logical_experts,
                old_num_physical_experts,
            ) = self.elastic_ep_executor.receive_expert_mapping()
            # 读取扩容后物理专家数量。
            num_physical_experts = expanded_physical_to_logical.shape[1]
            # 回写冗余专家数量给并行配置。
            self.parallel_config.eplb_config.num_redundant_experts = (
                num_physical_experts - num_logical_experts
            )

        with (
            self._maybe_get_memory_pool_context(tag="weights"),
            set_current_cfie_config(self.cfie_config),
        ):
            # 在权重内存池上下文中实际加载模型。
            self.model_runner.load_model(load_dummy_weights=dummy_weights)

        if dummy_weights:
            # dummy 权重模式下继续按映射关系补齐 EPLB 状态。
            self.model_runner.setup_eplb_from_mapping(
                expanded_physical_to_logical, old_num_physical_experts
            )
            # 扩容过渡阶段先抑制 EPLB。
            self.model_runner.eep_eplb_suppressed = True

    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def reload_weights(self, *args, **kwargs) -> None:
        self.model_runner.reload_weights(*args, **kwargs)

    @torch.inference_mode()
    # 通过一次 profiling 估算出还能留给 KV cache 的显存预算。
    def determine_available_memory(self) -> int:
        """
        通过一次启动期 profiling，推导“真正可分给 KV cache 的显存预算”。

        这里不是简单地看一次 `free_memory`，而是把显存拆成两类：

        - steady_non_kv_cache_memory:
          profile 结束后仍会常驻的非 KV 显存，例如权重、常驻 workspace、
          backend 元数据等。
        - runtime_peak_memory:
          只会在 prefill/decode 或图捕获时冲高的动态峰值，需要额外留在
          `runtime_headroom` 里，不能拿去分配 KV cache。

        最终公式是：

        `available_kv_cache_memory_bytes = static_memory_budget - steady_non_kv_cache_memory`

        在满足运行时峰值安全余量的前提下，还允许按需从未使用的
        `runtime_headroom` 借一部分预算给 KV cache。借用只在以下条件同时
        满足时触发：

        - 当前静态预算下的 KV cache 无法覆盖 `max_model_len * max_num_seqs`
          的目标容量；
        - `runtime_headroom` 仍有剩余（`runtime_peak_memory` 未挤满）。

        同时还要校验：

        - `runtime_peak_memory <= runtime_memory_headroom`
        - `static_memory_budget + runtime_peak_memory <= startup_free_memory`
        """
        # 若用户直接指定了 KV cache 大小，则跳过自动估算。
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            # still need a profile run which compiles the model for
            # max_num_batched_tokens
            # 仍要做一次 profile_run 来完成编译和 warmup。
            self.model_runner.profile_run()

            msg = (
                f"Initial free memory {format_gib(self.init_snapshot.free_memory)} "
                f"GiB, reserved {format_gib(kv_cache_memory_bytes)} GiB memory for "
                "KV Cache as specified by kv_cache_memory_bytes config and "
                "skipped memory profiling. This does not respect the "
                "gpu_memory_utilization config. Only use kv_cache_memory_bytes "
                "config when you want manual control of KV cache memory "
                "size. If OOM'ed, check the difference of initial free "
                "memory between the current run and the previous run "
                "where kv_cache_memory_bytes is suggested and update it "
                "correspondingly."
            )
            logger.info(msg)
            # 直接返回用户手工指定的 KV cache 大小。
            return kv_cache_memory_bytes

        # ------------------ 第 1 段：测出“非 KV”显存真实占用 ------------------
        # 通过 dummy forward + memory_profiling，得到权重、激活峰值、非 torch
        # 显存等信息，为后续把静态预算与运行时峰值拆开做准备。
        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            # 跑一次 dummy profile 触发真实内存峰值。
            self.model_runner.profile_run()

            # 读取 profiling 期间 PyTorch 侧的峰值显存。
            profile_torch_peak = current_platform.memory_stats(self.device).get(
                "allocated_bytes.all.peak", 0
            )

            # Profile CUDA graph memory if graphs will be captured.
            # 若后续会捕获 cudagraph，则额外估算其显存开销。
            cudagraph_memory_estimate = 0
            if not self.model_config.enforce_eager:
                cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()

        # ------------------ 第 2 段：把 profile 结果折叠成 steady / peak 两类口径 ------------------
        # Use the pre-cudagraph torch peak to avoid double-counting.
        # 计算 profiling 带来的 torch 峰值增量。
        profile_result.torch_peak_increase = (
            profile_torch_peak - profile_result.before_profile.torch_peak
        )
        # 汇总出 profile 期间观测到的非 KV cache 峰值。
        profile_result.non_kv_cache_memory = (
            profile_result.non_torch_increase
            + profile_result.torch_peak_increase
            + profile_result.weights_memory
        )

        # 仅在环境变量开启时，才把 cudagraph 估算值计入预算。
        cudagraph_memory_estimate_applied = (
            cudagraph_memory_estimate
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
            else 0
        )

        # steady_non_kv_cache_memory 只看 profile 结束后仍常驻的非 KV 显存。
        steady_non_kv_cache_memory = max(
            0,
            profile_result.after_profile.cuda_memory
            - profile_result.before_create.cuda_memory,
        )
        # runtime_peak_memory 表示 profile 期间额外冲高的动态显存。
        runtime_peak_memory = max(
            0,
            profile_result.non_kv_cache_memory - steady_non_kv_cache_memory,
        ) + cudagraph_memory_estimate_applied

        # 记录非 torch 常驻显存占用。
        self.non_torch_memory = profile_result.non_torch_increase
        # 记录 steady-state 非 KV 常驻显存。
        self.steady_non_kv_cache_memory = steady_non_kv_cache_memory
        # 记录运行时峰值需要额外预留的显存。
        self.runtime_peak_memory = runtime_peak_memory
        # 记录激活峰值与可能的 cudagraph 额外占用。
        self.peak_activation_memory = (
            profile_result.torch_peak_increase + cudagraph_memory_estimate_applied
        )
        # 保存 cudagraph 的估算结果。
        self.cudagraph_memory_estimate = cudagraph_memory_estimate

        # 读取 profiling 后剩余的空闲显存。
        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory >= free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {format_gib(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {format_gib(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container."
        )
        # ------------------ 第 3 段：校验“静态预算 + 运行时余量”是否站得住 ------------------
        # ratio 外的空间现在只保留给运行时峰值，不再拿来承接静态预算。
        if self.runtime_peak_memory > self.runtime_memory_headroom:
            raise ValueError(
                "Profiled runtime peak exceeds the GPU headroom left by "
                f"gpu_memory_utilization={self.cache_config.gpu_memory_utilization}. "
                f"Profiled runtime peak={format_gib(self.runtime_peak_memory)} GiB, "
                f"configured runtime headroom="
                f"{format_gib(self.runtime_memory_headroom)} GiB. "
                "Decrease gpu_memory_utilization or reduce runtime peak "
                "memory usage."
            )

        # 若有外部进程占用显存，则还要确保“静态预算 + 运行时峰值”在真实空闲显存内。
        required_free_memory = (
            self.static_memory_budget + self.runtime_peak_memory
        )
        if self.init_snapshot.free_memory < required_free_memory:
            raise ValueError(
                f"Free memory on device {self.init_snapshot.device_} "
                f"({format_gib(self.init_snapshot.free_memory)}/"
                f"{format_gib(self.init_snapshot.total_memory)} GiB) on startup "
                "is insufficient for the configured static budget plus the "
                f"profiled runtime peak. Static budget="
                f"{format_gib(self.static_memory_budget)} GiB, "
                f"profiled runtime peak={format_gib(self.runtime_peak_memory)} GiB. "
                "Reduce gpu_memory_utilization or free GPU memory used by "
                "other processes."
            )

        # ------------------ 第 4 段：导出最终可给 KV cache 的预算 ------------------
        # 用“静态预算 - 常驻非 KV 显存”得到可用于 KV cache 的空间。
        self.available_kv_cache_memory_bytes = (
            self.static_memory_budget - self.steady_non_kv_cache_memory
        )

        # ------------------ 第 4.1 段：在不破坏运行时峰值安全余量前提下，按需借用 headroom 扩容 KV ------------------
        # 只有当当前 KV 预算无法覆盖 max_model_len * max_num_seqs 时，才尝试借用。
        # 借用上限同时受两条约束：
        # 1) runtime headroom 剩余：runtime_memory_headroom - runtime_peak_memory
        # 2) startup free memory 剩余：init_free_memory - (static_budget + runtime_peak_memory)
        from cfie.v1.core.kv_cache_utils import max_memory_usage_bytes

        # 读取当前 worker 持有层的 KV 规格，并估算“单请求满上下文”所需 KV 字节数。
        worker_kv_cache_spec = self.model_runner.get_kv_cache_spec()
        single_req_kv_bytes = max_memory_usage_bytes(
            self.cfie_config, worker_kv_cache_spec.values()
        )
        required_req_count = max(1, self.scheduler_config.max_num_seqs)
        required_kv_bytes = single_req_kv_bytes * required_req_count

        # 当静态预算内的 KV 不足以覆盖 max_model_len * reqs 时，再尝试借用 headroom。
        # 这里允许初始 KV 预算 <= 0 的场景也参与借用。
        if (
            single_req_kv_bytes > 0
            and self.available_kv_cache_memory_bytes < required_kv_bytes
        ):
            headroom_spare = max(
                0,
                self.runtime_memory_headroom - self.runtime_peak_memory,
            )
            free_memory_spare = max(
                0,
                self.init_snapshot.free_memory
                - (self.static_memory_budget + self.runtime_peak_memory),
            )
            expandable_kv_bytes = min(headroom_spare, free_memory_spare)

            if expandable_kv_bytes > 0:
                expand_delta = min(
                    expandable_kv_bytes,
                    required_kv_bytes - self.available_kv_cache_memory_bytes,
                )
                self.available_kv_cache_memory_bytes += expand_delta
                logger.info_once(
                    "Expanded KV cache budget by borrowing runtime headroom: "
                    "delta=%s GiB, target=%s GiB, final=%s GiB.",
                    format_gib(expand_delta),
                    format_gib(required_kv_bytes),
                    format_gib(self.available_kv_cache_memory_bytes),
                    scope="local",
                )

        logger.debug(
            "Initial free memory: %s GiB; Static budget: %f (util), %s GiB; "
            "Runtime headroom: %s GiB",
            format_gib(self.init_snapshot.free_memory),
            self.cache_config.gpu_memory_utilization,
            format_gib(self.static_memory_budget),
            format_gib(self.runtime_memory_headroom),
        )
        logger.debug(
            "Free memory after profiling: %s GiB (total); steady non-KV: %s GiB; "
            "runtime peak: %s GiB",
            format_gib(free_gpu_memory),
            format_gib(self.steady_non_kv_cache_memory),
            format_gib(self.runtime_peak_memory),
        )
        logger.debug(profile_result)
        logger.info_once(
            "Available KV cache memory: %s GiB",
            format_gib(self.available_kv_cache_memory_bytes),
            scope="local",
        )
        logger.info_once(
            "GPU memory split: static budget=%s GiB, runtime headroom=%s GiB, "
            "steady non-KV=%s GiB, profiled runtime peak=%s GiB",
            format_gib(self.static_memory_budget),
            format_gib(self.runtime_memory_headroom),
            format_gib(self.steady_non_kv_cache_memory),
            format_gib(self.runtime_peak_memory),
            scope="local",
        )

        if cudagraph_memory_estimate > 0:
            total_mem = self.init_snapshot.total_memory
            current_util = self.cache_config.gpu_memory_utilization
            cg_util_delta = cudagraph_memory_estimate / total_mem
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:
                equiv_util = round(current_util - cg_util_delta, 4)
                suggested_util = min(
                    round(current_util + cg_util_delta, 4),
                    1.0,
                )
                logger.info(
                    "CUDA graph memory profiling is enabled "
                    "(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1). "
                    "This will become the default in v0.19. "
                    "The current --gpu-memory-utilization=%.4f is equivalent "
                    "to --gpu-memory-utilization=%.4f without CUDA graph "
                    "memory profiling. To maintain the same effective KV "
                    "cache size as before, increase "
                    "--gpu-memory-utilization to %.4f.",
                    current_util,
                    equiv_util,
                    suggested_util,
                )
            else:
                suggested_util = min(
                    round(current_util + cg_util_delta, 4),
                    1.0,
                )
                logger.info(
                    "In v0.19, CUDA graph memory profiling will be enabled "
                    "by default (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1), "
                    "which more accurately accounts for CUDA graph memory "
                    "during KV cache allocation. To try it now, set "
                    "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 and increase "
                    "--gpu-memory-utilization from %.4f to %.4f to maintain "
                    "the same effective KV cache size.",
                    current_util,
                    suggested_util,
                )

        # 返回最终可用于 KV cache 的字节数。
        return int(self.available_kv_cache_memory_bytes)

    # 从 worker 侧导出 KV connector 握手信息，供 scheduler/connector 对齐状态。
    def get_kv_connector_handshake_metadata(self) -> dict | None:
        """Get KV connector metadata from this worker if available."""

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        tp_rank = get_tp_group().rank_in_group
        return {tp_rank: metadata}

    # 向上层暴露当前模型需要的 KV cache 规格。
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    # 在 auto-fit 调整了上下文长度后，把新的 max_model_len 同步到 runner。
    def update_max_model_len(self, max_model_len: int) -> None:
        """Update max_model_len after auto-fit to GPU memory.

        This is called when max_model_len=-1 is used and the engine
        automatically determines the maximum context length that fits
        in GPU memory. Workers need to update their cached max_model_len
        to match the engine's decision.
        """
        self.model_config.max_model_len = max_model_len
        if self.model_runner is not None:
            self.model_runner.update_max_model_len(max_model_len)
        logger.debug("Updated max_model_len to %d", max_model_len)

    @instrument(span_name="Allocate KV cache")
    # 按 engine core 给出的最终配置分配 KV cache，并完成缓存相关初始化。
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """
        按已经规划好的 `KVCacheConfig` 真正分配并绑定 GPU KV cache。

        这一步不再重新做预算决策，而是把上游已经算好的：

        - `num_blocks`
        - `kv_cache_groups`
        - 每个 tensor 的共享关系与大小

        materialize 成实际的 KV tensor、metadata builder 和 transfer 侧注册项。
        """

        # Update local config with adjusted num blocks after profiling,
        # so that it's available to the warmup stage.
        # 把 profiling 后确定的 num_gpu_blocks 回写到本地 cache_config。
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g. kv cache sharing layers).
        # 先初始化 KV transfer/connector，确保其看见最终配置。
        ensure_kv_transfer_initialized(self.cfie_config, kv_cache_config)

        if self.cfie_config.model_config.enable_sleep_mode:
            from cfie.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            with allocator.use_memory_pool(tag="kv_cache"):
                # sleep mode 下在专用内存池中分配 KV cache。
                self.model_runner.initialize_kv_cache(kv_cache_config)
        else:
            # 常规模式下直接初始化 KV cache。
            self.model_runner.initialize_kv_cache(kv_cache_config)

        if (
            self.model_config.enable_return_routed_experts
            or self.model_config.enable_return_router_logits
        ):
            # 若需要返回 routed experts，则初始化捕获器。
            self.model_runner.init_routed_experts_capturer()

        # Build KV-zero metadata outside the CuMem pool so the bookkeeping
        # GPU tensors (seg_addrs, block-id buffers) use the standard PyTorch
        # allocator and are not discarded during sleep/wake cycles.
        if kv_cache_config.needs_kv_cache_zeroing and hasattr(
            self.model_runner, "_init_kv_zero_meta"
        ):
            # 如需 KV zeroing，则补建 zeroing 元数据。
            self.model_runner._init_kv_zero_meta()

    @instrument(span_name="Warmup (GPU)")
    # 在 KV cache 建好后做编译或 warmup，为后续正式执行准备运行时状态。
    def compile_or_warm_up_model(self) -> float:
        warmup_sizes: list[int] = []

        if self.cfie_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
            # warm up sizes that are not in cudagraph capture sizes,
            # but users still want to compile for better performance,
            # e.g. for the max-num-batched token size in chunked prefill.
            compile_sizes = self.cfie_config.compilation_config.compile_sizes
            warmup_sizes = compile_sizes.copy() if compile_sizes is not None else []  # type: ignore[assignment]
            cg_capture_sizes: list[int] = []

            if self.cfie_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                cg_sizes = self.cfie_config.compilation_config.cudagraph_capture_sizes
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]

            compile_ranges = self.cfie_config.compilation_config.get_compile_ranges()
            # For each compile_range, if none of the batch sizes
            # in warmup_sizes or cudagraph_capture_sizes are in the range,
            # add the end of the range to ensure compilation/warmup.
            all_sizes = set(cg_capture_sizes)
            all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])
            for compile_range in compile_ranges:
                if not any(x in compile_range for x in all_sizes):
                    warmup_sizes.append(compile_range.end)

        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        kernel_warmup(self)

        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            cuda_graph_memory_bytes = self.model_runner.capture_model()

        # Compare actual vs estimated CUDA graph memory (if we did profiling)
        if (
            hasattr(self, "cudagraph_memory_estimate")
            and self.cudagraph_memory_estimate > 0
        ):
            GiB = lambda b: round(b / GiB_bytes, 2)
            diff = abs(cuda_graph_memory_bytes - self.cudagraph_memory_estimate)
            logger.info(
                "CUDA graph pool memory: %s GiB (actual), %s GiB (estimated), "
                "difference: %s GiB (%.1f%%).",
                GiB(cuda_graph_memory_bytes),
                GiB(self.cudagraph_memory_estimate),
                GiB(diff),
                100 * diff / max(cuda_graph_memory_bytes, 1),
            )

        if self.cache_config.kv_cache_memory_bytes is None and hasattr(
            self, "peak_activation_memory"
        ):
            # Suggests optimal kv cache memory size if we rely on
            # memory_profiling to guess the kv cache memory size which
            # provides peak_activation_memory and a few other memory
            # consumption. `memory_profiling` does not consider
            # CUDAGraph memory size and may not utilize all gpu memory.
            # Users may want fine-grained control to specify kv cache
            # memory size.

            # empirically observed that the memory profiling may
            # slightly underestimate the memory consumption.
            # So leave a small buffer (=150MiB) to avoid OOM.
            redundancy_buffer_memory = 150 * (1 << 20)
            kv_cache_memory_bytes_to_gpu_limit = (
                self.init_snapshot.free_memory
                - self.runtime_peak_memory
                - self.steady_non_kv_cache_memory
                - redundancy_buffer_memory
            )
            kv_cache_memory_bytes_to_requested_limit = (
                int(self.static_memory_budget)
                - self.steady_non_kv_cache_memory
                - redundancy_buffer_memory
            )

            msg = (
                f"Free memory on device "
                f"({format_gib(self.init_snapshot.free_memory)}/"
                f"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Configured static GPU budget is "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.static_memory_budget)} GiB). "
                f"Configured runtime headroom is "
                f"{format_gib(self.runtime_memory_headroom)} GiB; profiled runtime "
                f"peak is {format_gib(self.runtime_peak_memory)} GiB. "
                f"Steady non-KV memory is "
                f"{format_gib(self.steady_non_kv_cache_memory)} GiB. "
                f"Actual usage is {format_gib(self.model_runner.model_memory_usage)} "
                f"GiB for weight, {format_gib(self.peak_activation_memory)} GiB "
                f"for peak activation, {format_gib(self.non_torch_memory)} GiB "
                f"for non-torch memory, and {format_gib(cuda_graph_memory_bytes)} "
                f"GiB for CUDAGraph memory. Replace gpu_memory_utilization "
                f"config with `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_requested_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_requested_limit)} GiB) to fit "
                f"into the configured static budget, or `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_gpu_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_gpu_limit)} GiB) to fully "
                f"utilize startup-free gpu memory while preserving the profiled "
                f"runtime peak. Current kv cache memory in use is "
                f"{format_gib(self.available_kv_cache_memory_bytes)} GiB."
            )

            logger.debug(msg)

        if self.use_v2_model_runner:
            # V2: Run full execute_model + sample_tokens to JIT compile triton kernels.
            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
        elif get_pp_group().is_last_rank:
            # V1: Warm up sampler and preallocate memory buffer for logits and other
            # sampling related tensors of max possible shape to avoid memory
            # fragmentation issue.
            # NOTE: This is called after `capture_model` on purpose to prevent
            # memory buffers from being cleared by `torch.accelerator.empty_cache`.
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )

            # We skip EPLB here since we don't want to record dummy metrics
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                num_tokens=max_num_reqs,
                skip_eplb=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

        return self.compilation_config.compilation_time

    def reset_mm_cache(self) -> None:
        self.model_runner.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        self.model_runner.reset_encoder_cache()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    # ------------------------------- 打开 predictor trace 捕获 -------------------------------
    def enable_predictor_capture(self, layer_ids: tuple[int, ...]) -> None:
        enable_fn = getattr(self.model_runner, "enable_predictor_capture", None)
        if not callable(enable_fn):
            raise RuntimeError("current model runner does not support predictor capture")
        enable_fn(layer_ids)

    # ------------------------------- 关闭 predictor trace 捕获 -------------------------------
    def disable_predictor_capture(self) -> None:
        disable_fn = getattr(self.model_runner, "disable_predictor_capture", None)
        if callable(disable_fn):
            disable_fn()

    # ------------------------------- 取走已完成的 predictor hidden states -------------------------------
    def take_predictor_hidden_states(
        self,
        request_ids: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, dict[str, Any]]:
        take_fn = getattr(self.model_runner, "take_predictor_hidden_states", None)
        if not callable(take_fn):
            raise RuntimeError("current model runner does not support predictor capture")
        return take_fn(request_ids)

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def get_encoder_timing_stats(self) -> dict[str, dict[str, float | int]]:
        """Get encoder timing stats from model runner."""
        return self.model_runner.get_encoder_timing_stats()

    def annotate_profile(self, scheduler_output):
        # add trace annotation so that we can easily distinguish
        # context/generation request numbers in each iteration.
        # A context request is a request that has not yet generated any tokens
        if not self.profiler:
            return nullcontext()

        self.profiler.step()

        iteration_details = compute_iteration_details(scheduler_output)

        annotation = "".join(
            [
                "execute_context_",
                str(iteration_details.num_ctx_requests),
                "(",
                str(iteration_details.num_ctx_tokens),
                ")_generation_",
                str(iteration_details.num_generation_requests),
                "(",
                str(iteration_details.num_generation_tokens),
                ")",
            ]
        )
        return self.profiler.annotate_context_manager(annotation)

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    @torch.inference_mode()
    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        # ----------------- worker 侧执行入口 -----------------
        # 先确保上一轮 pipeline parallel 的异步发送已经完成，避免交叉覆盖缓冲区。
        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []

        intermediate_tensors = None
        # 只要本轮有 token 工作量，就需要真正触发前向。
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        all_gather_tensors = {}
        compilation_config = self.cfie_config.compilation_config
        parallel_config = self.cfie_config.parallel_config

        if (
            parallel_config.pipeline_parallel_size > 1
            and compilation_config.pass_config.enable_sp
            and forward_pass
        ):
            # 目前 SP 只接在 V1 GPUModelRunner 路径上。
            assert not self.use_v2_model_runner
            num_scheduled_tokens_np = np.array(
                list(scheduler_output.num_scheduled_tokens.values()),
                dtype=np.int32,
            )
            # 这里预先推导 batch 形状，只为确定 PP/SP 需要 all-gather 哪些张量。
            _, batch_desc, _, _, _ = (
                self.model_runner._determine_batch_execution_and_padding(
                    num_tokens=num_scheduled_tokens,
                    num_reqs=len(num_scheduled_tokens_np),
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    max_num_scheduled_tokens=num_scheduled_tokens_np.max(),
                    use_cascade_attn=False,  # TODO(lucas): Handle cascade attention
                )
            )
            all_gather_tensors = {
                "residual": not is_residual_scattered_for_sp(
                    self.cfie_config, batch_desc.num_tokens
                )
            }

        if forward_pass and not get_pp_group().is_first_rank:
            # 非首个 PP rank 需要先从上一段接收中间张量，再继续本段前向。
            tensor_dict, comm_handles, comm_postprocess = (
                get_pp_group().irecv_tensor_dict(
                    all_gather_group=get_tp_group(),
                    all_gather_tensors=all_gather_tensors,
                )
            )
            assert tensor_dict is not None
            intermediate_tensors = AsyncIntermediateTensors(
                tensor_dict,
                comm_handles=comm_handles,
                comm_postprocess=comm_postprocess,
            )

        with self.annotate_profile(scheduler_output):
            # scheduler_output 在这里原样交给 GPUModelRunner 消费。
            output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            if (
                self.use_v2_model_runner
                and self.model_runner.is_pooling_model
                and output is None
            ):
                output = self.model_runner.pool()  # type: ignore
            if isinstance(
                output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType
            ):
                return output

        assert isinstance(output, IntermediateTensors)
        parallel_config = self.cfie_config.parallel_config
        assert (
            parallel_config.distributed_executor_backend != "external_launcher"
            and not get_pp_group().is_last_rank
        )

        # 非最后一个 PP rank 只负责继续异步把中间张量发给下游阶段。
        self._pp_send_work = get_pp_group().isend_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group(),
            all_gather_tensors=all_gather_tensors,
        )

        return None

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.model_runner.take_draft_token_ids()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # Check if profiling is enabled
        if self.profiler_config is None or self.profiler_config.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. Please set --profiler-config to enable "
                "profiling. Example: "
                "'--profiler-config.profiler=torch --profiler-config.torch_profiler_dir"
                "=YOUR_DIR_PATH_TO_DUMP_TRACE'"
            )

        if is_start:
            # Generate the trace name by combining prefix with comprehensive rank suffix
            from cfie.distributed.utils import get_worker_rank_suffix

            rank_suffix = get_worker_rank_suffix(global_rank=self.rank)

            # Build the full trace name
            if profile_prefix:
                trace_name = f"{profile_prefix}_{rank_suffix}"
            else:
                trace_name = rank_suffix

            # Create the profiler wrapper only on the first start call
            if self.profiler is None:
                profiler_type = self.profiler_config.profiler
                if profiler_type == "torch":
                    self.profiler = TorchProfilerWrapper(
                        self.profiler_config,
                        worker_name=trace_name,
                        local_rank=self.local_rank,
                        activities=["CPU", "CUDA"],
                    )
                    logger.debug(
                        "Starting torch profiler with trace name: %s", trace_name
                    )
                elif profiler_type == "cuda":
                    self.profiler = CudaProfilerWrapper(self.profiler_config)
                    logger.debug("Starting CUDA profiler")
                else:
                    # Config validation should prevent this code being reached
                    raise ValueError(
                        f"Invalid profiler value of {self.profiler_config.profiler}"
                    )

            # If profiler already initialized, restart profiling but keep
            # the original trace name from the first initialization.
            self.profiler.start()
        else:
            if self.profiler is None:
                logger.warning("Profiler was not started, nothing to stop.")
                return
            self.profiler.stop()

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1, uniform_decode=True)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from cfie.model_executor.model_loader import ShardedStateLoader

        ShardedStateLoader.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(self, tensorizer_config: "TensorizerConfig") -> None:
        TensorizerLoader.save_model(
            self.get_model(),
            tensorizer_config=tensorizer_config,
            model_config=self.model_config,
        )

    def init_weight_transfer_engine(self, init_info: dict) -> None:
        """
        Initialize weight transfer mechanism.
        For NCCL backend, this creates a process group with the trainer.

        Args:
            init_info: Dictionary containing backend-specific initialization info
        """
        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. "
                "Please set weight_transfer_config to enable weight transfer."
            )
        # Parse dict into backend-specific typed dataclass
        typed_init_info = self.weight_transfer_engine.parse_init_info(init_info)
        self.weight_transfer_engine.init_transfer_engine(typed_init_info)

    def update_weights(self, update_info: dict) -> None:
        """
        Batched weight update from the trainer.

        Args:
            update_info: Dictionary containing backend-specific update info
        """
        if self.weight_transfer_engine is None:
            raise RuntimeError(
                "Weight transfer not configured. "
                "Please set weight_transfer_config to enable weight transfer."
            )

        # Parse dict into backend-specific typed dataclass
        typed_update_info = self.weight_transfer_engine.parse_update_info(update_info)

        model = self.model_runner.model

        if typed_update_info.is_checkpoint_format:
            from cfie.model_executor.model_loader.reload import (
                finalize_layerwise_reload,
                initialize_layerwise_reload,
            )

            # Use layerwise reload pattern for checkpoint format weights
            with torch.device(self.device):
                initialize_layerwise_reload(model)
                self.weight_transfer_engine.receive_weights(
                    typed_update_info,
                    load_weights=model.load_weights,
                )
                finalize_layerwise_reload(model, self.model_config)
        else:
            # Weights are already in kernel format, copy directly
            def load_weights_direct(
                weights: list[tuple[str, torch.Tensor]],
            ) -> None:
                for name, weight in weights:
                    param = model.get_parameter(name)
                    param.copy_(weight)

            self.weight_transfer_engine.receive_weights(
                typed_update_info,
                load_weights=load_weights_direct,
            )

        # NCCL broadcast/packed path are asynchronous.
        # Sync here so the next step uses the new weights.
        torch.accelerator.synchronize()

    def shutdown(self) -> None:
        # has_kv_transfer_group can be None during interpreter shutdown.
        if ensure_kv_transfer_shutdown is not None:
            ensure_kv_transfer_shutdown()
        if self.profiler is not None:
            self.profiler.shutdown()

        if weight_transfer_engine := getattr(self, "weight_transfer_engine", None):
            weight_transfer_engine.shutdown()

    def elastic_ep_execute(self, execute_method: str, *args, **kwargs):
        return self.elastic_ep_executor.execute(execute_method, *args, **kwargs)


def init_worker_distributed_environment(
    cfie_config: CfieConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment."""
    attention_config = cfie_config.attention_config
    parallel_config = cfie_config.parallel_config
    from cfie.model_executor.layers.batch_invariant import init_batch_invariance

    init_batch_invariance(attention_config.backend)
    override_envs_for_eplb(parallel_config)
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_method = distributed_init_method or "env://"

    timeout = None
    if parallel_config.distributed_timeout_seconds is not None:
        timeout = timedelta(seconds=parallel_config.distributed_timeout_seconds)

    init_distributed_environment(
        parallel_config.world_size,
        rank,
        init_method,
        local_rank,
        backend,
        timeout,
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
    )

    # Init ec connector here before KV caches init
    # NOTE: We do not init KV caches for Encoder-only instance in EPD disagg mode
    ensure_ec_transfer_initialized(cfie_config)
