# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cached_property
from multiprocessing import Lock
from typing import Any

import torch
import torch.distributed as dist

import cfie.envs as envs
from cfie.logger import init_logger
from cfie.platforms import current_platform
from cfie.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from cfie.v1.core.sched.output import GrammarOutput, SchedulerOutput
from cfie.v1.executor.abstract import Executor
from cfie.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from cfie.v1.serial_utils import run_method
from cfie.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class UniProcExecutor(Executor):
    # 初始化单进程执行器，创建 driver worker 并在需要时直接加载模型。
    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        self.driver_worker = WorkerWrapperBase(rpc_rank=0)
        distributed_init_method, rank, local_rank = self._distributed_args()
        kwargs = dict(
            cfie_config=self.cfie_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=True,
            shared_worker_lock=Lock(),
        )

        self.async_output_thread: ThreadPoolExecutor | None = None
        if self.max_concurrent_batches > 1:
            self.async_output_thread = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="WorkerAsyncOutput"
            )

        is_eep_new_worker = envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH
        self.driver_worker.init_worker(all_kwargs=[kwargs])
        if not is_eep_new_worker:
            self.driver_worker.init_device()
            self.driver_worker.load_model()
            current_platform.update_block_size_for_backend(self.cfie_config)

    # 生成单进程执行器所需的分布式初始化参数。
    def _distributed_args(self) -> tuple[str, int, int]:
        """Return (distributed_init_method, rank, local_rank)."""
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
        # 若 device 已显式绑定到某张卡，则把其设备索引直接作为 local rank。
        device_info = self.cfie_config.device_config.device.__str__().split(":")
        local_rank = int(device_info[1]) if len(device_info) > 1 else 0
        return distributed_init_method, 0, local_rank

    @cached_property
    # 根据是否开启异步调度决定可并发的 batch 数量。
    def max_concurrent_batches(self) -> int:
        return 2 if self.scheduler_config.async_scheduling else 1

    # 在单 worker 上执行“伪 collective RPC”，兼容统一的 executor 接口。
    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        if not non_block:
            result = run_method(self.driver_worker, method, args, kwargs)
            return result if single_value else [result]

        try:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                if (async_thread := self.async_output_thread) is not None:
                    if single_value:
                        return async_thread.submit(result.get_output)

                    def get_output_list() -> list[Any]:
                        return [result.get_output()]

                    return async_thread.submit(get_output_list)
                result = result.get_output()
            future = Future[Any]()
            future.set_result(result if single_value else [result])
        except Exception as e:
            future = Future[Any]()
            future.set_exception(e)
        return future

    # 把一次调度结果转发给 driver worker 执行模型。
    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        output = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            single_value=True,
        )
        # 非阻塞模式下若底层立即失败，这里直接把异常抛出来，避免 future 静默挂着。
        if non_block and output.done():
            # 直接在当前调用栈暴露 worker 侧异常。
            output.result()
        return output

    # 在需要二段采样时，把 grammar 输出转发给 worker 继续采样。
    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            single_value=True,
        )

    # 读取 speculative decoding 草稿 token 的回传结果。
    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.collective_rpc("take_draft_token_ids", single_value=True)

    # 单进程执行器没有独立远端 worker，因此只要进程活着就视为健康。
    def check_health(self) -> None:
        # 单进程模式下没有额外心跳目标，只要当前进程仍存活就视为健康。
        return

    # 关闭 driver worker 并回收执行器资源。
    def shutdown(self) -> None:
        if worker := self.driver_worker:
            worker.shutdown()


class ExecutorWithExternalLauncher(UniProcExecutor):
    """An executor that uses external launchers to launch engines,
    specially designed for torchrun-compatible launchers, for
    offline inference with tensor parallelism.

    see https://github.com/cfie-project/cfie/issues/11400 for
    the motivation, and examples/offline_inference/torchrun_example.py
    for the usage example.

    The key idea: although it is tensor-parallel inference, we only
    create one worker per executor, users will launch multiple
    engines with torchrun-compatible launchers, and all these engines
    work together to process the same prompts. When scheduling is
    deterministic, all the engines will generate the same outputs,
    and they don't need to synchronize the states with each other.
    """

    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        assert not envs.VLLM_ENABLE_V1_MULTIPROCESSING, (
            "To get deterministic execution, "
            "please set VLLM_ENABLE_V1_MULTIPROCESSING=0"
        )
        super()._init_executor()

    def _distributed_args(self) -> tuple[str, int, int]:
        # engines are launched in torchrun-compatible launchers
        # so we can use the env:// method.
        # required env vars:
        # - RANK
        # - LOCAL_RANK
        # - MASTER_ADDR
        # - MASTER_PORT
        distributed_init_method = "env://"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        return distributed_init_method, rank, local_rank

    def determine_available_memory(self) -> list[int]:  # in bytes
        # we need to get the min across all ranks.
        memory = super().determine_available_memory()
        from cfie.distributed.parallel_state import get_world_group

        cpu_group = get_world_group().cpu_group
        memory_tensor = torch.tensor([memory], device="cpu", dtype=torch.int64)
        dist.all_reduce(memory_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        return [memory_tensor.item()]
