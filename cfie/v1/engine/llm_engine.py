# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Callable, Mapping
from copy import copy
from typing import Any

import torch.nn as nn
from typing_extensions import TypeVar

import cfie.envs as envs
from cfie.config import ParallelConfig, CfieConfig
from cfie.distributed import stateless_destroy_torch_distributed_process_group
from cfie.distributed.parallel_state import get_dp_group
from cfie.engine.arg_utils import EngineArgs
from cfie.inputs import ProcessorInputs, PromptType
from cfie.logger import init_logger
from cfie.lora.request import LoRARequest
from cfie.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from cfie.outputs import PoolingRequestOutput, RequestOutput
from cfie.plugins.io_processors import get_io_processor
from cfie.pooling_params import PoolingParams
from cfie.renderers import renderer_from_config
from cfie.renderers.inputs.preprocess import extract_prompt_components
from cfie.sampling_params import SamplingParams
from cfie.tasks import SupportedTask
from cfie.tokenizers import TokenizerLike
from cfie.tracing import init_tracer
from cfie.usage.usage_lib import UsageContext
from cfie.v1.engine import EngineCoreRequest, PauseMode
from cfie.v1.engine.core_client import EngineCoreClient
from cfie.v1.engine.input_processor import InputProcessor
from cfie.v1.engine.output_processor import OutputProcessor
from cfie.v1.engine.parallel_sampling import ParentRequest
from cfie.v1.executor import Executor
from cfie.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager
from cfie.v1.metrics.reader import Metric, get_metrics_snapshot
from cfie.v1.metrics.stats import IterationStats
from cfie.v1.utils import record_function_or_nullcontext
from cfie.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


def _should_enable_v1_multiprocessing(
        cfie_config: CfieConfig,
        *,
        requested: bool,
) -> bool:
    parallel_config = cfie_config.parallel_config
    # 多 rank 场景必须保留 multiprocessing 路径；单 rank 再允许走本地直通。
    requires_multiprocessing = (
        parallel_config.world_size > 1
        or parallel_config.data_parallel_size > 1
    )
    return requested or requires_multiprocessing


class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""

    # 初始化高层引擎对象，串起 renderer、processor、engine core client 与统计组件。
    def __init__(
            self,
            cfie_config: CfieConfig,
            executor_class: type[Executor],
            log_stats: bool,
            aggregate_engine_logging: bool = False,
            usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
            stat_loggers: list[StatLoggerFactory] | None = None,
            mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
            use_cached_outputs: bool = False,
            multiprocess_mode: bool = False,
    ) -> None:
        # 保存全局配置对象，供后续所有子组件复用。
        self.cfie_config = cfie_config

        # 缓存模型配置，减少层层取属性。
        self.model_config = cfie_config.model_config

        # 缓存观测配置，决定 tracing 和指标行为。
        self.observability_config = cfie_config.observability_config

        # 读取 OTLP tracing 上报地址。
        tracing_endpoint = self.observability_config.otlp_traces_endpoint

        # 若开启 tracing，则先初始化 tracer。
        if tracing_endpoint is not None:
            init_tracer("cfie.llm_engine", tracing_endpoint)

        # 记录是否启用统计输出。
        self.log_stats = log_stats

        # 取出并行配置。
        parallel_config = cfie_config.parallel_config

        # 取出执行器后端类型。
        executor_backend = parallel_config.distributed_executor_backend

        # 标记是否处于 external launcher 的数据并行模式。
        self.external_launcher_dp = (
                parallel_config.data_parallel_size > 1
                and executor_backend == "external_launcher"
        )

        if (
                not multiprocess_mode
                and parallel_config.data_parallel_size > 1
                and not self.external_launcher_dp
        ):
            # 当前进程内直接初始化 DP group。
            self.dp_group = parallel_config.stateless_init_dp_group()
        else:
            # 非本地 DP 场景下先不持有本地 dp_group。
            self.dp_group = None

        # 标记下轮是否需要执行 dummy batch 来保持 DP 同步。
        self.should_execute_dummy_batch = False

        # 根据配置创建 prompt/chat renderer。
        self.renderer = renderer = renderer_from_config(self.cfie_config)

        # 创建输入输出层面的 IO processor。
        self.io_processor = get_io_processor(
            self.cfie_config,
            self.renderer,
            self.model_config.io_processor_plugin,
        )

        # Convert TokPrompt --> EngineCoreRequest.
        # 创建输入处理器，把外部 prompt 转成 EngineCoreRequest。
        self.input_processor = InputProcessor(self.cfie_config, renderer)

        # Converts EngineCoreOutputs --> RequestOutput.
        # 创建输出处理器，把底层增量输出还原成用户可见结果。
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.cfie_config.scheduler_config.stream_interval,
            tracing_enabled=tracing_endpoint is not None,
        )

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        # 创建 engine core client，按是否多进程选择 inproc/mp 实现。
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,  # True
            asyncio_mode=False,
            cfie_config=cfie_config,
            executor_class=executor_class,  # UniProcExec
            log_stats=self.log_stats,
        )

        # 默认先不创建统计 logger manager。
        self.logger_manager: StatLoggerManager | None = None

        # 若开启统计，则初始化各类指标记录器。
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                cfie_config=cfie_config,
                custom_stat_loggers=stat_loggers,
                enable_default_loggers=log_stats,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        if not multiprocess_mode:
            # 非多进程模式下暴露底层 model_executor 以兼容旧调用。
            self.model_executor = self.engine_core.engine_core.model_executor  # type: ignore

        if self.external_launcher_dp:
            # 复用外部 launcher 已经建立好的 DP group。
            self.dp_group = get_dp_group().cpu_group

        # Don't keep the dummy data in memory
        # 清空多模态临时缓存，避免初始化后长期占内存。
        self.reset_mm_cache()

    @classmethod
    # 直接从现成的 `CfieConfig` 构建 `LLMEngine`。
    def from_cfie_config(
            cls,
            cfie_config: CfieConfig,
            usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
            stat_loggers: list[StatLoggerFactory] | None = None,
            disable_log_stats: bool = False,
    ) -> "LLMEngine":
        # 直接用现成配置实例化 LLMEngine。
        return cls(
            cfie_config=cfie_config,
            executor_class=Executor.get_class(cfie_config),
            log_stats=(not disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=_should_enable_v1_multiprocessing(
                cfie_config,
                requested=envs.VLLM_ENABLE_V1_MULTIPROCESSING,
            ),
        )

    @classmethod
    # 从 `EngineArgs` 出发创建 `CfieConfig`，再实例化 `LLMEngine`。
    def from_engine_args(
            cls,
            engine_args: EngineArgs,
            usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
            stat_loggers: list[StatLoggerFactory] | None = None,
            enable_multiprocessing: bool = False,
    ) -> "LLMEngine":

        # 先把 EngineArgs 展开成完整的 CfieConfig。
        cfie_config = engine_args.create_engine_config(usage_context)

        # 再根据配置选择具体执行器类型。
        executor_class = Executor.get_class(cfie_config)

        # 若环境变量强制开启多进程，则覆盖调用方传参。
        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.debug("Enabling multiprocessing for LLMEngine.")
            enable_multiprocessing = True
        enable_multiprocessing = _should_enable_v1_multiprocessing(
            cfie_config,
            requested=enable_multiprocessing,
        )

        # 用最终配置实例化高层引擎对象。
        return cls(
            cfie_config=cfie_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=enable_multiprocessing,
        )

    # 返回当前仍未完成的请求数量。
    def get_num_unfinished_requests(self) -> int:
        # 直接返回输出处理器维护的未完成请求数。
        return self.output_processor.get_num_unfinished_requests()

    # 判断本地与分布式侧是否还有未完成请求，用于驱动 chat 主循环继续 step。
    def has_unfinished_requests(self) -> bool:
        # 先看当前 rank 本地是否仍有未完成请求。
        has_unfinished = self.output_processor.has_unfinished_requests()
        # 非 DP 模式下，再结合 engine core 的全局运行状态返回。
        if self.dp_group is None:
            return has_unfinished or self.engine_core.dp_engines_running()
        # DP 模式下需要汇总各 rank 状态。
        return self.has_unfinished_requests_dp(has_unfinished)

    # 在数据并行场景下汇总各 rank 的未完成状态。
    def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool:
        # 通过 DP 通信把各 rank 的未完成状态做一次聚合。
        aggregated_has_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, has_unfinished
        )
        # 本地空闲但全局仍忙时，下轮需要执行 dummy batch 参与同步。
        if not has_unfinished and aggregated_has_unfinished:
            self.should_execute_dummy_batch = True
        # 返回聚合后的全局状态。
        return aggregated_has_unfinished

    # 懒加载并缓存底层模型支持的任务类型。
    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        # 首次访问时从 engine core 拉取支持的任务列表。
        if not hasattr(self, "_supported_tasks"):
            # Cache the result
            self._supported_tasks = self.engine_core.get_supported_tasks()

        # 后续直接复用缓存结果。
        return self._supported_tasks

    # 同时从输出处理器和 engine core 中中止指定请求。
    def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        # 先让输出处理器清理本地请求状态。
        request_ids = self.output_processor.abort_requests(request_ids, internal)
        # 再通知 engine core 中止底层执行。
        self.engine_core.abort_requests(request_ids)

    # 把外部 prompt/params 包装成 `EngineCoreRequest` 并送入 engine core。
    def add_request(
            self,
            request_id: str,
            prompt: EngineCoreRequest | PromptType | ProcessorInputs,
            params: SamplingParams | PoolingParams,
            arrival_time: float | None = None,
            lora_request: LoRARequest | None = None,
            tokenization_kwargs: dict[str, Any] | None = None,
            trace_headers: Mapping[str, str] | None = None,
            priority: int = 0,
            prompt_text: str | None = None,
    ) -> str:
        # Validate the request_id type.
        # 约束 request_id 必须是字符串，避免后续调度键类型混乱。
        if not isinstance(request_id, str):
            raise TypeError(f"request_id must be a string, got {type(request_id)}")

        # Process raw inputs into the request.
        # 若调用方直接传入底层 EngineCoreRequest，则尽量兼容。
        if isinstance(prompt, EngineCoreRequest):
            logger.warning_once(
                "Passing EngineCoreRequest to LLMEngine.generate() and .add_requests() "
                "is deprecated and will be removed in v0.18. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "LLMEngine.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            # 正常路径下，把 prompt 与采样参数编码成 EngineCoreRequest。
            request = self.input_processor.process_inputs(
                request_id,
                prompt,
                params,
                supported_tasks=self.get_supported_tasks(),
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
            )
            # 同步提取 prompt 文本，供输出层记录与回显使用。
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

        # 统一由 input processor 最终分配内部 request_id。
        self.input_processor.assign_request_id(request)

        # 取出最终 request_id，后续返回给调用方。
        req_id = request.request_id

        # Use cloned params that may have been updated in process_inputs()
        # 取回经过输入处理后可能已修正的 params。
        params = request.params

        # n>1 表示需要做并行采样扇出。
        n = params.n if isinstance(params, SamplingParams) else 1

        if n == 1:
            # Make a new RequestState and queue.
            # 在输出处理器中登记这条请求的状态。
            self.output_processor.add_request(request, prompt_text, None, 0)
            # Add the request to EngineCore.
            # 把请求正式交给 engine core 调度执行。
            self.engine_core.add_request(request)
            # 单请求场景直接返回 request_id。
            return req_id

        # Fan out child requests (for n>1).
        # 多采样场景下先创建父请求管理器。
        parent_req = ParentRequest(request)
        # 逐个生成子请求并分别送入输出处理器和 engine core。
        for idx in range(n):
            # 为每个子样本分配独立 request_id 与采样参数。
            request_id, child_params = parent_req.get_child_info(idx)
            # 最后一个子请求复用原对象，其余子请求做浅拷贝。
            child_request = request if idx == n - 1 else copy(request)
            # 写入子请求 request_id。
            child_request.request_id = request_id
            # 写入子请求对应的采样参数。
            child_request.sampling_params = child_params

            # Make a new RequestState and queue.
            # 为子请求登记输出状态并记录其在父请求中的位置。
            self.output_processor.add_request(
                child_request, prompt_text, parent_req, idx
            )
            # Add the request to EngineCore.
            # 把子请求逐个提交给 engine core。
            self.engine_core.add_request(child_request)

        # 返回父请求对应的 request_id。
        return req_id

    # 执行一次高层 step：取 engine core 输出、后处理并返回用户可见结果。
    def step(self) -> list[RequestOutput | PoolingRequestOutput]:
        # 若上一轮 DP 聚合要求执行 dummy batch，则先补这一轮同步。
        if self.should_execute_dummy_batch:
            self.should_execute_dummy_batch = False
            self.engine_core.execute_dummy_batch()
            return []

        # 1) Get EngineCoreOutput from the EngineCore.
        with record_function_or_nullcontext("llm_engine step: get_output"):
            # 从 engine core 拉取这一轮的原始输出。
            outputs = self.engine_core.get_output()

        # 2) Process EngineCoreOutputs.
        with record_function_or_nullcontext("llm_engine step: process_outputs"):
            # 若开启统计，则为这一轮新建统计收集器。
            iteration_stats = IterationStats() if self.log_stats else None
            # 将底层输出转换成用户可见的 RequestOutput。
            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=iteration_stats,
            )
            # 同步更新调度器统计信息。
            self.output_processor.update_scheduler_stats(outputs.scheduler_stats)

        # 3) Abort any reqs that finished due to stop strings.
        with record_function_or_nullcontext("llm_engine step: abort_requests"):
            # 对命中停止条件的请求执行底层中止，避免继续占资源。
            self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        # 4) Record stats
        with record_function_or_nullcontext("llm_engine step: record_stats"):
            if (
                    self.logger_manager is not None
                    and outputs.scheduler_stats is not None
                    and len(outputs.outputs) > 0
            ):
                # 记录这一轮调度、迭代和多模态缓存统计。
                self.logger_manager.record(
                    scheduler_stats=outputs.scheduler_stats,
                    iteration_stats=iteration_stats,
                    mm_cache_stats=self.renderer.stat_mm_cache(),
                )
                # 按时间间隔决定是否实际落日志。
                self.do_log_stats_with_interval()

        # 返回这一轮可直接交给调用方的请求输出列表。
        return processed_outputs.request_outputs

    # 通知 engine core 开始做 profile。
    def start_profile(self, profile_prefix: str | None = None):
        # 转发 profile 开始命令到底层 engine core。
        self.engine_core.profile(True, profile_prefix)

    # 通知 engine core 结束 profile。
    def stop_profile(self):
        # 转发 profile 结束命令到底层 engine core。
        self.engine_core.profile(False)

    # 清空多模态缓存，避免无关临时状态在 chat 生命周期内长驻。
    def reset_mm_cache(self):
        # 先清空 renderer 层的多模态缓存。
        self.renderer.clear_mm_cache()
        # 再清空 engine core 内部的多模态缓存。
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(
            self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.engine_core.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings computed with old weights are not reused.
        """
        self.engine_core.reset_encoder_cache()

    def sleep(self, level: int = 1, mode: PauseMode = "abort"):
        self.engine_core.sleep(level, mode)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    def wake_up(self, tags: list[str] | None = None):
        self.engine_core.wake_up(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def get_metrics(self) -> list[Metric]:
        assert self.log_stats, "Stat logging disabled"
        return get_metrics_snapshot()

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    def do_log_stats(self) -> None:
        """Log stats if logging is enabled."""
        if self.logger_manager:
            self.logger_manager.log()

    def do_log_stats_with_interval(self) -> None:
        """Log stats when the time interval has passed."""
        now = time.time()
        if not hasattr(self, "_last_log_time"):
            self._last_log_time = now
        if now - self._last_log_time >= envs.VLLM_LOG_STATS_INTERVAL:
            self.do_log_stats()
            self._last_log_time = now

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return self.engine_core.pin_lora(lora_id)

    def collective_rpc(
            self,
            method: str | Callable[[WorkerBase], _R],
            timeout: float | None = None,
            args: tuple = (),
            kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        return self.collective_rpc("apply_model", args=(func,))

    def __del__(self):
        dp_group = getattr(self, "dp_group", None)
        if dp_group is not None and not self.external_launcher_dp:
            stateless_destroy_torch_distributed_process_group(dp_group)
