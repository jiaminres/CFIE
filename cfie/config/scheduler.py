# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import InitVar
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from pydantic import Field, field_validator
from typing_extensions import Self

from cfie.config.utils import config
from cfie.logger import init_logger
from cfie.utils.hashing import safe_hash
from cfie.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from cfie.v1.core.sched.interface import SchedulerInterface

logger = init_logger(__name__)

RunnerType = Literal["generate", "pooling", "draft"]
SchedulerPolicy = Literal["fcfs", "priority"]


@config
class SchedulerConfig:
    """Scheduler configuration."""

    # 从 ModelConfig 透传进来的上下文窗口上限，只用于默认值补齐和参数校验。
    max_model_len: InitVar[int]
    """Maximum length of a sequence (including prompt and generated text).

    Note: This is stored in the ModelConfig, and is used only here to
    provide fallbacks and validate other attributes."""

    # 标记模型是否为 encoder-decoder；该标记会影响 chunked prefill 等能力开关。
    is_encoder_decoder: InitVar[bool]
    """True if the model is an encoder-decoder model.

    Note: This is stored in the ModelConfig, and is used only here to
    disable chunked prefill and prefix caching for encoder-decoder models.
    """

    # 调度器在未显式配置时采用的默认 token / req 预算。
    DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048
    DEFAULT_MAX_NUM_SEQS: ClassVar[int] = 128

    # 指定当前运行的是生成、池化还是 draft runner。
    runner_type: RunnerType = "generate"
    """The runner type to launch for the model."""

    # 单轮调度允许送入执行栈的最大 token 数。
    max_num_batched_tokens: int = Field(default=DEFAULT_MAX_NUM_BATCHED_TOKENS, ge=1)
    """Maximum number of tokens that can be processed in a single iteration.

    The default value here is mainly for convenience when testing.
    In real usage, this should be set in `EngineArgs.create_engine_config`.
    """

    # 单轮真正允许 scheduler 发出的 token 数；默认为 batched token 上限。
    max_num_scheduled_tokens: int | None = Field(default=None)
    """Maximum number of tokens that the scheduler may issue in a single iteration.
    
    This is usually equal to max_num_batched_tokens, but can be smaller in cases
    when the model might append tokens into the batch (such as speculative decoding).
    Defaults to max_num_batched_tokens."""

    # 单轮可并发推进的请求数上限。
    max_num_seqs: int = Field(default=DEFAULT_MAX_NUM_SEQS, ge=1)
    """Maximum number of sequences to be processed in a single iteration.

    The default value here is mainly for convenience when testing.
    In real usage, this should be set in `EngineArgs.create_engine_config`.
    """

    # chunked prefill 下，允许同时处于“部分 prefill”状态的请求数量。
    max_num_partial_prefills: int = Field(default=1, ge=1)
    """For chunked prefill, the maximum number of sequences that can be
    partially prefilled concurrently."""

    # 在“长 prompt”子集里，允许并发部分 prefill 的请求数量。
    max_long_partial_prefills: int = Field(default=1, ge=1)
    """For chunked prefill, the maximum number of prompts longer than
    long_prefill_token_threshold that will be prefilled concurrently. Setting
    this less than max_num_partial_prefills will allow shorter prompts to jump
    the queue in front of longer prompts in some cases, improving latency."""

    # 超过该阈值的 prompt 会被视为 long prefill 请求。
    long_prefill_token_threshold: int = 0
    """For chunked prefill, a request is considered long if the prompt is
    longer than this number of tokens."""

    # 是否允许把 prefill 按剩余 token budget 切成多轮调度。
    enable_chunked_prefill: bool = True
    """If True, prefill requests can be chunked based
    on the remaining `max_num_batched_tokens`.

    The default value here is mainly for convenience when testing.
    In real usage, this should be set in `EngineArgs.create_engine_config`.
    """

    # 标记模型是否为多模态模型，便于复用编码侧预算规则。
    is_multimodal_model: bool = False
    """True if the model is multimodal."""

    # 多模态 encoder 的计算预算，当前默认直接复用 batched token 预算。
    # TODO (ywang96): Make this configurable.
    max_num_encoder_input_tokens: int = Field(init=False)
    """Multimodal encoder compute budget, only used in V1.

    NOTE: This is not currently configurable. It will be overridden by
    max_num_batched_tokens in case max multimodal embedding size is larger."""

    # 多模态 encoder cache 的容量，当前默认与 batched token 预算对齐。
    # TODO (ywang96): Make this configurable.
    encoder_cache_size: int = Field(init=False)
    """Multimodal encoder cache size, only used in V1.

    NOTE: This is not currently configurable. It will be overridden by
    max_num_batched_tokens in case max multimodal embedding size is larger."""

    # 调度策略，目前支持 FCFS 和 priority 两种。
    policy: SchedulerPolicy = "fcfs"
    """The scheduling policy to use:\n
    - "fcfs" means first come first served, i.e. requests are handled in order
    of arrival.\n
    - "priority" means requests are handled based on given priority (lower
    value means earlier handling) and time of arrival deciding any ties)."""

    # chunked prefill 开启时，是否禁止把单个多模态条目拆开调度。
    disable_chunked_mm_input: bool = False
    """If set to true and chunked prefill is enabled, we do not want to
    partially schedule a multimodal item. Only used in V1
    This ensures that if a request has a mixed prompt
    (like text tokens TTTT followed by image tokens IIIIIIIIII) where only
    some image tokens can be scheduled (like TTTTIIIII, leaving IIIII),
    it will be scheduled as TTTT in one step and IIIIIIIIII in the next."""

    # 可显式指定 scheduler 类本身，或传入类的 qualname 字符串。
    scheduler_cls: str | type[object] | None = Field(default=None)
    """The scheduler class to use. "cfie.v1.core.sched.scheduler.Scheduler" is
    the default scheduler. Can be a class directly or the path to a class of
    form "mod.custom_class"."""

    # 是否关闭 hybrid KV cache manager；None 表示交给系统按环境自动判定。
    disable_hybrid_kv_cache_manager: bool | None = None
    """If set to True, KV cache manager will allocate the same size of KV cache
    for all attention layers even if there are multiple type of attention layers
    like full attention and sliding window attention.
    If set to None, the default value will be determined based on the environment
    and starting configuration.
    """

    # 是否启用 async scheduling；None 表示交给系统按运行条件自动选择。
    async_scheduling: bool | None = Field(default=None)
    """If set to False, disable async scheduling. Async scheduling helps to
    avoid gaps in GPU utilization, leading to better latency and throughput.
    """

    # 控制流式输出回传的粒度。
    stream_interval: int = Field(default=1, ge=1)
    """The interval (or buffer size) for streaming in terms of token length.
    A smaller value (1) makes streaming smoother by sending each token immediately,
    while a larger value (e.g., 10) reduces host overhead and may increase throughput
    by batching multiple tokens before sending."""

    @staticmethod
    def default_factory(**kwargs):
        # 为 InitVar 字段补入兜底值，便于外部在缺省上下文下直接构造配置。
        # 这主要服务于测试、文档和某些延迟补齐上下文的调用点。
        if "max_model_len" not in kwargs:
            kwargs["max_model_len"] = 8192
        if "is_encoder_decoder" not in kwargs:
            kwargs["is_encoder_decoder"] = False
        return SchedulerConfig(**kwargs)

    def get_scheduler_cls(self) -> type["SchedulerInterface"]:
        # ----------------- 决定最终实例化哪个 scheduler 类 -----------------
        # 未显式指定 scheduler 类时，按 async_scheduling 决定同步/异步版本。
        if self.scheduler_cls is None:
            if self.async_scheduling:
                from cfie.v1.core.sched.async_scheduler import AsyncScheduler

                return AsyncScheduler
            from cfie.v1.core.sched.scheduler import Scheduler

            return Scheduler

        # 自定义 scheduler 仍属内部接口，兼容性目前不承诺长期稳定。
        logger.warning_once(
            "Using custom scheduler class %s. This scheduler interface is "
            "not public and compatibility may not be maintained.",
            self.scheduler_cls,
        )
        if not isinstance(self.scheduler_cls, str):
            return cast(type["SchedulerInterface"], self.scheduler_cls)
        return resolve_obj_by_qualname(self.scheduler_cls)

    def compute_hash(self) -> str:
        # 计算影响执行图结构的配置哈希，用于缓存和编译图区分。
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []

        # `max_num_batched_tokens` 会影响 LoRA 静态缓冲和编译图索引宽度，
        # 因此必须进入结构哈希。
        factors.append(self.max_num_batched_tokens)

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @field_validator("scheduler_cls", "async_scheduling", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        # 延迟初始化场景允许这两个字段先保持 None，等后续上下文补齐。
        """Skip validation if the value is `None` when initialisation is delayed."""
        return None if value is None else handler(value)

    def __post_init__(self, max_model_len: int, is_encoder_decoder: bool) -> None:
        # ----------------- encoder-decoder 兼容性收口 -----------------
        if is_encoder_decoder:
            # Chunked prefill should be disabled for encoder-decoder models.
            self.disable_chunked_mm_input = True
            self.enable_chunked_prefill = False
            self.long_prefill_token_threshold = 0
            logger.info(
                "Encoder-decoder models do not support chunked prefill nor"
                " prefix caching; disabling both."
            )

        # 多模态 encoder 预算默认与主调度 token 预算保持一致。
        self.max_num_encoder_input_tokens = self.max_num_batched_tokens
        self.encoder_cache_size = self.max_num_batched_tokens

        # ----------------- 运行期日志与并发阈值补齐 -----------------
        if self.enable_chunked_prefill:
            logger.info(
                "Chunked prefill is enabled with max_num_batched_tokens=%d.",
                self.max_num_batched_tokens,
            )

        if self.max_num_partial_prefills > 1:
            # 若允许并发部分 prefill，但阈值未显式给出，则按窗口长度的 4% 估算。
            if self.long_prefill_token_threshold == 0:
                self.long_prefill_token_threshold = int(max_model_len * 0.04)

            logger.info(
                "Concurrent partial prefills enabled with "
                "max_num_partial_prefills=%d, max_long_partial_prefills=%d, "
                "long_prefill_token_threshold=%d",
                self.max_num_partial_prefills,
                self.max_long_partial_prefills,
                self.long_prefill_token_threshold,
            )

        # ----------------- 最终约束校验 -----------------
        self.verify_max_model_len(max_model_len)

    def verify_max_model_len(self, max_model_len: int) -> Self:
        # 关闭 chunked prefill 时，batched token 上限必须能容纳完整上下文。
        if (
            self.max_num_batched_tokens < max_model_len
            and not self.enable_chunked_prefill
        ):
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len."
            )

        # 基本一致性：单轮 token 预算不能小于单轮可并发请求数。
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs})."
            )

        # 超大 batched token 预算虽非硬错误，但通常意味着配置异常。
        if self.max_num_batched_tokens > self.max_num_seqs * max_model_len:
            logger.warning(
                "max_num_batched_tokens (%d) exceeds max_num_seqs "
                "* max_model_len (%d). This may lead to unexpected behavior.",
                self.max_num_batched_tokens,
                self.max_num_seqs * max_model_len,
            )

        # 并发 partial prefill 相关参数只在 chunked prefill 打开时有效。
        if self.max_num_partial_prefills > 1:
            if not self.enable_chunked_prefill:
                raise ValueError(
                    "Chunked prefill must be enabled to set "
                    "max_num_partial_prefills > 1."
                )

            # “长 prompt”阈值不允许超过上下文窗口上限。
            if self.long_prefill_token_threshold > max_model_len:
                raise ValueError(
                    "long_prefill_token_threshold "
                    f"({self.long_prefill_token_threshold}) cannot be greater "
                    f"than the max_model_len ({max_model_len})."
                )

        # 长 prompt 并发数不能超过整体 partial prefill 并发数。
        if self.max_long_partial_prefills > self.max_num_partial_prefills:
            raise ValueError(
                f"{self.max_long_partial_prefills=} must be less than or equal to "
                f"{self.max_num_partial_prefills=}."
            )

        return self
