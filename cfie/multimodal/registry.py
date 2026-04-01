# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading  # 线程相关，用于 timing 统计时加锁
from collections import defaultdict  # 默认字典，用于按 request_id 保存 timing 上下文
from collections.abc import Mapping  # 抽象映射类型，用于类型标注
from dataclasses import dataclass  # dataclass 装饰器，用于简化数据类定义
from multiprocessing.synchronize import Lock as LockType  # 多进程锁类型，用于 shm worker cache
from typing import TYPE_CHECKING, Generic, Literal, Protocol, TypeVar, cast  # 类型系统工具

from cfie.logger import init_logger  # vLLM 日志工具
from cfie.tokenizers import TokenizerLike, cached_tokenizer_from_config  # tokenizer 类型与按配置获取 tokenizer

from .cache import (
    BaseMultiModalProcessorCache,  # 多模态 processor cache 抽象基类
    BaseMultiModalReceiverCache,  # 多模态 receiver cache 抽象基类
    MultiModalProcessorOnlyCache,  # 仅 processor 侧缓存
    MultiModalProcessorSenderCache,  # processor 侧 sender cache（通常配合 LRU）
    MultiModalReceiverCache,  # engine 侧 receiver cache
    ShmObjectStoreReceiverCache,  # 基于共享内存对象存储的 receiver cache
    ShmObjectStoreSenderCache,  # 基于共享内存对象存储的 sender cache
)
from .inputs import MultiModalInputs  # 多模态输入结果类型
from .processing import (
    BaseDummyInputsBuilder,  # 伪输入构造器基类，用于 profiling 等场景
    BaseMultiModalProcessor,  # 多模态处理器基类
    BaseProcessingInfo,  # 多模态处理信息基类
    InputProcessingContext,  # 输入处理上下文：模型配置 + tokenizer
    TimingContext,  # 多模态处理耗时统计上下文
)

if TYPE_CHECKING:
    # 仅用于类型检查，运行时不会真的 import，避免循环依赖和额外开销
    from cfie.config import ModelConfig, ObservabilityConfig, CfieConfig
    from cfie.model_executor.models.interfaces import SupportsMultiModal

logger = init_logger(__name__)  # 初始化当前模块的 logger

# 泛型类型变量：
# N：绑定到“支持多模态的模型类类型”
N = TypeVar("N", bound=type["SupportsMultiModal"])
# _I：绑定到 BaseProcessingInfo 的具体子类
_I = TypeVar("_I", bound=BaseProcessingInfo)
# _I_co：协变版本的 BaseProcessingInfo 子类
_I_co = TypeVar("_I_co", bound=BaseProcessingInfo, covariant=True)


class ProcessingInfoFactory(Protocol[_I_co]):
    """
    一个“工厂协议”：
    给定 InputProcessingContext，构造一个 ProcessingInfo 实例。
    """

    def __call__(
            self,
            ctx: InputProcessingContext,
    ) -> _I_co: ...


class DummyInputsBuilderFactory(Protocol[_I]):  # type: ignore[misc]
    """
    一个“工厂协议”：
    给定 ProcessingInfo，构造一个 DummyInputsBuilder 实例。
    """

    def __call__(self, info: _I) -> BaseDummyInputsBuilder[_I]: ...


class MultiModalProcessorFactory(Protocol[_I]):  # type: ignore[misc]
    """
    一个“工厂协议”：
    给定 ProcessingInfo + DummyInputsBuilder (+ 可选 cache)，
    构造一个 MultiModalProcessor 实例。
    """

    def __call__(
            self,
            info: _I,
            dummy_inputs: BaseDummyInputsBuilder[_I],
            *,
            cache: BaseMultiModalProcessorCache | None = None,
    ) -> BaseMultiModalProcessor[_I]: ...


@dataclass(frozen=True)
class _ProcessorFactories(Generic[_I]):
    """绑定某个模型类的一组多模态处理工厂。"""

    # 保存三种工厂：
    # 1. info 工厂
    # 2. processor 工厂
    # 3. dummy_inputs 工厂
    # 根据 `InputProcessingContext` 构造该模型专属的 `ProcessingInfo`。
    info: ProcessingInfoFactory[_I]
    # 根据 `ProcessingInfo` + `DummyInputsBuilder` 真正构造 processor。
    processor: MultiModalProcessorFactory[_I]
    # 根据 `ProcessingInfo` 构造 profiling / 占位计算用的伪输入构造器。
    dummy_inputs: DummyInputsBuilderFactory[_I]

    def build_processor(
            self,
            ctx: InputProcessingContext,
            *,
            cache: BaseMultiModalProcessorCache | None = None,
    ):
        # 先根据上下文构造 info
        info = self.info(ctx)

        # 再根据 info 构造 dummy input builder
        dummy_inputs_builder = self.dummy_inputs(info)

        # 最后用 info + dummy builder + cache 构造真正的 processor
        return self.processor(info, dummy_inputs_builder, cache=cache)


class MultiModalRegistry:
    """
    多模态注册表：
    根据模型类型，分发对应的数据处理逻辑。
    """

    def supports_multimodal_inputs(self, model_config: "ModelConfig") -> bool:
        """
        判断某个模型是否真正需要启用多模态输入基础设施。

        返回 True 的情况：
        - 模型本身是多模态模型
        - 且存在至少一种支持的模态，其 per-prompt 限制不为 0
        - 或者虽然模态限制全为 0，但 enable_mm_embeds=True，
          仍需要多模态基础设施去处理预计算 embedding
        """
        # 若模型本身不是多模态模型，直接返回 False
        if not model_config.is_multimodal_model:
            return False

        # 取出多模态配置
        mm_config = model_config.get_multimodal_config()

        # 创建 processing info，用于获取该模型支持哪些模态
        info = self._create_processing_info(model_config, tokenizer=None)

        # 如果该模型所有支持的模态，其 limit_per_prompt 都被设置为 0
        if all(
                mm_config.get_limit_per_prompt(modality) == 0
                for modality in info.supported_mm_limits
        ):
            # 即便 encoder 不跑，如果启用了预计算多模态 embedding，
            # 仍需要保留多模态基础设施
            if mm_config.enable_mm_embeds:
                return True

            logger.info_once(
                "All limits of multimodal modalities supported by the model "
                "are set to 0, running in text-only mode."
            )
            return False

        # 其他情况说明确实支持并启用多模态
        return True

    def register_processor(
            self,
            processor: MultiModalProcessorFactory[_I],
            *,
            info: ProcessingInfoFactory[_I],
            dummy_inputs: DummyInputsBuilderFactory[_I],
    ):
        """
        给某个模型类注册多模态处理器。

        注意这里不是直接传“实例”，而是传“工厂函数”，
        因为 processor 是延迟构造的。
        """

        def wrapper(model_cls: N) -> N:
            # 如果该模型类之前已经注册过 processor，则给出警告
            if "_processor_factory" in model_cls.__dict__:
                logger.warning(
                    "Model class %s already has a multi-modal processor "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls,
                    self,
                )

            # 把三种工厂封装成 _ProcessorFactories，
            # 挂到模型类的 _processor_factory 属性上
            model_cls._processor_factory = _ProcessorFactories(
                info=info,
                dummy_inputs=dummy_inputs,
                processor=processor,
            )

            return model_cls

        # 返回装饰器
        return wrapper

    def _get_model_cls(self, model_config: "ModelConfig") -> "SupportsMultiModal":
        # 避免顶层 import 导致循环依赖，这里延迟导入
        from cfie.model_executor.model_loader import get_model_architecture

        # 根据 model_config 获取模型架构类
        model_cls, _ = get_model_architecture(model_config)

        # 要求该类必须已经注册了 _processor_factory
        assert hasattr(model_cls, "_processor_factory")

        # 强转为 SupportsMultiModal 类型
        return cast("SupportsMultiModal", model_cls)

    def _create_processing_ctx(
            self,
            model_config: "ModelConfig",
            tokenizer: TokenizerLike | None = None,
    ) -> InputProcessingContext:
        # 如果未显式传 tokenizer，则根据配置缓存获取一个 tokenizer
        if tokenizer is None:
            tokenizer = cached_tokenizer_from_config(model_config)

        # 构造输入处理上下文：模型配置 + tokenizer
        return InputProcessingContext(model_config, tokenizer)

    def _create_processing_info(
            self,
            model_config: "ModelConfig",
            tokenizer: TokenizerLike | None = None,
    ) -> BaseProcessingInfo:
        # 找到模型类
        model_cls = self._get_model_cls(model_config)
        # 取出注册在模型类上的 factories
        factories = model_cls._processor_factory
        # 构造处理上下文
        ctx = self._create_processing_ctx(model_config, tokenizer)
        # 用 info factory 构造 processing info
        return factories.info(ctx)

    def create_processor(
            self,
            model_config: "ModelConfig",
            *,
            tokenizer: TokenizerLike | None = None,
            cache: BaseMultiModalProcessorCache | None = None,
    ) -> BaseMultiModalProcessor[BaseProcessingInfo]:
        """
        为某个具体模型 + tokenizer 创建一个多模态处理器。
        """
        # 如果不是多模态模型，不允许创建 processor
        if not model_config.is_multimodal_model:
            raise ValueError(f"{model_config.model} is not a multimodal model")

        # 找到模型类
        model_cls = self._get_model_cls(model_config)

        # 取出工厂集合
        factories = model_cls._processor_factory

        # 构造处理上下文
        ctx = self._create_processing_ctx(model_config, tokenizer)

        # 用工厂集合真正构造 processor
        return factories.build_processor(ctx, cache=cache)

    def get_dummy_mm_inputs(
            self,
            model_config: "ModelConfig",
            mm_counts: Mapping[str, int],
            *,
            cache: BaseMultiModalProcessorCache | None = None,
            processor: BaseMultiModalProcessor | None = None,
    ) -> MultiModalInputs:
        """
        构造 profiling 用的“伪多模态输入”。

        常用于内存使用分析、profile 跑通等场景。
        """
        # 以模型最大序列长度作为 dummy 输入长度
        seq_len = model_config.max_model_len

        # 如果没有传 processor，则现场创建一个
        if processor is None:
            processor = self.create_processor(model_config, cache=cache)

        # 取出多模态配置
        mm_config = model_config.get_multimodal_config()

        # 让 dummy input builder 构造“处理器输入前”的伪输入
        processor_inputs = processor.dummy_inputs.get_dummy_processor_inputs(
            seq_len=seq_len,  # 目标总序列长度
            mm_counts=mm_counts,  # 各模态数量，例如 {"image": 1}
            mm_options=mm_config.limit_per_prompt,  # 各模态上限配置
        )

        # 把伪输入真正过一遍 processor，得到最终多模态输入
        mm_inputs = processor.apply(
            processor_inputs,
            timing_ctx=TimingContext(enabled=False),  # profiling 场景一般不统计 timing
        )

        # 取出 prompt token ids
        prompt_token_ids = mm_inputs["prompt_token_ids"]
        total_len = len(prompt_token_ids)

        # 如果长度不足 max_model_len，则补 0 到固定长度
        if total_len < seq_len:
            prompt_token_ids.extend([0] * (seq_len - total_len))

        return mm_inputs

    def _get_cache_type(
            self,
            cfie_config: "CfieConfig",
    ) -> Literal[None, "processor_only", "lru", "shm"]:
        """
        根据配置判断当前多模态缓存应采用哪种类型：
        - None：不启用缓存
        - processor_only：仅 processor 内部缓存
        - lru：使用 sender/receiver 方式的 LRU 缓存
        - shm：使用共享内存对象存储缓存
        """
        model_config = cfie_config.model_config

        # 如果当前模型根本不支持/不启用多模态，则不需要 cache
        if not self.supports_multimodal_inputs(model_config):
            return None

        # 如果 cache 大小 <= 0，表示显式禁用 cache
        mm_config = model_config.get_multimodal_config()
        if mm_config.mm_processor_cache_gb <= 0:
            return None

        # 判断是否支持 IPC（跨进程）缓存
        parallel_config = cfie_config.parallel_config
        is_ipc_supported = parallel_config._api_process_count == 1 and (
                parallel_config.data_parallel_size == 1
                or parallel_config.data_parallel_external_lb
        )

        # 如果不支持 IPC，则退化成仅 processor 侧缓存
        if not is_ipc_supported:
            return "processor_only"

        # 否则按配置返回 cache 类型（通常是 "lru" 或 "shm"）
        mm_config = model_config.get_multimodal_config()
        return mm_config.mm_processor_cache_type

    def processor_cache_from_config(
            self,
            cfie_config: "CfieConfig",
    ) -> BaseMultiModalProcessorCache | None:
        """
        按当前配置创建“前端/processor 侧”多模态缓存。

        这个函数通常在 renderer 初始化多模态 processor 时调用，
        用来决定 processor 产出的多模态中间结果应如何缓存：
        - 不缓存：返回 None
        - 仅当前进程内缓存：返回 `MultiModalProcessorOnlyCache`
        - IPC + LRU 元数据缓存：返回 `MultiModalProcessorSenderCache`
        - 共享内存对象缓存：返回 `ShmObjectStoreSenderCache`
        """
        # 先统一根据多模态开关、cache 大小、并行/IPC 能力判断 cache 类型。
        cache_type = self._get_cache_type(cfie_config)

        # None 表示当前模型不走多模态，或显式关闭了 mm processor cache。
        if cache_type is None:
            return None
        # `processor_only` 用于无法做跨进程 IPC 的场景；
        # 此时只在当前 processor 所在进程内保存完整多模态结果。
        elif cache_type == "processor_only":
            return MultiModalProcessorOnlyCache(cfie_config.model_config)
        # `lru` 表示 sender 侧只保留元数据和 LRU 驱逐信息，
        # 真正的数据由对端 receiver cache 协同复用，尽量减少 P0 内存占用。
        elif cache_type == "lru":  # 默认
            return MultiModalProcessorSenderCache(cfie_config.model_config)
        # `shm` 表示把多模态结果直接放进共享内存对象存储，
        # 后续进程通过共享内存地址读取，避免重复拷贝大对象。
        elif cache_type == "shm":
            return ShmObjectStoreSenderCache(cfie_config)
        else:
            raise ValueError(f"Unknown cache type: {cache_type!r}")

    def processor_only_cache_from_config(
            self,
            cfie_config: "CfieConfig",
    ) -> MultiModalProcessorOnlyCache | None:
        """按配置返回一个仅 processor 内部使用的 cache（如果启用）。"""
        cache_type = self._get_cache_type(cfie_config)
        if cache_type is None:
            return None

        return MultiModalProcessorOnlyCache(cfie_config.model_config)

    def engine_receiver_cache_from_config(
            self,
            cfie_config: "CfieConfig",
    ) -> BaseMultiModalReceiverCache | None:
        """按配置返回 engine 进程侧使用的 receiver cache。"""
        cache_type = self._get_cache_type(cfie_config)
        if cache_type in (None, "processor_only", "shm"):
            # shm 模式下 engine 侧不直接使用 receiver cache
            return None
        elif cache_type == "lru":
            return MultiModalReceiverCache(cfie_config.model_config)
        else:
            raise ValueError(f"Unknown cache type: {cache_type!r}")

    def worker_receiver_cache_from_config(
            self,
            cfie_config: "CfieConfig",
            shared_worker_lock: LockType,
    ) -> BaseMultiModalReceiverCache | None:
        """按配置返回 worker 进程侧使用的 receiver cache。"""
        cache_type = self._get_cache_type(cfie_config)
        if cache_type in (None, "processor_only", "lru"):
            return None
        elif cache_type == "shm":
            return ShmObjectStoreReceiverCache(cfie_config, shared_worker_lock)
        else:
            raise ValueError(f"Unknown cache type: {cache_type!r}")


class MultiModalTimingRegistry:
    """
    多模态处理耗时统计注册表。

    用 request_id 维度保存 TimingContext，
    供 observability / profiling / stats 上报使用。
    """

    def __init__(self, observability_config: "ObservabilityConfig | None") -> None:
        super().__init__()

        # 只有显式开启 mm processor 统计时才启用
        if observability_config and observability_config.enable_mm_processor_stats:
            self._lock = threading.Lock()  # 线程锁，保护共享字典
            self._ctx_by_request_id = defaultdict[str, TimingContext](TimingContext)
            self._enabled = True
        else:
            self._enabled = False

    def get(self, request_id: str) -> TimingContext:
        """
        获取某个 request_id 对应的 TimingContext。
        若未启用统计，则返回一个 disabled 的 TimingContext。
        """
        if not self._enabled:
            return TimingContext(enabled=False)

        with self._lock:
            return self._ctx_by_request_id[request_id]

    def stat(self) -> dict[str, dict[str, float]]:
        """
        取出当前所有 request 的 timing 统计结果，并清空内部缓存。
        """
        if not self._enabled:
            return {}

        with self._lock:
            stats = {
                req_id: ctx.get_stats_dict()
                for req_id, ctx in self._ctx_by_request_id.items()
            }
            self._ctx_by_request_id.clear()  # 取完即清空
            return stats
