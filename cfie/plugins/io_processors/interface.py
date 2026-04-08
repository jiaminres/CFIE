# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from typing import Generic, TypeVar

from cfie.config import CfieConfig
from cfie.inputs.data import PromptType
from cfie.outputs import PoolingRequestOutput
from cfie.pooling_params import PoolingParams
from cfie.renderers import BaseRenderer
from cfie.sampling_params import SamplingParams

IOProcessorInput = TypeVar("IOProcessorInput")
IOProcessorOutput = TypeVar("IOProcessorOutput")


class IOProcessor(ABC, Generic[IOProcessorInput, IOProcessorOutput]):
    """Engine 输入输出预处理/后处理的抽象接口。"""

    def __init__(self, cfie_config: CfieConfig, renderer: BaseRenderer):
        super().__init__()

        # 保存全局配置，供插件在预处理或后处理阶段查询模型与运行参数。
        self.cfie_config = cfie_config
        # 保存 renderer，插件可借助它做 prompt 渲染或 tokenizer 相关工作。
        self.renderer = renderer

    def parse_data(self, data: object) -> IOProcessorInput:
        # 兼容旧插件：若仍实现的是 parse_request，则转调旧接口并给出废弃告警。
        if callable(parse_request := getattr(self, "parse_request", None)):
            warnings.warn(
                "`parse_request` has been renamed to `parse_data`. "
                "Please update your IO Processor Plugin to use the new name. "
                "The old name will be removed in v0.19.",
                DeprecationWarning,
                stacklevel=2,
            )

            return parse_request(data)  # type: ignore

        # 新接口必须由具体插件自行实现。
        raise NotImplementedError

    def merge_sampling_params(
        self,
        params: SamplingParams | None = None,
    ) -> SamplingParams:
        # 兼容旧插件：若仍提供 validate_or_generate_params，则复用旧逻辑。
        if callable(
            validate_or_generate_params := getattr(
                self, "validate_or_generate_params", None
            )
        ):
            warnings.warn(
                "`validate_or_generate_params` has been split into "
                "`merge_sampling_params` and `merge_pooling_params`."
                "Please update your IO Processor Plugin to use the new methods. "
                "The old name will be removed in v0.19.",
                DeprecationWarning,
                stacklevel=2,
            )

            return validate_or_generate_params(params)  # type: ignore

        # 默认情况下，若调用方未提供 sampling 参数，则给一个默认 SamplingParams。
        return params or SamplingParams()

    def merge_pooling_params(
        self,
        params: PoolingParams | None = None,
    ) -> PoolingParams:
        # 兼容旧插件：若仍提供 validate_or_generate_params，则继续转调旧接口。
        if callable(
            validate_or_generate_params := getattr(
                self, "validate_or_generate_params", None
            )
        ):
            warnings.warn(
                "`validate_or_generate_params` has been split into "
                "`merge_sampling_params` and `merge_pooling_params`."
                "Please update your IO Processor Plugin to use the new methods. "
                "The old name will be removed in v0.19.",
                DeprecationWarning,
                stacklevel=2,
            )

            return validate_or_generate_params(params)  # type: ignore

        # 默认情况下，若未提供 pooling 参数，则生成一个 task="plugin" 的默认值。
        return params or PoolingParams(task="plugin")

    @abstractmethod
    def pre_process(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        # 由具体插件把外部输入对象转换成一个或多个 PromptType。
        raise NotImplementedError

    async def pre_process_async(
        self,
        prompt: IOProcessorInput,
        request_id: str | None = None,
        **kwargs,
    ) -> PromptType | Sequence[PromptType]:
        # 默认异步实现直接复用同步 pre_process。
        return self.pre_process(prompt, request_id, **kwargs)

    @abstractmethod
    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        # 由具体插件把模型输出转换成最终对外返回的结构。
        raise NotImplementedError

    async def post_process_async(
        self,
        model_output: AsyncGenerator[tuple[int, PoolingRequestOutput]],
        request_id: str | None = None,
        **kwargs,
    ) -> IOProcessorOutput:
        # 异步输出的返回顺序不一定与输入顺序一致，因此先按索引排序再后处理。
        sorted_output = sorted(
            [(i, item) async for i, item in model_output], key=lambda output: output[0]
        )
        # 把排序后的输出对象取出来，转交同步 post_process 处理。
        collected_output = [output[1] for output in sorted_output]
        return self.post_process(collected_output, request_id=request_id, **kwargs)
