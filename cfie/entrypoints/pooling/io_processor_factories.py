# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from cfie.config import ModelConfig
from cfie.entrypoints.chat_utils import ChatTemplateConfig
from cfie.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from cfie.renderers import BaseRenderer
from cfie.tasks import SupportedTask


def init_pooling_io_processors(
    supported_tasks: tuple[SupportedTask, ...],
    model_config: ModelConfig,
    renderer: BaseRenderer,
    chat_template_config: ChatTemplateConfig,
) -> dict[str, PoolingIOProcessor]:
    processors: list[tuple[str, type[PoolingIOProcessor]]] = []
    if "classify" in supported_tasks:
        from cfie.entrypoints.pooling.classify.io_processor import ClassifyIOProcessor

        processors.append(("classify", ClassifyIOProcessor))
    if "embed" in supported_tasks:
        from cfie.entrypoints.pooling.embed.io_processor import EmbedIOProcessor

        processors.append(("classify", EmbedIOProcessor))

    return {
        task: processor_cls(
            model_config=model_config,
            renderer=renderer,
            chat_template_config=chat_template_config,
        )
        for task, processor_cls in processors
    }
