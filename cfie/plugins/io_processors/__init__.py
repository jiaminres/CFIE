# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

from cfie.config import CfieConfig
from cfie.plugins import IO_PROCESSOR_PLUGINS_GROUP, load_plugins_by_group
from cfie.plugins.io_processors.interface import IOProcessor
from cfie.renderers import BaseRenderer
from cfie.utils.import_utils import resolve_obj_by_qualname

logger = logging.getLogger(__name__)


def get_io_processor(
    # 全局引擎配置，供插件构造时使用。
    cfie_config: CfieConfig,
    # 当前 renderer，插件可能需要借助它做 prompt 渲染或 tokenizer 操作。
    renderer: BaseRenderer,
    # 初始化阶段显式指定的插件名，优先级高于模型配置。
    plugin_from_init: str | None = None,
) -> IOProcessor | None:
    # IO processor 通过 `cfie.io_processor_plugins` 这组 entry points 作为插件加载。
    # 每个插件入口返回一个字符串，表示最终要实例化的处理器类全限定名。

    # 若调用方在初始化阶段显式指定了插件，则直接采用。
    if plugin_from_init:
        model_plugin = plugin_from_init
    else:
        # 否则尝试从模型的 hf_config 中读取模型声明的 io_processor_plugin。
        hf_config = cfie_config.model_config.hf_config.to_dict()
        config_plugin = hf_config.get("io_processor_plugin")
        model_plugin = config_plugin

    # 若模型与初始化参数都未要求插件，则返回 None。
    if model_plugin is None:
        logger.debug("No IOProcessor plugins requested by the model")
        return None

    # 打印即将加载的插件名。
    logger.debug("IOProcessor plugin to be loaded %s", model_plugin)

    # 发现并加载该插件组下所有已安装的 entry point。
    multimodal_data_processor_plugins = load_plugins_by_group(
        IO_PROCESSOR_PLUGINS_GROUP
    )

    # 保存“插件名 -> 可实例化处理器类全限定名”的映射。
    loadable_plugins = {}
    # 逐个调用插件入口函数，拿到真正的处理器类名。
    for name, func in multimodal_data_processor_plugins.items():
        try:
            # entry point 加载出来的对象必须是可调用的。
            assert callable(func)
            # 调用插件入口函数，获取处理器类的全限定名。
            processor_cls_qualname = func()
            # 只有返回了类名的插件才视为可加载。
            if processor_cls_qualname is not None:
                loadable_plugins[name] = processor_cls_qualname
        except Exception:
            # 单个插件失败时打印告警，但不阻断其他插件的发现。
            logger.warning("Failed to load plugin %s.", name, exc_info=True)

    # 统计当前真正可用的 IOProcessor 插件数量。
    num_available_plugins = len(loadable_plugins.keys())
    # 如果模型要求插件，但一个可用插件都没有，直接报错。
    if num_available_plugins == 0:
        raise ValueError(
            f"No IOProcessor plugins installed but one is required ({model_plugin})."
        )

    # 如果模型要求的插件不在可用列表里，也直接报错并给出候选项。
    if model_plugin not in loadable_plugins:
        raise ValueError(
            f"The model requires the '{model_plugin}' IO Processor plugin "
            "but it is not installed. "
            f"Available plugins: {list(loadable_plugins.keys())}"
        )

    # 解析目标插件类。
    activated_plugin_cls = resolve_obj_by_qualname(loadable_plugins[model_plugin])

    # 用 CfieConfig 与当前 renderer 实例化最终的 IOProcessor。
    return activated_plugin_cls(cfie_config, renderer)
