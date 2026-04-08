# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections.abc import Callable
from typing import Any

import cfie.envs as envs

# 当前模块的日志器
logger = logging.getLogger(__name__)

# 默认插件分组：
# 所有进程都会加载（包括 process0、engine core 进程、worker 进程）
DEFAULT_PLUGINS_GROUP = "cfie.general_plugins"

# IO 处理插件分组：
# 只会在 process0 中加载
IO_PROCESSOR_PLUGINS_GROUP = "cfie.io_processor_plugins"

# 平台插件分组：
# 当调用 `cfie.platforms.current_platform` 且当前平台值尚未初始化时，
# 所有进程都会尝试加载这一组插件
PLATFORM_PLUGINS_GROUP = "cfie.platform_plugins"

# 统计日志插件分组：
# 仅在 process0 中、且以 async 模式启动服务时加载
STAT_LOGGER_PLUGINS_GROUP = "cfie.stat_logger_plugins"

# 确保同一个进程只加载一次插件
plugins_loaded = False


def load_plugins_by_group(group: str) -> dict[str, Callable[[], Any]]:
    """加载指定 entry point 分组下注册的插件。"""
    from importlib.metadata import entry_points

    # 从环境变量中读取允许加载的插件名单
    # 若为 None，表示不做限制，允许加载该分组下的所有插件
    allowed_plugins = envs.VLLM_PLUGINS

    # 发现当前 group 下所有注册的 entry points
    discovered_plugins = entry_points(group=group)

    # 如果一个插件都没找到，则直接返回空字典
    if len(discovered_plugins) == 0:
        logger.debug("分组 %s 下未找到插件。", group)
        return {}

    # 判断当前是不是“默认插件组”
    is_default_group = group == DEFAULT_PLUGINS_GROUP

    # 默认插件组用 DEBUG 级别打印
    # 非默认插件组用 INFO 级别打印
    log_level = logger.debug if is_default_group else logger.info

    # 打印当前分组下可用插件
    log_level("分组 %s 下可用的插件如下：", group)
    for plugin in discovered_plugins:
        log_level("- %s -> %s", plugin.name, plugin.value)

    # 如果没有配置 allowed_plugins，说明当前分组下所有插件都允许加载
    if allowed_plugins is None:
        log_level(
            "该分组下的所有插件都会被加载。"
            "如需控制加载哪些插件，请设置 `VLLM_PLUGINS`。"
        )

    # 保存最终成功加载的插件
    # key: 插件名
    # value: 插件对应的可调用对象（通常是一个无参函数）
    plugins = dict[str, Callable[[], Any]]()

    # 逐个尝试加载发现到的插件
    for plugin in discovered_plugins:
        # 若未设置 allowed_plugins，则全部允许
        # 若设置了，则只加载名字在 allowed_plugins 里的插件
        if allowed_plugins is None or plugin.name in allowed_plugins:
            if allowed_plugins is not None:
                log_level("正在加载插件 %s", plugin.name)

            try:
                # 真正导入并加载插件对象
                func = plugin.load()

                # 保存到结果字典
                plugins[plugin.name] = func
            except Exception:
                # 某个插件加载失败时，打印异常但不中断整体流程
                logger.exception("加载插件 %s 失败", plugin.name)

    return plugins


def load_general_plugins():
    """加载默认通用插件。

    警告：
    插件可能会在多个不同进程中被重复加载。
    因此插件本身应设计成“可重复加载而不会出问题”的形式。
    """
    global plugins_loaded

    # 若当前进程已经加载过插件，则直接返回，避免重复加载
    if plugins_loaded:
        return

    # 标记当前进程已经完成插件加载
    plugins_loaded = True

    # 加载默认插件组下的所有插件
    plugins = load_plugins_by_group(group=DEFAULT_PLUGINS_GROUP)

    # 对于 general plugins，只需要执行加载出来的函数即可
    for func in plugins.values():
        func()
