# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 改编自 sglang 的 offloader 相关实现
"""平台检测与当前平台对象解析逻辑。"""  # :contentReference[oaicite:0]{index=0}

import logging
import traceback
from itertools import chain
from typing import TYPE_CHECKING

from cfie import envs
from cfie.plugins import PLATFORM_PLUGINS_GROUP, load_plugins_by_group
from cfie.utils.import_utils import resolve_obj_by_qualname
from cfie.utils.torch_utils import supports_xccl

from .interface import CpuArchEnum, Platform, PlatformEnum

# 模块级日志器
logger = logging.getLogger(__name__)


def cfie_version_matches_substr(substr: str) -> bool:
    """
    检查 cfie 包版本字符串里是否包含指定子串。
    常用于判断当前安装的是不是 cpu 版本等。
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        # 读取已安装 cfie 包的版本号字符串
        cfie_version = version("cfie")
    except PackageNotFoundError:
        # 支持直接从源码树运行而未安装 wheel/sdist 的场景。
        from cfie import __version__ as cfie_version

        logger.debug(
            "The CFIE package metadata was not found; falling back to "
            "cfie.__version__ for platform detection."
        )

    # 只做简单子串包含判断
    return substr in cfie_version


def tpu_platform_plugin() -> str | None:
    """
    检测当前环境是否可用 TPU。
    若可用，则返回 TPU 平台类的“完整限定名”字符串；
    否则返回 None。
    """
    logger.debug("Checking if TPU platform is available.")

    # 先检查是否通过 Pathways TPU proxy 使用 TPU
    if envs.VLLM_TPU_USING_PATHWAYS:
        logger.debug("Confirmed TPU platform is available via Pathways proxy.")
        return "tpu_inference.platforms.tpu_platform.TpuPlatform"

    # 再检查本机是否安装了 libtpu
    try:
        # 一般认为：装了 libtpu，基本就意味着机器有 TPU 能力
        import libtpu  # noqa: F401

        logger.debug("Confirmed TPU platform is available.")
        return "cfie.platforms.tpu.TpuPlatform"
    except Exception as e:
        logger.debug("TPU platform is not available because: %s", str(e))
        return None


def cuda_platform_plugin() -> str | None:
    """
    检测当前环境是否可用 CUDA。
    若可用，则返回 CUDA 平台类的完整限定名；
    否则返回 None。
    """
    is_cuda = False
    logger.debug("Checking if CUDA platform is available.")
    try:
        # 动态导入 pynvml，用于检测 NVIDIA GPU
        from cfie.utils.import_utils import import_pynvml

        pynvml = import_pynvml()
        pynvml.nvmlInit()
        try:
            # 这里除了检查 GPU 数量，还要额外检查：
            # 当前 cfie 是否是 cpu build。
            # 否则在“机器上有 GPU 但 cfie 是 CPU 版”的情况下会误判为 CUDA 平台。
            is_cuda = (
                pynvml.nvmlDeviceGetCount() > 0
                and not cfie_version_matches_substr("cpu")
            )

            # 没 GPU
            if pynvml.nvmlDeviceGetCount() <= 0:
                logger.debug("CUDA platform is not available because no GPU is found.")

            # 是 cpu build
            if cfie_version_matches_substr("cpu"):
                logger.debug(
                    "CUDA platform is not available because vLLM is built with CPU."
                )

            if is_cuda:
                logger.debug("Confirmed CUDA platform is available.")
        finally:
            # 结束 NVML 会话
            pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug("Exception happens when checking CUDA platform: %s", str(e))

        # 如果异常不是 NVML 相关，就直接继续抛出
        if "nvml" not in e.__class__.__name__.lower():
            raise e

        # Jetson 平台上 CUDA 可用，但 NVML 不一定可用
        import os

        def cuda_is_jetson() -> bool:
            # 通过 Jetson 常见系统文件判断
            return os.path.isfile("/etc/nv_tegra_release") or os.path.exists(
                "/sys/class/tegra-firmware"
            )

        if cuda_is_jetson():
            logger.debug("Confirmed CUDA platform is available on Jetson.")
            is_cuda = True
        else:
            logger.debug("CUDA platform is not available because: %s", str(e))

    return "cfie.platforms.cuda.CudaPlatform" if is_cuda else None


def rocm_platform_plugin() -> str | None:
    """
    检测当前环境是否可用 ROCm。
    若可用，则返回 ROCm 平台类完整限定名；
    否则返回 None。
    """
    is_rocm = False
    logger.debug("Checking if ROCm platform is available.")
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        try:
            # 若能检测到 AMD 处理器句柄，则认为 ROCm 可用
            if len(amdsmi.amdsmi_get_processor_handles()) > 0:
                is_rocm = True
                logger.debug("Confirmed ROCm platform is available.")
            else:
                logger.debug("ROCm platform is not available because no GPU is found.")
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as e:
        logger.debug("ROCm platform is not available because: %s", str(e))

    return "cfie.platforms.rocm.RocmPlatform" if is_rocm else None


def xpu_platform_plugin() -> str | None:
    """
    检测当前环境是否可用 XPU（如 Intel GPU）。
    若可用，则返回 XPU 平台类完整限定名；
    否则返回 None。
    """
    is_xpu = False
    logger.debug("Checking if XPU platform is available.")
    try:
        import torch

        # 如果支持 xccl，则给 XPUPlatform 设置分布式后端
        if supports_xccl():
            dist_backend = "xccl"
            from cfie.platforms.xpu import XPUPlatform

            XPUPlatform.dist_backend = dist_backend
            logger.debug("Confirmed %s backend is available.", XPUPlatform.dist_backend)

        # 检查 torch.xpu 能否使用
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            is_xpu = True
            logger.debug("Confirmed XPU platform is available.")
    except Exception as e:
        logger.debug("XPU platform is not available because: %s", str(e))

    return "cfie.platforms.xpu.XPUPlatform" if is_xpu else None


def cpu_platform_plugin() -> str | None:
    """
    检测当前环境是否应使用 CPU 平台。
    若可用，则返回 CPU 平台类完整限定名；
    否则返回 None。
    """
    is_cpu = False
    logger.debug("Checking if CPU platform is available.")
    try:
        # 若版本字符串里包含 "cpu"，则认为当前是 cpu build
        is_cpu = cfie_version_matches_substr("cpu")
        if is_cpu:
            logger.debug(
                "Confirmed CPU platform is available because vLLM is built with CPU."
            )

        # 如果不是 cpu build，再检查是否是 macOS
        # 在 macOS 上通常也走 CPU 平台
        if not is_cpu:
            import sys

            is_cpu = sys.platform.startswith("darwin")
            if is_cpu:
                logger.debug(
                    "Confirmed CPU platform is available because the machine is MacOS."
                )

    except Exception as e:
        logger.debug("CPU platform is not available because: %s", str(e))

    return "cfie.platforms.cpu.CpuPlatform" if is_cpu else None


# 内建平台插件表：
# key 是插件名字
# value 是检测函数
builtin_platform_plugins = {
    "tpu": tpu_platform_plugin,
    "cuda": cuda_platform_plugin,
    "rocm": rocm_platform_plugin,
    "xpu": xpu_platform_plugin,
    "cpu": cpu_platform_plugin,
}


def resolve_current_platform_cls_qualname() -> str:
    """
    解析“当前平台类”的完整限定名字符串。

    解析顺序：
    1. 先加载 out-of-tree 插件
    2. 再遍历内建插件和外部插件，看看哪些被激活
    3. 最终要求只能有一个平台插件激活
    """
    # 加载用户/外部插件, 默认没有外部platform插件
    platform_plugins = load_plugins_by_group(PLATFORM_PLUGINS_GROUP)

    # 记录成功激活的平台插件名字
    activated_plugins = []

    # 把“内建插件”和“外部插件”串起来统一检查
    for name, func in chain(builtin_platform_plugins.items(), platform_plugins.items()):
        try:
            assert callable(func)
            platform_cls_qualname = func()

            # 返回非 None 说明插件认为当前平台可用
            if platform_cls_qualname is not None:
                activated_plugins.append(name)
        except Exception:
            # 这里直接吞掉异常，表示该插件未激活
            pass

    # 过滤出“被激活的内建插件”
    activated_builtin_plugins = list(
        set(activated_plugins) & set(builtin_platform_plugins.keys())
    )

    # 过滤出“被激活的外部插件”
    activated_oot_plugins = list(set(activated_plugins) & set(platform_plugins.keys()))

    # 外部插件不能同时激活多个
    if len(activated_oot_plugins) >= 2:
        raise RuntimeError(
            "Only one platform plugin can be activated, but got: "
            f"{activated_oot_plugins}"
        )

    # 若有且仅有一个外部插件激活，则优先使用它
    elif len(activated_oot_plugins) == 1:
        platform_cls_qualname = platform_plugins[activated_oot_plugins[0]]()
        logger.info("Platform plugin %s is activated", activated_oot_plugins[0])

    # 内建插件也不能同时激活多个
    elif len(activated_builtin_plugins) >= 2:
        raise RuntimeError(
            "Only one platform plugin can be activated, but got: "
            f"{activated_builtin_plugins}"
        )

    # 若有且仅有一个内建插件激活，则使用它
    elif len(activated_builtin_plugins) == 1:
        platform_cls_qualname = builtin_platform_plugins[activated_builtin_plugins[0]]()
        logger.debug(
            "Automatically detected platform %s.", activated_builtin_plugins[0]
        )

    # 如果一个都没有检测到，则退化到 UnspecifiedPlatform
    else:
        platform_cls_qualname = "cfie.platforms.interface.UnspecifiedPlatform"
        logger.debug("No platform detected, vLLM is running on UnspecifiedPlatform")

    return platform_cls_qualname


# 当前平台对象，采用懒初始化
_current_platform = None

# 记录 current_platform 第一次初始化时的调用栈，便于调试
_init_trace: str = ""

if TYPE_CHECKING:
    # 仅供静态类型检查器使用
    current_platform: Platform


def __getattr__(name: str):
    """
    模块级动态属性访问。
    这里主要用来实现 current_platform 的懒初始化。
    """
    if name == "current_platform":
        # current_platform 采用懒初始化，原因：
        #
        # 1. 外部平台插件通常会写：
        #    from cfie.platforms import Platform
        #    用于继承 Platform 类
        #    所以在 cfie.platforms 模块导入过程中，不能太早解析 current_platform，
        #    否则会引发导入时序问题。
        #
        # 2. 用户可能仅仅执行 import cfie，
        #    而 cfie 内部某些代码会在导入时访问 current_platform。
        #    我们必须保证 current_platform 只在插件加载完成后才解析。
        global _current_platform

        # 若尚未初始化，则现在初始化
        if _current_platform is None:
            # 先解析出平台类的完整限定名
            platform_cls_qualname = resolve_current_platform_cls_qualname()

            # 再动态导入并实例化该平台类
            _current_platform = resolve_obj_by_qualname(platform_cls_qualname)()

            # 记录初始化调用栈
            global _init_trace
            _init_trace = "".join(traceback.format_stack())

        return _current_platform

    elif name in globals():
        # 若访问的是模块内已有全局变量，则直接返回
        return globals()[name]
    else:
        # 否则抛标准属性不存在异常
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


def __setattr__(name: str, value):
    """
    模块级动态属性赋值。
    主要允许外部显式覆盖 current_platform。
    """
    if name == "current_platform":
        global _current_platform
        _current_platform = value
    elif name in globals():
        globals()[name] = value
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


# 对外导出的符号
__all__ = ["Platform", "PlatformEnum", "current_platform", "CpuArchEnum", "_init_trace"]
