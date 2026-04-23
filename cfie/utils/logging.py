"""CFIE 统一日志工具。"""

from __future__ import annotations

import logging
import os
import sys

_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str | int | None = None) -> None:
    """初始化 CFIE 日志。

    该函数是幂等的；重复调用只会更新日志级别，不会重复添加 handler。
    """

    # 获取名为 "cfie" 的根 logger。
    # 后续像 "cfie.xxx" 这样的子 logger 一般都会继承它的配置。
    root_logger = logging.getLogger("cfie")

    # 确定最终要使用的日志级别：
    # 1. 优先使用函数参数 level
    # 2. 如果没传，则读取环境变量 CFIE_LOG_LEVEL
    # 3. 如果环境变量也没有，则默认使用 "INFO"
    resolved_level = (
        level if level is not None else os.environ.get("CFIE_LOG_LEVEL", "INFO")
    )

    # 把字符串或整数形式的日志级别转换成 logging 模块内部使用的数值级别。
    # 例如 "DEBUG" -> logging.DEBUG，"INFO" -> logging.INFO。
    # 如果传入值无法识别，则回退到 logging.INFO。
    numeric_level = logging._nameToLevel.get(
        str(resolved_level).upper(),
        logging.INFO,
    )
    resolved_level_name = str(logging.getLevelName(numeric_level)).upper()

    # 将最终日志级别同步回环境变量，确保后续拉起的子进程继承同一配置。
    os.environ["CFIE_LOG_LEVEL"] = resolved_level_name
    os.environ["VLLM_LOGGING_LEVEL"] = resolved_level_name

    # 设置 "cfie" 根 logger 的日志级别。
    root_logger.setLevel(numeric_level)

    # 禁止该 logger 把日志继续向上层 logger 传播，
    # 避免和外部/全局日志系统重复打印。
    root_logger.propagate = False

    # 如果这个 logger 已经有 handler，说明之前初始化过：
    # 此时不再重复创建 handler，只更新已有 handler 的级别和输出流。
    if root_logger.handlers:
        for handler in root_logger.handlers:
            # 同步更新每个 handler 的日志级别。
            handler.setLevel(numeric_level)

            # 如果是输出到流的 handler（通常是控制台输出），
            # 则把输出流重新绑定到当前的 sys.stderr。
            if isinstance(handler, logging.StreamHandler):
                try:
                    handler.setStream(sys.stderr)
                except ValueError:
                    # 某些测试场景中，旧的 stderr 流对象可能已经关闭，
                    # setStream 会失败。此时直接强制替换底层 stream。
                    handler.stream = sys.stderr
        return

    # 如果还没有 handler，说明是第一次初始化：
    # 创建一个输出到标准错误流的 StreamHandler。
    handler = logging.StreamHandler(stream=sys.stderr)

    # 设置 handler 自身的日志级别。
    handler.setLevel(numeric_level)

    # 设置日志输出格式和时间格式。
    handler.setFormatter(
        logging.Formatter(
            fmt=_DEFAULT_FORMAT,
            datefmt=_DEFAULT_DATE_FORMAT,
        )
    )

    # 把这个 handler 挂到 "cfie" 根 logger 上。
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """获取 CFIE 命名空间 logger。"""

    configure_logging()
    if name.startswith("cfie"):
        return logging.getLogger(name)
    return logging.getLogger(f"cfie.{name}")
