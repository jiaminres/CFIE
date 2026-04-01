"""CLI bootstrap for CFIE."""

from __future__ import annotations

# `argparse` 负责构建整个命令行解析器。
import argparse
# `Sequence` 让 `main(argv=...)` 可以接收多种参数序列类型。
from collections.abc import Sequence

# 注册原生 v1 聊天子命令。
from cfie.cli.native_chat import add_native_chat_parser
# 注册原生 v1 单次生成子命令。
from cfie.cli.native_generate import add_native_generate_parser
# `run-local` 当前复用 `serve` 的本地执行逻辑。
from cfie.cli.run_local import run_local
# `serve` 是轻量推理引擎入口。
from cfie.cli.serve import run_serve
# 读取 CLI 默认值与枚举范围。
from cfie.config import defaults
# 在真正执行前统一初始化日志等级。
from cfie.utils.logging import configure_logging


def _add_common_engine_args(parser: argparse.ArgumentParser) -> None:
    # 共用一套参数定义，确保 `serve` 与 `run-local` 行为一致。
    # 指定要加载的模型路径或模型名。
    parser.add_argument("--model", required=True)
    # 可选地固定模型 revision。
    parser.add_argument("--revision", default=None)
    # 是否允许加载远端自定义代码。
    parser.add_argument("--trust-remote-code", action="store_true")
    # 是否仅使用本地已存在文件。
    parser.add_argument("--local-files-only", action="store_true")
    # 指定权重加载格式。
    parser.add_argument("--load-format",
                        default=defaults.DEFAULT_LOAD_FORMAT,
                        choices=defaults.SUPPORTED_LOAD_FORMATS)
    # 可选地覆盖模型下载目录。
    parser.add_argument("--download-dir", default=None)
    # 指定推理 dtype。
    parser.add_argument("--dtype",
                        default=defaults.DEFAULT_DTYPE,
                        choices=defaults.SUPPORTED_DTYPES)
    # 设置模型可见的最大上下文长度。
    parser.add_argument("--max-model-len",
                        type=int,
                        default=defaults.DEFAULT_MAX_MODEL_LEN)
    # 设置调度器允许并发的最大序列数。
    parser.add_argument("--max-num-seqs",
                        type=int,
                        default=defaults.DEFAULT_MAX_NUM_SEQS)
    # 设置显存使用比例上限。
    parser.add_argument("--gpu-memory-utilization",
                        type=float,
                        default=defaults.DEFAULT_GPU_MEMORY_UTILIZATION)
    # 指定量化后端。
    parser.add_argument("--quantization",
                        default=defaults.DEFAULT_QUANTIZATION,
                        choices=defaults.SUPPORTED_QUANTIZATION)
    # 指定 KV cache 的数据类型。
    parser.add_argument("--kv-cache-dtype",
                        default=defaults.DEFAULT_KV_CACHE_DTYPE,
                        choices=defaults.SUPPORTED_KV_CACHE_DTYPES)
    # 指定权重 offload 后端。
    parser.add_argument("--weight-offload-backend",
                        default=defaults.DEFAULT_WEIGHT_OFFLOAD_BACKEND,
                        choices=defaults.SUPPORTED_WEIGHT_OFFLOAD_BACKENDS)
    # 指定 KV offload 后端。
    parser.add_argument("--kv-offload-backend",
                        default=defaults.DEFAULT_KV_OFFLOAD_BACKEND,
                        choices=defaults.SUPPORTED_KV_OFFLOAD_BACKENDS)
    # 为通用 CPU offload 预留内存预算。
    parser.add_argument("--cpu-offload-gb",
                        type=float,
                        default=defaults.DEFAULT_CPU_OFFLOAD_GB)
    # 为 MoE CPU expert cache 设置硬上限。
    parser.add_argument("--moe-cpu-budget-gb",
                        type=float,
                        default=defaults.DEFAULT_MOE_CPU_BUDGET_GB)
    # 为系统剩余可用内存设置保底线。
    parser.add_argument("--moe-cpu-min-free-gb",
                        type=float,
                        default=defaults.DEFAULT_MOE_CPU_MIN_FREE_GB)
    # 指定 NVMe offload 存放路径。
    parser.add_argument("--nvme-offload-path",
                        default=defaults.DEFAULT_NVME_OFFLOAD_PATH)
    # 设置 offload 预取窗口大小。
    parser.add_argument("--offload-prefetch-window",
                        type=int,
                        default=defaults.DEFAULT_OFFLOAD_PREFETCH_WINDOW)
    # 设置日志输出等级。
    parser.add_argument("--log-level",
                        default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    # 可选地直接传入单条 prompt。
    parser.add_argument("--prompt", default=None)
    # 控制本次生成的最大新 token 数。
    parser.add_argument("--max-new-tokens", type=int, default=64)
    # 为请求关联一个会话 ID。
    parser.add_argument("--session-id", default="cli")
    # 在无 prompt 时执行多少个空步。
    parser.add_argument("--steps", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    # 创建顶层 `cfie` 命令解析器。
    parser = argparse.ArgumentParser(prog="cfie")
    # 要求用户必须选择一个子命令。
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 注册轻量引擎 `serve` 子命令。
    serve_parser = subparsers.add_parser("serve", help="Start local CFIE engine")
    # 为 `serve` 注入通用推理参数。
    _add_common_engine_args(serve_parser)
    # 将 `serve` 分发到 `run_serve()`。
    serve_parser.set_defaults(handler=run_serve)

    # 注册本地运行别名 `run-local`。
    run_local_parser = subparsers.add_parser("run-local",
                                             help="Run CFIE in local mode")
    # `run-local` 与 `serve` 复用同一组参数。
    _add_common_engine_args(run_local_parser)
    # 将 `run-local` 分发到 `run_local()`。
    run_local_parser.set_defaults(handler=run_local)

    # 注册原生 v1 的交互式聊天命令。
    add_native_chat_parser(subparsers)
    # 注册原生 v1 的单次生成命令。
    add_native_generate_parser(subparsers)

    # 返回构造完成的命令解析器。
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # 先构建完整的 CLI 解析器。
    parser = build_parser()
    # 解析命令行参数；`argv=None` 时读取真实命令行。
    args = parser.parse_args(argv)
    # 在执行具体子命令前先配置日志系统。
    configure_logging(args.log_level)
    # 调用子命令预先绑定好的处理函数。
    return args.handler(args)


if __name__ == "__main__":  # pragma: no cover
    # 允许通过 `python -m cfie.cli.main ...` 直接启动 CLI。
    raise SystemExit(main())
