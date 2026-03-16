"""CLI bootstrap for CFIE."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from cfie.cli.run_local import run_local
from cfie.cli.serve import run_serve
from cfie.config import defaults


def _add_common_engine_args(parser: argparse.ArgumentParser) -> None:
    # 共用一套参数定义，确保 `serve` 与 `run-local` 行为一致。
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype",
                        default=defaults.DEFAULT_DTYPE,
                        choices=defaults.SUPPORTED_DTYPES)
    parser.add_argument("--max-model-len",
                        type=int,
                        default=defaults.DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs",
                        type=int,
                        default=defaults.DEFAULT_MAX_NUM_SEQS)
    parser.add_argument("--gpu-memory-utilization",
                        type=float,
                        default=defaults.DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--quantization",
                        default=defaults.DEFAULT_QUANTIZATION,
                        choices=defaults.SUPPORTED_QUANTIZATION)
    parser.add_argument("--kv-cache-dtype",
                        default=defaults.DEFAULT_KV_CACHE_DTYPE,
                        choices=defaults.SUPPORTED_KV_CACHE_DTYPES)
    parser.add_argument("--weight-offload-backend",
                        default=defaults.DEFAULT_WEIGHT_OFFLOAD_BACKEND,
                        choices=defaults.SUPPORTED_WEIGHT_OFFLOAD_BACKENDS)
    parser.add_argument("--kv-offload-backend",
                        default=defaults.DEFAULT_KV_OFFLOAD_BACKEND,
                        choices=defaults.SUPPORTED_KV_OFFLOAD_BACKENDS)
    parser.add_argument("--cpu-offload-gb",
                        type=float,
                        default=defaults.DEFAULT_CPU_OFFLOAD_GB)
    parser.add_argument("--nvme-offload-path",
                        default=defaults.DEFAULT_NVME_OFFLOAD_PATH)
    parser.add_argument("--offload-prefetch-window",
                        type=int,
                        default=defaults.DEFAULT_OFFLOAD_PREFETCH_WINDOW)
    parser.add_argument("--steps", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfie")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 使用 `set_defaults(handler=<...)` 统一命令分发入口。
    serve_parser = subparsers.add_parser("serve", help="Start local CFIE engine")
    _add_common_engine_args(serve_parser)
    serve_parser.set_defaults(handler=run_serve)

    run_local_parser = subparsers.add_parser("run-local",
                                             help="Run CFIE in local mode")
    _add_common_engine_args(run_local_parser)
    run_local_parser.set_defaults(handler=run_local)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
