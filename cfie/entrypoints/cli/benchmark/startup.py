# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

from cfie.benchmarks.startup import add_cli_args, main
from cfie.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase


class BenchmarkStartupSubcommand(BenchmarkSubcommandBase):
    """The `startup` subcommand for `vllm bench`."""

    name = "startup"
    help = "Benchmark the startup time of vLLM models."

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
