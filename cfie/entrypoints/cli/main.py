# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoints of vLLM

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import importlib.metadata
import sys

from cfie.logger import init_logger

logger = init_logger(__name__)


def main():
    import cfie.entrypoints.cli.benchmark.main
    import cfie.entrypoints.cli.collect_env
    import cfie.entrypoints.cli.launch
    import cfie.entrypoints.cli.openai
    import cfie.entrypoints.cli.run_batch
    import cfie.entrypoints.cli.serve
    from cfie.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
    from cfie.utils.argparse_utils import FlexibleArgumentParser

    CMD_MODULES = [
        cfie.entrypoints.cli.openai,
        cfie.entrypoints.cli.serve,
        cfie.entrypoints.cli.launch,
        cfie.entrypoints.cli.benchmark.main,
        cfie.entrypoints.cli.collect_env,
        cfie.entrypoints.cli.run_batch,
    ]

    cli_env_setup()

    # For 'vllm bench *': use CPU instead of UnspecifiedPlatform by default
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        logger.debug(
            "Bench command detected, must ensure current platform is not "
            "UnspecifiedPlatform to avoid device type inference error"
        )
        from cfie import platforms

        if platforms.current_platform.is_unspecified():
            from cfie.platforms.cpu import CpuPlatform

            platforms.current_platform = CpuPlatform()
            logger.info(
                "Unspecified platform detected, switching to CPU Platform instead."
            )

    parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=importlib.metadata.version("vllm"),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
