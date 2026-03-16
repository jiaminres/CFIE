"""Local run command entrypoint for CFIE."""

from __future__ import annotations

from cfie.cli.serve import run_serve

from argparse import Namespace


def run_local(args: Namespace) -> int:
    return run_serve(args)
