"""Unit tests for CFIE-specific engine CLI arguments."""

from __future__ import annotations

import pytest

from cfie.engine.arg_utils import EngineArgs
from cfie.utils.argparse_utils import FlexibleArgumentParser


def test_engine_arg_utils_accepts_current_tiered_cache_args() -> None:
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)

    args = parser.parse_args(
        [
            "--model",
            "./model",
            "--gpu-slots-per-layer",
            "32",
            "--prefill-burst-slots",
            "256",
            "--prepare-cpu-copy-batch-size",
            "8",
        ]
    )

    engine_args = EngineArgs.from_cli_args(args)

    assert engine_args.gpu_slots_per_layer == 32
    assert engine_args.prefill_burst_slots == 256
    assert engine_args.prepare_cpu_copy_batch_size == 8


@pytest.mark.parametrize(
    "legacy_arg",
    [
        "--stage-base-slots",
        "--prefetch-base-slots",
        "--stage-min-resident",
        "--prefetch-min-resident",
        "--prepare-deviation-tolerance",
        "--allow-missing-experts",
    ],
)
def test_engine_arg_utils_rejects_removed_tiered_cache_args(
    legacy_arg: str,
) -> None:
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "./model", legacy_arg, "8"])
