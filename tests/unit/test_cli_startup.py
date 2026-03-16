"""Unit tests for CFIE CLI bootstrap."""

from __future__ import annotations

import pytest

from cfie.cli.main import main


def test_cli_serve_command_starts_engine(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["serve", "--model", "./model", "--steps", "2"])

    assert code == 0
    output = capsys.readouterr().out
    assert "CFIE engine started" in output
    assert "CFIE engine stopped" in output


def test_cli_propagates_config_validation_error() -> None:
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        main([
            "serve",
            "--model",
            "./model",
            "--gpu-memory-utilization",
            "2.0",
        ])
