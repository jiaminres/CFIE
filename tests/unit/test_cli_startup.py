"""Unit tests for CFIE CLI bootstrap."""

from __future__ import annotations

import pytest

from cfie.cli.main import main


def test_cli_serve_command_starts_engine(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([
        "serve",
        "--model",
        "./model",
        "--steps",
        "2",
        "--log-level",
        "INFO",
    ])

    assert code == 0
    stderr = capsys.readouterr().err
    assert "CFIE engine started" in stderr
    assert "CFIE engine stopped" in stderr


def test_cli_propagates_config_validation_error() -> None:
    with pytest.raises(ValueError, match="gpu_memory_utilization"):
        main([
            "serve",
            "--model",
            "./model",
            "--gpu-memory-utilization",
            "2.0",
        ])


def test_cli_chat_command_dispatches_to_handler(monkeypatch) -> None:
    import cfie.cli.native_chat as native_chat

    monkeypatch.setattr(native_chat, "run_native_chat", lambda args: 7)

    code = main([
        "chat",
        "--model",
        "./model",
    ])

    assert code == 7
