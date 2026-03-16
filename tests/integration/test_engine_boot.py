"""Minimal integration test for phase-0 engine startup."""

from __future__ import annotations

from cfie.config.schema import EngineConfig
from cfie.runtime.engine import Engine


def test_engine_can_boot_and_step() -> None:
    cfg = EngineConfig.from_flat_kwargs(model="./model")
    engine = Engine(cfg)

    engine.run(steps=1)

    assert engine.step_count == 1
    assert engine.running is False
