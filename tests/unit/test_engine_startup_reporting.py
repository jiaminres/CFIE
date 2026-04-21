"""Unit tests for engine startup failure reporting."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cfie.v1.engine.utils import EngineZmqAddresses, wait_for_engine_startup


class _FakeProcess:

    def __init__(self, name: str, sentinel: int, exitcode: int):
        self.name = name
        self.sentinel = sentinel
        self._exitcode_after_join = exitcode
        self.exitcode: int | None = None

    def join(self, timeout: float | None = None) -> None:
        self.exitcode = self._exitcode_after_join


class _FakeProcManager:

    def __init__(self, *processes: _FakeProcess):
        self.processes = list(processes)

    def sentinels(self) -> list[int]:
        return [proc.sentinel for proc in self.processes]

    def finished_procs(self) -> dict[str, int]:
        return {
            proc.name: proc.exitcode
            for proc in self.processes
            if proc.exitcode is not None
        }


class _FakePoller:

    def __init__(self, events: list[tuple[object, int]]):
        self._events = events

    def register(self, _obj: object, _mask: int) -> None:
        return None

    def poll(self, _timeout: int) -> list[tuple[object, int]]:
        return self._events


def test_wait_for_engine_startup_reports_triggered_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handshake_socket = object()
    proc = _FakeProcess("EngineCore", sentinel=101, exitcode=-9)
    proc_manager = _FakeProcManager(proc)

    monkeypatch.setattr(
        "cfie.v1.engine.utils.zmq.Poller",
        lambda: _FakePoller([(proc.sentinel, 1)]),
    )

    with pytest.raises(RuntimeError, match=r"Failed core proc\(s\): .*EngineCore.*-9"):
        wait_for_engine_startup(
            handshake_socket=handshake_socket,
            addresses=EngineZmqAddresses(inputs=[], outputs=[]),
            core_engines=[],
            parallel_config=SimpleNamespace(
                data_parallel_size_local=1,
                data_parallel_hybrid_lb=False,
                data_parallel_external_lb=False,
            ),
            coordinated_dp=False,
            cache_config=SimpleNamespace(),
            proc_manager=proc_manager,
            coord_process=None,
        )
