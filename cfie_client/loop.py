from __future__ import annotations

from pathlib import Path

from cfie_client.executor import ComputerBackend, ComputerExecutor
from cfie_client.protocol import (
    ComputerCall,
    ComputerCallOutput,
    build_computer_call_output,
    find_computer_calls,
)
from cfie_client.safety import SafetyGate
from cfie_client.screen import PillowScreenCapture, ScreenCapture
from cfie_client.trace import TraceStore


class ComputerLoop:
    def __init__(
        self,
        *,
        executor: ComputerExecutor | None = None,
        backend: ComputerBackend | None = None,
        screen: ScreenCapture | None = None,
        safety: SafetyGate | None = None,
        trace_path: str | Path | None = None,
        trace_artifact_dir: str | Path | None = None,
    ) -> None:
        if executor is not None and backend is not None:
            raise ValueError("Pass either executor or backend, not both")
        self.executor = executor if executor is not None else ComputerExecutor(backend)
        self.screen = screen if screen is not None else PillowScreenCapture()
        self.safety = safety if safety is not None else SafetyGate()
        self.trace = TraceStore(
            Path(trace_path) if trace_path is not None else None,
            Path(trace_artifact_dir) if trace_artifact_dir is not None else None,
        )

    def handle_call(self, call_like) -> ComputerCallOutput:
        call = (
            call_like
            if isinstance(call_like, ComputerCall)
            else ComputerCall.from_openai(call_like)
        )
        screen_size = self.screen.size()
        self.safety.check_call(call, screen_size=screen_size)
        self.trace.record_tool_call(call.to_openai_dict())
        self.executor.execute_all(call.actions)
        screenshot = self.screen.screenshot()
        output = build_computer_call_output(
            call_id=call.call_id,
            image_url=screenshot.image_url,
            acknowledged_safety_checks=call.pending_safety_checks,
        )
        self.trace.record_tool_result(output.to_openai_dict())
        return output

    def handle_response(self, response_or_items) -> tuple[ComputerCallOutput, ...]:
        return tuple(
            self.handle_call(call)
            for call in find_computer_calls(response_or_items)
        )
