from __future__ import annotations

from dataclasses import dataclass

from cfie_client.protocol import ACTION_TYPES, ComputerCall


class SafetyViolation(RuntimeError):
    pass


@dataclass(slots=True)
class SafetyGate:
    max_actions_per_call: int = 32
    allowed_action_types: tuple[str, ...] = ACTION_TYPES
    require_screen_bounds: bool = True

    def check_call(
        self,
        call: ComputerCall,
        *,
        screen_size: tuple[int, int] | None = None,
    ) -> None:
        if len(call.actions) > self.max_actions_per_call:
            raise SafetyViolation(
                "computer_call contains more actions than the local limit"
            )

        allowed = set(self.allowed_action_types)
        for action in call.actions:
            if action.type not in allowed:
                raise SafetyViolation(f"blocked computer action: {action.type}")

            if self.require_screen_bounds and screen_size is not None:
                self._check_bounds(action.coordinate_points, screen_size)

    def _check_bounds(
        self,
        points: tuple[tuple[int, int], ...],
        screen_size: tuple[int, int],
    ) -> None:
        width, height = screen_size
        for x, y in points:
            if x < 0 or y < 0 or x >= width or y >= height:
                raise SafetyViolation(
                    f"coordinate ({x}, {y}) is outside screen bounds {screen_size}"
                )
