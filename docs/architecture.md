# CFIE Architecture (Phase 0)

This phase provides a runnable project skeleton and a validated configuration entrypoint.

## Runtime Path

`CLI -> EngineConfig validation -> Engine startup -> no-op step loop`

## Implemented Modules

- `cfie/config/*`: defaults, validators, dataclass schema.
- `cfie/cli/*`: `serve` and `run-local` command entrypoints.
- `cfie/runtime/engine.py`: minimal engine lifecycle (`start/step/stop`).

## Pending for Next Phase

- Real model loader integration.
- Request scheduler and token generation loop.
- Streaming output and API server integration.
