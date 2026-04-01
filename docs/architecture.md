# CFIE Architecture (Phase 1)

Phase 1 delivers a GPU-only minimal inference path for single-sequence generation.

## Runtime Path

`CLI/Python SDK -> EngineConfig validation -> Request queue -> FCFS scheduler -> InputBuilder -> Executor(ModelRunner) -> OutputProcessor -> Streamer`

## Implemented Modules

- `cfie/loader/*`: 内置模型注册表优先加载（含 vLLM 借鉴的 `load_weights` 逻辑），缺失时自动回退 HF AutoModel。
- `vllm/*`（CFIE 内置快照）: vLLM 源码随 CFIE 项目内置分发，避免外部包依赖。
- `cfie/model_executor/vllm_models/*`: vLLM `model_executor/models` 源码镜像（含 Qwen3.5 与其余内置模型）。
- `cfie/vendor/vllm_shims/*`: vLLM 依赖边界层（先收口后逐步内生化，当前已覆盖 Qwen3.5）。
- `cfie/request/*`: request state machine, FIFO queue, session manager.
- `cfie/scheduler/*`: FCFS scheduler (`max_num_seqs` default `1`).
- `cfie/runtime/*`: engine loop, input builder, executor, greedy prefill/decode runner.
- `cfie/api/*`: protocol dataclasses, token streamer, local Python SDK entry.
- `cfie/cli/*`: `serve`/`run-local`, supports prompt streaming and interrupt.
- `cfie/utils/logging.py`: 统一日志初始化与命名空间 logger（支持日志级别控制）。

## Current Constraints

- Single process / single device execution path.
- Default policy is FCFS and optimized for `max_num_seqs=1`.
- Offload modules remain placeholders for Phase 2+.
