# 2026-05-24 GUI Agent Workflow: Doubao Web Validation

## Scope

This entry records the first real-model validation scenario for the generic GUI
Agent runtime.

Important boundary:

- The product is still **GUI Agent**.
- Doubao Web is only the current APP/session task configuration used to test the
  agent loop.
- Runtime code should stay generic: workflow, task, trace, tool, artifact,
  human request, monitor, and policy.
- Scenario names such as Doubao or AI app testing may appear in benchmark
  scripts, user-created APP/session configuration, logs, or sample data, but not
  in core product abstractions.

## Server

The 122B VL service was already running on Windows:

- Endpoint: `http://127.0.0.1:8000/v1`
- Model id: `qwen35-vl`
- Model root: `D:\models\Qwen3.5-122B-A10B-GPTQ-Int4\snapshots\5b9f0050d3ec98b0c81a7716776533c5eacebb64`
- Key runtime options:
  - `--max-model-len 128000`
  - `--max-num-seqs 1`
  - `--max-num-batched-tokens 8192`
  - `--kv-cache-memory-bytes 4000000000`
  - `--gpu-slots-per-layer 16`
  - `--prefill-burst-slots 256`
  - `--prepare-cpu-copy-threads 32`
  - `--cpu-static-pinned-gb 40`
  - `--enable-prefix-caching`
  - `--marlin-input-dtype fp8`
  - MTP one draft token
  - `--default-chat-template-kwargs {"enable_thinking":false}`
  - `--allowed-local-media-path C:\Users\13642\PycharmProjects\vllm\CFIE\.bench_logs\gui_agent_workflow\artifacts`

## Changes

### Screenshot Transport

The earlier Responses path used base64 `data:image` screenshots. The local
server counted those base64 bytes as text input and rejected requests with a
context-length error. The workflow path now supports file-backed screenshots:

- `ScaledPillowScreenCapture(url_mode="file")`
- screenshot output directory under the artifact root
- benchmark default `--screenshot-url-mode file`
- local server started with `--allowed-local-media-path`

The right-monitor crop used for this run:

```text
--screenshot-crop 1280,0,1280,1440
```

The captured image is downscaled to the configured screenshot size, and
`CoordinateScalingBackend` maps model coordinates back to the physical crop
origin before executing mouse actions.

### Responses Compatibility

The local Responses implementation did not accept native
`computer_call_output` items in the same way as OpenAI-hosted computer-use
models. The adapter now normalizes tool outputs into user-visible screenshot
messages, and maps `developer` to `system` for local protocol compatibility.

Qwen emits `computer_use` as a normal function tool. The runner now accepts both
native `computer_call` and function-style `computer_use` calls.

Tool argument normalization was extended for Qwen-style output:

- `action` -> `type`
- `coordinate` / `coordinates` / `point` -> `x`, `y`
- `left_click`, `right_click`, `mouse_click`
- `input`, `input_text`, `text`
- `press`, `hotkey`, `key`
- `sleep`
- JSON-array strings under `computer_use.actions`
- simple repair for missing commas in tool-call JSON

### Harness Guard

The harness now detects repeated ineffective computer actions. When the same
computer action repeats at least three times in the same task, the runner:

- records the step as `repeated_action`;
- creates a human request;
- moves the subtask to `waiting_human`;
- returns `GuiAgentResult(status="waiting_human")`.

This is intentionally generic. It is not tied to Doubao or testing. It prevents
the agent from spending all steps on a stuck modal, hidden login wall, frozen
page, invisible control, or other unrecoverable local UI state.

## Real Run

Command shape:

```powershell
..\.venv\Scripts\python.exe benchmarks\run_gui_agent_workflow_responses.py `
  --base-url http://127.0.0.1:8000/v1 `
  --model qwen35-vl `
  --target-url https://www.doubao.com/chat/ `
  --input-path .bench_logs\gui_agent_workflow\doubao_items.jsonl `
  --trace-path .bench_logs\gui_agent_workflow\doubao_trace_crop_login_guard2.jsonl `
  --artifact-dir .bench_logs\gui_agent_workflow\artifacts `
  --item-limit 1 `
  --max-steps 6 `
  --max-output-tokens 256 `
  --timeout 900 `
  --screenshot-max-width 960 `
  --screenshot-max-height 540 `
  --screenshot-jpeg-quality 85 `
  --screenshot-url-mode file `
  --screenshot-crop 1280,0,1280,1440 `
  --image-detail low `
  --result-json .bench_logs\gui_agent_workflow\doubao_crop_login_guard2_result.json
```

Result:

| Field | Value |
|---|---|
| Status | `waiting_human` |
| Steps | 4 |
| Reason | `Harness requested human input after repeated computer actions.` |
| Trace | `.bench_logs\gui_agent_workflow\doubao_trace_crop_login_guard2.jsonl` |
| Result JSON | `.bench_logs\gui_agent_workflow\doubao_crop_login_guard2_result.json` |
| Artifact root | `.bench_logs\gui_agent_workflow\artifacts` |

Observed sequence:

1. The model called `read_text_file` and read the input list correctly.
2. The model attempted to operate the Doubao page via `computer_use`.
3. The page was blocked by a login/modal state, so the model repeated the same
   click.
4. On the third repeated click, the harness converted the task into a human
   intervention request instead of continuing the loop.

This is an acceptable result for the current environment because a real logged
in Doubao session was not available. The important validation point is that the
agent loop stopped safely and left a structured trace for the client UI.

## Verification

Unit tests:

```powershell
..\.venv\Scripts\python.exe -m pytest `
  tests\unit\test_cfie_client_gui_agent.py `
  tests\unit\test_gui_agent_architecture.py `
  tests\unit\test_gui_agent_openai_responses.py `
  tests\unit\test_gui_agent_workflow.py -q
```

Result:

```text
51 passed in 1.00s
```

## Next Work

- Keep Doubao Web as the first validation APP/session.
- Improve the desktop client so the same trace is shown as user-friendly recent
  actions, screenshots, tool calls, and human requests instead of raw JSON.
- Add a logged-in or mock web-app target so the workflow can complete a full
  input -> submit -> observe -> record-result cycle.
- Keep game-specific low-latency controls as a later scenario after the generic
  APP/session loop is stable.

## 2026-05-25 Follow-Up

### Output Budget And Concision

The workflow benchmark now uses a larger model output budget for development
(`--max-output-tokens 768`) so tool-call JSON has enough room. This must not be
treated as permission for verbose model behavior. The runner now injects an
efficiency policy into the developer/runtime context:

- do not emit long thinking text;
- do not describe screenshots or videos unless that description is the task
  result;
- prefer tool calls over prose;
- for `record_workflow_result`, prefer short arguments:
  `item_id`, `output_text`, `status`, `reason`, `artifact_refs`.

The harness now records `model_response` trace events for every model round:

- model response latency;
- response JSON size;
- visible output text length;
- tool argument character count;
- function-call count;
- warnings such as visible `<think>` text or unusually large output.

This gives a concrete way to supervise whether higher output budgets are causing
slow, verbose visual reasoning instead of direct automation actions.

### Tool Robustness

Real runs exposed malformed tool-call shapes from the local model:

- truncated `record_workflow_result.arguments`;
- malformed `computer_use.actions` JSON-array strings;
- ordinary tools accidentally nested under `computer_use.actions`;
- final structured result emitted as normal text instead of as a tool call.

The runner now handles these as recoverable protocol issues:

- malformed tool arguments are returned to the model as rejected tool outputs
  instead of crashing the process;
- malformed `computer_use.actions` is surfaced as a `computer_use` tool error so
  the model can retry;
- ordinary tools nested inside `computer_use.actions` are routed back to the
  normal agent-tool path when safely identifiable;
- final-message JSON with `item_id`, `output_text`, and `status` is persisted as
  a `workflow_result` trace event.

### Item Limit

`--item-limit` is now a hard input constraint. The benchmark materializes a
limited JSONL file under:

```text
.bench_logs\gui_agent_workflow\artifacts\inputs\
```

The model only sees the limited file path. This prevents a workflow run configured
for one item from continuing into later rows just because the source file
contains more examples.

### Real Run

Run:

```text
.bench_logs\gui_agent_workflow\doubao_tool_error_retry_result.json
```

Result:

| Field | Value |
|---|---|
| Status | `completed` |
| Steps | 5 |
| Trace | `.bench_logs\gui_agent_workflow\doubao_trace_tool_error_retry.jsonl` |
| Limited input | `.bench_logs\gui_agent_workflow\artifacts\inputs\doubao_items.limit1.jsonl` |

Observed behavior:

1. The model initially produced malformed `computer_use.actions`.
2. The harness rejected that tool call with a structured JSON error instead of
   exiting.
3. The model recovered, read the limited input file, clicked the input field,
   typed the question, and pressed Enter.
4. The model observed the page answer and produced a compact final JSON result:
   `2 + 3 等于 5。`, status `passed`.

The model-response metrics showed no long visible thinking text. The visible
text lengths stayed small: 0, 0, 19, 61, and 268 characters. Response latency was
still high for interactive GUI use, around 47-64 seconds per model round on the
current 122B local setup.

### Verification

```powershell
..\.venv\Scripts\python.exe -m pytest `
  tests\unit\test_cfie_client_gui_agent.py `
  tests\unit\test_gui_agent_architecture.py `
  tests\unit\test_gui_agent_openai_responses.py `
  tests\unit\test_gui_agent_workflow.py -q
```

Result:

```text
57 passed in 0.96s
```
