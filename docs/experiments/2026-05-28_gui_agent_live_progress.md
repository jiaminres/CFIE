# 2026-05-28 GUI Agent Live Progress

## Scope

- Windows local run.
- Goal: restart the 122B OpenAI/Responses service, launch the GUI Agent desktop client, verify the live Responses path, and continue the Doubao Web GUI Agent workflow work.

## Engine Startup

Command shape:

```powershell
..\.venv\Scripts\python.exe -m cfie.entrypoints.openai.api_server `
  --model D:\models\Qwen3.5-122B-A10B-GPTQ-Int4\snapshots\5b9f0050d3ec98b0c81a7716776533c5eacebb64 `
  --served-model-name qwen35 qwen35-vl `
  --host 127.0.0.1 `
  --port 8000 `
  --max-model-len 128000 `
  --max-num-seqs 1 `
  --max-num-batched-tokens 2096 `
  --kv-cache-memory-bytes 4000000000 `
  --gpu-slots-per-layer 16 `
  --prefill-burst-slots 256 `
  --prepare-cpu-copy-threads 32 `
  --cpu-static-pinned-gb 40 `
  --enable-prefix-caching `
  --marlin-input-dtype fp8 `
  --reasoning-parser qwen3 `
  --default-chat-template-kwargs '{"enable_thinking":false}' `
  --limit-mm-per-prompt '{"image":8,"video":0}' `
  --enforce-eager
```

Runtime notes:

- Port: `127.0.0.1:8000`.
- Served model aliases: `qwen35`, `qwen35-vl`.
- Startup ready line: `Application startup complete`.
- Weight loading: `29.87s`.
- Full model load and tiered expert cache attachment: `112.34s`.
- Multi-modal warmup: `4.25s`.
- Ready around `2m29s` after launch.
- GPU memory after ready: about `24.7 GiB / 31.8 GiB`.
- KV cache: `3.69 GiB`; estimated single-request max context `148,816` tokens.
- Prefix cache: enabled.
- Thinking: default disabled through `enable_thinking=false`; `--reasoning-parser qwen3` should be configured whenever reasoning may be enabled by a request.

## Client Startup

Command:

```powershell
..\.venv\Scripts\python.exe -m cfie_gui_agent.desktop_client
```

Observed state:

- The desktop client launched successfully.
- It loaded persisted state from `runs/gui_agent/state.json`.
- Selected session: `Doubao Web`.
- Trace path from persisted state: `runs\gui_agent\traces\doubao_eval_20260528_174134.jsonl`.
- Screenshot after launch: `.bench_logs\20260528_gui_agent_client\desktop_after_launch.png`.

## Verification

Smoke request:

- `/health`: success.
- `/v1/models`: returned `qwen35`.
- Text `/v1/responses`: success.

Unit tests:

```text
tests/unit/test_gui_agent_desktop_client.py
tests/unit/test_gui_agent_architecture.py
tests/unit/test_cfie_client_gui_agent.py
tests/unit/test_gui_agent_openai_responses.py
tests/unit/test_responses_tool_call_normalizer.py
```

Result:

```text
113 passed
```

After the `input_image.detail` adapter fix:

```text
tests/unit/test_gui_agent_openai_responses.py
tests/unit/test_cfie_client_gui_agent.py
```

Result:

```text
40 passed
```

## Issue Found: Responses Image Detail

Live test found that a typed Responses image part without `detail` is rejected by the server with HTTP 400.

Failing input shape:

```json
{
  "type": "input_image",
  "image_url": "data:image/jpeg;base64,..."
}
```

Fix:

- `cfie_client.responses_adapter` now adds `detail: "auto"` to `input_image` parts when the caller omitted it.
- This keeps GUI Agent and other client-side protocol users from hitting server-side validation errors.

Post-fix visual smoke result:

- 1280x720 JPEG screenshot input.
- `input_tokens=906`.
- Latency: `8.074s` with `max_output_tokens=40`.
- Adapter debug confirmed `detail="auto"`.

## Issue Fixed: Reasoning Parser Requirement

Request-side thinking control is injected before generation. The correct
Responses API behavior is to parse Qwen `<think>...</think>` output into a
standard `type="reasoning"` output item, not to add CFIE-private audit fields
or let `</think>` leak into ordinary output text.

Fix:

- Removed the temporary `cfie_*` reasoning audit fields.
- Responses output now keeps the request-level `reasoning` object standard.
- If a request enables reasoning but the server was not launched with a
  reasoning parser, the request is rejected with a clear error.
- For Qwen3/Qwen3.5, launch with `--reasoning-parser qwen3`.

Verification:

```text
tests/unit/test_openai_reasoning_template.py
tests/unit/test_gui_agent_openai_responses.py
tests/unit/test_responses_tool_call_normalizer.py
```

Result:

```text
28 passed before the direction change; after removing CFIE-private audit fields,
the targeted reasoning-template suite passed with 11 tests.
```

Live restart:

- Restarted the live service with `--reasoning-parser qwen3`.
- `/v1/models` returned both `qwen35` and `qwen35-vl`.
- Default no-thinking smoke:
  - No CFIE-private reasoning audit fields.
  - `reasoning` is `null`.
  - Output is a normal `type="message"` item.
- `reasoning.effort="low"` smoke:
  - Top-level `reasoning` remains the standard request object:
    `{"effort":"low"}`.
  - The model immediately closed the think block, so no `type="reasoning"`
    item was emitted.
  - `</think>` did not leak into ordinary output text.
- `reasoning.effort="high"` smoke:
  - The parser emitted a standard `type="reasoning"` output item with
    `content[].type="reasoning_text"`.
  - The response hit `max_output_tokens` while still reasoning, which confirms
    why GUI Agent should keep thinking disabled or use only short efforts for
    interactive operation.

## Current GUI Agent Notes

- Client persistence is working.
- The visible client session still contains older trace items, including historical `</think>` fragments from previous experiments. Those are old trace records, not new output from the restarted no-thinking engine.
- Main next development target remains the Doubao Web session through the client, but product code must stay generic: Doubao is only a configured session/task, not a hardcoded workflow.

## 2026-05-29 Protocol And Reasoning Control Tasks

### Architectural Constraints

- Keep the project clean: remove obsolete experiments, dead code, and misleading compatibility shims instead of leaving parallel paths active.
- GUI Agent must not rewrite or invent Responses protocol output items. It may read standard `response.output` items and render or execute them, but Qwen text tool-call normalization belongs in the OpenAI Responses protocol layer under `cfie/entrypoints/openai/responses/`.
- If the protocol layer cannot parse a tool call safely, it must expose an incomplete/error state instead of letting the Agent layer silently recover a different tool.
- Doubao/Web automation is only a configured GUI Agent session used for validation. Product code must remain generic.

### Task 1: Reasoning Prompt Design Evaluation

Evaluate whether the current `low` / `medium` / `high` reasoning preambles actually control output length and behavior.

Acceptance criteria:

- Each effort has a clear target behavior:
  - `low`: one short state check plus next action, minimal latency.
  - `medium`: concise verification of state, action, and failure risk.
  - `high`: longer verification only when correctness matters, still bounded.
- Measure for each effort:
  - reasoning character count;
  - visible output character count;
  - whether `</think>` closes before final/tool output;
  - whether a valid standard Responses tool call is produced;
  - whether tool-call XML/text leaks into visible output.
- If current prompts do not reliably bound reasoning length, redesign the preambles and retest before using them in GUI Agent flows.

### Task 2: Remove Agent-Side Protocol Rewriting

Remove GUI Agent code that normalizes Qwen text tool calls by mutating Responses objects.

Acceptance criteria:

- `OpenAIResponsesAgent` returns the server Responses object unchanged except for private request debug metadata used by local tracing.
- `GuiAgentRunner` does not call a protocol-normalization function before parsing.
- Client-side helper functions only parse already-standard `function_call` / `computer_call` / `message` items for execution and display.
- Tests that expected Agent-side text tool-call normalization are moved to the Responses protocol-layer tests or removed.

### Task 3: Reasoning Preamble Must Be Visible In Standard Protocol

When reasoning is enabled, the response `reasoning.content` should include both the injected reasoning preamble and the model-generated reasoning text.

Acceptance criteria:

- For Qwen reasoning, `response.output` contains a standard `type="reasoning"` item when thinking is enabled and any reasoning/preamble exists.
- The `reasoning.content[].text` starts with the selected preamble or otherwise clearly includes it.
- Empty model reasoning still records the preamble so callers can audit which reasoning mode was used.
- No CFIE-private reasoning audit fields are added to the public protocol object.

### Task 4: Reasoning Context Writeback

Every GUI Agent turn must write the assistant's standard reasoning item back into the next-turn context along with the assistant tool call and tool result.

Acceptance criteria:

- If a response has `type="reasoning"`, the next request context includes that reasoning item before the related assistant function call.
- Tool call outputs still follow the matching `call_id`.
- The next-turn prompt is complete enough for the model to understand its previous reasoning and action.

### Task 5: Low / Medium / High GUI Task Comparison

After protocol fixes, run the same generic GUI Agent task under `low`, `medium`, and `high` reasoning.

Acceptance criteria:

- Use the same screenshot policy, tool profile, max steps, and target task.
- Record completion status, number of model turns, number of executed tools, average response latency, reasoning chars, visible chars, and whether human intervention was requested.
- Report which effort is best for interactive GUI automation.

### 2026-05-29 Implementation Update

- Agent-side Responses rewriting has been removed from `cfie_gui_agent`.
  - `OpenAIResponsesAgent` now returns the server response object unchanged, except local `_cfie_request_debug` tracing metadata.
  - `GuiAgentRunner` no longer calls Agent-side Qwen text tool-call normalization.
  - Agent helpers only consume already-standard `function_call` / `tool_call` output items.
- Qwen text tool-call normalization is owned by the Responses protocol layer.
  - `<tool_call>...</tool_call>`, bare `<function=...>`, `<tool_code>...</tool_code>`, and nested agent-tool actions inside `computer_use.actions` are handled in `cfie/entrypoints/openai/responses/tool_call_normalizer.py`.
- Reasoning preambles have been redesigned for bounded GUI automation behavior.
  - `low`: previous state plus next action only.
  - `medium`: state, uncertainty, action, and risk.
  - `high`: state, target, risk, and fallback verification, still bounded.
- Qwen reasoning parser now prepends the selected preamble to standard `reasoning.content`.
  - If the model immediately emits `</think>` with no generated reasoning, the preamble is still emitted as the reasoning content.
  - This replaces the previous temporary CFIE-private audit fields.
- GUI Agent now writes standard `reasoning` output items back into the next-turn context before the matching tool call and tool result.

Verification:

```text
tests/unit/test_responses_tool_call_normalizer.py
tests/unit/test_openai_reasoning_template.py
tests/unit/test_gui_agent_architecture.py
tests/unit/test_gui_agent_openai_responses.py
tests/unit/test_cfie_client_gui_agent.py::test_gui_agent_runner_writes_reasoning_back_before_tool_result
```

Result:

```text
24 passed for Responses protocol/reasoning parser suites.
55 passed for GUI Agent architecture/openai-responses/reasoning-writeback targeted suites.
```

Remaining:

- Restart the live engine with the updated code and `--reasoning-parser qwen3`.
- Run the same generic GUI Agent task with `reasoning.effort=low`, `medium`, and `high`.
- Compare completion, turns, tool count, latency, reasoning length, visible output length, and intervention rate.

### 2026-05-29 Live Reasoning Smoke

Restarted the Windows OpenAI/Responses service with the documented GUI Agent
command and `--reasoning-parser qwen3`.

Startup:

- Health endpoint became ready.
- Latest logs:
  - `.bench_logs/gui_agent_engine_restart_20260529_003708.out.log`
  - `.bench_logs/gui_agent_engine_restart_20260529_003708.err.log`
- Windows stderr still reports a GBK logging error for the startup banner
  glyphs. This is a logging/display issue, not a service startup failure.

Small `/v1/responses` reasoning smoke:

| Effort | Status | Output Types | Latency | Reasoning Chars | Visible Chars |
|---|---|---|---:|---:|---:|
| `low` | completed | `reasoning`, `message` | 5.003s | 171 | 47 |
| `medium` | completed | `reasoning`, `message` | 8.900s | 158 | 154 |
| `high` | completed | `reasoning`, `message` | 10.553s | 152 | 221 |

Interpretation:

- The current code path does emit standard `type="reasoning"` output items.
- `reasoning.content` includes the selected preamble even when the model does
  not generate additional reasoning before `</think>`.
- `usage.output_tokens_details.reasoning_tokens` can remain `0` in this case,
  because the preamble is service-side audit/context text rather than generated
  model reasoning tokens.
- This smoke only validates protocol shape. It is not a GUI Agent completion
  comparison; task-level `low` / `medium` / `high` evaluation still needs the
  same client workflow and screenshots.
