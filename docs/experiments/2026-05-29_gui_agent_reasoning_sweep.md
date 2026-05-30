# 2026-05-29 GUI Agent Reasoning Sweep

## Scope

This entry records the Doubao Web GUI Agent workflow experiments used to tune
reasoning mode and prompt policy.

Scenario: the GUI Agent reads a JSONL question set, submits each item to
Doubao Web, records the visible answer to JSONL, and calls `finish_subtask`.
This is only a validation scenario for the generic GUI Agent client; the
product code must not hard-code Doubao or evaluation-specific behavior.

Current active sweep directory:

```text
.bench_logs/gui_agent_reasoning_overnight_20260529_044106
```

## Code Fixes During Sweep

- `cfie_gui_agent.runner`
  - Normalizes `write_text_file` / `append_text_file` text arguments.
  - If the model passes a JSON object as `text`, the harness serializes it to a
    UTF-8 JSON string.
  - For `.jsonl` `append_text_file`, the harness appends a trailing newline if
    missing, preventing multiple JSON objects from being glued into one line.
- `cfie_gui_agent.verifier`
  - Repeated text-submit detection now includes a hash of submitted text.
  - Re-submitting the same item can still trigger loop protection.
  - Submitting different questions in sequence no longer falsely triggers
    repeated-action blocking.
- `benchmarks/run_gui_agent_reasoning_sweep.py`
  - Added `guided`, `default`, and `off` reasoning-mode sweeps.
  - Added `low_guided_keyboard_s900` repeats.
  - The keyboard prompt now explicitly says: after a visible answer appears for
    the current item, record success/mismatch and do not resubmit that item.

Verification:

```text
python -m pytest tests/unit/test_gui_agent_architecture.py tests/unit/test_gui_agent_desktop_client.py tests/unit/test_cfie_client_gui_agent.py -q
107 passed
```

## Interim Results

These are from the latest clean run after all three fixes above. The run was
stopped after the six-repeat core sweep finished; later exploratory cases were
not needed for the default decision.

| Config | Runs | Status | Correct | Failures | Avg response | Avg steps | Reasoning chars |
|---|---:|---|---:|---:|---:|---:|---:|
| `medium_guided_keyboard_s900` | 6 | completed | 18 / 18 | 0 | 27.413s | 11.0 | 158 |
| `off_keyboard_s900` | 6 | mixed | 10 / 18 | 3 | 29.312s | 9.2 | 0 |
| `low_guided_keyboard_s900` | 6 | mixed | 12 / 18 | 1 | 29.910s | 10.8 | 171 |
| `low_default_keyboard_s900` | 6 | mixed | 15 / 18 | 1 | 35.965s | 9.8 | 155-218 |
| `medium_default_thinking_keyboard_s900` | 1 | failed | 0 / 3 | 1 | 37.246s | 4.0 | 185 |

## Interpretation

`medium_guided_keyboard_s900` is the current best default candidate. It is the
only tested mode that completed six consecutive core runs with all answers
correct, no human-block failure, and no obvious latency penalty compared with
thinking-off mode.

`off_keyboard_s900` is fast, but it has already produced both a mismatch and a
human-wait failure across repeated runs. For this workflow, disabling thinking
does not look reliable enough as a default.

`low_default_keyboard_s900` uses Qwen default thinking without the CFIE guided
preamble. It tends to reason longer, sometimes over-corrects visible UI state,
and is slower. It is useful for debugging but not a good latency-oriented
default.

`low_guided_keyboard_s900` keeps reasoning short, but the tested runs showed
answer-recording mistakes. It is less attractive than `medium_guided`.

## Current Recommendation

For GUI Agent automation tasks that need correctness and acceptable latency:

```text
reasoning_mode = guided
reasoning_effort = medium
max_output_tokens = 1024
max_visual_frames = 24
screenshot_size = 900
prompt_variant = keyboard_submit
```

Keep the guided reasoning preamble short. The current medium preamble is:

```text
Current think mode: medium. Use two to four short clauses: state, uncertainty,
action, and risk. Keep reasoning compact; avoid enumerating visible UI details.
```

The core sweep is complete. The recommendation should be treated as the current
default until a broader task family contradicts it.
## Human Intervention And Macro Proposal Update

- `request_human_help` now carries a `blocking` boolean.
  - `blocking=true`: current subtask moves to `waiting_human`, used for captcha, login, account risk, or any state where continuing would be unsafe.
  - `blocking=false`: request is queued in the shared human loop while the active workflow may continue other work.
- Non-blocking human requests store a resume snapshot:
  - active job/subtask;
  - current screenshot reference;
  - recent step summaries;
  - compact conversation text preview.
  The intent is to let the harness restore the request-time context when a manager reply arrives, process that reply as human feedback, then switch back to the current task context.
- Added `propose_action_macro` as a model-callable generic tool.
  - A macro proposal contains a name, purpose, dynamic parameters, ordered steps, and per-step action/purpose.
  - Click-like steps include a small preview crop centered on the proposed click point when the screen backend supports local crops.
  - Macro proposals are non-blocking human requests, so approval/revision does not stop unrelated work.
- Client-side approval can convert a macro proposal into an `ActionMacro`.
  - Approved macros are added to the active macro registry and included in future runtime context.
  - Macro execution supports simple `{{parameter_name}}` replacement for dynamic text fields.

Validation:

```text
135 passed, 2 warnings
```

## Click Local Crop And Context Binding Fix

- Click fallback rule changed to match the production expectation: every `click` or `double_click` inside `computer_use` now produces one local high-resolution crop around that click point and appends it to the next model input.
- Compound actions are handled per click. For example, `click -> type -> click` appends two local crops, and the last crop becomes the active `local_refinement_1000` coordinate space for a follow-up correction.
- The inspector detail binding was fixed:
  - `operation` cards no longer inherit a random previous model request context.
  - `step` cards prefer the recorded `model_response_step`.
  - legacy traces fall back to the nearest previous model response, not a future response with the same numeric `step_id`.
- New trace records include `task_id`, `app_id`, and `model_response_step`, so future UI inspection can distinguish "first request" from later accumulated-context requests.

Validation:

```text
71 passed
```

## Desktop UI Refresh And Macro Smoke

Scope:

- Smooth the desktop client refresh path so focus changes and polling do not
  rebuild the full app list, timeline, and inspector when the underlying data
  has not changed.
- Keep terminal sessions visually stable: completed sessions do not show a
  runnable play icon, and app status no longer changes between focused and
  unfocused states.
- Verify the generic macro proposal path in the Doubao Web 5-question workflow.

Code changes:

- `cfie_gui_agent.desktop_client_ui`
  - Added app/timeline/inspector content signatures.
  - Debounced focus refresh.
  - Added session deletion from the app context menu.
  - Prevented stale loaded `running` traces from overriding persisted terminal
    app status.
- `cfie_gui_agent.runner`
  - Added a generic macro reminder after two repeated successful submit/record
    cycles.
  - Added a stricter rule that active tasks must end each model turn with an
    executable tool call, not reasoning-only empty output.
- `cfie_gui_agent.tools`
  - Tightened `propose_action_macro` so macro steps are computer-use actions
    only. File writes, shell calls, memory, and trace records stay outside the
    low-latency macro.
- `cfie.entrypoints.openai.reasoning_template`
  - Added the same "reasoning must be followed by an action" hint to the Qwen
    reasoning preamble. This requires service restart before it affects the
    running OpenAI server.

Validation:

```text
124 passed
```

Doubao Web 5-question run after UI refresh:

```text
trace: runs/gui_agent/traces/doubao_5q_smooth_20260529_201336.jsonl
result: .bench_logs/gui_agent_client_doubao/doubao_5q_ui_verify_results.jsonl
outcome: completed
questions: 5/5 pass
client stderr: empty
macro proposal: not requested in this run
empty-response retry: 5
```

Doubao Web 5-question macro run:

```text
trace: runs/gui_agent/traces/doubao_5q_macro2_20260529_205428.jsonl
outcome: completed
questions: 5/5 pass
macro_proposal_reminder: 1
propose_action_macro: 1, non-blocking, macro_approval_requested
empty-response retry: 3
```

Doubao Web macro schema tightening run:

```text
trace: runs/gui_agent/traces/doubao_5q_macro3_20260529_211151.jsonl
outcome: completed
questions: 5/5 pass
macro_proposal_reminder: 1
propose_action_macro: 1, non-blocking, macro_approval_requested
empty-response retry: 4
client stderr: empty
```

Macro proposal in the tightened run used only computer-use steps:

```text
macro_name: doubao_submit_and_record
steps:
  1. submit_text(text="{{input_text}}")
  2. wait(seconds=3)
```

Remaining issue:

- The currently running OpenAI server had already loaded the older Qwen
  reasoning preamble, so the server-side "reasoning must be followed by action"
  hint was not active during these runs. The next engine restart should retest
  whether `empty_response_without_tool_call` retries drop.
