# 2026-05-25 GUI Agent Desktop Client Iteration

## Scope

- Continue GUI Agent desktop-client development with the client window open.
- Use Codex desktop UI as a visual reference for calmer spacing, lighter scrollbars, rounded cards, and concise execution history.
- Keep the product model generic: the current Doubao workflow is only one APP session/task definition, not a hard-coded product mode.

## Changes

- The desktop client can now be launched with an APP/workflow session and an existing trace:
  - `--app-name`
  - `--target-url`
  - `--input-path`
  - `--trace-path`
  - `--limit`
- Loading an existing trace no longer appends another `workflow_configured` event. The client now treats existing traces as read-first state for inspection.
- Repeated workflow configuration cards are deduplicated in the visible operation stream.
- Main window task definition is condensed to a brief summary. Full task definition remains available through the task detail view.
- Removed the explanatory "recent operations" helper card from the main conversation surface.
- Operation cards now translate internal trace kinds into user-facing labels:
  - `model_response` -> 模型响应
  - `computer_call` / `computer_use` -> 电脑操作
  - `read_text_file` -> 读取文件
  - `record_workflow_result` -> 记录结果
- Model-response cards expose concise latency/tool/output metrics instead of raw JSON-like event names.
- Tool parse failures are shown as a normal recoverable operation instead of exposing raw internal event names.
- Scrollbars were reduced from 6 px to 4 px and shifted to a lighter gray.

## Visual Checks

- Before comparison screenshot:
  - `.bench_logs/gui_agent_client/desktop_compare_current.png`
- After first UI pass:
  - `.bench_logs/gui_agent_client/desktop_client_after_ui_pass.png`
  - `.bench_logs/gui_agent_client/desktop_client_after_ui_pass2.png`
  - `.bench_logs/gui_agent_client/desktop_client_after_ui_pass3.png`
  - `.bench_logs/gui_agent_client/desktop_client_after_ui_pass4.png`

## Verification

```powershell
..\.venv\Scripts\python.exe -m pytest `
  tests\unit\test_gui_agent_desktop_client.py `
  tests\unit\test_gui_agent_workflow.py `
  tests\unit\test_cfie_client_gui_agent.py `
  tests\unit\test_gui_agent_architecture.py `
  tests\unit\test_gui_agent_openai_responses.py -q
```

Result:

```text
63 passed
```

## Remaining UI Notes

- The client is now usable for visual iteration, but it is still not as polished as Codex:
  - card density can be improved;
  - the right inspector can use richer media thumbnails instead of text-only cards;
  - action cards should eventually render screenshots/video evidence inline;
  - the settings and APP task-definition dialogs still need the same visual treatment as the main window.

## Follow-up: Trace UI and Latency Policy

User feedback:

- Configuration cards should not appear in the main execution timeline.
- Trace/path management belongs in the APP right-click menu.
- The detail pane must not expose raw internal payload objects.
- GUI Agent response latency target is under 10 seconds per step.

Changes:

- Main and inspector execution streams now hide workflow-configuration events.
  Task definition remains visible as APP metadata, while trace-file operations are
  available from the APP context menu.
- APP right-click menu now includes:
  - copy trace-file path;
  - open trace-file directory;
  - copy input-list path.
- Trace detail rendering now uses user-facing text:
  - model response latency/output/thinking counters;
  - formatted computer actions;
  - screenshot evidence refs;
  - workflow results.
  It no longer shows `kind`, raw payload fields, or nested JSON object dumps.
- Responses requests now default to:
  - `reasoning.effort=none`;
  - `chat_template_kwargs.enable_thinking=false`.
- `GuiAgentRunner` no longer rewrites the first runtime-context message on every
  step by default. This keeps the conversation prefix stable so the server-side
  prefix cache can reuse previous prompt work.
- Latency warnings now trigger at 10 seconds instead of 120 seconds.

Verification:

```text
63 passed
```

Screenshot:

- `.bench_logs/gui_agent_client/desktop_client_after_trace_ui_speed_pass.png`

## Cached History Context and APP Viewport Crop

Reason:

- Real 122B Windows timing shows 2096-token prefill is the practical low-latency
  target and 4192-token prefill is the upper interactive target.
- Sending a changing video window every step destroys that target. The GUI Agent
  should keep history images stable in the conversation and normally add only
  one new operation-result keyframe per step.
- If the useful APP occupies only part of the desktop, the first full-desktop
  observation should be used to locate the APP outline. Later screenshots should
  be cropped to that region.

Implementation:

- Added `VisionContextPolicy.keyframe()`.
- `GuiAgentRunner` defaults to the after-frame history policy:
  - no routine recent video;
  - current frame plus bounded historical after-action screenshot references;
  - existing history stays in the conversation for KV-cache reuse;
  - each new step adds one result screenshot and a compact tool/result message.
- Added model-callable `set_app_viewport`.
- Added screenshot viewport metadata to the runtime context.
- `ScaledPillowScreenCapture` can now expose and update its crop box through
  `set_crop_box()` and `viewport_context()`.
- `ComputerLoop` now maps scaled/cropped screenshot coordinates back to physical
  desktop coordinates before executing mouse actions.

Verification:

```text
60 passed
```

## Cached-History Regression Tests

User correction:

- Do not reduce model-visible history to only two images.
- Historical screenshots must remain in the conversation prefix so the model can
  use them to avoid action loops and inconsistent decisions.
- The latency target is achieved by keeping the historical prefix stable and
  appending only a small per-step delta.

Additional tests:

- `test_gui_agent_runner_keeps_stable_history_prefix_between_steps`
  - confirms the developer/runtime context message is not rewritten between
    steps;
  - confirms each `computer_use` appends one new screenshot result;
  - confirms prior screenshot outputs stay in the conversation.
- `test_gui_agent_runner_clicks_inside_cropped_app_viewport`
  - confirms `set_app_viewport` can crop the APP area;
  - confirms later `computer_use` coordinates are interpreted inside the
    cropped screenshot and mapped back to physical desktop coordinates.

Verification:

```text
tests/unit/test_cfie_client_gui_agent.py: 19 passed
GUI Agent suite: 73 passed, 2 warnings
```

## Desktop Client Surface Cleanup

User-facing direction:

- The product is a general GUI Agent client. Specific scenarios such as AI app
  evaluation are only APP/session configurations, not top-level product
  concepts.
- Main surface should feel like a conversation/workspace, not a debug console.
- APP task definition and full execution records belong in the left APP object's
  context menu. They should not be permanently exposed as large top-level tabs or
  header buttons.
- The composer should behave like a single input unit. The primary send control
  should be short (`发送` or an icon), not `发送给 Agent`.

Implementation:

- Removed visible main-surface labels/buttons such as `任务定义`, `检查器`,
  `执行轨迹`, and `向当前 APP 发送输入`.
- Left APP context menu now contains:
  - task definition view/edit;
  - execution record view;
  - trace path copy/open actions;
  - input list path copy;
  - APP ID copy and settings.
- The right record panel is hidden by default and opened from the icon or APP
  context menu.
- Composer send button is now integrated into the input shell and uses the short
  label `发送` / `提交`.
- Canvas-based rounded buttons now use Pillow high-resolution rendering when
  available to reduce jagged edges; the old Tk Canvas path remains as fallback.
- Empty trace text is reduced to a compact chat message.

Verification:

```text
tests/unit/test_gui_agent_desktop_client.py tests/unit/test_cfie_client_gui_agent.py:
25 passed
```

Screenshots:

- `.bench_logs/gui_agent_dev/desktop_client_clean_composer.png`
- `.bench_logs/gui_agent_dev/desktop_client_send_button_short_fixed.png`

## Manual APP Viewport Marker

Problem found during live Doubao workflow:

- Letting the model choose the initial APP viewport can fail badly. In the live
  run, the model set a crop that landed on the desktop wallpaper instead of the
  browser page. After that, all later screenshots and click coordinates were
  based on the wrong visual input.

Design decision:

- The target APP/window region should be mechanically or manually guaranteed
  before the model is asked to click inside it.
- The model may still call `set_app_viewport`, but the desktop client now also
  supports a human-calibrated viewport for each APP session.

Implementation:

- Added per-APP `manual_viewport` metadata:
  - `x`, `y`, `width`, `height`
  - `viewport_source=desktop_marker`
- Added APP context-menu actions:
  - `标定 APP 视野`
  - `隐藏视野框`
  - `复制视野坐标`
  - `清除视野标定`
- The marker is a topmost desktop overlay:
  - drag the top bar to move it;
  - drag borders/corners to resize it;
  - `保存隐藏` writes the viewport into the current APP config and hides the
    overlay;
  - `隐藏` closes the overlay without changing the saved viewport.
- `复制视野坐标` exports `x,y,width,height`, matching the existing
  `--screenshot-crop` format used by workflow experiments.
- Saving the viewport records a normal operation event, so the main stream and
  detail panel can show that the APP view was calibrated without exposing raw
  internal JSON.

Rationale:

- Coordinate grids can improve click precision inside a correct screenshot.
- They do not solve the larger failure mode where the cropped screenshot is the
  wrong window or wallpaper.
- Manual APP viewport marking gives the operator a low-cost fallback before
  adding more complex Windows title/handle based focusing.

## Screenshot Grid And Window Focus Hook

Additional implementation:

- `ScaledPillowScreenCapture` now supports `grid_overlay=off|coarse|fine`.
  - The grid is drawn after crop and resize.
  - Therefore the visible grid coordinates match the screenshot coordinate
    system used by `computer_use`.
  - The workflow default was changed back to `off` after coordinate-space
    normalization stabilized. The grid remains only as a debugging option.
- `benchmarks/run_gui_agent_workflow_responses.py` added:
  - `--screenshot-grid off|coarse|fine`
  - `--focus-window-title-pattern REGEX`
  - `--crop-focused-window`
- `cfie_client.executor.windows` now exposes a minimal Windows helper:
  - enumerate visible windows;
  - match by title regex;
  - bring the target window to front;
  - return the physical window crop box.

Usage direction for live GUI Agent experiments:

```powershell
--focus-window-title-pattern "Doubao|豆包|Chrome" `
--crop-focused-window `
--screenshot-grid off
```

This gives the workflow two safeguards:

- mechanical title-based focus/crop when the target window is discoverable;
- human manual viewport marker when the window title/focus route is unreliable.

Verification:

```text
tests/unit/test_gui_agent_desktop_client.py
tests/unit/test_cfie_client_gui_agent.py
tests/unit/test_gui_agent_architecture.py
tests/unit/test_gui_agent_workflow.py
tests/unit/test_gui_agent_openai_responses.py

73 passed
```

Grid smoke artifact:

- `.bench_logs/gui_agent_dev/grid_smoke/screen_0e36a0d525ce4d4fa41030daa850d951.jpg`

## Live Doubao Workflow Retest

Service state:

- Local OpenAI-compatible service was still running on `127.0.0.1:8000`.
- Served model: `qwen35-vl`.
- GPU memory during service check: about `31.96 GiB / 32.61 GiB`.

Run 1:

- Trace: `.bench_logs\20260525_gui_agent_live\doubao_live_trace_grid_focus_1.jsonl`
- Result: `.bench_logs\20260525_gui_agent_live\doubao_live_result_grid_focus_1.json`
- Config:
  - `--screenshot-grid coarse`
  - `--focus-window-title-pattern "Doubao|豆包|Chrome|Google Chrome"`
  - `--crop-focused-window`
- Result:
  - `max_steps_exceeded`, 6 steps.
  - The crop no longer landed on pure wallpaper, but the title pattern was too
    broad. The crop included a large desktop area and a small browser area.
  - Model repeatedly tried click/type/Enter instead of moving to result
    recording.
  - Response latencies: `17.185, 16.601, 25.220, 17.327, 17.100, 24.742` s.

Run 2:

- Trace: `.bench_logs\20260525_gui_agent_live\doubao_live_trace_grid_focus_2.jsonl`
- Result: `.bench_logs\20260525_gui_agent_live\doubao_live_result_grid_focus_2.json`
- Config:
  - `--focus-window-title-pattern "豆包|Doubao"`
  - `--maximize-focused-window`
  - `--crop-focused-window`
  - `--screenshot-grid coarse`
- Result:
  - Matched title:
    `豆包 - 字节跳动旗下 AI 智能助手 ... Microsoft Edge`
  - Crop: `[-8, -8, 2576, 1408]`.
  - `max_steps_exceeded`, 4 steps.
  - Initial screenshot was correct; subsequent behavior still exposed tool-call
    formatting issues and model hesitation.
  - Response latencies: `13.576, 16.312, 20.380, 13.408` s.

## Client Detail And Image Interaction

Latest UI direction:

- Latency remains an application-layer measurement around the Responses call.
  The OpenAI Responses protocol object itself is stored separately.
- The inspector renders one selected step from:
  - the input text preview sent to the model;
  - screenshots before/after the operation;
  - the normalized Responses output object;
  - app-layer latency and token/character statistics;
  - the executed tool call detail.
- Main timeline and inspector panes both support mouse-wheel scrolling in
  addition to dragging the slim scroll thumb.
- Screenshot thumbnails in both panes open a preview window on click.
  The preview supports fit, 100%, zoom in/out, Ctrl + wheel zoom, wheel scroll,
  and drag-to-pan.
- Visual coordinate grids are no longer enabled by default:
  - desktop workflow command builder now sends `--screenshot-grid off`;
  - workflow and coordinate smoke scripts also default to `off`;
  - `coarse/fine` are retained only for explicit debugging runs.

Run 3:

- Trace: `.bench_logs\20260525_gui_agent_live\doubao_live_trace_grid_focus_3.jsonl`
- Result: `.bench_logs\20260525_gui_agent_live\doubao_live_result_grid_focus_3.json`
- Result before code fix:
  - Marked `completed`, 3 steps, but this was a false completion.
  - The model response was truncated by `max_output_tokens` in the middle of a
    `<tool_call>`.
  - Because the parser could not extract a complete tool call, runner treated
    the leftover visible text as final output.
- Code fix:
  - Runner now detects incomplete `<tool_call>`, `<function=...>`, or
    `<parameter=...>` text and sends a retry instruction instead of completing
    the task.
  - This prevents truncated tool calls from being treated as successful final
    answers.

Other fixes from the live run:

- Windows window helper now:
  - skips minimized windows;
  - skips DWM-cloaked windows;
  - supports Alt-key foreground fallback for `SetForegroundWindow`;
  - supports `--maximize-focused-window`.
- Workflow script now forces UTF-8 stdout/stderr to avoid Windows GBK failures
  when printing titles containing hidden Unicode characters.
- Qwen text tool-call parser now repairs the narrow coordinate typo:
  - from `{"type":"click","x":313,464}`
  - to `{"type":"click","x":313,"y":464}`
- Model response metrics now count text-form `<tool_call>` blocks as tool calls
  instead of reporting `function_call_count=0`.
- Step verification now detects semantic repeated text submission:
  - consecutive `click + type + Enter` submissions are treated as the same
    high-level action even if coordinates or punctuation differ;
  - after the configured threshold, the runner can move to human unblock instead
    of letting the model loop until `max_steps`.

Current conclusion:

- Correct-window visual input is now much better than the earlier wallpaper
  crop failure.
- The main remaining bottleneck in this workflow is model behavior:
  - visible English planning text still appears despite the low-latency prompt;
  - response latency is usually `13-28s`;
  - the model can hesitate or repeat UI actions after submission;
  - low `max_output_tokens` can truncate tool calls.
- Next improvements should focus on:
  - stronger response-format prompting;
  - validating the repeated-submit guard in a longer live run;
  - optionally using a deterministic workflow state machine for this specific
    input-submit-record pattern while keeping GUI Agent generic.

## Live Doubao Workflow Retest - Parser And Safety Hardening

Additional fixes after Run 3:

- Runner no longer treats prose planning as completion:
  - responses such as `Thinking Process` / `Plan: call read_text_file` are
    recorded as `tool_call_parse_retry`;
  - the model is asked to return one complete tool call instead.
- Benchmark workflow no longer double-scales computer coordinates:
  - removed the extra `CoordinateScalingBackend` wrapper;
  - `ComputerLoop` alone maps screenshot coordinates to physical screen
    coordinates.
- Protocol compatibility:
  - `key_press` is normalized to `keypress`;
  - invalid tool arguments no longer crash the runner; they are fed back to the
    model as parse retry messages;
  - bare visible `{"actions": [...]}` JSON is parsed as `computer_use`.
- Verifier hardening:
  - repeated text-submit actions are detected semantically;
  - after a text-submit, repeated click-only probing is treated as stuck UI and
    moves the task to human unblock.
- The screenshot grid overlay was made lighter so it assists coordinates
  without obscuring modal text.

Latest live validation:

- Trace: `.bench_logs\20260525_gui_agent_live\doubao_live_trace_grid_focus_11.jsonl`
- Result: `.bench_logs\20260525_gui_agent_live\doubao_live_result_grid_focus_11.json`
- Status: `waiting_human`, 4 steps.
- Observed sequence:
  - step 1: `read_text_file` succeeded;
  - step 2: model clicked the chat input, typed the question, and pressed Enter;
  - step 3/4: model repeatedly clicked the send/modal region after submission;
  - verifier marked `computer_click_only` as repeated after the text-submit
    context and runner requested human input.
- This is the desired failure mode for the current Doubao session because the
  page shows login/unlock friction after submission. The agent no longer false
  completes, crashes on malformed tool calls, or runs until `max_steps`.
- Response latencies in the latest run:
  - `15.284s`, `16.258s`, `13.572s`, `15.558s`.

Remaining issues:

- The model still sometimes emits short visible prose before tool calls.
- The workflow should eventually get a deterministic helper for the common
  input-submit-record pattern, while keeping the GUI Agent product generic.
- For real production runs, the target web app should start from an authenticated
  account state, or the first login popup should become a structured human
  request immediately.

Latest verification:

```text
GUI Agent unit suite: 79 passed
```

## Desktop-Client Integrated Run - 2026-05-25 Evening

Goal:

- Run the Doubao workflow from the desktop client itself, not by bypassing the
  client with a direct benchmark command.
- Keep the client visible while testing so UI/trace issues are discovered in
  the same path a user would operate.

Client integration added:

- Header play button and APP context-menu action `运行当前任务`.
- Background subprocess execution of
  `benchmarks/run_gui_agent_workflow_responses.py`.
- Per-run artifacts:
  - `.client_run_YYYYMMDD_HHMMSS.log`
  - `.client_run_YYYYMMDD_HHMMSS.json`
- Client reloads the trace after the subprocess exits and records a visible
  operation card for the run result.
- Initial window geometry now fits and centers inside the current screen instead
  of letting the run/inspector buttons drift offscreen.

Failures found through the client path:

1. API server process had died while the EngineCore child was still around.
   Restarting the server via PowerShell `Start-Process -ArgumentList` broke JSON
   quoting for `--speculative-config`, so the reliable launcher is now a Python
   `subprocess.Popen(args_list)` command.
2. Windows command-line forwarding through the current Python shim can degrade
   Chinese window-title regexes such as `豆包|Doubao` into `??|Doubao`.
   `??|Doubao` is an invalid regex and previously crashed window focus.
   Fix:
   - client command construction sanitizes invalid regex alternatives;
   - `cfie_client.executor.windows.find_visible_window()` catches `re.error`
     and returns no match instead of crashing.
3. The client defaulted screenshots to `file://` URLs, but the OpenAI server was
   not started with `--allowed-local-media-path`.
   Fix:
   - desktop-client launched workflow commands now default to
     `--screenshot-url-mode data`;
   - file mode remains available for services explicitly launched with local
     media permission.
4. Qwen3.5 sometimes emits executable tools as
   `<tool_code>print(read_text_file(...))</tool_code>` instead of native
   Responses `function_call` items or `<tool_call><function=...>`.
   Fix:
   - parser now supports Qwen `<tool_code>` Python-call style;
   - supports normal keyword arguments and `computer_use(actions=[...])`;
   - model-response metrics count `<tool_code>` as text tool calls.
5. A run can return process exit code 0 while the GUI task is actually
   `waiting_human`.
   Fix:
   - runner result metadata now includes pending/completed human requests;
   - desktop client reads `result.status` from the result JSON;
   - `waiting_human` is displayed as `Agent 等待人工`, not `Agent 执行完成`;
   - if the result lacks serialized human requests, the client creates a
     fallback local human request so the composer can unblock the job.

Live client run:

- Trace:
  `.bench_logs\20260525_gui_agent_live\doubao_live_trace_client_7.jsonl`
- Result:
  `.bench_logs\20260525_gui_agent_live\doubao_live_trace_client_7.client_run_20260525_171912.json`
- Final task status:
  `waiting_human`
- Steps:
  1. `read_text_file` accepted;
  2. `computer_use`: click input, type `请用一句话回答：2 + 3 等于几？`, press Enter;
  3. `computer_use`: click probing;
  4. invalid coordinate was rejected by harness;
  5. repeated click led to human unblock.
- Latencies:
  - step 1: `76.221s`
  - step 2: `16.461s`
  - step 3: `10.476s`
  - step 4: `12.903s`
  - step 5: `16.664s`

Interpretation:

- Client-to-model-to-tool-to-trace loop is now functional.
- The current Doubao page still contains login/unlock friction, so
  `waiting_human` is the correct product behavior.
- Windows VL prefill remains too slow for a responsive GUI Agent loop. The first
  step is especially expensive; later turns benefit from prefix/MM cache but are
  still usually above the 10s interaction target on Windows.

Verification after fixes:

```text
tests/unit/test_cfie_client_gui_agent.py
tests/unit/test_gui_agent_architecture.py
tests/unit/test_gui_agent_desktop_client.py
tests/unit/test_gui_agent_workflow.py
tests/unit/test_gui_agent_openai_responses.py

86 passed
```

## Coordinate Protocol Fix

Problem found in live Doubao runs:

- Qwen3.5-VL can recognize the input box and send control semantically, but its
  mouse coordinates are not consistently raw screenshot pixels.
- A controlled localization prompt on a `960x524` screenshot returned roughly
  `(700, 950)` for the send button. This is not valid pixel `y`, but it maps
  correctly when interpreted as Qwen-style `0..1000` normalized image
  coordinates.
- The previous `auto` mode can only be a compatibility fallback because
  coordinates inside both ranges are ambiguous.

Protocol decision:

- `computer_use` now has an explicit required `coordinate_space` argument.
- Qwen GUI workflows must use:

```json
{
  "coordinate_space": "qwen_normalized_1000",
  "actions": [
    {"type": "click", "x": 700, "y": 950}
  ]
}
```

- `(0,0)` means the current model image top-left.
- `(1000,1000)` means the current model image bottom-right.
- The harness converts normalized coordinates to screenshot pixels, then to the
  physical desktop or cropped APP viewport.
- `screenshot` remains available for non-Qwen models or internal tools that
  already produce pixel coordinates.
- `auto` remains only as a defensive compatibility path in `ComputerLoop`, not
  as the product protocol.

Related implementation changes:

- `ComputerCall` carries `coordinate_space`.
- `find_computer_tool_calls()` preserves the tool-level coordinate space.
- `ComputerLoop` lets the call-level coordinate space override the loop
  default.
- `run_gui_agent_workflow_responses.py` sets the Qwen workflow loop default to
  `qwen_normalized_1000`.
- The tool declaration and Qwen workflow prompt both require
  `coordinate_space="qwen_normalized_1000"`.
- Parser repair now also handles Qwen malformed coordinates like
  `{"x":549,460,"button":"left"}`.

Verification:

```text
tests/unit/test_cfie_client_gui_agent.py
tests/unit/test_gui_agent_architecture.py
tests/unit/test_gui_agent_desktop_client.py
tests/unit/test_gui_agent_workflow.py
tests/unit/test_gui_agent_openai_responses.py

92 passed
```

Follow-up verification after action-level coordinate repair:

- Qwen sometimes places `coordinate_space` inside each action instead of as a
  top-level tool argument. The parser now accepts the action-level form when all
  actions agree on the same coordinate space.
- Added a regression test for:

```json
{
  "actions": [
    {
      "type": "click",
      "x": 500,
      "y": 900,
      "coordinate_space": "qwen_normalized_1000"
    }
  ]
}
```

Unit verification:

```text
tests/unit/test_cfie_client_gui_agent.py
tests/unit/test_gui_agent_architecture.py
tests/unit/test_gui_agent_desktop_client.py
tests/unit/test_gui_agent_workflow.py
tests/unit/test_gui_agent_openai_responses.py

93 passed
```

Controlled live coordinate smoke:

1. Card layout target button
   - Command:
     `benchmarks/run_gui_agent_coordinate_smoke.py`
   - Artifact:
     `.bench_logs\20260525_coordinate_smoke\result.json`
   - Screenshot size:
     `960x524`
   - Model coordinate:
     `qwen_normalized_1000 (387,258)`
   - Mapped screenshot coordinate:
     `(372,135)`
   - Mapped physical coordinate:
     `(990,355)`
   - Result:
     local page recorded the button click.

2. Bottom composer send button
   - Command:
     `benchmarks/run_gui_agent_coordinate_smoke.py --layout bottom_composer`
   - Artifact:
     `.bench_logs\20260525_coordinate_smoke_bottom\result.json`
   - Screenshot size:
     `960x524`
   - Model coordinate:
     `qwen_normalized_1000 (638,925)`
   - Mapped screenshot coordinate:
     `(612,485)`
   - Mapped physical coordinate:
     `(1634,1295)`
   - Result:
     local page recorded the send-button click.

Conclusion:

- The coordinate protocol and harness mapping path are now correct in controlled
  desktop tests.
- The remaining Doubao workflow failures are mainly task-following and
  page-state issues: the model can repeat clicks or skip required workflow
  tools. They should be handled through workflow prompt/tool discipline and
  verifier feedback, not by changing coordinate scaling again.
