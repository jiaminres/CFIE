# 2026-05-22 Responses Video and GUI Testing Harness

## Scope

- Add CFIE Responses input support for video content parts.
- Review and repair `cfie_client` as a generic computer-use harness.
- Keep `cfie_gui_agent` as the application-level GUI automation agent layer.

## Protocol Change

CFIE now accepts a Responses input content part:

```json
{
  "type": "input_video",
  "video_url": "file:///D:/videos/gui_trace.mp4"
}
```

The request validator accepts `input_video`, and `chat_utils` maps it to the existing internal `video` multimodal modality. Qwen3.5-VL therefore receives video through the native `pixel_values_videos` / `video_grid_thw` path, not as an ad hoc list of images.

Supported `video_url` forms:

- HTTP URL
- `file://...` URL, subject to `--allowed-local-media-path`
- `data:video/...;base64,...`

`input_video.file_id` is intentionally rejected at parsing time for now, because this project does not yet implement an OpenAI file store.

## cfie_client Changes

- Fixed single-action `actions: {...}` handling.
- Added `Qwen35ComputerAdapter` for Qwen text JSON outputs. It accepts fenced JSON, `action` aliases, and `coordinate: [x, y]`, then converts them into `computer_call`.
- Added `wait.seconds` / `wait.duration` / `wait.timeout` parsing.
- Changed default wait execution from fixed 2.0s to 1.0s when no duration is supplied.
- Added `CTRL+L` style key-chord normalization.
- Added SafetyGate limits for typed text length, scroll delta, and wait duration.
- Switched Windows screen size query to `GetSystemMetrics` instead of taking an extra screenshot.
- Added video suffix handling in `TraceStore` data URL artifacts.
- Rebuilt Windows Unicode keyboard input around the full `SendInput` structure and return-value checks.
- Added a small default action delay in `ComputerExecutor` so real desktop UI has time to process click/type events before the next action.
- Rewrote the mojibake README with the correct package boundary: protocol harness, not business app.

## cfie_gui_agent Changes

- Added `GuiAgentRunner`.
- `GuiAgentRunner` builds the initial Responses input with `input_text` + `input_image`, calls an injected agent function, executes any returned `computer_call` through `cfie_client`, and stops when the model returns a final message.
- Extended `GuiAgentResult` with `final_text` and `steps`.
- Rewrote README to describe the app-layer boundary.

## Verification

Commands:

```powershell
..\.venv\Scripts\python.exe -m pytest tests\unit\test_responses_video_input.py tests\unit\test_cfie_client_gui_agent.py -q
..\.venv\Scripts\python.exe -m py_compile cfie\entrypoints\chat_utils.py cfie\entrypoints\openai\responses\protocol.py cfie_client\__init__.py cfie_client\protocol.py cfie_client\loop.py cfie_client\screen.py cfie_client\safety.py cfie_client\trace.py cfie_client\executor\computer.py cfie_client\executor\keyboard.py cfie_client\executor\mouse.py cfie_client\executor\scroll.py cfie_client\executor\wait.py cfie_client\executor\windows.py cfie_gui_agent\__init__.py cfie_gui_agent\specs.py cfie_gui_agent\runner.py
```

Result:

```text
7 passed
py_compile passed
git diff --check passed
```

Additional local-file video smoke test:

- Generated `.tmp/video_response_smoke/gui_smoke.mp4`.
- Loaded it through `MediaConnector.fetch_video(file://...)` with `allowed_local_media_path`.
- Result shape: `(2, 64, 64, 3)`.

## Desktop Computer-Use Smoke

Benchmark helper:

```powershell
..\.venv\Scripts\python.exe benchmarks\run_responses_computer_use_smoke.py `
  --base-url http://127.0.0.1:8000/v1 `
  --model qwen35-vl `
  --artifact-dir .bench_logs\20260522_responses_computer_use\model_run_8_large_button `
  --open-browser `
  --crop-right-half `
  --execute `
  --timeout 900 `
  --max-output-tokens 192 `
  --visual-max-width 960 `
  --video-frames 2
```

Server configuration used for this smoke:

- Qwen3.5-122B-A10B-GPTQ-Int4 from `D:\models\Qwen3.5-122B-A10B-GPTQ-Int4\snapshots\5b9f0050d3ec98b0c81a7716776533c5eacebb64`
- `max_model_len=8192`
- `max_num_seqs=1`
- `max_num_batched_tokens=4096`
- `kv_cache_memory_bytes=1000000000`
- `gpu_slots_per_layer=16`
- `prefill_burst_slots=256`
- `prepare_cpu_copy_threads=32`
- `cpu_static_pinned_gb=40`
- `enable_prefix_caching`
- `marlin_input_dtype=fp8`
- `enforce_eager`

Result:

- Responses status: `200`
- Model latency: `32.88s`
- Input tokens: `2183`
- Output tokens: `105`
- Model output:

```json
{
  "actions": [
    {"action": "click", "coordinate": [499, 245]},
    {"action": "type", "text": "hello cfie"},
    {"action": "click", "coordinate": [832, 320]}
  ]
}
```

Execution result:

- Visual crop scale: `0.75`
- Visual crop offset: `[1280, 0]`
- Scaled actions:
  - click `(1945, 327)`
  - type `hello cfie`
  - click `(2389, 427)`
- Final screenshot showed `submitted: hello cfie`.

Important findings:

- The first real desktop run proved `input_video` and `input_image` reached Qwen3.5-VL and produced usable GUI action JSON.
- The original keyboard backend did not reliably type into Edge because `SendInput` used an incomplete Windows input structure and did not check failure.
- Without a small per-action delay, click/type/click could finish before the browser processed keyboard events, causing screenshots to miss typed text.
- The model's approximate button y-coordinate was about 20-30px high on the original narrow button. The controlled smoke page now uses a larger button so the protocol and tool execution path is tested without being dominated by a one-pixel-edge coordinate miss.
