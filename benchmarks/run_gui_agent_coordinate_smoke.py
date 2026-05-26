from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cfie_client import ComputerLoop, ScaledPillowScreenCapture, create_default_backend
from cfie_client.executor.windows import focus_window_by_title
from cfie_gui_agent import ModelToolRegistry, OpenAIResponsesAgent, find_computer_tool_calls


CARD_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CFIE Coordinate Smoke</title>
  <style>
    body {
      margin: 0;
      background: #f7f3ee;
      color: #111827;
      font-family: Arial, sans-serif;
    }
    main {
      width: 760px;
      margin: 88px auto;
      padding: 36px;
      border-radius: 22px;
      background: #fffaf4;
      box-shadow: 0 20px 80px rgba(17, 24, 39, 0.18);
    }
    h1 {
      margin: 0 0 12px;
      font-size: 28px;
    }
    p {
      margin: 0 0 28px;
      font-size: 16px;
      color: #475569;
    }
    .row {
      display: flex;
      align-items: center;
      gap: 18px;
    }
    button {
      width: 250px;
      height: 76px;
      border: 0;
      border-radius: 18px;
      background: #155eef;
      color: white;
      font-size: 24px;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 12px 30px rgba(21, 94, 239, 0.32);
    }
    #status {
      min-width: 180px;
      padding: 20px 22px;
      border-radius: 16px;
      background: #eef6ff;
      color: #0f172a;
      font-size: 20px;
      font-weight: 700;
    }
  </style>
</head>
<body>
  <main>
    <h1>Coordinate accuracy check</h1>
    <p>Click the large blue button labeled TARGET SEND.</p>
    <div class="row">
      <button id="target" onclick="markClicked()">TARGET SEND</button>
      <div id="status">waiting</div>
    </div>
  </main>
  <script>
    async function markClicked() {
      document.getElementById('status').textContent = 'CLICKED OK';
      await fetch('/clicked', {method: 'POST'});
    }
  </script>
</body>
</html>
"""

BOTTOM_COMPOSER_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CFIE Coordinate Smoke</title>
  <style>
    body {
      margin: 0;
      background: #fbfaf8;
      color: #111827;
      font-family: Arial, sans-serif;
    }
    .content {
      width: 820px;
      margin: 70px auto 0;
      font-size: 20px;
      line-height: 1.6;
    }
    .composer {
      position: fixed;
      left: 50%;
      bottom: 42px;
      transform: translateX(-50%);
      width: min(880px, calc(100vw - 120px));
      height: 96px;
      border: 1px solid #e7dfd6;
      border-radius: 24px;
      background: #ffffff;
      box-shadow: 0 18px 60px rgba(15, 23, 42, 0.16);
      display: flex;
      align-items: center;
      padding: 0 18px 0 24px;
      gap: 16px;
    }
    .input {
      flex: 1;
      color: #64748b;
      font-size: 20px;
    }
    button {
      width: 128px;
      height: 56px;
      border: 0;
      border-radius: 18px;
      background: #111827;
      color: white;
      font-size: 19px;
      font-weight: 700;
      cursor: pointer;
    }
    #status {
      margin-top: 28px;
      font-weight: 700;
      color: #0f766e;
    }
  </style>
</head>
<body>
  <div class="content">
    <h1>Chat style coordinate check</h1>
    <p>The target is the dark SEND button in the bottom composer.</p>
    <div id="status">waiting</div>
  </div>
  <div class="composer">
    <div class="input">Type a message...</div>
    <button id="target" onclick="markClicked()">SEND</button>
  </div>
  <script>
    async function markClicked() {
      document.getElementById('status').textContent = 'CLICKED OK';
      await fetch('/clicked', {method: 'POST'});
    }
  </script>
</body>
</html>
"""


class CoordinateServer(BaseHTTPRequestHandler):
    clicked_count = 0
    page_html = CARD_HTML

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/status":
            self._send_json({"clicked_count": CoordinateServer.clicked_count})
            return
        body = CoordinateServer.page_html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path == "/clicked":
            CoordinateServer.clicked_count += 1
            self._send_json({"ok": True, "clicked_count": CoordinateServer.clicked_count})
            return
        self.send_error(404)

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GUI Agent coordinate accuracy smoke.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35-vl")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--screenshot-max-width", type=int, default=1920)
    parser.add_argument("--screenshot-max-height", type=int, default=1080)
    parser.add_argument("--screenshot-grid", choices=("off", "coarse", "fine"), default="off")
    parser.add_argument(
        "--layout",
        choices=("card", "bottom_composer"),
        default="card",
    )
    parser.add_argument("--artifact-dir", default=".bench_logs/coordinate_smoke")
    parser.add_argument("--result-json", default=None)
    parser.add_argument(
        "--browser-path",
        default=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    CoordinateServer.clicked_count = 0
    CoordinateServer.page_html = (
        BOTTOM_COMPOSER_HTML if args.layout == "bottom_composer" else CARD_HTML
    )

    server = ThreadingHTTPServer(("127.0.0.1", args.port), CoordinateServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    target_url = f"http://127.0.0.1:{args.port}/"
    browser = Path(args.browser_path)
    if browser.exists():
        subprocess.Popen([str(browser), "--new-window", target_url])
    else:
        import webbrowser

        webbrowser.open(target_url)
    time.sleep(2.5)

    window = focus_window_by_title("CFIE Coordinate Smoke", maximize=True)
    if window is None:
        raise RuntimeError("failed to focus coordinate smoke browser window")
    time.sleep(0.5)

    screen = ScaledPillowScreenCapture(
        max_width=args.screenshot_max_width,
        max_height=args.screenshot_max_height,
        jpeg_quality=90,
        url_mode="data",
        crop_box=window.crop_box,
        output_dir=artifact_dir,
        filename_prefix="coordinate_smoke",
        grid_overlay=args.screenshot_grid,
    )
    screenshot = screen.screenshot()
    initial_screenshot_path = artifact_dir / "initial_screenshot.jpg"
    _write_data_url_image(screenshot.image_url, initial_screenshot_path)

    agent = OpenAIResponsesAgent(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        tool_registry=ModelToolRegistry(allowed_tool_names=("computer_use",)),
        max_output_tokens=args.max_output_tokens,
        timeout=600,
        reasoning_effort="none",
        chat_template_kwargs={"enable_thinking": False},
    )
    conversation = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Look at this screenshot and click the large blue button "
                        "labeled TARGET SEND or the SEND button in the bottom "
                        "composer. Return exactly one computer_use tool "
                        "call and no prose. The tool call must include top-level "
                        "coordinate_space=\"qwen_normalized_1000\". Coordinates "
                        "must be normalized on a 0..1000 image grid where (0,0) "
                        "is the screenshot top-left and (1000,1000) is the "
                        "screenshot bottom-right."
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": screenshot.image_url,
                    "detail": "low",
                },
            ],
        }
    ]
    started = time.perf_counter()
    response = agent(conversation)
    latency = time.perf_counter() - started
    calls = find_computer_tool_calls(response)
    output_text = _response_text(response)
    result: dict[str, Any] = {
        "ok": False,
        "model_latency_seconds": latency,
        "screenshot_size": [screenshot.width, screenshot.height],
        "physical_crop": list(window.crop_box),
        "initial_screenshot": str(initial_screenshot_path),
        "response_output_text": output_text,
        "call_count": len(calls),
    }
    if calls:
        call = calls[0]
        result["model_call"] = call.to_openai_dict()
        result["mapped_preview"] = _mapped_preview(
            call.to_openai_dict(),
            screenshot_size=(screenshot.width, screenshot.height),
            physical_crop=window.crop_box,
        )
        loop = ComputerLoop(
            backend=create_default_backend(),
            screen=screen,
            model_coordinate_mode="qwen_normalized_1000",
            trace_artifact_dir=artifact_dir / "trace_artifacts",
            trace_path=artifact_dir / "trace.jsonl",
        )
        loop.handle_call(call)
        time.sleep(1.0)
        result["clicked_count"] = CoordinateServer.clicked_count
        result["ok"] = CoordinateServer.clicked_count > 0
    else:
        result["response"] = response

    result_path = Path(args.result_json) if args.result_json else artifact_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    server.shutdown()


def _write_data_url_image(image_url: str, path: Path) -> None:
    if not image_url.startswith("data:"):
        return
    _, encoded = image_url.split(",", 1)
    path.write_bytes(base64.b64decode(encoded))


def _response_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in response.get("output") or ():
        for content in item.get("content") or ():
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _mapped_preview(
    call: dict[str, Any],
    *,
    screenshot_size: tuple[int, int],
    physical_crop: tuple[int, int, int, int],
) -> list[dict[str, Any]]:
    width, height = screenshot_size
    origin_x, origin_y, physical_width, physical_height = physical_crop
    preview: list[dict[str, Any]] = []
    for action in call.get("actions") or ():
        if "x" not in action or "y" not in action:
            continue
        normalized_x = int(action["x"])
        normalized_y = int(action["y"])
        screenshot_x = max(0, min(width - 1, int(round(normalized_x * width / 1000))))
        screenshot_y = max(0, min(height - 1, int(round(normalized_y * height / 1000))))
        physical_x = int(round(screenshot_x * physical_width / width)) + origin_x
        physical_y = int(round(screenshot_y * physical_height / height)) + origin_y
        preview.append(
            {
                "type": action.get("type"),
                "normalized": [normalized_x, normalized_y],
                "screenshot": [screenshot_x, screenshot_y],
                "physical": [physical_x, physical_y],
            }
        )
    return preview


if __name__ == "__main__":
    main()
