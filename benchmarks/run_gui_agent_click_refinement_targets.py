from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cfie_client import ComputerLoop, ScaledPillowScreenCapture, create_default_backend
from cfie_client.executor.windows import focus_window_by_title
from cfie_gui_agent import (
    AgentTraceStore,
    GuiAgentRunner,
    GuiAgentTaskSpec,
    ModelToolRegistry,
    OpenAIResponsesAgent,
)


PAGE_TITLE_PREFIX = "CFIE Tiny Target Click Test"
HWND_NOTOPMOST = -2
HWND_TOPMOST = -1
WM_CLOSE = 0x0010
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
DEFAULT_TARGETS = (
    ("A", 18, 24, "#ef4444"),
    ("B", 72, 22, "#0ea5e9"),
    ("C", 46, 42, "#22c55e"),
    ("D", 27, 71, "#f59e0b"),
    ("E", 82, 74, "#8b5cf6"),
)


@dataclass(slots=True, frozen=True)
class TargetSpec:
    target_id: str
    x_pct: int
    y_pct: int
    color: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.target_id,
            "xPct": self.x_pct,
            "yPct": self.y_pct,
            "color": self.color,
        }


class TinyTargetServer(BaseHTTPRequestHandler):
    targets: tuple[TargetSpec, ...] = ()
    target_size_px: int = 18
    page_title: str = PAGE_TITLE_PREFIX
    clicked_ids: set[str] = set()
    click_log: list[dict[str, Any]] = []

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/status":
            self._send_json(_status_payload())
            return
        if self.path == "/":
            body = _page_html(
                targets=TinyTargetServer.targets,
                target_size_px=TinyTargetServer.target_size_px,
                page_title=TinyTargetServer.page_title,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/click":
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {"raw": raw}
            hit_id = str(payload.get("hit_id") or "").strip()
            if hit_id:
                TinyTargetServer.clicked_ids.add(hit_id)
            TinyTargetServer.click_log.append(
                {
                    "x": payload.get("x"),
                    "y": payload.get("y"),
                    "hit_id": hit_id or None,
                    "nearest_id": payload.get("nearest_id"),
                    "nearest_distance_px": payload.get("nearest_distance_px"),
                    "clicked_after": sorted(TinyTargetServer.clicked_ids),
                }
            )
            self._send_json(_status_payload())
            return
        if self.path == "/reset":
            TinyTargetServer.clicked_ids.clear()
            TinyTargetServer.click_log.clear()
            self._send_json(_status_payload())
            return
        self.send_error(404)

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a GUI Agent tiny-target click benchmark that stresses local "
            "click refinement crops."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35-vl")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--target-count", type=int, default=5)
    parser.add_argument("--target-size-px", type=int, default=18)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-output-tokens", type=int, default=768)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--screenshot-max-width", type=int, default=1920)
    parser.add_argument("--screenshot-max-height", type=int, default=1080)
    parser.add_argument("--screenshot-detail", default="low")
    parser.add_argument("--screenshot-grid", choices=("off", "coarse", "fine"), default="off")
    parser.add_argument("--local-click-refinement-radius", type=int, default=120)
    parser.add_argument(
        "--no-local-click-refinement-upscale",
        action="store_true",
        help="Send the raw local crop size instead of upscaling it for the model.",
    )
    parser.add_argument("--local-click-refinement-max-size", type=int, default=1080)
    parser.add_argument(
        "--local-click-refinement-resample",
        choices=("nearest", "bicubic", "bilinear", "lanczos"),
        default="nearest",
    )
    parser.add_argument(
        "--local-click-refinement-center-marker",
        action="store_true",
        help=(
            "Draw magenta center brackets on local click refinement crops. "
            "This is off by default because the benchmark is testing whether "
            "the model can re-locate the target from the local crop itself."
        ),
    )
    parser.add_argument(
        "--local-click-refinement-image-format",
        choices=("PNG", "JPEG", "png", "jpeg", "jpg"),
        default="PNG",
        help="Encode local refinement crops separately from full screenshots.",
    )
    parser.add_argument("--artifact-dir", default=".bench_logs/click_refinement_targets")
    parser.add_argument("--browser-mode", choices=("app", "window"), default="app")
    parser.add_argument("--no-topmost", action="store_true")
    parser.add_argument(
        "--browser-path",
        default=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    targets = _target_specs(args.target_count)
    page_title = f"{PAGE_TITLE_PREFIX} {int(time.time())}"
    TinyTargetServer.targets = targets
    TinyTargetServer.target_size_px = max(1, int(args.target_size_px))
    TinyTargetServer.page_title = page_title
    TinyTargetServer.clicked_ids.clear()
    TinyTargetServer.click_log.clear()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), TinyTargetServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    target_url = f"http://127.0.0.1:{args.port}/"
    window = None
    try:
        _open_browser(target_url, Path(args.browser_path), mode=args.browser_mode)
        window = _focus_window_or_raise(page_title)
        if not args.no_topmost:
            _set_window_topmost(window.hwnd, topmost=True)
        screen = ScaledPillowScreenCapture(
            max_width=args.screenshot_max_width,
            max_height=args.screenshot_max_height,
            jpeg_quality=92,
            url_mode="data",
            crop_box=window.crop_box,
            output_dir=artifact_dir / "screens",
            filename_prefix="tiny_target",
            grid_overlay=args.screenshot_grid,
            draw_cursor=True,
        )
        computer_trace_path = artifact_dir / "computer_trace.jsonl"
        agent_trace_path = artifact_dir / "agent_trace.jsonl"
        runner = GuiAgentRunner(
            computer_loop=ComputerLoop(
                backend=create_default_backend(),
                screen=screen,
                trace_path=computer_trace_path,
                trace_artifact_dir=artifact_dir / "computer_artifacts",
                screenshot_detail=args.screenshot_detail,
                model_coordinate_mode="qwen_normalized_1000",
            ),
            tool_registry=ModelToolRegistry(
                allowed_tool_names=("computer_use", "finish_subtask")
            ),
            trace_store=AgentTraceStore(path=agent_trace_path),
            image_detail=args.screenshot_detail,
            max_steps=args.max_steps,
            auto_human_repeated_action_threshold=0,
            require_finish_tool_for_completion=True,
            guard_untrusted_clicks=True,
            response_latency_warning_seconds=30.0,
            response_text_warning_chars=1200,
            local_click_refinement_radius=max(
                24,
                int(args.local_click_refinement_radius),
            ),
            local_click_refinement_upscale=not args.no_local_click_refinement_upscale,
            local_click_refinement_max_size=args.local_click_refinement_max_size,
            local_click_refinement_resample=args.local_click_refinement_resample,
            local_click_refinement_image_format=(
                "JPEG"
                if str(args.local_click_refinement_image_format).lower() == "jpg"
                else str(args.local_click_refinement_image_format).upper()
            ),
            local_click_refinement_draw_center_marker=(
                args.local_click_refinement_center_marker
            ),
        )
        agent = OpenAIResponsesAgent(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            tool_registry=runner.tool_registry,
            max_output_tokens=args.max_output_tokens,
            timeout=900,
            reasoning_effort=args.reasoning_effort,
            chat_template_kwargs={"enable_thinking": bool(args.enable_thinking)},
        )
        task = GuiAgentTaskSpec(
            task_id=f"click_refinement_{int(time.time())}",
            target_app=page_title,
            instruction=_task_instruction(targets, target_url),
            expected_outcome=(
                "All tiny targets are clicked exactly once, and the task ends "
                "with finish_subtask."
            ),
            metadata={
                "app_id": "click_refinement_targets",
                "target_url": target_url,
                "target_count": len(targets),
                "target_size_px": TinyTargetServer.target_size_px,
            },
        )
        started = time.perf_counter()
        result = runner.run_task(task, agent)
        elapsed = time.perf_counter() - started
        status = _status_payload()
        summary = _build_summary(
            result=result.to_trace_payload(),
            status=status,
            targets=targets,
            elapsed_seconds=elapsed,
            artifact_dir=artifact_dir,
            agent_trace_path=agent_trace_path,
            computer_trace_path=computer_trace_path,
        )
        result_path = artifact_dir / "result.json"
        result_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if window is not None and not args.no_topmost:
            _set_window_topmost(window.hwnd, topmost=False)
        if window is not None:
            _close_window(window.hwnd)
        server.shutdown()


def _target_specs(count: int) -> tuple[TargetSpec, ...]:
    count = max(1, min(24, int(count)))
    specs = [
        TargetSpec(target_id, x_pct, y_pct, color)
        for target_id, x_pct, y_pct, color in DEFAULT_TARGETS[:count]
    ]
    if len(specs) >= count:
        return tuple(specs)
    colors = ("#ef4444", "#0ea5e9", "#22c55e", "#f59e0b", "#8b5cf6", "#14b8a6")
    index = len(specs)
    for row in range(3):
        for col in range(6):
            if len(specs) >= count:
                return tuple(specs)
            specs.append(
                TargetSpec(
                    chr(ord("A") + index),
                    14 + col * 14,
                    20 + row * 24,
                    colors[index % len(colors)],
                )
            )
            index += 1
    return tuple(specs)


def _page_html(
    *,
    targets: tuple[TargetSpec, ...],
    target_size_px: int,
    page_title: str,
) -> str:
    target_json = json.dumps([target.to_dict() for target in targets], ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN" translate="no">
<head>
  <meta charset="utf-8">
  <meta name="google" content="notranslate">
  <title>{page_title}</title>
  <style>
    :root {{
      --target-size: {target_size_px}px;
      --target-radius: {target_size_px / 2:.1f}px;
      color-scheme: light;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at 18% 12%, #fff7ed 0, transparent 26%),
        linear-gradient(135deg, #f8fafc 0%, #fff7ed 100%);
      color: #111827;
      font-family: "Segoe UI", Arial, sans-serif;
      min-height: 100vh;
      overflow: hidden;
    }}
    header {{
      position: absolute;
      left: 48px;
      top: 36px;
      width: 720px;
      padding: 24px 28px;
      border: 1px solid rgba(17, 24, 39, 0.08);
      border-radius: 24px;
      background: rgba(255, 255, 255, 0.84);
      box-shadow: 0 20px 70px rgba(15, 23, 42, 0.10);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.2;
    }}
    p {{
      margin: 0;
      color: #475569;
      font-size: 16px;
      line-height: 1.55;
    }}
    #board {{
      position: absolute;
      inset: 146px 48px 48px 48px;
      border: 1px solid rgba(17, 24, 39, 0.08);
      border-radius: 28px;
      background: rgba(255, 255, 255, 0.74);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.7);
    }}
    .target-dot {{
      position: absolute;
      width: var(--target-size);
      height: var(--target-size);
      margin-left: calc(var(--target-size) / -2);
      margin-top: calc(var(--target-size) / -2);
      border-radius: 999px;
      border: 2px solid #111827;
      background: var(--target-color);
      box-shadow: 0 0 0 7px rgba(255,255,255,0.95),
                  0 10px 22px rgba(15, 23, 42, 0.22);
    }}
    .target-dot.done {{
      background: #16a34a;
      border-color: #14532d;
      box-shadow: 0 0 0 8px rgba(220, 252, 231, 0.95),
                  0 10px 22px rgba(22, 163, 74, 0.22);
    }}
    .target-label {{
      position: absolute;
      transform: translate(18px, -52%);
      padding: 5px 10px;
      border-radius: 999px;
      background: #111827;
      color: white;
      font-size: 15px;
      font-weight: 700;
      pointer-events: none;
      white-space: nowrap;
    }}
    .target-label.done {{
      background: #dcfce7;
      color: #14532d;
    }}
    #status {{
      position: absolute;
      right: 48px;
      top: 36px;
      min-width: 280px;
      padding: 22px 24px;
      border-radius: 24px;
      background: #111827;
      color: white;
      font-size: 18px;
      line-height: 1.6;
      box-shadow: 0 20px 70px rgba(15, 23, 42, 0.22);
    }}
  </style>
</head>
<body>
  <header>
    <h1>小目标点击测试</h1>
    <p>只点击彩色小圆点 A、B、C、D、E。旁边的文字标签不是目标。若点击偏移，请根据 harness 返回的局部高清图修正下一次点击。</p>
  </header>
  <section id="board"></section>
  <aside id="status">Clicked: <span id="clickedText">none</span><br>Remaining: <span id="remainingText"></span></aside>
  <script>
    const targets = {target_json};
    const targetSize = {target_size_px};
    const radius = targetSize / 2;
    const clicked = new Set();
    const board = document.getElementById("board");

    function renderTargets() {{
      board.innerHTML = "";
      for (const target of targets) {{
        const dot = document.createElement("div");
        dot.className = "target-dot" + (clicked.has(target.id) ? " done" : "");
        dot.dataset.targetId = target.id;
        dot.style.left = target.xPct + "%";
        dot.style.top = target.yPct + "%";
        dot.style.setProperty("--target-color", target.color);
        board.appendChild(dot);

        const label = document.createElement("div");
        label.className = "target-label" + (clicked.has(target.id) ? " done" : "");
        label.textContent = clicked.has(target.id) ? target.id + " done" : "目标 " + target.id;
        label.style.left = target.xPct + "%";
        label.style.top = target.yPct + "%";
        board.appendChild(label);
      }}
      const clickedIds = [...clicked].sort();
      const remaining = targets.map(t => t.id).filter(id => !clicked.has(id));
      document.getElementById("clickedText").textContent = clickedIds.length ? clickedIds.join(", ") : "none";
      document.getElementById("remainingText").textContent = remaining.length ? remaining.join(", ") : "none";
    }}

    function targetCenter(target) {{
      const rect = board.getBoundingClientRect();
      return {{
        x: rect.left + rect.width * target.xPct / 100,
        y: rect.top + rect.height * target.yPct / 100
      }};
    }}

    document.addEventListener("click", async (event) => {{
      const x = event.clientX;
      const y = event.clientY;
      let hit = null;
      let nearest = null;
      let nearestDistance = 1e9;
      for (const target of targets) {{
        const center = targetCenter(target);
        const distance = Math.hypot(x - center.x, y - center.y);
        if (distance < nearestDistance) {{
          nearest = target;
          nearestDistance = distance;
        }}
        if (!clicked.has(target.id) && distance <= radius) {{
          hit = target;
        }}
      }}
      if (hit) {{
        clicked.add(hit.id);
        renderTargets();
      }}
      await fetch("/click", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{
          x,
          y,
          hit_id: hit ? hit.id : null,
          nearest_id: nearest ? nearest.id : null,
          nearest_distance_px: Math.round(nearestDistance * 10) / 10,
          clicked_ids: [...clicked].sort()
        }})
      }});
    }}, true);

    renderTargets();
  </script>
</body>
</html>"""


def _task_instruction(targets: tuple[TargetSpec, ...], target_url: str) -> str:
    ids = ", ".join(target.target_id for target in targets)
    return (
        f"当前浏览器已经打开本地点击靶场：{target_url}\n"
        f"目标：依次点击页面中的小圆点目标 {ids}，每个目标只点一次。"
        "小圆点本身才是目标，旁边的 target A/B/C 标签不是目标。\n\n"
        "执行规则：\n"
        "1. 每轮优先返回一个 computer_use 工具调用；如果要连续点击多个目标，"
        "每个 action 必须写 index；每个 action 都必须写 intent，例如“点击目标 A 的红色小圆点”。\n"
        "2. 默认使用 coordinate_space=\"qwen_normalized_1000\"，坐标是当前截图"
        "0..1000 归一化坐标。\n"
        "3. 每次 click 后，harness 会返回操作后截图，并给出该点击附近的局部高清图。"
        "如果目标未变成 done 或绿色，下一轮必须检查局部图；若需要基于局部图修正点击，"
        "使用 coordinate_space=\"local_refinement_1000\"，x/y 是局部图 0..1000 坐标，"
        "不要重复上一次全图坐标。局部高清图不绘制鼠标光标；请把局部图当作新的完整图，"
        "重新选择操作意图对应目标的中心点，不要默认点击图像中心。\n"
        "4. 全部目标都显示 done 或状态栏 remaining 为 none 后，调用 finish_subtask。"
    )


def _open_browser(target_url: str, browser: Path, *, mode: str) -> None:
    if browser.exists():
        if mode == "app":
            subprocess.Popen([str(browser), f"--app={target_url}"])
        else:
            subprocess.Popen([str(browser), "--new-window", target_url])
        return
    import webbrowser

    webbrowser.open(target_url)


def _focus_window_or_raise(page_title: str):
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        window = focus_window_by_title(page_title, maximize=True)
        if window is not None:
            time.sleep(0.8)
            refreshed = focus_window_by_title(page_title, maximize=True)
            return refreshed or window
        time.sleep(0.5)
    raise RuntimeError(f"failed to focus browser window titled {page_title!r}")


def _set_window_topmost(hwnd: int, *, topmost: bool) -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetWindowPos(
            int(hwnd),
            HWND_TOPMOST if topmost else HWND_NOTOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE,
        )
    except Exception:
        return


def _close_window(hwnd: int) -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.PostMessageW(int(hwnd), WM_CLOSE, 0, 0)
    except Exception:
        return


def _status_payload() -> dict[str, Any]:
    target_ids = [target.target_id for target in TinyTargetServer.targets]
    clicked = sorted(TinyTargetServer.clicked_ids)
    return {
        "target_count": len(target_ids),
        "target_ids": target_ids,
        "clicked_ids": clicked,
        "remaining_ids": [target_id for target_id in target_ids if target_id not in clicked],
        "click_log": TinyTargetServer.click_log,
    }


def _build_summary(
    *,
    result: dict[str, Any],
    status: dict[str, Any],
    targets: tuple[TargetSpec, ...],
    elapsed_seconds: float,
    artifact_dir: Path,
    agent_trace_path: Path,
    computer_trace_path: Path,
) -> dict[str, Any]:
    step_records = (
        result.get("metadata", {})
        .get("step_records", [])
        if isinstance(result.get("metadata"), dict)
        else []
    )
    local_refinement_outputs = _count_local_refinement_outputs(step_records)
    local_refinement_tool_calls = _count_local_refinement_tool_calls(step_records)
    clicked_ids = set(status.get("clicked_ids") or [])
    target_ids = [target.target_id for target in targets]
    return {
        "ok": all(target_id in clicked_ids for target_id in target_ids),
        "status": result.get("status"),
        "reason": result.get("reason"),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "steps": result.get("steps"),
        "targets": [target.to_dict() for target in targets],
        "target_size_px": TinyTargetServer.target_size_px,
        "clicked_ids": status.get("clicked_ids", []),
        "remaining_ids": status.get("remaining_ids", []),
        "click_count": len(status.get("click_log", [])),
        "miss_count": len(
            [item for item in status.get("click_log", []) if not item.get("hit_id")]
        ),
        "local_refinement_outputs": local_refinement_outputs,
        "local_refinement_tool_calls": local_refinement_tool_calls,
        "click_log": status.get("click_log", []),
        "model_step_metrics": _compact_model_response_metrics(result),
        "artifact_dir": str(artifact_dir),
        "agent_trace_path": str(agent_trace_path),
        "computer_trace_path": str(computer_trace_path),
    }


def _compact_model_response_metrics(result: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = result.get("metadata") if isinstance(result, dict) else None
    metrics = metadata.get("model_response_metrics", []) if isinstance(metadata, dict) else []
    if not isinstance(metrics, list):
        return []
    compacted: list[dict[str, Any]] = []
    for item in metrics:
        if not isinstance(item, dict):
            continue
        compacted.append(
            {
                "step": item.get("step"),
                "latency_seconds": item.get("latency_seconds"),
                "input_tokens": item.get("input_tokens"),
                "output_tokens": item.get("output_tokens"),
                "total_tokens": item.get("total_tokens"),
                "reasoning_text_chars": item.get("reasoning_text_chars"),
                "output_text_chars": item.get("output_text_chars"),
                "function_call_count": item.get("function_call_count"),
                "tool_argument_chars": item.get("tool_argument_chars"),
                "warnings": item.get("warnings", []),
                "reasoning_preview": item.get("reasoning_generated_text_preview")
                or item.get("reasoning_text_preview"),
                "output_preview": item.get("output_text_preview"),
            }
        )
    return compacted


def _count_local_refinement_outputs(step_records: Any) -> int:
    if not isinstance(step_records, list):
        return 0
    count = 0
    for step in step_records:
        metadata = step.get("metadata") if isinstance(step, dict) else None
        refinements = metadata.get("local_refinements") if isinstance(metadata, dict) else None
        if isinstance(refinements, list):
            count += len(refinements)
    return count


def _count_local_refinement_tool_calls(step_records: Any) -> int:
    if not isinstance(step_records, list):
        return 0
    count = 0
    for step in step_records:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        if not isinstance(action, dict):
            continue
        if action.get("coordinate_space") == "local_refinement_1000":
            count += 1
    return count


if __name__ == "__main__":
    main()
