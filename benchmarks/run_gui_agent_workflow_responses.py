from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cfie_client import (
    ComputerLoop,
    ScaledPillowScreenCapture,
    create_default_backend,
)
from cfie_gui_agent import (
    GuiAgentRunner,
    GuiAgentTaskSpec,
    ModelToolRegistry,
    OpenAIResponsesAgent,
)
from cfie_gui_agent.workflow import load_workflow_items


WORKFLOW_TOOL_NAMES = (
    "computer_use",
    "read_text_file",
    "record_workflow_result",
    "request_human_help",
    "finish_subtask",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a GUI Agent workflow through a local OpenAI Responses service."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35-vl")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--app-name", default="Doubao Web")
    parser.add_argument("--target-url", default="https://www.doubao.com/chat/")
    parser.add_argument(
        "--input-path",
        default=".bench_logs/gui_agent_workflow/doubao_items.jsonl",
    )
    parser.add_argument(
        "--trace-path",
        default=".bench_logs/gui_agent_workflow/doubao_trace.jsonl",
    )
    parser.add_argument(
        "--artifact-dir",
        default=".bench_logs/gui_agent_workflow/artifacts",
    )
    parser.add_argument(
        "--chrome-path",
        default=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=768)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high"),
        default="none",
        help="Responses reasoning effort. Keep none/low for low-latency GUI control.",
    )
    parser.add_argument(
        "--tool-profile",
        choices=("workflow", "all"),
        default="workflow",
        help="Use a smaller tool schema for workflow runs to reduce prompt cost.",
    )
    parser.add_argument("--item-limit", type=int, default=1)
    parser.add_argument("--screenshot-max-width", type=int, default=1920)
    parser.add_argument("--screenshot-max-height", type=int, default=1080)
    parser.add_argument("--screenshot-jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--screenshot-crop",
        default=None,
        help="Optional physical crop as x,y,width,height before scaling.",
    )
    parser.add_argument(
        "--focus-window-title-pattern",
        default=None,
        help="Optional Windows title regex to bring the target window to front.",
    )
    parser.add_argument(
        "--crop-focused-window",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the focused matched window rect as screenshot crop.",
    )
    parser.add_argument(
        "--maximize-focused-window",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Maximize the matched target window before measuring its crop.",
    )
    parser.add_argument(
        "--screenshot-grid",
        choices=("off", "coarse", "fine"),
        default="off",
        help="Optional coordinate grid after crop/resize for debugging localization.",
    )
    parser.add_argument(
        "--screenshot-url-mode",
        choices=("file", "data"),
        default="file",
        help=(
            "Use file URLs for local model servers started with "
            "--allowed-local-media-path, or data URLs for remote servers."
        ),
    )
    parser.add_argument(
        "--image-detail",
        choices=("low", "auto", "high", "original", "none"),
        default="high",
        help="Responses image detail for initial screenshots and computer outputs.",
    )
    parser.add_argument("--open-url", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--result-json", default=None)
    return parser


def ensure_default_input(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "item_id": "sample_001",
            "input_text": "请用一句话回答：2 + 3 等于几？",
            "expected_output": "5",
        },
        {
            "item_id": "sample_002",
            "input_text": "请用一句话回答：中国的首都是哪里？",
            "expected_output": "北京",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


def materialize_run_input(
    source_path: Path,
    *,
    artifact_dir: Path,
    item_limit: int,
) -> Path:
    if item_limit <= 0:
        return source_path
    items = load_workflow_items(source_path, limit=item_limit)
    if len(items) == item_limit and source_path.suffix.lower() == ".jsonl":
        limited_dir = artifact_dir / "inputs"
        limited_dir.mkdir(parents=True, exist_ok=True)
        limited_path = limited_dir / f"{source_path.stem}.limit{item_limit}.jsonl"
        limited_path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "item_id": item.item_id,
                        "input_text": item.input_text,
                        "expected_output": item.expected_output,
                        **item.metadata,
                    },
                    ensure_ascii=False,
                )
                for item in items
            ),
            encoding="utf-8",
        )
        return limited_path
    return source_path


def open_target_url(*, chrome_path: str, target_url: str) -> None:
    if sys.platform == "win32" and Path(chrome_path).exists():
        subprocess.Popen([chrome_path, target_url])
        return
    webbrowser.open(target_url)


def parse_crop(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--screenshot-crop must be x,y,width,height")
    x, y, width, height = (int(part) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError("--screenshot-crop width/height must be positive")
    return (x, y, width, height)


def focus_target_window(
    title_pattern: str | None,
    *,
    maximize: bool,
) -> dict[str, Any] | None:
    if not title_pattern or sys.platform != "win32":
        return None
    from cfie_client.executor.windows import focus_window_by_title

    window = focus_window_by_title(title_pattern, maximize=maximize)
    if window is None:
        return None
    return {"title": window.title, "crop": window.crop_box, "rect": window.rect}


def build_instruction(
    *,
    app_name: str,
    target_url: str,
    input_path: Path,
    trace_path: Path,
    item_limit: int,
    items: tuple[Any, ...],
) -> str:
    item_lines = _workflow_items_prompt(items)
    return "\n".join(
        [
            f"当前 APP：{app_name}",
            f"目标网址：{target_url}",
            f"输入清单：{input_path}",
            f"轨迹文件：{trace_path}",
            f"本轮最多处理 {item_limit} 条输入。",
            "",
            "当前待处理条目已经由 harness 读取，不需要再读文件：",
            item_lines,
            "",
            "你是 GUI Agent，目标是在当前浏览器页面完成这个任务流。",
            "先观察屏幕。如果页面未打开或不可用，使用 computer_use 打开/定位目标页面。",
            "如果浏览器或目标网页只占当前截图的一部分，先调用 set_app_viewport 记录应用区域；后续截图会裁剪到该区域，以降低每轮视觉输入延迟。",
            "computer_use 坐标一律使用 coordinate_space=\"qwen_normalized_1000\"，鼠标坐标是当前模型图像的 0..1000 归一化坐标，不要使用物理屏幕坐标或截图像素坐标。",
            "对每个条目：用 computer_use 点击并聚焦输入框，然后用 computer_use 的 type 动作输入该条目的 input_text，之后用 computer_use 点击可见发送按钮；如果应用明确支持 Enter 发送，也可以用 computer_use 发送 Enter。",
            "如果输入框已经可见，不要连续只点击输入框；应在同一次或下一次 computer_use 中完成 type。如果输入框里已经有待发送文本，下一步必须提交或等待输出，不要再次输入同一段文字。",
            "如果两次点击后仍无法输入或提交，应调用 request_human_help。",
            "每个条目结束时调用 record_workflow_result。为了保持响应快速，优先只传 item_id、output_text、status、reason；不要重复 input_text 和 expected_output，harness 会按 item_id 从输入清单回填。",
            "协议硬约束：如果当前任务还没完成，普通文本不会被视为完成；必须调用工具推进。",
            "工具协议：如果下一步需要工具，只返回一个完整工具调用，不要在工具调用前输出“思考过程”“计划”“分析”或解释性文字。",
            "坐标协议：每个 computer_use 都必须包含 coordinate_space=\"qwen_normalized_1000\"。鼠标 x/y 和拖拽路径坐标必须使用当前图像的 0..1000 归一化坐标：(0,0) 是当前图像左上角，(1000,1000) 是右下角。harness 会在本地转换为截图像素坐标。不要使用物理桌面坐标，也不要在这个任务流中使用原始截图像素坐标。",
            "不要输出长篇思考过程，也不要复述截图或视频内容；除非这是任务结果，否则只给必要工具调用或极短摘要。",
            "如果下一步是调用工具，不要先输出 visible previous state / next action 文本；直接调用工具。",
            "如果登录、验证码、页面卡住、控件不可见或无法判断输出完成，调用 request_human_help。",
            "如果提交后出现登录、授权、解锁更多功能等弹窗，立即调用 request_human_help；不要关闭弹窗，也不要继续尝试点击页面。",
            "完成本轮所有条目后调用 finish_subtask。",
            "computer_use 每次返回的截图已经在下一轮输入中直接给你；不要为了检查刚刚的截图再调用 read_image。",
            "不要把当前场景当成产品模式；这只是当前 APP 会话的任务流配置。",
        ]
    )


def _workflow_items_prompt(items: tuple[Any, ...]) -> str:
    rows: list[dict[str, str]] = []
    for item in items:
        rows.append(
            {
                "item_id": str(getattr(item, "item_id", "")),
                "input_text": str(getattr(item, "input_text", "")),
                "expected_output": str(getattr(item, "expected_output", "")),
            }
        )
    return json.dumps(rows, ensure_ascii=False)


def _workflow_item_manifest_prompt(items: tuple[Any, ...]) -> str:
    rows: list[dict[str, Any]] = []
    for item in items:
        rows.append(
            {
                "item_id": str(getattr(item, "item_id", "")),
                "source": str(getattr(item, "source", "")),
                "has_expected_output": bool(getattr(item, "expected_output", "")),
                "has_attachment": bool(getattr(item, "attachment_path", None)),
            }
        )
    return json.dumps(rows, ensure_ascii=False)


def build_instruction_v2(
    *,
    app_name: str,
    target_url: str,
    input_path: Path,
    trace_path: Path,
    item_limit: int,
    items: tuple[Any, ...],
) -> str:
    item_manifest = _workflow_item_manifest_prompt(items)
    return "\n".join(
        [
            f"当前 APP：{app_name}",
            f"目标网址：{target_url}",
            f"输入清单：{input_path}",
            f"轨迹文件：{trace_path}",
            f"本轮最多处理 {item_limit} 条输入。",
            "",
            "第一步必须调用 read_text_file 读取输入清单；不要根据下面的清单概览直接执行任务。",
            "下面只给出 item_id 和来源概览，用于核对进度，不包含题目正文：",
            item_manifest,
            "",
            "你是通用 GUI Agent，目标是在当前浏览器页面完成这个 APP 会话的任务流。",
            "先观察屏幕。如果页面未打开或不可用，用 computer_use 打开或定位目标页面。",
            "不要调用任何 APP 专用工具；只能使用通用工具读取文件、操作电脑、记录结果、请求人工和结束子任务。",
            "每个 computer_use 都必须包含 coordinate_space=\"qwen_normalized_1000\"。鼠标 x/y 和拖拽路径必须使用当前模型图像的 0..1000 归一化坐标：(0,0) 是当前图像左上角，(1000,1000) 是右下角。不要使用物理屏幕坐标，也不要使用截图像素坐标。",
            "对每个条目：读取 input_text 或 question 字段，点击并聚焦输入框，输入该文本，提交给网页应用，等待并读取网页输出，然后调用 record_workflow_result。",
            "网页聊天输入框完成 type 后，优先用 computer_use 的 keypress Enter 提交；只有明确 Enter 不能发送时才点击可见发送按钮。不要反复点击同一个发送位置。",
            "如果输入框已经可见，不要连续只点击输入框；应在同一次或下一次 computer_use 中完成 type。如果输入框里已有待发送文本，下一步必须提交或等待输出，不要再次输入同一段文字。",
            "如果两次尝试后仍无法输入、提交或判断输出完成，调用 request_human_help。",
            "record_workflow_result 为了保持响应快，优先只传 item_id、output_text、status、reason；不要重复 input_text 和 expected_output，harness 会按 item_id 从输入清单回填。",
            "协议约束：如果当前任务还没完成，普通文本不会被视为完成；必须调用工具推进。",
            "工具约束：如果下一步需要工具，只返回一个完整工具调用，不要在工具调用前输出“思考过程”“计划”“分析”或解释性文字。",
            "不要输出长篇思考过程，也不要复述截图或视频内容；除非这是任务结果，否则只给必要工具调用或极短摘要。",
            "如果登录、验证码、页面卡住、控件不可见、需要人工判断、或提交后出现登录/授权/解锁功能弹窗，立即调用 request_human_help。",
            "完成本轮所有条目并记录结果后，调用 finish_subtask。",
            "computer_use 每次返回的截图已经会在下一轮输入中直接给你；不要为了检查刚才的截图再调用 read_image。",
            "不要把当前场景当成产品模式；这只是当前 APP 会话的任务流配置。",
        ]
    )


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = build_parser().parse_args()
    input_path = Path(args.input_path)
    trace_path = Path(args.trace_path)
    artifact_dir = Path(args.artifact_dir)
    ensure_default_input(input_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_input_path = materialize_run_input(
        input_path,
        artifact_dir=artifact_dir,
        item_limit=args.item_limit,
    )
    run_items = load_workflow_items(run_input_path, limit=args.item_limit)

    if args.open_url:
        open_target_url(chrome_path=args.chrome_path, target_url=args.target_url)
        time.sleep(3.0)
    focused_window = focus_target_window(
        args.focus_window_title_pattern,
        maximize=args.maximize_focused_window,
    )
    if focused_window is not None:
        time.sleep(0.5)
        print(
            json.dumps(
                {
                    "focused_window_title": focused_window["title"],
                    "focused_window_crop": focused_window["crop"],
                    "focused_window_rect": focused_window["rect"],
                    "focus_window_title_pattern": args.focus_window_title_pattern,
                },
                ensure_ascii=False,
            )
        )
    crop_box = parse_crop(args.screenshot_crop)
    if crop_box is None and args.crop_focused_window:
        crop_box = (
            tuple(focused_window["crop"]) if focused_window is not None else None
        )

    task = GuiAgentTaskSpec(
        task_id=f"workflow:{args.app_name.lower().replace(' ', '_')}",
        target_app=args.app_name,
        instruction=build_instruction_v2(
            app_name=args.app_name,
            target_url=args.target_url,
            input_path=run_input_path,
            trace_path=trace_path,
            item_limit=args.item_limit,
            items=run_items,
        ),
        expected_outcome="Workflow item results are recorded in the trace.",
        metadata={
            "target_url": args.target_url,
            "input_path": str(run_input_path),
            "source_input_path": str(input_path),
            "trace_path": str(trace_path),
            "item_limit": args.item_limit,
        },
    )
    screen = ScaledPillowScreenCapture(
        max_width=args.screenshot_max_width,
        max_height=args.screenshot_max_height,
        jpeg_quality=args.screenshot_jpeg_quality,
        url_mode=args.screenshot_url_mode,
        output_dir=artifact_dir / "screenshots",
        crop_box=crop_box,
        grid_overlay=args.screenshot_grid,
    )
    tool_registry = (
        ModelToolRegistry(allowed_tool_names=WORKFLOW_TOOL_NAMES)
        if args.tool_profile == "workflow"
        else ModelToolRegistry()
    )
    runner = GuiAgentRunner(
        computer_loop=ComputerLoop(
            backend=create_default_backend(),
            screen=screen,
            trace_path=trace_path,
            trace_artifact_dir=artifact_dir,
            screenshot_detail=None if args.image_detail == "none" else args.image_detail,
            model_coordinate_mode="qwen_normalized_1000",
        ),
        max_steps=args.max_steps,
        image_detail=None if args.image_detail == "none" else args.image_detail,
        tool_registry=tool_registry,
    )
    runner.trace_store.path = trace_path
    agent = OpenAIResponsesAgent(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_output_tokens=args.max_output_tokens,
        timeout=args.timeout,
        reasoning_effort=args.reasoning_effort,
        chat_template_kwargs={"enable_thinking": args.reasoning_effort != "none"},
        tool_registry=tool_registry,
    )
    result = runner.run_task(task, agent)
    payload = {
        "result": result.to_trace_payload(),
        "trace_path": str(trace_path),
        "artifact_dir": str(artifact_dir),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.result_json:
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
