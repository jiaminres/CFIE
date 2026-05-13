"""HTTP 服务器模式: predictor + OpenAI chat completions 端到端测试。

用 multiprocessing.Process 启动服务器 → 轮询端口就绪 → 发送 HTTP 请求 → 校验结果 → 关闭。

用法:
    python scripts/test_predictor_server.py
"""
from __future__ import annotations

import json
import multiprocessing
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil

MODEL = (
    "C:/Users/13642/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-122B-A10B-GPTQ-Int4/snapshots/"
    "5b9f0050d3ec98b0c81a7716776533c5eacebb64"
)
PREDICTOR = (
    "C:/Users/13642/PycharmProjects/vllm/CFIE/.tmp/predictor_122b_books/"
    "mask_layer0_frozen_router_delta_384_purekl_e200/"
    "predictor_runtime_epoch_200.ckpt"
)
HOST = "127.0.0.1"
PORT = 18882
BASE_URL = f"http://{HOST}:{PORT}"

RESULT_FILE = Path(
    "C:/Users/13642/PycharmProjects/vllm/CFIE/.tmp/predictor_server_result.json"
)

STARTUP_TIMEOUT_S = 600
INFERENCE_TIMEOUT_S = 120


# ---------------------------------------------------------------
# 服务器进程入口
# ---------------------------------------------------------------
def _run_server():
    """在独立进程中运行 API 服务器。"""
    import asyncio

    from cfie.entrypoints.openai.api_server import (
        run_server,
        validate_parsed_serve_args,
    )
    from cfie.entrypoints.utils import cli_env_setup
    from cfie.utils.argparse_utils import FlexibleArgumentParser
    from cfie.entrypoints.openai.cli_args import make_arg_parser

    cli_env_setup()

    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)

    argv = [
        MODEL,
        "--dtype", "float16",
        "--quantization", "gptq_marlin",
        "--max-model-len", "128",
        "--gpu-memory-utilization", "0.60",
        "--enforce-eager",
        "--max-num-seqs", "1",
        "--max-num-batched-tokens", "128",
        "--hf-overrides", json.dumps({"predictor_checkpoint_path": PREDICTOR}),
        "--host", HOST,
        "--port", str(PORT),
    ]
    args = parser.parse_args(argv)
    # model_tag → model 同步（通常由 ServeSubcommand.cmd() 处理，
    # 直接调用 run_server 时需要手动同步）
    if hasattr(args, "model_tag") and args.model_tag is not None:
        args.model = args.model_tag
    validate_parsed_serve_args(args)

    try:
        import uvloop
        uvloop.run(run_server(args))
    except ImportError:
        # uvloop 在 Windows 上不可用，回退到 asyncio
        # Windows ProactorEventLoop 与 ZMQ 不兼容，需要 SelectorPolicy
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(
                asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(run_server(args))


# ---------------------------------------------------------------
# HTTP 客户端
# ---------------------------------------------------------------
def _port_is_open() -> bool:
    import socket
    try:
        sock = socket.create_connection((HOST, PORT), timeout=3)
        sock.close()
        return True
    except OSError:
        return False


def _server_is_ready(path: str = "/health") -> bool:
    try:
        req = urllib.request.Request(f"{BASE_URL}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def wait_for_server(timeout_s: int = STARTUP_TIMEOUT_S) -> bool:
    print(f"Waiting for server at {BASE_URL}/health ...", flush=True)
    deadline = time.monotonic() + timeout_s
    t0 = time.monotonic()
    last_log = t0
    while time.monotonic() < deadline:
        if _port_is_open() and _server_is_ready():
            elapsed = time.monotonic() - t0
            print(f"    server ready! ({elapsed:.0f}s)", flush=True)
            return True
        if time.monotonic() - last_log > 20:
            print(f"    still waiting... ({time.monotonic() - t0:.0f}s)", flush=True)
            last_log = time.monotonic()
        time.sleep(2)
    return False


def send_chat_completion(prompt: str, stream: bool = False) -> dict:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": stream,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()

    if stream:
        return _handle_streaming(req, t0)
    else:
        return _handle_non_streaming(req, t0)


def _handle_non_streaming(req: urllib.request.Request, t0: float) -> dict:
    with urllib.request.urlopen(req, timeout=INFERENCE_TIMEOUT_S) as resp:
        raw = resp.read()
    elapsed = time.perf_counter() - t0
    data = json.loads(raw)
    content = ""
    choices = data.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "")
    return {
        "status": resp.status,
        "streaming": False,
        "content": content,
        "elapsed": round(elapsed, 1),
        "finish_reason": choices[0].get("finish_reason") if choices else None,
    }


def _handle_streaming(req: urllib.request.Request, t0: float) -> dict:
    chunks = 0
    full_content: list[str] = []
    finish_reason = None

    with urllib.request.urlopen(req, timeout=INFERENCE_TIMEOUT_S) as resp:
        for line_bytes in resp:
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                chunks += 1
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("content"):
                        full_content.append(delta["content"])
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]
            except json.JSONDecodeError:
                continue

    elapsed = time.perf_counter() - t0
    return {
        "status": resp.status,
        "streaming": True,
        "content": "".join(full_content),
        "elapsed": round(elapsed, 1),
        "chunk_count": chunks,
        "finish_reason": finish_reason,
    }


def _server_has_models(base_url: str) -> bool:
    try:
        req = urllib.request.Request(f"{base_url}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("object") == "list" and len(data.get("data", [])) > 0
    except Exception:
        return False


# ---------------------------------------------------------------
# ---------------------------------------------------------------
# 进程树清理
# ---------------------------------------------------------------
def _kill_proc_tree(proc: multiprocessing.Process, timeout: int = 15) -> None:
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        proc.terminate()
        _gone, alive = psutil.wait_procs(children + [parent], timeout=timeout)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
    except (psutil.NoSuchProcess, Exception):
        pass
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)


# 主流程
# ---------------------------------------------------------------
def main() -> int:
    print("=" * 60)
    print("[server test] predictor + OpenAI HTTP server E2E")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Predictor: {PREDICTOR}")
    print(f"Listen: {BASE_URL}")

    # 启动服务器进程
    print("\n[1] Starting server process...", flush=True)
    t_start = time.perf_counter()

    server_proc = multiprocessing.Process(
        target=_run_server,
        name="cfie-api-server",
    )
    server_proc.start()

    try:
        # 等待服务器就绪
        print("[2] Waiting for server ready...", flush=True)
        if not wait_for_server():
            print("[FAIL] server did not become ready within timeout", flush=True)
            _kill_proc_tree(server_proc)
            return 1

        startup_time = time.perf_counter() - t_start
        print(f"    startup: {startup_time:.1f}s", flush=True)

        # 检查 /v1/models 端点
        print("[3] GET /v1/models...", flush=True)
        try:
            req = urllib.request.Request(f"{BASE_URL}/v1/models")
            with urllib.request.urlopen(req, timeout=10) as resp:
                models_data = json.loads(resp.read())
            model_count = len(models_data.get("data", []))
            print(f"    {model_count} models listed", flush=True)
        except Exception as exc:
            print(f"    /v1/models failed: {exc}", flush=True)
            model_count = -1

        # 非流式请求
        print("[4] POST /v1/chat/completions (non-streaming)...", flush=True)
        try:
            result_ns = send_chat_completion(
                "请用一句话介绍你自己", stream=False)
            print(f"    status={result_ns['status']} "
                  f"content_len={len(result_ns['content'])} "
                  f"elapsed={result_ns['elapsed']}s "
                  f"finish={result_ns['finish_reason']}",
                  flush=True)
        except Exception as exc:
            print(f"    non-streaming failed: {exc}", flush=True)
            result_ns = {"error": str(exc), "status": 0}

        # 流式请求
        print("[5] POST /v1/chat/completions (streaming SSE)...", flush=True)
        try:
            result_s = send_chat_completion(
                "请复述：太阳从东方升起", stream=True)
            print(f"    status={result_s['status']} "
                  f"content_len={len(result_s['content'])} "
                  f"elapsed={result_s['elapsed']}s "
                  f"chunks={result_s['chunk_count']} "
                  f"finish={result_s['finish_reason']}",
                  flush=True)
        except Exception as exc:
            print(f"    streaming failed: {exc}", flush=True)
            result_s = {"error": str(exc), "status": 0}

        # 汇总
        response_ok = (
            result_ns.get("status") == 200
            and len(result_ns.get("content", "")) > 0
        )

        result = {
            "ok": response_ok,
            "startup_seconds": round(startup_time, 1),
            "model_count": model_count,
            "non_streaming": result_ns,
            "streaming": result_s,
        }
        RESULT_FILE.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n    result saved: {RESULT_FILE}")

        if response_ok:
            print("\n[PASS] predictor + OpenAI HTTP server E2E OK")
            return 0
        else:
            print("\n[FAIL] response check failed")
            return 1

    finally:
        print("    shutting down server process...", flush=True)
        _kill_proc_tree(server_proc)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
