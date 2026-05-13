"""Predictor + MTP 多配置对比基准测试。

每个配置（MTP=0/1/2/3）:
  1. 启动 OpenAI API 服务器 (predictor 始终开启)
  2. 发送多轮对话问题
  3. 记录输出内容 + 首 token 延迟 + 总耗时 + token 数
  4. 关闭服务器

最终生成对比报告。

用法:
    python scripts/benchmark_predictor_mtp.py
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
from typing import Any

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
PORT = 18883
BASE_URL = f"http://{HOST}:{PORT}"

RESULT_DIR = Path("C:/Users/13642/PycharmProjects/vllm/CFIE/.tmp")
RESULT_FILE = RESULT_DIR / "benchmark_predictor_mtp.json"

STARTUP_TIMEOUT_S = 600
INFERENCE_TIMEOUT_S = 180

# ---------------------------------------------------------------
# 测试问题集 (多轮对话 + 上下文延续)
# ---------------------------------------------------------------
CONVERSATIONS: list[dict[str, Any]] = [
    {
        "id": "conv_1_chinese_knowledge",
        "name": "中文知识问答",
        "messages": [
            {"role": "user", "content": "请用一句话介绍大语言模型的基本工作原理。"},
        ],
    },
    {
        "id": "conv_2_moe_architecture",
        "name": "MoE 架构讨论",
        "messages": [
            {"role": "user", "content": "MoE（混合专家）架构相比Dense架构有什么优势和缺点？"},
        ],
    },
    {
        "id": "conv_3_multi_turn_context",
        "name": "多轮上下文对话",
        "messages": [
            {"role": "user", "content": "什么是Transformer模型中的注意力机制？"},
            {"role": "assistant",
             "content": "注意力机制是Transformer的核心组件，它允许模型在处理序列时动态关注不同位置的信息。"},
            {"role": "user", "content": "那多头注意力是如何扩展这个机制的？"},
        ],
    },
    {
        "id": "conv_4_deployment",
        "name": "模型部署推理",
        "messages": [
            {"role": "user",
             "content": "在显存受限的设备上运行大语言模型，有哪些主流的技术方案？请简要列举。"},
        ],
    },
    {
        "id": "conv_5_code_generation",
        "name": "代码生成",
        "messages": [
            {"role": "user",
             "content": "用Python写一个函数，计算两个列表的余弦相似度。只输出代码，不要解释。"},
        ],
    },
]

# MTP 配置列表: (spec_method, num_spec_tokens, label)
# 注意: MTP=0 baseline 已跑完，可设 SKIP_BASELINE=True 跳过
SKIP_BASELINE = False
_ALL_CONFIGS: list[tuple[str, int | None, str]] = [
    ("none", None, "MTP=0 (baseline, predictor only)"),
    ("mtp", 1, "MTP=1"),
    ("mtp", 2, "MTP=2"),
    ("mtp", 3, "MTP=3"),
]
MTP_CONFIGS = _ALL_CONFIGS[1:] if SKIP_BASELINE else _ALL_CONFIGS


# ---------------------------------------------------------------
# 服务器启动/关闭
# ---------------------------------------------------------------
def _run_server(mtp_method: str, mtp_tokens: int | None):
    """在独立进程中运行 API 服务器。"""
    import asyncio, json as _json

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
        "--max-model-len", "256",
        "--gpu-memory-utilization", "0.60",
        "--enforce-eager",
        "--max-num-seqs", "1",
        "--max-num-batched-tokens", "256",
        "--hf-overrides",
        _json.dumps({"predictor_checkpoint_path": PREDICTOR}),
        "--host", HOST,
        "--port", str(PORT),
    ]
    # MTP speculative decoding 参数（通过 --speculative-config JSON dict 传入）
    if mtp_method == "mtp" and mtp_tokens:
        argv.extend([
            "--speculative-config",
            _json.dumps({"method": "mtp", "num_speculative_tokens": mtp_tokens}),
        ])

    args = parser.parse_args(argv)
    if hasattr(args, "model_tag") and args.model_tag is not None:
        args.model = args.model_tag
    validate_parsed_serve_args(args)

    try:
        import uvloop
        uvloop.run(run_server(args))
    except ImportError:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(
                asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(run_server(args))


def _port_is_open() -> bool:
    import socket
    try:
        s = socket.create_connection((HOST, PORT), timeout=3)
        s.close()
        return True
    except OSError:
        return False


def wait_for_server(timeout_s: int = STARTUP_TIMEOUT_S) -> tuple[bool, float]:
    print(f"  waiting for {BASE_URL}/health ...", flush=True)
    t0 = time.monotonic()
    deadline = t0 + timeout_s
    last_log = t0
    while time.monotonic() < deadline:
        if _port_is_open():
            try:
                req = urllib.request.Request(f"{BASE_URL}/health", method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        elapsed = time.monotonic() - t0
                        print(f"  ready ({elapsed:.0f}s)", flush=True)
                        return True, elapsed
            except Exception:
                pass
        if time.monotonic() - last_log > 30:
            print(f"  still waiting ({time.monotonic() - t0:.0f}s)...", flush=True)
            last_log = time.monotonic()
        time.sleep(2)
    return False, 0


# ---------------------------------------------------------------
# HTTP 请求
# ---------------------------------------------------------------


def send_chat_completion(messages: list[dict], max_tokens: int = 128,
                         temperature: float = 0.0, stream: bool = True) -> dict:
    """发送 chat completion 请求，返回包含内容、耗时、token 统计的结果。"""
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_req = time.perf_counter()

    if not stream:
        return _non_streaming(req, t_req)
    else:
        return _streaming_sse(req, t_req)


def _non_streaming(req: urllib.request.Request, t0: float) -> dict:
    with urllib.request.urlopen(req, timeout=INFERENCE_TIMEOUT_S) as resp:
        raw = resp.read()
    elapsed = time.perf_counter() - t0
    data = json.loads(raw)
    content = ""
    usage = {}
    choices = data.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        if choices[0].get("finish_reason"):
            pass
    if data.get("usage"):
        usage = dict(data["usage"])
    return {
        "ok": True,
        "content": content,
        "total_seconds": round(elapsed, 2),
        "ttft_seconds": round(elapsed, 2),  # 非流式只有总时间
        "usage": usage,
    }


def _streaming_sse(req: urllib.request.Request, t0: float) -> dict:
    content_parts: list[str] = []
    chunk_count = 0
    ttft: float | None = None
    finish_reason = None
    usage = {}

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
                chunk_count += 1
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("content") and ttft is None:
                        ttft = round(time.perf_counter() - t0, 3)
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]
                if chunk.get("usage"):
                    usage = dict(chunk["usage"])
            except json.JSONDecodeError:
                continue

    total_elapsed = time.perf_counter() - t0
    content = "".join(content_parts)
    return {
        "ok": True,
        "content": content,
        "content_len": len(content),
        "chunk_count": chunk_count,
        "ttft_seconds": ttft,
        "total_seconds": round(total_elapsed, 2),
        "finish_reason": finish_reason,
        "usage": usage,
        "tokens_per_second": round(len(content_parts) / total_elapsed, 1) if total_elapsed > 0 and content_parts else None,
    }


# ---------------------------------------------------------------
# 单次 MTP 配置的全部对话测试
# ---------------------------------------------------------------
def run_conversations(label: str) -> list[dict]:
    """发送所有对话，返回每个对话的结果列表。"""
    results: list[dict] = []
    for conv in CONVERSATIONS:
        conv_id = conv["id"]
        conv_name = conv["name"]
        messages = conv["messages"]
        print(f"    [{conv_name}] ", end="", flush=True)

        try:
            r = send_chat_completion(
                messages=messages,
                max_tokens=150 if conv_id == "conv_5_code_generation" else 128,
                temperature=0.0,
                stream=True,
            )
            content_preview = r.get("content", "")[:80].replace("\n", "\\n")
            print(f"TTFT={r.get('ttft_seconds', '?')}s "
                  f"total={r.get('total_seconds')}s "
                  f"len={r.get('content_len', 0)} "
                  f"finish={r.get('finish_reason', '?')} "
                  f"preview={content_preview}",
                  flush=True)
            r["conv_id"] = conv_id
            r["conv_name"] = conv_name
            r["messages"] = messages
            results.append(r)
        except Exception as exc:
            print(f"FAILED: {exc}", flush=True)
            results.append({
                "ok": False,
                "conv_id": conv_id,
                "conv_name": conv_name,
                "error": str(exc),
            })

    return results


# ---------------------------------------------------------------
# 进程树清理 (Windows 上 terminate() 不级联杀 EngineCore 子进程)
# ---------------------------------------------------------------
def _kill_proc_tree(proc: multiprocessing.Process, timeout: int = 15) -> None:
    """递归清理进程树，确保 EngineCore 等子进程也被 kill。"""
    pid = proc.pid
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        proc.terminate()
        # 等待所有子进程 + 自身退出
        _gone, alive = psutil.wait_procs(children + [parent], timeout=timeout)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
    except (psutil.NoSuchProcess, Exception):
        pass
    # 确保 multiprocessing 侧也 release
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)


def _wait_gpu_idle(
        target_mib: int = 1200, retries: int = 10, interval: float = 2.0
) -> bool:
    """等待 GPU 显存降回到接近空闲水平。"""
    for _ in range(retries):
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            used_mib = int(result.stdout.strip())
            if used_mib <= target_mib:
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


# ---------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------
def main() -> int:
    all_results: dict[str, Any] = {
        "model": MODEL,
        "predictor": PREDICTOR,
        "conversations": CONVERSATIONS,
        "mtp_results": {},
        "summary": {},
    }

    for mtp_method, mtp_tokens, label in MTP_CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"  [{label}]")
        print(f"{'=' * 60}")

        # 启动服务器
        print("[1] Starting server...", flush=True)
        server_proc = multiprocessing.Process(
            target=_run_server,
            args=(mtp_method, mtp_tokens),
            name="cfie-bench",
        )
        server_proc.start()

        config_start = time.perf_counter()
        try:
            ready, startup_time = wait_for_server()
            if not ready:
                print(f"  [FAIL] server did not become ready", flush=True)
                all_results["mtp_results"][label] = {
                    "error": "server startup timeout"}
                _kill_proc_tree(server_proc)
                _wait_gpu_idle()
                continue

            all_results["mtp_results"][label] = {
                "startup_seconds": round(startup_time, 1),
                "conversations": [],
            }

            # 执行对话
            print(f"[2] Running {len(CONVERSATIONS)} conversations...", flush=True)
            conv_results = run_conversations(label)
            all_results["mtp_results"][label]["conversations"] = conv_results

            total_inference = sum(
                r.get("total_seconds", 0) for r in conv_results if r.get("ok"))
            all_results["mtp_results"][label]["total_inference_seconds"] = round(
                total_inference, 1)

            # 打印本次摘要
            ok_count = sum(1 for r in conv_results if r.get("ok"))
            avg_ttft = (
                sum(r["ttft_seconds"] for r in conv_results
                    if r.get("ok") and r.get("ttft_seconds"))
                / max(ok_count, 1)
            )
            print(f"  [{label}] done: {ok_count}/{len(CONVERSATIONS)} OK, "
                  f"avg TTFT={avg_ttft:.2f}s, "
                  f"total inference={total_inference:.1f}s",
                  flush=True)

        finally:
            print(f"  shutting down server...", flush=True)
            _kill_proc_tree(server_proc)
            _wait_gpu_idle()

        config_elapsed = time.perf_counter() - config_start
        print(f"  config total time: {config_elapsed:.0f}s", flush=True)

    # -----------------------------------------------------------
    # 汇总对比
    # -----------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  SUMMARY - Predictor + MTP Benchmark")
    print(f"{'=' * 70}")
    print(f"{'Config':<10} {'OK':>3} {'TTFT':>7} {'Total':>7} {'Len':>5} {'TPS':>6}")
    print("-" * 45)

    best_config = None
    best_tps = -1.0
    for mtp_method, mtp_tokens, label in MTP_CONFIGS:
        data = all_results["mtp_results"].get(label, {})
        convs = data.get("conversations", [])
        ok_convs = [c for c in convs if c.get("ok")]
        if not ok_convs:
            print(f"{label:<10} {'0':>3} {'N/A':>7} {'N/A':>7} {'N/A':>5}")
            continue

        avg_ttft = sum(c.get("ttft_seconds", 0) for c in ok_convs) / len(ok_convs)
        avg_total = sum(c.get("total_seconds", 0) for c in ok_convs) / len(ok_convs)
        avg_len = sum(c.get("content_len", 0) for c in ok_convs) / len(ok_convs)
        avg_tps = sum(c.get("tokens_per_second", 0) or 0 for c in ok_convs) / len(ok_convs)
        print(f"{label:<10} {len(ok_convs):>3} {avg_ttft:>6.1f}s {avg_total:>6.1f}s {avg_len:>5.0f} {avg_tps:>5.1f}")

        if avg_tps > best_tps:
            best_tps = avg_tps
            best_config = label

    print("-" * 45)

    # 详情: 每轮对话的输入和输出
    if best_config:
        all_results["summary"]["best_config"] = best_config
        all_results["summary"]["best_tps"] = round(best_tps, 1)
        print(f"\n  Best config: {best_config} (avg {best_tps:.1f} tok/s)")
        print(f"\n  === Detailed I/O (best config: {best_config}) ===\n")

        best_data = all_results["mtp_results"].get(best_config, {})
        best_convs = best_data.get("conversations", [])
        for i, conv in enumerate(best_convs):
            conv_name = conv.get("conv_name", f"conv_{i}")
            msgs = conv.get("messages", [])
            print(f"  [{conv_name}]")
            for msg in msgs:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                # 截断过长的输入
                if len(content) > 120:
                    content = content[:120] + "..."
                print(f"    {role:>10}: {content}")
            output = conv.get("content", "")
            if len(output) > 200:
                output = output[:200] + "..."
            print(f"    output:    {output}")
            print(f"    (TTFT={conv.get('ttft_seconds','?')}s "
                  f"total={conv.get('total_seconds','?')}s "
                  f"len={conv.get('content_len',0)} "
                  f"finish={conv.get('finish_reason','?')})")
            print()

    # 保存
    RESULT_FILE.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  Full results: {RESULT_FILE}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
