"""Interactive native v1 engine chat entrypoint."""

from __future__ import annotations

import argparse
import copy
import contextlib
import io
import logging
import os
import re
import sys
import warnings
from argparse import Namespace
from collections.abc import Iterator
from typing import Any

from cfie.cli.native_generate import (
    _build_engine_args,
    _build_sampling_params,
    _iter_request_text,
    _resolve_runtime_symbols,
)
from cfie.logger import suppress_logging
from cfie.utils.logging import get_logger

# 初始化当前模块的 logger，供异常或调试日志使用。
logger = get_logger(__name__)

# 用户在交互式 chat 中可用的退出命令集合。
_EXIT_COMMANDS = frozenset({"/exit", "/quit"})

# 清空历史上下文的命令。
_CLEAR_COMMAND = "/clear"

# 默认不把 `<think>...</think>` 写回历史，避免后续 prompt 膨胀过快。
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_CLOSE_TOKEN = "</think>"


# 注册 `chat` / `native-chat` 子命令及其专属参数。
def add_native_chat_parser(subparsers: Any) -> None:
    # 在主命令解析器下新增 chat 子命令。
    parser = subparsers.add_parser(
        "chat",
        aliases=["native-chat"],
        help="Run an interactive streaming chat session",
    )

    # 必填：模型名称或模型路径。
    parser.add_argument("--model", required=True)

    # 日志级别，控制 CLI 层面输出详细程度。
    parser.add_argument("--log-level",
                        default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    # tokenizer 名称或路径；若不指定，默认复用 model。
    parser.add_argument("--tokenizer", default=None)

    # 模型/Tokenizer 的 revision（例如 huggingface 的分支、tag、commit）。
    parser.add_argument("--revision", default=None)

    # 下载目录。
    parser.add_argument("--download-dir", default=None)

    # 模型加载格式，例如 auto / safetensors 等。
    parser.add_argument("--load-format", default="auto")

    # 模型权重 dtype，例如 auto / float16 / bfloat16。
    parser.add_argument("--dtype", default="auto")

    # 量化方式，例如 awq / gptq；不启用则为 None。
    parser.add_argument("--quantization", default=None)

    # 模型最大上下文长度。
    # 同时兼容 --max-context-len 这个别名，统一映射到 max_model_len。
    parser.add_argument("--max-model-len",
                        "--max-context-len",
                        dest="max_model_len",
                        type=int,
                        default=32768)

    # 单次回答最多生成多少 token。
    parser.add_argument("--max-new-tokens", type=int, default=512)

    # GPU 显存使用率上限。
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    # 控制 prefix cache 是否启用；不显式传入时沿用引擎默认值。
    parser.add_argument("--enable-prefix-caching",
                        action=argparse.BooleanOptionalAction,
                        default=None)

    # MoE 场景下 CPU 侧预算参数。
    parser.add_argument("--moe-cpu-budget-gb", type=float, default=0.0)
    parser.add_argument("--moe-cpu-min-free-gb", type=float, default=0.0)

    # 允许卸载到 CPU 的显存大小（GB）。
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)

    # offload 后端。
    parser.add_argument("--offload-backend", default="auto")

    # 张量并行大小。
    parser.add_argument("--tensor-parallel-size", type=int, default=1)

    # 并发请求上限。对于本交互式 chat，通常为 1。
    parser.add_argument("--max-num-seqs", type=int, default=1)
    # 单轮调度的 batched token 预算上限。
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)

    # speculative decoding 方式。
    parser.add_argument("--spec-method",
                        choices=("none", "mtp"),
                        default="none")

    # speculative token 数量。
    parser.add_argument("--num-speculative-tokens", type=int, default=None)

    # attention / moe 后端配置。
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--moe-backend", default="auto")
    parser.add_argument("--mamba-cache-mode", default=None)
    parser.add_argument("--language-model-only",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--skip-mm-profiling",
                        action=argparse.BooleanOptionalAction,
                        default=False)

    # 采样参数；未显式传入时优先使用模型 generation_config.json。
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--frequency-penalty", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # 支持多次传入 --stop，形成 stop word 列表。
    parser.add_argument("--stop", action="append", default=None)

    # 系统提示词。
    parser.add_argument("--system-prompt", default=None)

    # 是否启用 thinking 模式。
    # BooleanOptionalAction 支持：
    #   --enable-thinking
    #   --no-enable-thinking
    parser.add_argument("--enable-thinking",
                        action=argparse.BooleanOptionalAction,
                        default=None)

    # 是否把 `<think>...</think>` 一并写回历史。
    parser.add_argument("--keep-thinking-in-history",
                        action=argparse.BooleanOptionalAction,
                        default=False)

    # 对交互式 chat 默认开启重复模式检测，避免 thinking 退化时刷满 max_new_tokens。
    parser.add_argument("--repetition-detection-max-pattern-size",
                        type=int,
                        default=64)
    parser.add_argument("--repetition-detection-min-pattern-size",
                        type=int,
                        default=4)
    parser.add_argument("--repetition-detection-min-count", type=int, default=3)

    # 在进入交互循环前先做一次隐藏 warmup，减少首轮固定冷启动开销。
    parser.add_argument("--startup-warmup",
                        action=argparse.BooleanOptionalAction,
                        default=True)

    # 是否展示底层 runtime 输出。
    parser.add_argument("--show-runtime-output", action="store_true")

    # 是否强制 eager 模式。
    parser.add_argument("--enforce-eager", action="store_true")

    # 是否启用多进程。
    parser.add_argument("--enable-multiprocessing", action="store_true")

    # 是否打印统计信息。
    parser.add_argument("--log-stats", action="store_true")

    # 指定该子命令实际执行的处理函数。
    parser.set_defaults(handler=run_native_chat)


# 按模型路径加载 tokenizer，供 chat template 渲染与 token 计数使用。
def _load_tokenizer(args: Namespace):
    from transformers import AutoTokenizer

    # 优先使用显式传入的 tokenizer；否则直接用 model 对应的 tokenizer。
    tokenizer_name = args.tokenizer or args.model

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=args.revision,
        trust_remote_code=False,
        use_fast=True,
    )

    # 若 tokenizer 没有 pad_token，但存在 eos_token，则复用 eos_token 作为 pad_token。
    # 这在部分模型上是常见兼容处理，可避免后续模板渲染或批处理时报错。
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# 组装 chat template 渲染参数，并按需透传 thinking 开关。
def _build_chat_template_kwargs(args: Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        # 这里只需要渲染成最终 prompt 字符串，不直接返回 token ids。
        "tokenize": False,

        # 追加 generation prompt，让模型知道接下来该生成 assistant 回复。
        "add_generation_prompt": True,
    }

    # 仅在用户显式指定时透传 enable_thinking。
    # 若为 None，则交由 tokenizer / 模板默认行为处理。
    if args.enable_thinking is not None:
        kwargs["enable_thinking"] = args.enable_thinking

    return kwargs


def _build_renderer_chat_params(args: Namespace) -> Any:
    from cfie.renderers.params import ChatParams

    return ChatParams(chat_template_kwargs=_build_chat_template_kwargs(args))


def _thinking_mode_enabled(args: Namespace) -> bool:
    return args.enable_thinking is not False


def _render_engine_chat_prompt(
        engine: Any,
        messages: list[dict[str, str]],
        args: Namespace,
) -> Any:
    _, prompts = engine.renderer.render_chat(
        [messages],
        _build_renderer_chat_params(args),
    )
    return prompts[0]


# 计算一段 prompt 在当前 tokenizer 下会占用多少 token。
def _count_prompt_tokens(tokenizer: Any, prompt: str) -> int:
    # 不添加额外 special tokens，因为 prompt 已经是模板渲染后的最终文本。
    encoded = tokenizer(prompt, add_special_tokens=False)

    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None:
        raise TypeError("tokenizer did not return input_ids for prompt counting")

    return len(input_ids)


# 把消息列表渲染成最终 prompt 文本，并顺手统计 prompt token 数。
def _render_prompt(
        tokenizer: Any,
        messages: list[dict[str, str]],
        args: Namespace,
) -> tuple[str, int]:
    # 使用 tokenizer 内置的 chat template，将 messages 渲染成模型实际接收的 prompt。
    prompt = tokenizer.apply_chat_template(
        messages,
        **_build_chat_template_kwargs(args),
    )

    # 当前逻辑要求渲染结果必须是字符串。
    if not isinstance(prompt, str):
        raise TypeError("chat template rendering must return a string prompt")

    # 返回 prompt 本身及其 token 数。
    return prompt, _count_prompt_tokens(tokenizer, prompt)


# 从会话历史里丢弃最早的一轮对话，用于把上下文裁进窗口内。
def _drop_oldest_turn(messages: list[dict[str, str]], start_index: int) -> bool:
    # 若 start_index 已经越界，说明没有可删内容。
    if len(messages) <= start_index:
        return False

    # 删除最早的一条消息（通常是 user）。
    del messages[start_index]

    # 如果删完后该位置还是 assistant，则说明这是同一轮对话的回答，
    # 继续删除，保证“按轮”裁剪，而不是只删半轮。
    if len(messages) > start_index and messages[start_index]["role"] == "assistant":
        del messages[start_index]

    return True


# 在保留 system prompt 的前提下，把消息历史裁剪到 `max_model_len` 可接受范围。
def _fit_messages_to_context(
        tokenizer: Any,
        messages: list[dict[str, str]],
        args: Namespace,
) -> tuple[list[dict[str, str]], str, int]:
    # prompt token 预算 = 总上下文窗口 - 预留给新生成内容的 token 数。
    prompt_budget = args.max_model_len - args.max_new_tokens
    if prompt_budget <= 0:
        raise ValueError("max_model_len must be greater than max_new_tokens")

    # 拷贝一份 messages，避免原地修改调用方的历史数据。
    trimmed_messages = [dict(message) for message in messages]

    # 如果第一条是 system，则从索引 1 开始裁剪，确保 system prompt 永远保留。
    preserve_index = 1 if trimmed_messages and trimmed_messages[0]["role"] == "system" else 0

    # 初次渲染并计算 token 数。
    prompt, prompt_tokens = _render_prompt(tokenizer, trimmed_messages, args)

    # 若超过预算，就不断删除最早的一轮 user/assistant 对话，直到放得下。
    while prompt_tokens > prompt_budget:
        if not _drop_oldest_turn(trimmed_messages, preserve_index):
            break
        prompt, prompt_tokens = _render_prompt(tokenizer, trimmed_messages, args)

    # 如果已经删无可删，仍旧超长，则说明：
    # 1) system prompt 太长，或
    # 2) 当前用户输入本身太长。
    if prompt_tokens > prompt_budget:
        raise ValueError(
            "latest prompt still exceeds the configured context window; "
            "increase --max-model-len or shorten the current input"
        )

    return trimmed_messages, prompt, prompt_tokens


# 初始化会话的 system 消息列表。
def _initial_messages(args: Namespace) -> list[dict[str, str]]:
    # 未设置 system prompt，则初始历史为空。
    if not args.system_prompt:
        return []

    # 若指定了 system prompt，则作为第一条系统消息注入历史。
    return [{
        "role": "system",
        "content": args.system_prompt,
    }]


# 将 assistant 输出压成适合写回历史的版本，默认丢弃 think 块。
def _final_assistant_content(text: str, thinking_enabled: bool) -> str:
    if thinking_enabled:
        if _THINK_CLOSE_TOKEN in text:
            return text.rsplit(_THINK_CLOSE_TOKEN, 1)[1].strip()
        return ""
    return _THINK_BLOCK_RE.sub("", text).strip()


def _history_assistant_content(
        text: str,
        keep_thinking_in_history: bool,
        thinking_enabled: bool,
) -> str:
    if keep_thinking_in_history:
        return text

    return _final_assistant_content(text, thinking_enabled)


def _needs_answer_retry(text: str, thinking_enabled: bool) -> bool:
    return (
        thinking_enabled
        and _THINK_CLOSE_TOKEN in text
        and not _final_assistant_content(text, thinking_enabled)
    )


def _remaining_retry_tokens(
        tokenizer: Any,
        generated_text: str,
        args: Namespace,
) -> int:
    generated_ids = tokenizer(generated_text, add_special_tokens=False).input_ids
    return max(0, int(args.max_new_tokens) - len(generated_ids))


def _build_answer_retry_prompt(prompt: str, generated_text: str) -> str:
    retry_prompt = prompt + generated_text
    if generated_text.rstrip().endswith(_THINK_CLOSE_TOKEN):
        return retry_prompt.rstrip() + "\n\n"
    return retry_prompt


def _clone_sampling_params_with_max_tokens(
        sampling_params: Any,
        max_tokens: int,
) -> Any:
    cloned = sampling_params.clone() if hasattr(sampling_params, "clone") else copy.copy(
        sampling_params
    )
    cloned.max_tokens = max_tokens
    return cloned


def _stream_request_text(
        engine: Any,
        request_id: str,
        prompt: Any,
        sampling_params: Any,
        show_runtime_output: bool,
) -> list[str]:
    chunks: list[str] = []

    with _runtime_output_suppressed(show_runtime_output):
        engine.add_request(request_id, prompt, sampling_params)

    while engine.has_unfinished_requests():
        with _runtime_output_suppressed(show_runtime_output):
            outputs = engine.step()

        for text in _iter_request_text(outputs, request_id):
            chunks.append(text)
            print(text, end="", flush=True)

    return chunks


# 在真正进入交互前跑一次最小请求，把首轮固定冷启动前移到启动阶段。
def _maybe_warmup_chat_engine(
        engine: Any,
        args: Namespace,
) -> None:
    if not args.startup_warmup:
        return

    warmup_messages = _initial_messages(args) + [{
        "role": "user",
        "content": "warmup",
    }]
    warmup_prompt = _render_engine_chat_prompt(engine, warmup_messages, args)
    warmup_sampling_params = _build_sampling_params(args)
    warmup_sampling_params.max_tokens = 1

    # warmup 属于初始化阶段：按需求默认不抑制 runtime 输出。
    engine.add_request("native-chat-warmup", warmup_prompt, warmup_sampling_params)
    while engine.has_unfinished_requests():
        engine.step()


# 尝试获取流对象的文件描述符，供后续静默输出时重定向使用。
def _stream_fileno(stream: Any) -> int | None:
    try:
        return stream.fileno()
    # 某些流对象可能不支持 fileno，例如 StringIO；
    # 某些环境下也可能抛出 OSError / ValueError。
    except (AttributeError, io.UnsupportedOperation, OSError, ValueError):
        return None


# 临时把 stdout/stderr 重定向到 `/dev/null`，屏蔽底层运行时噪声。
@contextlib.contextmanager
def _suppress_stream_output() -> Iterator[None]:
    # 获取 stdout / stderr 对应的底层文件描述符。
    stdout_fd = _stream_fileno(sys.stdout)
    stderr_fd = _stream_fileno(sys.stderr)

    # 如果两者都拿不到，说明当前环境不支持 fd 级别重定向，直接放行。
    if stdout_fd is None and stderr_fd is None:
        yield
        return

    # 打开 /dev/null，用于吞掉运行时噪声输出。
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    # 备份原始 stdout/stderr fd，便于 finally 中恢复。
    stdout_dup = os.dup(stdout_fd) if stdout_fd is not None else None
    stderr_dup = os.dup(stderr_fd) if stderr_fd is not None else None

    try:
        # 先尽量 flush 一下，避免已有缓冲内容丢失或错位。
        with contextlib.suppress(Exception):
            sys.stdout.flush()
        with contextlib.suppress(Exception):
            sys.stderr.flush()

        # 将 stdout/stderr 重定向到 /dev/null。
        if stdout_fd is not None:
            os.dup2(devnull_fd, stdout_fd)
        if stderr_fd is not None:
            os.dup2(devnull_fd, stderr_fd)

        yield
    finally:
        # 退出上下文时再次 flush，尽量保证状态一致。
        with contextlib.suppress(Exception):
            sys.stdout.flush()
        with contextlib.suppress(Exception):
            sys.stderr.flush()

        # 恢复原来的 stdout/stderr。
        if stdout_fd is not None and stdout_dup is not None:
            os.dup2(stdout_dup, stdout_fd)
            os.close(stdout_dup)
        if stderr_fd is not None and stderr_dup is not None:
            os.dup2(stderr_dup, stderr_fd)
            os.close(stderr_dup)

        # 关闭 /dev/null fd，避免泄露。
        os.close(devnull_fd)


# 仅在“请求生成阶段”按 `show_runtime_output` 开关决定是否静默 runtime 输出。
@contextlib.contextmanager
def _runtime_output_suppressed(show_runtime_output: bool) -> Iterator[None]:
    # 若用户要求展示 runtime 输出，则不做任何抑制。
    if show_runtime_output:
        yield
        return

    # 否则同时抑制：
    # 1) warnings
    # 2) logging
    # 3) stdout/stderr 的底层输出
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with suppress_logging(logging.CRITICAL):
            with _suppress_stream_output():
                yield


# 运行交互式 chat 主循环：构建引擎、渲染 prompt、提交请求并流式打印输出。
def run_native_chat(args: Namespace) -> int:
    # 动态解析 runtime 依赖符号，避免模块导入阶段产生副作用。
    _, _, _, LLMEngine = _resolve_runtime_symbols()

    # 根据命令行参数构建 engine，并初始化。
    engine = LLMEngine.from_engine_args(
        _build_engine_args(args),
        enable_multiprocessing=args.enable_multiprocessing,
    )

    # 加载 tokenizer，后续用于模板渲染和 token 统计。
    tokenizer = _load_tokenizer(args)

    # 交互前先跑一次隐藏 warmup，把首轮固定冷启动前移。
    _maybe_warmup_chat_engine(engine, args)

    # 初始化会话历史（可能包含 system prompt）。
    messages = _initial_messages(args)

    # 构建采样参数，如 temperature、top_p、stop 等。
    sampling_params = _build_sampling_params(args)

    # 自增请求编号，用于区分每一轮请求。
    turn_index = 0

    # 在 stderr 打印交互提示信息。
    print(
        "Interactive chat ready. Commands: /clear, /quit",
        file=sys.stderr,
        flush=True,
    )

    try:
        while True:
            try:
                # 读取用户输入。
                user_text = input("user> ")
            except EOFError:
                # 例如用户按下 Ctrl-D，优雅退出。
                print(file=sys.stderr)
                return 0

            # 去掉首尾空白。
            user_text = user_text.strip()

            # 空输入直接忽略。
            if not user_text:
                continue

            # 退出命令。
            if user_text in _EXIT_COMMANDS:
                return 0

            # 清空历史命令，仅保留 system prompt。
            if user_text == _CLEAR_COMMAND:
                messages = _initial_messages(args)
                print("history cleared", file=sys.stderr, flush=True)
                continue

            # 构造“当前用户输入加入后的候选历史”。
            previous_messages = messages
            candidate_messages = messages + [{
                "role": "user",
                "content": user_text,
            }]

            try:
                # 尝试将消息裁剪到上下文窗口内，并得到最终 prompt。
                prompt_messages, prompt, _ = _fit_messages_to_context(
                    tokenizer,
                    candidate_messages,
                    args,
                )
            except ValueError as exc:
                # 若超出上下文限制，则提示用户，但不退出整个会话。
                print(f"context error: {exc}", file=sys.stderr, flush=True)
                continue

            # 为本轮生成唯一 request_id。
            request_id = f"native-chat-{turn_index}"
            turn_index += 1

            # 用于收集 assistant 的所有流式输出片段，便于最后写回历史。
            assistant_chunks: list[str] = []

            # 打印 assistant 前缀，后续流式内容会直接接在后面。
            print("assistant> ", end="", flush=True)

            engine_prompt = _render_engine_chat_prompt(engine, prompt_messages, args)
            assistant_chunks.extend(
                _stream_request_text(
                    engine,
                    request_id,
                    engine_prompt,
                    sampling_params,
                    args.show_runtime_output,
                )
            )

            assistant_text = "".join(assistant_chunks)
            if _needs_answer_retry(assistant_text, _thinking_mode_enabled(args)):
                retry_tokens = _remaining_retry_tokens(tokenizer, assistant_text, args)
                if retry_tokens > 0:
                    retry_sampling_params = _clone_sampling_params_with_max_tokens(
                        sampling_params,
                        retry_tokens,
                    )
                    assistant_chunks.extend(
                        _stream_request_text(
                            engine,
                            f"{request_id}-answer",
                            _build_answer_retry_prompt(prompt, assistant_text),
                            retry_sampling_params,
                            args.show_runtime_output,
                        )
                    )

            # 一轮输出结束后补一个换行。
            print(flush=True)

            history_content = _history_assistant_content(
                "".join(assistant_chunks),
                args.keep_thinking_in_history,
                _thinking_mode_enabled(args),
            )

            if history_content:
                # 将裁剪后的消息历史作为新的基础历史。
                # 这里的 prompt_messages 已经包含了当前 user 消息。
                messages = prompt_messages
                # 把 assistant 完整回答追加到历史中，供下一轮继续对话。
                messages.append({
                    "role": "assistant",
                    "content": history_content,
                })
            else:
                # 如果本轮只有 thinking 没有最终回答，则丢弃这一轮，
                # 避免“未闭合/无正文的思考输出”污染下一轮上下文。
                messages = previous_messages

    except KeyboardInterrupt:
        # Ctrl-C 时优雅中断。
        print("\ninterrupted", file=sys.stderr, flush=True)
        return 130
    finally:
        # 无论如何都尽量关闭底层 engine，避免资源泄漏。
        try:
            # 关闭属于退出阶段，默认不做 runtime 输出抑制。
            engine.engine_core.shutdown()
        except Exception:
            logger.exception("failed to shutdown native CFIE engine cleanly")
