"""Unit tests for the interactive native CFIE/v1 chat CLI."""

from __future__ import annotations

import contextlib
from argparse import Namespace
from types import SimpleNamespace

from cfie.cli.native_chat import (
    _fit_messages_to_context,
    _history_assistant_content,
    run_native_chat,
)


class _FakeRequestOutputKind:
    DELTA = "delta"


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.max_tokens = kwargs.get("max_tokens")

    def clone(self):
        return _FakeSamplingParams(**self.kwargs)


class _FakeEngineArgs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeRenderer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def render_chat(self, conversations, chat_params):
        prompts = [
            self.tokenizer.apply_chat_template(
                conversation,
                **chat_params.chat_template_kwargs,
            ) for conversation in conversations
        ]
        return conversations, prompts


class _FakeEngine:
    renderer_tokenizer = None

    def __init__(self):
        self.requests = []
        self._step_count = 0
        self._current_request_id = None
        self.engine_core = SimpleNamespace(shutdown=self._shutdown)
        self.shutdown_called = False
        self.renderer = _FakeRenderer(self.renderer_tokenizer)

    @classmethod
    def from_engine_args(cls, engine_args, enable_multiprocessing=False):
        engine = cls()
        engine.engine_args = engine_args
        engine.enable_multiprocessing = enable_multiprocessing
        return engine

    def add_request(self, request_id, prompt, sampling_params):
        self.requests.append((request_id, prompt, sampling_params))
        self._current_request_id = request_id
        self._step_count = 0

    def has_unfinished_requests(self):
        return self._step_count == 0

    def step(self):
        self._step_count += 1
        return [
            SimpleNamespace(
                request_id=self._current_request_id,
                outputs=[SimpleNamespace(text=f"reply-{len(self.requests)}")],
            )
        ]

    def _shutdown(self):
        self.shutdown_called = True


class _FakeTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.render_calls = []

    def apply_chat_template(self, conversation, **kwargs):
        self.render_calls.append({
            "conversation": [dict(message) for message in conversation],
            "kwargs": dict(kwargs),
        })
        prompt = "|".join(
            f"{message['role']}:{message['content']}" for message in conversation
        )
        if kwargs.get("add_generation_prompt", False):
            prompt += "|assistant:"
        return prompt

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=list(text))


def _make_args(**overrides):
    base = dict(
        model="/tmp/model",
        tokenizer=None,
        revision=None,
        download_dir=None,
        load_format="auto",
        dtype="auto",
        quantization=None,
        max_model_len=128,
        max_new_tokens=16,
        gpu_memory_utilization=0.9,
        moe_cpu_budget_gb=0.0,
        moe_cpu_min_free_gb=0.0,
        cpu_offload_gb=0.0,
        offload_backend="auto",
        tensor_parallel_size=1,
        max_num_seqs=1,
        max_num_batched_tokens=None,
        spec_method="mtp",
        num_speculative_tokens=1,
        attention_backend=None,
        moe_backend="auto",
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        presence_penalty=None,
        frequency_penalty=None,
        repetition_penalty=None,
        seed=None,
        stop=None,
        system_prompt="You are terse.",
        enable_thinking=False,
        keep_thinking_in_history=False,
        repetition_detection_max_pattern_size=64,
        repetition_detection_min_pattern_size=4,
        repetition_detection_min_count=3,
        show_runtime_output=False,
        startup_warmup=False,
        enforce_eager=False,
        enable_multiprocessing=False,
        log_stats=False,
        log_level="INFO",
    )
    base.update(overrides)
    return Namespace(**base)


def test_native_chat_streams_output_and_forwards_thinking_flag(
    monkeypatch,
    capsys,
):
    fake_engine_holder = {}
    fake_tokenizer = _FakeTokenizer()
    _FakeEngine.renderer_tokenizer = fake_tokenizer
    inputs = iter(["hello", "/quit"])

    class _EngineFactory(_FakeEngine):
        @classmethod
        def from_engine_args(cls, engine_args, enable_multiprocessing=False):
            engine = super().from_engine_args(engine_args, enable_multiprocessing)
            fake_engine_holder["engine"] = engine
            return engine

    runtime_symbols = lambda: (
        _FakeEngineArgs,
        _FakeSamplingParams,
        _FakeRequestOutputKind,
        _EngineFactory,
    )
    monkeypatch.setattr("cfie.cli.native_generate._resolve_runtime_symbols",
                        runtime_symbols)
    monkeypatch.setattr("cfie.cli.native_chat._resolve_runtime_symbols",
                        runtime_symbols)
    monkeypatch.setattr("cfie.cli.native_chat._load_tokenizer",
                        lambda args: fake_tokenizer)
    monkeypatch.setattr(
        "cfie.cli.native_chat._build_renderer_chat_params",
        lambda args: SimpleNamespace(
            chat_template_kwargs={"enable_thinking": args.enable_thinking}
        ),
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    code = run_native_chat(_make_args())

    assert code == 0
    stdout, stderr = capsys.readouterr()
    assert "assistant> reply-1" in stdout
    assert "Interactive chat ready" in stderr

    engine = fake_engine_holder["engine"]
    request_id, prompt, sampling_params = engine.requests[0]
    assert request_id == "native-chat-0"
    assert "system:You are terse." in prompt
    assert "user:hello" in prompt
    assert sampling_params.kwargs["output_kind"] == "delta"
    assert engine.shutdown_called is True
    assert fake_tokenizer.render_calls[0]["kwargs"]["enable_thinking"] is False


def test_native_chat_suppresses_runtime_output_by_default(
    monkeypatch,
):
    fake_tokenizer = _FakeTokenizer()
    _FakeEngine.renderer_tokenizer = fake_tokenizer
    inputs = iter(["hello", "/quit"])
    suppression_calls = []

    runtime_symbols = lambda: (
        _FakeEngineArgs,
        _FakeSamplingParams,
        _FakeRequestOutputKind,
        _FakeEngine,
    )

    @contextlib.contextmanager
    def _fake_suppress(show_runtime_output):
        suppression_calls.append(show_runtime_output)
        yield

    monkeypatch.setattr("cfie.cli.native_generate._resolve_runtime_symbols",
                        runtime_symbols)
    monkeypatch.setattr("cfie.cli.native_chat._resolve_runtime_symbols",
                        runtime_symbols)
    monkeypatch.setattr("cfie.cli.native_chat._load_tokenizer",
                        lambda args: fake_tokenizer)
    monkeypatch.setattr(
        "cfie.cli.native_chat._build_renderer_chat_params",
        lambda args: SimpleNamespace(
            chat_template_kwargs={"enable_thinking": args.enable_thinking}
        ),
    )
    monkeypatch.setattr("cfie.cli.native_chat._runtime_output_suppressed",
                        _fake_suppress)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    code = run_native_chat(_make_args())

    assert code == 0
    assert len(suppression_calls) >= 2
    assert set(suppression_calls) == {False}


def test_fit_messages_to_context_drops_oldest_turns():
    tokenizer = _FakeTokenizer()
    args = _make_args(max_model_len=72, max_new_tokens=12)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old question that should be trimmed"},
        {"role": "assistant", "content": "old answer that should be trimmed"},
        {"role": "user", "content": "latest"},
    ]

    trimmed_messages, prompt, prompt_tokens = _fit_messages_to_context(
        tokenizer,
        messages,
        args,
    )

    assert prompt_tokens <= args.max_model_len - args.max_new_tokens
    assert trimmed_messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "latest"},
    ]
    assert "old question" not in prompt


def test_history_assistant_content_drops_thinking_only_text():
    assert (
        _history_assistant_content(
            "这里是思考过程</think>\n",
            keep_thinking_in_history=False,
            thinking_enabled=True,
        )
        == ""
    )


def test_native_chat_retries_when_thinking_has_no_final_answer(
    monkeypatch,
    capsys,
):
    fake_tokenizer = _FakeTokenizer()
    _FakeEngine.renderer_tokenizer = fake_tokenizer
    inputs = iter(["hello", "/quit"])

    class _ThinkingRetryEngine(_FakeEngine):
        def step(self):
            self._step_count += 1
            reply_text = (
                "先想一下</think>\n"
                if self._current_request_id == "native-chat-0"
                else "最终回答"
            )
            return [
                SimpleNamespace(
                    request_id=self._current_request_id,
                    outputs=[SimpleNamespace(text=reply_text)],
                )
            ]

    runtime_symbols = lambda: (
        _FakeEngineArgs,
        _FakeSamplingParams,
        _FakeRequestOutputKind,
        _ThinkingRetryEngine,
    )
    monkeypatch.setattr("cfie.cli.native_generate._resolve_runtime_symbols",
                        runtime_symbols)
    monkeypatch.setattr("cfie.cli.native_chat._resolve_runtime_symbols",
                        runtime_symbols)
    monkeypatch.setattr("cfie.cli.native_chat._load_tokenizer",
                        lambda args: fake_tokenizer)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    code = run_native_chat(_make_args(enable_thinking=True))

    assert code == 0
    stdout = capsys.readouterr().out
    assert "最终回答" in stdout
