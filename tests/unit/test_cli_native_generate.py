"""Unit tests for the native CFIE/v1 generation CLI."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

from cfie.cli.native_generate import _build_sampling_params, run_native_generate


class _FakeRequestOutputKind:
    DELTA = "delta"


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeEngineArgs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeEngine:
    def __init__(self):
        self.added = None
        self._step_count = 0
        self.engine_core = SimpleNamespace(shutdown=self._shutdown)
        self.shutdown_called = False

    @classmethod
    def from_engine_args(cls, engine_args, enable_multiprocessing=False):
        engine = cls()
        engine.engine_args = engine_args
        engine.enable_multiprocessing = enable_multiprocessing
        return engine

    def add_request(self, request_id, prompt, sampling_params):
        self.added = (request_id, prompt, sampling_params)

    def has_unfinished_requests(self):
        return self._step_count == 0

    def step(self):
        self._step_count += 1
        return [
            SimpleNamespace(
                request_id="native-generate",
                outputs=[SimpleNamespace(text="hello world")],
            )
        ]

    def _shutdown(self):
        self.shutdown_called = True


def _make_args(**overrides):
    base = dict(
        model="/tmp/model",
        prompt="ping",
        tokenizer=None,
        revision=None,
        download_dir=None,
        load_format="auto",
        dtype="auto",
        quantization=None,
        max_model_len=32768,
        max_new_tokens=16,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=None,
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
        mamba_cache_mode=None,
        language_model_only=False,
        skip_mm_profiling=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        presence_penalty=None,
        frequency_penalty=None,
        repetition_penalty=None,
        seed=None,
        stop=None,
        repetition_detection_max_pattern_size=0,
        repetition_detection_min_pattern_size=4,
        repetition_detection_min_count=3,
        enforce_eager=False,
        enable_multiprocessing=False,
        log_stats=False,
    )
    base.update(overrides)
    return Namespace(**base)


def _patch_runtime_symbols(
    monkeypatch,
    *,
    engine_args_cls=_FakeEngineArgs,
    sampling_params_cls=_FakeSamplingParams,
    request_output_kind_cls=_FakeRequestOutputKind,
    engine_cls=_FakeEngine,
):
    monkeypatch.setattr(
        "cfie.cli.native_generate._resolve_runtime_symbols",
        lambda: (
            engine_args_cls,
            sampling_params_cls,
            request_output_kind_cls,
            engine_cls,
        ),
    )
    monkeypatch.setattr(
        "cfie.cli.native_generate._render_engine_prompt",
        lambda engine, args: args.prompt,
    )


def test_native_generate_builds_native_engine_and_streams_output(
    monkeypatch,
    capsys,
):
    fake_engine_holder = {}

    class _EngineFactory(_FakeEngine):
        @classmethod
        def from_engine_args(cls, engine_args, enable_multiprocessing=False):
            engine = super().from_engine_args(engine_args, enable_multiprocessing)
            fake_engine_holder["engine"] = engine
            return engine

    _patch_runtime_symbols(
        monkeypatch,
        engine_args_cls=_FakeEngineArgs,
        sampling_params_cls=_FakeSamplingParams,
        request_output_kind_cls=_FakeRequestOutputKind,
        engine_cls=_EngineFactory,
    )

    code = run_native_generate(_make_args())

    assert code == 0
    stdout = capsys.readouterr().out
    assert "hello world" in stdout

    engine = fake_engine_holder["engine"]
    assert engine.added[0] == "native-generate"
    assert engine.added[1] == "ping"
    assert engine.added[2].kwargs["output_kind"] == "delta"
    assert engine.engine_args.kwargs["speculative_config"] == {
        "method": "mtp",
        "num_speculative_tokens": 1,
    }
    assert engine.engine_args.kwargs["dtype"] == "auto"
    assert engine.shutdown_called is True


def test_native_generate_omits_speculative_config_when_disabled(monkeypatch):
    captured = {}

    class _EngineArgsCapture(_FakeEngineArgs):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.update(kwargs)

    _patch_runtime_symbols(
        monkeypatch,
        engine_args_cls=_EngineArgsCapture,
        sampling_params_cls=_FakeSamplingParams,
        request_output_kind_cls=_FakeRequestOutputKind,
        engine_cls=_FakeEngine,
    )

    run_native_generate(_make_args(spec_method="none", num_speculative_tokens=None))

    assert captured["speculative_config"] is None


def test_native_generate_forces_float16_for_gptq_auto_dtype(monkeypatch):
    captured = {}

    class _EngineArgsCapture(_FakeEngineArgs):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.update(kwargs)

    _patch_runtime_symbols(
        monkeypatch,
        engine_args_cls=_EngineArgsCapture,
        sampling_params_cls=_FakeSamplingParams,
        request_output_kind_cls=_FakeRequestOutputKind,
        engine_cls=_FakeEngine,
    )

    run_native_generate(_make_args(quantization="gptq", dtype="auto"))

    assert captured["dtype"] == "float16"


def test_native_generate_passes_moe_cpu_planner_args(monkeypatch):
    captured = {}

    class _EngineArgsCapture(_FakeEngineArgs):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.update(kwargs)

    _patch_runtime_symbols(
        monkeypatch,
        engine_args_cls=_EngineArgsCapture,
        sampling_params_cls=_FakeSamplingParams,
        request_output_kind_cls=_FakeRequestOutputKind,
        engine_cls=_FakeEngine,
    )

    run_native_generate(
        _make_args(
            moe_cpu_budget_gb=24.0,
            moe_cpu_min_free_gb=18.0,
        )
    )

    assert captured["moe_cpu_budget_gb"] == 24.0
    assert captured["moe_cpu_min_free_gb"] == 18.0


def test_native_generate_passes_mm_and_cache_args(monkeypatch):
    captured = {}

    class _EngineArgsCapture(_FakeEngineArgs):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.update(kwargs)

    _patch_runtime_symbols(
        monkeypatch,
        engine_args_cls=_EngineArgsCapture,
        sampling_params_cls=_FakeSamplingParams,
        request_output_kind_cls=_FakeRequestOutputKind,
        engine_cls=_FakeEngine,
    )

    run_native_generate(
        _make_args(
            enable_prefix_caching=False,
            mamba_cache_mode="none",
            language_model_only=True,
            skip_mm_profiling=True,
        )
    )

    assert captured["enable_prefix_caching"] is False
    assert captured["mamba_cache_mode"] == "none"
    assert captured["language_model_only"] is True
    assert captured["skip_mm_profiling"] is True


def test_native_generate_passes_max_num_batched_tokens(monkeypatch):
    captured = {}

    class _EngineArgsCapture(_FakeEngineArgs):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.update(kwargs)

    _patch_runtime_symbols(
        monkeypatch,
        engine_args_cls=_EngineArgsCapture,
        sampling_params_cls=_FakeSamplingParams,
        request_output_kind_cls=_FakeRequestOutputKind,
        engine_cls=_FakeEngine,
    )

    run_native_generate(_make_args(max_num_batched_tokens=2048))

    assert captured["max_num_batched_tokens"] == 2048


def test_build_sampling_params_uses_generation_defaults_when_unset(monkeypatch):
    monkeypatch.setattr(
        "cfie.cli.native_generate._load_generation_sampling_defaults",
        lambda args: {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.05,
        },
    )

    params = _build_sampling_params(
        _make_args(
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
        )
    )

    assert params.temperature == 0.6
    assert params.top_p == 0.95
    assert params.top_k == 20
    assert params.repetition_penalty == 1.05
