from __future__ import annotations

from types import SimpleNamespace

from cfie.model_executor.models.interfaces import SupportsEagle3
from cfie.v1.worker.gpu_model_runner import GPUModelRunner


class _ProtocolOnlyWrapper(SupportsEagle3):
    def __init__(self, language_model):
        self.language_model = language_model


class _RealCaptureModel(SupportsEagle3):
    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, *args, **kwargs):
        raise AssertionError("predictor capture should not call language_model()")

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.calls.append(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return ()


def _build_fake_runner(root_model):
    return SimpleNamespace(
        get_model=lambda: root_model,
        predictor_capture_enabled=False,
        predictor_capture_layer_ids=(),
        _in_progress_predictor_hidden_states={},
        _ready_predictor_hidden_states={},
        use_aux_hidden_state_outputs=False,
        _resolve_predictor_capture_model=lambda: GPUModelRunner._resolve_predictor_capture_model(  # noqa: E501
            SimpleNamespace(get_model=lambda: root_model)
        ),
    )


def test_predictor_capture_prefers_nested_model_override() -> None:
    inner_model = _RealCaptureModel()
    outer_model = _ProtocolOnlyWrapper(language_model=inner_model)

    resolved = GPUModelRunner._resolve_predictor_capture_model(
        SimpleNamespace(get_model=lambda: outer_model)
    )

    assert resolved is inner_model


def test_predictor_capture_enable_and_disable_use_nested_override() -> None:
    inner_model = _RealCaptureModel()
    outer_model = _ProtocolOnlyWrapper(language_model=inner_model)
    runner = _build_fake_runner(outer_model)

    GPUModelRunner.enable_predictor_capture(runner, (2, 20, 37))

    assert runner.predictor_capture_enabled is True
    assert runner.predictor_capture_layer_ids == (2, 20, 37)
    assert runner.use_aux_hidden_state_outputs is True
    assert inner_model.calls == [(2, 20, 37)]

    GPUModelRunner.disable_predictor_capture(runner)

    assert runner.predictor_capture_enabled is False
    assert runner.predictor_capture_layer_ids == ()
    assert runner.use_aux_hidden_state_outputs is False
    assert inner_model.calls[-1] == ()


def test_predictor_capture_local_layers_follow_pp_bounds() -> None:
    backbone = SimpleNamespace(start_layer=4, end_layer=8)
    capture_model = SimpleNamespace(model=backbone)
    runner = SimpleNamespace(
        _resolve_predictor_capture_model=lambda: capture_model,
    )

    local_layer_ids = GPUModelRunner._resolve_predictor_capture_local_layer_ids(
        runner,
        tuple(range(12)),
    )

    assert local_layer_ids == (4, 5, 6, 7)
