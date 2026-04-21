from __future__ import annotations

import pytest
import torch

from cfie_training.predictor.trainer import EngineRouterTeacherModelBackend


def _build_backend(
    layer_ids: tuple[int, ...],
) -> EngineRouterTeacherModelBackend:
    backend = object.__new__(EngineRouterTeacherModelBackend)
    backend._capture_layer_ids = layer_ids
    backend._capture_fragments_by_request = {}
    backend._engine = None
    backend._engine_capacity = None
    return backend


def test_merge_prefers_tp_rank_zero_duplicate_layers() -> None:
    backend = _build_backend((0, 1))

    rank_one_layer_zero = torch.tensor([[10.0]])
    rank_one_layer_one = torch.tensor([[11.0]])
    rank_zero_layer_zero = torch.tensor([[0.0]])
    rank_zero_layer_one = torch.tensor([[1.0]])

    merged = backend._merge_captured_hidden_states(
        [
            {
                "request-a": {
                    "layer_ids": (0, 1),
                    "hidden_states": (rank_one_layer_zero, rank_one_layer_one),
                    "pp_rank": 0,
                    "tp_rank": 1,
                }
            },
            {
                "request-a": {
                    "layer_ids": (0, 1),
                    "hidden_states": (rank_zero_layer_zero, rank_zero_layer_one),
                    "pp_rank": 0,
                    "tp_rank": 0,
                }
            },
        ]
    )

    assert torch.equal(merged["request-a"][0], rank_zero_layer_zero)
    assert torch.equal(merged["request-a"][1], rank_zero_layer_one)


def test_merge_accumulates_pp_layer_fragments_across_calls() -> None:
    backend = _build_backend((0, 1, 2))

    layer_zero = torch.tensor([[0.0]])
    layer_one = torch.tensor([[1.0]])
    layer_two = torch.tensor([[2.0]])

    first_poll = backend._merge_captured_hidden_states(
        [
            {
                "request-b": {
                    "layer_ids": (0,),
                    "hidden_states": (layer_zero,),
                    "pp_rank": 0,
                    "tp_rank": 0,
                }
            }
        ]
    )

    assert first_poll == {}
    assert set(backend._capture_fragments_by_request["request-b"]) == {0}

    second_poll = backend._merge_captured_hidden_states(
        [
            {
                "request-b": {
                    "layer_ids": (1, 2),
                    "hidden_states": (layer_one, layer_two),
                    "pp_rank": 1,
                    "tp_rank": 0,
                }
            }
        ]
    )

    assert torch.equal(second_poll["request-b"][0], layer_zero)
    assert torch.equal(second_poll["request-b"][1], layer_one)
    assert torch.equal(second_poll["request-b"][2], layer_two)
    assert "request-b" not in backend._capture_fragments_by_request


def test_merge_rejects_unstructured_worker_payload() -> None:
    backend = _build_backend((0, 1))

    layer_zero = torch.tensor([[0.0]])
    layer_one = torch.tensor([[1.0]])

    with pytest.raises(ValueError, match="payload must be a dict"):
        backend._merge_captured_hidden_states(
            [{"request-c": (layer_zero, layer_one)}]  # type: ignore[dict-item]
        )


def test_merge_converts_rpc_list_payload_to_tensors() -> None:
    backend = _build_backend((0, 1))

    merged = backend._merge_captured_hidden_states(
        [
            {
                "request-d": {
                    "layer_ids": (0, 1),
                    "hidden_states": (
                        [[0.0, 0.1]],
                        [[1.0, 1.1]],
                    ),
                    "pp_rank": 0,
                    "tp_rank": 0,
                }
            }
        ]
    )

    assert isinstance(merged["request-d"][0], torch.Tensor)
    assert torch.equal(merged["request-d"][0], torch.tensor([[0.0, 0.1]]))
    assert torch.equal(merged["request-d"][1], torch.tensor([[1.0, 1.1]]))
