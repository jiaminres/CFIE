from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cfie.model_executor.models.interfaces import supports_mrope
from cfie.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase


def test_qwen35_text_model_provides_text_only_mrope_positions() -> None:
    model = Qwen3_5ForCausalLMBase.__new__(Qwen3_5ForCausalLMBase)

    positions, delta = model.get_mrope_input_positions([11, 22, 33], [])

    assert supports_mrope(model)
    assert delta == 0
    assert positions.shape == (3, 3)
    assert torch.equal(positions[0], torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(positions[1], positions[0])
    assert torch.equal(positions[2], positions[0])


def test_qwen35_text_model_rejects_multimodal_mrope_features() -> None:
    model = Qwen3_5ForCausalLMBase.__new__(Qwen3_5ForCausalLMBase)

    with pytest.raises(
        ValueError,
        match="do not accept multimodal features",
    ):
        model.get_mrope_input_positions(
            [11, 22],
            [SimpleNamespace(modality="image")],
        )
