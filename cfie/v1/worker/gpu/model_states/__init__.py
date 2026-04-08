# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from cfie.config import CfieConfig
from cfie.v1.worker.gpu.mm.encoder_cache import EncoderCache


def init_model_state(
    cfie_config: CfieConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
):
    if "WhisperForConditionalGeneration" in cfie_config.model_config.architectures:
        from cfie.v1.worker.gpu.model_states.whisper import WhisperModelState

        return WhisperModelState(cfie_config, model, encoder_cache, device)

    from cfie.v1.worker.gpu.model_states.default import DefaultModelState

    return DefaultModelState(cfie_config, model, encoder_cache, device)
