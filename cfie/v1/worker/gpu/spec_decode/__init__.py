# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from cfie.config import CfieConfig


def init_speculator(cfie_config: CfieConfig, device: torch.device):
    speculative_config = cfie_config.speculative_config
    assert speculative_config is not None
    if speculative_config.use_eagle():
        from cfie.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

        return EagleSpeculator(cfie_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
