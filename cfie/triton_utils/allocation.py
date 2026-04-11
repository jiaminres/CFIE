# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from cfie.logger import init_logger
from cfie.triton_utils import HAS_TRITON, triton

logger = init_logger(__name__)


def set_triton_allocator(device: torch.device):
    if not HAS_TRITON or not hasattr(triton, "set_allocator"):
        logger.warning_once(
            "Triton allocator API is unavailable in the current runtime; "
            "skipping Triton allocator setup."
        )
        return

    def alloc_fn(size: int, alignment: int, stream: int | None):
        return torch.empty(size, device=device, dtype=torch.int8)

    triton.set_allocator(alloc_fn)
