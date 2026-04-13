# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from cfie.triton_utils import HAS_TRITON, tl, triton


def _fast_exp_reference(x):
    if isinstance(x, torch.Tensor):
        return torch.exp(x)
    return math.exp(x)


if HAS_TRITON:

    @triton.jit
    def fast_exp(x):
        """Faster alternative to tl.exp() using the hardware exp2 instruction.

        tl.math.exp2 maps directly to a single ex2.approx.f32 PTX instruction,
        while tl.exp goes through libdevice __nv_expf which adds function call
        overhead and extra range checking.
        """
        LOG2E = tl.constexpr(1.4426950408889634)
        return tl.math.exp2(LOG2E * x)
else:
    fast_exp = _fast_exp_reference
