# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

from cfie.triton_utils import tl, tldevice, triton

from .utils import is_gather_supported


def _make_unavailable_triton_helper(name: str):
    def _helper(*args, **kwargs):
        raise RuntimeError(
            f"{name} requires a Triton runtime. "
            "This placeholder exists only to keep the import path "
            "stable when Triton is unavailable."
        )

    _helper.__name__ = f"{name}_unavailable"
    return _helper


def _resolve_triton_helper(*candidates, name: str):
    for candidate in candidates:
        if callable(candidate):
            return candidate
    return _make_unavailable_triton_helper(name)


if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    exp = _resolve_triton_helper(
        getattr(tldevice, "fast_expf", None),
        getattr(tl, "exp", None),
        name="exp",
    )
    log = _resolve_triton_helper(
        getattr(tldevice, "fast_logf", None),
        getattr(tl, "log", None),
        name="log",
    )
    log2 = _resolve_triton_helper(
        getattr(tldevice, "fast_log2f", None),
        getattr(tl, "log2", None),
        name="log2",
    )
else:
    exp = _resolve_triton_helper(getattr(tl, "exp", None), name="exp")
    log = _resolve_triton_helper(getattr(tl, "log", None), name="log")
    log2 = _resolve_triton_helper(getattr(tl, "log2", None), name="log2")


if not is_gather_supported:

    @triton.jit
    def gather(src, index, axis, _builder=None):
        """
        Gather operation that works when tl.gather is not supported.
        This is a fallback implementation that returns None.
        Just to make triton compiler happy.
        """
        return None
else:
    gather = tl.gather

if hasattr(triton.language, "_experimental_make_tensor_descriptor"):
    # For Triton 3.3.x
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, "make_tensor_descriptor"):
    # For Triton 3.4.x and later
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:
    """
    Fallback implementation when TMA is not supported.
    Returns None to indicate TMA descriptors are unavailable.
    Just make triton compiler happy.
    """

    @triton.jit
    def make_tensor_descriptor(
        base,
        shape,
        strides,
        block_shape,
        _builder=None,
    ):
        return None
