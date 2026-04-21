# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from cfie import _custom_ops
from cfie.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch


def _require_cuda_precompiled() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not _custom_ops.has_precompiled_apply_top_k_top_p():
        pytest.skip("Precompiled op is not available in this build")


@pytest.mark.parametrize(
    ("k", "p"),
    [
        (torch.tensor([3, 4, 2], device="cuda", dtype=torch.int32), None),
        (torch.tensor([6, 4, 6], device="cuda", dtype=torch.int32), None),
        (None, torch.tensor([0.80, 0.65, 0.95], device="cuda", dtype=torch.float32)),
        (None, torch.tensor([1.00, 0.65, 1.00], device="cuda", dtype=torch.float32)),
        (
            torch.tensor([3, 2, 4], device="cuda", dtype=torch.int32),
            torch.tensor([0.80, 0.70, 0.90], device="cuda", dtype=torch.float32),
        ),
        (
            torch.tensor([6, 2, 4], device="cuda", dtype=torch.int32),
            torch.tensor([0.85, 0.70, 0.90], device="cuda", dtype=torch.float32),
        ),
        (
            torch.tensor([6, 2, 6], device="cuda", dtype=torch.int32),
            torch.tensor([1.00, 0.70, 1.00], device="cuda", dtype=torch.float32),
        ),
    ],
)
def test_apply_top_k_top_p_precompiled_matches_pytorch(
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> None:
    _require_cuda_precompiled()

    logits = torch.tensor(
        [
            [0.1, -1.2, 2.4, 0.7, -0.3, 1.8],
            [1.3, 0.2, -0.5, 2.1, 1.1, -1.4],
            [-0.7, 0.9, 1.6, -2.0, 0.4, 1.2],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    actual = logits.clone()
    expected = logits.clone()

    _custom_ops.apply_top_k_top_p_precompiled(actual, k, p)
    expected = apply_top_k_top_p_pytorch(expected, k, p)

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0.0, atol=0.0)
