# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from cfie import _custom_ops
from cfie.v1.attention.ops import common


def _require_cuda_precompiled() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not _custom_ops.has_precompiled_correct_attn_out():
        pytest.skip("Precompiled op is not available in this build")


def _reference_correct_attn_out(
    out: torch.Tensor,
    lses: torch.Tensor,
    cp_rank: int,
    is_lse_base_on_e: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_ref = out.clone()
    neg_inf = lses.new_full((), -float("inf"))
    sanitized = torch.where(torch.isnan(lses) | torch.isinf(lses), neg_inf, lses)
    lse_max = sanitized.amax(dim=0)
    lse_max = torch.where(lse_max == neg_inf, torch.zeros_like(lse_max), lse_max)
    shifted = sanitized - lse_max.unsqueeze(0)
    if is_lse_base_on_e:
        final_lse = torch.log(torch.exp(shifted).sum(dim=0)) + lse_max
        factor = torch.exp(sanitized[cp_rank] - final_lse)
    else:
        final_lse = torch.log2(torch.exp2(shifted).sum(dim=0)) + lse_max
        factor = torch.exp2(sanitized[cp_rank] - final_lse)
    factor = torch.where(torch.isnan(factor) | torch.isinf(factor), 0.0, factor)
    out_ref.mul_(factor.unsqueeze(-1).to(out_ref.dtype))
    return out_ref, final_lse


@pytest.mark.parametrize("is_lse_base_on_e", [True, False])
@pytest.mark.parametrize(
    ("out_shape", "lses_shape"),
    [
        ("3d", "3d"),
        ("4d", "n_b_h_1"),
        ("4d", "n_1_b_h"),
    ],
)
def test_correct_attn_out_precompiled_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    is_lse_base_on_e: bool,
    out_shape: str,
    lses_shape: str,
) -> None:
    _require_cuda_precompiled()
    monkeypatch.setattr(common, "HAS_TRITON", False)

    base_out = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    base_lses = torch.tensor(
        [
            [[0.0, float("nan")], [2.0, -1.0]],
            [[1.0, float("inf")], [0.5, -0.5]],
            [[-2.0, 1.5], [0.0, -3.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    cp_rank = 1

    out = base_out.clone()
    if out_shape == "4d":
        out = out.unsqueeze(1)

    lses = base_lses.clone()
    if lses_shape == "n_b_h_1":
        lses = lses.unsqueeze(-1)
    elif lses_shape == "n_1_b_h":
        lses = lses.unsqueeze(1)

    actual_out, actual_lse = common.correct_attn_out(
        out,
        lses,
        cp_rank,
        ctx=None,
        is_lse_base_on_e=is_lse_base_on_e,
    )
    expected_out, expected_lse = _reference_correct_attn_out(
        base_out,
        base_lses,
        cp_rank,
        is_lse_base_on_e,
    )

    torch.testing.assert_close(actual_out.cpu(), expected_out.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_lse.cpu(), expected_lse.cpu(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("is_lse_base_on_e", [True, False])
def test_correct_attn_out_torch_fallback_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    is_lse_base_on_e: bool,
) -> None:
    monkeypatch.setattr(common, "HAS_TRITON", False)
    monkeypatch.setattr(_custom_ops, "has_precompiled_correct_attn_out", lambda: False)

    out = torch.tensor(
        [[[1.0, 0.5], [2.0, 1.5]]],
        dtype=torch.float32,
    )
    lses = torch.tensor(
        [
            [[0.0, 1.0]],
            [[-1.0, float("inf")]],
        ],
        dtype=torch.float32,
    )

    actual_out, actual_lse = common.correct_attn_out(
        out.clone(),
        lses.clone(),
        cp_rank=0,
        ctx=None,
        is_lse_base_on_e=is_lse_base_on_e,
    )
    expected_out, expected_lse = _reference_correct_attn_out(
        out,
        lses,
        cp_rank=0,
        is_lse_base_on_e=is_lse_base_on_e,
    )

    torch.testing.assert_close(actual_out, expected_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-5)
