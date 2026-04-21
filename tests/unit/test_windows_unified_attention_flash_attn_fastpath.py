from __future__ import annotations

import torch

import cfie.v1.attention.ops.triton_unified_attention as unified_attn


class _KernelShouldNotRun:
    def __getitem__(self, _grid):
        def _launcher(**_kwargs):
            raise AssertionError("Triton unified attention kernel should not run")

        return _launcher


class _KernelSpy:
    def __init__(self) -> None:
        self.grid = None
        self.kwargs = None

    def __getitem__(self, grid):
        self.grid = grid

        def _launcher(**kwargs):
            self.kwargs = kwargs

        return _launcher


def _make_inputs():
    q = torch.randn(2, 4, 8)
    k = torch.randn(1, 16, 2, 8)
    v = torch.randn(1, 16, 2, 8)
    out = torch.empty_like(q)
    cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int32)
    seqused_k = torch.tensor([1, 1], dtype=torch.int32)
    block_table = torch.zeros((2, 1), dtype=torch.int32)
    return q, k, v, out, cu_seqlens_q, seqused_k, block_table


def _make_single_seq_inputs(*, q_len: int, seq_len: int):
    q = torch.ones(q_len, 1, 1, dtype=torch.float32)
    k = torch.ones(1, seq_len, 1, 1, dtype=torch.float32)
    v = torch.zeros(1, seq_len, 1, 1, dtype=torch.float32)
    out = torch.empty_like(q)
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32)
    seqused_k = torch.tensor([seq_len], dtype=torch.int32)
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    return q, k, v, out, cu_seqlens_q, seqused_k, block_table


def test_unified_attention_prefers_flash_attn_fastpath(monkeypatch) -> None:
    q, k, v, out, cu_seqlens_q, seqused_k, block_table = _make_inputs()
    calls: dict[str, object] = {}

    def _fake_flash_attn_varlen_func(**kwargs):
        calls.update(kwargs)
        kwargs["out"].fill_(5.0)

    monkeypatch.setattr(unified_attn, "is_batch_invariant", False)
    monkeypatch.setattr(unified_attn, "is_flash_attn_varlen_func_available", lambda: True)
    monkeypatch.setattr(unified_attn, "flash_attn_supports_sinks", lambda: False)
    monkeypatch.setattr(unified_attn, "get_flash_attn_version", lambda **_: 2)
    monkeypatch.setattr(
        unified_attn,
        "flash_attn_varlen_func",
        _fake_flash_attn_varlen_func,
    )
    monkeypatch.setattr(
        unified_attn,
        "kernel_unified_attention_2d",
        _KernelShouldNotRun(),
    )
    monkeypatch.setattr(
        unified_attn,
        "kernel_unified_attention_3d",
        _KernelShouldNotRun(),
    )

    unified_attn.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=1,
        softmax_scale=0.25,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=torch.ones((2, 2), dtype=torch.float32),
        v_descale=torch.ones((2, 2), dtype=torch.float32),
    )

    assert calls["fa_version"] == 2
    assert calls["block_table"] is block_table
    assert calls["seqused_k"] is seqused_k
    assert calls["softmax_scale"] == 0.25
    assert torch.equal(out, torch.full_like(out, 5.0))


def test_unified_attention_keeps_triton_path_when_qq_bias_present(monkeypatch) -> None:
    q, k, v, out, cu_seqlens_q, seqused_k, block_table = _make_inputs()
    kernel_spy = _KernelSpy()

    monkeypatch.setattr(unified_attn, "HAS_TRITON", True)
    monkeypatch.setattr(unified_attn, "is_batch_invariant", False)
    monkeypatch.setattr(unified_attn, "is_flash_attn_varlen_func_available", lambda: True)
    monkeypatch.setattr(unified_attn, "get_flash_attn_version", lambda **_: 2)
    monkeypatch.setattr(
        unified_attn,
        "flash_attn_varlen_func",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("flash-attn fastpath should be disabled for qq_bias")
        ),
    )
    monkeypatch.setattr(unified_attn, "kernel_unified_attention_2d", kernel_spy)

    unified_attn.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=1,
        softmax_scale=0.25,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=torch.ones((2, 2), dtype=torch.float32),
        v_descale=torch.ones((2, 2), dtype=torch.float32),
        qq_bias=torch.zeros((1, 1), dtype=torch.float32),
    )

    assert kernel_spy.grid == (2, 2)
    assert kernel_spy.kwargs is not None


def test_unified_attention_uses_reference_fallback_without_triton(monkeypatch) -> None:
    q, k, v, out, cu_seqlens_q, seqused_k, block_table = _make_inputs()
    calls: dict[str, object] = {}

    monkeypatch.setattr(unified_attn, "HAS_TRITON", False)
    monkeypatch.setattr(unified_attn, "is_flash_attn_varlen_func_available", lambda: False)
    monkeypatch.setattr(
        unified_attn,
        "_reference_unified_attention",
        lambda **kwargs: calls.update(kwargs),
    )
    monkeypatch.setattr(
        unified_attn,
        "kernel_unified_attention_2d",
        _KernelShouldNotRun(),
    )
    monkeypatch.setattr(
        unified_attn,
        "kernel_unified_attention_3d",
        _KernelShouldNotRun(),
    )

    unified_attn.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=1,
        softmax_scale=0.25,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=torch.ones((2, 2), dtype=torch.float32),
        v_descale=torch.ones((2, 2), dtype=torch.float32),
        qq_bias=torch.zeros((1, 1), dtype=torch.float32),
    )

    assert calls["qq_bias"] is not None
    assert calls["block_table"] is block_table
    assert calls["seqused_k"] is seqused_k


def test_unified_attention_uses_paged_attention_v1_fastpath_without_triton(
    monkeypatch,
) -> None:
    q, k, v, out, cu_seqlens_q, seqused_k, block_table = _make_inputs()
    calls: dict[str, object] = {}

    def _fake_paged_attention_v1(**kwargs):
        calls.update(kwargs)
        kwargs["out"].fill_(7.0)

    monkeypatch.setattr(unified_attn, "HAS_TRITON", False)
    monkeypatch.setattr(unified_attn, "is_flash_attn_varlen_func_available", lambda: False)
    monkeypatch.setattr(
        unified_attn,
        "_reference_unified_attention",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("reference fallback should not run")
        ),
    )
    monkeypatch.setattr(unified_attn.ops, "paged_attention_v1", _fake_paged_attention_v1)

    unified_attn.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=1,
        softmax_scale=0.25,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=torch.ones((2, 2), dtype=torch.float32),
        v_descale=torch.ones((2, 2), dtype=torch.float32),
    )

    assert calls["num_kv_heads"] == 2
    assert calls["query"].shape == q.shape
    assert calls["key_cache"].shape == (1, 2, 2, 16, 4)
    assert calls["value_cache"].shape == (1, 2, 8, 16)
    assert torch.equal(calls["block_tables"], torch.zeros_like(block_table))
    assert torch.equal(calls["seq_lens"], seqused_k)
    assert calls["kv_cache_dtype"] == "auto"
    assert torch.equal(out, torch.full_like(out, 7.0))


def test_unified_attention_decode_fastpath_skips_prefill_without_triton(
    monkeypatch,
) -> None:
    q, k, v, out, cu_seqlens_q, seqused_k, block_table = _make_single_seq_inputs(
        q_len=2,
        seq_len=2,
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(unified_attn, "HAS_TRITON", False)
    monkeypatch.setattr(unified_attn, "is_flash_attn_varlen_func_available", lambda: False)
    monkeypatch.setattr(
        unified_attn.ops,
        "paged_attention_v1",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("paged_attention_v1 fastpath should not run for prefill")
        ),
    )
    monkeypatch.setattr(
        unified_attn,
        "_reference_unified_attention",
        lambda **kwargs: calls.update(kwargs),
    )

    unified_attn.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=2,
        seqused_k=seqused_k,
        max_seqlen_k=2,
        softmax_scale=1.0,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )

    assert calls["block_table"] is block_table
    assert calls["seqused_k"] is seqused_k
    assert calls["softmax_scale"] == 1.0


def test_reference_fallback_applies_qq_bias_semantics(monkeypatch) -> None:
    q, k, v, out, cu_seqlens_q, seqused_k, block_table = _make_single_seq_inputs(
        q_len=2,
        seq_len=2,
    )
    v[0, 0, 0, 0] = 1.0
    v[0, 1, 0, 0] = 2.0
    qq_bias = torch.tensor(
        [
            [0.0, float("-inf")],
            [float("-inf"), 0.0],
        ],
        dtype=torch.float32,
    )

    monkeypatch.setattr(unified_attn, "HAS_TRITON", False)
    monkeypatch.setattr(unified_attn, "is_flash_attn_varlen_func_available", lambda: False)

    unified_attn.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=2,
        seqused_k=seqused_k,
        max_seqlen_k=2,
        softmax_scale=1.0,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        qq_bias=qq_bias,
    )

    assert torch.allclose(out[:, 0, 0], torch.tensor([1.0, 2.0]))


def test_reference_fallback_supports_mm_prefix_and_output_scale(
    monkeypatch,
) -> None:
    q, k, v, out, cu_seqlens_q, seqused_k, block_table = _make_single_seq_inputs(
        q_len=2,
        seq_len=4,
    )
    v[0, 3, 0, 0] = 4.0
    mm_prefix_range = torch.tensor([[[2, 3]]], dtype=torch.int32)

    monkeypatch.setattr(unified_attn, "HAS_TRITON", False)
    monkeypatch.setattr(unified_attn, "is_flash_attn_varlen_func_available", lambda: False)

    unified_attn.unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=2,
        seqused_k=seqused_k,
        max_seqlen_k=4,
        softmax_scale=1.0,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        mm_prefix_range=mm_prefix_range,
        output_scale=torch.tensor([[2.0]], dtype=torch.float32),
    )

    assert torch.allclose(
        out[:, 0, 0],
        torch.tensor([0.5, 0.5], dtype=torch.float32),
        atol=1e-5,
    )
