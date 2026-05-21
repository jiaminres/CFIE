"""Unit tests for GPTQ-cache requantization placeholders."""

from __future__ import annotations

import pytest
import torch

from cfie_training.training_base import (
    GptqCacheRecord,
    GptqMarlinBundleRequantizer,
    GptqCacheRequantizer,
    GptqCacheStore,
    decode_gptq_marlin_bundle,
    encode_gptq_marlin_bundle_sections,
    SymmetricInt4GptqCodec,
    SymmetricInt4GptqLayout,
    gptq_bundle_num_bytes,
    gptq_layout_hash,
)
from cfie_training.training_base.gptq_checkpoint import (
    pack_gptq_int4_qweight,
    pack_gptq_int4_qzeros,
)


def test_symmetric_int4_codec_round_trips_with_expected_size() -> None:
    layout = SymmetricInt4GptqLayout(group_size=4)
    codec = SymmetricInt4GptqCodec(layout)
    values = torch.tensor(
        [0.0, 1.0, -2.0, 3.5, -4.0, 5.0, -6.0],
        dtype=torch.float32,
    )

    payload = codec.encode(values)
    decoded = codec.decode(payload, values.numel())

    assert len(payload) == gptq_bundle_num_bytes(values.numel(), group_size=4)
    assert len(payload) == 12
    assert torch.allclose(decoded, values, rtol=0.16, atol=0.16)
    assert codec.layout_hash == gptq_layout_hash(group_size=4)


def test_symmetric_int4_codec_rejects_wrong_payload_size() -> None:
    codec = SymmetricInt4GptqCodec(SymmetricInt4GptqLayout(group_size=4))

    with pytest.raises(ValueError, match="expected"):
        codec.decode(b"short", num_elements=8)


def test_gptq_requantizer_updates_only_touched_mapped_params(tmp_path) -> None:
    layout = SymmetricInt4GptqLayout(group_size=4)
    codec = SymmetricInt4GptqCodec(layout)
    store = GptqCacheStore.create(
        tmp_path,
        {
            "bundle_a": GptqCacheRecord(
                "bundle_a",
                "gptq_0000.bin",
                0,
                codec.payload_num_bytes(4),
                quant_layout_hash=codec.layout_hash,
            ),
            "bundle_b": GptqCacheRecord(
                "bundle_b",
                "gptq_0000.bin",
                codec.payload_num_bytes(4),
                codec.payload_num_bytes(4),
                quant_layout_hash=codec.layout_hash,
            ),
        },
    )
    requantizer = GptqCacheRequantizer(
        store=store,
        param_to_bundle={
            "param_a": "bundle_a",
            "param_b": "bundle_b",
        },
        codec=codec,
    )

    updates = requantizer.requantize_touched(
        {
            "param_a": torch.tensor([1.0, -2.0, 3.0, -4.0]),
            "param_b": torch.tensor([5.0, -6.0, 7.0, -8.0]),
            "dense_param": torch.ones(4),
        },
        touched_param_ids=("param_a", "dense_param"),
    )
    store.flush_touched(updates, generation=1)

    assert set(updates) == {"bundle_a"}
    assert codec.decode(store.read_bundle("bundle_a"), 4).shape == (4,)
    assert store.read_bundle("bundle_b") == b"\0" * codec.payload_num_bytes(4)


def test_gptq_requantizer_rejects_layout_mismatch(tmp_path) -> None:
    codec = SymmetricInt4GptqCodec(SymmetricInt4GptqLayout(group_size=4))
    store = GptqCacheStore.create(
        tmp_path,
        {
            "bundle_a": GptqCacheRecord(
                "bundle_a",
                "gptq_0000.bin",
                0,
                codec.payload_num_bytes(4),
                quant_layout_hash="sha256:wrong",
            ),
        },
    )
    requantizer = GptqCacheRequantizer(
        store=store,
        param_to_bundle={"param_a": "bundle_a"},
        codec=codec,
    )

    with pytest.raises(ValueError, match="layout mismatch"):
        requantizer.requantize_touched(
            {"param_a": torch.ones(4)},
            touched_param_ids=("param_a",),
        )


def test_gptq_requantizer_rejects_payload_size_mismatch(tmp_path) -> None:
    codec = SymmetricInt4GptqCodec(SymmetricInt4GptqLayout(group_size=4))
    store = GptqCacheStore.create(
        tmp_path,
        {
            "bundle_a": GptqCacheRecord(
                "bundle_a",
                "gptq_0000.bin",
                0,
                codec.payload_num_bytes(4) + 1,
                quant_layout_hash=codec.layout_hash,
            ),
        },
    )
    requantizer = GptqCacheRequantizer(
        store=store,
        param_to_bundle={"param_a": "bundle_a"},
        codec=codec,
    )

    with pytest.raises(ValueError, match="requantized payload"):
        requantizer.requantize_touched(
            {"param_a": torch.ones(4)},
            touched_param_ids=("param_a",),
        )


def test_gptq_requantizer_can_require_mapping_for_touched_params(tmp_path) -> None:
    store = GptqCacheStore.create(tmp_path, {})
    requantizer = GptqCacheRequantizer(
        store=store,
        param_to_bundle={},
        require_mapping_for_touched=True,
    )

    with pytest.raises(KeyError, match="missing GPTQ bundle mapping"):
        requantizer.requantize_touched(
            {"param_a": torch.ones(4)},
            touched_param_ids=("param_a",),
        )


def test_marlin_bundle_requantizer_updates_single_bundle_schema(tmp_path) -> None:
    initial_payload = encode_gptq_marlin_bundle_sections(
        bundle_id="bundle_a",
        group_size=2,
        size_k=4,
        size_n=2,
        sections={
            "qweight": pack_gptq_int4_qweight(torch.full((4, 2), 8)),
            "scales": torch.ones(2, 2, dtype=torch.float32),
            "qzeros": pack_gptq_int4_qzeros(torch.full((2, 2), 7)),
            "g_idx": torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        },
        act_order=True,
    )
    layout_hash = decode_gptq_marlin_bundle(initial_payload).metadata.layout_hash
    store = GptqCacheStore.create(
        tmp_path,
        {
            "bundle_a": GptqCacheRecord(
                "bundle_a",
                "gptq_0000.bin",
                0,
                len(initial_payload),
                quant_layout_hash=layout_hash,
            ),
        },
    )
    store.flush_touched({"bundle_a": initial_payload}, generation=0)
    requantizer = GptqMarlinBundleRequantizer(
        store=store,
        param_to_bundle={"param_a": "bundle_a"},
    )

    updates = requantizer.requantize_touched(
        {
            "param_a": torch.tensor(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                dtype=torch.float32,
            ),
        },
        touched_param_ids=("param_a",),
    )

    decoded = decode_gptq_marlin_bundle(updates["bundle_a"])

    assert len(updates["bundle_a"]) == len(initial_payload)
    assert decoded.metadata.layout_hash == layout_hash
    assert set(decoded.tensors) == {"g_idx", "qweight", "qzeros", "scales"}
    assert decoded.tensors["qweight"].shape == (1, 2)
    assert decoded.tensors["scales"].shape == (2, 2)
    assert not torch.equal(
        decoded.tensors["qweight"],
        torch.zeros_like(decoded.tensors["qweight"]),
    )


def test_marlin_bundle_requantizer_updates_prefixed_w13_sections(tmp_path) -> None:
    base_sections = {
        "w1.qweight": pack_gptq_int4_qweight(torch.full((2, 2), 8)),
        "w1.scales": torch.ones(1, 2, dtype=torch.float32),
        "w1.qzeros": pack_gptq_int4_qzeros(torch.full((1, 2), 7)),
        "w1.g_idx": torch.tensor([0, 0], dtype=torch.int32),
        "w3.qweight": pack_gptq_int4_qweight(torch.full((2, 2), 8)),
        "w3.scales": torch.ones(1, 2, dtype=torch.float32),
        "w3.qzeros": pack_gptq_int4_qzeros(torch.full((1, 2), 7)),
        "w3.g_idx": torch.tensor([0, 0], dtype=torch.int32),
    }
    initial_payload = encode_gptq_marlin_bundle_sections(
        bundle_id="bundle_w13",
        group_size=2,
        size_k=2,
        size_n=4,
        sections=base_sections,
        act_order=True,
    )
    layout_hash = decode_gptq_marlin_bundle(initial_payload).metadata.layout_hash
    store = GptqCacheStore.create(
        tmp_path,
        {
            "bundle_w13": GptqCacheRecord(
                "bundle_w13",
                "gptq_0000.bin",
                0,
                len(initial_payload),
                quant_layout_hash=layout_hash,
            ),
        },
    )
    store.flush_touched({"bundle_w13": initial_payload}, generation=0)
    requantizer = GptqMarlinBundleRequantizer(
        store=store,
        param_to_bundle={"param_w13": "bundle_w13"},
    )

    updates = requantizer.requantize_touched(
        {
            "param_w13": torch.tensor(
                [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0],
                dtype=torch.float32,
            ),
        },
        touched_param_ids=("param_w13",),
    )

    decoded = decode_gptq_marlin_bundle(updates["bundle_w13"])

    assert len(updates["bundle_w13"]) == len(initial_payload)
    assert decoded.metadata.layout_hash == layout_hash
    assert set(decoded.tensors) == set(base_sections)
    assert decoded.tensors["w1.qweight"].shape == (1, 2)
    assert decoded.tensors["w3.qweight"].shape == (1, 2)


def test_marlin_bundle_requantizer_rejects_master_shape_mismatch(tmp_path) -> None:
    initial_payload = encode_gptq_marlin_bundle_sections(
        bundle_id="bundle_a",
        group_size=2,
        size_k=4,
        size_n=2,
        sections={
            "qweight": pack_gptq_int4_qweight(torch.full((4, 2), 8)),
            "scales": torch.ones(2, 2, dtype=torch.float32),
        },
    )
    store = GptqCacheStore.create(
        tmp_path,
        {
            "bundle_a": GptqCacheRecord(
                "bundle_a",
                "gptq_0000.bin",
                0,
                len(initial_payload),
                quant_layout_hash=decode_gptq_marlin_bundle(
                    initial_payload
                ).metadata.layout_hash,
            ),
        },
    )
    store.flush_touched({"bundle_a": initial_payload}, generation=0)
    requantizer = GptqMarlinBundleRequantizer(
        store=store,
        param_to_bundle={"param_a": "bundle_a"},
    )

    with pytest.raises(ValueError, match="FP32 master size"):
        requantizer.requantize_touched(
            {"param_a": torch.ones(7)},
            touched_param_ids=("param_a",),
        )
