"""Unit tests for GPTQ/Marlin cache bundle containers."""

from __future__ import annotations

import pytest
import torch

from cfie_training.training_base import (
    GptqCacheRecord,
    GptqCacheStore,
    GptqMarlinBundleCodec,
    GptqMarlinBundleSection,
    GptqMarlinBundleTensors,
    decode_gptq_marlin_bundle,
    gptq_marlin_bundle_layout_hash,
)


def _bundle_tensors() -> GptqMarlinBundleTensors:
    return GptqMarlinBundleTensors(
        qweight=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        scales=torch.tensor([[0.5, 0.25]], dtype=torch.float16),
        qzeros=torch.tensor([[8, 8]], dtype=torch.int32),
        g_idx=torch.tensor([0, 0], dtype=torch.int32),
        perm=torch.tensor([1, 0], dtype=torch.int32),
    )


def test_gptq_marlin_bundle_codec_round_trips_sections() -> None:
    codec = GptqMarlinBundleCodec(
        bundle_id="layers.0.experts.1.w13_weight",
        group_size=128,
        size_k=2,
        size_n=2,
        act_order=True,
    )

    payload = codec.encode(_bundle_tensors())
    decoded = codec.decode(payload)

    assert decoded.metadata.bundle_id == "layers.0.experts.1.w13_weight"
    assert decoded.metadata.has_zero_points
    assert decoded.metadata.has_g_idx
    assert decoded.metadata.has_perm
    assert decoded.metadata.act_order
    assert decoded.metadata.payload_num_bytes == len(payload)
    assert set(decoded.tensors) == {"g_idx", "perm", "qweight", "qzeros", "scales"}
    assert torch.equal(
        decoded.tensors["qweight"],
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
    )
    assert torch.equal(
        decoded.tensors["scales"],
        torch.tensor([[0.5, 0.25]], dtype=torch.float16),
    )


def test_gptq_marlin_bundle_layout_hash_tracks_flags() -> None:
    with_perm = gptq_marlin_bundle_layout_hash(
        group_size=128,
        size_k=2,
        size_n=2,
        act_order=True,
        has_zero_points=True,
        has_g_idx=True,
        has_perm=True,
    )
    without_perm = gptq_marlin_bundle_layout_hash(
        group_size=128,
        size_k=2,
        size_n=2,
        act_order=True,
        has_zero_points=True,
        has_g_idx=True,
        has_perm=False,
    )

    assert with_perm.startswith("sha256:")
    assert with_perm != without_perm


def test_gptq_marlin_bundle_payload_can_be_stored_in_gptq_cache(tmp_path) -> None:
    codec = GptqMarlinBundleCodec(
        bundle_id="bundle_a",
        group_size=128,
        size_k=2,
        size_n=2,
    )
    payload = codec.encode(_bundle_tensors())
    layout_hash = decode_gptq_marlin_bundle(payload).metadata.layout_hash
    store = GptqCacheStore.create(
        tmp_path,
        {
            "bundle_a": GptqCacheRecord(
                bundle_id="bundle_a",
                shard_name="gptq_0000.bin",
                offset_bytes=0,
                num_bytes=len(payload),
                quant_layout_hash=layout_hash,
            ),
        },
    )

    store.flush_touched({"bundle_a": payload}, generation=1)
    decoded = decode_gptq_marlin_bundle(store.read_bundle("bundle_a"))

    assert decoded.metadata.bundle_id == "bundle_a"
    assert decoded.metadata.layout_hash == layout_hash
    assert torch.equal(decoded.tensors["g_idx"], torch.tensor([0, 0], dtype=torch.int32))


def test_gptq_marlin_bundle_decode_rejects_invalid_payload() -> None:
    with pytest.raises(ValueError, match="magic"):
        decode_gptq_marlin_bundle(b"not-a-valid-bundle")


def test_gptq_marlin_bundle_decode_rejects_truncated_section() -> None:
    codec = GptqMarlinBundleCodec(
        bundle_id="bundle_a",
        group_size=128,
        size_k=2,
        size_n=2,
    )
    payload = codec.encode(_bundle_tensors())

    with pytest.raises(ValueError, match="truncated"):
        decode_gptq_marlin_bundle(payload[:-1])


def test_gptq_marlin_bundle_section_rejects_duplicate_names() -> None:
    section = GptqMarlinBundleSection(
        name="qweight",
        dtype="int32",
        shape=(1,),
        offset_bytes=0,
        num_bytes=4,
    )

    with pytest.raises(ValueError, match="unique"):
        from cfie_training.training_base import GptqMarlinBundleMetadata

        GptqMarlinBundleMetadata(
            bundle_id="bundle_a",
            group_size=128,
            size_k=1,
            size_n=1,
            sections=(section, section),
        )
