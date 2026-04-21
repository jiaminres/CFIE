from __future__ import annotations

import torch

from cfie.model_executor.layers.fused_moe.routed_experts_capturer import (
    get_routed_experts_attention_group_index,
    get_routed_experts_buffer_num_slots,
)
from cfie.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)


def test_routed_experts_buffer_uses_full_attention_slot_space() -> None:
    kv_cache_config = KVCacheConfig(
        num_blocks=524,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layers.0.attn"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=8,
                    head_size=128,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["layers.1.attn"],
                kv_cache_spec=SlidingWindowSpec(
                    block_size=8,
                    num_kv_heads=8,
                    head_size=128,
                    dtype=torch.float16,
                    sliding_window=4096,
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["layers.2.attn"],
                kv_cache_spec=SlidingWindowSpec(
                    block_size=8,
                    num_kv_heads=8,
                    head_size=128,
                    dtype=torch.float16,
                    sliding_window=4096,
                ),
            ),
            KVCacheGroupSpec(
                layer_names=["layers.3.attn"],
                kv_cache_spec=SlidingWindowSpec(
                    block_size=8,
                    num_kv_heads=8,
                    head_size=128,
                    dtype=torch.float16,
                    sliding_window=4096,
                ),
            ),
        ],
    )

    assert get_routed_experts_attention_group_index(kv_cache_config) == 0
    assert get_routed_experts_buffer_num_slots(kv_cache_config) == 8384
