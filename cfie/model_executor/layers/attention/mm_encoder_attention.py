# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import numpy as np
import torch

from cfie.logger import init_logger
from cfie.model_executor.custom_op import CustomOp, get_oot_class_by_name
from cfie.model_executor.models.vision import get_vit_attn_backend
from cfie.utils.math_utils import round_up
from cfie.v1.attention.backends.fa_utils import get_flash_attn_version
from cfie.v1.attention.backends.registry import AttentionBackendEnum
from cfie.v1.attention.ops.vit_attn_wrappers import (
    vit_flash_attn_wrapper,
    vit_flashinfer_wrapper,
    vit_torch_sdpa_wrapper,
    vit_triton_attn_wrapper,
)

logger = init_logger(__name__)

# Batch buckets for cuDNN graph caching.
# Graphs use batch size and max sequence length as cache key.
# This avoids creating a new graph for each unique set of
# batch size and max sequence length at runtime.
# From the cuDNN team's performance measurements, there
# is no significant kernel performance difference between padding
# to a smaller batch size/seq length and padding to larger
# ones. The bucketing here is solely used to avoid memory
# operation overhead, which won't be needed if we have CUDA
# graph support in the future.
# TODO: Remove buckets after issue #34763
# (cuda graph support) is addressed.
FLASHINFER_BATCH_BUCKETS = [8, 16, 32, 64]
FLASHINFER_MAX_SEQLEN_BUCKETS = [
    1 * 1024,
    2 * 1024,
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
]

# Workspace buffer for FlashInfer CuDNN backend
FLASHINFER_CUDNN_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024
_flashinfer_workspace_buffer: torch.Tensor | None = None


def _get_flashinfer_workspace_buffer() -> torch.Tensor:
    global _flashinfer_workspace_buffer
    if _flashinfer_workspace_buffer is None:
        _flashinfer_workspace_buffer = torch.zeros(
            FLASHINFER_CUDNN_WORKSPACE_SIZE_BYTES,
            dtype=torch.uint8,
            device="cuda",
        )
    return _flashinfer_workspace_buffer


def add_padding_to_seqlens(
        seq: np.ndarray,
        batch_size: int,
        padding_value: int,
) -> np.ndarray:
    batch_size_padded = next(
        (b for b in FLASHINFER_BATCH_BUCKETS if b >= batch_size),
        round_up(batch_size, FLASHINFER_BATCH_BUCKETS[0]),
    )
    if batch_size_padded == batch_size:
        return seq
    return np.concatenate(
        [
            seq,
            np.full((batch_size_padded - batch_size,), padding_value, dtype=seq.dtype),
        ]
    )


def bucket_flashinfer_max_seqlen(
        real_max_seqlen: int,
) -> int:
    if real_max_seqlen <= 0:
        return FLASHINFER_MAX_SEQLEN_BUCKETS[0]
    return next(
        (s for s in FLASHINFER_MAX_SEQLEN_BUCKETS if s >= real_max_seqlen),
        round_up(real_max_seqlen, FLASHINFER_MAX_SEQLEN_BUCKETS[-1]),
    )


# --8<-- [start:mm_encoder_attn]
@CustomOp.register("mm_encoder_attn")
class MMEncoderAttention(CustomOp):
    """不带任何 KV cache 的多头注意力，用于多模态编码器。"""

    # --8<-- [end:mm_encoder_attn]

    @classmethod
    def compute_max_seqlen(
            cls,
            attn_backend: AttentionBackendEnum,  # 当前使用的 attention 后端
            cu_seqlens: np.ndarray,  # 前缀和形式的序列边界，形状通常为 [batch + 1]
    ) -> int:
        # 默认最大序列长度为 0
        max_seqlen = 0

        # 只有以下这些后端需要显式计算 max_seqlen：
        # - FLASH_ATTN
        # - ROCM_AITER_FA
        # - TRITON_ATTN
        # - FLASHINFER
        #
        # 并且 cu_seqlens 至少要有两个元素，
        # 因为长度是由相邻边界做差得到的
        if (
                attn_backend
                in (
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.ROCM_AITER_FA,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLASHINFER,
        )
                and len(cu_seqlens) >= 2
        ):
            # 由前缀和边界还原每条序列长度：
            # 例如 cu_seqlens = [0, 3, 8, 10]
            # 则各序列长度 = [3, 5, 2]
            #
            # 再从中取最大值，得到当前 batch 的最大序列长度
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())

        # 如果后端是 FLASHINFER，
        # 则进一步把 max_seqlen 映射到 FLASHINFER 期望的 bucket 大小
        # 这样做通常是为了适配其 kernel / workspace / 调度要求
        if attn_backend == AttentionBackendEnum.FLASHINFER:
            max_seqlen = bucket_flashinfer_max_seqlen(max_seqlen)

        # 返回最终的最大序列长度
        return max_seqlen

    @classmethod
    def maybe_compute_seq_lens(
            cls,
            attn_backend: AttentionBackendEnum,  # 当前使用的 attention 后端
            cu_seqlens: np.ndarray,  # 前缀和形式的序列边界，形状通常为 [batch + 1]
            device: torch.device,  # 输出张量要放置的设备
    ) -> torch.Tensor | None:
        # 如果当前类存在 out-of-tree(OOT) 的外部替代实现，
        # 则优先调用外部实现的 maybe_compute_seq_lens
        if (oot_class := get_oot_class_by_name(cls.__name__)) is not None:
            return oot_class.maybe_compute_seq_lens(attn_backend, cu_seqlens, device)  # type: ignore[attr-defined]

        # 只有 FLASHINFER 后端需要显式的 sequence_lengths
        # 其他后端不需要时，直接返回 None
        if attn_backend != AttentionBackendEnum.FLASHINFER:
            return None

        # 由前缀和边界 cu_seqlens 还原出每条序列的实际长度
        # 例如：
        # cu_seqlens = [0, 3, 8, 10]
        # 则 sequence_lengths = [3, 5, 2]
        sequence_lengths = cu_seqlens[1:] - cu_seqlens[:-1]

        # 对 sequence_lengths 做 padding 处理
        # 这里填充值是 0，长度目标是当前序列条数 len(sequence_lengths)
        # 具体作用取决于 add_padding_to_seqlens 的实现，一般是为了适配后端输入格式
        sequence_lengths = add_padding_to_seqlens(
            sequence_lengths, len(sequence_lengths), 0
        )

        # 把 numpy 数组转成 torch.Tensor，并搬到目标设备
        # non_blocking=True 表示若条件允许，使用异步拷贝
        sequence_lengths = torch.from_numpy(sequence_lengths).to(
            device, non_blocking=True
        )

        # 返回每条序列长度张量
        return sequence_lengths

    @classmethod
    def maybe_recompute_cu_seqlens(
            cls,
            attn_backend: AttentionBackendEnum,  # 当前使用的 attention 后端
            cu_seqlens: np.ndarray,  # 原始前缀和序列边界，形状通常为 [batch + 1]
            hidden_size: int,  # 模型隐藏维度
            tp_size: int,  # tensor parallel 大小
            device: torch.device,  # 最终张量放置的设备
    ) -> torch.Tensor:
        # 如果当前类存在 out-of-tree(OOT) 外部替代实现，
        # 则优先调用外部实现
        if (oot_class := get_oot_class_by_name(cls.__name__)) is not None:
            return oot_class.maybe_recompute_cu_seqlens(  # type: ignore[attr-defined]
                attn_backend, cu_seqlens, hidden_size, tp_size, device
            )

        # 只有 FLASHINFER 后端需要对 cu_seqlens 做特殊重算
        if attn_backend == AttentionBackendEnum.FLASHINFER:
            # batch 大小 = 序列条数 = 边界数 - 1
            batch_size = len(cu_seqlens) - 1

            # 每个 token 在当前 TP rank 上展开后的特征长度
            # 等于 hidden_size 按 tp_size 切分后的本地维度
            scale = hidden_size // tp_size

            # 把“按 token 计数的前缀和”转换成“按元素数/展平长度计数的前缀和”
            # 例如如果每个 token 展平后占 scale 个元素，
            # 那么序列边界也要整体乘以 scale
            cu_seqlens = cu_seqlens * scale

            # q/k/o 共用一套边界
            cu_seqlens_qko = cu_seqlens

            # v 部分的展开长度是 q/k/o 的 3 倍，所以边界整体再乘 3
            cu_seqlens_v = cu_seqlens * 3

            # 对 q/k/o 的边界做 padding，补到 FLASHINFER 期望的格式
            # 填充值使用最后一个边界值，即总长度
            cu_seqlens_qko = add_padding_to_seqlens(
                cu_seqlens_qko, batch_size, cu_seqlens_qko[-1]
            )

            # 对 v 的边界也做同样 padding
            cu_seqlens_v = add_padding_to_seqlens(
                cu_seqlens_v, batch_size, cu_seqlens_v[-1]
            )

            # 把 q/k/o 边界和 v 边界拼起来，
            # 形成 FLASHINFER 需要的最终 cu_seqlens 格式
            cu_seqlens = np.concatenate([cu_seqlens_qko, cu_seqlens_v])

        # 把 numpy 数组转成 torch.Tensor，并搬到目标设备
        # non_blocking=True 表示如果条件允许则异步拷贝
        cu_seqlens = torch.from_numpy(cu_seqlens).to(device, non_blocking=True)

        return cu_seqlens

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float | None = None,
            num_kv_heads: int | None = None,
            prefix: str = "",
    ) -> None:
        """
        Args:
            num_heads: number of attention heads per partition.
            head_size: hidden_size per attention head.
            scale: scale factor.
            num_kv_heads: number of kv heads.
            prefix: This has no effect, it is only here to make it easier to
                    swap between Attention and MultiHeadAttention
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = 1.0 / (head_size ** 0.5) if scale is None else scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.layer_name = prefix
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) is not "
            f"divisible by num_kv_heads ({self.num_kv_heads})"
        )
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()

        # Get device-specific vision attention backend.
        self.attn_backend = get_vit_attn_backend(
            head_size=head_size,
            dtype=dtype,
        )

        self.is_flash_attn_backend = self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        }

        self._fa_version = (
            get_flash_attn_version(head_size=head_size)
            if self.is_flash_attn_backend
            else None
        )

        if self.attn_backend == AttentionBackendEnum.FLASHINFER:
            _get_flashinfer_workspace_buffer()

        logger.info_once(f"Using {self.attn_backend} for MMEncoderAttention.")

    @classmethod
    def enabled(cls) -> bool:
        return True

    def view_qkv_to_4d(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            bsz: int,
            q_len: int,
            kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reshape query, key, value to 4D tensors:
        (batch_size, seq_len, num_heads, head_size)
        """
        query = query.view(bsz, q_len, self.num_heads, self.head_size)
        key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)

        return query, key, value

    def _forward_sdpa(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Input shape:
        (batch_size x seq_len x hidden_size) or
        (batch_size x seq_len x num_heads x head_size)
        """
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4

        query, key, value = self.view_qkv_to_4d(query, key, value, bsz, q_len, kv_len)

        output = vit_torch_sdpa_wrapper(
            q=query,
            k=key,
            v=value,
            scale=self.scale,
            cu_seqlens=cu_seqlens,
            enable_gqa=self.num_heads > self.num_kv_heads,
        )
        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        return output

    def _forward_fa(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        """Input shape:
        (batch_size x seq_len x hidden_size) or
        (batch_size x seq_len x num_heads x head_size)
        """
        assert (cu_seqlens is not None and max_seqlen is not None) or (
                cu_seqlens is None and max_seqlen is None
        ), "cu_seqlens and max_seqlen should be both set or both None."

        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4

        query, key, value = self.view_qkv_to_4d(query, key, value, bsz, q_len, kv_len)

        output = vit_flash_attn_wrapper(
            q=query,
            k=key,
            v=value,
            batch_size=bsz,
            is_rocm_aiter=(self.attn_backend == AttentionBackendEnum.ROCM_AITER_FA),
            fa_version=self._fa_version,
            scale=self.scale,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        return output

    def _forward_triton(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        """Input shape:
        (batch_size x seq_len x hidden_size) or
        (batch_size x seq_len x num_heads x head_size)
        """
        assert (cu_seqlens is not None and max_seqlen is not None) or (
                cu_seqlens is None and max_seqlen is None
        ), "cu_seqlens and max_seqlen should be both set or both None."

        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4

        query, key, value = self.view_qkv_to_4d(query, key, value, bsz, q_len, kv_len)

        output = vit_triton_attn_wrapper(
            q=query,
            k=key,
            v=value,
            batch_size=bsz,
            scale=self.scale,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        return output

    def _forward_flashinfer(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,
            sequence_lengths: torch.Tensor
                              | None = None,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        return vit_flashinfer_wrapper(
            q=query,
            k=key,
            v=value,
            scale=self.scale,
            workspace_buffer=_get_flashinfer_workspace_buffer(),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )

    def forward_native(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
            sequence_lengths: torch.Tensor
                              | None = None,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        return self._forward_sdpa(query, key, value, cu_seqlens)

    def forward_cuda(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
            sequence_lengths: torch.Tensor
                              | None = None,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        if self.is_flash_attn_backend:
            return self._forward_fa(query, key, value, cu_seqlens, max_seqlen)
        elif self.attn_backend == AttentionBackendEnum.TRITON_ATTN:
            return self._forward_triton(query, key, value, cu_seqlens, max_seqlen)
        elif self.attn_backend == AttentionBackendEnum.FLASHINFER:
            return self._forward_flashinfer(
                query, key, value, cu_seqlens, max_seqlen, sequence_lengths
            )
        elif self.attn_backend == AttentionBackendEnum.TORCH_SDPA:
            return self._forward_sdpa(query, key, value, cu_seqlens)
        else:
            raise ValueError(
                f"Unsupported multi-modal encoder attention backend for CUDA: "
                f"{self.attn_backend}."
            )

    def forward_cpu(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
            sequence_lengths: torch.Tensor
                              | None = None,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        return self._forward_sdpa(query, key, value, cu_seqlens)

    def forward_xpu(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            cu_seqlens: torch.Tensor | None = None,
            max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
            sequence_lengths: torch.Tensor
                              | None = None,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        if self.attn_backend == AttentionBackendEnum.FLASH_ATTN:
            return self._forward_fa(query, key, value, cu_seqlens, max_seqlen)
        elif self.attn_backend == AttentionBackendEnum.TRITON_ATTN:
            return self._forward_triton(query, key, value, cu_seqlens, max_seqlen)
        elif self.attn_backend == AttentionBackendEnum.TORCH_SDPA:
            return self._forward_sdpa(query, key, value, cu_seqlens)
        else:
            raise ValueError(
                f"Unsupported multi-modal encoder attention backend for XPU: "
                f"{self.attn_backend}."
            )
