# Adapted from https://github.com/cfie-project/flash-attention/blob/main/flash_attn/layers/rotary.py
# Modified lines are marked with `# modified from original` comment
# Copyright (c) 2023, Tri Dao.

import math
from typing import Optional, Tuple, Union

import torch
from einops import rearrange, repeat
from ..ops.triton.rotary import apply_rotary   # modified from original


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]],
        dim=-1,
    )


class ApplyRotaryEmb(torch.autograd.Function):
    # 自定义 autograd.Function
    # 作用：把底层 apply_rotary 前向/反向封装成一个可参与自动求导的算子

    @staticmethod
    def forward(
        ctx,                                   # autograd 上下文对象，用于在 forward 中保存 backward 需要的信息
        x,                                     # 输入张量，形状常见：
                                               # - [batch_size, seqlen, nheads, headdim]
                                               # - 或 [total_seqlen, nheads, headdim]
        cos,                                   # cos 表，形状: [seqlen_rotary, rotary_dim / 2]
        sin,                                   # sin 表，形状: [seqlen_rotary, rotary_dim / 2]
        interleaved=False,                     # True: GPT-J 风格；False: NeoX 风格
        inplace=False,                         # 是否原地修改 x
        seqlen_offsets: Union[int, torch.Tensor] = 0,  # 位置偏移；可为 int 或 [batch_size]
        cu_seqlens: Optional[torch.Tensor] = None,     # 变长序列边界，形状: [batch + 1]
        max_seqlen: Optional[int] = None,              # 变长 batch 中最大序列长度
    ):
        # 调用底层 apply_rotary 完成真正的前向旋转
        out = apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,  # 位置偏移
            cu_seqlens=cu_seqlens,          # 变长序列边界
            max_seqlen=max_seqlen,          # 最大序列长度
            interleaved=interleaved,        # 旋转排布风格
            inplace=inplace,                # 是否原地
        )

        # backward 里需要再次调用 apply_rotary，所以要把必要张量存进 ctx
        if isinstance(seqlen_offsets, int):
            # save_for_backward 只能保存 Tensor，不能保存 Python int
            # 所以 int 型 seqlen_offsets 单独挂在 ctx 上
            ctx.save_for_backward(cos, sin, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            # 如果 seqlen_offsets 本身是 Tensor，就一起存进 saved_tensors
            # 常见形状: [batch_size]
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None

        # 保存若干非 Tensor 配置，供 backward 使用
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen

        # 若不是原地模式，返回 out
        # 若是原地模式，底层已经把结果写回 x，所以直接返回 x
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        # do = d(output)，即来自上游的梯度
        # 形状与 forward 输出相同：
        # - [batch_size, seqlen, nheads, headdim]
        # - 或 [total_seqlen, nheads, headdim]

        seqlen_offsets = ctx.seqlen_offsets

        # 根据 forward 里保存方式的不同，恢复保存的张量
        if seqlen_offsets is None:
            # 说明 seqlen_offsets 在 forward 中是 Tensor，
            # 所以它保存在 saved_tensors 中
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            # 说明 seqlen_offsets 在 forward 中是 Python int，
            # saved_tensors 里只有 cos / sin / cu_seqlens
            cos, sin, cu_seqlens = ctx.saved_tensors

        # 历史兼容处理：
        # 某些 Triton 版本在这个分支下会报 invalid device context，
        # clone 一份梯度后可规避
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()

        # 反向本质上还是再次调用 apply_rotary，
        # 但这次传 conjugate=True，表示应用“共轭旋转/逆旋转”
        # 从而把输出梯度映射回输入梯度 dx
        dx = apply_rotary(
            do,                              # 上游梯度 d(output)
            cos,                             # [seqlen_rotary, rotary_dim / 2]
            sin,                             # [seqlen_rotary, rotary_dim / 2]
            seqlen_offsets=seqlen_offsets,   # int 或 [batch_size]
            cu_seqlens=cu_seqlens,           # [batch + 1] 或 None
            max_seqlen=ctx.max_seqlen,       # int 或 None
            interleaved=ctx.interleaved,     # 保持与 forward 一致
            inplace=ctx.inplace,             # 保持与 forward 一致
            conjugate=True,                  # 反向时做逆旋转
        )

        # backward 需要返回与 forward 输入一一对应的梯度
        # forward 输入共有 8 个：
        #   x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
        #
        # 只有 x 需要梯度，其他参数都不需要梯度，所以返回 None
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x,  # 输入张量
    cos,  # cos 位置编码表
    sin,  # sin 位置编码表
    interleaved=False,  # 是否使用交错维旋转（GPT-J 风格）
    inplace=False,  # 是否原地修改 x
    seqlen_offsets: Union[int, torch.Tensor] = 0,  # 每条序列的位置偏移
    cu_seqlens: Optional[torch.Tensor] = None,  # 变长序列的前缀和边界
    max_seqlen: Optional[int] = None,  # 变长 batch 中的最大序列长度
):
    """
    参数说明：

    x:
        如果 cu_seqlens is None，
        则 x 形状为：
            (batch_size, seqlen, nheads, headdim)

        否则 x 形状为：
            (total_seqlen, nheads, headdim)

        也就是说：
        - 定长 batch 时，x 按 [批大小, 序列长度, 头数, 每头维度] 组织
        - 变长 batch 时，所有序列会沿 token 维拼接成 total_seqlen

    cos, sin:
        形状为：
            (seqlen_rotary, rotary_dim / 2)

        表示预先计算好的 cos / sin 表。
        注意：
        - seqlen_rotary 是位置表长度
        - 不一定等于当前输入 x 的 seqlen
        - 只要能覆盖当前实际使用到的位置即可

    interleaved:
        如果为 True，
        则按偶数维 / 奇数维成对旋转（GPT-J 风格）

        如果为 False，
        则按前半维 / 后半维成对旋转（GPT-NeoX 风格）

    inplace:
        如果为 True，则直接在输入 x 上原地应用 rotary embedding，
        不额外创建输出张量。

    seqlen_offsets:
        可以是：
        - 一个 int
        - 或一个形状为 (batch_size,) 的张量

        含义：
        每条序列的位置都会整体平移这个偏移量。

        最常见用途：
        推理时结合 KV cache 使用。
        例如前面已经缓存了很多 token，
        那当前新 token 的位置不能从 0 开始，而要从缓存长度开始。

    cu_seqlens:
        形状：
            (batch + 1,)
        或者为 None

        当不为 None 时，表示输入 x 采用“变长拼接”格式。
        它保存每条序列在拼接大张量中的起止边界。

    max_seqlen:
        一个整数。
        当 cu_seqlens 不为 None 时，表示当前 batch 中最长序列的长度。

    返回值：

    out:
        如果 cu_seqlens is None，
        输出形状为：
            (batch_size, seqlen, nheads, headdim)

        否则输出形状为：
            (total_seqlen, nheads, headdim)

    额外说明：

    rotary_dim 必须满足：
        rotary_dim <= headdim

    实际作用：
        只对 x 的前 rotary_dim 个维度施加旋转位置编码，
        剩余维度保持不变。
    """

    # 调用底层 autograd.Function 版本的 ApplyRotaryEmb
    # 这里会根据：
    # - 是否变长输入
    # - 是否有位置偏移
    # - 是否交错旋转
    # - 是否原地计算
    # 来选择相应实现
    return ApplyRotaryEmb.apply(
        x,              # 输入张量
        cos,            # cos 表
        sin,            # sin 表
        interleaved,    # 旋转风格
        inplace,        # 是否原地
        seqlen_offsets, # 位置偏移
        cu_seqlens,     # 变长序列边界
        max_seqlen,     # 最大序列长度
    )


# 为了兼容旧代码，保留一个旧名字别名
apply_rotary_emb_func = apply_rotary_emb


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cos_k=None,
        sin_k=None,
        interleaved=False,
        seqlen_offsets: Union[int, torch.Tensor] = 0,
        num_heads_q: Union[int] = None,
    ):
        if cos_k is None and sin_k is None and qkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need qkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            if qkv.dim() == 5:
                batch, seqlen, three, nheads, headdim = qkv.shape
                assert three == 3
                # qk = rearrange(qkv[:, :, :2], "b s t h d -> b s (t h) d")
                qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
            else:
                assert qkv.dim() == 4
                assert num_heads_q is not None
                num_heads_k = (qkv.shape[2] - num_heads_q) // 2
                assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
                qk = qkv[:, :, :num_heads_q + num_heads_k]
            apply_rotary(
                qk, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved, inplace=True
            )
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            if qkv.dim() == 5:
                q, k = qkv[:, :, 0], qkv[:, :, 1]
            else:
                assert qkv.dim() == 4
                assert num_heads_q is not None
                num_heads_k = (qkv.shape[2] - num_heads_q) // 2
                assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
                q, k = qkv[:, :, :num_heads_q], qkv[:, :, num_heads_q : num_heads_q + num_heads_k]
            apply_rotary(q, cos, sin, seqlen_offsets, interleaved=interleaved, inplace=True)
            apply_rotary(k, cos_k, sin_k, seqlen_offsets, interleaved=interleaved, inplace=True)
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cos_k, sin_k, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.num_heads_q = num_heads_q
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cos_k, sin_k, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cos_k, sin_k = ctx.saved_tensors
        if cos_k is None and sin_k is None and dqkv.is_contiguous():
            # Call 1 kernel instead of 2 kernels
            # We need dqkv to be contiguous so that when we reshape to combine (3, nheads)
            # dimensions, we get the same tensor
            if dqkv.dim() == 5:
                dqk = rearrange(dqkv[:, :, :2], "b s t h d -> b s (t h) d")
            else:
                assert dqkv.dim() == 4
                assert ctx.num_heads_q is not None
                num_heads_k = (dqkv.shape[2] - ctx.num_heads_q) // 2
                assert dqkv.shape[2] == ctx.num_heads_q + 2 * num_heads_k
                dqk = dqkv[:, :, : ctx.num_heads_q + num_heads_k]
            apply_rotary(
                dqk,
                cos,
                sin,
                seqlen_offsets=seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
            )
        else:
            cos_k = cos if cos_k is None else cos_k
            sin_k = sin if sin_k is None else sin_k
            if dqkv.dim() == 5:
                dq, dk = dqkv[:, :, 0], dqkv[:, :, 1]
            else:
                assert dqkv.dim() == 4
                assert ctx.num_heads_q is not None
                num_heads_k = (dqkv.shape[2] - ctx.num_heads_q) // 2
                assert dqkv.shape[2] == ctx.num_heads_q + 2 * num_heads_k
                dq = dqkv[:, :, : ctx.num_heads_q]
                dk = dqkv[:, :, ctx.num_heads_q : ctx.num_heads_q + num_heads_k]
            apply_rotary(
                dq,
                cos,
                sin,
                seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
            )
            apply_rotary(
                dk,
                cos_k,
                sin_k,
                seqlen_offsets,
                interleaved=ctx.interleaved,
                inplace=True,
                conjugate=True,
            )
        return dqkv, None, None, None, None, None, None, None


def apply_rotary_emb_qkv_(
    qkv,
    cos,
    sin,
    cos_k=None,
    sin_k=None,
    interleaved=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    num_heads_q: Optional[int] = None,
):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim).
            If qkv has shape (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim) (e.g. MQA / GQA),
            then num_heads_q must be provided.
        cos, sin: (seqlen, rotary_dim / 2)
        cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of Q and K.
    """
    return ApplyRotaryEmbQKV_.apply(
        qkv, cos, sin, cos_k, sin_k, interleaved, seqlen_offsets, num_heads_q
    )


class ApplyRotaryEmbKV_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, cos, sin, interleaved=False, seqlen_offsets: Union[int, torch.Tensor] = 0):
        batch, seqlen, two, nheads, headdim = kv.shape
        assert two == 2
        k = kv[:, :, 0]
        apply_rotary(
            k, cos, sin, seqlen_offsets=seqlen_offsets, interleaved=interleaved, inplace=True
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin)  # Can't save int with save_for_backward
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        return kv

    @staticmethod
    def backward(ctx, dkv):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin = ctx.saved_tensors
        apply_rotary(
            dkv[:, :, 0],
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=ctx.interleaved,
            inplace=True,
            conjugate=True,
        )
        return dkv, None, None, None, None


apply_rotary_emb_kv_ = ApplyRotaryEmbKV_.apply


def apply_rotary_emb_kv_(
    kv,
    cos,
    sin,
    interleaved=False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
):
    """
    Arguments:
        kv: (batch_size, seqlen, 2, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead of
            1st half and 2nd half (GPT-NeoX style).
        seqlen_offsets: (batch_size,) or int. Each sequence in Q and K is shifted by this amount.
            Most commonly used in inference when we have KV cache.
    Return:
        kv: (batch_size, seqlen, 2, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding *inplace* to the first rotary_dim of K.
    """
    return ApplyRotaryEmbKV_.apply(kv, cos, sin, interleaved, seqlen_offsets)


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        pos_idx_in_fp32=True,
        device=None,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self._compute_inv_freq(device=device)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        max_seqlen: Optional[int] = None,
        num_heads_q: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim) or (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim)
            if kv is none, else it's just q of shape (batch, seqlen, nheads, headdim).
            If qkv has shape (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim) (e.g. MQA / GQA),
            then num_heads_q must be provided.
        kv: (batch, seqlen, 2, nheads, headdim)
        seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
            If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
            should pass in max_seqlen, which will update the cos / sin cache up to that length.
        Apply rotary embedding *inplace* to qkv and / or kv.
        """
        seqlen = qkv.shape[1]
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype)
        if kv is None:
            if self.scale is None:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    num_heads_q=num_heads_q,
                )
            else:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    num_heads_q=num_heads_q,
                )
        else:
            q = qkv
            q = apply_rotary_emb_func(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
            )
            if self.scale is None:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            else:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            return q, kv
