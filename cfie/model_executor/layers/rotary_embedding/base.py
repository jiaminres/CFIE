# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rotary Positional Embeddings Base Class."""

import torch

from cfie._aiter_ops import rocm_aiter_ops
from cfie.model_executor.custom_op import CustomOp

from .common import ApplyRotaryEmb


# --8<-- [start:rotary_embedding]
# --8<-- [start:rotary_embedding]
@CustomOp.register("rotary_embedding")
class RotaryEmbeddingBase(CustomOp):
    """原始的旋转位置编码（Rotary Positional Embedding）基类。"""

    # --8<-- [end:rotary_embedding]

    def __init__(
            self,
            head_size: int,  # 每个 attention head 的总维度
            rotary_dim: int,  # 实际参与旋转的位置维度
            max_position_embeddings: int,  # 预先缓存的最大位置长度
            base: float,  # RoPE 的 base，通常是 10000
            is_neox_style: bool,  # 是否采用 GPT-NeoX 风格的旋转排布
            dtype: torch.dtype,  # cos/sin cache 希望使用的数据类型
            init_cache: bool = True,  # 初始化时是否立即构建 cos/sin cache
    ) -> None:
        # 初始化 CustomOp 基类
        super().__init__()

        # 保存每个 head 的总维度
        self.head_size = head_size

        # 保存实际参与旋转的维度数
        self.rotary_dim = rotary_dim

        # 保存最大位置长度
        self.max_position_embeddings = max_position_embeddings

        # 保存 RoPE base
        self.base = base

        # 保存是否使用 NeoX 风格旋转
        self.is_neox_style = is_neox_style

        # 保存目标 dtype
        self.dtype = dtype

        # 如果子类或外部没有提前设置 use_flashinfer，则默认关闭
        if not hasattr(self, "use_flashinfer"):
            self.use_flashinfer = False

        # 是否启用 ROCm + aiter 的 Triton rotary embedding 实现
        # 条件：
        # 1. 当前 CustomOp 处于 enabled 状态
        # 2. rocm_aiter_ops 表示对应 Triton rotary op 可用
        self.use_aiter = (
                self.enabled() and rocm_aiter_ops.is_triton_rotary_embed_enabled()
        )

        # 若启用了 aiter，则取出对应的 rotary embedding 算子
        if self.use_aiter:
            self.rocm_aiter_triton_rotary_embedding = (
                rocm_aiter_ops.get_triton_rotary_embedding_op()
            )

        # 若要求初始化时就构建 cache
        if init_cache:
            # 先计算 cos/sin cache
            cache = self._compute_cos_sin_cache()
            # cache 初始形状:
            # [max_position_embeddings, rotary_dim]
            # 因为内部其实是把 cos 与 sin 在最后一维拼起来：
            # [max_position_embeddings, rotary_dim/2] + [max_position_embeddings, rotary_dim/2]
            # 最终总维度仍是 rotary_dim

            # 若不用 flashinfer，则把 cache 转到指定 dtype
            # flashinfer 可能要求保留特定 dtype / layout
            if not self.use_flashinfer:
                cache = cache.to(dtype)

            # 显式声明 cos_sin_cache 是一个 torch.Tensor buffer
            self.cos_sin_cache: torch.Tensor

            # 注册成 buffer，而不是 Parameter
            # persistent=False 表示不把它写入 state_dict
            self.register_buffer("cos_sin_cache", cache, persistent=False)

        # 构造 ApplyRotaryEmb 辅助模块
        # 它负责真正把 cos/sin 施加到 query/key 的前 rotary_dim 维上
        self.apply_rotary_emb = ApplyRotaryEmb(
            is_neox_style=self.is_neox_style,
        )

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """计算逆频率 inv_freq。"""

        # NOTE:
        # 为了和 HF 实现完全对齐，严格来说应该先在 CPU 上算 cache，
        # 再搬到 GPU；但这里为了初始化更快，直接在目标设备上计算。
        # 这样可能会与 HF 版本有极小数值差异。

        inv_freq = 1.0 / (
                base
                ** (
                        torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
                )
        )
        # 形状: [rotary_dim / 2]
        #
        # 含义：
        # 对 rotary_dim 中每两个维度对应一个频率，
        # 这里生成这些频率的倒数

        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """计算 cos / sin 缓存表。"""

        # 先算逆频率
        inv_freq = self._compute_inv_freq(self.base)
        # 形状: [rotary_dim / 2]

        # 所有位置下标 [0, 1, 2, ..., max_position_embeddings - 1]
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        # 形状: [max_position_embeddings]

        # 外积：
        # 每个位置 × 每个逆频率
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # 形状: [max_position_embeddings, rotary_dim / 2]

        # 对这些相位取 cos
        cos = freqs.cos()
        # 形状: [max_position_embeddings, rotary_dim / 2]

        # 对这些相位取 sin
        sin = freqs.sin()
        # 形状: [max_position_embeddings, rotary_dim / 2]

        # 在最后一维拼接 cos 和 sin
        cache = torch.cat((cos, sin), dim=-1)
        # 形状: [max_position_embeddings, rotary_dim]

        return cache

    def _match_cos_sin_cache_dtype(self, query: torch.Tensor) -> torch.Tensor:
        # 先取当前缓存
        cos_sin_cache = self.cos_sin_cache

        # 如果 cache 的 device 和 dtype 已经和 query 完全一致，
        # 直接返回，避免不必要转换
        if (
                cos_sin_cache.device == query.device
                and self.cos_sin_cache.dtype == query.dtype
        ):
            return cos_sin_cache

        # 否则把 cache 转到 query 所在 device 和 dtype
        cos_sin_cache = cos_sin_cache.to(query.device, dtype=query.dtype)

        # 如果当前处于 torch.compile / cudagraph tracing 期间，
        # 为了避免在 tracing 时修改 module buffer，
        # 这里只返回转换后的临时张量，不回写 self.cos_sin_cache
        if torch.compiler.is_compiling():
            return cos_sin_cache

        # 非 tracing 场景下，直接更新模块里的 cache，
        # 这样下次若 query 还在同一 device/dtype，就可以直接复用
        self.cos_sin_cache = cos_sin_cache

        return cos_sin_cache

    def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 取前 seqlen 个位置对应的 cache
        cos_sin = self.cos_sin_cache[:seqlen]
        # 形状: [seqlen, rotary_dim]

        # 沿最后一维一分为二：
        # 前半部分是 cos，后半部分是 sin
        cos, sin = cos_sin.chunk(2, dim=-1)
        # cos 形状: [seqlen, rotary_dim / 2]
        # sin 形状: [seqlen, rotary_dim / 2]

        return cos, sin


class RotaryEmbedding(RotaryEmbeddingBase):
    # 最基础的 RoPE 实现类
    # 负责把 rotary position embedding 施加到 query / key 上

    def __init__(
            self,
            head_size: int,  # 每个 attention head 的总维度
            rotary_dim: int,  # 实际参与旋转的位置维度
            max_position_embeddings: int,  # 预先缓存的最大位置长度
            base: float,  # RoPE 的 base，一般是 10000
            is_neox_style: bool,  # 是否采用 GPT-NeoX 风格的旋转排布
            dtype: torch.dtype,  # cos/sin cache 的 dtype
            init_cache: bool = True,  # 初始化时是否立即构建 cos/sin cache
    ) -> None:
        # 直接调用父类初始化
        # 父类里通常会保存这些配置，并准备 cos/sin cache
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
            init_cache=init_cache,
        )

    @staticmethod
    def forward_static(
            positions: torch.Tensor,  # 位置下标，形状常见为 [num_tokens] 或可展平成 [num_tokens]
            query: torch.Tensor,  # query，形状常见为 [num_tokens, num_heads * head_size] 或类似展平形式
            key: torch.Tensor | None,  # key，形状与 query 类似；某些场景可为 None
            head_size: int,  # 每个 head 的总维度
            rotary_dim: int,  # 前 rotary_dim 维参与旋转，后面的维度直通
            cos_sin_cache: torch.Tensor,  # 预先缓存好的 cos/sin 表，形状常见为 [max_position, 2 * rotary_dim]
            is_neox_style: bool,  # 是否用 NeoX 风格旋转
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """forward() 的纯 PyTorch 实现。"""

        # 把 positions 压平成一维
        positions = positions.flatten()
        # positions 形状: [num_tokens]

        # token 数
        num_tokens = positions.shape[0]

        # 从 cos/sin cache 中取出这些位置对应的行
        cos_sin = cos_sin_cache.index_select(0, positions)
        # cos_sin 形状: [num_tokens, 2 * rotary_dim]

        # 一分为二：
        # 前半是 cos，后半是 sin
        cos, sin = cos_sin.chunk(2, dim=-1)
        # cos 形状: [num_tokens, rotary_dim]
        # sin 形状: [num_tokens, rotary_dim]

        # 先保存 query 原始形状，后面还原
        query_shape = query.shape

        # 把 query 重新看成 [num_tokens, num_heads, head_size]
        query = query.view(num_tokens, -1, head_size)
        # query 形状: [num_tokens, num_heads, head_size]

        # 取前 rotary_dim 维，这部分要做旋转
        query_rot = query[..., :rotary_dim]
        # 形状: [num_tokens, num_heads, rotary_dim]

        # 剩余维度不参与旋转，直接透传
        query_pass = query[..., rotary_dim:]
        # 形状: [num_tokens, num_heads, head_size - rotary_dim]

        # 对 query_rot 真正应用旋转位置编码
        query_rot = ApplyRotaryEmb.forward_static(
            query_rot,
            cos,
            sin,
            is_neox_style,
        )
        # 输出形状仍为: [num_tokens, num_heads, rotary_dim]

        # 把旋转后的部分和未旋转部分拼回去
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
        # 还原到 query 原始形状

        # 某些场景下 key 可能为 None，例如跨层 KV 共享
        if key is not None:
            # 保存 key 原始形状
            key_shape = key.shape

            # 重新看成 [num_tokens, num_heads, head_size]
            key = key.view(num_tokens, -1, head_size)
            # key 形状: [num_tokens, num_heads, head_size]

            # 前 rotary_dim 维参与旋转
            key_rot = key[..., :rotary_dim]
            # 形状: [num_tokens, num_heads, rotary_dim]

            # 其余维度直通
            key_pass = key[..., rotary_dim:]
            # 形状: [num_tokens, num_heads, head_size - rotary_dim]

            # 对 key_rot 应用旋转位置编码
            key_rot = ApplyRotaryEmb.forward_static(
                key_rot,
                cos,
                sin,
                is_neox_style,
            )
            # 形状仍为: [num_tokens, num_heads, rotary_dim]

            # 拼回去并恢复原始形状
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

        # 返回施加 RoPE 后的 query / key
        return query, key

    def forward_native(
            self,
            positions: torch.Tensor,  # 位置下标
            query: torch.Tensor,  # query 张量
            key: torch.Tensor | None = None,  # key 张量，可为 None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """forward() 的纯 PyTorch 实现。"""

        # 根据 query 的 dtype / device，匹配出合适的 cos/sin cache
        cos_sin_cache = self._match_cos_sin_cache_dtype(query)

        # 调用静态纯 PyTorch 版本
        return self.forward_static(
            positions,
            query,
            key,
            self.head_size,
            self.rotary_dim,
            cos_sin_cache,
            self.is_neox_style,
        )

    def forward_cuda(
            self,
            positions: torch.Tensor,  # 位置下标
            query: torch.Tensor,  # query 张量
            key: torch.Tensor | None = None,  # key 张量，可为 None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # 如果启用了 flashinfer 路径，就直接调用 flashinfer 的 rotary embedding custom op
        if self.use_flashinfer:
            torch.ops.cfie.flashinfer_rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
            )
            # 这个 op 是原地修改 query / key
            return query, key

        # 否则走 cfie 自己的 custom op
        from cfie import _custom_ops as ops

        # 匹配合适 dtype 的 cos/sin cache
        cos_sin_cache = self._match_cos_sin_cache_dtype(query)

        # ops.rotary_embedding() 是原地操作
        # 它会直接修改 query / key
        ops.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            cos_sin_cache,
            self.is_neox_style,
        )
        return query, key

    def forward_hip(
            self,
            positions: torch.Tensor,  # 位置下标
            query: torch.Tensor,  # query 张量
            key: torch.Tensor | None = None,  # key 张量
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # 如果启用了 aiter 路径，则走 ROCm 上专用的 Triton/aiter 实现
        if self.use_aiter:
            cos_sin_cache = self._match_cos_sin_cache_dtype(query)
            self.rocm_aiter_triton_rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                cos_sin_cache,
                self.is_neox_style,
            )
            # 同样是原地更新
            return query, key

        # 否则直接复用 CUDA 路径
        return self.forward_cuda(positions, query, key)

    def forward_xpu(
            self,
            positions: torch.Tensor,  # 位置下标
            query: torch.Tensor,  # query 张量
            key: torch.Tensor | None = None,  # key 张量
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # 先确保 cache dtype/device 匹配
        self._match_cos_sin_cache_dtype(query)

        # 当 key 为 None 时，直接退回纯 PyTorch 路径
        if key is None:
            return self.forward_native(positions, query, key)
        else:
            # key 不为 None 时，使用 custom op
            from cfie import _custom_ops as ops

            cos_sin_cache = self._match_cos_sin_cache_dtype(query)

            # 原地更新 query / key
            ops.rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                cos_sin_cache,
                self.is_neox_style,
            )
        return query, key

    def forward_cpu(
            self,
            positions: torch.Tensor,  # 位置下标
            query: torch.Tensor,  # query 张量
            key: torch.Tensor | None = None,  # key 张量
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # CPU 路径也使用 custom op
        from cfie import _custom_ops as ops

        # 匹配 cos/sin cache dtype
        cos_sin_cache = self._match_cos_sin_cache_dtype(query)

        # 原地更新 query / key
        ops.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            cos_sin_cache,
            self.is_neox_style,
        )
        return query, key

    def extra_repr(self) -> str:
        # 打印模块时额外展示的关键信息
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s
