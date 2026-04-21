# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""中文说明：与 HuggingFace 权重兼容的 Qwen3VL 仅推理实现。"""

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import lru_cache, partial
from itertools import islice
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature
from transformers.models.qwen2_vl import Qwen2VLImageProcessorFast
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    smart_resize as image_smart_resize,
)
from transformers.models.qwen3_vl import Qwen3VLProcessor, Qwen3VLVideoProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLVisionConfig,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    smart_resize as video_smart_resize,
)
from transformers.video_utils import VideoMetadata

from cfie.compilation.decorators import support_torch_compile
from cfie.config import CfieConfig
from cfie.config.multimodal import BaseDummyOptions, VideoDummyOptions
from cfie.distributed import get_pp_group, parallel_state
from cfie.logger import init_logger
from cfie.model_executor.layers.activation import _ACTIVATION_REGISTRY
from cfie.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from cfie.model_executor.layers.conv import Conv3dLayer
from cfie.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from cfie.model_executor.layers.logits_processor import LogitsProcessor
from cfie.model_executor.layers.quantization import QuantizationConfig
from cfie.model_executor.layers.rotary_embedding import get_rope
from cfie.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from cfie.model_executor.model_loader.weight_utils import default_weight_loader
from cfie.model_executor.models.module_mapping import MultiModelKeys
from cfie.multimodal import MULTIMODAL_REGISTRY
from cfie.multimodal.evs import (
    compute_mrope_for_media,
    compute_retained_tokens_count,
    compute_retention_mask,
    recompute_mrope_positions,
)
from cfie.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
    VideoItem,
)
from cfie.multimodal.parse import ImageSize, MultiModalDataItems
from cfie.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from cfie.sequence import IntermediateTensors
from cfie.tokenizers.protocol import TokenizerLike
from cfie.tokenizers.registry import cached_tokenizer_from_config
from cfie.utils.collection_utils import is_list_of
from cfie.utils.math_utils import round_up
from .interfaces import (
    MultiModalEmbeddings,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsMultiModalPruning,
    SupportsPP,
    _require_is_multimodal,
)
from .qwen2_5_vl import (
    Qwen2_5_VisionAttention,
    Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoInputs,
    Qwen2_5_VLVideoPixelInputs,
)
from .qwen2_vl import (
    Qwen2VLMultiModalDataParser,
    Qwen2VLProcessingInfo,
    _create_qwen2vl_field_factory,
)
from .qwen3 import Qwen3ForCausalLM, Qwen3Model
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from .vision import (
    get_vit_attn_backend,
    is_vit_use_data_parallel,
    run_dp_sharded_mrope_vision_model,
)

logger = init_logger(__name__)

# 中文注释：We use 2048 dummy video frames that would generate vision embeddings
# 中文注释：of the maximum size.
DUMMY_VIDEO_NUM_FRAMES = 2048


class Qwen3_VisionPatchEmbed(nn.Module):
    # 视觉 Patch Embedding 模块
    # 作用：将输入的图像/视频 patch 映射为 hidden_size 维的视觉 token 表示

    def __init__(
            self,
            patch_size: int = 14,  # 空间 patch 大小，例如 14x14
            temporal_patch_size: int = 2,  # 时间维 patch 大小，例如连续 2 帧作为一个时间 patch
            in_channels: int = 3,  # 输入通道数，RGB 通常为 3
            hidden_size: int = 1152,  # 输出的视觉 token 维度
    ) -> None:
        # 初始化 nn.Module 基类
        super().__init__()

        # 保存空间 patch 大小
        self.patch_size = patch_size

        # 保存时间维 patch 大小
        self.temporal_patch_size = temporal_patch_size

        # 保存输出特征维度
        self.hidden_size = hidden_size

        # Conv3D 的卷积核大小：
        # (时间维 patch 大小, 空间高 patch 大小, 空间宽 patch 大小)
        kernel_size = (temporal_patch_size, patch_size, patch_size)

        # 使用 3D 卷积完成 patch 投影
        # kernel_size = stride 表示按 patch 大小无重叠切块
        # 输入通道数为 in_channels，输出通道数为 hidden_size
        self.proj = Conv3dLayer(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 的形状是 [L, C]
        # 其中：
        #   L 表示 patch/token 数量
        #   C 表示每个 patch 展平后的像素长度
        L, C = x.shape

        # 将每个展平 patch 恢复为 5 维形状：
        # [L, in_channels, temporal_patch_size, patch_size, patch_size]
        # 其中 -1 会自动推导为 in_channels
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)

        # 通过 3D 卷积将每个 patch 投影到 hidden_size 维
        # 由于 kernel_size 和 stride 完全一致，
        # 每个 patch 最终只产生一个输出位置
        # 输出形状可视为 [L, hidden_size, 1, 1, 1]
        # 再 reshape 成 [L, hidden_size]
        x = self.proj(x).view(L, self.hidden_size)

        # 返回每个 patch 对应的视觉 token 表示
        return x


class Qwen3_VisionMLP(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            bias: bool = False,
            act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
    ):
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.linear_fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc1",
            disable_tp=use_data_parallel,
        )
        self.linear_fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc2",
            disable_tp=use_data_parallel,
        )
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        mlp_output = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return mlp_output


class Qwen3_VisionBlock(nn.Module):
    def __init__(
            self,
            dim: int,  # token 特征维度 = hidden size = embed dim
            num_heads: int,  # attention 头数
            mlp_hidden_dim: int,  # MLP 中间层维度
            act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,  # 激活函数
            norm_layer: Callable[[int], nn.Module] | None = None,  # 归一化层构造器，输入 int(dim) 返回一个 norm 模块
            quant_config: QuantizationConfig | None = None,  # 量化配置
            prefix: str = "",  # 模块名前缀
    ) -> None:
        super().__init__()

        # 默认使用 LayerNorm(dim, eps=1e-6)
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # 两个归一化层
        # 输入输出形状都不变：[..., dim] -> [..., dim]
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # 注意力模块
        # 典型输入输出形状：
        # [N_all, dim] -> [N_all, dim]
        self.attn = Qwen2_5_VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        # MLP 模块
        # 典型内部形状：
        # [N_all, dim] -> [N_all, mlp_hidden_dim] -> [N_all, dim]
        self.mlp = Qwen3_VisionMLP(
            dim,
            mlp_hidden_dim,
            act_fn=act_fn,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
            self,
            x: torch.Tensor,  # 形状: [N_all, dim]
            # N_all 表示当前 batch 中所有视觉 token 拼接后的总数

            cu_seqlens: torch.Tensor,  # 形状: [num_seq + 1]
            # 每条图像/帧序列在 x 中的前缀和边界
            # 例如 [0, 16, 40, 56]

            rotary_pos_emb_cos: torch.Tensor,  # [N_all, rotary_dim/2]
            rotary_pos_emb_sin: torch.Tensor,  # [N_all, rotary_dim/2]

            max_seqlen: torch.Tensor,  # 标量，或 shape: []
            # 当前 batch 中最长那条视觉序列长度
            # 仅某些 FlashAttention 后端需要

            sequence_lengths: torch.Tensor,  # 形状常见: [num_seq]
            # 每条序列的实际长度
            # 例如 [16, 24, 16]
            # 仅某些 FlashInfer/CuDNN 后端需要
    ) -> torch.Tensor:
        # [N_all, dim] + [N_all, dim] -> [N_all, dim]
        x = x + self.attn(
            self.norm1(x),  # [N_all, dim] -> [N_all, dim]
            cu_seqlens=cu_seqlens, # [num_seq + 1]
            rotary_pos_emb_cos=rotary_pos_emb_cos, #  [N_all, rotary_dim/2]
            rotary_pos_emb_sin=rotary_pos_emb_sin, #  [N_all, rotary_dim/2]
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )

        # self.norm2(x)
        # [N_all, dim] -> [N_all, dim]
        #
        # self.mlp(...)
        # [N_all, dim] -> [N_all, dim]
        #
        # 残差相加后形状仍不变
        # [N_all, dim] + [N_all, dim] -> [N_all, dim]
        x = x + self.mlp(self.norm2(x))

        # 输出形状: [N_all, dim]
        return x


class Qwen3_VisionPatchMerger(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            norm_layer: Callable[[int], nn.Module] | None = None,
            spatial_merge_size: int = 2,
            use_postshuffle_norm: bool = False,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.hidden_size = context_dim * (spatial_merge_size ** 2)

        self.use_postshuffle_norm = use_postshuffle_norm
        if self.use_postshuffle_norm:
            context_dim = self.hidden_size

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(context_dim)
        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc1",
            disable_tp=use_data_parallel,
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(
            self.hidden_size,
            d_model,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc2",
            disable_tp=use_data_parallel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)

        x_parallel, _ = self.linear_fc1(x)
        x_parallel = self.act_fn(x_parallel)
        out, _ = self.linear_fc2(x_parallel)
        return out


class Qwen3_VisionTransformer(nn.Module):
    # Qwen3 的视觉 Transformer 主体
    # 负责把图像/视频输入编码成视觉特征，供后续语言模型使用

    def __init__(
            self,
            vision_config: Qwen3VLVisionConfig,  # 视觉模块配置
            norm_eps: float = 1e-6,  # LayerNorm 的 eps
            quant_config: QuantizationConfig | None = None,  # 量化配置
            prefix: str = "",  # 参数名前缀
    ) -> None:
        # 初始化 nn.Module 基类
        super().__init__()

        # 视觉隐藏维度
        self.hidden_size = vision_config.hidden_size

        # 注意力头数
        self.num_heads = vision_config.num_heads

        # 位置 embedding 总表长
        self.num_position_embeddings = vision_config.num_position_embeddings

        # 空间 patch 尺寸
        self.patch_size = vision_config.patch_size

        # 空间 merge 尺寸
        # 后续会把 spatial_merge_size x spatial_merge_size 个 patch 做聚合
        self.spatial_merge_size = vision_config.spatial_merge_size

        # 一个空间 merge 单元包含的 patch 数量
        self.spatial_merge_unit = self.spatial_merge_size ** 2

        # 时间维 patch 尺寸，视频场景会用到
        self.temporal_patch_size = vision_config.temporal_patch_size

        # 指定哪些视觉层需要做 deepstack 特征提取
        # 若配置中没有该字段，则默认为空列表
        self.deepstack_visual_indexes = (
            vision_config.deepstack_visual_indexes
            if hasattr(vision_config, "deepstack_visual_indexes")
            else []
        )

        # 假设位置 embedding 对应一个正方形网格
        # 这里计算每边网格数
        self.num_grid_per_side = int(self.num_position_embeddings ** 0.5)

        # 判断视觉塔是否走数据并行
        use_data_parallel = is_vit_use_data_parallel()

        # 若视觉塔走数据并行，则视觉塔自身的 TP 大小视为 1
        # 否则取当前全局 tensor parallel world size
        self.tp_size = (
            1
            if use_data_parallel
            else parallel_state.get_tensor_model_parallel_world_size()
        )

        # NOTE:
        # 这里用于为 DP ViT 的 all_gather 创建空张量时确定输出维度
        # 若启用了 deepstack，则输出维度要乘上 (1 + deepstack层数)
        self.out_hidden_size = vision_config.out_hidden_size * (
                1 + len(self.deepstack_visual_indexes)
        )

        # Patch Embedding 模块
        # 将输入图像/视频分块后映射到 hidden_size 维空间
        self.patch_embed = Qwen3_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )

        # 可学习的位置 embedding 表
        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)

        # 构造 norm 层工厂
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)

        # 每个注意力头的维度
        head_dim = self.hidden_size // self.num_heads

        # 构造视觉侧使用的 RoPE
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"partial_rotary_factor": 0.5},
        )

        # 最终输出使用的 patch merger
        # 将视觉 block 的输出进一步聚合到 out_hidden_size
        self.merger = Qwen3_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,  # merger 输出维度
            context_dim=self.hidden_size,  # 输入上下文维度
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )

        # 为 deepstack 中间层输出准备多个 merger
        # 每个被选中的 deepstack 层对应一个 merger
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3_VisionPatchMerger(
                    d_model=vision_config.out_hidden_size,
                    context_dim=self.hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                    use_postshuffle_norm=True,  # deepstack 版本额外启用 postshuffle norm
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.deepstack_merger_list.{layer_idx}",
                )
                for layer_idx in range(len(self.deepstack_visual_indexes))
            ]
        )

        # 在 Vision Transformer 初始化时，选择当前视觉注意力要使用的后端实现
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,  # 每个 attention head 的维度
            dtype=torch.get_default_dtype(),  # 当前默认计算 dtype，例如 float16 / bfloat16 / float32
        )

        # 构造视觉 Transformer block 列表
        self.blocks = nn.ModuleList(
            [
                Qwen3_VisionBlock(
                    dim=self.hidden_size,  # 隐藏维度
                    num_heads=self.num_heads,  # 头数
                    mlp_hidden_dim=vision_config.intermediate_size,  # MLP 中间层维度
                    act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],  # 激活函数
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(vision_config.depth)
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        # 当前视觉模块参数所使用的 dtype
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        # 当前视觉模块所在 device
        return self.patch_embed.proj.weight.device

    @staticmethod
    @lru_cache(maxsize=1024)
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        # 生成二维旋转位置编码索引
        # 返回 shape 为 [h*w, 2]，每个位置包含 (h_pos, w_pos)

        # 先生成 h 方向坐标矩阵，形状 [h, w]
        # 第 i 行的所有元素都等于 i
        # 一共 h 行、w 列
        hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))

        # merge 后的高、宽块数
        h_div = h // spatial_merge_size
        w_div = w // spatial_merge_size

        # 按 merge block 重排
        hpos_ids = hpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )

        # [h_div, w_div,  spatial_merge_size,  spatial_merge_size]
        hpos_ids = hpos_ids.transpose(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        # 同理生成 w 方向坐标矩阵
        wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        wpos_ids = wpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.transpose(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()

        # 拼成 [N, 2]，每一项为 (h_pos, w_pos)
        return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))

    def rot_pos_emb(self, grid_thw: list[list[int]]):
        # 根据 grid_thw 生成视觉 RoPE 所需的 cos/sin
        # grid_thw 中每个元素形如 [t, h, w]

        # 找到所有样本中最大的 h/w，用于一次性取出足够长的 cos/sin cache
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)

        # 为每个样本生成位置 id
        # 若 t == 1 表示图像，直接生成一次
        # 若 t > 1 表示视频，对同一空间位置重复 t次
        pos_ids = [
            # 拼成 [N, 2]，每一项为 (h_pos, w_pos)
            self.rot_pos_ids(h, w, self.spatial_merge_size)
            if t == 1
            else self.rot_pos_ids(h, w, self.spatial_merge_size).repeat(t, 1)
            for t, h, w in grid_thw
        ]

        # 拼接全部样本的位置 id 并搬到当前 device
        # [N_all, 2]
        pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True)

        # 使用 RotaryEmbedding 中预先缓存好的 cos/sin
        # [max_grid_size, rotary_dim/2]
        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)

        # 取出对应位置的 cos/sin，并展平成后续注意力使用的格式
        # [N_all, 2, rotary_dim/2] --> [N_all, rotary_dim]
        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)

        return cos_combined, sin_combined

    def fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
        # 对可学习位置 embedding 做快速双线性插值
        # 使训练时固定网格的位置表能适配当前任意 h, w

        # 原始位置表每边网格数
        # 标量
        num_grid_per_side = self.num_grid_per_side

        # spatial merge 大小
        # 标量
        m_size = self.spatial_merge_size

        # 位置 embedding 维度
        # 标量
        hidden_dim = self.pos_embed.embedding_dim

        # 保存每个样本插值后的位置 embedding
        # 列表中每项形状大致为 [t * (h*w), D]
        outputs = []

        for t, h, w in grid_thw:
            # t: 时间帧数
            # h: 当前样本网格高
            # w: 当前样本网格宽
            # N = h * w

            # 在原始位置表坐标范围 [0, G-1] 内均匀采样 h 个纵向坐标
            # shape: [h]
            h_idxs = torch.linspace(
                0, num_grid_per_side - 1, h, dtype=torch.float32, device=self.device
            )

            # 在原始位置表坐标范围 [0, G-1] 内均匀采样 w 个横向坐标
            # shape: [w]
            w_idxs = torch.linspace(
                0, num_grid_per_side - 1, w, dtype=torch.float32, device=self.device
            )

            # 双线性插值的 floor / ceil 索引
            # h_floor, h_ceil: [h]
            # w_floor, w_ceil: [w]
            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

            # 与 floor 的距离，用于计算插值权重
            # dh: [h]
            # dw: [w]
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            # 为所有 h,w 点构造 meshgrid 视图
            # dh_grid, dw_grid: [h, w]
            # h_floor_grid, w_floor_grid: [h, w]
            # h_ceil_grid, w_ceil_grid: [h, w]
            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            # 原始四个角点权重计算公式：
            # w00 = (1 - dh_grid) * (1 - dw_grid)
            # w01 = (1 - dh_grid) * dw_grid
            # w10 = dh_grid * (1 - dw_grid)
            # w11 = dh_grid * dw_grid
            #
            # 这里复用 w11，减少重复乘法
            #
            # 四个权重的 shape 都是: [h, w]
            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            # 四个角的 h / w 索引
            # h_grid: [4, h, w]
            # w_grid: [4, h, w]
            #
            # 4 个位置依次对应：
            # 0: (floor_h, floor_w) 左上
            # 1: (floor_h, ceil_w)  右上
            # 2: (ceil_h,  floor_w) 左下
            # 3: (ceil_h,  ceil_w)  右下
            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])

            # 转成一维 embedding 表索引
            # h_grid_idx: [4, h, w]
            h_grid_idx = h_grid * num_grid_per_side

            # indices: [4, h*w] = [4, N]
            # 每列对应一个目标位置的四个角点 embedding 索引
            indices = (h_grid_idx + w_grid).reshape(4, -1)

            # 四个角对应的权重
            # 先 stack 后 shape: [4, h, w]
            # reshape 后 shape: [4, h*w, 1] = [4, N, 1]
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
            weights = weights.to(dtype=self.dtype)

            # 从位置 embedding 表中取四个角的 embedding
            # self.pos_embed(indices) 的 shape: [4, N, D]
            embeds = self.pos_embed(indices)

            # 乘对应权重（广播乘法，不是矩阵乘法）
            # embeds:  [4, N, D]
            # weights: [4, N, 1]
            # 结果仍是 [4, N, D]
            embeds *= weights

            # 对四个角求和，得到最终插值结果
            # combined: [N, D] = [h*w, D]
            combined = embeds.sum(dim=0)

            # 按 spatial_merge_size 重排，和 patch merge 的 token 顺序对齐
            #
            # 第一步 reshape:
            # [N, D] -> [h//m, m, w//m, m, D]
            combined = combined.reshape(
                h // m_size, m_size, w // m_size, m_size, hidden_dim
            )

            # 第二步 permute:
            # [h//m, m, w//m, m, D]
            # -> [h//m, w//m, m, m, D]
            combined = combined.permute(0, 2, 1, 3, 4)

            # 第三步 reshape:
            # -> [1, N, D]
            combined = combined.reshape(1, -1, hidden_dim)

            # 若 t > 1（视频），沿时间维重复 t 次
            # expand 后:  [t, N, D]
            # reshape 后: [t*N, D]
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)

            # repeated: [t*h*w, D]
            outputs.append(repeated)

        # 拼接所有样本
        # 若 batch 中各样本 token 总数之和为 M
        # 返回 shape: [M, D]
        return torch.cat(outputs, dim=0)

    def forward(
            self,
            x: torch.Tensor,  # 输入图像/视频张量
            grid_thw: torch.Tensor | list[list[int]],  # 每个样本的 [t, h, w] 网格信息
    ) -> torch.Tensor:
        # 将输入搬到视觉模块所在 device / dtype
        hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)

        # 将原始像素输入转成 patch token
        hidden_states = self.patch_embed(hidden_states)

        # 统一把 grid_thw 转成 list 和 numpy 两种形式，便于后续处理
        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = np.array(grid_thw, dtype=np.int32)
        else:
            grid_thw_list = grid_thw.tolist()
            grid_thw = grid_thw.numpy()

        # 计算插值后的位置 embedding，并加到 patch token 上
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw_list)
        # [N, hidden_size]
        hidden_states = hidden_states + pos_embeds

        # 计算视觉 RoPE 所需的 cos/sin
        # 拼成 [N,rotary_dim]
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

        # 计算 cu_seqlens：
        # 每帧 token 数为 h*w，视频重复 t 次，再做前缀和

        # grid_thw = [
        #     [2, 3, 4],   # t=2, h=3, w=4
        #     [1, 5, 6],   # t=1, h=5, w=6
        # ]
        # --> [12, 30] --> [12, 12, 30] --> [12, 24, 54]
        cu_seqlens = np.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            axis=0, dtype=np.int32
        )

        # --> [0, 12, 24, 54]
        # 前面补 0，形成标准 cu_seqlens 格式
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])

        # 某些注意力后端需要 sequence lengths
        sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend,
            cu_seqlens,
            self.device
        )

        # 当前 batch 内最大序列长度
        max_seqlen = torch.tensor(
            MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu_seqlens),
            dtype=torch.int32,
            device=self.device,
        )

        # 根据 backend / hidden_size / tp_size 需要，可能对 cu_seqlens 重新整理
        cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend,
            cu_seqlens,
            self.hidden_size,
            self.tp_size,
            self.device,
        )

        # 增加一个维度，适配后续 block 的输入格式
        hidden_states = hidden_states.unsqueeze(1)

        # 用来收集 deepstack 中间层特征
        deepstack_feature_lists = []

        # 依次通过所有视觉 Transformer block
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen,  # FDSP
                sequence_lengths=sequence_lengths,
            )

            # 如果当前层被配置为 deepstack 层
            # 则将该层输出通过对应的 merger 保存为额外视觉特征
            if layer_num in self.deepstack_visual_indexes:
                deepstack_merger_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[deepstack_merger_idx](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        # 最后一层输出经过主 merger
        hidden_states = self.merger(hidden_states)

        # 将主输出与所有 deepstack 输出在特征维拼接
        # 最终形状近似为：
        # [seq_len, hidden_size * (1 + deepstack层数)]
        hidden_states = torch.cat(
            [hidden_states] + deepstack_feature_lists, dim=1
        )

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # 该映射用于将 checkpoint 中分开的 q/k/v 参数
        # 加载到内部融合的 qkv 参数中
        stacked_params_mapping = [
            # (内部参数名片段, checkpoint 参数名片段, shard_id)
            ("attn.qkv.", "attn.q.", "q"),
            ("attn.qkv.", "attn.k.", "k"),
            ("attn.qkv.", "attn.v.", "v"),
        ]

        # 当前模块的参数字典
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        # 记录成功加载的参数名
        loaded_params: set[str] = set()

        # 遍历传入的所有权重
        for name, loaded_weight in weights:
            # 先尝试是否属于 q/k/v 融合加载场景
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                # 把 checkpoint 名字替换为内部参数名字
                name = name.replace(weight_name, param_name)

                # 取到目标参数
                param = params_dict[name]

                # 调用参数自带的 weight_loader，按 shard_id 加载
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # 非 q/k/v 融合参数，走普通加载逻辑
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            # 记录已加载参数
            loaded_params.add(name)

        return loaded_params


class Qwen3VLProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3VLConfig)

    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_image_processor(self, **kwargs: object) -> Qwen2VLImageProcessorFast:
        return self.get_hf_processor(**kwargs).image_processor

    def get_video_processor(self, **kwargs: object) -> Qwen3VLVideoProcessor:
        return self.get_hf_processor(**kwargs).video_processor

    def get_data_parser(self):
        return Qwen2VLMultiModalDataParser(
            self.get_hf_config().vision_config.spatial_merge_size,
            video_needs_metadata=True,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def _get_vision_info(
            self,
            *,
            image_width: int,
            image_height: int,
            num_frames: int = 2,
            do_resize: bool = True,
            image_processor: Qwen2VLImageProcessorFast | Qwen3VLVideoProcessor,
            mm_kwargs: Mapping[str, object],
    ) -> tuple[ImageSize, int]:
        is_video = isinstance(image_processor, Qwen3VLVideoProcessor)

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        mm_kwargs = self.ctx.get_merged_mm_kwargs(mm_kwargs)
        size = image_processor.size
        if override_size := mm_kwargs.get("size"):
            size = size | override_size
        if (override_min_pixels := mm_kwargs.get("min_pixels")) is not None:
            size = size | {"shortest_edge": override_min_pixels}
        if (override_max_pixels := mm_kwargs.get("max_pixels")) is not None:
            size = size | {"longest_edge": override_max_pixels}

        if do_resize:
            if is_video:
                smart_resize = video_smart_resize
                extra_kwargs = {
                    "num_frames": num_frames,
                    "temporal_factor": temporal_patch_size,
                }
            else:
                smart_resize = image_smart_resize
                extra_kwargs = {}

            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=size["shortest_edge"],
                max_pixels=size["longest_edge"],
                **extra_kwargs,
            )
            preprocessed_size = ImageSize(width=resized_width, height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width, height=image_height)

        padded_num_frames = round_up(num_frames, temporal_patch_size)

        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size ** 2)

        return preprocessed_size, num_vision_tokens

    def _get_max_video_frames(self, max_tokens: int, start_num_frames: int = 2) -> int:
        return super()._get_max_video_frames(
            max_tokens, start_num_frames=start_num_frames
        )

    def get_num_frames_with_most_features(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
    ) -> int:
        return super().get_num_frames_with_most_features(
            seq_len, mm_counts, max_frames_per_video=DUMMY_VIDEO_NUM_FRAMES
        )

    def get_max_video_tokens(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
    ) -> int:
        video_processor = self.get_video_processor()

        mm_kwargs = self.ctx.get_merged_mm_kwargs({})
        video_size = mm_kwargs.get("size", video_processor.size)
        temporal_patch_size = mm_kwargs.get(
            "temporal_patch_size", video_processor.temporal_patch_size
        )

        # 中文注释：video_max_pixels contains the temporal compression factor,
        # 中文注释：so we divide by 2 to get the maximum number of image pixels.
        video_max_pixels = video_size["longest_edge"]
        target_width, target_height = self.get_image_size_with_most_features(
            max_pixels=video_max_pixels // temporal_patch_size
        )
        num_video_soft_tokens = self.get_num_video_tokens(
            image_width=target_width,
            image_height=target_height,
            num_frames=2,
            image_processor=video_processor,
            mm_kwargs={},
        )
        return num_video_soft_tokens

    def _calculate_timestamps(
            self, indices: list[int] | torch.Tensor, video_fps: float, merge_size: int
    ):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            # 中文注释：don't update metadata's frames_indices directly
            indices = indices + [indices[-1]] * (merge_size - len(indices) % merge_size)
        timestamps = [idx / video_fps for idx in indices]
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2
            for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    def _get_video_second_idx(
            self,
            metadata: dict[str, Any],
            do_sample_frames: bool | None = None,
            sampled_fps: float | None = None,
            sampled_num_frames: int | None = None,
    ) -> list[int]:
        video_processor = self.get_video_processor()
        merge_size = video_processor.merge_size
        indices = metadata["frames_indices"]

        # 中文注释：metadata["fps"] refers to the true fps of the input video.
        video_fps = metadata["fps"]
        if do_sample_frames is None:
            do_sample_frames = metadata.get("do_sample_frames", False)

        # 中文注释：If video frames are sampled in HF processor (instead of vLLM
        # 中文注释：video loader), we need to re-calculate the indices from original
        # 中文注释：metadata.
        if do_sample_frames:
            total_num_frames = metadata["total_num_frames"]

            # 中文注释：When num_frames is explicitly provided, use it directly
            # 中文注释：instead of computing from fps. This mirrors the behavior of
            # 中文注释：HF's Qwen3VLVideoProcessor.sample_frames where num_frames
            # 中文注释：and fps are mutually exclusive.
            if sampled_num_frames is not None:
                num_frames = sampled_num_frames
            else:
                # 中文注释：here video_fps is the fps of the sampled video, and
                # 中文注释：metadata["fps"] refers to the fps of the original video.
                sampled_fps = sampled_fps if sampled_fps else video_processor.fps
                num_frames = int(total_num_frames / metadata["fps"] * sampled_fps)

            num_frames = min(
                min(
                    max(num_frames, video_processor.min_frames),
                    video_processor.max_frames,
                ),
                total_num_frames,
            )
            indices = (
                np.linspace(0, total_num_frames - 1, num_frames)
                .round()
                .astype(int)
                .tolist()
            )
        timestamps = self._calculate_timestamps(indices, video_fps, merge_size)
        return timestamps


class Qwen3VLDummyInputsBuilder(BaseDummyInputsBuilder[Qwen3VLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        video_token = "<|vision_start|><|video_pad|><|vision_end|>"

        return image_token * num_images + video_token * num_videos

    def get_dummy_mm_data(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
            mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        image_overrides = mm_options.get("image")
        video_overrides = mm_options.get("video")

        target_image_width, target_image_height = (
            self.info.get_image_size_with_most_features()
        )

        # 中文注释：treat videos as special images
        target_num_frames = 2
        if video_overrides:
            assert isinstance(video_overrides, VideoDummyOptions)
            num_frames_override = video_overrides.num_frames
            if num_frames_override:
                if num_frames_override > target_num_frames:
                    logger.warning(
                        "video.num_frames override (%d) exceeds model's "
                        "maximum number of frames (%d), will be ignored",
                        num_frames_override,
                        target_num_frames,
                    )
                if num_frames_override < 2:
                    logger.warning(
                        "video.num_frames override (%d) cannot be less "
                        "than 2, will be ignored",
                        num_frames_override,
                    )
                target_num_frames = min(target_num_frames, num_frames_override)
        target_num_frames = max(target_num_frames, 2)

        video_processor = self.info.get_video_processor()

        mm_kwargs = self.info.ctx.get_merged_mm_kwargs({})
        video_size = mm_kwargs.get("size", video_processor.size)
        temporal_patch_size = mm_kwargs.get(
            "temporal_patch_size", video_processor.temporal_patch_size
        )

        # 中文注释：video_max_pixels contains the temporal compression factor,
        # 中文注释：so we divide by 2 to get the maximum number of image pixels.
        video_max_pixels = video_size["longest_edge"]
        target_video_width, target_video_height = (
            self.info.get_image_size_with_most_features(
                max_pixels=video_max_pixels // temporal_patch_size
            )
        )
        target_video_size, _ = self.info._get_vision_info(
            image_width=target_video_width,
            image_height=target_video_height,
            num_frames=target_num_frames,
            image_processor=video_processor,
            mm_kwargs={},
        )
        # 中文注释：NOTE: we need to do this check here since Qwen3-VL resizes video
        # 中文注释：frames depending on how many frames there are.
        target_video_width, target_video_height = (
            target_video_size.width,
            target_video_size.height,
        )
        if video_overrides:
            assert isinstance(video_overrides, VideoDummyOptions)
            width_override = video_overrides.width
            if width_override:
                if width_override > target_video_width:
                    logger.warning(
                        "video.width override (%d) exceeds model's "
                        "maximum width (%d), will be ignored",
                        width_override,
                        target_video_width,
                    )
                target_video_width = min(target_video_width, width_override)
            height_override = video_overrides.height
            if height_override:
                if height_override > target_video_height:
                    logger.warning(
                        "video.height override (%d) exceeds model's "
                        "maximum height (%d), will be ignored",
                        height_override,
                        target_video_height,
                    )
                target_video_height = min(target_video_height, height_override)

        return {
            "image": self._get_dummy_images(
                width=target_image_width,
                height=target_image_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
            "video": self._get_dummy_videos(
                width=target_video_width,
                height=target_video_height,
                num_frames=target_num_frames,
                num_videos=num_videos,
            ),
        }

    def _get_dummy_videos(
            self,
            *,
            width: int,
            height: int,
            num_frames: int,
            num_videos: int,
    ) -> list[VideoItem]:
        video = np.full((num_frames, width, height, 3), 255, dtype=np.uint8)
        video_items = []
        for i in range(num_videos):
            video_metadata = {
                "fps": 2.0,
                "duration": num_frames / 2.0,
                "total_num_frames": num_frames,
                "frames_indices": [i for i in range(num_frames)],
                "video_backend": "opencv",
                "do_sample_frames": False,
            }
            video_item = (video.copy(), video_metadata)
            video_items.append(video_item)
        return video_items


class Qwen3VLMultiModalProcessor(BaseMultiModalProcessor[Qwen3VLProcessingInfo]):
    def _call_hf_processor(
            self,
            prompt: str,
            mm_data: Mapping[str, object],
            mm_kwargs: Mapping[str, object],
            tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        processor = self.info.get_hf_processor(**mm_kwargs)

        # 中文注释：Separate video processing from image processing. Because the videos
        # 中文注释：are processed into several image patches
        if videos := mm_data.pop("videos", []):
            video_grid_thw_lst = []
            pixel_values_videos_lst = []
            timestamps_per_video = []

            for item in videos:
                video_array, metadata = item

                # 中文注释：NOTE: @JJJYmmm new attr metadata.frames_indices indicates
                # 中文注释：the sampled frames indices of pre-sampled videos, which is
                # 中文注释：used to calculate the timestamps. Make sure that
                # 中文注释：do_sample_frames in mm_kwargs is false for presampled videos.

                # 中文注释：NOTE: a copy of is created to update do_sample_frames,
                # 中文注释：otherwise mm_hash for the object will be incorrect.
                video_mm_kwargs = dict(**mm_kwargs)
                if "do_sample_frames" not in video_mm_kwargs:
                    # 中文注释：qwen_vl_utils already has "do_sample_frames" in
                    # 中文注释：mm_kwargs, don't overwrite it.
                    video_mm_kwargs["do_sample_frames"] = metadata.get(
                        "do_sample_frames", False
                    )

                metadata = VideoMetadata(
                    **{k: metadata[k] for k in metadata if k != "do_sample_frames"}
                )

                # 中文注释：Compute timestamps here where we have access to metadata
                timestamps = self.info._get_video_second_idx(
                    metadata=metadata,
                    do_sample_frames=video_mm_kwargs["do_sample_frames"],
                    sampled_fps=video_mm_kwargs.get("fps"),
                    sampled_num_frames=video_mm_kwargs.get("num_frames"),
                )
                timestamps_per_video.append(timestamps)

                video_mm_data = dict()
                video_mm_data["videos"] = [[video_array]]
                video_mm_data["video_metadata"] = [[metadata]]

                # 中文注释：When num_frames is specified, explicitly set fps=None
                # 中文注释：to prevent HF's BaseVideoProcessor.preprocess() from
                # 中文注释：filling in the class default (fps=2) via setdefault(),
                # 中文注释：which would conflict with num_frames (mutually exclusive).
                if "num_frames" in video_mm_kwargs and "fps" not in video_mm_kwargs:
                    video_mm_kwargs["fps"] = None

                video_outputs = super()._call_hf_processor(
                    prompt="<|vision_start|><|video_pad|><|vision_end|>",
                    mm_data=video_mm_data,
                    mm_kwargs=video_mm_kwargs,
                    tok_kwargs=tok_kwargs,
                )

                merge_size = processor.video_processor.merge_size
                # 中文注释：Get video grid info for EVS calculation.
                video_grid_thw = video_outputs["video_grid_thw"]
                num_frames = int(video_grid_thw[0, 0])
                tokens_per_frame_base = int(video_grid_thw[0, 1:].prod()) // (
                        merge_size ** 2
                )

                # 中文注释：Apply EVS if enabled.
                video_pruning_rate = self.info.ctx.get_mm_config().video_pruning_rate
                if video_pruning_rate is not None and video_pruning_rate > 0.0:
                    num_tokens = compute_retained_tokens_count(
                        tokens_per_frame=tokens_per_frame_base,
                        num_frames=num_frames,
                        q=video_pruning_rate,
                    )
                    # 中文注释：Here we just need placeholders that won't actually be replaced -
                    # 中文注释：we just need to make sure the total number of tokens is correct
                    # 中文注释：assign all tokens to the first frame.
                    tokens_per_frame = [num_tokens] + [0] * (num_frames - 1)
                    select_token_id = False
                else:
                    tokens_per_frame = [tokens_per_frame_base] * num_frames
                    select_token_id = True

                # 中文注释：Generate the video replacement with EVS-adjusted token counts
                tokenizer = self.info.get_tokenizer()
                hf_config = self.info.get_hf_config()
                video_repl = Qwen3VLMultiModalProcessor.get_video_repl(
                    tokens_per_frame=tokens_per_frame,
                    timestamps=timestamps,
                    tokenizer=tokenizer,
                    vision_start_token_id=hf_config.vision_start_token_id,
                    vision_end_token_id=hf_config.vision_end_token_id,
                    video_token_id=hf_config.video_token_id,
                    select_token_id=select_token_id,
                )

                # 中文注释：Convert token IDs to text for the HF processor flow
                video_placeholder = tokenizer.decode(
                    video_repl.full, skip_special_tokens=False
                )
                input_ids = video_outputs.pop("input_ids")
                video_placeholder = processor.tokenizer.batch_decode(input_ids)[0]
                prompt = prompt.replace(
                    "<|vision_start|><|video_pad|><|vision_end|>",
                    video_placeholder,
                    1,
                )

                video_grid_thw_lst.append(video_outputs["video_grid_thw"])
                pixel_values_videos_lst.append(video_outputs["pixel_values_videos"])
            video_outputs = dict(
                pixel_values_videos=torch.cat(pixel_values_videos_lst),
                video_grid_thw=torch.cat(video_grid_thw_lst),
                timestamps=timestamps_per_video,
            )
        else:
            video_outputs = dict()

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        combined_outputs = dict(
            processed_outputs,
            **video_outputs,
        )
        return BatchFeature(combined_outputs)

    def _get_mm_fields_config(
            self,
            hf_inputs: BatchFeature,
            hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _create_qwen2vl_field_factory(
            self.info.get_hf_config().vision_config.spatial_merge_size
        )(hf_inputs)

    def _get_prompt_updates(
            self,
            mm_items: MultiModalDataItems,
            hf_processor_mm_kwargs: Mapping[str, Any],
            out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        hf_config = self.info.get_hf_config()

        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        vision_end_token_id = hf_config.vision_end_token_id

        merge_length = image_processor.merge_size ** 2

        def get_image_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            return [hf_processor.image_token_id] * num_tokens

        def get_video_replacement_qwen3vl(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            sampled_fps = hf_processor_mm_kwargs.get("fps")
            if is_list_of(sampled_fps, float):
                sampled_fps = sampled_fps[item_idx]

            timestamps = out_item["timestamps"].data
            assert len(timestamps) == grid_thw[0], (
                f"The timestamps length({len(timestamps)}) should be equal "
                f"video length ({grid_thw[0]})."
            )

            # 中文注释：Compute tokens per frame, with EVS support
            num_frames = int(grid_thw[0])
            tokens_per_frame_base = int(grid_thw[1:].prod()) // merge_length

            video_pruning_rate = self.info.ctx.get_mm_config().video_pruning_rate
            if video_pruning_rate is not None and video_pruning_rate > 0.0:
                num_tokens = compute_retained_tokens_count(
                    tokens_per_frame=tokens_per_frame_base,
                    num_frames=num_frames,
                    q=video_pruning_rate,
                )
                tokens_per_frame = [num_tokens] + [0] * (num_frames - 1)
                select_token_id = False
            else:
                tokens_per_frame = [tokens_per_frame_base] * num_frames
                select_token_id = True

            return Qwen3VLMultiModalProcessor.get_video_repl(
                tokens_per_frame=tokens_per_frame,
                timestamps=timestamps,
                tokenizer=tokenizer,
                vision_start_token_id=vision_start_token_id,
                vision_end_token_id=vision_end_token_id,
                video_token_id=video_token_id,
                select_token_id=select_token_id,
            )

        return [
            PromptReplacement(
                modality="image",
                target=hf_processor.image_token,
                replacement=get_image_replacement_qwen3vl,
            ),
            # 中文注释：NOTE: We match string on purpose since searching sequence of
            # 中文注释：token ids takes more time.
            PromptReplacement(
                modality="video",
                target="<|vision_start|><|video_pad|><|vision_end|>",
                replacement=get_video_replacement_qwen3vl,
            ),
        ]

    @staticmethod
    def get_video_repl(
            *,
            tokens_per_frame: list[int],
            timestamps: list[float | int],
            tokenizer: TokenizerLike,
            vision_start_token_id: int,
            vision_end_token_id: int,
            video_token_id: int,
            select_token_id: bool = False,
    ) -> PromptUpdateDetails[list[int]]:
        """构建 Qwen3VL 视频格式的 prompt 替换序列。

        中文说明：The replacement structure for each frame is:
        中文说明：timestamp_tokens + vision_start_token + video_tokens + vision_end_token

        中文说明：Args:
            中文说明：tokens_per_frame: Number of video tokens per frame (can vary per frame for
                中文说明：EVS).
            中文说明：timestamps: List of timestamps in seconds for each frame
            中文说明：tokenizer: Tokenizer to encode timestamp strings
            中文说明：vision_start_token_id: Token ID for vision start marker
            中文说明：vision_end_token_id: Token ID for vision end marker
            中文说明：video_token_id: Token ID for video content

        中文说明：Returns:
            中文说明：PromptUpdateDetails with full token sequence
        """
        assert len(timestamps) == len(tokens_per_frame), (
            "timestamps and tokens_per_frame must have the same length"
        )

        # 中文注释：Tokenize timestamp strings independently to avoid tokenizer merging
        # 中文注释：tokens across boundaries.
        # 中文注释：TODO: switch to `_seq2tokens` which has some caching.
        timestamp_token_ids = [
            tokenizer.encode(f"<{timestamp:.1f} seconds>", add_special_tokens=False)
            for timestamp in timestamps
        ]

        # 中文注释：Build the full token sequence
        all_token_ids = []
        for frame_timestamp_ids, num_tokens in zip(
                timestamp_token_ids, tokens_per_frame
        ):
            # 中文注释：Add timestamp tokens
            all_token_ids.extend(frame_timestamp_ids)

            # 中文注释：Add vision tokens: vision_start + video_tokens + vision_end
            all_token_ids.append(vision_start_token_id)
            all_token_ids.extend([video_token_id] * num_tokens)
            all_token_ids.append(vision_end_token_id)

        if select_token_id:
            return PromptUpdateDetails.select_token_id(all_token_ids, video_token_id)

        # 中文注释：NOTE: we use `from_seq` instead of `select_token_id` because we want all
        # 中文注释：tokens in the placeholder to be initially marked as candidates. Then
        # 中文注释：in `get_input_embeddings``, we refine the mask to only replace
        # 中文注释：`video_token_id` / `image_token_id`` positions with video/image embeddings,
        # 中文注释：keeping text embeddings for timestamps and structural tokens.
        return PromptUpdateDetails.from_seq(all_token_ids)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # 中文注释：positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # 中文注释：otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        # 中文注释：the same shape as input_embeds
        "deepstack_input_embeds": 0,
    }
)
class Qwen3LLMModel(Qwen3Model):
    def forward(
            self,
            input_ids: torch.Tensor | None,
            positions: torch.Tensor,
            intermediate_tensors: IntermediateTensors | None = None,
            inputs_embeds: torch.Tensor | None = None,
            # 中文注释：args for deepstack
            deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in islice(
                enumerate(self.layers), self.start_layer, self.end_layer
        ):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

            if deepstack_input_embeds is not None and layer_idx in range(
                    0, len(deepstack_input_embeds)
            ):
                hidden_states = (
                        hidden_states
                        + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
                )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            intermediate_tensors = IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
            # PP 非最后 stage 的 aux hidden states 只用于本地 capture / speculative 侧路；
            # 主干 PP 通信仍保持 hidden_states/residual 两个核心张量。
            if aux_hidden_states:
                return intermediate_tensors, aux_hidden_states
            return intermediate_tensors
        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


class Qwen3LLMForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, cfie_config: CfieConfig, prefix: str = ""):
        super(Qwen3ForCausalLM, self).__init__()
        config = cfie_config.model_config.hf_config
        quant_config = cfie_config.quant_config

        self.config = config

        self.quant_config = quant_config
        self.model = Qwen3LLMModel(
            cfie_config=cfie_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix="lm_head",
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3VLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3VLForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsMRoPE,
    SupportsEagle,
    SupportsEagle3,
    SupportsMultiModalPruning,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        "qkv": ["qkv"],  # For vision tower's already-packed QKV
    }

    supports_encoder_tp_data = True

    # 中文注释：To ensure correct weight loading and mapping.
    hf_to_cfie_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"

        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, cfie_config: CfieConfig, prefix: str = "model"):
        super().__init__()
        config: Qwen3VLConfig = cfie_config.model_config.hf_config
        quant_config = cfie_config.quant_config
        multimodal_config = cfie_config.model_config.multimodal_config

        self.config = config
        self._tokenizer = cached_tokenizer_from_config(cfie_config.model_config)
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(cfie_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

            # 中文注释：register buffer for deepstack
            if self.use_deepstack:
                self.deepstack_input_embeds = [
                    torch.zeros(
                        cfie_config.scheduler_config.max_num_batched_tokens,
                        config.text_config.hidden_size,
                    )
                    for _ in range(self.deepstack_num_level)
                ]

        with self._mark_language_model(cfie_config):
            self.language_model = Qwen3LLMForCausalLM(
                cfie_config=cfie_config.with_hf_config(config.text_config),
                prefix=maybe_prefix(prefix, "language_model"),
            )

        if not get_pp_group().is_first_rank and hasattr(
                config.vision_config, "deepstack_visual_indexes"
        ):
            assert self.language_model.start_layer >= len(
                config.vision_config.deepstack_visual_indexes
            ), (
                "start_layer should be greater than or equal to "
                "len(deepstack_visual_indexes)"
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _get_deepstack_input_embeds(
            self,
            num_tokens: int,
    ) -> IntermediateTensors | None:
        if not getattr(self, "deepstack_input_embeds", None):
            return None  # If vision tower is skipped

        # 中文注释：get deepstack_input_embeds from buffer, and clear the buffer
        return IntermediateTensors(
            {
                f"deepstack_input_embeds_{idx}": self.deepstack_input_embeds[idx][
                    :num_tokens
                ]
                for idx in range(self.deepstack_num_level)
            }
        )

    def _set_deepstack_input_embeds(self, deepstack_input_embeds: torch.Tensor) -> None:
        if not getattr(self, "deepstack_input_embeds", None):
            return

        # 中文注释：set deepstack_input_embeds to buffer
        num_tokens = deepstack_input_embeds.size(1)
        if num_tokens > self.deepstack_input_embeds[0].size(0):
            self.deepstack_input_embeds = [
                torch.zeros(
                    num_tokens,
                    self.config.text_config.hidden_size,
                    device=self.deepstack_input_embeds[0].device,
                    dtype=self.deepstack_input_embeds[0].dtype,
                )
                for _ in range(self.deepstack_num_level)
            ]
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[idx][:num_tokens].copy_(
                deepstack_input_embeds[idx]
            )

    def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
        if not getattr(self, "deepstack_input_embeds", None):
            return

        # 中文注释：clear deepstack_input_embeds in buffer
        if num_tokens > 0:
            for idx in range(self.deepstack_num_level):
                self.deepstack_input_embeds[idx][:num_tokens].zero_()

    def _parse_and_validate_image_input(
            self, **kwargs: object
    ) -> Qwen2_5_VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return Qwen2_5_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _parse_and_validate_video_input(
            self, **kwargs: object
    ) -> Qwen2_5_VLVideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)
        timestamps = kwargs.pop("timestamps", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                timestamps=timestamps,
            )

        if video_embeds is not None:
            return Qwen2_5_VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
                timestamps=timestamps,
            )

    def _process_image_input(
            self, image_input: Qwen2_5_VLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.visual, pixel_values, grid_thw.tolist(), rope_type="rope_3d"
                )
            else:
                image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        # 中文注释：Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(
            self, video_input: Qwen2_5_VLVideoInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(
                self.visual.dtype
            )
            if self.use_data_parallel:
                grid_thw_list = grid_thw.tolist()
                return run_dp_sharded_mrope_vision_model(
                    self.visual, pixel_values_videos, grid_thw_list, rope_type="rope_3d"
                )
            else:
                video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

        # 中文注释：Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return video_embeds.split(sizes)

    def _postprocess_image_embeds_evs(
            self,
            image_embeds_split: tuple[torch.Tensor, ...],
            image_input: Qwen2_5_VLImageInputs,
    ) -> tuple[torch.Tensor, ...]:
        """
        中文说明：Append mrope positions for each for images.
        中文说明：This is necessary to recover correct mrope
        中文说明：positions after video pruning

        中文说明：Args:
            中文说明：image_embeds_split: Tuple of image embeddings for
                中文说明：each image item.
            中文说明：image_input: Image input data.

        中文说明：Returns:
            中文说明：Tuple of image embeddings for each image item.
            中文说明：Resulting embeddings will have extra 5 channels for
            中文说明：computed mrope positions, consistent with video embeddings.
        """
        if self.is_multimodal_pruning_enabled:
            merge_size = self.visual.spatial_merge_size
            grid_thw = image_input["image_grid_thw"]
            grid_thw_list = grid_thw.tolist()
            image_embeds_out = []
            for emb, size in zip(image_embeds_split, grid_thw_list):
                positions = compute_mrope_for_media(size, merge_size).to(emb.device)
                positions = torch.cat(
                    [
                        positions,
                        torch.zeros_like(
                            positions[:, 0:1]
                        ),  # Dummy extra fifth channel
                    ],
                    dim=1,
                )
                emb = torch.cat([emb, positions], dim=1)
                image_embeds_out.append(emb)
            image_embeds_split = tuple(image_embeds_out)
        return image_embeds_split

    def _postprocess_video_embeds_evs(
            self,
            video_embeds_split: tuple[torch.Tensor, ...],
            video_input: Qwen2_5_VLVideoInputs,
    ) -> tuple[torch.Tensor, ...]:
        """
        中文说明：Prunes video embeddings via Efficient Video Sampling (EVS)
        中文说明：and then appends mrope positions for each retained embeddings

        中文说明：Args:
            中文说明：video_embeds_split: Tuple of video embeddings for each video item.
            中文说明：video_input: Video input data.

        中文说明：Returns:
            中文说明：Tuple of video embeddings for each video item.
            中文说明：Resulting embeddings will have extra 5 channels for computed mrope
            中文说明：positions, and whether the index corresponds to a video embedding.
        """
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()
        merge_size = self.visual.spatial_merge_size

        # 中文注释：Apply EVS to each video.
        video_embeds_out = []
        for video_idx, (emb, size) in enumerate(zip(video_embeds_split, grid_thw_list)):
            # 中文注释：Compute positions.
            timestamps = video_input.timestamps[video_idx]
            num_frames = len(timestamps)

            t, h, w = size
            if self.is_multimodal_pruning_enabled:
                # 中文注释：For each video, compute retention mask using EVS.
                # 中文注释：retention_mask: [11424].
                retention_mask = compute_retention_mask(
                    emb,
                    size,
                    spatial_merge_size=self.visual.spatial_merge_size,
                    q=self.video_pruning_rate,
                )
                # 中文注释：Apply retention mask.
                emb = emb[retention_mask]

                # 中文注释：Calculate the actual number of retained tokens per frame.
                num_frames, rows, cols = (
                    t,
                    h // merge_size,
                    w // merge_size,
                )
                retention_mask_thw = retention_mask.reshape(num_frames, rows, cols)
                num_tokens_per_frame = (
                    retention_mask_thw.sum(dim=(1, 2)).long().tolist()
                )
            else:
                feature_size = emb.shape[0] // num_frames
                num_tokens_per_frame = [feature_size] * num_frames
                retention_mask = None

            emb = self._create_final_video_embeddings(
                video_embeddings=emb,
                num_tokens_per_frame=num_tokens_per_frame,
                timestamps=timestamps,
                video_grid_thw=size,
                retention_mask=retention_mask,
            )

            video_embeds_out.append(emb)

        return tuple(video_embeds_out)

    def _create_final_video_embeddings(
            self,
            video_embeddings: torch.Tensor,
            num_tokens_per_frame: list[int],
            timestamps: list[float],
            video_grid_thw: list[int],
            retention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """创建最终融合 embedding，将视频 embedding 与
        中文说明：text embeddings of indicator tokens.

        中文说明：These final embeddings contain:
        中文说明：- Actual video embeddings in positions corresponding to video content
        中文说明：- Text embeddings for indicator tokens (<img>, </img>, and
          中文说明：frame separation text) in their respective positions

        中文说明：These embeddings will replace the placeholder embeddings to create
        中文说明：input_embeds for the LLM.
        """
        device = video_embeddings.device

        # 中文注释：Generate video replacement token IDs using get_video_repl
        # 中文注释：This tokenizes each frame separator independently, then uses pre-tokenized
        # 中文注释：special tokens to ensure consistent tokenization regardless of
        # 中文注释：num_tokens_per_frame values.
        video_repl = Qwen3VLMultiModalProcessor.get_video_repl(
            tokens_per_frame=num_tokens_per_frame,
            tokenizer=self._tokenizer,
            timestamps=timestamps,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            video_token_id=self.config.video_token_id,
            select_token_id=self.is_multimodal_pruning_enabled,
        )

        repl_token_ids = torch.tensor(video_repl.full, device=device)
        embed_token_id = _cached_tensor(self.config.video_token_id, device=device)
        is_video_embed = torch.isin(repl_token_ids, embed_token_id)

        # 中文注释：Get text embeddings for indicator tokens (has only `visual_dim``).
        text_embeddings = self.get_language_model().embed_input_ids(repl_token_ids)

        if self.use_deepstack:
            (
                deepstack_input_embeds,
                multimodal_embeddings,
            ) = self._compute_deepstack_embeds(
                inputs_embeds=text_embeddings,
                multimodal_embeddings=[video_embeddings],
                is_multimodal=is_video_embed,
            )
        else:
            deepstack_input_embeds = None
            multimodal_embeddings = [video_embeddings]

        merged_embeddings = _merge_multimodal_embeddings(
            inputs_embeds=text_embeddings,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_video_embed,
        )

        to_concat = [merged_embeddings]
        if deepstack_input_embeds is not None:
            to_concat.append(
                deepstack_input_embeds.permute(1, 0, 2).reshape(
                    deepstack_input_embeds.shape[1], -1
                )
            )

        expanded_positions = None
        if self.is_multimodal_pruning_enabled:
            is_vision_start = repl_token_ids.eq(self.config.vision_start_token_id)
            expanded_positions = self._get_expanded_positions(
                device=merged_embeddings.device,
                seq_len=merged_embeddings.shape[0],
                video_grid_thw=video_grid_thw,
                num_tokens_per_frame=num_tokens_per_frame,
                timestamps=timestamps,
                is_video_embed=is_video_embed,
                is_vision_start=is_vision_start,
                retention_mask=retention_mask,
            )
            to_concat.append(expanded_positions)

        final_video_embeddings = torch.cat(to_concat, dim=-1)

        return final_video_embeddings

    def _get_expanded_positions(
            self,
            device,
            seq_len,
            video_grid_thw,
            num_tokens_per_frame,
            timestamps,
            is_video_embed,
            is_vision_start,
            retention_mask,
    ):
        embed_token_id = _cached_tensor(self.config.video_token_id, device=device)

        # 中文注释：Expand positions to match the full sequence length
        # 中文注释：(includes both video tokens and indicator tokens)
        # 中文注释：Shape: [full_length, 5] where positions are filled for video tokens
        # 中文注释：and zeros for indicator tokens.
        # 中文注释：Channel 3 flags VISION_START tokens so that
        # 中文注释：recompute_mrope_positions can reliably count timestamp tokens
        # 中文注释：(even when early frames have all video tokens pruned).
        # 中文注释：Channel 4 flags video-embedding tokens.
        expanded_positions = torch.zeros(
            seq_len,
            5,  # [t_index, h_index, w_index, is_vision_start, is_video]
            device=device,
            dtype=torch.long,
        )
        _, h, w = video_grid_thw
        merge_size = self.visual.spatial_merge_size
        num_frames = len(num_tokens_per_frame)
        unpruned_token_ids = Qwen3VLMultiModalProcessor.get_video_repl(
            tokens_per_frame=[(h // merge_size) * (w // merge_size)] * num_frames,
            tokenizer=self._tokenizer,
            timestamps=timestamps,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            video_token_id=self.config.video_token_id,
        ).full
        unpruned_token_ids_tensor = torch.tensor(unpruned_token_ids, device=device)
        mm_feature = MultiModalFeatureSpec(
            data=MultiModalKwargsItem(
                {
                    "video_grid_thw": MultiModalFieldElem(
                        data=torch.tensor(video_grid_thw),
                        field=None,  # HACK.
                    ),
                }
            ),
            modality="video",
            identifier="DUMMY",
            mm_position=PlaceholderRange(offset=0, length=len(unpruned_token_ids)),
        )
        original_mrope = (
            self.get_mrope_input_positions(
                input_tokens=unpruned_token_ids,
                mm_features=[mm_feature],
            )[0]
            .to(device)
            .permute(1, 0)
        )
        full_is_video_embed = unpruned_token_ids_tensor == embed_token_id
        expanded_positions[is_video_embed, :3] = original_mrope[full_is_video_embed][
            retention_mask
        ]
        expanded_positions[~is_video_embed, :3] = original_mrope[~full_is_video_embed]
        expanded_positions[..., 3] = is_vision_start
        expanded_positions[..., 4] = is_video_embed

        return expanded_positions

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}
        for input_key in kwargs:
            if (
                    input_key in ("pixel_values", "image_embeds")
                    and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                    input_key in ("pixel_values_videos", "video_embeds")
                    and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
                    **kwargs
                )
        return mm_input_by_modality

    @staticmethod
    def _iter_mm_grid_hw(
            input_tokens: list[int],
            mm_features: list[MultiModalFeatureSpec],
            video_token_id: int,
            vision_start_token_id: int,
            vision_end_token_id: int,
            spatial_merge_size: int,
    ) -> Iterator[tuple[int, int, int, int]]:
        """遍历多模态特征并产出位置信息。

        中文说明：Args:
            中文说明：input_tokens: List of token IDs in the input sequence.
            中文说明：mm_features: List of multimodal feature specifications containing
                中文说明：image/video data and position information.
            中文说明：video_token_id: Token ID used for video tokens.
            中文说明：vision_start_token_id: Token ID marking the start of a vision sequence.
            中文说明：vision_end_token_id: Token ID marking the end of a vision sequence.
            中文说明：spatial_merge_size: Size of the spatial merge operation used to
                中文说明：compute logical grid dimensions from the original feature grid.

        中文说明：Yields:
            中文说明：offset: Position of the first video/image token in the sequence.
            中文说明：llm_grid_h: Logical grid height (may not match actual token count with EVS).
            中文说明：llm_grid_w: Logical grid width (may not match actual token count with EVS).
            中文说明：actual_num_tokens: Actual number of video/image tokens in the placeholder.
        """
        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            offset = mm_feature.mm_position.offset
            if mm_feature.modality == "image":
                t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
                assert t == 1, f"Image must have 1 frame, got {t}"
                llm_grid_h = h // spatial_merge_size
                llm_grid_w = w // spatial_merge_size
                yield offset, llm_grid_h, llm_grid_w, llm_grid_h * llm_grid_w
            elif mm_feature.modality == "video":
                t, h, w = mm_feature.data["video_grid_thw"].data.tolist()
                llm_grid_h = h // spatial_merge_size
                llm_grid_w = w // spatial_merge_size

                for _ in range(t):
                    # 中文注释：When EVS is enabled, some frames may have 0 video tokens in the
                    # 中文注释：placeholder. We use `vision_start_token_id` to locate each frame
                    # 中文注释：since it is always present for every frame.
                    # 中文注释：We then look for the first `video_token_id` after
                    # 中文注释：`vision_start_token_id` and before `vision_end_token_id`.
                    offset = input_tokens.index(vision_start_token_id, offset)
                    vision_end_offset = input_tokens.index(vision_end_token_id, offset)

                    try:
                        actual_num_tokens = 0
                        video_offset = input_tokens.index(
                            video_token_id, offset, vision_end_offset
                        )
                        # 中文注释：NOTE: looking at the
                        # 中文注释：`Qwen3VLMultiModalProcessor.get_video_repl` code, we can
                        # 中文注释：see that we can use the below formula to get the token
                        # 中文注释：count, since everything in between `video_offset` and
                        # 中文注释：`vision_end_offset` is populated as `video_token_id`.
                        # 中文注释：This saves us from manually counting the number tokens
                        # 中文注释：that match `video_token_id` in between.
                        actual_num_tokens += vision_end_offset - video_offset
                    except ValueError:
                        # 中文注释：No `video_token_id` in this frame (EVS with 0 tokens for
                        # 中文注释：this frame) -> use `offset + 1`` to move past
                        # 中文注释：`vision_start_token_id`.
                        video_offset = offset + 1

                    yield video_offset, llm_grid_h, llm_grid_w, actual_num_tokens
                    # 中文注释：Move offset past this frame for next iteration.
                    offset = vision_end_offset + 1
            else:
                raise ValueError(f"Unsupported modality: {mm_feature.modality}")

    def _get_evs_mask_segments(
            self, mm_position: PlaceholderRange, expected_frames: int
    ) -> list[torch.Tensor] | None:
        """从 EVS 的 is_embed 掩码中提取连续片段。

        中文说明：The EVS (Efficient Video Sampling) mask marks which placeholder
        中文说明：positions should be filled with video embeddings. This method splits
        中文说明：the mask into contiguous segments, where each segment represents one
        中文说明：retained frame.

        中文说明：This is a pure function - it does not modify any state and always
        中文说明：returns the same output for the same input (idempotent).

        中文说明：Args:
            中文说明：mm_position: MultiModal position containing the is_embed mask
            中文说明：expected_frames: Expected number of frame segments

        中文说明：Returns:
            中文说明：List of tensors, each containing indices for one frame segment,
            中文说明：or None if EVS is not enabled or validation fails.
        """
        is_embed_mask = getattr(mm_position, "is_embed", None)
        if is_embed_mask is None:
            return None

        # 中文注释：Find all True positions in the mask
        mask_tensor = torch.as_tensor(is_embed_mask, dtype=torch.bool).view(-1)
        true_indices = torch.nonzero(mask_tensor, as_tuple=False).flatten()
        if true_indices.numel() == 0:
            return None

        # 中文注释：Split into contiguous segments (where diff > 1 indicates a gap)
        if true_indices.numel() == 1:
            segments = [true_indices]
        else:
            diffs = torch.diff(true_indices)
            split_points = torch.nonzero(diffs != 1, as_tuple=False).flatten()
            if split_points.numel() == 0:
                segments = [true_indices]
            else:
                segments = torch.tensor_split(
                    true_indices, split_points.add(1).tolist()
                )

        # 中文注释：Validate segment count matches expected frames
        if len(segments) < expected_frames:
            logger.debug(
                "EVS mask segments (%d) do not match expected frames (%d)",
                len(segments),
                expected_frames,
            )
            return None

        return segments[:expected_frames]

    def _extract_frame_offsets_from_mask(
            self, mm_position: PlaceholderRange, expected_frames: int
    ) -> list[int] | None:
        """返回每个 EVS 保留帧的相对偏移量。

        中文说明：The prompt processor stores a boolean mask inside ``mm_position`` that
        中文说明：marks which placeholder locations should be populated with video
        中文说明：embeddings. By splitting that mask into contiguous runs we can recover
        中文说明：the start of every retained frame without probing ``input_tokens``.

        中文说明：Args:
            中文说明：mm_position: MultiModal position containing the is_embed mask
            中文说明：expected_frames: Expected number of frames

        中文说明：Returns:
            中文说明：List of starting offsets (relative to mm_position) for each frame,
            中文说明：or None if EVS is not enabled.
        """
        segments = self._get_evs_mask_segments(mm_position, expected_frames)
        if segments is None:
            return None

        return [int(segment[0].item()) for segment in segments]

    def _get_actual_frame_token_counts(
            self, mm_position: PlaceholderRange, expected_frames: int
    ) -> list[int] | None:
        """返回每个 EVS 保留帧的实际 token 数量。

        中文说明：This function calculates the actual number of tokens per frame by
        中文说明：analyzing the is_embed mask, accounting for EVS pruning. Each frame
        中文说明：may have a different token count due to content-aware pruning.

        中文说明：Args:
            中文说明：mm_position: MultiModal position containing the is_embed mask
            中文说明：expected_frames: Expected number of frames

        中文说明：Returns:
            中文说明：List of token counts for each frame, or None if EVS is not enabled.
        """
        segments = self._get_evs_mask_segments(mm_position, expected_frames)
        if segments is None:
            return None

        return [len(seg) for seg in segments]

    def get_mrope_input_positions(
            self,
            input_tokens: list[int],
            mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        return self._get_mrope_input_positions(
            input_tokens=input_tokens,
            mm_features=mm_features,
            config=self.config,
        )

    @staticmethod
    def _get_mrope_input_positions(
            input_tokens: list[int],
            mm_features: list[MultiModalFeatureSpec],
            config: Qwen3VLConfig,
    ):
        llm_pos_ids_list = []
        st = 0
        for (
                offset,
                llm_grid_h,
                llm_grid_w,
                actual_num_tokens,
        ) in Qwen3VLForConditionalGeneration._iter_mm_grid_hw(
            input_tokens,
            mm_features,
            video_token_id=config.video_token_id,
            vision_start_token_id=config.vision_start_token_id,
            vision_end_token_id=config.vision_end_token_id,
            spatial_merge_size=config.vision_config.spatial_merge_size,
        ):
            # 中文注释：Skip frames with 0 tokens (EVS placeholder with tokens lumped elsewhere)
            if actual_num_tokens == 0:
                continue

            text_len = offset - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )

            # 中文注释：Check if this is a "lumped placeholder" (all tokens from multiple frames
            # 中文注释：assigned to the 0-th frame - see
            # 中文注释：`Qwen3VLMultiModalProcessor.get_video_repl`.
            expected_tokens_per_frame = llm_grid_h * llm_grid_w
            if actual_num_tokens > expected_tokens_per_frame:
                # 中文注释：Lumped placeholder: create grid positions for all "logical" frames
                # 中文注释：represented.
                num_logical_frames = actual_num_tokens // expected_tokens_per_frame
                remainder = actual_num_tokens % expected_tokens_per_frame

                # 中文注释：Create positions for complete frames.
                for _ in range(num_logical_frames):
                    grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(
                        3, -1
                    )
                    llm_pos_ids_list.append(grid_indices + text_len + st_idx)
                    st_idx = llm_pos_ids_list[-1].max() + 1
                    text_len = 0  # No text between frames within the lump

                # 中文注释：Handle remainder tokens if any (partial frame).
                # 中文注释：NOTE: this should never be the case. Should we have an assert?
                if remainder > 0:
                    # 中文注释：Create a partial grid - take first 'remainder' positions
                    full_grid = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
                    grid_indices = full_grid[:, :remainder]
                    llm_pos_ids_list.append(grid_indices + text_len + st_idx)
            else:
                # 中文注释：Normal case: frame has exactly the expected tokens (after actual EVS
                # 中文注释：pruning).
                grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
                llm_pos_ids_list.append(grid_indices + text_len + st_idx)

            st = offset + actual_num_tokens

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
        return torch.from_numpy(llm_positions), mrope_position_delta

    def recompute_mrope_positions(
            self,
            input_ids: list[int],
            multimodal_embeddings: MultiModalEmbeddings,
            mrope_positions: torch.LongTensor,
            num_computed_tokens: int,
    ) -> tuple[MultiModalEmbeddings, torch.Tensor, int]:
        """
        中文说明：Update part of input mrope positions (starting with
        中文说明：num_computed_tokens index). Original mrope_positions are computed
        中文说明：for unpruned sequence and becomes incorrect once pruning occurs,
        中文说明：so once we prune media tokens we should reflect this in the
        中文说明：mrope_positions before we feed it to LLM.

        中文说明：Args:
            中文说明：input_ids: (N,) All input tokens of the prompt containing
                中文说明：entire sequence.
            中文说明：multimodal_embeddings: Tuple of multimodal embeddings that
                中文说明：fits into the prefill chunk that is being processed.
            中文说明：mrope_positions: Existing mrope positions (3, N) for entire
                中文说明：sequence
            中文说明：num_computed_tokens: A number of computed tokens so far.

        中文说明：Returns:
            中文说明：Tuple of (multimodal_embeddings, mrope_positions,
                中文说明：mrope_position_delta).
        """
        return self._recompute_mrope_positions(
            input_ids=input_ids,
            multimodal_embeddings=multimodal_embeddings,
            mrope_positions=mrope_positions,
            num_computed_tokens=num_computed_tokens,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
        )

    @staticmethod
    def _recompute_mrope_positions(
            input_ids: list[int],
            multimodal_embeddings: MultiModalEmbeddings,
            mrope_positions: torch.LongTensor,
            num_computed_tokens: int,
            vision_start_token_id: int,
            image_token_id: int,
            video_token_id: int,
    ) -> tuple[MultiModalEmbeddings, torch.Tensor, int]:
        # 中文注释：Device
        device = (
            multimodal_embeddings[0].device
            if len(multimodal_embeddings)
            else mrope_positions.device
        )

        # 中文注释：Tensors
        input_ids_t = torch.as_tensor(input_ids, device=device, dtype=torch.long)

        mm_embeddings_out = []
        mm_embeddings_pos = []
        # 中文注释：Strip position information from embeddings (last 5 channels)
        # 中文注释：For Qwen3 VL, handle potentially empty frames (from unpacking)
        for mm in multimodal_embeddings:
            if mm.shape[0] > 0:  # Only process non-empty frames
                mm_embeddings_out.append(mm[:, :-5])
                mm_embeddings_pos.append(mm[:, -5:].permute(1, 0).long())
            else:
                # 中文注释：Empty frame - keep as is
                mm_embeddings_out.append(mm)
                # 中文注释：Create empty position tensor with correct shape
                mm_embeddings_pos.append(
                    torch.empty(5, 0, device=device, dtype=torch.long)
                )

        positions, mrope_positions_delta = recompute_mrope_positions(
            input_ids_t,
            mm_embeddings_pos,
            mrope_positions,
            num_computed_tokens,
            vision_start_token_id,
            image_token_id,
            video_token_id,
        )

        return tuple(mm_embeddings_out), positions, mrope_positions_delta

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return None

        # 中文注释：The result multimodal_embeddings is tuple of tensors, with each
        # 中文注释：tensor corresponding to a multimodal data item (image or video).
        multimodal_embeddings: list[torch.Tensor] = []

        # 中文注释：NOTE: It is important to iterate over the keys in this dictionary
        # 中文注释：to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                image_embeddings = self._postprocess_image_embeds_evs(
                    image_embeddings, multimodal_input
                )
                multimodal_embeddings.extend(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                if self.is_multimodal_pruning_enabled:
                    video_embeddings = self._postprocess_video_embeds_evs(
                        video_embeddings, multimodal_input
                    )
                multimodal_embeddings.extend(video_embeddings)

        embeddings_tuple = tuple(multimodal_embeddings)
        return embeddings_tuple

    def _compute_deepstack_embeds(
            self,
            inputs_embeds: torch.Tensor,
            multimodal_embeddings: MultiModalEmbeddings,
            is_multimodal: torch.Tensor,
    ) -> tuple[torch.Tensor, MultiModalEmbeddings]:
        visual_lens = [len(x) for x in multimodal_embeddings]
        multimodal_embeddings_cat = torch.cat(multimodal_embeddings, dim=0)

        (
            multimodal_embeddings_main,
            multimodal_embeddings_multiscale,
        ) = torch.split(
            multimodal_embeddings_cat,
            [self.visual_dim, self.multiscale_dim],
            dim=-1,
        )

        multimodal_embeddings = torch.split(
            multimodal_embeddings_main, visual_lens, dim=0
        )
        multimodal_embeddings_multiscale = torch.split(
            multimodal_embeddings_multiscale, visual_lens, dim=0
        )

        deepstack_input_embeds = inputs_embeds.new_zeros(
            inputs_embeds.size(0), self.deepstack_num_level * inputs_embeds.size(1)
        )

        deepstack_input_embeds = _merge_multimodal_embeddings(
            inputs_embeds=deepstack_input_embeds,
            multimodal_embeddings=multimodal_embeddings_multiscale,
            is_multimodal=is_multimodal,
        )
        deepstack_input_embeds = deepstack_input_embeds.view(
            inputs_embeds.shape[0], self.deepstack_num_level, self.visual_dim
        )
        deepstack_input_embeds = deepstack_input_embeds.permute(1, 0, 2)

        return deepstack_input_embeds, multimodal_embeddings

    def embed_input_ids(
            self,
            input_ids: torch.Tensor,
            multimodal_embeddings: MultiModalEmbeddings | None = None,
            *,
            is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        if self.use_deepstack:
            (
                deepstack_input_embeds,
                multimodal_embeddings,
            ) = self._compute_deepstack_embeds(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
        else:
            deepstack_input_embeds = None

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        if deepstack_input_embeds is not None:
            self._set_deepstack_input_embeds(deepstack_input_embeds)

        return inputs_embeds

    def forward(
            self,
            input_ids: torch.Tensor | None,
            positions: torch.Tensor,
            intermediate_tensors: IntermediateTensors | None = None,
            inputs_embeds: torch.Tensor | None = None,
            **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """执行 Qwen3VL 的前向计算。

        中文说明：Args:
            中文说明：input_ids: Flattened (concatenated) input_ids corresponding to a
                中文说明：batch.
            中文说明：positions: Flattened (concatenated) position ids corresponding to a
                中文说明：batch.
                中文说明：**NOTE**: If mrope is enabled (default setting for Qwen3VL
                中文说明：opensource models), the shape will be `(3, seq_len)`,
                中文说明：otherwise it will be `(seq_len,).
            中文说明：intermediate_tensors: Intermediate tensors from previous pipeline
                中文说明：stages.
            中文说明：inputs_embeds: Pre-computed input embeddings.
            中文说明：**kwargs: Additional keyword arguments including:
                中文说明：- pixel_values: Pixel values to be fed to a model.
                    中文说明：`None` if no images are passed.
                中文说明：- image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in
                    中文说明：LLM. `None` if no images are passed.
                中文说明：- pixel_values_videos: Pixel values of videos to be fed to a
                    中文说明：model. `None` if no videos are passed.
                中文说明：- video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in
                    中文说明：LLM. `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        if inputs_embeds is not None and get_pp_group().is_first_rank:
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0)
            )
        else:
            deepstack_input_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            # 中文注释：args for deepstack
            deepstack_input_embeds=deepstack_input_embeds,
        )

        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))

        return hidden_states

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_cfie_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        中文说明：Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector=["visual.merger", "visual.deepstack_merger_list"],
            tower_model="visual.",
        )

    def get_num_mm_encoder_tokens(
            self,
            num_image_tokens: int,
    ) -> int:
        hf_config = self.config
        vision_config = hf_config.vision_config
        merge_size = vision_config.spatial_merge_size

        return num_image_tokens * merge_size ** 2

    def get_num_mm_connector_tokens(
            self,
            num_vision_tokens: int,
    ) -> int:
        hf_config = self.config
        vision_config = hf_config.vision_config
        merge_size = vision_config.spatial_merge_size
        return num_vision_tokens // merge_size ** 2


@lru_cache
def _cached_tensor(x, device) -> torch.Tensor:
    return torch.tensor(x, device=device)
