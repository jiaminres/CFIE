# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal, get_args

from cfie.config.utils import config
from cfie.logger import init_logger
from cfie.utils.hashing import safe_hash

logger = init_logger(__name__)

# 序列级 pooling 类型集合。
SequencePoolingType = Literal["CLS", "LAST", "MEAN"]
# 把 SequencePoolingType 展开成运行期 tuple 便于判定。
SEQ_POOLING_TYPES: tuple[SequencePoolingType, ...] = get_args(SequencePoolingType)

# token 级 pooling 类型集合。
TokenPoolingType = Literal["ALL", "STEP"]
# 把 TokenPoolingType 展开成运行期 tuple 便于判定。
TOK_POOLING_TYPES: tuple[TokenPoolingType, ...] = get_args(TokenPoolingType)


@config
class PoolerConfig:
    """Controls the behavior of output pooling in pooling models."""

    # 统一入口字段，允许用户只用 pooling_type 指定序列/Token pooling。
    pooling_type: SequencePoolingType | TokenPoolingType | None = None
    """
    The pooling method used for pooling.

    If set, `seq_pooling_type` or `tok_pooling_type` are automatically populated
    with this field. Alternatively, users can set `seq_pooling_type` and
    `tok_pooling_type` explicitly.

    This field is mainly for user convenience. Internal code should always use
    `seq_pooling_type` or `tok_pooling_type` instead of `pooling_type`.
    """

    # 序列级 pooling 的最终类型。
    seq_pooling_type: SequencePoolingType | None = None
    """
    The pooling method used for sequence pooling.
    """

    # token 级 pooling 的最终类型。
    tok_pooling_type: TokenPoolingType | None = None
    """
    The pooling method used for tokenwise pooling.
    """

    # pooling 结果是否继续过激活函数。
    use_activation: bool | None = None
    """
    Whether to apply activation function to the pooler outputs.
    `None` uses the pooler's default, which is `True` in most cases.
    """

    ## for embedding models
    # embedding 模型下，是否把向量降到指定维度。
    dimensions: int | None = None
    """
    Reduce the dimensions of embeddings if model
    support matryoshka representation. Defaults to None.
    """
    # 是否允许对超长输入分块做 embedding。
    enable_chunked_processing: bool = False
    """
    Whether to enable chunked processing for long inputs that exceed the model's
    maximum position embeddings. When enabled, long inputs will be split into
    chunks, processed separately, and then aggregated using weighted averaging.
    This allows embedding models to handle arbitrarily long text without CUDA
    errors. Defaults to False.
    """
    # embedding 场景允许的最大输入长度。
    max_embed_len: int | None = None
    """
    Maximum input length allowed for embedding generation. When set, allows
    inputs longer than max_embed_len to be accepted for embedding models.
    When an input exceeds max_embed_len, it will be handled according to 
    the original max_model_len validation logic. 
    Defaults to None (i.e. set to max_model_len).
    """

    ## for classification models
    # 分类模型输出 logit 时使用的 bias。
    logit_bias: float | None = None
    """
    If provided, apply classification logit biases. Defaults to None.
    """

    ## for reward models
    # reward model 下，若设置则只返回 step_tag 对应位置的分数。
    step_tag_id: int | None = None
    """
    If set, only the score corresponding to the `step_tag_id` in the
    generated sentence should be returned. Otherwise, the scores for all tokens
    are returned.
    """
    # reward model 下，允许只抽取部分 token id 对应的维度。
    returned_token_ids: list[int] | None = None
    """
    A list of indices for the vocabulary dimensions to be extracted,
    such as the token IDs of `good_token` and `bad_token` in the
    `math-shepherd-mistral-7b-prm` model.
    """

    def __post_init__(self) -> None:
        # 若用户提供了统一入口 pooling_type，则把它解析成 seq/tok 的具体字段。
        if pooling_type := self.pooling_type:
            # pooling_type 与 seq_pooling_type 不能同时设置。
            if self.seq_pooling_type is not None:
                raise ValueError(
                    "Cannot set both `pooling_type` and `seq_pooling_type`"
                )
            # pooling_type 与 tok_pooling_type 也不能同时设置。
            if self.tok_pooling_type is not None:
                raise ValueError(
                    "Cannot set both `pooling_type` and `tok_pooling_type`"
                )

            # 若该值属于序列级 pooling，则写入 seq_pooling_type。
            if pooling_type in SEQ_POOLING_TYPES:
                logger.debug(
                    "Resolved `pooling_type=%r` to `seq_pooling_type=%r`.",
                    pooling_type,
                    pooling_type,
                )
                self.seq_pooling_type = pooling_type
            # 若该值属于 token 级 pooling，则写入 tok_pooling_type。
            elif pooling_type in TOK_POOLING_TYPES:
                logger.debug(
                    "Resolved `pooling_type=%r` to `tok_pooling_type=%r`.",
                    pooling_type,
                    pooling_type,
                )
                self.tok_pooling_type = pooling_type
            else:
                # 落不到任一枚举集合时，说明调用方给了未支持的 pooling_type。
                raise NotImplementedError(pooling_type)

    def get_seq_pooling_type(self) -> SequencePoolingType:
        # 调用到这里时，seq_pooling_type 应已在 ModelConfig 中完成解析。
        assert self.seq_pooling_type is not None, "Should be resolved by ModelConfig"
        # 返回最终的序列级 pooling 类型。
        return self.seq_pooling_type

    def get_tok_pooling_type(self) -> TokenPoolingType:
        # 调用到这里时，tok_pooling_type 应已在 ModelConfig 中完成解析。
        assert self.tok_pooling_type is not None, "Should be resolved by ModelConfig"
        # 返回最终的 token 级 pooling 类型。
        return self.tok_pooling_type

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        # pooler 配置目前不参与图结构哈希。
        factors: list[Any] = []
        # 用空 factors 生成稳定哈希。
        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        # 返回哈希结果。
        return hash_str
