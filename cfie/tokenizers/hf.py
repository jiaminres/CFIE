# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import copy
from pathlib import Path
from typing import TypeAlias

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from cfie.transformers_utils.config import get_sentence_transformer_tokenizer_config

from .protocol import TokenizerLike

HfTokenizer: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast


def get_cached_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    # 这里不会重新构造一个 tokenizer，而是浅拷贝一个同类型对象，
    # 后面仅替换其 class，让若干频高属性直接返回缓存值。
    cached_tokenizer = copy.copy(tokenizer)

    # 先把后续高频访问的属性和词表结果一次性算好。
    tokenizer_all_special_ids = tokenizer.all_special_ids
    tokenizer_all_special_tokens = tokenizer.all_special_tokens
    tokenizer_vocab = tokenizer.get_vocab()
    tokenizer_len = len(tokenizer)

    # `max_token_id` / `max_chars_per_token` 常被下游预处理和长度估算逻辑访问，
    # 提前缓存后可避免 tokenizer 反复扫描词表。
    max_token_id = max(tokenizer_vocab.values())
    max_chars_per_token = max(len(tok) for tok in tokenizer_vocab)

    # 某些 tokenizer（例如 QwenTokenizer）会把额外 special tokens
    # 计入 `vocab_size` 的实现结果，但这些 token 不一定出现在 `get_vocab()`
    # 返回的词表里；因此如果对象实现了 `vocab_size`，这里应取两者中的较大值。
    if hasattr(tokenizer, "vocab_size"):
        with contextlib.suppress(NotImplementedError):
            max_token_id = max(max_token_id, tokenizer.vocab_size)

    # 动态派生一个“缓存版 tokenizer 类”，并覆盖几个高频属性访问接口。
    class CachedTokenizer(tokenizer.__class__):  # type: ignore
        @property
        def all_special_ids(self) -> list[int]:
            return tokenizer_all_special_ids

        @property
        def all_special_tokens(self) -> list[str]:
            return tokenizer_all_special_tokens

        @property
        def max_token_id(self) -> int:
            return max_token_id

        @property
        def max_chars_per_token(self) -> int:
            return max_chars_per_token

        def get_vocab(self) -> dict[str, int]:
            return tokenizer_vocab

        def __len__(self) -> int:
            return tokenizer_len

        def __reduce__(self):
            # 让 pickle / multiprocessing 仍能还原成缓存包装后的 tokenizer。
            return get_cached_tokenizer, (tokenizer,)

    CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"

    # 仅替换实例的运行时类型，不改原始 tokenizer 的底层分词行为。
    cached_tokenizer.__class__ = CachedTokenizer
    return cached_tokenizer


class CachedHfTokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(
            cls,
            path_or_repo_id: str | Path,
            *args,
            trust_remote_code: bool = False,
            revision: str | None = None,
            download_dir: str | None = None,
            **kwargs,
    ) -> HfTokenizer:
        try:
            # 主路径：直接复用 transformers 的 AutoTokenizer 选择具体 tokenizer 类。
            # `download_dir` 在 HF 侧对应 `cache_dir`；其余 kwargs 原样透传。
            tokenizer = AutoTokenizer.from_pretrained(
                path_or_repo_id,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                cache_dir=download_dir,
                **kwargs,
            )
        except ValueError as e:
            # 若仓库中的 tokenizer 依赖远端自定义 Python 代码，而当前没开
            # `trust_remote_code`，transformers 往往会在这里抛出 ValueError。
            # 我们把它改写成更明确的报错，提示用户开启对应开关。
            if not trust_remote_code and (
                    "does not exist or is not currently imported." in str(e)
                    or "requires you to execute the tokenizer file" in str(e)
            ):
                err_msg = (
                    "Failed to load the tokenizer. If the tokenizer "
                    "is a custom tokenizer not yet available in the "
                    "HuggingFace transformers library, consider "
                    "setting `trust_remote_code=True` in LLM or using "
                    "the `--trust-remote-code` flag in the CLI."
                )
                raise RuntimeError(err_msg) from e
            else:
                raise e

        # sentence-transformers 风格仓库可能额外提供一份 encoder tokenizer 配置；
        # 若其中要求 `do_lower_case=True`，特殊 token 也要同步转成小写，
        # 否则 tokenizer.special_tokens_map 与编码行为可能不一致。
        encoder_config = get_sentence_transformer_tokenizer_config(
            path_or_repo_id, revision
        )
        if isinstance(encoder_config, dict) and encoder_config.get(
                "do_lower_case", False
        ):
            # 这里只重写 special tokens 的大小写，不改普通词表内容。
            special_tokens_map = {
                k: v.lower() for k, v in tokenizer.special_tokens_map.items()
            }
            tokenizer.add_special_tokens(special_tokens_map)

        # 最后再套一层缓存代理，减少后续频繁读取 tokenizer 属性的开销。
        return get_cached_tokenizer(tokenizer)
