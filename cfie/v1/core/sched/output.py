# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from cfie.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
    from cfie.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from cfie.lora.request import LoRARequest
    from cfie.multimodal.inputs import MultiModalFeatureSpec
    from cfie.pooling_params import PoolingParams
    from cfie.sampling_params import SamplingParams
    from cfie.v1.request import Request
else:
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    LoRARequest = object
    MultiModalFeatureSpec = object
    PoolingParams = object
    SamplingParams = object
    Request = object


@dataclass
class NewRequestData:
    # worker 首次见到某个请求时，需要这份全量初始化数据来建立本地缓存状态。
    req_id: str
    external_req_id: str | None
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: "torch.Tensor | None" = None

    # 仅 v2 model runner 使用；用于显式传递本轮 prefill 要消费的 token 视图。
    prefill_token_ids: list[int] | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "NewRequestData":
        # 从调度器内部 Request 快照提取出 worker 首次建态所需的最小完整载荷。
        return cls(
            req_id=request.request_id,
            external_req_id=getattr(request, "external_req_id", None),
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prefill_token_ids=prefill_token_ids,
        )

    def __repr__(self) -> str:
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"external_req_id={self.external_req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )

    # 隐去具体 prompt 内容的调试打印版本，避免日志直接泄露原始文本。
    def anon_repr(self) -> str:
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None
        )
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        prefill_token_ids_len = (
            len(self.prefill_token_ids) if self.prefill_token_ids is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"prefill_token_ids_len={prefill_token_ids_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )


@dataclass
class CachedRequestData:
    # worker 已缓存过这些请求后，后续 step 只需要发送增量更新。
    req_ids: list[str]
    # resumed_req_ids 中的请求是“恢复执行”的请求；其 block 视图要整体替换而不是追加。
    resumed_req_ids: set[str]
    # 仅 pipeline parallel 场景需要显式回传新增 token id；其余场景通常为空。
    new_token_ids: list[list[int]]
    # 上一轮未被调度的请求要补发 all_token_ids，便于 worker/connector 重建状态。
    all_token_ids: dict[str, list[int]]
    # 与 req_ids 按位置对齐，表示本轮新增或替换后的 block id 视图。
    new_block_ids: list[tuple[list[int], ...] | None]
    # 与 req_ids 按位置对齐，表示请求在本轮调度前的已计算 token 数。
    num_computed_tokens: list[int]
    # 与 req_ids 按位置对齐，表示请求当前已生成的输出 token 数。
    num_output_tokens: list[int]

    # 隐去具体 token id 内容的调试打印版本。
    def anon_repr(self) -> str:
        new_token_ids_lens = [len(toks) for toks in self.new_token_ids]
        all_token_ids_lens = {
            req_id: len(toks) for req_id, toks in self.all_token_ids.items()
        }
        return (
            f"CachedRequestData("
            f"req_ids={self.req_ids},"
            f"resumed_req_ids={self.resumed_req_ids},"
            f"new_token_ids_lens={new_token_ids_lens},"
            f"all_token_ids_lens={all_token_ids_lens},"
            f"new_block_ids={self.new_block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"num_output_tokens={self.num_output_tokens}"
            f")"
        )

    def __repr__(self) -> str:
        return self.anon_repr()

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @cached_property
    def _req_id_to_num_output_tokens(self) -> dict[str, int]:
        """缓存 req_id 到 num_output_tokens 的映射，便于 O(1) 查询。"""
        return dict(zip(self.req_ids, self.num_output_tokens))

    def is_context_phase(self, req_id: str) -> bool:
        num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
        return num_output_tokens is not None and num_output_tokens == 0

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


@dataclass
class SchedulerOutput:
    # 本轮首次发给 worker 的请求集合；worker 会据此建立本地请求状态。
    scheduled_new_reqs: list[NewRequestData]
    # 本轮继续执行的老请求集合；这里发送的是增量信息而不是完整请求。
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens，表示每个请求本轮实际下发了多少 token 工作量。
    num_scheduled_tokens: dict[str, int]
    # 本轮总工作量，等于 num_scheduled_tokens 的求和。
    total_num_scheduled_tokens: int
    # req_id -> spec_token_ids；只有本轮实际安排了 spec token 的请求才会出现在这里。
    scheduled_spec_decode_tokens: dict[str, list[int]]
    # req_id -> 需要在本轮处理的 encoder 输入索引。
    scheduled_encoder_inputs: dict[str, list[int]]
    # running 集合在每个 KV cache group 上的最长公共前缀 block 数。
    num_common_prefix_blocks: list[int]

    # 上一轮到这一轮之间刚结束的请求；worker 收到后可释放其本地缓存状态。
    finished_req_ids: set[str]
    # 需要从 encoder cache 释放的多模态特征哈希列表。
    free_encoder_mm_hashes: list[str]

    # 本轮被抢占的请求，仅 v2 model runner 使用。
    preempted_req_ids: set[str] | None = None

    # 本轮是否存在已进入 structured output 采样阶段的请求。
    has_structured_output_requests: bool = False

    # 本轮请求是否已经具备 grammar bitmask 计算所需的全部输出 token。
    pending_structured_output_tokens: bool = False

    # 用于修正 speculative decoding 的 acceptance rate 统计。
    num_invalid_spec_tokens: dict[str, int] | None = None

    # KV connector 在本轮需要执行的 load/store 元数据。
    kv_connector_metadata: KVConnectorMetadata | None = None

    # encoder cache connector 的元数据。
    ec_connector_metadata: ECConnectorMetadata | None = None

    # 本轮新分配出的 block id；worker 使用前会先清零对应显存，避免脏数据污染计算。
    new_block_ids_to_zero: list[int] | None = None

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )


@dataclass
class GrammarOutput:
    # 需要做 structured output 约束的请求 id 列表。
    structured_output_request_ids: list[str]
    # 与 structured_output_request_ids 同序对齐的 grammar bitmask。
    grammar_bitmask: "npt.NDArray[np.int32]"
