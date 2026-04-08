# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import cfie.envs as envs
from cfie.distributed import (
    get_ep_group,
    get_pcp_group,
    tensor_model_parallel_all_reduce,
)
from cfie.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
)
from cfie.logger import init_logger
from cfie.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from cfie.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from cfie.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from cfie.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from cfie.platforms import current_platform
from cfie.utils.math_utils import cdiv
from cfie.utils.torch_utils import (
    HAS_OPAQUE_TYPE,
    ModuleName,
    aux_stream,
    current_stream,
    direct_register_custom_op,
)
from cfie.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)


def get_layer_from_name(layer_name: str) -> torch.nn.Module:
    # 从 forward context 中按名字取回真实的 FusedMoE 层对象。
    forward_context: ForwardContext = get_forward_context()
    if layer_name == "from_forward_context":
        # 某些编译路径不会直接传真实层名，而是要求按调用顺序回放当前层。
        all_moe_layers = forward_context.all_moe_layers
        assert all_moe_layers is not None
        moe_layer_index = forward_context.moe_layer_index
        if moe_layer_index >= len(all_moe_layers):
            raise AssertionError(
                "We expected the number of MOE layers in `all_moe_layers` "
                "to be equal to the number of "
                "{cfie.moe_forward, cfie.moe_forward_shared} calls."
            )
        layer_name = all_moe_layers[moe_layer_index]
        # 每取一次都把索引向前推进，保证下一次能拿到后续 MoE 层。
        forward_context.moe_layer_index += 1
    return forward_context.no_compile_layers[layer_name]


# torch >= 2.11 时，layer_name 会被提升成 ModuleName 不透明对象；
# 更早版本里，它仍然只是普通字符串。
if TYPE_CHECKING:
    from typing import TypeAlias

    _layer_name_type: TypeAlias = str | ModuleName
else:
    _layer_name_type = ModuleName if HAS_OPAQUE_TYPE else str


def _resolve_layer_name(layer_name: str | ModuleName) -> str:
    # 兼容 torch 新旧版本对 layer_name 的不同封装形式。
    return layer_name.value if isinstance(layer_name, ModuleName) else layer_name


def _pack_token_ranges_by_expert_capacity(
    topk_ids: torch.Tensor,
    capacity: int,
) -> list[tuple[int, int]]:
    # 按“一个 chunk 内允许出现的唯一 experts 上限”把 token 切成多个连续范围。
    if topk_ids.numel() == 0:
        return []

    ranges: list[tuple[int, int]] = []
    current_start = 0
    current_experts: set[int] = set()

    for token_idx in range(topk_ids.shape[0]):
        # 当前 token 可能路由到多个 expert，这里先去重再统计。
        token_unique_ids = {
            int(expert_id)
            for expert_id in torch.unique(topk_ids[token_idx].detach()).tolist()
        }
        if len(token_unique_ids) > capacity:
            raise RuntimeError(
                f"token {token_idx} requested {len(token_unique_ids)} experts, "
                f"which exceeds tiered-cache capacity {capacity}"
            )

        # 尝试把当前 token 合并进现有 chunk，并估算合并后的唯一 experts 数。
        candidate_experts = current_experts | token_unique_ids
        if token_idx > current_start and len(candidate_experts) > capacity:
            # 一旦超容量，就在前一个 token 处截断，开启新的 chunk。
            ranges.append((current_start, token_idx))
            current_start = token_idx
            current_experts = set(token_unique_ids)
        else:
            # 仍未超容量时，把当前 token 继续并入当前 chunk。
            current_experts = candidate_experts

    # 收尾补上最后一个 chunk。
    ranges.append((current_start, topk_ids.shape[0]))
    return ranges


def _moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> torch.Tensor:
    # custom op 入口：先按 layer_name 找回层对象，再转交给 runner 真正执行。
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    # TODO(bnell): 等 MK 迁移完成后，这里的兼容初始化可删除。
    layer.ensure_moe_quant_config_init()
    return layer.runner.forward_impl(
        layer, hidden_states, router_logits, shared_experts_input
    )


def _moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> torch.Tensor:
    # fake impl 只用于 shape / tracing 推导，不参与真实计算。
    return torch.empty_like(hidden_states)


def _moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> tuple[torch.Tensor, torch.Tensor]:
    # shared-experts 版本会返回两个输出：shared 分支和 routed experts 分支。
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    # TODO(bnell): 等 MK 迁移完成后，这里的兼容初始化可删除。
    layer.ensure_moe_quant_config_init()
    return layer.runner.forward_impl(
        layer, hidden_states, router_logits, shared_experts_input
    )


def _moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> tuple[torch.Tensor, torch.Tensor]:
    # fake impl 同样要保持真实算子的返回结构一致。
    # 输出形状约定如下：
    # - fused_out 与 hidden_states 同形；routed experts 若做过变换，则使用变换后的维度。
    # - shared_out 若提供了 shared_experts_input，则与它同形；否则与 hidden_states 同形。
    # - latent MoE 下，shared experts 仍使用原始 hidden_size，而非 latent size。
    fused_out = torch.empty_like(hidden_states)
    if shared_experts_input is not None:
        shared_out = torch.empty_like(shared_experts_input)
    else:
        shared_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


direct_register_custom_op(
    op_name="moe_forward",
    op_func=_moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=_moe_forward_shared,
    mutates_args=["hidden_states"],
    fake_impl=_moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


class DefaultMoERunner(MoERunner):
    # 默认的 MoE 执行器实现，负责串起一层 Mixture-of-Experts 的完整前向流程。
    # 当前统一覆盖 expert routing、token dispatch、shared experts、DP chunking、
    # TP / EP 并行、量化执行，以及 monolithic / decomposed 两类专家执行路径。
    # 它的职责是把“路由 token -> 执行 experts -> 合并输出”这一整条链路跑通。
    # 后续可再按 shared experts、gate 等配置差异继续拆成更细的专用 runner。

    def __init__(
        self,
        layer: torch.nn.Module,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_input_transform: torch.nn.Module | None,
        gate: torch.nn.Module | None,
        shared_experts: torch.nn.Module | None,
        quant_method: FusedMoEMethodBase,
        reduce_results: bool,
        enable_dbo: bool,
    ):
        super().__init__()
        # 保存执行期所需的配置对象和子模块引用。
        self.moe_config = moe_config
        self.router = router
        self.routed_input_transform = routed_input_transform
        self.gate = gate
        self.shared_experts = shared_experts
        self.quant_method = quant_method
        self.reduce_results = reduce_results
        self.enable_dbo = enable_dbo

        # -----------------
        # shared experts 独立 stream 相关初始化。
        # -----------------
        # 出于调试目的，允许通过环境变量禁用 shared experts 的独立 stream。
        # TODO: 等 TP / DP 与其他执行模式验证更充分后，可移除这条调试开关。
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.debug_once("Disabling MoE shared_experts cuda stream", scope="local")
            self.shared_experts_stream = None
        else:
            # TODO(rob): 为非 cuda-alike 平台补上 shared expert overlap 支持。
            # 非 cuda-alike 平台上，aux_stream() 会直接返回 None。
            self.shared_experts_stream = aux_stream()
            if self.shared_experts_stream is not None:
                logger.debug_once(
                    "Enabled separate cuda stream for MoE shared_experts", scope="local"
                )

        # 记录层名，供 custom op 反查真实层对象。
        self.layer_name = layer.layer_name

        # 在 TPU / CPU 上直接绑定 Python 函数；在 CUDA 路径上优先走注册过的 custom op。
        if current_platform.is_tpu() or current_platform.is_cpu():
            # TODO: TPU 后端的 OOM 问题解决后，再切回 moe_forward custom op。
            # CPU 路径不需要额外包一层 forward_impl。
            if self.shared_experts is None:
                self.moe_forward = _moe_forward
            else:
                self.moe_forward = _moe_forward_shared
        else:
            if self.shared_experts is None:
                self.moe_forward = torch.ops.cfie.moe_forward
            else:
                self.moe_forward = torch.ops.cfie.moe_forward_shared

        # DP chunking 场景下会复用这两块 staging buffer，避免每个 chunk 重新分配显存。
        self.batched_hidden_states: torch.Tensor | None = None
        self.batched_router_logits: torch.Tensor | None = None

    def _apply_with_tiered_cache(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 这是 CFIE tiered cache 的执行桥接层：
        # 若当前层没挂 controller，就直接走原始 quant_method.apply；
        # 否则先确保本次请求涉及的 experts 已经在 GPU resident slots 中。
        controller = getattr(layer, "_cfie_tiered_cache_controller", None)
        if controller is None:
            return self.quant_method.apply(
                layer=layer,
                x=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
            )

        # -----------------
        # 单个 chunk 的执行逻辑。
        # -----------------
        def apply_chunk(
            chunk_x: torch.Tensor,
            chunk_topk_weights: torch.Tensor,
            chunk_topk_ids: torch.Tensor,
            chunk_shared_input: torch.Tensor | None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            # 统计当前 chunk 一共触达多少个唯一 experts。
            unique_requested = int(torch.unique(chunk_topk_ids.detach()).numel())
            num_chunk_tokens = int(chunk_topk_ids.shape[0])
            if unique_requested <= layer.local_num_experts:
                # resident slots 足够时，先 prepare 把缺失 experts 换入，再正常执行。
                controller.prepare(chunk_topk_ids)
                return self.quant_method.apply(
                    layer=layer,
                    x=chunk_x,
                    topk_weights=chunk_topk_weights,
                    topk_ids=chunk_topk_ids,
                    shared_experts_input=chunk_shared_input,
                )

            if controller.can_run_prefill_burst(unique_requested, num_chunk_tokens):
                # 若主 resident slots 装不下，但 burst pool 能兜住，就走临时 burst 执行区。
                return controller.run_prefill_burst(
                    x=chunk_x,
                    topk_weights=chunk_topk_weights,
                    topk_ids=chunk_topk_ids,
                    shared_experts_input=chunk_shared_input,
                )

            # 既装不进 resident，也不能用 burst，就只能报容量错误。
            raise RuntimeError(
                f"tiered-cache chunk requires {unique_requested} experts but no "
                "execution path can handle that capacity"
            )

        # -----------------
        # 先判断“整块输入”能否直接跑。
        # -----------------
        full_unique_requested = int(torch.unique(topk_ids.detach()).numel())
        full_num_tokens = int(topk_ids.shape[0])
        full_can_use_burst = controller.can_run_prefill_burst(
            full_unique_requested,
            full_num_tokens,
        )
        if (
            full_unique_requested <= layer.local_num_experts
            or full_can_use_burst
        ):
            # 整块可直接执行时，不再额外切 token chunk。
            return apply_chunk(x, topk_weights, topk_ids, shared_experts_input)

        # -----------------
        # 整块超容量时，按 expert 容量重新切 token 范围。
        # -----------------
        burst_capacity = getattr(controller, "prefill_burst_capacity", 0)
        burst_min_tokens = getattr(controller, "prefill_burst_min_tokens", 0)
        capacity = (
            max(layer.local_num_experts, burst_capacity)
            if burst_capacity > 0 and full_num_tokens >= burst_min_tokens
            else layer.local_num_experts
        )
        # 每个切片都要保证其唯一 experts 数不超过可执行容量。
        token_ranges = _pack_token_ranges_by_expert_capacity(topk_ids, capacity)

        outputs: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        for start, end in token_ranges:
            # shared experts 输入也必须同步切成相同 token 范围。
            chunk_shared_input = (
                shared_experts_input[start:end]
                if shared_experts_input is not None
                else None
            )
            outputs.append(
                apply_chunk(
                    x[start:end],
                    topk_weights[start:end],
                    topk_ids[start:end],
                    chunk_shared_input,
                )
            )

        # 把所有子 chunk 的输出重新沿 token 维拼回完整结果。
        first_output = outputs[0]
        if isinstance(first_output, tuple):
            return (
                torch.cat([output[0] for output in outputs], dim=0),
                torch.cat([output[1] for output in outputs], dim=0),
            )
        return torch.cat(outputs, dim=0)

    @property
    def use_dp_chunking(self) -> bool:
        # 只有部分 all2all backend 支持 / 受益于 DP chunking，并且还要受环境变量开关控制。
        return (
            self.moe_config.moe_parallel_config.use_deepep_ll_kernels
            or self.moe_config.moe_parallel_config.use_mori_kernels
            or self.moe_config.moe_parallel_config.use_fi_all2allv_kernels
            or self.moe_config.moe_parallel_config.use_nixl_ep_kernels
        ) and envs.VLLM_ENABLE_MOE_DP_CHUNK

    def _maybe_setup_shared_experts_stream(
        self,
        hidden_states: torch.Tensor,
        shared_input: torch.Tensor | None,
        has_separate_shared_experts: bool,
        use_chunked_impl: bool,
    ) -> tuple[bool, torch.Tensor | None]:
        # 决定是否让 shared experts 在独立 CUDA stream 上和 routed experts 并行执行。
        use_shared_experts_stream = (
            current_platform.is_cuda()
            and has_separate_shared_experts
            and not use_chunked_impl
            and self.shared_experts_stream is not None
            and (
                hidden_states.shape[0]
                <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            )
        )

        shared_experts_input: torch.Tensor | None = None
        if use_shared_experts_stream:
            assert self.shared_experts_stream is not None
            assert self.moe_config.disable_inplace

            shared_experts_input = (
                shared_input if shared_input is not None else hidden_states
            )

            # 标记 shared_experts_input 会在另一条 stream 上被消费，避免张量过早释放。
            # 这里不需要对 shared_output 再额外做 record_stream，
            # 因为后面在使用 shared_output 前会先同步两条 stream。
            shared_experts_input.record_stream(self.shared_experts_stream)

            # 在这里记录独立 shared experts stream 的同步起点，
            # 让它能够与下面的 router / gate 路径并行执行。
            assert self.shared_experts_stream is not None
            self.shared_experts_stream.wait_stream(current_stream())

        return use_shared_experts_stream, shared_experts_input

    def ensure_dp_chunking_init(self):
        # 按需懒初始化 DP chunking 的 staging tensor，重复 forward 时直接复用。
        if not self.use_dp_chunking or self.batched_hidden_states is not None:
            return

        states_shape: tuple[int, ...]
        logits_shape: tuple[int, ...]

        moe = self.moe_config

        if self.enable_dbo:
            # DBO 打开时，多预留一个 ubatch 维度。
            states_shape = (2, moe.max_num_tokens, self.moe_config.hidden_dim)
            logits_shape = (2, moe.max_num_tokens, self.moe_config.num_logical_experts)
        else:
            states_shape = (moe.max_num_tokens, self.moe_config.hidden_dim)
            logits_shape = (moe.max_num_tokens, self.moe_config.num_logical_experts)

        # 在当前设备上一次性分配 chunk staging buffer。
        device = torch.accelerator.current_device_index()
        self.batched_hidden_states = torch.zeros(
            states_shape,
            dtype=moe.in_dtype,
            device=device,
        )

        self.batched_router_logits = torch.zeros(
            logits_shape,
            dtype=moe.router_logits_dtype,
            device=device,
        )

    def must_reduce_shared_expert_outputs(self) -> bool:
        # shared experts 一般由 RowParallelLinear 计算。
        # 纯 TP 场景下可延后到 MoE 末尾再规约；
        # 但 EP + all2all 场景下，各 DP rank 会持有完整 hidden_states，
        # 因此需要尽早规约 shared experts 输出。
        assert self.quant_method is not None
        # 某些 kernel 已经在内部完成规约，这时 shared experts 输出无需再额外 reduce。
        return (
            self.quant_method.moe_kernel is not None
            and self.quant_method.moe_kernel.output_is_reduced()
        )

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        # 某些 combine kernel 默认已完成跨 GPU rank 的规约。
        # 若当前 kernel 没有帮我们规约，就在这里补一次 TP all-reduce。
        if self.must_reduce_shared_expert_outputs():
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def apply_routed_input_transform(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 仅对 routed experts 输入做额外变换，例如 latent projection。
        # FusedMoE.forward_native 会保留原始 hidden_states 给 shared experts，
        # 而 routed experts 则使用变换后的 [S, moe_latent_size] 输入。
        # TODO: 为了进一步降低 latent MoE 的带宽开销，fc2_latent_proj 未来可考虑
        # 下沉到 SharedFusedMoE 内部，在更小的 latent 维度上做 all-reduce。
        # routed experts 和 shared experts 的输入维可能不同，因此 routed 分支可单独变换。
        if self.routed_input_transform is not None:
            result = self.routed_input_transform(hidden_states)
            # ReplicatedLinear 会返回 (output, extra_bias)；
            # 这里只需要真正的输出张量。
            if isinstance(result, tuple):
                return result[0]
            return result
        return hidden_states

    def _reduce_output(
        self,
        states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        trunc_sizes: list[int],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 这个函数统一处理两件事：
        # 1. 必要时做 TP all-reduce；
        # 2. 把对齐 / padding 后的 hidden_dim 裁回原始尺寸。
        def trunc(x: torch.Tensor, trunc_size: int) -> torch.Tensor:
            return x[..., :trunc_size]

        def reduce_and_trunc(x: torch.Tensor, trunc_size: int) -> torch.Tensor:
            return trunc(self.maybe_all_reduce_tensor_model_parallel(x), trunc_size)

        if (
            not self.moe_config.is_sequence_parallel
            and not self.use_dp_chunking
            and self.reduce_results
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
        ):
            # 满足条件时，说明当前输出仍需要额外跨 rank 规约。
            func = reduce_and_trunc
        else:
            func = trunc

        if isinstance(states, tuple):
            return tuple(
                [func(s, trunc_size) for s, trunc_size in zip(states, trunc_sizes)]
            )
        else:
            assert len(trunc_sizes) == 1
            return func(states, trunc_sizes[0])

    def _encode_layer_name(self) -> str | ModuleName:
        # 编码 layer_name，供 custom op 在 forward context 中反查真实层对象。
        if HAS_OPAQUE_TYPE:
            return ModuleName(self.layer_name)
        # 单测环境里 forward context 可能不存在，或 all_moe_layers 为空。
        if (
            is_forward_context_available()
            and get_forward_context().all_moe_layers is not None
        ):
            return "from_forward_context"
        return self.layer_name

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # -----------------
        # 入口：准备 routed / shared experts 各自需要的输入。
        # -----------------
        # latent MoE 下先保留原始 hidden_states：
        # shared experts 仍用原始维度，routed experts 则走变换后的维度。
        if self.shared_experts is not None:
            original_hidden_states = hidden_states
            original_hidden_dim = hidden_states.shape[-1]
        else:
            original_hidden_states = None

        # 先对 routed experts 输入做可选变换，例如 latent projection。
        hidden_states = self.apply_routed_input_transform(hidden_states)

        # routed 分支若被 pad 到 kernel 需要的 hidden_dim，最后还要再裁回去。
        # 这里记录的是变换后的维度，后面裁剪 routed 输出时会用到。
        transformed_hidden_dim = hidden_states.shape[-1]
        if self.moe_config.hidden_dim != transformed_hidden_dim:
            hidden_states = F.pad(
                hidden_states,
                (0, self.moe_config.hidden_dim - transformed_hidden_dim),
                mode="constant",
                value=0.0,
            )

        # 真正执行 routed experts 的 forward；shared_experts_input 作为第三参数透传。
        fused_output = self.moe_forward(
            hidden_states,
            router_logits,
            original_hidden_states,
            self._encode_layer_name(),
        )

        # 根据是否存在 shared experts，准备最后的裁剪尺寸列表。
        if self.shared_experts is not None:
            orig_hidden_dims = [original_hidden_dim, transformed_hidden_dim]
        else:
            orig_hidden_dims = [transformed_hidden_dim]

        # 最终统一在这里做 reduce + trunc。
        return self._reduce_output(fused_output, orig_hidden_dims)

    def forward_impl_chunked(
        self,
        layer: torch.nn.Module,
        full_hidden_states: torch.Tensor,
        full_router_logits: torch.Tensor,
        full_shared_input: torch.Tensor | None,
        has_separate_shared_experts: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # -----------------
        # DP chunking 路径：把超大 token 批次切成多个 chunk，重复执行再拼回。
        # -----------------
        assert self.batched_hidden_states is not None
        assert self.batched_router_logits is not None
        assert self.batched_hidden_states.dtype == full_hidden_states.dtype, (
            f"{self.batched_hidden_states.dtype} == {full_hidden_states.dtype}"
        )
        assert self.batched_router_logits.dtype == full_router_logits.dtype, (
            f"{self.batched_router_logits.dtype} == {full_router_logits.dtype}"
        )
        # 确认 staging buffer 的最后一维和完整输入一致。
        assert self.batched_hidden_states.size(-1) == full_hidden_states.size(-1)
        assert self.batched_router_logits.size(-1) == full_router_logits.size(-1)

        # TODO(bnell): 修复 DP chunking 下的 shared_expert_inputs 支持。
        # assert shared_input is None, (
        #    "Routed input transform is not currently supported with DP chunking."
        # )

        full_fused_final_hidden_states = torch.empty_like(full_hidden_states)
        if self.shared_experts is not None:
            full_shared_final_hidden_states = torch.empty_like(full_hidden_states)

        # -----------------
        # 单个 chunk 的处理函数。
        # -----------------
        def process_chunk(chunk_start, chunk_end, skip_result_store=False):
            chunk_size = chunk_end - chunk_start
            # 先从完整输入中切出当前 chunk。
            hidden_states = full_hidden_states[chunk_start:chunk_end, :]
            router_logits = full_router_logits[chunk_start:chunk_end, :]
            shared_input = (
                full_shared_input[chunk_start:chunk_end, :]
                if full_shared_input is not None
                else None
            )

            assert self.batched_hidden_states is not None
            assert self.batched_router_logits is not None
            # 只有 DBO 打开时，staging tensor 才会多出一层 ubatch 维度。
            if self.batched_hidden_states.dim() == 3:
                assert self.batched_router_logits.dim() == 3
                # DBO 模式下按当前 ubatch 选择对应的 staging buffer。
                batch_buffer_idx = dbo_current_ubatch_id()
                batched_hidden_states = self.batched_hidden_states[batch_buffer_idx, :]
                batched_router_logits = self.batched_router_logits[batch_buffer_idx, :]
            else:
                batched_hidden_states = self.batched_hidden_states
                batched_router_logits = self.batched_router_logits

            # 只截取当前 chunk 实际需要的前半段 staging 空间。
            assert (
                batched_hidden_states.size(0)  # type: ignore
                >= chunk_size
            )
            assert (
                batched_router_logits.size(0)  # type: ignore
                >= chunk_size
            )
            staged_hidden_states = batched_hidden_states[:chunk_size, :]  # type: ignore
            staged_router_logits = batched_router_logits[:chunk_size, :]  # type: ignore
            # 把当前 chunk 拷进 staging buffer，后续 kernel 都直接读 staging tensor。
            staged_hidden_states.copy_(hidden_states, non_blocking=True)
            staged_router_logits.copy_(router_logits, non_blocking=True)

            shared_input = (
                shared_input if shared_input is not None else staged_hidden_states
            )

            # -----------------
            # chunk 内部的 MoE 主计算。
            # -----------------
            # 核心专家计算阶段。
            if self.quant_method.is_monolithic:
                assert has_separate_shared_experts or self.shared_experts is None
                final_hidden_states = self.quant_method.apply_monolithic(
                    layer=layer,
                    x=staged_hidden_states,
                    router_logits=staged_router_logits,
                )
            else:
                # 先做 router top-k 选择，再进入 tiered cache / quant method 执行。
                topk_weights, topk_ids = self.router.select_experts(
                    hidden_states=staged_hidden_states,
                    router_logits=staged_router_logits,
                )

                final_hidden_states = self._apply_with_tiered_cache(
                    layer=layer,
                    x=staged_hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    shared_experts_input=shared_input,
                )

            if has_separate_shared_experts:
                assert not isinstance(final_hidden_states, tuple)
                assert self.shared_experts is not None

                # shared experts 单独跑完后，和 routed experts 输出打包成二元组。
                shared_output = self.shared_experts(shared_input)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )

            if not skip_result_store:
                # 把当前 chunk 的结果写回完整输出张量。
                if self.shared_experts is None:
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states, non_blocking=True
                    )
                else:
                    full_shared_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[0], non_blocking=True
                    )
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[1], non_blocking=True
                    )

        # -----------------
        # 计算 chunk 循环边界。
        # -----------------
        ctx = get_forward_context()
        # flashinfer_cutlass kernel 可同时覆盖可选的 DP 与 TP/EP 组合。
        max_tokens_across_dispatchers = ctx.dp_metadata.max_tokens_across_dp_cpu
        moe_dp_chunk_size_per_rank = self.moe_config.max_num_tokens

        # 若输入本身走了 sequence parallel，需要先除以 sp_size，
        # 才能得到单个 dispatcher 实际可能看到的最大 token 数。
        if self.moe_config.is_sequence_parallel:
            max_tokens_across_dispatchers = cdiv(
                max_tokens_across_dispatchers, self.moe_config.sp_size
            )

        num_tokens = full_hidden_states.size(0)
        # 逐个 chunk 进入前面的 process_chunk。
        for chunk_idx, chunk_start_ in enumerate(
            range(0, max_tokens_across_dispatchers, moe_dp_chunk_size_per_rank)
        ):
            chunk_start = chunk_start_
            chunk_end = min(
                chunk_start + moe_dp_chunk_size_per_rank, max_tokens_across_dispatchers
            )
            # 再把 chunk 边界裁到当前真实 token 范围内。
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)
            with ctx.dp_metadata.chunked_sizes(
                self.moe_config.sp_size, moe_dp_chunk_size_per_rank, chunk_idx
            ):
                process_chunk(
                    chunk_start, chunk_end, skip_result_store=chunk_start_ >= num_tokens
                )

        # 所有 chunk 处理完后，按是否存在 shared experts 决定返回结构。
        if self.shared_experts is None:
            return full_fused_final_hidden_states
        else:
            return (full_shared_final_hidden_states, full_fused_final_hidden_states)

    def forward_impl(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # -----------------
        # 这是 runner 真正的主执行函数，所有 moe_forward custom op 最终都落到这里。
        # -----------------
        assert self.quant_method is not None

        # 按需初始化 DP chunking 复用缓冲区。
        self.ensure_dp_chunking_init()

        # shared experts 若不由 MK 内部接管，就需要 runner 自己单独执行这条分支。
        has_separate_shared_experts = (
            not self.quant_method.mk_owns_shared_expert
            and self.shared_experts is not None
        )

        # 某些 all2all backend 会强制启用 chunked 实现。
        use_chunked_impl = self.use_dp_chunking

        # 决定 shared experts 是否启用独立 stream 并准备其输入。
        use_shared_experts_stream, shared_experts_input = (
            self._maybe_setup_shared_experts_stream(
                hidden_states,
                shared_input,
                has_separate_shared_experts,
                use_chunked_impl,
            )
        )

        # -----------------
        # gate / router 预处理。
        # -----------------
        # 若当前层显式提供了 gate / router，就在这里先产出 router_logits。
        # 这段逻辑主要服务于 overlapped 模式，方便 shared experts 与 FusedMoE
        # 借助独立 CUDA stream 并行执行。
        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)

        if use_chunked_impl:
            # chunked 模式下，后续逻辑全部转交给专门的 chunked 实现。
            return self.forward_impl_chunked(
                layer,
                hidden_states,
                router_logits,
                shared_input,
                has_separate_shared_experts,
            )

        # -----------------
        # 非 chunked 主路径。
        # -----------------
        # TODO(rob): 等所有 quant method 都迁移到 MK 后，
        # 这里的 naive dispatch/combine 分支可以删除。
        do_naive_dispatch_combine = (
            self.moe_config.dp_size > 1 and not self.quant_method.supports_internal_mk
        )

        ctx = get_forward_context()
        sp_ctx = (
            ctx.dp_metadata.sp_local_sizes(self.moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

        with sp_ctx:
            # -----------------
            # shared experts 预执行路径。
            # -----------------
            # 若 shared experts 不走独立 stream，就先于主专家计算执行，
            # 避免后续矩阵计算修改 hidden_states。
            if has_separate_shared_experts and not use_shared_experts_stream:
                assert self.shared_experts is not None
                shared_input = (
                    shared_input if shared_input is not None else hidden_states
                )
                shared_output = self.shared_experts(shared_input)

            # -----------------
            # dispatch 阶段。
            # -----------------
            # naive DP/EP 路径下，需要先把 hidden states 与 router logits
            # 分发到对应的 expert rank。
            # TODO: 等所有 kernel 都迁移到 MoEKernel 框架后，这条分支可删除。
            if do_naive_dispatch_combine:
                # DP+EP 的 naive 路径下，需要先把 token / logits 分发到 expert 所在 rank。
                hidden_states, router_logits = get_ep_group().dispatch_router_logits(
                    hidden_states,
                    router_logits,
                    self.moe_config.is_sequence_parallel,
                )

            # PCP 打开时，还要额外在 prefill context 维上做一次 gather。
            # PCP 与 DP 类似，同样需要 dispatch / combine；
            # 当前为了简化实现，先单独给 PCP 接了一套 AgRsAll2All 路径，
            # 后续可再考虑把 All2AllManager 抽象扩展得更统一。
            if self.moe_config.pcp_size > 1:
                hidden_states = get_pcp_group().all_gather(
                    hidden_states,
                    dim=0,
                )
                router_logits = get_pcp_group().all_gather(
                    router_logits,
                    dim=0,
                )

            # -----------------
            # expert 主计算阶段。
            # -----------------
            # 核心专家计算阶段。
            if self.quant_method.is_monolithic:
                final_hidden_states = self.quant_method.apply_monolithic(
                    layer=layer,
                    x=hidden_states,
                    router_logits=router_logits,
                )
            else:
                # 非 monolithic 路径统一先由 router 产出 topk，再进入执行逻辑。
                topk_weights, topk_ids = self.router.select_experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                )

                final_hidden_states = self._apply_with_tiered_cache(
                    layer=layer,
                    x=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    shared_experts_input=shared_input,
                )

            if has_separate_shared_experts:
                assert self.shared_experts is not None

                if use_shared_experts_stream:
                    # shared experts 在独立 stream 上与 routed experts 并行计算。
                    # 这里启动独立 stream，并在算完后立刻标记同步终点，
                    # 以免后续 cuda graph replay 过程中额外分配过多 stream。
                    with torch.cuda.stream(self.shared_experts_stream):
                        # 这里必须 clone，避免与主 stream 发生读写冲突。
                        shared_output = self.shared_experts(shared_experts_input)
                    current_stream().wait_stream(self.shared_experts_stream)

                # 输出统一组织成 (shared_output, routed_output)。
                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )

            # -----------------
            # combine 阶段。
            # -----------------
            def combine_output(states: torch.Tensor) -> torch.Tensor:
                if do_naive_dispatch_combine:
                    # naive 路径下，把各 expert rank 的结果重新合并回原 token 顺序。
                    states = get_ep_group().combine(
                        states, self.moe_config.is_sequence_parallel
                    )

                if self.moe_config.pcp_size > 1:
                    # PCP 路径下，再把 gather 过的结果通过 reduce_scatter 收回到本 rank。
                    states = get_pcp_group().reduce_scatter(
                        states,
                        dim=0,
                    )

                return states

            # shared experts 分支只 combine routed 输出；shared 输出由调用方决定何时规约。
            if self.shared_experts is not None:
                return (
                    final_hidden_states[0],
                    combine_output(final_hidden_states[1]),
                )
            else:
                return combine_output(final_hidden_states)
