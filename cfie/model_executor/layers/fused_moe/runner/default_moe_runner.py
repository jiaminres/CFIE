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
    """
    FusedMoE 的默认执行编排器。

    这个类本身不持有专家权重；权重、kernel 配置和并行元数据仍属于外层
    `FusedMoE` / `SharedFusedMoE` layer。`DefaultMoERunner` 的职责更像
    “运行时调度器”：

    - 接收上层传入的 `hidden_states`、`router_logits`
    - 视配置决定 gate、router、shared experts 的执行顺序
    - 在需要时接入 CFIE tiered cache、DP chunking、独立 shared stream
    - 调用 quant method / monolithic kernel 完成 routed experts 主计算
    - 在末尾完成 dispatch/combine、规约与输出裁剪

    因而它统一覆盖了 routed experts、shared experts、TP/EP/SP 并行、
    monolithic / decomposed kernel，以及 CFIE 特有 tiered cache 的组合路径。
    """

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
        # 先初始化抽象基类 `MoERunner`。
        # 这里主要是让父类有机会建立自己的基础状态。
        super().__init__()

        # 保存当前层对应的 MoE 静态配置，例如 hidden_dim、tp/ep/sp 拓扑等。
        self.moe_config = moe_config
        # 保存 router 对象；后续非 monolithic 路径会用它把 router logits 转成 top-k experts。
        self.router = router
        # 保存 routed experts 的可选输入变换模块，例如 latent MoE 的前置投影。
        self.routed_input_transform = routed_input_transform
        # 保存可选的 gate 模块；若不为空，runner 会在内部先调用它产出 router logits。
        self.gate = gate
        # 保存 shared experts 分支模块；若为空，说明当前层没有 shared expert。
        self.shared_experts = shared_experts
        # 保存底层量化 / kernel 执行入口；真正的专家计算最终都会落到这里。
        self.quant_method = quant_method
        # 记录当前层是否希望在更外层返回前完成规约。
        self.reduce_results = reduce_results
        # 记录是否启用 DBO；它会影响 DP chunking staging buffer 的布局。
        self.enable_dbo = enable_dbo

        # -----------------
        # shared experts 独立 stream 相关初始化。
        # -----------------
        # 出于调试目的，允许通过环境变量禁用 shared experts 的独立 stream。
        # TODO: 等 TP / DP 与其他执行模式验证更充分后，可移除这条调试开关。
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            # 若环境变量显式禁用，就完全不为 shared experts 单独分配辅助 stream。
            logger.debug_once("Disabling MoE shared_experts cuda stream", scope="local")
            # 记为 None，后续分支会自然退化为“与主流串行执行”。
            self.shared_experts_stream = None
        else:
            # TODO(rob): 为非 cuda-alike 平台补上 shared expert overlap 支持。
            # 非 cuda-alike 平台上，aux_stream() 会直接返回 None。
            # 尝试申请一条辅助 stream，供 shared experts 与 routed experts 并行执行。
            self.shared_experts_stream = aux_stream()
            if self.shared_experts_stream is not None:
                # 只有真正拿到辅助 stream 时才打印启用日志。
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
                # 没有 shared experts 时，绑定只返回 routed 输出的入口。
                self.moe_forward = _moe_forward
            else:
                # 有 shared experts 时，绑定会返回 `(shared, routed)` 的入口。
                self.moe_forward = _moe_forward_shared
        else:
            if self.shared_experts is None:
                # CUDA/自定义算子路径下，优先走注册好的 `torch.ops.cfie.moe_forward`。
                self.moe_forward = torch.ops.cfie.moe_forward
            else:
                # shared 版本的 custom op 会在内部转发到 runner 的 shared 路径。
                self.moe_forward = torch.ops.cfie.moe_forward_shared

        # DP chunking 场景下会复用这两块 staging buffer，避免每个 chunk 重新分配显存。
        # 这里先置空，等第一次真正需要 chunking 时再懒初始化。
        self.batched_hidden_states: torch.Tensor | None = None
        # router logits 也对应维护一块 staging buffer，与 hidden_states 同步复用。
        self.batched_router_logits: torch.Tensor | None = None

    def _apply_with_tiered_cache(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # -----------------
        # decomposed/top-k 路径下的 tiered cache 桥接。
        # -----------------
        # 这条路径对应“router 先选出 top-k experts，再调用 quant_method.apply(...)”
        # 的执行方式。若当前层未挂 tiered cache controller，就直接把 top-k 结果交给
        # quant_method；否则先根据本批 token 触达的 experts 集合决定：
        # - 直接 prepare 后执行
        # - 走 burst 执行区
        # - 或拆成多个 token chunk 分批执行
        # 这是 CFIE tiered cache 的执行桥接层：
        # 若当前层没挂 controller，就直接走原始 quant_method.apply；
        # 否则先确保本次请求涉及的 experts 已经在 GPU resident slots 中。
        # 尝试从层对象上拿到 CFIE tiered cache controller。
        controller = getattr(layer, "_cfie_tiered_cache_controller", None)
        if controller is None:
            # 若当前层没有 tiered cache，就直接把 top-k 路由结果交给 quant method 执行。
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
            # `detach()` 只是为了避免这类统计逻辑意外进入 autograd 图。
            unique_requested = int(torch.unique(chunk_topk_ids.detach()).numel())
            # 当前 chunk 的 token 数也会影响 burst 策略判断。
            num_chunk_tokens = int(chunk_topk_ids.shape[0])
            if unique_requested <= layer.local_num_experts:
                # resident slots 足够时，先 prepare 把缺失 experts 换入，再正常执行。
                # `prepare(...)` 会确保本 chunk 需要的 experts 已经驻留到可执行槽位。
                controller.prepare(chunk_topk_ids)
                # resident 容量足够时，后续执行与普通 quant_method.apply 完全一致。
                return self.quant_method.apply(
                    layer=layer,
                    x=chunk_x,
                    topk_weights=chunk_topk_weights,
                    topk_ids=chunk_topk_ids,
                    shared_experts_input=chunk_shared_input,
                )

            if controller.can_run_prefill_burst(unique_requested, num_chunk_tokens):
                # 若主 resident slots 装不下，但 burst pool 能兜住，就走临时 burst 执行区。
                # 这条路径不要求把 experts 全部换入常驻槽位，而是借助 burst 区临时执行。
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
        # 统计整块输入一共会命中多少个不同 expert。
        full_unique_requested = int(torch.unique(topk_ids.detach()).numel())
        # 统计整块输入的 token 总数。
        full_num_tokens = int(topk_ids.shape[0])
        # 预先判断整块输入是否可以整体走 burst 路径。
        full_can_use_burst = controller.can_run_prefill_burst(
            full_unique_requested,
            full_num_tokens,
        )
        if (
            full_unique_requested <= layer.local_num_experts
            or full_can_use_burst
        ):
            # 整块可直接执行时，不再额外切 token chunk。
            # 这样可以避免不必要的 token 维拆分和后续再拼接。
            return apply_chunk(x, topk_weights, topk_ids, shared_experts_input)

        # -----------------
        # 整块超容量时，按 expert 容量重新切 token 范围。
        # -----------------
        # 读取 burst 区最多还能额外承载多少个 experts。
        burst_capacity = getattr(controller, "prefill_burst_capacity", 0)
        # 读取启用 burst 路径所要求的最小 token 数阈值。
        burst_min_tokens = getattr(controller, "prefill_burst_min_tokens", 0)
        # 根据当前 batch 大小和 burst 配置，决定单个 token chunk 允许触达的最大 expert 数。
        capacity = (
            max(layer.local_num_experts, burst_capacity)
            if burst_capacity > 0 and full_num_tokens >= burst_min_tokens
            else layer.local_num_experts
        )
        # 每个切片都要保证其唯一 experts 数不超过可执行容量。
        # 返回的是若干 `(start, end)` token 区间。
        token_ranges = _pack_token_ranges_by_expert_capacity(topk_ids, capacity)

        # 依次收集各个 token chunk 的执行结果。
        outputs: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        for start, end in token_ranges:
            # shared experts 输入也必须同步切成相同 token 范围。
            chunk_shared_input = (
                shared_experts_input[start:end]
                if shared_experts_input is not None
                else None
            )
            # 对当前 token 区间执行一次完整的 apply_chunk。
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
            # 若 quant method 返回 `(shared, routed)` 二元组，就需要分别拼接两个分支。
            return (
                torch.cat([output[0] for output in outputs], dim=0),
                torch.cat([output[1] for output in outputs], dim=0),
            )
        # 普通 routed-only 路径则直接拼接单个输出张量。
        return torch.cat(outputs, dim=0)

    def _apply_monolithic_with_tiered_cache(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # -----------------
        # monolithic kernel 路径下的 tiered cache 桥接。
        # -----------------
        # monolithic kernel 会把“router + experts”打成一个整体执行，因此正式计算时
        # 不会显式经过 top-k prepare/finalize 流程；但 tiered cache 仍然需要事先知道
        # 当前 batch 会触达哪些 experts，才能决定是否 prepare / 分 chunk。
        # monolithic 路径同样先尝试取得 tiered cache controller。
        controller = getattr(layer, "_cfie_tiered_cache_controller", None)
        if controller is None:
            # 若没有 tiered cache，就直接调用 monolithic kernel。
            return self.quant_method.apply_monolithic(
                layer=layer,
                x=x,
                router_logits=router_logits,
            )

        # monolithic kernel 虽然把 router + experts 合在一起执行，
        # 但为了判断 tiered cache 容量，仍需先在 Python 侧做一次 expert 选择。
        _, topk_ids = self.router.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )

        # 统计整块输入会触达多少个唯一 experts。
        full_unique_requested = int(torch.unique(topk_ids.detach()).numel())
        if full_unique_requested <= layer.local_num_experts:
            # 若常驻槽位足够，就先 prepare 再整体执行 monolithic kernel。
            controller.prepare(topk_ids)
            return self.quant_method.apply_monolithic(
                layer=layer,
                x=x,
                router_logits=router_logits,
            )

        # 若整块输入超出容量，就按 expert 容量重新切 token 区间。
        token_ranges = _pack_token_ranges_by_expert_capacity(
            topk_ids,
            layer.local_num_experts,
        )

        # 逐个 token chunk 调 monolithic kernel，并收集各 chunk 输出。
        outputs: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        for start, end in token_ranges:
            # 每个 chunk 执行前都要让 controller 先把这一段需要的 experts prepare 到位。
            controller.prepare(topk_ids[start:end])
            outputs.append(
                self.quant_method.apply_monolithic(
                    layer=layer,
                    x=x[start:end],
                    router_logits=router_logits[start:end],
                )
            )

        first_output = outputs[0]
        if isinstance(first_output, tuple):
            # shared/routed 二元组输出时，两个分支分别沿 token 维拼接。
            return (
                torch.cat([output[0] for output in outputs], dim=0),
                torch.cat([output[1] for output in outputs], dim=0),
            )
        # 纯 routed 输出则直接拼接。
        return torch.cat(outputs, dim=0)

    @property
    def use_dp_chunking(self) -> bool:
        # 返回当前 backend 是否应该启用 DP chunking。
        # 只有部分 all2all kernel 支持 / 受益于把大 batch 切块执行，并且还要受
        # 环境变量开关控制。
        # 这里前半部分判断“当前 backend 能不能 / 要不要 chunk”，
        # 最后的环境变量则是人工总开关。
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
        # 判断 shared experts 是否要放到独立 CUDA stream 上和 routed experts 并行。
        # 返回值：
        # - `use_shared_experts_stream`：本次 forward 是否真的启用独立 stream
        # - `shared_experts_input`：若启用独立 stream，需要提前固定好它要消费的输入
        # 决定是否让 shared experts 在独立 CUDA stream 上和 routed experts 并行执行。
        # 只有满足以下条件时，shared experts 才值得放到独立 CUDA stream：
        # 1. 当前确实是 CUDA 平台；
        # 2. shared experts 没被 monolithic kernel 接管；
        # 3. 当前不走 chunked 路径；
        # 4. 成功申请到了辅助 stream；
        # 5. token 数没有超过环境变量给定阈值。
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

        # 默认先记为 None；只有真的启用独立 stream 时才准备专门的输入张量引用。
        shared_experts_input: torch.Tensor | None = None
        if use_shared_experts_stream:
            # 既然决定启用独立 stream，这里就要求辅助 stream 必须存在。
            assert self.shared_experts_stream is not None
            # shared experts 和 routed experts 并行时，主路径不能原地覆写输入。
            assert self.moe_config.disable_inplace

            # 若调用方已显式给了 shared_input，就沿用它；
            # 否则 shared experts 默认直接消费当前的 hidden_states。
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
        # 按需分配 DP chunking 复用缓冲区。
        # 这些 staging tensor 会在一次 forward 的多个 chunk 之间反复复用，
        # 避免每个 chunk 都重新申请显存。
        # 若当前 backend 不启用 DP chunking，或已经初始化过 staging buffer，就直接返回。
        if not self.use_dp_chunking or self.batched_hidden_states is not None:
            return

        # 先声明好 hidden states / router logits 两块 staging buffer 的形状变量。
        states_shape: tuple[int, ...]
        logits_shape: tuple[int, ...]

        # 简写当前 MoE 配置，后面多处都会用到。
        moe = self.moe_config

        if self.enable_dbo:
            # DBO 打开时，多预留一个 ubatch 维度。
            states_shape = (2, moe.max_num_tokens, self.moe_config.hidden_dim)
            logits_shape = (2, moe.max_num_tokens, self.moe_config.num_logical_experts)
        else:
            # 非 DBO 模式下，staging buffer 只有 `[tokens, hidden/logits]` 两维。
            states_shape = (moe.max_num_tokens, self.moe_config.hidden_dim)
            logits_shape = (moe.max_num_tokens, self.moe_config.num_logical_experts)

        # 在当前设备上一次性分配 chunk staging buffer。
        # 后续所有 chunk 都会反复写入这两块张量。
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
        # 这个历史接口名容易误解。
        # 当前实现里，返回 True 更接近“shared expert 输出已经满足最终规约要求，
        # 调用方无需再额外 all-reduce”；返回 False 才表示后面还要补一次 TP 规约。
        # shared experts 一般由 RowParallelLinear 计算。
        # 纯 TP 场景下可延后到 MoE 末尾再规约；
        # 但 EP + all2all 场景下，各 DP rank 会持有完整 hidden_states，
        # 因此需要尽早规约 shared experts 输出。
        # 这里仍要求 quant_method 必须存在，因为判断逻辑依赖底层 kernel 能力。
        assert self.quant_method is not None
        # 某些 kernel 已经在内部完成规约，这时 shared experts 输出无需再额外 reduce。
        return (
            # `moe_kernel is not None` 表示当前 quant method 已经挂上具体内核实现。
            self.quant_method.moe_kernel is not None
            # 若内核声明输出已经规约完成，则这里返回 True，外层无需再补 all-reduce。
            and self.quant_method.moe_kernel.output_is_reduced()
        )

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        # 对外提供一个“按需规约”的统一入口。
        # 若底层 kernel 已经保证输出满足最终规约要求，就原样返回；
        # 否则这里补一次 TP all-reduce。
        if self.must_reduce_shared_expert_outputs():
            # 若底层已经规约完，就直接把输入结果向外透传。
            return final_hidden_states
        else:
            # 否则显式在 TP group 上做一次 all-reduce。
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def apply_routed_input_transform(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对 routed experts 输入做可选的前置变换。
        # 常见场景是 latent MoE：shared experts 继续消费原始 hidden_states，
        # 但 routed experts 先投影到更小的 latent 维度再进入 MoE kernel。
        # 仅对 routed experts 输入做额外变换，例如 latent projection。
        # FusedMoE.forward_native 会保留原始 hidden_states 给 shared experts，
        # 而 routed experts 则使用变换后的 [S, moe_latent_size] 输入。
        # TODO: 为了进一步降低 latent MoE 的带宽开销，fc2_latent_proj 未来可考虑
        # 下沉到 SharedFusedMoE 内部，在更小的 latent 维度上做 all-reduce。
        # routed experts 和 shared experts 的输入维可能不同，因此 routed 分支可单独变换。
        if self.routed_input_transform is not None:
            # 若配置了 routed_input_transform，就先对 routed 分支输入做一次变换。
            result = self.routed_input_transform(hidden_states)
            # ReplicatedLinear 会返回 (output, extra_bias)；
            # 这里只需要真正的输出张量。
            if isinstance(result, tuple):
                return result[0]
            # 普通模块直接返回单个张量时，原样透传。
            return result
        # 若未配置额外变换，就直接使用原始 hidden_states。
        return hidden_states

    def _reduce_output(
        self,
        states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        trunc_sizes: list[int],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 统一处理输出后收尾逻辑。
        # 这里同时负责：
        # - 必要时执行 TP all-reduce
        # - 把为了适配 kernel 而 pad 的 hidden_dim 裁回真实维度
        # - 同时兼容“仅 routed 输出”和“(shared, routed) 二元组输出”
        def trunc(x: torch.Tensor, trunc_size: int) -> torch.Tensor:
            # 只保留真实 hidden 维，去掉为了适配 kernel 补出来的尾部维度。
            return x[..., :trunc_size]

        def reduce_and_trunc(x: torch.Tensor, trunc_size: int) -> torch.Tensor:
            # 先按需做 TP all-reduce，再裁回真实 hidden 维。
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
            # 其余情况下只需要裁剪，不需要在这里补额外规约。
            func = trunc

        if isinstance(states, tuple):
            # `(shared, routed)` 二元组输出时，对两个分支分别应用同一套后处理逻辑。
            return tuple(
                [func(s, trunc_size) for s, trunc_size in zip(states, trunc_sizes)]
            )
        else:
            # 单输出场景下，`trunc_sizes` 里只应有一个目标维度。
            assert len(trunc_sizes) == 1
            return func(states, trunc_sizes[0])

    def _encode_layer_name(self) -> str | ModuleName:
        # 把 Python 层对象的 `layer_name` 编码成 custom op 可以识别的句柄。
        # 某些路径下 custom op 只拿得到一个轻量标识，后面再通过 forward context
        # 反查真实层对象。
        if HAS_OPAQUE_TYPE:
            # 若当前运行环境支持 opaque 类型，就直接封装成 `ModuleName` 传给 custom op。
            return ModuleName(self.layer_name)
        # 单测环境里 forward context 可能不存在，或 all_moe_layers 为空。
        if (
            is_forward_context_available()
            and get_forward_context().all_moe_layers is not None
        ):
            # 若 forward context 可用，就让 custom op 走“从上下文反查层对象”的分支。
            return "from_forward_context"
        # 最后兜底返回普通字符串层名。
        return self.layer_name

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # -----------------
        # runner 的公开前向入口。
        # -----------------
        # 这里主要负责两件事：
        # 1. 准备 routed / shared 两条分支各自需要的输入形状
        # 2. 调用注册好的 `moe_forward` custom op 或 Python fallback 入口，
        #    再在末尾统一做规约与维度裁剪
        # -----------------
        # 入口：准备 routed / shared experts 各自需要的输入。
        # -----------------
        # latent MoE 下先保留原始 hidden_states：
        # shared experts 仍用原始维度，routed experts 则走变换后的维度。
        if self.shared_experts is not None:
            # shared experts 存在时，需要保留一份原始输入给 shared 分支使用。
            original_hidden_states = hidden_states
            # 同时记录原始 hidden 维，后面给 shared 输出裁剪时会用到。
            original_hidden_dim = hidden_states.shape[-1]
        else:
            # 没有 shared experts 时，这两个值后面都不会参与实际计算。
            original_hidden_states = None

        # 先对 routed experts 输入做可选变换，例如 latent projection。
        hidden_states = self.apply_routed_input_transform(hidden_states)

        # routed 分支若被 pad 到 kernel 需要的 hidden_dim，最后还要再裁回去。
        # 这里记录的是变换后的维度，后面裁剪 routed 输出时会用到。
        transformed_hidden_dim = hidden_states.shape[-1]
        if self.moe_config.hidden_dim != transformed_hidden_dim:
            # 若变换后维度小于 kernel 期望维度，就在尾部补 0 对齐到 kernel 的 hidden_dim。
            hidden_states = F.pad(
                hidden_states,
                (0, self.moe_config.hidden_dim - transformed_hidden_dim),
                mode="constant",
                value=0.0,
            )

        # 真正执行 routed experts 的 forward；shared_experts_input 作为第三参数透传。
        # 这里的 `self.moe_forward` 可能是 Python fallback，也可能是注册好的 custom op。
        fused_output = self.moe_forward(
            hidden_states,
            router_logits,
            original_hidden_states,
            self._encode_layer_name(),
        )

        # 根据是否存在 shared experts，准备最后的裁剪尺寸列表。
        if self.shared_experts is not None:
            # shared/routed 双输出时，要分别把两个分支裁回各自真实维度。
            orig_hidden_dims = [original_hidden_dim, transformed_hidden_dim]
        else:
            # 只有 routed 输出时，只需要保留一个目标维度。
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
        # DP chunking 专用执行路径。
        # -----------------
        # 当单次 token 批次过大、backend 又支持 DP chunking 时，会把完整输入切成多个
        # token chunk，重复执行“stage -> route -> experts -> write back”，最后再把所有
        # chunk 的输出拼回完整张量。
        # 进入这里前，外层已经决定当前 backend 需要 / 支持 DP chunking。
        assert self.batched_hidden_states is not None
        assert self.batched_router_logits is not None
        # staging buffer 与当前输入必须使用同一 dtype，否则后续 copy_/kernel 调用会出错。
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

        # 预先分配完整 routed 输出张量；每个 chunk 执行完后会把自己的结果写回对应区间。
        full_fused_final_hidden_states = torch.empty_like(full_hidden_states)
        if self.shared_experts is not None:
            # 若存在 shared experts，也为 shared 分支单独预分配一块完整输出张量。
            full_shared_final_hidden_states = torch.empty_like(full_hidden_states)

        # -----------------
        # 单个 chunk 的处理函数。
        # -----------------
        def process_chunk(chunk_start, chunk_end, skip_result_store=False):
            # 当前 chunk 覆盖的 token 数。
            chunk_size = chunk_end - chunk_start
            # 先从完整输入中切出当前 chunk。
            hidden_states = full_hidden_states[chunk_start:chunk_end, :]
            router_logits = full_router_logits[chunk_start:chunk_end, :]
            # shared_input 若存在，也必须按同样的 token 区间切片。
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
                # 非 DBO 模式直接复用整块 staging buffer。
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
            # 使用 `non_blocking=True` 允许在满足条件时走异步拷贝。
            staged_hidden_states.copy_(hidden_states, non_blocking=True)
            staged_router_logits.copy_(router_logits, non_blocking=True)

            # shared 分支若没有单独输入，就默认直接使用 staged 后的 hidden_states。
            shared_input = (
                shared_input if shared_input is not None else staged_hidden_states
            )

            # -----------------
            # chunk 内部的 MoE 主计算。
            # -----------------
            # 核心专家计算阶段。
            if self.quant_method.is_monolithic:
                # monolithic kernel 会在一个入口里同时完成 router + experts 主计算。
                assert has_separate_shared_experts or self.shared_experts is None
                final_hidden_states = self._apply_monolithic_with_tiered_cache(
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

                # top-k 结果会决定 routed token 实际命中的 experts 以及后续 tiered cache prepare。
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
                # 这里 shared 分支和 routed 分支对齐在同一 token chunk 上。
                shared_output = self.shared_experts(shared_input)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )

            if not skip_result_store:
                # 把当前 chunk 的结果写回完整输出张量。
                if self.shared_experts is None:
                    # routed-only 场景下，直接写回 routed 输出。
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states, non_blocking=True
                    )
                else:
                    # shared/routed 双输出场景下，两块完整输出张量分别写回。
                    full_shared_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[0], non_blocking=True
                    )
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[1], non_blocking=True
                    )

        # -----------------
        # 计算 chunk 循环边界。
        # -----------------
        # forward context 中会记录当前 DP dispatcher 在整个 batch 上看到的 token 上界。
        ctx = get_forward_context()
        # flashinfer_cutlass kernel 可同时覆盖可选的 DP 与 TP/EP 组合。
        max_tokens_across_dispatchers = ctx.dp_metadata.max_tokens_across_dp_cpu
        # 单个 rank 每次最多处理多少 token，由 moe 配置里的 `max_num_tokens` 决定。
        moe_dp_chunk_size_per_rank = self.moe_config.max_num_tokens

        # 若输入本身走了 sequence parallel，需要先除以 sp_size，
        # 才能得到单个 dispatcher 实际可能看到的最大 token 数。
        if self.moe_config.is_sequence_parallel:
            max_tokens_across_dispatchers = cdiv(
                max_tokens_across_dispatchers, self.moe_config.sp_size
            )

        # 当前真实输入里实际包含多少 token。
        num_tokens = full_hidden_states.size(0)
        # 逐个 chunk 进入前面的 process_chunk。
        for chunk_idx, chunk_start_ in enumerate(
            range(0, max_tokens_across_dispatchers, moe_dp_chunk_size_per_rank)
        ):
            # 初始 chunk 起点就是这轮循环对应的 token 偏移。
            chunk_start = chunk_start_
            # 初始 chunk 终点受“chunk 大小”和“dispatcher 最大 token 上界”双重限制。
            chunk_end = min(
                chunk_start + moe_dp_chunk_size_per_rank, max_tokens_across_dispatchers
            )
            # 再把 chunk 边界裁到当前真实 token 范围内。
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)
            # 进入 `chunked_sizes(...)` 上下文后，下游 kernel / communicator 可以读取当前
            # chunk 在 DP/SP 视角下的元信息。
            with ctx.dp_metadata.chunked_sizes(
                self.moe_config.sp_size, moe_dp_chunk_size_per_rank, chunk_idx
            ):
                # 若这轮循环的起点已经超出真实 token 范围，就只更新元信息而跳过结果写回。
                process_chunk(
                    chunk_start, chunk_end, skip_result_store=chunk_start_ >= num_tokens
                )

        # 所有 chunk 处理完后，按是否存在 shared experts 决定返回结构。
        if self.shared_experts is None:
            # routed-only 场景直接返回完整 routed 输出。
            return full_fused_final_hidden_states
        else:
            # shared/routed 双输出场景保持 `(shared, routed)` 返回结构。
            return (full_shared_final_hidden_states, full_fused_final_hidden_states)

    def forward_impl(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # ------------------------------- 主流程入口 -------------------------------
        # 执行单层 MoE 的完整前向主线：
        # 先准备路由输入与 shared 分支状态，再执行 routed 主计算，
        # 最后按并行拓扑规约输出并返回。
        assert self.quant_method is not None

        # 若当前后端可能走 DP chunking，这里先确保 staging buffer 已就绪。
        # 这样后续无论是否真正进入 chunked 分支，都不会访问未初始化状态。
        self.ensure_dp_chunking_init()

        # 只有 shared experts 未被内核内部融合接管时，
        # runner 才需要显式调 shared 分支并管理其执行时机。
        has_separate_shared_experts = (
            not self.quant_method.mk_owns_shared_expert
            and self.shared_experts is not None
        )

        # 根据并行后端能力判断本次是否走 chunked 专用实现。
        use_chunked_impl = self.use_dp_chunking

        # 决定 shared experts 是否放到独立 CUDA stream 异步执行，
        # 并提前确定 shared 分支要消费的输入张量。
        use_shared_experts_stream, shared_experts_input = (
            self._maybe_setup_shared_experts_stream(
                hidden_states,
                shared_input,
                has_separate_shared_experts,
                use_chunked_impl,
            )
        )

        # ------------------------------- gate/router 预处理 -------------------------------
        # 若当前层内置了 gate，就以层内 gate 的结果为准覆盖外部 logits，
        # 让后续 shared/routed 两条路径读取同一份路由信息。
        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)

        # ------------------------------- chunked 快速分支 -------------------------------
        # chunked 模式下直接切到专用实现，避免与非 chunked 逻辑交叉。
        if use_chunked_impl:
            return self.forward_impl_chunked(
                layer,
                hidden_states,
                router_logits,
                shared_input,
                has_separate_shared_experts,
            )

        # ------------------------------- 非 chunked 主路径 -------------------------------
        # 仅在 DP>1 且量化实现不支持内核内部分发时，才启用 naive dispatch/combine。
        # TODO(rob): 等所有 quant method 迁移到 MK 后，可删除该兼容分支。
        do_naive_dispatch_combine = (
            self.moe_config.dp_size > 1 and not self.quant_method.supports_internal_mk
        )

        # 读取当前 forward 的并行上下文元数据。
        ctx = get_forward_context()
        # 若存在 DP metadata，则按 SP 口径设置本地 token 视图；
        # 若不存在，则用空上下文保持后续写法一致。
        sp_ctx = (
            ctx.dp_metadata.sp_local_sizes(self.moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

        with sp_ctx:
            # ------------------------------- shared experts 预执行 -------------------------------
            # 若 shared 分支不走独立 stream，则先在主 stream 同步执行。
            # 这样可以在 routed 路径改写输入前拿到稳定的 shared 输出。
            if has_separate_shared_experts and not use_shared_experts_stream:
                assert self.shared_experts is not None
                # 若未显式提供 shared_input，则默认复用当前 hidden_states。
                shared_input = (
                    shared_input if shared_input is not None else hidden_states
                )
                shared_output = self.shared_experts(shared_input)

            # ------------------------------- dispatch 阶段 -------------------------------
            # naive 路径下先把 token 与路由 logits 分发到目标 expert rank，
            # 让本 rank 只处理自己负责的专家输入。
            # TODO: 等所有 kernel 迁移到 MoEKernel 框架后，可删除该分支。
            if do_naive_dispatch_combine:
                hidden_states, router_logits = get_ep_group().dispatch_router_logits(
                    hidden_states,
                    router_logits,
                    self.moe_config.is_sequence_parallel,
                )

            # ------------------------------- PCP gather 阶段 -------------------------------
            # PCP 开启时，在 token 维收集各 rank 分片，
            # 让后续 routed 计算基于完整上下文视图执行。
            if self.moe_config.pcp_size > 1:
                hidden_states = get_pcp_group().all_gather(
                    hidden_states,
                    dim=0,
                )
                router_logits = get_pcp_group().all_gather(
                    router_logits,
                    dim=0,
                )

            # ------------------------------- routed experts 主计算 -------------------------------
            # 核心专家计算阶段：
            # monolithic 路径由统一入口完成路由与专家计算；
            # 非 monolithic 路径先选 top-k，再按 top-k 执行专家前向。
            if self.quant_method.is_monolithic:
                final_hidden_states = self._apply_monolithic_with_tiered_cache(
                    layer=layer,
                    x=hidden_states,
                    router_logits=router_logits,
                )
            else:
                topk_weights, topk_ids = self.router.select_experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                )

                # top-k 结果会作为 routed 路径后续计算与合并的驱动信号。
                final_hidden_states = self._apply_with_tiered_cache(
                    layer=layer,
                    x=hidden_states,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    shared_experts_input=shared_input,
                )

            # ------------------------------- shared experts 并行收敛 -------------------------------
            # 若存在独立 shared 分支，这里把 shared 与 routed 两路结果对齐并组织成统一结构。
            if has_separate_shared_experts:
                assert self.shared_experts is not None

                if use_shared_experts_stream:
                    # shared experts 在辅助 stream 上执行，与 routed 计算并行。
                    with torch.cuda.stream(self.shared_experts_stream):
                        shared_output = self.shared_experts(shared_experts_input)
                    # 在读取 shared_output 前，主 stream 必须等待辅助 stream 完成。
                    current_stream().wait_stream(self.shared_experts_stream)

                # 输出统一组织为 (shared_output, routed_output)。
                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )

            # ------------------------------- combine 阶段 -------------------------------
            # 按并行拓扑把 routed 输出规约回当前 rank 的最终布局。
            def combine_output(states: torch.Tensor) -> torch.Tensor:
                # naive 路径下先做 EP 侧 combine，把分发后的 routed 输出拼回原 token 视图。
                if do_naive_dispatch_combine:
                    states = get_ep_group().combine(
                        states, self.moe_config.is_sequence_parallel
                    )

                # PCP 路径下再做 reduce_scatter，把上下文维聚合结果切回本 rank。
                if self.moe_config.pcp_size > 1:
                    states = get_pcp_group().reduce_scatter(
                        states,
                        dim=0,
                    )

                return states

            # shared+routed 模式下，仅 routed 分支在此处 combine；
            # shared 分支保留给外层决定规约/相加时机。
            if self.shared_experts is not None:
                return (
                    final_hidden_states[0],
                    combine_output(final_hidden_states[1]),
                )
            else:
                # routed-only 模式下，直接返回 combine 后的单张量输出。
                return combine_output(final_hidden_states)
