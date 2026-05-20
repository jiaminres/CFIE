# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
import os
import time as _time

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
from cfie.config import CUDAGraphMode
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
    # 浠?forward context 涓寜鍚嶅瓧鍙栧洖鐪熷疄鐨?FusedMoE 灞傚璞°€?
    forward_context: ForwardContext = get_forward_context()
    if layer_name == "from_forward_context":
        # 鏌愪簺缂栬瘧璺緞涓嶄細鐩存帴浼犵湡瀹炲眰鍚嶏紝鑰屾槸瑕佹眰鎸夎皟鐢ㄩ『搴忓洖鏀惧綋鍓嶅眰銆?
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
        # 姣忓彇涓€娆￠兘鎶婄储寮曞悜鍓嶆帹杩涳紝淇濊瘉涓嬩竴娆¤兘鎷垮埌鍚庣画 MoE 灞傘€?
        forward_context.moe_layer_index += 1
    return forward_context.no_compile_layers[layer_name]


# torch >= 2.11 鏃讹紝layer_name 浼氳鎻愬崌鎴?ModuleName 涓嶉€忔槑瀵硅薄锛?
# 鏇存棭鐗堟湰閲岋紝瀹冧粛鐒跺彧鏄櫘閫氬瓧绗︿覆銆?
if TYPE_CHECKING:
    from typing import TypeAlias

    _layer_name_type: TypeAlias = str | ModuleName
else:
    _layer_name_type = ModuleName if HAS_OPAQUE_TYPE else str


def _resolve_layer_name(layer_name: str | ModuleName) -> str:
    # 鍏煎 torch 鏂版棫鐗堟湰瀵?layer_name 鐨勪笉鍚屽皝瑁呭舰寮忋€?
    return layer_name.value if isinstance(layer_name, ModuleName) else layer_name


def _pack_token_ranges_by_expert_capacity(
    topk_ids: torch.Tensor,
    capacity: int,
) -> list[tuple[int, int]]:
    # 鎸夆€滀竴涓?chunk 鍐呭厑璁稿嚭鐜扮殑鍞竴 experts 涓婇檺鈥濇妸 token 鍒囨垚澶氫釜杩炵画鑼冨洿銆?
    if topk_ids.numel() == 0:
        return []

    ranges: list[tuple[int, int]] = []
    current_start = 0
    current_experts: set[int] = set()

    for token_idx in range(topk_ids.shape[0]):
        # 褰撳墠 token 鍙兘璺敱鍒板涓?expert锛岃繖閲屽厛鍘婚噸鍐嶇粺璁°€?
        token_unique_ids = {
            int(expert_id)
            for expert_id in torch.unique(topk_ids[token_idx].detach()).tolist()
        }
        if len(token_unique_ids) > capacity:
            raise RuntimeError(
                f"token {token_idx} requested {len(token_unique_ids)} experts, "
                f"which exceeds tiered-cache capacity {capacity}"
            )

        # 灏濊瘯鎶婂綋鍓?token 鍚堝苟杩涚幇鏈?chunk锛屽苟浼扮畻鍚堝苟鍚庣殑鍞竴 experts 鏁般€?
        candidate_experts = current_experts | token_unique_ids
        if token_idx > current_start and len(candidate_experts) > capacity:
            # 涓€鏃﹁秴瀹归噺锛屽氨鍦ㄥ墠涓€涓?token 澶勬埅鏂紝寮€鍚柊鐨?chunk銆?
            ranges.append((current_start, token_idx))
            current_start = token_idx
            current_experts = set(token_unique_ids)
        else:
            # 浠嶆湭瓒呭閲忔椂锛屾妸褰撳墠 token 缁х画骞跺叆褰撳墠 chunk銆?
            current_experts = candidate_experts

    # 鏀跺熬琛ヤ笂鏈€鍚庝竴涓?chunk銆?
    ranges.append((current_start, topk_ids.shape[0]))
    return ranges


def _should_stabilize_moe_output_for_cudagraph() -> bool:
    if not is_forward_context_available():
        return False
    forward_context = get_forward_context()
    return (
        forward_context.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
        and forward_context.batch_descriptor is not None
    )


def _copy_to_stable_moe_output(
    layer: torch.nn.Module,
    output: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    buffers = getattr(layer, "_cfie_piecewise_moe_output_buffers", None)
    if buffers is None:
        buffers = {}
        setattr(layer, "_cfie_piecewise_moe_output_buffers", buffers)

    key = (
        name,
        tuple(output.shape),
        tuple(output.stride()),
        output.dtype,
        output.device,
    )
    stable = buffers.get(key)
    if stable is None:
        stable = torch.empty_strided(
            tuple(output.shape),
            tuple(output.stride()),
            dtype=output.dtype,
            device=output.device,
        )
        buffers[key] = stable
    stable.copy_(output)
    return stable


def _stabilize_moe_output_for_cudagraph(
    layer: torch.nn.Module,
    output: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if not _should_stabilize_moe_output_for_cudagraph():
        return output
    if isinstance(output, torch.Tensor):
        return _copy_to_stable_moe_output(layer, output, name="output")
    return tuple(
        _copy_to_stable_moe_output(layer, value, name=f"output_{idx}")
        if isinstance(value, torch.Tensor)
        else value
        for idx, value in enumerate(output)
    )


def _moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> torch.Tensor:
    # custom op 鍏ュ彛锛氬厛鎸?layer_name 鎵惧洖灞傚璞★紝鍐嶈浆浜ょ粰 runner 鐪熸鎵ц銆?
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    # TODO(bnell): 绛?MK 杩佺Щ瀹屾垚鍚庯紝杩欓噷鐨勫吋瀹瑰垵濮嬪寲鍙垹闄ゃ€?
    layer.ensure_moe_quant_config_init()
    output = layer.runner.forward_impl(
        layer, hidden_states, router_logits, shared_experts_input
    )
    return _stabilize_moe_output_for_cudagraph(layer, output)


def _moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> torch.Tensor:
    # fake impl 鍙敤浜?shape / tracing 鎺ㄥ锛屼笉鍙備笌鐪熷疄璁＄畻銆?
    return torch.empty_like(hidden_states)


def _moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> tuple[torch.Tensor, torch.Tensor]:
    # shared-experts 鐗堟湰浼氳繑鍥炰袱涓緭鍑猴細shared 鍒嗘敮鍜?routed experts 鍒嗘敮銆?
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    # TODO(bnell): 绛?MK 杩佺Щ瀹屾垚鍚庯紝杩欓噷鐨勫吋瀹瑰垵濮嬪寲鍙垹闄ゃ€?
    layer.ensure_moe_quant_config_init()
    output = layer.runner.forward_impl(
        layer, hidden_states, router_logits, shared_experts_input
    )
    return _stabilize_moe_output_for_cudagraph(layer, output)


def _moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> tuple[torch.Tensor, torch.Tensor]:
    # fake impl 鍚屾牱瑕佷繚鎸佺湡瀹炵畻瀛愮殑杩斿洖缁撴瀯涓€鑷淬€?
    # 杈撳嚭褰㈢姸绾﹀畾濡備笅锛?
    # - fused_out 涓?hidden_states 鍚屽舰锛況outed experts 鑻ュ仛杩囧彉鎹紝鍒欎娇鐢ㄥ彉鎹㈠悗鐨勭淮搴︺€?
    # - shared_out 鑻ユ彁渚涗簡 shared_experts_input锛屽垯涓庡畠鍚屽舰锛涘惁鍒欎笌 hidden_states 鍚屽舰銆?
    # - latent MoE 涓嬶紝shared experts 浠嶄娇鐢ㄥ師濮?hidden_size锛岃€岄潪 latent size銆?
    fused_out = torch.empty_like(hidden_states)
    if shared_experts_input is not None:
        shared_out = torch.empty_like(shared_experts_input)
    else:
        shared_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


def _moe_prepare_tiered(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: _layer_name_type,
) -> torch.Tensor:
    # PIECE split boundary: dynamic tiered-cache prepare/H2D/scatter stays
    # eager, while the following compute op can still be captured.
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    controller = getattr(layer, "_cfie_tiered_cache_controller", None)
    if controller is not None:
        setattr(controller, "_active_prefill_burst_pool", None)
        unique_requested = int(torch.unique(topk_ids.detach()).numel())
        num_tokens = int(topk_ids.shape[0])
        if (
            unique_requested > int(getattr(layer, "local_num_experts", 0))
            and controller.can_run_prefill_burst(unique_requested, num_tokens)
        ):
            pool = controller.prefill_burst_pool
            if pool is None:
                raise RuntimeError(
                    f"{controller.layer_key}: prefill burst pool is not available"
                )
            pool.prepare(controller, topk_ids)
            setattr(controller, "_active_prefill_burst_pool", pool)
        else:
            layer.runner._prepare_tiered_controller(
                controller,
                topk_ids,
                topk_weights,
                router_logits,
            )
    return topk_ids


def _moe_prepare_tiered_fake(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: _layer_name_type,
) -> torch.Tensor:
    return topk_ids


def _moe_compute(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> torch.Tensor:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    layer.ensure_moe_quant_config_init()
    quant_method = layer.runner.quant_method
    controller = getattr(layer, "_cfie_tiered_cache_controller", None)
    burst_pool = (
        getattr(controller, "_active_prefill_burst_pool", None)
        if controller is not None
        else None
    )
    execution_layer = (
        burst_pool._execution_layer if burst_pool is not None else layer
    )
    moe_config = getattr(quant_method, "moe", None)
    old_disable_inplace = getattr(moe_config, "disable_inplace", None)
    if old_disable_inplace is not None:
        moe_config.disable_inplace = True
    try:
        try:
            return quant_method.apply(
                layer=execution_layer,
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
            )
        finally:
            if burst_pool is not None:
                burst_pool.release(record_use=True)
                setattr(controller, "_active_prefill_burst_pool", None)
    finally:
        if old_disable_inplace is not None:
            moe_config.disable_inplace = old_disable_inplace


def _moe_compute_fake(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _moe_compute_shared(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    layer.ensure_moe_quant_config_init()
    quant_method = layer.runner.quant_method
    controller = getattr(layer, "_cfie_tiered_cache_controller", None)
    burst_pool = (
        getattr(controller, "_active_prefill_burst_pool", None)
        if controller is not None
        else None
    )
    execution_layer = (
        burst_pool._execution_layer if burst_pool is not None else layer
    )
    moe_config = getattr(quant_method, "moe", None)
    old_disable_inplace = getattr(moe_config, "disable_inplace", None)
    if old_disable_inplace is not None:
        moe_config.disable_inplace = True
    try:
        try:
            return quant_method.apply(
                layer=execution_layer,
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
            )
        finally:
            if burst_pool is not None:
                burst_pool.release(record_use=True)
                setattr(controller, "_active_prefill_burst_pool", None)
    finally:
        if old_disable_inplace is not None:
            moe_config.disable_inplace = old_disable_inplace


def _moe_compute_shared_fake(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    layer_name: _layer_name_type,
) -> tuple[torch.Tensor, torch.Tensor]:
    fused_out = torch.empty_like(hidden_states)
    if shared_experts_input is not None:
        shared_out = torch.empty_like(shared_experts_input)
    else:
        shared_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


direct_register_custom_op(
    op_name="moe_prepare_tiered",
    op_func=_moe_prepare_tiered,
    mutates_args=[],
    fake_impl=_moe_prepare_tiered_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


direct_register_custom_op(
    op_name="moe_compute",
    op_func=_moe_compute,
    mutates_args=[],
    fake_impl=_moe_compute_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


direct_register_custom_op(
    op_name="moe_compute_shared",
    op_func=_moe_compute_shared,
    mutates_args=[],
    fake_impl=_moe_compute_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


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
    FusedMoE 鐨勯粯璁ゆ墽琛岀紪鎺掑櫒銆?

    杩欎釜绫绘湰韬笉鎸佹湁涓撳鏉冮噸锛涙潈閲嶃€乲ernel 閰嶇疆鍜屽苟琛屽厓鏁版嵁浠嶅睘浜庡灞?
    `FusedMoE` / `SharedFusedMoE` layer銆俙DefaultMoERunner` 鐨勮亴璐ｆ洿鍍?
    鈥滆繍琛屾椂璋冨害鍣ㄢ€濓細

    - 鎺ユ敹涓婂眰浼犲叆鐨?`hidden_states`銆乣router_logits`
    - 瑙嗛厤缃喅瀹?gate銆乺outer銆乻hared experts 鐨勬墽琛岄『搴?
    - 鍦ㄩ渶瑕佹椂鎺ュ叆 CFIE tiered cache銆丏P chunking銆佺嫭绔?shared stream
    - 璋冪敤 quant method / monolithic kernel 瀹屾垚 routed experts 涓昏绠?
    - 鍦ㄦ湯灏惧畬鎴?dispatch/combine銆佽绾︿笌杈撳嚭瑁佸壀

    鍥犺€屽畠缁熶竴瑕嗙洊浜?routed experts銆乻hared experts銆乀P/EP/SP 骞惰銆?
    monolithic / decomposed kernel锛屼互鍙?CFIE 鐗规湁 tiered cache 鐨勭粍鍚堣矾寰勩€?
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
        # 鍏堝垵濮嬪寲鎶借薄鍩虹被 `MoERunner`銆?
        # 杩欓噷涓昏鏄鐖剁被鏈夋満浼氬缓绔嬭嚜宸辩殑鍩虹鐘舵€併€?
        super().__init__()

        # 淇濆瓨褰撳墠灞傚搴旂殑 MoE 闈欐€侀厤缃紝渚嬪 hidden_dim銆乼p/ep/sp 鎷撴墤绛夈€?
        self.moe_config = moe_config
        # 淇濆瓨 router 瀵硅薄锛涘悗缁潪 monolithic 璺緞浼氱敤瀹冩妸 router logits 杞垚 top-k experts銆?
        self.router = router
        # 淇濆瓨 routed experts 鐨勫彲閫夎緭鍏ュ彉鎹㈡ā鍧楋紝渚嬪 latent MoE 鐨勫墠缃姇褰便€?
        self.routed_input_transform = routed_input_transform
        # 淇濆瓨鍙€夌殑 gate 妯″潡锛涜嫢涓嶄负绌猴紝runner 浼氬湪鍐呴儴鍏堣皟鐢ㄥ畠浜у嚭 router logits銆?
        self.gate = gate
        # 淇濆瓨 shared experts 鍒嗘敮妯″潡锛涜嫢涓虹┖锛岃鏄庡綋鍓嶅眰娌℃湁 shared expert銆?
        self.shared_experts = shared_experts
        # 淇濆瓨搴曞眰閲忓寲 / kernel 鎵ц鍏ュ彛锛涚湡姝ｇ殑涓撳璁＄畻鏈€缁堥兘浼氳惤鍒拌繖閲屻€?
        self.quant_method = quant_method
        # 璁板綍褰撳墠灞傛槸鍚﹀笇鏈涘湪鏇村灞傝繑鍥炲墠瀹屾垚瑙勭害銆?
        self.reduce_results = reduce_results
        # 璁板綍鏄惁鍚敤 DBO锛涘畠浼氬奖鍝?DP chunking staging buffer 鐨勫竷灞€銆?
        self.enable_dbo = enable_dbo

        # -----------------
        # shared experts 鐙珛 stream 鐩稿叧鍒濆鍖栥€?
        # -----------------
        # 鍑轰簬璋冭瘯鐩殑锛屽厑璁搁€氳繃鐜鍙橀噺绂佺敤 shared experts 鐨勭嫭绔?stream銆?
        # TODO: 绛?TP / DP 涓庡叾浠栨墽琛屾ā寮忛獙璇佹洿鍏呭垎鍚庯紝鍙Щ闄よ繖鏉¤皟璇曞紑鍏炽€?
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            # 鑻ョ幆澧冨彉閲忔樉寮忕鐢紝灏卞畬鍏ㄤ笉涓?shared experts 鍗曠嫭鍒嗛厤杈呭姪 stream銆?
            logger.debug_once("Disabling MoE shared_experts cuda stream", scope="local")
            # 璁颁负 None锛屽悗缁垎鏀細鑷劧閫€鍖栦负鈥滀笌涓绘祦涓茶鎵ц鈥濄€?
            self.shared_experts_stream = None
        else:
            # TODO(rob): 涓洪潪 cuda-alike 骞冲彴琛ヤ笂 shared expert overlap 鏀寔銆?
            # 闈?cuda-alike 骞冲彴涓婏紝aux_stream() 浼氱洿鎺ヨ繑鍥?None銆?
            # 灏濊瘯鐢宠涓€鏉¤緟鍔?stream锛屼緵 shared experts 涓?routed experts 骞惰鎵ц銆?
            self.shared_experts_stream = aux_stream()
            if self.shared_experts_stream is not None:
                # 鍙湁鐪熸鎷垮埌杈呭姪 stream 鏃舵墠鎵撳嵃鍚敤鏃ュ織銆?
                logger.debug_once(
                    "Enabled separate cuda stream for MoE shared_experts", scope="local"
                )

        # 璁板綍灞傚悕锛屼緵 custom op 鍙嶆煡鐪熷疄灞傚璞°€?
        self.layer_name = layer.layer_name
        self.layer = layer

        # 鍦?TPU / CPU 涓婄洿鎺ョ粦瀹?Python 鍑芥暟锛涘湪 CUDA 璺緞涓婁紭鍏堣蛋娉ㄥ唽杩囩殑 custom op銆?
        if current_platform.is_tpu() or current_platform.is_cpu():
            # TODO: TPU 鍚庣鐨?OOM 闂瑙ｅ喅鍚庯紝鍐嶅垏鍥?moe_forward custom op銆?
            # CPU 璺緞涓嶉渶瑕侀澶栧寘涓€灞?forward_impl銆?
            if self.shared_experts is None:
                # 娌℃湁 shared experts 鏃讹紝缁戝畾鍙繑鍥?routed 杈撳嚭鐨勫叆鍙ｃ€?
                self.moe_forward = _moe_forward
            else:
                # 鏈?shared experts 鏃讹紝缁戝畾浼氳繑鍥?`(shared, routed)` 鐨勫叆鍙ｃ€?
                self.moe_forward = _moe_forward_shared
        else:
            if self.shared_experts is None:
                # CUDA/鑷畾涔夌畻瀛愯矾寰勪笅锛屼紭鍏堣蛋娉ㄥ唽濂界殑 `torch.ops.cfie.moe_forward`銆?
                self.moe_forward = torch.ops.cfie.moe_forward
            else:
                # shared 鐗堟湰鐨?custom op 浼氬湪鍐呴儴杞彂鍒?runner 鐨?shared 璺緞銆?
                self.moe_forward = torch.ops.cfie.moe_forward_shared

        # DP chunking 鍦烘櫙涓嬩細澶嶇敤杩欎袱鍧?staging buffer锛岄伩鍏嶆瘡涓?chunk 閲嶆柊鍒嗛厤鏄惧瓨銆?
        # 杩欓噷鍏堢疆绌猴紝绛夌涓€娆＄湡姝ｉ渶瑕?chunking 鏃跺啀鎳掑垵濮嬪寲銆?
        self.batched_hidden_states: torch.Tensor | None = None
        # router logits 涔熷搴旂淮鎶や竴鍧?staging buffer锛屼笌 hidden_states 鍚屾澶嶇敤銆?        self.batched_router_logits: torch.Tensor | None = None

    def _apply_with_tiered_cache(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
        router_logits: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # -----------------
        # decomposed/top-k 璺緞涓嬬殑 tiered cache 妗ユ帴銆?
        # -----------------
        # 杩欐潯璺緞瀵瑰簲鈥渞outer 鍏堥€夊嚭 top-k experts锛屽啀璋冪敤 quant_method.apply(...)鈥?
        # 鐨勬墽琛屾柟寮忋€傝嫢褰撳墠灞傛湭鎸?tiered cache controller锛屽氨鐩存帴鎶?top-k 缁撴灉浜ょ粰
        # quant_method锛涘惁鍒欏厛鏍规嵁鏈壒 token 瑙﹁揪鐨?experts 闆嗗悎鍐冲畾锛?
        # - 鐩存帴 prepare 鍚庢墽琛?
        # - 璧?burst 鎵ц鍖?
        # - 鎴栨媶鎴愬涓?token chunk 鍒嗘壒鎵ц
        # 杩欐槸 CFIE tiered cache 鐨勬墽琛屾ˉ鎺ュ眰锛?
        # 鑻ュ綋鍓嶅眰娌℃寕 controller锛屽氨鐩存帴璧板師濮?quant_method.apply锛?
        # 鍚﹀垯鍏堢‘淇濇湰娆¤姹傛秹鍙婄殑 experts 宸茬粡鍦?GPU resident slots 涓€?
        # 灏濊瘯浠庡眰瀵硅薄涓婃嬁鍒?CFIE tiered cache controller銆?
        controller = getattr(layer, "_cfie_tiered_cache_controller", None)
        if controller is None:
            # 鑻ュ綋鍓嶅眰娌℃湁 tiered cache锛屽氨鐩存帴鎶?top-k 璺敱缁撴灉浜ょ粰 quant method 鎵ц銆?
            return self.quant_method.apply(
                layer=layer,
                x=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
            )

        # 鍩哄噯娴嬭瘯璁℃椂锛氬彈 CFIE_BENCH_TIMING 鐜鍙橀噺鎺у埗锛屼粎鍦ㄦ帓鏌ユ椂寮€鍚€?
        _bench_timing = os.getenv("CFIE_BENCH_TIMING", "") == "1"
        if router_logits is None:
            max_expert_id = int(topk_ids.max().item()) if topk_ids.numel() else 0
            num_experts = max(
                max_expert_id + 1,
                int(getattr(layer, "global_num_experts", 0) or 0),
                int(getattr(layer, "num_experts", 0) or 0),
                1,
            )
            router_logits = torch.zeros(
                (int(topk_ids.shape[0]), num_experts),
                dtype=torch.float32,
                device=topk_ids.device,
            )

        # -----------------
        # 鍗曚釜 chunk 鐨勬墽琛岄€昏緫銆?
        # -----------------
        def apply_chunk(
            chunk_x: torch.Tensor,
            chunk_router_logits: torch.Tensor,
            chunk_topk_weights: torch.Tensor,
            chunk_topk_ids: torch.Tensor,
            chunk_shared_input: torch.Tensor | None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            # 缁熻褰撳墠 chunk 涓€鍏辫Е杈惧灏戜釜鍞竴 experts銆?
            # `detach()` 鍙槸涓轰簡閬垮厤杩欑被缁熻閫昏緫鎰忓杩涘叆 autograd 鍥俱€?
            unique_requested = int(torch.unique(chunk_topk_ids.detach()).numel())
            # 褰撳墠 chunk 鐨?token 鏁颁篃浼氬奖鍝?burst 绛栫暐鍒ゆ柇銆?
            num_chunk_tokens = int(chunk_topk_ids.shape[0])
            if unique_requested <= layer.local_num_experts:
                # resident slots fit this chunk; prepare loads missing experts.
                if _bench_timing:
                    _t0 = _time.perf_counter()
                self._prepare_tiered_controller(
                    controller,
                    chunk_topk_ids,
                    chunk_topk_weights,
                    chunk_router_logits,
                )
                if _bench_timing:
                    _t1 = _time.perf_counter()
                    torch.cuda.synchronize()
                    _t1_sync = _time.perf_counter()
                result = self.quant_method.apply(
                    layer=layer,
                    x=chunk_x,
                    topk_weights=chunk_topk_weights,
                    topk_ids=chunk_topk_ids,
                    shared_experts_input=chunk_shared_input,
                )
                if _bench_timing:
                    _t2 = _time.perf_counter()
                    torch.cuda.synchronize()
                    _t2_sync = _time.perf_counter()
                    _layer_key = getattr(controller, "layer_key", "?")
                    logger.info(
                        "CFIE_BENCH_TIMING layer=%s prepare=%.3fs prepare_sync=%.3fs "
                        "apply=%.3fs apply_sync=%.3fs requested=%d",
                        _layer_key,
                        _t1 - _t0, _t1_sync - _t0,
                        _t2 - _t1, _t2_sync - _t0,
                        unique_requested,
                    )
                return result

            if controller.can_run_prefill_burst(unique_requested, num_chunk_tokens):
                # 鑻ヤ富 resident slots 瑁呬笉涓嬶紝浣?burst pool 鑳藉厹浣忥紝灏辫蛋涓存椂 burst 鎵ц鍖恒€?
                # 杩欐潯璺緞涓嶈姹傛妸 experts 鍏ㄩ儴鎹㈠叆甯搁┗妲戒綅锛岃€屾槸鍊熷姪 burst 鍖轰复鏃舵墽琛屻€?
                return controller.run_prefill_burst(
                    x=chunk_x,
                    topk_weights=chunk_topk_weights,
                    topk_ids=chunk_topk_ids,
                    shared_experts_input=chunk_shared_input,
                )

            # 鏃㈣涓嶈繘 resident锛屼篃涓嶈兘鐢?burst锛屽氨鍙兘鎶ュ閲忛敊璇€?
            raise RuntimeError(
                f"tiered-cache chunk requires {unique_requested} experts but no "
                "execution path can handle that capacity"
            )

        # -----------------
        # 鍏堝垽鏂€滄暣鍧楄緭鍏モ€濊兘鍚︾洿鎺ヨ窇銆?
        # -----------------
        # 缁熻鏁村潡杈撳叆涓€鍏变細鍛戒腑澶氬皯涓笉鍚?expert銆?
        full_unique_requested = int(torch.unique(topk_ids.detach()).numel())
        # 缁熻鏁村潡杈撳叆鐨?token 鎬绘暟銆?
        full_num_tokens = int(topk_ids.shape[0])
        # 棰勫厛鍒ゆ柇鏁村潡杈撳叆鏄惁鍙互鏁翠綋璧?burst 璺緞銆?
        full_can_use_burst = controller.can_run_prefill_burst(
            full_unique_requested,
            full_num_tokens,
        )
        if (
            full_unique_requested <= layer.local_num_experts
            or full_can_use_burst
        ):
            # 鏁村潡鍙洿鎺ユ墽琛屾椂锛屼笉鍐嶉澶栧垏 token chunk銆?
            # 杩欐牱鍙互閬垮厤涓嶅繀瑕佺殑 token 缁存媶鍒嗗拰鍚庣画鍐嶆嫾鎺ャ€?
            return apply_chunk(
                x,
                router_logits,
                topk_weights,
                topk_ids,
                shared_experts_input,
            )

        raise RuntimeError(
            f"tiered-cache chunk requires {full_unique_requested} experts, "
            f"resident slots={layer.local_num_experts}, burst capacity="
            f"{getattr(controller, 'prefill_burst_capacity', 0)}. Increase "
            "--prefill-burst-slots or reduce max_num_batched_tokens."
        )

    def _use_split_tiered_for_piecewise(self) -> bool:
        if current_platform.is_tpu() or current_platform.is_cpu():
            return False
        if self.routed_input_transform is not None:
            return False
        controller = getattr(self.layer, "_cfie_tiered_cache_controller", None)
        if controller is None:
            return False
        # The shared prefill burst pool is intentionally reused across layers.
        # Split prepare/compute can hold that pool across custom-op boundaries,
        # while PIECE partitioning may advance to the next layer before compute
        # releases it. Keep burst mode as a single eager MoE boundary instead.
        if getattr(controller, "prefill_burst_pool", None) is not None:
            return False
        try:
            from cfie.config import get_current_cfie_config_or_none

            cfie_config = get_current_cfie_config_or_none()
        except Exception:
            cfie_config = None
        return bool(
            cfie_config is not None
            and cfie_config.compilation_config.allow_tiered_moe_compile
        )

    def _forward_split_tiered_for_piecewise(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.quant_method.is_monolithic:
            return self._apply_monolithic_with_tiered_cache(
                layer=self.layer,
                x=hidden_states,
                router_logits=router_logits,
            )

        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)

        topk_weights, topk_ids = self.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        layer_name = ModuleName(self.layer_name) if HAS_OPAQUE_TYPE else self.layer_name
        prepared_topk_ids = torch.ops.cfie.moe_prepare_tiered(
            topk_ids,
            topk_weights,
            router_logits,
            layer_name,
        )
        if self.shared_experts is None:
            return torch.ops.cfie.moe_compute(
                hidden_states,
                topk_weights,
                prepared_topk_ids,
                shared_experts_input,
                layer_name,
            )
        if not self.quant_method.mk_owns_shared_expert:
            shared_input = (
                shared_experts_input
                if shared_experts_input is not None
                else hidden_states
            )
            shared_output = self.shared_experts(shared_input)
            routed_output = torch.ops.cfie.moe_compute(
                hidden_states,
                topk_weights,
                prepared_topk_ids,
                shared_experts_input,
                layer_name,
            )
            return shared_output, routed_output
        return torch.ops.cfie.moe_compute_shared(
            hidden_states,
            topk_weights,
            prepared_topk_ids,
            shared_experts_input,
            layer_name,
        )

    @staticmethod
    def _prepare_tiered_controller(
        controller: object,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor | None,
        router_logits: torch.Tensor | None,
    ) -> None:
        router_probs = None
        if router_logits is not None:
            router_probs = torch.softmax(
                router_logits.detach().to(dtype=torch.float32),
                dim=-1,
            )

        prepare = controller.prepare
        try:
            signature = inspect.signature(prepare)
        except (TypeError, ValueError):
            kwargs: dict[str, object] = {}
            if topk_weights is not None:
                kwargs["topk_weights"] = topk_weights
            if router_probs is not None:
                kwargs["router_probs"] = router_probs
            prepare(topk_ids, **kwargs)
            return

        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        kwargs: dict[str, object] = {}
        if topk_weights is not None and (
                accepts_var_kwargs or "topk_weights" in signature.parameters
        ):
            kwargs["topk_weights"] = topk_weights
        if router_probs is not None and (
                accepts_var_kwargs or "router_probs" in signature.parameters
        ):
            kwargs["router_probs"] = router_probs
        prepare(topk_ids, **kwargs)

    def _apply_monolithic_with_tiered_cache(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # -----------------
        # monolithic kernel 璺緞涓嬬殑 tiered cache 妗ユ帴銆?
        # -----------------
        # monolithic kernel 浼氭妸鈥渞outer + experts鈥濇墦鎴愪竴涓暣浣撴墽琛岋紝鍥犳姝ｅ紡璁＄畻鏃?
        # 涓嶄細鏄惧紡缁忚繃 top-k prepare/finalize 娴佺▼锛涗絾 tiered cache 浠嶇劧闇€瑕佷簨鍏堢煡閬?
        # 褰撳墠 batch 浼氳Е杈惧摢浜?experts锛屾墠鑳藉喅瀹氭槸鍚?prepare / 鍒?chunk銆?
        # monolithic 璺緞鍚屾牱鍏堝皾璇曞彇寰?tiered cache controller銆?
        controller = getattr(layer, "_cfie_tiered_cache_controller", None)
        if controller is None:
            # 鑻ユ病鏈?tiered cache锛屽氨鐩存帴璋冪敤 monolithic kernel銆?
            return self.quant_method.apply_monolithic(
                layer=layer,
                x=x,
                router_logits=router_logits,
            )

        # monolithic kernel 铏界劧鎶?router + experts 鍚堝湪涓€璧锋墽琛岋紝
        # 浣嗕负浜嗗垽鏂?tiered cache 瀹归噺锛屼粛闇€鍏堝湪 Python 渚у仛涓€娆?expert 閫夋嫨銆?
        topk_weights, topk_ids = self.router.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )

        # 缁熻鏁村潡杈撳叆浼氳Е杈惧灏戜釜鍞竴 experts銆?
        full_unique_requested = int(torch.unique(topk_ids.detach()).numel())
        if full_unique_requested <= layer.local_num_experts:
            # 鑻ュ父椹绘Ы浣嶈冻澶燂紝灏卞厛 prepare 鍐嶆暣浣撴墽琛?monolithic kernel銆?
            self._prepare_tiered_controller(
                controller,
                topk_ids,
                topk_weights,
                router_logits,
            )
            return self.quant_method.apply_monolithic(
                layer=layer,
                x=x,
                router_logits=router_logits,
            )

        # 鑻ユ暣鍧楄緭鍏ヨ秴鍑哄閲忥紝灏辨寜 expert 瀹归噺閲嶆柊鍒?token 鍖洪棿銆?
        token_ranges = _pack_token_ranges_by_expert_capacity(
            topk_ids,
            layer.local_num_experts,
        )

        # 閫愪釜 token chunk 璋?monolithic kernel锛屽苟鏀堕泦鍚?chunk 杈撳嚭銆?
        outputs: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        for start, end in token_ranges:
            # 姣忎釜 chunk 鎵ц鍓嶉兘瑕佽 controller 鍏堟妸杩欎竴娈甸渶瑕佺殑 experts prepare 鍒颁綅銆?
            self._prepare_tiered_controller(
                controller,
                topk_ids[start:end],
                topk_weights[start:end],
                router_logits[start:end],
            )
            outputs.append(
                self.quant_method.apply_monolithic(
                    layer=layer,
                    x=x[start:end],
                    router_logits=router_logits[start:end],
                )
            )

        first_output = outputs[0]
        if isinstance(first_output, tuple):
            # shared/routed 浜屽厓缁勮緭鍑烘椂锛屼袱涓垎鏀垎鍒部 token 缁存嫾鎺ャ€?
            return (
                torch.cat([output[0] for output in outputs], dim=0),
                torch.cat([output[1] for output in outputs], dim=0),
            )
        # 绾?routed 杈撳嚭鍒欑洿鎺ユ嫾鎺ャ€?
        return torch.cat(outputs, dim=0)

    @property
    def use_dp_chunking(self) -> bool:
        # 杩斿洖褰撳墠 backend 鏄惁搴旇鍚敤 DP chunking銆?
        # 鍙湁閮ㄥ垎 all2all kernel 鏀寔 / 鍙楃泭浜庢妸澶?batch 鍒囧潡鎵ц锛屽苟涓旇繕瑕佸彈
        # 鐜鍙橀噺寮€鍏虫帶鍒躲€?
        # 杩欓噷鍓嶅崐閮ㄥ垎鍒ゆ柇鈥滃綋鍓?backend 鑳戒笉鑳?/ 瑕佷笉瑕?chunk鈥濓紝
        # 鏈€鍚庣殑鐜鍙橀噺鍒欐槸浜哄伐鎬诲紑鍏炽€?
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
        # 鍒ゆ柇 shared experts 鏄惁瑕佹斁鍒扮嫭绔?CUDA stream 涓婂拰 routed experts 骞惰銆?
        # 杩斿洖鍊硷細
        # - `use_shared_experts_stream`锛氭湰娆?forward 鏄惁鐪熺殑鍚敤鐙珛 stream
        # - `shared_experts_input`锛氳嫢鍚敤鐙珛 stream锛岄渶瑕佹彁鍓嶅浐瀹氬ソ瀹冭娑堣垂鐨勮緭鍏?
        # 鍐冲畾鏄惁璁?shared experts 鍦ㄧ嫭绔?CUDA stream 涓婂拰 routed experts 骞惰鎵ц銆?
        # 鍙湁婊¤冻浠ヤ笅鏉′欢鏃讹紝shared experts 鎵嶅€煎緱鏀惧埌鐙珛 CUDA stream锛?
        # 1. 褰撳墠纭疄鏄?CUDA 骞冲彴锛?
        # 2. shared experts 娌¤ monolithic kernel 鎺ョ锛?
        # 3. 褰撳墠涓嶈蛋 chunked 璺緞锛?
        # 4. 鎴愬姛鐢宠鍒颁簡杈呭姪 stream锛?
        # 5. token 鏁版病鏈夎秴杩囩幆澧冨彉閲忕粰瀹氶槇鍊笺€?
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

        # 榛樿鍏堣涓?None锛涘彧鏈夌湡鐨勫惎鐢ㄧ嫭绔?stream 鏃舵墠鍑嗗涓撻棬鐨勮緭鍏ュ紶閲忓紩鐢ㄣ€?
        shared_experts_input: torch.Tensor | None = None
        if use_shared_experts_stream:
            # 鏃㈢劧鍐冲畾鍚敤鐙珛 stream锛岃繖閲屽氨瑕佹眰杈呭姪 stream 蹇呴』瀛樺湪銆?
            assert self.shared_experts_stream is not None
            # shared experts 鍜?routed experts 骞惰鏃讹紝涓昏矾寰勪笉鑳藉師鍦拌鍐欒緭鍏ャ€?
            assert self.moe_config.disable_inplace

            # 鑻ヨ皟鐢ㄦ柟宸叉樉寮忕粰浜?shared_input锛屽氨娌跨敤瀹冿紱
            # 鍚﹀垯 shared experts 榛樿鐩存帴娑堣垂褰撳墠鐨?hidden_states銆?
            shared_experts_input = (
                shared_input if shared_input is not None else hidden_states
            )

            # 鏍囪 shared_experts_input 浼氬湪鍙︿竴鏉?stream 涓婅娑堣垂锛岄伩鍏嶅紶閲忚繃鏃╅噴鏀俱€?
            # 杩欓噷涓嶉渶瑕佸 shared_output 鍐嶉澶栧仛 record_stream锛?
            # 鍥犱负鍚庨潰鍦ㄤ娇鐢?shared_output 鍓嶄細鍏堝悓姝ヤ袱鏉?stream銆?
            shared_experts_input.record_stream(self.shared_experts_stream)

            # 鍦ㄨ繖閲岃褰曠嫭绔?shared experts stream 鐨勫悓姝ヨ捣鐐癸紝
            # 璁╁畠鑳藉涓庝笅闈㈢殑 router / gate 璺緞骞惰鎵ц銆?
            assert self.shared_experts_stream is not None
            self.shared_experts_stream.wait_stream(current_stream())

        return use_shared_experts_stream, shared_experts_input

    def ensure_dp_chunking_init(self):
        # 鎸夐渶鍒嗛厤 DP chunking 澶嶇敤缂撳啿鍖恒€?
        # 杩欎簺 staging tensor 浼氬湪涓€娆?forward 鐨勫涓?chunk 涔嬮棿鍙嶅澶嶇敤锛?
        # 閬垮厤姣忎釜 chunk 閮介噸鏂扮敵璇锋樉瀛樸€?
        # 鑻ュ綋鍓?backend 涓嶅惎鐢?DP chunking锛屾垨宸茬粡鍒濆鍖栬繃 staging buffer锛屽氨鐩存帴杩斿洖銆?
        if not self.use_dp_chunking or self.batched_hidden_states is not None:
            return

        # 鍏堝０鏄庡ソ hidden states / router logits 涓ゅ潡 staging buffer 鐨勫舰鐘跺彉閲忋€?
        states_shape: tuple[int, ...]
        logits_shape: tuple[int, ...]

        # 绠€鍐欏綋鍓?MoE 閰嶇疆锛屽悗闈㈠澶勯兘浼氱敤鍒般€?
        moe = self.moe_config

        if self.enable_dbo:
            # DBO 鎵撳紑鏃讹紝澶氶鐣欎竴涓?ubatch 缁村害銆?
            states_shape = (2, moe.max_num_tokens, self.moe_config.hidden_dim)
            logits_shape = (2, moe.max_num_tokens, self.moe_config.num_logical_experts)
        else:
            # 闈?DBO 妯″紡涓嬶紝staging buffer 鍙湁 `[tokens, hidden/logits]` 涓ょ淮銆?
            states_shape = (moe.max_num_tokens, self.moe_config.hidden_dim)
            logits_shape = (moe.max_num_tokens, self.moe_config.num_logical_experts)

        # 鍦ㄥ綋鍓嶈澶囦笂涓€娆℃€у垎閰?chunk staging buffer銆?
        # 鍚庣画鎵€鏈?chunk 閮戒細鍙嶅鍐欏叆杩欎袱鍧楀紶閲忋€?
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
        # 杩欎釜鍘嗗彶鎺ュ彛鍚嶅鏄撹瑙ｃ€?
        # 褰撳墠瀹炵幇閲岋紝杩斿洖 True 鏇存帴杩戔€渟hared expert 杈撳嚭宸茬粡婊¤冻鏈€缁堣绾﹁姹傦紝
        # 璋冪敤鏂规棤闇€鍐嶉澶?all-reduce鈥濓紱杩斿洖 False 鎵嶈〃绀哄悗闈㈣繕瑕佽ˉ涓€娆?TP 瑙勭害銆?
        # shared experts 涓€鑸敱 RowParallelLinear 璁＄畻銆?
        # 绾?TP 鍦烘櫙涓嬪彲寤跺悗鍒?MoE 鏈熬鍐嶈绾︼紱
        # 浣?EP + all2all 鍦烘櫙涓嬶紝鍚?DP rank 浼氭寔鏈夊畬鏁?hidden_states锛?
        # 鍥犳闇€瑕佸敖鏃╄绾?shared experts 杈撳嚭銆?
        # 杩欓噷浠嶈姹?quant_method 蹇呴』瀛樺湪锛屽洜涓哄垽鏂€昏緫渚濊禆搴曞眰 kernel 鑳藉姏銆?
        assert self.quant_method is not None
        # 鏌愪簺 kernel 宸茬粡鍦ㄥ唴閮ㄥ畬鎴愯绾︼紝杩欐椂 shared experts 杈撳嚭鏃犻渶鍐嶉澶?reduce銆?
        return (
            # `moe_kernel is not None` 琛ㄧず褰撳墠 quant method 宸茬粡鎸備笂鍏蜂綋鍐呮牳瀹炵幇銆?
            self.quant_method.moe_kernel is not None
            # 鑻ュ唴鏍稿０鏄庤緭鍑哄凡缁忚绾﹀畬鎴愶紝鍒欒繖閲岃繑鍥?True锛屽灞傛棤闇€鍐嶈ˉ all-reduce銆?
            and self.quant_method.moe_kernel.output_is_reduced()
        )

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        # 瀵瑰鎻愪緵涓€涓€滄寜闇€瑙勭害鈥濈殑缁熶竴鍏ュ彛銆?
        # 鑻ュ簳灞?kernel 宸茬粡淇濊瘉杈撳嚭婊¤冻鏈€缁堣绾﹁姹傦紝灏卞師鏍疯繑鍥烇紱
        # 鍚﹀垯杩欓噷琛ヤ竴娆?TP all-reduce銆?
        if self.must_reduce_shared_expert_outputs():
            # 鑻ュ簳灞傚凡缁忚绾﹀畬锛屽氨鐩存帴鎶婅緭鍏ョ粨鏋滃悜澶栭€忎紶銆?
            return final_hidden_states
        else:
            # 鍚﹀垯鏄惧紡鍦?TP group 涓婂仛涓€娆?all-reduce銆?
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def apply_routed_input_transform(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 瀵?routed experts 杈撳叆鍋氬彲閫夌殑鍓嶇疆鍙樻崲銆?
        # 甯歌鍦烘櫙鏄?latent MoE锛歴hared experts 缁х画娑堣垂鍘熷 hidden_states锛?
        # 浣?routed experts 鍏堟姇褰卞埌鏇村皬鐨?latent 缁村害鍐嶈繘鍏?MoE kernel銆?
        # 浠呭 routed experts 杈撳叆鍋氶澶栧彉鎹紝渚嬪 latent projection銆?
        # FusedMoE.forward_native 浼氫繚鐣欏師濮?hidden_states 缁?shared experts锛?
        # 鑰?routed experts 鍒欎娇鐢ㄥ彉鎹㈠悗鐨?[S, moe_latent_size] 杈撳叆銆?
        # TODO: 涓轰簡杩涗竴姝ラ檷浣?latent MoE 鐨勫甫瀹藉紑閿€锛宖c2_latent_proj 鏈潵鍙€冭檻
        # 涓嬫矇鍒?SharedFusedMoE 鍐呴儴锛屽湪鏇村皬鐨?latent 缁村害涓婂仛 all-reduce銆?
        # routed experts 鍜?shared experts 鐨勮緭鍏ョ淮鍙兘涓嶅悓锛屽洜姝?routed 鍒嗘敮鍙崟鐙彉鎹€?
        if self.routed_input_transform is not None:
            # 鑻ラ厤缃簡 routed_input_transform锛屽氨鍏堝 routed 鍒嗘敮杈撳叆鍋氫竴娆″彉鎹€?
            result = self.routed_input_transform(hidden_states)
            # ReplicatedLinear 浼氳繑鍥?(output, extra_bias)锛?
            # 杩欓噷鍙渶瑕佺湡姝ｇ殑杈撳嚭寮犻噺銆?
            if isinstance(result, tuple):
                return result[0]
            # 鏅€氭ā鍧楃洿鎺ヨ繑鍥炲崟涓紶閲忔椂锛屽師鏍烽€忎紶銆?
            return result
        # 鑻ユ湭閰嶇疆棰濆鍙樻崲锛屽氨鐩存帴浣跨敤鍘熷 hidden_states銆?
        return hidden_states

    def _reduce_output(
        self,
        states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        trunc_sizes: list[int],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 缁熶竴澶勭悊杈撳嚭鍚庢敹灏鹃€昏緫銆?
        # 杩欓噷鍚屾椂璐熻矗锛?
        # - 蹇呰鏃舵墽琛?TP all-reduce
        # - 鎶婁负浜嗛€傞厤 kernel 鑰?pad 鐨?hidden_dim 瑁佸洖鐪熷疄缁村害
        # - 鍚屾椂鍏煎鈥滀粎 routed 杈撳嚭鈥濆拰鈥?shared, routed) 浜屽厓缁勮緭鍑衡€?
        def trunc(x: torch.Tensor, trunc_size: int) -> torch.Tensor:
            # 鍙繚鐣欑湡瀹?hidden 缁达紝鍘绘帀涓轰簡閫傞厤 kernel 琛ュ嚭鏉ョ殑灏鹃儴缁村害銆?
            return x[..., :trunc_size]

        def reduce_and_trunc(x: torch.Tensor, trunc_size: int) -> torch.Tensor:
            # 鍏堟寜闇€鍋?TP all-reduce锛屽啀瑁佸洖鐪熷疄 hidden 缁淬€?
            return trunc(self.maybe_all_reduce_tensor_model_parallel(x), trunc_size)

        if (
            not self.moe_config.is_sequence_parallel
            and not self.use_dp_chunking
            and self.reduce_results
            and (self.moe_config.tp_size > 1 or self.moe_config.ep_size > 1)
        ):
            # 婊¤冻鏉′欢鏃讹紝璇存槑褰撳墠杈撳嚭浠嶉渶瑕侀澶栬法 rank 瑙勭害銆?
            func = reduce_and_trunc
        else:
            # 鍏朵綑鎯呭喌涓嬪彧闇€瑕佽鍓紝涓嶉渶瑕佸湪杩欓噷琛ラ澶栬绾︺€?
            func = trunc

        if isinstance(states, tuple):
            # `(shared, routed)` 浜屽厓缁勮緭鍑烘椂锛屽涓や釜鍒嗘敮鍒嗗埆搴旂敤鍚屼竴濂楀悗澶勭悊閫昏緫銆?
            return tuple(
                [func(s, trunc_size) for s, trunc_size in zip(states, trunc_sizes)]
            )
        else:
            # 鍗曡緭鍑哄満鏅笅锛宍trunc_sizes` 閲屽彧搴旀湁涓€涓洰鏍囩淮搴︺€?
            assert len(trunc_sizes) == 1
            return func(states, trunc_sizes[0])

    def _encode_layer_name(self) -> str | ModuleName:
        # 鎶?Python 灞傚璞＄殑 `layer_name` 缂栫爜鎴?custom op 鍙互璇嗗埆鐨勫彞鏌勩€?
        # 鏌愪簺璺緞涓?custom op 鍙嬁寰楀埌涓€涓交閲忔爣璇嗭紝鍚庨潰鍐嶉€氳繃 forward context
        # 鍙嶆煡鐪熷疄灞傚璞°€?
        if HAS_OPAQUE_TYPE:
            # 鑻ュ綋鍓嶈繍琛岀幆澧冩敮鎸?opaque 绫诲瀷锛屽氨鐩存帴灏佽鎴?`ModuleName` 浼犵粰 custom op銆?
            return ModuleName(self.layer_name)
        # 鍗曟祴鐜閲?forward context 鍙兘涓嶅瓨鍦紝鎴?all_moe_layers 涓虹┖銆?
        if (
            is_forward_context_available()
            and get_forward_context().all_moe_layers is not None
        ):
            # 鑻?forward context 鍙敤锛屽氨璁?custom op 璧扳€滀粠涓婁笅鏂囧弽鏌ュ眰瀵硅薄鈥濈殑鍒嗘敮銆?
            return "from_forward_context"
        # 鏈€鍚庡厹搴曡繑鍥炴櫘閫氬瓧绗︿覆灞傚悕銆?
        return self.layer_name

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # -----------------
        # runner 鐨勫叕寮€鍓嶅悜鍏ュ彛銆?
        # -----------------
        # 杩欓噷涓昏璐熻矗涓や欢浜嬶細
        # 1. 鍑嗗 routed / shared 涓ゆ潯鍒嗘敮鍚勮嚜闇€瑕佺殑杈撳叆褰㈢姸
        # 2. 璋冪敤娉ㄥ唽濂界殑 `moe_forward` custom op 鎴?Python fallback 鍏ュ彛锛?
        #    鍐嶅湪鏈熬缁熶竴鍋氳绾︿笌缁村害瑁佸壀
        # -----------------
        # 鍏ュ彛锛氬噯澶?routed / shared experts 鍚勮嚜闇€瑕佺殑杈撳叆銆?
        # -----------------
        use_split_tiered = self._use_split_tiered_for_piecewise()

        # latent MoE 涓嬪厛淇濈暀鍘熷 hidden_states锛?
        # shared experts 浠嶇敤鍘熷缁村害锛宺outed experts 鍒欒蛋鍙樻崲鍚庣殑缁村害銆?
        if self.shared_experts is not None:
            # shared experts 瀛樺湪鏃讹紝闇€瑕佷繚鐣欎竴浠藉師濮嬭緭鍏ョ粰 shared 鍒嗘敮浣跨敤銆?
            original_hidden_states = hidden_states
            # 鍚屾椂璁板綍鍘熷 hidden 缁达紝鍚庨潰缁?shared 杈撳嚭瑁佸壀鏃朵細鐢ㄥ埌銆?
            original_hidden_dim = (
                self.moe_config.hidden_dim if use_split_tiered
                else hidden_states.shape[-1]
            )
        else:
            # 娌℃湁 shared experts 鏃讹紝杩欎袱涓€煎悗闈㈤兘涓嶄細鍙備笌瀹為檯璁＄畻銆?
            original_hidden_states = None

        # 鍏堝 routed experts 杈撳叆鍋氬彲閫夊彉鎹紝渚嬪 latent projection銆?
        hidden_states = self.apply_routed_input_transform(hidden_states)

        # routed 鍒嗘敮鑻ヨ pad 鍒?kernel 闇€瑕佺殑 hidden_dim锛屾渶鍚庤繕瑕佸啀瑁佸洖鍘汇€?
        # PIECE split 路径避免把 Python `torch.Size` 对象跨 FX/AOT 边界传递；
        # 当前只在无 routed_input_transform 时启用 split，因此 routed dim 可静态确定。
        if use_split_tiered:
            transformed_hidden_dim = self.moe_config.hidden_dim
            routed_output_dim = self.moe_config.hidden_dim
        else:
            transformed_hidden_dim = hidden_states.shape[-1]
            routed_output_dim = transformed_hidden_dim
        if not use_split_tiered and self.moe_config.hidden_dim != transformed_hidden_dim:
            # 鑻ュ彉鎹㈠悗缁村害灏忎簬 kernel 鏈熸湜缁村害锛屽氨鍦ㄥ熬閮ㄨˉ 0 瀵归綈鍒?kernel 鐨?hidden_dim銆?
            hidden_states = F.pad(
                hidden_states,
                (0, self.moe_config.hidden_dim - transformed_hidden_dim),
                mode="constant",
                value=0.0,
            )

        # 鐪熸鎵ц routed experts 鐨?forward锛泂hared_experts_input 浣滀负绗笁鍙傛暟閫忎紶銆?
        # 杩欓噷鐨?`self.moe_forward` 鍙兘鏄?Python fallback锛屼篃鍙兘鏄敞鍐屽ソ鐨?custom op銆?
        if use_split_tiered:
            fused_output = self._forward_split_tiered_for_piecewise(
                hidden_states,
                router_logits,
                original_hidden_states,
            )
        else:
            fused_output = self.moe_forward(
                hidden_states,
                router_logits,
                original_hidden_states,
                self._encode_layer_name(),
            )

        # 鏍规嵁鏄惁瀛樺湪 shared experts锛屽噯澶囨渶鍚庣殑瑁佸壀灏哄鍒楄〃銆?
        if self.shared_experts is not None:
            # shared/routed 鍙岃緭鍑烘椂锛岃鍒嗗埆鎶婁袱涓垎鏀鍥炲悇鑷湡瀹炵淮搴︺€?
            orig_hidden_dims = [
                original_hidden_dim,
                routed_output_dim if use_split_tiered else transformed_hidden_dim,
            ]
        else:
            # 鍙湁 routed 杈撳嚭鏃讹紝鍙渶瑕佷繚鐣欎竴涓洰鏍囩淮搴︺€?
            orig_hidden_dims = [routed_output_dim]

        # 鏈€缁堢粺涓€鍦ㄨ繖閲屽仛 reduce + trunc銆?
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
        # DP chunking 涓撶敤鎵ц璺緞銆?
        # -----------------
        # 褰撳崟娆?token 鎵规杩囧ぇ銆乥ackend 鍙堟敮鎸?DP chunking 鏃讹紝浼氭妸瀹屾暣杈撳叆鍒囨垚澶氫釜
        # token chunk锛岄噸澶嶆墽琛屸€渟tage -> route -> experts -> write back鈥濓紝鏈€鍚庡啀鎶婃墍鏈?
        # chunk 鐨勮緭鍑烘嫾鍥炲畬鏁村紶閲忋€?
        # 杩涘叆杩欓噷鍓嶏紝澶栧眰宸茬粡鍐冲畾褰撳墠 backend 闇€瑕?/ 鏀寔 DP chunking銆?
        assert self.batched_hidden_states is not None
        assert self.batched_router_logits is not None
        # staging buffer 涓庡綋鍓嶈緭鍏ュ繀椤讳娇鐢ㄥ悓涓€ dtype锛屽惁鍒欏悗缁?copy_/kernel 璋冪敤浼氬嚭閿欍€?
        assert self.batched_hidden_states.dtype == full_hidden_states.dtype, (
            f"{self.batched_hidden_states.dtype} == {full_hidden_states.dtype}"
        )
        assert self.batched_router_logits.dtype == full_router_logits.dtype, (
            f"{self.batched_router_logits.dtype} == {full_router_logits.dtype}"
        )
        # 纭 staging buffer 鐨勬渶鍚庝竴缁村拰瀹屾暣杈撳叆涓€鑷淬€?
        assert self.batched_hidden_states.size(-1) == full_hidden_states.size(-1)
        assert self.batched_router_logits.size(-1) == full_router_logits.size(-1)

        # TODO(bnell): 淇 DP chunking 涓嬬殑 shared_expert_inputs 鏀寔銆?
        # assert shared_input is None, (
        #    "Routed input transform is not currently supported with DP chunking."
        # )

        # 棰勫厛鍒嗛厤瀹屾暣 routed 杈撳嚭寮犻噺锛涙瘡涓?chunk 鎵ц瀹屽悗浼氭妸鑷繁鐨勭粨鏋滃啓鍥炲搴斿尯闂淬€?
        full_fused_final_hidden_states = torch.empty_like(full_hidden_states)
        if self.shared_experts is not None:
            # 鑻ュ瓨鍦?shared experts锛屼篃涓?shared 鍒嗘敮鍗曠嫭棰勫垎閰嶄竴鍧楀畬鏁磋緭鍑哄紶閲忋€?
            full_shared_final_hidden_states = torch.empty_like(full_hidden_states)

        # -----------------
        # 鍗曚釜 chunk 鐨勫鐞嗗嚱鏁般€?
        # -----------------
        def process_chunk(chunk_start, chunk_end, skip_result_store=False):
            # 褰撳墠 chunk 瑕嗙洊鐨?token 鏁般€?
            chunk_size = chunk_end - chunk_start
            # 鍏堜粠瀹屾暣杈撳叆涓垏鍑哄綋鍓?chunk銆?
            hidden_states = full_hidden_states[chunk_start:chunk_end, :]
            router_logits = full_router_logits[chunk_start:chunk_end, :]
            # shared_input 鑻ュ瓨鍦紝涔熷繀椤绘寜鍚屾牱鐨?token 鍖洪棿鍒囩墖銆?
            shared_input = (
                full_shared_input[chunk_start:chunk_end, :]
                if full_shared_input is not None
                else None
            )

            assert self.batched_hidden_states is not None
            assert self.batched_router_logits is not None
            # 鍙湁 DBO 鎵撳紑鏃讹紝staging tensor 鎵嶄細澶氬嚭涓€灞?ubatch 缁村害銆?
            if self.batched_hidden_states.dim() == 3:
                assert self.batched_router_logits.dim() == 3
                # DBO 妯″紡涓嬫寜褰撳墠 ubatch 閫夋嫨瀵瑰簲鐨?staging buffer銆?
                batch_buffer_idx = dbo_current_ubatch_id()
                batched_hidden_states = self.batched_hidden_states[batch_buffer_idx, :]
                batched_router_logits = self.batched_router_logits[batch_buffer_idx, :]
            else:
                # 闈?DBO 妯″紡鐩存帴澶嶇敤鏁村潡 staging buffer銆?
                batched_hidden_states = self.batched_hidden_states
                batched_router_logits = self.batched_router_logits

            # 鍙埅鍙栧綋鍓?chunk 瀹為檯闇€瑕佺殑鍓嶅崐娈?staging 绌洪棿銆?
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
            # 鎶婂綋鍓?chunk 鎷疯繘 staging buffer锛屽悗缁?kernel 閮界洿鎺ヨ staging tensor銆?
            # 浣跨敤 `non_blocking=True` 鍏佽鍦ㄦ弧瓒虫潯浠舵椂璧板紓姝ユ嫹璐濄€?
            staged_hidden_states.copy_(hidden_states, non_blocking=True)
            staged_router_logits.copy_(router_logits, non_blocking=True)

            # shared 鍒嗘敮鑻ユ病鏈夊崟鐙緭鍏ワ紝灏遍粯璁ょ洿鎺ヤ娇鐢?staged 鍚庣殑 hidden_states銆?
            shared_input = (
                shared_input if shared_input is not None else staged_hidden_states
            )

            # -----------------
            # chunk 鍐呴儴鐨?MoE 涓昏绠椼€?
            # -----------------
            # 鏍稿績涓撳璁＄畻闃舵銆?
            if self.quant_method.is_monolithic:
                # monolithic kernel 浼氬湪涓€涓叆鍙ｉ噷鍚屾椂瀹屾垚 router + experts 涓昏绠椼€?
                assert has_separate_shared_experts or self.shared_experts is None
                final_hidden_states = self._apply_monolithic_with_tiered_cache(
                    layer=layer,
                    x=staged_hidden_states,
                    router_logits=staged_router_logits,
                )
            else:
                # 鍏堝仛 router top-k 閫夋嫨锛屽啀杩涘叆 tiered cache / quant method 鎵ц銆?
                topk_weights, topk_ids = self.router.select_experts(
                    hidden_states=staged_hidden_states,
                    router_logits=staged_router_logits,
                )

                # top-k 缁撴灉浼氬喅瀹?routed token 瀹為檯鍛戒腑鐨?experts 浠ュ強鍚庣画 tiered cache prepare銆?
                final_hidden_states = self._apply_with_tiered_cache(
                    layer=layer,
                    x=staged_hidden_states,
                    router_logits=staged_router_logits,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    shared_experts_input=shared_input,
                )

            if has_separate_shared_experts:
                assert not isinstance(final_hidden_states, tuple)
                assert self.shared_experts is not None

                # shared experts 鍗曠嫭璺戝畬鍚庯紝鍜?routed experts 杈撳嚭鎵撳寘鎴愪簩鍏冪粍銆?
                # 杩欓噷 shared 鍒嗘敮鍜?routed 鍒嗘敮瀵归綈鍦ㄥ悓涓€ token chunk 涓娿€?
                shared_output = self.shared_experts(shared_input)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )

            if not skip_result_store:
                # 鎶婂綋鍓?chunk 鐨勭粨鏋滃啓鍥炲畬鏁磋緭鍑哄紶閲忋€?
                if self.shared_experts is None:
                    # routed-only 鍦烘櫙涓嬶紝鐩存帴鍐欏洖 routed 杈撳嚭銆?
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states, non_blocking=True
                    )
                else:
                    # shared/routed 鍙岃緭鍑哄満鏅笅锛屼袱鍧楀畬鏁磋緭鍑哄紶閲忓垎鍒啓鍥炪€?
                    full_shared_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[0], non_blocking=True
                    )
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[1], non_blocking=True
                    )

        # -----------------
        # 璁＄畻 chunk 寰幆杈圭晫銆?
        # -----------------
        # forward context 涓細璁板綍褰撳墠 DP dispatcher 鍦ㄦ暣涓?batch 涓婄湅鍒扮殑 token 涓婄晫銆?
        ctx = get_forward_context()
        # flashinfer_cutlass kernel 鍙悓鏃惰鐩栧彲閫夌殑 DP 涓?TP/EP 缁勫悎銆?
        max_tokens_across_dispatchers = ctx.dp_metadata.max_tokens_across_dp_cpu
        # 鍗曚釜 rank 姣忔鏈€澶氬鐞嗗灏?token锛岀敱 moe 閰嶇疆閲岀殑 `max_num_tokens` 鍐冲畾銆?
        moe_dp_chunk_size_per_rank = self.moe_config.max_num_tokens

        # 鑻ヨ緭鍏ユ湰韬蛋浜?sequence parallel锛岄渶瑕佸厛闄や互 sp_size锛?
        # 鎵嶈兘寰楀埌鍗曚釜 dispatcher 瀹為檯鍙兘鐪嬪埌鐨勬渶澶?token 鏁般€?
        if self.moe_config.is_sequence_parallel:
            max_tokens_across_dispatchers = cdiv(
                max_tokens_across_dispatchers, self.moe_config.sp_size
            )

        # 褰撳墠鐪熷疄杈撳叆閲屽疄闄呭寘鍚灏?token銆?
        num_tokens = full_hidden_states.size(0)
        # 閫愪釜 chunk 杩涘叆鍓嶉潰鐨?process_chunk銆?
        for chunk_idx, chunk_start_ in enumerate(
            range(0, max_tokens_across_dispatchers, moe_dp_chunk_size_per_rank)
        ):
            # 鍒濆 chunk 璧风偣灏辨槸杩欒疆寰幆瀵瑰簲鐨?token 鍋忕Щ銆?
            chunk_start = chunk_start_
            # 鍒濆 chunk 缁堢偣鍙椻€渃hunk 澶у皬鈥濆拰鈥渄ispatcher 鏈€澶?token 涓婄晫鈥濆弻閲嶉檺鍒躲€?
            chunk_end = min(
                chunk_start + moe_dp_chunk_size_per_rank, max_tokens_across_dispatchers
            )
            # 鍐嶆妸 chunk 杈圭晫瑁佸埌褰撳墠鐪熷疄 token 鑼冨洿鍐呫€?
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)
            # 杩涘叆 `chunked_sizes(...)` 涓婁笅鏂囧悗锛屼笅娓?kernel / communicator 鍙互璇诲彇褰撳墠
            # chunk 鍦?DP/SP 瑙嗚涓嬬殑鍏冧俊鎭€?
            with ctx.dp_metadata.chunked_sizes(
                self.moe_config.sp_size, moe_dp_chunk_size_per_rank, chunk_idx
            ):
                # 鑻ヨ繖杞惊鐜殑璧风偣宸茬粡瓒呭嚭鐪熷疄 token 鑼冨洿锛屽氨鍙洿鏂板厓淇℃伅鑰岃烦杩囩粨鏋滃啓鍥炪€?
                process_chunk(
                    chunk_start, chunk_end, skip_result_store=chunk_start_ >= num_tokens
                )

        # 鎵€鏈?chunk 澶勭悊瀹屽悗锛屾寜鏄惁瀛樺湪 shared experts 鍐冲畾杩斿洖缁撴瀯銆?
        if self.shared_experts is None:
            # routed-only 鍦烘櫙鐩存帴杩斿洖瀹屾暣 routed 杈撳嚭銆?
            return full_fused_final_hidden_states
        else:
            # shared/routed 鍙岃緭鍑哄満鏅繚鎸?`(shared, routed)` 杩斿洖缁撴瀯銆?
            return (full_shared_final_hidden_states, full_fused_final_hidden_states)

    def forward_impl(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # ------------------------------- 涓绘祦绋嬪叆鍙?-------------------------------
        # 鎵ц鍗曞眰 MoE 鐨勫畬鏁村墠鍚戜富绾匡細
        # 鍏堝噯澶囪矾鐢辫緭鍏ヤ笌 shared 鍒嗘敮鐘舵€侊紝鍐嶆墽琛?routed 涓昏绠楋紝
        # 鏈€鍚庢寜骞惰鎷撴墤瑙勭害杈撳嚭骞惰繑鍥炪€?
        assert self.quant_method is not None

        # 鑻ュ綋鍓嶅悗绔彲鑳借蛋 DP chunking锛岃繖閲屽厛纭繚 staging buffer 宸插氨缁€?
        # 杩欐牱鍚庣画鏃犺鏄惁鐪熸杩涘叆 chunked 鍒嗘敮锛岄兘涓嶄細璁块棶鏈垵濮嬪寲鐘舵€併€?
        self.ensure_dp_chunking_init()

        # 鍙湁 shared experts 鏈鍐呮牳鍐呴儴铻嶅悎鎺ョ鏃讹紝
        # runner 鎵嶉渶瑕佹樉寮忚皟 shared 鍒嗘敮骞剁鐞嗗叾鎵ц鏃舵満銆?
        has_separate_shared_experts = (
            not self.quant_method.mk_owns_shared_expert
            and self.shared_experts is not None
        )

        # 鏍规嵁骞惰鍚庣鑳藉姏鍒ゆ柇鏈鏄惁璧?chunked 涓撶敤瀹炵幇銆?
        use_chunked_impl = self.use_dp_chunking

        # 鍐冲畾 shared experts 鏄惁鏀惧埌鐙珛 CUDA stream 寮傛鎵ц锛?
        # 骞舵彁鍓嶇‘瀹?shared 鍒嗘敮瑕佹秷璐圭殑杈撳叆寮犻噺銆?
        use_shared_experts_stream, shared_experts_input = (
            self._maybe_setup_shared_experts_stream(
                hidden_states,
                shared_input,
                has_separate_shared_experts,
                use_chunked_impl,
            )
        )

        # ------------------------------- gate/router 棰勫鐞?-------------------------------
        # 鑻ュ綋鍓嶅眰鍐呯疆浜?gate锛屽氨浠ュ眰鍐?gate 鐨勭粨鏋滀负鍑嗚鐩栧閮?logits锛?
        # 璁╁悗缁?shared/routed 涓ゆ潯璺緞璇诲彇鍚屼竴浠借矾鐢变俊鎭€?
        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)

        # ------------------------------- chunked 蹇€熷垎鏀?-------------------------------
        # chunked 妯″紡涓嬬洿鎺ュ垏鍒颁笓鐢ㄥ疄鐜帮紝閬垮厤涓庨潪 chunked 閫昏緫浜ゅ弶銆?
        if use_chunked_impl:
            return self.forward_impl_chunked(
                layer,
                hidden_states,
                router_logits,
                shared_input,
                has_separate_shared_experts,
            )

        # ------------------------------- 闈?chunked 涓昏矾寰?-------------------------------
        # 浠呭湪 DP>1 涓旈噺鍖栧疄鐜颁笉鏀寔鍐呮牳鍐呴儴鍒嗗彂鏃讹紝鎵嶅惎鐢?naive dispatch/combine銆?
        # TODO(rob): 绛夋墍鏈?quant method 杩佺Щ鍒?MK 鍚庯紝鍙垹闄よ鍏煎鍒嗘敮銆?
        do_naive_dispatch_combine = (
            self.moe_config.dp_size > 1 and not self.quant_method.supports_internal_mk
        )

        # 璇诲彇褰撳墠 forward 鐨勫苟琛屼笂涓嬫枃鍏冩暟鎹€?
        ctx = get_forward_context()
        # 鑻ュ瓨鍦?DP metadata锛屽垯鎸?SP 鍙ｅ緞璁剧疆鏈湴 token 瑙嗗浘锛?
        # 鑻ヤ笉瀛樺湪锛屽垯鐢ㄧ┖涓婁笅鏂囦繚鎸佸悗缁啓娉曚竴鑷淬€?
        sp_ctx = (
            ctx.dp_metadata.sp_local_sizes(self.moe_config.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

        with sp_ctx:
            # ------------------------------- shared experts 棰勬墽琛?-------------------------------
            # 鑻?shared 鍒嗘敮涓嶈蛋鐙珛 stream锛屽垯鍏堝湪涓?stream 鍚屾鎵ц銆?
            # 杩欐牱鍙互鍦?routed 璺緞鏀瑰啓杈撳叆鍓嶆嬁鍒扮ǔ瀹氱殑 shared 杈撳嚭銆?
            if has_separate_shared_experts and not use_shared_experts_stream:
                assert self.shared_experts is not None
                # 鑻ユ湭鏄惧紡鎻愪緵 shared_input锛屽垯榛樿澶嶇敤褰撳墠 hidden_states銆?
                shared_input = (
                    shared_input if shared_input is not None else hidden_states
                )
                shared_output = self.shared_experts(shared_input)

            # ------------------------------- dispatch 闃舵 -------------------------------
            # naive 璺緞涓嬪厛鎶?token 涓庤矾鐢?logits 鍒嗗彂鍒扮洰鏍?expert rank锛?
            # 璁╂湰 rank 鍙鐞嗚嚜宸辫礋璐ｇ殑涓撳杈撳叆銆?
            # TODO: 绛夋墍鏈?kernel 杩佺Щ鍒?MoEKernel 妗嗘灦鍚庯紝鍙垹闄よ鍒嗘敮銆?
            if do_naive_dispatch_combine:
                hidden_states, router_logits = get_ep_group().dispatch_router_logits(
                    hidden_states,
                    router_logits,
                    self.moe_config.is_sequence_parallel,
                )

            # ------------------------------- PCP gather 闃舵 -------------------------------
            # PCP 寮€鍚椂锛屽湪 token 缁存敹闆嗗悇 rank 鍒嗙墖锛?
            # 璁╁悗缁?routed 璁＄畻鍩轰簬瀹屾暣涓婁笅鏂囪鍥炬墽琛屻€?
            if self.moe_config.pcp_size > 1:
                hidden_states = get_pcp_group().all_gather(
                    hidden_states,
                    dim=0,
                )
                router_logits = get_pcp_group().all_gather(
                    router_logits,
                    dim=0,
                )

            # ------------------------------- routed experts 涓昏绠?-------------------------------
            # 鏍稿績涓撳璁＄畻闃舵锛?
            # monolithic 璺緞鐢辩粺涓€鍏ュ彛瀹屾垚璺敱涓庝笓瀹惰绠楋紱
            # 闈?monolithic 璺緞鍏堥€?top-k锛屽啀鎸?top-k 鎵ц涓撳鍓嶅悜銆?
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

                # top-k 缁撴灉浼氫綔涓?routed 璺緞鍚庣画璁＄畻涓庡悎骞剁殑椹卞姩淇″彿銆?
                final_hidden_states = self._apply_with_tiered_cache(
                    layer=layer,
                    x=hidden_states,
                    router_logits=router_logits,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    shared_experts_input=shared_input,
                )

            # ------------------------------- shared experts 骞惰鏀舵暃 -------------------------------
            # 鑻ュ瓨鍦ㄧ嫭绔?shared 鍒嗘敮锛岃繖閲屾妸 shared 涓?routed 涓よ矾缁撴灉瀵归綈骞剁粍缁囨垚缁熶竴缁撴瀯銆?
            if has_separate_shared_experts:
                assert self.shared_experts is not None

                if use_shared_experts_stream:
                    # shared experts 鍦ㄨ緟鍔?stream 涓婃墽琛岋紝涓?routed 璁＄畻骞惰銆?
                    with torch.cuda.stream(self.shared_experts_stream):
                        shared_output = self.shared_experts(shared_experts_input)
                    # 鍦ㄨ鍙?shared_output 鍓嶏紝涓?stream 蹇呴』绛夊緟杈呭姪 stream 瀹屾垚銆?
                    current_stream().wait_stream(self.shared_experts_stream)

                # 杈撳嚭缁熶竴缁勭粐涓?(shared_output, routed_output)銆?
                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )

            # ------------------------------- combine 闃舵 -------------------------------
            # 鎸夊苟琛屾嫇鎵戞妸 routed 杈撳嚭瑙勭害鍥炲綋鍓?rank 鐨勬渶缁堝竷灞€銆?
            def combine_output(states: torch.Tensor) -> torch.Tensor:
                # naive 璺緞涓嬪厛鍋?EP 渚?combine锛屾妸鍒嗗彂鍚庣殑 routed 杈撳嚭鎷煎洖鍘?token 瑙嗗浘銆?
                if do_naive_dispatch_combine:
                    states = get_ep_group().combine(
                        states, self.moe_config.is_sequence_parallel
                    )

                # PCP 璺緞涓嬪啀鍋?reduce_scatter锛屾妸涓婁笅鏂囩淮鑱氬悎缁撴灉鍒囧洖鏈?rank銆?
                if self.moe_config.pcp_size > 1:
                    states = get_pcp_group().reduce_scatter(
                        states,
                        dim=0,
                    )

                return states

            # shared+routed 妯″紡涓嬶紝浠?routed 鍒嗘敮鍦ㄦ澶?combine锛?
            # shared 鍒嗘敮淇濈暀缁欏灞傚喅瀹氳绾?鐩稿姞鏃舵満銆?
            if self.shared_experts is not None:
                return (
                    final_hidden_states[0],
                    combine_output(final_hidden_states[1]),
                )
            else:
                # routed-only 妯″紡涓嬶紝鐩存帴杩斿洖 combine 鍚庣殑鍗曞紶閲忚緭鍑恒€?
                return combine_output(final_hidden_states)
