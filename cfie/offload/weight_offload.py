"""Tiered MoE expert cache controllers."""

from __future__ import annotations

import ctypes
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
import os
from pathlib import Path
import sys
import time as _time
import threading
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import safe_open as safetensors_safe_open
from safetensors.torch import save_file as safetensors_save_file

from cfie import _custom_ops as ops
from cfie.logger import init_logger
from cfie.model_executor.layers.fused_moe.layer import FusedMoE
from cfie.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    convert_to_unquantized_kernel_format,
)
from cfie.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from cfie.model_executor.layers.quantization.gptq_marlin import GPTQMarlinMoEMethod
from cfie.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_moe_permute_scales,
)
from cfie.offload.cpu_backend import (
    ExpertBundle,
    PackedExpertTensorSpec,
    allocate_packed_cpu_tensor_views,
    bundle_nbytes,
    pack_cpu_bundles_by_expert,
    pack_batched_cpu_tensor_dicts_by_expert,
    pack_cpu_tensor_dict,
)
from cfie.offload.nvme_backend import SafetensorExpertStore
from cfie.offload.policy import (
    PLAN_KEY,
    get_moe_tiered_cache_plan,
)
from cfie.utils.platform_utils import is_pin_memory_available
from cfie.utils.torch_utils import current_stream

logger = init_logger(__name__)

DEFAULT_PREFILL_BURST_MIN_TOKENS = 8
DEFAULT_PREFILL_BURST_TOKENS_PER_GPU_SLOT = 4
DEFAULT_CPU_STATIC_PREPROCESS_BATCH_SIZE_CAP = 0
DEFAULT_CPU_STATIC_PREPROCESS_CPU_RESERVE_BYTES = 1 << 30
DEFAULT_CPU_STATIC_PREPROCESS_GPU_RESERVE_BYTES = 512 << 20
MARLIN_READY_EXPERT_CACHE_VERSION = 1
MARLIN_READY_FP8_WEIGHT_SCALE_FACTOR = 512
MARLIN_READY_FP8_PREPROCESS_SCHEMA_VERSION = 2
SMALL_TOPK_UNIQUE_CPU_THRESHOLD = 256


def _callable_accepts_keyword_argument(callback: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == keyword:
            return True
    return False



def _bench_timing_enabled() -> bool:
    return os.getenv("CFIE_BENCH_TIMING", "") == "1"


def _is_cuda_oom_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        message = str(current).lower()
        if (
            "out of memory" in message
            or "cudaerrormemoryallocation" in message
            or ("memory allocation" in message and "cuda" in message)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _synchronize_torch_device_best_effort(
    device: torch.device | str | None,
) -> None:
    if device is None:
        return
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError, ValueError):
        return
    if resolved.type != "cuda":
        return
    try:
        torch.cuda.synchronize(resolved)
    except (RuntimeError, TypeError, ValueError):
        return


def _tensor_debug_summary(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "None"
    return (
        f"shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} "
        f"device={tensor.device} "
        f"contiguous={tensor.is_contiguous()} "
        f"pinned={tensor.is_pinned() if tensor.device.type == 'cpu' else False}"
    )


def _debug_layer_key(obj: Any) -> str:
    return str(getattr(obj, "layer_key", "<unknown-layer>"))




# 闂傚洠鍋撻悷鏇氱窔閸ｇ顕欐ウ璺ㄧ缂備緡鍘藉Ο澶屸偓娑欘焽濞?MoE 闁哄鍟撮崳鍝モ偓娑欘殕椤斿矂宕ュ鍛仚閻炴稏鍔婇埀?
_GPU_EXPERT_FIELD_NAMES = (
    "w13_qweight",
    "w2_qweight",
    "w13_scales",
    "w2_scales",
    "w13_qzeros",
    "w2_qzeros",
)

# desc_act 婵☆垪鈧磭纭€濞戞挸顑夐·鍌涘緞閺嶎偅鐣?g_idx 閻庢稒顨嗛宀勫Υ?
_GPU_EXPERT_GIDX_FIELD_NAMES = (
    "w13_g_idx",
    "w2_g_idx",
    "w13_g_idx_sort_indices",
    "w2_g_idx_sort_indices",
)


def _build_window_groups(
    total_layers: int,
    window_layers: int,
    stride_layers: int,
    min_insertion: int,
) -> list[tuple[int, int]]:
    """閻犱緤绱曢悾?GPU 閺夆晝鍋熼悽濠氬及閹呮憼闁汇劌瀚悰銉╁矗閿濆懎鐎荤紓?(start_layer, end_layer) 鐎殿喒鍋撻柛鏍ㄦそ濡?"""
    first_start = min_insertion + 1
    if first_start >= total_layers:
        return []
    groups = [(first_start, min(first_start + window_layers, total_layers))]
    g = first_start + window_layers
    while g < total_layers:
        groups.append((g, min(g + stride_layers, total_layers)))
        g += stride_layers
    return groups


def _build_stride_windows(
    total_layers: int,
    window_layers: int,
    stride_layers: int,
    min_insertion: int,
) -> list[list[tuple[int, int]]]:
    """閻忓繐妫欓惁锛勭磼閸曨剙顎曞☉?stride 缂佹劖顨呰ぐ? 閺夆晜鏌ㄥú?[[(start, end), ...], ...].

    缂佹鍏涚粩瀵哥玻濡も偓瑜?(window_layers=8, stride_layers=4):
      闁?[[(5,9), (9,13)]]  (2 濞?stride 缂佹劖顨呰ぐ?
    闁告艾娴烽悽鑽ょ玻濡も偓瑜?(stride_layers=4):
      闁?[[(13,17)], [(17,21)], ...]
    """
    groups = _build_window_groups(total_layers, window_layers, stride_layers, min_insertion)
    result: list[list[tuple[int, int]]] = []
    for g_start, g_end in groups:
        sws: list[tuple[int, int]] = []
        s = g_start
        while s < g_end:
            e = min(s + stride_layers, g_end)
            sws.append((s, e))
            s = e
        result.append(sws)
    return result


# ---- GPU 閻忕偟鍋為埀顑啯鍊?闁?bundle tensor 闁告艾绉跺▓鎴﹀及閻樿尙娈?----
_GPU_TO_BUNDLE_FIELD: dict[str, str] = {
    "w13_qweight": "runtime.w13_qweight",
    "w2_qweight": "runtime.w2_qweight",
    "w13_scales": "runtime.w13_scales",
    "w2_scales": "runtime.w2_scales",
    "w13_qzeros": "runtime.w13_qzeros",
    "w2_qzeros": "runtime.w2_qzeros",
    "w13_g_idx": "runtime.w13_g_idx",
    "w2_g_idx": "runtime.w2_g_idx",
    "w13_g_idx_sort_indices": "runtime.w13_g_idx_sort_indices",
    "w2_g_idx_sort_indices": "runtime.w2_g_idx_sort_indices",
}


def _format_bytes(num_bytes: int) -> str:
    if num_bytes >= (1 << 30):
        return f"{num_bytes / (1 << 30):.2f} GiB"
    if num_bytes >= (1 << 20):
        return f"{num_bytes / (1 << 20):.2f} MiB"
    if num_bytes >= (1 << 10):
        return f"{num_bytes / (1 << 10):.2f} KiB"
    return f"{num_bytes} B"


def _extract_moe_layer_index(layer_key: str) -> int | None:
    marker = ".layers."
    if marker not in layer_key:
        return None
    tail = layer_key.split(marker, 1)[1]
    raw = tail.split(".", 1)[0]
    try:
        return int(raw)
    except ValueError:
        return None



def _get_available_cpu_memory_bytes_best_effort() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception:
        psutil = None

    if psutil is not None:
        try:
            return int(psutil.virtual_memory().available)
        except Exception:
            pass

    if sys.platform == "win32":
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        memory_status = _MemoryStatusEx()
        memory_status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
            return int(memory_status.ullAvailPhys)
        return None

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * available_pages
    except (AttributeError, OSError, ValueError):
        return None


def _empty_torch_host_allocator_cache_best_effort() -> None:
    try:
        empty_host_cache = getattr(torch._C, "_host_emptyCache", None)
    except Exception:
        empty_host_cache = None
    if empty_host_cache is None:
        return
    try:
        empty_host_cache()
    except Exception:
        return


@dataclass(slots=True)
class _RawExpertWeights:
    # 闂佹彃绻愮€靛弶绋夐幘鎰佸晙闁?CPU staging/raw buffer 濞戞搩鍙冮崳鎼佸棘閻楀牆顏婚悷浣告噹閹鎯?w13 闁告ê顑囩紓澶愬级閸愵喖娅㈤柕?
    w13_qweight: torch.Tensor
    # 闂佹彃绻愮€靛弶绋夐幘鎰佸晙闁?CPU staging/raw buffer 濞戞搩鍙冮崳鎼佸棘閻楀牆顏婚悷浣告噹閹鎯?w2 闁告ê顑囩紓澶愬级閸愵喖娅㈤柕?
    w2_qweight: torch.Tensor
    # 闂佹彃绻愮€靛弶绋夐幘鎰佸晙闁?CPU staging/raw buffer 濞戞搩鍙冮崳鎼佸棘閻楀牆顏婚悷浣告噹閹鎯?w13 scale闁?
    w13_scales: torch.Tensor
    # 闂佹彃绻愮€靛弶绋夐幘鎰佸晙闁?CPU staging/raw buffer 濞戞搩鍙冮崳鎼佸棘閻楀牆顏婚悷浣告噹閹鎯?w2 scale闁?
    w2_scales: torch.Tensor
    # 闂佹彃绻愮€靛弶绋夐幘鎰佸晙闁?CPU staging/raw buffer 濞戞搩鍙冮崳鎼佸棘閻楀牆顏婚悷浣告噹閹鎯?w13 qzeros闁?
    w13_qzeros: torch.Tensor
    # 闂佹彃绻愮€靛弶绋夐幘鎰佸晙闁?CPU staging/raw buffer 濞戞搩鍙冮崳鎼佸棘閻楀牆顏婚悷浣告噹閹鎯?w2 qzeros闁?
    w2_qzeros: torch.Tensor
    # desc_act=True 闁哄啳顔愮槐婕te/up 閻犱警鍨扮欢鐐存交濮椻偓濞撳墎鎲版担椋庣闁伙絾鐟︾敮鎾存償韫囨挸顤呴柣?g_idx闁?
    w13_g_idx: torch.Tensor | None = None
    # desc_act=True 闁哄啳顔愮槐婕漮wn_proj 閻犱警鍨扮欢鐐存交濮椻偓濞撳墎鎲版担椋庣闁伙絾鐟︾敮鎾存償韫囨挸顤呴柣?g_idx闁?
    w2_g_idx: torch.Tensor | None = None


@dataclass(slots=True)
class _RawUnquantizedExpertWeights:
    # 闂傚牏鍋ら崳娲礌閺嶏妇鐟╅悗纭呮硾濠€?CPU staging/raw buffer 濞戞搩鍙冮崳鎼佸棘閻楀牆顏婚悷浣告噹閹鎯冮崟顐ｅ€ゆ?w13 闁哄鍟撮崳鎼佸Υ?
    w13_weight: torch.Tensor
    # 闂傚牏鍋ら崳娲礌閺嶏妇鐟╅悗纭呮硾濠€?CPU staging/raw buffer 濞戞搩鍙冮崳鎼佸棘閻楀牆顏婚悷浣告噹閹鎯?w2 闁哄鍟撮崳鎼佸Υ?
    w2_weight: torch.Tensor


@dataclass(slots=True)
class _PrefillBurstExecutionStats:
    # 闁哄牜鍓氶?burst 闁圭瑳鍡╂斀闁烩晛鐡ㄧ敮鎾川閹存帟鍘?resident GPU experts 闁汇劌瀚濂稿极閼割兘鍋?
    resident_hits: int = 0
    # 闁哄牜鍓氶?burst 闁圭瑳鍡╂斀闁告稒鍨濋懙?CPU static experts 闁汇劌瀚濂稿极閼割兘鍋?
    cpu_hits: int = 0
    # 闁哄牜鍓氶?burst 闁圭瑳鍡╂斀闂傚洠鍋撻悷鏇氭缁?NVMe 闁瑰嘲顦ぐ?experts 闁汇劌瀚濂稿极閼割兘鍋?
    nvme_loads: int = 0


@dataclass(slots=True)
class _MarlinReadyLayerCache:
    tensors: dict[str, torch.Tensor] | None = None
    bundles: list[ExpertBundle] | None = None


class _PrefillBurstExecutionLayer:
    """Execution view backed by shared prefill burst storage."""

    def __init__(
            self,
            base_layer: FusedMoE,
            target: Any,
            expert_map: torch.Tensor,
            num_slots: int,
    ) -> None:
        # ------------------------------- 濞ｅ洦绻傞悺銊╁礂閸欐﹢鐓?burst pool 閻庣數顢婇挅鍕嵁鐠鸿櫣鏆氶柟瀛樺姍椤╄鈻庨敍鍕嫧閻?-------------------------------
        # 閻犱焦婢樼紞宥夊礂閸欐﹢鐓?burst pool 闁哄牜鍏涚紞瀣晬鐏炶姤鍊电紓渚囧幗婢х晫鎮扮仦鐐槯濞村吋姘ㄥú鍧楀箳閵夈倗鐭ら弶鈺傜懇閸ｉ鎷犵拠鎻掔悼濞戞挸鐡ㄥ鍌炲箥瑜戦、鎴濐嚕閻樿娅ら柕?
        self._target = target

        # 濞达綀娉曢弫銈堛亹閹惧啿顤呴柛鈺勬椤㈠懐浠﹂崒妯峰亾娴ｉ鐟╅悗纭呭煐濡惭呬焊閸曨喓鈧啴宕仦鎷橆偅鎷呭鍡樻閻庣懓鏈崹姘閿濆洦鍊為悘鐐插€诲▓鎴烇純閺嶎煈鍋х紓浣瑰灥閻ｉ箖濡?
        self.bind(base_layer, expert_map, num_slots)

    def bind(
            self,
            base_layer: FusedMoE,
            expert_map: torch.Tensor,
            num_slots: int,
    ) -> None:
        # ------------------------------- 缂備焦鍨甸悾楣冨春閾忚鏀ㄩ悘鐐插€风粭宀冦亹閹惧啿顤?burst expert 闁哄嫮濮撮惃鐘诲礂瀹曞洭鍏?-------------------------------
        # 濞ｅ洦绻傞悺銊╂儑閻旈鏉介柣銊ュ閻斺偓缁绢厸鍋?FusedMoE 閻忕偛鍋婄槐婵嬪触鎼达絿鏁鹃柡鍫簼濡顕ｈ箛姘兼船闁烩晜鐗滃▓鎴犱沪閻愮补鍋撹缁辨壆绱掗悢鍓侇伇闂侇偄绻嬬槐鍓佺磼濞嗗繒鏆婇柕?
        self._base_layer = base_layer

        # 濞达綀娉曢弫銈堛亹閹惧啿顤?burst pool 闁?expert 闁哄嫮濮撮惃鐘垫偘閵婎煈娲柣鈺傜墪鐢偅鎱ㄧ€ｎ亞婀村☉鎾筹功濞?resident expert 闁哄嫮濮撮惃鐘诲Υ?
        self._expert_map = expert_map

        # 閻忓繐妫楃紞瀣礈瀹ュ嫬鏁╅柣鐐叉閻即鎯冮崟顒佹嫳闁革箓顣︾粭鎾垛偓纭呭煐閺嗙喖鏌岃箛鏃€鏆柛鎰懁鐠?burst pool 闁汇劌瀚径宥夊籍閼告晫顐ｆ媴瀹ュ棙娈堕柕?
        self.local_num_experts = num_slots

        # 濞ｅ洦绻冪€垫棃宕楅妸銉ф拱濞戞挻鎸搁宥夊极娴兼潙娅ゅ☉鎾抽鐢偅鎱ㄧ€ｎ亞婀村☉鎾亾闁肩柉鎻槐婵堟兜椤旇崵绠?router 闁?top-k expert id 闁汇劌瀚銏＄▕婢跺鐟濋柛娆惽滈埀?
        self.global_num_experts = base_layer.global_num_experts

        # ------------------------------- 闁革负鍔戦崳娲礌閺嶎剛鐔呯€垫澘瀚粭鍛村箮婵犲啫鈷旈悶娑樻湰濞煎牓鏌屽鍛倞闂佹彃绻橀崳鍝モ偓瑙勮壘閹粓宕?burst pool -------------------------------
        # 鐟滅増鎸搁崣鈩冪椤愩倖绐楅柡宥呮搐椤曨喚鎸掗垾鑼憪閻庢稒锚濠€顏堟煂韫囨挸顕ч柡澶婂暣閸ｇ顕ｉ悩璇叉闁哄啳顔愮槐婵堟嫚鐎涙ɑ顫栫憸鐗堟尭婢х姴顔忛妷銈囩▕闁革负鍔戦崳娲礌閺嶃劌鈷旈悶娑樼焷閻儳顕ラ崟顏嗙憮闁?
        if hasattr(self._target, "w13_qweight"):
            # 閻?w13 闂佹彃绻愮€垫煡寮堕崘顔兼闂佹彃绉撮悾楣冨触閹存繂鐓?burst pool 闁汇劌瀚径宥夊籍鐠鸿櫣鐐婇梺鎻掔箞閳?
            self.w13_qweight = self._target.w13_qweight

            # 閻?w2 闂佹彃绻愮€垫煡寮堕崘顔兼闂佹彃绉撮悾楣冨触閹存繂鐓?burst pool 闁汇劌瀚径宥夊籍鐠鸿櫣鐐婇梺鎻掔箞閳?
            self.w2_qweight = self._target.w2_qweight

            # 閻?w13 闁?scale 鐎殿喚濞€閸ｆ椽鏌屽鍛毎闁告碍鍨甸崺?burst pool 闁汇劌瀚径宥夊籍鐠鸿櫣鐐婇梺鎻掔箞閳?
            self.w13_scales = self._target.w13_scales

            # 閻?w2 闁?scale 鐎殿喚濞€閸ｆ椽鏌屽鍛毎闁告碍鍨甸崺?burst pool 闁汇劌瀚径宥夊籍鐠鸿櫣鐐婇梺鎻掔箞閳?
            self.w2_scales = self._target.w2_scales

            # 閻?w13 闁?qzeros 鐎殿喚濞€閸ｆ椽鏌屽鍛毎闁告碍鍨甸崺?burst pool 闁汇劌瀚径宥夊籍鐠鸿櫣鐐婇梺鎻掔箞閳?
            self.w13_qzeros = self._target.w13_qzeros

            # 閻?w2 闁?qzeros 鐎殿喚濞€閸ｆ椽鏌屽鍛毎闁告碍鍨甸崺?burst pool 闁汇劌瀚径宥夊籍鐠鸿櫣鐐婇梺鎻掔箞閳?
            self.w2_qzeros = self._target.w2_qzeros

            # 閻?w13 闁?g_idx 鐎殿喚濞€閸ｆ椽鏌屽鍛毎闁告碍鍨甸崺?burst pool 闁汇劌瀚径宥夊籍鐠鸿櫣鐐婇梺鎻掔箞閳?
            self.w13_g_idx = self._target.w13_g_idx

            # 閻?w2 闁?g_idx 鐎殿喚濞€閸ｆ椽鏌屽鍛毎闁告碍鍨甸崺?burst pool 闁汇劌瀚径宥夊籍鐠鸿櫣鐐婇梺鎻掔箞閳?
            self.w2_g_idx = self._target.w2_g_idx

            # 閻?w13 闁?g_idx_sort_indices 鐎殿喚濞€閸ｆ椽鏌屽鍛毎闁告碍鍨甸崺?burst pool 闁汇劌瀚径宥夊籍鐠鸿櫣鐐婇梺鎻掔箞閳?
            self.w13_g_idx_sort_indices = self._target.w13_g_idx_sort_indices

            # 閻?w2 闁?g_idx_sort_indices 鐎殿喚濞€閸ｆ椽鏌屽鍛毎闁告碍鍨甸崺?burst pool 闁汇劌瀚径宥夊籍鐠鸿櫣鐐婇梺鎻掔箞閳?
            self.w2_g_idx_sort_indices = self._target.w2_g_idx_sort_indices

            # 闁哄本鍔掔花娲焻濮樿鲸鏆忛柟绗涘棭鏀介悹渚灠缁剁偞瀵煎鍨涘亾濮樺磭绠?w13_weight 閻犱礁娼″Λ鍫曞级閸愵喖娅㈤柨娑樼焷缁绘牠鏌岀仦鎯╅悗鐟板暙閸╁棝宕ュ鍛厒闂佹彃绻愮€垫煡寮堕崘顔兼鐎殿喚濞€閸ｇ儤绋夋繛搴撳亾?
            self.w13_weight = self._target.w13_qweight

            # 闁哄本鍔掔花娲焻濮樿鲸鏆忛柟绗涘棭鏀介悹渚灠缁剁偞瀵煎鍨涘亾濮樺磭绠?w2_weight 閻犱礁娼″Λ鍫曞级閸愵喖娅㈤柨娑樼焷缁绘牠鏌岀仦鎯╅悗鐟板暙閸╁棝宕ュ鍛厒闂佹彃绻愮€垫煡寮堕崘顔兼鐎殿喚濞€閸ｇ儤绋夋繛搴撳亾?
            self.w2_weight = self._target.w2_qweight
        else:
            # ------------------------------- 闁革负鍔戝顏堟煂韫囨挸顕ч悹渚灠缁剁偞绋夌€ｎ偄惟闁圭瑳鍡╂斀闁哄鍟撮崳绋款嚕閻樿娅ら梺鎻掔Т閻ｉ箖宕ラ幋婵嗙厒 burst pool -------------------------------
            # 閻?w13 dense 闁哄鍟撮崳鎼佹煂瀹ュ懐鏆伴柛姘灥閸?burst pool 闁汇劌瀚径宥夊籍鐠鸿櫣鐐婇梺鎻掔箞閳?
            self.w13_weight = self._target.w13_weight

            # 閻?w2 dense 闁哄鍟撮崳鎼佹煂瀹ュ懐鏆伴柛姘灥閸?burst pool 闁汇劌瀚径宥夊籍鐠鸿櫣鐐婇梺鎻掔箞閳?
            self.w2_weight = self._target.w2_weight

    @property
    def expert_map(self) -> torch.Tensor:
        # ------------------------------- 閺夆晜鏌ㄥú鏍亹閹惧啿顤?burst pool 濞达綀娉曢弫銈夋儍閸曨亜顦查柡?expert 闁哄嫮濮撮惃鐘垫偘?-------------------------------
        # 閺夆晜鍔橀、鎴﹀籍?kernel 閻犲洩顕цぐ鍥儍?expert_map 閹煎瓨鏌ㄧ紞瀣级閵夈劌娈扮憸鐗堟尭婢?burst pool 闁汇劌瀚径宥夊籍閼稿灚衼閻忓繐瀚妴鍐Υ?
        return self._expert_map

    def __getattr__(self, name: str) -> Any:
        # ------------------------------- 閻忓繐妫欏﹢顓㈠及閹呯閻熸洖妫涘ú濠囨儍閸曨偆娼ｉ柟顑懐鍩犲☉鎾亾闂侇偄绻嬬槐鍓佺磼濞嗗繐鏂у┑?FusedMoE 閻?-------------------------------
        # 鐟滅増鎸烽崬顒勬偠閸℃婀撮柤濂変海闂娾晛鈻介埄鍐╃畳閻犲洢鍎遍惈姗€骞€瑜庡鍌炴晬鐏炶姤绀€闂侇偀鍋撻柛鎺撴緲鐢偅鎱ㄧ€ｎ亞鍞ㄧ痪顓涘亾閻忕偛鍊风粭鍌滅磼瑜忛悽濠氬蓟閵夛箑顥濋柕?
        return getattr(self._base_layer, name)


class SharedPrefillBurstPool:
    """Shared pool for prefill MoE batch staging."""

    def __init__(self, template_layer: FusedMoE, num_slots: int) -> None:
        # ------------------------------- 闁哄稄绻濋悰?burst pool 婵″弶鍨濈紞鍛村极閺夎儻瀚欓柛鎺撶箓椤劙宕犻弽褏鍞ㄧ痪顓涘亾闁稿繐鍟弳鐔煎箲?-------------------------------
        # 闁稿繐褰夐棅?prefill burst pool 闁煎嘲鍟块惃顖炴閳ь剛鎲?1 濞戞搩浜欐径宥夊籍閼告晫顐ｆ媴瀹ュ繒绀夐柛姘剧畱閸垰鈻介埄鍐╃畳閻庡湱鍋ゅ顖炲箛韫囧海鐤呴柕?
        if num_slots <= 0:
            raise ValueError("Shared prefill burst pool requires a positive slot count")

        # 閻犱焦婢樼紞宥堛亹閹惧啿顤?burst pool 闁告瑯鍨伴鎰棯瀹曞洦鐣卞☉鎾崇摠濡炲倸危閹存帞绉撮柡浣峰嵆閸ｆ椽濡?
        self.num_slots = int(num_slots)

        # 閻犱焦婢樼紞宥呂熼埄鍐╃凡閻忕偛鍊搁顔芥償閺冨倹鐣遍柛蹇嬪妼閻剚绋夐幘鎰佸晙闁诡剛绮弳鐔兼晬鐏炶姤鍊电紓渚囧幖椤︽煡鎮介妸銉﹀€卞☉鎾亾濠靛倹顨呴崣蹇曚沪閳?expert id 閻犲浂鍘虹粻鐔煎Υ?
        self.global_num_experts = template_layer.global_num_experts

        # 閻犱焦婢樼紞宥呂熼埄鍐╃凡閻忕偛鍊搁幃鏇犵矓鐢喚绀夊☉鎾存椤╋箓鎮介妸銈囪壘闁哄啨鍎辩换鏃€娼忛幘鍐叉瘔濞戞挸姘﹂惃鐔烘嫚閺囩偟鏆板ù锝呯Р閳?
        self.layer_name = template_layer.layer_name

        # 闁稿繐褰夐棅鈺佇ч悩鍙夊€卞☉鎾亾闁哄啳娉涢崺銏ゅ矗椤忓嫬甯掗悹渚€鏅茬粩瀛樼▔?layer 闁?request 濞达綀娉曢弫銈夋晬瀹€鍕╂慨婵愭線婢跺秹寮捄铏圭倞闂佹彃绻楅～锕傜嵁鐠哄搫绲洪悷鏇炴濞插﹪濡?
        self._busy = False
        # 閻犱焦婢樼紞宥嗙▔婵犱胶顏遍弶鐑嗗枟瑜颁焦绂嶉妶鍛厒鐟滅増鎸告晶鐘诲礂閸欐﹢鐓╂慨鍦Х濞?GPU 鐎规悶鍎扮紞鏃傗偓鐟版湰閸ㄦ碍绂嶇€ｂ晜顐介柨娑欑☉椤︽煡鎮介妸銉ヮ枀闂傚洠鍋撻悷鏇氳兌閻℃垵顕ラ崨顓犳殜閻庣懓鏈崹姘舵晬?
        # 闁告熬绠戦崹顖炲触鎼达絿鏁鹃悘鐐插€歌ぐ鏌ユ嚄閽樺韬柛鎾崇С缁斿浠﹂崒娆戠煗婵炴垵鐗愰崹?burst 鐎殿喚濞€閸ｆ椽寮懜闈涚倒闁告挸绉烽々顐﹀礃濞嗘帞绠瑰ù婊勭〒缁憋箓宕橀幓鎺戦殬闁?
        self._last_use_event: torch.cuda.Event | None = None

        # ------------------------------- 闁圭顦拌啯闁哄鐏濋惇浼存儍閸曨垰娅ら柛鏍ㄧ墬鑶╃€殿喖绻愰崹搴ㄦ煀瀹ュ嫬顦查柡鍐煐婢х晫鎮扮仦鐣岀倞闂?-------------------------------
        # 鐟滅増鎸昏啯闁哄鐏濋惇浼存煂閸モ晜鏆?GPTQ Marlin 闂佹彃绻愮€佃尙鎹勯姘辩獮闁哄啳顔愮槐婵嬫閳ь剛鎲版担鐤闂佹彃绻愮€垫煡寮堕崘顔兼闁靛棔澶焎ale 濞戞挸姘︾欢鐔煎礉閳哄啫鍋嶇€殿喗娲熼崗姗€宕欓崱妤婃У burst 鐎殿喚濞€閸ｆ椽濡?
        if isinstance(template_layer.quant_method, GPTQMarlinMoEMethod):
            # 閻犱焦婢樼紞宥堛亹閹惧啿顤?burst pool 鐎规悶鍎扮紞鏃堝捶?GPTQ Marlin 婵☆垪鈧磭纭€濞戞挸顑冮埀?
            self._mode = "gptq_marlin"

            # 濞?w13 闂佹彃绻愮€垫煡寮堕崘顔兼闁告帒妫濋崢銈夊箰?slot 缂備礁瀚划鎰版儍閸曨亜顦查柡鍐硾缁卞爼鏌岃箛瀣у亾?
            self.w13_qweight = torch.empty(
                (self.num_slots, *template_layer.w13_qweight.shape[1:]),
                dtype=template_layer.w13_qweight.dtype,
                device=template_layer.w13_qweight.device,
            )

            # 濞?w2 闂佹彃绻愮€垫煡寮堕崘顔兼闁告帒妫濋崢銈夊箰?slot 缂備礁瀚划鎰版儍閸曨亜顦查柡鍐硾缁卞爼鏌岃箛瀣у亾?
            self.w2_qweight = torch.empty(
                (self.num_slots, *template_layer.w2_qweight.shape[1:]),
                dtype=template_layer.w2_qweight.dtype,
                device=template_layer.w2_qweight.device,
            )

            # 濞?w13 闁?scale 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋闁?slot 缂備礁瀚划鎰版儍閸曨亜顦查柡鍐硾缁卞爼鏌岃箛瀣у亾?
            self.w13_scales = torch.empty(
                (self.num_slots, *template_layer.w13_scales.shape[1:]),
                dtype=template_layer.w13_scales.dtype,
                device=template_layer.w13_scales.device,
            )

            # 濞?w2 闁?scale 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋闁?slot 缂備礁瀚划鎰版儍閸曨亜顦查柡鍐硾缁卞爼鏌岃箛瀣у亾?
            self.w2_scales = torch.empty(
                (self.num_slots, *template_layer.w2_scales.shape[1:]),
                dtype=template_layer.w2_scales.dtype,
                device=template_layer.w2_scales.device,
            )

            # 濞?w13 闁?qzeros 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋闁?slot 缂備礁瀚划鎰版儍閸曨亜顦查柡鍐硾缁卞爼鏌岃箛瀣у亾?
            self.w13_qzeros = torch.empty(
                (self.num_slots, *template_layer.w13_qzeros.shape[1:]),
                dtype=template_layer.w13_qzeros.dtype,
                device=template_layer.w13_qzeros.device,
            )

            # 濞?w2 闁?qzeros 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋闁?slot 缂備礁瀚划鎰版儍閸曨亜顦查柡鍐硾缁卞爼鏌岃箛瀣у亾?
            self.w2_qzeros = torch.empty(
                (self.num_slots, *template_layer.w2_qzeros.shape[1:]),
                dtype=template_layer.w2_qzeros.dtype,
                device=template_layer.w2_qzeros.device,
            )

            # 濞?w13 闁?g_idx 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋闁?slot 缂備礁瀚划鎰版儍閸曨亜顦查柡鍐硾缁卞爼鏌岃箛瀣у亾?
            self.w13_g_idx = torch.empty(
                (self.num_slots, *template_layer.w13_g_idx.shape[1:]),
                dtype=template_layer.w13_g_idx.dtype,
                device=template_layer.w13_g_idx.device,
            )

            # 濞?w2 闁?g_idx 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋闁?slot 缂備礁瀚划鎰版儍閸曨亜顦查柡鍐硾缁卞爼鏌岃箛瀣у亾?
            self.w2_g_idx = torch.empty(
                (self.num_slots, *template_layer.w2_g_idx.shape[1:]),
                dtype=template_layer.w2_g_idx.dtype,
                device=template_layer.w2_g_idx.device,
            )

            # 濞?w13 闁?g_idx_sort_indices 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋闁?slot 缂備礁瀚划鎰版儍閸曨亜顦查柡鍐硾缁卞爼鏌岃箛瀣у亾?
            self.w13_g_idx_sort_indices = torch.empty(
                (self.num_slots, *template_layer.w13_g_idx_sort_indices.shape[1:]),
                dtype=template_layer.w13_g_idx_sort_indices.dtype,
                device=template_layer.w13_g_idx_sort_indices.device,
            )

            # 濞?w2 闁?g_idx_sort_indices 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋闁?slot 缂備礁瀚划鎰版儍閸曨亜顦查柡鍐硾缁卞爼鏌岃箛瀣у亾?
            self.w2_g_idx_sort_indices = torch.empty(
                (self.num_slots, *template_layer.w2_g_idx_sort_indices.shape[1:]),
                dtype=template_layer.w2_g_idx_sort_indices.dtype,
                device=template_layer.w2_g_idx_sort_indices.device,
            )

            # 閻犱焦婢樼紞?expert 闁哄嫮濮撮惃鐘垫偘閵娿儳瀹夐柡鈧崜褏鏋傞柣銊ュ椤旀洘寰勯崶椋庣婵炲矁娉曢弫銈夋煂韫囨挸顕ч柡澶婂暣閸ｆ悂骞嶉埀顒勫捶閵婎煈鍟庡璺烘储閳?
            expert_map_device = template_layer.w13_qweight.device

        # 鐟滅増鎸昏啯闁哄鐏濋惇浼存煂閸モ晜鏆忛梻鍫㈠仱閸ｆ椽宕犻弽顒傜唴鐎垫澘瀚鍌炴晬鐏炶棄娑ч梻鍥ｅ亾閻熸洑绀侀崳顖涘緞閸ヮ亞绠ラ悶娑樻湰濡炲倿鎯勭€涙ê澶嶆繛鎴濈墣閸ㄥ倿鎯?w13 闁?w2 闁哄鍟撮崳绋款嚕閻樿娅ら柕?
        elif isinstance(template_layer.quant_method, UnquantizedFusedMoEMethod):
            # 閻犱焦婢樼紞宥堛亹閹惧啿顤?burst pool 鐎规悶鍎扮紞鏃堝捶閵娾晜濮滈梺鎻掔箰鐎垫彃螣閳ュ磭纭€濞戞挸顑冮埀?
            self._mode = "unquantized"

            # 濞?w13 闁哄鍟撮崳鎼佸礆閸℃稑甯抽柟?slot 缂備礁瀚划鎰版儍閸曨亜顦查柡鍐硾缁卞爼鏌岃箛瀣у亾?
            self.w13_weight = torch.empty(
                (self.num_slots, *template_layer.w13_weight.shape[1:]),
                dtype=template_layer.w13_weight.dtype,
                device=template_layer.w13_weight.device,
            )

            # 濞?w2 闁哄鍟撮崳鎼佸礆閸℃稑甯抽柟?slot 缂備礁瀚划鎰版儍閸曨亜顦查柡鍐硾缁卞爼鏌岃箛瀣у亾?
            self.w2_weight = torch.empty(
                (self.num_slots, *template_layer.w2_weight.shape[1:]),
                dtype=template_layer.w2_weight.dtype,
                device=template_layer.w2_weight.device,
            )

            # 閻犱焦婢樼紞?expert 闁哄嫮濮撮惃鐘垫偘閵娿儳瀹夐柡鈧崜褏鏋傞柣銊ュ椤旀洘寰勯崶椋庣婵炲矁娉曢弫銈夋閻愬搫娅ら柛鏍ㄧ墬濞煎牓鏌屽鍡楊暡闁革负鍔忛鏇熷緞閸ャ儮鍋?
            expert_map_device = template_layer.w13_weight.device
        else:
            # 鐟滅増鎸告晶鐘诲礂閸欐﹢鐓?burst pool 濞寸姴鎳忛弫顕€骞?GPTQ Marlin 闁告粌鐭傚顏堟煂韫囨挸顕у☉鎾卞€楃悮?FusedMoE 閻犱警鍨扮欢鐐哄Υ?
            raise TypeError(
                "Shared prefill burst pool only supports GPTQMarlinMoEMethod "
                "and UnquantizedFusedMoEMethod"
            )

        # ------------------------------- 闁哄瀚伴埀顒傚Т閸欏繒浠﹂埀?expert 闁?burst slot 闁汇劌瀚Σ褏浜搁崟顔衡偓鍐╃▔鎼淬垹鈷旈悶娑樺閸烆剟鎮堕崱妤冩勾 -------------------------------
        # 濞戞挸鎼紞瀣礈瀹ュ懎褰欏ù?burst pool 闁告垵妫楅ˇ顒佺▔閳ь剙顕?闁稿繈鍔岄惇?expert id -> burst slot id"闁汇劌瀚Σ褏浜搁崟顔衡偓鍐Υ?
        self._expert_map = torch.full(
            (self.global_num_experts,),
            -1,
            dtype=torch.int32,
            device=expert_map_device,
        )

        # 闁糕晞妗ㄧ花顒€螣閳╁啯绶查悘鐐插€归悗顖炴焻閻樿京顏卞☉鎿冧海娴溿倝鏌岃箛鏇㈢崜闁圭瑳鍡╂斀濞寸媴绲块幃濠勪沪閸岋妇绀夐柟?kernel 闁稿繈鍎辫ぐ娑㈡煂瀹ュ懐鏆伴柛姘灥閸?burst pool 濞戞挸顭堥埀?
        self._execution_layer = _PrefillBurstExecutionLayer(
            base_layer=template_layer,
            target=self,
            expert_map=self._expert_map,
            num_slots=self.num_slots,
        )

        # ------------------------------- 闁瑰灚鎸稿畵?burst pool 闁告帗绻傞～鎰板礌閺嵮呮殮闁瑰瓨鍔栧Λ鈺勭疀?-------------------------------
        # 閺夊牊鎸搁崵顓°亹閹惧啿顤呴柛蹇撳綁闂?burst pool 闁汇劌瀚惇浼村触瀹ュ啠鍋撴担榧撲礁顕ｈ箛瀣у亾娴ｅ墣顐ｆ媴瀹ュ棙娈跺☉鎾冲濡鈧稒锚瀹曚即鎮介妸銉ｄ海閻忓繐绻堥埀?
        logger.info(
            "Initialized shared prefill burst pool: layer=%s mode=%s slots=%d bytes=%.2f MiB",
            template_layer.layer_name,
            self._mode,
            self.num_slots,
            self.nbytes / (1 << 20),
        )

    def _supports_reuse_events(self) -> bool:
        return self._expert_map.device.type == "cuda"

    def _wait_for_previous_use(self) -> None:
        if self._last_use_event is None or not self._supports_reuse_events():
            return
        current_stream().wait_event(self._last_use_event)

    def _record_current_use(self) -> None:
        if not self._supports_reuse_events():
            return
        event = torch.cuda.Event()
        event.record(current_stream())
        self._last_use_event = event

    def prepare(
            self,
            controller: LayerTieredExpertCacheController,
            topk_ids: torch.Tensor,
    ) -> _PrefillBurstExecutionStats:
        if self._busy:
            raise RuntimeError(
                f"{controller.layer_key}: shared prefill burst pool is already in use"
            )

        requested = controller._unique_experts(topk_ids)
        if len(requested) > self.num_slots:
            raise RuntimeError(
                f"{controller.layer_key}: requested {len(requested)} experts but only "
                f"{self.num_slots} burst slots are available"
            )

        self._busy = True
        try:
            self._wait_for_previous_use()
            self._expert_map.fill_(-1)
            stats = controller._populate_prefill_burst_pool(self, requested)
            for slot, expert_id in enumerate(requested):
                self._expert_map[expert_id] = slot
            self._execution_layer.bind(
                base_layer=controller.layer,
                expert_map=self._expert_map,
                num_slots=self.num_slots,
            )
            return stats
        except Exception:
            self._busy = False
            raise

    def release(self, *, record_use: bool) -> None:
        try:
            if record_use:
                self._record_current_use()
        finally:
            self._busy = False

    @property
    def nbytes(self) -> int:
        # ------------------------------- 缂備胶鍠曢姝屻亹閹惧啿顤?burst pool 闁汇劌瀚埀顒冾嚙閻⊙囨嚍閸屾艾绐楅柣?-------------------------------
        # 鐟滅増鎸告导鎰媴濠婂啯韬?GPTQ Marlin 婵☆垪鈧磭纭€闁哄啳顔愮槐婵嬫閳ь剛鎲版担鐟拔╅柟纰樺亾闁哄牆顦径宥夊籍閸洖娅ら柛鏍ㄧ墪缁卞爼鏌岃箛娑樺幋缂佹儳鍟块崣鍡欑磼閻旀椿鍚€闁?
        if self._mode == "gptq_marlin":
            tensors = (
                self.w13_qweight,
                self.w2_qweight,
                self.w13_scales,
                self.w2_scales,
                self.w13_qzeros,
                self.w2_qzeros,
                self.w13_g_idx,
                self.w2_g_idx,
                self.w13_g_idx_sort_indices,
                self.w2_g_idx_sort_indices,
            )
        else:
            # 鐟滅増鎸告导鎰媴濠婂啯韬梻鍫㈠仱閸ｆ椽宕犻弽顭屼礁顕ｈ箛鏃€顦ч柨娑樿嫰瑜把呯磼閻旀椿鍚€ w13 闁?w2 濞戞挶鍊曞鈩冪▔鐎涙ɑ顦ч柡澶婂暣閸ｇ顕ｉ悩璇叉闁?
            tensors = (self.w13_weight, self.w2_weight)

        # 閻庝絻顫夋晶宥夊嫉婢跺顦查柡鍐硾缁卞爼鏌岃箛娑掑亾閹邦亪鍤嬮柟?闁稿繐鍟扮粈宀勫极?* 闁告娲栭崢鎾舵閻樿尙鎽熼柤鍝勫€归弳?婵懓鍊搁幏浼存晬鐏炵晫绻侀柛鎺斿閳ь剝顕у畷浼存偨閵娿儳鎽熼柤鍝勫€归弳鐔煎Υ?
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def supports_layer(self, layer: FusedMoE) -> bool:
        # ------------------------------- 闁告帇鍊栭弻鍥亹閹惧啿顤呴柛蹇撳綁闂?burst pool 闁哄嫷鍨伴幆渚€宕ｉ婵愭蕉闁圭娲ら悾鍓т沪閸屾凹妲婚柣?-------------------------------
        # 鐟滅増鎸惧ú浼村冀閸パ呮勾闁汇劌瀚崣蹇曚沪閳ь剚绋夐幘鎰佸晙闁诡剛绮弳鐔哥▔鎼达紕绉奸柛?burst pool 濞戞挸绉崇粩鎾嚊鐎涙ɑ顦ч柨娑樼焷椤曗晠寮?top-k expert id 閻犲浂鍘虹粻鐔哥▔瀹ュ懏鍊遍柨娑樺缁楀鎳楅挊澶樻Щ闁活潿鍔婇埀?
        if layer.global_num_experts != self.global_num_experts:
            return False

        # 鐟滅増鎸告导鎰媴濠婂啯韬?GPTQ Marlin 婵☆垪鈧磭纭€闁哄啳顔愮槐婵嬫閳ь剛鎲版笟鈧·鍌涘緞閺嶃劎澧″Δ鐘叉湰婢у秹寮垫径瀣闊洤鍟撮崳娲礌閺嵮呯倞闂佹彃绻掑▓鎴ｃ亹閵忋垹笑闁稿繒鍘ч鎰板箑瑜嬮埀?
        if self._mode == "gptq_marlin":
            return (
                    isinstance(layer.quant_method, GPTQMarlinMoEMethod)
                    and layer.w13_qweight.shape[1:] == self.w13_qweight.shape[1:]
                    and layer.w2_qweight.shape[1:] == self.w2_qweight.shape[1:]
                    and layer.w13_scales.shape[1:] == self.w13_scales.shape[1:]
                    and layer.w2_scales.shape[1:] == self.w2_scales.shape[1:]
                    and layer.w13_qzeros.shape[1:] == self.w13_qzeros.shape[1:]
                    and layer.w2_qzeros.shape[1:] == self.w2_qzeros.shape[1:]
            )

        # 鐟滅増鎸告导鎰媴濠婂啯韬梻鍫㈠仱閸ｆ椽宕犻弽顭屼礁顕ｈ箛鏃€顦ч柨娑樺缁酣妫侀埀顒勫冀閿熺姷宕?w13 闁?w2 闁哄鍟撮崳绋款嚕閻樿娅ら柣銊ュ閼镐即鎮╃捄鍝勬倯閻庣顫夐埀顑讲鍋?
        return (
                isinstance(layer.quant_method, UnquantizedFusedMoEMethod)
                and layer.w13_weight.shape[1:] == self.w13_weight.shape[1:]
                and layer.w2_weight.shape[1:] == self.w2_weight.shape[1:]
        )

    def execute(
            self,
            controller: LayerTieredExpertCacheController,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        stats = self.prepare(controller, topk_ids)
        try:
            # 闁瑰灚鎸稿畵鍐亹閹惧啿顤?burst 闁圭瑳鍡╂斀闁汇劌瀚弳鐔煎箲椤旇姤闄嶆繝褎鍔曢幊鈩冪▔椤撶喎鍓伴柛鎰暜缁辨繈宕犻崨顔碱仾 resident 闁告稒鍨濋懙鎴﹀Υ娑撳摉U 闁告稒鍨濋懙鎴炵▔?NVMe 闁告梻濮惧ù鍥р枎閳╁啯娈堕柕?
            logger.debug(
                "Tiered MoE prefill burst execution: layer=%s unique=%d slots=%d "
                "resident_hits=%d cpu_hits=%d nvme_loads=%d",
                controller.layer_key,
                int(torch.unique(topk_ids.detach()).numel()),
                self.num_slots,
                stats.resident_hits,
                stats.cpu_hits,
                stats.nvme_loads,
            )

            # 闂侇偅淇虹换鍐亹閹惧啿顤呴柟璨夊啫鐓戦柛锝冨妿濞堟垿鏌岃箛鎾愁嚙闁哄倽顫夌涵鍫曟晬鐏炴儳惟闁圭瑳鍡╂斀闁稿繈鍎辫ぐ娑㈡儎鐎涙ê澶嶉梺鎻掔Т閻ｉ箖宕ラ幋婵嗙厒 burst proxy layer 濞戞挸锕ら悾顒勫箣閹邦剛鏉介梻鍕嚀椤撳摜绮诲灏栧亾?
            result = controller.quant_method.apply(
                layer=self._execution_layer,
                x=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=shared_experts_input,
            )
            self.release(record_use=True)
            return result
        finally:
            # ------------------------------- 闂佹彃锕ラ弬渚€宕楅崣姗€鐓?burst pool 闁汇劌瀚畷浼存偨閵婏妇鍨奸悹?-------------------------------
            # 闁哄啰濮鹃鎴﹀嫉椤掍緡鍋ч柟绗涘棭鏀介柟瀛樺姇婵稒娼诲Ο缁樞﹀鎯扮簿鐟欙箓鏁嶅畝鍕幋闊洤鎳橀妴蹇撱€掗崨瀛樼彑 busy 闁哄秴娲╅鍥晬鐏炶棄甯掗悹浣侯焾閹绱掗鈽嗗殲婵懓鍊婚幋椋庣磼椤撶儐妲婚柣顫姀椤曟岸宕楅崣姗€鐓╂慨鍦О閳?
            if self._busy:
                self.release(record_use=False)


class SharedRuntimeExpertStagePool:
    """Model-wide reusable CPU/GPU expert-major staging storage."""

    def __init__(self) -> None:
        self._cpu_storage: torch.Tensor | None = None
        self._cpu_capacity = 0
        self._cpu_signature: tuple[Any, ...] | None = None
        self._cpu_lock = threading.Lock()

        self._gpu_storage: torch.Tensor | None = None
        self._gpu_capacity = 0
        self._gpu_signature: tuple[Any, ...] | None = None
        self._gpu_device: torch.device | None = None
        self._gpu_lock = threading.Lock()
        self._copy_executor: ThreadPoolExecutor | None = None
        self._copy_executor_workers = 0
        self._copy_executor_lock = threading.Lock()

    def get_cpu_storage(
            self,
            *,
            capacity: int,
            per_expert_bytes: int,
            pin_memory: bool,
            signature: tuple[Any, ...],
    ) -> torch.Tensor:
        required_capacity = max(1, int(capacity))
        required_numel = required_capacity * int(per_expert_bytes)
        with self._cpu_lock:
            if (
                self._cpu_storage is not None
                and self._cpu_signature == signature
                and self._cpu_capacity >= required_capacity
                and (not pin_memory or self._cpu_storage.is_pinned())
            ):
                return self._cpu_storage

            kwargs: dict[str, Any] = {
                "device": "cpu",
                "dtype": torch.uint8,
            }
            if pin_memory:
                kwargs["pin_memory"] = True
            try:
                if pin_memory:
                    gc.collect()
                    _empty_torch_host_allocator_cache_best_effort()
                self._cpu_storage = torch.empty(required_numel, **kwargs)
            except Exception as exc:
                if not pin_memory:
                    raise
                raise RuntimeError(
                    "Failed to allocate pinned runtime expert stage: "
                    f"capacity={required_capacity} "
                    f"bytes={required_numel / (1 << 20):.2f} MiB"
                ) from exc
            self._cpu_capacity = required_capacity
            self._cpu_signature = signature
            return self._cpu_storage

    def get_gpu_storage(
            self,
            *,
            capacity: int,
            required_numel: int,
            per_expert_numel: int,
            dtype: torch.dtype,
            device: torch.device,
    ) -> torch.Tensor:
        required_capacity = max(1, int(capacity))
        required_storage_numel = max(
            int(required_numel),
            required_capacity * int(per_expert_numel),
        )
        signature = (dtype, int(per_expert_numel))
        with self._gpu_lock:
            if (
                self._gpu_storage is not None
                and self._gpu_signature == signature
                and self._gpu_capacity >= required_storage_numel
                and self._gpu_device == device
            ):
                return self._gpu_storage

            self._gpu_storage = torch.empty(
                required_storage_numel,
                dtype=dtype,
                device=device,
            )
            self._gpu_capacity = required_storage_numel
            self._gpu_signature = signature
            self._gpu_device = device
            return self._gpu_storage

    def preallocate(
            self,
            *,
            capacity: int,
            per_expert_bytes: int,
            pin_memory: bool,
            cpu_signature: tuple[Any, ...],
            device: torch.device,
    ) -> dict[str, Any]:
        cpu_storage = self.get_cpu_storage(
            capacity=capacity,
            per_expert_bytes=per_expert_bytes,
            pin_memory=pin_memory,
            signature=cpu_signature,
        )
        gpu_storage = self.get_gpu_storage(
            capacity=capacity,
            required_numel=capacity * per_expert_bytes,
            per_expert_numel=per_expert_bytes,
            dtype=torch.uint8,
            device=device,
        )
        return {
            "capacity": int(capacity),
            "per_expert_bytes": int(per_expert_bytes),
            "cpu_bytes": int(cpu_storage.numel() * cpu_storage.element_size()),
            "gpu_bytes": int(gpu_storage.numel() * gpu_storage.element_size()),
            "cpu_pinned": bool(cpu_storage.is_pinned()),
            "device": str(device),
        }

    def map_cpu_stage_copies(
            self,
            copy_fn: Any,
            items: list[Any],
            *,
            max_workers: int,
    ) -> None:
        if not items:
            return
        workers = max(1, min(int(max_workers), len(items)))
        if workers <= 1:
            for item in items:
                copy_fn(item)
            return

        with self._copy_executor_lock:
            if (
                self._copy_executor is None
                or self._copy_executor_workers < workers
            ):
                if self._copy_executor is not None:
                    self._copy_executor.shutdown(wait=True)
                self._copy_executor = ThreadPoolExecutor(max_workers=workers)
                self._copy_executor_workers = workers
            executor = self._copy_executor

        def _copy_in_inference_mode(item: Any) -> None:
            with torch.inference_mode():
                copy_fn(item)

        list(executor.map(_copy_in_inference_mode, items))


class LayerTieredExpertCacheController:
    """Per-layer GPU/CPU tiered cache controller for routed experts."""

    def __init__(
            self,
            layer: FusedMoE,
            plan: dict[str, Any],
            expert_store: SafetensorExpertStore,
            prefill_burst_pool: SharedPrefillBurstPool | None = None,
    ) -> None:
        # ------------------------------- 闁哄稄绻濋悰娆掋亹閹惧啿顤呴悘鐐插€瑰Σ鎼佸触閿曗偓閸戔€愁嚈閾忓湱褰岄柛蹇嬪妼閻?expert 闁告帞澧楀﹢浼村捶?slot 闁汇劌瀚Σ褏浜?-------------------------------
        # Tiered MoE cache 濞撴碍绻嗙粋?FusedMoE 濡澘瀚崢娑㈠几閸曨垪鍋撻悩灞傚仺闁汇劌瀚崣蹇曚沪閳?expert 闁告帞澧楀﹢浼村捶?slot 闁哄嫮濮撮惃鐘垫偘閵婏絺鍋?
        if layer._expert_map is None:
            raise ValueError("Tiered MoE cache requires a global-to-local expert map")

        # ------------------------------- 濞ｅ洦绻傞悺銊╁箳瑜嶉崺妤呭闯閵娧呮嫧閻庤姘ㄥ▓鎴﹀冀缁嬭法濡囬悗鐢殿攰閽栧嫭绋夋惔锛勫敤缁绢厸鍋撻柛蹇撳暞閺嗙喖骞?-------------------------------
        # 濞ｅ洦绻傞悺銊ㄣ亹閹惧啿顤呴柟璨夊啫鐓戦柛锝冨妿缁妇鈧姘ㄥ▓?FusedMoE 閻忕偛鍊搁顔炬寬鎺抽埀?
        self.layer = layer

        # 濞ｅ洦绻傞悺銊╁触椤栨艾袟闁哄牏鍠曢～澶愬礆閹烘垶鐝ら柣銏㈠枑閸ㄦ岸鎯冮崟顕呭悁闁告帗甯掗悺褔宕楅幖鐐╁亾?
        self.plan = plan

        # 濞ｅ洦绻傞悺銊︾▔閹炬剚鍟€閻庢稒锚閸嬪秶鈧數顢婇挅鍕晬鐏炶姤鍊电紓渚囧幖閸犲孩绋夐幘鎰佸晙闁哄鍟撮崳鎼佹焾娴犲鍋撳宕囩畺閻庣懓鍟€垫粓妫侀埀顒傛嫚鐠囨彃绲块柕?
        self.expert_store = expert_store

        # 濞ｅ洦绻傞悺銊╁礂閸欐﹢鐓?prefill burst pool闁挎稒绋栫€氥垻鈧稒锚濠€顏堟晬鐏炶棄鐏熼柛娆樺灣閺併倖绂嶆惔掳浜?prefill batch 闁汇劌瀚径宥夊籍閼搁潧鈷旈悶娑樼焷閻儳顕ラ崟鈹惧亾?
        self.prefill_burst_pool = prefill_burst_pool

        # 濞ｅ洦绻傞悺銊ㄣ亹閹惧啿顤呴悘鐐插€诲▓鎴犱沪閸屾粓鐛撳☉鎾愁煼閺侇參鏁嶇仦鑲╄繑闁哄牜鍓欏﹢瀵糕偓娑櫭崑宥夊蓟閵夘煈鍤勫☉鎾冲濡晞绠涘Δ鍕炕闁告垼妗ㄦ繛鍥偨閵婏絺鍋?
        self.layer_key = layer.layer_name

        # 濞ｅ洦绻傞悺銊ㄣ亹閹惧啿顤呴悘鐐插€搁顔芥償閺冨倹鐣遍梺鎻掔箰鐎垫煡寮憴鍕€婇悗鐢殿攰閽栧嫰鏁嶇仦鑺ュ€电紓渚囧幘閺併倖绂嶆惔鈥虫瀫閻庤纰嶅鍫ユ煂瀹ュ懎鏅搁柛銉у仒缁楀矂骞嶈椤㈡垹鎹勯姘辩獮闁?
        self.quant_method = layer.quant_method

        # resident GPU slot 闁轰焦澹嗛悺鎴炵鎼达紕绉奸柛?FusedMoE 閻忕偛鍊稿﹢?CFIE 閻犱警鍨扮欢鐐寸▔鐎ｎ亞鏉介梻鍕噹閸ㄥ崬顕欓搹瑙勭暠闁哄牜鍓欏﹢?expert 闁轰浇鍩囬埀?
        self.num_slots = layer.local_num_experts
        self._compute_slots = 8

        # 闁告帗绻傞～鎰板礌閺嶎厸鍋撻弰蹇曞竼闁哄啫鐖煎Λ鍨潰閵夘煈鍚€闁轰焦婢樺▍鎺楁晬鐏炶偐鐭岄柣顫妺缁剛鎷犳繝鍐╃劷闁哄啨鍎辩换鏃堝Υ?
        self._step = 0

        # 闁告帗绻傞～鎰板礌?resident GPU slot 闁告帗婢橀崣蹇曚沪閳?expert 闁汇劌瀚浠嬪触閹寸偞衼閻忓繐瀚妴鍐Υ?
        self._slot_to_global = [-1] * self.num_slots
        self._expert_to_slot = [-1] * int(layer.global_num_experts)
        self._victim_cursor = 0

        # ------------------------------- 闁告帗绻傞～鎰板礌閺嶎剛绠ラ悶娑樻湰濠€锛勭磼閻旀椿鍚€闁圭娲﹂悥?-------------------------------
        # 閻犱焦婢樼紞宥夊箳瑜嶉崺妤呭闯閵娧呮焾閻犱讲鍓濇晶鐣屾偘瀹€鍐畺闁汇劌瀚粭鎾垛偓纭呮硾婵偞娼懞銉仹闁轰浇鍩囬埀?
        self._total_loads = 0

        # 閻犱焦婢樼紞宥夊川閹存帟鍘?CPU 閻㈩垱鎮傞埞妤冪磽閹惧磭鎽犻柣銊ュ椤愬ジ寮懜顑藉亾?
        self._cpu_hits = 0

        # 閻犱焦婢樼紞宥団偓鍦仱濡绢垱绂掓惔鈥虫瀻閻庢稒锚閸嬪秹宕濋悩鐑樼グ濞戞挻鎸搁宥夋儍閸曨剦鍋ч柡浣藉焽閳?
        self._nvme_loads = 0

        # 閻犱焦婢樼紞宥夊矗閹寸姵鏅?expert 濡炲綊浜堕埀顒佸姉濞堟垵鈻庨埄鍐╂闁?
        self._evictions = 0

        # 鐟滅増鎸告晶鐘诲箳瑜嶉崺妤呭闯閵娿倕鈻忛柣顫妿濞堟垿寮堕崘顔兼婵☆垪鈧磭纭€闁哄秴娲╅惁鎴︽晬瀹€鈧埣銏ゅ触鎼存繄绐楅柡宥堫潐瀹?quant_method 閺夆晜绋栭、鎴犳導鐎ｎ亖鍋撶粭琛″亾?
        self._mode: str

        # ------------------------------- 闁告帗绻傞～鎰板礌?CPU 濞撴皜鍛閻庢稒眉缁楀瞼绱撻幘鍐叉毐闁告牞娅ｆ慨鎼佸箑?-------------------------------
        # 妤犵偛鍟胯ぐ鎾绩椤栨稑鐦?pinned memory 闁哄啫澧庨弫銈嗙?stage buffer; CPU static mirror 婵ɑ鐡曠换?pageable闁?
        self._use_pinned_cpu_static = bool(
            self.plan.get(
                "use_pinned_cpu_static",
                getattr(self, "_use_pinned_cpu_static", False),
            )
        )
        self._cpu_static_pinned_layers = frozenset(
            int(layer)
            for layer in self.plan.get("cpu_static_pinned_layers", ())
        )
        self._use_pinned_cpu = is_pin_memory_available()
        self._moe_layer_index = _extract_moe_layer_index(str(self.layer_key))
        # 濞ｅ洦绻傞悺銊ヮ啅閼碱剙鈷栭柛鏍ㄧ墪閸?CPU 閻㈩垱鎮傞埞妤呭礃閸涱厾鎽犲☉鎿冨幘濞堟垿妫冨▎鎰ㄥ亾?expert bundle闁?
        self._cpu_static_bundles: dict[int, ExpertBundle] = {}
        self._cpu_static_layer_field_views: dict[str, torch.Tensor] = {}
        self._cpu_static_layer_expert_ids: tuple[int, ...] = ()
        self._cpu_static_layer_expert_index_by_id: dict[int, int] = {}
        self._marlin_ready_cache_disabled = (
            os.getenv("CFIE_MARLIN_READY_CACHE", "1").strip().lower()
            in {"0", "false", "no", "off"}
        )
        self._marlin_ready_cache_root = self._resolve_marlin_ready_cache_root()
        # 濞ｅ洦绻勯弳鈧柡鍐勫懎顣奸悹浣测偓鍐茬亰闁稿繒鍘ч鎰偓娑欘殕椤斿矂鏁嶅☉妯肩Ъ闁告挸绉崇€靛瞼鐥径鍝ヮ伇闁煎壊鍏涚粭澶愬礃瀹ュ嫮璐╅悹褎鐗曞畷鐔兼偑椤掑倹鐣?staging bundle闁?
        self._cpu_stage_bundle: ExpertBundle | None = None

        # 濞ｅ洦绻傞悺銊ф媼閳ュ啿鐏婂☉鎿冨幗鐎垫氨鈧姘ㄥ▓?CPU 閻㈩垱鎮傞埞?expert 闂傚棗妫楅幃搴ㄥΥ?
        self._cpu_static_experts: frozenset[int] = frozenset()

        # 缂備胶鍠曢姝屻亹閹惧啿顤呴柟璨夊啫鐓戦柛锝冨妽鐎垫棃寮垫径灞剧暠闁稿繈鍔戦崕?CPU buffer 闁诡剝顕ч悺褔鎳為崒娑欐闁?
        self._cpu_buffer_bytes = 0

        # 闂佹彃绻愮€佃尙鎹勯姘辩獮濞戞挸顑囬弫銈嗙鎼粹剝韬?CPU 濞撴皜鍕暞閻庢稒锚閼荤喖鏌屽鍥╃煁闁告鍠庨～鎰▔閹炬剚鍟€闁哄鍟撮崳鎼佹儍?raw buffer闁?
        self._cpu_quantized_raw_buffer: _RawExpertWeights | None = None

        # 闂傚牏鍋ら崳娲礌閺嶎剛鐔呯€垫澘瀚粭鍛存偨閵娿倗鑹鹃柛?CPU 濞撴皜鍕暞閻庢稒锚閼荤喖鏌屽鍥╃煁闁告鍠庨～鎰▔閹炬剚鍟€闁哄鍟撮崳鎼佹儍?raw buffer闁?
        self._cpu_unquantized_raw_buffer: _RawUnquantizedExpertWeights | None = None
        self._cpu_runtime_batch_buffers: dict[str, torch.Tensor] = {}
        self._cpu_static_bundle_lock = threading.Lock()
        self._cpu_runtime_batch_buffers_lock = threading.Lock()
        self._runtime_target_field_cache: dict[
            int, tuple[list[tuple[str, torch.Tensor]], torch.device]
        ] = {}
        self._runtime_slot_ids_cpu: torch.Tensor | None = None
        self._runtime_slot_ids_gpu: torch.Tensor | None = None
        self._runtime_slot_ids_gpu_device: torch.device | None = None
        self._runtime_slot_ids_lock = threading.Lock()
        self._cpu_runtime_expert_stage_storage: torch.Tensor | None = None
        self._cpu_runtime_expert_stage_capacity = 0
        self._cpu_runtime_expert_stage_signature: tuple[Any, ...] | None = None
        self._cpu_runtime_expert_stage_lock = threading.Lock()
        self._gpu_runtime_expert_stage_storage: torch.Tensor | None = None
        self._gpu_runtime_expert_stage_capacity = 0
        self._gpu_runtime_expert_stage_signature: tuple[Any, ...] | None = None
        self._gpu_runtime_expert_stage_device: torch.device | None = None
        self._gpu_runtime_expert_stage_lock = threading.Lock()
        self._runtime_stage_pool: SharedRuntimeExpertStagePool | None = None
        # Stage storage is reusable capacity only. Each H2D copy uses the
        # prefix matching the actual number of missing experts in prepare().
        self._runtime_stage_slots = max(1, int(self._compute_slots or 1))
        self._prepare_cpu_copy_threads = max(
            1,
            int(
                self.plan.get(
                    "prepare_cpu_copy_batch_size",
                    0,
                )
                or self.plan.get(
                    "prepare_cpu_copy_threads",
                    DEFAULT_CPU_STATIC_PREPROCESS_BATCH_SIZE_CAP,
                )
                or 1
            ),
        )
        self._prepare_cpu_copy_batch_size = self._prepare_cpu_copy_threads

        # ------------------------------- 閻犱緤绱曢悾?prefill burst 闁汇劌瀚〒鍓佷焊?token 閻熸瑱绠戣ぐ鍌炴⒓閸績鍋?-------------------------------
        # 濞村吋锚閸樻稓鎷犵拠鎻掔悼閻犱讲鈧啿鐏婂☉鎿冨幗濡顕ｈ箛娑樺赋缂傚喚鍠氬▓?burst 闁哄牃鍋撻悘?token 闂傚啫鐗嗛埀顒傤儠閳?
        configured_burst_min_tokens = int(self.plan.get("prefill_burst_min_tokens", 0))

        # 鐟滅増鎹侀鎼佸礆閹烘挻寮撻柡鍕劤缁憋繝鏌婂鍥╂瀭闁哄啳顔愮槐婵嬪箰婢舵劗甯涢悹浣靛€楃划鈩冾殽鐏炶棄褰嗙€殿喖绻楅鍝ョ不?burst 闁哄牃鍋撻悘?token 闂傚啫鐗嗛埀顒傤儠閳?
        self._prefill_burst_min_tokens = (
            configured_burst_min_tokens
            if configured_burst_min_tokens > 0
            else max(
                DEFAULT_PREFILL_BURST_MIN_TOKENS,
                self.num_slots * DEFAULT_PREFILL_BURST_TOKENS_PER_GPU_SLOT,
            )
        )
        self._cpu_static_preprocess_batch_size_cap = max(
            0,
            int(
                self.plan.get(
                    "cpu_static_preprocess_batch_size",
                    DEFAULT_CPU_STATIC_PREPROCESS_BATCH_SIZE_CAP,
                )
                or 0
            ),
        )
        self._cpu_static_preprocess_batch_size = int(
            self.plan.get(
                "cpu_static_preprocess_batch_size",
                DEFAULT_CPU_STATIC_PREPROCESS_BATCH_SIZE_CAP,
            )
            or 0
        )

        # ------------------------------- 闁哄秷顫夊畵渚€鏌岃箛鎾愁嚙闁哄倽顫夌涵鍓佹嫚閸℃鐒肩憸鐗堟尭婢х姷浠﹂崒婊勭暠闁哄鍟撮崳绋课熼垾宕囩妤犵偠鍩栭悧搴㈩殽瀹€鈧€规娊寮?-------------------------------
        # 鐟滅増鎸哥紞瀣礈瀹ュ懐婀撮梺鎻掓川閺?GPTQ Marlin 闂佹彃绻愮€垫煡寮憴鍕€婇柡鍐啇缁辨繃娼诲☉妯哄汲闂佹彃绻愮€?expert 闁告柣鍔嶉埀顑跨婵偞娼崐鐔虹唴鐎垫澘瀚ㄩ埀?
        if isinstance(layer.quant_method, GPTQMarlinMoEMethod):
            # 閻犱焦婢樼紞宥堛亹閹惧啿顤呴柟璨夊啫鐓戦柛锝冨妼娴兼劖鎷呭鍐╄含 GPTQ Marlin 婵☆垪鈧磭纭€濞戞挸顑冮埀?
            self._mode = "gptq_marlin"
            input_dtype = layer.quant_method.input_dtype
            if input_dtype == torch.float8_e4m3fn:
                self._marlin_input_dtype_name = "fp8"
            elif input_dtype == torch.int8:
                self._marlin_input_dtype_name = "int8"
            elif input_dtype is None:
                self._marlin_input_dtype_name = "a16"
            else:
                self._marlin_input_dtype_name = str(input_dtype).replace("torch.", "")

            # 閻犱焦婢樼紞?GPTQ 闁哄嫷鍨伴幆渚€宕ラ婊勬殢濞?desc_act 闂佹澘绉堕悿鍡涙晬鐏炶姤鍊电紓渚囧幗婢х晫鎮板畝鍐唴鐎垫澘瀚ぐ鏌ュ箰婢跺寒鍤夐柡宥呮搐缁绘棃宕橀崘鑼毎濠㈣泛瀚幃濠囧礆閸℃ɑ鏆滈柕?
            self._gptq_desc_act = bool(layer.quant_method.quant_config.desc_act)

            # 閻犱焦婢樼紞宥堛亹閹惧啿顤呴悘鐐插€块崳娲礌閺嶃劍缍€闂佹彃绉垫晶宥夊捶閵娧勭暠闁烩晩鍠楅悥?GPU 閻犱焦鍎抽ˇ顒勫Υ?
            self.device = layer.w13_qweight.device

            # 濞ｅ洦绻傞悺銊╂煂韫囨挸顕ч柡澶婂暣閸ｆ悂鎯?pack_factor 闁告瑥鍊归弳鐔煎Υ?
            self.pack_factor = layer.quant_method.quant_config.pack_factor

            # 濞ｅ洦绻傞悺銊╂煂韫囨挸顕х紒顐ヮ嚙閻庨鈧數鎳撶花鏌ユ儍?bit 闁轰浇鍩囬埀?
            self.num_bits = layer.quant_method.quant_config.quant_type.size_bits

            # 濞ｅ洦绻傞悺銊╂煂韫囨挸顕?group size 闁告瑥鍊归弳鐔煎Υ?
            self.group_size = layer.quant_method.quant_config.group_size

            # 闁哄秴娲╅鍥亹閹惧啿顤呴悹渚灠缁剁偞绋夊鍡樞?8bit 婵犵鍋撴繛鍙夋閻儳顕ラ崟鈹惧亾?
            self.is_a_8bit = bool(
                input_dtype is not None and getattr(input_dtype, "itemsize", 0) == 1
            )

        # 鐟滅増鎸哥紞瀣礈瀹ュ懐婀撮梺鎻掓川閺併倝妫冮悙鍝勬闁?FusedMoE 闁哄倽顫夌涵鍫曞籍鐠佸湱绀夐弶鈺傜☉閸欏棝妫冮悙鍝勬闁告牗鐗曟慨鈺呭箑娴ｇ顫ｉ弶鐐测偓鐔虹唴鐎垫澘瀚ㄩ埀?
        elif isinstance(layer.quant_method, UnquantizedFusedMoEMethod):
            # 鐟滅増鎸告晶鐘绘閻愬搫娅ら柛鏍ㄧ墪婵晠骞€娴ｇ顫ｉ弶鐐测偓鐔虹唴鐎垫澘瀚粭澶愬绩椤栨稑鐦悽?bias 闁?MoE闁?
            if layer.quant_method.moe.has_bias:
                raise ValueError(
                    "Tiered MoE cache currently does not support biased unquantized MoE"
                )

            # 鐟滅増鎸告晶鐘绘閻愬搫娅ら柛鏍ㄧ墳閻儳顕ラ崟顒佹殰闁?Triton闁靛棔绔糢DA ATen 濞?PyTorch 闁搞儳鍋ら埀顑藉亾 backend闁?
            if layer.quant_method.unquantized_backend not in (
                    UnquantizedMoeBackend.TRITON,
                    UnquantizedMoeBackend.CUDA_ATEN,
                    UnquantizedMoeBackend.TORCH,
            ):
                raise TypeError(
                    "Tiered MoE cache currently only supports TRITON, "
                    "CUDA_ATEN, and TORCH fallback unquantized MoE"
                )

            # 閻犱焦婢樼紞宥堛亹閹惧啿顤呴柟璨夊啫鐓戦柛锝冨妼娴兼劖鎷呭鍐╄含闂傚牏鍋ら崳娲礌閺嶎煂浣割嚕韫囧海鐟撻柕?
            self._mode = "unquantized"

            # 闂傚牏鍋ら崳娲礌閺嶎剛鐔呯€垫澘瀚粭鍛▔瀹ュ棛效闁?GPTQ desc_act闁挎稑鏈Ο澶婎嚕韫囨凹鍞剁憸鐗堟磻鐠?False闁?
            self._gptq_desc_act = False

            # 閻犱焦婢樼紞宥堛亹閹惧啿顤呴悘鐐插€垮顏堟煂韫囨挸顕ч柡澶婂暣閸ｆ悂骞嶉埀顒勫捶閵娧勭暠闁烩晩鍠楅悥?GPU 閻犱焦鍎抽ˇ顒勫Υ?
            self.device = layer.w13_weight.device
        else:
            # 鐟滅増鎸告晶鐘诲箳瑜嶉崺妤呭闯閵娿倗鐭岄柡鈧娑樼槷 GPTQ Marlin 闁告粌鐭傚顏堟煂韫囨挸顕?FusedMoE 濞戞挶鍊楃悮顐ゆ崉椤栨氨绐為柕?
            raise TypeError(
                "Tiered MoE cache currently supports GPTQMarlinMoEMethod "
                "and UnquantizedFusedMoEMethod only"
            )

        # ------------------------------- 闁哄秷顫夊畵?layer._expert_map 闁告瑥绉电敮?resident GPU slot 闁汇劌瀚崹鍨叏鐎ｂ晝鐟╅悗纭呮硾缁旈浠﹂埀?-------------------------------
        # 闂侇剙绉村濠氬箥閳ь剟寮垫径濠傚伎閻忕偐鍋?expert闁挎稑鐭傞埀顒佷亢缁?layer._expert_map 闁告瑥绉电敮褰掑礄閸濆嫮绉奸柛?resident GPU slot 濞戞搩鍘奸悿鍕⒔閸涚补鏁嗛柣锝嗙懅濞堟垿寮伴姘喛濞?expert闁?
        for global_expert in range(layer.global_num_experts):
            # 閻犲洩顕цぐ鍥亹閹惧啿顤呴柛蹇嬪妼閻?expert 闁哄嫮濮撮惃鐘诲礆閹殿喗鐣遍柡鍫墮濠€?slot 缂傚倹鐗曡ぐ鍧楀Υ?
            slot = int(layer._expert_map[global_expert].item())

            # 鐟?slot 缂傚倹鐗曡ぐ鍧楁媰閽樺韬憸鐗堟尭婢?resident slot 闁肩厧鍟ú鍧楀礃閸涱喗顦ч柨娑樼焷椤斿洩銇愰弴鐐插緭闁告瑥绉撮幃婊堝及閻樿尙娈搁柛蹇撶－闁挳濡?
            if 0 <= slot < self.num_slots:
                # 闁哄牜鍓欏﹢纾條ot --> 闁稿繈鍔岄惇鐟俵ot
                self._slot_to_global[slot] = global_expert
                self._expert_to_slot[global_expert] = slot

        # ------------------------------- 闁告帗绻傞～鎰板礌?CPU 閻㈩垱鎮傞埞?expert 婵湱濮崇粭宀勫储閻斿娼楅柡澶婂暣閸ｅ摜绱撻幘鍐叉毐闁?-------------------------------
        # 闁哄秷顫夊畵浣姐亹閹惧啿顤呴悹浣测偓鍐茬亰濞戞挸瀛╁鍫ユ煂瀹ュ枺浣割嚕韫囨挸鐏ュ┑顔碱儏鐎?CPU 濞撴皜鍐╃ゼ閻?expert 婵湱濮撮幏浼村储閻斿娼楅柡澶婂暣閸ｅ摜绱撻幘鍐叉毐闁告牗浜介埀?
        self._init_cpu_fixed_pools()

    def _uses_marlin_fp8_activation(self) -> bool:
        return str(getattr(self, "_marlin_input_dtype_name", "a16")) == "fp8"

    def _preprocess_marlin_fp8_raw_weights(self, raw_gpu: _RawExpertWeights) -> None:
        if not self._uses_marlin_fp8_activation():
            return
        ops.marlin_int4_fp8_preprocess(raw_gpu.w13_qweight, inplace=True)
        ops.marlin_int4_fp8_preprocess(raw_gpu.w2_qweight, inplace=True)
        raw_gpu.w13_scales = raw_gpu.w13_scales * MARLIN_READY_FP8_WEIGHT_SCALE_FACTOR
        raw_gpu.w2_scales = raw_gpu.w2_scales * MARLIN_READY_FP8_WEIGHT_SCALE_FACTOR

    @property
    def slot_to_global(self) -> tuple[int, ...]:
        # 閻庣數鎳撻ˇ濠氬汲閹绢喗鑻熼柛娆樹海椤曟壆鎲撮崱妤佺闁挎稑鐭傛导鈺呭礂瀹ュ牏娈堕柣顫妽閺岀喖鎯勭€涙ê澶嶉柡鈧悷鏉挎櫢闁告劕鎳橀崕?resident 闁哄嫮濮撮惃鐘诲Υ?
        return tuple(self._slot_to_global)

    @property
    def prefill_burst_capacity(self) -> int:
        # 闁哄牜浜崢銈囩磾?burst pool 闁哄啳顔愮槐婵堚偓鐟扮秺閸ｈ櫣鎲撮崱鏇＄ 0闁?
        if self.prefill_burst_pool is None:
            return 0
        # 闁告熬绠戦崹顖涙交閺傛寧绀€闁稿繐褰夐棅?burst pool 闁告瑯鍨径宥夊籍鐠轰警鍟囩紒鎯х－濞?experts 闁轰浇鍩囬埀?
        return self.prefill_burst_pool.num_slots

    @property
    def prefill_burst_min_tokens(self) -> int:
        # 闁?burst pool 闁哄啳顔愮槐婵堟嫚閵夆晜顫岄柛濠傚悑閻ュ懘寮垫径瀣濞戞柨顦埀?
        if self.prefill_burst_pool is None:
            return 0
        # 閺夆晜鏌ㄥú?burst path 閻熸瑱绠戣ぐ鍌炲箥閳ь剟妫侀埀顒勬儍閸曨剚浠橀悘?token 闁轰浇鍩囬埀?
        return self._prefill_burst_min_tokens

    def can_run_prefill_burst(self, num_unique_experts: int, num_tokens: int) -> bool:
        # 闁告瑯浜濆﹢?burst pool 閻庢稒锚濠€顏堝Υ娴ｅ壊鍟囬梺鎻掔箺閸愮粯寰勯悢绮瑰亾娑旂磤ken 闁轰焦婢橀¨鍕緞瑜岀粭鏍沪閸屾俺鍩岄柣妯挎硾閸氬鈧顫夊鍌炴晬鐏炴儳顤呴柤鍐测偓鐔绘巢 burst闁?
        return (
                self.prefill_burst_pool is not None
                and num_unique_experts <= self.prefill_burst_pool.num_slots
                and self.prefill_burst_pool.supports_layer(self.layer)
        )

    def run_prefill_burst(
            self,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 婵炲备鍓濆﹢渚€宕楅崣姗€鐓?burst pool 闁哄啳顔愮槐婵堟嫚鐎涙ɑ顫栭悹鍥ュ劚閻増绋夊鍡樻殰闁?闁哄牜浜滈幆搴ㄦ偨閵婎煈鍤夐柟绗涘棭鏀介悹渚灠缁剁偤濡?
        if self.prefill_burst_pool is None:
            raise RuntimeError(f"{self.layer_key}: prefill burst pool is not available")
        return self.prefill_burst_pool.execute(
            controller=self,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts_input=shared_experts_input,
        )


    def _normalize_requested_experts(
            self,
            expert_ids: torch.Tensor | list[int] | tuple[int, ...],
    ) -> list[int]:
        if torch.is_tensor(expert_ids):
            requested = self._unique_experts(expert_ids)
        else:
            requested = []
            seen: set[int] = set()
            for expert_id in expert_ids:
                parsed = int(expert_id)
                if parsed < 0 or parsed in seen:
                    continue
                requested.append(parsed)
                seen.add(parsed)
        return requested

    def prepare(
            self,
            topk_ids: torch.Tensor,
            topk_weights: torch.Tensor | None = None,
            router_probs: torch.Tensor | None = None,
    ) -> None:
        bench_timing = _bench_timing_enabled()
        prepare_t0 = _time.perf_counter() if bench_timing else 0.0
        request_t0 = _time.perf_counter() if bench_timing else 0.0
        requested = self._normalize_requested_experts(topk_ids)
        request_seconds = (
            _time.perf_counter() - request_t0
            if bench_timing else 0.0
        )
        if not requested:
            return
        execution_slots = self.num_slots
        if len(requested) > execution_slots:
            raise RuntimeError(
                f"{self.layer_key}: requested {len(requested)} unique experts but "
                f"only {execution_slots} resident slots are executable"
            )

        self._step += 1
        plan_t0 = _time.perf_counter() if bench_timing else 0.0
        resident_t0 = _time.perf_counter() if bench_timing else 0.0
        expert_to_slot = self._ensure_cpu_expert_to_slot()
        resident_requested_slots: list[int] = []
        load_missing: list[int] = []
        for expert_id in requested:
            slot = (
                int(expert_to_slot[expert_id])
                if 0 <= expert_id < len(expert_to_slot)
                else -1
            )
            if slot >= 0:
                resident_requested_slots.append(slot)
            else:
                load_missing.append(expert_id)
        resident_check_seconds = (
            _time.perf_counter() - resident_t0
            if bench_timing else 0.0
        )

        load_plan: list[tuple[int, int]] = []
        victim_select_seconds = 0.0
        load_plan_zip_seconds = 0.0
        if load_missing:
            victim_t0 = _time.perf_counter() if bench_timing else 0.0
            victim_slots = self._choose_victim_slots_round_robin(
                count=len(load_missing),
                protected_slots=resident_requested_slots,
            )
            victim_select_seconds = (
                _time.perf_counter() - victim_t0
                if bench_timing else 0.0
            )
            load_plan_zip_t0 = _time.perf_counter() if bench_timing else 0.0
            load_plan = list(zip(load_missing, victim_slots, strict=True))
            load_plan_zip_seconds = (
                _time.perf_counter() - load_plan_zip_t0
                if bench_timing else 0.0
            )
        plan_seconds = (
            _time.perf_counter() - plan_t0
            if bench_timing else 0.0
        )
        if load_plan:
            load_t0 = _time.perf_counter() if bench_timing else 0.0
            load_stats = self._load_experts_into_slots(load_plan) or {}
            load_seconds = (_time.perf_counter() - load_t0) if bench_timing else 0.0
        else:
            load_stats = {}
            load_seconds = 0.0

        final_requested_count = len(requested)
        cpu_copy_threads = max(
            1,
            int(
                getattr(
                    self,
                    "_prepare_cpu_copy_threads",
                    getattr(self, "_prepare_cpu_copy_batch_size", 1),
                )
                or 1
            ),
        )
        if (
            self._step % 50 == 0
            and not bench_timing
            and logger.isEnabledFor(10)
        ):
            resident_hit = (len(requested) - len(load_plan)) / max(len(requested), 1) * 100
            logger.debug(
                "CFIE_PREPARE_STATS layer=%s step=%d requested=%d final_requested=%d "
                "staged=%d stage_resident_hit=%.0f%% missing=%d load_missing=%d "
                "cpu_copy_threads=%d cpu_hits=%d evictions=%d",
                self.layer_key,
                self._step,
                len(requested),
                final_requested_count,
                len(load_plan),
                resident_hit,
                len(load_missing),
                len(load_missing),
                cpu_copy_threads,
                self._cpu_hits,
                self._evictions,
            )
        if bench_timing:
            load_materialize_seconds = float(
                load_stats.get("materialize_seconds", 0.0)
            )
            load_write_seconds = float(load_stats.get("write_seconds", 0.0))
            load_cpu_pack_seconds = float(load_stats.get("cpu_pack_seconds", 0.0))
            load_h2d_seconds = float(load_stats.get("h2d_seconds", 0.0))
            load_gpu_scatter_seconds = float(
                load_stats.get("gpu_scatter_seconds", 0.0)
            )
            load_install_seconds = float(load_stats.get("install_seconds", 0.0))
            load_cpu_stage_mode = str(load_stats.get("cpu_stage_mode", ""))
            load_h2d_mode = str(load_stats.get("h2d_mode", ""))
            logger.info(
                "CFIE_BENCH_TIMING prepare layer=%s step=%d plan=%.3fms "
                "request_unique=%.3fms plan_resident=%.3fms "
                "plan_victim=%.3fms plan_zip=%.3fms "
                "stage=%.3fms stage_materialize=%.3fms stage_write=%.3fms "
                "stage_cpu_pack=%.3fms stage_h2d=%.3fms stage_gpu_scatter=%.3fms "
                "stage_install=%.3fms total=%.3fms requested=%d staged=%d "
                "final_requested=%d missing=%d load_missing=%d "
                "cpu_copy_threads=%d cpu_stage_mode=%s cpu_stage_native=%d "
                "cpu_stage_pinned=%d cpu_source_pinned=%d cpu_source_pageable=%d "
                "cpu_source_shared=%d cpu_source_contiguous=%d h2d_mode=%s "
                "h2d_source_pinned=%d h2d_non_blocking=%d h2d_source_shared=%d "
                "h2d_single_contiguous=%d h2d_bytes=%d native_scatter_install=%d "
                "installed_mappings=%d",
                self.layer_key,
                self._step,
                plan_seconds * 1000.0,
                request_seconds * 1000.0,
                resident_check_seconds * 1000.0,
                victim_select_seconds * 1000.0,
                load_plan_zip_seconds * 1000.0,
                load_seconds * 1000.0,
                load_materialize_seconds * 1000.0,
                load_write_seconds * 1000.0,
                load_cpu_pack_seconds * 1000.0,
                load_h2d_seconds * 1000.0,
                load_gpu_scatter_seconds * 1000.0,
                load_install_seconds * 1000.0,
                (_time.perf_counter() - prepare_t0) * 1000.0,
                len(requested),
                len(load_plan),
                final_requested_count,
                len(load_missing),
                len(load_missing),
                cpu_copy_threads,
                load_cpu_stage_mode,
                int(load_stats.get("cpu_stage_native", 0) or 0),
                int(load_stats.get("cpu_stage_pinned", 0) or 0),
                int(load_stats.get("cpu_source_pinned", 0) or 0),
                int(load_stats.get("cpu_source_pageable", 0) or 0),
                int(load_stats.get("cpu_source_shared", 0) or 0),
                int(load_stats.get("cpu_source_contiguous", 0) or 0),
                load_h2d_mode,
                int(load_stats.get("h2d_source_pinned", 0) or 0),
                int(load_stats.get("h2d_non_blocking", 0) or 0),
                int(load_stats.get("h2d_source_shared", 0) or 0),
                int(load_stats.get("h2d_single_contiguous", 0) or 0),
                int(load_stats.get("h2d_bytes", 0) or 0),
                int(load_stats.get("native_scatter_install", 0) or 0),
                int(load_stats.get("installed_mappings", 0) or 0),
            )

    def _ensure_cpu_expert_to_slot(self) -> list[int]:
        expert_to_slot = getattr(self, "_expert_to_slot", None)
        if expert_to_slot is not None:
            return expert_to_slot
        expert_map = getattr(self.layer, "_expert_map", None)
        num_experts = int(getattr(self.layer, "global_num_experts", 0) or 0)
        if torch.is_tensor(expert_map):
            num_experts = max(num_experts, int(expert_map.numel()))
        for expert_id in getattr(self, "_slot_to_global", ()):
            if int(expert_id) >= 0:
                num_experts = max(num_experts, int(expert_id) + 1)
        expert_to_slot = [-1] * num_experts
        for slot, expert_id in enumerate(getattr(self, "_slot_to_global", ())):
            expert_id = int(expert_id)
            if 0 <= expert_id < num_experts:
                expert_to_slot[expert_id] = int(slot)
        self._expert_to_slot = expert_to_slot
        return expert_to_slot

    def _choose_victim_slots_round_robin(
            self,
            *,
            count: int,
            protected_slots: list[int] | tuple[int, ...],
    ) -> list[int]:
        if count <= 0:
            return []
        num_slots = min(self.num_slots, len(self._slot_to_global))
        if num_slots <= 0:
            raise RuntimeError(f"{self.layer_key}: no resident slots are available")

        protected = [False] * num_slots
        for slot in protected_slots:
            parsed = int(slot)
            if 0 <= parsed < num_slots:
                protected[parsed] = True

        selected: list[int] = []
        cursor = int(getattr(self, "_victim_cursor", 0) or 0) % num_slots
        scanned = 0
        while len(selected) < count and scanned < num_slots:
            slot = cursor
            cursor = (cursor + 1) % num_slots
            scanned += 1
            if protected[slot]:
                continue
            selected.append(slot)
            protected[slot] = True

        if len(selected) < count:
            raise RuntimeError(
                f"{self.layer_key}: no evictable slot available for requested experts"
            )
        self._victim_cursor = cursor
        return selected

    def _unique_experts(self, topk_ids: torch.Tensor) -> list[int]:
        # 缂佸矂缂氱欢顓㈠礂閵壯勭函闁规亽鍎寸换鎴﹀炊閻愮鏁勯柛鎺擃殙閵嗗啴濡?
        if topk_ids.numel() == 0:
            return []
        flat_ids = topk_ids.detach().reshape(-1)
        if flat_ids.numel() <= SMALL_TOPK_UNIQUE_CPU_THRESHOLD:
            cpu_ids = flat_ids.to(device="cpu", dtype=torch.int64)
            requested: list[int] = []
            seen: set[int] = set()
            for expert_id in cpu_ids.tolist():
                parsed = int(expert_id)
                if parsed < 0 or parsed in seen:
                    continue
                seen.add(parsed)
                requested.append(parsed)
            requested.sort()
            return requested
        # 闁稿繐鐗嗘禒?unique闁挎稑鑻崯鈧柛?CPU 閺夌儐鍓氶崹?Python int 闁告帗顨夐妴鍐晬鐏炵偓鐓欏〒姘仢閹绱掗鐔蜂粯闁告帟鍩栫粊锔芥媴鐠恒劍鏆忛柕?
        unique_ids = torch.unique(flat_ids).to(device="cpu")
        return [
            int(expert_id)
            for expert_id in unique_ids.tolist()
            if int(expert_id) >= 0
        ]

    def _is_resident(self, expert_id: int) -> bool:
        # Keep prepare metadata on CPU. Reading layer._expert_map[expert].item()
        # here would synchronize with the GPU for every resident check.
        expert_to_slot = self._ensure_cpu_expert_to_slot()
        if expert_id < 0 or expert_id >= len(expert_to_slot):
            return False
        return int(expert_to_slot[expert_id]) >= 0

    def _materialize_cpu_static_bundle(
            self,
            expert_id: int,
            *,
            eager: bool = False,
            private_raw: bool = False,
    ) -> ExpertBundle | None:
        # ------------------------------- 濞寸姴鎳嶇拹鐔烘媼閳ュ啿鐏婇柟绋挎搐閻ｉ箖鎯?CPU 閻㈩垱鎮傞埞?expert 闁绘せ鏅涚€垫煡妫冨▎鎰ㄥ亾?bundle -------------------------------
        # 鐟滅増鎸哥紞瀣礈?expert 濞戞挸绉撮惈妯荤?CPU 閻㈩垱鎮傞埞?expert 闂傚棗妫楅幃搴ㄥ籍鐠佸湱绀夐柣鈺佺摠鐢瓨娼婚弬鎸庣 None闁?
        if expert_id not in self._cpu_static_experts:
            return None

        # ------------------------------- 濞村吋锚閸樻稒寰勫鍥ㄦ殢鐎规瓕灏欑划锟犳偋閳轰礁顕ч悗鐟版湰閸ㄦ岸鎯?CPU 闂傚牊鐟﹂埀?bundle -------------------------------
        # 闁稿繐鐗呯划?CPU 闂傚牊鐟﹂埀顑胯兌缁憋妇鈧稒锚閻⊙囧礂闂€鎰幀闁哄被鍎叉竟妯裤亹閹惧啿顤?expert 閻庣數鎳撶花鏌ユ儍?bundle闁?
        bundle = self._cpu_static_bundles.get(expert_id)

        # 鐟滅増鎸哥紞瀣礈?expert 闁?CPU 闂傚牊鐟﹂埀?bundle 鐎规瓕灏欑划锛勨偓娑櫭﹢顏堝籍鐠佸湱绀夐柣鈺佺摠鐢瓨寰勫鍥ㄦ殢閻?bundle闁?
        if bundle is not None:
            return bundle

        bundle = self._load_source_expert_bundle(expert_id)
        bundle = self._preprocess_cpu_static_bundle_compat(
            bundle,
            private_raw=private_raw,
        )
        return self._register_cpu_static_bundle(
            expert_id,
            bundle,
            eager=eager,
        )

    def _preprocess_cpu_static_bundle_compat(
            self,
            bundle: ExpertBundle,
            *,
            private_raw: bool = False,
    ) -> ExpertBundle:
        preprocess = self._preprocess_cpu_static_bundle
        try:
            signature = inspect.signature(preprocess)
        except (TypeError, ValueError):
            return preprocess(bundle, private_raw=private_raw)

        accepts_private_raw = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            or name == "private_raw"
            for name, param in signature.parameters.items()
        )
        if accepts_private_raw:
            return preprocess(bundle, private_raw=private_raw)
        return preprocess(bundle)

    def _materialize_cpu_static_bundle_safe(
            self,
            expert_id: int,
            *,
            eager: bool = False,
            private_raw: bool = False,
    ) -> ExpertBundle | None:
        return self._materialize_cpu_static_bundle(
            expert_id,
            eager=eager,
            private_raw=private_raw,
        )

    def _load_source_expert_bundle(self, expert_id: int) -> ExpertBundle:
        bundle = self._allocate_source_bundle()
        skip_suffixes = ("g_idx",) if not getattr(self, "_gptq_desc_act", False) else ()
        self.expert_store.copy_expert_into(
            self.layer_key,
            expert_id,
            bundle.tensors,
            skip_suffixes=skip_suffixes,
        )
        return bundle

    def _register_cpu_static_bundle(
            self,
            expert_id: int,
            bundle: ExpertBundle,
            *,
            eager: bool,
    ) -> ExpertBundle:
        lock = getattr(self, "_cpu_static_bundle_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._cpu_static_bundle_lock = lock
        with lock:
            existing = self._cpu_static_bundles.get(expert_id)
            if existing is not None:
                return existing

            self._cpu_static_bundles[expert_id] = bundle
            self._cpu_buffer_bytes += bundle.nbytes

            return bundle

    def _pack_cpu_static_bundles_layer_wide(
            self,
            *,
            pin_memory: bool,
    ) -> None:
        if not self._cpu_static_bundles:
            return

        expert_ids = sorted(self._cpu_static_bundles)
        bundles = [self._cpu_static_bundles[expert_id] for expert_id in expert_ids]
        if not all(bundle.runtime_ready for bundle in bundles):
            raise RuntimeError(
                f"{self.layer_key}: layer-wide CPU static packing requires "
                "runtime-ready expert bundles"
            )
        self._cpu_static_layer_expert_ids = tuple(expert_ids)
        self._cpu_static_layer_expert_index_by_id = {
            expert_id: index for index, expert_id in enumerate(expert_ids)
        }

        if (
                len(bundles) >= 2
                and all(bundle.storage is bundles[0].storage for bundle in bundles)
                and bundles[0].pinned == pin_memory
        ):
            expected_stride = int(bundles[0].nbytes)
            if all(
                    int(bundle.storage_offset_bytes) == index * expected_stride
                    for index, bundle in enumerate(bundles)
            ):
                self._cpu_static_layer_field_views = (
                    self._build_cpu_static_layer_field_views(bundles)
                )
                return

        if pin_memory and all(
            bundle.pinned
            and bundle.runtime_ready
            and self._source_has_contiguous_expert_storage(
                bundle,
                int(bundle.nbytes),
            )
            for bundle in bundles
        ):
            self._cpu_static_layer_field_views = {}
            logger.info(
                "Using chunked pinned CPU static experts without layer-wide repack: "
                "layer=%s experts=%d bytes=%.2f MiB chunks=%d",
                self.layer_key,
                len(expert_ids),
                sum(bundle.nbytes for bundle in bundles) / (1 << 20),
                len({id(bundle.storage) for bundle in bundles}),
            )
            gc.collect()
            return

        packed_bundles = pack_cpu_bundles_by_expert(
            bundles,
            pin_memory=pin_memory,
            runtime_ready=True,
        )
        self._cpu_static_bundles = {
            expert_id: packed_bundle
            for expert_id, packed_bundle in zip(
                expert_ids,
                packed_bundles,
                strict=True,
            )
        }
        self._cpu_static_layer_field_views = (
            self._build_cpu_static_layer_field_views(packed_bundles)
        )
        logger.info(
            "Packed layer CPU static experts into contiguous storage: "
            "layer=%s experts=%d pinned=%s bytes=%.2f MiB",
            self.layer_key,
            len(expert_ids),
            pin_memory,
            sum(bundle.nbytes for bundle in packed_bundles) / (1 << 20),
        )
        gc.collect()

    def _should_pin_cpu_static_layer(self) -> bool:
        use_pinned = bool(getattr(self, "_use_pinned_cpu_static", False))
        layer_index = getattr(self, "_moe_layer_index", None)
        pinned_layers = getattr(self, "_cpu_static_pinned_layers", frozenset())
        if layer_index is not None and int(layer_index) in pinned_layers:
            use_pinned = True
        return use_pinned

    def _pack_cpu_static_bundles_layer_wide_best_effort(self) -> None:
        # CPU static mirror 婵ɑ鐡曠换?pageable, 濞戞挸绉撮崯鈧悘蹇旂箚閻?pinned packing闁?
        if not self._cpu_static_bundles:
            return
        use_pinned = self._should_pin_cpu_static_layer()
        if not use_pinned:
            self._pack_cpu_static_bundles_layer_wide(pin_memory=False)
            return
        try:
            gc.collect()
            _empty_torch_host_allocator_cache_best_effort()
            self._pack_cpu_static_bundles_layer_wide(pin_memory=True)
        except Exception as exc:
            gc.collect()
            _empty_torch_host_allocator_cache_best_effort()
            raise RuntimeError(
                "Failed to allocate pinned CPU static storage for configured "
                f"tiered MoE layer {getattr(self, 'layer_key', '<unknown>')}"
            ) from exc

    def _build_cpu_static_layer_field_views(
            self,
            packed_bundles: list[ExpertBundle],
    ) -> dict[str, torch.Tensor]:
        if not packed_bundles:
            return {}
        if not all(bundle.runtime_ready for bundle in packed_bundles):
            return {}

        bundles_and_sources = [
            (index, index, bundle, "cpu_static")
            for index, bundle in enumerate(packed_bundles)
        ]
        field_views: dict[str, torch.Tensor] = {}
        for field_name in packed_bundles[0].tensors:
            packed_view = self._view_packed_runtime_ready_cpu_field(
                bundles_and_sources,
                field_name,
            )
            if packed_view is None:
                return {}
            field_views[field_name] = packed_view
        return field_views

    def _build_runtime_ready_expert_indices(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
    ) -> torch.Tensor | None:
        expert_index_by_id = getattr(
            self,
            "_cpu_static_layer_expert_index_by_id",
            None,
        )
        if not expert_index_by_id:
            return None

        expert_indices: list[int] = []
        for expert_id, _slot, _bundle, _source in bundles_and_sources:
            expert_index = expert_index_by_id.get(int(expert_id))
            if expert_index is None:
                return None
            expert_indices.append(expert_index)
        return torch.tensor(expert_indices, dtype=torch.int64)

    def _materialize_cpu_static_bundles_eager(
            self,
            expert_ids: tuple[int, ...],
    ) -> None:
        pending = [
            expert_id
            for expert_id in expert_ids
            if expert_id in self._cpu_static_experts
               and expert_id not in self._cpu_static_bundles
        ]
        if not pending:
            return

        t_start = _time.perf_counter()
        configured_batch = int(
            getattr(self, "_cpu_static_preprocess_batch_size", 0) or 0
        )
        requested_batch = (
            len(pending)
            if configured_batch <= 0
            else min(len(pending), configured_batch)
        )
        effective_batch = self._resolve_cpu_static_preprocess_batch_size(
            requested_batch
        )

        if self._mode != "gptq_marlin":
            chunk_size = effective_batch
            if chunk_size <= 1:
                for expert_id in pending:
                    self._materialize_cpu_static_bundle_safe(
                        expert_id,
                        eager=True,
                        private_raw=True,
                    )
            else:
                chunks = [
                    pending[start:start + chunk_size]
                    for start in range(0, len(pending), chunk_size)
                ]
                with ThreadPoolExecutor(max_workers=min(len(chunks), chunk_size)) as executor:
                    futures = [
                        executor.submit(
                            lambda expert_chunk=expert_chunk: [
                                self._materialize_cpu_static_bundle_safe(
                                    expert_id,
                                    eager=True,
                                    private_raw=True,
                                )
                                for expert_id in expert_chunk
                            ]
                        )
                        for expert_chunk in chunks
                    ]
                    for future in as_completed(futures):
                        future.result()
        else:
            chunk_size = effective_batch
            if chunk_size <= 1:
                for expert_id in pending:
                    self._materialize_cpu_static_bundle_safe(
                        expert_id,
                        eager=True,
                        private_raw=True,
                    )
            else:
                for start in range(0, len(pending), chunk_size):
                    batch_expert_ids = pending[start:start + chunk_size]
                    self._materialize_quantized_cpu_static_batch_with_retry(
                        batch_expert_ids
                    )

        logger.debug(
            "CPU static mirror: layer=%s experts=%d batch=%d "
            "cpu_bytes=%.2f MiB elapsed=%.1fs",
            self.layer_key,
            len(pending),
            effective_batch,
            int(getattr(self, "_cpu_buffer_bytes", 0)) / (1 << 20),
            _time.perf_counter() - t_start,
        )
        _empty_torch_host_allocator_cache_best_effort()

    def _materialize_quantized_cpu_static_batch_with_retry(
            self,
            expert_ids: list[int],
    ) -> None:
        if not expert_ids:
            return

        try:
            self._materialize_quantized_cpu_static_batch(expert_ids)
            return
        except Exception as exc:
            if not _is_cuda_oom_error(exc) or len(expert_ids) <= 1:
                raise

            _synchronize_torch_device_best_effort(getattr(self, "device", None))
            torch.accelerator.empty_cache()

            next_cap = max(1, len(expert_ids) // 2)
            configured_cap = int(
                getattr(self, "_cpu_static_preprocess_batch_size_cap", 0) or 0
            )
            if configured_cap <= 0 or next_cap < configured_cap:
                self._cpu_static_preprocess_batch_size_cap = next_cap

            logger.warning(
                "Retrying CPU static preprocess with smaller batch: layer=%s "
                "failed=%d next_cap=%d",
                self.layer_key,
                len(expert_ids),
                next_cap,
            )

        split_index = max(1, len(expert_ids) // 2)
        self._materialize_quantized_cpu_static_batch_with_retry(
            expert_ids[:split_index]
        )
        self._materialize_quantized_cpu_static_batch_with_retry(
            expert_ids[split_index:]
        )

    def _resolve_cpu_static_preprocess_batch_size(
            self,
            requested_batch_size: int,
    ) -> int:
        if requested_batch_size <= 1:
            return max(1, requested_batch_size)

        configured_cap = int(
            getattr(self, "_cpu_static_preprocess_batch_size_cap", 0) or 0
        )
        upper_bound = (
            min(requested_batch_size, configured_cap)
            if configured_cap > 0
            else requested_batch_size
        )
        if upper_bound <= 1:
            return 1
        if getattr(self, "_mode", "gptq_marlin") != "gptq_marlin":
            return upper_bound

        cpu_available = self._get_available_cpu_memory_bytes()
        gpu_available = self._get_available_device_memory_bytes()
        if cpu_available is None and gpu_available is None:
            return upper_bound

        cpu_reserve = int(
            getattr(
                self,
                "_cpu_static_preprocess_cpu_reserve_bytes",
                DEFAULT_CPU_STATIC_PREPROCESS_CPU_RESERVE_BYTES,
            )
        )
        gpu_reserve = int(
            getattr(
                self,
                "_cpu_static_preprocess_gpu_reserve_bytes",
                DEFAULT_CPU_STATIC_PREPROCESS_GPU_RESERVE_BYTES,
            )
        )

        best = 1
        low = 1
        high = upper_bound
        while low <= high:
            batch_size = (low + high) // 2
            fits_cpu = (
                    cpu_available is None
                    or self._estimate_quantized_cpu_static_preprocess_cpu_bytes(
                batch_size
            ) <= max(0, cpu_available - cpu_reserve)
            )
            fits_gpu = (
                    gpu_available is None
                    or self._estimate_quantized_cpu_static_preprocess_gpu_bytes(
                batch_size
            ) <= max(0, gpu_available - gpu_reserve)
            )
            if fits_cpu and fits_gpu:
                best = batch_size
                low = batch_size + 1
            else:
                high = batch_size - 1

        configured_batch = int(
            getattr(self, "_cpu_static_preprocess_batch_size", 0) or 0
        )
        if configured_batch <= 0:
            logger.info_once(
                "Auto-selected CPU static preprocess batch: layer=%s "
                "requested=%d selected=%d cpu_avail=%.2f GiB gpu_avail=%.2f GiB",
                self.layer_key,
                upper_bound,
                best,
                -1.0 if cpu_available is None else cpu_available / (1 << 30),
                -1.0 if gpu_available is None else gpu_available / (1 << 30),
            )
        elif best < upper_bound:
            logger.info_once(
                "Shrinking CPU static preprocess batch: layer=%s "
                "requested=%d selected=%d cpu_avail=%.2f GiB gpu_avail=%.2f GiB",
                self.layer_key,
                upper_bound,
                best,
                -1.0 if cpu_available is None else cpu_available / (1 << 30),
                -1.0 if gpu_available is None else gpu_available / (1 << 30),
            )
        return best

    @staticmethod
    def _get_available_cpu_memory_bytes() -> int | None:
        return _get_available_cpu_memory_bytes_best_effort()

    def _get_available_device_memory_bytes(self) -> int | None:
        if getattr(self, "device", None) is None:
            return None

        device = torch.device(self.device)
        device_module = getattr(torch, device.type, None)
        if device_module is None or not hasattr(device_module, "mem_get_info"):
            return None

        device_index = device.index
        try:
            free_bytes, _total_bytes = (
                device_module.mem_get_info(device_index)
                if device_index is not None
                else device_module.mem_get_info()
            )
            return int(free_bytes)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    def _estimate_quantized_checkpoint_source_bundle_nbytes(self) -> int:
        hidden_size = int(self.layer.hidden_size)
        intermediate_size = int(self.layer.intermediate_size_per_partition)
        w13_scale_bytes = int(self.layer.w13_scales.element_size())
        w2_scale_bytes = int(self.layer.w2_scales.element_size())
        w13_qzeros_bytes = int(self.layer.w13_qzeros.element_size())
        w2_qzeros_bytes = int(self.layer.w2_qzeros.element_size())

        total_bytes = (
                2 * (hidden_size // self.pack_factor) * intermediate_size * 4
                + 2 * int(self.layer.num_groups_w13) * intermediate_size * w13_scale_bytes
                + 2
                * int(self.layer.num_groups_w13)
                * (intermediate_size // self.pack_factor)
                * w13_qzeros_bytes
                + (intermediate_size // self.pack_factor) * hidden_size * 4
                + int(self.layer.num_groups_w2) * hidden_size * w2_scale_bytes
                + int(self.layer.num_groups_w2)
                * (hidden_size // self.pack_factor)
                * w2_qzeros_bytes
        )
        if self._gptq_desc_act:
            total_bytes += (
                    2 * hidden_size * int(self.layer.w13_g_idx.element_size())
                    + intermediate_size * int(self.layer.w2_g_idx.element_size())
            )
        return total_bytes

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor | None) -> int:
        if tensor is None:
            return 0
        return int(tensor.numel() * tensor.element_size())

    def _estimate_quantized_runtime_bundle_nbytes(self) -> int:
        total_bytes = (
                self._tensor_nbytes(self.layer.w13_qweight[0])
                + self._tensor_nbytes(self.layer.w2_qweight[0])
                + self._tensor_nbytes(self.layer.w13_scales[0])
                + self._tensor_nbytes(self.layer.w2_scales[0])
                + self._tensor_nbytes(self.layer.w13_qzeros[0])
                + self._tensor_nbytes(self.layer.w2_qzeros[0])
        )
        if self._gptq_desc_act:
            total_bytes += (
                    self._tensor_nbytes(self.layer.w13_g_idx[0])
                    + self._tensor_nbytes(self.layer.w2_g_idx[0])
                    + self._tensor_nbytes(self.layer.w13_g_idx_sort_indices[0])
                    + self._tensor_nbytes(self.layer.w2_g_idx_sort_indices[0])
            )
        return total_bytes

    def _estimate_quantized_gpu_preprocess_output_nbytes(self) -> int:
        total_bytes = (
                self._tensor_nbytes(self.layer.w13_qweight[0])
                + self._tensor_nbytes(self.layer.w2_qweight[0])
                + self._tensor_nbytes(self.layer.w13_scales[0])
                + self._tensor_nbytes(self.layer.w2_scales[0])
        )
        if self._gptq_desc_act:
            total_bytes += (
                    self._tensor_nbytes(self.layer.w13_g_idx[0])
                    + self._tensor_nbytes(self.layer.w2_g_idx[0])
                    + self._tensor_nbytes(self.layer.w13_g_idx_sort_indices[0])
                    + self._tensor_nbytes(self.layer.w2_g_idx_sort_indices[0])
            )
        return total_bytes

    def _estimate_quantized_cpu_static_preprocess_cpu_bytes(
            self,
            batch_size: int,
    ) -> int:
        raw_bytes_per_expert = self._raw_quantized_nbytes(self._cpu_quantized_raw_buffer)
        return (
                self._estimate_quantized_checkpoint_source_bundle_nbytes()
                + batch_size * raw_bytes_per_expert
                + batch_size * self._estimate_quantized_runtime_bundle_nbytes()
        )

    def _estimate_quantized_cpu_static_preprocess_gpu_bytes(
            self,
            batch_size: int,
    ) -> int:
        raw_bytes_per_expert = self._raw_quantized_nbytes(self._cpu_quantized_raw_buffer)
        return batch_size * (
                raw_bytes_per_expert
                + self._estimate_quantized_gpu_preprocess_output_nbytes()
        )

    def _materialize_quantized_cpu_static_batch(
            self,
            expert_ids: list[int],
    ) -> None:
        if not expert_ids:
            return

        pin_cpu_static = self._should_pin_cpu_static_layer()
        cache = self._load_marlin_ready_layer_cache(
            expert_ids,
            copy_storage=not pin_cpu_static,
            pin_storage=pin_cpu_static,
        )
        runtime_tensors: dict[str, torch.Tensor] | None = None
        runtime_bundles: list[ExpertBundle] | None = None
        if cache is not None:
            runtime_tensors = cache.tensors
            runtime_bundles = cache.bundles

        if runtime_tensors is None and runtime_bundles is None:
            raw = self._allocate_quantized_raw_buffer(
                batch_size=len(expert_ids),
                pin_memory=False,
            )
            for expert_index, expert_id in enumerate(expert_ids):
                bundle = self._load_source_expert_bundle(expert_id)
                self._assemble_raw_weights(
                    bundle,
                    raw=raw,
                    expert_index=expert_index,
                )
                del bundle

            runtime_tensors = self._preprocess_quantized_raw_batch(raw)
            del raw
            self._save_marlin_ready_layer_cache(expert_ids, runtime_tensors)
        if pin_cpu_static:
            gc.collect()
            _empty_torch_host_allocator_cache_best_effort()
            torch.accelerator.empty_cache()
        source_runtime_bundles = runtime_bundles
        try:
            if runtime_bundles is not None:
                if pin_cpu_static and not all(
                        bundle.pinned for bundle in runtime_bundles
                ):
                    runtime_bundles = (
                        self._pin_quantized_runtime_bundles_with_retry(
                            runtime_bundles
                        )
                    )
            elif runtime_tensors is not None:
                runtime_bundles = (
                    self._build_quantized_runtime_bundles_pinned_with_retry(
                        runtime_tensors
                    )
                    if pin_cpu_static
                    else self._build_quantized_runtime_bundles(
                        runtime_tensors,
                        pin_memory=False,
                    )
                )
            else:
                runtime_bundles = []
        except Exception as exc:
            if not pin_cpu_static:
                raise
            gc.collect()
            _empty_torch_host_allocator_cache_best_effort()
            torch.accelerator.empty_cache()
            raise RuntimeError(
                "Failed to allocate pinned CPU static runtime bundles for "
                f"configured tiered MoE layer "
                f"{getattr(self, 'layer_key', '<unknown>')} "
                f"(experts={len(expert_ids)})"
            ) from exc
        for expert_id, runtime_bundle in zip(expert_ids, runtime_bundles, strict=False):
            self._register_cpu_static_bundle(
                expert_id,
                runtime_bundle,
                eager=True,
            )
        del runtime_tensors
        del runtime_bundles
        gc.collect()
        _empty_torch_host_allocator_cache_best_effort()

    def _preprocess_cpu_static_bundle(
            self,
            bundle: ExpertBundle,
            *,
            private_raw: bool = False,
    ) -> ExpertBundle:
        # 鐎规瓕灏欑划锟犲及?runtime-ready 闁?bundle 濞戞挸绉归崳鍛婂緞瀹ュ拋妲遍柣鐐叉閳?
        if bundle.runtime_ready:
            return bundle

        # 闂佹彃绻愮€佃尙鎹勯姘辩獮濞戞挸顑戠槐婊砅U static mirror 閹煎瓨姊荤槐锔锯偓?Marlin MoE runtime 闁哄秶鍘х槐锟犳晬?
        # gate/up 闁稿繐鐗嗛幃搴ㄧ嵁閺堜絻绀?w13闁挎稑鐦?3/w2 qweight 闁?repack闁挎稑顔抍ale 闁?permute闁?
        if getattr(self, "_mode", "") == "gptq_marlin":
            return self._preprocess_quantized_static_bundle(
                bundle,
                private_raw=private_raw,
            )

        # 闂傚牏鍋ら崳娲礌閺嶎剛鐔呯€垫澘瀚粭鍛存晬鐎涘└U static mirror 閹煎瓨姊荤槐锔锯偓娑櫳戞晶鐣屾偘鐏炵晫婀撮柣鈺佺摠鐢潙鈽夐崼锝呯€柣?w13/w2闁?
        if getattr(self, "_mode", "") == "unquantized":
            return self._preprocess_unquantized_static_bundle(
                bundle,
                private_raw=private_raw,
            )

        # 闁告娲樼粊鎾箣閺嵮冩倯閻庣顫夊Λ顐ゆ崉椤栨氨绐炲☉鎿冨幖瑜版煡鎳楁禒瀣у亾濮樺磭绠?__new__ 闁哄瀚伴埀顒傚Т瀹曟劙宕氬┑鍡╂綏闁告牗鐗曢顔炬寬閳藉懐骞㈡慨婵勫€栧鍌涚┍濠靛洤鐦柛妯煎枑閻楅亶濡?
        return bundle

    def _preprocess_quantized_static_bundle(
            self,
            bundle: ExpertBundle,
            *,
            private_raw: bool = False,
    ) -> ExpertBundle:
        return self._preprocess_quantized_static_bundles_batch(
            [bundle],
            private_raw=private_raw,
        )[0]

    def _preprocess_quantized_static_bundles_batch(
            self,
            bundles: list[ExpertBundle],
            *,
            private_raw: bool = False,
    ) -> list[ExpertBundle]:
        if not bundles:
            return []

        if len(bundles) == 1:
            if private_raw:
                raw = self._allocate_quantized_raw_buffer(batch_size=1)
                raw = self._assemble_raw_weights(
                    bundles[0],
                    raw=raw,
                    expert_index=0,
                )
            else:
                raw = self._assemble_raw_weights(bundles[0])
        else:
            raw = self._allocate_quantized_raw_buffer(
                batch_size=len(bundles),
            )
            for expert_index, bundle in enumerate(bundles):
                self._assemble_raw_weights(
                    bundle,
                    raw=raw,
                    expert_index=expert_index,
                )

        runtime_tensors = self._preprocess_quantized_raw_batch(raw)
        return self._build_quantized_runtime_bundles(
            runtime_tensors,
            pin_memory=bool(
                getattr(
                    self,
                    "_use_pinned_cpu_static",
                    getattr(self, "_use_pinned_cpu", False),
                )
            ),
        )

    def _preprocess_quantized_raw_batch(
            self,
            raw: _RawExpertWeights,
    ) -> dict[str, torch.Tensor]:
        raw_gpu = None
        repacked_w13 = None
        repacked_w2 = None
        permuted_w13_scales = None
        permuted_w2_scales = None
        w13_g_idx_sort_indices = None
        w2_g_idx_sort_indices = None
        w13_sorted_g_idx = None
        w2_sorted_g_idx = None

        try:
            raw_gpu = self._move_raw_weights_to_device(raw)
            self._preprocess_marlin_fp8_raw_weights(raw_gpu)
            num_experts = int(raw_gpu.w13_qweight.shape[0])

            if self._gptq_desc_act:
                if raw_gpu.w13_g_idx is None or raw_gpu.w2_g_idx is None:
                    raise RuntimeError(
                        f"{self.layer_key}: desc_act=True requires g_idx tensors "
                        "during CPU static preprocessing"
                    )

                w13_g_idx_sort_indices = torch.argsort(raw_gpu.w13_g_idx, dim=-1).to(
                    torch.int32
                )
                w2_g_idx_sort_indices = torch.argsort(raw_gpu.w2_g_idx, dim=-1).to(
                    torch.int32
                )
                w13_sorted_g_idx = torch.gather(
                    raw_gpu.w13_g_idx,
                    -1,
                    w13_g_idx_sort_indices,
                )
                w2_sorted_g_idx = torch.gather(
                    raw_gpu.w2_g_idx,
                    -1,
                    w2_g_idx_sort_indices,
                )
            else:
                w13_g_idx_sort_indices = self._empty_perm(
                    raw_gpu.w13_qweight.device,
                    num_experts,
                )
                w2_g_idx_sort_indices = self._empty_perm(
                    raw_gpu.w2_qweight.device,
                    num_experts,
                )

            repacked_w13 = ops.gptq_marlin_moe_repack(
                raw_gpu.w13_qweight,
                w13_g_idx_sort_indices,
                raw_gpu.w13_qweight.shape[1] * self.pack_factor,
                raw_gpu.w13_qweight.shape[2],
                self.num_bits,
                is_a_8bit=self.is_a_8bit,
            )
            repacked_w2 = ops.gptq_marlin_moe_repack(
                raw_gpu.w2_qweight,
                w2_g_idx_sort_indices,
                raw_gpu.w2_qweight.shape[1] * self.pack_factor,
                raw_gpu.w2_qweight.shape[2],
                self.num_bits,
                is_a_8bit=self.is_a_8bit,
            )
            permuted_w13_scales = marlin_moe_permute_scales(
                s=raw_gpu.w13_scales,
                size_k=self.layer.intermediate_size_per_partition,
                size_n=raw_gpu.w13_scales.shape[2],
                group_size=self.group_size,
                is_a_8bit=self.is_a_8bit,
            )
            permuted_w2_scales = marlin_moe_permute_scales(
                s=raw_gpu.w2_scales,
                size_k=raw_gpu.w2_scales.shape[1]
                       * (self.group_size if self.group_size != -1 else self.pack_factor),
                size_n=raw_gpu.w2_scales.shape[2],
                group_size=self.group_size,
                is_a_8bit=self.is_a_8bit,
            )
            tensors = {
                "runtime.w13_qweight": self._to_cpu_static_tensor(
                    repacked_w13,
                ),
                "runtime.w2_qweight": self._to_cpu_static_tensor(
                    repacked_w2,
                ),
                "runtime.w13_scales": self._to_cpu_static_tensor(
                    permuted_w13_scales,
                ),
                "runtime.w2_scales": self._to_cpu_static_tensor(
                    permuted_w2_scales,
                ),
                "runtime.w13_qzeros": self._to_cpu_static_tensor(
                    raw_gpu.w13_qzeros,
                ),
                "runtime.w2_qzeros": self._to_cpu_static_tensor(
                    raw_gpu.w2_qzeros,
                ),
            }
            if (
                    w13_sorted_g_idx is not None
                    and w2_sorted_g_idx is not None
                    and self._gptq_desc_act
            ):
                tensors.update(
                    {
                        "runtime.w13_g_idx": self._to_cpu_static_tensor(
                            w13_sorted_g_idx,
                        ),
                        "runtime.w2_g_idx": self._to_cpu_static_tensor(
                            w2_sorted_g_idx,
                        ),
                        "runtime.w13_g_idx_sort_indices": self._to_cpu_static_tensor(
                            w13_g_idx_sort_indices,
                        ),
                        "runtime.w2_g_idx_sort_indices": self._to_cpu_static_tensor(
                            w2_g_idx_sort_indices,
                        ),
                    }
                )
            return tensors
        finally:
            _synchronize_torch_device_best_effort(getattr(self, "device", None))
            del raw_gpu
            del repacked_w13
            del repacked_w2
            del permuted_w13_scales
            del permuted_w2_scales
            del w13_g_idx_sort_indices
            del w2_g_idx_sort_indices
            del w13_sorted_g_idx
            del w2_sorted_g_idx
            gc.collect()
            torch.accelerator.empty_cache()

    @classmethod
    def _build_quantized_runtime_bundles(
            cls,
            runtime_tensors: dict[str, torch.Tensor],
            *,
            pin_memory: bool = False,
    ) -> list[ExpertBundle]:
        if not runtime_tensors:
            return []
        return pack_batched_cpu_tensor_dicts_by_expert(
            runtime_tensors,
            pin_memory=pin_memory,
            runtime_ready=True,
        )

    def _build_quantized_runtime_bundles_pinned_with_retry(
            self,
            runtime_tensors: dict[str, torch.Tensor],
    ) -> list[ExpertBundle]:
        if not runtime_tensors:
            return []
        first = next(iter(runtime_tensors.values()))
        batch_size = int(first.shape[0])

        def _build_range(start: int, end: int) -> list[ExpertBundle]:
            sliced = {
                name: tensor[start:end]
                for name, tensor in runtime_tensors.items()
            }
            try:
                gc.collect()
                _empty_torch_host_allocator_cache_best_effort()
                return self._build_quantized_runtime_bundles(
                    sliced,
                    pin_memory=True,
                )
            except Exception as exc:
                if end - start <= 1:
                    raise RuntimeError(
                        "Failed to allocate pinned CPU static runtime bundle for "
                        f"layer={getattr(self, 'layer_key', '<unknown>')} "
                        f"expert_index={start}"
                    ) from exc
                gc.collect()
                _empty_torch_host_allocator_cache_best_effort()
                torch.accelerator.empty_cache()
                mid = start + (end - start) // 2
                logger.warning(
                    "Retrying CPU static pinned runtime bundle with smaller chunk: "
                    "layer=%s failed=%d next=%d error=%s",
                    getattr(self, "layer_key", "<unknown>"),
                    end - start,
                    mid - start,
                    exc,
                )
                return _build_range(start, mid) + _build_range(mid, end)

        return _build_range(0, batch_size)

    def _pin_quantized_runtime_bundles_with_retry(
            self,
            runtime_bundles: list[ExpertBundle],
    ) -> list[ExpertBundle]:
        if not runtime_bundles:
            return []

        def _build_range(start: int, end: int) -> list[ExpertBundle]:
            sliced = runtime_bundles[start:end]
            try:
                gc.collect()
                _empty_torch_host_allocator_cache_best_effort()
                return pack_cpu_bundles_by_expert(
                    sliced,
                    pin_memory=True,
                    runtime_ready=True,
                )
            except Exception as exc:
                if end - start <= 1:
                    raise RuntimeError(
                        "Failed to pin CPU static runtime bundle for "
                        f"layer={getattr(self, 'layer_key', '<unknown>')} "
                        f"expert_index={start}"
                    ) from exc
                gc.collect()
                _empty_torch_host_allocator_cache_best_effort()
                torch.accelerator.empty_cache()
                mid = start + (end - start) // 2
                logger.warning(
                    "Retrying CPU static pinned runtime bundle with smaller chunk: "
                    "layer=%s failed=%d next=%d error=%s",
                    getattr(self, "layer_key", "<unknown>"),
                    end - start,
                    mid - start,
                    exc,
                )
                return _build_range(start, mid) + _build_range(mid, end)

        return _build_range(0, len(runtime_bundles))

    def _preprocess_unquantized_static_bundle(
            self,
            bundle: ExpertBundle,
            *,
            private_raw: bool = False,
    ) -> ExpertBundle:
        if private_raw:
            raw = self._allocate_unquantized_raw_buffer()
            raw = self._assemble_unquantized_weights(
                bundle,
                raw=raw,
                expert_index=0,
            )
        else:
            raw = self._assemble_unquantized_weights(bundle)
        tensors = {
            "runtime.w13_weight": self._to_cpu_static_tensor(
                raw.w13_weight[0],
            ),
            "runtime.w2_weight": self._to_cpu_static_tensor(
                raw.w2_weight[0],
            ),
        }
        use_pinned_cpu_static = bool(
            getattr(
                self,
                "_use_pinned_cpu_static",
                False,
            )
        )
        return pack_cpu_tensor_dict(
            tensors,
            pin_memory=use_pinned_cpu_static,
            runtime_ready=True,
        )

    def _runtime_requires_pinned_cpu(self) -> bool:
        return bool(getattr(self, "_use_pinned_cpu", False))

    def runtime_stage_allocation_spec(
            self,
            *,
            pin_memory: bool,
    ) -> tuple[int, tuple[Any, ...]]:
        template_bundle = self._allocate_runtime_ready_stage_bundle(pin_memory=False)
        return (
            int(template_bundle.nbytes),
            self._runtime_ready_stage_signature(
                [template_bundle],
                pin_memory=pin_memory,
            ),
        )

    def _allocate_runtime_ready_stage_bundle(
            self,
            *,
            pin_memory: bool,
    ) -> ExpertBundle:
        if self._mode == "gptq_marlin":
            specs = [
                PackedExpertTensorSpec(
                    name="runtime.w13_qweight",
                    shape=tuple(self.layer.w13_qweight.shape[1:]),
                    dtype=self.layer.w13_qweight.dtype,
                ),
                PackedExpertTensorSpec(
                    name="runtime.w2_qweight",
                    shape=tuple(self.layer.w2_qweight.shape[1:]),
                    dtype=self.layer.w2_qweight.dtype,
                ),
                PackedExpertTensorSpec(
                    name="runtime.w13_scales",
                    shape=tuple(self.layer.w13_scales.shape[1:]),
                    dtype=self.layer.w13_scales.dtype,
                ),
                PackedExpertTensorSpec(
                    name="runtime.w2_scales",
                    shape=tuple(self.layer.w2_scales.shape[1:]),
                    dtype=self.layer.w2_scales.dtype,
                ),
                PackedExpertTensorSpec(
                    name="runtime.w13_qzeros",
                    shape=tuple(self.layer.w13_qzeros.shape[1:]),
                    dtype=self.layer.w13_qzeros.dtype,
                ),
                PackedExpertTensorSpec(
                    name="runtime.w2_qzeros",
                    shape=tuple(self.layer.w2_qzeros.shape[1:]),
                    dtype=self.layer.w2_qzeros.dtype,
                ),
            ]
            if hasattr(self.layer, "w13_g_idx"):
                specs.extend(
                    (
                        PackedExpertTensorSpec(
                            name="runtime.w13_g_idx",
                            shape=tuple(self.layer.w13_g_idx.shape[1:]),
                            dtype=self.layer.w13_g_idx.dtype,
                        ),
                        PackedExpertTensorSpec(
                            name="runtime.w2_g_idx",
                            shape=tuple(self.layer.w2_g_idx.shape[1:]),
                            dtype=self.layer.w2_g_idx.dtype,
                        ),
                        PackedExpertTensorSpec(
                            name="runtime.w13_g_idx_sort_indices",
                            shape=tuple(self.layer.w13_g_idx_sort_indices.shape[1:]),
                            dtype=self.layer.w13_g_idx_sort_indices.dtype,
                        ),
                        PackedExpertTensorSpec(
                            name="runtime.w2_g_idx_sort_indices",
                            shape=tuple(self.layer.w2_g_idx_sort_indices.shape[1:]),
                            dtype=self.layer.w2_g_idx_sort_indices.dtype,
                        ),
                    )
                )
        else:
            specs = [
                PackedExpertTensorSpec(
                    name="runtime.w13_weight",
                    shape=tuple(self.layer.w13_weight.shape[1:]),
                    dtype=self.layer.w13_weight.dtype,
                ),
                PackedExpertTensorSpec(
                    name="runtime.w2_weight",
                    shape=tuple(self.layer.w2_weight.shape[1:]),
                    dtype=self.layer.w2_weight.dtype,
                ),
            ]

        storage, tensors = allocate_packed_cpu_tensor_views(
            specs,
            pin_memory=pin_memory,
        )
        return ExpertBundle(
            tensors=tensors,
            nbytes=bundle_nbytes(tensors),
            pinned=bool(storage.is_pinned()),
            runtime_ready=True,
            storage=storage,
            storage_offset_bytes=0,
        )

    def _ensure_runtime_ready_stage_bundle(self) -> ExpertBundle:
        stage_bundle = getattr(self, "_cpu_stage_bundle", None)
        if stage_bundle is not None:
            return stage_bundle
        stage_bundle = self._allocate_runtime_ready_stage_bundle(pin_memory=True)
        self._cpu_stage_bundle = stage_bundle
        self._cpu_buffer_bytes = int(getattr(self, "_cpu_buffer_bytes", 0)) + int(
            stage_bundle.nbytes
        )
        return stage_bundle

    def _move_runtime_ready_bundles_to_device(
            self,
            bundles: list[ExpertBundle],
            *,
            device: torch.device | str,
            stage_slot_capacity: int | None = None,
    ) -> list[ExpertBundle]:
        if not bundles:
            return []

        resolved_device = torch.device(device)
        first_bundle = bundles[0]
        source_storage = first_bundle.storage
        if source_storage is None:
            raise RuntimeError(
                f"{self.layer_key}: runtime-ready bundles are missing packed storage"
            )

        if (
                source_storage.device == resolved_device
                and all(bundle.storage is source_storage for bundle in bundles)
        ):
            self._last_runtime_h2d_info = {
                "mode": "already_device",
                "source_pinned": False,
                "non_blocking": False,
                "source_shared": True,
                "single_contiguous": False,
                "bytes": 0,
            }
            return bundles

        required_numel = self._runtime_ready_stage_required_numel(
            bundles,
            source_storage,
        )
        if required_numel <= 0:
            return []

        non_blocking = bool(
            source_storage.device.type == "cpu" and source_storage.is_pinned()
        )
        h2d_mode = "unknown"
        source_shared = all(bundle.storage is source_storage for bundle in bundles)
        source_is_single_contiguous_run = False
        target_storage = self._get_runtime_ready_gpu_stage_storage(
            required_numel=required_numel,
            per_expert_numel=self._runtime_ready_stage_per_expert_numel(bundles),
            dtype=source_storage.dtype,
            device=resolved_device,
            stage_slot_capacity=stage_slot_capacity,
        )
        if source_shared:
            used_native_h2d = False
            per_expert_bytes = int(bundles[0].nbytes)
            source_base_offset = int(bundles[0].storage_offset_bytes)
            source_is_single_contiguous_run = all(
                int(bundle.nbytes) == per_expert_bytes
                and int(bundle.storage_offset_bytes)
                == source_base_offset + index * per_expert_bytes
                for index, bundle in enumerate(bundles)
            )
            if source_is_single_contiguous_run:
                h2d_mode = "direct_contiguous"
                source_end = source_base_offset + required_numel
                target_storage[:required_numel].copy_(
                    source_storage[source_base_offset:source_end],
                    non_blocking=non_blocking,
                )
                used_native_h2d = True
            if (
                not used_native_h2d
                and
                source_storage.device.type == "cpu"
                and source_storage.dtype == torch.uint8
                and target_storage.dtype == torch.uint8
                and target_storage.device == resolved_device
                and source_storage.is_contiguous()
                and target_storage.is_contiguous()
                and all(int(bundle.nbytes) == per_expert_bytes for bundle in bundles)
                and ops.has_copy_expert_slices_to_stage_device()
            ):
                offsets = torch.empty(len(bundles), device="cpu", dtype=torch.int64)
                for index, bundle in enumerate(bundles):
                    offsets[index] = int(bundle.storage_offset_bytes)
                ops.copy_expert_slices_to_stage_device(
                    source_storage,
                    offsets,
                    target_storage[:len(bundles) * per_expert_bytes],
                    per_expert_bytes,
                    non_blocking,
                )
                h2d_mode = "native_gather_device"
                used_native_h2d = True
            if not used_native_h2d:
                h2d_mode = "loop_shared_storage"
                dest_offset = 0
                for bundle in bundles:
                    source_start = int(bundle.storage_offset_bytes)
                    source_end = source_start + int(bundle.nbytes)
                    dest_end = dest_offset + int(bundle.nbytes)
                    target_storage[dest_offset:dest_end].copy_(
                        source_storage[source_start:source_end],
                        non_blocking=non_blocking,
                    )
                    dest_offset = dest_end
        else:
            h2d_mode = "loop_multi_storage"
            dest_offset = 0
            for bundle in bundles:
                bundle_storage = bundle.storage
                if bundle_storage is None:
                    raise RuntimeError(
                        f"{self.layer_key}: runtime-ready bundle is missing storage"
                    )
                source_start = int(bundle.storage_offset_bytes)
                source_end = source_start + int(bundle.nbytes)
                dest_end = dest_offset + int(bundle.nbytes)
                target_storage[dest_offset:dest_end].copy_(
                    bundle_storage[source_start:source_end],
                    non_blocking=bool(
                        bundle_storage.device.type == "cpu"
                        and bundle_storage.is_pinned()
                    ),
                )
                dest_offset = dest_end

        self._last_runtime_h2d_info = {
            "mode": h2d_mode,
            "source_pinned": bool(non_blocking),
            "non_blocking": bool(non_blocking),
            "source_shared": bool(source_shared),
            "single_contiguous": bool(source_is_single_contiguous_run),
            "bytes": int(required_numel),
        }

        moved_bundles: list[ExpertBundle] = []
        dest_offset = 0
        for bundle in bundles:
            bundle_storage = bundle.storage
            if bundle_storage is None:
                raise RuntimeError(
                    f"{self.layer_key}: runtime-ready bundle is missing storage"
                )
            views: dict[str, torch.Tensor] = {}
            for field_name, tensor in bundle.tensors.items():
                field_offset_bytes = (
                    tensor.data_ptr()
                    - (bundle_storage.data_ptr() + int(bundle.storage_offset_bytes))
                )
                num_bytes = int(tensor.numel() * tensor.element_size())
                start = dest_offset + field_offset_bytes
                views[field_name] = target_storage[start:start + num_bytes].view(
                    tensor.dtype
                ).view(tensor.shape)
            moved_bundles.append(
                ExpertBundle(
                    tensors=views,
                    nbytes=bundle.nbytes,
                    pinned=False,
                    runtime_ready=bundle.runtime_ready,
                    storage=target_storage,
                    storage_offset_bytes=dest_offset,
                )
            )
            dest_offset += int(bundle.nbytes)
        return moved_bundles

    @staticmethod
    def _runtime_ready_stage_per_expert_numel(
            bundles: list[ExpertBundle],
    ) -> int:
        if not bundles:
            return 0
        first = bundles[0]
        return int(
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in first.tensors.values()
            )
        )

    def _get_runtime_ready_gpu_stage_storage(
            self,
            *,
            required_numel: int,
            per_expert_numel: int,
            dtype: torch.dtype,
            device: torch.device,
            stage_slot_capacity: int | None = None,
    ) -> torch.Tensor:
        stage_slots = max(
            1,
            int(getattr(self, "_runtime_stage_slots", 0) or 0),
            int(getattr(self, "_compute_slots", 0) or 0),
            int(stage_slot_capacity or 0),
        )
        capacity = max(
            int(required_numel),
            int(per_expert_numel) * stage_slots,
        )
        shared_pool = getattr(self, "_runtime_stage_pool", None)
        if shared_pool is not None:
            return shared_pool.get_gpu_storage(
                capacity=stage_slots,
                required_numel=required_numel,
                per_expert_numel=per_expert_numel,
                dtype=dtype,
                device=device,
            )
        signature = (dtype, int(per_expert_numel))

        def _allocate_or_get() -> torch.Tensor:
            storage = getattr(self, "_gpu_runtime_expert_stage_storage", None)
            storage_signature = getattr(
                self,
                "_gpu_runtime_expert_stage_signature",
                None,
            )
            storage_capacity = int(
                getattr(self, "_gpu_runtime_expert_stage_capacity", 0) or 0
            )
            storage_device = getattr(
                self,
                "_gpu_runtime_expert_stage_device",
                None,
            )
            if (
                storage is not None
                and storage_signature == signature
                and storage_capacity >= capacity
                and storage_device == device
            ):
                return storage

            storage = torch.empty(capacity, dtype=dtype, device=device)
            self._gpu_runtime_expert_stage_storage = storage
            self._gpu_runtime_expert_stage_capacity = capacity
            self._gpu_runtime_expert_stage_signature = signature
            self._gpu_runtime_expert_stage_device = device
            return storage

        lock = getattr(self, "_gpu_runtime_expert_stage_lock", None)
        if lock is None:
            return _allocate_or_get()
        with lock:
            return _allocate_or_get()

    def _runtime_ready_stage_required_numel(
            self,
            bundles: list[ExpertBundle],
            source_storage: torch.Tensor,
    ) -> int:
        if any(bundle.storage is not source_storage for bundle in bundles):
            return int(sum(int(bundle.nbytes) for bundle in bundles))
        if any(
            int(bundle.storage_offset_bytes) != index * int(bundle.nbytes)
            for index, bundle in enumerate(bundles)
        ):
            return int(sum(int(bundle.nbytes) for bundle in bundles))
        required_numel = 0
        for bundle in bundles:
            bundle_storage = bundle.storage
            if bundle_storage is None:
                raise RuntimeError(
                    f"{self.layer_key}: runtime-ready bundle is missing storage"
                )
            if bundle_storage is not source_storage:
                raise RuntimeError(
                    f"{self.layer_key}: runtime-ready bundles must share packed "
                    "stage storage before H2D"
                )
            for tensor in bundle.tensors.values():
                field_offset_bytes = (
                    tensor.data_ptr()
                    - (bundle_storage.data_ptr() + int(bundle.storage_offset_bytes))
                )
                num_bytes = int(tensor.numel() * tensor.element_size())
                required_numel = max(
                    required_numel,
                    int(bundle.storage_offset_bytes) + field_offset_bytes + num_bytes,
                )
        return required_numel

    def _write_runtime_ready_bundles_via_gpu_stage(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
            target: Any,
            *,
            install_mappings: bool = False,
    ) -> dict[str, Any]:
        if not bundles_and_sources:
            return {
                "cpu_pack_seconds": 0.0,
                "h2d_seconds": 0.0,
                "gpu_scatter_seconds": 0.0,
            }

        stage_slot_capacity = (
            int(getattr(target, "num_slots", 0) or 0)
            if self._is_prefill_burst_target(target)
            else None
        )

        stage_t0 = _time.perf_counter()
        runtime_ready_stage_bundles = self._prepare_runtime_ready_stage_bundles(
            bundles_and_sources,
            pin_memory=self._runtime_requires_pinned_cpu(),
            stage_slot_capacity=stage_slot_capacity,
        )
        cpu_pack_seconds = _time.perf_counter() - stage_t0
        cpu_stage_info = getattr(self, "_last_runtime_cpu_stage_info", {})
        if not runtime_ready_stage_bundles:
            return {
                "cpu_pack_seconds": cpu_pack_seconds,
                "h2d_seconds": 0.0,
                "gpu_scatter_seconds": 0.0,
            }

        target_fields, target_device = self._runtime_target_fields(
            target,
            runtime_ready_stage_bundles[0],
        )

        h2d_t0 = _time.perf_counter()
        gpu_stage_bundles = self._move_runtime_ready_bundles_to_device(
            runtime_ready_stage_bundles,
            device=target_device,
            stage_slot_capacity=stage_slot_capacity,
        )
        h2d_seconds = _time.perf_counter() - h2d_t0
        h2d_info = getattr(self, "_last_runtime_h2d_info", {})
        slot_ids = self._runtime_slot_ids_for_target(
            [slot for _expert_id, slot, _bundle, _source in bundles_and_sources],
            target_device,
        )

        scatter_t0 = _time.perf_counter()
        native_installed = False
        with torch.no_grad():
            if install_mappings:
                native_installed = self._try_native_runtime_scatter_and_install(
                    bundles_and_sources,
                    gpu_stage_bundles,
                    target,
                    slot_ids,
                )
            if not native_installed:
                for field_name, target_tensor in target_fields:
                    source_view = self._runtime_ready_stage_field_view(
                        gpu_stage_bundles,
                        field_name,
                    )
                    if source_view is None:
                        raise RuntimeError(
                            f"{self.layer_key}: failed to build GPU runtime-ready "
                            f"stage view for {field_name}"
                        )
                    target_tensor.index_copy_(0, slot_ids, source_view)
        gpu_scatter_seconds = _time.perf_counter() - scatter_t0
        return {
            "cpu_pack_seconds": cpu_pack_seconds,
            "h2d_seconds": h2d_seconds,
            "gpu_scatter_seconds": gpu_scatter_seconds,
            "cpu_stage_mode": str(cpu_stage_info.get("mode", "")),
            "cpu_stage_native": int(bool(cpu_stage_info.get("native_copy", False))),
            "cpu_stage_pinned": int(bool(cpu_stage_info.get("stage_pinned", False))),
            "cpu_source_pinned": int(cpu_stage_info.get("source_pinned", 0) or 0),
            "cpu_source_pageable": int(
                cpu_stage_info.get("source_pageable", 0) or 0
            ),
            "cpu_source_shared": int(bool(cpu_stage_info.get("source_shared", False))),
            "cpu_source_contiguous": int(
                cpu_stage_info.get("source_contiguous", 0) or 0
            ),
            "h2d_mode": str(h2d_info.get("mode", "")),
            "h2d_source_pinned": int(bool(h2d_info.get("source_pinned", False))),
            "h2d_non_blocking": int(bool(h2d_info.get("non_blocking", False))),
            "h2d_source_shared": int(bool(h2d_info.get("source_shared", False))),
            "h2d_single_contiguous": int(
                bool(h2d_info.get("single_contiguous", False))
            ),
            "h2d_bytes": int(h2d_info.get("bytes", 0) or 0),
            "native_scatter_install": int(native_installed),
            "installed_mappings": int(native_installed),
        }

    def _runtime_target_fields(
            self,
        target: Any,
        template_bundle: ExpertBundle,
    ) -> tuple[list[tuple[str, torch.Tensor]], torch.device]:
        if not hasattr(self, "_runtime_target_field_cache"):
            self._runtime_target_field_cache = {}
        cache_key = id(target)
        cached = self._runtime_target_field_cache.get(cache_key)
        if cached is not None:
            return cached

        fields: list[tuple[str, torch.Tensor]] = []
        target_device: torch.device | None = None
        for field_name in template_bundle.tensors.keys():
            target_attr = field_name.removeprefix("runtime.")
            target_tensor = getattr(target, target_attr, None)
            if target_tensor is None:
                continue
            if target_device is None:
                target_device = target_tensor.device
            fields.append((field_name, target_tensor))
        if target_device is None or not fields:
            raise RuntimeError(
                f"{self.layer_key}: runtime-ready target has no writable tensors"
            )
        resolved = (fields, target_device)
        self._runtime_target_field_cache[cache_key] = resolved
        return resolved

    def _runtime_slot_ids_for_target(
            self,
            slots: list[int],
            device: torch.device,
    ) -> torch.Tensor:
        count = len(slots)
        if count <= 0:
            return torch.empty((0,), dtype=torch.int64, device=device)
        if device.type != "cuda":
            return torch.tensor(slots, dtype=torch.int64, device=device)

        if not hasattr(self, "_runtime_slot_ids_lock"):
            self._runtime_slot_ids_lock = threading.Lock()
            self._runtime_slot_ids_cpu = None
            self._runtime_slot_ids_gpu = None
            self._runtime_slot_ids_gpu_device = None
        with self._runtime_slot_ids_lock:
            capacity = max(
                count,
                int(getattr(self, "_runtime_stage_slots", 0) or 0),
                int(getattr(self, "_compute_slots", 0) or 0),
            )
            cpu_buffer = self._runtime_slot_ids_cpu
            gpu_buffer = self._runtime_slot_ids_gpu
            if cpu_buffer is None or int(cpu_buffer.numel()) < capacity:
                cpu_buffer = torch.empty(
                    (capacity,),
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=is_pin_memory_available(),
                )
                self._runtime_slot_ids_cpu = cpu_buffer
            if (
                gpu_buffer is None
                or int(gpu_buffer.numel()) < capacity
                or self._runtime_slot_ids_gpu_device != device
            ):
                gpu_buffer = torch.empty(
                    (capacity,),
                    dtype=torch.int64,
                    device=device,
                )
                self._runtime_slot_ids_gpu = gpu_buffer
                self._runtime_slot_ids_gpu_device = device

            for index, slot in enumerate(slots):
                cpu_buffer[index] = int(slot)
            gpu_buffer[:count].copy_(
                cpu_buffer[:count],
                non_blocking=bool(cpu_buffer.is_pinned()),
            )
            return gpu_buffer[:count]

    def _runtime_ready_stage_field_view(
            self,
            bundles: list[ExpertBundle],
            field_name: str,
    ) -> torch.Tensor | None:
        if not bundles:
            return None
        first_bundle = bundles[0]
        storage = first_bundle.storage
        if storage is None:
            return None
        first_tensor = first_bundle.tensors[field_name]
        itemsize = int(first_tensor.element_size())
        per_expert_bytes = int(first_bundle.nbytes)
        field_offset_bytes = (
            first_tensor.data_ptr()
            - (storage.data_ptr() + int(first_bundle.storage_offset_bytes))
        )
        if (
            field_offset_bytes < 0
            or field_offset_bytes % itemsize != 0
            or per_expert_bytes % itemsize != 0
        ):
            return None
        start_byte = int(first_bundle.storage_offset_bytes) + field_offset_bytes
        if start_byte < 0 or start_byte >= int(storage.numel()):
            return None
        typed_storage = storage[start_byte:].view(first_tensor.dtype)
        batch_stride = per_expert_bytes // itemsize
        return typed_storage.as_strided(
            (len(bundles), *tuple(first_tensor.shape)),
            (batch_stride, *tuple(first_tensor.stride())),
        )

    def _runtime_int64_ids_for_target(
            self,
            values: list[int],
            device: torch.device,
            *,
            name: str,
    ) -> torch.Tensor:
        count = len(values)
        if count <= 0:
            return torch.empty((0,), dtype=torch.int64, device=device)
        if device.type != "cuda":
            return torch.tensor(values, dtype=torch.int64, device=device)

        cpu_attr = f"_runtime_{name}_ids_cpu"
        gpu_attr = f"_runtime_{name}_ids_gpu"
        gpu_device_attr = f"_runtime_{name}_ids_gpu_device"
        lock_attr = f"_runtime_{name}_ids_lock"
        lock = getattr(self, lock_attr, None)
        if lock is None:
            lock = threading.Lock()
            setattr(self, lock_attr, lock)
            setattr(self, cpu_attr, None)
            setattr(self, gpu_attr, None)
            setattr(self, gpu_device_attr, None)

        with lock:
            capacity = max(
                count,
                int(getattr(self, "_runtime_stage_slots", 0) or 0),
                int(getattr(self, "_compute_slots", 0) or 0),
            )
            cpu_buffer = getattr(self, cpu_attr)
            gpu_buffer = getattr(self, gpu_attr)
            gpu_buffer_device = getattr(self, gpu_device_attr)
            if cpu_buffer is None or int(cpu_buffer.numel()) < capacity:
                cpu_buffer = torch.empty(
                    (capacity,),
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=is_pin_memory_available(),
                )
                setattr(self, cpu_attr, cpu_buffer)
            if (
                gpu_buffer is None
                or int(gpu_buffer.numel()) < capacity
                or gpu_buffer_device != device
            ):
                gpu_buffer = torch.empty(
                    (capacity,),
                    dtype=torch.int64,
                    device=device,
                )
                setattr(self, gpu_attr, gpu_buffer)
                setattr(self, gpu_device_attr, device)

            for index, value in enumerate(values):
                cpu_buffer[index] = int(value)
            gpu_buffer[:count].copy_(
                cpu_buffer[:count],
                non_blocking=bool(cpu_buffer.is_pinned()),
            )
            return gpu_buffer[:count]

    def _try_native_runtime_scatter_and_install(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
            gpu_stage_bundles: list[ExpertBundle],
            target: Any,
            slot_ids: torch.Tensor,
    ) -> bool:
        if self._is_prefill_burst_target(target):
            return False
        expert_map = getattr(target, "_expert_map", None)
        if not torch.is_tensor(expert_map):
            return False
        if not gpu_stage_bundles or len(gpu_stage_bundles) != len(bundles_and_sources):
            return False

        device = expert_map.device
        old_expert_ids = [
            int(self._slot_to_global[int(slot)])
            for _expert_id, slot, _bundle, _source in bundles_and_sources
        ]
        new_expert_ids = [
            int(expert_id)
            for expert_id, _slot, _bundle, _source in bundles_and_sources
        ]
        old_ids = self._runtime_int64_ids_for_target(
            old_expert_ids,
            device,
            name="old_expert",
        )
        new_ids = self._runtime_int64_ids_for_target(
            new_expert_ids,
            device,
            name="new_expert",
        )

        def _view(field_name: str) -> torch.Tensor:
            source_view = self._runtime_ready_stage_field_view(
                gpu_stage_bundles,
                field_name,
            )
            if source_view is None:
                raise RuntimeError(
                    f"{self.layer_key}: failed to build GPU runtime-ready "
                    f"stage view for {field_name}"
                )
            return source_view

        if self._mode == "unquantized":
            if not ops.has_moe_batch_load_unquantized_runtime_and_install():
                return False
            ops.moe_batch_load_unquantized_runtime_and_install(
                slot_ids,
                old_ids,
                new_ids,
                _view("runtime.w13_weight"),
                _view("runtime.w2_weight"),
                target.w13_weight,
                target.w2_weight,
                expert_map,
            )
            return True

        if self._mode != "gptq_marlin":
            return False
        if not ops.has_moe_batch_load_gptq_runtime_and_install():
            return False

        has_g_idx = (
            "runtime.w13_g_idx" in gpu_stage_bundles[0].tensors
            and hasattr(target, "w13_g_idx")
        )
        ops.moe_batch_load_gptq_runtime_and_install(
            slot_ids,
            old_ids,
            new_ids,
            _view("runtime.w13_qweight"),
            _view("runtime.w2_qweight"),
            _view("runtime.w13_scales"),
            _view("runtime.w2_scales"),
            _view("runtime.w13_qzeros"),
            _view("runtime.w2_qzeros"),
            target.w13_qweight,
            target.w2_qweight,
            target.w13_scales,
            target.w2_scales,
            target.w13_qzeros,
            target.w2_qzeros,
            _view("runtime.w13_g_idx") if has_g_idx else None,
            _view("runtime.w2_g_idx") if has_g_idx else None,
            _view("runtime.w13_g_idx_sort_indices") if has_g_idx else None,
            _view("runtime.w2_g_idx_sort_indices") if has_g_idx else None,
            target.w13_g_idx if has_g_idx else None,
            target.w2_g_idx if has_g_idx else None,
            target.w13_g_idx_sort_indices if has_g_idx else None,
            target.w2_g_idx_sort_indices if has_g_idx else None,
            expert_map,
        )
        return True

    def _copy_runtime_ready_bundle_tensors(
            self,
            source_bundle: ExpertBundle,
            target_bundle: ExpertBundle,
    ) -> None:
        for field_name, source_tensor in source_bundle.tensors.items():
            target_bundle.tensors[field_name].copy_(source_tensor)

    # 闁冲厜鍋撻柍鍏夊亾 鐎瑰憡褰冮崹褰掓⒔閵堝洦鐣?batch H2D 闁哄倽顫夌涵?闁冲厜鍋撻柍鍏夊亾
    # _get_runtime_ready_batch_buffer, _try_write_runtime_ready_bundles_batch,
    # _view_packed_runtime_ready_cpu_field,
    # _stack_runtime_ready_cpu_field, _maybe_stack_runtime_ready_cpu_field,
    # _write_unquantized_runtime_ready_bundles_batch, _write_gptq_runtime_ready_bundles_batch
    # Shared expert-major staging view reused across fields before the GPU copy.

    def _get_runtime_ready_batch_buffer(
            self,
            field_name: str,
            first_tensor: torch.Tensor,
            batch_size: int,
            *,
            pin_memory: bool,
    ) -> torch.Tensor:
        def _get_or_create() -> torch.Tensor:
            buffers = getattr(self, "_cpu_runtime_batch_buffers", None)
            if buffers is None:
                buffers = {}
                self._cpu_runtime_batch_buffers = buffers

            existing = buffers.get(field_name)
            batch_shape = (batch_size, *first_tensor.shape)
            if (
                    existing is not None
                    and existing.dtype == first_tensor.dtype
                    and tuple(existing.shape[1:]) == tuple(first_tensor.shape)
                    and existing.shape[0] >= batch_size
                    and (not pin_memory or existing.is_pinned())
            ):
                return existing[:batch_size]

            batch_kwargs: dict[str, Any] = {
                "device": "cpu",
                "dtype": first_tensor.dtype,
            }
            if pin_memory:
                batch_kwargs["pin_memory"] = True
            try:
                batch = torch.empty(batch_shape, **batch_kwargs)
            except RuntimeError as exc:
                if pin_memory:
                    raise RuntimeError(
                        f"{self.layer_key}: failed to allocate pinned CPU staging "
                        f"buffer for {field_name}"
                    ) from exc
                batch = torch.empty(batch_shape, device="cpu", dtype=first_tensor.dtype)
            buffers[field_name] = batch
            return batch

        lock = getattr(self, "_cpu_runtime_batch_buffers_lock", None)
        if lock is None:
            return _get_or_create()
        with lock:
            return _get_or_create()

    def _to_cpu_static_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        # CPU static mirror 婵ɑ鐡曠换?pageable; pinned memory 濞寸姴鎳庡﹢?stage buffer 濞戞搩鍘烘繛鍥偨閵婏絺鍋?
        cpu_tensor = tensor.detach().to(device="cpu").contiguous()
        _synchronize_torch_device_best_effort(tensor.device)
        return cpu_tensor

    def _get_source_bundle(self, expert_id: int) -> tuple[ExpertBundle, str]:
        # 闁稿繐鐗嗛惃鍓ф嫚閺囩偞鍤掑☉?CPU static experts闁?
        bundle = self._cpu_static_bundles.get(expert_id)
        source = "cpu_static"
        if bundle is None:
            raise RuntimeError(
                f"{self.layer_key}: expert {expert_id} is not present in the CPU "
                "static expert pool. Current MoE tiered cache mainline requires "
                "initial_cpu_experts to cover every expert before prepare()."
            )
        # 閺夆晜鏌ㄥú?bundle 濞寸姰鍎卞鐑藉级閵夛妇鐖遍柡宥呮川椤掔兘鏁嶇仦鑲╄繑缂備胶鍠曢鍛婄▔鎼淬垺锛夐煫鍥ㄣ仦婵炲洭鎮介妸锝傚亾?
        if not bundle.runtime_ready:
            raise RuntimeError(
                f"{self.layer_key}: expert {expert_id} in CPU static pool is "
                "not runtime-ready; prepare() no longer materializes or repacks "
                "on demand."
            )
        return bundle, source
    def _load_expert_into_slot(self, expert_id: int, slot: int) -> None:
        self._load_experts_into_slots([(expert_id, slot)])

    def _load_experts_into_slots(
            self,
            assignments: list[tuple[int, int]],
    ) -> dict[str, Any]:
        if not assignments:
            return {}

        load_stats: dict[str, Any] = {
            "materialize_seconds": 0.0,
            "write_seconds": 0.0,
            "cpu_pack_seconds": 0.0,
            "h2d_seconds": 0.0,
            "gpu_scatter_seconds": 0.0,
            "install_seconds": 0.0,
        }
        materialize_t0 = _time.perf_counter()
        bundles_and_sources = self._resolve_batch_sources_for_plan(assignments)
        load_stats["materialize_seconds"] = _time.perf_counter() - materialize_t0

        if not bundles_and_sources:
            return load_stats

        write_t0 = _time.perf_counter()
        batch_stats = self._write_expert_bundles(
            bundles_and_sources,
            self.layer,
            install_mappings=True,
        ) or {}
        load_stats["write_seconds"] += _time.perf_counter() - write_t0
        load_stats["cpu_pack_seconds"] += float(
            batch_stats.get("cpu_pack_seconds", 0.0)
        )
        load_stats["h2d_seconds"] += float(batch_stats.get("h2d_seconds", 0.0))
        load_stats["gpu_scatter_seconds"] += float(
            batch_stats.get("gpu_scatter_seconds", 0.0)
        )
        for key in (
            "cpu_stage_native",
            "cpu_stage_pinned",
            "cpu_source_pinned",
            "cpu_source_pageable",
            "cpu_source_shared",
            "cpu_source_contiguous",
            "h2d_source_pinned",
            "h2d_non_blocking",
            "h2d_source_shared",
            "h2d_single_contiguous",
            "h2d_bytes",
            "native_scatter_install",
            "installed_mappings",
        ):
            load_stats[key] = int(batch_stats.get(key, 0) or 0)
        for key in ("cpu_stage_mode", "h2d_mode"):
            value = str(batch_stats.get(key, "") or "")
            if value:
                load_stats[key] = value
        for _expert_id, _slot, _bundle, source in bundles_and_sources:
            if source == "nvme_stage":
                self._nvme_loads += 1
            else:
                self._cpu_hits += 1

        install_t0 = _time.perf_counter()
        if int(batch_stats.get("installed_mappings", 0) or 0):
            self._install_cpu_mappings_after_native(bundles_and_sources)
        else:
            self._install_mappings(bundles_and_sources)
        load_stats["install_seconds"] += _time.perf_counter() - install_t0

        return load_stats

    def _resolve_batch_sources_for_plan(
            self,
            assignments: list[tuple[int, int]],
    ) -> list[tuple[int, int, ExpertBundle, str]]:
        bundles_and_sources: list[tuple[int, int, ExpertBundle, str]] = []
        cpu_static_bundles = getattr(self, "_cpu_static_bundles", {})
        for expert_id, slot in assignments:
            bundle = cpu_static_bundles.get(expert_id)
            if bundle is None:
                bundle, source = self._get_source_bundle(expert_id)
            else:
                source = "cpu_static"
            bundles_and_sources.append((expert_id, slot, bundle, source))
        return bundles_and_sources

    def _populate_prefill_burst_pool(
            self,
            pool: SharedPrefillBurstPool,
            requested: list[int],
    ) -> _PrefillBurstExecutionStats:
        stats = _PrefillBurstExecutionStats()
        bundles_and_sources: list[tuple[int, int, ExpertBundle, str]] = []
        for slot, expert_id in enumerate(requested):
            if self._is_resident(expert_id):
                self._copy_resident_expert_to_target(expert_id, slot, pool)
                stats.resident_hits += 1
                continue
            bundle, source = self._get_source_bundle(expert_id)
            bundles_and_sources.append((expert_id, slot, bundle, source))
            if source == "nvme_stage":
                stats.nvme_loads += 1
            else:
                stats.cpu_hits += 1
        if bundles_and_sources:
            self._write_expert_bundles(bundles_and_sources, pool)
        return stats

    @staticmethod
    def _is_prefill_burst_target(target: Any) -> bool:
        return isinstance(target, SharedPrefillBurstPool)

    def _write_expert_bundles(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
            target: Any,
            *,
            install_mappings: bool = False,
    ) -> dict[str, float]:
        batch_stats = self._try_write_runtime_ready_bundles_batch(
            bundles_and_sources,
            target,
            install_mappings=install_mappings,
        )
        if batch_stats is not None:
            return batch_stats
        for _expert_id, slot, bundle, _source in bundles_and_sources:
            self._write_expert_bundle(slot, bundle, target)
        return {
            "cpu_pack_seconds": 0.0,
            "h2d_seconds": 0.0,
            "gpu_scatter_seconds": 0.0,
        }

    def _try_write_runtime_ready_bundles_batch(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
            target: Any,
            *,
            install_mappings: bool = False,
    ) -> dict[str, float] | None:
        if not bundles_and_sources:
            return None

        bundles = [bundle for _expert_id, _slot, bundle, _source in bundles_and_sources]
        if not all(bundle.runtime_ready for bundle in bundles):
            return None

        if self._mode == "unquantized":
            return self._write_unquantized_runtime_ready_bundles_batch(
                bundles_and_sources,
                target,
                install_mappings=install_mappings,
            )

        if self._mode == "gptq_marlin":
            return self._write_gptq_runtime_ready_bundles_batch(
                bundles_and_sources,
                target,
                install_mappings=install_mappings,
            )

        return self._write_runtime_ready_bundles_via_gpu_stage(
            bundles_and_sources,
            target,
            install_mappings=install_mappings,
        )

    def _view_packed_runtime_ready_cpu_field(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
            field_name: str,
    ) -> torch.Tensor | None:
        if not bundles_and_sources:
            return None

        bundles = [bundle for _expert_id, _slot, bundle, _source in bundles_and_sources]
        first_bundle = bundles[0]
        first_tensor = first_bundle.tensors[field_name]
        storage = first_bundle.storage
        if storage is None:
            return None

        base_storage_ptr = storage.data_ptr()
        base_field_offset_bytes = (
            first_tensor.data_ptr()
            - (base_storage_ptr + int(first_bundle.storage_offset_bytes))
        )
        if base_field_offset_bytes < 0:
            return None

        tensor_signature = (
            first_tensor.dtype,
            tuple(first_tensor.shape),
            tuple(first_tensor.stride()),
        )
        per_bundle_bytes = int(first_bundle.nbytes)
        offsets = [int(first_bundle.storage_offset_bytes)]
        for bundle in bundles[1:]:
            if bundle.storage is None or bundle.storage.data_ptr() != base_storage_ptr:
                return None
            if int(bundle.nbytes) != per_bundle_bytes:
                return None
            tensor = bundle.tensors[field_name]
            if (
                    tensor.dtype,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
            ) != tensor_signature:
                return None
            field_offset_bytes = (
                tensor.data_ptr()
                - (bundle.storage.data_ptr() + int(bundle.storage_offset_bytes))
            )
            if field_offset_bytes != base_field_offset_bytes:
                return None
            offsets.append(int(bundle.storage_offset_bytes))

        if len(offsets) >= 2:
            stride_bytes = offsets[1] - offsets[0]
            if stride_bytes <= 0:
                return None
            if any(
                    offsets[index] - offsets[index - 1] != stride_bytes
                    for index in range(2, len(offsets))
            ):
                return None
        else:
            stride_bytes = per_bundle_bytes

        itemsize = first_tensor.element_size()
        if base_field_offset_bytes % itemsize != 0 or stride_bytes % itemsize != 0:
            return None

        field_num_bytes = int(first_tensor.numel() * itemsize)
        last_field_end = (
            offsets[0]
            + base_field_offset_bytes
            + (len(offsets) - 1) * stride_bytes
            + field_num_bytes
        )
        if last_field_end > int(storage.numel()):
            return None

        typed_base = storage[offsets[0] + base_field_offset_bytes:].view(
            first_tensor.dtype
        )
        return torch.as_strided(
            typed_base,
            size=(len(bundles), *first_tensor.shape),
            stride=(stride_bytes // itemsize, *first_tensor.stride()),
        )

    def _prepare_runtime_ready_stage_bundles(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
            *,
            pin_memory: bool,
            stage_slot_capacity: int | None = None,
    ) -> list[ExpertBundle]:
        bundles = [bundle for _expert_id, _slot, bundle, _source in bundles_and_sources]
        if not bundles:
            return []

        first_packed_nbytes = int(
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in bundles[0].tensors.values()
            )
        )
        source_pinned = sum(
            1
            for bundle in bundles
            if bool(bundle.pinned)
            or (
                bundle.storage is not None
                and bundle.storage.device.type == "cpu"
                and bool(bundle.storage.is_pinned())
            )
        )
        source_shared = (
            bundles[0].storage is not None
            and all(bundle.storage is bundles[0].storage for bundle in bundles)
        )
        source_contiguous = sum(
            1
            for bundle in bundles
            if self._source_has_contiguous_expert_storage(
                bundle,
                first_packed_nbytes,
            )
        )

        def _record_cpu_stage_info(
                mode: str,
                *,
                stage_pinned: bool,
                native_copy: bool = False,
        ) -> None:
            self._last_runtime_cpu_stage_info = {
                "mode": mode,
                "native_copy": bool(native_copy),
                "stage_pinned": bool(stage_pinned),
                "source_pinned": int(source_pinned),
                "source_pageable": int(len(bundles) - source_pinned),
                "source_shared": bool(source_shared),
                "source_contiguous": int(source_contiguous),
            }

        if len(bundles) == 1:
            storage = bundles[0].storage
            if (
                    storage is not None
                    and (not pin_memory or bundles[0].pinned)
                    and int(bundles[0].storage_offset_bytes) == 0
                    and int(storage.numel()) == first_packed_nbytes
            ):
                _record_cpu_stage_info(
                    "already_packed_single",
                    stage_pinned=bool(
                        storage.device.type == "cpu" and storage.is_pinned()
                    ),
                )
                return bundles
        else:
            first = bundles[0]
            first_storage = first.storage
            if (
                    first_storage is not None
                    and (not pin_memory or first.pinned)
            ):
                expected_stride = first_packed_nbytes
                expected_numel = len(bundles) * expected_stride
                if int(first_storage.numel()) == expected_numel and all(
                        bundle.storage is first.storage
                        and int(bundle.storage_offset_bytes) == index * expected_stride
                        for index, bundle in enumerate(bundles)
                ):
                    _record_cpu_stage_info(
                        "already_packed_batch",
                        stage_pinned=bool(
                            first_storage.device.type == "cpu"
                            and first_storage.is_pinned()
                        ),
                    )
                    return bundles
        if pin_memory and all(
            bundle.pinned
            and self._source_has_contiguous_expert_storage(
                bundle,
                first_packed_nbytes,
            )
            for bundle in bundles
        ):
            _record_cpu_stage_info(
                "direct_pinned_static",
                stage_pinned=True,
            )
            return bundles

        return self._pack_runtime_ready_bundles_into_reused_stage(
            bundles,
            pin_memory=pin_memory,
            stage_slot_capacity=stage_slot_capacity,
        )

    @staticmethod
    def _runtime_ready_stage_signature(
            bundles: list[ExpertBundle],
            *,
            pin_memory: bool,
    ) -> tuple[Any, ...]:
        first = bundles[0]
        ordered_names = tuple(first.tensors.keys())
        per_expert_bytes = int(
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in first.tensors.values()
            )
        )
        field_specs = tuple(
            (
                name,
                tuple(first.tensors[name].shape),
                first.tensors[name].dtype,
                tuple(first.tensors[name].stride()),
            )
            for name in ordered_names
        )
        return (bool(pin_memory), per_expert_bytes, field_specs)

    @staticmethod
    def _source_has_contiguous_expert_storage(
            bundle: ExpertBundle,
            per_expert_bytes: int,
    ) -> bool:
        storage = bundle.storage
        offset = int(bundle.storage_offset_bytes)
        return (
            storage is not None
            and storage.device.type == "cpu"
            and storage.dtype == torch.uint8
            and offset >= 0
            and offset + per_expert_bytes <= int(storage.numel())
        )

    def _copy_contiguous_expert_storage_to_stage_native(
            self,
            bundles: list[ExpertBundle],
            storage: torch.Tensor,
            per_expert_bytes: int,
            copy_batch_size: int,
    ) -> bool:
        if not bundles or not ops.has_copy_expert_slices_to_stage_cpu():
            return False
        source_storage = bundles[0].storage
        if source_storage is None:
            return False
        if not all(
            bundle.storage is source_storage
            and self._source_has_contiguous_expert_storage(
                bundle,
                per_expert_bytes,
            )
            for bundle in bundles
        ):
            return False

        offsets = torch.empty(len(bundles), device="cpu", dtype=torch.int64)
        for index, bundle in enumerate(bundles):
            offsets[index] = int(bundle.storage_offset_bytes)
        required_numel = len(bundles) * per_expert_bytes
        ops.copy_expert_slices_to_stage_cpu(
            source_storage,
            offsets,
            storage[:required_numel],
            per_expert_bytes,
            copy_batch_size,
        )
        return True

    def _pack_runtime_ready_bundles_into_reused_stage(
            self,
            bundles: list[ExpertBundle],
            *,
            pin_memory: bool,
            stage_slot_capacity: int | None = None,
    ) -> list[ExpertBundle]:
        if not bundles:
            return []

        first = bundles[0]
        ordered_names = list(first.tensors.keys())
        per_expert_bytes = int(
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in first.tensors.values()
            )
        )
        signature = self._runtime_ready_stage_signature(
            bundles,
            pin_memory=pin_memory,
        )
        for bundle in bundles:
            if list(bundle.tensors.keys()) != ordered_names:
                raise ValueError("all expert bundles must share the same tensor fields")
            for name in ordered_names:
                source = bundle.tensors[name]
                reference = first.tensors[name]
                if source.dtype != reference.dtype or tuple(source.shape) != tuple(
                    reference.shape
                ):
                    raise ValueError(
                        "all expert bundles must share tensor shapes and dtypes"
                    )

        capacity = max(
            len(bundles),
            int(getattr(self, "_runtime_stage_slots", 0) or 0),
            int(getattr(self, "_compute_slots", 0) or 0),
            int(stage_slot_capacity or 0),
        )
        shared_pool = getattr(self, "_runtime_stage_pool", None)
        if shared_pool is not None:
            shared_storage = shared_pool.get_cpu_storage(
                capacity=capacity,
                per_expert_bytes=per_expert_bytes,
                pin_memory=pin_memory,
                signature=signature,
            )
        else:
            shared_storage = None

        def _allocate_or_get_stage() -> torch.Tensor:
            if shared_storage is not None:
                return shared_storage
            storage = getattr(self, "_cpu_runtime_expert_stage_storage", None)
            storage_signature = getattr(
                self,
                "_cpu_runtime_expert_stage_signature",
                None,
            )
            storage_capacity = int(
                getattr(self, "_cpu_runtime_expert_stage_capacity", 0) or 0
            )
            if (
                storage is not None
                and storage_signature == signature
                and storage_capacity >= capacity
                and (not pin_memory or storage.is_pinned())
            ):
                return storage

            kwargs: dict[str, Any] = {
                "device": "cpu",
                "dtype": torch.uint8,
            }
            if pin_memory:
                kwargs["pin_memory"] = True
            storage = torch.empty(capacity * per_expert_bytes, **kwargs)
            self._cpu_runtime_expert_stage_storage = storage
            self._cpu_runtime_expert_stage_capacity = capacity
            self._cpu_runtime_expert_stage_signature = signature
            self._cpu_buffer_bytes = int(
                getattr(self, "_cpu_buffer_bytes", 0)
            ) + int(storage.numel())
            return storage

        lock = getattr(self, "_cpu_runtime_expert_stage_lock", None)
        if lock is None:
            storage = _allocate_or_get_stage()
        else:
            with lock:
                storage = _allocate_or_get_stage()

        def _copy_one(index_and_bundle: tuple[int, ExpertBundle]) -> None:
            expert_index, bundle = index_and_bundle
            dest_start = expert_index * per_expert_bytes
            dest = storage[dest_start:dest_start + per_expert_bytes]
            source_storage = bundle.storage
            source_offset = int(bundle.storage_offset_bytes)
            if self._source_has_contiguous_expert_storage(bundle, per_expert_bytes):
                dest.copy_(
                    source_storage[
                        source_offset:source_offset + per_expert_bytes
                    ]
                )
                return

            offset = 0
            for name in ordered_names:
                source = bundle.tensors[name].detach()
                if source.device.type != "cpu":
                    source = source.to(device="cpu")
                source = source.contiguous()
                num_bytes = int(source.numel() * source.element_size())
                view = dest[offset:offset + num_bytes].view(source.dtype).view(
                    source.shape
                )
                view.copy_(source)
                offset += num_bytes

        copy_threads = max(
            1,
            int(
                getattr(
                    self,
                    "_prepare_cpu_copy_threads",
                    getattr(self, "_prepare_cpu_copy_batch_size", 1),
                )
                or 1
            ),
        )
        copy_items = list(enumerate(bundles))
        used_native_copy = self._copy_contiguous_expert_storage_to_stage_native(
            bundles,
            storage,
            per_expert_bytes,
            copy_threads,
        )
        copy_mode = "native_cpu_copy"
        shared_pool = getattr(self, "_runtime_stage_pool", None)
        if used_native_copy:
            pass
        elif shared_pool is not None:
            copy_mode = "shared_pool_copy"
            shared_pool.map_cpu_stage_copies(
                _copy_one,
                copy_items,
                max_workers=copy_threads,
            )
        elif copy_threads > 1 and len(bundles) > 1:
            copy_mode = "python_threadpool_copy"
            def _copy_one_in_inference_mode(item: tuple[int, ExpertBundle]) -> None:
                with torch.inference_mode():
                    _copy_one(item)

            with ThreadPoolExecutor(
                max_workers=min(copy_threads, len(bundles))
            ) as executor:
                list(executor.map(_copy_one_in_inference_mode, copy_items))
        else:
            copy_mode = "serial_copy"
            for item in copy_items:
                _copy_one(item)

        stage_bundles: list[ExpertBundle] = []
        resolved_pinned = bool(storage.is_pinned())
        source_pinned = sum(
            1
            for bundle in bundles
            if bool(bundle.pinned)
            or (
                bundle.storage is not None
                and bundle.storage.device.type == "cpu"
                and bool(bundle.storage.is_pinned())
            )
        )
        self._last_runtime_cpu_stage_info = {
            "mode": copy_mode,
            "native_copy": bool(used_native_copy),
            "stage_pinned": bool(resolved_pinned),
            "source_pinned": int(source_pinned),
            "source_pageable": int(len(bundles) - source_pinned),
            "source_shared": bool(
                bundles[0].storage is not None
                and all(bundle.storage is bundles[0].storage for bundle in bundles)
            ),
            "source_contiguous": int(
                sum(
                    1
                    for bundle in bundles
                    if self._source_has_contiguous_expert_storage(
                        bundle,
                        per_expert_bytes,
                    )
                )
            ),
        }
        for expert_index in range(len(bundles)):
            expert_start = expert_index * per_expert_bytes
            expert_slice = storage[expert_start:expert_start + per_expert_bytes]
            views: dict[str, torch.Tensor] = {}
            offset = 0
            for name in ordered_names:
                source = first.tensors[name]
                num_bytes = int(source.numel() * source.element_size())
                views[name] = expert_slice[offset:offset + num_bytes].view(
                    source.dtype
                ).view(source.shape)
                offset += num_bytes
            stage_bundles.append(
                ExpertBundle(
                    tensors=views,
                    nbytes=per_expert_bytes,
                    pinned=resolved_pinned,
                    runtime_ready=True,
                    storage=storage,
                    storage_offset_bytes=expert_start,
                )
            )
        return stage_bundles

    def _stack_runtime_ready_cpu_field(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
            field_name: str,
            expert_indices: torch.Tensor | None = None,
            *,
            runtime_ready_stage_bundles: list[ExpertBundle] | None = None,
    ) -> torch.Tensor:
        require_pinned_batch = self._runtime_requires_pinned_cpu()
        packed_view = self._view_packed_runtime_ready_cpu_field(
            bundles_and_sources,
            field_name,
        )
        if packed_view is not None:
            if packed_view.is_pinned() or not require_pinned_batch:
                return packed_view
            batch = self._get_runtime_ready_batch_buffer(
                field_name,
                packed_view[0],
                len(bundles_and_sources),
                pin_memory=True,
            )
            batch.copy_(packed_view)
            return batch

        if runtime_ready_stage_bundles is not None:
            stage_bundles_and_sources = [
                (index, index, bundle, "cpu_static")
                for index, bundle in enumerate(runtime_ready_stage_bundles)
            ]
            stage_packed_view = self._view_packed_runtime_ready_cpu_field(
                stage_bundles_and_sources,
                field_name,
            )
            if stage_packed_view is not None:
                if stage_packed_view.is_pinned() or not require_pinned_batch:
                    return stage_packed_view

        layer_field_view = getattr(
            self,
            "_cpu_static_layer_field_views",
            {},
        ).get(field_name)
        if layer_field_view is not None:
            if expert_indices is None:
                expert_indices = self._build_runtime_ready_expert_indices(
                    bundles_and_sources
                )
            if expert_indices is not None and int(expert_indices.numel()) == len(
                bundles_and_sources
            ):
                use_pinned_batch = require_pinned_batch or layer_field_view.is_pinned()
                batch = self._get_runtime_ready_batch_buffer(
                    field_name,
                    layer_field_view[0],
                    len(bundles_and_sources),
                    pin_memory=use_pinned_batch,
                )
                torch.index_select(layer_field_view, 0, expert_indices, out=batch)
                return batch

        field_tensors = [
            bundle.tensors[field_name]
            for _expert_id, _slot, bundle, _source in bundles_and_sources
        ]
        first = field_tensors[0]
        use_pinned_batch = (
            require_pinned_batch
            or all(
                tensor.device.type == "cpu" and tensor.is_pinned()
                for tensor in field_tensors
            )
        )
        batch = self._get_runtime_ready_batch_buffer(
            field_name,
            first,
            len(field_tensors),
            pin_memory=use_pinned_batch,
        )

        for batch_idx, tensor in enumerate(field_tensors):
            cpu_tensor = (
                tensor
                if tensor.device.type == "cpu"
                else tensor.to(device="cpu", non_blocking=use_pinned_batch)
            )
            batch[batch_idx].copy_(cpu_tensor, non_blocking=use_pinned_batch)
        return batch

    def _stage_runtime_ready_cpu_fields_parallel(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
            field_specs: list[tuple[str, bool]],
            expert_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if not field_specs:
            return {}

        runtime_ready_stage_bundles = None
        if self._runtime_requires_pinned_cpu():
            runtime_ready_stage_bundles = self._prepare_runtime_ready_stage_bundles(
                bundles_and_sources,
                pin_memory=True,
            )

        if len(field_specs) == 1:
            field_name, optional = field_specs[0]
            stack_fn = (
                self._maybe_stack_runtime_ready_cpu_field
                if optional
                else self._stack_runtime_ready_cpu_field
            )
            stack_kwargs: dict[str, Any] = {}
            if (
                    runtime_ready_stage_bundles is not None
                    and _callable_accepts_keyword_argument(
                stack_fn,
                "runtime_ready_stage_bundles",
            )
            ):
                stack_kwargs["runtime_ready_stage_bundles"] = (
                    runtime_ready_stage_bundles
                )
            tensor = (
                stack_fn(
                    bundles_and_sources,
                    field_name,
                    expert_indices,
                    **stack_kwargs,
                )
            )
            return {field_name: tensor}

        field_batches: dict[str, torch.Tensor | None] = {}
        max_workers = min(len(field_specs), max(1, os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            def _submit_field(
                    fn: Any,
                    field_name: str,
                    optional: bool,
            ) -> Any:
                stack_kwargs: dict[str, Any] = {}
                if (
                        runtime_ready_stage_bundles is not None
                        and _callable_accepts_keyword_argument(
                    fn,
                    "runtime_ready_stage_bundles",
                )
                ):
                    stack_kwargs["runtime_ready_stage_bundles"] = (
                        runtime_ready_stage_bundles
                    )
                return executor.submit(
                    fn,
                    bundles_and_sources,
                    field_name,
                    expert_indices,
                    **stack_kwargs,
                )

            futures = {
                _submit_field(
                    self._maybe_stack_runtime_ready_cpu_field
                    if optional
                    else self._stack_runtime_ready_cpu_field,
                    field_name,
                    optional,
                ): field_name
                for field_name, optional in field_specs
            }
            for future in as_completed(futures):
                field_batches[futures[future]] = future.result()
        return field_batches

    def _maybe_stack_runtime_ready_cpu_field(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
            field_name: str,
            expert_indices: torch.Tensor | None = None,
            *,
            runtime_ready_stage_bundles: list[ExpertBundle] | None = None,
    ) -> torch.Tensor | None:
        if any(
                field_name not in bundle.tensors
                for _expert_id, _slot, bundle, _source in bundles_and_sources
        ):
            return None
        return self._stack_runtime_ready_cpu_field(
            bundles_and_sources,
            field_name,
            expert_indices,
            runtime_ready_stage_bundles=runtime_ready_stage_bundles,
        )

    def _write_unquantized_runtime_ready_bundles_batch(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
            target: Any,
            *,
            install_mappings: bool = False,
    ) -> dict[str, float]:
        return self._write_runtime_ready_bundles_via_gpu_stage(
            bundles_and_sources,
            target,
            install_mappings=install_mappings,
        )

    def _write_gptq_runtime_ready_bundles_batch(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
            target: Any,
            *,
            install_mappings: bool = False,
    ) -> dict[str, float]:
        return self._write_runtime_ready_bundles_via_gpu_stage(
            bundles_and_sources,
            target,
            install_mappings=install_mappings,
        )

    def _copy_resident_expert_to_target(
            self,
            expert_id: int,
            slot: int,
            target: Any,
    ) -> None:
        layer_key = _debug_layer_key(self)
        # 闂侇偅淇虹换鍐╃▔鐠囪尙婀撮柣?expert_map 闁瑰灚鍎抽崺宀€鎷?resident expert 鐟滅増鎸告晶鐘虫媴瀹ュ嫮鑹鹃柛婵愪簷闁?GPU slot闁?
        expert_to_slot = self._ensure_cpu_expert_to_slot()
        src_slot = (
            int(expert_to_slot[expert_id])
            if 0 <= expert_id < len(expert_to_slot)
            else -1
        )
        # 闁兼眹鍎叉竟妯荤▔瀹ュ懎鐓?resident slot闁挎稑鐭侀鈺呭及鎼淬倗娈堕柣顫妽閺岀喖鎮╅懜纰樺亾娴ｉ鐟濆☉鎾亾闁奸攱鐣埀?
        if src_slot < 0:
            raise RuntimeError(
                f"{self.layer_key}: expert {expert_id} is not resident in GPU slots"
            )
        with torch.no_grad():
            # 闂佹彃绻愮€佃尙鎹勯姘辩獮濞戞挸顑嗘俊鎼佸箥閳ь剟寮?runtime 闁烩晝顭堥崣褍顕ｉ悩璇叉閻庣懓鏈弳锝夊箯閻ゎ垳顦柛鎺撳濞蹭即寮介崶褜鍤犻悹鐏烘壋鍋?
            if self._mode == "gptq_marlin":
                target.w13_qweight[slot].copy_(self.layer.w13_qweight[src_slot])
                target.w2_qweight[slot].copy_(self.layer.w2_qweight[src_slot])
                target.w13_scales[slot].copy_(self.layer.w13_scales[src_slot])
                target.w2_scales[slot].copy_(self.layer.w2_scales[src_slot])
                target.w13_qzeros[slot].copy_(self.layer.w13_qzeros[src_slot])
                target.w2_qzeros[slot].copy_(self.layer.w2_qzeros[src_slot])
                if hasattr(target, "w13_g_idx"):
                    target.w13_g_idx[slot].copy_(self.layer.w13_g_idx[src_slot])
                    target.w2_g_idx[slot].copy_(self.layer.w2_g_idx[src_slot])
                    target.w13_g_idx_sort_indices[slot].copy_(
                        self.layer.w13_g_idx_sort_indices[src_slot]
                    )
                    target.w2_g_idx_sort_indices[slot].copy_(
                        self.layer.w2_g_idx_sort_indices[src_slot]
                    )
                return
            # 闂傚牏鍋ら崳娲礌閺嶎剛鐔呯€垫澘瀚ぐ褔妫侀埀顒傛啺娴ｇ懓顏?w13/w2 濞戞挶鍊曞?runtime 闁哄鍟撮崳鎼佸Υ?
            target.w13_weight[slot].copy_(self.layer.w13_weight[src_slot])
            target.w2_weight[slot].copy_(self.layer.w2_weight[src_slot])

    def _write_expert_bundle(self, slot: int, bundle: ExpertBundle, target: Any) -> None:
        if bundle.runtime_ready:
            self._write_runtime_ready_bundle(slot, bundle, target)
            return

        # ----------------- 闂佹彃绻愮€佃尙鎹勯姘辩獮闁挎稒鐡瑄ndle -> CPU raw -> GPU raw -> Marlin runtime 闁哄秶鍘х槐?-----------------
        if self._mode == "gptq_marlin":
            # 闁稿繐鐗婃俊?gate/up/down 濞戞挸顦崬銈夋煂韫囨挸顕х€殿喚濞€閸ｆ椽鏌屽鍥╃煁闁搞儳鍋涘畷?expert 闁汇劌瀚敮顐ｆ叏鐎ｎ剛娉㈤柡瀣閳?
            raw = self._assemble_raw_weights(bundle)
            # 闁告劕绉垫俊?raw 闁哄鍟撮崳绋款嚕閸屾侗鍔勯柟纰卞墮閸╁矂鎯勯鐣屽灱 GPU 閻犱焦鍎抽ˇ顒勫Υ?
            raw_gpu = self._move_raw_weights_to_device(raw)
            self._preprocess_marlin_fp8_raw_weights(raw_gpu)
            if self._gptq_desc_act:
                if raw_gpu.w13_g_idx is None or raw_gpu.w2_g_idx is None:
                    raise RuntimeError(
                        f"{self.layer_key}: desc_act=True requires g_idx tensors "
                        "during dynamic expert load"
                    )
                w13_g_idx_sort_indices = torch.argsort(
                    raw_gpu.w13_g_idx, dim=-1
                ).to(torch.int32)
                w2_g_idx_sort_indices = torch.argsort(
                    raw_gpu.w2_g_idx, dim=-1
                ).to(torch.int32)
                w13_sorted_g_idx = torch.gather(
                    raw_gpu.w13_g_idx, -1, w13_g_idx_sort_indices
                )
                w2_sorted_g_idx = torch.gather(
                    raw_gpu.w2_g_idx, -1, w2_g_idx_sort_indices
                )
            else:
                w13_g_idx_sort_indices = self._empty_perm(raw_gpu.w13_qweight.device)
                w2_g_idx_sort_indices = self._empty_perm(raw_gpu.w2_qweight.device)
                w13_sorted_g_idx = None
                w2_sorted_g_idx = None
            # qweight 闂傚洠鍋撻悷鏇氱窔閸ｆ悂寮?repack 闁?Marlin MoE kernel 闁告瑯鍨冲ú鍧楀箳閵夛妇啸閻犳劕婀卞▓鎴犳暜閸愩劎婀伴柕?
            repacked_w13 = ops.gptq_marlin_moe_repack(
                raw_gpu.w13_qweight,
                w13_g_idx_sort_indices,
                raw_gpu.w13_qweight.shape[1] * self.pack_factor,
                raw_gpu.w13_qweight.shape[2],
                self.num_bits,
                is_a_8bit=self.is_a_8bit,
            )
            # w2 濞戞梻鍠曢々锕傛偑椤掑倻褰岄柛瀣煯缁旀挳鏌?repack闁?
            repacked_w2 = ops.gptq_marlin_moe_repack(
                raw_gpu.w2_qweight,
                w2_g_idx_sort_indices,
                raw_gpu.w2_qweight.shape[1] * self.pack_factor,
                raw_gpu.w2_qweight.shape[2],
                self.num_bits,
                is_a_8bit=self.is_a_8bit,
            )
            # scale 閺夆晜锕㈠〒鍓佹啺娴ｇ懓鐦婚柣?Marlin 闁汇劌瀚鏍偓娑櫭粩椋庝沪閳ь剟宕戝顐ゎ伇婵?permute闁?
            permuted_w13_scales = marlin_moe_permute_scales(
                s=raw_gpu.w13_scales,
                size_k=self.layer.intermediate_size_per_partition,
                size_n=raw_gpu.w13_scales.shape[2],
                group_size=self.group_size,
                is_a_8bit=self.is_a_8bit,
            )
            # w2 scale 闁?size_k 閻庤鐭粻鐔哥▔?w13 濞戞挸绉撮幃鎾绘晬瀹€鍐闂佹彃鏈€垫粍绋夌€ｎ偄顫岀憸鎷屼含濞ｎ喗鎯旈敃鈧畷鐔兼偑椤掑喚鍚€缂佺姵銇滈埀?
            permuted_w2_scales = marlin_moe_permute_scales(
                s=raw_gpu.w2_scales,
                size_k=raw_gpu.w2_scales.shape[1]
                       * (self.group_size if self.group_size != -1 else self.pack_factor),
                size_n=raw_gpu.w2_scales.shape[2],
                group_size=self.group_size,
                is_a_8bit=self.is_a_8bit,
            )
            # 闁哄牃鍋撻柛姘婵?repack/permute 闁告艾娴峰▓鎴犵磼閹惧浜柛鎰懆缁绘﹢鎯勯鐣屽灱 GPU slot闁?
            self._write_quantized_target_slot(
                target=target,
                slot=slot,
                repacked_w13=repacked_w13[0],
                repacked_w2=repacked_w2[0],
                permuted_w13_scales=permuted_w13_scales[0],
                permuted_w2_scales=permuted_w2_scales[0],
                raw_gpu=raw_gpu,
                w13_g_idx=w13_sorted_g_idx[0] if w13_sorted_g_idx is not None else None,
                w2_g_idx=w2_sorted_g_idx[0] if w2_sorted_g_idx is not None else None,
                w13_g_idx_sort_indices=(
                    w13_g_idx_sort_indices[0]
                    if self._gptq_desc_act
                    else None
                ),
                w2_g_idx_sort_indices=(
                    w2_g_idx_sort_indices[0]
                    if self._gptq_desc_act
                    else None
                ),
            )
            return

        # ----------------- 闂傚牏鍋ら崳娲礌閺嶎剛鐔呯€垫澘瀚哥槐鐧皍ndle -> CPU raw -> GPU raw -> runtime kernel 闁哄秶鍘х槐?-----------------
        # 闁稿繐鐗婃俊?gate/up/down 濞戞挸顦崬銈夊储閻斿娼楅柡澶婂暣閸ｆ悂骞忛崗鐓庣亣閺夆晜鍔橀、鎴﹀籍閸洘浠橀悷鏇氳兌濞?w13/w2 缂備焦鎸婚悗顖炲Υ?
        raw = self._assemble_unquantized_weights(bundle)
        # 闁告劕绉垫俊鎼佸箯閻撳簺鍋ㄩ柣銊ュ鐢偅鎱ㄧ€ｎ偅缍€闂佹彃绉甸幆澶愬礆?GPU闁?
        raw_gpu = self._move_unquantized_weights_to_device(raw)
        # 濞戞挸绉撮幃?backend 闁告瑯鍨甸崗姗€妫侀埀顒傛啺娓氣偓椤ゅ倹寰勯弽銊у鐎殿喖绻楀ù鍡涘箲椤喚绀夐弶鈺傜懇閸ｉ绱掗悢鍓侇伇閻?helper闁?
        runtime_w13, runtime_w2 = convert_to_unquantized_kernel_format(
            self.quant_method.unquantized_backend,
            layer=self.layer,
            w13_weight=raw_gpu.w13_weight,
            w2_weight=raw_gpu.w2_weight,
        )
        # 闁?runtime 闁哄秶鍘х槐锟犳儍閸曨剚缍€闂佹彃绉撮崯鎾存交濞戞碍绐楅柡?slot闁?
        self._write_unquantized_target_slot(
            target=target,
            slot=slot,
            runtime_w13=runtime_w13[0],
            runtime_w2=runtime_w2[0],
        )

    def _write_runtime_ready_bundle(
            self,
            slot: int,
            bundle: ExpertBundle,
            target: Any,
    ) -> None:
        self._write_runtime_ready_bundles_via_gpu_stage(
            [(slot, slot, bundle, "cpu_static")],
            target,
        )

    def _write_quantized_target_slot(
            self,
            *,
            target: Any,
            slot: int,
            repacked_w13: torch.Tensor,
            repacked_w2: torch.Tensor,
            permuted_w13_scales: torch.Tensor,
            permuted_w2_scales: torch.Tensor,
            raw_gpu: _RawExpertWeights,
            w13_g_idx: torch.Tensor | None = None,
            w2_g_idx: torch.Tensor | None = None,
            w13_g_idx_sort_indices: torch.Tensor | None = None,
            w2_g_idx_sort_indices: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            # 濞撴碍绻冮鑲╂啺閸℃瑦纾伴柣鈺婂枟閻?slot 濞戞搩鍘惧▓?runtime 闂佹彃绻愮€垫煡寮堕崘顔兼濞?scale闁?
            target.w13_qweight[slot].copy_(repacked_w13)
            target.w2_qweight[slot].copy_(repacked_w2)
            target.w13_scales[slot].copy_(permuted_w13_scales)
            target.w2_scales[slot].copy_(permuted_w2_scales)
            # qzeros 濞戞挸绉村顒佺▔?repack/permute闁挎稑鐬煎ú鍧楀箳閵夛箓鍎撮柣?raw 闁哄秶鍘х槐锟犲礃濞嗗繐寮抽柛妤€鍟胯ぐ鏌ュΥ?
            target.w13_qzeros[slot].copy_(raw_gpu.w13_qzeros[0])
            target.w2_qzeros[slot].copy_(raw_gpu.w2_qzeros[0])
            if (
                    w13_g_idx is not None
                    and w2_g_idx is not None
                    and w13_g_idx_sort_indices is not None
                    and w2_g_idx_sort_indices is not None
                    and hasattr(target, "w13_g_idx")
            ):
                target.w13_g_idx[slot].copy_(w13_g_idx)
                target.w2_g_idx[slot].copy_(w2_g_idx)
                target.w13_g_idx_sort_indices[slot].copy_(w13_g_idx_sort_indices)
                target.w2_g_idx_sort_indices[slot].copy_(w2_g_idx_sort_indices)

    def _write_unquantized_target_slot(
            self,
            *,
            target: Any,
            slot: int,
            runtime_w13: torch.Tensor,
            runtime_w2: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            # 闂傚牏鍋ら崳娲礌閺嶎剛鐔呯€垫澘瀚ú鍧楀箳閵夘煈娲柣鈺傜墱濞蹭即寮?slot 閻庣數鎳撶花鏌ユ儍?w13/w2 runtime 闁哄鍟撮崳鎼佸Υ?
            target.w13_weight[slot].copy_(runtime_w13)
            target.w2_weight[slot].copy_(runtime_w2)

    def _assemble_raw_weights(
            self,
            bundle: ExpertBundle,
            *,
            raw: _RawExpertWeights | None = None,
            expert_index: int = 0,
    ) -> _RawExpertWeights:
        # ------------------------------- 閻犲洩顕цぐ鍥亹閹惧啿顤?bundle 闁汇劌瀚槐鍫曟煂韫囨挾鎽熼柛蹇曨焾閼荤喐锛愰崟顕呭悁缂?w13 闁告艾鐗嗛懟鐔烘暜閸愩劎婀伴柛娆忓€归弳?-------------------------------
        # 闁告瑦鐗曢崵顓°亹閹惧啿顤?expert bundle 濞戞搩鍘虹换姘扁偓娑欘焽濞堟垵顕ｉ悩璇叉閻庢稒顨呴崥鈧柨娑樿嫰閹绱掗鐔风樆閻庢稒顨嗛宀勫触瀹ュ鍋撻幇顏堝殝闁圭柉澹堥ˉ濠囧Υ?
        tensors = bundle.tensors
        """
        qweight 闁?pack闁哄倻鎳撻幃婊勭▔妤︽鏀?
        qzero 闁?pack闁哄倻鎳撻幃婊勭▔閸濆嫬鐏?
        """
        # 閻犱緤绱曢悾濠氬触閸繆瀚欓柛?w13 闁汇劌瀚埀顒冾嚙閸亪寮敮顔剧閻庣數鎳撶花?gate_proj 濞?up_proj 濞戞挶鍊曞畷鎰板箯閸忕厧澶嶉柛姘捣濞堟垿骞€鐠囧樊鍟嶉幖杈捐礋閳?
        w13_cols = 2 * self.layer.intermediate_size_per_partition

        # 閻犱緤绱曢悾?w13 闁告挸绉村畷鎰版焾閵娿儱鐎婚柣銊ュ閸亪寮敮顔剧閻庣數鎳撶花鏌ュ础閺囨岸鍤?gate_proj 闁?up_proj 闁汇劌瀚鏃€鎯旈敂琛″亾?
        w13_half_cols = self.layer.intermediate_size_per_partition

        # 閻犱緤绱曢悾濠氬触閸繆瀚欓柛?w13_qzeros 闁汇劌瀚埀顒冾嚙閸亪寮敮顔剧闁稿繗娉涢崹顏堝极閻楀牆鐦?pack_factor 闁告ê顑囩紓澶愬Υ?
        w13_qzeros_cols = w13_cols // self.pack_factor

        # 閻犱緤绱曢悾?w13 闁告挸绉村畷鎰版焾閵娿儱鐎?qzeros 闁汇劌瀚崹顏堝极鐢喚绀夐柛蹇氭硾閸亪寮弶鎸庡€遍柡宥夋敱鐎?pack_factor 闁告ê顑囩紓澶愬Υ?
        w13_half_qzeros_cols = w13_half_cols // self.pack_factor

        # ------------------------------- 闁哄稄绻濋悰娆撴煂韫囨挸顕ч悹渚灠缁剁偤骞嶉埀顒勬閳ь剟鎯?CPU raw buffer 闁哄嫷鍨伴幆浣割啅閼碱剛鐥呴柛鎺戞閸?-------------------------------
        # 鐟滅増鎹囬崳娲礌閺嶎剛鐔呯€垫澘瀚粭鍛存儍?CPU raw buffer 閻忓繑纰嶅﹢顓㈠礆濠靛棭娼楅柛鏍ㄧ墬濡炲倿鏁嶇仦鐣岀Ъ闁?expert 濞戞挸绉烽崗妯肩磼瑜忛悽濠氬箥瑜戦、鎴﹀礉閵婏腹鍋撴担绋款潱閺夌偠濮ょ€氬墽鎲楅崨顐熷亾?
        if raw is None:
            if self._cpu_quantized_raw_buffer is None:
                raise RuntimeError(f"{self.layer_key}: quantized CPU raw buffer is missing")
            raw = self._cpu_quantized_raw_buffer

        # ------------------------------- 闁告帗绻傞～鎰板礌閺嵮呮憻婵炲牓娼ч悾顒勫极鐎涙ǚ鍋撹椤ュ懘寮婚妷褍笑闁?-------------------------------
        # 闁活潿鍔嬬花顒傛媼閺夎法绉?gate_proj闁靛棔鑿噋_proj闁靛棔榫歰wn_proj 闁汇劌瀚崣褔鏌ㄩ鑲╂憻婵炲牆鐏氬Σ鎼佸触閿曗偓閸戯紕绱掕箛鎾冲伎闂侇喓鍔岄崯鎾诲礂?raw buffer闁?
        seen_fields: set[tuple[str, str]] = set()

        # 閻犱焦婢樼紞?gate/up 闁告艾鐗嗛懟鐔烘崉椤栨氨绐為柡鍕靛灠閹礁顔忛懠顒傜梾闁活亜顑呴崺?w13 閻庣數鎳撶花鏌ユ儍?g_idx 閻庢稒顨嗛宀勫Υ?
        seen_w13_g_idx = False

        # 閻犱焦婢樼紞?down 閻犱警鍨扮欢鐐哄及椤栨碍鍎婄€规瓕灏欑划锟犳儑鐎ｎ亜鐓?w2 閻庣數鎳撶花鏌ユ儍?g_idx 閻庢稒顨嗛宀勫Υ?
        seen_w2_g_idx = False

        # ------------------------------- 闂侇偅鍔曠槐鍫曟煂韫囨挾娈?gate闁靛棔鑿噋闁靛棔榫歰wn 闁哄鍟撮崳鎼佸箯閼归偊妫呴柛?raw buffer 濞?-------------------------------
        # 闂侇剙绉村鏄忋亹閹惧啿顤?bundle 濞戞搩鍘惧▓鎴﹀礂閵娾晛鍔ョ€殿喚濞€閸ｆ椽寮堕敍鍕獥闁挎稑鏈€垫粎鈧稒顨嗛宀勫触瀹ュ懐娈洪柛蹇氭硾閸熸捇宕楅妷銉殸閹煎瓨姊诲▓?raw buffer 闁告牕鎼悡娆撳Υ?
        for relative_name, tensor in tensors.items():
            # 闁告绮敮鈧?slot 闁告挸绉剁槐鎴︽晬鐏炶棄娑уǎ鍥ㄧ箘閺嗏偓闁硅埖娲栨總鏍触瀹ュ嫮鐟㈤悗娑欘殕椤斿矂宕ュ澶婂姤闁告帒妫庨埀?
            _, suffix = relative_name.split(".", 1)

            # 閻忓繐妫楁晶鎸庢媴濞嗘挸鍔ラ柛鎺戞婵爼骞嬮幇顓烆潓鐟滄澘宕幃鏇㈠椽鐏炵晫鎽熸繛鍫ユ涧閹洘绋夐妶鍡╁斀闁?
            proj_name, field_name = suffix.split(".", 1)

            # 濞戞捁妗ㄧ换姘辨嫚娴ｅ憡鍊电紓?H2D 閻犱警鍨扮欢鐐寸▔閳ь剟鎳涙潏鍓х闁稿繐鐗忛垾妯荤┍濠靛棛绉奸柛鎾崇У缁喖顕ｉ悩璇叉濞达絽绉崇花?CPU 濞戞挸顭堥埀?
            cpu_tensor = tensor if tensor.device.type == "cpu" else tensor.to(device="cpu")

            # 鐟滅増鎸哥紞瀣礈瀹ュ懐鐐婇梺鎻掔箰閻ɑ绂?gate_proj 闁哄啳顔愮槐婵堜焊閸℃寰撻柛鎰懃閸?w13 闁汇劌瀚晶鐘诲础婵犲洤鍔ラ柛鎺戞閳?
            if proj_name == "gate_proj":
                self._copy_merged_half(
                    field_name,
                    cpu_tensor,
                    raw,
                    expert_index=expert_index,
                    offset=0,
                    qzeros_offset=0,
                )
                seen_fields.add((proj_name, field_name))

            # 鐟滅増鎸哥紞瀣礈瀹ュ懐鐐婇梺鎻掔箰閻ɑ绂?up_proj 闁哄啳顔愮槐婵堜焊閸℃寰撻柛鎰懃閸?w13 闁汇劌瀚幃妤呭础婵犲洤鍔ラ柛鎺戞閳?
            elif proj_name == "up_proj":
                self._copy_merged_half(
                    field_name,
                    cpu_tensor,
                    raw,
                    expert_index=expert_index,
                    offset=w13_half_cols,
                    qzeros_offset=w13_half_qzeros_cols,
                )
                seen_fields.add((proj_name, field_name))

            # 鐟滅増鎸哥紞瀣礈瀹ュ懐鐐婇梺鎻掔箰閻ɑ绂?down_proj 闁哄啳顔愮槐婵堜焊閸℃寰撻柣鈺佺摠鐢挳宕樺▎蹇撳汲 w2 閻庣數鎳撶花鑼偓娑欘殕椤斿矂濡?
            elif proj_name == "down_proj":
                self._copy_direct(
                    field_name,
                    cpu_tensor,
                    raw,
                    expert_index=expert_index,
                )
                seen_fields.add((proj_name, field_name))

            # ------------------------------- 閻犱焦婢樼紞?g_idx 閻庢稒顨嗛宀勬儍閸曨偅鍤掑☉鎿冨幗閸庡繘宕?-------------------------------
            # 鐟滅増鎸哥紞瀣礈瀹ュ懐鎽熸繛鍫ユ涧閹洘绋?g_idx 闁哄啳顔愮槐婵嬪箰婢跺顫岀憸鎷屼含鐞氼偊宕圭€ｎ亜鐎婚柛鎺濆亯椤斿洩銇?w13 闁?w2 閻犱警鍨扮欢鐐哄及椤栨碍鍎婄€规瓕灏欏﹢鍛村礆閺夋鍤犻幖瀛樻煥閻⊙冣枔閻愬厜鍋?
            if field_name == "g_idx":
                if proj_name in ("gate_proj", "up_proj"):
                    seen_w13_g_idx = True
                elif proj_name == "down_proj":
                    seen_w2_g_idx = True

        # ------------------------------- 闁哄稄绻濋悰娆撳礉閵婏腹鍋撴担绋款潱閺夌偠濮ゆ晶宥夋閳ь剟鎯冮崟顐㈠綘闂佹鍠栭悺褍鈻撻崹顐Ｐ﹂柛姘剧畵缂嶅牓宕?-------------------------------
        # 閻庤鐭粻鐔兼煂韫囨挸顕ч柛鏂诲妽閳ь兛绀佹慨鐐存姜閸婄喓鐔呯€垫澘瀚々锕€效閸屾氨绠戝銈堫嚙閸欐寧寰勯崶鈺傜暠闁糕晞娅ｉ、鍛偓娑欘殕椤斿矂姊块崱妤佸€ら柕?
        required_fields = {
            ("gate_proj", "qweight"),
            ("gate_proj", "scales"),
            ("gate_proj", "qzeros"),
            ("up_proj", "qweight"),
            ("up_proj", "scales"),
            ("up_proj", "qzeros"),
            ("down_proj", "qweight"),
            ("down_proj", "scales"),
            ("down_proj", "qzeros"),
        }

        # 閻犱緤绱曢悾鏄忋亹閹惧啿顤?bundle 濞戞搩鍘惧杈ㄥ緞鏉堚晜鐣遍柛蹇斿▕閺侇厾鈧稒顨嗛宀勬⒖閸℃鍊ら柕?
        missing_fields = required_fields - seen_fields

        # 鐟滅増鎸搁悺銊╁捶閵娧冪箒濠㈣泛宕悺褍鈻撻崹顐ｎ槯闁挎稑鐬煎ú鍧楀箳閵夛箑袚闂佹寧鐟ラ懟鐔虹磼閸噥鍓剧憸鐗堟尭婢х娀宕濋妸锔瑰亾娴ｇ顫ｉ弶鐐差潟閳?
        if missing_fields:
            raise KeyError(
                f"{self.layer_key}: missing expert tensors for dynamic load: "
                f"{sorted(missing_fields)}"
            )

        # 鐟滅増鎸哥紞瀣礈瀹ュ牏鐔呯€垫澘瀚幆搴ㄦ偨閵娿倗鍟?desc_act闁挎稑濂旂徊鍓х磽閸濆嫮姣?w13 闁?w2 閻犱警鍨扮欢鐐哄箥閳ь剟妫侀埀顒勬儍?g_idx 閻庢稒顨嗛宀勫籍鐠佸湱绀夐柣鈺佺摠鐢挳骞庨妷鈺傛櫓闁?
        if self._gptq_desc_act and (not seen_w13_g_idx or not seen_w2_g_idx):
            raise KeyError(
                f"{self.layer_key}: missing expert g_idx tensors for dynamic load"
            )

        # ------------------------------- 閺夆晜鏌ㄥú鏍亹閹惧啿顤呯€圭寮剁€氬墽鎲楅崨顓犳殮闁瑰瓨鍔楀▓鎴︽煂韫囨挸顕?raw buffer -------------------------------
        # 閺夆晜鏌ㄥú鏍ь啅閼碱剛鐥呴悗鐟版湰閸?gate闁靛棔鑿噋闁靛棔榫歰wn 闁告艾鐗嗛懟鐔煎礃濞嗗繐寮抽柣銊ュ閸ｆ椽宕?CPU raw buffer闁?
        return raw

    def _copy_merged_half(
            self,
            field_name: str,
            tensor: torch.Tensor,
            raw: _RawExpertWeights,
            *,
            expert_index: int,
            offset: int,
            qzeros_offset: int,
    ) -> None:
        # ------------------------------- 閻?gate 闁?up 闁告帒妫欓弫顔锯偓娑欘殕椤斿矂宕樺▎蹇撳汲闁告艾鐗嗛懟鐔煎触鎼达絾鐣?w13 閻庣數鎳撶花鏌ュ础婵犲倸闅?-------------------------------
        # 鐟滅増鎸哥紞瀣礈瀹ュ懐鎽熸繛鍫濆悁鐠?qweight 闁哄啳顔愮槐婵堜焊閸℃凹鍤夌€殿喚濞€閸ｆ椽骞愭径濠傜仚闁稿绻掍簺闁告劖鐟ラ崣?raw.w13_qweight 闁汇劌瀚顔芥償閺傚灝纾归柛鏍ㄤ航閳?
        if field_name == "qweight":
            raw.w13_qweight[
            expert_index,
            :,
            offset: offset + tensor.shape[-1],
            ].copy_(tensor)

        # 鐟滅増鎸哥紞瀣礈瀹ュ懐鎽熸繛鍫濆悁鐠?scales 闁哄啳顔愮槐婵堜焊閸℃凹鍤夌€殿喚濞€閸ｆ椽骞愭径濠傜仚闁稿绻掍簺闁告劖鐟ラ崣?raw.w13_scales 闁汇劌瀚顔芥償閺傚灝纾归柛鏍ㄤ航閳?
        elif field_name == "scales":
            raw.w13_scales[
            expert_index,
            :,
            offset: offset + tensor.shape[-1],
            ].copy_(tensor)

        # ------------------------------- 閻忓繐妫楃敮鍥╃磽閳轰礁鐏欓柡浣瑰濞?qzeros 闁告劖鐟ラ崣鍡涘触閸繆瀚欓柛姘捣濞?w13 閻庣數鎳撶花鏌ュ础婵犲倸闅?-------------------------------
        # 鐟滅増鎸哥紞瀣礈瀹ュ懐鎽熸繛鍫濆悁鐠?qzeros 闁哄啳顔愮槐婵嬪箰?qzeros 濞戞挻鎸鹃弫銈夊礆濡も偓娴滃摜绮旂拠鎻掓櫢闁?raw.w13_qzeros 闁汇劌瀚顔芥償閺傚灝纾归柛鏍ㄤ航閳?
        elif field_name == "qzeros":
            raw.w13_qzeros[
            expert_index,
            :,
            qzeros_offset: qzeros_offset + tensor.shape[-1],
            ].copy_(tensor)

        # ------------------------------- 閻?g_idx 闁告劖鐟ラ崣?w13 闁汇劌瀚敮顐ｆ叏鐎ｎ剙鍋嶇€殿喗娲滅槐锕傚礃閹绘帒闅?-------------------------------
        # 鐟滅増鎸哥紞瀣礈瀹ュ懐鎽熸繛鍫濆悁鐠?g_idx 闁哄啳顔愮槐婵嬫閳ь剛鎲版担绋垮弗缁绢収鍠涢?w13_g_idx 闁告鍠庨～鎰磽閹惧啿鏆遍柛鏍ф惈閸戯紕绱掕箛鎾崇€婚梺鏉跨Т閻ｎ剟骞嬮幇鈹惧亾?
        elif field_name == "g_idx":
            if raw.w13_g_idx is None:
                raise RuntimeError(f"{self.layer_key}: w13_g_idx raw buffer is missing")

            # 閻忓繐妫楃紞瀣礈?g_idx 鐎殿喚濞€閸ｆ椽寮弶鎴炲仴闁告劖鐟ラ崣?raw.w13_g_idx闁?
            raw.w13_g_idx[expert_index].copy_(tensor)

    def _copy_direct(
            self,
            field_name: str,
            tensor: torch.Tensor,
            raw: _RawExpertWeights,
            *,
            expert_index: int,
    ) -> None:
        # down_proj 濞戞挸绉瑰〒鍓佹啺娴ｅ憡鍊ゆ鐐舵硾娴滃摜绮旀导娆戠闁烩晛鐡ㄧ敮鎾极閺夋垶鍋ラ柛鎰懃閸?w2 閻庣數鎳撶花鑼偓娑欘殕椤斿矂濡?
        if field_name == "qweight":
            raw.w2_qweight[expert_index].copy_(tensor)
        elif field_name == "scales":
            raw.w2_scales[expert_index].copy_(tensor)
        elif field_name == "qzeros":
            raw.w2_qzeros[expert_index].copy_(tensor)
        elif field_name == "g_idx":
            if raw.w2_g_idx is None:
                raise RuntimeError(f"{self.layer_key}: w2_g_idx raw buffer is missing")
            raw.w2_g_idx[expert_index].copy_(tensor)

    def _move_raw_weights_to_device(self, raw: _RawExpertWeights) -> _RawExpertWeights:
        # ------------------------------- 閻忓繐妫濋崳娲礌閺嵮冩枾濠殿喖顑嗗鍫ユ煂瀹ュ洨澶勯柛鎰絻鐏忣垶寮紙鐘电Ъ闁圭⒈鍓濈换宥夊礆閹殿喗绐楅柡?GPU 閻犱焦鍎抽ˇ?-------------------------------
        # 闁哄瀚伴埀顒傚Т閼荤喐娼婚弬鎸庣濞戞挴鍋撳ù鐘哄Г閺屽﹪鎯冮崟顐㈡枾濠殿喖顑嗗鍫ユ煂瀹ュ拋鍤犻悹鐐╂缁辨繈宕楅張浣冨幀闁告艾瀚悺褍鈻撻悽绋垮幋鐎规瓕寮撶划?CPU 鐎殿喖鍊归鐐哄箹椤掑啰绠ラ柛鎺撴緲缂嶅宕滃鍡椾粯闁告帟娉涘▍鎺旂磼閹存繄鏆伴柣銊ュ濞蹭即寮?GPU 閻犱焦鍎抽ˇ顒勫Υ?
        return _RawExpertWeights(
            # 閻?w13 闂佹彃绻愮€垫煡寮堕崘顔兼鐎殿喖鍊归鐐哄箹椤掑啰绠ラ柛鎺撳濞蹭即寮?GPU闁?
            w13_qweight=raw.w13_qweight.to(device=self.device, non_blocking=True),

            # 閻?w2 闂佹彃绻愮€垫煡寮堕崘顔兼鐎殿喖鍊归鐐哄箹椤掑啰绠ラ柛鎺撳濞蹭即寮?GPU闁?
            w2_qweight=raw.w2_qweight.to(device=self.device, non_blocking=True),

            # 閻?w13 闁?scale 鐎殿喚濞€閸ｅ搫顕ｉ崒娑卞妱闁圭⒈鍓濈换宥夊礆閹殿喗绐楅柡?GPU闁?
            w13_scales=raw.w13_scales.to(device=self.device, non_blocking=True),

            # 閻?w2 闁?scale 鐎殿喚濞€閸ｅ搫顕ｉ崒娑卞妱闁圭⒈鍓濈换宥夊礆閹殿喗绐楅柡?GPU闁?
            w2_scales=raw.w2_scales.to(device=self.device, non_blocking=True),

            # 閻?w13 闁?qzeros 鐎殿喚濞€閸ｅ搫顕ｉ崒娑卞妱闁圭⒈鍓濈换宥夊礆閹殿喗绐楅柡?GPU闁?
            w13_qzeros=raw.w13_qzeros.to(device=self.device, non_blocking=True),

            # 閻?w2 闁?qzeros 鐎殿喚濞€閸ｅ搫顕ｉ崒娑卞妱闁圭⒈鍓濈换宥夊礆閹殿喗绐楅柡?GPU闁?
            w2_qzeros=raw.w2_qzeros.to(device=self.device, non_blocking=True),

            # 鐟?w13_g_idx 閻庢稒锚濠€顏堝籍鐠佸湱绀夐悘蹇撴閸欐儳顕ｉ崒娑卞妱闁圭⒈鍓濈换宥夊礆閹殿喗绐楅柡?GPU闁挎稒绋戦幆渚€宕氬▎搴ｇ闁归晲妞掔拹?None闁?
            w13_g_idx=(
                raw.w13_g_idx.to(device=self.device, non_blocking=True)
                if raw.w13_g_idx is not None
                else None
            ),

            # 鐟?w2_g_idx 閻庢稒锚濠€顏堝籍鐠佸湱绀夐悘蹇撴閸欐儳顕ｉ崒娑卞妱闁圭⒈鍓濈换宥夊礆閹殿喗绐楅柡?GPU闁挎稒绋戦幆渚€宕氬▎搴ｇ闁归晲妞掔拹?None闁?
            w2_g_idx=(
                raw.w2_g_idx.to(device=self.device, non_blocking=True)
                if raw.w2_g_idx is not None
                else None
            ),
        )

    def _assemble_unquantized_weights(
            self, bundle: ExpertBundle
    ) -> _RawUnquantizedExpertWeights:
        # ----------------- 閻犱緤绱曢悾濠氭閻愬搫娅ら柛?w13 闁汇劌瀚€氶箖骞掗妷銉︽閻?-----------------
        tensors = bundle.tensors
        is_act_and_mul = bool(self.layer.moe_config.is_act_and_mul)
        w13_up_dim = (
            2 * self.layer.intermediate_size_per_partition
            if is_act_and_mul
            else self.layer.intermediate_size_per_partition
        )
        half_dim = self.layer.intermediate_size_per_partition
        # 闂傚牏鍋ら崳娲礌閺嶎剛鐔呯€垫澘瀚幃鎾诲冀閻ゎ垼娲ｆ慨鐟板€块。鈺呭礆閸℃稑甯?CPU raw buffer闁?
        if self._cpu_unquantized_raw_buffer is None:
            raise RuntimeError(
                f"{self.layer_key}: unquantized CPU raw buffer is missing"
            )
        raw = self._cpu_unquantized_raw_buffer
        seen_fields: set[tuple[str, str]] = set()

        # ----------------- 闁?gate/up/down 闁哄鍟撮崳鎼佸箯閸忕厧鐏?runtime 闁圭鍋撻梻鍥ｅ亾闁?w13/w2 -----------------
        for relative_name, tensor in tensors.items():
            _, suffix = relative_name.split(".", 1)
            proj_name, field_name = suffix.split(".", 1)
            # 闂傚牏鍋ら崳娲礌閺嶎剛鐔呯€垫澘瀚ぐ褍鈽夐崼锝呯€柛妯煎枎椤?weight 閻庢稒顨嗛宀勫Υ?
            if field_name != "weight":
                continue
            cpu_tensor = tensor if tensor.device.type == "cpu" else tensor.to(device="cpu")
            # gate_proj 闁告劖鐟ラ崣?w13 闁告挸绉村畷鎰版焾閵娿儱鐎婚柕?
            if proj_name == "gate_proj":
                raw.w13_weight[0, :half_dim].copy_(cpu_tensor)
                seen_fields.add((proj_name, field_name))
            # up_proj 闁告劖鐟ラ崣?w13 闁告艾楠稿畷鎰版焾閵娿儱鐎婚柕?
            elif proj_name == "up_proj":
                raw.w13_weight[0, half_dim: half_dim + cpu_tensor.shape[0]].copy_(
                    cpu_tensor
                )
                seen_fields.add((proj_name, field_name))
            # down_proj 闁烩晛鐡ㄧ敮鎾礃濞嗗繐寮?w2闁?
            elif proj_name == "down_proj":
                raw.w2_weight[0].copy_(cpu_tensor)
                seen_fields.add((proj_name, field_name))

        # 缂傚倽妗ㄩ幑銏ゅ箛韫囧海顏遍柛褎顨嗗鍫ユ煂瀹ュ鍘村☉鎾崇Х閸忔鈧懓鏈崹姘跺礉閵婏腹鍋撴担绋款潱閺夌偛顫曢埀?
        required_fields = {
            ("gate_proj", "weight"),
            ("up_proj", "weight"),
            ("down_proj", "weight"),
        }
        missing_fields = required_fields - seen_fields
        if missing_fields:
            raise KeyError(
                f"{self.layer_key}: missing expert tensors for dynamic load: "
                f"{sorted(missing_fields)}"
            )

        return raw

    def _move_unquantized_weights_to_device(
            self, raw: _RawUnquantizedExpertWeights
    ) -> _RawUnquantizedExpertWeights:
        # 闁硅泛锕顏堟煂韫囨挸顕?raw buffer 鐎殿喖鍊归鐐哄箹椤掆偓閸╁矂鎯勯鐣屽灱 GPU闁?
        return _RawUnquantizedExpertWeights(
            w13_weight=raw.w13_weight.to(device=self.device, non_blocking=True),
            w2_weight=raw.w2_weight.to(device=self.device, non_blocking=True),
        )

    def _empty_perm(
            self,
            device: torch.device,
            num_experts: int = 1,
    ) -> torch.Tensor:
        # 鐟滅増鎸告晶鐘诲礉閵婏腹鍋撴担绋款潱閺夌偛鈧喓鐔呯€垫澘瀚粭澶嬪緞瀹ュ洦鏆忓Λ鏉垮椤撳摜绮?perm闁挎稑濂旂槐鑸电▔閳ь剚绋夐鍡忔晞闁告濮崇紞鍛嚕閻樿娅ょ紓?Marlin repack 闁规亽鍎辫ぐ娑㈠Υ?
        return torch.empty((num_experts, 0), dtype=torch.int32, device=device)

    def _resolve_marlin_ready_cache_root(self) -> Path | None:
        if self._marlin_ready_cache_disabled:
            return None
        configured = os.getenv("CFIE_MARLIN_READY_CACHE_DIR", "").strip()
        if configured:
            return Path(configured).expanduser()
        model_path = str(self.plan.get("model_path", "") or "").strip()
        if not model_path:
            return None
        return (
            Path(model_path).expanduser()
            / ".cfie_marlin_ready_cache"
            / f"v{MARLIN_READY_EXPERT_CACHE_VERSION}"
        )

    def _marlin_ready_layer_cache_path(self) -> Path | None:
        return self._marlin_ready_layer_cache_path_with_suffix(".safetensors")

    def _marlin_ready_layer_cache_path_with_suffix(
            self,
            suffix: str,
    ) -> Path | None:
        root = getattr(self, "_marlin_ready_cache_root", None)
        layer_index = getattr(self, "_moe_layer_index", None)
        if root is None or layer_index is None:
            return None
        input_dtype_name = getattr(self, "_marlin_input_dtype_name", "a16")
        if input_dtype_name and input_dtype_name != "a16":
            root = root / input_dtype_name
        return root / f"layer_{int(layer_index):03d}{suffix}"

    def _source_model_stamp(self) -> dict[str, Any]:
        model_path = str(self.plan.get("model_path", "") or "")
        stamp: dict[str, Any] = {"model_path": model_path}
        if not model_path:
            return stamp
        root = Path(model_path)
        for name in ("config.json", "model.safetensors.index.json"):
            file_path = root / name
            try:
                stat = file_path.stat()
            except OSError:
                continue
            stamp[name] = {
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        return stamp

    def _marlin_ready_cache_metadata(
            self,
            expert_ids: list[int],
            runtime_tensors: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        tensor_meta: dict[str, Any] = {}
        if runtime_tensors is not None:
            tensor_meta = {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
                for name, tensor in runtime_tensors.items()
            }
        marlin_input_dtype = str(getattr(self, "_marlin_input_dtype_name", "a16"))
        fp8_metadata: dict[str, Any] = {}
        if marlin_input_dtype == "fp8":
            # FP8-activation Marlin stores the preprocessed int4 weights in an
            # fp8-friendly encoding. Weight scales must carry the compensating
            # factor, otherwise the two MoE GEMMs collapse toward zero.
            fp8_metadata["fp8_weight_scale_factor"] = (
                MARLIN_READY_FP8_WEIGHT_SCALE_FACTOR
            )
            fp8_metadata["fp8_preprocess_schema_version"] = (
                MARLIN_READY_FP8_PREPROCESS_SCHEMA_VERSION
            )
        return {
            "format": "cfie_marlin_ready_expert_cache",
            "version": MARLIN_READY_EXPERT_CACHE_VERSION,
            "source": self._source_model_stamp(),
            "layer_key": str(self.layer_key),
            "layer_index": getattr(self, "_moe_layer_index", None),
            "mode": getattr(self, "_mode", None),
            "quantization": self.plan.get("quantization", None),
            "expert_ids": [int(expert_id) for expert_id in expert_ids],
            "global_num_experts": int(self.layer.global_num_experts),
            "hidden_size": int(self.layer.hidden_size),
            "intermediate_size_per_partition": int(
                self.layer.intermediate_size_per_partition
            ),
            "num_bits": int(getattr(self, "num_bits", 0) or 0),
            "group_size": int(getattr(self, "group_size", 0) or 0),
            "pack_factor": int(getattr(self, "pack_factor", 0) or 0),
            "desc_act": bool(getattr(self, "_gptq_desc_act", False)),
            "is_a_8bit": bool(getattr(self, "is_a_8bit", False)),
            "marlin_input_dtype": marlin_input_dtype,
            "tensor_meta": tensor_meta,
            **fp8_metadata,
        }

    def _marlin_ready_cache_metadata_matches(
            self,
            metadata: Any,
            expert_ids: list[int],
            tensors: dict[str, torch.Tensor] | None = None,
    ) -> bool:
        if not isinstance(metadata, dict):
            return False
        expected = self._marlin_ready_cache_metadata(expert_ids)
        keys = (
            "format",
            "version",
            "layer_key",
            "layer_index",
            "mode",
            "expert_ids",
            "global_num_experts",
            "hidden_size",
            "intermediate_size_per_partition",
            "num_bits",
            "group_size",
            "pack_factor",
            "desc_act",
            "is_a_8bit",
            "marlin_input_dtype",
        )
        for key in keys:
            if metadata.get(key) != expected.get(key):
                return False
        if expected.get("marlin_input_dtype") == "fp8":
            if (
                metadata.get("fp8_weight_scale_factor")
                != expected.get("fp8_weight_scale_factor")
            ):
                return False
            if (
                metadata.get("fp8_preprocess_schema_version")
                != expected.get("fp8_preprocess_schema_version")
            ):
                return False
        expected_source = expected.get("source", {})
        cached_source = metadata.get("source", {})
        if cached_source.get("model_path") != expected_source.get("model_path"):
            return False
        if tensors is not None:
            tensor_meta = metadata.get("tensor_meta", {})
            if not isinstance(tensor_meta, dict):
                return False
            for name, tensor in tensors.items():
                meta = tensor_meta.get(name)
                if not isinstance(meta, dict):
                    return False
                if list(tensor.shape) != list(meta.get("shape", [])):
                    return False
                if str(tensor.dtype) != str(meta.get("dtype", "")):
                    return False
        return True

    @staticmethod
    def _torch_dtype_from_name(name: Any) -> torch.dtype | None:
        if not isinstance(name, str):
            return None
        dtype_name = name.removeprefix("torch.")
        dtype = getattr(torch, dtype_name, None)
        if isinstance(dtype, torch.dtype):
            return dtype
        return None

    def _build_marlin_ready_expert_major_cache_payload(
            self,
            runtime_tensors: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, Any], list[str], int] | None:
        bundles = self._build_quantized_runtime_bundles(
            runtime_tensors,
            pin_memory=False,
        )
        if not bundles:
            return None
        first_bundle = bundles[0]
        storage = first_bundle.storage
        per_expert_bytes = int(first_bundle.nbytes)
        if (
            storage is None
            or storage.device.type != "cpu"
            or storage.dtype != torch.uint8
            or per_expert_bytes <= 0
        ):
            return None
        if int(storage.numel()) < per_expert_bytes * len(bundles):
            return None

        field_order = list(first_bundle.tensors.keys())
        field_meta: dict[str, Any] = {}
        expected_offset = 0
        base_ptr = int(storage.data_ptr()) + int(first_bundle.storage_offset_bytes)
        for name in field_order:
            tensor = first_bundle.tensors[name]
            offset = int(tensor.data_ptr()) - base_ptr
            num_bytes = int(tensor.numel() * tensor.element_size())
            if offset != expected_offset or num_bytes <= 0:
                return None
            field_meta[name] = {
                "offset": int(offset),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "nbytes": int(num_bytes),
            }
            expected_offset += num_bytes
        if expected_offset != per_expert_bytes:
            return None

        for expert_index, bundle in enumerate(bundles):
            if bundle.storage is not storage:
                return None
            if int(bundle.storage_offset_bytes) != expert_index * per_expert_bytes:
                return None
            if int(bundle.nbytes) != per_expert_bytes:
                return None

        expert_storage = storage[
            : per_expert_bytes * len(bundles)
        ].view(len(bundles), per_expert_bytes).contiguous()
        return expert_storage, field_meta, field_order, per_expert_bytes

    def _build_marlin_ready_bundles_from_expert_major_storage(
            self,
            storage: torch.Tensor,
            metadata: dict[str, Any],
            expert_ids: list[int],
            *,
            copy_storage: bool = False,
            pin_storage: bool = False,
    ) -> list[ExpertBundle] | None:
        if metadata.get("layout") != "expert_major":
            return None
        if not self._marlin_ready_cache_metadata_matches(metadata, expert_ids):
            return None
        field_meta = metadata.get("field_meta")
        field_order = metadata.get("field_order")
        if not isinstance(field_meta, dict) or not isinstance(field_order, list):
            return None
        try:
            per_expert_bytes = int(metadata.get("per_expert_bytes", 0))
        except (TypeError, ValueError):
            return None
        if per_expert_bytes <= 0:
            return None

        flat_storage = storage.detach()
        if flat_storage.device.type != "cpu":
            flat_storage = flat_storage.to(device="cpu")
        if flat_storage.dtype != torch.uint8:
            return None
        flat_storage = flat_storage.contiguous().view(torch.uint8).view(-1)
        expected_nbytes = per_expert_bytes * len(expert_ids)
        if int(flat_storage.numel()) < expected_nbytes:
            return None
        flat_storage = flat_storage[:expected_nbytes]
        if copy_storage:
            # safetensors can return a file-backed CPU tensor. Runtime pageable
            # static mirrors must live in DRAM, otherwise decode pays random
            # mmap/page-fault cost during prepare.
            flat_storage = flat_storage.clone()
        elif pin_storage and not flat_storage.is_pinned():
            try:
                gc.collect()
                _empty_torch_host_allocator_cache_best_effort()
                flat_storage = flat_storage.pin_memory()
            except Exception as exc:
                logger.warning_once(
                    "Failed to pin Marlin-ready expert-major cache storage "
                    "directly; falling back to bundle-level pinned packing: "
                    "layer=%s error=%s",
                    self.layer_key,
                    exc,
                )

        specs: list[tuple[str, int, tuple[int, ...], torch.dtype, int]] = []
        expected_offset = 0
        for raw_name in field_order:
            name = str(raw_name)
            meta = field_meta.get(name)
            if not isinstance(meta, dict):
                return None
            dtype = self._torch_dtype_from_name(meta.get("dtype"))
            if dtype is None:
                return None
            try:
                offset = int(meta.get("offset"))
                shape = tuple(int(dim) for dim in meta.get("shape", ()))
                nbytes = int(meta.get("nbytes"))
            except (TypeError, ValueError):
                return None
            expected_nbytes_for_field = (
                int(torch.Size(shape).numel())
                * torch.empty((), dtype=dtype).element_size()
            )
            if (
                offset != expected_offset
                or nbytes != expected_nbytes_for_field
                or nbytes <= 0
            ):
                return None
            specs.append((name, offset, shape, dtype, nbytes))
            expected_offset += nbytes
        if expected_offset != per_expert_bytes:
            return None

        bundles: list[ExpertBundle] = []
        pinned = bool(flat_storage.is_pinned())
        for expert_index in range(len(expert_ids)):
            expert_offset = expert_index * per_expert_bytes
            expert_slice = flat_storage[
                expert_offset: expert_offset + per_expert_bytes
            ]
            views: dict[str, torch.Tensor] = {}
            for name, offset, shape, dtype, nbytes in specs:
                view = expert_slice[offset: offset + nbytes].view(dtype).view(shape)
                views[name] = view
            bundles.append(
                ExpertBundle(
                    tensors=views,
                    nbytes=per_expert_bytes,
                    pinned=pinned,
                    runtime_ready=True,
                    storage=flat_storage,
                    storage_offset_bytes=expert_offset,
                )
            )
        return bundles

    def _load_marlin_ready_layer_cache(
            self,
            expert_ids: list[int],
            *,
            copy_storage: bool = False,
            pin_storage: bool = False,
    ) -> _MarlinReadyLayerCache | None:
        path = self._marlin_ready_layer_cache_path()
        if path is not None and path.is_file():
            cache = self._load_marlin_ready_layer_safetensors_cache(
                path,
                expert_ids,
                copy_storage=copy_storage,
                pin_storage=pin_storage,
            )
            if cache is not None:
                return cache
        return None

    def _load_marlin_ready_layer_safetensors_cache(
            self,
            path: Path,
            expert_ids: list[int],
            *,
            copy_storage: bool = False,
            pin_storage: bool = False,
    ) -> _MarlinReadyLayerCache | None:
        try:
            with safetensors_safe_open(
                str(path),
                framework="pt",
                device="cpu",
            ) as handle:
                if "expert_storage" not in set(handle.keys()):
                    return None
                safetensors_metadata = handle.metadata() or {}
                encoded_metadata = safetensors_metadata.get("cfie_metadata")
                if not encoded_metadata:
                    return None
                metadata = json.loads(encoded_metadata)
                if not isinstance(metadata, dict):
                    return None
                storage = handle.get_tensor("expert_storage")
            bundles = self._build_marlin_ready_bundles_from_expert_major_storage(
                storage,
                metadata,
                expert_ids,
                copy_storage=copy_storage,
                pin_storage=pin_storage,
            )
            if bundles is None:
                logger.warning_once(
                    "Ignoring stale Marlin-ready expert-major cache: "
                    "layer=%s path=%s",
                    self.layer_key,
                    path,
                )
                return None
            logger.info(
                "Loaded Marlin-ready expert-major cache: "
                "layer=%s experts=%d path=%s",
                self.layer_key,
                len(expert_ids),
                path,
            )
            return _MarlinReadyLayerCache(bundles=bundles)
        except Exception as exc:
            logger.warning_once(
                "Failed to load Marlin-ready expert-major cache; rebuilding: "
                "layer=%s path=%s error=%s",
                self.layer_key,
                path,
                exc,
            )
            return None

    def _save_marlin_ready_layer_cache(
            self,
            expert_ids: list[int],
            runtime_tensors: dict[str, torch.Tensor],
    ) -> None:
        path = self._marlin_ready_layer_cache_path()
        if path is None or not runtime_tensors:
            return
        if list(expert_ids) != list(range(int(self.layer.global_num_experts))):
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tensors = {
                name: tensor.detach().to(device="cpu").contiguous()
                for name, tensor in runtime_tensors.items()
            }
            payload = self._build_marlin_ready_expert_major_cache_payload(tensors)
            if payload is None:
                return
            expert_storage, field_meta, field_order, per_expert_bytes = payload
            metadata = self._marlin_ready_cache_metadata(expert_ids, tensors)
            metadata.update(
                {
                    "layout": "expert_major",
                    "per_expert_bytes": int(per_expert_bytes),
                    "field_order": field_order,
                    "field_meta": field_meta,
                }
            )
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            safetensors_save_file(
                {"expert_storage": expert_storage},
                str(tmp_path),
                metadata={
                    "format": "cfie_marlin_ready_expert_cache",
                    "cfie_metadata": json.dumps(
                        metadata,
                        separators=(",", ":"),
                    ),
                },
            )
            os.replace(tmp_path, path)
            logger.info(
                "Saved Marlin-ready expert-major cache: "
                "layer=%s experts=%d path=%s",
                self.layer_key,
                len(expert_ids),
                path,
            )
        except Exception as exc:
            logger.warning_once(
                "Failed to save Marlin-ready expert cache: "
                "layer=%s path=%s error=%s",
                self.layer_key,
                path,
                exc,
            )

    def _init_cpu_fixed_pools(self) -> None:
        # ------------------------------- 闁哄秷顫夊畵浣烘媼閳ュ啿鐏婇悷娆欑稻閻?CPU 閻㈩垱鎮傞埞?expert 闂傚棗妫楅幃?-------------------------------
        # 濞寸姴姘﹂鎼佸礆閹哄秷鍘柣?initial_cpu_experts 閻犲洩顕цぐ鍥閳ь剛鎲版担鍛婅含 CPU 濞撴皜鍐煑濡炵宕靛▓?expert 缂傚倹鐗曡ぐ鍧楁晬鐏炲€熷珯閺夆晛娲﹂幎銈夊箳婢跺海楔闁伙絽鐬煎▓?expert闁?
        cpu_static_experts = tuple(
            int(expert_id)
            for expert_id in self.plan.get("initial_cpu_experts", ())
            if int(expert_id) < self.layer.global_num_experts
        )

        # 閻?CPU 閻㈩垱鎮傞埞?expert 闂傚棗妫楅幃搴ㄥ礃閼姐倗娉㈠☉?frozenset闁挎稑濂旂粚鑸电鎼粹剝鍊电紓渚囧幖閹烩晠鏌呴悢宄扮伈闁哄偆鍘虹粭宀勬焼閸喖甯抽悹鍥跺灟閹便劑寮ㄩ獮搴撳亾?
        self._cpu_static_experts = frozenset(cpu_static_experts)

        # ------------------------------- 闁告帗绻傞～鎰板礌?CPU 濞?staging bundle 闁绘鍩栭埀?-------------------------------
        # 鐟滅増鎸告晶鐘崇▔閼姐倕娈犻悹渚灠缁剁偞绋夊鍛櫃闁告娲滅€氼厾绱掔€涙ê袘 staging bundle闁挎稑鑻ú婊冾潰閵堝牏绠归梺鎻掓湰濡顕ｈ箛鏇犳瀭濞?None闁?
        self._cpu_stage_bundle = None
        self._cpu_runtime_batch_buffers = {}

        # ------------------------------- 闁圭顦紞瀣礈瀹ュ棙缍€闂佹彃绉佃啯鐎殿喖绻愰崹鍨叏鐎ｎ亜顕?CPU 濞撴皜鍐ㄦ枾濠殿喖顑嗗鍫ユ煂瀹ュ洨澶勯柛鎰絻鐏?-------------------------------
        # 鐟滅増鎸哥紞瀣礈瀹ュ棗浠橀柛鎺曟硾濞呮帒顔忛妷銈囩▕闁?GPTQ Marlin 婵☆垪鈧磭纭€濞戞挸顑嗗鍌炴晬鐏炶棄鐎婚梺鏉跨Ч閸ｆ椽宕犻弽褍鏂у┑顔碱儐濞煎牓鏌屽鍥╁闁告劖褰冪亸顖炲Υ?
        if self._mode == "gptq_marlin":
            # 闁告帒妫濋崢?CPU 濞撴皜鍥ф闁告牗鐗曠敮顐ｆ叏鐎ｎ偅缍€闂佹彃绉剁槐锕傚礃閹绘帒闅橀柕?
            self._cpu_quantized_raw_buffer = self._allocate_quantized_raw_buffer()

            # 閻忓繐妫濋崳娲礌閺嵮冩枾濠殿喖顑嗗鍫ユ煂瀹ュ洨澶勯柛鎰絻鐏忣垶宕￠悩鍨殢闁汇劌瀚悺褔鎳為崒娑欐缂侀硸鍨甸鎼佸礆?CPU buffer 闁诡剝顕ч妵鍥╀焊韫囧氦鍘柕?
            self._cpu_buffer_bytes += self._raw_quantized_nbytes(
                self._cpu_quantized_raw_buffer
            )
        else:
            # 鐟滅増鎸哥紞瀣礈瀹ュ棗浠橀柛鎺曟硾濞呮帒顔忛妷銈囩▕闁革负鍔戝顏堟煂韫囨挸顕ф俊顖椻偓宕囩濞戞挸顑嗗鍌炴晬鐏炶棄鐎婚梺鏉跨Ч濞碱亪鏌岃箛鎾愁嚙闁告鍠庨～鎰板级閸愵喖娅㈢紓鍌涙尭閸熷潡宕犻幁鎺嗗亾?
            self._cpu_unquantized_raw_buffer = self._allocate_unquantized_raw_buffer()

            # 閻忓繐妫濆顏堟煂韫囨挸顕ч柛妯煎枎椤劙寮堕崘顔兼缂傚倹鎸搁崯鍧楀礌閸濆嫬绐楅柣顫妿濞堟垹鈧稒顨夋俊顓㈠极閹殿喚鏌堥悹浣测偓鍐茬厒 CPU buffer 闁诡剝顕ч妵鍥╀焊韫囧氦鍘柕?
            self._cpu_buffer_bytes += self._raw_unquantized_nbytes(
                self._cpu_unquantized_raw_buffer
            )



        # ------------------------------- 闁圭顦抽鎼佸礆閹烘垹娈?CPU 閻㈩垱鎮傞埞?expert 濡澘瀚崕褰掑礆閺夊灝鏁堕悗娑櫭奸懙?-------------------------------
        self._materialize_cpu_static_bundles_eager(cpu_static_experts)
        if self._cpu_static_bundles and all(
                bundle.runtime_ready
                for bundle in self._cpu_static_bundles.values()
        ):
            self._pack_cpu_static_bundles_layer_wide_best_effort()

        # ------------------------------- 闁革负鍔嶇€垫棃寮?CPU 缂傚倹鎸搁悺銊╁箣閺嶎偆澶勯柛鎰絻鐏忣垶寮幆鎵炕闁告垵鎼崹鍨叏鐎ｎ亜顕ч柡鍐﹀劚缁?-------------------------------
        # 鐟滅増鎸哥紞瀣礈瀹ュ棗浠橀柛鎺曟硾濞呮帞娑甸鑲╂澖闁归晲鐒﹀﹢?CPU 闂傚牊鐟﹂埀?bundle 闁?staging bundle 闁哄啳顔愮槐婵囨綇閹惧啿姣夐柛鎺撶箓椤劙宕犻弽銊︼級闊洦銇滈埀?
        if self._cpu_static_bundles or self._cpu_stage_bundle is not None:
            pinned_static_bytes = sum(
                int(bundle.nbytes)
                for bundle in self._cpu_static_bundles.values()
                if bundle.pinned
            )
            pageable_static_bytes = sum(
                int(bundle.nbytes)
                for bundle in self._cpu_static_bundles.values()
                if not bundle.pinned
            )
            logger.info(
                "Initialized fixed CPU expert pool: layer=%s static=%d/%d staging=%s "
                "cpu_bytes=%.2f MiB pinned_static=%.2f MiB "
                "pageable_static=%.2f MiB burst_min_tokens=%d",
                self.layer_key,
                len(self._cpu_static_bundles),
                len(self._cpu_static_experts),
                self._cpu_stage_bundle is not None,
                int(getattr(self, "_cpu_buffer_bytes", 0)) / (1 << 20),
                pinned_static_bytes / (1 << 20),
                pageable_static_bytes / (1 << 20),
                self.prefill_burst_min_tokens,
            )

    def _allocate_source_bundle(self) -> ExpertBundle:
        if self._mode == "gptq_marlin":
            specs: list[PackedExpertTensorSpec] = [
                PackedExpertTensorSpec(
                    name="slot.gate_proj.qweight",
                    shape=(
                        self.layer.hidden_size // self.pack_factor,
                        self.layer.intermediate_size_per_partition,
                    ),
                    dtype=torch.int32,
                ),
                PackedExpertTensorSpec(
                    name="slot.gate_proj.scales",
                    shape=(
                        self.layer.num_groups_w13,
                        self.layer.intermediate_size_per_partition,
                    ),
                    dtype=self.layer.w13_scales.dtype,
                ),
                PackedExpertTensorSpec(
                    name="slot.gate_proj.qzeros",
                    shape=(
                        self.layer.num_groups_w13,
                        self.layer.intermediate_size_per_partition // self.pack_factor,
                    ),
                    dtype=self.layer.w13_qzeros.dtype,
                ),
                PackedExpertTensorSpec(
                    name="slot.up_proj.qweight",
                    shape=(
                        self.layer.hidden_size // self.pack_factor,
                        self.layer.intermediate_size_per_partition,
                    ),
                    dtype=torch.int32,
                ),
                PackedExpertTensorSpec(
                    name="slot.up_proj.scales",
                    shape=(
                        self.layer.num_groups_w13,
                        self.layer.intermediate_size_per_partition,
                    ),
                    dtype=self.layer.w13_scales.dtype,
                ),
                PackedExpertTensorSpec(
                    name="slot.up_proj.qzeros",
                    shape=(
                        self.layer.num_groups_w13,
                        self.layer.intermediate_size_per_partition // self.pack_factor,
                    ),
                    dtype=self.layer.w13_qzeros.dtype,
                ),
                PackedExpertTensorSpec(
                    name="slot.down_proj.qweight",
                    shape=(
                        self.layer.intermediate_size_per_partition // self.pack_factor,
                        self.layer.hidden_size,
                    ),
                    dtype=torch.int32,
                ),
                PackedExpertTensorSpec(
                    name="slot.down_proj.scales",
                    shape=(self.layer.num_groups_w2, self.layer.hidden_size),
                    dtype=self.layer.w2_scales.dtype,
                ),
                PackedExpertTensorSpec(
                    name="slot.down_proj.qzeros",
                    shape=(
                        self.layer.num_groups_w2,
                        self.layer.hidden_size // self.pack_factor,
                    ),
                    dtype=self.layer.w2_qzeros.dtype,
                ),
            ]
            if self._gptq_desc_act:
                specs.extend(
                    (
                        PackedExpertTensorSpec(
                            name="slot.gate_proj.g_idx",
                            shape=(self.layer.hidden_size,),
                            dtype=self.layer.w13_g_idx.dtype,
                        ),
                        PackedExpertTensorSpec(
                            name="slot.up_proj.g_idx",
                            shape=(self.layer.hidden_size,),
                            dtype=self.layer.w13_g_idx.dtype,
                        ),
                        PackedExpertTensorSpec(
                            name="slot.down_proj.g_idx",
                            shape=(self.layer.intermediate_size_per_partition,),
                            dtype=self.layer.w2_g_idx.dtype,
                        ),
                    )
                )
        else:
            specs = [
                PackedExpertTensorSpec(
                    name="slot.gate_proj.weight",
                    shape=(
                        self.layer.intermediate_size_per_partition,
                        self.layer.hidden_size,
                    ),
                    dtype=self.layer.w13_weight.dtype,
                ),
                PackedExpertTensorSpec(
                    name="slot.up_proj.weight",
                    shape=(
                        self.layer.intermediate_size_per_partition,
                        self.layer.hidden_size,
                    ),
                    dtype=self.layer.w13_weight.dtype,
                ),
                PackedExpertTensorSpec(
                    name="slot.down_proj.weight",
                    shape=(
                        self.layer.hidden_size,
                        self.layer.intermediate_size_per_partition,
                    ),
                    dtype=self.layer.w2_weight.dtype,
                ),
            ]

        storage, tensors = allocate_packed_cpu_tensor_views(specs)
        return ExpertBundle(
            tensors=tensors,
            nbytes=bundle_nbytes(tensors),
            pinned=False,
            storage=storage,
        )
        # ------------------------------- 濞?checkpoint 鐟滆埇鍨洪埀?expert 闁告帒妫濋崢銈嗙▔鐎涙ɑ顦?source bundle 閻庡湱鎳撳▍?-------------------------------
        # 閺夆晜鐟ら柌?bundle 闁告瑯浜為弫銈嗙鎼淬垹顥為柟?safetensors 濞戞搩鍘惧▓?gate/up/down 闁告鍠庨～鎰偓娑欘殕椤斿矂鏁嶅畝鍕吂闁告艾绨肩槐鎵偖椤愶紕顏辨繛鍡忓墲閳ь儸鍥舵殨濠㈣泛瀚幃濠囧箣?
        # runtime-ready CPU static bundle闁靛棗鍊块弳閬嶅嫉閻斿皷鏁嗛柣锝嗙懅濞?CPU static mirror 濞戞挸绉撮崯鈧ǎ鍥ㄧ箓閻?checkpoint 鐟滆埇鍨洪埀顑跨筏缁?
        # 闁兼澘鏈Σ鍛婄┍濠靛棛鎽犻柛娆樺灣濞插潡骞掗妷銉ユ櫢闁稿繈鍎茬敮褰掓偠?resident slot 闁?runtime 闁哄秶鍘х槐锟犲Υ?
        #
        # 闁搞儳濮甸婵囨交濞嗘挸娅″ǎ鍥ㄧ箖鐎垫棃寮查鈧埀?CPU 闁告劕鎳庨悺銊╁础閸愭彃璁查柨娑欑〒濠€鈥愁潰閿濆懐鐟栭柡鍫熺☉婵偤鏌?H2D 闁汇劌瀚Σ鍛婏紣閸曨偒妲遍柣鐐叉閹鎯?runtime-ready bundle闁?
        cpu_tensor_args = {"device": "cpu"}

        # 濡澘瀚敍鎰板及?source bundle 闁告劕鎳橀崕瀛樼┍濠靛棛鎽犻柣銊ュ缁卞爼鏌岃箛鎾舵憻闁稿繑鎮堕埀?
        tensors: dict[str, torch.Tensor]

        # ------------------------------- 闁革负鍔戦崳娲礌閺嶎剛鐔呯€垫澘瀚粭鍛村礆閸℃稑甯?gate/up/down 闁汇劌瀚伴崳娲礌閺嶃劍缍€闂佹彃绉寸槐鍫曟煂?-------------------------------
        # 鐟滅増鎸哥紞瀣礈瀹ュ棗浠橀柛鎺曟硾濞呮帒顔忛妷銈囩▕闁?GPTQ Marlin 婵☆垪鈧磭纭€濞戞挸顑嗗鍌炴晬鐏炶壈绀?gate闁靛棔鑿噋闁靛棔榫歰wn 濞戞挸顦抽惌楣冨礆閸℃鐒奸柛鎺戞閸?qweight闁靛棔澶焎ales 闁?qzeros 鐎殿喚濞€閸ｆ椽濡?
        if self._mode == "gptq_marlin":
            tensors = {
                # 濞?gate 闁硅埖娲栨總鏍儍閸曨垰娅ら柛鏍ㄧ墬濞煎牓鏌屽鍛€婚梺?CPU 鐎殿喚濞€閸ｆ椽濡?
                "slot.gate_proj.qweight": torch.empty(
                    (
                        self.layer.hidden_size // self.pack_factor,
                        self.layer.intermediate_size_per_partition,
                    ),
                    dtype=torch.int32,
                    **cpu_tensor_args,
                ),

                # 濞?gate 闁硅埖娲栨總鏍儍?scale 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋 CPU 鐎殿喚濞€閸ｆ椽濡?
                "slot.gate_proj.scales": torch.empty(
                    (self.layer.num_groups_w13, self.layer.intermediate_size_per_partition),
                    dtype=self.layer.w13_scales.dtype,
                    **cpu_tensor_args,
                ),

                # 濞?gate 闁硅埖娲栨總鏍儍?qzeros 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋 CPU 鐎殿喚濞€閸ｆ椽濡?
                "slot.gate_proj.qzeros": torch.empty(
                    (
                        self.layer.num_groups_w13,
                        self.layer.intermediate_size_per_partition // self.pack_factor,
                    ),
                    dtype=self.layer.w13_qzeros.dtype,
                    **cpu_tensor_args,
                ),

                # 濞?up 闁硅埖娲栨總鏍儍閸曨垰娅ら柛鏍ㄧ墬濞煎牓鏌屽鍛€婚梺?CPU 鐎殿喚濞€閸ｆ椽濡?
                "slot.up_proj.qweight": torch.empty(
                    (
                        self.layer.hidden_size // self.pack_factor,
                        self.layer.intermediate_size_per_partition,
                    ),
                    dtype=torch.int32,
                    **cpu_tensor_args,
                ),

                # 濞?up 闁硅埖娲栨總鏍儍?scale 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋 CPU 鐎殿喚濞€閸ｆ椽濡?
                "slot.up_proj.scales": torch.empty(
                    (
                        self.layer.num_groups_w13,
                        self.layer.intermediate_size_per_partition
                    ),
                    dtype=self.layer.w13_scales.dtype,
                    **cpu_tensor_args,
                ),

                # 濞?up 闁硅埖娲栨總鏍儍?qzeros 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋 CPU 鐎殿喚濞€閸ｆ椽濡?
                "slot.up_proj.qzeros": torch.empty(
                    (
                        self.layer.num_groups_w13,
                        self.layer.intermediate_size_per_partition // self.pack_factor,
                    ),
                    dtype=self.layer.w13_qzeros.dtype,
                    **cpu_tensor_args,
                ),

                # 濞?down 闁硅埖娲栨總鏍儍閸曨垰娅ら柛鏍ㄧ墬濞煎牓鏌屽鍛€婚梺?CPU 鐎殿喚濞€閸ｆ椽濡?
                "slot.down_proj.qweight": torch.empty(
                    (
                        self.layer.intermediate_size_per_partition // self.pack_factor,
                        self.layer.hidden_size,
                    ),
                    dtype=torch.int32,
                    **cpu_tensor_args,
                ),

                # 濞?down 闁硅埖娲栨總鏍儍?scale 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋 CPU 鐎殿喚濞€閸ｆ椽濡?
                "slot.down_proj.scales": torch.empty(
                    (self.layer.num_groups_w2, self.layer.hidden_size),
                    dtype=self.layer.w2_scales.dtype,
                    **cpu_tensor_args,
                ),

                # 濞?down 闁硅埖娲栨總鏍儍?qzeros 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋 CPU 鐎殿喚濞€閸ｆ椽濡?
                "slot.down_proj.qzeros": torch.empty(
                    (self.layer.num_groups_w2, self.layer.hidden_size // self.pack_factor),
                    dtype=self.layer.w2_qzeros.dtype,
                    **cpu_tensor_args,
                ),
            }
            if self._gptq_desc_act:
                tensors.update(
                    {
                        "slot.gate_proj.g_idx": torch.empty(
                            (self.layer.hidden_size,),
                            dtype=self.layer.w13_g_idx.dtype,
                            **cpu_tensor_args,
                        ),
                        "slot.up_proj.g_idx": torch.empty(
                            (self.layer.hidden_size,),
                            dtype=self.layer.w13_g_idx.dtype,
                            **cpu_tensor_args,
                        ),
                        "slot.down_proj.g_idx": torch.empty(
                            (self.layer.intermediate_size_per_partition,),
                            dtype=self.layer.w2_g_idx.dtype,
                            **cpu_tensor_args,
                        ),
                    }
                )
        else:
            # ------------------------------- 闁革负鍔戝顏堟煂韫囨挸顕ч悹渚灠缁剁偞绋夌€ｎ亜鐎婚梺?gate/up/down 闁汇劌瀚敮顐ｆ叏鐎ｎ偅缍€闂佹彃绉寸槐鍫曟煂?-------------------------------
            # 鐟滅増鎸哥紞瀣礈瀹ュ棗浠橀柛鎺曟硾濞呮帒顔忛妷銈囩▕闁革负鍔戝顏堟煂韫囨挸顕ф俊顖椻偓宕囩濞戞挸顑嗗鍌炴晬鐏炶棄娑ч梻鍥ｅ亾閻熸洑妞掔拹?gate闁靛棔鑿噋闁靛棔榫歰wn 濞戞挸顦抽惌楣冨礆閸℃稑甯抽柛妯煎枎椤?weight 鐎殿喚濞€閸ｆ椽濡?
            tensors = {
                # 濞?gate 闁硅埖娲栨總鏍儍閸曨偄鏂у┑顔碱儐濞煎牓鏌屽鍛€婚梺?CPU 鐎殿喚濞€閸ｆ椽濡?
                "slot.gate_proj.weight": torch.empty(
                    (
                        self.layer.intermediate_size_per_partition,
                        self.layer.hidden_size
                    ),
                    dtype=self.layer.w13_weight.dtype,
                    **cpu_tensor_args,
                ),

                # 濞?up 闁硅埖娲栨總鏍儍閸曨偄鏂у┑顔碱儐濞煎牓鏌屽鍛€婚梺?CPU 鐎殿喚濞€閸ｆ椽濡?
                "slot.up_proj.weight": torch.empty(
                    (
                        self.layer.intermediate_size_per_partition,
                        self.layer.hidden_size
                    ),
                    dtype=self.layer.w13_weight.dtype,
                    **cpu_tensor_args,
                ),

                # 濞?down 闁硅埖娲栨總鏍儍閸曨偄鏂у┑顔碱儐濞煎牓鏌屽鍛€婚梺?CPU 鐎殿喚濞€閸ｆ椽濡?
                "slot.down_proj.weight": torch.empty(
                    (
                        self.layer.hidden_size,
                        self.layer.intermediate_size_per_partition
                    ),
                    dtype=self.layer.w2_weight.dtype,
                    **cpu_tensor_args,
                ),
            }

        # ------------------------------- 閻忓繐妫楃槐鍫曟煂韫囨挾鎽熼柛蹇曨焾閻ㄦ繄鎲楅崨顒冪缂備胶鍠嶇粩鎾儍?ExpertBundle 閺夆晜鏌ㄥú?-------------------------------
        # 闁糕晞妗ㄧ花顒冦亹閹惧啿顤呯€殿喚濞€閸ｈ櫣鈧稒顨呴崥鈧柡瀣閳ь剛濮风划鐑樼▔閳ь剟鎯?ExpertBundle闁挎稑濂旂欢?CPU static 濞戞挸楠告慨鈺呭箑娴ｇ顫ｉ弶鐐测偓鐔虹唴鐎垫澘瀚ˇ鏌ユ偨閵婏絺鍋?
        return ExpertBundle(
            tensors=tensors,
            nbytes=bundle_nbytes(tensors),
            pinned=False,
        )

    def _allocate_quantized_raw_buffer(
            self,
            batch_size: int = 1,
            *,
            pin_memory: bool | None = None,
    ) -> _RawExpertWeights:
        # ------------------------------- 濞戞挻妞介崳娲礌閺嶎剛鐔呯€垫澘瀚崹搴ㄦ煀瀹ュ懎绀?expert 缂?CPU 闁告鍠庨～鎰板级閸愵喖娅㈢紓鍌涙尭閸熷潡宕?-------------------------------
        # 闁哄瀚伴埀?CPU 濞撴皜鍐倞闂佹彃绻愰崹搴ㄦ煀瀹ュ懎妫橀柡浣稿簻缁辫精銇愰幘鎶芥尙闁告瑧澧楅弫顕€骞愭担瑙勵槯闁告凹鍨抽弫?pinned memory 濞寸姰鍎辨慨鐐烘焻閻旈攱鍊电紓?H2D 濞磋偐濮剧欢顓㈠Υ?
        cpu_tensor_args = {
            "device": "cpu",
            "pin_memory": (
                bool(
            getattr(
                self,
                "_use_pinned_cpu_static",
                False,
            )
                )
                if pin_memory is None
                else bool(pin_memory)
            ),
        }

        # 閺夆晜鏌ㄥú鏍閵忕姵鍊婚梺鎻掔箰鐎佃尙鎹勯姘辩獮闁汇劌瀚畷?expert 闁告鍠庨～鎰板级閸愵喖娅㈢紓鍌涙尭閸熷潡宕犻崫鍕靛殸閻犵伜鎵冲亾?
        return _RawExpertWeights(
            # 濞?gate/up 闁告艾鐗嗛懟鐔煎触鎼达絾鐣?w13 闂佹彃绻愮€垫煡寮堕崘顔兼闁告帒妫濋崢?CPU 缂傚倹鎸搁崯鍧楀礌閹巻鍋?
            w13_qweight=torch.empty(
                (
                    batch_size,
                    self.layer.hidden_size // self.pack_factor,
                    2 * self.layer.intermediate_size_per_partition,
                ),
                dtype=torch.int32,
                **cpu_tensor_args,
            ),
            # 濞?down 閻犱警鍨扮欢鐐烘儍?w2 闂佹彃绻愮€垫煡寮堕崘顔兼闁告帒妫濋崢?CPU 缂傚倹鎸搁崯鍧楀礌閹巻鍋?
            w2_qweight=torch.empty(
                (
                    batch_size,
                    self.layer.intermediate_size_per_partition // self.pack_factor,
                    self.layer.hidden_size,
                ),
                dtype=torch.int32,
                **cpu_tensor_args,
            ),
            # 濞?gate/up 闁告艾鐗嗛懟鐔煎触鎼达絾鐣?w13 scale 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋 CPU 缂傚倹鎸搁崯鍧楀礌閹巻鍋?
            w13_scales=torch.empty(
                (
                    batch_size,
                    self.layer.num_groups_w13,
                    2 * self.layer.intermediate_size_per_partition
                ),
                dtype=self.layer.w13_scales.dtype,
                **cpu_tensor_args,
            ),
            # 濞?down 閻犱警鍨扮欢鐐烘儍?w2 scale 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋 CPU 缂傚倹鎸搁崯鍧楀礌閹巻鍋?
            w2_scales=torch.empty(
                (
                    batch_size,
                    self.layer.num_groups_w2,
                    self.layer.hidden_size
                ),
                dtype=self.layer.w2_scales.dtype,
                **cpu_tensor_args,
            ),
            # 濞?gate/up 闁告艾鐗嗛懟鐔煎触鎼达絾鐣?w13 qzeros 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋 CPU 缂傚倹鎸搁崯鍧楀礌閹巻鍋?
            w13_qzeros=torch.empty(
                (
                    batch_size,
                    self.layer.num_groups_w13,
                    (2 * self.layer.intermediate_size_per_partition) // self.pack_factor,
                ),
                dtype=self.layer.w13_qzeros.dtype,
                **cpu_tensor_args,
            ),
            # 濞?down 閻犱警鍨扮欢鐐烘儍?w2 qzeros 鐎殿喚濞€閸ｆ椽宕氶崱娑樺赋 CPU 缂傚倹鎸搁崯鍧楀礌閹巻鍋?
            w2_qzeros=torch.empty(
                (
                    batch_size,
                    self.layer.num_groups_w2,
                    self.layer.hidden_size // self.pack_factor
                ),
                dtype=self.layer.w2_qzeros.dtype,
                **cpu_tensor_args,
            ),
            w13_g_idx=(
                torch.empty(
                    (batch_size, self.layer.hidden_size),
                    dtype=self.layer.w13_g_idx.dtype,
                    **cpu_tensor_args,
                )
                if self._gptq_desc_act
                else None
            ),
            w2_g_idx=(
                torch.empty(
                    (batch_size, self.layer.intermediate_size_per_partition),
                    dtype=self.layer.w2_g_idx.dtype,
                    **cpu_tensor_args,
                )
                if self._gptq_desc_act
                else None
            ),
        )

    def _allocate_unquantized_raw_buffer(self) -> _RawUnquantizedExpertWeights:
        # ------------------------------- 濞戞挻妞藉顏堟煂韫囨挸顕ч悹渚灠缁剁偤宕氶崱娑樺赋闁?expert 缂?CPU 闁告鍠庨～鎰板级閸愵喖娅㈢紓鍌涙尭閸熷潡宕?-------------------------------
        # 闁哄瀚伴埀?CPU 濞撴皜鍐倞闂佹彃绻愰崹搴ㄦ煀瀹ュ懎妫橀柡浣稿簻缁辫精銇愰幘鎶芥尙闁告瑧澧楅弫顕€骞愭担瑙勵槯闁告凹鍨抽弫?pinned memory 濞寸姰鍎辨慨鐐烘焻閻旈攱鍊电紓?H2D 濞磋偐濮剧欢顓㈠Υ?
        cpu_tensor_args = {
            "device": "cpu",
            "pin_memory": bool(
                getattr(
                    self,
                    "_use_pinned_cpu_static",
                    getattr(self, "_use_pinned_cpu", False),
                )
            ),
        }

        # 闁告帇鍊栭弻鍥亹閹惧啿顤?MoE 閻犱警鍨扮欢鐐哄及椤栨碍鍎婇梺鎻掓川閺?act-and-mul 鐟滆埇鍨圭槐锟犳晬鐏炶偐鐭ら柤鏉胯嫰閸犲懐鈧?w13 闁汇劌瀚粭鍌氥€掑宀€缈婚柛鎴ｆ濞ｎ喗鎯旈敂琛″亾?
        is_act_and_mul = bool(self.layer.moe_config.is_act_and_mul)

        # 闁哄秷顫夊畵渚€寮伴姘剨闁告凹鍨抽弫?act-and-mul闁挎稑鐭侀鍝ョ不?w13 闁告艾鐗嗛懟鐔煎级閸愵喖娅㈤柣銊ュ缁额參宕欓搹瑙勬▕閹艰揪璐熼埀?
        w13_up_dim = (
            2 * self.layer.intermediate_size_per_partition
            if is_act_and_mul
            else self.layer.intermediate_size_per_partition
        )

        # 閺夆晜鏌ㄥú鏍閵忕姵鍊婚梻鍫㈠仱閸ｆ椽宕犻弽顒傜唴鐎垫澘瀚▓鎴﹀础?expert 闁告鍠庨～鎰板级閸愵喖娅㈢紓鍌涙尭閸熷潡宕犻崫鍕靛殸閻犵伜鎵冲亾?
        return _RawUnquantizedExpertWeights(
            # 濞?gate/up 闁告艾鐗嗛懟鐔煎触鎼达絾鐣?w13 闁哄鍟撮崳鎼佸礆閸℃稑甯?CPU 缂傚倹鎸搁崯鍧楀礌閹巻鍋?
            w13_weight=torch.empty(
                (
                    1,
                    w13_up_dim,
                    self.layer.hidden_size
                ),
                dtype=self.layer.w13_weight.dtype,
                **cpu_tensor_args,
            ),
            # 濞?down 閻犱警鍨扮欢鐐烘儍?w2 闁哄鍟撮崳鎼佸礆閸℃稑甯?CPU 缂傚倹鎸搁崯鍧楀礌閹巻鍋?
            w2_weight=torch.empty(
                (
                    1,
                    self.layer.hidden_size,
                    self.layer.intermediate_size_per_partition,
                ),
                dtype=self.layer.w2_weight.dtype,
                **cpu_tensor_args,
            ),
        )

    def _install_mapping(self, expert_id: int, slot: int) -> None:
        # 閻犱焦婢樼紞宥囨嫚?slot 濞戞柨顑呮晶鐘炽仚閼姐倖娈岄柣銊ュ濡叉悂宕鍐殝 expert闁?
        expert_to_slot = self._ensure_cpu_expert_to_slot()
        previous_global = self._slot_to_global[slot]
        with torch.no_grad():
            # 闁兼眹鍎撮?slot 闁告鍠愬﹢鏉款啅閸欏绠?expert闁挎稑鑻崹顖炲礂閸喎惟闁?expert 濞?expert_map 濞戞搩鍘介悥锝囨媼妫颁浇绀嬮柡鍫邯閳规鎮惧▎宥佸亾?
            if previous_global >= 0:
                self.layer._expert_map[previous_global] = -1
                if previous_global < len(expert_to_slot):
                    expert_to_slot[previous_global] = -1
                self._evictions += 1
            # 闁告劕绉垫俊鎼佸棘?expert 闁圭鍊搁崺宀€鎷?slot 濞戞挸顭堥埀?
            self.layer._expert_map[expert_id] = slot
            if expert_id >= len(expert_to_slot):
                expert_to_slot.extend([-1] * (expert_id + 1 - len(expert_to_slot)))
            if expert_id >= 0:
                expert_to_slot[expert_id] = slot
        # 闁哄洤鐡ㄩ弻濠囧矗瀹ュ懏鍊荤紒渚垮灩缁扁晝鎮伴…鎺旂閻炴稏鍔庨妵姘交濞嗗酣鍤?slot 闁绘粍婢樺﹢顏嗘啑閸涱垱绲婚柡?expert闁?
        self._slot_to_global[slot] = expert_id

    def _install_mappings(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
    ) -> None:
        for expert_id, slot, _bundle, source in bundles_and_sources:
            self._install_mapping(expert_id, slot)
            self._total_loads += 1
            if self._total_loads <= 3 or self._total_loads % 200 == 0:
                logger.debug(
                    "Tiered MoE cache event: layer=%s load=%d batch=%d source=%s "
                    "expert=%d slot=%d cpu_hits=%d nvme_loads=%d evictions=%d",
                    self.layer_key,
                    self._total_loads,
                    len(bundles_and_sources),
                    source,
                    expert_id,
                    slot,
                    self._cpu_hits,
                    self._nvme_loads,
                    self._evictions,
                )

    def _install_cpu_mappings_after_native(
            self,
            bundles_and_sources: list[tuple[int, int, ExpertBundle, str]],
    ) -> None:
        expert_to_slot = self._ensure_cpu_expert_to_slot()
        for expert_id, slot, _bundle, source in bundles_and_sources:
            expert_id = int(expert_id)
            slot = int(slot)
            previous_global = int(self._slot_to_global[slot])
            if previous_global >= 0:
                if previous_global < len(expert_to_slot):
                    expert_to_slot[previous_global] = -1
                self._evictions += 1
            if expert_id >= len(expert_to_slot):
                expert_to_slot.extend([-1] * (expert_id + 1 - len(expert_to_slot)))
            if expert_id >= 0:
                expert_to_slot[expert_id] = slot
            self._slot_to_global[slot] = expert_id

            self._total_loads += 1
            if self._total_loads <= 3 or self._total_loads % 200 == 0:
                logger.debug(
                    "Tiered MoE cache event: layer=%s load=%d batch=%d source=%s "
                    "expert=%d slot=%d cpu_hits=%d nvme_loads=%d evictions=%d",
                    self.layer_key,
                    self._total_loads,
                    len(bundles_and_sources),
                    source,
                    expert_id,
                    slot,
                    self._cpu_hits,
                    self._nvme_loads,
                    self._evictions,
                )

    @staticmethod
    def _raw_quantized_nbytes(raw: _RawExpertWeights) -> int:
        # ------------------------------- 缂備胶鍠曢鎼佹煂韫囨挸顕ч柛妯煎枎椤劙寮堕崘顔兼缂傚倹鎸搁崯鍧楀礌閾忚鐣遍柟顒冾嚙閻⊙囨嚍閸屾稒娈?-------------------------------
        # 閻庨潧缍婇崳娲礌閺嵮冩枾濠殿喖顑嗗鍫ユ煂瀹ュ洨澶勯柛鎰絻鐏忣垱绋夐鐘崇暠闁告艾瀚柌婊冾嚕閻樿娅ら柟?闁稿繐鍟扮粈宀勫极妫颁胶顔掑ù鐘劚瀹曠喖宕楅崘顏嗩槺閻庢稒顨夋俊顓㈠极?婵懓鍊搁幏浼存晬鐏炵晫绻侀柛鎺斿閳ь剝顕ч悺褔鎳為崒姘獥闁活潿鍔婇埀?
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                raw.w13_qweight,
                raw.w2_qweight,
                raw.w13_scales,
                raw.w2_scales,
                raw.w13_qzeros,
                raw.w2_qzeros,
                raw.w13_g_idx,
                raw.w2_g_idx,
            )
            if tensor is not None
        )

    @staticmethod
    def _raw_unquantized_nbytes(raw: _RawUnquantizedExpertWeights) -> int:
        # 缂備胶鍠曢鎼佹閻愬搫娅ら柛?raw buffer 闁诡剝顕ч悺褔鎳為崒娑欐闁?
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (raw.w13_weight, raw.w2_weight)
        )


def maybe_enable_tiered_moe_cache(model: nn.Module, cfie_config: Any) -> None:
    # ------------------------------- 閻犲洩顕цぐ?tiered MoE cache 閻犱讲鈧啿鐏婃鐐舵硾閸ㄤ粙寮鐔感﹂柛姘剧畱閹酣鎮?-------------------------------
    # 濞寸姴閰ｉ崢銈囩磾椤旂⒈鍤犻悹鐐┾偓鑼跺幀閻犲洩顕цぐ鍥触椤栨艾袟闁?planner 婵炲鍔岄崣鍡涙儍?tiered MoE cache 閻犱讲鈧啿鐏婇柕?
    plan = get_moe_tiered_cache_plan(cfie_config)

    # 鐟滅増鎹侀鎼佸礆閹哄秶鐟濋悗娑櫭﹢顏堟晬鐏炴儳鐏楅悹浣测偓鍐茬亰濞戞搩鍘藉Ο澶婎嚕韫囨梻鍨奸悹渚€顣︾拹鐔煎嫉椤忓嫭鍎欓柣顫妽濡炲倿鏁嶅畝鈧ú鍧楀箳閵夈劎绠查柛銉у剳缁辨繃绋夊鍡楃槸閺夌偞鍨濋幑銏℃媴閺囩喎浠橀柛鎺曟硾濞呮帡濡?
    if not plan or not bool(plan.get("enabled", False)):
        return

    # ------------------------------- 闁瑰灚鎸稿畵?tiered MoE cache 闁圭鍊藉ù鍥礈瀹ュ洦鐣遍柟顒冾唺缂嶅鎷嬮垾鍐茬亰濞ｅ洠鍓濇导?-------------------------------
    # 閺夊牊鎸搁崵顓°亹閹惧啿顤?tiered MoE cache 閻犱讲鈧啿鐏婇柣銊ュ閸櫻囨煥椤旇姤鍠呴悷鏇氭娣囧﹪骞侀銈囩濞撴艾銇樼花顒傛兜椤旀鍚囨俊顖椻偓宕団偓椋庣尵鐠囪尙鈧兘濡存稉鐑禪 婵″弶鍨濈紞鍛村Υ娑旑櫥rst 婵″弶鍨濈紞鍛村椽?CPU 婵″弶鍨濈紞鍛驳婢跺﹤妫橀柡浣藉焽閳?
    logger.info(
        "Preparing tiered MoE expert cache attachment: model_type=%s "
        "gpu_slots/layer=%d prefill_burst_slots=%d cpu_slots/layer=%d model=%s",
        plan.get("model_type", ""),
        int(plan.get("gpu_slots_per_layer", 0)),
        int(plan.get("prefill_burst_slots", 0)),
        int(plan.get("cpu_slots_per_layer", 0)),
        plan.get("model_path", ""),
    )

    # ------------------------------- 闁告帗绻傞～鎰板礌閺嵮冩瀻闁轰胶澧楀畵浣烘嫚鐠囨彃绲块悗娑櫭崑宥嗙▔鎼粹€冲伎閻忕偐鍋撶紓浣哄枙椤撴悂鎮╅懜纰樺亾?-------------------------------
    # 闁糕晞妗ㄧ花顒傛媼閳ュ啿鐏婂☉鎿冨幘濞堟垵螣閳ュ磭鈧鎹勯姘辩獮闁告帗绋戠紓?safetensors 濞戞挻鎸搁宥団偓娑櫭崑宥夋晬瀹€鈧弫銈嗙鎼粹剝鍊电紓渚囧幗鐎垫粎浠﹂崒妯峰亾娴ｇ懓鐦诲☉鎾存尭椤斿秶鎷犵拠鎻掔悼闁告劙鏀遍弳鐔煎箲椤旇　鍋?
    expert_store = SafetensorExpertStore(plan["model_path"])

    # 閻犱焦婢樼紞宥夊嫉閳ь剛绱掗崼銏″焸婵繐绲剧€垫洘娼?tiered cache controller 闁汇劌瀚惇浼村极閼割兘鍋?
    enabled_layers = 0

    # 閻犱焦婢樼紞宥呂熼垾宕団偓閿嬬▔椤撯槅娼堕柡宥呮穿椤斿洦绋夐崫鍕畨閻犲洢鍎遍幆搴ㄦ偨?tiered cache 闁?FusedMoE 閻忕偛鍊归弳鐔煎Υ?
    marked_layers = 0

    # 閻犲洩顕цぐ鍥╂媼閳ュ啿鐏婂☉鎿冨弮閸樸倗绱旈鐐暠闁稿繐褰夐棅?prefill burst pool 婵″弶鍨濈紞鍛村极閼割兘鍋?
    prefill_burst_slots = int(plan.get("prefill_burst_slots", 0))

    # 閻忓繋绮欓崳娲捶閵娿儲鍊卞☉鎾亾婵☆垪鈧磭鈧兘宕橀崨顓фЩ闁活潿鍔嬬粩瀛樼▔椤忓嫬褰欏ù?prefill burst pool闁挎稑濂旀禍鎺楀礄韫囨挾姣屽Λ鐗堢箓椤﹀寮伴幆褏鎽犻柛妤冨Х閺併倝濡?
    shared_prefill_burst_pool: SharedPrefillBurstPool | None = None

    # ------------------------------- 闂侇剙绉村璇参熼垾宕団偓宄拔熼垾铏仴妤犵偞婀圭拹鐔兼儎椤旂晫鍨?FusedMoE 閻忕偛鍊圭€垫洘娼懞銉ヤ粯闁告帟娉涘▍?-------------------------------
    # 闂傚啳鍩栭?1: 鐎点倖鍎肩换?burst pool 闁告帗绋戠紓? 闁稿繐鐗嗛悾顒勫箣閹邦厼顣查柡鍫濐槸閻即鎯?CPU 闂傚牊鐟﹂埀顑跨窔閺嗗懘宕撹箛鏃€缍忛柡鍌涚懃鐎垫煡濡?
    # 閺夆晜鐟﹂悧?repack 閺夆晛娲ㄩ埢鍏肩▔椤撴繄鐟濆ù鍏肩煯缁?burst pool 闁?~1.2 GiB GPU 闁告帒妫濋崢銈嗙婢跺顫ラ柡鍕劤閻°劑濡?
    controllers: list[LayerTieredExpertCacheController] = []
    shared_runtime_stage_pool = SharedRuntimeExpertStagePool()
    for module in model.modules():
        # 濞寸姴鎳庨ˇ鈺呮偠?FusedMoE 缂侇偉顕ч悗鐑芥儍閸曨偆婀撮柕?
        if not isinstance(module, FusedMoE):
            continue

        # 濞寸姴鎳庨ˇ鈺呮偠閸℃顤呴梻鍫涘灩閸ㄥ灚鎱ㄧ€ｎ亜顕ч梻鍐煐椤斿苯顔忛懠棰濇蕉闁哄秴娲╅鍥触椤栨粍鏆?CFIE tiered cache 闁汇劌瀚惇浼村Υ?
        if not getattr(module, "_cfie_tiered_cache_enabled", False):
            continue

        # 缂備胶鍠曢姝屻亹閹惧啿顤呮俊顖椻偓宕団偓閿嬬▔椤撯槅娼堕柡宥呮穿椤斿洭妫侀埀顒傛啺娴ｅ憡鍎欓柣?tiered cache 闁汇劌瀚惇浼村极閼割兘鍋?
        marked_layers += 1

        # 闂傚啳鍩栭?1 濞戞挸绉撮崹鍗烆嚈?burst pool (濞?None), 閻?repack 闁绘瑯鍓欏畷?GPU 濞戞挸鐡ㄥ鍌滅矚濞差亝锛? 闂侇剙鐏濋崢?2閼?闁告帒妫濋崢銈夊Υ?
        controller = LayerTieredExpertCacheController(
            layer=module,
            plan=plan,
            expert_store=expert_store,
            prefill_burst_pool=None,
        )
        module._cfie_tiered_cache_controller = controller
        controller._runtime_stage_pool = shared_runtime_stage_pool
        controllers.append(controller)
        enabled_layers += 1

    if controllers:
        _synchronize_torch_device_best_effort(getattr(controllers[0], "device", None))
        gc.collect()
        torch.accelerator.empty_cache()

    # 闂傚啳鍩栭?2: CPU 闂傗偓濠婂啫鍓奸柛蹇嬪姂閸庡鈧懓鏈崹姘跺触鎼粹€虫櫃闁告帒妫濋崢?shared prefill burst pool, 妤犵偛澧庣划锔锯偓瑙勮壘閸╁矂宕ラ崟顐ゆ勾闁?
    if prefill_burst_slots > 0 and controllers:
        template = controllers[0]
        shared_prefill_burst_pool = SharedPrefillBurstPool(
            template_layer=template.layer,
            num_slots=prefill_burst_slots,
        )
        for controller in controllers:
            if shared_prefill_burst_pool.supports_layer(controller.layer):
                controller.prefill_burst_pool = shared_prefill_burst_pool
            else:
                logger.warning(
                    "Skipping shared prefill burst pool on incompatible MoE layer: %s",
                    controller.layer.layer_name,
                )

    if controllers:
        runtime_stage_slots = max(
            1,
            prefill_burst_slots,
            *(
                max(
                    int(getattr(controller, "_runtime_stage_slots", 0) or 0),
                    int(getattr(controller, "_compute_slots", 0) or 0),
                )
                for controller in controllers
            ),
        )
        pin_runtime_stage = any(
            controller._runtime_requires_pinned_cpu()
            for controller in controllers
        )
        template = controllers[0]
        per_expert_bytes, cpu_signature = template.runtime_stage_allocation_spec(
            pin_memory=pin_runtime_stage,
        )
        stage_device = torch.device(template.device)
        mismatched_stage_layers: list[str] = []
        for controller in controllers[1:]:
            other_bytes, other_signature = controller.runtime_stage_allocation_spec(
                pin_memory=pin_runtime_stage,
            )
            if other_bytes != per_expert_bytes or other_signature != cpu_signature:
                mismatched_stage_layers.append(controller.layer_key)
        if mismatched_stage_layers:
            logger.warning(
                "Shared runtime expert stage was preallocated from %s, but %d "
                "layers have a different runtime bundle layout. The stage may "
                "need to reallocate lazily for those layers; first mismatched=%s",
                template.layer_key,
                len(mismatched_stage_layers),
                mismatched_stage_layers[0],
            )
        try:
            stage_info = shared_runtime_stage_pool.preallocate(
                capacity=runtime_stage_slots,
                per_expert_bytes=per_expert_bytes,
                pin_memory=pin_runtime_stage,
                cpu_signature=cpu_signature,
                device=stage_device,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to preallocate shared runtime expert CPU/GPU stage: "
                f"slots={runtime_stage_slots} "
                f"per_expert={per_expert_bytes / (1 << 20):.2f} MiB "
                f"pin_memory={pin_runtime_stage} device={stage_device}"
            ) from exc
        logger.info(
            "Preallocated shared runtime expert stage: slots=%d "
            "per_expert=%.2f MiB cpu=%.2f MiB pinned=%s gpu=%.2f MiB device=%s",
            int(stage_info["capacity"]),
            float(stage_info["per_expert_bytes"]) / (1 << 20),
            float(stage_info["cpu_bytes"]) / (1 << 20),
            bool(stage_info["cpu_pinned"]),
            float(stage_info["gpu_bytes"]) / (1 << 20),
            stage_info["device"],
        )

    # ------------------------------- 闁哄稄绻濋悰娆撴儎椤旂晫鍨奸悘鐐插€归弳鐔哥▔鎼达紕鏉介梻鍕噺鐎垫洘娼挊澶屾勾闁轰胶澧楀Σ鎼佸触閿旇法顏遍柤?-------------------------------
    if marked_layers and enabled_layers != marked_layers:
        raise RuntimeError(
            f"Tiered MoE cache expected {marked_layers} layers, but attached only "
            f"{enabled_layers}"
        )

    if enabled_layers > 0:
        logger.info(
            "Enabled tiered MoE expert cache on %d layers (plan=%s)",
            enabled_layers,
            PLAN_KEY,
        )
