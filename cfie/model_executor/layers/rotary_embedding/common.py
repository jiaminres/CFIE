# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from importlib.util import find_spec
from typing import Callable

import torch

from cfie.logger import init_logger
from cfie.model_executor.custom_op import CustomOp
from cfie.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_cfie_flash_attn_apply_rotary_emb: Callable | None = None
_cfie_flash_attn_rotary_unavailable = False


def _is_optional_cfie_flash_attn_import_error(exc: BaseException) -> bool:
    module_name = getattr(exc, "name", None)
    if isinstance(module_name, str) and module_name.startswith("triton"):
        return True

    return (
        "cfie.cfie_flash_attn requires the CUDA flash attention extensions"
        in str(exc)
    )


def _get_cfie_flash_attn_apply_rotary_emb() -> Callable | None:
    global _cfie_flash_attn_apply_rotary_emb
    global _cfie_flash_attn_rotary_unavailable

    if _cfie_flash_attn_apply_rotary_emb is not None:
        return _cfie_flash_attn_apply_rotary_emb

    if _cfie_flash_attn_rotary_unavailable:
        return None

    if find_spec("triton") is None:
        _cfie_flash_attn_rotary_unavailable = True
        logger.warning_once(
            "cfie_flash_attn rotary is unavailable because Triton is not "
            "installed; falling back to the native rotary embedding path."
        )
        return None

    try:
        from cfie.cfie_flash_attn.layers.rotary import apply_rotary_emb
    except (ImportError, ModuleNotFoundError) as exc:
        if not _is_optional_cfie_flash_attn_import_error(exc):
            raise

        _cfie_flash_attn_rotary_unavailable = True
        logger.warning_once(
            "cfie_flash_attn rotary is unavailable in the current runtime; "
            "falling back to the native rotary embedding path."
        )
        return None

    _cfie_flash_attn_apply_rotary_emb = apply_rotary_emb
    return _cfie_flash_attn_apply_rotary_emb


# common functions
def rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


# yarn functions
# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
    truncate: bool = True,
) -> tuple[float | int, float | int]:
    low = yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype
) -> torch.Tensor:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def yarn_get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


def _flashinfer_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """Custom op wrapper for flashinfer's rotary embedding.

    This is an in-place operation that modifies query and key tensors directly.
    """
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
    )


def _flashinfer_rotary_embedding_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    return


# Register flashinfer rotary embedding custom op
direct_register_custom_op(
    op_name="flashinfer_rotary_embedding",
    op_func=_flashinfer_rotary_embedding,
    mutates_args=["query", "key"],  # These tensors are modified in-place
    fake_impl=_flashinfer_rotary_embedding_fake,
)


# --8<-- [start:apply_rotary_emb]
@CustomOp.register("apply_rotary_emb")
class ApplyRotaryEmb(CustomOp):
    # 注册一个名为 "apply_rotary_emb" 的 custom op
    # 作用：把已经准备好的 cos / sin 应用到输入张量 x 上
    # 也就是执行“真正的旋转”这一步，而不是生成 cos/sin cache

    # --8<-- [end:apply_rotary_emb]

    def __init__(
        self,
        enforce_enable: bool = False,      # 是否强制启用 custom op 路径
        is_neox_style: bool = True,        # 是否使用 NeoX 风格；False 时用 GPT-J 风格
        enable_fp32_compute: bool = False, # 是否临时转成 FP32 计算以提高精度
    ) -> None:
        # 初始化 CustomOp 基类
        super().__init__(enforce_enable=enforce_enable)

        # 保存 RoPE 排布风格
        self.is_neox_style = is_neox_style

        # 保存是否启用 FP32 计算
        self.enable_fp32_compute = enable_fp32_compute

        # 先默认没有 flash_attn 的 rotary 实现
        self.apply_rotary_emb_flash_attn = None

        # 如果当前环境安装了可用的 flash_attn，就导入其中的 Triton rotary 实现。
        # 这里必须真的尝试 import，而不能只看 find_spec：
        # 某些源码树（例如 .deps/vllm-flash-attn-src）会让 find_spec 命中，
        # 但其底层 CUDA 扩展并未真正编译好，继续导入反而会在启动期崩溃。
        if find_spec("flash_attn") is not None:
            try:
                from flash_attn.ops.triton.rotary import apply_rotary
            except (ImportError, ModuleNotFoundError, OSError) as exc:
                logger.info_once(
                    "flash_attn rotary backend is unavailable, falling back to "
                    "the CFIE rotary path: %s",
                    exc,
                    scope="local",
                )
            else:
                # 保存 flash_attn 提供的 apply_rotary 函数
                self.apply_rotary_emb_flash_attn = apply_rotary

    @staticmethod
    def forward_static(
        x: torch.Tensor,                   # 形状: [batch_size(可选), seq_len, num_heads, head_size]
        cos: torch.Tensor,                 # 形状: [seq_len, head_size // 2]
        sin: torch.Tensor,                 # 形状: [seq_len, head_size // 2]
        is_neox_style: bool = True,        # 是否使用 NeoX 风格
        enable_fp32_compute: bool = False, # 是否临时转 FP32 计算
    ) -> torch.Tensor:
        """
        参数说明：
            x:
                [batch_size(可选), seq_len, num_heads, head_size]

            cos:
                [seq_len, head_size // 2]

            sin:
                [seq_len, head_size // 2]

            is_neox_style:
                True  -> NeoX 风格
                False -> GPT-J 风格（偶数维/奇数维交错）

            enable_fp32_compute:
                若为 True，则先把 x / cos / sin 临时转成 FP32 计算，
                最后再转回原 dtype，以提高数值精度
        """

        # 记录输入原始 dtype，后面如有需要再转回去
        origin_dtype = x.dtype

        # 若开启 FP32 计算，则先把 x 转成 float32
        if enable_fp32_compute:
            x = x.float()

        # 给 cos/sin 增加一个维度，使其可在 num_heads 维上广播
        # 原始形状: [seq_len, head_size // 2]
        # 变成:     [seq_len, 1, head_size // 2]
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)

        # -------- 按不同风格切分 x --------
        if is_neox_style:
            # NeoX 风格：直接把最后一维一分为二
            # x1: 前半部分
            # x2: 后半部分
            x1, x2 = torch.chunk(x, 2, dim=-1)
            # 若 x 最后一维是 head_size
            # 则 x1 / x2 形状均为:
            # [..., head_size // 2]
        else:
            # GPT-J 风格：取偶数位和奇数位
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            # 形状同样是:
            # [..., head_size // 2]

        # -------- 旋转公式 --------
        # 对每一对维度做二维旋转：
        # [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        # o1 / o2 形状:
        # 与 x1 / x2 相同

        # -------- 按风格拼回输出 --------
        if is_neox_style:
            # NeoX 风格：直接前后拼回去
            output = torch.cat((o1, o2), dim=-1)
            # 输出形状与 x 相同
        else:
            # GPT-J 风格：交错恢复
            # 先 stack 成 [..., head_size // 2, 2]
            # 再 flatten(-2) 变回 [..., head_size]
            output = torch.stack((o1, o2), dim=-1).flatten(-2)

        # 若启用了 FP32 计算，最后转回原 dtype
        if enable_fp32_compute:
            output = output.to(origin_dtype)

        return output

    def _pre_process(
        self,
        x: torch.Tensor,   # 可能是 [seq_len, num_heads, head_size] 或 [batch, seq_len, num_heads, head_size]
        cos: torch.Tensor, # [seq_len, rotary_dim / 2]
        sin: torch.Tensor, # [seq_len, rotary_dim / 2]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size, torch.dtype]:
        # 保存原始形状
        origin_shape = x.shape

        if len(origin_shape) == 3:
            # 如果输入 x 没有 batch 维：
            # [seq_len, num_heads, head_size]
            # 则补一个 batch 维，变成:
            # [1, seq_len, num_heads, head_size]
            x = x.unsqueeze(0)

        # 保存原始 dtype
        origin_dtype = x.dtype

        # 若启用 FP32 计算，则 x / cos / sin 都转成 float32
        if self.enable_fp32_compute:
            x = x.float()
            cos = cos.float()
            sin = sin.float()

        # 返回预处理后的张量、原始形状、原始 dtype
        return x, cos, sin, origin_shape, origin_dtype

    def _post_process(
        self,
        output: torch.Tensor,   # 处理后的输出，可能是 4 维
        origin_shape: torch.Size,# 原始输入形状
        origin_dtype: torch.dtype,# 原始输入 dtype
    ) -> torch.Tensor:
        if len(origin_shape) == 3:
            # 如果原始输入没有 batch 维，则去掉刚才补上的 batch 维
            # [1, seq_len, num_heads, head_size] -> [seq_len, num_heads, head_size]
            output = output.squeeze(0)

        if self.enable_fp32_compute:
            # 若之前临时转了 FP32，这里恢复原 dtype
            output = output.to(origin_dtype)

        return output

    def forward_native(
        self,
        x: torch.Tensor,   # 输入张量
        cos: torch.Tensor, # cos 表
        sin: torch.Tensor, # sin 表
    ) -> torch.Tensor:
        # 直接走纯 PyTorch 实现
        output = self.forward_static(
            x, cos, sin, self.is_neox_style, self.enable_fp32_compute
        )
        return output

    def forward_cuda(
        self,
        x: torch.Tensor,   # 形状常见: [seq_len, num_heads, head_size] 或 [batch, seq_len, num_heads, head_size]
        cos: torch.Tensor, # [seq_len_rotary, rotary_dim / 2]
        sin: torch.Tensor, # [seq_len_rotary, rotary_dim / 2]
    ) -> torch.Tensor:
        # CUDA 路径优先使用 cfie_flash_attn rotary；
        # 若当前运行时没有 Triton 或 flash attention 扩展，则回退到共享的原生实现。
        apply_rotary_emb = _get_cfie_flash_attn_apply_rotary_emb()
        if apply_rotary_emb is None:
            return self.forward_native(x, cos, sin)

        # 统一输入形状与 dtype
        x, cos, sin, origin_shape, origin_dtype = self._pre_process(x, cos, sin)

        """
        cfie_flash_attn 里的 apply_rotary_emb 参数约定：
            x:   [batch_size, seq_len, nheads, headdim]
            cos: [seqlen_rotary, rotary_dim / 2]
            sin: [seqlen_rotary, rotary_dim / 2]
            interleaved:
                False 表示 NeoX 风格
                True  表示 GPT-J 风格
        """

        # 注意：
        # is_neox_style=True  -> interleaved=False
        # is_neox_style=False -> interleaved=True
        interleaved = not self.is_neox_style

        # 调用 CUDA 高性能实现
        output = apply_rotary_emb(x, cos, sin, interleaved)

        # 恢复原始形状与 dtype
        output = self._post_process(output, origin_shape, origin_dtype)
        return output

    def forward_hip(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # ROCm 路径：
        # 若系统里有 flash_attn 的 Triton rotary 实现，则优先用它
        if self.apply_rotary_emb_flash_attn is not None:
            # 统一输入形状与 dtype
            x, cos, sin, origin_shape, origin_dtype = self._pre_process(x, cos, sin)

            """
            flash_attn 里的 apply_rotary 参数约定：
                x:   [batch_size, seq_len, nheads, headdim]
                cos: [seqlen_rotary, rotary_dim / 2]
                sin: [seqlen_rotary, rotary_dim / 2]
                interleaved:
                    False 表示 NeoX 风格
                    True  表示 GPT-J 风格
            """
            interleaved = not self.is_neox_style

            # 调用 flash_attn 的 Triton rotary 实现
            output = self.apply_rotary_emb_flash_attn(
                x, cos, sin, interleaved=interleaved
            ).type_as(x)

            # 恢复原始形状与 dtype
            output = self._post_process(output, origin_shape, origin_dtype)
        else:
            # 如果没有可用的 flash_attn 实现，则退回纯 PyTorch 实现
            output = self.forward_native(x, cos, sin)

        return output

    def forward_cpu(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # CPU 路径当前没有 fused 版本
        # 直接退回纯 PyTorch 实现
        # TODO: 以后可在这里启用 fused CPU ROPE
        return self.forward_native(x, cos, sin)

    def extra_repr(self) -> str:
        # 打印模块时显示的额外信息
        s = f"is_neox_style={self.is_neox_style}"
        s += f", enable_fp32_compute={self.enable_fp32_compute}"
        return s
