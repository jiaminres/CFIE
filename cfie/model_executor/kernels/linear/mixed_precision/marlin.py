# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from cfie import _custom_ops as ops
from cfie.model_executor.layers.quantization.utils.marlin_utils import (
    MARLIN_SUPPORTED_GROUP_SIZES,
    apply_gptq_marlin_linear,
    check_marlin_supports_shape,
    marlin_act_int8_process_scales,
    marlin_is_k_full,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_permute_bias,
    marlin_permute_scales,
    marlin_sort_g_idx,
    marlin_zero_points,
    query_marlin_supported_quant_types,
    unpack_cols,
)
from cfie.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from cfie.platforms import current_platform
from cfie.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class MarlinLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        # 返回当前 kernel backend 对 GPU 的最低算力要求
        # 75 表示至少需要 SM75（Turing）及以上
        # choose_mp_linear_kernel(...) 会先用这个值做一轮粗筛
        return 75

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        # -------------------------------------------------------------
        # 1) 平台检查：Marlin 只支持 NVIDIA CUDA
        # -------------------------------------------------------------
        # 因为 Marlin 用了 inline PTX，所以不能跑在非 CUDA 平台
        if not current_platform.is_cuda():
            return False, "Marlin only supported on CUDA"

        # -------------------------------------------------------------
        # 2) 检查量化类型是否被 Marlin 支持
        # -------------------------------------------------------------
        # c.weight_type: 权重量化类型，例如 uint4 / uint4b8 等
        # c.zero_points: 当前配置是否启用 zero-points
        # 支持的量化类型集合可能和是否启用 zero_points 有关
        quant_types = query_marlin_supported_quant_types(c.zero_points)
        if c.weight_type not in quant_types:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by"
                f"  Marlin, supported types are: {quant_types}",
            )

        # -------------------------------------------------------------
        # 3) 检查 group_size 是否支持
        # -------------------------------------------------------------
        # c.group_size: 分组量化的组大小
        # Marlin 只支持一组固定集合里的 group_size
        if c.group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"Group size ({c.group_size}) not supported by "
                "Marlin, supported group sizes are: "
                f"{MARLIN_SUPPORTED_GROUP_SIZES}",
            )

        # -------------------------------------------------------------
        # 4) 检查 shape 是否满足 Marlin kernel 的约束
        # -------------------------------------------------------------
        # c.partition_weight_shape = (K_local, N_local)
        #   K_local: 当前 rank 上本地输入维
        #   N_local: 当前 rank 上本地输出维
        #
        # c.full_weight_shape = (K_global, N_global)
        #   K_global: 全局输入维
        #
        # 这里把:
        # - N_local（out_features）
        # - K_local（local in_features）
        # - K_global（global in_features）
        # - group_size
        # 传给 shape 检查器
        #
        # 为什么要同时传 K_local 和 K_global？
        # 因为 Marlin 需要区分：
        # - 当前是不是 row parallel（K_local != K_global）
        # - group 的划分是否和 local/global K 匹配
        return check_marlin_supports_shape(
            c.partition_weight_shape[1],  # out_features = N_local
            c.partition_weight_shape[0],  # in_features  = K_local
            c.full_weight_shape[0],  # full in_features = K_global
            c.group_size,
        )

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # 当前权重所在 device
        device = getattr(layer, self.w_q_name).device

        # kernel 配置对象，后面简写为 c
        c = self.config

        # -------------------------------------------------------------
        # 1) 判断输入激活是不是“1字节类型”
        # -------------------------------------------------------------
        # 比如 int8 / fp8 这种 itemsize == 1 的激活类型
        is_a_8bit = c.act_type is not None and c.act_type.itemsize == 1

        # 如果激活是 8bit，这里要求权重量化类型必须是 uint4b8
        # 否则当前 Marlin 路径不支持
        if is_a_8bit:
            assert c.weight_type == scalar_types.uint4b8, (
                "W8A8 is not supported by marlin kernel."
            )

        # -------------------------------------------------------------
        # 2) 若输入激活是 FP8(e4m3fn)，做额外预处理
        # -------------------------------------------------------------
        # 这和前面 AWQ/Marlin 那条路径类似：
        # - 先对量化权重做 fp8 预处理
        # - 再把 scales 放大 512
        if c.act_type == torch.float8_e4m3fn:
            ops.marlin_int4_fp8_preprocess(getattr(layer, self.w_q_name), inplace=True)
            getattr(layer, self.w_s_name).data = (
                    getattr(layer, self.w_s_name).data * 512
            )

        # -------------------------------------------------------------
        # 3) 判断当前是不是 row-parallel
        # -------------------------------------------------------------
        # 若 K_local != K_global，说明输入维被切分，属于 row parallel 语义
        row_parallel = c.partition_weight_shape[0] != c.full_weight_shape[0]

        # is_k_full 表示：
        # 当前 kernel 在解释 K 维时，能不能把它看成“完整 K”
        #
        # 它会综合：
        # - c.has_g_idx （是否启用 desc_act / activation-order）
        # - row_parallel（输入维是否被切）
        #
        # 后面 apply_gptq_marlin_linear(...) 会用到这个标记来决定
        # 如何解释 g_idx / scales / zp / K维关系
        self.is_k_full = marlin_is_k_full(c.has_g_idx, row_parallel)

        # -------------------------------------------------------------
        # 4) 分配 Marlin workspace
        # -------------------------------------------------------------
        # 这是后续 kernel 调用的临时工作区
        self.workspace = marlin_make_workspace_new(device)

        # -------------------------------------------------------------
        # 5) 给 g_idx / zp 准备默认名字
        # -------------------------------------------------------------
        # Marlin 这条路径要求这些参数“总是存在”
        # 即使当前逻辑上没有 g_idx / zero_points，也会塞一个空占位 tensor
        if self.w_gidx_name is None:
            self.w_gidx_name = "g_idx"
        if self.w_zp_name is None:
            self.w_zp_name = "w_zp"

        # -------------------------------------------------------------
        # 6) 定义 qweight 的变换函数
        # -------------------------------------------------------------
        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)

            # 先把参数布局规范化为：
            #   input_dim=0, output_dim=1, packed_dim=0
            #
            # 逻辑语义：
            #   qweight 对应本地权重 [K_local, N_local]
            #   只是 packed 发生在输入维 K 上
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)

            # 把当前 qweight repack 成 Marlin kernel 需要的格式
            #
            # x.data 当前逻辑上对应：
            #   packed([K_local, N_local])
            #
            # perm=layer.g_idx_sort_indices:
            #   若开启 desc_act，这里会把 activation-order 的重排顺序编码进 repack 过程
            #   若没开 desc_act，这通常是空占位
            #
            # size_k = K_local
            # size_n = N_local
            x.data = ops.gptq_marlin_repack(
                x.data.contiguous(),
                perm=layer.g_idx_sort_indices,
                size_k=c.partition_weight_shape[0],  # K_local
                size_n=c.partition_weight_shape[1],  # N_local
                num_bits=c.weight_type.size_bits,
                is_a_8bit=is_a_8bit,
            )
            return x

        # -------------------------------------------------------------
        # 7) 定义 scales 的变换函数
        # -------------------------------------------------------------
        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)

            # 规范化 scales 的布局：
            #   input_dim=0, output_dim=1
            #
            # 逻辑上 scales 常见 shape:
            #   [num_groups(或其变体), N_local]
            permute_param_layout_(x, input_dim=0, output_dim=1)

            # 把 checkpoint 里的 scales 排布变成 Marlin kernel 需要的排布
            x.data = marlin_permute_scales(
                x.data.contiguous(),
                size_k=c.partition_weight_shape[0],  # K_local
                size_n=c.partition_weight_shape[1],  # N_local
                group_size=c.group_size,
                is_a_8bit=is_a_8bit,
            )

            # 计算 group 数
            # 若 group_size == -1，表示 channelwise/整维一组，视作 1 组
            if c.group_size == -1:
                num_groups = 1
            else:
                num_groups = c.partition_weight_shape[0] // c.group_size

            # 对 int8 激活且多组量化的情况，还要额外生成 input_global_scale
            # 说明 Marlin 的 int8 激活路径还需要一个全局输入缩放参数
            if c.act_type == torch.int8 and num_groups > 1:
                x.data, input_global_scale = marlin_act_int8_process_scales(x.data)
                layer.register_parameter(
                    "input_global_scale",
                    torch.nn.Parameter(input_global_scale, requires_grad=False),
                )
            else:
                layer.input_global_scale = None
            return x

        # -------------------------------------------------------------
        # 8) 处理 g_idx / activation-order
        # -------------------------------------------------------------
        if c.has_g_idx:
            # 从 layer 上取出原始 g_idx，然后排序并得到：
            # - g_idx: 处理后的 g_idx 参数
            # - g_idx_sort_indices: 排序索引
            #
            # 这个排序索引后面会被用于：
            # - qweight repack
            # - kernel 执行阶段
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(
                getattr(layer, self.w_gidx_name)
            )

            # 把 layer 上的 g_idx 替换成处理后的版本
            self._transform_param(layer, self.w_gidx_name, lambda _: g_idx)

            # 保存排序索引
            layer.g_idx_sort_indices = g_idx_sort_indices
        else:
            # 没有 g_idx 时，也放一个空占位 tensor
            setattr(layer, self.w_gidx_name, marlin_make_empty_g_idx(device))
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        # -------------------------------------------------------------
        # 9) 处理 zero-points
        # -------------------------------------------------------------
        if c.zero_points:
            # grouped_k = K 方向上的 group 数
            # 若 group_size == -1，则按 1 组处理
            grouped_k = (
                c.partition_weight_shape[0] // c.group_size if c.group_size != -1 else 1
            )

            # 对 qzeros 做变换：
            # 1. x.t() 后交给 unpack_cols(...) 解包
            # 2. 再通过 marlin_zero_points(...) 转成 Marlin kernel 要的布局
            #
            # unpack 后的逻辑 shape 近似可理解为：
            #   [grouped_k, N_local]
            self._transform_param(
                layer,
                self.w_zp_name,
                lambda x: marlin_zero_points(
                    unpack_cols(
                        x.t(),  # 倒置
                        c.weight_type.size_bits,
                        grouped_k,
                        c.partition_weight_shape[1],  # N_local
                    ),
                    size_k=grouped_k,
                    size_n=c.partition_weight_shape[1],  # N_local
                    num_bits=c.weight_type.size_bits,
                    is_a_8bit=is_a_8bit,
                ),
            )
        else:
            # 没有 zero_points 时，同样放空占位
            setattr(layer, self.w_zp_name, marlin_make_empty_g_idx(device))

        # -------------------------------------------------------------
        # 10) 真正对 qweight / scales 执行变换
        # -------------------------------------------------------------
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        # -------------------------------------------------------------
        # 11) bias 也要跟着输出通道顺序一起 permute
        # -------------------------------------------------------------
        # 因为 qweight/scales/zp 的重排会改变输出通道在 kernel 里的布局，
        # 所以 bias 也必须同步重排，才能和输出列对齐
        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.data = marlin_permute_bias(layer.bias)

    def apply_weights(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,  # 输入激活，shape 常见: [..., K_local]
            bias: torch.Tensor | None = None,  # 本地 bias，shape 常见: [N_local]
    ) -> torch.Tensor:
        c = self.config

        # 取出当前 layer 上准备好的参数
        # - w_q    : Marlin 格式的 qweight
        # - w_s    : Marlin 格式的 scales
        # - w_zp   : Marlin 格式的 zero-points（若没启用，也会是空占位）
        # - w_gidx : g_idx（若没启用，也会是空占位）
        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)

        # process_weights_after_loading() 已经保证：
        # - w_zp 不是 None（真实参数或空占位）
        # - w_gidx 不是 None（真实参数或空占位）

        # 调用 GPTQ + Marlin 的核心 wrapper
        #
        # 输入:
        #   x shape 常见: [..., K_local]
        #
        # 本地形状:
        #   input_size_per_partition  = K_local
        #   output_size_per_partition = N_local
        #
        # kernel 输出逻辑 shape:
        #   [..., N_local]
        return apply_gptq_marlin_linear(
            input=x,
            weight=w_q,
            weight_scale=w_s,
            weight_zp=w_zp,  # type: ignore
            g_idx=w_gidx,  # type: ignore
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=self.workspace,
            wtype=c.weight_type,
            input_size_per_partition=c.partition_weight_shape[0],  # K_local
            output_size_per_partition=c.partition_weight_shape[1],  # N_local
            is_k_full=self.is_k_full,
            input_global_scale=getattr(layer, "input_global_scale", None),
            bias=bias,
            input_dtype=c.act_type,
        )
