# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from typing import Any

import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE

import cfie.model_executor.layers.fused_moe  # noqa
from cfie import _custom_ops as ops
from cfie.logger import init_logger
from cfie.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from cfie.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from cfie.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
from cfie.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    UnquantizedFusedMoEMethod,
)
from cfie.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from cfie.model_executor.layers.quantization import QuantizationMethods
from cfie.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from cfie.model_executor.layers.quantization.utils import replace_parameter
from cfie.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override,
    get_linear_quant_method,
    override_config,
)
from cfie.model_executor.layers.quantization.utils.marlin_utils import (
    check_marlin_supported,
    check_moe_marlin_supports_layer,
    get_marlin_input_dtype,
    marlin_act_int8_process_scales,
    marlin_make_workspace_new,
    marlin_moe_permute_scales,
    marlin_permute_bias,
    marlin_repeat_scales_on_all_ranks,
    verify_marlin_supported,
)
from cfie.model_executor.parameter import (
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from cfie.platforms import current_platform
from cfie.scalar_type import scalar_types
from cfie.transformers_utils.config import get_safetensors_params_metadata
from cfie.utils.collection_utils import is_list_of

logger = init_logger(__name__)


def get_moe_quant_method(
        config: "GPTQMarlinConfig",
        layer: torch.nn.Module,
        prefix: str,
        moe_method_cls: type,
):
    cloned_config = deepcopy(config)

    if isinstance(layer, FusedMoE):
        # 当前 122B-A10B-GPTQ-Int4 的 routed experts 一般不会命中 dynamic 负匹配，
        # 所以这里通常不会退回 UnquantizedFusedMoEMethod；
        # 当前真正的 Int4 主体正是从这里进入 GPTQMarlinMoEMethod。
        if (
                get_dynamic_override(
                    cloned_config,
                    layer_name=prefix,
                )
                == False
        ):
            return UnquantizedFusedMoEMethod(layer.moe_config)

        # 若当前层带有明确前缀，则继续按模块名前缀匹配 dynamic 规则。
        # 这里可能把当前层的 bits / group_size / desc_act / sym 等字段
        # 改写成“这个前缀专属”的量化配置。
        if prefix:
            override_config(cloned_config, prefix=prefix)

        return moe_method_cls(cloned_config, layer.moe_config)
    return None


class GPTQMarlinConfig(QuantizationConfig):
    """Config class for GPTQ Marlin"""

    # (num_bits, is_sym) -> quant_type
    TYPE_MAP = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
    }

    def __init__(
            self,
            weight_bits: int,
            group_size: int,
            desc_act: bool,
            is_sym: bool,
            lm_head_quantized: bool,
            dynamic: dict[str, dict[str, int | bool]],
            full_config: dict[str, Any],
            modules_in_block_to_quantize: list[str] | None = None,
    ) -> None:
        # 先初始化 QuantizationConfig 基类部分。
        super().__init__()
        # 当 group_size=-1 时表示整条输出通道只分成一个组，此时 desc_act 没有实际意义。
        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            # 直接把 desc_act 归一化为 False，避免后续走无效分支。
            desc_act = False

        # GPTQModel use `dynamic` config property to allow per module
        # quantization config so each module can be individually optimized.
        # Format is dict[str, dict] where key is a regex string that can
        # perform both positive ("+:" prefixed) or negative ("-:" prefixed)
        # matching of a module.
        # Default to positive match, override base quant config mode, if no
        # prefix is used. Value is in dict format of field key and override
        # value.
        # Negative matching will skip quantization init for this module
        # entirely:
        # non-quantized inference. More details and quantization examples can be
        # found at: https://github.com/ModelCloud/GPTQModel
        # Example:
        #  # last 1/2 of the layers 10-21 has 8bit vs 4bit for 0-9
        #  # last 1/4 of the layers 16-21 has 8bit and group_size 64
        # dynamic = {
        #  #`.*\.` matches the layers_node prefix
        #  # positive match layer 10-15
        #  r"+:.*\.(?:1[0-5])\..*": {"bits": 8,},
        #  # positive match layer 16-21
        #  r"+:.*\.(?:1[6-9]|20|21)\..*": {"bits": 8, "group_size": 64,},
        #  r"-:.*\.moe\..*": {}, # negative match (skip) all `moe` layers
        # }
        # 保存按模块粒度覆盖量化配置的 dynamic 规则表。
        self.dynamic = dynamic

        # 记录权重量化 bit 数。
        self.weight_bits = weight_bits
        # 记录当前量化是否是对称量化。
        self.is_sym = is_sym

        # 根据 bit 数计算一个 int32 里能打包多少个量化值。
        self.pack_factor = 32 // weight_bits  # packed into int32
        # 保存 group-wise quant 的分组大小。
        self.group_size = group_size
        # 保存是否启用 activation-order / desc_act。
        self.desc_act = desc_act
        # 保存 lm_head 是否也参与量化。
        self.lm_head_quantized = lm_head_quantized
        # 保留完整原始配置，后面某些回退路径会直接复用。
        self.full_config = full_config

        # 校验当前 (bits, sym) 组合是否在 GPTQ Marlin 支持表里。
        if (weight_bits, is_sym) not in self.TYPE_MAP:
            # 不支持的量化组合直接在配置构造阶段报错。
            raise ValueError(
                f"Unsupported quantization config: bits={weight_bits}, sym={is_sym}"
            )

        # 把 (bits, sym) 映射成底层 kernel 使用的 quant_type。
        self.quant_type = self.TYPE_MAP[(weight_bits, is_sym)]

        # 记录哪些模块应该被视为量化模块；未提供时默认空列表。
        self.modules_in_block_to_quantize = modules_in_block_to_quantize or []
        # used to identify GPTQ model quantized by autoround
        # 读取 autoround 版本号，用于识别这类 GPTQ checkpoint。
        self.autoround_version = full_config.get("autoround_version", "")

    def __repr__(self) -> str:
        # 返回便于日志和调试查看的配置摘要字符串。
        return (
            f"GPTQMarlinConfig(quant_type={self.quant_type}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}, "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"dynamic={self.dynamic}, "
            f"modules_in_block_to_quantize={self.modules_in_block_to_quantize})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        # 返回这一路量化方法在框架中的注册名。
        return "gptq_marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        # 声明 GPTQ Marlin 支持的激活 dtype。
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # 声明运行该 kernel 所需的最小 CUDA capability。
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        # 声明从模型目录搜索量化配置时接受的文件名列表。
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GPTQMarlinConfig":
        # 先读取 dynamic 配置，没有则默认空字典。
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        # 某些 checkpoint 可能把 dynamic 显式写成 None，这里归一化为空字典。
        dynamic = {} if dynamic is None else dynamic

        # 读取权重量化 bit 数。
        weight_bits = cls.get_from_keys(config, ["bits"])
        # 读取 group-wise quant 的组大小。
        group_size = cls.get_from_keys(config, ["group_size"])
        # 读取是否启用 desc_act。
        desc_act = cls.get_from_keys(config, ["desc_act"])
        # 读取是否使用对称量化。
        is_sym = cls.get_from_keys(config, ["sym"])
        # 读取 lm_head 是否量化，缺省时按 False 处理。
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        # 读取模块白名单/清单，缺省时保留 None 让构造函数再做归一化。
        modules_in_block_to_quantize = cls.get_from_keys_or(
            config, ["modules_in_block_to_quantize"], default=None
        )
        # 用解析出的字段构造最终 GPTQMarlinConfig 实例。
        return cls(
            weight_bits,
            group_size,
            desc_act,
            is_sym,
            lm_head_quantized,
            dynamic,
            config,
            modules_in_block_to_quantize,
        )

    @classmethod
    def override_quantization_method(
            cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None:
        # 先判断这份 HF 量化配置是否可以在运行时自动转换到 gptq_marlin。
        can_convert = cls.is_gptq_marlin_compatible(hf_quant_cfg)

        # 用户未显式指定量化方法，或指定的是 marlin / gptq_marlin，都算允许自动切换。
        is_valid_user_quant = (
                user_quant is None or user_quant == "marlin" or user_quant == "gptq_marlin"
        )

        # 同时满足“可转换”与“用户没有阻止”时，直接覆盖到 gptq_marlin。
        if can_convert and is_valid_user_quant:
            # 组织一条运行时转换提示日志。
            msg = (
                "The model is convertible to {} during runtime."
                " Using {} kernel.".format(cls.get_name(), cls.get_name())
            )
            # 把转换结果打印到日志。
            logger.info(msg)
            # 返回新的量化方法名，让上层改走 gptq_marlin。
            return cls.get_name()

        # 如果模型可转换，但用户显式要求继续走 gptq，就只提示而不强制切换。
        if can_convert and user_quant == "gptq":
            # 提醒用户显式选择了较慢的 gptq 路径。
            logger.info(
                "Detected that the model can run with gptq_marlin"
                ", however you specified quantization=gptq explicitly,"
                " so forcing gptq. Use quantization=gptq_marlin for"
                " faster inference"
            )
        # 其他情况都不覆盖用户或模型当前的量化方法。
        return None

    def get_quant_method(
            self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        # -------------------- 先按层类型分流量化方法 --------------------
        # FusedMoE 不能直接复用普通 Linear 的 GPTQMarlinLinearMethod。
        # 原因是：
        # 1. FusedMoE 的权重布局不是单个线性矩阵，而是多 expert 打包后的专用布局
        # 2. 运行时执行也不是普通 GEMM，而是 routed experts 对应的 MoE kernel
        # 3. 支持性检查、fallback 路径以及 dynamic 规则落点都和线性层不同
        # 因此这里必须先把 FusedMoE 单独分出来，走专用的 MoE quant method 选择逻辑。
        if isinstance(layer, FusedMoE):
            # -------------------- FusedMoE 专用分支 --------------------
            # 只有进入 MoE 分支时才延迟导入 MoeWNA16Config，避免无关路径增加依赖。
            from cfie.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config

            # 先检查当前 FusedMoE 层是否满足 GPTQ Marlin MoE kernel 的约束。
            # 这里检查的是“MoE 专用 kernel 是否能接这个层”，而不是普通线性层
            # 是否能做 GPTQ Marlin，因此必须放在 FusedMoE 分支单独判断。
            if not check_moe_marlin_supports_layer(layer, self.group_size):
                # 若当前层不满足 GPTQMarlinMoE 的限制，则不能硬套普通 Linear
                # 量化路径，否则权重布局和执行 kernel 都会对不上。
                # 这里统一回退到 MoeWNA16，让该 MoE 层仍能以专用 MoE 量化内核执行。
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by GPTQMoeMarlin. "
                    "Falling back to Moe WNA16 kernels."
                )
                # 用同一份 full_config 构造 MoeWNA16 配置，并返回它的 quant method。
                return MoeWNA16Config.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )

            # 支持时，为当前 FusedMoE 层构造 GPTQMarlinMoEMethod。
            # 这条路径会保留 MoE 语义，包括 routed experts 的专用权重组织方式。
            moe_quant_method = get_moe_quant_method(
                self,
                layer,
                prefix,
                GPTQMarlinMoEMethod
            )
            # 如果动态规则或层类型判定后认为不应量化，就返回 None。
            if moe_quant_method is None:
                return None

            # 当前 routed experts 进入这里后，会继续保留 GPTQ 配置值：
            # - bits = 4
            # - group_size = 128
            # - desc_act = false
            # - sym = true
            # 给 MoE quant method 注入当前前缀对应的输入激活 dtype。
            moe_quant_method.input_dtype = get_marlin_input_dtype(prefix)
            # 返回为该 MoE 层准备好的量化执行方法。
            return moe_quant_method

        # -------------------- 普通线性层分支 --------------------
        # 非 FusedMoE 层统一按普通线性层处理。
        # 这里走的是 GPTQMarlinLinearMethod，对应普通矩阵乘线性层的量化执行路径。
        # 非 MoE 场景统一走线性层量化方法选择逻辑。
        quant_method = get_linear_quant_method(
            self, layer, prefix, GPTQMarlinLinearMethod
        )
        # 如果该层不该量化或不支持 GPTQ Marlin 线性路径，就返回 None。
        if quant_method is None:
            return None
        # 当前 122B-A10B MoE 模型里，线性层若能走到这里，说明它既在
        # checkpoint 标记为量化层，又没有命中 -:.*attn.* / -:.*shared_expert.*
        # 之类的 dynamic 排除。实际数量通常远少于 routed experts。
        # 给线性层 quant method 注入当前前缀对应的输入激活 dtype。
        quant_method.input_dtype = get_marlin_input_dtype(prefix)
        # 返回为该线性层准备好的量化执行方法。
        return quant_method

    @classmethod
    def is_gptq_marlin_compatible(cls, quant_config: dict[str, Any]):
        # 读取量化方法名，并统一转成小写便于比较。
        quant_method = quant_config.get("quant_method", "").lower()
        # 读取量化 bit 数。
        num_bits = quant_config.get("bits")
        # 读取 group_size。
        group_size = quant_config.get("group_size")
        # 读取对称量化标记。
        sym = quant_config.get("sym")
        # 读取 desc_act 标记。
        desc_act = quant_config.get("desc_act")

        # 当前平台既不是 CUDA 也不是 CPU 时，直接判定不兼容。
        if not (current_platform.is_cuda() or current_platform.is_cpu()):
            return False

        # 只有原始量化方法是 gptq，才有转成 gptq_marlin 的意义。
        if quant_method != "gptq":
            return False

        # Marlin conversion is only valid if required properties are found
        # 缺少任何关键字段时，都无法安全判断兼容性。
        if num_bits is None or group_size is None or sym is None or desc_act is None:
            return False

        # 如果 (bits, sym) 组合不在支持表内，也不能转成 Marlin。
        if (num_bits, sym) not in cls.TYPE_MAP:
            return False

        # 最后再调用底层能力检查，确认 quant_type 与 group_size 组合真的受支持。
        return check_marlin_supported(
            quant_type=cls.TYPE_MAP[(num_bits, sym)], group_size=group_size
        )

    def apply_cfie_mapper(self, hf_to_cfie_mapper):
        # 只有在模块列表非空时，才需要把 HF 路径命名映射成 CFIE 命名。
        if self.modules_in_block_to_quantize is not None:
            # 对模块名列表整体应用映射器，便于后续按 CFIE 前缀匹配层名。
            self.modules_in_block_to_quantize = hf_to_cfie_mapper.apply_list(
                self.modules_in_block_to_quantize
            )

    def maybe_update_config(self, model_name: str, revision: str | None = None):
        # 如果配置里已经带了模块列表，就优先复用，不再重新扫描 checkpoint。
        if self.modules_in_block_to_quantize:
            # 某些配置会把模块列表写成 list[list[str]]，这里先做一次扁平化。
            if is_list_of(self.modules_in_block_to_quantize, list):
                # original modules_in_block_to_quantize: list[list[str]]
                # flatten original modules_in_block_to_quantize
                # 把嵌套列表展开成一维模块名列表。
                self.modules_in_block_to_quantize = [
                    item
                    for sublist in self.modules_in_block_to_quantize
                    for item in sublist
                ]
            # 已有模块列表时到此为止，不再执行后面的 checkpoint 扫描。
            return

        # 这些 dtype 视为未量化权重 dtype，后面会用来筛掉非量化参数。
        unquant_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        # 读取 safetensors 参数元数据，用于判断每个参数实际存储 dtype。
        metadata = get_safetensors_params_metadata(model_name, revision=revision)
        # 收集所有“dtype 不是未量化浮点类型”的参数所属层名前缀。
        quant_layers: set[str] = {
            param_name.rsplit(".", 1)[0]
            for param_name, info in metadata.items()
            if (dtype := info.get("dtype", None))
               and _SAFETENSORS_TO_TORCH_DTYPE[dtype] not in unquant_dtypes
        }
        # 把扫描出的量化层名前缀保存回配置，供后续层匹配使用。
        self.modules_in_block_to_quantize = list(quant_layers)


class GPTQMarlinLinearMethod(LinearMethodBase):
    """Linear method for GPTQ Marlin.

    这是 GPTQ + Marlin 路径对应的线性层量化实现方法。
    它不是一个 nn.Module 层本身，而是被某个 Linear 层挂载为 self.quant_method。

    主要职责：
    1. create_weights:
       为当前层注册 GPTQ 所需的量化参数：
       - qweight : 量化后的 packed 权重
       - g_idx   : activation order / group index 信息
       - scales  : scale 参数
       - qzeros  : zero-point 参数
    2. process_weights_after_loading:
       checkpoint 加载后，交给所选 kernel backend 做必要的后处理/重排
    3. apply:
       forward 时调用所选 kernel backend 执行量化线性计算

    和 AWQMarlinLinearMethod 的重要区别：
    - GPTQ 这里支持 desc_act（activation-order），因此需要 g_idx
    - 会根据 shape / dtype / group_size / desc_act 等自动选择合适的 MPLinear kernel
    """

    # 类级别集合：记录本进程已经打印过“正在使用哪个 kernel backend”
    # 避免同一个 backend 被重复 logger.info 很多次
    _kernel_backends_being_used: set[str] = set()

    def __init__(self, quant_config: GPTQMarlinConfig) -> None:
        # 保存 GPTQ Marlin 配置
        # 常见会包含：
        # - quant_type（例如 uint4 / uint8 等）
        # - group_size
        # - desc_act（是否使用 activation-order / descending activation）
        # - pack_factor
        self.quant_config = quant_config

        # 输入激活 dtype，默认先未知
        # 后续可能由外部按运行环境设成 fp16 / bf16 / int8 / fp8 等
        self.input_dtype = None

        # 量化类型直接来自配置
        # 例如可能是 int4 / uint4 等
        self.quant_type = self.quant_config.quant_type

        # 在初始化阶段就校验：
        # 当前平台是否支持这种 GPTQ Marlin 配置
        #
        # 例如会检查：
        # - 当前 quant_type 是否支持
        # - group_size 是否合法
        # - 平台 / kernel 能否覆盖这种组合
        verify_marlin_supported(
            quant_type=self.quant_config.quant_type,
            group_size=self.quant_config.group_size,
        )

    def create_weights(
            self,
            layer: torch.nn.Module,  # 当前被量化的线性层对象
            input_size_per_partition: int,  # 当前 TP rank 本地输入维，记作 K_local
            output_partition_sizes: list[int],  # 当前 TP rank 各逻辑输出块的 local 输出维列表
            input_size: int,  # 全局输入维，记作 K_global
            output_size: int,  # 全局输出维，记作 N_global
            params_dtype: torch.dtype,  # scale 参数等浮点参数 dtype
            **extra_weight_attrs,  # 额外信息，通常包含 weight_loader
    ) -> None:
        # 当前本地 checkpoint 的 GPTQ 配置可确定为：
        # - bits = 4
        # - group_size = 128
        # - desc_act = false
        # - sym = true
        # 因此：
        # - pack_factor = 32 / 4 = 8
        # - has_g_idx = False（虽然仍会先创建 g_idx 参数位）
        # - qzeros 在对称量化下通常不会真正参与 kernel 计算，但为了统一加载接口仍会创建
        # 当前 rank 本地总输出维，记作 N_local
        #
        # 普通线性层时：
        #   output_partition_sizes = [N_local]
        #
        # QKV / merged 时可能是：
        #   [q_local, k_local, v_local]
        # 最后求和得到总 local 输出维
        output_size_per_partition = sum(output_partition_sizes)

        # 判断当前层是不是 row-parallel
        #
        # 这里的判定逻辑是：
        # - 如果 input_size != input_size_per_partition
        #   说明输入维 K 被切分了
        #   那就是 row parallel 风格
        #
        # - 如果 input_size == input_size_per_partition
        #   说明输入维没切
        #   更像 column parallel 或无 TP
        is_row_parallel = input_size != input_size_per_partition

        # 从外部拿 weight_loader，后面挂到参数对象上
        weight_loader = extra_weight_attrs.get("weight_loader")

        # 当前激活 dtype
        # 若 self.input_dtype 尚未设定，则后面会退回 params_dtype
        input_dtype = self.input_dtype

        # -------------------------------------------------------------
        # Step 1. 组织一个“多并行线性层 kernel 配置”
        # -------------------------------------------------------------
        # full_weight_shape:
        #   全局逻辑权重 shape = [K_global, N_global]
        #
        # partition_weight_shape:
        #   当前 rank 本地逻辑权重 shape = [K_local, N_local]
        #
        # weight_type:
        #   权重量化类型（比如 int4/uint4）
        #
        # act_type:
        #   激活类型。若 input_dtype 未显式指定，则默认用 params_dtype
        #
        # group_size:
        #   GPTQ group-wise quant 的组大小
        #
        # zero_points=False:
        #   这里是传给 backend kernel config 的一个语义标记；
        #   虽然本类确实创建了 qzeros 参数，但该字段更多表示 kernel config 的某种路径选择
        #
        # has_g_idx=self.quant_config.desc_act:
        #   若 desc_act=True，表示需要 activation-order / g_idx

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),  # [K_global, N_global]
            partition_weight_shape=(
                input_size_per_partition,  # K_local
                output_size_per_partition,  # N_local
            ),
            weight_type=self.quant_config.quant_type,
            act_type=params_dtype if input_dtype is None else input_dtype,
            group_size=self.quant_config.group_size,
            zero_points=False,
            has_g_idx=self.quant_config.desc_act,
        )

        # -------------------------------------------------------------
        # Step 2. 根据配置自动选择最合适的 MPLinear kernel backend
        # -------------------------------------------------------------
        # 选择依据通常包括：
        # - 当前是 row/column parallel
        # - K/N shape
        # - quant_type
        # - act_type
        # - group_size
        # - 是否有 g_idx
        #
        # 返回的是一个“kernel 类”，后面会实例化
        kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)

        # 只在某个 backend 第一次出现时打印日志，避免刷屏
        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for GPTQMarlinLinearMethod", kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        # -------------------------------------------------------------
        # Step 3. 归一化 group_size
        # -------------------------------------------------------------
        # 若 group_size != -1:
        #   直接使用配置中的 group_size
        #
        # 若 group_size == -1:
        #   视为整条输入维作为一个组
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        # -------------------------------------------------------------
        # Step 4. 决定 scales / qzeros 在 TP 下如何处理
        # -------------------------------------------------------------
        # 核心问题：
        #   在 TP>1 时，scales / qzeros 是：
        #   A) 每个 rank 都保留完整一份（repeat）
        #   B) 像普通权重那样按 rank 切分（shard）
        #
        # 这个决策由以下因素共同决定：
        # - desc_act 是否开启
        # - group_size
        # - 当前层是否 row parallel
        #
        # 如果 marlin_repeat_scales_on_all_ranks(...) 返回 True：
        #   scales / qzeros 在每个 GPU 上都放完整重复副本
        #   -> scales_and_zp_input_dim = None
        #   -> weight_loader 会知道“不要沿输入维切分它”
        #
        # 此时 scales_and_zp_size 按全局输入维计算：
        #   = K_global / group_size
        if marlin_repeat_scales_on_all_ranks(
                self.quant_config.desc_act,
                self.quant_config.group_size,
                is_row_parallel
        ):
            # scale_dim=None 的语义：
            # “加载器请不要把 scales / zp 按输入维切 shard，而是每个 rank 放完整一份”
            scales_and_zp_input_dim = None
            scales_and_zp_size = input_size // group_size

        else:
            # 否则，scales / qzeros 在 TP 下按本地输入维进行切分
            # 即当前 rank 只存自己那部分
            #
            # scale_dim=0 的语义：
            # “加载器可以沿第 0 维（输入/group 维）切 shard”
            scales_and_zp_input_dim = 0
            scales_and_zp_size = input_size_per_partition // group_size

        # -------------------------------------------------------------
        # Step 5. 创建 qweight（GPTQ packed 权重）
        # -------------------------------------------------------------
        # 这里和前面 AWQMarlin 的一个重要区别是 packed 方向不同。
        #
        # 当前 qweight shape:
        #   [K_local / pack_factor, N_local]
        #
        # 即：
        # - packed_dim = 0
        # - packed 发生在输入维 K 上
        #
        # 这说明 GPTQ+Marlin 这里的 checkpoint/kernel 约定里，
        # 权重是沿第 0 维进行 packed 存储的
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,  # K_local / pack_factor
                output_size_per_partition,  # N_local
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        # -------------------------------------------------------------
        # Step 6. 创建 g_idx（activation order / group index）
        # -------------------------------------------------------------
        # g_idx shape:
        #   [K_local]
        #
        # 它通常用于 desc_act=True 的 GPTQ 路径：
        # - 表示输入通道/组的重排顺序
        # - 或者用于 kernel 在恢复 activation-order 时的索引信息
        #
        # 若 desc_act=False，虽然这里也创建了 g_idx，
        # 但实际 kernel 可能不会真正使用它
        g_idx = RowvLLMParameter(
            data=torch.empty(
                input_size_per_partition,  # K_local
                dtype=torch.int32,
            ),
            input_dim=0,
            weight_loader=weight_loader,
        )

        # -------------------------------------------------------------
        # Step 7. 构造 qzeros / scales 的公共参数字典
        # -------------------------------------------------------------
        # qzeros:
        #   shape = [scales_and_zp_size, N_local / pack_factor]
        #
        # scales:
        #   shape = [scales_and_zp_size, N_local]
        #
        # 这里 scales_and_zp_size 可能是：
        # - K_global / group_size （若每 rank 重复完整一份）
        # - K_local  / group_size （若按 rank shard）
        qzeros_args = {
            "data": torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            "weight_loader": weight_loader,
        }
        weight_scale_args = {
            "data": torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,  # scale的精度和激活精度一致
            ),
            "weight_loader": weight_loader,
        }

        # -------------------------------------------------------------
        # Step 8. 按“是否重复到所有 rank”选择 scales / qzeros 的参数类型
        # -------------------------------------------------------------
        if scales_and_zp_input_dim is None:
            # 情况 A：scales / qzeros 在每个 rank 上都放完整一份
            #
            # scales:
            #   ChannelQuantScaleParameter(output_dim=1)
            #   说明它主要按输出维 N 组织，不沿输入维做 TP shard
            #
            # qzeros:
            #   PackedColumnParameter(...)
            #   packed_dim=1 表示 packed 发生在输出维
            #
            # 这里可以把它理解成：
            #   “这是每个 rank 都完整持有的一套按输出列组织的量化元信息”
            scales = ChannelQuantScaleParameter(output_dim=1, **weight_scale_args)
            qzeros = PackedColumnParameter(
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )

        else:
            # 情况 B：scales / qzeros 在 TP 下按输入/group 维切分
            #
            # scales:
            #   GroupQuantScaleParameter(input_dim=0, output_dim=1)
            #
            # qzeros:
            #   PackedvLLMParameter(input_dim=0, output_dim=1, packed_dim=1)
            #
            # 这里表达的含义更接近：
            #   “当前 rank 只持有自己那部分 group 对应的 scale/zp”
            scales = GroupQuantScaleParameter(
                output_dim=1,
                input_dim=0,
                **weight_scale_args
            )
            qzeros = PackedvLLMParameter(
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )

        # -------------------------------------------------------------
        # Step 9. 把 GPTQ 所需参数注册到 layer 上
        # -------------------------------------------------------------
        layer.register_parameter("qweight", qweight)  # [K_local/pack_factor, N_local]
        layer.register_parameter("g_idx", g_idx)  # [K_local]
        layer.register_parameter("scales", scales)  # [groups_or_global_groups, N_local]
        layer.register_parameter("qzeros", qzeros)  # [groups_or_global_groups, N_local/pack_factor]

        # -------------------------------------------------------------
        # Step 10. 实例化所选 kernel backend
        # -------------------------------------------------------------
        # 这里 self.kernel 不是函数，而是一个 backend kernel 对象。
        #
        # 它知道：
        # - 当前 layer 的权重量化配置
        # - 哪些参数名对应 qweight / scales / qzeros / g_idx
        #
        # 后面：
        # - process_weights_after_loading -> 调 self.kernel.process_weights_after_loading(layer)
        # - apply -> 调 self.kernel.apply_weights(layer, x, bias)
        self.kernel = kernel_type(
            mp_linear_kernel_config,
            w_q_param_name="qweight",
            w_s_param_name="scales",
            w_zp_param_name="qzeros",
            w_gidx_param_name="g_idx",
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # checkpoint 加载完之后，让具体 kernel backend 去做后处理
        #
        # 这一步通常可能包含：
        # - 重排 qweight 布局
        # - 处理 scales / qzeros
        # - 处理 g_idx / activation-order
        # - 分配 workspace
        #
        # 和 AWQ 那边“在 method 里显式写 repack”不同，
        # GPTQ 这里把后处理逻辑进一步下沉到了 backend kernel 对象里
        #
        # 但要注意：对当前 122B-A10B MoE 模型，这个 GPTQMarlinLinearMethod
        # 并不是最重的后处理来源。真正重量级的后处理主要还是 routed experts 的
        # GPTQMarlinMoEMethod.process_weights_after_loading(...)。
        self.kernel.process_weights_after_loading(layer)

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,  # 输入激活，shape 常见: [..., K_local] 或 [..., K]
            bias: torch.Tensor | None = None,  # 本地 bias，shape 常见: [N_local]
    ) -> torch.Tensor:
        # forward 时，直接委托给具体 kernel backend
        #
        # 逻辑输出 shape:
        #   [..., N_local]
        #
        # 若外层线性层是 ColumnParallelLinear 且 gather_output=False：
        #   这里得到的就是当前 TP rank 的 local 输出
        #
        # 若外层后续还做 all-gather，才会再拼成 [..., N_global]
        return self.kernel.apply_weights(layer, x, bias)


class GPTQMarlinMoEMethod(FusedMoEMethodBase):
    """GPTQ Marlin 对应的 MoE 量化执行方法。"""

    def __init__(
            self,
            quant_config: GPTQMarlinConfig,
            moe: FusedMoEConfig,
    ) -> None:
        # 先初始化 FusedMoeMethodBase 负责的通用 MoE 状态。
        super().__init__(moe)
        # 保存当前层对应的 GPTQ Marlin 量化配置。
        self.quant_config = quant_config
        # 按量化 bit 数选择当前真正使用的 Marlin quant_type。
        if self.quant_config.quant_type.size_bits == 4:
            self.quant_type = scalar_types.uint4b8
        elif self.quant_config.quant_type.size_bits == 8:
            self.quant_type = scalar_types.uint8b128
        else:
            raise ValueError("GPTQMarlinMoEMethod only supports int4 and int8 now.")
        # 输入激活 dtype 由外层在构造 quant method 后再注入。
        self.input_dtype = None
        # 标记当前方法走 Marlin 内核路径。
        self.use_marlin = True

    def create_weights(
            self,
            layer: torch.nn.Module,
            num_experts: int,
            hidden_size: int,
            intermediate_size_per_partition: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs,
    ):
        # ----------------- 基础输入与运行模式 -----------------
        # 把当前输入激活 dtype 回写到 layer，供后续执行路径使用。
        layer.input_dtype = self.input_dtype  # None
        # itemsize==1 通常表示 int8 / fp8 这类 8-bit 激活输入。
        is_a_8bit = self.input_dtype is not None and self.input_dtype.itemsize == 1

        # 当前 Marlin MoE kernel 不支持 W8A8-INT8 这一路径。
        if is_a_8bit:
            assert self.quant_type == scalar_types.uint4b8, (
                "W8A8-INT8 is not supported by marlin kernel."
            )

        # 读取完整 intermediate size，desc_act 场景下某些张量尺寸要按 full size 计算。
        intermediate_size_full = extra_weight_attrs.pop("intermediate_size_full")  # 1024

        # desc_act 关闭时，当前层天然视为 full-k 布局。
        # desc_act 开启时，只有局部分片等于完整尺寸时才视为 full-k。
        self.is_k_full = (not self.quant_config.desc_act) or (
                intermediate_size_per_partition == intermediate_size_full
        )

        # ----------------- 量化分组布局 -----------------
        # 按 group_size 计算 w13 / w2 的 scale 分组布局。
        if self.quant_config.group_size != -1:  # 128
            # gate/up 侧按 hidden_size 维度和 group_size 计算 group 数。
            # gate/up属于列切割, 无需考虑is_k_full的问题
            scales_size13 = hidden_size // self.quant_config.group_size
            # down_proj 在 desc_act 场景可能要按 full w2 尺寸处理。
            w2_scales_size = (
                intermediate_size_full
                if self.quant_config.desc_act
                else intermediate_size_per_partition
            )
            scales_size2 = w2_scales_size // self.quant_config.group_size
            strategy = FusedMoeWeightScaleSupported.GROUP.value
        else:
            # group_size=-1 表示走按通道量化，每个 expert 只保留一组 scales。
            scales_size13 = 1
            scales_size2 = 1
            strategy = FusedMoeWeightScaleSupported.CHANNEL.value

        # 把两条投影路径各自的分组数记录到 layer 上。
        layer.num_groups_w13 = scales_size13
        layer.num_groups_w2 = scales_size2

        # 给后续注册的参数统一打上量化相关元信息。
        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": True})

        # ----------------- packed qweight -----------------
        # 注册 gate_proj + up_proj 融合后的 packed qweight。
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.quant_config.pack_factor,  # 量化方向k维度
                2 * intermediate_size_per_partition,  # n维度切割
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        # 注册 down_proj 对应的 packed qweight。
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.quant_config.pack_factor,  # 量化方向k维度， k维度切割
                hidden_size,  #
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        # ----------------- scales 与 qzeros -----------------
        # 注册 gate/up 对应的 scales。
        w13_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        # 注册 down_proj 对应的 scales。
        w2_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size2,
                hidden_size,
                dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)
        # desc_act 场景下，w2 scales 需要按完整权重视图加载，不能只按局部分片加载。
        set_weight_attrs(w2_scales, {"load_full_w2": self.quant_config.desc_act})

        # 注册 gate/up 对应的 qzeros。
        w13_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)

        # 注册 down_proj 对应的 qzeros。
        w2_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size2,
                hidden_size // self.quant_config.pack_factor,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)
        # desc_act 场景下，w2 qzeros 同样需要按完整权重视图加载。
        set_weight_attrs(w2_qzeros, {"load_full_w2": self.quant_config.desc_act})

        # ----------------- act-order 索引 -----------------
        # 注册 gate/up 的 g_idx。
        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)
        # 注册 down_proj 的 g_idx。
        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        # 注册 gate/up 的 g_idx 排序索引。
        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        # 注册 down_proj 的 g_idx 排序索引。
        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        # ----------------- kernel 工作区 -----------------
        # 为 Marlin MoE kernel 申请工作区。
        device = layer.w13_qweight.device
        layer.workspace = marlin_make_workspace_new(device, 4)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # ----------------- 入口：checkpoint 加载完成后的 GPTQ Marlin 后处理 -----------------
        # 这一阶段会把“按 checkpoint 友好布局保存的参数”转换成 Marlin kernel 真正期望的运行时布局。
        # 典型工作包括：
        # - 根据 desc_act 处理 g_idx / 排序索引
        # - repack qweight
        # - 重排 scales
        # - 必要时同步 bias 布局

        # ----------------- 基础输入检查 -----------------
        # 标记当前输入是否为 8-bit 激活类型。
        # itemsize==1 时，常见就是 int8 / fp8 这类单字节激活。
        is_a_8bit = self.input_dtype is not None and self.input_dtype.itemsize == 1

        # Marlin MoE 不支持 W8A8-INT8。
        if is_a_8bit:
            assert self.quant_type == scalar_types.uint4b8, (
                "W8A8-INT8 is not supported by marlin kernel."
            )

        # ----------------- 特殊输入分支：FP8 激活预处理 -----------------
        # FP8 输入场景下，需要先对 packed qweight 做额外预处理，并同步缩放 scales。
        # 这里的预处理不是重新量化，而是把现有 int4 packed 权重整理成
        # Marlin-FP8 混合路径更适合直接消费的表示。
        if self.input_dtype == torch.float8_e4m3fn:
            # gate/up 路径的 packed qweight 预处理。
            ops.marlin_int4_fp8_preprocess(layer.w13_qweight, inplace=True)
            # down_proj 路径的 packed qweight 预处理。
            ops.marlin_int4_fp8_preprocess(layer.w2_qweight, inplace=True)
            # FP8 路径要求 scales 额外乘一个固定缩放因子。
            layer.w13_scales.data = layer.w13_scales.data * 512
            layer.w2_scales.data = layer.w2_scales.data * 512

        # ----------------- g_idx / act-order 后处理 -----------------
        if self.quant_config.desc_act:
            # desc_act=true 时，运行时需要按 g_idx 的排序结果访问权重。
            # 因此这里先为每个 expert 计算排序索引，再把 g_idx 自身重排成排序后视图。
            num_experts = layer.w13_g_idx.shape[0]
            # 保存“排序后位置 -> 原始位置”的索引。
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_g_idx)
            # 保存按索引重排后的 g_idx，供后续 repack 直接使用。
            w13_sorted_g_idx = torch.empty_like(layer.w13_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_g_idx)
            for e in range(num_experts):
                # 分别为每个 expert 计算 gate/up 与 down_proj 的 g_idx 排序次序。
                w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_g_idx[e]).to(
                    torch.int32
                )
                w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_g_idx[e]).to(
                    torch.int32
                )
                # 把原始 g_idx 重排成单调有序视图，后续 kernel/repack 会直接依赖它。
                w13_sorted_g_idx[e] = layer.w13_g_idx[e][w13_g_idx_sort_indices[e]]
                w2_sorted_g_idx[e] = layer.w2_g_idx[e][w2_g_idx_sort_indices[e]]
            # 用 replace_parameter 覆盖原参数，保留 weight_loader 等运行时属性。
            replace_parameter(layer, "w13_g_idx", w13_sorted_g_idx)
            replace_parameter(layer, "w2_g_idx", w2_sorted_g_idx)
            replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
            replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        else:
            # desc_act=false 时，g_idx 相关张量在运行时无实际用途。
            # 当前 Qwen3.5-122B-A10B-GPTQ-Int4 就是这一路径：
            # - bits = 4
            # - group_size = 128
            # - desc_act = false
            # - sym = true
            # 因而 g_idx 虽然会先按统一接口加载，但这里会直接缩成空张量释放运行时负担。
            num_experts = layer.w13_g_idx.shape[0]
            device = layer.w13_g_idx.device
            # 仍然保留参数名，但把实体缩成 [num_experts, 0] 的空张量。
            layer.w13_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )

        # ----------------- qweight repack -----------------
        # 把 checkpoint 里的 packed qweight 重排成 Marlin kernel 直接消费的布局。
        # 这里传给 repack 的 size_k / size_n 对应“解包前的逻辑矩阵尺寸”，
        # 而不是当前 packed tensor 的物理 shape。
        marlin_w13_qweight = ops.gptq_marlin_moe_repack(
            layer.w13_qweight,
            # gate/up 路径若开启 desc_act，会用到事先准备好的排序索引。
            layer.w13_g_idx_sort_indices,
            # packed 之前的 K 维大小。
            layer.w13_qweight.shape[1] * self.quant_config.pack_factor,
            # 逻辑 N 维大小。
            layer.w13_qweight.shape[2],
            self.quant_config.quant_type.size_bits,
            is_a_8bit=is_a_8bit,
        )
        # 用 repack 后的参数替换原始 checkpoint 布局参数。
        replace_parameter(layer, "w13_qweight", marlin_w13_qweight)

        # down_proj 的 qweight 同样做对应的 repack。
        marlin_w2_qweight = ops.gptq_marlin_moe_repack(
            layer.w2_qweight,
            layer.w2_g_idx_sort_indices,
            # w2 的 K 维在存储上同样是按 pack_factor 压缩过的。
            layer.w2_qweight.shape[1] * self.quant_config.pack_factor,
            layer.w2_qweight.shape[2],
            self.quant_config.quant_type.size_bits,
            is_a_8bit=is_a_8bit,
        )
        replace_parameter(layer, "w2_qweight", marlin_w2_qweight)

        # ----------------- modular kernel 兼容别名 -----------------
        # 给 modular kernel 暴露统一的 w13_weight / w2_weight 访问名。
        layer.w13_weight = layer.w13_qweight
        layer.w2_weight = layer.w2_qweight

        # ----------------- scales 重排与输入 scale 提取 -----------------
        # 把 w13_scales 重排成 Marlin 期望布局。
        marlin_w13_scales = marlin_moe_permute_scales(
            s=layer.w13_scales,
            # w13 的逻辑 K 维就是当前 TP 分片后的 intermediate_size。
            size_k=layer.intermediate_size_per_partition,
            # gate/up 融合后，逻辑 N 维直接取 scales 最后一维。
            size_n=layer.w13_scales.shape[2],
            group_size=self.quant_config.group_size,
            is_a_8bit=is_a_8bit,
        )
        # INT8 输入且存在多组 scales 时，需要额外提取全局输入 scale。
        if self.input_dtype == torch.int8 and layer.num_groups_w13 > 1:
            marlin_w13_scales, w13_input_global_scale = marlin_act_int8_process_scales(
                marlin_w13_scales
            )
            layer.register_parameter(
                "w13_input_global_scale",
                torch.nn.Parameter(w13_input_global_scale, requires_grad=False),
            )

        # 用重排后的 scales 替换原始 checkpoint 布局。
        replace_parameter(layer, "w13_scales", marlin_w13_scales)

        # down_proj 的 scales 也要按自己的 K/N 语义重排。
        marlin_w2_scales = marlin_moe_permute_scales(
            s=layer.w2_scales,
            # group 量化时，K 维要从“group 数”还原回逻辑输入维；
            # channel 量化时则退回按 pack_factor 恢复。
            size_k=layer.w2_scales.shape[1]
                   * (
                       self.quant_config.group_size
                       if self.quant_config.group_size != -1
                       else self.quant_config.pack_factor
                   ),
            size_n=layer.w2_scales.shape[2],
            group_size=self.quant_config.group_size,
            is_a_8bit=is_a_8bit,
        )
        # INT8 输入且存在多组 scales 时，同样提取 w2 的全局输入 scale。
        if self.input_dtype == torch.int8 and layer.num_groups_w2 > 1:
            marlin_w2_scales, w2_input_global_scale = marlin_act_int8_process_scales(
                marlin_w2_scales
            )
            layer.register_parameter(
                "w2_input_global_scale",
                torch.nn.Parameter(w2_input_global_scale, requires_grad=False),
            )

        # 用重排后的 w2 scales 替换原始参数。
        replace_parameter(layer, "w2_scales", marlin_w2_scales)

        # ----------------- bias 重排 -----------------
        # 若存在 bias，也同步转成 Marlin 期望布局。
        if hasattr(layer, "w13_bias") and layer.w13_bias is not None:
            # w13_bias 对应 gate/up 融合后的偏置。
            layer.w13_bias.data = marlin_permute_bias(layer.w13_bias)

        if hasattr(layer, "w2_bias") and layer.w2_bias is not None:
            # w2_bias 对应 down_proj 偏置。
            layer.w2_bias.data = marlin_permute_bias(layer.w2_bias)

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        # 延迟导入 FusedMoE 侧的量化配置构造器，避免无关路径增加依赖。
        from cfie.model_executor.layers.fused_moe.config import (
            gptq_marlin_moe_quant_config,
        )

        # 把 layer 上已经整理好的量化张量打包成 FusedMoE 运行时量化配置对象。
        return gptq_marlin_moe_quant_config(
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            weight_bits=self.quant_config.weight_bits,
            group_size=self.quant_config.group_size,
            w1_zp=getattr(layer, "w13_qzeros", None)
            if not self.quant_config.is_sym
            else None,
            w2_zp=getattr(layer, "w2_qzeros", None)
            if not self.quant_config.is_sym
            else None,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
        )

    def select_gemm_impl(
            self,
            prepare_finalize,
            layer: torch.nn.Module,
    ):
        # modular kernel 只服务于 LoRA 场景。
        if not self.moe.is_lora_enabled:
            raise NotImplementedError(
                "GPTQ-Marlin uses its own apply() method when LoRA is not enabled. "
                "Modular kernels are only used for LoRA support."
            )

        # modular 版 Marlin experts 当前不支持 8-bit 权重。
        if self.quant_config.weight_bits == 8:
            raise NotImplementedError(
                "GPTQ-Marlin kernel does not support 8-bit weights."
            )

        # 延迟导入 modular kernel 枚举和 Marlin expert 实现。
        from cfie.model_executor.layers.fused_moe import modular_kernel as mk
        from cfie.model_executor.layers.fused_moe.fused_marlin_moe import (
            BatchedMarlinExperts,
            MarlinExperts,
        )

        # 调用这里前，外层必须已经准备好 MoE 运行时量化配置。
        assert self.moe_quant_config is not None, (
            "moe_quant_config must be initialized before select_gemm_impl"
        )

        # desc_act=true 时保留 g_idx。
        w13_g_idx = (
            getattr(layer, "w13_g_idx", None) if self.quant_config.desc_act else None
        )
        # desc_act=true 时保留 g_idx。
        w2_g_idx = (
            getattr(layer, "w2_g_idx", None) if self.quant_config.desc_act else None
        )
        # desc_act=true 时保留排序索引。
        w13_g_idx_sort_indices = (
            getattr(layer, "w13_g_idx_sort_indices", None)
            if self.quant_config.desc_act
            else None
        )
        # desc_act=true 时保留排序索引。
        w2_g_idx_sort_indices = (
            getattr(layer, "w2_g_idx_sort_indices", None)
            if self.quant_config.desc_act
            else None
        )

        # BatchedExperts 格式通常用于 EP 场景。
        if (
                prepare_finalize.activation_format
                == mk.FusedMoEActivationFormat.BatchedExperts
        ):
            # batched 格式下，需要提供每个 rank 的最大 token 数上限。
            max_num_tokens_per_rank = prepare_finalize.max_num_tokens_per_rank()
            assert max_num_tokens_per_rank is not None
            return BatchedMarlinExperts(
                max_num_tokens=max_num_tokens_per_rank,
                num_dispatchers=prepare_finalize.num_dispatchers(),
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                w13_g_idx=w13_g_idx,
                w2_g_idx=w2_g_idx,
                w13_g_idx_sort_indices=w13_g_idx_sort_indices,
                w2_g_idx_sort_indices=w2_g_idx_sort_indices,
                is_k_full=self.is_k_full,
            )
        else:
            # 非 batched 格式直接使用标准 Marlin experts。
            return MarlinExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                w13_g_idx=w13_g_idx,
                w2_g_idx=w2_g_idx,
                w13_g_idx_sort_indices=w13_g_idx_sort_indices,
                w2_g_idx_sort_indices=w2_g_idx_sort_indices,
                is_k_full=self.is_k_full,
            )

    def apply(
            self,
            layer: FusedMoE,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 直接调用 fused Marlin MoE 主 kernel。
        return fused_marlin_moe(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            getattr(layer, "w13_bias", None),
            getattr(layer, "w2_bias", None),
            layer.w13_scales,
            layer.w2_scales,
            topk_weights,
            topk_ids,
            input_global_scale1=getattr(layer, "w13_input_global_scale", None),
            input_global_scale2=getattr(layer, "w2_input_global_scale", None),
            quant_type_id=self.quant_type.id,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            g_idx1=layer.w13_g_idx,
            g_idx2=layer.w2_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            workspace=layer.workspace,
            is_k_full=self.is_k_full,
            input_dtype=self.input_dtype,
            inplace=not self.moe.disable_inplace,
        )
