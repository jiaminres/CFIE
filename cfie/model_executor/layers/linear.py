# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from abc import abstractmethod

import torch
from torch.nn.parameter import Parameter, UninitializedParameter

from cfie.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from cfie.logger import init_logger
from cfie.model_executor.custom_op import PluggableLayer
from cfie.model_executor.layers.batch_invariant import (
    linear_batch_invariant,
    cfie_is_batch_invariant,
)
from cfie.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from cfie.model_executor.layers.utils import (
    dispatch_unquantized_gemm,
)
from cfie.model_executor.parameter import (
    BasevLLMParameter,
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    PerTensorScaleParameter,
    RowvLLMParameter,
)
from cfie.model_executor.utils import set_weight_attrs
from cfie.platforms import current_platform

logger = init_logger(__name__)

WEIGHT_LOADER_V2_SUPPORTED = [
    "UnquantizedLinearMethod",
    "CompressedTensorsLinearMethod",
    "CompressedTensorsLinearTransformMethod",
    "AWQMarlinLinearMethod",
    "AWQLinearMethod",
    "GPTQMarlinLinearMethod",
    "Fp8LinearMethod",
    "MarlinLinearMethod",
    "GPTQMarlin24LinearMethod",
    "TPUInt8LinearMethod",
    "GPTQLinearMethod",
    "FBGEMMFp8LinearMethod",
    "ModelOptFp8LinearMethod",
    "ModelOptFp8PcPtLinearMethod",
    "ModelOptFp8PbWoLinearMethod",
    "QuarkLinearMethod",
    "ModelOptNvFp4LinearMethod",
    "PetitNvFp4LinearMethod",
]


def register_weight_loader_v2_supported_method(cls):
    """Decorator to register a LinearMethod as supporting weight_loader_v2."""
    WEIGHT_LOADER_V2_SUPPORTED.append(cls.__name__)
    return cls


def adjust_marlin_shard(
        param: Parameter,
        shard_size: int,
        shard_offset: int,
) -> tuple[int, int]:
    marlin_tile_size: int | None = getattr(param, "marlin_tile_size", None)
    if marlin_tile_size is None:
        return shard_size, shard_offset

    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size


def adjust_block_scale_shard(
        weight_block_size: tuple[int, ...] | None,
        shard_size: int,
        shard_offset: int,
) -> tuple[int, int]:
    assert weight_block_size is not None
    block_n = weight_block_size[0]
    shard_offset = (shard_offset + block_n - 1) // block_n
    shard_size = (shard_size + block_n - 1) // block_n
    return shard_size, shard_offset


def adjust_bitsandbytes_4bit_shard(
        param: Parameter,
        shard_offsets: dict[str, tuple[int, int]],
        loaded_shard_id: str,
) -> tuple[int, int]:
    """Adjust the quantization offsets and sizes for BitsAndBytes sharding."""

    total, _ = shard_offsets["total"]
    orig_offset, orig_size = shard_offsets[loaded_shard_id]

    quantized_total = param.data.shape[0]
    quantized_offset = orig_offset * quantized_total // total
    quantized_size = orig_size * quantized_total // total

    return quantized_size, quantized_offset


def adjust_scalar_to_fused_array(
        param_data: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_id: int | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For fused modules (QKV and MLP) we have an array of length
    N that holds 1 scale for each "logical" matrix. So the param
    is an array of length N. The loaded_weight corresponds to
    one of the shards on disk. Here, we slice the param based on
    the shard_id for loading.
    """
    qkv_idxs = {"q": 0, "k": 1, "v": 2}

    if isinstance(shard_id, str):
        shard_id = qkv_idxs[shard_id]
    elif not isinstance(shard_id, int):
        raise ValueError(f"Unknown Shard Id {shard_id}")

    # AutoFP8 scales do not have a shape
    # compressed-tensors scales do have a shape
    if len(loaded_weight.shape) != 0:
        assert loaded_weight.shape[0] == 1
        loaded_weight = loaded_weight[0]

    return param_data[shard_id], loaded_weight


class LinearMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def create_weights(
            self,
            layer: torch.nn.Module,
            input_size_per_partition: int,
            output_partition_sizes: list[int],
            input_size: int,
            output_size: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs,
    ):
        """Create weights for a linear layer.
           The weights will be set as attributes of the layer.

        Args:
            layer: The layer that is using the LinearMethodBase factory.
            input_size_per_partition: Size of the weight input dim on rank X.
            output_partition_sizes: Sizes of the output dim of each logical
                weight on rank X. E.g., output_partition_sizes for QKVLinear
                is a list contains the width of Wq, Wk, Wv on rank X.
            input_size: Size of the input dim of the weight across all ranks.
            output_size: Size of the output dim of the weight across all ranks.
            params_dtype: Datatype of the parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
            self,
            layer: torch.nn.Module,
            input_size_per_partition: int,
            output_partition_sizes: list[int],
            input_size: int,
            output_size: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs,
    ):
        # This method creates unquantized linear weights.
        # The weights are not quantized, and they are not sharded.
        # The amount of memory allocated for the weights is
        # sum(output_partition_sizes) * input_size_per_partition.
        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if current_platform.is_cpu():
            from cfie.model_executor.layers.utils import dispatch_cpu_unquantized_gemm

            dispatch_cpu_unquantized_gemm(layer, remove_weight=True)

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cfie_is_batch_invariant() and current_platform.is_cuda_alike():
            return linear_batch_invariant(x, layer.weight, bias)
        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)


class LinearBase(PluggableLayer):
    """
    线性层基类。

    参数说明：
        input_size: 线性层输入维度。
        output_size: 线性层输出维度。
        skip_bias_add: 若为 True，则前向时不直接把 bias 加到输出上，而是由调用方后处理。
        params_dtype: 参数数据类型。
        quant_config: 量化配置对象。
        prefix: 参数名前缀，用于权重加载和 state dict 对齐。
        return_bias: 若为 True，则前向时允许把 bias 一并返回。
        disable_tp: 若为 True，则对当前层禁用 tensor parallel 语义。
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            skip_bias_add: bool = False,
            params_dtype: torch.dtype | None = None,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
            *,
            return_bias: bool = True,
            disable_tp: bool = False,
    ):
        # 先初始化 PluggableLayer / nn.Module 基类状态。
        super().__init__()

        # -------------------- 记录基础线性层超参数 --------------------
        # 保存输入维度。
        self.input_size = input_size
        # 保存输出维度。
        self.output_size = output_size
        # 保存“是否跳过 bias 加法”的行为开关。
        self.skip_bias_add = skip_bias_add
        # 若外部未显式指定参数 dtype，则退化到当前默认浮点类型。
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        # 记录最终生效的参数 dtype。
        self.params_dtype = params_dtype
        # 保存量化配置对象，后续具体线性层实现会据此选择 quant_method。
        self.quant_config = quant_config
        # 保存参数名前缀。
        self.prefix = prefix
        # 默认不允许 FP8 block shape mismatch，某些子类会按需改成 True。
        self.allow_fp8_block_shape_mismatch = False
        # -------------------- 决定当前层使用的 quant_method --------------------
        # 未开启量化时，默认走非量化线性方法。
        if quant_config is None:
            self.quant_method: QuantizeMethodBase | None = UnquantizedLinearMethod()
        else:
            # 开启量化时，从 quant_config 中为当前层选择具体量化实现。
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
        # 保存前向是否返回 bias 的行为开关。
        self.return_bias = return_bias
        # 保存是否禁用 TP 的开关。
        self.disable_tp = disable_tp
        # -------------------- 解析当前层看到的 TP rank / TP size --------------------
        # 若未禁用 TP，则从全局并行状态读取当前 TP rank；否则退化成单 rank 视角。
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        # 若未禁用 TP，则读取 TP world size；否则视为 1。
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1

    def update_param_tp_status(self):
        # 遍历当前层注册的全部参数。
        for param in self.parameters():
            # 只有 BasevLLMParameter 体系的参数才会携带 tp_rank/tp_size 这些额外元信息。
            if isinstance(param, BasevLLMParameter):
                # 把当前层解析出的 TP rank 写回参数对象。
                param.tp_rank = self.tp_rank
                # 把当前层解析出的 TP world size 写回参数对象。
                param.tp_size = self.tp_size


# --8<-- [start:replicated_linear]
@PluggableLayer.register("replicated_linear")
class ReplicatedLinear(LinearBase):
    """
    复制式线性层。

    参数说明：
        input_size: 线性层输入维度。
        output_size: 线性层输出维度。
        bias: 是否创建 bias 参数。
        skip_bias_add: 若为 True，则前向中不把 bias 加到输出上，而是单独返回。
        params_dtype: 参数数据类型。
        quant_config: 量化配置对象。
        prefix: 该层在 state dict 中的完整路径前缀，
            例如 `model.layers.0.qkv_proj`。
        return_bias: 前向时是否把 bias 一并返回。
        disable_tp: 对 replicated linear 无实际作用，仅为接口兼容保留。
    """

    # --8<-- [end:replicated_linear]

    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = True,
            skip_bias_add: bool = False,
            params_dtype: torch.dtype | None = None,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
            *,
            return_bias: bool = True,
            disable_tp: bool = False,
    ):
        # -------------------- 先解析输出分片信息 --------------------
        # 若当前实例其实是 MergedReplicatedLinear 这类融合变体，则直接复用其预先定义的输出分片尺寸。
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = self.output_sizes
        else:
            # 普通 ReplicatedLinear 没有逻辑分片，整个输出维只视为一个分片。
            self.output_partition_sizes = [output_size]

        # -------------------- 调用 LinearBase 完成通用线性层初始化 --------------------
        # 基类会负责记录 input/output size、选择 quant_method、注册通用属性等。
        super().__init__(
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        # -------------------- 创建主权重参数 --------------------
        # ReplicatedLinear 必须已经拿到某个 quant_method；无论是否真的量化，都统一通过它创建权重。
        assert self.quant_method is not None
        self.quant_method.create_weights(
            # 当前线性层实例，供 quant_method 在其上注册参数。
            self,
            # replicated linear 的每个 rank 都持有完整输入维，因此 input_size_per_partition 等于 input_size。
            self.input_size,
            # 输出分片尺寸列表；普通层只有一个分片，融合层可能有多个逻辑分片。
            self.output_partition_sizes,
            # 全局输入维度。
            self.input_size,
            # 全局输出维度。
            self.output_size,
            # 权重参数的数据类型。
            self.params_dtype,
            # 指定统一的权重加载回调，所有主权重都通过本类的 weight_loader 落盘。
            weight_loader=self.weight_loader,
        )

        # -------------------- 按需创建 bias 参数 --------------------
        if bias:
            # bias 的长度等于完整输出维度，因为 replicated linear 不做输出切分。
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=self.params_dtype)
            )
            # 给 bias 挂上额外元数据，便于统一的权重加载逻辑识别其输出维与 loader。
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            # 若显式关闭 bias，则注册一个值为 None 的占位参数名，保持模块接口一致。
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # -------------------- 先处理 GGUF 特殊元数据 --------------------
        # 某些 GGUF 参数不是普通权重张量，而是附带类型信息或延迟 materialize 的占位参数。

        # 标记当前参数是否来自 GGUF 权重。
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        # 标记当前参数是否实际上承载的是 GGUF 的 weight_type 元信息。
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            # 这类参数不拷贝整张量，而是把磁盘里的标量值记到 param.weight_type 上。
            param.weight_type = loaded_weight.item()

        # -------------------- 按需把 GGUF 的未初始化参数实体化 --------------------
        # GGUF 路径里某些参数一开始是 UninitializedParameter，需要先按实际 shape/dtype materialize。
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            param.materialize(loaded_weight.shape, dtype=loaded_weight.dtype)

        # -------------------- 统一标量权重的形状 --------------------
        # 某些磁盘权重本身是 0 维标量（例如 AutoFP8 的 scale），这里统一 reshape 成长度为 1 的张量。
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        # -------------------- 校验形状并拷贝权重 --------------------
        # 只有目标参数形状与磁盘权重完全一致时，才允许执行 data copy。
        assert param.size() == loaded_weight.size(), (
            f"Tried to load weights of size {loaded_weight.size()}"
            f"to a parameter of size {param.size()}"
        )
        # 把磁盘权重直接拷贝到参数存储中。
        param.data.copy_(loaded_weight)

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # -------------------- 先根据 skip_bias_add 决定本轮是否在算子内加 bias --------------------
        # 若 skip_bias_add=True，则把 bias 置空，交给调用方后处理；否则在 quant_method.apply 内直接加上。
        bias = self.bias if not self.skip_bias_add else None
        # ReplicatedLinear 的前向一定依赖 quant_method 统一执行 GEMM/量化路径。
        assert self.quant_method is not None

        # -------------------- 调用 quant_method 执行线性变换 --------------------
        # quant_method 会根据实际实现选择普通 matmul 或量化 kernel。
        output = self.quant_method.apply(self, x, bias)

        # -------------------- 按配置决定返回值形态 --------------------
        # 若调用方不需要 bias，则直接返回输出张量本身。
        if not self.return_bias:
            return output
        # 只有 skip_bias_add=True 时，才需要把原始 bias 单独返回给上游手动相加。
        output_bias = self.bias if self.skip_bias_add else None
        # 返回 (output, output_bias) 二元组，保持与 LinearBase 家族其它实现一致。
        return output, output_bias

    def extra_repr(self) -> str:
        # 先输出输入特征维度。
        s = f"in_features={self.input_size}"
        # 再补充输出特征维度。
        s += f", output_features={self.output_size}"
        # 最后标记当前是否带 bias。
        s += f", bias={self.bias is not None}"
        # 返回给 nn.Module.__repr__ 使用的附加字符串。
        return s


# --8<-- [start:column_parallel_linear]
@PluggableLayer.register("column_parallel_linear")
class ColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism.

    数学定义:
        Y = X A + b

    其中:
    - X 的最后一维是 input_size
    - A 的形状可理解为 [input_size, output_size]
    - Y 的最后一维是 output_size

    “Column parallelism” 的意思是:
    - 沿 A 的“输出维”切分，也就是按列块切分
    - 若 tp_size = p，则:
          A = [A_1, A_2, ..., A_p]
      每个 rank 只持有其中一块 A_i

    所以每个 TP rank 本地计算的是:
          Y_i = X A_i
    本地输出 shape:
          [..., output_size / tp_size]

    若 gather_output=True:
    - 会把所有 rank 的局部输出 all-gather，得到完整输出 [..., output_size]

    若 gather_output=False:
    - 每个 rank 只保留自己那一份局部输出 [..., output_size_per_partition]

    这也是为什么前面的 QKVParallelLinear 传入“全局 output_size”，
    但 forward 实际返回的却是当前 rank 的 local shard。
    """

    def __init__(
            self,
            input_size: int,                      # 输入维，记作 C_in
            output_size: int,                     # 全局逻辑输出维，记作 C_out_global
            bias: bool = True,
            gather_output: bool = False,          # 是否把各 rank 的局部输出 all-gather 成完整输出
            skip_bias_add: bool = False,          # 若为 True，forward 不把 bias 加到输出里，而是单独返回 bias
            params_dtype: torch.dtype | None = None,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
            *,
            return_bias: bool = True,             # forward 是否返回 bias（常与 skip_bias_add 配合）
            disable_tp: bool = False,             # 若为 True，则不做 TP 切分
    ):
        # ------------------------------------------------------------
        # 1) 确定当前 TP rank 和 TP world size
        # ------------------------------------------------------------
        # 若 disable_tp=True，则逻辑上退化成单卡:
        #   tp_rank = 0
        #   tp_size = 1
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1

        # ------------------------------------------------------------
        # 2) 计算本地分片的输入/输出维
        # ------------------------------------------------------------
        # ColumnParallel 的特点:
        # - 输入维不切分，所有 rank 都看到完整 input_size
        # - 输出维沿 TP 切分
        #
        # 因此:
        #   input_size_per_partition = input_size
        #   output_size_per_partition = output_size / tp_size
        #
        # 也就是说，本地权重块形状可理解为:
        #   [input_size, output_size_per_partition]
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)

        # 默认只有一个输出分块，即整个 local output 一块
        # 例如普通 ColumnParallelLinear:
        #   output_partition_sizes = [output_size_per_partition]
        self.output_partition_sizes = [self.output_size_per_partition]

        # ------------------------------------------------------------
        # 3) 特殊情况：如果是 QKVParallelLinear / MergedColumnParallelLinear
        # ------------------------------------------------------------
        #         # 这些子类会在 super().__init__ 前提前设置 self.output_sizes，
        # 表示逻辑上输出由多段拼接而成，比如:
        #   output_sizes = [q_out, k_out, v_out]
        #
        # 那么这里就把每一段都各自按 TP 切开，得到每段 local size:
        #   [q_out/tp_size, k_out/tp_size, v_out/tp_size]
        #
        # 这正是前面 QKVParallelLinear 能够维护
        # “本地 [q_local | k_local | v_local] 拼接布局”的基础。
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size) for output_size in self.output_sizes
            ]

        # ------------------------------------------------------------
        # 4) 调用 LinearBase 初始化
        # ------------------------------------------------------------
        # 注意这里传给父类的是“全局 input_size / 全局 output_size”
        # 但当前层自己已经记录了 local partition size。
        #
        # 后面真正创建权重时，会通过 quant_method.create_weights(...)
        # 使用 input_size_per_partition / output_partition_sizes 来创建本地参数。
        super().__init__(
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        # 某些 FP8 block quant 场景下，block 大小可能与输出分片大小不整除
        # 这里允许这种 mismatch，避免某些 block layout 校验直接失败
        self._maybe_allow_fp8_block_shape_mismatch()

        # 是否在 forward 后把所有 TP rank 的局部输出 all-gather 成完整输出
        self.gather_output = gather_output

        # quant_method 是由 LinearBase / quant_config 决定的权重量化/应用策略对象
        assert self.quant_method is not None

        # ------------------------------------------------------------
        # 5) 创建本地权重参数
        # ------------------------------------------------------------
        # 这里非常关键：
        # create_weights(...) 并不是创建一个“全局完整权重”，
        # 而是根据当前 partition 信息创建“当前 rank 的本地权重”
        #
        # 传入的关键信息:
        # - input_size_per_partition = 完整 input_size
        # - output_partition_sizes   = 当前 rank 各输出分块的 local 大小
        #
        # 对普通 ColumnParallelLinear:
        #   output_partition_sizes = [output_size_per_partition]
        #
        # 对 QKVParallelLinear:
        #   output_partition_sizes = [q_local_dim, k_local_dim, v_local_dim]
        #
        # weight_loader 会绑定成:
        # - 新版量化方法支持时 -> self.weight_loader_v2
        # - 否则               -> self.weight_loader
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )

        # ------------------------------------------------------------
        # 6) 创建 bias
        # ------------------------------------------------------------
        # 因为这是 column parallel，所以 bias 也只为当前 rank 创建本地那一段:
        #   shape = [output_size_per_partition]
        #
        # 也就是说:
        # - 如果 gather_output=False，则这个 local bias 直接对应 local output
        # - 如果 gather_output=True，则 all-gather 后完整输出的 bias 实际是由各 rank
        #   的 local bias 拼起来构成的
        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition, dtype=params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,                  # bias 的切分维就是它唯一那一维
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

        # 给参数补充 TP 状态标记（例如参数是否是分片参数）
        self.update_param_tp_status()

    def _maybe_allow_fp8_block_shape_mismatch(self) -> None:
        # 这个函数只处理一种很特定的情况:
        # FP8 block quant 时，block 大小要求输出维按 block_n 对齐，
        # 但如果当前各 local partition size 有的不能整除 block_n，
        # 就允许这种 mismatch。
        quant_config = getattr(self, "quant_config", None)
        weight_block = getattr(quant_config, "weight_block_size", None)

        # 若没有 block quant 信息，或者只有单一输出分块，则无需处理
        if (
                weight_block is None
                or len(weight_block) < 1
                or len(self.output_partition_sizes) <= 1
        ):
            return

        try:
            block_n = int(weight_block[0])
        except (ValueError, TypeError):
            return

        if block_n <= 0:
            return

        # 如果任意一个 local output 分块大小不能被 block_n 整除，
        # 就允许 FP8 block shape mismatch
        if any(size % block_n != 0 for size in self.output_partition_sizes):
            self.allow_fp8_block_shape_mismatch = True
            logger.debug(
                "Allowing FP8 block shape mismatch for %s (block_n=%d, partitions=%s)",
                getattr(self, "prefix", "<unknown>"),
                block_n,
                self.output_partition_sizes,
            )

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # ------------------------------------------------------------
        # 旧版/通用权重加载
        # ------------------------------------------------------------
        # 对于 ColumnParallelLinear，核心逻辑很简单:
        # - 若 loaded_weight 是全量输出维权重，就按 tp_rank 沿 output_dim 切出本地那一段
        # - 若 loaded_weight 已经是当前 rank 对应的 sharded weight，就直接 copy
        output_dim = getattr(param, "output_dim", None)

        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

        # bitsandbytes 4bit 常常已经直接提供本地那一份权重，不需要再次 narrow
        is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

        # ------------------------------------------------------------
        # GGUF 特殊情况
        # ------------------------------------------------------------
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)

        # 若当前 param 存的是 GGUF 的“权重量化类型元信息”
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # 若是 GGUF 权重，且 param 还是未初始化参数，
        # 则先按当前 TP 分片大小 materialize 出本地参数形状
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            final_shape = list(loaded_weight.shape)

            # 若 output_dim 存在，则沿该维按 tp_size 切分
            if output_dim is not None:
                assert final_shape[output_dim] % self.tp_size == 0
                final_shape[output_dim] = final_shape[output_dim] // self.tp_size

            # 这里 materialize 出的是当前 rank 的本地参数形状
            param.materialize(final_shape, dtype=loaded_weight.dtype)

        # param_data 是目标参数实际存储张量（本地 shard）
        param_data = param.data

        # ------------------------------------------------------------
        # 标准 ColumnParallel 切分逻辑
        # ------------------------------------------------------------
        # 若:
        # - output_dim 存在（说明知道沿哪个维度切）
        # - loaded_weight 还不是预先分好的 local shard
        #
        # 则按 tp_rank 从 loaded_weight 沿 output_dim 切出当前 rank 本地那一段
        if output_dim is not None and not is_sharded_weight:
            # 当前 param_data 在 output_dim 上的大小，就是本地 shard_size
            shard_size = param_data.shape[output_dim]

            # 当前 rank 对应这段在全量 loaded_weight 上的起始位置
            start_idx = self.tp_rank * shard_size

            # 从全量 loaded_weight 中 narrow 出当前 rank 本地 shard
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        # ------------------------------------------------------------
        # 特殊情况：某些 scale 从磁盘上读出来是标量（0-d tensor）
        # ------------------------------------------------------------
        # 例如 AutoFP8 的某些 scale 参数，没有形状，需要 reshape 成 [1]
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        # 最终本地参数块和当前本地 loaded_weight 块形状必须一致
        assert param_data.shape == loaded_weight.shape

        # copy 到本地参数
        param_data.copy_(loaded_weight)

    def weight_loader_v2(self, param: BasevLLMParameter, loaded_weight: torch.Tensor):
        # ------------------------------------------------------------
        # 新版 loader
        # ------------------------------------------------------------
        # 把具体 column parallel 权重加载细节下沉给 param 自己处理:
        #   param.load_column_parallel_weight(...)
        #
        # 这里自己只处理一下“标量 scale -> [1]”这种情况
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)

        param.load_column_parallel_weight(loaded_weight=loaded_weight)

    def forward(
            self,
            input_,   # 输入张量，常见 shape: [..., input_size]
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # ------------------------------------------------------------
        # 1) 决定是否把 bias 直接传给 quant_method.apply
        # ------------------------------------------------------------
        # 若 skip_bias_add=False:
        #   bias 会直接参与线性层/量化kernel计算，输出里已经加上 bias
        #
        # 若 skip_bias_add=True:
        #   这里传入 bias=None，不在本层里做加法
        #   bias 会在最后作为 output_bias 单独返回，供外部 fuse
        bias = self.bias if not self.skip_bias_add else None

        # ------------------------------------------------------------
        # 2) 做矩阵乘法（可能是量化 kernel）
        # ------------------------------------------------------------
        # 这是当前 rank 的“本地分片线性层”计算:
        #
        # 若 input_ shape:
        #   [..., input_size]
        #
        # 则 output_parallel shape:
        #   [..., output_size_per_partition]
        #
        # 对 QKVParallelLinear 这类子类，更准确地说最后一维可能是:
        #   sum(output_partition_sizes)
        # 即本地 q_local / k_local / v_local 拼接后的维度
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, bias)

        # ------------------------------------------------------------
        # 3) 若需要 gather_output，则把各 rank 本地输出拼成完整输出
        # ------------------------------------------------------------
        if self.gather_output and self.tp_size > 1:
            # all-gather 后，输出最后一维恢复为全局 output_size
            #
            # 例如:
            #   output_parallel: [..., output_size_per_partition]
            # ->output:         [..., output_size]
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            # 否则直接返回本地输出
            # shape 保持为 [..., output_size_per_partition]
            output = output_parallel

        # ------------------------------------------------------------
        # 4) 按 return_bias / skip_bias_add 约定返回
        # ------------------------------------------------------------
        # return_bias=False:
        #   直接只返回 output
        if not self.return_bias:
            return output

        # 若 skip_bias_add=True，则把本地 bias 作为第二返回值吐出去
        # 否则 output_bias=None，因为 bias 已经加到 output 里了
        output_bias = self.bias if self.skip_bias_add else None

        # 返回:
        # - output:      [..., local_out_dim] 或 [..., global_out_dim]
        # - output_bias: [local_out_dim] 或 None
        return output, output_bias

    def extra_repr(self) -> str:
        # 打印模块时展示的信息
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size_per_partition}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", gather_output={self.gather_output}"
        return s


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Args:
        input_size: input dimension of the linear layer.
        output_sizes: list of output dimensions of the linear layer.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make the output
                       available to all GPUs, otherwise, every GPU will have
                       its own output.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, all weights matrix won't be sharded, this layer
                    will be treated as a "Replicated" MergedLinear.
    """

    def __init__(
            self,
            input_size: int,
            output_sizes: list[int],
            bias: bool = True,
            gather_output: bool = False,
            skip_bias_add: bool = False,
            params_dtype: torch.dtype | None = None,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
            *,
            return_bias: bool = True,
            disable_tp: bool = False,
    ):
        self.output_sizes = output_sizes
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0

        assert all(output_size % self.tp_size == 0 for output_size in output_sizes)
        super().__init__(
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def validate_shard_id(self, loaded_shard_id: int | tuple[int, ...] | None):
        if loaded_shard_id is None:
            return
        if isinstance(loaded_shard_id, tuple):
            for idx in loaded_shard_id:
                if not (0 <= idx < len(self.output_sizes)):
                    raise ValueError(
                        f"Shard id index {idx} should be between 0 and "
                        f"{len(self.output_sizes) - 1}. Got shard id {loaded_shard_id}."
                    )
            if len(loaded_shard_id) > 1 and any(
                    b - a != 1 for a, b in zip(loaded_shard_id[:-1], loaded_shard_id[1:])
            ):
                raise ValueError(
                    "Shard id with multiple indices should be consecutive. "
                    f"Got shard id {loaded_shard_id}."
                )
            return
        elif isinstance(loaded_shard_id, int):
            if loaded_shard_id < 0 or loaded_shard_id >= len(self.output_sizes):
                raise ValueError(
                    f"Shard id should be between 0 and {len(self.output_sizes) - 1}. "
                    f"Got shard id {loaded_shard_id}."
                )
            return
        raise ValueError("This line should not be reached")

    def weight_loader(
            self,
            param: Parameter,
            loaded_weight: torch.Tensor,
            loaded_shard_id: tuple[int, ...] | int | None = None,
    ):
        self.validate_shard_id(loaded_shard_id)
        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if isinstance(loaded_shard_id, tuple) and (
                is_gguf_weight or is_gguf_weight_type
        ):
            raise NotImplementedError(
                "Shard id with multiple indices is not supported for GGUF."
            )
        if is_gguf_weight_type:
            if loaded_shard_id is not None:
                param.data[loaded_shard_id].copy_(loaded_weight)
                param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            else:
                param.shard_weight_type = {
                    i: loaded_weight.item() for i, _ in enumerate(self.output_sizes)
                }
            return

        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)
            shard_size = loaded_weight.size(output_dim) // self.tp_size
            start_idx = self.tp_rank * shard_size

            if loaded_shard_id is not None:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
                param.shard_id.append(loaded_shard_id)
                param.shard_id_map[loaded_shard_id] = len(param.data_container)
                param.data_container.append(loaded_weight)
                return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for per-tensor scale to load scalar into fused array.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):
            # Loaded weight is already fused on disk (mlp).
            # (e.g., Phi-3's gate_up_proj).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0
                    )

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return

            output_sizes = (
                self.output_sizes[loaded_shard_id[0]: loaded_shard_id[-1] + 1]
                if loaded_shard_id is not None
                else self.output_sizes
            )
            current_shard_offset = 0
            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            if (
                    use_bitsandbytes_4bit
                    and isinstance(loaded_shard_id, tuple)
                    and self.tp_size > 1
            ):
                raise NotImplementedError(
                    "Shard id with multiple indices is not supported "
                    "for BNB quantization with TP yet."
                )
            shard_offsets: list[tuple[int, int, int]] = []
            for i, output_size in enumerate(output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantization.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.packed_factor
                    shard_offset = shard_offset // param.packed_factor
                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset
                    )

                if use_bitsandbytes_4bit:
                    index = list(itertools.accumulate([0] + self.output_sizes))
                    orig_offsets = {
                        str(i): (index[i], size)
                        for i, size in enumerate(self.output_sizes)
                    }
                    orig_offsets["total"] = (self.output_size, 0)
                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_offsets, str(shard_id)
                    )

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id < len(self.output_sizes)
        if output_dim is not None:
            shard_offset = sum(self.output_sizes[:loaded_shard_id])
            shard_size = self.output_sizes[loaded_shard_id]
            shard_offset //= self.tp_size
            shard_size //= self.tp_size

            if isinstance(param, BlockQuantScaleParameter):
                weight_block_size = getattr(self, "weight_block_size", None)
                shard_size, shard_offset = adjust_block_scale_shard(
                    weight_block_size, shard_size, shard_offset
                )

            # Special case for quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.packed_factor
                shard_offset = shard_offset // param.packed_factor
                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset
                )

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            # bitsandbytes loads the weights of the specific portion
            # no need to narrow
            is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

            if use_bitsandbytes_4bit:
                index = list(itertools.accumulate([0] + self.output_sizes))
                orig_offsets = {
                    str(i): (index[i], size) for i, size in enumerate(self.output_sizes)
                }
                orig_offsets["total"] = (self.output_size, 0)
                shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                    param, orig_offsets, str(loaded_shard_id)
                )
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            start_idx = self.tp_rank * shard_size
            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )

        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions."
                )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def _load_fused_module_from_checkpoint(
            self,
            param: BasevLLMParameter,
            loaded_weight: torch.Tensor,
            output_sizes: list[int] | None = None,
    ):
        """
        Handle special case for models where MLP layers are already
        fused on disk. In this case, we have no shard id. This function
        determines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """

        current_shard_offset = 0
        shard_offsets: list[tuple[int, int, int]] = []
        output_sizes = output_sizes or self.output_sizes
        for i, output_size in enumerate(output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size

        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            if (
                    isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
                    and param.packed_dim == param.output_dim
            ):
                shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset
                )

            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_offset, shard_size
            )
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(
            self,
            param: BasevLLMParameter,
            loaded_weight: torch.Tensor,
            loaded_shard_id: tuple[int, ...] | int | None = None,
    ):
        # MergedColumnParallelLinear 对应“按输出维拼起来的多个逻辑矩阵”。
        #
        # 当前 Qwen3.5 链路里，典型使用点有两类：
        # 1. 非量化：
        #    - shared_expert.gate_up_proj
        #    - self_attn.in_proj_qkvz / in_proj_ba
        # 2. 若某个 fused MLP/attention 模块未被 dynamic 排除量化，
        #    也可能承接 GPTQ qweight/scales/qzeros/g_idx 的加载
        #
        # 但对你当前 122B-A10B-GPTQ-Int4 checkpoint：
        # - shared_expert.* 被 -:.*shared_expert.* 排除量化
        # - attn.*          被 -:.*attn.* 排除量化
        # 所以这条 MergedColumnParallelLinear 路径当前更常见的是“非量化 BF16”加载。
        self.validate_shard_id(loaded_shard_id)
        if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):
            if isinstance(param, PerTensorScaleParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight, shard_id=0)
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight)
                return
            output_sizes = (
                [self.output_sizes[idx] for idx in loaded_shard_id]
                if loaded_shard_id
                else None
            )
            if isinstance(param, BlockQuantScaleParameter):
                weight_block_size = getattr(self, "weight_block_size", None)
                output_sizes = [
                    adjust_block_scale_shard(weight_block_size, size, 0)[0]
                    for size in (output_sizes or self.output_sizes)
                ]
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(
                param, loaded_weight, output_sizes=output_sizes
            )
            return

        assert loaded_shard_id < len(self.output_sizes)

        shard_offset = sum(self.output_sizes[:loaded_shard_id])
        shard_size = self.output_sizes[loaded_shard_id]
        shard_offset //= self.tp_size
        shard_size //= self.tp_size

        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        param.load_merged_column_weight(
            loaded_weight=loaded_weight,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
        )


class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    这个类本质上是一个“Q/K/V 融合的大线性层”：
    - 输入: hidden_states，最后一维是 hidden_size，常见 shape: [..., hidden_size]
    - 输出: 把 Q / K / V 三个投影沿“输出维”拼到一起，常见 shape: [..., q_dim + k_dim + v_dim]
    - 在 TP（tensor parallel）下，权重按输出维切分，所以每个 rank 只持有一部分输出列

    对普通 MHA:
    - total_num_heads == total_num_kv_heads
    - Q / K / V 头数相同

    对 GQA / MQA:
    - total_num_kv_heads < total_num_heads
    - Q 头数更多，K/V 头数更少
    - TP 下 K/V 头可能“复制”到多个 rank，而不是严格均分
    """

    def __init__(
            self,
            hidden_size: int,  # 输入隐藏维，记作 C_in
            head_size: int,  # 每个 Q/K 头的维度，记作 D_qk
            total_num_heads: int,  # 全局 Q 头数，记作 Hq_total
            total_num_kv_heads: int | None = None,  # 全局 K/V 头数，记作 Hkv_total；若为 None，则默认等于 Hq_total
            bias: bool = True,
            skip_bias_add: bool = False,
            params_dtype: torch.dtype | None = None,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
            *,
            return_bias: bool = True,
            disable_tp: bool = False,  # 若为 True，则本层内部不做 TP 切分
            v_head_size: int | None = None,  # 每个 V 头维度，记作 D_v；若为 None，则默认等于 D_qk
    ):
        # 保存输入/头维配置
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        # 上面允许 V 的 head dim 和 Q/K 不一样，但大多数模型里 D_v == D_qk

        self.total_num_heads = total_num_heads

        # 若未指定 total_num_kv_heads，则默认 K/V 头数 = Q 头数，即普通 MHA
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads

        # ------------------------------------------------------------
        # 1) 计算 TP world size
        # ------------------------------------------------------------
        # 若 disable_tp=True，则把 TP 视为关闭，tp_size=1
        # 否则从全局并行环境拿 TP world size
        #
        # tp_size 表示当前张量并行组里一共有多少个 rank
        # 后面 Q 头 / KV 头会基于它做切分或复制
        tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1

        # ------------------------------------------------------------
        # 2) 计算当前 rank 上本地持有多少个 Q 头
        # ------------------------------------------------------------
        # Q 头始终按 TP 严格等分
        #
        # Hq_local = Hq_total / tp_size
        self.num_heads = divide(self.total_num_heads, tp_size)

        # ------------------------------------------------------------
        # 3) 计算当前 rank 上本地持有多少个 KV 头，以及 KV 头复制倍数
        # ------------------------------------------------------------
        # 分两种情况：
        #
        # 情况 A: tp_size >= total_num_kv_heads
        #   说明 TP rank 比 KV 头还多
        #   这时每个 rank 最多只放 1 个 KV 头
        #   同一个“真实 KV 头”会被复制到多个 rank
        #
        #   例:
        #     Hkv_total = 2, tp_size = 8
        #   则:
        #     num_kv_heads = 1
        #     num_kv_head_replicas = 8 / 2 = 4
        #   含义:
        #     每个真实 KV 头被复制到 4 个 rank 上
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)

        # 情况 B: tp_size < total_num_kv_heads
        #   KV 头可以被正常切分
        #
        #   例:
        #     Hkv_total = 8, tp_size = 2
        #   则:
        #     num_kv_heads = 4
        #     num_kv_head_replicas = 1
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1

        # 输入维就是 hidden_size
        input_size = self.hidden_size  # = C_in

        # ------------------------------------------------------------
        # 4) 计算“全局逻辑输出维”
        # ------------------------------------------------------------
        # 当前 rank 本地 q/k/v 输出维分别是:
        #   q_local_dim = num_heads    * head_size
        #   k_local_dim = num_kv_heads * head_size
        #   v_local_dim = num_kv_heads * v_head_size
        #
        # 它们加起来，是当前 rank 持有的 local 输出维。
        # 再乘 tp_size，得到全局逻辑上的总输出维 output_size。
        #
        # 所以全局输出最后一维逻辑上是:
        #   Hq_total * D_qk + Hkv_total * D_qk + Hkv_total * D_v
        #
        # 对普通 MHA 且 D_v = D_qk = D:
        #   output_size = 3 * H * D
        output_size = (
                              self.num_heads * self.head_size
                              + self.num_kv_heads * self.head_size
                              + self.num_kv_heads * self.v_head_size
                      ) * tp_size

        # output_sizes 记录 q / k / v 这三段“全局逻辑输出维”的大小
        #
        # 对普通 MHA:
        #   [H*D, H*D, H*D]
        #
        # 对 GQA:
        #   [Hq_total*D_qk, Hkv_total*D_qk, Hkv_total*D_v]
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj 的全局输出维
            self.num_kv_heads * self.head_size * tp_size,  # k_proj 的全局输出维
            self.num_kv_heads * self.v_head_size * tp_size,  # v_proj 的全局输出维
        ]

        # ------------------------------------------------------------
        # 5) 调用父类 ColumnParallelLinear 初始化
        # ------------------------------------------------------------
        # ColumnParallelLinear 的特点:
        # - 权重按“输出维”切分
        # - 每个 TP rank 只持有输出列的一部分
        # - gather_output=False 表示 forward 后不自动把各 rank 输出 all-gather 成完整输出
        #
        # 因此 forward 时，当前 rank 实际返回的最后一维通常是:
        #   q_local_dim + k_local_dim + v_local_dim
        #
        # 而不是上面那个全局 output_size
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def validate_shard_id(self, loaded_shard_id: str | None):
        # 校验加载时的 shard_id
        #
        # 含义:
        # - None: 当前 loaded_weight 是 fused 的 qkv 整体
        # - "q": 当前 loaded_weight 对应 q 部分
        # - "k": 当前 loaded_weight 对应 k 部分
        # - "v": 当前 loaded_weight 对应 v 部分
        if loaded_shard_id is None:
            return
        if isinstance(loaded_shard_id, str):
            if loaded_shard_id not in ["q", "k", "v"]:
                raise ValueError(
                    "Shard id for QKVParallelLinear should be 'q', 'k', or 'v', "
                    f"got shard id {loaded_shard_id}."
                )
            return
        raise ValueError("This line should not be reached")

    def _get_shard_offset_mapping(self, loaded_shard_id: str):
        # 返回“当前 rank 本地 qkv 拼接布局”里，各 shard 的起始 offset
        #
        # 当前 rank 本地输出最后一维布局可以理解为:
        #   [ q_local | k_local | v_local ]
        #
        # 其中:
        #   q_local_dim = self.num_heads * self.head_size
        #   k_local_dim = self.num_kv_heads * self.head_size
        #   v_local_dim = self.num_kv_heads * self.v_head_size
        shard_offset_mapping = {
            "q": 0,
            "k": self.num_heads * self.head_size,
            "v": (self.num_heads + self.num_kv_heads) * self.head_size,
            "total": (self.num_heads + self.num_kv_heads) * self.head_size
                     + self.num_kv_heads * self.v_head_size,
        }
        return shard_offset_mapping.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str):
        # 返回当前 rank 本地 q / k / v 这三段的大小
        shard_size_mapping = {
            "q": self.num_heads * self.head_size,  # q_local_dim
            "k": self.num_kv_heads * self.head_size,  # k_local_dim
            "v": self.num_kv_heads * self.v_head_size,  # v_local_dim
        }
        return shard_size_mapping.get(loaded_shard_id)

    def _load_fused_module_from_checkpoint(
            self, param: BasevLLMParameter, loaded_weight: torch.Tensor
    ):
        """
        处理一种特殊情况:
        checkpoint 磁盘上存的已经是 fused qkv，而不是分开的 q / k / v。

        也就是 loaded_weight 在 output_dim 上长这样:
            [ q_full | k_full | v_full ]

        这里需要:
        1. 按全局 q/k/v 大小把 loaded_weight 切成三段
        2. 分别递归调用 weight_loader_v2(..., shard_id="q"/"k"/"v")

        当前 Qwen3.5-122B-A10B-GPTQ-Int4 的实际 checkpoint 一般不走这里：
        - 磁盘上通常是分开的
          `...self_attn.q_proj.weight / k_proj.weight / v_proj.weight`
        - 因此 load_weights 时更常直接带着
          `loaded_shard_id="q" / "k" / "v"` 进入下面的 Case 2
        - 这个 fused 分支主要是兼容“磁盘上已经把 qkv 合并存储”的模型
        """

        # 这里定义的是“磁盘上 fused qkv 权重”的全局 offset / size
        #
        # 注意这里用的是 total_num_heads / total_num_kv_heads，
        # 即全局头数，而不是当前 rank 的 local 头数
        #
        # 额外注意当前 Qwen3.5 full-attention 的特殊点：
        # - 配置里 num_attention_heads = 32
        # - 但 attn_output_gate = true
        # - 所以传给 QKVParallelLinear 的 total_num_heads 实际是 32 * (1 + 1) = 64
        #   这里的 q 段其实对应 [q | gate] 的融合输出
        #
        # 在当前 tp_size=1 下，fused qkv 的全局逻辑维度是：
        # - q: 64 * 256 = 16384
        # - k:  2 * 256 =   512
        # - v:  2 * 256 =   512
        # 合计 17408
        shard_offsets = [
            # (shard_id, shard_offset, shard_size)
            ("q", 0, self.total_num_heads * self.head_size),
            (
                "k",
                self.total_num_heads * self.head_size,
                self.total_num_kv_heads * self.head_size,
            ),
            (
                "v",
                (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                self.total_num_kv_heads * self.v_head_size,
            ),
        ]

        for shard_id, shard_offset, shard_size in shard_offsets:
            # 量化特殊情况:
            # 如果 param 是 packed 存储，而且 packed_dim 恰好是 output_dim，
            # 则原始逻辑维度上的 shard_offset/shard_size 不能直接用，
            # 需要先折算到 packed 后的物理存储索引
            if (
                    isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
                    and param.packed_dim == param.output_dim
            ):
                shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset
                )

            # 沿 param.output_dim 从 fused 权重里切出当前 q/k/v 那一段
            #
            # 若 loaded_weight shape 类似:
            #   [out_dim, in_dim]
            # 则 loaded_weight_shard shape 类似:
            #   [shard_size, in_dim]
            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_offset, shard_size
            )

            # 递归交给 weight_loader_v2，显式带上 shard_id
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(
            self,
            param: BasevLLMParameter,
            loaded_weight: torch.Tensor,
            loaded_shard_id: str | None = None,
    ):
        # QKVParallelLinear 的新版加载入口。
        #
        # 设计目标：
        # - 不在这里手写过多“按参数类型分支 + narrow + copy”的细节
        # - 而是把 Q/K/V 的真正写入逻辑尽量下沉到
        #   `param.load_qkv_weight(...)`
        #
        # 当前项目上下文里，需要特别注意两点：
        # 1. 这是“支持新版量化参数接口”的通用入口，不等于“当前一定在加载 GPTQ QKV”。
        # 2. 对当前 Qwen3.5-122B-A10B-GPTQ-Int4 checkpoint 来说，
        #    full-attention 的 q/k/v 实际被 dynamic 规则 `-:.*attn.*` 排除量化，
        #    因此这几份权重通常仍是 BF16，而不是 GPTQ packed 权重。
        #
        # 例如当前本地 checkpoint 中，某个 full-attention 层实际可以看到：
        # - q_proj.weight: [16384, 3072] BF16
        # - k_proj.weight: [  512, 3072] BF16
        # - v_proj.weight: [  512, 3072] BF16
        #
        # 这里 q 的 16384 不是普通 32 * 256，而是：
        # - config.num_attention_heads = 32
        # - attn_output_gate = true
        # - 传给 QKVParallelLinear 的 total_num_heads 变成 32 * (1 + 1) = 64
        #   因而 q 段实际是 [q | gate] 融合后的输出
        self.validate_shard_id(loaded_shard_id)

        # ------------------------------------------------------------
        # Case 1: loaded_shard_id is None
        # ------------------------------------------------------------
        # 说明当前拿到的 loaded_weight 还是 fused 的 qkv 整体
        #
        # 当前 Qwen3.5-122B-A10B 的主流 checkpoint 一般不走这里，
        # 因为磁盘上更常见的是分开的：
        # - q_proj.weight
        # - k_proj.weight
        # - v_proj.weight
        #
        # 这个分支主要服务于“磁盘上已经把 qkv 合并存储”的模型。
        if loaded_shard_id is None:
            # 若当前参数是“per-tensor scale”这种量化参数
            # 直接交给 param.load_qkv_weight 处理 fused 加载逻辑
            if isinstance(param, PerTensorScaleParameter):
                param.load_qkv_weight(
                    loaded_weight=loaded_weight, shard_id=0, tp_rank=self.tp_rank
                )
                return

            # 若是普通 Base/Row 参数，也直接让 param 自己处理 fused qkv 加载
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight, tp_rank=self.tp_rank)
                return

            # 否则走手工拆 fused qkv 的路径
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        # ------------------------------------------------------------
        # Case 2: loaded_shard_id 已经明确是 q / k / v 之一
        # ------------------------------------------------------------
        # 当前 Qwen3.5-122B-A10B checkpoint 的 full-attention 层通常走这里：
        # - 加载 q_proj.weight -> loaded_shard_id = "q"
        # - 加载 k_proj.weight -> loaded_shard_id = "k"
        # - 加载 v_proj.weight -> loaded_shard_id = "v"
        #
        # 在当前启动配置下还能确定：
        # - tp_size = 1
        # - tp_rank = 0
        # - self.total_num_heads = 64   （32 个 q 头 + 32 个 gate 头）
        # - self.total_num_kv_heads = 2
        # - self.head_size = self.v_head_size = 256
        # - self.num_kv_head_replicas = 1
        #
        # 因而当前 rank 上的本地 shard 大小就是：
        # - q shard_size = 64 * 256 = 16384
        # - k shard_size =  2 * 256 =   512
        # - v shard_size =  2 * 256 =   512
        # 下面算的是“当前 rank 本地 qkv 拼接布局”里的 local offset / local size
        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)

        # 若参数是 block quant 的 scale 参数，则 shard 的 offset/size
        # 还要根据 quant block 大小再做一次调整
        # 当前这份 122B-A10B checkpoint 的 attention q/k/v 多数是 BF16，
        # 一般不会命中这个 block-quant scale 分支。
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        # 最终交给 param 自己实现的 load_qkv_weight
        #
        # 关键参数:
        # - loaded_weight: 当前这份 q/k/v shard 的权重
        # - num_heads=self.num_kv_head_replicas:
        #     这里主要是告诉底层 K/V 是否存在 replica 情况
        #     当前 tp_size=1，所以这个值可确定为 1
        # - shard_id:
        #     当前是 q / k / v 的哪一段
        # - shard_offset/shard_size:
        #     当前段在本地 fused qkv 布局中的位置
        #     当前可确定为：
        #     - q: offset=0,     size=16384
        #     - k: offset=16384, size=512
        #     - v: offset=16896, size=512
        # - tp_rank:
        #     当前 TP rank 编号；当前启动命令未开 TP，可确定为 0
        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=self.num_kv_head_replicas,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
        )

    def weight_loader(
            self,
            
            param: Parameter,
            loaded_weight: torch.Tensor,
            loaded_shard_id: str | None = None,
    ):
        # 旧版/通用 loader
        # 相比 v2，这里自己手写了更多 shard narrow / packed / gguf / bnb4bit 逻辑
        self.validate_shard_id(loaded_shard_id)

        # ------------------------------------------------------------
        # GGUF 特殊情况 1: 当前 param 存的是“权重量化类型”
        # ------------------------------------------------------------
        # 比如 q/k/v 各自的量化类型 metadata
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)

        if is_gguf_weight_type:
            idx_map = {"q": 0, "k": 1, "v": 2}

            # 如果 shard_id 已知，就只给对应 q/k/v 那个槽位赋值
            if loaded_shard_id is not None:
                param.data[idx_map[loaded_shard_id]].copy_(loaded_weight)
                param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            else:
                # 如果 shard_id 不知道，就把同一个类型值复制给 q/k/v 三段
                param.shard_weight_type = {k: loaded_weight.item() for k in idx_map}
            return

        # ------------------------------------------------------------
        # GGUF 特殊情况 2: 真正的 GGUF 权重
        # ------------------------------------------------------------
        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)

            # 假设 GGUF loaded_weight 在 output_dim 上还是全量，需要先按 TP 切
            shard_size = loaded_weight.size(output_dim) // self.tp_size
            start_idx = self.tp_rank * shard_size

            if loaded_shard_id is not None:
                # 切出当前 TP rank 对应那一段
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

                # 把它存到 param 的容器中，按 q/k/v 分别记录
                param.shard_id.append(loaded_shard_id)
                param.shard_id_map[loaded_shard_id] = len(param.data_container)
                param.data_container.append(loaded_weight)
                return

        # param_data 是目标参数实际存储张量
        param_data = param.data

        # output_dim 表示参数的哪个维度对应“输出维”
        # 对线性层权重，常见就是 0（比如 [out_dim, in_dim]）
        output_dim = getattr(param, "output_dim", None)

        # 某些 fused per-tensor scale 参数，需要把标量扩成数组后再加载
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        # ------------------------------------------------------------
        # Case 1: loaded_shard_id is None
        # ------------------------------------------------------------
        # 说明 loaded_weight 本身已经是 fused qkv 格式:
        #   [ q_full | k_full | v_full ]
        if loaded_shard_id is None:
            # 若参数本身没有 output_dim 概念，
            # 那就只能整体 copy
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0
                    )

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return

            # 定义 fused 磁盘权重中 q/k/v 的全局 offset / size
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                ("q", 0, self.total_num_heads * self.head_size),
                (
                    "k",
                    self.total_num_heads * self.head_size,
                    self.total_num_kv_heads * self.head_size,
                ),
                (
                    "v",
                    (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                    self.total_num_kv_heads * self.v_head_size,
                ),
            ]

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            packed_dim = getattr(param, "packed_dim", None)

            # 逐段切出 q/k/v，然后递归再加载
            for shard_id, shard_offset, shard_size in shard_offsets:
                # 若是 packed 量化格式，则逻辑维 offset/size 要先映射到物理 packed 维
                if packed_dim == output_dim:
                    shard_size = shard_size // param.packed_factor
                    shard_offset = shard_offset // param.packed_factor

                    # Marlin 的物理布局比较特殊，还要再调一次
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset
                    )

                # bitsandbytes 4bit 有自己独特的 shard 物理布局
                if use_bitsandbytes_4bit:
                    orig_qkv_offsets = {
                        "q": (0, self.total_num_heads * self.head_size),
                        "k": (
                            self.total_num_heads * self.head_size,
                            self.total_num_kv_heads * self.head_size,
                        ),
                        "v": (
                            (self.total_num_heads + self.total_num_kv_heads)
                            * self.head_size,
                            self.total_num_kv_heads * self.v_head_size,
                        ),
                        "total": (
                            (self.total_num_heads + self.total_num_kv_heads)
                            * self.head_size
                            + self.total_num_kv_heads * self.v_head_size,
                            0,
                        ),
                    }

                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_qkv_offsets, shard_id
                    )

                # 从 fused loaded_weight 中切出当前这一段
                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )

                # 再递归进入 “已知 shard_id” 的路径
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        # ------------------------------------------------------------
        # Case 2: loaded_shard_id 已知，是 q/k/v 某一段
        # ------------------------------------------------------------
        # 如果 param 有 output_dim，就按标准线性层方式处理
        if output_dim is not None:
            # 这里的 shard_offset / shard_size 是“当前 rank 本地 fused qkv 布局”里的 local offset/size
            if loaded_shard_id == "q":
                shard_offset = 0
                shard_size = self.num_heads * self.head_size
            elif loaded_shard_id == "k":
                shard_offset = self.num_heads * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            elif loaded_shard_id == "v":
                shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
                shard_size = self.num_kv_heads * self.v_head_size

            # block quant scale 参数还要根据 block 粒度调 offset/size
            if isinstance(param, BlockQuantScaleParameter):
                weight_block_size = getattr(self, "weight_block_size", None)
                shard_size, shard_offset = adjust_block_scale_shard(
                    weight_block_size, shard_size, shard_offset
                )

            # packed quant 权重: 逻辑维 -> packed 物理维
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.packed_factor
                shard_offset = shard_offset // param.packed_factor

                # Marlin 物理布局特殊，再调一次
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset
                )

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            is_sharded_weight = getattr(param, "is_sharded_weight", False)

            # bnb 4bit 通常 loaded_weight 已经是当前 portion，不需要再 narrow
            is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

            if use_bitsandbytes_4bit:
                orig_qkv_offsets = {
                    "q": (0, self.num_heads * self.head_size),
                    "k": (
                        self.num_heads * self.head_size,
                        self.num_kv_heads * self.head_size,
                    ),
                    "v": (
                        (self.num_heads + self.num_kv_heads) * self.head_size,
                        self.num_kv_heads * self.v_head_size,
                    ),
                    "total": (
                        (self.num_heads + self.num_kv_heads) * self.head_size
                        + self.num_kv_heads * self.v_head_size,
                        0,
                    ),
                }
                shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                    param, orig_qkv_offsets, loaded_shard_id
                )

            # 先把目标参数 param_data 自己 narrow 到当前 q/k/v 这段 local 区域
            #
            # 若 param_data shape 类似 [local_qkv_dim, in_dim]
            # 则 narrow 后变成 [shard_size, in_dim]
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)

            # 决定从 loaded_weight 里取哪一段给当前 TP rank
            #
            # 对 q:
            #   q 头总是严格按 TP 切，所以 shard_rank = tp_rank
            if loaded_shard_id == "q":
                shard_rank = self.tp_rank
            else:
                # 对 k/v:
                # 如果存在 KV replica，则多个 tp_rank 对应同一个真实 kv shard
                # 所以要把 tp_rank 压缩成“真实 KV shard 的 rank”
                shard_rank = self.tp_rank // self.num_kv_head_replicas

            start_idx = shard_rank * shard_size

            # 如果 loaded_weight 还不是预先切好的当前 rank 本地 shard，
            # 则从 loaded_weight 再 narrow 出当前 rank 需要的那一段
            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        # ------------------------------------------------------------
        # Case 3: 没有 output_dim，但需要 scalar -> array
        # ------------------------------------------------------------
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )

        # ------------------------------------------------------------
        # Case 4: 既没有 output_dim，也不是 scalar-to-array
        # ------------------------------------------------------------
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "QKVParallelLinear, assume the weight is the same "
                    "for all partitions."
                )

        # 最终要求:
        # - 目标参数块 param_data
        # - 当前加载权重块 loaded_weight
        # 二者 shape 必须完全一致
        assert param_data.shape == loaded_weight.shape

        # 真实拷贝到参数里
        param_data.copy_(loaded_weight)


# --8<-- [start:row_parallel_linear]
@PluggableLayer.register("row_parallel_linear")
class RowParallelLinear(LinearBase):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        reduce_results: If true, call all-reduce on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y = X_iA_i
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.down_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, weights matrix won't be sharded through tp rank.
    """

    # --8<-- [end:row_parallel_linear]

    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = True,
            input_is_parallel: bool = True,
            skip_bias_add: bool = False,
            params_dtype: torch.dtype | None = None,
            reduce_results: bool = True,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
            *,
            return_bias: bool = True,
            disable_tp: bool = False,
    ):
        # Divide the weight matrix along the first dimension.
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        super().__init__(
            input_size,
            output_size,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        assert self.quant_method is not None
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError(
                "When not reduce the results, adding bias to the "
                "results can lead to incorrect results"
            )

        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)
        self.update_param_tp_status()

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        input_dim = getattr(param, "input_dim", None)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow
        is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            weight_shape = list(loaded_weight.shape)
            if input_dim:
                weight_shape[input_dim] = weight_shape[input_dim] // self.tp_size
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)

        param_data = param.data
        if input_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def weight_loader_v2(self, param: BasevLLMParameter, loaded_weight: torch.Tensor):
        # RowParallelLinear 常见于 down_proj / o_proj 这类“沿输入维切分”的层。
        #
        # 当前 Qwen3.5-122B-A10B-GPTQ-Int4 中：
        # - shared_expert.down_proj 命中 -:.*shared_expert.*，通常走 BF16 非量化
        # - attention 的 o_proj      命中 -:.*attn.*，通常走 BF16 非量化
        # - routed experts 的 down_proj 不走这里，而走 FusedMoE.weight_loader(...)
        #
        # 因而当前 RowParallelLinear 的主职责，更多是在 attention/shared expert
        # 这些非量化层上把磁盘全量 weight 沿 input_dim 切成本地 shard。
        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)

        param.load_row_parallel_weight(loaded_weight=loaded_weight)

    def forward(
            self,
            input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            split_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = split_input[self.tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self, input_parallel, bias_)

        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s
