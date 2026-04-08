# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Hashable
from fractions import Fraction
from weakref import WeakValueDictionary

import torch
from torch.nn import Parameter

from cfie.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from cfie.logger import init_logger

__all__ = [
    "BasevLLMParameter",
    "PackedvLLMParameter",
    "PerTensorScaleParameter",
    "ModelWeightParameter",
    "ChannelQuantScaleParameter",
    "GroupQuantScaleParameter",
    "PackedColumnParameter",
    "RowvLLMParameter",
]

logger = init_logger(__name__)


class BasevLLMParameter(Parameter):
    """
    Base parameter for vLLM linear layers. Extends the torch.nn.parameter
    by taking in a linear weight loader. Will copy the loaded weight
    into the parameter when the provided weight loader is called.
    """

    def __new__(cls, data: torch.Tensor | None, **kwargs):
        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(self, data: torch.Tensor, weight_loader: Callable):
        """
        Initialize the BasevLLMParameter

        :param data: torch tensor with the parameter data
        :param weight_loader: weight loader callable

        :returns: a torch.nn.parameter
        """

        # During weight loading, we often do something like:
        # narrowed_tensor = param.data.narrow(0, offset, len)
        # narrowed_tensor.copy_(real_weight)
        # expecting narrowed_tensor and param.data to share the same storage.
        # However, on TPUs, narrowed_tensor will lazily propagate to the base
        # tensor, which is param.data, leading to the redundant memory usage.
        # This sometimes causes OOM errors during model loading. To avoid this,
        # we sync the param tensor after its weight loader is called.
        from cfie.platforms import current_platform

        if current_platform.use_sync_weight_loader():
            weight_loader = current_platform.make_synced_weight_loader(weight_loader)

        self._weight_loader = weight_loader
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

    @property
    def weight_loader(self) -> Callable:
        # NOTE(@ksayers) some models such as mamba_mixer2 override the
        # weight loader to support custom loading. In the future, model-specific
        # weight loading should be implemented via Model.load_weights. In the
        # meantime, support deleting and overriding `weight_loader` attribute
        if self._weight_loader is None:
            raise AttributeError(
                f"{self.__class__.__name__} weight_loader attribute has been deleted"
            )
        return self._weight_loader

    @weight_loader.setter
    def weight_loader(self, value: Callable):
        self._weight_loader = value

    @weight_loader.deleter
    def weight_loader(self):
        self._weight_loader = None  # type: ignore[assignment]

    def _is_1d_and_scalar(self, loaded_weight: torch.Tensor):
        cond1 = self.data.ndim == 1 and self.data.numel() == 1
        cond2 = loaded_weight.ndim == 0 and loaded_weight.numel() == 1
        return cond1 and cond2

    def _assert_and_load(self, loaded_weight: torch.Tensor):
        assert self.data.shape == loaded_weight.shape or self._is_1d_and_scalar(
            loaded_weight
        )
        self.data.copy_(loaded_weight)

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        self._assert_and_load(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        self._assert_and_load(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._assert_and_load(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        self._assert_and_load(loaded_weight)

    def _shard_id_as_int(self, shard_id: str | int) -> int:
        if isinstance(shard_id, int):
            return shard_id

        # if not int, assume shard_id for qkv
        # map to int and return
        qkv_idxs = {"q": 0, "k": 1, "v": 2}
        assert isinstance(shard_id, str)
        assert shard_id in qkv_idxs
        return qkv_idxs[shard_id]

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)


class _ColumnvLLMParameter(BasevLLMParameter):
    """
    Private class defining weight loading functionality
    (load_merged_column_weight, load_qkv_weight)
    for parameters being loaded into linear layers with column
    parallelism. This includes QKV and MLP layers which are
    not already fused on disk. Requires an output dimension
    to be defined. Called within the weight loader of
    each of the column parallel linear layers.
    """

    def __init__(self, output_dim: int, **kwargs):
        self._output_dim = output_dim
        super().__init__(**kwargs)

    @property
    def output_dim(self):
        return self._output_dim

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        shard_size = self.data.shape[self.output_dim]
        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, shard_size
        )
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_offset = kwargs.get("shard_offset")
        shard_size = kwargs.get("shard_size")

        # 这是 fused MLP / fused projection 在参数级别的真正写入逻辑。
        #
        # 当前 Qwen3.5 里常见的调用方：
        # - MergedColumnParallelLinear.weight_loader_v2(...)
        #
        # 当前 122B-A10B-GPTQ-Int4 里更常见的是两类：
        # 1. shared_expert.gate_up_proj 的 BF16 非量化权重
        # 2. 若某些 fused linear 未被 dynamic 排除量化，则也可承接其 qweight/scales/qzeros
        #
        # 这里不会自己判断“这是 gate 还是 up”，这些语义已经在上一层
        # MergedColumnParallelLinear.weight_loader_v2(...) 中用 shard_id 换算成
        # shard_offset / shard_size 了；这里仅负责把本地那一段 copy 进去。

        # TODO: move these to PackedColumnParameter and PackedvLLMParameter
        if (
            isinstance(self, (PackedColumnParameter, PackedvLLMParameter))
            and self.packed_dim == self.output_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size
            )

        param_data = self.data

        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, shard_size
        )
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        shard_offset = kwargs.get("shard_offset")
        shard_size = kwargs.get("shard_size")
        shard_id = kwargs.get("shard_id")
        num_heads = kwargs.get("num_heads")

        # TODO: move these to PackedColumnParameter and PackedvLLMParameter
        if (
            isinstance(self, (PackedColumnParameter, PackedvLLMParameter))
            and self.output_dim == self.packed_dim
        ):
            shard_size, shard_offset = self.adjust_shard_indexes_for_packing(
                shard_offset=shard_offset, shard_size=shard_size
            )

        param_data = self.data
        shard_id = self.tp_rank if shard_id == "q" else self.tp_rank // num_heads
        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(
            self.output_dim, shard_id * shard_size, shard_size
        )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class RowvLLMParameter(BasevLLMParameter):
    """
    这是一个带“row parallel 权重加载语义”的参数类。

    适用场景：
    - 某个线性层是 RowParallelLinear
    - 这类层的权重在“输入维”上做切分
    - 因此当前 TP rank 只持有全局权重在 input_dim 上的一段

    它的核心职责：
    - 知道哪个维度是输入维（input_dim）
    - 在加载 checkpoint 时，沿 input_dim 取出当前 tp_rank 对应的 shard
    - 再把这段 shard 拷进本地参数 self.data

    直观理解：
    如果一个全局权重逻辑 shape 是：
        [K_global, N]
    并且 row parallel 按 K 维切分，
    那每个 rank 只保存：
        [K_local, N]
    其中：
        K_local = K_global / tp_size
    """

    def __init__(self, input_dim: int, **kwargs):
        # input_dim 表示“哪个维度是输入维”
        #
        # 例如：
        # - 如果参数逻辑 shape 是 [K, N]，通常 input_dim = 0
        # - 如果参数逻辑 shape 是 [N, K]，那 input_dim 可能就是 1
        #
        # RowParallel 的关键就是：沿 input_dim 切 shard
        self._input_dim = input_dim

        # 其余通用参数初始化交给 BasevLLMParameter
        super().__init__(**kwargs)

    @property
    def input_dim(self):
        # 返回当前参数的输入维编号
        return self._input_dim

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        # -------------------------------------------------------------
        # Step 1. 计算当前 rank 本地 shard 在 input_dim 上的长度
        # -------------------------------------------------------------
        #
        # 当前 Qwen3.5-122B-A10B 场景下，这条路径通常服务于：
        # - shared_expert.down_proj（BF16，因 -:.*shared_expert.* 被排除量化）
        # - attention.o_proj         （BF16，因 -:.*attn.* 被排除量化）
        #
        # routed experts 的 w2 不会走这里，而是走 FusedMoE.weight_loader。
        # self.data 是当前 rank 本地参数存储
        #
        # 假设：
        #   self.data.shape[input_dim] = K_local
        #
        # 那么 shard_size = K_local
        #
        # 也就是说：
        # 当前 rank 需要从全局 loaded_weight 中，沿 input_dim 取出长度为 K_local 的一段
        shard_size = self.data.shape[self.input_dim]

        # -------------------------------------------------------------
        # Step 2. 从全局 loaded_weight 中切出当前 tp_rank 对应的那一段
        # -------------------------------------------------------------
        # 起始位置：
        #   start = tp_rank * shard_size
        #
        # 取法：
        #   沿 input_dim 做 narrow
        #
        # 例如：
        #   loaded_weight 逻辑 shape = [K_global, N]
        #   tp_size = 4
        #   tp_rank = 2
        #   shard_size = K_local = K_global / 4
        #
        # 那这里就等价于取：
        #   loaded_weight[2*K_local : 3*K_local, :]
        loaded_weight = loaded_weight.narrow(
            self.input_dim, self.tp_rank * shard_size, shard_size
        )

        # -------------------------------------------------------------
        # Step 3. 特殊情况：如果磁盘上加载的是 0 维标量，统一 reshape 成 [1]
        # -------------------------------------------------------------
        # 这通常是为了兼容某些 scale / 特殊参数加载路径
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        # -------------------------------------------------------------
        # Step 4. 校验本地参数 shape 和切出来的 shard shape 一致
        # -------------------------------------------------------------
        assert self.data.shape == loaded_weight.shape

        # -------------------------------------------------------------
        # Step 5. 真正拷贝到当前 rank 的本地参数存储
        # -------------------------------------------------------------
        self.data.copy_(loaded_weight)


class ModelWeightParameter(_ColumnvLLMParameter, RowvLLMParameter):
    """
    Parameter class for linear layer weights. Uses both column and
    row parallelism.
    """

    pass


class GroupQuantScaleParameter(_ColumnvLLMParameter, RowvLLMParameter):
    """
    Parameter class for weight scales loaded for weights with
    grouped quantization. Uses both column and row parallelism.
    """

    pass


class ChannelQuantScaleParameter(_ColumnvLLMParameter):
    """
    Parameter class for weight scales loaded for weights with
    channel-wise quantization. Equivalent to _ColumnvLLMParameter.
    """

    pass


class PerTensorScaleParameter(BasevLLMParameter):
    """
    Parameter class for scales where the number of scales is
    equivalent to the number of logical matrices in fused linear
    layers (e.g. for QKV, there are 3 scales loaded from disk).
    This is relevant to weights with per-tensor quantization.
    Adds functionality to map the scalers to a shard during
    weight loading.

    Note: additional parameter manipulation may be handled
    for each quantization config specifically, within
    process_weights_after_loading
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # For row parallel layers, no sharding needed
    # load weight into parameter as is
    def load_row_parallel_weight(self, *args, **kwargs):
        super().load_row_parallel_weight(*args, **kwargs)

    def load_merged_column_weight(self, *args, **kwargs):
        self._load_into_shard_id(*args, **kwargs)

    def load_qkv_weight(self, *args, **kwargs):
        self._load_into_shard_id(*args, **kwargs)

    def load_column_parallel_weight(self, *args, **kwargs):
        super().load_row_parallel_weight(*args, **kwargs)

    def _load_into_shard_id(
        self, loaded_weight: torch.Tensor, shard_id: str | int, **kwargs
    ):
        """
        Slice the parameter data based on the shard id for
        loading.
        """

        param_data = self.data
        shard_id = self._shard_id_as_int(shard_id)

        # AutoFP8 scales do not have a shape
        # compressed-tensors scales do have a shape
        if len(loaded_weight.shape) != 0:
            assert loaded_weight.shape[0] == 1
            loaded_weight = loaded_weight[0]

        param_data = param_data[shard_id]
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class PackedColumnParameter(_ColumnvLLMParameter):
    """
    Parameter for model parameters which are packed on disk
    and support column parallelism only. See PackedvLLMParameter
    for more details on the packed properties.
    """

    def __init__(
        self,
        packed_factor: int | Fraction,
        packed_dim: int,
        marlin_tile_size: int | None = None,
        **kwargs,
    ):
        self._packed_factor = packed_factor
        self._packed_dim = packed_dim
        self._marlin_tile_size = marlin_tile_size
        super().__init__(**kwargs)

    @property
    def packed_dim(self):
        return self._packed_dim

    @property
    def packed_factor(self):
        return self._packed_factor

    @property
    def marlin_tile_size(self):
        return self._marlin_tile_size

    def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
        return _adjust_shard_indexes_for_packing(
            shard_size=shard_size,
            shard_offset=shard_offset,
            packed_factor=self.packed_factor,
            marlin_tile_size=self.marlin_tile_size,
        )


class PackedvLLMParameter(ModelWeightParameter):
    """
    表示一种“磁盘/参数中以 packed 形式存储”的模型权重参数。

    典型例子：
    - GPTQ Marlin 的 qweight：int4 / int8 量化值被 packed 进 int32
    - AWQ Marlin 的 qweight / qzeros：也是 packed 存储

    为什么需要这个类？
    因为对于 packed 权重来说：

    逻辑上你可能认为某个维度大小是：
        original_dim = 1024

    但物理上如果是 int4 packed 到 int32，
    例如一个 int32 里装 8 个 4bit 值，那么实际存储维度可能只有：
        packed_dim = 1024 / 8 = 128

    所以：
    - shard_size
    - shard_offset
    - fused q/k/v 分段位置
    都不能直接按“逻辑维度”切，
    必须折算到“packed 后的物理维度”。
    """

    def __init__(
        self,
        packed_factor: int | Fraction,         # 打包因子：一个物理存储单元里包含多少个逻辑量化值
        packed_dim: int,                       # 哪个维度发生了 packed
        marlin_tile_size: int | None = None,   # （可选）Marlin kernel 的 tile size，用于某些额外对齐修正
        **kwargs,
    ):
        # packed_factor:
        # 例如 int4 -> int32 时，常见 packed_factor = 8
        # 因为 32bit / 4bit = 8
        #
        # 也就是说：
        #   逻辑上 8 个 int4 值
        #   物理上占 1 个 int32 元素
        self._packed_factor = packed_factor

        # packed_dim:
        # 指明“发生 packed 的是哪个维度”
        #
        # 例如：
        # - GPTQMarlin qweight 常见 packed_dim = 0
        #   形状像 [K_local / pack_factor, N_local]
        #
        # - AWQMarlin qweight / qzeros 常见 packed_dim = 1
        #   形状像 [K_local, N_local / pack_factor]
        self._packed_dim = packed_dim

        # marlin_tile_size:
        # 某些 Marlin kernel 除了 packed_factor 外，
        # 还会要求权重分片在 tile 粒度上做额外对齐/换算
        #
        # 如果这个值不为 None，则 adjust_shard_indexes_for_packing(...)
        # 在折算 shard_size/shard_offset 时还会把 tile 规则考虑进去
        self._marlin_tile_size = marlin_tile_size

        # 其余公共参数初始化交给父类 ModelWeightParameter
        super().__init__(**kwargs)

    @property
    def packed_dim(self):
        # 返回哪个维度是 packed 维
        return self._packed_dim

    @property
    def packed_factor(self):
        # 返回 packed 因子
        #
        # 例如：
        # - int4 packed into int32 -> 8
        # - int8 packed into int32 -> 4
        return self._packed_factor

    @property
    def marlin_tile_size(self):
        # 返回 Marlin tile size（若有）
        return self._marlin_tile_size

    def adjust_shard_indexes_for_packing(self, shard_size, shard_offset):
        # 这个函数是这类参数最核心的方法。
        #
        # 输入的 shard_size / shard_offset 通常是“逻辑维度下”的分片信息，
        # 也就是说它们还没考虑 packed 存储。
        #
        # 例如逻辑上你想取：
        #   某维度上从 offset=128 开始，长度 size=256
        #
        # 但如果这个维度 packed_factor=8，
        # 那物理上真正应该取的是：
        #   offset=128/8=16
        #   size=256/8=32
        #
        # 如果再叠加 Marlin tile 对齐要求，
        # 还要进一步按 tile 大小修正。
        #
        # 返回值：
        #   (adjusted_shard_size, adjusted_shard_offset)
        #
        # 它们已经是“packed 后物理存储视角”的 shard 信息，
        # 后续才能安全地用 tensor.narrow(...) 去切参数。
        return _adjust_shard_indexes_for_packing(
            shard_size=shard_size,
            shard_offset=shard_offset,
            packed_factor=self.packed_factor,
            marlin_tile_size=self.marlin_tile_size,
        )


class BlockQuantScaleParameter(_ColumnvLLMParameter, RowvLLMParameter):
    """
    Parameter class for weight scales loaded for weights with
    block-wise quantization. Uses both column and row parallelism.
    """

    pass


class SharedWeightParameter(BasevLLMParameter):
    """
    Parameter for weights with many shared tensors across a model

    For example, when applying transforms to the "gate" and "up" partitions of
    `MergedColumnParallelLinear`, the transform weights must stay separate
    tensors in order to allow for tensor memory sharing between layers.
    """

    # global registry for sharing tensors based on passed `data_key`
    # this dict holds weaksrefs to avoid memory leak after model cleanup
    tensors_registry: WeakValueDictionary = WeakValueDictionary()

    # local container for strong references to shared tensors
    # this set compensates for the fact that torch.nn.Parameter
    # and Parameter subclasses do not hold reliable references to tensors
    local_tensors: set[torch.Tensor]

    # dictionary mapping partition indices to associated parameters
    partitions: dict[int, ModelWeightParameter | Parameter]

    def __new__(cls, **kwargs):
        return super().__new__(cls, data=None, **kwargs)

    def __init__(self, input_dim: int = 1, output_dim: int = 0, **kwargs):
        weight_loader: Callable = kwargs.get("weight_loader")  # type: ignore[assignment]
        super().__init__(data=None, weight_loader=weight_loader)

        self.local_tensors = set()
        self.partitions = {}
        self.kwargs = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "weight_loader": self._fake_weight_loader,
        }

        if self.tp_size > 1:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not "
                "currently support tensor parallelism"
            )

    def add_partition(self, index: int, data_key: Hashable, *args, **kwargs):
        """
        Add a partition to the weight parameter. Partitions whose `data_key`
        is the same will share tensor data

        :param index: index of partition to add
        :param data_key: hashable key used to key shared tensors
        :param *args: arguments for `torch.empty`
        :param **kwargs: keyword arguments for `torch.empty`
        """
        # load (shared) tensor using `data_key`
        if data_key not in self.tensors_registry:
            data = torch.empty(*args, **kwargs)
            self.tensors_registry[data_key] = data
        else:
            data = self.tensors_registry[data_key]

        # create associated model parameter
        self.partitions[index] = ModelWeightParameter(data=data, **self.kwargs)  # type: ignore[arg-type]

        # hold local reference, since ModelWeightParameter does not
        # see https://github.com/pytorch/pytorch/issues/75932
        self.local_tensors.add(data)

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        assert len(self.partitions) == 1 and 0 in self.partitions
        partition = self.partitions[0]

        ModelWeightParameter.load_column_parallel_weight(partition, loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        assert len(self.partitions) == 1 and 0 in self.partitions
        partition = self.partitions[0]

        ModelWeightParameter.load_row_parallel_weight(partition, loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        partition_id = kwargs.pop("shard_id")
        partition_id = self._shard_id_as_int(partition_id)
        partition = self.partitions[partition_id]

        input_dim = self.kwargs.get("input_dim")
        shard_size = partition.data.size(input_dim) // self.tp_size
        shard_offset = self.tp_rank * shard_size

        ModelWeightParameter.load_merged_column_weight(
            partition, loaded_weight, shard_offset=shard_offset, shard_size=shard_size
        )

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        partition_id = self._shard_id_as_int(kwargs.pop("shard_id"))
        partition = self.partitions[partition_id]

        input_dim = self.kwargs.get("input_dim")
        shard_size = partition.data.size(input_dim) // self.tp_size
        shard_offset = self.tp_rank * shard_size
        shard_id = "q"  # fake first partition
        num_heads = kwargs.get("num_heads")

        ModelWeightParameter.load_qkv_weight(
            partition,
            loaded_weight,
            shard_offset=shard_offset,
            shard_size=shard_size,
            shard_id=shard_id,
            num_heads=num_heads,
        )

    def process_weights_after_loading(self):
        for key in self.partitions:
            self.partitions[key] = torch.nn.Parameter(
                data=self.partitions[key].data, requires_grad=False
            )

    @property
    def data(self):
        raise ValueError(
            "Accessing `data` of a `SharedWeightParameter` is not allowed. "
            "Instead, use `get_partition` to get the weight of "
            "the particular partition you want to access"
        )

    def _fake_weight_loader(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_weight_shard_id: str | int | None,
    ):
        raise ValueError(
            "When loading partition weights of "
            f"{self.__class__.__name__}, use methods provided by "
            f"{self.__class__.__name__}, not partition loader"
        )


def permute_param_layout_(
    param: BasevLLMParameter, input_dim: int, output_dim: int, **kwargs
) -> BasevLLMParameter:
    """
    Permute a parameter's layout to the specified input and output dimensions,
    useful for forcing the parameter into a known layout, for example, if I need
    a packed (quantized) weight matrix to be in the layout
        {input_dim = 0, output_dim = 1, packed_dim = 0}
    then I can call:
        permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
    to ensure x is in the correct layout (permuting it to the correct layout if
    required, asserting if it cannot get it to the correct layout)
    """

    curr_input_dim = getattr(param, "input_dim", None)
    curr_output_dim = getattr(param, "output_dim", None)

    if curr_input_dim is None or curr_output_dim is None:
        assert param.data.dim() == 2, (
            "permute_param_layout_ only supports 2D parameters when either "
            "input_dim or output_dim is not set"
        )

    # if one of the dimensions is not set, set it to the opposite of the other
    #  we can only do this since we asserted the parameter is 2D above
    if curr_input_dim is None:
        assert curr_output_dim is not None, "either input or output dim must be set"
        curr_input_dim = (curr_output_dim + 1) % 2
    if curr_output_dim is None:
        assert curr_input_dim is not None, "either input or output dim must be set"
        curr_output_dim = (curr_input_dim + 1) % 2

    # create permutation from the current layout to the layout with
    # self.input_dim at input_dim and self.output_dim at output_dim preserving
    # other dimensions
    perm = [
        i for i in range(param.data.dim()) if i not in [curr_input_dim, curr_output_dim]
    ]
    perm.insert(input_dim, curr_input_dim)
    perm.insert(output_dim, curr_output_dim)

    if "packed_dim" in kwargs:
        assert (
            hasattr(param, "packed_dim")
            and param.packed_dim == perm[kwargs["packed_dim"]]
        ), "permute_param_layout_ currently doesn't support repacking"

    param.data = param.data.permute(*perm)
    if hasattr(param, "_input_dim"):
        param._input_dim = input_dim
    if hasattr(param, "_output_dim"):
        param._output_dim = output_dim
    if "packed_dim" in kwargs and hasattr(param, "_packed_dim"):
        param._packed_dim = kwargs["packed_dim"]

    return param


def _adjust_shard_indexes_for_marlin(shard_size, shard_offset, marlin_tile_size):
    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size


def _adjust_shard_indexes_for_packing(
    shard_size, shard_offset, packed_factor, marlin_tile_size
):
    shard_size = shard_size // packed_factor
    shard_offset = shard_offset // packed_factor
    if marlin_tile_size is not None:
        return _adjust_shard_indexes_for_marlin(
            shard_size=shard_size,
            shard_offset=shard_offset,
            marlin_tile_size=marlin_tile_size,
        )

    return shard_size, shard_offset
