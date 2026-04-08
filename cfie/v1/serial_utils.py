# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import importlib
import pickle
from collections.abc import Callable, Sequence
from functools import partial
from inspect import isclass
from types import FunctionType
from typing import Any, ClassVar, TypeAlias, cast, get_type_hints

import cloudpickle
import msgspec
import numpy as np
import torch
import zmq
from msgspec import msgpack
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from cfie import envs
from cfie.logger import init_logger
from cfie.multimodal.inputs import (
    BaseMultiModalField,
    MultiModalBatchedField,
    MultiModalFieldConfig,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalSharedField,
    NestedTensors,
)
from cfie.utils.platform_utils import is_pin_memory_available
from cfie.v1.utils import tensor_data

logger = init_logger(__name__)

CUSTOM_TYPE_PICKLE = 1
CUSTOM_TYPE_CLOUDPICKLE = 2
CUSTOM_TYPE_RAW_VIEW = 3

# MultiModalField class serialization type map.
# These need to list all possible field types and match them
# to factory methods in `MultiModalFieldConfig`.
MMF_CLASS_TO_FACTORY: dict[type[BaseMultiModalField], str] = {
    MultiModalFlatField: "flat",
    MultiModalSharedField: "shared",
    MultiModalBatchedField: "batched",
}

bytestr: TypeAlias = bytes | bytearray | memoryview | zmq.Frame


def _log_insecure_serialization_warning():
    logger.warning_once(
        "Allowing insecure serialization using pickle due to "
        "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
    )


def _typestr(val: Any) -> tuple[str, str] | None:
    if val is None:
        return None
    t = type(val)
    return t.__module__, t.__qualname__


def _encode_type_info_recursive(obj: Any) -> Any:
    """Recursively encode type information for nested structures of
    lists/dicts."""
    if obj is None:
        return None
    if type(obj) is list:
        return [_encode_type_info_recursive(item) for item in obj]
    if type(obj) is dict:
        return {k: _encode_type_info_recursive(v) for k, v in obj.items()}
    return _typestr(obj)


def _decode_type_info_recursive(
    type_info: Any, data: Any, convert_fn: Callable[[Sequence[str], Any], Any]
) -> Any:
    """Recursively decode type information for nested structures of
    lists/dicts."""
    if type_info is None:
        return data
    if isinstance(type_info, dict):
        assert isinstance(data, dict)
        return {
            k: _decode_type_info_recursive(type_info[k], data[k], convert_fn)
            for k in type_info
        }
    if isinstance(type_info, list) and (
        # Exclude serialized tensors/numpy arrays.
        len(type_info) != 2 or not isinstance(type_info[0], str)
    ):
        assert isinstance(data, list)
        return [
            _decode_type_info_recursive(ti, d, convert_fn)
            for ti, d in zip(type_info, data)
        ]
    return convert_fn(type_info, data)


class UtilityResult:
    """Wrapper for special handling when serializing/deserializing."""

    def __init__(self, r: Any = None):
        self.result = r


# 负责把请求/输出对象编码成 msgpack + 零拷贝附加 buffer 的序列化器。
class MsgpackEncoder:
    """
    Msgpack 编码器，支持自定义的 torch tensor 与 numpy array 序列化。

    需要注意的是，与原生 `msgspec` Encoder 不同，
    这个接口在编码 tensor / numpy array 时通常不是线程安全的。

    默认情况下，小于阈值的数组会被直接内联编码；
    更大的数组则会拆到独立消息里发送。
    这里的阈值是按“单个 tensor/array”计算的。
    """

    # 初始化 msgpack 编码器，并准备零拷贝附加 buffer 暂存区。
    def __init__(self, size_threshold: int | None = None):
        # 若调用方未显式传入阈值，则使用环境变量中的默认阈值。
        if size_threshold is None:
            size_threshold = envs.VLLM_MSGPACK_ZERO_COPY_THRESHOLD

        # 创建底层 msgpack 编码器，并挂上自定义对象编码钩子。
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)

        # 这里是一次编码过程中的临时 buffer 暂存区，供 enc_hook 访问。
        # `msgspec` 的 hook 接口本身无法额外传入上下文数据，因此只能借助实例字段暂存。
        self.aux_buffers: list[bytestr] | None = None

        # 保存当前实例采用的零拷贝阈值。
        self.size_threshold = size_threshold

        # 若允许不安全序列化，则提前打印一次告警。
        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            _log_insecure_serialization_warning()

    # 把对象编码成“主 msgpack buffer + 若干附加 backing buffer”。
    def encode(self, obj: Any) -> Sequence[bytestr]:
        try:
            # 先创建 buffer 列表，并预留第 0 项给主 msgpack buffer。
            self.aux_buffers = bufs = [b""]

            # 把顶层对象编码进主 buffer。
            bufs[0] = self.encoder.encode(obj)

            # `bufs` 里除了主 buffer 外，还可能保存 tensor/ndarray 的底层内存视图，
            # 从而避免把大块数据复制进新的 msgpack buffer。
            return bufs
        finally:
            # 无论成功还是失败，都清空本轮编码的临时 buffer 状态。
            self.aux_buffers = None

    # 直接把对象编码进调用方提供的 bytearray。
    def encode_into(self, obj: Any, buf: bytearray) -> Sequence[bytestr]:
        try:
            # 把外部提供的 bytearray 当作主 buffer 放进临时列表。
            self.aux_buffers = [buf]
            # 复用同一个列表变量，便于最后直接返回。
            bufs = self.aux_buffers
            # 把顶层对象直接编码进传入的 bytearray。
            self.encoder.encode_into(obj, buf)
            # 返回主 buffer 与所有附加 buffer。
            return bufs
        finally:
            # 无论成功还是失败，都清空本轮编码的临时 buffer 状态。
            self.aux_buffers = None

    # `msgspec` 在遇到非原生可编码对象时会回调到这里。
    def enc_hook(self, obj: Any) -> Any:
        # torch.Tensor 走自定义 tensor 编码路径。
        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)

        # numpy ndarray 且不是 object/void 类型时，走 ndarray 专用编码路径。
        # object/void 数组无法安全零拷贝编码，后面会退回 pickle。
        if isinstance(obj, np.ndarray) and obj.dtype.kind not in ("O", "V"):
            return self._encode_ndarray(obj)

        # slice 会被编码成 (start, stop, step) 三元组。
        if isinstance(obj, slice):
            # 这里假设 slice 内只会出现基于 int 的边界值。
            return tuple(
                int(v) if v is not None else None
                for v in (obj.start, obj.stop, obj.step)
            )

        # 单个多模态 kwargs item 走专用编码路径。
        if isinstance(obj, MultiModalKwargsItem):
            return self._encode_mm_item(obj)

        # 多个多模态 kwargs items 走批量编码路径。
        if isinstance(obj, MultiModalKwargsItems):
            return self._encode_mm_items(obj)

        # UtilityResult 需要为远端 utility 返回值做特殊处理。
        if isinstance(obj, UtilityResult):
            # 取出被包装的真正结果对象。
            result = obj.result
            # 禁止不安全序列化时，只原样返回结果，不附加类型信息。
            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                return None, result
            # utility 结果往往没有强类型声明，因此需要递归记录嵌套 list/dict 中的类型信息，
            # 以便解码端能更准确地恢复对象类型。
            return _encode_type_info_recursive(result), result

        # 若当前环境禁止不安全序列化，则未知对象一律报错。
        if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            raise TypeError(
                f"Object of type {type(obj)} is not serializable"
                "Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 to allow "
                "fallback to pickle-based serialization."
            )

        # 函数对象优先走 cloudpickle，因为普通 pickle 对方法/闭包支持较差。
        if isinstance(obj, FunctionType):
            # `pickle` 通常比 `cloudpickle` 更快，但序列化方法对象时可能失败。
            return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

        # 其他兜底对象走 pickle 扩展类型编码。
        return msgpack.Ext(
            CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        )

    # 编码 numpy ndarray，必要时把原始数据放进附加 buffer 列表。
    def _encode_ndarray(
        self, obj: np.ndarray
    ) -> tuple[str, tuple[int, ...], int | memoryview]:
        # ndarray 编码必须发生在一次有效的 encode/encode_into 调用内部。
        assert self.aux_buffers is not None
        # 非连续数组无法直接零拷贝引用其内存，因此先转成连续字节。
        arr_data = obj.data if obj.flags.c_contiguous else obj.tobytes()
        # 标量或小数组直接内联到主消息里。
        if not obj.shape or obj.nbytes < self.size_threshold:
            # 这里使用扩展类型包住 raw view，便于解码端避免额外复制。
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, arr_data)
        else:
            # 大数组只在主消息里写入附加 buffer 的索引，真实数据放到 aux_buffers。
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr_data)

        # ndarray 会被编码成 (dtype, shape, data) 三元组。
        # 其中 data 要么是内联的 raw view，要么是 aux_buffers 里的索引。
        return obj.dtype.str, obj.shape, data

    # 编码 torch.Tensor，必要时把底层字节视图放进附加 buffer 列表。
    def _encode_tensor(
        self,
        obj: torch.Tensor
    ) -> tuple[str, tuple[int, ...], int | memoryview]:
        # tensor 编码必须发生在一次有效的 encode/encode_into 调用内部。
        assert self.aux_buffers is not None

        # 取出 tensor 的底层连续字节视图。
        arr_data = tensor_data(obj)

        # 小 tensor 直接内联到主消息里。
        if obj.nbytes < self.size_threshold:
            # 小 tensor 与 ndarray 一样，直接编码成 raw view。
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, arr_data)
        else:
            # 大 tensor 只在主消息中记录 aux_buffers 下标，真实数据单独发送。
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr_data)

        # 把 `torch.float16` 这类 dtype 转成不带前缀的字符串。
        dtype = str(obj.dtype).removeprefix("torch.")

        # tensor 同样编码成 (dtype, shape, data) 三元组。
        return dtype, obj.shape, data

    # 编码一组按模态分桶的多模态 kwargs items。
    def _encode_mm_items(self, items: MultiModalKwargsItems) -> dict[str, Any]:
        # 对每个模态下的 item 列表逐项编码。
        return {
            modality: [self._encode_mm_item(item) for item in itemlist]
            for modality, itemlist in items.items()
        }

    # 编码一个多模态 kwargs item。
    def _encode_mm_item(self, item: MultiModalKwargsItem) -> dict[str, Any]:
        # 对 item 中每个字段元素分别编码。
        return {key: self._encode_mm_field_elem(elem) for key, elem in item.items()}

    # 编码单个多模态字段元素。
    def _encode_mm_field_elem(self, elem: MultiModalFieldElem) -> dict[str, Any]:
        # `data` 保存真正的多模态张量结构，`field` 保存字段描述信息。
        return {
            "data": (
                None if elem.data is None else self._encode_nested_tensors(elem.data)
            ),
            "field": self._encode_mm_field(elem.field),
        }

    # 递归编码嵌套张量结构。
    def _encode_nested_tensors(self, nt: NestedTensors) -> Any:
        # 叶子节点若是 tensor，则走 tensor 编码路径。
        if isinstance(nt, torch.Tensor):
            return self._encode_tensor(nt)
        # 叶子节点若是纯数值，则直接原样返回。
        if isinstance(nt, (int, float)):
            # 虽然这不完全符合 NestedTensors 的类型定义，
            # 但 MultiModalKwargs 的值里有时确实会直接出现 float。
            return nt
        # 其余情况按嵌套容器递归编码。
        return [self._encode_nested_tensors(x) for x in nt]

    # 编码多模态字段描述对象。
    def _encode_mm_field(self, field: BaseMultiModalField):
        # 先根据字段具体类型查出对应的工厂方法名。
        name = MMF_CLASS_TO_FACTORY.get(field.__class__)
        # 若字段类型未注册到映射表中，则直接报错。
        if not name:
            raise TypeError(f"Unsupported field type: {field.__class__}")

        # 按 dataclass 字段顺序把所有字段值复制出来，
        # 解码端会用这些参数重新构造出同样的 field。
        factory_kw = {f.name: getattr(field, f.name) for f in dataclasses.fields(field)}
        # 返回 (工厂名, 工厂参数) 二元组。
        return name, factory_kw


class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Decoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.
    """

    def __init__(self, t: Any | None = None, share_mem: bool = True):
        self.share_mem = share_mem
        self.pin_tensors = is_pin_memory_available()
        args = () if t is None else (t,)
        self.decoder = msgpack.Decoder(
            *args, ext_hook=self.ext_hook, dec_hook=self.dec_hook
        )
        self.aux_buffers: Sequence[bytestr] = ()
        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            _log_insecure_serialization_warning()

    def decode(self, bufs: bytestr | Sequence[bytestr]) -> Any:
        if isinstance(bufs, bytestr):  # type: ignore
            return self.decoder.decode(bufs)

        self.aux_buffers = bufs
        try:
            return self.decoder.decode(bufs[0])
        finally:
            self.aux_buffers = ()

    def dec_hook(self, t: type, obj: Any) -> Any:
        # Given native types in `obj`, convert to type `t`.
        if isclass(t):
            if issubclass(t, np.ndarray):
                return self._decode_ndarray(obj)
            if issubclass(t, torch.Tensor):
                return self._decode_tensor(obj)
            if t is slice:
                return slice(*obj)
            if issubclass(t, MultiModalKwargsItem):
                return self._decode_mm_item(obj)
            if issubclass(t, MultiModalKwargsItems):
                return self._decode_mm_items(obj)
            if t is UtilityResult:
                return self._decode_utility_result(obj)
        return obj

    def _decode_utility_result(self, obj: Any) -> UtilityResult:
        result_type, result = obj
        if result_type is not None:
            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                raise TypeError(
                    "VLLM_ALLOW_INSECURE_SERIALIZATION must "
                    "be set to use custom utility result types"
                )
            # Use recursive decoding to handle nested structures
            result = _decode_type_info_recursive(
                result_type, result, self._convert_result
            )
        return UtilityResult(result)

    def _convert_result(self, result_type: Sequence[str], result: Any) -> Any:
        if result_type is None:
            return result
        mod_name, name = result_type
        mod = importlib.import_module(mod_name)
        result_type = getattr(mod, name)
        return msgspec.convert(result, result_type, dec_hook=self.dec_hook)

    def _decode_ndarray(self, arr: Any) -> np.ndarray:
        dtype, shape, data = arr
        # zero-copy decode. We assume the ndarray will not be kept around,
        # as it now locks the whole received message buffer in memory.
        buffer = self.aux_buffers[data] if isinstance(data, int) else data
        arr = np.frombuffer(buffer, dtype=dtype)
        if not self.share_mem:
            arr = arr.copy()
        return arr.reshape(shape)

    def _decode_tensor(self, arr: Any) -> torch.Tensor:
        dtype, shape, data = arr
        is_aux = isinstance(data, int)
        buffer = self.aux_buffers[data] if is_aux else data
        buffer = buffer if isinstance(buffer, memoryview) else memoryview(buffer)
        torch_dtype = getattr(torch, dtype)
        assert isinstance(torch_dtype, torch.dtype)
        if not buffer.nbytes:  # torch.frombuffer doesn't like empty buffers
            assert 0 in shape
            return torch.empty(shape, dtype=torch_dtype)
        # Create uint8 array
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        # Clone ensures tensor is backed by pytorch-owned memory for safe
        # future async CPU->GPU transfer.
        # Pin larger tensors for more efficient CPU->GPU transfer.
        if not is_aux:
            arr = arr.clone()
        elif not self.share_mem:
            arr = arr.pin_memory() if self.pin_tensors else arr.clone()
        # Convert back to proper shape & type
        return arr.view(torch_dtype).view(shape)

    def _decode_mm_items(self, obj: dict[str, Any]) -> MultiModalKwargsItems:
        return MultiModalKwargsItems(
            {
                modality: [self._decode_mm_item(item) for item in itemlist]
                for modality, itemlist in obj.items()
            }
        )

    def _decode_mm_item(self, obj: dict[str, Any]) -> MultiModalKwargsItem:
        return MultiModalKwargsItem(
            {key: self._decode_mm_field_elem(elem) for key, elem in obj.items()}
        )

    def _decode_mm_field_elem(self, obj: dict[str, Any]) -> MultiModalFieldElem:
        if obj["data"] is not None:
            obj["data"] = self._decode_nested_tensors(obj["data"])

        # Reconstruct the field processor using MultiModalFieldConfig
        factory_meth_name, factory_kw = obj["field"]
        factory_meth = getattr(MultiModalFieldConfig, factory_meth_name)

        # Special case: decode the union "slices" field of
        # MultiModalFlatField
        if factory_meth_name == "flat":
            factory_kw["slices"] = self._decode_nested_slices(factory_kw["slices"])

        obj["field"] = factory_meth("", **factory_kw).field
        return MultiModalFieldElem(**obj)

    def _decode_nested_tensors(self, obj: Any) -> NestedTensors:
        if isinstance(obj, (int, float)):
            # Although it violates NestedTensors type, MultiModalKwargs
            # values are sometimes floats.
            return obj
        if not isinstance(obj, list):
            raise TypeError(f"Unexpected NestedTensors contents: {type(obj)}")
        if obj and isinstance(obj[0], str):
            return self._decode_tensor(obj)
        return [self._decode_nested_tensors(x) for x in obj]

    def _decode_nested_slices(self, obj: Any) -> Any:
        assert isinstance(obj, (list, tuple))
        if obj and not isinstance(obj[0], (list, tuple)):
            return slice(*obj)
        return [self._decode_nested_slices(x) for x in obj]

    def ext_hook(self, code: int, data: memoryview) -> Any:
        if code == CUSTOM_TYPE_RAW_VIEW:
            return data

        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            if code == CUSTOM_TYPE_PICKLE:
                return pickle.loads(data)
            if code == CUSTOM_TYPE_CLOUDPICKLE:
                return cloudpickle.loads(data)

        raise NotImplementedError(f"Extension type code {code} is not supported")


def run_method(
    obj: Any,
    method: str | bytes | Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """
    Run a method of an object with the given arguments and keyword arguments.
    If the method is string, it will be converted to a method using getattr.
    If the method is serialized bytes and will be deserialized using
    cloudpickle.
    If the method is a callable, it will be called directly.
    """
    if isinstance(method, bytes):
        func = partial(cloudpickle.loads(method), obj)
    elif isinstance(method, str):
        try:
            func = getattr(obj, method)
        except AttributeError:
            raise NotImplementedError(
                f"Method {method!r} is not implemented."
            ) from None
    else:
        func = partial(method, obj)  # type: ignore
    return func(*args, **kwargs)


class PydanticMsgspecMixin:
    """Make a ``msgspec.Struct`` compatible with Pydantic for both
    **validation** (JSON/dict -> Struct) and **serialization**
    (Struct -> JSON-safe dict).

    Subclasses may set ``__pydantic_msgspec_exclude__`` (a ``set[str]``)
    to list non-underscore field names that should also be stripped from
    serialized output.  Fields whose names start with ``_`` are always
    excluded automatically.
    """

    # Subclasses can override to exclude additional public-but-internal keys.
    __pydantic_msgspec_exclude__: ClassVar[set[str]] = set()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """
        Make msgspec.Struct compatible with Pydantic, respecting defaults.
        Handle JSON=>msgspec.Struct. Used when exposing msgspec.Struct to the
        API as input or in `/docs`. Note this is cached by Pydantic and not
        called on every validation.
        """
        msgspec_fields = {f.name: f for f in msgspec.structs.fields(source_type)}
        type_hints = get_type_hints(source_type)

        # Build the Pydantic typed_dict_field for each msgspec field
        fields = {}
        for name, hint in type_hints.items():
            if name not in msgspec_fields:
                # Skip ClassVar and other non-struct annotations.
                continue
            # Skip private fields — they are excluded from serialization
            # and should not appear in the generated JSON/OpenAPI schema.
            if name.startswith("_"):
                continue
            msgspec_field = msgspec_fields[name]

            # typed_dict_field using the handler to get the schema
            field_schema = handler(hint)

            # Add default value to the schema.
            # Mark fields with defaults as not required so the generated
            # JSON Schema stays consistent with ``omit_defaults=True``
            # serialization (fields at their default value may be absent).
            if msgspec_field.default_factory is not msgspec.NODEFAULT:
                wrapped_schema = core_schema.with_default_schema(
                    schema=field_schema,
                    default_factory=msgspec_field.default_factory,
                )
                fields[name] = core_schema.typed_dict_field(
                    wrapped_schema, required=False
                )
            elif msgspec_field.default is not msgspec.NODEFAULT:
                wrapped_schema = core_schema.with_default_schema(
                    schema=field_schema,
                    default=msgspec_field.default,
                )
                fields[name] = core_schema.typed_dict_field(
                    wrapped_schema, required=False
                )
            else:
                # No default, so Pydantic will treat it as required
                fields[name] = core_schema.typed_dict_field(field_schema)
        typed_dict_then_convert = core_schema.no_info_after_validator_function(
            cls._validate_msgspec,
            core_schema.typed_dict_schema(fields),
        )

        # Build a serializer that strips private / excluded fields.
        serializer = core_schema.plain_serializer_function_ser_schema(
            cls._serialize_msgspec,
            info_arg=False,
        )

        # Accept either an already-constructed msgspec.Struct instance or a
        # JSON/dict-like payload.
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(source_type),
                typed_dict_then_convert,
            ],
            serialization=serializer,
        )

    @classmethod
    def _validate_msgspec(cls, value: Any) -> Any:
        """Validate and convert input to msgspec.Struct instance."""
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        return msgspec.convert(value, type=cls)

    @staticmethod
    def _serialize_msgspec(value: Any) -> Any:
        """Serialize a msgspec.Struct to a JSON-compatible dict, stripping
        private (``_``-prefixed) and explicitly excluded fields.

        Uses ``msgspec.to_builtins`` which respects ``omit_defaults=True``,
        so only fields that differ from their declared defaults are included.
        """
        raw = msgspec.to_builtins(value)
        if not isinstance(raw, dict):
            return raw

        exclude: set[str] = cast(
            set[str],
            getattr(type(value), "__pydantic_msgspec_exclude__", set()),
        )
        for key in list(raw):
            if key.startswith("_") or key in exclude:
                del raw[key]

        return raw
