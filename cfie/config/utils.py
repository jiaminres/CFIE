# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for vLLM config dataclasses."""

import ast
import enum
import hashlib
import inspect
import json
import os
import pathlib
import textwrap
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import MISSING, field, fields, is_dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from pydantic.fields import Field as PydanticField
from pydantic.fields import FieldInfo
from typing_extensions import dataclass_transform, runtime_checkable

import cfie.envs as envs
from cfie.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
else:
    DataclassInstance = Any

ConfigType = type[DataclassInstance]
ConfigT = TypeVar("ConfigT", bound=DataclassInstance)


# 告诉类型检查器：被 `@config` 修饰后的类应按 dataclass 规则理解。
@dataclass_transform(field_specifiers=(PydanticField,))
def config(
    # 直接被装饰的类；无参使用 `@config` 时会传进来。
    cls: type[ConfigT] | None = None,
    *,
    # 可选的 Pydantic ConfigDict，会与默认配置合并。
    config: ConfigDict | None = None,
    # 透传给 `pydantic.dataclasses.dataclass` 的其他参数。
    **kwargs: Any,
) -> type[ConfigT] | Callable[[type[ConfigT]], type[ConfigT]]:

    # 先构造默认配置：所有 config dataclass 默认禁止未声明字段。
    # 所有配置类默认都禁止未声明字段。
    merged_config = ConfigDict(extra="forbid")
    if config is not None:
        # 若调用方显式传入配置，则叠加到默认配置上。
        merged_config.update(config)

    def decorator(cls):
        # 把普通类转换成带 Pydantic 校验能力的 dataclass。
        # 真正把普通类包装成带 Pydantic 校验能力的 dataclass。
        return dataclass(cls, config=merged_config, **kwargs)

    # Called with arguments: @config(config=...)
    if cls is None:
        # 有参装饰器场景，先返回闭包，等待后续接收类对象。
        return decorator
    # Called without arguments: @config
    # 无参装饰器场景，立即对类执行包装。
    return decorator(cls)


def get_field(cls: ConfigType, name: str) -> Any:
    """Get the default factory field of a dataclass by name. Used for getting
    default factory fields in `EngineArgs`."""
    if not is_dataclass(cls):
        raise TypeError("The given class is not a dataclass.")
    try:
        # 在 dataclass 字段列表里按名字查找目标字段。
        named_field = next(f for f in fields(cls) if f.name == name)
    except StopIteration as e:
        raise ValueError(f"Field '{name}' not found in {cls.__name__}.") from e

    # 先取出 dataclass 原生 default/default_factory/init 信息。
    # The arguments to copy to the new field
    default = named_field.default
    default_factory = named_field.default_factory
    init = named_field.init

    # 若默认值本身是 pydantic.Field，还需要把其中的信息拆出来。
    # Handle pydantic.Field
    if isinstance(default, FieldInfo):
        if default.init is not None:
            init = default.init
        if default.default_factory is not None:
            default_factory = cast(Callable[[], Any], default.default_factory)
            default = MISSING
        else:
            default = default.default

    # 如果既没有默认值也没有 default_factory，则打印一次 warning 方便排查。
    if default is MISSING and default_factory is MISSING:
        logger.warning_once(
            "%s.%s has no default or default factory.", cls.__name__, name
        )
    # 重新构造一个等价 field 对象并返回。
    return field(default=default, default_factory=default_factory, init=init)


def is_init_field(cls: ConfigType, name: str) -> bool:
    # 复用 get_field 的结果，判断这个字段是否属于 __init__ 参数。
    return get_field(cls, name).init


def replace(dataclass_instance: ConfigT, /, **kwargs) -> ConfigT:
    """Like [`dataclasses.replace`](https://docs.python.org/3/library/dataclasses.html#dataclasses.replace),
    but compatible with Pydantic dataclasses which use `pydantic.fields.Field` instead
    of `dataclasses.field`"""
    # 取出实例类型。
    cls = type(dataclass_instance)
    # 先读取实例当前的全部字段值。
    dataclass_dict = dataclass_instance.__dict__
    # 仅保留真正属于 __init__ 参数的字段，兼容 pydantic dataclass。
    dataclass_dict = {k: v for k, v in dataclass_dict.items() if is_init_field(cls, k)}
    # 用调用方传入的 overrides 覆盖对应字段。
    dataclass_dict.update(kwargs)
    # 重新构造一个同类型新实例。
    return cls(**dataclass_dict)


def getattr_iter(
    object: object,
    names: Sequence[str],
    default: Any | None = None,
    default_factory: Callable[[], Any] | None = None,
    warn: bool = False,
) -> Any:
    """
    A helper function that retrieves an attribute from an object which may
    have multiple possible names. This is useful when fetching attributes from
    arbitrary `transformers.PretrainedConfig` instances.

    In the case where the first name in `names` is the preferred name, and
    any other names are deprecated aliases, setting `warn=True` will log a
    warning when a deprecated name is used.
    """
    # 依次尝试多个可能的属性名，直到命中为止。
    for i, name in enumerate(names):
        if hasattr(object, name):
            if warn and i > 0:
                # 若命中的不是首选名，则视为走到了 deprecated alias，并给出一次 warning。
                logger.warning_once(
                    "%s contains a deprecated attribute name '%s'. "
                    "Please use the preferred attribute name '%s' instead.",
                    type(object).__name__,
                    name,
                    names[0],
                )
            return getattr(object, name)
    # 全部未命中时，优先调用 default_factory，否则退回 default。
    return default_factory() if default_factory is not None else default


def get_attr_docs(cls: type[Any]) -> dict[str, str]:
    """
    Get any docstrings placed after attribute assignments in a class body.

    https://davidism.com/mit-license/
    """

    # 读取类源码并解析成 AST。
    cls_node = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0]

    if not isinstance(cls_node, ast.ClassDef):
        raise TypeError("Given object was not a class.")

    # 输出结果：字段名 -> 紧随其后的文档字符串。
    out = {}

    # Consider each pair of nodes.
    for a, b in pairwise(cls_node.body):
        # Must be an assignment then a constant string.
        if (
            not isinstance(a, (ast.Assign, ast.AnnAssign))
            or not isinstance(b, ast.Expr)
            or not isinstance(b.value, ast.Constant)
            or not isinstance(b.value.value, str)
        ):
            continue

        # 清理 docstring 的缩进。
        doc = inspect.cleandoc(b.value.value)

        # An assignment can have multiple targets (a = b = v), but an
        # annotated assignment only has one target.
        targets = a.targets if isinstance(a, ast.Assign) else [a.target]

        for target in targets:
            # Must be assigning to a plain name.
            if not isinstance(target, ast.Name):
                continue

            # 记录这个属性对应的文档字符串。
            out[target.id] = doc

    return out


@runtime_checkable
class SupportsHash(Protocol):
    # 约定实现方提供 compute_hash，用于配置哈希或 cache key。
    def compute_hash(self) -> str: ...


class SupportsMetricsInfo(Protocol):
    # 约定实现方提供 metrics_info，用于导出指标标签信息。
    def metrics_info(self) -> dict[str, str]: ...


def update_config(config: ConfigT, overrides: dict[str, Any]) -> ConfigT:
    # 先准备一个处理过的 overrides 字典。
    processed_overrides = {}
    for field_name, value in overrides.items():
        # 所有 override 字段都必须存在于目标 config 上。
        assert hasattr(config, field_name), (
            f"{type(config)} has no field `{field_name}`"
        )
        current_value = getattr(config, field_name)
        # 若当前字段本身还是 dataclass，而 override 传的是 dict，则递归更新。
        if is_dataclass(current_value) and not is_dataclass(value):
            assert isinstance(value, dict), (
                f"Overrides to {type(config)}.{field_name} must be a dict"
                f"  or {type(current_value)}, but got {type(value)}"
            )
            value = update_config(
                current_value,  # type: ignore[type-var]
                value,
            )
        processed_overrides[field_name] = value
    # 复用上面的 replace 构造更新后的 config 副本。
    return replace(config, **processed_overrides)


def normalize_value(x):
    """Return a stable, JSON-serializable canonical form for hashing.
    Order: primitives, special types (Enum, callable, torch.dtype, Path), then
    generic containers (Mapping/Set/Sequence) with recursion.
    """
    # Fast path
    if x is None or isinstance(x, (bool, int, float, str)):
        return x

    # 枚举值需要带上类型全名，避免和普通整型/字符串值冲突。
    # Enums: tag with FQN to avoid primitive collisions.
    # Ex: Enum(1) vs int(1) -> ("module.QualName", value).
    if isinstance(x, enum.Enum):
        enum_type = f"{x.__class__.__module__}.{x.__class__.__qualname__}"
        return (enum_type, normalize_value(x.value))

    # Classes (types) are accepted and canonicalized by their fully-qualified
    # name (module.qualname) for a stable identifier.
    # Instances are only accepted if they expose uuid(); otherwise they are
    # rejected to avoid under-hashing object state.

    # 类对象按 module.qualname 归一化，保证稳定可哈希。
    # Callables: accept classes only; reject funcs/lambdas/methods.
    # Used by LogitsProcessor types and ModelConfig.hf_overrides.
    if isinstance(x, type):
        module = getattr(x, "__module__", "")
        qual = getattr(x, "__qualname__", getattr(x, "__name__", ""))
        return ".".join([p for p in (module, qual) if p]) or repr(x)

    # 若对象提供 uuid()，优先用它作为稳定标识。
    # Prefer stable uuid identifiers for objects that provide them, even if
    # they are callable instances (e.g., InductorPass wrappers).
    if hasattr(x, "uuid") and callable(getattr(x, "uuid", None)):
        return x.uuid()

    # 其余 callable 实例/函数默认不支持，以免漏掉内部状态。
    if callable(x):
        raise TypeError("normalize_value: function or callable instance unsupported")

    # torch.dtype 直接转成字符串。
    # Torch dtype: stringify (torch.float64 -> "torch.float64").
    # We rely on the string form here; dtype-bearing fields that need additional
    # disambiguation should encode that at the config layer.
    if isinstance(x, torch.dtype):
        return str(x)

    # bytes/bytearray 转成十六进制字符串。
    # Bytes
    if isinstance(x, (bytes, bytearray)):
        return x.hex()

    # Path 尽量解析成绝对规范路径。
    # Paths (canonicalize)
    if isinstance(x, pathlib.Path):
        try:
            return str(x.expanduser().resolve())
        except Exception:
            return str(x)

    # dataclass 统一表示成 (类型全名, 排序后的字段值元组)。
    # Dataclasses: represent as (FQN, sorted(field,value) tuple) for stability.
    if is_dataclass(x):
        type_fqn = f"{x.__class__.__module__}.{x.__class__.__qualname__}"
        items = tuple(
            (f.name, normalize_value(getattr(x, f.name)))
            for f in sorted(fields(x), key=lambda f: f.name)
        )
        return (type_fqn, items)

    # Mapping / Set / Sequence 都递归归一化，保证顺序稳定。
    # Containers (generic)
    if isinstance(x, Mapping):
        return tuple(sorted((str(k), normalize_value(v)) for k, v in x.items()))
    if isinstance(x, Set):
        return tuple(sorted(repr(normalize_value(v)) for v in x))
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        return tuple(normalize_value(v) for v in x)

    # Hugging Face PretrainedConfig 走 to_json_string，避免展开复杂内部对象。
    # PretrainedConfig
    if hasattr(x, "to_json_string") and callable(x.to_json_string):
        return x.to_json_string()

    # Unsupported type: e.g., modules, generators, open files, or objects
    # without a stable JSON/UUID representation. Hard-error to avoid
    # under-hashing.
    # If you hit this, either reshape your config to use supported primitives
    # and containers, or extend normalize_value to provide a stable encoding
    # (e.g., via uuid() or to_json_string()) for this type.
    raise TypeError(
        f"normalize_value: unsupported type '{type(x).__name__}'. "
        "Ensure config values use supported primitives/containers or add a "
        "stable representation for this type."
    )


def get_hash_factors(config: ConfigT, ignored_factors: set[str]) -> dict[str, object]:
    """Gets the factors used for hashing a config class.
    - Includes all dataclass fields not in `ignored_factors`.
    - Errors on non-normalizable values.
    """
    # 遍历 dataclass 字段，收集所有参与哈希的规范化值。
    factors: dict[str, object] = {}
    for dc_field in fields(config):
        factor = dc_field.name
        # 明确在 ignored_factors 里的字段直接跳过。
        if factor in ignored_factors:
            continue
        value = getattr(config, factor, None)
        try:
            # 每个字段都要先过 normalize_value，确保哈希稳定。
            factors[factor] = normalize_value(value)
        except TypeError as e:
            raise TypeError(
                f"get_hash_factors: unsupported type for key '{factor}' "
                f"({type(value).__name__})"
            ) from e
    return factors


def hash_factors(items: dict[str, object]) -> str:
    """Return a SHA-256 hex digest of the canonical items structure."""
    # 统一对 JSON 序列化后的 canonical 结构做 SHA-256。
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


@dataclass
class Range:
    """
    A range of numbers.
    Inclusive of start, inclusive of end.
    """

    # 区间起点。
    start: int
    # 区间终点。
    end: int

    def is_single_size(self) -> bool:
        # 起点等于终点时，这个区间只表示单个数值。
        return self.start == self.end

    def __contains__(self, size: int) -> bool:
        # 区间两端都按闭区间处理。
        # Inclusive of start, inclusive of end
        return self.start <= size <= self.end

    def __eq__(self, other: object) -> bool:
        # 仅当另一个对象也是 Range 且起止点都相等时，才视为相等。
        if not isinstance(other, Range):
            return False
        return self.start == other.start and self.end == other.end

    def __hash__(self) -> int:
        # 用起止点元组生成哈希。
        return hash((self.start, self.end))

    def __str__(self) -> str:
        # 统一输出成 `(start, end)` 字符串形式。
        return f"({self.start}, {self.end})"

    def __repr__(self) -> str:
        # repr 复用 __str__，保持展示一致。
        return self.__str__()


def handle_deprecated(
    config: ConfigT,
    old_name: str,
    new_name_or_names: str | list[str],
    removal_version: str,
) -> None:
    # 读取旧字段当前值；若为空则无需迁移。
    old_val = getattr(config, old_name)
    if old_val is None:
        return

    # 统一把“新字段名”整理成列表形式，兼容单字段或多字段迁移。
    if isinstance(new_name_or_names, str):
        new_names = [new_name_or_names]
    else:
        new_names = new_name_or_names

    # 先打印一次弃用 warning。
    msg = (
        f"{old_name} is deprecated and will be removed in {removal_version}. "
        f"Use {', '.join(new_names)} instead."
    )
    logger.warning(msg)

    # 再把旧值复制到所有新字段上。
    for new_name in new_names:
        setattr(config, new_name, old_val)


def get_from_deprecated_env_if_set(
    env_name: str,
    removal_version: str,
    field_name: str | None = None,
) -> str | None:
    """
    Get value from deprecated environment variable with warning.

    Args:
        env_name: Name of the deprecated environment variable
        removal_version: Version when it will be removed
        field_name: Name of the field to suggest as alternative

    Returns:
        The environment variable value if set, None otherwise
    """
    # 若旧环境变量已设置，则给出 warning 并返回其值。
    if envs.is_set(env_name):
        value = os.environ.get(env_name)
        alt_msg = f" Please use {field_name} instead." if field_name else ""
        logger.warning_once(
            "Using %s environment variable is deprecated and will be removed in %s.%s",
            env_name,
            removal_version,
            alt_msg,
        )
        return value
    # 未设置时返回 None。
    return None


def set_from_deprecated_env_if_set(
    config: ConfigT,
    env_name: str,
    removal_version: str,
    field_name: str,
    to_bool: bool = False,
    to_int: bool = False,
) -> None:
    """
    Set object field from deprecated environment variable with warning.

    Args:
        config: Config object to set the field on
        env_name: Name of the deprecated environment variable
        removal_version: Version when the env var will be removed
        field_name: Name of the field to set
        to_bool: Whether to convert the environment variable value to boolean
        to_int: Whether to convert the environment variable value to integer
    Returns:
        None
    """
    # 一个环境变量值只能按一种目标类型转换。
    if to_bool and to_int:
        raise ValueError("Cannot convert to both boolean and integer.")

    # 先读取旧环境变量值，并在命中时自动打印弃用 warning。
    env_value = get_from_deprecated_env_if_set(env_name, removal_version, field_name)
    if env_value is not None:
        field_value: str | bool | int = env_value
        # 按调用方要求转换成 bool。
        if to_bool:
            field_value = env_value.lower() in ("1", "true")
        # 或转换成 int。
        elif to_int:
            field_value = int(env_value)
        # 最终把转换后的值写回配置对象对应字段。
        setattr(config, field_name, field_value)
