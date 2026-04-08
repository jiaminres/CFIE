# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import inspect

import torch
import torch.nn as nn

from cfie.config import get_cached_compilation_config
from cfie.logger import init_logger
from cfie.model_executor.utils import maybe_disable_graph_partition
from cfie.platforms import current_platform

logger = init_logger(__name__)

# 所有自定义算子类的注册表（按“注册名”索引）
# value 一般是 CustomOp 子类或 PluggableLayer 子类
#
# 注意：
# 这里只是“已注册类”的字典，不代表它一定启用
# 某个 op 是否真正启用，需要调用类上的 .enabled() 判断
#
# 例如：
# - MyOp.enabled()
# - op_registry["my_op"].enabled()
op_registry: dict[str, type["CustomOp"] | type["PluggableLayer"]] = {}

# out-of-tree(OOT) 自定义算子注册表
# 这里保存的是“外部/插件式覆盖实现”
#
# 如果某个类名在 op_registry_oot 中有对应实现，
# 框架在实例化时会优先使用 OOT 版本，而不是内建版本
op_registry_oot: dict[str, type["CustomOp"] | type["PluggableLayer"]] = {}


def get_oot_class_by_name(class_name: str) -> type | None:
    # 根据类名查询是否存在对应的 OOT 覆盖实现

    # 如果在 OOT 注册表中找到了，就返回对应类
    if class_name in op_registry_oot:
        return op_registry_oot[class_name]

    # 否则返回 None，表示没有 OOT 覆盖实现
    return None


class PluggableLayer(nn.Module):
    """
    Base class for pluggable layers.

    A PluggableLayer is a *module-composing* abstraction: it may instantiate other
    ``torch.nn.Module`` objects as sub-layers, and its functionality depends on
    these sub-layers following a generalized invocation sequence. Also, it is stateful
    and may hold parameters or buffers.

    Unlike :class:`CustomOp`, PluggableLayer does NOT provide per-platform
    ``forward_*`` dispatch. Instead, it supports out-of-tree (OOT) replacement
    of the entire layer class at instantiation time, allowing customized
    initialization and submodule composition.
    """

    def __new__(cls, *args, **kwargs):
        try:
            layer_class_name = cls.__name__
        except AttributeError:
            raise TypeError(
                f"Cannot instantiate '{cls.__name__}': its 'name' attribute "
                f"was not set, possibly because it was not decorated with "
                f"@PluggableLayer.register, or it's the PluggableLayer itself."
            ) from None

        if layer_class_name not in op_registry_oot:
            layer_cls_to_instantiate = cls
        else:
            layer_cls_to_instantiate = op_registry_oot[layer_class_name]
            logger.debug(
                "Instantiating pluggable layer: %s using %s",
                layer_class_name,
                str(layer_cls_to_instantiate),
            )
        return super().__new__(layer_cls_to_instantiate)

    # Decorator to register pluggable layers.
    @classmethod
    def register(cls, name: str):
        def decorator(op_cls):
            assert name not in op_registry, f"Duplicate op name: {name}"
            op_cls.name = name
            op_registry[name] = op_cls
            return op_cls

        return decorator

    # Decorator to register out-of-tree(oot) pluggable layers.
    # For OOT pluggable layers:
    #   if in-tree layer class is registered with an oot_custom_layer,
    #   the oot_custom_layer will be used instead.
    @classmethod
    def register_oot(cls, _decorated_layer_cls=None, name: str | None = None):
        def decorator(layer_cls):
            reg_name = name if name is not None else cls.__name__
            assert reg_name not in op_registry_oot, f"Duplicate layer name: {reg_name}"
            layer_cls.name = reg_name
            op_registry_oot[reg_name] = layer_cls
            return layer_cls

        if _decorated_layer_cls is None:
            # Called with parentheses: @PluggableLayer.register_oot()
            # or @PluggableLayer.register_oot(name="...")
            return decorator
        elif isinstance(_decorated_layer_cls, type):  # Check if it's a class
            # Called without parentheses: @PluggableLayer.register_oot
            return decorator(_decorated_layer_cls)
        else:
            raise TypeError("Decorator can only be applied to classes.")


class CustomOp(nn.Module):
    """
    自定义算子的基类。
    负责根据当前运行后端（CUDA / CPU / XPU / TPU / OOT 等）
    分发到对应的 forward 实现。
    """

    def __new__(cls, *args, **kwargs):
        # 在实例化对象之前，先决定“真正要实例化哪个类”
        try:
            # 这里要求类上应该已经有一个可识别的 op 名称
            # 实际上后面 register/register_oot 会给类设置 name
            op_name = cls.__name__
        except AttributeError:
            raise TypeError(
                f"Cannot instantiate '{cls.__name__}': its 'name' attribute "
                f"was not set, possibly because it was not decorated with "
                f"@CustomOp.register, or it's the CustomOp base class itself."
            ) from None

        # 如果这个 op_name 没有在 out-of-tree 注册表中
        # 就实例化当前类本身
        if op_name not in op_registry_oot:
            op_cls_to_instantiate = cls
        else:
            # 否则优先实例化 OOT（out-of-tree）版本
            # 即外部平台/插件覆盖的实现
            op_cls_to_instantiate = op_registry_oot[op_name]
            logger.debug(
                "Instantiating custom op: %s using %s",
                op_name,
                str(op_cls_to_instantiate),
            )

        # 真正创建实例对象
        return super().__new__(op_cls_to_instantiate)

    def __init__(self, *, enforce_enable: bool = False, compile_native: bool = False):
        # 初始化 nn.Module
        super().__init__()

        # 是否强制启用该自定义算子
        self._enforce_enable = enforce_enable

        # 根据当前平台和配置，提前选好 forward 分发方法
        self._forward_method = self.dispatch_forward(compile_native=compile_native)

    def forward(self, *args, **kwargs):
        # forward 统一调用已经分发好的后端实现
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        """PyTorch 原生实现。
        这是可选的。如果实现了：
        - 可用于 torch.compile
        - 可用于 PyTorch XLA
        - 可用于测试/回退路径
        """
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        # CUDA 后端实现，默认要求子类自己实现
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        # 默认认为 HIP(ROCm) 实现与 CUDA 兼容
        return self.forward_cuda(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        # 默认认为 XPU 实现可回退到 PyTorch 原生实现
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        # 默认认为 CPU 实现可回退到 PyTorch 原生实现
        return self.forward_native(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs):
        # 默认认为 TPU 实现可回退到 PyTorch 原生实现
        # 目前这里只是预留扩展点
        return self.forward_native(*args, **kwargs)

    def forward_oot(self, *args, **kwargs):
        # 默认认为 OOT(out-of-tree) 实现可回退到 PyTorch 原生实现
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self, compile_native: bool):
        # 根据当前平台和配置，决定 forward 应该绑定到哪个实现
        # 注意：这里假设 vLLM/cfie 只为一个固定后端构建，不支持动态多后端切换

        compilation_config = get_cached_compilation_config()

        # enforce_enable 允许强制启用某个 device-specific kernel
        # 默认情况下是否启用，由配置和 enabled() 决定
        enabled = self._enforce_enable or self.enabled()

        # 记录这个 custom op 是启用还是禁用，便于编译配置跟踪
        if enabled:
            compilation_config.enabled_custom_ops.update([self.__class__.name])
        else:
            compilation_config.disabled_custom_ops.update([self.__class__.name])

        # 如果当前算子未启用，则走 native 路径
        if not enabled:
            # 如果处于某些不透明torch custom op 内部，
            # model-level torch.compile 可能看不到这层 forward
            # 所以这里可选择单独 compile forward_native
            return self.maybe_compile(self.forward_native, enable=compile_native)

        # 按运行平台分发到对应实现
        if current_platform.is_rocm():
            return self.forward_hip
        elif current_platform.is_cpu():
            return self.forward_cpu
        elif current_platform.is_tpu():
            return self.forward_tpu
        elif current_platform.is_xpu():
            return self.forward_xpu
        elif current_platform.is_out_of_tree():
            return self.forward_oot
        else:
            # 默认走 CUDA
            return self.forward_cuda

    def maybe_compile(self, fn, *, enable: bool = True):
        """
        在需要时对 fn 做 torch.compile。

        适用场景：
        某个 CustomOp 是在另一个 torch custom op 内部被调用的，
        这样 model-level torch.compile 可能无法看到它，
        所以这里提供局部 compile 的机会。

        注意：
        这种 compile 不能跨 opaque custom op 做融合，
        所以仍然应尽量减少不透明 custom op 的包裹。
        """
        from cfie.config.compilation import CompilationMode

        # 如果显式关闭，则不编译
        if not enable:
            return fn

        # 如果全局 compilation mode 关闭，也不编译
        compilation_config = get_cached_compilation_config()
        if compilation_config.mode == CompilationMode.NONE:
            return fn

        # 如果 backend 是 eager，也不编译
        if compilation_config.backend == "eager":
            return fn

        # 获取 compile 选项，并根据平台可能关闭 graph partition
        compile_options = maybe_disable_graph_partition(
            current_platform.simple_compile_backend
        )
        backend = current_platform.simple_compile_backend

        # dynamic_arg_dims 用于标记哪些输入维度是动态维
        dynamic_arg_dims = getattr(self.__class__, "_dynamic_arg_dims", None)

        if dynamic_arg_dims is not None:
            # 若显式给了动态维规则，则这里先用 dynamic=False 编译，
            # 再在 wrapper 中手动 mark_dynamic
            compiled_fn = torch.compile(
                fn,
                dynamic=False,
                backend=backend,
                options=compile_options,
            )
            sig = inspect.signature(fn)

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                # 绑定实参到函数签名，便于按名字找到对应参数
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                # 对指定参数的指定维度做动态维标记
                for name, dims in dynamic_arg_dims.items():
                    arg = bound.arguments.get(name)
                    if arg is not None and isinstance(arg, torch.Tensor):
                        dims_list = [dims] if isinstance(dims, int) else dims
                        for d in dims_list:
                            real_d = arg.ndim + d if d < 0 else d
                            torch._dynamo.mark_dynamic(arg, real_d)

                return compiled_fn(*args, **kwargs)

            return wrapper

        # 若未指定动态维规则，则用 dynamic=True 避免频繁重编译
        return torch.compile(
            fn,
            dynamic=True,
            backend=backend,
            options=compile_options,
        )

    @classmethod
    def enabled(cls) -> bool:
        # 判断当前 custom op 是否启用
        # 依据 compilation_config.custom_ops 配置决定

        compilation_config = get_cached_compilation_config()
        custom_ops = compilation_config.custom_ops

        # 如果类上没有 name，说明它没有通过 register 装饰器注册
        if not hasattr(cls, "name"):
            logger.warning_once(
                "Custom op %s was not registered, which means it won't appear "
                "in the op registry. It will be enabled/disabled based on the "
                "global settings.",
                cls.__name__,
            )
            return CustomOp.default_on()

        # 支持配置中显式 +op_name 开启，-op_name 关闭
        enabled = f"+{cls.name}" in custom_ops
        disabled = f"-{cls.name}" in custom_ops

        # 不能同时显式开启和关闭
        assert not (enabled and disabled), f"Cannot enable and disable {cls.name}"

        # 启用逻辑：
        # (默认开启 或 显式开启) 且 没有被显式关闭
        return (CustomOp.default_on() or enabled) and not disabled

    @staticmethod
    def default_on() -> bool:
        """
        默认是否开启 custom op。

        由 CompilationConfig.custom_ops 控制：
        - 若包含 'all'：默认开启
        - 若包含 'none'：默认关闭

        通常在使用 PyTorch Inductor 时，默认会是 'none'；
        否则一般默认 'all'
        """
        compilation_config = get_cached_compilation_config()
        count_none = compilation_config.custom_ops.count("none")
        count_all = compilation_config.custom_ops.count("all")

        # 要求二者恰好出现一个
        assert count_none + count_all == 1

        return not count_none > 0 or count_all > 0

    # 注册 in-tree custom op 的装饰器
    @classmethod
    def register(
        cls,
        name: str,  # 注册名
        dynamic_arg_dims: dict[str, int | list[int]] | None = None,  # 动态维规则
    ):
        def decorator(op_cls):
            # 注册名不能重复
            assert name not in op_registry, f"Duplicate op name: {name}"

            # 给类设置名字
            op_cls.name = name

            # 保存动态维配置信息
            op_cls._dynamic_arg_dims = dynamic_arg_dims

            # 加入全局 op 注册表
            op_registry[name] = op_cls
            return op_cls

        return decorator

    # 注册 out-of-tree(OOT) custom op 的装饰器
    # OOT 的含义是：如果某个 in-tree 层类有对应的 OOT 替代实现，
    # 则优先使用 OOT 实现
    @classmethod
    def register_oot(cls, _decorated_op_cls=None, name: str | None = None):
        def decorator(op_cls):
            # 若未显式传 name，则默认用当前类名
            reg_name = name if name is not None else cls.__name__

            # 注册名不能重复
            assert reg_name not in op_registry_oot, f"Duplicate op name: {reg_name}"

            # 给类设置名字
            op_cls.name = reg_name

            # 加入 OOT 注册表
            op_registry_oot[reg_name] = op_cls
            return op_cls

        if _decorated_op_cls is None:
            # 带括号写法：
            # @CustomOp.register_oot()
            # @CustomOp.register_oot(name="...")
            # 此时先返回真正的 decorator
            return decorator
        elif isinstance(_decorated_op_cls, type):
            # 不带括号写法：
            # @CustomOp.register_oot
            # 此时传进来的第一个参数就是类本身，直接装饰
            return decorator(_decorated_op_cls)
        else:
            # 非法用法
            raise TypeError("Decorator can only be applied to classes.")
