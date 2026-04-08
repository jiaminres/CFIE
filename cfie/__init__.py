"""CFIE package."""

from __future__ import annotations

import typing
import warnings

from .version import __version__, __version_tuple__

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"The cuda\.cudart module is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"The cuda\.nvrtc module is deprecated.*",
)

MODULE_ATTRS = {
    "PoolingParams": ".pooling_params:PoolingParams",
    "SamplingParams": ".sampling_params:SamplingParams",
}

if typing.TYPE_CHECKING:
    from cfie.pooling_params import PoolingParams
    from cfie.sampling_params import SamplingParams
else:

    def __getattr__(name: str) -> typing.Any:
        from importlib import import_module

        if name in MODULE_ATTRS:
            module_name, attr_name = MODULE_ATTRS[name].split(":")
            module = import_module(module_name, __package__)
            return getattr(module, attr_name)
        raise AttributeError(f"module {__package__} has no attribute {name}")


__all__ = [
    "__version__",
    "__version_tuple__",
    "PoolingParams",
    "SamplingParams",
]
