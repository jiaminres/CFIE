"""CFIE package."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import typing
import warnings

from .version import __version__, __version_tuple__


_WINDOWS_DLL_DIRECTORIES: list[os.PathLike[str] | object] = []


def _iter_windows_dll_dirs() -> list[Path]:
    candidates: list[Path] = []
    package_root = Path(__file__).resolve().parent

    # Editable installs place extension modules directly under the package tree.
    candidates.extend([package_root, package_root / "vllm_flash_attn"])

    for env_name in ("CUDA_PATH", "CUDA_HOME"):
        raw_path = os.environ.get(env_name)
        if raw_path:
            candidates.append(Path(raw_path) / "bin")

    default_cuda_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if default_cuda_root.exists():
        cuda_versions = sorted(
            (entry for entry in default_cuda_root.iterdir() if entry.is_dir()),
            reverse=True,
        )
        if cuda_versions:
            candidates.append(cuda_versions[0] / "bin")

    def add_package_bin(module_name: str, suffix: str = "bin") -> None:
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            return
        if spec is None or not spec.submodule_search_locations:
            return
        for location in spec.submodule_search_locations:
            candidates.append(Path(location) / suffix)

    add_package_bin("torch", "lib")
    add_package_bin("nvidia.cublas")
    add_package_bin("nvidia.cuda_runtime")
    add_package_bin("nvidia.cuda_nvrtc")

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            normalized = os.path.normcase(str(candidate.resolve(strict=False)))
        except OSError:
            normalized = os.path.normcase(str(candidate))
        if normalized in seen or not candidate.is_dir():
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


def _configure_windows_dll_search_path() -> None:
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    for dll_dir in _iter_windows_dll_dirs():
        try:
            # Keep handles alive for the interpreter lifetime.
            _WINDOWS_DLL_DIRECTORIES.append(os.add_dll_directory(str(dll_dir)))
        except OSError:
            continue


_configure_windows_dll_search_path()

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
