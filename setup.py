from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys
import sysconfig
import ctypes
import functools
from pathlib import Path
from shutil import which

import torch
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import CUDA_HOME


def _raise_cuda_torch_required() -> None:
    torch_version = getattr(torch, "__version__", "unknown")
    torch_cuda = getattr(torch.version, "cuda", None)
    raise RuntimeError(
        "CFIE native build requires a CUDA-enabled PyTorch environment. "
        f"Detected torch={torch_version!s} with torch.version.cuda={torch_cuda!r}. "
        "If you already installed a CUDA wheel into your virtual environment, "
        "pip build isolation may be hiding it inside a temporary build env. "
        "Install a CUDA build of torch in your target environment and rerun "
        "`python -m pip install --no-build-isolation -e .`."
    )


def load_module_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ROOT_DIR = Path(__file__).parent.resolve()
envs = load_module_from_path("cfie_build_envs", str(ROOT_DIR / "cfie" / "envs.py"))


def to_cmake_path(path: str | os.PathLike[str]) -> str:
    return str(path).replace("\\", "/")


def default_fetchcontent_base_dir() -> Path:
    machine = platform.machine() or "unknown"
    platform_dir = f"{sys.platform}-{machine}".replace(" ", "_")
    return ROOT_DIR / ".deps" / platform_dir


def is_supported_platform() -> bool:
    return sys.platform.startswith("linux") or sys.platform.startswith("win")


def find_tool(name: str) -> str | None:
    script_dir = Path(sysconfig.get_path("scripts") or "")
    executable = f"{name}.exe" if os.name == "nt" else name
    script_tool = script_dir / executable
    if script_tool.exists():
        return str(script_tool)
    return which(executable) or which(name)


def iter_cuda_roots() -> list[Path]:
    candidates: list[Path] = []
    for raw_path in (
        CUDA_HOME,
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
    ):
        if raw_path:
            normalized = str(raw_path).strip()
            if normalized:
                candidates.append(Path(normalized))

    spec = importlib.util.find_spec("nvidia.cuda_nvcc")
    if spec and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            candidates.append(Path(location))

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = os.path.normcase(str(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


def resolve_cuda_home() -> Path | None:
    nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
    for candidate in iter_cuda_roots():
        if (candidate / "bin" / nvcc_name).exists():
            return candidate
    return None


def get_windows_short_path(path: Path) -> Path | None:
    if os.name != "nt":
        return None

    buffer = ctypes.create_unicode_buffer(32768)
    result = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
    if result == 0:
        return None
    short_path = buffer.value.strip()
    return Path(short_path) if short_path else None


def prepare_cuda_root(cuda_root: Path) -> Path:
    if os.name != "nt" or " " not in str(cuda_root):
        return cuda_root

    short_path = get_windows_short_path(cuda_root)
    if short_path is not None and " " not in str(short_path):
        return short_path

    version_suffix = torch.version.cuda or "cuda"
    alias_root = ROOT_DIR / ".deps" / "toolchains" / f"cuda-{version_suffix}"
    alias_root.parent.mkdir(parents=True, exist_ok=True)

    if alias_root.exists():
        try:
            if alias_root.resolve() == cuda_root.resolve():
                return alias_root
        except OSError:
            pass
        return cuda_root

    try:
        subprocess.check_call(
            ["cmd", "/c", "mklink", "/J", str(alias_root), str(cuda_root)]
        )
        return alias_root
    except (OSError, subprocess.CalledProcessError):
        return cuda_root


def _find_vcvars64_bat() -> Path | None:
    if os.name != "nt":
        return None

    direct_candidates: list[Path] = []
    vsinstalldir = os.environ.get("VSINSTALLDIR")
    if vsinstalldir:
        direct_candidates.append(
            Path(vsinstalldir) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        )

    default_vs_root = Path("C:/Program Files/Microsoft Visual Studio/2022")
    for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
        direct_candidates.append(
            default_vs_root / edition / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        )

    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if not program_files_x86:
        return None

    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None

    try:
        installation_path = subprocess.check_output(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

    if not installation_path:
        return None

    candidate = Path(installation_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    return candidate if candidate.is_file() else None


@functools.lru_cache(maxsize=1)
def _load_msvc_build_env() -> dict[str, str]:
    vcvars64 = _find_vcvars64_bat()
    if vcvars64 is None:
        return {}

    command = f'cmd /d /s /c ""{vcvars64}" >nul && set"'
    cmd_env = os.environ.copy()
    cmd_env.setdefault("VSCMD_SKIP_SENDTELEMETRY", "1")

    try:
        output = subprocess.check_output(
            command,
            text=True,
            env=cmd_env,
            shell=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}

    parsed_env: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            parsed_env[key] = value
    return parsed_env


if not is_supported_platform():
    raise RuntimeError(
        f"Unsupported platform for CFIE native build: {sys.platform}. "
        "Supported platforms are Windows and Linux."
    )
if torch.version.cuda is None:
    _raise_cuda_torch_required()

CUDA_ROOT = resolve_cuda_home()
if CUDA_ROOT is None:
    raise RuntimeError(
        "Unable to locate a CUDA toolkit with an nvcc compiler. "
        "Install the CUDA toolkit and set CUDA_HOME or CUDA_PATH."
    )
CUDA_ALIAS_ROOT = prepare_cuda_root(CUDA_ROOT)

CMAKE_EXECUTABLE = find_tool("cmake")
if CMAKE_EXECUTABLE is None:
    raise RuntimeError("Unable to locate CMake. Install cmake or add it to PATH.")

NINJA_EXECUTABLE = find_tool("ninja")


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwargs) -> None:
        super().__init__(
            name,
            sources=[],
            py_limited_api=not bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
            **kwargs,
        )
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class CMakeBuildExt(build_ext):
    configured: set[str] = set()

    @staticmethod
    def cmake_subprocess_env() -> dict[str, str]:
        env = os.environ.copy()
        if os.name == "nt":
            env.update(_load_msvc_build_env())
        cuda_root = str(CUDA_ROOT)
        cuda_alias_root = str(CUDA_ALIAS_ROOT)
        env.setdefault("CUDA_PATH", cuda_root)
        env.setdefault("CUDA_HOME", cuda_root)
        env.setdefault("CUDAToolkit_ROOT", cuda_alias_root)
        env.setdefault("CUDA_TOOLKIT_ROOT_DIR", cuda_alias_root)
        if os.name == "nt":
            env.setdefault("CudaToolkitDir", cuda_alias_root)
        return env

    def cmake_build_dir(self) -> str:
        override = os.environ.get("CFIE_CMAKE_BUILD_DIR")
        if override:
            return str(Path(override).resolve())
        return self.build_temp

    def compute_num_jobs(self) -> tuple[int, int | None]:
        max_jobs = getattr(envs, "MAX_JOBS", None)
        if max_jobs is not None:
            num_jobs = int(max_jobs)
        else:
            try:
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count() or 1

        nvcc_threads = getattr(envs, "NVCC_THREADS", None)
        if nvcc_threads is not None:
            nvcc_threads = int(nvcc_threads)
            num_jobs = max(1, num_jobs // max(1, nvcc_threads))
        return num_jobs, nvcc_threads

    def configure(self, ext: CMakeExtension) -> None:
        if ext.cmake_lists_dir in self.configured:
            return
        self.configured.add(ext.cmake_lists_dir)

        cfg = getattr(envs, "CMAKE_BUILD_TYPE", None) or (
            "Debug" if self.debug else "RelWithDebInfo"
        )
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DVLLM_TARGET_DEVICE=cuda",
            f"-DVLLM_PYTHON_EXECUTABLE={to_cmake_path(sys.executable)}",
            f"-DPython_EXECUTABLE={to_cmake_path(sys.executable)}",
            f"-DPython3_EXECUTABLE={to_cmake_path(sys.executable)}",
            f"-DVLLM_PYTHON_PATH={os.pathsep.join(to_cmake_path(path) for path in sys.path)}",
            f"-DFETCHCONTENT_BASE_DIR={to_cmake_path(os.environ.get('FETCHCONTENT_BASE_DIR', default_fetchcontent_base_dir()))}",
            f"-DVLLM_CUTLASS_SRC_DIR={to_cmake_path(ROOT_DIR / 'third_party' / 'cutlass')}",
            f"-DVLLM_FLASH_ATTN_SRC_DIR={to_cmake_path(ROOT_DIR / 'third_party' / 'vllm-flash-attn')}",
        ]

        if getattr(envs, "VERBOSE", False):
            cmake_args.append("-DCMAKE_VERBOSE_MAKEFILE=ON")

        num_jobs, nvcc_threads = self.compute_num_jobs()
        if nvcc_threads:
            cmake_args.append(f"-DNVCC_THREADS={nvcc_threads}")

        build_tool: list[str] = []
        requested_generator = (
            os.environ.get("CFIE_CMAKE_GENERATOR")
            or os.environ.get("CMAKE_GENERATOR")
        )
        if requested_generator:
            build_tool = ["-G", requested_generator]
            if os.name == "nt" and requested_generator.startswith("Visual Studio"):
                build_tool.extend(
                    ["-A", os.environ.get("CMAKE_GENERATOR_PLATFORM", "x64")]
                )
            if requested_generator == "Ninja" and NINJA_EXECUTABLE is not None:
                cmake_args.extend(
                    [
                        f"-DCMAKE_MAKE_PROGRAM={to_cmake_path(NINJA_EXECUTABLE)}",
                        "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                        f"-DCMAKE_JOB_POOLS:STRING=compile={num_jobs}",
                    ]
                )
        elif NINJA_EXECUTABLE is not None:
            build_tool = ["-G", "Ninja"]
            cmake_args.extend(
                [
                    f"-DCMAKE_MAKE_PROGRAM={to_cmake_path(NINJA_EXECUTABLE)}",
                    "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                    f"-DCMAKE_JOB_POOLS:STRING=compile={num_jobs}",
                ]
            )
        elif os.name == "nt":
            build_tool = [
                "-G",
                "Visual Studio 17 2022",
                "-A",
                os.environ.get("CMAKE_GENERATOR_PLATFORM", "x64"),
            ]

        nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
        cmake_args.append(
            f"-DCMAKE_CUDA_COMPILER={to_cmake_path(CUDA_ALIAS_ROOT / 'bin' / nvcc_name)}"
        )
        cmake_args.extend(
            [
                f"-DCUDAToolkit_ROOT={to_cmake_path(CUDA_ALIAS_ROOT)}",
                f"-DCUDA_TOOLKIT_ROOT_DIR={to_cmake_path(CUDA_ALIAS_ROOT)}",
            ]
        )
        if CUDA_ALIAS_ROOT != CUDA_ROOT:
            cmake_args.append(
                f"-DCFIE_CUDA_TOOLKIT_ALIAS={to_cmake_path(CUDA_ALIAS_ROOT)}"
            )

        extra_cmake_args = os.environ.get("CMAKE_ARGS")
        if extra_cmake_args:
            cmake_args.extend(extra_cmake_args.split())

        build_dir = self.cmake_build_dir()
        os.makedirs(build_dir, exist_ok=True)
        subprocess.check_call(
            [CMAKE_EXECUTABLE, ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=build_dir,
            env=self.cmake_subprocess_env(),
        )

    @staticmethod
    def target_name(ext_name: str) -> str:
        return ext_name.removeprefix("cfie.").removeprefix("vllm_flash_attn.")

    def build_extensions(self) -> None:
        subprocess.check_call([CMAKE_EXECUTABLE, "--version"])
        self.build_temp = self.cmake_build_dir()
        os.makedirs(self.build_temp, exist_ok=True)
        cmake_env = self.cmake_subprocess_env()
        cfg = getattr(envs, "CMAKE_BUILD_TYPE", None) or (
            "Debug" if self.debug else "RelWithDebInfo"
        )

        targets: list[str] = []
        for ext in self.extensions:
            self.configure(ext)
            targets.append(self.target_name(ext.name))

        num_jobs, _ = self.compute_num_jobs()
        build_cmd = [
            CMAKE_EXECUTABLE,
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={target}" for target in targets],
        ]
        if os.name == "nt":
            build_cmd.extend(["--config", cfg])
        subprocess.check_call(build_cmd, cwd=self.build_temp, env=cmake_env)

        for ext in self.extensions:
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
            if outdir == Path(self.build_temp).resolve():
                continue

            prefix = outdir
            for _ in range(ext.name.count(".")):
                prefix = prefix.parent

            subprocess.check_call(
                [
                    CMAKE_EXECUTABLE,
                    "--install",
                    ".",
                    "--prefix",
                    str(prefix),
                    "--component",
                    self.target_name(ext.name),
                    *(["--config", cfg] if os.name == "nt" else []),
                ],
                cwd=self.build_temp,
                env=cmake_env,
            )


ext_modules = [
    CMakeExtension("cfie._C"),
    CMakeExtension("cfie._moe_C"),
    CMakeExtension("cfie.cumem_allocator"),
    CMakeExtension("cfie.vllm_flash_attn._vllm_fa2_C"),
]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuildExt},
)
