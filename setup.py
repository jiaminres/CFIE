from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys
import sysconfig
import ctypes
import functools
import shutil
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


def default_generator_tag() -> str:
    requested_generator = (
        os.environ.get("CFIE_CMAKE_GENERATOR")
        or os.environ.get("CMAKE_GENERATOR")
    )
    if requested_generator:
        return requested_generator.replace(" ", "-").lower()
    if find_tool("ninja") is not None:
        return "ninja"
    if os.name == "nt":
        return "vs2022"
    return "default"


def default_fetchcontent_base_dir() -> Path:
    machine = platform.machine() or "unknown"
    platform_dir = f"{sys.platform}-{machine}-{default_generator_tag()}".replace(
        " ", "_"
    )
    return ROOT_DIR / ".deps" / platform_dir


def is_supported_platform() -> bool:
    return sys.platform.startswith("linux") or sys.platform.startswith("win")


def find_tool(name: str) -> str | None:
    def iter_windows_fallback_candidates(executable_name: str) -> list[Path]:
        if os.name != "nt":
            return []

        candidates: list[Path] = []
        program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        program_files_x86 = Path(
            os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")
        )

        if executable_name == "cmake.exe":
            candidates.append(program_files / "CMake" / "bin" / executable_name)
            jetbrains_root = program_files / "JetBrains"
            if jetbrains_root.is_dir():
                candidates.extend(
                    sorted(
                        jetbrains_root.glob("CLion* /bin/cmake/win/x64/bin/cmake.exe"),
                        reverse=True,
                    )
                )
                candidates.extend(
                    sorted(
                        jetbrains_root.glob("CLion*/bin/cmake/win/x64/bin/cmake.exe"),
                        reverse=True,
                    )
                )
            for root in (program_files, program_files_x86):
                vs_root = root / "Microsoft Visual Studio"
                if not vs_root.is_dir():
                    continue
                candidates.extend(
                    sorted(
                        vs_root.glob(
                            "*/**/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
                        ),
                        reverse=True,
                    )
                )
        elif executable_name == "ninja.exe":
            jetbrains_root = program_files / "JetBrains"
            if jetbrains_root.is_dir():
                candidates.extend(
                    sorted(
                        jetbrains_root.glob("CLion*/bin/ninja/win/x64/ninja.exe"),
                        reverse=True,
                    )
                )
            for root in (program_files, program_files_x86):
                vs_root = root / "Microsoft Visual Studio"
                if not vs_root.is_dir():
                    continue
                candidates.extend(
                    sorted(
                        vs_root.glob(
                            "*/**/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"
                        ),
                        reverse=True,
                    )
                )
        return candidates

    def is_usable_tool(candidate: Path) -> bool:
        if not candidate.is_file():
            return False
        try:
            subprocess.run(
                [str(candidate), "--version"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return True

    executable = f"{name}.exe" if os.name == "nt" else name
    candidate_paths: list[Path] = []

    for env_name in (f"CFIE_{name.upper()}_EXECUTABLE", f"{name.upper()}_EXECUTABLE"):
        raw_override = os.environ.get(env_name)
        if not raw_override:
            continue
        candidate_paths.append(Path(str(raw_override).strip()).expanduser())

    script_dir = Path(sysconfig.get_path("scripts") or "")
    candidate_paths.append(script_dir / executable)

    for resolved in (which(executable), which(name)):
        if resolved:
            candidate_paths.append(Path(resolved))

    candidate_paths.extend(iter_windows_fallback_candidates(executable))

    seen: set[str] = set()
    for candidate in candidate_paths:
        normalized = os.path.normcase(str(candidate))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if is_usable_tool(candidate):
            return str(candidate)
    return None


def iter_default_cuda_roots() -> list[Path]:
    candidates: list[Path] = []
    if os.name == "nt":
        default_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
        if default_root.is_dir():
            candidates.extend(sorted(default_root.glob("v*"), reverse=True))
    else:
        canonical_root = Path("/usr/local/cuda")
        if canonical_root.is_dir():
            candidates.append(canonical_root)
        cuda_parent = Path("/usr/local")
        if cuda_parent.is_dir():
            candidates.extend(sorted(cuda_parent.glob("cuda-*"), reverse=True))
    return candidates


def iter_cuda_roots() -> list[Path]:
    candidates: list[Path] = []
    for raw_path in (
        os.environ.get("CFIE_CUDA_HOME"),
        CUDA_HOME,
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        os.environ.get("CUDAToolkit_ROOT"),
        os.environ.get("CUDA_TOOLKIT_ROOT_DIR"),
    ):
        if raw_path:
            normalized = str(raw_path).strip()
            if normalized:
                candidates.append(Path(normalized))

    spec = importlib.util.find_spec("nvidia.cuda_nvcc")
    if spec and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            candidates.append(Path(location))

    candidates.extend(iter_default_cuda_roots())

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
    explicit_vcvars = os.environ.get("VCVARS64_BAT")
    if explicit_vcvars:
        direct_candidates.append(Path(explicit_vcvars))
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


def _find_msvc_cl() -> str | None:
    return which("cl.exe") or which("cl")


def _raise_missing_windows_toolchain() -> None:
    raise RuntimeError(
        "Unable to locate a usable MSVC x64 build toolchain for the CFIE native "
        "build. Install Visual Studio 2022 Build Tools with the 'Desktop "
        "development with C++' workload, or launch the build from a Developer "
        "PowerShell. If your Build Tools are installed in a custom location, set "
        "`VCVARS64_BAT` to the full path of `vcvars64.bat`, or set "
        "`VSINSTALLDIR` to the Visual Studio installation root before rerunning "
        "`python -m pip install --no-build-isolation -e . --verbose`."
    )


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

if os.name == "nt" and _find_msvc_cl() is None and _find_vcvars64_bat() is None:
    _raise_missing_windows_toolchain()

CUDA_ROOT = resolve_cuda_home()
if CUDA_ROOT is None:
    if os.name == "nt":
        raise RuntimeError(
            "Unable to locate a CUDA toolkit with an nvcc compiler. CFIE checked "
            "`CFIE_CUDA_HOME`, `CUDA_HOME`, `CUDA_PATH`, `CUDAToolkit_ROOT`, "
            "`CUDA_TOOLKIT_ROOT_DIR`, the `nvidia.cuda_nvcc` Python package, and "
            "the standard Windows CUDA install directory under "
            "`C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA`. Install the "
            "CUDA toolkit and/or set `CUDA_PATH` or `CUDA_HOME` to the toolkit "
            "root before rerunning the editable install."
        )
    raise RuntimeError(
        "Unable to locate a CUDA toolkit with an nvcc compiler. CFIE checked "
        "`CFIE_CUDA_HOME`, `CUDA_HOME`, `CUDA_PATH`, `CUDAToolkit_ROOT`, "
        "`CUDA_TOOLKIT_ROOT_DIR`, the `nvidia.cuda_nvcc` Python package, and "
        "common Linux toolkit locations such as `/usr/local/cuda`. On Linux/WSL, "
        "install the NVIDIA CUDA toolkit (for example `cuda-toolkit-13-1`) or "
        "install `nvidia-cuda-nvcc-cu13` into the target environment, then set "
        "`CUDA_HOME` if needed before rerunning the editable install."
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
    def _is_wsl_drvfs_path(path: str | os.PathLike[str]) -> bool:
        if not sys.platform.startswith("linux"):
            return False
        return Path(path).resolve().as_posix().startswith("/mnt/")

    @staticmethod
    def _copy_file_without_copystat(
        src: str | os.PathLike[str], dst: str | os.PathLike[str]
    ) -> tuple[str, bool]:
        src_path = Path(src)
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_path, dst_path)
        try:
            shutil.copymode(src_path, dst_path)
        except OSError:
            pass
        return str(dst_path), True

    @staticmethod
    def cmake_subprocess_env() -> dict[str, str]:
        env = os.environ.copy()
        if os.name == "nt":
            env.update(_load_msvc_build_env())
        cuda_root = str(CUDA_ROOT)
        cuda_alias_root = str(CUDA_ALIAS_ROOT)
        nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
        cuda_bin_dir = str(CUDA_ROOT / "bin")
        cuda_libnvvp_dir = str(CUDA_ROOT / "libnvvp")
        env.setdefault("CUDA_PATH", cuda_root)
        env.setdefault("CUDA_HOME", cuda_root)
        env.setdefault("CUDAToolkit_ROOT", cuda_alias_root)
        env.setdefault("CUDA_TOOLKIT_ROOT_DIR", cuda_alias_root)
        env["CUDA_NVCC_EXECUTABLE"] = str(CUDA_ROOT / "bin" / nvcc_name)
        env["CUDACXX"] = str(CUDA_ROOT / "bin" / nvcc_name)
        path_entries = env.get("PATH", "").split(os.pathsep)
        prepend_entries = [
            entry for entry in (cuda_bin_dir, cuda_libnvvp_dir) if entry not in path_entries
        ]
        if prepend_entries:
            env["PATH"] = os.pathsep.join([*prepend_entries, *path_entries])
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

        build_dir = str(Path(self.cmake_build_dir()).resolve())
        os.makedirs(build_dir, exist_ok=True)
        subprocess.check_call(
            [
                CMAKE_EXECUTABLE,
                "-S",
                ext.cmake_lists_dir,
                "-B",
                build_dir,
                *build_tool,
                *cmake_args,
            ],
            cwd=ROOT_DIR,
            env=self.cmake_subprocess_env(),
        )

    @staticmethod
    def target_name(ext_name: str) -> str:
        return ext_name.removeprefix("cfie.").removeprefix("vllm_flash_attn.")

    def build_extensions(self) -> None:
        subprocess.check_call([CMAKE_EXECUTABLE, "--version"])
        self.build_temp = str(Path(self.cmake_build_dir()).resolve())
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
            self.build_temp,
            f"-j={num_jobs}",
            *[f"--target={target}" for target in targets],
        ]
        if getattr(envs, "VERBOSE", False):
            build_cmd.append("--verbose")
        if os.name == "nt":
            build_cmd.extend(["--config", cfg])
        subprocess.check_call(build_cmd, cwd=ROOT_DIR, env=cmake_env)

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
                    self.build_temp,
                    "--prefix",
                    str(prefix),
                    "--component",
                    self.target_name(ext.name),
                    *(["--config", cfg] if os.name == "nt" else []),
                ],
                cwd=ROOT_DIR,
                env=cmake_env,
            )

    def copy_extensions_to_source(self) -> None:
        build_py = self.get_finalized_command("build_py")
        for ext in self.extensions:
            inplace_file, regular_file = self._get_inplace_equivalent(build_py, ext)

            if os.path.exists(regular_file) or not ext.optional:
                try:
                    self.copy_file(regular_file, inplace_file, level=self.verbose)
                except PermissionError:
                    if not (
                        self._is_wsl_drvfs_path(inplace_file)
                        and self._is_wsl_drvfs_path(ROOT_DIR)
                    ):
                        raise
                    self.announce(
                        "WSL DrvFs does not support shutil.copy2/copystat for "
                        f"{inplace_file}; falling back to plain copy.",
                        level=2,
                    )
                    self._copy_file_without_copystat(regular_file, inplace_file)

            if ext._needs_stub:
                inplace_stub = self._get_equivalent_stub(ext, inplace_file)
                self._write_stub_file(inplace_stub, ext, compile=True)


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
