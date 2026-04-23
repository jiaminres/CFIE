from __future__ import annotations

import importlib.util
import hashlib
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


# ------------------------------- 报告 PyTorch CUDA 环境缺失 -------------------------------
def _raise_cuda_torch_required() -> None:
    # 读取当前虚拟环境中的 torch 版本，错误信息需要同时展示 Python 包版本。
    torch_version = getattr(torch, "__version__", "unknown")
    # 读取 torch 绑定的 CUDA 版本，用来区分 CPU wheel 与 CUDA wheel。
    torch_cuda = getattr(torch.version, "cuda", None)
    # 直接中断构建，避免后续 CMake 在缺少 CUDA torch 的环境里给出更隐晦的错误。
    raise RuntimeError(
        "CFIE native build requires a CUDA-enabled PyTorch environment. "
        f"Detected torch={torch_version!s} with torch.version.cuda={torch_cuda!r}. "
        "If you already installed a CUDA wheel into your virtual environment, "
        "pip build isolation may be hiding it inside a temporary build env. "
        "Install a CUDA build of torch in your target environment and rerun "
        "`python -m pip install --no-build-isolation -e .`."
    )


# ------------------------------- 从指定路径加载构建配置模块 -------------------------------
def load_module_from_path(module_name: str, path: str):
    # 根据文件路径创建模块加载规范，使 setup 可以读取仓库内的 envs.py。
    spec = importlib.util.spec_from_file_location(module_name, path)
    # 按规范对象创建临时模块实例，后续会把文件内容执行到这个模块里。
    module = importlib.util.module_from_spec(spec)
    # 提前注册到 sys.modules，保证模块内部相对引用或重复加载时能找到同一实例。
    sys.modules[module_name] = module
    # loader 不存在说明模块规范无效，此处提前失败能避免后续空指针式错误。
    assert spec.loader is not None
    # 执行模块文件，把 envs.py 中的构建开关落到 module 对象上。
    spec.loader.exec_module(module)
    # 返回模块对象，调用方通过属性读取 CMAKE_BUILD_TYPE、MAX_JOBS 等配置。
    return module


# ------------------------------- 初始化 setup 全局路径与环境配置 -------------------------------
# 记录 CFIE 包根目录，后续 CMake 源码目录、third_party 路径和临时目录都以它为基准。
ROOT_DIR = Path(__file__).parent.resolve()
# 通过显式路径加载 cfie/envs.py，避免导入外部同名模块影响构建参数。
envs = load_module_from_path("cfie_build_envs", str(ROOT_DIR / "cfie" / "envs.py"))


# ------------------------------- 转换 CMake 可接受的路径格式 -------------------------------
def to_cmake_path(path: str | os.PathLike[str]) -> str:
    # CMake 在 Windows 下也稳定接受正斜杠路径，统一转换可减少转义与空格解析问题。
    return str(path).replace("\\", "/")


# ------------------------------- 判断是否位于 WSL 挂载盘 -------------------------------
def is_wsl_drvfs_path(path: str | os.PathLike[str]) -> bool:
    # 只有 Linux/WSL 需要识别 /mnt 挂载盘，Windows 与普通 Linux 路径直接返回否。
    if not sys.platform.startswith("linux"):
        return False
    # DrvFs 路径在 chmod/copy metadata 上行为不同，后续复制扩展文件时要降级处理。
    return Path(path).resolve().as_posix().startswith("/mnt/")


# ------------------------------- 生成依赖缓存目录的生成器标签 -------------------------------
def default_generator_tag() -> str:
    # 优先尊重外部显式指定的 CMake 生成器，保证 pip 构建和手动构建可复现。
    requested_generator = (
        os.environ.get("CFIE_CMAKE_GENERATOR")
        or os.environ.get("CMAKE_GENERATOR")
    )
    # 将生成器名规范化为目录安全的标签，避免 FetchContent 缓存互相覆盖。
    if requested_generator:
        return requested_generator.replace(" ", "-").lower()
    # Ninja 可用时优先使用 Ninja 标签，因为 setup.py 默认也会优先选择 Ninja。
    if find_tool("ninja") is not None:
        return "ninja"
    # Windows 没有 Ninja 时会退回 Visual Studio 生成器，因此缓存标签也同步为 vs2022。
    if os.name == "nt":
        return "vs2022"
    # 其他平台保留默认标签，避免对未知生成器做错误推断。
    return "default"


# ------------------------------- 计算 FetchContent 依赖缓存目录 -------------------------------
def default_fetchcontent_base_dir() -> Path:
    # 平台目录包含系统、架构和生成器，避免不同工具链共用同一份 CMake 子构建缓存。
    machine = platform.machine() or "unknown"
    platform_dir = f"{sys.platform}-{machine}-{default_generator_tag()}".replace(
        " ", "_"
    )
    # WSL 挂载盘上频繁创建小文件较慢，依赖缓存改放 Linux 用户缓存目录。
    if is_wsl_drvfs_path(ROOT_DIR):
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        # 用仓库路径 hash 隔离不同 checkout，避免多个 CFIE 仓库争用同一依赖缓存。
        repo_key = hashlib.sha1(str(ROOT_DIR).encode("utf-8")).hexdigest()[:12]
        return cache_root / "cfie" / "fetchcontent" / repo_key / platform_dir
    # 非 WSL 场景把依赖缓存放在 CFIE/.deps，便于 CLion 与 pip 构建共用依赖源码。
    return ROOT_DIR / ".deps" / platform_dir


# ------------------------------- 校验当前操作系统是否支持原生扩展构建 -------------------------------
def is_supported_platform() -> bool:
    # 当前 CFIE 原生扩展只维护 Linux 与 Windows 构建链，其他平台提前阻断。
    return sys.platform.startswith("linux") or sys.platform.startswith("win")


# ------------------------------- 查找 CMake 或 Ninja 可执行文件 -------------------------------
def find_tool(name: str) -> str | None:
    # ------------------------------- 枚举 Windows 常见安装位置 -------------------------------
    def iter_windows_fallback_candidates(executable_name: str) -> list[Path]:
        # 非 Windows 平台只依赖 PATH 和虚拟环境，不需要枚举 Visual Studio/CLion 目录。
        if os.name != "nt":
            return []

        # 候选列表保留顺序，后续会按“显式配置优先、环境路径其次、IDE 兜底”的策略使用。
        candidates: list[Path] = []
        # ProgramFiles 是 CMake、CLion 和 Visual Studio 默认安装位置的共同根。
        program_files = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
        # ProgramFiles(x86) 主要用于查找 32 位根目录下的 Visual Studio 安装。
        program_files_x86 = Path(
            os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")
        )

        # CMake 的兜底来源包括独立安装、CLion 内置版本和 Visual Studio 内置版本。
        if executable_name == "cmake.exe":
            candidates.append(program_files / "CMake" / "bin" / executable_name)
            jetbrains_root = program_files / "JetBrains"
            if jetbrains_root.is_dir():
                # 兼容历史上误写的 CLion glob 形式，避免已有环境依赖该路径时失效。
                candidates.extend(
                    sorted(
                        jetbrains_root.glob("CLion* /bin/cmake/win/x64/bin/cmake.exe"),
                        reverse=True,
                    )
                )
                # 优先选择最新 CLion 内置 CMake，通常能覆盖没有独立安装 CMake 的机器。
                candidates.extend(
                    sorted(
                        jetbrains_root.glob("CLion*/bin/cmake/win/x64/bin/cmake.exe"),
                        reverse=True,
                    )
                )
            # Visual Studio 自带 CMake 作为最后兜底，保证 Build Tools 环境也能启动配置。
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
        # Ninja 的兜底来源包括 CLion 内置 Ninja 和 Visual Studio 内置 Ninja。
        elif executable_name == "ninja.exe":
            jetbrains_root = program_files / "JetBrains"
            if jetbrains_root.is_dir():
                # CLion 自带 Ninja 最常见，优先用于保持 IDE 与 pip 构建行为一致。
                candidates.extend(
                    sorted(
                        jetbrains_root.glob("CLion*/bin/ninja/win/x64/ninja.exe"),
                        reverse=True,
                    )
                )
            # 没有 CLion Ninja 时，Visual Studio 的 CMake 扩展目录也可能带有 Ninja。
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

    # ------------------------------- 验证候选工具是否能被当前系统执行 -------------------------------
    def is_usable_tool(candidate: Path) -> bool:
        # 先确认候选路径是实际文件，避免 subprocess 对目录或不存在路径报复杂错误。
        if not candidate.is_file():
            return False
        try:
            # 通过 --version 做轻量执行探测，可同时发现 Device Guard 或策略拦截问题。
            subprocess.run(
                [str(candidate), "--version"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            # 候选工具无法启动时跳过，让后续候选继续参与选择。
            return False
        # 能启动即认为可用，具体版本兼容性由 CMake 配置阶段继续判断。
        return True

    # Windows 需要 .exe 后缀参与显式路径和 PATH 查找，Linux 保持原工具名。
    executable = f"{name}.exe" if os.name == "nt" else name
    # 候选路径按优先级收集，后续去重后逐个验证。
    candidate_paths: list[Path] = []

    # 显式环境变量优先级最高，便于 CLion 或 CI 精确指定 CMake/Ninja。
    for env_name in (f"CFIE_{name.upper()}_EXECUTABLE", f"{name.upper()}_EXECUTABLE"):
        raw_override = os.environ.get(env_name)
        if not raw_override:
            continue
        # strip/expanduser 让用户传入带空白或 ~ 的路径时仍可被解析。
        candidate_paths.append(Path(str(raw_override).strip()).expanduser())

    # 虚拟环境 scripts 目录优先于系统 PATH，保证 python -m pip 安装的 cmake/ninja 被复用。
    script_dir = Path(sysconfig.get_path("scripts") or "")
    candidate_paths.append(script_dir / executable)

    # PATH 中的可执行文件作为常规来源，兼容开发者手动准备的工具链。
    for resolved in (which(executable), which(name)):
        if resolved:
            candidate_paths.append(Path(resolved))

    # Windows 再追加 IDE 与 Visual Studio 常见路径，降低纯 Build Tools 环境的配置门槛。
    candidate_paths.extend(iter_windows_fallback_candidates(executable))

    # seen 防止同一个工具路径被多种来源重复探测，减少不必要的 subprocess 调用。
    seen: set[str] = set()
    for candidate in candidate_paths:
        # normcase 用于 Windows 路径大小写归一，避免同一文件被重复检查。
        normalized = os.path.normcase(str(candidate))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        # 返回第一个能执行的候选，使显式配置和虚拟环境工具具备最高优先级。
        if is_usable_tool(candidate):
            return str(candidate)
    # 所有来源都不可用时返回 None，由调用方生成面向用户的构建错误。
    return None


# ------------------------------- 枚举默认 CUDA Toolkit 安装根目录 -------------------------------
def iter_default_cuda_roots() -> list[Path]:
    # 候选根目录按新版本优先排序，适配机器上同时安装多个 CUDA Toolkit 的情况。
    candidates: list[Path] = []
    if os.name == "nt":
        # Windows 官方安装器默认把 CUDA 放在 Program Files 下的 CUDA/vX.Y 目录。
        default_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
        if default_root.is_dir():
            candidates.extend(sorted(default_root.glob("v*"), reverse=True))
    else:
        # Linux 常见软链接 /usr/local/cuda 表示当前默认 Toolkit。
        canonical_root = Path("/usr/local/cuda")
        if canonical_root.is_dir():
            candidates.append(canonical_root)
        # 同时枚举 /usr/local/cuda-*，用于没有默认软链接但安装了版本目录的机器。
        cuda_parent = Path("/usr/local")
        if cuda_parent.is_dir():
            candidates.extend(sorted(cuda_parent.glob("cuda-*"), reverse=True))
    return candidates


# ------------------------------- 汇总所有可能的 CUDA Toolkit 根目录 -------------------------------
def iter_cuda_roots() -> list[Path]:
    # candidates 会按可信度从高到低收集，resolve_cuda_home 只需要找到第一个含 nvcc 的目录。
    candidates: list[Path] = []
    # 外部显式变量优先于默认安装路径，便于在 CUDA 多版本机器上锁定构建版本。
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

    # 某些环境通过 nvidia-cuda-nvcc Python 包提供 nvcc，这里把包目录也加入搜索。
    spec = importlib.util.find_spec("nvidia.cuda_nvcc")
    if spec and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            candidates.append(Path(location))

    # 默认安装路径作为兜底，避免用户没有设置 CUDA_HOME 时直接失败。
    candidates.extend(iter_default_cuda_roots())

    # 去重后保持原优先级，避免同一路径来自 CUDA_HOME 和 CUDA_PATH 时重复检查。
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = os.path.normcase(str(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


# ------------------------------- 选择实际可用的 CUDA Toolkit 根目录 -------------------------------
def resolve_cuda_home() -> Path | None:
    # Windows 与类 Unix 的 nvcc 文件名不同，先确定当前平台要检查的编译器文件名。
    nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
    # 只接受根目录下存在 bin/nvcc 的候选，避免把 CUDA runtime 或 torch 目录误当 Toolkit。
    for candidate in iter_cuda_roots():
        if (candidate / "bin" / nvcc_name).exists():
            return candidate
    # 没有任何候选包含 nvcc 时交给调用方报出更完整的排查路径。
    return None


# ------------------------------- 获取 Windows 无空格短路径 -------------------------------
def get_windows_short_path(path: Path) -> Path | None:
    # 非 Windows 不存在 8.3 短路径语义，直接返回 None 让调用方使用原路径。
    if os.name != "nt":
        return None

    # Windows API 需要调用方提供缓冲区，32768 覆盖常见最大路径长度。
    buffer = ctypes.create_unicode_buffer(32768)
    # GetShortPathNameW 会把带空格的 Program Files 路径转换成 PROGRA~1 形式。
    result = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
    if result == 0:
        return None
    # 空字符串视为失败，避免把无效路径继续传给 CMake。
    short_path = buffer.value.strip()
    return Path(short_path) if short_path else None


# ------------------------------- 准备传给 CMake 的 CUDA 根路径 -------------------------------
def prepare_cuda_root(cuda_root: Path) -> Path:
    # 非 Windows 或路径本身没有空格时无需额外处理，直接保持真实 Toolkit 根目录。
    if os.name != "nt" or " " not in str(cuda_root):
        return cuda_root

    # Windows 优先使用系统提供的 8.3 短路径，避免 nvcc/CMake/FindCUDA 在空格路径上解析失败。
    short_path = get_windows_short_path(cuda_root)
    if short_path is not None and " " not in str(short_path):
        return short_path

    # 若短路径不可用，则在 CFIE/.deps/toolchains 下准备一个无空格的 CUDA 目录别名。
    version_suffix = torch.version.cuda or "cuda"
    alias_root = ROOT_DIR / ".deps" / "toolchains" / f"cuda-{version_suffix}"
    # 创建别名父目录，使后续 mklink /J 可以直接落盘。
    alias_root.parent.mkdir(parents=True, exist_ok=True)

    # 已存在的别名只在指向当前 CUDA 根目录时复用，避免误用旧版本 Toolkit。
    if alias_root.exists():
        try:
            if alias_root.resolve() == cuda_root.resolve():
                return alias_root
        except OSError:
            pass
        # 现有路径无法确认指向当前 Toolkit 时保守退回原路径，不擅自删除用户文件。
        return cuda_root

    try:
        # junction 不需要管理员权限的概率更高，且能让 CMake 看到一个无空格 CUDA 根目录。
        subprocess.check_call(
            ["cmd", "/c", "mklink", "/J", str(alias_root), str(cuda_root)]
        )
        return alias_root
    except (OSError, subprocess.CalledProcessError):
        # 创建别名失败时仍继续使用真实路径，让后续 CMake 给出具体失败位置。
        return cuda_root


# ------------------------------- 定位 Visual Studio vcvars64.bat -------------------------------
def _find_vcvars64_bat() -> Path | None:
    # 只有 Windows 需要通过 vcvars64.bat 注入 MSVC 桌面 x64 编译环境。
    if os.name != "nt":
        return None

    # direct_candidates 保存无需 vswhere 即可判断的高可信路径。
    direct_candidates: list[Path] = []
    # VCVARS64_BAT 允许用户在自定义安装位置时精确指定批处理脚本。
    explicit_vcvars = os.environ.get("VCVARS64_BAT")
    if explicit_vcvars:
        direct_candidates.append(Path(explicit_vcvars))
    # VSINSTALLDIR 来自已有开发者命令行，复用它可保持与当前 shell 的 VS 实例一致。
    vsinstalldir = os.environ.get("VSINSTALLDIR")
    if vsinstalldir:
        direct_candidates.append(
            Path(vsinstalldir) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        )

    # 官方默认安装目录下按常见 edition 枚举，覆盖 BuildTools 与完整 IDE。
    default_vs_root = Path("C:/Program Files/Microsoft Visual Studio/2022")
    for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
        direct_candidates.append(
            default_vs_root / edition / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        )

    # 先返回第一个真实存在的直接候选，避免每次都启动 vswhere。
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    # 没有 ProgramFiles(x86) 时无法定位 vswhere，直接返回 None 交给上层报错。
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if not program_files_x86:
        return None

    # vswhere 是 Visual Studio 官方安装发现工具，可处理自定义安装目录。
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None

    try:
        # 查询带 C++ x64 工具组件的最新 VS 实例，避免选到没有 cl.exe 的安装。
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
        # vswhere 执行失败时不猜测路径，避免错误注入半套 MSVC 环境。
        return None

    # 空输出代表没有符合条件的 VS 安装。
    if not installation_path:
        return None

    # 从 VS 安装根拼出 vcvars64.bat，存在才视为可用。
    candidate = Path(installation_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    return candidate if candidate.is_file() else None


# ------------------------------- 检查当前 PATH 是否已有 cl 编译器 -------------------------------
def _find_msvc_cl() -> str | None:
    # Developer PowerShell 中可能已经有 cl.exe，setup 不必强制再找 vcvars64.bat。
    return which("cl.exe") or which("cl")


# ------------------------------- 报告 Windows MSVC 工具链缺失 -------------------------------
def _raise_missing_windows_toolchain() -> None:
    # 在进入 CMake 前给出明确安装建议，避免用户只看到 nvcc 或 link 的底层错误。
    raise RuntimeError(
        "Unable to locate a usable MSVC x64 build toolchain for the CFIE native "
        "build. Install Visual Studio 2022 Build Tools with the 'Desktop "
        "development with C++' workload, or launch the build from a Developer "
        "PowerShell. If your Build Tools are installed in a custom location, set "
        "`VCVARS64_BAT` to the full path of `vcvars64.bat`, or set "
        "`VSINSTALLDIR` to the Visual Studio installation root before rerunning "
        "`python -m pip install --no-build-isolation -e . --verbose`."
    )


# ------------------------------- 加载 MSVC 桌面 x64 构建环境 -------------------------------
@functools.lru_cache(maxsize=1)
def _load_msvc_build_env() -> dict[str, str]:
    # setup 构建每次进程只需要加载一次 vcvars64 输出，缓存可避免重复启动 cmd。
    vcvars64 = _find_vcvars64_bat()
    if vcvars64 is None:
        # 找不到 vcvars64 时返回空环境，允许已有 Developer Shell 环境继续工作。
        return {}

    # 先执行 vcvars64.bat 再输出 set，拿到 PATH/INCLUDE/LIB/LIBPATH 等完整变量。
    command = f'cmd /d /s /c ""{vcvars64}" >nul && set"'
    # 以当前环境为基础运行 vcvars64，保留用户已设置的 CUDA/Python 等变量。
    cmd_env = os.environ.copy()
    # 跳过 VS 命令行遥测，减少构建过程中的无关输出和网络侧效应。
    cmd_env.setdefault("VSCMD_SKIP_SENDTELEMETRY", "1")

    try:
        # text=True 直接得到字符串输出，便于逐行解析 key=value。
        output = subprocess.check_output(
            command,
            text=True,
            env=cmd_env,
            shell=True,
        )
    except (OSError, subprocess.CalledProcessError):
        # vcvars64 加载失败时返回空环境，让后续 CMake 使用当前 shell 环境并暴露真实错误。
        return {}

    # 解析后的环境会覆盖 subprocess env 中的同名变量，确保 MSVC 桌面 x64 路径生效。
    parsed_env: dict[str, str] = {}
    for line in output.splitlines():
        # set 输出中理论上都是 key=value，仍保留防御以跳过异常行。
        if "=" not in line:
            continue
        # 只按第一个等号切分，避免 PATH 等变量值中包含等号时被截断。
        key, value = line.split("=", 1)
        if key:
            parsed_env[key] = value
    return parsed_env


# ------------------------------- 执行构建入口前的全局环境校验 -------------------------------
if not is_supported_platform():
    # 其他系统没有维护过 CFIE 原生扩展链路，因此在 setup 初始化阶段直接阻断。
    raise RuntimeError(
        f"Unsupported platform for CFIE native build: {sys.platform}. "
        "Supported platforms are Windows and Linux."
    )
if torch.version.cuda is None:
    # 缺少 CUDA torch 时无法确定目标 CUDA ABI，也无法链接 torch CUDA 库。
    _raise_cuda_torch_required()

if os.name == "nt" and _find_msvc_cl() is None and _find_vcvars64_bat() is None:
    # Windows 构建必须能找到 cl.exe 或 vcvars64.bat，否则 nvcc 的 host 编译阶段无法运行。
    _raise_missing_windows_toolchain()

# ------------------------------- 解析并规范化 CUDA Toolkit 根目录 -------------------------------
CUDA_ROOT = resolve_cuda_home()
if CUDA_ROOT is None:
    if os.name == "nt":
        # Windows 下明确列出检查过的 CUDA 来源，方便用户修正 CUDA_PATH 或安装 Toolkit。
        raise RuntimeError(
            "Unable to locate a CUDA toolkit with an nvcc compiler. CFIE checked "
            "`CFIE_CUDA_HOME`, `CUDA_HOME`, `CUDA_PATH`, `CUDAToolkit_ROOT`, "
            "`CUDA_TOOLKIT_ROOT_DIR`, the `nvidia.cuda_nvcc` Python package, and "
            "the standard Windows CUDA install directory under "
            "`C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA`. Install the "
            "CUDA toolkit and/or set `CUDA_PATH` or `CUDA_HOME` to the toolkit "
            "root before rerunning the editable install."
        )
    # Linux/WSL 下同时提示系统 Toolkit 与 Python nvcc 包两条可行路径。
    raise RuntimeError(
        "Unable to locate a CUDA toolkit with an nvcc compiler. CFIE checked "
        "`CFIE_CUDA_HOME`, `CUDA_HOME`, `CUDA_PATH`, `CUDAToolkit_ROOT`, "
        "`CUDA_TOOLKIT_ROOT_DIR`, the `nvidia.cuda_nvcc` Python package, and "
        "common Linux toolkit locations such as `/usr/local/cuda`. On Linux/WSL, "
        "install the NVIDIA CUDA toolkit (for example `cuda-toolkit-13-1`) or "
        "install `nvidia-cuda-nvcc-cu13` into the target environment, then set "
        "`CUDA_HOME` if needed before rerunning the editable install."
    )
# CUDA_ALIAS_ROOT 是传给 CMake 的稳定路径，Windows 上会优先变成无空格短路径或 junction。
CUDA_ALIAS_ROOT = prepare_cuda_root(CUDA_ROOT)

# ------------------------------- 选择 CMake 与 Ninja 前端工具 -------------------------------
CMAKE_EXECUTABLE = find_tool("cmake")
if CMAKE_EXECUTABLE is None:
    # 没有 CMake 无法生成原生扩展工程，因此在 build_ext 前提前失败。
    raise RuntimeError("Unable to locate CMake. Install cmake or add it to PATH.")

# Ninja 可选；存在时 setup 会优先使用它，否则 Windows 可退回 Visual Studio 生成器。
NINJA_EXECUTABLE = find_tool("ninja")


# ------------------------------- 定义 setuptools 的 CMake 扩展占位对象 -------------------------------
class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwargs) -> None:
        # setuptools 仍需要 Extension 对象描述产物名，但实际源码由 CMakeLists.txt 管理。
        super().__init__(
            name,
            sources=[],
            # Python free-threading 场景不使用稳定 ABI，其他场景启用 py_limited_api。
            py_limited_api=not bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
            **kwargs,
        )
        # CMake 源码目录保存为绝对路径，避免 build_ext 在不同 cwd 下重复配置。
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


# ------------------------------- 驱动 CMake 配置、编译与安装 -------------------------------
class CMakeBuildExt(build_ext):
    # 同一个 CMakeLists 只配置一次，多个扩展目标共用同一个 build 目录。
    configured: set[str] = set()

    # ------------------------------- 判断复制目标是否位于 WSL 挂载盘 -------------------------------
    @staticmethod
    def _is_wsl_drvfs_path(path: str | os.PathLike[str]) -> bool:
        # 复用全局判断逻辑，让 copy_extensions_to_source 的降级路径保持一致。
        return is_wsl_drvfs_path(path)

    # ------------------------------- 在 WSL 挂载盘上执行无元数据复制 -------------------------------
    @staticmethod
    def _copy_file_without_copystat(
        src: str | os.PathLike[str], dst: str | os.PathLike[str]
    ) -> tuple[str, bool]:
        # 将输入路径转为 Path，后续创建父目录和复制文件都以 pathlib 操作。
        src_path = Path(src)
        dst_path = Path(dst)
        # 先创建目标父目录，避免 copyfile 因目录不存在失败。
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        # 只复制文件内容，不复制 stat 元数据，绕开 WSL DrvFs 对 copystat 的限制。
        shutil.copyfile(src_path, dst_path)
        try:
            # 尝试复制权限位，失败时忽略，因为 Windows 挂载盘通常不支持完整 POSIX mode。
            shutil.copymode(src_path, dst_path)
        except OSError:
            pass
        # 返回 setuptools copy_file 兼容的二元组，表示目标文件已被更新。
        return str(dst_path), True

    # ------------------------------- 构造传给 CMake 子进程的环境变量 -------------------------------
    @staticmethod
    def cmake_subprocess_env() -> dict[str, str]:
        # 以当前 Python 进程环境为基础，保留用户传入的代理、PATH 与构建开关。
        env = os.environ.copy()
        if os.name == "nt":
            # Windows 注入 vcvars64 环境，确保 cl/link/rc/mt 和 LIB/INCLUDE 都来自桌面 x64。
            env.update(_load_msvc_build_env())
        # cuda_root 保留真实安装目录，供 PATH 与 CUDA_HOME 指向真实 Toolkit。
        cuda_root = str(CUDA_ROOT)
        # cuda_alias_root 是 CMake 看到的无空格路径，主要解决 Windows 空格路径兼容性。
        cuda_alias_root = str(CUDA_ALIAS_ROOT)
        # nvcc 文件名按平台区分，后续多个变量都要指向同一个编译器。
        nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
        # CUDA bin 需要进入 PATH，Torch 的 try_run 和 nvcc 子进程都依赖它。
        cuda_bin_dir = str(CUDA_ROOT / "bin")
        # libnvvp 是 CUDA 旧工具目录，加入 PATH 可兼容部分依赖仍查找该路径的情况。
        cuda_libnvvp_dir = str(CUDA_ROOT / "libnvvp")
        # setdefault 保留用户显式设置，同时在未设置时补齐标准 CUDA 环境变量。
        env.setdefault("CUDA_PATH", cuda_root)
        env.setdefault("CUDA_HOME", cuda_root)
        # CUDAToolkit_ROOT 与 CUDA_TOOLKIT_ROOT_DIR 指向别名路径，指导 CMake 查 include/lib。
        env.setdefault("CUDAToolkit_ROOT", cuda_alias_root)
        env.setdefault("CUDA_TOOLKIT_ROOT_DIR", cuda_alias_root)
        # CUDA_NVCC_EXECUTABLE 是 Torch FindCUDA 的旧变量，指向真实 nvcc 避免找错版本。
        env["CUDA_NVCC_EXECUTABLE"] = str(CUDA_ROOT / "bin" / nvcc_name)
        # CUDACXX 是 CMake CUDA language 的入口变量，同样固定到真实 nvcc。
        env["CUDACXX"] = str(CUDA_ROOT / "bin" / nvcc_name)
        # 拆分 PATH 后只追加缺失目录，避免每次 build_ext 重复前置 CUDA 路径。
        path_entries = env.get("PATH", "").split(os.pathsep)
        prepend_entries = [
            entry for entry in (cuda_bin_dir, cuda_libnvvp_dir) if entry not in path_entries
        ]
        if prepend_entries:
            # CUDA 工具目录放在 PATH 最前面，保证 nvcc、cudart DLL 和辅助工具版本一致。
            env["PATH"] = os.pathsep.join([*prepend_entries, *path_entries])
        if os.name == "nt":
            # CudaToolkitDir 是部分 Windows CMake/VS 规则识别 CUDA Toolkit 的传统变量。
            env.setdefault("CudaToolkitDir", cuda_alias_root)
        return env

    # ------------------------------- 计算 CMake build 目录 -------------------------------
    def cmake_build_dir(self) -> str:
        # CFIE_CMAKE_BUILD_DIR 允许开发者和 CLion 指定固定 build 目录，便于复用缓存。
        override = os.environ.get("CFIE_CMAKE_BUILD_DIR")
        if override:
            return str(Path(override).resolve())
        # 默认使用 setuptools 的 build_temp，使 pip 构建产物不污染源码目录。
        return self.build_temp

    # ------------------------------- 计算并行构建参数 -------------------------------
    def compute_num_jobs(self) -> tuple[int, int | None]:
        # envs.MAX_JOBS 显式控制总并发，适合显存/内存紧张机器限制编译压力。
        max_jobs = getattr(envs, "MAX_JOBS", None)
        if max_jobs is not None:
            num_jobs = int(max_jobs)
        else:
            try:
                # Linux 优先使用 CPU affinity，尊重容器或 CI 限制的可用核心集合。
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                # Windows 没有 sched_getaffinity，退回 os.cpu_count 估算并发。
                num_jobs = os.cpu_count() or 1

        # NVCC_THREADS 控制单个 nvcc 进程内部并发，需要反向压低外层 Ninja 并发。
        nvcc_threads = getattr(envs, "NVCC_THREADS", None)
        if nvcc_threads is not None:
            nvcc_threads = int(nvcc_threads)
            # 至少保留一个外层任务，避免 nvcc_threads 大于 CPU 数时并发变成 0。
            num_jobs = max(1, num_jobs // max(1, nvcc_threads))
        # 返回外层 job 数和 nvcc 内部线程数，调用方分别传给 CMake build 与 NVCC_THREADS。
        return num_jobs, nvcc_threads

    # ------------------------------- 执行 CMake configure 阶段 -------------------------------
    def configure(self, ext: CMakeExtension) -> None:
        # 同一 CMakeLists 已配置过时直接复用，避免多个扩展目标重复生成工程。
        if ext.cmake_lists_dir in self.configured:
            return
        self.configured.add(ext.cmake_lists_dir)

        # 构建类型优先使用 envs.py 显式配置，否则跟随 setuptools debug 标志。
        cfg = getattr(envs, "CMAKE_BUILD_TYPE", None) or (
            "Debug" if self.debug else "RelWithDebInfo"
        )
        # cmake_args 是 setup.py 注入给 CMakeLists.txt 的核心变量集合。
        cmake_args = [
            # 单配置生成器需要 CMAKE_BUILD_TYPE，多配置生成器后续 build 仍会传 --config。
            f"-DCMAKE_BUILD_TYPE={cfg}",
            # CFIE 当前原生扩展只维护 CUDA 后端，显式传入避免默认值被外部覆盖。
            "-DVLLM_TARGET_DEVICE=cuda",
            # VLLM_PYTHON_EXECUTABLE 让 CMake 精确匹配当前虚拟环境的 Python。
            f"-DVLLM_PYTHON_EXECUTABLE={to_cmake_path(sys.executable)}",
            # Python_EXECUTABLE/Python3_EXECUTABLE 兼容 CMake FindPython 的不同变量名。
            f"-DPython_EXECUTABLE={to_cmake_path(sys.executable)}",
            f"-DPython3_EXECUTABLE={to_cmake_path(sys.executable)}",
            # 将当前 sys.path 传入 CMake，代码生成脚本可复用 pip 构建环境中的 Python 包。
            f"-DVLLM_PYTHON_PATH={os.pathsep.join(to_cmake_path(path) for path in sys.path)}",
            # FetchContent 基目录固定后，CUTLASS 等依赖不会在不同 build 目录重复下载/展开。
            f"-DFETCHCONTENT_BASE_DIR={to_cmake_path(os.environ.get('FETCHCONTENT_BASE_DIR', default_fetchcontent_base_dir()))}",
            # 默认使用仓库内置 cutlass，避免 configure 阶段访问网络。
            f"-DVLLM_CUTLASS_SRC_DIR={to_cmake_path(ROOT_DIR / 'third_party' / 'cutlass')}",
            # 默认使用仓库内置 vllm-flash-attn，保证 CUDA attention 扩展可离线构建。
            f"-DVLLM_FLASH_ATTN_SRC_DIR={to_cmake_path(ROOT_DIR / 'third_party' / 'vllm-flash-attn')}",
        ]

        if getattr(envs, "VERBOSE", False):
            # 开启 CMake 生成的详细构建命令，便于排查 nvcc/cl/link 实际参数。
            cmake_args.append("-DCMAKE_VERBOSE_MAKEFILE=ON")

        # 计算外层构建并发和 nvcc 内部线程数，避免两个层级同时满载导致机器卡死。
        num_jobs, nvcc_threads = self.compute_num_jobs()
        if nvcc_threads:
            # NVCC_THREADS 只传给 CMakeLists，实际会追加到 CUDA 编译 flags。
            cmake_args.append(f"-DNVCC_THREADS={nvcc_threads}")

        # build_tool 保存 -G/-A 等生成器参数，后续与 cmake_args 一起传给 CMake。
        build_tool: list[str] = []
        # 外部可通过 CFIE_CMAKE_GENERATOR/CMAKE_GENERATOR 强制指定生成器。
        requested_generator = (
            os.environ.get("CFIE_CMAKE_GENERATOR")
            or os.environ.get("CMAKE_GENERATOR")
        )
        if requested_generator:
            # 显式生成器优先级最高，setup 不再自行选择 Ninja 或 Visual Studio。
            build_tool = ["-G", requested_generator]
            if os.name == "nt" and requested_generator.startswith("Visual Studio"):
                # Visual Studio 生成器需要平台参数，默认 x64 避免落到 Win32。
                build_tool.extend(
                    ["-A", os.environ.get("CMAKE_GENERATOR_PLATFORM", "x64")]
                )
            if requested_generator == "Ninja" and NINJA_EXECUTABLE is not None:
                # Ninja 生成器显式绑定可执行文件，避免 CMake 从 PATH 找到被策略拦截的 ninja。
                cmake_args.extend(
                    [
                        f"-DCMAKE_MAKE_PROGRAM={to_cmake_path(NINJA_EXECUTABLE)}",
                        # Job pool 限制编译阶段并发，链接和生成阶段仍由 Ninja 默认处理。
                        "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                        f"-DCMAKE_JOB_POOLS:STRING=compile={num_jobs}",
                    ]
                )
        elif NINJA_EXECUTABLE is not None:
            # 未显式指定生成器且 Ninja 可用时，优先使用 Ninja 获得更快更稳定的单配置构建。
            build_tool = ["-G", "Ninja"]
            cmake_args.extend(
                [
                    f"-DCMAKE_MAKE_PROGRAM={to_cmake_path(NINJA_EXECUTABLE)}",
                    # 将 CMake 编译 job pool 绑定到 setup 计算出的安全并发。
                    "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                    f"-DCMAKE_JOB_POOLS:STRING=compile={num_jobs}",
                ]
            )
        elif os.name == "nt":
            # Windows 没有 Ninja 时退回 VS2022 生成器，并保持 x64 平台。
            build_tool = [
                "-G",
                "Visual Studio 17 2022",
                "-A",
                os.environ.get("CMAKE_GENERATOR_PLATFORM", "x64"),
            ]

        # nvcc 路径在 CMake 变量中使用别名根，避免 CMake 生成命令时再次遇到空格路径。
        nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
        cmake_args.append(
            f"-DCMAKE_CUDA_COMPILER={to_cmake_path(CUDA_ALIAS_ROOT / 'bin' / nvcc_name)}"
        )
        # 标准 CUDA root 变量统一指向别名根，供 Torch FindCUDA 和 CUDAToolkit 同时使用。
        cmake_args.extend(
            [
                f"-DCUDAToolkit_ROOT={to_cmake_path(CUDA_ALIAS_ROOT)}",
                f"-DCUDA_TOOLKIT_ROOT_DIR={to_cmake_path(CUDA_ALIAS_ROOT)}",
            ]
        )

        # CMAKE_ARGS 作为最后追加的逃生口，允许开发者临时注入诊断或覆盖变量。
        extra_cmake_args = os.environ.get("CMAKE_ARGS")
        if extra_cmake_args:
            cmake_args.extend(extra_cmake_args.split())

        # build_dir 是实际 -B 目录，先创建可避免 CMake 在权限或路径不存在时报错。
        build_dir = str(Path(self.cmake_build_dir()).resolve())
        os.makedirs(build_dir, exist_ok=True)
        # 启动 CMake configure，cwd 固定到 ROOT_DIR 以保证相对 third_party 路径稳定。
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

    # ------------------------------- 将 Python 扩展名映射到 CMake 目标名 -------------------------------
    @staticmethod
    def target_name(ext_name: str) -> str:
        # CMake 目标不带 Python 包前缀，因此需要去掉 cfie. 或 vllm_flash_attn.。
        return ext_name.removeprefix("cfie.").removeprefix("vllm_flash_attn.")

    # ------------------------------- 执行 CMake build 与 install 阶段 -------------------------------
    def build_extensions(self) -> None:
        # 先打印 CMake 版本，构建日志中能直接看到当前使用的是哪个 CMake 前端。
        subprocess.check_call([CMAKE_EXECUTABLE, "--version"])
        # setuptools 会设置 build_temp；这里再应用 CFIE_CMAKE_BUILD_DIR 覆盖并创建目录。
        self.build_temp = str(Path(self.cmake_build_dir()).resolve())
        os.makedirs(self.build_temp, exist_ok=True)
        # build 和 configure 使用同一套环境，避免配置阶段与编译阶段的 MSVC/CUDA 不一致。
        cmake_env = self.cmake_subprocess_env()
        # build 配置必须与 configure 配置一致，多配置生成器会通过 --config 使用该值。
        cfg = getattr(envs, "CMAKE_BUILD_TYPE", None) or (
            "Debug" if self.debug else "RelWithDebInfo"
        )

        # targets 保存本轮 setuptools 请求构建的所有 CMake 目标。
        targets: list[str] = []
        for ext in self.extensions:
            # 第一个扩展会触发 configure，后续同源目录扩展只登记目标名。
            self.configure(ext)
            targets.append(self.target_name(ext.name))

        # build 阶段继续使用同一并发计算，确保实际编译压力与 configure 注入的 job pool 一致。
        num_jobs, _ = self.compute_num_jobs()
        # CMake build 命令按目标构建，避免编译未声明为 Python 扩展的辅助目标。
        build_cmd = [
            CMAKE_EXECUTABLE,
            "--build",
            self.build_temp,
            f"-j={num_jobs}",
            *[f"--target={target}" for target in targets],
        ]
        if getattr(envs, "VERBOSE", False):
            # verbose 让 Ninja/MSBuild 展开 nvcc、cl 和 link 的真实命令行。
            build_cmd.append("--verbose")
        if os.name == "nt":
            # Windows 多配置生成器需要 --config，Ninja 会忽略或按 CMake 约定处理。
            build_cmd.extend(["--config", cfg])
        # 执行实际编译，失败时 subprocess 会把 CMake 返回码直接传回 pip。
        subprocess.check_call(build_cmd, cwd=ROOT_DIR, env=cmake_env)

        for ext in self.extensions:
            # 计算 setuptools 期望的扩展输出目录，后续 cmake --install 会安装到对应包目录。
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
            if outdir == Path(self.build_temp).resolve():
                # 如果 setuptools 输出目录就是 build 目录，无需额外 install 复制。
                continue

            # prefix 需要回退到包根，确保 CMake install 的 DESTINATION cfie 能落到正确位置。
            prefix = outdir
            for _ in range(ext.name.count(".")):
                prefix = prefix.parent

            # 每个扩展单独 install 对应 component，避免把无关目标安装到源码树。
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

    # ------------------------------- 将构建产物复制回源码树供 editable 使用 -------------------------------
    def copy_extensions_to_source(self) -> None:
        # build_py 提供 editable 模式下扩展模块的源码树等价路径。
        build_py = self.get_finalized_command("build_py")
        for ext in self.extensions:
            # setuptools 计算 build 产物路径与源码树目标路径，后续按需复制。
            inplace_file, regular_file = self._get_inplace_equivalent(build_py, ext)

            if os.path.exists(regular_file) or not ext.optional:
                try:
                    # 默认使用 setuptools copy_file，保留其日志和时间戳判断行为。
                    self.copy_file(regular_file, inplace_file, level=self.verbose)
                except PermissionError:
                    if not (
                        self._is_wsl_drvfs_path(inplace_file)
                        and self._is_wsl_drvfs_path(ROOT_DIR)
                    ):
                        # 非 WSL DrvFs 的权限错误应原样抛出，避免隐藏真实文件权限问题。
                        raise
                    # WSL DrvFs 不支持完整 copystat 时，降级为只复制文件内容。
                    self.announce(
                        "WSL DrvFs does not support shutil.copy2/copystat for "
                        f"{inplace_file}; falling back to plain copy.",
                        level=2,
                    )
                    self._copy_file_without_copystat(regular_file, inplace_file)

            if ext._needs_stub:
                # 稳定 ABI 扩展需要生成 .pyi stub，让 editable 安装下的导入元数据完整。
                inplace_stub = self._get_equivalent_stub(ext, inplace_file)
                self._write_stub_file(inplace_stub, ext, compile=True)


# ------------------------------- 声明 Python 包需要构建的 CMake 扩展目标 -------------------------------
ext_modules = [
    # cfie._C 聚合基础 CUDA kernel、量化 kernel、采样和 torch binding。
    CMakeExtension("cfie._C"),
    # cfie._moe_C 聚合 MoE 路由、topk、Marlin MoE 与 router GEMM 相关 kernel。
    CMakeExtension("cfie._moe_C"),
    # cfie.cumem_allocator 构建 CUDA driver memory allocator 的轻量 C++ 扩展。
    CMakeExtension("cfie.cumem_allocator"),
    # vendored vllm-flash-attn 扩展由外部 CMake 脚本定义并通过同一 build_ext 驱动。
    CMakeExtension("cfie.vllm_flash_attn._vllm_fa2_C"),
]


# ------------------------------- 注册 setuptools 构建入口 -------------------------------
setup(
    # ext_modules 只声明 Python 产物名，具体源码、宏和链接库由 CMakeLists.txt 维护。
    ext_modules=ext_modules,
    # build_ext 被替换为 CMakeBuildExt，使 pip install -e . 进入 CMake 构建链路。
    cmdclass={"build_ext": CMakeBuildExt},
)
