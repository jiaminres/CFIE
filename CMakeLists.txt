cmake_minimum_required(VERSION 3.26)

# 可单独设置如下参数 和 环境变量 单独编译调试
#-DVLLM_TARGET_DEVICE=cuda
#-DVLLM_PYTHON_EXECUTABLE=C:/Users/13642/PycharmProjects/vllm/.venv/Scripts/python.exe
#-DPython_EXECUTABLE=C:/Users/13642/PycharmProjects/vllm/.venv/Scripts/python.exe
#-DPython3_EXECUTABLE=C:/Users/13642/PycharmProjects/vllm/.venv/Scripts/python.exe
#-DFETCHCONTENT_BASE_DIR=C:/Users/13642/PycharmProjects/vllm/CFIE/.deps/win32-AMD64-ninja
#-DVLLM_CUTLASS_SRC_DIR=C:/Users/13642/PycharmProjects/vllm/CFIE/third_party/cutlass
#-DVLLM_FLASH_ATTN_SRC_DIR=C:/Users/13642/PycharmProjects/vllm/CFIE/third_party/vllm-flash-attn
#-DCMAKE_MAKE_PROGRAM=C:/Users/13642/PycharmProjects/vllm/.venv/Scripts/ninja.exe
#-DCMAKE_JOB_POOL_COMPILE:STRING=compile
#-DCMAKE_JOB_POOLS:STRING=compile=8
#-DCMAKE_CUDA_COMPILER=C:/PROGRA~1/NVIDIA~2/CUDA/v13.1/bin/nvcc.exe
#-DCUDAToolkit_ROOT=C:/PROGRA~1/NVIDIA~2/CUDA/v13.1
#-DCUDA_TOOLKIT_ROOT_DIR=C:/PROGRA~1/NVIDIA~2/CUDA/v13.1

# CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1;CUDA_NVCC_EXECUTABLE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe;CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1;CUDA_TOOLKIT_ROOT_DIR=C:\PROGRA~1\NVIDIA~2\CUDA\v13.1;CUDACXX=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe;CUDAToolkit_ROOT=C:\PROGRA~1\NVIDIA~2\CUDA\v13.1;CudaToolkitDir=C:\PROGRA~1\NVIDIA~2\CUDA\v13.1;PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin
# ------------------------------- 初始化 CMake 与 Windows 编译探测策略 -------------------------------
if(POLICY CMP0141)
  # 使用新版 MSVC 调试信息策略，使后续 CMAKE_MSVC_DEBUG_INFORMATION_FORMAT 能稳定生效。
  cmake_policy(SET CMP0141 NEW)
endif()

if(WIN32)
  # Windows 下让本项目沿用 PyTorch 注入的 /Z7 调试信息格式，避免 /Zi 与 /Z7 互相覆盖。
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
      "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
  # CMake 的编译器自检和 Torch 的 try_run 会临时链接 exe，提前补 runtimeobject 解决 RoInitialize 依赖。
  string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " /DEFAULTLIB:runtimeobject.lib")
endif()

# 直接使用 CMake 构建时，configure/build 只生成二进制，install 才会把扩展放回 Python 包目录。
# 示例主线：cmake -S <源码目录> -B <构建目录> 后执行 cmake --build，再执行 cmake --install。
# 若只构建单个扩展目标，也必须安装同名 component，editable 包才能导入对应 .pyd/.so。
project(cfie_extensions LANGUAGES CXX)

# 项目主机侧 C++ 代码使用 C++17，CUDA host 代码也会通过 Torch/CMake 继承该标准。
set(CMAKE_CXX_STANDARD 17)
# 强制要求编译器支持 C++17，避免静默降级后在模板或 PyTorch 头文件处失败。
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
  # Windows 主机编译统一使用 UTF-8，并屏蔽第三方头文件常见的无关窄化与编码告警。
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
    $<$<COMPILE_LANGUAGE:CXX>:/wd4819>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4267>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819>)
  # CCCL 在 MSVC 传统预处理器模式下会产生提示，这里显式声明使用兼容路径。
  add_compile_definitions(
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING=1)

  # ------------------------------- 给 Windows 扩展目标补齐统一编译卫生参数 -------------------------------
  function(cfie_apply_windows_extension_hygiene target_name)
    # CXX 目标强制包含兼容头，集中修补 Windows/CUDA/PyTorch 头文件差异。
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/FI${CFIE_WINDOWS_FORCE_INCLUDE_HEADER}>
      # /bigobj 避免模板展开后的对象文件节数量超过 MSVC 默认限制。
      $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
      # /Zc:preprocessor 使用标准预处理器，减少 CUDA/CCCL 宏展开兼容问题。
      $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
      # CUDA 文件通过 --pre-include 注入同一兼容头，使 nvcc 的 host 编译阶段也一致。
      $<$<COMPILE_LANGUAGE:CUDA>:--pre-include=${CFIE_WINDOWS_FORCE_INCLUDE_HEADER_CMAKE}>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
  endfunction()
endif()


# ------------------------------- 读取基础构建入口变量 -------------------------------
# VLLM_TARGET_DEVICE 是 setup.py 注入的目标后端，CFIE 当前只允许 CUDA 原生扩展构建。
set(VLLM_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for vLLM")
# 打印构建类型，便于从 pip 或 CLion 日志快速判断当前是 Release/Debug/RelWithDebInfo。
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
# 打印目标设备，确认 setup.py 或 CLion 传入的后端变量没有被外部覆盖。
message(STATUS "Target device: ${VLLM_TARGET_DEVICE}")
if (NOT VLLM_TARGET_DEVICE STREQUAL "cuda")
  # 非 CUDA 后端在当前 CFIE Windows 适配链路中未维护，提前失败避免进入错误 target 分支。
  message(FATAL_ERROR "CFIE native build currently supports CUDA only.")
endif()

# 引入 vLLM/CFIE 复用的 CMake helper，后续 define_extension_target 和架构筛选都来自这里。
include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# VLLM_PYTHON_PATH 主要给代码生成脚本使用，这里读取一次避免 CMake 报手动变量未使用。
set(ignoreMe "${VLLM_PYTHON_PATH}")

# 禁止 FetchContent 依赖默认参与 install，避免 cutlass 等第三方目标被安装进 Python 包。
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

# ------------------------------- 约束 Python 与 PyTorch 版本 -------------------------------
# Python 版本列表用于 find_python_from_executable 精确匹配当前虚拟环境。
# 该列表需要与 setup.py 支持范围保持一致，否则 pip 与手动 CMake 会选出不同解释器。
set(PYTHON_SUPPORTED_VERSIONS "3.10" "3.11" "3.12" "3.13")

# HIP 分支保留 AMD GPU 架构列表，虽然当前 CFIE 入口强制 CUDA，但复用逻辑仍需要该变量。
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201")

# ROCm 默认安装前缀用于 HIP 兼容分支，允许调用方通过 -DROCM_PATH 覆盖。
if(NOT DEFINED ROCM_PATH)
  set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm installation prefix")
else()
  set(ROCM_PATH ${ROCM_PATH} CACHE PATH "ROCm installation prefix" FORCE)
endif()
# 期望的 CUDA/ROCm PyTorch 版本只用于告警，不阻断构建，方便本地调试新 wheel。
# CUDA 版本来源应与 pyproject 和 requirements 保持一致，ROCm 版本来源应与 rocm 镜像保持一致。
set(TORCH_SUPPORTED_VERSION_CUDA "2.10.0")
set(TORCH_SUPPORTED_VERSION_ROCM "2.10.0")

# ------------------------------- 锁定当前 pip/CLion 使用的 Python 解释器 -------------------------------
# 只接受 VLLM_PYTHON_EXECUTABLE 指向的解释器，避免 CMake 自动找到系统 Python。
if (VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set VLLM_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

# 将当前 Python 环境中的 torch CMake 前缀追加到 CMAKE_PREFIX_PATH，供 find_package(Torch) 使用。
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

# ------------------------------- 规范化 CUDA Toolkit 输入并修补 Windows 探测状态 -------------------------------
# 先探测 PATH 中的 nvcc，配合 CUDA_FOUND 分支给出更直接的缺失提示。
find_program(NVCC_EXECUTABLE nvcc)
if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif()

# CFIE_CUDA_COMPILE_ROOT 表示本次构建认定的 CUDA 根目录，后续 include/lib 都从它派生。
set(CFIE_CUDA_COMPILE_ROOT "")
if(DEFINED CUDAToolkit_ROOT AND EXISTS "${CUDAToolkit_ROOT}")
  # 新版 CMake 的 CUDAToolkit_ROOT 优先级最高，通常由 setup.py 传入无空格短路径。
  file(TO_CMAKE_PATH "${CUDAToolkit_ROOT}" CFIE_CUDA_COMPILE_ROOT)
elseif(DEFINED CUDA_TOOLKIT_ROOT_DIR AND EXISTS "${CUDA_TOOLKIT_ROOT_DIR}")
  # 旧版 FindCUDA 使用 CUDA_TOOLKIT_ROOT_DIR，因此也接受该变量作为根目录来源。
  file(TO_CMAKE_PATH "${CUDA_TOOLKIT_ROOT_DIR}" CFIE_CUDA_COMPILE_ROOT)
elseif(DEFINED ENV{CUDAToolkit_ROOT} AND EXISTS "$ENV{CUDAToolkit_ROOT}")
  # CLion 可能只通过环境变量注入 CUDAToolkit_ROOT，这里补齐到 CMake 普通变量。
  file(TO_CMAKE_PATH "$ENV{CUDAToolkit_ROOT}" CFIE_CUDA_COMPILE_ROOT)
elseif(DEFINED ENV{CUDA_TOOLKIT_ROOT_DIR} AND EXISTS "$ENV{CUDA_TOOLKIT_ROOT_DIR}")
  # 兼容只设置旧环境变量的手动 CMake 调用。
  file(TO_CMAKE_PATH "$ENV{CUDA_TOOLKIT_ROOT_DIR}" CFIE_CUDA_COMPILE_ROOT)
endif()

if(CFIE_CUDA_COMPILE_ROOT)
  # 强制写回旧变量，保证 Torch vendored FindCUDA 和后续 CUDAToolkit 查找看到同一根目录。
  set(CUDA_TOOLKIT_ROOT_DIR "${CFIE_CUDA_COMPILE_ROOT}" CACHE PATH
      "CUDA toolkit root used by CFIE" FORCE)
  # 强制写回新变量，保证 find_package(CUDAToolkit) 与 FindCUDA 不会选到不同版本。
  set(CUDAToolkit_ROOT "${CFIE_CUDA_COMPILE_ROOT}" CACHE PATH
      "CUDAToolkit root used by CFIE" FORCE)
endif()

if(WIN32)
  # Windows 下 CMake/CLion 偶尔在早期自检阶段未写出 CMAKE_SIZEOF_VOID_P，需要额外判断 x64。
  set(CFIE_WINDOWS_IS_64_BIT FALSE)
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # CMake 已探测到 64 位指针时直接认定为 x64 构建。
    set(CFIE_WINDOWS_IS_64_BIT TRUE)
  elseif(CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64")
    # MSVC 编译器架构显示 x64 时，即使 CMAKE_SIZEOF_VOID_P 尚未缓存也按 64 位处理。
    set(CFIE_WINDOWS_IS_64_BIT TRUE)
  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "x64")
    # Visual Studio 多配置生成器通过 -A x64 表达目标平台，这里也纳入判断。
    set(CFIE_WINDOWS_IS_64_BIT TRUE)
  elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(AMD64|x86_64|ARM64|arm64)$")
    # 主机处理器为 64 位时作为最后兜底，避免旧 FindCUDA 误走 lib/Win32。
    set(CFIE_WINDOWS_IS_64_BIT TRUE)
  endif()

  if(CFIE_WINDOWS_IS_64_BIT)
    if(NOT DEFINED CMAKE_SIZEOF_VOID_P OR NOT CMAKE_SIZEOF_VOID_P)
      # 给旧 FindCUDA 补齐指针宽度，使其搜索 CUDA/lib/x64 而不是 CUDA/lib/Win32。
      set(CMAKE_SIZEOF_VOID_P 8)
    endif()
    # 强制 CUDA device code 使用 64 位模式，匹配 Windows x64 的 host 编译与库目录。
    set(CUDA_64_BIT_DEVICE_CODE ON CACHE BOOL
        "Compile device code in 64 bit mode" FORCE)
  endif()
endif()

# ------------------------------- 导入 Torch 并启用 CUDA/HIP 语言环境 -------------------------------
# Torch CMake 包会内部调用 FindCUDA/enable_language(CUDA)，因此这里不手动 check_language。
find_package(Torch REQUIRED)

if(CFIE_CUDA_COMPILE_ROOT)
  # Torch 导入后再次写回 CUDA 根目录，避免 Torch 内部查找逻辑修改后续 CFIE 的 include/lib 来源。
  set(CUDA_TOOLKIT_ROOT_DIR "${CFIE_CUDA_COMPILE_ROOT}" CACHE PATH
      "CUDA toolkit root used by CFIE" FORCE)
  set(CUDAToolkit_ROOT "${CFIE_CUDA_COMPILE_ROOT}" CACHE PATH
      "CUDAToolkit root used by CFIE" FORCE)
endif()

# CFIE_CUDA_INCLUDE_DIRS 汇总传给 CXX 与 CUDA 的 CUDA 头文件路径。
set(CFIE_CUDA_INCLUDE_DIRS "")
if(CFIE_CUDA_COMPILE_ROOT)
  # CUDA include 是所有 CUDA/C++ 扩展都需要的基础头文件目录。
  list(APPEND CFIE_CUDA_INCLUDE_DIRS "${CFIE_CUDA_COMPILE_ROOT}/include")
  if(EXISTS "${CFIE_CUDA_COMPILE_ROOT}/include/cccl")
    # CUDA 13 把 CCCL 头文件独立到 include/cccl，显式加入可避免 MSVC 找不到 cuda/std 相关头。
    list(APPEND CFIE_CUDA_INCLUDE_DIRS "${CFIE_CUDA_COMPILE_ROOT}/include/cccl")
  endif()
  # 同步写入 CMake CUDA language、CUDAToolkit 和旧 FindCUDA 变量，保持三套查找逻辑一致。
  set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES ${CFIE_CUDA_INCLUDE_DIRS})
  set(CUDAToolkit_INCLUDE_DIRS ${CFIE_CUDA_INCLUDE_DIRS})
  set(CUDA_INCLUDE_DIRS ${CFIE_CUDA_INCLUDE_DIRS})
endif()

# 这些变量在 Windows 下保存 CUDA 静态导入库绝对路径，避免 target_link_libraries 搜错目录。
set(CFIE_CUDA_CUDART_LIBRARY "")
set(CFIE_CUDA_DRIVER_LIBRARY "")
set(CFIE_CUDA_BLAS_LIBRARIES "")
if(WIN32 AND CFIE_CUDA_COMPILE_ROOT)
  # cudart.lib 是 CUDA runtime 的导入库，部分第三方 CMake 目标需要显式路径。
  set(CFIE_CUDA_CUDART_LIBRARY "${CFIE_CUDA_COMPILE_ROOT}/lib/x64/cudart.lib")
  # cuda.lib 是 driver API 导入库，cumem_allocator 直接依赖该库。
  set(CFIE_CUDA_DRIVER_LIBRARY "${CFIE_CUDA_COMPILE_ROOT}/lib/x64/cuda.lib")
  # cuBLAS 与 cuBLASLt 在 Windows 下显式加入绝对路径，规避 Store/UWP LIB 环境污染。
  list(APPEND CFIE_CUDA_BLAS_LIBRARIES
       "${CFIE_CUDA_COMPILE_ROOT}/lib/x64/cublas.lib"
       "${CFIE_CUDA_COMPILE_ROOT}/lib/x64/cublasLt.lib")
endif()

if(WIN32)
  # 兼容头集中存放 Windows/CUDA 适配宏，后续对每个扩展目标统一强制包含。
  set(CFIE_WINDOWS_FORCE_INCLUDE_HEADER
      "${CMAKE_CURRENT_SOURCE_DIR}/csrc/windows_cuda_compat.h")
  # CUDA 的 --pre-include 使用 CMake 风格路径，避免反斜杠被 nvcc 解析成转义。
  file(TO_CMAKE_PATH "${CFIE_WINDOWS_FORCE_INCLUDE_HEADER}"
       CFIE_WINDOWS_FORCE_INCLUDE_HEADER_CMAKE)
endif()

# ------------------------------- 根据 CUDA 编译器版本确定支持的 NVIDIA 架构集合 -------------------------------
# 该判断必须在 find_package(Torch) 后执行，因为 CMAKE_CUDA_COMPILER_VERSION 由 CUDA language 初始化写入。
if(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  # CUDA 13 支持 Blackwell 与更新的 arch 表达，因此保留 10.x/11.x/12.0 等新架构。
  set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")
elseif(DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  # CUDA 12.8 支持 Blackwell 早期架构，同时仍保留 7.x 兼容项。
  set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0")
else()
  # 旧 CUDA 编译器不支持 Blackwell 目标，只保留 Ampere/Hopper 之前常见架构。
  set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0")
endif()

# ------------------------------- 拦截非 CUDA 目标后端 -------------------------------
# 当前 CFIE 入口通常走 CUDA；该分支保留给历史 CPU/其他后端的外部脚本转发。
if (NOT VLLM_TARGET_DEVICE STREQUAL "cuda" AND
    NOT VLLM_TARGET_DEVICE STREQUAL "rocm")
    if (VLLM_TARGET_DEVICE STREQUAL "cpu")
        # CPU 后端交给独立 CMake 脚本处理，避免主 CUDA 扩展逻辑继续执行。
        include(${CMAKE_CURRENT_LIST_DIR}/cmake/cpu_extension.cmake)
    else()
        # 未识别后端不定义任何扩展目标，直接结束 configure。
        return()
    endif()
    return()
endif()

# ------------------------------- 确定 GPU 编译语言并校验 Torch 版本 -------------------------------
# CUDA_FOUND/HIP_FOUND 来自 Torch CMake 包，决定后续 define_extension_target 使用 CUDA 还是 HIP。
if (NOT HIP_FOUND AND CUDA_FOUND)
  # CUDA 路径是 CFIE Windows 适配和主要推理算子构建路径。
  set(VLLM_GPU_LANG "CUDA")

  if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
    # 版本不匹配只告警，方便开发者使用本地 nightly 或临时 wheel 验证构建链路。
    message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
      "expected for CUDA build, saw ${Torch_VERSION} instead.")
  endif()
elseif(HIP_FOUND)
  # HIP 分支保留 ROCm 构建能力，后续扩展目标会按 HIP 语言生成。
  set(VLLM_GPU_LANG "HIP")

  # Torch 只提供 ROCm 配置变量，不会自动让 CMake 识别 .hip 文件，因此这里显式启用 HIP 语言。
  enable_language(HIP)

  # ROCm 5/6 需要 PyTorch 版本满足最低预期，否则仅告警提示兼容风险。
  if (ROCM_VERSION_DEV_MAJOR GREATER_EQUAL 5 AND
      Torch_VERSION VERSION_LESS ${TORCH_SUPPORTED_VERSION_ROCM})
    message(WARNING "Pytorch version >= ${TORCH_SUPPORTED_VERSION_ROCM} "
      "expected for ROCm build, saw ${Torch_VERSION} instead.")
  endif()
else()
  # 既没有 CUDA 也没有 HIP 时无法构建任何 GPU 原生扩展。
  message(FATAL_ERROR "Can't find CUDA or HIP installation.")
endif()


# ------------------------------- 解析并筛选目标 GPU 架构 -------------------------------
if(VLLM_GPU_LANG STREQUAL "CUDA")
  # CUDA 架构从全局 flags 中抽取出来，后续按源文件单独设置 gencode 以减少不必要编译。
  clear_cuda_arches(CUDA_ARCH_FLAGS)
  # 将抽出的 gencode flags 归一化成升序唯一架构列表，便于后续 loose intersection。
  extract_unique_cuda_archs_ascending(CUDA_ARCHS "${CUDA_ARCH_FLAGS}")
  message(STATUS "CUDA target architectures: ${CUDA_ARCHS}")
  # 过滤掉当前 nvcc 不支持的架构，避免后续所有源文件继承无效 arch 导致构建失败。
  cuda_archs_loose_intersection(CUDA_ARCHS
    "${CUDA_SUPPORTED_ARCHS}" "${CUDA_ARCHS}")
  message(STATUS "CUDA supported target architectures: ${CUDA_ARCHS}")
else()
  # 非 CUDA 后端使用 helper 按当前语言筛选 GPU 架构，结果写入 VLLM_GPU_ARCHES。
  override_gpu_arches(VLLM_GPU_ARCHES
    ${VLLM_GPU_LANG}
    "${${VLLM_GPU_LANG}_SUPPORTED_ARCHS}")
endif()

# ------------------------------- 汇总 GPU 编译 flags -------------------------------
# 从 torch 查询当前 GPU 语言需要的基础编译参数，并写入 VLLM_GPU_FLAGS 供所有扩展复用。
get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})

# NVCC_THREADS 来自 setup.py/envs.py，用于把 nvcc 内部并发传到 CUDA 编译命令。
if(NVCC_THREADS AND VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()

# CUDA 13 使用新的 --compress-mode 写法，替代旧版压缩参数以匹配 nvcc 13 语义。
if(VLLM_GPU_LANG STREQUAL "CUDA" AND
   DEFINED CMAKE_CUDA_COMPILER_VERSION AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  list(APPEND VLLM_GPU_FLAGS "--compress-mode=size")
endif()

# CUDA 头文件也要加入 CXX flags，因为部分 .cpp binding 会直接包含 CUDA/CCCL 头。
if(VLLM_GPU_LANG STREQUAL "CUDA")
  foreach(CFIE_CUDA_INCLUDE_DIR IN LISTS CFIE_CUDA_INCLUDE_DIRS)
    # 使用显式 -I 保证 MSVC 编译 .cpp 时不依赖 CMake CUDA language 的 include 传播。
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -I\"${CFIE_CUDA_INCLUDE_DIR}\"")
  endforeach()
endif()

# ------------------------------- 准备 FetchContent 依赖缓存 -------------------------------
# C++ 第三方依赖通过 FetchContent 纳入构建，setup.py 会传入稳定 base dir 以复用缓存。
# 产生二进制的依赖应使用各自 BINARY_DIR，避免 Debug/Release 或不同生成器产物互相覆盖。
include(FetchContent)
# 提前创建依赖缓存目录，避免 FetchContent 在首次展开依赖时因父目录不存在失败。
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR})
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

if(VLLM_GPU_LANG STREQUAL "HIP")
  # HIP Debug 构建关闭优化并增加 ggdb3，便于 ROCm 侧调试生成的 host/device 代码。
  set(CMAKE_${VLLM_GPU_LANG}_FLAGS_DEBUG "${CMAKE_${VLLM_GPU_LANG}_FLAGS_DEBUG} -O0 -ggdb3")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb3")

  # HIP 头文件大量 [[nodiscard]] 告警会淹没有效错误，暂时统一屏蔽未使用结果告警。
  set(CMAKE_${VLLM_GPU_LANG}_FLAGS "${CMAKE_${VLLM_GPU_LANG}_FLAGS} -Wno-unused-result")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-result")
endif()

# ------------------------------- 定义 cumem_allocator 扩展目标 -------------------------------
# cumem_allocator 只包含轻量 C++ 源文件，用于封装 CUDA/ROCm driver memory allocator。

set(VLLM_CUMEM_EXT_SRC
  "csrc/cumem_allocator.cpp")

# 给 cumem_allocator 源文件设置架构 flags，使其与主扩展目标的 CUDA_ARCHS 保持一致。
set_gencode_flags_for_srcs(
  SRCS "${VLLM_CUMEM_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "HIP")
  message(STATUS "Enabling cumem allocator extension.")
  if(VLLM_GPU_LANG STREQUAL "CUDA")
    # CUDA 分支链接 driver API；Windows 直接使用前面解析出的 cuda.lib 绝对路径。
    if(WIN32 AND CFIE_CUDA_DRIVER_LIBRARY)
      list(APPEND CUMEM_LIBS "${CFIE_CUDA_DRIVER_LIBRARY}")
    else()
      # 非 Windows 使用 CUDAToolkit 导出的 CUDA::cuda_driver target。
      list(APPEND CUMEM_LIBS CUDA::cuda_driver)
    endif()
  else()
    # HIP 分支优先在 ROCM_PATH/lib 下查找 amdhip64 绝对路径，找不到再退回按库名链接。
    find_library(AMDHIP64_LIB
      NAMES amdhip64 libamdhip64.so
      PATHS ${ROCM_PATH}/lib
      NO_DEFAULT_PATH)
    if(AMDHIP64_LIB)
      # 找到绝对路径时直接加入链接列表，减少系统库搜索路径差异。
      message(STATUS "Found libamdhip64 at ${AMDHIP64_LIB}")
      list(APPEND CUMEM_LIBS ${AMDHIP64_LIB})
    else()
      # 未找到绝对路径时保留链接名，让系统 linker 在默认库路径中解析。
      message(WARNING "libamdhip64 not found in ${ROCM_PATH}/lib; falling back to linking 'amdhip64' by name")
      list(APPEND CUMEM_LIBS amdhip64)
    endif()
  endif()
  # define_extension_target 统一创建 Python 扩展模块并注册 install component。
  define_extension_target(
    cumem_allocator
    DESTINATION cfie
    LANGUAGE CXX
    SOURCES ${VLLM_CUMEM_EXT_SRC}
    LIBRARIES ${CUMEM_LIBS}
    USE_SABI 3.8
    WITH_SOABI)
endif()

# ------------------------------- 定义主扩展 _C 的基础源码集合 -------------------------------
# VLLM_EXT_SRC 先放所有平台共享的核心算子、量化算子和 torch binding。

set(VLLM_EXT_SRC
  "csrc/mamba/mamba_ssm/selective_scan_fwd.cu"
  "csrc/cache_kernels.cu"
  "csrc/cache_kernels_fused.cu"
  "csrc/attention/paged_attention_v1.cu"
  "csrc/attention/paged_attention_v2.cu"
  "csrc/attention/merge_attn_states.cu"
  "csrc/attention/vertical_slash_index.cu"
  "csrc/pos_encoding_kernels.cu"
  "csrc/activation_kernels.cu"
  "csrc/layernorm_kernels.cu"
  "csrc/fused_qknorm_rope_kernel.cu"
  "csrc/layernorm_quant_kernels.cu"
  "csrc/sampler.cu"
  "csrc/topk.cu"
  "csrc/cuda_view.cu"
  "csrc/quantization/gptq/q_gemm.cu"
  "csrc/quantization/w8a8/int8/scaled_quant.cu"
  "csrc/quantization/w8a8/fp8/common.cu"
  "csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu"
  "csrc/quantization/gguf/gguf_kernel.cu"
  "csrc/quantization/activation_kernels.cu"
  "csrc/cuda_utils_kernels.cu"
  "csrc/custom_all_reduce.cu"
  "csrc/windows_triton_compat_ops.cpp"
  "csrc/torch_bindings.cpp")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  # CUTLASS 只作为头文件库参与编译，避免额外构建 CUTLASS 示例或测试目标。
  SET(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")

  # CUTLASS_REVISION 用于远程 FetchContent 兜底路径，并保持日志中的版本信息稳定。
  set(CUTLASS_REVISION "v4.2.1")

  # 环境变量中的 VLLM_CUTLASS_SRC_DIR 优先级高于默认 third_party，便于本地替换 CUTLASS。
  if (DEFINED ENV{VLLM_CUTLASS_SRC_DIR})
    set(VLLM_CUTLASS_SRC_DIR $ENV{VLLM_CUTLASS_SRC_DIR})
  endif()

  if(NOT VLLM_CUTLASS_SRC_DIR)
    # 没有外部指定时优先使用仓库 vendored cutlass，避免 configure 阶段联网下载。
    set(VLLM_BUNDLED_CUTLASS_SRC_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass")
    if(EXISTS "${VLLM_BUNDLED_CUTLASS_SRC_DIR}/CMakeLists.txt")
      # 只有 vendored cutlass 目录完整时才启用，避免半初始化子模块导致误用。
      set(VLLM_CUTLASS_SRC_DIR "${VLLM_BUNDLED_CUTLASS_SRC_DIR}")
    endif()
  endif()

  if(VLLM_CUTLASS_SRC_DIR)
    if(NOT IS_ABSOLUTE VLLM_CUTLASS_SRC_DIR)
      # 将相对路径转成绝对路径，避免 FetchContent 在二级 build 目录下解析出错。
      get_filename_component(VLLM_CUTLASS_SRC_DIR "${VLLM_CUTLASS_SRC_DIR}" ABSOLUTE)
    endif()
    message(STATUS "The VLLM_CUTLASS_SRC_DIR is set, using ${VLLM_CUTLASS_SRC_DIR} for compilation")
    # CUTLASS_DIR 给后续 MLA/Machete 代码生成和 include 设置复用。
    set(CUTLASS_DIR "${VLLM_CUTLASS_SRC_DIR}"
        CACHE PATH "CUTLASS Repository Directory" FORCE)
    # CUTLASS_INCLUDE_DIR 是大多数 CUTLASS kernel 源文件需要的主 include 目录。
    set(CUTLASS_INCLUDE_DIR "${VLLM_CUTLASS_SRC_DIR}/include"
        CACHE PATH "CUTLASS Header Library" FORCE)
    # tools/util/include 提供部分示例/工具头，Machete 与 CUTLASS 扩展会依赖它。
    set(CUTLASS_TOOLS_UTIL_INCLUDE_DIR
        "${VLLM_CUTLASS_SRC_DIR}/tools/util/include"
        CACHE PATH "CUTLASS Tools util include directory" FORCE)
    # 使用本地源码目录声明 FetchContent，避免下载并让依赖目标进入当前构建图。
    FetchContent_Declare(cutlass SOURCE_DIR ${VLLM_CUTLASS_SRC_DIR})
  else()
    # 本地 cutlass 不存在时才走远程仓库兜底，适用于没有 vendored 子模块的环境。
    FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/nvidia/cutlass.git
        # GIT_TAG 必须与上方 CUTLASS_REVISION 保持一致，避免日志版本和实际拉取版本不一致。
        GIT_TAG ${CUTLASS_REVISION}
        GIT_PROGRESS TRUE

        # GIT_SHALLOW 只拉取指定 tag，减少首次 configure 下载体积。
        # 若未来 GIT_TAG 改为 commit hash，需要关闭 GIT_SHALLOW，否则 CMake 无法浅克隆该提交。
        GIT_SHALLOW TRUE
    )
  endif()
  # 展开 cutlass 依赖并创建它提供的 CMake target/include 变量。
  FetchContent_MakeAvailable(cutlass)

  # CUDA 后端追加 CUTLASS、FP4、稀疏、per-token group quant 等源文件到主扩展。
  list(APPEND VLLM_EXT_SRC
    "csrc/quantization/awq/gemm_kernels.cu"
    "csrc/permute_cols.cu"
    "csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu"
    "csrc/quantization/fp4/nvfp4_quant_entry.cu"
    "csrc/quantization/fp4/nvfp4_scaled_mm_entry.cu"
    "csrc/sparse/cutlass/sparse_scaled_mm_entry.cu"
    "csrc/cutlass_extensions/common.cpp"
    "csrc/quantization/w8a8/fp8/per_token_group_quant.cu"
    "csrc/quantization/w8a8/int8/per_token_group_quant.cu")

  # 对主扩展当前已有源文件设置基础架构 flags，后续按 kernel 家族还会覆盖更窄架构。
  set_gencode_flags_for_srcs(
    SRCS "${VLLM_EXT_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")

  # ------------------------------- 筛选并生成 Marlin kernel 源码 -------------------------------
  # 只有目标架构包含 Marlin 支持范围时才生成/编译 Marlin；sm90 仍保留以覆盖 Machete 不支持的形状。

  # FP16 输出 Marlin 模板主要面向 sm80+，PTX 用于前向兼容同代架构。
  cuda_archs_loose_intersection(MARLIN_ARCHS "8.0+PTX" "${CUDA_ARCHS}")
  # Turing 只支持 Marlin 的受限模板，单独筛出 sm75 源文件。
  cuda_archs_loose_intersection(MARLIN_SM75_ARCHS "7.5" "${CUDA_ARCHS}")
  # BF16 输出需要覆盖 sm80 与 sm90 PTX，因为部分路径依赖 BF16 atomicAdd。
  cuda_archs_loose_intersection(MARLIN_BF16_ARCHS "8.0+PTX;9.0+PTX" "${CUDA_ARCHS}")
  # FP8 Marlin 只启用 sm89 与 sm120，避开 sm80 不支持 FP8 和 sm90/sm100 缺少对应 QMMA 指令的问题。
  cuda_archs_loose_intersection(MARLIN_FP8_ARCHS "8.9;12.0" "${CUDA_ARCHS}")
  # 其他 Marlin 公共源覆盖 sm75 与 sm80+PTX，作为模板 kernel 外的入口实现。
  cuda_archs_loose_intersection(MARLIN_OTHER_ARCHS "7.5;8.0+PTX" "${CUDA_ARCHS}")

  if (MARLIN_OTHER_ARCHS)

    # Marlin 会按预置输入类型和调度组合生成源码，脚本 hash 与架构共同决定是否重新生成。
    set(MARLIN_GEN_SCRIPT
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantization/marlin/generate_kernels.py)
    # 脚本 MD5 用于检测生成逻辑变化，变化时强制重新生成 kernel 源码。
    file(MD5 ${MARLIN_GEN_SCRIPT} MARLIN_GEN_SCRIPT_HASH)
    # 架构列表也会影响生成结果，因此拼入 cache key。
    list(JOIN CUDA_ARCHS "," CUDA_ARCHS_STR)
    set(MARLIN_GEN_SCRIPT_HASH_AND_ARCH "${MARLIN_GEN_SCRIPT_HASH}(ARCH:${CUDA_ARCHS_STR})")

    message(STATUS "Marlin generation script hash: ${MARLIN_GEN_SCRIPT_HASH_AND_ARCH}")
    message(STATUS "Last run Marlin generate script hash: $CACHE{MARLIN_GEN_SCRIPT_HASH_AND_ARCH}")

    if (NOT DEFINED CACHE{MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
        OR NOT $CACHE{MARLIN_GEN_SCRIPT_HASH_AND_ARCH} STREQUAL ${MARLIN_GEN_SCRIPT_HASH_AND_ARCH})
      # 生成脚本通过 CMake env 执行，继承 PYTHONPATH 以便找到仓库内 Python 依赖。
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=$ENV{PYTHONPATH}
          ${Python_EXECUTABLE} ${MARLIN_GEN_SCRIPT} ${CUDA_ARCHS_STR}
        RESULT_VARIABLE marlin_generation_result
        OUTPUT_VARIABLE marlin_generation_result
        OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log
        ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log
      )

      if (NOT marlin_generation_result EQUAL 0)
        # 生成失败直接终止 configure，并提示查看落盘日志定位 Python 生成脚本错误。
        message(FATAL_ERROR "Marlin generation failed."
                            " Result: \"${marlin_generation_result}\""
                            "\nCheck the log for details: "
                            "${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log")
      else()
        # 生成成功后更新 cache key，下次 configure 可跳过同一组源码生成。
        set(MARLIN_GEN_SCRIPT_HASH_AND_ARCH ${MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
            CACHE STRING "Last run Marlin generate script hash and arch" FORCE)
        message(STATUS "Marlin generation completed successfully.")
      endif()
    else()
      # hash 与架构未变化时跳过生成，避免每次 configure 重写大量生成文件。
      message(STATUS "Marlin generation script has not changed, skipping generation.")
    endif()

    if (MARLIN_ARCHS)
      # sm80 FP16 模板 kernel 由生成脚本产出，按 MARLIN_ARCHS 设置独立 gencode。
      file(GLOB MARLIN_TEMPLATE_KERNEL_SRC "csrc/quantization/marlin/sm80_kernel_*_float16.cu")
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_TEMPLATE_KERNEL_SRC}"
        CUDA_ARCHS "${MARLIN_ARCHS}")
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        # CUDA 12.8+ 需要关闭静态全局模板 stub，避免生成模板 kernel 时触发 nvcc 链接问题。
        set_source_files_properties(${MARLIN_TEMPLATE_KERNEL_SRC}
          PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
      endif()
      list(APPEND VLLM_EXT_SRC ${MARLIN_TEMPLATE_KERNEL_SRC})

      # BF16 模板 kernel 使用单独架构集合，确保 BF16 atomic 路径只在支持架构上编译。
      file(GLOB MARLIN_TEMPLATE_BF16_KERNEL_SRC "csrc/quantization/marlin/sm80_kernel_*_bfloat16.cu")
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_TEMPLATE_BF16_KERNEL_SRC}"
        CUDA_ARCHS "${MARLIN_BF16_ARCHS}")
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        # 与 FP16 模板保持相同 nvcc 兼容参数，避免 CUDA 12.8+ 模板 stub 问题。
        set_source_files_properties(${MARLIN_TEMPLATE_BF16_KERNEL_SRC}
          PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
      endif()
      list(APPEND VLLM_EXT_SRC ${MARLIN_TEMPLATE_BF16_KERNEL_SRC})
    endif()

    if (MARLIN_SM75_ARCHS)
      # sm75 模板单独编译，避免把 Turing 受限 kernel 混入 sm80+ 架构列表。
      file(GLOB MARLIN_TEMPLATE_SM75_KERNEL_SRC "csrc/quantization/marlin/sm75_kernel_*.cu")
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_TEMPLATE_SM75_KERNEL_SRC}"
        CUDA_ARCHS "${MARLIN_SM75_ARCHS}")
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        # CUDA 12.8+ 对生成模板源统一关闭静态全局模板 stub。
        set_source_files_properties(${MARLIN_TEMPLATE_SM75_KERNEL_SRC}
          PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
      endif()
      list(APPEND VLLM_EXT_SRC ${MARLIN_TEMPLATE_SM75_KERNEL_SRC})
    endif()

    if (MARLIN_FP8_ARCHS)
      # FP8 模板只在前面筛出的 sm89/sm120 架构上编译。
      file(GLOB MARLIN_TEMPLATE_FP8_KERNEL_SRC "csrc/quantization/marlin/sm89_kernel_*.cu")
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_TEMPLATE_FP8_KERNEL_SRC}"
        CUDA_ARCHS "${MARLIN_FP8_ARCHS}")
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        # 保持 FP8 模板与其他 Marlin 生成模板的 nvcc 兼容参数一致。
        set_source_files_properties(${MARLIN_TEMPLATE_FP8_KERNEL_SRC}
          PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
      endif()
      list(APPEND VLLM_EXT_SRC ${MARLIN_TEMPLATE_FP8_KERNEL_SRC})
    endif()

    # Marlin 公共入口源文件不由模板 glob 产生，需要显式维护并设置 MARLIN_OTHER_ARCHS。
    set(MARLIN_SRCS
       "csrc/quantization/marlin/marlin.cu"
       "csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu"
       "csrc/quantization/marlin/gptq_marlin_repack.cu"
       "csrc/quantization/marlin/awq_marlin_repack.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_SRCS}"
      CUDA_ARCHS "${MARLIN_OTHER_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      # 公共入口源也包含模板实例化，CUDA 12.8+ 同样关闭静态全局模板 stub。
      set_source_files_properties(${MARLIN_SRCS}
        PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND VLLM_EXT_SRC "${MARLIN_SRCS}")

    message(STATUS "Building Marlin kernels for archs: ${MARLIN_OTHER_ARCHS}")
  else()
    message(STATUS "Not building Marlin kernels as no compatible archs found"
                   " in CUDA target architectures")
  endif()

  # ------------------------------- 按架构启用 AllSpark 量化 kernel -------------------------------
  # AllSpark kernel 只在 Ampere/Ada 等支持架构上编译，避免无效 arch 拉长构建时间。
  cuda_archs_loose_intersection(ALLSPARK_ARCHS "8.0;8.6;8.7;8.9" "${CUDA_ARCHS}")
  if (ALLSPARK_ARCHS)
    # AllSpark 源文件负责 GPTQ repack 与 w8a16 GEMM，作为主扩展的一部分编译。
    set(ALLSPARK_SRCS
       "csrc/quantization/gptq_allspark/allspark_repack.cu"
       "csrc/quantization/gptq_allspark/allspark_qgemm_w8a16.cu")
    set_gencode_flags_for_srcs(
      SRCS "${ALLSPARK_SRCS}"
      CUDA_ARCHS "${ALLSPARK_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${ALLSPARK_SRCS}")
    message(STATUS "Building AllSpark kernels for archs: ${ALLSPARK_ARCHS}")
  else()
    message(STATUS "Not building AllSpark kernels as no compatible archs found"
                   " in CUDA target architectures")
  endif()


  # ------------------------------- 按架构启用 CUTLASS scaled_mm kernel -------------------------------
  # SCALED_MM_3X_ARCHS 记录已由 CUTLASS 3.x 覆盖的架构，后续 CUTLASS 2.x 需要排除它们。
  set(SCALED_MM_3X_ARCHS)
  # Hopper 的 CUTLASS 3.x scaled_mm 需要 CUDA 12.0+ 与 sm90a 架构。
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a;" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND SCALED_MM_ARCHS)
    # sm90 scaled_mm 覆盖 FP8、INT8、AZP 和 blockwise FP8 多种入口。
    set(SRCS
       "csrc/quantization/w8a8/cutlass/scaled_mm_c3x_sm90.cu"
       "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8.cu"
       "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_int8.cu"
       "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_azp_sm90_int8.cu"
       "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm90_fp8.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_SM90=1")
    # 记录 sm90a 已由 CUTLASS 3.x 覆盖，后续 CUTLASS 2.x 不再重复构建该架构。
    list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
    message(STATUS "Building scaled_mm_c3x_sm90 for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND SCALED_MM_ARCHS)
      message(STATUS "Not building scaled_mm_c3x_sm90 as CUDA Compiler version is "
                     "not >= 12.0, we recommend upgrading to CUDA 12.0 or "
                     "later if you intend on running FP8 quantized models on "
                     "Hopper.")
    else()
      message(STATUS "Not building scaled_mm_c3x_sm90 as no compatible archs found "
                     "in CUDA target architectures")
    endif()
  endif()


  # GeForce Blackwell SM120 的 CUTLASS 3.x scaled_mm 需要 CUDA 12.8+。
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    # CUDA 13 使用 12.0f 表达 Blackwell consumer forward-compatible 架构。
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0f" "${CUDA_ARCHS}")
  else()
    # CUDA 12.8 使用 12.0a 表达同类架构，需与 nvcc 版本匹配。
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "12.0a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    # sm120 scaled_mm 源文件覆盖 Blackwell consumer 的 FP8 与 blockwise FP8 路径。
    set(SRCS
      "csrc/quantization/w8a8/cutlass/scaled_mm_c3x_sm120.cu"
      "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8.cu"
      "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu"
    )
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_SM120=1")
    # 记录 sm120 已由 CUTLASS 3.x 覆盖，后续 CUTLASS 2.x 不再重复构建该架构。
    list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
    message(STATUS "Building scaled_mm_c3x_sm120 for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
      message(STATUS "Not building scaled_mm_c3x_sm120 as CUDA Compiler version is "
                     "not >= 12.8, we recommend upgrading to CUDA 12.8 or "
                     "later if you intend on running FP8 quantized models on "
                     "Blackwell.")
    else()
      message(STATUS "Not building scaled_mm_c3x_120 as no compatible archs found "
                     "in CUDA target architectures")
    endif()
  endif()


  # Blackwell SM100/SM110 的 CUTLASS 3.x scaled_mm 同样需要 CUDA 12.8+。
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    # CUDA 13 使用 10.0f/11.0f 表达数据中心 Blackwell forward-compatible 架构。
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
  else()
    # CUDA 12.8 使用 10.x a 后缀表达 Blackwell 数据中心架构。
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    # sm100 scaled_mm 源文件覆盖 Blackwell 数据中心 FP8 与 blockwise FP8 路径。
    set(SRCS
      "csrc/quantization/w8a8/cutlass/scaled_mm_c3x_sm100.cu"
      "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm100_fp8.cu"
      "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm100_fp8.cu"
    )
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_SM100=1")
    # 记录 sm100/sm110 已由 CUTLASS 3.x 覆盖，后续 CUTLASS 2.x 不再重复构建该架构。
    list(APPEND SCALED_MM_3X_ARCHS "${SCALED_MM_ARCHS}")
    message(STATUS "Building scaled_mm_c3x_sm100 for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
      message(STATUS "Not building scaled_mm_c3x_sm100 as CUDA Compiler version is "
                     "not >= 12.8, we recommend upgrading to CUDA 12.8 or "
                     "later if you intend on running FP8 quantized models on "
                     "Blackwell.")
    else()
      message(STATUS "Not building scaled_mm_c3x_100 as no compatible archs found "
                     "in CUDA target architectures")
    endif()
  endif()

  # CUTLASS 2.x scaled_mm 只覆盖未被 CUTLASS 3.x 接管的剩余架构，并额外保留 8.9+PTX 的 FP8 路径。
  cuda_archs_loose_intersection(SCALED_MM_2X_ARCHS
    "7.5;8.0;8.7;8.9+PTX" "${CUDA_ARCHS}")
  # 从 2.x 架构列表中移除 3.x 已覆盖的架构，避免重复注册同名能力宏和重复编译。
  list(REMOVE_ITEM SCALED_MM_2X_ARCHS ${SCALED_MM_3X_ARCHS})
  if (SCALED_MM_2X_ARCHS)
    # CUTLASS 2.x scaled_mm 只有一个入口源文件，内部按宏区分能力。
    set(SRCS "csrc/quantization/w8a8/cutlass/scaled_mm_c2x.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${SCALED_MM_2X_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_C2X=1")
    message(STATUS "Building scaled_mm_c2x for archs: ${SCALED_MM_2X_ARCHS}")
  else()
    if (SCALED_MM_3X_ARCHS)
      message(STATUS "Not building scaled_mm_c2x as all archs are already built"
                     " for and covered by scaled_mm_c3x")
    else()
      message(STATUS "Not building scaled_mm_c2x as no compatible archs found "
                    "in CUDA target architectures")
    endif()
  endif()

  # ------------------------------- 按架构启用 2:4 稀疏 CUTLASS kernel -------------------------------
  # 2:4 稀疏 scaled_mm/compressor 依赖 CUDA 12.2+，且当前只支持 Hopper sm90a。
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a;" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.2 AND SCALED_MM_ARCHS)
    # 稀疏 scaled_mm 源文件进入主扩展，并通过宏暴露运行时入口。
    set(SRCS "csrc/sparse/cutlass/sparse_scaled_mm_c3x.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_SPARSE_SCALED_MM_C3X=1")
    message(STATUS "Building sparse_scaled_mm_c3x for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.2 AND SCALED_MM_ARCHS)
      message(STATUS "Not building sparse_scaled_mm_c3x kernels as CUDA Compiler version is "
                     "not >= 12.2, we recommend upgrading to CUDA 12.2 or later "
                     "if you intend on running FP8 sparse quantized models on Hopper.")
    else()
      message(STATUS "Not building sparse_scaled_mm_c3x as no compatible archs found "
                     "in CUDA target architectures")
    endif()
  endif()

  # ------------------------------- 按架构启用 NVFP4 kernel -------------------------------
  # GeForce Blackwell SM120 的 NVFP4 scaled_mm 需要 CUDA 12.8+。
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    # CUDA 13 使用 12.0f 表达 SM120 forward-compatible 架构。
    cuda_archs_loose_intersection(FP4_ARCHS "12.0f" "${CUDA_ARCHS}")
  else()
    # CUDA 12.8 使用 12.0a 表达 SM120 架构。
    cuda_archs_loose_intersection(FP4_ARCHS "12.0a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND FP4_ARCHS)
    # SM120 NVFP4 源文件同时覆盖量化、activation fusion、experts quant、scaled_mm 与 blockwise MoE。
    set(SRCS
      "csrc/quantization/fp4/nvfp4_quant_kernels.cu"
      "csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu"
      "csrc/quantization/fp4/nvfp4_experts_quant.cu"
      "csrc/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu"
      "csrc/quantization/fp4/nvfp4_blockwise_moe_kernel.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${FP4_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_NVFP4_SM120=1")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM120=1")
    message(STATUS "Building NVFP4 for archs: ${FP4_ARCHS}")
  else()
    message(STATUS "Not building NVFP4 as no compatible archs were found.")
    # 清空 FP4_ARCHS，防止后续逻辑误认为 SM120 FP4 已启用。
    set(FP4_ARCHS)
  endif()

  # Blackwell SM100/SM110 的 NVFP4 kernel 使用另一组架构表达，也需要 CUDA 12.8+。
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(FP4_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(FP4_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND FP4_ARCHS)
    set(SRCS
      "csrc/quantization/fp4/nvfp4_quant_kernels.cu"
      "csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu"
      "csrc/quantization/fp4/nvfp4_experts_quant.cu"
      "csrc/quantization/fp4/nvfp4_scaled_mm_kernels.cu"
      "csrc/quantization/fp4/nvfp4_blockwise_moe_kernel.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${FP4_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_NVFP4_SM100=1")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1")
    message(STATUS "Building NVFP4 for archs: ${FP4_ARCHS}")
  else()
    message(STATUS "Not building NVFP4 as no compatible archs were found.")
    # 清空 FP4_ARCHS，防止后续逻辑误认为 SM100 FP4 已启用。
    set(FP4_ARCHS)
  endif()

  # ------------------------------- 按架构启用 CUTLASS MLA kernel -------------------------------
  # MLA kernel 依赖 Blackwell/Hopper 新架构表达，并要求 CUDA 12.8+。
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(MLA_ARCHS "10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(MLA_ARCHS "10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND MLA_ARCHS)
    set(SRCS
      "csrc/attention/mla/sm100_cutlass_mla_kernel.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${MLA_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MLA=1")
    # MLA 额外依赖 CUTLASS 示例目录头文件，只挂到 MLA 源文件，避免污染其他编译单元。
    set_source_files_properties(${SRCS}
      PROPERTIES INCLUDE_DIRECTORIES "${CUTLASS_DIR}/examples/77_blackwell_fmha;${CUTLASS_DIR}/examples/common")
    message(STATUS "Building CUTLASS MLA for archs: ${MLA_ARCHS}")
  else()
    message(STATUS "Not building CUTLASS MLA as no compatible archs were found.")
    # 清空 MLA_ARCHS，防止后续逻辑误认为 MLA 已启用。
    set(MLA_ARCHS)
  endif()

  # ------------------------------- 按架构启用 CUTLASS MoE kernel -------------------------------
  # Hopper MoE grouped_mm 需要 CUDA 12.3+，moe_data 只在至少一个 MoE kernel 可用时才有意义。
  cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND SCALED_MM_ARCHS)
    # sm90 MoE grouped_mm 进入主扩展，并通过 ENABLE_CUTLASS_MOE_SM90 控制运行时注册。
    set(SRCS "csrc/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm90.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM90=1")
    message(STATUS "Building grouped_mm_c3x for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND SCALED_MM_ARCHS)
      message(STATUS "Not building grouped_mm_c3x kernels as CUDA Compiler version is "
                     "not >= 12.3, we recommend upgrading to CUDA 12.3 or later "
                     "if you intend on running FP8 quantized MoE models on Hopper.")
    else()
      message(STATUS "Not building grouped_mm_c3x as no compatible archs found "
                     "in CUDA target architectures.")
    endif()
  endif()

  # Blackwell MoE grouped_mm 需要 CUDA 12.8+，并根据 nvcc 版本选择 f/a 后缀架构表达。
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
    set(SRCS "csrc/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm100.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${SCALED_MM_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM100=1")
    message(STATUS "Building grouped_mm_c3x for archs: ${SCALED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND SCALED_MM_ARCHS)
      message(STATUS "Not building grouped_mm_c3x kernels as CUDA Compiler version is "
                     "not >= 12.8, we recommend upgrading to CUDA 12.8 or later "
                     "if you intend on running FP8 quantized MoE models on Blackwell.")
    else()
      message(STATUS "Not building grouped_mm_c3x as no compatible archs found "
                     "in CUDA target architectures.")
    endif()
  endif()

  # Expert-specialization MXFP8 blockscaled grouped kernel 只在 SM100+ 且 CUDA 12.8+ 时启用。
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(ES_MXFP8_GROUPED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(ES_MXFP8_GROUPED_MM_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND ES_MXFP8_GROUPED_MM_ARCHS)
    set(SRCS
      "csrc/moe/mxfp8_moe/cutlass_mxfp8_grouped_mm.cu"
      "csrc/moe/mxfp8_moe/mxfp8_experts_quant.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${ES_MXFP8_GROUPED_MM_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_ES_MXFP8_GROUPED_MM_SM100=1")
    message(STATUS "Building ES MXFP8 grouped kernels for archs: ${ES_MXFP8_GROUPED_MM_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8
        AND ES_MXFP8_GROUPED_MM_ARCHS)
      message(STATUS "Not building ES MXFP8 grouped kernels as CUDA Compiler version is "
                     "not >= 12.8.")
    else()
      message(STATUS "Not building ES MXFP8 grouped kernels as no compatible archs found "
                     "in CUDA target architectures.")
    endif()
  endif()

  # DeepSeek V3 fused A GEMM 需要 Hopper 及更新架构，CUDA 12.0+ 才参与编译。
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(DSV3_FUSED_A_GEMM_ARCHS "9.0a;10.0f;11.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(DSV3_FUSED_A_GEMM_ARCHS "9.0a;10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND DSV3_FUSED_A_GEMM_ARCHS)
    set(DSV3_FUSED_A_GEMM_SRC "csrc/dsv3_fused_a_gemm.cu")
    set_gencode_flags_for_srcs(
      SRCS "${DSV3_FUSED_A_GEMM_SRC}"
      CUDA_ARCHS "${DSV3_FUSED_A_GEMM_ARCHS}")
    list(APPEND VLLM_EXT_SRC ${DSV3_FUSED_A_GEMM_SRC})
    message(STATUS "Building dsv3_fused_a_gemm for archs: ${DSV3_FUSED_A_GEMM_ARCHS}")
  else()
    message(STATUS "Not building dsv3_fused_a_gemm as no compatible archs found "
                   "in CUDA target architectures.")
  endif()

  # moe_data.cu 为所有 CUTLASS MoE kernel 提供共享数据布局逻辑，仅在 MoE 支持架构上编译。
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0f;11.0f;12.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(CUTLASS_MOE_DATA_ARCHS "9.0a;10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND CUTLASS_MOE_DATA_ARCHS)
    set(SRCS "csrc/quantization/w8a8/cutlass/moe/moe_data.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${CUTLASS_MOE_DATA_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    message(STATUS "Building moe_data for archs: ${CUTLASS_MOE_DATA_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND CUTLASS_MOE_DATA_ARCHS)
      message(STATUS "Not building moe_data as CUDA Compiler version is "
                     "not >= 12.3, we recommend upgrading to CUDA 12.3 or later "
                     "if you intend on running FP8 quantized MoE models on Hopper or Blackwell.")
    else()
      message(STATUS "Not building moe_data as no compatible archs found "
                     "in CUDA target architectures.")
    endif()
  endif()

  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    # 为后续 Blackwell 相关 kernel 预筛 scaled_mm 架构，复用同一变量名承接目标集合。
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
  endif()

  # ------------------------------- 生成并启用 Machete kernel -------------------------------
  # Machete 当前只支持 Hopper sm90a，并要求 CUDA 12.0+。
  cuda_archs_loose_intersection(MACHETE_ARCHS "9.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND MACHETE_ARCHS)
    # Machete 根据预置类型和 schedule 生成源码，脚本 hash 变化时才重新生成。
    set(MACHETE_GEN_SCRIPT
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantization/machete/generate.py)
    file(MD5 ${MACHETE_GEN_SCRIPT} MACHETE_GEN_SCRIPT_HASH)

    message(STATUS "Machete generation script hash: ${MACHETE_GEN_SCRIPT_HASH}")
    message(STATUS "Last run machete generate script hash: $CACHE{MACHETE_GEN_SCRIPT_HASH}")

    if (NOT DEFINED CACHE{MACHETE_GEN_SCRIPT_HASH}
        OR NOT $CACHE{MACHETE_GEN_SCRIPT_HASH} STREQUAL ${MACHETE_GEN_SCRIPT_HASH})
      # 生成脚本需要 cutlass_extensions 与 CUTLASS python 目录，因此先拼接专用 PYTHONPATH。
      set(MACHETE_PYTHONPATH_ENTRIES
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass_extensions"
        "${CUTLASS_DIR}/python")
      if (DEFINED VLLM_PYTHON_PATH AND NOT "${VLLM_PYTHON_PATH}" STREQUAL "")
        # setup.py 传入的 VLLM_PYTHON_PATH 保证生成脚本能访问当前虚拟环境依赖。
        list(APPEND MACHETE_PYTHONPATH_ENTRIES "${VLLM_PYTHON_PATH}")
      endif()
      if (DEFINED ENV{PYTHONPATH} AND NOT "$ENV{PYTHONPATH}" STREQUAL "")
        # 保留用户原本的 PYTHONPATH，避免本地调试依赖被覆盖。
        list(APPEND MACHETE_PYTHONPATH_ENTRIES "$ENV{PYTHONPATH}")
      endif()

      if (WIN32)
        # Windows 的 PYTHONPATH 分隔符是分号。
        string(JOIN ";" MACHETE_PYTHONPATH ${MACHETE_PYTHONPATH_ENTRIES})
      else()
        # Linux/WSL 的 PYTHONPATH 分隔符是冒号。
        string(JOIN ":" MACHETE_PYTHONPATH ${MACHETE_PYTHONPATH_ENTRIES})
      endif()

      # 运行 Machete 生成脚本，并把 stdout/stderr 都落到同一个日志文件便于排查。
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E env
        "PYTHONPATH=${MACHETE_PYTHONPATH}"
          ${Python_EXECUTABLE} ${MACHETE_GEN_SCRIPT}
        RESULT_VARIABLE machete_generation_result
        OUTPUT_VARIABLE machete_generation_output
        OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log
        ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log
      )

      if (NOT machete_generation_result EQUAL 0)
        # 生成失败阻断 configure，避免后续 file(GLOB) 找不到生成源却继续构建。
        message(FATAL_ERROR "Machete generation failed."
                            " Result: \"${machete_generation_result}\""
                            "\nCheck the log for details: "
                            "${CMAKE_CURRENT_BINARY_DIR}/machete_generation.log")
      else()
        # 生成成功后写入脚本 hash，下次 configure 可跳过相同生成工作。
        set(MACHETE_GEN_SCRIPT_HASH ${MACHETE_GEN_SCRIPT_HASH}
            CACHE STRING "Last run machete generate script hash" FORCE)
        message(STATUS "Machete generation completed successfully.")
      endif()
    else()
      # 生成脚本未变化时保持已有 generated 源文件，减少 configure 时间。
      message(STATUS "Machete generation script has not changed, skipping generation.")
    endif()

    # 将 Machete 生成目录中的 CUDA 源加入主扩展，使生成结果参与后续编译。
    file(GLOB MACHETE_GEN_SOURCES "csrc/quantization/machete/generated/*.cu")
    list(APPEND VLLM_EXT_SRC ${MACHETE_GEN_SOURCES})

    # 生成源按 MACHETE_ARCHS 设置 gencode，PTX 路径用于 Hopper 前向兼容。
    set_gencode_flags_for_srcs(
      SRCS "${MACHETE_GEN_SOURCES}"
      CUDA_ARCHS "${MACHETE_ARCHS}")

    # Machete PyTorch binding 入口不是生成文件，需要显式加入主扩展源码。
    list(APPEND VLLM_EXT_SRC
      csrc/quantization/machete/machete_pytorch.cu)

    message(STATUS "Building Machete kernels for archs: ${MACHETE_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0
        AND MACHETE_ARCHS)
      message(STATUS "Not building Machete kernels as CUDA Compiler version is "
                     "not >= 12.0, we recommend upgrading to CUDA 12.0 or "
                     "later if you intend on running w4a16 quantized models on "
                     "Hopper.")
    else()
      message(STATUS "Not building Machete kernels as no compatible archs "
                     "found in CUDA target architectures")
    endif()
  endif()

  # ------------------------------- 按架构启用 W4A8 CUTLASS kernel -------------------------------
  # W4A8 kernel 当前只在 Hopper sm90a 且 CUDA 12.0+ 时编译。
  cuda_archs_loose_intersection(W4A8_ARCHS "9.0a" "${CUDA_ARCHS}")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND W4A8_ARCHS)
    # W4A8 源文件覆盖普通 GEMM、grouped GEMM 与工具函数。
    set(SRCS
       "csrc/quantization/cutlass_w4a8/w4a8_mm_entry.cu"
       "csrc/quantization/cutlass_w4a8/w4a8_grouped_mm_entry.cu"
       "csrc/quantization/cutlass_w4a8/w4a8_utils.cu"
       )

    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${W4A8_ARCHS}")

    list(APPEND VLLM_EXT_SRC "${SRCS}")

    message(STATUS "Building W4A8 kernels for archs: ${W4A8_ARCHS}")
  else()
    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0
        AND W4A8_ARCHS)
      message(STATUS "Not building W4A8 kernels as CUDA Compiler version is "
                     "not >= 12.0, we recommend upgrading to CUDA 12.0 or "
                     "later if you intend on running w4a16 quantized models on "
                     "Hopper.")
    else()
      message(STATUS "Not building W4A8 kernels as no compatible archs "
                     "found in CUDA target architectures")
    endif()
  endif()

  # ------------------------------- 按架构启用 Hadacore Hadamard kernel -------------------------------
  # Hadacore 支持 sm80/sm90 PTX，作为量化相关 Hadamard transform CUDA 实现加入主扩展。
  cuda_archs_loose_intersection(HADACORE_ARCHS "8.0+PTX;9.0+PTX" "${CUDA_ARCHS}")
  if(HADACORE_ARCHS)
    set(SRCS "csrc/quantization/hadamard/hadacore/hadamard_transform_cuda.cu")
    set_gencode_flags_for_srcs(
      SRCS "${SRCS}"
      CUDA_ARCHS "${HADACORE_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${SRCS}")
    message(STATUS "Building hadacore")
  endif()

# CUDA 专属 kernel 追加逻辑到这里结束，后续会统一定义 _C 扩展目标。
endif()

if (VLLM_GPU_LANG STREQUAL "HIP")
  # HIP 后端追加 QuickReduce kernel，使 ROCm 主扩展包含对应通信优化实现。
  list(APPEND VLLM_EXT_SRC
    "csrc/custom_quickreduce.cu"
  )
# HIP 专属源码追加逻辑到这里结束。
endif()

# ------------------------------- 创建主扩展 _C -------------------------------
message(STATUS "Enabling C extension.")
define_extension_target(
  _C
  DESTINATION cfie
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

if(WIN32)
  # Windows 下给 _C 统一追加强制 include、/bigobj 和标准预处理器等兼容参数。
  cfie_apply_windows_extension_hygiene(_C)
endif()

# CUTLASS 在 NVCC 12.5+ 默认通过 cudaGetDriverEntryPointByVersion 包装 driver API。
# 该包装在较旧 CUDA driver 链接场景下可能出问题，因此强制直接调用 driver API。
target_compile_definitions(_C PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)

# ------------------------------- 定义 MoE 扩展 _moe_C 的基础源码集合 -------------------------------
# VLLM_MOE_EXT_SRC 先收集所有 MoE 后端共享的 torch binding、align/sum 和 topk softmax kernel。

set(VLLM_MOE_EXT_SRC
  "csrc/moe/torch_bindings.cpp"
  "csrc/moe/moe_align_sum_kernels.cu"
  "csrc/moe/topk_softmax_kernels.cu")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  # CUDA MoE 分支追加 wna16、grouped topk 和 router GEMM 基础 kernel。
  list(APPEND VLLM_MOE_EXT_SRC
    "csrc/moe/moe_wna16.cu"
    "csrc/moe/grouped_topk_kernels.cu"
    "csrc/moe/router_gemm.cu")
endif()

if(VLLM_GPU_LANG STREQUAL "CUDA")
  # permute/unpermute 源文件成对加入，分别提供 CUDA kernel 与 PyTorch op 包装。
  set(MOE_PERMUTE_SRC
      "csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu"
      "csrc/moe/moe_permute_unpermute_op.cu")

  list(APPEND VLLM_MOE_EXT_SRC "${MOE_PERMUTE_SRC}")
endif()

# 先给 MoE 当前源码集合设置通用 CUDA 架构 flags，后续 Marlin MoE 会按源文件细分。
set_gencode_flags_for_srcs(
  SRCS "${VLLM_MOE_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

if(VLLM_GPU_LANG STREQUAL "CUDA")
  # moe_wna16 单独设置一次架构 flags，确保该核心源在目标架构集合上完整编译。
  set(VLLM_MOE_WNA16_SRC
    "csrc/moe/moe_wna16.cu")

  set_gencode_flags_for_srcs(
    SRCS "${VLLM_MOE_WNA16_SRC}"
    CUDA_ARCHS "${CUDA_ARCHS}")

  list(APPEND VLLM_MOE_EXT_SRC "${VLLM_MOE_WNA16_SRC}")
  # ------------------------------- 筛选并生成 Marlin MoE kernel 源码 -------------------------------
  # Marlin MoE 当前固定 use_atomic_add=False，因此不需要为了 BF16 atomicAdd 单独保留 sm90 PTX。
  cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0+PTX" "${CUDA_ARCHS}")
  # Turing 对 Marlin MoE 支持有限，sm75 源文件单独筛选。
  cuda_archs_loose_intersection(MARLIN_MOE_SM75_ARCHS "7.5" "${CUDA_ARCHS}")
  # FP8 Marlin MoE 只启用 sm89 和 sm120，避开不支持 FP8 或缺少对应 QMMA 指令的架构。
  cuda_archs_loose_intersection(MARLIN_MOE_FP8_ARCHS "8.9;12.0" "${CUDA_ARCHS}")
  # Marlin MoE 公共 ops 源覆盖 sm75 与 sm80+PTX。
  cuda_archs_loose_intersection(MARLIN_MOE_OTHER_ARCHS "7.5;8.0+PTX" "${CUDA_ARCHS}")
  if (MARLIN_MOE_OTHER_ARCHS)

    # Marlin MoE 按预置输入类型和 schedule 生成源码，脚本 hash 与架构共同决定是否重新生成。
    set(MOE_MARLIN_GEN_SCRIPT
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc/moe/marlin_moe_wna16/generate_kernels.py)
    # 记录生成脚本 hash，脚本变化时强制重跑生成。
    file(MD5 ${MOE_MARLIN_GEN_SCRIPT} MOE_MARLIN_GEN_SCRIPT_HASH)
    # 目标架构会影响生成文件集合，因此加入 cache key。
    list(JOIN CUDA_ARCHS "," CUDA_ARCHS_STR)
    set(MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH "${MOE_MARLIN_GEN_SCRIPT_HASH}(ARCH:${CUDA_ARCHS_STR})")

    message(STATUS "Marlin MOE generation script hash with arch: ${MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}")
    message(STATUS "Last run Marlin MOE generate script hash with arch: $CACHE{MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}")

    if (NOT DEFINED CACHE{MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
        OR NOT $CACHE{MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH} STREQUAL ${MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH})
      # 通过当前 Python 运行生成脚本，并继承 PYTHONPATH 以访问仓库内依赖。
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=$ENV{PYTHONPATH}
          ${Python_EXECUTABLE} ${MOE_MARLIN_GEN_SCRIPT} ${CUDA_ARCHS_STR}
        RESULT_VARIABLE moe_marlin_generation_result
        OUTPUT_VARIABLE moe_marlin_generation_output
        OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
        ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
      )

      if (NOT moe_marlin_generation_result EQUAL 0)
        # 生成失败直接终止 configure，避免后续 glob 到旧文件产生错误构建。
        message(FATAL_ERROR "Marlin MOE generation failed."
                            " Result: \"${moe_marlin_generation_result}\""
                            "\nCheck the log for details: "
                            "${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log")
      else()
        # 生成成功后更新 cache key，使同一脚本与架构组合下可以跳过生成。
        set(MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH ${MOE_MARLIN_GEN_SCRIPT_HASH_AND_ARCH}
            CACHE STRING "Last run Marlin MOE generate script hash" FORCE)
        message(STATUS "Marlin MOE generation completed successfully.")
      endif()
    else()
      # cache key 未变化时复用已有生成文件，减少 configure 时间。
      message(STATUS "Marlin MOE generation script has not changed, skipping generation.")
    endif()

    if (MARLIN_MOE_ARCHS)
      # sm80 Marlin MoE 模板 kernel 由生成脚本产出，并按 MARLIN_MOE_ARCHS 编译。
      file(GLOB MARLIN_MOE_SRC "csrc/moe/marlin_moe_wna16/sm80_kernel_*.cu")
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_MOE_SRC}"
        CUDA_ARCHS "${MARLIN_MOE_ARCHS}")
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        # CUDA 12.8+ 对生成模板源关闭静态全局模板 stub，避免 nvcc 链接问题。
        set_source_files_properties(${MARLIN_MOE_SRC}
          PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
      endif()
      list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_SRC})
    endif()

    if (MARLIN_MOE_SM75_ARCHS)
      # sm75 Marlin MoE 模板单独按 Turing 架构编译。
      file(GLOB MARLIN_MOE_SM75_SRC "csrc/moe/marlin_moe_wna16/sm75_kernel_*.cu")
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_MOE_SM75_SRC}"
        CUDA_ARCHS "${MARLIN_MOE_SM75_ARCHS}")
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        # 保持 sm75 生成模板与其他 Marlin MoE 模板的 nvcc 兼容参数一致。
        set_source_files_properties(${MARLIN_MOE_SM75_SRC}
          PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
      endif()
      list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_SM75_SRC})
    endif()

    if (MARLIN_MOE_FP8_ARCHS)
      # FP8 Marlin MoE 模板只在 sm89/sm120 支持架构上编译。
      file(GLOB MARLIN_MOE_FP8_SRC "csrc/moe/marlin_moe_wna16/sm89_kernel_*.cu")
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_MOE_FP8_SRC}"
        CUDA_ARCHS "${MARLIN_MOE_FP8_ARCHS}")
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        # 保持 FP8 生成模板与其他 Marlin MoE 模板的 nvcc 兼容参数一致。
        set_source_files_properties(${MARLIN_MOE_FP8_SRC}
          PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
      endif()
      list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_FP8_SRC})
    endif()

    # ops.cu 是 Marlin MoE 公共入口源文件，按 MARLIN_MOE_OTHER_ARCHS 编译。
    set(MARLIN_MOE_OTHER_SRC "csrc/moe/marlin_moe_wna16/ops.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_MOE_OTHER_SRC}"
      CUDA_ARCHS "${MARLIN_MOE_OTHER_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      # 公共入口源同样关闭静态全局模板 stub，以兼容 CUDA 12.8+。
      set_source_files_properties(${MARLIN_MOE_OTHER_SRC}
        PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND VLLM_MOE_EXT_SRC "${MARLIN_MOE_OTHER_SRC}")

    message(STATUS "Building Marlin MOE kernels for archs: ${MARLIN_MOE_OTHER_ARCHS}")
  else()
    message(STATUS "Not building Marlin MOE kernels as no compatible archs found"
                   " in CUDA target architectures")
  endif()

  # DeepSeek V3 router GEMM 需要 SM90+，根据 CUDA 版本选择 Blackwell 架构后缀表达。
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(DSV3_ROUTER_GEMM_ARCHS "9.0a;10.0f;11.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(DSV3_ROUTER_GEMM_ARCHS "9.0a;10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND DSV3_ROUTER_GEMM_ARCHS)
    set(DSV3_ROUTER_GEMM_SRC
      "csrc/moe/dsv3_router_gemm_entry.cu"
      "csrc/moe/dsv3_router_gemm_float_out.cu"
      "csrc/moe/dsv3_router_gemm_bf16_out.cu")
    set_gencode_flags_for_srcs(
      SRCS "${DSV3_ROUTER_GEMM_SRC}"
      CUDA_ARCHS "${DSV3_ROUTER_GEMM_ARCHS}")
    list(APPEND VLLM_MOE_EXT_SRC "${DSV3_ROUTER_GEMM_SRC}")
    message(STATUS "Building DSV3 router GEMM kernel for archs: ${DSV3_ROUTER_GEMM_ARCHS}")
  else()
    message(STATUS "Not building DSV3 router GEMM kernel as no compatible archs found"
                   " (requires SM90+ and CUDA >= 12.0)")
  endif()
endif()

# ------------------------------- 创建 MoE 扩展 _moe_C -------------------------------
message(STATUS "Enabling moe extension.")
define_extension_target(
  _moe_C
  DESTINATION cfie
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_MOE_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

if(WIN32)
  # Windows 下给 _moe_C 追加与 _C 相同的编译卫生参数，保持所有 CUDA 扩展一致。
  cfie_apply_windows_extension_hygiene(_moe_C)
endif()

if(VLLM_GPU_LANG STREQUAL "HIP")
  # ------------------------------- 创建 ROCm 专用扩展 _rocm_C -------------------------------
  # HIP 后端额外构建 ROCm skinny gemm 与 attention binding。
  set(VLLM_ROCM_EXT_SRC
    "csrc/rocm/torch_bindings.cpp"
    "csrc/rocm/skinny_gemms.cu"
    "csrc/rocm/attention.cu")

  define_extension_target(
    _rocm_C
    DESTINATION cfie
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_ROCM_EXT_SRC}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    USE_SABI 3
    WITH_SOABI)
endif()

# CUDA 后端构建 vendored vllm-flash-attn 扩展，具体目标由外部 CMake 脚本维护。
if (VLLM_GPU_LANG STREQUAL "CUDA")
    include(cmake/external_projects/vllm_flash_attn.cmake)
endif ()
