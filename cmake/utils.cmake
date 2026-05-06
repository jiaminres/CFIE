# ------------------------------- 绑定指定 Python 可执行文件 -------------------------------
# 让后续 Python/CMake 探测严格跟随外部传入的解释器路径。
macro (find_python_from_executable EXECUTABLE SUPPORTED_VERSIONS)
  # 先把输入路径归一化成 CMake 路径风格。
  file(TO_CMAKE_PATH "${EXECUTABLE}" _EXECUTABLE_CMAKE_PATH)
  # 再解析真实路径，消掉符号链接和相对路径差异。
  file(REAL_PATH "${_EXECUTABLE_CMAKE_PATH}" EXECUTABLE)
  # 把归一化后的解释器写回给后续 find_package(Python) 使用。
  set(Python_EXECUTABLE "${EXECUTABLE}")
  # 按当前解释器探测解释器、开发头和模块构建组件。
  find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule)
  if (NOT Python_FOUND)
    # 找不到匹配的 Python 时立刻终止 configure。
    message(FATAL_ERROR "Unable to find python matching: ${EXECUTABLE}.")
  endif()
  # 取出当前解释器的主次版本号。
  set(_VER "${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}")
  # 把首参和可变参数一起拼成允许的版本白名单。
  set(_SUPPORTED_VERSIONS_LIST ${SUPPORTED_VERSIONS} ${ARGN})
  if (NOT _VER IN_LIST _SUPPORTED_VERSIONS_LIST)
    # 解释器版本不在白名单时拒绝继续构建。
    message(FATAL_ERROR
      "Python version (${_VER}) is not one of the supported versions: "
      "${_SUPPORTED_VERSIONS_LIST}.")
  endif()
  # 输出最终绑定到的 Python 路径，便于排查多解释器环境。
  message(STATUS "Found python matching: ${EXECUTABLE}.")
endmacro()

# ------------------------------- 执行一段 Python 表达式 -------------------------------
# 用当前绑定的 Python 解释器执行表达式，并把标准输出回传给调用方。
function (run_python OUT EXPR ERR_MSG)
  # 启动外部 Python 进程执行传入的一段代码。
  execute_process(
    COMMAND
    "${Python_EXECUTABLE}" "-c" "${EXPR}"
    # 把标准输出捕获到变量，供上层继续解析。
    OUTPUT_VARIABLE PYTHON_OUT
    # 记录子进程退出码，用于判断是否执行成功。
    RESULT_VARIABLE PYTHON_ERROR_CODE
    # 捕获标准错误，失败时拼进报错信息。
    ERROR_VARIABLE PYTHON_STDERR
    # 去掉输出末尾换行，避免后续列表或路径处理带脏字符。
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT PYTHON_ERROR_CODE EQUAL 0)
    # Python 执行失败时把调用方给出的上下文错误一起抛出。
    message(FATAL_ERROR "${ERR_MSG}: ${PYTHON_STDERR}")
  endif()
  # 把执行结果写回调用方作用域。
  set(${OUT} ${PYTHON_OUT} PARENT_SCOPE)
endfunction()

# ------------------------------- 扩展 CMAKE_PREFIX_PATH -------------------------------
# 先导入指定 Python 包，再把表达式结果追加到 CMAKE_PREFIX_PATH。
macro (append_cmake_prefix_path PKG EXPR)
  # 通过 Python 取出包对应的 CMake 前缀路径。
  run_python(_PREFIX_PATH
    "import ${PKG}; print(${EXPR})" "Failed to locate ${PKG} path")
  # 把探测到的路径追加到 CMAKE_PREFIX_PATH，供后续 find_package 使用。
  list(APPEND CMAKE_PREFIX_PATH ${_PREFIX_PATH})
endmacro()

# ------------------------------- 生成 HIP 预处理目标 -------------------------------
# 把一组 CUDA 源文件转换成 HIP 版本，并把输出源码列表回传给调用方。
function (hipify_sources_target OUT_SRCS NAME ORIG_SRCS)
  # 先复制一份原始源码列表，后面按扩展名分流。
  set(SRCS ${ORIG_SRCS})
  # 单独保留 C++/HIP 源文件，避免它们被当成待 hipify 的 CUDA 文件。
  set(CXX_SRCS ${ORIG_SRCS})
  # 从待转换列表里排除 .cc/.cpp/.hip，只留下真正的 CUDA 源。
  list(FILTER SRCS EXCLUDE REGEX "\.(cc)|(cpp)|(hip)$")
  # 从原始列表里筛出无需转换的 C++/HIP 源。
  list(FILTER CXX_SRCS INCLUDE REGEX "\.(cc)|(cpp)|(hip)$")

  # 初始化生成后的 HIP 源文件列表。
  set(HIP_SRCS)
  foreach (SRC ${SRCS})
    # 先把 .cu 后缀替换成 .hip。
    string(REGEX REPLACE "\.cu$" "\.hip" SRC ${SRC})
    # 再把路径中的 cuda 目录名替换成 hip。
    string(REGEX REPLACE "cuda" "hip" SRC ${SRC})
    # 生成文件统一落到当前 build 目录下，而不是源码树。
    list(APPEND HIP_SRCS "${CMAKE_CURRENT_BINARY_DIR}/${SRC}")
  endforeach()

  # 约定所有 hipify 输出都放到 build/csrc 下。
  set(CSRC_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/csrc)
  # 创建一个自定义目标，专门负责运行 hipify 预处理脚本。
  add_custom_target(
    hipify${NAME}
    # 用当前 Python 解释器执行仓库里的 hipify.py。
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/hipify.py -p ${CMAKE_SOURCE_DIR}/csrc -o ${CSRC_BUILD_DIR} ${SRCS}
    # 脚本本身和原始 CUDA 源变化时都需要重新生成。
    DEPENDS ${CMAKE_SOURCE_DIR}/cmake/hipify.py ${SRCS}
    # 告诉构建系统这批 HIP 文件是该目标的副产物。
    BYPRODUCTS ${HIP_SRCS}
    # 构建日志里显示当前正在执行的 hipify 任务。
    COMMENT "Running hipify on ${NAME} extension source files.")

  # 把无需转换的 C++/HIP 源追加回最终源码列表。
  list(APPEND HIP_SRCS ${CXX_SRCS})
  # 把转换后的完整源码列表回传给调用方。
  set(${OUT_SRCS} ${HIP_SRCS} PARENT_SCOPE)
endfunction()

# ------------------------------- 读取 Torch GPU 编译参数 -------------------------------
# 从当前 torch wheel 提取 CUDA/HIP 扩展构建所需的基础编译选项。
function (get_torch_gpu_compiler_flags OUT_GPU_FLAGS GPU_LANG)
  if (${GPU_LANG} STREQUAL "CUDA")
    # 从 torch 读取通用 NVCC flags，保证扩展口径与 torch 一致。
    run_python(GPU_FLAGS
      "from torch.utils.cpp_extension import COMMON_NVCC_FLAGS; print(';'.join(COMMON_NVCC_FLAGS))"
      "Failed to determine torch nvcc compiler flags")

    if (CUDA_VERSION VERSION_GREATER_EQUAL 11.8)
      # CUDA 11.8+ 打开 FP8 宏，允许后续编译对应代码路径。
      list(APPEND GPU_FLAGS "-DENABLE_FP8")
    endif()
    if (CUDA_VERSION VERSION_GREATER_EQUAL 12.0)
      # CUDA 12+ 删除 torch 默认禁掉 half/bfloat16 运算的旧宏。
      list(REMOVE_ITEM GPU_FLAGS
        "-D__CUDA_NO_HALF_OPERATORS__"
        "-D__CUDA_NO_HALF_CONVERSIONS__"
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__"
        "-D__CUDA_NO_HALF2_OPERATORS__")
    endif()

  elseif(${GPU_LANG} STREQUAL "HIP")
    # 从 torch 读取通用 HIP/HIPCC flags，沿用 torch 的 ROCm 构建口径。
    run_python(GPU_FLAGS
      "import torch.utils.cpp_extension as t; print(';'.join(t.COMMON_HIP_FLAGS + t.COMMON_HIPCC_FLAGS))"
      "Failed to determine torch nvcc compiler flags")

    # 再补上项目自己需要的 ROCm 编译宏和兼容参数。
    list(APPEND GPU_FLAGS
      "-DUSE_ROCM"
      "-DENABLE_FP8"
      "-U__HIP_NO_HALF_CONVERSIONS__"
      "-U__HIP_NO_HALF_OPERATORS__"
      "-Werror=unused-variable"
      "-fno-gpu-rdc")

  endif()
  # 把最终 flags 列表回传给调用方。
  set(${OUT_GPU_FLAGS} ${GPU_FLAGS} PARENT_SCOPE)
endfunction()

# ------------------------------- 准备 torch 自带 libgomp shim -------------------------------
# 在 build 目录下创建一层软链接目录，统一导出 libgomp.so 和 libgomp.so.1。
function(vllm_prepare_torch_gomp_shim TORCH_GOMP_SHIM_DIR)
  # 默认先把输出变量清空，表示暂未找到可用 shim。
  set(${TORCH_GOMP_SHIM_DIR} "" PARENT_SCOPE)

  # 通过 Python 探测 torch wheel 内部是否自带 libgomp。
  run_python(_VLLM_TORCH_GOMP_PATH
    "
import os, glob
import torch
# 先定位 torch 包目录。
torch_pkg = os.path.dirname(torch.__file__)
# 再回到 site-packages 根目录，兼容 torch.libs 布局。
site_root = os.path.dirname(torch_pkg)

# 同时扫描 torch.libs 和 torch/lib 两种常见布局。
roots = [os.path.join(site_root, 'torch.libs'), os.path.join(torch_pkg, 'lib')]
# 收集所有匹配到的 libgomp 候选路径。
candidates = []
for root in roots:
    # 目录不存在时直接跳过当前根目录。
    if not os.path.isdir(root):
        continue
    # 把该根目录下的 libgomp* 统统加入候选列表。
    candidates.extend(glob.glob(os.path.join(root, 'libgomp*.so*')))

# 找到候选时返回第一个，否则返回空串。
print(candidates[0] if candidates else '')
"
    "failed to probe for libgomp")

  if(_VLLM_TORCH_GOMP_PATH STREQUAL "" OR NOT EXISTS "${_VLLM_TORCH_GOMP_PATH}")
    # 没找到 vendored libgomp 时直接返回，保持输出为空。
    return()
  endif()

  # 把 shim 目录固定创建在当前 build 根目录下。
  set(_shim "${CMAKE_BINARY_DIR}/gomp_shim")
  # 预先创建 shim 目录。
  file(MAKE_DIRECTORY "${_shim}")

  # 先删掉旧的软链接，避免残留指向过期文件。
  execute_process(COMMAND ${CMAKE_COMMAND} -E rm -f "${_shim}/libgomp.so")
  # 同时清理 libgomp.so.1 这个兼容名字。
  execute_process(COMMAND ${CMAKE_COMMAND} -E rm -f "${_shim}/libgomp.so.1")
  # 创建标准名 libgomp.so 到真实 wheel 内部库文件的软链接。
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${_VLLM_TORCH_GOMP_PATH}" "${_shim}/libgomp.so")
  # 再创建 soname 兼容名 libgomp.so.1 的软链接。
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink "${_VLLM_TORCH_GOMP_PATH}" "${_shim}/libgomp.so.1")

  # 把最终 shim 目录路径回传给调用方。
  set(${TORCH_GOMP_SHIM_DIR} "${_shim}" PARENT_SCOPE)
endfunction()

# ------------------------------- 归一化 gencode 版本串 -------------------------------
# 把 80、90a 这类紧凑形式转换成 8.0、9.0a 这类 CMake 版本形式。
macro(string_to_ver OUT_VER IN_STR)
  # 在首位和末位数字之间插入小数点，保留可能存在的后缀字母。
  string(REGEX REPLACE "\([0-9]+\)\([0-9]\)" "\\1.\\2" ${OUT_VER} ${IN_STR})
endmacro()

# ------------------------------- 提取全局 CUDA gencode -------------------------------
# 先把 CMAKE_CUDA_FLAGS 里的 -gencode 取出来，后续再按源码文件单独回灌。
macro(clear_cuda_arches CUDA_ARCH_FLAGS)
    # 从全局 CUDA flags 里提取所有 -gencode 片段。
    string(REGEX MATCHALL "-gencode arch=[^ ]+" CUDA_ARCH_FLAGS
      ${CMAKE_CUDA_FLAGS})

    # 再把这些 -gencode 从全局 flags 里删掉，避免后续和文件级设置重复。
    string(REGEX REPLACE "-gencode arch=[^ ]+ *" "" CMAKE_CUDA_FLAGS
      ${CMAKE_CUDA_FLAGS})
endmacro()

# ------------------------------- 解析唯一 CUDA 架构列表 -------------------------------
# 把一组 gencode flags 提取成去重、升序的 CMake 架构列表。
function(extract_unique_cuda_archs_ascending OUT_ARCHES CUDA_ARCH_FLAGS)
  # 初始化输出用的临时架构列表。
  set(_CUDA_ARCHES)
  foreach(_ARCH ${CUDA_ARCH_FLAGS})
    # 从单条 gencode 中抽出 compute_XX 或 compute_XXa 片段。
    string(REGEX MATCH "arch=compute_\([0-9]+a?\)" _COMPUTE ${_ARCH})
    if (_COMPUTE)
      # 只保留架构编号本体，去掉前缀。
      set(_COMPUTE ${CMAKE_MATCH_1})
    endif()

    # 把紧凑编号转成 CMake 习惯的 x.y 版本形式。
    string_to_ver(_COMPUTE_VER ${_COMPUTE})
    # 把解析出的架构版本追加到临时列表。
    list(APPEND _CUDA_ARCHES ${_COMPUTE_VER})
  endforeach()

  # 删除重复架构，避免同一 sm 被重复编译。
  list(REMOVE_DUPLICATES _CUDA_ARCHES)
  # 按自然顺序升序排列，便于后续做区间和交集计算。
  list(SORT _CUDA_ARCHES COMPARE NATURAL ORDER ASCENDING)
  # 把最终架构列表回传给调用方。
  set(${OUT_ARCHES} ${_CUDA_ARCHES} PARENT_SCOPE)
endfunction()

# ------------------------------- 给源码追加单条 gencode -------------------------------
# 把一条 gencode 规则挂到指定源文件的 CUDA 编译选项上。
macro(set_gencode_flag_for_srcs)
  # 当前 helper 没有布尔开关参数。
  set(options)
  # ARCH/CODE 分别表示 compute 端和产物端编号。
  set(oneValueArgs ARCH CODE)
  # SRCS 是要追加该 gencode 的源码文件列表。
  set(multiValueArgs SRCS)
  # 解析调用方传入的参数。
  cmake_parse_arguments(arg "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN} )
  # 组装成单条 nvcc 可识别的 -gencode 选项。
  set(_FLAG -gencode arch=${arg_ARCH},code=${arg_CODE})
  # 只对 CUDA 语言的编译动作追加这条 gencode。
  set_property(
    SOURCE ${arg_SRCS}
    APPEND PROPERTY
    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:${_FLAG}>"
  )

  # 在 debug 日志里输出当前追加了哪条 gencode。
  message(DEBUG "Setting gencode flag for ${arg_SRCS}: ${_FLAG}")
endmacro(set_gencode_flag_for_srcs)

# ------------------------------- 给源码列表追加 gencode 集合 -------------------------------
# 把一组 CUDA 架构列表展开成源文件级 gencode 选项。
macro(set_gencode_flags_for_srcs)
  # 当前 helper 没有布尔开关参数。
  set(options)
  # BUILD_PTX_FOR_ARCH 用于额外请求一份特定 PTX 产物。
  set(oneValueArgs BUILD_PTX_FOR_ARCH)
  # SRCS 是目标源码列表，CUDA_ARCHS 是待展开的架构列表。
  set(multiValueArgs SRCS CUDA_ARCHS)
  # 解析调用方传入的参数。
  cmake_parse_arguments(arg "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN} )

  foreach(_ARCH ${arg_CUDA_ARCHS})
    # 先判断当前架构是否显式请求了 +PTX。
    string(FIND "${_ARCH}" "+PTX" _HAS_PTX)
    if(NOT _HAS_PTX EQUAL -1)
      # 去掉 +PTX 后缀，得到基础架构版本。
      string(REPLACE "+PTX" "" _BASE_ARCH "${_ARCH}")
      # 去掉小数点，转成 compute_XX / sm_XX 需要的紧凑格式。
      string(REPLACE "." "" _STRIPPED_ARCH "${_BASE_ARCH}")
      # 先追加面向具体硬件的 sm 产物。
      set_gencode_flag_for_srcs(
        SRCS ${arg_SRCS}
        ARCH "compute_${_STRIPPED_ARCH}"
        CODE "sm_${_STRIPPED_ARCH}")
      # 再追加一份 PTX 产物，供更高同代架构前向兼容。
      set_gencode_flag_for_srcs(
        SRCS ${arg_SRCS}
        ARCH "compute_${_STRIPPED_ARCH}"
        CODE "compute_${_STRIPPED_ARCH}")
    else()
      # 普通架构只需要生成 sm 产物。
      string(REPLACE "." "" _STRIPPED_ARCH "${_ARCH}")
      set_gencode_flag_for_srcs(
        SRCS ${arg_SRCS}
        ARCH "compute_${_STRIPPED_ARCH}"
        CODE "sm_${_STRIPPED_ARCH}")
    endif()
  endforeach()

  if (${arg_BUILD_PTX_FOR_ARCH})
    # 先把架构列表排序，便于找到最高架构版本。
    list(SORT arg_CUDA_ARCHS COMPARE NATURAL ORDER ASCENDING)
    # 取出当前列表中的最高架构。
    list(GET arg_CUDA_ARCHS -1 _HIGHEST_ARCH)
    if (_HIGHEST_ARCH VERSION_GREATER_EQUAL ${arg_BUILD_PTX_FOR_ARCH})
      # 把目标 PTX 架构转成紧凑格式。
      string(REPLACE "." "" _PTX_ARCH "${arg_BUILD_PTX_FOR_ARCH}")
      # 在满足条件时额外补一份指定基线架构的 PTX。
      set_gencode_flag_for_srcs(
        SRCS ${arg_SRCS}
        ARCH "compute_${_PTX_ARCH}"
        CODE "compute_${_PTX_ARCH}")
    endif()
  endif()
endmacro()

# ------------------------------- 计算宽松 CUDA 架构交集 -------------------------------
# 把源代码支持架构与目标环境架构做一次“尽量保留可兼容版本”的交集筛选。
function(cuda_archs_loose_intersection OUT_CUDA_ARCHS SRC_CUDA_ARCHS TGT_CUDA_ARCHS)
  # 先复制一份源架构列表，后面会原地修改。
  set(_SRC_CUDA_ARCHS "${SRC_CUDA_ARCHS}")
  # 再复制一份目标架构列表，避免直接污染调用方输入。
  set(_TGT_CUDA_ARCHS ${TGT_CUDA_ARCHS})

  # 单独记录哪些源架构请求了 +PTX。
  set(_PTX_ARCHS)
  foreach(_arch ${_SRC_CUDA_ARCHS})
    if(_arch MATCHES "\\+PTX$")
      # 去掉 +PTX 后缀后再参与普通架构匹配。
      string(REPLACE "+PTX" "" _base "${_arch}")
      # 把基础架构记进 PTX 请求列表。
      list(APPEND _PTX_ARCHS "${_base}")
      # 从原始源架构列表里删掉带后缀版本。
      list(REMOVE_ITEM _SRC_CUDA_ARCHS "${_arch}")
      # 再把无后缀版本补回去参与比较。
      list(APPEND _SRC_CUDA_ARCHS "${_base}")
    endif()
  endforeach()
  # 去重 PTX 请求列表。
  list(REMOVE_DUPLICATES _PTX_ARCHS)
  # 去重普通源架构列表。
  list(REMOVE_DUPLICATES _SRC_CUDA_ARCHS)

  # 先单独处理 a/f 后缀架构，保留它们比普通 x.0 更具体的语义。
  set(_CUDA_ARCHS)
  foreach(_arch ${_SRC_CUDA_ARCHS})
    if(_arch MATCHES "[af]$")
      # 先从普通待匹配列表里移除当前带后缀架构。
      list(REMOVE_ITEM _SRC_CUDA_ARCHS "${_arch}")
      # 取出不带 a/f 的基础版本号。
      string(REGEX REPLACE "[af]$" "" _base "${_arch}")
      if ("${_base}" IN_LIST TGT_CUDA_ARCHS)
        # 目标列表里若已有基础版本，就用更具体的 a/f 版本替换它。
        list(REMOVE_ITEM _TGT_CUDA_ARCHS "${_base}")
        # 把更具体的架构直接加入结果列表。
        list(APPEND _CUDA_ARCHS "${_arch}")
      endif()
    endif()
  endforeach()

  # 把剩余普通源架构按升序排列，便于后续找不大于目标的最近版本。
  list(SORT _SRC_CUDA_ARCHS COMPARE NATURAL ORDER ASCENDING)

  # 对每个目标架构，选择一个不超过它的最近可兼容源架构。
  foreach(_ARCH ${_TGT_CUDA_ARCHS})
    # 用临时变量保存当前目标最终命中的源架构。
    set(_TMP_ARCH)
    # 取出目标架构的 major 版本号。
    string(REGEX REPLACE "^([0-9]+)\\..*$" "\\1" TGT_ARCH_MAJOR "${_ARCH}")
    foreach(_SRC_ARCH ${_SRC_CUDA_ARCHS})
      # 取出候选源架构的 major 版本号。
      string(REGEX REPLACE "^([0-9]+)\\..*$" "\\1" SRC_ARCH_MAJOR "${_SRC_ARCH}")
      # 只接受版本不高于目标的源架构。
      if (_SRC_ARCH VERSION_LESS_EQUAL _ARCH)
        # PTX 请求允许跨 major 命中，否则要求 major 完全一致。
        if (_SRC_ARCH IN_LIST _PTX_ARCHS OR SRC_ARCH_MAJOR STREQUAL TGT_ARCH_MAJOR)
          # 由于列表升序遍历，后命中的会自然成为“最近且不大于目标”的版本。
          set(_TMP_ARCH "${_SRC_ARCH}")
        endif()
      else()
        # 一旦超过当前目标版本，后面的候选只会更大，可以提前结束。
        break()
      endif()
    endforeach()

    # 命中候选时把它加入结果列表。
    if (_TMP_ARCH)
      list(APPEND _CUDA_ARCHS "${_TMP_ARCH}")
    endif()
  endforeach()

  # 去重最终命中的架构列表。
  list(REMOVE_DUPLICATES _CUDA_ARCHS)

  # 重新给原本请求了 PTX 的架构补回 +PTX 后缀。
  set(_FINAL_ARCHS)
  foreach(_arch ${_CUDA_ARCHS})
    if(_arch IN_LIST _PTX_ARCHS)
      # PTX 请求过的架构恢复成 x.y+PTX 形式。
      list(APPEND _FINAL_ARCHS "${_arch}+PTX")
    else()
      # 其他普通架构原样保留。
      list(APPEND _FINAL_ARCHS "${_arch}")
    endif()
  endforeach()
  # 用补完后缀后的最终列表覆盖临时结果。
  set(_CUDA_ARCHS ${_FINAL_ARCHS})

  # 把宽松交集结果回传给调用方。
  set(${OUT_CUDA_ARCHS} ${_CUDA_ARCHS} PARENT_SCOPE)
endfunction()

# ------------------------------- 覆盖并筛选 GPU 架构 -------------------------------
# 目前主要用于 HIP 分支，把探测到的架构收敛到项目支持范围内。
macro(override_gpu_arches GPU_ARCHES GPU_LANG GPU_SUPPORTED_ARCHES)
  # 把首参和可变参数拼成完整的支持架构白名单。
  set(_GPU_SUPPORTED_ARCHES_LIST ${GPU_SUPPORTED_ARCHES} ${ARGN})
  # 打印当前后端支持的架构集合，便于诊断配置问题。
  message(STATUS "${GPU_LANG} supported arches: ${_GPU_SUPPORTED_ARCHES_LIST}")

  if (${GPU_LANG} STREQUAL "HIP")
    # 优先读取用户显式传入的 ROCm 架构列表。
    if(DEFINED ENV{PYTORCH_ROCM_ARCH})
      # 使用环境变量时，直接按用户给定值作为原始架构列表。
      set(HIP_ARCHITECTURES $ENV{PYTORCH_ROCM_ARCH})
    else()
      # 否则退回 CMake 在启用 HIP 语言时探测出的架构列表。
      set(HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
    endif()
    # 先清空输出变量，后面只保留支持且被探测到的架构。
    set(${GPU_ARCHES})
    foreach (_ARCH ${HIP_ARCHITECTURES})
      if (_ARCH IN_LIST _GPU_SUPPORTED_ARCHES_LIST)
        # 只有命中项目支持白名单的架构才写入最终结果。
        list(APPEND ${GPU_ARCHES} ${_ARCH})
      endif()
    endforeach()

    if(NOT ${GPU_ARCHES})
      # 过滤后一个可用架构都没有时直接终止配置。
      message(FATAL_ERROR
        "None of the detected ROCm architectures: ${HIP_ARCHITECTURES} is"
        " supported. Supported ROCm architectures are: ${_GPU_SUPPORTED_ARCHES_LIST}.")
    endif()
  endif()
endmacro()

# ------------------------------- 定义单个 Python 扩展目标 -------------------------------
# 用统一封装创建 CUDA/HIP/CXX 扩展，避免主 CMakeLists 重复写样板逻辑。
function (define_extension_target MOD_NAME)
  # 解析扩展目标的布尔开关、单值参数和列表参数。
  cmake_parse_arguments(PARSE_ARGV 1
    ARG
    "WITH_SOABI;NO_INSTALL"
    "DESTINATION;LANGUAGE;USE_SABI"
    "SOURCES;ARCHITECTURES;COMPILE_FLAGS;INCLUDE_DIRECTORIES;LIBRARIES")

  # HIP 构建时，先把 CUDA 源预处理成 HIP 源。
  if (ARG_LANGUAGE STREQUAL "HIP")
    hipify_sources_target(ARG_SOURCES ${MOD_NAME} "${ARG_SOURCES}")
  endif()

  if (ARG_WITH_SOABI)
    # 需要把 Python SOABI 拼进模块文件名时启用对应关键字。
    set(SOABI_KEYWORD WITH_SOABI)
  else()
    # 不启用 SOABI 时传空串，保持调用参数结构统一。
    set(SOABI_KEYWORD "")
  endif()

  # 运行 Python 检查当前解释器是否开启了 free-threaded 模式。
  run_python(IS_FREETHREADED_PYTHON
    "import sysconfig; print(1 if sysconfig.get_config_var(\"Py_GIL_DISABLED\") else 0)"
    "Failed to determine whether interpreter is free-threaded")

  # 只有非 free-threaded Python 才允许打开稳定 ABI。
  if (ARG_USE_SABI AND NOT IS_FREETHREADED_PYTHON)
    # 需要稳定 ABI 时按 USE_SABI 版本创建 Python 模块目标。
    Python_add_library(${MOD_NAME} MODULE USE_SABI ${ARG_USE_SABI} ${SOABI_KEYWORD} "${ARG_SOURCES}")
  else()
    # 否则按普通 Python 模块扩展创建目标。
    Python_add_library(${MOD_NAME} MODULE ${SOABI_KEYWORD} "${ARG_SOURCES}")
  endif()

  if (ARG_LANGUAGE STREQUAL "HIP")
    # HIP 扩展必须先依赖 hipify 生成步骤。
    add_dependencies(${MOD_NAME} hipify${MOD_NAME})
    # 先包含 build 目录下的 hipified 头，再追加调用方传入的额外头路径。
    target_include_directories(${MOD_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/csrc
      ${ARG_INCLUDE_DIRECTORIES})
  else()
    # 非 HIP 扩展统一包含源码目录、CUDA 头目录和调用方额外头路径。
    target_include_directories(${MOD_NAME} PRIVATE csrc
      ${CFIE_CUDA_INCLUDE_DIRS}
      ${ARG_INCLUDE_DIRECTORIES})
  endif()

  if (ARG_ARCHITECTURES)
    # 调用方传了架构列表时，把它写成 target 级架构属性。
    set_target_properties(${MOD_NAME} PROPERTIES
      ${ARG_LANGUAGE}_ARCHITECTURES "${ARG_ARCHITECTURES}")
  endif()

  # 只给指定语言的编译动作追加额外编译参数。
  target_compile_options(${MOD_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:${ARG_LANGUAGE}>:${ARG_COMPILE_FLAGS}>)

  # 把扩展目标名写进宏，供 torch binding 在运行时注册正确的 op namespace。
  target_compile_definitions(${MOD_NAME} PRIVATE
    "-DTORCH_EXTENSION_NAME=${MOD_NAME}")

  # CUDA 分支避免直接链接 TORCH_LIBRARIES，减少无关依赖被一起拉入。
  if (ARG_LANGUAGE STREQUAL "CUDA")
    if(WIN32 AND CFIE_CUDA_CUDART_LIBRARY AND CFIE_CUDA_DRIVER_LIBRARY)
      # Windows 上优先链接显式解析出的 cudart/driver/BLAS 路径。
      target_link_libraries(
        ${MOD_NAME} PRIVATE torch
        "${CFIE_CUDA_CUDART_LIBRARY}"
        "${CFIE_CUDA_DRIVER_LIBRARY}"
        ${CFIE_CUDA_BLAS_LIBRARIES}
        ${ARG_LIBRARIES})
    else()
      # 非 Windows 或未解析出绝对路径时，退回标准 CUDA target 名链接。
      target_link_libraries(${MOD_NAME} PRIVATE torch CUDA::cudart CUDA::cuda_driver ${ARG_LIBRARIES})
    endif()
  else()
    # 非 CUDA 分支沿用 torch 导出的标准依赖集合。
    target_link_libraries(${MOD_NAME} PRIVATE torch ${TORCH_LIBRARIES} ${ARG_LIBRARIES})
  endif()

  if (NOT ARG_NO_INSTALL)
    # 这里控制 install 阶段把 build tree 里的产物拷到哪，不影响前面的 OUTPUT_DIRECTORY。
    # 默认把扩展安装到调用方指定的 Python 包目录下。
    install(
      TARGETS ${MOD_NAME}
      LIBRARY DESTINATION ${ARG_DESTINATION}
      COMPONENT ${MOD_NAME})
  endif()
endfunction()
