#pragma once

#include "quantization/vectorization.cuh"
#include "quantization/utils.cuh"

#include <cmath>

#ifndef USE_ROCM
  #include "nvidia/quant_utils.cuh"
#else
  #include "amd/quant_utils.cuh"
#endif

// ------------------------------- 判断当前平台 FP8 语义 -------------------------------

static bool is_fp8_ocp() {
#ifndef USE_ROCM
  // CUDA 路径默认使用 OCP FP8 语义，因此直接返回 true。
  return true;
#else
  // ROCm 路径需要读取当前设备属性，区分 MI300 等特殊 FP8 表示。
  auto dprops = at::cuda::getCurrentDeviceProperties();

  // ROCm 设备架构名中包含 gfx94 时，对应 MI300 系列的 fnuz FP8 口径。
  std::string device_arch = dprops->gcnArchName;

  // 查找 gfx94 子串，用于判断是否应避开 OCP FP8 语义。
  size_t substring = device_arch.find("gfx94");

  // 未命中 gfx94 时返回 true，表示使用 OCP FP8；命中则返回 false。
  return substring == std::string::npos;
#endif
}

namespace vllm {

// ------------------------------- float 原子最大值 helper -------------------------------

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  // 保存 atomic 操作前的旧值，保持和 CUDA atomic API 的返回语义一致。
  float old;

  // 对非负 float 可直接复用 int 原子最大值；负数路径需要用 unsigned + atomicMin 保持排序语义。
  old = (value >= 0)
            ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int*)addr, __float_as_uint(value)));

  // 返回原子更新前的旧值，调用方当前不依赖该返回值但保留通用 helper 语义。
  return old;
}

// ------------------------------- 标量到 FP8 的缩放转换 helper -------------------------------

template <bool is_scale_inverted, typename fp8_type>
__device__ __forceinline__ fp8_type scaled_fp8_conversion(float const val,
                                                          float const scale) {
  // 初始化缩放后的中间值，后续会被裁剪到 FP8 可表示范围。
  float x = 0.0f;

  // `is_scale_inverted=true` 表示调用方已经传入 `1 / scale`，因此走乘法路径。
  if constexpr (is_scale_inverted) {
    // 用倒数 scale 缩放输入值，减少 kernel 内除法开销。
    x = val * scale;
  } else {
    // `is_scale_inverted=false` 表示调用方传入原始 scale，需要执行 `val / scale`。
    x = val / scale;
  }

  // 读取目标 FP8 类型的最大可表示幅度，用作饱和裁剪边界。
  const float qmax = static_cast<float>(quant_type_max_value<fp8_type>());

  // 将缩放结果裁剪到 `[-qmax, qmax]`，避免转换成 FP8 时溢出。
  float r = fmaxf(-qmax, fminf(x, qmax));
#ifndef USE_ROCM
  // NVIDIA 路径使用硬件 FP8 转换指令，当前主要面向 `c10::Float8_e4m3fn`。
  return fp8::vec_conversion<fp8_type, float>(r);
#else
  // ROCm 路径使用 ROCm 对应的 FP8 转换 helper。
  return fp8::cvt_c10<fp8_type>(r);
#endif
}

}  // namespace vllm
