#pragma once

// ------------------------------- 引入量化工具依赖 -------------------------------

// 提供 `std::numeric_limits` 与基础数学函数，供最大值和最小 scale 计算使用。
#include <cmath>

// 引入 PyTorch/C10 类型宏，例如 `C10_HOST_DEVICE`、`C10_DEVICE` 与 `C10_ALWAYS_INLINE`。
#include <torch/types.h>

// ------------------------------- 平台相关 FP8 类型与 host/device 修饰 -------------------------------

#ifndef USE_ROCM
  // CUDA 路径只需要 OCP E4M3 FP8 类型。
  #include <c10/util/Float8_e4m3fn.h>

  #if defined(_WIN32)
    // Windows 下某些静态 constexpr 变量不能稳定携带 `C10_HOST_DEVICE`，因此置空该兼容宏。
    #define MAYBE_HOST_DEVICE
  #else
    // 非 Windows CUDA 路径保留 host/device 修饰，使 constexpr 变量可在 host 与 device 侧使用。
    #define MAYBE_HOST_DEVICE C10_HOST_DEVICE
  #endif
#else
  // ROCm 路径需要 HIP 上下文头文件，供设备属性与 HIP 编译环境使用。
  #include <ATen/hip/HIPContext.h>

  // ROCm 路径同时可能使用 OCP E4M3 与 FNUZ E4M3 两种 FP8 类型。
  #include <c10/util/Float8_e4m3fn.h>
  #include <c10/util/Float8_e4m3fnuz.h>

  // ROCm 的静态 constexpr 路径不需要额外 `C10_HOST_DEVICE` 修饰。
  #define MAYBE_HOST_DEVICE
#endif

// ------------------------------- 量化类型最大值 traits -------------------------------

// `quant_type_max` 为 int8 / FP8 量化类型提供统一的最大可表示值入口。
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, c10::Float8_e4m3fn> ||
                                      std::is_same_v<T, c10::Float8_e4m3fnuz> ||
                                      std::is_same_v<T, int8_t>>>
struct quant_type_max {
  // 默认路径直接使用 C++ 类型系统提供的 `numeric_limits<T>::max()`。
  C10_HOST_DEVICE static constexpr T val() {
    // 返回当前量化类型的默认最大可表示值。
    return std::numeric_limits<T>::max();
  }
};

// ------------------------------- ROCm FNUZ FP8 最大值特化 -------------------------------

// FNUZ E4M3 使用 PyTorch 默认最大值 240.0/0x7F 时，动态量化精度会受影响。
// 因此该类型特化为 224.0/0x7E，降低动态量化时的溢出与精度风险。
template <>
struct quant_type_max<c10::Float8_e4m3fnuz> {
  // 返回 FNUZ E4M3 量化路径使用的调整后最大值。
  C10_HOST_DEVICE static constexpr c10::Float8_e4m3fnuz val() {
    // 直接按 bit 构造 0x7E，对应该路径期望的 224.0 最大幅度。
    return c10::Float8_e4m3fnuz(0x7E, c10::Float8_e4m3fnuz::from_bits());
  }
};

// ------------------------------- 最大值变量与函数入口 -------------------------------

// `quant_type_max_v<T>` 提供变量模板形式的量化最大值，便于 constexpr 场景使用。
template <typename T>
MAYBE_HOST_DEVICE static constexpr T quant_type_max_v =
    quant_type_max<T>::val();

// `quant_type_max_value<T>()` 提供函数形式的量化最大值，便于 device 代码中调用。
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE constexpr T quant_type_max_value() {
  // 统一转发到 `quant_type_max<T>` traits，避免不同调用点重复写特化逻辑。
  return quant_type_max<T>::val();
}

// ------------------------------- 最小 scale traits -------------------------------

// `min_scaling_factor` 为动态量化提供最小 scale 下界，避免 scale 过小或除零。
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, c10::Float8_e4m3fn> ||
                                      std::is_same_v<T, c10::Float8_e4m3fnuz> ||
                                      std::is_same_v<T, int8_t>>>
struct min_scaling_factor {
  // FP8 默认使用 `1 / (qmax * 512)` 作为最小 scale，下界与 FP8 动态范围绑定。
  C10_DEVICE C10_ALWAYS_INLINE static float val() {
    // 通过 qmax 推导 scale 下界，避免动态 absmax 为 0 时生成不可用 scale。
    return 1.0f / (quant_type_max_value<T>() * 512.0f);
  }
};

// ------------------------------- int8 最小 scale 特化 -------------------------------

// int8 动态量化使用 float epsilon 作为最小 scale，保持传统 int8 路径语义。
template <>
struct min_scaling_factor<int8_t> {
  // 返回 int8 路径的最小 scale 下界。
  C10_DEVICE C10_ALWAYS_INLINE static float val() {
    // 使用 float epsilon 防止除零，同时避免给 int8 路径引入 FP8 专用下界。
    return std::numeric_limits<float>::epsilon();
  }
};
