#pragma once

// ------------------------------- 引入 FP8 基础类型 -------------------------------

// 同时引入 AMD 与 NVIDIA 两种 FP8 类型声明，避免量化公共头文件之间形成循环依赖。
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e4m3fn.h>

namespace vllm {

// ------------------------------- 通用向量化存储容器 -------------------------------

// `vec_n_t` 把 `vec_size` 个 `scalar_t` 元素打包成一个对齐后的向量容器。
template <typename scalar_t, size_t vec_size>
struct __align__(vec_size * sizeof(scalar_t)) vec_n_t {
  // 实际元素数组，逻辑形状为 `[vec_size]`，供向量化 load/store 一次性读写。
  scalar_t val[vec_size];
};

// ------------------------------- 8-bit 量化向量化存储容器 -------------------------------

// `q8_n_t` 专门用于 int8 / FP8 量化元素的向量化打包。
template <typename quant_type_t, size_t vec_size>
struct __align__(vec_size * sizeof(quant_type_t)) q8_n_t {
  // 约束量化元素类型，防止非 8-bit 量化类型误用该容器。
  static_assert(std::is_same_v<quant_type_t, int8_t> ||
                std::is_same_v<quant_type_t, c10::Float8_e4m3fn> ||
                std::is_same_v<quant_type_t, c10::Float8_e4m3fnuz>);

  // 实际量化元素数组，逻辑形状为 `[vec_size]`。
  quant_type_t val[vec_size];
};

// ------------------------------- 常用 4 元素向量别名 -------------------------------

// `vec4_t<T>` 表示 4 个普通标量元素组成的对齐向量容器。
template <typename scalar_t>
using vec4_t = vec_n_t<scalar_t, 4>;

// `q8x4_t<T>` 表示 4 个 int8 / FP8 量化元素组成的对齐向量容器。
template <typename quant_type_t>
using q8x4_t = q8_n_t<quant_type_t, 4>;

}  // namespace vllm
