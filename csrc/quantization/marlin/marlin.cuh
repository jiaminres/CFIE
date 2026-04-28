#pragma once  // 防止头文件被重复包含。

#ifndef _marlin_cuh  // 如果还没有定义头文件宏。
#define _marlin_cuh  // 定义头文件宏。

#include <torch/all.h>  // 引入 PyTorch C++ API。

#include <ATen/cuda/CUDAContext.h>  // CUDA stream/context 工具。
#include <c10/cuda/CUDAGuard.h>  // CUDA device guard 工具。
#include <cuda.h>  // CUDA driver API。
#include <cuda_fp16.h>  // CUDA half 类型。
#include <cuda_runtime.h>  // CUDA runtime API。
#include <iostream>  // 标准 IO。

#ifndef MARLIN_NAMESPACE_NAME  // 如果外部没有指定 namespace。
#define MARLIN_NAMESPACE_NAME marlin  // 默认 namespace 为 marlin。
#endif  // 结束 namespace 宏判断。

namespace MARLIN_NAMESPACE_NAME {  // 进入 Marlin namespace。

// ------------------------------- Marlin 基础参数 -------------------------------

    // 默认线程数：256 threads = 8 warps。
    static constexpr int default_threads = 256;

    // 主 kernel pipeline stage 数。
    static constexpr int pipe_stages = 4;

    // 每个线程块最小 N 维工作粒度。
    static constexpr int min_thread_n = 64;

    // 每个线程块最小 K 维工作粒度。
    static constexpr int min_thread_k = 64;

    // 每个线程块最大 N 维工作粒度。
    static constexpr int max_thread_n = 256;

    // Marlin 基础 tile 大小。
    static constexpr int tile_size = 16;

    // 最大并行分块数。
    static constexpr int max_par = 16;

// ------------------------------- Repack 参数 -------------------------------

    // repack kernel 的 pipeline stage 数。
    static constexpr int repack_stages = 8;

    // repack kernel 默认线程数。
    static constexpr int repack_threads = 256;

    // repack 的 K tile 大小。
    static constexpr int tile_k_size = tile_size;

    // repack 的 N tile 大小。
    static constexpr int tile_n_size = tile_k_size * 4;

// ------------------------------- 工具类型与函数 -------------------------------

    // 固定长度向量封装。
    template<typename T, int n>
    struct Vec {
        // 实际元素数组。
        T elems[n];

        // device 侧下标访问。
        __device__ T &operator[](int i) { return elems[i]; }
    };

    // 4 个 int 的向量别名。
    using I4 = Vec<int, 4>;

    // 整数向上取整除法。
    constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

// ------------------------------- cp.async 兼容路径：SM < 80 -------------------------------

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800  // Ampere 以前无 cp.async。

    // 拷贝 4 字节到 shared memory，带 predicate。
__device__ inline void cp_async1_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  // pred 为 true 才执行拷贝。
  if (pred) {
    // 用普通 global load + shared store 模拟。
    reinterpret_cast<int32_t*>(smem_ptr)[0] =
        reinterpret_cast<const int32_t*>(glob_ptr)[0];
  }
}

// 拷贝 8 字节到 shared memory，带 predicate。
__device__ inline void cp_async2_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  // pred 为 true 才执行拷贝。
  if (pred) {
    // 用普通 64-bit 拷贝模拟。
    reinterpret_cast<int64_t*>(smem_ptr)[0] =
        reinterpret_cast<const int64_t*>(glob_ptr)[0];
  }
}

// 拷贝 16 字节到 shared memory，带 predicate。
__device__ inline void cp_async4_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  // pred 为 true 才执行拷贝。
  if (pred) {
    // 用 int4 拷贝模拟 16 字节搬运。
    reinterpret_cast<int4*>(smem_ptr)[0] =
        reinterpret_cast<const int4*>(glob_ptr)[0];
  }
}

// 拷贝 16 字节到 shared memory，带 predicate。
__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr,
                                      bool pred = true) {
  // pred 为 true 才执行拷贝。
  if (pred) {
    // 旧架构下仍用普通 int4 拷贝。
    reinterpret_cast<int4*>(smem_ptr)[0] =
        reinterpret_cast<const int4*>(glob_ptr)[0];
  }
}

// 拷贝 16 字节到 shared memory。
__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  // 旧架构下用普通 int4 拷贝。
  reinterpret_cast<int4*>(smem_ptr)[0] =
      reinterpret_cast<const int4*>(glob_ptr)[0];
}

// 旧架构没有 async group，fence 为空操作。
__device__ inline void cp_async_fence() {}

// 旧架构没有 async wait，wait 为空操作。
template <int n>
__device__ inline void cp_async_wait() {}

// ------------------------------- cp.async 原生路径：SM >= 80 -------------------------------

#else  // Ampere 及之后支持 cp.async。

    // 拷贝 4 字节到 shared memory，cache all，带 predicate。
    __device__ inline void cp_async1_ca_pred(void *smem_ptr, const void *glob_ptr,
                                             bool pred = true) {
        // 单次拷贝字节数。
        const int BYTES = 4;

        // 将 generic shared 指针转成 shared 地址。
        uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

        // 发出带 predicate 的 cp.async.ca 指令。
        asm volatile(
                "{\n"
                "   .reg .pred p;\n"  // 定义 predicate 寄存器。
                "   setp.ne.b32 p, %0, 0;\n"  // pred != 0 时 p=true。
                "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"  // 条件异步拷贝。
                "}\n"::"r"((int) pred),  // 输入 pred。
        "r"(smem), "l"(glob_ptr), "n"(BYTES));  // 输入地址和字节数。
    }

    // 拷贝 8 字节到 shared memory，cache all，带 predicate。
    __device__ inline void cp_async2_ca_pred(void *smem_ptr, const void *glob_ptr,
                                             bool pred = true) {
        // 单次拷贝字节数。
        const int BYTES = 8;

        // 转换 shared memory 地址。
        uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

        // 发出带 predicate 的 cp.async.ca 指令。
        asm volatile(
                "{\n"
                "   .reg .pred p;\n"  // 定义 predicate。
                "   setp.ne.b32 p, %0, 0;\n"  // 根据 pred 设置 p。
                "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"  // 条件拷贝。
                "}\n"::"r"((int) pred),  // 输入 pred。
        "r"(smem), "l"(glob_ptr), "n"(BYTES));  // 输入地址和大小。
    }

    // 拷贝 16 字节到 shared memory，cache all，带 predicate。
    __device__ inline void cp_async4_ca_pred(void *smem_ptr, const void *glob_ptr,
                                             bool pred = true) {
        // 单次拷贝字节数。
        const int BYTES = 16;

        // 转换 shared memory 地址。
        uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

        // 发出带 predicate 的 cp.async.ca 指令。
        asm volatile(
                "{\n"
                "   .reg .pred p;\n"  // 定义 predicate。
                "   setp.ne.b32 p, %0, 0;\n"  // 根据 pred 设置 p。
                "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"  // 条件拷贝。
                "}\n"::"r"((int) pred),  // 输入 pred。
        "r"(smem), "l"(glob_ptr), "n"(BYTES));  // 输入地址和大小。
    }

    // 拷贝 16 字节到 shared memory，cache global，带 predicate。
    __device__ inline void cp_async4_pred(void *smem_ptr, const void *glob_ptr,
                                          bool pred = true) {
        // 单次拷贝字节数。
        const int BYTES = 16;

        // 转换 shared memory 地址。
        uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

        // 发出带 predicate 的 cp.async.cg 指令。
        asm volatile(
                "{\n"
                "   .reg .pred p;\n"  // 定义 predicate。
                "   setp.ne.b32 p, %0, 0;\n"  // 根据 pred 设置 p。
                "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"  // 条件拷贝。
                "}\n"::"r"((int) pred),  // 输入 pred。
        "r"(smem), "l"(glob_ptr), "n"(BYTES));  // 输入地址和大小。
    }

    // 拷贝 16 字节到 shared memory，不带 predicate。
    __device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
        // 单次拷贝字节数。
        const int BYTES = 16;

        // 转换 shared memory 地址。
        uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

        // 发出无 predicate 的 cp.async.cg 指令。
        asm volatile(
                "{\n"
                "   cp.async.cg.shared.global [%0], [%1], %2;\n"  // 异步拷贝 16 字节。
                "}\n"::"r"(smem),  // shared memory 地址。
        "l"(glob_ptr), "n"(BYTES));  // global 地址和大小。
    }

     // 提交当前 cp.async group。
    __device__ inline void cp_async_fence() {
        // commit 当前 async copy group。
        asm volatile("cp.async.commit_group;\n"::);
    }

    // 等待最多保留 n 个未完成 cp.async group。
    template<int n>
    __device__ inline void cp_async_wait() {
        // 等待 async copy group 完成。
        asm volatile("cp.async.wait_group %0;\n"::"n"(n));
    }

#endif  // 结束架构分支。

}  // namespace MARLIN_NAMESPACE_NAME

#endif  // 结束头文件宏。