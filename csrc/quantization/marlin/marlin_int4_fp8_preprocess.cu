#include "marlin.cuh"

#include "core/registration.h"

// ------------------------------- GPTQ / 非 zero-point 格式预处理 kernel -------------------------------
// 仅用于非 zero-point 格式，例如 GPTQ。
// 输入 qweight 是 int4 packed 权重，每个 int32 打包 8 个 4-bit 值。
__global__ void marlin_int4_fp8_preprocess_kernel_without_zp(
    // qweight: 一维 packed 权重，形状逻辑上为 (size_k * size_n // 8,)
    const int32_t* __restrict__ qweight,
    // output: 输出形状与 qweight 相同，可以与 qweight 原地复用
    int32_t* __restrict__ output) {
  // 每个线程处理一个 int32，即 8 个 int4 权重。
  // 一个block 32线程， 则每block处理32 * int32
  int32_t val = qweight[blockIdx.x * 32 + threadIdx.x];
  int32_t new_val = 0;

#pragma unroll
  for (int32_t i = 0; i < 8; i++) {
    // 取最低 4 bit，得到当前 int4 权重值。
    int32_t single_val = val & 0xF;

    // 将 GPTQ int4 表示转换为 Marlin FP8 路径需要的编码。
    // 这里相当于对 4-bit 值做半区翻转 / 符号重映射。
    single_val = single_val >= 8 ? single_val - 8 : 15 - single_val;

    // 写回 new_val 的第 i 个 4-bit 槽位。
    new_val |= single_val << (i * 4);

    // 右移 4 bit，处理下一个 int4。
    val >>= 4;
  }

  // 写回当前线程负责的 packed int32。
  output[blockIdx.x * 32 + threadIdx.x] = new_val;
}

// ------------------------------- AWQ / zero-point 格式预处理 kernel -------------------------------
// 仅用于 AWQ 格式：带 zero-point，且 qweight 使用 AWQ 权重布局。
__global__ void marlin_int4_fp8_preprocess_kernel_awq(
    // AWQ qweight: [size_k, size_n // 8]，每个 int32 打包 8 个 N 维 int4
    const int32_t* __restrict__ qweight,
    // output: 与 qweight 形状相同
    int32_t* __restrict__ output,
    // AWQ qzeros: [size_k // group_size, size_n // 8]
    const int32_t* __restrict__ qzeros,
    // size_n: 解包前的 N 维大小
    int32_t size_n,
    // size_k: K 维大小
    int32_t size_k,
    // group_size: 每组 zero-point 覆盖的 K 维行数
    int32_t group_size) {
  // 当前线程负责的 K 维行号。
  int32_t k_idx = blockIdx.x * 32 + threadIdx.x;

  // 当前线程负责的 packed N 维块号。
  int32_t n_pack_idx = blockIdx.y;

  // 读取一个 packed int32 权重，里面包含 8 个 int4。
  int32_t val = qweight[k_idx * size_n / 8 + n_pack_idx];

  // 根据 k_idx 所属 group 读取对应的 packed zero-point。
  int32_t zero = qzeros[k_idx / group_size * size_n / 8 + n_pack_idx];

  int32_t new_val = 0;

#pragma unroll
  for (int32_t i = 0; i < 8; i++) {
    // 取当前 4-bit 权重值和对应 4-bit zero-point。
    int32_t single_val = val & 0xF;
    int32_t single_zero = zero & 0xF;

    // 先应用 AWQ zero-point，再转换为 Marlin FP8 路径需要的编码。
    single_val =
        single_val >= single_zero ? single_val - single_zero : 15 - single_val;

    // 写入 new_val 的第 i 个 4-bit 槽位。
    new_val |= single_val << (i * 4);

    // 继续处理下一个 int4 权重和 zero-point。
    val >>= 4;
    zero >>= 4;
  }

  // 写回当前 K 行、当前 packed N 块。
  output[k_idx * size_n / 8 + n_pack_idx] = new_val;
}

// ------------------------------- Python/C++ 入口函数 -------------------------------
// 对 int4 packed qweight 做 FP8 激活路径所需的预处理。
// qzeros_or_none 为空时走 GPTQ/非 zero-point 路径；否则走 AWQ 路径。
torch::Tensor marlin_int4_fp8_preprocess(
    torch::Tensor& qweight,
    std::optional<torch::Tensor> qzeros_or_none,
    bool inplace) {
  // qweight 必须在 CUDA 上。
  TORCH_CHECK(qweight.device().is_cuda(), "qweight is not on GPU");

  // qweight 必须是 int32 packed 格式。
  TORCH_CHECK(qweight.scalar_type() == at::ScalarType::Int,
              "qweight.dtype != torch.int32");

  // 切换到 qweight 所在 CUDA device。
  const at::cuda::OptionalCUDAGuard device_guard(device_of(qweight));

  // inplace=true 时原地写回；否则新建同形状输出。
  torch::Tensor output = inplace ? qweight : torch::empty_like(qweight);

  if (!qzeros_or_none.has_value()) {
    // ------------------------------- 非 zero-point 路径：GPTQ 等 -------------------------------
    // 每个 block 有 32 个线程，每个线程处理 8 个 int4，
    // 因此每个 block 处理 256 个 int4。
    TORCH_CHECK(qweight.numel() * 8 % 256 == 0,
                "qweight.numel() * 8 % 256 != 0");

    int blocks = qweight.numel() * 8 / 256;

    // 启动非 zero-point 预处理 kernel。
    marlin_int4_fp8_preprocess_kernel_without_zp<<<blocks, 32>>>(
        (const int32_t*)qweight.data_ptr(), (int32_t*)output.data_ptr());

  } else {
    // ------------------------------- AWQ zero-point 路径 -------------------------------
    // AWQ qweight 物理形状为 [size_k, size_n // 8]。
    int32_t size_k = qweight.size(0);
    int32_t size_n = qweight.size(1) * 8;
    torch::Tensor qzeros = qzeros_or_none.value();

    // 每个 block 沿 K 维处理 32 行。
    TORCH_CHECK(size_k % 32 == 0, "size_k % 32 != 0");

    // qzeros 也必须在 CUDA 上，且为 int32 packed 格式。
    TORCH_CHECK(qzeros.device().is_cuda(), "qzeros is not on GPU");
    TORCH_CHECK(qzeros.scalar_type() == at::ScalarType::Int,
                "qweight.dtype != torch.int32");

    // qweight 和 qzeros 必须在同一张 GPU 上。
    TORCH_CHECK(device_of(qweight) == device_of(qzeros),
                "qzeros is not on the same device with qweight");

    // qzeros.shape[0] 对应 K 维 group 数，因此 group_size = size_k / num_groups。
    int32_t group_size = qweight.size(0) / qzeros.size(0);

    // qweight 和 qzeros 的 packed N 维必须一致。
    TORCH_CHECK(qweight.size(1) == qzeros.size(1),
                "qweight.size(1) != qzeros.size(1)");

    // K 维必须能被 zero-point group 数整除。
    TORCH_CHECK(qweight.size(0) % qzeros.size(0) == 0,
                "qweight.size(0) % qzeros.size(0) != 0");

    // AWQ 这里要求 group_size 是 8 的倍数。
    TORCH_CHECK(group_size % 8 == 0, "group_size % 8 != 0");

    // blockIdx.x 遍历 K 维，每个 block 32 行；
    // blockIdx.y 遍历 packed N 维。
    dim3 blocks(size_k / 32, size_n / 8);

    // 启动 AWQ zero-point 预处理 kernel。
    marlin_int4_fp8_preprocess_kernel_awq<<<blocks, 32>>>(
        (const int32_t*)qweight.data_ptr(), (int32_t*)output.data_ptr(),
        (const int32_t*)qzeros.data_ptr(), size_n, size_k, group_size);
  }

  return output;
}

// ------------------------------- PyTorch CUDA 算子注册 -------------------------------
// 将 C++/CUDA 函数注册为 torch extension 算子实现。
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("marlin_int4_fp8_preprocess", &marlin_int4_fp8_preprocess);
}
