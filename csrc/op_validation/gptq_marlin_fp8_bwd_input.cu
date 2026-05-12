#include "core/registration.h"

#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp8.h>

namespace {

// ------------------------------- MMA tile 形状 -------------------------------

// 当前验证反向使用 FP8 Tensor Core 指令：
//   mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
// 因此单个 warp 一次计算 `dInput` 的 16 行 M、8 列 K，并沿 N 方向每次归约 32 列。
constexpr int kMmaRows = 16;
constexpr int kMmaCols = 8;
constexpr int kMmaReduce = 32;

// GPTQ-Marlin FP8 forward repack 以 32x32 逻辑 tile 存储权重。
constexpr int kForwardTileK = 32;
constexpr int kForwardTileN = 32;
constexpr int kForwardWarpCountPerTile = 4;
constexpr int kForwardPackedIntsPerTile =
    (kForwardTileK * kForwardTileN) / 8;

// ------------------------------- packed int4 解码 -------------------------------

// 把 Marlin FP8 路径预处理后的 4-bit 编码恢复为 GPTQ 逻辑整数值 `[-8, 7]`。
__device__ inline int decode_preprocessed_u4b8(uint32_t encoded_value) {
  return encoded_value < 8 ? static_cast<int>(encoded_value)
                           : 7 - static_cast<int>(encoded_value);
}

// 从 forward Marlin 32x32 tile 布局中读取一个逻辑权重 `W[row_k, col_n]`。
__device__ inline uint32_t read_forward_marlin_u4_fp8(
    const uint32_t* qweight_fwd,
    int size_n,
    int row_k,
    int col_n) {
  // 定位逻辑 32x32 tile。
  int k_tile = row_k / kForwardTileK;
  int n_tile = col_n / kForwardTileN;

  // 定位 tile 内部坐标。
  int local_k = row_k % kForwardTileK;
  int local_n = col_n % kForwardTileN;

  // forward Marlin 权重张量的 N tile 数。
  int n_tiles = size_n / kForwardTileN;

  // 当前 32x32 tile 在 packed qweight 中的起始 int32 偏移。
  int tile_offset =
      (k_tile * n_tiles + n_tile) * kForwardPackedIntsPerTile;

  // FP8 激活路径下，一个 32x32 tile 被 4 个 warp 分摊，每个 warp 覆盖 8 个 N 列。
  int warp_id = local_n / 8;

  // `tc_col` 是当前 N 列在 warp 内 8 列范围中的列号。
  int tc_col = local_n % 8;

  // 每个线程保存 4 个 K 行的前半段和 4 个 K 行的后半段。
  int row_group = (local_k % 16) / 4;
  int pos_in_group = local_k % 4;

  // 反推 forward repack 中负责该元素的逻辑线程号。
  int th_id = tc_col * 4 + row_group;

  // 一个线程在该 tile 中给每个 warp 写出一个 packed int32。
  int packed_index =
      tile_offset + th_id * kForwardWarpCountPerTile + warp_id;

  // FP8 路径把 `[k, k + 16]` 交错进同一个 int32 的相邻 4-bit 槽。
  int nibble_slot = local_k < 16 ? pos_in_group * 2 : pos_in_group * 2 + 1;

  // 返回单个 4-bit 编码值。
  return (qweight_fwd[packed_index] >> (nibble_slot * 4)) & 0xF;
}

// ------------------------------- FP8 fragment 构造 -------------------------------

// 把 4 个 FP8 byte 按 MMA fragment 的寄存器格式打包进一个 uint32。
__device__ inline uint32_t pack_fp8x4(
    uint8_t x0,
    uint8_t x1,
    uint8_t x2,
    uint8_t x3) {
  return static_cast<uint32_t>(x0) | (static_cast<uint32_t>(x1) << 8) |
      (static_cast<uint32_t>(x2) << 16) |
      (static_cast<uint32_t>(x3) << 24);
}

// 把普通 float 转成 E4M3 FP8 的原始 bit 表示。
__device__ inline uint8_t float_to_e4m3_bits(float value) {
  __nv_fp8_e4m3 fp8_value(value);
  return fp8_value.__x;
}

// 读取上游梯度 `dY[m, n]` 的 FP8 bit；越界位置按 0 填充。
__device__ inline uint8_t read_grad_output_fp8_or_zero(
    const uint8_t* grad_output_fp8,
    int size_m,
    int size_n,
    int row_m,
    int col_n) {
  if (row_m >= size_m || col_n >= size_n) {
    return 0;
  }
  return grad_output_fp8[row_m * size_n + col_n];
}

// 构造 MMA operand A fragment，对应逻辑 tile `dY[M: M+16, N: N+32]`。
__device__ inline void build_operand_a_fragment(
    const uint8_t* grad_output_fp8,
    uint32_t (&frag_a)[4],
    int size_m,
    int size_n,
    int tile_m,
    int tile_n,
    int lane_id) {
  // PTX 文档中的 fragment 线程分组。
  int group_id = lane_id >> 2;
  int thread_in_group = lane_id & 3;

#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    uint8_t values[4];

#pragma unroll
    for (int item = 0; item < 4; ++item) {
      // 当前寄存器内的第几个 FP8 元素。
      int frag_index = reg * 4 + item;

      // A fragment 映射：
      //   a0..a3   -> row=group,   col=t*4+[0..3]
      //   a4..a7   -> row=group+8, col=t*4+[0..3]
      //   a8..a11  -> row=group,   col=t*4+16+[0..3]
      //   a12..a15 -> row=group+8, col=t*4+16+[0..3]
      int local_m = group_id + ((frag_index & 4) ? 8 : 0);
      int local_n = thread_in_group * 4 + item + ((frag_index & 8) ? 16 : 0);

      values[item] = read_grad_output_fp8_or_zero(
          grad_output_fp8,
          size_m,
          size_n,
          tile_m + local_m,
          tile_n + local_n);
    }

    // 一个 A 寄存器保存 4 个连续 FP8 byte。
    frag_a[reg] = pack_fp8x4(values[0], values[1], values[2], values[3]);
  }
}

// 构造 MMA operand B fragment，对应逻辑 tile `W^T[N: N+32, K: K+8]`。
template <typename scale_t>
__device__ inline void build_operand_b_fragment(
    const uint32_t* qweight_fwd,
    const scale_t* scales_bwd,
    uint32_t (&frag_b)[2],
    int size_k,
    int size_n,
    int group_size,
    int tile_k,
    int tile_n,
    int lane_id) {
  // PTX 文档中的 fragment 线程分组。
  int group_id = lane_id >> 2;
  int thread_in_group = lane_id & 3;

#pragma unroll
  for (int reg = 0; reg < 2; ++reg) {
    uint8_t values[4];

#pragma unroll
    for (int item = 0; item < 4; ++item) {
      // B fragment 映射：
      //   b0..b3 -> row=t*4+[0..3],    col=group
      //   b4..b7 -> row=t*4+16+[0..3], col=group
      int local_n = thread_in_group * 4 + item + (reg ? 16 : 0);
      int col_k = tile_k + group_id;
      int row_n = tile_n + local_n;

      uint8_t fp8_bits = 0;
      if (col_k < size_k && row_n < size_n) {
        // 当前输出 K 列所属的 GPTQ group。
        int group_index = group_size == -1 ? 0 : (col_k / group_size);

        // 从唯一一份 forward Marlin qweight 中读取逻辑 `W[col_k, row_n]`。
        uint32_t encoded_weight =
            read_forward_marlin_u4_fp8(qweight_fwd, size_n, col_k, row_n);

        // 恢复 int4 逻辑值，并乘以反向专用 scale。
        float scale =
            static_cast<float>(scales_bwd[group_index * size_n + row_n]);
        float weight_value =
            static_cast<float>(decode_preprocessed_u4b8(encoded_weight)) *
            scale;

        // 反向 MMA 路径把当前 tile 内的反向权重临时转成 FP8 fragment。
        fp8_bits = float_to_e4m3_bits(weight_value);
      }

      values[item] = fp8_bits;
    }

    // 一个 B 寄存器保存 4 个连续 FP8 byte。
    frag_b[reg] = pack_fp8x4(values[0], values[1], values[2], values[3]);
  }
}

// ------------------------------- FP8 MMA 指令封装 -------------------------------

// 执行一次 warp 级 FP8 MMA：
//   C[16x8] += A[16x32] @ B[32x8]
__device__ inline void mma_m16n8k32_fp8(
    const uint32_t (&frag_a)[4],
    const uint32_t (&frag_b)[2],
    float (&frag_c)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(frag_c[0]),
        "=f"(frag_c[1]),
        "=f"(frag_c[2]),
        "=f"(frag_c[3])
      : "r"(frag_a[0]),
        "r"(frag_a[1]),
        "r"(frag_a[2]),
        "r"(frag_a[3]),
        "r"(frag_b[0]),
        "r"(frag_b[1]),
        "f"(frag_c[0]),
        "f"(frag_c[1]),
        "f"(frag_c[2]),
        "f"(frag_c[3]));
#endif
}

// ------------------------------- 输出 fragment 写回 -------------------------------

template <typename output_t>
__device__ inline void store_accumulator_fragment(
    output_t* grad_input,
    const float (&frag_c)[4],
    const float* grad_output_scales,
    int grad_scale_count,
    int size_m,
    int size_k,
    int tile_m,
    int tile_k,
    int lane_id) {
  // PTX 文档中的 C fragment 映射。
  int group_id = lane_id >> 2;
  int thread_in_group = lane_id & 3;

#pragma unroll
  for (int item = 0; item < 4; ++item) {
    // C fragment 映射：
    //   c0 -> row=group,   col=t*2
    //   c1 -> row=group,   col=t*2+1
    //   c2 -> row=group+8, col=t*2
    //   c3 -> row=group+8, col=t*2+1
    int local_m = group_id + ((item & 2) ? 8 : 0);
    int local_k = thread_in_group * 2 + (item & 1);
    int row_m = tile_m + local_m;
    int col_k = tile_k + local_k;

    if (row_m < size_m && col_k < size_k) {
      // `scaled_fp8_quant` 生成的是 per-tensor 或 per-token scale。
      float grad_scale =
          grad_output_scales[grad_scale_count == 1 ? 0 : row_m];

      // MMA 累加的是 FP8 编码值本身，写回前恢复上游梯度 scale。
      float value = frag_c[item] * grad_scale;
      grad_input[row_m * size_k + col_k] = static_cast<output_t>(value);
    }
  }
}

// ------------------------------- 反向 dInput MMA kernel -------------------------------

template <typename output_t, typename scale_t>
__global__ void gptq_marlin_fp8_bwd_input_mma_kernel(
    const uint8_t* __restrict__ grad_output_fp8,
    const float* __restrict__ grad_output_scales,
    int grad_scale_count,
    const uint32_t* __restrict__ qweight_fwd,
    const scale_t* __restrict__ scales_bwd,
    output_t* __restrict__ grad_input,
    int size_m,
    int size_k,
    int size_n,
    int group_size) {
  // 一个 warp 负责一个 `16x8` 的 dInput tile。
  int lane_id = threadIdx.x & 31;
  int tile_k = blockIdx.x * kMmaCols;
  int tile_m = blockIdx.y * kMmaRows;

  // 每个线程持有 4 个 C fragment 元素。
  float frag_c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  // 沿原始输出维 N 做 `k=32` 的分块归约。
  for (int tile_n = 0; tile_n < size_n; tile_n += kMmaReduce) {
    uint32_t frag_a[4];
    uint32_t frag_b[2];

    // A = dY[M, N]，已经由上层按 token 动态量化为 FP8。
    build_operand_a_fragment(
        grad_output_fp8,
        frag_a,
        size_m,
        size_n,
        tile_m,
        tile_n,
        lane_id);

    // B = W^T[N, K]，从 forward Marlin qweight 中按 tile 临时恢复成 FP8 fragment。
    build_operand_b_fragment(
        qweight_fwd,
        scales_bwd,
        frag_b,
        size_k,
        size_n,
        group_size,
        tile_k,
        tile_n,
        lane_id);

    // Tensor Core 执行 `dY_fp8 @ W_tile_fp8^T`。
    mma_m16n8k32_fp8(frag_a, frag_b, frag_c);
  }

  // 把 C fragment 写回 `grad_input[M, K]`。
  store_accumulator_fragment(
      grad_input,
      frag_c,
      grad_output_scales,
      grad_scale_count,
      size_m,
      size_k,
      tile_m,
      tile_k,
      lane_id);
}

// ------------------------------- Python/C++ 入口 -------------------------------

torch::Tensor gptq_marlin_fp8_bwd_input(
    torch::Tensor grad_output_fp8,
    torch::Tensor grad_output_scales,
    torch::Tensor qweight_fwd,
    torch::Tensor scales_bwd,
    int64_t size_k,
    int64_t size_n,
    int64_t group_size) {
  TORCH_CHECK(
      grad_output_fp8.is_cuda(), "grad_output_fp8 must be a CUDA tensor");
  TORCH_CHECK(
      grad_output_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "grad_output_fp8 must use torch.float8_e4m3fn");
  TORCH_CHECK(grad_output_fp8.dim() == 2, "grad_output_fp8 must be 2D");
  TORCH_CHECK(
      grad_output_fp8.is_contiguous(), "grad_output_fp8 must be contiguous");

  TORCH_CHECK(
      grad_output_scales.is_cuda(),
      "grad_output_scales must be a CUDA tensor");
  TORCH_CHECK(
      grad_output_scales.scalar_type() == at::ScalarType::Float,
      "grad_output_scales must use torch.float32");
  TORCH_CHECK(
      grad_output_scales.is_contiguous(),
      "grad_output_scales must be contiguous");
  TORCH_CHECK(
      grad_output_scales.numel() == 1 ||
          grad_output_scales.numel() == grad_output_fp8.size(0),
      "grad_output_scales must be per-tensor or per-token");

  TORCH_CHECK(qweight_fwd.is_cuda(), "qweight_fwd must be a CUDA tensor");
  TORCH_CHECK(
      qweight_fwd.scalar_type() == at::kInt,
      "qweight_fwd must use torch.int32 storage");
  TORCH_CHECK(qweight_fwd.dim() == 2, "qweight_fwd must be 2D");
  TORCH_CHECK(qweight_fwd.is_contiguous(), "qweight_fwd must be contiguous");

  TORCH_CHECK(scales_bwd.is_cuda(), "scales_bwd must be a CUDA tensor");
  TORCH_CHECK(scales_bwd.dim() == 2, "scales_bwd must be 2D");
  TORCH_CHECK(scales_bwd.is_contiguous(), "scales_bwd must be contiguous");

  TORCH_CHECK(size_k > 0, "size_k must be positive");
  TORCH_CHECK(size_n > 0, "size_n must be positive");
  TORCH_CHECK(
      size_k % kForwardTileK == 0,
      "size_k must be divisible by 32 for FP8 backward validation");
  TORCH_CHECK(
      size_n % kMmaReduce == 0,
      "size_n must be divisible by 32 for FP8 backward validation");
  TORCH_CHECK(
      group_size == -1 || group_size > 0,
      "group_size must be -1 or a positive integer");
  TORCH_CHECK(
      group_size == -1 || size_k % group_size == 0,
      "size_k must be divisible by group_size");

  int64_t size_m = grad_output_fp8.size(0);
  TORCH_CHECK(
      grad_output_fp8.size(1) == size_n,
      "grad_output_fp8 second dimension must equal size_n");
  TORCH_CHECK(
      qweight_fwd.size(0) == size_k / 16 && qweight_fwd.size(1) == size_n * 2,
      "qweight_fwd must have forward Marlin shape [size_k / 16, size_n * 2]");

  int64_t num_groups = group_size == -1 ? 1 : size_k / group_size;
  TORCH_CHECK(
      scales_bwd.size(0) == num_groups && scales_bwd.size(1) == size_n,
      "scales_bwd must have row-major shape [num_groups, size_n]");

  c10::cuda::CUDAGuard device_guard(grad_output_fp8.device());
  auto* device_prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(
      device_prop->major > 8 ||
          (device_prop->major == 8 && device_prop->minor >= 9),
      "gptq_marlin_fp8_bwd_input MMA path requires SM89 or newer");

  auto grad_output_fp8_contiguous = grad_output_fp8.contiguous();
  auto grad_output_scales_contiguous = grad_output_scales.contiguous().view({-1});
  auto qweight_fwd_contiguous = qweight_fwd.contiguous();
  auto scales_bwd_contiguous = scales_bwd.contiguous();
  auto grad_input =
      torch::empty({size_m, size_k}, scales_bwd_contiguous.options());

  dim3 block(32);
  dim3 grid(
      static_cast<unsigned int>((size_k + kMmaCols - 1) / kMmaCols),
      static_cast<unsigned int>((size_m + kMmaRows - 1) / kMmaRows));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_bwd_contiguous.scalar_type(),
      "gptq_marlin_fp8_bwd_input_mma",
      [&] {
        gptq_marlin_fp8_bwd_input_mma_kernel<scalar_t, scalar_t>
            <<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
                reinterpret_cast<const uint8_t*>(
                    grad_output_fp8_contiguous.data_ptr()),
                grad_output_scales_contiguous.data_ptr<float>(),
                static_cast<int>(grad_output_scales_contiguous.numel()),
                reinterpret_cast<const uint32_t*>(
                    qweight_fwd_contiguous.data_ptr<int32_t>()),
                scales_bwd_contiguous.data_ptr<scalar_t>(),
                grad_input.data_ptr<scalar_t>(),
                static_cast<int>(size_m),
                static_cast<int>(size_k),
                static_cast<int>(size_n),
                static_cast<int>(group_size));
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_input;
}

}  // namespace

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("gptq_marlin_fp8_bwd_input", &gptq_marlin_fp8_bwd_input);
}
