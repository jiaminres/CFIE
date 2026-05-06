#include "core/registration.h"

#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass/numeric_types.h"

namespace {

constexpr int kTileKSizeFp8 = 32;
constexpr int kTileNSizeFp8 = 32;
constexpr int kWarpCountPerTile = 4;
constexpr int kPackedIntsPerTile = (kTileKSizeFp8 * kTileNSizeFp8) / 8;
constexpr float kFp8WeightScaleDivisor = 512.0f;

__device__ inline int scale_perm_inverse_index(int local_n) {
  constexpr int kScalePermInvLocal[32] = {
      0,  1,  8,  9,  16, 17, 24, 25, 2,  3,  10, 11, 18, 19, 26, 27,
      4,  5,  12, 13, 20, 21, 28, 29, 6,  7,  14, 15, 22, 23, 30, 31,
  };
  return kScalePermInvLocal[local_n];
}

__device__ inline int decode_preprocessed_u4b8(uint32_t encoded_value) {
  return encoded_value < 8 ? static_cast<int>(encoded_value)
                           : 7 - static_cast<int>(encoded_value);
}

__device__ inline uint32_t read_forward_marlin_u4_fp8(
    const uint32_t* qweight_fwd,
    int size_n,
    int row_k,
    int col_n) {
  int k_tile = row_k / kTileKSizeFp8;
  int n_tile = col_n / kTileNSizeFp8;
  int local_k = row_k % kTileKSizeFp8;
  int local_n = col_n % kTileNSizeFp8;
  int n_tiles = size_n / kTileNSizeFp8;
  int tile_offset = (k_tile * n_tiles + n_tile) * kPackedIntsPerTile;

  int warp_id = local_n / 8;
  int tc_col = local_n % 8;
  int row_group = (local_k % 16) / 4;
  int pos_in_group = local_k % 4;
  int th_id = tc_col * 4 + row_group;
  int packed_index = tile_offset + th_id * kWarpCountPerTile + warp_id;
  int nibble_slot = local_k < 16 ? pos_in_group * 2 : pos_in_group * 2 + 1;
  return (qweight_fwd[packed_index] >> (nibble_slot * 4)) & 0xF;
}

template <typename scale_t>
__device__ inline float read_forward_marlin_scale(
    const scale_t* scales_fwd,
    int size_n,
  int group_index,
  int col_n) {
  int tile_n = col_n / 32;
  int local_n = col_n % 32;
  int permuted_col = tile_n * 32 + scale_perm_inverse_index(local_n);
  return static_cast<float>(scales_fwd[group_index * size_n + permuted_col]) /
         kFp8WeightScaleDivisor;
}

template <typename output_t, typename scale_t>
__global__ void gptq_marlin_fp8_bwd_input_kernel(
    const uint8_t* __restrict__ grad_output_fp8,
    const float* __restrict__ grad_output_scales,
    int grad_scale_count,
    const uint32_t* __restrict__ qweight_fwd,
    const scale_t* __restrict__ scales_fwd,
    output_t* __restrict__ grad_input,
    int size_m,
    int size_k,
    int size_n,
    int group_size) {
  int row_m = blockIdx.y * blockDim.y + threadIdx.y;
  int col_k = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_m >= size_m || col_k >= size_k) {
    return;
  }

  int group_index = group_size == -1 ? 0 : (col_k / group_size);
  float grad_scale =
      grad_output_scales[grad_scale_count == 1 ? 0 : row_m];

  float acc = 0.0f;
  for (int n = 0; n < size_n; ++n) {
    uint8_t fp8_bits = grad_output_fp8[row_m * size_n + n];
    float grad_value =
        static_cast<float>(cutlass::float_e4m3_t::bitcast(fp8_bits)) *
        grad_scale;
    uint32_t encoded_weight =
        read_forward_marlin_u4_fp8(qweight_fwd, size_n, col_k, n);
    float weight_scale =
        read_forward_marlin_scale(scales_fwd, size_n, group_index, n);
    float weight_value =
        static_cast<float>(decode_preprocessed_u4b8(encoded_weight)) *
        weight_scale;
    acc += grad_value * weight_value;
  }

  grad_input[row_m * size_k + col_k] = static_cast<output_t>(acc);
}

torch::Tensor gptq_marlin_fp8_bwd_input(
    torch::Tensor grad_output_fp8,
    torch::Tensor grad_output_scales,
    torch::Tensor qweight_fwd,
    torch::Tensor scales_fwd,
    int64_t size_k,
    int64_t size_n,
    int64_t group_size) {
  TORCH_CHECK(grad_output_fp8.is_cuda(), "grad_output_fp8 must be a CUDA tensor");
  TORCH_CHECK(
      grad_output_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "grad_output_fp8 must use torch.float8_e4m3fn");
  TORCH_CHECK(grad_output_fp8.dim() == 2, "grad_output_fp8 must be 2D");
  TORCH_CHECK(grad_output_fp8.is_contiguous(), "grad_output_fp8 must be contiguous");

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

  TORCH_CHECK(scales_fwd.is_cuda(), "scales_fwd must be a CUDA tensor");
  TORCH_CHECK(scales_fwd.dim() == 2, "scales_fwd must be 2D");
  TORCH_CHECK(scales_fwd.is_contiguous(), "scales_fwd must be contiguous");

  TORCH_CHECK(size_k > 0, "size_k must be positive");
  TORCH_CHECK(size_n > 0, "size_n must be positive");
  TORCH_CHECK(
      size_k % kTileKSizeFp8 == 0,
      "size_k must be divisible by 32 for FP8 backward validation");
  TORCH_CHECK(
      size_n % kTileNSizeFp8 == 0,
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
      scales_fwd.size(0) == num_groups && scales_fwd.size(1) == size_n,
      "scales_fwd must have forward Marlin shape [num_groups, size_n]");

  c10::cuda::CUDAGuard device_guard(grad_output_fp8.device());
  auto grad_output_fp8_contiguous = grad_output_fp8.contiguous();
  auto grad_output_scales_contiguous = grad_output_scales.contiguous().view({-1});
  auto qweight_fwd_contiguous = qweight_fwd.contiguous();
  auto scales_fwd_contiguous = scales_fwd.contiguous();
  auto grad_input = torch::empty(
      {size_m, size_k},
      scales_fwd.options());

  constexpr int block_x = 16;
  constexpr int block_y = 16;
  dim3 block(block_x, block_y);
  dim3 grid(
      static_cast<unsigned int>((size_k + block_x - 1) / block_x),
      static_cast<unsigned int>((size_m + block_y - 1) / block_y));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_fwd_contiguous.scalar_type(),
      "gptq_marlin_fp8_bwd_input",
      [&] {
        gptq_marlin_fp8_bwd_input_kernel<scalar_t, scalar_t>
            <<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
                reinterpret_cast<const uint8_t*>(
                    grad_output_fp8_contiguous.data_ptr()),
                grad_output_scales_contiguous.data_ptr<float>(),
                static_cast<int>(grad_output_scales_contiguous.numel()),
                reinterpret_cast<const uint32_t*>(
                    qweight_fwd_contiguous.data_ptr<int32_t>()),
                scales_fwd_contiguous.data_ptr<scalar_t>(),
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
