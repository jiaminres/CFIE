#include "core/registration.h"

#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int kNumBits = 4;
constexpr int kPackFactor = 32 / kNumBits;

template <typename scalar_t>
__global__ void gptq_marlin_fp8_dequantize_kernel(
    const uint32_t* __restrict__ qweight,
    const scalar_t* __restrict__ scales,
    scalar_t* __restrict__ output,
    int size_k,
    int size_n,
    int num_groups,
    int group_size,
    bool transpose) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (k >= size_k || n >= size_n) {
    return;
  }

  int qweight_row = k / kPackFactor;
  int qweight_shift = (k % kPackFactor) * kNumBits;
  uint32_t packed = qweight[qweight_row * size_n + n];
  int qvalue = static_cast<int>((packed >> qweight_shift) & 0xF) - 8;

  int group_index = 0;
  if (num_groups > 1) {
    group_index = group_size == -1 ? 0 : (k / group_size);
  }

  float scale = static_cast<float>(scales[group_index * size_n + n]);
  scalar_t value = static_cast<scalar_t>(static_cast<float>(qvalue) * scale);

  if (transpose) {
    output[n * size_k + k] = value;
  } else {
    output[k * size_n + n] = value;
  }
}

torch::Tensor gptq_marlin_fp8_dequantize(
    torch::Tensor qweight,
    torch::Tensor scales,
    int64_t size_k,
    int64_t size_n,
    int64_t group_size,
    bool transpose) {
  TORCH_CHECK(qweight.is_cuda(), "qweight must be a CUDA tensor");
  TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
  TORCH_CHECK(
      qweight.scalar_type() == at::kInt,
      "qweight must use torch.int32 storage");
  TORCH_CHECK(size_k > 0, "size_k must be positive");
  TORCH_CHECK(size_n > 0, "size_n must be positive");
  TORCH_CHECK(
      size_k % kPackFactor == 0,
      "size_k must be divisible by ",
      kPackFactor);
  TORCH_CHECK(
      qweight.dim() == 2 && qweight.size(0) == size_k / kPackFactor &&
          qweight.size(1) == size_n,
      "qweight must have shape [size_k / 8, size_n]");
  TORCH_CHECK(
      group_size == -1 || group_size > 0,
      "group_size must be -1 or a positive integer");
  TORCH_CHECK(
      group_size == -1 || size_k % group_size == 0,
      "size_k must be divisible by group_size");

  int64_t num_groups = group_size == -1 ? 1 : size_k / group_size;
  TORCH_CHECK(
      scales.dim() == 2 && scales.size(0) == num_groups &&
          scales.size(1) == size_n,
      "scales must have shape [num_groups, size_n]");

  c10::cuda::CUDAGuard device_guard(qweight.device());
  auto qweight_contiguous = qweight.contiguous();
  auto scales_contiguous = scales.contiguous();
  auto output = torch::empty(
      transpose ? std::vector<int64_t>{size_n, size_k}
                : std::vector<int64_t>{size_k, size_n},
      scales.options());

  constexpr int block_x = 16;
  constexpr int block_y = 16;
  dim3 block(block_x, block_y);
  dim3 grid(
      static_cast<unsigned int>((size_n + block_x - 1) / block_x),
      static_cast<unsigned int>((size_k + block_y - 1) / block_y));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      scales_contiguous.scalar_type(),
      "gptq_marlin_fp8_dequantize",
      [&] {
        gptq_marlin_fp8_dequantize_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getDefaultCUDAStream()>>>(
                reinterpret_cast<const uint32_t*>(
                    qweight_contiguous.data_ptr<int32_t>()),
                scales_contiguous.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<int>(size_k),
                static_cast<int>(size_n),
                static_cast<int>(num_groups),
                static_cast<int>(group_size),
                transpose);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("gptq_marlin_fp8_dequantize", &gptq_marlin_fp8_dequantize);
}
