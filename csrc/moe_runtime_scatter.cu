#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include "ops.h"

#include <array>
#include <cstdint>
#include <optional>

namespace {

constexpr int kCopyThreads = 256;

struct RuntimeFieldCopySpec {
  const uint8_t* src;
  uint8_t* dst;
  int64_t src_stride_bytes;
  int64_t dst_stride_bytes;
  int64_t slot_bytes;
};

__device__ __forceinline__ RuntimeFieldCopySpec select_spec(
    int field_idx, RuntimeFieldCopySpec f0, RuntimeFieldCopySpec f1,
    RuntimeFieldCopySpec f2, RuntimeFieldCopySpec f3, RuntimeFieldCopySpec f4,
    RuntimeFieldCopySpec f5, RuntimeFieldCopySpec f6, RuntimeFieldCopySpec f7,
    RuntimeFieldCopySpec f8, RuntimeFieldCopySpec f9) {
  switch (field_idx) {
    case 0:
      return f0;
    case 1:
      return f1;
    case 2:
      return f2;
    case 3:
      return f3;
    case 4:
      return f4;
    case 5:
      return f5;
    case 6:
      return f6;
    case 7:
      return f7;
    case 8:
      return f8;
    default:
      return f9;
  }
}

__device__ __forceinline__ bool aligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

__global__ void runtime_scatter_install_kernel(
    const int64_t* __restrict__ slot_ids,
    const int64_t* __restrict__ old_expert_ids,
    const int64_t* __restrict__ new_expert_ids, int32_t* __restrict__ expert_map,
    int64_t num_experts, RuntimeFieldCopySpec f0, RuntimeFieldCopySpec f1,
    RuntimeFieldCopySpec f2, RuntimeFieldCopySpec f3, RuntimeFieldCopySpec f4,
    RuntimeFieldCopySpec f5, RuntimeFieldCopySpec f6, RuntimeFieldCopySpec f7,
    RuntimeFieldCopySpec f8, RuntimeFieldCopySpec f9) {
  const int64_t expert_idx = static_cast<int64_t>(blockIdx.x);
  const int field_idx = static_cast<int>(blockIdx.y);
  if (expert_idx >= num_experts) {
    return;
  }

  const RuntimeFieldCopySpec spec =
      select_spec(field_idx, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9);
  const int64_t slot = slot_ids[expert_idx];

  if (field_idx == 0 && threadIdx.x == 0) {
    const int64_t old_id = old_expert_ids[expert_idx];
    const int64_t new_id = new_expert_ids[expert_idx];
    if (old_id >= 0) {
      expert_map[old_id] = -1;
    }
    expert_map[new_id] = static_cast<int32_t>(slot);
  }

  if (spec.slot_bytes <= 0 || spec.src == nullptr || spec.dst == nullptr) {
    return;
  }

  const uint8_t* src = spec.src + expert_idx * spec.src_stride_bytes;
  uint8_t* dst = spec.dst + slot * spec.dst_stride_bytes;
  const bool use_vec4 =
      aligned16(src) && aligned16(dst) && (spec.slot_bytes % 16 == 0);

  if (use_vec4) {
    const int64_t vec_count = spec.slot_bytes / 16;
    const uint4* src_vec = reinterpret_cast<const uint4*>(src);
    uint4* dst_vec = reinterpret_cast<uint4*>(dst);
    for (int64_t i = threadIdx.x; i < vec_count; i += blockDim.x) {
      dst_vec[i] = src_vec[i];
    }
  } else {
    for (int64_t i = threadIdx.x; i < spec.slot_bytes; i += blockDim.x) {
      dst[i] = src[i];
    }
  }
}

bool has_slot_contiguous_payload(const torch::Tensor& tensor) {
  if (tensor.dim() < 1 || tensor.size(0) < 1) {
    return false;
  }
  int64_t expected = 1;
  for (int64_t dim = tensor.dim() - 1; dim >= 1; --dim) {
    if (tensor.stride(dim) != expected) {
      return false;
    }
    expected *= tensor.size(dim);
  }
  return true;
}

bool check_field_pair(const torch::Tensor& src, const torch::Tensor& dst,
                      int64_t batch) {
  if (!src.defined() || !dst.defined() || !src.is_cuda() || !dst.is_cuda()) {
    return false;
  }
  if (src.scalar_type() != dst.scalar_type() || src.dim() != dst.dim() ||
      src.size(0) != batch) {
    return false;
  }
  for (int64_t dim = 1; dim < src.dim(); ++dim) {
    if (src.size(dim) != dst.size(dim)) {
      return false;
    }
  }
  return has_slot_contiguous_payload(src) && has_slot_contiguous_payload(dst);
}

bool check_optional_pair(const std::optional<torch::Tensor>& src,
                         const std::optional<torch::Tensor>& dst,
                         int64_t batch) {
  if (!src.has_value() && !dst.has_value()) {
    return true;
  }
  if (!src.has_value() || !dst.has_value()) {
    return false;
  }
  return check_field_pair(src.value(), dst.value(), batch);
}

RuntimeFieldCopySpec empty_spec() {
  return RuntimeFieldCopySpec{nullptr, nullptr, 0, 0, 0};
}

RuntimeFieldCopySpec make_spec(const torch::Tensor& src,
                               const torch::Tensor& dst) {
  const int64_t elem_size = src.element_size();
  const int64_t slot_numel = src.numel() / src.size(0);
  return RuntimeFieldCopySpec{
      static_cast<const uint8_t*>(src.data_ptr()),
      static_cast<uint8_t*>(dst.data_ptr()),
      src.stride(0) * elem_size,
      dst.stride(0) * dst.element_size(),
      slot_numel * elem_size,
  };
}

RuntimeFieldCopySpec make_optional_spec(
    const std::optional<torch::Tensor>& src,
    const std::optional<torch::Tensor>& dst) {
  if (!src.has_value() || !dst.has_value()) {
    return empty_spec();
  }
  return make_spec(src.value(), dst.value());
}

bool base_inputs_supported(const torch::Tensor& slot_ids,
                           const torch::Tensor& old_expert_ids,
                           const torch::Tensor& new_expert_ids,
                           const torch::Tensor& expert_map) {
  const int64_t batch = slot_ids.numel();
  if (batch <= 0) {
    return false;
  }
  if (!slot_ids.is_cuda() || !old_expert_ids.is_cuda() ||
      !new_expert_ids.is_cuda() || !expert_map.is_cuda()) {
    return false;
  }
  if (slot_ids.scalar_type() != torch::kInt64 ||
      old_expert_ids.scalar_type() != torch::kInt64 ||
      new_expert_ids.scalar_type() != torch::kInt64 ||
      expert_map.scalar_type() != torch::kInt32) {
    return false;
  }
  return old_expert_ids.numel() == batch && new_expert_ids.numel() == batch;
}

bool launch_runtime_scatter_install(
    const torch::Tensor& slot_ids, const torch::Tensor& old_expert_ids,
    const torch::Tensor& new_expert_ids, torch::Tensor& expert_map,
    int field_count, RuntimeFieldCopySpec f0, RuntimeFieldCopySpec f1,
    RuntimeFieldCopySpec f2, RuntimeFieldCopySpec f3, RuntimeFieldCopySpec f4,
    RuntimeFieldCopySpec f5, RuntimeFieldCopySpec f6 = empty_spec(),
    RuntimeFieldCopySpec f7 = empty_spec(), RuntimeFieldCopySpec f8 = empty_spec(),
    RuntimeFieldCopySpec f9 = empty_spec()) {
  const int64_t batch = slot_ids.numel();
  if (batch <= 0 || field_count <= 0) {
    return false;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(expert_map));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned int>(batch),
                  static_cast<unsigned int>(field_count));
  runtime_scatter_install_kernel<<<grid, kCopyThreads, 0, stream>>>(
      slot_ids.data_ptr<int64_t>(), old_expert_ids.data_ptr<int64_t>(),
      new_expert_ids.data_ptr<int64_t>(), expert_map.data_ptr<int32_t>(), batch,
      f0, f1, f2, f3, f4, f5, f6, f7, f8, f9);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

}  // namespace

bool moe_batch_load_unquantized_runtime_and_install_fused_cuda(
    const torch::Tensor& slot_ids, const torch::Tensor& old_expert_ids,
    const torch::Tensor& new_expert_ids, const torch::Tensor& w13_src,
    const torch::Tensor& w2_src, torch::Tensor& w13_dst, torch::Tensor& w2_dst,
    torch::Tensor& expert_map) {
  const int64_t batch = slot_ids.numel();
  if (!base_inputs_supported(slot_ids, old_expert_ids, new_expert_ids,
                             expert_map)) {
    return false;
  }
  if (!check_field_pair(w13_src, w13_dst, batch) ||
      !check_field_pair(w2_src, w2_dst, batch)) {
    return false;
  }
  return launch_runtime_scatter_install(
      slot_ids, old_expert_ids, new_expert_ids, expert_map, 2,
      make_spec(w13_src, w13_dst), make_spec(w2_src, w2_dst), empty_spec(),
      empty_spec(), empty_spec(), empty_spec());
}

bool moe_batch_load_gptq_runtime_and_install_fused_cuda(
    const torch::Tensor& slot_ids, const torch::Tensor& old_expert_ids,
    const torch::Tensor& new_expert_ids, const torch::Tensor& w13_qweight_src,
    const torch::Tensor& w2_qweight_src, const torch::Tensor& w13_scales_src,
    const torch::Tensor& w2_scales_src, const torch::Tensor& w13_qzeros_src,
    const torch::Tensor& w2_qzeros_src, torch::Tensor& w13_qweight_dst,
    torch::Tensor& w2_qweight_dst, torch::Tensor& w13_scales_dst,
    torch::Tensor& w2_scales_dst, torch::Tensor& w13_qzeros_dst,
    torch::Tensor& w2_qzeros_dst,
    const std::optional<torch::Tensor>& w13_g_idx_src,
    const std::optional<torch::Tensor>& w2_g_idx_src,
    const std::optional<torch::Tensor>& w13_g_idx_sort_indices_src,
    const std::optional<torch::Tensor>& w2_g_idx_sort_indices_src,
    const std::optional<torch::Tensor>& w13_g_idx_dst,
    const std::optional<torch::Tensor>& w2_g_idx_dst,
    const std::optional<torch::Tensor>& w13_g_idx_sort_indices_dst,
    const std::optional<torch::Tensor>& w2_g_idx_sort_indices_dst,
    torch::Tensor& expert_map) {
  const int64_t batch = slot_ids.numel();
  if (!base_inputs_supported(slot_ids, old_expert_ids, new_expert_ids,
                             expert_map)) {
    return false;
  }
  if (!check_field_pair(w13_qweight_src, w13_qweight_dst, batch) ||
      !check_field_pair(w2_qweight_src, w2_qweight_dst, batch) ||
      !check_field_pair(w13_scales_src, w13_scales_dst, batch) ||
      !check_field_pair(w2_scales_src, w2_scales_dst, batch) ||
      !check_field_pair(w13_qzeros_src, w13_qzeros_dst, batch) ||
      !check_field_pair(w2_qzeros_src, w2_qzeros_dst, batch) ||
      !check_optional_pair(w13_g_idx_src, w13_g_idx_dst, batch) ||
      !check_optional_pair(w2_g_idx_src, w2_g_idx_dst, batch) ||
      !check_optional_pair(w13_g_idx_sort_indices_src,
                           w13_g_idx_sort_indices_dst, batch) ||
      !check_optional_pair(w2_g_idx_sort_indices_src,
                           w2_g_idx_sort_indices_dst, batch)) {
    return false;
  }

  const int field_count = w13_g_idx_src.has_value() ? 10 : 6;
  return launch_runtime_scatter_install(
      slot_ids, old_expert_ids, new_expert_ids, expert_map, field_count,
      make_spec(w13_qweight_src, w13_qweight_dst),
      make_spec(w2_qweight_src, w2_qweight_dst),
      make_spec(w13_scales_src, w13_scales_dst),
      make_spec(w2_scales_src, w2_scales_dst),
      make_spec(w13_qzeros_src, w13_qzeros_dst),
      make_spec(w2_qzeros_src, w2_qzeros_dst),
      make_optional_spec(w13_g_idx_src, w13_g_idx_dst),
      make_optional_spec(w2_g_idx_src, w2_g_idx_dst),
      make_optional_spec(w13_g_idx_sort_indices_src,
                         w13_g_idx_sort_indices_dst),
      make_optional_spec(w2_g_idx_sort_indices_src,
                         w2_g_idx_sort_indices_dst));
}
