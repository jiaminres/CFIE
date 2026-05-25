#include <torch/all.h>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "ops.h"

namespace {

inline int next_power_of_2_int(int value) {
  int out = 1;
  while (out < value) {
    out <<= 1;
  }
  return out;
}

__device__ float block_sum(float value, float* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  const float result = shared[0];
  __syncthreads();
  return result;
}

template <typename scalar_t, typename gate_t, typename beta_t>
__global__ void gated_delta_recurrent_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const gate_t* __restrict__ g,
    const beta_t* __restrict__ beta,
    const float* __restrict__ initial_state,
    const int64_t* __restrict__ cu_seqlens,
    scalar_t* __restrict__ output,
    float* __restrict__ final_state,
    int64_t B,
    int64_t T,
    int64_t Hq,
    int64_t H,
    int64_t K,
    int64_t V,
    int64_t N,
    float scale,
    bool has_cu_seqlens,
    bool output_final_state,
    bool normalize_qk) {
  extern __shared__ float shared[];

  const int64_t v_col = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t seq_idx = blockIdx.z;
  const int tid = threadIdx.x;
  const bool lane_valid = tid < K;

  const int64_t heads_per_group = H / Hq;
  const int64_t key_head = head / heads_per_group;

  int64_t batch_idx = seq_idx;
  int64_t start = 0;
  int64_t end = T;
  if (has_cu_seqlens) {
    batch_idx = 0;
    start = cu_seqlens[seq_idx];
    end = cu_seqlens[seq_idx + 1];
  }

  float state = 0.0f;
  if (lane_valid) {
    const int64_t state_idx = (((seq_idx * H + head) * V + v_col) * K + tid);
    state = initial_state[state_idx];
  }

  for (int64_t tok = start; tok < end; ++tok) {
    float q_val = 0.0f;
    float k_val = 0.0f;
    if (lane_valid) {
      const int64_t qk_idx = (((batch_idx * T + tok) * Hq + key_head) * K + tid);
      q_val = static_cast<float>(q[qk_idx]);
      k_val = static_cast<float>(k[qk_idx]);
    }

    if (normalize_qk) {
      const float q_norm_sq = block_sum(q_val * q_val, shared);
      const float k_norm_sq = block_sum(k_val * k_val, shared);
      const float q_inv_norm = rsqrtf(q_norm_sq + 1.0e-6f);
      const float k_inv_norm = rsqrtf(k_norm_sq + 1.0e-6f);
      q_val *= q_inv_norm;
      k_val *= k_inv_norm;
    }

    const float delta_dot = block_sum(lane_valid ? state * k_val : 0.0f, shared);
    const int64_t gate_idx = (batch_idx * T + tok) * H + head;
    const float beta_val = static_cast<float>(beta[gate_idx]);
    const float decay = expf(static_cast<float>(g[gate_idx]));
    const int64_t v_idx = (((batch_idx * T + tok) * H + head) * V + v_col);
    const float v_val = static_cast<float>(v[v_idx]);
    const float delta_v = (v_val - delta_dot) * beta_val;

    if (lane_valid) {
      state = state * decay + delta_v * k_val;
    }

    const float out_dot = block_sum(lane_valid ? state * q_val : 0.0f, shared);
    if (tid == 0) {
      output[v_idx] = static_cast<scalar_t>(out_dot * scale);
    }
  }

  if (output_final_state && lane_valid) {
    const int64_t state_idx = (((seq_idx * H + head) * V + v_col) * K + tid);
    final_state[state_idx] = state;
  }
}

bool is_supported_dtype(const torch::Tensor& t) {
  return t.scalar_type() == torch::kBFloat16 || t.scalar_type() == torch::kFloat16;
}

bool is_supported_gate_dtype(const torch::Tensor& t,
                             const torch::ScalarType q_dtype) {
  return t.scalar_type() == q_dtype || t.scalar_type() == torch::kFloat32;
}

template <typename scalar_t, typename gate_t, typename beta_t>
void launch_gated_delta_recurrent_cuda(
    const torch::Tensor& q_c, const torch::Tensor& k_c,
    const torch::Tensor& v_c, const torch::Tensor& g_c,
    const torch::Tensor& beta_c, const torch::Tensor& initial_c,
    const std::optional<torch::Tensor>& cu_c, torch::Tensor& output,
    torch::Tensor& final_state, int64_t B, int64_t T, int64_t Hq, int64_t H,
    int64_t K, int64_t V, int64_t N, float scale, bool output_final_state,
    bool use_qk_l2norm_in_kernel, dim3 grid, int threads, size_t shared_bytes,
    cudaStream_t stream) {
  gated_delta_recurrent_kernel<scalar_t, gate_t, beta_t>
      <<<grid, threads, shared_bytes, stream>>>(
          q_c.data_ptr<scalar_t>(), k_c.data_ptr<scalar_t>(),
          v_c.data_ptr<scalar_t>(), g_c.data_ptr<gate_t>(),
          beta_c.data_ptr<beta_t>(), initial_c.data_ptr<float>(),
          cu_c.has_value() ? cu_c.value().data_ptr<int64_t>() : nullptr,
          output.data_ptr<scalar_t>(), final_state.data_ptr<float>(), B, T, Hq,
          H, K, V, N, scale, cu_c.has_value(), output_final_state,
          use_qk_l2norm_in_kernel);
}

}  // namespace

bool chunk_gated_delta_rule_recurrent_cuda_fast_supported(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& g, const torch::Tensor& beta,
    const torch::Tensor& initial_state,
    const std::optional<torch::Tensor>& cu_seqlens) {
  if (!q.is_cuda() || !k.is_cuda() || !v.is_cuda() || !g.is_cuda() ||
      !beta.is_cuda() || !initial_state.is_cuda()) {
    return false;
  }
  if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4 || g.dim() != 3 ||
      beta.dim() != 3 || initial_state.dim() != 4) {
    return false;
  }
  if (!is_supported_dtype(q) || k.scalar_type() != q.scalar_type() ||
      v.scalar_type() != q.scalar_type() ||
      !is_supported_gate_dtype(g, q.scalar_type()) ||
      !is_supported_gate_dtype(beta, q.scalar_type())) {
    return false;
  }
  if (initial_state.scalar_type() != torch::kFloat32) {
    return false;
  }
  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t Hq = q.size(2);
  const int64_t K = q.size(3);
  const int64_t H = v.size(2);
  const int64_t V = v.size(3);
  if (T <= 0 || Hq <= 0 || H <= 0 || V <= 0 || K <= 0 || K > 256) {
    return false;
  }
  if (k.sizes() != q.sizes()) {
    return false;
  }
  if (v.size(0) != B || v.size(1) != T || g.size(0) != B || g.size(1) != T ||
      g.size(2) != H || beta.sizes() != g.sizes()) {
    return false;
  }
  if (H % Hq != 0) {
    return false;
  }
  const int64_t N = cu_seqlens.has_value() ? cu_seqlens.value().size(0) - 1 : B;
  if (N <= 0 || initial_state.size(0) != N || initial_state.size(1) != H ||
      initial_state.size(2) != V || initial_state.size(3) != K) {
    return false;
  }
  if (cu_seqlens.has_value()) {
    const auto& cu = cu_seqlens.value();
    if (!cu.is_cuda() || cu.dim() != 1 || cu.scalar_type() != torch::kInt64 ||
        B != 1) {
      return false;
    }
  }
  return true;
}

std::tuple<torch::Tensor, torch::Tensor>
chunk_gated_delta_rule_recurrent_cuda_fast(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& g, const torch::Tensor& beta, double scale,
    const torch::Tensor& initial_state, bool output_final_state,
    const std::optional<torch::Tensor>& cu_seqlens,
    bool use_qk_l2norm_in_kernel) {
  TORCH_CHECK(chunk_gated_delta_rule_recurrent_cuda_fast_supported(
                  q, k, v, g, beta, initial_state, cu_seqlens),
              "chunk_gated_delta_rule_recurrent_cuda_fast got unsupported inputs");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));

  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto v_c = v.contiguous();
  auto g_c = g.contiguous();
  auto beta_c = beta.contiguous();
  auto initial_c = initial_state.contiguous();
  std::optional<torch::Tensor> cu_c = std::nullopt;
  if (cu_seqlens.has_value()) {
    cu_c = cu_seqlens.value().contiguous();
  }

  const int64_t B = q_c.size(0);
  const int64_t T = q_c.size(1);
  const int64_t Hq = q_c.size(2);
  const int64_t K = q_c.size(3);
  const int64_t H = v_c.size(2);
  const int64_t V = v_c.size(3);
  const int64_t N = cu_c.has_value() ? cu_c.value().size(0) - 1 : B;

  auto output = torch::empty({B, T, H, V}, q_c.options());
  auto final_state =
      output_final_state
          ? torch::empty({N, H, V, K}, initial_c.options().dtype(torch::kFloat32))
          : torch::empty({0}, initial_c.options().dtype(torch::kFloat32));

  const int threads = next_power_of_2_int(static_cast<int>(K));
  const dim3 grid(static_cast<unsigned int>(V), static_cast<unsigned int>(H),
                  static_cast<unsigned int>(N));
  const size_t shared_bytes = static_cast<size_t>(threads) * sizeof(float);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q_c.scalar_type(),
      "gated_delta_recurrent_kernel", [&] {
        if (g_c.scalar_type() == torch::kFloat32) {
          if (beta_c.scalar_type() == torch::kFloat32) {
            launch_gated_delta_recurrent_cuda<scalar_t, float, float>(
                q_c, k_c, v_c, g_c, beta_c, initial_c, cu_c, output,
                final_state, B, T, Hq, H, K, V, N, static_cast<float>(scale),
                output_final_state, use_qk_l2norm_in_kernel, grid, threads,
                shared_bytes, stream);
          } else {
            launch_gated_delta_recurrent_cuda<scalar_t, float, scalar_t>(
                q_c, k_c, v_c, g_c, beta_c, initial_c, cu_c, output,
                final_state, B, T, Hq, H, K, V, N, static_cast<float>(scale),
                output_final_state, use_qk_l2norm_in_kernel, grid, threads,
                shared_bytes, stream);
          }
        } else {
          if (beta_c.scalar_type() == torch::kFloat32) {
            launch_gated_delta_recurrent_cuda<scalar_t, scalar_t, float>(
                q_c, k_c, v_c, g_c, beta_c, initial_c, cu_c, output,
                final_state, B, T, Hq, H, K, V, N, static_cast<float>(scale),
                output_final_state, use_qk_l2norm_in_kernel, grid, threads,
                shared_bytes, stream);
          } else {
            launch_gated_delta_recurrent_cuda<scalar_t, scalar_t, scalar_t>(
                q_c, k_c, v_c, g_c, beta_c, initial_c, cu_c, output,
                final_state, B, T, Hq, H, K, V, N, static_cast<float>(scale),
                output_final_state, use_qk_l2norm_in_kernel, grid, threads,
                shared_bytes, stream);
          }
        }
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, final_state};
}
