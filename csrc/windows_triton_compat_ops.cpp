#include "ops.h"

#include <torch/extension.h>

#include <vector>

namespace {

torch::Tensor apply_gate_activation(const torch::Tensor& x,
                                    const std::string& activation) {
  if (activation == "swish" || activation == "silu") {
    return x * torch::sigmoid(x);
  }
  if (activation == "sigmoid") {
    return torch::sigmoid(x);
  }
  TORCH_CHECK(false, "Unsupported gating activation: ", activation);
}

torch::Tensor softplus_with_threshold(const torch::Tensor& x, double beta,
                                      double threshold) {
  auto beta_x = x * beta;
  return torch::where(beta_x <= threshold, torch::log1p(torch::exp(beta_x)) / beta,
                      x);
}

torch::Tensor prepare_mrope_cache(const torch::Tensor& cache,
                                  c10::List<int64_t> mrope_section,
                                  bool mrope_interleaved) {
  TORCH_CHECK(cache.dim() == 3 && cache.size(0) == 3,
              "MRoPE cache must have shape [3, num_tokens, rotary_dim/2]");
  TORCH_CHECK(mrope_section.size() == 3,
              "mrope_section must contain [t, h, w]");

  const int64_t section_t = mrope_section.get(0);
  const int64_t section_h = mrope_section.get(1);
  const int64_t section_w = mrope_section.get(2);
  const int64_t total = cache.size(-1);
  TORCH_CHECK(section_t + section_h + section_w == total,
              "mrope_section must sum to rotary_dim / 2");

  if (mrope_interleaved) {
    auto merged = cache.select(0, 0).clone();
    if (section_h > 0) {
      merged.slice(-1, 1, section_h * 3, 3)
          .copy_(cache.select(0, 1).slice(-1, 1, section_h * 3, 3));
    }
    if (section_w > 0) {
      merged.slice(-1, 2, section_w * 3, 3)
          .copy_(cache.select(0, 2).slice(-1, 2, section_w * 3, 3));
    }
    return merged;
  }

  auto splits = cache.split_with_sizes({section_t, section_h, section_w}, -1);
  return torch::cat(
      {splits[0].select(0, 0), splits[1].select(0, 1), splits[2].select(0, 2)},
      -1);
}

torch::Tensor apply_rotary_emb_native(const torch::Tensor& x,
                                      const torch::Tensor& cos,
                                      const torch::Tensor& sin,
                                      bool is_neox) {
  auto cos_expanded = cos.unsqueeze(1).to(x.scalar_type());
  auto sin_expanded = sin.unsqueeze(1).to(x.scalar_type());

  if (is_neox) {
    auto x1 = x.slice(-1, 0, x.size(-1) / 2);
    auto x2 = x.slice(-1, x.size(-1) / 2, x.size(-1));
    auto o1 = x1 * cos_expanded - x2 * sin_expanded;
    auto o2 = x2 * cos_expanded + x1 * sin_expanded;
    return torch::cat({o1, o2}, -1);
  }

  auto reshaped = x.reshape({x.size(0), x.size(1), x.size(2) / 2, 2});
  auto x1 = reshaped.select(-1, 0);
  auto x2 = reshaped.select(-1, 1);
  auto o1 = x1 * cos_expanded - x2 * sin_expanded;
  auto o2 = x2 * cos_expanded + x1 * sin_expanded;
  return torch::stack({o1, o2}, -1).reshape_as(x);
}

int64_t resolve_state_index(const std::optional<torch::Tensor>& indices,
                            int64_t seq_idx, int64_t token_offset) {
  if (!indices.has_value()) {
    return -2;
  }

  const auto& idx = indices.value();
  if (idx.dim() == 1) {
    const int64_t flat_index = seq_idx + token_offset;
    if (flat_index >= idx.numel()) {
      return -1;
    }
    return idx[flat_index].item<int64_t>();
  }
  return idx.index({seq_idx, token_offset}).item<int64_t>();
}

torch::Tensor l2norm_last_dim(const torch::Tensor& x) {
  return x / torch::sqrt((x * x).sum(-1, true) + 1e-6);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> mrope_rotary_embedding(
    const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& cos, const torch::Tensor& sin, int64_t head_size,
    int64_t rotary_dim, c10::List<int64_t> mrope_section, bool is_neox,
    bool mrope_interleaved) {
  TORCH_CHECK(query.is_cuda(), "mrope_rotary_embedding expects CUDA query");
  TORCH_CHECK(key.is_cuda(), "mrope_rotary_embedding expects CUDA key");
  TORCH_CHECK(query.dim() == 2 && key.dim() == 2,
              "mrope_rotary_embedding expects flattened [num_tokens, hidden]");

  const int64_t num_tokens = query.size(0);
  auto merged_cos = prepare_mrope_cache(cos, mrope_section, mrope_interleaved);
  auto merged_sin = prepare_mrope_cache(sin, mrope_section, mrope_interleaved);

  auto query_view = query.view({num_tokens, -1, head_size});
  auto query_rot = query_view.slice(-1, 0, rotary_dim);
  auto query_pass = query_view.slice(-1, rotary_dim, head_size);
  auto query_out = torch::cat(
      {apply_rotary_emb_native(query_rot, merged_cos, merged_sin, is_neox),
       query_pass},
      -1)
                       .reshape_as(query);

  auto key_view = key.view({num_tokens, -1, head_size});
  auto key_rot = key_view.slice(-1, 0, rotary_dim);
  auto key_pass = key_view.slice(-1, rotary_dim, head_size);
  auto key_out = torch::cat(
      {apply_rotary_emb_native(key_rot, merged_cos, merged_sin, is_neox),
       key_pass},
      -1)
                     .reshape_as(key);

  return {query_out, key_out};
}

torch::Tensor gated_layer_norm(
    const torch::Tensor& input, const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& gate, double epsilon,
    int64_t group_size, bool norm_before_gate, bool is_rms_norm,
    const std::string& activation) {
  TORCH_CHECK(input.is_cuda(), "gated_layer_norm expects CUDA input");
  TORCH_CHECK(weight.is_cuda(), "gated_layer_norm expects CUDA weight");
  TORCH_CHECK(input.size(-1) == weight.size(0),
              "weight must match the hidden dimension");

  auto original_shape = input.sizes().vec();
  const auto original_dtype = input.scalar_type();
  const int64_t hidden = input.size(-1);
  const int64_t resolved_group_size =
      group_size > 0 ? group_size : hidden;
  TORCH_CHECK(hidden % resolved_group_size == 0,
              "group_size must divide hidden size");
  const int64_t num_groups = hidden / resolved_group_size;

  auto x = input.reshape({-1, hidden}).to(torch::kFloat32);
  auto w = weight.to(torch::kFloat32);
  auto z = gate.has_value() ? gate.value().reshape({-1, hidden}).to(torch::kFloat32)
                            : torch::Tensor();

  if (gate.has_value() && !norm_before_gate) {
    x = x * apply_gate_activation(z, activation);
  }

  torch::Tensor y;
  if (num_groups == 1) {
    if (is_rms_norm) {
      auto rstd = torch::rsqrt(x.square().mean(-1, true) + epsilon);
      y = x * rstd * w;
    } else {
      auto mean = x.mean(-1, true);
      auto centered = x - mean;
      auto var = centered.square().mean(-1, true);
      y = centered * torch::rsqrt(var + epsilon) * w;
    }
  } else {
    auto x_group = x.reshape({-1, num_groups, resolved_group_size});
    auto w_group = w.reshape({num_groups, resolved_group_size});
    if (is_rms_norm) {
      auto rstd = torch::rsqrt(x_group.square().mean(-1, true) + epsilon);
      y = (x_group * rstd * w_group).reshape({-1, hidden});
    } else {
      auto mean = x_group.mean(-1, true);
      auto centered = x_group - mean;
      auto var = centered.square().mean(-1, true);
      y = (centered * torch::rsqrt(var + epsilon) * w_group)
              .reshape({-1, hidden});
    }
  }

  if (bias.has_value()) {
    y = y + bias.value().to(torch::kFloat32);
  }
  if (gate.has_value() && norm_before_gate) {
    y = y * apply_gate_activation(z, activation);
  }

  return y.reshape(original_shape).to(original_dtype);
}

std::tuple<torch::Tensor, torch::Tensor>
fused_sigmoid_gating_delta_rule_update_precompiled(
    const torch::Tensor& A_log, const torch::Tensor& a,
    const torch::Tensor& b, const torch::Tensor& dt_bias,
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    double beta, double threshold, double scale,
    const torch::Tensor& initial_state, bool inplace_final_state,
    const std::optional<torch::Tensor>& cu_seqlens,
    const std::optional<torch::Tensor>& ssm_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    bool use_qk_l2norm_in_kernel, bool is_kda) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "precompiled fused sigmoid gating expects CUDA tensors");
  TORCH_CHECK(initial_state.is_cuda(),
              "precompiled fused sigmoid gating expects CUDA initial_state");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "q, k, v must have shape [B, T, H/HV, K/V]");

  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t H = q.size(2);
  const int64_t K = q.size(3);
  const int64_t HV = v.size(2);
  const int64_t V = v.size(3);
  const int64_t hv_per_h = HV / H;
  auto output = torch::empty(v.sizes(), q.options());
  auto final_state = inplace_final_state
                         ? initial_state
                         : q.new_empty({T, HV, V, K}, initial_state.options());

  auto head_index =
      torch::arange(HV, torch::TensorOptions().dtype(torch::kLong).device(q.device())) /
      hv_per_h;
  auto A_log_f32 = A_log.to(torch::kFloat32);
  auto dt_bias_f32 = dt_bias.to(torch::kFloat32);

  torch::Tensor q_tokens;
  torch::Tensor k_tokens;
  torch::Tensor v_tokens;
  torch::Tensor a_tokens;
  torch::Tensor b_tokens;
  std::vector<std::pair<int64_t, int64_t>> seq_ranges;

  if (!cu_seqlens.has_value()) {
    q_tokens = q.reshape({B * T, H, K});
    k_tokens = k.reshape({B * T, H, K});
    v_tokens = v.reshape({B * T, HV, V});
    a_tokens = is_kda ? a.reshape({B * T, HV, K}) : a.reshape({B * T, HV});
    b_tokens = b.reshape({B * T, HV});
    seq_ranges.reserve(B);
    for (int64_t seq_idx = 0; seq_idx < B; ++seq_idx) {
      seq_ranges.emplace_back(seq_idx * T, (seq_idx + 1) * T);
    }
  } else {
    const auto& seq = cu_seqlens.value();
    q_tokens = q[0];
    k_tokens = k[0];
    v_tokens = v[0];
    a_tokens = is_kda ? a.reshape({-1, HV, K}) : a.reshape({-1, HV});
    b_tokens = b.reshape({-1, HV});
    const int64_t num_seqs = seq.size(0) - 1;
    seq_ranges.reserve(num_seqs);
    for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      seq_ranges.emplace_back(seq[seq_idx].item<int64_t>(),
                              seq[seq_idx + 1].item<int64_t>());
    }
  }

  auto out_tokens = output.reshape({-1, HV, V});

  for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(seq_ranges.size());
       ++seq_idx) {
    const auto [bos, eos] = seq_ranges[seq_idx];
    const int64_t seq_len = eos - bos;
    if (seq_len <= 0) {
      continue;
    }

    torch::Tensor h;
    if (!initial_state.defined()) {
      h = torch::zeros({HV, V, K},
                       torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
    } else if (ssm_state_indices.has_value()) {
      int64_t accepted_offset = 0;
      if (num_accepted_tokens.has_value()) {
        accepted_offset = num_accepted_tokens.value()[seq_idx].item<int64_t>() - 1;
      }
      const int64_t state_idx =
          resolve_state_index(ssm_state_indices, seq_idx, accepted_offset);
      if (state_idx == -2) {
        h = torch::zeros({HV, V, K},
                         torch::TensorOptions().dtype(torch::kFloat32)
                             .device(q.device()));
      } else if (state_idx < 0) {
        out_tokens.slice(0, bos, eos).zero_();
        continue;
      } else {
        h = initial_state[state_idx].to(torch::kFloat32).clone();
      }
    } else {
      int64_t state_base = seq_idx;
      if (initial_state.size(0) > bos) {
        state_base = bos;
      }
      h = initial_state[state_base].to(torch::kFloat32).clone();
    }

    for (int64_t token_offset = 0; token_offset < seq_len; ++token_offset) {
      const int64_t token_idx = bos + token_offset;

      auto q_token =
          q_tokens[token_idx].to(torch::kFloat32).index_select(0, head_index);
      auto k_token =
          k_tokens[token_idx].to(torch::kFloat32).index_select(0, head_index);
      auto v_token = v_tokens[token_idx].to(torch::kFloat32);
      auto beta_token = torch::sigmoid(b_tokens[token_idx].to(torch::kFloat32));

      torch::Tensor g;
      if (!is_kda) {
        auto x = a_tokens[token_idx].to(torch::kFloat32) + dt_bias_f32;
        g = -torch::exp(A_log_f32) * softplus_with_threshold(x, beta, threshold);
      } else {
        auto x =
            a_tokens[token_idx].to(torch::kFloat32) + dt_bias_f32.unsqueeze(1);
        g = -torch::exp(A_log_f32).unsqueeze(1) *
            softplus_with_threshold(x, beta, threshold);
      }

      if (use_qk_l2norm_in_kernel) {
        q_token = l2norm_last_dim(q_token);
        k_token = l2norm_last_dim(k_token);
      }

      q_token = q_token * scale;
      if (!is_kda) {
        h = h * torch::exp(g).view({HV, 1, 1});
      } else {
        h = h * torch::exp(g).unsqueeze(1);
      }

      auto updated_v =
          v_token - (h * k_token.unsqueeze(1)).sum(-1);
      updated_v = updated_v * beta_token.unsqueeze(1);
      h = h + updated_v.unsqueeze(-1) * k_token.unsqueeze(1);
      out_tokens[token_idx].copy_(
          (h * q_token.unsqueeze(1)).sum(-1).to(output.scalar_type()));

      if (inplace_final_state) {
        const int64_t final_state_idx =
            resolve_state_index(ssm_state_indices, seq_idx, token_offset);
        if (final_state_idx >= 0) {
          final_state[final_state_idx].copy_(h.to(final_state.scalar_type()));
        }
      } else {
        final_state[token_idx].copy_(h.to(final_state.scalar_type()));
      }
    }
  }

  if (ssm_state_indices.has_value()) {
    auto valid_mask = ssm_state_indices.value().reshape({-1}) >= 0;
    if (out_tokens.size(0) == valid_mask.size(0)) {
      auto zeros_output = torch::zeros(
          {1, HV, V},
          torch::TensorOptions().dtype(output.scalar_type()).device(output.device()));
      out_tokens.copy_(torch::where(valid_mask.view({-1, 1, 1}), out_tokens,
                                    zeros_output.expand_as(out_tokens)));
    }
  }

  return {output, final_state};
}
