#include <torch/all.h>

#include "ops.h"

#include <algorithm>
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
  return x;
}

torch::Tensor apply_optional_activation(const torch::Tensor& x,
                                        const std::string& activation) {
  if (activation.empty()) {
    return x;
  }
  return apply_gate_activation(x, activation);
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

int64_t resolve_cache_line(const std::optional<torch::Tensor>& cache_indices,
                           int64_t seq_idx, int64_t block_offset = 0) {
  if (!cache_indices.has_value()) {
    return seq_idx;
  }

  const auto& idx = cache_indices.value();
  if (idx.dim() == 1) {
    return idx[seq_idx].item<int64_t>();
  }
  return idx.index({seq_idx, block_offset}).item<int64_t>();
}

torch::Tensor load_initial_history(const torch::Tensor& state,
                                   int64_t history_len,
                                   bool load_initial_state,
                                   int64_t start_offset = 0) {
  if (!state.defined() || history_len <= 0 || !load_initial_state) {
    const int64_t dim = state.defined() ? state.size(0) : 0;
    auto options = state.defined()
                       ? state.options()
                       : torch::TensorOptions().dtype(torch::kFloat32);
    return torch::zeros({dim, history_len}, options);
  }

  auto history = state.slice(-1, start_offset, start_offset + history_len);
  if (history.size(-1) == history_len) {
    return history;
  }

  auto padded = torch::zeros({state.size(0), history_len}, state.options());
  if (history.numel() > 0) {
    padded.slice(-1, 0, history.size(-1)).copy_(history);
  }
  return padded;
}

torch::Tensor update_conv_state_ref(const torch::Tensor& state,
                                    const torch::Tensor& seq_tokens,
                                    int64_t shift_tokens) {
  const int64_t state_len = state.size(-1);
  if (state_len == 0) {
    return state;
  }

  const int64_t safe_shift = std::max<int64_t>(shift_tokens, 0);
  auto retained_state = state.slice(-1, std::min<int64_t>(safe_shift, state_len));
  auto updated = torch::cat({retained_state, seq_tokens}, -1);

  if (updated.size(-1) >= state_len) {
    return updated.slice(-1, updated.size(-1) - state_len, updated.size(-1));
  }

  auto padded = torch::zeros_like(state);
  padded.slice(-1, state_len - updated.size(-1), state_len).copy_(updated);
  return padded;
}

torch::Tensor causal_conv1d_sequence_ref(
    const torch::Tensor& seq_tokens, const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias, const torch::Tensor& history,
    const std::string& activation) {
  auto combined = torch::cat({history, seq_tokens}, -1);
  auto windows = combined.unfold(-1, weight.size(1), 1);
  auto output =
      (windows * weight.unsqueeze(1)).sum(-1);
  if (bias.has_value()) {
    output = output + bias.value().unsqueeze(1);
  }
  return apply_optional_activation(output, activation);
}

torch::Tensor l2norm_last_dim(const torch::Tensor& x) {
  return x / torch::sqrt((x * x).sum(-1, true) + 1e-6);
}

torch::Tensor expand_qk_heads(const torch::Tensor& x, int64_t target_heads) {
  if (x.size(2) == target_heads) {
    return x;
  }
  TORCH_CHECK(target_heads % x.size(2) == 0, "Cannot expand ", x.size(2),
              " query/key heads to ", target_heads, " value heads.");
  return x.repeat_interleave(target_heads / x.size(2), 2);
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
          q_tokens[token_idx].to(torch::kFloat32).repeat_interleave(hv_per_h, 0);
      auto k_token =
          k_tokens[token_idx].to(torch::kFloat32).repeat_interleave(hv_per_h, 0);
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

torch::Tensor apply_rotary_emb_precompiled(const torch::Tensor& x,
                                           const torch::Tensor& cos,
                                           const torch::Tensor& sin,
                                           bool is_neox_style,
                                           bool enable_fp32_compute) {
  TORCH_CHECK(x.is_cuda(), "apply_rotary_emb_precompiled expects CUDA x");
  TORCH_CHECK(cos.is_cuda(), "apply_rotary_emb_precompiled expects CUDA cos");
  TORCH_CHECK(sin.is_cuda(), "apply_rotary_emb_precompiled expects CUDA sin");
  TORCH_CHECK(x.dim() == 3 || x.dim() == 4,
              "apply_rotary_emb_precompiled expects x with rank 3 or 4");

  auto working_x = enable_fp32_compute ? x.to(torch::kFloat32) : x;
  auto working_cos = enable_fp32_compute ? cos.to(torch::kFloat32) : cos;
  auto working_sin = enable_fp32_compute ? sin.to(torch::kFloat32) : sin;
  const bool added_batch = working_x.dim() == 3;
  if (added_batch) {
    working_x = working_x.unsqueeze(0);
  }

  auto cos_expanded = working_cos.unsqueeze(-2).to(working_x.scalar_type());
  auto sin_expanded = working_sin.unsqueeze(-2).to(working_x.scalar_type());

  torch::Tensor output;
  if (is_neox_style) {
    auto parts = working_x.split(working_x.size(-1) / 2, -1);
    auto x1 = parts[0];
    auto x2 = parts[1];
    auto o1 = x1 * cos_expanded - x2 * sin_expanded;
    auto o2 = x2 * cos_expanded + x1 * sin_expanded;
    output = torch::cat({o1, o2}, -1);
  } else {
    auto x1 = working_x.slice(-1, 0, working_x.size(-1), 2);
    auto x2 = working_x.slice(-1, 1, working_x.size(-1), 2);
    auto o1 = x1 * cos_expanded - x2 * sin_expanded;
    auto o2 = x2 * cos_expanded + x1 * sin_expanded;
    output = torch::stack({o1, o2}, -1).flatten(-2);
  }

  if (added_batch) {
    output = output.squeeze(0);
  }
  if (enable_fp32_compute) {
    output = output.to(x.scalar_type());
  }
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> chunk_gated_delta_rule_precompiled(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& g, const torch::Tensor& beta, double scale,
    const torch::Tensor& initial_state, bool output_final_state,
    const std::optional<torch::Tensor>& cu_seqlens,
    bool use_qk_l2norm_in_kernel) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && g.is_cuda() &&
                  beta.is_cuda() && initial_state.is_cuda(),
              "chunk_gated_delta_rule_precompiled expects CUDA tensors");

  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t H = v.size(2);
  const int64_t K = q.size(3);
  const int64_t V = v.size(3);
  const int64_t N = cu_seqlens.has_value() ? cu_seqlens.value().size(0) - 1 : B;

  auto q_work = expand_qk_heads(q.to(torch::kFloat32), H);
  auto k_work = expand_qk_heads(k.to(torch::kFloat32), H);
  if (use_qk_l2norm_in_kernel) {
    q_work = l2norm_last_dim(q_work);
    k_work = l2norm_last_dim(k_work);
  }
  auto v_work = v.to(torch::kFloat32);
  auto g_work = g.to(torch::kFloat32);
  auto beta_work = beta.to(torch::kFloat32);

  auto output = torch::empty({B, T, H, V}, q.options());
  auto final_state = output_final_state
                         ? torch::empty(
                               {N, H, V, K},
                               torch::TensorOptions().dtype(torch::kFloat32).device(q.device()))
                         : torch::empty(
                               {0},
                               torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));

  for (int64_t seq_idx = 0; seq_idx < N; ++seq_idx) {
    int64_t batch_idx = seq_idx;
    int64_t start = 0;
    int64_t end = T;
    if (cu_seqlens.has_value()) {
      batch_idx = 0;
      start = cu_seqlens.value()[seq_idx].item<int64_t>();
      end = cu_seqlens.value()[seq_idx + 1].item<int64_t>();
    }

    auto state = initial_state[seq_idx].to(torch::kFloat32).clone();
    for (int64_t tok_idx = start; tok_idx < end; ++tok_idx) {
      auto q_tok = q_work.index({batch_idx, tok_idx});
      auto k_tok = k_work.index({batch_idx, tok_idx});
      auto v_tok = v_work.index({batch_idx, tok_idx});
      auto beta_tok = beta_work.index({batch_idx, tok_idx}).unsqueeze(-1);
      auto decay =
          torch::exp(g_work.index({batch_idx, tok_idx})).unsqueeze(-1).unsqueeze(-1);
      state = state * decay;
      auto delta_v =
          v_tok - torch::matmul(state, k_tok.unsqueeze(-1)).squeeze(-1);
      delta_v = delta_v * beta_tok;
      state = state + delta_v.unsqueeze(-1) * k_tok.unsqueeze(-2);
      output.index_put_({batch_idx, tok_idx},
                        (torch::matmul(state, q_tok.unsqueeze(-1)).squeeze(-1) * scale)
                            .to(output.scalar_type()));
    }

    if (output_final_state) {
      final_state.index_put_({seq_idx}, state);
    }
  }

  return {output, final_state};
}

std::tuple<torch::Tensor, torch::Tensor> fused_gdn_gating_precompiled(
    const torch::Tensor& A_log, const torch::Tensor& a,
    const torch::Tensor& b, const torch::Tensor& dt_bias, double beta,
    double threshold) {
  TORCH_CHECK(A_log.is_cuda() && a.is_cuda() && b.is_cuda() && dt_bias.is_cuda(),
              "fused_gdn_gating_precompiled expects CUDA tensors");
  auto x = a.to(torch::kFloat32) + dt_bias.to(torch::kFloat32);
  auto g =
      (-torch::exp(A_log.to(torch::kFloat32)) *
       softplus_with_threshold(x, beta, threshold))
          .unsqueeze(0);
  auto beta_output =
      torch::sigmoid(b.to(torch::kFloat32)).to(b.scalar_type()).unsqueeze(0);
  return {g, beta_output};
}

torch::Tensor causal_conv1d_fn_precompiled(
    const torch::Tensor& x, const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias, torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const std::optional<torch::Tensor>& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const std::string& activation, int64_t pad_slot_id) {
  TORCH_CHECK(x.is_cuda(), "causal_conv1d_fn_precompiled expects CUDA x");
  TORCH_CHECK(weight.is_cuda(),
              "causal_conv1d_fn_precompiled expects CUDA weight");
  TORCH_CHECK(conv_states.is_cuda(),
              "causal_conv1d_fn_precompiled expects CUDA conv_states");
  TORCH_CHECK(x.dim() == 2,
              "causal_conv1d_fn_precompiled expects x with shape [dim, tokens]");

  auto out = torch::zeros_like(x);
  const int64_t history_len = weight.size(1) - 1;
  const int64_t batch = query_start_loc.numel() - 1;

  for (int64_t seq_idx = 0; seq_idx < batch; ++seq_idx) {
    const int64_t seq_start = query_start_loc[seq_idx].item<int64_t>();
    const int64_t seq_end = query_start_loc[seq_idx + 1].item<int64_t>();
    if (seq_end <= seq_start) {
      continue;
    }

    const int64_t cache_line = resolve_cache_line(cache_indices, seq_idx);
    if (cache_line == pad_slot_id) {
      continue;
    }

    const auto state = conv_states[cache_line];
    const bool load_initial_state =
        !has_initial_state.has_value() ||
        has_initial_state.value()[seq_idx].item<bool>();
    auto history = load_initial_history(state, history_len, load_initial_state);
    auto seq_tokens = x.slice(1, seq_start, seq_end);
    out.slice(1, seq_start, seq_end)
        .copy_(causal_conv1d_sequence_ref(seq_tokens, weight, bias, history,
                                          activation));
    conv_states[cache_line].copy_(update_conv_state_ref(
        conv_states[cache_line], seq_tokens, seq_tokens.size(-1)));
  }

  return out;
}

void zero_kv_blocks_precompiled(const torch::Tensor& block_ids,
                                c10::List<torch::Tensor> kv_tensors,
                                c10::List<int64_t> block_dims,
                                c10::List<int64_t> ratios) {
  TORCH_CHECK(block_ids.is_cuda(),
              "zero_kv_blocks_precompiled expects CUDA block_ids");
  TORCH_CHECK(kv_tensors.size() == block_dims.size() &&
                  kv_tensors.size() == ratios.size(),
              "kv_tensors, block_dims, and ratios must have the same length");

  auto block_ids_cpu = block_ids.to(torch::kCPU);
  for (size_t tensor_idx = 0; tensor_idx < kv_tensors.size(); ++tensor_idx) {
    auto kv = kv_tensors.get(tensor_idx);
    TORCH_CHECK(kv.is_cuda(),
                "zero_kv_blocks_precompiled expects CUDA kv tensors");
    const int64_t block_dim = block_dims.get(tensor_idx);
    const int64_t ratio = ratios.get(tensor_idx);
    for (int64_t idx = 0; idx < block_ids_cpu.numel(); ++idx) {
      const int64_t block_id = block_ids_cpu[idx].item<int64_t>();
      kv.narrow(block_dim, block_id * ratio, ratio).zero_();
    }
  }
}

torch::Tensor causal_conv1d_update_precompiled(
    const torch::Tensor& x, torch::Tensor& conv_state,
    const torch::Tensor& weight, const std::optional<torch::Tensor>& bias,
    const std::string& activation,
    const std::optional<torch::Tensor>& conv_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t pad_slot_id,
    const std::optional<torch::Tensor>& block_idx_last_scheduled_token,
    const std::optional<torch::Tensor>& initial_state_idx) {
  TORCH_CHECK(x.is_cuda(), "causal_conv1d_update_precompiled expects CUDA x");
  TORCH_CHECK(weight.is_cuda(),
              "causal_conv1d_update_precompiled expects CUDA weight");
  TORCH_CHECK(conv_state.is_cuda(),
              "causal_conv1d_update_precompiled expects CUDA conv_state");

  const bool use_fast_path =
      !query_start_loc.has_value() && !num_accepted_tokens.has_value() &&
      !block_idx_last_scheduled_token.has_value() &&
      !initial_state_idx.has_value() &&
      (!conv_state_indices.has_value() || conv_state_indices.value().dim() == 1);

  if (use_fast_path) {
    TORCH_CHECK(x.dim() == 3,
                "fast-path causal_conv1d_update_precompiled expects [B, D, T]");
    const int64_t batch = x.size(0);
    const int64_t dim = x.size(1);
    const int64_t seqlen = x.size(2);
    const int64_t history_len = weight.size(1) - 1;

    auto safe_indices =
        conv_state_indices.has_value()
            ? conv_state_indices.value().to(torch::TensorOptions()
                                                .dtype(torch::kLong)
                                                .device(x.device()))
            : torch::arange(batch, torch::TensorOptions()
                                       .dtype(torch::kLong)
                                       .device(x.device()));
    auto valid_mask = safe_indices >= 0;
    auto gather_indices = safe_indices.clamp_min(0);
    auto state = conv_state.index_select(0, gather_indices);
    auto history = state.slice(-1, 0, history_len);
    auto combined = torch::cat({history, x}, -1);
    auto windows = combined.unfold(-1, weight.size(1), 1);
    auto out =
        (windows * weight.view({1, dim, 1, weight.size(1)})).sum(-1);
    if (bias.has_value()) {
      out = out + bias.value().view({1, dim, 1});
    }
    out = apply_optional_activation(out, activation);
    out = out * valid_mask.view({batch, 1, 1}).to(out.scalar_type());

    auto updated_state =
        update_conv_state_ref(state.reshape({batch * dim, state.size(-1)}),
                              x.reshape({batch * dim, seqlen}), seqlen)
            .reshape({batch, dim, state.size(-1)});
    conv_state.index_copy_(0, gather_indices, updated_state);
    return out;
  }

  auto out = x.clone();
  const int64_t history_len = weight.size(1) - 1;
  const int64_t batch = query_start_loc.has_value()
                            ? query_start_loc.value().numel() - 1
                            : x.size(0);

  for (int64_t seq_idx = 0; seq_idx < batch; ++seq_idx) {
    torch::Tensor seq_tokens;
    int64_t seq_len = 0;
    int64_t seq_start = 0;
    int64_t seq_end = 0;
    if (!query_start_loc.has_value()) {
      seq_tokens = x[seq_idx];
      seq_len = seq_tokens.size(-1);
    } else {
      seq_start = query_start_loc.value()[seq_idx].item<int64_t>();
      seq_end = query_start_loc.value()[seq_idx + 1].item<int64_t>();
      if (seq_end <= seq_start) {
        continue;
      }
      seq_tokens =
          x.slice(0, seq_start, seq_end).transpose(0, 1).contiguous();
      seq_len = seq_end - seq_start;
    }

    int64_t input_block_offset = 0;
    if (initial_state_idx.has_value()) {
      input_block_offset = initial_state_idx.value()[seq_idx].item<int64_t>();
    }
    const int64_t input_cache_line =
        resolve_cache_line(conv_state_indices, seq_idx, input_block_offset);
    if (input_cache_line == pad_slot_id) {
      continue;
    }

    int64_t output_block_offset = 0;
    if (block_idx_last_scheduled_token.has_value()) {
      output_block_offset =
          block_idx_last_scheduled_token.value()[seq_idx].item<int64_t>();
    }
    const int64_t output_cache_line =
        resolve_cache_line(conv_state_indices, seq_idx, output_block_offset);

    const auto state = conv_state[input_cache_line];
    int64_t history_offset = 0;
    int64_t shift_tokens = seq_len;
    if (num_accepted_tokens.has_value()) {
      history_offset =
          std::max<int64_t>(num_accepted_tokens.value()[seq_idx].item<int64_t>() - 1,
                            0);
      shift_tokens = 1;
    }

    auto history =
        load_initial_history(state, history_len, true, history_offset);
    auto seq_output =
        causal_conv1d_sequence_ref(seq_tokens, weight, bias, history, activation);

    if (!query_start_loc.has_value()) {
      out[seq_idx].slice(-1, 0, seq_len).copy_(seq_output);
    } else {
      out.slice(0, seq_start, seq_end).copy_(seq_output.transpose(0, 1));
    }

    if (output_cache_line != pad_slot_id) {
      conv_state[output_cache_line].copy_(update_conv_state_ref(
          conv_state[input_cache_line], seq_tokens, shift_tokens));
    }
  }

  return out;
}
