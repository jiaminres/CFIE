#include <torch/all.h>

#include "ops.h"

#include <ATen/ops/scaled_dot_product_attention.h>

#include <algorithm>
#include <limits>
#include <vector>

namespace {

constexpr double kLog2E = 1.4426950408889634;
constexpr double kLogE2 = 0.6931471805599453;

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

torch::Tensor normalize_rowwise_param(const torch::Tensor& value,
                                      const torch::Tensor& logits,
                                      torch::ScalarType dtype) {
  TORCH_CHECK(value.defined(), "row-wise parameter tensor must be defined");
  TORCH_CHECK(value.dim() <= 1,
              "row-wise parameter tensor must be scalar or 1D");

  auto normalized =
      value.to(logits.device(), dtype, false, false).contiguous().view(-1);
  if (normalized.numel() == 1 && logits.size(0) != 1) {
    normalized = normalized.expand({logits.size(0)}).contiguous();
  }

  TORCH_CHECK(normalized.numel() == logits.size(0),
              "row-wise parameter tensor size must match batch size");
  return normalized;
}

torch::Tensor build_shifted_top_p_mask(const torch::Tensor& probs_cumsum,
                                       const torch::Tensor& resolved_p) {
  auto top_p_mask = probs_cumsum > resolved_p.unsqueeze(1);
  if (probs_cumsum.size(1) > 1) {
    auto shifted = top_p_mask.clone();
    top_p_mask.slice(1, 1, probs_cumsum.size(1))
        .copy_(shifted.slice(1, 0, probs_cumsum.size(1) - 1));
  }
  top_p_mask.select(1, 0).fill_(false);
  return top_p_mask;
}

bool try_apply_top_k_only_with_cuda_topk_per_row(torch::Tensor& logits,
                                                 const torch::Tensor& resolved_k,
                                                 double mask_value) {
  if (logits.scalar_type() != torch::kFloat32 || !logits.is_contiguous()) {
    return false;
  }

  auto no_top_k_mask = resolved_k == logits.size(1);
  if (no_top_k_mask.all().item<bool>()) {
    return true;
  }

  auto effective_k = resolved_k.masked_fill(no_top_k_mask, 1);
  const int64_t max_k = effective_k.max().item<int64_t>();
  auto topk_indices = torch::empty(
      {logits.size(0), max_k},
      torch::TensorOptions().dtype(torch::kInt32).device(logits.device()));
  auto seq_lens = torch::full(
      {logits.size(0)}, logits.size(1),
      torch::TensorOptions().dtype(torch::kInt32).device(logits.device()));
  top_k_per_row_decode(logits, 1, seq_lens, topk_indices, logits.size(0),
                       logits.stride(0), logits.stride(1), max_k);

  auto k_index = effective_k.sub(1).unsqueeze(1);
  auto topk_indices_long = topk_indices.to(torch::kLong);
  auto topk_values = logits.gather(1, topk_indices_long);
  auto topk_values_sorted = std::get<0>(torch::sort(topk_values, 1, true));
  auto top_k_threshold = topk_values_sorted.gather(1, k_index);
  top_k_threshold.masked_fill_(no_top_k_mask.unsqueeze(1), mask_value);
  logits.masked_fill_(logits < top_k_threshold, mask_value);
  return true;
}

void apply_top_p_only_iterative_topk_precompiled(torch::Tensor& logits,
                                                 const torch::Tensor& resolved_p,
                                                 double mask_value) {
  TORCH_CHECK(logits.dim() == 2,
              "apply_top_p_only_iterative_topk_precompiled expects 2D logits");
  TORCH_CHECK(resolved_p.dim() == 1,
              "apply_top_p_only_iterative_topk_precompiled expects 1D p");
  TORCH_CHECK(logits.size(0) == resolved_p.numel(),
              "apply_top_p_only_iterative_topk_precompiled expects matching "
              "batch size");

  auto no_top_p_mask = resolved_p >= 1.0;
  if (no_top_p_mask.all().item<bool>()) {
    return;
  }

  auto active_rows = torch::nonzero(~no_top_p_mask).view(-1);
  auto active_logits = logits.index_select(0, active_rows);
  auto active_p = resolved_p.index_select(0, active_rows);
  const int64_t vocab_size = active_logits.size(1);

  int64_t current_k = std::min<int64_t>(128, vocab_size);
  auto active_logsumexp =
      torch::logsumexp(active_logits.to(torch::kFloat32), 1, true);

  torch::Tensor topk_values;
  torch::Tensor topk_indices;
  torch::Tensor probs_cumsum;
  while (true) {
    auto topk = torch::topk(active_logits, current_k, 1, true, true);
    topk_values = std::get<0>(topk);
    topk_indices = std::get<1>(topk);
    auto topk_probs =
        torch::exp(topk_values.to(torch::kFloat32) - active_logsumexp);
    probs_cumsum = torch::cumsum(topk_probs, -1);

    if (current_k == vocab_size ||
        probs_cumsum.select(1, current_k - 1).ge(active_p).all().item<bool>()) {
      break;
    }
    current_k = std::min<int64_t>(current_k * 2, vocab_size);
  }

  auto top_p_mask = build_shifted_top_p_mask(probs_cumsum, active_p);
  auto filtered_values = topk_values.masked_fill(top_p_mask, mask_value);
  active_logits.fill_(mask_value);
  active_logits.scatter_(1, topk_indices, filtered_values);
  logits.index_copy_(0, active_rows, active_logits);
}

void apply_top_k_then_top_p_small_k_precompiled(
    torch::Tensor& logits, const torch::Tensor& resolved_k,
    const torch::Tensor& resolved_p, double mask_value) {
  TORCH_CHECK(logits.dim() == 2,
              "apply_top_k_then_top_p_small_k_precompiled expects 2D logits");
  TORCH_CHECK(resolved_k.dim() == 1 && resolved_p.dim() == 1,
              "apply_top_k_then_top_p_small_k_precompiled expects 1D inputs");
  TORCH_CHECK(logits.size(0) == resolved_k.numel() &&
                  logits.size(0) == resolved_p.numel(),
              "apply_top_k_then_top_p_small_k_precompiled expects matching "
              "batch size");
  TORCH_CHECK(resolved_k.max().item<int64_t>() < logits.size(1),
              "apply_top_k_then_top_p_small_k_precompiled expects k < vocab");

  const int64_t max_k = resolved_k.max().item<int64_t>();
  auto topk = torch::topk(logits, max_k, 1, true, true);
  auto topk_values = std::get<0>(topk);
  auto topk_indices = std::get<1>(topk);

  auto positions = torch::arange(max_k, resolved_k.options()).unsqueeze(0);
  auto valid_k_mask = positions < resolved_k.unsqueeze(1);
  auto filtered_values = topk_values.masked_fill(~valid_k_mask, mask_value);

  auto probs_desc = filtered_values.softmax(-1, torch::kFloat32);
  auto probs_cumsum = torch::cumsum(probs_desc, -1);
  auto top_p_mask = build_shifted_top_p_mask(probs_cumsum, resolved_p);

  filtered_values.masked_fill_(top_p_mask | (~valid_k_mask), mask_value);
  logits.fill_(mask_value);
  logits.scatter_(1, topk_indices, filtered_values);
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

torch::Tensor move_tensor_to_target_device(const torch::Tensor& src,
                                           const torch::Tensor& dst) {
  auto contiguous_src = src.contiguous();
  if (contiguous_src.device() == dst.device()) {
    return contiguous_src;
  }

  auto dst_src = torch::empty(
      contiguous_src.sizes(),
      contiguous_src.options().device(dst.device()));
  dst_src.copy_(contiguous_src, true);
  return dst_src;
}

torch::Tensor move_slot_ids_to_target_device(const torch::Tensor& slot_ids,
                                             const torch::Tensor& dst) {
  auto slot_ids_long = slot_ids.to(torch::kLong).contiguous();
  if (slot_ids_long.device() == dst.device()) {
    return slot_ids_long;
  }

  auto slot_ids_device = torch::empty(
      slot_ids_long.sizes(),
      slot_ids_long.options().device(dst.device()));
  slot_ids_device.copy_(slot_ids_long, true);
  return slot_ids_device;
}

void batch_copy_into_slots(const torch::Tensor& slot_ids,
                           const torch::Tensor& src,
                           torch::Tensor& dst,
                           const char* field_name) {
  TORCH_CHECK(dst.is_cuda(), field_name, " destination must be CUDA");
  TORCH_CHECK(src.dim() == dst.dim(), field_name,
              " source and destination rank must match");
  TORCH_CHECK(src.size(0) == slot_ids.numel(), field_name,
              " source batch size must match slot_ids");
  for (int64_t dim = 1; dim < src.dim(); ++dim) {
    TORCH_CHECK(src.size(dim) == dst.size(dim), field_name,
                " source and destination shapes must match after batch dim");
  }

  auto slot_ids_device = move_slot_ids_to_target_device(slot_ids, dst);
  auto src_device = move_tensor_to_target_device(src, dst);
  dst.index_copy_(0, slot_ids_device, src_device);
}

void batch_copy_into_slots_optional(
    const torch::Tensor& slot_ids,
    const std::optional<torch::Tensor>& src,
    const std::optional<torch::Tensor>& dst,
    const char* field_name) {
  if (!src.has_value() && !dst.has_value()) {
    return;
  }
  TORCH_CHECK(src.has_value() && dst.has_value(), field_name,
              " source and destination must either both exist or both be None");
  auto dst_tensor = dst.value();
  batch_copy_into_slots(slot_ids, src.value(), dst_tensor, field_name);
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

torch::Tensor correct_attn_out_precompiled(torch::Tensor& out,
                                           const torch::Tensor& lses,
                                           int64_t cp_rank,
                                           bool is_lse_base_on_e) {
  TORCH_CHECK(out.is_cuda(),
              "correct_attn_out_precompiled expects CUDA out");
  TORCH_CHECK(lses.is_cuda(),
              "correct_attn_out_precompiled expects CUDA lses");
  TORCH_CHECK(out.dim() == 3,
              "correct_attn_out_precompiled expects out with shape [B, H, D]");
  TORCH_CHECK(lses.dim() == 3,
              "correct_attn_out_precompiled expects lses with shape [N, B, H]");
  TORCH_CHECK(lses.size(1) == out.size(0) && lses.size(2) == out.size(1),
              "correct_attn_out_precompiled expects lses [N, B, H] to match "
              "out [B, H, D]");
  TORCH_CHECK(cp_rank >= 0 && cp_rank < lses.size(0),
              "correct_attn_out_precompiled expects cp_rank within [0, N)");

  const double neg_inf = -std::numeric_limits<double>::infinity();
  auto neg_inf_scalar = torch::full({}, neg_inf, lses.options());

  auto sanitized =
      torch::where(torch::isnan(lses) | torch::isinf(lses), neg_inf_scalar, lses);
  auto lse_max = sanitized.amax(0);
  lse_max = torch::where(lse_max == neg_inf_scalar, torch::zeros_like(lse_max),
                         lse_max);

  auto shifted = sanitized - lse_max.unsqueeze(0);
  torch::Tensor final_lse;
  if (is_lse_base_on_e) {
    final_lse = torch::log(torch::exp(shifted).sum(0)) + lse_max;
  } else {
    final_lse =
        torch::log(torch::exp(shifted * kLogE2).sum(0)) * kLog2E + lse_max;
  }

  auto lse = torch::empty_strided({out.size(0), out.size(1)},
                                  {lses.stride(1), lses.stride(2)},
                                  lses.options());
  lse.copy_(final_lse);

  auto local_lse = sanitized.select(0, cp_rank) - lse;
  local_lse =
      torch::where(torch::isnan(local_lse) | torch::isinf(local_lse),
                   torch::full({}, neg_inf, local_lse.options()), local_lse);
  auto factor = is_lse_base_on_e ? torch::exp(local_lse)
                                 : torch::exp(local_lse * kLogE2);
  out.mul_(factor.unsqueeze(-1).to(out.scalar_type()));
  return lse;
}

std::tuple<torch::Tensor, torch::Tensor> dcp_lse_combine_precompiled(
    const torch::Tensor& recv_output, const torch::Tensor& recv_lse,
    bool return_lse, bool is_lse_base_on_e) {
  TORCH_CHECK(recv_output.is_cuda(),
              "dcp_lse_combine_precompiled expects CUDA recv_output");
  TORCH_CHECK(recv_lse.is_cuda(),
              "dcp_lse_combine_precompiled expects CUDA recv_lse");
  TORCH_CHECK(recv_output.dim() == 4,
              "dcp_lse_combine_precompiled expects recv_output with shape "
              "[N, B, H, D]");
  TORCH_CHECK(recv_lse.dim() == 3,
              "dcp_lse_combine_precompiled expects recv_lse with shape "
              "[N, B, H]");
  TORCH_CHECK(recv_output.size(0) == recv_lse.size(0) &&
                  recv_output.size(1) == recv_lse.size(1) &&
                  recv_output.size(2) == recv_lse.size(2),
              "dcp_lse_combine_precompiled expects recv_output [N, B, H, D] "
              "and recv_lse [N, B, H] to agree on [N, B, H]");

  const double neg_inf = -std::numeric_limits<double>::infinity();
  auto neg_inf_scalar = torch::full({}, neg_inf, recv_lse.options());

  auto sanitized = torch::where(
      torch::isnan(recv_lse) | torch::isinf(recv_lse), neg_inf_scalar, recv_lse);
  auto lse_max = sanitized.amax(0);
  lse_max = torch::where(lse_max == neg_inf_scalar, torch::zeros_like(lse_max),
                         lse_max);

  auto shifted = sanitized - lse_max.unsqueeze(0);
  torch::Tensor weights;
  torch::Tensor global_lse;
  if (is_lse_base_on_e) {
    weights = torch::exp(shifted);
    global_lse = torch::log(weights.sum(0)) + lse_max;
  } else {
    weights = torch::exp(shifted * kLogE2);
    global_lse = torch::log(weights.sum(0)) * kLog2E + lse_max;
  }

  weights = torch::where(torch::isnan(weights), torch::zeros_like(weights), weights);
  auto weight_sum = weights.sum(0, true);
  auto normalized = weights / weight_sum.clamp_min(1e-10);
  auto result =
      (recv_output.to(torch::kFloat32) *
       normalized.unsqueeze(-1).to(torch::kFloat32))
          .sum(0)
          .to(recv_output.scalar_type());

  if (!return_lse) {
    global_lse = torch::empty({0}, recv_lse.options());
  }

  return {result, global_lse};
}

void prefill_attention_precompiled(torch::Tensor& output,
                                   const torch::Tensor& q,
                                   const torch::Tensor& k,
                                   const torch::Tensor& v,
                                   const torch::Tensor& b_start_loc,
                                   const torch::Tensor& b_seq_len,
                                   bool is_causal, double softmax_scale,
                                   int64_t sliding_window_q,
                                   int64_t sliding_window_k) {
  TORCH_CHECK(output.is_cuda(),
              "prefill_attention_precompiled expects CUDA output");
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "prefill_attention_precompiled expects CUDA q/k/v");
  TORCH_CHECK(b_start_loc.is_cuda() && b_seq_len.is_cuda(),
              "prefill_attention_precompiled expects CUDA sequence metadata");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && output.dim() == 3,
              "prefill_attention_precompiled expects q/k/v/output with "
              "shape [T, H, D]");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0) &&
                  q.size(0) == output.size(0),
              "prefill_attention_precompiled expects q/k/v/output to agree "
              "on token dimension");
  TORCH_CHECK(output.size(1) == q.size(1),
              "prefill_attention_precompiled expects output heads to match "
              "query heads");
  TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) == v.size(2) &&
                  q.size(2) == output.size(2),
              "prefill_attention_precompiled expects q/k/v/output to agree "
              "on head dimension");
  TORCH_CHECK(k.size(1) == v.size(1),
              "prefill_attention_precompiled expects matching KV heads");
  TORCH_CHECK(q.size(1) % k.size(1) == 0,
              "prefill_attention_precompiled expects query heads to be a "
              "multiple of KV heads");
  TORCH_CHECK(b_start_loc.dim() == 1 && b_seq_len.dim() == 1,
              "prefill_attention_precompiled expects 1D sequence metadata");
  TORCH_CHECK(b_start_loc.numel() == b_seq_len.numel(),
              "prefill_attention_precompiled expects b_start_loc and "
              "b_seq_len to have the same batch size");

  const int64_t batch = b_seq_len.numel();
  const int64_t kv_group_num = q.size(1) / k.size(1);
  const int64_t q_window = std::max<int64_t>(sliding_window_q, 0);
  const int64_t k_window = std::max<int64_t>(sliding_window_k, 0);
  auto starts_cpu = b_start_loc.to(torch::kLong).cpu();
  auto seq_lens_cpu = b_seq_len.to(torch::kLong).cpu();
  auto long_options =
      torch::TensorOptions().dtype(torch::kLong).device(q.device());
  auto bool_options =
      torch::TensorOptions().dtype(torch::kBool).device(q.device());

  for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
    const int64_t seq_start = starts_cpu[batch_idx].item<int64_t>();
    const int64_t seq_len = seq_lens_cpu[batch_idx].item<int64_t>();
    const int64_t seq_stop = seq_start + seq_len;

    TORCH_CHECK(seq_len >= 0,
                "prefill_attention_precompiled expects non-negative "
                "sequence lengths");
    if (seq_len == 0) {
      continue;
    }

    auto q_seq = q.slice(0, seq_start, seq_stop).to(torch::kFloat32);
    auto k_seq = k.slice(0, seq_start, seq_stop).to(torch::kFloat32);
    auto v_seq = v.slice(0, seq_start, seq_stop).to(torch::kFloat32);

    auto q_heads = q_seq.permute({1, 0, 2}).contiguous();
    auto k_heads = k_seq.permute({1, 0, 2}).contiguous();
    auto v_heads = v_seq.permute({1, 0, 2}).contiguous();

    if (kv_group_num > 1) {
      k_heads = k_heads.repeat_interleave(kv_group_num, 0);
      v_heads = v_heads.repeat_interleave(kv_group_num, 0);
    }

    auto positions = torch::arange(seq_len, long_options);
    auto q_pos = positions.unsqueeze(1);
    auto k_pos = positions.unsqueeze(0);
    auto mask = torch::ones({seq_len, seq_len}, bool_options);

    if (is_causal) {
      mask = mask & (q_pos >= k_pos);
    }
    if (q_window > 0) {
      mask = mask & ((q_pos - k_pos) <= q_window);
    }
    if (k_window > 0) {
      mask = mask & ((k_pos - q_pos) <= k_window);
    }

    std::optional<at::Tensor> attn_mask(mask.unsqueeze(0).unsqueeze(0));
    auto out = at::scaled_dot_product_attention(
                   q_heads.unsqueeze(0), k_heads.unsqueeze(0),
                   v_heads.unsqueeze(0), attn_mask, 0.0, false,
                   std::optional<double>(softmax_scale), false)
                   .squeeze(0);
    output.slice(0, seq_start, seq_stop)
        .copy_(out.permute({1, 0, 2}).to(output.scalar_type()));
  }
}

void prefix_prefill_attention_precompiled(
    torch::Tensor& output, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& gathered_ctx_k,
    const torch::Tensor& gathered_ctx_v, const torch::Tensor& cu_ctx_lens,
    const torch::Tensor& b_start_loc, const torch::Tensor& b_seq_len,
    double sm_scale, int64_t sliding_window, bool skip_decode) {
  TORCH_CHECK(output.is_cuda(),
              "prefix_prefill_attention_precompiled expects CUDA output");
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
              "prefix_prefill_attention_precompiled expects CUDA q/k/v");
  TORCH_CHECK(gathered_ctx_k.is_cuda() && gathered_ctx_v.is_cuda(),
              "prefix_prefill_attention_precompiled expects CUDA gathered "
              "context tensors");
  TORCH_CHECK(cu_ctx_lens.is_cuda(),
              "prefix_prefill_attention_precompiled expects CUDA cu_ctx_lens");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && output.dim() == 3,
              "prefix_prefill_attention_precompiled expects q/k/v/output with "
              "shape [T, H, D]");
  TORCH_CHECK(gathered_ctx_k.dim() == 3 && gathered_ctx_v.dim() == 3,
              "prefix_prefill_attention_precompiled expects gathered context "
              "tensors with shape [Tctx, Hkv, D]");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0) &&
                  q.size(0) == output.size(0),
              "prefix_prefill_attention_precompiled expects q/k/v/output to "
              "agree on token dimension");
  TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) == v.size(2) &&
                  q.size(2) == output.size(2),
              "prefix_prefill_attention_precompiled expects q/k/v/output to "
              "agree on head dimension");
  TORCH_CHECK(k.size(1) == v.size(1) && gathered_ctx_k.size(1) == k.size(1) &&
                  gathered_ctx_v.size(1) == v.size(1),
              "prefix_prefill_attention_precompiled expects matching KV "
              "heads");
  TORCH_CHECK(q.size(1) % k.size(1) == 0,
              "prefix_prefill_attention_precompiled expects query heads to be "
              "a multiple of KV heads");
  TORCH_CHECK(b_seq_len.dim() == 1 && b_start_loc.dim() == 1 &&
                  cu_ctx_lens.dim() == 1,
              "prefix_prefill_attention_precompiled expects 1D sequence "
              "metadata tensors");
  TORCH_CHECK(b_start_loc.numel() == b_seq_len.numel() + 1 &&
                  cu_ctx_lens.numel() == b_seq_len.numel() + 1,
              "prefix_prefill_attention_precompiled expects metadata prefix "
              "sums with batch + 1 elements");

  const int64_t batch = b_seq_len.numel();
  const int64_t num_q_heads = q.size(1);
  const int64_t num_kv_heads = k.size(1);
  const int64_t kv_group_num = num_q_heads / num_kv_heads;
  auto starts_cpu = b_start_loc.to(torch::kLong).cpu();
  auto seq_lens_cpu = b_seq_len.to(torch::kLong).cpu();
  auto cu_ctx_lens_cpu = cu_ctx_lens.to(torch::kLong).cpu();
  auto long_options =
      torch::TensorOptions().dtype(torch::kLong).device(q.device());

  for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
    const int64_t seq_start = starts_cpu[batch_idx].item<int64_t>();
    const int64_t seq_stop = starts_cpu[batch_idx + 1].item<int64_t>();
    const int64_t query_len = seq_stop - seq_start;
    const int64_t seq_len = seq_lens_cpu[batch_idx].item<int64_t>();
    const int64_t ctx_start = cu_ctx_lens_cpu[batch_idx].item<int64_t>();
    const int64_t ctx_stop = cu_ctx_lens_cpu[batch_idx + 1].item<int64_t>();
    const int64_t ctx_len = ctx_stop - ctx_start;

    TORCH_CHECK(query_len >= 0 && ctx_len >= 0,
                "prefix_prefill_attention_precompiled expects non-negative "
                "sequence lengths");
    TORCH_CHECK(seq_len == ctx_len + query_len,
                "prefix_prefill_attention_precompiled expects seq_len to "
                "equal ctx_len + query_len");
    if (query_len <= 0) {
      continue;
    }
    if (skip_decode && query_len == 1) {
      continue;
    }

    auto ctx_k = gathered_ctx_k.slice(0, ctx_start, ctx_stop).to(torch::kFloat32);
    auto ctx_v = gathered_ctx_v.slice(0, ctx_start, ctx_stop).to(torch::kFloat32);
    auto q_seq = q.slice(0, seq_start, seq_stop).to(torch::kFloat32);
    auto k_seq = k.slice(0, seq_start, seq_stop).to(torch::kFloat32);
    auto v_seq = v.slice(0, seq_start, seq_stop).to(torch::kFloat32);

    auto all_k = torch::cat({ctx_k, k_seq}, 0);
    auto all_v = torch::cat({ctx_v, v_seq}, 0);
    auto q_heads = q_seq.permute({1, 0, 2}).contiguous();
    auto k_heads = all_k.permute({1, 0, 2}).contiguous();
    auto v_heads = all_v.permute({1, 0, 2}).contiguous();
    if (kv_group_num > 1) {
      k_heads = k_heads.repeat_interleave(kv_group_num, 0);
      v_heads = v_heads.repeat_interleave(kv_group_num, 0);
    }

    auto query_positions =
        torch::arange(ctx_len, ctx_len + query_len, long_options);
    auto key_positions = torch::arange(0, ctx_len + query_len, long_options);
    auto mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1);
    if (sliding_window > 0) {
      mask = mask & ((query_positions.unsqueeze(1) - key_positions.unsqueeze(0)) <
                     sliding_window);
    }

    std::optional<at::Tensor> attn_mask(mask.unsqueeze(0).unsqueeze(0));
    auto out = at::scaled_dot_product_attention(
                   q_heads.unsqueeze(0), k_heads.unsqueeze(0),
                   v_heads.unsqueeze(0), attn_mask, 0.0, false,
                   std::optional<double>(sm_scale), false)
                   .squeeze(0);
    output.slice(0, seq_start, seq_stop)
        .copy_(out.permute({1, 0, 2}).to(output.scalar_type()));
  }
}

torch::Tensor pack_seq_precompiled(const torch::Tensor& x,
                                   const torch::Tensor& lengths,
                                   double pad_value) {
  TORCH_CHECK(x.is_cuda(), "pack_seq_precompiled expects CUDA x");
  TORCH_CHECK(lengths.is_cuda(), "pack_seq_precompiled expects CUDA lengths");
  TORCH_CHECK(x.dim() == 2, "pack_seq_precompiled expects x with shape [N, D]");
  TORCH_CHECK(lengths.dim() == 1,
              "pack_seq_precompiled expects 1D sequence lengths");

  auto lengths_i64 = lengths.to(torch::kLong).contiguous();
  auto lengths_cpu = lengths_i64.cpu();
  const int64_t batch = lengths_cpu.numel();
  const int64_t feature_dim = x.size(1);
  const int64_t max_len =
      batch > 0 ? lengths_cpu.max().item<int64_t>() : 0;
  const int64_t total_tokens =
      batch > 0 ? lengths_cpu.sum().item<int64_t>() : 0;

  TORCH_CHECK(total_tokens == x.size(0),
              "pack_seq_precompiled expects sum(lengths) to equal x.size(0)");

  auto out = torch::full({batch, max_len, feature_dim}, pad_value, x.options());
  int64_t start = 0;
  for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
    const int64_t seq_len = lengths_cpu[batch_idx].item<int64_t>();
    TORCH_CHECK(seq_len >= 0,
                "pack_seq_precompiled expects non-negative lengths");
    if (seq_len > 0) {
      out[batch_idx].slice(0, 0, seq_len)
          .copy_(x.slice(0, start, start + seq_len));
    }
    start += seq_len;
  }
  return out;
}

torch::Tensor unpack_seq_precompiled(const torch::Tensor& packed_tensor,
                                     const torch::Tensor& lengths) {
  TORCH_CHECK(packed_tensor.is_cuda(),
              "unpack_seq_precompiled expects CUDA packed_tensor");
  TORCH_CHECK(lengths.is_cuda(),
              "unpack_seq_precompiled expects CUDA lengths");
  TORCH_CHECK(packed_tensor.dim() == 3,
              "unpack_seq_precompiled expects packed_tensor with shape "
              "[B, Lmax, D]");
  TORCH_CHECK(lengths.dim() == 1,
              "unpack_seq_precompiled expects 1D sequence lengths");
  TORCH_CHECK(packed_tensor.size(0) == lengths.size(0),
              "unpack_seq_precompiled expects packed batch size to match "
              "lengths");

  auto lengths_i64 = lengths.to(torch::kLong).contiguous();
  auto lengths_cpu = lengths_i64.cpu();
  const int64_t batch = lengths_cpu.numel();
  const int64_t max_len = packed_tensor.size(1);
  const int64_t feature_dim = packed_tensor.size(2);
  const int64_t total_tokens =
      batch > 0 ? lengths_cpu.sum().item<int64_t>() : 0;

  auto out = torch::empty({total_tokens, feature_dim}, packed_tensor.options());
  int64_t start = 0;
  for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
    const int64_t seq_len = lengths_cpu[batch_idx].item<int64_t>();
    TORCH_CHECK(seq_len >= 0,
                "unpack_seq_precompiled expects non-negative lengths");
    TORCH_CHECK(seq_len <= max_len,
                "unpack_seq_precompiled expects each length <= packed "
                "sequence max length");
    if (seq_len > 0) {
      out.slice(0, start, start + seq_len)
          .copy_(packed_tensor[batch_idx].slice(0, 0, seq_len));
    }
    start += seq_len;
  }
  return out;
}

torch::Tensor expand_batch_to_tokens_precompiled(
    const torch::Tensor& x, const torch::Tensor& cu_num_tokens,
    int64_t replace_from, int64_t replace_to) {
  TORCH_CHECK(x.is_cuda(), "expand_batch_to_tokens_precompiled expects CUDA x");
  TORCH_CHECK(cu_num_tokens.is_cuda(),
              "expand_batch_to_tokens_precompiled expects CUDA cu_num_tokens");
  TORCH_CHECK(x.dim() == 1,
              "expand_batch_to_tokens_precompiled expects 1D x");
  TORCH_CHECK(cu_num_tokens.dim() == 1,
              "expand_batch_to_tokens_precompiled expects 1D cu_num_tokens");
  TORCH_CHECK(x.size(0) == cu_num_tokens.size(0),
              "expand_batch_to_tokens_precompiled expects matching batch size");

  auto cu_num_tokens_i64 = cu_num_tokens.to(torch::kLong);
  auto counts = torch::empty_like(cu_num_tokens_i64);
  if (cu_num_tokens_i64.numel() > 0) {
    counts.slice(0, 0, 1).copy_(cu_num_tokens_i64.slice(0, 0, 1));
  }
  if (cu_num_tokens_i64.numel() > 1) {
    counts.slice(0, 1, cu_num_tokens_i64.size(0))
        .copy_(cu_num_tokens_i64.slice(0, 1, cu_num_tokens_i64.size(0)) -
               cu_num_tokens_i64.slice(0, 0, cu_num_tokens_i64.size(0) - 1));
  }

  auto expanded_x = x.repeat_interleave(counts);
  if (replace_from != replace_to) {
    auto replacement = torch::full({}, replace_to, expanded_x.options());
    expanded_x = torch::where(expanded_x == replace_from, replacement, expanded_x);
  }
  return expanded_x;
}

torch::Tensor sample_recovered_tokens_precompiled(
    const torch::Tensor& cu_num_draft_tokens,
    const torch::Tensor& draft_token_ids,
    const std::optional<torch::Tensor>& draft_probs,
    const torch::Tensor& target_probs, const torch::Tensor& inv_q) {
  TORCH_CHECK(cu_num_draft_tokens.is_cuda(),
              "sample_recovered_tokens_precompiled expects CUDA "
              "cu_num_draft_tokens");
  TORCH_CHECK(draft_token_ids.is_cuda(),
              "sample_recovered_tokens_precompiled expects CUDA "
              "draft_token_ids");
  TORCH_CHECK(target_probs.is_cuda(),
              "sample_recovered_tokens_precompiled expects CUDA target_probs");
  TORCH_CHECK(inv_q.is_cuda(),
              "sample_recovered_tokens_precompiled expects CUDA inv_q");
  TORCH_CHECK(cu_num_draft_tokens.dim() == 1,
              "sample_recovered_tokens_precompiled expects 1D "
              "cu_num_draft_tokens");
  TORCH_CHECK(draft_token_ids.dim() == 1,
              "sample_recovered_tokens_precompiled expects 1D draft_token_ids");
  TORCH_CHECK(target_probs.dim() == 2,
              "sample_recovered_tokens_precompiled expects 2D target_probs");
  TORCH_CHECK(inv_q.dim() == 2,
              "sample_recovered_tokens_precompiled expects 2D inv_q");
  TORCH_CHECK(target_probs.size(0) == draft_token_ids.size(0),
              "sample_recovered_tokens_precompiled expects target_probs rows "
              "to match draft_token_ids");
  TORCH_CHECK(inv_q.size(0) == cu_num_draft_tokens.size(0),
              "sample_recovered_tokens_precompiled expects inv_q rows to "
              "match batch size");

  auto counts = torch::empty_like(cu_num_draft_tokens.to(torch::kLong));
  auto cu_num_draft_tokens_i64 = cu_num_draft_tokens.to(torch::kLong);
  if (cu_num_draft_tokens_i64.numel() > 0) {
    counts.slice(0, 0, 1).copy_(cu_num_draft_tokens_i64.slice(0, 0, 1));
  }
  if (cu_num_draft_tokens_i64.numel() > 1) {
    counts.slice(0, 1, cu_num_draft_tokens_i64.size(0))
        .copy_(cu_num_draft_tokens_i64.slice(0, 1, cu_num_draft_tokens_i64.size(0)) -
               cu_num_draft_tokens_i64.slice(0, 0,
                                             cu_num_draft_tokens_i64.size(0) - 1));
  }

  auto req_ids =
      torch::arange(cu_num_draft_tokens_i64.size(0), counts.options());
  auto req_indices = req_ids.repeat_interleave(counts);
  auto expanded_inv_q = inv_q.index_select(0, req_indices);

  torch::Tensor scores;
  if (draft_probs.has_value() && draft_probs.value().defined()) {
    TORCH_CHECK(draft_probs.value().is_cuda(),
                "sample_recovered_tokens_precompiled expects CUDA draft_probs");
    TORCH_CHECK(draft_probs.value().sizes() == target_probs.sizes(),
                "sample_recovered_tokens_precompiled expects draft_probs to "
                "match target_probs");
    scores = torch::clamp_min(target_probs - draft_probs.value(), 0.0);
  } else {
    scores = target_probs.clone();
    auto draft_ids_long = draft_token_ids.to(torch::kLong).unsqueeze(1);
    auto zeros = torch::zeros({scores.size(0), 1}, scores.options());
    scores = scores.scatter(1, draft_ids_long, zeros);
  }

  auto weighted_scores = scores * expanded_inv_q;
  return std::get<1>(weighted_scores.max(1)).to(draft_token_ids.scalar_type());
}

void apply_top_k_top_p_precompiled(
    torch::Tensor& logits, const std::optional<torch::Tensor>& k,
    const std::optional<torch::Tensor>& p, double mask_value) {
  TORCH_CHECK(logits.is_cuda(), "apply_top_k_top_p_precompiled expects CUDA logits");
  TORCH_CHECK(logits.dim() == 2,
              "apply_top_k_top_p_precompiled expects 2D logits");

  if (!k.has_value() && !p.has_value()) {
    return;
  }

  if (k.has_value() && !p.has_value()) {
    auto resolved_k =
        normalize_rowwise_param(k.value(), logits, torch::kLong)
            .clamp(1, logits.size(1));
    if (try_apply_top_k_only_with_cuda_topk_per_row(logits, resolved_k,
                                                    mask_value)) {
      return;
    }

    auto no_top_k_mask = resolved_k == logits.size(1);
    if (no_top_k_mask.all().item<bool>()) {
      return;
    }
    auto effective_k = resolved_k.masked_fill(no_top_k_mask, 1);
    const int64_t max_k = effective_k.max().item<int64_t>();
    auto topk_values = std::get<0>(torch::topk(logits, max_k, 1, true, true));
    auto k_index = effective_k.sub(1).unsqueeze(1);
    auto top_k_threshold = topk_values.gather(1, k_index);
    top_k_threshold.masked_fill_(no_top_k_mask.unsqueeze(1), mask_value);
    logits.masked_fill_(logits < top_k_threshold, mask_value);
    return;
  }

  if (k.has_value() && p.has_value()) {
    auto resolved_k =
        normalize_rowwise_param(k.value(), logits, torch::kLong)
            .clamp(1, logits.size(1));
    auto resolved_p =
        normalize_rowwise_param(p.value(), logits, torch::kFloat32)
            .clamp(0.0, 1.0);
    auto small_k_mask = resolved_k < logits.size(1);
    auto full_k_mask = resolved_k == logits.size(1);

    if (small_k_mask.all().item<bool>()) {
      apply_top_k_then_top_p_small_k_precompiled(logits, resolved_k,
                                                 resolved_p, mask_value);
      return;
    }

    if (small_k_mask.any().item<bool>()) {
      auto small_rows = torch::nonzero(small_k_mask).view(-1);
      auto small_logits = logits.index_select(0, small_rows);
      auto small_k = resolved_k.index_select(0, small_rows);
      auto small_p = resolved_p.index_select(0, small_rows);
      apply_top_k_then_top_p_small_k_precompiled(small_logits, small_k, small_p,
                                                 mask_value);
      logits.index_copy_(0, small_rows, small_logits);
    }

    if (full_k_mask.any().item<bool>()) {
      auto full_rows = torch::nonzero(full_k_mask).view(-1);
      auto full_logits = logits.index_select(0, full_rows);
      auto full_p = resolved_p.index_select(0, full_rows);
      apply_top_p_only_iterative_topk_precompiled(full_logits, full_p,
                                                  mask_value);
      logits.index_copy_(0, full_rows, full_logits);
      return;
    }
  }

  if (!k.has_value() && p.has_value()) {
    auto resolved_p =
        normalize_rowwise_param(p.value(), logits, torch::kFloat32)
            .clamp(0.0, 1.0);
    apply_top_p_only_iterative_topk_precompiled(logits, resolved_p, mask_value);
    return;
  }

  torch::Tensor logits_sort;
  torch::Tensor logits_idx;
  std::tie(logits_sort, logits_idx) = torch::sort(logits, -1, false);

  if (k.has_value()) {
    auto resolved_k =
        normalize_rowwise_param(k.value(), logits, torch::kLong)
            .clamp(1, logits_sort.size(1));
    auto top_k_index =
        (logits_sort.size(1) - resolved_k).unsqueeze(1);
    auto top_k_threshold = logits_sort.gather(1, top_k_index);
    auto top_k_mask = logits_sort < top_k_threshold;
    logits_sort.masked_fill_(top_k_mask, mask_value);
  }

  if (p.has_value()) {
    auto resolved_p =
        normalize_rowwise_param(p.value(), logits, torch::kFloat32)
            .clamp(0.0, 1.0);
    auto probs_sort = logits_sort.softmax(-1);
    auto probs_cumsum = torch::cumsum(probs_sort, -1);
    auto top_p_mask = probs_cumsum <= (1 - resolved_p.unsqueeze(1));
    top_p_mask.select(1, top_p_mask.size(1) - 1).fill_(false);
    logits_sort.masked_fill_(top_p_mask, mask_value);
  }

  logits.scatter_(1, logits_idx, logits_sort);
}

void input_batch_prepare_prefill_inputs_precompiled(
    torch::Tensor& input_ids, torch::Tensor& next_prefill_tokens,
    const torch::Tensor& idx_mapping, const torch::Tensor& query_start_loc,
    const torch::Tensor& all_token_ids, const torch::Tensor& prefill_len,
    const torch::Tensor& num_computed_tokens) {
  TORCH_CHECK(input_ids.is_cuda(),
              "input_batch_prepare_prefill_inputs_precompiled expects CUDA "
              "input_ids");
  TORCH_CHECK(next_prefill_tokens.is_cuda(),
              "input_batch_prepare_prefill_inputs_precompiled expects CUDA "
              "next_prefill_tokens");
  TORCH_CHECK(idx_mapping.is_cuda(),
              "input_batch_prepare_prefill_inputs_precompiled expects CUDA "
              "idx_mapping");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "input_batch_prepare_prefill_inputs_precompiled expects CUDA "
              "query_start_loc");
  TORCH_CHECK(all_token_ids.is_cuda(),
              "input_batch_prepare_prefill_inputs_precompiled expects CUDA "
              "all_token_ids");
  TORCH_CHECK(prefill_len.is_cuda(),
              "input_batch_prepare_prefill_inputs_precompiled expects CUDA "
              "prefill_len");
  TORCH_CHECK(num_computed_tokens.is_cuda(),
              "input_batch_prepare_prefill_inputs_precompiled expects CUDA "
              "num_computed_tokens");

  auto idx_cpu = idx_mapping.to(torch::kCPU);
  auto query_start_cpu = query_start_loc.to(torch::kCPU);
  auto prefill_len_cpu = prefill_len.to(torch::kCPU);
  auto computed_cpu = num_computed_tokens.to(torch::kCPU);

  const int64_t num_reqs = idx_mapping.size(0);
  for (int64_t batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
    const int64_t req_state_idx = idx_cpu[batch_idx].item<int64_t>();
    const int64_t prefill =
        prefill_len_cpu[req_state_idx].item<int64_t>();
    const int64_t num_computed =
        computed_cpu[req_state_idx].item<int64_t>();
    if (num_computed >= prefill) {
      continue;
    }

    const int64_t query_start =
        query_start_cpu[batch_idx].item<int64_t>();
    const int64_t query_end =
        query_start_cpu[batch_idx + 1].item<int64_t>();
    const int64_t query_len = query_end - query_start;
    if (query_len > 0) {
      input_ids.slice(0, query_start, query_end)
          .copy_(all_token_ids.select(0, req_state_idx)
                     .slice(0, num_computed, num_computed + query_len));
    }

    const int64_t next_pos = num_computed + query_len;
    if (next_pos < prefill) {
      next_prefill_tokens.select(0, req_state_idx)
          .copy_(all_token_ids.select(0, req_state_idx).select(0, next_pos));
    }
  }
}

void input_batch_prepare_pos_seq_lens_precompiled(
    const torch::Tensor& idx_mapping, const torch::Tensor& query_start_loc,
    const torch::Tensor& num_computed_tokens, torch::Tensor& pos,
    torch::Tensor& seq_lens) {
  TORCH_CHECK(idx_mapping.is_cuda(),
              "input_batch_prepare_pos_seq_lens_precompiled expects CUDA "
              "idx_mapping");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "input_batch_prepare_pos_seq_lens_precompiled expects CUDA "
              "query_start_loc");
  TORCH_CHECK(num_computed_tokens.is_cuda(),
              "input_batch_prepare_pos_seq_lens_precompiled expects CUDA "
              "num_computed_tokens");
  TORCH_CHECK(pos.is_cuda(),
              "input_batch_prepare_pos_seq_lens_precompiled expects CUDA pos");
  TORCH_CHECK(seq_lens.is_cuda(),
              "input_batch_prepare_pos_seq_lens_precompiled expects CUDA "
              "seq_lens");

  auto idx_cpu = idx_mapping.to(torch::kCPU);
  auto query_start_cpu = query_start_loc.to(torch::kCPU);
  auto computed_cpu = num_computed_tokens.to(torch::kCPU);

  const int64_t num_reqs = idx_mapping.size(0);
  if (seq_lens.size(0) > num_reqs) {
    seq_lens.slice(0, num_reqs, seq_lens.size(0)).zero_();
  }

  for (int64_t req_id = 0; req_id < num_reqs; ++req_id) {
    const int64_t req_state_idx = idx_cpu[req_id].item<int64_t>();
    const int64_t num_computed =
        computed_cpu[req_state_idx].item<int64_t>();
    const int64_t start = query_start_cpu[req_id].item<int64_t>();
    const int64_t end = query_start_cpu[req_id + 1].item<int64_t>();
    const int64_t query_len = end - start;

    seq_lens.select(0, req_id).fill_(num_computed + query_len);
    if (query_len > 0) {
      pos.slice(0, start, end)
          .copy_(torch::arange(num_computed, num_computed + query_len,
                               pos.options()));
    }
  }
}

torch::Tensor input_batch_combine_sampled_and_draft_tokens_precompiled(
    torch::Tensor& input_ids, const torch::Tensor& idx_mapping,
    const torch::Tensor& last_sampled_tokens,
    const torch::Tensor& query_start_loc, const torch::Tensor& seq_lens,
    const torch::Tensor& prefill_len, const torch::Tensor& draft_tokens,
    const torch::Tensor& cu_num_logits, int64_t num_logits) {
  TORCH_CHECK(input_ids.is_cuda(),
              "input_batch_combine_sampled_and_draft_tokens_precompiled "
              "expects CUDA input_ids");
  TORCH_CHECK(idx_mapping.is_cuda(),
              "input_batch_combine_sampled_and_draft_tokens_precompiled "
              "expects CUDA idx_mapping");
  TORCH_CHECK(last_sampled_tokens.is_cuda(),
              "input_batch_combine_sampled_and_draft_tokens_precompiled "
              "expects CUDA last_sampled_tokens");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "input_batch_combine_sampled_and_draft_tokens_precompiled "
              "expects CUDA query_start_loc");
  TORCH_CHECK(seq_lens.is_cuda(),
              "input_batch_combine_sampled_and_draft_tokens_precompiled "
              "expects CUDA seq_lens");
  TORCH_CHECK(prefill_len.is_cuda(),
              "input_batch_combine_sampled_and_draft_tokens_precompiled "
              "expects CUDA prefill_len");
  TORCH_CHECK(draft_tokens.is_cuda(),
              "input_batch_combine_sampled_and_draft_tokens_precompiled "
              "expects CUDA draft_tokens");
  TORCH_CHECK(cu_num_logits.is_cuda(),
              "input_batch_combine_sampled_and_draft_tokens_precompiled "
              "expects CUDA cu_num_logits");

  auto logits_indices =
      torch::empty({num_logits},
                   torch::TensorOptions().device(input_ids.device()).dtype(
                       torch::kInt64));
  auto idx_cpu = idx_mapping.to(torch::kCPU);
  auto query_start_cpu = query_start_loc.to(torch::kCPU);
  auto seq_lens_cpu = seq_lens.to(torch::kCPU);
  auto prefill_len_cpu = prefill_len.to(torch::kCPU);
  auto cu_num_logits_cpu = cu_num_logits.to(torch::kCPU);

  const int64_t num_reqs = idx_mapping.size(0);
  for (int64_t batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
    const int64_t req_state_idx = idx_cpu[batch_idx].item<int64_t>();
    const int64_t logits_start =
        cu_num_logits_cpu[batch_idx].item<int64_t>();
    const int64_t logits_end =
        cu_num_logits_cpu[batch_idx + 1].item<int64_t>();
    const int64_t req_num_logits = logits_end - logits_start;
    const int64_t query_end =
        query_start_cpu[batch_idx + 1].item<int64_t>();
    const int64_t input_start = query_end - req_num_logits;

    if (req_num_logits > 0) {
      logits_indices.slice(0, logits_start, logits_end)
          .copy_(torch::arange(input_start, input_start + req_num_logits,
                               logits_indices.options()));
    }

    const int64_t seq_len = seq_lens_cpu[batch_idx].item<int64_t>();
    const int64_t prefill =
        prefill_len_cpu[req_state_idx].item<int64_t>();
    if (seq_len <= prefill || req_num_logits == 0) {
      continue;
    }

    input_ids.select(0, input_start)
        .copy_(last_sampled_tokens.select(0, req_state_idx));

    const int64_t num_draft_tokens = req_num_logits - 1;
    if (num_draft_tokens > 0) {
      input_ids.slice(0, query_end - num_draft_tokens, query_end)
          .copy_(draft_tokens.select(0, req_state_idx)
                     .slice(0, 0, num_draft_tokens));
    }
  }
  return logits_indices;
}

std::tuple<torch::Tensor, torch::Tensor>
input_batch_get_num_sampled_and_rejected_precompiled(
    torch::Tensor& num_sampled, const torch::Tensor& seq_lens,
    const torch::Tensor& cu_num_logits, const torch::Tensor& idx_mapping,
    const torch::Tensor& prefill_len) {
  TORCH_CHECK(num_sampled.is_cuda(),
              "input_batch_get_num_sampled_and_rejected_precompiled expects "
              "CUDA num_sampled");
  TORCH_CHECK(seq_lens.is_cuda(),
              "input_batch_get_num_sampled_and_rejected_precompiled expects "
              "CUDA seq_lens");
  TORCH_CHECK(cu_num_logits.is_cuda(),
              "input_batch_get_num_sampled_and_rejected_precompiled expects "
              "CUDA cu_num_logits");
  TORCH_CHECK(idx_mapping.is_cuda(),
              "input_batch_get_num_sampled_and_rejected_precompiled expects "
              "CUDA idx_mapping");
  TORCH_CHECK(prefill_len.is_cuda(),
              "input_batch_get_num_sampled_and_rejected_precompiled expects "
              "CUDA prefill_len");

  auto num_rejected = torch::empty_like(num_sampled);
  auto num_sampled_cpu = num_sampled.to(torch::kCPU);
  auto seq_lens_cpu = seq_lens.to(torch::kCPU);
  auto cu_num_logits_cpu = cu_num_logits.to(torch::kCPU);
  auto idx_cpu = idx_mapping.to(torch::kCPU);
  auto prefill_len_cpu = prefill_len.to(torch::kCPU);

  const int64_t num_reqs = idx_mapping.size(0);
  for (int64_t batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
    const int64_t req_state_idx = idx_cpu[batch_idx].item<int64_t>();
    const int64_t seq_len = seq_lens_cpu[batch_idx].item<int64_t>();
    const int64_t prefill =
        prefill_len_cpu[req_state_idx].item<int64_t>();
    const bool is_chunked_prefilling = seq_len < prefill;
    int64_t sampled = num_sampled_cpu[batch_idx].item<int64_t>();
    if (is_chunked_prefilling) {
      sampled = 0;
      num_sampled.select(0, batch_idx).fill_(0);
    }

    const int64_t logits_start =
        cu_num_logits_cpu[batch_idx].item<int64_t>();
    const int64_t logits_end =
        cu_num_logits_cpu[batch_idx + 1].item<int64_t>();
    const int64_t rejected =
        is_chunked_prefilling ? 0 : (logits_end - logits_start - sampled);
    num_rejected.select(0, batch_idx).fill_(rejected);
  }
  return std::make_tuple(num_sampled, num_rejected);
}

void input_batch_post_update_precompiled(
    const torch::Tensor& idx_mapping, torch::Tensor& num_computed_tokens,
    torch::Tensor& last_sampled_tokens,
    const std::optional<torch::Tensor>& output_bin_counts,
    const torch::Tensor& sampled_tokens, const torch::Tensor& num_sampled,
    const torch::Tensor& num_rejected, const torch::Tensor& query_start_loc,
    torch::Tensor& all_token_ids, torch::Tensor& total_len) {
  TORCH_CHECK(idx_mapping.is_cuda(),
              "input_batch_post_update_precompiled expects CUDA idx_mapping");
  TORCH_CHECK(num_computed_tokens.is_cuda(),
              "input_batch_post_update_precompiled expects CUDA "
              "num_computed_tokens");
  TORCH_CHECK(last_sampled_tokens.is_cuda(),
              "input_batch_post_update_precompiled expects CUDA "
              "last_sampled_tokens");
  TORCH_CHECK(sampled_tokens.is_cuda(),
              "input_batch_post_update_precompiled expects CUDA "
              "sampled_tokens");
  TORCH_CHECK(num_sampled.is_cuda(),
              "input_batch_post_update_precompiled expects CUDA num_sampled");
  TORCH_CHECK(num_rejected.is_cuda(),
              "input_batch_post_update_precompiled expects CUDA num_rejected");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "input_batch_post_update_precompiled expects CUDA "
              "query_start_loc");
  TORCH_CHECK(all_token_ids.is_cuda(),
              "input_batch_post_update_precompiled expects CUDA all_token_ids");
  TORCH_CHECK(total_len.is_cuda(),
              "input_batch_post_update_precompiled expects CUDA total_len");
  if (output_bin_counts.has_value() && output_bin_counts.value().defined()) {
    TORCH_CHECK(output_bin_counts.value().is_cuda(),
                "input_batch_post_update_precompiled expects CUDA "
                "output_bin_counts");
  }

  auto idx_cpu = idx_mapping.to(torch::kCPU);
  auto num_sampled_cpu = num_sampled.to(torch::kCPU);
  auto num_rejected_cpu = num_rejected.to(torch::kCPU);
  auto query_start_cpu = query_start_loc.to(torch::kCPU);
  auto total_len_cpu = total_len.to(torch::kCPU);

  const int64_t num_reqs = idx_mapping.size(0);
  for (int64_t req_id = 0; req_id < num_reqs; ++req_id) {
    const int64_t req_state_idx = idx_cpu[req_id].item<int64_t>();
    const int64_t old_total_len = total_len_cpu[req_state_idx].item<int64_t>();
    const int64_t sampled = num_sampled_cpu[req_id].item<int64_t>();
    if (sampled > 0) {
      last_sampled_tokens.select(0, req_state_idx)
          .copy_(sampled_tokens.select(0, req_id).select(0, sampled - 1));
      total_len.select(0, req_state_idx).fill_(old_total_len + sampled);
    }

    for (int64_t i = 0; i < sampled; ++i) {
      auto token = sampled_tokens.select(0, req_id).select(0, i);
      all_token_ids.select(0, req_state_idx)
          .select(0, old_total_len + i)
          .copy_(token);

      if (output_bin_counts.has_value() &&
          output_bin_counts.value().defined()) {
        const int64_t token_id = token.item<int64_t>();
        output_bin_counts.value()
            .select(0, req_state_idx)
            .select(0, token_id)
            .add_(1);
      }
    }

    const int64_t query_start = query_start_cpu[req_id].item<int64_t>();
    const int64_t query_end = query_start_cpu[req_id + 1].item<int64_t>();
    const int64_t query_len = query_end - query_start;
    const int64_t rejected = num_rejected_cpu[req_id].item<int64_t>();
    num_computed_tokens.select(0, req_state_idx)
        .add_(query_len - rejected);
  }
}

void input_batch_post_update_pool_precompiled(
    const torch::Tensor& idx_mapping, torch::Tensor& num_computed_tokens,
    const torch::Tensor& query_start_loc) {
  TORCH_CHECK(idx_mapping.is_cuda(),
              "input_batch_post_update_pool_precompiled expects CUDA "
              "idx_mapping");
  TORCH_CHECK(num_computed_tokens.is_cuda(),
              "input_batch_post_update_pool_precompiled expects CUDA "
              "num_computed_tokens");
  TORCH_CHECK(query_start_loc.is_cuda(),
              "input_batch_post_update_pool_precompiled expects CUDA "
              "query_start_loc");

  auto idx_cpu = idx_mapping.to(torch::kCPU);
  auto query_start_cpu = query_start_loc.to(torch::kCPU);
  const int64_t num_reqs = idx_mapping.size(0);
  for (int64_t batch_id = 0; batch_id < num_reqs; ++batch_id) {
    const int64_t req_state_idx = idx_cpu[batch_id].item<int64_t>();
    const int64_t query_start = query_start_cpu[batch_id].item<int64_t>();
    const int64_t query_end = query_start_cpu[batch_id + 1].item<int64_t>();
    num_computed_tokens.select(0, req_state_idx)
        .add_(query_end - query_start);
  }
}

std::tuple<torch::Tensor, torch::Tensor>
input_batch_expand_idx_mapping_precompiled(const torch::Tensor& idx_mapping,
                                           int64_t total_num_logits,
                                           const torch::Tensor& cu_num_logits) {
  TORCH_CHECK(idx_mapping.is_cuda(),
              "input_batch_expand_idx_mapping_precompiled expects CUDA "
              "idx_mapping");
  TORCH_CHECK(cu_num_logits.is_cuda(),
              "input_batch_expand_idx_mapping_precompiled expects CUDA "
              "cu_num_logits");

  auto expanded_idx_mapping =
      torch::empty({total_num_logits}, idx_mapping.options());
  auto expanded_local_pos =
      torch::empty({total_num_logits},
                   torch::TensorOptions().device(idx_mapping.device()).dtype(
                       torch::kInt32));

  auto idx_cpu = idx_mapping.to(torch::kCPU);
  auto cu_num_logits_cpu = cu_num_logits.to(torch::kCPU);
  const int64_t num_reqs = idx_mapping.size(0);
  for (int64_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
    const int64_t start = cu_num_logits_cpu[req_idx].item<int64_t>();
    const int64_t end = cu_num_logits_cpu[req_idx + 1].item<int64_t>();
    const int64_t num_tokens = end - start;
    if (num_tokens <= 0) {
      continue;
    }
    expanded_idx_mapping.slice(0, start, end)
        .fill_(idx_cpu[req_idx].item<int64_t>());
    expanded_local_pos.slice(0, start, end)
        .copy_(torch::arange(0, num_tokens, expanded_local_pos.options()));
  }

  return std::make_tuple(expanded_idx_mapping, expanded_local_pos);
}

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

std::tuple<torch::Tensor, torch::Tensor>
fused_recurrent_gated_delta_rule_packed_decode_precompiled(
    const torch::Tensor& mixed_qkv, const torch::Tensor& a,
    const torch::Tensor& b, const torch::Tensor& A_log,
    const torch::Tensor& dt_bias, double scale, torch::Tensor& initial_state,
    torch::Tensor& out, const torch::Tensor& ssm_state_indices,
    bool use_qk_l2norm_in_kernel) {
  TORCH_CHECK(mixed_qkv.is_cuda() && a.is_cuda() && b.is_cuda() &&
                  A_log.is_cuda() && dt_bias.is_cuda() &&
                  initial_state.is_cuda() && out.is_cuda() &&
                  ssm_state_indices.is_cuda(),
              "fused_recurrent_gated_delta_rule_packed_decode_precompiled "
              "expects CUDA tensors");
  TORCH_CHECK(mixed_qkv.dim() == 2,
              "mixed_qkv must be 2D for packed decode");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
              "a and b must be 2D for packed decode");
  TORCH_CHECK(A_log.dim() == 1 && dt_bias.dim() == 1,
              "A_log and dt_bias must be 1D for packed decode");
  TORCH_CHECK(initial_state.dim() == 4,
              "initial_state must be 4D for packed decode");
  TORCH_CHECK(out.dim() == 4 && out.size(1) == 1,
              "out must have shape [B, 1, HV, V] for packed decode");
  TORCH_CHECK(ssm_state_indices.dim() == 1,
              "ssm_state_indices must be 1D for packed decode");

  const int64_t B = mixed_qkv.size(0);
  const int64_t HV = initial_state.size(1);
  const int64_t V = initial_state.size(2);
  const int64_t K = initial_state.size(3);

  TORCH_CHECK(a.size(0) == B && b.size(0) == B,
              "a and b batch size must match mixed_qkv");
  TORCH_CHECK(a.size(1) == HV && b.size(1) == HV,
              "a and b hidden dimension must match initial_state");
  TORCH_CHECK(A_log.numel() == HV && dt_bias.numel() == HV,
              "A_log and dt_bias must have HV elements");
  TORCH_CHECK(ssm_state_indices.numel() == B,
              "ssm_state_indices must have B elements");
  TORCH_CHECK(out.size(0) == B && out.size(2) == HV && out.size(3) == V,
              "out shape must match [B, 1, HV, V]");

  const int64_t qkv_dim = mixed_qkv.size(1);
  const int64_t qk_dim = qkv_dim - HV * V;
  TORCH_CHECK(qk_dim > 0 && qk_dim % 2 == 0,
              "Invalid packed mixed_qkv size for packed decode");
  const int64_t q_dim = qk_dim / 2;
  TORCH_CHECK(q_dim % K == 0,
              "Packed decode q_dim must be divisible by K");
  const int64_t H = q_dim / K;
  TORCH_CHECK(H > 0 && HV % H == 0,
              "Packed decode inferred invalid head configuration");
  const int64_t hv_per_h = HV / H;

  auto index_options =
      torch::TensorOptions().dtype(torch::kLong).device(mixed_qkv.device());

  auto safe_indices =
      ssm_state_indices.to(mixed_qkv.device(), torch::kLong, false, false)
          .contiguous();
  auto valid_mask = safe_indices >= 0;
  auto gather_indices = torch::clamp_min(safe_indices, 0);

  auto mixed_qkv_f = mixed_qkv.to(torch::kFloat32);
  auto packed_q = mixed_qkv_f.slice(1, 0, q_dim).view({B, H, K});
  auto packed_k = mixed_qkv_f.slice(1, q_dim, 2 * q_dim).view({B, H, K});
  auto packed_v = mixed_qkv_f.slice(1, 2 * q_dim, qkv_dim).view({B, HV, V});

  std::vector<int64_t> h_index_vec(HV);
  for (int64_t idx = 0; idx < HV; ++idx) {
    h_index_vec[idx] = idx / hv_per_h;
  }
  auto h_indices = torch::tensor(h_index_vec, index_options);

  auto q = packed_q.index_select(1, h_indices);
  auto k = packed_k.index_select(1, h_indices);
  if (use_qk_l2norm_in_kernel) {
    q = l2norm_last_dim(q);
    k = l2norm_last_dim(k);
  }
  q = q * scale;

  auto h = initial_state.index_select(0, gather_indices).to(torch::kFloat32);
  auto g_input =
      a.to(torch::kFloat32) + dt_bias.to(torch::kFloat32).view({1, HV});
  auto g =
      (-torch::exp(A_log.to(torch::kFloat32)).view({1, HV, 1, 1}) *
       softplus_with_threshold(g_input, 1.0, 20.0).view({B, HV, 1, 1}));
  auto beta = torch::sigmoid(b.to(torch::kFloat32)).view({B, HV, 1});

  h = h * torch::exp(g);
  auto v = (packed_v - (h * k.unsqueeze(-2)).sum(-1)) * beta;
  auto updated_h = h + v.unsqueeze(-1) * k.unsqueeze(-2);
  auto out_values = (updated_h * q.unsqueeze(-2)).sum(-1);

  auto out_slice = out.select(1, 0);
  out_slice.copy_(torch::where(valid_mask.view({B, 1, 1}), out_values,
                               torch::zeros_like(out_values))
                      .to(out.scalar_type()));

  if (valid_mask.any().item<bool>()) {
    auto valid_rows = torch::nonzero(valid_mask).view(-1);
    auto valid_gather_indices = gather_indices.index_select(0, valid_rows);
    auto valid_updated_h =
        updated_h.to(initial_state.scalar_type()).index_select(0, valid_rows);
    initial_state.index_copy_(0, valid_gather_indices, valid_updated_h);
  }

  return {out, initial_state};
}

torch::Tensor l2norm_precompiled(
    const torch::Tensor& x, double eps,
    const std::optional<torch::ScalarType>& output_dtype) {
  TORCH_CHECK(x.is_cuda(), "l2norm_precompiled expects CUDA input");
  TORCH_CHECK(x.dim() >= 1, "l2norm_precompiled expects rank >= 1");

  const auto original_sizes = x.sizes().vec();
  auto x_contiguous = x.contiguous();
  auto x_flat = x_contiguous.view({-1, x_contiguous.size(-1)});
  auto y = l2norm_last_dim(x_flat.to(torch::kFloat32));
  auto resolved_dtype = output_dtype.value_or(x.scalar_type());
  return y.to(resolved_dtype).view(original_sizes);
}

torch::Tensor chunk_local_cumsum_precompiled(
    const torch::Tensor& g, int64_t chunk_size, bool reverse,
    const std::optional<torch::Tensor>& cu_seqlens, bool head_first,
    const std::optional<torch::ScalarType>& output_dtype) {
  TORCH_CHECK(g.is_cuda(), "chunk_local_cumsum_precompiled expects CUDA input");
  TORCH_CHECK(g.dim() == 3 || g.dim() == 4,
              "chunk_local_cumsum_precompiled expects rank 3 or 4 input");
  TORCH_CHECK(chunk_size > 0 && (chunk_size & (chunk_size - 1)) == 0,
              "chunk_size must be a power of 2");
  if (cu_seqlens.has_value()) {
    TORCH_CHECK(g.size(0) == 1,
                "Only batch size 1 is supported when cu_seqlens are "
                "provided");
    TORCH_CHECK(cu_seqlens.value().dim() == 1 && cu_seqlens.value().numel() >= 2,
                "cu_seqlens must be a 1D tensor with at least 2 elements");
  }

  auto seq_major = g.contiguous();
  if (head_first) {
    seq_major = g.dim() == 3 ? seq_major.permute({0, 2, 1}).contiguous()
                             : seq_major.permute({0, 2, 1, 3}).contiguous();
  }

  const auto resolved_dtype = output_dtype.value_or(g.scalar_type());
  auto out = torch::empty(seq_major.sizes(),
                          seq_major.options().dtype(resolved_dtype));

  auto write_sequence = [&](const torch::Tensor& src_seq,
                            torch::Tensor dst_seq) {
    const int64_t token_count = src_seq.size(0);
    if (token_count == 0) {
      return;
    }
    const int64_t flat_width = src_seq.numel() / token_count;
    auto src_flat =
        src_seq.reshape({token_count, flat_width}).to(torch::kFloat32);
    auto dst_flat = torch::empty({token_count, flat_width},
                                 src_flat.options().dtype(torch::kFloat32));

    for (int64_t chunk_start = 0; chunk_start < token_count;
         chunk_start += chunk_size) {
      const int64_t chunk_end = std::min(chunk_start + chunk_size, token_count);
      auto chunk = src_flat.slice(0, chunk_start, chunk_end);
      auto chunk_out =
          reverse ? torch::flip(torch::cumsum(torch::flip(chunk, {0}), 0), {0})
                  : torch::cumsum(chunk, 0);
      dst_flat.slice(0, chunk_start, chunk_end).copy_(chunk_out);
    }

    dst_seq.copy_(dst_flat.reshape(src_seq.sizes().vec()).to(resolved_dtype));
  };

  if (cu_seqlens.has_value()) {
    auto cu = cu_seqlens.value().to(g.device(), torch::kLong, false, false)
                  .contiguous();
    const int64_t num_sequences = cu.numel() - 1;
    for (int64_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
      const int64_t seq_start = cu[seq_idx].item<int64_t>();
      const int64_t seq_end = cu[seq_idx + 1].item<int64_t>();
      if (seq_end <= seq_start) {
        continue;
      }
      write_sequence(seq_major[0].slice(0, seq_start, seq_end),
                     out[0].slice(0, seq_start, seq_end));
    }
  } else {
    for (int64_t batch_idx = 0; batch_idx < seq_major.size(0); ++batch_idx) {
      write_sequence(seq_major[batch_idx], out[batch_idx]);
    }
  }

  if (head_first) {
    return g.dim() == 3 ? out.permute({0, 2, 1}).contiguous()
                        : out.permute({0, 2, 1, 3}).contiguous();
  }
  return out;
}

torch::Tensor chunk_fwd_o_precompiled(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& h, const std::optional<torch::Tensor>& g,
    double scale, const std::optional<torch::Tensor>& cu_seqlens,
    int64_t block_size) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && h.is_cuda(),
              "chunk_fwd_o_precompiled expects CUDA q/k/v/h");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && h.dim() == 5,
              "chunk_fwd_o_precompiled expects q/k/v rank 4 and h rank 5");
  TORCH_CHECK(q.sizes() == k.sizes(),
              "chunk_fwd_o_precompiled expects q and k to share shape");
  TORCH_CHECK(q.size(0) == v.size(0) && q.size(1) == v.size(1),
              "chunk_fwd_o_precompiled expects q/k and v to share batch/time");
  TORCH_CHECK(block_size > 0, "block_size must be positive");

  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t Hg = q.size(2);
  const int64_t K = q.size(3);
  const int64_t H = v.size(2);
  const int64_t V = v.size(3);
  TORCH_CHECK(Hg > 0 && H % Hg == 0,
              "chunk_fwd_o_precompiled expects H divisible by Hg");
  TORCH_CHECK(h.size(0) == B && h.size(2) == H && h.size(3) == V &&
                  h.size(4) == K,
              "chunk_fwd_o_precompiled expects h shape [B, NT, H, V, K]");
  if (g.has_value()) {
    TORCH_CHECK(g.value().is_cuda(),
                "chunk_fwd_o_precompiled expects CUDA g when provided");
    TORCH_CHECK(g.value().dim() == 3 && g.value().size(0) == B &&
                    g.value().size(1) == T && g.value().size(2) == H,
                "chunk_fwd_o_precompiled expects g shape [B, T, H]");
  }

  auto q_f = q.to(torch::kFloat32);
  auto k_f = k.to(torch::kFloat32);
  auto v_f = v.to(torch::kFloat32);
  auto h_f = h.to(torch::kFloat32);
  std::optional<torch::Tensor> g_f = std::nullopt;
  if (g.has_value()) {
    g_f = g.value().to(torch::kFloat32);
  }
  auto out = torch::empty_like(v);

  const int64_t heads_per_group = H / Hg;

  auto write_chunk = [&](int64_t batch_idx, int64_t chunk_h_idx,
                         int64_t start, int64_t end) {
    if (end <= start) {
      return;
    }
    auto q_chunk_all = q_f[batch_idx].slice(0, start, end);
    auto k_chunk_all = k_f[batch_idx].slice(0, start, end);
    auto v_chunk_all = v_f[batch_idx].slice(0, start, end);
    std::optional<torch::Tensor> g_chunk_all = std::nullopt;
    if (g_f.has_value()) {
      g_chunk_all = g_f.value()[batch_idx].slice(0, start, end);
    }

    for (int64_t head_idx = 0; head_idx < H; ++head_idx) {
      const int64_t group_idx = head_idx / heads_per_group;
      auto q_chunk = q_chunk_all.select(1, group_idx);
      auto k_chunk = k_chunk_all.select(1, group_idx);
      auto v_chunk = v_chunk_all.select(1, head_idx);
      auto h_chunk = h_f[batch_idx][chunk_h_idx][head_idx];

      auto out_chunk = torch::matmul(q_chunk, h_chunk.transpose(0, 1));
      auto attn_chunk = torch::matmul(q_chunk, k_chunk.transpose(0, 1));
      if (g_chunk_all.has_value()) {
        auto g_chunk = g_chunk_all.value().select(1, head_idx);
        auto g_exp = torch::exp(g_chunk).unsqueeze(-1);
        out_chunk = out_chunk * g_exp;
        attn_chunk =
            attn_chunk *
            torch::exp(g_chunk.unsqueeze(-1) - g_chunk.unsqueeze(0));
      }
      attn_chunk = torch::tril(attn_chunk);
      out_chunk = (out_chunk + torch::matmul(attn_chunk, v_chunk)) * scale;
      out[batch_idx].slice(0, start, end).select(1, head_idx).copy_(
          out_chunk.to(out.scalar_type()));
    }
  };

  if (cu_seqlens.has_value()) {
    TORCH_CHECK(B == 1,
                "Only batch size 1 is supported when cu_seqlens are provided");
    auto cu = cu_seqlens.value().to(q.device(), torch::kLong, false, false)
                  .contiguous();
    TORCH_CHECK(cu.dim() == 1 && cu.numel() >= 2,
                "cu_seqlens must be a 1D tensor with at least 2 elements");
    int64_t global_chunk_idx = 0;
    for (int64_t seq_idx = 0; seq_idx < cu.numel() - 1; ++seq_idx) {
      const int64_t seq_start = cu[seq_idx].item<int64_t>();
      const int64_t seq_end = cu[seq_idx + 1].item<int64_t>();
      const int64_t seq_len = seq_end - seq_start;
      const int64_t local_chunks =
          (seq_len + block_size - 1) / block_size;
      for (int64_t local_chunk_idx = 0; local_chunk_idx < local_chunks;
           ++local_chunk_idx) {
        const int64_t start = seq_start + local_chunk_idx * block_size;
        const int64_t end = std::min(start + block_size, seq_end);
        write_chunk(0, global_chunk_idx, start, end);
        ++global_chunk_idx;
      }
    }
    TORCH_CHECK(h.size(1) == global_chunk_idx,
                "chunk_fwd_o_precompiled got unexpected h chunk dimension for "
                "varlen input");
  } else {
    const int64_t total_chunks = (T + block_size - 1) / block_size;
    TORCH_CHECK(h.size(1) == total_chunks,
                "chunk_fwd_o_precompiled got unexpected h chunk dimension");
    for (int64_t batch_idx = 0; batch_idx < B; ++batch_idx) {
      for (int64_t chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
        const int64_t start = chunk_idx * block_size;
        const int64_t end = std::min(start + block_size, T);
        write_chunk(batch_idx, chunk_idx, start, end);
      }
    }
  }

  return out;
}

torch::Tensor chunk_scaled_dot_kkt_fwd_precompiled(
    const torch::Tensor& k, const std::optional<torch::Tensor>& g,
    const torch::Tensor& beta, const std::optional<torch::Tensor>& cu_seqlens,
    int64_t chunk_size,
    const std::optional<torch::ScalarType>& output_dtype) {
  TORCH_CHECK(k.is_cuda() && beta.is_cuda(),
              "chunk_scaled_dot_kkt_fwd_precompiled expects CUDA k/beta");
  TORCH_CHECK(k.dim() == 4 && beta.dim() == 3,
              "chunk_scaled_dot_kkt_fwd_precompiled expects k rank 4 and "
              "beta rank 3");
  TORCH_CHECK(k.size(0) == beta.size(0) && k.size(1) == beta.size(1),
              "chunk_scaled_dot_kkt_fwd_precompiled expects shared batch/time");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

  const int64_t B = k.size(0);
  const int64_t T = k.size(1);
  const int64_t Hg = k.size(2);
  const int64_t K = k.size(3);
  const int64_t H = beta.size(2);
  TORCH_CHECK(Hg > 0 && H % Hg == 0,
              "chunk_scaled_dot_kkt_fwd_precompiled expects H divisible by Hg");
  if (g.has_value()) {
    TORCH_CHECK(g.value().is_cuda(),
                "chunk_scaled_dot_kkt_fwd_precompiled expects CUDA g when "
                "provided");
    TORCH_CHECK(g.value().dim() == 3 && g.value().size(0) == B &&
                    g.value().size(1) == T && g.value().size(2) == H,
                "chunk_scaled_dot_kkt_fwd_precompiled expects g shape [B, T, H]");
  }

  const auto resolved_dtype = output_dtype.value_or(torch::kFloat32);
  auto out = torch::zeros({B, T, H, chunk_size},
                          k.options().dtype(resolved_dtype));
  auto k_f = k.to(torch::kFloat32);
  auto beta_f = beta.to(torch::kFloat32);
  std::optional<torch::Tensor> g_f = std::nullopt;
  if (g.has_value()) {
    g_f = g.value().to(torch::kFloat32);
  }

  const int64_t heads_per_group = H / Hg;

  auto write_chunk = [&](int64_t batch_idx, int64_t start, int64_t end) {
    if (end <= start) {
      return;
    }
    auto k_chunk_all = k_f[batch_idx].slice(0, start, end);
    auto beta_chunk_all = beta_f[batch_idx].slice(0, start, end);
    std::optional<torch::Tensor> g_chunk_all = std::nullopt;
    if (g_f.has_value()) {
      g_chunk_all = g_f.value()[batch_idx].slice(0, start, end);
    }
    const int64_t chunk_len = end - start;

    for (int64_t head_idx = 0; head_idx < H; ++head_idx) {
      const int64_t group_idx = head_idx / heads_per_group;
      auto k_chunk = k_chunk_all.select(1, group_idx);
      auto beta_chunk = beta_chunk_all.select(1, head_idx);
      auto attn_chunk = torch::matmul(k_chunk * beta_chunk.unsqueeze(-1),
                                      k_chunk.transpose(0, 1));
      if (g_chunk_all.has_value()) {
        auto g_chunk = g_chunk_all.value().select(1, head_idx);
        attn_chunk =
            attn_chunk *
            torch::exp(g_chunk.unsqueeze(-1) - g_chunk.unsqueeze(0));
      }
      attn_chunk = torch::tril(attn_chunk, -1);
      out[batch_idx].slice(0, start, end).select(1, head_idx)
          .slice(1, 0, chunk_len)
          .copy_(attn_chunk.to(resolved_dtype));
    }
  };

  if (cu_seqlens.has_value()) {
    TORCH_CHECK(B == 1,
                "Only batch size 1 is supported when cu_seqlens are provided");
    auto cu = cu_seqlens.value().to(k.device(), torch::kLong, false, false)
                  .contiguous();
    TORCH_CHECK(cu.dim() == 1 && cu.numel() >= 2,
                "cu_seqlens must be a 1D tensor with at least 2 elements");
    for (int64_t seq_idx = 0; seq_idx < cu.numel() - 1; ++seq_idx) {
      const int64_t seq_start = cu[seq_idx].item<int64_t>();
      const int64_t seq_end = cu[seq_idx + 1].item<int64_t>();
      const int64_t seq_len = seq_end - seq_start;
      const int64_t local_chunks = (seq_len + chunk_size - 1) / chunk_size;
      for (int64_t chunk_idx = 0; chunk_idx < local_chunks; ++chunk_idx) {
        const int64_t start = seq_start + chunk_idx * chunk_size;
        const int64_t end = std::min(start + chunk_size, seq_end);
        write_chunk(0, start, end);
      }
    }
  } else {
    const int64_t total_chunks = (T + chunk_size - 1) / chunk_size;
    for (int64_t batch_idx = 0; batch_idx < B; ++batch_idx) {
      for (int64_t chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
        const int64_t start = chunk_idx * chunk_size;
        const int64_t end = std::min(start + chunk_size, T);
        write_chunk(batch_idx, start, end);
      }
    }
  }

  return out;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>,
           std::optional<torch::Tensor>>
chunk_gated_delta_rule_fwd_h_precompiled(
    const torch::Tensor& k, const torch::Tensor& w, const torch::Tensor& u,
    const std::optional<torch::Tensor>& g,
    const std::optional<torch::Tensor>& gk,
    const std::optional<torch::Tensor>& initial_state, bool output_final_state,
    int64_t chunk_size, bool save_new_value,
    const std::optional<torch::Tensor>& cu_seqlens) {
  TORCH_CHECK(k.is_cuda() && w.is_cuda() && u.is_cuda(),
              "chunk_gated_delta_rule_fwd_h_precompiled expects CUDA k/w/u");
  TORCH_CHECK(k.dim() == 4 && w.dim() == 4 && u.dim() == 4,
              "chunk_gated_delta_rule_fwd_h_precompiled expects k/w/u rank 4");
  TORCH_CHECK(k.size(0) == w.size(0) && k.size(0) == u.size(0) &&
                  k.size(1) == w.size(1) && k.size(1) == u.size(1),
              "chunk_gated_delta_rule_fwd_h_precompiled expects shared "
              "batch/time dimensions");
  TORCH_CHECK(w.size(2) == u.size(2),
              "chunk_gated_delta_rule_fwd_h_precompiled expects shared head "
              "dimension for w/u");
  TORCH_CHECK(w.size(3) == k.size(3),
              "chunk_gated_delta_rule_fwd_h_precompiled expects shared K "
              "dimension for k/w");
  TORCH_CHECK(chunk_size > 0,
              "chunk_gated_delta_rule_fwd_h_precompiled expects positive "
              "chunk_size");

  const int64_t B = k.size(0);
  const int64_t T = k.size(1);
  const int64_t Hg = k.size(2);
  const int64_t K = k.size(3);
  const int64_t H = u.size(2);
  const int64_t V = u.size(3);
  TORCH_CHECK(Hg > 0 && H % Hg == 0,
              "chunk_gated_delta_rule_fwd_h_precompiled expects H divisible "
              "by Hg");
  TORCH_CHECK(K <= 256,
              "chunk_gated_delta_rule_fwd_h_precompiled only supports K <= 256");

  if (g.has_value()) {
    TORCH_CHECK(g.value().is_cuda(),
                "chunk_gated_delta_rule_fwd_h_precompiled expects CUDA g when "
                "provided");
    TORCH_CHECK(g.value().dim() == 3 && g.value().size(0) == B &&
                    g.value().size(1) == T && g.value().size(2) == H,
                "chunk_gated_delta_rule_fwd_h_precompiled expects g shape "
                "[B, T, H]");
  }
  if (gk.has_value()) {
    TORCH_CHECK(gk.value().is_cuda(),
                "chunk_gated_delta_rule_fwd_h_precompiled expects CUDA gk when "
                "provided");
    TORCH_CHECK(gk.value().dim() == 4 && gk.value().size(0) == B &&
                    gk.value().size(1) == T && gk.value().size(2) == H &&
                    gk.value().size(3) == K,
                "chunk_gated_delta_rule_fwd_h_precompiled expects gk shape "
                "[B, T, H, K]");
  }

  int64_t N = B;
  int64_t total_chunks = 0;
  int64_t chunk_dim = (T + chunk_size - 1) / chunk_size;
  std::optional<torch::Tensor> cu = std::nullopt;
  if (cu_seqlens.has_value()) {
    TORCH_CHECK(B == 1,
                "Only batch size 1 is supported when cu_seqlens are provided");
    cu = cu_seqlens.value().to(k.device(), torch::kLong, false, false)
             .contiguous();
    TORCH_CHECK(cu.value().dim() == 1 && cu.value().numel() >= 2,
                "cu_seqlens must be a 1D tensor with at least 2 elements");
    N = cu.value().numel() - 1;
    for (int64_t seq_idx = 0; seq_idx < N; ++seq_idx) {
      const int64_t seq_start = cu.value()[seq_idx].item<int64_t>();
      const int64_t seq_end = cu.value()[seq_idx + 1].item<int64_t>();
      TORCH_CHECK(seq_end >= seq_start,
                  "chunk_gated_delta_rule_fwd_h_precompiled expects "
                  "non-decreasing cu_seqlens");
      const int64_t seq_len = seq_end - seq_start;
      total_chunks += (seq_len + chunk_size - 1) / chunk_size;
    }
    chunk_dim = total_chunks;
  } else {
    total_chunks = B * chunk_dim;
  }

  if (initial_state.has_value()) {
    TORCH_CHECK(initial_state.value().is_cuda(),
                "chunk_gated_delta_rule_fwd_h_precompiled expects CUDA "
                "initial_state when provided");
    TORCH_CHECK(initial_state.value().dim() == 4 &&
                    initial_state.value().size(0) == N &&
                    initial_state.value().size(1) == H &&
                    initial_state.value().size(2) == V &&
                    initial_state.value().size(3) == K,
                "chunk_gated_delta_rule_fwd_h_precompiled expects "
                "initial_state shape [N, H, V, K]");
  }

  auto h = torch::empty({B, chunk_dim, H, V, K}, k.options());
  std::optional<torch::Tensor> v_new = std::nullopt;
  if (save_new_value) {
    v_new = torch::empty_like(u);
  }
  std::optional<torch::Tensor> final_state = std::nullopt;
  if (output_final_state) {
    final_state = torch::empty({N, H, V, K}, k.options().dtype(torch::kFloat32));
  }

  auto k_f = k.to(torch::kFloat32);
  auto w_f = w.to(torch::kFloat32);
  auto u_f = u.to(torch::kFloat32);
  std::optional<torch::Tensor> g_f = std::nullopt;
  if (g.has_value()) {
    g_f = g.value().to(torch::kFloat32);
  }
  std::optional<torch::Tensor> gk_f = std::nullopt;
  if (gk.has_value()) {
    gk_f = gk.value().to(torch::kFloat32);
  }
  std::optional<torch::Tensor> initial_state_f = std::nullopt;
  if (initial_state.has_value()) {
    initial_state_f = initial_state.value().to(torch::kFloat32);
  }

  const int64_t heads_per_group = H / Hg;
  int64_t global_chunk_idx = 0;

  auto process_sequence = [&](int64_t seq_idx, int64_t batch_idx, int64_t seq_start,
                              int64_t seq_end) {
    std::vector<torch::Tensor> state_per_head;
    state_per_head.reserve(H);
    for (int64_t head_idx = 0; head_idx < H; ++head_idx) {
      if (initial_state_f.has_value()) {
        state_per_head.push_back(initial_state_f.value()[seq_idx][head_idx].clone());
      } else {
        state_per_head.push_back(
            torch::zeros({V, K}, k.options().dtype(torch::kFloat32)));
      }
    }

    const int64_t seq_len = seq_end - seq_start;
    const int64_t local_chunks = (seq_len + chunk_size - 1) / chunk_size;
    for (int64_t local_chunk_idx = 0; local_chunk_idx < local_chunks;
         ++local_chunk_idx) {
      const int64_t chunk_start = seq_start + local_chunk_idx * chunk_size;
      const int64_t chunk_end = std::min(chunk_start + chunk_size, seq_end);
      for (int64_t head_idx = 0; head_idx < H; ++head_idx) {
        const int64_t group_idx = head_idx / heads_per_group;
        auto state = state_per_head[head_idx];
        const int64_t h_chunk_idx = cu.has_value() ? global_chunk_idx : local_chunk_idx;
        h[batch_idx][h_chunk_idx][head_idx].copy_(
            state.to(h.scalar_type()));

        auto k_chunk = k_f[batch_idx].slice(0, chunk_start, chunk_end).select(
            1, group_idx);
        auto w_chunk = w_f[batch_idx].slice(0, chunk_start, chunk_end).select(
            1, head_idx);
        auto u_chunk = u_f[batch_idx].slice(0, chunk_start, chunk_end).select(
            1, head_idx);
        auto delta_value =
            u_chunk - torch::matmul(w_chunk, state.transpose(0, 1));

        if (v_new.has_value()) {
          v_new.value()[batch_idx].slice(0, chunk_start, chunk_end)
              .select(1, head_idx)
              .copy_(delta_value.to(v_new.value().scalar_type()));
        }

        if (g_f.has_value()) {
          auto g_chunk = g_f.value()[batch_idx].slice(0, chunk_start, chunk_end)
                             .select(1, head_idx);
          auto g_last = g_chunk[-1];
          delta_value =
              delta_value * torch::exp(g_last - g_chunk).unsqueeze(-1);
          state = state * torch::exp(g_last);
        }
        if (gk_f.has_value()) {
          auto gk_last = gk_f.value()[batch_idx][chunk_end - 1][head_idx];
          state = state * torch::exp(gk_last).unsqueeze(0);
        }

        state = state + torch::matmul(delta_value.transpose(0, 1), k_chunk);
        state_per_head[head_idx] = state;
      }
      ++global_chunk_idx;
    }

    if (final_state.has_value()) {
      for (int64_t head_idx = 0; head_idx < H; ++head_idx) {
        final_state.value()[seq_idx][head_idx].copy_(state_per_head[head_idx]);
      }
    }
  };

  if (cu.has_value()) {
    for (int64_t seq_idx = 0; seq_idx < N; ++seq_idx) {
      const int64_t seq_start = cu.value()[seq_idx].item<int64_t>();
      const int64_t seq_end = cu.value()[seq_idx + 1].item<int64_t>();
      process_sequence(seq_idx, 0, seq_start, seq_end);
    }
  } else {
    for (int64_t batch_idx = 0; batch_idx < B; ++batch_idx) {
      process_sequence(batch_idx, batch_idx, 0, T);
      TORCH_CHECK(global_chunk_idx == (batch_idx + 1) * chunk_dim,
                  "chunk_gated_delta_rule_fwd_h_precompiled internal chunk "
                  "accounting mismatch");
    }
  }

  TORCH_CHECK(global_chunk_idx == total_chunks,
              "chunk_gated_delta_rule_fwd_h_precompiled internal chunk "
              "accounting mismatch");
  return {h, v_new, final_state};
}

torch::Tensor solve_tril_precompiled(
    const torch::Tensor& A, const std::optional<torch::Tensor>& cu_seqlens,
    const std::optional<torch::ScalarType>& output_dtype) {
  TORCH_CHECK(A.is_cuda(), "solve_tril_precompiled expects CUDA input");
  TORCH_CHECK(A.dim() == 4,
              "solve_tril_precompiled expects rank-4 input [B, T, H, BT]");

  const int64_t B = A.size(0);
  const int64_t T = A.size(1);
  const int64_t H = A.size(2);
  const int64_t BT = A.size(3);
  TORCH_CHECK(BT == 16 || BT == 32 || BT == 64,
              "solve_tril_precompiled expects BT in {16, 32, 64}");

  const auto resolved_dtype = output_dtype.value_or(torch::kFloat32);
  auto out = torch::zeros_like(A, A.options().dtype(resolved_dtype));

  auto write_chunk = [&](int64_t batch_idx, int64_t start, int64_t end) {
    if (end <= start) {
      return;
    }
    const int64_t chunk_len = end - start;
    auto A_chunk = A[batch_idx].slice(0, start, end).to(torch::kFloat32);
    auto identity = torch::eye(chunk_len, A.options().dtype(torch::kFloat32));

    for (int64_t head_idx = 0; head_idx < H; ++head_idx) {
      auto lower =
          torch::tril(A_chunk.select(1, head_idx).slice(1, 0, chunk_len), -1);
      auto inverse = torch::inverse(identity + lower);
      out[batch_idx].slice(0, start, end).select(1, head_idx).slice(
          1, 0, chunk_len).copy_(inverse.to(resolved_dtype));
    }
  };

  if (cu_seqlens.has_value()) {
    TORCH_CHECK(B == 1,
                "Only batch size 1 is supported when cu_seqlens are provided");
    auto cu = cu_seqlens.value().to(A.device(), torch::kLong, false, false)
                  .contiguous();
    TORCH_CHECK(cu.dim() == 1 && cu.numel() >= 2,
                "cu_seqlens must be a 1D tensor with at least 2 elements");
    for (int64_t seq_idx = 0; seq_idx < cu.numel() - 1; ++seq_idx) {
      const int64_t seq_start = cu[seq_idx].item<int64_t>();
      const int64_t seq_end = cu[seq_idx + 1].item<int64_t>();
      TORCH_CHECK(seq_end >= seq_start,
                  "solve_tril_precompiled expects non-decreasing cu_seqlens");
      const int64_t seq_len = seq_end - seq_start;
      const int64_t local_chunks = (seq_len + BT - 1) / BT;
      for (int64_t chunk_idx = 0; chunk_idx < local_chunks; ++chunk_idx) {
        const int64_t start = seq_start + chunk_idx * BT;
        const int64_t end = std::min(start + BT, seq_end);
        write_chunk(0, start, end);
      }
    }
  } else {
    const int64_t total_chunks = (T + BT - 1) / BT;
    for (int64_t batch_idx = 0; batch_idx < B; ++batch_idx) {
      for (int64_t chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
        const int64_t start = chunk_idx * BT;
        const int64_t end = std::min(start + BT, T);
        write_chunk(batch_idx, start, end);
      }
    }
  }

  return out;
}

std::tuple<torch::Tensor, torch::Tensor> recompute_w_u_fwd_precompiled(
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& beta, const torch::Tensor& g_cumsum,
    const torch::Tensor& A, const std::optional<torch::Tensor>& cu_seqlens) {
  TORCH_CHECK(k.is_cuda() && v.is_cuda() && beta.is_cuda() &&
                  g_cumsum.is_cuda() && A.is_cuda(),
              "recompute_w_u_fwd_precompiled expects CUDA inputs");
  TORCH_CHECK(k.dim() == 4 && v.dim() == 4 && beta.dim() == 3 &&
                  g_cumsum.dim() == 3 && A.dim() == 4,
              "recompute_w_u_fwd_precompiled expects k/v/A rank 4 and "
              "beta/g_cumsum rank 3");
  TORCH_CHECK(k.size(0) == v.size(0) && k.size(1) == v.size(1) &&
                  k.size(0) == beta.size(0) && k.size(1) == beta.size(1) &&
                  k.size(0) == g_cumsum.size(0) &&
                  k.size(1) == g_cumsum.size(1) &&
                  k.size(0) == A.size(0) && k.size(1) == A.size(1),
              "recompute_w_u_fwd_precompiled expects shared batch/time "
              "dimensions");

  const int64_t B = k.size(0);
  const int64_t T = k.size(1);
  const int64_t Hg = k.size(2);
  const int64_t K = k.size(3);
  const int64_t H = v.size(2);
  const int64_t V = v.size(3);
  const int64_t BT = A.size(3);
  TORCH_CHECK(beta.size(2) == H && g_cumsum.size(2) == H && A.size(2) == H,
              "recompute_w_u_fwd_precompiled expects head dimensions to "
              "match v");
  TORCH_CHECK(Hg > 0 && H % Hg == 0,
              "recompute_w_u_fwd_precompiled expects H divisible by Hg");

  auto w = torch::empty({B, T, H, K}, k.options());
  auto u = torch::empty_like(v);

  auto k_f = k.to(torch::kFloat32);
  auto v_f = v.to(torch::kFloat32);
  auto beta_f = beta.to(torch::kFloat32);
  auto g_f = torch::exp(g_cumsum.to(torch::kFloat32));
  auto A_f = A.to(torch::kFloat32);

  const int64_t heads_per_group = H / Hg;

  auto write_chunk = [&](int64_t batch_idx, int64_t start, int64_t end) {
    if (end <= start) {
      return;
    }
    const int64_t chunk_len = end - start;
    auto A_chunk_all =
        A_f[batch_idx].slice(0, start, end).slice(2, 0, chunk_len);

    for (int64_t head_idx = 0; head_idx < H; ++head_idx) {
      const int64_t group_idx = head_idx / heads_per_group;
      auto beta_chunk =
          beta_f[batch_idx].slice(0, start, end).select(1, head_idx).unsqueeze(-1);
      auto g_chunk =
          g_f[batch_idx].slice(0, start, end).select(1, head_idx).unsqueeze(-1);
      auto A_chunk = A_chunk_all.select(1, head_idx);
      auto v_chunk =
          v_f[batch_idx].slice(0, start, end).select(1, head_idx);
      auto k_chunk =
          k_f[batch_idx].slice(0, start, end).select(1, group_idx);

      auto u_chunk = torch::matmul(A_chunk, v_chunk * beta_chunk);
      auto w_chunk = torch::matmul(A_chunk, k_chunk * beta_chunk * g_chunk);

      u[batch_idx].slice(0, start, end).select(1, head_idx).copy_(
          u_chunk.to(u.scalar_type()));
      w[batch_idx].slice(0, start, end).select(1, head_idx).copy_(
          w_chunk.to(w.scalar_type()));
    }
  };

  if (cu_seqlens.has_value()) {
    TORCH_CHECK(B == 1,
                "Only batch size 1 is supported when cu_seqlens are provided");
    auto cu = cu_seqlens.value().to(k.device(), torch::kLong, false, false)
                  .contiguous();
    TORCH_CHECK(cu.dim() == 1 && cu.numel() >= 2,
                "cu_seqlens must be a 1D tensor with at least 2 elements");
    for (int64_t seq_idx = 0; seq_idx < cu.numel() - 1; ++seq_idx) {
      const int64_t seq_start = cu[seq_idx].item<int64_t>();
      const int64_t seq_end = cu[seq_idx + 1].item<int64_t>();
      TORCH_CHECK(seq_end >= seq_start,
                  "recompute_w_u_fwd_precompiled expects non-decreasing "
                  "cu_seqlens");
      const int64_t seq_len = seq_end - seq_start;
      const int64_t local_chunks = (seq_len + BT - 1) / BT;
      for (int64_t chunk_idx = 0; chunk_idx < local_chunks; ++chunk_idx) {
        const int64_t start = seq_start + chunk_idx * BT;
        const int64_t end = std::min(start + BT, seq_end);
        write_chunk(0, start, end);
      }
    }
  } else {
    const int64_t total_chunks = (T + BT - 1) / BT;
    for (int64_t batch_idx = 0; batch_idx < B; ++batch_idx) {
      for (int64_t chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
        const int64_t start = chunk_idx * BT;
        const int64_t end = std::min(start + BT, T);
        write_chunk(batch_idx, start, end);
      }
    }
  }

  return {w, u};
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

void moe_batch_load_unquantized_runtime_precompiled(
    const torch::Tensor& slot_ids, const torch::Tensor& w13_src,
    const torch::Tensor& w2_src, torch::Tensor& w13_dst,
    torch::Tensor& w2_dst) {
  batch_copy_into_slots(slot_ids, w13_src, w13_dst, "w13_weight");
  batch_copy_into_slots(slot_ids, w2_src, w2_dst, "w2_weight");
}

void moe_batch_load_gptq_runtime_precompiled(
    const torch::Tensor& slot_ids, const torch::Tensor& w13_qweight_src,
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
    const std::optional<torch::Tensor>& w2_g_idx_sort_indices_dst) {
  batch_copy_into_slots(slot_ids, w13_qweight_src, w13_qweight_dst,
                        "w13_qweight");
  batch_copy_into_slots(slot_ids, w2_qweight_src, w2_qweight_dst,
                        "w2_qweight");
  batch_copy_into_slots(slot_ids, w13_scales_src, w13_scales_dst,
                        "w13_scales");
  batch_copy_into_slots(slot_ids, w2_scales_src, w2_scales_dst,
                        "w2_scales");
  batch_copy_into_slots(slot_ids, w13_qzeros_src, w13_qzeros_dst,
                        "w13_qzeros");
  batch_copy_into_slots(slot_ids, w2_qzeros_src, w2_qzeros_dst,
                        "w2_qzeros");
  batch_copy_into_slots_optional(slot_ids, w13_g_idx_src, w13_g_idx_dst,
                                 "w13_g_idx");
  batch_copy_into_slots_optional(slot_ids, w2_g_idx_src, w2_g_idx_dst,
                                 "w2_g_idx");
  batch_copy_into_slots_optional(slot_ids, w13_g_idx_sort_indices_src,
                                 w13_g_idx_sort_indices_dst,
                                 "w13_g_idx_sort_indices");
  batch_copy_into_slots_optional(slot_ids, w2_g_idx_sort_indices_src,
                                 w2_g_idx_sort_indices_dst,
                                 "w2_g_idx_sort_indices");
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
