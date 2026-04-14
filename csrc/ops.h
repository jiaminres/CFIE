#pragma once

#include <optional>
#include <torch/library.h>
#include <tuple>

#include <ATen/core/List.h>
#include "core/scalar_type.hpp"

#include <vector>

inline torch::Tensor weak_ref_tensor(torch::Tensor& tensor) {
  // Ensure tensor is on CUDA
  if (!tensor.is_cuda()) {
    throw std::runtime_error("Tensor must be on CUDA device");
  }

  // Get the raw data pointer
  void* data_ptr = tensor.data_ptr();

  // Get tensor sizes and strides
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();

  // Get tensor options (dtype, device)
  auto options = tensor.options();

  // Create a new tensor from the raw data pointer
  auto new_tensor = torch::from_blob(data_ptr, sizes, strides, options);

  return new_tensor;
}

void paged_attention_v1(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void paged_attention_v2(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const int64_t tp_rank,
    const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step);

void merge_attn_states(torch::Tensor& output,
                       std::optional<torch::Tensor> output_lse,
                       const torch::Tensor& prefix_output,
                       const torch::Tensor& prefix_lse,
                       const torch::Tensor& suffix_output,
                       const torch::Tensor& suffix_lse);
#ifndef USE_ROCM
void convert_vertical_slash_indexes(
    torch::Tensor& block_count,      // [BATCH, N_HEADS, NUM_ROWS]
    torch::Tensor& block_offset,     // [BATCH, N_HEADS, NUM_ROWS, NNZ_S]
    torch::Tensor& column_count,     // [BATCH, N_HEADS, NUM_ROWS]
    torch::Tensor& column_index,     // [BATCH, N_HEADS, NUM_ROWS, NNZ_V]
    torch::Tensor q_seqlens,         // [BATCH, ]
    torch::Tensor kv_seqlens,        // [BATCH, ]
    torch::Tensor vertical_indexes,  // [BATCH, N_HEADS, NNZ_V]
    torch::Tensor slash_indexes,     // [BATCH, N_HEADS, NNZ_S]
    int64_t context_size, int64_t block_size_M, int64_t block_size_N,
    bool causal);

void convert_vertical_slash_indexes_mergehead(
    torch::Tensor& block_count,            // [BATCH, N_HEADS, NUM_ROWS]
    torch::Tensor& block_offset,           // [BATCH, N_HEADS, NUM_ROWS, NNZ_S]
    torch::Tensor& column_count,           // [BATCH, N_HEADS, NUM_ROWS]
    torch::Tensor& column_index,           // [BATCH, N_HEADS, NUM_ROWS, NNZ_V]
    torch::Tensor q_seqlens,               // [BATCH, ]
    torch::Tensor kv_seqlens,              // [BATCH, ]
    torch::Tensor vertical_indexes,        // [BATCH, N_HEADS, NNZ_V]
    torch::Tensor slash_indexes,           // [BATCH, N_HEADS, NNZ_S]
    torch::Tensor vertical_indices_count,  // [N_HEADS, ]
    torch::Tensor slash_indices_count, int64_t context_size,
    int64_t block_size_M, int64_t block_size_N, bool causal);
#endif

void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon);

void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon);

void fused_qk_norm_rope(torch::Tensor& qkv, int64_t num_heads_q,
                        int64_t num_heads_k, int64_t num_heads_v,
                        int64_t head_dim, double eps, torch::Tensor& q_weight,
                        torch::Tensor& k_weight, torch::Tensor& cos_sin_cache,
                        bool is_neox, torch::Tensor& position_ids);

void apply_repetition_penalties_(torch::Tensor& logits,
                                 const torch::Tensor& prompt_mask,
                                 const torch::Tensor& output_mask,
                                 const torch::Tensor& repetition_penalties);

void top_k_per_row_prefill(const torch::Tensor& logits,
                           const torch::Tensor& rowStarts,
                           const torch::Tensor& rowEnds, torch::Tensor& indices,
                           int64_t numRows, int64_t stride0, int64_t stride1,
                           int64_t topK);

void top_k_per_row_decode(const torch::Tensor& logits, int64_t next_n,
                          const torch::Tensor& seqLens, torch::Tensor& indices,
                          int64_t numRows, int64_t stride0, int64_t stride1,
                          int64_t topK);

void large_context_topk(const torch::Tensor& score, torch::Tensor& indices,
                        const torch::Tensor& lengths,
                        std::optional<torch::Tensor> row_starts_opt);

torch::Tensor correct_attn_out_precompiled(torch::Tensor& out,
                                           const torch::Tensor& lses,
                                           int64_t cp_rank,
                                           bool is_lse_base_on_e);

std::tuple<torch::Tensor, torch::Tensor> dcp_lse_combine_precompiled(
    const torch::Tensor& recv_output, const torch::Tensor& recv_lse,
    bool return_lse, bool is_lse_base_on_e);

void prefill_attention_precompiled(torch::Tensor& output,
                                   const torch::Tensor& q,
                                   const torch::Tensor& k,
                                   const torch::Tensor& v,
                                   const torch::Tensor& b_start_loc,
                                   const torch::Tensor& b_seq_len,
                                   bool is_causal, double softmax_scale,
                                   int64_t sliding_window_q,
                                   int64_t sliding_window_k);

void prefix_prefill_attention_precompiled(
    torch::Tensor& output, const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& v, const torch::Tensor& gathered_ctx_k,
    const torch::Tensor& gathered_ctx_v, const torch::Tensor& cu_ctx_lens,
    const torch::Tensor& b_start_loc, const torch::Tensor& b_seq_len,
    double sm_scale, int64_t sliding_window, bool skip_decode);

torch::Tensor pack_seq_precompiled(const torch::Tensor& x,
                                   const torch::Tensor& lengths,
                                   double pad_value);

torch::Tensor unpack_seq_precompiled(const torch::Tensor& packed_tensor,
                                     const torch::Tensor& lengths);

torch::Tensor expand_batch_to_tokens_precompiled(
    const torch::Tensor& x, const torch::Tensor& cu_num_tokens,
    int64_t replace_from, int64_t replace_to);

torch::Tensor sample_recovered_tokens_precompiled(
    const torch::Tensor& cu_num_draft_tokens,
    const torch::Tensor& draft_token_ids,
    const std::optional<torch::Tensor>& draft_probs,
    const torch::Tensor& target_probs, const torch::Tensor& inv_q);

void apply_top_k_top_p_precompiled(
    torch::Tensor& logits, const std::optional<torch::Tensor>& k,
    const std::optional<torch::Tensor>& p, double mask_value);

void rejection_greedy_sample_precompiled(
    torch::Tensor& output_token_ids, const torch::Tensor& cu_num_draft_tokens,
    const torch::Tensor& draft_token_ids, const torch::Tensor& target_argmax,
    const torch::Tensor& bonus_token_ids,
    const std::optional<torch::Tensor>& is_greedy, int64_t max_spec_len);

void rejection_random_sample_precompiled(
    torch::Tensor& output_token_ids, const torch::Tensor& cu_num_draft_tokens,
    const torch::Tensor& draft_token_ids,
    const std::optional<torch::Tensor>& draft_probs,
    const torch::Tensor& target_probs, const torch::Tensor& bonus_token_ids,
    const torch::Tensor& recovered_token_ids,
    const torch::Tensor& uniform_probs,
    const std::optional<torch::Tensor>& is_greedy, int64_t max_spec_len);

void input_batch_prepare_prefill_inputs_precompiled(
    torch::Tensor& input_ids, torch::Tensor& next_prefill_tokens,
    const torch::Tensor& idx_mapping, const torch::Tensor& query_start_loc,
    const torch::Tensor& all_token_ids, const torch::Tensor& prefill_len,
    const torch::Tensor& num_computed_tokens);

void input_batch_prepare_pos_seq_lens_precompiled(
    const torch::Tensor& idx_mapping, const torch::Tensor& query_start_loc,
    const torch::Tensor& num_computed_tokens, torch::Tensor& pos,
    torch::Tensor& seq_lens);

torch::Tensor input_batch_combine_sampled_and_draft_tokens_precompiled(
    torch::Tensor& input_ids, const torch::Tensor& idx_mapping,
    const torch::Tensor& last_sampled_tokens,
    const torch::Tensor& query_start_loc, const torch::Tensor& seq_lens,
    const torch::Tensor& prefill_len, const torch::Tensor& draft_tokens,
    const torch::Tensor& cu_num_logits, int64_t num_logits);

std::tuple<torch::Tensor, torch::Tensor>
input_batch_get_num_sampled_and_rejected_precompiled(
    torch::Tensor& num_sampled, const torch::Tensor& seq_lens,
    const torch::Tensor& cu_num_logits, const torch::Tensor& idx_mapping,
    const torch::Tensor& prefill_len);

void input_batch_post_update_precompiled(
    const torch::Tensor& idx_mapping, torch::Tensor& num_computed_tokens,
    torch::Tensor& last_sampled_tokens,
    const std::optional<torch::Tensor>& output_bin_counts,
    const torch::Tensor& sampled_tokens, const torch::Tensor& num_sampled,
    const torch::Tensor& num_rejected, const torch::Tensor& query_start_loc,
    torch::Tensor& all_token_ids, torch::Tensor& total_len);

void input_batch_post_update_pool_precompiled(
    const torch::Tensor& idx_mapping, torch::Tensor& num_computed_tokens,
    const torch::Tensor& query_start_loc);

std::tuple<torch::Tensor, torch::Tensor>
input_batch_expand_idx_mapping_precompiled(const torch::Tensor& idx_mapping,
                                           int64_t total_num_logits,
                                           const torch::Tensor& cu_num_logits);

void eagle_step_update_slot_mapping_and_metadata_precompiled(
    const torch::Tensor& positions_1d, const torch::Tensor& block_table_tensor,
    torch::Tensor& seq_lens, int64_t block_size, int64_t max_model_len,
    torch::Tensor& out_clamped_positions, torch::Tensor& out_slot_mapping,
    int64_t input_batch_size);

void eagle_prepare_inputs_padded_precompiled(
    const torch::Tensor& cu_num_draft_tokens,
    const torch::Tensor& valid_sampled_tokens_count,
    const torch::Tensor& query_start_loc_gpu,
    torch::Tensor& token_indices_to_sample,
    torch::Tensor& num_rejected_tokens_gpu);

void eagle_prepare_next_token_padded_precompiled(
    const torch::Tensor& sampled_token_ids,
    const torch::Tensor& discard_request_mask,
    const torch::Tensor& backup_next_token_ids, torch::Tensor& next_token_ids,
    torch::Tensor& valid_sampled_tokens_count, int64_t vocab_size);

void copy_and_expand_eagle_inputs_precompiled(
    const torch::Tensor& target_token_ids,
    const torch::Tensor& target_positions, const torch::Tensor& next_token_ids,
    torch::Tensor& out_input_ids, torch::Tensor& out_positions,
    torch::Tensor& out_is_rejected_token_mask,
    torch::Tensor& out_is_masked_token_mask,
    torch::Tensor& out_new_token_indices,
    torch::Tensor& out_hidden_state_mapping,
    const torch::Tensor& query_start_loc, const torch::Tensor& query_end_loc,
    int64_t padding_token_id, int64_t parallel_drafting_token_id,
    int64_t total_input_tokens, int64_t num_padding_slots_per_request,
    bool shift_input_ids);

void prepare_eagle_inputs_precompiled(
    torch::Tensor& last_token_indices, torch::Tensor& eagle_input_ids,
    torch::Tensor& eagle_positions, const torch::Tensor& target_input_ids,
    const torch::Tensor& target_positions, const torch::Tensor& idx_mapping,
    const torch::Tensor& last_sampled,
    const torch::Tensor& next_prefill_tokens, const torch::Tensor& num_sampled,
    const torch::Tensor& num_rejected, const torch::Tensor& query_start_loc);

void prepare_eagle_decode_precompiled(
    const torch::Tensor& draft_tokens, const torch::Tensor& output_hidden_states,
    const torch::Tensor& last_token_indices,
    const torch::Tensor& target_seq_lens, const torch::Tensor& num_rejected,
    torch::Tensor& input_ids, torch::Tensor& positions,
    torch::Tensor& query_start_loc, torch::Tensor& seq_lens,
    torch::Tensor& input_hidden_states, int64_t max_model_len,
    int64_t max_num_reqs);

void update_eagle_inputs_precompiled(
    const torch::Tensor& draft_tokens, const torch::Tensor& output_hidden_states,
    torch::Tensor& input_ids, torch::Tensor& positions,
    torch::Tensor& seq_lens, torch::Tensor& hidden_states,
    int64_t max_model_len);

void rms_norm_static_fp8_quant(torch::Tensor& out, torch::Tensor& input,
                               torch::Tensor& weight, torch::Tensor& scale,
                               double epsilon);

void fused_add_rms_norm_static_fp8_quant(torch::Tensor& out,
                                         torch::Tensor& input,
                                         torch::Tensor& residual,
                                         torch::Tensor& weight,
                                         torch::Tensor& scale, double epsilon);

void rms_norm_dynamic_per_token_quant(torch::Tensor& out,
                                      torch::Tensor const& input,
                                      torch::Tensor const& weight,
                                      torch::Tensor& scales,
                                      double const epsilon,
                                      std::optional<torch::Tensor> scale_ub,
                                      std::optional<torch::Tensor> residual);

void rms_norm_per_block_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor const& weight,
                              torch::Tensor& scales, double const epsilon,
                              std::optional<torch::Tensor> scale_ub,
                              std::optional<torch::Tensor> residual,
                              int64_t group_size, bool is_scale_transposed);

void rotary_embedding(torch::Tensor& positions, torch::Tensor& query,
                      std::optional<torch::Tensor> key, int64_t head_size,
                      torch::Tensor& cos_sin_cache, bool is_neox);

std::tuple<torch::Tensor, torch::Tensor> mrope_rotary_embedding(
    const torch::Tensor& query, const torch::Tensor& key,
    const torch::Tensor& cos, const torch::Tensor& sin, int64_t head_size,
    int64_t rotary_dim, c10::List<int64_t> mrope_section, bool is_neox,
    bool mrope_interleaved);

torch::Tensor gated_layer_norm(
    const torch::Tensor& input, const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& gate, double epsilon,
    int64_t group_size, bool norm_before_gate, bool is_rms_norm,
    const std::string& activation);

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
    bool use_qk_l2norm_in_kernel, bool is_kda);

torch::Tensor apply_rotary_emb_precompiled(const torch::Tensor& x,
                                           const torch::Tensor& cos,
                                           const torch::Tensor& sin,
                                           bool is_neox_style,
                                           bool enable_fp32_compute);

std::tuple<torch::Tensor, torch::Tensor> chunk_gated_delta_rule_precompiled(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& g, const torch::Tensor& beta, double scale,
    const torch::Tensor& initial_state, bool output_final_state,
    const std::optional<torch::Tensor>& cu_seqlens,
    bool use_qk_l2norm_in_kernel);

std::tuple<torch::Tensor, torch::Tensor>
fused_recurrent_gated_delta_rule_packed_decode_precompiled(
    const torch::Tensor& mixed_qkv, const torch::Tensor& a,
    const torch::Tensor& b, const torch::Tensor& A_log,
    const torch::Tensor& dt_bias, double scale, torch::Tensor& initial_state,
    torch::Tensor& out, const torch::Tensor& ssm_state_indices,
    bool use_qk_l2norm_in_kernel);

torch::Tensor l2norm_precompiled(
    const torch::Tensor& x, double eps,
    const std::optional<torch::ScalarType>& output_dtype);

torch::Tensor chunk_local_cumsum_precompiled(
    const torch::Tensor& g, int64_t chunk_size, bool reverse,
    const std::optional<torch::Tensor>& cu_seqlens, bool head_first,
    const std::optional<torch::ScalarType>& output_dtype);

torch::Tensor chunk_fwd_o_precompiled(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& h, const std::optional<torch::Tensor>& g,
    double scale, const std::optional<torch::Tensor>& cu_seqlens,
    int64_t block_size);

torch::Tensor chunk_scaled_dot_kkt_fwd_precompiled(
    const torch::Tensor& k, const std::optional<torch::Tensor>& g,
    const torch::Tensor& beta, const std::optional<torch::Tensor>& cu_seqlens,
    int64_t chunk_size,
    const std::optional<torch::ScalarType>& output_dtype);

std::tuple<torch::Tensor, std::optional<torch::Tensor>,
           std::optional<torch::Tensor>>
chunk_gated_delta_rule_fwd_h_precompiled(
    const torch::Tensor& k, const torch::Tensor& w, const torch::Tensor& u,
    const std::optional<torch::Tensor>& g,
    const std::optional<torch::Tensor>& gk,
    const std::optional<torch::Tensor>& initial_state, bool output_final_state,
    int64_t chunk_size, bool save_new_value,
    const std::optional<torch::Tensor>& cu_seqlens);

torch::Tensor solve_tril_precompiled(
    const torch::Tensor& A, const std::optional<torch::Tensor>& cu_seqlens,
    const std::optional<torch::ScalarType>& output_dtype);

std::tuple<torch::Tensor, torch::Tensor> recompute_w_u_fwd_precompiled(
    const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& beta, const torch::Tensor& g_cumsum,
    const torch::Tensor& A, const std::optional<torch::Tensor>& cu_seqlens);

std::tuple<torch::Tensor, torch::Tensor> fused_gdn_gating_precompiled(
    const torch::Tensor& A_log, const torch::Tensor& a,
    const torch::Tensor& b, const torch::Tensor& dt_bias, double beta,
    double threshold);

torch::Tensor causal_conv1d_fn_precompiled(
    const torch::Tensor& x, const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias, torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const std::optional<torch::Tensor>& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const std::string& activation, int64_t pad_slot_id);

torch::Tensor causal_conv1d_update_precompiled(
    const torch::Tensor& x, torch::Tensor& conv_state,
    const torch::Tensor& weight, const std::optional<torch::Tensor>& bias,
    const std::string& activation,
    const std::optional<torch::Tensor>& conv_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t pad_slot_id,
    const std::optional<torch::Tensor>& block_idx_last_scheduled_token,
    const std::optional<torch::Tensor>& initial_state_idx);

torch::Tensor count_expert_num_tokens_precompiled(
    const torch::Tensor& topk_ids, int64_t num_local_experts,
    const std::optional<torch::Tensor>& expert_map);

torch::Tensor zero_experts_compute_identity_precompiled(
    torch::Tensor& expert_indices, torch::Tensor& expert_scales,
    int64_t num_experts, const torch::Tensor& hidden_states);

void zero_kv_blocks_precompiled(const torch::Tensor& block_ids,
                                c10::List<torch::Tensor> kv_tensors,
                                c10::List<int64_t> block_dims,
                                c10::List<int64_t> ratios);

void moe_batch_load_unquantized_runtime_precompiled(
    const torch::Tensor& slot_ids, const torch::Tensor& w13_src,
    const torch::Tensor& w2_src, torch::Tensor& w13_dst,
    torch::Tensor& w2_dst);

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
    const std::optional<torch::Tensor>& w2_g_idx_sort_indices_dst);

void moe_batched_mm_precompiled(
    const torch::Tensor& A, const torch::Tensor& B, torch::Tensor& C,
    const torch::Tensor& expert_num_tokens,
    const std::optional<torch::Tensor>& A_scale,
    const std::optional<torch::Tensor>& B_scale, bool use_fp8_w8a8,
    bool per_act_token_quant);

void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

void silu_and_mul_quant(torch::Tensor& out, torch::Tensor& input,
                        torch::Tensor& scale);

#ifndef USE_ROCM
void silu_and_mul_nvfp4_quant(torch::Tensor& out,
                              torch::Tensor& output_block_scale,
                              torch::Tensor& input,
                              torch::Tensor& input_global_scale);
#endif
void persistent_masked_m_silu_mul_quant(
    const at::Tensor& input,   // (E, T, 2*H)
    const at::Tensor& counts,  // (E)
    at::Tensor& y_q,           // (E, T, H) [OUT]
    at::Tensor& y_s,           // (E, T, H//group_size) [OUT]
    bool use_ue8m0);

void mul_and_silu(torch::Tensor& out, torch::Tensor& input);

void gelu_and_mul(torch::Tensor& out, torch::Tensor& input);

void gelu_tanh_and_mul(torch::Tensor& out, torch::Tensor& input);

void fatrelu_and_mul(torch::Tensor& out, torch::Tensor& input,
                     double threshold);
void swiglustep_and_mul(torch::Tensor& out, torch::Tensor& input,
                        double limit = 7.0);
void swigluoai_and_mul(torch::Tensor& out, torch::Tensor& input,
                       double alpha = 1.702, double limit = 7.0);

void gelu_new(torch::Tensor& out, torch::Tensor& input);

void gelu_fast(torch::Tensor& out, torch::Tensor& input);

void gelu_quick(torch::Tensor& out, torch::Tensor& input);

void cutlass_mla_decode(torch::Tensor const& out, torch::Tensor const& q_nope,
                        torch::Tensor const& q_pe,
                        torch::Tensor const& kv_c_and_k_pe_cache,
                        torch::Tensor const& seq_lens,
                        torch::Tensor const& page_table, double scale);

torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor);

#ifndef USE_ROCM

torch::Tensor awq_gemm(torch::Tensor _in_feats, torch::Tensor _kernel,
                       torch::Tensor _scaling_factors, torch::Tensor _zeros,
                       int64_t split_k_iters);

torch::Tensor awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, int64_t split_k_iters,
                             int64_t thx, int64_t thy);

torch::Tensor permute_cols(torch::Tensor const& A, torch::Tensor const& perm);
#endif

torch::Tensor ggml_dequantize(torch::Tensor W, int64_t type, int64_t m,
                              int64_t n,
                              std::optional<at::ScalarType> const& dtype);

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W, torch::Tensor X,
                                  int64_t type, int64_t row);

torch::Tensor ggml_mul_mat_a8(torch::Tensor W, torch::Tensor X, int64_t type,
                              int64_t row);

torch::Tensor ggml_moe_a8(torch::Tensor X, torch::Tensor W,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor expert_ids,
                          torch::Tensor num_tokens_post_padded, int64_t type,
                          int64_t row, int64_t top_k, int64_t tokens);

torch::Tensor ggml_moe_a8_vec(torch::Tensor X, torch::Tensor W,
                              torch::Tensor topk_ids, int64_t top_k,
                              int64_t type, int64_t row, int64_t tokens);

int64_t ggml_moe_get_block_size(int64_t type);

#ifndef USE_ROCM

bool cutlass_scaled_mm_supports_fp4(int64_t cuda_device_capability);
bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability);
bool cutlass_scaled_mm_supports_block_fp8(int64_t cuda_device_capability);
bool cutlass_group_gemm_supported(int64_t cuda_device_capability);

void cutlass_scaled_fp4_mm(torch::Tensor& D, torch::Tensor const& A,
                           torch::Tensor const& B, torch::Tensor const& A_sf,
                           torch::Tensor const& B_sf,
                           torch::Tensor const& alpha);

void cutlass_scaled_mm(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       std::optional<torch::Tensor> const& bias);

void cutlass_moe_mm(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch);

void cutlass_fp4_group_mm(
    torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b,
    const torch::Tensor& a_blockscale, const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas, const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets);

void get_cutlass_moe_mm_data(
    const torch::Tensor& topk_ids, torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation, torch::Tensor& output_permutation,
    const int64_t num_experts, const int64_t n, const int64_t k,
    const std::optional<torch::Tensor>& blockscale_offsets);

void get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
    const torch::Tensor& expert_first_token_offset,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    const int64_t n, const int64_t k, const bool swap_ab);

void get_cutlass_batched_moe_mm_data(torch::Tensor& expert_offsets,
                                     torch::Tensor& problem_sizes1,
                                     torch::Tensor& problem_sizes2,
                                     const torch::Tensor& expert_num_tokens,
                                     const int64_t num_local_experts,
                                     const int64_t padded_m, const int64_t n,
                                     const int64_t k);

void cutlass_scaled_mm_azp(torch::Tensor& out, torch::Tensor const& a,
                           torch::Tensor const& b,
                           torch::Tensor const& a_scales,
                           torch::Tensor const& b_scales,
                           torch::Tensor const& azp_adj,
                           std::optional<torch::Tensor> const& azp,
                           std::optional<torch::Tensor> const& bias);

bool cutlass_sparse_scaled_mm_supported(int64_t cuda_device_capability);

void cutlass_scaled_sparse_mm(torch::Tensor& out, torch::Tensor const& a,
                              torch::Tensor const& b, torch::Tensor const& e,
                              torch::Tensor const& a_scales,
                              torch::Tensor const& b_scales,
                              std::optional<torch::Tensor> const& bias);

std::vector<torch::Tensor> cutlass_sparse_compress(torch::Tensor const& a);

void scaled_fp4_quant(torch::Tensor& output, torch::Tensor const& input,
                      torch::Tensor& output_scale,
                      torch::Tensor const& input_scale,
                      bool is_sf_swizzled_layout);

void scaled_fp4_experts_quant(
    torch::Tensor& output, torch::Tensor& output_scale,
    torch::Tensor const& input, torch::Tensor const& input_global_scale,
    torch::Tensor const& input_offset_by_experts,
    torch::Tensor const& output_scale_offset_by_experts);

void silu_and_mul_scaled_fp4_experts_quant(
    torch::Tensor& output, torch::Tensor& output_scale,
    torch::Tensor const& input, torch::Tensor const& input_global_scale,
    torch::Tensor const& input_offset_by_experts,
    torch::Tensor const& output_scale_offset_by_experts);

void per_token_group_quant_fp8(const torch::Tensor& input,
                               torch::Tensor& output_q, torch::Tensor& output_s,
                               int64_t group_size, double eps, double fp8_min,
                               double fp8_max, bool scale_ue8m0,
                               bool dummy_is_scale_transposed,
                               bool dummy_is_tma_aligned);

void per_token_group_quant_int8(const torch::Tensor& input,
                                torch::Tensor& output_q,
                                torch::Tensor& output_s, int64_t group_size,
                                double eps, double int8_min, double int8_max);

// Fused activation quantisation + DeepGEMM-compatible UE8M0-packed scales.
void per_token_group_quant_8bit_packed(const torch::Tensor& input,
                                       torch::Tensor& output_q,
                                       torch::Tensor& output_s_packed,
                                       int64_t group_size, double eps,
                                       double min_8bit, double max_8bit);

#endif

void static_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor const& scale,
                              std::optional<torch::Tensor> const& azp);

void dynamic_scaled_int8_quant(torch::Tensor& out, torch::Tensor const& input,
                               torch::Tensor& scales,
                               std::optional<torch::Tensor> const& azp);

torch::Tensor gptq_gemm(torch::Tensor a, torch::Tensor b_q_weight,
                        torch::Tensor b_gptq_qzeros,
                        torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                        bool use_exllama, bool use_v2_format, int64_t bit);

void gptq_shuffle(torch::Tensor q_weight, torch::Tensor q_perm, int64_t bit);

void static_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor const& scale,
    std::optional<std::tuple<int64_t, int64_t>> group_shape = std::nullopt);

void dynamic_scaled_fp8_quant(torch::Tensor& out, torch::Tensor const& input,
                              torch::Tensor& scale);

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scale,
    std::optional<torch::Tensor> const& scale_ub);

void selective_scan_fwd(
    const torch::Tensor& u, const torch::Tensor& delta, const torch::Tensor& A,
    const torch::Tensor& B, const torch::Tensor& C,
    const std::optional<torch::Tensor>& D_,
    const std::optional<torch::Tensor>& z_,
    const std::optional<torch::Tensor>& delta_bias_, bool delta_softplus,
    const std::optional<torch::Tensor>& query_start_loc,
    const std::optional<torch::Tensor>& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const torch::Tensor& ssm_states, int64_t pad_slot_id, int64_t block_size,
    const std::optional<torch::Tensor>& block_idx_first_scheduled_token,
    const std::optional<torch::Tensor>& block_idx_last_scheduled_token,
    const std::optional<torch::Tensor>& initial_state_idx,
    const std::optional<torch::Tensor>& cu_chunk_seqlen,
    const std::optional<torch::Tensor>& last_chunk_indices);

torch::Tensor dynamic_4bit_int_moe_cpu(
    torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
    torch::Tensor w13_packed, torch::Tensor w2_packed, int64_t H, int64_t I,
    int64_t I2, int64_t group_size, bool apply_router_weight_on_input,
    int64_t activation_kind);

using fptr_t = int64_t;
fptr_t init_custom_ar(const std::vector<int64_t>& fake_ipc_ptrs,
                      torch::Tensor& rank_data, int64_t rank,
                      bool fully_connected);
void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                fptr_t reg_buffer, int64_t reg_buffer_sz_bytes);
void dispose(fptr_t _fa);
int64_t meta_size();
void register_buffer(fptr_t _fa, const std::vector<int64_t>& fake_ipc_ptrs);
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa);
void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets);
std::tuple<int64_t, torch::Tensor> allocate_shared_buffer_and_handle(
    int64_t size);
int64_t open_mem_handle(torch::Tensor& mem_handle);
void free_shared_buffer(int64_t buffer);

torch::Tensor hadacore_transform(torch::Tensor& x, bool inplace);

#ifdef USE_ROCM
fptr_t init_custom_qr(int64_t rank, int64_t world_size,
                      std::optional<int64_t> qr_max_size = std::nullopt);
void qr_destroy(fptr_t _fa);
torch::Tensor qr_get_handle(fptr_t _fa);
void qr_open_handles(fptr_t _fa, const std::vector<torch::Tensor>& handles);
void qr_all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out,
                   int64_t quant_level, bool cast_bf2half = false);
int64_t qr_max_size();
#endif

#ifndef USE_ROCM
void dsv3_fused_a_gemm(torch::Tensor& output, torch::Tensor const& mat_a,
                       torch::Tensor const& mat_b);
#endif
