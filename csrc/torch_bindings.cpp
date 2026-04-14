#include "cache.h"
#include "cuda_utils.h"
#include "ops.h"
#include "core/registration.h"

#include <torch/library.h>
#include <torch/version.h>

// 本文件集中把 C++ / CUDA 算子暴露给 PyTorch；
// Python 层看到的 `torch.ops.<namespace>.*` 入口，最终都会从这里拿到 schema 和后端实现。
//
// `ops.def(...)` 负责声明 PyTorch 侧 schema，决定算子名称、参数签名和返回值形态。
// `ops.impl(...)` 负责把同名 schema 绑定到具体后端实现；本文件大多数实现绑定到 CUDA，
// 也有少量 helper 绑定到 CPU。
//
// 如果某个算子在这里只有 `def(...)`、没有对应的 `impl(...)`，
// 通常表示它依赖条件编译，并会在满足编译条件的源文件里补做实现注册。
//
// 关于 meta 函数的说明：
// `X_meta` 这一类签名表示与算子 `X` 对应的 meta 函数签名。
// 它们必须与 `X` 本身的函数签名保持同步。
// 一般来说，只有返回 `Tensor` 的函数才需要配套的 meta 函数。
//
// 关于算子注册和函数 schema 的详细文档，可参考下面链接。
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations


// ------------------------------- 注册主命名空间下的自定义算子 -------------------------------
// 下面这一整段是主算子表；
// 推理热链、量化热链、MoE 热链以及若干运行时辅助算子都会先在这里声明 schema。
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // ------------------------------- 注册基础辅助算子与轻量工具入口 -------------------------------
  // 这几项算子会被更高层的 Python 逻辑直接探测或复用，
  // 负责提供量化 helper、弱引用张量包装以及 CPU Tensor 到 CUDA 视图的桥接能力。
  ops.def(
      "persistent_masked_m_silu_mul_quant(Tensor input, Tensor counts, Tensor! "
      "y_q, Tensor! y_s,"
      "bool use_ue8m0) -> ()");
  ops.impl("persistent_masked_m_silu_mul_quant", torch::kCUDA,
           &persistent_masked_m_silu_mul_quant);

  ops.def("weak_ref_tensor(Tensor input) -> Tensor");
  ops.impl("weak_ref_tensor", torch::kCUDA, &weak_ref_tensor);

  ops.def("get_cuda_view_from_cpu_tensor(Tensor cpu_tensor) -> Tensor");
  ops.impl("get_cuda_view_from_cpu_tensor", torch::kCPU,
           &get_cuda_view_from_cpu_tensor);

  // ------------------------------- 注册 Attention 与稀疏索引相关算子 -------------------------------
  // 这一段负责把分页注意力、注意力结果合并以及稀疏注意力索引转换入口注册到 PyTorch；
  // 调度器和 worker 在构造好 block table / cache 之后，会沿这些入口进入底层 CUDA 实现。
  ops.def(
      "paged_attention_v1("
      "    Tensor! out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);

  // `paged_attention_v2` 相比 v1 额外暴露中间缓冲区，
  // 方便底层实现分步累计 softmax 所需的统计量。
  ops.def(
      "paged_attention_v2("
      "    Tensor! out, Tensor! exp_sums, Tensor! max_logits,"
      "    Tensor! tmp_out, Tensor query, Tensor key_cache,"
      "    Tensor value_cache, int num_kv_heads, float scale,"
      "    Tensor block_tables, Tensor seq_lens, int block_size,"
      "    int max_seq_len, Tensor? alibi_slopes,"
      "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
      "    int tp_rank, int blocksparse_local_blocks,"
      "    int blocksparse_vert_stride, int blocksparse_block_size,"
      "    int blocksparse_head_sliding_step) -> ()");
  ops.impl("paged_attention_v2", torch::kCUDA, &paged_attention_v2);

  // 这个入口用于合并拆分 KV 之后得到的部分 attention 结果，
  // 对应 split-KV 场景下把 prefix / suffix 两段结果重新拼成统一输出。
  ops.def(
      "merge_attn_states("
      "    Tensor! output,"
      "    Tensor!? output_lse,"
      "    Tensor prefix_output,"
      "    Tensor prefix_lse,"
      "    Tensor suffix_output,"
      "    Tensor suffix_lse) -> ()");
  ops.impl("merge_attn_states", torch::kCUDA, &merge_attn_states);
#ifndef USE_ROCM
  ops.def(
      "convert_vertical_slash_indexes("
      "   Tensor! block_count, Tensor! block_offset, "
      "   Tensor! column_count, Tensor! column_index, "
      "   Tensor q_seqlens, Tensor q_seqlens, "
      "   Tensor vertical_indexes, Tensor slash_indexes, "
      "   int context_size, int block_size_M, int block_size_N, "
      "   bool causal) -> ()");
  ops.impl("convert_vertical_slash_indexes", torch::kCUDA,
           &convert_vertical_slash_indexes);

  ops.def(
      "convert_vertical_slash_indexes_mergehead("
      "   Tensor! block_count, Tensor! block_offset, "
      "   Tensor! column_count, Tensor! column_index, "
      "   Tensor q_seqlens, Tensor q_seqlens, "
      "   Tensor vertical_indexes, Tensor slash_indexes, "
      "   Tensor vertical_indices_count, Tensor slash_indices_count, "
      "   int context_size, int block_size_M, int block_size_N, "
      "   bool causal) -> ()");
  ops.impl("convert_vertical_slash_indexes_mergehead", torch::kCUDA,
           &convert_vertical_slash_indexes_mergehead);
#endif

  // ------------------------------- 注册激活函数与常规归一化算子 -------------------------------
  // 这一段主要服务 MLP / 门控前馈层；
  // 多个 `def + impl` 对虽然形式重复，但它们分别对应不同激活族和量化族的真实 CUDA kernel。
  ops.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  ops.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  ops.def(
      "silu_and_mul_quant(Tensor! result, Tensor input, Tensor scale) -> ()");
  ops.impl("silu_and_mul_quant", torch::kCUDA, &silu_and_mul_quant);

#ifndef USE_ROCM
  ops.def(
      "silu_and_mul_nvfp4_quant(Tensor! result, Tensor! result_block_scale, "
      "Tensor input, Tensor input_global_scale) -> ()");
  ops.impl("silu_and_mul_nvfp4_quant", torch::kCUDA, &silu_and_mul_nvfp4_quant);
#endif

  ops.def("mul_and_silu(Tensor! out, Tensor input) -> ()");
  ops.impl("mul_and_silu", torch::kCUDA, &mul_and_silu);

  // `gelu_and_mul` 对应 GeGLU 的精确近似路径。
  ops.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

  // `gelu_tanh_and_mul` 对应 GeGLU 的 `tanh` 近似路径。
  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  // `fatrelu_and_mul` 负责 FATReLU 变体。
  ops.def("fatrelu_and_mul(Tensor! out, Tensor input, float threshold) -> ()");
  ops.impl("fatrelu_and_mul", torch::kCUDA, &fatrelu_and_mul);

  ops.def("swiglustep_and_mul(Tensor! out, Tensor input, float limit=7.0) -> ()");
  ops.impl("swiglustep_and_mul", torch::kCUDA, &swiglustep_and_mul);

  ops.def(
      "swigluoai_and_mul(Tensor! out, Tensor input, float alpha=1.702, float "
      "limit=7.0) "
      "-> ()");
  ops.impl("swigluoai_and_mul", torch::kCUDA, &swigluoai_and_mul);

  // `gelu_new` 对应 GPT-2 风格 GELU。
  ops.def("gelu_new(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_new", torch::kCUDA, &gelu_new);

  // `gelu_fast` 提供更快的 GELU 近似实现。
  ops.def("gelu_fast(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_fast", torch::kCUDA, &gelu_fast);

  // `gelu_quick` 提供另一条 quick GELU 路径。
  ops.def("gelu_quick(Tensor! out, Tensor input) -> ()");
  ops.impl("gelu_quick", torch::kCUDA, &gelu_quick);

  // `rms_norm` 和后续 `fused_add_rms_norm` 是最常见的归一化基础入口，
  // 上层模型构图会先通过这些名字拿到对应 kernel。
  ops.def(
      "rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> "
      "()");
  ops.impl("rms_norm", torch::kCUDA, &rms_norm);

  // 这里注册就地 add + RMSNorm 融合版本，
  // 让 residual 更新和归一化能复用同一条 CUDA 热链。
  ops.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, "
      "float epsilon) -> ()");
  ops.impl("fused_add_rms_norm", torch::kCUDA, &fused_add_rms_norm);

  // 这一项把 QK 归一化与 RoPE 旋转合并到同一个入口，
  // 供需要降低中间 Tensor 往返的注意力路径使用。
  ops.def(
      "fused_qk_norm_rope(Tensor! qkv, int num_heads_q, "
      "int num_heads_k, int num_heads_v, int head_dim, float eps, "
      "Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache, "
      "bool is_neox, Tensor position_ids) -> ()");
  ops.impl("fused_qk_norm_rope", torch::kCUDA, &fused_qk_norm_rope);

  // 这里注册 logits 原地重复惩罚，
  // 供采样前的后处理逻辑直接在 GPU 上调整分布。
  ops.def(
      "apply_repetition_penalties_(Tensor! logits, Tensor prompt_mask, "
      "Tensor output_mask, Tensor repetition_penalties) -> ()");
  ops.impl("apply_repetition_penalties_", torch::kCUDA,
           &apply_repetition_penalties_);

  // 这一组 top-k helper 主要服务 prefill / decode 采样前的候选截断。
  ops.def(
      "top_k_per_row_prefill(Tensor logits, Tensor rowStarts, Tensor rowEnds, "
      "Tensor! indices, int numRows, int stride0, "
      "int stride1, int topK) -> ()");
  ops.impl("top_k_per_row_prefill", torch::kCUDA, &top_k_per_row_prefill);

  ops.def(
      "top_k_per_row_decode(Tensor logits, int next_n, "
      "Tensor seq_lens, Tensor! indices, "
      "int numRows, int stride0, int stride1, int topK) -> ()");
  ops.impl("top_k_per_row_decode", torch::kCUDA, &top_k_per_row_decode);

  ops.def(
      "large_context_topk(Tensor score, Tensor indices, Tensor lengths, "
      "Tensor? "
      "row_starts_opt) -> ()");
  ops.impl("large_context_topk", torch::kCUDA, &large_context_topk);

  // ------------------------------- 注册 Windows / 共享预编译推理热链算子 -------------------------------
  // 下面这批 `_precompiled` 入口是当前 Windows 无 Triton 场景的重要统一落点；
  // Python 层会先走统一 selector，再决定是否命中这些 `_C` / CUDA 实现。

  // 这一组 attention helper 负责在共享路径下修正注意力输出、
  // 处理 DCP/LSE 合并，以及 prefill / prefix prefill 的基础算子。
  ops.def("correct_attn_out_precompiled("
          "Tensor! out, Tensor lses, int cp_rank, "
          "bool is_lse_base_on_e=True) -> Tensor");
  ops.impl("correct_attn_out_precompiled", torch::kCUDA,
           &correct_attn_out_precompiled);

  ops.def("dcp_lse_combine_precompiled("
          "Tensor recv_output, Tensor recv_lse, bool return_lse=False, "
          "bool is_lse_base_on_e=True) -> (Tensor, Tensor)");
  ops.impl("dcp_lse_combine_precompiled", torch::kCUDA,
           &dcp_lse_combine_precompiled);

  ops.def("prefill_attention_precompiled("
          "Tensor! output, Tensor q, Tensor k, Tensor v, "
          "Tensor b_start_loc, Tensor b_seq_len, bool is_causal, "
          "float softmax_scale, int sliding_window_q, "
          "int sliding_window_k) -> ()");
  ops.impl("prefill_attention_precompiled", torch::kCUDA,
           &prefill_attention_precompiled);

  ops.def("prefix_prefill_attention_precompiled("
          "Tensor! output, Tensor q, Tensor k, Tensor v, "
          "Tensor gathered_ctx_k, Tensor gathered_ctx_v, Tensor cu_ctx_lens, "
          "Tensor b_start_loc, Tensor b_seq_len, float sm_scale, "
          "int sliding_window, bool skip_decode) -> ()");
  ops.impl("prefix_prefill_attention_precompiled", torch::kCUDA,
           &prefix_prefill_attention_precompiled);

  ops.def("pack_seq_precompiled("
          "Tensor x, Tensor lengths, float pad_value) -> Tensor");
  ops.impl("pack_seq_precompiled", torch::kCUDA, &pack_seq_precompiled);

  ops.def("unpack_seq_precompiled("
          "Tensor packed_tensor, Tensor lengths) -> Tensor");
  ops.impl("unpack_seq_precompiled", torch::kCUDA, &unpack_seq_precompiled);

  ops.def("expand_batch_to_tokens_precompiled("
          "Tensor x, Tensor cu_num_tokens, int replace_from, "
          "int replace_to) -> Tensor");
  ops.impl("expand_batch_to_tokens_precompiled", torch::kCUDA,
           &expand_batch_to_tokens_precompiled);

  ops.def("sample_recovered_tokens_precompiled("
          "Tensor cu_num_draft_tokens, Tensor draft_token_ids, "
          "Tensor? draft_probs, Tensor target_probs, "
          "Tensor inv_q) -> Tensor");
  ops.impl("sample_recovered_tokens_precompiled", torch::kCUDA,
           &sample_recovered_tokens_precompiled);

  ops.def("apply_top_k_top_p_precompiled("
          "Tensor! logits, Tensor? k, Tensor? p, float mask_value) -> ()");
  ops.impl("apply_top_k_top_p_precompiled", torch::kCUDA,
           &apply_top_k_top_p_precompiled);

  // 这一组 rejection / sample helper 用于 speculative decode 的接受-拒绝流程，
  // 让草稿 token 的保留、替换和恢复逻辑留在 GPU 热链中完成。
  ops.def("rejection_greedy_sample_precompiled("
          "Tensor! output_token_ids, Tensor cu_num_draft_tokens, "
          "Tensor draft_token_ids, Tensor target_argmax, "
          "Tensor bonus_token_ids, Tensor? is_greedy, "
          "int max_spec_len) -> ()");
  ops.impl("rejection_greedy_sample_precompiled", torch::kCUDA,
           &rejection_greedy_sample_precompiled);

  ops.def("rejection_random_sample_precompiled("
          "Tensor! output_token_ids, Tensor cu_num_draft_tokens, "
          "Tensor draft_token_ids, Tensor? draft_probs, "
          "Tensor target_probs, Tensor bonus_token_ids, "
          "Tensor recovered_token_ids, Tensor uniform_probs, "
          "Tensor? is_greedy, int max_spec_len) -> ()");
  ops.impl("rejection_random_sample_precompiled", torch::kCUDA,
           &rejection_random_sample_precompiled);

  ops.def("input_batch_prepare_prefill_inputs_precompiled("
          "Tensor! input_ids, Tensor! next_prefill_tokens, "
          "Tensor idx_mapping, Tensor query_start_loc, "
          "Tensor all_token_ids, Tensor prefill_len, "
          "Tensor num_computed_tokens) -> ()");
  ops.impl("input_batch_prepare_prefill_inputs_precompiled", torch::kCUDA,
           &input_batch_prepare_prefill_inputs_precompiled);

  // 这一组 `input_batch_*` 入口负责把调度器输出的索引、位置信息和采样结果
  // 整理成 worker 可直接消费的批输入结构。
  ops.def("input_batch_prepare_pos_seq_lens_precompiled("
          "Tensor idx_mapping, Tensor query_start_loc, "
          "Tensor num_computed_tokens, Tensor! pos, Tensor! seq_lens) -> ()");
  ops.impl("input_batch_prepare_pos_seq_lens_precompiled", torch::kCUDA,
           &input_batch_prepare_pos_seq_lens_precompiled);

  ops.def("input_batch_combine_sampled_and_draft_tokens_precompiled("
          "Tensor! input_ids, Tensor idx_mapping, "
          "Tensor last_sampled_tokens, Tensor query_start_loc, "
          "Tensor seq_lens, Tensor prefill_len, Tensor draft_tokens, "
          "Tensor cu_num_logits, int num_logits) -> Tensor");
  ops.impl("input_batch_combine_sampled_and_draft_tokens_precompiled",
           torch::kCUDA,
           &input_batch_combine_sampled_and_draft_tokens_precompiled);

  ops.def("input_batch_get_num_sampled_and_rejected_precompiled("
          "Tensor! num_sampled, Tensor seq_lens, Tensor cu_num_logits, "
          "Tensor idx_mapping, Tensor prefill_len) -> (Tensor, Tensor)");
  ops.impl("input_batch_get_num_sampled_and_rejected_precompiled",
           torch::kCUDA,
           &input_batch_get_num_sampled_and_rejected_precompiled);

  ops.def("input_batch_post_update_precompiled("
          "Tensor idx_mapping, Tensor! num_computed_tokens, "
          "Tensor! last_sampled_tokens, Tensor? output_bin_counts, "
          "Tensor sampled_tokens, Tensor num_sampled, Tensor num_rejected, "
          "Tensor query_start_loc, Tensor! all_token_ids, "
          "Tensor! total_len) -> ()");
  ops.impl("input_batch_post_update_precompiled", torch::kCUDA,
           &input_batch_post_update_precompiled);

  ops.def("input_batch_post_update_pool_precompiled("
          "Tensor idx_mapping, Tensor! num_computed_tokens, "
          "Tensor query_start_loc) -> ()");
  ops.impl("input_batch_post_update_pool_precompiled", torch::kCUDA,
           &input_batch_post_update_pool_precompiled);

  ops.def("input_batch_expand_idx_mapping_precompiled("
          "Tensor idx_mapping, int total_num_logits, "
          "Tensor cu_num_logits) -> (Tensor, Tensor)");
  ops.impl("input_batch_expand_idx_mapping_precompiled", torch::kCUDA,
           &input_batch_expand_idx_mapping_precompiled);

  ops.def("eagle_step_update_slot_mapping_and_metadata_precompiled("
          "Tensor positions_1d, Tensor block_table_tensor, "
          "Tensor! seq_lens, int block_size, int max_model_len, "
          "Tensor! out_clamped_positions, Tensor! out_slot_mapping, "
          "int input_batch_size) -> ()");
  ops.impl("eagle_step_update_slot_mapping_and_metadata_precompiled",
           torch::kCUDA,
           &eagle_step_update_slot_mapping_and_metadata_precompiled);

  // 这一组 `eagle_*` / `prepare_*` 入口服务 EAGLE speculative decode，
  // 负责把草稿输入、slot mapping 和 decode 元数据整理成后续模型执行所需的形态。
  ops.def("eagle_prepare_inputs_padded_precompiled("
          "Tensor cu_num_draft_tokens, Tensor valid_sampled_tokens_count, "
          "Tensor query_start_loc_gpu, Tensor! token_indices_to_sample, "
          "Tensor! num_rejected_tokens_gpu) -> ()");
  ops.impl("eagle_prepare_inputs_padded_precompiled", torch::kCUDA,
           &eagle_prepare_inputs_padded_precompiled);

  ops.def("eagle_prepare_next_token_padded_precompiled("
          "Tensor sampled_token_ids, Tensor discard_request_mask, "
          "Tensor backup_next_token_ids, Tensor! next_token_ids, "
          "Tensor! valid_sampled_tokens_count, int vocab_size) -> ()");
  ops.impl("eagle_prepare_next_token_padded_precompiled", torch::kCUDA,
           &eagle_prepare_next_token_padded_precompiled);

  ops.def("copy_and_expand_eagle_inputs_precompiled("
          "Tensor target_token_ids, Tensor target_positions, "
          "Tensor next_token_ids, Tensor! out_input_ids, "
          "Tensor! out_positions, Tensor! out_is_rejected_token_mask, "
          "Tensor! out_is_masked_token_mask, Tensor! out_new_token_indices, "
          "Tensor! out_hidden_state_mapping, Tensor query_start_loc, "
          "Tensor query_end_loc, int padding_token_id, "
          "int parallel_drafting_token_id, int total_input_tokens, "
          "int num_padding_slots_per_request, bool shift_input_ids) -> ()");
  ops.impl("copy_and_expand_eagle_inputs_precompiled", torch::kCUDA,
           &copy_and_expand_eagle_inputs_precompiled);

  ops.def("prepare_eagle_inputs_precompiled("
          "Tensor! last_token_indices, Tensor! eagle_input_ids, "
          "Tensor! eagle_positions, Tensor target_input_ids, "
          "Tensor target_positions, Tensor idx_mapping, "
          "Tensor last_sampled, Tensor next_prefill_tokens, "
          "Tensor num_sampled, Tensor num_rejected, "
          "Tensor query_start_loc) -> ()");
  ops.impl("prepare_eagle_inputs_precompiled", torch::kCUDA,
           &prepare_eagle_inputs_precompiled);

  ops.def("prepare_eagle_decode_precompiled("
          "Tensor draft_tokens, Tensor output_hidden_states, "
          "Tensor last_token_indices, Tensor target_seq_lens, "
          "Tensor num_rejected, Tensor! input_ids, Tensor! positions, "
          "Tensor! query_start_loc, Tensor! seq_lens, "
          "Tensor! input_hidden_states, int max_model_len, "
          "int max_num_reqs) -> ()");
  ops.impl("prepare_eagle_decode_precompiled", torch::kCUDA,
           &prepare_eagle_decode_precompiled);

  ops.def("update_eagle_inputs_precompiled("
          "Tensor draft_tokens, Tensor output_hidden_states, "
          "Tensor! input_ids, Tensor! positions, Tensor! seq_lens, "
          "Tensor! hidden_states, int max_model_len) -> ()");
  ops.impl("update_eagle_inputs_precompiled", torch::kCUDA,
           &update_eagle_inputs_precompiled);

  // ------------------------------- 注册归一化与量化融合热链 -------------------------------
  // 这几项入口把 RMSNorm 与静态 / 动态量化拼成单次 kernel，
  // 目的是减少中间 Tensor 落地，给量化推理主链直接复用。
  ops.def(
      "rms_norm_static_fp8_quant(Tensor! result, Tensor input, Tensor weight, "
      "Tensor scale, float epsilon) -> "
      "()");
  ops.impl("rms_norm_static_fp8_quant", torch::kCUDA,
           &rms_norm_static_fp8_quant);

  // 这里把 residual 融合更新和静态 FP8 量化版 RMSNorm 合在一起注册。
  ops.def(
      "fused_add_rms_norm_static_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! residual, Tensor weight, "
      "Tensor scale, float epsilon) -> ()");
  ops.impl("fused_add_rms_norm_static_fp8_quant", torch::kCUDA,
           &fused_add_rms_norm_static_fp8_quant);

  // 动态逐 token 量化版本会在运行时回填 scale，
  // 适合需要随输入波动动态估计量化尺度的路径。
  ops.def(
      "rms_norm_dynamic_per_token_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual) -> ()");
  ops.impl("rms_norm_dynamic_per_token_quant", torch::kCUDA,
           &rms_norm_dynamic_per_token_quant);

  // block 量化版本需要额外的分组信息，
  // 供更细粒度的块级量化内核直接消费。
  ops.def(
      "rms_norm_per_block_quant(Tensor! result, Tensor input, "
      "Tensor weight, Tensor! scale, float epsilon, "
      "Tensor? scale_ub, Tensor!? residual, int group_size, "
      "bool is_scale_transposed) -> ()");
  ops.impl("rms_norm_per_block_quant", torch::kCUDA, &rms_norm_per_block_quant);

  // ------------------------------- 注册 RoPE、FLA 与 Mamba 共享预编译算子 -------------------------------
  // 这一段集中服务旋转位置编码、FLA 门控 / 状态更新以及 Mamba 卷积热链，
  // 是 Windows 无 Triton 时最关键的一批共享 `_precompiled` 入口。

  // `rotary_embedding` 负责传统 GPT-NeoX / GPT-J 风格的 RoPE 旋转。
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");
  ops.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);

  // `mrope_rotary_embedding` 对应多段式 mRoPE 变体，
  // 直接返回新的 query / key 张量对。
  ops.def(
      "mrope_rotary_embedding("
      "    Tensor query, Tensor key, Tensor cos, Tensor sin, int head_size,"
      "    int rotary_dim, int[] mrope_section, bool is_neox,"
      "    bool mrope_interleaved) -> (Tensor, Tensor)");
  ops.impl("mrope_rotary_embedding", torch::kCUDA, &mrope_rotary_embedding);

  ops.def(
      "gated_layer_norm("
      "    Tensor input, Tensor weight, Tensor? bias, Tensor? gate,"
      "    float epsilon, int group_size, bool norm_before_gate,"
      "    bool is_rms_norm, str activation) -> Tensor");
  ops.impl("gated_layer_norm", torch::kCUDA, &gated_layer_norm);

  // 下面这些 FLA `_precompiled` 入口覆盖门控更新、局部 cumsum、
  // recurrent / chunk 前向和若干线性代数 helper，
  // 供 Python selector 在无 Triton 时命中共享 CUDA 闭环。
  ops.def(
      "fused_sigmoid_gating_delta_rule_update_precompiled("
      "    Tensor A_log, Tensor a, Tensor b, Tensor dt_bias,"
      "    Tensor q, Tensor k, Tensor v, float beta, float threshold,"
      "    float scale, Tensor initial_state, bool inplace_final_state,"
      "    Tensor? cu_seqlens, Tensor? ssm_state_indices,"
      "    Tensor? num_accepted_tokens, bool use_qk_l2norm_in_kernel,"
      "    bool is_kda) -> (Tensor, Tensor)");
  ops.impl("fused_sigmoid_gating_delta_rule_update_precompiled", torch::kCUDA,
           &fused_sigmoid_gating_delta_rule_update_precompiled);

  ops.def(
      "apply_rotary_emb_precompiled("
      "    Tensor x, Tensor cos, Tensor sin, bool is_neox_style,"
      "    bool enable_fp32_compute) -> Tensor");
  ops.impl("apply_rotary_emb_precompiled", torch::kCUDA,
           &apply_rotary_emb_precompiled);

  ops.def(
      "chunk_gated_delta_rule_precompiled("
      "    Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float scale,"
      "    Tensor initial_state, bool output_final_state, Tensor? cu_seqlens,"
      "    bool use_qk_l2norm_in_kernel) -> (Tensor, Tensor)");
  ops.impl("chunk_gated_delta_rule_precompiled", torch::kCUDA,
           &chunk_gated_delta_rule_precompiled);

  ops.def(
      "fused_recurrent_gated_delta_rule_packed_decode_precompiled("
      "    Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log,"
      "    Tensor dt_bias, float scale, Tensor! initial_state, Tensor! out,"
      "    Tensor ssm_state_indices, bool use_qk_l2norm_in_kernel)"
      " -> (Tensor, Tensor)");
  ops.impl("fused_recurrent_gated_delta_rule_packed_decode_precompiled",
           torch::kCUDA,
           &fused_recurrent_gated_delta_rule_packed_decode_precompiled);

  ops.def(
      "l2norm_precompiled("
      "    Tensor x, float eps, ScalarType? output_dtype=None) -> Tensor");
  ops.impl("l2norm_precompiled", torch::kCUDA, &l2norm_precompiled);

  ops.def(
      "chunk_local_cumsum_precompiled("
      "    Tensor g, int chunk_size, bool reverse=False,"
      "    Tensor? cu_seqlens=None, bool head_first=False,"
      "    ScalarType? output_dtype=None) -> Tensor");
  ops.impl("chunk_local_cumsum_precompiled", torch::kCUDA,
           &chunk_local_cumsum_precompiled);

  ops.def(
      "chunk_fwd_o_precompiled("
      "    Tensor q, Tensor k, Tensor v, Tensor h, Tensor? g,"
      "    float scale, Tensor? cu_seqlens, int block_size) -> Tensor");
  ops.impl("chunk_fwd_o_precompiled", torch::kCUDA, &chunk_fwd_o_precompiled);

  ops.def(
      "chunk_scaled_dot_kkt_fwd_precompiled("
      "    Tensor k, Tensor? g, Tensor beta, Tensor? cu_seqlens,"
      "    int chunk_size, ScalarType? output_dtype=None) -> Tensor");
  ops.impl("chunk_scaled_dot_kkt_fwd_precompiled", torch::kCUDA,
           &chunk_scaled_dot_kkt_fwd_precompiled);

  ops.def(
      "chunk_gated_delta_rule_fwd_h_precompiled("
      "    Tensor k, Tensor w, Tensor u, Tensor? g=None, Tensor? gk=None,"
      "    Tensor? initial_state=None, bool output_final_state=False,"
      "    int chunk_size=64, bool save_new_value=True,"
      "    Tensor? cu_seqlens=None) -> (Tensor, Tensor?, Tensor?)");
  ops.impl("chunk_gated_delta_rule_fwd_h_precompiled", torch::kCUDA,
           &chunk_gated_delta_rule_fwd_h_precompiled);

  ops.def(
      "solve_tril_precompiled("
      "    Tensor A, Tensor? cu_seqlens=None,"
      "    ScalarType? output_dtype=None) -> Tensor");
  ops.impl("solve_tril_precompiled", torch::kCUDA, &solve_tril_precompiled);

  ops.def(
      "recompute_w_u_fwd_precompiled("
      "    Tensor k, Tensor v, Tensor beta, Tensor g_cumsum,"
      "    Tensor A, Tensor? cu_seqlens=None) -> (Tensor, Tensor)");
  ops.impl("recompute_w_u_fwd_precompiled", torch::kCUDA,
           &recompute_w_u_fwd_precompiled);

  ops.def(
      "fused_gdn_gating_precompiled("
      "    Tensor A_log, Tensor a, Tensor b, Tensor dt_bias, float beta,"
      "    float threshold) -> (Tensor, Tensor)");
  ops.impl("fused_gdn_gating_precompiled", torch::kCUDA,
           &fused_gdn_gating_precompiled);

  ops.def(
      "causal_conv1d_fn_precompiled("
      "    Tensor x, Tensor weight, Tensor? bias, Tensor! conv_states,"
      "    Tensor query_start_loc, Tensor? cache_indices,"
      "    Tensor? has_initial_state, str activation,"
      "    int pad_slot_id) -> Tensor");
  ops.impl("causal_conv1d_fn_precompiled", torch::kCUDA,
           &causal_conv1d_fn_precompiled);

  ops.def(
      "causal_conv1d_update_precompiled("
      "    Tensor x, Tensor! conv_state, Tensor weight, Tensor? bias,"
      "    str activation, Tensor? conv_state_indices,"
      "    Tensor? num_accepted_tokens, Tensor? query_start_loc,"
      "    int pad_slot_id, Tensor? block_idx_last_scheduled_token,"
      "    Tensor? initial_state_idx) -> Tensor");
  ops.impl("causal_conv1d_update_precompiled", torch::kCUDA,
           &causal_conv1d_update_precompiled);

  // 这一组 MoE helper 负责统计专家 token 数、处理零专家快路径、
  // 清零 KV block，以及把 runtime-ready 专家权重批量装入 GPU slot。
  ops.def(
      "count_expert_num_tokens_precompiled("
      "    Tensor topk_ids, int num_local_experts, Tensor? expert_map) -> Tensor");
  ops.impl("count_expert_num_tokens_precompiled", torch::kCUDA,
           &count_expert_num_tokens_precompiled);

  ops.def(
      "zero_experts_compute_identity_precompiled("
      "    Tensor(a!) expert_indices, Tensor(b!) expert_scales,"
      "    int num_experts, Tensor hidden_states) -> Tensor");
  ops.impl("zero_experts_compute_identity_precompiled", torch::kCUDA,
           &zero_experts_compute_identity_precompiled);

  ops.def(
      "zero_kv_blocks_precompiled("
      "    Tensor block_ids, Tensor[] kv_tensors, int[] block_dims,"
      "    int[] ratios) -> ()");
  ops.impl("zero_kv_blocks_precompiled", torch::kCUDA,
           &zero_kv_blocks_precompiled);

  ops.def(
      "moe_batch_load_unquantized_runtime_precompiled("
      "    Tensor slot_ids, Tensor w13_src, Tensor w2_src,"
      "    Tensor! w13_dst, Tensor! w2_dst) -> ()");
  ops.impl("moe_batch_load_unquantized_runtime_precompiled", torch::kCUDA,
           &moe_batch_load_unquantized_runtime_precompiled);

  ops.def(
      "moe_batch_load_gptq_runtime_precompiled("
      "    Tensor slot_ids, Tensor w13_qweight_src, Tensor w2_qweight_src,"
      "    Tensor w13_scales_src, Tensor w2_scales_src,"
      "    Tensor w13_qzeros_src, Tensor w2_qzeros_src,"
      "    Tensor! w13_qweight_dst, Tensor! w2_qweight_dst,"
      "    Tensor! w13_scales_dst, Tensor! w2_scales_dst,"
      "    Tensor! w13_qzeros_dst, Tensor! w2_qzeros_dst,"
      "    Tensor? w13_g_idx_src, Tensor? w2_g_idx_src,"
      "    Tensor? w13_g_idx_sort_indices_src,"
      "    Tensor? w2_g_idx_sort_indices_src,"
      "    Tensor!? w13_g_idx_dst, Tensor!? w2_g_idx_dst,"
      "    Tensor!? w13_g_idx_sort_indices_dst,"
      "    Tensor!? w2_g_idx_sort_indices_dst) -> ()");
  ops.impl("moe_batch_load_gptq_runtime_precompiled", torch::kCUDA,
           &moe_batch_load_gptq_runtime_precompiled);

  ops.def(
      "moe_batched_mm_precompiled("
      "    Tensor A, Tensor B, Tensor! C, Tensor expert_num_tokens,"
      "    Tensor? A_scale, Tensor? B_scale, bool use_fp8_w8a8,"
      "    bool per_act_token_quant) -> ()");
  ops.impl("moe_batched_mm_precompiled", torch::kCUDA,
           &moe_batched_mm_precompiled);

  // ------------------------------- 注册量化、GEMM 与专用后端算子 -------------------------------
  // 这一大段同时承接 AWQ / Marlin / CUTLASS / GGML / GPTQ / FP8 / INT8 / AllSpark 等族的入口；
  // Python 层会按模型类型、设备能力和量化配置选择其中合适的 schema。
#ifndef USE_ROCM
  // `dsv3_fused_a_gemm` 只在满足设备条件时提供实现，
  // 这里先暴露 schema，真正的 `impl` 会在对应源文件里按条件补注册。
  ops.def(
      "dsv3_fused_a_gemm(Tensor! output, Tensor mat_a, Tensor mat_b) -> ()");
  // 具体实现依赖条件编译，因此不在本文件直接绑定 `impl(...)`。

  // 这一对入口负责 AWQ 的量化 GEMM 与反量化。
  ops.def(
      "awq_gemm(Tensor _in_feats, Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, SymInt split_k_iters) -> Tensor");
  ops.impl("awq_gemm", torch::kCUDA, &awq_gemm);

  ops.def(
      "awq_dequantize(Tensor _kernel, Tensor _scaling_factors, "
      "Tensor _zeros, SymInt split_k_iters, int thx, int thy) -> Tensor");
  ops.impl("awq_dequantize", torch::kCUDA, &awq_dequantize);

  // 关于 Marlin kernel 的 `workspace` 参数，这里需要特别说明：
  // 从底层执行看，kernel 的确会临时改写 `workspace`；
  // 但它在返回前会把内容恢复，因此从 PyTorch schema 视角可以把它视作“净效果不变”的输入。
  //
  // 这里故意不把 `workspace` 标成可变参数，
  // 是为了避免它干扰 `ScalarType` 一类自定义参数参与 schema 推导。
  // 如果把它声明成可变参数，PyTorch 会在
  // `torch._higher_order_ops._register_effectful_op` 处触发断言，
  // 从而阻断这些 kernel 进入 `torch.compile` 路径。
  //
  // 更完整的背景可参考下面文档。
  // https://docs.google.com/document/d/18fBMPuOJ0fY5ZQ6YyrHUppw9FA332CpNtgB6SOIgyuA

  // 这一组是 Hopper 上的 Machete dense mixed-precision GEMM 能力；
  // 只先注册 schema，实际实现按编译条件在其它源文件中补绑定。
  ops.def(
      "machete_supported_schedules("
      "   ScalarType a_type,"
      "   int b_type,"
      "   ScalarType? maybe_group_scales_type,"
      "   ScalarType? maybe_group_zeros_type,"
      "   ScalarType? maybe_channel_scales_type,"
      "   ScalarType? maybe_token_scales_type,"
      "   ScalarType? maybe_out_type"
      ") -> str[]");
  ops.def(
      "machete_mm("
      "   Tensor A,"
      "   Tensor B,"
      "   int b_type,"
      "   ScalarType? out_type,"
      "   Tensor? group_scales,"
      "   Tensor? group_zeros,"
      "   int?    group_size,"
      "   Tensor? channel_scales,"
      "   Tensor? token_scales,"
      "   str?    schedule"
      ") -> Tensor");
  ops.def(
      "machete_prepack_B("
      "   Tensor B,"
      "   ScalarType a_type,"
      "   int b_type,"
      "   ScalarType? group_scales_type"
      ") -> Tensor");
  // 具体实现依赖条件编译，因此本文件只声明 schema。

  ops.def("permute_cols(Tensor A, Tensor perm) -> Tensor");
  ops.impl("permute_cols", torch::kCUDA, &permute_cols);

  // 这一组是 Marlin 系列量化 GEMM 与相关预处理入口，
  // 覆盖 GPTQ、AWQ、FP8、NVFP4、MXFP4 等多种量化格式。
  ops.def(
      "marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, "
      "Tensor? b_bias_or_none,Tensor b_scales, "
      "Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, "
      "Tensor? "
      "g_idx_or_none, Tensor? perm_or_none, Tensor workspace, int b_type_id, "
      "SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full, "
      "bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor");
  // 具体实现依赖条件编译，因此本文件只声明 schema。

  // 把 GPTQ 权重重排到 Marlin kernel 期望的布局。
  ops.def(
      "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, "
      "SymInt size_k, SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor");
  // 具体实现依赖条件编译，因此本文件只声明 schema。

  // 把 AWQ 权重重排到 Marlin kernel 期望的布局。
  ops.def(
      "awq_marlin_repack(Tensor b_q_weight, SymInt size_k, "
      "SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor");
  // 具体实现依赖条件编译，因此本文件只声明 schema。

  // 预处理 `W-int4A-fp8` 权重，使其能直接进入 Marlin kernel。
  ops.def(
      "marlin_int4_fp8_preprocess(Tensor qweight, "
      "Tensor? qzeros_or_none, bool inplace) -> Tensor");
  // 具体实现依赖条件编译，因此本文件只声明 schema。

  // 这一组是 CUTLASS 的 `w4a8` dense / grouped GEMM 入口。
  ops.def(
      "cutlass_w4a8_mm("
      "   Tensor A,"
      "   Tensor B,"
      "   Tensor group_scales,"
      "   int    group_size,"
      "   Tensor channel_scales,"
      "   Tensor token_scales,"
      "   ScalarType? out_type,"
      "   str?   maybe_schedule"
      ") -> Tensor");
  // 把 scale 打包成 CUTLASS 侧更易消费的布局。
  ops.def("cutlass_pack_scale_fp8(Tensor scales) -> Tensor");
  // 把 INT4 权重编码并重排成 CUTLASS kernel 期望的布局。
  ops.def("cutlass_encode_and_reorder_int4b(Tensor B) -> Tensor");
  // 具体实现依赖条件编译，因此本文件只声明 schema。

  ops.def(
      "cutlass_w4a8_moe_mm("
      "   Tensor! out_tensors,"
      "   Tensor a_tensors,"
      "   Tensor b_tensors,"
      "   Tensor a_scales,"
      "   Tensor b_scales,"
      "   Tensor b_group_scales,"
      "   int b_group_size,"
      "   Tensor expert_offsets,"
      "   Tensor problem_sizes,"
      "   Tensor a_strides,"
      "   Tensor b_strides,"
      "   Tensor c_strides,"
      "   Tensor group_scale_strides,"
      "   str? maybe_schedule"
      ") -> ()");
  ops.def(
      "cutlass_encode_and_reorder_int4b_grouped(Tensor b_tensors) -> (Tensor, "
      "Tensor)");
  // 具体实现依赖条件编译，因此本文件只声明 schema。

#endif

  // 这一组 GGML 入口负责反量化、矩阵乘和 MoE 相关辅助 kernel。
  ops.def(
      "ggml_dequantize(Tensor W, int type, SymInt m, SymInt n, ScalarType? "
      "dtype) -> Tensor");
  ops.impl("ggml_dequantize", torch::kCUDA, &ggml_dequantize);

  ops.def(
      "ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, SymInt row) "
      "-> Tensor");
  ops.impl("ggml_mul_mat_vec_a8", torch::kCUDA, &ggml_mul_mat_vec_a8);

  ops.def(
      "ggml_mul_mat_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.impl("ggml_mul_mat_a8", torch::kCUDA, &ggml_mul_mat_a8);

  ops.def(
      "ggml_moe_a8(Tensor X, Tensor W, "
      "Tensor sorted_token_ids, Tensor expert_ids, Tensor "
      "num_tokens_post_padded, "
      "int type, SymInt row, SymInt top_k, SymInt tokens) -> Tensor");
  ops.impl("ggml_moe_a8", torch::kCUDA, &ggml_moe_a8);

  ops.def(
      "ggml_moe_a8_vec(Tensor X, Tensor W, "
      "Tensor topk_ids, int top_k, "
      "int type, SymInt row, SymInt tokens) -> Tensor");
  ops.impl("ggml_moe_a8_vec", torch::kCUDA, &ggml_moe_a8_vec);

  ops.def("ggml_moe_get_block_size", &ggml_moe_get_block_size);

#ifndef USE_ROCM
  // 下面这一组是 CUTLASS / DeepSeek / MLA 的专用入口，
  // 覆盖 NVFP4、MXFP8、稀疏 GEMM 和 SM100 MLA decode 等路径。
  ops.def(
      "cutlass_scaled_fp4_mm(Tensor! out, Tensor a, Tensor b,"
      "                      Tensor block_scale_a, Tensor block_scale_b,"
      "                      Tensor alpha) -> ()");
  ops.impl("cutlass_scaled_fp4_mm", torch::kCUDA, &cutlass_scaled_fp4_mm);

  ops.def(
      "cutlass_fp4_group_mm(Tensor! out, Tensor a, Tensor b,"
       " Tensor a_blockscale, Tensor b_blockscales, Tensor alphas,"
       " Tensor problem_sizes, Tensor expert_offsets, Tensor sf_offsets) -> ()");
  // 具体实现依赖条件编译，因此本文件只声明 schema。

  // 面向专家并行分组格式的 MXFP8 分块量化入口。
  ops.def(
      "mxfp8_experts_quant("
      " Tensor input, Tensor problem_sizes, Tensor expert_offsets,"
      " Tensor blockscale_offsets, Tensor! quant_output, Tensor! scale_factor)"
      " -> ()");
  // 具体实现依赖条件编译，因此本文件只声明 schema。

  // 面向专家并行分组格式的 MXFP8 grouped GEMM 入口。
  ops.def(
      "cutlass_mxfp8_grouped_mm("
      " Tensor a, Tensor b, Tensor sfa, Tensor sfb, Tensor! out,"
      " Tensor problem_sizes, Tensor expert_offsets, Tensor blockscale_offsets)"
      " -> ()");
  // 具体实现依赖条件编译，因此本文件只声明 schema。

  // `cutlass_scaled_mm` 与 `cutlass_scaled_mm_azp` 是当前最常用的 `w8a8` dense GEMM 入口，
  // 分别覆盖对称量化和带零点补偿的非对称量化路径。
  ops.def(
      "cutlass_scaled_mm(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales, Tensor? bias) -> ()");
  ops.impl("cutlass_scaled_mm", torch::kCUDA, &cutlass_scaled_mm);

  ops.def(
      "cutlass_scaled_mm_azp(Tensor! out, Tensor a,"
      "                  Tensor b, Tensor a_scales,"
      "                  Tensor b_scales, Tensor azp_adj,"
      "                  Tensor? azp, Tensor? bias) -> ()");
  ops.impl("cutlass_scaled_mm_azp", torch::kCUDA, &cutlass_scaled_mm_azp);

  // 这些 capability 查询入口会在 Python 层做算子选择时先被调用，
  // 用来判断当前设备是否满足对应 CUTLASS kernel 的运行前提。
  ops.def("cutlass_scaled_mm_supports_fp8(int cuda_device_capability) -> bool");
  ops.impl("cutlass_scaled_mm_supports_fp8", &cutlass_scaled_mm_supports_fp8);

  ops.def("cutlass_group_gemm_supported(int cuda_device_capability) -> bool");
  ops.impl("cutlass_group_gemm_supported", &cutlass_group_gemm_supported);

  ops.def(
      "cutlass_moe_mm(Tensor! out_tensors, Tensor a_tensors, Tensor b_tensors, "
      "               Tensor a_scales, Tensor b_scales, Tensor expert_offsets, "
      "               Tensor problem_sizes, Tensor a_strides, "
      "               Tensor b_strides, Tensor c_strides, bool per_act_token, "
      "               bool per_out_ch) -> ()");
  ops.impl("cutlass_moe_mm", torch::kCUDA, &cutlass_moe_mm);

  // 这组数据准备入口并不直接做 GEMM，
  // 而是先根据 `topk_ids` 或 `expert_num_tokens` 计算 experts 偏移、problem size 以及输入输出重排信息，
  // 供 fused MoE 的两次矩阵乘直接消费。
  ops.def(
      "get_cutlass_moe_mm_data(Tensor topk_ids, Tensor! expert_offsets, "
      "                        Tensor! problem_sizes1, Tensor! problem_sizes2, "
      "                        Tensor! input_permutation, "
      "                        Tensor! output_permutation, int num_experts, "
      "                        int n, int k, Tensor? blockscale_offsets) -> "
      "()");
  ops.impl("get_cutlass_moe_mm_data", torch::kCUDA, &get_cutlass_moe_mm_data);

  ops.def(
      "get_cutlass_moe_mm_problem_sizes_from_expert_offsets("
      "    Tensor expert_first_token_offset, "
      "    Tensor! problem_sizes1, "
      "    Tensor! problem_sizes2, "
      "    int n, int k, bool swap_ab) -> ()");
  ops.impl("get_cutlass_moe_mm_problem_sizes_from_expert_offsets", torch::kCUDA,
           &get_cutlass_moe_mm_problem_sizes_from_expert_offsets);

  ops.def(
      "get_cutlass_batched_moe_mm_data(Tensor! expert_offsets, "
      "                             Tensor! problem_sizes1, "
      "                             Tensor! problem_sizes2, "
      "                             Tensor expert_num_tokens, "
      "                             int num_local_experts, int padded_m, "
      "                             int n, int k) -> ()");
  ops.impl("get_cutlass_batched_moe_mm_data", torch::kCUDA,
           &get_cutlass_batched_moe_mm_data);

  ops.def(
      "cutlass_scaled_mm_supports_block_fp8(int cuda_device_capability) -> "
      "bool");
  ops.impl("cutlass_scaled_mm_supports_block_fp8",
           &cutlass_scaled_mm_supports_block_fp8);

  ops.def(
      "cutlass_sparse_scaled_mm_supported(int cuda_device_capability) -> bool");
  ops.impl("cutlass_sparse_scaled_mm_supported",
           &cutlass_sparse_scaled_mm_supported);

  // `cutlass_scaled_sparse_mm` 与 `cutlass_sparse_compress` 负责稀疏量化矩阵乘链路。
  ops.def(
      "cutlass_scaled_sparse_mm(Tensor! out, Tensor a,"
      "                         Tensor bt_nzs,"
      "                         Tensor bt_meta, Tensor a_scales,"
      "                         Tensor b_scales, Tensor? bias) -> ()");
  ops.impl("cutlass_scaled_sparse_mm", torch::kCUDA, &cutlass_scaled_sparse_mm);

  ops.def("cutlass_sparse_compress(Tensor a) -> Tensor[]");
  ops.impl("cutlass_sparse_compress", &cutlass_sparse_compress);

  // 这两项是 SM100 MLA decode 专用入口，
  // 仍然采用“先声明 schema，再按条件在实现文件中补注册”的模式。
  ops.def(
      "sm100_cutlass_mla_decode(Tensor! out, Tensor! lse, Tensor q_nope,"
      "                         Tensor q_pe, Tensor kv_c_and_k_pe_cache,"
      "                         Tensor seq_lens, Tensor page_table,"
      "                         Tensor workspace, float scale,"
      "                         int num_kv_splits) -> ()");
  // 具体实现依赖条件编译，因此本文件只声明 schema。

  ops.def(
      "sm100_cutlass_mla_get_workspace_size(int max_seq_len, int num_batches,"
      "                                     int sm_count, int num_kv_splits) "
      "-> int");
  // 具体实现依赖条件编译，因此本文件只声明 schema。

  // 下面这一组是 NVFP4 / FP4 量化与能力探测入口。
  ops.def(
      "scaled_fp4_quant(Tensor! output, Tensor input,"
      "                 Tensor! output_scale, Tensor input_scale, bool "
      "is_sf_swizzled_layout) -> ()");
  ops.impl("scaled_fp4_quant", torch::kCUDA, &scaled_fp4_quant);

  ops.def(
      "scaled_fp4_experts_quant(Tensor! output, Tensor! output_scale,"
      "Tensor input, Tensor input_global_scale, Tensor input_offset_by_experts,"
      "Tensor output_scale_offset_by_experts) -> ()");
  ops.impl("scaled_fp4_experts_quant", torch::kCUDA, &scaled_fp4_experts_quant);

  ops.def(
      "silu_and_mul_scaled_fp4_experts_quant(Tensor! output, Tensor! "
      "output_scale,"
      "Tensor input, Tensor input_global_scale, Tensor input_offset_by_experts,"
      "Tensor output_scale_offset_by_experts) -> ()");
  ops.impl("silu_and_mul_scaled_fp4_experts_quant", torch::kCUDA,
           &silu_and_mul_scaled_fp4_experts_quant);

  ops.def("cutlass_scaled_mm_supports_fp4(int cuda_device_capability) -> bool");
  ops.impl("cutlass_scaled_mm_supports_fp4", &cutlass_scaled_mm_supports_fp4);
#endif

  // 下面这一组是 GPTQ、FP8、INT8 和扫描类 kernel 的通用入口。
  // `gptq_gemm` 这里显式写 schema，而不是完全依赖 C++ 自动推断，
  // 是为了避开 meta function 注册链上的兼容问题。
  ops.def(
      "gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, "
      "Tensor b_gptq_scales, Tensor b_g_idx, bool use_exllama, bool "
      "use_v2_format, int bit) "
      "-> Tensor");
  ops.impl("gptq_gemm", torch::kCUDA, &gptq_gemm);

  // `gptq_shuffle` 用于 GPTQ 权重的后处理重排。
  ops.def("gptq_shuffle(Tensor! q_weight, Tensor q_perm, int bit) -> ()");
  ops.impl("gptq_shuffle", torch::kCUDA, &gptq_shuffle);

  // 静态 FP8 量化入口支持按 tensor、按通道、按 token 以及二维 group 缩放；
  // 当 scale 是一维时，需要用 `group_shape` 消歧到底是按通道还是按 token。
  ops.def(
      "static_scaled_fp8_quant(Tensor! result, Tensor input, Tensor scale, "
      "(int, int)? group_shape=None) -> ()");
  ops.impl("static_scaled_fp8_quant", torch::kCUDA, &static_scaled_fp8_quant);

  // 这一项会在运行时回写整 tensor 的动态 scale。
  ops.def(
      "dynamic_scaled_fp8_quant(Tensor! result, Tensor input, Tensor! scale) "
      "-> "
      "()");
  ops.impl("dynamic_scaled_fp8_quant", torch::kCUDA, &dynamic_scaled_fp8_quant);

  // 这一项会按 token 粒度回写动态 scale，适合波动更大的输入分布。
  ops.def(
      "dynamic_per_token_scaled_fp8_quant(Tensor! result, Tensor input, "
      "Tensor! scale, Tensor? scale_ub) -> "
      "()");
  ops.impl("dynamic_per_token_scaled_fp8_quant", torch::kCUDA,
           &dynamic_per_token_scaled_fp8_quant);

  // 静态 INT8 量化入口。
  ops.def(
      "static_scaled_int8_quant(Tensor! result, Tensor input, Tensor scale,"
      "Tensor? azp) -> ()");
  ops.impl("static_scaled_int8_quant", torch::kCUDA, &static_scaled_int8_quant);

  // 动态 INT8 量化入口，同时回写 scale 和可选零点补偿。
  ops.def(
      "dynamic_scaled_int8_quant(Tensor! result, Tensor input, Tensor! scale, "
      "Tensor!? azp) -> ()");
  ops.impl("dynamic_scaled_int8_quant", torch::kCUDA,
           &dynamic_scaled_int8_quant);

  // `selective_scan_fwd` 是 Mamba 状态扫描热链的核心入口。
  ops.def(
      "selective_scan_fwd(Tensor! u, Tensor! delta,"
      "Tensor! A, Tensor! B, Tensor! C,"
      "Tensor? D_, Tensor!? z_, Tensor? delta_bias_,"
      "bool delta_softplus,"
      "Tensor? query_start_loc,"
      "Tensor? cache_indices,"
      "Tensor? has_initial_state,"
      "Tensor! ssm_states,"
      "int pad_slot_id,"
      "int block_size,"
      "Tensor? block_idx_first_scheduled_token,"
      "Tensor? block_idx_last_scheduled_token,"
      "Tensor? initial_state_idx,"
      "Tensor? cu_chunk_seqlen,"
      "Tensor? last_chunk_indices) -> ()");
  ops.impl("selective_scan_fwd", torch::kCUDA, &selective_scan_fwd);

  // `hadacore_transform` 负责 Hadamard 变换类 helper。
  ops.def("hadacore_transform(Tensor! x, bool inplace) -> Tensor");

#ifndef USE_ROCM
  // 这一组 DeepGEMM / token-group 量化入口的 dummy 参数不是冗余字段，
  // 而是为了让它们在图编译与 RMSNorm 融合路径里保持正确的 schema。
  ops.def(
      "per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! "
      "output_s, "
      "int group_size, float eps, float fp8_min, float fp8_max, bool "
      "scale_ue8m0, bool dummy_is_scale_transposed, bool dummy_is_tma_aligned "
      ") -> ()");
  ops.impl("per_token_group_fp8_quant", torch::kCUDA,
           &per_token_group_quant_fp8);

  // 这里输出的是 UE8M0 打包且 TMA 对齐的 scale，
  // 专门服务 DeepGEMM 的后续消费格式。
  ops.def(
      "per_token_group_fp8_quant_packed(Tensor input, Tensor! output_q, "
      "Tensor! output_s_packed, int group_size, float eps, float fp8_min, "
      "float fp8_max) -> ()");
  ops.impl("per_token_group_fp8_quant_packed", torch::kCUDA,
           &per_token_group_quant_8bit_packed);

  // 这是 token-group 粒度的 INT8 量化版本。
  ops.def(
      "per_token_group_quant_int8(Tensor input, Tensor! output_q, Tensor! "
      "output_s, int group_size, float eps, float int8_min, float int8_max) -> "
      "()");
  ops.impl("per_token_group_quant_int8", torch::kCUDA,
           &per_token_group_quant_int8);

  // 下面两项服务 AllSpark Ampere `W8A16` fused GEMM；
  // 第一项先把权重重排到指定布局，第二项再暴露真正的 GEMM schema。
  ops.def(
      "rearrange_kn_weight_as_n32k16_order(Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_zeros, "
      "bool has_zp, Tensor! b_qweight_reorder, Tensor! b_scales_reorder, "
      "Tensor!? b_zeros_reorder, "
      "int K, int N, int N_32align) -> ()");
  // 具体实现依赖条件编译，因此本文件只声明 schema。

  ops.def(
      "allspark_w8a16_gemm(Tensor a, Tensor b_qweight, Tensor b_scales, "
      "Tensor? b_qzeros, "
      "SymInt n, SymInt group_size, SymInt sm_count, SymInt sm_version, SymInt "
      "CUBLAS_M_THRESHOLD, bool has_zp, bool n32k16_reorder) -> Tensor");
  // 具体实现依赖条件编译，因此本文件只声明 schema。
#endif
}

// ------------------------------- 注册 KV Cache 与页缓存相关算子 -------------------------------
// 这一组入口负责缓存 block 的搬运、重排、量化写入和按页 gather，
// 是 scheduler / worker 把 block table 落成真实 KV 内存布局时会直接触达的底层接口。
TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cache_ops), cache_ops) {
    // `swap_blocks` 负责在两个缓存张量之间按 block 映射做交换或拷贝。
  cache_ops.def(
      "swap_blocks(Tensor src, Tensor! dst,"
      "            int block_size_in_bytes, Tensor block_mapping) -> ()");
  cache_ops.impl("swap_blocks", torch::kCUDA, &swap_blocks);

  // 这一组 `reshape_and_cache*` 入口负责把当前 step 生成的 key / value
  // 整理成页缓存布局，并按 `slot_mapping` 写入目标 cache。
  cache_ops.def(
      "reshape_and_cache(Tensor key, Tensor value,"
      "                  Tensor! key_cache, Tensor! value_cache,"
      "                  Tensor slot_mapping,"
      "                  str kv_cache_dtype,"
      "                  Tensor k_scale, Tensor v_scale) -> ()");
  cache_ops.impl("reshape_and_cache", torch::kCUDA, &reshape_and_cache);

  cache_ops.def(
      "reshape_and_cache_flash(Tensor key, Tensor value,"
      "                        Tensor! key_cache,"
      "                        Tensor! value_cache,"
      "                        Tensor slot_mapping,"
      "                        str kv_cache_dtype,"
      "                        Tensor k_scale, Tensor v_scale) -> ()");
  cache_ops.impl("reshape_and_cache_flash", torch::kCUDA,
                 &reshape_and_cache_flash);

  cache_ops.def(
      "reshape_and_cache_flash_diffkv(Tensor key, Tensor value,"
      "                               Tensor! kv_cache,"
      "                               Tensor slot_mapping,"
      "                               str kv_cache_dtype,"
      "                               Tensor k_scale, Tensor v_scale) -> ()");
  cache_ops.impl("reshape_and_cache_flash_diffkv", torch::kCUDA,
                 &reshape_and_cache_flash_diffkv);

  // 这一组 MLA helper 会先拼接或旋转 Q/K，再把结果落入专用 cache 形态。
  cache_ops.def(
      "concat_and_cache_mla(Tensor kv_c, Tensor k_pe,"
      "                     Tensor! kv_cache,"
      "                     Tensor slot_mapping,"
      "                     str kv_cache_dtype,"
      "                     Tensor scale) -> ()");
  cache_ops.impl("concat_and_cache_mla", torch::kCUDA, &concat_and_cache_mla);

  cache_ops.def(
      "concat_and_cache_mla_rope_fused("
      "                     Tensor positions,"
      "                     Tensor! q_pe,"
      "                     Tensor! k_pe,"
      "                     Tensor kv_c,"
      "                     Tensor cos_sin_cache,"
      "                     bool is_neox,"
      "                     Tensor slot_mapping,"
      "                     Tensor! kv_cache,"
      "                     str kv_cache_dtype,"
      "                     Tensor kv_cache_scale) -> ()");
  cache_ops.impl("concat_and_cache_mla_rope_fused", torch::kCUDA,
                 &concat_and_cache_mla_rope_fused);

  // `convert_fp8` 负责把现有 cache 转成 FP8 表示。
  cache_ops.def(
      "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
      "str kv_cache_dtype) -> ()");
  cache_ops.impl("convert_fp8", torch::kCUDA, &convert_fp8);

  // 下面这一组 gather 入口负责按 block_table 把缓存块收集出来；
  // 如果源缓存是量化格式，还会在需要时顺带做反量化或升精度。
  cache_ops.def(
      "gather_and_maybe_dequant_cache(Tensor src_cache, Tensor! dst, "
      "                               Tensor block_table, Tensor cu_seq_lens, "
      "                               Tensor token_to_seq, "
      "                               int num_tokens, "
      "                               str kv_cache_dtype, "
      "                               Tensor scale, Tensor? seq_starts) -> ()");
  cache_ops.impl("gather_and_maybe_dequant_cache", torch::kCUDA,
                 &gather_and_maybe_dequant_cache);

  cache_ops.def(
      "cp_gather_cache(Tensor src_cache, Tensor! dst, Tensor block_table, "
      "Tensor cu_seq_lens, int batch_size, Tensor? seq_starts) -> ()");
  cache_ops.impl("cp_gather_cache", torch::kCUDA, &cp_gather_cache);

  cache_ops.def(
      "gather_paged_kv_cache(Tensor key_cache, Tensor value_cache, "
      "Tensor! gathered_key, Tensor! gathered_value, Tensor block_table, "
      "Tensor cu_seq_lens, int batch_size, Tensor? seq_starts) -> ()");
  cache_ops.impl("gather_paged_kv_cache", torch::kCUDA,
                 &gather_paged_kv_cache);

  cache_ops.def(
      "cp_gather_and_upconvert_fp8_kv_cache(Tensor src_cache, Tensor! dst, "
      "Tensor block_table, Tensor seq_lens, Tensor workspace_starts, int "
      "batch_size) -> ()");
  cache_ops.impl("cp_gather_and_upconvert_fp8_kv_cache", torch::kCUDA,
                 &cp_gather_and_upconvert_fp8_kv_cache);

  cache_ops.def(
      "indexer_k_quant_and_cache(Tensor k, Tensor! kv_cache, Tensor "
      "slot_mapping, "
      "int quant_block_size, str kv_cache_dtype) -> ()");
  cache_ops.impl("indexer_k_quant_and_cache", torch::kCUDA,
                 &indexer_k_quant_and_cache);

  cache_ops.def(
      "concat_mla_q(Tensor ql_nope, Tensor q_pe, Tensor! q_out) -> ()");
  cache_ops.impl("concat_mla_q", torch::kCUDA, &concat_mla_q);

  cache_ops.def(
      "cp_gather_indexer_k_quant_cache(Tensor kv_cache, Tensor! dst_k, Tensor! "
      "dst_scale, Tensor block_table, Tensor cu_seq_lens) -> ()");
  cache_ops.impl("cp_gather_indexer_k_quant_cache", torch::kCUDA,
                 &cp_gather_indexer_k_quant_cache);
}


// ------------------------------- 注册 CUDA 设备信息查询辅助入口 -------------------------------
// 这一小段不参与模型热链计算，
// 主要给 Python 层做设备能力判断和 kernel 选择时提供只读查询接口。
TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _cuda_utils), cuda_utils) {
  // 查询指定设备属性值。
  cuda_utils.def("get_device_attribute(int attribute, int device_id) -> int");
  cuda_utils.impl("get_device_attribute", &get_device_attribute);

  // 查询单个 block 可用的最大 shared memory。
  cuda_utils.def(
      "get_max_shared_memory_per_block_device_attribute(int device_id) -> int");
  cuda_utils.impl("get_max_shared_memory_per_block_device_attribute",
                  &get_max_shared_memory_per_block_device_attribute);
}


// ------------------------------- 注册自定义 all-reduce 与共享缓冲算子 -------------------------------
// 这组入口负责初始化通信句柄、注册共享缓冲区以及触发自定义 all-reduce；
// 分布式或张量并行路径会通过这里把底层通信能力暴露给 Python 层。
TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _custom_ar), custom_ar) {
  custom_ar.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool fully_connected) -> int");
  custom_ar.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);
  custom_ar.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  custom_ar.impl("all_reduce", torch::kCUDA, &all_reduce);

  custom_ar.def("dispose", &dispose);
  custom_ar.def("meta_size", &meta_size);

  custom_ar.def("register_buffer", &register_buffer);
  custom_ar.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
  custom_ar.def("register_graph_buffers", &register_graph_buffers);

  custom_ar.def("allocate_shared_buffer_and_handle",
                &allocate_shared_buffer_and_handle);
  custom_ar.def("open_mem_handle(Tensor mem_handle) -> int", &open_mem_handle);
  custom_ar.impl("open_mem_handle", torch::kCPU, &open_mem_handle);

  custom_ar.def("free_shared_buffer", &free_shared_buffer);
#ifdef USE_ROCM
  // ROCm 场景下额外补 Quick Reduce 相关入口。
  custom_ar.def(
      "qr_all_reduce(int fa, Tensor inp, Tensor out, int quant_level, bool "
      "cast_bf2half) -> ()");
  custom_ar.impl("qr_all_reduce", torch::kCUDA, &qr_all_reduce);

  custom_ar.def("init_custom_qr", &init_custom_qr);
  custom_ar.def("qr_destroy", &qr_destroy);

  custom_ar.def("qr_get_handle", &qr_get_handle);

  custom_ar.def("qr_open_handles(int _fa, Tensor[](b!) handles) -> ()");
  custom_ar.impl("qr_open_handles", torch::kCPU, &qr_open_handles);

  // 查询 Quick Reduce 当前支持的最大输入字节数。
  custom_ar.def("qr_max_size", &qr_max_size);
#endif
}


REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
