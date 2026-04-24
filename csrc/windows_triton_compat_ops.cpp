// 本文件提供 Windows 环境下的一组原生兜底算子。
// 当 Triton 或专用 CUDA kernel 在当前平台不可用时，Python 层会回退到这里的
// ATen/CUDA 参考实现，以保证接口语义、张量形状和状态更新流程保持一致。
#include <torch/all.h>

#include "ops.h"

#include <ATen/ops/scaled_dot_product_attention.h>

#include <algorithm>
#include <limits>
#include <vector>

namespace {

    constexpr double kLog2E = 1.4426950408889634;
    constexpr double kLogE2 = 0.6931471805599453;

// ------------------------------- 基础激活与位置编码辅助逻辑 -------------------------------

    torch::Tensor apply_gate_activation(const torch::Tensor &x,
                                        const std::string &activation) {
        if (activation == "swish" || activation == "silu") {
            // x: [..., hidden]
            return x * torch::sigmoid(x);
        }
        if (activation == "sigmoid") {
            return torch::sigmoid(x);
        }
        TORCH_CHECK(false, "Unsupported gating activation: ", activation);
        return x;
    }

    torch::Tensor apply_optional_activation(const torch::Tensor &x,
                                            const std::string &activation) {
        if (activation.empty()) {
            return x;
        }
        return apply_gate_activation(x, activation);
    }

    torch::Tensor softplus_with_threshold(const torch::Tensor &x, double beta,
                                          double threshold) {
        // beta_x: 与 x 同形状
        auto beta_x = x * beta;
        return torch::where(beta_x <= threshold, torch::log1p(torch::exp(beta_x)) / beta,
                            x);
    }

    // 将三路 MRoPE cache 重新整理成后续旋转时可直接广播的布局。
    // 这里同时兼容交织和非交织两种缓存格式，避免 Python 层再额外搬运数据。
    torch::Tensor prepare_mrope_cache(const torch::Tensor &cache,
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
            // merged: [num_tokens, rotary_dim/2]
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

        // splits[i]: [3, num_tokens, section_i]
        auto splits = cache.split_with_sizes({section_t, section_h, section_w}, -1);
        return torch::cat(
                {splits[0].select(0, 0), splits[1].select(0, 1), splits[2].select(0, 2)},
                -1);
    }

    torch::Tensor apply_rotary_emb_native(const torch::Tensor &x,
                                          const torch::Tensor &cos,
                                          const torch::Tensor &sin,
                                          bool is_neox) {
        // cos/sin: [T, rotary_dim/2] -> [T, 1, rotary_dim/2]
        auto cos_expanded = cos.unsqueeze(1).to(x.scalar_type());
        auto sin_expanded = sin.unsqueeze(1).to(x.scalar_type());

        if (is_neox) {
            // x1/x2: [T, H, rotary_dim/2]
            auto x1 = x.slice(-1, 0, x.size(-1) / 2);
            auto x2 = x.slice(-1, x.size(-1) / 2, x.size(-1));
            auto o1 = x1 * cos_expanded - x2 * sin_expanded;
            auto o2 = x2 * cos_expanded + x1 * sin_expanded;
            return torch::cat({o1, o2}, -1);
        }

        // reshaped: [T, H, rotary_dim/2, 2]
        auto reshaped = x.reshape({x.size(0), x.size(1), x.size(2) / 2, 2});
        auto x1 = reshaped.select(-1, 0);
        auto x2 = reshaped.select(-1, 1);
        auto o1 = x1 * cos_expanded - x2 * sin_expanded;
        auto o2 = x2 * cos_expanded + x1 * sin_expanded;
        return torch::stack({o1, o2}, -1).reshape_as(x);
    }

    // 将标量或 1D 的按行参数 value 规范化为形状 [B] 的连续 Tensor，其中 B = logits.size(0)
    torch::Tensor normalize_rowwise_param(const torch::Tensor &value,
                                          const torch::Tensor &logits,
                                          torch::ScalarType dtype) {
        // 检查 value 必须是已定义 Tensor，不能是默认构造的 undefined Tensor
        TORCH_CHECK(value.defined(), "row-wise parameter tensor must be defined");

        // 检查 value 只能是标量 [] 或一维 Tensor [N]，不能是二维及以上
        TORCH_CHECK(value.dim() <= 1,
                    "row-wise parameter tensor must be scalar or 1D");

        // 将 value 转到 logits 所在 device，并转成指定 dtype；形状保持为 [] 或 [N]
        auto normalized =
                value.to(logits.device(), dtype, false, false)
                                // 保证内存连续，方便后续 view(-1) 和 CUDA kernel 按连续数组访问
                        .contiguous()
                                // 将标量 [] 变成 [1]，将一维 [N] 保持为 [N]
                        .view(-1);

        // 如果 normalized 是单元素 [1]，并且 batch size B != 1，则认为它是全 batch 共用的标量参数
        if (normalized.numel() == 1 && logits.size(0) != 1) {
            // 将 [1] 扩展为 [B]，其中 B = logits.size(0)，逻辑值为 [v, v, ..., v]
            normalized = normalized.expand({logits.size(0)})
                            // expand 可能产生 stride=0 的非连续视图，这里复制成真正连续的 [B]
                    .contiguous();
        }

        // 检查最终 normalized 的元素个数必须等于 batch size B，即 normalized.shape == [logits.size(0)]
        TORCH_CHECK(normalized.numel() == logits.size(0),
                    "row-wise parameter tensor size must match batch size");

        // 返回形状为 [B]、device 与 logits 一致、dtype 为指定 dtype、内存连续的参数 Tensor
        return normalized;
    }

    // 保留“累计概率刚刚超过 p 的那个 token”，把它后面的 token 都 mask 掉。
    torch::Tensor build_shifted_top_p_mask(const torch::Tensor &probs_cumsum,
                                           const torch::Tensor &resolved_p) {
        // probs_cumsum: [B, V]，每一行是排序后概率的累积和
        // resolved_p:   [B]，每个 batch/request 对应一个 top_p 阈值

        // resolved_p.unsqueeze(1): [B] -> [B, 1]
        // probs_cumsum > resolved_p.unsqueeze(1): [B, V] > [B, 1]，广播得到 [B, V]
        // top_p_mask[b, v] = true 表示第 b 行中，当前位置累计概率已经超过 top_p 阈值
        auto top_p_mask = probs_cumsum > resolved_p.unsqueeze(1);

        // 如果 vocab 维度 V > 1，才需要做右移操作
        if (probs_cumsum.size(1) > 1) {
            // clone 一份原始 mask，避免后面原地 copy_ 时读写同一个 Tensor 导致覆盖问题
            auto shifted = top_p_mask.clone();

            // 将第 0 到 V-2 列的 mask 拷贝到第 1 到 V-1 列
            // 等价于：
            // top_p_mask[:, 1:V] = shifted[:, 0:V-1]
            // 也就是把 mask 整体向右移动一位
            top_p_mask.slice(1, 1, probs_cumsum.size(1))
                    .copy_(shifted.slice(1, 0, probs_cumsum.size(1) - 1));
        }

        // 第 0 列强制设为 false
        // 保证每一行至少保留概率最大的那个 token，避免所有 token 都被过滤
        top_p_mask.select(1, 0).fill_(false);

        // 返回形状 [B, V] 的 bool mask
        // true  表示该 token 要被过滤掉
        // false 表示该 token 可以保留
        return top_p_mask;
    }

    // 尝试使用 CUDA top-k-per-row kernel 对 logits 原地执行 top-k 过滤；成功处理返回 true，不适用则返回 false
    bool try_apply_top_k_only_with_cuda_topk_per_row(torch::Tensor &logits,
                                                     const torch::Tensor &resolved_k,
                                                     double mask_value) {
        // logits: [B, V]，B=batch size，V=vocab size；该优化路径只支持 float32 且连续内存布局
        if (logits.scalar_type() != torch::kFloat32 || !logits.is_contiguous()) {
            // 如果 dtype 不是 float32 或 logits 不是 contiguous，则交给外层 fallback 逻辑处理
            return false;
        }

        // resolved_k: [B]；no_top_k_mask: [B]；当 resolved_k[b] == V 时，表示第 b 行不需要 top-k 过滤
        auto no_top_k_mask = resolved_k == logits.size(1);

        // 如果所有 batch 行都是 k == V，则所有 token 都保留，无需修改 logits
        if (no_top_k_mask.all().item<bool>()) {
            // top-k-only 已经处理完成，只是没有任何元素需要被 mask
            return true;
        }

        // effective_k: [B]；把不需要过滤的行的 k 临时改成 1，避免后续 max_k 被 V 拉大造成多余 top-k 计算
        auto effective_k = resolved_k.masked_fill(no_top_k_mask, 1);

        // max_k: 标量；当前 batch 内实际需要计算的最大 top-k 数量
        const int64_t max_k = effective_k.max().item<int64_t>();

        // topk_indices: [B, max_k]，int32，存放每一行 logits 中 top max_k 的 vocab 下标
        auto topk_indices = torch::empty(
                {logits.size(0), max_k},
                torch::TensorOptions().dtype(torch::kInt32).device(logits.device()));

        // seq_lens: [B]，int32，每一行有效长度都是 V = logits.size(1)
        auto seq_lens = torch::full(
                {logits.size(0)}, logits.size(1),
                torch::TensorOptions().dtype(torch::kInt32).device(logits.device()));

        // 调用自定义 CUDA kernel：对 logits: [B, V] 的每一行找 top max_k，并把下标写入 topk_indices: [B, max_k]
        top_k_per_row_decode(logits, 1, seq_lens, topk_indices, logits.size(0),
                             logits.stride(0), logits.stride(1), max_k);

        // k_index: [B, 1]；每行第 k 大元素在 0-based 排序数组中的位置是 k - 1
        auto k_index = effective_k.sub(1).unsqueeze(1);

        // topk_indices_long: [B, max_k]，gather 要求 index 通常为 int64，因此把 int32 下标转为 int64
        auto topk_indices_long = topk_indices.to(torch::kLong);

        // topk_values: [B, max_k]；根据 topk_indices 从 logits: [B, V] 中取出每行 top max_k 的实际 logit 值
        auto topk_values = logits.gather(1, topk_indices_long);

        // topk_values_sorted: [B, max_k]；按 dim=1 对每行 topk_values 降序排序，确保能准确取第 k 大值
        auto topk_values_sorted = std::get<0>(torch::sort(topk_values, 1, true));

        // top_k_threshold: [B, 1]；每一行自己的 top-k 阈值，即第 k 大 logit 值
        auto top_k_threshold = topk_values_sorted.gather(1, k_index);

        // 对原本 k == V 的行，把阈值设置成 mask_value；若 mask_value 为 -inf，则 logits < -inf 恒为 false，不会过滤该行
        top_k_threshold.masked_fill_(no_top_k_mask.unsqueeze(1), mask_value);

        // logits: [B, V]，top_k_threshold: [B, 1]；广播比较后将小于本行阈值的 token 原地填成 mask_value
        logits.masked_fill_(logits < top_k_threshold, mask_value);

        // 返回 true 表示 top-k-only 过滤已在当前函数中成功完成
        return true;
    }

    // 仅应用 top-p 过滤：对 logits 原地修改；每行保留累计概率达到 top_p 所需的最小候选集合，其余填成 mask_value
    void apply_top_p_only_iterative_topk_precompiled(torch::Tensor &logits,
                                                     const torch::Tensor &resolved_p,
                                                     double mask_value) {
        // logits: [B, V]，B=batch size，V=vocab size；该函数要求 logits 必须是二维
        TORCH_CHECK(logits.dim() == 2,
                    "apply_top_p_only_iterative_topk_precompiled expects 2D logits");

        // resolved_p: [B]；每个 batch 行对应一个 top_p 参数
        TORCH_CHECK(resolved_p.dim() == 1,
                    "apply_top_p_only_iterative_topk_precompiled expects 1D p");

        // 检查 logits 的 batch size 必须和 resolved_p 的元素个数一致
        TORCH_CHECK(logits.size(0) == resolved_p.numel(),
                    "apply_top_p_only_iterative_topk_precompiled expects matching "
                    "batch size");

        // no_top_p_mask: [B]；当 p >= 1.0 时，表示该行不需要 top-p 过滤，保留所有 token
        auto no_top_p_mask = resolved_p >= 1.0;

        // 如果所有行的 p 都 >= 1.0，则没有任何行需要过滤，直接返回
        if (no_top_p_mask.all().item<bool>()) {
            return;
        }

        // active_rows: [B_active]；找出需要执行 top-p 过滤的行号，即 p < 1.0 的 batch 行
        auto active_rows = torch::nonzero(~no_top_p_mask).view(-1);

        // active_logits: [B_active, V]；从 logits: [B, V] 中取出需要过滤的行
        auto active_logits = logits.index_select(0, active_rows);

        // active_p: [B_active]；取出需要过滤行对应的 top_p 参数
        auto active_p = resolved_p.index_select(0, active_rows);

        // vocab_size: 标量；V = active_logits.size(1)，表示词表大小
        const int64_t vocab_size = active_logits.size(1);

        // current_k: 标量；迭代 top-k 的初始候选数量，最多从 128 个 token 开始
        int64_t current_k = std::min<int64_t>(128, vocab_size);

        // active_logsumexp: [B_active, 1]；每行 logits 的 log(sum(exp(logits)))，用于把 topk logits 转成全词表归一化概率
        auto active_logsumexp =
                torch::logsumexp(active_logits.to(torch::kFloat32), 1, true);

        // topk_values: [B_active, current_k]；每轮保存每行 top current_k 的 logit 值
        torch::Tensor topk_values;

        // topk_indices: [B_active, current_k]；每轮保存每行 top current_k 的 vocab 下标
        torch::Tensor topk_indices;

        // probs_cumsum: [B_active, current_k]；每轮保存 top current_k 概率按降序排列后的累计概率
        torch::Tensor probs_cumsum;

        // 迭代扩大 current_k，直到 top current_k 的累计概率足以覆盖每行 top_p，或者已经覆盖全词表
        while (true) {
            // topk_values/topk_indices: [B_active, current_k]；dim=1 上取每行最大的 current_k 个 token，并按值降序排序
            auto topk = torch::topk(active_logits, current_k, 1, true, true);

            // topk_values: [B_active, current_k]；每行 top current_k 的 logit 值，通常已按从大到小排列
            topk_values = std::get<0>(topk);

            // topk_indices: [B_active, current_k]；每行 top current_k 对应的原始 vocab 下标
            topk_indices = std::get<1>(topk);

            // topk_probs: [B_active, current_k]；
            // 用 exp(topk_logit - logsumexp(all_logits)) 得到这些 top-k token 在完整 vocab softmax 下的概率
            auto topk_probs =
                    torch::exp(topk_values.to(torch::kFloat32) - active_logsumexp);

            // probs_cumsum: [B_active, current_k]；对 topk_probs 沿 vocab 候选维做累计和
            probs_cumsum = torch::cumsum(topk_probs, -1);

            // 如果 current_k 已经等于 vocab_size，说明候选已覆盖全词表，可以停止
            // 或者每一行 top current_k 的累计概率都 >= active_p，说明当前候选已经足够构造 top-p 集合，可以停止
            if (current_k == vocab_size ||
                probs_cumsum.select(1, current_k - 1).ge(active_p).all().item<bool>()) {
                break;
            }

            // 如果当前 top current_k 的累计概率还不够覆盖 top_p，则把候选数量翻倍，但不能超过 vocab_size
            current_k = std::min<int64_t>(current_k * 2, vocab_size);
        }

        // top_p_mask: [B_active, current_k]；
        // true 表示该 top-k 候选位置需要被过滤，false 表示保留
        // build_shifted_top_p_mask 会保留“累计概率首次超过 p 的那个 token”
        auto top_p_mask = build_shifted_top_p_mask(probs_cumsum, active_p);

        // filtered_values: [B_active, current_k]；
        // 在 topk_values 中把 top-p 集合之外的候选填成 mask_value，top-p 集合内的候选保留原 logit
        auto filtered_values = topk_values.masked_fill(top_p_mask, mask_value);

        // active_logits: [B_active, V]；先把所有 active 行全部填成 mask_value，相当于默认过滤所有 token
        active_logits.fill_(mask_value);

        // active_logits: [B_active, V]；
        // 根据 topk_indices: [B_active, current_k] 把 filtered_values 写回原 vocab 位置
        // top-p 内 token 写回原 logit，top-p 外 top-k 候选写回 mask_value
        active_logits.scatter_(1, topk_indices, filtered_values);

        // logits: [B, V]；
        // 把处理后的 active_logits: [B_active, V] 写回 logits 对应的 active_rows 行，完成原地更新
        logits.index_copy_(0, active_rows, active_logits);
    }

    // 先执行 top-k，再在 top-k 候选集合内部执行 top-p；适合 k 较小的场景，会原地修改 logits
    void apply_top_k_then_top_p_small_k_precompiled(
            torch::Tensor &logits, const torch::Tensor &resolved_k,
            const torch::Tensor &resolved_p, double mask_value) {
        // logits: [B, V]，B=batch size，V=vocab size；该函数要求 logits 必须是二维
        TORCH_CHECK(logits.dim() == 2,
                    "apply_top_k_then_top_p_small_k_precompiled expects 2D logits");

        // resolved_k: [B]，resolved_p: [B]；每一行分别对应自己的 top_k 和 top_p 参数
        TORCH_CHECK(resolved_k.dim() == 1 && resolved_p.dim() == 1,
                    "apply_top_k_then_top_p_small_k_precompiled expects 1D inputs");

        // 检查 logits 的 batch size 必须同时等于 resolved_k 和 resolved_p 的元素个数
        TORCH_CHECK(logits.size(0) == resolved_k.numel() &&
                    logits.size(0) == resolved_p.numel(),
                    "apply_top_k_then_top_p_small_k_precompiled expects matching "
                    "batch size");

        // 要求所有行的 k 都小于 vocab size；也就是每一行都真的需要 top-k 过滤，而不是 k == V
        TORCH_CHECK(resolved_k.max().item<int64_t>() < logits.size(1),
                    "apply_top_k_then_top_p_small_k_precompiled expects k < vocab");

        // max_k: 标量；当前 batch 中最大的 top_k，用于统一取每行 top max_k
        const int64_t max_k = resolved_k.max().item<int64_t>();

        // topk: tuple(values, indices)；
        // 对 logits: [B, V] 在 dim=1 上取每行最大的 max_k 个 token，并按 logit 值降序排列
        auto topk = torch::topk(logits, max_k, 1, true, true);

        // topk_values: [B, max_k]；每一行 top max_k 的 logit 值，通常按从大到小排序
        auto topk_values = std::get<0>(topk);

        // topk_indices: [B, max_k]；每一行 top max_k 对应的原始 vocab 下标
        auto topk_indices = std::get<1>(topk);

        // positions: [1, max_k]；内容为 [0, 1, 2, ..., max_k - 1]，用于和每行自己的 k 比较
        auto positions = torch::arange(max_k, resolved_k.options()).unsqueeze(0);

        // valid_k_mask: [B, max_k]；
        // valid_k_mask[b, j] = true 表示第 b 行第 j 个 top-k 候选仍在该行自己的 resolved_k[b] 范围内
        auto valid_k_mask = positions < resolved_k.unsqueeze(1);

        // filtered_values: [B, max_k]；
        // 对于某一行，如果它自己的 k 小于 max_k，则把超过该行 k 范围的位置先填成 mask_value
        auto filtered_values = topk_values.masked_fill(~valid_k_mask, mask_value);

        // probs_desc: [B, max_k]；
        // 在 top-k 后的候选集合内部做 softmax，得到每行 top-k 候选的归一化概率
        // 注意这里的概率是在 top-k 截断后的集合内重新归一化，不是全 vocab softmax 概率
        auto probs_desc = filtered_values.softmax(-1, torch::kFloat32);

        // probs_cumsum: [B, max_k]；
        // 对按 logit 降序排列的 top-k 概率做累计和，用于后续 top-p 判断
        auto probs_cumsum = torch::cumsum(probs_desc, -1);

        // top_p_mask: [B, max_k]；
        // true 表示该位置需要被 top-p 过滤；false 表示保留
        // build_shifted_top_p_mask 会保留“累计概率首次超过 p 的那个 token”
        auto top_p_mask = build_shifted_top_p_mask(probs_cumsum, resolved_p);

        // filtered_values: [B, max_k]；
        // 同时应用 top-p mask 和 top-k 有效范围 mask：
        // 1. top_p_mask 为 true 的位置填成 mask_value
        // 2. valid_k_mask 为 false 的位置也填成 mask_value
        filtered_values.masked_fill_(top_p_mask | (~valid_k_mask), mask_value);

        // logits: [B, V]；先把整张 logits 全部填成 mask_value，相当于默认过滤所有 token
        logits.fill_(mask_value);

        // logits: [B, V]；
        // 根据 topk_indices: [B, max_k] 把 filtered_values: [B, max_k] 写回原始 vocab 位置
        // 最终只保留同时通过 top-k 和 top-p 的 token，其余位置保持 mask_value
        logits.scatter_(1, topk_indices, filtered_values);
    }

// ------------------------------- 状态索引与缓存搬运辅助逻辑 -------------------------------

    int64_t resolve_state_index(const std::optional<torch::Tensor> &indices,
                                int64_t seq_idx, int64_t token_offset) {
        if (!indices.has_value()) {
            return -2;
        }

        const auto &idx = indices.value();
        if (idx.dim() == 1) {
            const int64_t flat_index = seq_idx + token_offset;
            if (flat_index >= idx.numel()) {
                return -1;
            }
            return idx[flat_index].item<int64_t>();
        }
        return idx.index({seq_idx, token_offset}).item<int64_t>();
    }

    int64_t resolve_cache_line(const std::optional<torch::Tensor> &cache_indices,
                               int64_t seq_idx, int64_t block_offset = 0) {
        if (!cache_indices.has_value()) {
            return seq_idx;
        }

        const auto &idx = cache_indices.value();
        if (idx.dim() == 1) {
            return idx[seq_idx].item<int64_t>();
        }
        return idx.index({seq_idx, block_offset}).item<int64_t>();
    }

    torch::Tensor move_tensor_to_target_device(const torch::Tensor &src,
                                               const torch::Tensor &dst) {
        // contiguous_src: 与 src 同形状
        auto contiguous_src = src.contiguous();
        if (contiguous_src.device() == dst.device()) {
            return contiguous_src;
        }

        // dst_src: 与 src 同形状，但设备对齐到 dst.device()
        auto dst_src = torch::empty(
                contiguous_src.sizes(),
                contiguous_src.options().device(dst.device()));
        dst_src.copy_(contiguous_src, true);
        return dst_src;
    }

    torch::Tensor move_slot_ids_to_target_device(const torch::Tensor &slot_ids,
                                                 const torch::Tensor &dst) {
        // slot_ids_long: [N_slot]
        auto slot_ids_long = slot_ids.to(torch::kLong).contiguous();
        if (slot_ids_long.device() == dst.device()) {
            return slot_ids_long;
        }

        // slot_ids_device: [N_slot]
        auto slot_ids_device = torch::empty(
                slot_ids_long.sizes(),
                slot_ids_long.options().device(dst.device()));
        slot_ids_device.copy_(slot_ids_long, true);
        return slot_ids_device;
    }

    void batch_copy_into_slots(const torch::Tensor &slot_ids,
                               const torch::Tensor &src,
                               torch::Tensor &dst,
                               const char *field_name) {
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
        // index_copy_: src [N_slot, ...] -> dst[slot_ids, ...]
        dst.index_copy_(0, slot_ids_device, src_device);
    }

    void batch_copy_into_slots_optional(
            const torch::Tensor &slot_ids,
            const std::optional<torch::Tensor> &src,
            const std::optional<torch::Tensor> &dst,
            const char *field_name) {
        if (!src.has_value() && !dst.has_value()) {
            return;
        }
        TORCH_CHECK(src.has_value() && dst.has_value(), field_name,
                    " source and destination must either both exist or both be None");
        auto dst_tensor = dst.value();
        batch_copy_into_slots(slot_ids, src.value(), dst_tensor, field_name);
    }

    torch::Tensor load_initial_history(const torch::Tensor &state,
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

        // history: [D, history_len_actual]
        auto history = state.slice(-1, start_offset, start_offset + history_len);
        if (history.size(-1) == history_len) {
            return history;
        }

        // padded: [D, history_len]
        auto padded = torch::zeros({state.size(0), history_len}, state.options());
        if (history.numel() > 0) {
            padded.slice(-1, 0, history.size(-1)).copy_(history);
        }
        return padded;
    }

    torch::Tensor update_conv_state_ref(const torch::Tensor &state,
                                        const torch::Tensor &seq_tokens,
                                        int64_t shift_tokens) {
        const int64_t state_len = state.size(-1);
        if (state_len == 0) {
            return state;
        }

        const int64_t safe_shift = std::max<int64_t>(shift_tokens, 0);
        // retained_state/updated: [D, <=state_len] / [D, <=state_len + T]
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
            const torch::Tensor &seq_tokens, const torch::Tensor &weight,
            const std::optional<torch::Tensor> &bias, const torch::Tensor &history,
            const std::string &activation) {
        // history/seq_tokens: [D, T_hist] / [D, T_seq]
        auto combined = torch::cat({history, seq_tokens}, -1);
        // windows: [D, T_seq, K]
        auto windows = combined.unfold(-1, weight.size(1), 1);
        auto output =
                (windows * weight.unsqueeze(1)).sum(-1);
        if (bias.has_value()) {
            output = output + bias.value().unsqueeze(1);
        }
        return apply_optional_activation(output, activation);
    }

    torch::Tensor l2norm_last_dim(const torch::Tensor &x) {
        // x: [..., K]
        return x / torch::sqrt((x * x).sum(-1, true) + 1e-6);
    }

    torch::Tensor expand_qk_heads(const torch::Tensor &x, int64_t target_heads) {
        if (x.size(2) == target_heads) {
            return x;
        }
        TORCH_CHECK(target_heads % x.size(2) == 0, "Cannot expand ", x.size(2),
                    " query/key heads to ", target_heads, " value heads.");
        // x: [B, T, Hg, K] -> [B, T, H, K]
        return x.repeat_interleave(target_heads / x.size(2), 2);
    }

    torch::Tensor align_scale_tail_dims(const torch::Tensor &scale,
                                        int64_t target_dim) {
        // scale: [..] -> [..., 1, 1]
        auto aligned = scale;
        while (aligned.dim() < target_dim) {
            aligned = aligned.unsqueeze(-1);
        }
        return aligned;
    }

    torch::Tensor group_broadcast_reference(const torch::Tensor &scale,
                                            c10::IntArrayRef target_shape) {
        // broadcasted: 对齐到 target_shape 的前缀广播视图
        auto broadcasted = align_scale_tail_dims(scale, target_shape.size());
        for (int64_t dim = 0; dim < static_cast<int64_t>(target_shape.size()); ++dim) {
            const int64_t current = broadcasted.size(dim);
            const int64_t target = target_shape[dim];
            if (current == target || current == 1) {
                continue;
            }
            TORCH_CHECK(target % current == 0,
                        "Scale cannot be group-broadcast to target shape");
            // repeat_interleave: 把 group 维复制到目标大小
            broadcasted = broadcasted.repeat_interleave(target / current, dim);
        }
        return broadcasted;
    }

    torch::Tensor dequantize_batched_moe_tensor_reference(
            const torch::Tensor &tensor, const std::optional<torch::Tensor> &scale,
            bool per_act_token_quant) {
        auto tensor_f = tensor.to(torch::kFloat32);
        if (!scale.has_value()) {
            return tensor_f;
        }

        // scale_f: [1] 或与 group/token 对齐的缩放张量
        auto scale_f =
                scale.value().to(tensor.device(), torch::kFloat32, false, false)
                        .contiguous();
        if (per_act_token_quant || scale_f.numel() == 1) {
            return tensor_f * align_scale_tail_dims(scale_f, tensor.dim());
        }
        return tensor_f * group_broadcast_reference(scale_f, tensor.sizes());
    }

}  // namespace

// ------------------------------- Attention 与 LSE 聚合兜底算子 -------------------------------

// 重新计算当前 cp 分片应当保留的缩放系数，并返回合并后的全局 LSE。
// 这条兜底路径不重新执行 attention，只根据各分片的 lse 恢复出正确的归一化比例。
torch::Tensor correct_attn_out_precompiled(torch::Tensor &out,
                                           const torch::Tensor &lses,
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
    // neg_inf_scalar: 标量 -inf，占位到 lses dtype/device
    auto neg_inf_scalar = torch::full({}, neg_inf, lses.options());

    // 先把 NaN/Inf 压成负无穷，避免后续最大值和指数归一化被异常值污染。
    // sanitized: [N, B, H]
    auto sanitized =
            torch::where(torch::isnan(lses) | torch::isinf(lses), neg_inf_scalar, lses);
    // lse_max: [B, H]
    auto lse_max = sanitized.amax(0);
    // 全是 -inf 的位置改回 0，避免后续减法继续传播无效值。
    lse_max = torch::where(lse_max == neg_inf_scalar, torch::zeros_like(lse_max),
                           lse_max);

    // 先做数值稳定的 log-sum-exp 合并，再恢复成与输入底数一致的全局 lse。
    // shifted: [N, B, H]
    auto shifted = sanitized - lse_max.unsqueeze(0);
    torch::Tensor final_lse;
    if (is_lse_base_on_e) {
        // final_lse: [B, H]
        final_lse = torch::log(torch::exp(shifted).sum(0)) + lse_max;
    } else {
        // final_lse: [B, H]
        final_lse =
                torch::log(torch::exp(shifted * kLogE2).sum(0)) * kLog2E + lse_max;
    }

    // lse: [B, H]
    auto lse = torch::empty_strided({out.size(0), out.size(1)},
                                    {lses.stride(1), lses.stride(2)},
                                    lses.options());
    // 把最终 lse 拷回和上游 kernel 一致的 stride 布局。
    lse.copy_(final_lse);

    // 当前 rank 只保留属于自己的局部贡献，所以要按局部 lse 与全局 lse 的差值回缩。
    // local_lse: [B, H]
    auto local_lse = sanitized.select(0, cp_rank) - lse;
    // 无效差值继续压成 -inf，保证 exp 后正好得到 0 权重。
    local_lse =
            torch::where(torch::isnan(local_lse) | torch::isinf(local_lse),
                         torch::full({}, neg_inf, local_lse.options()), local_lse);
    // factor: [B, H]
    auto factor = is_lse_base_on_e ? torch::exp(local_lse)
                                   : torch::exp(local_lse * kLogE2);
    // out: [B, H, D]，按 head 维权重缩放当前 rank 输出。
    out.mul_(factor.unsqueeze(-1).to(out.scalar_type()));
    return lse;
}

// 合并多个上下文并行分片的输出和 lse，得到最终 attention 输出。
// 兜底实现沿用 log-sum-exp 权重重标定思路，保证和正式 kernel 的归一化语义一致。
std::tuple<torch::Tensor, torch::Tensor> dcp_lse_combine_precompiled(
        const torch::Tensor &recv_output, const torch::Tensor &recv_lse,
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
    // neg_inf_scalar: 标量 -inf，占位到 recv_lse dtype/device
    auto neg_inf_scalar = torch::full({}, neg_inf, recv_lse.options());

    // 先合成全局 lse，并同步得到每个分片对最终结果的相对权重。
    // sanitized: [N, B, H]
    auto sanitized = torch::where(
            torch::isnan(recv_lse) | torch::isinf(recv_lse), neg_inf_scalar, recv_lse);
    // lse_max: [B, H]
    auto lse_max = sanitized.amax(0);
    // 全无效位置回填 0，避免后续归一化继续扩散 -inf。
    lse_max = torch::where(lse_max == neg_inf_scalar, torch::zeros_like(lse_max),
                           lse_max);

    // shifted: [N, B, H]
    auto shifted = sanitized - lse_max.unsqueeze(0);
    torch::Tensor weights;
    torch::Tensor global_lse;
    if (is_lse_base_on_e) {
        // weights/global_lse: [N, B, H] / [B, H]
        weights = torch::exp(shifted);
        global_lse = torch::log(weights.sum(0)) + lse_max;
    } else {
        // weights/global_lse: [N, B, H] / [B, H]
        weights = torch::exp(shifted * kLogE2);
        global_lse = torch::log(weights.sum(0)) * kLog2E + lse_max;
    }

    // 对分片权重做一次归一化，再按分片维度加权求和输出张量。
    // NaN 权重直接归零，避免分片全无效时污染结果。
    weights = torch::where(torch::isnan(weights), torch::zeros_like(weights), weights);
    // normalized: [N, B, H]
    // weight_sum: [1, B, H]
    auto weight_sum = weights.sum(0, true);
    auto normalized = weights / weight_sum.clamp_min(1e-10);
    // result: [B, H, D]
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

// 在 Windows 上用逐序列 SDPA 参考实现兜底 prefill attention。
// 这里显式展开每条请求的 token 区间、滑窗限制和 causal mask，输出形状与正式算子保持一致。
void prefill_attention_precompiled(torch::Tensor &output,
                                   const torch::Tensor &q,
                                   const torch::Tensor &k,
                                   const torch::Tensor &v,
                                   const torch::Tensor &b_start_loc,
                                   const torch::Tensor &b_seq_len,
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

    // batch/kv_group_num/q_window/k_window: 请求数 / KV 复用组数 / 双侧滑窗大小
    const int64_t batch = b_seq_len.numel();
    const int64_t kv_group_num = q.size(1) / k.size(1);
    const int64_t q_window = std::max<int64_t>(sliding_window_q, 0);
    const int64_t k_window = std::max<int64_t>(sliding_window_k, 0);
    // starts_cpu/seq_lens_cpu: [B]
    auto starts_cpu = b_start_loc.to(torch::kLong).cpu();
    auto seq_lens_cpu = b_seq_len.to(torch::kLong).cpu();
    auto long_options =
            torch::TensorOptions().dtype(torch::kLong).device(q.device());
    auto bool_options =
            torch::TensorOptions().dtype(torch::kBool).device(q.device());

    // 逐条请求切片，避免把不同长度的样本硬拼成一个统一 attention 图。
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

        // q_seq/k_seq/v_seq: [T_seq, Hq|Hkv, D]
        auto q_seq = q.slice(0, seq_start, seq_stop).to(torch::kFloat32);
        auto k_seq = k.slice(0, seq_start, seq_stop).to(torch::kFloat32);
        auto v_seq = v.slice(0, seq_start, seq_stop).to(torch::kFloat32);

        // q_heads: [Hq, T_seq, D], k_heads/v_heads: [Hkv, T_seq, D]
        auto q_heads = q_seq.permute({1, 0, 2}).contiguous();
        auto k_heads = k_seq.permute({1, 0, 2}).contiguous();
        auto v_heads = v_seq.permute({1, 0, 2}).contiguous();

        if (kv_group_num > 1) {
            // 扩头后 k/v: [Hq, T_seq, D]
            k_heads = k_heads.repeat_interleave(kv_group_num, 0);
            v_heads = v_heads.repeat_interleave(kv_group_num, 0);
        }

        // 先构造窗口约束，再叠加 causal 限制，使参考路径和正式 kernel 的可见性规则一致。
        // positions/q_pos/k_pos: [T_seq] / [T_seq, 1] / [1, T_seq]
        auto positions = torch::arange(seq_len, long_options);
        auto q_pos = positions.unsqueeze(1);
        auto k_pos = positions.unsqueeze(0);
        // mask: [T_seq, T_seq]
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

        // 这里直接复用 ATen 的 SDPA 参考实现，只负责把输入整理成它要求的布局。
        std::optional<at::Tensor> attn_mask(mask.unsqueeze(0).unsqueeze(0));
        // out: [Hq, T_seq, D]
        auto out = at::scaled_dot_product_attention(
                q_heads.unsqueeze(0), k_heads.unsqueeze(0),
                v_heads.unsqueeze(0), attn_mask, 0.0, false,
                std::optional<double>(softmax_scale), false)
                .squeeze(0);
        output.slice(0, seq_start, seq_stop)
                .copy_(out.permute({1, 0, 2}).to(output.scalar_type()));
    }
}

// 为 prefix prefill 构造“历史上下文 + 当前 query”联合 attention。
// 这条兜底路径会把历史 KV 和当前 batch 内的 KV 手动拼接，再交给 SDPA 参考实现处理。
void prefix_prefill_attention_precompiled(
        torch::Tensor &output, const torch::Tensor &q, const torch::Tensor &k,
        const torch::Tensor &v, const torch::Tensor &gathered_ctx_k,
        const torch::Tensor &gathered_ctx_v, const torch::Tensor &cu_ctx_lens,
        const torch::Tensor &b_start_loc, const torch::Tensor &b_seq_len,
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

    // batch/num_q_heads/num_kv_heads/kv_group_num: 批次与头分组元数据
    const int64_t batch = b_seq_len.numel();
    const int64_t num_q_heads = q.size(1);
    const int64_t num_kv_heads = k.size(1);
    const int64_t kv_group_num = num_q_heads / num_kv_heads;
    auto starts_cpu = b_start_loc.to(torch::kLong).cpu();
    auto seq_lens_cpu = b_seq_len.to(torch::kLong).cpu();
    // cu_ctx_lens_cpu: [B + 1]
    auto cu_ctx_lens_cpu = cu_ctx_lens.to(torch::kLong).cpu();
    auto long_options =
            torch::TensorOptions().dtype(torch::kLong).device(q.device());

    // 逐条请求拼接历史上下文和本轮 query 对应的 KV，再按 query 长度回写输出。
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

        // ctx_k/ctx_v: [T_ctx, Hkv, D], q_seq/k_seq/v_seq: [T_q, Hq|Hkv, D]
        auto ctx_k = gathered_ctx_k.slice(0, ctx_start, ctx_stop).to(torch::kFloat32);
        auto ctx_v = gathered_ctx_v.slice(0, ctx_start, ctx_stop).to(torch::kFloat32);
        auto q_seq = q.slice(0, seq_start, seq_stop).to(torch::kFloat32);
        auto k_seq = k.slice(0, seq_start, seq_stop).to(torch::kFloat32);
        auto v_seq = v.slice(0, seq_start, seq_stop).to(torch::kFloat32);

        // all_k/all_v: [T_ctx + T_q, Hkv, D]
        auto all_k = torch::cat({ctx_k, k_seq}, 0);
        auto all_v = torch::cat({ctx_v, v_seq}, 0);
        // q_heads/k_heads/v_heads: [Hq, T_q|T_ctx+T_q, D]
        auto q_heads = q_seq.permute({1, 0, 2}).contiguous();
        auto k_heads = all_k.permute({1, 0, 2}).contiguous();
        auto v_heads = all_v.permute({1, 0, 2}).contiguous();
        if (kv_group_num > 1) {
            // 扩头后 k/v: [Hq, T_ctx + T_q, D]
            k_heads = k_heads.repeat_interleave(kv_group_num, 0);
            v_heads = v_heads.repeat_interleave(kv_group_num, 0);
        }

        // 当前 query 可以看见全部 prefix 上下文；滑窗限制只在 query 到 key 的相对距离上生效。
        // query_positions/key_positions: [T_q] / [T_ctx + T_q]
        auto query_positions =
                torch::arange(ctx_len, ctx_len + query_len, long_options);
        auto key_positions = torch::arange(0, ctx_len + query_len, long_options);
        // mask: [T_q, T_ctx + T_q]
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

// ------------------------------- 序列打包与采样过滤兜底算子 -------------------------------

// 把按 token 连续排布的二维张量，重新装配成 [B, Lmax, D] 的 padded 布局。
// 这在 Windows 上主要用于给缺失的变长 kernel 提供一个易于调试的参考路径。
torch::Tensor pack_seq_precompiled(const torch::Tensor &x,
                                   const torch::Tensor &lengths,
                                   double pad_value) {
    TORCH_CHECK(x.is_cuda(), "pack_seq_precompiled expects CUDA x");
    TORCH_CHECK(lengths.is_cuda(), "pack_seq_precompiled expects CUDA lengths");
    TORCH_CHECK(x.dim() == 2, "pack_seq_precompiled expects x with shape [N, D]");
    TORCH_CHECK(lengths.dim() == 1,
                "pack_seq_precompiled expects 1D sequence lengths");

    auto lengths_i64 = lengths.to(torch::kLong).contiguous();
    auto lengths_cpu = lengths_i64.cpu();
    // batch/feature_dim/max_len/total_tokens: pack 后的输出尺寸元数据
    const int64_t batch = lengths_cpu.numel();
    const int64_t feature_dim = x.size(1);
    const int64_t max_len =
            batch > 0 ? lengths_cpu.max().item<int64_t>() : 0;
    const int64_t total_tokens =
            batch > 0 ? lengths_cpu.sum().item<int64_t>() : 0;

    TORCH_CHECK(total_tokens == x.size(0),
                "pack_seq_precompiled expects sum(lengths) to equal x.size(0)");

    // 先分配满长 padded 输出，再按每条序列自己的长度逐段拷入真实 token。
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

// 将 padded 序列按 lengths 还原回连续 token 布局。
// 这里与 pack_seq_precompiled 互为逆操作，便于测试和参考实现对拍。
torch::Tensor unpack_seq_precompiled(const torch::Tensor &packed_tensor,
                                     const torch::Tensor &lengths) {
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
    // batch/max_len/feature_dim/total_tokens: unpack 后的连续 token 布局尺寸
    const int64_t batch = lengths_cpu.numel();
    const int64_t max_len = packed_tensor.size(1);
    const int64_t feature_dim = packed_tensor.size(2);
    const int64_t total_tokens =
            batch > 0 ? lengths_cpu.sum().item<int64_t>() : 0;

    // 输出重新回到连续 token 布局，总长度就是所有序列长度之和。
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

// 按每条请求对应的 token 数，把批级别标量展开成 token 级别向量。
// replace_from/replace_to 用于在展开阶段顺手改写特殊标记值，减少额外 kernel 调用。
torch::Tensor expand_batch_to_tokens_precompiled(
        const torch::Tensor &x, const torch::Tensor &cu_num_tokens,
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

    // cu_num_tokens 存的是前缀和，这里先还原出每条请求真实拥有的 token 数。
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

    // repeat_interleave 把批级值复制成 token 级值，后续逻辑就能直接按 token 消费。
    auto expanded_x = x.repeat_interleave(counts);
    if (replace_from != replace_to) {
        auto replacement = torch::full({}, replace_to, expanded_x.options());
        expanded_x = torch::where(expanded_x == replace_from, replacement, expanded_x);
    }
    return expanded_x;
}

// 依据目标分布与 draft 分布的差值，恢复被拒绝 token 的重采样结果。
// 如果没有提供 draft_probs，就退化成“屏蔽原 draft token 后从 target_probs 里重新选最大值”。
torch::Tensor sample_recovered_tokens_precompiled(
        const torch::Tensor &cu_num_draft_tokens,
        const torch::Tensor &draft_token_ids,
        const std::optional<torch::Tensor> &draft_probs,
        const torch::Tensor &target_probs, const torch::Tensor &inv_q) {
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
    // inv_q 是按请求存的，这里展开成和每个 draft token 一一对应的行数。
    auto expanded_inv_q = inv_q.index_select(0, req_indices);

    torch::Tensor scores;
    if (draft_probs.has_value() && draft_probs.value().defined()) {
        TORCH_CHECK(draft_probs.value().is_cuda(),
                    "sample_recovered_tokens_precompiled expects CUDA draft_probs");
        TORCH_CHECK(draft_probs.value().sizes() == target_probs.sizes(),
                    "sample_recovered_tokens_precompiled expects draft_probs to "
                    "match target_probs");
        // 有 draft_probs 时，真正可恢复的质量来自 target 与 draft 的正向差值。
        scores = torch::clamp_min(target_probs - draft_probs.value(), 0.0);
    } else {
        // 没有 draft_probs 时，至少要把原始 draft token 自己屏蔽掉，避免直接原样选回去。
        scores = target_probs.clone();
        auto draft_ids_long = draft_token_ids.to(torch::kLong).unsqueeze(1);
        auto zeros = torch::zeros({scores.size(0), 1}, scores.options());
        scores = scores.scatter(1, draft_ids_long, zeros);
    }

    auto weighted_scores = scores * expanded_inv_q;
    return std::get<1>(weighted_scores.max(1)).to(draft_token_ids.scalar_type());
}

// 在 Windows 上提供统一的 top-k / top-p logits 过滤参考实现。
// 这里按参数组合拆成几条分支，优先复用已有小算子，最后再回落到纯 ATen 逻辑。
void apply_top_k_top_p_precompiled(
        torch::Tensor &logits, const std::optional<torch::Tensor> &k,
        const std::optional<torch::Tensor> &p, double mask_value) {
    TORCH_CHECK(logits.is_cuda(), "apply_top_k_top_p_precompiled expects CUDA logits");
    TORCH_CHECK(logits.dim() == 2,
                "apply_top_k_top_p_precompiled expects 2D logits");

    if (!k.has_value() && !p.has_value()) {
        return;
    }

    // 只有 top-k 时优先走项目里现成的逐行 top-k kernel，减少不必要的全量排序。
    if (k.has_value() && !p.has_value()) {
        // resolved_k: [B]
        auto resolved_k =
                normalize_rowwise_param(k.value(), logits, torch::kLong)
                        .clamp(1, logits.size(1));
        // 先尝试复用已有 CUDA top-k per-row 实现。
        if (try_apply_top_k_only_with_cuda_topk_per_row(logits, resolved_k,
                                                        mask_value)) {
            return;
        }

        // no_top_k_mask/effective_k: [B]
        auto no_top_k_mask = resolved_k == logits.size(1);
        if (no_top_k_mask.all().item<bool>()) {
            return;
        }
        auto effective_k = resolved_k.masked_fill(no_top_k_mask, 1);
        const int64_t max_k = effective_k.max().item<int64_t>();
        // topk_values: [B, max_k]
        auto topk_values = std::get<0>(torch::topk(logits, max_k, 1, true, true));
        // k_index/top_k_threshold: [B, 1]
        auto k_index = effective_k.sub(1).unsqueeze(1);
        auto top_k_threshold = topk_values.gather(1, k_index);
        // 没有限 top-k 的行直接把阈值设回 mask_value，保持原 logits。
        top_k_threshold.masked_fill_(no_top_k_mask.unsqueeze(1), mask_value);
        logits.masked_fill_(logits < top_k_threshold, mask_value);
        return;
    }

    // top-k 与 top-p 同时存在时，先裁出候选集，再在候选集内部执行累计概率过滤。
    if (k.has_value() && p.has_value()) {
        // resolved_k/resolved_p: [B]
        auto resolved_k =
                normalize_rowwise_param(k.value(), logits, torch::kLong)
                        .clamp(1, logits.size(1));
        auto resolved_p =
                normalize_rowwise_param(p.value(), logits, torch::kFloat32)
                        .clamp(0.0, 1.0);
        // small_k_mask/full_k_mask: [B]
        auto small_k_mask = resolved_k < logits.size(1);
        auto full_k_mask = resolved_k == logits.size(1);

        if (small_k_mask.all().item<bool>()) {
            apply_top_k_then_top_p_small_k_precompiled(logits, resolved_k,
                                                       resolved_p, mask_value);
            return;
        }

        if (small_k_mask.any().item<bool>()) {
            // small_logits: [B_small, V]
            auto small_rows = torch::nonzero(small_k_mask).view(-1);
            // small_k/small_p: [B_small]
            auto small_logits = logits.index_select(0, small_rows);
            auto small_k = resolved_k.index_select(0, small_rows);
            auto small_p = resolved_p.index_select(0, small_rows);
            apply_top_k_then_top_p_small_k_precompiled(small_logits, small_k, small_p,
                                                       mask_value);
            logits.index_copy_(0, small_rows, small_logits);
        }

        if (full_k_mask.any().item<bool>()) {
            // full_logits: [B_full, V]
            auto full_rows = torch::nonzero(full_k_mask).view(-1);
            // full_p: [B_full]
            auto full_logits = logits.index_select(0, full_rows);
            auto full_p = resolved_p.index_select(0, full_rows);
            apply_top_p_only_iterative_topk_precompiled(full_logits, full_p,
                                                        mask_value);
            logits.index_copy_(0, full_rows, full_logits);
            return;
        }
    }

    // 只有 top-p 时直接按累计概率截断，不引入额外的 top-k 先验。
    if (!k.has_value() && p.has_value()) {
        // resolved_p: [B]
        auto resolved_p =
                normalize_rowwise_param(p.value(), logits, torch::kFloat32)
                        .clamp(0.0, 1.0);
        apply_top_p_only_iterative_topk_precompiled(logits, resolved_p, mask_value);
        return;
    }

    torch::Tensor logits_sort;
    torch::Tensor logits_idx;
    // logits_sort/logits_idx: [B, V]
    std::tie(logits_sort, logits_idx) = torch::sort(logits, -1, false);

    if (k.has_value()) {
        // resolved_k: [B]
        auto resolved_k =
                normalize_rowwise_param(k.value(), logits, torch::kLong)
                        .clamp(1, logits_sort.size(1));
        // top_k_index/top_k_threshold/top_k_mask: [B, 1] / [B, 1] / [B, V]
        auto top_k_index =
                (logits_sort.size(1) - resolved_k).unsqueeze(1);
        auto top_k_threshold = logits_sort.gather(1, top_k_index);
        auto top_k_mask = logits_sort < top_k_threshold;
        logits_sort.masked_fill_(top_k_mask, mask_value);
    }

    if (p.has_value()) {
        // resolved_p: [B]
        auto resolved_p =
                normalize_rowwise_param(p.value(), logits, torch::kFloat32)
                        .clamp(0.0, 1.0);
        // probs_sort/probs_cumsum/top_p_mask: [B, V]
        auto probs_sort = logits_sort.softmax(-1);
        auto probs_cumsum = torch::cumsum(probs_sort, -1);
        auto top_p_mask = probs_cumsum <= (1 - resolved_p.unsqueeze(1));
        top_p_mask.select(1, top_p_mask.size(1) - 1).fill_(false);
        logits_sort.masked_fill_(top_p_mask, mask_value);
    }

    logits.scatter_(1, logits_idx, logits_sort);
}

// ------------------------------- InputBatch 请求状态兜底算子 -------------------------------

// 按请求状态把下一轮 prefill 所需的 token 填进连续输入缓冲区。
// 同时预取每条请求在本轮 query 之后紧跟着要补进来的 next_prefill token。
void input_batch_prepare_prefill_inputs_precompiled(
        torch::Tensor &input_ids, torch::Tensor &next_prefill_tokens,
        const torch::Tensor &idx_mapping, const torch::Tensor &query_start_loc,
        const torch::Tensor &all_token_ids, const torch::Tensor &prefill_len,
        const torch::Tensor &num_computed_tokens) {
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

    // idx_cpu/query_start_cpu/prefill_len_cpu/computed_cpu: 请求级 CPU 元数据
    auto idx_cpu = idx_mapping.to(torch::kCPU);
    auto query_start_cpu = query_start_loc.to(torch::kCPU);
    auto prefill_len_cpu = prefill_len.to(torch::kCPU);
    auto computed_cpu = num_computed_tokens.to(torch::kCPU);

    // num_reqs: 当前批次里参与 prefill 的请求数
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

        // 先把本轮还未计算的 prefill token 拷入输入缓冲，供后续一次性执行 prefill。
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

        // 再把 query 之后的下一个待填充 token 单独缓存出来，供 chunked prefill 续跑时复用。
        const int64_t next_pos = num_computed + query_len;
        if (next_pos < prefill) {
            next_prefill_tokens.select(0, req_state_idx)
                    .copy_(all_token_ids.select(0, req_state_idx).select(0, next_pos));
        }
    }
}

// 结合请求映射和 query 区间，补齐当前位置编码和本轮有效 seq_lens。
// 这样后续 kernel 即使在 Windows 参考路径里，也能拿到和正式调度器一致的位置信息。
void input_batch_prepare_pos_seq_lens_precompiled(
        const torch::Tensor &idx_mapping, const torch::Tensor &query_start_loc,
        const torch::Tensor &num_computed_tokens, torch::Tensor &pos,
        torch::Tensor &seq_lens) {
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
    // computed_cpu: [N_req_state]
    auto computed_cpu = num_computed_tokens.to(torch::kCPU);

    // num_reqs: 当前批次请求数
    const int64_t num_reqs = idx_mapping.size(0);
    if (seq_lens.size(0) > num_reqs) {
        // 未使用的尾部槽位必须清零，避免后续批处理逻辑误读历史残留值。
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
            // 每条请求的位置从“已计算 token 数”继续向后递增。
            pos.slice(0, start, end)
                    .copy_(torch::arange(num_computed, num_computed + query_len,
                                         pos.options()));
        }
    }
}

// 把上一轮 sample 结果和 draft token 拼回 input_ids，并返回 logits 对应的输入下标。
// 这条兜底路径复刻的是 speculative decode 里“sampled token + draft token”拼接逻辑。
torch::Tensor input_batch_combine_sampled_and_draft_tokens_precompiled(
        torch::Tensor &input_ids, const torch::Tensor &idx_mapping,
        const torch::Tensor &last_sampled_tokens,
        const torch::Tensor &query_start_loc, const torch::Tensor &seq_lens,
        const torch::Tensor &prefill_len, const torch::Tensor &draft_tokens,
        const torch::Tensor &cu_num_logits, int64_t num_logits) {
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
    // logits_indices: [num_logits]
    auto idx_cpu = idx_mapping.to(torch::kCPU);
    auto query_start_cpu = query_start_loc.to(torch::kCPU);
    // seq_lens_cpu/prefill_len_cpu/cu_num_logits_cpu: [B] / [N_req_state] / [B + 1]
    auto seq_lens_cpu = seq_lens.to(torch::kCPU);
    auto prefill_len_cpu = prefill_len.to(torch::kCPU);
    auto cu_num_logits_cpu = cu_num_logits.to(torch::kCPU);

    // num_reqs: 当前批次请求数
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
            // logits 只对应当前请求尾部那一段输入，所以这里先建好全局到局部的索引映射。
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

        // decode 阶段第一个位置写入上轮真正采样出的 token，后续位置再接 draft token。
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

// 根据本轮 sample / reject 情况，统计每条请求实际接受了多少 token。
// chunked prefill 请求不应被当成正常 decode，因此这里会强制把 sampled 数量改回 0。
std::tuple<torch::Tensor, torch::Tensor>
input_batch_get_num_sampled_and_rejected_precompiled(
        torch::Tensor &num_sampled, const torch::Tensor &seq_lens,
        const torch::Tensor &cu_num_logits, const torch::Tensor &idx_mapping,
        const torch::Tensor &prefill_len) {
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

    // num_reqs: 当前批次请求数
    const int64_t num_reqs = idx_mapping.size(0);
    for (int64_t batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
        const int64_t req_state_idx = idx_cpu[batch_idx].item<int64_t>();
        const int64_t seq_len = seq_lens_cpu[batch_idx].item<int64_t>();
        const int64_t prefill =
                prefill_len_cpu[req_state_idx].item<int64_t>();
        const bool is_chunked_prefilling = seq_len < prefill;
        int64_t sampled = num_sampled_cpu[batch_idx].item<int64_t>();
        // chunked prefill 还没真正进入 decode，所以 sample 数必须被视为 0。
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
        // rejected 数只在正常 decode 阶段有意义，prefill 补块时一律不记 reject。
        num_rejected.select(0, batch_idx).fill_(rejected);
    }
    return std::make_tuple(num_sampled, num_rejected);
}

// 在请求级状态表上回写本轮 sample 结果，并同步更新 all_token_ids / total_len。
// 如果启用了 output_bin_counts，这里也顺手完成每个 token 的计数累计。
void input_batch_post_update_precompiled(
        const torch::Tensor &idx_mapping, torch::Tensor &num_computed_tokens,
        torch::Tensor &last_sampled_tokens,
        const std::optional<torch::Tensor> &output_bin_counts,
        const torch::Tensor &sampled_tokens, const torch::Tensor &num_sampled,
        const torch::Tensor &num_rejected, const torch::Tensor &query_start_loc,
        torch::Tensor &all_token_ids, torch::Tensor &total_len) {
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
    // query_start_cpu/total_len_cpu: [B + 1] / [N_req_state]
    auto query_start_cpu = query_start_loc.to(torch::kCPU);
    auto total_len_cpu = total_len.to(torch::kCPU);

    // num_reqs: 当前批次请求数
    const int64_t num_reqs = idx_mapping.size(0);
    for (int64_t req_id = 0; req_id < num_reqs; ++req_id) {
        const int64_t req_state_idx = idx_cpu[req_id].item<int64_t>();
        const int64_t old_total_len = total_len_cpu[req_state_idx].item<int64_t>();
        const int64_t sampled = num_sampled_cpu[req_id].item<int64_t>();
        if (sampled > 0) {
            // last_sampled_tokens 只记录最终被接受的最后一个 token，供下一轮 decode 直接取用。
            last_sampled_tokens.select(0, req_state_idx)
                    .copy_(sampled_tokens.select(0, req_id).select(0, sampled - 1));
            total_len.select(0, req_state_idx).fill_(old_total_len + sampled);
        }

        for (int64_t i = 0; i < sampled; ++i) {
            // all_token_ids 是请求历史的真值表，所以每个新接受 token 都要逐个追加入列。
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
        // 已计算 token 数只增加真正进入主序列的部分，reject 的尾部 token 不能记账。
        num_computed_tokens.select(0, req_state_idx)
                .add_(query_len - rejected);
    }
}

// pool 模式下不涉及 sample/reject，只需要把 query 长度直接累加回 num_computed_tokens。
void input_batch_post_update_pool_precompiled(
        const torch::Tensor &idx_mapping, torch::Tensor &num_computed_tokens,
        const torch::Tensor &query_start_loc) {
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
    // idx_cpu/query_start_cpu: [B] / [B + 1]
    const int64_t num_reqs = idx_mapping.size(0);
    for (int64_t batch_id = 0; batch_id < num_reqs; ++batch_id) {
        const int64_t req_state_idx = idx_cpu[batch_id].item<int64_t>();
        const int64_t query_start = query_start_cpu[batch_id].item<int64_t>();
        const int64_t query_end = query_start_cpu[batch_id + 1].item<int64_t>();
        // 每条请求只把本轮 query 长度累加回 num_computed_tokens。
        num_computed_tokens.select(0, req_state_idx)
                .add_(query_end - query_start);
    }
}

// 把请求级 idx_mapping 按 logits 数量展开成 token 级映射，并返回局部位置下标。
// 这为后续“按 logits 行回写到请求状态”提供了一个直接可用的索引表。
std::tuple<torch::Tensor, torch::Tensor>
input_batch_expand_idx_mapping_precompiled(const torch::Tensor &idx_mapping,
                                           int64_t total_num_logits,
                                           const torch::Tensor &cu_num_logits) {
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
    // expanded_idx_mapping/expanded_local_pos: [num_logits_total]

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
        // 先把这一段 logits 全部标记成来自哪条请求。
        expanded_idx_mapping.slice(0, start, end)
                .fill_(idx_cpu[req_idx].item<int64_t>());
        // 再在请求内部从 0 开始编号，后续就能按“局部第几个 logits”回写状态。
        expanded_local_pos.slice(0, start, end)
                .copy_(torch::arange(0, num_tokens, expanded_local_pos.options()));
    }

    return std::make_tuple(expanded_idx_mapping, expanded_local_pos);
}

// ------------------------------- EAGLE 草稿解码兜底算子 -------------------------------

// 根据当前位置和 block table 更新下一步的 slot_mapping、clamped position 与 seq_lens。
// 这一步把 EAGLE 草稿分支要写入 KV cache 的目标槽位提前算好，方便后续一步落盘。
void eagle_step_update_slot_mapping_and_metadata_precompiled(
        const torch::Tensor &positions_1d, const torch::Tensor &block_table_tensor,
        torch::Tensor &seq_lens, int64_t block_size, int64_t max_model_len,
        torch::Tensor &out_clamped_positions, torch::Tensor &out_slot_mapping,
        int64_t input_batch_size) {
    TORCH_CHECK(positions_1d.is_cuda(),
                "eagle_step_update_slot_mapping_and_metadata_precompiled expects "
                "CUDA positions_1d");
    TORCH_CHECK(block_table_tensor.is_cuda(),
                "eagle_step_update_slot_mapping_and_metadata_precompiled expects "
                "CUDA block_table_tensor");
    TORCH_CHECK(seq_lens.is_cuda(),
                "eagle_step_update_slot_mapping_and_metadata_precompiled expects "
                "CUDA seq_lens");
    TORCH_CHECK(out_clamped_positions.is_cuda(),
                "eagle_step_update_slot_mapping_and_metadata_precompiled expects "
                "CUDA out_clamped_positions");
    TORCH_CHECK(out_slot_mapping.is_cuda(),
                "eagle_step_update_slot_mapping_and_metadata_precompiled expects "
                "CUDA out_slot_mapping");
    TORCH_CHECK(positions_1d.dim() == 1,
                "eagle_step_update_slot_mapping_and_metadata_precompiled expects "
                "1D positions_1d");
    TORCH_CHECK(block_table_tensor.dim() == 2,
                "eagle_step_update_slot_mapping_and_metadata_precompiled expects "
                "2D block_table_tensor");

    // batch_size/n_blocks_per_req: 请求数与每请求 block table 宽度
    const int64_t batch_size = positions_1d.size(0);
    const int64_t n_blocks_per_req = block_table_tensor.size(1);
    TORCH_CHECK(seq_lens.size(0) >= batch_size,
                "eagle_step_update_slot_mapping_and_metadata_precompiled expects "
                "seq_lens to cover batch");
    TORCH_CHECK(out_clamped_positions.size(0) >= batch_size,
                "eagle_step_update_slot_mapping_and_metadata_precompiled expects "
                "out_clamped_positions to cover batch");
    TORCH_CHECK(out_slot_mapping.size(0) >= input_batch_size,
                "eagle_step_update_slot_mapping_and_metadata_precompiled expects "
                "out_slot_mapping to cover input_batch_size");

    if (batch_size == 0) {
        if (input_batch_size > 0) {
            out_slot_mapping.slice(0, 0, input_batch_size).fill_(-1);
        }
        return;
    }

    // 先得到“理论上的下一位置”，超出模型长度的请求会被重置成无效槽位。
    auto new_position = positions_1d + 1;
    // exceeds_max: [B]
    auto exceeds_max = new_position >= max_model_len;
    // clamped_position: [B]
    auto clamped_position =
            torch::where(exceeds_max, torch::zeros_like(new_position), new_position);
    auto block_number =
            torch::floor_divide(clamped_position, block_size)
                    .clamp(0, std::max<int64_t>(n_blocks_per_req - 1, 0))
                    .to(torch::kLong);
    // req_idx/block_number/block_id/slot_id: [B]
    auto req_idx = torch::arange(
            batch_size,
            torch::TensorOptions().device(positions_1d.device()).dtype(torch::kLong));
    auto block_id = block_table_tensor.index({req_idx, block_number});
    // slot_id: [B]
    auto slot_id = block_id * block_size + torch::remainder(clamped_position, block_size)
            .to(block_id.scalar_type());
    slot_id = torch::where(exceeds_max,
                           torch::full_like(slot_id, -1),
                           slot_id);

    out_clamped_positions.slice(0, 0, batch_size)
            .copy_(clamped_position.to(out_clamped_positions.scalar_type()));
    out_slot_mapping.slice(0, 0, batch_size)
            .copy_(slot_id.to(out_slot_mapping.scalar_type()));
    if (input_batch_size > batch_size) {
        out_slot_mapping.slice(0, batch_size, input_batch_size).fill_(-1);
    }

    auto next_seq_lens =
            torch::where(exceeds_max, torch::ones_like(seq_lens.slice(0, 0, batch_size)),
                         seq_lens.slice(0, 0, batch_size) + 1);
    // 超过上限后 seq_len 回退到 1，否则只做单步递增。
    next_seq_lens =
            torch::minimum(next_seq_lens,
                           torch::full({}, max_model_len, next_seq_lens.options()));
    seq_lens.slice(0, 0, batch_size)
            .copy_(next_seq_lens.to(seq_lens.scalar_type()));
}

// 为 padded EAGLE 输入准备“应该从哪里取下一个 token”以及“有多少 token 被拒绝”。
void eagle_prepare_inputs_padded_precompiled(
        const torch::Tensor &cu_num_draft_tokens,
        const torch::Tensor &valid_sampled_tokens_count,
        const torch::Tensor &query_start_loc_gpu,
        torch::Tensor &token_indices_to_sample,
        torch::Tensor &num_rejected_tokens_gpu) {
    TORCH_CHECK(cu_num_draft_tokens.is_cuda(),
                "eagle_prepare_inputs_padded_precompiled expects CUDA "
                "cu_num_draft_tokens");
    TORCH_CHECK(valid_sampled_tokens_count.is_cuda(),
                "eagle_prepare_inputs_padded_precompiled expects CUDA "
                "valid_sampled_tokens_count");
    TORCH_CHECK(query_start_loc_gpu.is_cuda(),
                "eagle_prepare_inputs_padded_precompiled expects CUDA "
                "query_start_loc_gpu");
    TORCH_CHECK(token_indices_to_sample.is_cuda(),
                "eagle_prepare_inputs_padded_precompiled expects CUDA "
                "token_indices_to_sample");
    TORCH_CHECK(num_rejected_tokens_gpu.is_cuda(),
                "eagle_prepare_inputs_padded_precompiled expects CUDA "
                "num_rejected_tokens_gpu");

    const int64_t num_reqs = valid_sampled_tokens_count.size(0);
    if (num_reqs == 0) {
        return;
    }
    // 先把 prefix-sum 形式的 cu_num_draft_tokens 还原成每条请求自己的 draft 数。
    auto cu_i64 = cu_num_draft_tokens.to(torch::kLong);
    auto num_draft_tokens = torch::empty_like(cu_i64);
    // num_draft_tokens: [B]
    num_draft_tokens.slice(0, 0, 1).copy_(cu_i64.slice(0, 0, 1));
    if (num_reqs > 1) {
        num_draft_tokens.slice(0, 1, num_reqs)
                .copy_(cu_i64.slice(0, 1, num_reqs) - cu_i64.slice(0, 0, num_reqs - 1));
    }

    auto valid_i64 = valid_sampled_tokens_count.to(torch::kLong);
    // num_rejected_tokens: [B]
    auto num_rejected_tokens = num_draft_tokens + 1 - valid_i64;
    num_rejected_tokens =
            torch::where(num_draft_tokens > 0, num_rejected_tokens,
                         torch::zeros_like(num_rejected_tokens));
    // 下一个真正要 sample 的位置，等于 query 尾部再减去被 reject 掉的那些 token。
    auto index_to_sample =
            query_start_loc_gpu.to(torch::kLong).slice(0, 1, num_reqs + 1) - 1 -
            num_rejected_tokens;

    token_indices_to_sample.copy_(
            index_to_sample.to(token_indices_to_sample.scalar_type()));
    num_rejected_tokens_gpu.copy_(
            num_rejected_tokens.to(num_rejected_tokens_gpu.scalar_type()));
}

// 从 sampled_token_ids 里挑出每条请求最后一个有效 token，作为下一步真正输入的 token。
// 如果请求被 discard 或根本没有有效 sample，则退回到 backup_next_token_ids。
void eagle_prepare_next_token_padded_precompiled(
        const torch::Tensor &sampled_token_ids,
        const torch::Tensor &discard_request_mask,
        const torch::Tensor &backup_next_token_ids, torch::Tensor &next_token_ids,
        torch::Tensor &valid_sampled_tokens_count, int64_t vocab_size) {
    TORCH_CHECK(sampled_token_ids.is_cuda(),
                "eagle_prepare_next_token_padded_precompiled expects CUDA "
                "sampled_token_ids");
    TORCH_CHECK(discard_request_mask.is_cuda(),
                "eagle_prepare_next_token_padded_precompiled expects CUDA "
                "discard_request_mask");
    TORCH_CHECK(backup_next_token_ids.is_cuda(),
                "eagle_prepare_next_token_padded_precompiled expects CUDA "
                "backup_next_token_ids");
    TORCH_CHECK(next_token_ids.is_cuda(),
                "eagle_prepare_next_token_padded_precompiled expects CUDA "
                "next_token_ids");
    TORCH_CHECK(valid_sampled_tokens_count.is_cuda(),
                "eagle_prepare_next_token_padded_precompiled expects CUDA "
                "valid_sampled_tokens_count");
    TORCH_CHECK(sampled_token_ids.dim() == 2,
                "eagle_prepare_next_token_padded_precompiled expects 2D "
                "sampled_token_ids");

    // batch_size/num_sampled_tokens_per_req: 请求数与每请求草稿采样宽度
    const int64_t batch_size = sampled_token_ids.size(0);
    const int64_t num_sampled_tokens_per_req = sampled_token_ids.size(1);
    if (batch_size == 0) {
        return;
    }

    // 先标记出哪些 sample 位置是有效词表 id。
    auto valid_mask = (sampled_token_ids != -1) & (sampled_token_ids < vocab_size);
    // valid_mask: [B, K_draft]
    auto valid_count =
            valid_mask.sum(1).to(valid_sampled_tokens_count.scalar_type());
    auto offsets = torch::arange(
            num_sampled_tokens_per_req,
            torch::TensorOptions()
                    .device(sampled_token_ids.device())
                    .dtype(torch::kLong))
            .unsqueeze(0)
            .expand({batch_size, num_sampled_tokens_per_req});
    // last_valid_index: [B]
    auto last_valid_index =
            torch::where(valid_mask, offsets, torch::full_like(offsets, -1))
                    .amax(1);
    // gather_index/last_valid_token: [B, 1] / [B]
    auto gather_index = last_valid_index.clamp_min(0).unsqueeze(1);
    auto last_valid_token = sampled_token_ids.gather(1, gather_index).squeeze(1);
    // 如果这一行有有效 sample，就取最后一个有效 token；否则退回 backup token。
    auto next_token =
            torch::where(valid_count > 0,
                         last_valid_token.to(backup_next_token_ids.scalar_type()),
                         backup_next_token_ids);

    auto discard_mask = discard_request_mask.to(torch::kBool);
    // 被丢弃的请求要强制恢复成 backup token，同时把 valid_count 归零。
    next_token = torch::where(discard_mask, backup_next_token_ids, next_token);
    valid_count =
            torch::where(discard_mask, torch::zeros_like(valid_count), valid_count);

    next_token_ids.copy_(next_token.to(next_token_ids.scalar_type()));
    valid_sampled_tokens_count.copy_(
            valid_count.to(valid_sampled_tokens_count.scalar_type()));
}

// 把目标输入扩展成 EAGLE 需要的“有效 token + bonus token + 并行草稿槽位 + reject 槽位”布局。
// 这里同时回写多种掩码和索引，方便后续一步完成 draft 分支的拼装。
void copy_and_expand_eagle_inputs_precompiled(
        const torch::Tensor &target_token_ids,
        const torch::Tensor &target_positions, const torch::Tensor &next_token_ids,
        torch::Tensor &out_input_ids, torch::Tensor &out_positions,
        torch::Tensor &out_is_rejected_token_mask,
        torch::Tensor &out_is_masked_token_mask,
        torch::Tensor &out_new_token_indices,
        torch::Tensor &out_hidden_state_mapping,
        const torch::Tensor &query_start_loc, const torch::Tensor &query_end_loc,
        int64_t padding_token_id, int64_t parallel_drafting_token_id,
        int64_t total_input_tokens, int64_t num_padding_slots_per_request,
        bool shift_input_ids) {
    TORCH_CHECK(target_token_ids.is_cuda(),
                "copy_and_expand_eagle_inputs_precompiled expects CUDA "
                "target_token_ids");
    TORCH_CHECK(target_positions.is_cuda(),
                "copy_and_expand_eagle_inputs_precompiled expects CUDA "
                "target_positions");
    TORCH_CHECK(next_token_ids.is_cuda(),
                "copy_and_expand_eagle_inputs_precompiled expects CUDA "
                "next_token_ids");
    TORCH_CHECK(out_input_ids.is_cuda() && out_positions.is_cuda() &&
                out_is_rejected_token_mask.is_cuda() &&
                out_is_masked_token_mask.is_cuda() &&
                out_new_token_indices.is_cuda() &&
                out_hidden_state_mapping.is_cuda(),
                "copy_and_expand_eagle_inputs_precompiled expects CUDA outputs");
    TORCH_CHECK(query_start_loc.is_cuda() && query_end_loc.is_cuda(),
                "copy_and_expand_eagle_inputs_precompiled expects CUDA query "
                "metadata");
    TORCH_CHECK(total_input_tokens > 0,
                "copy_and_expand_eagle_inputs_precompiled expects positive "
                "total_input_tokens");

    auto query_start_cpu = query_start_loc.to(torch::kCPU);
    auto query_end_cpu = query_end_loc.to(torch::kCPU);
    // batch_size: 当前 EAGLE padded 批次请求数
    const int64_t batch_size = query_end_loc.size(0);
    // index_options: 生成输出布局索引用的 [long] 配置
    auto index_options = torch::TensorOptions()
            .device(target_token_ids.device())
            .dtype(torch::kLong);

    for (int64_t request_idx = 0; request_idx < batch_size; ++request_idx) {
        const int64_t query_start = query_start_cpu[request_idx].item<int64_t>();
        const int64_t next_query_start =
                query_start_cpu[request_idx + 1].item<int64_t>();
        const int64_t query_end = query_end_cpu[request_idx].item<int64_t>();

        int64_t num_valid_tokens;
        int64_t input_offset;
        int64_t output_start;
        if (shift_input_ids) {
            num_valid_tokens = query_end - query_start;
            input_offset = 1;
            output_start = query_start +
                           request_idx * (num_padding_slots_per_request - 1);
        } else {
            num_valid_tokens = query_end - query_start + 1;
            input_offset = 0;
            output_start =
                    query_start + request_idx * num_padding_slots_per_request;
        }

        const int64_t num_rejected = next_query_start - query_end - 1;
        const int64_t total_output_tokens =
                num_valid_tokens + num_padding_slots_per_request + num_rejected;
        if (total_output_tokens <= 0) {
            continue;
        }

        // 先按区域类型构造输出布局，再把不同来源的 token 映射到对应位置。
        // j/out_idx: [T_out_req]
        auto j = torch::arange(total_output_tokens, index_options);
        auto out_idx = output_start + j;
        auto is_valid_region = j < num_valid_tokens;
        auto is_bonus_region = j == num_valid_tokens;
        auto is_parallel_draft_region =
                (j > num_valid_tokens) &
                (j < num_valid_tokens + num_padding_slots_per_request);
        auto is_rejected_region =
                j >= (num_valid_tokens + num_padding_slots_per_request);

        auto in_idx = query_start + input_offset + j;
        // in_idx_clamped: [T_out_req]
        auto in_idx_clamped =
                torch::clamp(in_idx, 0, total_input_tokens - 1);
        // source_tokens/token_ids: [T_out_req]
        auto source_tokens = target_token_ids.index({in_idx_clamped});
        auto token_ids =
                torch::where(is_valid_region, source_tokens, torch::zeros_like(source_tokens));
        auto bonus_token =
                next_token_ids.index({request_idx}).expand_as(token_ids).to(token_ids.scalar_type());
        token_ids = torch::where(is_bonus_region, bonus_token, token_ids);
        token_ids =
                torch::where(is_parallel_draft_region,
                             torch::full_like(token_ids, parallel_drafting_token_id),
                             token_ids);
        token_ids = torch::where(is_rejected_region,
                                 torch::full_like(token_ids, padding_token_id),
                                 token_ids);

        torch::Tensor start_pos;
        if (target_positions.dim() == 1) {
            start_pos = target_positions.index({query_start});
        } else {
            start_pos = target_positions.index({0, query_start});
        }
        auto positions =
                start_pos.to(out_positions.scalar_type()) + j.to(out_positions.scalar_type());
        // reject 区域位置清零，避免后续位置编码误读无效槽位。
        positions =
                torch::where(is_rejected_region, torch::zeros_like(positions), positions);

        out_input_ids.index_put_({out_idx}, token_ids.to(out_input_ids.scalar_type()));
        out_positions.index_put_({out_idx}, positions.to(out_positions.scalar_type()));
        out_is_rejected_token_mask.index_put_(
                {out_idx}, is_rejected_region.to(out_is_rejected_token_mask.scalar_type()));
        out_is_masked_token_mask.index_put_(
                {out_idx}, is_parallel_draft_region.to(out_is_masked_token_mask.scalar_type()));

        auto is_new_token_region =
                (j >= num_valid_tokens) &
                (j < num_valid_tokens + num_padding_slots_per_request);
        auto new_token_local_idx =
                (j - num_valid_tokens).masked_select(is_new_token_region);
        if (new_token_local_idx.numel() > 0) {
            auto new_token_out_idx =
                    request_idx * num_padding_slots_per_request + new_token_local_idx;
            auto new_token_values = out_idx.masked_select(is_new_token_region);
            out_new_token_indices.index_put_(
                    {new_token_out_idx},
                    new_token_values.to(out_new_token_indices.scalar_type()));
        }

        if (shift_input_ids) {
            // 需要复用 hidden_states 时，顺手建立原始输入索引到新布局索引的映射。
            const int64_t num_input_tokens_this_request =
                    next_query_start - query_start;
            if (num_input_tokens_this_request > 0) {
                // src_idx/mapped_out_idx: [T_in_req] / [T_map]
                auto src_idx =
                        torch::arange(query_start, next_query_start, index_options);
                auto mapped_out_idx = out_idx.slice(
                        0, 0, std::min<int64_t>(num_input_tokens_this_request,
                                                out_idx.size(0)));
                out_hidden_state_mapping.index_put_(
                        {src_idx.slice(0, 0, mapped_out_idx.size(0))},
                        mapped_out_idx.to(out_hidden_state_mapping.scalar_type()));
            }
        }
    }
}

// 根据 target 输入与本轮 sample 结果，构造 EAGLE 分支真正要吃的 input_ids / positions。
// 每条请求都会把“最后一个位置”替换成下一步要预测的 token，并记录这个位置的全局下标。
void prepare_eagle_inputs_precompiled(
        torch::Tensor &last_token_indices, torch::Tensor &eagle_input_ids,
        torch::Tensor &eagle_positions, const torch::Tensor &target_input_ids,
        const torch::Tensor &target_positions, const torch::Tensor &idx_mapping,
        const torch::Tensor &last_sampled,
        const torch::Tensor &next_prefill_tokens, const torch::Tensor &num_sampled,
        const torch::Tensor &num_rejected, const torch::Tensor &query_start_loc) {
    TORCH_CHECK(last_token_indices.is_cuda(),
                "prepare_eagle_inputs_precompiled expects CUDA "
                "last_token_indices");
    TORCH_CHECK(eagle_input_ids.is_cuda() && eagle_positions.is_cuda(),
                "prepare_eagle_inputs_precompiled expects CUDA EAGLE buffers");
    TORCH_CHECK(target_input_ids.is_cuda() && target_positions.is_cuda(),
                "prepare_eagle_inputs_precompiled expects CUDA target buffers");
    TORCH_CHECK(idx_mapping.is_cuda() && last_sampled.is_cuda() &&
                next_prefill_tokens.is_cuda() && num_sampled.is_cuda() &&
                num_rejected.is_cuda() && query_start_loc.is_cuda(),
                "prepare_eagle_inputs_precompiled expects CUDA metadata");

    auto idx_cpu = idx_mapping.to(torch::kCPU);
    auto sampled_cpu = num_sampled.to(torch::kCPU);
    auto rejected_cpu = num_rejected.to(torch::kCPU);
    // query_start_cpu: [B + 1]
    auto query_start_cpu = query_start_loc.to(torch::kCPU);
    // num_reqs: 当前批次请求数
    const int64_t num_reqs = idx_mapping.size(0);

    for (int64_t batch_idx = 0; batch_idx < num_reqs; ++batch_idx) {
        const int64_t req_state_idx = idx_cpu[batch_idx].item<int64_t>();
        const int64_t query_start = query_start_cpu[batch_idx].item<int64_t>();
        const int64_t query_end = query_start_cpu[batch_idx + 1].item<int64_t>();
        const int64_t query_len =
                query_end - query_start - rejected_cpu[batch_idx].item<int64_t>();
        TORCH_CHECK(query_len > 0,
                    "prepare_eagle_inputs_precompiled expects positive query_len");

        torch::Tensor next_token;
        if (sampled_cpu[batch_idx].item<int64_t>() > 0) {
            // decode 请求优先拿上轮真实 sample 的最后一个 token。
            next_token = last_sampled.select(0, req_state_idx);
        } else {
            // chunked prefill 请求则继续拿预取好的 next_prefill token。
            next_token = next_prefill_tokens.select(0, req_state_idx);
        }

        // 除了最后一个 token 外，其余位置都沿用目标输入的对齐结果。
        if (query_len > 1) {
            eagle_input_ids.slice(0, query_start, query_start + query_len - 1)
                    .copy_(target_input_ids
                                   .slice(0, query_start + 1, query_start + query_len)
                                   .to(eagle_input_ids.scalar_type()));
        }

        const int64_t last_token_index = query_start + query_len - 1;
        // last_token_indices: [B]
        last_token_indices.select(0, batch_idx).fill_(last_token_index);
        eagle_input_ids.select(0, last_token_index)
                .copy_(next_token.to(eagle_input_ids.scalar_type()));
        eagle_positions.slice(0, query_start, query_start + query_len)
                .copy_(target_positions
                               .slice(0, query_start, query_start + query_len)
                               .to(eagle_positions.scalar_type()));
    }
}

// 把 EAGLE 草稿 decode 的输入缓冲改造成“单 token decode”布局。
// 同时把 query_start_loc、seq_lens 和 hidden state 指针都推进到下一轮状态。
void prepare_eagle_decode_precompiled(
        const torch::Tensor &draft_tokens, const torch::Tensor &output_hidden_states,
        const torch::Tensor &last_token_indices,
        const torch::Tensor &target_seq_lens, const torch::Tensor &num_rejected,
        torch::Tensor &input_ids, torch::Tensor &positions,
        torch::Tensor &query_start_loc, torch::Tensor &seq_lens,
        torch::Tensor &input_hidden_states, int64_t max_model_len,
        int64_t max_num_reqs) {
    TORCH_CHECK(draft_tokens.is_cuda(),
                "prepare_eagle_decode_precompiled expects CUDA draft_tokens");
    TORCH_CHECK(output_hidden_states.is_cuda(),
                "prepare_eagle_decode_precompiled expects CUDA "
                "output_hidden_states");
    TORCH_CHECK(last_token_indices.is_cuda() && target_seq_lens.is_cuda() &&
                num_rejected.is_cuda(),
                "prepare_eagle_decode_precompiled expects CUDA metadata");
    TORCH_CHECK(input_ids.is_cuda() && positions.is_cuda() &&
                query_start_loc.is_cuda() && seq_lens.is_cuda() &&
                input_hidden_states.is_cuda(),
                "prepare_eagle_decode_precompiled expects CUDA output buffers");

    // num_reqs: 进入单步 decode 的请求数
    const int64_t num_reqs = draft_tokens.size(0);
    auto arange_options = torch::TensorOptions()
            .device(query_start_loc.device())
            .dtype(torch::kInt32);
    // query_start_loc: [B + 1] -> [0, 1, 2, ...]
    query_start_loc.slice(0, 0, num_reqs + 1)
            .copy_(torch::arange(0, num_reqs + 1, arange_options));
    if (max_num_reqs > num_reqs) {
        query_start_loc.slice(0, num_reqs + 1, max_num_reqs + 1).fill_(num_reqs);
        seq_lens.slice(0, num_reqs, max_num_reqs).zero_();
    }
    if (num_reqs == 0) {
        return;
    }

    // decode 阶段每条请求只保留一个输入 token，因此 query_start_loc 会退化成 [0, 1, 2, ...]。
    input_ids.slice(0, 0, num_reqs)
            .copy_(draft_tokens.to(input_ids.scalar_type()));
    auto src_indices = last_token_indices.to(torch::kLong);
    // src_indices: [B]
    // hidden state 需要按“上一轮最后一个有效 token”的位置取回，而不是简单顺序截断。
    input_hidden_states.slice(0, 0, num_reqs)
            .copy_(output_hidden_states.index_select(0, src_indices)
                           .to(input_hidden_states.scalar_type()));

    // 位置编码统一向前推进一格，并在模型上限处截断。
    auto next_positions = positions.slice(0, 0, num_reqs) + 1;
    next_positions =
            torch::minimum(next_positions,
                           torch::full({}, max_model_len - 1,
                                       next_positions.options()));
    positions.slice(0, 0, num_reqs)
            .copy_(next_positions.to(positions.scalar_type()));

    auto next_seq_lens =
            target_seq_lens.slice(0, 0, num_reqs) -
            num_rejected.slice(0, 0, num_reqs) + 1;
    // 新的 seq_lens 要扣掉 reject 的 token，再加回这一步真正进入 decode 的那个 token。
    next_seq_lens =
            torch::minimum(next_seq_lens,
                           torch::full({}, max_model_len, next_seq_lens.options()));
    seq_lens.slice(0, 0, num_reqs)
            .copy_(next_seq_lens.to(seq_lens.scalar_type()));
}

// 在连续草稿 decode 中，把最新的 draft token / hidden state 直接推到下一步输入缓冲。
void update_eagle_inputs_precompiled(
        const torch::Tensor &draft_tokens, const torch::Tensor &output_hidden_states,
        torch::Tensor &input_ids, torch::Tensor &positions,
        torch::Tensor &seq_lens, torch::Tensor &hidden_states,
        int64_t max_model_len) {
    TORCH_CHECK(draft_tokens.is_cuda(),
                "update_eagle_inputs_precompiled expects CUDA draft_tokens");
    TORCH_CHECK(output_hidden_states.is_cuda(),
                "update_eagle_inputs_precompiled expects CUDA "
                "output_hidden_states");
    TORCH_CHECK(input_ids.is_cuda() && positions.is_cuda() &&
                seq_lens.is_cuda() && hidden_states.is_cuda(),
                "update_eagle_inputs_precompiled expects CUDA output buffers");

    // num_reqs: 连续草稿 decode 的请求数
    const int64_t num_reqs = draft_tokens.size(0);
    if (num_reqs == 0) {
        return;
    }
    // draft token 和对应 hidden state 直接覆盖下一轮 decode 的输入缓冲。
    input_ids.slice(0, 0, num_reqs)
            .copy_(draft_tokens.to(input_ids.scalar_type()));
    // output_hidden_states: [B, hidden] -> hidden_states 前 B 行
    hidden_states.slice(0, 0, num_reqs)
            .copy_(output_hidden_states.to(hidden_states.scalar_type()));

    // positions / seq_lens 都按单步 decode 规则同步前推。
    auto next_positions = positions.slice(0, 0, num_reqs) + 1;
    next_positions =
            torch::minimum(next_positions,
                           torch::full({}, max_model_len - 1,
                                       next_positions.options()));
    positions.slice(0, 0, num_reqs)
            .copy_(next_positions.to(positions.scalar_type()));

    auto next_seq_lens = seq_lens.slice(0, 0, num_reqs) + 1;
    next_seq_lens =
            torch::minimum(next_seq_lens,
                           torch::full({}, max_model_len, next_seq_lens.options()));
    seq_lens.slice(0, 0, num_reqs)
            .copy_(next_seq_lens.to(seq_lens.scalar_type()));
}

// ------------------------------- 位置编码、归一化与门控状态兜底算子 -------------------------------

// 把多路 MRoPE cache 应用到 query / key 上，并保留非旋转维度不变。
// 这条参考实现主要负责把 cache 重排与 rotary 乘法的语义对齐到正式 kernel。
std::tuple<torch::Tensor, torch::Tensor> mrope_rotary_embedding(
        const torch::Tensor &query, const torch::Tensor &key,
        const torch::Tensor &cos, const torch::Tensor &sin, int64_t head_size,
        int64_t rotary_dim, c10::List<int64_t> mrope_section, bool is_neox,
        bool mrope_interleaved) {
    TORCH_CHECK(query.is_cuda(), "mrope_rotary_embedding expects CUDA query");
    TORCH_CHECK(key.is_cuda(), "mrope_rotary_embedding expects CUDA key");
    TORCH_CHECK(query.dim() == 2 && key.dim() == 2,
                "mrope_rotary_embedding expects flattened [num_tokens, hidden]");

    // num_tokens: flatten 后的总 token 数
    const int64_t num_tokens = query.size(0);
    // 先把三路 cache 合成真正参与旋转的 cos/sin 视图，避免 query 和 key 各自重复整理。
    // merged_cos/merged_sin: [T, rotary_dim/2]
    auto merged_cos = prepare_mrope_cache(cos, mrope_section, mrope_interleaved);
    auto merged_sin = prepare_mrope_cache(sin, mrope_section, mrope_interleaved);

    // 旋转维度单独做 rotary，剩余维度直接原样拼回，确保 hidden 布局不被破坏。
    // query/key view: [T, H, head_size]
    auto query_view = query.view({num_tokens, -1, head_size});
    // query_rot/query_pass: [T, H, rotary_dim] / [T, H, head_size-rotary_dim]
    auto query_rot = query_view.slice(-1, 0, rotary_dim);
    auto query_pass = query_view.slice(-1, rotary_dim, head_size);
    // query_out: [T, hidden]
    auto query_out = torch::cat(
            {apply_rotary_emb_native(query_rot, merged_cos, merged_sin, is_neox),
             query_pass},
            -1)
            .reshape_as(query);

    // key_rot/key_pass: [T, H, rotary_dim] / [T, H, head_size-rotary_dim]
    auto key_view = key.view({num_tokens, -1, head_size});
    auto key_rot = key_view.slice(-1, 0, rotary_dim);
    auto key_pass = key_view.slice(-1, rotary_dim, head_size);
    // key_out: [T, hidden]
    auto key_out = torch::cat(
            {apply_rotary_emb_native(key_rot, merged_cos, merged_sin, is_neox),
             key_pass},
            -1)
            .reshape_as(key);

    return {query_out, key_out};
}

// 在 Windows 上兜底 gated layer norm / gated RMS norm。
// 这里统一走 float32 参考计算，再在函数末尾转回调用方原始 dtype。
torch::Tensor gated_layer_norm(
        const torch::Tensor &input, const torch::Tensor &weight,
        const std::optional<torch::Tensor> &bias,
        const std::optional<torch::Tensor> &gate, double epsilon,
        int64_t group_size, bool norm_before_gate, bool is_rms_norm,
        const std::string &activation) {
    TORCH_CHECK(input.is_cuda(), "gated_layer_norm expects CUDA input");
    TORCH_CHECK(weight.is_cuda(), "gated_layer_norm expects CUDA weight");
    TORCH_CHECK(input.size(-1) == weight.size(0),
                "weight must match the hidden dimension");

    auto original_shape = input.sizes().vec();
    // original_dtype/hidden/resolved_group_size/num_groups: 归一化维度元数据
    const auto original_dtype = input.scalar_type();
    const int64_t hidden = input.size(-1);
    const int64_t resolved_group_size =
            group_size > 0 ? group_size : hidden;
    TORCH_CHECK(hidden % resolved_group_size == 0,
                "group_size must divide hidden size");
    const int64_t num_groups = hidden / resolved_group_size;

    // 先把输入展平成二维，便于统一处理 group norm 与后续门控分支。
    // x/z: [N, hidden], w: [hidden]
    auto x = input.reshape({-1, hidden}).to(torch::kFloat32);
    auto w = weight.to(torch::kFloat32);
    // z: [N, hidden]，gate 缺失时保持未定义
    auto z = gate.has_value() ? gate.value().reshape({-1, hidden}).to(torch::kFloat32)
                              : torch::Tensor();

    // norm_before_gate 为 false 时，先对输入施加门控，再做归一化。
    if (gate.has_value() && !norm_before_gate) {
        x = x * apply_gate_activation(z, activation);
    }

    torch::Tensor y;
    if (num_groups == 1) {
        // 单组时分别兜底 RMSNorm 和 LayerNorm 的标准公式。
        if (is_rms_norm) {
            // rstd: [N, 1]
            auto rstd = torch::rsqrt(x.square().mean(-1, true) + epsilon);
            y = x * rstd * w;
        } else {
            // mean/centered/var: [N, 1] / [N, hidden] / [N, 1]
            auto mean = x.mean(-1, true);
            auto centered = x - mean;
            auto var = centered.square().mean(-1, true);
            y = centered * torch::rsqrt(var + epsilon) * w;
        }
    } else {
        // x_group: [N, G, group_size], w_group: [G, group_size]
        auto x_group = x.reshape({-1, num_groups, resolved_group_size});
        auto w_group = w.reshape({num_groups, resolved_group_size});
        if (is_rms_norm) {
            // rstd: [N, G, 1]
            auto rstd = torch::rsqrt(x_group.square().mean(-1, true) + epsilon);
            y = (x_group * rstd * w_group).reshape({-1, hidden});
        } else {
            // mean/centered/var: [N, G, 1] / [N, G, group_size] / [N, G, 1]
            auto mean = x_group.mean(-1, true);
            auto centered = x_group - mean;
            auto var = centered.square().mean(-1, true);
            y = (centered * torch::rsqrt(var + epsilon) * w_group)
                    .reshape({-1, hidden});
        }
    }

    // bias 和后置 gate 都放在归一化之后，和正式实现的执行顺序保持一致。
    if (bias.has_value()) {
        y = y + bias.value().to(torch::kFloat32);
    }
    if (gate.has_value() && norm_before_gate) {
        y = y * apply_gate_activation(z, activation);
    }

    return y.reshape(original_shape).to(original_dtype);
}

// 以纯 ATen 方式复刻 recurrent gated delta rule 的逐 token 状态推进。
// 参考实现的重点不是极致性能，而是完整保留 state 读取、更新与输出写回的语义。
std::tuple<torch::Tensor, torch::Tensor>
fused_sigmoid_gating_delta_rule_update_precompiled(
        const torch::Tensor &A_log, const torch::Tensor &a,
        const torch::Tensor &b, const torch::Tensor &dt_bias,
        const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v,
        double beta, double threshold, double scale,
        const torch::Tensor &initial_state, bool inplace_final_state,
        const std::optional<torch::Tensor> &cu_seqlens,
        const std::optional<torch::Tensor> &ssm_state_indices,
        const std::optional<torch::Tensor> &num_accepted_tokens,
        bool use_qk_l2norm_in_kernel, bool is_kda) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
                "precompiled fused sigmoid gating expects CUDA tensors");
    TORCH_CHECK(initial_state.is_cuda(),
                "precompiled fused sigmoid gating expects CUDA initial_state");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q, k, v must have shape [B, T, H/HV, K/V]");

    // B/T/H/K/HV/V/hv_per_h: 批次、长度、Q头、K维、V头、V维和头展开倍数
    const int64_t B = q.size(0);
    const int64_t T = q.size(1);
    const int64_t H = q.size(2);
    const int64_t K = q.size(3);
    const int64_t HV = v.size(2);
    const int64_t V = v.size(3);
    const int64_t hv_per_h = HV / H;
    // output: [B, T, HV, V]
    auto output = torch::empty(v.sizes(), q.options());
    auto final_state = inplace_final_state
                       ? initial_state
                       : q.new_empty({T, HV, V, K}, initial_state.options());
    // output: [B, T, HV, V] 或 [1, T_total, HV, V]

    // A_log_f32/dt_bias_f32: [HV] 或 [HV, K] 的 float32 计算视图
    auto A_log_f32 = A_log.to(torch::kFloat32);
    auto dt_bias_f32 = dt_bias.to(torch::kFloat32);
    // q_tokens/k_tokens/v_tokens/a_tokens/b_tokens: 统一成 token-major 视图
    torch::Tensor q_tokens;
    torch::Tensor k_tokens;
    torch::Tensor v_tokens;
    torch::Tensor a_tokens;
    torch::Tensor b_tokens;
    std::vector<std::pair<int64_t, int64_t>> seq_ranges;

    // 先把定长 batch 和 varlen 两种输入布局统一成 token 级扫描视图。
    if (!cu_seqlens.has_value()) {
        // q_tokens/k_tokens: [B*T, H, K], v_tokens: [B*T, HV, V]
        q_tokens = q.reshape({B * T, H, K});
        k_tokens = k.reshape({B * T, H, K});
        v_tokens = v.reshape({B * T, HV, V});
        a_tokens = is_kda ? a.reshape({B * T, HV, K}) : a.reshape({B * T, HV});
        b_tokens = b.reshape({B * T, HV});
        // seq_ranges: 每条序列在 token-major 视图中的 [bos, eos)
        seq_ranges.reserve(B);
        for (int64_t seq_idx = 0; seq_idx < B; ++seq_idx) {
            seq_ranges.emplace_back(seq_idx * T, (seq_idx + 1) * T);
        }
    } else {
        // varlen 模式直接取 batch=1 的 token 视图。
        const auto &seq = cu_seqlens.value();
        q_tokens = q[0];
        k_tokens = k[0];
        v_tokens = v[0];
        a_tokens = is_kda ? a.reshape({-1, HV, K}) : a.reshape({-1, HV});
        b_tokens = b.reshape({-1, HV});
        const int64_t num_seqs = seq.size(0) - 1;
        // seq_ranges: varlen 模式下直接来自 cu_seqlens
        seq_ranges.reserve(num_seqs);
        for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
            seq_ranges.emplace_back(seq[seq_idx].item<int64_t>(),
                                    seq[seq_idx + 1].item<int64_t>());
        }
    }

    auto out_tokens = output.reshape({-1, HV, V});

    // 逐序列、逐 token 推进隐状态，确保 Windows 参考路径和正式状态机顺序完全一致。
    for (int64_t seq_idx = 0; seq_idx < static_cast<int64_t>(seq_ranges.size());
         ++seq_idx) {
        const auto [bos, eos] = seq_ranges[seq_idx];
        const int64_t seq_len = eos - bos;
        if (seq_len <= 0) {
            continue;
        }

        torch::Tensor h;
        if (!initial_state.defined()) {
            // h: [HV, V, K]，没有初始状态时从全零开始
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
                // -2 表示没有显式索引表，回退到全零状态。
                h = torch::zeros({HV, V, K},
                                 torch::TensorOptions().dtype(torch::kFloat32)
                                         .device(q.device()));
            } else if (state_idx < 0) {
                // 负索引表示整条序列无效，直接把输出清零后跳过。
                out_tokens.slice(0, bos, eos).zero_();
                continue;
            } else {
                // 从请求状态表里取出对应的初始 h: [HV, V, K]
                h = initial_state[state_idx].to(torch::kFloat32).clone();
            }
        } else {
            // 没有显式 state index 时，优先按 seq_idx 对齐；必要时回退到 token base。
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
            // q_token/k_token: [HV, K], v_token: [HV, V]
            auto beta_token = torch::sigmoid(b_tokens[token_idx].to(torch::kFloat32));

            torch::Tensor g;
            if (!is_kda) {
                // x: [HV]
                auto x = a_tokens[token_idx].to(torch::kFloat32) + dt_bias_f32;
                // g: [HV]
                g = -torch::exp(A_log_f32) * softplus_with_threshold(x, beta, threshold);
            } else {
                // x: [HV, K]
                auto x =
                        a_tokens[token_idx].to(torch::kFloat32) + dt_bias_f32.unsqueeze(1);
                // g: [HV, K]
                g = -torch::exp(A_log_f32).unsqueeze(1) *
                    softplus_with_threshold(x, beta, threshold);
            }

            if (use_qk_l2norm_in_kernel) {
                q_token = l2norm_last_dim(q_token);
                k_token = l2norm_last_dim(k_token);
            }

            // q_token: [HV, K]
            q_token = q_token * scale;
            if (!is_kda) {
                // 非 KDA 模式下 gate 只沿 HV 生效。
                h = h * torch::exp(g).view({HV, 1, 1});
            } else {
                // KDA 模式下 gate 带 K 维，需要保留到状态最后一维。
                h = h * torch::exp(g).unsqueeze(1);
            }

            auto updated_v =
                    v_token - (h * k_token.unsqueeze(1)).sum(-1);
            // updated_v: [HV, V], h: [HV, V, K]
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

    // 某些 state index 可能被标记成无效，这里需要把对应输出明确清零。
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

// 对普通 RoPE 提供一条不依赖 Triton 的参考实现。
// 这里统一先整理输入 rank，再按 NeoX 与交错两种布局分别应用旋转。
torch::Tensor apply_rotary_emb_precompiled(const torch::Tensor &x,
                                           const torch::Tensor &cos,
                                           const torch::Tensor &sin,
                                           bool is_neox_style,
                                           bool enable_fp32_compute) {
    TORCH_CHECK(x.is_cuda(), "apply_rotary_emb_precompiled expects CUDA x");
    TORCH_CHECK(cos.is_cuda(), "apply_rotary_emb_precompiled expects CUDA cos");
    TORCH_CHECK(sin.is_cuda(), "apply_rotary_emb_precompiled expects CUDA sin");
    TORCH_CHECK(x.dim() == 3 || x.dim() == 4,
                "apply_rotary_emb_precompiled expects x with rank 3 or 4");

    // working_x/working_cos/working_sin: 需要时统一提升到 float32 计算
    auto working_x = enable_fp32_compute ? x.to(torch::kFloat32) : x;
    auto working_cos = enable_fp32_compute ? cos.to(torch::kFloat32) : cos;
    auto working_sin = enable_fp32_compute ? sin.to(torch::kFloat32) : sin;
    const bool added_batch = working_x.dim() == 3;
    if (added_batch) {
        // 统一补成四维，避免后续分支再区分 batch 缺失场景。
        working_x = working_x.unsqueeze(0);
    }

    // cos_expanded/sin_expanded: [B|1, T, 1, rotary_dim/2]
    auto cos_expanded = working_cos.unsqueeze(-2).to(working_x.scalar_type());
    auto sin_expanded = working_sin.unsqueeze(-2).to(working_x.scalar_type());

    torch::Tensor output;
    if (is_neox_style) {
        // parts/x1/x2: [..., rotary_dim/2]
        auto parts = working_x.split(working_x.size(-1) / 2, -1);
        auto x1 = parts[0];
        auto x2 = parts[1];
        // o1/o2: [..., rotary_dim/2]
        auto o1 = x1 * cos_expanded - x2 * sin_expanded;
        auto o2 = x2 * cos_expanded + x1 * sin_expanded;
        output = torch::cat({o1, o2}, -1);
    } else {
        // x1/x2: [..., rotary_dim/2]
        auto x1 = working_x.slice(-1, 0, working_x.size(-1), 2);
        auto x2 = working_x.slice(-1, 1, working_x.size(-1), 2);
        // o1/o2: [..., rotary_dim/2]
        auto o1 = x1 * cos_expanded - x2 * sin_expanded;
        auto o2 = x2 * cos_expanded + x1 * sin_expanded;
        // 交错布局旋转后再还原回 [..., rotary_dim]
        output = torch::stack({o1, o2}, -1).flatten(-2);
    }

    if (added_batch) {
        // 原输入没有 batch 维时，把补出来的维度再去掉。
        output = output.squeeze(0);
    }
    if (enable_fp32_compute) {
        // 最终输出 dtype 与输入 x 保持一致。
        output = output.to(x.scalar_type());
    }
    return output;
}

// ------------------------------- Chunked Gated Delta Rule 与矩阵参考算子 -------------------------------

// 逐 token 复刻 chunk gated delta rule 的状态衰减、增量更新和输出投影。
// 该实现把 query/key 头数先扩展到 value 头数，再按序列顺序串行推进状态。
std::tuple<torch::Tensor, torch::Tensor> chunk_gated_delta_rule_precompiled(
        const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v,
        const torch::Tensor &g, const torch::Tensor &beta, double scale,
        const torch::Tensor &initial_state, bool output_final_state,
        const std::optional<torch::Tensor> &cu_seqlens,
        bool use_qk_l2norm_in_kernel) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && g.is_cuda() &&
                beta.is_cuda() && initial_state.is_cuda(),
                "chunk_gated_delta_rule_precompiled expects CUDA tensors");

    // B/T/H/K/V/N: 批次、长度、value 头数、状态 K/V 维与序列数
    const int64_t B = q.size(0);
    const int64_t T = q.size(1);
    const int64_t H = v.size(2);
    const int64_t K = q.size(3);
    const int64_t V = v.size(3);
    const int64_t N = cu_seqlens.has_value() ? cu_seqlens.value().size(0) - 1 : B;

    // q_work/k_work: [B, T, H, K]
    auto q_work = expand_qk_heads(q.to(torch::kFloat32), H);
    auto k_work = expand_qk_heads(k.to(torch::kFloat32), H);
    if (use_qk_l2norm_in_kernel) {
        q_work = l2norm_last_dim(q_work);
        k_work = l2norm_last_dim(k_work);
    }
    // v_work/g_work/beta_work: [B, T, H, V] / [B, T, H] / [B, T, H]
    auto v_work = v.to(torch::kFloat32);
    auto g_work = g.to(torch::kFloat32);
    auto beta_work = beta.to(torch::kFloat32);

    // output/final_state: [B, T, H, V] / [N, H, V, K]
    auto output = torch::empty({B, T, H, V}, q.options());
    auto final_state = output_final_state
                       ? torch::empty(
                    {N, H, V, K},
                    torch::TensorOptions().dtype(torch::kFloat32).device(q.device()))
                       : torch::empty(
                    {0},
                    torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));

    // varlen 模式下按 cu_seqlens 切每条序列，定长模式则直接逐 batch 扫描。
    for (int64_t seq_idx = 0; seq_idx < N; ++seq_idx) {
        int64_t batch_idx = seq_idx;
        int64_t start = 0;
        int64_t end = T;
        if (cu_seqlens.has_value()) {
            batch_idx = 0;
            start = cu_seqlens.value()[seq_idx].item<int64_t>();
            end = cu_seqlens.value()[seq_idx + 1].item<int64_t>();
        }

        // state: [H, V, K]
        auto state = initial_state[seq_idx].to(torch::kFloat32).clone();
        for (int64_t tok_idx = start; tok_idx < end; ++tok_idx) {
            // q_tok/k_tok/v_tok: [H, K] / [H, K] / [H, V]
            auto q_tok = q_work.index({batch_idx, tok_idx});
            auto k_tok = k_work.index({batch_idx, tok_idx});
            auto v_tok = v_work.index({batch_idx, tok_idx});
            // beta_tok/decay: [H, 1] / [H, 1, 1]
            auto beta_tok = beta_work.index({batch_idx, tok_idx}).unsqueeze(-1);
            auto decay =
                    torch::exp(g_work.index({batch_idx, tok_idx})).unsqueeze(-1).unsqueeze(-1);
            // 每一步都先按门控衰减历史状态，再写入当前 token 的增量。
            state = state * decay;
            // delta_v: [H, V]
            auto delta_v =
                    v_tok - torch::matmul(state, k_tok.unsqueeze(-1)).squeeze(-1);
            delta_v = delta_v * beta_tok;
            state = state + delta_v.unsqueeze(-1) * k_tok.unsqueeze(-2);
            // 输出当前 token 的投影结果: [H, V]
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

// 针对 packed decode 场景，把混排的 qkv / gating 输入拆开并执行一步 recurrent 更新。
// 这条兜底路径直接对 initial_state 原地回写，保持与正式 decode kernel 的副作用一致。
std::tuple<torch::Tensor, torch::Tensor>
fused_recurrent_gated_delta_rule_packed_decode_precompiled(
        const torch::Tensor &mixed_qkv, const torch::Tensor &a,
        const torch::Tensor &b, const torch::Tensor &A_log,
        const torch::Tensor &dt_bias, double scale, torch::Tensor &initial_state,
        torch::Tensor &out, const torch::Tensor &ssm_state_indices,
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

    // B/HV/V/K: packed decode 的批次与状态张量维度
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

    // index_options: 生成 head 映射索引用的 [long] 配置
    auto index_options =
            torch::TensorOptions().dtype(torch::kLong).device(mixed_qkv.device());

    // safe_indices/valid_mask/gather_indices: [B]
    auto safe_indices =
            ssm_state_indices.to(mixed_qkv.device(), torch::kLong, false, false)
                    .contiguous();
    auto valid_mask = safe_indices >= 0;
    auto gather_indices = torch::clamp_min(safe_indices, 0);

    // mixed_qkv 里先是 q，再是 k，最后是 v；这里需要按推导出的 head 结构把它拆开。
    // mixed_qkv_f: [B, q_dim + q_dim + HV*V]
    auto mixed_qkv_f = mixed_qkv.to(torch::kFloat32);
    // packed_q/packed_k/packed_v: [B, H, K] / [B, H, K] / [B, HV, V]
    auto packed_q = mixed_qkv_f.slice(1, 0, q_dim).view({B, H, K});
    auto packed_k = mixed_qkv_f.slice(1, q_dim, 2 * q_dim).view({B, H, K});
    auto packed_v = mixed_qkv_f.slice(1, 2 * q_dim, qkv_dim).view({B, HV, V});

    std::vector<int64_t> h_index_vec(HV);
    for (int64_t idx = 0; idx < HV; ++idx) {
        h_index_vec[idx] = idx / hv_per_h;
    }
    // h_indices: [HV]
    auto h_indices = torch::tensor(h_index_vec, index_options);

    // q/k: [B, HV, K]
    auto q = packed_q.index_select(1, h_indices);
    auto k = packed_k.index_select(1, h_indices);
    if (use_qk_l2norm_in_kernel) {
        q = l2norm_last_dim(q);
        k = l2norm_last_dim(k);
    }
    q = q * scale;

    // h: [B, HV, V, K]
    auto h = initial_state.index_select(0, gather_indices).to(torch::kFloat32);
    // g_input: [B, HV]
    auto g_input =
            a.to(torch::kFloat32) + dt_bias.to(torch::kFloat32).view({1, HV});
    // g/beta: [B, HV, 1, 1] / [B, HV, 1]
    auto g =
            (-torch::exp(A_log.to(torch::kFloat32)).view({1, HV, 1, 1}) *
             softplus_with_threshold(g_input, 1.0, 20.0).view({B, HV, 1, 1}));
    auto beta = torch::sigmoid(b.to(torch::kFloat32)).view({B, HV, 1});

    h = h * torch::exp(g);
    // v/updated_h/out_values: [B, HV, V] / [B, HV, V, K] / [B, HV, V]
    auto v = (packed_v - (h * k.unsqueeze(-2)).sum(-1)) * beta;
    auto updated_h = h + v.unsqueeze(-1) * k.unsqueeze(-2);
    auto out_values = (updated_h * q.unsqueeze(-2)).sum(-1);

    // out_slice: [B, HV, V]
    auto out_slice = out.select(1, 0);
    out_slice.copy_(torch::where(valid_mask.view({B, 1, 1}), out_values,
                                 torch::zeros_like(out_values))
                            .to(out.scalar_type()));

    // 只有 state index 有效的行才允许回写，避免 pad 请求污染真实状态。
    if (valid_mask.any().item<bool>()) {
        // valid_rows/valid_gather_indices: [B_valid]
        auto valid_rows = torch::nonzero(valid_mask).view(-1);
        auto valid_gather_indices = gather_indices.index_select(0, valid_rows);
        // valid_updated_h: [B_valid, HV, V, K]
        auto valid_updated_h =
                updated_h.to(initial_state.scalar_type()).index_select(0, valid_rows);
        initial_state.index_copy_(0, valid_gather_indices, valid_updated_h);
    }

    return {out, initial_state};
}

// 对最后一维执行 L2 归一化，并保持外层批次结构不变。
torch::Tensor l2norm_precompiled(
        const torch::Tensor &x, double eps,
        const std::optional<torch::ScalarType> &output_dtype) {
    TORCH_CHECK(x.is_cuda(), "l2norm_precompiled expects CUDA input");
    TORCH_CHECK(x.dim() >= 1, "l2norm_precompiled expects rank >= 1");

    // original_sizes: 原始外层形状
    const auto original_sizes = x.sizes().vec();
    // x_contiguous/x_flat: [..., K] -> [N_flat, K]
    auto x_contiguous = x.contiguous();
    auto x_flat = x_contiguous.view({-1, x_contiguous.size(-1)});
    // y: [N_flat, K]
    auto y = l2norm_last_dim(x_flat.to(torch::kFloat32));
    auto resolved_dtype = output_dtype.value_or(x.scalar_type());
    return y.to(resolved_dtype).view(original_sizes);
}

// 在每个 chunk 内独立做前向或反向 cumsum，不跨 chunk 传播历史。
// 这与正式 kernel 的局部块语义一致，便于后续按 chunk 拼接更大状态机。
torch::Tensor chunk_local_cumsum_precompiled(
        const torch::Tensor &g, int64_t chunk_size, bool reverse,
        const std::optional<torch::Tensor> &cu_seqlens, bool head_first,
        const std::optional<torch::ScalarType> &output_dtype) {
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

    // seq_major: [B, T, ...]
    auto seq_major = g.contiguous();
    if (head_first) {
        seq_major = g.dim() == 3 ? seq_major.permute({0, 2, 1}).contiguous()
                                 : seq_major.permute({0, 2, 1, 3}).contiguous();
    }

    const auto resolved_dtype = output_dtype.value_or(g.scalar_type());
    // out: 与 seq_major 同形状
    auto out = torch::empty(seq_major.sizes(),
                            seq_major.options().dtype(resolved_dtype));

    // 每条序列单独处理，保证 chunk 边界不会跨请求泄漏累计结果。
    auto write_sequence = [&](const torch::Tensor &src_seq,
                              torch::Tensor dst_seq) {
        const int64_t token_count = src_seq.size(0);
        if (token_count == 0) {
            return;
        }
        const int64_t flat_width = src_seq.numel() / token_count;
        // src_flat/dst_flat: [T_seq, flat_width]
        auto src_flat =
                src_seq.reshape({token_count, flat_width}).to(torch::kFloat32);
        auto dst_flat = torch::empty({token_count, flat_width},
                                     src_flat.options().dtype(torch::kFloat32));

        for (int64_t chunk_start = 0; chunk_start < token_count;
             chunk_start += chunk_size) {
            const int64_t chunk_end = std::min(chunk_start + chunk_size, token_count);
            // chunk/chunk_out: [T_chunk, flat_width]
            auto chunk = src_flat.slice(0, chunk_start, chunk_end);
            // reverse 模式等价于先翻转、再做正向 cumsum、最后翻回原顺序。
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

// 根据 chunk 状态张量 h、局部 attention 和可选门控，恢复每个 token 的输出。
// 这里的实现是逐 chunk、逐 head 参考计算，重点在语义正确而不是吞吐。
torch::Tensor chunk_fwd_o_precompiled(
        const torch::Tensor &q, const torch::Tensor &k, const torch::Tensor &v,
        const torch::Tensor &h, const std::optional<torch::Tensor> &g,
        double scale, const std::optional<torch::Tensor> &cu_seqlens,
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

    // B/T/Hg/K/H/V: chunk 输出恢复的维度元数据
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
    // h_f: [B, N_chunk, H, V, K]
    auto h_f = h.to(torch::kFloat32);
    std::optional<torch::Tensor> g_f = std::nullopt;
    if (g.has_value()) {
        g_f = g.value().to(torch::kFloat32);
    }
    // out: [B, T, H, V]
    auto out = torch::empty_like(v);

    const int64_t heads_per_group = H / Hg;

    // 一个 chunk 内先取出对应的 q/k/v 视图，再按 head 独立拼出输出。
    auto write_chunk = [&](int64_t batch_idx, int64_t chunk_h_idx,
                           int64_t start, int64_t end) {
        if (end <= start) {
            return;
        }
        auto q_chunk_all = q_f[batch_idx].slice(0, start, end);
        auto k_chunk_all = k_f[batch_idx].slice(0, start, end);
        auto v_chunk_all = v_f[batch_idx].slice(0, start, end);
        // q_chunk_all/k_chunk_all: [T_chunk, Hg, K], v_chunk_all: [T_chunk, H, V]
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
            // q_chunk/k_chunk/v_chunk/h_chunk: [T_chunk, K] / [T_chunk, K] / [T_chunk, V] / [V, K]

            // out_chunk 是状态项贡献，attn_chunk 是本 chunk 内的下三角局部注意力贡献。
            // out_chunk/attn_chunk: [T_chunk, V] / [T_chunk, T_chunk]
            auto out_chunk = torch::matmul(q_chunk, h_chunk.transpose(0, 1));
            auto attn_chunk = torch::matmul(q_chunk, k_chunk.transpose(0, 1));
            if (g_chunk_all.has_value()) {
                // g_chunk/g_exp: [T_chunk] / [T_chunk, 1]
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

// 计算每个 chunk 内的下三角 K * K^T 参考结果，并按 beta / g 施加缩放。
torch::Tensor chunk_scaled_dot_kkt_fwd_precompiled(
        const torch::Tensor &k, const std::optional<torch::Tensor> &g,
        const torch::Tensor &beta, const std::optional<torch::Tensor> &cu_seqlens,
        int64_t chunk_size,
        const std::optional<torch::ScalarType> &output_dtype) {
    TORCH_CHECK(k.is_cuda() && beta.is_cuda(),
                "chunk_scaled_dot_kkt_fwd_precompiled expects CUDA k/beta");
    TORCH_CHECK(k.dim() == 4 && beta.dim() == 3,
                "chunk_scaled_dot_kkt_fwd_precompiled expects k rank 4 and "
                "beta rank 3");
    TORCH_CHECK(k.size(0) == beta.size(0) && k.size(1) == beta.size(1),
                "chunk_scaled_dot_kkt_fwd_precompiled expects shared batch/time");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

    // B/T/Hg/K/H: chunk K*K^T 参考实现的维度元数据
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
    // out: [B, T, H, chunk_size]
    auto out = torch::zeros({B, T, H, chunk_size},
                            k.options().dtype(resolved_dtype));
    // k_f/beta_f: [B, T, Hg, K] / [B, T, H]
    auto k_f = k.to(torch::kFloat32);
    auto beta_f = beta.to(torch::kFloat32);
    std::optional<torch::Tensor> g_f = std::nullopt;
    if (g.has_value()) {
        g_f = g.value().to(torch::kFloat32);
    }

    const int64_t heads_per_group = H / Hg;

    // 逐 chunk 构造下三角相关矩阵，多余列保持为零，方便与固定 chunk_size 输出对齐。
    auto write_chunk = [&](int64_t batch_idx, int64_t start, int64_t end) {
        if (end <= start) {
            return;
        }
        auto k_chunk_all = k_f[batch_idx].slice(0, start, end);
        auto beta_chunk_all = beta_f[batch_idx].slice(0, start, end);
        // k_chunk_all/beta_chunk_all: [T_chunk, Hg, K] / [T_chunk, H]
        std::optional<torch::Tensor> g_chunk_all = std::nullopt;
        if (g_f.has_value()) {
            g_chunk_all = g_f.value()[batch_idx].slice(0, start, end);
        }
        const int64_t chunk_len = end - start;

        for (int64_t head_idx = 0; head_idx < H; ++head_idx) {
            const int64_t group_idx = head_idx / heads_per_group;
            // k_chunk/beta_chunk: [T_chunk, K] / [T_chunk]
            auto k_chunk = k_chunk_all.select(1, group_idx);
            auto beta_chunk = beta_chunk_all.select(1, head_idx);
            // attn_chunk: [T_chunk, T_chunk]
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

// 以前向扫描方式为每个 chunk 生成状态快照 h，并可选返回增量值 v_new 与最终状态。
// 这是后续 chunk 输出恢复算子的上游参考实现，负责把 chunk 边界上的状态语义固定下来。
std::tuple<torch::Tensor, std::optional<torch::Tensor>,
        std::optional<torch::Tensor>>
chunk_gated_delta_rule_fwd_h_precompiled(
        const torch::Tensor &k, const torch::Tensor &w, const torch::Tensor &u,
        const std::optional<torch::Tensor> &g,
        const std::optional<torch::Tensor> &gk,
        const std::optional<torch::Tensor> &initial_state, bool output_final_state,
        int64_t chunk_size, bool save_new_value,
        const std::optional<torch::Tensor> &cu_seqlens) {
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

    // B/T/Hg/K/H/V: chunk 状态前向扫描的维度元数据
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
    // chunk_dim: 每条 batch 或全局 varlen 的 chunk 数
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

    // h 保存每个 chunk 开始前的状态快照；可选的 v_new / final_state 分别服务于调试与后续算子。
    // h: [B, N_chunk, H, V, K]
    auto h = torch::empty({B, chunk_dim, H, V, K}, k.options());
    std::optional<torch::Tensor> v_new = std::nullopt;
    if (save_new_value) {
        // v_new: [B, T, H, V]
        v_new = torch::empty_like(u);
    }
    std::optional<torch::Tensor> final_state = std::nullopt;
    if (output_final_state) {
        // final_state: [N, H, V, K]
        final_state = torch::empty({N, H, V, K}, k.options().dtype(torch::kFloat32));
    }

    auto k_f = k.to(torch::kFloat32);
    auto w_f = w.to(torch::kFloat32);
    auto u_f = u.to(torch::kFloat32);
    // k_f/w_f/u_f: [B, T, Hg|H, K|V]
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
        // initial_state_f: [N, H, V, K]
        initial_state_f = initial_state.value().to(torch::kFloat32);
    }

    const int64_t heads_per_group = H / Hg;
    int64_t global_chunk_idx = 0;

    // 每条序列内部逐 chunk 推进状态，并在 chunk 边界记录快照。
    auto process_sequence = [&](int64_t seq_idx, int64_t batch_idx, int64_t seq_start,
                                int64_t seq_end) {
        std::vector<torch::Tensor> state_per_head;
        // state_per_head[head]: [V, K]
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
                // h 记录的是进入当前 chunk 之前的状态，供后续 chunk_fwd_o 等算子复用。
                h[batch_idx][h_chunk_idx][head_idx].copy_(
                        state.to(h.scalar_type()));

                auto k_chunk = k_f[batch_idx].slice(0, chunk_start, chunk_end).select(
                        1, group_idx);
                auto w_chunk = w_f[batch_idx].slice(0, chunk_start, chunk_end).select(
                        1, head_idx);
                auto u_chunk = u_f[batch_idx].slice(0, chunk_start, chunk_end).select(
                        1, head_idx);
                // k_chunk/w_chunk/u_chunk: [T_chunk, K] / [T_chunk, K] / [T_chunk, V]
                // delta_value: [T_chunk, V]
                auto delta_value =
                        u_chunk - torch::matmul(w_chunk, state.transpose(0, 1));

                if (v_new.has_value()) {
                    // v_new 保存尚未乘上当前 chunk key 的增量，便于后续排查数值链路。
                    v_new.value()[batch_idx].slice(0, chunk_start, chunk_end)
                            .select(1, head_idx)
                            .copy_(delta_value.to(v_new.value().scalar_type()));
                }

                if (g_f.has_value()) {
                    // g_chunk: [T_chunk]
                    auto g_chunk = g_f.value()[batch_idx].slice(0, chunk_start, chunk_end)
                            .select(1, head_idx);
                    auto g_last = g_chunk[-1];
                    delta_value =
                            delta_value * torch::exp(g_last - g_chunk).unsqueeze(-1);
                    state = state * torch::exp(g_last);
                }
                if (gk_f.has_value()) {
                    // gk_last: [K]
                    auto gk_last = gk_f.value()[batch_idx][chunk_end - 1][head_idx];
                    state = state * torch::exp(gk_last).unsqueeze(0);
                }

                // state: [V, K]
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

// 对每个 chunk 的严格下三角矩阵求 (I + tril(A, -1)) 的逆。
// 这是 Windows 参考路径里最直接的线性代数写法，便于验证 chunk 求解逻辑。
torch::Tensor solve_tril_precompiled(
        const torch::Tensor &A, const std::optional<torch::Tensor> &cu_seqlens,
        const std::optional<torch::ScalarType> &output_dtype) {
    TORCH_CHECK(A.is_cuda(), "solve_tril_precompiled expects CUDA input");
    TORCH_CHECK(A.dim() == 4,
                "solve_tril_precompiled expects rank-4 input [B, T, H, BT]");

    // B/T/H/BT: 三角求解输入的批次、长度、头数与块大小
    const int64_t B = A.size(0);
    const int64_t T = A.size(1);
    const int64_t H = A.size(2);
    const int64_t BT = A.size(3);
    TORCH_CHECK(BT == 16 || BT == 32 || BT == 64,
                "solve_tril_precompiled expects BT in {16, 32, 64}");

    const auto resolved_dtype = output_dtype.value_or(torch::kFloat32);
    // out: [B, T, H, BT]
    auto out = torch::zeros_like(A, A.options().dtype(resolved_dtype));

    // 每个 chunk 独立求解，避免不同 chunk 之间互相污染矩阵结构。
    auto write_chunk = [&](int64_t batch_idx, int64_t start, int64_t end) {
        if (end <= start) {
            return;
        }
        const int64_t chunk_len = end - start;
        auto A_chunk = A[batch_idx].slice(0, start, end).to(torch::kFloat32);
        // A_chunk: [T_chunk, H, BT]
        // identity: [T_chunk, T_chunk]
        auto identity = torch::eye(chunk_len, A.options().dtype(torch::kFloat32));

        for (int64_t head_idx = 0; head_idx < H; ++head_idx) {
            // lower/inverse: [T_chunk, T_chunk]
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

// 根据 A、beta 与累计门控值，重建 chunk 内部用到的 w / u 中间量。
// 这样后续状态推进算子就可以直接消费统一格式的参考张量。
std::tuple<torch::Tensor, torch::Tensor> recompute_w_u_fwd_precompiled(
        const torch::Tensor &k, const torch::Tensor &v,
        const torch::Tensor &beta, const torch::Tensor &g_cumsum,
        const torch::Tensor &A, const std::optional<torch::Tensor> &cu_seqlens) {
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

    // B/T/Hg/K/H/V/BT: 重建 w/u 时使用的核心维度
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
    // w/u: [B, T, H, K] / [B, T, H, V]

    auto k_f = k.to(torch::kFloat32);
    auto v_f = v.to(torch::kFloat32);
    // beta_f/g_f/A_f: [B, T, H] / [B, T, H] / [B, T, H, BT]
    auto beta_f = beta.to(torch::kFloat32);
    auto g_f = torch::exp(g_cumsum.to(torch::kFloat32));
    auto A_f = A.to(torch::kFloat32);

    const int64_t heads_per_group = H / Hg;

    // 每个 chunk 独立恢复 w/u，保证 varlen 与定长批处理共享同一套语义。
    auto write_chunk = [&](int64_t batch_idx, int64_t start, int64_t end) {
        if (end <= start) {
            return;
        }
        const int64_t chunk_len = end - start;
        // A_chunk_all: [T_chunk, H, T_chunk]
        auto A_chunk_all =
                A_f[batch_idx].slice(0, start, end).slice(2, 0, chunk_len);

        for (int64_t head_idx = 0; head_idx < H; ++head_idx) {
            const int64_t group_idx = head_idx / heads_per_group;
            // beta_chunk/g_chunk: [T_chunk, 1]
            auto beta_chunk =
                    beta_f[batch_idx].slice(0, start, end).select(1, head_idx).unsqueeze(-1);
            auto g_chunk =
                    g_f[batch_idx].slice(0, start, end).select(1, head_idx).unsqueeze(-1);
            auto A_chunk = A_chunk_all.select(1, head_idx);
            auto v_chunk =
                    v_f[batch_idx].slice(0, start, end).select(1, head_idx);
            auto k_chunk =
                    k_f[batch_idx].slice(0, start, end).select(1, group_idx);
            // A_chunk/v_chunk/k_chunk: [T_chunk, T_chunk] / [T_chunk, V] / [T_chunk, K]

            // u_chunk/w_chunk: [T_chunk, V] / [T_chunk, K]
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

// 单独提取 gated delta network 里最常见的 g / beta 计算。
// 这个小兜底算子用于把门控公式从大 kernel 中拆出来，方便 Python 侧组合调用。
std::tuple<torch::Tensor, torch::Tensor> fused_gdn_gating_precompiled(
        const torch::Tensor &A_log, const torch::Tensor &a,
        const torch::Tensor &b, const torch::Tensor &dt_bias, double beta,
        double threshold) {
    TORCH_CHECK(A_log.is_cuda() && a.is_cuda() && b.is_cuda() && dt_bias.is_cuda(),
                "fused_gdn_gating_precompiled expects CUDA tensors");
    // x: [H] 或 [B, H]
    auto x = a.to(torch::kFloat32) + dt_bias.to(torch::kFloat32);
    // g: [1, H] 或 [1, B, H]
    auto g =
            (-torch::exp(A_log.to(torch::kFloat32)) *
             softplus_with_threshold(x, beta, threshold))
                    .unsqueeze(0);
    // beta_output: [1, H] 或 [1, B, H]
    auto beta_output =
            torch::sigmoid(b.to(torch::kFloat32)).to(b.scalar_type()).unsqueeze(0);
    // 返回值和正式算子保持一致：第一项是衰减项 g，第二项是门控权重 beta。
    return {g, beta_output};
}

// ------------------------------- 卷积缓存与 MoE 运行时兜底算子 -------------------------------

// 对每条请求执行一次因果一维卷积，并把新的历史窗口回写到 conv_states。
// 这条实现使用显式的历史拼接和逐序列扫描，重点保持 cache 行为与正式 kernel 一致。
torch::Tensor causal_conv1d_fn_precompiled(
        const torch::Tensor &x, const torch::Tensor &weight,
        const std::optional<torch::Tensor> &bias, torch::Tensor &conv_states,
        const torch::Tensor &query_start_loc,
        const std::optional<torch::Tensor> &cache_indices,
        const std::optional<torch::Tensor> &has_initial_state,
        const std::string &activation, int64_t pad_slot_id) {
    TORCH_CHECK(x.is_cuda(), "causal_conv1d_fn_precompiled expects CUDA x");
    TORCH_CHECK(weight.is_cuda(),
                "causal_conv1d_fn_precompiled expects CUDA weight");
    TORCH_CHECK(conv_states.is_cuda(),
                "causal_conv1d_fn_precompiled expects CUDA conv_states");
    TORCH_CHECK(x.dim() == 2,
                "causal_conv1d_fn_precompiled expects x with shape [dim, tokens]");

    // out: [D, T_total]
    auto out = torch::zeros_like(x);
    // history_len/batch: 卷积历史窗口长度与请求数
    const int64_t history_len = weight.size(1) - 1;
    const int64_t batch = query_start_loc.numel() - 1;

    // 每条请求独立读取自己的缓存槽位，卷积结束后再把新窗口写回。
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
        // history: [D, K-1], seq_tokens: [D, T_seq]
        auto history = load_initial_history(state, history_len, load_initial_state);
        auto seq_tokens = x.slice(1, seq_start, seq_end);
        out.slice(1, seq_start, seq_end)
                .copy_(causal_conv1d_sequence_ref(seq_tokens, weight, bias, history,
                                                  activation));
        // cache 中始终保留最新 history_len 个历史 token，供下一次 decode 继续衔接。
        conv_states[cache_line].copy_(update_conv_state_ref(
                conv_states[cache_line], seq_tokens, seq_tokens.size(-1)));
    }

    return out;
}

// 统计每个本地 expert 实际接收到的 token 数。
// expert_map 存在时，先把全局 expert id 映射成当前 rank 的本地 expert id。
torch::Tensor count_expert_num_tokens_precompiled(
        const torch::Tensor &topk_ids, int64_t num_local_experts,
        const std::optional<torch::Tensor> &expert_map) {
    TORCH_CHECK(topk_ids.is_cuda(),
                "count_expert_num_tokens_precompiled expects CUDA topk_ids");
    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32 ||
                topk_ids.scalar_type() == torch::kInt64,
                "count_expert_num_tokens_precompiled expects signed int32/int64 "
                "topk_ids");
    TORCH_CHECK(num_local_experts >= 0,
                "count_expert_num_tokens_precompiled expects non-negative "
                "num_local_experts");

    auto output = torch::zeros(
            {num_local_experts},
            topk_ids.options().dtype(torch::kInt32).memory_format(c10::MemoryFormat::Contiguous));
    if (num_local_experts == 0 || topk_ids.numel() == 0) {
        return output;
    }

    auto flat_ids = topk_ids.reshape({-1}).to(
            torch::TensorOptions().device(topk_ids.device()).dtype(torch::kLong));
    // valid_mask: [N_topk]
    auto valid_mask = flat_ids.ge(0);
    if (!valid_mask.any().item<bool>()) {
        return output;
    }

    auto valid_ids = flat_ids.masked_select(valid_mask);
    // 先过滤无效 expert id，再按需把全局 id 映射到当前 rank 的本地 id。
    if (expert_map.has_value()) {
        TORCH_CHECK(expert_map.value().is_cuda(),
                    "count_expert_num_tokens_precompiled expects CUDA expert_map");
        // expert_map_long: [N_global_expert]
        auto expert_map_long = expert_map.value().to(
                torch::TensorOptions().device(topk_ids.device()).dtype(torch::kLong));
        valid_ids = expert_map_long.index_select(0, valid_ids);
    }

    auto in_range_mask = valid_ids.ge(0) & valid_ids.lt(num_local_experts);
    if (!in_range_mask.any().item<bool>()) {
        return output;
    }

    auto local_ids = valid_ids.masked_select(in_range_mask);
    // bincount 的结果天然就是“每个 expert 收到多少 token”的计数表。
    auto counts = torch::bincount(local_ids, /*weights=*/{}, num_local_experts);
    return counts.to(output.options());
}

// 把“零专家”分支转换成恒等映射输出，同时清理 expert_indices / expert_scales 中的占位值。
// 返回值是零专家分支对 hidden_states 的直接贡献，方便后续和真正 expert 输出相加。
torch::Tensor zero_experts_compute_identity_precompiled(
        torch::Tensor &expert_indices, torch::Tensor &expert_scales,
        int64_t num_experts, const torch::Tensor &hidden_states) {
    TORCH_CHECK(
            expert_indices.is_cuda(),
            "zero_experts_compute_identity_precompiled expects CUDA expert_indices");
    TORCH_CHECK(
            expert_scales.is_cuda(),
            "zero_experts_compute_identity_precompiled expects CUDA expert_scales");
    TORCH_CHECK(
            hidden_states.is_cuda(),
            "zero_experts_compute_identity_precompiled expects CUDA hidden_states");
    TORCH_CHECK(
            num_experts >= 0,
            "zero_experts_compute_identity_precompiled expects non-negative "
            "num_experts");
    TORCH_CHECK(
            expert_indices.sizes() == expert_scales.sizes(),
            "zero_experts_compute_identity_precompiled expects expert_indices and "
            "expert_scales to have matching shapes");
    TORCH_CHECK(
            expert_indices.size(0) == hidden_states.size(0),
            "zero_experts_compute_identity_precompiled expects expert tensors and "
            "hidden_states to agree on the token dimension");

    // zero_expert_mask/zero_expert_scales: [N_token, topk]
    auto zero_expert_mask = expert_indices.ge(num_experts);
    auto zero_expert_scales =
            expert_scales.masked_fill(zero_expert_mask.logical_not(), 0.0);

    // 零专家分支会被重定向成恒等映射，因此索引和缩放都要改写成安全占位值。
    expert_indices.masked_fill_(zero_expert_mask, 0);
    expert_scales.masked_fill_(zero_expert_mask, 0.0);

    // 返回的贡献值等价于把所有零专家权重累加后，直接乘到原 hidden_states 上。
    auto scale_sum =
            zero_expert_scales.sum(-1, /*keepdim=*/true).to(hidden_states.scalar_type());
    return hidden_states * scale_sum;
}

// 对输入后一半做乘法门控，并在乘前把两路值裁到 limit 范围内。
void swiglustep_and_mul(torch::Tensor &out, torch::Tensor &input,
                        double limit) {
    TORCH_CHECK(input.is_cuda(),
                "swiglustep_and_mul expects CUDA input");
    TORCH_CHECK(out.is_cuda(),
                "swiglustep_and_mul expects CUDA output");
    TORCH_CHECK(input.size(-1) % 2 == 0,
                "swiglustep_and_mul expects the last dimension to be even");

    // 输入前半段是 gate，后半段是 up 投影，两者在做完裁剪后再逐元素相乘。
    const auto hidden_size = input.size(-1) / 2;
    // gate/up: [..., hidden/2]
    auto gate = input.slice(-1, 0, hidden_size);
    auto up = input.slice(-1, hidden_size);
    // gate_silu/result: [..., hidden/2]
    auto gate_silu = gate * torch::sigmoid(gate);
    auto result = torch::clamp_max(gate_silu, limit) *
                  torch::clamp(up, -limit, limit);
    out.copy_(result.to(out.scalar_type()));
}

// 把指定 block_id 覆盖到一组 KV 张量的对应区间上，常用于回收或重置 cache block。
void zero_kv_blocks_precompiled(const torch::Tensor &block_ids,
                                c10::List<torch::Tensor> kv_tensors,
                                c10::List<int64_t> block_dims,
                                c10::List<int64_t> ratios) {
    TORCH_CHECK(block_ids.is_cuda(),
                "zero_kv_blocks_precompiled expects CUDA block_ids");
    TORCH_CHECK(kv_tensors.size() == block_dims.size() &&
                kv_tensors.size() == ratios.size(),
                "kv_tensors, block_dims, and ratios must have the same length");

    // block_ids_cpu: [N_block]
    auto block_ids_cpu = block_ids.to(torch::kCPU);
    for (size_t tensor_idx = 0; tensor_idx < kv_tensors.size(); ++tensor_idx) {
        auto kv = kv_tensors.get(tensor_idx);
        TORCH_CHECK(kv.is_cuda(),
                    "zero_kv_blocks_precompiled expects CUDA kv tensors");
        const int64_t block_dim = block_dims.get(tensor_idx);
        const int64_t ratio = ratios.get(tensor_idx);
        for (int64_t idx = 0; idx < block_ids_cpu.numel(); ++idx) {
            const int64_t block_id = block_ids_cpu[idx].item<int64_t>();
            // ratio 表示一个逻辑 block 在当前张量维度上实际占用的连续长度。
            kv.narrow(block_dim, block_id * ratio, ratio).zero_();
        }
    }
}

// 把未量化 expert 权重按 slot_ids 搬进运行时槽位。
void moe_batch_load_unquantized_runtime_precompiled(
        const torch::Tensor &slot_ids, const torch::Tensor &w13_src,
        const torch::Tensor &w2_src, torch::Tensor &w13_dst,
        torch::Tensor &w2_dst) {
    // 两组投影权重共用同一套 slot_ids，对应的运行时槽位必须同步更新。
    batch_copy_into_slots(slot_ids, w13_src, w13_dst, "w13_weight");
    batch_copy_into_slots(slot_ids, w2_src, w2_dst, "w2_weight");
}

// 把 GPTQ 量化 expert 的各路权重与元数据批量搬进运行时槽位。
// 可选的 g_idx 及其排序索引也会在这里同步更新，避免 Python 层拆成多次调用。
void moe_batch_load_gptq_runtime_precompiled(
        const torch::Tensor &slot_ids, const torch::Tensor &w13_qweight_src,
        const torch::Tensor &w2_qweight_src, const torch::Tensor &w13_scales_src,
        const torch::Tensor &w2_scales_src, const torch::Tensor &w13_qzeros_src,
        const torch::Tensor &w2_qzeros_src, torch::Tensor &w13_qweight_dst,
        torch::Tensor &w2_qweight_dst, torch::Tensor &w13_scales_dst,
        torch::Tensor &w2_scales_dst, torch::Tensor &w13_qzeros_dst,
        torch::Tensor &w2_qzeros_dst,
        const std::optional<torch::Tensor> &w13_g_idx_src,
        const std::optional<torch::Tensor> &w2_g_idx_src,
        const std::optional<torch::Tensor> &w13_g_idx_sort_indices_src,
        const std::optional<torch::Tensor> &w2_g_idx_sort_indices_src,
        const std::optional<torch::Tensor> &w13_g_idx_dst,
        const std::optional<torch::Tensor> &w2_g_idx_dst,
        const std::optional<torch::Tensor> &w13_g_idx_sort_indices_dst,
        const std::optional<torch::Tensor> &w2_g_idx_sort_indices_dst) {
    // 先搬运量化主权重与缩放参数，确保后续 expert 命中槽位时即可直接解码使用。
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
    // g_idx 相关元数据是可选的，只有量化布局需要时才跟着一起同步。
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

// 逐 expert 执行 batched matmul，必要时先把 fp8 / 量化输入反量化成 float32。
// 这条路径主要服务于 Windows 上缺少高性能 MoE kernel 时的功能兜底。
void moe_batched_mm_precompiled(
        const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C,
        const torch::Tensor &expert_num_tokens,
        const std::optional<torch::Tensor> &A_scale,
        const std::optional<torch::Tensor> &B_scale, bool use_fp8_w8a8,
        bool per_act_token_quant) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(),
                "moe_batched_mm_precompiled expects CUDA tensors");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3 && C.dim() == 3,
                "moe_batched_mm_precompiled expects rank-3 A/B/C tensors");
    TORCH_CHECK(expert_num_tokens.dim() == 1,
                "moe_batched_mm_precompiled expects 1D expert_num_tokens");
    TORCH_CHECK(A.size(0) == B.size(0) && A.size(0) == C.size(0) &&
                A.size(0) == expert_num_tokens.numel(),
                "moe_batched_mm_precompiled expects expert dimension to match");
    TORCH_CHECK(A.size(1) == C.size(1),
                "moe_batched_mm_precompiled expects A/C token dimensions to "
                "match");
    TORCH_CHECK(B.size(1) == C.size(2),
                "moe_batched_mm_precompiled expects B/C output dimensions to "
                "match");
    TORCH_CHECK(A.size(2) == B.size(2),
                "moe_batched_mm_precompiled expects A/B K dimensions to match");

    auto expert_num_tokens_cpu =
            expert_num_tokens.to(torch::kCPU, torch::kLong, false, false)
                    .contiguous();

    // resolved_A_scale/resolved_B_scale: 把可选缩放张量对齐到对应设备
    auto resolved_A_scale =
            A_scale.has_value()
            ? std::optional<torch::Tensor>(
                    A_scale.value().to(A.device(), A_scale.value().scalar_type(),
                                       false, false)
                            .contiguous())
            : std::nullopt;
    auto resolved_B_scale =
            B_scale.has_value()
            ? std::optional<torch::Tensor>(
                    B_scale.value().to(B.device(), B_scale.value().scalar_type(),
                                       false, false)
                            .contiguous())
            : std::nullopt;

    // C: [N_expert, T_max, N_out]，先清零保证未命中区域保持空输出。
    C.zero_();

    for (int64_t expert_idx = 0; expert_idx < expert_num_tokens_cpu.numel();
         ++expert_idx) {
        const int64_t num_tokens = expert_num_tokens_cpu[expert_idx].item<int64_t>();
        if (num_tokens <= 0) {
            continue;
        }

        // 每个 expert 只计算自己实际接收的前 num_tokens 行，未使用尾部保持零值。
        // input_tensor/weight_tensor: [T_used, K] / [N_out, K]
        auto input_tensor = A[expert_idx].slice(0, 0, num_tokens);
        auto weight_tensor = B[expert_idx];

        torch::Tensor input_f;
        torch::Tensor weight_f;
        if (use_fp8_w8a8) {
            input_f = dequantize_batched_moe_tensor_reference(
                    input_tensor,
                    resolved_A_scale.has_value()
                    ? std::optional<torch::Tensor>(
                            resolved_A_scale.value()[expert_idx].slice(0, 0, num_tokens))
                    : std::nullopt,
                    per_act_token_quant);
            weight_f = dequantize_batched_moe_tensor_reference(
                    weight_tensor,
                    resolved_B_scale.has_value()
                    ? std::optional<torch::Tensor>(resolved_B_scale.value()[expert_idx])
                    : std::nullopt,
                    false);
        } else {
            input_f = input_tensor.to(torch::kFloat32);
            weight_f = weight_tensor.to(torch::kFloat32);
        }

        // result: [T_used, N_out]
        auto result =
                torch::matmul(input_f, weight_f.transpose(0, 1)).to(C.scalar_type());
        C[expert_idx].slice(0, 0, num_tokens).copy_(result);
    }
}

// 针对 decode / varlen / 接受 token 回放等多种场景，统一兜底 causal_conv1d_update。
// fast path 直接批量卷积，慢路径则显式处理 cache line、accepted token 和 query 切片。
torch::Tensor causal_conv1d_update_precompiled(
        const torch::Tensor &x, torch::Tensor &conv_state,
        const torch::Tensor &weight, const std::optional<torch::Tensor> &bias,
        const std::string &activation,
        const std::optional<torch::Tensor> &conv_state_indices,
        const std::optional<torch::Tensor> &num_accepted_tokens,
        const std::optional<torch::Tensor> &query_start_loc, int64_t pad_slot_id,
        const std::optional<torch::Tensor> &block_idx_last_scheduled_token,
        const std::optional<torch::Tensor> &initial_state_idx) {
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

    // 简单批量 decode 时直接走向量化 fast path，减少不必要的逐请求 CPU 参与。
    if (use_fast_path) {
        TORCH_CHECK(x.dim() == 3,
                    "fast-path causal_conv1d_update_precompiled expects [B, D, T]");
        // batch/dim/seqlen/history_len: fast path 输入与卷积窗口元数据
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
        // safe_indices/gather_indices: [B]
        auto valid_mask = safe_indices >= 0;
        auto gather_indices = safe_indices.clamp_min(0);
        // state/history: [B, D, T_hist_all] / [B, D, K-1]
        auto state = conv_state.index_select(0, gather_indices);
        auto history = state.slice(-1, 0, history_len);
        auto combined = torch::cat({history, x}, -1);
        // combined/windows: [B, D, T_hist + T] / [B, D, T, K]
        auto windows = combined.unfold(-1, weight.size(1), 1);
        auto out =
                (windows * weight.view({1, dim, 1, weight.size(1)})).sum(-1);
        if (bias.has_value()) {
            out = out + bias.value().view({1, dim, 1});
        }
        // 无效 cache 行对应的输出整行清零。
        out = apply_optional_activation(out, activation);
        out = out * valid_mask.view({batch, 1, 1}).to(out.scalar_type());

        // 先展平到 [B*D, T_hist] 再复用单序列状态更新参考实现。
        auto updated_state =
                update_conv_state_ref(state.reshape({batch * dim, state.size(-1)}),
                                      x.reshape({batch * dim, seqlen}), seqlen)
                        .reshape({batch, dim, state.size(-1)});
        // updated_state: [B, D, T_hist]
        conv_state.index_copy_(0, gather_indices, updated_state);
        return out;
    }

    // 复杂场景改走慢路径，逐条请求决定输入 cache line、输出 cache line 与历史偏移。
    auto out = x.clone();
    // history_len/batch: slow path 卷积历史窗口长度与请求数
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
            // 简单 decode: seq_tokens 直接是 [D, T_seq]
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
            // seq_tokens: [D, T_seq]
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
        // input_cache_line/output_cache_line: 当前读取/写回的 cache 槽位
        const int64_t output_cache_line =
                resolve_cache_line(conv_state_indices, seq_idx, output_block_offset);

        const auto state = conv_state[input_cache_line];
        int64_t history_offset = 0;
        int64_t shift_tokens = seq_len;
        if (num_accepted_tokens.has_value()) {
            history_offset =
                    std::max<int64_t>(num_accepted_tokens.value()[seq_idx].item<int64_t>() - 1,
                                      0);
            // accepted-token 回放时，这一轮只把最后一个真正接受的 token 推进缓存。
            shift_tokens = 1;
        }

        auto history =
                load_initial_history(state, history_len, true, history_offset);
        auto seq_output =
                causal_conv1d_sequence_ref(seq_tokens, weight, bias, history, activation);
        // seq_output: [D, T_seq]

        if (!query_start_loc.has_value()) {
            out[seq_idx].slice(-1, 0, seq_len).copy_(seq_output);
        } else {
            out.slice(0, seq_start, seq_end).copy_(seq_output.transpose(0, 1));
        }

        if (output_cache_line != pad_slot_id) {
            // accepted token 回放场景只推进真正进入缓存的那一部分 token。
            conv_state[output_cache_line].copy_(update_conv_state_ref(
                    conv_state[input_cache_line], seq_tokens, shift_tokens));
        }
    }

    return out;
}
