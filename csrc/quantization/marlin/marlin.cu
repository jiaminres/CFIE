/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * 改编自 https://github.com/IST-DASLab/marlin
 */

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME marlin
#endif

#include "kernel.h"
#include "core/registration.h"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

namespace marlin {
    // 低架构占位 kernel 只用于让符号存在，真正可运行 kernel 由 selector 生成。
    __global__ void MarlinDefault(MARLIN_KERNEL_PARAMS) {
    };

    // Marlin kernel selector 返回的统一函数指针类型。
    using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750

    // SM75 以下不支持 Marlin 主 kernel，这里保留空实现避免编译期缺符号。
    __global__ void permute_cols_kernel(int4 const * __restrict__ a_int4_ptr,
                                        int const * __restrict__ perm_int_ptr,
                                        int4 * __restrict__ out_int4_ptr, int size_m,
                                        int size_k, int lda, int block_rows) {
    }

}  // 命名空间 marlin

    // SM75 以下直接拒绝执行，避免调用方误走不支持的 Marlin 路径。
    // PyTorch 入口：校验输入 Tensor，推导 Marlin 类型，并调用底层 CUDA kernel。
    torch::Tensor marlin_gemm(
        torch::Tensor &a, std::optional<torch::Tensor> c_or_none,
        torch::Tensor &b_q_weight,
        std::optional<torch::Tensor> const &b_bias_or_none, torch::Tensor &b_scales,
        std::optional<torch::Tensor> const &b_zeros_or_none,
        std::optional<torch::Tensor> const &g_idx_or_none,
        std::optional<torch::Tensor> const &perm_or_none, torch::Tensor &workspace,
        vllm::ScalarTypeId const &b_type_id, int64_t size_m, int64_t size_n,
        int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
        bool is_zp_float) {
        // 当前编译目标不满足 Marlin 最低架构要求，运行时直接报错。
        TORCH_CHECK_NOT_IMPLEMENTED(false,
                                    "marlin_gemm(..) requires CUDA_ARCH >= 8.0");
        // 保留不可达返回值，满足函数签名与编译器控制流分析。
        return torch::empty({1, 1});
    }

#else

    // a: [M, K]，根据 perm: [K] 把 A 的 K 维列重排到 out。
    __global__ void permute_cols_kernel(int4 const * __restrict__ a_int4_ptr,
                                        int const * __restrict__ perm_int_ptr,
                                        int4 * __restrict__ out_int4_ptr, int size_m,
                                        int size_k, int lda, int block_rows) {
        // 当前 block 负责的起始 M 行。
        auto start_row = block_rows * blockIdx.x;
        // 当前 block 理论结束行，用于分块覆盖整个 M 维。
        int finish_row = start_row + block_rows;
        // 尾块可能越过真实 M，需要裁剪到 size_m。
        if (finish_row > size_m) {
            finish_row = size_m;
        }
        // 当前 block 实际需要处理的行数。
        int cur_block_rows = finish_row - start_row;

        // A 按 int4 访问，输入行跨度需要从 half 元素数换算成 16B 单位。
        int input_row_stride = lda * sizeof(half) / 16;
        // out 是重排后的紧凑 [M, K]，输出行跨度同样按 16B 单位计算。
        int output_row_stride = size_k * sizeof(half) / 16;

        // 对单行执行 K 维列重排，每个线程负责若干 K 位置。
        auto permute_row = [&](int row) {
            // 每轮处理 default_threads 个 K 元素。
            int iters = size_k / default_threads;
            // rest 保存最后不足一个线程块宽度的 K 元素数。
            int rest = size_k % default_threads;

            // 输入行首地址偏移，单位为 int4。
            int input_offset = row * input_row_stride;
            // 输出行首地址偏移，单位为 int4。
            int output_offset = row * output_row_stride;

            // 把 int4 指针转成 half 指针，方便按 K 维逐元素重排。
            half const *a_row_half =
                    reinterpret_cast<half const *>(a_int4_ptr + input_offset);
            // 输出同样按 half 逐列写入。
            half *out_half = reinterpret_cast<half *>(out_int4_ptr + output_offset);

            // base_k 记录当前循环块在 K 维的起点。
            int base_k = 0;

            // 主循环覆盖完整的 default_threads 宽度块。
            for (int i = 0; i < iters; i++) {
                // 当前线程负责的目标 K 位置。
                auto cur_k = base_k + threadIdx.x;
                // perm[cur_k] 给出目标位置对应的源 K 位置。
                int src_pos = perm_int_ptr[cur_k];

                // 把源列值写入重排后的目标列。
                out_half[cur_k] = a_row_half[src_pos];

                // 推进到下一个 K 维线程块。
                base_k += default_threads;
            }

            // 处理最后不足 default_threads 个元素的 K 维尾块。
            if (rest) {
                // 只有前 rest 个线程参与尾块写入，避免越界访问。
                if (threadIdx.x < rest) {
                    // 当前线程负责尾块中的目标 K 位置。
                    auto cur_k = base_k + threadIdx.x;
                    // 读取该目标位置对应的源 K 位置。
                    int src_pos = perm_int_ptr[cur_k];

                    // 写入尾块重排结果。
                    out_half[cur_k] = a_row_half[src_pos];
                }
            }
        };

        // 当前 block 内按行遍历，逐行完成列重排。
        for (int i = 0; i < cur_block_rows; i++) {
            // 计算当前局部行对应的全局 M 行。
            int cur_row = start_row + i;
            // 再次保护尾块边界，确保不访问 size_m 之外的行。
            if (cur_row < size_m) {
                // 对当前行执行 K 维 perm 重排。
                permute_row(cur_row);
            }
        }
    }

    // 单个候选 kernel 的线程 tile 配置。
    typedef struct {
        // thread_k 表示单个线程块覆盖的 K 维 tile 大小。
        int thread_k;
        // thread_n 表示单个线程块覆盖的 N 维 tile 大小。
        int thread_n;
        // num_threads 表示 kernel launch 的线程数。
        int num_threads;
    } thread_config_t;

    // 小 batch 优先使用更大的 K/N tile，减少调度开销。
    thread_config_t small_batch_thread_configs[] = {
        // 配置按优先级排列：thread_k, thread_n, num_threads。
        {128, 128, 256},
        {64, 128, 128},
        {128, 64, 128}
    };

    // 大 batch 优先扩大 N tile，提升并行覆盖。
    thread_config_t large_batch_thread_configs[] = {
        // 配置按优先级排列：thread_k, thread_n, num_threads。
        {64, 256, 256},
        {64, 128, 128},
        {128, 64, 128}
    };

    // 执行配置同时描述每个 SM 的 block 数与线程 tile。
    typedef struct {
        // 每个 SM 同时驻留的 block 数。
        int blocks_per_sm;
        // 当前执行配置选择的线程 tile。
        thread_config_t tb_cfg;
    } exec_config_t;

    // 估算 scale 在共享内存中的缓存大小，单位为字节。
    int get_scales_cache_size(thread_config_t const &th_config, int prob_m,
                              int prob_n, int prob_k, int num_bits, int group_size,
                              bool has_act_order, bool is_k_full, int stages) {
        // act-order 且不是完整 K 时，scale 需要按 K chunk 动态缓存。
        bool cache_scales_chunk = has_act_order && !is_k_full;

        // 当前线程块覆盖的 N 维 tile 大小。
        int tb_n = th_config.thread_n;
        // 当前线程块覆盖的 K 维 tile 大小。
        int tb_k = th_config.thread_k;

        // tb_groups 表示一个 K tile 内最多涉及多少个量化 group。
        int tb_groups;
        // group_size=-1 表示 per-channel 或单组 scale。
        if (group_size == -1) {
            tb_groups = 1;
            // group_size=0 表示 act-order 下局部 K 分片无法静态确定 group，按 32 的最坏情况估算。
        } else if (group_size == 0) {
            tb_groups = div_ceil(tb_k, 32);
            // 普通 grouped quant 按真实 group_size 计算当前 K tile 触达的 group 数。
        } else {
            tb_groups = div_ceil(tb_k, group_size);
        }

        // 动态缓存路径需要覆盖 pipeline 前后 chunk 的 scale。
        if (cache_scales_chunk) {
            // load_groups 按 K 维 pipeline 深度扩大，保证预取 chunk 足够。
            int load_groups =
                    tb_groups * stages * 2;
            // scale 加载至少按 32 个 group 组织，匹配 kernel 内部批量加载粒度。
            load_groups = max(load_groups, 32);
            // 每个 scale 以 2 字节计，返回当前 N tile 下的 scale 缓存字节数。
            return load_groups * tb_n * 2;
        } else {
            // 静态缓存路径只需要缓存当前线程块涉及的 group 与 N tile。
            int tb_scales = tb_groups * tb_n * 2;

            // pipeline 每个 stage 都需要一份 scale 缓存。
            return tb_scales * stages;
        }
    }

    // 估算一个 Marlin kernel 实例需要的动态共享内存大小。
    int get_kernel_cache_size(thread_config_t const &th_config, int thread_m_blocks,
                              int prob_m, int prob_n, int prob_k, int num_bits,
                              int group_size, bool has_act_order, bool is_k_full,
                              int has_zp, bool is_zp_float, bool is_a_8bit,
                              int stages) {
        // pack_factor 表示一个 32 bit 权重 word 中能打包多少个量化值。
        int pack_factor = 32 / num_bits;

        // tb_k 是当前线程块覆盖的 K 维 tile。
        int tb_k = th_config.thread_k;
        // tb_n 是当前线程块覆盖的 N 维 tile。
        int tb_n = th_config.thread_n;
        // tb_m 由 thread_m_blocks 决定，每个 m block 固定 16 行。
        int tb_m = thread_m_blocks * 16;
        // A cache 大小随 activation 位宽变化：8bit A 用 1 字节，16bit A 用 2 字节。
        int sh_a_size = stages * (tb_m * tb_k) * (is_a_8bit ? 1 : 2);
        // B cache 保存打包权重，按 32 bit word 数换算成字节。
        int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
        // reduce cache 保存局部累加结果，N 维额外 +8 用于 kernel 内部对齐。
        int sh_red_size = tb_m * (tb_n + 8) * 2;
        // bias cache 按 N tile 保存半精度 bias。
        int sh_bias_size = tb_n * 2;
        // tmp cache 需要同时容纳 B cache 或 reduce cache，并叠加 bias cache。
        int tmp_size =
                (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
        // tmp cache 取三者最大值，保证复用缓冲区不会覆盖最大使用阶段。
        tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

        // scale cache 大小由 group 布局、act-order 和 pipeline stage 决定。
        int sh_s_size =
                get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits,
                                      group_size, has_act_order, is_k_full, stages);
        // g_idx 只在 act-order 且非完整 K 路径下需要缓存。
        int sh_g_idx_size = has_act_order && !is_k_full ? stages * tb_k / 4 : 0;
        // zero-point cache 默认没有，只有 has_zp 时按量化格式补充。
        int sh_zp_size = 0;
        // has_zp 表示权重使用非对称量化，需要额外加载 zero-point。
        if (has_zp) {
            // float zero-point 与 scale 形状一致，因此缓存大小等于 scale cache。
            if (is_zp_float)
                sh_zp_size = sh_s_size;
                // int4 zero-point 每个字节打包 2 个值，相对 scale 缓存缩小 4 倍。
            else if (num_bits == 4)
                sh_zp_size = sh_s_size / 4;
                // int8 zero-point 相对 scale 缓存缩小 2 倍。
            else if (num_bits == 8)
                sh_zp_size = sh_s_size / 2;
        }

        // 汇总 kernel 所需动态共享内存。
        int total_size =
                tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size;

        // 返回共享内存字节数，供配置过滤和 cudaFuncSetAttribute 使用。
        return total_size;
    }

    // 判断某个线程 tile 配置是否能合法运行当前 Marlin GEMM。
    bool is_valid_config(thread_config_t const &th_config, int thread_m_blocks,
                         int prob_m, int prob_n, int prob_k, int num_bits,
                         int group_size, bool has_act_order, bool is_k_full,
                         int has_zp, bool is_zp_float, bool is_a_8bit, int stages,
                         int max_shared_mem) {
        // -1 表示未初始化配置，不能进入 kernel selector。
        if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
            th_config.num_threads == -1) {
            return false;
        }

        // K/N 必须能被线程 tile 整除，否则 kernel 内部 tile 索引会越界。
        if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
            return false;
        }

        // Marlin kernel 对 thread_n/thread_k 有最小 tile 约束。
        if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
            return false;
        }

        // 至少 128 线程即 4 个 warp，满足 Marlin kernel 的并行加载/计算结构。
        if (th_config.num_threads < 128) {
            return false;
        }

        // 先估算该配置需要的动态共享内存。
        int cache_size = get_kernel_cache_size(
            th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits, group_size,
            has_act_order, is_k_full, has_zp, is_zp_float, is_a_8bit, stages);
        // 只有共享内存需求不超过设备限制时，该配置才可用。
        return cache_size <= max_shared_mem;
    }

    // 根据类型、tile 和量化特征从自动生成的 selector 中取出具体 Marlin kernel。
    MarlinFuncPtr get_marlin_kernel(
        const vllm::ScalarType a_type, const vllm::ScalarType b_type,
        const vllm::ScalarType c_type, const vllm::ScalarType s_type,
        int thread_m_blocks, int thread_n_blocks, int thread_k_blocks,
        bool m_block_size_8, bool has_act_order, bool has_zp, int group_blocks,
        int threads, bool is_zp_float, int stages) {
        // num_bits 用于 selector 区分 int4/int8/fp8/fp4 等权重位宽。
        int num_bits = b_type.size_bits();
        // 默认指向空 kernel，selector 未命中时用于显式报错。
        auto kernel = MarlinDefault;

        // kernel_selector.h 根据上面的局部变量展开匹配逻辑并更新 kernel。
#include "kernel_selector.h"

        // 返回匹配到的 kernel；未匹配时仍为 MarlinDefault。
        return kernel;
    }

    // 自动选择当前问题规模可用的 Marlin 执行配置。
    exec_config_t determine_exec_config(
        const vllm::ScalarType &a_type, const vllm::ScalarType &b_type,
        const vllm::ScalarType &c_type, const vllm::ScalarType &s_type, int prob_m,
        int prob_n, int prob_k, int thread_m_blocks, bool m_block_size_8,
        int num_bits, int group_size, bool has_act_order, bool is_k_full,
        bool has_zp, bool is_zp_float, int is_a_8bit, int stages,
        int max_shared_mem, int sms) {
        // 默认配置保持无效状态，调用方据此判断自动选择是否失败。
        exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
        // M tile 较大时使用 large batch 配置，否则使用 small batch 配置。
        thread_config_t *thread_configs = thread_m_blocks > 1
                                              ? large_batch_thread_configs
                                              : small_batch_thread_configs;
        // 计算当前候选配置数组长度。
        int thread_configs_size =
                thread_m_blocks > 1
                    ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
                    : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

        // 按优先级遍历候选线程 tile。
        for (int i = 0; i < thread_configs_size; i++) {
            // 取出当前候选配置。
            thread_config_t th_config = thread_configs[i];

            // 过滤共享内存、tile 整除和最小线程数不满足的配置。
            if (!is_valid_config(th_config, thread_m_blocks, prob_m, prob_n, prob_k,
                                 num_bits, group_size, has_act_order, is_k_full, has_zp,
                                 is_zp_float, is_a_8bit, stages,
                                 max_shared_mem - 512)) {
                continue;
            }

            // 计算该候选配置实际需要的共享内存，用于后续调试和配置验证。
            int cache_size = get_kernel_cache_size(th_config, thread_m_blocks, prob_m,
                                                   prob_n, prob_k, num_bits, group_size,
                                                   has_act_order, is_k_full, has_zp,
                                                   is_zp_float, is_a_8bit, stages);

            // 非 act-order 路径可以把 group_size 转成 kernel selector 使用的 16 元素 group block。
            int group_blocks = 0;
            // act-order 路径由 kernel 内部 g_idx 处理 group，不在 selector 中静态绑定 group_blocks。
            if (!has_act_order) {
                group_blocks = group_size == -1 ? -1 : group_size / 16;
            }

            // 尝试用当前候选配置匹配一个真实 Marlin kernel。
            auto kernel =
                    get_marlin_kernel(a_type, b_type, c_type, s_type, thread_m_blocks,
                                      th_config.thread_n / 16, th_config.thread_k / 16,
                                      m_block_size_8, has_act_order, has_zp, group_blocks,
                                      th_config.num_threads, is_zp_float, stages);

            // selector 未命中时继续尝试下一组候选配置。
            if (kernel == MarlinDefault) continue;

            // 命中真实 kernel 后返回当前配置。
            return {1, th_config};
        }

        // 所有候选都失败时返回无效配置。
        return exec_cfg;
    }

    // 执行底层 Marlin GEMM kernel，负责配置选择、act-order 预处理和分段 launch。
    void marlin_mm(const void *A,
                   const void *B,
                   void *C,
                   void *C_tmp,
                   void *b_bias,
                   void *a_s,
                   void *b_s,
                   void *g_s,
                   void *zp,
                   void *g_idx,
                   void *perm,
                   void *a_tmp,
                   int prob_m,
                   int prob_n,
                   int prob_k,
                   int lda,
                   void *workspace,
                   vllm::ScalarType const &a_type,
                   vllm::ScalarType const &b_type,
                   vllm::ScalarType const &c_type,
                   vllm::ScalarType const &s_type,
                   bool has_bias,
                   bool has_act_order,
                   bool is_k_full,
                   bool has_zp,
                   int num_groups,
                   int group_size,
                   int dev,
                   cudaStream_t stream,
                   int thread_k_init,
                   int thread_n_init,
                   int sms,
                   bool use_atomic_add,
                   bool use_fp32_reduce,
                   bool is_zp_float) {
        // A 为 8bit dtype 时，kernel 内部按 1 字节 activation 加载。
        bool is_a_8bit = a_type.size_bits() == 8;

        // GEMM 三个维度必须为正，否则 kernel tile 计算没有合法范围。
        TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m,
                    ", ", prob_n, ", ", prob_k, "]");

        // group_blocks 是以 16 个 K 元素为单位的量化 group 大小。
        int group_blocks = 0;

        // act-order 路径需要区分完整 K 与 TP 分片 K。
        if (has_act_order) {
            // 完整 K 时每个 rank 拥有完整 group，可退化成静态 group_blocks。
            if (is_k_full) {
                // 完整 K 的 act-order 不能使用 per-channel 单组标记。
                TORCH_CHECK(group_size != -1);
                // kernel selector 使用 16 元素为单位的 group block。
                group_blocks = group_size / 16;
                // K 维必须能按 group block 整除，保证 kernel 内部 group 索引合法。
                TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                            " is not divisible by group_blocks = ", group_blocks);
                // 非完整 K 的 act-order 由 g_idx 描述本地 K 到 group 的映射。
            } else {
                // group_size=0 表示不能静态推导本地 group 大小。
                TORCH_CHECK(group_size == 0);
                // selector 使用 0 标记动态 group 映射。
                group_blocks = 0;
            }
            // 非 act-order 路径中 group 布局可以直接由 group_size 静态确定。
        } else {
            // group_size=-1 表示整条 K 维共享一组 scale。
            if (group_size == -1) {
                group_blocks = -1;
                // grouped quant 路径按 16 元素单位换算 group block。
            } else {
                group_blocks = group_size / 16;
                // K 维必须能按 group block 整除，保证 scale/zero-point 索引合法。
                TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                            " is not divisible by group_blocks = ", group_blocks);
            }
        }

        // num_bits 表示 B 权重量化位宽，用于共享内存估算与 selector 匹配。
        int num_bits = b_type.size_bits();
        // A/B/C 都按 int4 进行 16B 向量化访问。
        const int4 *A_ptr = (const int4 *) A;
        const int4 *B_ptr = (const int4 *) B;
        int4 *C_ptr = (int4 *) C;
        int4 *C_tmp_ptr = (int4 *) C_tmp;

        // bias 按 int4 加载，内部再解释为对应 c_type。
        const int4 *bias_ptr = (const int4 *) b_bias;
        // activation scale 始终以 float 指针传给 kernel。
        const float *a_s_ptr = (const float *) a_s;
        // weight scale 按 int4 加载，实际解释由 s_type 决定。
        const int4 *b_s_ptr = (const int4 *) b_s;
        // global scale 用于 NVFP4/MXFP4 等额外全局缩放路径。
        const uint16_t *g_s_ptr = (const uint16_t *) g_s;

        // zero-point 按 int4 加载，实际布局由 has_zp/is_zp_float 决定。
        const int4 *zp_ptr = (const int4 *) zp;
        // g_idx 描述 act-order 路径下本地 K 位置对应的 group id。
        const int *g_idx_ptr = (const int *) g_idx;
        // perm 描述 A 的 K 维重排顺序。
        const int *perm_ptr = (const int *) perm;
        // a_tmp 是 act-order 重排后的 A 临时缓冲。
        int4 *a_tmp_ptr = (int4 *) a_tmp;
        // workspace 首段作为 kernel 间同步/归约锁使用。
        int *locks = (int *) workspace;

        // act-order 需要先把 A 的 K 维按 perm 重排，使其与重排后的权重行一致。
        if (has_act_order) {
            // 每个 SM 负责一段 M 行，尽量让重排 kernel 覆盖所有 SM。
            int block_rows = div_ceil(prob_m, sms);
            // 避免 clang-format 把 CUDA launch 的 >>> 拆成 > > >。
    // clang-format off
    permute_cols_kernel<<<sms, default_threads, 0, stream>>>(
        A_ptr, perm_ptr, a_tmp_ptr, prob_m, prob_k, lda, block_rows);
            // clang-format on
            // 后续 GEMM 读取重排后的 A。
            A_ptr = a_tmp_ptr;
            // 重排后的 A 是紧凑 [M, K]，行跨度变为 prob_k。
            lda = prob_k;

            // 完整 K 时权重已按 group id 重排，A 也已重排，可切回非 act-order kernel。
            if (is_k_full) has_act_order = false;
        }

        // 查询当前设备单 block 可申请的最大动态共享内存。
        int max_shared_mem = 0;
        cudaDeviceGetAttribute(&max_shared_mem,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        // 设备必须返回有效共享内存上限，否则无法验证 kernel 配置。
        TORCH_CHECK(max_shared_mem > 0);

        // 查询 CUDA 计算能力主版本。
        int major_capability, minor_capability;
        cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                               dev);
        // 查询 CUDA 计算能力次版本。
        cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                               dev);
        // Marlin kernel 最低要求 SM75。
        TORCH_CHECK(major_capability * 10 + minor_capability >= 75,
                    "marlin kernel only support Turing or newer GPUs.");
        // 默认使用 4 stage pipeline。
        int stages = 4;
        // Turing SM75 共享内存和指令能力较弱，降低 pipeline stage。
        if (major_capability == 7 && minor_capability == 5) {
            // Turing 上使用 2 stage，避免共享内存和调度压力过大。
            stages = 2;
            // Turing 只允许 FP16 或 INT8 activation 的 Marlin 路径。
            TORCH_CHECK(a_type == vllm::kFloat16 || a_type == vllm::kS8,
                        "Turing only support FP16 or INT8 activation.");
        }
        // W4A8-FP8 Marlin 只在 Ada SM89 和 Blackwell SM120 上启用。
        if (a_type == vllm::kFE4M3fn) {
            TORCH_CHECK(
                major_capability * 10 + minor_capability == 89 ||
                major_capability * 10 + minor_capability == 120,
                "Marlin W4A8-FP8 only support SM89 or SM120 device (It is slower than "
                "Marlin W4A16 on other devices).");
        }

        // max_par 控制 M 维拆分并行度，避免一次 launch 覆盖过多 M 分片。
        int max_par = 16;
        // 小 N 场景可放宽 M 维并行拆分，提高 SM 利用率。
        if (prob_n <= 4096) max_par = 16 * 8;
        // max_shared_mem_new 会随 blocks_per_sm 调整为每个 block 可用的共享内存。
        int max_shared_mem_new = max_shared_mem;
        // rest_m 记录尚未 launch 的 M 行数。
        int rest_m = prob_m;
        // 每个 kernel 分段最多覆盖 4 个 16 行 M block。
        int max_thread_m_blocks = 4;
        // 按 M 维分段 launch，直到所有 M 行处理完毕。
        while (rest_m) {
            // par_count 表示当前剩余 M 能切出多少个最大 M tile。
            int par_count = rest_m / (max_thread_m_blocks * 16);
            // 限制单次 launch 的 M tile 数，防止过大 grid 降低调度效率。
            if (par_count > max_par) par_count = max_par;
            // prob_m_split 是本轮 launch 实际处理的 M 行数。
            int prob_m_split =
                    par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

            // 用户指定 thread_k 时优先使用，否则保持 -1 进入自动配置。
            int thread_k = thread_k_init;
            // 用户指定 thread_n 时优先使用，否则保持 -1 进入自动配置。
            int thread_n = thread_n_init;

            // 当前 M 分段需要的 16 行 block 数，不超过 max_thread_m_blocks。
            int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
            // 小 M 且 16bit activation 可选择 8 行特化 kernel。
            int m_block_size_8 = prob_m_split <= 8 && a_type.size_bits() == 16;

            // ------------------------------- 选择线程 tile 配置 -------------------------------
            // exec_cfg 保存最终 blocks_per_sm 和线程 tile。
            exec_config_t exec_cfg;
            // thread_tfg 保存当前轮 launch 的 thread_k/thread_n/num_threads。
            thread_config_t thread_tfg;
            // 同时指定 thread_k/thread_n 时，跳过自动配置。
            if (thread_k != -1 && thread_n != -1) {
                // 手动配置默认使用 default_threads。
                thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
                // 手动配置每个 SM 只放一个 block。
                exec_cfg = exec_config_t{1, thread_tfg};
                // N 维必须能被手动 thread_n 整除。
                TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
                            " is not divisible by thread_n = ", thread_n);
                // K 维必须能被手动 thread_k 整除。
                TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
                            " is not divisible by thread_k = ", thread_k);
            } else {
                // 自动配置按问题规模和共享内存限制选择可用 kernel。
                exec_cfg = determine_exec_config(
                    a_type, b_type, c_type, s_type, prob_m_split, prob_n, prob_k,
                    thread_m_blocks, m_block_size_8, num_bits, group_size, has_act_order,
                    is_k_full, has_zp, is_zp_float, is_a_8bit, stages, max_shared_mem,
                    sms);
                // 提取自动配置选出的线程 tile。
                thread_tfg = exec_cfg.tb_cfg;
                // 如果自动配置已命中，可在小 grid 场景尝试更小 N tile 提升 SM 覆盖。
                if (thread_tfg.thread_n != -1) {
                    // 估算当前 grid 是否过小；过小时优先尝试 {128,64,128} 配置。
                    if (prob_n / thread_tfg.thread_n *
                        div_ceil(prob_m_split, thread_m_blocks * 16) * 4 <=
                        sms) {
                        // 确认 {128,64,128} 在当前共享内存限制下合法。
                        if (is_valid_config({128, 64, 128}, thread_m_blocks, prob_m_split,
                                            prob_n, prob_k, num_bits, group_size,
                                            has_act_order, is_k_full, has_zp, is_zp_float,
                                            is_a_8bit, stages, max_shared_mem_new)) {
                            // 使用更小 N tile 增加 block 数，提高小 N 场景并行度。
                            thread_tfg = {128, 64, 128};
                            // blocks_per_sm 仍保持 1。
                            exec_cfg = {1, thread_tfg};
                        }
                    }
                }

                // 自动配置失败且 M tile 还能缩小时，缩小 M tile 后重新尝试。
                if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1) {
                    // 减少每个 block 覆盖的 M block，降低共享内存需求。
                    max_thread_m_blocks--;
                    // 回到 while 开头重新计算 prob_m_split 与配置。
                    continue;
                }
            }

            // 最终 launch 线程数来自选中的线程 tile。
            int num_threads = thread_tfg.num_threads;
            // 最终 thread_k 来自手动或自动配置。
            thread_k = thread_tfg.thread_k;
            // 最终 thread_n 来自手动或自动配置。
            thread_n = thread_tfg.thread_n;
            // grid block 数按 SM 数和每 SM block 数确定。
            int blocks = sms * exec_cfg.blocks_per_sm;
            // 多 block 驻留同一 SM 时，每个 block 可用共享内存需要均分并预留余量。
            if (exec_cfg.blocks_per_sm > 1)
                max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

            // selector 使用 16 元素单位描述 K tile。
            int thread_k_blocks = thread_k / 16;
            // selector 使用 16 元素单位描述 N tile。
            int thread_n_blocks = thread_n / 16;

            // launch 前再次校验最终配置，错误信息带出所有关键维度。
            TORCH_CHECK(
                is_valid_config(thread_tfg, thread_m_blocks, prob_m_split, prob_n,
                    prob_k, num_bits, group_size, has_act_order, is_k_full,
                    has_zp, is_zp_float, is_a_8bit, stages,
                    max_shared_mem_new),
                "Invalid thread config: thread_m_blocks = ", thread_m_blocks,
                ", thread_k = ", thread_tfg.thread_k,
                ", thread_n = ", thread_tfg.thread_n,
                ", num_threads = ", thread_tfg.num_threads, " for MKN = [", prob_m,
                ", ", prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
                ", prob_m_split = ", prob_m_split, ", group_size = ", group_size,
                ", has_act_order = ", has_act_order, ", is_k_full = ", is_k_full,
                ", has_zp = ", has_zp, ", is_zp_float = ", is_zp_float,
                ", stages = ", stages, ", max_shared_mem_new = ", max_shared_mem_new);

            // 根据最终配置选出具体的模板实例 kernel。
            auto kernel = get_marlin_kernel(
                a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
                thread_k_blocks, m_block_size_8, has_act_order, has_zp, group_blocks,
                num_threads, is_zp_float, stages);

            // selector 仍未命中时，说明当前形状和量化组合不受支持。
            if (kernel == MarlinDefault) {
                TORCH_CHECK(false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n,
                            ", ", prob_k, "]", ", has_act_order = ", has_act_order,
                            ", num_groups = ", num_groups, ", group_size = ", group_size,
                            ", prob_m_split = ", prob_m_split,
                            ", thread_m_blocks = ", thread_m_blocks,
                            ", thread_n_blocks = ", thread_n_blocks,
                            ", thread_k_blocks = ", thread_k_blocks,
                            ", num_threads = ", num_threads, ", num_bits = ", num_bits);
            }

            // 允许该 kernel 使用 max_shared_mem_new 字节动态共享内存。
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 max_shared_mem_new);

            // 小输出规模下 atomicAdd reduce 可能快于全局 reduce。
            bool part_use_atomic_add =
                    use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

            // 避免 clang-format 把 CUDA launch 的 >>> 拆成 > > >。
    // clang-format off
    kernel<<<blocks, num_threads, max_shared_mem_new, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr,
        g_idx_ptr, num_groups,
        prob_m_split, prob_n, prob_k, lda, locks, has_bias, part_use_atomic_add,
        use_fp32_reduce, max_shared_mem_new);
            // clang-format on

            // 重新确认 A 位宽，用于计算本轮 M 分段之后的 A 指针推进量。
            bool is_a_8bit = a_type.size_bits() == 8;
            // A_ptr 推进 prob_m_split 行；8bit A 每 16 个元素占一个 int4，16bit A 每 8 个元素占一个 int4。
            A_ptr += prob_m_split * (lda / (is_a_8bit ? 16 : 8));
            // activation scale 按 M 行逐 token 存储，同步推进 prob_m_split。
            a_s_ptr += prob_m_split;
            // C_ptr 按输出 [M, N] 推进，int4 一次覆盖 8 个 half/BF16 元素。
            C_ptr += prob_m_split * (prob_n / 8);
            // 扣减本轮已处理的 M 行数。
            rest_m -= prob_m_split;
        }
    }
} // 命名空间 marlin

torch::Tensor marlin_gemm(
    torch::Tensor &a,
    std::optional<torch::Tensor> c_or_none,
    torch::Tensor &b_q_weight,
    std::optional<torch::Tensor> const &b_bias_or_none,
    torch::Tensor &b_scales,
    std::optional<torch::Tensor> const &a_scales_or_none,
    std::optional<torch::Tensor> const &global_scale_or_none,
    std::optional<torch::Tensor> const &b_zeros_or_none,
    std::optional<torch::Tensor> const &g_idx_or_none,
    std::optional<torch::Tensor> const &perm_or_none,
    torch::Tensor &workspace,
    vllm::ScalarTypeId const &b_type_id,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float) {
    // a/c/scale 的 ScalarTypeId 在入口根据 PyTorch dtype 动态推导。
    vllm::ScalarTypeId a_type_id, c_type_id, s_type_id;

    // 默认输出 dtype 跟随 activation A。
    auto c_dtype = a.dtype();

    // FP16 activation 路径输出也使用 FP16。
    if (a.scalar_type() == at::ScalarType::Half) {
        a_type_id = vllm::kFloat16.id();
        c_type_id = vllm::kFloat16.id();
    } else if (a.scalar_type() == at::ScalarType::BFloat16) {
        // BF16 activation 路径输出也使用 BF16。
        a_type_id = vllm::kBFloat16.id();
        c_type_id = vllm::kBFloat16.id();
    } else {
        // 其余 activation 属于 W4A8 / FP4 等路径，需要从 scale 或显式 C 推导输出 dtype。
        // 默认先让输出 dtype 跟随 b_scales。
        c_dtype = b_scales.dtype();

        // b_scales 为 FP16 时，输出计算类型设为 FP16。
        if (b_scales.scalar_type() == at::ScalarType::Half) {
            c_type_id = vllm::kFloat16.id();
        } else if (b_scales.scalar_type() == at::ScalarType::BFloat16) {
            // b_scales 为 BF16 时，输出计算类型设为 BF16。
            c_type_id = vllm::kBFloat16.id();
        } else {
            // FP4 scale 不是最终输出 dtype，需要依赖调用方传入 C。
            // 先给 c_type 一个默认 BF16，随后会用 c 的真实 dtype 覆盖。
            c_type_id = vllm::kBFloat16.id();

            // W4A8-FP4 路径必须传入 C，以明确输出 dtype。
            TORCH_CHECK(c_or_none.has_value(), "c must be passed for W4A8-FP4");

            // 取出调用方提供的输出 Tensor。
            torch::Tensor c = c_or_none.value();

            // 输出 dtype 以显式 C 为准。
            c_dtype = c.dtype();

            // 显式 C 为 FP16 时，输出计算类型设为 FP16。
            if (c.scalar_type() == at::ScalarType::Half) {
                c_type_id = vllm::kFloat16.id();
            } else if (c.scalar_type() == at::ScalarType::BFloat16) {
                // 显式 C 为 BF16 时，输出计算类型设为 BF16。
                c_type_id = vllm::kBFloat16.id();
            } else {
                // 其他输出 dtype 没有对应 Marlin kernel。
                TORCH_CHECK(false, "unsupported c dtype");
            }
        }

        // FP8 activation 映射为 E4M3 输入类型。
        if (a.scalar_type() == at::ScalarType::Float8_e4m3fn) {
            a_type_id = vllm::kFE4M3fn.id();
        } else if (a.scalar_type() == at::ScalarType::Char) {
            // int8 activation 在 PyTorch 侧用 Char 表示。
            a_type_id = vllm::kS8.id();
        } else {
            // 非 16bit 且非 int8/fp8 activation 不受 Marlin 支持。
            TORCH_CHECK(false, "unsupported `a` scalar_type");
        }
    }

    // 默认 scale 类型与输出类型一致。
    s_type_id = c_type_id;
    // FP4 权重路径的 scale 可能使用 FP8 scale 类型，需要单独推导。
    if (b_type_id == vllm::kFE2M1f.id()) {
        // NVFP4 使用 E4M3 scale。
        if (b_scales.scalar_type() == at::ScalarType::Float8_e4m3fn) {
            s_type_id = vllm::kFE4M3fn.id();
        } else if (b_scales.scalar_type() == at::ScalarType::Float8_e8m0fnu) {
            // MXFP4 使用 E8M0 scale。
            s_type_id = vllm::kFE8M0fnu.id();
        } else {
            // FP4 权重只允许这两种 scale dtype。
            TORCH_CHECK(false,
                        "When b_type = float4_e2m1f, b_scale scalar type must be",
                        "float8_e4m3fn (for NVFP4) or float8_e8m0fnu (for MXFP4).");
        }
    }

    // 将入口推导出的 id 转成内部 ScalarType 对象，供 selector 与 kernel 使用。
    vllm::ScalarType a_type = vllm::ScalarType::from_id(a_type_id);

    // B 的量化类型由 Python 侧显式传入。
    vllm::ScalarType b_type = vllm::ScalarType::from_id(b_type_id);

    // C 的输出/累加类型来自上面的 dtype 推导。
    vllm::ScalarType c_type = vllm::ScalarType::from_id(c_type_id);

    // scale 类型可能等于 C，也可能是 FP4 专用 scale 类型。
    vllm::ScalarType s_type = vllm::ScalarType::from_id(s_type_id);

    // pack_factor 表示一个 32 bit 权重 word 对应的输出元素数。
    int pack_factor = 32 / b_type.size_bits();

    // ------------------------------- 校验 A 与 B 的基础形状 -------------------------------
    // A: [size_m, size_k]，M 维必须匹配调用方传入的 size_m。
    TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
                ", size_m = ", size_m);

    // A: [size_m, size_k]，K 维必须匹配调用方传入的 size_k。
    TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
                ", size_k = ", size_k);

    // B 的 K 维必须按 Marlin tile_size 对齐。
    TORCH_CHECK(
        size_k % MARLIN_NAMESPACE_NAME::tile_size == 0, "size_k = ", size_k,
        " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);

    // b_q_weight 第 0 维对应 K tile 数，必须等于 size_k / tile_size。
    TORCH_CHECK((size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_weight.size(0),
                "Shape mismatch: b_q_weight.size(0) = ", b_q_weight.size(0),
                ", size_k = ", size_k,
                ", tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);

    // b_q_weight 第 1 维也按 tile_size 对齐，便于 kernel 按 tile 访问 N 维。
    TORCH_CHECK(
        b_q_weight.size(1) % MARLIN_NAMESPACE_NAME::tile_size == 0,
        "b_q_weight.size(1) = ", b_q_weight.size(1),
        " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);

    // 从压缩权重形状和 pack_factor 反推出真实 N 维。
    int actual_size_n =
            (b_q_weight.size(1) / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;

    // 调用方传入的 size_n 必须与压缩权重反推结果一致。
    TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
                ", actual_size_n = ", actual_size_n);

    // ------------------------------- 校验设备、连续性和对齐 -------------------------------
    // A 必须在 CUDA 设备上，因为底层只注册 CUDA kernel。
    TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");

    // A 的 K 维必须连续，保证 kernel 按行内连续加载。
    TORCH_CHECK(a.stride(1) == 1, "A.stride(1) is not 1");

    // A 使用 int4 即 16B 向量化加载，需按元素字节数计算真实行跨度。
    int64_t a_row_stride_bytes =
            a.stride(0) * static_cast<int64_t>(a.element_size());

    // FP16/BF16 要求 stride(0)%8==0；FP8/INT8 要求 stride(0)%16==0。
    TORCH_CHECK(a_row_stride_bytes % 16 == 0,
                "A row stride in bytes must be divisible by 16, got ",
                a_row_stride_bytes);

    // A 起始地址也必须 16B 对齐，避免 int4 加载未对齐。
    TORCH_CHECK(((uint64_t)a.data_ptr()) % 16 == 0, "A must aligned to 16 bytes");

    // 压缩权重必须在 CUDA 上。
    TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");

    // 压缩权重必须连续，kernel 直接按压缩布局线性读取。
    TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

    // scale 张量必须在 CUDA 上。
    TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");

    // scale 张量必须连续，kernel 按 group/N 维顺序读取。
    TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");

    // a_scales 用于 8bit activation 的 per-token 输入缩放。
    torch::Tensor a_scales;

    // options 用于分配输出和空占位 Tensor，dtype 跟随 c_dtype。
    auto options = torch::TensorOptions().dtype(c_dtype).device(a.device());

    // options_fp32 用于 FP32 reduce 临时缓冲和空 scale 占位。
    auto options_fp32 =
            torch::TensorOptions().dtype(at::kFloat).device(a.device());

    // 8bit activation 路径必须由调用方传入 a_scales。
    if (a_scales_or_none.has_value()) {
        // 保存调用方传入的 a_scales。
        a_scales = a_scales_or_none.value();

        // a_scales 只能服务于 8bit activation。
        TORCH_CHECK(a_type.size_bits() == 8,
                    "a_scales can only be used for 8bit activation.");
    } else {
        // 非 8bit activation 使用空 Tensor 占位，底层不会读取。
        a_scales = torch::empty({0}, options_fp32);

        // 8bit activation 缺少 a_scales 时无法反量化输入。
        TORCH_CHECK(a_type.size_bits() != 8,
                    "the a_scales parameter must be passed for 8bit activation.");
    }

    // thread_k=-1 表示让 marlin_mm 自动选择 K 维 thread tile。
    int thread_k = -1;

    // thread_n=-1 表示让 marlin_mm 自动选择 N 维 thread tile。
    int thread_n = -1;

    // sms 记录当前设备 SM 数，用于 grid 大小与 workspace 校验。
    int sms = -1;

    // 从当前 A 所在设备查询 SM 数。
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, a.get_device());

    // ------------------------------- 分配输出与临时缓冲 -------------------------------
    // 设置 CUDA guard，确保后续 Tensor 分配落在 A 所在设备。
    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));

    // c 是最终输出 Tensor，可能由调用方提供，也可能在此处分配。
    torch::Tensor c;

    // 若调用方传入 c，则直接写入该输出缓冲。
    if (c_or_none.has_value()) {
        // 取出外部输出 Tensor。
        c = c_or_none.value();

        // 输出 Tensor 必须位于 CUDA。
        TORCH_CHECK(c.device().is_cuda(), "c is not on GPU");

        // 输出 Tensor 必须连续，kernel 按行主序写回。
        TORCH_CHECK(c.is_contiguous(), "c is not contiguous");

        // 输出 M 维必须等于 size_m。
        TORCH_CHECK(c.size(0) == size_m, "Shape mismatch: c.size(0) = ", c.size(0),
                    ", size_m = ", size_m);

        // 输出 N 维必须等于 size_n。
        TORCH_CHECK(c.size(1) == size_n, "Shape mismatch: c.size(1) = ", c.size(1),
                    ", size_n = ", size_n);
    } else {
        // 未提供输出时，按 [size_m, size_n] 分配一个新的输出 Tensor。
        c = torch::empty({size_m, size_n}, options);
    }

    // 空 M 直接返回输出 Tensor，避免启动非法空 kernel。
    if (size_m == 0) return c;

    // c_tmp 只在 FP32 全局 reduce 路径使用。
    torch::Tensor c_tmp;

    // FP32 reduce 需要额外临时缓冲保存每个 SM 的部分结果。
    if (use_fp32_reduce) {
        // max_m_block_size 向上对齐到 16，并限制到 kernel 最高 64 行。
        int max_m_block_size = (size_m + 16 - 1) / 16 * 16;

        max_m_block_size = min(max_m_block_size, 64);

        // c_tmp 大小按 SM 数、最大 M block 和最大 N tile 估算。
        int max_c_tmp_size =
                sms * max_m_block_size * MARLIN_NAMESPACE_NAME::max_thread_n;

        // 分配 FP32 临时缓冲，供 global reduce 写入。
        c_tmp = torch::empty({max_c_tmp_size}, options_fp32);
    } else {
        // 非 FP32 reduce 路径传空 Tensor 占位，底层不会读取。
        c_tmp = torch::empty({0}, options_fp32);
    }

    // ------------------------------- 推导 group_size 与 act-order 状态 -------------------------------
    // num_groups 初始为 -1，后续由 b_scales 第 0 维确定。
    int num_groups = -1;

    // group_size 初始为 -1，表示 per-channel 或单组量化。
    int group_size = -1;

    // b_scales 期望为二维：[num_groups, size_n]。
    int rank = b_scales.sizes().size();

    // b_scales 必须是二维 Tensor。
    TORCH_CHECK(rank == 2, "b_scales rank = ", rank, " is not 2");

    // b_scales 第 1 维对应真实输出 N 维。
    TORCH_CHECK(b_scales.size(1) == size_n, "b_scales dim 1 = ", b_scales.size(1),
                " is not size_n = ", size_n);

    // b_scales 第 0 维表示 scale group 数。
    num_groups = b_scales.size(0);

    // g_idx/perm/a_tmp 用于 act-order 路径；非 act-order 时为空占位。
    torch::Tensor g_idx, perm, a_tmp;

    // g_idx 和 perm 必须成对出现，分别描述 group id 与 K 维重排。
    if (g_idx_or_none.has_value() && perm_or_none.has_value()) {
        // 取出 act-order 的 group 索引。
        g_idx = g_idx_or_none.value();

        // 取出 act-order 的 K 维重排索引。
        perm = perm_or_none.value();

        // g_idx 必须在 CUDA 上，底层 kernel 直接读取。
        TORCH_CHECK(g_idx.device().is_cuda(), "g_idx is not on GPU");

        // g_idx 必须连续，保证线性访问合法。
        TORCH_CHECK(g_idx.is_contiguous(), "g_idx is not contiguous");

        // perm 必须在 CUDA 上，列重排 kernel 直接读取。
        TORCH_CHECK(perm.device().is_cuda(), "perm is not on GPU");

        // perm 必须连续，保证线性访问合法。
        TORCH_CHECK(perm.is_contiguous(), "perm is not contiguous");

        // g_idx/perm 要么同时为空，要么都覆盖完整 K 维。
        TORCH_CHECK((g_idx.size(-1) == 0 && perm.size(-1) == 0) ||
                    (g_idx.size(-1) == size_k && perm.size(-1) == size_k),
                    "Unexpected g_idx.size(-1) = ", g_idx.size(-1),
                    " and perm.size(-1) = ", perm.size(-1),
                    ", where size_k = ", size_k);
    } else {
        // 非 act-order 路径使用空 g_idx 占位。
        g_idx = torch::empty({0}, options);

        // 非 act-order 路径使用空 perm 占位。
        perm = torch::empty({0}, options);

        // 非 act-order 路径不需要 A 重排临时缓冲。
        a_tmp = torch::empty({0}, options);
    }

    // 同时存在 g_idx 和 perm 时才启用 act-order。
    bool has_act_order = g_idx.size(-1) > 0 && perm.size(-1) > 0;

    // act-order 路径需要准备 A 的重排缓冲并推导 group_size。
    if (has_act_order) {
        // a_tmp: [size_m, size_k]，保存 permute_cols_kernel 重排后的 A。
        a_tmp = torch::empty({size_m, size_k}, options);

        // 完整 K 场景可以从 scale group 数直接还原 group_size。
        if (is_k_full) {
            // act-order 完整 K 至少需要多个 group，否则 g_idx 没有意义。
            TORCH_CHECK(num_groups > 1, "For act_order, num_groups must be > 1");

            // 完整 K 必须能均分到各 scale group。
            TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                        ", is not divisible by num_groups = ", num_groups);

            // group_size 表示每个 scale group 覆盖的 K 元素数。
            group_size = size_k / num_groups;
        } else {
            // 非完整 K 的 TP 分片无法静态确定 group_size，用 0 交给 g_idx 动态处理。
            group_size = 0;
        }

        // 非 act-order 路径不需要 A 重排，group_size 由 scale group 数静态推导。
    } else {
        // 非 act-order 不需要 A 临时缓冲。
        a_tmp = torch::empty({0}, options);

        // 多 group scale 路径按 size_k / num_groups 推导 group_size。
        if (num_groups > 1) {
            // K 维必须能被 group 数整除，保证每个 group 等宽。
            TORCH_CHECK(
                size_k % num_groups == 0, "size_k = ", size_k,
                ", is not divisible by b_scales.size(0) = ", b_scales.size(0));

            // 得到普通 grouped quant 的每组 K 元素数。
            group_size = size_k / num_groups;
        } else {
            // 单 group 路径用 -1 传给底层，表示不需要按 group 切分 K。
            group_size = -1;
        }
    }

    // global_scale 只用于 NVFP4 额外全局缩放路径。
    torch::Tensor global_scale;

    // 调用方传入 global_scale 时必须匹配 NVFP4 格式。
    if (global_scale_or_none.has_value()) {
        // 取出全局缩放 Tensor。
        global_scale = global_scale_or_none.value();

        // 只有 FE2M1 权重 + E4M3 scale 的 NVFP4 路径允许 global_scale。
        TORCH_CHECK(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn,
                    "global_scale can only be used for nvfp4 format.");
    } else {
        // 非 NVFP4 路径使用空 Tensor 占位。
        global_scale = torch::empty({0}, options);

        // NVFP4 路径必须传入 global_scale，否则无法还原最终缩放。
        TORCH_CHECK(!(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn),
                    "the global_scale parameter must be passed for nvfp4 format.");
    }

    // bias 可选，存在时底层 kernel 会在输出前加上 bias。
    bool has_bias = b_bias_or_none.has_value();

    // b_bias 使用 Tensor 占位，保证 data_ptr 总是可传给底层函数。
    torch::Tensor b_bias;

    // 调用方传入 bias 时校验其设备和形状。
    if (has_bias) {
        // 取出 bias Tensor。
        b_bias = b_bias_or_none.value();

        // bias 必须在 CUDA 上。
        TORCH_CHECK(b_bias.device().is_cuda(), "b_bias is not on GPU");

        // bias 必须连续，kernel 按 N 维线性读取。
        TORCH_CHECK(b_bias.is_contiguous(), "b_bias is not contiguous");

        // bias: [size_n]，长度必须等于输出 N 维。
        TORCH_CHECK(b_bias.size(0) == size_n, "b_bias.size(0) != size_n");

        // bias 一维 stride 必须为 1。
        TORCH_CHECK(b_bias.stride(0) == 1, "b_bias.stride(0) != 1");
    } else {
        // 无 bias 时传空 Tensor 占位，底层由 has_bias 控制不读取。
        b_bias = torch::empty({0}, options);
    }

    // b_zeros 保存非对称量化 zero-point，可选。
    torch::Tensor b_zeros;

    // 调用方传入 zero-point 时校验其设备与连续性。
    if (b_zeros_or_none.has_value()) {
        // 取出 zero-point Tensor。
        b_zeros = b_zeros_or_none.value();

        // zero-point 必须在 CUDA 上。
        TORCH_CHECK(b_zeros.device().is_cuda(), "b_zeros is not on GPU");

        // zero-point 必须连续，kernel 按 group/N 维线性读取。
        TORCH_CHECK(b_zeros.is_contiguous(), "b_zeros is not contiguous");
    } else {
        // 对称量化或无 zero-point 路径使用空 Tensor 占位。
        b_zeros = torch::empty({0}, options);
    }

    // zero-point Tensor 非空时启用 has_zp 路径。
    bool has_zp = b_zeros.size(-1) > 0;

    // 非对称量化只支持无 block 打包后缀的 u4/u8 权重。
    if (has_zp) {
        TORCH_CHECK(
            b_type == vllm::kU4 || b_type == vllm::kU8,
            "b_type must be u4 or u8 when has_zp = True. Got = ", b_type.str());
    } else {
        // 无 zero-point 时允许对称 int、FP8、FP4 以及带固定 block scale 的无 zp 格式。
        TORCH_CHECK(b_type == vllm::kU4B8 || b_type == vllm::kU8B128 ||
                    b_type == vllm::kS4 || b_type == vllm::kS8 ||
                    b_type == vllm::kFE4M3fn || b_type == vllm::kFE2M1f,
                    "b_type must be uint4b8, uint8b128, int4, int8, "
                    "float8_e4m3fn or float4_e2m1f when has_zp = False. Got = ",
                    b_type.str());
    }

    // float zero-point 路径只支持 half 计算，避免 BF16/int8 组合缺少对应 kernel。
    if (has_zp && is_zp_float) {
        TORCH_CHECK(a.scalar_type() == at::ScalarType::Half,
                    "Computation type must be float16 (half) when using float zero "
                    "points.");
    }

    // ------------------------------- 校验 zero-point 形状 -------------------------------
    // has_zp 时，b_zeros 必须与 scale group 和输出 N 维匹配。
    if (has_zp) {
        // b_zeros 期望为二维。
        int rank = b_zeros.sizes().size();

        // zero-point 必须是 [num_groups, ...] 的二维布局。
        TORCH_CHECK(rank == 2, "b_zeros rank = ", rank, " is not 2");

        // float zero-point 不打包 N 维，第二维直接等于 size_n。
        if (is_zp_float) {
            // b_zeros: [num_groups, size_n]。
            TORCH_CHECK(b_zeros.size(1) == size_n,
                        "b_zeros dim 1 = ", b_zeros.size(1),
                        " is not size_n = ", size_n);

            // 第 0 维必须等于 scale group 数。
            TORCH_CHECK(num_groups == b_zeros.size(0),
                        "b_zeros dim 0 = ", b_zeros.size(0),
                        " is not num_groups = ", num_groups);

            // float zero-point 需要明确 group 数，不能保留默认 -1。
            TORCH_CHECK(num_groups != -1, "num_groups must be != -1");

            // 整数 zero-point 按权重位宽打包 N 维。
        } else {
            // 第 0 维必须等于 scale group 数。
            TORCH_CHECK(b_zeros.size(0) == num_groups,
                        "b_zeros dim 0 = ", b_zeros.size(0),
                        " is not num_groups = ", num_groups);

            // 第 1 维按 pack_factor 压缩后的 N 维长度校验。
            TORCH_CHECK(b_zeros.size(1) == size_n / pack_factor,
                        "b_zeros dim 1 = ", b_zeros.size(1),
                        " is not size_n / pack_factor = ", size_n / pack_factor);
        }
    }

    // ------------------------------- 校验 workspace 与最终 dtype 约束 -------------------------------
    // N 维必须满足 Marlin 最小 N tile 对齐。
    TORCH_CHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0,
                "size_n = ", size_n, ", is not divisible by min_thread_n = ",
                MARLIN_NAMESPACE_NAME::min_thread_n);

    // workspace 至少需要 sms 个 int 锁位。
    int min_workspace_size = sms;

    // 调用方传入 workspace 太小时，底层归约锁会越界。
    TORCH_CHECK(workspace.numel() >= min_workspace_size,
                "workspace.numel = ", workspace.numel(),
                " is below min_workspace_size = ", min_workspace_size);

    // 记录 A 所在 CUDA 设备，后续用于流和底层设备属性查询。
    int dev = a.get_device();

    // a_scales 即使为空占位也必须是 float，保持底层指针类型一致。
    TORCH_CHECK(a_scales.scalar_type() == at::ScalarType::Float,
                "scalar type of a_scales must be float");

    // global_scale 即使为空占位也必须与 C dtype 一致，匹配底层参数解释。
    TORCH_CHECK(global_scale.scalar_type() == c.scalar_type(),
                "scalar type of global_scale must be the same with c");

    // 16bit activation 路径要求 A/C dtype 一致，避免 kernel 输出解释错误。
    if (a_type.size_bits() == 16) {
        TORCH_CHECK(
            a.scalar_type() == c.scalar_type(),
            "scalar type of a must be the same with c for 16 bit activation");
    }

    // 将已校验的 Tensor 指针和推导出的类型/形状交给底层 Marlin 调度函数。
    marlin::marlin_mm(
        // ------------------------------- 输入、输出和临时缓冲 -------------------------------
        // A：activation 矩阵，形状按 [M, K] 解释。
        a.data_ptr(),
        // B：已经按 Marlin 布局 pack 后的量化权重。
        b_q_weight.data_ptr(),
        // C：最终输出矩阵，形状按 [M, N] 解释。
        c.data_ptr(),
        // c_tmp：FP32/global reduce 路径的分块累加缓冲，普通路径为空占位。
        c_tmp.data_ptr(),
        // b_bias：可选 bias；has_bias=false 时底层不会读取。
        b_bias.data_ptr(),
        // a_scales：A=int8/fp8 等量化 activation 路径使用的 scale。
        a_scales.data_ptr(),
        // b_scales：B 权重量化 scale，GPTQ/Marlin 路径通常按 group 或 channel 存储。
        b_scales.data_ptr(),
        // global_scale：全局缩放因子；无该语义时传空 Tensor 占位。
        global_scale.data_ptr(),
        // b_zeros：非对称量化 zero-point；has_zp=false 时底层不会读取。
        b_zeros.data_ptr(),
        // g_idx：act-order/group 索引，描述每个 K 位置使用哪个量化 group。
        g_idx.data_ptr(),
        // perm：act-order 下从排序后 K 位置还原到原始 K 位置的索引。
        perm.data_ptr(),
        // a_tmp：act-order 需要重排 A 时使用的临时缓冲，非 act-order 路径为空占位。
        a_tmp.data_ptr(),

        // ------------------------------- 问题规模和内存步长 -------------------------------
        // size_m/size_n/size_k 对应 GEMM 的 M/N/K。
        size_m,
        size_n,
        size_k,
        // A 的行跨度按元素数传入，底层结合 a_type 解释真实字节跨度。
        a.stride(0),
        // workspace：threadblock 调度和归约状态缓冲。
        workspace.data_ptr(),

        // ------------------------------- 数据类型描述 -------------------------------
        // a/b/c/s 分别描述 activation、权重、输出和 scale 的内部标量类型。
        a_type,
        b_type,
        c_type,
        s_type,

        // ------------------------------- 量化和布局开关 -------------------------------
        // has_bias：是否读取并加上 b_bias。
        has_bias,
        // has_act_order：是否启用 GPTQ act-order 对应的 K 重排路径。
        has_act_order,
        // is_k_full：当前 rank 是否持有完整 K，影响 act-order 的处理方式。
        is_k_full,
        // has_zp：是否存在 zero-point。
        has_zp,
        // num_groups/group_size 描述 B scale/zero 的 group 布局。
        num_groups,
        group_size,

        // ------------------------------- 执行配置 -------------------------------
        // dev/stream 指定当前 CUDA 设备和提交 kernel 的 stream。
        dev,
        at::cuda::getCurrentCUDAStream(dev),
        // thread_k/thread_n 指定单个 threadblock 覆盖的 K/N tile 大小。
        thread_k,
        thread_n,
        // sms 为当前设备 SM 数，用于 workspace 和 reduce 调度。
        sms,
        // use_atomic_add 控制小 M*N、大 K 场景下是否用 atomicAdd reduce。
        use_atomic_add,
        // use_fp32_reduce 控制是否启用 FP32 临时缓冲归约。
        use_fp32_reduce,
        // is_zp_float 标记 zero-point 是否按浮点数解释。
        is_zp_float);

    // 返回写入后的输出 Tensor。
    return c;
}

#endif

// 注册 CUDA 实现，使 Python 侧可以通过 torch ops 调用 marlin_gemm。
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
    // 将 C++ 入口绑定到当前扩展命名空间下的 marlin_gemm op。
    m.impl("marlin_gemm", &marlin_gemm);
}
