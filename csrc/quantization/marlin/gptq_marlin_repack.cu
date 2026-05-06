#include "marlin.cuh"

#include "core/registration.h"

namespace marlin {

// ------------------------------- GPTQ Marlin qweight repack kernel -------------------------------
    // GPTQ packed qweight -> Marlin tile-interleaved 布局。
    // num_bits: 权重量化位宽，4/8 bit。
    // has_perm: 是否使用 act-order 排序。
    // is_a_8bit: 激活是否为 8-bit。
    template<int const num_threads, int const num_bits, bool const has_perm,
            bool is_a_8bit>
    __global__ void gptq_marlin_repack_kernel(
            // 原始 packed 权重: [size_k / pack_factor, size_n]
            uint32_t const *__restrict__ b_q_weight_ptr,
            // act-order 排序索引；无 perm 时为空
            uint32_t const *__restrict__ perm_ptr,
            // repack 后的 Marlin 布局输出
            uint32_t *__restrict__ out_ptr,
            // 逻辑 K/N 矩阵尺寸
            int size_k,
            int size_n) {

        // 每个 uint32 能打包的量化值个数。
        constexpr int pack_factor = 32 / num_bits;

        /*
         * 非8_bit: 16 * 64 每 block
         * 8_bit: 32 * 32 每 block
         * */
        // 8-bit 激活时，N tile 减半。
        constexpr int target_tile_n_size = tile_n_size / (is_a_8bit ? 2 : 1);

        // 8-bit 激活时，K tile 加倍。
        constexpr int target_tile_k_size = tile_k_size * (is_a_8bit ? 2 : 1);

        // K 方向 tile 总数。
        int k_tiles = size_k / target_tile_k_size;

        // N 方向 tile 总数。
        int n_tiles = size_n / target_tile_n_size;

        // 每个 block 负责的 K tile 数。
        // gridDim 共有多少block
        int block_k_tiles = div_ceil(k_tiles, gridDim.x);

        // 当前 block 负责的起始 K tile。
        auto start_k_tile = blockIdx.x * block_k_tiles;

        // block 超出有效 K tile 范围则退出。
        if (start_k_tile >= k_tiles) {
            return;
        }

        // 当前 block 负责的结束 K tile。
        int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

        // 定义等待 shared memory stage 就绪的函数。
        auto wait_for_stage = [&]() {

            // 等待 cp.async 搬运完成。
            // 最多允许6个还未搬运完成，即至少搬运了2个stage
            cp_async_wait<repack_stages - 2>();

            // 同步 block 内所有线程。
            __syncthreads();
        };

        // ------------------------------- shared memory 布局 -------------------------------

        // 声明动态 shared memory。
        extern __shared__ int4 sh[];

        // perm 区大小，单位是 int4(int32 * 4)。
        constexpr int perm_size = target_tile_k_size / 4;

        // shared memory 起始位置作为 perm 区。
        int4 *sh_perm_ptr = sh;

        // pipe 区默认从 shared memory 起点开始。
        int4 *sh_pipe_ptr = sh_perm_ptr;

        // 如果有 perm，需要跳过 perm 存储区。
        if constexpr (has_perm) {

            // pipe 区移动到 perm 区之后。
            sh_pipe_ptr += perm_size;
        }

        // 当前 K tile packed 后的 uint32 行数。
        constexpr int tile_ints = target_tile_k_size / pack_factor;

        // N 方向每 4 个 uint32 用一个 int4 搬运。
        constexpr int stage_n_threads = target_tile_n_size / 4;

        // perm 路径按解包 K 元素搬；普通路径按 packed 行搬。
        constexpr int stage_k_threads = has_perm ? target_tile_k_size : tile_ints;

        // 一个 pipeline stage 占用的 int4 数量。
        constexpr int stage_size = stage_k_threads * stage_n_threads;

        // ------------------------------- 加载 perm -------------------------------

        // 定义把当前 K tile 的 perm 加载到 shared memory 的函数。
        [[maybe_unused]] auto load_perm_to_shared = [&](int k_tile_id) {

            // 当前 K tile 的 perm 起点，单位 int4。
            int first_k_int4 = (k_tile_id * target_tile_k_size) / 4;

            // 把 perm 指针转换成 int4 指针，方便 16 字节搬运。
            int4 const *perm_int4_ptr = reinterpret_cast<int4 const *>(perm_ptr);

            // 只有前 perm_size 个线程负责加载 perm。
            if (threadIdx.x < perm_size) {

                // 每个线程加载一个 int4 perm 块。
                sh_perm_ptr[threadIdx.x] = perm_int4_ptr[first_k_int4 + threadIdx.x];
            }

            // 确保 perm 已经写入 shared memory。
            __syncthreads();
        };

        // ------------------------------- 加载 qweight -------------------------------

        // 定义异步加载 qweight tile 到 shared memory 的函数。
        auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {

            // 如果 N tile 越界，只提交 fence 后返回。
            if (n_tile_id >= n_tiles) {

                // 结束当前 async copy 批次。
                cp_async_fence();

                // 越界 tile 不加载。
                return;
            }

            // 当前 N tile 的起始 N 下标。
            int first_n = n_tile_id * target_tile_n_size;

            // 当前 pipe 对应的 shared memory stage。
            int4 *sh_ptr = sh_pipe_ptr + stage_size * pipe;

            // act-order 路径。
            if constexpr (has_perm) {

                // 有效线程参与加载。
                if (threadIdx.x < stage_size) {

                    // k_id在是否has_perm下, 不同stage_size下，有不同的范围和含义
                    // 当前线程负责的 K 元素编号。
                    auto k_id = threadIdx.x / stage_n_threads;

                    // 当前线程负责的 N 方向 int4 编号。
                    auto n_id = threadIdx.x % stage_n_threads;

                    // shared memory 中的 perm 按 uint32 读取。
                    uint32_t const *sh_perm_int_ptr =
                            reinterpret_cast<uint32_t const *>(sh_perm_ptr);

                    // 排序后 K 位置映射到原始 K 位置。
                    int src_k = sh_perm_int_ptr[k_id];

                    // 原始 K 位置对应的 packed 行号。
                    int src_k_packed = src_k / pack_factor;

                    // 异步搬运一个 int4，也就是 16 字节。
                    cp_async4(
                            // 目标地址：当前 shared stage。
                            &sh_ptr[k_id * stage_n_threads + n_id],
                            // 源地址：原始 qweight 的 packed 位置。
                            reinterpret_cast<int4 const *>(&(
                                    b_q_weight_ptr[src_k_packed * size_n
                                                   + first_n + (n_id * 4)])));
                }

                // 非 act-order 路径。按照一个线程int4单位读取
            } else {

                // 有效线程参与加载。
                if (threadIdx.x < stage_size) {

                    // 当前线程负责的 packed K 行编号。(stage定位)
                    auto k_id = threadIdx.x / stage_n_threads;

                    // 当前线程负责的 N 方向 int4 编号。(stage定位)
                    auto n_id = threadIdx.x % stage_n_threads;

                    // 当前 K tile 的起始逻辑 K。(全局定位)
                    int first_k = k_tile_id * target_tile_k_size;

                    // 当前 K tile 的起始 packed 行。
                    int first_k_packed = first_k / pack_factor;

                    // 异步搬运一个 int4 到 shared memory。
                    cp_async4(
                            // 目标地址：当前 shared stage。
                            &sh_ptr[k_id * stage_n_threads + n_id],
                            // 源地址：连续 packed qweight。
                            reinterpret_cast<int4 const *>(
                                    &(b_q_weight_ptr[(first_k_packed + k_id) * size_n
                                                     + first_n + (n_id * 4)])));
                }
            }

            // 提交当前 pipe 的 async copy。
            cp_async_fence();
        };

        // ------------------------------- unpack + repack -------------------------------

        // 定义单个 tile 的 unpack + repack 逻辑。
        auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {

            // N tile 越界则不处理。
            if (n_tile_id >= n_tiles) {
                return;
            }

            // 当前线程所在 warp 编号。
            auto warp_id = threadIdx.x / 32;

            // 当前线程在 warp 内的 lane 编号。
            auto th_id = threadIdx.x % 32;

            // 只用前 4 个 warp 执行 repack。
            if (warp_id >= 4) {
                return;
            }

            // 当前线程负责的 tile 内列编号。
            int tc_col = th_id / 4;

            // 当前线程负责的 tile 内 K 行起点。
            int tc_row = (th_id % 4) * (is_a_8bit ? 4 : 2);

            // 普通路径下的 K 行交错偏移。
            constexpr int tc_offsets[4] = {0, 1, 8, 9};

            // 当前线程负责的 N 维位置。
            int cur_n = (warp_id / (is_a_8bit ? 2 : 1)) * 16 + tc_col;

            // shared memory 中一行的跨度。
            constexpr int sh_stride = target_tile_n_size;

            // 用于取出 num_bits 位的掩码。
            constexpr uint32_t mask = (1 << num_bits) - 1;

            // 当前 pipe 的 shared stage 起点。
            int4 *sh_stage_ptr = sh_pipe_ptr + stage_size * pipe;

            // shared stage 按 uint32 访问。
            uint32_t *sh_stage_int_ptr = reinterpret_cast<uint32_t *>(sh_stage_ptr);

            // shared perm 按 uint32 访问。
            [[maybe_unused]] uint32_t *sh_perm_int_ptr =
                    reinterpret_cast<uint32_t *>(sh_perm_ptr);

            // 当前线程抽取的 8 个量化值。
            // 一个线程负责两列，每列4个元素
            uint32_t vals[8];

            // act-order 路径。
            if constexpr (has_perm) {

                // act-order 不支持 8-bit 激活。
                static_assert(!is_a_8bit);

                // 每个线程先取 4 个 K 位置。
                for (int i = 0; i < 4; i++) {

                    // 当前要读取的排序后 K 下标。
                    int k_idx = tc_row + tc_offsets[i];

                    // 排序后 K 对应的原始 K。
                    uint32_t src_k = sh_perm_int_ptr[k_idx];

                    // 原始 K 在 packed uint32 内的位置。
                    uint32_t src_k_pos = src_k % pack_factor;

                    // 第一个 N 半区的 packed 值。
                    uint32_t b1_val = sh_stage_int_ptr[k_idx * sh_stride + cur_n];

                    // 第二个 N 半区的 packed 值。
                    uint32_t b2_val = sh_stage_int_ptr[k_idx * sh_stride + cur_n + 8];

                    // 从第一个半区抽取量化值。
                    vals[i] = (b1_val >> (src_k_pos * num_bits)) & mask;

                    // 从第二个半区抽取量化值。
                    vals[4 + i] = (b2_val >> (src_k_pos * num_bits)) & mask;
                }

                // 非 act-order 路径。
            } else {

                // 保存第一个 N 半区的 packed 行。
                uint32_t b1_vals[tile_ints];

                // 保存第二个 N 半区的 packed 行。
                [[maybe_unused]] uint32_t b2_vals[tile_ints];

                // 展开循环读取 packed 行。
#pragma unroll

                // 遍历当前 tile 的 packed K 行。
                for (int i = 0; i < tile_ints; i++) {

                    // 8-bit 激活路径。
                    if constexpr (is_a_8bit) {

                        // 按 warp 半区读取 packed 值。
                        b1_vals[i] =
                                sh_stage_int_ptr[cur_n + sh_stride * i + (warp_id % 2) * 8];

                        // 普通激活路径。
                    } else {

                        // 读取第一个 N 半区。
                        b1_vals[i] = sh_stage_int_ptr[cur_n + sh_stride * i];

                        // 读取第二个 N 半区。
                        b2_vals[i] = sh_stage_int_ptr[cur_n + 8 + sh_stride * i];
                    }
                }

                // 展开循环抽取量化值。
#pragma unroll

                // 每次抽取一对半区值。
                for (int i = 0; i < 4; i++) {

                    // 当前逻辑 K 元素编号。
                    int cur_elem = tc_row + (is_a_8bit ? i : tc_offsets[i]);

                    // 当前元素所在的 packed uint32 编号。
                    int cur_int = cur_elem / pack_factor;

                    // 当前元素在 packed uint32 内的槽位。
                    int cur_pos = cur_elem % pack_factor;

                    // 从第一个半区抽取量化值。
                    vals[i] = (b1_vals[cur_int] >> (cur_pos * num_bits)) & mask;

                    // 8-bit 激活路径取后半个 K tile。
                    if constexpr (is_a_8bit) {

                        // 从后半 K tile 抽取量化值。
                        vals[4 + i] =
                                (b1_vals[cur_int + tile_ints / 2]
                                        >> (cur_pos * num_bits)) & mask;

                        // 普通激活路径取第二个 N 半区。
                    } else {

                        // 从第二个N半区抽取量化值。
                        vals[4 + i] = (b2_vals[cur_int] >> (cur_pos * num_bits)) & mask;
                    }
                }
            }

            // 当前 tile 的 packed 输出大小。
            constexpr int tile_size =
                    target_tile_k_size * target_tile_n_size / pack_factor;

            // 当前输出 tile 的起始 offset。
            int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_size;

            // ------------------------------- 写出目标布局 -------------------------------

            // int4 weight + 普通激活路径。
            if constexpr (!is_a_8bit && num_bits == 4) {

                // Marlin int4 普通路径的打包顺序。
                int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};

                // 初始化 packed 输出值。
                uint32_t res = 0;

                // 展开打包循环。
#pragma unroll

                // 将 8 个 int4 重新打包进 uint32。
                for (int i = 0; i < 8; i++) {

                    // 按目标顺序写入 4-bit 槽位。
                    res |= vals[pack_idx[i]] << (i * 4);
                }

                // 写出一个 packed uint32。
                // 同一个 th_id 的 4 个 warp 写在连续的 4 个位置：
                /*
                 *
                 * th_id=0: warp0, warp1, warp2, warp3
                 * th_id=1: warp0, warp1, warp2, warp3
                 * th_id=2: warp0, warp1, warp2, warp3
                 */
                out_ptr[out_offset + th_id * 4 + warp_id] = res;

                // int4 weight + 8-bit 激活路径。
            } else if constexpr (is_a_8bit && num_bits == 4) {

                // Marlin int4 + 8-bit 激活打包顺序。
                /*
                 * Marlin int4 + 8-bit activation 的专用打包顺序。
                 *
                 * 这一分支与普通 int4 路径的关键区别：
                 * - 普通路径的 vals[] 更接近按 K 偏移 [0, 1, 8, 9, ...] 取数
                 * - 本分支的 vals[] 先取前半块的连续 K 偏移 [0, 1, 2, 3]
                 *   再取后半块对应位置 [16, 17, 18, 19]
                 *
                 * 以一个 32x32 逻辑 tile 为例，单个线程在交错前看到的
                 * vals[0..7] 可理解为：
                 *
                 * +---------+--------+-----------------------------------+
                 * | th_id%4 | tc_row | vals[0..7] 对应的逻辑 K 行        |
                 * +---------+--------+-----------------------------------+
                 * |   0     |   0    | 0,1,2,3,16,17,18,19              |
                 * |   1     |   4    | 4,5,6,7,20,21,22,23              |
                 * |   2     |   8    | 8,9,10,11,24,25,26,27            |
                 * |   3     |   12   | 12,13,14,15,28,29,30,31          |
                 * +---------+--------+-----------------------------------+
                 *
                 * 但写入 uint32 时不会按这个顺序直接打包，而是通过 pack_idx
                 * 先做“前半块/后半块交错”：
                 *
                 *   交错前:   0, 1, 2, 3, 16, 17, 18, 19
                 *   pack_idx: 0, 4, 1, 5,  2,  6,  3,  7
                 *   交错后:   0,16, 1,17,  2, 18,  3, 19
                 *
                 * 最终这 8 个 int4 nibble 会按“交错后”的顺序，依次落入同一个
                 * uint32 的 bit[3:0], bit[7:4], ..., bit[31:28]。
                 */
                int pack_idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};

                // 初始化 packed 输出值。
                uint32_t res = 0;

                // 展开打包循环。
#pragma unroll

                // 将 8 个 int4 重新打包进 uint32。
                for (int i = 0; i < 8; i++) {

                    // 按目标顺序写入 4-bit 槽位。
                    res |= vals[pack_idx[i]] << (i * 4);
                }

                /*
                *
                * th_id=0: warp0, warp1, warp2, warp3
                * th_id=1: warp0, warp1, warp2, warp3
                * th_id=2: warp0, warp1, warp2, warp3
                * ...
                * th_id=31: warp0, warp1, warp2, warp3
                */
                // 写出一个 packed uint32。
                out_ptr[out_offset + th_id * 4 + warp_id] = res;

                // int8 weight 路径。
            } else {

                // int8 weight 的打包顺序。
                constexpr int pack_idx[4] = {0, 2, 1, 3};

                // 第一个 packed int8 输出。
                uint32_t res1 = 0;

                // 第二个 packed int8 输出。
                uint32_t res2 = 0;

                // 展开打包循环。
#pragma unroll

                // 每个 uint32 打包 4 个 int8。
                for (int i = 0; i < 4; i++) {

                    // 8-bit 激活路径不使用 pack_idx 重排。
                    const int ii = is_a_8bit ? i : pack_idx[i];

                    // 写入第一个 uint32。
                    res1 |= vals[ii] << (i * 8);

                    // 写入第二个 uint32。
                    res2 |= vals[4 + ii] << (i * 8);
                }

                // 写出第一个 packed uint32。
                out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;

                // 写出第二个 packed uint32。
                out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
            }
        };

        // ------------------------------- pipeline 预取 -------------------------------

        // 定义 pipeline 预填充函数。
        auto start_pipes = [&](int k_tile_id, int n_tile_id) {
            // 展开预取循环。
#pragma unroll

            // 预先填充前 repack_stages - 1 个 pipe。
            for (int pipe = 0; pipe < repack_stages - 1; pipe++) {
                // 预取对应 N tile。
                fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
            }

            // 等待预取数据就绪。
            wait_for_stage();
        };

        // ------------------------------- 主循环 -------------------------------

        // 展开 K tile 循环。
#pragma unroll

        // 遍历当前 block 负责的 K tiles。
        for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
            // 从第 0 个 N tile 开始。
            int n_tile_id = 0;

            // 如果有 perm，先加载当前 K tile 的 perm。
            if constexpr (has_perm) {
                // 加载 perm 到 shared memory。
                load_perm_to_shared(k_tile_id);
            }

            // 预填充 pipeline。
            start_pipes(k_tile_id, n_tile_id);

            // 遍历当前 K tile 下的所有 N tiles。
            while (n_tile_id < n_tiles) {
                // 展开 pipe 循环。
#pragma unroll
                // 每轮处理 repack_stages 个 N tiles。
                for (int pipe = 0; pipe < repack_stages; pipe++) {
                    // 预取后续 N tile。这里从末尾的stage
                    fetch_to_shared((pipe + repack_stages - 1) % repack_stages,
                                    k_tile_id,
                                    n_tile_id + pipe + repack_stages - 1);

                    // 处理当前已就绪的 N tile。
                    repack_tile(pipe, k_tile_id, n_tile_id + pipe);

                    // 等待下一 stage 就绪。
                    wait_for_stage();
                }

                // 前进到下一组 N tiles。
                n_tile_id += repack_stages;
            }
        }
    }

}  // namespace marlin

// ------------------------------- 模板分发宏 -------------------------------

// 根据 num_bits / has_perm / is_a_8bit 选择模板实例并启动 kernel。

#define CALL_IF(NUM_BITS, HAS_PERM, IS_A_8BIT)                             \
  else if (num_bits == NUM_BITS && has_perm == HAS_PERM &&                  \
           is_a_8bit == IS_A_8BIT) {                                        \
    cudaFuncSetAttribute(                                                    \
        marlin::gptq_marlin_repack_kernel<marlin::repack_threads, NUM_BITS, \
                                          HAS_PERM, IS_A_8BIT>,             \
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);       \
    marlin::gptq_marlin_repack_kernel<marlin::repack_threads, NUM_BITS,     \
                                      HAS_PERM, IS_A_8BIT>                  \
        <<<blocks, marlin::repack_threads, max_shared_mem, stream>>>(       \
            b_q_weight_ptr, perm_ptr, out_ptr, size_k, size_n);             \
  }

// ------------------------------- Python/C++ 入口函数 -------------------------------
// b_q_weight: GPTQ packed qweight，形状 [size_k / pack_factor, size_n]。
// perm: act-order 排序索引；为空张量表示不启用 perm。
// size_k / size_n: 解包前的逻辑矩阵尺寸。
// num_bits: 4 或 8。
// is_a_8bit: 输入激活是否为 8-bit。
torch::Tensor gptq_marlin_repack(torch::Tensor &b_q_weight,
                                 torch::Tensor &perm,
                                 int64_t size_k,
                                 int64_t size_n,
                                 int64_t num_bits,
                                 bool is_a_8bit) {
    // ------------------------------- 基础形状检查 -------------------------------
    // Marlin tile 要求 K/N 维能被 tile 大小整除。
    TORCH_CHECK(size_k % marlin::tile_k_size == 0, "size_k = ", size_k,
                " is not divisible by tile_k_size = ", marlin::tile_k_size);

    TORCH_CHECK(size_n % marlin::tile_n_size == 0, "size_n = ", size_n,
                " is not divisible by tile_n_size = ", marlin::tile_n_size);

    // 当前 repack 只支持 int4 / int8 weight。
    TORCH_CHECK(num_bits == 4 || num_bits == 8,
                "num_bits must be 4 or 8. Got = ", num_bits);

    int const pack_factor = 32 / num_bits;

    // ------------------------------- qweight 形状检查 -------------------------------
    TORCH_CHECK((size_k / pack_factor) == b_q_weight.size(0),
                "Shape mismatch: b_q_weight.size(0) = ", b_q_weight.size(0),
                ", size_k = ", size_k, ", pack_factor = ", pack_factor);
    TORCH_CHECK(b_q_weight.size(1) == size_n,
                "b_q_weight.size(1) = ", b_q_weight.size(1),
                " is not size_n = ", size_n);

    // ------------------------------- device / dtype / contiguous 检查 -------------------------------
    TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
    TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");
    TORCH_CHECK(b_q_weight.dtype() == at::kInt, "b_q_weight type is not kInt");

    TORCH_CHECK(perm.device().is_cuda(), "perm is not on GPU");
    TORCH_CHECK(perm.is_contiguous(), "perm is not contiguous");
    TORCH_CHECK(perm.dtype() == at::kInt, "perm type is not at::kInt");

    // ------------------------------- 分配输出 buffer -------------------------------
    // 把当前 CUDA device 临时切换到 b_q_weight 所在的 GPU，并在作用域结束时自动恢复原来的 device。
    const at::cuda::OptionalCUDAGuard device_guard(device_of(b_q_weight));

    auto options = torch::TensorOptions()
            .dtype(b_q_weight.dtype())
            .device(b_q_weight.device());

    /*
     * 与python侧对应:
     *
     *   # 为所有 experts 预先分配输出张量；输出形状中的第二维按 16 行一组压缩，
     *  output = torch.empty(
     *    # 相当于把行方向的16一组 沿着列方向展开
     *   (num_experts, size_k // 16, size_n * (num_bits // 2)),
     *   device=b_q_weight.device,
     *   dtype=b_q_weight.dtype,
     *  )
     *
     * */
    // 输出按 Marlin tile 布局展开。
    torch::Tensor out = torch::empty(
            {
                    size_k / marlin::tile_size,
                    size_n * marlin::tile_size / pack_factor
            },
            options);

    // perm 非空表示启用 act-order。
    bool has_perm = perm.size(0) != 0;

    // ------------------------------- 获取裸指针 -------------------------------
    uint32_t const *b_q_weight_ptr =
            reinterpret_cast<uint32_t const *>(b_q_weight.data_ptr());

    uint32_t const *perm_ptr = reinterpret_cast<uint32_t const *>(perm.data_ptr());

    uint32_t *out_ptr = reinterpret_cast<uint32_t *>(out.data_ptr());

    // ------------------------------- CUDA launch 配置 -------------------------------
    int dev = b_q_weight.get_device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dev);

    // 用 SM 数作为 block 数，kernel 内部再分配 K tiles。
    int blocks;
    cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev);

    // 获取该设备单个 block 可 opt-in 的最大 shared memory。
    int max_shared_mem = 0;
    cudaDeviceGetAttribute(&max_shared_mem,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

    TORCH_CHECK(max_shared_mem > 0);

    // ------------------------------- 模板实例分发 -------------------------------
    if (false) {
    } CALL_IF(4, false, false)CALL_IF(4, true, false)CALL_IF(8, false, false)CALL_IF(8, true, false)

        // 8-bit 激活路径不支持 has_perm=true。
    CALL_IF(4, false, true)CALL_IF(8, false, true)

    else {
        TORCH_CHECK(false, "Unsupported repack config: num_bits = ", num_bits,
                    ", has_perm = ", has_perm, ", is_a_8bit = ", is_a_8bit);
    }

    return out;
}

// ------------------------------- PyTorch CUDA 算子注册 -------------------------------
// 注册为 torch extension 的 CUDA 实现。
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
    m.impl("gptq_marlin_repack", &gptq_marlin_repack);
}
