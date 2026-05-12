#pragma once
#include "vectorization.cuh"

namespace vllm {
    // ------------------------------- 默认向量写入操作封装 -------------------------------

    // 将只定义了单元素处理逻辑的 `scalar_op` 包装成可处理 `VEC_SIZE` 个元素的向量操作。
    template<int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
    struct DefaultVecOp {
        // 保存调用方提供的标量转换逻辑，供向量路径逐元素复用。
        ScaOp scalar_op;

        // 对一个 `vec_n_t<InT, VEC_SIZE>` 输入包逐元素转换，并写入输出向量包。
        __device__ __forceinline__ void operator()(
            vec_n_t<OutT, VEC_SIZE> &dst, const vec_n_t<InT, VEC_SIZE> &src) const {
#pragma unroll
            // 展开 `VEC_SIZE` 个元素，保持向量 load/store 外壳下的逐元素语义。
            for (int i = 0; i < VEC_SIZE; ++i) {
                // 对第 i 个元素执行调用方定义的标量转换逻辑。
                scalar_op(dst.val[i], src.val[i]);
            }
        }
    };

    // ------------------------------- 对齐感知的读写向量化主路径 -------------------------------

    // 在保证语义正确的前提下，尽量使用 `vec_n_t<*, VEC_SIZE>` 做向量化读写。
    template<int VEC_SIZE, typename InT, typename OutT, typename VecOp,
        typename ScaOp>
    __device__ inline void vectorize_with_alignment(
        const InT *in,
        OutT *out,
        int len,
        int tid,
        int stride,
        VecOp &&vec_op,
        ScaOp &&scalar_op) {
        // `VEC_SIZE` 必须是 2 的幂，才能用位运算快速判断对齐和整除关系。
        static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                      "VEC_SIZE must be a positive power-of-two");

        // `WIDTH` 是一次向量化读写覆盖的字节数，例如 FP8 下 16 个元素对应 16B。
        constexpr int WIDTH = VEC_SIZE * sizeof(InT);

        // 将输入指针转成整数地址，用于计算 `in` 是否满足 `WIDTH` 字节对齐。
        uintptr_t addr = reinterpret_cast<uintptr_t>(in);

        // 当输入起始地址对齐 && 总长度能被 `VEC_SIZE` 整除时，整段都可以安全向量化。
        bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);

        // 完全对齐路径直接把 `[len]` 拆成 `[len / VEC_SIZE]` 个向量包处理。
        if (can_vec) {
            // 计算当前连续区间中总共有多少个向量包。
            int num_vec = len / VEC_SIZE;

            // 输入向量包类型，每个元素包含 `VEC_SIZE` 个 `InT`。
            using vin_t = vec_n_t<InT, VEC_SIZE>;

            // 输出向量包类型，每个元素包含 `VEC_SIZE` 个 `OutT`。
            using vout_t = vec_n_t<OutT, VEC_SIZE>;

            // 把输入标量指针解释成向量包指针，用于触发向量化 load。
            auto *v_in = reinterpret_cast<const vin_t *>(in);

            // 把输出标量指针解释成向量包指针，用于触发向量化 store。
            auto *v_out = reinterpret_cast<vout_t *>(out);

            // 当前线程按 `stride` 间隔处理向量包，通常 `stride == blockDim.x`。
            for (int i = tid; i < num_vec; i += stride) {
                // 为当前线程负责的输出向量包分配寄存器临时变量。
                vout_t tmp;

                // 一次性读取一个输入向量包，鼓励编译器生成单条向量化 load。
                vin_t src = v_in[i];

                // 对当前向量包执行调用方提供的向量化转换逻辑。
                vec_op(tmp, src);

                // 一次性写回一个输出向量包，鼓励编译器生成单条向量化 store。
                v_out[i] = tmp;
            }

            // 完全对齐路径已经处理全部元素，直接结束函数。
            return;
        }

        // 计算输入地址相对 `WIDTH` 对齐边界的偏移字节数。
        int misalignment_offset = addr & (WIDTH - 1);

        // 计算从当前地址前进到下一个 `WIDTH` 对齐边界还需要多少字节。
        int alignment_bytes = WIDTH - misalignment_offset;

        // 若当前地址本来已经对齐，则 prefix 应为 0；这里通过按位与处理 `alignment_bytes == WIDTH` 的情况。
        int prefix_elems = alignment_bytes & (WIDTH - 1);

        // 把 prefix 字节数转换成输入元素个数。
        prefix_elems /= sizeof(InT);

        // prefix 不能超过实际长度，避免短数组场景访问越界。
        prefix_elems = min(prefix_elems, len);

        // ------------------------------- 处理未对齐前缀 -------------------------------

        // 未对齐前缀不能安全 reinterpret 成向量包，因此先用标量路径处理。
        for (int i = tid; i < prefix_elems; i += stride) {
            // 对 prefix 中第 i 个元素执行标量转换并写入输出。
            scalar_op(out[i], in[i]);
        }

        // 把输入指针推进到已经处理完的 prefix 之后，此时地址满足向量对齐或 len 已耗尽。
        in += prefix_elems;

        // 把输出指针同步推进到 prefix 之后，保持输入输出元素位置一致。
        out += prefix_elems;

        // 从剩余长度中扣除已经处理的 prefix 元素数。
        len -= prefix_elems;

        // ------------------------------- 处理对齐主体 -------------------------------

        // 剩余区间中可完整向量化处理的向量包数量。
        int num_vec = len / VEC_SIZE;

        // 输入向量包类型，每个包包含 `VEC_SIZE` 个输入元素。
        using vin_t = vec_n_t<InT, VEC_SIZE>;

        // 输出向量包类型，每个包包含 `VEC_SIZE` 个输出元素。
        using vout_t = vec_n_t<OutT, VEC_SIZE>;

        // 将已经对齐的输入指针解释成向量包指针。
        auto *v_in = reinterpret_cast<const vin_t *>(in);

        // 将对应输出指针解释成向量包指针。
        auto *v_out = reinterpret_cast<vout_t *>(out);

        // 当前线程按 `stride` 间隔处理主体区间中的向量包。
        for (int i = tid; i < num_vec; i += stride) {
            // 为当前输出向量包创建寄存器临时变量。
            vout_t tmp;

            // 一次性读取一个输入向量包，鼓励编译器生成向量 load。
            vin_t src = v_in[i];

            // 对当前输入向量包执行调用方提供的向量操作。
            vec_op(tmp, src);

            // 一次性写回一个输出向量包，鼓励编译器生成向量 store。
            v_out[i] = tmp;
        }

        // ------------------------------- 处理尾部剩余元素 -------------------------------

        // 计算主体向量化区间覆盖了多少个标量元素。
        int tail_start = num_vec * VEC_SIZE;

        // 尾部不足一个向量包的元素只能用标量路径处理。
        for (int i = tid + tail_start; i < len; i += stride) {
            // 对尾部第 i 个元素执行标量转换并写入输出。
            scalar_op(out[i], in[i]);
        }
    }

    // ------------------------------- 仅提供标量写入操作的便捷重载 -------------------------------

    // 调用方只提供 `scalar_op` 时，用默认向量操作自动把它包装成向量路径。
    template<int VEC_SIZE, typename InT, typename OutT, typename ScaOp>
    __device__ __forceinline__ void vectorize_with_alignment(const InT *in,
                                                             OutT *out,
                                                             int len,
                                                             int tid,
                                                             int stride,
                                                             ScaOp &&scalar_op) {
        // 构造默认向量操作类型，内部会对向量包逐元素调用 `scalar_op`。
        using Vec = DefaultVecOp<VEC_SIZE, InT, OutT, std::decay_t<ScaOp> >;

        // 复用主实现，让它统一处理前缀、主体向量化和尾部标量路径。
        vectorize_with_alignment<VEC_SIZE>(in,
                                           out,
                                           len,
                                           tid,
                                           stride,
                                           Vec{scalar_op},
                                           std::forward<ScaOp>(scalar_op));
    }

    // ------------------------------- 默认向量只读操作封装 -------------------------------

    // 将只定义了单元素读取逻辑的 `scalar_op` 包装成可读取 `VEC_SIZE` 个元素的向量操作。
    template<int VEC_SIZE, typename InT, typename ScaOp>
    struct DefaultReadVecOp {
        // 保存调用方提供的标量读取逻辑，供向量读路径逐元素复用。
        ScaOp scalar_op;

        // 对一个 `vec_n_t<InT, VEC_SIZE>` 输入包逐元素调用只读标量逻辑。
        __device__ __forceinline__ void operator()(
            const vec_n_t<InT, VEC_SIZE> &src) const {
#pragma unroll
            // 展开 `VEC_SIZE` 个元素，保持一次向量 load 后逐元素消费。
            for (int i = 0; i < VEC_SIZE; ++i) {
                // 对第 i 个元素执行调用方定义的只读标量逻辑。
                scalar_op(src.val[i]);
            }
        }
    };

    // ------------------------------- 对齐感知的只读向量化主路径 -------------------------------

    // 只读取输入 `[len]`，在对齐区间尽量使用 `vec_n_t<InT, VEC_SIZE>` 向量化 load。
    template<int VEC_SIZE, typename InT, typename VecOp, typename ScaOp>
    __device__ inline void vectorize_read_with_alignment(const InT *in,
                                                         int len,
                                                         int tid,
                                                         int stride,
                                                         VecOp &&vec_op,
                                                         ScaOp &&scalar_op) {
        /*
         * 当val是2的幂, 则other_val & (val-1) == 0 --> other_val % val == 0
         *
         */

        // `VEC_SIZE` 必须是 2 的幂，保证对齐判断和取余判断可用位运算表达。
        static_assert(VEC_SIZE > 0 && (VEC_SIZE & (VEC_SIZE - 1)) == 0,
                      "VEC_SIZE must be a positive power-of-two");

        // `WIDTH` 是一个向量包覆盖的字节数。
        constexpr int WIDTH = VEC_SIZE * sizeof(InT);

        // 将输入指针转成整数地址，用于判断对齐。
        uintptr_t addr = reinterpret_cast<uintptr_t>(in);

        // 当起始地址对齐 && 长度可整除 `VEC_SIZE` 时，整段都可走向量化只读路径。
        bool can_vec = ((addr & (WIDTH - 1)) == 0) && ((len & (VEC_SIZE - 1)) == 0);

        // 完全对齐路径直接按向量包遍历整段输入。
        if (can_vec) {
            // 计算总共有多少个输入向量包。
            int num_vec = len / VEC_SIZE;

            // 输入向量包类型，每个包包含 `VEC_SIZE` 个输入元素。
            using vin_t = vec_n_t<InT, VEC_SIZE>;

            // 将输入标量指针解释成向量包指针。
            auto *v_in = reinterpret_cast<const vin_t *>(in);

            // 当前线程按 `stride` 间隔处理输入向量包。
            for (int i = tid; i < num_vec; i += stride) {
                // 一次性读取一个向量包，鼓励编译器生成向量化 load。
                vin_t tmp = v_in[i];

                // 对读取到的向量包执行调用方提供的只读向量逻辑。
                vec_op(tmp);
            }

            // 完全对齐路径已经消费所有输入元素，直接返回。
            return;
        }

        // 计算输入地址相对 `WIDTH` 对齐边界的偏移字节数。
        int misalignment_offset = addr & (WIDTH - 1);

        // 计算到下一个对齐边界还需要的字节数。
        int alignment_bytes = WIDTH - misalignment_offset;

        // 若当前地址已经对齐，则 prefix 字节数应归零。
        int prefix_elems = alignment_bytes & (WIDTH - 1);

        // 把 prefix 字节数换算成输入元素个数。
        prefix_elems /= sizeof(InT);

        // prefix 元素数不能超过总长度，避免短输入越界。
        prefix_elems = min(prefix_elems, len);

        // ------------------------------- 处理未对齐只读前缀 -------------------------------

        // 未对齐 prefix 使用标量读取，避免不安全的向量化 load。
        for (int i = tid; i < prefix_elems; i += stride) {
            // 对 prefix 中第 i 个元素执行只读标量逻辑。
            scalar_op(in[i]);
        }

        // 将输入指针推进到 prefix 之后。
        in += prefix_elems;

        // 从剩余长度中扣除已经处理的 prefix。
        len -= prefix_elems;

        // ------------------------------- 处理对齐只读主体 -------------------------------

        // 剩余区间可完整向量化读取的向量包数量。
        int num_vec = len / VEC_SIZE;

        // 输入向量包类型，每个包包含 `VEC_SIZE` 个输入元素。
        using vin_t = vec_n_t<InT, VEC_SIZE>;

        // 将对齐后的输入指针解释成向量包指针。
        auto *v_in = reinterpret_cast<const vin_t *>(in);

        // 当前线程按 `stride` 间隔读取主体向量包。
        for (int i = tid; i < num_vec; i += stride) {
            // 对当前向量包执行只读向量逻辑。
            vec_op(v_in[i]);
        }

        // ------------------------------- 处理只读尾部剩余元素 -------------------------------

        // 计算主体向量化区间覆盖的标量元素数。
        int tail_start = num_vec * VEC_SIZE;

        // 尾部不足一个向量包的元素走标量读取路径。
        for (int i = tid + tail_start; i < len; i += stride) {
            // 对尾部第 i 个元素执行只读标量逻辑。
            scalar_op(in[i]);
        }
    }

    // ------------------------------- 仅提供标量只读操作的便捷重载 -------------------------------

    // 调用方只提供 `scalar_op` 时，用默认只读向量操作自动包装成向量读取路径。
    template<int VEC_SIZE, typename InT, typename ScaOp>
    __device__ __forceinline__ void vectorize_read_with_alignment(
        const InT *in,
        int len,
        int tid,
        int stride,
        ScaOp &&scalar_op) {
        // 构造默认只读向量操作类型，内部会对向量包逐元素调用 `scalar_op`。
        using Vec = DefaultReadVecOp<VEC_SIZE, InT, std::decay_t<ScaOp> >;

        // 复用主实现，让它统一处理未对齐前缀、对齐主体和尾部剩余元素。
        vectorize_read_with_alignment<VEC_SIZE>(
            in,
            len,
            tid,
            stride,
            Vec{scalar_op},
            std::forward<ScaOp>(scalar_op));
    }
} // namespace vllm
