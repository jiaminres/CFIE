#include "common.cuh"
#include "dispatch_utils.h"
#include "cub_helpers.h"
#include "quantization/vectorization_utils.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Exceptions.h>
#include <tuple>

namespace vllm {
    // ------------------------------- 静态 scale FP8 量化 kernel -------------------------------

    // `STRIDE_I_ZERO` 表示 scale 在 token/group_m 方向不移动，典型场景是 per-tensor 或 per-channel。
    // `STRIDE_J_ZERO` 表示 scale 在 hidden/group_n 方向不移动，典型场景是 per-tensor 或 per-token。
    template<typename scalar_t, typename fp8_type, bool STRIDE_I_ZERO,
        bool STRIDE_J_ZERO>
    __global__ void scaled_fp8_quant_kernel_strided_group_shape(
        fp8_type * __restrict__ out, const scalar_t * __restrict__ input,
        const float * __restrict__ scale, int hidden_size, int64_t in_row_stride,
        int64_t out_row_stride, int group_m, int group_n, int64_t scale_stride_i,
        int64_t scale_stride_j) {
        // 一个 CUDA block 负责一行 token，因此 `blockIdx.x` 对应展平后的 M 维行号。
        const int64_t token_idx = blockIdx.x;

        // 当前线程在线程块内的编号，用于在 hidden 维 `[N]` 上做 stride 遍历。
        const int tid = threadIdx.x;

        // 根据输入行 stride 定位当前 token 的输入起始地址 `input[token_idx, :]`。
        const scalar_t *token_in = input + token_idx * in_row_stride;

        // 根据输出行 stride 定位当前 token 的输出起始地址 `out[token_idx, :]`。
        fp8_type *token_out = out + token_idx * out_row_stride;

        // 预计算当前 token 行对应的 scale 行基址；若 token 方向 stride 为 0，编译期会消去该分支。
        const int64_t scale_row_base =
                STRIDE_I_ZERO
                    ? 0
                    : static_cast<int>(token_idx) / group_m * scale_stride_i;

        // 根据 hidden group 编号读取 scale 并取倒数，供 `val * inv_scale` 量化路径使用。
        auto get_inv_scale = [&](int gj) {
            // `gj` 是 hidden 维 group 编号，对应 scale 的列向分组位置。
            return 1.0f / scale[scale_row_base + gj * scale_stride_j];
        };

        // 缓存最近一次访问的 hidden group 编号，降低小 group 标量路径中的重复 scale 读取。
        int cached_gj = -1;

        // 缓存最近一次访问的 scale 倒数，配合 `cached_gj` 复用。
        float cached_inv_scale = 0.0f;

        // 在标量路径中按需更新 scale 缓存，避免每个 hidden 元素都访问全局 scale。
        auto get_inv_scale_cached = [&](int gj) {
            // 只有当前 hidden 元素进入新的 group 时才重新读取 scale。
            if (gj != cached_gj) {
                // 读取当前 hidden group 的 scale 并转换为倒数。
                cached_inv_scale = 1.0f / scale[scale_row_base + gj * scale_stride_j];

                // 记录已缓存的 group 编号，供后续相同 group 元素复用。
                cached_gj = gj;
            }

            // 返回当前 hidden group 对应的 scale 倒数。
            return cached_inv_scale;
        };

        // FP8 每个元素 1 byte，因此 16 个元素刚好组成 128-bit 向量化访问粒度。
        constexpr int VEC_SIZE = 16;

        // 对连续片段执行向量化静态 scale 量化：输入 `[size]` 写入输出 `[size]`。
        auto scaled_fp8_conversion_vectorized = [&](const scalar_t *in, fp8_type *out,
                                                    int size, float inv_scale) {
            // 使用对齐感知的向量化 helper，让每个线程处理当前片段中的若干元素。
            vectorize_with_alignment<VEC_SIZE>(
                in, out, size, tid, blockDim.x,
                [=] __device__(fp8_type &dst, const scalar_t &src) {
                    // 将源值转成 float 后乘以 scale 倒数，并裁剪/转换到目标 FP8 类型。
                    dst = scaled_fp8_conversion<true, fp8_type>(static_cast<float>(src),
                                                                inv_scale);
                });
        };

        // ------------------------------- 选择静态量化访问路径 -------------------------------

        // hidden 方向 scale stride 为 0 且整行可 128-bit 对齐时，整行共享一个 scale 并走全行向量化。
        if (STRIDE_J_ZERO && hidden_size % VEC_SIZE == 0) {
            // 对 per-tensor 或 per-token 场景，`gj=0` 即当前行唯一 scale。
            scaled_fp8_conversion_vectorized(token_in, token_out, hidden_size,
                                             get_inv_scale(0));
        } else if (group_n % VEC_SIZE == 0) {
            // hidden group 大小满足 128-bit 对齐时，逐 group 做向量化量化。
            const int num_groups_n = hidden_size / group_n;

            // 遍历当前 token 行上的所有 hidden 维 group。
            for (int gj = 0; gj < num_groups_n; gj++) {
                // 对当前 hidden group `[gj * group_n, (gj + 1) * group_n)` 使用对应 scale。
                scaled_fp8_conversion_vectorized(token_in + gj * group_n,
                                                 token_out + gj * group_n, group_n,
                                                 get_inv_scale(gj));
            }
        } else {
            // hidden group 小于向量化粒度时走标量路径，避免跨 group 错用 scale。
            for (int n = tid; n < hidden_size; n += blockDim.x) {
                // 当前 hidden 元素所属的列向 group 编号。
                const int gj = n / group_n;

                // 对单个元素应用该 group 的 scale 倒数并写入 `token_out[n]`。
                token_out[n] = scaled_fp8_conversion<true, fp8_type>(
                    static_cast<float>(token_in[n]), get_inv_scale_cached(gj));
            }
        }
    }

    // ------------------------------- per-tensor 动态 scale 归约 kernel -------------------------------

    template<typename scalar_t, typename fp8_type>
    __global__ void segmented_max_reduction_strided(
        float * __restrict__ scale, const scalar_t * __restrict__ input,
        int hidden_size, int64_t in_row_stride, int64_t num_tokens) {
        // 每个 block 内使用共享内存保存线程局部 absmax，最终归约成当前 token 行的 absmax。
        __shared__ float cache[256];

        // 当前线程编号用于 stride 扫描 hidden 维 `[N]`。
        const int tid = threadIdx.x;

        // 一个 block 对应一个 token 行，`token_idx` 是展平后的 M 维行号。
        int64_t token_idx = blockIdx.x;

        // grid 可能被外部按更大范围启动，这里保护越界 token 行。
        if (token_idx >= num_tokens) {
            // 越界 block 不参与全局 scale 归约。
            return;
        }

        // 定位当前 token 行输入起始地址 `input[token_idx, :]`。
        const scalar_t *row_ptr = input + token_idx * in_row_stride;

        // 初始化当前线程负责元素的局部绝对值最大值。
        float thread_max = 0.0f;

        // 当前线程按 `blockDim.x` stride 扫描 hidden 维，覆盖当前 token 行的若干列。
        for (int e = tid; e < hidden_size; e += blockDim.x) {
            // 将输入元素转成 float 后取绝对值，作为 FP8 动态 scale 的候选幅度。
            float v = fabsf(static_cast<float>(row_ptr[e]));

            // 更新当前线程的局部最大绝对值。
            thread_max = fmaxf(thread_max, v);
        }

        // 把线程局部最大值写入共享内存，供 block 内归约使用。
        cache[tid] = thread_max;

        // 等待所有线程完成共享内存写入。
        __syncthreads();

        // 使用二分归约把当前 token 行的最大绝对值收敛到 `cache[0]`。
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            // 前半线程读取后半线程结果并更新本位置最大值。
            if (tid < offset) {
                // 合并两个归约片段的最大绝对值。
                cache[tid] = fmaxf(cache[tid], cache[tid + offset]);
            }

            // 每轮归约后同步，确保下一轮读到的是已更新值。
            __syncthreads();
        }

        // 每个 token block 的 0 号线程把当前行 absmax 合并到全局 per-tensor scale。
        if (tid == 0) {
            // 全局 scale 存储的是 `absmax / qmax`，所有 token 行通过 atomic max 取最大值。
            atomicMaxFloat(scale, cache[0] / quant_type_max_value<fp8_type>());
        }
    }

    // ------------------------------- per-tensor 动态 scale FP8 量化 kernel -------------------------------

    template<typename scalar_t, typename fp8_type>
    __global__ void scaled_fp8_quant_kernel_strided_dynamic(
        fp8_type * __restrict__ out, const scalar_t * __restrict__ input,
        const float * __restrict__ scale, int hidden_size, int64_t in_row_stride,
        int64_t out_row_stride) {
        // 一个 block 负责一个 token 行。
        const int64_t token_idx = blockIdx.x;

        // 当前线程编号用于在 hidden 维上并行处理元素。
        const int tid = threadIdx.x;

        // 定位当前 token 的输入行起始地址。
        const scalar_t *token_in = input + token_idx * in_row_stride;

        // 定位当前 token 的输出行起始地址。
        fp8_type *token_out = out + token_idx * out_row_stride;

        // per-tensor 动态量化全张量共享 `scale[0]`，kernel 内取倒数以复用乘法路径。
        const float reciprocal_scale = 1.0f / (*scale);

        // 使用 16 个 FP8 元素为粒度的向量化路径量化当前 token 行。
        vectorize_with_alignment<16>(
            token_in, token_out, hidden_size, tid, blockDim.x,
            [=] __device__(fp8_type &dst, const scalar_t &src) {
                // 将输入值转为 float 后乘以全局 scale 倒数，并写成 FP8。
                dst = scaled_fp8_conversion<true, fp8_type>(static_cast<float>(src),
                                                            reciprocal_scale);
            });
    }

    // ------------------------------- per-token 动态 scale FP8 量化 kernel -------------------------------

    template<typename scalar_t, typename fp8_type>
    __global__ void dynamic_per_token_scaled_fp8_quant_kernel_strided(
        fp8_type * __restrict__ out,
        float * __restrict__ scale,
        const scalar_t * __restrict__ input,
        const float * __restrict__ scale_ub,
        int hidden_size,
        int64_t in_row_stride,
        int64_t out_row_stride
    ) {
        // 一个 block 负责一个 token 行。
        const int64_t token_idx = blockIdx.x;

        // 当前线程编号用于扫描当前 token 行的 hidden 维。
        const int tid = threadIdx.x;

        // 使用 int64 计算输入偏移，避免长序列或大 hidden 情况下 int32 溢出。
        int64_t in_offset = static_cast<int64_t>(token_idx) * in_row_stride;

        // 使用 int64 计算输出偏移，保持与输入 stride 计算口径一致。
        int64_t out_offset = static_cast<int64_t>(token_idx) * out_row_stride;

        // 当前 token 行输入指针，逻辑形状为 `input[token_idx, :]`。
        const scalar_t *token_in = input + in_offset;

        // 当前 token 行输出指针，逻辑形状为 `out[token_idx, :]`。
        fp8_type *token_out = out + out_offset;

        // ------------------------------- 计算当前 token 的动态 scale -------------------------------

        // 初始化当前线程扫描片段的局部最大绝对值。
        float absmax_val = 0.f;

        // 向量化读取当前 token 行，并在当前线程内累积局部 absmax。
        vectorize_read_with_alignment<16>(
            token_in,
            hidden_size,
            tid,
            blockDim.x,
            [&] __device__(scalar_t v) {
                // 将输入值转成 float 后取绝对值，用于动态 per-token scale。
                absmax_val = fmaxf(absmax_val, fabsf(static_cast<float>(v)));
            }
        );

        // 使用 CUB block reduce 在当前 token block 内归约所有线程的 absmax。
        using BlockReduce = cub::BlockReduce<float, 256>;

        // 为 CUB block reduce 分配共享内存临时区。
        __shared__ typename BlockReduce::TempStorage tmp;

        // 得到当前 token 行的最大绝对值 `block_max`。
        const float block_max =
                BlockReduce(tmp).Reduce(absmax_val, CubMaxOp{}, blockDim.x);

        // 共享内存保存当前 token 的最终 scale，供同一 block 内所有线程量化复用。
        __shared__ float token_scale;

        // 只有 0 号线程负责把归约结果转换成 scale 并写回全局 `scale[token_idx]`。
        if (tid == 0) {
            // 当前 FP8 类型可表示的最大幅度，用于把 absmax 归一化成 scale。
            const float qmax = quant_type_max_value<fp8_type>();

            // 如果调用方提供上界，则先限制当前 token 的 absmax，避免 scale 过大。
            token_scale = scale_ub ? fminf(block_max, *scale_ub) : block_max;

            // 将 absmax 转换为 scale，并用最小 scale 下界防止除零或过小 scale。
            token_scale =
                    fmaxf(token_scale / qmax, min_scaling_factor<fp8_type>::val());

            // 把当前 token 的 scale 写回 `scale: [M, 1]` 的扁平存储。
            scale[token_idx] = token_scale;
        }

        // 等待 0 号线程完成 `token_scale` 写入后，其他线程才能开始量化。
        __syncthreads();

        // ------------------------------- 使用当前 token scale 执行量化 -------------------------------

        // 向量化量化当前 token 行，输入 `[N]` 写入输出 `[N]`。
        vectorize_with_alignment<16>(
            token_in,
            token_out,
            hidden_size,
            tid,
            blockDim.x,
            [=] __device__(fp8_type &dst, const scalar_t &src) {
                // per-token kernel 传入的是正向 scale，转换函数内部执行 `val / token_scale`。
                dst = scaled_fp8_conversion<false, fp8_type>(static_cast<float>(src),
                                                             token_scale);
            });
    }
} // namespace vllm

// ------------------------------- 静态 scale FP8 量化入口 -------------------------------

void static_scaled_fp8_quant(
    torch::Tensor &out, // 输出 FP8 张量，逻辑形状 `[..., d]`。
    torch::Tensor const &input, // 输入浮点张量，逻辑形状 `[..., d]`。
    torch::Tensor const &scale, // 静态 scale，支持 0D、1D 或 2D。
    std::optional<std::tuple<int64_t, int64_t> >
    opt_group_shape) // 可选显式 group 形状 `(group_m, group_n)`。
{
    // ------------------------------- 校验输入输出基础布局 -------------------------------

    // 输入最后一维必须连续，保证每个 token 行的 hidden 维可被 kernel 线性扫描。
    TORCH_CHECK(input.stride(-1) == 1,
                "last dimension of input must be contiguous");

    // 输出最后一维必须连续，保证 FP8 结果按 hidden 维顺序写回。
    TORCH_CHECK(out.stride(-1) == 1,
                "last dimension of output must be contiguous");

    // `hidden_size` 是展平二维视角下的 N 维，即每个 token 行的 hidden 宽度。
    const int hidden_size = input.size(-1);

    // `num_tokens` 是展平二维视角下的 M 维，即除最后一维外的总行数。
    const int num_tokens = input.numel() / hidden_size;

    // ------------------------------- 解析静态 scale 的 group 布局 -------------------------------

    // `group_m/group_n` 分别表示一个 scale 覆盖多少 token 行和多少 hidden 列。
    int group_m, group_n;

    // `scale_stride_i/scale_stride_j` 分别描述 scale 在行向 group 和列向 group 上的存储步长。
    int64_t scale_stride_i, scale_stride_j;

    // 0D 或单元素 scale 表示全输入 `[M, N]` 共享一个 per-tensor scale。
    if (scale.dim() == 0 || scale.numel() == 1) {
        // per-tensor scale 覆盖所有 token 行。
        group_m = num_tokens;

        // per-tensor scale 覆盖整条 hidden 维。
        group_n = hidden_size;

        // scale 在 token group 方向不移动。
        scale_stride_i = 0;

        // scale 在 hidden group 方向不移动。
        scale_stride_j = 0;
    } else if (scale.dim() == 1) {
        // 1D scale 必须显式提供 group_shape，避免 `M == N` 时无法区分 per-token 与 per-channel。
        TORCH_CHECK(opt_group_shape.has_value(),
                    "1D scale requires explicit group_shape to disambiguate "
                    "per-channel vs per-token quantization. "
                    "Use group_shape=(-1, 1) for per-channel or group_shape=(1, "
                    "-1) for per-token.");

        // 读取调用方给出的 `(group_m, group_n)`，其中 -1 表示覆盖该维全长。
        const auto &[opt_group_m, opt_group_n] = opt_group_shape.value();

        // 解析 token 方向 group 大小，-1 表示一个 group 覆盖所有 token 行。
        group_m = opt_group_m == -1 ? num_tokens : static_cast<int>(opt_group_m);

        // 解析 hidden 方向 group 大小，-1 表示一个 group 覆盖整条 hidden 维。
        group_n = opt_group_n == -1 ? hidden_size : static_cast<int>(opt_group_n);

        // 记录 1D scale 的实际元素数，用于校验 group_shape 是否匹配输入形状。
        const int64_t scale_len = scale.numel();

        // 根据 group_m 计算 token 方向应该产生多少个 scale group。
        const int64_t expected_scale_m = num_tokens / group_m;

        // 根据 group_n 计算 hidden 方向应该产生多少个 scale group。
        const int64_t expected_scale_n = hidden_size / group_n;

        // 1D scale 的期望元素数等于二维 group 网格展平后的大小。
        const int64_t expected_scale_numel = expected_scale_m * expected_scale_n;

        // 校验调用方传入的 1D scale 长度是否与显式 group_shape 推导结果一致。
        TORCH_CHECK(scale_len == expected_scale_numel, "1D scale length (",
                    scale_len, ") does not match expected size (",
                    expected_scale_numel, ") for group_shape (", opt_group_m, ", ",
                    opt_group_n, ") with input shape (", num_tokens, ", ",
                    hidden_size, ")");

        // token 方向只有一个 scale group 时，1D scale 只能沿 hidden group 变化，即 per-channel。
        if (expected_scale_m == 1) {
            // per-channel 语义下行向 group 固定为 0。
            scale_stride_i = 0;

            // per-channel 语义下列向 group 沿 1D scale 连续前进。
            scale_stride_j = scale.stride(0);
        } else if (expected_scale_n == 1) {
            // hidden 方向只有一个 scale group 时，1D scale 只能沿 token group 变化，即 per-token。
            scale_stride_i = scale.stride(0);

            // per-token 语义下列向 group 固定为 0。
            scale_stride_j = 0;
        } else {
            // 1D scale 不能同时表达 token 与 hidden 两个方向都变化的二维 group 网格。
            TORCH_CHECK(
                false,
                "1D scale can only be used when one of the scale dimensions is 1. "
                "For 2D group scaling, use a 2D scale tensor.");
        }
    } else if (scale.dim() == 2) {
        // 2D scale 直接表达 `[M/group_m, N/group_n]` 的二维 group 网格。
        const int64_t scale_size_0 = scale.size(0);

        // 2D scale 的第二维对应 hidden 方向 group 数。
        const int64_t scale_size_1 = scale.size(1);

        // token 数必须能被 scale 第一维整除，才能推导每个 scale 覆盖多少 token 行。
        TORCH_CHECK(num_tokens % scale_size_0 == 0, "num_tokens (", num_tokens,
                    ") must be divisible by scale.size(0) (", scale_size_0, ")");

        // hidden 宽度必须能被 scale 第二维整除，才能推导每个 scale 覆盖多少 hidden 列。
        TORCH_CHECK(hidden_size % scale_size_1 == 0, "hidden_size (", hidden_size,
                    ") must be divisible by scale.size(1) (", scale_size_1, ")");

        // 从 2D scale 第一维推导 token 方向 group 大小。
        int inferred_group_m = num_tokens / scale_size_0;

        // 从 2D scale 第二维推导 hidden 方向 group 大小。
        int inferred_group_n = hidden_size / scale_size_1;

        // 若调用方显式提供 group_shape，则必须和 2D scale 形状推导结果一致。
        if (opt_group_shape.has_value()) {
            // 读取显式 group_shape，-1 仍表示覆盖该维全长。
            const auto &[opt_group_m, opt_group_n] = opt_group_shape.value();

            // 解析显式 token 方向 group 大小。
            group_m = opt_group_m == -1 ? num_tokens : static_cast<int>(opt_group_m);

            // 解析显式 hidden 方向 group 大小。
            group_n = opt_group_n == -1 ? hidden_size : static_cast<int>(opt_group_n);

            // 防止调用方传入的 group_shape 与 scale 二维形状不一致。
            TORCH_CHECK(group_m == inferred_group_m && group_n == inferred_group_n,
                        "Explicit group_shape (", opt_group_m, ", ", opt_group_n,
                        ") does not match inferred group shape (", inferred_group_m,
                        ", ", inferred_group_n, ") from 2D scale tensor shape (",
                        scale_size_0, ", ", scale_size_1, ")");
        } else {
            // 未显式提供时，直接采用 2D scale 形状推导出的 token group 大小。
            group_m = inferred_group_m;

            // 未显式提供时，直接采用 2D scale 形状推导出的 hidden group 大小。
            group_n = inferred_group_n;
        }

        // 2D scale 的第一维 stride 对应 token group 方向步长。
        scale_stride_i = scale.stride(0);

        // 2D scale 的第二维 stride 对应 hidden group 方向步长。
        scale_stride_j = scale.stride(1);
    } else {
        // 当前静态 FP8 量化入口只支持 0D、1D、2D scale。
        TORCH_CHECK(false, "scale must be 0D, 1D, or 2D tensor, but got ",
                    scale.dim(), "D");
    }

    // ------------------------------- 准备 CUDA launch 参数 -------------------------------

    // 每个 token 行使用 256 线程处理 hidden 维。
    const int block_size = 256;

    // grid.x 对应展平后的 token 行数 M。
    dim3 grid(num_tokens);

    // block.x 对应每个 token 行内的并行线程数。
    dim3 block(block_size);

    // 输入行 stride 用于支持非完全 contiguous 但最后一维 contiguous 的二维视图。
    const int64_t in_row_stride = input.stride(-2);

    // 输出行 stride 用于支持带 padding 或外部预分配的输出视图。
    const int64_t out_row_stride = out.stride(-2);

    // 切换到输入张量所在 CUDA 设备，避免多 GPU 场景 launch 到错误设备。
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    // 使用当前 PyTorch CUDA stream，保持与上游算子提交顺序一致。
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ------------------------------- 分派并启动静态量化 kernel -------------------------------

    // 根据输入浮点 dtype 选择 kernel 模板实参。
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        // 根据输出 FP8 dtype 选择平台对应的 FP8 类型。
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
            // 根据 scale 行向 stride 是否为 0 生成编译期 bool，消除不必要地址计算。
            VLLM_DISPATCH_BOOL(scale_stride_i == 0, S0_ZERO, [&] {
                // 根据 scale 列向 stride 是否为 0 生成编译期 bool，优化 per-token/per-tensor 路径。
                VLLM_DISPATCH_BOOL(scale_stride_j == 0, S1_ZERO, [&] {
                    // 启动静态 scale FP8 量化 kernel，按 token 行并行处理。
                    vllm::scaled_fp8_quant_kernel_strided_group_shape<
                    scalar_t, fp8_t, S0_ZERO, S1_ZERO>
                    <<<grid, block, 0, stream>>>(
                        out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                        scale.data_ptr<float>(), hidden_size, in_row_stride,
                        out_row_stride, group_m, group_n, scale_stride_i,
                        scale_stride_j);
                    });
                });
            });
        });
}

// ------------------------------- per-tensor 动态 scale FP8 量化入口 -------------------------------

void dynamic_scaled_fp8_quant(torch::Tensor &out, // 输出 FP8 张量，逻辑形状 `[..., d]`。
                              torch::Tensor const &input, // 输入浮点张量，逻辑形状 `[..., d]`。
                              torch::Tensor &scale) // 输出 per-tensor scale，形状 `[1]`。
{
    // ------------------------------- 校验输入输出基础布局 -------------------------------

    // 输入最后一维必须连续，保证 kernel 可以把每个 token 行当作连续 hidden 向量处理。
    TORCH_CHECK(input.stride(-1) == 1,
                "last dimension of input must be contiguous");

    // 输出最后一维必须连续，保证 FP8 量化结果按 hidden 维顺序写回。
    TORCH_CHECK(out.stride(-1) == 1,
                "last dimension of output must be contiguous");

    // `hidden_size` 是展平二维视角中的 N 维。
    const int hidden_size = input.size(-1);

    // `num_tokens` 是展平二维视角中的 M 维。
    const int num_tokens = input.numel() / hidden_size;

    // 每个 token 行使用 256 线程参与归约和量化。
    const int block_size = 256;

    // grid.x 对应 token 行数。
    dim3 grid(num_tokens);

    // block.x 对应每行内部并行线程数。
    dim3 block(block_size);

    // 输入行 stride 用于定位 `input[token_idx, :]`。
    const int64_t in_row_stride = input.stride(-2);

    // 输出行 stride 用于定位 `out[token_idx, :]`。
    const int64_t out_row_stride = out.stride(-2);

    // 切换到输入所在 CUDA 设备。
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    // 获取当前 PyTorch CUDA stream，保证与上游操作顺序一致。
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ------------------------------- 计算 per-tensor 动态 scale -------------------------------

    // 归约 kernel 通过 atomic max 写 scale，因此启动前必须把 `scale[0]` 清零。
    AT_CUDA_CHECK(
        cudaMemsetAsync(scale.data_ptr<float>(), 0, sizeof(float), stream));

    // 根据输入浮点 dtype 选择归约与量化 kernel 的模板实参。
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "scaled_fp8_quant_kernel_scalar_type", [&] {
        // 根据输出 FP8 dtype 选择平台对应的 FP8 类型。
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "scaled_fp8_quant_kernel_fp8_type", [&] {
            // 第一阶段：每个 token 行归约 absmax，并通过 atomic max 生成全局 per-tensor scale。
            vllm::segmented_max_reduction_strided<scalar_t, fp8_t>
            <<<grid, block, 0, stream>>>(
                scale.data_ptr<float>(), input.data_ptr<scalar_t>(),
                hidden_size, in_row_stride,
                static_cast<int64_t>(num_tokens));

            // 第二阶段：使用刚生成的 `scale[0]` 对全部 token 行执行 FP8 量化。
            vllm::scaled_fp8_quant_kernel_strided_dynamic<scalar_t, fp8_t>
            <<<grid, block, 0, stream>>>(
                out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                scale.data_ptr<float>(), hidden_size, in_row_stride,
                out_row_stride);
            });
        });
}

// ------------------------------- per-token 动态 scale FP8 量化入口 -------------------------------

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor &out, // 输出 FP8 张量，逻辑形状 `[..., d]`。
    torch::Tensor const &input, // 输入浮点张量，逻辑形状 `[..., d]`。
    torch::Tensor &scales, // 输出 per-token scale，逻辑形状 `[M, 1]`。
    std::optional<at::Tensor> const &scale_ub) {
    // ------------------------------- 校验输入输出基础布局 -------------------------------

    // 输入最后一维必须连续，保证每个 token 行的 hidden 维可线性扫描。
    TORCH_CHECK(input.stride(-1) == 1,
                "last dimension of input must be contiguous");

    // 输出最后一维必须连续，保证量化结果写回时不需要复杂 gather/scatter。
    TORCH_CHECK(out.stride(-1) == 1,
                "last dimension of output must be contiguous");

    // `hidden_size` 是展平二维视角下的 N 维。
    const int hidden_size = input.size(-1);

    // `num_tokens` 是展平二维视角下的 M 维。
    const int num_tokens = input.numel() / hidden_size;

    // per-token kernel 最多使用 256 线程处理一行。
    const int block_size = 256;

    // grid.x 对应 token 行数，每个 block 计算一个 token 的 scale 并量化该行。
    dim3 grid(num_tokens);

    // block.x 不超过 hidden_size，避免 hidden 很小时启动无意义线程。
    dim3 block(std::min(hidden_size, block_size));

    // 输入行 stride 用于支持最后一维 contiguous 的非完全 contiguous 视图。
    const int64_t in_row_stride = input.stride(-2);

    // 输出行 stride 用于支持 padding 后输出或外部预分配输出。
    const int64_t out_row_stride = out.stride(-2);

    // 切换到输入所在 CUDA 设备。
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    // 使用当前 PyTorch CUDA stream 提交 kernel。
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ------------------------------- 分派并启动 per-token 动态量化 kernel -------------------------------

    /*
     * 下面两层 dispatch 宏的作用，是把运行时 dtype 枚举转换成编译期 C++ 类型别名。
     *
     * `input.scalar_type()` 返回的是运行时枚举，例如 `at::ScalarType::Half`。
     * `scalar_t` 是宏命中某个 case 后声明出来的编译期类型别名，用于：
     *   - 实例化 CUDA 模板 kernel。
     *   - 正确解释 `input.data_ptr<scalar_t>()` 指向的底层元素。
     *
     * `out.scalar_type()` 同样是运行时枚举，例如 `at::ScalarType::Float8_e4m3fn`。
     * `fp8_t` 是内层 FP8 dispatch 声明出来的编译期类型别名，用于：
     *   - 实例化输出 FP8 kernel 路径。
     *   - 正确解释 `out.data_ptr<fp8_t>()` 指向的底层元素。
     *
     * 在 CUDA 非 ROCm 路径下，宏展开后的主要逻辑可近似理解为：
     *
     * switch (input.scalar_type()) {
     *   case at::ScalarType::Float: {
     *     using scalar_t = float;
     *
     *     switch (out.scalar_type()) {
     *       case at::ScalarType::Float8_e4m3fn: {
     *         using fp8_t = c10::Float8_e4m3fn;
     *
     *         vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<
     *             scalar_t, fp8_t><<<grid, block, 0, stream>>>(
     *             out.data_ptr<fp8_t>(),
     *             scales.data_ptr<float>(),
     *             input.data_ptr<scalar_t>(),
     *             scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
     *             hidden_size,
     *             in_row_stride,
     *             out_row_stride);
     *         break;
     *       }
     *       default:
     *         TORCH_CHECK(false, "unsupported FP8 output dtype");
     *     }
     *     break;
     *   }
     *
     *   case at::ScalarType::Half: {
     *     using scalar_t = at::Half;
     *
     *     switch (out.scalar_type()) {
     *       case at::ScalarType::Float8_e4m3fn: {
     *         using fp8_t = c10::Float8_e4m3fn;
     *         vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<
     *             scalar_t, fp8_t><<<grid, block, 0, stream>>>(...);
     *         break;
     *       }
     *       default:
     *         TORCH_CHECK(false, "unsupported FP8 output dtype");
     *     }
     *     break;
     *   }
     *
     *   case at::ScalarType::BFloat16: {
     *     using scalar_t = at::BFloat16;
     *
     *     switch (out.scalar_type()) {
     *       case at::ScalarType::Float8_e4m3fn: {
     *         using fp8_t = c10::Float8_e4m3fn;
     *         vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<
     *             scalar_t, fp8_t><<<grid, block, 0, stream>>>(...);
     *         break;
     *       }
     *       default:
     *         TORCH_CHECK(false, "unsupported FP8 output dtype");
     *     }
     *     break;
     *   }
     *
     *   default:
     *     TORCH_CHECK(false, "unsupported input dtype");
     * }
     *
     * 关键点：`at::ScalarType::Half` 是运行时 dtype 标签；`scalar_t` 才是模板实参。
     * 因此不能把 `input.scalar_type()` 直接写进 `kernel<...>`，必须先通过 dispatch 转成类型。
     */
    VLLM_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "dynamic_per_token_scaled_fp8_quant_kernel_scalar_type",
        [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(),
            "dynamic_per_token_scaled_fp8_quant_kernel_fp8_type",
            [&] {
            vllm::dynamic_per_token_scaled_fp8_quant_kernel_strided<
            scalar_t, fp8_t><<<grid, block, 0, stream>>>(
                out.data_ptr<fp8_t>(),
                scales.data_ptr<float>(),
                input.data_ptr<scalar_t>(),
                scale_ub.has_value() ? scale_ub->data_ptr<float>()
                : nullptr,
                hidden_size,
                in_row_stride,
                out_row_stride);
            });
        });
}
