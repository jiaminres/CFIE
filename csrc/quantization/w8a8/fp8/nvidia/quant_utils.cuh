#pragma once

#include "../../../../attention/attention_dtypes.h"
#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <type_traits>

namespace vllm {
#ifndef USE_ROCM

    namespace fp8 {
        // 本文件只为 NVIDIA CUDA 路径提供 FP8 KV cache 转换工具。
        // ROCm 使用独立实现，因此不会实例化这里的 CUDA FP8 intrinsic 逻辑。
#ifdef ENABLE_FP8

        // ------------------------------- 定义未缩放 FP8 转换模板 -------------------------------
        // 默认模板保留原值，具体 dtype 与向量宽度由下面的特化实现。
        template<typename Tout, typename Tin>
        __inline__ __device__ Tout vec_conversion(
            const Tin &x, const __nv_fp8_interpretation_t fp8_type = __NV_E4M3) {
            // 未命中特化时保持输入不变，避免普通 dtype 路径额外改写位模式。
            return x;
        }

        // 将标量 float 转成 PyTorch 使用的 E4M3 FP8 包装类型。
        template<>
        __inline__ __device__ c10::Float8_e4m3fn vec_conversion<c10::Float8_e4m3fn, float>(
            const float &a,
            const __nv_fp8_interpretation_t fp8_type
        ) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
            // SM80 之前没有 CUDA FP8 intrinsic，退回 PyTorch 包装类型的静态转换。
            return static_cast<c10::Float8_e4m3fn>(a);
#else
            // SM80+ 使用 CUDA intrinsic 生成 FP8 bit，再按 PyTorch wrapper 语义封装。
            return c10::Float8_e4m3fn(__nv_cvt_float_to_fp8(a, __NV_SATFINITE, fp8_type),
                                      c10::Float8_e4m3fn::from_bits());
#endif
        }

#if 0  // 为降低二进制体积，保留但不编译未缩放转换路径。
        // 下面的未缩放特化只作为参考路径，当前 kernel 实际使用缩放转换。
        // FP8 标量解码成 half 标量。
        template<>
        __inline__ __device__ uint16_t vec_conversion<uint16_t, uint8_t>(
            const uint8_t &a, const __nv_fp8_interpretation_t fp8_type) {
            // 调用 CUDA intrinsic 把 1 个 FP8 lane 解码成 half 原始位。
            __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
            // 返回 half 的 16 bit 存储，供后续向量打包继续复用。
            return res.x;
        }

        // 两个打包 FP8 lane 解码成 half2。
        template<>
        __inline__ __device__ uint32_t vec_conversion<uint32_t, uint16_t>(
            const uint16_t &a, const __nv_fp8_interpretation_t fp8_type) {
            // 使用 union 把两个 half lane 直接按 uint32_t 位模式返回。
            union {
                uint16_t u16[2];
                uint32_t u32;
            } tmp;
            // 一次性把 2 个 FP8 lane 解码成 2 个 half lane。
            __half2_raw res = __nv_cvt_fp8x2_to_halfraw2(a, fp8_type);
            // 低 16 bit 保存第 0 个 half lane。
            tmp.u16[0] = res.x;
            // 高 16 bit 保存第 1 个 half lane。
            tmp.u16[1] = res.y;
            // 返回打包后的 half2 位模式。
            return tmp.u32;
        }

        // 四个打包 FP8 lane 解码成两个 half2。
        template<>
        __inline__ __device__ uint2 vec_conversion<uint2, uint32_t>(
            const uint32_t &a, const __nv_fp8_interpretation_t fp8_type) {
            // 使用 uint2 承载两个 half2，保持后续向量化访问对齐。
            union {
                uint2 u32x2;
                uint32_t u32[2];
            } tmp;
            // 低 16 bit 包含前两个 FP8 lane，先解码成第一个 half2。
            tmp.u32[0] = vec_conversion<uint32_t, uint16_t>((uint16_t) a, fp8_type);
            // 高 16 bit 包含后两个 FP8 lane，解码成第二个 half2。
            tmp.u32[1] =
                    vec_conversion<uint32_t, uint16_t>((uint16_t) (a >> 16U), fp8_type);
            // 返回两个 half2 组成的向量位模式。
            return tmp.u32x2;
        }

        // 八个打包 FP8 lane 解码成四个 half2。
        template<>
        __inline__ __device__ uint4 vec_conversion<uint4, uint2>(
            const uint2 &a, const __nv_fp8_interpretation_t fp8_type) {
            // 使用 uint4 承载四个 half2，匹配 8 lane 解码后的 128 bit 输出。
            union {
                uint4 u64x2;
                uint2 u64[2];
            } tmp;
            // a.x 存放前 4 个 FP8 lane，解码成前两个 half2。
            tmp.u64[0] = vec_conversion<uint2, uint32_t>(a.x, fp8_type);
            // a.y 存放后 4 个 FP8 lane，解码成后两个 half2。
            tmp.u64[1] = vec_conversion<uint2, uint32_t>(a.y, fp8_type);
            // 返回 8 个 half lane 的打包位模式。
            return tmp.u64x2;
        }

        // FP8 标量解码成 BF16 标量。
        template<>
        __inline__ __device__ __nv_bfloat16 vec_conversion<__nv_bfloat16, uint8_t>(
            const uint8_t &a, const __nv_fp8_interpretation_t fp8_type) {
            // CUDA 没有 FP8 到 BF16 的直接 intrinsic，先转成 half 原始位。
            __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
            // half 先提升到 float，避免手写 BF16 round 逻辑。
            float tmp = half_to_float(res.x);
            // 最后由 CUDA BF16 intrinsic 完成 float 到 BF16 的舍入。
            return __float2bfloat16(tmp);
        }

        // 两个打包 FP8 lane 解码成 BF16x2。
        template<>
        __inline__ __device__ __nv_bfloat162 vec_conversion<__nv_bfloat162, uint16_t>(
            const uint16_t &a, const __nv_fp8_interpretation_t fp8_type) {
            // 准备 BF16x2 输出，两个 lane 分别从低/高 8 bit 解码。
            __nv_bfloat162 res;
            // 低 8 bit 对应第 0 个 FP8 lane。
            res.x = vec_conversion<__nv_bfloat16, uint8_t>((uint8_t) a, fp8_type);
            // 高 8 bit 对应第 1 个 FP8 lane。
            res.y = vec_conversion<__nv_bfloat16, uint8_t>((uint8_t) (a >> 8U), fp8_type);
            // 返回两个 BF16 lane 的组合结果。
            return res;
        }

        // 四个打包 FP8 lane 解码成 bf16_4_t。
        template<>
        __inline__ __device__ bf16_4_t vec_conversion<bf16_4_t, uint32_t>(
            const uint32_t &a, const __nv_fp8_interpretation_t fp8_type) {
            // 准备 4 lane BF16 输出，内部按两个 BF16x2 组织。
            bf16_4_t res;
            // 低 16 bit 包含前两个 FP8 lane，写入 res.x。
            res.x = vec_conversion<__nv_bfloat162, uint16_t>((uint16_t) a, fp8_type);
            // 高 16 bit 包含后两个 FP8 lane，写入 res.y。
            res.y =
                    vec_conversion<__nv_bfloat162, uint16_t>((uint16_t) (a >> 16U), fp8_type);
            // 返回 4 lane BF16 向量。
            return res;
        }

        // 八个打包 FP8 lane 解码成 bf16_8_t。
        template<>
        __inline__ __device__ bf16_8_t vec_conversion<bf16_8_t, uint2>(
            const uint2 &a, const __nv_fp8_interpretation_t fp8_type) {
            // 先用两个 4 lane 临时结果承接 a.x 与 a.y 的解码结果。
            bf16_4_t tmp1, tmp2;
            // a.x 存放前 4 个 FP8 lane。
            tmp1 = vec_conversion<bf16_4_t, uint32_t>(a.x, fp8_type);
            // a.y 存放后 4 个 FP8 lane。
            tmp2 = vec_conversion<bf16_4_t, uint32_t>(a.y, fp8_type);
            // 准备 8 lane BF16 输出结构。
            bf16_8_t res;
            // 前 4 个 BF16 lane 写入 x/y。
            res.x = tmp1.x;
            res.y = tmp1.y;
            // 后 4 个 BF16 lane 写入 z/w。
            res.z = tmp2.x;
            res.w = tmp2.y;
            // 返回完整的 8 lane BF16 向量。
            return res;
        }

        // FP8 标量解码成 float 标量。
        template<>
        __inline__ __device__ float
        vec_conversion<float, uint8_t>(const uint8_t &a,
                                       const __nv_fp8_interpretation_t fp8_type) {
            // 复用 FP8 到 half 的路径，减少重复解码逻辑。
            uint16_t tmp = vec_conversion<uint16_t, uint8_t>(a, fp8_type);
            // half 原始位提升成 float 作为未缩放输出。
            return half_to_float(tmp);
        }

        // 两个打包 FP8 lane 解码成 float2。
        template<>
        __inline__ __device__ float2 vec_conversion<float2, uint16_t>(
            const uint16_t &a, const __nv_fp8_interpretation_t fp8_type) {
            // 先解码成 half2 位模式，复用统一的 FP8x2 解码逻辑。
            uint32_t tmp = vec_conversion<uint32_t, uint16_t>(a, fp8_type);
            // half2 再提升成 float2，作为计算域输入。
            return half2_to_float2(tmp);
        }

        // 四个打包 FP8 lane 解码成 Float4_。
        template<>
        __inline__ __device__ Float4_ vec_conversion<Float4_, uint32_t>(
            const uint32_t &a, const __nv_fp8_interpretation_t fp8_type) {
            // Float4_ 用两个 float2 表达 4 lane 结果。
            Float4_ res;
            // 低 16 bit 解码成前两个 float lane。
            res.x = vec_conversion<float2, uint16_t>((uint16_t) a, fp8_type);
            // 高 16 bit 解码成后两个 float lane。
            res.y = vec_conversion<float2, uint16_t>((uint16_t) (a >> 16U), fp8_type);
            // 返回 4 lane float 向量。
            return res;
        }

        // 八个打包 FP8 lane 解码成 Float8_。
        template<>
        __inline__ __device__ Float8_ vec_conversion<Float8_, uint2>(
            const uint2 &a, const __nv_fp8_interpretation_t fp8_type) {
            // 先用两个 4 lane 临时结果承接 a.x 与 a.y 的解码结果。
            Float4_ tmp1, tmp2;
            // a.x 存放前 4 个 FP8 lane。
            tmp1 = vec_conversion<Float4_, uint32_t>(a.x, fp8_type);
            // a.y 存放后 4 个 FP8 lane。
            tmp2 = vec_conversion<Float4_, uint32_t>(a.y, fp8_type);
            // 准备 8 lane float 输出结构。
            Float8_ res;
            // 前 4 个 float lane 写入 x/y。
            res.x = tmp1.x;
            res.y = tmp1.y;
            // 后 4 个 float lane 写入 z/w。
            res.z = tmp2.x;
            res.w = tmp2.y;
            // 返回完整的 8 lane float 向量。
            return res;
        }

        // half 标量编码成 FP8 标量。
        template<>
        __inline__ __device__ uint8_t vec_conversion<uint8_t, uint16_t>(
            const uint16_t &a, const __nv_fp8_interpretation_t fp8_type) {
            // 把 half 的 16 bit 位模式放入 CUDA half_raw 包装。
            __half_raw tmp;
            // a 本身已经是 half 原始位，直接赋给 half_raw.x。
            tmp.x = a;
            // 由 CUDA intrinsic 执行饱和舍入并生成 FP8 存储位。
            __nv_fp8_storage_t res =
                    __nv_cvt_halfraw_to_fp8(tmp, __NV_SATFINITE, fp8_type);
            // 返回 FP8 的 8 bit 存储。
            return (uint8_t) res;
        }

        // BF16 标量编码成 FP8 标量。
        template<>
        __inline__ __device__ uint8_t vec_conversion<uint8_t, __nv_bfloat16>(
            const __nv_bfloat16 &a, const __nv_fp8_interpretation_t fp8_type) {


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
        // SM80 之前不支持 BF16 到 FP8 的 CUDA intrinsic。
        assert(false);
#else
        // SM80+ 直接把 BF16 原始位按目标 FP8 格式饱和转换。
        __nv_fp8_storage_t res = __nv_cvt_bfloat16raw_to_fp8(
            __nv_bfloat16_raw(a), __NV_SATFINITE, fp8_type);
        // 返回 FP8 的 8 bit 存储。
  return (uint8_t)res;
#endif
        }

        // float 标量编码成 FP8 标量。
        template<>
        __inline__ __device__ uint8_t vec_conversion<uint8_t, float>(
            const float &a, const __nv_fp8_interpretation_t fp8_type) {
            // float 直接按目标 FP8 格式执行饱和舍入。
            __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(a, __NV_SATFINITE, fp8_type);
            // 返回 FP8 的 8 bit 存储。
            return (uint8_t) res;
        }

        // 四个打包 FP8 lane 解码成 CUDA float4。
        template<>
        __inline__ __device__ float4 vec_conversion<float4, uint32_t>(
            const uint32_t &a, const __nv_fp8_interpretation_t fp8_type) {
            // 先复用 Float4_ 解码路径得到两个 float2。
            Float4_ tmp = vec_conversion<Float4_, uint32_t>(a, fp8_type);
            // 把内部 Float4_ 展平成 CUDA float4，便于 kernel 直接使用。
            float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
            // 返回 CUDA float4 结果。
            return res;
        }

        template<>
        __inline__ __device__ uint32_t vec_conversion<uint32_t, float2>(
            const float2 &a, const __nv_fp8_interpretation_t fp8_type) {
            // 使用 union 让 half2 结果能够按 uint32_t 位模式返回。
            union {
                half2 float16;
                uint32_t uint32;
            };

            // float2 按 round-to-nearest 转成 half2，作为后续打包存储格式。
            float16 = __float22half2_rn(a);
            // 返回 half2 的 32 bit 原始位。
            return uint32;
        }

        template<>
        __inline__ __device__ uint2 vec_conversion<uint2, Float4_>(
            const Float4_ &a, const __nv_fp8_interpretation_t fp8_type) {
            // 准备两个 half2 的打包输出。
            uint2 b;
            // 临时 float2 用来逐半边承接 Float4_ 的两个 float2 成员。
            float2 val;
            // 取出前两个 float lane。
            val.x = a.x.x;
            val.y = a.x.y;
            // 前两个 float lane 转成第一个 half2 位模式。
            b.x = vec_conversion<uint32_t, float2>(val, fp8_type);

            // 取出后两个 float lane。
            val.x = a.y.x;
            val.y = a.y.y;
            // 后两个 float lane 转成第二个 half2 位模式。
            b.y = vec_conversion<uint32_t, float2>(val, fp8_type);

            // 返回 4 lane half 的打包结果。
            return b;
        }

        template<>
        __inline__ __device__ float4 vec_conversion<float4, Float4_>(
            const Float4_ &a, const __nv_fp8_interpretation_t fp8_type) {
            // 准备 CUDA float4 输出，便于调用方按原生向量访问。
            float4 b;
            // 前两个 lane 来自 Float4_.x。
            b.x = a.x.x;
            b.y = a.x.y;
            // 后两个 lane 来自 Float4_.y。
            b.z = a.y.x;
            b.w = a.y.y;
            // 返回展平后的 float4。
            return b;
        }

        template<>
        __inline__ __device__ uint4 vec_conversion<uint4, Float8_>(
            const Float8_ &a, const __nv_fp8_interpretation_t fp8_type) {
            // 准备四个 half2 的打包输出。
            uint4 b;
            // 每个 Float8_ 成员都是 float2，分别转成一个 half2 位模式。
            b.x = vec_conversion<uint32_t, float2>(a.x, fp8_type);
            b.y = vec_conversion<uint32_t, float2>(a.y, fp8_type);
            b.z = vec_conversion<uint32_t, float2>(a.z, fp8_type);
            b.w = vec_conversion<uint32_t, float2>(a.w, fp8_type);
            // 返回 8 lane half 的打包结果。
            return b;
        }

        template<>
        __inline__ __device__ __nv_bfloat162 vec_conversion<__nv_bfloat162, float2>(
            const float2 &a, const __nv_fp8_interpretation_t fp8_type) {
            // 准备 BF16x2 输出。
            __nv_bfloat162 b;
            // 复用通用 float 到 BF16x2 工具完成舍入。
            from_float(b, a);
            // 返回两个 BF16 lane。
            return b;
        }

        template<>
        __inline__ __device__ bf16_4_t vec_conversion<bf16_4_t, Float4_>(
            const Float4_ &a, const __nv_fp8_interpretation_t fp8_type) {
            // 准备 4 lane BF16 输出。
            bf16_4_t b;
            // 复用通用 float 到 BF16 向量工具完成逐 lane 舍入。
            from_float(b, a);
            // 返回 4 lane BF16 结果。
            return b;
        }

        template<>
        __inline__ __device__ bf16_8_t vec_conversion<bf16_8_t, Float8_>(
            const Float8_ &a, const __nv_fp8_interpretation_t fp8_type) {
            // 准备 8 lane BF16 输出。
            bf16_8_t b;
            // 复用通用 float 到 BF16 向量工具完成逐 lane 舍入。
            from_float(b, a);
            // 返回 8 lane BF16 结果。
            return b;
        }
#endif

        // ------------------------------- 定义带 scale 的 FP8 转换模板 -------------------------------
        // scale 约定：写入 FP8 时先除以 scale，读回高精度时再乘以 scale。
        // 打包宽度约定：uint8_t 表示 1 lane，uint16_t 表示 2 lane，uint32_t 表示 4 lane，uint2 表示 8 lane。
        template<typename Tout, typename Tin>
        __inline__ __device__ Tout scaled_vec_conversion(
            const Tin &x, const float scale, const __nv_fp8_interpretation_t fp8_type) {
            // 默认模板不改变输入值，未命中特化的 dtype 维持原始位模式。
            return x;
        }

        // FP8 标量按 scale 解码成 half 标量。
        template<>
        __inline__ __device__ uint16_t scaled_vec_conversion<uint16_t, uint8_t>(
            const uint8_t &a, const float scale,
            const __nv_fp8_interpretation_t fp8_type) {
            // 先把 FP8 lane 解码成 half 原始位。
            __half_raw tmp = __nv_cvt_fp8_to_halfraw(a, fp8_type);
            // half 提升成 float 后乘回 scale，再舍入回 half 存储。
            return float_to_half(half_to_float(tmp.x) * scale);
        }

        // 两个打包 FP8 lane 按 scale 解码成 half2。
        template<>
        __inline__ __device__ uint32_t scaled_vec_conversion<uint32_t, uint16_t>(
            const uint16_t &a, const float scale,
            const __nv_fp8_interpretation_t fp8_type) {
            // 使用 union 把两个 half lane 重新打包成 uint32_t。
            union {
                uint16_t u16[2];
                uint32_t u32;
            } tmp;
            // 一次性把两个 FP8 lane 解码成两个 half 原始位。
            __half2_raw res = __nv_cvt_fp8x2_to_halfraw2(a, fp8_type);
            // 第 0 个 lane 乘回 scale 后写入低 16 bit。
            tmp.u16[0] = float_to_half(half_to_float(res.x) * scale);
            // 第 1 个 lane 乘回 scale 后写入高 16 bit。
            tmp.u16[1] = float_to_half(half_to_float(res.y) * scale);
            // 返回 half2 的打包位模式。
            return tmp.u32;
        }

        // 四个打包 FP8 lane 按 scale 解码成两个 half2。
        template<>
        __inline__ __device__ uint2 scaled_vec_conversion<uint2, uint32_t>(
            const uint32_t &a, const float scale,
            const __nv_fp8_interpretation_t fp8_type) {
            // 使用 uint2 承载两个 half2，保持向量化写回对齐。
            union {
                uint2 u32x2;
                uint32_t u32[2];
            } tmp;
            // 低 16 bit 包含前两个 FP8 lane，解码成第一个 scaled half2。
            tmp.u32[0] =
                    scaled_vec_conversion<uint32_t, uint16_t>((uint16_t) a, scale, fp8_type);
            // 高 16 bit 包含后两个 FP8 lane，解码成第二个 scaled half2。
            tmp.u32[1] = scaled_vec_conversion<uint32_t, uint16_t>((uint16_t) (a >> 16U),
                                                                   scale, fp8_type);
            // 返回 4 lane half 的打包结果。
            return tmp.u32x2;
        }

        // 八个打包 FP8 lane 按 scale 解码成四个 half2。
        template<>
        __inline__ __device__ uint4
        scaled_vec_conversion<uint4, uint2>(const uint2 &a, const float scale,
                                            const __nv_fp8_interpretation_t fp8_type) {
            // 使用 uint4 承载四个 half2，对应 8 个 half lane。
            union {
                uint4 u64x2;
                uint2 u64[2];
            } tmp;
            // a.x 存放前 4 个 FP8 lane，解码成前两个 scaled half2。
            tmp.u64[0] = scaled_vec_conversion<uint2, uint32_t>(a.x, scale, fp8_type);
            // a.y 存放后 4 个 FP8 lane，解码成后两个 scaled half2。
            tmp.u64[1] = scaled_vec_conversion<uint2, uint32_t>(a.y, scale, fp8_type);
            // 返回 8 lane half 的打包结果。
            return tmp.u64x2;
        }

        // FP8 标量按 scale 解码成 BF16 标量。
        template<>
        __inline__ __device__ __nv_bfloat16
        scaled_vec_conversion<__nv_bfloat16, uint8_t>(
            const uint8_t &a, const float scale,
            const __nv_fp8_interpretation_t fp8_type) {
            // CUDA 没有 FP8 到 BF16 的直接 intrinsic，先解码成 half 原始位。
            __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
            // half 提升成 float，便于统一乘回 scale。
            float tmp = half_to_float(res.x);
            // 乘回 scale 后再舍入成 BF16。
            return __float2bfloat16(tmp * scale);
        }

        // 两个打包 FP8 lane 按 scale 解码成 BF16x2。
        template<>
        __inline__ __device__ __nv_bfloat162
        scaled_vec_conversion<__nv_bfloat162, uint16_t>(
            const uint16_t &a, const float scale,
            const __nv_fp8_interpretation_t fp8_type) {
            // 准备 BF16x2 输出，两个 lane 分别处理低/高 8 bit。
            __nv_bfloat162 res;
            // 低 8 bit 解码成第 0 个 scaled BF16 lane。
            res.x = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t) a, scale,
                                                                  fp8_type);
            // 高 8 bit 解码成第 1 个 scaled BF16 lane。
            res.y = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t) (a >> 8U),
                                                                  scale, fp8_type);
            // 返回两个 BF16 lane 的组合结果。
            return res;
        }

        // 四个打包 FP8 lane 按 scale 解码成 bf16_4_t。
        template<>
        __inline__ __device__ bf16_4_t scaled_vec_conversion<bf16_4_t, uint32_t>(
            const uint32_t &a, const float scale,
            const __nv_fp8_interpretation_t fp8_type) {
            // 准备 4 lane BF16 输出，内部按两个 BF16x2 组织。
            bf16_4_t res;
            // 低 16 bit 解码成前两个 scaled BF16 lane。
            res.x = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t) a, scale,
                                                                    fp8_type);
            // 高 16 bit 解码成后两个 scaled BF16 lane。
            res.y = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t) (a >> 16U),
                                                                    scale, fp8_type);
            // 返回 4 lane BF16 向量。
            return res;
        }

        // 八个打包 FP8 lane 按 scale 解码成 bf16_8_t。
        template<>
        __inline__ __device__ bf16_8_t scaled_vec_conversion<bf16_8_t, uint2>(
            const uint2 &a, const float scale,
            const __nv_fp8_interpretation_t fp8_type) {
            // 先用两个 4 lane 临时结果承接 a.x 与 a.y 的解码结果。
            bf16_4_t tmp1, tmp2;
            // a.x 存放前 4 个 FP8 lane。
            tmp1 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.x, scale, fp8_type);
            // a.y 存放后 4 个 FP8 lane。
            tmp2 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.y, scale, fp8_type);
            // 准备 8 lane BF16 输出结构。
            bf16_8_t res;
            // 前 4 个 BF16 lane 写入 x/y。
            res.x = tmp1.x;
            res.y = tmp1.y;
            // 后 4 个 BF16 lane 写入 z/w。
            res.z = tmp2.x;
            res.w = tmp2.y;
            // 返回完整的 8 lane BF16 向量。
            return res;
        }

        // FP8 标量按 scale 解码成 float 标量。
        template<>
        __inline__ __device__ float scaled_vec_conversion<float, uint8_t>(
            const uint8_t &a, const float scale,
            const __nv_fp8_interpretation_t fp8_type) {
            // 先把 FP8 解码成 half 原始位，复用 CUDA 的 FP8 解码 intrinsic。
            __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
            // 取出 half 的 16 bit 存储，供 half_to_float 使用。
            uint16_t tmp = res.x;

            // half 提升成 float 后乘回 scale，恢复高精度域数值。
            return half_to_float(tmp) * scale;
        }

        // 两个打包 FP8 lane 按 scale 解码成 float2。
        template<>
        __inline__ __device__ float2 scaled_vec_conversion<float2, uint16_t>(
            const uint16_t &a, const float scale,
            const __nv_fp8_interpretation_t fp8_type) {
            // 先解码成 scaled half2 位模式，复用两 lane 路径的 scale 处理。
            uint32_t tmp = scaled_vec_conversion<uint32_t, uint16_t>(a, scale, fp8_type);
            // half2 提升成 float2 作为计算域输入。
            return half2_to_float2(tmp);
        }

        // 四个打包 FP8 lane 按 scale 解码成 Float4_。
        template<>
        __inline__ __device__ Float4_ scaled_vec_conversion<Float4_, uint32_t>(
            const uint32_t &a, const float scale,
            const __nv_fp8_interpretation_t fp8_type) {
            // Float4_ 用两个 float2 保存 4 lane 结果。
            Float4_ res;
            // 低 16 bit 解码成前两个 scaled float lane。
            res.x = scaled_vec_conversion<float2, uint16_t>((uint16_t) a, scale, fp8_type);
            // 高 16 bit 解码成后两个 scaled float lane。
            res.y = scaled_vec_conversion<float2, uint16_t>((uint16_t) (a >> 16U), scale,
                                                            fp8_type);
            // 返回 4 lane float 向量。
            return res;
        }

        // 八个打包 FP8 lane 按 scale 解码成 Float8_。
        template<>
        __inline__ __device__ Float8_ scaled_vec_conversion<Float8_, uint2>(
            const uint2 &a, const float scale,
            const __nv_fp8_interpretation_t fp8_type) {
            // 先用两个 4 lane 临时结果承接 a.x 与 a.y 的解码结果。
            Float4_ tmp1, tmp2;
            // a.x 存放前 4 个 FP8 lane。
            tmp1 = scaled_vec_conversion<Float4_, uint32_t>(a.x, scale, fp8_type);
            // a.y 存放后 4 个 FP8 lane。
            tmp2 = scaled_vec_conversion<Float4_, uint32_t>(a.y, scale, fp8_type);
            // 准备 8 lane float 输出结构。
            Float8_ res;
            // 前 4 个 float lane 写入 x/y。
            res.x = tmp1.x;
            res.y = tmp1.y;
            // 后 4 个 float lane 写入 z/w。
            res.z = tmp2.x;
            res.w = tmp2.y;
            // 返回完整的 8 lane float 向量。
            return res;
        }

        // half 标量按 scale 编码成 FP8 标量。
        template<>
        __inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, uint16_t>(
            const uint16_t &a, const float scale,
            const __nv_fp8_interpretation_t fp8_type) {
            // 写入 FP8 前先除以 scale，再按目标 FP8 格式饱和舍入。
            __nv_fp8_storage_t res =
                    __nv_cvt_float_to_fp8(half_to_float(a) / scale, __NV_SATFINITE, fp8_type);
            // 返回 FP8 的 8 bit 存储。
            return (uint8_t) res;
        }

        // BF16 标量按 scale 编码成 FP8 标量。
        template<>
        __inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, __nv_bfloat16>(
            const __nv_bfloat16 &a, const float scale,
            const __nv_fp8_interpretation_t fp8_type) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
            // SM80 之前没有 BF16 到 FP8 的 CUDA intrinsic，此路径不应在设备端执行。
            assert(false);
#else
            // BF16 提升成 float 后除以 scale，再按目标 FP8 格式饱和舍入。
            __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(__bfloat162float(a) / scale,
                                                           __NV_SATFINITE, fp8_type);
            // 返回 FP8 的 8 bit 存储。
            return (uint8_t) res;
#endif
            // 告诉编译器前面的预处理分支已经覆盖所有合法返回路径。
            __builtin_unreachable();
        }

        // float 标量按 scale 编码成 FP8 标量。
        template<>
        __inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, float>(
            const float &a, const float scale,
            const __nv_fp8_interpretation_t fp8_type) {
            // 写入 FP8 前先除以 scale，再按目标 FP8 格式饱和舍入。
            __nv_fp8_storage_t res =
                    __nv_cvt_float_to_fp8(a / scale, __NV_SATFINITE, fp8_type);
            // 返回 FP8 的 8 bit 存储。
            return (uint8_t) res;
        }

        // 四个打包 FP8 lane 按 scale 解码成 CUDA float4。
        template<>
        __inline__ __device__ float4 scaled_vec_conversion<float4, uint32_t>(
            const uint32_t &a, const float scale,
            const __nv_fp8_interpretation_t fp8_type) {
            // 先复用 Float4_ 解码路径得到两个 float2。
            Float4_ tmp = scaled_vec_conversion<Float4_, uint32_t>(a, scale, fp8_type);
            // 把内部 Float4_ 展平成 CUDA float4，便于 kernel 直接使用。
            float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
            // 返回 CUDA float4 结果。
            return res;
        }
#endif  // 结束 ENABLE_FP8 分支

        // ------------------------------- 按 KV cache dtype 分派转换路径 -------------------------------
        // 未缩放转换入口当前只保留接口形态，实际转换分支因二进制体积原因被禁用。
        template<typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
        __inline__ __device__ Tout convert(const Tin &x) {
#if 0  // 为降低二进制体积，保留但不编译未缩放分派分支。
            if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
                return vec_conversion<Tout, Tin>(x, __NV_E4M3);
            } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
                return vec_conversion<Tout, Tin>(x, __NV_E5M2);
            }
#endif
            // 未缩放转换分支被禁用时，任何运行时调用都应视为非法路径。
            assert(false);
            // 告诉编译器该非法路径不会继续返回，避免缺失 return 诊断。
            __builtin_unreachable();
        }

        // 带 scale 转换入口根据 KV cache FP8 格式选择 E4M3 或 E5M2 转换。
        template<typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
        __inline__ __device__ Tout scaled_convert(const Tin &x, const float scale) {
#ifdef ENABLE_FP8
            // E4M3 路径用于默认 FP8 KV cache 与 DeepSeek MLA FP8 cache。
            if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
                return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E4M3);
                // E5M2 路径用于显式选择 fp8_e5m2 的 KV cache。
            } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
                return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E5M2);
            }
#endif
            // 未启用 FP8 或 dtype 未覆盖时，设备端不应继续执行。
            assert(false);
            // 告诉编译器该非法路径不会继续返回，避免缺失 return 诊断。
            __builtin_unreachable();
        }

        // 依据源 dtype 与 KV cache dtype 展开转换函数模板参数。
        // auto 模式保持 cache dtype 与源 dtype 一致；fp8_ds_mla 固定使用 E4M3 存储。
        // FN 必须展开为带有以下模板参数的调用：
        //   <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
#define DISPATCH_BY_KV_CACHE_DTYPE(SRC_DTYPE, KV_DTYPE, FN)                  \
    if (KV_DTYPE == "auto") {                                                  \
      if (SRC_DTYPE == at::ScalarType::Float) {                                \
        FN(float, float, vllm::Fp8KVCacheDataType::kAuto);                     \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                          \
        FN(uint16_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);               \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(__nv_bfloat16, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto);     \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
      }                                                                        \
    } else {                                                                   \
      if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") {                       \
        if (SRC_DTYPE == at::ScalarType::Float) {                              \
          FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);              \
        } else if (SRC_DTYPE == at::ScalarType::Half) {                        \
          FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);           \
        } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                    \
          FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);      \
        } else {                                                               \
          TORCH_CHECK(false,                                                   \
                      "Unsupported input type of kv cache: ", SRC_DTYPE);      \
        }                                                                      \
      } else if (KV_DTYPE == "fp8_e5m2") {                                     \
        if (SRC_DTYPE == at::ScalarType::Float) {                              \
          FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);              \
        } else if (SRC_DTYPE == at::ScalarType::Half) {                        \
          FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);           \
        } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                    \
          FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);      \
        } else {                                                               \
          TORCH_CHECK(false,                                                   \
                      "Unsupported input type of kv cache: ", SRC_DTYPE);      \
        }                                                                      \
      } else if (KV_DTYPE == "fp8_ds_mla") {                                   \
        if (SRC_DTYPE == at::ScalarType::Float) {                              \
          FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);              \
        } else if (SRC_DTYPE == at::ScalarType::Half) {                        \
          FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);           \
        } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                    \
          FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);      \
        } else {                                                               \
          TORCH_CHECK(false,                                                   \
                      "Unsupported input type of kv cache: ", SRC_DTYPE);      \
        }                                                                      \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE);   \
      }                                                                        \
    }
    } // 命名空间 fp8
#endif  // 结束非 USE_ROCM 分支
} // 命名空间 vllm
