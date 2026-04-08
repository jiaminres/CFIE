
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import struct
from dataclasses import dataclass
from enum import Enum

_SCALAR_TYPES_ID_MAP = {}


# Mirrors enum in `core/scalar_type.hpp`
class NanRepr(Enum):
    NONE = 0  # nans are not supported
    IEEE_754 = 1  # nans are: Exp all 1s, mantissa not all 0s
    EXTD_RANGE_MAX_MIN = 2  # nans are: Exp all 1s, mantissa all 1s


# This ScalarType class is a parallel implementation of the C++ ScalarType
# class found in csrc/core/scalar_type.hpp.  These two classes should be kept
# in sync until the inductor fully supports custom C++ classes.
@dataclass(frozen=True)
class ScalarType:
    """
    ScalarType 用来描述“一个标量元素的数值格式”。

    它比 torch.dtype 更通用，因为它不仅能表示：
    - 普通整数类型
    - 普通浮点类型
    - 子字节类型（例如 int4 / uint4 / fp6 / fp4）

    还可以表示“带 bias 的整数类型”。

    这里的 bias 的含义非常重要：
        stored_value = value + bias
    等价地：
        value = stored_value - bias

    这特别适合量化类型。
    例如 GPTQ 常见的 4bit 量化类型可以写成 uint4b8，
    它表示：
    - 物理上按 4bit 无符号整数存储，stored_value 范围是 [0, 15]
    - 真实逻辑值要减去 bias=8
    - 因此逻辑值范围变成 [-8, 7]

    也就是说：
    - 存储时没有符号位
    - 但通过 bias，可以表达“以 0 为中心”的有正有负数值
    """

    exponent: int
    """
    若这是浮点类型，则表示指数位 bit 数；
    若这是整数类型，则该值为 0。
    """

    mantissa: int
    """
    若这是浮点类型，则表示尾数位 bit 数；
    若这是整数类型，则表示“数值位数”：
    - 对无符号整数：就是全部 bit 数
    - 对有符号整数：是不含 sign bit 的数值位
    """

    signed: bool
    """
    是否有符号位。
    True 表示该类型带 sign bit。
    """

    bias: int
    """
    该类型的存储偏移量。

    定义关系：
        value = stored_value - bias

    例如：
    - 若是 uint8b128，则 bias=128
    - 存储值 128 表示真实值 0
    - 存储值 127 表示真实值 -1
    - 存储值 129 表示真实值 +1

    所以这里的 b 不表示 block，不表示 pack，
    而是 bias。
    """

    _finite_values_only: bool = False
    """
    私有字段：若为 True，表示浮点类型只支持有限值，不支持 inf。
    """

    nan_repr: NanRepr = NanRepr.IEEE_754
    """
    浮点类型里 NaN 的表示方式。
    整数类型不适用。
    """

    def _floating_point_max_int(self) -> int:
        # 这个函数的作用：
        # 先用“位级构造”的方式，构造出当前浮点格式能表示的最大正数，
        # 再把它映射到 IEEE double 的位布局里，最后转成 Python float。
        #
        # 这里要求 exponent / mantissa 不能太大，
        # 否则没法安全映射到 64bit double 表示。
        assert self.mantissa <= 52 and self.exponent <= 11, (
            f"Cannot represent max/min as a double for type {self.__str__()}"
        )

        # 最大尾数，全 1
        max_mantissa = (1 << self.mantissa) - 1

        # 若采用扩展范围最大最小值编码，则最后一个 mantissa 模式可能保留给特殊值
        if self.nan_repr == NanRepr.EXTD_RANGE_MAX_MIN:
            max_mantissa = max_mantissa - 1

        # 普通 IEEE 风格里，最大正常指数通常是 “全 1 减 1”
        max_exponent = (1 << self.exponent) - 2

        # 某些非标准格式允许把指数再往上扩一点
        if self.nan_repr == NanRepr.EXTD_RANGE_MAX_MIN or self.nan_repr == NanRepr.NONE:
            assert self.exponent < 11, (
                f"Cannot represent max/min as a double for type {self.__str__()}"
            )
            max_exponent = max_exponent + 1

        # 这里假设指数 bias 采用标准浮点偏置：
        #   exponent_bias = 2^(e-1) - 1
        exponent_bias = (1 << (self.exponent - 1)) - 1

        # double 的指数位宽是 11，因此它的 bias 是 1023
        exponent_bias_double = (1 << 10) - 1

        # 把当前类型的最大指数映射到 double 的指数空间
        max_exponent_double = max_exponent - exponent_bias + exponent_bias_double

        # 组装成 IEEE double 的原始 bit 模式：
        # - mantissa 放到低 52 位区域
        # - exponent 放到 exponent 区域
        return (max_mantissa << (52 - self.mantissa)) | (max_exponent_double << 52)

    def _floating_point_max(self) -> float:
        # 把上面构造出的 double 原始位模式真正转成 Python float
        double_raw = self._floating_point_max_int()
        return struct.unpack("!d", struct.pack("!Q", double_raw))[0]

    def _raw_max(self) -> int | float:
        # 返回“未扣除 bias 前”的最大原始值
        if self.is_floating_point():
            return self._floating_point_max()
        else:
            # 整数路径里，这里只允许能安全表示成 Python int 的情况
            assert self.size_bits < 64 or self.size_bits == 64 and self.is_signed(), (
                "Cannot represent max as an int"
            )
            # 对整数来说，raw max 是全 1 数值
            return (1 << self.mantissa) - 1

    def _raw_min(self) -> int | float:
        # 返回“未扣除 bias 前”的最小原始值
        if self.is_floating_point():
            # 当前实现默认所有浮点类型都有符号
            assert self.is_signed(), (
                "We currently assume all floating point types are signed"
            )

            # 在 double 表示里补上 sign bit，构造最小负数
            sign_bit_double = 1 << 63
            max_raw = self._floating_point_max_int()
            min_raw = max_raw | sign_bit_double
            return struct.unpack("!d", struct.pack("!Q", min_raw))[0]
        else:
            assert not self.is_signed() or self.size_bits <= 64, (
                "Cannot represent min as a int64_t"
            )

            if self.is_signed():
                # 有符号整数最小值，例如 int4 -> -8
                return -(1 << (self.size_bits - 1))
            else:
                # 无符号整数最小原始值是 0
                return 0

    @functools.cached_property
    def id(self) -> int:
        """
        把 ScalarType 编码成一个 int64 风格的 ID，
        供 pytorch custom op / C++ 扩展使用。

        这个编码布局必须和 C++ 侧 scalar_type.hpp 保持一致。
        """
        val = 0
        offset = 0

        def or_and_advance(member, bit_width):
            nonlocal val
            nonlocal offset
            bit_mask = (1 << bit_width) - 1
            val = val | (int(member) & bit_mask) << offset
            offset = offset + bit_width

        # 把各字段按固定 bit 宽编码进一个整数里
        or_and_advance(self.exponent, 8)
        or_and_advance(self.mantissa, 8)
        or_and_advance(self.signed, 1)
        or_and_advance(self.bias, 32)
        or_and_advance(self._finite_values_only, 1)
        or_and_advance(self.nan_repr.value, 8)

        assert offset <= 64, f"ScalarType fields too big {offset} to fit into an int64"

        # 建立从 id -> ScalarType 的映射缓存
        _SCALAR_TYPES_ID_MAP[val] = self
        return val

    @property
    def size_bits(self) -> int:
        # 总 bit 数：
        # 浮点：exponent + mantissa + sign
        # 整数：mantissa + sign（因为 exponent=0）
        return self.exponent + self.mantissa + int(self.signed)

    def min(self) -> int | float:
        """
        当前类型可表示的最小真实值。
        注意这里已经考虑了 bias：
            min = raw_min - bias
        """
        return self._raw_min() - self.bias

    def max(self) -> int | float:
        """
        当前类型可表示的最大真实值。
        注意这里已经考虑了 bias：
            max = raw_max - bias
        """
        return self._raw_max() - self.bias

    def is_signed(self) -> bool:
        """
        是否有 sign bit。
        """
        return self.signed

    def is_floating_point(self) -> bool:
        """
        是否为浮点类型。
        判定规则：exponent != 0
        """
        return self.exponent != 0

    def is_integer(self) -> bool:
        """
        是否为整数类型。
        判定规则：exponent == 0
        """
        return self.exponent == 0

    def has_bias(self) -> bool:
        """
        是否带非零 bias。
        """
        return self.bias != 0

    def has_infs(self) -> bool:
        """
        是否支持无穷大。
        """
        return not self._finite_values_only

    def has_nans(self) -> bool:
        # 注意：这里源码写的是和 nan_repr.value 比较
        return self.nan_repr != NanRepr.NONE.value

    def is_ieee_754(self) -> bool:
        """
        是否是标准 IEEE754 风格浮点类型。
        """
        return self.nan_repr == NanRepr.IEEE_754.value and not self._finite_values_only

    def __str__(self) -> str:
        """
        生成类型名字符串。

        浮点类型命名规则：
            float<size_bits>_e<exp_bits>m<mantissa_bits>[flags]

        整数类型命名规则：
            [u]int<size_bits>[b<bias>]

        这里非常关键：
        - 整数类型后缀里的 bXXX 表示 bias
        - 例如 uint4b8 表示：
            4bit 无符号整数 + bias=8
        - 不是 block，不是 pack
        """
        if self.is_floating_point():
            ret = (
                "float"
                + str(self.size_bits)
                + "_e"
                + str(self.exponent)
                + "m"
                + str(self.mantissa)
            )

            if not self.is_ieee_754():
                if self._finite_values_only:
                    ret = ret + "f"
                if self.nan_repr != NanRepr.NONE:
                    ret = ret + "n"

            return ret
        else:
            ret = ("int" if self.is_signed() else "uint") + str(self.size_bits)
            if self.has_bias():
                # 这里的 b 就是 bias
                ret = ret + "b" + str(self.bias)
            return ret

    def __repr__(self) -> str:
        return "ScalarType." + self.__str__()

    def __len__(self) -> int:
        # 为了兼容 pytorch opcheck，需要定义 __len__，
        # 但这里故意抛 TypeError
        raise TypeError

    #
    # Convenience Constructors
    #

    @classmethod
    def int_(cls, size_bits: int, bias: int | None) -> "ScalarType":
        """
        构造一个有符号整数类型。
        注意：
        - size_bits 包含 sign bit
        - bias 若为 None，则按 0 处理
        """
        ret = cls(0, size_bits - 1, True, bias if bias else 0)
        ret.id
        return ret

    @classmethod
    def uint(cls, size_bits: int, bias: int | None) -> "ScalarType":
        """
        构造一个无符号整数类型。
        注意：
        - 若 bias 非零，则这个“无符号存储类型”可以表达带正负的真实值
        - 例如 uint4b8:
            stored ∈ [0, 15]
            value  = stored - 8 ∈ [-8, 7]
        """
        ret = cls(0, size_bits, False, bias if bias else 0)
        ret.id
        return ret

    @classmethod
    def float_IEEE754(cls, exponent: int, mantissa: int) -> "ScalarType":
        """
        构造标准 IEEE754 风格浮点类型。
        """
        assert mantissa > 0 and exponent > 0
        ret = cls(exponent, mantissa, True, 0)
        ret.id
        return ret

    @classmethod
    def float_(
        cls, exponent: int, mantissa: int, finite_values_only: bool, nan_repr: NanRepr
    ) -> "ScalarType":
        """
        构造非标准浮点类型。
        """
        assert mantissa > 0 and exponent > 0
        assert nan_repr != NanRepr.IEEE_754, (
            "use `float_IEEE754` constructor for floating point types that "
            "follow IEEE 754 conventions"
        )
        ret = cls(exponent, mantissa, True, 0, finite_values_only, nan_repr)
        ret.id
        return ret

    @classmethod
    def from_id(cls, scalar_type_id: int):
        # 通过编码后的 id 反查对应 ScalarType
        if scalar_type_id not in _SCALAR_TYPES_ID_MAP:
            raise ValueError(f"scalar_type_id {scalar_type_id} doesn't exists.")
        return _SCALAR_TYPES_ID_MAP[scalar_type_id]


# naming generally follows: https://github.com/jax-ml/ml_dtypes
# for floating point types (leading f) the scheme is:
#  `float<size_bits>_e<exponent_bits>m<mantissa_bits>[flags]`
#  flags:
#  - no-flags: means it follows IEEE 754 conventions
#  - f: means finite values only (no infinities)
#  - n: means nans are supported (non-standard encoding)
# for integer types the scheme is:
#  `[u]int<size_bits>[b<bias>]`
#  - if bias is not present it means its zero


class scalar_types:
    # -------------------------------------------------------------
    # 1) 普通整数类型
    # -------------------------------------------------------------

    # 有符号 4bit 整数
    # size_bits = 4
    # signed = True
    # bias = 0
    #
    # 真实值范围大致：
    #   [-8, 7]
    int4 = ScalarType.int_(4, None)

    # 无符号 4bit 整数
    # size_bits = 4
    # signed = False
    # bias = 0
    #
    # 真实值范围：
    #   [0, 15]
    uint4 = ScalarType.uint(4, None)

    # 有符号 8bit 整数
    int8 = ScalarType.int_(8, None)

    # 无符号 8bit 整数
    uint8 = ScalarType.uint(8, None)

    # -------------------------------------------------------------
    # 2) 8bit 浮点类型
    # -------------------------------------------------------------

    # FP8 e4m3fn
    # - exponent bits = 4
    # - mantissa bits = 3
    # - finite-values-only / nan 表示策略由参数决定
    float8_e4m3fn = ScalarType.float_(4, 3, True, NanRepr.EXTD_RANGE_MAX_MIN)

    # 标准 IEEE754 风格 FP8 e5m2
    float8_e5m2 = ScalarType.float_IEEE754(5, 2)

    # 特殊 8bit 浮点变体
    float8_e8m0fnu = ScalarType(8, 0, False, 0, True, NanRepr.EXTD_RANGE_MAX_MIN)

    # -------------------------------------------------------------
    # 3) 16bit 浮点类型
    # -------------------------------------------------------------

    # bfloat16 风格：e8m7
    float16_e8m7 = ScalarType.float_IEEE754(8, 7)

    # IEEE half / float16 风格：e5m10
    float16_e5m10 = ScalarType.float_IEEE754(5, 10)

    # -------------------------------------------------------------
    # 4) 更低位浮点类型
    # -------------------------------------------------------------

    float6_e3m2f = ScalarType.float_(3, 2, True, NanRepr.NONE)
    float6_e2m3f = ScalarType.float_(2, 3, True, NanRepr.NONE)
    float4_e2m1f = ScalarType.float_(2, 1, True, NanRepr.NONE)

    # -------------------------------------------------------------
    # 5) GPTQ 常见“带 bias 的无符号整数类型”
    # -------------------------------------------------------------
    # 注意：
    # 这里命名中的 b 不是 block，不是 pack，
    # 而是 bias。
    #
    # 例如 uint4b8:
    # - 存储类型是 4bit unsigned，stored_value ∈ [0, 15]
    # - bias = 8
    # - 真实值 value = stored_value - 8
    # - 因此真实值范围是 [-8, 7]
    #
    # 这正适合 GPTQ 这类量化场景：
    # - 物理存储用无符号整数更方便
    # - 但逻辑上又想表达围绕 0 对称的量化值

    # 2bit unsigned + bias=2
    # stored ∈ [0, 3]
    # value  ∈ [-2, 1]
    uint2b2 = ScalarType.uint(2, 2)

    # 3bit unsigned + bias=4
    # stored ∈ [0, 7]
    # value  ∈ [-4, 3]
    uint3b4 = ScalarType.uint(3, 4)

    # 4bit unsigned + bias=8
    # stored ∈ [0, 15]
    # value  ∈ [-8, 7]
    #
    # 这是 GPTQ / Marlin 路径里非常常见的一种类型
    uint4b8 = ScalarType.uint(4, 8)

    # 8bit unsigned + bias=128
    # stored ∈ [0, 255]
    # value  ∈ [-128, 127]
    uint8b128 = ScalarType.uint(8, 128)

    # -------------------------------------------------------------
    # 6) 常用别名
    # -------------------------------------------------------------

    # bfloat16 别名
    bfloat16 = float16_e8m7

    # float16 / half 别名
    float16 = float16_e5m10
