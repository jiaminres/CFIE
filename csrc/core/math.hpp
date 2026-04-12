#pragma once

#include <climits>
#include <iostream>

inline constexpr uint32_t bit_width(uint32_t value) {
  uint32_t width = 0;
  while (value != 0) {
    ++width;
    value >>= 1;
  }
  return width;
}

inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1u << bit_width(num - 1);
}

template <typename A, typename B>
static inline constexpr auto div_ceil(A a, B b) {
  return (a + b - 1) / b;
}

// Round a down to the next multiple of b. The caller is responsible for making
// sure that b is non-zero
template <typename T>
inline constexpr T round_to_previous_multiple_of(T a, T b) {
  return a % b == 0 ? a : (a / b) * b;
}

// Round a up to the next multiple of b. The caller is responsible for making
// sure that b is non-zero
template <typename T>
inline constexpr T round_to_next_multiple_of(T a, T b) {
  return a % b == 0 ? a : ((a / b) + 1) * b;
}
