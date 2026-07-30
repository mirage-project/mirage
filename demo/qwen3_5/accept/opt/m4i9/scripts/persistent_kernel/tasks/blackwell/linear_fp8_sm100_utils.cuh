#pragma once

#include <cuda/std/cstdint>
#include <cuda/std/utility>
#include <cuda_bf16.h>
#include <cute/container/tuple.hpp>

#include "linear_fp8_sm100_cute_tie.cuh"

#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("Assertion failed: %s:%d, condition: %s\n",                       \
             __FILE__,                                                         \
             __LINE__,                                                         \
             #cond);                                                           \
      asm("trap;");                                                            \
    }                                                                          \
  } while (0)
#endif

#ifndef DG_TRAP_ONLY_DEVICE_ASSERT
#define DG_TRAP_ONLY_DEVICE_ASSERT(cond)                                       \
  do {                                                                         \
    if (!(cond))                                                               \
      asm("trap;");                                                            \
  } while (0)
#endif

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

namespace mirage::blackwell::linear_fp8_sm100 {

template <typename FuncT>
struct PatternVisitor {
  FuncT func;

  __device__ __host__ explicit PatternVisitor(FuncT &&func)
      : func(std::forward<FuncT>(func)) {}

  __device__ __host__ auto operator[](uint32_t const &i) {
    return func(i);
  }
};

template <typename T>
__device__ __host__ T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
__device__ __host__ T align(T a, T b) {
  return ceil_div(a, b) * b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_align(T a, T b) {
  return constexpr_ceil_div(a, b) * b;
}

template <typename T>
__device__ __host__ constexpr T constexpr_gcd(T a, T b) {
  return b == 0 ? a : constexpr_gcd(b, a % b);
}

template <typename T>
__forceinline__ __device__ void swap(T &a, T &b) {
  T temp = a;
  a = b;
  b = temp;
}

__forceinline__ __device__ uint32_t get_sm_idx() {
  uint32_t sm_idx;
  asm("mov.u32 %0, %%smid;" : "=r"(sm_idx));
  return sm_idx;
}

__forceinline__ __device__ uint32_t get_lane_idx() {
  uint32_t lane_id;
  asm("mov.u32 %0, %laneid;" : "=r"(lane_id));
  return lane_id;
}

__device__ __forceinline__ uint32_t ld_shared(uint32_t const *ptr) {
  uint32_t ret;
  asm volatile("ld.shared.u32 %0, [%1];"
               : "=r"(ret)
               : "l"(__cvta_generic_to_shared(ptr)));
  return ret;
}

__device__ __forceinline__ void st_shared(uint32_t const *ptr, uint32_t val) {
  asm volatile("st.shared.u32 [%0], %1;"
               :
               : "l"(__cvta_generic_to_shared(ptr)), "r"(val));
}

__device__ __forceinline__ void
    st_shared(void const *ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile(
      "st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
      :
      : "l"(__cvta_generic_to_shared(ptr)), "r"(x), "r"(y), "r"(z), "r"(w));
}

template <typename old_t>
__device__ __forceinline__ int cast_into_bf16_and_pack(old_t &x, old_t &y) {
  auto bf16x2 = __float22bfloat162_rn(
      {*reinterpret_cast<float *>(&x), *reinterpret_cast<float *>(&y)});
  return *reinterpret_cast<int *>(&bf16x2);
}

__device__ __forceinline__ void add_packed_bf16x2_into_fp32_bits(
    uint32_t packed_residual, uint32_t &x_bits, uint32_t &y_bits) {
  union BitsFloat {
    uint32_t bits;
    float value;
  };
  union PackedBf16x2 {
    uint32_t bits;
    __nv_bfloat162 value;
  };

  BitsFloat x{.bits = x_bits};
  BitsFloat y{.bits = y_bits};
  PackedBf16x2 residual{.bits = packed_residual};
  float2 const residual_vals = __bfloat1622float2(residual.value);
  x.value += residual_vals.x;
  y.value += residual_vals.y;
  x_bits = x.bits;
  y_bits = y.bits;
}

} // namespace mirage::blackwell::linear_fp8_sm100
