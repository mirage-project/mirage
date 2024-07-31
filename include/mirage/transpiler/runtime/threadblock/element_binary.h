// element_binary.h - Implementation of thread block level element wise binary
// operators
#pragma once

#include <cassert>

#include <cute/layout.hpp>
using namespace cute;

#include "utils.h"

namespace tb {

enum class ElementBinaryOpType { ADD, MUL, DIV };

template <typename T, ElementBinaryOpType OP>
static __device__ __forceinline__ T perform_element_binary_op(T a, T b) {
  if constexpr (OP == ElementBinaryOpType::ADD) {
    return a + b;
  } else if constexpr (OP == ElementBinaryOpType::MUL) {
    return a * b;
  } else if constexpr (OP == ElementBinaryOpType::DIV) {
    return a / b;
  } else {
    assert(0);
  }
}

class ReplacementFunctionClass {
public:
  template <auto cur_dim_shape, auto cur_dim_stride>
  constexpr Int<cur_dim_shape == 1 ? 0 : cur_dim_stride>
      operator()(Int<cur_dim_shape>, Int<cur_dim_stride>) {
    return {};
  }
};

template <typename T,
          ElementBinaryOpType OP,
          typename DstLayout,
          typename Src0Layout,
          typename Src1Layout,
          int NUM_THREADS>
class ElementBinaryKernel {
public:
  using Numel = decltype(cute::size(DstLayout{}));

  // Layouts that take an logical index from dst's layout, and return the
  // physical index in src0 and src1. We use this to implement broadcast: both
  // two layouts have the same shape as DstLayout, while for broadcast
  // dimentions, we set the stride to 0
  using Src0InDstLayout =
      decltype(make_layout(shape(DstLayout{}),
                           transform(shape(Src0Layout{}),
                                     stride(Src0Layout{}),
                                     ReplacementFunctionClass{})));
  using Src1InDstLayout =
      decltype(make_layout(shape(DstLayout{}),
                           transform(shape(Src1Layout{}),
                                     stride(Src1Layout{}),
                                     ReplacementFunctionClass{})));
  static_assert(is_static_v<Src0InDstLayout>);
  static_assert(is_static_v<Src1InDstLayout>);

  // TODO(intlsy): Use half2
  static __device__ __forceinline__ void run(T *__restrict__ dst,
                                             T const *__restrict__ src0,
                                             T const *__restrict__ src1,
                                             int thread_idx) {
    constexpr auto numel = Numel{}.value;
    auto dst_layout = DstLayout{};
    auto src0_in_dst_layout = Src0InDstLayout{};
    auto src1_in_dst_layout = Src1InDstLayout{};
    for (int elem_idx = thread_idx; elem_idx < numel; elem_idx += NUM_THREADS) {
      int64_t src0_phy_pos = src0_in_dst_layout(elem_idx);
      int64_t src1_phy_pos = src1_in_dst_layout(elem_idx);
      int64_t dst_phy_pos = dst_layout(elem_idx);
      dst[dst_phy_pos] = perform_element_binary_op<T, OP>(src0[src0_phy_pos],
                                                          src1[src1_phy_pos]);
    }
  }
};

} // namespace tb