// element_unary.h - Implementation of thread block level element wise unary
// operators
#pragma once

#include <cassert>

#include <cute/layout.hpp>
using namespace cute;

#include "utils.h"

namespace tb {

enum class ElementUnaryOpType { EXP };

template <typename T, ElementUnaryOpType OP>
static __device__ __forceinline__ T perform_element_unary_op(T a) {
  if constexpr (OP == ElementUnaryOpType::EXP) {
    if constexpr (std::is_same_v<T, cutlass::half_t> ||
                  std::is_same_v<T, __half>) {
      return (T)expf((float)a);
    } else {
      assert(0);
    }
  } else {
    assert(0);
  }
}

template <typename T,
          ElementUnaryOpType OP,
          typename DstLayout,
          typename SrcLayout,
          int NUM_THREADS>
class ElementUnaryKernel {
public:
  using Numel = decltype(cute::size(DstLayout{}));

  // TODO(intlsy): Use half2
  static __device__ __forceinline__ void
      run(T *__restrict__ dst, T const *__restrict__ src, int thread_idx) {
    constexpr auto numel = Numel{}.value;
    auto dst_layout = DstLayout{};
    auto src_layout = SrcLayout{};
    for (int elem_idx = thread_idx; elem_idx < numel; elem_idx += NUM_THREADS) {
      int64_t dst_phy_pos = dst_layout(elem_idx);
      int64_t src_phy_pos = src_layout(elem_idx);
      dst[dst_phy_pos] = perform_element_unary_op<T, OP>(src[src_phy_pos]);
    }
  }
};

} // namespace tb