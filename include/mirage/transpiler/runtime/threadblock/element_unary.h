// element_unary.h - Implementation of thread block level element wise unary
// operators
#pragma once

#include <cassert>

#include <cute/layout.hpp>
using namespace cute;

#include "utils.h"

namespace tb {

enum class ElementUnaryOpType { EXP, SILU, SQUARE, SQRT, MULSCALAR };

template <typename T, ElementUnaryOpType OP>
static __device__ __forceinline__ T
    perform_element_unary_op(T a, float scalar = 0.0f) {
  if constexpr (!(std::is_same_v<T, cutlass::half_t> ||
                  std::is_same_v<T, __half>)) {
    assert(0 && "unsupport datatype in tb elementunary");
  }
  if constexpr (OP == ElementUnaryOpType::EXP) {
    return (T)expf((float)a);
  } else if constexpr (OP == ElementUnaryOpType::SILU) {
    return (T)(a * (T(1) / (T(1) + fast_exp(-a))));
  } else if (OP == ElementUnaryOpType::SQUARE) {
    return (T)((float)a * (float)a);
  } else if (OP == ElementUnaryOpType::SQRT) {
    return (T)(sqrtf((float)a));
  } else if (OP == ElementUnaryOpType::MULSCALAR) {
    return (T)(scalar * (float)a);
  }

  assert(0 && "unsupport optype in tb elementunary");
}

template <typename T,
          ElementUnaryOpType OP,
          typename DstLayout,
          typename SrcLayout,
          int NUM_THREADS,
          class Epilogue>
class ElementUnaryKernel {
public:
  using Numel = decltype(cute::size(DstLayout{}));

  // TODO(intlsy): Use half2
  static __device__ __forceinline__ void run(T *__restrict__ dst,
                                             T const *__restrict__ src,
                                             int thread_idx,
                                             float scalar) {
    constexpr auto numel = Numel{}.value;
    auto dst_layout = DstLayout{};
    auto src_layout = SrcLayout{};
    for (int elem_idx = thread_idx; elem_idx < numel; elem_idx += NUM_THREADS) {
      int64_t dst_phy_pos = dst_layout(elem_idx);
      int64_t src_phy_pos = src_layout(elem_idx);
      T res = perform_element_unary_op<T, OP>(src[src_phy_pos], scalar);
      Epilogue::run(res, dst, dst_phy_pos);
    }
  }
};

} // namespace tb