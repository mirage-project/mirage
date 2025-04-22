// element_unary.h - Implementation of thread block level element wise unary
// operators
#pragma once

#include <cassert>

#include <cute/layout.hpp>
using namespace cute;

#include "utils.h"

namespace tb {

enum class ElementUnaryOpType {
  EXP,
  SILU,
  GELU,
  RELU,
  CLAMP,
  SQUARE,
  SQRT,
  MULSCALAR
};

template <typename T, ElementUnaryOpType OP>
static __device__ __forceinline__ T
    perform_element_unary_op(T a, float scalar = 0.0f) {
  if constexpr (!(std::is_same_v<T, cutlass::half_t> ||
                  std::is_same_v<T, cutlass::bfloat16_t> ||
                  std::is_same_v<T, float> || std::is_same_v<T, __half>)) {
    assert(0 && "unsupport datatype in tb elementunary");
  }
  if constexpr (OP == ElementUnaryOpType::EXP) {
    return (T)expf((float)a);
  } else if constexpr (OP == ElementUnaryOpType::SILU) {
    return (T)(((float)a) * (1.0f / (1.0f + expf((float)-a))));
  } else if constexpr (OP == ElementUnaryOpType::GELU) {
    return (T)((((float)a) / 2.0f) * (1.0f + erff(((float)a) / sqrtf(2.0f))));
  } else if constexpr (OP == ElementUnaryOpType::RELU) {
    return (T)(fmaxf(0.f, (float)a));
  } else if constexpr (OP == ElementUnaryOpType::CLAMP) {
    return (T)(fmaxf(0.f, fminf((float)a, 1.f)));
  } else if constexpr (OP == ElementUnaryOpType::SQUARE) {
    return (T)((float)a * (float)a);
  } else if constexpr (OP == ElementUnaryOpType::SQRT) {
    return (T)(sqrtf((float)a));
  } else if constexpr (OP == ElementUnaryOpType::MULSCALAR) {
    return (T)(scalar * (float)a);
  } else {
    assert(0 && "unsupport optype in tb elementunary");
  }

  return (T)0.0;
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
                                             float scalar,
                                             float const *epilogue_scalars) {

    constexpr auto numel = Numel{}.value;
    auto dst_layout = DstLayout{};
    auto src_layout = SrcLayout{};
    for (int elem_idx = thread_idx; elem_idx < numel; elem_idx += NUM_THREADS) {
      int64_t dst_phy_pos = dst_layout(elem_idx);
      int64_t src_phy_pos = src_layout(elem_idx);
      T res = perform_element_unary_op<T, OP>(src[src_phy_pos], scalar);
      Epilogue::run(res, dst, dst_phy_pos, epilogue_scalars);
    }
  }
};

} // namespace tb