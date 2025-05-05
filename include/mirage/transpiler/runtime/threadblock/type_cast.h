// type_cast.h - Implementation of thread block level type casting operators
#pragma once

#include <cassert>

#include <cute/layout.hpp>
using namespace cute;

#include "utils.h"

namespace tb {

template <typename Tout, typename Tin>
static __device__ __forceinline__ Tout perform_type_cast_op(Tin f) {
  if constexpr (std::is_same_v<Tin, cutlass::bfloat16_t> ||
                  std::is_same_v<Tin, float>) {
    return (Tout)cutlass::float_e4m3_t::from_float(f);
  } else if constexpr (std::is_same_v<Tin, cutlass::half_t> ||
                  std::is_same_v<Tin, __half>) {
    return (Tout)cutlass::float_e4m3_t::from_half(f);
  } else {
    assert(0 && "unsupport input datatype in tb typecast");
  }

  return (Tout)0.0;
}

template <typename Tout,
          typename Tin,
          typename DstLayout,
          typename SrcLayout,
          int NUM_THREADS>
class TypeCastKernel {
public:
  using Numel = decltype(cute::size(DstLayout{}));

  // TODO(intlsy): Use half2
  static __device__ __forceinline__ void run(Tout *__restrict__ dst,
                                             Tin const *__restrict__ src,
                                             int thread_idx) {

    constexpr auto numel = Numel{}.value;
    auto dst_layout = DstLayout{};
    auto src_layout = SrcLayout{};
    for (int elem_idx = thread_idx; elem_idx < numel; elem_idx += NUM_THREADS) {
      int64_t dst_phy_pos = dst_layout(elem_idx);
      int64_t src_phy_pos = src_layout(elem_idx);
      Tout res = perform_type_cast_op<Tout, Tin>(src[src_phy_pos]);
      dst[dst_phy_pos] = res;
    }
  }
};

} // namespace tb