// redyctuib.h - Implementation of thread block level reduction operators
#pragma once

#include <cassert>

#include <cute/layout.hpp>
using namespace cute;

#include "utils.h"

namespace tb {

template <typename T,
          typename DstLayout,
          typename SrcLayout,
          int REDUCTION_DIM,
          int NUM_THREADS,
          class Epilogue>
class ReductionKernel {
public:
  static constexpr int NUM_DIMS = rank(SrcLayout{});
  static constexpr int DST_NUMEL = size(DstLayout{});
  CUTE_STATIC_ASSERT_V(rank(SrcLayout{}) == rank(DstLayout{}));

  CUTE_STATIC_ASSERT_V(get<REDUCTION_DIM>(shape(SrcLayout{})) %
                           get<REDUCTION_DIM>(shape(DstLayout{})) ==
                       _0{});
  static constexpr int REDUCTION_FACTOR =
      get<REDUCTION_DIM>(shape(SrcLayout{})) /
      get<REDUCTION_DIM>(shape(DstLayout{}));
  static constexpr int REDUCTION_DIM_STRIDE =
      get<REDUCTION_DIM>(stride(SrcLayout{}));

  // Refer to runtime/kernel/reduction.h for the wisdom of the following code
  using DstInSrcLayout = decltype(make_layout(
      shape(DstLayout{}),
      replace<REDUCTION_DIM>(stride(SrcLayout{}),
                             Int<REDUCTION_DIM_STRIDE * REDUCTION_FACTOR>{})));
  static_assert(is_static_v<DstInSrcLayout>);

  static __device__ __forceinline__ void
      run(T *__restrict__ dst, T const *__restrict__ src, int thread_idx) {
    auto dst_in_src_layout = DstInSrcLayout{};
    auto dst_layout = DstLayout{};
    for (int dst_elem_idx = thread_idx; dst_elem_idx < DST_NUMEL;
         dst_elem_idx += NUM_THREADS) {
      int src_elem_idx = dst_in_src_layout(dst_elem_idx);
      T result = (T)0;
      CUTE_UNROLL
      for (int i = 0; i < REDUCTION_FACTOR; ++i) {
        result += src[src_elem_idx + i * REDUCTION_DIM_STRIDE];
      }
      auto dst_phy_pos = dst_layout(dst_elem_idx);
      Epilogue::run(result, dst, dst_phy_pos);
    }
  }
};

} // namespace tb
