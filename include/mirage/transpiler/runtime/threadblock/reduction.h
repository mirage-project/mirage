// reduction.h - Implementation of thread block level reduction operators
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

  using SrcShapeStride =
      decltype(stride(make_layout(shape(SrcLayout{}), LayoutLeft{})));
  using SrcReductionDimCoordStride =
      decltype(get<REDUCTION_DIM>(SrcShapeStride{}));
  static constexpr int SRC_REDUCTION_DIM_COORD_STRIDE =
      SrcReductionDimCoordStride::value;
  using DstCoord2SrcCoord = decltype(make_layout(
      shape(DstLayout{}),
      replace<REDUCTION_DIM>(
          SrcShapeStride{},
          Int<REDUCTION_FACTOR * SRC_REDUCTION_DIM_COORD_STRIDE>{})));
  static_assert(is_static_v<DstCoord2SrcCoord>);

  static __device__ __forceinline__ void run(T *__restrict__ dst,
                                             T const *__restrict__ src,
                                             int thread_idx,
                                             float const *epilogue_scalars) {
    auto src_layout = SrcLayout{};
    auto dst_layout = DstLayout{};
    auto dst_coord2src_coord = DstCoord2SrcCoord{};
    for (int dst_elem_idx = thread_idx; dst_elem_idx < DST_NUMEL;
         dst_elem_idx += NUM_THREADS) {
      int src_elem_idx =
          dst_coord2src_coord(dst_elem_idx); // The logical index of the first
                                             // element in the reduction group
      T result = (T)0;
      CUTE_UNROLL
      for (int i = 0; i < REDUCTION_FACTOR; ++i) {
        result +=
            src[src_layout(src_elem_idx + i * SRC_REDUCTION_DIM_COORD_STRIDE)];
      }
      auto dst_phy_pos = dst_layout(dst_elem_idx);
      Epilogue::run(result, dst, dst_phy_pos, epilogue_scalars);
    }
  }
};

} // namespace tb
