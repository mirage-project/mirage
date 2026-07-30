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
      float result = 0;
      CUTE_UNROLL
      for (int i = 0; i < REDUCTION_FACTOR; ++i) {
        result += (float)
            src[src_layout(src_elem_idx + i * SRC_REDUCTION_DIM_COORD_STRIDE)];
      }
      auto dst_phy_pos = dst_layout(dst_elem_idx);
      Epilogue::run((T)result, dst, dst_phy_pos, epilogue_scalars);
    }
  }
};

// Initialize the reduction max operator
template <typename T, int NUM_ELEMS, int NUM_THREADS>
class InitReductionMaxKernel {
public:
  static __device__ __forceinline__ void run(T *__restrict__ updated_max,
                                             int thread_idx) {
    for (int elem_idx = thread_idx; elem_idx < NUM_ELEMS;
         elem_idx += NUM_THREADS) {
      updated_max[elem_idx] = std::numeric_limits<T>::lowest();
    }
  }
};

template <typename T,
          typename UpdatedMaxLayout,
          typename DiffLayout,
          typename SrcLayout,
          int REDUCTION_DIM,
          int NUM_THREADS> // Should not have epilogue
class ReductionMaxKernel {
public:
  static constexpr int NUM_DIMS = rank(SrcLayout{});
  static constexpr int UPDATED_MAX_NUMEL = size(UpdatedMaxLayout{});
  static constexpr int DIFF_NUMEL = size(DiffLayout{});
  CUTE_STATIC_ASSERT_V(rank(SrcLayout{}) == rank(UpdatedMaxLayout{}));
  CUTE_STATIC_ASSERT_V(rank(SrcLayout{}) == rank(DiffLayout{}));

  CUTE_STATIC_ASSERT_V(get<REDUCTION_DIM>(shape(UpdatedMaxLayout{})) == _1{});
  CUTE_STATIC_ASSERT_V(get<REDUCTION_DIM>(shape(DiffLayout{})) == _1{});

  static constexpr int REDUCTION_FACTOR =
      get<REDUCTION_DIM>(shape(SrcLayout{}));

  using SrcShapeStride =
      decltype(stride(make_layout(shape(SrcLayout{}), LayoutLeft{})));
  using SrcReductionDimCoordStride =
      decltype(get<REDUCTION_DIM>(SrcShapeStride{}));
  static constexpr int SRC_REDUCTION_DIM_COORD_STRIDE =
      SrcReductionDimCoordStride::value;

  using DstCoord2SrcCoord = decltype(make_layout(
      shape(UpdatedMaxLayout{}),
      replace<REDUCTION_DIM>(
          SrcShapeStride{},
          Int<REDUCTION_FACTOR * SRC_REDUCTION_DIM_COORD_STRIDE>{})));
  static_assert(is_static_v<DstCoord2SrcCoord>);

  static __device__ __forceinline__ void run(T *__restrict__ updated_max,
                                             T *__restrict__ diff,
                                             T const *__restrict__ src,
                                             int thread_idx) {
    auto src_layout = SrcLayout{};
    auto updated_max_layout = UpdatedMaxLayout{};
    auto diff_layout = DiffLayout{};
    auto dst_coord2src_coord = DstCoord2SrcCoord{};
    for (int dst_elem_idx = thread_idx; dst_elem_idx < UPDATED_MAX_NUMEL;
         dst_elem_idx += NUM_THREADS) {
      int src_elem_idx =
          dst_coord2src_coord(dst_elem_idx); // The logical index of the first
                                             // element in the reduction group
      float max_val = (float)updated_max[updated_max_layout(dst_elem_idx)];
      float diff_val = max_val;
      // updated_max = max(updated_max, src)
      // diff = unupdated_max - updated_max
      CUTE_UNROLL
      for (int i = 0; i < REDUCTION_FACTOR; ++i) {
        max_val =
            max_val > (float)src[src_layout(src_elem_idx +
                                            i * SRC_REDUCTION_DIM_COORD_STRIDE)]
                ? max_val
                : (float)src[src_layout(src_elem_idx +
                                        i * SRC_REDUCTION_DIM_COORD_STRIDE)];
      }
      updated_max[updated_max_layout(dst_elem_idx)] = (T)max_val;
      diff[diff_layout(dst_elem_idx)] =
          diff_val - max_val > (float)std::numeric_limits<T>::lowest() &&
                  (T)diff_val - (T)max_val <= 0
              ? (T)diff_val - (T)max_val
              : (T)0.0f;
    }
  }
};

} // namespace tb
