// element_binary.h - Implementation of thread block level element wise binary
// operators
#pragma once

#include <cassert>
#include <utility>

#include <cute/layout.hpp>
using namespace cute;

#include "utils.h"

namespace tb {

enum class ElementBinaryOpType { ADD, SUB, MUL, DIV, POW };

template <typename T, ElementBinaryOpType OP>
static __device__ __forceinline__ T perform_element_binary_op(T a, T b) {
  if constexpr (OP == ElementBinaryOpType::ADD) {
    return a + b;
  } else if constexpr (OP == ElementBinaryOpType::SUB) {
    return a - b;
  } else if constexpr (OP == ElementBinaryOpType::MUL) {
    return a * b;
  } else if constexpr (OP == ElementBinaryOpType::DIV) {
    return a / b;
  } else if constexpr (OP == ElementBinaryOpType::POW) {
    return (T)powf((float)a, (float)b);
  } else {
    assert(0);
  }
}

// Get a layout that converts a logical coordinate in DstLayout to a logical
// coordinate in SrcLayout, so that the composition of this layout and SrcLayout
// results in a layout that converts a logical coordinate in DstLayout to a
// physical coordinate in SrcLayout.
//
// A src dim of size 1 under a larger dst dim is a BROADCAST and must get
// stride 0, so every dst coordinate along it maps to src coordinate 0. The
// previous version used the src's plain column-major strides and compensated
// with `elem_idx % src_numel` at the call site -- which is only correct when
// the broadcast dim happens to be the slowest-varying in the dst enumeration.
// A (1,H) operand met that by accident; an (M,1) operand did not, and
// mul/div by a per-row vector returned rel ~0.6-0.8 silently.
template <typename SrcShape, typename ColMajorStride>
struct BroadcastStrideGetter {
  template <size_t... Is>
  static auto get(std::index_sequence<Is...>) {
    return cute::make_tuple(
        std::conditional_t<(cute::get<Is>(SrcShape{}) == cute::Int<1>{}),
                           cute::Int<0>,
                           cute::tuple_element_t<Is, ColMajorStride>>{}...);
  }
  using Result = decltype(get(
      std::make_index_sequence<cute::tuple_size_v<SrcShape>>{}));
};

template <typename DstLayout, typename SrcLayout>
class DstCoord2SrcCoordGetter {
  using DstShape = decltype(shape(DstLayout{}));
  using SrcShape = decltype(shape(SrcLayout{}));
  using ColMajor = decltype(stride(Layout<SrcShape>{}));
  using BStride = typename BroadcastStrideGetter<SrcShape, ColMajor>::Result;
  using Result_ = Layout<DstShape, BStride>;

public:
  using Result = decltype(coalesce(Result_{}));
};

template <typename T,
          ElementBinaryOpType OP,
          typename DstLayout,
          typename Src0Layout,
          typename Src1Layout,
          int NUM_THREADS,
          typename Epilogue>
class ElementBinaryKernel {
public:
  using Numel = decltype(cute::size(DstLayout{}));
  using Src0Numel = decltype(cute::size(Src0Layout{}));
  using Src1Numel = decltype(cute::size(Src1Layout{}));

  using DstCoord2Src0Coord =
      typename DstCoord2SrcCoordGetter<DstLayout, Src0Layout>::Result;
  using DstCoord2Src1Coord =
      typename DstCoord2SrcCoordGetter<DstLayout, Src1Layout>::Result;
  static_assert(is_static_v<DstCoord2Src0Coord>);
  static_assert(is_static_v<DstCoord2Src1Coord>);

  using DstCoord2Src0PhyPos =
      decltype(coalesce(composition(Src0Layout{}, DstCoord2Src0Coord{})));
  using DstCoord2Src1PhyPos =
      decltype(coalesce(composition(Src1Layout{}, DstCoord2Src1Coord{})));
  static_assert(is_static_v<DstCoord2Src0PhyPos>);
  static_assert(is_static_v<DstCoord2Src1PhyPos>);

  // TODO(intlsy): Use half2
  static __device__ __forceinline__ void run(T *__restrict__ dst,
                                             T const *__restrict__ src0,
                                             T const *__restrict__ src1,
                                             int thread_idx,
                                             float const *epilogue_scalars) {
    constexpr auto numel = Numel{}.value;
    auto dst_layout = DstLayout{};
    auto src0_layout_dst_coord = DstCoord2Src0PhyPos{};
    auto src1_layout_dst_coord = DstCoord2Src1PhyPos{};
    for (int elem_idx = thread_idx; elem_idx < numel; elem_idx += NUM_THREADS) {
      // Full index: broadcast dims carry stride 0 in the composed map (see
      // BroadcastStrideGetter), so no modulo trickery is needed and the result
      // is independent of which dim the dst enumeration varies fastest.
      int64_t src0_phy_pos = src0_layout_dst_coord(elem_idx);
      int64_t src1_phy_pos = src1_layout_dst_coord(elem_idx);
      int64_t dst_phy_pos = dst_layout(elem_idx);
      T res = perform_element_binary_op<T, OP>(src0[src0_phy_pos],
                                               src1[src1_phy_pos]);
      Epilogue::run(res, dst, dst_phy_pos, epilogue_scalars);
    }
  }
};

} // namespace tb