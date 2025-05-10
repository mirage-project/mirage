#pragma once

#include <cassert>

#include <cute/layout.hpp>
using namespace cute;

#include "utils.h"

namespace tb {

template <typename T,
          typename Dst0Layout,
          typename Dst1Layout,
          typename SrcLayout,
          int CHUNK_SIZE,
          int CHUNK_DIM,
          int NUM_THREADS,
          typename Epilogue>
class ChunkKernel {
public:
  using Numel = decltype(cute::size(SrcLayout{}));

  static constexpr int SRC_SIZE = get<CHUNK_DIM>(shape(SrcLayout{}));
  static constexpr int DST0_SIZE = get<CHUNK_DIM>(shape(Dst0Layout{}));
  static constexpr int DST1_SIZE = get<CHUNK_DIM>(shape(Dst1Layout{}));
  static_assert(DST0_SIZE == SRC_SIZE / 2);
  static_assert(DST1_SIZE == SRC_SIZE / 2);

  static __device__ __forceinline__ void run(T *__restrict__ dst0,
                                             T const *__restrict__ dst1,
                                             T const *__restrict__ src,
                                             int thread_idx,
                                             float const *epilogue_scalars) {
    constexpr auto numel = Numel{}.value;
    auto src_layout = SrcLayout{};
    auto dst0_layout = Dst0Layout{};
    auto dst1_layout = Dst1Layout{};

    for (int elem_idx = thread_idx; elem_idx < numel; elem_idx += NUM_THREADS) {
      auto src_coord = src_layout.get_flat_coord(elem_idx);
      auto src_phy_pos = src_layout(elem_idx);
      if (get<CHUNK_DIM>(src_coord) < DST0_SIZE) {
        auto dst0_phy_pos = dst0_layout(src_coord);
        Epilogue::run(src[src_phy_pos], dst0, dst0_phy_pos, epilogue_scalars);
      } else {
        replace<CHUNK_DIM>(src_coord, get<CHUNK_DIM>(src_coord) - DST0_SIZE);
        auto dst1_phy_pos = dst1_layout(src_coord);
        Epilogue::run(src[src_phy_pos], dst1, dst1_phy_pos, epilogue_scalars);
      }
    }
  }
};

} // namespace tb
