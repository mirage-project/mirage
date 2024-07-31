// output.h - Implementation of accumulating the output
#pragma once

#include <cute/layout.hpp>
using namespace cute;

namespace tb {

template <typename T, int NUM_ELEMS, int NUM_THREADS>
class ClearOutputAccumKernel {
public:
  // TODO(intlsy): Use half2
  static __device__ __forceinline__ void run(T *__restrict__ accum,
                                             int thread_idx) {
    for (int elem_idx = thread_idx; elem_idx < NUM_ELEMS;
         elem_idx += NUM_THREADS) {
      accum[elem_idx] = 0;
    }
  }
};

template <typename T, class DstSrcLayout, int NUM_THREADS>
class AccumOutputKernel {
public:
  using Numel = decltype(cute::size(DstSrcLayout{}));

  // TODO(intlsy) Use half2
  static __device__ __forceinline__ void
      run(T *__restrict__ accum, T const *__restrict__ src, int thread_idx) {
    constexpr auto numel = Numel{};
    auto dst_src_layout = DstSrcLayout{};
    for (int elem_idx = thread_idx; elem_idx < numel; elem_idx += NUM_THREADS) {
      int phy_addr = dst_src_layout(elem_idx);
      accum[phy_addr] += src[phy_addr];
    }
  }
};

} // namespace tb
