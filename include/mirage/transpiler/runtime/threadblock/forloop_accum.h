// forloop_accum.h - Implementation of accumulating the output
#pragma once

#include "cute/config.hpp"
#include <cute/layout.hpp>
using namespace cute;

namespace tb {

// Clear the accumulator
// (Just fill the accumulator with zeros)
template <typename T, int NUM_ELEMS, int NUM_THREADS>
class ClearAccumlatorKernel {
public:
  static constexpr int GROUP_SIZE = 16 / sizeof(T);
  static_assert(NUM_ELEMS % GROUP_SIZE ==
                0); // NUM_ELEMS should always be multiple of GROUP_SIZE
                    // (guaranteed by layout resolution)

  static __device__ __forceinline__ void run(T *__restrict__ accum,
                                             int thread_idx) {
    uint128_t *accum_128 = reinterpret_cast<uint128_t *>(accum);
    for (int elem_idx = thread_idx; elem_idx < NUM_ELEMS / GROUP_SIZE;
         elem_idx += NUM_THREADS) {
      accum_128[elem_idx] = 0ul;
    }
  }
};

template <typename T, class AccumLayout, class SrcLayout, int NUM_THREADS>
class ForloopAccumKernel {
public:
  using Numel = decltype(size(AccumLayout{}));
  CUTE_STATIC_ASSERT_V(Numel{} == size(SrcLayout{}));

  // TODO(intlsy) Use half2
  static __device__ __forceinline__ void
      run(T *__restrict__ accum, T const *__restrict__ src, int thread_idx) {
    constexpr auto numel = Numel{};
    auto accum_layout = AccumLayout{};
    auto src_layout = SrcLayout{};
    for (int elem_idx = thread_idx; elem_idx < numel; elem_idx += NUM_THREADS) {
      accum[accum_layout(elem_idx)] += src[src_layout(elem_idx)];
    }
  }
};

} // namespace tb
