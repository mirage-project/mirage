// amax.h - Implementation of thread block level absolute maximum
#pragma once

#include <cassert>

#include <cute/layout.hpp>
using namespace cute;

#include "utils.h"

namespace tb {

template <typename T,
          typename DstLayout,
          typename SrcLayout,
          int NUM_THREADS>
class AmaxKernel {
public:
  using Numel = decltype(cute::size(DstLayout{}));

  // TODO(intlsy): Use half2
  static __device__ __forceinline__ void run(T *__restrict__ dst,
                                             T const *__restrict__ src,
                                             int thread_idx) {
    // each thread marches through the tile with a simple strided loop
    T local_amax = T{0};
    for (int i = thread_idx; i < Numel{}; i += NUM_THREADS) {
      T v = src[i];
      T a = v < T{0} ? -v : v;
      local_amax = cute::max(local_amax, a);
    }

    // shared-memory tree reduction
    __shared__ T smem[NUM_THREADS];
    smem[thread_idx] = local_amax;
    __syncthreads();

    for (int off = NUM_THREADS>>1; off > 0; off >>= 1) {
      if (thread_idx < off) {
        smem[thread_idx] = cute::max(smem[thread_idx],
                                     smem[thread_idx + off]);
      }
      __syncthreads();
    }

    if (thread_idx == 0) {
      dst[0] = smem[0];
    }
  }
};

} // namespace tb