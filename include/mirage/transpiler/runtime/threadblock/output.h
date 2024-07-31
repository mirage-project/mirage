// output.h - Implementation of threadblock level output operators
//
// We provide these implementations:
// - Non-chunked, synchronous copy
// - Chunked, synchronous copy
// - Copy using the Tensor Memory Accelerator (TMA)
//
// For the meaning of "chunked" and "asynchronous", please refer to `input.h`
// in the same directory.

#pragma once

#include <cute/layout.hpp>
using namespace cute;

namespace tb {

// Type 1: Non-chunked, synchronous copy
template <typename T, class DstLayout, class SrcLayout, int NUM_THREADS>
class OutputNonChunkedSyncCopy {
public:
  CUTE_STATIC_ASSERT_V(cute::size(SrcLayout{}) == cute::size(DstLayout{}));
  using Numel = decltype(cute::size(DstLayout{}));

  static __device__ __forceinline__ void
      run(T *__restrict__ dst, T const *__restrict__ src, int thread_idx) {
    constexpr auto numel = Numel{};
    auto dst_layout = DstLayout{};
    auto src_layout = SrcLayout{};
    for (int elem_idx = thread_idx; elem_idx < numel; elem_idx += NUM_THREADS) {
      dst[dst_layout(elem_idx)] = src[src_layout(elem_idx)];
    }
  }
};

// Type 2: Chunked, synchronous copy

// Type 3: Copy using the Tensor Memory Accelerator (TMA)

} // namespace tb
