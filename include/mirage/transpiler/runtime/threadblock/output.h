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

// A converter that converts a layout to a "chunked" layout, which groups every
// `GROUP_SIZE` elements in the 0-th dimension into a single element.
template<class InputLayout, int GROUP_SIZE>
class OutputChunkedLayoutConverter {
  using InputRank = decltype(rank(InputLayout{}));
  using GroupSize = Int<GROUP_SIZE>;

  using InputShape = decltype(shape(InputLayout{}));
  using OutputShape = decltype(make_shape(
    ceil_div_cute(get<0>(InputShape{}), GroupSize{}),
    take<1, InputRank::value>(InputShape{})
  ));
  using InputStride = decltype(stride(InputLayout{}));
  using OutputStride = decltype(make_stride(
    get<0>(InputStride{}) * GroupSize{},
    take<1, InputRank::value>(InputStride{})
  ));

public:
  using Result = decltype(coalesce(flatten(Layout<OutputShape, OutputStride>{})));
};

// Type 2: Chunked, synchronous copy
template <typename T, class DstLayout, class SrcLayout, int NUM_THREADS>
class OutputChunkedSyncCopy {
public:
  CUTE_STATIC_ASSERT_V(size(SrcLayout{}) == size(DstLayout{}));

  static constexpr int GROUP_SIZE = 16 / sizeof(T);
  using DstChunkedLayout = typename OutputChunkedLayoutConverter<DstLayout, GROUP_SIZE>::Result;
  using SrcChunkedLayout = typename OutputChunkedLayoutConverter<SrcLayout, GROUP_SIZE>::Result;
  using Numel = decltype(size(DstChunkedLayout{}));
  CUTE_STATIC_ASSERT_V(size(DstChunkedLayout{}) == size(SrcChunkedLayout{}));

  static __device__ __forceinline__ void
    run(T *__restrict__ dst, T const *__restrict__ src, int thread_idx) {
    constexpr auto numel = Numel{};
    auto dst_chunked_layout = DstChunkedLayout{};
    auto src_chunked_layout = SrcChunkedLayout{};
    #pragma unroll
    for (int elem_idx = thread_idx; elem_idx < numel; elem_idx += NUM_THREADS) {
      uint128_t res = *((const uint128_t*)(src + src_chunked_layout(elem_idx)));
      *((uint128_t*)(dst + dst_chunked_layout(elem_idx))) = res;
    }
  }
};

// Type 3: Copy using the Tensor Memory Accelerator (TMA)

} // namespace tb
