// input.h - Implementation of threadblock level input operators
//
// We provide these implementations:
// - Non-chunked, synchronous copy
// - Chunked, synchronous copy
// - Chunked, asynchronous copy
// - Copy using the Tensor Memory Accelerator (TMA)
//
// "Chunked" means that the copy is performed in uint128_t-sized chunks, while
// "asynchronous" means that the copy is performed using asynchronous
// instructions like cp.async. The Mirage Transpiler will choose the best
// implementation based on the layout of the input tensors and the available
// hardware.

#pragma once

#include <cstdint>
#include <cute/layout.hpp>
using namespace cute;

namespace tb {

// Type 1: Non-chunked, synchronous copy
template <typename T, class DstLayout, class SrcLayout, int NUM_THREADS>
class InputNonChunkedSyncCopy {
public:
  CUTE_STATIC_ASSERT_V(cute::size(SrcLayout{}) == cute::size(DstLayout{}));
  using Numel = decltype(cute::size(DstLayout{}));

  static __device__ __forceinline__ void
      run(T *__restrict__ dst, T const *__restrict__ src, int thread_idx) {
    constexpr auto numel = Numel{};
    auto dst_layout = DstLayout{};
    auto src_layout = SrcLayout{};
    #pragma unroll
    for (int elem_idx = thread_idx; elem_idx < numel; elem_idx += NUM_THREADS) {
      T res = src[src_layout(elem_idx)];
      dst[dst_layout(elem_idx)] = res;
    }
  }
};

// A converter that converts a layout to a "chunked" layout, which groups every
// `GROUP_SIZE` elements in the 0-th dimension into a single element.
template<class InputLayout, int GROUP_SIZE>
class ChunkedLayoutConverter {
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
// The real innermost dim should be the first dimension
// Every element in DstLayout and SrcLayout will be treated as a uint128_t
template <typename T, class DstLayout, class SrcLayout, int NUM_THREADS>
class InputChunkedSyncCopy {
public:
  CUTE_STATIC_ASSERT_V(size(SrcLayout{}) == size(DstLayout{}));

  static constexpr int GROUP_SIZE = 16 / sizeof(T);
  using DstChunkedLayout = typename ChunkedLayoutConverter<DstLayout, GROUP_SIZE>::Result;
  using SrcChunkedLayout = typename ChunkedLayoutConverter<SrcLayout, GROUP_SIZE>::Result;
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

// Type 3: Chunked, asynchronous copy
template <typename T, class DstLayout, class SrcLayout, int NUM_THREADS>
class InputChunkedAsyncCopy {
public:
  CUTE_STATIC_ASSERT_V(size(SrcLayout{}) == size(DstLayout{}));

  static constexpr int GROUP_SIZE = 16 / sizeof(T);
  using DstChunkedLayout = typename ChunkedLayoutConverter<DstLayout, GROUP_SIZE>::Result;
  using SrcChunkedLayout = typename ChunkedLayoutConverter<SrcLayout, GROUP_SIZE>::Result;
  using Numel = decltype(size(DstChunkedLayout{}));
  CUTE_STATIC_ASSERT_V(size(DstChunkedLayout{}) == size(SrcChunkedLayout{}));

  static __device__ __forceinline__ void
    run(T *__restrict__ dst, T const *__restrict__ src, int thread_idx) {
    constexpr auto numel = Numel{};
    auto dst_chunked_layout = DstChunkedLayout{};
    auto src_chunked_layout = SrcChunkedLayout{};
    #pragma unroll
    for (int elem_idx = thread_idx; elem_idx < numel; elem_idx += NUM_THREADS) {
      size_t src_addr = (size_t)(src + src_chunked_layout(elem_idx));
      uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst + dst_chunked_layout(elem_idx));
      asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;"
                   :: "r"(dst_addr), "l"(src_addr));
    }
  }
};

// Type 4: Copy using the Tensor Memory Accelerator (TMA)

} // namespace tb
