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

#include "cute/arch/cluster_sm90.hpp"
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

// Get a mapping from chunk coordinate to original coordinate
//
// Assume the shape of the tensor is (A0, A1, ...) (where A0 is the innermost
// dim), then if we merge elements in the same chunk into one element, the shape
// would be (A0', A1, ...), where A0' is the number of chunks in the innermost
// dim (ceil(A0 / CHUNK_SIZE)).
//
// This class returns such a mapping. It takes a coordinate in the "chunk space"
// and converts it to a coordinate in the "original space".
template <class InputLayout, int CHUNK_SIZE>
class GetChunkedCoord2Coord {
  using InputShape = decltype(shape(InputLayout{}));
  static constexpr int INNERMOST_DIM_SIZE = get<0>(InputShape{}).value;
  static constexpr int INNERMOST_DIM_NUM_CHUNKS =
      ceil_div(INNERMOST_DIM_SIZE, CHUNK_SIZE);

  using Result_ = decltype(make_layout(
      replace<0>(InputShape{}, Int<INNERMOST_DIM_NUM_CHUNKS>{}),
      replace<0>(stride(make_layout(InputShape{}, LayoutLeft{})),
                 Int<CHUNK_SIZE>{})));

public:
  using Result = decltype(coalesce(Result_{}));
};

// Type 2: Chunked, synchronous copy
// The real innermost dim should be the first dimension
template <typename T, class DstLayout, class SrcLayout, int NUM_THREADS>
class InputChunkedSyncCopy {
public:
  CUTE_STATIC_ASSERT_V(size(SrcLayout{}) == size(DstLayout{}));

  static constexpr int CHUNK_SIZE = 16 / sizeof(T);
  using SrcChunkedCoord2Coord =
      typename GetChunkedCoord2Coord<SrcLayout, CHUNK_SIZE>::Result;
  using DstChunkedCoord2Coord =
      typename GetChunkedCoord2Coord<DstLayout, CHUNK_SIZE>::Result;
  static constexpr int NUM_CHUNKS = size(SrcChunkedCoord2Coord{}).value;

  static __device__ __forceinline__ void
      run(T *__restrict__ dst, T const *__restrict__ src, int thread_idx) {
    auto src_layout = SrcLayout{};
    auto dst_layout = DstLayout{};
    auto src_chunked_coord2coord = SrcChunkedCoord2Coord{};
    auto dst_chunked_coord2coord = DstChunkedCoord2Coord{};
#pragma unroll
    for (int chunk_idx = thread_idx; chunk_idx < NUM_CHUNKS;
         chunk_idx += NUM_THREADS) {
      uint128_t res =
          *((uint128_t const *)(src + src_layout(
                                          src_chunked_coord2coord(chunk_idx))));
      *((uint128_t *)(dst + dst_layout(dst_chunked_coord2coord(chunk_idx)))) =
          res;
    }
  }
};

// Type 3: Chunked, asynchronous copy
template <typename T, class DstLayout, class SrcLayout, int NUM_THREADS>
class InputChunkedAsyncCopy {
public:
  CUTE_STATIC_ASSERT_V(size(SrcLayout{}) == size(DstLayout{}));

  static constexpr int CHUNK_SIZE = 16 / sizeof(T);
  using SrcChunkedCoord2Coord =
      typename GetChunkedCoord2Coord<SrcLayout, CHUNK_SIZE>::Result;
  using DstChunkedCoord2Coord =
      typename GetChunkedCoord2Coord<DstLayout, CHUNK_SIZE>::Result;
  static constexpr int NUM_CHUNKS = size(SrcChunkedCoord2Coord{}).value;

  static __device__ __forceinline__ void
      run(T *__restrict__ dst, T const *__restrict__ src, int thread_idx) {
    auto src_layout = SrcLayout{};
    auto dst_layout = DstLayout{};
    auto src_chunked_coord2coord = SrcChunkedCoord2Coord{};
    auto dst_chunked_coord2coord = DstChunkedCoord2Coord{};
    uint32_t dst_base_addr = cute::cast_smem_ptr_to_uint(dst);

#pragma unroll
    for (int chunk_idx = thread_idx; chunk_idx < NUM_CHUNKS;
         chunk_idx += NUM_THREADS) {
      // if (threadIdx.x == 0 && blockIdx.x == 0 && chunk_idx == 0) {
      //   printf("blockIdx.x %d, blockIdx.y %d, blockIdx.z %d, chunk_idx %d, "
      //          "NUM_CHUNKS %d,  %d \n",
      //          blockIdx.x,
      //          blockIdx.y,
      //          blockIdx.z,
      //          chunk_idx,
      //          NUM_CHUNKS,
      //          size(SrcLayout{}));
      //   printf("\n");
      //   // print(size(shape(SrcLayout{})));
      //   print(size(SrcLayout{}));
      //   printf("\n");
      // }
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        // cute::print(SrcLayout{});
        // cute::print(DstLayout{});
        // cute::print(GMMA::Layout_MN_SW128_Atom<half>{});
      }
      size_t src_addr =
          (size_t)(src + src_layout(src_chunked_coord2coord(chunk_idx)));
      uint32_t dst_addr =
          dst_base_addr +
          dst_layout(dst_chunked_coord2coord(chunk_idx)) * sizeof(T);
      asm volatile(
          "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" ::"r"(dst_addr),
          "l"(src_addr));
    }
  }
};

// Type 4 : Copy using the Tensor Memory
//          Accelerator(TMA)
template <typename T, class DstLayout, class SrcLayout, class TMA>
class InputTMAAsyncCopy {
public:
  CUTE_STATIC_ASSERT_V(rank(SrcLayout{}) == rank(DstLayout{}));
  static constexpr int tmaTransactionBytes = sizeof(T) * size(DstLayout{});
  using CTA_TILER = decltype(shape(DstLayout{}));
  static __device__ __forceinline__ void run(TMA const &tma,
                                             T *dst,
                                             T const *src,
                                             uint64_t *tma_load_mbar,
                                             unsigned forloop_idx) {
    Tensor mA = tma.get_tma_tensor(shape(SrcLayout{}));

    auto cta_coord = make_coord(forloop_idx, blockIdx.x);
    auto cta_coord_t = make_coord(blockIdx.x, forloop_idx);

    Tensor gA =
        local_tile(mA,
                   CTA_TILER{},
                   size<1>(SrcLayout{}) == 2048 ? cta_coord_t : cta_coord,
                   Step<_1, _1>{});

    // Tensor gA = local_tile(mA, CTA_TILER{}, cta_coord);
    // Tensor gA = flat_divide(mA, shape(DstLayout{}));
    Tensor sA = make_tensor(make_smem_ptr(dst), DstLayout{});

    auto cta_tma = tma.get_slice(Int<0>{});  // CTA slice
    Tensor tAgA_x = cta_tma.partition_S(gA); // (TMA,TMA_M,TMA_N,REST_M,REST_N)
    Tensor tAsA_x = cta_tma.partition_D(sA); // (TMA,TMA_M,TMA_N)

    Tensor tAsA = group_modes<1, rank(tAsA_x)>(tAsA_x);
    Tensor tAgA = group_modes<1, rank(tAgA_x)>(tAgA_x); // (TMA,REST)
    if (thread0() && forloop_idx == 0) {
      printf("gA:   ");
      print(gA);
      print("\n");
      print("dst layout: ");
      print(DstLayout{});
      print("\n");
      print(SrcLayout{});
      print("\n");
    }
    if (threadIdx.x == 0) {
      tma_load_mbar[0] = 0;
      initialize_barrier(tma_load_mbar[0], 1);
      set_barrier_transaction_bytes(tma_load_mbar[0], tmaTransactionBytes);
      copy(tma.with(tma_load_mbar[0]), tAgA(_, 0), tAsA(_, 0));
    }
    __syncthreads();
    wait_barrier(tma_load_mbar[0], 0);
    __syncthreads();

    // if(threadIdx.x == 0){
    //   print("mA: \n");
    //   printf("(uintptr_t)ptr1 - (uintptr_t)ptr2; %lu\n", 0x7a79d5fffb80 -
    //   0x7a79aa604000); print("mbar %p\n", (void*)tma.get_tma_descriptor());
    //   // print("TMA desc     : "); print(tma.tma_desc_); print("\n");
    //   print("output dtensor %p", (void*)dtensor);
    //   print(mA);
    //   print("\n");
    //   print("coord: ");
    //   print(cta_coord);
    //   print("\n");
    //   print("tAgA_x: ");
    //   print(tAgA_x);
    //   print("\n");
    //   // print("\n");
    //   print("tAsA_x: ");
    //   print(tAsA_x);

    //   print("\n");
    //   print("tAsA: ");
    //   print(tAsA(_, 0));
    //   print("\n");
    //   print("tAgA: ");
    //   print(tAgA);
    //   print("\n");
    //  print("tgaaaa: ");
    //   print(tAgA(_, 0));
    //   print("\n");
    //   print("tma_load_mbar_x: ");
    //   print(tma_load_mbar[0]);

    // printf("Address of src: %p\n", (void*)src);
    //   printf("Address of dst: %p\n", (void*)dst);
    //   print("ma: ");
    //   print(mA);

    //   print(tma);
    //   print("\n");

    //   print("first ele of src: %f\n", (float)(src[0]));
    //   print("first ele of src: %f\n", (float)(src[64 * 64 - 1]));
    //   print("first ele of src: %f\n", (float)(src[64]));

    //   print("first ele of dst: %f\n", (float)(dst[0]));
    //   print("first ele of dst: %f\n", (float)(dst[64 * 64 - 1]));
    //   print("first ele of dst: %f\n", (float)(dst[64]));

    // }

    // int warp_idx = cutlass::canonical_warp_idx_sync();
    // int lane_predicate = cute::elect_one_sync();
  }
};

} // namespace tb
