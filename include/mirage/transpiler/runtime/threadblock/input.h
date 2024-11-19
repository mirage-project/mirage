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
#include "cutlass/pipeline/pipeline.hpp"
#include "hopper_mainloop_params.h"
#include "cutlass/gemm/gemm.h"
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

// static __device__ __forceinline__ auto tma_make_coord(int forloop_dim,
//                                                       int forloop_idx,
//                                                       int imap_x,
//                                                       int imap_y,
//                                                       int imap_z,
//                                                       int rank,
//                                                       int *res) {

//   assert(rank == 2);
//   res[0] = -1;
//   res[1] = -1;
//   res[2] = -1;
//   if (forloop_dim == 0) {
//     res[0] = forloop_idx;
//   } else if (forloop_dim == 1) {
//     res[1] = forloop_idx;
//   } else if (forloop_dim == 2) {
//     res[2] = forloop_idx;
//   }

//   if (imap_x != -1) {
//     int div_dim = imap_x == 0 ? 0 : imap_x == 1 ? 1 : 2;
//     res[div_dim] =
//         imap_x == forloop_dim ? forloop_idx * blockIdx.x : blockIdx.x;
//   }
//   if (imap_y != -1) {
//     int div_dim = imap_y == 0 ? 0 : imap_y == 1 ? 1 : 2;
//     res[div_dim] =
//         imap_y == forloop_dim ? forloop_idx * blockIdx.y : blockIdx.y;
//   }
//   if (imap_z != -1) {
//     // z = imap_z == forloop_dim ? forloop_idx * blockIdx.y : blockIdx.y;
//     int div_dim = imap_z == 0 ? 0 : imap_z == 1 ? 1 : 2;
//     res[div_dim] =
//         imap_z == forloop_dim ? forloop_idx * blockIdx.z : blockIdx.z;
//   }

//   // if (thread0()) {
//   //   printf("imap xyz %d %d %d\n", imap_x, imap_y, imap_z);
//   //   printf("xyz %d %d %d\n", res[0], res[1], res[2]);
//   // }

//   if (res[0] >= 0 && res[1] >= 0) {
//     return make_coord(res[0], res[1]);
//   } else if (res[0] >= 0 && res[2] >= 0) {
//     assert(false);
//   } else if (res[1] >= 0 && res[2] >= 0) {
//     assert(false);
//   } else if (res[0] >= 0) {
//     return make_coord(res[0], 0);
//   } else if (res[1] >= 0) {
//     return make_coord(0, res[1]);
//   } else if (res[2] >= 0) {
//     assert(false);
//   } else {
//     return make_coord(0, 0);
//   }

//   return make_coord(0, 0);
// }

// static __device__ __forceinline__ auto tma_make_coord(int imap_x,
//                                                       int imap_y,
//                                                       int imap_z) {

//   if(imap_x != 0){
//     return make_coord(blockIdx.x, _);
//   }

//   return make_coord(_, _);
// }
// Type 4 : Copy using the Tensor Memory
//          Accelerator(TMA)
template <typename T, class DstLayout, class SrcLayout, class TMA, class MainloopPipeline, class PipelineState>
class InputTMAAsyncCopy {
public:
  CUTE_STATIC_ASSERT_V(rank(SrcLayout{}) == rank(DstLayout{}));
  static constexpr int tmaTransactionBytes = sizeof(T) * size(DstLayout{});
  using CTA_TILER = decltype(shape(DstLayout{}));

  // using CTA_TILER = decltype(replace<0>(SrcLayout{}, shape<1>(DstLayout{})));
  // using CTA_TILER = decltype(make_shape(shape<0>(SrcLayout{}), shape<1>(DstLayout{})));

  //tile shape = 64, 4096
  //TODO, change it
  using SrcPipeLayput = decltype(make_shape(shape<0>(SrcLayout{}), shape<0>(DstLayout{})));
  
  using DstPipeLayout = decltype(tile_to_shape(
        DstLayout{},
        make_shape(shape<0>(DstLayout{}), shape<1>(DstLayout{}), Int<kStages>{}), Step<_1, _2, _3>{}));
  // 4096, 64, 2
  static __device__ __forceinline__ void prefetch(TMA const &tma){
    cute::prefetch_tma_descriptor(tma.get_tma_descriptor());
  }

  static __device__ __forceinline__ void run(TMA const &tma,
                                             T *dst,
                                             T const *src,
                                             MainloopPipeline pipeline,
                                             PipelineState &smem_pipe_write,
                                             unsigned k_tile_count,
                                             unsigned forloop_dim,
                                             int imapx,
                                             int imapy,
                                             int imapz,
                                             int warp_idx, tb::WarpGroupRole warp_group_role, int warp_idx_in_warp_group, int warp_group_thread_idx, int lane_predicate) {

    if(lane_predicate){
      // Tensor mA = tma.get_tma_tensor(append<3>(shape(SrcLayout{}), Int<1>{}));

      Tensor mA = tma.get_tma_tensor(shape(SrcPipeLayput{}));
      // Coord<2> cta_coord = imapx == -1 ? make_coord(_, _) : make_coord(blockIdx.x, _); 
      
      // auto cta_coord = make_coord(0, 0);
      Tensor gA = local_tile(mA, CTA_TILER{}, make_coord(_, _)); 
      // Tensor gA = local_tile(mA, CTA_TILER{}, cta_coord);
      Tensor sA = make_tensor(make_smem_ptr(dst), DstPipeLayout{});

      auto cta_tma = tma.get_slice(Int<0>{});  // CTA slice
      Tensor tAgA_x = cta_tma.partition_S(gA); // (TMA,TMA_M,TMA_N,REST_M,REST_N)
      Tensor tAsA_x = cta_tma.partition_D(sA); // (TMA,TMA_M,TMA_N)
      Tensor tAsA = group_modes<1, rank(tAsA_x)>(tAsA_x);
      Tensor tAgA = group_modes<1, rank(tAgA_x)>(tAgA_x); // (TMA,REST)

        
      if(blockIdx.x == 0){
        // print("sdsdsds: ");
        // print(warp_idx);
        // print("\n");
        // print(warp_idx_in_warp_group);
        // print("\n");
        // print("mA: ");
        // print(mA);
        // print("\n");

        // print("gA: ");
        // print(gA);
        // print("\n");

        // print("sA: ");
        // print(sA);
        // print("\n");


        // print("tAgA: ");
        // print(tAgA);
        // print("\n");

        // print("tAsA: ");
        // print(tAsA);
        // print("\n");

        // print("tma");
        // print(tma);
        // print("\n");
      }
      auto k_tile_iter  = cute::make_coord_iterator(k_tile_count);
      CUTLASS_PRAGMA_NO_UNROLL
      for (; k_tile_count > 0; --k_tile_count){
        pipeline.producer_acquire(smem_pipe_write);
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType *tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();

        // printf("write stage: %d\n", write_stage);
        // copy(tma.with(*tma_barrier), tAgA(_, _, *k_tile_iter), tAsA(_, _, write_stage));
        ++k_tile_iter;

        // Advance smem_pipe_write
        ++smem_pipe_write;
      }

      if(lane_predicate){
          pipeline.producer_tail(smem_pipe_write);
      }

          // int coord_value[3];
          // auto cta_coord = make_coord(0, 0);
          // coord_value);
          // auto cta_coord = tma_make_coord(forloop_dim,
          //                                 forloop_idx,
          //                                 imapx,
          //                                 imapy,
          //                                 imapz,
          //                                 rank(SrcLayout{}),
          //                                 coord_value);

          

          // Tensor sA = make_tensor(make_smem_ptr(dst), DstLayout{});

          // auto cta_tma = tma.get_slice(Int<0>{});  // CTA slice
          // Tensor tAgA_x = cta_tma.partition_S(gA); // (TMA,TMA_M,TMA_N,REST_M,REST_N)
          // Tensor tAsA_x = cta_tma.partition_D(sA); // (TMA,TMA_M,TMA_N)

          // Tensor tAsA = group_modes<1, rank(tAsA_x)>(tAsA_x);
          // Tensor tAgA = group_modes<1, rank(tAgA_x)>(tAgA_x); // (TMA,REST)
          // // if (threadIdx.x == 0) {
          // //   tma_load_mbar[0] = 0;
          // //   initialize_barrier(tma_load_mbar[0], 1);
          // //   set_barrier_transaction_bytes(tma_load_mbar[0], tmaTransactionBytes);
          // //   copy(tma.with(tma_load_mbar[0]), tAgA, tAsA);
          // // }
          // // __syncthreads();
          // // wait_barrier(tma_load_mbar[0], 0);
    }
    
  }
};

} // namespace tb
