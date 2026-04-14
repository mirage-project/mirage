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

#include "cute/arch/cluster_sm100.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/gemm/gemm.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "utils.h"
#include <cstdint>
#include <cute/layout.hpp>
#include <cutlass/arch/reg_reconfig.h>
using namespace cute;

namespace tb {

template <class InputLayout>
class InputDim01Swapper {
  CUTE_STATIC_ASSERT_V(rank(InputLayout{}) == _2{});

  using A0 = decltype(get<0>(shape(InputLayout{})));
  using A1 = decltype(get<1>(shape(InputLayout{})));
  using TransposeCoordLayout = Layout<Shape<A1, A0>, Stride<A0, _1>>;
  using Result_ = decltype(composition(InputLayout{}, TransposeCoordLayout{}));

public:
  using Result =
      decltype(coalesce(Result_{}, Step<_1, _1>{})); // By-mode coalescing
};

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
    for (int chunk_idx = thread_idx; chunk_idx < NUM_CHUNKS; chunk_idx += 128) {
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

template <typename T,
          class DstLayout,
          class SrcLayout,
          class TMA,
          // class MainloopPipeline,
          // class PipelineState,
          class HopperAsyncPipeline,
          bool MInput,
          int K_ITER>
class InputTMAAsyncCopy {
public:
  using CTA_TILER =
      decltype(make_shape(shape<0>(DstLayout{}), shape<1>(DstLayout{})));

  static constexpr cute::GMMA::Major GmmaMajor = GMMA::Major::MN;
  using DstMNKLayout = DstLayout;
  using SmemLayoutAtom =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GmmaMajor,
               half_t,
               decltype(get<0>(DstMNKLayout{})),
               decltype(get<1>(DstMNKLayout{}))>());

  // A, B, X, Y, Z, Stage
  using DstPipeLayout =
      decltype(tile_to_shape(SmemLayoutAtom{},
                             make_shape(shape<0>(DstMNKLayout{}),
                                        shape<1>(DstMNKLayout{}),
                                        Int<HopperAsyncPipeline::Stage>{})));

  static constexpr int tmaTransactionBytes =
      sizeof(T) * size(DstPipeLayout{}) / HopperAsyncPipeline::Stage;

  static __device__ __forceinline__ void prefetch(TMA const &tma) {
    cute::prefetch_tma_descriptor(tma.get_tma_descriptor());
  }

  template <int SrcLayoutSize>
  static __device__ auto
      make_coord_runtime(int imapx_a, int imapy_a, int imapz_a) {
    if constexpr (SrcLayoutSize == 6) {
      return make_coord(_,
                        _,
                        imapx_a >= 0 ? blockIdx.x : 0,
                        imapy_a >= 0 ? blockIdx.y : 0,
                        imapz_a >= 0 ? blockIdx.z : 0,
                        _);
    } else if constexpr (SrcLayoutSize == 7) {
      return make_coord(_,
                        _,
                        _,
                        imapx_a >= 0 ? blockIdx.x : 0,
                        imapy_a >= 0 ? blockIdx.y : 0,
                        imapz_a >= 0 ? blockIdx.z : 0,
                        _);
    } else {
      static_assert(SrcLayoutSize == 6 || SrcLayoutSize == 7,
                    "Unsupported layout size");
    }
  }

  static __device__ __forceinline__ void run(TMA const &tma_a,
                                             T *dst_a,
                                             int imapx_a,
                                             int imapy_a,
                                             int imapz_a,
                                             int k_tile_iter,
                                             HopperAsyncPipeline &pipeline) {
    if (lane_id() == 0) {
      Tensor mA = tma_a.get_tma_tensor(shape(SrcLayout{}));
      // （CTA_M, CTA_K, X, Y, Z, FORLOOP）
      auto blkCoordA = make_coord_runtime<decltype(rank(SrcLayout{}))::value>(
          imapx_a, imapy_a, imapz_a);
      // auto blkCoordA = make_coord(_,
      //                             _,
      //                             imapx_a >= 0 ? blockIdx.x : 0,
      //                             imapy_a >= 0 ? blockIdx.y : 0,
      //                             imapz_a >= 0 ? blockIdx.z : 0,
      //                             _);

      Tensor gA = mA(blkCoordA);

      Tensor sA = make_tensor(make_smem_ptr(dst_a), DstPipeLayout{});
      auto cta_tma_a = tma_a.get_slice(Int<0>{}); // CTA slice

      Tensor tAgA = cta_tma_a.partition_S(gA);
      Tensor tAsA = cta_tma_a.partition_D(sA);
      Tensor tAgAX = group_modes<0, rank(tAgA) - 1>(tAgA); // REST, Forloop
      Tensor tAsAX = group_modes<0, rank(tAsA) - 1>(tAsA);

      auto [tma_barrier, write_stage] = pipeline.producer_acquire();
      copy(tma_a.with(*tma_barrier),
           tAgAX(_, k_tile_iter),
           tAsAX(_, write_stage));
      pipeline.producer_advance();
    }
  }
};

// Blackwell
template <typename T,
          class DstLayout,
          class SrcLayout,
          class TMA,
          class BlackwellAsyncPipeline,
          bool MInput,
          int K_ITER,
          class TiledMMA_,
          class Mma_Tiler_,
          class ClusterShape_MNK_>
class InputTMAAsyncCopy_Blackwell {

  using TiledMMA = TiledMMA_;
  using Mma_Tiler = Mma_Tiler_;
  using ClusterShape = ClusterShape_MNK_;

  static constexpr UMMA::Major UMMAMajor =
      MInput ? UMMA::Major::K : UMMA::Major::MN;

  using SmemLayoutAtom =
      decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
               UMMAMajor,
               T,
               std::conditional_t<MInput,
                                  decltype(get<0>(Mma_Tiler{})),
                                  decltype(get<1>(Mma_Tiler{}))>,
               decltype(get<2>(Mma_Tiler{}))>());

  using DstMNKLayout = std::conditional_t<
      MInput,
      decltype(partition_shape_A(
          TiledMMA{},
          make_shape(shape<0>(Mma_Tiler{}), shape<2>(Mma_Tiler{})))),
      decltype(partition_shape_B(
          TiledMMA{},
          make_shape(shape<1>(Mma_Tiler{}), shape<2>(Mma_Tiler{}))))>;

  using DstPipeLayout = decltype(UMMA::tile_to_mma_shape(
      SmemLayoutAtom{},
      append(DstMNKLayout{}, Int<BlackwellAsyncPipeline::Stage>{}),
      std::conditional_t<MInput, Step<_1, _2, _3>, Step<_2, _1, _3>>{}));

public:
  static __device__ __forceinline__ void prefetch(TMA const &tma) {
    cute::prefetch_tma_descriptor(tma.get_tma_descriptor());
  }

  static __device__ __forceinline__ void
      run(TMA const &tma_a,
          T *dst_smem, // SMEM destination address
          TiledMMA const &tiled_mma,
          Mma_Tiler const &mma_tiler,
          int k_iter,
          BlackwellAsyncPipeline &pipeline) {

    if (lane_id() == 0) {
      Tensor mA = tma_a.get_tma_tensor(shape(SrcLayout{}));

      Tensor tCsA = make_tensor(make_smem_ptr(dst_smem), DstPipeLayout{});

      TiledMMA tiled_mma;
      Layout cluster_layout_vmnk = get_cluster_layout<TiledMMA, ClusterShape>();
      auto cta_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
          int(cute::block_rank_in_cluster()));

      auto mma_coord_vmnk =
          get_mma_coord_vmnk<TiledMMA, ClusterShape>(blockIdx.x, blockIdx.y);
      auto mma_coord = select<1, 2, 3>(mma_coord_vmnk);
      decltype(auto) gA = [&]() {
        if constexpr (MInput) {
          return local_tile(
              mA,
              mma_tiler,
              mma_coord,
              Step<_1, X, _1>{}); // (MmaTile_M, MmaTile_K, Tiles_K)
        } else {
          return local_tile(
              mA,
              mma_tiler,
              mma_coord,
              Step<X, _1, _1>{}); // (MmaTile_M, MmaTile_K, Tiles_K)
        }
      }();

      auto mma_v = get<0>(mma_coord_vmnk);
      ThrMMA cta_mma = tiled_mma.get_slice(mma_v); // Use Peer CTA coordinate

      decltype(auto) tCgA = [&]() {
        if constexpr (MInput) {
          return cta_mma.partition_A(gA); // (MmaA, NumMma_M, NumMma_K, Tiles_K)
        } else {
          return cta_mma.partition_B(gA); // (MmaA, NumMma_M, NumMma_K, Tiles_K)
        }
      }();

      // Project the cluster_layout for tma_A along the N-modes
      uint16_t tma_mcast_mask;
      if constexpr (MInput) {
        tma_mcast_mask = create_tma_multicast_mask<2>(
            cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
      } else {
        tma_mcast_mask = create_tma_multicast_mask<1>(
            cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
      }

      auto [tAgAX, tAsAX] = [&]() {
        if constexpr (MInput) {
          return tma_partition(
              tma_a,
              get<2>(cta_in_cluster_coord_vmnk), // The CTA coordinate along N
                                                 // mode of the cluster
              make_layout(
                  size<2>(cluster_layout_vmnk)), // The CTA layout along N mode
                                                 // of the cluster
              group_modes<0, rank(tCsA) - 1>(tCsA),
              group_modes<0, rank(tCgA) - 1>(tCgA));
        } else {
          return tma_partition(
              tma_a,
              get<1>(cta_in_cluster_coord_vmnk), // The CTA coordinate along M
                                                 // mode of the cluster
              make_layout(
                  size<1>(cluster_layout_vmnk)), // The CTA layout along M mode
                                                 // of the cluster
              group_modes<0, rank(tCsA) - 1>(tCsA),
              group_modes<0, rank(tCgA) - 1>(tCgA));
        }
      }();

      auto [tma_barrier, write_stage] = pipeline.producer_acquire();

      copy(tma_a.with(*tma_barrier, tma_mcast_mask),
           tAgAX(_, k_iter),
           tAsAX(_, write_stage));

      pipeline.producer_advance();
    }
  }
};

} // namespace tb
