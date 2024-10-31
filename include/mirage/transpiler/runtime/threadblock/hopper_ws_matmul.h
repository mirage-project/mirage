// matmul.h - Implementation of threadblock level matmul operators
//
// WARNING To understand code in this file, you have to be familiar with the
// CuTe library. The CuTe library is a part of the Cutlass library by NVIDIA
// which provides (not so) easy-to-use primitives for writing high-performance
// kernels. You may read the CuTe documentation at
// https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md
// or you can find some amazing blogs in Chinese here:
// https://zhuanlan.zhihu.com/p/661182311
//

#pragma once

#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <threadblock/matmul.h>
using namespace cute;

#include "element_unary.h"

namespace tb {

template <typename T,
          class SrcLayoutA,
          class SrcLayoutB,
          class TMA,
          class MMAAtom,
          class TiledMMAThrLayout,
          bool IS_LDMATRIX_AVAIL,
          bool IS_STMATRIX_AVAIL,
          class SmemLayoutA_, // [K, M]
          class SmemLayoutB_, // [N, K]
          class SmemLayoutC_, // [N, M]
          int NUM_THREADS,
          int NUM_EXPS_BEFORE_STORE, // Since matmul may use some advanced
                                     // instructions (like stmatrix) to store
                                     // data, it does not use the standard
                                     // "epilogue" semantic
          bool IS_STORE_ACCUM>
class Hopper_Input_Ws_Matmul {
public:
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutA_{}) == _2{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutB_{}) == _2{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutC_{}) == _2{});

  static_assert(NUM_THREADS / NumThreadsPerWarpGroup == _2{});

  using SmemLayoutA = typename Dim01Swapper<SmemLayoutA_>::Result; // [M, K]
  using SmemLayoutB = SmemLayoutB_;                                // [N, K]
  using SmemLayoutC = typename Dim01Swapper<SmemLayoutC_>::Result; // [M, N]

  using SmemAtomLayoutA = smem_layout_selector<size<0> SmemLayoutA{}>;
  using SmemAtomLayoutB = smem_layout_selector<size<0> SmemLayoutB{}>;

  // change it to dynamic;
  static constexpr int KStages = 3;

  // using Input_A = tb::InputTMAAsyncCopy<T, SmemLayoutA, SrcLayoutA, TMA_A>;
  // using Input_B = tb::InputTMAAsyncCopy<T, SmemLayoutB, SrcLayoutB, TMA_B>;

  // using MMA = tb::Hopper_Matmul<>;

  enum class WarpGroupRole {
    Producer = 0,
    Consumer = 1,
  };
  enum class ProducerWarpRole {
    MainloopEpilogue = 0,
    Warp1 = 1,
    Warp2 = 2,
    Warp3 = 3
  };

  // Shape checking
  // Expect A have a shape of [M, K], B have a shape of [N, K], and
  // C have a shape of [M, N]
  using M = decltype(get<0>(shape(SmemLayoutA{})));
  using K = decltype(get<1>(shape(SmemLayoutA{})));
  using N = decltype(get<0>(shape(SmemLayoutB{})));
  CUTE_STATIC_ASSERT_V(K{} == get<1>(shape(SmemLayoutB{})));
  CUTE_STATIC_ASSERT_V(M{} == get<0>(shape(SmemLayoutC{})));
  CUTE_STATIC_ASSERT_V(N{} == get<1>(shape(SmemLayoutC{})));

  // Mainloop Load pipeline, TODO xinhaoc change the stages
  using MainloopPipeline = cutlass::PipelineTmaAsync<_1{}>;

  struct SharedStorage {

    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename MainloopPipeline::SharedStorage;

      alignas(16) MainloopPipelineStorage mainloop;
    } pipelines;
  };

  using TiledMMA = decltype(make_tiled_mma(
      SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{}));

  static constexpr int TILED_MMA_NUM_THREADS = thr_size(TiledMMA{});
  static_assert(TILED_MMA_NUM_THREADS <= NUM_THREADS);

  using R2STiledCopyCSelector =
      R2STiledCopySelector<T, IS_STMATRIX_AVAIL, SmemLayoutC>;
  using R2STiledCopyCAtom = typename R2STiledCopyCSelector::Result;
  static constexpr R2STiledCopyType R2S_TILED_COPY_C_TYPE =
      R2STiledCopyCSelector::TYPE;
  using R2STiledCopyC =
      decltype(make_tiled_copy_C(R2STiledCopyCAtom{}, TiledMMA{}));

  static __device__ __forceinline__ auto get_mma_rC(int thread_idx) {
    // Make a fake tensor
    Tensor sC_fake =
        make_tensor(make_smem_ptr((half_t *)nullptr), SmemLayoutC{});
    TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(thread_idx);
    Tensor mma_rC =
        thr_mma.partition_fragment_C(sC_fake); // (MMA, MMA_M, MMA_N)
    clear(mma_rC);
    return mma_rC;
  }

  template <class AccumRegFrag>
  static __device__ __forceinline__ void write_back_mma_rC(
      T *__restrict__ c_ptr, AccumRegFrag const &mma_rC, int thread_idx) {
    if (thread_idx >= TILED_MMA_NUM_THREADS) {
      return;
    }
    Tensor sC = make_tensor(make_smem_ptr(c_ptr), SmemLayoutC{}); // [M, N]
    R2STiledCopyC r2s_tiled_copy_C;
    ThrCopy r2s_tiled_copy_C_thr = r2s_tiled_copy_C.get_slice(thread_idx);
    Tensor r2s_rC =
        r2s_tiled_copy_C_thr.retile_S(mma_rC);            // (R2S, R2S_M, R2S_N)
    Tensor r2s_sC = r2s_tiled_copy_C_thr.partition_D(sC); // (R2S, R2S_M, R2S_N)
    r2s_copy_with_oob_protection<T,
                                 M,
                                 N,
                                 NUM_EXPS_BEFORE_STORE,
                                 IS_STORE_ACCUM>(
        r2s_tiled_copy_C, r2s_rC, r2s_sC, thread_idx);
  }

  template <typename MMARc>
  static __device__ __forceinline__ void
      run(MMARc &mma_rC,
          TMA const &tma_load_a,
          TMA const &tma_load_b,
          T *__restrict__ a_ptr, // Do not define a_ptr and b_ptr as const here,
                                 // since we may pad remaining part on the
                                 // k-axis with 0
          T *__restrict__ b_ptr,
          char const *__restrict__ smem_allzero_ptr,
          int thread_idx) {
    if (thread_idx >= TILED_MMA_NUM_THREADS) {
      return;
    }

    auto k_tile_iter = cute::make_coord_iterator(shape<2>(SrcLayoutA{}));
    auto k_tile_count = size<2>(SrcLayoutA{});

    // Shared memory;
    SharedStorage &shared_storage =
        *reinterpret_cast<SharedStorage *>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int lane_idx = cutlass::canonical_lane_idx();
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    typename MainloopPipeline::Params mainloop_pipeline_params;

    if (warp_group_role == WarpGroupRole::Producer &&
        producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
      mainloop_pipeline_params.role =
          MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
      mainloop_pipeline_params.role =
          MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = NumThreadsPerWarpGroup;
    mainloop_pipeline_params.transaction_bytes =
        params.mainloop.tma_transaction_bytes;

    // cluster size
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop,
                                       mainloop_pipeline_params,
                                       Int<_1, _1, _1>{});

    // Copy
    if (warp_group_role == WarpGroupRole::Producer) {
      if (producer_warp_role == ProducerWarpRole::MainloopEpilogue) {

        for (; k_tile_count > 0; --k_tile_count) {
          // init barrier
          mainloop_pipeline.producer_acquire(smem_pipe_write);
          using BarrierType = typename MainloopPipeline::ProducerBarrierType;
          BarrierType *tma_barrier =
              mainloop_pipeline.producer_get_barrier(smem_pipe_write);

          int write_stage = smem_pipe_write.index();

          auto cta_tma_a = tma_load_a.get_slice(Int<0>{}); // CTA slice
          Tensor tAgA_x =
              cta_tma_a.partition_S(gA); // (TMA,TMA_M,TMA_N,REST_M,REST_N)
          Tensor tAsA_x = cta_tma_a.partition_D(sA); // (TMA,TMA_M,TMA_N)

          auto cta_tma_b = tma_load_a.get_slice(Int<0>{}); // CTA slice
          Tensor tBgB_x =
              cta_tma_b.partition_S(gA); // (TMA,TMA_M,TMA_N,REST_M,REST_N)
          Tensor tBsB_x = cta_tma_b.partition_D(sA); // (TMA,TMA_M,TMA_N)

          Tensor tAsA = group_modes<1, rank(tAsA_x)>(tAsA_x);
          Tensor tAgA = group_modes<1, rank(tAgA_x)>(tAgA_x); // (TMA,REST)

          Tensor tBsB = group_modes<1, rank(tAsB_x)>(tBsB_x);
          Tensor tBgB = group_modes<1, rank(tAgB_x)>(tBgB_x); // (TMA,REST)

          copy(tma_load_a.with(*tma_barrier),
               tAgA(_, _, *k_tile_iter),
               tAsA(_, _, write_stage));
          copy(mainloop_params.tma_load_b.with(*tma_barrier),
               tBgB(_, _, *k_tile_iter),
               tBsB(_, _, write_stage));

          ++smem_pipe_write;
        }

        // what's this
        // mainloop_pipe_producer_state.advance(k_tile_count);
        //   // Make sure mainloop consumer has been waited upon before issuing
        //   epilogue load collective_mainloop.load_tail(mainloop_pipeline,
        //   mainloop_pipe_producer_state);
      }
    } else if (warp_group_role == WarpGroupRole::Consumer) {
      // MMA
      PipelineState smem_pipe_release = smem_pipe_read;

      auto barrier_token = mainloop_pipeline.consumer_try_wait(smem_pipe_read);
      mainloop_pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();
      warpgroup_arrive();

      // gemm

      TiledMMA tiled_mma;

      // make stages here

      auto sA_l = tile_to_shape(
          SmemAtomLayoutA{},
          make_shape(size<0> SmemLayoutA{}, size<1> SmemLayoutA{}, KStages));
      auto sB_l = tile_to_shape(
          SmemAtomLayoutB{},
          shape(size<0> SmemLayoutB{}, size<1> SmemLayoutB{}, KStages));

      Tensor sA =
          make_tensor(make_smem_ptr(a_ptr), sA_l); // [TILE_M, TILE_K, KStage]
      Tensor sB =
          make_tensor(make_smem_ptr(b_ptr), sB_l); // [TILE_N, TILE_K, KStage]

      ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
      Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
      Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)

      Tensor tCrA = thr_mma.make_fragment_A(tCsA); // (MMA,MMA_M,MMA_K,PIPE)
      Tensor tCrB = thr_mma.make_fragment_B(tCsB); // (MMA,MMA_N,MMA_K,PIPE)

      CUTE_STATIC_ASSERT_V(Int<KStages>{} == size<2>(sA)); // PIPE
      CUTE_STATIC_ASSERT_V(Int<KStages>{} == size<2>(sB)); // PIPE

      // auto k_tile_count = 1;

      CUTE_UNROLL
      for (int i_k = 0; i_k < k_tile_count; ++i_k) {
        gemm(tiled_mma,
             mma_rC,
             tCrA(_, _, read_stage),
             tCrB(_, _, read_stage),
             mma_rC);
      }

      warpgroup_commit_batch();
      warpgroup_wait<0>();
      mainloop_pipeline.consumer_release(smem_pipe_release);

      ++smem_pipe_read;
    }
  }
};

} // namespace tb
