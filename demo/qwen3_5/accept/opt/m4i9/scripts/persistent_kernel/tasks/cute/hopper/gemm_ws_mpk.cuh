/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "cutlass/arch/mma_sm90.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/conv/detail.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/tensor.hpp"

#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/arch/memory.h"

// MPK Settings
#include "../../hopper/smem_layout_tma.cuh"
#include "../../hopper/tma.cuh"
#include "../../hopper/utils.cuh"
#include "epilogue.cuh"
#include "kernel_traits.cuh"
#include "mma_tma_ws_mainloop.cuh"

namespace kernel {

using namespace cute;

template <class CollectiveMainloop,
          class CollectiveEpilogue,
          bool TMAOnHost,
          typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          typename TMA_A,
          typename TMA_B,
          int OUTPUT_STRIDE = OUTPUT_SIZE,
          bool WITH_RESIDUAL = false>
CUTLASS_DEVICE void linear_cutlass_ws_hopper(const TMA_A &tma_a,
                                             const TMA_B &tma_b,
                                             void *output_ptr,
                                             void const *residual_ptr) {

  struct SharedStorage {
    // Mainloop and epilogue don't use smem concurrently since kernel is
    // non-persistent, so we can use a union
    union TensorStorage {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;

    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage =
          typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage =
          typename CollectiveEpilogue::PipelineStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
    } pipelines;
  };

  enum class WarpGroupRole {
    Producer = 1,
    Consumer = 0,
  };
  enum class ProducerWarpRole {
    MainloopEpilogue = 0,
    Warp1 = 1,
    Warp2 = 2,
    Warp3 = 3
  };

  extern __shared__ char smem[];
  uintptr_t aligned_smem =
      (reinterpret_cast<uintptr_t>(smem) + 1023) / 1024 * 1024;

  using ClusterShape = typename CollectiveMainloop::ClusterShape;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using TileShape = typename CollectiveMainloop::TileShape;
  using SmemLayoutA =
      typename CollectiveMainloop::SmemLayoutA; // (BLK_M,BLK_K,PIPE)
  using SmemLayoutB =
      typename CollectiveMainloop::SmemLayoutB; // (BLK_N,BLK_K,PIPE)

  // Kernel level shared memory storage
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(aligned_smem);
  T *d_output = static_cast<T *>(output_ptr);
  T const *d_residual = static_cast<T const *>(residual_ptr);

  cutlass::bfloat16_t *shared_weight =
      shared_storage.tensors.mainloop.smem_A.begin();
  cutlass::bfloat16_t *shared_input =
      shared_storage.tensors.mainloop.smem_B.begin();
  //  T_ *mm_output = shared_storage.tensors.epilogue.begin();

  constexpr int INPUT_TMA_TILE_SIZE = 64;
  constexpr int WEIGHT_TMA_TILE_SIZE = INPUT_TMA_TILE_SIZE;
  constexpr int OUTPUT_ATOM_SIZE = 64; // this is padded if OUTPUT_SIZE < 64
  constexpr int SWIZZLE_B = 3, SWIZZLE_M = 3, SWIZZLE_S = 3;
  constexpr int TILE_SIZE = 128;

  // NOTE(Yu): Assume batch size is smaller than 16, and padding the batch size
  // to 16
  static_assert(BATCH_SIZE <= 16,
                "Batch size must be smaller or equal to 16 in swapAB");
  constexpr int SMEM_M_SIZE = BATCH_SIZE;
  using InputSmem = smem_tma<cutlass::bfloat16_t,
                             SWIZZLE_B,
                             SWIZZLE_M,
                             SWIZZLE_S,
                             SMEM_M_SIZE,
                             INPUT_TMA_TILE_SIZE,
                             TILE_SIZE / INPUT_TMA_TILE_SIZE>;
  InputSmem input_smem(shared_input);

  using WeightSmem = smem_tma<cutlass::bfloat16_t,
                              SWIZZLE_B,
                              SWIZZLE_M,
                              SWIZZLE_S,
                              OUTPUT_ATOM_SIZE,
                              WEIGHT_TMA_TILE_SIZE,
                              TILE_SIZE / WEIGHT_TMA_TILE_SIZE>;
  WeightSmem input_weight_smem(shared_weight);

  int thread_idx = int(threadIdx.x);
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
  int warp_group_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
  auto warp_group_role = WarpGroupRole(cutlass::canonical_warp_group_idx());
  auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
  int lane_predicate = cute::elect_one_sync();

  // Issue Tma Descriptor Prefetch from a single thread
  if ((warp_idx == 0) && lane_predicate) {
    prefetch_tma_descriptor(tma_a.desc_ptr);
    prefetch_tma_descriptor(tma_b.desc_ptr);

    //  NOTE(Yu): prefetch epilogue tma descriptor if needed
    // CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
  }

  // Mainloop Load pipeline
  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  typename MainloopPipeline::Params mainloop_pipeline_params;
  if (warp_group_role == WarpGroupRole::Producer &&
      producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
    mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
  }
  if (warp_group_role == WarpGroupRole::Consumer) {
    mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  }

  mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
  mainloop_pipeline_params.num_consumers = cutlass::NumThreadsPerWarpGroup;
  // compute tma transaction bytes
  static constexpr uint32_t TmaTransactionBytesMK =
      size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * sizeof(T);
  static constexpr uint32_t TmaTransactionBytesNK =
      size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * sizeof(T);
  static constexpr uint32_t TmaTransactionBytes =
      TmaTransactionBytesMK + TmaTransactionBytesNK;

  mainloop_pipeline_params.transaction_bytes = TmaTransactionBytes;
  // mainloop_pipeline_params.transaction_bytes =
  //     mainloop_params.tma_transaction_bytes;
  MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop,
                                     mainloop_pipeline_params,
                                     ClusterShape{});

  // Epilogue Load pipeline
  // using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
  // typename EpiLoadPipeline::Params epi_load_pipeline_params;
  // if (warp_group_role == WarpGroupRole::Producer &&
  //     producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
  //   epi_load_pipeline_params.role =
  //   EpiLoadPipeline::ThreadCategory::Producer;
  // }
  // if (warp_group_role == WarpGroupRole::Consumer) {
  //   epi_load_pipeline_params.role =
  //   EpiLoadPipeline::ThreadCategory::Consumer;
  // }
  // epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
  // epi_load_pipeline_params.producer_arv_count = cutlass::NumThreadsPerWarp;
  // epi_load_pipeline_params.consumer_arv_count =
  // cutlass::NumThreadsPerWarpGroup; if constexpr
  // (CollectiveEpilogue::RequiresTransactionBytes) {
  // epi_load_pipeline_params.transaction_bytes =
  //     epilogue_params.tma_transaction_bytes;
  // }
  // EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load,
  //                                   epi_load_pipeline_params);
  // Epilogue Store pipeline
  // using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
  // typename EpiStorePipeline::Params epi_store_pipeline_params;
  // epi_store_pipeline_params.always_wait = true;
  // EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);
  // Initialize starting pipeline states for the collectives
  // Epilogue store pipe is producer-only (consumer is TMA unit, waits via
  // scoreboarding)
  typename CollectiveMainloop::PipelineState smem_pipe_read;
  typename CollectiveMainloop::PipelineState smem_pipe_release = smem_pipe_read;
  // typename CollectiveEpilogue::LoadPipelineState
  // epi_load_pipe_consumer_state;

  // For the DMA Load (producer) we start with an opposite phase
  // i.e., we skip all waits since we know that the buffer is indeed empty
  cutlass::PipelineState mainloop_pipe_producer_state =
      cutlass::make_producer_start_state<MainloopPipeline>();
  // cutlass::PipelineState epi_load_pipe_producer_state =
  //     cutlass::make_producer_start_state<EpiLoadPipeline>();
  // cutlass::PipelineState epi_store_pipe_producer_state =
  //     cutlass::make_producer_start_state<EpiStorePipeline>();
  auto blk_shape = TileShape{}; // (BLK_M,BLK_N,BLK_K)
  TiledMma tiled_mma;

  constexpr auto problem_shape_mnkl = make_shape(cute::Int<OUTPUT_SIZE>{},
                                                 cute::Int<BATCH_SIZE>{},
                                                 cute::Int<REDUCTION_SIZE>{},
                                                 cute::Int<1>{});

  auto blk_coord = make_coord(Int<0>{}, Int<0>{}, _, Int<0>{});

  // Get pipeline iterators and increments from tensor shapes
  constexpr int NUM_ITERS_K = (REDUCTION_SIZE + TILE_SIZE - 1) / TILE_SIZE;
  auto k_tile_count = NUM_ITERS_K;

  // Wait for all thread blocks in the Cluster
  __syncthreads();

  if (warp_group_role == WarpGroupRole::Producer) {
    if (producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();
      int lane_predicate = cute::elect_one_sync();

      if (lane_predicate) {
        // Mainloop
        CUTLASS_PRAGMA_NO_UNROLL
        for (int k_iter = 0; k_iter < k_tile_count; k_iter++) {
          // LOCK smem_pipe_write for _writing_
          mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state);

          //
          // Copy gmem to smem for *k_iter
          //

          using BarrierType = typename MainloopPipeline::ProducerBarrierType;
          BarrierType *tma_barrier = mainloop_pipeline.producer_get_barrier(
              mainloop_pipe_producer_state);

          int write_stage = mainloop_pipe_producer_state.index();
          int tma_coords_A[2] = {k_iter * TILE_SIZE, 0 * OUTPUT_ATOM_SIZE};
          int tma_coords_B[2] = {k_iter * TILE_SIZE, 0};
          input_weight_smem.set_ptr(shared_weight +
                                    write_stage * OUTPUT_ATOM_SIZE * TILE_SIZE);
          input_smem.set_ptr(shared_input +
                             write_stage * SMEM_M_SIZE * TILE_SIZE);
          tma_a.tma_cp_async(
              *tma_barrier, input_weight_smem(0, 0), tma_coords_A);
          tma_b.tma_cp_async(*tma_barrier, input_smem(0, 0), tma_coords_B);

          // Advance smem_pipe_write
          ++mainloop_pipe_producer_state;
        }
      }

      // Issue the epilogue waits
      if (lane_predicate) {
        //  pipeline.producer_tail(smem_pipe_write);
        mainloop_pipeline.producer_tail(mainloop_pipe_producer_state);
      }
    }
  } else if (warp_group_role == WarpGroupRole::Consumer) {
    Tensor accum = partition_fragment_C(
        tiled_mma, take<0, 2>(blk_shape)); // (MMA,MMA_M,MMA_N)

    Tensor sA = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_A.data()),
        SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_B.data()),
        SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
    constexpr int MmaWarpGroups =
        size(TiledMma{}) / cutlass::NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(
        Int<MmaWarpGroups>{}, Int<cutlass::NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(
        0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
    auto thread_mma =
        tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    Tensor tCsA = thread_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB); // (MMA,MMA_N,MMA_K,PIPE)
    //
    // PIPELINED MAIN LOOP
    //
    static_assert(
        (0 <= CollectiveMainloop::K_PIPE_MMAS) &&
            (CollectiveMainloop::K_PIPE_MMAS < CollectiveMainloop::NUM_STAGES),
        "ERROR : Incorrect number of MMAs in flight");

    // We release buffers to producer warps(dma load) with some mmas in flight

    // Prologue GMMAs
    int prologue_mma_count = min(CollectiveMainloop::K_PIPE_MMAS, k_tile_count);
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    warpgroup_fence_operand(accum);
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips
      // from rdPhaseBit value)
      auto barrier_token = mainloop_pipeline.consumer_try_wait(smem_pipe_read);

      mainloop_pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();
      warpgroup_arrive();
      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma,
                   tCrA(_, _, k_block, read_stage),
                   tCrB(_, _, k_block, read_stage),
                   accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }

      warpgroup_commit_batch();

      ++smem_pipe_read;
    }

    tiled_mma.accumulate_ = GMMA::ScaleOut::One;

    warpgroup_fence_operand(accum);
    CUTLASS_PRAGMA_UNROLL
    for (int k_tile_prologue = prologue_mma_count - 1; k_tile_prologue > 0;
         --k_tile_prologue) {
      // WAIT on smem_pipe_read until its data are available (phase bit flips
      // from rdPhaseBit value)
      auto barrier_token = mainloop_pipeline.consumer_try_wait(smem_pipe_read);
      mainloop_pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();
      warpgroup_arrive();
      // (V,M,K) x (V,N,K) => (V,M,N)
      cute::gemm(tiled_mma,
                 tCrA(_, _, _, read_stage),
                 tCrB(_, _, _, read_stage),
                 accum);
      warpgroup_commit_batch();

      ++smem_pipe_read;
    }

    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    k_tile_count -= prologue_mma_count;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 0; --k_tile_count) {
      // WAIT on smem_pipe_read until its data are available (phase bit flips
      // from rdPhaseBit value)
      auto barrier_token = mainloop_pipeline.consumer_try_wait(smem_pipe_read);
      mainloop_pipeline.consumer_wait(smem_pipe_read, barrier_token);

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();
      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      // (V,M,K) x (V,N,K) => (V,M,N)
      cute::gemm(tiled_mma,
                 tCrA(_, _, _, read_stage),
                 tCrB(_, _, _, read_stage),
                 accum);
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to
      /// ensure smem_pipe_write is consumed
      warpgroup_wait<CollectiveMainloop::K_PIPE_MMAS>();
      warpgroup_fence_operand(accum);

      // UNLOCK smem_pipe_release, done _computing_ on it
      mainloop_pipeline.consumer_release(smem_pipe_release);

      // Advance smem_pipe_read and smem_pipe_release
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    warpgroup_fence_operand(accum);

    // Make sure the math instructions are done and free buffers before entering
    // the epilogue
    // Prologue GMMAs
    // int prologue_mma_count = min(CollectiveMainloop::K_PIPE_MMAS,
    // k_tile_count); k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      mainloop_pipeline.consumer_release(
          smem_pipe_release); // UNLOCK smem_pipe_release,
                              // done _computing_ on it
      ++smem_pipe_release;
    }

    // Hint on an early release of global memory resources.
    // The timing of calling this function only influences performance,
    // not functional correctness.
    cutlass::arch::launch_dependent_grids();

    // start store op
    using X = Underscore;

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);
    auto M_STRIDE = cute::Int<OUTPUT_STRIDE>{};

    // NOTE(Yu): stride_c and stride_d are global view of the output tensor
    // tranpose stride_c and stride_d
    auto stride_c = cute::make_stride(M_STRIDE, cute::Int<1>{}, 0);
    auto stride_d = cute::make_stride(M_STRIDE, cute::Int<1>{}, 0);
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    if constexpr (::cutlass::gemm::kernel::detail::Has_SwapAB_v<
                      CollectiveMainloop>) {
      auto dC_T = cute::make_stride(
          get<1>(stride_c), get<0>(stride_c), get<2>(stride_c));
      auto dD_T = cute::make_stride(
          get<1>(stride_d), get<0>(stride_d), get<2>(stride_d));
      Tensor mD_mnl_T = cute::make_tensor(
          cute::make_gmem_ptr(d_output), cute::make_shape(M, N, L), dD_T);

      Tensor gD_T = local_tile(
          mD_mnl_T, blk_shape, make_coord(_, _, _), Step<_1, _1, X>{})(
          _, _, m_coord, n_coord, l_coord); // (BLK_M, BLK_N)

      Tensor mC_mnl_T = cute::make_tensor(
          cute::make_gmem_ptr<typename CollectiveEpilogue::DataTypeC>(
              d_residual),
          cute::make_shape(M, N, L),
          dC_T);
      Tensor gC_T = local_tile(
          mC_mnl_T, blk_shape, make_coord(_, _, _), Step<_1, _1, X>{})(
          _, _, m_coord, n_coord, l_coord);

      // Partition source and destination tiles to match the accumulator
      // partitioning
      Tensor tCgD = thr_mma.partition_C(gD_T); // (VEC,THR_M,THR_N)
      Tensor tCgC = thr_mma.partition_C(gC_T); // (VEC,THR_M,THR_N)

      // OOB predication for tile quantization "residue"
      // Absolute coordinate tensors (dynamic)
      auto shape_MN = make_shape(M, N);
      Tensor mD_crd = make_identity_tensor(shape_MN); // (M,N)
      Tensor cD_mn = local_tile(mD_crd,
                                take<0, 2>(blk_shape),
                                make_coord(m_coord, n_coord)); // (BLK_M,BLK_N)
      Tensor tCcD_mn = thr_mma.partition_C(cD_mn); // (VEC,THR_M,THR_N)
      // Relative coordinate tensors (static)
      Tensor cD = cute::make_coord_tensor(cD_mn.layout()); // (BLK_M,BLK_N)
      Tensor tCcD =
          cute::make_coord_tensor(tCcD_mn.layout()); // (VEC,THR_M,THR_N)
      // Subtract the global "bottom right" corner from the local "top left"
      // corner to get the max relative coordinate
      auto residue_cD = shape_MN - cD_mn(_0{});     // (m,n)
      auto residue_tCcD = shape_MN - tCcD_mn(_0{}); // (m,n)

      // Fully OOB tile
      if (not elem_less(repeat_like(residue_cD, _0{}), residue_cD)) {
        return;
      }

      using FragCType = remove_cvref_t<decltype(tCgC(0))>;
      using FragDType = remove_cvref_t<decltype(tCgD(0))>;

      using ThreadOp = typename CollectiveEpilogue::ThreadEpilogueOp;

      typename ThreadOp::Params thread_params{};
      thread_params.alpha = 1.0f;
      thread_params.beta = WITH_RESIDUAL ? 1.0f : 0.0f;
      ThreadOp epilogue_op(thread_params);

      // source is needed
      if constexpr (WITH_RESIDUAL) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
          FragCType fragC;
          bool pred = elem_less(tCcD(i), residue_tCcD);
          cutlass::arch::global_load<FragCType, sizeof(FragCType)>(
              fragC, &tCgC(i), pred);
          FragDType fragD = epilogue_op(accum(i), fragC);

          cutlass::arch::global_store<FragDType, sizeof(FragDType)>(
              fragD, &tCgD(i), pred);
          // }
        }
      }
      // source is not needed, avoid load
      else {

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
          FragCType fragC;
          bool pred = elem_less(tCcD(i), residue_tCcD);
          FragDType fragD = epilogue_op(accum(i), fragC);
          cutlass::arch::global_store<FragDType, sizeof(FragDType)>(
              fragD, &tCgD(i), pred);
        }
      }
    }

    // collective_epilogue.store_tail(epi_load_pipeline,
    //                                epi_load_pipe_consumer_state_next,
    //                                epi_store_pipeline,
    //                                epi_store_pipe_producer_state_next);
  }
}
// };

} // namespace kernel
