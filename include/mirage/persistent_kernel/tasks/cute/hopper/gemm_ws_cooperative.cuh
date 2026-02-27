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

#include "cute/arch/cluster_sm90.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal_decl.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"
#include "cutlass/workspace.h"

namespace kernel {

using namespace cute;

template <class CollectiveMainloop, class CollectiveEpilogue>
CUTLASS_DEVICE void gemm_kernel_tma_warp_specialized_cooperative(
    typename CollectiveMainloop::Params const &mainloop_params,
    typename CollectiveEpilogue::Params const &epilogue_params,
    typename CollectiveMainloop::TileScheduler::Params const
        &scheduler_params) {
  using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1, 2>;

  enum class WarpGroupRole {
    Producer = 0,
    Consumer0 = 1,
    Consumer1 = 2,
  };
  enum class ProducerWarpRole {
    Mainloop = 0,
    Warp1 = 1,
    Epilogue = 2,
    MainloopAux = 3
  };
  extern __shared__ char smem[];

  using ClusterShape = typename CollectiveMainloop::ClusterShape;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using TileShape = typename CollectiveMainloop::TileShape;
  // using PipelineState = typename CollectiveMainloop::PipelineState;

  static constexpr uint32_t TileSchedulerPipelineStageCount = 0;
  //   using TileSchedulerTag = TileSchedulerTag_;

  using TileScheduler = typename CollectiveMainloop::TileScheduler;

  using TileSchedulerPipeline = typename TileScheduler::Pipeline;
  using TileSchedulerPipelineState =
      typename TileSchedulerPipeline::PipelineState;
  using TileSchedulerStorage = typename TileScheduler::SharedStorage;
  using TileSchedulerThrottlePipeline =
      typename TileScheduler::ThrottlePipeline;
  using TileSchedulerThrottlePipelineState =
      typename TileSchedulerThrottlePipeline::PipelineState;

  struct SharedStorage {
    // Mainloop and epilogue don't use smem concurrently since kernel is
    // non-persistent, so we can use a union
    union TensorStorage {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;

    alignas(16) TileSchedulerStorage scheduler;

    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage =
          typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage =
          typename CollectiveEpilogue::PipelineStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order;
    } pipelines;
  };

  //   typename detail::TileSchedulerSelector<
  //                                           TileSchedulerTag,
  //                                           ArchTag,
  //                                           TileShape,
  //                                           ClusterShape
  //                                           ,TileSchedulerPipelineStageCount
  //                                           >::Scheduler;

  // 1 stage ordered sequence between mainloop and epilogue producer load
  // threads

  /// Register requirement for Load and Math WGs
  static constexpr uint32_t NumMMAThreads = size(TiledMma{}); // 8 warps
  static constexpr int RegsPerThread = size<0>(TileShape{}) *
                                       size<1>(TileShape{}) / NumMMAThreads *
                                       sizeof(float) / sizeof(uint32_t);
  static constexpr bool HeavyRegisterPressure = RegsPerThread >= 208;
  static constexpr uint32_t LoadRegisterRequirement =
      !HeavyRegisterPressure ? 40 : 24;
  static constexpr uint32_t MmaRegisterRequirement =
      !HeavyRegisterPressure ? 232 : 240;
  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups =
      NumMMAThreads / cutlass::NumThreadsPerWarpGroup;

  static_assert(
      NumMMAThreads == 256,
      "Cooperative kernel must have TiledMMA operating using 256 threads.");
  static_assert(size<0>(TileShape{}) >= 128,
                "Cooperative kernel requires Tile Size to be greater than or "
                "equal to 128 along the M-dimension.");

  // Kernel level shared memory storage
  SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem);

  int thread_idx = int(threadIdx.x);
  int lane_idx = cutlass::canonical_lane_idx();
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
  int warp_group_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
  int mma_thread_idx = thread_idx % NumMMAThreads;
  auto warp_group_role = WarpGroupRole(cutlass::canonical_warp_group_idx());
  auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
  int lane_predicate = cute::elect_one_sync();
  uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

  // Issue Tma Descriptor Prefetch from a single thread
  if ((warp_idx == 0) && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
    // CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
  }

  // TODO, add this TileScheduler pipeline
  typename TileSchedulerPipeline::Params scheduler_pipeline_params;
  typename TileSchedulerThrottlePipeline::Params
      scheduler_throttle_pipeline_params;

  TileSchedulerPipeline scheduler_pipeline(shared_storage.scheduler.pipeline(),
                                           scheduler_pipeline_params);
  TileSchedulerPipelineState scheduler_pipe_consumer_state;

  TileSchedulerThrottlePipeline scheduler_throttle_pipeline(
      shared_storage.scheduler.throttle_pipeline(),
      scheduler_throttle_pipeline_params);
  TileSchedulerThrottlePipelineState scheduler_pipe_throttle_consumer_state;
  TileSchedulerThrottlePipelineState scheduler_pipe_throttle_producer_state =
      cutlass::make_producer_start_state<TileSchedulerThrottlePipeline>();

  // Mainloop Load pipeline
  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  typename MainloopPipeline::Params mainloop_pipeline_params;
  if (warp_group_role == WarpGroupRole::Producer &&
      (producer_warp_role == ProducerWarpRole::Mainloop ||
       producer_warp_role == ProducerWarpRole::MainloopAux)) {
    mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
  }
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
  }

  mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
  mainloop_pipeline_params.num_consumers = NumMMAThreads;
  mainloop_pipeline_params.num_producers = 1;
  mainloop_pipeline_params.transaction_bytes =
      mainloop_params.tma_transaction_bytes;
  MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop,
                                     mainloop_pipeline_params,
                                     ClusterShape{});

  // Epilogue Load pipeline
  using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
  typename EpiLoadPipeline::Params epi_load_pipeline_params;
  if (warp_group_role == WarpGroupRole::Producer &&
      producer_warp_role == ProducerWarpRole::Epilogue) {
    epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
  }
  if (warp_group_role == WarpGroupRole::Consumer0 ||
      warp_group_role == WarpGroupRole::Consumer1) {
    epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
  }
  epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
  epi_load_pipeline_params.producer_arv_count = cutlass::NumThreadsPerWarp;
  epi_load_pipeline_params.consumer_arv_count = cutlass::NumThreadsPerWarpGroup;
  if constexpr (CollectiveEpilogue::RequiresTransactionBytes) {
    epi_load_pipeline_params.transaction_bytes =
        epilogue_params.tma_transaction_bytes;
  }
  EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load,
                                    epi_load_pipeline_params);
  // Epilogue Store pipeline
  using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
  typename EpiStorePipeline::Params epi_store_pipeline_params;
  epi_store_pipeline_params.always_wait = true;
  EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

  // todo, add this barrier
  typename LoadWarpOrderBarrier::Params params_load_order_barrier;
  params_load_order_barrier.group_id =
      producer_warp_role == ProducerWarpRole::Mainloop ? 0 : 1;
  params_load_order_barrier.group_size = cutlass::NumThreadsPerWarp;
  LoadWarpOrderBarrier load_order_barrier(shared_storage.pipelines.load_order,
                                          params_load_order_barrier);

  // Initialize starting pipeline states for the collectives
  // Epilogue store pipe is producer-only (consumer is TMA unit, waits via
  // scoreboarding)
  typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
  typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

  // For the DMA Load (producer) we start with an opposite phase
  // i.e., we skip all waits since we know that the buffer is indeed empty
  cutlass::PipelineState mainloop_pipe_producer_state =
      cutlass::make_producer_start_state<MainloopPipeline>();
  cutlass::PipelineState epi_load_pipe_producer_state =
      cutlass::make_producer_start_state<EpiLoadPipeline>();
  cutlass::PipelineState epi_store_pipe_producer_state =
      cutlass::make_producer_start_state<EpiStorePipeline>();
  auto blk_shape = TileShape{}; // (BLK_M,BLK_N,BLK_K)
  TiledMma tiled_mma;

  auto problem_shape_MNKL =
      append<4>(mainloop_params.problem_shape, cute::Int<1>{});

  // todo, define the TILE Scheduler collective
  TileScheduler scheduler{scheduler_params};
  // Declare work_tile_info, then define it in each of warps that use it.
  typename TileScheduler::WorkTileInfo work_tile_info;

  CollectiveMainloop collective_mainloop;
  CollectiveEpilogue collective_epilogue(epilogue_params);

  auto load_inputs =
      collective_mainloop.load_init(problem_shape_MNKL, mainloop_params);
  static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2,
                "Output of load_init must have at least two elements (A, B)");

  // Extract out partitioned A and B.
  Tensor gA_mkl = get<0>(load_inputs);
  Tensor gB_nkl = get<1>(load_inputs);

  // Wait for all thread blocks in the Cluster
  __syncthreads();

  if (warp_group_role == WarpGroupRole::Producer) {
    work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});
    cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
    // Mainloop Producer Warp
    if (producer_warp_role == ProducerWarpRole::Mainloop) {
      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();
      bool do_load_order_arrive = true;
      bool requires_clc_query = true;
      while (work_tile_info.is_valid()) {
        if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
          auto [next_work_tile_info, increment_pipe] =
              scheduler.fetch_next_work(work_tile_info);
          work_tile_info = next_work_tile_info;
          continue;
        }

        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and
        // n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

        // Get the number of K tiles to compute for this work as well as the
        // starting K tile offset of the work.
        auto work_k_tile_count = TileScheduler::get_work_k_tile_count(
            work_tile_info, problem_shape_MNKL, blk_shape);
        auto work_k_tile_start =
            TileScheduler::get_work_k_tile_start(work_tile_info);
        auto k_tile_iter = cute::make_coord_iterator(
            idx2crd(work_k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));

        if (requires_clc_query) {
          scheduler_throttle_pipeline.producer_acquire(
              scheduler_pipe_throttle_producer_state);
          scheduler_throttle_pipeline.producer_commit(
              scheduler_pipe_throttle_producer_state);
          ++scheduler_pipe_throttle_producer_state;
        }

        collective_mainloop.load(mainloop_params,
                                 mainloop_pipeline,
                                 mainloop_pipe_producer_state,
                                 load_inputs,
                                 blk_coord,
                                 k_tile_iter,
                                 work_k_tile_count,
                                 lane_idx,
                                 block_rank_in_cluster,
                                 shared_storage.tensors.mainloop);
        // Update starting pipeline state for the next tile
        mainloop_pipe_producer_state.advance(work_k_tile_count);

        // Signal for the epilogue load warp to begin
        if (do_load_order_arrive) {
          load_order_barrier.arrive();
          do_load_order_arrive = false;
        }
        // Get next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
            work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);

        work_tile_info = next_work_tile_info;
      } // Scheduler work fetch loop

      // Make sure all Consumer Warp Groups have been waited upon
      collective_mainloop.load_tail(mainloop_pipeline,
                                    mainloop_pipe_producer_state);
    }
  } else if (warp_group_role == WarpGroupRole::Consumer0 ||
             warp_group_role == WarpGroupRole::Consumer1) {
    work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});
    cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

    CollectiveEpilogue collective_epilogue(epilogue_params);

    // Do we potentially issue tail arrives for TMA stores, if epilogue load is
    // waiting for it
    bool do_store_tail = false;
    while (work_tile_info.is_valid()) {
      // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and
      // n-shape
      auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
      auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
      auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
      auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);
      auto work_k_tile_count = TileScheduler::get_work_k_tile_count(
          work_tile_info, problem_shape_MNKL, blk_shape);
      // Allocate the accumulators for the (M,N) blk_shape
      //
      // MSVC CTAD breaks if we say "Tensor" here, so we use "auto" instead.
      auto accumulators = partition_fragment_C(
          tiled_mma, take<0, 2>(blk_shape)); // (MMA,MMA_M,MMA_N)
      if (TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {

        collective_mainloop.mma(mainloop_pipeline,
                                mainloop_pipe_consumer_state,
                                accumulators,
                                work_k_tile_count,
                                mma_thread_idx,
                                shared_storage.tensors.mainloop,
                                mainloop_params);

        // Make sure the math instructions are done and free buffers before
        // entering the epilogue
        collective_mainloop.mma_tail(
            mainloop_pipeline, mainloop_pipe_consumer_state, work_k_tile_count);

        // Update starting mainloop pipeline state for the next tile
        mainloop_pipe_consumer_state.advance(work_k_tile_count);
      }
      // Index of warp group within consumer warp groups
      int consumer_warp_group_idx =
          cutlass::canonical_warp_group_idx() - NumLoadWarpGroups;

      // Perform reduction across splits, if needed
      TileScheduler::fixup(scheduler_params,
                           work_tile_info,
                           accumulators,
                           NumMmaWarpGroups,
                           consumer_warp_group_idx);

      if (TileScheduler::compute_epilogue(work_tile_info, scheduler_params)) {
        // Epilogue and write to gD
        auto [epi_load_pipe_consumer_state_next,
              epi_store_pipe_producer_state_next] =
            collective_epilogue.store(epi_load_pipeline,
                                      epi_load_pipe_consumer_state,
                                      epi_store_pipeline,
                                      epi_store_pipe_producer_state,
                                      problem_shape_MNKL,
                                      blk_shape,
                                      blk_coord,
                                      accumulators,
                                      tiled_mma,
                                      mma_thread_idx,
                                      shared_storage.tensors.epilogue,
                                      work_tile_info.reduction_subtile_idx());
        epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;
        epi_store_pipe_producer_state = epi_store_pipe_producer_state_next;
        do_store_tail = true;
      }

      // Get next work tile
      auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);
      work_tile_info = next_work_tile_info;
    } // Scheduler work fetch loop

    if (do_store_tail) {
      // collective_epilogue.store_tail(
      //   epi_load_pipeline,
      //   epi_load_pipe_consumer_state,
      //   epi_store_pipeline,
      //   epi_store_pipe_producer_state
      // );
    }
  } // Consumer Warp Groups End
}
} // namespace kernel
