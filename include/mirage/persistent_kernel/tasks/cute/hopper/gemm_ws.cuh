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
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"

#include "cutlass/conv/detail.hpp"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/tensor.hpp"

#include "cutlass/arch/grid_dependency_control.h"

namespace kernel {

using namespace cute;

template <class CollectiveMainloop, class CollectiveEpilogue>
CUTLASS_DEVICE void gemm_kernel_tma_warp_specialized(
    typename CollectiveMainloop::template Params<true> const &mainloop_params,
    typename CollectiveEpilogue::Params const &epilogue_params) {

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
    Producer = 0,
    Consumer = 1,
  };
  enum class ProducerWarpRole {
    MainloopEpilogue = 0,
    Warp1 = 1,
    Warp2 = 2,
    Warp3 = 3
  };
  extern __shared__ char smem[];

  using ClusterShape = typename CollectiveMainloop::ClusterShape;
  using TiledMma = typename CollectiveMainloop::TiledMma;
  using TileShape = typename CollectiveMainloop::TileShape;
  // using PipelineState = typename CollectiveMainloop::PipelineState;

  // Kernel level shared memory storage
  SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem);

  int thread_idx = int(threadIdx.x);
  int lane_idx = cutlass::canonical_lane_idx();
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int warp_idx_in_warp_group = warp_idx % cutlass::NumWarpsPerWarpGroup;
  int warp_group_thread_idx = thread_idx % cutlass::NumThreadsPerWarpGroup;
  auto warp_group_role = WarpGroupRole(cutlass::canonical_warp_group_idx());
  auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
  int lane_predicate = cute::elect_one_sync();
  uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

  // Issue Tma Descriptor Prefetch from a single thread
  if ((warp_idx == 0) && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
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
  mainloop_pipeline_params.transaction_bytes =
      mainloop_params.tma_transaction_bytes;
  MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop,
                                     mainloop_pipeline_params,
                                     ClusterShape{});

  // Epilogue Load pipeline
  using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
  typename EpiLoadPipeline::Params epi_load_pipeline_params;
  if (warp_group_role == WarpGroupRole::Producer &&
      producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
    epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
  }
  if (warp_group_role == WarpGroupRole::Consumer) {
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

  CollectiveMainloop collective_mainloop;
  CollectiveEpilogue collective_epilogue(epilogue_params);

  auto load_inputs =
      collective_mainloop.load_init(problem_shape_MNKL, mainloop_params);
  static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2,
                "Output of load_init must have at least two elements (A, B)");

  // Extract out partitioned A and B.
  Tensor gA_mkl = get<0>(load_inputs);
  Tensor gB_nkl = get<1>(load_inputs);

  // Compute m_coord, n_coord, and l_coord with their post-tiled shapes
  auto m_coord = idx2crd(int(blockIdx.x), shape<2>(gA_mkl));
  auto n_coord = idx2crd(int(blockIdx.y), shape<2>(gB_nkl));
  // handles the difference between the rank of Tensor returned by load_input in
  // case they do not have a batch mode auto l_coord = [&] (auto const& gB_nkl_)
  // {
  //   // gB_nkl needs to be passed into the lambda because C++17
  //   // does not permit lambda capture of structured bindings.
  //   if constexpr (not IsConvProblemShape) {
  //     // This needs to be inside an `if constexpr`,
  //     // because shape<4>(gB_nkl) is not well-formed otherwise.
  //     return idx2crd(int(blockIdx.z), shape<4>(gB_nkl_));
  //   }
  //   else {
  //     return Int<0>{};
  //   }
  // } (gB_nkl);

  auto blk_coord = make_coord(m_coord, n_coord, _, Int<0>{});

  // Get pipeline iterators and increments from tensor shapes
  auto k_tile_iter = cute::make_coord_iterator(shape<3>(gA_mkl));
  auto k_tile_count = size<3>(gA_mkl);

  // Wait for all thread blocks in the Cluster
  __syncthreads();

  if (warp_group_role == WarpGroupRole::Producer) {
    if (producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
      // Ensure that the prefetched kernel does not touch
      // unflushed global memory prior to this instruction
      cutlass::arch::wait_on_dependent_grids();
      collective_mainloop.load(mainloop_params,
                               mainloop_pipeline,
                               mainloop_pipe_producer_state,
                               load_inputs,
                               blk_coord,
                               k_tile_iter,
                               k_tile_count,
                               lane_idx,
                               block_rank_in_cluster,
                               shared_storage.tensors.mainloop);
      // Update starting mainloop pipeline state for the pipeline drain
      mainloop_pipe_producer_state.advance(k_tile_count);
      // Make sure mainloop consumer has been waited upon before issuing
      // epilogue load
      collective_mainloop.load_tail(mainloop_pipeline,
                                    mainloop_pipe_producer_state);
    }
  } else if (warp_group_role == WarpGroupRole::Consumer) {
    Tensor accumulators = partition_fragment_C(
        tiled_mma, take<0, 2>(blk_shape)); // (MMA,MMA_M,MMA_N)

    collective_mainloop.mma(mainloop_pipeline,
                            mainloop_pipe_consumer_state,
                            accumulators,
                            k_tile_count,
                            warp_group_thread_idx,
                            shared_storage.tensors.mainloop,
                            mainloop_params);

    // Make sure the math instructions are done and free buffers before entering
    // the epilogue
    collective_mainloop.mma_tail(
        mainloop_pipeline, mainloop_pipe_consumer_state, k_tile_count);

    // Hint on an early release of global memory resources.
    // The timing of calling this function only influences performance,
    // not functional correctness.
    cutlass::arch::launch_dependent_grids();

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
                                  warp_group_thread_idx,
                                  shared_storage.tensors.epilogue);

    // collective_epilogue.store_tail(epi_load_pipeline,
    //                                epi_load_pipe_consumer_state_next,
    //                                epi_store_pipeline,
    //                                epi_store_pipe_producer_state_next);
  }
}
// };

///////////////////////////////////////////////////////////////////////////////

} // namespace kernel
