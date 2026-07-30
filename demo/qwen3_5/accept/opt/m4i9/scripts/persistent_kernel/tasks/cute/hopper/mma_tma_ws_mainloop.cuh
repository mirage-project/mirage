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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "kernel_traits.cuh"

namespace kernel {

template <class KernelTraits, bool OnHost>
struct MainloopParamsImpl;

// NOTE(Yu): this is for normal mainloop where tma is allocated on host and pass
// from global function
template <class KernelTraits>
struct MainloopParamsImpl<KernelTraits, /*OnHost*/ true> {
  using DataType = typename KernelTraits::DataType;
  using ProblemShape = typename KernelTraits::ProblemShape;

  using StrideA = typename KernelTraits::StrideA;
  using StrideB = typename KernelTraits::StrideB;
  using SmemLayoutA = typename KernelTraits::SmemLayoutA; // (BLK_M,BLK_K,PIPE)
  using SmemLayoutB = typename KernelTraits::SmemLayoutB; // (BLK_N,BLK_K,PIPE)
  using TileShape = typename KernelTraits::TileShape_MNK;
  using ClusterShape = typename KernelTraits::ClusterShape_MNK;

  using GmemTiledCopyA = cute::SM90_TMA_LOAD;
  using GmemTiledCopyB = cute::SM90_TMA_LOAD;

  using TMA_A = decltype(make_tma_copy_A_sm90(
      GmemTiledCopyA{},
      make_tensor(static_cast<DataType const *>(nullptr),
                  repeat_like(StrideA{}, int32_t(0)),
                  StrideA{}),
      SmemLayoutA{}(_, _, cute::Int<0>{}),
      TileShape{},
      ClusterShape{}));

  using TMA_B = decltype(make_tma_copy_B_sm90(
      GmemTiledCopyB{},
      make_tensor(static_cast<DataType const *>(nullptr),
                  repeat_like(StrideB{}, int32_t(0)),
                  StrideB{}),
      SmemLayoutB{}(_, _, cute::Int<0>{}),
      TileShape{},
      ClusterShape{}));

  TMA_A tma_load_a;
  TMA_B tma_load_b;
  uint32_t tma_transaction_bytes;
  uint32_t tma_transaction_bytes_mk;
  uint32_t tma_transaction_bytes_nk;
  ProblemShape problem_shape;
};

// NOTE(Yu): this is for mpk device call
template <class KernelTraits>
struct MainloopParamsImpl<KernelTraits, /*OnHost*/ false> {
  using ProblemShape = typename KernelTraits::ProblemShape;

  uint32_t tma_transaction_bytes;
  uint32_t tma_transaction_bytes_mk;
  uint32_t tma_transaction_bytes_nk;
  ProblemShape problem_shape;
};

template <typename KernelTraits>
struct CollectiveMainloop {
  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = typename KernelTraits::SmemLayoutA; // (BLK_M,BLK_K,PIPE)
  using SmemLayoutB = typename KernelTraits::SmemLayoutB; // (BLK_N,BLK_K,PIPE)

  using GmemTiledCopyA = cute::SM90_TMA_LOAD;
  using GmemTiledCopyB = cute::SM90_TMA_LOAD;

  using DataType = typename KernelTraits::DataType;
  using DTypeAccum = typename KernelTraits::DTypeAccum;

  using StrideA = typename KernelTraits::StrideA;
  using StrideB = typename KernelTraits::StrideB;

  using TileShape = typename KernelTraits::TileShape_MNK;
  using ClusterShape = typename KernelTraits::ClusterShape_MNK;

  using MainloopPipeline = typename KernelTraits::MainloopPipeline;
  using PipelineState = typename MainloopPipeline::PipelineState;

  using TiledMma = typename KernelTraits::TiledMma;

  using ProblemShape = typename KernelTraits::ProblemShape;

  using TileScheduler = typename KernelTraits::TileScheduler;

  static constexpr bool SwapAB = KernelTraits::SwapAB;
  static constexpr int K_PIPE_MMAS = KernelTraits::K_PIPE_MMAS;
  static constexpr int NUM_STAGES = KernelTraits::NUM_STAGES;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::array_aligned<typename TiledMma::ValTypeA,
                          cute::cosize_v<SmemLayoutA>>
          smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB,
                          cute::cosize_v<SmemLayoutB>>
          smem_B;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr uint32_t TmaTransactionBytesMK = cutlass::bits_to_bytes(
      size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) *
      static_cast<uint32_t>(cutlass::sizeof_bits<DataType>::value));
  static constexpr uint32_t TmaTransactionBytesNK = cutlass::bits_to_bytes(
      size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) *
      static_cast<uint32_t>(cutlass::sizeof_bits<DataType>::value));
  static constexpr uint32_t TmaTransactionBytes =
      TmaTransactionBytesMK + TmaTransactionBytesNK;

  // Host side kernel arguments
  struct Arguments {
    DataType const *ptr_A;
    StrideA dA;
    DataType const *ptr_B;
    StrideB dB;
  };

  template <bool onHost>
  using Params = MainloopParamsImpl<KernelTraits, onHost>;

  template <bool onHost = false, class ProblemShapeT>
  CUTLASS_HOST_DEVICE static Params<onHost>
      to_underlying_arguments(ProblemShapeT const &problem_shape,
                              Arguments const &args) {
    uint32_t transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t transaction_bytes_nk = TmaTransactionBytesNK;
    uint32_t transaction_bytes = transaction_bytes_mk + transaction_bytes_nk;

    if constexpr (onHost) {
      auto problem_shape_MNKL = append<4>(problem_shape, 1);
      auto [M, N, K, L] = problem_shape_MNKL;

      auto ptr_A = reinterpret_cast<DataType const *>(args.ptr_A);
      auto ptr_B = reinterpret_cast<DataType const *>(args.ptr_B);

      Tensor tensor_a =
          make_tensor(ptr_A, make_layout(make_shape(M, K, L), args.dA));
      Tensor tensor_b =
          make_tensor(ptr_B, make_layout(make_shape(N, K, L), args.dB));

      Params<true> p{};
      p.problem_shape = problem_shape;
      p.tma_transaction_bytes = transaction_bytes;
      p.tma_transaction_bytes_mk = transaction_bytes_mk;
      p.tma_transaction_bytes_nk = transaction_bytes_nk;

      p.tma_load_a = make_tma_copy_A_sm90(GmemTiledCopyA{},
                                          tensor_a,
                                          SmemLayoutA{}(_, _, cute::Int<0>{}),
                                          TileShape{},
                                          ClusterShape{});

      p.tma_load_b = make_tma_copy_B_sm90(GmemTiledCopyB{},
                                          tensor_b,
                                          SmemLayoutB{}(_, _, cute::Int<0>{}),
                                          TileShape{},
                                          ClusterShape{});

      return p;
    } else {
      Params<false> p{};
      p.problem_shape = problem_shape;
      p.tma_transaction_bytes = transaction_bytes;
      p.tma_transaction_bytes_mk = transaction_bytes_mk;
      p.tma_transaction_bytes_nk = transaction_bytes_nk;
      return p;
    }
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best
  /// performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params<true> const &mainloop_params) {
    cute::prefetch_tma_descriptor(
        mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(
        mainloop_params.tma_load_b.get_tma_descriptor());
  }

  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto load_init(ProblemShape_MNKL const &problem_shape_MNKL,
                                Params<true> const &mainloop_params) const {
    using X = Underscore;
    auto [M, N, K, L] = problem_shape_MNKL;
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(
        make_shape(M, K, L)); // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(
        make_shape(N, K, L)); // (n,k,l)
    Tensor gA_mkl = local_tile(mA_mkl,
                               TileShape{},
                               make_coord(_, _, _),
                               Step<_1, X, _1>{}); // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl,
                               TileShape{},
                               make_coord(_, _, _),
                               Step<X, _1, _1>{}); // (BLK_N,BLK_K,n,k,l)

    return cute::make_tuple(gA_mkl, gB_nkl);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <class TensorA, class TensorB, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void load(Params<true> const &mainloop_params,
                           MainloopPipeline pipeline,
                           PipelineState smem_pipe_write,
                           cute::tuple<TensorA, TensorB> const &load_inputs,
                           BlockCoord const &blk_coord,
                           KTileIterator k_tile_iter,
                           int k_tile_count,
                           int thread_idx,
                           uint32_t block_rank_in_cluster,
                           TensorStorage &shared_tensors) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()),
                              SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()),
                              SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

      Tensor gA_mkl = get<0>(load_inputs);
      Tensor gB_nkl = get<1>(load_inputs);
      // CUTE_STATIC_ASSERT_V(size<2>(gB_nkl) == 0); //
      // int m_tile_count = size<2>(gA_mkl);
      // int n_tile_count = size<2>(gB_nkl);

      auto block_tma_a = mainloop_params.tma_load_a.get_slice(0);
      auto block_tma_b = mainloop_params.tma_load_b.get_slice(0);

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

      Tensor gA = gA_mkl(_, _, m_coord, _, l_coord); // (BLK_M,BLK_K,k)
      Tensor gB = gB_nkl(_, _, n_coord, _, l_coord); // (BLK_N,BLK_K,k)

      // Applies the mapping from block_tma_a
      Tensor tAgA = block_tma_a.partition_S(gA); // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA); // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB); // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB); // (TMA,TMA_N,TMA_K,PIPE)

      uint16_t mcast_mask_a = 0;
      uint16_t mcast_mask_b = 0;

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for (; k_tile_count > 0; --k_tile_count) {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType *tma_barrier =
            pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        // printf("producer really start\n");
        copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a),
             tAgA(_, _, _, *k_tile_iter),
             tAsA(_, _, _, write_stage));
        copy(mainloop_params.tma_load_b.with(*tma_barrier, mcast_mask_b),
             tBgB(_, _, _, *k_tile_iter),
             tBsB(_, _, _, write_stage));
        ++k_tile_iter;

        // Advance smem_pipe_write
        ++smem_pipe_write;
      }
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline,
                                PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <class FrgTensorC>
  CUTLASS_DEVICE void mma(MainloopPipeline pipeline,
                          PipelineState smem_pipe_read,
                          FrgTensorC &accum,
                          int k_tile_count,
                          int thread_idx,
                          TensorStorage &shared_tensors,
                          Params<true> const & /*mainloop_params*/) {

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()),
                            SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()),
                            SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)
    constexpr int MmaWarpGroups =
        size(TiledMma{}) / cutlass::NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(
        Int<MmaWarpGroups>{}, Int<cutlass::NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(
        0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
    TiledMma tiled_mma;
    auto thread_mma =
        tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    Tensor tCsA = thread_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB); // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum)); // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum)); // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));  // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));  // PIPE
    CUTE_STATIC_ASSERT_V(Int<KernelTraits::NUM_STAGES>{} ==
                         size<2>(sA)); // PIPE
    CUTE_STATIC_ASSERT_V(Int<KernelTraits::NUM_STAGES>{} ==
                         size<2>(sB)); // PIPE

    //
    // PIPELINED MAIN LOOP
    //
    static_assert((0 <= KernelTraits::K_PIPE_MMAS) &&
                      (KernelTraits::K_PIPE_MMAS < KernelTraits::NUM_STAGES),
                  "ERROR : Incorrect number of MMAs in flight");

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    // Prologue GMMAs
    int prologue_mma_count = min(KernelTraits::K_PIPE_MMAS, k_tile_count);
    assert(k_tile_count >= 1);
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    warpgroup_fence_operand(accum);
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips
      // from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);

      pipeline.consumer_wait(smem_pipe_read, barrier_token);

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
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

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
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

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
      warpgroup_wait<KernelTraits::K_PIPE_MMAS>();
      warpgroup_fence_operand(accum);

      // UNLOCK smem_pipe_release, done _computing_ on it
      pipeline.consumer_release(smem_pipe_release);

      // Advance smem_pipe_read and smem_pipe_release
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    warpgroup_fence_operand(accum);
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void mma_tail(MainloopPipeline pipeline,
                               PipelineState smem_pipe_release,
                               int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = min(KernelTraits::K_PIPE_MMAS, k_tile_count);
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release); // UNLOCK smem_pipe_release,
                                                    // done _computing_ on it
      ++smem_pipe_release;
    }
  }
};

} // namespace kernel
