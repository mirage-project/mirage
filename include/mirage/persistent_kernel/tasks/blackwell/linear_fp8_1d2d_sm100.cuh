/* Copyright 2026 CMU
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
#include <cstdio>
#include <iostream>

// Cutlass includes
#include <cutlass/half.h> // F16 data type
// #include <cutlass/util/print_error.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

// CuTe includes
// #include <cute/algorithm/cooperative_copy.hpp> // Auto vectorized copy operation
#include <cute/arch/cluster_sm90.hpp> // CuTe functions for querying the details of cluster launched
#include <cute/arch/tmem_allocator_sm100.hpp> // TMEM allocator for SM100
#include <cute/numeric/integral_constant.hpp> // Compile time in constants such as _1, _256 etc.
#include <cute/tensor.hpp>                    // CuTe tensor implementation

#include "../common/dmem_layout.cuh"
#include "../common/worker_config.h"
#include "../hopper/barrier.cuh"
#include "../hopper/smem_layout_tma.cuh"
#include "../hopper/tma.cuh"
#include "storage.cuh"

namespace kernel {

template <typename T_,
          typename TMA_A,
          typename TMA_B,
          typename TMA_SFA,
          typename TMA_SFB,
          class BiasTensor,
          typename TMA_OUT,
          int MMA_M,
          int MMA_N,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          bool NOBIAS,
          bool SplitK,
          int NUM_AB_STAGE = 8,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE = 4>
__device__ __noinline__ void
linear_fp8_1d2d_sm100_task_impl(
                         const TMA_A &tma_a,
                         const TMA_B &tma_b,
                         const TMA_SFA &tma_sfa,
                         const TMA_SFB &tma_sfb,
                         BiasTensor mBias,
                         const TMA_OUT &tma_out) {
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = kernel::lane_id();

    auto mma_coord_vmnk = cute::make_coord(0,        // Peer CTA coordinate
        cute::_,  //    MMA-M coordinate
        cute::_,  //    MMA-N coordinate
        cute::_); //    MMA-K coordinate

    constexpr int num_tmem_columns = MMA_N * NUM_ACC_STAGE;

    // cute::TiledMMA tiled_mma = cute::make_tiled_mma(
    //     cute::SM100_MMA_F8F6F4_SS<T_,
    //                                T_,
    //                                float, // Mma's A, B, and Accumulator types
    //                                MMA_M,
    //                                MMA_N, // Mma M and N dimensions
    //                                cute::UMMA::Major::K,
    //                                cute::UMMA::Major::K>{}); // A and B layouts
    cute::TiledMMA tiled_mma = cute::make_tiled_mma(
        cute::SM100_MMA_F16BF16_SS<T_,
                                   T_,
                                   float, // Mma's A, B, and Accumulator types
                                   MMA_M,
                                   MMA_N, // Mma M and N dimensions
                                   cute::UMMA::Major::K,
                                   cute::UMMA::Major::K>{}); // A and B layouts

    auto bM = cute::tile_size<0>(
        tiled_mma); // MMA Tile M. We'll use 1 MMAs per MMA Tile M.
    auto bN = cute::tile_size<1>(
        tiled_mma); // MMA Tile N. We'll use 1 MMAs per MMA Tile N.
    auto bK = cute::tile_size<2>(tiled_mma) *
                cute::Int<4>{}; // MMA Tile K. We'll use 4 MMAs per MMA Tile K. For
                                // 16b types, tcgen05.mma has K16.
    
    auto mma_tiler = cute::make_shape(bM, bN, bK); // (MMA_M, MMA_N, MMA_K)

    auto mma_coord = cute::select<1, 2, 3>(mma_coord_vmnk);
    auto cd_tiler =
        cute::make_shape(bN, bM, bK); // (MmaTile_N, MmaTile_M, MmaTile_K)

    cute::Tensor mA = cute::make_coord_tensor(cute::make_layout(
        cute::make_shape(OUTPUT_SIZE, REDUCTION_SIZE),
        cute::make_stride(
            cute::E<1>{},
            cute::E<0>{}))); // ArithTuple(_0,_0) o
                            // (output_size,reduction_size):(_1@1,_1@0)
    cute::Tensor mB = cute::make_coord_tensor(cute::make_layout(
        cute::make_shape(BATCH_SIZE, REDUCTION_SIZE),
        cute::make_stride(
            cute::E<1>{},
            cute::E<0>{}))); // ArithTuple(_0,_0) o
                            // (batch_size,reduction_size):(_1@1,_1@0)
    cute::Tensor mC = cute::make_coord_tensor(cute::make_layout(
        cute::make_shape(BATCH_SIZE, OUTPUT_SIZE),
        cute::make_stride(cute::E<1>{},
                            cute::E<0>{}))); // ArithTuple(_0,_0) o
                                            // (batch_size,output_size):(_1@1,_1@0)

    cute::Tensor gA = cute::local_tile(
        mA,
        mma_tiler,
        mma_coord,
        cute::Step<cute::_1,
                    cute::X,
                    cute::_1>{}); // ArithTuple(_0,_0) o
                                // (_128,_64,8,64):(_1@1,_1@0,_128@1,_64@0)
    cute::Tensor gB = cute::local_tile(
        mB,
        mma_tiler,
        mma_coord,
        cute::Step<cute::X,
                    cute::_1,
                    cute::_1>{}); // ArithTuple(_0,_0) o
                                // (_32,_64,1,64):(_1@1,_1@0,_32@1,_64@0)
    cute::Tensor gBias = cute::local_tile(
        mBias,
        cd_tiler,
        mma_coord,
        cute::Step<cute::_1,
                    cute::_1,
                    cute::X>{}); // gmem_ptr[16b](0x7704bd040000) o
                                // (_32,_128,1,8):(1024,_1,32768,_128)
    cute::Tensor gC = cute::local_tile(
        mC,
        cd_tiler,
        mma_coord,
        cute::Step<cute::_1,
                    cute::_1,
                    cute::X>{}); // ArithTuple(_0,_0) o
                                // (_32,_128,1,8):(_1@1,_1@0,_32@1,_128@0)

    // Pre-partitioned Tile Shape (MmaTile_M, MmaTile_K) to post-partitioned
    // (MmaA, NumMma_M, NumMma_K)
    auto mma_shape_A =
        cute::partition_shape_A(tiled_mma,
                                cute::make_shape(cute::Int<MMA_M>{},
                                                cute::size<2>(mma_tiler),
                                                cute::Int<NUM_AB_STAGE>{}));
    // Pre-partitioned Tile Shape (MmaTile_N, MmaTile_K) to post-partitioned
    // (MmaB, NumMma_N, NumMma_K)
    auto mma_shape_B =
        cute::partition_shape_B(tiled_mma,
                                cute::make_shape(cute::Int<MMA_N>{},
                                                cute::size<2>(mma_tiler),
                                                cute::Int<NUM_AB_STAGE>{}));
    // Pre-partitioned Tile Shape (MmaTile_N, MmaTile_M) to post-partitioned
    // (MmaC, NumMma_N, NumMma_K)
    auto mma_shape_C =
        cute::make_shape(cute::make_shape(cute::Int<MMA_N>{}, cute::Int<MMA_M>{}),
                        cute::Int<1>{},
                        cute::Int<1>{},
                        cute::Int<NUM_C_STAGE>{});


    auto sA_layout = cute::UMMA::tile_to_mma_shape(
        cute::UMMA::Layout_K_SW128_Atom<T_>{}, mma_shape_A);
    auto sB_layout = cute::UMMA::tile_to_mma_shape(
        cute::UMMA::Layout_K_SW128_Atom<T_>{}, mma_shape_B);
    // constuct unswizzled layout for C
    auto sC_layout_fake = cute::UMMA::tile_to_mma_shape(
        cute::UMMA::Layout_K_INTER_Atom<T_>{}, mma_shape_C);
    auto sC_shape = cute::make_shape(
        cute::make_shape(cute::Int<MMA_N>{}, cute::Int<MMA_M>{}),
        cute::Int<1>{},
        cute::Int<1>{},
        cute::make_shape(cute::Int<1>{}, cute::Int<NUM_C_STAGE>{}));
    auto sC_stride = cute::make_stride(
        cute::make_stride(cute::Int<MMA_M>{}, cute::Int<1>{}),
        cute::Int<0>{},
        cute::Int<0>{},
        cute::make_stride(cute::Int<0>{}, cute::Int<MMA_M * MMA_N>{}));
    auto sC_layout = cute::composition(sC_layout_fake.layout_a(),
                                        sC_layout_fake.offset(),
                                        cute::make_layout(sC_shape, sC_stride));


                              

    using SharedStorage = PipedSharedStorage<T_,
        T_,
        T_,
        decltype(sA_layout),
        decltype(sB_layout),
        decltype(sC_layout),
        NUM_AB_STAGE,
        NUM_ACC_STAGE>;



    extern __shared__ char shared_memory[];
    uintptr_t aligned_smem =
    (reinterpret_cast<uintptr_t>(shared_memory) + 127) / 128 * 128;
    SharedStorage &shared_storage =
    *reinterpret_cast<SharedStorage *>(aligned_smem);

    // Prefetch TMA descriptors at the very beginning
    // if (warp_idx == 0 && lane_idx == 0) {
    //     kernel::tma::prefetch_tma_descriptor(tma_a.desc_ptr);
    //     kernel::tma::prefetch_tma_descriptor(tma_b.desc_ptr);
    //     // kernel::tma::prefetch_tma_descriptor(tma_sfb.desc_ptr);
    //     kernel::tma::prefetch_tma_descriptor(tma_out.desc_ptr);
    // }

    // Initialize barriers
    if (warp_idx == 0) {
        cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        NUM_AB_STAGE>(shared_storage.ab_full_mbar_ptr, /* arrival count */ 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        NUM_AB_STAGE>(shared_storage.ab_empty_mbar_ptr, /* arrival count */ 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        NUM_ACC_STAGE>(shared_storage.acc_full_mbar_ptr, /* arrival count */ 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        NUM_ACC_STAGE>(shared_storage.acc_empty_mbar_ptr,
                       /* arrival count */ 4);
    }

    // Sync tmem allocation status between MMA and epilogue warps within CTA
    // 32 threads (mma) + 128 threads (epilog) to sync
    cutlass::arch::NamedBarrier tmem_allocation_result_barrier(
        32 + 128, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
    cutlass::arch::NamedBarrier epilogue_wg_barrier(
        128, /*bar-id*/ cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);


  // Represent the SMEM buffers for A and B
  cute::Tensor tCsA =
  shared_storage.tensor_sA(); // (MmaA, NumMma_M, NumMma_K, Tiles_K)
cute::Tensor tCsB =
  shared_storage.tensor_sB(); // (MmaB, NumMma_M, NumMma_K, Tiles_K)
cute::Tensor sC_epi = shared_storage.tensor_sC(); // (EpiTile)

    //
    // Mma partitioning for A and B
    //

    auto mma_v = cute::get<0>(mma_coord_vmnk);
    cute::ThrMMA cta_mma = tiled_mma.get_slice(mma_v); // Use Peer CTA coordinate
    cute::Tensor tCgA =
    cta_mma.partition_A(gA); // (MmaA, NumMma_M, NumMma_K, Tiles_K)
    cute::Tensor tCgB =
    cta_mma.partition_B(gB); // (MmaB, NumMma_N, NumMma_K, Tiles_K)

    int tma_transaction_bytes =
    sizeof(T_) * cute::size<1>(mma_tiler) * cute::size<2>(mma_tiler) +
    sizeof(T_) * cute::size<0>(mma_tiler) * cute::size<2>(mma_tiler);

    constexpr int TILE_SIZE = 64;
    constexpr int INPUT_TMA_TILE_SIZE = 64;
    constexpr int WEIGHT_TMA_TILE_SIZE = 64;
    constexpr int OUTPUT_ATOM_SIZE = 128;
    constexpr int B = 3;
    constexpr int M = 3;
    constexpr int S = 3;

    T_ *shared_weight = shared_storage.A.begin();
    T_ *shared_input = shared_storage.B.begin();
    T_ *mm_output = shared_storage.C.begin();

    Barrier *ab_full_mbar_ptr =
    reinterpret_cast<Barrier *>(shared_storage.ab_full_mbar_ptr);

    using InputSmem = smem_tma<T_,
                            B,
                            M,
                            S,
                            MMA_N,
                            INPUT_TMA_TILE_SIZE,
                            1>; // 64/64 = 1
    InputSmem input_smem(shared_input);

    using WeightSmem = smem_tma<T_,
                            B,
                            M,
                            S,
                            OUTPUT_ATOM_SIZE,
                            WEIGHT_TMA_TILE_SIZE,
                            1>; // 64/64 = 1
    WeightSmem input_weight_smem(shared_weight);

    using OutputSmem = smem_tma<T_, 0, M, S, MMA_N, OUTPUT_ATOM_SIZE, 1>;
    OutputSmem mm_output_smem(mm_output);

    // MMA Fragment Allocation
    // We allocate "fragments" which are SMEM descriptors that serve as inputs to
    // cute::gemm operations. For tcgen05.mma operations:
    // - Matrices A and B are sourced from SMEM
    // - tCrA and tCrB provide descriptor views of tCsA and tCsB respectively
    // - The first mode of each descriptor represents the SMEM for a single MMA
    // operation
    cute::Tensor tCrA =
    cta_mma.make_fragment_A(tCsA); // (MmaA, NumMma_M, NumMma_K, Tiles_K)
    cute::Tensor tCrB =
    cta_mma.make_fragment_B(tCsB); // (MmaB, NumMma_M, NumMma_K, Tiles_K)
    auto acc_shape = cute::partition_shape_C(
    tiled_mma,
    cute::make_shape(cute::size<0>(mma_tiler),
                    cute::size<1>(mma_tiler),
                    cute::Int<NUM_ACC_STAGE>{})); // (MmaC, NumMma_M,
                                                    // NumMma_N, NumAcc_Stage)
    // (MMA, MMA_M, MMA_N, STAGE)
    auto tCtAcc = tiled_mma.make_fragment_C(acc_shape);

    cutlass::arch::fence_barrier_init();
    __syncthreads();

    int k_tile_count = cute::size<4>(tCgA);

    using TmemAllocator = cute::TMEM::Allocator1Sm;
    TmemAllocator tmem_allocator{};

    // Make initialized barrier visible in async proxy
    cutlass::arch::fence_barrier_init();
    __syncthreads(); // Wait for all threads until warp0 allocates TMEM

    if (warp_idx == 5) {
        // TMA warp (1)
        int total_k_tile_count = 0;
        for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
          for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {

            if (lane_idx == 0) {
                printf("in TMA warp, m_tile: %d, n_tile: %d, before ab empty barrier\n", m_tile, n_tile);
            }
    
            int num_prev_k_blk = total_k_tile_count;
            total_k_tile_count += k_tile_count;
    
            int tma_wr_k_tile = 0;
            int smem_wr_buffer = (num_prev_k_blk + tma_wr_k_tile) % NUM_AB_STAGE;
            int tma_wr_ab_empty_phase =
                (num_prev_k_blk + tma_wr_k_tile) / NUM_AB_STAGE % 2 ^ 1;
    
            bool peek_ab_empty_status = kernel::try_wait_barrier(
                shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                tma_wr_ab_empty_phase);
    
            // CUTE_UNROLL
            for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
    
              int tma_wr_k_tile_next = tma_wr_k_tile + 1;
              int smem_wr_buffer_next =
                  (num_prev_k_blk + tma_wr_k_tile_next) % NUM_AB_STAGE;
              int tma_wr_ab_empty_phase_next = smem_wr_buffer_next == 0
                                                   ? tma_wr_ab_empty_phase ^ 1
                                                   : tma_wr_ab_empty_phase;

            if (lane_idx == 0) {
                printf("in TMA warp, m_tile: %d, n_tile: %d, after ab empty barrier\n", m_tile, n_tile);
            }
    
              // Wait for an empty buffer
              if (!peek_ab_empty_status) {
                cute::wait_barrier(shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                                   tma_wr_ab_empty_phase);
              }
    
              if (cute::elect_one_sync()) {
                int tma_coords_A[2] = {k_tile * TILE_SIZE,
                                       m_tile * OUTPUT_ATOM_SIZE};
                int tma_coords_B[2] = {k_tile * TILE_SIZE, n_tile * MMA_N};
                input_weight_smem.set_ptr(
                    shared_weight + smem_wr_buffer * OUTPUT_ATOM_SIZE * TILE_SIZE);
                input_smem.set_ptr(shared_input +
                                   smem_wr_buffer * MMA_N * TILE_SIZE);
                cute::set_barrier_transaction_bytes(
                    shared_storage.ab_full_mbar_ptr[smem_wr_buffer],
                    tma_transaction_bytes);
                // set_barrier_transaction_bytes(ab_full_mbar_ptr[smem_wr_buffer],
                // tma_transaction_bytes);
                tma_a.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer],
                                   input_weight_smem.base_ptr,
                                   tma_coords_A);
                tma_b.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer],
                                   input_smem.base_ptr,
                                   tma_coords_B);
              }
    
              if (tma_wr_k_tile_next < k_tile_count) {
                if (lane_idx == 0) {
                    printf("in TMA warp, m_tile: %d, n_tile: %d, before try wait ab empty barrier\n", m_tile, n_tile);
                }
                peek_ab_empty_status = kernel::try_wait_barrier(
                    shared_storage.ab_empty_mbar_ptr[smem_wr_buffer_next],
                    tma_wr_ab_empty_phase_next);
                if (lane_idx == 0) {
                    printf("in TMA warp, m_tile: %d, n_tile: %d, after try wait ab empty barrier\n", m_tile, n_tile);
                }
              }
    
              tma_wr_k_tile = tma_wr_k_tile_next;
              smem_wr_buffer = smem_wr_buffer_next;
              tma_wr_ab_empty_phase = tma_wr_ab_empty_phase_next;
    
            } // end for k_tile
    
          } // end for n_tile
        }
      } else if (warp_idx == 4) {
        // MMA warp (1)

        // Wait for TMEM allocation to complete
        tmem_allocation_result_barrier.arrive_and_wait();
        tCtAcc.data() = shared_storage.tmem_base_ptr;
    
        int total_k_tile_count = 0;
        int num_tiles_executed = 0;
        for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
          for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {
    
            if (lane_idx == 0) {
                printf("in MMA warp, m_tile: %d, n_tile: %d, before acc empty barrier\n", m_tile, n_tile);
            }

            int acc_buf_idx = num_tiles_executed % NUM_ACC_STAGE;
            auto tCtAcc_Slice = tCtAcc(cute::_, cute::_, cute::_, acc_buf_idx);
    
            int num_prev_k_blk = total_k_tile_count;
            total_k_tile_count += k_tile_count;
    
            int mma_rd_k_tile = 0;
            int smem_rd_buffer = (num_prev_k_blk + mma_rd_k_tile) % NUM_AB_STAGE;
            int mma_rd_ab_full_phase =
                (num_prev_k_blk + mma_rd_k_tile) / NUM_AB_STAGE % 2;
    
            // Peek full phase
            bool peek_ab_full_status = kernel::try_wait_barrier(
                shared_storage.ab_full_mbar_ptr[smem_rd_buffer],
                mma_rd_ab_full_phase);
    
            int acc_empty_phase = num_tiles_executed / NUM_ACC_STAGE % 2 ^ 1;
            cute::wait_barrier(shared_storage.acc_empty_mbar_ptr[acc_buf_idx],
                               acc_empty_phase);
    
            if (lane_idx == 0) {
                printf("in MMA warp, m_tile: %d, n_tile: %d, after acc empty barrier\n", m_tile, n_tile);
            }

            // Initialize the accumulator to zero
            tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;
    
            for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
              int mma_rd_k_tile_next = mma_rd_k_tile + 1;
              int smem_rd_buffer_next =
                  (num_prev_k_blk + mma_rd_k_tile_next) % NUM_AB_STAGE;
              int mma_rd_ab_full_phase_next = smem_rd_buffer_next == 0
                                                  ? mma_rd_ab_full_phase ^ 1
                                                  : mma_rd_ab_full_phase;
    
              if (lane_idx == 0) {
                printf("in MMA warp, m_tile: %d, n_tile: %d, before ab full barrier\n", m_tile, n_tile);
              }

              if (!peek_ab_full_status) {
                cute::wait_barrier(shared_storage.ab_full_mbar_ptr[smem_rd_buffer],
                                   mma_rd_ab_full_phase);
              }
    
              // if (!peek_ab_full_status){
              //   wait(ab_full_mbar_ptr[smem_rd_buffer], mma_rd_ab_full_phase);
              // }
    
              if (lane_idx == 0) {
                printf("in MMA warp, m_tile: %d, n_tile: %d, after ab full barrier\n", m_tile, n_tile);
              }

              // Perform MMA operation
              for (int k_block = 0; k_block < cute::size<2>(tCrA); ++k_block) {
                cute::gemm(tiled_mma,
                           tCrA(cute::_, cute::_, k_block, smem_rd_buffer),
                           tCrB(cute::_, cute::_, k_block, smem_rd_buffer),
                           tCtAcc_Slice);
                tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
              }
    
              cutlass::arch::umma_arrive(
                  &shared_storage.ab_empty_mbar_ptr[smem_rd_buffer]);
    
              if (mma_rd_k_tile_next < k_tile_count) {
                peek_ab_full_status = kernel::try_wait_barrier(
                    shared_storage.ab_full_mbar_ptr[smem_rd_buffer_next],
                    mma_rd_ab_full_phase_next);
              }
    
              mma_rd_k_tile = mma_rd_k_tile_next;
              smem_rd_buffer = smem_rd_buffer_next;
              mma_rd_ab_full_phase = mma_rd_ab_full_phase_next;
    
            } // end for k_tile

            if (lane_idx == 0) {
                printf("in MMA warp, m_tile: %d, n_tile: %d, before arrive acc full barrier\n", m_tile, n_tile);
            }
    
            cutlass::arch::umma_arrive(
                &shared_storage.acc_full_mbar_ptr[acc_buf_idx]);
            num_tiles_executed++;

            if (lane_idx == 0) {
                printf("in MMA warp, m_tile: %d, n_tile: %d, after arrive acc full barrier\n", m_tile, n_tile);
            }
    
          } // end for n_tile
        }
      } else if (warp_idx < 4) {
        // Epilogue warps (4)
    
        // Allocate TMEM for accumulators
        if (warp_idx == 0) {
          tmem_allocator.allocate(num_tmem_columns, &shared_storage.tmem_base_ptr);
        }
        tmem_allocation_result_barrier.arrive_and_wait();
        tCtAcc.data() = shared_storage.tmem_base_ptr;
    
        using AccType = typename decltype(tCtAcc)::value_type;
        using TypeBias = T_;
        using TypeC = T_;
    
        cutlass::NumericConverter<AccType, TypeBias> converterBias;
        cutlass::NumericConverter<TypeC, AccType> converter;
    
        cute::TiledCopy tiled_copy_t2r =
            cute::make_tmem_copy(cute::SM100_TMEM_LOAD_32dp32b1x{},
                                 tCtAcc(cute::_, cute::_, cute::_, 0)); // (128,32)
        cute::ThrCopy thr_copy_t2r = tiled_copy_t2r.get_slice(threadIdx.x);
        cute::Tensor tTR_tAcc = thr_copy_t2r.partition_S(
            tCtAcc); // tmem_[32b](0x0000.0000) o
                     // ((_32,_1),_32,_1,_1,_2):((_65536,_0),_1,_0,_0,_32)
    
        cute::Tensor tCgC_fake = cute::make_tensor<TypeC>(cute::shape(
            tCtAcc(cute::_, cute::_, cute::_, 0))); // (T2R, T2R_M, T2R_N)
        cute::Tensor tTR_rAcc_fake = thr_copy_t2r.partition_D(tCgC_fake);
        cute::Tensor tTR_rAcc = cute::make_tensor<AccType>(
            cute::shape(tTR_rAcc_fake)); // ptr[32b](some_ptr) o
                                         // ((_1,_1),_32,_1,_1):((_0,_0),_1,_0,_0)
    
    
        int num_tiles_executed = 0;
        for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
          for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {
            int acc_buf_idx = num_tiles_executed % NUM_ACC_STAGE;
            int acc_full_phase = num_tiles_executed / NUM_ACC_STAGE % 2;
            int c_smem_wr_buffer_idx = num_tiles_executed % NUM_C_STAGE;
    
            cute::Tensor tCgBias =
                gBias(cute::_, cute::_, n_tile, m_tile); // (Mma_M, Mma_N)
            cute::Tensor tCrBiasTypeBias = cute::make_tensor<TypeBias>(
                cute::shape(tTR_rAcc(0, cute::_, 0, 0))); // (T2R_M, T2R_N)
            cute::Tensor tCrBiasTypeAcc =
                cute::make_tensor<AccType>(cute::shape(tCrBiasTypeBias));
            cute::Tensor tCrC =
                cute::make_tensor<TypeC>(cute::shape(tCrBiasTypeBias));

    
            // T2R and register operations
            if constexpr (!NOBIAS) {
              // this copy might conflict with TMA load, might add a wait barrier if
              // needed
              cute::copy(tCgBias(cute::_, threadIdx.x), tCrBiasTypeBias);
              // optimize with vectorized type conversion
    
              CUTE_UNROLL
              for (int i = 0; i < tCrBiasTypeBias.size(); i++) {
                tCrBiasTypeAcc[i] = converterBias(tCrBiasTypeBias[i]);
              }
            }
    
            mm_output_smem.set_ptr(mm_output +
                                   c_smem_wr_buffer_idx * MMA_N * OUTPUT_ATOM_SIZE);
    
            if (lane_idx == 0) {
                printf("in epilogue warp, m_tile: %d, n_tile: %d, before acc full barrier\n", m_tile, n_tile);
            }

            cute::wait_barrier(shared_storage.acc_full_mbar_ptr[acc_buf_idx],
                               acc_full_phase);
            // T2R copy
            cute::copy(tiled_copy_t2r,
                       tTR_tAcc(cute::_, cute::_, cute::_, cute::_, acc_buf_idx),
                       tTR_rAcc);
    
            if (lane_idx == 0) {
                printf("in epilogue warp, m_tile: %d, n_tile: %d, after acc full barrier\n", m_tile, n_tile);
            }

            // arrive acc empty buffer
            epilogue_wg_barrier.arrive_and_wait();
            if (cute::elect_one_sync()) {
              printf("in epilogue warp, before arrive acc empty barrier\n");
              cute::arrive_barrier(shared_storage.acc_empty_mbar_ptr[acc_buf_idx]);
              printf("in epilogue warp, after arrive acc empty barrier\n");
            }
    
            if constexpr (!NOBIAS) {
              CUTE_UNROLL
              for (int i = 0; i < tTR_rAcc.size(); i++) {
                tTR_rAcc[i] += tCrBiasTypeAcc[i];
              }
            }
    
            CUTE_UNROLL
            for (int i = 0; i < tCrC.size(); i++) {
              tCrC[i] = converter(tTR_rAcc[i]);
            }

            if (lane_idx == 0) {
                printf("in epilogue warp, m_tile: %d, n_tile: %d, before R2S copy\n", m_tile, n_tile);
            }
    
            // R2S copy
            cute::Tensor sC_epi_slice =
                cute::flatten(sC_epi(cute::_, 0, 0, c_smem_wr_buffer_idx));
            cute::copy(tCrC, sC_epi_slice(cute::_, threadIdx.x));
    
            if (lane_idx == 0) {
                printf("in epilogue warp, m_tile: %d, n_tile: %d, after R2S copy\n", m_tile, n_tile);
            }

            // S2G TMA
            cute::tma_store_fence(); // Ensure C smem stores are visible to TMA
            if (lane_idx == 0) {
                printf("in epilogue warp, m_tile: %d, n_tile: %d, before arrive epilogue_wg_barrier\n", m_tile, n_tile);
            }
            epilogue_wg_barrier
                .arrive_and_wait(); // Ensure all threads have issued fence
            if (lane_idx == 0) {
                printf("in epilogue warp, m_tile: %d, n_tile: %d, after arrive epilogue_wg_barrier\n", m_tile, n_tile);
            }

            if (warp_idx == 0 && cute::elect_one_sync()) {
              if constexpr (SplitK) {
                tma_out.tma_reduce_add_async(
                    mm_output_smem.base_ptr,
                    {m_tile * OUTPUT_ATOM_SIZE, n_tile * MMA_N});
              } else {
                printf("mm_output_smem.base_ptr: %p, m_tile*OUTPUT_ATOM_SIZE: %d, n_tile*MMA_N: %d\n", mm_output_smem.base_ptr, m_tile*OUTPUT_ATOM_SIZE, n_tile*MMA_N);
                tma_out.tma_store_async(
                    mm_output_smem.base_ptr,
                    {m_tile * OUTPUT_ATOM_SIZE, n_tile * MMA_N});
              }

              if (lane_idx == 0) {
                printf("in epilogue warp, before tma store arrive\n");
              }
    
              cute::tma_store_arrive();
              cute::tma_store_wait<NUM_C_STAGE - 1>();
              if (lane_idx == 0) {
                printf("in epilogue warp, after tma store wait\n");
              }
            }
    
            num_tiles_executed++;
          }
        }
        if (lane_idx == 0) {
            printf("in epilogue warp, before tma store wait\n");
        }
        // wait all TMA stores to complete
        if (warp_idx == 0 && cute::elect_one_sync()) {
          cute::tma_store_wait<0>();
        }
        if (lane_idx == 0) {
            printf("in epilogue warp, after tma store wait\n");
        }
      }
      __syncthreads();

    // Release the right to allocate before deallocations so that the next CTA can
    // rasterize Then deallocate TMEM
    if (warp_idx == 0) {
      // don't do relinquish for megakernel
      // tmem_allocator.release_allocation_lock();
      tmem_allocator.free(shared_storage.tmem_base_ptr, num_tmem_columns);
    }
} // end linear_fp8_1d2d_sm100_mpk_task_impl

};  // namespace kernel