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
#include "cute/tensor.hpp"
#include "cute/util/debug.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "../common/dmem_layout.cuh"
#include "../common/worker_config.h"
#include "../hopper/barrier.cuh"
#include "../hopper/smem_layout_tma.cuh"
#include "../hopper/tma.cuh"
#include "storage.cuh"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)


namespace kernel {

using namespace cute;   
using namespace cutlass;

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

//  C = (A * SFA) x (B * SFB) + bias
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
          int SCALE_VECTOR_SIZE,
          bool NOBIAS,
          bool SplitK,
          int NUM_AB_STAGE = 8,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE = 4>
__device__ __noinline__ void
linear_nvfp4_1d2d_sm100_task_impl(const TMA_A &tma_a,
                                  const TMA_B &tma_b,
                                  const TMA_SFA &tma_sfa,
                                  const TMA_SFB &tma_sfb,
                                  BiasTensor mBias,
                                  const TMA_OUT &tma_out,
                                  uint32_t* debug_smem_sfa,
                                  uint32_t* debug_smem_sfb,
                                  uint32_t* debug_sfa_out,
                                  uint32_t* debug_sfb_out) {

    /*
        Naming convention:
            m{Matrix}   - Logical/full matrix view (coordinate tensor)
            g{Matrix}   - Global memory
            s{Matrix}   - Shared memory
            r{Matrix}   - Register
            t{Matrix}   - TMEM storage

            tCg{Matrix} - This CTA's Global view (thread-partitioned)
            tCs{Matrix} - This CTA's Shared view
            tCf{Matrix} - This CTA's Shared fragment/descriptor view (required for SMEM -> MMA)
            tCr{Matrix} - This CTA's Register view
            tTR_{...}   - Thread's T2R (TMEM-to-Register) view

            A     - Left matrix (weights)
            B     - Right matrix (input/activation)
            C     - Output matrix
            Acc   - Accumulator
            SFA   - Scale factors for A
            SFB   - Scale factors for B
            Bias  - Bias vector
    */

    // MMA_N >= 32 for SFB SMEM -> TMEM transfer correctness
    static_assert(std::is_same_v<T_, cutlass::float_e2m1_t>, "T_ must be cutlass::float_e2m1_t");
    static_assert(SCALE_VECTOR_SIZE == 16, "SCALE_VECTOR_SIZE must be 16");
    static_assert(MMA_M == 128, "MMA_M must be 128");
    static_assert(MMA_N % 8 == 0 && MMA_N != 0 && MMA_N <=256, "MMA_N must be {8, 16, … 256} steps of 8"); 
    constexpr int MMA_K = 64; // SM100_MMA_MXF4_SS forces MMA_K to be 64
    
    // tCgA - Matrix A in global memory (source for TMA loads)
    // tCsA - Matrix A in shared memory (staging buffer, receives TMA data)
    // tCfA - SMEM descriptor pointing to tCsA (direct input to MMA instructions)

    using A_type = cutlass::float_e2m1_t;
    using B_type = cutlass::float_e2m1_t;
    using SF_type = cutlass::float_ue4m3_t;
    using C_type = float;

    constexpr int B = 0;
    constexpr int M = 3;
    constexpr int S = 3;

    constexpr int NUM_MMA_M = 1;
    constexpr int NUM_MMA_N = 1;
    constexpr int NUM_MMA_K = 1;

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = kernel::lane_id();

    // TODO: SM100_MMA_MXF4_SS vs SM100_MMA_MXF4_2x1SM_SS
    cute::TiledMMA tiled_mma = cute::make_tiled_mma(
                cute::SM100_MMA_MXF4_SS<A_type,
                                        B_type,
                                        C_type,
                                        SF_type,
                                        MMA_M,
                                        MMA_N,
                                        SCALE_VECTOR_SIZE,
                                        cute::UMMA::Major::K,
                                        cute::UMMA::Major::K>{}
    );

    constexpr auto bM = MMA_M * NUM_MMA_M;
    constexpr auto bN = MMA_N * NUM_MMA_N;
    constexpr auto bK = MMA_K * NUM_MMA_K;

    if (cute::thread0()) {
        cute::print("\nMMA_M: %d", MMA_M);
        cute::print("\nMMA_N: %d", MMA_N);
        cute::print("\nMMA_K: %d", MMA_K);

        cute::print("\nNUM_MMA_M: %d", NUM_MMA_M);
        cute::print("\nNUM_MMA_N: %d", NUM_MMA_N);
        cute::print("\nNUM_MMA_K: %d", NUM_MMA_K);
        
        cute::print("\nSCALE_VECTOR_SIZE: %d", SCALE_VECTOR_SIZE);

        cute::print("\nbM : %d", bM);
        cute::print("\nbN : %d", bN);
        cute::print("\nbK : %d", bK);
    }

    using TiledMma              = decltype(tiled_mma);
    using ThrBlkTileShape_MNK   = cute::Shape<cute::Int<bM>, cute::Int<bN>, cute::Int<bK>>;
    using Sm1xxBlkScaledConfig  = cutlass::detail::Sm1xxBlockScaledConfig<SCALE_VECTOR_SIZE>;
    using SmemLayoutAtomSFA     = decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFA(TiledMma{}, ThrBlkTileShape_MNK{}));
    using SmemLayoutAtomSFB     = decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFB(TiledMma{}, ThrBlkTileShape_MNK{}));

    auto tb_mma_coord_vmnk = cute::make_coord(0,           // Peer CTA coordinate
                                              cute::_,     //    MMA-M coordinate
                                              cute::_,     //    MMA-N coordinate
                                              cute::_);    //    MMA-K coordinate
    auto tb_mma_coord = cute::select<1, 2, 3>(tb_mma_coord_vmnk);
    auto tb_mma_tiler = ThrBlkTileShape_MNK{};
    auto tb_cd_tiler  = cute::make_shape(cute::Int<bN>{}, cute::Int<bM>{}, cute::Int<bK>{});
    auto tb_sf_tiler  = cute::make_shape(cute::Int<bM>{}, cute::Int<bN>{}, cute::Int<bK / SCALE_VECTOR_SIZE>{});

    // Coordinate tensor for matrix
    cute::Tensor mA = cute::make_coord_tensor(
        cute::make_layout(
            cute::make_shape(OUTPUT_SIZE, REDUCTION_SIZE),  // (128, 768) : (1,0)
            cute::make_stride(cute::E<1>{}, cute::E<0>{})
        )
    ); 
    cute::Tensor mB = cute::make_coord_tensor(
        cute::make_layout(
            cute::make_shape(BATCH_SIZE, REDUCTION_SIZE),   // (1,768):(1,0)
            cute::make_stride(cute::E<1>{}, cute::E<0>{})
        )
    ); 
    cute::Tensor mC = cute::make_coord_tensor(
        cute::make_layout(
            cute::make_shape(BATCH_SIZE, OUTPUT_SIZE),      // (1,128):(1,0)
            cute::make_stride(cute::E<1>{}, cute::E<0>{})
        )
    ); 
    cute::Tensor mSFA = cute::make_coord_tensor(
        cute::make_layout(
            cute::make_shape(OUTPUT_SIZE, REDUCTION_SIZE / SCALE_VECTOR_SIZE),  // (128, 768 / SV):(1,0)
            cute::make_stride(cute::E<1>{},cute::E<0>{})
        )
    ); 
    cute::Tensor mSFB = cute::make_coord_tensor(
        cute::make_layout(
            cute::make_shape(BATCH_SIZE, REDUCTION_SIZE / SCALE_VECTOR_SIZE),   // (1, 768 / SV):(1,0)
            cute::make_stride(cute::E<1>{},cute::E<0>{})
        )
    ); 

    // CTA's global memory tile
    cute::Tensor gA = cute::local_tile(
        mA,
        tb_mma_tiler,
        tb_mma_coord,
        cute::Step<cute::_1,    // M
                    cute::X,    // N
                    cute::_1>{} // K
    );
    cute::Tensor gB = cute::local_tile(
        mB,
        tb_mma_tiler,
        tb_mma_coord,
        cute::Step<cute::X,     // M
                    cute::_1,   // N
                    cute::_1>{} // K
    );
    cute::Tensor gBias = cute::local_tile(
        mBias,
        tb_cd_tiler,
        tb_mma_coord,
        cute::Step<cute::_1,    // M
                   cute::_1,    // N
                   cute::X>{}   // K
    );
    cute::Tensor gSFA = cute::local_tile(
        mSFA,
        tb_sf_tiler,
        tb_mma_coord,
        cute::Step<cute::_1,    // M
                   cute::X,     // N
                   cute::_1>{}  // K
    );
    cute::Tensor gSFB = cute::local_tile(
        mSFB,
        tb_sf_tiler,
        tb_mma_coord,
        cute::Step<cute::X,     // M
                   cute::_1,    // N
                   cute::_1>{}  // K
    );

    // Partition global tensors into tiles
    auto cta_ind = cute::get<0>(tb_mma_coord_vmnk);
    cute::ThrMMA cta_mma = tiled_mma.get_slice(cta_ind);    // Use Peer CTA coordinate
    cute::Tensor tCgA    = cta_mma.partition_A(gA);         // ((MMA_M, MMA_K), NUM_MMA_M, NUM_MMA_K, REDUCTION_SIZE      / bK)
    cute::Tensor tCgB    = cta_mma.partition_B(gB);         // ((MMA_N, MMA_K), NUM_MMA_N, NUM_MMA_K, REDUCTION_SIZE      / bK)
    cute::Tensor tCgSFA  = cta_mma.partition_A(gSFA);       // ((MMA_M, MMA_K), NUM_MMA_M, NUM_MMA_K, REDUCTION_SIZE / 16 / bK)
    cute::Tensor tCgSFB  = cta_mma.partition_B(gSFB);       // ((MMA_N, MMA_K), NUM_MMA_N, NUM_MMA_K, REDUCTION_SIZE / 16 / bK)
    if (cute::thread0()) {
        cute::print("\n");
        cute::print("tCgA layout/shape:     "); cute::print(tCgA); cute::print("\n");
        cute::print("tCgB layout/shape:     "); cute::print(tCgB); cute::print("\n");
        cute::print("tCgSFA layout/shape:   "); cute::print(tCgSFA); cute::print("\n");
        cute::print("tCgSFB layout/shape:   "); cute::print(tCgSFB); cute::print("\n");
        cute::print("\n");
    }


    // CTA's shared memory tile
    // ((MMA_M, MMA_K), NUM_MMA_M, NUM_MMA_K, NUM_AB_STAGE)
    auto mma_shape_A = cute::partition_shape_A(
        tiled_mma,                          // (MMA_M, MMA_K)
        cute::make_shape(
            cute::Int<bM>{},                // MMA_M * NUM_MMA_M
            cute::Int<bK>{},                // MMA_K * NUM_MMA_K
            cute::Int<NUM_AB_STAGE>{}       // NUM_AB_STAGE
        )
    );
    // ((MMA_N, MMA_K), NUM_MMA_N, NUM_MMA_K, NUM_AB_STAGE)
    auto mma_shape_B = cute::partition_shape_B(
        tiled_mma,                          // (MMA_N, MMA_K)
        cute::make_shape(
            cute::Int<bN>{},                // MMA_N * NUM_MMA_N
            cute::Int<bK>{},                // MMA_K * NUM_MMA_K
            cute::Int<NUM_AB_STAGE>{}       // NUM_AB_STAGE
        )
    );
    // ((MMA_N, MMA_M), 1, 1, NUM_C_STAGE)
    auto mma_shape_C = cute::make_shape(
        cute::make_shape(cute::Int<MMA_N>{}, cute::Int<MMA_M>{}),
        cute::Int<1>{},
        cute::Int<1>{},
        cute::Int<NUM_C_STAGE>{}
    );
    if (cute::thread0()) {
        cute::print("\n");
        cute::print("mma_shape_A: "); cute::print(mma_shape_A); cute::print("\n");
        cute::print("mma_shape_B: "); cute::print(mma_shape_B); cute::print("\n");
        cute::print("mma_shape_C: "); cute::print(mma_shape_C); cute::print("\n");
        cute::print("\n");
    }

    // Construct SMEM layout w/ swizzling mode
    // For float_e2m1_t (4-bit) with K-major, BLK_K=64:
    //   sm100_smem_selector picks Layout_K_SW32_Atom (size<1>=64, 64%64==0).
    //   SW128 needs K>=256, SW64 needs K>=128; SW32 is the correct choice.
    //   K_INTER_Atom has size<1>=32 so the u128-recast K-dim=1 which fails the
    //   canonical ((8,n),(2,1)) check and causes OOB descriptors at runtime.
    auto sA_layout      = cute::UMMA::tile_to_mma_shape(cute::UMMA::Layout_K_SW32_Atom<A_type>{}, mma_shape_A);
    auto sB_layout      = cute::UMMA::tile_to_mma_shape(cute::UMMA::Layout_K_SW32_Atom<B_type>{}, mma_shape_B);
    auto sC_layout_fake = cute::UMMA::tile_to_mma_shape(cute::UMMA::Layout_K_INTER_Atom<C_type>{}, mma_shape_C);
    auto sC_shape = cute::make_shape(
        cute::make_shape(cute::Int<MMA_N>{}, cute::Int<MMA_M>{}),
        cute::Int<1>{},
        cute::Int<1>{},
        cute::make_shape(cute::Int<1>{}, cute::Int<NUM_C_STAGE>{})
    );  // ((MMA_N, MMA_M), 1, 1, (1, NUM_C_STAGE))
    auto sC_stride = cute::make_stride(
        cute::make_stride(cute::Int<MMA_M>{}, cute::Int<1>{}),
        cute::Int<0>{},
        cute::Int<0>{},
        cute::make_stride(cute::Int<0>{}, cute::Int<MMA_M * MMA_N>{})
    );  // ((MMA_M, 1), 0, 0, (0, MMA_M * MMA_N))
    auto sC_layout = cute::composition(
        sC_layout_fake.layout_a(),
        sC_layout_fake.offset(),
        cute::make_layout(sC_shape, sC_stride)
    );
    auto sSFA_layout = cute::tile_to_shape(
        SmemLayoutAtomSFA{},
        cute::append(cute::shape(SmemLayoutAtomSFA{}), cute::Int<NUM_AB_STAGE>{})
    );
    auto sSFB_layout = cute::tile_to_shape(
        SmemLayoutAtomSFB{},
        cute::append(cute::shape(SmemLayoutAtomSFB{}), cute::Int<NUM_AB_STAGE>{})
    );
               
    if (cute::thread0()) {
        cute::print("\n");
        cute::print("sA_layout:    "); cute::print(sA_layout);    cute::print("\n");
        cute::print("sB_layout:    "); cute::print(sB_layout);    cute::print("\n");
        cute::print("sC_layout:    "); cute::print(sC_layout);    cute::print("\n");
        cute::print("SmemLayoutAtomSFA: "); cute::print(SmemLayoutAtomSFA{}); cute::print("\n");
        cute::print("SmemLayoutAtomSFB: "); cute::print(SmemLayoutAtomSFB{}); cute::print("\n");
        cute::print("sSFA_layout:  "); cute::print(sSFA_layout);  cute::print("\n");
        cute::print("sSFB_layout:  "); cute::print(sSFB_layout);  cute::print("\n");
        cute::print("  cosize(sSFA_layout) = %d\n", (int)cute::cosize(sSFA_layout));
        cute::print("  cosize(sSFB_layout) = %d\n", (int)cute::cosize(sSFB_layout));
        cute::print("  cosize(sSFA_layout)/NUM_AB_STAGE = %d  (bytes per stage)\n",
                    (int)cute::cosize(sSFA_layout) / NUM_AB_STAGE);
        cute::print("  cosize(sSFB_layout)/NUM_AB_STAGE = %d  (bytes per stage)\n",
                    (int)cute::cosize(sSFB_layout) / NUM_AB_STAGE);
        cute::print("\n");
    }

    using SharedStorage = PipedScaledSharedStorage<A_type,
                                                   B_type,
                                                   C_type,
                                                   SF_type,
                                                   decltype(sA_layout),
                                                   decltype(sB_layout),
                                                   decltype(sC_layout),
                                                   decltype(sSFA_layout),
                                                   decltype(sSFB_layout),
                                                   NUM_AB_STAGE,
                                                   NUM_ACC_STAGE>;

    if (cute::thread0()) {
        printf("sizeof(SharedStorage) = %zu bytes (%.1f KB)\n",
               sizeof(SharedStorage), sizeof(SharedStorage) / 1024.0f);
    }

    extern __shared__ char shared_memory[];
    uintptr_t aligned_smem = (reinterpret_cast<uintptr_t>(shared_memory) + 127) / 128 * 128;
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(aligned_smem);

    cute::Tensor tCsA   = shared_storage.tensor_sA();
    cute::Tensor tCsB   = shared_storage.tensor_sB();
    cute::Tensor tCsSFA = shared_storage.tensor_sSFA(); 
    cute::Tensor tCsSFB = shared_storage.tensor_sSFB();
    cute::Tensor tCsC   = shared_storage.tensor_sC();

    if (cute::thread0()) {
        cute::print("\n");
        cute::print("tCsA   : "); cute::print(tCsA);   cute::print("\n");
        cute::print("tCsB   : "); cute::print(tCsB);   cute::print("\n");
        cute::print("tCsSFA : "); cute::print(tCsSFA); cute::print("\n");
        cute::print("tCsSFB : "); cute::print(tCsSFB); cute::print("\n");
        cute::print("tCsC   : "); cute::print(tCsC); cute::print("\n");
    }

    // Initialize barriers
    if (warp_idx == 0) {
        cutlass::arch::detail::initialize_barrier_array_aligned<
            cutlass::arch::ClusterTransactionBarrier,
            NUM_AB_STAGE>(shared_storage.ab_full_mbar_ptr, 1);
        cutlass::arch::detail::initialize_barrier_array_aligned<
            cutlass::arch::ClusterBarrier,
            NUM_AB_STAGE>(shared_storage.ab_empty_mbar_ptr, 1);
        cutlass::arch::detail::initialize_barrier_array_aligned<
            cutlass::arch::ClusterBarrier,
            NUM_ACC_STAGE>(shared_storage.acc_full_mbar_ptr, 1);
        cutlass::arch::detail::initialize_barrier_array_aligned<
            cutlass::arch::ClusterBarrier,
            NUM_ACC_STAGE>(shared_storage.acc_empty_mbar_ptr, 4);
    }

    if (cute::thread0()) {
        cute::print("\tbarriers init");
    }

    // Sync tmem allocation status between MMA and epilogue warps within CTA
    // 32 threads (mma) + 128 threads (epilog) to sync
    cutlass::arch::NamedBarrier tmem_allocation_result_barrier(32 + 128, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
    cutlass::arch::NamedBarrier epilogue_wg_barrier(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

    // A and B: one TMA call loads MMA_K = 64 FP4 elements per slot (4-bit packed, 2/byte)
    // SFA/SFB: TMA tile is always (MMA_M, 16) and (MMA_N, 16) bytes regardless of bK.
    //   The tile covers 4 consecutive SMEM slots (4*MMA_K/SV SF columns) at once,
    //   so SFA/SFB TMA fires only every 4 outer k-tiles (k_tile % 4 == 0).
    constexpr int tma_transaction_bytes = MMA_M * MMA_K / 2   
                                        + MMA_N * MMA_K / 2   
                                        + MMA_M * 16
                                        + MMA_N * 16;
                                                              
    constexpr int tma_transaction_bytes_2 = MMA_M * MMA_K / 2
                                          + MMA_N * MMA_K / 2;

    // A_type *sA_ptr = shared_storage.A.begin();
    // B_type *sB_ptr = shared_storage.B.begin();
    void    *sA_ptr      = static_cast<void*>(&shared_storage.A);  // Has to be done since type is fp4 (TODO: fix?)
    void    *sB_ptr      = static_cast<void*>(&shared_storage.B);
    C_type  *sC_ptr      = shared_storage.C.begin();
    SF_type *sSFA_ptr    = shared_storage.SFA.begin();
    SF_type *sSFB_ptr    = shared_storage.SFB.begin();

    Barrier *ab_full_mbar_ptr = reinterpret_cast<Barrier *>(shared_storage.ab_full_mbar_ptr);

    // TODO: Create subbyte iterator?
    using A_smem_TMA = smem_tma<A_type, B, M, S, MMA_M, MMA_K, 1>;
    using B_smem_TMA = smem_tma<B_type, B, M, S, MMA_N, MMA_K, 1>;
    using C_smem_TMA = smem_tma<C_type, 0, M, S, MMA_N, MMA_M, 1>;
    using SFA_smem_TMA = smem_tma<SF_type, B, M, S, MMA_M, MMA_K / SCALE_VECTOR_SIZE, 1>;
    using SFB_smem_TMA = smem_tma<SF_type, B, M, S, MMA_N, MMA_K / SCALE_VECTOR_SIZE, 1>;

    A_smem_TMA sA(sA_ptr);
    B_smem_TMA sB(sB_ptr);
    C_smem_TMA sC(sC_ptr);
    SFA_smem_TMA sSFA(sSFA_ptr);
    SFB_smem_TMA sSFB(sSFB_ptr);

    cutlass::arch::fence_barrier_init();
    __syncthreads();

    // Prime the acc_empty pipeline: MMA waits on acc_empty[i] at phase=1 for the first
    // NUM_ACC_STAGE tiles. The barrier starts at phase 0 with count=4. To make phase=1
    // immediately satisfiable, we need:
    //   4 arrivals to complete phase 0 (transitions to phase 1 in-progress), then
    //   4 arrivals to complete phase 1 (transitions to phase 0 again, satisfying wait(1)).
    // Total: 8 arrivals per slot. 4 warps x 2 rounds x elect_one = 8 arrivals.
    // if (warp_idx < 4) {
    //     if (cute::elect_one_sync()) {
    //         for (int i = 0; i < NUM_ACC_STAGE; ++i) {
    //             // First round: complete phase 0
    //             cute::arrive_barrier(shared_storage.acc_empty_mbar_ptr[i]);
    //         }
    //     }
    // }
    // __syncthreads();
    // if (warp_idx < 4) {
    //     if (cute::elect_one_sync()) {
    //         for (int i = 0; i < NUM_ACC_STAGE; ++i) {
    //             // Second round: complete phase 1 (now wait(phase=1) passes immediately)
    //             cute::arrive_barrier(shared_storage.acc_empty_mbar_ptr[i]);
    //         }
    //     }
    // }
    // __syncthreads();

    // MMA Fragment Allocation
    // We allocate "fragments" which are SMEM descriptors that serve as inputs to
    // cute::gemm operations. For tcgen05.mma operations:
    // - Matrices A and B are sourced from SMEM
    // - tCfA and tCfB provide descriptor views of tCsA and tCsB respectively
    // - The first mode of each descriptor represents the SMEM for a single MMA
    // operation
    // For SFA/B, we need to move SMEM -> TMEM manually, so we create tensor memory

    // TODO: What should be the expected shape of tCfSFA/B?
    cute::Tensor tCfA = cta_mma.make_fragment_A(tCsA); // (1, NUM_MMA_M, NUM_MMA_K, NUM_AB_STAGE)
    cute::Tensor tCfB = cta_mma.make_fragment_B(tCsB); // (1, NUM_MMA_N, NUM_MMA_K, NUM_AB_STAGE)
    auto acc_shape = cute::partition_shape_C(
        tiled_mma,                          // MMA
        cute::make_shape(
            cute::size<0>(tb_mma_tiler),       // MMA_M,
            cute::size<1>(tb_mma_tiler),       // MMA_N
            cute::Int<NUM_ACC_STAGE>{}      // NUM_ACC_STAGE
        )
    ); // (MmaC, NumMma_M, NumMma_N, NumAcc_Stage)
    cute::Tensor tCtC = tiled_mma.make_fragment_C(acc_shape); // (1, MMA_M, MMA_N, NUM_ACC_STAGE)
    if (cute::thread0()) {
        cute::print("\n\tMMA Fragment Allocation");
    }

    // TODO: Allocator1Sm vs Allocator2Sm
    // TODO: Use num_tmem_columns, calculate this value - tmem_allocator.allocate
    // TODO: Check LayoutSFA/B

    // TMEM Allocation
    using UtccpOp = cute::SM100_UTCCP_4x32dp128bit_1cta;
    using TmemAllocator = cute::TMEM::Allocator1Sm;
    TmemAllocator tmem_allocator{};
    if (warp_idx == 0) {
        tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_acc_ptr);
    }
    __syncthreads();

    // Point tCtC at the allocated TMEM base
    tCtC.data() = cute::make_tmem_ptr<C_type>(shared_storage.tmem_acc_ptr);

    // DEBUG Check B: SMEM size, TMEM offsets, loop structure
    // if (cute::thread0()) {
    //     printf("sizeof(SharedStorage)=%zu  limit=%d  A=%zu  B=%zu  C=%zu  SFA=%zu  SFB=%zu\n",
    //            sizeof(shared_storage), 224 * 1024,
    //            sizeof(shared_storage.A), sizeof(shared_storage.B),
    //            sizeof(shared_storage.C), sizeof(shared_storage.SFA), sizeof(shared_storage.SFB));
    //     printf("tmem_acc_ptr=%u\n", shared_storage.tmem_acc_ptr);
    //     printf("k_tile_count=%d  outer_m_loop=%d  outer_n_loop=%d\n",
    //            (int)cute::size<4>(tCgA),
    //            (int)cute::size<3>(tCgA),
    //            (int)cute::size<3>(tCgB));
    //     cute::print("tCgA shape: "); cute::print(tCgA.shape()); cute::print("\n");
    // }

    static constexpr int kSFTmemCols = 4 * (MMA_K / SCALE_VECTOR_SIZE);  // = 4 * 4 = 16
    uint32_t sfa_offset = cutlass::detail::find_tmem_tensor_col_offset(tCtC);
    uint32_t sfb_offset = sfa_offset + kSFTmemCols;
    shared_storage.tmem_sfa_ptr = shared_storage.tmem_acc_ptr + sfa_offset;
    shared_storage.tmem_sfb_ptr = shared_storage.tmem_acc_ptr + sfb_offset;

    if (cute::thread0()) {
        printf("sfa_offset=%u  sfb_offset=%u  tmem_sfa_ptr=%u  tmem_sfb_ptr=%u\n",
               sfa_offset, sfb_offset,
               shared_storage.tmem_acc_ptr + sfa_offset,
               shared_storage.tmem_acc_ptr + sfb_offset);
    }


    // Create prototype TMEM tensor
    // TODO: MMA_K = TILE_SIZE ?
    // using LayoutSFA = Layout<Shape <Int<MMA_M>, Int<MMA_K/SCALE_VECTOR_SIZE>>, Stride<Int<MMA_K/SCALE_VECTOR_SIZE>, _1>>;
    // using LayoutSFB = Layout<Shape <Int<MMA_N>, Int<MMA_K/SCALE_VECTOR_SIZE>>, Stride<Int<MMA_K/SCALE_VECTOR_SIZE>, _1>>;
    using FrgTypeSFA = cute::UMMA::tmem_sf_frg<SF_type, SCALE_VECTOR_SIZE, 1, true>;
    using FrgTypeSFB = cute::UMMA::tmem_sf_frg<SF_type, SCALE_VECTOR_SIZE, 1, false>;
    static constexpr int MMA_M_SFB = ((MMA_M + 127) / 128) * 128;  // = 128
    auto sfa_tmem_shape = cute::make_shape(     // ((MMA_M, (16, 4)), NUM_MMA_M, NUM_MMA_K)
        cute::make_shape(
            cute::Int<MMA_M_SFB>{},
            cute::make_shape(
                cute::Int<SCALE_VECTOR_SIZE>{},
                cute::Int<MMA_K / SCALE_VECTOR_SIZE>{}
            )
        ),
        cute::Int<NUM_MMA_M>{},
        cute::Int<NUM_MMA_K>{}
    );
    static constexpr int MMA_N_SFB = ((MMA_N + 127) / 128) * 128;  // = 128
    auto sfb_tmem_shape = cute::make_shape(     // ((MMA_N_SFB, (16, 4)), NUM_MMA_N, NUM_MMA_K)
        cute::make_shape(
            cute::Int<MMA_N_SFB>{},
            cute::make_shape(
                cute::Int<SCALE_VECTOR_SIZE>{},
                cute::Int<MMA_K / SCALE_VECTOR_SIZE>{}
            )
        ),
        cute::Int<NUM_MMA_N>{},
        cute::Int<NUM_MMA_K>{}
    );
    auto tCtSFA = FrgTypeSFA::make(sfa_tmem_shape);
    auto tCtSFB = FrgTypeSFB::make(sfb_tmem_shape);
    tCtSFA.data() = make_tmem_ptr<SF_type>(shared_storage.tmem_sfa_ptr);
    tCtSFB.data() = make_tmem_ptr<SF_type>(shared_storage.tmem_sfb_ptr);
    // if (cute::thread0()) {
    //     cute::print("\n");
    //     cute::print("tCtSFA:"); cute::print(tCtSFA); cute::print("\n");
    //     cute::print("tCtSFB:"); cute::print(tCtSFB); cute::print("\n");
    //     cute::print("\n");
    // }

    // Precompute TMEM column start for SFA/SFB
    uint32_t tmem_col_sfa = shared_storage.tmem_sfa_ptr;
    uint32_t tmem_col_sfb = shared_storage.tmem_sfb_ptr;
    int k_tile_count = cute::size<4>(tCgA);
    if (lane_idx == 0) {
        printf("[warp %d] entering warp branch, k_tile_count=%d\n", warp_idx, k_tile_count);
    }
    if (warp_idx == 5) {
        // TMA warp (1)
        if (lane_idx == 0) printf("[TMA warp] entering TMA loop\n");
        int total_k_tile_count = 0;
        for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
            for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {
        
                int num_prev_k_blk = total_k_tile_count;
                total_k_tile_count += k_tile_count;
        
                int tma_wr_k_tile = 0;
                int smem_wr_buffer = (num_prev_k_blk + tma_wr_k_tile) % NUM_AB_STAGE;
                int tma_wr_ab_empty_phase = (num_prev_k_blk + tma_wr_k_tile) / NUM_AB_STAGE % 2 ^ 1;
                bool peek_ab_empty_status = kernel::try_wait_barrier(
                    shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                    tma_wr_ab_empty_phase
                );
        
                // CUTE_UNROLL
                for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        
                    int tma_wr_k_tile_next  = tma_wr_k_tile + 1;
                    int smem_wr_buffer_next = (num_prev_k_blk + tma_wr_k_tile_next) % NUM_AB_STAGE;
                    int tma_wr_ab_empty_phase_next = smem_wr_buffer_next == 0
                                                        ? tma_wr_ab_empty_phase ^ 1
                                                        : tma_wr_ab_empty_phase;

                    // Wait for an empty buffer
                    if (!peek_ab_empty_status) {
                        cute::wait_barrier(shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                                        tma_wr_ab_empty_phase);
                    }
        
                    if (cute::elect_one_sync()) {
                        // TMA tile sizes:
                        // A:   MMA_M x 32 bytes = MMA_M x 64 elements
                        // B:   MMA_N x 32 bytes = MMA_N x 64 elements
                        // SFA: MMA_M x 16 bytes = MMA_M x 16 elements 
                        // SFB: MMA_N x 16 bytes = MMA_N x 16 elements 
                        // One MMA tile needs 4 SF elements, so only transfer scaling factor once every 4 tiles
                        int tma_coords_A[2] = {
                            k_tile * MMA_K,
                            m_tile * MMA_M
                        };
                        int tma_coords_B[2] = {
                            k_tile * MMA_K, 
                            n_tile * MMA_N
                        };
                        // SF TMA fires every 4 k_tiles; each call advances by MMA_K/SV=4 SF columns.
                        // k_tile * (MMA_K/SV) would skip 4× too far — use (k_tile/4) instead.
                        int tma_coords_SFA[2] = {
                            (k_tile) * (MMA_K / SCALE_VECTOR_SIZE),
                            m_tile * MMA_M
                        };
                        int tma_coords_SFB[2] = {
                            (k_tile) * (MMA_K / SCALE_VECTOR_SIZE),
                            n_tile * MMA_N
                        };

                        constexpr int A_buffer_bytes = MMA_M * MMA_K / 2;  // 4-bit = 2 elements per byte
                        constexpr int B_buffer_bytes  = MMA_N * MMA_K / 2;
                        // SF TMA tile width = 16 bytes (UINT8 TMA minimum)
                        // This must be 4
                        constexpr int sf_smem_col      = 4;
                        constexpr int SFA_buffer_bytes = MMA_M * sf_smem_col;
                        constexpr int SFB_buffer_bytes = MMA_N * sf_smem_col;

                        sA.set_ptr(static_cast<char*>(sA_ptr) + smem_wr_buffer * A_buffer_bytes);
                        sB.set_ptr(static_cast<char*>(sB_ptr) + smem_wr_buffer * B_buffer_bytes);
                        sSFA.set_ptr(sSFA_ptr + smem_wr_buffer * SFA_buffer_bytes);
                        sSFB.set_ptr(sSFB_ptr + smem_wr_buffer * SFB_buffer_bytes);

                        cute::set_barrier_transaction_bytes(
                            shared_storage.ab_full_mbar_ptr[smem_wr_buffer],
                            (k_tile % 4 == 0) ? tma_transaction_bytes : tma_transaction_bytes_2
                        );

                        tma_a.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer], sA.base_ptr, tma_coords_A);
                        tma_b.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer], sB.base_ptr, tma_coords_B);
                        if (k_tile % 4 == 0) {
                            tma_sfa.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer], sSFA.base_ptr, tma_coords_SFA);
                            tma_sfb.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer], sSFB.base_ptr, tma_coords_SFB);
                        }
                    }
    
                    if (tma_wr_k_tile_next < k_tile_count) {
                        peek_ab_empty_status = kernel::try_wait_barrier(
                            shared_storage.ab_empty_mbar_ptr[smem_wr_buffer_next],
                            tma_wr_ab_empty_phase_next);
                    }
            
                    tma_wr_k_tile = tma_wr_k_tile_next;
                    smem_wr_buffer = smem_wr_buffer_next;
                    tma_wr_ab_empty_phase = tma_wr_ab_empty_phase_next;
        
            } // end for k_tile
          } // end for n_tile
        } // end for m_tile
      } else if (warp_idx == 4) {
        // MMA warp (1)
        if (lane_idx == 0) printf("[MMA warp] before tmem_allocation_result_barrier\n");
        tmem_allocation_result_barrier.arrive_and_wait();
        if (lane_idx == 0) printf("[MMA warp] after tmem_allocation_result_barrier\n");
        
        int total_k_tile_count = 0;
        int num_tiles_executed = 0;
        for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
            for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {    
                int acc_buf_idx = num_tiles_executed % NUM_ACC_STAGE;
                int num_prev_k_blk = total_k_tile_count;
                total_k_tile_count += k_tile_count;
        
                int mma_rd_k_tile        = 0;
                int smem_rd_buffer       = (num_prev_k_blk + mma_rd_k_tile) % NUM_AB_STAGE;
                int mma_rd_ab_full_phase = (num_prev_k_blk + mma_rd_k_tile) / NUM_AB_STAGE % 2;
        
                // Wait until accumulation buffer is free
                int acc_empty_phase = num_tiles_executed / NUM_ACC_STAGE % 2 ^ 1;
                cute::wait_barrier(
                    shared_storage.acc_empty_mbar_ptr[acc_buf_idx],
                    acc_empty_phase
                );
        
                // Initialize the TMEM accumulator to zero
                tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;
                for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
                    int mma_rd_k_tile_next = mma_rd_k_tile + 1;
                    int smem_rd_buffer_next = (num_prev_k_blk + mma_rd_k_tile_next) % NUM_AB_STAGE;
                    int mma_rd_ab_full_phase_next = smem_rd_buffer_next == 0
                                                    ? mma_rd_ab_full_phase ^ 1
                                                    : mma_rd_ab_full_phase;
            
                    // Wait for A, B, SFA, SFB to load into SMEM
                    cute::wait_barrier(
                        shared_storage.ab_full_mbar_ptr[smem_rd_buffer],
                        mma_rd_ab_full_phase
                    );

                    // UTCCP: SMEM→TMEM scale-factor copy (done by MMA warp)
                    {
                        constexpr int kUTCCPSFCols = kSFTmemCols;  // 16
                        constexpr int MMA_K_SF = MMA_K / SCALE_VECTOR_SIZE;  // 4

                        // Index into the staged SMEM tensor to get the base pointer for this pipeline slot
                        SF_type* sfa = tCsSFA(cute::_, cute::_, cute::_, smem_rd_buffer).data().get();
                        SF_type* sfb = tCsSFB(cute::_, cute::_, cute::_, smem_rd_buffer).data().get();

                        if (cute::elect_one_sync()) {
                            uint32_t sfa_addr = cute::cast_smem_ptr_to_uint(sfa);
                            uint64_t sfa_desc = matrix_descriptor_encode((uint64_t)(sfa_addr)) |
                                                (1llu << 46) |
                                                matrix_descriptor_encode((uint64_t)16) << 16 | // LBO
                                                matrix_descriptor_encode((uint64_t)128) << 32 | // SBO
                                                (uint64_t) 0 << 62;                             // swizzle
                            printf("[MMA warp k=%d] BEFORE tcgen05.cp SFA: sfa_addr=0x%08x tmem_col_sfa=%u desc=0x%016llx\n",
                                k_tile, sfa_addr, tmem_col_sfa, (unsigned long long)sfa_desc);
                            asm volatile("{tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;}" :: "r"(tmem_col_sfa), "l"(sfa_desc));
                            printf("[MMA warp k=%d] AFTER tcgen05.cp SFA\n", k_tile);

                            uint32_t sfb_addr = cute::cast_smem_ptr_to_uint(sfb);
                            uint64_t sfb_desc = matrix_descriptor_encode((uint64_t)(sfb_addr)) |
                                                (1llu << 46) |
                                                matrix_descriptor_encode((uint64_t)16) << 16 | // LBO
                                                matrix_descriptor_encode((uint64_t)128) << 32 | // SBO
                                                (uint64_t) 0 << 62;                             // swizzle
                            printf("[MMA warp k=%d] BEFORE tcgen05.cp SFB: sfb_addr=0x%08x tmem_col_sfb=%u desc=0x%016llx\n",
                                k_tile, sfb_addr, tmem_col_sfb, (unsigned long long)sfb_desc);
                            asm volatile("{tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;}" :: "r"(tmem_col_sfb), "l"(sfb_desc));
                            printf("[MMA warp k=%d] AFTER tcgen05.cp SFB\n", k_tile);
                        }
                    }

                    auto accumulate = tiled_mma.accumulate_;
                    if (lane_idx == 0) printf("[MMA warp k=%d] BEFORE gemm loop size<2>(tCfA)=%d\n", k_tile, (int)cute::size<2>(tCfA));
                    for (int k_block = 0; k_block < cute::size<2>(tCfA); ++k_block) {
                        if (lane_idx == 0) printf("[MMA warp k=%d k_block=%d] BEFORE cute::gemm\n", k_tile, k_block);
                        cute::gemm(
                            tiled_mma.with(
                                accumulate,
                                tCtSFA(cute::_, cute::_, k_block),
                                tCtSFB(cute::_, cute::_, k_block)
                            ),
                            tCfA(cute::_, cute::_, k_block, smem_rd_buffer),
                            tCfB(cute::_, cute::_, k_block, smem_rd_buffer),
                            tCtC(cute::_, cute::_, cute::_, acc_buf_idx)
                        );
                        if (lane_idx == 0) printf("[MMA warp k=%d k_block=%d] AFTER cute::gemm\n", k_tile, k_block);
                        accumulate = cute::UMMA::ScaleOut::One;
                    }
                    tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
            
                    cutlass::arch::umma_arrive(
                        &shared_storage.ab_empty_mbar_ptr[smem_rd_buffer]
                    );
            
                    mma_rd_k_tile = mma_rd_k_tile_next;
                    smem_rd_buffer = smem_rd_buffer_next;
                    mma_rd_ab_full_phase = mma_rd_ab_full_phase_next;
        
                } // end for k_tile
                cutlass::arch::umma_arrive(&shared_storage.acc_full_mbar_ptr[acc_buf_idx]);
                num_tiles_executed++;
          } // end for n_tile
        }
      } else if (warp_idx < 4) {
        // Epilogue warps (4)
        if (lane_idx == 0) printf("[Epi warp %d] before tmem_allocation_result_barrier\n", warp_idx);
        tmem_allocation_result_barrier.arrive_and_wait();
        if (lane_idx == 0) printf("[Epi warp %d] after tmem_allocation_result_barrier\n", warp_idx);
    
        using AccType = typename decltype(tCtC)::value_type;
        using TypeBias = typename BiasTensor::value_type;
        // using TypeC = T_;
    
        cutlass::NumericConverter<AccType, TypeBias> converterBias;
        cutlass::NumericConverter<C_type, AccType> converter;
    
        // Extract a (MMA_M, MMA_N) TMEM view of the accumulator (first acc stage).
        // make_coord(_, _) unpacks the MmaC atom into logical (M, N) dimensions,
        // matching the pattern in CUTLASS sm100_epilogue: accumulators(make_coord(_,_), _0{}, _0{}).
        if (lane_idx == 0) printf("[Epi warp %d] making tAcc\n", warp_idx);
        cute::Tensor tAcc = tCtC(cute::make_coord(cute::_, cute::_), cute::_0{}, cute::_0{}, cute::_0{});  // (MMA_M, MMA_N)

        if (lane_idx == 0) printf("[Epi warp %d] make_tmem_copy\n", warp_idx);
        cute::TiledCopy tiled_copy_t2r = cute::make_tmem_copy(cute::SM100_TMEM_LOAD_32dp32b1x{}, tAcc);
        if (lane_idx == 0) printf("[Epi warp %d] get_slice\n", warp_idx);
        cute::ThrCopy thr_copy_t2r = tiled_copy_t2r.get_slice(threadIdx.x);

        if (lane_idx == 0) printf("[Epi warp %d] partition_S\n", warp_idx);
        // tTR_tAcc: per-thread TMEM source view across all acc stages; last mode = stage index
        cute::Tensor tTR_tAcc_all = thr_copy_t2r.partition_S(
            tCtC(cute::make_coord(cute::_, cute::_), cute::_0{}, cute::_0{}, cute::_)
        );  // (T2R, T2R_M, T2R_N, NUM_ACC_STAGE)

        if (lane_idx == 0) printf("[Epi warp %d] make tTR_rAcc\n", warp_idx);
        // tTR_rAcc: per-thread register buffer, shape matches T2R partition of (MMA_M, MMA_N)
        cute::Tensor tCrC_mn  = cute::make_tensor<AccType>(cute::make_shape(cute::Int<MMA_M>{}, cute::Int<MMA_N>{}));
        cute::Tensor tTR_rAcc = cute::make_tensor<AccType>(cute::shape(thr_copy_t2r.partition_D(tCrC_mn)));
        if (lane_idx == 0) printf("[Epi warp %d] entering m/n tile loop\n", warp_idx);

        int num_tiles_executed = 0;
        for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
          for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {
            int acc_buf_idx = num_tiles_executed % NUM_ACC_STAGE;
            int acc_full_phase = num_tiles_executed / NUM_ACC_STAGE % 2;
            int c_smem_wr_buffer_idx = num_tiles_executed % NUM_C_STAGE;

            cute::Tensor tCgBias = gBias(cute::_, cute::_, n_tile, m_tile); // (Mma_M, Mma_N)
            cute::Tensor tCrBiasTypeBias = cute::make_tensor<TypeBias>(cute::shape(tTR_rAcc)); // (T2R, T2R_M, T2R_N)
            cute::Tensor tCrBiasTypeAcc = cute::make_tensor<AccType>(cute::shape(tCrBiasTypeBias));
            cute::Tensor tCrC = cute::make_tensor<C_type>(cute::shape(tCrBiasTypeBias));

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

            sC.set_ptr(sC_ptr + c_smem_wr_buffer_idx * MMA_N * MMA_M);

            if (lane_idx == 0) printf("[Epi warp %d m=%d n=%d] before acc_full wait\n", warp_idx, m_tile, n_tile);
            cute::wait_barrier(shared_storage.acc_full_mbar_ptr[acc_buf_idx],
                               acc_full_phase);
            if (lane_idx == 0) printf("[Epi warp %d m=%d n=%d] before T2R copy\n", warp_idx, m_tile, n_tile);
            // T2R copy: TMEM → registers for this acc stage
            cute::copy(tiled_copy_t2r,
                       tTR_tAcc_all(cute::_, cute::_, cute::_, acc_buf_idx),
                       tTR_rAcc);
            // DEBUG: Print first few accumulator values after T2R to verify gemm produced non-zero output
            if (warp_idx == 0 && lane_idx == 0) {
                printf("TMEM->REG[m=%d,n=%d] rAcc[0]=%f rAcc[1]=%f rAcc[2]=%f rAcc[3]=%f\n",
                    m_tile, n_tile,
                    (float)tTR_rAcc(0), (float)tTR_rAcc(1),
                    (float)tTR_rAcc(2), (float)tTR_rAcc(3));
            }

            // arrive acc empty buffer
            epilogue_wg_barrier.arrive_and_wait();
            if (cute::elect_one_sync()) {
              cute::arrive_barrier(shared_storage.acc_empty_mbar_ptr[acc_buf_idx]);
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

            // R2S copy
            cute::Tensor tCsC_slice =
                cute::flatten(tCsC(cute::_, 0, 0, c_smem_wr_buffer_idx));
            cute::copy(tCrC, tCsC_slice(cute::_, threadIdx.x));

            // S2G TMA
            cute::tma_store_fence(); // Ensure C smem stores are visible to TMA
            epilogue_wg_barrier.arrive_and_wait(); // Ensure all threads have issued fence

            if (warp_idx == 0 && cute::elect_one_sync()) {
              if constexpr (SplitK) {
                tma_out.tma_reduce_add_async(
                    sC.base_ptr,
                    {m_tile * MMA_M, n_tile * MMA_N});
              } else {
                tma_out.tma_store_async(
                    sC.base_ptr,
                    {m_tile * MMA_M, n_tile * MMA_N});
              }
              cute::tma_store_arrive();
              cute::tma_store_wait<NUM_C_STAGE - 1>();
            }

            num_tiles_executed++;
          }
        }
        // wait all TMA stores to complete
        if (warp_idx == 0 && cute::elect_one_sync()) {
          cute::tma_store_wait<0>();
        }
      }
      __syncthreads();

    // Release the right to allocate before deallocations so that the next CTA can
    // rasterize Then deallocate TMEM
    if (warp_idx == 0) {
      // don't do relinquish for megakernel
      // tmem_allocator.release_allocation_lock(); 
      tmem_allocator.free(shared_storage.tmem_acc_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }
} // end linear_fp8_1d2d_sm100_mpk_task_impl

};  // namespace kernel

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
