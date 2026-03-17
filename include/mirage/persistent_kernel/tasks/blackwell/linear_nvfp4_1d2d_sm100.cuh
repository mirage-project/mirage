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
                                  const TMA_OUT &tma_out) {

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

    if (threadIdx.x == 0) {
        cute::print("[KERNEL] Start\n");
    }

    // MMA_N >= 32 for SFB SMEM -> TMEM transfer correctness
    static_assert(std::is_same_v<T_, cutlass::float_e2m1_t>, "T_ must be cutlass::float_e2m1_t");
    static_assert(SCALE_VECTOR_SIZE == 16, "SCALE_VECTOR_SIZE must be 16");
    static_assert(MMA_M == 128, "MMA_M must be 128");
    static_assert(MMA_N % 8 == 0 && MMA_N != 0 && MMA_N <=256, "MMA_N must be {8, 16, … 256} steps of 8"); 
    constexpr int MMA_K = 64; // SM100_MMA_MXF4_SS forces MMA_K to be 64
    
    if (threadIdx.x == 0) {
        cute::print("[KERNEL] static assert\n");
    }

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
            cute::make_shape(OUTPUT_SIZE, REDUCTION_SIZE),
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

    // Using Layout_K_SW32_Atom as MMA_K = 64 FP4 = 32 bytes
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

    extern __shared__ char shared_memory[];
    uintptr_t aligned_smem = (reinterpret_cast<uintptr_t>(shared_memory) + 127) / 128 * 128;
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(aligned_smem);

    cute::Tensor tCsA   = shared_storage.tensor_sA();
    cute::Tensor tCsB   = shared_storage.tensor_sB();
    cute::Tensor tCsSFA = shared_storage.tensor_sSFA();
    cute::Tensor tCsSFB = shared_storage.tensor_sSFB();
    cute::Tensor tCsC   = shared_storage.tensor_sC();

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
    __syncthreads();

    if (threadIdx.x == 0) {
        cute::print("[KERNEL] Here\n");
    }

    // Sync tmem allocation status between MMA and epilogue warps within CTA
    // 32 threads (mma) + 128 threads (epilog) to sync
    if (threadIdx.x == 0) {
        cute::print("[KERNEL] DEBUG 0.1\n");
    }
    cutlass::arch::NamedBarrier tmem_allocation_result_barrier(32 + 128, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
    if (threadIdx.x == 0) {
        cute::print("[KERNEL] DEBUG 0.2\n");
    }
    cutlass::arch::NamedBarrier epilogue_wg_barrier(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

    if (threadIdx.x == 0) {
        cute::print("[KERNEL] DEBUG 1\n");
    }

    constexpr int MMA_K_SF = MMA_K / SCALE_VECTOR_SIZE;  // = 4
    constexpr int tma_transaction_bytes = MMA_M * MMA_K / 2      // A (FP4 packed)
                                        + MMA_N * MMA_K / 2      // B (FP4 packed)
                                        + MMA_M * MMA_K_SF       // SFA (UE4M3 = 1 byte each)
                                        + MMA_N * MMA_K_SF;      // SFB

    if (threadIdx.x == 0) {
        cute::print("[KERNEL] DEBUG 2\n");
    }

    // A_type *sA_ptr = shared_storage.A.begin();
    // B_type *sB_ptr = shared_storage.B.begin();
    void    *sA_ptr      = static_cast<void*>(&shared_storage.A);  // Has to be done since type is fp4 (TODO: fix?)
    void    *sB_ptr      = static_cast<void*>(&shared_storage.B);
    C_type  *sC_ptr      = shared_storage.C.begin();
    SF_type *sSFA_ptr    = shared_storage.SFA.begin();
    SF_type *sSFB_ptr    = shared_storage.SFB.begin();

    if (threadIdx.x == 0) {
        cute::print("[KERNEL] DEBUG 3\n");
    }

    Barrier *ab_full_mbar_ptr = reinterpret_cast<Barrier *>(shared_storage.ab_full_mbar_ptr);

    if (threadIdx.x == 0) {
        cute::print("[KERNEL] smme ptrs inits\n");
    }

    // TODO: Create subbyte iterator?
    using A_smem_TMA = smem_tma<A_type, B, M, S, MMA_M, MMA_K, 1>;
    using B_smem_TMA = smem_tma<B_type, B, M, S, MMA_N, MMA_K, 1>;
    using C_smem_TMA = smem_tma<C_type, 0, M, S, MMA_N, MMA_M, 1>;
    using SFA_smem_TMA = smem_tma<SF_type, B, M, S, MMA_M, MMA_K_SF, 1>;
    using SFB_smem_TMA = smem_tma<SF_type, B, M, S, MMA_N, MMA_K_SF, 1>;

    A_smem_TMA sA(sA_ptr);
    B_smem_TMA sB(sB_ptr);
    C_smem_TMA sC(sC_ptr);
    SFA_smem_TMA sSFA(sSFA_ptr);
    SFB_smem_TMA sSFB(sSFB_ptr);

    if (threadIdx.x == 0) {
        cute::print("[KERNEL] tma inits\n");
    }

    cutlass::arch::fence_barrier_init();
    __syncthreads();
    if (threadIdx.x == 0) {
        printf("[CKPT] 1: barriers init done\n");
        printf("[CKPT] 1: smem base=%p A=%p B=%p SFA=%p SFB=%p C=%p\n",
            (void*)aligned_smem, sA_ptr, sB_ptr, (void*)sSFA_ptr, (void*)sSFB_ptr, (void*)sC_ptr);
        printf("[CKPT] 1: ab_full_mbar=%p ab_empty_mbar=%p acc_full_mbar=%p acc_empty_mbar=%p\n",
            (void*)shared_storage.ab_full_mbar_ptr,
            (void*)shared_storage.ab_empty_mbar_ptr,
            (void*)shared_storage.acc_full_mbar_ptr,
            (void*)shared_storage.acc_empty_mbar_ptr);
        printf("[CKPT] 1: tma_transaction_bytes=%d (A=%d B=%d SFA=%d SFB=%d)\n",
            tma_transaction_bytes,
            MMA_M * MMA_K / 2, MMA_N * MMA_K / 2,
            MMA_M * MMA_K_SF, MMA_N * MMA_K_SF);
    }

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
    if (threadIdx.x == 0) {
        printf("[CKPT] 2: tmem alloc done, tmem_acc_ptr=%u (capacity=%d)\n",
            shared_storage.tmem_acc_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }

    tCtC.data() = cute::make_tmem_ptr<C_type>(shared_storage.tmem_acc_ptr);

    if (threadIdx.x == 0) { printf("[CKPT] 2b: tCtC.data set, calling find_tmem_tensor_col_offset\n"); }
    uint32_t sfa_offset = cutlass::detail::find_tmem_tensor_col_offset(tCtC);
    if (threadIdx.x == 0) { printf("[CKPT] 2c: sfa_offset=%u\n", sfa_offset); }
    uint32_t sfb_offset = sfa_offset + MMA_K_SF * 4;
    shared_storage.tmem_sfa_ptr = shared_storage.tmem_acc_ptr + sfa_offset;
    shared_storage.tmem_sfb_ptr = shared_storage.tmem_acc_ptr + sfb_offset;
    if (threadIdx.x == 0) {
        printf("[CKPT] 3: tmem offsets: acc=%u sfa_off=%u sfb_off=%u sfa_ptr=%u sfb_ptr=%u capacity=%u\n",
            shared_storage.tmem_acc_ptr, sfa_offset, sfb_offset,
            shared_storage.tmem_sfa_ptr, shared_storage.tmem_sfb_ptr,
            TmemAllocator::Sm100TmemCapacityColumns);
        printf("[CKPT] 3: sfa_offset+capacity_check: sfa_ptr+MMA_K_SF=%u sfb_ptr+MMA_K_SF=%u max_col=%u\n",
            shared_storage.tmem_sfa_ptr + MMA_K_SF,
            shared_storage.tmem_sfb_ptr + MMA_K_SF,
            shared_storage.tmem_acc_ptr + TmemAllocator::Sm100TmemCapacityColumns);
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

    // Precompute TMEM column start for SFA/SFB
    uint32_t tmem_col_sfa = shared_storage.tmem_sfa_ptr;
    uint32_t tmem_col_sfb = shared_storage.tmem_sfb_ptr;
    int k_tile_count = cute::size<4>(tCgA);
    if (threadIdx.x == 0) {
        printf("[CKPT] 4: tmem frg setup done, k_tile_count=%d num_m_tiles=%d num_n_tiles=%d\n",
            k_tile_count, (int)cute::size<3>(tCgA), (int)cute::size<3>(tCgB));
        printf("[CKPT] 4: tCgA shape=(%d,%d,%d,%d,%d) tCgB shape=(%d,%d,%d,%d,%d)\n",
            (int)cute::size<0>(tCgA), (int)cute::size<1>(tCgA), (int)cute::size<2>(tCgA),
            (int)cute::size<3>(tCgA), (int)cute::size<4>(tCgA),
            (int)cute::size<0>(tCgB), (int)cute::size<1>(tCgB), (int)cute::size<2>(tCgB),
            (int)cute::size<3>(tCgB), (int)cute::size<4>(tCgB));
        printf("[CKPT] 4: warp_idx=%d lane_idx=%d\n", warp_idx, lane_idx);
    }

    __syncthreads();
    if (threadIdx.x == 0) { printf("[CKPT] 5: final syncthreads done, entering warp dispatch warp_idx=%d\n", warp_idx); }

    if (warp_idx == 5) {
        // TMA warp (1)
        if (cute::elect_one_sync()) printf("[TMA] entered, k_tile_count=%d num_m=%d num_n=%d\n",
            k_tile_count, (int)cute::size<3>(tCgA), (int)cute::size<3>(tCgB));
        int total_k_tile_count = 0;
        for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
            for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {

                int num_prev_k_blk = total_k_tile_count;
                total_k_tile_count += k_tile_count;

                int tma_wr_k_tile = 0;
                int smem_wr_buffer = (num_prev_k_blk + tma_wr_k_tile) % NUM_AB_STAGE;
                int tma_wr_ab_empty_phase = (num_prev_k_blk + tma_wr_k_tile) / NUM_AB_STAGE % 2 ^ 1;
                if (cute::elect_one_sync()) printf("[TMA] m=%d n=%d: peek ab_empty slot=%d phase=%d\n",
                    m_tile, n_tile, smem_wr_buffer, tma_wr_ab_empty_phase);
                bool peek_ab_empty_status = kernel::try_wait_barrier(
                    shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                    tma_wr_ab_empty_phase
                );
                if (cute::elect_one_sync()) printf("[TMA] m=%d n=%d: peek result=%d\n",
                    m_tile, n_tile, (int)peek_ab_empty_status);

                // CUTE_UNROLL
                for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
                    int tma_wr_k_tile_next  = tma_wr_k_tile + 1;
                    int smem_wr_buffer_next = (num_prev_k_blk + tma_wr_k_tile_next) % NUM_AB_STAGE;
                    int tma_wr_ab_empty_phase_next = smem_wr_buffer_next == 0
                                                        ? tma_wr_ab_empty_phase ^ 1
                                                        : tma_wr_ab_empty_phase;

                    if (cute::elect_one_sync()) printf("[TMA] k=%d: waiting ab_empty slot=%d phase=%d peek=%d\n",
                        k_tile, smem_wr_buffer, tma_wr_ab_empty_phase, (int)peek_ab_empty_status);
                    if (!peek_ab_empty_status) {
                        cute::wait_barrier(shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                                        tma_wr_ab_empty_phase);
                    }
                    if (cute::elect_one_sync()) printf("[TMA] k=%d: ab_empty done, setting smem ptrs\n", k_tile);

                    if (cute::elect_one_sync()) {
                        int tma_coords_A[2] = {k_tile * MMA_K, m_tile};
                        int tma_coords_B[2] = {k_tile * MMA_K, n_tile};
                        int tma_coords_SFA[3] = {0, k_tile, m_tile};
                        int tma_coords_SFB[3] = {0, k_tile, n_tile};

                        sA.set_ptr(static_cast<void*>(cute::raw_pointer_cast(tCsA(cute::_, cute::_, cute::_, smem_wr_buffer).data())));
                        sB.set_ptr(static_cast<void*>(cute::raw_pointer_cast(tCsB(cute::_, cute::_, cute::_, smem_wr_buffer).data())));
                        sSFA.set_ptr(static_cast<void*>(tCsSFA(cute::_, cute::_, cute::_, smem_wr_buffer).data().get()));
                        sSFB.set_ptr(static_cast<void*>(tCsSFB(cute::_, cute::_, cute::_, smem_wr_buffer).data().get()));

                        printf("[TMA] k=%d: sA=%p sB=%p sSFA=%p sSFB=%p\n",
                            k_tile, sA.base_ptr, sB.base_ptr, sSFA.base_ptr, sSFB.base_ptr);
                        printf("[TMA] k=%d: coordsA={%d,%d} coordsB={%d,%d} coordsSFA={%d,%d,%d} coordsSFB={%d,%d,%d}\n",
                            k_tile,
                            tma_coords_A[0], tma_coords_A[1],
                            tma_coords_B[0], tma_coords_B[1],
                            tma_coords_SFA[0], tma_coords_SFA[1], tma_coords_SFA[2],
                            tma_coords_SFB[0], tma_coords_SFB[1], tma_coords_SFB[2]);
                        printf("[TMA] k=%d: set_barrier_tx_bytes slot=%d bytes=%d desc_a=%p desc_b=%p desc_sfa=%p desc_sfb=%p\n",
                            k_tile, smem_wr_buffer, tma_transaction_bytes,
                            tma_a.desc_ptr, tma_b.desc_ptr, tma_sfa.desc_ptr, tma_sfb.desc_ptr);

                        cute::set_barrier_transaction_bytes(
                            shared_storage.ab_full_mbar_ptr[smem_wr_buffer],
                            tma_transaction_bytes
                        );

                        printf("[TMA] k=%d: issuing tma_a\n", k_tile);
                        tma_a.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer], reinterpret_cast<cutlass::half_t*>(sA.base_ptr), tma_coords_A);
                        printf("[TMA] k=%d: tma_a done, issuing tma_b\n", k_tile);
                        tma_b.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer], reinterpret_cast<cutlass::half_t*>(sB.base_ptr), tma_coords_B);
                        printf("[TMA] k=%d: tma_b done, issuing tma_sfa\n", k_tile);
                        tma_sfa.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer], reinterpret_cast<cutlass::half_t*>(sSFA.base_ptr), tma_coords_SFA);
                        printf("[TMA] k=%d: tma_sfa done, issuing tma_sfb\n", k_tile);
                        tma_sfb.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer], reinterpret_cast<cutlass::half_t*>(sSFB.base_ptr), tma_coords_SFB);
                        printf("[TMA] k=%d: tma_sfb done, arriving on ab_full\n", k_tile);

                        cute::arrive_barrier(shared_storage.ab_full_mbar_ptr[smem_wr_buffer]);
                        printf("[TMA] k=%d: all TMA done, ab_full arrived\n", k_tile);
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
        if (cute::elect_one_sync()) printf("[TMA] all tiles done\n");
      } else if (warp_idx == 4) {
        // MMA warp (1)
        if (cute::elect_one_sync()) printf("[CKPT] warp4 MMA: before tmem_allocation_result_barrier\n");
        tmem_allocation_result_barrier.arrive_and_wait();
        if (cute::elect_one_sync()) printf("[CKPT] warp4 MMA: after tmem_allocation_result_barrier\n");
        
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
                    if (cute::elect_one_sync()) printf("[CKPT] warp4 MMA: k=%d waiting ab_full slot=%d\n", k_tile, smem_rd_buffer);
                    cute::wait_barrier(
                        shared_storage.ab_full_mbar_ptr[smem_rd_buffer],
                        mma_rd_ab_full_phase
                    );
                    if (cute::elect_one_sync()) printf("[CKPT] warp4 MMA: k=%d ab_full done\n", k_tile);

                    // UTCCP: copy SFA/SFB from SMEM -> TMEM
                    if (cute::elect_one_sync()) {
                        SF_type* sfa = tCsSFA(cute::_, cute::_, cute::_, smem_rd_buffer).data().get();
                        SF_type* sfb = tCsSFB(cute::_, cute::_, cute::_, smem_rd_buffer).data().get();

                        uint32_t sfa_addr = cute::cast_smem_ptr_to_uint(sfa);
                        uint64_t sfa_desc = matrix_descriptor_encode((uint64_t)(sfa_addr)) |
                                            (1llu << 46) |
                                            matrix_descriptor_encode((uint64_t)16)  << 16 |
                                            matrix_descriptor_encode((uint64_t)512) << 32 |
                                            (uint64_t) 0 << 62;
                        printf("[CKPT] warp4 MMA: k=%d issuing UTCCP sfa sfa_addr=%u tmem_col=%u sfa_desc=0x%llx\n", k_tile, sfa_addr, tmem_col_sfa, (unsigned long long)sfa_desc);
                        asm volatile("{tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;}" :: "r"(tmem_col_sfa), "l"(sfa_desc));
                        printf("[CKPT] warp4 MMA: k=%d UTCCP sfa done\n", k_tile);

                        uint32_t sfb_addr = cute::cast_smem_ptr_to_uint(sfb);
                        uint64_t sfb_desc = matrix_descriptor_encode((uint64_t)(sfb_addr)) |
                                            (1llu << 46) |
                                            matrix_descriptor_encode((uint64_t)16)  << 16 |
                                            matrix_descriptor_encode((uint64_t)512) << 32 |
                                            (uint64_t) 0 << 62;
                        printf("[CKPT] warp4 MMA: k=%d issuing UTCCP sfb sfb_addr=%u tmem_col=%u sfb_desc=0x%llx\n", k_tile, sfb_addr, tmem_col_sfb, (unsigned long long)sfb_desc);
                        asm volatile("{tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;}" :: "r"(tmem_col_sfb), "l"(sfb_desc));
                        printf("[CKPT] warp4 MMA: k=%d UTCCP sfb done\n", k_tile);
                    }

                    if (cute::elect_one_sync()) printf("[CKPT] warp4 MMA: k=%d issuing gemm\n", k_tile);
                    auto accumulate = tiled_mma.accumulate_;
                    for (int k_block = 0; k_block < cute::size<2>(tCfA); ++k_block) {
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
        if (lane_idx == 0) printf("[CKPT] warp%d epilogue: before tmem_allocation_result_barrier\n", warp_idx);
        tmem_allocation_result_barrier.arrive_and_wait();
        if (lane_idx == 0) printf("[CKPT] warp%d epilogue: after tmem_allocation_result_barrier\n", warp_idx);
    
        using AccType = typename decltype(tCtC)::value_type;
        using TypeBias = typename BiasTensor::value_type;
    
        cutlass::NumericConverter<AccType, TypeBias> converterBias;
        // cutlass::NumericConverter<C_type, AccType> converter;
    
        cute::Tensor tAcc = tCtC(cute::make_coord(cute::_, cute::_), cute::_0{}, cute::_0{}, cute::_0{});  // (MMA_M, MMA_N)
        cute::TiledCopy tiled_copy_t2r = cute::make_tmem_copy(cute::SM100_TMEM_LOAD_32dp32b1x{}, tAcc);
        cute::ThrCopy thr_copy_t2r = tiled_copy_t2r.get_slice(threadIdx.x);

        cute::Tensor tTR_tAcc_all = thr_copy_t2r.partition_S(
            tCtC(cute::make_coord(cute::_, cute::_), cute::_0{}, cute::_0{}, cute::_)
        );  // (T2R, T2R_M, T2R_N, NUM_ACC_STAGE)

        // tTR_rAcc: per-thread register buffer, shape matches T2R partition of (MMA_M, MMA_N)
        cute::Tensor tCrC_mn  = cute::make_tensor<AccType>(cute::make_shape(cute::Int<MMA_M>{}, cute::Int<MMA_N>{}));
        cute::Tensor tTR_rAcc = cute::make_tensor<AccType>(cute::shape(thr_copy_t2r.partition_D(tCrC_mn)));

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

            cute::wait_barrier(shared_storage.acc_full_mbar_ptr[acc_buf_idx],
                               acc_full_phase);
            // T2R copy: TMEM → registers for this acc stage
            cute::copy(tiled_copy_t2r,
                       tTR_tAcc_all(cute::_, cute::_, cute::_, acc_buf_idx),
                       tTR_rAcc);
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
              tCrC[i] = tTR_rAcc[i];
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
