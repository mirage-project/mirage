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

// C = (A * SFA) x (B * SFB) + bias
template <typename T_,
          typename TMA_A,
          typename TMA_B,
          typename TMA_SFA,
          typename TMA_SFB,
          class BiasTensor,
          class OutputTensor,
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
                                  OutputTensor mOutput,
                                  const TMA_OUT &tma_out) {

    static_assert(std::is_same_v<T_, cutlass::float_e2m1_t>, "T_ must be cutlass::float_e2m1_t");
    static_assert(SCALE_VECTOR_SIZE == 16, "SCALE_VECTOR_SIZE must be 16");
    static_assert(MMA_M == 128, "MMA_M must be 128");
    static_assert(MMA_N % 8 == 0 && MMA_N != 0 && MMA_N <= 256, "MMA_N must be {8, 16, ... 256} in steps of 8");

    constexpr int MMA_K = 64;
    constexpr int EPI_PIPE_DEPTH = 4;
    constexpr int EPI_N = MMA_N / EPI_PIPE_DEPTH;
    using A_type  = cutlass::float_e2m1_t;
    using B_type  = cutlass::float_e2m1_t;
    using SF_type = cutlass::float_ue4m3_t;
    using C_type  = float;

    constexpr int B_FP4 = 1;
    constexpr int B_SF  = 0;
    constexpr int M = 3;
    constexpr int S = 3;
    constexpr int OUTPUT_SWIZZLE = 128 / sizeof(C_type);

    constexpr int NUM_MMA_M = 1;
    constexpr int NUM_MMA_N = 1;
    constexpr int NUM_MMA_K = 4;

    static_assert(MMA_N % EPI_PIPE_DEPTH == 0, "EPI_PIPE_DEPTH must evenly divide MMA_N");
    static_assert(NUM_C_STAGE >= EPI_PIPE_DEPTH, "NUM_C_STAGE must cover all epilogue subtiles");
    static_assert(EPI_N % OUTPUT_SWIZZLE == 0, "EPI subtile must align with the output swizzle width");

    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = kernel::lane_id();

    cute::TiledMMA tiled_mma = cute::make_tiled_mma(
        cute::SM100_MMA_MXF4_SS<A_type, B_type, C_type, SF_type,
                                 MMA_M, MMA_N, SCALE_VECTOR_SIZE,
                                 cute::UMMA::Major::K, cute::UMMA::Major::K>{}
    );

    constexpr auto bM = MMA_M * NUM_MMA_M;
    constexpr auto bN = MMA_N * NUM_MMA_N;
    constexpr auto bK = MMA_K * NUM_MMA_K;

    constexpr int NUM_M_TILE_PER_CTA = 1;
    constexpr int NUM_N_TILE_PER_CTA = 1;

    const int cta_m_tile = blockIdx.x;
    const int cta_n_tile = blockIdx.y;

    const int tile_m_begin = NUM_M_TILE_PER_CTA * cta_m_tile;
    const int tile_m_end   = NUM_M_TILE_PER_CTA * (cta_m_tile + 1);
    const int tile_n_begin = NUM_N_TILE_PER_CTA * cta_n_tile;
    const int tile_n_end   = NUM_N_TILE_PER_CTA * (cta_n_tile + 1);

    using TiledMma             = decltype(tiled_mma);
    using ThrBlkTileShape_MNK  = cute::Shape<cute::Int<bM>, cute::Int<bN>, cute::Int<bK>>;
    using GmemStrideTypeD      = cute::Stride<cute::Int<OUTPUT_SIZE>, cute::Int<1>>;
    using AccLoadOp            = cute::SM100_TMEM_LOAD_32dp32b2x;
    using EpilogueTileMN       = cute::Shape<cute::Int<MMA_M>, cute::Int<EPI_N>>;
    using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SCALE_VECTOR_SIZE>;
    using SmemLayoutAtomSFA    = decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFA(TiledMma{}, ThrBlkTileShape_MNK{}));
    using SmemLayoutAtomSFB    = decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFB(TiledMma{}, ThrBlkTileShape_MNK{}));
    using SmemLayoutAtomD      = decltype(cutlass::epilogue::collective::detail::sm100_get_epilogue_smem_swizzle_layout_atom<
        GmemStrideTypeD, C_type, EpilogueTileMN>());
    static constexpr bool is_m_major_D = cutlass::detail::is_major<0>(GmemStrideTypeD{});
    using SmemLayoutStageD = decltype(cute::tile_to_shape(
        SmemLayoutAtomD{},
        cute::product_each(cute::shape(EpilogueTileMN{})),
        cute::conditional_t<is_m_major_D, cute::Step<cute::_2, cute::_1>, cute::Step<cute::_1, cute::_2>>{}));
    constexpr static int StageDBits = cute::cosize_v<SmemLayoutStageD> * cute::sizeof_bits_v<C_type>;
    constexpr static int StrideStageD = StageDBits / cute::sizeof_bits_v<C_type>;
    using SmemLayoutD = decltype(cute::append<3>(
        SmemLayoutStageD{}, cute::Layout<cute::Int<NUM_C_STAGE>, cute::Int<StrideStageD>>{}));
    using CopyOpR2S = decltype(cutlass::epilogue::collective::detail::sm100_get_smem_store_op<
        GmemStrideTypeD, C_type, C_type, AccLoadOp>());

    auto tb_mma_coord_vmnk = cute::make_coord(0, cute::_, cute::_, cute::_);
    auto tb_mma_coord      = cute::select<1, 2, 3>(tb_mma_coord_vmnk);
    auto tb_mma_tiler      = ThrBlkTileShape_MNK{};
    auto tb_cd_tiler       = cute::make_shape(cute::Int<bM>{}, cute::Int<bN>{}, cute::Int<bK>{});
    auto tb_sf_tiler       = cute::make_shape(cute::Int<bM>{}, cute::Int<bN>{}, cute::Int<bK / SCALE_VECTOR_SIZE>{});

    cute::Tensor mA = cute::make_coord_tensor(
        cute::make_layout(cute::make_shape(BATCH_SIZE, REDUCTION_SIZE),
                          cute::make_stride(cute::E<1>{}, cute::E<0>{})));
    cute::Tensor mB = cute::make_coord_tensor(
        cute::make_layout(cute::make_shape(OUTPUT_SIZE, REDUCTION_SIZE),
                          cute::make_stride(cute::E<1>{}, cute::E<0>{})));
    cute::Tensor mC = cute::make_coord_tensor(
        cute::make_layout(cute::make_shape(BATCH_SIZE, OUTPUT_SIZE),
                          cute::make_stride(cute::E<1>{}, cute::E<0>{})));
    cute::Tensor mSFA = cute::make_coord_tensor(
        cute::make_layout(cute::make_shape(BATCH_SIZE, REDUCTION_SIZE / SCALE_VECTOR_SIZE),
                          cute::make_stride(cute::E<1>{}, cute::E<0>{})));
    cute::Tensor mSFB = cute::make_coord_tensor(
        cute::make_layout(cute::make_shape(OUTPUT_SIZE, REDUCTION_SIZE / SCALE_VECTOR_SIZE),
                          cute::make_stride(cute::E<1>{}, cute::E<0>{})));

    cute::Tensor gA = cute::local_tile(mA, tb_mma_tiler, tb_mma_coord,
                                       cute::Step<cute::_1, cute::X, cute::_1>{});
    cute::Tensor gB = cute::local_tile(mB, tb_mma_tiler, tb_mma_coord,
                                       cute::Step<cute::X, cute::_1, cute::_1>{});
    cute::Tensor gBias = cute::local_tile(mBias, tb_cd_tiler, tb_mma_coord,
                                          cute::Step<cute::_1, cute::_1, cute::X>{});
    cute::Tensor gSFA = cute::local_tile(mSFA, tb_sf_tiler, tb_mma_coord,
                                         cute::Step<cute::_1, cute::X, cute::_1>{});
    cute::Tensor gSFB = cute::local_tile(mSFB, tb_sf_tiler, tb_mma_coord,
                                         cute::Step<cute::X, cute::_1, cute::_1>{});

    auto cta_ind  = cute::get<0>(tb_mma_coord_vmnk);
    cute::ThrMMA cta_mma = tiled_mma.get_slice(cta_ind);
    cute::Tensor tCgA   = cta_mma.partition_A(gA);
    cute::Tensor tCgB   = cta_mma.partition_B(gB);
    cute::Tensor tCgSFA = cta_mma.partition_A(gSFA);
    cute::Tensor tCgSFB = cta_mma.partition_B(gSFB);

    auto mma_shape_A = cute::partition_shape_A(
        tiled_mma, cute::make_shape(cute::Int<bM>{}, cute::Int<bK>{}, cute::Int<NUM_AB_STAGE>{}));
    auto mma_shape_B = cute::partition_shape_B(
        tiled_mma, cute::make_shape(cute::Int<bN>{}, cute::Int<bK>{}, cute::Int<NUM_AB_STAGE>{}));
    // bK=256 FP4 = 128 bytes = one 128B swizzle stripe, matches B_FP4=3 TMA descriptor
    auto sA_layout = cute::UMMA::tile_to_mma_shape(
        cute::UMMA::Layout_K_SW128_Atom<A_type>{}, mma_shape_A);
    auto sB_layout = cute::UMMA::tile_to_mma_shape(
        cute::UMMA::Layout_K_SW128_Atom<B_type>{}, mma_shape_B);
    auto sSFA_layout = cute::tile_to_shape(
        SmemLayoutAtomSFA{},
        cute::append(cute::shape(SmemLayoutAtomSFA{}), cute::Int<NUM_AB_STAGE>{}));
    auto sSFB_layout = cute::tile_to_shape(
        SmemLayoutAtomSFB{},
        cute::append(cute::shape(SmemLayoutAtomSFB{}), cute::Int<NUM_AB_STAGE>{}));

    using SharedStorage = PipedScaledEpilogueSharedStorage<A_type, B_type, C_type, SF_type,
                                                           decltype(sA_layout), decltype(sB_layout),
                                                           SmemLayoutD, decltype(sSFA_layout),
                                                           decltype(sSFB_layout), NUM_AB_STAGE, NUM_ACC_STAGE>;

    extern __shared__ char shared_memory[];
    uintptr_t aligned_smem = (reinterpret_cast<uintptr_t>(shared_memory) + 127) / 128 * 128;
    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(aligned_smem);

    cute::Tensor tCsA   = shared_storage.tensor_sA();
    cute::Tensor tCsB   = shared_storage.tensor_sB();
    cute::Tensor tCsC   = cute::as_position_independent_swizzle_tensor(shared_storage.tensor_sC());
    cute::Tensor tCsSFA = shared_storage.tensor_sSFA();
    cute::Tensor tCsSFB = shared_storage.tensor_sSFB();

    if (warp_idx == 0) {
        cutlass::arch::detail::initialize_barrier_array_aligned<
            cutlass::arch::ClusterTransactionBarrier, NUM_AB_STAGE>(shared_storage.ab_full_mbar_ptr, 1);
        cutlass::arch::detail::initialize_barrier_array_aligned<
            cutlass::arch::ClusterBarrier, NUM_AB_STAGE>(shared_storage.ab_empty_mbar_ptr, 1);
        cutlass::arch::detail::initialize_barrier_array_aligned<
            cutlass::arch::ClusterTransactionBarrier, NUM_AB_STAGE>(shared_storage.sf_full_mbar_ptr, 1);
        cutlass::arch::detail::initialize_barrier_array_aligned<
            cutlass::arch::ClusterBarrier, NUM_AB_STAGE>(shared_storage.sf_empty_mbar_ptr, 1);
        cutlass::arch::detail::initialize_barrier_array_aligned<
            cutlass::arch::ClusterBarrier, NUM_ACC_STAGE>(shared_storage.acc_full_mbar_ptr, 1);
        cutlass::arch::detail::initialize_barrier_array_aligned<
            cutlass::arch::ClusterBarrier, NUM_ACC_STAGE>(shared_storage.acc_empty_mbar_ptr, 4);
    }
    __syncthreads();

    cutlass::arch::NamedBarrier tmem_allocation_result_barrier(32 + 128, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
    cutlass::arch::NamedBarrier epilogue_wg_barrier(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    Barrier *ab_full_mbar_ptr = reinterpret_cast<Barrier *>(shared_storage.ab_full_mbar_ptr);
    Barrier *sf_full_mbar_ptr = reinterpret_cast<Barrier *>(shared_storage.sf_full_mbar_ptr);

    constexpr int MMA_K_SF = MMA_K / SCALE_VECTOR_SIZE;
    constexpr int tma_transaction_bytes_AB = MMA_M * MMA_K / 2 * NUM_MMA_K
                                           + MMA_N * MMA_K / 2 * NUM_MMA_K;
    constexpr int tma_transaction_bytes_SF = MMA_M * MMA_K_SF * NUM_MMA_K
                                           + MMA_N * MMA_K_SF * NUM_MMA_K;
    void    *sA_ptr   = static_cast<void*>(&shared_storage.A);
    void    *sB_ptr   = static_cast<void*>(&shared_storage.B);
    SF_type *sSFA_ptr = shared_storage.SFA.begin();
    SF_type *sSFB_ptr = shared_storage.SFB.begin();

    using A_smem_TMA   = smem_tma<A_type,  B_FP4, M, S, MMA_M, MMA_K, 1>;
    using B_smem_TMA   = smem_tma<B_type,  B_FP4, M, S, MMA_N, MMA_K, 1>;
    using SFA_smem_TMA = smem_tma<SF_type, B_SF,  M, S, MMA_M, MMA_K_SF, 1>;
    using SFB_smem_TMA = smem_tma<SF_type, B_SF,  M, S, MMA_N, MMA_K_SF, 1>;

    A_smem_TMA   sA(sA_ptr);
    B_smem_TMA   sB(sB_ptr);
    SFA_smem_TMA sSFA(sSFA_ptr);
    SFB_smem_TMA sSFB(sSFB_ptr);

    cutlass::arch::fence_barrier_init();
    __syncthreads();

    cute::Tensor tCfA = cta_mma.make_fragment_A(tCsA);
    cute::Tensor tCfB = cta_mma.make_fragment_B(tCsB);
    auto acc_shape = cute::partition_shape_C(
        tiled_mma,
        cute::make_shape(cute::size<0>(tb_mma_tiler), cute::size<1>(tb_mma_tiler),
                         cute::Int<NUM_ACC_STAGE>{}));
    cute::Tensor tCtC = tiled_mma.make_fragment_C(acc_shape);

    using UtccpOp = cute::SM100_UTCCP_4x32dp128bit_1cta;
    using TmemAllocator = cute::TMEM::Allocator1Sm;
    TmemAllocator tmem_allocator{};
    if (warp_idx == 0) {
        tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_acc_ptr);
    }
    __syncthreads();
    tCtC.data() = cute::make_tmem_ptr<C_type>(shared_storage.tmem_acc_ptr);

    using FrgTypeSFA = cute::UMMA::tmem_sf_frg<SF_type, SCALE_VECTOR_SIZE, 1, true>;
    using FrgTypeSFB = cute::UMMA::tmem_sf_frg<SF_type, SCALE_VECTOR_SIZE, 1, false>;
    static constexpr int MMA_M_SFB = ((MMA_M + 127) / 128) * 128;
    static constexpr int MMA_N_SFB = ((MMA_N + 127) / 128) * 128;
    auto sfa_tmem_shape = cute::make_shape(
        cute::make_shape(cute::Int<MMA_M_SFB>{},
                         cute::make_shape(cute::Int<SCALE_VECTOR_SIZE>{}, cute::Int<MMA_K / SCALE_VECTOR_SIZE>{})),
        cute::Int<NUM_MMA_M>{}, cute::Int<NUM_MMA_K>{});
    auto sfb_tmem_shape = cute::make_shape(
        cute::make_shape(cute::Int<MMA_N_SFB>{},
                         cute::make_shape(cute::Int<SCALE_VECTOR_SIZE>{}, cute::Int<MMA_K / SCALE_VECTOR_SIZE>{})),
        cute::Int<NUM_MMA_N>{}, cute::Int<NUM_MMA_K>{});
    auto tCtSFA = FrgTypeSFA::make(sfa_tmem_shape);
    auto tCtSFB = FrgTypeSFB::make(sfb_tmem_shape);

    uint32_t sfa_offset = cutlass::detail::find_tmem_tensor_col_offset(tCtC);
    uint32_t sfb_offset = sfa_offset + cutlass::detail::find_tmem_tensor_col_offset(tCtSFA);
    shared_storage.tmem_sfa_ptr = shared_storage.tmem_acc_ptr + sfa_offset;
    shared_storage.tmem_sfb_ptr = shared_storage.tmem_acc_ptr + sfb_offset;
    tCtSFA.data() = make_tmem_ptr<SF_type>(shared_storage.tmem_sfa_ptr);
    tCtSFB.data() = make_tmem_ptr<SF_type>(shared_storage.tmem_sfb_ptr);

    uint32_t tmem_sfa_ptr = shared_storage.tmem_sfa_ptr;
    uint32_t tmem_sfb_ptr = shared_storage.tmem_sfb_ptr;
    int k_tile_count = cute::size<4>(tCgA);

    if (warp_idx == 6) {
        // TMA A/B: GMEM -> SMEM
        int total_k_tile_count = 0;
        for (int m_tile = tile_m_begin; m_tile < tile_m_end; ++m_tile) {
            for (int n_tile = tile_n_begin; n_tile < tile_n_end; ++n_tile) {
                int num_prev_k_blk = total_k_tile_count;
                total_k_tile_count += k_tile_count;

                int tma_wr_k_tile = 0;
                int smem_wr_buffer = (num_prev_k_blk + tma_wr_k_tile) % NUM_AB_STAGE;
                int tma_wr_ab_empty_phase = (num_prev_k_blk + tma_wr_k_tile) / NUM_AB_STAGE % 2 ^ 1;
                bool peek_ab_empty_status = kernel::try_wait_barrier(
                    shared_storage.ab_empty_mbar_ptr[smem_wr_buffer], tma_wr_ab_empty_phase);

                for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
                    int tma_wr_k_tile_next        = tma_wr_k_tile + 1;
                    int smem_wr_buffer_next       = (num_prev_k_blk + tma_wr_k_tile_next) % NUM_AB_STAGE;
                    int tma_wr_ab_empty_phase_next = smem_wr_buffer_next == 0
                                                        ? tma_wr_ab_empty_phase ^ 1
                                                        : tma_wr_ab_empty_phase;
                    int tma_coords_A[2] = {k_tile * bK, m_tile * MMA_M};
                    int tma_coords_B[2] = {k_tile * bK, n_tile * MMA_N};
                    void *sA_stage_ptr = static_cast<void*>(
                        cute::raw_pointer_cast(tCsA(cute::_, cute::_, cute::_, smem_wr_buffer).data()));
                    void *sB_stage_ptr = static_cast<void*>(
                        cute::raw_pointer_cast(tCsB(cute::_, cute::_, cute::_, smem_wr_buffer).data()));

                    if (!peek_ab_empty_status) {
                        cute::wait_barrier(shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                                           tma_wr_ab_empty_phase);
                    }

                    if (cute::elect_one_sync()) {
                        sA.set_ptr(sA_stage_ptr);
                        sB.set_ptr(sB_stage_ptr);
                        cute::set_barrier_transaction_bytes(
                            shared_storage.ab_full_mbar_ptr[smem_wr_buffer], tma_transaction_bytes_AB);
                        tma_a.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer], sA.base_ptr, tma_coords_A);
                        tma_b.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer], sB.base_ptr, tma_coords_B);
                    }

                    if (tma_wr_k_tile_next < k_tile_count) {
                        peek_ab_empty_status = kernel::try_wait_barrier(
                            shared_storage.ab_empty_mbar_ptr[smem_wr_buffer_next],
                            tma_wr_ab_empty_phase_next);
                    }

                    tma_wr_k_tile        = tma_wr_k_tile_next;
                    smem_wr_buffer       = smem_wr_buffer_next;
                    tma_wr_ab_empty_phase = tma_wr_ab_empty_phase_next;
                }
            }
        }
    } else if (warp_idx == 5) {
        // TMA SFA/SFB: GMEM -> SMEM
        int total_k_tile_count = 0;
        for (int m_tile = tile_m_begin; m_tile < tile_m_end; ++m_tile) {
            for (int n_tile = tile_n_begin; n_tile < tile_n_end; ++n_tile) {
                int num_prev_k_blk = total_k_tile_count;
                total_k_tile_count += k_tile_count;

                int tma_wr_k_tile = 0;
                int smem_wr_buffer = (num_prev_k_blk + tma_wr_k_tile) % NUM_AB_STAGE;
                int tma_wr_sf_empty_phase = (num_prev_k_blk + tma_wr_k_tile) / NUM_AB_STAGE % 2 ^ 1;
                bool peek_sf_empty_status = kernel::try_wait_barrier(
                    shared_storage.sf_empty_mbar_ptr[smem_wr_buffer], tma_wr_sf_empty_phase);

                for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
                    int tma_wr_k_tile_next        = tma_wr_k_tile + 1;
                    int smem_wr_buffer_next       = (num_prev_k_blk + tma_wr_k_tile_next) % NUM_AB_STAGE;
                    int tma_wr_sf_empty_phase_next = smem_wr_buffer_next == 0
                                                        ? tma_wr_sf_empty_phase ^ 1
                                                        : tma_wr_sf_empty_phase;
                    int tma_coords_SFA[3] = {0, k_tile * NUM_MMA_K, m_tile};
                    int tma_coords_SFB[3] = {0, k_tile * NUM_MMA_K, n_tile};
                    void *sSFA_stage_ptr = static_cast<void*>(
                        tCsSFA(cute::_, cute::_, cute::_, smem_wr_buffer).data().get());
                    void *sSFB_stage_ptr = static_cast<void*>(
                        tCsSFB(cute::_, cute::_, cute::_, smem_wr_buffer).data().get());

                    if (!peek_sf_empty_status) {
                        cute::wait_barrier(shared_storage.sf_empty_mbar_ptr[smem_wr_buffer],
                                           tma_wr_sf_empty_phase);
                    }

                    if (cute::elect_one_sync()) {
                        sSFA.set_ptr(sSFA_stage_ptr);
                        sSFB.set_ptr(sSFB_stage_ptr);
                        cute::set_barrier_transaction_bytes(
                            shared_storage.sf_full_mbar_ptr[smem_wr_buffer], tma_transaction_bytes_SF);
                        tma_sfa.tma_cp_async(sf_full_mbar_ptr[smem_wr_buffer], reinterpret_cast<cute::half_t*>(sSFA.base_ptr), tma_coords_SFA);
                        tma_sfb.tma_cp_async(sf_full_mbar_ptr[smem_wr_buffer], reinterpret_cast<cute::half_t*>(sSFB.base_ptr), tma_coords_SFB);
                    }

                    if (tma_wr_k_tile_next < k_tile_count) {
                        peek_sf_empty_status = kernel::try_wait_barrier(
                            shared_storage.sf_empty_mbar_ptr[smem_wr_buffer_next],
                            tma_wr_sf_empty_phase_next);
                    }

                    tma_wr_k_tile        = tma_wr_k_tile_next;
                    smem_wr_buffer       = smem_wr_buffer_next;
                    tma_wr_sf_empty_phase = tma_wr_sf_empty_phase_next;
                }
            }
        }
    } else if (warp_idx == 4) {
        // SF SMEM -> TMEM + MMA
        tmem_allocation_result_barrier.arrive_and_wait();
        auto copy_sf_s2t = [&](int stage) {
            using UtccpOp = SM100_UTCCP_4x32dp128bit_1cta;
            auto copy_one = [&](auto& tCsSF, auto& tCtSF) {
                auto tCsSF_stage   = tCsSF(cute::_, cute::_, cute::_, stage);
                auto tCsSF_compact = make_tensor(tCsSF_stage.data(), filter_zeros(tCsSF_stage.layout()));
                auto tCtSF_compact = make_tensor(tCtSF.data(), filter_zeros(tCtSF.layout()));
                auto tiled_s2t     = make_utccp_copy(UtccpOp{}, tCtSF_compact);
                auto thr_s2t       = tiled_s2t.get_slice(0);
                auto src_          = thr_s2t.partition_S(tCsSF_compact);
                auto src           = get_utccp_smem_desc_tensor<UtccpOp>(src_);
                auto dst           = thr_s2t.partition_D(tCtSF_compact);
                cute::copy(tiled_s2t, src, dst);
            };
            copy_one(tCsSFA, tCtSFA);
            copy_one(tCsSFB, tCtSFB);
        };

        int total_k_tile_count = 0;
        int num_tiles_executed = 0;
        for (int m_tile = tile_m_begin; m_tile < tile_m_end; ++m_tile) {
            for (int n_tile = tile_n_begin; n_tile < tile_n_end; ++n_tile) {
                int acc_buf_idx    = num_tiles_executed % NUM_ACC_STAGE;
                int num_prev_k_blk = total_k_tile_count;
                total_k_tile_count += k_tile_count;

                int mma_rd_k_tile        = 0;
                int smem_rd_buffer       = (num_prev_k_blk + mma_rd_k_tile) % NUM_AB_STAGE;
                int mma_rd_ab_full_phase = (num_prev_k_blk + mma_rd_k_tile) / NUM_AB_STAGE % 2;
                int mma_rd_sf_full_phase = (num_prev_k_blk + mma_rd_k_tile) / NUM_AB_STAGE % 2;
                int acc_empty_phase      = num_tiles_executed / NUM_ACC_STAGE % 2 ^ 1;
                bool peek_ab_full_status = kernel::try_wait_barrier(
                    shared_storage.ab_full_mbar_ptr[smem_rd_buffer], mma_rd_ab_full_phase);
                bool peek_sf_full_status = kernel::try_wait_barrier(
                    shared_storage.sf_full_mbar_ptr[smem_rd_buffer], mma_rd_sf_full_phase);

                cute::wait_barrier(shared_storage.acc_empty_mbar_ptr[acc_buf_idx], acc_empty_phase);

                tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;
                for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
                    int mma_rd_k_tile_next        = mma_rd_k_tile + 1;
                    int smem_rd_buffer_next       = (num_prev_k_blk + mma_rd_k_tile_next) % NUM_AB_STAGE;
                    int mma_rd_ab_full_phase_next = smem_rd_buffer_next == 0
                                                        ? mma_rd_ab_full_phase ^ 1
                                                        : mma_rd_ab_full_phase;
                    int mma_rd_sf_full_phase_next = smem_rd_buffer_next == 0
                                                        ? mma_rd_sf_full_phase ^ 1
                                                        : mma_rd_sf_full_phase;

                    if (!peek_ab_full_status) {
                        cute::wait_barrier(shared_storage.ab_full_mbar_ptr[smem_rd_buffer], mma_rd_ab_full_phase);
                    }
                    if (!peek_sf_full_status) {
                        cute::wait_barrier(shared_storage.sf_full_mbar_ptr[smem_rd_buffer], mma_rd_sf_full_phase);
                    }
                    if (cute::elect_one_sync()) {
                        copy_sf_s2t(smem_rd_buffer);
                    }

                    auto accumulate = tiled_mma.accumulate_;
                    for (int k_block = 0; k_block < cute::size<2>(tCfA); ++k_block) {
                        cute::gemm(
                            tiled_mma.with(accumulate,
                                           tCtSFA(cute::_, cute::_, k_block),
                                           tCtSFB(cute::_, cute::_, k_block)),
                            tCfA(cute::_, cute::_, k_block, smem_rd_buffer),
                            tCfB(cute::_, cute::_, k_block, smem_rd_buffer),
                            tCtC(cute::_, cute::_, cute::_, acc_buf_idx));
                        accumulate = cute::UMMA::ScaleOut::One;
                    }
                    tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;

                    cutlass::arch::umma_arrive(&shared_storage.ab_empty_mbar_ptr[smem_rd_buffer]);
                    cutlass::arch::umma_arrive(&shared_storage.sf_empty_mbar_ptr[smem_rd_buffer]);

                    if (mma_rd_k_tile_next < k_tile_count) {
                        peek_ab_full_status = kernel::try_wait_barrier(
                            shared_storage.ab_full_mbar_ptr[smem_rd_buffer_next],
                            mma_rd_ab_full_phase_next);
                        peek_sf_full_status = kernel::try_wait_barrier(
                            shared_storage.sf_full_mbar_ptr[smem_rd_buffer_next],
                            mma_rd_sf_full_phase_next);
                    }
                    mma_rd_k_tile        = mma_rd_k_tile_next;
                    smem_rd_buffer       = smem_rd_buffer_next;
                    mma_rd_ab_full_phase = mma_rd_ab_full_phase_next;
                    mma_rd_sf_full_phase = mma_rd_sf_full_phase_next;
                }

                cutlass::arch::umma_arrive(&shared_storage.acc_full_mbar_ptr[acc_buf_idx]);
                num_tiles_executed++;
            }
        }
    } else if (warp_idx < 4) {
        // Epilogue: drain the 128x128 accumulator tile as four 128x32 subtiles.
        tmem_allocation_result_barrier.arrive_and_wait();

        auto epi_tile_shape = cute::make_shape(cute::Int<MMA_M>{}, cute::Int<EPI_N>{});
        cute::Tensor tCtAcc_proto = tCtC(cute::make_coord(cute::_, cute::_), cute::_0{}, cute::_0{}, cute::_0{});
        cute::Tensor tCtAcc_epi_proto = cute::flat_divide(tCtAcc_proto, epi_tile_shape);
        cute::Tensor tCtAcc_subtile_proto = tCtAcc_epi_proto(cute::_, cute::_, cute::_0{}, cute::_0{});

        cute::TiledCopy tiled_copy_t2r = cute::make_tmem_copy(
            AccLoadOp{}, tCtAcc_subtile_proto);
        cute::ThrCopy thr_copy_t2r = tiled_copy_t2r.get_slice(threadIdx.x);
        cute::Tensor tCrSubtile_ref = cute::make_tensor<C_type>(epi_tile_shape);
        cute::Tensor tCrAcc0 = cute::make_tensor<C_type>(cute::shape(thr_copy_t2r.partition_D(tCrSubtile_ref)));
        cute::Tensor tCrAcc1 = cute::make_tensor<C_type>(cute::shape(thr_copy_t2r.partition_D(tCrSubtile_ref)));
        cute::Tensor tCrBias0 = cute::make_tensor<C_type>(cute::shape(tCrAcc0));
        cute::Tensor tCrBias1 = cute::make_tensor<C_type>(cute::shape(tCrAcc1));
        cute::TiledCopy tiled_r2s = cute::make_tiled_copy_D(
            cute::Copy_Atom<CopyOpR2S, C_type>{}, tiled_copy_t2r);
        cute::ThrCopy thread_r2s = tiled_r2s.get_slice(threadIdx.x);
        auto tRS_rAcc0 = thread_r2s.retile_S(tCrAcc0);
        auto tRS_rAcc1 = thread_r2s.retile_S(tCrAcc1);
        auto tRS_sC = thread_r2s.partition_D(tCsC);

        int num_tiles_executed = 0;
        for (int m_tile = tile_m_begin; m_tile < tile_m_end; ++m_tile) {
          for (int n_tile = tile_n_begin; n_tile < tile_n_end; ++n_tile) {
            int epi_rd_acc_buf        = num_tiles_executed % NUM_ACC_STAGE;
            int epi_rd_acc_full_phase = num_tiles_executed / NUM_ACC_STAGE % 2;
            cute::wait_barrier(shared_storage.acc_full_mbar_ptr[epi_rd_acc_buf], epi_rd_acc_full_phase);

            auto tCtAcc_tile = tCtC(cute::make_coord(cute::_, cute::_), cute::_0{}, cute::_0{}, epi_rd_acc_buf);
            auto tCtAcc_epi = cute::flat_divide(tCtAcc_tile, epi_tile_shape);

            auto tCgBias_tile = gBias(cute::_, cute::_, m_tile, n_tile);
            auto tCgBias_epi = cute::flat_divide(tCgBias_tile, epi_tile_shape);

            auto load_subtile = [&](int subtile_idx, auto &tCrAccStage, auto &tCrBiasStage) {
              auto tCtAcc_subtile = tCtAcc_epi(cute::_, cute::_, cute::_0{}, subtile_idx);
              cute::copy(tiled_copy_t2r, thr_copy_t2r.partition_S(tCtAcc_subtile), tCrAccStage);
              if constexpr (!NOBIAS) {
                auto tCgBias_subtile = tCgBias_epi(cute::_, cute::_, cute::_0{}, subtile_idx);
                cute::copy(thr_copy_t2r.partition_D(tCgBias_subtile), tCrBiasStage);
              }
            };

            load_subtile(0, tCrAcc0, tCrBias0);
            epilogue_wg_barrier.arrive_and_wait();
            if (cute::elect_one_sync()) {
              cute::arrive_barrier(shared_storage.acc_empty_mbar_ptr[epi_rd_acc_buf]);
            }

            for (int subtile_idx = 0; subtile_idx < EPI_PIPE_DEPTH; ++subtile_idx) {
              int stage_idx = subtile_idx & 1;
              int next_stage_idx = (subtile_idx + 1) & 1;
              auto &tCrAcc = stage_idx == 0 ? tCrAcc0 : tCrAcc1;
              auto &tCrBias = stage_idx == 0 ? tCrBias0 : tCrBias1;
              auto &tCrAccNext = next_stage_idx == 0 ? tCrAcc0 : tCrAcc1;
              auto &tCrBiasNext = next_stage_idx == 0 ? tCrBias0 : tCrBias1;

              if (subtile_idx + 1 < EPI_PIPE_DEPTH) {
                load_subtile(subtile_idx + 1, tCrAccNext, tCrBiasNext);
              }

              if constexpr (!NOBIAS) {
                CUTE_UNROLL
                for (int i = 0; i < tCrAcc.size(); i++) {
                  tCrAcc[i] += tCrBias[i];
                }
              }

              int c_smem_wr_buffer = subtile_idx % NUM_C_STAGE;
              auto &tRS_rAcc = stage_idx == 0 ? tRS_rAcc0 : tRS_rAcc1;
              cute::copy(tiled_r2s, tRS_rAcc, tRS_sC(cute::_, cute::_, cute::_, c_smem_wr_buffer));

              cute::tma_store_fence();
              epilogue_wg_barrier.arrive_and_wait();

              if (warp_idx == 0 && cute::elect_one_sync()) {
                C_type *sC_stage_ptr = static_cast<C_type *>(
                    cute::raw_pointer_cast(tCsC(cute::_, cute::_, c_smem_wr_buffer).data()));
                int out_n_coord = n_tile * EPI_PIPE_DEPTH + subtile_idx;
                if constexpr (SplitK) {
                  tma_out.tma_reduce_add_async(
                      sC_stage_ptr, {0, out_n_coord, m_tile * MMA_M});
                } else {
                  tma_out.tma_store_async(
                      sC_stage_ptr, {0, out_n_coord, m_tile * MMA_M});
                }
                cute::tma_store_arrive();
                cute::tma_store_wait<NUM_C_STAGE - 1>();
              }
              epilogue_wg_barrier.arrive_and_wait();
            }
            num_tiles_executed++;
          }
        }
        if (warp_idx == 0 && cute::elect_one_sync()) {
          cute::tma_store_wait<0>();
        }
    }
    __syncthreads();

    if (warp_idx == 0) {
      tmem_allocator.free(shared_storage.tmem_acc_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    }
}

};  // namespace kernel

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
