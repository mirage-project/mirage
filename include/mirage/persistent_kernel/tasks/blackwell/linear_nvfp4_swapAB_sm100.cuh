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

// Computes C = X * W^T by running the MMA as C^T = W * X^T so the small batch
// dimension maps to MMA_N. The output is written back directly in row-major
// [batch, output] order during the epilogue.
template <typename T_,
          typename TMA_A,
          typename TMA_B,
          typename TMA_SFA,
          typename TMA_SFB,
          class BiasTensor,
          class OutputTensor,
          int MMA_M,
          int MMA_N,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int SCALE_VECTOR_SIZE,
          bool NOBIAS,
          int NUM_AB_STAGE = 8,
          int NUM_ACC_STAGE = 2>
__device__ __noinline__ void
linear_nvfp4_smallm_swapAB_sm100_task_impl(const TMA_A &tma_a,
                                           const TMA_B &tma_b,
                                           const TMA_SFA &tma_sfa,
                                           const TMA_SFB &tma_sfb,
                                           BiasTensor mBias,
                                           OutputTensor mOutput,
                                           int logical_batch_size) {

  static_assert(std::is_same_v<T_, cutlass::float_e2m1_t>,
                "T_ must be cutlass::float_e2m1_t");
  static_assert(SCALE_VECTOR_SIZE == 16, "SCALE_VECTOR_SIZE must be 16");
  static_assert(MMA_M == 128, "MMA_M must be 128");
  static_assert(MMA_N % 8 == 0 && MMA_N != 0 && MMA_N <= 128,
                "MMA_N must be {8, 16, ... 128} in steps of 8");
  static_assert(REDUCTION_SIZE % 256 == 0,
                "REDUCTION_SIZE must be divisible by 256");

  constexpr int MMA_K = 64;
  using A_type = cutlass::float_e2m1_t;
  using B_type = cutlass::float_e2m1_t;
  using SF_type = cutlass::float_ue4m3_t;
  using C_type = float;

  constexpr int B_FP4 = 1;
  constexpr int B_SF = 0;
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int NUM_MMA_M = 1;
  constexpr int NUM_MMA_N = 1;
  constexpr int NUM_MMA_K = 4;
  constexpr int bM = MMA_M * NUM_MMA_M;
  constexpr int bN = MMA_N * NUM_MMA_N;
  constexpr int bK = MMA_K * NUM_MMA_K;
  constexpr int MMA_K_SF = MMA_K / SCALE_VECTOR_SIZE;
  constexpr int MMA_N_SFB = ((MMA_N + 127) / 128) * 128;

  int warp_idx = cutlass::canonical_warp_idx_sync();

  cute::TiledMMA tiled_mma = cute::make_tiled_mma(
      cute::SM100_MMA_MXF4_SS<A_type,
                              B_type,
                              C_type,
                              SF_type,
                              MMA_M,
                              MMA_N,
                              SCALE_VECTOR_SIZE,
                              cute::UMMA::Major::K,
                              cute::UMMA::Major::K>{});

  using TiledMma = decltype(tiled_mma);
  using ThrBlkTileShape_MNK =
      cute::Shape<cute::Int<bM>, cute::Int<bN>, cute::Int<bK>>;
  using AccLoadOp = cute::SM100_TMEM_LOAD_32dp32b2x;
  using Sm1xxBlkScaledConfig =
      cutlass::detail::Sm1xxBlockScaledConfig<SCALE_VECTOR_SIZE>;
  using SmemLayoutAtomSFA =
      decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFA(
          TiledMma{}, ThrBlkTileShape_MNK{}));
  using SmemLayoutAtomSFB =
      decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFB(
          TiledMma{}, ThrBlkTileShape_MNK{}));
  using DummySmemLayout =
      decltype(cute::make_layout(cute::make_shape(cute::Int<1>{}),
                                 cute::make_stride(cute::Int<1>{})));

  auto tb_mma_coord_vmnk = cute::make_coord(0, cute::_, cute::_, cute::_);
  auto tb_mma_coord = cute::select<1, 2, 3>(tb_mma_coord_vmnk);
  auto tb_mma_tiler = ThrBlkTileShape_MNK{};
  auto tb_sf_tiler = cute::make_shape(cute::Int<bM>{},
                                      cute::Int<bN>{},
                                      cute::Int<bK / SCALE_VECTOR_SIZE>{});

  cute::Tensor mA = cute::make_coord_tensor(
      cute::make_layout(cute::make_shape(OUTPUT_SIZE, REDUCTION_SIZE),
                        cute::make_stride(cute::E<1>{}, cute::E<0>{})));
  cute::Tensor mB = cute::make_coord_tensor(
      cute::make_layout(cute::make_shape(MMA_N, REDUCTION_SIZE),
                        cute::make_stride(cute::E<1>{}, cute::E<0>{})));
  cute::Tensor mSFA = cute::make_coord_tensor(
      cute::make_layout(cute::make_shape(OUTPUT_SIZE,
                                         REDUCTION_SIZE / SCALE_VECTOR_SIZE),
                        cute::make_stride(cute::E<1>{}, cute::E<0>{})));

  cute::Tensor gA = cute::local_tile(mA, tb_mma_tiler, tb_mma_coord,
                                     cute::Step<cute::_1, cute::X, cute::_1>{});
  cute::Tensor gB = cute::local_tile(mB, tb_mma_tiler, tb_mma_coord,
                                     cute::Step<cute::X, cute::_1, cute::_1>{});
  cute::Tensor gSFA = cute::local_tile(mSFA, tb_sf_tiler, tb_mma_coord,
                                       cute::Step<cute::_1, cute::X, cute::_1>{});

  auto cta_ind = cute::get<0>(tb_mma_coord_vmnk);
  cute::ThrMMA cta_mma = tiled_mma.get_slice(cta_ind);
  cute::Tensor tCgA = cta_mma.partition_A(gA);
  cute::Tensor tCgB = cta_mma.partition_B(gB);
  cute::Tensor tCgSFA = cta_mma.partition_A(gSFA);

  auto mma_shape_A = cute::partition_shape_A(
      tiled_mma,
      cute::make_shape(cute::Int<bM>{}, cute::Int<bK>{},
                       cute::Int<NUM_AB_STAGE>{}));
  auto mma_shape_B = cute::partition_shape_B(
      tiled_mma,
      cute::make_shape(cute::Int<bN>{}, cute::Int<bK>{},
                       cute::Int<NUM_AB_STAGE>{}));
  auto sA_layout = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<A_type>{}, mma_shape_A);
  auto sB_layout = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<B_type>{}, mma_shape_B);
  auto sSFA_layout = cute::tile_to_shape(
      SmemLayoutAtomSFA{},
      cute::append(cute::shape(SmemLayoutAtomSFA{}),
                   cute::Int<NUM_AB_STAGE>{}));
  auto sSFB_layout = cute::tile_to_shape(
      SmemLayoutAtomSFB{},
      cute::append(cute::shape(SmemLayoutAtomSFB{}),
                   cute::Int<NUM_AB_STAGE>{}));

  using SharedStorage = PipedScaledSharedStorage<A_type,
                                                 B_type,
                                                 C_type,
                                                 SF_type,
                                                 decltype(sA_layout),
                                                 decltype(sB_layout),
                                                 DummySmemLayout,
                                                 decltype(sSFA_layout),
                                                 decltype(sSFB_layout),
                                                 NUM_AB_STAGE,
                                                 NUM_ACC_STAGE>;

  extern __shared__ char shared_memory[];
  uintptr_t aligned_smem =
      (reinterpret_cast<uintptr_t>(shared_memory) + 127) / 128 * 128;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(aligned_smem);

  cute::Tensor tCsA = shared_storage.tensor_sA();
  cute::Tensor tCsB = shared_storage.tensor_sB();
  cute::Tensor tCsSFA = shared_storage.tensor_sSFA();
  cute::Tensor tCsSFB = shared_storage.tensor_sSFB();

  if (warp_idx == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier, NUM_AB_STAGE>(
        shared_storage.ab_full_mbar_ptr, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, NUM_AB_STAGE>(
        shared_storage.ab_empty_mbar_ptr, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier, NUM_AB_STAGE>(
        shared_storage.sf_full_mbar_ptr, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, NUM_AB_STAGE>(
        shared_storage.sf_empty_mbar_ptr, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, NUM_ACC_STAGE>(
        shared_storage.acc_full_mbar_ptr, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, NUM_ACC_STAGE>(
        shared_storage.acc_empty_mbar_ptr, 4);
  }
  __syncthreads();

  cutlass::arch::NamedBarrier tmem_allocation_result_barrier(
      32 + 128, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
  cutlass::arch::NamedBarrier epilogue_wg_barrier(
      128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
  Barrier *ab_full_mbar_ptr =
      reinterpret_cast<Barrier *>(shared_storage.ab_full_mbar_ptr);
  Barrier *sf_full_mbar_ptr =
      reinterpret_cast<Barrier *>(shared_storage.sf_full_mbar_ptr);

  constexpr int tma_transaction_bytes_AB =
      MMA_M * MMA_K / 2 * NUM_MMA_K + MMA_N * MMA_K / 2 * NUM_MMA_K;
  constexpr int tma_transaction_bytes_SF =
      MMA_M * MMA_K_SF * NUM_MMA_K +
      MMA_N_SFB * MMA_K_SF * NUM_MMA_K;
  void *sA_ptr = static_cast<void *>(&shared_storage.A);
  void *sB_ptr = static_cast<void *>(&shared_storage.B);
  SF_type *sSFA_ptr = shared_storage.SFA.begin();
  SF_type *sSFB_ptr = shared_storage.SFB.begin();

  using A_smem_TMA = smem_tma<A_type, B_FP4, M, S, MMA_M, MMA_K, 1>;
  using B_smem_TMA = smem_tma<B_type, B_FP4, M, S, MMA_N, MMA_K, 1>;
  using SFA_smem_TMA = smem_tma<SF_type, B_SF, M, S, MMA_M, MMA_K_SF, 1>;
  using SFB_smem_TMA =
      smem_tma<SF_type, B_SF, M, S, MMA_N_SFB, MMA_K_SF, 1>;

  A_smem_TMA sA(sA_ptr);
  B_smem_TMA sB(sB_ptr);
  SFA_smem_TMA sSFA(sSFA_ptr);
  SFB_smem_TMA sSFB(sSFB_ptr);

  cutlass::arch::fence_barrier_init();
  __syncthreads();

  cute::Tensor tCfA = cta_mma.make_fragment_A(tCsA);
  cute::Tensor tCfB = cta_mma.make_fragment_B(tCsB);
  auto acc_shape = cute::partition_shape_C(
      tiled_mma,
      cute::make_shape(cute::size<0>(tb_mma_tiler),
                       cute::size<1>(tb_mma_tiler),
                       cute::Int<NUM_ACC_STAGE>{}));
  cute::Tensor tCtC = tiled_mma.make_fragment_C(acc_shape);

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (warp_idx == 0) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &shared_storage.tmem_acc_ptr);
  }
  __syncthreads();
  tCtC.data() = cute::make_tmem_ptr<C_type>(shared_storage.tmem_acc_ptr);

  using FrgTypeSFA =
      cute::UMMA::tmem_sf_frg<SF_type, SCALE_VECTOR_SIZE, 1, true>;
  using FrgTypeSFB =
      cute::UMMA::tmem_sf_frg<SF_type, SCALE_VECTOR_SIZE, 1, false>;
  auto sfa_tmem_shape = cute::make_shape(
      cute::make_shape(
          cute::Int<MMA_M>{},
          cute::make_shape(cute::Int<SCALE_VECTOR_SIZE>{},
                           cute::Int<MMA_K / SCALE_VECTOR_SIZE>{})),
      cute::Int<NUM_MMA_M>{}, cute::Int<NUM_MMA_K>{});
  auto sfb_tmem_shape = cute::make_shape(
      cute::make_shape(
          cute::Int<MMA_N_SFB>{},
          cute::make_shape(cute::Int<SCALE_VECTOR_SIZE>{},
                           cute::Int<MMA_K / SCALE_VECTOR_SIZE>{})),
      cute::Int<NUM_MMA_N>{}, cute::Int<NUM_MMA_K>{});
  auto tCtSFA = FrgTypeSFA::make(sfa_tmem_shape);
  auto tCtSFB = FrgTypeSFB::make(sfb_tmem_shape);

  uint32_t sfa_offset = cutlass::detail::find_tmem_tensor_col_offset(tCtC);
  uint32_t sfb_offset =
      sfa_offset + cutlass::detail::find_tmem_tensor_col_offset(tCtSFA);
  shared_storage.tmem_sfa_ptr = shared_storage.tmem_acc_ptr + sfa_offset;
  shared_storage.tmem_sfb_ptr = shared_storage.tmem_acc_ptr + sfb_offset;
  tCtSFA.data() = make_tmem_ptr<SF_type>(shared_storage.tmem_sfa_ptr);
  tCtSFB.data() = make_tmem_ptr<SF_type>(shared_storage.tmem_sfb_ptr);

  int k_tile_count = cute::size<4>(tCgA);
  int m_tile = blockIdx.x;
  // n_tile: which MMA_N-row chunk of the input batch this CTA processes.
  // blockIdx.y == 0 for M<=8 (single tile); blockIdx.y >= 1 for larger M.
  int n_tile = blockIdx.y;
  // sfb_outer = n_tile: prepared_x_sf[t] already has the right SFs for tile t.
  int sfb_outer = n_tile;

  if (warp_idx == 6) {
    int tma_wr_k_tile = 0;
    int smem_wr_buffer = 0;
    int tma_wr_ab_empty_phase = 1;
    bool peek_ab_empty_status = kernel::try_wait_barrier(
        shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
        tma_wr_ab_empty_phase);

    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
      int tma_wr_k_tile_next = tma_wr_k_tile + 1;
      int smem_wr_buffer_next = tma_wr_k_tile_next % NUM_AB_STAGE;
      int tma_wr_ab_empty_phase_next =
          smem_wr_buffer_next == 0 ? tma_wr_ab_empty_phase ^ 1
                                   : tma_wr_ab_empty_phase;
      int tma_coords_A[2] = {k_tile * bK, m_tile * MMA_M};
      int tma_coords_B[2] = {k_tile * bK, n_tile * MMA_N};
      void *sA_stage_ptr = static_cast<void *>(cute::raw_pointer_cast(
          tCsA(cute::_, cute::_, cute::_, smem_wr_buffer).data()));
      void *sB_stage_ptr = static_cast<void *>(cute::raw_pointer_cast(
          tCsB(cute::_, cute::_, cute::_, smem_wr_buffer).data()));

      if (!peek_ab_empty_status) {
        cute::wait_barrier(shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                           tma_wr_ab_empty_phase);
      }

      if (cute::elect_one_sync()) {
        sA.set_ptr(sA_stage_ptr);
        sB.set_ptr(sB_stage_ptr);
        cute::set_barrier_transaction_bytes(
            shared_storage.ab_full_mbar_ptr[smem_wr_buffer],
            tma_transaction_bytes_AB);
        tma_a.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer], sA.base_ptr,
                           tma_coords_A);
        tma_b.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer], sB.base_ptr,
                           tma_coords_B);
      }

      if (tma_wr_k_tile_next < k_tile_count) {
        peek_ab_empty_status = kernel::try_wait_barrier(
            shared_storage.ab_empty_mbar_ptr[smem_wr_buffer_next],
            tma_wr_ab_empty_phase_next);
      }

      tma_wr_k_tile = tma_wr_k_tile_next;
      smem_wr_buffer = smem_wr_buffer_next;
      tma_wr_ab_empty_phase = tma_wr_ab_empty_phase_next;
    }
  } else if (warp_idx == 5) {
    int tma_wr_k_tile = 0;
    int smem_wr_buffer = 0;
    int tma_wr_sf_empty_phase = 1;
    bool peek_sf_empty_status = kernel::try_wait_barrier(
        shared_storage.sf_empty_mbar_ptr[smem_wr_buffer],
        tma_wr_sf_empty_phase);

    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
      int tma_wr_k_tile_next = tma_wr_k_tile + 1;
      int smem_wr_buffer_next = tma_wr_k_tile_next % NUM_AB_STAGE;
      int tma_wr_sf_empty_phase_next =
          smem_wr_buffer_next == 0 ? tma_wr_sf_empty_phase ^ 1
                                   : tma_wr_sf_empty_phase;
      int tma_coords_SFA[3] = {0, k_tile * NUM_MMA_K, m_tile};
      int tma_coords_SFB[3] = {0, k_tile * NUM_MMA_K, sfb_outer};
      void *sSFA_stage_ptr = static_cast<void *>(
          tCsSFA(cute::_, cute::_, cute::_, smem_wr_buffer).data().get());
      void *sSFB_stage_ptr = static_cast<void *>(
          tCsSFB(cute::_, cute::_, cute::_, smem_wr_buffer).data().get());

      if (!peek_sf_empty_status) {
        cute::wait_barrier(shared_storage.sf_empty_mbar_ptr[smem_wr_buffer],
                           tma_wr_sf_empty_phase);
      }

      if (cute::elect_one_sync()) {
        sSFA.set_ptr(sSFA_stage_ptr);
        sSFB.set_ptr(sSFB_stage_ptr);
        cute::set_barrier_transaction_bytes(
            shared_storage.sf_full_mbar_ptr[smem_wr_buffer],
            tma_transaction_bytes_SF);
        tma_sfa.tma_cp_async(
            sf_full_mbar_ptr[smem_wr_buffer],
            reinterpret_cast<cute::half_t *>(sSFA.base_ptr),
            tma_coords_SFA);
        tma_sfb.tma_cp_async(
            sf_full_mbar_ptr[smem_wr_buffer],
            reinterpret_cast<cute::half_t *>(sSFB.base_ptr),
            tma_coords_SFB);
      }

      if (tma_wr_k_tile_next < k_tile_count) {
        peek_sf_empty_status = kernel::try_wait_barrier(
            shared_storage.sf_empty_mbar_ptr[smem_wr_buffer_next],
            tma_wr_sf_empty_phase_next);
      }

      tma_wr_k_tile = tma_wr_k_tile_next;
      smem_wr_buffer = smem_wr_buffer_next;
      tma_wr_sf_empty_phase = tma_wr_sf_empty_phase_next;
    }
  } else if (warp_idx == 4) {
    tmem_allocation_result_barrier.arrive_and_wait();
    auto copy_sf_s2t = [&](int stage) {
      using UtccpOp = SM100_UTCCP_4x32dp128bit_1cta;
      auto copy_one = [&](auto &tCsSF, auto &tCtSF) {
        auto tCsSF_stage = tCsSF(cute::_, cute::_, cute::_, stage);
        auto tCsSF_compact =
            make_tensor(tCsSF_stage.data(), filter_zeros(tCsSF_stage.layout()));
        auto tCtSF_compact =
            make_tensor(tCtSF.data(), filter_zeros(tCtSF.layout()));
        auto tiled_s2t = make_utccp_copy(UtccpOp{}, tCtSF_compact);
        auto thr_s2t = tiled_s2t.get_slice(0);
        auto src_ = thr_s2t.partition_S(tCsSF_compact);
        auto src = get_utccp_smem_desc_tensor<UtccpOp>(src_);
        auto dst = thr_s2t.partition_D(tCtSF_compact);
        cute::copy(tiled_s2t, src, dst);
      };
      copy_one(tCsSFA, tCtSFA);
      copy_one(tCsSFB, tCtSFB);
    };

    int acc_buf_idx = 0;
    int acc_empty_phase = 1;
    cute::wait_barrier(shared_storage.acc_empty_mbar_ptr[acc_buf_idx],
                       acc_empty_phase);

    tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;
    int mma_rd_k_tile = 0;
    int smem_rd_buffer = 0;
    int mma_rd_ab_full_phase = 0;
    int mma_rd_sf_full_phase = 0;
    bool peek_ab_full_status = kernel::try_wait_barrier(
        shared_storage.ab_full_mbar_ptr[smem_rd_buffer],
        mma_rd_ab_full_phase);
    bool peek_sf_full_status = kernel::try_wait_barrier(
        shared_storage.sf_full_mbar_ptr[smem_rd_buffer],
        mma_rd_sf_full_phase);

    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
      int mma_rd_k_tile_next = mma_rd_k_tile + 1;
      int smem_rd_buffer_next = mma_rd_k_tile_next % NUM_AB_STAGE;
      int mma_rd_ab_full_phase_next =
          smem_rd_buffer_next == 0 ? mma_rd_ab_full_phase ^ 1
                                   : mma_rd_ab_full_phase;
      int mma_rd_sf_full_phase_next =
          smem_rd_buffer_next == 0 ? mma_rd_sf_full_phase ^ 1
                                   : mma_rd_sf_full_phase;

      if (!peek_ab_full_status) {
        cute::wait_barrier(shared_storage.ab_full_mbar_ptr[smem_rd_buffer],
                           mma_rd_ab_full_phase);
      }
      if (!peek_sf_full_status) {
        cute::wait_barrier(shared_storage.sf_full_mbar_ptr[smem_rd_buffer],
                           mma_rd_sf_full_phase);
      }
      if (cute::elect_one_sync()) {
        copy_sf_s2t(smem_rd_buffer);
      }

      auto accumulate = tiled_mma.accumulate_;
      for (int k_block = 0; k_block < cute::size<2>(tCfA); ++k_block) {
        cute::gemm(
            tiled_mma.with(accumulate, tCtSFA(cute::_, cute::_, k_block),
                           tCtSFB(cute::_, cute::_, k_block)),
            tCfA(cute::_, cute::_, k_block, smem_rd_buffer),
            tCfB(cute::_, cute::_, k_block, smem_rd_buffer),
            tCtC(cute::_, cute::_, cute::_, acc_buf_idx));
        accumulate = cute::UMMA::ScaleOut::One;
      }
      tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;

      cutlass::arch::umma_arrive(
          &shared_storage.ab_empty_mbar_ptr[smem_rd_buffer]);
      cutlass::arch::umma_arrive(
          &shared_storage.sf_empty_mbar_ptr[smem_rd_buffer]);

      if (mma_rd_k_tile_next < k_tile_count) {
        peek_ab_full_status = kernel::try_wait_barrier(
            shared_storage.ab_full_mbar_ptr[smem_rd_buffer_next],
            mma_rd_ab_full_phase_next);
        peek_sf_full_status = kernel::try_wait_barrier(
            shared_storage.sf_full_mbar_ptr[smem_rd_buffer_next],
            mma_rd_sf_full_phase_next);
      }
      mma_rd_k_tile = mma_rd_k_tile_next;
      smem_rd_buffer = smem_rd_buffer_next;
      mma_rd_ab_full_phase = mma_rd_ab_full_phase_next;
      mma_rd_sf_full_phase = mma_rd_sf_full_phase_next;
    }

    cutlass::arch::umma_arrive(&shared_storage.acc_full_mbar_ptr[acc_buf_idx]);
  } else if (warp_idx < 4) {
    tmem_allocation_result_barrier.arrive_and_wait();

    auto epi_tile_shape =
        cute::make_shape(cute::Int<MMA_M>{}, cute::Int<MMA_N>{});
    cute::Tensor tCtAcc_proto = tCtC(cute::make_coord(cute::_, cute::_),
                                     cute::_0{}, cute::_0{}, cute::_0{});
    cute::Tensor tCtAcc_epi_proto =
        cute::flat_divide(tCtAcc_proto, epi_tile_shape);
    cute::Tensor tCtAcc_subtile_proto =
        tCtAcc_epi_proto(cute::_, cute::_, cute::_0{}, cute::_0{});

    cute::TiledCopy tiled_copy_t2r =
        cute::make_tmem_copy(AccLoadOp{}, tCtAcc_subtile_proto);
    cute::ThrCopy thr_copy_t2r = tiled_copy_t2r.get_slice(threadIdx.x);
    cute::Tensor tCrCoord =
        thr_copy_t2r.partition_D(cute::make_identity_tensor(epi_tile_shape));
    cute::Tensor tCrAcc =
        cute::make_tensor<C_type>(cute::shape(tCrCoord));

    int epi_rd_acc_buf = 0;
    int epi_rd_acc_full_phase = 0;
    cute::wait_barrier(shared_storage.acc_full_mbar_ptr[epi_rd_acc_buf],
                       epi_rd_acc_full_phase);

    auto tCtAcc_tile = tCtC(cute::make_coord(cute::_, cute::_), cute::_0{},
                            cute::_0{}, epi_rd_acc_buf);
    auto tCtAcc_epi = cute::flat_divide(tCtAcc_tile, epi_tile_shape);
    auto tCtAcc_subtile = tCtAcc_epi(cute::_, cute::_, cute::_0{}, cute::_0{});
    cute::copy(tiled_copy_t2r, thr_copy_t2r.partition_S(tCtAcc_subtile), tCrAcc);
    cutlass::arch::fence_view_async_tmem_load();

    epilogue_wg_barrier.arrive_and_wait();
    if (warp_idx == 0 && cute::elect_one_sync()) {
      cute::arrive_barrier(shared_storage.acc_empty_mbar_ptr[epi_rd_acc_buf]);
    }
    epilogue_wg_barrier.arrive_and_wait();

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tCrAcc); ++i) {
      auto coord = tCrCoord(i);
      int output_col = static_cast<int>(cute::get<0>(coord)) + m_tile * MMA_M;
      int batch_row = n_tile * MMA_N + static_cast<int>(cute::get<1>(coord));
      bool pred = output_col < OUTPUT_SIZE && batch_row < logical_batch_size;
      if (!pred) {
        continue;
      }

      C_type value = tCrAcc(i);
      if constexpr (!NOBIAS) {
        value += mBias(batch_row, output_col);
      }
      mOutput(batch_row, output_col) = value;
    }
  }

  __syncthreads();
  if (warp_idx == 0) {
    tmem_allocator.free(shared_storage.tmem_acc_ptr,
                        TmemAllocator::Sm100TmemCapacityColumns);
  }
}


} // namespace kernel

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
