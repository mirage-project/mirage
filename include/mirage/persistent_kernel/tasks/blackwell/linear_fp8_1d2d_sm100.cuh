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
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <type_traits>

// Cutlass includes
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

// CuTe includes
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>

#include "../common/dmem_layout.cuh"
#include "../common/worker_config.h"
#include "../hopper/barrier.cuh"
#include "../hopper/smem_layout_tma.cuh"
#include "../hopper/tma.cuh"
#include "sm100_utils.cuh"
#include "storage.cuh"

namespace kernel {

namespace detail {

// 128 uint32 elements per aligned scale chunk, same assumption as deepgemm.
static constexpr int kNumUTCCPAlignedElems = 128;

// Warp-local transpose required before UTCCP.
// Input/output is a contiguous 128 x uint32 region in shared memory.
__device__ __forceinline__ void
    utccp_required_smem_warp_transpose(uint32_t *smem_ptr, int lane_idx) {
  uint32_t values[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    values[i] = smem_ptr[(i ^ (lane_idx >> 3)) * 32 + lane_idx];
  }
  __syncwarp();
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    smem_ptr[lane_idx * 4 + (i ^ (lane_idx >> 3))] = values[i];
  }
}

} // namespace detail

template <typename T_,
          typename TMA_A,
          typename TMA_B,
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
    linear_fp8_1d2d_sm100_task_impl(const TMA_A &tma_a,
                                    const TMA_B &tma_b,
                                    uint32_t const *weight_scale_ptr,
                                    uint32_t const *input_scale_ptr,
                                    BiasTensor mBias,
                                    const TMA_OUT &tma_out) {
  using Barrier = cutlass::arch::ClusterTransactionBarrier;
  using TypeScale = uint32_t;
  using TypeC = cute::bfloat16_t;
  using TypeBias = cute::bfloat16_t;

  constexpr int OUTPUT_ATOM_SIZE = 128;
  constexpr int TILE_K = 128;
  constexpr int UMMA_K = 32;
  constexpr int NUM_K_SUBTILES = TILE_K / UMMA_K; // 4
  constexpr int SCALE_K = REDUCTION_SIZE / TILE_K;
  constexpr int PADDED_SCALE_K = ((SCALE_K + 3) / 4) * 4;

  // We assume per-128-K packed block scales:
  // one uint32 per row/col per K-tile, containing 4x UE8M0 bytes.
  constexpr int SF_BLOCK_M =
      ((OUTPUT_ATOM_SIZE + detail::kNumUTCCPAlignedElems - 1) /
       detail::kNumUTCCPAlignedElems) *
      detail::kNumUTCCPAlignedElems;
  constexpr int SF_BLOCK_N = ((MMA_N + detail::kNumUTCCPAlignedElems - 1) /
                              detail::kNumUTCCPAlignedElems) *
                             detail::kNumUTCCPAlignedElems;

  constexpr int kNumSFATmemCols = SF_BLOCK_M / 32;
  constexpr int kNumSFBTmemCols = SF_BLOCK_N / 32;
  constexpr int kNumAccumTmemCols = MMA_N * NUM_ACC_STAGE;
  constexpr int kTmemStartColOfSFA = kNumAccumTmemCols;
  constexpr int kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
  constexpr int num_tmem_columns =
      kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols; // TODO

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_idx = kernel::lane_id();

  auto mma_coord_vmnk = cute::make_coord(0, // peer CTA coordinate
                                         cute::_,
                                         cute::_,
                                         cute::_);

  // --------------------------------------------------------------------------
  // Block-scaled FP8 MMA
  // --------------------------------------------------------------------------
  using mma_issue_t = kernel::sm100::SM100_MMA_MXF8F6F4_SS;

  using mma_plumbing_t = cute::SM100_MMA_MXF8F6F4_SS<T_,
                                                     T_,
                                                     float,
                                                     cutlass::float_ue8m0_t,
                                                     MMA_M,
                                                     MMA_N,
                                                     cute::UMMA::Major::K,
                                                     cute::UMMA::Major::K>;

  auto tiled_mma = cute::make_tiled_mma(mma_plumbing_t{});

  auto bM = cute::Int<MMA_M>{};
  auto bN = cute::Int<MMA_N>{};
  auto bK = cute::Int<TILE_K>{};
  auto mma_tiler = cute::make_shape(bM, bN, bK);
  auto mma_coord = cute::select<1, 2, 3>(mma_coord_vmnk);
  cute::Tensor mA = cute::make_coord_tensor(
      cute::make_layout(cute::make_shape(OUTPUT_SIZE, REDUCTION_SIZE),
                        cute::make_stride(cute::E<1>{}, cute::E<0>{})));

  cute::Tensor mB = cute::make_coord_tensor(
      cute::make_layout(cute::make_shape(BATCH_SIZE, REDUCTION_SIZE),
                        cute::make_stride(cute::E<1>{}, cute::E<0>{})));

  cute::Tensor gA = cute::local_tile(
      mA, mma_tiler, mma_coord, cute::Step<cute::_1, cute::X, cute::_1>{});

  cute::Tensor gB = cute::local_tile(
      mB, mma_tiler, mma_coord, cute::Step<cute::X, cute::_1, cute::_1>{});

  auto mma_shape_A =
      cute::partition_shape_A(tiled_mma,
                              cute::make_shape(cute::Int<MMA_M>{},
                                               cute::size<2>(mma_tiler),
                                               cute::Int<NUM_AB_STAGE>{}));

  auto mma_shape_B =
      cute::partition_shape_B(tiled_mma,
                              cute::make_shape(cute::Int<MMA_N>{},
                                               cute::size<2>(mma_tiler),
                                               cute::Int<NUM_AB_STAGE>{}));

  auto mma_shape_C =
      cute::make_shape(cute::make_shape(cute::Int<MMA_N>{}, cute::Int<MMA_M>{}),
                       cute::Int<1>{},
                       cute::Int<1>{},
                       cute::Int<NUM_C_STAGE>{});

  auto sA_layout = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<T_>{}, mma_shape_A);
  auto sB_layout = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<T_>{}, mma_shape_B);

  auto sC_layout_fake = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_INTER_Atom<TypeC>{}, mma_shape_C);

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

  auto sSFA_layout = cute::make_layout(
      cute::make_shape(cute::Int<SF_BLOCK_M>{}, cute::Int<NUM_AB_STAGE>{}),
      cute::make_stride(cute::Int<1>{}, cute::Int<SF_BLOCK_M>{}));

  auto sSFB_layout = cute::make_layout(
      cute::make_shape(cute::Int<SF_BLOCK_N>{}, cute::Int<NUM_AB_STAGE>{}),
      cute::make_stride(cute::Int<1>{}, cute::Int<SF_BLOCK_N>{}));

  using SharedStorage = PipedSharedStorageWithSF<T_,
                                                 T_,
                                                 TypeC,
                                                 TypeScale,
                                                 decltype(sA_layout),
                                                 decltype(sB_layout),
                                                 decltype(sC_layout),
                                                 decltype(sSFA_layout),
                                                 decltype(sSFB_layout),
                                                 NUM_AB_STAGE,
                                                 NUM_ACC_STAGE>;

  extern __shared__ char shared_memory[];
  uintptr_t aligned_smem =
      (reinterpret_cast<uintptr_t>(shared_memory) + 127) / 128 * 128;

  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(aligned_smem);

  // Access staged scale-factor buffers stored inside PipedSharedStorageWithSF.
  auto smem_sfa = [&](int stage) -> TypeScale * {
    return shared_storage.SFA.begin() + stage * SF_BLOCK_M;
  };

  auto smem_sfb = [&](int stage) -> TypeScale * {
    return shared_storage.SFB.begin() + stage * SF_BLOCK_N;
  };

  auto load_packed_scale_tile = [&](TypeScale *dst,
                                    TypeScale const *src,
                                    int row_base,
                                    int total_rows,
                                    int block_rows,
                                    int k_tile) {
#pragma unroll
    for (int i = lane_idx; i < block_rows; i += 32) {
      int global_row = row_base + i;
      dst[i] = global_row < total_rows
                   ? src[global_row * PADDED_SCALE_K + k_tile]
                   : TypeScale(0);
    }
  };

  // Prefetch TMA descriptors
  if (warp_idx == 0 && cute::elect_one_sync()) {
    kernel::tma::prefetch_tma_descriptor(tma_a.desc_ptr);
    kernel::tma::prefetch_tma_descriptor(tma_b.desc_ptr);
    kernel::tma::prefetch_tma_descriptor(tma_out.desc_ptr);
  }

  // --------------------------------------------------------------------------
  // Barrier init
  // --------------------------------------------------------------------------
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

  cutlass::arch::NamedBarrier tmem_allocation_result_barrier(
      32 + 128, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
  cutlass::arch::NamedBarrier epilogue_wg_barrier(
      128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

  // SMEM tensors
  cute::Tensor tCsA = shared_storage.tensor_sA();
  cute::Tensor tCsB = shared_storage.tensor_sB();
  cute::Tensor sC_epi = shared_storage.tensor_sC();

  auto mma_v = cute::get<0>(mma_coord_vmnk);
  cute::ThrMMA cta_mma = tiled_mma.get_slice(mma_v);

  cute::Tensor tCgA = cta_mma.partition_A(gA);
  cute::Tensor tCgB = cta_mma.partition_B(gB);

  // NOTE: tCrA/tCrB are still useful for shape/k-looping, but actual FP8
  // block-scaled issue below uses low-level descriptors, not
  // cute::gemm(tiled_mma, ...).
  cute::Tensor tCrA = cta_mma.make_fragment_A(tCsA);
  cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);

  auto acc_shape =
      cute::partition_shape_C(tiled_mma,
                              cute::make_shape(cute::size<0>(mma_tiler),
                                               cute::size<1>(mma_tiler),
                                               cute::Int<NUM_ACC_STAGE>{}));
  auto tCtAcc = tiled_mma.make_fragment_C(acc_shape);

  int tma_transaction_bytes =
      sizeof(T_) * cute::size<1>(mma_tiler) * cute::size<2>(mma_tiler) +
      sizeof(T_) * cute::size<0>(mma_tiler) * cute::size<2>(mma_tiler);

  constexpr int TILE_SIZE = 128;
  constexpr int INPUT_TMA_TILE_SIZE = 128;
  constexpr int WEIGHT_TMA_TILE_SIZE = 128;
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  T_ *shared_weight = shared_storage.A.begin();
  T_ *shared_input = shared_storage.B.begin();
  TypeC *mm_output = shared_storage.C.begin();
  Barrier *ab_full_mbar_ptr =
      reinterpret_cast<Barrier *>(shared_storage.ab_full_mbar_ptr);

  using InputSmem = smem_tma<T_, B, M, S, MMA_N, INPUT_TMA_TILE_SIZE, 1>;
  using WeightSmem =
      smem_tma<T_, B, M, S, OUTPUT_ATOM_SIZE, WEIGHT_TMA_TILE_SIZE, 1>;
  using OutputSmem = smem_tma<TypeC, 0, M, S, MMA_N, OUTPUT_ATOM_SIZE, 1>;

  InputSmem input_smem(shared_input);
  WeightSmem weight_smem(shared_weight);
  OutputSmem output_smem(mm_output);

  cutlass::arch::fence_barrier_init();
  __syncthreads();

  int k_tile_count = cute::size<4>(tCgA);
  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  cutlass::arch::fence_barrier_init();
  __syncthreads();

  // --------------------------------------------------------------------------
  // Warp 5: TMA loader for A/B and global-to-SMEM loader for SFA/SFB
  // --------------------------------------------------------------------------
  if (warp_idx == 5) {
    int total_k_tile_count = 0;
    for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
      for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {
        int num_prev_k_blk = total_k_tile_count;
        total_k_tile_count += k_tile_count;

        int tma_wr_k_tile = 0;
        int smem_wr_buffer = (num_prev_k_blk + tma_wr_k_tile) % NUM_AB_STAGE;
        int tma_wr_ab_empty_phase =
            ((num_prev_k_blk + tma_wr_k_tile) / NUM_AB_STAGE) % 2 ^ 1;

        bool peek_ab_empty_status = kernel::try_wait_barrier(
            shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
            tma_wr_ab_empty_phase);

        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
          int tma_wr_k_tile_next = tma_wr_k_tile + 1;
          int smem_wr_buffer_next =
              (num_prev_k_blk + tma_wr_k_tile_next) % NUM_AB_STAGE;
          int tma_wr_ab_empty_phase_next = smem_wr_buffer_next == 0
                                               ? (tma_wr_ab_empty_phase ^ 1)
                                               : tma_wr_ab_empty_phase;

          if (!peek_ab_empty_status) {
            cute::wait_barrier(shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                               tma_wr_ab_empty_phase);
          }

          int m_base = m_tile * OUTPUT_ATOM_SIZE;
          int n_base = n_tile * MMA_N;

          load_packed_scale_tile(smem_sfa(smem_wr_buffer),
                                 weight_scale_ptr,
                                 m_base,
                                 OUTPUT_SIZE,
                                 SF_BLOCK_M,
                                 k_tile);
          load_packed_scale_tile(smem_sfb(smem_wr_buffer),
                                 input_scale_ptr,
                                 n_base,
                                 BATCH_SIZE,
                                 SF_BLOCK_N,
                                 k_tile);
          __syncwarp();

#pragma unroll
          for (int i = 0; i < SF_BLOCK_M / detail::kNumUTCCPAlignedElems; ++i) {
            detail::utccp_required_smem_warp_transpose(
                smem_sfa(smem_wr_buffer) + i * detail::kNumUTCCPAlignedElems,
                lane_idx);
          }
#pragma unroll
          for (int i = 0; i < SF_BLOCK_N / detail::kNumUTCCPAlignedElems; ++i) {
            detail::utccp_required_smem_warp_transpose(
                smem_sfb(smem_wr_buffer) + i * detail::kNumUTCCPAlignedElems,
                lane_idx);
          }
          __syncwarp();
          cutlass::arch::fence_view_async_shared();
          __syncwarp();

          if (cute::elect_one_sync()) {
            int tma_coords_A[2] = {k_tile * TILE_SIZE,
                                   m_tile * OUTPUT_ATOM_SIZE};
            int tma_coords_B[2] = {k_tile * TILE_SIZE, n_tile * MMA_N};

            weight_smem.set_ptr(shared_weight +
                                smem_wr_buffer * OUTPUT_ATOM_SIZE * TILE_SIZE);
            input_smem.set_ptr(shared_input +
                               smem_wr_buffer * MMA_N * TILE_SIZE);

            cute::set_barrier_transaction_bytes(
                shared_storage.ab_full_mbar_ptr[smem_wr_buffer],
                tma_transaction_bytes);

            tma_a.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer],
                               weight_smem.base_ptr,
                               tma_coords_A);
            tma_b.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer],
                               input_smem.base_ptr,
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
      }
    }
  }
  // --------------------------------------------------------------------------
  // Warp 4: MMA issue warp
  // --------------------------------------------------------------------------
  else if (warp_idx == 4) {
    // Allocate TMEM
    tmem_allocation_result_barrier.arrive_and_wait();
    tCtAcc.data() = shared_storage.tmem_base_ptr;

    // These helpers are expected to exist in Mirage, or be ported from
    // deepgemm.
    auto instr_desc =
        cute::UMMA::make_instr_desc_block_scaled<T_,
                                                 T_,
                                                 float,
                                                 cutlass::float_ue8m0_t,
                                                 MMA_M,
                                                 MMA_N,
                                                 cute::UMMA::Major::K,
                                                 cute::UMMA::Major::K>();

    auto sf_desc = kernel::sm100::make_sf_desc(nullptr);

    auto a_desc = kernel::sm100::
        make_umma_desc<cute::UMMA::Major::K, OUTPUT_ATOM_SIZE, TILE_K, 128>(
            shared_weight, 0, 0);

    auto b_desc =
        kernel::sm100::make_umma_desc<cute::UMMA::Major::K, MMA_N, TILE_K, 128>(
            shared_input, 0, 0);

    uint32_t a_desc_lo =
        lane_idx < NUM_AB_STAGE
            ? (a_desc.lo +
               lane_idx * (OUTPUT_ATOM_SIZE * TILE_K * sizeof(T_)) / 16)
            : 0u;
    uint32_t b_desc_lo =
        lane_idx < NUM_AB_STAGE
            ? (b_desc.lo + lane_idx * (MMA_N * TILE_K * sizeof(T_)) / 16)
            : 0u;

    int total_k_tile_count = 0;
    int num_tiles_executed = 0;

    for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
      for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {
        int acc_buf_idx = num_tiles_executed % NUM_ACC_STAGE;
        int acc_empty_phase = (num_tiles_executed / NUM_ACC_STAGE) % 2 ^ 1;
        cute::wait_barrier(shared_storage.acc_empty_mbar_ptr[acc_buf_idx],
                           acc_empty_phase);

        auto tCtAcc_Slice = tCtAcc(cute::_, cute::_, cute::_, acc_buf_idx);

        int num_prev_k_blk = total_k_tile_count;
        total_k_tile_count += k_tile_count;

        int mma_rd_k_tile = 0;
        int smem_rd_buffer = (num_prev_k_blk + mma_rd_k_tile) % NUM_AB_STAGE;
        int mma_rd_ab_full_phase =
            ((num_prev_k_blk + mma_rd_k_tile) / NUM_AB_STAGE) % 2;

        bool peek_ab_full_status = kernel::try_wait_barrier(
            shared_storage.ab_full_mbar_ptr[smem_rd_buffer],
            mma_rd_ab_full_phase);

        // Zero init accumulator at beginning of tile
        // For block-scaled low-level mma_t::fma, this is controlled by the bool
        // "accumulate".
        bool accumulate = false;

        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
          int mma_rd_k_tile_next = mma_rd_k_tile + 1;
          int smem_rd_buffer_next =
              (num_prev_k_blk + mma_rd_k_tile_next) % NUM_AB_STAGE;
          int mma_rd_ab_full_phase_next = smem_rd_buffer_next == 0
                                              ? (mma_rd_ab_full_phase ^ 1)
                                              : mma_rd_ab_full_phase;

          if (!peek_ab_full_status) {
            cute::wait_barrier(shared_storage.ab_full_mbar_ptr[smem_rd_buffer],
                               mma_rd_ab_full_phase);
          }

          // UTCCP SFA -> TMEM columns
          if (cute::elect_one_sync()) {
            using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_1cta;

#pragma unroll
            for (int i = 0; i < SF_BLOCK_M / detail::kNumUTCCPAlignedElems;
                 ++i) {
              auto smem_ptr =
                  smem_sfa(smem_rd_buffer) + i * detail::kNumUTCCPAlignedElems;
              kernel::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
              cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
            }

#pragma unroll
            for (int i = 0; i < SF_BLOCK_N / detail::kNumUTCCPAlignedElems;
                 ++i) {
              auto smem_ptr =
                  smem_sfb(smem_rd_buffer) + i * detail::kNumUTCCPAlignedElems;
              kernel::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
              cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
            }
          }
          __syncwarp();

          // Base descriptors for this AB stage.
          auto const &a_desc_base_lo =
              __shfl_sync(0xffffffff, a_desc_lo, smem_rd_buffer);
          auto const &b_desc_base_lo =
              __shfl_sync(0xffffffff, b_desc_lo, smem_rd_buffer);

          if (cute::elect_one_sync()) {
#pragma unroll
            for (int k_sub = 0; k_sub < NUM_K_SUBTILES; ++k_sub) {
              const uint32_t sfa_id = k_sub;
              const uint32_t sfb_id = k_sub;
              auto const runtime_instr_desc =
                  kernel::sm100::make_runtime_instr_desc_with_sf_id(
                      instr_desc, sfa_id, sfb_id);

              b_desc.lo = kernel::sm100::
                  advance_umma_desc_lo<cute::UMMA::Major::K, MMA_N, 128, T_>(
                      b_desc_base_lo, 0, k_sub * UMMA_K);

              a_desc.lo =
                  kernel::sm100::advance_umma_desc_lo<cute::UMMA::Major::K,
                                                      OUTPUT_ATOM_SIZE,
                                                      128,
                                                      T_>(
                      a_desc_base_lo, 0, k_sub * UMMA_K);

              mma_issue_t::fma(a_desc,
                               b_desc,
                               acc_buf_idx * MMA_N,
                               accumulate,
                               runtime_instr_desc,
                               kTmemStartColOfSFA,
                               kTmemStartColOfSFB);
              accumulate = true;
            }
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
        }

        cutlass::arch::umma_arrive(
            &shared_storage.acc_full_mbar_ptr[acc_buf_idx]);
        ++num_tiles_executed;
      }
    }
  }
  // --------------------------------------------------------------------------
  // Epilogue warps
  // --------------------------------------------------------------------------
  else if (warp_idx < 4) {
    if (warp_idx == 0) {
      tmem_allocator.allocate(num_tmem_columns, &shared_storage.tmem_base_ptr);
    }
    tmem_allocation_result_barrier.arrive_and_wait();
    tCtAcc.data() = shared_storage.tmem_base_ptr;

    using AccType = typename decltype(tCtAcc)::value_type;

    cutlass::NumericConverter<AccType, TypeBias> converterBias;
    cutlass::NumericConverter<TypeC, AccType> converter;
    cutlass::NumericConverter<AccType, TypeC> output_to_acc_converter;

    cute::TiledCopy tiled_copy_t2r =
        cute::make_tmem_copy(cute::SM100_TMEM_LOAD_32dp32b1x{},
                             tCtAcc(cute::_, cute::_, cute::_, 0));

    cute::ThrCopy thr_copy_t2r = tiled_copy_t2r.get_slice(threadIdx.x);
    cute::Tensor tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc);

    cute::Tensor tCgC_fake = cute::make_tensor<TypeC>(
        cute::shape(tCtAcc(cute::_, cute::_, cute::_, 0)));
    cute::Tensor tTR_rAcc_fake = thr_copy_t2r.partition_D(tCgC_fake);
    cute::Tensor tTR_rAcc =
        cute::make_tensor<AccType>(cute::shape(tTR_rAcc_fake));

    int num_tiles_executed = 0;
    for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
      for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {
        int acc_buf_idx = num_tiles_executed % NUM_ACC_STAGE;
        int acc_full_phase = (num_tiles_executed / NUM_ACC_STAGE) % 2;
        int c_smem_wr_buffer_idx = num_tiles_executed % NUM_C_STAGE;
        cute::Tensor tCrC =
            cute::make_tensor<TypeC>(cute::shape(tTR_rAcc(0, cute::_, 0, 0)));

        output_smem.set_ptr(mm_output +
                            c_smem_wr_buffer_idx * MMA_N * OUTPUT_ATOM_SIZE);

        cute::wait_barrier(shared_storage.acc_full_mbar_ptr[acc_buf_idx],
                           acc_full_phase);

        cute::copy(tiled_copy_t2r,
                   tTR_tAcc(cute::_, cute::_, cute::_, cute::_, acc_buf_idx),
                   tTR_rAcc);

        epilogue_wg_barrier.arrive_and_wait();
        if (cute::elect_one_sync()) {
          cute::arrive_barrier(shared_storage.acc_empty_mbar_ptr[acc_buf_idx]);
        }

        CUTE_UNROLL
        for (int i = 0; i < tCrC.size(); i++) {
          tCrC[i] = converter(tTR_rAcc[i]);
        }

        cute::Tensor sC_epi_slice =
            cute::flatten(sC_epi(cute::_, 0, 0, c_smem_wr_buffer_idx));
        cute::copy(tCrC, sC_epi_slice(cute::_, threadIdx.x));

        if constexpr (!NOBIAS) {
          epilogue_wg_barrier.arrive_and_wait();

          for (int idx = threadIdx.x; idx < MMA_N * OUTPUT_ATOM_SIZE;
               idx += 128) {
            int local_row = idx / OUTPUT_ATOM_SIZE;
            int local_col = idx % OUTPUT_ATOM_SIZE;
            int global_row = n_tile * MMA_N + local_row;
            int global_col = m_tile * OUTPUT_ATOM_SIZE + local_col;

            if (global_row < BATCH_SIZE && global_col < OUTPUT_SIZE) {
              AccType acc =
                  output_to_acc_converter(output_smem.at(local_row, local_col));
              AccType bias = converterBias(mBias(global_row, global_col));
              output_smem.at(local_row, local_col) = converter(acc + bias);
            }
          }

          epilogue_wg_barrier.arrive_and_wait();
        }

        cute::tma_store_fence();
        epilogue_wg_barrier.arrive_and_wait();

        if (warp_idx == 0 && cute::elect_one_sync()) {
          if constexpr (SplitK) {
            tma_out.tma_reduce_add_async(
                output_smem.base_ptr,
                {m_tile * OUTPUT_ATOM_SIZE, n_tile * MMA_N});
          } else {
            tma_out.tma_store_async(
                output_smem.base_ptr,
                {m_tile * OUTPUT_ATOM_SIZE, n_tile * MMA_N});
          }

          cute::tma_store_arrive();
          cute::tma_store_wait<NUM_C_STAGE - 1>();
        }

        ++num_tiles_executed;
      }
    }

    if (warp_idx == 0 && cute::elect_one_sync()) {
      cute::tma_store_wait<0>();
    }
  }

  __syncthreads();

  if (warp_idx == 0) {
    tmem_allocator.free(shared_storage.tmem_base_ptr, num_tmem_columns);
  }
}

} // namespace kernel
