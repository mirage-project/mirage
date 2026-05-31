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

#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>

#include "../common/dmem_layout.cuh"
#include "../common/worker_config.h"
#include "../hopper/barrier.cuh"
#include "../hopper/smem_layout_tma.cuh"
#include "../hopper/tma.cuh"
#include "storage.cuh"

// mHC K1 (linear half): tcgen05 + TMA + TMEM bf16 GEMM, MPK-fusable.
//
// Adapted from linear_sm100_mpk.cuh with these deltas for the mHC shape regime:
//   - One CTA per batch tile (n_tile). The n_tile loop is replaced by reading
//     blockIdx.x. Caller launches gridDim.x = ceil_div(BATCH_SIZE, MMA_N) so
//     all SMs are busy in one launch (vs the original single-CTA-serial-tiles
//     design that left ~149 SMs idle on B200).
//   - No bias path (mHC's K2 task handles affine + activation downstream).
//   - No SplitK path.
//   - m_tile is always 0 (mix_hc <= MMA_M=128 with output pad to 128).
//   - Output is bf16 [BATCH_SIZE, MMA_M] with the user's first n_actual cols
//     valid; caller slices the rest.

namespace kernel {

template <typename T_,
          typename TMA_A,
          typename TMA_B,
          typename TMA_OUT,
          int MMA_M,
          int MMA_N,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int NUM_AB_STAGE = 8,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE = 4>
__device__ __noinline__ void
    mHC_linear_task_impl(const TMA_A &tma_a,
                         const TMA_B &tma_b,
                         const TMA_OUT &tma_out,
                         // MPK-mode override: if >=0, run a single iteration
                         // at this fixed n_tile (caller must ensure tma_b
                         // already targets the correct rows). Standalone
                         // callers pass -1 for normal grid-stride behavior.
                         int n_tile_override = -1) {
  int warp_idx = cutlass::canonical_warp_idx_sync();

  // Grid-strided over n_tiles: each CTA processes (num_n_tiles / gridDim.x)
  // batch tiles in sequence, reusing TMEM and barrier rings across them.
  constexpr int NUM_N_TILES = BATCH_SIZE / MMA_N;
  static_assert(BATCH_SIZE % MMA_N == 0,
                "BATCH_SIZE must be a multiple of MMA_N");

  // MMA grid coord: peer-CTA = 0, m/n/k unspecified (we'll explicitly pick the
  // batch tile via gB partitioning below).
  auto mma_coord_vmnk = cute::make_coord(0, cute::_, cute::_, cute::_);

  constexpr int num_tmem_columns = MMA_N * NUM_ACC_STAGE;

  cute::TiledMMA tiled_mma =
      cute::make_tiled_mma(cute::SM100_MMA_F16BF16_SS<T_,
                                                      T_,
                                                      float,
                                                      MMA_M,
                                                      MMA_N,
                                                      cute::UMMA::Major::K,
                                                      cute::UMMA::Major::K>{});
  auto bM = cute::tile_size<0>(tiled_mma);
  auto bN = cute::tile_size<1>(tiled_mma);
  auto bK = cute::tile_size<2>(tiled_mma) * cute::Int<4>{};
  auto mma_tiler = cute::make_shape(bM, bN, bK);

  auto mma_coord = cute::select<1, 2, 3>(mma_coord_vmnk);
  auto cd_tiler = cute::make_shape(bN, bM, bK);

  cute::Tensor mA = cute::make_coord_tensor(
      cute::make_layout(cute::make_shape(OUTPUT_SIZE, REDUCTION_SIZE),
                        cute::make_stride(cute::E<1>{}, cute::E<0>{})));
  cute::Tensor mB = cute::make_coord_tensor(
      cute::make_layout(cute::make_shape(BATCH_SIZE, REDUCTION_SIZE),
                        cute::make_stride(cute::E<1>{}, cute::E<0>{})));
  cute::Tensor mC = cute::make_coord_tensor(
      cute::make_layout(cute::make_shape(BATCH_SIZE, OUTPUT_SIZE),
                        cute::make_stride(cute::E<1>{}, cute::E<0>{})));

  cute::Tensor gA = cute::local_tile(
      mA, mma_tiler, mma_coord, cute::Step<cute::_1, cute::X, cute::_1>{});
  cute::Tensor gB = cute::local_tile(
      mB, mma_tiler, mma_coord, cute::Step<cute::X, cute::_1, cute::_1>{});
  cute::Tensor gC = cute::local_tile(
      mC, cd_tiler, mma_coord, cute::Step<cute::_1, cute::_1, cute::X>{});

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

  cute::Tensor tCsA = shared_storage.tensor_sA();
  cute::Tensor tCsB = shared_storage.tensor_sB();
  cute::Tensor sC_epi = shared_storage.tensor_sC();

  auto mma_v = cute::get<0>(mma_coord_vmnk);
  cute::ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
  cute::Tensor tCgA = cta_mma.partition_A(gA);
  cute::Tensor tCgB = cta_mma.partition_B(gB);

  // TMA transaction bytes (clamp BATCH dim per tile).
  constexpr int kClampedBN = (BATCH_SIZE < MMA_N) ? BATCH_SIZE : MMA_N;
  int tma_transaction_bytes =
      sizeof(T_) * kClampedBN * cute::size<2>(mma_tiler) +
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

  using InputSmem = smem_tma<T_, B, M, S, MMA_N, INPUT_TMA_TILE_SIZE, 1>;
  InputSmem input_smem(shared_input);
  using WeightSmem =
      smem_tma<T_, B, M, S, OUTPUT_ATOM_SIZE, WEIGHT_TMA_TILE_SIZE, 1>;
  WeightSmem input_weight_smem(shared_weight);
  using OutputSmem = smem_tma<T_, 0, M, S, MMA_N, OUTPUT_ATOM_SIZE, 1>;
  OutputSmem mm_output_smem(mm_output);

  cute::Tensor tCrA = cta_mma.make_fragment_A(tCsA);
  cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
  auto acc_shape =
      cute::partition_shape_C(tiled_mma,
                              cute::make_shape(cute::size<0>(mma_tiler),
                                               cute::size<1>(mma_tiler),
                                               cute::Int<NUM_ACC_STAGE>{}));
  auto tCtAcc = tiled_mma.make_fragment_C(acc_shape);

  cutlass::arch::fence_barrier_init();
  __syncthreads();

  int k_tile_count = cute::size<4>(tCgA);

  // Single m_tile (mix_hc <= MMA_M with output pad), single n_tile per CTA.
  constexpr int kNumMTiles = OUTPUT_SIZE / MMA_M;
  static_assert(OUTPUT_SIZE % MMA_M == 0,
                "OUTPUT_SIZE must be a multiple of MMA_M");
  static_assert(kNumMTiles == 1, "mHC_linear assumes single m_tile per CTA");

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  __syncthreads();

  if (warp_idx == 5) {
    // ---- TMA producer warp ----
    int total_k_tile_count = 0;
    int const _nt_start =
        (n_tile_override >= 0) ? n_tile_override : (int)blockIdx.x;
    int const _nt_step = (n_tile_override >= 0) ? NUM_N_TILES : (int)gridDim.x;
    int const _nt_end =
        (n_tile_override >= 0) ? (n_tile_override + 1) : NUM_N_TILES;
    for (int n_tile = _nt_start; n_tile < _nt_end; n_tile += _nt_step) {
      int num_prev_k_blk = total_k_tile_count;
      total_k_tile_count += k_tile_count;

      int tma_wr_k_tile = 0;
      int smem_wr_buffer = (num_prev_k_blk + tma_wr_k_tile) % NUM_AB_STAGE;
      int tma_wr_ab_empty_phase =
          (num_prev_k_blk + tma_wr_k_tile) / NUM_AB_STAGE % 2 ^ 1;

      bool peek_ab_empty_status =
          try_wait_barrier(shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                           tma_wr_ab_empty_phase);

      for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        int tma_wr_k_tile_next = tma_wr_k_tile + 1;
        int smem_wr_buffer_next =
            (num_prev_k_blk + tma_wr_k_tile_next) % NUM_AB_STAGE;
        int tma_wr_ab_empty_phase_next = smem_wr_buffer_next == 0
                                             ? tma_wr_ab_empty_phase ^ 1
                                             : tma_wr_ab_empty_phase;

        if (!peek_ab_empty_status) {
          cute::wait_barrier(shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                             tma_wr_ab_empty_phase);
        }

        if (cute::elect_one_sync()) {
          int tma_coords_A[2] = {k_tile * TILE_SIZE, /*m_tile=*/0};
          int tma_coords_B[2] = {k_tile * TILE_SIZE, n_tile * MMA_N};
          input_weight_smem.set_ptr(
              shared_weight + smem_wr_buffer * OUTPUT_ATOM_SIZE * TILE_SIZE);
          input_smem.set_ptr(shared_input + smem_wr_buffer * MMA_N * TILE_SIZE);
          cute::set_barrier_transaction_bytes(
              shared_storage.ab_full_mbar_ptr[smem_wr_buffer],
              tma_transaction_bytes);
          tma_a.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer],
                             input_weight_smem.base_ptr,
                             tma_coords_A);
          tma_b.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer],
                             input_smem.base_ptr,
                             tma_coords_B);
        }

        if (tma_wr_k_tile_next < k_tile_count) {
          peek_ab_empty_status = try_wait_barrier(
              shared_storage.ab_empty_mbar_ptr[smem_wr_buffer_next],
              tma_wr_ab_empty_phase_next);
        }

        tma_wr_k_tile = tma_wr_k_tile_next;
        smem_wr_buffer = smem_wr_buffer_next;
        tma_wr_ab_empty_phase = tma_wr_ab_empty_phase_next;
      }
    }
  } else if (warp_idx == 4) {
    // ---- MMA consumer warp ----
    tmem_allocation_result_barrier.arrive_and_wait();
    tCtAcc.data() = shared_storage.tmem_base_ptr;

    int total_k_tile_count = 0;
    int num_tiles_executed = 0;
    int const _nt_start =
        (n_tile_override >= 0) ? n_tile_override : (int)blockIdx.x;
    int const _nt_step = (n_tile_override >= 0) ? NUM_N_TILES : (int)gridDim.x;
    int const _nt_end =
        (n_tile_override >= 0) ? (n_tile_override + 1) : NUM_N_TILES;
    for (int n_tile = _nt_start; n_tile < _nt_end; n_tile += _nt_step) {
      int acc_buf_idx = num_tiles_executed % NUM_ACC_STAGE;
      auto tCtAcc_Slice = tCtAcc(cute::_, cute::_, cute::_, acc_buf_idx);

      int num_prev_k_blk = total_k_tile_count;
      total_k_tile_count += k_tile_count;

      int mma_rd_k_tile = 0;
      int smem_rd_buffer = (num_prev_k_blk + mma_rd_k_tile) % NUM_AB_STAGE;
      int mma_rd_ab_full_phase =
          (num_prev_k_blk + mma_rd_k_tile) / NUM_AB_STAGE % 2;

      bool peek_ab_full_status =
          try_wait_barrier(shared_storage.ab_full_mbar_ptr[smem_rd_buffer],
                           mma_rd_ab_full_phase);

      int acc_empty_phase = num_tiles_executed / NUM_ACC_STAGE % 2 ^ 1;
      cute::wait_barrier(shared_storage.acc_empty_mbar_ptr[acc_buf_idx],
                         acc_empty_phase);

      tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;

      for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        int mma_rd_k_tile_next = mma_rd_k_tile + 1;
        int smem_rd_buffer_next =
            (num_prev_k_blk + mma_rd_k_tile_next) % NUM_AB_STAGE;
        int mma_rd_ab_full_phase_next = smem_rd_buffer_next == 0
                                            ? mma_rd_ab_full_phase ^ 1
                                            : mma_rd_ab_full_phase;

        if (!peek_ab_full_status) {
          cute::wait_barrier(shared_storage.ab_full_mbar_ptr[smem_rd_buffer],
                             mma_rd_ab_full_phase);
        }

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
          peek_ab_full_status = try_wait_barrier(
              shared_storage.ab_full_mbar_ptr[smem_rd_buffer_next],
              mma_rd_ab_full_phase_next);
        }

        mma_rd_k_tile = mma_rd_k_tile_next;
        smem_rd_buffer = smem_rd_buffer_next;
        mma_rd_ab_full_phase = mma_rd_ab_full_phase_next;
      }

      cutlass::arch::umma_arrive(
          &shared_storage.acc_full_mbar_ptr[acc_buf_idx]);
      num_tiles_executed++;
    }
  } else if (warp_idx < 4) {
    // ---- Epilogue warps ----
    if (warp_idx == 0) {
      tmem_allocator.allocate(num_tmem_columns, &shared_storage.tmem_base_ptr);
    }
    tmem_allocation_result_barrier.arrive_and_wait();
    tCtAcc.data() = shared_storage.tmem_base_ptr;

    using AccType = typename decltype(tCtAcc)::value_type;
    using TypeC = T_;
    cutlass::NumericConverter<TypeC, AccType> converter;

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

    cute::Tensor tCrC =
        cute::make_tensor<TypeC>(cute::shape(tTR_rAcc(0, cute::_, 0, 0)));

    int num_tiles_executed = 0;
    int const _nt_start =
        (n_tile_override >= 0) ? n_tile_override : (int)blockIdx.x;
    int const _nt_step = (n_tile_override >= 0) ? NUM_N_TILES : (int)gridDim.x;
    int const _nt_end =
        (n_tile_override >= 0) ? (n_tile_override + 1) : NUM_N_TILES;
    for (int n_tile = _nt_start; n_tile < _nt_end; n_tile += _nt_step) {
      int acc_buf_idx = num_tiles_executed % NUM_ACC_STAGE;
      int acc_full_phase = num_tiles_executed / NUM_ACC_STAGE % 2;
      int c_smem_wr_buffer_idx = num_tiles_executed % NUM_C_STAGE;

      mm_output_smem.set_ptr(mm_output +
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

      cute::tma_store_fence();
      epilogue_wg_barrier.arrive_and_wait();

      if (warp_idx == 0 && cute::elect_one_sync()) {
        tma_out.tma_store_async(mm_output_smem.base_ptr,
                                {/*m_tile=*/0, n_tile * MMA_N});
        cute::tma_store_arrive();
        cute::tma_store_wait<NUM_C_STAGE - 1>();
      }
      num_tiles_executed++;
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
