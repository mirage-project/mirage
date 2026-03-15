#pragma once
#include <cstdio>
#include <iostream>

// Cutlass includes
#include <cutlass/half.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

// CuTe includes
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

namespace kernel {

// ============================================================================
// Helper: reconstructs CuTe MMA setup from template params.
// All compile-time / zero runtime cost. Used by each __noinline__ stage
// so that no CuTe tensors cross function boundaries (avoids LMEM spills).
// ============================================================================
template <typename T_, int MMA_M, int MMA_N,
          int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE,
          int NUM_AB_STAGE, int NUM_ACC_STAGE>
struct LinearSm100Setup {
  static constexpr int TILE_SIZE = 64;
  static constexpr int OUTPUT_ATOM_SIZE = 128;
  static constexpr int INPUT_TMA_TILE_SIZE = 64;
  static constexpr int WEIGHT_TMA_TILE_SIZE = 64;

  using InputSmem = smem_tma<T_, 3, 3, 3, MMA_N, INPUT_TMA_TILE_SIZE, 1>;
  using WeightSmem = smem_tma<T_, 3, 3, 3, OUTPUT_ATOM_SIZE, WEIGHT_TMA_TILE_SIZE, 1>;
  using OutputSmem = smem_tma<T_, 0, 3, 3, MMA_N, OUTPUT_ATOM_SIZE, 1>;

  // Everything below is __device__ __forceinline__ so it vanishes at the call site
  static __device__ __forceinline__ auto make_tiled_mma() {
    return cute::make_tiled_mma(
        cute::SM100_MMA_F16BF16_SS<T_, T_, float, MMA_M, MMA_N,
                                   cute::UMMA::Major::K,
                                   cute::UMMA::Major::K>{});
  }

  static __device__ __forceinline__ auto make_mma_tiler() {
    auto tiled_mma = make_tiled_mma();
    auto bM = cute::tile_size<0>(tiled_mma);
    auto bN = cute::tile_size<1>(tiled_mma);
    auto bK = cute::tile_size<2>(tiled_mma) * cute::Int<4>{};
    return cute::make_shape(bM, bN, bK);
  }

  static __device__ __forceinline__ auto make_mma_coord() {
    auto mma_coord_vmnk = cute::make_coord(0, cute::_, cute::_, cute::_);
    return cute::select<1, 2, 3>(mma_coord_vmnk);
  }

  static __device__ __forceinline__ auto make_cd_tiler() {
    auto mma_tiler = make_mma_tiler();
    return cute::make_shape(cute::size<1>(mma_tiler),
                            cute::size<0>(mma_tiler),
                            cute::size<2>(mma_tiler));
  }

  static __device__ __forceinline__ auto make_gA() {
    cute::Tensor mA = cute::make_coord_tensor(cute::make_layout(
        cute::make_shape(OUTPUT_SIZE, REDUCTION_SIZE),
        cute::make_stride(cute::E<1>{}, cute::E<0>{})));
    return cute::local_tile(mA, make_mma_tiler(), make_mma_coord(),
        cute::Step<cute::_1, cute::X, cute::_1>{});
  }

  static __device__ __forceinline__ auto make_gB() {
    cute::Tensor mB = cute::make_coord_tensor(cute::make_layout(
        cute::make_shape(BATCH_SIZE, REDUCTION_SIZE),
        cute::make_stride(cute::E<1>{}, cute::E<0>{})));
    return cute::local_tile(mB, make_mma_tiler(), make_mma_coord(),
        cute::Step<cute::X, cute::_1, cute::_1>{});
  }

  static __device__ __forceinline__ auto make_cta_mma() {
    auto tiled_mma = make_tiled_mma();
    return tiled_mma.get_slice(0);  // mma_v = 0
  }

  static __device__ __forceinline__ auto make_tCgA() {
    return make_cta_mma().partition_A(make_gA());
  }

  static __device__ __forceinline__ auto make_tCgB() {
    return make_cta_mma().partition_B(make_gB());
  }

  static __device__ __forceinline__ int get_k_tile_count() {
    return cute::size<4>(make_tCgA());
  }

  static __device__ __forceinline__ int get_tma_transaction_bytes() {
    auto mma_tiler = make_mma_tiler();
    return sizeof(T_) * cute::size<1>(mma_tiler) * cute::size<2>(mma_tiler) +
           sizeof(T_) * cute::size<0>(mma_tiler) * cute::size<2>(mma_tiler);
  }
};


// Use __noinline__ for static worker (needed for warp-specialized pipeline),
// __forceinline__ for dynamic worker (recovers ~4% from cross-stage optimization).
#ifdef MPK_STATIC_WORKER
#define LINEAR_STAGE_ATTR __device__ __noinline__
#else
#define LINEAR_STAGE_ATTR __device__ __forceinline__
#endif

// ── Loader stage (warp 5) ──────────────────────────────────────────────────
template <typename T_, int MMA_M, int MMA_N,
          int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE,
          int NUM_AB_STAGE, int NUM_ACC_STAGE,
          typename TMA_A, typename TMA_B, class SharedStorage>
LINEAR_STAGE_ATTR void
linear_sm100_loader_stage(SharedStorage &shared_storage,
                          const TMA_A &tma_a,
                          const TMA_B &tma_b) {

  using S = LinearSm100Setup<T_, MMA_M, MMA_N, BATCH_SIZE, OUTPUT_SIZE,
                              REDUCTION_SIZE, NUM_AB_STAGE, NUM_ACC_STAGE>;

  auto tCgA = S::make_tCgA();
  auto tCgB = S::make_tCgB();
  int k_tile_count = S::get_k_tile_count();
  int tma_transaction_bytes = S::get_tma_transaction_bytes();

  T_ *shared_weight = shared_storage.A.begin();
  T_ *shared_input = shared_storage.B.begin();
  Barrier *ab_full_mbar_ptr =
      reinterpret_cast<Barrier *>(shared_storage.ab_full_mbar_ptr);

  typename S::InputSmem input_smem(shared_input);
  typename S::WeightSmem input_weight_smem(shared_weight);

  // ── Loader loop (identical to original) ──
  int total_k_tile_count = 0;
  for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
    for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {

      int num_prev_k_blk = total_k_tile_count;
      total_k_tile_count += k_tile_count;

      int tma_wr_k_tile = 0;
      int smem_wr_buffer = (num_prev_k_blk + tma_wr_k_tile) % NUM_AB_STAGE;
      int tma_wr_ab_empty_phase =
          (num_prev_k_blk + tma_wr_k_tile) / NUM_AB_STAGE % 2 ^ 1;

      bool peek_ab_empty_status = kernel::try_wait_barrier(
          shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
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
          int tma_coords_A[2] = {k_tile * S::TILE_SIZE,
                                 m_tile * S::OUTPUT_ATOM_SIZE};
          int tma_coords_B[2] = {k_tile * S::TILE_SIZE, n_tile * MMA_N};
          input_weight_smem.set_ptr(
              shared_weight + smem_wr_buffer * S::OUTPUT_ATOM_SIZE * S::TILE_SIZE);
          input_smem.set_ptr(shared_input +
                             smem_wr_buffer * MMA_N * S::TILE_SIZE);
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
          peek_ab_empty_status = kernel::try_wait_barrier(
              shared_storage.ab_empty_mbar_ptr[smem_wr_buffer_next],
              tma_wr_ab_empty_phase_next);
        }

        tma_wr_k_tile = tma_wr_k_tile_next;
        smem_wr_buffer = smem_wr_buffer_next;
        tma_wr_ab_empty_phase = tma_wr_ab_empty_phase_next;

      } // end for k_tile
    } // end for n_tile
  }
}


// ── Mainloop stage (warp 4) ────────────────────────────────────────────────
template <typename T_, int MMA_M, int MMA_N,
          int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE,
          int NUM_AB_STAGE, int NUM_ACC_STAGE,
          class SharedStorage>
LINEAR_STAGE_ATTR void
linear_sm100_mainloop_stage(SharedStorage &shared_storage) {

  using S = LinearSm100Setup<T_, MMA_M, MMA_N, BATCH_SIZE, OUTPUT_SIZE,
                              REDUCTION_SIZE, NUM_AB_STAGE, NUM_ACC_STAGE>;

  auto tiled_mma = S::make_tiled_mma();
  auto mma_tiler = S::make_mma_tiler();
  auto tCgA = S::make_tCgA();
  auto tCgB = S::make_tCgB();

  // SMEM tensor views + MMA fragments
  cute::Tensor tCsA = shared_storage.tensor_sA();
  cute::Tensor tCsB = shared_storage.tensor_sB();
  auto cta_mma = S::make_cta_mma();
  cute::Tensor tCrA = cta_mma.make_fragment_A(tCsA);
  cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
  auto acc_shape = cute::partition_shape_C(
      tiled_mma,
      cute::make_shape(cute::size<0>(mma_tiler),
                       cute::size<1>(mma_tiler),
                       cute::Int<NUM_ACC_STAGE>{}));
  auto tCtAcc = tiled_mma.make_fragment_C(acc_shape);

  int k_tile_count = S::get_k_tile_count();

  // Wait for TMEM allocation to complete
  cutlass::arch::NamedBarrier tmem_allocation_result_barrier(
      32 + 128, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
  tmem_allocation_result_barrier.arrive_and_wait();
  tCtAcc.data() = shared_storage.tmem_base_ptr;

  // ── Mainloop (identical to original) ──
  int total_k_tile_count = 0;
  int num_tiles_executed = 0;
  for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
    for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {

      int acc_buf_idx = num_tiles_executed % NUM_ACC_STAGE;
      auto tCtAcc_Slice = tCtAcc(cute::_, cute::_, cute::_, acc_buf_idx);

      int num_prev_k_blk = total_k_tile_count;
      total_k_tile_count += k_tile_count;

      int mma_rd_k_tile = 0;
      int smem_rd_buffer = (num_prev_k_blk + mma_rd_k_tile) % NUM_AB_STAGE;
      int mma_rd_ab_full_phase =
          (num_prev_k_blk + mma_rd_k_tile) / NUM_AB_STAGE % 2;

      bool peek_ab_full_status = kernel::try_wait_barrier(
          shared_storage.ab_full_mbar_ptr[smem_rd_buffer],
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
          peek_ab_full_status = kernel::try_wait_barrier(
              shared_storage.ab_full_mbar_ptr[smem_rd_buffer_next],
              mma_rd_ab_full_phase_next);
        }

        mma_rd_k_tile = mma_rd_k_tile_next;
        smem_rd_buffer = smem_rd_buffer_next;
        mma_rd_ab_full_phase = mma_rd_ab_full_phase_next;

      } // end for k_tile

      cutlass::arch::umma_arrive(
          &shared_storage.acc_full_mbar_ptr[acc_buf_idx]);
      num_tiles_executed++;

    } // end for n_tile
  }
}


// ── Storer stage (warps 0-3) ───────────────────────────────────────────────
template <typename T_, int MMA_M, int MMA_N,
          int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE,
          bool NOBIAS, bool SplitK,
          int NUM_AB_STAGE, int NUM_ACC_STAGE, int NUM_C_STAGE,
          class BiasTensor, typename TMA_OUT, class SharedStorage>
LINEAR_STAGE_ATTR void
linear_sm100_storer_stage(SharedStorage &shared_storage,
                          BiasTensor mBias,
                          const TMA_OUT &tma_out) {

  using S = LinearSm100Setup<T_, MMA_M, MMA_N, BATCH_SIZE, OUTPUT_SIZE,
                              REDUCTION_SIZE, NUM_AB_STAGE, NUM_ACC_STAGE>;

  int warp_idx = cutlass::canonical_warp_idx_sync();

  auto tiled_mma = S::make_tiled_mma();
  auto mma_tiler = S::make_mma_tiler();
  auto tCgA = S::make_tCgA();
  auto tCgB = S::make_tCgB();

  // gBias from mBias
  auto cd_tiler = S::make_cd_tiler();
  auto mma_coord = S::make_mma_coord();
  cute::Tensor gBias = cute::local_tile(
      mBias, cd_tiler, mma_coord,
      cute::Step<cute::_1, cute::_1, cute::X>{});

  // SMEM views
  cute::Tensor sC_epi = shared_storage.tensor_sC();
  T_ *mm_output = shared_storage.C.begin();
  typename S::OutputSmem mm_output_smem(mm_output);

  // Accumulator fragment
  auto acc_shape = cute::partition_shape_C(
      tiled_mma,
      cute::make_shape(cute::size<0>(mma_tiler),
                       cute::size<1>(mma_tiler),
                       cute::Int<NUM_ACC_STAGE>{}));
  auto tCtAcc = tiled_mma.make_fragment_C(acc_shape);

  constexpr int num_tmem_columns = MMA_N * NUM_ACC_STAGE;

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  cutlass::arch::NamedBarrier tmem_allocation_result_barrier(
      32 + 128, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
  cutlass::arch::NamedBarrier epilogue_wg_barrier(
      128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

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
                           tCtAcc(cute::_, cute::_, cute::_, 0));
  auto thr_copy_t2r = tiled_copy_t2r.get_slice(threadIdx.x);
  auto tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc);

  auto tCgC_fake = cute::make_tensor<TypeC>(cute::shape(
      tCtAcc(cute::_, cute::_, cute::_, 0)));
  auto tTR_rAcc_fake = thr_copy_t2r.partition_D(tCgC_fake);
  auto tTR_rAcc = cute::make_tensor<AccType>(cute::shape(tTR_rAcc_fake));

  // ── Storer loop (identical to original) ──
  int num_tiles_executed = 0;
  for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
    for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {
      int acc_buf_idx = num_tiles_executed % NUM_ACC_STAGE;
      int acc_full_phase = num_tiles_executed / NUM_ACC_STAGE % 2;
      int c_smem_wr_buffer_idx = num_tiles_executed % NUM_C_STAGE;

      auto tCgBias = gBias(cute::_, cute::_, n_tile, m_tile);
      auto tCrBiasTypeBias = cute::make_tensor<TypeBias>(
          cute::shape(tTR_rAcc(0, cute::_, 0, 0)));
      auto tCrBiasTypeAcc =
          cute::make_tensor<AccType>(cute::shape(tCrBiasTypeBias));
      auto tCrC =
          cute::make_tensor<TypeC>(cute::shape(tCrBiasTypeBias));

      if constexpr (!NOBIAS) {
        cute::copy(tCgBias(cute::_, threadIdx.x), tCrBiasTypeBias);

        CUTE_UNROLL
        for (int i = 0; i < tCrBiasTypeBias.size(); i++) {
          tCrBiasTypeAcc[i] = converterBias(tCrBiasTypeBias[i]);
        }
      }

      mm_output_smem.set_ptr(mm_output +
                             c_smem_wr_buffer_idx * MMA_N * S::OUTPUT_ATOM_SIZE);

      cute::wait_barrier(shared_storage.acc_full_mbar_ptr[acc_buf_idx],
                         acc_full_phase);
      // T2R copy
      cute::copy(tiled_copy_t2r,
                 tTR_tAcc(cute::_, cute::_, cute::_, cute::_, acc_buf_idx),
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
        tCrC[i] = converter(tTR_rAcc[i]);
      }

      // R2S copy
      auto sC_epi_slice =
          cute::flatten(sC_epi(cute::_, 0, 0, c_smem_wr_buffer_idx));
      cute::copy(tCrC, sC_epi_slice(cute::_, threadIdx.x));

      // S2G TMA
      cute::tma_store_fence();
      epilogue_wg_barrier.arrive_and_wait();

      if (warp_idx == 0 && cute::elect_one_sync()) {
        if constexpr (SplitK) {
          tma_out.tma_reduce_add_async(
              mm_output_smem.base_ptr,
              {m_tile * S::OUTPUT_ATOM_SIZE, n_tile * MMA_N});
        } else {
          tma_out.tma_store_async(
              mm_output_smem.base_ptr,
              {m_tile * S::OUTPUT_ATOM_SIZE, n_tile * MMA_N});
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


// ── Entry point (unchanged interface) ──────────────────────────────────────
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
    linear_sm100_mpk_task_impl(const TMA_A &tma_a,
                               const TMA_B &tma_b,
                               BiasTensor mBias,
                               const TMA_OUT &tma_out) {
  int warp_idx = cutlass::canonical_warp_idx_sync();

  using S = LinearSm100Setup<T_, MMA_M, MMA_N, BATCH_SIZE, OUTPUT_SIZE,
                              REDUCTION_SIZE, NUM_AB_STAGE, NUM_ACC_STAGE>;

  // ── Compute SharedStorage type (all constexpr) ──
  auto tiled_mma = S::make_tiled_mma();
  auto mma_tiler = S::make_mma_tiler();

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
                       cute::Int<1>{}, cute::Int<1>{},
                       cute::Int<NUM_C_STAGE>{});

  auto sA_layout = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<T_>{}, mma_shape_A);
  auto sB_layout = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<T_>{}, mma_shape_B);
  auto sC_layout_fake = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_INTER_Atom<T_>{}, mma_shape_C);
  auto sC_shape = cute::make_shape(
      cute::make_shape(cute::Int<MMA_N>{}, cute::Int<MMA_M>{}),
      cute::Int<1>{}, cute::Int<1>{},
      cute::make_shape(cute::Int<1>{}, cute::Int<NUM_C_STAGE>{}));
  auto sC_stride = cute::make_stride(
      cute::make_stride(cute::Int<MMA_M>{}, cute::Int<1>{}),
      cute::Int<0>{}, cute::Int<0>{},
      cute::make_stride(cute::Int<0>{}, cute::Int<MMA_M * MMA_N>{}));
  auto sC_layout = cute::composition(sC_layout_fake.layout_a(),
                                     sC_layout_fake.offset(),
                                     cute::make_layout(sC_shape, sC_stride));

  using SharedStorage = PipedSharedStorage<T_, T_, T_,
                                           decltype(sA_layout),
                                           decltype(sB_layout),
                                           decltype(sC_layout),
                                           NUM_AB_STAGE, NUM_ACC_STAGE>;

  extern __shared__ char shared_memory[];
  uintptr_t aligned_smem =
      (reinterpret_cast<uintptr_t>(shared_memory) + 127) / 128 * 128;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(aligned_smem);

  // Initialize the barriers in shared memory
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

  cutlass::arch::fence_barrier_init();
  MPK_CONSUMER_SYNC();

  constexpr int num_tmem_columns = MMA_N * NUM_ACC_STAGE;

  MPK_CONSUMER_SYNC(); // Wait for barrier init + TMEM alloc

  // ── Dispatch to warp-specialized stages ──────────────────────────────────
  // Each stage reconstructs its own CuTe tensors internally.
  // Only SharedStorage ref + TMA descriptors cross the __noinline__ boundary.
  if (warp_idx == 5) {
    linear_sm100_loader_stage<T_, MMA_M, MMA_N, BATCH_SIZE, OUTPUT_SIZE,
                              REDUCTION_SIZE, NUM_AB_STAGE, NUM_ACC_STAGE>(
        shared_storage, tma_a, tma_b);
  } else if (warp_idx == 4) {
    linear_sm100_mainloop_stage<T_, MMA_M, MMA_N, BATCH_SIZE, OUTPUT_SIZE,
                                REDUCTION_SIZE, NUM_AB_STAGE, NUM_ACC_STAGE>(
        shared_storage);
  } else if (warp_idx < 4) {
    linear_sm100_storer_stage<T_, MMA_M, MMA_N, BATCH_SIZE, OUTPUT_SIZE,
                              REDUCTION_SIZE, NOBIAS, SplitK,
                              NUM_AB_STAGE, NUM_ACC_STAGE, NUM_C_STAGE>(
        shared_storage, mBias, tma_out);
  }
  MPK_CONSUMER_SYNC();

  // Release TMEM after all warps have synced
  if (warp_idx == 0) {
    using TmemAllocator = cute::TMEM::Allocator1Sm;
    TmemAllocator tmem_allocator{};
    tmem_allocator.free(shared_storage.tmem_base_ptr, num_tmem_columns);
  }

} // end linear_sm100_mpk_task_impl

} // namespace kernel
