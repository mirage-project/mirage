#pragma once
#include <cstdio>
#include <iostream>

// Use Thrust to handle host/device allocations
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Cutlass includes
#include <cutlass/half.h> // F16 data type
// #include <cutlass/util/print_error.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

// CuTe includes
#include <cute/algorithm/cooperative_copy.hpp> // Auto vectorized copy operation
#include <cute/arch/cluster_sm90.hpp> // CuTe functions for querying the details of cluster launched
#include <cute/arch/tmem_allocator_sm100.hpp> // TMEM allocator for SM100
#include <cute/numeric/integral_constant.hpp> // Compile time in constants such as _1, _256 etc.
#include <cute/tensor.hpp>                    // CuTe tensor implementation
// using namespace cute;

#include "utils.cuh"

namespace kernel {

template <class SharedStorage,
          class ATensor,
          class BTensor,
          class BiasTensor,
          class CTensor,
          class MmaTiler_MNK,
          class EpiTiler_MN,
          class TiledMMA,
          class TmaAtomA,
          class TmaAtomB,
          class TmaAtomC,
          int Num_AB_Stage,
          int Num_ACC_Stage,
          int Num_C_Stage,
          bool NoBias>
__device__ __noinline__ void linear_kernel_ws_sm100(
    ATensor mA,             // (Gemm_M, Gemm_K)
    BTensor mB,             // (Gemm_N, Gemm_K)
    BiasTensor mBias,       // (Gemm_N, Gemm_M)
    CTensor mC,             // (Gemm_N, Gemm_M)
    MmaTiler_MNK mma_tiler, // <MmaTile_M, MmaTile_N, MmaTile_K>
    EpiTiler_MN epi_tiler,  // <EpiTile_M, EpiTile_N>
    TiledMMA tiled_mma,     // <    Mma_M,     Mma_N,     Mma_K>
    TmaAtomA const *tma_atom_A,
    TmaAtomB const *tma_atom_B,
    TmaAtomC const *tma_atom_C,
    int const num_tmem_columns) {
  int warp_idx = cutlass::canonical_warp_idx_sync();

  // Construct the MMA grid coordinate from the CTA grid coordinate
  auto mma_coord_vmnk = cute::make_coord(0,        // Peer CTA coordinate
                                         cute::_,  //    MMA-M coordinate
                                         cute::_,  //    MMA-N coordinate
                                         cute::_); //    MMA-K coordinate

  // Partition the GMEM tensors with the mma_tiler and mma_coord to get the
  // slices processed
  //   by this mma tile.
  // CuTe provides local_tile partitioning function. local_tile accepts 4
  // parameters:
  //   * Tensor to partition
  //   * Tiler to use for partitioning
  //   * Coordinate to use for slicing the partitioned tensor
  //   * Projection to ignore unwanted modes of the Tiler and Coordinate
  auto mma_coord = cute::select<1, 2, 3>(mma_coord_vmnk);
  auto cd_tiler = cute::make_shape(
      cute::size<1>(mma_tiler),
      cute::size<0>(mma_tiler),
      cute::size<2>(mma_tiler)); // (MmaTile_N, MmaTile_M, MmaTile_K)
  cute::Tensor gA = cute::local_tile(
      mA,
      mma_tiler,
      mma_coord,
      cute::Step<cute::_1, cute::X, cute::_1>{}); // (MmaTile_M, MmaTile_K,
                                                  // Tiles_K)
  cute::Tensor gB = cute::local_tile(
      mB,
      mma_tiler,
      mma_coord,
      cute::Step<cute::X, cute::_1, cute::_1>{}); // (MmaTile_N, MmaTile_K,
                                                  // Tiles_K)
  cute::Tensor gBias = cute::local_tile(
      mBias,
      cd_tiler,
      mma_coord,
      cute::Step<cute::_1, cute::_1, cute::X>{}); // (MmaTile_M, MmaTile_N)
  cute::Tensor gC = cute::local_tile(
      mC,
      cd_tiler,
      mma_coord,
      cute::Step<cute::_1, cute::_1, cute::X>{}); // (MmaTile_M, MmaTile_N)

  // if (cute::thread0()) {
  //   cute::print("mA:\t"); cute::print(mA); cute::print("\n");   // mA:
  //   ArithTuple(_0,_0) o (output_size, reduction_size):(_1@1,_1@0)
  //   cute::print("mB:\t"); cute::print(mB); cute::print("\n");   // mB:
  //   ArithTuple(_0,_0) o (batch_size, reduction_size):(_1@1,_1@0)
  //   cute::print("mBias:\t"); cute::print(mBias); cute::print("\n");   //
  //   mBias:   gmem_ptr[32b](GMEM_ADDR_C) o (512,1024):(1024,_1)
  //   cute::print("mC:\t"); cute::print(mC); cute::print("\n");   // mC:
  //   gmem_ptr[32b](GMEM_ADDR_D) o (512,1024):(1024,_1)

  //   cute::print("gA:\t"); cute::print(gA); cute::print("\n");   // gA:
  //   ArithTuple(_0,0) o (_128,_64,4):(_1@1,_1@0,_64@0) cute::print("gB:\t");
  //   cute::print(gB); cute::print("\n");   // gB:   ArithTuple(_0,0) o
  //   (_256,_64,4):(_1@1,_1@0,_64@0) cute::print("gBias:\t");
  //   cute::print(gBias); cute::print("\n");   // gBias:
  //   gmem_ptr[32b](GMEM_ADDR_C + offset_for_mma_tile) o (_128,_256):(256,_1)
  //   cute::print("gC:\t"); cute::print(gC); cute::print("\n");   // gC:
  //   gmem_ptr[32b](GMEM_ADDR_D + offset_for_mma_tile) o (_128,_256):(256,_1)
  // } __syncthreads();

  // The SMEM tensors

  // Allocate SMEM
  extern __shared__ char shared_memory[];
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  // Initialize the barriers in shared memory
  if (warp_idx == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        Num_AB_Stage>(shared_storage.ab_full_mbar_ptr, /* arrival count */ 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        Num_AB_Stage>(shared_storage.ab_empty_mbar_ptr, /* arrival count */ 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        Num_ACC_Stage>(shared_storage.acc_full_mbar_ptr, /* arrival count */ 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        Num_ACC_Stage>(shared_storage.acc_empty_mbar_ptr,
                       /* arrival count */ 4);
  }
  // Sync tmem allocation status between MMA and epilogue warps within CTA
  // 32 threads (mma) + 128 threads (epilog) to sync
  cutlass::arch::NamedBarrier tmem_allocation_result_barrier(
      32 + 128, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
  cutlass::arch::NamedBarrier epilogue_wg_barrier(128, /*bar-id*/ 6);
  cutlass::arch::NamedBarrier mma_wg_barrier(32, /*bar-id*/ 7);

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

  cute::Tensor tCgC_epi = cute::tiled_divide(
      gC, epi_tiler); // (EpiTile_M, EpiTile_N, Tiles_M, Tiles_N)

  // if (cute::thread0()) {
  //   cute::print("tCgA:\t"); cute::print(tCgA); cute::print("\n");  // tCgA:
  //   ArithTuple(_0,0) o ((_128,_16),_1,_4,64):((_1@1,_1@0),_0,_16@0,_64@0)
  //   cute::print("tCgB:\t"); cute::print(tCgB); cute::print("\n");  // tCgB:
  //   ArithTuple(_0,0) o ((_32,_16),_1,_4,64):((_1@1,_1@0),_0,_16@0,_64@0)
  //   cute::print("tCgC:\t"); cute::print(tCgC); cute::print("\n");  // tCgC:
  //   ArithTuple(_0,_0) o
  //   ((_32,_128),_1,_1,1,8):((_1@1,_1@0),_0,_0,_32@1,_128@0)
  //   cute::print("tCsA:\t"); cute::print(tCsA); cute::print("\n");
  //   cute::print("tCsB:\t"); cute::print(tCsB); cute::print("\n");
  //   cute::print("sC_epi:\t"); cute::print(sC_epi); cute::print("\n");
  // } __syncthreads();

  // // TMA Setup
  // //

  auto [tAgA, tAsA] = cute::tma_partition(*tma_atom_A,
                                          cute::Int<0>{},
                                          cute::Layout<cute::_1>{},
                                          cute::group_modes<0, 3>(tCsA),
                                          cute::group_modes<0, 3>(tCgA));

  auto [tBgB, tBsB] = cute::tma_partition(*tma_atom_B,
                                          cute::Int<0>{},
                                          cute::Layout<cute::_1>{},
                                          cute::group_modes<0, 3>(tCsB),
                                          cute::group_modes<0, 3>(tCgB));

  auto [tCgC, tCsC] = cute::tma_partition(*tma_atom_C,
                                          cute::Int<0>{},
                                          cute::Layout<cute::_1>{},
                                          cute::group_modes<0, 3>(sC_epi),
                                          cute::group_modes<0, 3>(tCgC_epi));

  // Calculate total bytes that TMA will transfer each tile to track completion
  int tma_transaction_bytes = sizeof(cute::make_tensor_like(tAsA(cute::_, 0))) +
                              sizeof(cute::make_tensor_like(tBsB(cute::_, 0)));
  // if (cute::thread0()) {
  //   cute::print("tAgA:\t"); cute::print(tAgA); cute::print("\n");  // tAgA:
  //   ArithTuple(_0,0) o (((_64,_128),_1),4):(((_1@0,_1@1),_0),_64@0)
  //   cute::print("tAsA:\t"); cute::print(tAsA); cute::print("\n");  // tAsA:
  //   Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_A) o ((_8192,_1)):((_1,_0))
  //   cute::print("tBgB:\t"); cute::print(tBgB); cute::print("\n");  // tBgB:
  //   ArithTuple(_0,0) o (((_64,_256),_1),4):(((_1@0,_1@1),_0),_64@0)
  //   cute::print("tBsB:\t"); cute::print(tBsB); cute::print("\n");  // tBsB:
  //   Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_B) o ((_16384,_1)):((_1,_0))
  //   cute::print("tCgC:\t"); cute::print(tCgC); cute::print("\n");  // tCgC:
  //   ArithTuple(_0,0) o (((_64,_256),_1),4):(((_1@0,_1@1),_0),_64@0)
  //   cute::print("tCsC:\t"); cute::print(tCsC); cute::print("\n");  // tCsC:
  //   Sw<3,4,3>_smem_ptr[16b](SMEM_ADDR_C) o ((_16384,_1)):((_1,_0))
  //   printf("tma_transaction_bytes: %d\n", tma_transaction_bytes);
  // } __syncthreads();

  // TODO(Zhihao): continue here

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
                       cute::Int<Num_ACC_Stage>{})); // (MmaC, NumMma_M,
                                                     // NumMma_N, NumAcc_Stage)
  // (MMA, MMA_M, MMA_N, STAGE)
  auto tCtAcc = tiled_mma.make_fragment_C(acc_shape);
  // // TMEM Allocation
  // // On SM100 architecture, accumulators are stored exclusively in tensor
  // memory (TMEM).
  // // ThrMma's make_fragment_C() creates a TMEM tensor with the appropriate
  // layout for the accumulator. cute::Tensor tCtAcc =
  // cta_mma.make_fragment_C(tCgC(cute::_, cute::_, cute::_, 0, 0));    //
  // (MmaC, NumMma_M, NumMma_N)

  cutlass::arch::fence_barrier_init();
  __syncthreads();

  // if (cute::thread0()) {
  //   cute::print("tCrA:\t"); cute::print(tCrA); cute::print("\n");     //
  //   tCrA:   UMMA::DescriptorIterator o (_1,_1,_4):(_0,_0,_2)
  //   cute::print("tCrB:\t"); cute::print(tCrB); cute::print("\n");     //
  //   tCrB:   UMMA::DescriptorIterator o (_1,_1,_4):(_0,_0,_2)
  //   cute::print("tCtAcc:\t"); cute::print(tCtAcc); cute::print("\n"); //
  //   tCtAcc: tmem_[32b](TMEM_ADDR) o ((_128,_256),_1,_1):((_65536,_1),_0,_0)
  // } __syncthreads();

  int k_tile_count = cute::size<4>(tCgA);

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  __syncthreads(); // Wait for all threads until warp0 allocates TMEM
  // tCtAcc.data() = shared_storage.tmem_base_ptr;

  if (warp_idx == 5) {
    // TMA warp (1)
    int total_k_tile_count = 0;
    for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
      for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {
        auto tAgA_Slice = tAgA(cute::_, m_tile, cute::_);
        auto tBgB_Slice = tBgB(cute::_, n_tile, cute::_);

        int num_prev_k_blk = total_k_tile_count;
        total_k_tile_count += k_tile_count;

        int tma_wr_k_tile = 0;
        int smem_wr_buffer = (num_prev_k_blk + tma_wr_k_tile) % Num_AB_Stage;
        int tma_wr_ab_empty_phase =
            (num_prev_k_blk + tma_wr_k_tile) / Num_AB_Stage % 2 ^ 1;

        bool peek_ab_empty_status = kernel::try_wait_barrier(
            shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
            tma_wr_ab_empty_phase);

        CUTE_UNROLL
        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {

          int tma_wr_k_tile_next = tma_wr_k_tile + 1;
          int smem_wr_buffer_next =
              (num_prev_k_blk + tma_wr_k_tile_next) % Num_AB_Stage;
          int tma_wr_ab_empty_phase_next = smem_wr_buffer_next == 0
                                               ? tma_wr_ab_empty_phase ^ 1
                                               : tma_wr_ab_empty_phase;

          // Wait for an empty buffer
          if (!peek_ab_empty_status) {
            cute::wait_barrier(shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                               tma_wr_ab_empty_phase);
          }

          if (cute::elect_one_sync()) {
            // printf("loading ab tile m=%d n=%d k=%d into smem_buf=%d\n",
            // m_tile, n_tile, k_tile, smem_wr_buffer);
            cute::set_barrier_transaction_bytes(
                shared_storage.ab_full_mbar_ptr[smem_wr_buffer],
                tma_transaction_bytes);
            // Start TMA to load A and B tiles into the selected SMEM buffer
            copy(tma_atom_A->with(
                     shared_storage.ab_full_mbar_ptr[smem_wr_buffer]),
                 tAgA_Slice(cute::_, k_tile),
                 tAsA(cute::_, smem_wr_buffer));
            copy(tma_atom_B->with(
                     shared_storage.ab_full_mbar_ptr[smem_wr_buffer]),
                 tBgB_Slice(cute::_, k_tile),
                 tBsB(cute::_, smem_wr_buffer));
            // printf("loaded ab tile m=%d n=%d k=%d into smem_buf=%d\n",
            // m_tile, n_tile, k_tile, smem_wr_buffer);
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
  } else if (warp_idx == 4) {
    // MMA warp (1)

    // Wait for TMEM allocation to complete
    tmem_allocation_result_barrier.arrive_and_wait();
    tCtAcc.data() = shared_storage.tmem_base_ptr;

    int total_k_tile_count = 0;
    int num_tiles_executed = 0;
    for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
      for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {

        int acc_buf_idx = num_tiles_executed % Num_ACC_Stage;
        auto tCtAcc_Slice = tCtAcc(cute::_, cute::_, cute::_, acc_buf_idx);

        int num_prev_k_blk = total_k_tile_count;
        total_k_tile_count += k_tile_count;

        int mma_rd_k_tile = 0;
        int smem_rd_buffer = (num_prev_k_blk + mma_rd_k_tile) % Num_AB_Stage;
        int mma_rd_ab_full_phase =
            (num_prev_k_blk + mma_rd_k_tile) / Num_AB_Stage % 2;

        // Peek full phase
        bool peek_ab_full_status = kernel::try_wait_barrier(
            shared_storage.ab_full_mbar_ptr[smem_rd_buffer],
            mma_rd_ab_full_phase);

        int acc_empty_phase = num_tiles_executed / Num_ACC_Stage % 2 ^ 1;
        cute::wait_barrier(shared_storage.acc_empty_mbar_ptr[acc_buf_idx],
                           acc_empty_phase);

        // Initialize the accumulator to zero
        tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;

        // CUTE_UNROLL
        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
          int mma_rd_k_tile_next = mma_rd_k_tile + 1;
          int smem_rd_buffer_next =
              (num_prev_k_blk + mma_rd_k_tile_next) % Num_AB_Stage;
          int mma_rd_ab_full_phase_next = smem_rd_buffer_next == 0
                                              ? mma_rd_ab_full_phase ^ 1
                                              : mma_rd_ab_full_phase;

          if (!peek_ab_full_status) {
            cute::wait_barrier(shared_storage.ab_full_mbar_ptr[smem_rd_buffer],
                               mma_rd_ab_full_phase);
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

        cutlass::arch::umma_arrive(
            &shared_storage.acc_full_mbar_ptr[acc_buf_idx]);
        num_tiles_executed++;

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
    using TypeBias = typename decltype(gBias)::value_type;
    using TypeC = typename decltype(tCsC)::value_type;

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

    // if(threadIdx.x == 0) {
    //   cute::print("tiled_copy_t2r:\t"); cute::print(tiled_copy_t2r);
    //   cute::print("\n"); // cute::print("thr_copy_t2r:\t");
    //   cute::print(thr_copy_t2r); cute::print("\n"); //
    //   cute::print("tTR_tAcc:\t"); cute::print(tTR_tAcc); cute::print("\n");
    //   // cute::print("tTR_rAcc:\t"); cute::print(tTR_rAcc);
    //   cute::print("\n"); //
    // } epilogue_wg_barrier.arrive_and_wait();

    int num_tiles_executed = 0;
    for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
      for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {
        int acc_buf_idx = num_tiles_executed % Num_ACC_Stage;
        int acc_full_phase = num_tiles_executed / Num_ACC_Stage % 2;
        int c_smem_wr_buffer_idx = num_tiles_executed % Num_C_Stage;

        cute::Tensor tCgBias =
            gBias(cute::_, cute::_, n_tile, m_tile); // (Mma_M, Mma_N)
        cute::Tensor tCrBiasTypeBias = cute::make_tensor<TypeBias>(
            cute::shape(tTR_rAcc(0, cute::_, 0, 0))); // (T2R_M, T2R_N)
        cute::Tensor tCrBiasTypeAcc =
            cute::make_tensor<AccType>(cute::shape(tCrBiasTypeBias));
        cute::Tensor tCrC =
            cute::make_tensor<TypeC>(cute::shape(tCrBiasTypeBias));

        // if(threadIdx.x == 0 and m_tile == 0 and n_tile == 0) {
        //   cute::print("tCgBias:\t"); cute::print(tCgBias); cute::print("\n");
        //   // cute::print("tCrBiasTypeBias:\t"); cute::print(tCrBiasTypeBias);
        //   cute::print("\n"); // cute::print("tCrBiasTypeAcc:\t");
        //   cute::print(tCrBiasTypeAcc); cute::print("\n"); //
        //   cute::print("tCrC:\t"); cute::print(tCrC); cute::print("\n"); //
        // } epilogue_wg_barrier.arrive_and_wait();

        // T2R and register operations
        if constexpr (!NoBias) {
          // this copy might conflict with TMA load, might add a wait barrier if
          // needed
          cute::copy(tCgBias(cute::_, threadIdx.x), tCrBiasTypeBias);
          // optimize with vectorized type conversion

          CUTE_UNROLL
          for (int i = 0; i < tCrBiasTypeBias.size(); i++) {
            tCrBiasTypeAcc[i] = tCrBiasTypeBias[i];
          }
        }

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

        if constexpr (!NoBias) {
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
        cute::Tensor sC_epi_slice =
            cute::flatten(sC_epi(cute::_, 0, 0, c_smem_wr_buffer_idx));
        cute::copy(tCrC, sC_epi_slice(cute::_, threadIdx.x));

        // S2G TMA
        cute::tma_store_fence(); // Ensure C smem stores are visible to TMA
        epilogue_wg_barrier
            .arrive_and_wait(); // Ensure all threads have issued fence

        if (warp_idx == 0 && cute::elect_one_sync()) {
          cute::copy(*tma_atom_C,
                     tCsC(cute::_, c_smem_wr_buffer_idx),
                     tCgC(cute::_, n_tile, m_tile));
          cute::tma_store_arrive();
          cute::tma_store_wait<Num_C_Stage - 1>();
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
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(shared_storage.tmem_base_ptr, num_tmem_columns);
  }

} // end linear_kernel_sm100

} // namespace kernel
