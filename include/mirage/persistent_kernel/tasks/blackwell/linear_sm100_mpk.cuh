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
#include <cute/algorithm/cooperative_copy.hpp> // Auto vectorized copy operation
#include <cute/arch/cluster_sm90.hpp> // CuTe functions for querying the details of cluster launched
#include <cute/arch/tmem_allocator_sm100.hpp> // TMEM allocator for SM100
#include <cute/numeric/integral_constant.hpp> // Compile time in constants such as _1, _256 etc.
#include <cute/tensor.hpp>                    // CuTe tensor implementation
// using namespace cute;

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

  // Construct the MMA grid coordinate from the CTA grid coordinate
  auto mma_coord_vmnk = cute::make_coord(0,        // Peer CTA coordinate
                                         cute::_,  //    MMA-M coordinate
                                         cute::_,  //    MMA-N coordinate
                                         cute::_); //    MMA-K coordinate

  constexpr int num_tmem_columns = MMA_N * NUM_ACC_STAGE;

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

  // if (cute::thread0()) {
  //   cute::print("mA:\t"); cute::print(mA); cute::print("\n");       // mA:
  //   ArithTuple(_0,_0) o (output_size,reduction_size):(_1@1,_1@0)
  //   cute::print("mB:\t"); cute::print(mB); cute::print("\n");       // mB:
  //   ArithTuple(_0,_0) o (batch_size,reduction_size):(_1@1,_1@0)
  //   cute::print("mBias:\t"); cute::print(mBias); cute::print("\n");   //
  //   mBias:   gmem_ptr[32b](GMEM_ADDR_C) o (batch_size,output_size):(1024,_1)
  //   cute::print("mC:\t"); cute::print(mC); cute::print("\n");       // mC:
  //   ArithTuple(_0,_0) o (batch_size,output_size):(_1@1,_1@0)
  //   cute::print("gA:\t"); cute::print(gA); cute::print("\n");
  //   cute::print("gB:\t"); cute::print(gB); cute::print("\n");
  //   cute::print("gBias:\t"); cute::print(gBias); cute::print("\n");
  //   cute::print("gC:\t"); cute::print(gC); cute::print("\n");
  //   printf("num_ab_stage, num_acc_stage, num_c_stage, : %d, %d, %d\n",
  //   NUM_AB_STAGE, NUM_ACC_STAGE, NUM_C_STAGE); printf("NOBIAS: %d\n",
  //   NOBIAS);
  // } __syncthreads();

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

  // Print and inspect mma_shape_A, and mma_shape_B for this example.
  // if (cute::thread0()) {
  //     cute::print("mma_shape_A:\t"); cute::print(mma_shape_A);
  //     cute::print("\n");  // mma_shape_A:  ((_128,_16),_1,_4,_8)
  //     cute::print("mma_shape_B:\t"); cute::print(mma_shape_B);
  //     cute::print("\n");  // mma_shape_B:  ((_32,_16),_1,_4,_8)
  //     cute::print("mma_shape_C:\t"); cute::print(mma_shape_C);
  //     cute::print("\n");  // mma_shape_C:  ((_32,_128),_1,_1,_4)
  // } __syncthreads();

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

  // if (cute::thread0()){
  //     cute::print("sA_layout:\t"); cute::print(sA_layout); cute::print("\n");
  //     // sA_layout:   ArithTuple(_0,0) o
  //     ((_128,_16),_1,_4,8):((_1@1,_1@0),_0,_16@0,_8@0)
  //     cute::print("sB_layout:\t"); cute::print(sB_layout); cute::print("\n");
  //     // sB_layout:   ArithTuple(_0,0) o
  //     ((_32,_16),_1,_4,8):((_1@1,_1@0),_0,_16@0,_8@0)
  //     cute::print("sC_layout:\t"); cute::print(sC_layout); cute::print("\n");
  //     // sC_layout:   ArithTuple(_0,0) o
  //     ((_32,_128),_1,_1,4):((_1@1,_1@0),_0,_128@0,_4@0)
  // } __syncthreads();

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

  // Initialize the barriers in shared memory
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

  // if (cute::thread0()) {
  //   cute::print("tCgA:\t"); cute::print(tCgA); cute::print("\n");  // tCgA:
  //   ArithTuple(_0,0) o ((_128,_16),_1,_4,64):((_1@1,_1@0),_0,_16@0,_64@0)
  //   cute::print("tCgB:\t"); cute::print(tCgB); cute::print("\n");  // tCgB:
  //   ArithTuple(_0,0) o ((_32,_16),_1,_4,64):((_1@1,_1@0),_0,_16@0,_64@0)
  //   cute::print("tCgC_epi:\t"); cute::print(tCgC_epi); cute::print("\n");  //
  //   tCgC:   ArithTuple(_0,_0) o
  //   ((_32,_128),_1,_1,1,8):((_1@1,_1@0),_0,_0,_32@1,_128@0)
  //   cute::print("tCsA:\t"); cute::print(tCsA); cute::print("\n");
  //   cute::print("tCsB:\t"); cute::print(tCsB); cute::print("\n");
  //   cute::print("sC_epi:\t"); cute::print(sC_epi); cute::print("\n");
  // } __syncthreads();

  // int tma_trans_bytes_A = sizeof(T_) * cute::size<1>(mma_tiler) *
  // cute::size<2>(mma_tiler); int tma_trans_bytes_B = sizeof(T_) *
  // cute::size<0>(mma_tiler) * cute::size<2>(mma_tiler);
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

  // // check tma descriptor:
  // if(threadIdx.x == 0){
  //     printf("smem start addr: %u\n",
  //     cute::cast_smem_ptr_to_uint(shared_memory)); printf("shared_weight:
  //     %u\n", cute::cast_smem_ptr_to_uint(shared_weight));
  //     printf("shared_input: %u\n",
  //     cute::cast_smem_ptr_to_uint(shared_input)); printf("mm_output: %u\n",
  //     cute::cast_smem_ptr_to_uint(mm_output));
  // } __syncthreads();

  // // Fence acquire tensor map:
  // ptx::n32_t<128> size_bytes;
  // // Since the tensor map was modified from the host using cudaMemcpy,
  // // the scope should be .sys.
  // ptx::fence_proxy_tensormap_generic(
  //   ptx::sem_acquire, ptx::scope_sys, tma_a.desc_ptr, size_bytes
  // );
  // ptx::fence_proxy_tensormap_generic(
  //   ptx::sem_acquire, ptx::scope_sys, tma_b.desc_ptr, size_bytes
  // );
  // ptx::fence_proxy_tensormap_generic(
  //   ptx::sem_acquire, ptx::scope_sys, tma_out.desc_ptr, size_bytes
  // );

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
  TASK_SYNC();

  // if (cute::thread0()) {
  //   printf("tma_trans_bytes_A: %d\n", tma_trans_bytes_A);
  //   printf("tma_trans_bytes_B: %d\n", tma_trans_bytes_B);
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

  TASK_SYNC(); // Wait for all threads until warp0 allocates TMEM

  if (warp_idx == 5) {
    // TMA warp (1)
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

        // CUTE_UNROLL
        for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {

          int tma_wr_k_tile_next = tma_wr_k_tile + 1;
          int smem_wr_buffer_next =
              (num_prev_k_blk + tma_wr_k_tile_next) % NUM_AB_STAGE;
          int tma_wr_ab_empty_phase_next = smem_wr_buffer_next == 0
                                               ? tma_wr_ab_empty_phase ^ 1
                                               : tma_wr_ab_empty_phase;

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

        // Initialize the accumulator to zero
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

          // if (!peek_ab_full_status){
          //   wait(ab_full_mbar_ptr[smem_rd_buffer], mma_rd_ab_full_phase);
          // }

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

    // if(threadIdx.x == 0) {
    //   cute::print("tiled_copy_t2r:\t"); cute::print(tiled_copy_t2r);
    //   cute::print("\n"); // cute::print("thr_copy_t2r:\t");
    //   cute::print(thr_copy_t2r); cute::print("\n"); //
    //   cute::print("tTR_tAcc:\t"); cute::print(tTR_tAcc); cute::print("\n");
    //   // cute::print("tTR_rAcc:\t"); cute::print(tTR_rAcc);
    //   cute::print("\n"); // cute::print("tCtAcc:\t"); cute::print(tCtAcc);
    //   cute::print("\n"); // printf("tmem_base_ptr: %u\n",
    //   shared_storage.tmem_base_ptr);
    // } epilogue_wg_barrier.arrive_and_wait();

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

        // if(threadIdx.x == 0 and m_tile == 0 and n_tile == 0) {
        //   cute::print("tCgBias:\t"); cute::print(tCgBias); cute::print("\n");
        //   // cute::print("tCrBiasTypeBias:\t"); cute::print(tCrBiasTypeBias);
        //   cute::print("\n"); // cute::print("tCrBiasTypeAcc:\t");
        //   cute::print(tCrBiasTypeAcc); cute::print("\n"); //
        //   cute::print("tCrC:\t"); cute::print(tCrC); cute::print("\n"); //
        // } epilogue_wg_barrier.arrive_and_wait();

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
        cute::Tensor sC_epi_slice =
            cute::flatten(sC_epi(cute::_, 0, 0, c_smem_wr_buffer_idx));
        cute::copy(tCrC, sC_epi_slice(cute::_, threadIdx.x));

        // S2G TMA
        cute::tma_store_fence(); // Ensure C smem stores are visible to TMA
        epilogue_wg_barrier
            .arrive_and_wait(); // Ensure all threads have issued fence

        if (warp_idx == 0 && cute::elect_one_sync()) {
          if constexpr (SplitK) {
            tma_out.tma_reduce_add_async(
                mm_output_smem.base_ptr,
                {m_tile * OUTPUT_ATOM_SIZE, n_tile * MMA_N});
          } else {
            tma_out.tma_store_async(
                mm_output_smem.base_ptr,
                {m_tile * OUTPUT_ATOM_SIZE, n_tile * MMA_N});
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
  TASK_SYNC();

  // Release the right to allocate before deallocations so that the next CTA can
  // rasterize Then deallocate TMEM
  if (warp_idx == 0) {
    // don't do relinquish for megakernel
    // tmem_allocator.release_allocation_lock();
    tmem_allocator.free(shared_storage.tmem_base_ptr, num_tmem_columns);
  }

} // end linear_sm100_mpk_task_impl

// ── Phase functions for streaming pipeline ───────────────────────────────────
// Each function handles one task's worth of work for a single warp role.
// They are self-contained: each reconstructs the CuTe types it needs
// (compile-time, zero cost) from shared_storage and template parameters.

// ── LOADER (warp 5): TMA loads for one task ──────────────────────────────────
template <typename T_, int MMA_M, int MMA_N,
          int NUM_AB_STAGE, typename SharedStorageT>
__device__ void linear_sm100_load_one_task(
    SharedStorageT &shared_storage,
    CUtensorMap *tma_a_ptr,
    CUtensorMap *tma_b_ptr,
    int k_tiles, int m_tiles, int n_tiles,
    int &global_ab_tile) {
  constexpr int TILE_SIZE = 64;
  constexpr int OUTPUT_ATOM_SIZE = MMA_M;
  constexpr int B = 3, M = 3, S = 3;

  int tma_transaction_bytes =
      sizeof(T_) * MMA_N * TILE_SIZE + sizeof(T_) * MMA_M * TILE_SIZE;

  T_ *shared_weight = shared_storage.A.begin();
  T_ *shared_input = shared_storage.B.begin();

  Barrier *ab_full_mbar_ptr =
      reinterpret_cast<Barrier *>(shared_storage.ab_full_mbar_ptr);

  using InputSmem = smem_tma<T_, B, M, S, MMA_N, TILE_SIZE, 1>;
  InputSmem input_smem(shared_input);

  using WeightSmem = smem_tma<T_, B, M, S, OUTPUT_ATOM_SIZE, TILE_SIZE, 1>;
  WeightSmem input_weight_smem(shared_weight);

  using TmaA = kernel::tma::tma_2d<T_, B, M, S,
      1, 1, MMA_M, TILE_SIZE, 1, 1, 1, 1, MMA_M * TILE_SIZE, true>;
  using TmaB = kernel::tma::tma_2d<T_, B, M, S,
      1, 1, MMA_N, TILE_SIZE, 1, 1, 1, 1, MMA_N * TILE_SIZE, true>;
  TmaA tma_a(tma_a_ptr);
  TmaB tma_b(tma_b_ptr);

  for (int m = 0; m < m_tiles; m++) {
    for (int n = 0; n < n_tiles; n++) {
      for (int k = 0; k < k_tiles; k++) {
        int smem_wr_buffer = global_ab_tile % NUM_AB_STAGE;
        int ab_empty_phase =
            (global_ab_tile / NUM_AB_STAGE) % 2 ^ 1;

        cute::wait_barrier(
            shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
            ab_empty_phase);

        if (cute::elect_one_sync()) {
          int tma_coords_A[2] = {k * TILE_SIZE, m * OUTPUT_ATOM_SIZE};
          int tma_coords_B[2] = {k * TILE_SIZE, n * MMA_N};
          input_weight_smem.set_ptr(
              shared_weight + smem_wr_buffer * OUTPUT_ATOM_SIZE * TILE_SIZE);
          input_smem.set_ptr(
              shared_input + smem_wr_buffer * MMA_N * TILE_SIZE);
          cute::set_barrier_transaction_bytes(
              shared_storage.ab_full_mbar_ptr[smem_wr_buffer],
              tma_transaction_bytes);
          tma_a.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer],
                             input_weight_smem.base_ptr, tma_coords_A);
          tma_b.tma_cp_async(ab_full_mbar_ptr[smem_wr_buffer],
                             input_smem.base_ptr, tma_coords_B);
        }

        global_ab_tile++;
      }
    }
  }
}

// ── COMPUTE (warp 4): WGMMA accumulation for one task ───────────────────────
template <typename T_, int MMA_M, int MMA_N,
          int NUM_AB_STAGE, int NUM_ACC_STAGE, typename SharedStorageT>
__device__ void linear_sm100_compute_one_task(
    SharedStorageT &shared_storage,
    int k_tiles, int m_tiles, int n_tiles,
    int &global_ab_tile, int &global_out_tile) {
  cute::TiledMMA tiled_mma = cute::make_tiled_mma(
      cute::SM100_MMA_F16BF16_SS<T_, T_, float, MMA_M, MMA_N,
                                  cute::UMMA::Major::K,
                                  cute::UMMA::Major::K>{});
  auto bM = cute::tile_size<0>(tiled_mma);
  auto bN = cute::tile_size<1>(tiled_mma);
  auto bK = cute::tile_size<2>(tiled_mma) * cute::Int<4>{};
  auto mma_tiler = cute::make_shape(bM, bN, bK);

  cute::ThrMMA cta_mma = tiled_mma.get_slice(0);
  cute::Tensor tCsA = shared_storage.tensor_sA();
  cute::Tensor tCsB = shared_storage.tensor_sB();
  cute::Tensor tCrA = cta_mma.make_fragment_A(tCsA);
  cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);

  auto acc_shape = cute::partition_shape_C(
      tiled_mma,
      cute::make_shape(cute::size<0>(mma_tiler), cute::size<1>(mma_tiler),
                       cute::Int<NUM_ACC_STAGE>{}));
  auto tCtAcc = tiled_mma.make_fragment_C(acc_shape);
  tCtAcc.data() = shared_storage.tmem_base_ptr;

  for (int m = 0; m < m_tiles; m++) {
    for (int n = 0; n < n_tiles; n++) {
      int acc_buf_idx = global_out_tile % NUM_ACC_STAGE;
      auto tCtAcc_Slice =
          tCtAcc(cute::_, cute::_, cute::_, acc_buf_idx);

      int acc_empty_phase =
          (global_out_tile / NUM_ACC_STAGE) % 2 ^ 1;
      cute::wait_barrier(
          shared_storage.acc_empty_mbar_ptr[acc_buf_idx],
          acc_empty_phase);

      tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;

      for (int k = 0; k < k_tiles; k++) {
        int smem_rd_buffer = global_ab_tile % NUM_AB_STAGE;
        int ab_full_phase =
            (global_ab_tile / NUM_AB_STAGE) % 2;

        cute::wait_barrier(
            shared_storage.ab_full_mbar_ptr[smem_rd_buffer],
            ab_full_phase);

        for (int k_block = 0;
             k_block < cute::size<2>(tCrA); ++k_block) {
          cute::gemm(
              tiled_mma,
              tCrA(cute::_, cute::_, k_block, smem_rd_buffer),
              tCrB(cute::_, cute::_, k_block, smem_rd_buffer),
              tCtAcc_Slice);
          tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
        }

        cutlass::arch::umma_arrive(
            &shared_storage.ab_empty_mbar_ptr[smem_rd_buffer]);

        global_ab_tile++;
      } // k

      cutlass::arch::umma_arrive(
          &shared_storage.acc_full_mbar_ptr[acc_buf_idx]);
      global_out_tile++;
    } // n
  }   // m
}

// ── EPILOGUE (warps 0-3): T2R → bias → convert → TMA store for one task ────
template <typename T_, int MMA_M, int MMA_N,
          int NUM_AB_STAGE, int NUM_ACC_STAGE, int NUM_C_STAGE,
          typename SharedStorageT>
__device__ void linear_sm100_store_one_task(
    SharedStorageT &shared_storage,
    CUtensorMap *tma_out_ptr,
    T_ *bias_ptr,
    int output_size,
    int m_tiles, int n_tiles,
    int &global_out_tile,
    int warp_idx) {
  constexpr int OUTPUT_ATOM_SIZE = MMA_M;
  constexpr int B = 3, M = 3, S = 3;

  // Reconstruct tCtAcc for T2R source shape
  cute::TiledMMA tiled_mma = cute::make_tiled_mma(
      cute::SM100_MMA_F16BF16_SS<T_, T_, float, MMA_M, MMA_N,
                                  cute::UMMA::Major::K,
                                  cute::UMMA::Major::K>{});
  auto bM = cute::tile_size<0>(tiled_mma);
  auto bN = cute::tile_size<1>(tiled_mma);
  auto bK = cute::tile_size<2>(tiled_mma) * cute::Int<4>{};
  auto mma_tiler = cute::make_shape(bM, bN, bK);

  auto acc_shape = cute::partition_shape_C(
      tiled_mma,
      cute::make_shape(cute::size<0>(mma_tiler), cute::size<1>(mma_tiler),
                       cute::Int<NUM_ACC_STAGE>{}));
  auto tCtAcc = tiled_mma.make_fragment_C(acc_shape);
  tCtAcc.data() = shared_storage.tmem_base_ptr;

  using AccType = typename decltype(tCtAcc)::value_type;
  using TypeBias = T_;
  using TypeC = T_;

  cutlass::NumericConverter<AccType, TypeBias> converterBias;
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

  cute::Tensor sC_epi = shared_storage.tensor_sC();
  T_ *mm_output = shared_storage.C.begin();

  using OutputSmem = smem_tma<T_, 0, M, S, MMA_N, OUTPUT_ATOM_SIZE, 1>;
  OutputSmem mm_output_smem(mm_output);

  using TmaOut = kernel::tma::tma_2d<T_, 0, M, S,
      1, 1, MMA_N, OUTPUT_ATOM_SIZE, 1, 1, 1, 1,
      MMA_N * OUTPUT_ATOM_SIZE, true>;
  TmaOut tma_out(tma_out_ptr);

  cutlass::arch::NamedBarrier epilogue_wg_barrier(
      128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

  for (int m = 0; m < m_tiles; m++) {
    for (int n = 0; n < n_tiles; n++) {
      int acc_buf_idx = global_out_tile % NUM_ACC_STAGE;
      int acc_full_phase = (global_out_tile / NUM_ACC_STAGE) % 2;
      int c_smem_wr_buffer_idx = global_out_tile % NUM_C_STAGE;

      cute::Tensor tCrBiasTypeBias = cute::make_tensor<TypeBias>(
          cute::shape(tTR_rAcc(0, cute::_, 0, 0)));
      cute::Tensor tCrBiasTypeAcc =
          cute::make_tensor<AccType>(cute::shape(tCrBiasTypeBias));
      cute::Tensor tCrC =
          cute::make_tensor<TypeC>(cute::shape(tCrBiasTypeBias));

      if (bias_ptr != nullptr) {
        T_ *bias_tile_base =
            bias_ptr + n * MMA_N * output_size + m * OUTPUT_ATOM_SIZE;
        CUTE_UNROLL
        for (int j = 0; j < tCrBiasTypeBias.size(); j++) {
          tCrBiasTypeBias[j] =
              bias_tile_base[j * output_size + threadIdx.x];
        }
        CUTE_UNROLL
        for (int i = 0; i < tCrBiasTypeBias.size(); i++) {
          tCrBiasTypeAcc[i] = converterBias(tCrBiasTypeBias[i]);
        }
      }

      mm_output_smem.set_ptr(
          mm_output + c_smem_wr_buffer_idx * MMA_N * OUTPUT_ATOM_SIZE);

      cute::wait_barrier(
          shared_storage.acc_full_mbar_ptr[acc_buf_idx],
          acc_full_phase);

      // T2R copy
      cute::copy(tiled_copy_t2r,
                 tTR_tAcc(cute::_, cute::_, cute::_, cute::_, acc_buf_idx),
                 tTR_rAcc);

      epilogue_wg_barrier.arrive_and_wait();
      if (cute::elect_one_sync()) {
        cute::arrive_barrier(
            shared_storage.acc_empty_mbar_ptr[acc_buf_idx]);
      }

      // Add bias
      if (bias_ptr != nullptr) {
        CUTE_UNROLL
        for (int i = 0; i < tTR_rAcc.size(); i++) {
          tTR_rAcc[i] += tCrBiasTypeAcc[i];
        }
      }

      // Convert to output type
      CUTE_UNROLL
      for (int i = 0; i < tCrC.size(); i++) {
        tCrC[i] = converter(tTR_rAcc[i]);
      }

      // R2S copy
      cute::Tensor sC_epi_slice =
          cute::flatten(sC_epi(cute::_, 0, 0, c_smem_wr_buffer_idx));
      cute::copy(tCrC, sC_epi_slice(cute::_, threadIdx.x));

      // S2G TMA store
      cute::tma_store_fence();
      epilogue_wg_barrier.arrive_and_wait();

      if (warp_idx == 0 && cute::elect_one_sync()) {
        tma_out.tma_store_async(
            mm_output_smem.base_ptr,
            {m * OUTPUT_ATOM_SIZE, n * MMA_N});
        cute::tma_store_arrive();
        cute::tma_store_wait<NUM_C_STAGE - 1>();
      }

      global_out_tile++;
    } // n
  }   // m

  // Drain all TMA stores for this task
  if (warp_idx == 0 && cute::elect_one_sync()) {
    cute::tma_store_wait<0>();
  }
  epilogue_wg_barrier.arrive_and_wait();
}

// ── Streaming pipeline for consecutive LINEAR_SM100 tasks ────────────────────
// Runs a sequence of LINEAR tasks through the same SMEM pipeline, reusing
// barriers and TMEM across task boundaries.  Warp 5 (loader) runs 1 task
// ahead of warp 4 (MMA), which runs ahead of warps 0-3 (epilogue).
//
// Parameters:
//   task_buf           – double-buffered TaskDesc in SMEM (from controller)
//   task_ready_mbar    – [2] controller → compute: "deps met, TaskDesc ready"
//   task_done_mbar     – [2] compute → controller: "task complete"
//   run_length         – # consecutive LINEAR tasks in this streaming run
//   first_task_loop_idx – loop index of the first task (for mbarrier phase)
//   config             – RuntimeConfig (for GMEM barriers)
//
// Caller must ensure:
//   - task_ready_mbar[first_task_loop_idx & 1] has already been signaled
//     for the first task (compute warps already waited on it).
//   - The function signals task_done_mbar and GMEM barriers per task.
//   - After return, callers may assume TMEM is freed and barriers are stale.

template <typename T_,
          int MMA_M = 128,
          int MMA_N = 16,
          int NUM_AB_STAGE = 8,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE = 4>
__device__ __noinline__ void
    linear_sm100_streaming_run(TaskDesc *task_buf,
                               uint64_t *task_ready_mbar,
                               uint64_t *task_done_mbar,
                               int *signal_barrier_idx_ptr,
                               int run_length,
                               int first_task_loop_idx,
                               RuntimeConfig const &config) {
  int warp_idx = cutlass::canonical_warp_idx_sync();

  // ── Fixed constants (same for ALL LINEAR_SM100 tasks) ─────────────────────
  constexpr int TILE_SIZE = 64;
  constexpr int OUTPUT_ATOM_SIZE = MMA_M; // 128
  constexpr int INPUT_TMA_TILE_SIZE = TILE_SIZE;
  constexpr int WEIGHT_TMA_TILE_SIZE = TILE_SIZE;
  constexpr int B = 3, M = 3, S = 3;

  constexpr int num_tmem_columns = MMA_N * NUM_ACC_STAGE;

  // ── CuTe MMA setup (compile-time, identical across tasks) ─────────────────
  cute::TiledMMA tiled_mma = cute::make_tiled_mma(
      cute::SM100_MMA_F16BF16_SS<T_, T_, float, MMA_M, MMA_N,
                                  cute::UMMA::Major::K,
                                  cute::UMMA::Major::K>{});
  auto bM = cute::tile_size<0>(tiled_mma);
  auto bN = cute::tile_size<1>(tiled_mma);
  auto bK = cute::tile_size<2>(tiled_mma) * cute::Int<4>{};

  auto mma_tiler = cute::make_shape(bM, bN, bK);
  auto mma_coord_vmnk = cute::make_coord(0, cute::_, cute::_, cute::_);
  auto mma_v = cute::get<0>(mma_coord_vmnk);
  cute::ThrMMA cta_mma = tiled_mma.get_slice(mma_v);

  // AB fragment shapes (compile-time)
  auto mma_shape_A = cute::partition_shape_A(
      tiled_mma,
      cute::make_shape(cute::Int<MMA_M>{}, cute::size<2>(mma_tiler),
                       cute::Int<NUM_AB_STAGE>{}));
  auto mma_shape_B = cute::partition_shape_B(
      tiled_mma,
      cute::make_shape(cute::Int<MMA_N>{}, cute::size<2>(mma_tiler),
                       cute::Int<NUM_AB_STAGE>{}));
  auto mma_shape_C = cute::make_shape(
      cute::make_shape(cute::Int<MMA_N>{}, cute::Int<MMA_M>{}),
      cute::Int<1>{}, cute::Int<1>{}, cute::Int<NUM_C_STAGE>{});

  // SMEM layouts
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

  using SharedStorage =
      PipedSharedStorage<T_, T_, T_, decltype(sA_layout), decltype(sB_layout),
                         decltype(sC_layout), NUM_AB_STAGE, NUM_ACC_STAGE>;

  extern __shared__ char shared_memory[];
  uintptr_t aligned_smem =
      (reinterpret_cast<uintptr_t>(shared_memory) + 127) / 128 * 128;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(aligned_smem);

  // SMEM tensor views
  cute::Tensor tCsA = shared_storage.tensor_sA();
  cute::Tensor tCsB = shared_storage.tensor_sB();
  cute::Tensor sC_epi = shared_storage.tensor_sC();

  // MMA fragments (SMEM descriptors for A/B, TMEM for accumulator)
  cute::Tensor tCrA = cta_mma.make_fragment_A(tCsA);
  cute::Tensor tCrB = cta_mma.make_fragment_B(tCsB);
  auto acc_shape = cute::partition_shape_C(
      tiled_mma,
      cute::make_shape(cute::size<0>(mma_tiler), cute::size<1>(mma_tiler),
                       cute::Int<NUM_ACC_STAGE>{}));
  auto tCtAcc = tiled_mma.make_fragment_C(acc_shape);

  int tma_transaction_bytes =
      sizeof(T_) * cute::size<1>(mma_tiler) * cute::size<2>(mma_tiler) +
      sizeof(T_) * cute::size<0>(mma_tiler) * cute::size<2>(mma_tiler);

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

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  // ── Phase 0: One-time initialization ────────────────────────────────────
  if (warp_idx == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier, NUM_AB_STAGE>(
        shared_storage.ab_full_mbar_ptr, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, NUM_AB_STAGE>(
        shared_storage.ab_empty_mbar_ptr, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, NUM_ACC_STAGE>(
        shared_storage.acc_full_mbar_ptr, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier, NUM_ACC_STAGE>(
        shared_storage.acc_empty_mbar_ptr, 4);
  }

  cutlass::arch::NamedBarrier tmem_allocation_result_barrier(
      32 + 128, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
  cutlass::arch::NamedBarrier epilogue_wg_barrier(
      128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

  cutlass::arch::fence_barrier_init();
  TASK_SYNC();

  // ── Phase 1: Streaming execution ───────────────────────────────────────

  if (warp_idx == 5) {
    // ── LOADER WARP ─────────────────────────────────────────────────────
    // Runs 1 task ahead of MMA.  Reads TMA descriptors and tile counts
    // from task_buf at each task boundary; streams tiles through the
    // 8-stage AB circular buffer continuously across tasks.

    int global_ab_tile = 0; // monotonic tile index across all tasks
    for (int ti = 0; ti < run_length; ti++) {
      int cur = (first_task_loop_idx + ti) & 1;
      int phase = ((first_task_loop_idx + ti) >> 1) & 1;

      // Wait until controller has loaded this task's TaskDesc
      if (ti > 0) {
        // For the first task, the caller already waited on task_ready
        if (cute::elect_one_sync())
          mbar_wait(&task_ready_mbar[cur], phase);
        __syncwarp();
      }

      // Read per-task info from TaskDesc
      CUtensorMap *tma_a_ptr = static_cast<CUtensorMap *>(
          task_buf[cur].input_tma_desc_ptrs[1][0]);
      CUtensorMap *tma_b_ptr = static_cast<CUtensorMap *>(
          task_buf[cur].input_tma_desc_ptrs[0][0]);
      int k_tiles = task_buf[cur].task_metadata.linear_tiles.k_tile_count;
      int m_tiles = task_buf[cur].task_metadata.linear_tiles.m_tile_count;
      int n_tiles = task_buf[cur].task_metadata.linear_tiles.n_tile_count;

      linear_sm100_load_one_task<T_, MMA_M, MMA_N, NUM_AB_STAGE>(
          shared_storage, tma_a_ptr, tma_b_ptr,
          k_tiles, m_tiles, n_tiles, global_ab_tile);
    } // end loader task loop

  } else if (warp_idx == 4) {
    // ── MMA WARP ────────────────────────────────────────────────────────
    // Follows loader through the AB buffer.  Resets accumulator at each
    // new (m,n) output tile; signals acc_full after reduction completes.

    tmem_allocation_result_barrier.arrive_and_wait();

    int global_ab_tile = 0;
    int global_out_tile = 0;
    for (int ti = 0; ti < run_length; ti++) {
      int cur = (first_task_loop_idx + ti) & 1;
      int phase = ((first_task_loop_idx + ti) >> 1) & 1;

      if (ti > 0) {
        if (cute::elect_one_sync())
          mbar_wait(&task_ready_mbar[cur], phase);
        __syncwarp();
      }

      int k_tiles = task_buf[cur].task_metadata.linear_tiles.k_tile_count;
      int m_tiles = task_buf[cur].task_metadata.linear_tiles.m_tile_count;
      int n_tiles = task_buf[cur].task_metadata.linear_tiles.n_tile_count;

      linear_sm100_compute_one_task<T_, MMA_M, MMA_N,
                                    NUM_AB_STAGE, NUM_ACC_STAGE>(
          shared_storage, k_tiles, m_tiles, n_tiles,
          global_ab_tile, global_out_tile);
    } // task loop

  } else if (warp_idx < 4) {
    // ── EPILOGUE WARPS (0-3) ────────────────────────────────────────────
    // Follow MMA through the ACC buffer.  For each output tile: T2R →
    // add bias → convert → SMEM → TMA store.  After all tiles of a task,
    // drain TMA stores and signal GMEM barrier + task_done.

    if (warp_idx == 0) {
      tmem_allocator.allocate(num_tmem_columns,
                              &shared_storage.tmem_base_ptr);
    }
    tmem_allocation_result_barrier.arrive_and_wait();

    int global_out_tile = 0;
    for (int ti = 0; ti < run_length; ti++) {
      int cur = (first_task_loop_idx + ti) & 1;
      int phase = ((first_task_loop_idx + ti) >> 1) & 1;

      if (ti > 0) {
        // Wait until controller loaded this task's TaskDesc
        if (threadIdx.x == 0)
          mbar_wait(&task_ready_mbar[cur], phase);
        epilogue_wg_barrier.arrive_and_wait();
      }

      int m_tiles = task_buf[cur].task_metadata.linear_tiles.m_tile_count;
      int n_tiles = task_buf[cur].task_metadata.linear_tiles.n_tile_count;

      CUtensorMap *tma_out_ptr = static_cast<CUtensorMap *>(
          task_buf[cur].output_tma_desc_ptrs[0][0]);
      T_ *bias_ptr = nullptr;
      if (task_buf[cur].task_type == TASK_LINEAR_WITH_RESIDUAL_SM100) {
        bias_ptr = static_cast<T_ *>(task_buf[cur].input_ptrs[2]);
      }
      int output_size = m_tiles * OUTPUT_ATOM_SIZE;

      linear_sm100_store_one_task<T_, MMA_M, MMA_N,
                                  NUM_AB_STAGE, NUM_ACC_STAGE, NUM_C_STAGE>(
          shared_storage, tma_out_ptr, bias_ptr,
          output_size, m_tiles, n_tiles,
          global_out_tile, warp_idx);

      // Signal GMEM completion barrier for cross-SM dependencies
      if (threadIdx.x == 0 && *signal_barrier_idx_ptr >= 0) {
        barrier_arrive(config.barriers, *signal_barrier_idx_ptr);
      }

      // Signal controller: this task is done
      if (threadIdx.x == 0) {
        mbar_arrive(&task_done_mbar[cur]);
      }
    } // task loop

    // ── Phase 2: Cleanup ────────────────────────────────────────────────
    epilogue_wg_barrier.arrive_and_wait();
    if (warp_idx == 0) {
      tmem_allocator.free(shared_storage.tmem_base_ptr, num_tmem_columns);
    }
  } else {
    // Warps 6-7: idle during LINEAR tasks.
    // Participate in the initial TASK_SYNC and the final one outside.
    // For TMEM allocation barrier: need to participate
    // (they're outside the 32+128 thread count, so they skip it).
  }

  // Final sync: ensure all warps are done before returning to the main loop
  TASK_SYNC();
}

} // namespace kernel
