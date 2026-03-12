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
          class InputTensor,
          class BiasTensor,
          class IndicesTensor,
          class MaskTensor,
          class OutputTensor,
          int MMA_M,
          int MMA_N,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int ORIG_OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int EXPERT_STRIDE,
          bool W13_LINEAR,
          bool NOBIAS,
          int NUM_AB_STAGE = 8,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE = 4>
__device__ __forceinline__ void
    moe_linear_sm100_task_impl(const TMA_A &tma_a,
                               InputTensor mInput,
                               BiasTensor mBias,
                               IndicesTensor mRoutingIndices,
                               MaskTensor mMask,
                               OutputTensor mOutput,
                               int const expert_offset) {
  int warp_idx = cutlass::canonical_warp_idx_sync();

  // Construct the MMA grid coordinate from the CTA grid coordinate
  auto mma_coord_vmnk = cute::make_coord(0,        // Peer CTA coordinate
                                         cute::_,  //    MMA-M coordinate
                                         cute::_,  //    MMA-N coordinate
                                         cute::_); //    MMA-K coordinate

  constexpr int num_tmem_columns = MMA_N * NUM_ACC_STAGE;
  using AccType = float;

  cute::TiledMMA tiled_mma = cute::make_tiled_mma(
      cute::SM100_MMA_F16BF16_SS<T_,
                                 T_,
                                 AccType, // Mma's A, B, and Accumulator types
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

  // TODO(Zhihao): remove the dependency of mA/gA/tCgA
  cute::Tensor mA = cute::make_coord_tensor(cute::make_layout(
      cute::make_shape(OUTPUT_SIZE, REDUCTION_SIZE, NUM_EXPERTS),
      cute::make_stride(cute::E<1>{},
                        cute::E<0>{},
                        cute::E<1>{} * cute::Int<OUTPUT_SIZE>{})));

  //   if (cute::thread0()) {
  //     cute::print("mA:\t");
  //     cute::print(mA);
  //     cute::print("\n");
  //     cute::print("mInput:\t");
  //     cute::print(mInput);
  //     cute::print("\n");
  //     cute::print("mRoutingIndices:\t");
  //     cute::print(mRoutingIndices);
  //     cute::print("\n");
  //     cute::print("mMask:\t");
  //     cute::print(mMask);
  //     cute::print("\n");
  //     cute::print("mOutput:\t");
  //     cute::print(mOutput);
  //     cute::print("\n");
  //   }
  //   __syncthreads();

  cute::Tensor gA = cute::local_tile(
      mA,
      mma_tiler,
      mma_coord,
      cute::Step<
          cute::_1,
          cute::X,
          cute::_1>{}); // ArithTuple(_0,_0) o
                        // (_128,_64,1,32,128):(_1@1,_1@0,_128@1,_64@0,_128@1)
  cute::Tensor gB = cute::local_tile(
      mInput,
      mma_tiler,
      mma_coord,
      cute::Step<cute::X,
                 cute::_1,
                 cute::_1>{}); // gmem_ptr[16b](0x792c0b000000) o
                               // (_16,_64,1,32):(2048,_1,32768,_64)
  cute::Tensor gBias = cute::local_tile(
      mBias, cd_tiler, mma_coord, cute::Step<cute::_1, cute::_1, cute::X>{});

  //   if (cute::thread0()) {
  //     cute::print("gA:\t");
  //     cute::print(gA);
  //     cute::print("\n"); // gA:     ArithTuple(_0,_0) o
  //                        //
  //                        (_128,_64,1,32,128):(_1@1,_1@0,_128@1,_64@0,_128@1)
  //     cute::print("gB:\t");
  //     cute::print(gB);
  //     cute::print("\n"); // gB:     gmem_ptr[16b](0x792c0b000000) o
  //                        // (_16,_64,1,32):(2048,_1,32768,_64)
  //     cute::print("gBias:\t");
  //     cute::print(gBias);
  //     cute::print("\n");
  //   }
  //   __syncthreads();

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

  //   // Print and inspect mma_shape_A, and mma_shape_B for this example.
  //   if (cute::thread0()) {
  //     cute::print("mma_shape_A:\t");
  //     cute::print(mma_shape_A);
  //     cute::print("\n"); // mma_shape_A:  ((_128,_16),_1,_4,_8)
  //     cute::print("mma_shape_B:\t");
  //     cute::print(mma_shape_B);
  //     cute::print("\n"); // mma_shape_B:  ((_32,_16),_1,_4,_8)
  //     cute::print("mma_shape_C:\t");
  //     cute::print(mma_shape_C);
  //     cute::print("\n"); // mma_shape_C:  ((_32,_128),_1,_1,_4)
  //   }
  //   __syncthreads();

  auto sA_layout = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<T_>{}, mma_shape_A);
  auto sB_layout = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<T_>{}, mma_shape_B);
  auto sB_cp_layout = cute::composition(
      sB_layout.layout_a(),
      sB_layout.offset(),
      cute::make_layout(cute::make_shape(cute::get<0>(cute::shape(gB)),
                                         cute::get<1>(cute::shape(gB)),
                                         cute::Int<1>{},
                                         cute::Int<NUM_AB_STAGE>{}),
                        cute::make_stride(cute::get<1>(cute::shape(gB)),
                                          cute::Int<1>{},
                                          cute::Int<0>{},
                                          cute::Int<MMA_N * bK>{})));
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

  //   if (cute::thread0()) {
  //     cute::print("sA_layout:\t");
  //     cute::print(sA_layout);
  //     cute::print("\n");
  //     cute::print("sB_layout:\t");
  //     cute::print(sB_layout);
  //     cute::print("\n");
  //     cute::print("sC_layout:\t");
  //     cute::print(sC_layout);
  //     cute::print("\n");
  //     cute::print("sB_cp_layout:\t");
  //     cute::print(sB_cp_layout);
  //     cute::print("\n");
  //   }
  //   __syncthreads();

  using SharedStorage = MoESharedStorage<T_,
                                         T_,
                                         decltype(sA_layout),
                                         decltype(sB_layout),
                                         decltype(sB_cp_layout),
                                         NUM_EXPERTS,
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
        NUM_AB_STAGE>(shared_storage.a_full_mbar_ptr,
                      /* arrival count
                       */
                      1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        NUM_AB_STAGE>(shared_storage.b_full_mbar_ptr,
                      /* arrival count
                       */
                      32);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        NUM_AB_STAGE>(shared_storage.ab_empty_mbar_ptr,
                      /* arrival count
                       */
                      1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        NUM_ACC_STAGE>(shared_storage.acc_full_mbar_ptr,
                       /* arrival count
                        */
                       1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        NUM_ACC_STAGE>(shared_storage.acc_empty_mbar_ptr,
                       /* arrival count */ 4);
  }

  // Sync tmem allocation status between MMA and epilogue warps within CTA
  // 32 threads (mma) + 128 threads (epilog) to sync
  cutlass::arch::NamedBarrier tmem_allocation_result_barrier(
      32 + 128,
      /*bar-id=*/cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
  cutlass::arch::NamedBarrier epilogue_wg_barrier(
      128, /*bar-id=*/cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

  // Represent the SMEM buffers for A and B
  cute::Tensor tCsA =
      shared_storage.tensor_sA(); // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  cute::Tensor tCsB =
      shared_storage.tensor_sB(); // (MmaB, NumMma_M, NumMma_K, Tiles_K)
  cute::Tensor sB = shared_storage.tensor_cp_sB(); // Layout for cp_async

  //
  // Mma partitioning for A and B
  //

  auto mma_v = cute::get<0>(mma_coord_vmnk);
  cute::ThrMMA cta_mma = tiled_mma.get_slice(mma_v); // Use Peer CTA coordinate
  cute::Tensor tCgA =
      cta_mma.partition_A(gA); // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  cute::Tensor tCgB =
      cta_mma.partition_B(gB); // (MmaB, NumMma_N, NumMma_K, Tiles_K)

  //   if (cute::thread0()) {
  //     cute::print("tCgA:\t");
  //     cute::print(tCgA);
  //     cute::print("\n");
  //     cute::print("tCgB:\t");
  //     cute::print(tCgB);
  //     cute::print("\n");
  //     cute::print("tCsA:\t");
  //     cute::print(tCsA);
  //     cute::print("\n");
  //     cute::print("tCsB:\t");
  //     cute::print(tCsB);
  //     cute::print("\n");
  //     cute::print("sB:\t");
  //     cute::print(sB);
  //     cute::print("\n");
  //     cute::print("\n");
  //   }
  //   __syncthreads();

  int tma_transaction_bytes_A =
      sizeof(T_) * cute::size<0>(mma_tiler) * cute::size<2>(mma_tiler);

  //   if (cute::thread0()) {
  //     printf("tma_transaction_bytes_A: %d\n", tma_transaction_bytes_A);
  //   }
  //   __syncthreads();

  constexpr int TILE_SIZE = 64;
  constexpr int WEIGHT_TMA_TILE_SIZE = 64;
  constexpr int OUTPUT_ATOM_SIZE = 128;
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int cp_async_group_size = 32 / MMA_N;

  T_ *shared_weight = shared_storage.A.begin();

  Barrier *a_full_mbar_ptr =
      reinterpret_cast<Barrier *>(shared_storage.a_full_mbar_ptr);

  using WeightSmem = smem_tma<T_,
                              B,
                              M,
                              S,
                              OUTPUT_ATOM_SIZE,
                              WEIGHT_TMA_TILE_SIZE,
                              1>; // 64/64 = 1
  WeightSmem weight_smem(shared_weight);

  // CP_ASYNC Atom for B, todo: use MMA_N instead of hardcoded 16
  cute::TiledCopy copyB = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, T_>{},
      Layout<Shape<Int<MMA_N>, Int<cp_async_group_size>>,
             Stride<Int<cp_async_group_size>, _1>>{}, // Thr layout
      Layout<Shape<_1, _8>>{});                       // Val layout

  //   if (cute::thread0()) {
  //     cute::print_latex(copyB);
  //   } __syncthreads();

  // // check tma descriptor:
  // if(threadIdx.x == 0){
  //     printf("smem start addr: %u\n",
  //     cute::cast_smem_ptr_to_uint(shared_memory)); printf("shared_weight:
  //     %u\n", cute::cast_smem_ptr_to_uint(shared_weight));
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
  __syncthreads();

  //   if (cute::thread0()) {
  //     cute::print("tCrA:\t");
  //     cute::print(tCrA);
  //     cute::print("\n"); // tCrA:   UMMA::DescriptorIterator o
  //     (_1,_1,_4):(_0,_0,_2) cute::print("tCrB:\t"); cute::print(tCrB);
  //     cute::print("\n"); // tCrB:   UMMA::DescriptorIterator o
  //     (_1,_1,_4):(_0,_0,_2) cute::print("tCtAcc:\t"); cute::print(tCtAcc);
  //     cute::print("\n"); // tCtAcc: tmem_[32b](TMEM_ADDR) o
  //                        // ((_128,_256),_1,_1):((_65536,_1),_0,_0)
  //   }
  //   __syncthreads();

  int k_tile_count = cute::size<4>(tCgA);
  int num_activated_experts =
      mMask(NUM_EXPERTS); // last element stores num activated experts

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  // // fetch expert mask and preprocessing
  // int32_t activated_expert_idx[(BATCH_SIZE * NUM_TOPK + EXPERT_STRIDE - 1) /
  // EXPERT_STRIDE]; int32_t total_activated_experts = 0; int32_t
  // num_activated_experts = 0;

  // if(threadIdx.x < NUM_EXPERTS) {
  //   shared_storage.expert_mask[threadIdx.x] = mMask[threadIdx.x];
  // }

  // __syncthreads();

  // for(int expert_idx = 0; expert_idx < NUM_EXPERTS; ++expert_idx) {
  //   int32_t expert_mask = shared_storage.expert_mask[expert_idx];
  //   if (expert_mask == 1 && (total_activated_experts) % EXPERT_STRIDE ==
  //   expert_offset) {
  //     activated_expert_idx[num_activated_experts] = expert_idx;
  //     num_activated_experts += 1;
  //   }
  //   total_activated_experts += expert_mask;
  // }
  __syncthreads(); // Wait for preprocessing done

  if (warp_idx == 5) {
    // DMA warp (1)

    const uint32_t lane_idx = cutlass::canonical_lane_idx();
    cute::ThrCopy thr_copy_b = copyB.get_slice(lane_idx);
    cute::Tensor tBgB = thr_copy_b.partition_S(gB); // (ThrB, ThrTile_N)
    cute::Tensor tBsB = thr_copy_b.partition_D(sB); // (ThrB, ThrTile_N)
    // if (lane_idx == 0) {
    //   cute::print("thr_copy_b:\t");
    //   cute::print(thr_copy_b);
    //   cute::print("\n");
    //   cute::print("tBgB:\t");
    //   cute::print(tBgB);
    //   cute::print("\n");
    //   cute::print("tBsB:\t");
    //   cute::print(tBsB);
    //   cute::print("\n");
    // } __syncwarp();

    int total_k_tile_count = 0;
    for (int activated_expert_offset = expert_offset;
         activated_expert_offset < num_activated_experts;
         activated_expert_offset += EXPERT_STRIDE) {
      int32_t expert_idx = mMask[activated_expert_offset];
      cute::Tensor tRoutingIndex = mRoutingIndices(expert_idx, cute::_);
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
              cute::wait_barrier(
                  shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                  tma_wr_ab_empty_phase);
            }

            // TMA for loading A
            if (cute::elect_one_sync()) {
              // TODO(Zhihao): add expert offset here
              int tma_coords_A[2] = {k_tile * TILE_SIZE,
                                     m_tile * OUTPUT_ATOM_SIZE +
                                         expert_idx * ORIG_OUTPUT_SIZE};
              weight_smem.set_ptr(shared_weight + smem_wr_buffer *
                                                      OUTPUT_ATOM_SIZE *
                                                      TILE_SIZE);
              cute::set_barrier_transaction_bytes(
                  shared_storage.a_full_mbar_ptr[smem_wr_buffer],
                  tma_transaction_bytes_A);
              tma_a.tma_cp_async(a_full_mbar_ptr[smem_wr_buffer],
                                 weight_smem.base_ptr,
                                 tma_coords_A);
            }

            // CP_ASYNC for loading B
            int32_t token_idx = n_tile * MMA_N + lane_idx / cp_async_group_size;
            int32_t topk_idx = tRoutingIndex(token_idx);
            if (token_idx < BATCH_SIZE && topk_idx > 0) {
              if constexpr (W13_LINEAR) {
                cute::copy(copyB,
                           tBgB(_, _, _, _, k_tile),
                           tBsB(_, _, _, _, smem_wr_buffer));
              } else {
                cute::copy(copyB,
                           tBgB(_, _, _, _, k_tile, topk_idx - 1),
                           tBsB(_, _, _, _, smem_wr_buffer));
              }
            }

            cutlass::arch::cpasync_barrier_arrive_noinc(
                &shared_storage.b_full_mbar_ptr[smem_wr_buffer]);

            if (tma_wr_k_tile_next < k_tile_count) {
              peek_ab_empty_status = kernel::try_wait_barrier(
                  shared_storage.ab_empty_mbar_ptr[smem_wr_buffer_next],
                  tma_wr_ab_empty_phase_next);
            }

            tma_wr_k_tile = tma_wr_k_tile_next;
            smem_wr_buffer = smem_wr_buffer_next;
            tma_wr_ab_empty_phase = tma_wr_ab_empty_phase_next;

          } // end for k_tile
        }   // end for n_tile
      }     // end for m_tile
    }       // end for expert_idx
  } else if (warp_idx == 4) {
    // MMA warp (1)

    // Wait for TMEM allocation to complete
    tmem_allocation_result_barrier.arrive_and_wait();
    tCtAcc.data() = shared_storage.tmem_base_ptr;

    int total_k_tile_count = 0;
    int num_tiles_executed = 0;
    for (int activated_expert_offset = expert_offset;
         activated_expert_offset < num_activated_experts;
         activated_expert_offset += EXPERT_STRIDE) {
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
          bool peek_a_full_status = kernel::try_wait_barrier(
              shared_storage.a_full_mbar_ptr[smem_rd_buffer],
              mma_rd_ab_full_phase);
          bool peek_b_full_status = kernel::try_wait_barrier(
              shared_storage.b_full_mbar_ptr[smem_rd_buffer],
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

            if (!peek_a_full_status) {
              cute::wait_barrier(shared_storage.a_full_mbar_ptr[smem_rd_buffer],
                                 mma_rd_ab_full_phase);
            }

            if (!peek_b_full_status) {
              cute::wait_barrier(shared_storage.b_full_mbar_ptr[smem_rd_buffer],
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
              peek_a_full_status = kernel::try_wait_barrier(
                  shared_storage.a_full_mbar_ptr[smem_rd_buffer_next],
                  mma_rd_ab_full_phase_next);
              peek_b_full_status = kernel::try_wait_barrier(
                  shared_storage.b_full_mbar_ptr[smem_rd_buffer_next],
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
      }   // end for m_tile
    }     // end for expert_idx
  } else if (warp_idx < 4) {
    // Epilogue warps (4)

    // Allocate TMEM for accumulators
    if (warp_idx == 0) {
      tmem_allocator.allocate(num_tmem_columns, &shared_storage.tmem_base_ptr);
    }
    tmem_allocation_result_barrier.arrive_and_wait();
    tCtAcc.data() = shared_storage.tmem_base_ptr;

    using TypeBias = T_;
    using TypeC = T_;

    cutlass::NumericConverter<AccType, TypeBias> converterBias;
    cutlass::NumericConverter<TypeC, AccType> converter;

    cute::TiledCopy tiled_copy_t2r =
        cute::make_tmem_copy(cute::SM100_TMEM_LOAD_32dp32b1x{},
                             tCtAcc(cute::_, cute::_, cute::_, 0));
    cute::ThrCopy thr_copy_t2r = tiled_copy_t2r.get_slice(threadIdx.x);
    cute::Tensor tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc);

    cute::Tensor tCgC_fake = cute::make_tensor<TypeC>(cute::shape(
        tCtAcc(cute::_, cute::_, cute::_, 0))); // (T2R, T2R_M, T2R_N)
    cute::Tensor tTR_rAcc_fake = thr_copy_t2r.partition_D(tCgC_fake);
    cute::Tensor tTR_rAcc =
        cute::make_tensor<AccType>(cute::shape(tTR_rAcc_fake));

    // if(threadIdx.x == 0) {
    //   cute::print("tiled_copy_t2r:\t");
    //   cute::print(tiled_copy_t2r);
    //   cute::print("\n");
    //   cute::print("thr_copy_t2r:\t");
    //   cute::print(thr_copy_t2r);
    //   cute::print("\n");
    //   cute::print("tTR_tAcc:\t");
    //   cute::print(tTR_tAcc);
    //   cute::print("\n");
    //   cute::print("tTR_rAcc:\t");
    //   cute::print(tTR_rAcc);
    //   cute::print("\n");
    //   cute::print("tCtAcc:\t");
    //   cute::print(tCtAcc);
    //   cute::print("\n");
    //   printf("tmem_base_ptr: %u\n", shared_storage.tmem_base_ptr);
    // } epilogue_wg_barrier.arrive_and_wait();

    int num_tiles_executed = 0;
    for (int activated_expert_offset = expert_offset;
         activated_expert_offset < num_activated_experts;
         activated_expert_offset += EXPERT_STRIDE) {
      int32_t expert_idx = mMask[activated_expert_offset];
      cute::Tensor tRoutingIndex = mRoutingIndices(expert_idx, cute::_);
      for (int m_tile = 0; m_tile < cute::size<3>(tCgA); ++m_tile) {
        for (int n_tile = 0; n_tile < cute::size<3>(tCgB); ++n_tile) {
          int acc_buf_idx = num_tiles_executed % NUM_ACC_STAGE;
          int acc_full_phase = num_tiles_executed / NUM_ACC_STAGE % 2;
          // int c_smem_wr_buffer_idx = num_tiles_executed % NUM_C_STAGE;

          cute::Tensor tCgBias = gBias(
              cute::_, cute::_, n_tile, m_tile, expert_idx); // (Mma_M, Mma_N)
          cute::Tensor tCrBiasTypeBias = cute::make_tensor<TypeBias>(
              cute::shape(tTR_rAcc(0, cute::_, 0, 0))); // (T2R_M, T2R_N)
          cute::Tensor tCrBiasTypeAcc =
              cute::make_tensor<AccType>(cute::shape(tCrBiasTypeBias));
          cute::Tensor tCrC =
              cute::make_tensor<TypeC>(cute::shape(tCrBiasTypeBias));

          // if(threadIdx.x == 0 and m_tile == 0 and n_tile == 0) {
          //   cute::print("tCgBias:\t");
          //   cute::print(tCgBias);
          //   cute::print("\n");
          //   cute::print("tCrBiasTypeBias:\t");
          //   cute::print(tCrBiasTypeBias);
          //   cute::print("\n"); //
          //   cute::print("tCrBiasTypeAcc:\t");
          //   cute::print(tCrBiasTypeAcc);
          //   cute::print("\n"); //
          //   cute::print("tCrC:\t");
          //   cute::print(tCrC);
          //   cute::print("\n");
          // } epilogue_wg_barrier.arrive_and_wait();

          // T2R and register operations
          if constexpr (!NOBIAS) {
            // this copy might conflict with TMA load, might add a wait
            // barrier if needed
            cute::copy(tCgBias(cute::_, threadIdx.x), tCrBiasTypeBias);
            // optimize with vectorized type conversion

            CUTE_UNROLL
            for (int i = 0; i < tCrBiasTypeBias.size(); i++) {
              tCrBiasTypeAcc[i] = converterBias(tCrBiasTypeBias[i]);
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
            cute::arrive_barrier(
                shared_storage.acc_empty_mbar_ptr[acc_buf_idx]);
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

          // R2G store, use cp.async.bulk here
          CUTE_UNROLL
          for (int i = 0; i < MMA_N; ++i) {
            int32_t m_idx = m_tile * MMA_M + threadIdx.x;
            int32_t n_idx = n_tile * MMA_N + i;
            int32_t topk_idx = tRoutingIndex(n_idx);
            if (n_idx < BATCH_SIZE && topk_idx > 0) {
              mOutput(n_idx, topk_idx - 1, m_idx) = tCrC[i];
            }
          }
          epilogue_wg_barrier.arrive_and_wait();

          num_tiles_executed++;
        } // end for n_tile
      }   // end for m_tile
    }     // end for expert_idx
  }       // end of epilogue warps
  __syncthreads();

  if (warp_idx == 0) {
    // don't do relinquish for megakernel
    // tmem_allocator.release_allocation_lock();
    tmem_allocator.free(shared_storage.tmem_base_ptr, num_tmem_columns);
  }

} // end moe_linear_sm100_task_impl

} // namespace kernel
