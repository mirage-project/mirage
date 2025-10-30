#pragma once
#include <cstdio>
#include <iostream>

// Cutlass includes
#include <cutlass/half.h> // F16 data type
// #include <cutlass/util/print_error.hpp>
#include "cutlass/gemm/collective/collective_builder.hpp"
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
// CuTe includes
#include <cute/algorithm/cooperative_copy.hpp> // Auto vectorized copy operation
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/cluster_sm90.hpp> // CuTe functions for querying the details of cluster launched
#include <cute/arch/copy_sm80.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/numeric/integral_constant.hpp> // Compile time in constants such as _1, _256 etc.
#include <cute/tensor.hpp>                    // CuTe tensor implementation

#include "../common/dmem_layout.cuh"
#include "../common/utils.cuh"
#include "../common/worker_config.h"
#include "barrier.cuh"
#include "smem_layout_tma.cuh"
#include "tma.cuh"
#include "utils.cuh"
namespace kernel {

// MoE Linear task storage. The shared memory buffers for A, B, and C matrices.
template <class TypeA, // Tensor A data type
          class TypeB, // Tensor B data type
          class ASmemLayout,
          class BSmemLayout,
          class BSmemCpLayout,
          int Num_AB_Stage>
struct MoESharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;

  alignas(16) cute::uint64_t a_full_mbar_ptr[Num_AB_Stage];
  alignas(16) cute::uint64_t b_full_mbar_ptr[Num_AB_Stage];
  alignas(16) cute::uint64_t ab_empty_mbar_ptr[Num_AB_Stage];

  CUTE_DEVICE constexpr auto tensor_sA() {
    return cute::make_tensor(cute::make_smem_ptr(A.begin()), ASmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sB() {
    return cute::make_tensor(cute::make_smem_ptr(B.begin()), BSmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_cp_sB() {
    return cute::make_tensor(cute::make_smem_ptr(B.begin()), BSmemCpLayout{});
  }
};

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
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int EXPERT_STRIDE,
          bool W13_LINEAR,
          bool NOBIAS,
          int NUM_AB_STAGE = 8>
__device__ __forceinline__ void
    moe_linear_sm90_task_impl(const TMA_A &tma_a,
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

  using TypeAcc = float;
  constexpr int NUM_THREAD_PER_WARPGROUP = 128;
  constexpr int CONSUMER_SYNC_BARRIER_ID = 5;

  // Define TileShape and AtomLayout
  constexpr int MMA_K = 64; // 16*4
  using TileShape_MNK = decltype(cute::make_shape(
      cute::Int<MMA_M>{}, cute::Int<MMA_N>{}, cute::Int<MMA_K>{}));
  using AtomLayoutMNK = cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>;

  // Create TiledMma type and instance
  auto tiled_mma =
      cute::make_tiled_mma(cute::GMMA::ss_op_selector<T_,
                                                      T_,
                                                      TypeAcc,
                                                      TileShape_MNK,
                                                      cute::GMMA::Major::K,
                                                      cute::GMMA::Major::K>(),
                           AtomLayoutMNK{});

  auto bM = cute::tile_size<0>(
      tiled_mma); // MMA Tile M. We'll use 1 MMAs per MMA Tile M.
  auto bN = cute::tile_size<1>(
      tiled_mma); // MMA Tile N. We'll use 1 MMAs per MMA Tile N.
  auto bK = cute::tile_size<2>(tiled_mma) *
            cute::Int<4>{}; // MMA Tile K. We'll use 4 MMAs per MMA Tile K. For
                            // 16b types, wgmma has K16.

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
  auto output_tiler =
      cute::make_shape(bN, bM, NUM_TOPK); // (MmaTile_N, MmaTile_M, NUM_TOPK)

  cute::Tensor mA = cute::make_coord_tensor(cute::make_layout(
      cute::make_shape(OUTPUT_SIZE, REDUCTION_SIZE, NUM_EXPERTS),
      cute::make_stride(cute::E<1>{},
                        cute::E<0>{},
                        cute::E<1>{} * cute::Int<OUTPUT_SIZE>{})));

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

  cute::Tensor gOutput =
      cute::local_tile(mOutput,
                       output_tiler,
                       mma_coord,
                       cute::Step<cute::_1, cute::_1, cute::X>{});
  // cute::Tensor tCgD = thread_mma.partition_C(gOutputTile);
#if 0
  if (cute::thread0()) {
    cute::print("mA:\t");
    cute::print(mA);
    cute::print("\n");
    cute::print("mInput:\t");
    cute::print(mInput);
    cute::print("\n");
    cute::print("mBias:\t");
    cute::print(mBias);
    cute::print("\n");
    cute::print("mOutput:\t");
    cute::print(mOutput);
    cute::print("\n");

    cute::print("gA:\t");
    cute::print(gA);
    cute::print("\n");
    cute::print("gB:\t");
    cute::print(gB);
    cute::print("\n");
    cute::print("gBias:\t");
    cute::print(gBias);
    cute::print("\n");

    printf("gOutput:\t");
    cute::print(gOutput);
    printf("\n");
  }
  __syncthreads();
#endif

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
  // auto mma_shape_C =
  //     cute::make_shape(cute::make_shape(cute::Int<MMA_N>{},
  //     cute::Int<MMA_M>{}),
  //                      cute::Int<1>{},
  //                      cute::Int<1>{},
  //                      cute::Int<NUM_C_STAGE>{});

  using SmemLayoutAtomA =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               cute::GMMA::Major::K,
               T_,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());

  using SmemLayoutA = decltype(cute::tile_to_shape(
      SmemLayoutAtomA{},
      cute::make_shape(cute::shape<0>(TileShape_MNK{}),
                       cute::shape<2>(TileShape_MNK{}),
                       cute::Int<NUM_AB_STAGE>{}),
      cute::Step<cute::_1, cute::_2, cute::_3>{}));

#if 0
    if (threadIdx.x == 0) {
        printf("mma_tiler:\t");
        cute::print(mma_tiler);
        printf("\n");
        cute::print("smem layout atom a:\t");
        cute::print(SmemLayoutAtomA{});
        cute::print("\n");
        cute::print("smem layout a:\t");
        cute::print(SmemLayoutA{});
        cute::print("\n");
        cute::print("mma_shape_A:\t");
        cute::print(mma_shape_A);
        cute::print("\n");
        cute::print("mma_shape_B:\t");
        cute::print(mma_shape_B);
        cute::print("\n");
    }
#endif

  using SmemLayoutAtomB =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               cute::GMMA::Major::K,
               T_,
               decltype(cute::get<1>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());

  using SmemLayoutB = decltype(cute::tile_to_shape(
      SmemLayoutAtomB{},
      cute::make_shape(cute::shape<1>(TileShape_MNK{}),
                       cute::shape<2>(TileShape_MNK{}),
                       cute::Int<NUM_AB_STAGE>{}),
      cute::Step<cute::_1, cute::_2, cute::_3>{}));

  SmemLayoutA sA_layout;
  SmemLayoutB sB_layout;

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

  // using SmemLayoutAtomC =
  // decltype(cutlass::gemm::collective::detail::ss_smem_selector<
  //          cute::GMMA::Major::K,
  //          T_,
  //          decltype(cute::get<0>(TileShape_MNK{})),
  //          decltype(cute::get<1>(TileShape_MNK{}))>());

  // using SmemLayoutC = decltype(cute::tile_to_shape(
  //           SmemLayoutAtomC{},
  //           cute::make_shape(cute::shape<0>(TileShape_MNK{}),
  //                      cute::shape<1>(TileShape_MNK{})),
  //                      cute::Step<cute::_1, cute::_2>{}));
  //     SmemLayoutC sC_layout;

#if 0
if (threadIdx.x == 0) {
    cute::print("smem layout atom c:\t");
    cute::print(SmemLayoutAtomC{});
    cute::print("\n");
    cute::print("smem layout c:\t");
    cute::print(SmemLayoutC{});
    cute::print("\n");
    printf("sA_layout: ");
    cute::print(sA_layout);
    printf("\n");
    printf("sB_layout: ");
    cute::print(sB_layout);
    printf("\n");
    printf("sB_cp_layout: ");
    cute::print(sB_cp_layout);
    printf("\n");
}
#endif

  using SharedStorage = MoESharedStorage<T_,
                                         T_,
                                         decltype(sA_layout),
                                         decltype(sB_layout),
                                         decltype(sB_cp_layout),
                                         NUM_AB_STAGE>;

  extern __shared__ char shared_memory[];
  uintptr_t aligned_smem =
      (reinterpret_cast<uintptr_t>(shared_memory) + 127) / 128 * 128;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(aligned_smem);
#if 0
  T_ *shared_output =
      (T_ *)(((uintptr_t)aligned_smem + sizeof(SharedStorage) + 1023) / 1024 *
             1024);
  using OutputSmem = smem_tma<T_, 0, 0, 0, 16, 64, 1>;
  OutputSmem output_smem(shared_output);
#endif
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
                      128);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        NUM_AB_STAGE>(shared_storage.ab_empty_mbar_ptr,
                      /* arrival count
                       */
                      1);
  }

  // Represent the SMEM buffers for A and B
  cute::Tensor sA =
      shared_storage.tensor_sA(); // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  cute::Tensor sB =
      shared_storage.tensor_cp_sB(); // (MmaB, NumMma_M, NumMma_K, Tiles_K)
  constexpr int MmaWarpGroups = size(tiled_mma) / NUM_THREAD_PER_WARPGROUP;
  cute::Layout warp_group_thread_layout = cute::make_layout(
      cute::Int<MmaWarpGroups>{}, cute::Int<NUM_THREAD_PER_WARPGROUP>{});
  int warp_group_idx =
      __shfl_sync(0xFFFFFFFF, threadIdx.x / NUM_THREAD_PER_WARPGROUP, 0);
  auto thread_mma =
      tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));
  cute::Tensor tCsA = thread_mma.partition_A(
      shared_storage.tensor_sA()); // (MMA,MMA_M,MMA_K,PIPE)
  cute::Tensor tCsB = thread_mma.partition_B(
      shared_storage.tensor_sB()); // (MMA,MMA_N,MMA_K,PIPE)

#if 0
    if (threadIdx.x == 0) {
      cute::print("mma warp groups:\t");
      cute::print(MmaWarpGroups);
      printf("\n");
      cute::print(warp_group_thread_layout(warp_group_idx));
      cute::print("\n");
      printf("sA: ");
      cute::print(sA);
      printf("\n");
      printf("sB: ");
      cute::print(sB);
      printf("\n");
      printf("tCsA: ");
      cute::print(tCsA);
      printf("\n");
      printf("tCsB: ");
      cute::print(tCsB);
      printf("\n");
    }
    __syncthreads();
#endif

  int tma_transaction_bytes_A =
      sizeof(T_) * cute::size<0>(mma_tiler) * cute::size<2>(mma_tiler);

  constexpr int TILE_SIZE = 64;
  constexpr int WEIGHT_TMA_TILE_SIZE = 64;
  constexpr int OUTPUT_ATOM_SIZE = 64;
  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;
  constexpr int cp_async_group_size =
      128 / MMA_N; // we use a whole wg for hopper cp_async

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

  // CP_ASYNC Atom for B
  auto copyB = cute::make_tiled_copy(
      cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, T_>{},
      cute::Layout<
          cute::Shape<cute::Int<MMA_N>, cute::Int<cp_async_group_size>>,
          cute::Stride<cute::Int<cp_async_group_size>, cute::_1>>{}, // Thr
                                                                     // layout
      cute::Layout<cute::Shape<cute::_1, cute::_8>>{}); // Val layout

  cute::Tensor tCrA =
      thread_mma.make_fragment_A(tCsA); // (MmaA, NumMma_M, NumMma_K, Tiles_K)
  cute::Tensor tCrB =
      thread_mma.make_fragment_B(tCsB); // (MmaB, NumMma_M, NumMma_K, Tiles_K)
#if 0
    if (threadIdx.x == 0) {
      printf("tCrA: ");
      cute::print(tCrA);
      printf("\n");
      printf("tCrB: ");
      cute::print(tCrB);
      printf("\n");
    }
#endif

  int k_tile_count = REDUCTION_SIZE / 64;

  if (warp_idx >= 4) {
    // DMA warp (4)

    const uint32_t lane_idx = cutlass::canonical_lane_idx();
    // cute::ThrCopy thr_copy_b = copyB.get_slice(lane_idx);
    const uint32_t tid_in_wg = threadIdx.x % NUM_THREAD_PER_WARPGROUP;
    cute::ThrCopy thr_copy_b = copyB.get_slice(tid_in_wg);
    cute::Tensor tBgB = thr_copy_b.partition_S(gB); // (ThrB, ThrTile_N)
    cute::Tensor tBsB = thr_copy_b.partition_D(
        sB); // (ThrB, ThrTile_N) NOTE(Yu): use cp_sb layout here
#if 0
    if (threadIdx.x == 128) {
      printf("cp_async_group_size: %d\n", cp_async_group_size);
      printf("sB:\t");
      cute::print(shared_storage.tensor_sB());
      printf("\n");
      printf("sB_cp_layout: ");
      cute::print(shared_storage.tensor_cp_sB());
      printf("\n");
      cute::print("copyB:\t");
      cute::print(copyB);
      cute::print("\n");
      cute::print("thr_copy_b:\t");
      cute::print(thr_copy_b);
      cute::print("\n");
      cute::print("tBgB:\t");
      cute::print(tBgB);
      cute::print("\n");

    } 
    __syncwarp();
#endif

    int total_k_tile_count = 0;
    int total_expert_count = 0;
    for (int expert_idx = 0; expert_idx < NUM_EXPERTS; ++expert_idx) {
      total_expert_count += mMask(expert_idx);
      if (mMask(expert_idx) == 1 &&
          (total_expert_count - 1) % EXPERT_STRIDE == expert_offset) {
        cute::Tensor tRoutingIndex = mRoutingIndices(expert_idx, cute::_);
        for (int m_tile = 0; m_tile < cute::size<2>(gA); ++m_tile) {
          for (int n_tile = 0; n_tile < cute::size<2>(gB); ++n_tile) {
            int num_prev_k_blk = total_k_tile_count;
            total_k_tile_count += k_tile_count;

            int tma_wr_k_tile = 0;
            int smem_wr_buffer =
                (num_prev_k_blk + tma_wr_k_tile) % NUM_AB_STAGE;
            int tma_wr_ab_empty_phase =
                (num_prev_k_blk + tma_wr_k_tile) / NUM_AB_STAGE % 2 ^ 1;

            bool peek_ab_empty_status = kernel::try_wait_barrier(
                shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                tma_wr_ab_empty_phase);

            CUTE_UNROLL
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
              // only one thread will issue the tma instruction
              if (threadIdx.x % NUM_THREAD_PER_WARPGROUP == 0) {
                int tma_coords_A[2] = {k_tile * TILE_SIZE,
                                       m_tile * OUTPUT_ATOM_SIZE +
                                           expert_idx * OUTPUT_SIZE};
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

              // int32_t token_idx = n_tile * MMA_N + lane_idx /
              // cp_async_group_size;
              int32_t token_idx =
                  n_tile * MMA_N + tid_in_wg / cp_async_group_size;
              int32_t topk_idx = tRoutingIndex(token_idx);
              if (token_idx < BATCH_SIZE && topk_idx > 0) {
                if constexpr (W13_LINEAR) {
                  cute::copy(
                      copyB,
                      tBgB(cute::_, cute::_, cute::_, cute::_, k_tile),
                      tBsB(cute::_, cute::_, cute::_, cute::_, smem_wr_buffer));
                } else {
                  cute::copy(
                      copyB,
                      tBgB(cute::_,
                           cute::_,
                           cute::_,
                           cute::_,
                           k_tile,
                           topk_idx - 1),
                      tBsB(cute::_, cute::_, cute::_, cute::_, smem_wr_buffer));
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
      }       // end if mask
    }         // end for expert_idx
  } else if (warp_idx < 4) {
    // MMA warp (4)

    int total_k_tile_count = 0;
    int num_tiles_executed = 0;
    int total_expert_count = 0;

    for (int expert_idx = 0; expert_idx < NUM_EXPERTS; ++expert_idx) {
      total_expert_count += mMask(expert_idx);
      cute::Tensor tRoutingIndex = mRoutingIndices(expert_idx, cute::_);
      if (mMask(expert_idx) == 1 &&
          (total_expert_count - 1) % EXPERT_STRIDE == expert_offset) {
        for (int m_tile = 0; m_tile < cute::size<2>(gA); ++m_tile) {
          for (int n_tile = 0; n_tile < cute::size<2>(gB); ++n_tile) {

            int num_prev_k_blk = total_k_tile_count;
            total_k_tile_count += k_tile_count;

            int mma_rd_k_tile = 0;
            int smem_rd_buffer =
                (num_prev_k_blk + mma_rd_k_tile) % NUM_AB_STAGE;
            int mma_rd_ab_full_phase =
                (num_prev_k_blk + mma_rd_k_tile) / NUM_AB_STAGE % 2;

            // Peek full phase
            bool peek_a_full_status = kernel::try_wait_barrier(
                shared_storage.a_full_mbar_ptr[smem_rd_buffer],
                mma_rd_ab_full_phase);
            bool peek_b_full_status = kernel::try_wait_barrier(
                shared_storage.b_full_mbar_ptr[smem_rd_buffer],
                mma_rd_ab_full_phase);

            cute::Tensor accum = cute::partition_fragment_C(
                tiled_mma, cute::take<0, 2>(mma_tiler)); // (MMA,MMA_M,MMA_N)
            // Clear the accumulator
            cute::clear(accum);
            // Initialize the accumulator to zero
            tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;

            for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
              int mma_rd_k_tile_next = mma_rd_k_tile + 1;
              int smem_rd_buffer_next =
                  (num_prev_k_blk + mma_rd_k_tile_next) % NUM_AB_STAGE;
              int mma_rd_ab_full_phase_next = smem_rd_buffer_next == 0
                                                  ? mma_rd_ab_full_phase ^ 1
                                                  : mma_rd_ab_full_phase;

              if (!peek_a_full_status) {
                cute::wait_barrier(
                    shared_storage.a_full_mbar_ptr[smem_rd_buffer],
                    mma_rd_ab_full_phase);
              }

              if (!peek_b_full_status) {
                cute::wait_barrier(
                    shared_storage.b_full_mbar_ptr[smem_rd_buffer],
                    mma_rd_ab_full_phase);
              }
#if 0
                        if (threadIdx.x == 0) {
                          printf("m_tile: %d, n_tile: %d, k_tile: %d, smem_rd_buffer: %d\n", m_tile, n_tile, k_tile, smem_rd_buffer);
                          for (int p = 0; p < 4; p++) {
                            auto tCsA_tile = tCsA(cute::_, cute::_, p, smem_rd_buffer);
                            printf("======== p = %d, n = %d, tCsA_tile: ========\n", p, n_tile);
                            cute::print(tCsA_tile);
                            printf("\n");
                            auto start_ptr = tCsA_tile.data();
                            printf("p = %d, start_ptr: %p\n", p, start_ptr);
                            printf("p = %d, tCsA_tile raw data: \n", p);
                            for (int i = 0; i < 64; i++) {
                              for (int j = 0; j < 16; j++) {
                                printf("%f ", static_cast<float>(*(start_ptr + i * 64 + j)));
                              }
                              printf("\n");
                            }
                            printf("\n");

                            auto tCsB_tile = tCsB(cute::_, cute::_, p, smem_rd_buffer);
                            printf("======== p = %d, n = %d, tCsB_tile: ========\n", p, n_tile);
                            cute::print(tCsB_tile);
                            printf("\n");
                            auto start_ptr_b = tCsB_tile.data();
                            cute::print(start_ptr_b);
                            printf("\n");
                            printf("start_ptr_b: %p\n", start_ptr_b);
                            printf("\n");

                            printf("p = %d, start_ptr_b: %p\n", p, start_ptr_b);
                            printf("p = %d, tCsB_tile raw data: \n", p);
                            for (int i = 0; i < 16; i++) {
                              // printf("start ptr_b[%d]: %p\n", i, start_ptr_b + i * 64);
                              for (int j = 0; j < 16; j++) {
                                printf("%f ", static_cast<float>(*(start_ptr_b + i * 64 + j)));
                              }
                              printf("\n");
                            }
                            printf("\n");
                            cute::print_tensor(tCrA(cute::_, cute::_, 0, smem_rd_buffer));
                          }
                        }
#endif

              cute::warpgroup_fence_operand(accum);
              {
                // Perform MMA operation
                cute::warpgroup_arrive();
                for (int k_block = 0; k_block < cute::size<2>(tCrA);
                     ++k_block) {
                  cute::gemm(tiled_mma,
                             tCrA(cute::_, cute::_, k_block, smem_rd_buffer),
                             tCrB(cute::_, cute::_, k_block, smem_rd_buffer),
                             accum);
                  tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
                }
                cute::warpgroup_commit_batch();
                cute::warpgroup_wait<0>();
              }
              cute::warpgroup_fence_operand(accum);

              // only one thread will issue the arrive instruction
              if (threadIdx.x % NUM_THREAD_PER_WARPGROUP == 0) {
                cute::arrive_barrier(
                    shared_storage.ab_empty_mbar_ptr[smem_rd_buffer]);
              }
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

            // Epilogue
            using TypeBias = T_;
            using TypeC = T_;
            cutlass::NumericConverter<TypeAcc, TypeBias> TypeBias_to_TypeAcc;
            cutlass::NumericConverter<TypeC, TypeAcc> TypeAcc_to_TypeC;

#if 0
            cute::Tensor gBiasTile = gBias(
                cute::_, cute::_, n_tile, m_tile, expert_idx); // (Mma_M, Mma_N)
              // NOTE(Yu): thread_mma.partition_C may not partition it to correct layout, this should be fixed if we want to use partition provided by cutlass
            auto tCgBias = thread_mma.partition_C(gBiasTile);

            // if constexpr (!NOBIAS) {
            //   CUTE_UNROLL
            //   for (int i = 0; i < tCgBias.size(); i++) {
            //     tCgBias(i) = TypeAcc_to_TypeC(accum(i));
            //   }
            // }
              if (threadIdx.x == 0) {
                printf("gBiasTile: ");
                cute::print(gBiasTile);
                printf("\n");
                printf("tCgBias: ");
                cute::print(tCgBias);
                printf("\n");
                printf("accum: ");
                cute::print(accum);
                printf("\n");

                printf("accum: ");
                for (int i = 0; i < accum.size(); i++) {
                  printf("%f ", static_cast<float>(accum(i)));
                }
                printf("\n");

                if (!NOBIAS) {
                  printf("tCgBias: ");
                  auto start_ptr = tCgBias.data();
                  for (int i = 0; i < tCgBias.size(); i++) {
                    printf("%p %f \n", start_ptr + i, static_cast<float>(tCgBias(i)));
                  }
                  printf("\n");

                  printf("gBiasTile start ptr: %p\n", gBiasTile.data());
                  for (int i = 0; i < gBiasTile.size(); i++) {
                    printf("%f ", static_cast<float>(gBiasTile(i)));
                  }
                  printf("\n");
                }

              }
#endif

            for (int i = 0; i < MMA_N / 2; i++) {
              int const idx_in_warp = threadIdx.x % 32;
              int m_idx =
                  (warp_idx % 4) * 16 + idx_in_warp / 4 + (i % 4 / 2) * 8;
              int n_idx = i / 4 * 8 + idx_in_warp % 4 * 2 + i % 2;

              int topk_idx = tRoutingIndex(n_idx);

              bool pred = (n_idx < BATCH_SIZE && tRoutingIndex(n_idx) > 0 &&
                           m_idx < OUTPUT_SIZE);
              TypeC fragD = TypeAcc_to_TypeC(accum(i));

              if constexpr (!NOBIAS) {
// TypeBias fragC{};
// cutlass::arch::global_load<TypeBias, sizeof(TypeBias)>(
//     fragC, &tCgBias(i), pred);
#if 0
                if (threadIdx.x == 0) {
                  printf("pred: %d, m_idx: %d, n_idx: %d, topk_idx: %d, fragC: %f, fragD: %f, tCgBias(i): %f, &tCgBias(i): %p\n",
                         pred,
                         m_idx,
                         n_idx,
                         topk_idx,
                         static_cast<float>(fragC),
                         static_cast<float>(fragD),
                         static_cast<float>(tCgBias(i)),
                         &tCgBias(i));
                }
#endif
                fragD += mBias(n_idx, m_idx, expert_idx);
              }
              if (pred) {
                mOutput(n_idx, m_idx, tRoutingIndex(n_idx) - 1) = fragD;
              }
            }
            num_tiles_executed++;
          } // end for n_tile
        }   // end for m_tile
      }     // end if mask
    }       // end for expert_idx
  }

} // end moe_linear_sm90_task_impl

} // namespace kernel