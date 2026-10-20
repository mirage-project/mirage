
/* Copyright 2025 CMU
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

// E2M1 values are stored low nibble first; MXFP4 scales use the E8M0
// exponent-only encoding (2^(encoded_exponent - 127)).

// The BF16 pair 2^(scale - 127) for an E8M0 scale; a scale of 0 decodes to 0.
CUTE_DEVICE __nv_bfloat162 mxfp4_scale_pair(uint32_t scale) {
  uint32_t const bits = scale == 0 ? 0u : scale << 7;
  uint32_t const pair = bits | bits << 16;
  return *reinterpret_cast<__nv_bfloat162 const *>(&pair);
}

// Decodes one packed byte, two E2M1 values adjacent along K, to a BF16 pair
// (low nibble in the low half) times `scale2`. A nibble s.ee.m placed as BF16
// bits (s << 15) | (ee.m << 6) is its value times 2^-126 (ee = 0 lands on a
// subnormal), so multiplying by 2^126 and then by the power-of-two scale is
// exact.
CUTE_DEVICE uint32_t decode_mxfp4_pair(uint32_t byte, __nv_bfloat162 scale2) {
  uint32_t const bits = ((byte & 0x08u) << 12) | ((byte & 0x07u) << 6) |
                        ((byte & 0x80u) << 24) | ((byte & 0x70u) << 18);
  uint32_t const unbias = 0x7e807e80u; // 2^126 in both halves
  __nv_bfloat162 const pair =
      __hmul2(__hmul2(*reinterpret_cast<__nv_bfloat162 const *>(&bits),
                      *reinterpret_cast<__nv_bfloat162 const *>(&unbias)),
              scale2);
  return *reinterpret_cast<uint32_t const *>(&pair);
}

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
          int OUTPUT_STRIDE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int EXPERT_STRIDE,
          bool W13_LINEAR,
          bool NOBIAS,
          bool MXFP4_WEIGHT = false,
          int NUM_AB_STAGE = 8>
__device__ __forceinline__ void
    moe_linear_sm90_task_impl(const TMA_A &tma_a,
                              InputTensor mInput,
                              BiasTensor mBias,
                              IndicesTensor mRoutingIndices,
                              MaskTensor mMask,
                              OutputTensor mOutput,
                              int const expert_offset,
                              uint8_t const *weight_scale = nullptr,
                              // the task's first weight row in tma_a's map:
                              // 0 when the map starts at the task's rows (MPK),
                              // its row offset in a whole-tensor map (the
                              // static megakernel)
                              int const weight_row0 = 0) {
  int warp_idx = cutlass::canonical_warp_idx_sync();

  // Construct the MMA grid coordinate from the CTA grid coordinate
  auto mma_coord_vmnk = cute::make_coord(0,        // Peer CTA coordinate
                                         cute::_,  //    MMA-M coordinate
                                         cute::_,  //    MMA-N coordinate
                                         cute::_); //    MMA-K coordinate

  using TypeAcc = float;
  constexpr int NUM_THREAD_PER_WARPGROUP = 128;
  int const idx_in_warp = threadIdx.x & 31;
  // constexpr int CONSUMER_SYNC_BARRIER_ID = 5;

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
  // MXFP4 decodes the weight tile into registers, so its MMA reads A from
  // registers (RS); B stays in shared memory.
  auto tiled_mma_rs =
      cute::make_tiled_mma(cute::GMMA::rs_op_selector<T_,
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
      mA, mma_tiler, mma_coord, cute::Step<cute::_1, cute::X, cute::_1>{});
  cute::Tensor gB = cute::local_tile(
      mInput, mma_tiler, mma_coord, cute::Step<cute::X, cute::_1, cute::_1>{});
  cute::Tensor gBias = cute::local_tile(
      mBias, cd_tiler, mma_coord, cute::Step<cute::_1, cute::_1, cute::X>{});

  cute::Tensor gOutput =
      cute::local_tile(mOutput,
                       output_tiler,
                       mma_coord,
                       cute::Step<cute::_1, cute::_1, cute::X>{});
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

  // MXFP4 tiles land packed (two values per byte) in the stages of A, which
  // hold BF16 tiles on the BF16 path. Past the shared storage, each consumer
  // warp keeps the E8M0 scales of its 16 rows of the current m tile.
  constexpr int MXFP4_SCALE_COLS = REDUCTION_SIZE / 32;
  uint8_t *mxfp4_scale = reinterpret_cast<uint8_t *>(
      (aligned_smem + sizeof(SharedStorage) + 15) / 16 * 16);
  static_assert(!MXFP4_WEIGHT ||
                    sizeof(SharedStorage) + 128 + 16 + 64 * MXFP4_SCALE_COLS <=
                        mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                "MXFP4 scales do not fit in shared memory");
  static_assert(!MXFP4_WEIGHT || (MMA_M == 64 && MMA_K == 64),
                "MXFP4 decodes 64x64 weight tiles");
  // 32-value blocks, two per k tile, so a row's scales come in pairs
  static_assert(!MXFP4_WEIGHT || REDUCTION_SIZE % 64 == 0,
                "MXFP4 needs whole 64-value k tiles");

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

  __syncthreads();

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

  int k_tile_count = REDUCTION_SIZE / 64;
  int num_activated_experts =
      mMask(NUM_EXPERTS); // last element stores num activated experts

  if (warp_idx >= 4) {
    // DMA warp (4)
    uint32_t const tid_in_wg = threadIdx.x % NUM_THREAD_PER_WARPGROUP;
    cute::ThrCopy thr_copy_b = copyB.get_slice(tid_in_wg);
    cute::Tensor tBgB = thr_copy_b.partition_S(gB); // (ThrB, ThrTile_N)
    cute::Tensor tBsB = thr_copy_b.partition_D(
        sB); // (ThrB, ThrTile_N) NOTE(Yu): use cp_sb layout here

    int total_k_tile_count = 0;
#pragma unroll 1
    for (int activated_expert_offset = expert_offset;
         activated_expert_offset < num_activated_experts;
         activated_expert_offset += EXPERT_STRIDE) {
      int32_t expert_idx = mMask[activated_expert_offset];
      cute::Tensor tRoutingIndex = mRoutingIndices(expert_idx, cute::_);
#pragma unroll 1
      for (int m_tile = 0; m_tile < cute::size<2>(gA); ++m_tile) {
#pragma unroll 1
        for (int n_tile = 0; n_tile < cute::size<2>(gB); ++n_tile) {
          int num_prev_k_blk = total_k_tile_count;
          total_k_tile_count += k_tile_count;

          int tma_wr_k_tile = 0;
          int smem_wr_buffer = (num_prev_k_blk + tma_wr_k_tile) % NUM_AB_STAGE;
          int tma_wr_ab_empty_phase =
              (num_prev_k_blk + tma_wr_k_tile) / NUM_AB_STAGE % 2 ^ 1;

          bool peek_ab_empty_status =
              try_wait_barrier(shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                               tma_wr_ab_empty_phase);

#pragma unroll 3
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

            // TMA for loading A; only one thread issues the instruction.
            if (threadIdx.x % NUM_THREAD_PER_WARPGROUP == 0) {
              int const tma_row = weight_row0 + m_tile * OUTPUT_ATOM_SIZE +
                                  expert_idx * OUTPUT_STRIDE;
              weight_smem.set_ptr(shared_weight + smem_wr_buffer *
                                                      OUTPUT_ATOM_SIZE *
                                                      TILE_SIZE);
              if constexpr (MXFP4_WEIGHT) {
                // packed, two values per byte; the consumer decodes it
                int tma_coords_A[2] = {k_tile * TILE_SIZE / 2, tma_row};
                cute::set_barrier_transaction_bytes(
                    shared_storage.a_full_mbar_ptr[smem_wr_buffer],
                    OUTPUT_ATOM_SIZE * TILE_SIZE / 2);
                tma_a.tma_cp_async(
                    a_full_mbar_ptr[smem_wr_buffer],
                    reinterpret_cast<uint8_t *>(weight_smem.base_ptr),
                    tma_coords_A);
              } else {
                int tma_coords_A[2] = {k_tile * TILE_SIZE, tma_row};
                cute::set_barrier_transaction_bytes(
                    shared_storage.a_full_mbar_ptr[smem_wr_buffer],
                    tma_transaction_bytes_A);
                tma_a.tma_cp_async(a_full_mbar_ptr[smem_wr_buffer],
                                   weight_smem.base_ptr,
                                   tma_coords_A);
              }
            }

            int32_t token_idx =
                n_tile * MMA_N + tid_in_wg / cp_async_group_size;
            int32_t topk_idx =
                (token_idx < BATCH_SIZE) ? tRoutingIndex(token_idx) : 0;
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
              peek_ab_empty_status = try_wait_barrier(
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
  } else if (warp_idx < 4) {
    // MMA warp (4)
    // MXFP4: the RS MMA's operands. Register j of k block s of the A
    // fragment holds two adjacent K values of row mxfp4_row (j even) or
    // mxfp4_row + 8 (j odd) at columns 16s + 2q (+ 8 for j >= 2), with
    // q = lane % 4: exactly one packed byte of the weight tile.
    auto thread_mma_rs =
        tiled_mma_rs.get_slice(warp_group_thread_layout(warp_group_idx));
    cute::Tensor tCrA_rs = thread_mma_rs.partition_fragment_A(
        shared_storage.tensor_sA()(cute::_, cute::_, 0)); // (MMA,MMA_M,MMA_K)
    cute::Tensor tCrA_rs_u32 = cute::recast<uint32_t>(tCrA_rs);
    cute::Tensor tCrB_rs = thread_mma_rs.make_fragment_B(
        thread_mma_rs.partition_B(shared_storage.tensor_sB()));
    int const mxfp4_row = ((warp_idx & 3) << 4) + (idx_in_warp >> 2);
    int const mxfp4_byte_shift = 8 * (idx_in_warp & 3);
    uint8_t *warp_scale = mxfp4_scale + (warp_idx & 3) * 16 * MXFP4_SCALE_COLS;
    // This warp's 16 rows of (expert, m_t)'s scales, contiguous in global
    // memory, loaded 2 bytes at a time with every load issued before any
    // store so their latencies overlap.
    constexpr int MXFP4_WARP_SCALE_PAIRS = 16 * MXFP4_SCALE_COLS / 2;
    constexpr int MXFP4_SCALE_PAIRS_PER_LANE =
        (MXFP4_WARP_SCALE_PAIRS + 31) / 32;
    auto load_warp_scales = [&](int expert, int m_t) {
      int const row0 = m_t * OUTPUT_ATOM_SIZE + ((warp_idx & 3) << 4);
      int const rows = max(0, min(16, OUTPUT_SIZE - row0));
      int const pairs = rows * MXFP4_SCALE_COLS / 2;
      uint16_t const *src = reinterpret_cast<uint16_t const *>(
          weight_scale + (static_cast<size_t>(expert) * OUTPUT_STRIDE + row0) *
                             MXFP4_SCALE_COLS);
      uint16_t vals[MXFP4_SCALE_PAIRS_PER_LANE];
#pragma unroll
      for (int i = 0; i < MXFP4_SCALE_PAIRS_PER_LANE; i++) {
        int const idx = idx_in_warp + i * 32;
        vals[i] = idx < pairs ? src[idx] : 0;
      }
#pragma unroll
      for (int i = 0; i < MXFP4_SCALE_PAIRS_PER_LANE; i++) {
        int const idx = idx_in_warp + i * 32;
        if (idx < pairs) {
          reinterpret_cast<uint16_t *>(warp_scale)[idx] = vals[i];
        }
      }
    };

    int total_k_tile_count = 0;
#pragma unroll 1
    for (int activated_expert_offset = expert_offset;
         activated_expert_offset < num_activated_experts;
         activated_expert_offset += EXPERT_STRIDE) {
      int32_t expert_idx = mMask[activated_expert_offset];
      cute::Tensor tRoutingIndex = mRoutingIndices(expert_idx, cute::_);
#pragma unroll 1
      for (int m_tile = 0; m_tile < cute::size<2>(gA); ++m_tile) {
        int const m_base = m_tile * OUTPUT_ATOM_SIZE;
#pragma unroll 1
        for (int n_tile = 0; n_tile < cute::size<2>(gB); ++n_tile) {

          int num_prev_k_blk = total_k_tile_count;
          total_k_tile_count += k_tile_count;

          int mma_rd_k_tile = 0;
          int smem_rd_buffer = (num_prev_k_blk + mma_rd_k_tile) % NUM_AB_STAGE;
          int mma_rd_ab_full_phase =
              (num_prev_k_blk + mma_rd_k_tile) / NUM_AB_STAGE % 2;

          // Peek full phase
          bool peek_a_full_status =
              try_wait_barrier(shared_storage.a_full_mbar_ptr[smem_rd_buffer],
                               mma_rd_ab_full_phase);
          bool peek_b_full_status =
              try_wait_barrier(shared_storage.b_full_mbar_ptr[smem_rd_buffer],
                               mma_rd_ab_full_phase);

          cute::Tensor accum = cute::partition_fragment_C(
              tiled_mma, cute::take<0, 2>(mma_tiler)); // (MMA,MMA_M,MMA_N)
          // Clear the accumulator
          cute::clear(accum);
          // Initialize the accumulator to zero
          tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
          tiled_mma_rs.accumulate_ = cute::GMMA::ScaleOut::Zero;
          if constexpr (MXFP4_WEIGHT) {
            // every lane is done with the previous k loop's scales
            __syncwarp();
            load_warp_scales(expert_idx, m_tile);
            __syncwarp();
          }
#pragma unroll 1
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

            if constexpr (MXFP4_WEIGHT) {
              // Decode this thread's bytes of the packed tile (64 rows of 32
              // bytes) into the A fragments; see tCrA_rs above.
              // the raw stage the producer's TMA wrote, not &sA(0, 0, stage):
              // sA's swizzle is address-based, and A is not 1024-byte aligned
              uint8_t const *tile = reinterpret_cast<uint8_t const *>(
                  shared_weight +
                  smem_rd_buffer * OUTPUT_ATOM_SIZE * TILE_SIZE);
              uint32_t words[2][8];
#pragma unroll
              for (int h = 0; h < 2; h++) {
                uint4 const *row = reinterpret_cast<uint4 const *>(
                    tile + (mxfp4_row + 8 * h) * (TILE_SIZE / 2));
                uint4 const lo = row[0], hi = row[1];
                uint32_t const w[8] = {
                    lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
                for (int i = 0; i < 8; i++) {
                  words[h][i] = w[i];
                }
              }
              // scales[h][b]: row mxfp4_row + 8h, 32-value block b of the tile
              __nv_bfloat162 scales[2][2];
#pragma unroll
              for (int h = 0; h < 2; h++) {
#pragma unroll
                for (int b = 0; b < 2; b++) {
                  scales[h][b] = mxfp4_scale_pair(
                      warp_scale[((mxfp4_row & 15) + 8 * h) * MXFP4_SCALE_COLS +
                                 2 * k_tile + b]);
                }
              }
#pragma unroll
              for (int kb = 0; kb < 4; kb++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                  // j: row half (j & 1), columns +8 (j >> 1)
                  uint32_t const byte =
                      (words[j & 1][2 * kb + (j >> 1)] >> mxfp4_byte_shift) &
                      0xffu;
                  tCrA_rs_u32(j, 0, kb) =
                      decode_mxfp4_pair(byte, scales[j & 1][kb >> 1]);
                }
              }
              cute::warpgroup_fence_operand(tCrA_rs);
            }
            cute::warpgroup_fence_operand(accum);
            {
              // Perform MMA operation
              cute::warpgroup_arrive();
              if constexpr (MXFP4_WEIGHT) {
#pragma unroll
                for (int k_block = 0; k_block < cute::size<2>(tCrA_rs);
                     ++k_block) {
                  cute::gemm(tiled_mma_rs,
                             tCrA_rs(cute::_, cute::_, k_block),
                             tCrB_rs(cute::_, cute::_, k_block, smem_rd_buffer),
                             accum);
                  tiled_mma_rs.accumulate_ = cute::GMMA::ScaleOut::One;
                }
              } else {
#pragma unroll 3
                for (int k_block = 0; k_block < cute::size<2>(tCrA);
                     ++k_block) {
                  cute::gemm(tiled_mma,
                             tCrA(cute::_, cute::_, k_block, smem_rd_buffer),
                             tCrB(cute::_, cute::_, k_block, smem_rd_buffer),
                             accum);
                  tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
                }
              }
              cute::warpgroup_commit_batch();
              cute::warpgroup_wait<0>();
            }
            cute::warpgroup_fence_operand(accum);
            if constexpr (MXFP4_WEIGHT) {
              cute::warpgroup_fence_operand(tCrA_rs);
            }

            // only one thread will issue the arrive instruction
            if (threadIdx.x % NUM_THREAD_PER_WARPGROUP == 0) {
              cute::arrive_barrier(
                  shared_storage.ab_empty_mbar_ptr[smem_rd_buffer]);
            }
            if (mma_rd_k_tile_next < k_tile_count) {
              peek_a_full_status = try_wait_barrier(
                  shared_storage.a_full_mbar_ptr[smem_rd_buffer_next],
                  mma_rd_ab_full_phase_next);
              peek_b_full_status = try_wait_barrier(
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

#pragma unroll
          for (int i = 0; i < (MMA_N >> 1); i++) {
            int m_idx = ((warp_idx & 3) << 4) + (idx_in_warp >> 2) +
                        (((i & 3) >> 1) << 3) + m_base;
            int n_idx = ((i >> 2) << 3) + ((idx_in_warp & 3) << 1) + (i & 1);

            int topk_idx = (n_idx < BATCH_SIZE) ? tRoutingIndex(n_idx) : 0;

            bool pred =
                (n_idx < BATCH_SIZE && topk_idx > 0 && m_idx < OUTPUT_SIZE);
            if (pred) {
              // Add the bias in fp32 and round once; rows past OUTPUT_SIZE
              // would read the next task's bias slice.
              TypeAcc acc = accum(i);
              if constexpr (!NOBIAS) {
                acc += TypeBias_to_TypeAcc(mBias(n_idx, m_idx, expert_idx));
              }
              mOutput(n_idx, m_idx, topk_idx - 1) = TypeAcc_to_TypeC(acc);
            }
          }
        } // end for n_tile
      }   // end for m_tile
    }     // end for expert_idx
  }

} // end moe_linear_sm90_task_impl

} // namespace kernel
