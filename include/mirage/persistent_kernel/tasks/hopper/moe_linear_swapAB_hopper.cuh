
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

// One run of 16 MXFP4 values along K: 8 packed bytes, low nibble first, and
// the E8M0 scale of the 32-value block it lies in.
struct Mxfp4Run {
  uint2 packed;
  uint32_t scale;
};

// Decodes a run to BF16 and writes it as two 16-byte stores, each holding 8
// values that stay contiguous under the K-major swizzle.
//
// An E2M1 nibble s.ee.m placed as BF16 bits (s << 15) | (ee.m << 6) is its
// value times 2^-126 (ee = 0 lands on a subnormal). Multiplying by 2^126 and
// then by the E8M0 scale 2^(scale - 127) is exact, two values per HMUL2.
CUTE_DEVICE __nv_bfloat162 bf162_from_bits(uint32_t bits) {
  return *reinterpret_cast<__nv_bfloat162 const *>(&bits);
}

template <typename T>
CUTE_DEVICE void store_mxfp4_run(T *dst_lo, T *dst_hi, Mxfp4Run const &run) {
  // the BF16 with exponent field `scale` is 2^(scale - 127); 0 decodes to 0
  uint32_t const scale_bits = run.scale == 0 ? 0u : run.scale << 7;
  __nv_bfloat162 const scale2 = bf162_from_bits(scale_bits | scale_bits << 16);
  __nv_bfloat162 const unbias2 = bf162_from_bits(0x7e807e80u); // 2^126
  uint32_t const words[2] = {run.packed.x, run.packed.y};
  uint32_t out[8];
#pragma unroll
  for (int w = 0; w < 2; w++) {
#pragma unroll
    for (int b = 0; b < 4; b++) {
      uint32_t const byte = (words[w] >> (8 * b)) & 0xffu;
      // low nibble (even K) to the low half, high nibble to the high half
      uint32_t const bits = ((byte & 0x08u) << 12) | ((byte & 0x07u) << 6) |
                            ((byte & 0x80u) << 24) | ((byte & 0x70u) << 18);
      __nv_bfloat162 const pair =
          __hmul2(__hmul2(bf162_from_bits(bits), unbias2), scale2);
      out[w * 4 + b] = *reinterpret_cast<uint32_t const *>(&pair);
    }
  }
  *reinterpret_cast<uint4 *>(dst_lo) =
      make_uint4(out[0], out[1], out[2], out[3]);
  *reinterpret_cast<uint4 *>(dst_hi) =
      make_uint4(out[4], out[5], out[6], out[7]);
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
                              uint8_t const *weight_scale = nullptr) {
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

  // MXFP4 staging past the shared storage: a ring of packed 64x64 weight
  // tiles (64 rows of 32 bytes) that TMA fills MXFP4_STAGES tiles ahead of
  // the decode, their barriers, and one m tile's E8M0 scales.
  constexpr int MXFP4_STAGES = 8;
  constexpr int MXFP4_STAGE_BYTES = 64 * 64 / 2;
  constexpr int MXFP4_SCALE_COLS = REDUCTION_SIZE / 32;
  uintptr_t const mxfp4_base =
      (aligned_smem + sizeof(SharedStorage) + 127) / 128 * 128;
  uint8_t *mxfp4_stage = reinterpret_cast<uint8_t *>(mxfp4_base);
  uint64_t *mxfp4_full = reinterpret_cast<uint64_t *>(
      mxfp4_base + MXFP4_STAGES * MXFP4_STAGE_BYTES);
  uint8_t *mxfp4_scale = reinterpret_cast<uint8_t *>(mxfp4_full + MXFP4_STAGES);
  static_assert(!MXFP4_WEIGHT ||
                    sizeof(SharedStorage) + 128 * 2 +
                            MXFP4_STAGES * (MXFP4_STAGE_BYTES + 8) +
                            64 * MXFP4_SCALE_COLS <=
                        mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                "MXFP4 staging does not fit in shared memory");

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
    if constexpr (MXFP4_WEIGHT) {
      cutlass::arch::detail::initialize_barrier_array_aligned<
          cutlass::arch::ClusterTransactionBarrier,
          MXFP4_STAGES>(mxfp4_full, /* arrival count */ 1);
    }
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

    // MXFP4: each producer thread decodes MXFP4_RUNS_PER_THREAD runs of 16
    // values of a 64x64 weight tile. Rows past OUTPUT_SIZE decode as zero.
    constexpr int MXFP4_RUNS_PER_ROW = TILE_SIZE / 16;
    constexpr int MXFP4_RUNS_PER_THREAD =
        OUTPUT_ATOM_SIZE * MXFP4_RUNS_PER_ROW / NUM_THREAD_PER_WARPGROUP;
    static_assert(
        OUTPUT_ATOM_SIZE * MXFP4_RUNS_PER_ROW % NUM_THREAD_PER_WARPGROUP == 0,
        "MXFP4 runs must split evenly over the producer");
    static_assert(!MXFP4_WEIGHT || (OUTPUT_ATOM_SIZE == 64 && TILE_SIZE == 64),
                  "MXFP4 staging holds 64x64 tiles");
    // 32-value blocks, and an even number of them per row for the 2-byte
    // scale loads
    static_assert(!MXFP4_WEIGHT || REDUCTION_SIZE % 64 == 0,
                  "MXFP4 blocks are 32 values, two per k tile");
    int const mxfp4_tid = threadIdx.x % NUM_THREAD_PER_WARPGROUP;
    int mxfp4_tiles_used = 0; // ring stages consumed so far
    // One thread stages tile k_t of (expert, m_t) into ring slot `slot`.
    auto issue_mxfp4_tile = [&](int expert, int m_t, int k_t, int slot) {
      if constexpr (MXFP4_WEIGHT) { // TMA_A is the bf16 weight map otherwise
        if (mxfp4_tid != 0) {
          return;
        }
        cute::set_barrier_transaction_bytes(mxfp4_full[slot],
                                            MXFP4_STAGE_BYTES);
        int const coords[2] = {k_t * (TILE_SIZE / 2),
                               expert * OUTPUT_STRIDE + m_t * OUTPUT_ATOM_SIZE};
        tma_a.tma_cp_async(
            mxfp4_full[slot], mxfp4_stage + slot * MXFP4_STAGE_BYTES, coords);
      }
    };
    // All producer threads load (expert, m_t)'s scales: contiguous rows of
    // REDUCTION_SIZE / 32 bytes, an even count, so 2-byte loads, all issued
    // before any is stored so their latencies overlap.
    constexpr int MXFP4_SCALE_PAIRS = 64 * MXFP4_SCALE_COLS / 2;
    constexpr int MXFP4_SCALE_PAIRS_PER_THREAD =
        (MXFP4_SCALE_PAIRS + NUM_THREAD_PER_WARPGROUP - 1) /
        NUM_THREAD_PER_WARPGROUP;
    auto load_mxfp4_scales = [&](int expert, int m_t) {
      int const rows =
          min(OUTPUT_ATOM_SIZE, OUTPUT_SIZE - m_t * OUTPUT_ATOM_SIZE);
      int const pairs = rows * MXFP4_SCALE_COLS / 2;
      uint16_t const *src = reinterpret_cast<uint16_t const *>(
          weight_scale + (static_cast<size_t>(expert) * OUTPUT_STRIDE +
                          m_t * OUTPUT_ATOM_SIZE) *
                             MXFP4_SCALE_COLS);
      uint16_t vals[MXFP4_SCALE_PAIRS_PER_THREAD];
#pragma unroll
      for (int i = 0; i < MXFP4_SCALE_PAIRS_PER_THREAD; i++) {
        int const idx = mxfp4_tid + i * NUM_THREAD_PER_WARPGROUP;
        vals[i] = idx < pairs ? src[idx] : 0;
      }
#pragma unroll
      for (int i = 0; i < MXFP4_SCALE_PAIRS_PER_THREAD; i++) {
        int const idx = mxfp4_tid + i * NUM_THREAD_PER_WARPGROUP;
        if (idx < pairs) {
          reinterpret_cast<uint16_t *>(mxfp4_scale)[idx] = vals[i];
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

          if constexpr (MXFP4_WEIGHT) {
            // The previous k loop's last decode passed the producer barrier,
            // so its scales are free to overwrite.
            load_mxfp4_scales(expert_idx, m_tile);
            wg_sync<NUM_THREAD_PER_WARPGROUP>(7);
            for (int k = 0; k < min(MXFP4_STAGES, k_tile_count); k++) {
              issue_mxfp4_tile(
                  expert_idx, m_tile, k, (mxfp4_tiles_used + k) % MXFP4_STAGES);
            }
          }
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

            if constexpr (MXFP4_WEIGHT) {
              // Decode this expert's staged 64x64 weight tile into the BF16
              // WGMMA shared-memory layout.
              int const tid = mxfp4_tid;
              int const slot = mxfp4_tiles_used % MXFP4_STAGES;
              cute::wait_barrier(mxfp4_full[slot],
                                 (mxfp4_tiles_used / MXFP4_STAGES) & 1);
              int const rows = min(OUTPUT_ATOM_SIZE,
                                   OUTPUT_SIZE - m_tile * OUTPUT_ATOM_SIZE);
#pragma unroll
              for (int r = 0; r < MXFP4_RUNS_PER_THREAD; r++) {
                int const run_idx = tid + r * NUM_THREAD_PER_WARPGROUP;
                int const row = run_idx / MXFP4_RUNS_PER_ROW;
                int const col = (run_idx % MXFP4_RUNS_PER_ROW) * 16;
                Mxfp4Run run;
                if (row < rows) {
                  run.packed = *reinterpret_cast<uint2 const *>(
                      mxfp4_stage + slot * MXFP4_STAGE_BYTES +
                      row * (TILE_SIZE / 2) + col / 2);
                  run.scale = mxfp4_scale[row * MXFP4_SCALE_COLS +
                                          (k_tile * TILE_SIZE + col) / 32];
                } else {
                  run.packed = make_uint2(0, 0);
                  run.scale = 0;
                }
                store_mxfp4_run(&sA(row, col, smem_wr_buffer),
                                &sA(row, col + 8, smem_wr_buffer),
                                run);
              }
              wg_sync<NUM_THREAD_PER_WARPGROUP>(7);
              tma::async_proxy_fence();
              wg_sync<NUM_THREAD_PER_WARPGROUP>(7);
              if (tid == 0) {
                cute::arrive_barrier(
                    shared_storage.a_full_mbar_ptr[smem_wr_buffer]);
              }
              // every producer thread has read this slot; refill it
              if (k_tile + MXFP4_STAGES < k_tile_count) {
                issue_mxfp4_tile(
                    expert_idx, m_tile, k_tile + MXFP4_STAGES, slot);
              }
              mxfp4_tiles_used++;
            } else {
              // TMA for loading A; only one thread issues the instruction.
              if (threadIdx.x % NUM_THREAD_PER_WARPGROUP == 0) {
                int tma_coords_A[2] = {k_tile * TILE_SIZE,
                                       m_tile * OUTPUT_ATOM_SIZE +
                                           expert_idx * OUTPUT_STRIDE};
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

            cute::warpgroup_fence_operand(accum);
            {
              // Perform MMA operation
              cute::warpgroup_arrive();
#pragma unroll 3
              for (int k_block = 0; k_block < cute::size<2>(tCrA); ++k_block) {
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
            TypeC fragD = TypeAcc_to_TypeC(accum(i));

            if constexpr (!NOBIAS) {
              fragD += mBias(n_idx, m_idx, expert_idx);
            }
            if (pred) {
              mOutput(n_idx, m_idx, topk_idx - 1) = fragD;
            }
          }
        } // end for n_tile
      }   // end for m_tile
    }     // end for expert_idx
  }

} // end moe_linear_sm90_task_impl

} // namespace kernel
