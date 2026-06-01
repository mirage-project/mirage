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

#include "common/bfloat16.h"
#include "blackwell/sm100_ptx.cuh"
#include "blackwell/linear_mxfp4_1d2d_sm100.cuh"  // SF_BYTES_PER_K_TILE, SF_TMEM_COLS_PER_MMA_K

#include <c10/util/Exception.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

// MXFP4 2SM (cluster_dim=(2,1,1)) kernel — port of linear_nvfp4_1d2d_2sm_sm100.cuh.
// Same warp-specialized + cluster-multicast structure; the format-specific
// deltas (all marked `// MXFP4 delta:`) are:
//   * tcgen05.mma uses scale_vec::2X (NVFP4 used scale_vec::4X) — MXFP4 group=32.
//   * SF gmem tile is 256 bytes per (128-row, 64-K) block (NVFP4: 512). This
//     exactly equals one TMA "half-tile" in the NVFP4 SF descriptor, so we
//     simplify: the MXFP4 SF tmap is a true 3D `{256B, K/64, rows/128}` tensor
//     with no inner-dim splitting (NVFP4 had to split to satisfy the 256-byte
//     SWIZZLE_NONE boxDim cap).
//   * SMEM SF region & TMA bytes halve: SFA_size = 128*BLOCK_K/32, etc.
//   * TMEM SF strides halve: each MMA-K consumes 2 columns (was 4).

inline void check_cu_mx_2sm(CUresult err) {
  if (err == CUDA_SUCCESS) {
    return;
  }
  const char *error_msg_ptr = nullptr;
  if (cuGetErrorString(err, &error_msg_ptr) != CUDA_SUCCESS) {
    error_msg_ptr = "unable to get error string";
  }
  TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", error_msg_ptr);
}

inline void check_cuda_mx_2sm(cudaError_t err) {
  if (err == cudaSuccess) {
    return;
  }
  TORCH_CHECK(false, cudaGetErrorString(err));
}

inline void init_AB_tmap_mx_2sm(CUtensorMap *tmap,
                                const char *ptr,
                                uint64_t global_height,
                                uint64_t global_width,
                                uint32_t shared_height,
                                uint32_t shared_width) {
  constexpr uint32_t rank = 3;
  uint64_t globalDim[rank]          = {256, global_height, global_width / 256};
  uint64_t globalStrides[rank - 1]  = {global_width / 2, 128};
  uint32_t boxDim[rank]             = {256, shared_height, shared_width / 256};
  uint32_t elementStrides[rank]     = {1, 1, 1};

  CUresult err = cuTensorMapEncodeTiled(
      tmap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
      rank,
      const_cast<char *>(ptr),
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  check_cu_mx_2sm(err);
}

// MXFP4 SF tmap (identical to NVFP4 SF tmap: atom = 512 B per (128 rows, 64 K),
// split into two 256-B half-tiles for SWIZZLE_NONE compatibility).
inline void init_SF_tmap_mx_2sm(CUtensorMap *tmap,
                                const char *ptr,
                                uint64_t rows,
                                uint64_t reduction_size,
                                uint32_t shared_k_blocks) {
  constexpr uint32_t rank = 3;
  uint64_t globalDim[rank]         = {256, 2 * (reduction_size / 64), rows / 128};
  uint64_t globalStrides[rank - 1] = {256, 512 * (reduction_size / 64)};
  uint32_t boxDim[rank]            = {256, 2 * shared_k_blocks, 1};
  uint32_t elementStrides[rank]    = {1, 1, 1};

  CUresult err = cuTensorMapEncodeTiled(
      tmap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
      rank,
      const_cast<char *>(ptr),
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  check_cu_mx_2sm(err);
}

inline void init_C_tmap_mx_2sm(CUtensorMap *tmap,
                               void *ptr,
                               uint64_t batch_size,
                               uint64_t output_size,
                               uint32_t tile_rows,
                               uint32_t tile_cols) {
  constexpr uint32_t rank = 2;
  uint64_t globalDim[rank] = {output_size, batch_size};
  uint64_t globalStrides[rank - 1] = {output_size * sizeof(type::bfloat16_t)};
  uint32_t boxDim[rank] = {tile_cols, tile_rows};
  uint32_t elementStrides[rank] = {1, 1};

  CUresult err = cuTensorMapEncodeTiled(
      tmap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      rank,
      ptr,
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  check_cu_mx_2sm(err);
}

namespace kernel {

using namespace ::kernel::sm100_ptx;

template <int REDUCTION_SIZE,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          int NUM_STAGES,
          int SUPERGROUP_SIZE,
          int EPI_TILE_N,
          int EPI_NUM_D_TILES,
          bool EPI_BATCHED,
          int EPI_BATCH_LA,
          bool OVERLAP_OUTPUT_MBAR,
          bool HAS_BIAS>
__global__ __launch_bounds__(BLOCK_M + 4 * 32) void
linear_mxfp4_1d2d_2sm_sm100_kernel(const __grid_constant__ CUtensorMap A_tmap,
                                   const __grid_constant__ CUtensorMap B_tmap,
                                   const __grid_constant__ CUtensorMap C_tmap,
                                   const __grid_constant__ CUtensorMap SFA_tmap,
                                   const __grid_constant__ CUtensorMap SFB_tmap,
                                   const type::bfloat16_t *bias_ptr,
                                   int M,
                                   int N) {
  static_assert(BLOCK_M == 128, "2SM SM100 MXFP4 uses 128 rows per CTA");
  static_assert(BLOCK_N == 256, "2SM kernel requires BLOCK_N == 256");
  static_assert(BLOCK_K % MMA_K == 0, "BLOCK_K must be divisible by MMA_K");
  static_assert(REDUCTION_SIZE % BLOCK_K == 0, "K must be divisible by BLOCK_K");

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int cta_group_m = static_cast<int>(cluster_ctaid_x());
  const int cluster_idx = static_cast<int>(blockIdx.x) / 2;
  const int num_clusters = static_cast<int>(gridDim.x) / 2;
  const int num_m_tiles = M / (2 * BLOCK_M);
  const int num_n_tiles = N / BLOCK_N;
  const int num_output_tiles = num_m_tiles * num_n_tiles;

  constexpr int EPILOGUE_WARPS = BLOCK_M / WARP_SIZE;
  constexpr int MMA_WARP = EPILOGUE_WARPS;
  constexpr int SCALE_TMA_WARP = EPILOGUE_WARPS + 2;
  constexpr int TILE_TMA_WARP = EPILOGUE_WARPS + 3;
  static_assert(SUPERGROUP_SIZE > 0, "SUPERGROUP_SIZE must be positive");
  static_assert(EPI_TILE_N == 32 || EPI_TILE_N == 64 || EPI_TILE_N == 128, "EPI_TILE_N must be 32, 64, or 128");
  static_assert(EPI_NUM_D_TILES > 0, "EPI_NUM_D_TILES must be positive");
  static_assert(BLOCK_N % EPI_TILE_N == 0, "BLOCK_N must be divisible by EPI_TILE_N");
  constexpr int EPI_PIPE_DEPTH = BLOCK_N / EPI_TILE_N;
  constexpr int EPI_TILE_BYTES = EPI_TILE_N * BLOCK_M * sizeof(type::bfloat16_t);
  static_assert(EPI_BATCH_LA >= 1, "EPI_BATCH_LA must be >= 1");
  static_assert(EPI_BATCH_LA <= EPI_PIPE_DEPTH, "EPI_BATCH_LA must not exceed EPI_PIPE_DEPTH");
  static_assert(EPI_PIPE_DEPTH % EPI_BATCH_LA == 0, "EPI_PIPE_DEPTH must be divisible by EPI_BATCH_LA");

  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int B_LOCAL_N = BLOCK_N / 2;
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = B_LOCAL_N * BLOCK_K / 2;
  // MXFP4 delta: halved (group 32 vs NVFP4 group 16).
  constexpr int SFA_size = 128 * BLOCK_K / 16;  // same as NVFP4
  constexpr int SFB_SCALE_TILES = (BLOCK_N + 127) / 128;
  constexpr int SFB_TILE_BYTES = 128 * BLOCK_K / 16;
  constexpr int SFB_size = SFB_SCALE_TILES * SFB_TILE_BYTES;  // per-CTA smem
  constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;
  constexpr int SCALE_EXPECTED_TX = SFA_size + SFB_size;

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES * 3 + 2];
  const int tile_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int scale_mbar_addr = tile_mbar_addr + NUM_STAGES * 8;
  const int mma_mbar_addr = scale_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
  const int output_mbar_addr = mainloop_mbar_addr + 8;

  constexpr int MMA_PER_TILE = BLOCK_K / MMA_K;
  // 4 TMEM cols per MMA-K (NVFP4-style; MXFP4 vec::2X uses only the first 2 of
  // them but addressing stride stays 4).
  constexpr int SFA_STAGE_STRIDE  = SF_TMEM_COLS_PER_MMA_K * MMA_PER_TILE;
  constexpr int SFB_STAGE_STRIDE  = SF_TMEM_COLS_PER_MMA_K * MMA_PER_TILE * SFB_SCALE_TILES;
  constexpr int SFA_K_STRIDE      = SF_TMEM_COLS_PER_MMA_K;
  constexpr int SFB_K_STRIDE      = SF_TMEM_COLS_PER_MMA_K * SFB_SCALE_TILES;
  constexpr int SFB_N_TILE_STRIDE = SF_TMEM_COLS_PER_MMA_K;
  constexpr int SFA_tmem = BLOCK_N;
  constexpr int SFB_tmem = SFA_tmem + SFA_STAGE_STRIDE * NUM_STAGES;
  constexpr int TMEM_ALLOC_COLS = BLOCK_N * 2;
  constexpr int TILE_EXPECTED_TX = 2 * A_size + 2 * B_size;

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES; i++) {
      mbar_init(tile_mbar_addr + i * 8, 1);
      mbar_init(scale_mbar_addr + i * 8, 1);
      mbar_init(mma_mbar_addr + i * 8, 1);
    }
    mbar_init(mainloop_mbar_addr, 1);
    mbar_init(output_mbar_addr, 2);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    tmem_alloc<2, TMEM_ALLOC_COLS>(smem);
  }
  __syncthreads();
  cluster_sync();

  constexpr int NUM_ITERS = REDUCTION_SIZE / BLOCK_K;
  const uint64_t cache_A = (M > N) ? EVICT_FIRST : EVICT_LAST;
  const uint64_t cache_B = (M > N) ? EVICT_LAST : EVICT_FIRST;

  auto make_desc_AB = [](int addr) -> uint64_t {
    const int SBO = 8 * 128;
    return desc_enc(addr) | (desc_enc(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
  };
  auto make_desc_SF = [](int addr) -> uint64_t {
    const int SBO = 8 * 16;
    return desc_enc(addr) | (desc_enc(SBO) << 32ULL) | (1ULL << 46ULL);
  };

  if (warp_id == TILE_TMA_WARP && elect_sync()) {
    for (int output_tile = cluster_idx, work_idx = 0; output_tile < num_output_tiles; output_tile += num_clusters, work_idx++) {
      const PersistentTile tile = map_supergroup_tile(output_tile, num_m_tiles, num_n_tiles, SUPERGROUP_SIZE);
      const int off_m = (tile.row_block * 2 + cta_group_m) * BLOCK_M;
      const int off_n = tile.col_block * BLOCK_N;
      for (int iter_k = 0; iter_k < NUM_ITERS; iter_k++) {
        const int pipeline_iter = work_idx * NUM_ITERS + iter_k;
        const int stage_id = pipeline_iter % NUM_STAGES;
        const int mma_phase = ((pipeline_iter - NUM_STAGES) / NUM_STAGES) % 2;

        const int mbar_addr = tile_mbar_addr + stage_id * 8;
        const int A_smem = smem + stage_id * STAGE_SIZE;
        const int B_smem = A_smem + A_size;
        const int off_k = iter_k * BLOCK_K;

        if (pipeline_iter >= NUM_STAGES) mbar_wait(mma_mbar_addr + stage_id * 8, mma_phase);
        // Only CTA 0 arms the cluster-scope tile mbar; cta_group::2 TMA reports there.
        if (cta_group_m == 0) {
          mbarrier_arrive_expect_tx_tile_cluster(mbar_addr, TILE_EXPECTED_TX);
        }
        tma_load<3, 2>(A_smem, &A_tmap, 0, off_m, off_k / 256, mbar_addr, cache_A);
        tma_load<3, 2>(B_smem, &B_tmap, 0, off_n + cta_group_m * B_LOCAL_N, off_k / 256, mbar_addr, cache_B);
      }
    }
  } else if (warp_id == SCALE_TMA_WARP && elect_sync()) {
    for (int output_tile = cluster_idx, work_idx = 0; output_tile < num_output_tiles; output_tile += num_clusters, work_idx++) {
      const PersistentTile tile = map_supergroup_tile(output_tile, num_m_tiles, num_n_tiles, SUPERGROUP_SIZE);
      const int off_m = (tile.row_block * 2 + cta_group_m) * BLOCK_M;
      const int off_n = tile.col_block * BLOCK_N;
      for (int iter_k = 0; iter_k < NUM_ITERS; iter_k++) {
        const int pipeline_iter = work_idx * NUM_ITERS + iter_k;
        const int stage_id  = pipeline_iter % NUM_STAGES;
        const int mma_phase = ((pipeline_iter - NUM_STAGES) / NUM_STAGES) % 2;

        const int mbar_addr = scale_mbar_addr + stage_id * 8;
        const int A_smem = smem + stage_id * STAGE_SIZE;
        const int SFA_smem = A_smem + A_size + B_size;
        const int SFB_smem = SFA_smem + SFA_size;
        const int off_k = iter_k * BLOCK_K;
        const uint16_t self_mask = static_cast<uint16_t>(1u << cta_group_m);

        if (pipeline_iter >= NUM_STAGES) mbar_wait(mma_mbar_addr + stage_id * 8, mma_phase);
        // SF atom covers 64 K-elements (same as NVFP4); 2 half-tiles per atom.
        tma_load_3d_multicast(SFA_smem, &SFA_tmap, 0, 2 * (off_k / 64), off_m / 128, mbar_addr, self_mask, cache_A);
        tma_load_3d_multicast(SFB_smem + cta_group_m * SFB_TILE_BYTES, &SFB_tmap, 0, 2 * (off_k / 64), (off_n / 128) + cta_group_m, mbar_addr, 0b11, cache_B);
      }
    }
  } else if (warp_id == MMA_WARP) {
    constexpr int MMA_M = 256;
    constexpr int MMA_N = BLOCK_N;
    // BlockScaled i_desc with scale_format = UE8M0 (bit 23) for MXFP4.
    constexpr uint32_t i_desc = (1U << 7U) | (1U << 10U) | ((uint32_t) MMA_N >> 3U << 17U) | (1U << 23U) | ((uint32_t) MMA_M >> 7U << 27U);
    const bool mma_leader = (cta_group_m == 0) && elect_sync();

    if (mma_leader) {
      for (int output_tile = cluster_idx, work_idx = 0; output_tile < num_output_tiles; output_tile += num_clusters, work_idx++) {
        const PersistentTile tile = map_supergroup_tile(output_tile, num_m_tiles, num_n_tiles, SUPERGROUP_SIZE);
        const int tile_n = tile.col_block;
        mbar_wait(output_mbar_addr, (work_idx - 1) % 2);
        asm volatile("tcgen05.fence::after_thread_sync;");

        for (int iter_k = 0; iter_k < NUM_ITERS; iter_k++) {
          const int pipeline_iter = work_idx * NUM_ITERS + iter_k;
          const int stage_id = pipeline_iter % NUM_STAGES;
          const int tma_phase = (pipeline_iter / NUM_STAGES) % 2;
          const int A_smem   = smem + stage_id * STAGE_SIZE;
          const int B_smem   = A_smem + A_size;
          const int SFA_smem = B_smem + B_size;
          const int SFB_smem = SFA_smem + SFA_size;
          const uint64_t SFA_desc = make_desc_SF(SFA_smem);
          const uint64_t SFB_desc = make_desc_SF(SFB_smem);

          // One cp per MMA-K (NVFP4-style). 4 TMEM cols per MMA-K; vec::2X reads
          // the first 2 (the rest are padding).
          auto copy_scale_k = [&](int k) {
            uint64_t sfa_desc = SFA_desc + static_cast<uint64_t>(k) * (SF_BYTES_PER_K_TILE >> 4ULL);
            tcgen05_cp_fp4<2>(SFA_tmem + stage_id * SFA_STAGE_STRIDE + k * SFA_K_STRIDE, sfa_desc);
            #pragma unroll
            for (int n_tile = 0; n_tile < SFB_SCALE_TILES; n_tile++) {
              uint64_t sfb_desc = SFB_desc + static_cast<uint64_t>(n_tile) * (SFB_TILE_BYTES >> 4ULL) + static_cast<uint64_t>(k) * (SF_BYTES_PER_K_TILE >> 4ULL);
              tcgen05_cp_fp4<2>(SFB_tmem + stage_id * SFB_STAGE_STRIDE + k * SFB_K_STRIDE + n_tile * SFB_N_TILE_STRIDE, sfb_desc);
            }
          };

          mbarrier_arrive_expect_tx_local(scale_mbar_addr + stage_id * 8, SCALE_EXPECTED_TX);
          mbar_wait(scale_mbar_addr + stage_id * 8, tma_phase);

          #pragma unroll
          for (int k_sf = 0; k_sf < BLOCK_K / MMA_K; k_sf++) {
            copy_scale_k(k_sf);
          }

          mbarrier_wait_cluster(tile_mbar_addr + stage_id * 8, tma_phase);

          #pragma unroll
          for (int k = 0; k < BLOCK_K / MMA_K; k++) {
            const int k1 = k / 4;
            const int k2 = k % 4;

            uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
            uint64_t b_desc = make_desc_AB(B_smem + k1 * B_LOCAL_N * 128 + k2 * 32);

            const int scale_A_tmem = SFA_tmem + stage_id * SFA_STAGE_STRIDE + k * SFA_K_STRIDE;
            const int scale_B_tmem = SFB_tmem + stage_id * SFB_STAGE_STRIDE + k * SFB_K_STRIDE + ((BLOCK_N < 128) ? (tile_n % (128 / BLOCK_N)) * (BLOCK_N / 32) : 0);
            const int enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;

            tcgen05_mma_mxfp4<2>(a_desc, b_desc, i_desc, scale_A_tmem, scale_B_tmem, enable_input_d);
          }

          tcgen05_commit_arrive<2>(mma_mbar_addr + stage_id * 8);
        }
        tcgen05_commit_arrive<2>(mainloop_mbar_addr);
      }
    }
  } else if (warp_id < EPILOGUE_WARPS) {
    for (int output_tile = cluster_idx, work_idx = 0; output_tile < num_output_tiles; output_tile += num_clusters, work_idx++) {
      const PersistentTile tile = map_supergroup_tile(output_tile, num_m_tiles, num_n_tiles, SUPERGROUP_SIZE);
      const int off_m = (tile.row_block * 2 + cta_group_m) * BLOCK_M;
      const int off_n = tile.col_block * BLOCK_N;

      mbar_wait(mainloop_mbar_addr, work_idx % 2);
      asm volatile("tcgen05.fence::after_thread_sync;");

      auto epilogue_M_major = [&]() {
        const int tmem_row_base = cta_group_m * BLOCK_M;
        const int out_smem_addr = smem + STAGE_SIZE * NUM_STAGES;
        type::bfloat16_t *out_smem = reinterpret_cast<type::bfloat16_t *>(smem_ptr + STAGE_SIZE * NUM_STAGES);

        auto load_subtile = [&](float *dst, int n) {
          if constexpr (EPI_TILE_N == 128) {
            tcgen05_ld_32x32bx128(dst, tmem_row_base + warp_id * 32, n * EPI_TILE_N);
          }
          if constexpr (EPI_TILE_N == 64) {
            tcgen05_ld_32x32bx64(dst, tmem_row_base + warp_id * 32, n * EPI_TILE_N);
          }
          if constexpr (EPI_TILE_N == 32) {
            tcgen05_ld_32x32bx32(dst, tmem_row_base + warp_id * 32, n * EPI_TILE_N);
          }
        };

        auto store_subtile = [&](const float *src, int n) {
          const int buffer_id = n % EPI_NUM_D_TILES;
          if (n >= EPI_NUM_D_TILES) {
            if (warp_id == 0 && elect_sync()) {
              tma_store_wait<EPI_NUM_D_TILES - 1>();
            }
            asm volatile("bar.sync 1, %0;" : : "r"(BLOCK_M) : "memory");
          }

          type::bfloat16_t *out_smem_tile = out_smem + buffer_id * EPI_TILE_N * BLOCK_M;
          if constexpr (HAS_BIAS) {
            const type::bfloat16_t *bias_row = bias_ptr + (off_n + n * EPI_TILE_N) * M + off_m + tid;
            for (int i = 0; i < EPI_TILE_N; i++) {
              type::bfloat16_t acc_bf16(src[i]);
              out_smem_tile[i * BLOCK_M + tid] = acc_bf16 + bias_row[i * M];
            }
          } else {
            for (int i = 0; i < EPI_TILE_N; i++) {
              out_smem_tile[i * BLOCK_M + tid] = type::bfloat16_t(src[i]);
            }
          }
          tma_store_fence();
          asm volatile("bar.sync 1, %0;" : : "r"(BLOCK_M) : "memory");

          if (warp_id == 0 && elect_sync()) {
            tma_store_2d(out_smem_addr + buffer_id * EPI_TILE_BYTES, &C_tmap, off_m, off_n + n * EPI_TILE_N);
            tma_store_commit();
          }
        };

        if constexpr (OVERLAP_OUTPUT_MBAR) {
          #pragma unroll
          for (int j = 0; j < EPI_PIPE_DEPTH; j++) {
            float tmp[EPI_TILE_N];
            load_subtile(tmp, j);
            if (j == EPI_PIPE_DEPTH - 1) {
              asm volatile("tcgen05.wait::ld.sync.aligned;");
              asm volatile("tcgen05.fence::before_thread_sync;\n");
              asm volatile("bar.sync 1, %0;" : : "r"(BLOCK_M) : "memory");
              if (warp_id == 0 && elect_sync()) {
                mbarrier_arrive_to_cta0(output_mbar_addr);
              }
            }
            store_subtile(tmp, j);
          }
          if (warp_id == 0 && elect_sync()) {
            tma_store_wait<0>();
          }
        }

        else if constexpr (!EPI_BATCHED) {
          for (int n = 0; n < EPI_PIPE_DEPTH; n++) {
            float tmp[EPI_TILE_N];
            load_subtile(tmp, n);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            store_subtile(tmp, n);
          }
          if (warp_id == 0 && elect_sync()) {
            tma_store_wait<0>();
          }
        }

        else {
          for (int g = 0; g < EPI_PIPE_DEPTH; g += EPI_BATCH_LA) {
            float tmp_batch[EPI_BATCH_LA][EPI_TILE_N];
            #pragma unroll
            for (int b = 0; b < EPI_BATCH_LA; b++) {
              load_subtile(tmp_batch[b], g + b);
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #pragma unroll
            for (int b = 0; b < EPI_BATCH_LA; b++) {
              store_subtile(tmp_batch[b], g + b);
            }
          }
          if (warp_id == 0 && elect_sync()) {
            tma_store_wait<0>();
          }
        }
      };

      epilogue_M_major();
      asm volatile("bar.sync 1, %0;" : : "r"(BLOCK_M) : "memory");
      if constexpr (!OVERLAP_OUTPUT_MBAR) {
        if (warp_id == 0 && elect_sync()) {
          mbarrier_arrive_to_cta0(output_mbar_addr);
        }
      }
    }
  }

  __syncthreads();
  cluster_sync();

  if (warp_id == 0) {
    tmem_dealloc<2, TMEM_ALLOC_COLS>(0);
  }
}

}  // namespace kernel
