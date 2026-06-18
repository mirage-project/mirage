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

#include "../common/bfloat16.h"
#include "sm100_ptx.cuh"

namespace kernel {

namespace nvfp4_1d2d_detail {

// Shared GEMM body. A/B arrive as TMA descriptors (by pointer); SFA/SFB/C/bias
// are raw pointers (C is written directly to gmem, no TMA store). The standalone
// launcher maps one output tile per block via a 2D grid (bid_m=output/BLOCK_M,
// bid_n=batch/BLOCK_N); here (tile_base, num_tasks) drive a persistent loop over
// that same tile space in row-major (bid_m, bid_n) order, so the __global__
// launcher passes (linear blockIdx, total blocks) and an MPK single-CTA task
// passes (0, 1) to sweep every tile itself.
template <int REDUCTION_SIZE,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          int NUM_STAGES,
          int EPI_BATCH_LA>
__device__ __forceinline__ void linear_nvfp4_1d2d_sm100_task_impl(
    CUtensorMap const *A_tmap,
    CUtensorMap const *B_tmap,
    char const *SFA_ptr,
    char const *SFB_ptr,
    type::bfloat16_t *C_ptr,
    type::bfloat16_t const *bias_ptr,
    int M,
    int N,
    int tile_base,
    int num_tasks) {
  using namespace ::kernel::sm100_ptx;
  constexpr int WARP_SIZE = 32;
  constexpr int MMA_K = 64;
  constexpr uint64_t EVICT_FIRST = 0x12F0000000000000ULL;
  constexpr uint64_t EVICT_LAST = 0x14F0000000000000ULL;
  static_assert(BLOCK_M == 128, "SM100 NVFP4 tcgen05 MMA uses BLOCK_M == 128");
  static_assert(BLOCK_K % MMA_K == 0, "BLOCK_K must be divisible by MMA_K");
  static_assert(REDUCTION_SIZE % BLOCK_K == 0,
                "K must be divisible by BLOCK_K");
  static_assert(BLOCK_N == 32 || BLOCK_N == 64 || BLOCK_N == 128,
                "BLOCK_N must be 32, 64, or 128");

  int const tid = threadIdx.x;
  int const warp_id = tid / WARP_SIZE;

  int const num_m_tiles = M / BLOCK_M;
  int const num_n_tiles = N / BLOCK_N;
  int const num_tiles = num_m_tiles * num_n_tiles;

  constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;

  extern __shared__ __align__(1024) char smem_ptr[];
  int const smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = BLOCK_N * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16;
  constexpr int SFB_size = 128 * BLOCK_K / 16;
  constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;

#pragma nv_diag_suppress static_var_with_dynamic_init
  // Single accumulator region in tmem (no room for two BLOCK_N-wide buffers
  // alongside the SF tmem). mainloop_mbar (MMA->epilogue) and output_mbar
  // (epilogue->MMA) form a 1-deep handshake so each tile's MMA waits for the
  // previous tile's epilogue to drain the accumulator; the K-stage pipeline
  // still flows continuously across tiles.
  __shared__ int64_t mbars[NUM_STAGES * 2 + 2];
  int const tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  int const mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  int const mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
  int const output_mbar_addr = mainloop_mbar_addr + 8;

  constexpr int SFA_tmem = BLOCK_N;
  constexpr int SFB_tmem = SFA_tmem + 4 * (BLOCK_K / MMA_K);

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES * 2; i++) {
      mbar_init(tma_mbar_addr + i * 8, 1);
    }
    mbar_init(mainloop_mbar_addr, 1);
    mbar_init(output_mbar_addr, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    tmem_alloc<1, BLOCK_N * 2>(smem);
  }
  __syncthreads();

  constexpr int NUM_ITERS = REDUCTION_SIZE / BLOCK_K;

  auto decode_tile = [&](int t, int &off_m, int &off_n) {
    off_m = (t % num_m_tiles) * BLOCK_M;
    off_n = (t / num_m_tiles) * BLOCK_N;
  };

  auto make_desc_AB = [](int addr) -> uint64_t {
    const int SBO = 8 * 128;
    return desc_enc(addr) | (desc_enc(SBO) << 32ULL) | (1ULL << 46ULL) |
           (2ULL << 61ULL);
  };

  auto make_desc_SF = [](int addr) -> uint64_t {
    const int SBO = 8 * 16;
    return desc_enc(addr) | (desc_enc(SBO) << 32ULL) | (1ULL << 46ULL);
  };

  if (warp_id == NUM_WARPS - 2 && elect_sync()) {
    uint64_t const cache_A = (M > N) ? EVICT_FIRST : EVICT_LAST;
    uint64_t const cache_B = (M > N) ? EVICT_LAST : EVICT_FIRST;

    auto issue_tma = [&](int off_m, int off_n, int iter_k, int stage_id) {
      const int mbar_addr = tma_mbar_addr + stage_id * 8;
      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;
      const int off_k = iter_k * BLOCK_K;

      tma_load<3, 1>(A_smem, A_tmap, 0, off_m, off_k / 256, mbar_addr, cache_A);
      tma_load<3, 1>(B_smem, B_tmap, 0, off_n, off_k / 256, mbar_addr, cache_B);

      const int rest_k = REDUCTION_SIZE / 64;
      const char *SFA_src =
          SFA_ptr + ((off_m / 128) * rest_k + off_k / 64) * 512;
      const char *SFB_src =
          SFB_ptr + ((off_n / 128) * rest_k + off_k / 64) * 512;

      tma_load_bulk(SFA_smem, SFA_src, SFA_size, mbar_addr, cache_A);
      tma_load_bulk(SFB_smem, SFB_src, SFB_size, mbar_addr, cache_B);

      mbarrier_arrive_expect_tx_tile_local(mbar_addr, STAGE_SIZE);
    };

    for (int t = tile_base, work_idx = 0; t < num_tiles;
         t += num_tasks, work_idx++) {
      int off_m, off_n;
      decode_tile(t, off_m, off_n);
      for (int iter_k = 0; iter_k < NUM_ITERS; iter_k++) {
        int const pipeline_iter = work_idx * NUM_ITERS + iter_k;
        int const stage_id = pipeline_iter % NUM_STAGES;
        if (pipeline_iter >= NUM_STAGES) {
          int const mma_phase = ((pipeline_iter - NUM_STAGES) / NUM_STAGES) % 2;
          mbar_wait(mma_mbar_addr + stage_id * 8, mma_phase);
        }
        issue_tma(off_m, off_n, iter_k, stage_id);
      }
    }
  } else if (warp_id == NUM_WARPS - 1 && elect_sync()) {
    constexpr int MMA_N = BLOCK_N;
    constexpr int MMA_M = 128;
    constexpr uint32_t i_desc = (1U << 7U) | (1U << 10U) |
                                ((uint32_t)MMA_N >> 3U << 17U) |
                                ((uint32_t)MMA_M >> 7U << 27U);

    for (int t = tile_base, work_idx = 0; t < num_tiles;
         t += num_tasks, work_idx++) {
      int off_m, off_n;
      decode_tile(t, off_m, off_n);
      int const bid_m = off_m / BLOCK_M;
      int const bid_n = off_n / BLOCK_N;
      // Wait for the previous tile's epilogue to release the accumulator.
      if (work_idx >= 1) {
        mbar_wait(output_mbar_addr, (work_idx - 1) % 2);
      }

      for (int iter_k = 0; iter_k < NUM_ITERS; iter_k++) {
        int const pipeline_iter = work_idx * NUM_ITERS + iter_k;
        int const stage_id = pipeline_iter % NUM_STAGES;
        int const tma_phase = (pipeline_iter / NUM_STAGES) % 2;

        int const A_smem = smem + stage_id * STAGE_SIZE;
        int const B_smem = A_smem + A_size;
        int const SFA_smem = B_smem + B_size;
        int const SFB_smem = SFA_smem + SFA_size;

        const uint64_t SF_desc = make_desc_SF(0);
        const uint64_t SFA_desc =
            SF_desc + (static_cast<uint64_t>(SFA_smem) >> 4ULL);
        const uint64_t SFB_desc =
            SF_desc + (static_cast<uint64_t>(SFB_smem) >> 4ULL);

        mbar_wait(tma_mbar_addr + stage_id * 8, tma_phase);

        for (int k = 0; k < BLOCK_K / MMA_K; k++) {
          uint64_t sfa_desc =
              SFA_desc + static_cast<uint64_t>(k) * (512ULL >> 4ULL);
          uint64_t sfb_desc =
              SFB_desc + static_cast<uint64_t>(k) * (512ULL >> 4ULL);
          tcgen05_cp_fp4<1>(SFA_tmem + k * 4, sfa_desc);
          tcgen05_cp_fp4<1>(SFB_tmem + k * 4, sfb_desc);
        }

        for (int k = 0; k < BLOCK_K / MMA_K; k++) {
          int const k1 = k / 4;
          int const k2 = k % 4;

          uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
          uint64_t b_desc = make_desc_AB(B_smem + k1 * BLOCK_N * 128 + k2 * 32);
          int const scale_A_tmem =
              SFA_tmem + k * 4 + (bid_m % (128 / BLOCK_M)) * (BLOCK_M / 32);
          int const scale_B_tmem =
              SFB_tmem + k * 4 + (bid_n % (128 / BLOCK_N)) * (BLOCK_N / 32);
          int const enable_input_d = (k == 0) ? iter_k : 1;

          tcgen05_mma_nvfp4<1>(a_desc,
                               b_desc,
                               i_desc,
                               scale_A_tmem,
                               scale_B_tmem,
                               enable_input_d);
        }
        tcgen05_commit_arrive<1>(mma_mbar_addr + stage_id * 8);
      }
      tcgen05_commit_arrive<1>(mainloop_mbar_addr);
    }
  } else if (tid < BLOCK_M) {
    for (int t = tile_base, work_idx = 0; t < num_tiles;
         t += num_tasks, work_idx++) {
      int off_m, off_n;
      decode_tile(t, off_m, off_n);

      mbar_wait(mainloop_mbar_addr, work_idx % 2);
      asm volatile("tcgen05.fence::after_thread_sync;");

      constexpr int WIDTH = (BLOCK_N < 64) ? BLOCK_N : 64;
      constexpr int NUM_SUBTILES = BLOCK_N / WIDTH;
      constexpr int BATCH =
          (EPI_BATCH_LA <= NUM_SUBTILES && NUM_SUBTILES % EPI_BATCH_LA == 0)
              ? EPI_BATCH_LA
              : 1;

      auto load_subtile = [&](float *dst, int n) {
        if constexpr (WIDTH == 128) {
          tcgen05_ld_32x32bx128(dst, warp_id * 32, n * WIDTH);
        }
        if constexpr (WIDTH == 64) {
          tcgen05_ld_32x32bx64(dst, warp_id * 32, n * WIDTH);
        }
        if constexpr (WIDTH == 32) {
          tcgen05_ld_32x32bx32(dst, warp_id * 32, n * WIDTH);
        }
      };

      auto store_subtile = [&](const float *src, int n) {
        for (int i = 0; i < WIDTH; i++) {
          const int row = off_n + n * WIDTH + i;
          const int col = off_m + tid;
          const int offset = row * M + col;
          type::bfloat16_t acc_bf16(src[i]);
          if (bias_ptr != nullptr) {
            C_ptr[offset] = acc_bf16 + bias_ptr[offset];
          } else {
            C_ptr[offset] = acc_bf16;
          }
        }
      };

      for (int g = 0; g < NUM_SUBTILES; g += BATCH) {
        float tmp_batch[BATCH][WIDTH];
#pragma unroll
        for (int b = 0; b < BATCH; b++) {
          load_subtile(tmp_batch[b], g + b);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;");
#pragma unroll
        for (int b = 0; b < BATCH; b++) {
          store_subtile(tmp_batch[b], g + b);
        }
      }

      // Release this accumulator back to the MMA warp.
      asm volatile("tcgen05.fence::before_thread_sync;");
      asm volatile("bar.sync 1, %0;" : : "r"(BLOCK_M) : "memory");
      if (warp_id == 0 && elect_sync()) {
        swapab_arrive_local(output_mbar_addr);
      }
    }
    asm volatile("bar.sync 1, %0;" : : "r"(BLOCK_M) : "memory");
    if (warp_id == 0) {
      tmem_dealloc<1, BLOCK_N * 2>(0);
    }
  }
}

// Standalone grid launcher: one output tile per block, A/B by value.
template <int REDUCTION_SIZE,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          int NUM_STAGES,
          int EPI_BATCH_LA = 1>
__global__
    __launch_bounds__(BLOCK_M + 2 * 32) void linear_nvfp4_1d2d_sm100_kernel(
        const __grid_constant__ CUtensorMap A_tmap,
        const __grid_constant__ CUtensorMap B_tmap,
        char const *SFA_ptr,
        char const *SFB_ptr,
        type::bfloat16_t *C_ptr,
        type::bfloat16_t const *bias_ptr,
        int M,
        int N) {
  // 2D grid (bid_m, bid_n) -> row-major linear tile; one tile per block.
  int const tile = blockIdx.y * gridDim.x + blockIdx.x;
  int const num_tasks = gridDim.x * gridDim.y;
  linear_nvfp4_1d2d_sm100_task_impl<REDUCTION_SIZE,
                                    BLOCK_M,
                                    BLOCK_N,
                                    BLOCK_K,
                                    NUM_STAGES,
                                    EPI_BATCH_LA>(&A_tmap,
                                                  &B_tmap,
                                                  SFA_ptr,
                                                  SFB_ptr,
                                                  C_ptr,
                                                  bias_ptr,
                                                  M,
                                                  N,
                                                  tile,
                                                  num_tasks);
}

} // namespace nvfp4_1d2d_detail

using nvfp4_1d2d_detail::linear_nvfp4_1d2d_sm100_kernel;
using nvfp4_1d2d_detail::linear_nvfp4_1d2d_sm100_task_impl;

} // namespace kernel
