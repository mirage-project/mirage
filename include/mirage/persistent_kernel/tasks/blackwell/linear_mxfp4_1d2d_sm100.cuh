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

#include "blackwell/sm100_ptx.cuh"
#include "common/bfloat16.h"

#include <c10/util/Exception.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

namespace kernel {

using namespace ::kernel::sm100_ptx;

template <int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          int NUM_STAGES,
          bool C_N_MAJOR,
          int EPI_BATCH_LA = 1>
__global__
    __launch_bounds__(BLOCK_M + 2 * 32) void linear_mxfp4_1d2d_sm100_kernel(
        const __grid_constant__ CUtensorMap A_tmap,
        const __grid_constant__ CUtensorMap B_tmap,
        char const *SFA_ptr,
        char const *SFB_ptr,
        type::bfloat16_t *C_ptr,
        type::bfloat16_t const *bias_ptr,
        int M,
        int N) {
  static_assert(BLOCK_M == 128, "SM100 MXFP4 tcgen05 MMA uses BLOCK_M == 128");
  static_assert(BLOCK_K % MMA_K == 0, "BLOCK_K must be divisible by MMA_K");
  static_assert(REDUCTION_SIZE % BLOCK_K == 0,
                "K must be divisible by BLOCK_K");
  static_assert(BLOCK_N == 32 || BLOCK_N == 64 || BLOCK_N == 128,
                "BLOCK_N must be 32, 64, or 128");

  // SF gmem atom = 128 rows × 64 K-elements = 512 B (same atom as NVFP4). TMEM
  // uses 4 cols per MMA-K; MXFP4 scale_vec::2X consumes only the first 2 but
  // the addressing stride stays 4.
  constexpr int SF_BYTES_PER_K_TILE = 32 * 4 * 4; // 512 B
  constexpr int SF_TMEM_COLS_PER_MMA_K = 4;

  int const tid = threadIdx.x;
  int const lane_id = tid % WARP_SIZE;
  int const warp_id = tid / WARP_SIZE;

  int const bid_m = blockIdx.x;
  int const bid_n = blockIdx.y;
  int const off_m = bid_m * BLOCK_M;
  int const off_n = bid_n * BLOCK_N;

  constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;

  extern __shared__ __align__(1024) char smem_ptr[];
  int const smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = BLOCK_N * BLOCK_K / 2;
  // SF size: 512 B per (128 rows, 64 K) atom — same as NVFP4. MXFP4's smaller
  // scale count is absorbed by zero-padding the unused 2 k_inner slots.
  constexpr int SFA_size = 128 * BLOCK_K / 16;
  constexpr int SFB_size = 128 * BLOCK_K / 16;
  constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES * 2 + 1];
  int const tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  int const mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  int const mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

  // SF TMEM region: 4 cols per MMA-K × MMA_PER_TILE MMA-Ks (same as NVFP4).
  constexpr int SFA_tmem = BLOCK_N;
  constexpr int SFB_tmem =
      SFA_tmem + SF_TMEM_COLS_PER_MMA_K * (BLOCK_K / MMA_K);

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES * 2 + 1; i++) {
      mbar_init(tma_mbar_addr + i * 8, 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    tmem_alloc<1, BLOCK_N * 2>(smem);
  }
  __syncthreads();

  constexpr int num_iters = REDUCTION_SIZE / BLOCK_K;

  if (warp_id == NUM_WARPS - 2 && elect_sync()) {
    uint64_t cache_A;
    uint64_t cache_B;
    if (M > N) {
      cache_A = EVICT_FIRST;
      cache_B = EVICT_LAST;
    } else {
      cache_A = EVICT_LAST;
      cache_B = EVICT_FIRST;
    }

    auto issue_tma = [&](int iter_k, int stage_id) {
      const int mbar_addr = tma_mbar_addr + stage_id * 8;
      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      const int off_k = iter_k * BLOCK_K;
      tma_load<3, 1>(
          A_smem, &A_tmap, 0, off_m, off_k / 256, mbar_addr, cache_A);
      tma_load<3, 1>(
          B_smem, &B_tmap, 0, off_n, off_k / 256, mbar_addr, cache_B);

      // SF atom = 128 rows × 64 K-elements = 512 B (same as NVFP4).
      const int rest_k = REDUCTION_SIZE / 64;
      const char *SFA_src =
          SFA_ptr + ((off_m / 128) * rest_k + off_k / 64) * SF_BYTES_PER_K_TILE;
      const char *SFB_src =
          SFB_ptr + ((off_n / 128) * rest_k + off_k / 64) * SF_BYTES_PER_K_TILE;
      tma_load_bulk(SFA_smem, SFA_src, SFA_size, mbar_addr, cache_A);
      tma_load_bulk(SFB_smem, SFB_src, SFB_size, mbar_addr, cache_B);

      mbarrier_arrive_expect_tx_tile_local(mbar_addr, STAGE_SIZE);
    };

    constexpr int prefetch_iters =
        (num_iters < NUM_STAGES) ? num_iters : NUM_STAGES;
    for (int iter_k = 0; iter_k < prefetch_iters; iter_k++) {
      issue_tma(iter_k, iter_k);
    }

    for (int iter_k = NUM_STAGES; iter_k < num_iters; iter_k++) {
      int const stage_id = iter_k % NUM_STAGES;
      int const mma_phase = (iter_k / NUM_STAGES - 1) % 2;
      mbar_wait(mma_mbar_addr + stage_id * 8, mma_phase);
      issue_tma(iter_k, stage_id);
    }
  } else if (warp_id == NUM_WARPS - 1 && elect_sync()) {
    constexpr int MMA_N = BLOCK_N;
    constexpr int MMA_M = 128;
    // BlockScaled i_desc: a_format/b_format = MXF4Format::E2M1 = 1,
    // scale_format = ScaleFormat::UE8M0 = 1 (bit 23).
    constexpr uint32_t i_desc = (1U << 7U) | (1U << 10U) |
                                ((uint32_t)MMA_N >> 3U << 17U) | (1U << 23U) |
                                ((uint32_t)MMA_M >> 7U << 27U);

    for (int iter_k = 0; iter_k < num_iters; iter_k++) {
      int const stage_id = iter_k % NUM_STAGES;
      int const tma_phase = (iter_k / NUM_STAGES) % 2;
      mbar_wait(tma_mbar_addr + stage_id * 8, tma_phase);

      int const A_smem = smem + stage_id * STAGE_SIZE;
      int const B_smem = A_smem + A_size;
      int const SFA_smem = B_smem + B_size;
      int const SFB_smem = SFA_smem + SFA_size;

      auto make_desc_AB = [](int addr) -> uint64_t {
        const int SBO = 8 * 128;
        return desc_enc(addr) | (desc_enc(SBO) << 32ULL) | (1ULL << 46ULL) |
               (2ULL << 61ULL);
      };
      auto make_desc_SF = [](int addr) -> uint64_t {
        const int SBO = 8 * 16;
        return desc_enc(addr) | (desc_enc(SBO) << 32ULL) | (1ULL << 46ULL);
      };

      const uint64_t SF_desc = make_desc_SF(0);
      const uint64_t SFA_desc =
          SF_desc + (static_cast<uint64_t>(SFA_smem) >> 4ULL);
      const uint64_t SFB_desc =
          SF_desc + (static_cast<uint64_t>(SFB_smem) >> 4ULL);

      // One cp per atom = 1 MMA-K (same as NVFP4); 4 cp-output cols per MMA-K.
      for (int k = 0; k < BLOCK_K / MMA_K; k++) {
        uint64_t sfa_desc =
            SFA_desc + static_cast<uint64_t>(k) * (SF_BYTES_PER_K_TILE >> 4ULL);
        uint64_t sfb_desc =
            SFB_desc + static_cast<uint64_t>(k) * (SF_BYTES_PER_K_TILE >> 4ULL);
        tcgen05_cp_fp4<1>(SFA_tmem + k * SF_TMEM_COLS_PER_MMA_K, sfa_desc);
        tcgen05_cp_fp4<1>(SFB_tmem + k * SF_TMEM_COLS_PER_MMA_K, sfb_desc);
      }

      for (int k1 = 0; k1 < BLOCK_K / 256; k1++) {
        for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
          uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
          uint64_t b_desc = make_desc_AB(B_smem + k1 * BLOCK_N * 128 + k2 * 32);

          int const k_sf = k1 * 4 + k2;
          // 4 TMEM cols per MMA-K (same stride as NVFP4); MXFP4 vec::2X only
          // reads the first 2 of those 4 cols, but the addressing stride
          // stays 4.
          int const scale_A_tmem = SFA_tmem + k_sf * SF_TMEM_COLS_PER_MMA_K +
                                   (bid_m % (128 / BLOCK_M)) * (BLOCK_M / 32);
          int const scale_B_tmem = SFB_tmem + k_sf * SF_TMEM_COLS_PER_MMA_K +
                                   (bid_n % (128 / BLOCK_N)) * (BLOCK_N / 32);

          int const enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;
          tcgen05_mma_mxfp4<1>(a_desc,
                               b_desc,
                               i_desc,
                               scale_A_tmem,
                               scale_B_tmem,
                               enable_input_d);
        }
      }

      tcgen05_commit_arrive<1>(mma_mbar_addr + stage_id * 8);
    }

    tcgen05_commit_arrive<1>(mainloop_mbar_addr);
  } else if (tid < BLOCK_M) {
    mbar_wait(mainloop_mbar_addr, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    auto epilogue_M_major = [&]() {
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
            C_ptr[offset] =
                type::bfloat16_t(float(acc_bf16) + float(bias_ptr[offset]));
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
    };

    auto epilogue_N_major = [&]() {
      for (int m = 0; m < 32 / 16; m++) {
        float tmp[BLOCK_N / 2];
        if constexpr (BLOCK_N == 128) {
          tcgen05_ld_16x256bx16(tmp, warp_id * 32 + m * 16, 0);
        }
        if constexpr (BLOCK_N == 64) {
          tcgen05_ld_16x256bx8(tmp, warp_id * 32 + m * 16, 0);
        }
        if constexpr (BLOCK_N == 32) {
          tcgen05_ld_16x256bx4(tmp, warp_id * 32 + m * 16, 0);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        for (int i = 0; i < BLOCK_N / 8; i++) {
          const int row = off_m + warp_id * 32 + m * 16 + lane_id / 4;
          const int col = off_n + i * 8 + (lane_id % 4) * 2;
          const int off0 = (row + 0) * N + col;
          const int off1 = (row + 8) * N + col;
          type::bfloat16_t a00(tmp[i * 4 + 0]), a01(tmp[i * 4 + 1]);
          type::bfloat16_t a10(tmp[i * 4 + 2]), a11(tmp[i * 4 + 3]);
          if (bias_ptr != nullptr) {
            C_ptr[off0 + 0] =
                type::bfloat16_t(float(a00) + float(bias_ptr[off0 + 0]));
            C_ptr[off0 + 1] =
                type::bfloat16_t(float(a01) + float(bias_ptr[off0 + 1]));
            C_ptr[off1 + 0] =
                type::bfloat16_t(float(a10) + float(bias_ptr[off1 + 0]));
            C_ptr[off1 + 1] =
                type::bfloat16_t(float(a11) + float(bias_ptr[off1 + 1]));
          } else {
            C_ptr[off0 + 0] = a00;
            C_ptr[off0 + 1] = a01;
            C_ptr[off1 + 0] = a10;
            C_ptr[off1 + 1] = a11;
          }
        }
      }
    };

    if constexpr (C_N_MAJOR) {
      epilogue_N_major();
    } else {
      epilogue_M_major();
    }

    asm volatile("bar.sync 1, %0;" : : "r"(BLOCK_M) : "memory");
    if (warp_id == 0) {
      tmem_dealloc<1, BLOCK_N * 2>(0);
    }
  }
}

} // namespace kernel
