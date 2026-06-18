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

// Small-batch (M<128) NVFP4 swapAB GEMM. Runs the MMA as C^T = W * X^T so
// weight rows (N) map to MMA_M=128 and the batch (M) maps to MMA_N. Versus
// the 1d2d 1SM kernel: MMA_N is narrow (8/16/32/64/128), SFA uses the per-
// tile layout indexed by batch tile, and the epilogue TMA-stores a (MMA_N,
// 128) smem tile to row-major C[M,N].

#pragma once

#include "../common/bfloat16.h"
#include "sm100_ptx.cuh" // templated PTX primitives shared across fp4 kernels

namespace kernel {

namespace nvfp4_swapAB_detail {

// Shared GEMM body. A/B/C are TMA descriptors (passed by pointer so both the
// __global__ launcher and the MPK task can call it); SFA/SFB/bias are raw
// pointers. (cta_idx, num_ctas) parametrize the persistent tile grid-stride:
// the __global__ kernel passes (blockIdx.x, gridDim.x); an MPK single-CTA task
// passes (0, 1) to sweep every tile itself.
template <int MMA_N,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int BLOCK_K,
          int NUM_STAGES,
          bool NOBIAS>
__device__ __forceinline__ void linear_nvfp4_swapAB_sm100_task_impl(
    CUtensorMap const *A_tmap,
    CUtensorMap const *B_tmap,
    CUtensorMap const *C_tmap,
    char const *SFA_ptr,
    char const *SFB_ptr,
    type::bfloat16_t const *bias_ptr,
    int M,
    int N,
    int cta_idx,
    int num_ctas) {
  // Function-local so sm100_ptx primitives are in scope without leaking into
  // kernel:: (other tasks in the shared megakernel TU define their own
  // WARP_SIZE/MMA_K). Tunables are owned here for the same reason.
  using namespace ::kernel::sm100_ptx;
  constexpr int WARP_SIZE = 32;
  constexpr int MMA_K = 64;
  constexpr uint64_t EVICT_FIRST = 0x12F0000000000000ULL;
  constexpr uint64_t EVICT_LAST = 0x14F0000000000000ULL;
  static_assert(BLOCK_K == 256, "BLOCK_K must be 256");
  static_assert(BLOCK_K % MMA_K == 0, "BLOCK_K must be divisible by MMA_K");
  static_assert(REDUCTION_SIZE % BLOCK_K == 0,
                "K must be divisible by BLOCK_K");
  static_assert(MMA_N % 8 == 0 && MMA_N >= 8 && MMA_N <= 128,
                "MMA_N must be a multiple of 8 in [8,128]");

  constexpr int BLOCK_M = 128;

  int const tid = threadIdx.x;
  int const warp_id = tid / WARP_SIZE;

  int const num_out_tiles = OUTPUT_SIZE / BLOCK_M;
  int const num_batch_tiles = (M + MMA_N - 1) / MMA_N;
  int const num_tiles = num_out_tiles * num_batch_tiles;

  constexpr int EPILOGUE_WARPS = BLOCK_M / WARP_SIZE;
  constexpr int MMA_WARP = EPILOGUE_WARPS;
  constexpr int TMA_WARP = EPILOGUE_WARPS + 1;
  constexpr int INIT_WARP = EPILOGUE_WARPS + 2;

  extern __shared__ __align__(1024) char smem_ptr[];
  int const smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = MMA_N * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16;
  constexpr int SFB_size = 128 * BLOCK_K / 16;
  constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;

  constexpr int ACC_COLS = (MMA_N <= 32)    ? 32
                           : (MMA_N <= 64)  ? 64
                           : (MMA_N <= 128) ? 128
                                            : 256;
  constexpr int MMA_PER_TILE = BLOCK_K / MMA_K;
  constexpr int SF_STAGE_STRIDE = 4 * MMA_PER_TILE;
  constexpr int SF_TOTAL = 2 * SF_STAGE_STRIDE * NUM_STAGES;
  constexpr int NUM_ACC_BUF = (2 * ACC_COLS + SF_TOTAL <= 512) ? 2 : 1;
  constexpr int SFA_tmem = NUM_ACC_BUF * ACC_COLS;
  constexpr int SFB_tmem = SFA_tmem + SF_STAGE_STRIDE * NUM_STAGES;
  constexpr int TMEM_USED_COLS = SFB_tmem + SF_STAGE_STRIDE * NUM_STAGES;
  static_assert(TMEM_USED_COLS <= 512, "TMEM oversubscribed; lower NUM_STAGES");
  constexpr int TMEM_ALLOC_COLS =
      (TMEM_USED_COLS <= 128) ? 128 : ((TMEM_USED_COLS <= 256) ? 256 : 512);

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES * 2 + 2 * NUM_ACC_BUF];
  int const tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  int const mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  int const mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
  int const output_mbar_addr = mainloop_mbar_addr + NUM_ACC_BUF * 8;

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES; i++) {
      mbar_init(tma_mbar_addr + i * 8, 1);
      mbar_init(mma_mbar_addr + i * 8, 1);
    }
    for (int i = 0; i < NUM_ACC_BUF; i++) {
      mbar_init(mainloop_mbar_addr + i * 8, 1);
      mbar_init(output_mbar_addr + i * 8, 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == INIT_WARP) {
    if constexpr (TMEM_ALLOC_COLS == 128) {
      tmem_alloc<1, 128>(smem);
    } else if constexpr (TMEM_ALLOC_COLS == 256) {
      tmem_alloc<1, 256>(smem);
    } else {
      tmem_alloc<1, 512>(smem);
    }
  }
  __syncthreads();

  constexpr int num_iters = REDUCTION_SIZE / BLOCK_K;
  constexpr int rest_k = REDUCTION_SIZE / 64;

  auto decode_tile = [&](int t, int &off_m, int &off_n, int &batch_tile) {
    const int out_tile = t / num_batch_tiles;
    batch_tile = t % num_batch_tiles;
    off_m = out_tile * BLOCK_M;
    off_n = batch_tile * MMA_N;
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

  if (warp_id == TMA_WARP && elect_sync()) {
    // Weight (A) is large; keep resident. Activation (B) is small; evict.
    const uint64_t cache_A = EVICT_LAST;
    const uint64_t cache_B = EVICT_FIRST;

    auto issue_tma = [&](int iter_k,
                         int stage_id,
                         int off_m,
                         int off_n,
                         int batch_tile) {
      const int ab_mbar = tma_mbar_addr + stage_id * 8;
      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      const int off_k = iter_k * BLOCK_K;
      tma_load<3, 1>(A_smem, A_tmap, 0, off_m, off_k / 256, ab_mbar, cache_A);
      tma_load<3, 1>(B_smem, B_tmap, 0, off_n, off_k / 256, ab_mbar, cache_B);

      const char *SFA_src =
          SFA_ptr + ((off_m / 128) * rest_k + off_k / 64) * 512;
      const char *SFB_src = SFB_ptr + (batch_tile * rest_k + off_k / 64) * 512;
      tma_load_bulk(SFA_smem, SFA_src, SFA_size, ab_mbar, cache_A);
      tma_load_bulk(SFB_smem, SFB_src, SFB_size, ab_mbar, cache_B);

      mbarrier_arrive_expect_tx_tile_local(ab_mbar, STAGE_SIZE);
    };

    for (int t = cta_idx, work_idx = 0; t < num_tiles;
         t += num_ctas, work_idx++) {
      int off_m, off_n, batch_tile;
      decode_tile(t, off_m, off_n, batch_tile);
      for (int iter_k = 0; iter_k < num_iters; iter_k++) {
        int const pipeline_iter = work_idx * num_iters + iter_k;
        int const stage_id = pipeline_iter % NUM_STAGES;
        if (pipeline_iter >= NUM_STAGES) {
          int const mma_phase = ((pipeline_iter - NUM_STAGES) / NUM_STAGES) % 2;
          mbar_wait(mma_mbar_addr + stage_id * 8, mma_phase);
        }
        issue_tma(iter_k, stage_id, off_m, off_n, batch_tile);
      }
    }
  } else if (warp_id == MMA_WARP && elect_sync()) {
    constexpr int MMA_M = 128;
    constexpr uint32_t i_desc = (1U << 7U) | (1U << 10U) |
                                ((uint32_t)MMA_N >> 3U << 17U) |
                                ((uint32_t)MMA_M >> 7U << 27U);

    for (int t = cta_idx, work_idx = 0; t < num_tiles;
         t += num_ctas, work_idx++) {
      int const acc_buf = work_idx % NUM_ACC_BUF;
      int const acc_base = acc_buf * ACC_COLS;

      if (work_idx >= NUM_ACC_BUF) {
        int const buf_phase = ((work_idx - NUM_ACC_BUF) / NUM_ACC_BUF) % 2;
        mbar_wait(output_mbar_addr + acc_buf * 8, buf_phase);
      }

      for (int iter_k = 0; iter_k < num_iters; iter_k++) {
        int const pipeline_iter = work_idx * num_iters + iter_k;
        int const stage_id = pipeline_iter % NUM_STAGES;
        int const tma_phase = (pipeline_iter / NUM_STAGES) % 2;

        int const A_smem = smem + stage_id * STAGE_SIZE;
        int const B_smem = A_smem + A_size;
        int const SFA_smem = B_smem + B_size;
        int const SFB_smem = SFA_smem + SFA_size;

        const uint64_t SFA_desc = make_desc_SF(SFA_smem);
        const uint64_t SFB_desc = make_desc_SF(SFB_smem);

        mbar_wait(tma_mbar_addr + stage_id * 8, tma_phase);

        for (int k = 0; k < BLOCK_K / MMA_K; k++) {
          uint64_t sfa_desc =
              SFA_desc + static_cast<uint64_t>(k) * (512ULL >> 4ULL);
          uint64_t sfb_desc =
              SFB_desc + static_cast<uint64_t>(k) * (512ULL >> 4ULL);
          tcgen05_cp_fp4<1>(SFA_tmem + stage_id * SF_STAGE_STRIDE + k * 4,
                            sfa_desc);
          tcgen05_cp_fp4<1>(SFB_tmem + stage_id * SF_STAGE_STRIDE + k * 4,
                            sfb_desc);
        }

        for (int k1 = 0; k1 < BLOCK_K / 256; k1++) {
          for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
            int const k_sf = k1 * 4 + k2;
            uint64_t a_desc =
                make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
            uint64_t b_desc = make_desc_AB(B_smem + k1 * MMA_N * 128 + k2 * 32);
            int const scale_A_tmem =
                SFA_tmem + stage_id * SF_STAGE_STRIDE + k_sf * 4;
            int const scale_B_tmem =
                SFB_tmem + stage_id * SF_STAGE_STRIDE + k_sf * 4;
            int const enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;
            tcgen05_mma_nvfp4<1>(a_desc,
                                 b_desc,
                                 i_desc,
                                 scale_A_tmem,
                                 scale_B_tmem,
                                 enable_input_d,
                                 acc_base);
          }
        }

        tcgen05_commit_arrive<1>(mma_mbar_addr + stage_id * 8);
      }

      tcgen05_commit_arrive<1>(mainloop_mbar_addr + acc_buf * 8);
    }
  } else if (warp_id < EPILOGUE_WARPS) {
    type::bfloat16_t *out_smem_base = reinterpret_cast<type::bfloat16_t *>(
        smem_ptr + STAGE_SIZE * NUM_STAGES);
    int const out_smem_base_addr = smem + STAGE_SIZE * NUM_STAGES;
    constexpr int OUT_TILE_ELEMS = MMA_N * BLOCK_M;
    constexpr int OUT_TILE_BYTES =
        OUT_TILE_ELEMS * (int)sizeof(type::bfloat16_t);
    int const out_row = warp_id * 32 + (tid % WARP_SIZE);

    for (int t = cta_idx, work_idx = 0; t < num_tiles;
         t += num_ctas, work_idx++) {
      int off_m, off_n, batch_tile;
      decode_tile(t, off_m, off_n, batch_tile);

      int const acc_buf = work_idx % NUM_ACC_BUF;
      int const acc_base = acc_buf * ACC_COLS;
      int const buf_phase = (work_idx / NUM_ACC_BUF) % 2;
      type::bfloat16_t *out_smem = out_smem_base + acc_buf * OUT_TILE_ELEMS;
      int const out_smem_addr = out_smem_base_addr + acc_buf * OUT_TILE_BYTES;

      mbar_wait(mainloop_mbar_addr + acc_buf * 8, buf_phase);
      asm volatile("tcgen05.fence::after_thread_sync;");

      float tmp[ACC_COLS];
      if constexpr (ACC_COLS == 256) {
        tcgen05_ld_32x32bx128(tmp, warp_id * 32, acc_base);
        tcgen05_ld_32x32bx128(tmp + 128, warp_id * 32, acc_base + 128);
      } else if constexpr (ACC_COLS == 128) {
        tcgen05_ld_32x32bx128(tmp, warp_id * 32, acc_base);
      } else if constexpr (ACC_COLS == 64) {
        tcgen05_ld_32x32bx64(tmp, warp_id * 32, acc_base);
      } else { // ACC_COLS == 32
        tcgen05_ld_32x32bx32(tmp, warp_id * 32, acc_base);
      }
      asm volatile("tcgen05.wait::ld.sync.aligned;");

      for (int j = 0; j < MMA_N; j++) {
        type::bfloat16_t acc_bf16(tmp[j]);
        if constexpr (!NOBIAS) {
          int const boff = (off_n + j) * N + (off_m + out_row);
          acc_bf16 = type::bfloat16_t(float(acc_bf16) + float(bias_ptr[boff]));
        }
        out_smem[j * BLOCK_M + out_row] = acc_bf16;
      }

      tma_store_fence();
      asm volatile("bar.sync 1, %0;" : : "r"(BLOCK_M) : "memory");
      if (warp_id == 0 && elect_sync()) {
        tma_store_2d(out_smem_addr, C_tmap, off_m, off_n);
        tma_store_commit();
        tma_store_wait<0>();
      }

      asm volatile("bar.sync 1, %0;" : : "r"(BLOCK_M) : "memory");
      if (warp_id == 0 && elect_sync()) {
        swapab_arrive_local(output_mbar_addr + acc_buf * 8);
      }
    }
  }

  __syncthreads();
  if (warp_id == 0) {
    if constexpr (TMEM_ALLOC_COLS == 128) {
      tmem_dealloc<1, 128>(0);
    } else if constexpr (TMEM_ALLOC_COLS == 256) {
      tmem_dealloc<1, 256>(0);
    } else {
      tmem_dealloc<1, 512>(0);
    }
  }
}

// Standalone grid launcher: TMA descriptors arrive by value (__grid_constant__),
// tiles are grid-strided over (blockIdx.x, gridDim.x).
template <int MMA_N,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int BLOCK_K,
          int NUM_STAGES,
          bool NOBIAS>
__global__
    __launch_bounds__(128 + 3 * 32) void linear_nvfp4_swapAB_sm100_kernel(
        const __grid_constant__ CUtensorMap A_tmap,
        const __grid_constant__ CUtensorMap B_tmap,
        const __grid_constant__ CUtensorMap C_tmap,
        char const *SFA_ptr,
        char const *SFB_ptr,
        type::bfloat16_t const *bias_ptr,
        int M,
        int N) {
  linear_nvfp4_swapAB_sm100_task_impl<MMA_N,
                                      OUTPUT_SIZE,
                                      REDUCTION_SIZE,
                                      BLOCK_K,
                                      NUM_STAGES,
                                      NOBIAS>(&A_tmap,
                                              &B_tmap,
                                              &C_tmap,
                                              SFA_ptr,
                                              SFB_ptr,
                                              bias_ptr,
                                              M,
                                              N,
                                              static_cast<int>(blockIdx.x),
                                              static_cast<int>(gridDim.x));
}

} // namespace nvfp4_swapAB_detail

using nvfp4_swapAB_detail::linear_nvfp4_swapAB_sm100_kernel;
using nvfp4_swapAB_detail::linear_nvfp4_swapAB_sm100_task_impl;

} // namespace kernel
