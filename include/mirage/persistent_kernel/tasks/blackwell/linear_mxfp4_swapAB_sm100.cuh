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

// Raw-CUDA rewrite of the small-batch (M<128) MXFP4 swapAB GEMM. Port of the
// NVFP4 swapAB kernel; the MXFP4 deltas (marked `// MXFP4 delta:`) are the same
// ones as the 1d2d 1SM port: scale_vec::2X tcgen05.mma, half-sized per-tile SF
// blocks (256 vs 512 bytes), and TMEM SF stride of 2 cols per MMA-K.

#pragma once

#include "blackwell/sm100_ptx.cuh"           // shared templated PTX primitives
#include "linear_mxfp4_1d2d_sm100.cuh"       // SF_BYTES_PER_K_TILE, SF_TMEM_COLS_PER_MMA_K
#include "linear_mxfp4_1d2d_2sm_sm100.cuh"   // init_C_tmap_mx_2sm

namespace kernel {

using namespace ::kernel::sm100_ptx;

namespace mxfp4_swapAB_detail {

__device__ __forceinline__ void swapab_arrive_local(int mbar_addr) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
               :
               : "r"(mbar_addr)
               : "memory");
}

template <int MMA_N,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int BLOCK_K,
          int NUM_STAGES,
          bool NOBIAS>
__global__ __launch_bounds__(128 + 3 * 32) void
linear_mxfp4_swapAB_sm100_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    const __grid_constant__ CUtensorMap C_tmap,
    const char *SFA_ptr,
    const char *SFB_ptr,
    const type::bfloat16_t *bias_ptr,
    int M,
    int N) {
  static_assert(BLOCK_K == 256, "BLOCK_K must be 256");
  static_assert(BLOCK_K % MMA_K == 0, "BLOCK_K must be divisible by MMA_K");
  static_assert(REDUCTION_SIZE % BLOCK_K == 0, "K must be divisible by BLOCK_K");
  static_assert(MMA_N % 8 == 0 && MMA_N >= 8 && MMA_N <= 128,
                "MMA_N must be a multiple of 8 in [8,128] for the 1SM kernel");

  constexpr int BLOCK_M = 128;

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int cta_idx = static_cast<int>(blockIdx.x);
  const int num_ctas = static_cast<int>(gridDim.x);

  const int num_out_tiles = OUTPUT_SIZE / BLOCK_M;
  const int num_batch_tiles = (M + MMA_N - 1) / MMA_N;
  const int num_tiles = num_out_tiles * num_batch_tiles;

  constexpr int EPILOGUE_WARPS = BLOCK_M / WARP_SIZE;  // 4
  constexpr int MMA_WARP = EPILOGUE_WARPS;
  constexpr int TMA_WARP = EPILOGUE_WARPS + 1;
  constexpr int INIT_WARP = EPILOGUE_WARPS + 2;

  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = MMA_N * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16;  // same atom as NVFP4
  constexpr int SFB_size = 128 * BLOCK_K / 16;
  constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;

  constexpr int ACC_COLS = (MMA_N <= 32)  ? 32
                         : (MMA_N <= 64)  ? 64
                         : (MMA_N <= 128) ? 128
                                          : 256;
  constexpr int MMA_PER_TILE = BLOCK_K / MMA_K;
  // MXFP4 delta: SF stage stride uses SF_TMEM_COLS_PER_MMA_K (=2) cols per mma_k.
  constexpr int SF_STAGE_STRIDE = SF_TMEM_COLS_PER_MMA_K * MMA_PER_TILE;
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
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
  const int output_mbar_addr = mainloop_mbar_addr + NUM_ACC_BUF * 8;

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
    if constexpr (TMEM_ALLOC_COLS == 128) tmem_alloc<1, 128>(smem);
    else if constexpr (TMEM_ALLOC_COLS == 256) tmem_alloc<1, 256>(smem);
    else tmem_alloc<1, 512>(smem);
  }
  __syncthreads();

  constexpr int num_iters = REDUCTION_SIZE / BLOCK_K;
  constexpr int rest_k = REDUCTION_SIZE / 64;  // atoms per row (same as NVFP4)

  auto decode_tile = [&](int t, int &off_m, int &off_n, int &batch_tile) {
    const int out_tile = t / num_batch_tiles;
    batch_tile = t % num_batch_tiles;
    off_m = out_tile * BLOCK_M;
    off_n = batch_tile * MMA_N;
  };

  if (warp_id == TMA_WARP && elect_sync()) {
    const uint64_t cache_A = EVICT_LAST;
    const uint64_t cache_B = EVICT_FIRST;

    auto issue_tma = [&](int iter_k, int stage_id, int off_m, int off_n,
                         int batch_tile) {
      const int ab_mbar = tma_mbar_addr + stage_id * 8;
      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      const int off_k = iter_k * BLOCK_K;
      tma_load<3, 1>(A_smem, &A_tmap, 0, off_m, off_k / 256, ab_mbar, cache_A);
      tma_load<3, 1>(B_smem, &B_tmap, 0, off_n, off_k / 256, ab_mbar, cache_B);

      // SF atom = 128 rows × 64 K-elements = 512 B (same as NVFP4).
      const char *SFA_src = SFA_ptr + ((off_m / 128) * rest_k + off_k / 64) * SF_BYTES_PER_K_TILE;
      const char *SFB_src = SFB_ptr + (batch_tile * rest_k + off_k / 64) * SF_BYTES_PER_K_TILE;
      tma_load_bulk(SFA_smem, SFA_src, SFA_size, ab_mbar, cache_A);
      tma_load_bulk(SFB_smem, SFB_src, SFB_size, ab_mbar, cache_B);

      mbarrier_arrive_expect_tx_tile_local(ab_mbar, STAGE_SIZE);
    };

    for (int t = cta_idx, work_idx = 0; t < num_tiles; t += num_ctas, work_idx++) {
      int off_m, off_n, batch_tile;
      decode_tile(t, off_m, off_n, batch_tile);
      for (int iter_k = 0; iter_k < num_iters; iter_k++) {
        const int pipeline_iter = work_idx * num_iters + iter_k;
        const int stage_id = pipeline_iter % NUM_STAGES;
        if (pipeline_iter >= NUM_STAGES) {
          const int mma_phase = ((pipeline_iter - NUM_STAGES) / NUM_STAGES) % 2;
          mbar_wait(mma_mbar_addr + stage_id * 8, mma_phase);
        }
        issue_tma(iter_k, stage_id, off_m, off_n, batch_tile);
      }
    }
  }
  else if (warp_id == MMA_WARP && elect_sync()) {
    constexpr int MMA_M = 128;
    // BlockScaled i_desc with scale_format=UE8M0 (bit 23) for MXFP4.
    constexpr uint32_t i_desc = (1U << 7U) | (1U << 10U) |
                                ((uint32_t)MMA_N >> 3U << 17U) |
                                (1U << 23U) |
                                ((uint32_t)MMA_M >> 7U << 27U);

    auto make_desc_AB = [](int addr) -> uint64_t {
      const int SBO = 8 * 128;
      return desc_enc(addr) | (desc_enc(SBO) << 32ULL) |
             (1ULL << 46ULL) | (2ULL << 61ULL);
    };
    auto make_desc_SF = [](int addr) -> uint64_t {
      const int SBO = 8 * 16;
      return desc_enc(addr) | (desc_enc(SBO) << 32ULL) | (1ULL << 46ULL);
    };

    for (int t = cta_idx, work_idx = 0; t < num_tiles; t += num_ctas, work_idx++) {
      const int acc_buf = work_idx % NUM_ACC_BUF;
      const int acc_base = acc_buf * ACC_COLS;

      if (work_idx >= NUM_ACC_BUF) {
        const int buf_phase = ((work_idx - NUM_ACC_BUF) / NUM_ACC_BUF) % 2;
        mbar_wait(output_mbar_addr + acc_buf * 8, buf_phase);
      }

      for (int iter_k = 0; iter_k < num_iters; iter_k++) {
        const int pipeline_iter = work_idx * num_iters + iter_k;
        const int stage_id = pipeline_iter % NUM_STAGES;
        const int tma_phase = (pipeline_iter / NUM_STAGES) % 2;

        const int A_smem = smem + stage_id * STAGE_SIZE;
        const int B_smem = A_smem + A_size;
        const int SFA_smem = B_smem + B_size;
        const int SFB_smem = SFA_smem + SFA_size;

        const uint64_t SFA_desc = make_desc_SF(SFA_smem);
        const uint64_t SFB_desc = make_desc_SF(SFB_smem);

        mbar_wait(tma_mbar_addr + stage_id * 8, tma_phase);

        // One cp per MMA-K (NVFP4-style; 4 cols per cp).
        for (int k = 0; k < BLOCK_K / MMA_K; k++) {
          uint64_t sfa_desc = SFA_desc + static_cast<uint64_t>(k) * (SF_BYTES_PER_K_TILE >> 4ULL);
          uint64_t sfb_desc = SFB_desc + static_cast<uint64_t>(k) * (SF_BYTES_PER_K_TILE >> 4ULL);
          tcgen05_cp_fp4<1>(SFA_tmem + stage_id * SF_STAGE_STRIDE + k * SF_TMEM_COLS_PER_MMA_K, sfa_desc);
          tcgen05_cp_fp4<1>(SFB_tmem + stage_id * SF_STAGE_STRIDE + k * SF_TMEM_COLS_PER_MMA_K, sfb_desc);
        }

        for (int k1 = 0; k1 < BLOCK_K / 256; k1++) {
          for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
            const int k_sf = k1 * 4 + k2;
            uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
            uint64_t b_desc = make_desc_AB(B_smem + k1 * MMA_N * 128 + k2 * 32);
            const int scale_A_tmem = SFA_tmem + stage_id * SF_STAGE_STRIDE + k_sf * SF_TMEM_COLS_PER_MMA_K;
            const int scale_B_tmem = SFB_tmem + stage_id * SF_STAGE_STRIDE + k_sf * SF_TMEM_COLS_PER_MMA_K;
            const int enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;
            tcgen05_mma_mxfp4<1>(a_desc, b_desc, i_desc, scale_A_tmem,
                                 scale_B_tmem, enable_input_d, acc_base);
          }
        }

        tcgen05_commit_arrive<1>(mma_mbar_addr + stage_id * 8);
      }

      tcgen05_commit_arrive<1>(mainloop_mbar_addr + acc_buf * 8);
    }
  }
  else if (warp_id < EPILOGUE_WARPS) {
    type::bfloat16_t *out_smem_base =
        reinterpret_cast<type::bfloat16_t *>(smem_ptr + STAGE_SIZE * NUM_STAGES);
    const int out_smem_base_addr = smem + STAGE_SIZE * NUM_STAGES;
    constexpr int OUT_TILE_ELEMS = MMA_N * BLOCK_M;
    constexpr int OUT_TILE_BYTES = OUT_TILE_ELEMS * (int)sizeof(type::bfloat16_t);
    const int out_row = warp_id * 32 + (tid % WARP_SIZE);

    for (int t = cta_idx, work_idx = 0; t < num_tiles; t += num_ctas, work_idx++) {
      int off_m, off_n, batch_tile;
      decode_tile(t, off_m, off_n, batch_tile);

      const int acc_buf = work_idx % NUM_ACC_BUF;
      const int acc_base = acc_buf * ACC_COLS;
      const int buf_phase = (work_idx / NUM_ACC_BUF) % 2;
      type::bfloat16_t *out_smem = out_smem_base + acc_buf * OUT_TILE_ELEMS;
      const int out_smem_addr = out_smem_base_addr + acc_buf * OUT_TILE_BYTES;

      mbar_wait(mainloop_mbar_addr + acc_buf * 8, buf_phase);
      asm volatile("tcgen05.fence::after_thread_sync;");

      float tmp[ACC_COLS];
      if constexpr (ACC_COLS == 256) {
        tcgen05_ld_32x32bx128(tmp,       warp_id * 32, acc_base);
        tcgen05_ld_32x32bx128(tmp + 128, warp_id * 32, acc_base + 128);
      } else if constexpr (ACC_COLS == 128) {
        tcgen05_ld_32x32bx128(tmp, warp_id * 32, acc_base);
      } else if constexpr (ACC_COLS == 64) {
        tcgen05_ld_32x32bx64(tmp, warp_id * 32, acc_base);
      } else {  // ACC_COLS == 32
        tcgen05_ld_32x32bx32(tmp, warp_id * 32, acc_base);
      }
      asm volatile("tcgen05.wait::ld.sync.aligned;");

      for (int j = 0; j < MMA_N; j++) {
        type::bfloat16_t acc_bf16(tmp[j]);
        if constexpr (!NOBIAS) {
          const int boff = (off_n + j) * N + (off_m + out_row);
          acc_bf16 = type::bfloat16_t(float(acc_bf16) + float(bias_ptr[boff]));
        }
        out_smem[j * BLOCK_M + out_row] = acc_bf16;
      }

      tma_store_fence();
      asm volatile("bar.sync 1, %0;" : : "r"(BLOCK_M) : "memory");
      if (warp_id == 0 && elect_sync()) {
        tma_store_2d(out_smem_addr, &C_tmap, off_m, off_n);
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
    if constexpr (TMEM_ALLOC_COLS == 128) tmem_dealloc<1, 128>(0);
    else if constexpr (TMEM_ALLOC_COLS == 256) tmem_dealloc<1, 256>(0);
    else tmem_dealloc<1, 512>(0);
  }
}

}  // namespace mxfp4_swapAB_detail

using mxfp4_swapAB_detail::linear_mxfp4_swapAB_sm100_kernel;

}  // namespace kernel
