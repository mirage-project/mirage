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
#include "blackwell/linear_fp4_primitives_sm100.cuh"

#include <c10/util/Exception.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

inline void check_cu(CUresult err) {
  if (err == CUDA_SUCCESS) {
    return;
  }
  const char *error_msg_ptr = nullptr;
  if (cuGetErrorString(err, &error_msg_ptr) != CUDA_SUCCESS) {
    error_msg_ptr = "unable to get error string";
  }
  TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", error_msg_ptr);
}

inline void check_cuda(cudaError_t err) {
  if (err == cudaSuccess) {
    return;
  }
  TORCH_CHECK(false, cudaGetErrorString(err));
}

inline void init_AB_tmap(CUtensorMap *tmap, 
                         const char *ptr,
                         uint64_t global_height,
                         uint64_t global_width,
                         uint32_t shared_height,
                         uint32_t shared_width) {
  constexpr uint32_t rank = 3;
  uint64_t globalDim[rank] = {256, global_height, global_width / 256};
  uint64_t globalStrides[rank - 1] = {global_width / 2, 128};
  uint32_t boxDim[rank] = {256, shared_height, shared_width / 256};
  uint32_t elementStrides[rank] = {1, 1, 1};

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
  check_cu(err);
}

namespace kernel {

template <int REDUCTION_SIZE,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          int NUM_STAGES,
          int EPI_BATCH_LA = 1>
__global__ __launch_bounds__(BLOCK_M + 2 * 32) void
linear_nvfp4_1d2d_sm100_kernel(const __grid_constant__ CUtensorMap A_tmap,
                               const __grid_constant__ CUtensorMap B_tmap,
                               const char *SFA_ptr,
                               const char *SFB_ptr,
                               type::bfloat16_t *C_ptr,
                               const type::bfloat16_t *bias_ptr,
                               int M,
                               int N) {
  static_assert(BLOCK_M == 128, "SM100 NVFP4 tcgen05 MMA uses BLOCK_M == 128");
  static_assert(BLOCK_K % MMA_K == 0, "BLOCK_K must be divisible by MMA_K");
  static_assert(REDUCTION_SIZE % BLOCK_K == 0, "K must be divisible by BLOCK_K");
  static_assert(BLOCK_N == 32 || BLOCK_N == 64 || BLOCK_N == 128, "BLOCK_N must be 32, 64, or 128");

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;

  const int bid_m = blockIdx.x;
  const int bid_n = blockIdx.y;
  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;

  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = BLOCK_N * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16;
  constexpr int SFB_size = 128 * BLOCK_K / 16;
  constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES * 2 + 1];
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

  constexpr int SFA_tmem = BLOCK_N;
  constexpr int SFB_tmem = SFA_tmem + 4 * (BLOCK_K / MMA_K);

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES * 2 + 1; i++) {
      mbarrier_init(tma_mbar_addr + i * 8, 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    tmem_alloc<1, BLOCK_N * 2>(smem);
  }
  __syncthreads();

  constexpr int NUM_ITERS = REDUCTION_SIZE / BLOCK_K;

  auto make_desc_AB = [](int addr) -> uint64_t {
    const int SBO = 8 * 128;
    return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
  };
  
  auto make_desc_SF = [](int addr) -> uint64_t {
    const int SBO = 8 * 16;
    return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
  };

  auto issue_tma = [&](int iter_k, int stage_id) {
    uint64_t cache_A = (M > N) ? EVICT_FIRST : EVICT_LAST;
    uint64_t cache_B = (M > N) ? EVICT_LAST : EVICT_FIRST;

    const int mbar_addr = tma_mbar_addr + stage_id * 8;
    const int A_smem = smem + stage_id * STAGE_SIZE;
    const int B_smem = A_smem + A_size;
    const int SFA_smem = B_smem + B_size;
    const int SFB_smem = SFA_smem + SFA_size;
    const int off_k = iter_k * BLOCK_K;

    tma_load<3, 1>(A_smem, &A_tmap, 0, off_m, off_k / 256, mbar_addr, cache_A);
    tma_load<3, 1>(B_smem, &B_tmap, 0, off_n, off_k / 256, mbar_addr, cache_B);

    const int rest_k = REDUCTION_SIZE / 64;
    const char *SFA_src = SFA_ptr + ((off_m / 128) * rest_k + off_k / 64) * 512;
    const char *SFB_src = SFB_ptr + ((off_n / 128) * rest_k + off_k / 64) * 512;

    tma_load_bulk(SFA_smem, SFA_src, SFA_size, mbar_addr, cache_A);
    tma_load_bulk(SFB_smem, SFB_src, SFB_size, mbar_addr, cache_B);

    mbarrier_arrive_expect_tx_tile_local(mbar_addr, STAGE_SIZE);
  };

  if (warp_id == NUM_WARPS - 2 && elect_sync()) {
    constexpr int PREFETCH_ITERS = (NUM_ITERS < NUM_STAGES) ? NUM_ITERS : NUM_STAGES;
    
    for (int iter_k = 0; iter_k < PREFETCH_ITERS; iter_k++) {
      issue_tma(iter_k, iter_k);
    }
    
    for (int iter_k = NUM_STAGES; iter_k < NUM_ITERS; iter_k++) {
      const int stage_id = iter_k % NUM_STAGES;
      const int mma_phase = (iter_k / NUM_STAGES - 1) % 2;
      mbarrier_wait_cta(mma_mbar_addr + stage_id * 8, mma_phase);
      issue_tma(iter_k, stage_id);
    }

  } else if (warp_id == NUM_WARPS - 1 && elect_sync()) {
    constexpr int MMA_N = BLOCK_N;
    constexpr int MMA_M = 128;
    constexpr uint32_t i_desc = (1U << 7U) | (1U << 10U) | ((uint32_t)MMA_N >> 3U << 17U) | ((uint32_t)MMA_M >> 7U << 27U);

    for (int iter_k = 0; iter_k < NUM_ITERS; iter_k++) {
      const int stage_id = iter_k % NUM_STAGES;
      const int tma_phase = (iter_k / NUM_STAGES) % 2;

      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      const uint64_t SF_desc = make_desc_SF(0);
      const uint64_t SFA_desc = SF_desc + (static_cast<uint64_t>(SFA_smem) >> 4ULL);
      const uint64_t SFB_desc = SF_desc + (static_cast<uint64_t>(SFB_smem) >> 4ULL);

      mbarrier_wait_cta(tma_mbar_addr + stage_id * 8, tma_phase);

      for (int k = 0; k < BLOCK_K / MMA_K; k++) {
        uint64_t sfa_desc = SFA_desc + static_cast<uint64_t>(k) * (512ULL >> 4ULL);
        uint64_t sfb_desc = SFB_desc + static_cast<uint64_t>(k) * (512ULL >> 4ULL);
        tcgen05_cp_fp4<1>(SFA_tmem + k * 4, sfa_desc);
        tcgen05_cp_fp4<1>(SFB_tmem + k * 4, sfb_desc);
      }

      for (int k = 0; k < BLOCK_K / MMA_K; k++) {
        const int k1 = k / 4;
        const int k2 = k % 4;

        uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
        uint64_t b_desc = make_desc_AB(B_smem + k1 * BLOCK_N * 128 + k2 * 32);
        const int scale_A_tmem = SFA_tmem + k * 4 + (bid_m % (128 / BLOCK_M)) * (BLOCK_M / 32);
        const int scale_B_tmem = SFB_tmem + k * 4 + (bid_n % (128 / BLOCK_N)) * (BLOCK_N / 32);
        const int enable_input_d = (k == 0) ? iter_k : 1;

        tcgen05_mma_nvfp4<1>(a_desc, b_desc, i_desc, scale_A_tmem, scale_B_tmem, enable_input_d);
      }
      tcgen05_commit_arrive<1>(mma_mbar_addr + stage_id * 8);
    }
    tcgen05_commit_arrive<1>(mainloop_mbar_addr);
  } else if (tid < BLOCK_M) {
    mbarrier_wait_cta(mainloop_mbar_addr, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    auto epilogue_M_major = [&]() {
      constexpr int WIDTH = (BLOCK_N < 64) ? BLOCK_N : 64;
      constexpr int NUM_SUBTILES = BLOCK_N / WIDTH;
      constexpr int BATCH = (EPI_BATCH_LA <= NUM_SUBTILES && NUM_SUBTILES % EPI_BATCH_LA == 0)
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
    };

    epilogue_M_major();

    asm volatile("bar.sync 1, %0;" : : "r"(BLOCK_M) : "memory");
    if (warp_id == 0) {
      tmem_dealloc<1, BLOCK_N * 2>(0);
    }
  }
}

}  // namespace kernel
