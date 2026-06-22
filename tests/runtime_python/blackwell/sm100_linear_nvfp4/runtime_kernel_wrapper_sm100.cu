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

#include "blackwell/linear_nvfp4_1d2d_2sm_sm100.cuh"
#include "blackwell/linear_nvfp4_1d2d_sm100.cuh"
#include "blackwell/linear_nvfp4_swapAB_sm100.cuh"
#include "blackwell/quantize_nvfp4_sm100.cuh"
#include "hopper/tma_3d.cuh"
#include "hopper/tma_fp4.cuh"
#include "runtime_header.h"
#include "tma.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/util/print_error.hpp>

#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/tensor.hpp>

using float_e2m1 = cute::float_e2m1_t;
using float_ue4m3 = cute::float_ue4m3_t;
using bfloat16 = cute::bfloat16_t;

template <typename T,
          int REDUCTION_SIZE,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          int NUM_STAGES,
          int EPI_BATCH_LA = 1>
void launch_linear_nvfp4_1d2d_sm100_config(void *input_ptr,
                                           void *input_sf_ptr,
                                           void *weight_ptr,
                                           void *weight_sf_ptr,
                                           void *output_ptr,
                                           void *residual_ptr,
                                           int batch_size,
                                           int output_size) {
  static_assert(BLOCK_M == 128, "1d2d NVFP4 uses a 128-row A tile");
  static_assert(BLOCK_K == 256, "1d2d NVFP4 dispatch table uses BLOCK_K=256");
  (void)sizeof(T);

  TORCH_CHECK(batch_size % 128 == 0,
              "SM100 NVFP4 1d2d requires batch_size divisible by 128, got ",
              batch_size);
  TORCH_CHECK(batch_size % BLOCK_N == 0,
              "SM100 NVFP4 1d2d requires swapped-frame N divisible by BLOCK_N=",
              BLOCK_N,
              ", got batch_size=",
              batch_size);
  TORCH_CHECK(output_size % BLOCK_M == 0,
              "SM100 NVFP4 1d2d requires output_size divisible by BLOCK_M=",
              BLOCK_M,
              ", got output_size=",
              output_size);
  TORCH_CHECK(REDUCTION_SIZE % BLOCK_K == 0,
              "SM100 NVFP4 1d2d requires K divisible by BLOCK_K");

  CUtensorMap A_tmap{};
  CUtensorMap B_tmap{};
  // Always SWAP_AB: A is weight [output, K], B is input [batch, K].
  kernel::tma::init_AB_tmap_fp4(&A_tmap,
                                reinterpret_cast<char const *>(weight_ptr),
                                static_cast<uint64_t>(output_size),
                                static_cast<uint64_t>(REDUCTION_SIZE),
                                BLOCK_M,
                                BLOCK_K);
  kernel::tma::init_AB_tmap_fp4(&B_tmap,
                                reinterpret_cast<char const *>(input_ptr),
                                static_cast<uint64_t>(batch_size),
                                static_cast<uint64_t>(REDUCTION_SIZE),
                                BLOCK_N,
                                BLOCK_K);

  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = BLOCK_N * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16;
  constexpr int SFB_size = 128 * BLOCK_K / 16;
  constexpr int stage_size = A_size + B_size + SFA_size + SFB_size;
  constexpr int smem_bytes = stage_size * NUM_STAGES;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid_dim(output_size / BLOCK_M, batch_size / BLOCK_N, 1);
  dim3 block_dim(BLOCK_M + 2 * 32, 1, 1);

  auto kernel_ptr = kernel::linear_nvfp4_1d2d_sm100_kernel<REDUCTION_SIZE,
                                                           BLOCK_M,
                                                           BLOCK_N,
                                                           BLOCK_K,
                                                           NUM_STAGES,
                                                           EPI_BATCH_LA>;
  if constexpr (smem_bytes > 48 * 1024) {
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  }

  kernel::linear_nvfp4_1d2d_sm100_kernel<REDUCTION_SIZE,
                                         BLOCK_M,
                                         BLOCK_N,
                                         BLOCK_K,
                                         NUM_STAGES,
                                         EPI_BATCH_LA>
      <<<grid_dim, block_dim, smem_bytes, stream>>>(
          A_tmap,
          B_tmap,
          reinterpret_cast<char const *>(weight_sf_ptr),
          reinterpret_cast<char const *>(input_sf_ptr),
          static_cast<type::bfloat16_t *>(output_ptr),
          static_cast<type::bfloat16_t const *>(residual_ptr),
          output_size,
          batch_size);
  CUTE_CHECK_ERROR(cudaGetLastError());
}

template <typename T>
void launch_linear_nvfp4_1d2d_sm100(void *input_ptr,
                                    void *input_sf_ptr,
                                    void *weight_ptr,
                                    void *weight_sf_ptr,
                                    void *output_ptr,
                                    void *residual_ptr,
                                    int batch_size,
                                    int output_size,
                                    int reduction_size) {
  // Unified default policy: BLOCK_N=128, NUM_STAGES=6, EPI_BATCH_LA=2 across
  // all K. Stage smem = 36KB at BN=128 BLOCK_K=256; 6*36 = 216KB fits the
  // ~228KB Blackwell limit. M (batch_size) must be divisible by 128.
  //
  // A BLOCK_N=64 variant (NUM_STAGES=8, fits 8*28=224KB) is available via
  // use_2sm_config=2 — see launch_linear_nvfp4_1d2d_sm100_bn64 below. It
  // accepts M divisible by 64 (relaxing the 128 constraint) and lets callers
  // use shapes with M ∈ {192, 320, 448, ...}. Empirically BN=64 does NOT
  // beat the BN=128 default on perf for M%128==0 shapes; it's an opt-in for
  // shape-coverage reasons, not a perf knob.
  switch (reduction_size) {
    case 256:
      return launch_linear_nvfp4_1d2d_sm100_config<T, 256, 128, 128, 256, 6, 2>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 512:
      return launch_linear_nvfp4_1d2d_sm100_config<T, 512, 128, 128, 256, 6, 2>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 768:
      return launch_linear_nvfp4_1d2d_sm100_config<T, 768, 128, 128, 256, 6, 2>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 1024:
      return launch_linear_nvfp4_1d2d_sm100_config<T,
                                                   1024,
                                                   128,
                                                   128,
                                                   256,
                                                   6,
                                                   2>(input_ptr,
                                                      input_sf_ptr,
                                                      weight_ptr,
                                                      weight_sf_ptr,
                                                      output_ptr,
                                                      residual_ptr,
                                                      batch_size,
                                                      output_size);
    case 1536:
      return launch_linear_nvfp4_1d2d_sm100_config<T,
                                                   1536,
                                                   128,
                                                   128,
                                                   256,
                                                   6,
                                                   2>(input_ptr,
                                                      input_sf_ptr,
                                                      weight_ptr,
                                                      weight_sf_ptr,
                                                      output_ptr,
                                                      residual_ptr,
                                                      batch_size,
                                                      output_size);
    case 2048:
      return launch_linear_nvfp4_1d2d_sm100_config<T,
                                                   2048,
                                                   128,
                                                   128,
                                                   256,
                                                   6,
                                                   2>(input_ptr,
                                                      input_sf_ptr,
                                                      weight_ptr,
                                                      weight_sf_ptr,
                                                      output_ptr,
                                                      residual_ptr,
                                                      batch_size,
                                                      output_size);
    case 2304:
      return launch_linear_nvfp4_1d2d_sm100_config<T,
                                                   2304,
                                                   128,
                                                   128,
                                                   256,
                                                   6,
                                                   2>(input_ptr,
                                                      input_sf_ptr,
                                                      weight_ptr,
                                                      weight_sf_ptr,
                                                      output_ptr,
                                                      residual_ptr,
                                                      batch_size,
                                                      output_size);
    case 3072:
      return launch_linear_nvfp4_1d2d_sm100_config<T,
                                                   3072,
                                                   128,
                                                   128,
                                                   256,
                                                   6,
                                                   2>(input_ptr,
                                                      input_sf_ptr,
                                                      weight_ptr,
                                                      weight_sf_ptr,
                                                      output_ptr,
                                                      residual_ptr,
                                                      batch_size,
                                                      output_size);
    case 4096:
      return launch_linear_nvfp4_1d2d_sm100_config<T,
                                                   4096,
                                                   128,
                                                   128,
                                                   256,
                                                   6,
                                                   2>(input_ptr,
                                                      input_sf_ptr,
                                                      weight_ptr,
                                                      weight_sf_ptr,
                                                      output_ptr,
                                                      residual_ptr,
                                                      batch_size,
                                                      output_size);
    case 6144:
      return launch_linear_nvfp4_1d2d_sm100_config<T,
                                                   6144,
                                                   128,
                                                   128,
                                                   256,
                                                   6,
                                                   2>(input_ptr,
                                                      input_sf_ptr,
                                                      weight_ptr,
                                                      weight_sf_ptr,
                                                      output_ptr,
                                                      residual_ptr,
                                                      batch_size,
                                                      output_size);
    case 7168:
      return launch_linear_nvfp4_1d2d_sm100_config<T,
                                                   7168,
                                                   128,
                                                   128,
                                                   256,
                                                   6,
                                                   2>(input_ptr,
                                                      input_sf_ptr,
                                                      weight_ptr,
                                                      weight_sf_ptr,
                                                      output_ptr,
                                                      residual_ptr,
                                                      batch_size,
                                                      output_size);
    case 8192:
      return launch_linear_nvfp4_1d2d_sm100_config<T,
                                                   8192,
                                                   128,
                                                   128,
                                                   256,
                                                   6,
                                                   2>(input_ptr,
                                                      input_sf_ptr,
                                                      weight_ptr,
                                                      weight_sf_ptr,
                                                      output_ptr,
                                                      residual_ptr,
                                                      batch_size,
                                                      output_size);
    case 16384:
      return launch_linear_nvfp4_1d2d_sm100_config<T,
                                                   16384,
                                                   128,
                                                   128,
                                                   256,
                                                   6,
                                                   2>(input_ptr,
                                                      input_sf_ptr,
                                                      weight_ptr,
                                                      weight_sf_ptr,
                                                      output_ptr,
                                                      residual_ptr,
                                                      batch_size,
                                                      output_size);
    default:
      TORCH_CHECK(false,
                  "SM100 NVFP4 1d2d unsupported K=",
                  reduction_size,
                  ". Supported K values: {256, 512, 768, 1024, 1536, 2048, "
                  "2304, 3072, 4096, 6144, 7168, 8192, 16384}.");
  }
}

// -----------------------------------------------------------------------------
// 1SM 1d2d BLOCK_N=64 variant (use_2sm_config=2). Same kernel as cfg=0 but
// with a narrower N tile + deeper pipeline. Accepts M divisible by 64 only.
// Stage smem = 28KB at BN=64 BLOCK_K=256; NUM_STAGES=8 fits in 224KB.
// EPI_BATCH_LA=1 because BN/EPI_WIDTH = 64/64 = 1 subtile.
// -----------------------------------------------------------------------------

template <typename T>
void launch_linear_nvfp4_1d2d_sm100_bn64(void *input_ptr,
                                         void *input_sf_ptr,
                                         void *weight_ptr,
                                         void *weight_sf_ptr,
                                         void *output_ptr,
                                         void *residual_ptr,
                                         int batch_size,
                                         int output_size,
                                         int reduction_size) {
  switch (reduction_size) {
    case 256:
      return launch_linear_nvfp4_1d2d_sm100_config<T, 256, 128, 64, 256, 8, 1>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 512:
      return launch_linear_nvfp4_1d2d_sm100_config<T, 512, 128, 64, 256, 8, 1>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 768:
      return launch_linear_nvfp4_1d2d_sm100_config<T, 768, 128, 64, 256, 8, 1>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 1024:
      return launch_linear_nvfp4_1d2d_sm100_config<T, 1024, 128, 64, 256, 8, 1>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 1536:
      return launch_linear_nvfp4_1d2d_sm100_config<T, 1536, 128, 64, 256, 8, 1>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 2048:
      return launch_linear_nvfp4_1d2d_sm100_config<T, 2048, 128, 64, 256, 8, 1>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 2304:
      return launch_linear_nvfp4_1d2d_sm100_config<T, 2304, 128, 64, 256, 8, 1>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 4096:
      return launch_linear_nvfp4_1d2d_sm100_config<T, 4096, 128, 64, 256, 8, 1>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 7168:
      return launch_linear_nvfp4_1d2d_sm100_config<T, 7168, 128, 64, 256, 8, 1>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 8192:
      return launch_linear_nvfp4_1d2d_sm100_config<T, 8192, 128, 64, 256, 8, 1>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 16384:
      return launch_linear_nvfp4_1d2d_sm100_config<T,
                                                   16384,
                                                   128,
                                                   64,
                                                   256,
                                                   8,
                                                   1>(input_ptr,
                                                      input_sf_ptr,
                                                      weight_ptr,
                                                      weight_sf_ptr,
                                                      output_ptr,
                                                      residual_ptr,
                                                      batch_size,
                                                      output_size);
    default:
      TORCH_CHECK(false,
                  "SM100 NVFP4 1d2d BN=64 unsupported K=",
                  reduction_size,
                  ". Supported K values: {256, 512, 768, 1024, 1536, 2048, "
                  "2304, 4096, 7168, 8192, 16384}.");
  }
}

template <typename T,
          int REDUCTION_SIZE,
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
          bool HAS_BIAS_DEPRECATED> // ignored; HAS_BIAS chosen at launch from
                                    // residual_ptr
void launch_linear_nvfp4_1d2d_2sm_sm100_config(void *input_ptr,
                                               void *input_sf_ptr,
                                               void *weight_ptr,
                                               void *weight_sf_ptr,
                                               void *output_ptr,
                                               void *residual_ptr,
                                               int batch_size,
                                               int output_size) {
  static_assert(BLOCK_M == 128, "2SM 1d2d NVFP4 uses a 128-row per-CTA A tile");
  static_assert(BLOCK_K == 256,
                "2SM 1d2d NVFP4 dispatch table uses BLOCK_K=256");
  static_assert(SUPERGROUP_SIZE > 0, "SUPERGROUP_SIZE must be positive");
  (void)sizeof(T);

  TORCH_CHECK(batch_size % BLOCK_N == 0,
              "SM100 NVFP4 2SM 1d2d requires batch_size divisible by BLOCK_N=",
              BLOCK_N,
              ", got batch_size=",
              batch_size);
  TORCH_CHECK(
      output_size % (2 * BLOCK_M) == 0,
      "SM100 NVFP4 2SM 1d2d requires output_size divisible by 2*BLOCK_M=",
      2 * BLOCK_M,
      ", got output_size=",
      output_size);
  TORCH_CHECK(REDUCTION_SIZE % BLOCK_K == 0,
              "SM100 NVFP4 2SM 1d2d requires K divisible by BLOCK_K");

  CUtensorMap A_tmap{};
  CUtensorMap B_tmap{};
  CUtensorMap C_tmap{};
  CUtensorMap SFA_tmap{};
  CUtensorMap SFB_tmap{};
  // Always SWAP_AB: A is weight [output, K], B is input [batch, K].
  kernel::tma::init_AB_tmap_fp4(&A_tmap,
                                reinterpret_cast<char const *>(weight_ptr),
                                static_cast<uint64_t>(output_size),
                                static_cast<uint64_t>(REDUCTION_SIZE),
                                BLOCK_M,
                                BLOCK_K);
  kernel::tma::init_AB_tmap_fp4(&B_tmap,
                                reinterpret_cast<char const *>(input_ptr),
                                static_cast<uint64_t>(batch_size),
                                static_cast<uint64_t>(REDUCTION_SIZE),
                                BLOCK_N / 2,
                                BLOCK_K);
  // Scale tensor maps: cluster-scope cta_group::2 TMA replaces the per-CTA
  // bulk TMA + scale_ready_flags spin. Both CTAs issue the same logical TMA
  // and report to CTA0's scale mbarrier. Each TMA fetches BLOCK_K/64 k_blocks
  // (= 128 * BLOCK_K / 16 bytes) for one 128-row block.
  kernel::tma::init_SF_tmap_fp4(&SFA_tmap,
                                reinterpret_cast<char const *>(weight_sf_ptr),
                                static_cast<uint64_t>(output_size),
                                static_cast<uint64_t>(REDUCTION_SIZE),
                                BLOCK_K / 64);
  kernel::tma::init_SF_tmap_fp4(&SFB_tmap,
                                reinterpret_cast<char const *>(input_sf_ptr),
                                static_cast<uint64_t>(batch_size),
                                static_cast<uint64_t>(REDUCTION_SIZE),
                                BLOCK_K / 64);
  static_assert(EPI_TILE_N == 32 || EPI_TILE_N == 64 || EPI_TILE_N == 128,
                "EPI_TILE_N must be 32, 64, or 128");
  static_assert(EPI_NUM_D_TILES > 0, "EPI_NUM_D_TILES must be positive");
  static_assert(BLOCK_N % EPI_TILE_N == 0,
                "BLOCK_N must be divisible by the epilogue tile width");
  static_assert(EPI_BATCH_LA >= 1 && EPI_BATCH_LA <= (BLOCK_N / EPI_TILE_N),
                "EPI_BATCH_LA must be in [1, EPI_PIPE_DEPTH]");
  static_assert((BLOCK_N / EPI_TILE_N) % EPI_BATCH_LA == 0,
                "EPI_PIPE_DEPTH must be divisible by EPI_BATCH_LA");
  kernel::tma::init_C_tmap_fp4(&C_tmap,
                               output_ptr,
                               static_cast<uint64_t>(batch_size),
                               static_cast<uint64_t>(output_size),
                               EPI_TILE_N,
                               BLOCK_M);

  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = (BLOCK_N / 2) * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16;
  constexpr int SFB_scale_tiles = (BLOCK_N + 127) / 128;
  constexpr int SFB_size = SFB_scale_tiles * 128 * BLOCK_K / 16;
  constexpr int stage_size = A_size + B_size + SFA_size + SFB_size;
  constexpr int output_smem_bytes =
      EPI_NUM_D_TILES * EPI_TILE_N * BLOCK_M * sizeof(type::bfloat16_t);
  constexpr int smem_bytes = stage_size * NUM_STAGES + output_smem_bytes;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int persistent_ctas = 148;
  dim3 grid_dim(persistent_ctas, 1, 1);
  dim3 block_dim(BLOCK_M + 4 * 32, 1, 1);
  dim3 cluster_dim(2, 1, 1);

  // HAS_BIAS chosen at launch time based on residual_ptr. Two kernel
  // instantiations per config (with/without bias) so the inner-loop has no
  // runtime branch on bias.
  (void)HAS_BIAS_DEPRECATED;
  bool const has_bias = (residual_ptr != nullptr);
  auto kernel_ptr_nobias =
      kernel::linear_nvfp4_1d2d_2sm_sm100_kernel<REDUCTION_SIZE,
                                                 BLOCK_M,
                                                 BLOCK_N,
                                                 BLOCK_K,
                                                 NUM_STAGES,
                                                 SUPERGROUP_SIZE,
                                                 EPI_TILE_N,
                                                 EPI_NUM_D_TILES,
                                                 EPI_BATCHED,
                                                 EPI_BATCH_LA,
                                                 OVERLAP_OUTPUT_MBAR,
                                                 /*HAS_BIAS=*/false>;
  auto kernel_ptr_bias =
      kernel::linear_nvfp4_1d2d_2sm_sm100_kernel<REDUCTION_SIZE,
                                                 BLOCK_M,
                                                 BLOCK_N,
                                                 BLOCK_K,
                                                 NUM_STAGES,
                                                 SUPERGROUP_SIZE,
                                                 EPI_TILE_N,
                                                 EPI_NUM_D_TILES,
                                                 EPI_BATCHED,
                                                 EPI_BATCH_LA,
                                                 OVERLAP_OUTPUT_MBAR,
                                                 /*HAS_BIAS=*/true>;
  if constexpr (smem_bytes > 48 * 1024) {
    CUTE_CHECK_ERROR(
        cudaFuncSetAttribute(kernel_ptr_nobias,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes));
    CUTE_CHECK_ERROR(
        cudaFuncSetAttribute(kernel_ptr_bias,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes));
  }

  cutlass::ClusterLaunchParams params = {
      grid_dim, block_dim, cluster_dim, smem_bytes, stream};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params,
      (void const *)(has_bias ? kernel_ptr_bias : kernel_ptr_nobias),
      A_tmap,
      B_tmap,
      C_tmap,
      SFA_tmap,
      SFB_tmap,
      static_cast<type::bfloat16_t const *>(residual_ptr),
      output_size,
      batch_size);
  CUTE_CHECK_ERROR(cudaGetLastError());
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "SM100 NVFP4 2SM 1d2d cluster launch failed: ",
              cutlassGetStatusString(status));
}

template <typename T>
void launch_linear_nvfp4_1d2d_2sm_sm100(void *input_ptr,
                                        void *input_sf_ptr,
                                        void *weight_ptr,
                                        void *weight_sf_ptr,
                                        void *output_ptr,
                                        void *residual_ptr,
                                        int batch_size,
                                        int output_size,
                                        int reduction_size,
                                        int config_id) {
  // Unified policy: BLOCK_M=128 (per-CTA; cta_group::2 makes MMA_M=256),
  // BLOCK_N=256 across all K. NUM_STAGES = 5 where smem permits (K large
  // enough to benefit from deeper pipeline), 4 otherwise. The small-M
  // BLOCK_N=128 branches were removed — callers must now satisfy
  // batch_size % 256 == 0 (validated at launch).
  //
  // K set matches the 1SM dispatcher for consistency: {256, 512, 768, 1024,
  // 1536, 2048, 2304, 4096, 7168, 8192, 16384}.

  // cfg=0 picks NUM_STAGES per K (data from stages-sweep study: S=5 wins
  // for K∈{4096, 8192, 16384}; smaller K is mainloop-light, S=4 is plenty).
  // cfgs 1..9 are tuning probes kept on K=4096 for the existing benches.
  switch (reduction_size) {
    case 256:
      return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                       256,
                                                       128,
                                                       256,
                                                       256,
                                                       4,
                                                       4,
                                                       64,
                                                       2,
                                                       false,
                                                       1,
                                                       true,
                                                       false>(input_ptr,
                                                              input_sf_ptr,
                                                              weight_ptr,
                                                              weight_sf_ptr,
                                                              output_ptr,
                                                              residual_ptr,
                                                              batch_size,
                                                              output_size);
    case 512:
      return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                       512,
                                                       128,
                                                       256,
                                                       256,
                                                       4,
                                                       4,
                                                       64,
                                                       2,
                                                       false,
                                                       1,
                                                       true,
                                                       false>(input_ptr,
                                                              input_sf_ptr,
                                                              weight_ptr,
                                                              weight_sf_ptr,
                                                              output_ptr,
                                                              residual_ptr,
                                                              batch_size,
                                                              output_size);
    case 768:
      return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                       768,
                                                       128,
                                                       256,
                                                       256,
                                                       4,
                                                       4,
                                                       64,
                                                       2,
                                                       false,
                                                       1,
                                                       true,
                                                       false>(input_ptr,
                                                              input_sf_ptr,
                                                              weight_ptr,
                                                              weight_sf_ptr,
                                                              output_ptr,
                                                              residual_ptr,
                                                              batch_size,
                                                              output_size);
    case 1024:
      switch (config_id) {
        case 0:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           1024,
                                                           128,
                                                           256,
                                                           256,
                                                           4,
                                                           4,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        // SG ablation: cfg=100+SG. Producer-arm OFF (matches cfg0).
        case 101:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           1024,
                                                           128,
                                                           256,
                                                           256,
                                                           4,
                                                           1,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 102:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           1024,
                                                           128,
                                                           256,
                                                           256,
                                                           4,
                                                           2,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 104:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           1024,
                                                           128,
                                                           256,
                                                           256,
                                                           4,
                                                           4,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        default:
          TORCH_CHECK(false,
                      "SM100 NVFP4 2SM 1d2d K=1024 unsupported config_id=",
                      config_id,
                      ".");
      }
      break;
    case 1536:
      return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                       1536,
                                                       128,
                                                       256,
                                                       256,
                                                       4,
                                                       4,
                                                       64,
                                                       2,
                                                       false,
                                                       1,
                                                       true,
                                                       false>(input_ptr,
                                                              input_sf_ptr,
                                                              weight_ptr,
                                                              weight_sf_ptr,
                                                              output_ptr,
                                                              residual_ptr,
                                                              batch_size,
                                                              output_size);
    case 2048:
      switch (config_id) {
        case 0:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           2048,
                                                           128,
                                                           256,
                                                           256,
                                                           4,
                                                           4,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        // SG ablation: cfg=100+SG. Producer-arm OFF (matches cfg0).
        case 101:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           2048,
                                                           128,
                                                           256,
                                                           256,
                                                           4,
                                                           1,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 102:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           2048,
                                                           128,
                                                           256,
                                                           256,
                                                           4,
                                                           2,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 104:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           2048,
                                                           128,
                                                           256,
                                                           256,
                                                           4,
                                                           4,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 108:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           2048,
                                                           128,
                                                           256,
                                                           256,
                                                           4,
                                                           8,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        default:
          TORCH_CHECK(false,
                      "SM100 NVFP4 2SM 1d2d K=2048 unsupported config_id=",
                      config_id,
                      ".");
      }
      break;
    case 2304:
      return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                       2304,
                                                       128,
                                                       256,
                                                       256,
                                                       4,
                                                       4,
                                                       64,
                                                       2,
                                                       false,
                                                       1,
                                                       true,
                                                       false>(input_ptr,
                                                              input_sf_ptr,
                                                              weight_ptr,
                                                              weight_sf_ptr,
                                                              output_ptr,
                                                              residual_ptr,
                                                              batch_size,
                                                              output_size);
    case 3072:
      return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                       3072,
                                                       128,
                                                       256,
                                                       256,
                                                       4,
                                                       4,
                                                       64,
                                                       2,
                                                       false,
                                                       1,
                                                       true,
                                                       false>(input_ptr,
                                                              input_sf_ptr,
                                                              weight_ptr,
                                                              weight_sf_ptr,
                                                              output_ptr,
                                                              residual_ptr,
                                                              batch_size,
                                                              output_size);
    case 4096:
      switch (config_id) {
        case 0:
          // S=5 was the empirical winner in the stages-sweep study.
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           4,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 1:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           4,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 2:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           4,
                                                           4,
                                                           128,
                                                           2,
                                                           false,
                                                           1,
                                                           false,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 3:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           4,
                                                           4,
                                                           64,
                                                           3,
                                                           false,
                                                           1,
                                                           false,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 4:
          // cfg=4 used to be BN=128; aliased to BN=256 S=5 (= cfg=1) under
          // the unified BLOCK_N=256 policy.
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           4,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 5:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           4,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 6:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           4,
                                                           64,
                                                           2,
                                                           true,
                                                           2,
                                                           false,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 7:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           4,
                                                           4,
                                                           128,
                                                           2,
                                                           true,
                                                           2,
                                                           false,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 8:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           4,
                                                           64,
                                                           2,
                                                           true,
                                                           4,
                                                           false,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 9:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           2,
                                                           4,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 10:
          // Experiment: EPI_TILE_N=32, EPI_PIPE_DEPTH=8 (deeper epilogue pipe).
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           4,
                                                           32,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 11:
          // Experiment: OVERLAP_OUTPUT_MBAR=false (TK pattern at large N).
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           4,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           false,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        // SG ablation: cfg=100+SG. Producer-arm OFF (matches existing K=4096
        // cfg0).
        case 101:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           1,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 102:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           2,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 104:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           4,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 108:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           8,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 116:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           16,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 124:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           24,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 132:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           32,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 148:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           48,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 164:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           4096,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           64,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        default:
          TORCH_CHECK(false,
                      "SM100 NVFP4 2SM 1d2d K=4096 unsupported config_id=",
                      config_id,
                      ".");
      }
      break;
    case 6144:
      return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                       6144,
                                                       128,
                                                       256,
                                                       256,
                                                       4,
                                                       4,
                                                       64,
                                                       2,
                                                       false,
                                                       1,
                                                       true,
                                                       false>(input_ptr,
                                                              input_sf_ptr,
                                                              weight_ptr,
                                                              weight_sf_ptr,
                                                              output_ptr,
                                                              residual_ptr,
                                                              batch_size,
                                                              output_size);
    case 7168:
      return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                       7168,
                                                       128,
                                                       256,
                                                       256,
                                                       4,
                                                       4,
                                                       64,
                                                       2,
                                                       false,
                                                       1,
                                                       true,
                                                       false>(input_ptr,
                                                              input_sf_ptr,
                                                              weight_ptr,
                                                              weight_sf_ptr,
                                                              output_ptr,
                                                              residual_ptr,
                                                              batch_size,
                                                              output_size);
    case 8192:
      switch (config_id) {
        case 0:
          // cfg0: SG=8 (divides 32-row tile grid evenly, stable); producer-arm.
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           8192,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           8,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 10:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           8192,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           4,
                                                           32,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 11:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           8192,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           4,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           false,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        // SG ablation: cfg=100+SG.
        case 101:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           8192,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           1,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 102:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           8192,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           2,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 104:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           8192,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           4,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 108:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           8192,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           8,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 116:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           8192,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           16,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 124:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           8192,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           24,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 132:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           8192,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           32,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        default:
          TORCH_CHECK(false,
                      "SM100 NVFP4 2SM 1d2d K=8192 unsupported config_id=",
                      config_id,
                      ".");
      }
      break;
    case 16384:
      switch (config_id) {
        case 0:
          // cfg0: SG=8 (divides 64-row tile grid, stable); producer-arm.
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           16384,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           8,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 3:
          // NUM_STAGES=3 sweep point.
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           16384,
                                                           128,
                                                           256,
                                                           256,
                                                           3,
                                                           4,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 9:
          // NUM_STAGES=2 sweep point.
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           16384,
                                                           128,
                                                           256,
                                                           256,
                                                           2,
                                                           4,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 10:
          // EPI_TILE_N=32 experiment.
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           16384,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           4,
                                                           32,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 11:
          // OVERLAP_OUTPUT_MBAR=false experiment.
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           16384,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           4,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           false,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        // SG ablation: cfg=100+SG.
        case 101:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           16384,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           1,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 102:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           16384,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           2,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 104:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           16384,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           4,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 108:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           16384,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           8,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 114:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           16384,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           14,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 116:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           16384,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           16,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 124:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           16384,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           24,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 132:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           16384,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           32,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 148:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           16384,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           48,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        case 164:
          return launch_linear_nvfp4_1d2d_2sm_sm100_config<T,
                                                           16384,
                                                           128,
                                                           256,
                                                           256,
                                                           5,
                                                           64,
                                                           64,
                                                           2,
                                                           false,
                                                           1,
                                                           true,
                                                           false>(input_ptr,
                                                                  input_sf_ptr,
                                                                  weight_ptr,
                                                                  weight_sf_ptr,
                                                                  output_ptr,
                                                                  residual_ptr,
                                                                  batch_size,
                                                                  output_size);
        default:
          TORCH_CHECK(false,
                      "SM100 NVFP4 2SM 1d2d K=16384 supports config_ids {0, 3, "
                      "9, 10, 11, 101..116}; got ",
                      config_id,
                      ".");
      }
      break;
    default:
      TORCH_CHECK(false,
                  "SM100 NVFP4 2SM 1d2d unsupported K=",
                  reduction_size,
                  ". Supported K values: {256, 512, 768, 1024, 1536, 2048, "
                  "2304, 4096, 7168, 8192, 16384}.");
  }
}

// Compile-time supported sets, dispatched as three nested fold expressions
// (mma_n → output_size → reduction_size) to keep each tuple size below the
// libstdc++ tuple-recursion limit. The swapab_mma_n occupancy formula only ever
// selects power-of-2 widths {8,16,32,64,128} (1SM caps MMA_N at 128), so only
// those are compiled. N and K cover the production shape set.
using nvfp4_swapAB_mma_ns = std::integer_sequence<int, 8, 16, 32, 64, 128>;
using nvfp4_swapAB_outs = std::integer_sequence<int,
                                                128,
                                                256,
                                                384,
                                                512,
                                                768,
                                                1024,
                                                1536,
                                                2048,
                                                4096,
                                                7168>;
using nvfp4_swapAB_ks = std::integer_sequence<int,
                                              256,
                                              512,
                                              768,
                                              1024,
                                              1536,
                                              2048,
                                              3072,
                                              4096,
                                              6144,
                                              7168>;

template <int... Values, class Func>
bool dispatch_int_sequence(std::integer_sequence<int, Values...>, Func &&fn) {
  return (fn(std::integral_constant<int, Values>{}) || ...);
}

// Canonical swapAB tile-width selection for the small-batch path (M < 128).
//
// The swapAB grid is (N/128 output tiles) x (ceil(M/MMA_N) batch tiles). Both
// dimensions occupy CTAs; choosing MMA_N only scales the batch-tile count.
// Each CTA pays a fixed setup cost (TMA descriptor fetch, TMEM alloc, pipeline
// fill) that is independent of K, so the cheapest schedule keeps every CTA in a
// single SM wave: total CTAs <= SM count. A larger N consumes more SMs via the
// output dimension, shrinking the per-N batch-tile budget and forcing a wider
// MMA_N (fewer batch tiles) at smaller M.
//
//   B(N)  = floor(SM_count / (N/128))     # one-wave batch-tile budget
//   MMA_N = smallest supported tile >= ceil(M / B(N))   in {8,16,32,64,128}
//
// This reproduces the empirically measured N=K crossovers on B200 (148 SMs):
//   N=2048 (B=9):  ->16 at M>72
//   N=4096 (B=4):  ->16 at M>32,  ->32 at M>64
//   N=7168 (B=2):  ->16 at M>16,  ->32 at M>32,  ->64 at M>64
// Using the runtime SM count keeps it correct across GPUs.
//
// The activation SF must be quantized with this same mma_n (the wrapper does
// this internally), or the per-tile SF layout mismatches.
inline int swapab_mma_n(int batch_size, int output_size) {
  static int sm_count = 0;
  if (sm_count == 0) {
    CUTE_CHECK_ERROR(
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  }
  int const output_tiles = (output_size + 127) / 128; // N/128
  int const budget = sm_count / output_tiles;         // B(N), floor
  // ceil(M / max(B,1)); B==0 (N alone exceeds the SM count) forces the widest
  // tile so a single batch tile fits the residual SM budget.
  int const needed = (budget >= 1) ? (batch_size + budget - 1) / budget : 128;
  // Round up to the smallest supported swapAB tile {8,16,32,64,128}, min 8.
  if (needed <= 8) {
    return 8;
  }
  if (needed <= 16) {
    return 16;
  }
  if (needed <= 32) {
    return 32;
  }
  if (needed <= 64) {
    return 64;
  }
  return 128;
}

// Raw-CUDA swapAB launcher (no CUTLASS). Builds A (weight) and B (activation)
// 3D fp4 TMA descriptors via init_AB_tmap_fp4 and an output TMA descriptor via
// init_C_tmap_fp4; passes raw SF byte pointers (loaded with cp.async.bulk in
// the kernel). Descriptors are rebuilt per call (graph-capture friendly: no
// device-side mutation, all on the kernel's stream).
template <typename T, int MMA_N, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_linear_nvfp4_swapAB_sm100(void *input_ptr,
                                      void *input_sf_ptr,
                                      void *weight_ptr,
                                      void *weight_sf_ptr,
                                      void *output_ptr,
                                      void *residual_ptr,
                                      int logical_batch_size) {
  constexpr int BLOCK_M = 128;
  constexpr int BLOCK_K = 256;
  static_assert(REDUCTION_SIZE % BLOCK_K == 0, "K must be divisible by 256");
  static_assert(OUTPUT_SIZE % BLOCK_M == 0, "N must be divisible by 128");
  // Wide MMA_N grows the per-stage B/out SMEM, so pick the deepest pipeline
  // (<=4, to keep the TMEM accumulator alloc at 256 cols) that fits the ~224KB
  // SMEM budget. Narrow MMA_N keeps NUM_STAGES=4; very wide drops to 3/2.
  constexpr int STAGE_B = MMA_N * BLOCK_K / 2;
  constexpr int STAGE_BYTES =
      BLOCK_M * BLOCK_K / 2 + STAGE_B + 2 * (128 * BLOCK_K / 16);
  constexpr int ACC_C = (MMA_N <= 32)    ? 32
                        : (MMA_N <= 64)  ? 64
                        : (MMA_N <= 128) ? 128
                                         : 256;
  constexpr int SMEM_LIMIT = 224 * 1024;
  // Compile-time stage chooser: largest s in {4,3,2} with stage*s + out <=
  // limit.
  constexpr int NS4_NAB =
      (2 * ACC_C + 2 * (4 * (BLOCK_K / 64)) * 4 <= 512) ? 2 : 1;
  constexpr int NS3_NAB =
      (2 * ACC_C + 2 * (4 * (BLOCK_K / 64)) * 3 <= 512) ? 2 : 1;
  constexpr int NS2_NAB =
      (2 * ACC_C + 2 * (4 * (BLOCK_K / 64)) * 2 <= 512) ? 2 : 1;
  constexpr int NUM_STAGES =
      (STAGE_BYTES * 4 + NS4_NAB * MMA_N * BLOCK_M * 2 <= SMEM_LIMIT)   ? 4
      : (STAGE_BYTES * 3 + NS3_NAB * MMA_N * BLOCK_M * 2 <= SMEM_LIMIT) ? 3
                                                                        : 2;

  CUtensorMap A_tmap{}; // weight [OUTPUT_SIZE, K], 128-wide tile
  CUtensorMap B_tmap{}; // activation [M, K], MMA_N-wide tile
  kernel::tma::init_AB_tmap_fp4(&A_tmap,
                                reinterpret_cast<char const *>(weight_ptr),
                                static_cast<uint64_t>(OUTPUT_SIZE),
                                static_cast<uint64_t>(REDUCTION_SIZE),
                                BLOCK_M,
                                BLOCK_K);
  kernel::tma::init_AB_tmap_fp4(&B_tmap,
                                reinterpret_cast<char const *>(input_ptr),
                                static_cast<uint64_t>(logical_batch_size),
                                static_cast<uint64_t>(REDUCTION_SIZE),
                                MMA_N,
                                BLOCK_K);

  // Output C [M, N] bf16, stored in [output, batch] tile orientation:
  // boxDim {output=BLOCK_M, batch=MMA_N}.
  CUtensorMap C_tmap{};
  kernel::tma::init_C_tmap_fp4(
      &C_tmap,
      output_ptr,
      /*batch_size=*/static_cast<uint64_t>(logical_batch_size),
      /*output_size=*/static_cast<uint64_t>(OUTPUT_SIZE),
      /*tile_rows=batch*/ MMA_N,
      /*tile_cols=output*/ BLOCK_M);

  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = MMA_N * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16;
  constexpr int SFB_size = 128 * BLOCK_K / 16;
  constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;
  // Mirror the kernel's accumulator buffering decision (must match exactly):
  // double-buffer the out_smem staging iff the accumulator double-buffers.
  constexpr int ACC_COLS = (MMA_N <= 32)    ? 32
                           : (MMA_N <= 64)  ? 64
                           : (MMA_N <= 128) ? 128
                                            : 256;
  constexpr int SF_TOTAL = 2 * (4 * (BLOCK_K / 64)) * NUM_STAGES;
  constexpr int NUM_ACC_BUF = (2 * ACC_COLS + SF_TOTAL <= 512) ? 2 : 1;
  constexpr int OUT_SMEM =
      NUM_ACC_BUF * MMA_N * BLOCK_M * (int)sizeof(type::bfloat16_t);
  constexpr int smem_bytes = STAGE_SIZE * NUM_STAGES + OUT_SMEM;

  // Persistent grid: one CTA per SM, grid-stride over the tile space. Cap to
  // the total tile count so we never launch idle CTAs.
  int const num_n_tiles = (logical_batch_size + MMA_N - 1) / MMA_N;
  int const num_tiles = (OUTPUT_SIZE / BLOCK_M) * num_n_tiles;
  static thread_local int sm_count = 0;
  if (sm_count == 0) {
    CUTE_CHECK_ERROR(
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  }
  int const grid_x = (num_tiles < sm_count) ? num_tiles : sm_count;
  dim3 grid_dim(grid_x, 1, 1);
  dim3 block_dim(BLOCK_M + 3 * 32, 1, 1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (num_tiles == 0) {
    return;
  }

  auto launch = [&](auto NoBiasTag) {
    constexpr bool NOBIAS = decltype(NoBiasTag)::value;
    auto kfn = kernel::linear_nvfp4_swapAB_sm100_kernel<MMA_N,
                                                        OUTPUT_SIZE,
                                                        REDUCTION_SIZE,
                                                        BLOCK_K,
                                                        NUM_STAGES,
                                                        NOBIAS>;
    if constexpr (smem_bytes > 48 * 1024) {
      static thread_local bool configured = false;
      if (!configured) {
        CUTE_CHECK_ERROR(cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
        configured = true;
      }
    }
    kfn<<<grid_dim, block_dim, smem_bytes, stream>>>(
        A_tmap,
        B_tmap,
        C_tmap,
        reinterpret_cast<char const *>(weight_sf_ptr),
        reinterpret_cast<char const *>(input_sf_ptr),
        static_cast<type::bfloat16_t const *>(residual_ptr),
        logical_batch_size,
        OUTPUT_SIZE);
    CUTE_CHECK_ERROR(cudaGetLastError());
  };
  if (residual_ptr != nullptr) {
    launch(std::false_type{});
  } else {
    launch(std::true_type{});
  }
}

template <typename T>
void dispatch_linear_nvfp4_swapAB(int output_size,
                                  int reduction_size,
                                  int batch_size,
                                  int mma_n,
                                  void *input_ptr,
                                  void *input_sf_ptr,
                                  void *weight_ptr,
                                  void *weight_sf_ptr,
                                  void *output_ptr,
                                  void *residual_ptr) {

  auto try_kk = [&](auto MmaNTag, auto OutTag, auto KTag) {
    constexpr int MMA_N = decltype(MmaNTag)::value;
    constexpr int OUT = decltype(OutTag)::value;
    constexpr int K = decltype(KTag)::value;
    if (MMA_N != mma_n || OUT != output_size || K != reduction_size) {
      return false;
    }
    launch_linear_nvfp4_swapAB_sm100<T, MMA_N, OUT, K>(input_ptr,
                                                       input_sf_ptr,
                                                       weight_ptr,
                                                       weight_sf_ptr,
                                                       output_ptr,
                                                       residual_ptr,
                                                       batch_size);
    return true;
  };
  auto try_oo = [&](auto MmaNTag, auto OutTag) {
    return dispatch_int_sequence(nvfp4_swapAB_ks{}, [&](auto KTag) {
      return try_kk(MmaNTag, OutTag, KTag);
    });
  };
  auto try_mm = [&](auto MmaNTag) {
    return dispatch_int_sequence(nvfp4_swapAB_outs{}, [&](auto OutTag) {
      return try_oo(MmaNTag, OutTag);
    });
  };
  bool const dispatched = dispatch_int_sequence(
      nvfp4_swapAB_mma_ns{}, [&](auto MmaNTag) { return try_mm(MmaNTag); });

  TORCH_CHECK(dispatched,
              "Small-M SM100 NVFP4 swapAB: unsupported shape (mma_n=",
              mma_n,
              ", N=",
              output_size,
              ", K=",
              reduction_size,
              ")");
}

// ============================================================
// Python entry points
// ============================================================

namespace {

constexpr int OUTPUT_SIZE = 128;
constexpr int REDUCTION_SIZE = 768;
constexpr int BATCH_SIZE = 4096;
constexpr int SCALE_VEC_SIZE = 16;
constexpr int QUANTIZE_THREADS = 128;

template <typename T, int HIDDEN_SIZE>
__global__ __launch_bounds__(
    QUANTIZE_THREADS,
    1) void quantize_nvfp4_sm100_wrapper(T const *input_ptr,
                                         uint8_t *output_q_ptr,
                                         uint8_t *output_s_ptr,
                                         int batch_size,
                                         int mma_n) {
  kernel::quantize_nvfp4_sm100_task_impl<HIDDEN_SIZE,
                                         SCALE_VEC_SIZE,
                                         HIDDEN_SIZE,
                                         T>(input_ptr,
                                            output_q_ptr,
                                            output_s_ptr,
                                            batch_size,
                                            1.0e-6f,
                                            /*min_4bit=*/-6.0f,
                                            /*max_4bit=*/6.0f,
                                            /*scale_outer_stride=*/32 * 4 * 4,
                                            mma_n);
}

// mma_n == 0  → interleaved layout [padded/128, K/64, 32, 4, 4]  (for 1d2d
// path) mma_n >  0  → per-tile swapAB layout [ceil(batch/mma_n), K/64, 32, 4,
// 4]
template <int HIDDEN_SIZE>
std::vector<torch::Tensor>
    launch_quantize_nvfp4_sm100(torch::Tensor const &input, int mma_n) {
  int const batch_size = static_cast<int>(input.size(0));
  int const padded_batch_size = ((batch_size + 127) / 128) * 128;
  int const sf_k_outer = HIDDEN_SIZE / 64;

  auto output_q = torch::empty({padded_batch_size, HIDDEN_SIZE / 2},
                               input.options().dtype(torch::kUInt8));

  at::Tensor output_s;
  if (mma_n > 0) {
    // layout for swapAB
    int const num_n_tiles = (batch_size + mma_n - 1) / mma_n;
    output_s = torch::empty({num_n_tiles, sf_k_outer, 32, 4, 4},
                            input.options().dtype(torch::kUInt8));
  } else {
    // layout for 1d2d path
    output_s = torch::empty({padded_batch_size / 128, sf_k_outer, 32, 4, 4},
                            input.options().dtype(torch::kUInt8));
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // quantize_nvfp4_sm100_task_impl now loops over the whole batch in one CTA
  // (shared with the MPK task), so launch a single block.
  quantize_nvfp4_sm100_wrapper<float, HIDDEN_SIZE>
      <<<dim3(1), dim3(QUANTIZE_THREADS), 0, stream>>>(
          static_cast<float const *>(input.data_ptr()),
          static_cast<uint8_t *>(output_q.data_ptr()),
          static_cast<uint8_t *>(output_s.data_ptr()),
          batch_size,
          mma_n);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "quantize_nvfp4_sm100 launch failed: ",
              cudaGetErrorString(err));
  return {output_q, output_s};
}

std::vector<torch::Tensor>
    dispatch_quantize_nvfp4_sm100(torch::Tensor const &input, int mma_n = 0) {
  int const hidden_size = static_cast<int>(input.size(1));
  TORCH_CHECK(hidden_size % 64 == 0, "input.shape[1] must be divisible by 64");

  switch (hidden_size) {
    case 128:
      return launch_quantize_nvfp4_sm100<128>(input, mma_n);
    case 256:
      return launch_quantize_nvfp4_sm100<256>(input, mma_n);
    case 384:
      return launch_quantize_nvfp4_sm100<384>(input, mma_n);
    case 512:
      return launch_quantize_nvfp4_sm100<512>(input, mma_n);
    case 768:
      return launch_quantize_nvfp4_sm100<768>(input, mma_n);
    case 1024:
      return launch_quantize_nvfp4_sm100<1024>(input, mma_n);
    case 1536:
      return launch_quantize_nvfp4_sm100<1536>(input, mma_n);
    case 2048:
      return launch_quantize_nvfp4_sm100<2048>(input, mma_n);
    case 3072:
      return launch_quantize_nvfp4_sm100<3072>(input, mma_n);
    case 4096:
      return launch_quantize_nvfp4_sm100<4096>(input, mma_n);
    case 6144:
      return launch_quantize_nvfp4_sm100<6144>(input, mma_n);
    case 7168:
      return launch_quantize_nvfp4_sm100<7168>(input, mma_n);
    case 8192:
      return launch_quantize_nvfp4_sm100<8192>(input, mma_n);
    default:
      TORCH_CHECK(
          false,
          "quantize_nvfp4_sm100 supports K in {128, 256, 384, 512, "
          "768, 1024, 1536, 2048, 3072, 4096, 6144, 7168, 8192}. Got K=",
          hidden_size);
  }
}

void launch_linear_nvfp4_small_batch(torch::Tensor const &input,
                                     torch::Tensor const &input_sf,
                                     torch::Tensor const &weight,
                                     torch::Tensor const &weight_sf,
                                     c10::optional<at::Tensor> const &residual,
                                     torch::Tensor const &output,
                                     int reduction_size,
                                     int batch_size,
                                     int mma_n = 8) {
  TORCH_CHECK(batch_size >= 1,
              "launch_linear_nvfp4_small_batch requires batch_size >= 1, got ",
              batch_size);
  int const output_size = static_cast<int>(weight.size(0));

  // input_sf is already in per-tile swapAB layout [num_n_tiles, sf_k_outer, 32,
  // 4, 4] produced directly by the quantizer — no restructuring needed.
  dispatch_linear_nvfp4_swapAB<cute::float_e2m1_t>(
      output_size,
      reduction_size,
      batch_size,
      mma_n,
      input.data_ptr(),
      input_sf.data_ptr(),
      weight.data_ptr(),
      weight_sf.data_ptr(),
      output.data_ptr(),
      residual.has_value() ? residual->data_ptr() : nullptr);
}

void launch_linear_nvfp4(torch::Tensor const &input,
                         torch::Tensor const &input_sf,
                         torch::Tensor const &weight,
                         torch::Tensor const &weight_sf,
                         c10::optional<at::Tensor> const &residual,
                         torch::Tensor const &output,
                         int batch_size,
                         int output_size,
                         int reduction_size,
                         bool use_2sm,
                         int use_2sm_config) {
  (void)OUTPUT_SIZE;
  (void)REDUCTION_SIZE;
  (void)BATCH_SIZE;
  if (use_2sm) {
    launch_linear_nvfp4_1d2d_2sm_sm100<cute::float_e2m1_t>(
        input.data_ptr(),
        input_sf.data_ptr(),
        weight.data_ptr(),
        weight_sf.data_ptr(),
        output.data_ptr(),
        residual.has_value() ? residual->data_ptr() : nullptr,
        batch_size,
        output_size,
        reduction_size,
        use_2sm_config);
  } else if (use_2sm_config == 2) {
    launch_linear_nvfp4_1d2d_sm100_bn64<cute::float_e2m1_t>(
        input.data_ptr(),
        input_sf.data_ptr(),
        weight.data_ptr(),
        weight_sf.data_ptr(),
        output.data_ptr(),
        residual.has_value() ? residual->data_ptr() : nullptr,
        batch_size,
        output_size,
        reduction_size);
  } else {
    launch_linear_nvfp4_1d2d_sm100<cute::float_e2m1_t>(
        input.data_ptr(),
        input_sf.data_ptr(),
        weight.data_ptr(),
        weight_sf.data_ptr(),
        output.data_ptr(),
        residual.has_value() ? residual->data_ptr() : nullptr,
        batch_size,
        output_size,
        reduction_size);
  }
}

// Auto-select 1SM vs 2SM by an occupancy criterion derived from sweeps.
//
// 1SM and 2SM are equally efficient per FLOP; 2SM just bundles 2 SMs into a
// 256-wide tile. So 2SM only wins once 1SM, even at its widest tile (BLOCK_N=
// 128), can no longer keep the launch in a single SM wave — i.e. when
//     (N/128) * (M/128) > SM_count
// 1SM must spill to multiple waves, and 2SM's wider tile finishes the same work
// in fewer cluster-waves. Equivalently, switch to 2SM once  M*N >
// SM_count*128². This replaces the old hardcoded ">= 32 tiles" threshold (which
// was a rough proxy for the same saturation point) and is SM-count portable.
//
// 2SM also requires M%256==0 and N%256==0 (cluster geometry); else fall back.
//
// The legacy use_2sm/use_2sm_config arguments are ignored for path selection —
// kept for ABI compatibility and explicit routing to experimental variants.
static bool should_use_2sm(int batch_size, int output_size) {
  if (batch_size % 256 != 0) {
    return false;
  }
  if (output_size % 256 != 0) {
    return false;
  }
  static int sm_count = 0;
  if (sm_count == 0) {
    CUTE_CHECK_ERROR(
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  }
  // 1SM saturates (>1 wave at BLOCK_N=128) once (N/128)*(M/128) > SM_count.
  long long const one_sm_tiles = (static_cast<long long>(output_size) / 128) *
                                 (static_cast<long long>(batch_size) / 128);
  return one_sm_tiles > sm_count;
}

void dispatch_linear_nvfp4(torch::Tensor const &input,
                           torch::Tensor const &input_sf,
                           torch::Tensor const &weight,
                           torch::Tensor const &weight_sf,
                           c10::optional<at::Tensor> const &residual,
                           torch::Tensor const &output,
                           int batch_size,
                           int output_size,
                           int reduction_size,
                           bool use_2sm,
                           int use_2sm_config) {
  // Explicit swapAB override (use_2sm=false, use_2sm_config ∈
  // {8,16,32,64,128}): route ANY batch_size through the swapAB kernel with the
  // requested MMA_N. Used by the MMA_N ablation benchmarks comparing swapAB at
  // various MMA_N against the 1d2d MMA_N=128 baseline. The caller MUST quantize
  // the activation SF with the same mma_n (per-tile layout, not interleaved).
  bool const swapab_override =
      !use_2sm &&
      (use_2sm_config == 8 || use_2sm_config == 16 || use_2sm_config == 32 ||
       use_2sm_config == 64 || use_2sm_config == 128);
  // M < 128 → swapAB small-batch kernel. M == 128 falls through to the 1SM
  // large-batch path (BLOCK_N=128), which is faster at that boundary.
  if (batch_size < 128 || swapab_override) {
    // Auto-select the swapAB tile width. Ablation (N=K=2048) showed mma_n=8 is
    // fastest for M<=72 (8 tiles fit one wave); for M in [73,127] mma_n=8
    // spills to a 2nd wave (~1.6x slower) while mma_n=16 stays single-wave at
    // parity. An explicit use_2sm_config in {8,16,32,64,128} overrides (for
    // benchmarking). The caller MUST quantize the activation SF with the same
    // mma_n (see swapab_mma_n in the Python wrapper).
    int mma_n = swapab_mma_n(batch_size, output_size);
    if (use_2sm_config == 8 || use_2sm_config == 16 || use_2sm_config == 32 ||
        use_2sm_config == 64 || use_2sm_config == 128) {
      mma_n = use_2sm_config;
    }
    launch_linear_nvfp4_small_batch(input,
                                    input_sf,
                                    weight,
                                    weight_sf,
                                    residual,
                                    output,
                                    reduction_size,
                                    batch_size,
                                    mma_n);
    return;
  }
  // For the experimental opt-in cfgs (persistent / BN=64), honor caller's
  // explicit choice; otherwise auto-select.
  bool const explicit_path = use_2sm || use_2sm_config != 0;
  bool const pick_2sm =
      explicit_path ? use_2sm : should_use_2sm(batch_size, output_size);
  launch_linear_nvfp4(input,
                      input_sf,
                      weight,
                      weight_sf,
                      residual,
                      output,
                      batch_size,
                      output_size,
                      reduction_size,
                      pick_2sm,
                      use_2sm_config);
}

void validate_linear_tensors(torch::Tensor const &weight,
                             torch::Tensor const &weight_sf,
                             c10::optional<at::Tensor> const &residual,
                             torch::Tensor const &output,
                             int batch_size,
                             bool use_2sm,
                             int use_2sm_config = 0) {
  // All SM100 NVFP4 dispatch paths produce bf16 output now and accept
  // bf16 residual.
  (void)use_2sm;
  (void)use_2sm_config;
  bool const is_swapab = batch_size <= 128;
  (void)is_swapab;
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(weight_sf.is_cuda(), "weight_sf must be a CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(weight.dim() == 2, "weight must be rank-2");
  TORCH_CHECK(output.dim() == 2, "output must be rank-2");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(weight_sf.is_contiguous(), "weight_sf must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(weight.scalar_type() == torch::kUInt8,
              "weight must have dtype uint8");
  TORCH_CHECK(weight_sf.scalar_type() == torch::kUInt8,
              "weight_sf must have dtype uint8");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16,
              "output must have dtype bfloat16 (all SM100 NVFP4 dispatch "
              "paths produce bfloat16 output)");
  TORCH_CHECK(output.size(0) == batch_size,
              "output.shape[0] must equal the logical batch size");
  TORCH_CHECK(output.size(1) == weight.size(0),
              "output.shape[1] must equal weight.shape[0]");
  // All paths (1SM, 1SM persistent, 2SM) accept residual now.
  if (residual.has_value()) {
    TORCH_CHECK(residual->is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(residual->is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(residual->scalar_type() == torch::kBFloat16,
                "residual must have dtype bfloat16 (matches output dtype)");
    TORCH_CHECK(residual->sizes() == output.sizes(),
                "residual must have the same shape as output");
  }
}

void check_cuda_sync(char const *label) {
  cudaError_t err = cudaPeekAtLastError();
  TORCH_CHECK(err == cudaSuccess, label, ": ", cudaGetErrorString(err));
}

} // namespace

// Exposed so callers (using the pre-quantized no_quantization entry) can
// quantize the activation SF with the same tile width the swapAB dispatch will
// use. The wrapper derives M and N from the tensors — the Python caller only
// passes the activation `input` [M, K/2] and `weight` [N, K/2].
int64_t swapab_mma_n_kernel(torch::Tensor input, torch::Tensor weight) {
  int const batch_size = static_cast<int>(input.size(0));
  int const output_size = static_cast<int>(weight.size(0));
  return swapab_mma_n(batch_size, output_size);
}

std::vector<torch::Tensor> quantize_nvfp4_sm100_kernel(torch::Tensor input,
                                                       int64_t mma_n = 0) {
  return dispatch_quantize_nvfp4_sm100(input, static_cast<int>(mma_n));
}

void linear_nvfp4_sm100_no_quantization_kernel(
    torch::Tensor input,
    torch::Tensor input_sf,
    torch::Tensor weight,
    torch::Tensor weight_sf,
    c10::optional<at::Tensor> residual,
    torch::Tensor output,
    bool use_2sm = false,
    int64_t use_2sm_config = 0) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input_sf.is_cuda(), "input_sf must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be rank-2");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input_sf.is_contiguous(), "input_sf must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kUInt8,
              "input must have dtype uint8");
  TORCH_CHECK(input_sf.scalar_type() == torch::kUInt8,
              "input_sf must have dtype uint8");
  TORCH_CHECK(input.size(1) == weight.size(1),
              "input.shape[1] and weight.shape[1] must match");

  int const batch_size = static_cast<int>(output.size(0));
  int const output_size = static_cast<int>(weight.size(0));
  int const reduction_size = static_cast<int>(input.size(1) * 2);
  TORCH_CHECK(input.size(0) >= batch_size,
              "input must provide at least output.shape[0] rows");
  TORCH_CHECK(use_2sm_config >= 0 && use_2sm_config <= 400,
              "use_2sm_config must be in [0, 400]; got ",
              use_2sm_config);
  validate_linear_tensors(weight,
                          weight_sf,
                          residual,
                          output,
                          batch_size,
                          use_2sm,
                          static_cast<int>(use_2sm_config));
  dispatch_linear_nvfp4(input,
                        input_sf,
                        weight,
                        weight_sf,
                        residual,
                        output,
                        batch_size,
                        output_size,
                        reduction_size,
                        use_2sm,
                        static_cast<int>(use_2sm_config));
  check_cuda_sync("linear_nvfp4_sm100_no_quantization");
}

// Auto-quantizing Python entry point.
// - input:     [M, K]    fp32 row-major
// - weight:    [N, K/2]  uint8
// - weight_sf: interleaved weight scale factors
// - output:    [M, N]    bfloat16 for swapAB (M <= 128) and optimized 2SM 1d2d,
//                       float32 for 1SM 1d2d
void linear_nvfp4_sm100_kernel(torch::Tensor input,
                               torch::Tensor weight,
                               torch::Tensor weight_sf,
                               c10::optional<at::Tensor> residual,
                               torch::Tensor output,
                               bool use_2sm = false,
                               int64_t use_2sm_config = 0) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be rank-2");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32,
              "input must have dtype float32");
  validate_linear_tensors(weight,
                          weight_sf,
                          residual,
                          output,
                          static_cast<int>(input.size(0)),
                          use_2sm,
                          static_cast<int>(use_2sm_config));
  TORCH_CHECK(weight.size(1) * 2 == input.size(1),
              "weight.shape[1] must equal input.shape[1] / 2");
  TORCH_CHECK(use_2sm_config >= 0 && use_2sm_config <= 400,
              "use_2sm_config must be in [0, 400]; got ",
              use_2sm_config);

  int const batch_size = static_cast<int>(input.size(0));
  int const output_size = static_cast<int>(weight.size(0));
  int const reduction_size = static_cast<int>(input.size(1));
  // M < 128 → swapAB: quantize SF with the tile width the dispatch will use.
  // M >= 128 → 1SM/2SM path → interleaved layout (mma_n=0).
  int const mma_n =
      (batch_size < 128) ? swapab_mma_n(batch_size, output_size) : 0;
  auto quantized_input = dispatch_quantize_nvfp4_sm100(input, mma_n);
  dispatch_linear_nvfp4(quantized_input[0],
                        quantized_input[1],
                        weight,
                        weight_sf,
                        residual,
                        output,
                        batch_size,
                        output_size,
                        reduction_size,
                        use_2sm,
                        static_cast<int>(use_2sm_config));
  check_cuda_sync("linear_nvfp4_sm100");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_nvfp4_sm100",
        &quantize_nvfp4_sm100_kernel,
        "SM100 NVFP4 quantize. mma_n=0 → interleaved layout for 1d2d path; "
        "mma_n>0 → per-tile swapAB layout [ceil(M/mma_n), K/64, 32, 4, 4].",
        pybind11::arg("input"),
        pybind11::arg("mma_n") = 0);
  m.def("swapab_mma_n",
        &swapab_mma_n_kernel,
        "Canonical swapAB tile width the dispatch auto-selects for these "
        "tensors (derives M from input, N from weight). Quantize activation SF "
        "with this mma_n to match the kernel.",
        pybind11::arg("input"),
        pybind11::arg("weight"));
  m.def("linear_nvfp4_sm100_no_quantization",
        &linear_nvfp4_sm100_no_quantization_kernel,
        "SM100 NVFP4 linear (pre-quantized inputs). All paths produce "
        "bfloat16 output and accept bfloat16 residual. Path is auto-"
        "selected from shape: M<128 uses swapAB (MMA_N by occupancy formula); "
        "large-batch picks 2SM when 1SM saturates one SM wave at BLOCK_N=128 "
        "((N/128)*(M/128) > SM_count) and M,N divisible by 256, else 1SM. "
        "use_2sm/use_2sm_config are honored only when non-default.",
        pybind11::arg("input"),
        pybind11::arg("input_sf"),
        pybind11::arg("weight"),
        pybind11::arg("weight_sf"),
        pybind11::arg("residual"),
        pybind11::arg("output"),
        pybind11::arg("use_2sm") = false,
        pybind11::arg("use_2sm_config") = 0);
  m.def("linear_nvfp4_sm100",
        &linear_nvfp4_sm100_kernel,
        "SM100 NVFP4 linear (quantizes fp32 activations on entry, then "
        "dispatches to the no-quantization kernel). All paths produce "
        "bfloat16 output and accept bfloat16 residual. Path is auto-"
        "selected from shape.",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("weight_sf"),
        pybind11::arg("residual"),
        pybind11::arg("output"),
        pybind11::arg("use_2sm") = false,
        pybind11::arg("use_2sm_config") = 0);
}
