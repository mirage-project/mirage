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

// Minimal MXFP4 PyTorch wrapper. Mirrors the NVFP4 entry points (quantize +
// linear) but with a compact dispatch table. Extend the switch statements
// below as needed.

#include "blackwell/linear_mxfp4_1d2d_2sm_sm100.cuh"
#include "blackwell/linear_mxfp4_1d2d_sm100.cuh"
#include "blackwell/linear_mxfp4_swapAB_sm100.cuh"
#include "blackwell/quantize_mxfp4_sm100.cuh"
#include "hopper/tma_fp4.cuh" // kernel::tma::init_*_tmap_fp4
#include "runtime_header.h"
#include "tma.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cute/numeric/numeric_types.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/util/print_error.hpp>

using float_e2m1 = cute::float_e2m1_t;
using bfloat16 = cute::bfloat16_t;

// ---------------------------------------------------------------------------
// Quantizer
// ---------------------------------------------------------------------------

constexpr int QUANTIZE_THREADS = 128;
constexpr int MXFP4_GROUP_SIZE = 32;

template <typename T, int HIDDEN_SIZE>
__global__ __launch_bounds__(
    QUANTIZE_THREADS,
    1) void quantize_mxfp4_sm100_wrapper(T const *input_ptr,
                                         uint8_t *output_q_ptr,
                                         uint8_t *output_s_ptr,
                                         int batch_size,
                                         int mma_n) {
  kernel::quantize_mxfp4_sm100_task_impl<HIDDEN_SIZE,
                                         MXFP4_GROUP_SIZE,
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

// mma_n == 0 → interleaved layout [padded/128, K/64, 32, 4, 4]
// mma_n  > 0 → per-tile swapAB layout [ceil(batch/mma_n), K/64, 32, 4, 4]
//
// Each 512-B SF atom covers 64 K-elements (= 2 MXFP4 scales × SFVecSize=32).
// The inner `4` axis holds the 2 active scales in positions 0,1 and zero
// padding in positions 2,3 (NSF=2 for MXFP4 vec::2X).
template <int HIDDEN_SIZE>
std::vector<torch::Tensor>
    launch_quantize_mxfp4_sm100(torch::Tensor const &input, int mma_n) {
  int const batch_size = static_cast<int>(input.size(0));
  int const padded_batch_size = ((batch_size + 127) / 128) * 128;
  int const sf_k_outer = HIDDEN_SIZE / 64;

  auto output_q = torch::empty({padded_batch_size, HIDDEN_SIZE / 2},
                               input.options().dtype(torch::kUInt8));

  at::Tensor output_s;
  if (mma_n > 0) {
    int const num_n_tiles = (batch_size + mma_n - 1) / mma_n;
    output_s = torch::empty({num_n_tiles, sf_k_outer, 32, 4, 4},
                            input.options().dtype(torch::kUInt8));
  } else {
    output_s = torch::empty({padded_batch_size / 128, sf_k_outer, 32, 4, 4},
                            input.options().dtype(torch::kUInt8));
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  quantize_mxfp4_sm100_wrapper<float, HIDDEN_SIZE>
      <<<dim3(padded_batch_size), dim3(QUANTIZE_THREADS), 0, stream>>>(
          static_cast<float const *>(input.data_ptr()),
          static_cast<uint8_t *>(output_q.data_ptr()),
          static_cast<uint8_t *>(output_s.data_ptr()),
          batch_size,
          mma_n);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "quantize_mxfp4_sm100 launch failed: ",
              cudaGetErrorString(err));
  return {output_q, output_s};
}

std::vector<torch::Tensor>
    dispatch_quantize_mxfp4_sm100(torch::Tensor const &input, int mma_n) {
  int const hidden_size = static_cast<int>(input.size(1));
  TORCH_CHECK(hidden_size % 64 == 0, "input.shape[1] must be divisible by 64");
  switch (hidden_size) {
    case 256:
      return launch_quantize_mxfp4_sm100<256>(input, mma_n);
    case 512:
      return launch_quantize_mxfp4_sm100<512>(input, mma_n);
    case 1024:
      return launch_quantize_mxfp4_sm100<1024>(input, mma_n);
    case 2048:
      return launch_quantize_mxfp4_sm100<2048>(input, mma_n);
    case 4096:
      return launch_quantize_mxfp4_sm100<4096>(input, mma_n);
    case 7168:
      return launch_quantize_mxfp4_sm100<7168>(input, mma_n);
    case 8192:
      return launch_quantize_mxfp4_sm100<8192>(input, mma_n);
    case 16384:
      return launch_quantize_mxfp4_sm100<16384>(input, mma_n);
    default:
      TORCH_CHECK(false,
                  "quantize_mxfp4_sm100 supports K in "
                  "{256,512,1024,2048,4096,7168,8192,16384}; got ",
                  hidden_size);
  }
}

// ---------------------------------------------------------------------------
// 1SM 1d2d MXFP4 launcher
// ---------------------------------------------------------------------------

template <int REDUCTION_SIZE,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          int NUM_STAGES,
          int EPI_BATCH_LA = 1>
void launch_linear_mxfp4_1d2d_sm100_config(void *input_ptr,
                                           void *input_sf_ptr,
                                           void *weight_ptr,
                                           void *weight_sf_ptr,
                                           void *output_ptr,
                                           void *residual_ptr,
                                           int batch_size,
                                           int output_size) {
  static_assert(BLOCK_M == 128, "1d2d MXFP4 uses a 128-row tile");
  TORCH_CHECK(batch_size % BLOCK_N == 0,
              "MXFP4 1d2d requires batch_size divisible by BLOCK_N=",
              BLOCK_N);
  TORCH_CHECK(output_size % BLOCK_M == 0,
              "MXFP4 1d2d requires output_size divisible by BLOCK_M=",
              BLOCK_M);

  CUtensorMap A_tmap{};
  CUtensorMap B_tmap{};
  // Swap-AB: A = weight [output, K], B = input [batch, K].
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
  constexpr int SFA_size = 128 * BLOCK_K / 16; // 512-B atom shared with NVFP4
  constexpr int SFB_size = 128 * BLOCK_K / 16;
  constexpr int stage_size = A_size + B_size + SFA_size + SFB_size;
  constexpr int smem_bytes = stage_size * NUM_STAGES;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid_dim(output_size / BLOCK_M, batch_size / BLOCK_N, 1);
  dim3 block_dim(BLOCK_M + 2 * 32, 1, 1);

  auto kernel_ptr = kernel::linear_mxfp4_1d2d_sm100_kernel<
      /*BATCH_SIZE=*/0,
      /*OUTPUT_SIZE=*/0,
      REDUCTION_SIZE,
      BLOCK_M,
      BLOCK_N,
      BLOCK_K,
      NUM_STAGES,
      /*C_N_MAJOR=*/false,
      EPI_BATCH_LA>;
  if constexpr (smem_bytes > 48 * 1024) {
    CUTE_CHECK_ERROR(cudaFuncSetAttribute(
        kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  }
  kernel_ptr<<<grid_dim, block_dim, smem_bytes, stream>>>(
      A_tmap,
      B_tmap,
      reinterpret_cast<char const *>(weight_sf_ptr),
      reinterpret_cast<char const *>(input_sf_ptr),
      reinterpret_cast<type::bfloat16_t *>(output_ptr),
      reinterpret_cast<type::bfloat16_t const *>(residual_ptr),
      output_size,
      batch_size);
  CUTE_CHECK_ERROR(cudaGetLastError());
}

void launch_linear_mxfp4_1d2d_sm100(void *input_ptr,
                                    void *input_sf_ptr,
                                    void *weight_ptr,
                                    void *weight_sf_ptr,
                                    void *output_ptr,
                                    void *residual_ptr,
                                    int batch_size,
                                    int output_size,
                                    int reduction_size) {
  // Compact dispatch table: BLOCK_M=128, BLOCK_N=128, BLOCK_K=256.
  switch (reduction_size) {
    case 256:
      return launch_linear_mxfp4_1d2d_sm100_config<256, 128, 128, 256, 6>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 512:
      return launch_linear_mxfp4_1d2d_sm100_config<512, 128, 128, 256, 6>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 1024:
      return launch_linear_mxfp4_1d2d_sm100_config<1024, 128, 128, 256, 6>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 2048:
      return launch_linear_mxfp4_1d2d_sm100_config<2048, 128, 128, 256, 6>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 4096:
      return launch_linear_mxfp4_1d2d_sm100_config<4096, 128, 128, 256, 6>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 7168:
      return launch_linear_mxfp4_1d2d_sm100_config<7168, 128, 128, 256, 6>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 8192:
      return launch_linear_mxfp4_1d2d_sm100_config<8192, 128, 128, 256, 6>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    case 16384:
      return launch_linear_mxfp4_1d2d_sm100_config<16384, 128, 128, 256, 6>(
          input_ptr,
          input_sf_ptr,
          weight_ptr,
          weight_sf_ptr,
          output_ptr,
          residual_ptr,
          batch_size,
          output_size);
    default:
      TORCH_CHECK(
          false, "linear_mxfp4_1d2d_sm100: unsupported K=", reduction_size);
  }
}

// ---------------------------------------------------------------------------
// 2SM 1d2d MXFP4 launcher
// ---------------------------------------------------------------------------

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
          bool OVERLAP_OUTPUT_MBAR>
void launch_linear_mxfp4_1d2d_2sm_sm100_config(void *input_ptr,
                                               void *input_sf_ptr,
                                               void *weight_ptr,
                                               void *weight_sf_ptr,
                                               void *output_ptr,
                                               void *residual_ptr,
                                               int batch_size,
                                               int output_size) {
  TORCH_CHECK(batch_size % BLOCK_N == 0,
              "MXFP4 2SM 1d2d requires batch_size divisible by BLOCK_N=",
              BLOCK_N);
  TORCH_CHECK(output_size % (2 * BLOCK_M) == 0,
              "MXFP4 2SM 1d2d requires output_size divisible by 2*BLOCK_M=",
              2 * BLOCK_M);

  CUtensorMap A_tmap{}, B_tmap{}, C_tmap{}, SFA_tmap{}, SFB_tmap{};
  kernel::tma::init_AB_tmap_fp4(&A_tmap,
                                reinterpret_cast<char const *>(weight_ptr),
                                output_size,
                                REDUCTION_SIZE,
                                BLOCK_M,
                                BLOCK_K);
  kernel::tma::init_AB_tmap_fp4(&B_tmap,
                                reinterpret_cast<char const *>(input_ptr),
                                batch_size,
                                REDUCTION_SIZE,
                                BLOCK_N / 2,
                                BLOCK_K);
  kernel::tma::init_SF_tmap_fp4(&SFA_tmap,
                                reinterpret_cast<char const *>(weight_sf_ptr),
                                output_size,
                                REDUCTION_SIZE,
                                BLOCK_K / 64);
  kernel::tma::init_SF_tmap_fp4(&SFB_tmap,
                                reinterpret_cast<char const *>(input_sf_ptr),
                                batch_size,
                                REDUCTION_SIZE,
                                BLOCK_K / 64);
  kernel::tma::init_C_tmap_fp4(
      &C_tmap, output_ptr, batch_size, output_size, EPI_TILE_N, BLOCK_M);

  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = (BLOCK_N / 2) * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16; // 512-B atom shared with NVFP4
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

  bool const has_bias = (residual_ptr != nullptr);
  auto kernel_ptr_nobias =
      kernel::linear_mxfp4_1d2d_2sm_sm100_kernel<REDUCTION_SIZE,
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
      kernel::linear_mxfp4_1d2d_2sm_sm100_kernel<REDUCTION_SIZE,
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
              "MXFP4 2SM 1d2d cluster launch failed: ",
              cutlassGetStatusString(status));
}

void launch_linear_mxfp4_1d2d_2sm_sm100(void *input_ptr,
                                        void *input_sf_ptr,
                                        void *weight_ptr,
                                        void *weight_sf_ptr,
                                        void *output_ptr,
                                        void *residual_ptr,
                                        int batch_size,
                                        int output_size,
                                        int reduction_size,
                                        int config_id = 0) {
  // SG-ablation path (cfg = 100 + SG). Wired for K=16384 only; the production
  // default (cfg=0) uses SG=4 across all K (empirical from the stages-sweep
  // study and mirrored from NVFP4).
  if (config_id >= 101 && config_id <= 164 && reduction_size == 16384) {
    auto launch_sg = [&](auto SgTag) -> bool {
      constexpr int SG = decltype(SgTag)::value;
      if (config_id - 100 != SG) {
        return false;
      }
      launch_linear_mxfp4_1d2d_2sm_sm100_config<16384,
                                                128,
                                                256,
                                                256,
                                                5,
                                                SG,
                                                64,
                                                2,
                                                false,
                                                1,
                                                true>(input_ptr,
                                                      input_sf_ptr,
                                                      weight_ptr,
                                                      weight_sf_ptr,
                                                      output_ptr,
                                                      residual_ptr,
                                                      batch_size,
                                                      output_size);
      return true;
    };
    bool dispatched = launch_sg(std::integral_constant<int, 1>{}) ||
                      launch_sg(std::integral_constant<int, 2>{}) ||
                      launch_sg(std::integral_constant<int, 4>{}) ||
                      launch_sg(std::integral_constant<int, 8>{}) ||
                      launch_sg(std::integral_constant<int, 14>{}) ||
                      launch_sg(std::integral_constant<int, 16>{}) ||
                      launch_sg(std::integral_constant<int, 24>{}) ||
                      launch_sg(std::integral_constant<int, 32>{}) ||
                      launch_sg(std::integral_constant<int, 48>{}) ||
                      launch_sg(std::integral_constant<int, 64>{});
    TORCH_CHECK(dispatched,
                "linear_mxfp4_1d2d_2sm_sm100: SG ablation supports SG ∈ "
                "{1,2,4,8,14,16,24,32,48,64}; got ",
                config_id - 100);
    return;
  }

  switch (reduction_size) {
    case 256:
      return launch_linear_mxfp4_1d2d_2sm_sm100_config<256,
                                                       128,
                                                       256,
                                                       256,
                                                       4,
                                                       4,
                                                       64,
                                                       2,
                                                       false,
                                                       1,
                                                       true>(input_ptr,
                                                             input_sf_ptr,
                                                             weight_ptr,
                                                             weight_sf_ptr,
                                                             output_ptr,
                                                             residual_ptr,
                                                             batch_size,
                                                             output_size);
    case 512:
      return launch_linear_mxfp4_1d2d_2sm_sm100_config<512,
                                                       128,
                                                       256,
                                                       256,
                                                       4,
                                                       4,
                                                       64,
                                                       2,
                                                       false,
                                                       1,
                                                       true>(input_ptr,
                                                             input_sf_ptr,
                                                             weight_ptr,
                                                             weight_sf_ptr,
                                                             output_ptr,
                                                             residual_ptr,
                                                             batch_size,
                                                             output_size);
    case 1024:
      return launch_linear_mxfp4_1d2d_2sm_sm100_config<1024,
                                                       128,
                                                       256,
                                                       256,
                                                       4,
                                                       4,
                                                       64,
                                                       2,
                                                       false,
                                                       1,
                                                       true>(input_ptr,
                                                             input_sf_ptr,
                                                             weight_ptr,
                                                             weight_sf_ptr,
                                                             output_ptr,
                                                             residual_ptr,
                                                             batch_size,
                                                             output_size);
    case 2048:
      return launch_linear_mxfp4_1d2d_2sm_sm100_config<2048,
                                                       128,
                                                       256,
                                                       256,
                                                       4,
                                                       4,
                                                       64,
                                                       2,
                                                       false,
                                                       1,
                                                       true>(input_ptr,
                                                             input_sf_ptr,
                                                             weight_ptr,
                                                             weight_sf_ptr,
                                                             output_ptr,
                                                             residual_ptr,
                                                             batch_size,
                                                             output_size);
    case 4096:
      return launch_linear_mxfp4_1d2d_2sm_sm100_config<4096,
                                                       128,
                                                       256,
                                                       256,
                                                       5,
                                                       4,
                                                       64,
                                                       2,
                                                       false,
                                                       1,
                                                       true>(input_ptr,
                                                             input_sf_ptr,
                                                             weight_ptr,
                                                             weight_sf_ptr,
                                                             output_ptr,
                                                             residual_ptr,
                                                             batch_size,
                                                             output_size);
    case 7168:
      return launch_linear_mxfp4_1d2d_2sm_sm100_config<7168,
                                                       128,
                                                       256,
                                                       256,
                                                       5,
                                                       4,
                                                       64,
                                                       2,
                                                       false,
                                                       1,
                                                       true>(input_ptr,
                                                             input_sf_ptr,
                                                             weight_ptr,
                                                             weight_sf_ptr,
                                                             output_ptr,
                                                             residual_ptr,
                                                             batch_size,
                                                             output_size);
    case 8192:
      return launch_linear_mxfp4_1d2d_2sm_sm100_config<8192,
                                                       128,
                                                       256,
                                                       256,
                                                       5,
                                                       4,
                                                       64,
                                                       2,
                                                       false,
                                                       1,
                                                       true>(input_ptr,
                                                             input_sf_ptr,
                                                             weight_ptr,
                                                             weight_sf_ptr,
                                                             output_ptr,
                                                             residual_ptr,
                                                             batch_size,
                                                             output_size);
    case 16384:
      return launch_linear_mxfp4_1d2d_2sm_sm100_config<16384,
                                                       128,
                                                       256,
                                                       256,
                                                       5,
                                                       4,
                                                       64,
                                                       2,
                                                       false,
                                                       1,
                                                       true>(input_ptr,
                                                             input_sf_ptr,
                                                             weight_ptr,
                                                             weight_sf_ptr,
                                                             output_ptr,
                                                             residual_ptr,
                                                             batch_size,
                                                             output_size);
    default:
      TORCH_CHECK(
          false, "linear_mxfp4_1d2d_2sm_sm100: unsupported K=", reduction_size);
  }
}

// ---------------------------------------------------------------------------
// PyTorch entry points
// ---------------------------------------------------------------------------

void launch_linear_mxfp4(torch::Tensor const &input,
                         torch::Tensor const &input_sf,
                         torch::Tensor const &weight,
                         torch::Tensor const &weight_sf,
                         c10::optional<at::Tensor> const &residual,
                         torch::Tensor const &output,
                         int batch_size,
                         int output_size,
                         int reduction_size,
                         bool use_2sm,
                         int config_id = 0) {
  void *res_ptr = residual.has_value() ? residual->data_ptr() : nullptr;
  if (use_2sm) {
    launch_linear_mxfp4_1d2d_2sm_sm100(input.data_ptr(),
                                       input_sf.data_ptr(),
                                       weight.data_ptr(),
                                       weight_sf.data_ptr(),
                                       output.data_ptr(),
                                       res_ptr,
                                       batch_size,
                                       output_size,
                                       reduction_size,
                                       config_id);
  } else {
    // 1SM has no SG ablation (no persistent loop / supergroup mapping).
    TORCH_CHECK(config_id == 0,
                "MXFP4 1SM does not support config_id != 0; got ",
                config_id);
    launch_linear_mxfp4_1d2d_sm100(input.data_ptr(),
                                   input_sf.data_ptr(),
                                   weight.data_ptr(),
                                   weight_sf.data_ptr(),
                                   output.data_ptr(),
                                   res_ptr,
                                   batch_size,
                                   output_size,
                                   reduction_size);
  }
}

torch::Tensor linear_mxfp4_sm100_no_quantization_kernel(
    torch::Tensor input,
    torch::Tensor input_sf,
    torch::Tensor weight,
    torch::Tensor weight_sf,
    c10::optional<at::Tensor> residual,
    bool use_2sm,
    int64_t config_id = 0) {
  int const batch_size = static_cast<int>(input.size(0));
  int const output_size = static_cast<int>(weight.size(0));
  int const reduction_size =
      static_cast<int>(weight.size(1)) * 2; // packed e2m1
  auto output = torch::empty({batch_size, output_size},
                             input.options().dtype(torch::kBFloat16));
  launch_linear_mxfp4(input,
                      input_sf,
                      weight,
                      weight_sf,
                      residual,
                      output,
                      batch_size,
                      output_size,
                      reduction_size,
                      use_2sm,
                      static_cast<int>(config_id));
  return output;
}

torch::Tensor linear_mxfp4_sm100_kernel(torch::Tensor input,
                                        torch::Tensor weight,
                                        torch::Tensor weight_sf,
                                        c10::optional<at::Tensor> residual,
                                        bool use_2sm) {
  // Quantize activations to MXFP4, then run the linear.
  auto qpack = dispatch_quantize_mxfp4_sm100(input, /*mma_n=*/0);
  return linear_mxfp4_sm100_no_quantization_kernel(
      qpack[0], qpack[1], weight, weight_sf, residual, use_2sm);
}

std::vector<torch::Tensor> quantize_mxfp4_sm100_kernel(torch::Tensor input,
                                                       int64_t mma_n) {
  return dispatch_quantize_mxfp4_sm100(input, static_cast<int>(mma_n));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_mxfp4_sm100",
        &quantize_mxfp4_sm100_kernel,
        "MXFP4 (e2m1 + e8m0) quantizer; returns {packed_e2m1, e8m0_scales}",
        pybind11::arg("input"),
        pybind11::arg("mma_n") = 0);
  m.def("linear_mxfp4_sm100_no_quantization",
        &linear_mxfp4_sm100_no_quantization_kernel,
        "MXFP4 linear with pre-quantized input",
        pybind11::arg("input"),
        pybind11::arg("input_sf"),
        pybind11::arg("weight"),
        pybind11::arg("weight_sf"),
        pybind11::arg("residual") = c10::optional<at::Tensor>(),
        pybind11::arg("use_2sm") = false,
        pybind11::arg("config_id") = 0);
  m.def("linear_mxfp4_sm100",
        &linear_mxfp4_sm100_kernel,
        "MXFP4 linear; quantizes the input on-the-fly",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("weight_sf"),
        pybind11::arg("residual") = c10::optional<at::Tensor>(),
        pybind11::arg("use_2sm") = false);
}
