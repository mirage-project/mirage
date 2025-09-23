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
#include "bfloat16.h"
#include "hopper/linear_hopper.cuh"
#include "hopper/linear_swapAB_hopper.cuh"
#include "hopper/multitoken_paged_attention_hopper.cuh"
#include "hopper/norm_linear_hopper.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>

using kernel::linear_kernel_hopper;
using kernel::linear_swapAB_kernel_hopper;
using kernel::multitoken_paged_attention_hopper_impl;
using kernel::norm_linear_kernel_hopper;
using bfloat16 = type::bfloat16_t;

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          typename TMA_A,
          typename TMA_B,
          typename TMA_RESIDUAL,
          typename TMA_OUT,
          int Kstages = 4>
__global__ __launch_bounds__(256, 1) void linear_kernel_swapAB_hopper_wrapper(
    const __grid_constant__ TMA_A tma_a,
    const __grid_constant__ TMA_B tma_b,
    const __grid_constant__ TMA_RESIDUAL tma_residual,
    const __grid_constant__ TMA_OUT tma_out) {

  linear_swapAB_kernel_hopper<T,
                              BATCH_SIZE,
                              OUTPUT_SIZE,
                              REDUCTION_SIZE,
                              Kstages,
                              TMA_A,
                              TMA_B,
                              TMA_OUT>(tma_a, tma_b, tma_out, &tma_residual);
}

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          typename TMA_A,
          typename TMA_B,
          typename TMA_OUT,
          int Kstages = 2>
__global__
    __launch_bounds__(256, 1) void linear_kernel_swapAB_no_residual_hopper_wrapper(
        const __grid_constant__ TMA_A tma_a,
        const __grid_constant__ TMA_B tma_b,
        const __grid_constant__ TMA_OUT tma_out) {

  linear_swapAB_kernel_hopper<T,
                              BATCH_SIZE,
                              OUTPUT_SIZE,
                              REDUCTION_SIZE,
                              Kstages,
                              TMA_A,
                              TMA_B,
                              TMA_OUT,
                              void>(tma_a, tma_b, tma_out, nullptr);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_linear_swapAB(void *input_ptr,
                          void *weight_ptr,
                          void *residual_ptr,
                          void *output_ptr = nullptr) {

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int TMA_CP_ASYNC_SIZE =
      64; // note that if swizzle 128 is used, 64 is maximal cp size
  constexpr int TILE_SIZE =
      128; // we should modify this param if we want larger tile size
  constexpr int TMA_CP_ASYNC_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;

  constexpr int OUTPUT_ATOM_SIZE = 64; // this is padded
  constexpr int OUTPUT_TMA_CP_SIZE = OUTPUT_SIZE < 64 ? OUTPUT_SIZE : 64;
  constexpr int OUTPUT_ATOM_REPEAT_COL = 1;

  constexpr int SMEM_M_SIZE = 16;
  using TMA_B =
      kernel::tma::tma_2d<bfloat16,
                          B,
                          M,
                          S,
                          BATCH_SIZE,                      /*GMEM_ROW_*/
                          REDUCTION_SIZE,                  /*GMEM_COL_*/
                          BATCH_SIZE,                      /*SMEM_ROW_*/
                          TMA_CP_ASYNC_SIZE,               /*SMEM_COL_*/
                          REDUCTION_SIZE,                  /*GMEM_STRIDE_ROW_*/
                          1,                               /*GMEM_STRIDE_COL_*/
                          1,                               /*SMEM_REPEAT_ROW_*/
                          TMA_CP_ASYNC_REPEAT_COL,         /*SMEM_REPEAT_COL_*/
                          SMEM_M_SIZE * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                          true>;
  using TMA_A =
      kernel::tma::tma_2d<bfloat16,
                          B,
                          M,
                          S,
                          OUTPUT_SIZE,             /*GMEM_ROW_*/
                          REDUCTION_SIZE,          /*GMEM_COL_*/
                          OUTPUT_ATOM_SIZE,        /*SMEM_ROW_*/
                          TMA_CP_ASYNC_SIZE,       /*SMEM_COL_*/
                          REDUCTION_SIZE,          /*GMEM_STRIDE_ROW_*/
                          1,                       /*GMEM_STRIDE_COL_*/
                          1,                       /*SMEM_REPEAT_ROW_*/
                          TMA_CP_ASYNC_REPEAT_COL, /*SMEM_REPEAT_COL_*/
                          OUTPUT_ATOM_SIZE * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                          true>;
  using TMA_RESIDUAL = kernel::tma::tma_2d<bfloat16,
                                           0,
                                           0,
                                           0,
                                           BATCH_SIZE,
                                           OUTPUT_SIZE,
                                           BATCH_SIZE,
                                           OUTPUT_TMA_CP_SIZE,
                                           OUTPUT_SIZE,
                                           1,
                                           1,
                                           OUTPUT_ATOM_REPEAT_COL,
                                           SMEM_M_SIZE * OUTPUT_TMA_CP_SIZE,
                                           true>;

  using TMA_OUT = kernel::tma::tma_2d<bfloat16,
                                      B,
                                      M,
                                      S,
                                      BATCH_SIZE,
                                      OUTPUT_SIZE,
                                      BATCH_SIZE,
                                      OUTPUT_TMA_CP_SIZE,
                                      OUTPUT_SIZE,
                                      1,
                                      1,
                                      OUTPUT_ATOM_REPEAT_COL,
                                      SMEM_M_SIZE * TMA_CP_ASYNC_SIZE,
                                      true>;
  TMA_A tma_a(weight_ptr);
  TMA_B tma_b(input_ptr);
  TMA_RESIDUAL tma_residual(residual_ptr);
  TMA_OUT tma_out(output_ptr);

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(256, 1, 1);
  size_t smem_size = 224 * 1024;

  if (residual_ptr != nullptr) {
    cudaFuncSetAttribute(linear_kernel_swapAB_hopper_wrapper<T,
                                                             BATCH_SIZE,
                                                             OUTPUT_SIZE,
                                                             REDUCTION_SIZE,
                                                             TMA_A,
                                                             TMA_B,
                                                             TMA_RESIDUAL,
                                                             TMA_OUT>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);

  } else {

    cudaFuncSetAttribute(
        linear_kernel_swapAB_no_residual_hopper_wrapper<T,
                                                        BATCH_SIZE,
                                                        OUTPUT_SIZE,
                                                        REDUCTION_SIZE,
                                                        TMA_A,
                                                        TMA_B,
                                                        TMA_OUT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
  }

#ifndef MIRAGE_PROFILE_HOPPER
  if (residual_ptr != nullptr) {

    linear_kernel_swapAB_hopper_wrapper<T,
                                        BATCH_SIZE,
                                        OUTPUT_SIZE,
                                        REDUCTION_SIZE,
                                        TMA_A,
                                        TMA_B,
                                        TMA_RESIDUAL,
                                        TMA_OUT>
        <<<grid_dim, block_dim, smem_size>>>(
            tma_a, tma_b, tma_residual, tma_out);
  } else {

    linear_kernel_swapAB_no_residual_hopper_wrapper<T,
                                                    BATCH_SIZE,
                                                    OUTPUT_SIZE,
                                                    REDUCTION_SIZE,
                                                    TMA_A,
                                                    TMA_B,
                                                    TMA_OUT>
        <<<grid_dim, block_dim, smem_size>>>(tma_a, tma_b, tma_out);
  }
#else

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  constexpr int WARMUP_RUNS = 16;
  constexpr int BENCHMARK_RUNS = 1000;

  printf("=== Kernel Performance Profiling ===\n");

  for (int i = 0; i < WARMUP_RUNS; i++) {
    if (residual_ptr != nullptr) {
      linear_kernel_swapAB_hopper_wrapper<T,
                                          BATCH_SIZE,
                                          OUTPUT_SIZE,
                                          REDUCTION_SIZE,
                                          TMA_A,
                                          TMA_B,
                                          TMA_RESIDUAL,
                                          TMA_OUT>
          <<<grid_dim, block_dim, smem_size>>>(
              tma_a, tma_b, tma_residual, tma_out);
    } else {
      linear_kernel_swapAB_no_residual_hopper_wrapper<T,
                                                      BATCH_SIZE,
                                                      OUTPUT_SIZE,
                                                      REDUCTION_SIZE,
                                                      TMA_A,
                                                      TMA_B,
                                                      TMA_OUT>
          <<<grid_dim, block_dim, smem_size>>>(tma_a, tma_b, tma_out);
    }
  }
  cudaDeviceSynchronize(); // Wait for all warmup runs to complete

  printf("Running %d benchmark iterations...\n", BENCHMARK_RUNS);

  float *iteration_times = new float[BENCHMARK_RUNS];
  float total_time_ms = 0.0f;

  for (int i = 0; i < BENCHMARK_RUNS; i++) {
    cudaEventRecord(start);
    if (residual_ptr != nullptr) {
      linear_kernel_swapAB_hopper_wrapper<T,
                                          BATCH_SIZE,
                                          OUTPUT_SIZE,
                                          REDUCTION_SIZE,
                                          TMA_A,
                                          TMA_B,
                                          TMA_RESIDUAL,
                                          TMA_OUT>
          <<<grid_dim, block_dim, smem_size>>>(
              tma_a, tma_b, tma_residual, tma_out);
    } else {
      linear_kernel_swapAB_no_residual_hopper_wrapper<T,
                                                      BATCH_SIZE,
                                                      OUTPUT_SIZE,
                                                      REDUCTION_SIZE,
                                                      TMA_A,
                                                      TMA_B,
                                                      TMA_OUT>
          <<<grid_dim, block_dim, smem_size>>>(tma_a, tma_b, tma_out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float iteration_time_ms;
    cudaEventElapsedTime(&iteration_time_ms, start, stop);

    total_time_ms += iteration_time_ms;
  }

  float avg_time_ms = total_time_ms / BENCHMARK_RUNS;

  printf("\n=== Performance Results ===\n");
  printf("Configuration:\n");
  printf("  BATCH_SIZE=%d, OUTPUT_SIZE=%d, REDUCTION_SIZE=%d\n",
         BATCH_SIZE,
         OUTPUT_SIZE,
         REDUCTION_SIZE);
  printf(" TILE SIZE: %d\n", TILE_SIZE);
  printf("  Average: %.3f ms\n", avg_time_ms);

  printf("===============================\n");

  delete[] iteration_times;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
#endif
}

#define DISPATCH_LINEAR_SWAPAB_REDUCTION_SIZE_CASE(                            \
    BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE)                                   \
  case REDUCTION_SIZE:                                                         \
    launch_linear_swapAB<bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(   \
        input_ptr, weight_ptr, residual_ptr, output_ptr);                      \
    break;

#define DISPATCH_LINEAR_SWAPAB_REDUCTION_SIZE(BATCH_SIZE, OUTPUT_SIZE)         \
  switch (input.size(1)) {                                                     \
    /*                                                                         \
    DISPATCH_LINEAR_SWAPAB_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 64)    \
    DISPATCH_LINEAR_SWAPAB_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 128)   \
    DISPATCH_LINEAR_SWAPAB_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 256)   \
    DISPATCH_LINEAR_SWAPAB_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 512)   \
    DISPATCH_LINEAR_SWAPAB_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 3072)  \
    */                                                                         \
    DISPATCH_LINEAR_SWAPAB_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 12288) \
    DISPATCH_LINEAR_SWAPAB_REDUCTION_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE, 4096)  \
    default:                                                                   \
      printf("Unsupported reduction size in test: %zu\n", input.size(1));      \
      break;                                                                   \
  }

#define DISPATCH_LINEAR_SWAPAB_OUTPUT_SIZE_CASE(BATCH_SIZE, OUTPUT_SIZE)       \
  case OUTPUT_SIZE:                                                            \
    DISPATCH_LINEAR_SWAPAB_REDUCTION_SIZE(BATCH_SIZE, OUTPUT_SIZE)             \
    break;

#define DISPATCH_LINEAR_SWAPAB_OUTPUT_SIZE(BATCH_SIZE)                         \
  switch (output.size(1)) {                                                    \
    DISPATCH_LINEAR_SWAPAB_OUTPUT_SIZE_CASE(BATCH_SIZE, 48)                    \
    DISPATCH_LINEAR_SWAPAB_OUTPUT_SIZE_CASE(BATCH_SIZE, 64)                    \
    DISPATCH_LINEAR_SWAPAB_OUTPUT_SIZE_CASE(BATCH_SIZE, 128)                   \
    DISPATCH_LINEAR_SWAPAB_OUTPUT_SIZE_CASE(BATCH_SIZE, 256)                   \
    DISPATCH_LINEAR_SWAPAB_OUTPUT_SIZE_CASE(BATCH_SIZE, 2048)                  \
    /*                                                                         \
    DISPATCH_LINEAR_SWAPAB_OUTPUT_SIZE_CASE(BATCH_SIZE, 192)                   \
    DISPATCH_LINEAR_SWAPAB_OUTPUT_SIZE_CASE(BATCH_SIZE, 16)                    \
    DISPATCH_LINEAR_SWAPAB_OUTPUT_SIZE_CASE(BATCH_SIZE, 1600)                  \
    DISPATCH_LINEAR_SWAPAB_OUTPUT_SIZE_CASE(BATCH_SIZE, 32)                    \
    DISPATCH_LINEAR_SWAPAB_OUTPUT_SIZE_CASE(BATCH_SIZE, 512)                   \
    DISPATCH_LINEAR_SWAPAB_OUTPUT_SIZE_CASE(BATCH_SIZE, 1024)                  \
    */                                                                         \
    default:                                                                   \
      printf("Unsupported output size in test: %zu\n", output.size(1));        \
      break;                                                                   \
  }

#define DISPATCH_LINEAR_SWAPAB_BATCH_SIZE_CASE(BATCH_SIZE)                     \
  case BATCH_SIZE:                                                             \
    DISPATCH_LINEAR_SWAPAB_OUTPUT_SIZE(BATCH_SIZE)                             \
    break;

void linear_swapAB_kernel(torch::Tensor input,
                          torch::Tensor weight,
                          c10::optional<at::Tensor> residual,
                          torch::Tensor output) {

  void *input_ptr = input.data_ptr();
  void *weight_ptr = weight.data_ptr();
  bool has_residual = residual.has_value();
  void *residual_ptr = has_residual ? residual->data_ptr() : nullptr;
  void *output_ptr = output.data_ptr();

  switch (input.size(0)) {
    DISPATCH_LINEAR_SWAPAB_BATCH_SIZE_CASE(8)
    DISPATCH_LINEAR_SWAPAB_BATCH_SIZE_CASE(16)
    default:
      printf("Unsupported batch size in test: %zu\n", output.size(0));
      break;
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

// template <typename T,
//           int BATCH_SIZE,
//           int OUTPUT_SIZE,
//           int REDUCTION_SIZE,
//           typename TMA_A,
//           typename TMA_B,
//           typename TMA_RESIDUAL,
//           typename TMA_OUT,
//           int Kstages = 2>
// __global__ __launch_bounds__(256, 1) void linear_kernel_hopper_wrapper(
//     const __grid_constant__ TMA_A tma_a,
//     const __grid_constant__ TMA_B tma_b,
//     const __grid_constant__ TMA_RESIDUAL tma_residual,
//     const __grid_constant__ TMA_OUT tma_out) {

//   linear_kernel_hopper<T,
//                        BATCH_SIZE,
//                        OUTPUT_SIZE,
//                        REDUCTION_SIZE,
//                        Kstages,
//                        TMA_A,
//                        TMA_B,
//                        TMA_OUT>(tma_a, tma_b, tma_out, &tma_residual);
// }

// template <typename T,
//           int BATCH_SIZE,
//           int OUTPUT_SIZE,
//           int REDUCTION_SIZE,
//           typename TMA_A,
//           typename TMA_B,
//           typename TMA_OUT,
//           int Kstages = 2>
// __global__
//     __launch_bounds__(256, 1) void linear_kernel_hopper_no_residual_wrapper(
//         const __grid_constant__ TMA_A tma_a,
//         const __grid_constant__ TMA_B tma_b,
//         const __grid_constant__ TMA_OUT tma_out) {

//   linear_kernel_hopper<T,
//                        BATCH_SIZE,
//                        OUTPUT_SIZE,
//                        REDUCTION_SIZE,
//                        Kstages,
//                        TMA_A,
//                        TMA_B,
//                        TMA_OUT,
//                        void>(tma_a, tma_b, tma_out, nullptr);
// }

// template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
// void launch_linear_hopper(void *input_ptr,
//                           void *weight_ptr,
//                           void *residual_ptr,
//                           void *output_ptr) {

//   constexpr int B = 3;
//   constexpr int M = 3;
//   constexpr int S = 3;

//   constexpr int TMA_CP_ASYNC_SIZE =
//       64; // note that if swizzle 128 is used, 64 is maximal cp size
//   constexpr int TILE_SIZE =
//       128; // we should modify this param if we want larger tile size
//   constexpr int TMA_CP_ASYNC_REPEAT_COL =
//       (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;

//   constexpr int OUTPUT_ATOM_SIZE = (OUTPUT_SIZE >= 256)   ? 256
//                                    : (OUTPUT_SIZE >= 128) ? 128
//                                    : (OUTPUT_SIZE >= 64)  ? 64
//                                    : (OUTPUT_SIZE >= 32)  ? 32
//                                                           : 16;
//   constexpr int OUTPUT_ATOM_REPEAT_COL =
//       (OUTPUT_ATOM_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;

//   constexpr int OUTPUT_TMA_CP_SIZE = OUTPUT_SIZE < 64 ? OUTPUT_SIZE : 64;
//   constexpr int SMEM_M_SIZE = BATCH_SIZE;
//   using TMA_A =
//       kernel::tma::tma_2d<bfloat16,
//                           B,
//                           M,
//                           S,
//                           BATCH_SIZE,                      /*GMEM_ROW_*/
//                           REDUCTION_SIZE,                  /*GMEM_COL_*/
//                           BATCH_SIZE,                      /*SMEM_ROW_*/
//                           TMA_CP_ASYNC_SIZE,               /*SMEM_COL_*/
//                           REDUCTION_SIZE, /*GMEM_STRIDE_ROW_*/ 1,
//                           /*GMEM_STRIDE_COL_*/ 1, /*SMEM_REPEAT_ROW_*/
//                           TMA_CP_ASYNC_REPEAT_COL, /*SMEM_REPEAT_COL_*/
//                           SMEM_M_SIZE * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
//                           true>;
//   using TMA_B =
//       kernel::tma::tma_2d<bfloat16,
//                           B,
//                           M,
//                           S,
//                           OUTPUT_SIZE,             /*GMEM_ROW_*/
//                           REDUCTION_SIZE,          /*GMEM_COL_*/
//                           OUTPUT_ATOM_SIZE,        /*SMEM_ROW_*/
//                           TMA_CP_ASYNC_SIZE,       /*SMEM_COL_*/
//                           REDUCTION_SIZE,          /*GMEM_STRIDE_ROW_*/
//                           1,                       /*GMEM_STRIDE_COL_*/
//                           1,                       /*SMEM_REPEAT_ROW_*/
//                           TMA_CP_ASYNC_REPEAT_COL, /*SMEM_REPEAT_COL_*/
//                           OUTPUT_ATOM_SIZE * TMA_CP_ASYNC_SIZE,
//                           /*SMEM_STRIDE_*/ true>;
//   using TMA_RESIDUAL = kernel::tma::tma_2d<bfloat16,
//                                            B,
//                                            M,
//                                            S,
//                                            BATCH_SIZE,
//                                            OUTPUT_SIZE,
//                                            BATCH_SIZE,
//                                            OUTPUT_TMA_CP_SIZE,
//                                            OUTPUT_SIZE,
//                                            1,
//                                            1,
//                                            OUTPUT_ATOM_REPEAT_COL,
//                                            SMEM_M_SIZE * TMA_CP_ASYNC_SIZE,
//                                            true>;

//   using TMA_OUT = kernel::tma::tma_2d<bfloat16,
//                                       B,
//                                       M,
//                                       S,
//                                       BATCH_SIZE,
//                                       OUTPUT_SIZE,
//                                       BATCH_SIZE,
//                                       OUTPUT_TMA_CP_SIZE,
//                                       OUTPUT_SIZE,
//                                       1,
//                                       1,
//                                       OUTPUT_ATOM_REPEAT_COL,
//                                       SMEM_M_SIZE * TMA_CP_ASYNC_SIZE,
//                                       true>;
//   TMA_A tma_a(input_ptr);
//   TMA_B tma_b(weight_ptr);
//   TMA_RESIDUAL tma_residual(residual_ptr);
//   TMA_OUT tma_out(output_ptr);

//   dim3 grid_dim(1, 1, 1);
//   dim3 block_dim(256, 1, 1);
//   size_t smem_size = 224 * 1024;

//   if (residual_ptr != nullptr) {
//     cudaFuncSetAttribute(linear_kernel_hopper_wrapper<T,
//                                                       BATCH_SIZE,
//                                                       OUTPUT_SIZE,
//                                                       REDUCTION_SIZE,
//                                                       TMA_A,
//                                                       TMA_B,
//                                                       TMA_RESIDUAL,
//                                                       TMA_OUT>,
//                          cudaFuncAttributeMaxDynamicSharedMemorySize,
//                          smem_size);

//   } else {

//     cudaFuncSetAttribute(
//         linear_kernel_hopper_no_residual_wrapper<T,
//                                                  BATCH_SIZE,
//                                                  OUTPUT_SIZE,
//                                                  REDUCTION_SIZE,
//                                                  TMA_A,
//                                                  TMA_B,
//                                                  TMA_OUT>,
//         cudaFuncAttributeMaxDynamicSharedMemorySize,
//         smem_size);
//   }

// #ifndef MIRAGE_PROFILE_HOPPER
//   if (residual_ptr != nullptr) {

//     linear_kernel_hopper_wrapper<T,
//                                  BATCH_SIZE,
//                                  OUTPUT_SIZE,
//                                  REDUCTION_SIZE,
//                                  TMA_A,
//                                  TMA_B,
//                                  TMA_RESIDUAL,
//                                  TMA_OUT><<<grid_dim, block_dim,
//                                  smem_size>>>(
//         tma_a, tma_b, tma_residual, tma_out);
//   } else {

//     linear_kernel_hopper_no_residual_wrapper<T,
//                                              BATCH_SIZE,
//                                              OUTPUT_SIZE,
//                                              REDUCTION_SIZE,
//                                              TMA_A,
//                                              TMA_B,
//                                              TMA_OUT>
//         <<<grid_dim, block_dim, smem_size>>>(tma_a, tma_b, tma_out);
//   }
// #else

//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);

//   constexpr int WARMUP_RUNS = 16;
//   constexpr int BENCHMARK_RUNS = 1000;

//   printf("=== Kernel Performance Profiling ===\n");

//   for (int i = 0; i < WARMUP_RUNS; i++) {
//     if (residual_ptr != nullptr) {
//       linear_kernel_hopper_wrapper<T,
//                                    BATCH_SIZE,
//                                    OUTPUT_SIZE,
//                                    REDUCTION_SIZE,
//                                    TMA_A,
//                                    TMA_B,
//                                    TMA_RESIDUAL,
//                                    TMA_OUT><<<grid_dim, block_dim,
//                                    smem_size>>>(
//           tma_a, tma_b, tma_residual, tma_out);
//     } else {
//       linear_kernel_hopper_no_residual_wrapper<T,
//                                                BATCH_SIZE,
//                                                OUTPUT_SIZE,
//                                                REDUCTION_SIZE,
//                                                TMA_A,
//                                                TMA_B,
//                                                TMA_OUT>
//           <<<grid_dim, block_dim, smem_size>>>(tma_a, tma_b, tma_out);
//     }
//   }
//   cudaDeviceSynchronize(); // Wait for all warmup runs to complete

//   printf("Running %d benchmark iterations...\n", BENCHMARK_RUNS);

//   float *iteration_times = new float[BENCHMARK_RUNS];
//   float total_time_ms = 0.0f;

//   for (int i = 0; i < BENCHMARK_RUNS; i++) {
//     cudaEventRecord(start);
//     if (residual_ptr != nullptr) {
//       linear_kernel_hopper_wrapper<T,
//                                    BATCH_SIZE,
//                                    OUTPUT_SIZE,
//                                    REDUCTION_SIZE,
//                                    TMA_A,
//                                    TMA_B,
//                                    TMA_RESIDUAL,
//                                    TMA_OUT><<<grid_dim, block_dim,
//                                    smem_size>>>(
//           tma_a, tma_b, tma_residual, tma_out);
//     } else {
//       linear_kernel_hopper_no_residual_wrapper<T,
//                                                BATCH_SIZE,
//                                                OUTPUT_SIZE,
//                                                REDUCTION_SIZE,
//                                                TMA_A,
//                                                TMA_B,
//                                                TMA_OUT>
//           <<<grid_dim, block_dim, smem_size>>>(tma_a, tma_b, tma_out);
//     }
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);

//     float iteration_time_ms;
//     cudaEventElapsedTime(&iteration_time_ms, start, stop);

//     total_time_ms += iteration_time_ms;
//   }

//   float avg_time_ms = total_time_ms / BENCHMARK_RUNS;

//   printf("\n=== Performance Results ===\n");
//   printf("Configuration:\n");
//   printf("  BATCH_SIZE=%d, OUTPUT_SIZE=%d, REDUCTION_SIZE=%d\n",
//          BATCH_SIZE,
//          OUTPUT_SIZE,
//          REDUCTION_SIZE);
//   printf(" TILE SIZE: %d\n", TILE_SIZE);
//   printf("  Average: %.3f ms\n", avg_time_ms);

//   printf("===============================\n");

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear", &linear_kernel, "Linear kernel");
    m.def("norm_linear", &norm_linear_kernel, "NormLinear kernel");
    m.def("multitoken_paged_attention",
          &multitoken_paged_attention_hopper,
          "Multitoken paged attention for Grace Hopper GPU");
  }
