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
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "epilogue.cuh"
#include "gemm_ws.cuh"
#include "gemm_ws_mpk.cuh"
#include "kernel_traits.cuh"
#include "mma_tma_ws_mainloop.cuh"

#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../common/bfloat16.h"
using bfloat16 = type::bfloat16_t;

// template <typename CollectiveMainloop, typename CollectiveEpilogue>
// __global__ __launch_bounds__(256, 1) void linear_kernel_hopper_cute_wrapper(
//     CUTE_GRID_CONSTANT
//     typename CollectiveMainloop::template Params<true> const mainloop_params,
//     CUTE_GRID_CONSTANT
//     typename CollectiveEpilogue::Params const epilogue_params) {
//   kernel::gemm_kernel_tma_warp_specialized<CollectiveMainloop,
//                                            CollectiveEpilogue>(mainloop_params,
//                                                                epilogue_params);
// }

// template <typename T, int OUTPUT_SIZE, int BATCH_SIZE, int REDUCTION_SIZE>
// void launch_linear_hopper_cute(void *weight_ptr,
//                                void *input_ptr,
//                                void *residual_ptr,
//                                void *output_ptr) {

//   using namespace cute;
//   auto problem_shape =
//       Shape<Int<OUTPUT_SIZE>, Int<BATCH_SIZE>, Int<REDUCTION_SIZE>>{};

//   using KernelTraits =
//       kernel::MMAKernelTraits<T,
//                               OUTPUT_SIZE,
//                               BATCH_SIZE,
//                               REDUCTION_SIZE,
//                               cutlass::layout::RowMajor,    // GmemLayoutATag
//                               cutlass::layout::ColumnMajor, // GmemLayoutBTag
//                               cutlass::layout::RowMajor,    // GmemLayoutCTag
//                               cutlass::layout::RowMajor,    // GmemLayoutDTag
//                               8,                            // NUM_WARPS
//                               64,                           // M
//                               BATCH_SIZE,                   // N
//                               64,                           // K
//                               decltype(problem_shape),
//                               OUTPUT_SIZE, // O_STRIDE
//                               4>;          // NUM_STAGES

//   using Mainloop = kernel::CollectiveMainloop<KernelTraits>;
//   using Epilogue = kernel::CollectiveEpilogue<KernelTraits>;

//   using StrideA = typename KernelTraits::StrideA;
//   using StrideB = typename KernelTraits::StrideB;
//   using StrideC = typename KernelTraits::StrideC;
//   //   using StrideD = typename KernelTraits::StrideD;

//   StrideA stride_A = cutlass::make_cute_packed_stride(
//       StrideA{}, {KernelTraits::OUTPUT_SIZE, KernelTraits::REDUCTION_SIZE,
//       1});
//   StrideB stride_B = cutlass::make_cute_packed_stride(
//       StrideB{}, {KernelTraits::BATCH_SIZE, KernelTraits::REDUCTION_SIZE,
//       1});
//   StrideC stride_C = cutlass::make_cute_packed_stride(
//       StrideC{}, {KernelTraits::BATCH_SIZE, KernelTraits::OUTPUT_SIZE, 1});
//   //   StrideD stride_D = cutlass::make_cute_packed_stride(
//   //       StrideD{}, {KernelTraits::M, KernelTraits::N, 1});

//   typename Mainloop::Arguments mainloop_args{
//       static_cast<T const *>(weight_ptr), // ptr_A
//       stride_A,                           // dA
//       static_cast<T const *>(input_ptr),  // ptr_B
//       stride_B,                           // dB
//   };

//   typename Epilogue::Arguments epilogue_args{
//       static_cast<T const *>(residual_ptr), // ptr_C
//       stride_C,                             // dC
//       static_cast<T *>(output_ptr),         // ptr_D
//       stride_C,                             // dD
//       {1.0f, 1.0f}                          // alpha and beta
//   };

//   typename Mainloop::template Params<true> mainloop_params =
//       Mainloop::template to_underlying_arguments<true>(problem_shape,
//                                                        mainloop_args);
//   typename Epilogue::Params epilogue_params =
//       Epilogue::to_underlying_arguments(problem_shape, epilogue_args);

//   dim3 grid(1);
//   dim3 block(256);

//   size_t shared_mem_size = 100000;
//   cudaFuncSetAttribute(linear_kernel_hopper_cute_wrapper<Mainloop, Epilogue>,
//                        cudaFuncAttributeMaxDynamicSharedMemorySize,
//                        shared_mem_size);
//   // linear_kernel_hopper_cute_wrapper<Mainloop, Epilogue>
//   //     <<<grid, block, shared_mem_size>>>(mainloop_params,
//   epilogue_params);

//   cudaEvent_t start, stop;
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);

//   constexpr int WARMUP_RUNS = 16;
//   constexpr int BENCHMARK_RUNS = 1000;

//   printf("=== Kernel Performance Profiling ===\n");

//   for (int i = 0; i < WARMUP_RUNS; i++) {
//     linear_kernel_hopper_cute_wrapper<Mainloop, Epilogue>
//         <<<grid, block, shared_mem_size>>>(mainloop_params, epilogue_params);
//   }
//   cudaDeviceSynchronize(); // Wait for all warmup runs to complete

//   printf("Running %d benchmark iterations...\n", BENCHMARK_RUNS);

//   float *iteration_times = new float[BENCHMARK_RUNS];
//   float total_time_ms = 0.0f;

//   for (int i = 0; i < BENCHMARK_RUNS; i++) {
//     cudaEventRecord(start);
//     linear_kernel_hopper_cute_wrapper<Mainloop, Epilogue>
//         <<<grid, block, shared_mem_size>>>(mainloop_params, epilogue_params);
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
//   printf("  Average: %.3f ms\n", avg_time_ms);

//   printf("===============================\n");

//   delete[] iteration_times;
//   cudaEventDestroy(start);
//   cudaEventDestroy(stop);
// }

// #define DISPATCH_LINEAR_CUTE_REDUCTION_SIZE_CASE(                              \
//     OUTPUT_SIZE, BATCH_SIZE, REDUCTION_SIZE)                                   \
//   case REDUCTION_SIZE:                                                         \
//     launch_linear_hopper_cute<cutlass::bfloat16_t,                             \
//                               OUTPUT_SIZE,                                     \
//                               BATCH_SIZE,                                      \
//                               REDUCTION_SIZE>(                                 \
//         weight_ptr, input_ptr, residual_ptr, output_ptr);                      \
//     break;

// #define DISPATCH_LINEAR_CUTE_REDUCTION_SIZE(OUTPUT_SIZE, BATCH_SIZE)           \
//   switch (weight.size(1)) {                                                    \
//     DISPATCH_LINEAR_CUTE_REDUCTION_SIZE_CASE(OUTPUT_SIZE, BATCH_SIZE, 4096)    \
//     /*                                                                         \
//     DISPATCH_LINEAR_CUTE_REDUCTION_SIZE_CASE(OUTPUT_SIZE, BATCH_SIZE, 12288)   \
//     */                                                                         \
//     default:                                                                   \
//       printf("Unsupported reduction size in test: %zu\n", input.size(1));      \
//       break;                                                                   \
//   }

// #define DISPATCH_LINEAR_CUTE_BATCH_SIZE_CASE(OUTPUT_SIZE, BATCH_SIZE)          \
//   case BATCH_SIZE:                                                             \
//     DISPATCH_LINEAR_CUTE_REDUCTION_SIZE(OUTPUT_SIZE, BATCH_SIZE)               \
//     break;

// #define DISPATCH_LINEAR_CUTE_BATCH_SIZE(OUTPUT_SIZE)                           \
//   switch (input.size(0)) {                                                     \
//     DISPATCH_LINEAR_CUTE_BATCH_SIZE_CASE(OUTPUT_SIZE, 8)                       \
//     DISPATCH_LINEAR_CUTE_BATCH_SIZE_CASE(OUTPUT_SIZE, 16)                      \
//     /*                                                                         \
//     DISPATCH_LINEAR_CUTE_BATCH_SIZE_CASE(OUTPUT_SIZE, 32)                      \
//     DISPATCH_LINEAR_CUTE_BATCH_SIZE_CASE(OUTPUT_SIZE, 64)                      \
//     */                                                                         \
//     default:                                                                   \
//       printf("Unsupported batch size in test: %zu\n", input.size(1));          \
//       break;                                                                   \
//   }

// #define DISPATCH_LINEAR_CUTE_OUTPUT_SIZE_CASE(OUTPUT_SIZE)                     \
//   case OUTPUT_SIZE:                                                            \
//     DISPATCH_LINEAR_CUTE_BATCH_SIZE(OUTPUT_SIZE)                               \
//     break;

// void linear_kernel(torch::Tensor weight,
//                    torch::Tensor input,
//                    torch::Tensor residual,
//                    torch::Tensor output) {

//   void *input_ptr = input.data_ptr();
//   void *weight_ptr = weight.data_ptr();
//   void *residual_ptr = residual.data_ptr();
//   void *output_ptr = output.data_ptr();

//   switch (weight.size(0)) {
//     DISPATCH_LINEAR_CUTE_OUTPUT_SIZE_CASE(64)
//     DISPATCH_LINEAR_CUTE_OUTPUT_SIZE_CASE(128)
//     default:
//       printf("Unsupported output size in test: %zu\n", weight.size(0));
//       break;
//   }

//   cudaError_t err = cudaDeviceSynchronize();
//   if (err != cudaSuccess) {
//     printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
//   }
// }

template <typename CollectiveMainloop,
          typename CollectiveEpilogue,
          typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          typename TMA_A,
          typename TMA_B>
__global__ __launch_bounds__(256, 1) void linear_kernel_hopper_cute_mpk_wrapper(
    const __grid_constant__ TMA_A tma_a,
    const __grid_constant__ TMA_B tma_b,
    void *output_ptr,
    void const *residual_ptr) {
  kernel::linear_cutlass_ws_hopper<CollectiveMainloop,
                                   CollectiveEpilogue,
                                   false,
                                   T,
                                   BATCH_SIZE,
                                   OUTPUT_SIZE,
                                   REDUCTION_SIZE,
                                   TMA_A,
                                   TMA_B,
                                   OUTPUT_SIZE,
                                   true>(
      tma_a, tma_b, output_ptr, residual_ptr);
}

template <typename T, int OUTPUT_SIZE, int BATCH_SIZE, int REDUCTION_SIZE>
void launch_linear_hopper_cute_mpk(void *weight_ptr,
                                   void *input_ptr,
                                   void *residual_ptr,
                                   void *output_ptr) {

  using namespace cute;
  auto problem_shape =
      Shape<Int<OUTPUT_SIZE>, Int<BATCH_SIZE>, Int<REDUCTION_SIZE>>{};

  constexpr int TILE_SIZE =
      128; // we should modify this param if we want larger tile size

  using KernelTraits =
      kernel::MMAKernelTraits<T,
                              OUTPUT_SIZE,
                              BATCH_SIZE,
                              REDUCTION_SIZE,
                              cutlass::layout::RowMajor,    // GmemLayoutATag
                              cutlass::layout::ColumnMajor, // GmemLayoutBTag
                              cutlass::layout::RowMajor,    // GmemLayoutCTag
                              cutlass::layout::RowMajor,    // GmemLayoutDTag
                              8,                            // NUM_WARPS
                              64,                           // M
                              BATCH_SIZE,                   // N
                              TILE_SIZE,                    // K
                              decltype(problem_shape),
                              OUTPUT_SIZE, // O_STRIDE
                              4>;          // NUM_STAGES

  using Mainloop = kernel::CollectiveMainloop<KernelTraits>;
  using Epilogue = kernel::CollectiveEpilogue<KernelTraits>;

  using StrideA = typename KernelTraits::StrideA;
  using StrideB = typename KernelTraits::StrideB;
  using StrideC = typename KernelTraits::StrideC;
  //   using StrideD = typename KernelTraits::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(
      StrideA{}, {KernelTraits::OUTPUT_SIZE, KernelTraits::REDUCTION_SIZE, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(
      StrideB{}, {KernelTraits::BATCH_SIZE, KernelTraits::REDUCTION_SIZE, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(
      StrideC{}, {KernelTraits::BATCH_SIZE, KernelTraits::OUTPUT_SIZE, 1});
  //   StrideD stride_D = cutlass::make_cute_packed_stride(
  //       StrideD{}, {KernelTraits::M, KernelTraits::N, 1});

  typename Mainloop::Arguments mainloop_args{
      static_cast<T const *>(weight_ptr), // ptr_A
      stride_A,                           // dA
      static_cast<T const *>(input_ptr),  // ptr_B
      stride_B,                           // dB
  };

  typename Epilogue::Arguments epilogue_args{
      static_cast<T const *>(residual_ptr), // ptr_C
      stride_C,                             // dC
      static_cast<T *>(output_ptr),         // ptr_D
      stride_C,                             // dD
      {1.0f, 1.0f}                          // alpha and beta
  };

  // typename Mainloop::Params mainloop_params =
  // Mainloop::template to_underlying_arguments</*onHost=*/true>(problem_shape,
  // mainloop_args);
  using MainloopParamsHost = typename Mainloop::template Params<false>;
  MainloopParamsHost mainloop_params =
      Mainloop::template to_underlying_arguments<false>(problem_shape,
                                                        mainloop_args);
  typename Epilogue::Params epilogue_params =
      Epilogue::to_underlying_arguments(problem_shape, epilogue_args);

  constexpr int B = 3;
  constexpr int M = 3;
  constexpr int S = 3;

  constexpr int TMA_CP_ASYNC_SIZE =
      64; // note that if swizzle 128 is used, 64 is maximal cp size

  constexpr int TMA_CP_ASYNC_REPEAT_COL =
      (TILE_SIZE + TMA_CP_ASYNC_SIZE - 1) / TMA_CP_ASYNC_SIZE;

  constexpr int OUTPUT_ATOM_SIZE = 64; // this is padded
  constexpr int SMEM_M_SIZE = BATCH_SIZE;
  using TMA_B =
      kernel::tma::tma_2d<T,
                          B,
                          M,
                          S,
                          SMEM_M_SIZE,                     /*GMEM_ROW_*/
                          REDUCTION_SIZE,                  /*GMEM_COL_*/
                          SMEM_M_SIZE,                     /*SMEM_ROW_*/
                          TMA_CP_ASYNC_SIZE,               /*SMEM_COL_*/
                          REDUCTION_SIZE,                  /*GMEM_STRIDE_ROW_*/
                          1,                               /*GMEM_STRIDE_COL_*/
                          1,                               /*SMEM_REPEAT_ROW_*/
                          TMA_CP_ASYNC_REPEAT_COL,         /*SMEM_REPEAT_COL_*/
                          SMEM_M_SIZE * TMA_CP_ASYNC_SIZE, /*SMEM_STRIDE_*/
                          true>;
  using TMA_A =
      kernel::tma::tma_2d<T,
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

  TMA_A tma_a(weight_ptr);
  TMA_B tma_b(input_ptr);

  dim3 grid(1);
  dim3 block(256);

  size_t shared_mem_size = 100000;
  cudaFuncSetAttribute(
      linear_kernel_hopper_cute_mpk_wrapper<Mainloop,
                                            Epilogue,
                                            cutlass::bfloat16_t,
                                            BATCH_SIZE,
                                            OUTPUT_SIZE,
                                            REDUCTION_SIZE,
                                            TMA_A,
                                            TMA_B>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_mem_size);
  // linear_kernel_hopper_cute_wrapper<Mainloop, Epilogue>
  //     <<<grid, block, shared_mem_size>>>(mainloop_params, epilogue_params);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  constexpr int WARMUP_RUNS = 0;
  constexpr int BENCHMARK_RUNS = 1;

  printf("=== Kernel Performance Profiling ===\n");

  for (int i = 0; i < WARMUP_RUNS; i++) {
    linear_kernel_hopper_cute_mpk_wrapper<Mainloop,
                                          Epilogue,
                                          T,
                                          BATCH_SIZE,
                                          OUTPUT_SIZE,
                                          REDUCTION_SIZE,
                                          TMA_A,
                                          TMA_B>
        <<<grid, block, shared_mem_size>>>(
            tma_a, tma_b, output_ptr, residual_ptr);
  }
  cudaDeviceSynchronize(); // Wait for all warmup runs to complete

  printf("Running %d benchmark iterations...\n", BENCHMARK_RUNS);

  float *iteration_times = new float[BENCHMARK_RUNS];
  float total_time_ms = 0.0f;

  for (int i = 0; i < BENCHMARK_RUNS; i++) {
    cudaEventRecord(start);
    linear_kernel_hopper_cute_mpk_wrapper<Mainloop,
                                          Epilogue,
                                          T,
                                          BATCH_SIZE,
                                          OUTPUT_SIZE,
                                          REDUCTION_SIZE,
                                          TMA_A,
                                          TMA_B>
        <<<grid, block, shared_mem_size>>>(
            tma_a, tma_b, output_ptr, residual_ptr);
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
  printf("  Average: %.3f ms\n", avg_time_ms);

  printf("===============================\n");

  delete[] iteration_times;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

#define DISPATCH_LINEAR_CUTE_MPK_REDUCTION_SIZE_CASE(                          \
    OUTPUT_SIZE, BATCH_SIZE, REDUCTION_SIZE)                                   \
  case REDUCTION_SIZE:                                                         \
    launch_linear_hopper_cute_mpk<cutlass::bfloat16_t,                         \
                                  OUTPUT_SIZE,                                 \
                                  BATCH_SIZE,                                  \
                                  REDUCTION_SIZE>(                             \
        weight_ptr, input_ptr, residual_ptr, output_ptr);                      \
    break;

#define DISPATCH_LINEAR_CUTE_MPK_REDUCTION_SIZE(OUTPUT_SIZE, BATCH_SIZE)       \
  switch (weight.size(1)) {                                                    \
    DISPATCH_LINEAR_CUTE_MPK_REDUCTION_SIZE_CASE(                              \
        OUTPUT_SIZE, BATCH_SIZE, 4096)                                         \
    /*                                                                         \
    DISPATCH_LINEAR_CUTE_MPK_REDUCTION_SIZE_CASE(OUTPUT_SIZE, BATCH_SIZE,      \
    12288)                                                                     \
    */                                                                         \
    default:                                                                   \
      printf("Unsupported reduction size in test: %zu\n", input.size(1));      \
      break;                                                                   \
  }

#define DISPATCH_LINEAR_CUTE_MPK_BATCH_SIZE_CASE(OUTPUT_SIZE, BATCH_SIZE)      \
  case BATCH_SIZE:                                                             \
    DISPATCH_LINEAR_CUTE_MPK_REDUCTION_SIZE(OUTPUT_SIZE, BATCH_SIZE)           \
    break;

#define DISPATCH_LINEAR_CUTE_MPK_BATCH_SIZE(OUTPUT_SIZE)                       \
  switch (input.size(0)) {                                                     \
    DISPATCH_LINEAR_CUTE_MPK_BATCH_SIZE_CASE(OUTPUT_SIZE, 8)                   \
    DISPATCH_LINEAR_CUTE_MPK_BATCH_SIZE_CASE(OUTPUT_SIZE, 16)                  \
    /*                                                                         \
    DISPATCH_LINEAR_CUTE_MPK_BATCH_SIZE_CASE(OUTPUT_SIZE, 32)                  \
    DISPATCH_LINEAR_CUTE_MPK_BATCH_SIZE_CASE(OUTPUT_SIZE, 64)                  \
    */                                                                         \
    default:                                                                   \
      printf("Unsupported batch size in test: %zu\n", input.size(1));          \
      break;                                                                   \
  }

#define DISPATCH_LINEAR_CUTE_MPK_OUTPUT_SIZE_CASE(OUTPUT_SIZE)                 \
  case OUTPUT_SIZE:                                                            \
    DISPATCH_LINEAR_CUTE_MPK_BATCH_SIZE(OUTPUT_SIZE)                           \
    break;

void linear_mpk_kernel(torch::Tensor weight,
                       torch::Tensor input,
                       torch::Tensor residual,
                       torch::Tensor output) {

  void *input_ptr = input.data_ptr();
  void *weight_ptr = weight.data_ptr();
  void *residual_ptr = residual.data_ptr();
  void *output_ptr = output.data_ptr();

  switch (weight.size(0)) {
    DISPATCH_LINEAR_CUTE_MPK_OUTPUT_SIZE_CASE(64)
    default:
      printf("Unsupported output size in test: %zu\n", weight.size(0));
      break;
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("linear", &linear_kernel, "Linear kernel");
  m.def("linear_mpk", &linear_mpk_kernel, "Linear mpk kernel");
}