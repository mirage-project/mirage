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
#include "blackwell/per_token_group_quantize_fp8.cuh"
#include "blackwell/task_header.cuh"
#include "hopper/tma_2d.cuh"
#include "runtime_header.h"
#include "tma.cuh"
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <cstdio>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <iostream>
#include <torch/extension.h>

// Cutlass includes
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/half.h> // F16 data type
#include <cutlass/util/print_error.hpp>

// CuTe includes
#include <cute/algorithm/cooperative_copy.hpp> // Auto vectorized copy operation
#include <cute/arch/cluster_sm90.hpp> // CuTe functions for querying the details of cluster launched
#include <cute/arch/tmem_allocator_sm100.hpp> // TMEM allocator for SM100
#include <cute/numeric/integral_constant.hpp> // Compile time in constants such as _1, _256 etc.
#include <cute/pointer_flagged.hpp>
#include <cute/tensor.hpp> // CuTe tensor implementation

using fp8_e4m3fn = __nv_fp8_e4m3;
using bfloat16 = type::bfloat16_t;

template <int HIDDEN_SIZE, int GROUP_SIZE>
__global__ __launch_bounds__(256) void quantize_fp8_sm100_kernel(
    void const *__restrict__ input_ptr,
    void *__restrict__ output_q_ptr,
    void *__restrict__ output_s_ptr,
    float eps,
    float min_8bit,
    float max_8bit) {
  int row = static_cast<int>(blockIdx.x);
  bfloat16 const *input =
      static_cast<bfloat16 const *>(input_ptr) + row * HIDDEN_SIZE;
  fp8_e4m3fn *output_q =
      static_cast<fp8_e4m3fn *>(output_q_ptr) + row * HIDDEN_SIZE;
  float *output_s =
      static_cast<float *>(output_s_ptr) + row * (HIDDEN_SIZE / GROUP_SIZE);

  kernel::per_token_group_quantize_fp8_task_impl</*BATCH_SIZE=*/1,
                                                 /*HIDDEN_SIZE=*/HIDDEN_SIZE,
                                                 /*GROUP_SIZE=*/GROUP_SIZE,
                                                 /*GLOBAL_STRIDE=*/HIDDEN_SIZE,
                                                 bfloat16,
                                                 fp8_e4m3fn,
                                                 /*SCALE_UE8M0=*/false>(
      input, output_q, output_s, eps, min_8bit, max_8bit);
}

#define DISPATCH_QUANTIZE_FP8_SM100_GROUP_SIZE_CASE(HIDDEN_SIZE, GROUP_SIZE)   \
  case GROUP_SIZE:                                                             \
    quantize_fp8_sm100_kernel<HIDDEN_SIZE, GROUP_SIZE>                         \
        <<<grid_dim, block_dim, 0>>>(                                          \
            input_ptr, output_q_ptr, output_s_ptr, kEps, kMin8, kMax8);        \
    break;

#define DISPATCH_QUANTIZE_FP8_SM100_GROUP_SIZE(HIDDEN_SIZE)                    \
  switch (group_size) {                                                        \
    DISPATCH_QUANTIZE_FP8_SM100_GROUP_SIZE_CASE(HIDDEN_SIZE, 128)              \
    default:                                                                   \
      TORCH_CHECK(false,                                                       \
                  "Unsupported group_size=",                                   \
                  group_size,                                                  \
                  " (must be one of {128})");                                  \
      break;                                                                   \
  }

#define DISPATCH_QUANTIZE_FP8_SM100_HIDDEN_SIZE_CASE(HIDDEN_SIZE)              \
  case HIDDEN_SIZE:                                                            \
    DISPATCH_QUANTIZE_FP8_SM100_GROUP_SIZE(HIDDEN_SIZE)                        \
    break;

// Quantize to FP8 and compute scale
void quantize_fp8_sm100(torch::Tensor input,
                        torch::Tensor output_q,
                        torch::Tensor output_s,
                        int group_size) {
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  TORCH_CHECK(output_q.sizes() == input.sizes(),
              "output_q must match input shape");
  TORCH_CHECK(output_s.dim() == 2, "output_s must be 2D (per-group scale)");
  TORCH_CHECK(output_s.size(0) == input.size(0),
              "output_s must have batch_size rows");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(output_q.is_contiguous(), "output_q must be contiguous");
  TORCH_CHECK(output_s.is_contiguous(), "output_s must be contiguous");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be bfloat16");
  TORCH_CHECK(output_q.scalar_type() == at::kFloat8_e4m3fn,
              "output_q must be float8_e4m3fn");
  TORCH_CHECK(output_s.scalar_type() == at::kFloat, "output_s must be float32");
  TORCH_CHECK(input.size(1) % group_size == 0,
              "hidden_size must be divisible by group_size");
  TORCH_CHECK(output_s.size(1) == input.size(1) / group_size,
              "output_s must have hidden_size/group_size columns");

  int const batch_size = static_cast<int>(input.size(0));
  int const hidden_size = static_cast<int>(input.size(1));

  void const *input_ptr = input.data_ptr();
  void *output_q_ptr = output_q.data_ptr();
  void *output_s_ptr = output_s.data_ptr();

  constexpr float kEps = 0.0f;
  constexpr float kMin8 = -448.0f;
  constexpr float kMax8 = 448.0f;

  dim3 grid_dim(batch_size, 1, 1);
  dim3 block_dim(256, 1, 1);

  switch (hidden_size) {
    DISPATCH_QUANTIZE_FP8_SM100_HIDDEN_SIZE_CASE(128)
    DISPATCH_QUANTIZE_FP8_SM100_HIDDEN_SIZE_CASE(256)
    DISPATCH_QUANTIZE_FP8_SM100_HIDDEN_SIZE_CASE(7168)
    default:
      TORCH_CHECK(false,
                  "Unsupported hidden_size=",
                  hidden_size,
                  " (must be one of {128, 256, 7168})");
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_fp8_sm100",
        &quantize_fp8_sm100,
        py::arg("input"),
        py::arg("output_q"),
        py::arg("output_s"),
        py::arg("group_size") = 128,
        "Quantize FP8 SM100");
}
