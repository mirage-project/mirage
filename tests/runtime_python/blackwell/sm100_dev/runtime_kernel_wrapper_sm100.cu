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
#include "runtime_header.h"
#include "blackwell/task_header.cuh"
#include "hopper/tma_2d.cuh"
#include "tma.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <iostream>
#include <cstdio>

// Cutlass includes
#include <cutlass/half.h>                       // F16 data type
#include <cutlass/util/print_error.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

// CuTe includes
#include <cute/tensor.hpp>                      // CuTe tensor implementation
#include <cute/arch/cluster_sm90.hpp>           // CuTe functions for querying the details of cluster launched
#include <cute/numeric/integral_constant.hpp>   // Compile time in constants such as _1, _256 etc.
#include <cute/algorithm/cooperative_copy.hpp>  // Auto vectorized copy operation
#include <cute/arch/tmem_allocator_sm100.hpp>   // TMEM allocator for SM100
#include <cute/pointer_flagged.hpp> 

using bfloat16 = cute::bfloat16_t;

// Fused TopK softmax for SM100
#include "blackwell/topk_softmax_sm100.cuh"

void topk_softmax_sm100_kernel(torch::Tensor gating_output,
                               torch::Tensor topk_indices,
                               torch::Tensor topk_weights);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_softmax_sm100", &topk_softmax_sm100_kernel, "TopK Softmax fused SM100");
}

// New: expose a direct fused TopK softmax without GEMM
void topk_softmax_sm100_kernel(torch::Tensor gating_output,
                               torch::Tensor topk_indices,
                               torch::Tensor topk_weights) {

  const int BATCH_SIZE = static_cast<int>(gating_output.size(0));
  const int OUTPUT_SIZE = static_cast<int>(gating_output.size(1));
  const int NUM_TOPK = static_cast<int>(topk_indices.size(1));

  assert(topk_indices.size(0) == BATCH_SIZE && topk_indices.size(1) == NUM_TOPK);
  assert(topk_weights.size(0) == BATCH_SIZE && topk_weights.size(1) == NUM_TOPK);

  // Ensure float input to fused kernel
  auto gating_output_f = gating_output.to(at::kFloat);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  kernel::topk_softmax_fused_sm100<float>(
      gating_output_f.data_ptr<float>(),
      topk_weights.data_ptr<float>(),
      topk_indices.data_ptr<int>(),
      BATCH_SIZE,
      OUTPUT_SIZE,
      NUM_TOPK,
      /*renormalize=*/true,
      stream);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}