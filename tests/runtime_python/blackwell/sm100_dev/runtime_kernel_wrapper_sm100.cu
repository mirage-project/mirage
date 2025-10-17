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

namespace {
template <typename T, int EXPERTS>
__global__ __launch_bounds__(256) void topk_softmax_kernel(
    const T* __restrict__ gating_output,
    float* __restrict__ topk_weights,
    int* __restrict__ topk_indices,
    int num_rows,
    int k,
    bool renormalize) {
  static constexpr std::size_t MAX_BYTES_PER_LDG = 16;
  static constexpr int BYTES_PER_LDG = (sizeof(T) * EXPERTS) < MAX_BYTES_PER_LDG ? (sizeof(T) * EXPERTS) : MAX_BYTES_PER_LDG;
  using C = kernel::detail::TopkConstants<T, EXPERTS, BYTES_PER_LDG>;
  static constexpr int VPT = C::VPT;
  static constexpr int WARPS_PER_TB = 8; // 256 threads
  kernel::topkGatingSoftmaxFused_device<T, VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG>(
      gating_output, /*finished*/ nullptr, topk_weights, num_rows, topk_indices, k, 0, EXPERTS, renormalize);
}
} // namespace

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
  // launch grid using 256-thread blocks
  auto launch = [&](auto experts_ct) {
    constexpr int EXP = decltype(experts_ct)::value;
    using T = float;
    using C = kernel::detail::TopkConstants<T, EXP, ((sizeof(T)*EXP)<16?(sizeof(T)*EXP):16)>;
    constexpr int ROWS_PER_WARP = C::ROWS_PER_WARP;
    constexpr int WARPS_PER_TB = 8;
    int num_warps = (BATCH_SIZE + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;
    dim3 block_dim(32, WARPS_PER_TB);
    topk_softmax_kernel<T, EXP><<<num_blocks, block_dim, 0, stream>>>(
        gating_output_f.data_ptr<float>(),
        topk_weights.data_ptr<float>(),
        topk_indices.data_ptr<int>(),
        BATCH_SIZE,
        NUM_TOPK,
        /*renormalize=*/true);
  };

  switch (OUTPUT_SIZE) {
    case 1: launch(std::integral_constant<int,1>{}); break;
    case 2: launch(std::integral_constant<int,2>{}); break;
    case 4: launch(std::integral_constant<int,4>{}); break;
    case 8: launch(std::integral_constant<int,8>{}); break;
    case 16: launch(std::integral_constant<int,16>{}); break;
    case 32: launch(std::integral_constant<int,32>{}); break;
    case 64: launch(std::integral_constant<int,64>{}); break;
    case 128: launch(std::integral_constant<int,128>{}); break;
    case 256: launch(std::integral_constant<int,256>{}); break;
    default:
      printf("Unsupported num_experts=%d (must be power-of-two <= 256)\n", OUTPUT_SIZE);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}