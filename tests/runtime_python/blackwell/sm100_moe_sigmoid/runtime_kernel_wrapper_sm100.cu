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
#include "blackwell/topk_sigmoid_sm100.cuh"
#include "blackwell/topk_softmax_sm100.cuh" // for TopkConstants
#include "runtime_header.h"
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdio>
#include <iostream>

using bfloat16 = cute::bfloat16_t;

// topk_sigmoid_sm100

template <typename T,
          int EXPERTS,
          int BYTES_PER_LDG,
          int NUM_GROUPS,
          int TOPK_GROUP,
          int EXPERTS_PER_GROUP,
          int TOPK_EXPERTS>
__global__ __launch_bounds__(256) void topk_sigmoid_kernel(
    void *__restrict__ gating_output,
    void *__restrict__ bias,
    void *__restrict__ topk_weights,
    void *__restrict__ mpk_routing_indices,
    void *__restrict__ mpk_active_expert_ids,
    int num_rows,
    float routed_scaling_factor) {
  using C = kernel::detail::TopkConstants<T, EXPERTS, BYTES_PER_LDG>;
  static constexpr int VPT = C::VPT;
  static constexpr int WARPS_PER_TB = 8;
  kernel::topk_sigmoid_task_impl<T,
                                  VPT,
                                  EXPERTS,
                                  WARPS_PER_TB,
                                  BYTES_PER_LDG,
                                  NUM_GROUPS,
                                  TOPK_GROUP,
                                  EXPERTS_PER_GROUP,
                                  TOPK_EXPERTS>(
      gating_output,
      bias,
      /*finished=*/nullptr,
      topk_weights,
      num_rows,
      mpk_routing_indices,
      mpk_active_expert_ids,
      /*start_expert=*/0,
      /*end_expert=*/EXPERTS,
      routed_scaling_factor);
  __syncthreads();
}

void topk_sigmoid_sm100_kernel(torch::Tensor gating_output,
                               torch::Tensor bias,
                               torch::Tensor topk_weights,
                               torch::Tensor mpk_routing_indices,
                               torch::Tensor mpk_active_expert_ids,
                               float routed_scaling_factor,
                               int num_groups,
                               int topk_group) {

  int const BATCH_SIZE = static_cast<int>(gating_output.size(0));
  int const OUTPUT_SIZE = static_cast<int>(gating_output.size(1));
  int const NUM_TOPK = static_cast<int>(topk_weights.size(1));

  assert(topk_weights.size(0) == BATCH_SIZE &&
         topk_weights.size(1) == NUM_TOPK);
  assert(mpk_routing_indices.size(0) == OUTPUT_SIZE &&
         mpk_routing_indices.size(1) == BATCH_SIZE);
  assert(mpk_active_expert_ids.size(0) == OUTPUT_SIZE + 1);
  assert(bias.size(0) == OUTPUT_SIZE);

  void *gating_output_ptr = gating_output.data_ptr();
  void *bias_ptr = bias.data_ptr();
  void *topk_weights_ptr = topk_weights.data_ptr();
  void *mpk_routing_indices_ptr = mpk_routing_indices.data_ptr();
  void *mpk_active_expert_ids_ptr = mpk_active_expert_ids.data_ptr();

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(256, 1, 1);

  // DeepSeek V3: 256 experts, 8 groups of 32, top-4 groups, top-8 experts
  if (OUTPUT_SIZE == 256 && num_groups == 8 && topk_group == 4) {
    using T = bfloat16;
    constexpr int EXP = 256;
    constexpr int BPL = 16;
    topk_sigmoid_kernel<T, EXP, BPL, 8, 4, 32, 8>
        <<<grid_dim, block_dim, 0>>>(gating_output_ptr,
                                     bias_ptr,
                                     topk_weights_ptr,
                                     mpk_routing_indices_ptr,
                                     mpk_active_expert_ids_ptr,
                                     BATCH_SIZE,
                                     routed_scaling_factor);
  } else {
    printf("Unsupported configuration: num_experts=%d num_groups=%d topk_group=%d\n",
           OUTPUT_SIZE, num_groups, topk_group);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

// topk_softmax_sm100 (for benchmark comparison at same config)

template <typename T, int EXPERTS, int BYTES_PER_LDG>
__global__ __launch_bounds__(256) void topk_softmax_kernel(
    void *__restrict__ gating_output,
    void *__restrict__ topk_weights,
    void *__restrict__ mpk_routing_indices,
    void *__restrict__ mpk_active_expert_ids,
    int num_rows,
    int k,
    bool renormalize) {
  using C = kernel::detail::TopkConstants<T, EXPERTS, BYTES_PER_LDG>;
  static constexpr int VPT = C::VPT;
  static constexpr int WARPS_PER_TB = 8;
  kernel::topk_softmax_task_impl<T, VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG>(
      gating_output,
      /*finished*/ nullptr,
      topk_weights,
      num_rows,
      k,
      mpk_routing_indices,
      mpk_active_expert_ids,
      /*start_expert=*/0,
      /*end_expert=*/EXPERTS,
      renormalize);
  __syncthreads();
}

void topk_softmax_sm100_kernel(torch::Tensor gating_output,
                               torch::Tensor topk_weights,
                               torch::Tensor mpk_routing_indices,
                               torch::Tensor mpk_active_expert_ids) {

  int const BATCH_SIZE = static_cast<int>(gating_output.size(0));
  int const OUTPUT_SIZE = static_cast<int>(gating_output.size(1));
  int const NUM_TOPK = static_cast<int>(topk_weights.size(1));

  assert(topk_weights.size(0) == BATCH_SIZE &&
         topk_weights.size(1) == NUM_TOPK);
  assert(mpk_routing_indices.size(0) == OUTPUT_SIZE &&
         mpk_routing_indices.size(1) == BATCH_SIZE);
  assert(mpk_active_expert_ids.size(0) == OUTPUT_SIZE + 1);

  void *gating_output_ptr = gating_output.data_ptr();
  void *topk_weights_ptr = topk_weights.data_ptr();
  void *mpk_routing_indices_ptr = mpk_routing_indices.data_ptr();
  void *mpk_active_expert_ids_ptr = mpk_active_expert_ids.data_ptr();

  auto launch = [&](auto experts_ct) {
    constexpr int EXP = decltype(experts_ct)::value;
    using T = bfloat16;
    dim3 grid_dim(1, 1, 1);
    dim3 block_dim(256, 1, 1);
    topk_softmax_kernel<T,
                        EXP,
                        ((sizeof(T) * EXP) < 16 ? (sizeof(T) * EXP) : 16)>
        <<<grid_dim, block_dim, 0>>>(gating_output_ptr,
                                     topk_weights_ptr,
                                     mpk_routing_indices_ptr,
                                     mpk_active_expert_ids_ptr,
                                     BATCH_SIZE,
                                     NUM_TOPK,
                                     /*renormalize=*/true);
  };

  switch (OUTPUT_SIZE) {
    case 128:
      launch(std::integral_constant<int, 128>{});
      break;
    case 256:
      launch(std::integral_constant<int, 256>{});
      break;
    default:
      printf("Unsupported num_experts=%d\n", OUTPUT_SIZE);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_sigmoid_sm100",
        &topk_sigmoid_sm100_kernel,
        "TopK Sigmoid group-aware fused SM100");
  m.def("topk_softmax_sm100",
        &topk_softmax_sm100_kernel,
        "TopK Softmax fused SM100 (for benchmark comparison)");
}
