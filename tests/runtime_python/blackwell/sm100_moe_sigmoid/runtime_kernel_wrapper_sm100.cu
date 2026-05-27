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
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdio>
#include <iostream>

using bfloat16 = cute::bfloat16_t;

// topk_sigmoid_sm100 — DeepSeek V3 group-aware top-K-of-sigmoid routing.
// Fixed at 256 experts, 8 groups, top-4 groups, top-8 experts.
// VPT is template-parametric (only VPT=8 is instantiated; group-aware
// routing requires each thread's experts to belong to one group, which
// only holds at VPT=8 with the interleaved load layout).

constexpr int kSigmoidNumExperts = 256;
constexpr int kSigmoidNumGroups = 8;
constexpr int kSigmoidTopkGroup = 4;
constexpr int kSigmoidExpertsPerGroup = 32;
constexpr int kSigmoidTopkExperts = 8;
constexpr int kSigmoidBytesPerLdg = 16;
constexpr int kSigmoidWarpsPerCta = 8;

template <typename T, int VPT, bool FUSE_COMPACTION>
__global__ __launch_bounds__(kSigmoidWarpsPerCta * 32) void topk_sigmoid_kernel(
    void *__restrict__ gating_output,
    void *__restrict__ bias,
    void *__restrict__ topk_weights,
    void *__restrict__ mpk_routing_indices,
    void *__restrict__ mpk_active_expert_ids,
    int num_rows,
    float routed_scaling_factor) {
  kernel::topk_sigmoid_task_impl<T,
                                 VPT,
                                 kSigmoidNumExperts,
                                 kSigmoidWarpsPerCta,
                                 kSigmoidBytesPerLdg,
                                 kSigmoidNumGroups,
                                 kSigmoidTopkGroup,
                                 kSigmoidExpertsPerGroup,
                                 kSigmoidTopkExperts,
                                 FUSE_COMPACTION>(
      gating_output,
      bias,
      /*finished=*/nullptr,
      topk_weights,
      num_rows,
      mpk_routing_indices,
      mpk_active_expert_ids,
      /*start_expert=*/0,
      /*end_expert=*/kSigmoidNumExperts,
      routed_scaling_factor);
}

__global__ __launch_bounds__(256) void compact_active_experts_kernel(
    int *mpk_active_expert_ids) {
  kernel::compact_active_experts_impl<kSigmoidNumExperts>(
      mpk_active_expert_ids,
      /*start_expert=*/0,
      /*end_expert=*/kSigmoidNumExperts);
}

// Test-only entry point: force the fused (single-CTA) path regardless of
// BATCH_SIZE, so we can validate the outer-loop's multi-chunk correctness
// (the MPK-style invocation shape).
void topk_sigmoid_sm100_kernel_force_fused(
    torch::Tensor gating_output, torch::Tensor bias,
    torch::Tensor topk_weights, torch::Tensor mpk_routing_indices,
    torch::Tensor mpk_active_expert_ids, float routed_scaling_factor) {
  int const BATCH_SIZE = static_cast<int>(gating_output.size(0));
  auto stream = at::cuda::getCurrentCUDAStream();
  using T = bfloat16;
  constexpr int VPT = 8;
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(kSigmoidWarpsPerCta * 32, 1, 1);
  topk_sigmoid_kernel<T, VPT, /*FUSE_COMPACTION=*/true>
      <<<grid_dim, block_dim, 0, stream>>>(
          gating_output.data_ptr(), bias.data_ptr(),
          topk_weights.data_ptr(), mpk_routing_indices.data_ptr(),
          mpk_active_expert_ids.data_ptr<int>(),
          BATCH_SIZE, routed_scaling_factor);
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

  if (OUTPUT_SIZE != kSigmoidNumExperts || num_groups != kSigmoidNumGroups ||
      topk_group != kSigmoidTopkGroup) {
    printf("Unsupported configuration: num_experts=%d num_groups=%d "
           "topk_group=%d (only 256/8/4 supported)\n",
           OUTPUT_SIZE,
           num_groups,
           topk_group);
    return;
  }

  void *gating_output_ptr = gating_output.data_ptr();
  void *bias_ptr = bias.data_ptr();
  void *topk_weights_ptr = topk_weights.data_ptr();
  void *mpk_routing_indices_ptr = mpk_routing_indices.data_ptr();
  int *mpk_active_expert_ids_ptr =
      mpk_active_expert_ids.data_ptr<int>();

  // mpk_active_expert_ids init (markers to -1, counter to 0) is fused into
  // the main kernel's Phase 0.
  auto stream = at::cuda::getCurrentCUDAStream();

  using T = bfloat16;
  constexpr int VPT = 8;
  constexpr int kRowsPerWarp =
      (32 * VPT) / kSigmoidNumExperts;  // = 1 at VPT=8
  constexpr int kRowsPerCta = kSigmoidWarpsPerCta * kRowsPerWarp;  // = 8

  dim3 block_dim(kSigmoidWarpsPerCta * 32, 1, 1);

  if (BATCH_SIZE <= kRowsPerCta) {
    // Fused path: one CTA does scoring + Phase-7 compaction inline.
    // No separate compaction kernel launch needed.
    dim3 grid_dim(1, 1, 1);
    topk_sigmoid_kernel<T, VPT, /*FUSE_COMPACTION=*/true>
        <<<grid_dim, block_dim, 0, stream>>>(
            gating_output_ptr,
            bias_ptr,
            topk_weights_ptr,
            mpk_routing_indices_ptr,
            mpk_active_expert_ids_ptr,
            BATCH_SIZE,
            routed_scaling_factor);
  } else {
    // Multi-CTA path: scoring spread across SMs, compaction in a separate
    // kernel using kernel-launch ordering as the grid-wide barrier.
    int const grid_x = (BATCH_SIZE + kRowsPerCta - 1) / kRowsPerCta;
    dim3 grid_dim(grid_x, 1, 1);
    topk_sigmoid_kernel<T, VPT, /*FUSE_COMPACTION=*/false>
        <<<grid_dim, block_dim, 0, stream>>>(
            gating_output_ptr,
            bias_ptr,
            topk_weights_ptr,
            mpk_routing_indices_ptr,
            mpk_active_expert_ids_ptr,
            BATCH_SIZE,
            routed_scaling_factor);
    compact_active_experts_kernel<<<1, 256, 0, stream>>>(
        mpk_active_expert_ids_ptr);
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
  m.def("topk_sigmoid_sm100_force_fused",
        &topk_sigmoid_sm100_kernel_force_fused,
        "TopK Sigmoid (force single-CTA fused path; test/MPK shape)");
  m.def("topk_softmax_sm100",
        &topk_softmax_sm100_kernel,
        "TopK Softmax fused SM100 (for benchmark comparison)");
}
