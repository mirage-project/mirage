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
#include "common/moe_routing_distributed.cuh"
#include "common/all_to_all_combine_task.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cstdio>

using bfloat16 = cute::bfloat16_t;

// ---------------------------------------------------------------------------
// EP MoE Routing: single-GPU test (WORLD_SIZE=1)
// ---------------------------------------------------------------------------
void moe_routing(torch::Tensor router_logits,
                 torch::Tensor routing_indices,
                 torch::Tensor routing_weights,
                 torch::Tensor dispatch_counts,
                 int num_experts,
                 int experts_per_rank,
                 int rank) {
  int batch_size = router_logits.size(0);
  assert(router_logits.size(1) == num_experts);

  auto *logits_ptr = reinterpret_cast<bfloat16 const *>(
      router_logits.data_ptr());
  int *idx_ptr = routing_indices.data_ptr<int>();
  auto *wts_ptr = reinterpret_cast<bfloat16 *>(routing_weights.data_ptr());
  int *dcounts_ptr = dispatch_counts.data_ptr<int>();

  cudaMemset(dcounts_ptr, 0, dispatch_counts.numel() * sizeof(int));

  int topk = routing_indices.size(1);

  // One block per token, topk threads per block.
  // Use the __global__ test wrapper.
  if (num_experts == 8 && topk == 2) {
    mirage::kernel::moe_routing_distributed_task_impl<
        bfloat16, /*BATCH_SIZE=*/8, /*NUM_EXPERTS=*/8, /*TOPK=*/2,
        /*WORLD_SIZE=*/1, /*NORMALIZE=*/true>
        <<<batch_size, topk>>>(logits_ptr, idx_ptr, wts_ptr, dcounts_ptr,
                               nullptr, experts_per_rank, rank, 1.0f);
  } else if (num_experts == 64 && topk == 6) {
    mirage::kernel::moe_routing_distributed_task_impl<
        bfloat16, /*BATCH_SIZE=*/1, /*NUM_EXPERTS=*/64, /*TOPK=*/6,
        /*WORLD_SIZE=*/1, /*NORMALIZE=*/true>
        <<<batch_size, topk>>>(logits_ptr, idx_ptr, wts_ptr, dcounts_ptr,
                               nullptr, experts_per_rank, rank, 1.0f);
  } else {
    printf("Unsupported config: num_experts=%d, topk=%d\n", num_experts, topk);
    assert(false);
  }

  cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// EP MoE Combine: single-GPU test (WORLD_SIZE=1)
// ---------------------------------------------------------------------------
void moe_combine(torch::Tensor expert_outputs,
                 torch::Tensor routing_weights,
                 torch::Tensor residual,
                 torch::Tensor output,
                 int num_experts,
                 int experts_per_rank,
                 int rank,
                 bool add_residual) {
  int batch_size = output.size(0);
  int hidden_dim = output.size(1);
  int topk = expert_outputs.size(1);

  auto *expert_ptr = reinterpret_cast<bfloat16 const *>(
      expert_outputs.data_ptr());
  auto *wts_ptr = reinterpret_cast<bfloat16 const *>(
      routing_weights.data_ptr());
  auto *res_ptr = reinterpret_cast<bfloat16 const *>(residual.data_ptr());
  auto *out_ptr = reinterpret_cast<bfloat16 *>(output.data_ptr());

  // Use the __global__ test wrapper: one block per token range.
  if (hidden_dim == 64 && topk == 2) {
    if (add_residual) {
      mirage::kernel::all_to_all_combine_task_impl<
          bfloat16, /*BATCH_SIZE=*/8, /*HIDDEN_DIM=*/64, /*TOPK=*/2,
          /*WORLD_SIZE=*/1, /*ADD_RESIDUAL=*/true>
          <<<1, 128>>>(expert_ptr, nullptr, wts_ptr, res_ptr, out_ptr,
                       nullptr, nullptr, num_experts, experts_per_rank, rank,
                       nullptr);
    } else {
      mirage::kernel::all_to_all_combine_task_impl<
          bfloat16, /*BATCH_SIZE=*/8, /*HIDDEN_DIM=*/64, /*TOPK=*/2,
          /*WORLD_SIZE=*/1, /*ADD_RESIDUAL=*/false>
          <<<1, 128>>>(expert_ptr, nullptr, wts_ptr, res_ptr, out_ptr,
                       nullptr, nullptr, num_experts, experts_per_rank, rank,
                       nullptr);
    }
  } else if (hidden_dim == 256 && topk == 2) {
    if (add_residual) {
      mirage::kernel::all_to_all_combine_task_impl<
          bfloat16, /*BATCH_SIZE=*/8, /*HIDDEN_DIM=*/256, /*TOPK=*/2,
          /*WORLD_SIZE=*/1, /*ADD_RESIDUAL=*/true>
          <<<1, 128>>>(expert_ptr, nullptr, wts_ptr, res_ptr, out_ptr,
                       nullptr, nullptr, num_experts, experts_per_rank, rank,
                       nullptr);
    } else {
      mirage::kernel::all_to_all_combine_task_impl<
          bfloat16, /*BATCH_SIZE=*/8, /*HIDDEN_DIM=*/256, /*TOPK=*/2,
          /*WORLD_SIZE=*/1, /*ADD_RESIDUAL=*/false>
          <<<1, 128>>>(expert_ptr, nullptr, wts_ptr, res_ptr, out_ptr,
                       nullptr, nullptr, num_experts, experts_per_rank, rank,
                       nullptr);
    }
  } else {
    printf("Unsupported config: hidden_dim=%d, topk=%d\n", hidden_dim, topk);
    assert(false);
  }

  cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_routing", &moe_routing, "EP MoE distributed routing (single-GPU)");
  m.def("moe_combine", &moe_combine, "EP MoE combine (single-GPU)");
}
