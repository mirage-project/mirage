/* Copyright 2023-2025 CMU
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

#pragma once

#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/threadblock/graph.h"
#include <string>
#include <vector>

namespace mirage {
namespace runtime {

struct TaskSmemRegion {
  std::string name;
  int size = 0;
  int alignment = 1;
  int page_count = -1;
  bool can_pack = false;
  int release_step = 0;
  bool contiguous = true;
};

struct TaskSmemInfo {
  int size = 0;
  int alignment = 1;
  std::vector<TaskSmemRegion> regions;
};

struct TaskRoleVariantCode {
  // Optional: op-declared body that the controller runs (single-thread)
  // once per published instruction, before role warps wake. Use to
  // mbar_init slots in runtime_smem->dynamic_semaphores[slot][i] that
  // the role bodies will arrive/wait on.
  std::string init_semaphores;
  std::string loader;
  std::string launcher;
  std::string consumer;
  std::string storer;

  // Page-lifecycle hooks. Defaults wire every task into the
  // generic page protocol (every task arrives every page exactly once).
  //   - auto_loader_page_lifecycle: codegen prepends every loader body
  //     with a lane-parallel "wait every page; for pages this task does
  //     not use, finish them immediately." If the user has no loader
  //     body, codegen emits one anyway with just this prefix.
  //   - auto_consumer_finish: codegen appends every consumer body with
  //     a runtime_finish_region_range_pages over the task's regions
  //     (i.e. the pages this task uses get released here). Set false
  //     for tasks that release pages incrementally inside their body
  //     (e.g. linear's per-stage release in 3.5b).
  bool auto_loader_page_lifecycle = true;
  bool auto_consumer_finish = true;
};

class TaskRegister {
public:
  static TaskRegister *singleton;
  TaskRegister();

public:
  static TaskRegister *get_instance();
  int register_embedding_task(threadblock::Graph const &bgraph,
                              std::vector<int> const &params);
  int register_rmsnorm_task(threadblock::Graph const &bgraph,
                            std::vector<int> const &params);
  int register_rmsnorm_linear_task(threadblock::Graph const &bgraph,
                                   std::vector<int> const &params);
  int register_attention_task(threadblock::Graph const &bgraph,
                              std::vector<int> const &params);
  int register_paged_attention_task(threadblock::Graph const &bgraph,
                                    std::vector<int> const &params);
  int register_single_batch_extend_attention_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_linear_task(threadblock::Graph const &bgraph,
                           std::vector<int> const &params,
                           bool with_residual);
  int register_silu_mul_task(threadblock::Graph const &bgraph,
                             std::vector<int> const &params);
  int register_identity_task(threadblock::Graph const &bgraph,
                             std::vector<int> const &params);
  int register_silu_mul_linear_with_residual_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_argmax_partial_task(threadblock::Graph const &bgraph,
                                   std::vector<int> const &params);
  int register_argmax_reduce_task(threadblock::Graph const &bgraph,
                                  std::vector<int> const &params);
  int register_reduction_task(threadblock::Graph const &bgraph,
                              std::vector<int> const &params);
  int register_find_ngram_partial_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params);
  int register_find_ngram_global_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  int register_target_verify_greedy_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);
  // Hopper tasks
  int register_linear_hopper_task(threadblock::Graph const &bgraph,
                                  std::vector<int> const &params,
                                  bool with_residual);
  int register_paged_attention_hopper_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);
  int register_rmsnorm_hopper_task(threadblock::Graph const &bgraph,
                                   std::vector<int> const &params);
  int register_linear_swapAB_hopper_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params,
                                         bool with_residual);
  int register_linear_cutlass_hopper_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params,
                                          bool with_residual);
  int register_silu_mul_hopper_task(threadblock::Graph const &bgraph,
                                    std::vector<int> const &params);
  int register_embedding_hopper_task(threadblock::Graph const &bgraph,
                                     std::vector<int> const &params);
  int register_moe_linear_sm90_task(threadblock::Graph const &bgraph,
                                    std::vector<int> const &params,
                                    bool w13_linear);
  int register_splitk_linear_swapAB_hopper_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params,
      bool with_residual);
  int register_paged_attention_split_kv_hopper_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  // SM100 tasks
  int register_splitk_linear_sm100_task(threadblock::Graph const &bgraph,
                                        std::vector<int> const &params,
                                        bool with_residual);
  int register_linear_sm100_task(threadblock::Graph const &bgraph,
                                 std::vector<int> const &params,
                                 bool with_residual);
  // v2 linear: typed Channel + TmemChannel primitives
  // (blackwell_v2/linear_sm100_v2.cuh + channel.cuh). Each role re-inits its
  // async edges at task start to clear stray arrivals on a reused ring slot.
  int register_linear_sm100_v2_task(threadblock::Graph const &bgraph,
                                    std::vector<int> const &params,
                                    bool with_residual);
  // v2 dispatch variants for non-linear tasks. Emit same kernel calls as the
  // v1 versions, but register under TASK_X_V2 enums so the whole pipeline
  // goes through v2 codegen (no mixed v1/v2 dispatch in the task graph).
  int register_rmsnorm_hopper_v2_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  int register_silu_mul_v2_task(threadblock::Graph const &bgraph,
                                std::vector<int> const &params);
  int register_embedding_v2_task(threadblock::Graph const &bgraph,
                                 std::vector<int> const &params);
  int register_paged_attention_sm100_v2_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params);
  int register_argmax_partial_sm100_v2_task(threadblock::Graph const &bgraph,
                                            std::vector<int> const &params);
  int register_argmax_reduce_sm100_v2_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);
  int register_paged_attention_sm100_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params);
  int register_argmax_partial_sm100_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);
  int register_argmax_reduce_sm100_task(threadblock::Graph const &bgraph,
                                        std::vector<int> const &params);
  int register_sampling_sm100_task(threadblock::Graph const &bgraph,
                                   std::vector<int> const &params);
  int register_tensor_init_task(threadblock::Graph const &bgraph,
                                std::vector<int> const &params);
  int register_moe_topk_softmax_sm100_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);
  int register_moe_topk_sigmoid_sm100_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);
  int register_moe_linear_sm100_task(threadblock::Graph const &bgraph,
                                     std::vector<int> const &params,
                                     bool w13_linear);
  int register_moe_fp8_sm100_task(threadblock::Graph const &bgraph,
                                  std::vector<int> const &params,
                                  bool w13_linear);
  int register_moe_silu_mul_task(threadblock::Graph const &bgraph,
                                 std::vector<int> const &params);
  int register_moe_mul_sum_add_sm100_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params);
  int register_paged_attention_split_kv_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_paged_attention_split_kv_merge_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_mla_decode_sm100_task(threadblock::Graph const &bgraph,
                                     std::vector<int> const &params);
  int register_mla_reduce_sm100_task(threadblock::Graph const &bgraph,
                                     std::vector<int> const &params);
  int register_mla_prefill_sm100_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  // SM100 tasks end
  // Multi-GPU tasks
  int register_nvshmem_allgather_strided_put_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_nvshmem_tile_allreduce_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);
  // Multi-GPU tasks end
  int register_task_variant(TaskType type, std::string const &code);
  void register_v2_task_role_variant(TaskType type,
                                     int variant_id,
                                     TaskRoleVariantCode code);

  // Register the total SMEM bytes a (TaskType, variant_id) consumes. Current
  // v2 codegen publishes this as metadata while keeping executable task bases
  // at offset 0. A future per-SM allocator can use these sizes for placement
  // once every task body supports non-zero offsets.
  void register_variant_smem_size(TaskType type, int variant_id, int size);
  void register_variant_smem_info(TaskType type,
                                  int variant_id,
                                  TaskSmemInfo info);
  TaskSmemInfo get_variant_smem_info(TaskType type, int variant_id) const;
  int get_variant_smem_size(TaskType type, int variant_id) const;

public:
  std::map<TaskType, std::vector<std::string>> all_task_variants;
  std::map<TaskType, std::vector<TaskRoleVariantCode>>
      all_v2_task_role_variants;
  std::map<TaskType, std::vector<TaskSmemInfo>> all_task_variant_smem_infos;
};

} // namespace runtime
} // namespace mirage
