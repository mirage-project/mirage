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

namespace mirage {
namespace runtime {

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
  int register_elementwise_add_sm100_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params);
  int register_softmax_gather_sm100_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);
  int register_mtp_verify_probabilistic_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params);
  int register_mtp_float_scatter_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  int register_prob_extract_sm100_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params);
  int register_prob_scatter_sm100_task(threadblock::Graph const &bgraph,
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
  int register_mla_mtp_decode_sm100_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);
  int register_mla_mtp_reduce_sm100_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);
  // MLA-MTP TP variants (ferret-derived, no-PDL). Each: TP=2/4/8, with paired
  // reduce. Differ in NUM_HEADS (64/32/16); TP=4 also splits V across two
  // CTAs (z=2); TP=8 takes Q_LEN_real (Q_LEN padded to even).
  int register_mla_mtp_decode_tp2_sm100_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params);
  int register_mla_mtp_decode_tp2_reduce_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_mla_mtp_decode_tp4_sm100_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params);
  int register_mla_mtp_decode_tp4_reduce_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_mla_mtp_decode_tp8_sm100_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params);
  int register_mla_mtp_decode_tp8_reduce_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_quantize_fp8_sm100_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params,
                                       bool scale_ue8m0);
  int register_linear_fp8_sm100_task(threadblock::Graph const &bgraph,
                                     std::vector<int> const &params,
                                     bool with_residual);
  int register_mla_kv_gather_sm100_task(threadblock::Graph const &bgraph,
                                        std::vector<int> const &params);
  // MTP tasks
  int register_mtp_verify_strict_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  int register_mtp_accept_commit_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  int register_mtp_token_scatter_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  int register_mtp_prepare_verify_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params);
  // SM100 tasks end
  // Multi-GPU tasks
  int register_nvshmem_allgather_strided_put_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_nvshmem_tile_allreduce_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);
  // Multi-GPU tasks end
  int register_task_variant(TaskType type, std::string const &code);

public:
  std::map<TaskType, std::vector<std::string>> all_task_variants;
};

} // namespace runtime
} // namespace mirage
