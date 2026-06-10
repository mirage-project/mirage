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


// Unified-entry selectors. Sub-variant ints are documented at each entry.
enum class MlaTaskKind {
  Decode,       // variant = tp_size (1/2/4/8)
  Reduce,       // variant = tp_size (1/2/4/8)
  DecodeCompat, // catalog-ABI decode (variant unused)
  ReduceCompat, // catalog-ABI reduce (variant unused)
  KvGather,     // 0=dense, 1=split, 2=unified
  Prefill,      // 0=plain, 1=absorbed, 2=tp8-chunked
  Rope,         // 0=q, 1=q_fused, 2=q_split, 3=k
  AssembleQ,    // variant unused
};
enum class AttentionTaskKind {
  Base, Paged, PagedHopper, PagedSm100,
  PagedSplitKv, PagedSplitKvMerge, PagedSplitKvHopper, SingleBatchExtend,
};

class TaskRegister {
public:
  // Unified MLA-family entry (kind + variant; see enum).
  int register_mla_task(threadblock::Graph const &bgraph,
                        std::vector<int> const &params,
                        MlaTaskKind kind,
                        int variant = 0);
  // Unified per-head FP8 BMM entry: dense selects swapAB vs dense body.
  int register_linear_fp8_bmm_unified_sm100_task(
      threadblock::Graph const &bgraph,
      std::vector<int> const &params,
      bool dense);
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
  // Unified attention-family entry (kind: Base/Paged/PagedHopper/PagedSm100/
  // PagedSplitKv/PagedSplitKvMerge/PagedSplitKvHopper/SingleBatchExtend).
  int register_attention_task(threadblock::Graph const &bgraph,
                              std::vector<int> const &params,
                              AttentionTaskKind kind);
  int register_linear_task(threadblock::Graph const &bgraph,
                           std::vector<int> const &params,
                           bool with_residual);
  int register_silu_mul_task(threadblock::Graph const &bgraph,
                             std::vector<int> const &params);
  int register_identity_task(threadblock::Graph const &bgraph,
                             std::vector<int> const &params);
  int register_identity_2in_task(threadblock::Graph const &bgraph,
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
  // SM100 tasks
  int register_splitk_linear_sm100_task(threadblock::Graph const &bgraph,
                                        std::vector<int> const &params,
                                        bool with_residual);
  int register_linear_sm100_task(threadblock::Graph const &bgraph,
                                 std::vector<int> const &params,
                                 bool with_residual);
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
  int register_quantize_fp8_sm100_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params,
                                       bool scale_ue8m0);
  int register_linear_fp8_sm100_task(threadblock::Graph const &bgraph,
                                     std::vector<int> const &params,
                                     bool with_residual);
  int register_linear_fp8_swapAB_sm100_task(threadblock::Graph const &bgraph,
                                            std::vector<int> const &params,
                                            bool with_residual);
  int register_splitk_linear_fp8_swapAB_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  // Unified dense BN128 entry (#201 L1): variant 0=smallm (NE2), 1=mediumm
  // (NE4). Byte-identical codegen to the prior two per-variant wrappers.
  // Unified dense FP8 GEMM entry. variant: 0=smallm, 1=mediumm,
  // 2=smallm fp8out, 3=mediumm fp8out.
  int register_fp8_gemm_dense_bn128_sm100_task(threadblock::Graph const &bgraph,
                                               std::vector<int> const &params,
                                               int variant);
  int register_fused_rmsnorm_quantize_fp8_sm100_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  // Unified grouped-GEMM entry (#201 L1): variant 0=smallm, 1=largem.
  // Byte-identical codegen to the prior two per-variant wrappers.
  int register_fp8_group_gemm_sm100_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params,
                                         int variant);
  int register_moe_permute_sm100_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);
  int register_moe_unpermute_sm100_task(threadblock::Graph const &bgraph,
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
  int register_mtp_build_embed_input_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params);
  // SM100 tasks end
  // Multi-GPU tasks
  int register_nvshmem_allgather_strided_put_task(
      threadblock::Graph const &bgraph, std::vector<int> const &params);
  int register_nvshmem_tile_allreduce_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);
  int register_nvshmem_global_argmax_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params);
  // Multi-GPU tasks end
  int register_task_variant(TaskType type, std::string const &code);

public:
  std::map<TaskType, std::vector<std::string>> all_task_variants;
};

} // namespace runtime
} // namespace mirage
