/* Copyright 2023-2024 CMU
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

#include <cassert>
#include <string>
#include <unordered_map>
#include <vector>

#include "mirage/kernel/graph.h"
#include "mirage/transpiler/common.h"
#include "mirage/transpiler/sched_tb_graph.h"
#include "mirage/transpiler/structs.h"
#include "mirage/transpiler/utils.h"
#include "mirage/type.h"

namespace mirage {
namespace transpiler {

using std::vector;

class Transpiler {
private:
  // The kernel graph
  std::shared_ptr<kn::Graph> g;

  // User-provided configuration
  TranspilerConfig config;
  vector<vector<size_t>> input_strides;
  // Note that for certain output tensors, the output_stride vector
  // is empty, indicating that users do not specify a layout and
  // the transpiler can choose an arbitrary layout for that output tensor
  vector<vector<size_t>> output_strides;
  vector<kn::DTensor> mugraph_output_tensors;

  // Distributed configuration
  int num_gpus;
  bool use_nvshmem; // Whether to use NVSHMEM (<=> whether the kernel is
                    // distributed)
  void resolve_distributed_config() {
    num_gpus = g->gpu_dim.x * g->gpu_dim.y * g->gpu_dim.z;
    use_nvshmem = num_gpus > 1;
  }

  // Tensor metadata
  // A list of all distinct (by guid) DTensors
  std::vector<kn::DTensor> all_dtensors;
  std::unordered_map<decltype(kn::DTensor::guid), DTensorMeta>
      dtensor_metas; // DTensor guid -> metadata
  std::unordered_map<decltype(tb::STensor::guid), STensorMeta>
      stensor_metas; // STensor guid -> metadata
  void resolve_dtensor_meta();
  void resolve_tensor_layout();

  // Fusion metadata
  std::unordered_map<tb::TBOperator const *, bool> is_fused_with_prev;
  std::unordered_map<tb::TBOperator const *, bool> is_fused_with_next;
  std::unordered_map<tb::TBOperator const *, tb::TBOperator const *>
      chain_leading_op; // op |-> leading op
  std::unordered_map<tb::TBOperator const *,
                     std::vector<tb::TBOperator const *>>
      fusion_chain; // Leading op |-> [fused ops]
  void resolve_tb_fusion();

  // Memory allocation metadata
  size_t d_buf_size; // Size of the buffer for intermediate DTensors
  void plan_dtensor_memory();

  // Utility functions for transpiling
  // Get the pointer to a DTensor. Return {pointer_name, code}
  std::pair<std::string, std::string>
      get_dtensor_ptr(kn::DTensor const &dtensor);

  std::pair<std::string, std::string>
      get_profiling_ptr(int const customized_idx);

  // Get the "optimal" schedule for a threadblock graph
  TBSched get_threadblock_schedule(tb::Graph const &tb_graph);

  // Get the swizzle plan for a threadblock graph
  // (This function modifies STensorMeta directly, so it returns void)
  void get_threadblock_swizzle_plan(tb::Graph const &tb_graph,
                                    TBSched const &sched);

  void get_threadblock_swizzle_plan_hopper(tb::Graph const &tb_graph,
                                           TBSched const &sched);

  // Get the "optimal" memory plan for a threadblock graph
  TBMemoryPlan get_threadblock_memory_plan(tb::Graph const &tb_graph,
                                           TBSched const &tb_sched,
                                           bool hopper_arch = false);

  // Transpile a custom KN operator (a custom block graph)
  CustomOPTranspileResult transpile_kn_custom_op(kn::KNCustomizedOp const *op);
  CustomOPTranspileResult
      transpile_kn_custom_op_hopper(kn::KNCustomizedOp const *op);

  void get_hopper_tmas(CodeKeeper &code, std::vector<TMAParams> tmaParamsList);

  // Transpile the whole uGraph
  TranspileResult transpile_ugraph();

public:
  // Initialize the transpiler and resolve all configurations
  Transpiler(kernel::Graph const *g,
             TranspilerConfig const &config,
             vector<vector<size_t>> const &input_strides);

  TranspileResult generate_code() {
    this->resolve_distributed_config();
    this->resolve_dtensor_meta();
    this->resolve_tb_fusion();
    this->resolve_tensor_layout();
    this->plan_dtensor_memory();
    return this->transpile_ugraph();
  }
};

} // namespace transpiler
} // namespace mirage
