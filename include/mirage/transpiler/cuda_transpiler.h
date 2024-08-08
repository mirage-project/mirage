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

#include "mirage/kernel/customized.h"
#include "mirage/kernel/graph.h"

#include <set>

namespace mirage {
namespace transpiler {

class CudaTranspiler {
public:
  CudaTranspiler(bool nvshmem);
  void define_stensor_from_offset(
      std::stringstream &ss,
      int offset,
      std::string name,
      std::string ind,
      mirage::type::DataType type = mirage::type::DT_FLOAT16);
  void gen_cuda_code_input_loader(std::string dtensor_name, std::string indent);
  void gen_cuda_code_output_saver(std::string dtensor_name, std::string indent);
  void gen_cuda_code_forloop_accum(std::string indent);
  void gen_cuda_code_matmul_op(std::string indent);
  void gen_cuda_code_exp_op(std::string indent);
  void gen_cuda_code_div_op(std::string indent);
  void gen_cuda_code_reduction_op(std::string indent);

  std::string generate_header_code(std::string indent);
  std::string generate_kernel_code(mirage::threadblock::NewKernelParams params,
                                   mirage::threadblock::Graph const *graph,
                                   std::string func_name,
                                   std::vector<std::string> input_names,
                                   std::vector<std::string> output_names,
                                   std::string indent);

public:
  mirage::threadblock::NewKernelParams params;
  mirage::threadblock::Graph const *bgraph;
  int param_idx, op;
  bool use_nvshmem;
  std::stringstream input_loader_func;
  std::stringstream output_saver_func;
  std::stringstream header;
  std::stringstream main;
  std::stringstream ending;
  std::set<int> input_loader_smem_offsets;
};

} // namespace transpiler
} // namespace mirage
