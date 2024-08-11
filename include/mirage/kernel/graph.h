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
#include "mirage/kernel/device_tensor.h"
#include "mirage/kernel/operator.h"
#include "mirage/threadblock/graph.h"
#include <vector>

namespace mirage {
namespace kernel {

class Graph {
private:
  struct pair_hash {
    size_t operator()(std::pair<int, int> const &p) const;
  };

public:
  Graph(dim3 gpu_dim = {1, 1, 1});
  ~Graph();
  Graph(Graph const &) = delete;
  Graph &operator=(Graph const &) = delete;
  // input operator
  DTensor new_input(std::vector<int> const &dims,
                    mirage::type::DataType data_type,
                    mirage::layout::DmemLayout layout);
  DTensor *new_input_ptr(std::vector<int> const &dims,
                         mirage::type::DataType data_type,
                         mirage::layout::DmemLayout layout);
  KNOperator *create_input_op(std::vector<int> const &dims,
                              mirage::type::DataType data_type,
                              mirage::layout::DmemLayout layout);
  // matmul operator
  DTensor matmul(DTensor const &A, DTensor const &B);
  DTensor *matmul(DTensor const *A, DTensor const *B);
  KNOperator *create_matmul_op(DTensor const &A, DTensor const &B);
  // elementunary operator
  DTensor exp(DTensor const &input);
  DTensor *exp(DTensor const *input);
  KNOperator *create_elementunary_op(DTensor const &input,
                                     mirage::type::KNOperatorType _type);
  // elementunary operator
  DTensor add(DTensor const &input1, DTensor const &input2);
  DTensor mul(DTensor const &input1, DTensor const &input2);
  DTensor div(DTensor const &input1, DTensor const &input2);
  DTensor *add(DTensor const *input1, DTensor const *input2);
  DTensor *mul(DTensor const *input1, DTensor const *input2);
  DTensor *div(DTensor const *input1, DTensor const *input2);
  KNOperator *create_elementbinary_op(DTensor const &input1,
                                      DTensor const &input2,
                                      mirage::type::KNOperatorType _type);
  // reduction operator
  DTensor reduction(DTensor const &input, int dim, int size = 1);
  DTensor *reduction(DTensor const *input, int dim, int size = 1);
  KNOperator *create_reduction_op(DTensor const &input, int dim, int factor);
  // allreduce operator
  DTensor all_reduce(DTensor const &input, bool inplace = true);
  DTensor *all_reduce(DTensor const *input, bool inplace = true);
  KNOperator *create_all_reduce_op(DTensor const &input, bool inplace);
  // customized operator
  std::vector<DTensor>
      customized(std::vector<DTensor> const &inputs,
                 mirage::threadblock::ExecutionPlan const &plan);
  KNOperator *
      create_customized_op(std::vector<DTensor> const &inputs,
                           mirage::threadblock::ExecutionPlan const &plan);
  std::vector<DTensor> customized(std::vector<DTensor> const &inputs,
                                  mirage::threadblock::Graph const &_graph);
  KNOperator *create_customized_op(std::vector<DTensor> const &inputs,
                                   mirage::threadblock::Graph const &_graph);
  // helper functions
  void generate_triton_program(char const *filepath);
  void generate_cuda_program(char const *filepath);
  bool can_allocate(DTensor const &tensor,
                    bool allocate_fingerprint = true) const;
  bool can_allocate(size_t data_size_in_bytes, size_t fp_size_in_bytes) const;
  bool allocate(DTensor &tensor, bool allocate_fingerprint = true);
  void free(DTensor &tensor);

public:
  std::vector<mirage::kernel::KNOperator *> operators;
  dim3 gpu_dim;
  // memory allocator
  // device memory offset manager
  off_t dmem_data_offset, dmem_fp_offset;
  std::vector<std::pair<off_t, size_t>> allocated_data_tensors,
      allocated_fp_tensors;
  // std::unordered_map<std::pair<int, int>, DTensor, pair_hash> tensors;
  // std::unordered_map<std::pair<int, int>, std::pair<int, int>, pair_hash>
  // edges; std::vector<std::vector<SrcEdge>> edges;
  // mirage::kernel::OperatorFactory *operator_factory;

  using OpType = KNOperator;
  using TensorType = DTensor;
};

void to_json(json &j, Graph const &g);
void from_json(json const &j, Graph &g);

} // namespace kernel
} // namespace mirage
