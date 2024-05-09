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
#include "mirage/kernel/operator.h"
// #include "mirage/kernel/operator_factory.h"
#include "mirage/kernel/device_tensor.h"
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
  Graph(void);
  // input operator
  DTensor new_input(std::vector<int> const &dims,
                    mirage::type::DataType data_type,
                    mirage::layout::DmemLayout layout);
  DTensor* new_input_ptr(std::vector<int> const &dims,
                         mirage::type::DataType data_type,
                         mirage::layout::DmemLayout layout);
  KNOperator *create_input_op(std::vector<int> const &dims,
                              mirage::type::DataType data_type,
                              mirage::layout::DmemLayout layout);
  // matmul operator
  DTensor matmul(DTensor const &A, DTensor const &B);
  DTensor* matmul(DTensor const *A, DTensor const *B);
  KNOperator *create_matmul_op(DTensor const &A, DTensor const &B);
  // elementunary operator
  DTensor exp(DTensor const &input);
  DTensor* exp(DTensor const *input);
  KNOperator *create_elementunary_op(DTensor const &input,
                                     mirage::type::KNOperatorType _type);
  // elementunary operator
  DTensor add(DTensor const &input1, DTensor const &input2);
  DTensor mul(DTensor const &input1, DTensor const &input2);
  DTensor div(DTensor const &input1, DTensor const &input2);
  DTensor* add(DTensor const *input1, DTensor const *input2);
  DTensor* mul(DTensor const *input1, DTensor const *input2);
  DTensor* div(DTensor const *input1, DTensor const *input2);
  KNOperator *create_elementbinary_op(DTensor const &input1,
                                      DTensor const &input2,
                                      mirage::type::KNOperatorType _type);
  // reduction operator
  DTensor reduction(DTensor const &input, int dim, int size = 1);
  DTensor* reduction(DTensor const *input, int dim, int size = 1);
  KNOperator *create_reduction_op(DTensor const &input, int dim, int factor);
  // customized operator
  std::vector<DTensor> customized(std::vector<DTensor> const &inputs,
                                  mirage::threadblock::ExecutionPlan const &plan);
  KNOperator *create_customized_op(std::vector<DTensor> const &inputs,
                                   mirage::threadblock::ExecutionPlan const &plan);
  std::vector<DTensor> customized(std::vector<DTensor> const &inputs,
                                  mirage::threadblock::Graph const &_graph);
  KNOperator *create_customized_op(std::vector<DTensor> const &inputs,
                                   mirage::threadblock::Graph const &_graph);
  // helper functions
  void generate_triton_program(char const *filepath);
  std::vector<mirage::kernel::KNOperator *> operators;
  // std::unordered_map<std::pair<int, int>, DTensor, pair_hash> tensors;
  // std::unordered_map<std::pair<int, int>, std::pair<int, int>, pair_hash>
  // edges; std::vector<std::vector<SrcEdge>> edges;
  // mirage::kernel::OperatorFactory *operator_factory;
};

void to_json(json &j, Graph const &g);
void from_json(json const &j, Graph &g);

} // namespace kernel
} // namespace mirage
