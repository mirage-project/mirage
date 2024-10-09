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

#include "mirage/kernel/operator.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"

namespace mirage {
namespace kernel {

KNOperator::KNOperator(Graph *_graph, mirage::type::KNOperatorType _type)
    : kgraph(_graph), op_type(_type) {}

KNOperator::KNOperator(Graph *_graph,
                       mirage::type::KNOperatorType _type,
                       DTensor const &A)
    : kgraph(_graph), op_type(_type) {
  input_tensors.push_back(A);
}

KNOperator::KNOperator(Graph *_graph,
                       mirage::type::KNOperatorType _type,
                       DTensor const &A,
                       DTensor const &B)
    : kgraph(_graph), op_type(_type) {
  input_tensors.push_back(A);
  input_tensors.push_back(B);
}

KNOperator::KNOperator(Graph *_graph,
                       mirage::type::KNOperatorType _type,
                       std::vector<DTensor> const &inputs)
    : kgraph(_graph), op_type(_type) {
  for (auto const &i : inputs) {
    input_tensors.push_back(i);
  }
}

KNOperator::~KNOperator() {}

} // namespace kernel
} // namespace mirage
