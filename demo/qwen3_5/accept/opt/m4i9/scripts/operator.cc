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
#include "mirage/utils/hash_utils.h"

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

int KNOperator::get_input_dtensors(DTensor **inputs) {
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    inputs[i] = &input_tensors[i];
  }
  return input_tensors.size();
}

int KNOperator::get_output_dtensors(DTensor **outputs) {
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    outputs[i] = &output_tensors[i];
  }
  return output_tensors.size();
}

/*virtual*/
size_t KNOperator::get_owner_independent_hash() const {
  size_t ret = std::hash<int>()(op_type);
  for (auto const &t : input_tensors) {
    size_t h = t.get_owner_independent_hash();
    hash_combine(ret, h);
  }
  for (auto const &t : output_tensors) {
    size_t h = t.get_owner_independent_hash();
    hash_combine(ret, h);
  }
  return ret;
}

} // namespace kernel
} // namespace mirage
