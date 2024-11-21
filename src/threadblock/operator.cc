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

#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/graph.h"

namespace mirage {
namespace threadblock {

TBOperator::TBOperator(Graph *_graph, mirage::type::TBOperatorType _type)
    : bgraph(_graph), op_type(_type) {}

TBOperator::TBOperator(Graph *_graph,
                       mirage::type::TBOperatorType _type,
                       STensor const &input1)
    : bgraph(_graph), op_type(_type) {
  input_tensors.push_back(input1);
}

TBOperator::TBOperator(Graph *_graph,
                       mirage::type::TBOperatorType _type,
                       STensor const &input1,
                       STensor const &input2)
    : bgraph(_graph), op_type(_type) {
  input_tensors.push_back(input1);
  input_tensors.push_back(input2);
}

int TBOperator::get_input_stensors(STensor** inputs) {
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    inputs[i] = &input_tensors[i];
  }
  return input_tensors.size();
}

int TBOperator::get_output_stensors(STensor** outputs) {
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    outputs[i] = &output_tensors[i];
  }
  return output_tensors.size();
}

TBOperator::~TBOperator() {}

} // namespace threadblock
} // namespace mirage
