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
#include "mirage/kernel/device_tensor.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/type.h"
#include <vector>
#include <vector_types.h>

namespace mirage {
namespace threadblock {

class Graph;

class TBOperator {
public:
  TBOperator(Graph *graph, mirage::type::TBOperatorType);
  TBOperator(Graph *graph, mirage::type::TBOperatorType, STensor const &input1);
  TBOperator(Graph *graph,
             mirage::type::TBOperatorType,
             STensor const &input1,
             STensor const &input2);
  TBOperator(Graph *graph,
             mirage::type::TBOperatorType,
             std::vector<STensor> const &inputs);
  virtual ~TBOperator();

  virtual operator json() const = 0;

public:
  Graph *bgraph;
  mirage::type::TBOperatorType op_type;
  std::vector<STensor> input_tensors;
  std::vector<STensor> output_tensors;
};

class TBInputOp : public TBOperator {
public:
  TBInputOp(Graph *_graph,
            mirage::kernel::DTensor const &dtensor,
            int3 input_map,
            int forloop_dim,
            mirage::layout::SmemLayout layout);
  ~TBInputOp();

  operator json() const override;

public:
  mirage::kernel::DTensor dtensor;
  int3 input_map;
  int forloop_dim;
};

class TBOutputOp : public TBOperator {
public:
  TBOutputOp(Graph *_graph, STensor const &stensor, int3 output_map);
  ~TBOutputOp();

  operator json() const override;

public:
  mirage::kernel::DTensor dtensor;
  int3 output_map;
};

} // namespace threadblock
} // namespace mirage
