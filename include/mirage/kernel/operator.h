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
#include "mirage/profile_result.h"
#include "mirage/utils/json_utils.h"
#include <vector>

namespace mirage {
namespace kernel {

class Graph;

class KNOperator {
public:
  KNOperator(Graph *graph, mirage::type::KNOperatorType _type);
  KNOperator(Graph *graph,
             mirage::type::KNOperatorType _type,
             DTensor const &input1);
  KNOperator(Graph *graph,
             mirage::type::KNOperatorType _type,
             DTensor const &input1,
             DTensor const &input2);
  KNOperator(Graph *graph,
             mirage::type::KNOperatorType _type,
             std::vector<DTensor> const &inputs);
  int get_input_dtensors(DTensor **inputs);
  int get_output_dtensors(DTensor **inputs);

  virtual ~KNOperator();
  virtual bool profile(ProfileResult &result) = 0;
  virtual bool fingerprint(void) = 0;
  virtual operator json() const = 0;

  // hash related functions
  virtual size_t get_owner_independent_hash() const;

public:
  Graph *kgraph;
  mirage::type::KNOperatorType op_type;
  std::vector<DTensor> input_tensors;
  std::vector<DTensor> output_tensors;
};

class KNInputOp : public KNOperator {
public:
  KNInputOp(Graph *_graph,
            std::vector<int> const &dims,
            std::vector<size_t> const &strides,
            mirage::type::DataType data_type,
            mirage::layout::DmemLayout layout,
            int3 input_map = {-1, -1, -1});
  ~KNInputOp();
  bool profile(ProfileResult &profile);
  bool fingerprint(void);

  operator json() const override;

public:
  std::vector<size_t> input_strides;
  int3 input_map;
};

class KNOutputOp : public KNOperator {
public:
  KNOutputOp(Graph *_graph,
             DTensor const &A,
             std::vector<size_t> const &strides,
             int3 output_map = {-1, -1, -1});
  ~KNOutputOp();
  bool profile(ProfileResult &profile);
  bool fingerprint(void);

  operator json() const override;

public:
  std::vector<size_t> output_strides;
  int3 output_map;
};

} // namespace kernel
} // namespace mirage
