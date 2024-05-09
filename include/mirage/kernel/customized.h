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
#include "mirage/kernel/operator.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"
#include <tuple>
#include <vector_types.h>

namespace mirage {
namespace kernel {

class KNCustomizedOp : public mirage::kernel::KNOperator {
public:
  KNCustomizedOp(std::vector<DTensor> const &inputs,
                 mirage::threadblock::ExecutionPlan const &plan);
  KNCustomizedOp(std::vector<DTensor> const &inputs,
                 mirage::threadblock::Graph const &_graph);
  virtual ~KNCustomizedOp();
  bool profile(ProfileResult &profile);
  void run(void);
  bool fingerprint(void);

  operator json() const override;

public:
  mirage::threadblock::ExecutionPlan plan;
  mirage::threadblock::Graph bgraph;
};

} // namespace kernel
} // namespace mirage
