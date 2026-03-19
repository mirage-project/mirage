/* Copyright 2023-2025 CMU
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

#include <memory>
#include <unordered_map>
#include <vector>

#include "mirage/kernel/device_tensor.h"
#include "mirage/kernel/graph.h"
#include "mirage/threadblock/graph.h"

namespace mirage {
namespace transpiler {

struct GraphNormalizationResult {
  std::shared_ptr<kernel::Graph> graph;
  std::vector<kernel::DTensor> mugraph_output_tensors;
  std::unordered_map<mirage::type::GuidType, kernel::DTensor> dtensor_mapping;
  std::unordered_map<mirage::type::GuidType, threadblock::STensor>
      stensor_mapping;
};

GraphNormalizationResult normalize_graph(kernel::Graph const *graph);

} // namespace transpiler
} // namespace mirage
