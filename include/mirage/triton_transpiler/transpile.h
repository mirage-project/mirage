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

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mirage/kernel/graph.h"
#include "mirage/threadblock/graph.h"

namespace mirage {
namespace triton_transpiler {

namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

struct STensorMeta {
  int partition_dim;
  size_t num_phy_elems;
};
// Transpiler configuration
struct TritonTranspilerConfig {
  // CUDA compute capability
  int target_cc;
};

// Result of transpiling a custom kernel operator
struct TritonCustomOPTranspileResult {
  // Generated kernel function name
  std::string func_name;
  // Generated Triton kernel code
  std::string code;
};

// Final result of transpilation
struct TritonTranspileResult {
  // The generated Triton code
  std::string code;
  std::vector<std::vector<int>> output_shapes;
};

class TritonTranspiler {
private:
  // The kernel graph to transpile
  std::shared_ptr<mirage::kernel::Graph> g;
  // Configuration
  TritonTranspilerConfig config;
  std::vector<mirage::kernel::DTensor> mugraph_output_tensors;
  std::unordered_map<decltype(tb::STensor::guid), STensorMeta>
      stensor_metas; // STensor guid -> metadata

  // Internal counter for kernel naming
  static int kernel_idx_counter;

public:
  TritonTranspiler(kernel::Graph const *_graph,
                   TritonTranspilerConfig const &_config);
  // Main entry point for code generation
  TritonTranspileResult generate_code();
  // Transpile a custom kernel operator
  TritonCustomOPTranspileResult
      transpile_kn_custom_op(kn::KNCustomizedOp const *op);

  // Transpile the kernel graph
  TritonTranspileResult transpile_ugraph();
};

TritonTranspileResult transpile(kernel::Graph const *g,
                                TritonTranspilerConfig const &config);

} // namespace triton_transpiler
} // namespace mirage