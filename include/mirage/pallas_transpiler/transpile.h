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
#include <optional>
#include <string>
#include <vector>

#include "mirage/kernel/graph.h"

namespace mirage {
namespace pallas_transpiler {

namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

struct PallasTranspilerConfig {
  std::string target_chip;
  bool debug = false;
};

struct PallasErrorInfo {
  std::vector<std::string> errors;
  PallasErrorInfo() = default;
  explicit PallasErrorInfo(std::vector<std::string> err)
      : errors(std::move(err)) {}
};

struct PallasCustomOPTranspileResult {
  std::string func_name;
  std::string code;
};

struct PallasTranspileResult {
  std::string code;
  std::vector<std::vector<int>> output_shapes;
  PallasErrorInfo error_state;

  PallasTranspileResult()
      : code(""), output_shapes(), error_state(PallasErrorInfo()) {}
  explicit PallasTranspileResult(std::string code_,
                                 std::vector<std::vector<int>> output_shapes_ = {},
                                 PallasErrorInfo err_ = PallasErrorInfo())
      : code(std::move(code_)),
        output_shapes(std::move(output_shapes_)),
        error_state(std::move(err_)) {}
};

class PallasTranspiler {
private:
  std::shared_ptr<mirage::kernel::Graph> g;
  PallasTranspilerConfig config;
  std::vector<mirage::kernel::DTensor> mugraph_output_tensors;
  std::vector<std::string> errors;

  static int kernel_idx_counter;

  std::optional<PallasErrorInfo> validate_graph();
  PallasCustomOPTranspileResult
      transpile_kn_custom_op(kn::KNCustomizedOp const *op);
  PallasTranspileResult transpile_ugraph();

public:
  PallasTranspiler(kernel::Graph const *graph,
                   PallasTranspilerConfig const &config);
  PallasTranspileResult generate_code();
};

PallasTranspileResult transpile(kernel::Graph const *g,
                                PallasTranspilerConfig const &config);

} // namespace pallas_transpiler
} // namespace mirage
