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
#include <optional>
#include <string>
#include <vector>

#include "mirage/kernel/element_binary.h"
#include "mirage/kernel/graph.h"
#include "mirage/transpiler/structs.h"

namespace mirage {
namespace nki_transpiler {

namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;
using dguid_t = decltype(kn::DTensor::guid); // Guid of a DTensor
using sguid_t = decltype(tb::STensor::guid); // Guid of a STensor

// Metadata for STensors during transpiling
// STensors with the same `guid` share one STensorMeta
struct STensorMeta {
  // Partition dim for NKI
  int partition_dim;

  // Physical size needed for the tensor (in number of elements)
  size_t num_phy_elems;
};

struct NKITranspilerConfig {
  // Target compute capability
  int target_cc;
};

struct NeuronArch {
  NeuronArch() = delete;
  NeuronArch(NeuronArch const &) = delete;
  NeuronArch(NeuronArch &&) = delete;
  static constexpr char const *pmax = "nki.language.tile_size.pmax";
  static constexpr char const *psum_fmax = "nki.language.tile_size.psum_fmax";
  static constexpr char const *gemm_mov_fmax =
      "nki.language.tile_size.gemm_moving_fmax";
  static constexpr char const *gemm_sta_fmax =
      "nki.language.tile_size.gemm_stationary_fmax";
};

// Descriptive transpiler errors.
// Currently, only layout errors are reported, further descriptive
// errors can be added to provide info to users.
struct NKIErrorInfo {
  std::vector<std::string> errors;
  NKIErrorInfo() = default;
  NKIErrorInfo(std::vector<std::string> err) : errors(err) {}
};

// Transpile a custom KN operator (a custom block graph)
struct NKICustomOPTranspileResult {
  // The name of the generated kernel function
  std::string func_name;
  // The kernel function code. Should be something like:
  // def func_name(InputDTensor0, ..., InputDTensorN):
  //  [kernel code]
  std::string code;
};

struct NKITranspileResult {
  // The generated NKI code
  std::string code;

  // The size of the buffer (should be an array on Trainium), in bytes
  // size_t buf_size;

  // Transpiler errors for a valid mugraphs.
  NKIErrorInfo error_state;

  // Default constructor to build py extensions
  NKITranspileResult() : code(""), error_state(NKIErrorInfo()) {}
  explicit NKITranspileResult(std::string code_,
                              NKIErrorInfo err_ = NKIErrorInfo())
      : code(std::move(code_)), error_state(err_) {}
};

class NKITranspiler {
private:
  // The kernel graph
  std::shared_ptr<mirage::kernel::Graph> g;
  // User-provided configuration
  NKITranspilerConfig config;
  std::vector<mirage::kernel::DTensor> mugraph_output_tensors;
  std::unordered_map<decltype(tb::STensor::guid), STensorMeta>
      stensor_metas; // STensor guid -> metadata
public:
  NKITranspiler(kernel::Graph const *_graph,
                NKITranspilerConfig const &_config);
  NKITranspileResult generate_code();
  std::optional<NKIErrorInfo> resolve_tensor_layout();
  NKICustomOPTranspileResult
      transpile_kn_custom_op(kn::KNCustomizedOp const *op);
  std::optional<NKICustomOPTranspileResult>
      transpile_kn_op(kn::KNOperator const *op);
  NKITranspileResult transpile_ugraph();
};

NKITranspileResult transpile(kernel::Graph const *g,
                             NKITranspilerConfig const &config);

} // namespace nki_transpiler
} // namespace mirage
