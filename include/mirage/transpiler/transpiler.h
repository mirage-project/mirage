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

#include <cassert>
#include <string>
#include <unordered_map>
#include <vector>

#include "mirage/kernel/graph.h"
#include "mirage/transpiler/common.h"
#include "mirage/transpiler/config.h"
#include "mirage/transpiler/tensor_meta.h"
#include "mirage/transpiler/utils.h"
#include "mirage/type.h"

namespace mirage {
namespace transpiler {

using std::vector;

class Transpiler {
private:
  // The kernel graph
  kernel::Graph const *g;

  // User-provided configuration
  TranspilerConfig config;
  vector<vector<size_t>> input_strides;
  vector<size_t> output_stride;

  // Distributed configuration
  int num_gpus;
  bool use_nvshmem; // Whether to use NVSHMEM (<=> whether the kernel is
                    // distributed)
  void resolve_distributed_config() {
    num_gpus = g->gpu_dim.x * g->gpu_dim.y * g->gpu_dim.z;
    use_nvshmem = num_gpus > 1;
  }

  // DTensor metadata
  std::unordered_map<decltype(kernel::DTensor::guid), DTensorMeta>
      dtensor_metas; // DTensor guid -> metadata
  void resolve_dtensor_meta();

  // STensor metadata
  std::unordered_map<decltype(threadblock::STensor::guid), STensorMeta>
      stensor_metas; // STensor guid -> metadata
  void resolve_stensor_meta();

  void resolve_all_config() {
    this->resolve_distributed_config();
    this->resolve_dtensor_meta();
    this->resolve_stensor_meta();
  }

  // Utility functions for transpiling
  // Get the pointer to a DTensor. Return {pointer_name, code}
  std::pair<std::string, std::string>
      get_dtensor_ptr(kernel::DTensor const &dtensor);

public:
  // Initialize the transpiler and resolve all configurations
  Transpiler(kernel::Graph const *g,
             TranspilerConfig const &config,
             vector<vector<size_t>> input_strides,
             vector<size_t> output_stride)
      : g(g), config(config), input_strides(input_strides),
        output_stride(output_stride) {
    // Currently we only support GPUs with compute capability >= 8.0 (A100+)
    // TODO(intlsy): Support older GPUs
    if (config.target_cc < GPU_CC::A100) {
      throw std::runtime_error("Unsupported target compute capability");
    }
    this->resolve_all_config();
  }

  TranspileResult generate_code();
};

} // namespace transpiler
} // namespace mirage
