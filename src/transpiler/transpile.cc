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

#include "mirage/transpiler/transpile.h"

#include <cassert>

#include "mirage/transpiler/transpiler.h"

namespace mirage {
namespace transpiler {

// Transpile a kernel graph into CUDA code
// Return (code, global memory buffer size (in bytes))
TranspileResult transpile(const kernel::Graph *g, const TranspilerConfig &config, const std::vector<std::vector<size_t>> &input_strides, const std::vector<size_t> &output_stride) {
	Transpiler transpiler(g, config, input_strides, output_stride);
	TranspileResult result = transpiler.generate_code();
	return result;
}

}
}
