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

#include "mirage/kernel/customized.h"
#include "mirage/kernel/graph.h"

namespace mirage {
namespace transpiler {

class CudaTranspiler {
public:
  std::string gen_header_code(std::string indent);
  std::string gen_kernel_code(mirage::threadblock::NewKernelParams params,
                              int forloop_range,
                              int reduction_dimx,
                              std::string func_name,
                              std::vector<std::string> input_names,
                              std::vector<std::string> output_names,
                              std::string indent);
};

} // namespace transpiler
} // namespace mirage
