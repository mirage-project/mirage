/* Copyright 2023-2026 CMU
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

// A host-only core (USE_NKI, USE_HIP: no CUDA sources, no transpiler): the
// functions the Python binding (python/mirage/_cython/core.pyx) links that
// exist only in the CUDA build. They fail with a clear message instead of an
// undefined symbol at import.
#ifndef MIRAGE_BACKEND_USE_CUDA

#include "mirage/kernel/device_memory_manager.h"
#include "mirage/transpiler/transpile.h"

#include <stdexcept>

namespace mirage {
namespace transpiler {

TranspileResult transpile(kernel::Graph const *,
                          TranspilerConfig const &,
                          std::vector<std::vector<size_t>> const &) {
  throw std::runtime_error(
      "transpile: the CUDA transpiler needs the CUDA build (USE_CUDA ON)");
}

} // namespace transpiler

namespace kernel {

// the CUDA build selects the device of the fingerprint buffers; a host-only
// core keeps them in host memory
void cython_set_gpu_device_id(int) {}

} // namespace kernel
} // namespace mirage

#endif // MIRAGE_BACKEND_USE_CUDA
