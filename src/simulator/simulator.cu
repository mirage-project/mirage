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

#include "mirage/simulator.h"
#include "mirage/utils/cuda_helper.h"

namespace mirage {
namespace simulator {

Simulator *Simulator::singleton = nullptr;

Simulator::Simulator() : offset(0) {
  // work_space_size = (size_t)2 * 1024 * 1024 * 1024 /*2GB*/;
  // checkCUDA(cudaMalloc(&base_ptr, work_space_size));
  // checkCUDA(cublasCreate(&blas));
  // checkCUDA(cublasSetMathMode(blas, CUBLAS_TENSOR_OP_MATH));
}

Simulator::~Simulator() {
  // checkCUDA(cudaFree(base_ptr));
  // checkCUDA(cublasDestroy(blas));
}

Simulator *Simulator::get_instance() {
  if (singleton == nullptr) {
    singleton = new Simulator();
  }
  return singleton;
}

void Simulator::free_all() {
  offset = 0;
}

void *Simulator::allocate(size_t size_in_bytes) {
  void *ret = static_cast<char *>(base_ptr) + offset;
  offset += size_in_bytes;
  assert(offset <= work_space_size);
  return ret;
}

} // namespace simulator
} // namespace mirage
