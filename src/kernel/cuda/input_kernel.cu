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

#include "mirage/kernel/operator.h"
#include "mirage/utils/cuda_helper.h"

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/matrix_coord.h"

namespace mirage {
namespace kernel {

using namespace mirage::type;

template <typename DT>
__global__ void init_input(DTensor const A, size_t num_elements) {
  int idx = (threadIdx.x + blockIdx.x * blockDim.x);
  int kColumn = A.dim[A.num_dims - 1];
  // int myRow = idx / kColumn;
  int myColumn = idx % kColumn;
  if (idx < num_elements) {
    ((DT *)A.data_ptr)[idx] = ((float)myColumn);
    // printf("idx(%d) v(%.f)\n", idx, (float)myRow);
  }
}

bool KNInputOp::profile(ProfileResult &profile) {
  profile.run_time = 0.0f;
  int const num_threads_per_blk = 1024;
  int num_blocks =
      (output_tensors[0].num_elements() + num_threads_per_blk - 1) /
      num_threads_per_blk;
  if (output_tensors[0].data_type == mirage::type::DT_FLOAT16) {
    init_input<cutlass::half_t><<<num_blocks, num_threads_per_blk>>>(
        output_tensors[0], output_tensors[0].num_elements());
  } else {
    assert(false && "Unsupported type");
  }
  checkCUDA(cudaDeviceSynchronize());
  return true;
}

__global__ void init_input_fingerprint(DTensor const A, size_t num_elements) {
  int idx = (threadIdx.x + blockIdx.x * blockDim.x);
  if (idx < num_elements) {
    // FIXME: replace this with curand to generate random numbers
    A.fp_ptr[idx] = idx % FP_PQ;
  }
}

bool KNInputOp::fingerprint(void) {
  int const num_threads_per_blk = 1024;
  int num_blocks =
      (output_tensors[0].num_elements() + num_threads_per_blk - 1) /
      num_threads_per_blk;
  init_input_fingerprint<<<num_blocks, num_threads_per_blk>>>(
      output_tensors[0], output_tensors[0].num_elements());
  checkCUDA(cudaDeviceSynchronize());
  return true;
}

} // namespace kernel
} // namespace mirage
