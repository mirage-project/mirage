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

#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/kernel/operator.h"
#include "mirage/utils/cuda_helper.h"

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/matrix_coord.h"

namespace mirage {
namespace kernel {

using namespace mirage::type;
using namespace mirage::config;

template <typename DT>
__global__ void
    init_input(char *dmem_base_ptr, DTensor const A, size_t num_elements) {
  int idx = (threadIdx.x + blockIdx.x * blockDim.x);
  int kColumn = A.dim[A.num_dims - 1];
  // int myRow = idx / kColumn;
  int myColumn = idx % kColumn;
  DT *data_ptr = (DT *)(dmem_base_ptr + A.data_offset);
  if (idx < num_elements) {
    data_ptr[idx] = ((float)myColumn);
    // printf("idx(%d) v(%.f)\n", idx, (float)myRow);
  }
}

bool KNInputOp::profile(ProfileResult &profile) {
  // assert a 1-D GPU mesh
  assert(kgraph->gpu_dim.y == 1);
  assert(kgraph->gpu_dim.z == 1);

  profile.run_time = 0.0f;
  int const num_threads_per_blk = 1024;
  mirage::kernel::DeviceMemoryManager *dmm =
      mirage::kernel::DeviceMemoryManager::get_instance();
  int num_blocks =
      (output_tensors[0].num_elements() + num_threads_per_blk - 1) /
      num_threads_per_blk;
  for (int gpu_id = 0; gpu_id < kgraph->gpu_dim.x; gpu_id++) {
    checkCUDA(cudaSetDevice(gpu_id));
    if (output_tensors[0].data_type == mirage::type::DT_FLOAT16) {
      init_input<cutlass::half_t><<<num_blocks, num_threads_per_blk>>>(
          dmm->data_base_ptr[gpu_id],
          output_tensors[0],
          output_tensors[0].num_elements());
    } else {
      assert(false && "Unsupported type");
    }
  }
  checkCUDA(cudaDeviceSynchronize());
  return true;
}

__global__ void init_input_fingerprint(char *fp_base_ptr,
                                       DTensor const A,
                                       size_t num_elements,
                                       int gpu_id) {
  int idx = (threadIdx.x + blockIdx.x * blockDim.x);
  mirage::type::FPType *fp_ptr =
      (mirage::type::FPType *)(fp_base_ptr + A.fp_offset);
  if (idx < num_elements) {
    // FIXME: replace this with curand to generate random numbers
    fp_ptr[idx] = (idx + gpu_id * num_elements) % FP_PQ;
  }
}

bool KNInputOp::fingerprint(void) {
  // assert a 1-D GPU mesh
  assert(kgraph->gpu_dim.y == 1);
  assert(kgraph->gpu_dim.z == 1);
  int const num_threads_per_blk = 1024;
  mirage::kernel::DeviceMemoryManager *dmm =
      mirage::kernel::DeviceMemoryManager::get_instance();
  int num_blocks =
      (output_tensors[0].num_elements() + num_threads_per_blk - 1) /
      num_threads_per_blk;
  // Use GPU dmm->gpu_id for computing fingerprint
  checkCUDA(cudaSetDevice(dmm->gpu_id));
  for (int gpu_id = 0; gpu_id < kgraph->gpu_dim.x; gpu_id++) {
    init_input_fingerprint<<<num_blocks, num_threads_per_blk>>>(
        dmm->fp_base_ptr[gpu_id],
        output_tensors[0],
        output_tensors[0].num_elements(),
        gpu_id);
    checkCUDA(cudaDeviceSynchronize());
  }
  return true;
}

} // namespace kernel
} // namespace mirage
