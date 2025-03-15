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

#include "cutlass/fast_math.h"
#include "mirage/config.h"
#include "mirage/kernel/all_reduce.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/utils/cuda_helper.h"
#include "mirage/utils/fingerprint_functions.h"
#include "mirage/utils/hash_utils.h"
#include <cassert>

namespace mirage {
namespace kernel {

using namespace mirage::type;
using namespace mirage::config;
using namespace mirage::utils;

bool KNAllReduceOp::profile(ProfileResult &result) {
  // TODO: to be implemented
  result.run_time = 0.0f;
  return true;
}

__global__ void compute_allreduce_fingerprint(
    mirage::utils::FpPointerList fp_ptr_list, int num_gpus, int num_elements) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_elements) {
    FPType x = 0;
    for (int k = 0; k < num_gpus; k++) {
      x = compute_add_fingerprint(x, fp_ptr_list.ptrs[k][i]);
    }
    for (int k = 0; k < num_gpus; k++) {
      fp_ptr_list.ptrs[k][i] = x;
    }
  }
}

bool KNAllReduceOp::fingerprint(void) {
  // assert a 1-D GPU mesh
  assert(kgraph->gpu_dim.y == 1);
  assert(kgraph->gpu_dim.z == 1);

  assert(input_tensors[0].num_elements() == output_tensors[0].num_elements());
  int num_elements = input_tensors[0].num_elements();
  int const num_threads_per_blk = 1024;
  int num_blocks =
      (num_elements + num_threads_per_blk - 1) / num_threads_per_blk;
  mirage::kernel::DeviceMemoryManager *dmm =
      mirage::kernel::DeviceMemoryManager::get_instance();
  // assert inplace optimization is enabled
  assert(inplace);
  // Use GPU dmm->gpu_id for computing fingerprint
  checkCUDA(cudaSetDevice(dmm->gpu_id));
  mirage::utils::FpPointerList fp_ptr_list;
  for (int gpu_id = 0; gpu_id < kgraph->gpu_dim.x; gpu_id++) {
    fp_ptr_list.ptrs[gpu_id] = reinterpret_cast<mirage::type::FPType *>(
        dmm->fp_base_ptr[gpu_id] + input_tensors[0].fp_offset);
  }
  compute_allreduce_fingerprint<<<num_blocks, num_threads_per_blk>>>(
      fp_ptr_list, kgraph->gpu_dim.x, num_elements);
  checkCUDA(cudaDeviceSynchronize());
  return true;
}

} // namespace kernel
} // namespace mirage
