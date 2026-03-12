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
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/kernel/reduction.h"
#include "mirage/utils/cuda_helper.h"
#include "mirage/utils/fingerprint_functions.h"
#include "mirage/utils/hash_utils.h"
#include <cassert>

namespace mirage {
namespace kernel {

using namespace mirage::type;
using namespace mirage::config;
using namespace mirage::utils;

#ifdef MIRAGE_FINGERPRINT_USE_CUDA
__global__ void compute_reduction_fingerprint(FPType *input_ptr,
                                              FPType *output_ptr,
                                              int num_elements,
                                              int reduction_factor,
                                              int input_stride,
                                              int output_stride) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_elements) {
    FPType result = 0;
    int n = i / output_stride;
    int m = i % output_stride;
    for (int k = 0; k < reduction_factor; k++) {
      FPType input = input_ptr[n * input_stride + m + k * output_stride];
      result = compute_add_fingerprint(result, input);
    }
    output_ptr[i] = result;
  }
}

bool KNReductionOp::fingerprint(void) {
  // assert a 1-D GPU mesh
  assert(kgraph->gpu_dim.y == 1);
  assert(kgraph->gpu_dim.z == 1);

  int num_elements = output_tensors[0].num_elements();
  int const num_threads_per_blk = 1024;
  int num_blocks =
      (num_elements + num_threads_per_blk - 1) / num_threads_per_blk;
  int output_stride = 1;
  int input_stride = 1;
  for (int i = reduction_dim_idx; i < output_tensors[0].num_dims; i++) {
    output_stride *= output_tensors[0].dim[i];
    input_stride *= input_tensors[0].dim[i];
  }
  int reduction_factor = input_tensors[0].dim[reduction_dim_idx] /
                         output_tensors[0].dim[reduction_dim_idx];
  assert(output_stride * reduction_factor == input_stride);
  mirage::kernel::DeviceMemoryManager *dmm =
      mirage::kernel::DeviceMemoryManager::get_instance();
  // Use GPU dmm->gpu_id for computing fingerprint
  checkCUDA(cudaSetDevice(dmm->gpu_id));

  for (int gpu_id = 0; gpu_id < kgraph->gpu_dim.x; gpu_id++) {
    mirage::type::FPType *input_fp_ptr =
        reinterpret_cast<mirage::type::FPType *>(dmm->fp_base_ptr[gpu_id] +
                                                 input_tensors[0].fp_offset);
    mirage::type::FPType *output_fp_ptr =
        reinterpret_cast<mirage::type::FPType *>(dmm->fp_base_ptr[gpu_id] +
                                                 output_tensors[0].fp_offset);
    compute_reduction_fingerprint<<<num_blocks, num_threads_per_blk>>>(
        input_fp_ptr,
        output_fp_ptr,
        num_elements,
        reduction_factor,
        input_stride,
        output_stride);
    checkCUDA(cudaDeviceSynchronize());
  }
  return true;
}
#endif // MIRAGE_FINGERPRINT_USE_CUDA

} // namespace kernel
} // namespace mirage
