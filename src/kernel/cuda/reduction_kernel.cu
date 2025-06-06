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

template <typename DT>
__global__ void execute_reduction(DT *input_ptr,
                                  DT *output_ptr,
                                  int num_input_elements,
                                  int num_output_elements) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  DT sum = static_cast<DT>(0.0f);
  if (idx < num_output_elements) {
    for (int i = 0; i < num_input_elements; i += num_output_elements) {
      sum += input_ptr[i];
    }
    output_ptr[idx] = sum;
  }
}

bool KNReductionOp::profile(ProfileResult &result) {
  assert(false);
  // assert a single GPU
  // assert(kgraph->gpu_dim.x == 1);
  int gpu_id = 0;
  // checkCUDA(cudaSetDevice(0));

  assert(input_tensors[0].data_type == DT_FLOAT16);
  assert(output_tensors[0].data_type == DT_FLOAT16);
  mirage::kernel::DeviceMemoryManager *dmm =
      mirage::kernel::DeviceMemoryManager::get_instance();
  cutlass::half_t *input_ptr = reinterpret_cast<cutlass::half_t *>(
      dmm->data_base_ptr[gpu_id] + input_tensors[0].data_offset);
  cutlass::half_t *output_ptr = reinterpret_cast<cutlass::half_t *>(
      dmm->data_base_ptr[gpu_id] + output_tensors[0].data_offset);
  int num_input_elements = input_tensors[0].num_elements();
  int num_output_elements = output_tensors[0].num_elements();
  int const num_threads_per_blk = 1024;
  int num_blocks =
      (num_output_elements + num_threads_per_blk - 1) / num_threads_per_blk;
  checkCUDA(cudaDeviceSynchronize());
  cudaEvent_t events[2];
  checkCUDA(cudaEventCreate(&events[0]));
  checkCUDA(cudaEventCreate(&events[1]));
  checkCUDA(cudaEventRecord(events[0]));
  for (int i = 0; i < ProfileResult::NUM_ITERATIONS; i++) {
    execute_reduction<<<num_blocks, num_threads_per_blk>>>(
        input_ptr, output_ptr, num_input_elements, num_output_elements);
  }
  float runtime_ms = 0;
  checkCUDA(cudaEventRecord(events[1]));
  checkCUDA(cudaEventSynchronize(events[1]));
  checkCUDA(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));
  result.run_time = runtime_ms / ProfileResult::NUM_ITERATIONS;
  printf("Reduction: runtime(%.8lfms)\n", result.run_time);
  checkCUDA(cudaEventDestroy(events[0]));
  checkCUDA(cudaEventDestroy(events[1]));

  return true;
}

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
      if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        // printf("result(%d) output_stride(%d) input_stride(%d) i(%d), n(%d) "
        //        "m(%d) k(%d)\n",
        //        result,
        //        output_stride,
        //        input_stride,
        //        i,
        //        n,
        //        m,
        //        k);
      }
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

} // namespace kernel
} // namespace mirage
