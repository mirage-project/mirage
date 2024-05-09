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
#include "mirage/kernel/element_unary.h"
#include "mirage/kernel/graph.h"
#include "mirage/utils/cuda_helper.h"
#include "mirage/utils/hash_utils.h"
#include "cutlass/fast_math.h"
#include <cassert>

namespace mirage {
namespace kernel {

using namespace mirage::type;

template <typename DT>
__global__ void execute_elementunary(mirage::type::KNOperatorType type,
                                     DT *input_ptr,
                                     DT *output_ptr,
                                     int num_elements) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (type == mirage::type::KN_EXP_OP) {
    if (i < num_elements) {
      output_ptr[i] = cutlass::fast_exp(input_ptr[i]);
    }
  } else {
    assert(false && "Unimplemented");
  }
}

bool KNElementUnaryOp::profile(ProfileResult &result) {
  // TODO: to be implemented
  assert(input_tensors[0].num_elements() == output_tensors[0].num_elements());
  assert(input_tensors[0].data_type == DT_FLOAT16);
  assert(output_tensors[0].data_type == DT_FLOAT16);
  cutlass::half_t *input_ptr =
      static_cast<cutlass::half_t *>(input_tensors[0].data_ptr);
  cutlass::half_t *output_ptr =
      static_cast<cutlass::half_t *>(output_tensors[0].data_ptr);
  int num_elements = input_tensors[0].num_elements();
  int const num_threads_per_blk = 1024;
  int num_blocks =
      (num_elements + num_threads_per_blk - 1) / num_threads_per_blk;
  checkCUDA(cudaDeviceSynchronize());
  cudaEvent_t events[2];
  checkCUDA(cudaEventCreate(&events[0]));
  checkCUDA(cudaEventCreate(&events[1]));
  checkCUDA(cudaEventRecord(events[0]));
  for (int i = 0; i < ProfileResult::NUM_ITERATIONS; i++) {
    execute_elementunary<<<num_blocks, num_threads_per_blk>>>(
        op_type, input_ptr, output_ptr, num_elements);
  }
  float runtime_ms = 0;
  checkCUDA(cudaEventRecord(events[1]));
  checkCUDA(cudaEventSynchronize(events[1]));
  checkCUDA(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));
  result.run_time = runtime_ms / ProfileResult::NUM_ITERATIONS;
  printf("ElementUnary: runtime(%.8lfms)\n", result.run_time);
  checkCUDA(cudaEventDestroy(events[0]));
  checkCUDA(cudaEventDestroy(events[1]));
  return true;
}

__global__ void compute_elementunary_fingerprint(mirage::type::KNOperatorType type,
                                                 FPType *exp_lookup_table,
                                                 mirage::type::FPType *input_ptr,
                                                 mirage::type::FPType *output_ptr,
                                                 int num_elements) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (type == mirage::type::KN_EXP_OP) {
    if (i < num_elements) {
      mirage::type::FPType val = input_ptr[i];
      mirage::type::FPType q_residual = val % FP_Q;
      uint32_t result = exp_lookup_table[q_residual];
      result = (result * FP_Q_MUL_P_MOD_1) % FP_PQ;
      output_ptr[i] = result;
    }
  } else {
    assert(false && "Unimplemented");
  }
}

bool KNElementUnaryOp::fingerprint(void) {
  assert(input_tensors[0].num_elements() == output_tensors[0].num_elements());
  int num_elements = input_tensors[0].num_elements();
  int const num_threads_per_blk = 1024;
  int num_blocks =
      (num_elements + num_threads_per_blk - 1) / num_threads_per_blk;
  mirage::kernel::DeviceMemoryManager *dmm =
      mirage::kernel::DeviceMemoryManager::get_instance();
  compute_elementunary_fingerprint<<<num_blocks, num_threads_per_blk>>>(
      op_type,
      dmm->exp_lookup_table,
      input_tensors[0].fp_ptr,
      output_tensors[0].fp_ptr,
      num_elements);
  checkCUDA(cudaDeviceSynchronize());
  return true;
}

} // namespace kernel
} // namespace mirage
