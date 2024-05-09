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
#include "mirage/kernel/element_binary.h"
#include "mirage/kernel/graph.h"
#include "mirage/utils/cuda_helper.h"
#include "mirage/utils/hash_utils.h"
#include "cutlass/fast_math.h"
#include <cassert>

namespace mirage {
namespace kernel {

using namespace mirage::type;

template <typename DT>
__global__ void execute_elementbinary(mirage::type::KNOperatorType type,
                                      DT *input1_ptr,
                                      DT *input2_ptr,
                                      DT *output_ptr,
                                      int factor1,
                                      int factor2,
                                      int num_elements) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_elements) {
    DT operand_A = input1_ptr[i / factor1];
    DT operand_B = input2_ptr[i / factor2];
    if (type == mirage::type::KN_ADD_OP) {
      output_ptr[i] = operand_A + operand_B;
    } else if (type == mirage::type::KN_MUL_OP) {
      output_ptr[i] = operand_A * operand_B;
    } else if (type == mirage::type::KN_DIV_OP) {
      output_ptr[i] = operand_A / operand_B;
    } else {
      assert(false && "Unimplemented");
    }
  }
}

bool KNElementBinaryOp::profile(ProfileResult &result) {
  assert(input_tensors[0].data_type == DT_FLOAT16);
  assert(input_tensors[1].data_type == DT_FLOAT16);
  assert(output_tensors[0].data_type == DT_FLOAT16);
  cutlass::half_t *input1_ptr =
      static_cast<cutlass::half_t *>(input_tensors[0].data_ptr);
  cutlass::half_t *input2_ptr =
      static_cast<cutlass::half_t *>(input_tensors[1].data_ptr);
  cutlass::half_t *output_ptr =
      static_cast<cutlass::half_t *>(output_tensors[0].data_ptr);

  int num_elements = output_tensors[0].num_elements();
  int factor1 =
      output_tensors[0].num_elements() / input_tensors[0].num_elements();
  int factor2 =
      output_tensors[0].num_elements() / input_tensors[1].num_elements();
  int const num_threads_per_blk = 1024;
  int num_blocks =
      (num_elements + num_threads_per_blk - 1) / num_threads_per_blk;
  checkCUDA(cudaDeviceSynchronize());
  cudaEvent_t events[2];
  checkCUDA(cudaEventCreate(&events[0]));
  checkCUDA(cudaEventCreate(&events[1]));
  checkCUDA(cudaEventRecord(events[0]));
  for (int i = 0; i < 16; i++) {
    execute_elementbinary<<<num_blocks, num_threads_per_blk>>>(op_type,
                                                               input1_ptr,
                                                               input2_ptr,
                                                               output_ptr,
                                                               factor1,
                                                               factor2,
                                                               num_elements);
  }
  float runtime_ms = 0;
  checkCUDA(cudaEventRecord(events[1]));
  checkCUDA(cudaEventSynchronize(events[1]));
  checkCUDA(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));
  result.run_time = runtime_ms / 16;
  printf("ElementBinary: runtime(%.8lfms)\n", result.run_time);
  checkCUDA(cudaEventDestroy(events[0]));
  checkCUDA(cudaEventDestroy(events[1]));
  return true;
}

__global__ void
    compute_elementbinary_fingerprint(mirage::type::KNOperatorType type,
                                      FPType *div_p_lookup_table,
                                      FPType *div_q_lookup_table,
                                      mirage::kernel::DTensor input1,
                                      mirage::kernel::DTensor input2,
                                      mirage::kernel::DTensor output,
                                      int num_elements) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (type == mirage::type::KN_ADD_OP) {
    if (i < num_elements) {
      int input1_stride = 1, input1_idx = 0;
      int input2_stride = 1, input2_idx = 0;
      for (int d = output.num_dims - 1; d >= 0; d--) {
        input1_idx += (i % input1.dim[d]) * input1_stride;
        input2_idx += (i % input2.dim[d]) * input2_stride;
        input1_stride *= input1.dim[d];
        input2_stride *= input2.dim[d];
        i /= output.dim[d];
      }
      uint32_t x = input1.fp_ptr[input1_idx];
      uint32_t y = input2.fp_ptr[input2_idx];
      uint32_t z = (x + y) % FP_PQ;
      output.fp_ptr[threadIdx.x + blockIdx.x * blockDim.x] = z;
      // printf("add: output[%d] = %d input1[%d] = %d input2[%d] = %d\n",
      //     threadIdx.x + blockIdx.x * blockDim.x, z % FP_PQ,
      //     input1_idx, x, input2_idx, y);
    }
  } else if (type == mirage::type::KN_MUL_OP) {
    if (i < num_elements) {
      int input1_stride = 1, input1_idx = 0;
      int input2_stride = 1, input2_idx = 0;
      for (int d = output.num_dims - 1; d >= 0; d--) {
        input1_idx += (i % input1.dim[d]) * input1_stride;
        input2_idx += (i % input2.dim[d]) * input2_stride;
        input1_stride *= input1.dim[d];
        input2_stride *= input2.dim[d];
        i /= output.dim[d];
      }
      uint32_t x = input1.fp_ptr[input1_idx];
      uint32_t y = input2.fp_ptr[input2_idx];
      uint32_t z = (x * y) % FP_PQ;
      output.fp_ptr[threadIdx.x + blockIdx.x * blockDim.x] = z;
      // printf("add: output[%d] = %d input1[%d] = %d input2[%d] = %d\n",
      //     threadIdx.x + blockIdx.x * blockDim.x, z % FP_PQ,
      //     input1_idx, x, input2_idx, y);
    }
  } else if (type == mirage::type::KN_DIV_OP) {
    if (i < num_elements) {
      int input1_stride = 1, input1_idx = 0;
      int input2_stride = 1, input2_idx = 0;
      for (int d = output.num_dims - 1; d >= 0; d--) {
        input1_idx += (i % input1.dim[d]) * input1_stride;
        input2_idx += (i % input2.dim[d]) * input2_stride;
        input1_stride *= input1.dim[d];
        input2_stride *= input2.dim[d];
        i /= output.dim[d];
      }
      uint32_t x = input1.fp_ptr[input1_idx];
      uint32_t y = input2.fp_ptr[input2_idx];
      uint32_t z =
          (x % FP_P) * div_p_lookup_table[y % FP_P] * FP_Q_MUL_P_MOD_1 +
          (x % FP_Q) * div_q_lookup_table[y % FP_Q] * FP_P_MUL_Q_MOD_1;
      output.fp_ptr[threadIdx.x + blockIdx.x * blockDim.x] = z % FP_PQ;
      // printf("div: output[%d] = %d input1[%d] = %d input2[%d] = %d\n",
      //     threadIdx.x + blockIdx.x * blockDim.x, z % FP_PQ,
      //     input1_idx, x, input2_idx, y);
    }
  } else {
    assert(false && "Unimplemented");
  }
}

bool KNElementBinaryOp::fingerprint(void) {
  assert(input_tensors[0].num_dims == output_tensors[0].num_dims);
  for (int i = 0; i < output_tensors[0].num_dims; i++) {
    if (input_tensors[0].dim[i] != output_tensors[0].dim[i]) {
      assert(input_tensors[0].dim[i] == 1);
    }
  }
  assert(input_tensors[1].num_dims == output_tensors[0].num_dims);
  for (int i = 0; i < output_tensors[0].num_dims; i++) {
    if (input_tensors[1].dim[i] != output_tensors[0].dim[i]) {
      assert(input_tensors[1].dim[i] == 1);
    }
  }
  int num_elements = output_tensors[0].num_elements();
  int const num_threads_per_blk = 1024;
  int num_blocks =
      (num_elements + num_threads_per_blk - 1) / num_threads_per_blk;
  mirage::kernel::DeviceMemoryManager *dmm =
      mirage::kernel::DeviceMemoryManager::get_instance();
  compute_elementbinary_fingerprint<<<num_blocks, num_threads_per_blk>>>(
      op_type,
      dmm->div_p_lookup_table,
      dmm->div_q_lookup_table,
      input_tensors[0],
      input_tensors[1],
      output_tensors[0],
      num_elements);
  checkCUDA(cudaDeviceSynchronize());
  return true;
}

} // namespace kernel
} // namespace mirage
