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
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/element_unary.h"
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

__constant__ float CLAMP_MIN_MAX_DEVICE[2];

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
  } else if (type == mirage::type::KN_SQUARE_OP) {
    if (i < num_elements) {
      output_ptr[i] = input_ptr[i] * input_ptr[i];
    }
  } else if (type == mirage::type::KN_SQRT_OP) {
    if (i < num_elements) {
      output_ptr[i] = cutlass::fast_sqrt(input_ptr[i]);
    }
  } else if (type == mirage::type::KN_SILU_OP) {
    if (i < num_elements) {
      DT x = input_ptr[i];
      output_ptr[i] = x / (1.0f + cutlass::fast_exp(-x));
    }
  } else if (type == mirage::type::KN_GELU_OP) {
    if (i < num_elements) {
      DT x = input_ptr[i];
      output_ptr[i] = (x / 2.0f) * (1.0f + erff(x / sqrtf(2.0f)));
    }
  } else if (type == mirage::type::KN_RELU_OP) {
    if (i < num_elements) {
      DT x = input_ptr[i];
      if (x > 0.0f) {
        output_ptr[i] = x;
      } else {
        output_ptr[i] = 0.0f;
      }
    }
  } else if (type == mirage::type::KN_CLAMP_OP) {
    if (i < num_elements) {
      DT x = input_ptr[i];
      if (x < CLAMP_MIN_MAX_DEVICE[0]) {
        output_ptr[i] = CLAMP_MIN_MAX_DEVICE[0];
      } else if (x > CLAMP_MIN_MAX_DEVICE[1]) {
        output_ptr[i] = CLAMP_MIN_MAX_DEVICE[1];
      } else {
        output_ptr[i] = x;
      }
    }
  } else {
    assert(false && "Unimplemented");
  }
}

bool KNElementUnaryOp::profile(ProfileResult &result) {
  // Only launch kernel on a single GPU for profiling
  // checkCUDA(cudaSetDevice(0));
  assert(input_tensors[0].num_elements() == output_tensors[0].num_elements());
  assert(input_tensors[0].data_type == DT_FLOAT16);
  assert(output_tensors[0].data_type == DT_FLOAT16);
  mirage::kernel::DeviceMemoryManager *dmm =
      mirage::kernel::DeviceMemoryManager::get_instance();
  cutlass::half_t *input_ptr = reinterpret_cast<cutlass::half_t *>(
      dmm->data_base_ptr[0] + input_tensors[0].data_offset);
  cutlass::half_t *output_ptr = reinterpret_cast<cutlass::half_t *>(
      dmm->data_base_ptr[0] + output_tensors[0].data_offset);
  int num_elements = input_tensors[0].num_elements();
  int const num_threads_per_blk = 1024;
  int num_blocks =
      (num_elements + num_threads_per_blk - 1) / num_threads_per_blk;
  checkCUDA(cudaDeviceSynchronize());
  cudaEvent_t events[2];
  checkCUDA(cudaEventCreate(&events[0]));
  checkCUDA(cudaEventCreate(&events[1]));
  checkCUDA(cudaEventRecord(events[0]));

  if (op_type == mirage::type::KNOperatorType::KN_CLAMP_OP) {
    KNClampUnaryOp *clamp_op = dynamic_cast<KNClampUnaryOp *>(this);
    float CLAMP_MIN_MAX_HOST[2] = {clamp_op->min_val, clamp_op->max_val};
    cudaMemcpyToSymbol(
        CLAMP_MIN_MAX_DEVICE, CLAMP_MIN_MAX_HOST, sizeof(float) * 2);
  }

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

__global__ void
    compute_elementunary_fingerprint(mirage::type::KNOperatorType type,
                                     FPType *exp_lookup_table,
                                     FPType *sqrt_p_lookup_table,
                                     FPType *sqrt_q_lookup_table,
                                     mirage::type::FPType *input_ptr,
                                     mirage::type::FPType *output_ptr,
                                     int num_elements) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_elements) {
    if (type == mirage::type::KN_EXP_OP) {
      output_ptr[i] = compute_exp_fingerprint(input_ptr[i], exp_lookup_table);
    } else if (type == mirage::type::KN_SQUARE_OP) {
      output_ptr[i] = compute_square_fingerprint(input_ptr[i]);
    } else if (type == mirage::type::KN_SQRT_OP) {
      output_ptr[i] = compute_sqrt_fingerprint(
          input_ptr[i], sqrt_p_lookup_table, sqrt_q_lookup_table);
    } else if (type == mirage::type::KN_SILU_OP) {
      output_ptr[i] = compute_silu_fingerprint(input_ptr[i], exp_lookup_table);
    } else if (type == mirage::type::KN_GELU_OP) {
      output_ptr[i] = compute_gelu_fingerprint(input_ptr[i], exp_lookup_table);
    } else if (type == mirage::type::KN_RELU_OP) {
      output_ptr[i] = compute_relu_fingerprint(input_ptr[i]);
    } else if (type == mirage::type::KN_CLAMP_OP) {
      output_ptr[i] = compute_clamp_fingerprint(input_ptr[i]);
    } else {
      assert(false && "Unimplemented");
    }
  }
}

bool KNElementUnaryOp::fingerprint(void) {
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
  // Use GPU dmm->gpu_id for computing fingerprint
  checkCUDA(cudaSetDevice(dmm->gpu_id));

  for (int gpu_id = 0; gpu_id < kgraph->gpu_dim.x; gpu_id++) {
    mirage::type::FPType *input_fp_ptr =
        reinterpret_cast<mirage::type::FPType *>(dmm->fp_base_ptr[gpu_id] +
                                                 input_tensors[0].fp_offset);
    mirage::type::FPType *output_fp_ptr =
        reinterpret_cast<mirage::type::FPType *>(dmm->fp_base_ptr[gpu_id] +
                                                 output_tensors[0].fp_offset);
    compute_elementunary_fingerprint<<<num_blocks, num_threads_per_blk>>>(
        op_type,
        dmm->exp_lookup_table,
        dmm->sqrt_p_lookup_table,
        dmm->sqrt_q_lookup_table,
        input_fp_ptr,
        output_fp_ptr,
        num_elements);
    checkCUDA(cudaDeviceSynchronize());
  }
  return true;
}

} // namespace kernel
} // namespace mirage
