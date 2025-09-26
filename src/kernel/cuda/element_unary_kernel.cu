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

#ifdef MIRAGE_FINGERPRINT_USE_CUDA
__constant__ float CLAMP_MIN_MAX_DEVICE[2];

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
#endif // MIRAGE_FINGERPRINT_USE_CUDA

} // namespace kernel
} // namespace mirage
