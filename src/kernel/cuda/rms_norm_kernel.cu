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
#include "mirage/kernel/rms_norm.h"
#include "mirage/utils/cuda_helper.h"
#include "mirage/utils/fingerprint_functions.h"
#include "mirage/utils/hash_utils.h"
#include <cassert>

namespace mirage {
namespace kernel {

using namespace mirage::utils;

bool KNRMSNormOp::profile(ProfileResult &result) {
  // TODO: add profile results
  return true;
}

__global__ void compute_rms_norm_fingerprint(FPType *input_ptr,
                                             FPType *output_ptr,
                                             FPType *div_p_lookup_table,
                                             FPType *div_q_lookup_table,
                                             FPType *sqrt_p_lookup_table,
                                             FPType *sqrt_q_lookup_table,
                                             int num_samples,
                                             int norm_size) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_samples) {
    FPType square_sum = 0;
    for (int k = 0; k < norm_size; k++) {
      FPType x = input_ptr[i * norm_size + k];
      x = compute_mul_fingerprint(x, x);
      square_sum = compute_add_fingerprint(square_sum, x);
    }
    // Compute rooted mean square
    FPType rms = 0;
    {
      FPType x = square_sum;
      FPType n = norm_size % FP_PQ;
      // Compute z = x / n
      FPType z =
          compute_div_fingerprint(x, n, div_p_lookup_table, div_q_lookup_table);
      // Perform sqrt for root-mean-square
      rms =
          compute_sqrt_fingerprint(z, sqrt_p_lookup_table, sqrt_q_lookup_table);
    }
    for (int k = 0; k < norm_size; k++) {
      FPType x = input_ptr[i * norm_size + k];
      // Compute x / rms
      FPType z = compute_div_fingerprint(
          x, rms, div_p_lookup_table, div_q_lookup_table);
      output_ptr[i * norm_size + k] = z;
    }
  }
}

bool KNRMSNormOp::fingerprint(void) {
  // assert a 1-D GPU mesh
  assert(kgraph->gpu_dim.y == 1);
  assert(kgraph->gpu_dim.z == 1);
  int num_samples = output_tensors[0].num_elements() / normalized_size;
  int const num_threads_per_blk = 128;
  int num_blocks =
      (num_samples + num_threads_per_blk - 1) / num_threads_per_blk;
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
    compute_rms_norm_fingerprint<<<num_blocks, num_threads_per_blk>>>(
        input_fp_ptr,
        output_fp_ptr,
        dmm->div_p_lookup_table,
        dmm->div_q_lookup_table,
        dmm->sqrt_p_lookup_table,
        dmm->sqrt_q_lookup_table,
        num_samples,
        normalized_size);
    checkCUDA(cudaDeviceSynchronize());
  }
  return true;
}

} // namespace kernel
} // namespace mirage
