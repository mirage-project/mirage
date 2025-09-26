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
#include "mirage/kernel/matmul.h"
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
__global__ void compute_matmul_fingerprint(mirage::type::FPType *A_ptr,
                                           mirage::type::FPType *B_ptr,
                                           mirage::type::FPType *C_ptr,
                                           int num_batches,
                                           int m,
                                           int n,
                                           int k) {
  int row_idx = (threadIdx.x + blockIdx.x * blockDim.x) / n;
  int col_idx = (threadIdx.x + blockIdx.x * blockDim.x) % n;
  int mk = m * k;
  int mn = m * n;
  int nk = n * k;
  if (row_idx < m) {
    for (int b = 0; b < num_batches; b++) {
      mirage::type::FPType result = 0;
      for (int i = 0; i < k; i++) {
        mirage::type::FPType x = A_ptr[b * mk + row_idx * k + i];
        mirage::type::FPType y = B_ptr[b * nk + i * n + col_idx];
        mirage::type::FPType z = utils::compute_mul_fingerprint(x, y);
        result = utils::compute_add_fingerprint(result, z);
      }
      if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        // printf("C[%d] = %d\n",
        //        b * mn + threadIdx.x + blockIdx.x * blockDim.x,
        //        result);
      }
      C_ptr[b * mn + threadIdx.x + blockIdx.x * blockDim.x] = result;
    }
  }
}

bool KNMatmulOp::fingerprint(void) {
  // Currently assert a single GPU
  assert(kgraph->gpu_dim.y == 1);
  assert(kgraph->gpu_dim.z == 1);

  int num_dims = input_tensors[0].num_dims;
  int row_A = input_tensors[0].dim[num_dims - 2];
  int column_A = input_tensors[0].dim[num_dims - 1];
  int row_B = input_tensors[1].dim[num_dims - 2];
  int column_B = input_tensors[1].dim[num_dims - 1];
  int row_C = output_tensors[0].dim[num_dims - 2];
  int column_C = output_tensors[0].dim[num_dims - 1];
  assert(column_A == row_B);
  assert(row_C == row_A);
  assert(column_C == column_B);
  int num_batches = 1;
  for (int i = 0; i < num_dims - 2; i++) {
    num_batches *= input_tensors[0].dim[i];
  }
  int const num_threads_per_blk = 1024;
  int num_blocks =
      (row_C * column_C + num_threads_per_blk - 1) / num_threads_per_blk;
  mirage::kernel::DeviceMemoryManager *dmm =
      mirage::kernel::DeviceMemoryManager::get_instance();
  // Use GPU dmm->gpu_id for computing fingerprint
  checkCUDA(cudaSetDevice(dmm->gpu_id));

  for (int gpu_id = 0; gpu_id < kgraph->gpu_dim.x; gpu_id++) {
    mirage::type::FPType *A_fp_ptr = reinterpret_cast<mirage::type::FPType *>(
        dmm->fp_base_ptr[gpu_id] + input_tensors[0].fp_offset);
    mirage::type::FPType *B_fp_ptr = reinterpret_cast<mirage::type::FPType *>(
        dmm->fp_base_ptr[gpu_id] + input_tensors[1].fp_offset);
    mirage::type::FPType *C_fp_ptr = reinterpret_cast<mirage::type::FPType *>(
        dmm->fp_base_ptr[gpu_id] + output_tensors[0].fp_offset);
    compute_matmul_fingerprint<<<num_blocks, num_threads_per_blk>>>(
        A_fp_ptr, B_fp_ptr, C_fp_ptr, num_batches, row_C, column_C, row_B);
    checkCUDA(cudaDeviceSynchronize());
  }
  return true;
}
#endif // MIRAGE_FINGERPRINT_USE_CUDA

} // namespace kernel
} // namespace mirage
