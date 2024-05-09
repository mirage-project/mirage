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
#include "mirage/utils/hash_utils.h"
#include <cassert>

namespace mirage {
namespace kernel {

using namespace mirage::type;

bool KNMatmulOp::profile(ProfileResult &result) {
  float alpha = 1.0f, beta = 0.0f;
  mirage::kernel::DeviceMemoryManager *dmm =
      mirage::kernel::DeviceMemoryManager::get_instance();
  void *A = input_tensors[0].data_ptr;
  void *B = input_tensors[1].data_ptr;
  void *C = output_tensors[0].data_ptr;
  int num_dims = input_tensors[0].num_dims;
  assert(input_tensors[1].num_dims == num_dims);
  assert(output_tensors[0].num_dims == num_dims);
  int batch = 1;
  for (int i = 0; i < num_dims - 2; i++) {
    assert(input_tensors[0].dim[i] == input_tensors[1].dim[i]);
    assert(input_tensors[0].dim[i] == output_tensors[0].dim[i]);
    batch *= input_tensors[0].dim[i];
  }
  int row_A = input_tensors[0].dim[num_dims - 2];
  int column_A = input_tensors[0].dim[num_dims - 1];
  int row_B = input_tensors[1].dim[num_dims - 2];
  int column_B = input_tensors[1].dim[num_dims - 1];
  int row_C = output_tensors[0].dim[num_dims - 2];
  int column_C = output_tensors[0].dim[num_dims - 1];
  assert(column_A == row_B);
  assert(row_C == row_A);
  assert(column_C == column_B);
  cudaDataType_t type_A =
      mirage::utils::to_cuda_datatype(input_tensors[0].data_type);
  cudaDataType_t type_B =
      mirage::utils::to_cuda_datatype(input_tensors[1].data_type);
  cudaDataType_t type_C =
      mirage::utils::to_cuda_datatype(output_tensors[0].data_type);
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  cublasOperation_t trans_A = CUBLAS_OP_N;
  cublasOperation_t trans_B = CUBLAS_OP_N;
  if (input_tensors[0].layout == layout::DmemColumnMajor) {
    trans_A = CUBLAS_OP_T;
  } else {
    assert(input_tensors[0].layout == layout::DmemRowMajor);
  }
  if (input_tensors[1].layout == layout::DmemColumnMajor) {
    trans_B = CUBLAS_OP_T;
  } else {
    assert(input_tensors[1].layout == layout::DmemRowMajor);
  }
  // Currently assume C must be in row major;
  assert(output_tensors[0].layout == layout::DmemRowMajor);
  int lda = input_tensors[0].layout == layout::DmemRowMajor ? row_A : column_A;
  int ldb = input_tensors[1].layout == layout::DmemRowMajor ? row_B : column_B;
  int ldc = row_C;

  checkCUDA(cudaDeviceSynchronize());
  cudaEvent_t events[2];
  checkCUDA(cudaEventCreate(&events[0]));
  checkCUDA(cudaEventCreate(&events[1]));
  checkCUDA(cudaEventRecord(events[0]));
  for (int i = 0; i < 16; i++) {
    if (batch == 1) {
      checkCUDA(cublasGemmEx(dmm->blas,
                             trans_A,
                             trans_B,
                             row_C,
                             column_C,
                             column_A,
                             &alpha,
                             A,
                             type_A,
                             lda,
                             B,
                             type_B,
                             ldb,
                             &beta,
                             C,
                             type_C,
                             ldc,
                             compute_type,
                             CUBLAS_GEMM_DEFAULT));
    } else {
      int strideA = row_A * column_A;
      int strideB = row_B * column_B;
      int strideC = row_C * column_C;
      checkCUDA(cublasGemmStridedBatchedEx(dmm->blas,
                                           trans_A,
                                           trans_B,
                                           row_C,
                                           column_C,
                                           column_A,
                                           &alpha,
                                           A,
                                           type_A,
                                           lda,
                                           strideA,
                                           B,
                                           type_B,
                                           ldb,
                                           strideB,
                                           &beta,
                                           C,
                                           type_C,
                                           ldc,
                                           strideC,
                                           batch,
                                           compute_type,
                                           CUBLAS_GEMM_DEFAULT));
    }
  }
  float runtime_ms = 0;
  checkCUDA(cudaEventRecord(events[1]));
  checkCUDA(cudaEventSynchronize(events[1]));
  checkCUDA(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));
  result.run_time = runtime_ms / 16;
  printf("BatchMatmul: runtime(%.8lfms)\n", result.run_time);
  checkCUDA(cudaEventDestroy(events[0]));
  checkCUDA(cudaEventDestroy(events[1]));
  return true;
}

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
      uint32_t result = 0;
      for (int i = 0; i < k; i++) {
        uint32_t A_value = A_ptr[b * mk + row_idx * k + i];
        uint32_t B_value = B_ptr[b * nk + i * n + col_idx];
        result = (result + A_value * B_value) % FP_PQ;
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
  compute_matmul_fingerprint<<<num_blocks, num_threads_per_blk>>>(
      input_tensors[0].fp_ptr,
      input_tensors[1].fp_ptr,
      output_tensors[0].fp_ptr,
      num_batches,
      row_C,
      column_C,
      row_B);
  checkCUDA(cudaDeviceSynchronize());
  return true;
}

} // namespace kernel
} // namespace mirage
