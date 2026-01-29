/* Copyright 2025 CMU
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
#pragma once
#include "tasks/common/common_header.cuh"

#ifdef USE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

namespace kernel {

#ifdef USE_NVSHMEM

/**
 * NVSHMEM-based Allgather using put operations.
 * To achieve allreduce, we also need reduction after allgather.
 */
template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
__device__ __forceinline__ void
    nvshmem_allgather_strided_put(void *buffer_ptr,
                                  void *local_data_ptr,
                                  void *sig_addr,
                                  size_t event_index,
                                  int target_gpu_id) {
  // TODO(Zepeng): remove event_index
  // TODO(Zepeng): transfer only active data
#pragma unroll
  for (int i = 0; i < BATCH_SIZE; i++) {
    nvshmemx_putmem_nbi_block(reinterpret_cast<char *>(buffer_ptr) +
                                  i * OUTPUT_STRIDE * sizeof(T),
                              reinterpret_cast<char *>(local_data_ptr) +
                                  i * OUTPUT_STRIDE * sizeof(T),
                              OUTPUT_SIZE * sizeof(T),
                              target_gpu_id);
  }

  nvshmem_quiet();
  __syncthreads();
  if (threadIdx.x == 0) {
    nvshmemx_signal_op(reinterpret_cast<uint64_t *>(sig_addr),
                      1,
                      NVSHMEM_SIGNAL_ADD,
                      target_gpu_id);
  }
}

// Assume that input/buffer/output have the same stride
template <typename T,
          int NUM_GPUS,
          int MY_GPU_ID,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE>
__device__ __forceinline__ void reduction_kernel(void const *input_ptr,
                                                 void const *buf_ptr,
                                                 void *output_ptr) {
  // We must force memory order here before reading data from other GPUs.
  // TODO(Zepeng): reduce only active data
  nvshmem_quiet();
  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_buffer = static_cast<T const *>(buf_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);
  for (int idx = threadIdx.x; idx < OUTPUT_SIZE * BATCH_SIZE;
       idx += blockDim.x) {
    float accum = 0.0;
    int batch = idx / OUTPUT_SIZE;
    int offset = idx % OUTPUT_SIZE;
    for (int i = 0; i < NUM_GPUS; i++) {
      if (i == MY_GPU_ID) {
        accum += static_cast<float>(d_input[batch * OUTPUT_STRIDE + offset]);
      } else {
        accum += static_cast<float>(d_buffer[i * BATCH_SIZE * OUTPUT_STRIDE +
                                             batch * OUTPUT_STRIDE + offset]);
      }
    }
    d_output[batch * OUTPUT_STRIDE + offset] = static_cast<T>(accum);
  }
}

#endif // USE_NVSHMEM

// Reduction function resides in reduction.cuh

} // namespace kernel