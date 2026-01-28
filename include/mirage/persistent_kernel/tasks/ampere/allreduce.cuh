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
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE>
__device__ __forceinline__ void nvshmem_allgather_strided_put(
    void* buffer_ptr,
    void* local_data_ptr,
    void* sig_addr,
    size_t event_index,
    int target_gpu_id) {
  #pragma unroll
  for (int i = 0; i < BATCH_SIZE; i++) {
      nvshmemx_putmem_signal_block(
          reinterpret_cast<char*>(buffer_ptr) + i * OUTPUT_STRIDE * sizeof(T),
          reinterpret_cast<char*>(local_data_ptr) + i * OUTPUT_STRIDE * sizeof(T),
          OUTPUT_SIZE * sizeof(T),
          reinterpret_cast<uint64_t *>(sig_addr),
          1 /*signal*/,
          NVSHMEM_SIGNAL_ADD,
          target_gpu_id);
  }
}

#endif // USE_NVSHMEM

// Reduction function resides in reduction.cuh

} // namespace kernel