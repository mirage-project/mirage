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
#include "../common/worker_config.h"
namespace kernel {
namespace v2 {

template <typename T, int BATCH_SIZE, int CHUNK_SIZE, int OUTPUT_DIM_SIZE>
__device__ __forceinline__ void
    embedding_kernel_hopper(void const *__restrict__ input_ptr,
                            void const *__restrict__ embedding_ptr,
                            void *__restrict__ output_ptr) {
  if (threadIdx.x >= CONSUMER_NUM_THREADS) {
    return;
  }
  int64_t const *__restrict__ input_ids =
      static_cast<int64_t const *>(input_ptr);
  T const *__restrict__ embedding = static_cast<T const *>(embedding_ptr);
  T *__restrict__ output = static_cast<T *>(output_ptr);

  // 16B-vectorized row copy: the compute owns only 128 threads (4 warps),
  // half of v1's 256-thread worker, so per-thread efficiency must carry the
  // difference — 8 bf16 per access instead of 1 (V2_TODO.md #16/#18).
  // Vector path requires 16B-aligned bases; row strides preserve alignment
  // because OUTPUT_DIM_SIZE elements are a multiple of 16B. Falls back to
  // the scalar copy otherwise.
  constexpr int VEC = 16 / sizeof(T);
  bool const aligned =
      ((reinterpret_cast<uintptr_t>(embedding) |
        reinterpret_cast<uintptr_t>(output)) & 15) == 0;
  if constexpr (CHUNK_SIZE % VEC == 0 && OUTPUT_DIM_SIZE % VEC == 0) {
    if (aligned) {
      constexpr int NVEC = CHUNK_SIZE / VEC;
#pragma unroll
      for (int batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {
        int64_t wordIdx = input_ids[batch_idx];
        uint4 *__restrict__ out_v =
            reinterpret_cast<uint4 *>(output + batch_idx * OUTPUT_DIM_SIZE);
        if (wordIdx >= 0) {
          uint4 const *__restrict__ emb_v = reinterpret_cast<uint4 const *>(
              embedding + wordIdx * OUTPUT_DIM_SIZE);
#pragma unroll
          for (int i = threadIdx.x; i < NVEC; i += CONSUMER_NUM_THREADS) {
            out_v[i] = emb_v[i];
          }
        } else {
          for (int i = threadIdx.x; i < NVEC; i += CONSUMER_NUM_THREADS) {
            out_v[i] = make_uint4(0, 0, 0, 0);
          }
        }
      }
      return;
    }
  }
#pragma unroll
  for (int batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {
    int64_t wordIdx = input_ids[batch_idx];
    if (wordIdx >= 0) {
#pragma unroll
      for (int i = threadIdx.x; i < CHUNK_SIZE; i += CONSUMER_NUM_THREADS) {
        output[batch_idx * OUTPUT_DIM_SIZE + i] =
            embedding[wordIdx * OUTPUT_DIM_SIZE + i];
      }
    } else {
      // TODO: This might not be necessary
      for (int i = threadIdx.x; i < CHUNK_SIZE;
           i += CONSUMER_NUM_THREADS) { // writing 0 to output
        output[batch_idx * OUTPUT_DIM_SIZE + i] = T(0.0f);
      }
    }
  }
}

} // namespace v2
} // namespace kernel
