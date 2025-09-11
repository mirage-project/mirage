/* Copyright 2025 Mirage Team
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
#include "common.h"
#include "utils.cuh"
namespace kernel {

template <typename T, int BATCH_SIZE, int HIDDEN_DIM>
__device__ __forceinline__ void rms_norm_impl(void const *input_ptr,
                                              void const *weight_ptr,
                                              void *output_ptr,
                                              float eps) {
  static_assert(BATCH_SIZE == 1);
  extern __shared__ char smem[];
  constexpr int CHUNK_SIZE = 16 / sizeof(T); // 128b copy-async
  int const total_threads = blockDim.x;
  int const tile_size = total_threads * CHUNK_SIZE;
  assert(HIDDEN_DIM % tile_size == 0);
  int const num_tiles = HIDDEN_DIM / tile_size;
  int const num_chunks_output = BATCH_SIZE * HIDDEN_DIM / CHUNK_SIZE;
  int const num_warps = total_threads / NUM_THREADS_PER_WARP;

  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  // using InputDmem =
  //     dmem_row_const<T, BATCH_SIZE, HIDDEN_DIM, HIDDEN_DIM>;
  // using OutputDmem =
  //     dmem_row<T, BATCH_SIZE, HIDDEN_DIM, HIDDEN_DIM>;

  // InputDmem input_dmem(d_input);
  // OutputDmem output_dmem(d_output);

  constexpr size_t SHARED_WEIGHT_BUFFER_OFFSET = sizeof(T) * HIDDEN_DIM;
  constexpr size_t SHARED_OUTPUT_BUFFER_OFFSET =
      SHARED_WEIGHT_BUFFER_OFFSET + sizeof(T) * HIDDEN_DIM;
  constexpr size_t REDUCE_BUFFER_OFFSET =
      SHARED_OUTPUT_BUFFER_OFFSET + sizeof(T) * HIDDEN_DIM;
  T *shared_input_buffer = (T *)(smem);
  T *shared_weight_buffer = (T *)(smem + SHARED_WEIGHT_BUFFER_OFFSET);
  T *shared_output_buffer = (T *)(smem + SHARED_OUTPUT_BUFFER_OFFSET);
  float *reduce_smem = reinterpret_cast<float *>(smem + REDUCE_BUFFER_OFFSET);

  // Warm up input tiles for the first atoms
  {
    load_smem(shared_input_buffer + threadIdx.x * CHUNK_SIZE,
              d_input + threadIdx.x * CHUNK_SIZE);
    load_smem(shared_weight_buffer + threadIdx.x * CHUNK_SIZE,
              d_weight + threadIdx.x * CHUNK_SIZE);
    cp_async_fence();
  }

  float sum = 0.0f;
#pragma unroll
  for (int for_idx = 0; for_idx < num_tiles; for_idx++) {
    // copy
    if (for_idx + 1 < num_tiles) {
      load_smem(shared_input_buffer + threadIdx.x * CHUNK_SIZE +
                    (for_idx + 1) * tile_size,
                d_input + threadIdx.x * CHUNK_SIZE + (for_idx + 1) * tile_size);
      load_smem(shared_weight_buffer + threadIdx.x * CHUNK_SIZE +
                    (for_idx + 1) * tile_size,
                d_weight + threadIdx.x * CHUNK_SIZE +
                    (for_idx + 1) * tile_size);
      cp_async_fence();
      cp_async_wait<1>();
    } else if (for_idx + 1 == num_tiles) {
      cp_async_wait<0>();
    }
    __syncthreads();
#pragma unroll
    for (int i = threadIdx.x; i < tile_size; i += total_threads) {
      float val = (float)shared_input_buffer[for_idx * tile_size + i];
      sum += val * val;
    }
  }

#pragma unroll
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    sum += shfl_xor_sync(sum, offset);
  }
  if (threadIdx.x % 32 == 0) {
    reduce_smem[threadIdx.x / 32] = sum;
  }
  __syncthreads();
  sum = threadIdx.x < num_warps ? reduce_smem[threadIdx.x] : 0.0f;
#pragma unroll
  for (int offset = num_warps / 2; offset > 0; offset /= 2) {
    sum += shfl_xor_sync(sum, offset);
  }
  if (threadIdx.x == 0) {
    reduce_smem[0] = sum;
  }
  __syncthreads();

  float rms_rcp = rsqrt(reduce_smem[0] / float(HIDDEN_DIM) + eps);

#pragma unroll
  for (int i = threadIdx.x; i < HIDDEN_DIM; i += total_threads) {
    float val = (float)shared_input_buffer[i];
    float w = (float)shared_weight_buffer[i];
    val *= rms_rcp * w;
    shared_output_buffer[i] = (T)val;
  }
  __syncthreads();
#pragma unroll
  for (int i = threadIdx.x; i < num_chunks_output; i += total_threads) {
    *((__uint128_t *)((void *)&d_output[i * CHUNK_SIZE])) =
        *((__uint128_t *)((void *)&shared_output_buffer[i * CHUNK_SIZE]));
  }
}

} // namespace kernel