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
#include "../common/utils.cuh"
namespace kernel {

template <typename T, int BATCH_SIZE, int HIDDEN_DIM, int NUM_THREADS = 256>
__device__ __forceinline__ void rms_norm_hopper_impl(void const *input_ptr,
                                                     void const *weight_ptr,
                                                     void *output_ptr,
                                                     float eps) {
  // static_assert(BATCH_SIZE == 1);
  extern __shared__ char smem[];
  static_assert(HIDDEN_DIM % NUM_THREADS == 0);
  constexpr int ELTS_PER_THREAD = HIDDEN_DIM / NUM_THREADS;
  constexpr int BYTES_PER_THREAD = ELTS_PER_THREAD * sizeof(T);
  constexpr int BYTES_PER_CP = []() {
    if constexpr (BYTES_PER_THREAD % 16 == 0) {
      return 16; // 128bit copy-async
    } else if constexpr (BYTES_PER_THREAD % 8 == 0) {
      return 8; // 64bit copy-async
    } else {
      static_assert(BYTES_PER_THREAD % 4 == 0);
      return 4; // 32bit copy-async
    }
  }();
  constexpr int CHUNK_SIZE = BYTES_PER_CP / sizeof(T);
  constexpr int TILE_SIZE = NUM_THREADS * CHUNK_SIZE;
  static_assert(HIDDEN_DIM % TILE_SIZE == 0);
  constexpr int NUM_TILES = HIDDEN_DIM / TILE_SIZE;
  constexpr int NUM_CHUNKS_OUTPUT =
      HIDDEN_DIM / CHUNK_SIZE; // NUM_CHUNKS_OUTPUT per batch
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;

  // T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  // T *__restrict__ d_output = static_cast<T *>(output_ptr);

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

  for (int batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {
    // get the current batch input, weight, and output pointers
    T const *__restrict__ curr_d_input =
        static_cast<T const *>(input_ptr) + batch_idx * HIDDEN_DIM;
    T *__restrict__ curr_d_output =
        static_cast<T *>(output_ptr) + batch_idx * HIDDEN_DIM;
    // Warm up input tiles for the first atoms
    {
      load_smem<T, BYTES_PER_CP>(shared_input_buffer + threadIdx.x * CHUNK_SIZE,
                                 curr_d_input + threadIdx.x * CHUNK_SIZE);
      load_smem<T, BYTES_PER_CP>(shared_weight_buffer +
                                     threadIdx.x * CHUNK_SIZE,
                                 d_weight + threadIdx.x * CHUNK_SIZE);
      cp_async_fence();
    }

    float sum = 0.0f;
#pragma unroll
    for (int for_idx = 0; for_idx < NUM_TILES; for_idx++) {
      // copy
      if (for_idx + 1 < NUM_TILES) {
        load_smem<T, BYTES_PER_CP>(shared_input_buffer +
                                       threadIdx.x * CHUNK_SIZE +
                                       (for_idx + 1) * TILE_SIZE,
                                   curr_d_input + threadIdx.x * CHUNK_SIZE +
                                       (for_idx + 1) * TILE_SIZE);
        load_smem<T, BYTES_PER_CP>(
            shared_weight_buffer + threadIdx.x * CHUNK_SIZE +
                (for_idx + 1) * TILE_SIZE,
            d_weight + threadIdx.x * CHUNK_SIZE + (for_idx + 1) * TILE_SIZE);
        cp_async_fence();
        cp_async_wait<1>();
      } else if (for_idx + 1 == NUM_TILES) {
        cp_async_wait<0>();
      }
      __syncthreads();
#pragma unroll
      for (int i = threadIdx.x; i < TILE_SIZE; i += NUM_THREADS) {
        float val = (float)shared_input_buffer[for_idx * TILE_SIZE + i];
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
    sum = threadIdx.x < NUM_WARPS ? reduce_smem[threadIdx.x] : 0.0f;
#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
      sum += shfl_xor_sync(sum, offset);
    }
    if (threadIdx.x == 0) {
      reduce_smem[0] = sum;
    }
    __syncthreads();

    float rms_rcp = rsqrt(reduce_smem[0] / float(HIDDEN_DIM) + eps);

#pragma unroll
    for (int i = threadIdx.x; i < HIDDEN_DIM; i += NUM_THREADS) {
      float val = (float)shared_input_buffer[i];
      float w = (float)shared_weight_buffer[i];
      val *= rms_rcp * w;
      shared_output_buffer[i] = (T)val;
    }
    __syncthreads();
#pragma unroll
    for (int i = threadIdx.x; i < NUM_CHUNKS_OUTPUT; i += NUM_THREADS) {
      if constexpr (BYTES_PER_CP == 16) {
        *((__uint128_t *)((void *)&curr_d_output[i * CHUNK_SIZE])) =
            *((__uint128_t *)((void *)&shared_output_buffer[i * CHUNK_SIZE]));
      } else if constexpr (BYTES_PER_CP == 8) {
        *((uint64_t *)((void *)&curr_d_output[i * CHUNK_SIZE])) =
            *((uint64_t *)((void *)&shared_output_buffer[i * CHUNK_SIZE]));
      } else { // BYTES_PER_CP == 4
        *((uint32_t *)((void *)&curr_d_output[i * CHUNK_SIZE])) =
            *((uint32_t *)((void *)&shared_output_buffer[i * CHUNK_SIZE]));
      }
    }
  } // end batch_idx
}

} // namespace kernel
