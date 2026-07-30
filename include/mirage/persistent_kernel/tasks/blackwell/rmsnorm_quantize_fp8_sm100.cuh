/* Copyright 2026 CMU
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
#include "per_token_group_quantize_fp8.cuh"

#include <cstdint>
#include <cuda_fp8.h>

namespace kernel {

// ---------------------------------------------------------------------------
// RMS norm + fp32-block-scale FP8 quantize, in ONE task.  M4-I9 flag A.
//
// WHY IT EXISTS.  Every layer's chain opens `rms_norm -> quantize -> dense_fp8`,
// and the two front tasks have IDENTICAL geometry: `rmsnorm_layer` splits dim 0
// by grid.x and `quantize_fp8_layer` does the same via
// `row_partition = (0,-1,-1)`, so both are one task per token row with the same
// [1, HIDDEN] tile.  Merging them removes one record from all 40 layers'
// dependency chains, and M4-I8 measured what a chain record costs beyond its own
// duration: ~1.15 us of event visibility or ~1.55 us of queue-pop latency plus a
// barrier pair.
//
// BIT-EXACT BY CONSTRUCTION, AND THE ARGUMENT IS STRUCTURAL RATHER THAN
// ARITHMETIC.  `rms_norm_hopper_impl` already stages its bf16 OUTPUT in shared
// memory (`shared_output_buffer[i] = (T)val`) and then copies smem -> global.
// So the bytes the standalone quantize would read back out of global are already
// sitting in shared memory when the norm finishes.  This task therefore
//
//   * runs the norm half as the EXACT code of `rms_norm_hopper_impl` -- same
//     tiling, same cp.async warm-up, same two-stage reduction, same
//     `(T)val` rounding, same smem->global store when the bf16 output is still
//     wanted; and
//   * runs the quantize half by CALLING `per_token_group_quantize_fp8_task_impl`
//     itself, the same function the standalone task calls, at the same
//     instantiation the standalone task uses at this site
//     (`BATCH_SIZE = 1, HIDDEN_SIZE = HIDDEN_DIM, GROUP_SIZE = 128,
//     GLOBAL_STRIDE = HIDDEN_DIM, SCALE_UE8M0 = false`), with its input pointer
//     redirected from global to the shared staging buffer.
//
// The only difference between the two arms is therefore the ADDRESS SPACE of one
// load.  No rounding position moves, so the CAST-POSITION RULE does not apply:
// the fp8 bytes are computed from the same bf16 values, rounded at the same
// place, and the amax/scale/clamp arithmetic is not re-derived here at all --
// it is the same instantiated function.
//
// NO EXTRA BARRIER.  The `__syncthreads()` the norm already executes between
// filling `shared_output_buffer` and copying it out is the same barrier the
// quantize needs, so the fused task has exactly the norm's barrier count.  That
// matters: M4-I8's arm O measured ~470 ns of makespan per extra scoped load +
// barrier pair per chain record.
//
// SOUNDNESS under the persistent work-queue scheduler (the M3-I3 test): each
// task owns a disjoint row, both halves are row-local, and the amax reduction
// stays inside one warp -- no cross-task reduction, no arrival counter, no
// co-residency requirement.
//
// WRITE_NORM selects whether the bf16 norm is still materialised.  It is needed
// at GDN layers (the bf16 `ba` projection reads it) and not at attention layers
// (the quantize was its only consumer), which is why the task has either 3 or 2
// outputs.
// ---------------------------------------------------------------------------
template <int BATCH_SIZE,
          int HIDDEN_DIM,
          int GROUP_SIZE,
          bool WRITE_NORM,
          typename T,
          typename DST_T,
          int NUM_THREADS = 256>
__device__ __forceinline__ void rms_norm_quantize_fp8_task_impl(
    void const *input_ptr,
    void const *weight_ptr,
    void *norm_out_ptr,
    void *out_q_ptr,
    void *out_s_ptr,
    float eps,
    float const q_eps,
    float const min_8bit,
    float const max_8bit) {
  extern __shared__ char smem[];
  static_assert(HIDDEN_DIM % NUM_THREADS == 0);
  static_assert(HIDDEN_DIM % GROUP_SIZE == 0,
                "a scale group must not straddle the row boundary");
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
  constexpr int NUM_CHUNKS_OUTPUT = HIDDEN_DIM / CHUNK_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
  constexpr int NUM_GROUPS_PER_ROW = HIDDEN_DIM / GROUP_SIZE;

  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);

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
    T const *__restrict__ curr_d_input =
        static_cast<T const *>(input_ptr) + batch_idx * HIDDEN_DIM;
    // ---- the norm half: rms_norm_hopper_impl, unchanged -----------------
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
    // The norm's OWN barrier, reused: it already separates the fill above from
    // the smem->global copy below, and it is exactly the barrier the quantize
    // half needs before reading a group a different warp may have written.
    __syncthreads();

    if constexpr (WRITE_NORM) {
      T *__restrict__ curr_d_output =
          static_cast<T *>(norm_out_ptr) + batch_idx * HIDDEN_DIM;
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
    }

    // ---- the quantize half: the SAME function, same instantiation, input
    //      pointer redirected from global to the staging buffer -------------
    per_token_group_quantize_fp8_task_impl</*BATCH_SIZE=*/1,
                                           /*HIDDEN_SIZE=*/HIDDEN_DIM,
                                           GROUP_SIZE,
                                           /*GLOBAL_STRIDE=*/HIDDEN_DIM,
                                           T,
                                           DST_T,
                                           /*SCALE_UE8M0=*/false>(
        shared_output_buffer,
        static_cast<DST_T *>(out_q_ptr) + batch_idx * HIDDEN_DIM,
        static_cast<float *>(out_s_ptr) + batch_idx * NUM_GROUPS_PER_ROW,
        q_eps,
        min_8bit,
        max_8bit,
        /*scale_outer_stride=*/1);
  } // end batch_idx
}

} // namespace kernel
