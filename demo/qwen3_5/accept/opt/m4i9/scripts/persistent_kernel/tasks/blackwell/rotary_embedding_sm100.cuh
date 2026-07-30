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
#include <cutlass/arch/barrier.h>

namespace kernel {

template <typename T,
          typename InputSmem,
          int NUM_HEAD,
          int WINDOW_SIZE,
          int HEAD_DIM = 128>
__device__ __forceinline__ void rotary_embedding_sm100(InputSmem smem_input,
                                                       T const *cos_ptr,
                                                       T const *sin_ptr,
                                                       int token_offset = 0) {
  static_assert(HEAD_DIM < NUM_THREADS || HEAD_DIM % NUM_THREADS == 0);
  constexpr int ROTARY_PARTICIPATING_THREADS =
      (NUM_THREADS < HEAD_DIM ? NUM_THREADS : HEAD_DIM);
  cutlass::arch::NamedBarrier wg_barrier_rotary(ROTARY_PARTICIPATING_THREADS,
                                                /*bar-id*/ 6);
  if (threadIdx.x < NUM_THREADS) {
#pragma unroll
    for (int win_idx = 0; win_idx < WINDOW_SIZE; ++win_idx) {

      int smem_seq_idx = token_offset + win_idx;

#pragma unroll
      for (int head_idx = 0; head_idx < NUM_HEAD; ++head_idx) {

        T const *cur_cos_ptr = cos_ptr + win_idx * HEAD_DIM;
        T const *cur_sin_ptr = sin_ptr + win_idx * HEAD_DIM;

#pragma unroll
        for (uint32_t i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
          int offset = (i / HEAD_DIM) * HEAD_DIM + i;

          int row = smem_seq_idx * NUM_HEAD + head_idx;
          int col = i;

          float cos = static_cast<float>(cur_cos_ptr[offset]);
          float sin = static_cast<float>(cur_sin_ptr[offset]);

          float v_rot;

          wg_barrier_rotary.arrive_and_wait();
          if (i < HEAD_DIM / 2) {
            float v1 = static_cast<float>(smem_input.at(row, col));
            float v2 =
                static_cast<float>(smem_input.at(row, col + HEAD_DIM / 2));
            v_rot = v1 * cos - v2 * sin;
          } else {
            float v1 = static_cast<float>(smem_input.at(row, col));
            float v2 =
                static_cast<float>(smem_input.at(row, col - HEAD_DIM / 2));
            v_rot = v1 * cos + v2 * sin;
          }
          wg_barrier_rotary.arrive_and_wait();
          smem_input.at(row, col) = static_cast<T>(v_rot);
        }
      }
    }
  }
}

} // namespace kernel
