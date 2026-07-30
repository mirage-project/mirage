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
#include "../common/utils.cuh"
#include "utils.cuh"
namespace kernel {

template <typename T,
          typename InputSmem,
          int NUM_HEAD,
          int WINDOW_SIZE,
          int HEAD_DIM = 128,
          int NUM_THREADS = 128,
          int BARRIER_ID = 9>
__device__ __forceinline__ void rotary_embedding_hopper(InputSmem smem_input,
                                                        T const *cos_ptr,
                                                        T const *sin_ptr,
                                                        int token_offset = 0) {
  // Avoid sync divergence dead lock.
  static_assert(HEAD_DIM < NUM_THREADS || HEAD_DIM % NUM_THREADS == 0);
  constexpr int ROTARY_PARTICIPATING_THREADS =
      (NUM_THREADS < HEAD_DIM ? NUM_THREADS : HEAD_DIM);
#pragma unroll
  for (int win_idx = 0; win_idx < WINDOW_SIZE; ++win_idx) {

    int smem_seq_idx = token_offset + win_idx;

#pragma unroll
    for (int head_idx = 0; head_idx < NUM_HEAD; ++head_idx) {

      T const *cur_cos_ptr = cos_ptr + win_idx * HEAD_DIM;
      T const *cur_sin_ptr = sin_ptr + win_idx * HEAD_DIM;

      // NeoX rotation pairs column i with i +/- HEAD_DIM/2, so it is a
      // read-modify-write across threads: every partner must be READ before
      // any write-back. When HEAD_DIM > NUM_THREADS a thread owns several
      // columns and this loop runs more than once, so writing back inside the
      // loop made the second trip read a partner the first trip had already
      // rotated. head_dim 128 == NUM_THREADS is a single trip, which is why
      // shipped models were correct and Qwen3.5's head_dim 256 is the first
      // shape to expose it. Staging in registers between the two barriers
      // fixes it for any HEAD_DIM that is a multiple of NUM_THREADS, and is
      // the SAME sequence of operations when HEAD_DIM <= NUM_THREADS.
      constexpr int COLS_PER_THREAD =
          (HEAD_DIM + NUM_THREADS - 1) / NUM_THREADS;
      float v_rot[COLS_PER_THREAD];
      int slot = 0;

      wg_sync<ROTARY_PARTICIPATING_THREADS>(BARRIER_ID);
#pragma unroll
      for (uint32_t i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS, ++slot) {
        int offset = (i / HEAD_DIM) * HEAD_DIM + i;

        int row = smem_seq_idx * NUM_HEAD + head_idx;
        int col = i;

        float cos = static_cast<float>(cur_cos_ptr[offset]);
        float sin = static_cast<float>(cur_sin_ptr[offset]);

        if (i < HEAD_DIM / 2) {
          float v1 = static_cast<float>(smem_input.at(row, col));
          float v2 = static_cast<float>(smem_input.at(row, col + HEAD_DIM / 2));
          v_rot[slot] = v1 * cos - v2 * sin;
        } else {
          float v1 = static_cast<float>(smem_input.at(row, col));
          float v2 = static_cast<float>(smem_input.at(row, col - HEAD_DIM / 2));
          v_rot[slot] = v1 * cos + v2 * sin;
        }
      }
      wg_sync<ROTARY_PARTICIPATING_THREADS>(BARRIER_ID);
      slot = 0;
#pragma unroll
      for (uint32_t i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS, ++slot) {
        int row = smem_seq_idx * NUM_HEAD + head_idx;
        smem_input.at(row, i) = static_cast<T>(v_rot[slot]);
      }
    }
  }
}

} // namespace kernel
