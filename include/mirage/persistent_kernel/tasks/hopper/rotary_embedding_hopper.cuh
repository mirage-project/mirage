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
          int BARRIER_ID = 9,
          // Partial RoPE (e.g. GLM-4.6: 64 of 128 dims): rotate dims
          // [0, ROTARY_DIM) pairing i <-> i + ROTARY_DIM/2, pass the rest
          // through. cos/sin rows are ROTARY_DIM wide.
          int ROTARY_DIM = HEAD_DIM>
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

      T const *cur_cos_ptr = cos_ptr + win_idx * ROTARY_DIM;
      T const *cur_sin_ptr = sin_ptr + win_idx * ROTARY_DIM;

#pragma unroll
      for (uint32_t i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
        int row = smem_seq_idx * NUM_HEAD + head_idx;
        int col = i;

        float v_rot = 0.0f;

        wg_sync<ROTARY_PARTICIPATING_THREADS>(BARRIER_ID);

        if (i < ROTARY_DIM / 2) {
          float cos = static_cast<float>(cur_cos_ptr[i]);
          float sin = static_cast<float>(cur_sin_ptr[i]);
          float v1 = static_cast<float>(smem_input.at(row, col));
          float v2 =
              static_cast<float>(smem_input.at(row, col + ROTARY_DIM / 2));
          v_rot = v1 * cos - v2 * sin;
        } else if (i < ROTARY_DIM) {
          float cos = static_cast<float>(cur_cos_ptr[i]);
          float sin = static_cast<float>(cur_sin_ptr[i]);
          float v1 = static_cast<float>(smem_input.at(row, col));
          float v2 =
              static_cast<float>(smem_input.at(row, col - ROTARY_DIM / 2));
          v_rot = v1 * cos + v2 * sin;
        }
        wg_sync<ROTARY_PARTICIPATING_THREADS>(BARRIER_ID);
        if (i < ROTARY_DIM) {
          smem_input.at(row, col) = static_cast<T>(v_rot);
        }
      }
    }
  }
}

// rotary_embedding_hopper's math (the same products, sums and roundings) for
// all NUM_HEAD heads of one token at once: cos / sin read once, every head's
// pair read before one sync and written after it (2 syncs, not 2 a head).
// HEAD_DIM threads at most one column each.
template <typename T,
          typename InputSmem,
          int NUM_HEAD,
          int HEAD_DIM,
          int NUM_THREADS,
          int BARRIER_ID,
          int ROTARY_DIM = HEAD_DIM>
__device__ __forceinline__ void rotary_embedding_hopper_heads(
    InputSmem smem_input, T const *cos_ptr, T const *sin_ptr, int token) {
  static_assert(HEAD_DIM <= NUM_THREADS,
                "rotary_embedding_hopper_heads: a column a thread");
  uint32_t const i = threadIdx.x;
  if (i >= HEAD_DIM) {
    return;
  }
  float cos = 0.f, sin = 0.f;
  if (i < ROTARY_DIM) {
    cos = static_cast<float>(cos_ptr[i]);
    sin = static_cast<float>(sin_ptr[i]);
  }
  float v_rot[NUM_HEAD];
  wg_sync<HEAD_DIM>(BARRIER_ID);
#pragma unroll
  for (int h = 0; h < NUM_HEAD; ++h) {
    int const row = token * NUM_HEAD + h;
    v_rot[h] = 0.0f;
    if (i < ROTARY_DIM / 2) {
      float v1 = static_cast<float>(smem_input.at(row, i));
      float v2 = static_cast<float>(smem_input.at(row, i + ROTARY_DIM / 2));
      v_rot[h] = v1 * cos - v2 * sin;
    } else if (i < ROTARY_DIM) {
      float v1 = static_cast<float>(smem_input.at(row, i));
      float v2 = static_cast<float>(smem_input.at(row, i - ROTARY_DIM / 2));
      v_rot[h] = v1 * cos + v2 * sin;
    }
  }
  wg_sync<HEAD_DIM>(BARRIER_ID);
  if (i < ROTARY_DIM) {
#pragma unroll
    for (int h = 0; h < NUM_HEAD; ++h) {
      smem_input.at(token * NUM_HEAD + h, i) = static_cast<T>(v_rot[h]);
    }
  }
}

} // namespace kernel
