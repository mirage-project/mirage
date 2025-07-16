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
#include "common.h"
#include "utils.cuh"
namespace kernel {
template <typename T,
          typename InputSmem,
          int NUM_HEAD,
          int WINDOW_SIZE,
          int HEAD_DIM = 128>
__device__ __forceinline__ void rms_norm(InputSmem smem_input,
                                         T const *weight_ptr,
                                         float *reduce_smem,
                                         float eps,
                                         int token_offset = 0,
                                         bool rotary_emd = false,
                                         T const *cos_ptr = nullptr,
                                         T const *sin_ptr = nullptr) {
  // For __syncthread divergence dead lock.
  static_assert(NUM_THREADS <= HEAD_DIM || HEAD_DIM % 32 == 0);
  // smem_input: NUM_HEADS * (WINDOW_SIZE or CHUNK_SIZE), HEAD_DIM
  // TODO(Wenqin): handle if speculative window of k span two chunks.
  int warp_idx = warp_id();
#pragma unroll
  for (int win_idx = 0; win_idx < WINDOW_SIZE; ++win_idx) {
    // token_offset is the offset for first token in input SMEM (auto-agressive
    // decoding or speculative decoding window), for Q, token_offset is always
    // 0. For K, token_offset is the offset of k (except speculative window) in
    // the SMEM chunk.
    int smem_seq_idx = token_offset + win_idx;
#pragma unroll
    for (int head_idx = 0; head_idx < NUM_HEAD; ++head_idx) {
      float sum = 0.0f;
#pragma unroll
      for (uint32_t i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
        int row = smem_seq_idx * NUM_HEAD + head_idx;
        int col = i;
        float val = (float)smem_input.at(row, col);
        sum += val * val;
      }

#pragma unroll
      for (uint32_t offset = NUM_THREADS_PER_WARP / 2; offset > 0;
           offset /= 2) {
        sum += shfl_xor_sync(sum, offset);
      }

      if (threadIdx.x % 32 == 0) {
        reduce_smem[warp_idx] = sum;
      }
      __syncthreads();
      sum = threadIdx.x < NUM_WARPS ? reduce_smem[threadIdx.x] : 0.0f;

#pragma unroll
      for (uint32_t offset = NUM_THREADS_PER_WARP / 2; offset > 0;
           offset /= 2) {
        sum += shfl_xor_sync(sum, offset);
      }

      if (threadIdx.x == 0) {
        reduce_smem[0] = sum;
      }

      __syncthreads();

      float rms_rcp = rsqrt(reduce_smem[0] / float(HEAD_DIM) + eps);

      // multiply with weight
      if (threadIdx.x < HEAD_DIM) {
#pragma unroll
        for (uint32_t i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
          int row = smem_seq_idx * NUM_HEAD + head_idx;
          int col = i;
          float val = (float)smem_input.at(row, col);
          float w = (float)weight_ptr[i];
          val *= rms_rcp * w;
          smem_input.at(row, col) = (T)val;

          if (rotary_emd) {
            // we should do rope for all the window size q and k, because they
            // came from hidden states, we didn't apply rope yet.
            __syncthreads();
            int offset = (i / HEAD_DIM) * HEAD_DIM + i;
            T const *cur_cos_ptr = cos_ptr + win_idx * HEAD_DIM;
            T const *cur_sin_ptr = sin_ptr + win_idx * HEAD_DIM;
            float cos = (float)cur_cos_ptr[offset];
            float sin = (float)cur_sin_ptr[offset];

            float v_rot;
            if (i < HEAD_DIM / 2) {
              float v1 = (float)smem_input.at(row, col);
              float v2 = (float)smem_input.at(row, col + HEAD_DIM / 2);
              v_rot = v1 * cos - v2 * sin;
            } else {
              float v1 = (float)smem_input.at(row, col);
              float v2 = (float)smem_input.at(row, col - HEAD_DIM / 2);
              v_rot = v1 * cos + v2 * sin;
            }
            __syncthreads();
            // output shape (window_size, head_num, head_dim)
            smem_input.at(row, col) = (T)v_rot;
          }
        } // i
      } else {
        // we should keep __syncthreads number same as the for loop when
        // HEAD_DIM smaller than NUM_THREAD
        if (rotary_emd) {
          __syncthreads();
          __syncthreads();
        }
      }
    } // head_idx
  }   // win_idx
}
} // namespace kernel