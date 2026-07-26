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
          int HEAD_DIM,
          int CONSUMER_WARPGROUP_SYNC_BARRIER_ID = 6,
          int ROTARY_SYNC_BARRIER_ID = 7>
__device__ __forceinline__ void rms_norm_sm100(InputSmem smem_input,
                                               T const *weight_ptr,
                                               float *reduce_smem,
                                               float eps,
                                               int window_size,
                                               int token_offset = 0,
                                               bool rotary_emd = false,
                                               T const *cos_ptr = nullptr,
                                               T const *sin_ptr = nullptr) {
  cutlass::arch::NamedBarrier wg_barrier(
      NUM_THREADS, /*bar-id*/ CONSUMER_WARPGROUP_SYNC_BARRIER_ID);
  constexpr int ROTARY_PARTICIPATING_THREADS =
      (NUM_THREADS < HEAD_DIM ? NUM_THREADS : HEAD_DIM);
  cutlass::arch::NamedBarrier wg_barrier_rotary(
      ROTARY_PARTICIPATING_THREADS,
      /*bar-id*/ ROTARY_SYNC_BARRIER_ID);
  if (threadIdx.x < NUM_THREADS) {
    // Avoid sync divergence dead lock.
    if (rotary_emd) {
      static_assert(HEAD_DIM < NUM_THREADS || HEAD_DIM % NUM_THREADS == 0);
    }
    // smem_input: NUM_HEADS * (WINDOW_SIZE or CHUNK_SIZE), HEAD_DIM
    // TODO(Wenqin): handle if speculative window of k span two chunks.
    int warp_idx = warp_id();
#pragma unroll
    for (int win_idx = 0; win_idx < window_size; ++win_idx) {
      // token_offset is the offset for first token in input SMEM
      // (auto-agressive decoding or speculative decoding window), for Q,
      // token_offset is always 0. For K, token_offset is the offset of k
      // (except speculative window) in the SMEM chunk.
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
        wg_barrier.arrive_and_wait();
        sum = threadIdx.x < NUM_WARPS ? reduce_smem[threadIdx.x] : 0.0f;

#pragma unroll
        for (uint32_t offset = NUM_THREADS_PER_WARP / 2; offset > 0;
             offset /= 2) {
          sum += shfl_xor_sync(sum, offset);
        }

        if (threadIdx.x == 0) {
          reduce_smem[0] = sum;
        }

        wg_barrier.arrive_and_wait();

        float rms_rcp = rsqrt(reduce_smem[0] / float(HEAD_DIM) + eps);

        int const row = smem_seq_idx * NUM_HEAD + head_idx;

        // multiply with weight -- for EVERY column of this head, before any
        // rotation reads a partner column (see the note below).
#pragma unroll
        for (uint32_t i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
          int col = i;
          float val = (float)smem_input.at(row, col);
          float w = (float)weight_ptr[i];
          val *= rms_rcp * w;
          smem_input.at(row, col) = (T)val;
        }

        if (rotary_emd) {
          // we should do rope for all the window size q and k, because they
          // came from hidden states, we didn't apply rope yet.
          //
          // NeoX rotation pairs column i with i +/- HEAD_DIM/2, so it is a
          // read-modify-write across threads and needs BOTH:
          //   (a) every column of this head normalised before any read, and
          //   (b) every partner read completed before any write-back.
          // When HEAD_DIM > NUM_THREADS each thread owns several columns and
          // the loop below runs more than once. Fusing the rotation into the
          // normalisation loop (as this code used to) breaks (a) and (b) at
          // once: on its first trip a thread read a partner that had not been
          // normalised yet, and on its second trip it read a partner that had
          // already been rotated. head_dim 128 == NUM_THREADS is a single
          // trip, which is why every shipped model was correct and Qwen3.5
          // (head_dim 256, docs/qwen35/vllm-graph.md §2.2.1) is the first
          // shape to expose it.
          //
          // Staging the rotated values in registers between the two barriers
          // fixes it for any HEAD_DIM that is a multiple of NUM_THREADS. For
          // HEAD_DIM <= NUM_THREADS this is the SAME operations in the same
          // order with the same barrier count, so existing models stay
          // bit-identical.
          constexpr int COLS_PER_THREAD =
              (HEAD_DIM + NUM_THREADS - 1) / NUM_THREADS;
          T const *cur_cos_ptr = cos_ptr + win_idx * HEAD_DIM;
          T const *cur_sin_ptr = sin_ptr + win_idx * HEAD_DIM;
          float v_rot[COLS_PER_THREAD];

          wg_barrier_rotary.arrive_and_wait();
          int slot = 0;
#pragma unroll
          for (uint32_t i = threadIdx.x; i < HEAD_DIM;
               i += NUM_THREADS, ++slot) {
            int col = i;
            float cos = (float)cur_cos_ptr[i];
            float sin = (float)cur_sin_ptr[i];
            if (i < HEAD_DIM / 2) {
              float v1 = (float)smem_input.at(row, col);
              float v2 = (float)smem_input.at(row, col + HEAD_DIM / 2);
              v_rot[slot] = v1 * cos - v2 * sin;
            } else {
              float v1 = (float)smem_input.at(row, col);
              float v2 = (float)smem_input.at(row, col - HEAD_DIM / 2);
              v_rot[slot] = v1 * cos + v2 * sin;
            }
          }
          wg_barrier_rotary.arrive_and_wait();
          slot = 0;
#pragma unroll
          for (uint32_t i = threadIdx.x; i < HEAD_DIM;
               i += NUM_THREADS, ++slot) {
            // output shape (window_size, head_num, head_dim)
            smem_input.at(row, i) = (T)v_rot[slot];
          }
        }
      } // head_idx
    }   // win_idx
  }
}
} // namespace kernel
