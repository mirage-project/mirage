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

// DeepSeek V4 C4 KV Compressor / Cache Insert for SM100.
//
// Correctness-first DeepSeek V4 Flash Base CSA C4 compressor/cache insert.
// This keeps the math simple and explicit so it can serve as the reference MPK
// implementation before a later performance-specialized kernel.
//
// DeepSeek V4 Flash Base constants targeted by this first task:
//   compress_ratio = 4
//   overlap        = true
//   head_dim       = 512
//   rope_head_dim  = 64
//   nope_head_dim  = 448
//   coff           = 2
//   kv_score_dim   = 4 * head_dim = 2048
//
// Input pointer contract:
//   input_ptrs[0] kv_score:
//     float32 [max_num_batched_tokens, KV_SCORE_DIM]
//     v1 intentionally fixes fp32 inputs to match the official HF Compressor
//     math. A later fused projection path may add bf16/fp8 input variants.
//
//   input_ptrs[1] token_meta:
//     int32 [max_num_batched_tokens, 2]
//     token_meta[token, 0] = absolute sequence position
//     token_meta[token, 1] = physical C4 cache slot, or -1 if this token does
//                            not emit a compressed KV entry.
//
//   input_ptrs[2] state_cache:
//     float32 [max_requests, 8, KV_SCORE_DIM]
//     Stores raw C4 overlap/current state. Scores are stored without APE and
//     APE is added during compression, following the SGLang-style packed state
//     while remaining mathematically equivalent to HF Compressor.forward.
//
//   input_ptrs[3] c4_cache:
//     bf16 [num_c4_pages, C4_PAGE_SIZE, HEAD_DIM]
//     Correctness-first cache format. A later performance pass should switch
//     to FlashMLA-compatible FP8-with-scale cache:
//       512 fp8 NoPE bytes + 4 fp32 scales + 64 bf16 RoPE values.
//
//   input_ptrs[4] ape:
//     float32 [8, HEAD_DIM]
//     Prepacked from the official HF ape [4, 1024]:
//       rows 0..3 = overlap half, rows 4..7 = current half.
//
//   input_ptrs[5] norm_weight:
//     float32 [HEAD_DIM]
//
//   input_ptrs[6] rope_cos_sin:
//     float32 [max_seq_len, ROPE_HEAD_DIM], GPT-J/interleaved style layout
//     with first half cos and second half sin.
//
// Runtime metadata:
//   qo_indptr_buffer[request_id:request_id+2] selects this request's token
//   window in kv_score/token_meta, matching existing MPK MLA gather tasks.
//
// DeepSeek official semantics, from HF inference/model.py Compressor:
//   1. Prefill: cutoff = seqlen - (seqlen % 4); remainder tokens stay in state.
//   2. C4 overlap transform: each compressed block uses 8 slots:
//      previous block's 4 overlap slots + current block's 4 current slots.
//   3. The first block has invalid overlap; use KV = 0 and score = -inf.
//   4. Decode: should_compress = ((absolute_position + 1) % 4 == 0).
//   5. Add APE before softmax.
//   6. Compute stable softmax over the 8 scores per hidden dimension and form
//      the weighted KV sum.
//   7. Apply RMSNorm after weighted pooling.
//   8. Apply RoPE only to the last ROPE_HEAD_DIM elements at position
//      absolute_position + 1 - 4.
//   9. Write compressed KV to c4_cache[token_meta[token, 1]].
//  10. After a write, shift current state into overlap state for the next C4.
//
// Implementation references:
//   - DeepSeek HF inference/model.py: Compressor.forward exact math.
//   - SGLang deepseek_v4 c4.cuh: 8-slot window and online softmax shape.
//   - vLLM deepseek_compressor.py: state cache and fused insert metadata.
//   - FlashMLA README: future physical index and FP8 KV cache format.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace kernel {

template <int HEAD_DIM, int ROPE_HEAD_DIM, int KV_SCORE_DIM, int C4_PAGE_SIZE>
__device__ __forceinline__ void dsv4_c4_compress_sm100_task_impl(
    void const *kv_score_ptr,
    void const *token_meta_ptr,
    void *state_cache_ptr,
    void *c4_cache_ptr,
    void const *ape_ptr,
    void const *norm_weight_ptr,
    void const *rope_cos_sin_ptr,
    int const *qo_indptr_buffer_ptr,
    int request_id) {
  static_assert(HEAD_DIM == 512,
                "DeepSeek V4 Flash Base C4 v1 only supports head_dim=512");
  static_assert(ROPE_HEAD_DIM == 64,
                "DeepSeek V4 Flash Base C4 v1 only supports rope_dim=64");
  static_assert(KV_SCORE_DIM == 4 * HEAD_DIM,
                "C4 kv_score layout must be [kv_overlap, kv, score_overlap, score]");
  static_assert(C4_PAGE_SIZE > 0, "C4 cache page size must be positive");

  constexpr int COMPRESS_RATIO = 4;
  constexpr int NUM_C4_SLOTS = 2 * COMPRESS_RATIO;
  constexpr int NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM;
  constexpr float RMS_EPS = 1.0e-6f;

  int const tid = threadIdx.x;
  int const num_threads = blockDim.x;

  float const *kv_score = reinterpret_cast<float const *>(kv_score_ptr);
  int const *token_meta = reinterpret_cast<int const *>(token_meta_ptr);
  float *state_cache = reinterpret_cast<float *>(state_cache_ptr);
  __nv_bfloat16 *c4_cache = reinterpret_cast<__nv_bfloat16 *>(c4_cache_ptr);
  float const *ape = reinterpret_cast<float const *>(ape_ptr);
  float const *norm_weight = reinterpret_cast<float const *>(norm_weight_ptr);
  float const *rope_cos_sin = reinterpret_cast<float const *>(rope_cos_sin_ptr);

  int const token_begin = qo_indptr_buffer_ptr[request_id];
  int const token_end = qo_indptr_buffer_ptr[request_id + 1];
  int const num_tokens = token_end - token_begin;
  if (token_end <= token_begin) {
    return;
  }

  float *request_state =
      state_cache + request_id * NUM_C4_SLOTS * KV_SCORE_DIM;

  __shared__ float pooled[HEAD_DIM];
  __shared__ float normed[HEAD_DIM];
  __shared__ float sumsq_partial[256];

  for (int token_idx = token_begin; token_idx < token_end; ++token_idx) {
    int const abs_pos = token_meta[token_idx * 2 + 0];
    int const c4_slot = token_meta[token_idx * 2 + 1];
    if (abs_pos < 0) {
      continue;
    }

    int const current_row = COMPRESS_RATIO + (abs_pos % COMPRESS_RATIO);
    float const *token_kv_score = kv_score + token_idx * KV_SCORE_DIM;
    float *current_state = request_state + current_row * KV_SCORE_DIM;

    for (int d = tid; d < KV_SCORE_DIM; d += num_threads) {
      current_state[d] = token_kv_score[d];
    }
    __syncthreads();

    if (c4_slot < 0) {
      continue;
    }

    bool const first_c4_block = (abs_pos + 1 == COMPRESS_RATIO);
    for (int d = tid; d < HEAD_DIM; d += num_threads) {
      float score_values[NUM_C4_SLOTS];
      float kv_values[NUM_C4_SLOTS];

#pragma unroll
      for (int slot = 0; slot < NUM_C4_SLOTS; ++slot) {
        if (first_c4_block && slot < COMPRESS_RATIO) {
          kv_values[slot] = 0.0f;
          score_values[slot] = -CUDART_INF_F;
        } else {
          float const *slot_state = request_state + slot * KV_SCORE_DIM;
          if (slot < COMPRESS_RATIO) {
            kv_values[slot] = slot_state[d];
            score_values[slot] = slot_state[2 * HEAD_DIM + d];
          } else {
            kv_values[slot] = slot_state[HEAD_DIM + d];
            score_values[slot] = slot_state[3 * HEAD_DIM + d];
          }
          score_values[slot] += ape[slot * HEAD_DIM + d];
        }
      }

      float max_score = score_values[0];
#pragma unroll
      for (int slot = 1; slot < NUM_C4_SLOTS; ++slot) {
        max_score = fmaxf(max_score, score_values[slot]);
      }

      float denom = 0.0f;
      float weighted_sum = 0.0f;
#pragma unroll
      for (int slot = 0; slot < NUM_C4_SLOTS; ++slot) {
        float const weight = __expf(score_values[slot] - max_score);
        denom += weight;
        weighted_sum += kv_values[slot] * weight;
      }
      pooled[d] = weighted_sum / denom;
    }
    __syncthreads();

    float local_sumsq = 0.0f;
    for (int d = tid; d < HEAD_DIM; d += num_threads) {
      float const value = pooled[d];
      local_sumsq += value * value;
    }
    sumsq_partial[tid] = local_sumsq;
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        sumsq_partial[tid] += sumsq_partial[tid + stride];
      }
      __syncthreads();
    }

    float const inv_rms = rsqrtf(sumsq_partial[0] / HEAD_DIM + RMS_EPS);
    for (int d = tid; d < HEAD_DIM; d += num_threads) {
      normed[d] = pooled[d] * inv_rms * norm_weight[d];
    }
    __syncthreads();

    int const rope_pos = abs_pos + 1 - COMPRESS_RATIO;
    int const c4_page = c4_slot / C4_PAGE_SIZE;
    int const c4_offset = c4_slot % C4_PAGE_SIZE;
    __nv_bfloat16 *cache_row =
        c4_cache + (c4_page * C4_PAGE_SIZE + c4_offset) * HEAD_DIM;

    for (int d = tid; d < NOPE_HEAD_DIM; d += num_threads) {
      cache_row[d] = __float2bfloat16(normed[d]);
    }
    for (int pair = tid; pair < ROPE_HEAD_DIM / 2; pair += num_threads) {
      int const even_d = NOPE_HEAD_DIM + 2 * pair;
      int const odd_d = even_d + 1;
      float const even = normed[even_d];
      float const odd = normed[odd_d];
      float const cos_v = rope_cos_sin[rope_pos * ROPE_HEAD_DIM + pair];
      float const sin_v =
          rope_cos_sin[rope_pos * ROPE_HEAD_DIM + ROPE_HEAD_DIM / 2 + pair];
      cache_row[even_d] = __float2bfloat16(even * cos_v - odd * sin_v);
      cache_row[odd_d] = __float2bfloat16(odd * cos_v + even * sin_v);
    }
    __syncthreads();

    for (int d = tid; d < COMPRESS_RATIO * KV_SCORE_DIM; d += num_threads) {
      request_state[d] = request_state[COMPRESS_RATIO * KV_SCORE_DIM + d];
    }
    __syncthreads();
  }

  // HF prefill (start_pos == 0, seqlen > 1) leaves only the last complete
  // block in overlap rows and any trailing remainder in current rows. Decode
  // updates are single-token windows and keep current rows until overwritten.
  if (num_tokens > 1) {
    int const remainder = num_tokens % COMPRESS_RATIO;
    int const cutoff = num_tokens - remainder;
    for (int d = tid; d < COMPRESS_RATIO * KV_SCORE_DIM; d += num_threads) {
      request_state[COMPRESS_RATIO * KV_SCORE_DIM + d] = 0.0f;
    }
    __syncthreads();
    for (int rem = 0; rem < remainder; ++rem) {
      float const *src = kv_score + (token_begin + cutoff + rem) * KV_SCORE_DIM;
      float *dst = request_state + (COMPRESS_RATIO + rem) * KV_SCORE_DIM;
      for (int d = tid; d < KV_SCORE_DIM; d += num_threads) {
        dst[d] = src[d];
      }
    }
    __syncthreads();
  }
}

} // namespace kernel
