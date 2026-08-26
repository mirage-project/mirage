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

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace kernel {

// (user #2 FuseTensor row-swap): when Q_ROW_STRIDE_OVERRIDE > 0,
// the kernel addresses q_fused/q_split_pe with the row-swap layout instead
// of the legacy [head, FUSED_HEAD_DIM] layout:
//   addr = base + row * Q_ROW_STRIDE_OVERRIDE
//          + Q_PE_BASE_IN_ROW + head_idx * Q_PE_HEAD_STRIDE
// This is what the chunked-prefill fused Q buffer (H*192 wide per row with
// nope/pe split blocks) needs. Defaults preserve legacy behavior.
template <int NUM_HEADS,
          int TILE_Q,
          bool HAS_SPLIT_Q,
          bool DO_Q,
          bool DO_K,
          int FUSED_HEAD_DIM = 576,
          int ROPE_DIM = 64,
          int K_PE_STRIDE = 128,
          int Q_ROW_STRIDE_OVERRIDE = 0,
          int Q_PE_BASE_IN_ROW = 0,
          int Q_PE_HEAD_STRIDE = 0>
__device__ __forceinline__ void
    deepseek_mla_rope_sm100_task_impl(__nv_bfloat16 *__restrict__ q_fused,
                                      __nv_bfloat16 *__restrict__ q_split_pe,
                                      __nv_bfloat16 *__restrict__ k_pe,
                                      __nv_bfloat16 const *__restrict__ cos,
                                      __nv_bfloat16 const *__restrict__ sin,
                                      int const *__restrict__ qo_indptr,
                                      int const *__restrict__ request_ids,
                                      int const *__restrict__ step,
                                      int request_slot,
                                      int head_idx,
                                      int q_tile_idx) {
  int const req_id = request_ids[request_slot];
  if (req_id < 0 || head_idx < 0 || head_idx >= NUM_HEADS) {
    return;
  }
  static_assert(DO_Q || DO_K);

  int const qo_begin = qo_indptr[request_slot];
  int const qo_end = qo_indptr[request_slot + 1];
  int const q_len = qo_end - qo_begin;
  int const token_begin = q_tile_idx * TILE_Q;
  if (q_len <= 0 || token_begin >= q_len) {
    return;
  }

  int const token_limit = token_begin + TILE_Q;
  int const token_end = q_len < token_limit ? q_len : token_limit;
  int const pairs_per_token = ROPE_DIM / 2;
  int const work = (token_end - token_begin) * pairs_per_token;
  int const position_begin = step[req_id] + token_begin;

  for (int idx = threadIdx.x; idx < work; idx += blockDim.x) {
    int const local_tok = idx / pairs_per_token;
    int const pair = idx - local_tok * pairs_per_token;
    int const row = qo_begin + token_begin + local_tok;
    int const pos = position_begin + local_tok;
    int const d0 = pair * 2;
    int const d1 = d0 + 1;

    float const c = __bfloat162float(cos[pos * ROPE_DIM + d0]);
    float const s = __bfloat162float(sin[pos * ROPE_DIM + d0]);

    if constexpr (DO_Q) {
      __nv_bfloat16 *q_tail;
      if constexpr (Q_ROW_STRIDE_OVERRIDE > 0) {
        // Row-swap fused layout: pe of head `head_idx` lives at
        //   base + row * ROW_STRIDE + (H * D_NOPE) + head_idx * D_PE
        q_tail = q_fused + static_cast<long long>(row) * Q_ROW_STRIDE_OVERRIDE +
                 Q_PE_BASE_IN_ROW + head_idx * Q_PE_HEAD_STRIDE;
      } else {
        q_tail = q_fused +
                 static_cast<long long>(row) * NUM_HEADS * FUSED_HEAD_DIM +
                 head_idx * FUSED_HEAD_DIM + (FUSED_HEAD_DIM - ROPE_DIM);
      }
      float const q0 = __bfloat162float(q_tail[d0]);
      float const q1 = __bfloat162float(q_tail[d1]);
      q_tail[d0] = __float2bfloat16(q0 * c - q1 * s);
      q_tail[d1] = __float2bfloat16(q1 * c + q0 * s);

      if constexpr (HAS_SPLIT_Q) {
        __nv_bfloat16 *q_pe =
            q_split_pe + static_cast<long long>(row) * NUM_HEADS * ROPE_DIM +
            head_idx * ROPE_DIM;
        float const p0 = __bfloat162float(q_pe[d0]);
        float const p1 = __bfloat162float(q_pe[d1]);
        q_pe[d0] = __float2bfloat16(p0 * c - p1 * s);
        q_pe[d1] = __float2bfloat16(p1 * c + p0 * s);
      }
    }

    if constexpr (DO_K) {
      if (head_idx != 0) {
        continue;
      }
      __nv_bfloat16 *k_tok = k_pe + static_cast<long long>(row) * K_PE_STRIDE;
      float const k0 = __bfloat162float(k_tok[d0]);
      float const k1 = __bfloat162float(k_tok[d1]);
      k_tok[d0] = __float2bfloat16(k0 * c - k1 * s);
      k_tok[d1] = __float2bfloat16(k1 * c + k0 * s);
    }
  }
}

} // namespace kernel
