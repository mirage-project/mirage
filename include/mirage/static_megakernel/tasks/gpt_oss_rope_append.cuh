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

// tasks/gpt_oss_rope_append.cuh -- the gpt_oss_rope_append task type
// (Blackwell GPT-OSS with an outside decode attention kernel,
// past 4K positions): what sm90_attention_split does before its KV loop, as
// a node of its own. Task x = request x (its one token t = qo_indptr[x]); warp
// h = KV head h:
//   q, k, v = bf16(the QKV GEMM's K parts summed in fp32 + the bias)
//   q_out[t][h QPK + i] = bf16(RoPE(q_i)) (fp32 at the token's position)
//   the cache's slot at the token's position = bf16(RoPE(k)), v
//   seq_lens[x] = the request's positions with the new token
// with the split task's sums, roundings and expressions (its appended K / V and
// its q, bit for bit), so that a decode attention kernel outside the layer
// kernel (FlashInfer's trtllm-gen) reads q_out, the caches and seq_lens.
//
// PARAMS {Q, KVH, D, max seq, page, R}; BUF_SLOTS {qkv [KP x T, KVH (QPK + 2)
// D] fp32 (the QKV GEMM's "store" parts), bias [KVH (QPK + 2) D] bf16, cos, sin
// [max seq, D] bf16, qo_indptr, kv_indptr, kv_indices, last_page_len, k cache,
// v cache [pages, page, KVH, D] bf16, seq_lens [R] int32, q_out [T, Q D] bf16
// (last: the node's output)}.
#pragma once
#include "sm90_attention_split.cuh"

namespace static_mk {

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_gpt_oss_rope_append(Maps const &,
                                                        G const &g,
                                                        KernelLocals &,
                                                        StaticTask const &tk) {
  using bf16 = __nv_bfloat16;
  constexpr int QH = PARAMS::v[0], KVH = PARAMS::v[1], D = PARAMS::v[2],
                PAGE = PARAMS::v[4], R = PARAMS::v[5];
  constexpr int QPK = QH / KVH, W = (QPK + 2) * D, STRIDE = KVH * W;
  constexpr int KP = Sm90First<IN...>::type::y;
  static_assert(D == 64 && QPK == 8 && KVH == SM90_WARPS,
                "gpt_oss_rope_append: 8 KV heads of 8 query heads of 64");
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2,
            qd = lane & 3;
  int const request = tk.x, head = warp;
  int const *qo = buf_at<int const>(g, slot_at<BUF_SLOTS, 4>());
  int const *kv_indptr = buf_at<int const>(g, slot_at<BUF_SLOTS, 5>());
  int const *kv_indices = buf_at<int const>(g, slot_at<BUF_SLOTS, 6>());
  int const *last_page_len = buf_at<int const>(g, slot_at<BUF_SLOTS, 7>());
  bf16 *k_cache = buf_at<bf16>(g, slot_at<BUF_SLOTS, 8>());
  bf16 *v_cache = buf_at<bf16>(g, slot_at<BUF_SLOTS, 9>());
  int *seq_lens = buf_at<int>(g, slot_at<BUF_SLOTS, 10>());
  bf16 *q_out = buf_at<bf16>(g, slot_at<BUF_SLOTS, 11>());
  // what needs no producer, before the wait: the position, the RoPE tables
  // there, the bias, the new token's page
  bool const active = request < R && qo[request + 1] > qo[request];
  int const tok = active ? qo[request] : 0;
  int const pos_new =
      request < R ? sm90_decode_cached<PAGE>(kv_indptr, last_page_len, request)
                  : 0;
  bf16 const *cosp =
      buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 2>()) + (size_t)pos_new * D;
  bf16 const *sinp =
      buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 3>()) + (size_t)pos_new * D;
  bf16 const *bias = buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 1>()) + head * W;
  __nv_bfloat162 qcs[8], qsn[8], qbs[8];
  float kcs[2], ksn[2], kbs[2], vbs[2];
  int new_page = 0;
  if (active) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      int const d = 8 * j + 2 * qd;
      qcs[j] = *reinterpret_cast<__nv_bfloat162 const *>(cosp + d);
      qsn[j] = *reinterpret_cast<__nv_bfloat162 const *>(sinp + d);
      qbs[j] = *reinterpret_cast<__nv_bfloat162 const *>(bias + gq * D + d);
    }
#pragma unroll
    for (int j = 0; j < 2; j++) {
      int const d = lane + 32 * j;
      kcs[j] = __bfloat162float(cosp[d]);
      ksn[j] = __bfloat162float(sinp[d]);
      kbs[j] = __bfloat162float(bias[QPK * D + d]);
      vbs[j] = __bfloat162float(bias[(QPK + 1) * D + d]);
    }
    new_page = kv_indices[kv_indptr[request] + pos_new / PAGE];
  }
  sm90_task_begin<IN...>(g);
  if (active) {
    // sm90_attention_split's q and new K / V, the same sums, roundings and
    // expressions (lane (gq, qd): q head gq's dims 8 j + 2 qd, + 1; K / V dims
    // lane, lane + 32)
    float const *qkv = buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()) +
                       (size_t)tok * STRIDE + head * W;
    float2 qp[KP][8];
    float kp[KP][2], vp[KP][2];
#pragma unroll
    for (int p = 0; p < KP; p++) {
#pragma unroll
      for (int j = 0; j < 8; j++) {
        qp[p][j] = *reinterpret_cast<float2 const *>(
            qkv + (size_t)p * T * STRIDE + gq * D + 8 * j + 2 * qd);
      }
#pragma unroll
      for (int j = 0; j < 2; j++) {
        kp[p][j] = qkv[(size_t)p * T * STRIDE + QPK * D + lane + 32 * j];
        vp[p][j] = qkv[(size_t)p * T * STRIDE + (QPK + 1) * D + lane + 32 * j];
      }
    }
    float qx[8][2];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float x = qp[0][j].x, y = qp[0][j].y;
#pragma unroll
      for (int p = 1; p < KP; p++) {
        x += qp[p][j].x, y += qp[p][j].y;
      }
      float2 const b = __bfloat1622float2(qbs[j]);
      qx[j][0] = __bfloat162float(__float2bfloat16_rn(x + b.x));
      qx[j][1] = __bfloat162float(__float2bfloat16_rn(y + b.y));
    }
    __nv_bfloat162 *qrow = reinterpret_cast<__nv_bfloat162 *>(
        q_out + ((size_t)tok * QH + head * QPK + gq) * D);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float2 const c = __bfloat1622float2(qcs[j]),
                   sn = __bfloat1622float2(qsn[j]);
      int const jp = (j + 4) & 7; // the partner dims (d +- D / 2)
      float r[2];
#pragma unroll
      for (int e = 0; e < 2; e++) {
        float const x = qx[j][e], y = qx[jp][e];
        float const ce = e ? c.y : c.x, se = e ? sn.y : sn.x;
        r[e] = __bfloat162float(
            __float2bfloat16_rn(j < 4 ? x * ce - y * se : x * ce + y * se));
      }
      qrow[4 * j + qd] = __floats2bfloat162_rn(r[0], r[1]);
    }
    float kx[2], vx[2];
#pragma unroll
    for (int j = 0; j < 2; j++) {
      float x = kp[0][j], v = vp[0][j];
#pragma unroll
      for (int p = 1; p < KP; p++) {
        x += kp[p][j], v += vp[p][j];
      }
      kx[j] = __bfloat162float(__float2bfloat16_rn(x + kbs[j]));
      vx[j] = __bfloat162float(__float2bfloat16_rn(v + vbs[j]));
    }
    size_t const off =
        (((size_t)new_page * PAGE + pos_new % PAGE) * KVH + head) * D;
#pragma unroll
    for (int j = 0; j < 2; j++) {
      int const d = lane + 32 * j;
      float const y = kx[1 - j];
      k_cache[off + d] = __float2bfloat16_rn(__bfloat162float(
          __float2bfloat16_rn(j == 0 ? kx[0] * kcs[0] - y * ksn[0]
                                     : kx[1] * kcs[1] + y * ksn[1])));
      v_cache[off + d] = __float2bfloat16_rn(vx[j]);
    }
  }
  if (request < R && threadIdx.x == 0) {
    seq_lens[request] = active ? pos_new + 1 : 0;
  }
  sm90_task_end<SELF>(g);
}

SM90_TASK_SMEM(gpt_oss_rope_append, 0)

} // namespace static_mk
