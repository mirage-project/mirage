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

// tasks/sm90_attention_merge.cuh -- the sm90_attention_merge task type: the
// output of a split attention (sm90_attention_split) from its splits' parts,
// with GPT-OSS's per-head sinks: for query head h of request r, with L_s the
// split's log2-sum-exp and s the sink (natural units),
//   out = sum_s 2^(L_s - M) O_s / (sum_s 2^(L_s - M) + 2^(s log2 e - M)),
// M = max(L_s, s log2 e), over the request's splits (sm90_attention_split's
// split_base). Task (x, y) of grid (R, Q / HPT, 1): request x (its one token),
// query heads [HPT y, HPT y + HPT); warp w takes head HPT y + w % HPT and that
// head's splits w / HPT, + 8 / HPT, ... (a lane two of the 64 values), the
// warps' sums added in a fixed order. HPT > 8 (many requests, few splits
// each: the grid on the SMs): warp w the heads w, w + 8, .. of the task's,
// each over all its splits.
#pragma once
#include "sm90_attention_split.cuh"

namespace static_mk {

// PARAMS {NUM_Q_HEADS, HEAD_DIM, P (parts' splits), HAS_SINK, HPT (heads a
// task), R (requests)}; BUF_SLOTS {parts (sm90_attention_split's), qo_indptr,
// sinks [Q] bf16 (-1: none), output [T, Q x D] bf16}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_attention_merge(Maps const &,
                                                         G const &g,
                                                         KernelLocals &,
                                                         StaticTask const &tk) {
  constexpr int QH = PARAMS::v[0], D = PARAMS::v[1], P = PARAMS::v[2],
                HPT = PARAMS::v[4], R = PARAMS::v[5];
  constexpr bool HAS_SINK = PARAMS::v[3] == 1;
  static_assert(D == 64 && (SM90_WARPS % HPT == 0 || HPT % SM90_WARPS == 0),
                "sm90_attention_merge: a lane two of 64 values");
  if constexpr (HPT > SM90_WARPS) {
    sm90_task_begin<IN...>(g);
    int const r = tk.x, lane = threadIdx.x & 31;
    int const *qo = buf_at<int const>(g, slot_at<BUF_SLOTS, 1>());
    if (qo[r + 1] > qo[r]) {
      float const *parts = buf_at<float const>(g, slot_at<BUF_SLOTS, 0>());
      int const *split_base =
          reinterpret_cast<int const *>(parts + (size_t)P * QH * (D + 1));
      int const base = split_base[r], n = split_base[r + 1] - base;
#pragma unroll 1
      for (int hh = threadIdx.x >> 5; hh < HPT; hh += SM90_WARPS) {
        int const h = tk.y * HPT + hh;
        float const *lse = parts + (size_t)P * QH * D + h;
        float m = -INFINITY;
        for (int k = lane; k < n; k += 32) {
          m = fmaxf(m, lse[(size_t)(base + k) * QH]);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
          m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
        }
        float sink = -INFINITY;
        if constexpr (HAS_SINK) {
          sink = __bfloat162float(buf_at<__nv_bfloat16 const>(
                     g, slot_at<BUF_SLOTS, 2>())[h]) *
                 1.4426950408889634f;
        }
        m = fmaxf(m, sink);
        float den = HAS_SINK ? exp2f(sink - m) : 0.f;
        float2 acc = make_float2(0.f, 0.f);
#pragma unroll 4
        for (int k = 0; k < n; k++) {
          float const w = exp2f(lse[(size_t)(base + k) * QH] - m);
          float2 const o = reinterpret_cast<float2 const *>(
              parts + ((size_t)(base + k) * QH + h) * D)[lane];
          acc.x += w * o.x, acc.y += w * o.y;
          den += w;
        }
        buf_at<__nv_bfloat162>(g, slot_at<BUF_SLOTS, 3>())
            [((size_t)qo[r] * QH + h) * (D / 2) + lane] =
                __floats2bfloat162_rn(acc.x / den, acc.y / den);
      }
    }
    sm90_task_end<SELF>(g);
    return;
  }
  constexpr int NJ = SM90_WARPS / HPT; // the warps a head
  __shared__ float red_m[SM90_WARPS], red_d[SM90_WARPS];
  __shared__ __align__(8) float2 red_o[SM90_WARPS][32];
  sm90_task_begin<IN...>(g);
  int const r = tk.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int const hh = warp % HPT, j = warp / HPT, h = tk.y * HPT + hh;
  int const *qo = buf_at<int const>(g, slot_at<BUF_SLOTS, 1>());
  if (qo[r + 1] > qo[r]) { // the request has its token this step
    float const *parts = buf_at<float const>(g, slot_at<BUF_SLOTS, 0>());
    float const *lse = parts + (size_t)P * QH * D + h; // [split * QH]
    int const *split_base =
        reinterpret_cast<int const *>(parts + (size_t)P * QH * (D + 1));
    int const base = split_base[r], n = split_base[r + 1] - base;
    // the head's max over all its splits (and the sink)
    float m = -INFINITY;
    for (int k = j + NJ * lane; k < n; k += NJ * 32) {
      m = fmaxf(m, lse[(size_t)(base + k) * QH]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
    }
    if (lane == 0) {
      red_m[warp] = m;
    }
    __syncthreads();
    float sink = -INFINITY;
    if constexpr (HAS_SINK) {
      sink = __bfloat162float(
                 buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>())[h]) *
             1.4426950408889634f;
    }
    m = sink;
#pragma unroll
    for (int k = 0; k < NJ; k++) {
      m = fmaxf(m, red_m[hh + HPT * k]);
    }
    // the warp's splits
    float den = 0.f;
    float2 acc = make_float2(0.f, 0.f);
#pragma unroll 4
    for (int k = j; k < n; k += NJ) {
      float const w = exp2f(lse[(size_t)(base + k) * QH] - m);
      float2 const o = reinterpret_cast<float2 const *>(
          parts + ((size_t)(base + k) * QH + h) * D)[lane];
      acc.x += w * o.x, acc.y += w * o.y;
      den += w;
    }
    red_o[warp][lane] = acc;
    if (lane == 0) {
      red_d[warp] = den;
    }
    __syncthreads();
    if (j == 0) {
      den = HAS_SINK ? exp2f(sink - m) : 0.f;
      acc = make_float2(0.f, 0.f);
#pragma unroll
      for (int k = 0; k < NJ; k++) {
        acc.x += red_o[hh + HPT * k][lane].x,
            acc.y += red_o[hh + HPT * k][lane].y;
        den += red_d[hh + HPT * k];
      }
      buf_at<__nv_bfloat162>(
          g,
          slot_at<BUF_SLOTS, 3>())[((size_t)qo[r] * QH + h) * (D / 2) + lane] =
          __floats2bfloat162_rn(acc.x / den, acc.y / den);
    }
  }
  sm90_task_end<SELF>(g);
}

SM90_TASK_SMEM(sm90_attention_merge, 0)

} // namespace static_mk
