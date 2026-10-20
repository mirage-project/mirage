// tasks/sm90_attention_merge.cuh -- the sm90_attention_merge task type: the
// output of a split attention (sm90_attention_split) from its chunks' parts,
// with GPT-OSS's per-head sinks: for query head h of token t, with L_c the
// chunk's log2-sum-exp and s the sink (natural units),
//   out = sum_c 2^(L_c - M) O_c / (sum_c 2^(L_c - M) + 2^(s log2 e - M)),
// M = max(L_c, s log2 e), over the chunks the sequence reaches. Task (x, y) of
// grid (T, Q / 8, 1): token x, query heads [8 y, 8 y + 8) (one warp a head).
#pragma once
#include "sm90_common.cuh"

namespace static_mk {

// PARAMS {NUM_Q_HEADS, HEAD_DIM, CHUNK, C, HAS_SINK, PAGE_SIZE}; BUF_SLOTS
// {parts (sm90_attention_split's: the chunks' outputs, then their
// log2-sum-exps), qo_indptr, kv_indptr, last_page_len (the page tables:
// request 0's), sinks [Q] bf16 (-1: none), output [T, Q x D] bf16}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_attention_merge(Maps const &,
                                                         G const &g,
                                                         KernelLocals &,
                                                         StaticTask const &tk) {
  constexpr int QH = PARAMS::v[0], D = PARAMS::v[1], CHUNK = PARAMS::v[2],
                C = PARAMS::v[3], PAGE = PARAMS::v[5];
  constexpr bool HAS_SINK = PARAMS::v[4] == 1;
  static_assert(D == 64, "sm90_attention_merge: a lane two of 64 values");
  sm90_task_begin<IN...>(g);
  int const t = tk.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int const h = tk.y * 8 + warp;
  int const *qo = buf_at<int const>(g, slot_at<BUF_SLOTS, 1>());
  if (t < qo[1]) { // the step's tokens (request 0)
    int const *kv = buf_at<int const>(g, slot_at<BUF_SLOTS, 2>());
    int const seq = (kv[1] - kv[0] - 1) * PAGE +
                    buf_at<int const>(g, slot_at<BUF_SLOTS, 3>())[0];
    int const nc = (seq + CHUNK - 1) / CHUNK;
    __nv_bfloat162 const *parts =
        buf_at<__nv_bfloat162 const>(g, slot_at<BUF_SLOTS, 0>());
    float const *lse =
        reinterpret_cast<float const *>(parts + (size_t)C * T * QH * D / 2);
    float m = -INFINITY;
    float lc[C];
#pragma unroll
    for (int c = 0; c < C; c++) {
      lc[c] = c < nc ? lse[((size_t)t * C + c) * QH + h] : -INFINITY;
      m = fmaxf(m, lc[c]);
    }
    float sink = -INFINITY;
    if constexpr (HAS_SINK) {
      sink = __bfloat162float(
                 buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 4>())[h]) *
             1.4426950408889634f;
      m = fmaxf(m, sink);
    }
    float den = HAS_SINK ? exp2f(sink - m) : 0.f;
    float2 acc = make_float2(0.f, 0.f);
#pragma unroll
    for (int c = 0; c < C; c++) {
      if (c < nc) {
        float const w = exp2f(lc[c] - m);
        float2 const o = __bfloat1622float2(
            parts[(((size_t)c * T + t) * QH + h) * (D / 2) + lane]);
        acc.x += w * o.x, acc.y += w * o.y;
        den += w;
      }
    }
    buf_at<__nv_bfloat162>(
        g, slot_at<BUF_SLOTS, 5>())[((size_t)t * QH + h) * (D / 2) + lane] =
        __floats2bfloat162_rn(acc.x / den, acc.y / den);
  }
  sm90_task_end<SELF>(g);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sm90_attention_merge() {
  return 0;
}

} // namespace static_mk
