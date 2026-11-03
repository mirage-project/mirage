// tasks/indexer_q.cuh -- DeepSeek-V4.1's indexer query (model.py L567-569,
// Indexer.forward): task t = token t, grid (T, 1, 1); H = 32 heads of HD = 128
// columns, thread i head i / 8, columns 16 (i % 8) .. + 15 (2 threads a group
// of 32). Add the indexer wq_b's P K parts (fp32, its buffer in slots
// [P][T][H HD], 0xFF before the launch: polled) in part order -> bf16 (wq_b's
// bf16 output) -> RoPE of the last RD = 64 columns at the token's position p
// (torch's contraction) -> bf16 -> fp4_act_quant per 32 columns (E8M0 scale,
// inplace: the dequantized values) -> q_idx [T][H HD] bf16, the node's buffer
// (every byte written each launch). Then the node's counter t += 1 (a reader
// of token t waits for 1).
#pragma once
#include "../core.cuh"
#include "act_quant.cuh"
#include "indexer_common.cuh"

namespace static_mk {

template <int H, int HD, int RD, int P>
__device__ __forceinline__ void indexer_q_task(float const *part,
                                               int const *pos,
                                               float const *freqs,
                                               __nv_bfloat16 *out,
                                               uint32_t *done_cnt,
                                               int t) {
  static_assert(H * HD == 256 * 16 && HD == 128 && RD == 64,
                "indexer_q: 32 heads of 128 columns, 16 per thread; RoPE on "
                "the last 64 (threads i % 8 >= 4)");
  constexpr size_t N = (size_t)H * HD;
  int const i = threadIdx.x, c = 16 * (i & 7);
  size_t const col = (size_t)(i >> 3) * HD + c;
  bool const rope = c >= HD - RD;
  // the thread's 8 RoPE pairs first: the load overlaps the wait for wq_b
  float4 fq[4];
  if (rope) {
    float4 const *f = reinterpret_cast<float4 const *>(
        freqs + ((size_t)pos[t] * (RD / 2) + (c - (HD - RD)) / 2) * 2);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      fq[k] = f[k];
    }
  }
  float4 v[P][4];
  bool ok[P];
#pragma unroll
  for (int p = 0; p < P; p++) {
    ok[p] = false;
  }
  bool all; // all P parts' loads in flight at once
  do {
    all = true;
#pragma unroll
    for (int p = 0; p < P; p++) {
      if (!ok[p]) {
#pragma unroll
        for (int k = 0; k < 4; k++) {
          v[p][k] = ldf4_poll(part + ((size_t)p * T + t) * N + col + 4 * k);
        }
      }
    }
#pragma unroll
    for (int p = 0; p < P; p++) {
      if (!ok[p]) {
        ok[p] = validf4(v[p][0]) && validf4(v[p][1]) && validf4(v[p][2]) &&
                validf4(v[p][3]);
      }
      all = all && ok[p];
    }
    if (!all) {
      poll_pause(STATIC_POLL_NS);
    }
  } while (!all);
  float x[16];
#pragma unroll
  for (int k = 0; k < 4; k++) {
    x[4 * k] = v[0][k].x;
    x[4 * k + 1] = v[0][k].y;
    x[4 * k + 2] = v[0][k].z;
    x[4 * k + 3] = v[0][k].w;
  }
#pragma unroll
  for (int p = 1; p < P; p++) { // in part order
#pragma unroll
    for (int k = 0; k < 4; k++) {
      x[4 * k] += v[p][k].x;
      x[4 * k + 1] += v[p][k].y;
      x[4 * k + 2] += v[p][k].z;
      x[4 * k + 3] += v[p][k].w;
    }
  }
#pragma unroll
  for (int j = 0; j < 16; j++) {
    x[j] = bf16r(x[j]); // wq_b's output
  }
  if (rope) {
#pragma unroll
    for (int k = 0; k < 8; k++) {
      float const a = x[2 * k], b = x[2 * k + 1];
      float const cr = (k & 1) ? fq[k / 2].z : fq[k / 2].x,
                  ci = (k & 1) ? fq[k / 2].w : fq[k / 2].y;
      x[2 * k] = bf16r(__fmaf_rn(a, cr, -__fmul_rn(b, ci)));
      x[2 * k + 1] = bf16r(__fmaf_rn(b, cr, __fmul_rn(a, ci)));
    }
  }
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 16; j++) {
    amax = fmaxf(amax, fabsf(x[j]));
  }
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1)); // the group's
  int const e = fp4_exponent(amax);
  float d[16];
#pragma unroll
  for (int k = 0; k < 4; k++) {
    fp4_quant4(x + 4 * k, e, d + 4 * k);
  }
  if constexpr (REARM) { // the one reader of these parts (config.cuh)
#pragma unroll
    for (int p = 0; p < P; p++) {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        rearm16(part + ((size_t)p * T + t) * N + col + 4 * k);
      }
    }
  }
  uint4 *o = reinterpret_cast<uint4 *>(out + (size_t)t * N + col);
  o[0] = pack_bf16x8(make_float4(d[0], d[1], d[2], d[3]),
                     make_float4(d[4], d[5], d[6], d[7]));
  o[1] = pack_bf16x8(make_float4(d[8], d[9], d[10], d[11]),
                     make_float4(d[12], d[13], d[14], d[15]));
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(done_cnt + t, 1); // release: the CTA's stores before (the barrier)
  }
}

// indexer_q task t = token t, grid (T, 1, 1); WB: the indexer wq_b's node (a
// gemm_fp8_tile in slots: WB::y K parts, WB::v[2] = H HD rows). PARAMS {H, HD,
// RD}; BUF_SLOTS {positions ([T] int32), freqs ([rows][RD / 2][2] fp32)},
// graph tensors; the node's buffer q_idx [T][H HD] bf16, a counter per token
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class WB,
          class... REST>
__device__ __forceinline__ void run_indexer_q(Maps const &,
                                              G const &g,
                                              KernelLocals &,
                                              StaticTask const &tk) {
  constexpr int H = PARAMS::v[0], HD = PARAMS::v[1], RD = PARAMS::v[2];
  static_assert(SELF::x == T && SELF::y == 1 && SELF::z == 1,
                "indexer_q: one task per token");
  static_assert(SELF::buf >= 0 && SELF::counter >= 0,
                "indexer_q: its output in its buffer, a counter per token");
  static_assert(WB::buf >= 0 && WB::v[0] == COMBINE_SLOTS &&
                    WB::v[2] == H * HD && WB::y >= 1 && WB::y <= 4,
                "indexer_q: the indexer wq_b's K parts in slots, at most 4 "
                "(in registers, 16 values each)");
  indexer_q_task<H, HD, RD, WB::y>(
      buf_at<float const>(g, WB::buf),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<__nv_bfloat16>(g, SELF::buf),
      g.cnt + SELF::counter,
      tk.x);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_indexer_q() {
  return 0;
}

} // namespace static_mk
