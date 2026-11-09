// tasks/index_k_post.cuh -- DeepSeek-V4.1's index keys (model.py L552-562,
// Indexer.forward of a layer that compresses its own KV): task t = token t,
// grid (T, 1, 1). Only when token t's compressor completed a group this step
// ((p + 1) % R == 0): add wk's P K parts (wk times the compressor's latent; a
// gemm_tile node's buffer in slots [P][T][HD], 0xFF before the launch: polled)
// in part order -> bf16 (wk's bf16 output) -> RMSNorm (k_norm, HD = 128) ->
// bf16 -> RoPE of the last RD = 64 columns at position p + 1 - R (the group's
// first token) -> bf16 -> fp4_act_quant per 32 columns (E8M0) -> row p / R of
// token t's index K cache [T][CAP][IX_ROW] (a graph tensor, kept across
// launches; indexer_common.cuh). The official keeps the bf16 dequantization
// of the same bytes. Warp 0: lane l columns 4 l .. 4 l + 3 (8 lanes a group of
// 32). Then the node's counter t += 1 (also when nothing is written; a reader
// of token t waits for 1).
#pragma once
#include "../core.cuh"
#include "act_quant.cuh"
#include "indexer_common.cuh"

namespace static_mk {

template <int HD, int RD, int R, int CAP, int P, int EPS_BITS>
__device__ __forceinline__ void index_k_post_task(float const *part,
                                                  __nv_bfloat16 const *gamma,
                                                  int const *pos,
                                                  float const *freqs,
                                                  unsigned char *cache,
                                                  uint32_t *done_cnt,
                                                  int t) {
  static_assert(HD == IX_HD && RD == 64,
                "index_k_post: 128 columns, 4 per lane of warp 0; RoPE on the "
                "last 64 (lanes 16..31)");
  int const l = threadIdx.x & 31, c = 4 * l, p = pos[t];
  bool const fires = (p + 1) % R == 0 && p / R < CAP; // past the cache: none
  // warp 0, a group completed; REARM: every token (its row's parts are re-armed
  // once wk wrote them, used or not)
  if (threadIdx.x < 32 && (fires || REARM)) {
    bool const rope = c >= HD - RD;
    float4 fq;
    if (rope &&
        fires) { // the lane's 2 RoPE pairs at the group's first position
      fq = *reinterpret_cast<float4 const *>(
          freqs + ((size_t)(p + 1 - R) * (RD / 2) + (c - (HD - RD)) / 2) * 2);
    }
    uint2 const gb = *reinterpret_cast<uint2 const *>(gamma + c);
    float4 v[P];
    bool ok[P];
#pragma unroll
    for (int q = 0; q < P; q++) {
      ok[q] = false;
    }
    bool all;
    do {
      all = true;
#pragma unroll
      for (int q = 0; q < P; q++) {
        if (!ok[q]) {
          v[q] = ldf4_poll(part + ((size_t)q * T + t) * HD + c);
        }
      }
#pragma unroll
      for (int q = 0; q < P; q++) {
        ok[q] = ok[q] || validf4(v[q]);
        all = all && ok[q];
      }
      if (!all) {
        poll_pause(STATIC_POLL_NS);
      }
    } while (!all);
    if constexpr (REARM) { // the one reader of these parts (config.cuh)
#pragma unroll
      for (int q = 0; q < P; q++) {
        rearm16(part + ((size_t)q * T + t) * HD + c);
      }
    }
    if (fires) { // the row written
      float x[4] = {v[0].x, v[0].y, v[0].z, v[0].w};
#pragma unroll
      for (int q = 1; q < P; q++) { // in part order
        x[0] += v[q].x;
        x[1] += v[q].y;
        x[2] += v[q].z;
        x[3] += v[q].w;
      }
      // RMSNorm as rms_norm_parts8 (q_norm.cuh): the squares rounded, added in
      // a fixed order (the lane's 4 in order, the warp by a butterfly)
      float ss = 0.f;
#pragma unroll
      for (int k = 0; k < 4; k++) {
        x[k] = bf16r(x[k]); // wk's output
        ss = __fadd_rn(ss, __fmul_rn(x[k], x[k]));
      }
      for (int d = 16; d > 0; d >>= 1) {
        ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, d));
      }
      float const r =
          rsqrtf(__fadd_rn(__fmul_rn(ss, 1.0f / HD), __int_as_float(EPS_BITS)));
      __nv_bfloat162 const *g2 = reinterpret_cast<__nv_bfloat162 const *>(&gb);
#pragma unroll
      for (int k = 0; k < 2; k++) {
        float2 const gm = __bfloat1622float2(g2[k]);
        x[2 * k] = bf16r(__fmul_rn(__fmul_rn(x[2 * k], r), gm.x));
        x[2 * k + 1] = bf16r(__fmul_rn(__fmul_rn(x[2 * k + 1], r), gm.y));
      }
      if (rope) {
#pragma unroll
        for (int k = 0; k < 2; k++) {
          float const a = x[2 * k], b = x[2 * k + 1];
          float const cr = k ? fq.z : fq.x, ci = k ? fq.w : fq.y;
          x[2 * k] = bf16r(__fmaf_rn(a, cr, -__fmul_rn(b, ci)));
          x[2 * k + 1] = bf16r(__fmaf_rn(b, cr, __fmul_rn(a, ci)));
        }
      }
      float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])),
                         fmaxf(fabsf(x[2]), fabsf(x[3])));
      for (int d = 1; d < 8; d <<= 1) { // the group's 8 lanes
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
      }
      int const e = fp4_exponent(amax);
      float deq[4];
      uint32_t const b2 = fp4_quant4(x, e, deq);
      unsigned char *row = cache + ((size_t)t * CAP + p / R) * IX_ROW;
      *reinterpret_cast<uint16_t *>(row + 2 * l) = (uint16_t)b2;
      if ((l & 7) == 0) {
        row[HD / 2 + l / 8] = (unsigned char)(e + 127);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(done_cnt + t, 1); // release: the CTA's stores before (the barrier)
  }
}

// index_k_post task t = token t, grid (T, 1, 1); WK: wk's node (a gemm_tile in
// slots: WK::y K parts, WK::v[2] = HD rows). PARAMS {HD, RD, R (compress
// ratio), CAP (the cache's positions), eps (fp32 bits)}; BUF_SLOTS {gamma
// (k_norm's weight, [HD] bf16), positions ([T] int32), freqs ([rows][RD /
// 2][2] fp32), the index K cache ([T][CAP][IX_ROW] uint8)}, graph tensors; the
// node's counters: one per token
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class WK,
          class... REST>
__device__ __forceinline__ void run_index_k_post(Maps const &,
                                                 G const &g,
                                                 KernelLocals &,
                                                 StaticTask const &tk) {
  constexpr int HD = PARAMS::v[0], RD = PARAMS::v[1], R = PARAMS::v[2],
                CAP = PARAMS::v[3];
  static_assert(SELF::x == T && SELF::y == 1 && SELF::z == 1,
                "index_k_post: one task per token");
  static_assert(SELF::buf < 0 && SELF::counter >= 0,
                "index_k_post: its output a graph tensor (the index K cache), "
                "a counter per token");
  static_assert(WK::buf >= 0 && WK::v[0] == COMBINE_SLOTS && WK::v[2] == HD &&
                    WK::y >= 1 && WK::y <= 8,
                "index_k_post: wk's K parts in slots, at most 8");
  index_k_post_task<HD, RD, R, CAP, WK::y, PARAMS::v[4]>(
      buf_at<float const>(g, WK::buf),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<unsigned char>(g, slot_at<BUF_SLOTS, 3>()),
      g.cnt + SELF::counter,
      tk.x);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_index_k_post() {
  return 0;
}

} // namespace static_mk
