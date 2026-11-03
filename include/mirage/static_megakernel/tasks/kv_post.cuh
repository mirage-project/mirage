// tasks/kv_post.cuh -- DeepSeek-V4.1's window K (model.py L704-706 and L723,
// _window_kv): task t = token t, grid (T, 1, 1); D = 512 (head_dim) columns
// over the first D / 8 = 64 threads (warps 0 and 1), thread i columns 8 i ..
// 8 i + 7 (4 threads a 32-column group; the other warps only take part in the
// barriers). Add wkv's P K parts in part order -> bf16 (wkv's bf16 output) ->
// RMSNorm (rms_norm_parts8, q_norm.cuh): bf16(x * rsqrt(mean(x^2) + eps) *
// gamma) -> RoPE of the last RD = 64 columns at the token's position p
// (threads 56..63, 4 pairs each: re = fma(a, c, -bd), im = fma(b, c, ad)) ->
// bf16 -> act_quant's formula per 32 columns -> row p % W of token t's window
// cache [T][W][D + D / 32] (a graph tensor, kept across launches): D e4m3
// bytes, then D / 32 e8m0 scale bytes (byte j: columns 32 j .. 32 j + 31). The
// official keeps the bf16 dequantization of the same bytes (act_quant inplace:
// e4m3 x 2^ex, exact in bf16). Only that row is written; the row it replaces is
// position p - W, outside token t's window. Then the node's counter t += 1 (a
// reader of token t waits for 1).
#pragma once
#include "../core.cuh"
#include "act_quant.cuh"
#include "q_norm.cuh"

namespace static_mk {

template <int D, int W, int RD, int P, int EPS_BITS>
__device__ __forceinline__ void kv_post_task(float const *part,
                                             __nv_bfloat16 const *gamma,
                                             int const *pos,
                                             float const *freqs,
                                             unsigned char *cache,
                                             uint32_t *done_cnt,
                                             int t) {
  static_assert(D == 512 && RD == 64,
                "kv_post: 8 columns per thread (threads 0..63); RoPE on the "
                "last 64 (threads 56..63)");
  int const l = threadIdx.x & 31, c = 8 * threadIdx.x, p = pos[t];
  bool const rope = c >= D - RD && c < D;
  // the thread's 4 RoPE pairs first: the load overlaps the wait for wkv
  float4 fq[2];
  if (rope) {
    float4 const *f = reinterpret_cast<float4 const *>(
        freqs + ((size_t)p * (RD / 2) + (c - (D - RD)) / 2) * 2);
    fq[0] = f[0];
    fq[1] = f[1];
  }
  float y[8]; // kv
  if (rms_norm_parts8<D, P, EPS_BITS>(part, gamma, t, y)) {
    if (rope) {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        float const a = y[2 * k], b = y[2 * k + 1];
        float const cr = (k & 1) ? fq[k / 2].z : fq[k / 2].x,
                    ci = (k & 1) ? fq[k / 2].w : fq[k / 2].y;
        y[2 * k] = __bfloat162float(
            __float2bfloat16_rn(__fmaf_rn(a, cr, -__fmul_rn(b, ci))));
        y[2 * k + 1] = __bfloat162float(
            __float2bfloat16_rn(__fmaf_rn(b, cr, __fmul_rn(a, ci))));
      }
    }
    float amax = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      amax = fmaxf(amax, fabsf(y[k]));
    }
    for (int d = 1; d < 4; d <<= 1) { // over the 4 threads of the group
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
    }
    int const ex = xq_exponent(amax);
    unsigned char *row = cache + ((size_t)t * W + p % W) * (D + D / 32);
    *reinterpret_cast<uint2 *>(row + c) =
        make_uint2(xq_pack4(y, ex), xq_pack4(y + 4, ex));
    uint32_t const s4 = xq_scale_word<16>(ex); // 16 threads = 4 groups
    if ((l & 15) == 0) {
      *reinterpret_cast<uint32_t *>(row + D + c / 32) = s4; // groups c / 32 ..
    }
    rearm_parts8<D, P>(part, t, c);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(done_cnt + t, 1); // release: the CTA's stores before (the barrier)
  }
}

// kv_post task t = token t, grid (T, 1, 1); KV: wkv's node (a gemm_fp8_tile in
// slots: KV::y K parts, KV::v[2] = D rows). PARAMS {D, W, RD, eps (fp32
// bits)}; BUF_SLOTS {gamma ([D] bf16), positions ([T] int32), freqs ([rows][RD
// / 2][2] fp32: cos, sin), the window cache ([T][W][D + D / 32] uint8)}, graph
// tensors; the node's counters: one per token
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class KV,
          class... REST>
__device__ __forceinline__ void run_kv_post(Maps const &,
                                            G const &g,
                                            KernelLocals &,
                                            StaticTask const &tk) {
  constexpr int D = PARAMS::v[0], W = PARAMS::v[1], RD = PARAMS::v[2];
  static_assert(SELF::x == T && SELF::y == 1 && SELF::z == 1,
                "kv_post: one task per token");
  static_assert(SELF::buf < 0 && SELF::counter >= 0,
                "kv_post: its output a graph tensor (the window cache), a "
                "counter per token");
  static_assert(KV::buf >= 0 && KV::v[0] == COMBINE_SLOTS && KV::v[2] == D &&
                    KV::y >= 1 && KV::y <= 10,
                "kv_post: wkv's K parts in slots, at most 10 (in registers, 8 "
                "values each)");
  kv_post_task<D, W, RD, KV::y, PARAMS::v[3]>(
      buf_at<float const>(g, KV::buf),
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
constexpr int smem_kv_post() {
  return 0;
}

} // namespace static_mk
