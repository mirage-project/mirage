// tasks/ckv_post.cuh -- DeepSeek-V4.1's compressed KV write (model.py
// Attention._compress_kv L746-757): task t = token t, grid (T, 1, 1). Only
// when token t's group completed ((p + 1) % R == 0): the compressor's latent
// row (bf16) -> RoPE of the last RD = 64 columns at position p + 1 - R (the
// group's first token; torch's contraction) -> bf16 -> fp4_act_quant per 16
// columns with an E4M3 scale (kernel.py fp4_quant_kernel, scale_dtype FP8: amax
// floored at 6 * 2^-9, s = e4m3(amax / 6), clamp(x / s, +-6) -> e2m1, to
// nearest even; plain IEEE divisions, as the generated kernel's) -> row p / R
// of token t's compressed KV cache [T][CAP][D / 2 + D / 16] (a graph tensor,
// kept across launches): D e2m1 (element 2 i in the low nibble), then D / 16
// e4m3 scales. The official keeps the bf16 dequantization of the same bytes.
// Warp 0: lane l columns 16 l .. 16 l + 15 (one group; the RoPE columns:
// lanes 28..31). Then the node's counter t += 1 (also when nothing is written:
// sparse_attn of token t waits for 1).
#pragma once
#include "../core.cuh"
#include "act_quant.cuh"

namespace static_mk {

template <int D, int RD, int R, int CAP>
__device__ __forceinline__ void ckv_post_task(__nv_bfloat16 const *latent,
                                              int const *pos,
                                              float const *freqs,
                                              unsigned char *cache,
                                              uint32_t const *l_cnt,
                                              uint32_t l_target,
                                              uint32_t *done_cnt,
                                              int t) {
  static_assert(D == 512 && RD == 64,
                "ckv_post: 512 columns, 16 per lane of warp 0; RoPE on the "
                "last 64 (lanes 28..31)");
  int const l = threadIdx.x & 31, c = 16 * l, p = pos[t];
  // warp 0, a group completed (its row in the cache: none past the cache)
  if (threadIdx.x < 32 && (p + 1) % R == 0 && p / R < CAP) {
    bool const rope = c >= D - RD;
    float4 fq[4];
    if (rope) { // the lane's 8 RoPE pairs at the group's first position
      float4 const *f = reinterpret_cast<float4 const *>(
          freqs + ((size_t)(p + 1 - R) * (RD / 2) + (c - (D - RD)) / 2) * 2);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        fq[k] = f[k];
      }
    }
    if (l == 0) {
      cnt_wait(l_cnt, l_target); // the compressor's tasks (all tokens)
    }
    __syncwarp();
    unsigned char const *src =
        reinterpret_cast<unsigned char const *>(latent + (size_t)t * D + c);
    uint4 const v0 = ld16_relaxed(src), v1 = ld16_relaxed(src + 16);
    float x[16];
    {
      uint32_t const wv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
      for (int k = 0; k < 8; k++) {
        float2 const f = __bfloat1622float2(
            *reinterpret_cast<__nv_bfloat162 const *>(&wv[k]));
        x[2 * k] = f.x;
        x[2 * k + 1] = f.y;
      }
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
    for (int k = 0; k < 16; k++) {
      amax = fmaxf(amax, fabsf(x[k]));
    }
    amax = fmaxf(amax, 6.0f * 0x1p-9f);
    uint16_t sb2;
    float const sv = __fdiv_rn(amax, 6.0f);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(sb2) : "f"(sv));
    uint32_t sh;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sh) : "h"(sb2));
    float const s = __half2float(__ushort_as_half((uint16_t)sh)); // e4m3, exact
    uint32_t q[2] = {0u, 0u};
#pragma unroll
    for (int k = 0; k < 8; k++) { // byte k: columns 2 k (low nibble), 2 k + 1
      uint16_t b;
      asm("{\n.reg .b8 b;\ncvt.rn.satfinite.e2m1x2.f32 b, %1, %2;\n"
          "cvt.u16.u8 %0, b;\n}"
          : "=h"(b)
          : "f"(__fdiv_rn(x[2 * k + 1], s)), "f"(__fdiv_rn(x[2 * k], s)));
      q[k / 4] |= (uint32_t)(b & 0xFFu) << (8 * (k & 3));
    }
    unsigned char *row = cache + ((size_t)t * CAP + p / R) * (D / 2 + D / 16);
    *reinterpret_cast<uint2 *>(row + 8 * l) = make_uint2(q[0], q[1]);
    row[D / 2 + l] = (unsigned char)(sb2 & 0xFFu);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(done_cnt + t, 1); // release: the CTA's stores before (the barrier)
  }
}

// ckv_post task t = token t, grid (T, 1, 1); LT: the compressor's node (its
// buffer: the latent [T][D] bf16; one counter of its LT::x tasks). PARAMS {D,
// RD, R, CAP}; BUF_SLOTS {positions ([T] int32), freqs ([rows][RD / 2][2]
// fp32), the compressed KV cache ([T][CAP][D / 2 + D / 16] uint8)}, graph
// tensors; the node's counters: one per token
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class LT,
          class... REST>
__device__ __forceinline__ void run_ckv_post(Maps const &,
                                             G const &g,
                                             KernelLocals &,
                                             StaticTask const &tk) {
  constexpr int D = PARAMS::v[0], RD = PARAMS::v[1], R = PARAMS::v[2],
                CAP = PARAMS::v[3];
  static_assert(SELF::x == T && SELF::y == 1 && SELF::z == 1,
                "ckv_post: one task per token");
  static_assert(SELF::buf < 0 && SELF::counter >= 0,
                "ckv_post: its output a graph tensor (the compressed KV "
                "cache), a counter per token");
  static_assert(LT::buf >= 0 && LT::counter >= 0 && LT::x == T &&
                    LT::v[0] == D && LT::v[1] == R,
                "ckv_post: the latent is the compressor's (one counter of its "
                "tasks)");
  ckv_post_task<D, RD, R, CAP>(
      buf_at<__nv_bfloat16 const>(g, LT::buf),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<unsigned char>(g, slot_at<BUF_SLOTS, 2>()),
      g.cnt + LT::counter,
      LT::x * LT::y * LT::z,
      g.cnt + SELF::counter,
      tk.x);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_ckv_post() {
  return 0;
}

} // namespace static_mk
