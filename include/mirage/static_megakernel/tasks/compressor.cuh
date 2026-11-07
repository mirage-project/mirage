// tasks/compressor.cuh -- DeepSeek-V4.1's compressor (model.py Compressor
// L458-485, decode): task t = token t, grid (T, 1, 1); D = 512 (head_dim)
// columns over the first D / 8 = 64 threads (warps 0 and 1), thread i columns
// 8 i .. 8 i + 7. Its GEMM (a gemm_tile node in slots, [P][T][NW] fp32: x
// times wkv, and for R = 2 also wgate) comes first:
//   R = 1: wkv's K parts added in part order -> bf16 (the bf16 linear's
//          output) -> RMSNorm(gamma) -> latent.
//   R = 2: the K parts added (fp32: the official promotes x and the weights to
//          fp32) -> kv, score into row p % 2 of the token's request slot in
//          the state (a graph tensor [slots][R][2][D] fp32, kv | score: the
//          framework's ring in a serving layer; kept across
//          launches) -> when the group completes ((p + 1) % 2 == 0), per
//          column softmax over the 2 slots (max, expf, the sum, a division:
//          torch's) and kv0 w0 + kv1 w1 (each product rounded) -> bf16 ->
//          RMSNorm(gamma) -> latent.
// The latent (before RoPE: the index keys need it) -> row t of the node's
// buffer [T][D] bf16, only for a token whose group completed (the other rows
// are left as they are: wk multiplies them, index_k_post and ckv_post skip
// them). Then the node's counter += 1 (its readers wait for T: wk's gemm_tile
// reads the latent through a tensor map).
#pragma once
#include "../core.cuh"
#include "act_quant.cuh"

namespace static_mk {

// RMSNorm of a row of K columns held 8 per thread by the first K / 8 threads
// (on), as rms_norm_parts8 (q_norm.cuh) does it: the squares rounded, added in
// a fixed order (a thread's 8 in order, the warp's 32 by a butterfly, the
// warps in order); y = bf16(x * rsqrt(mean(x^2) + eps) * gamma). x: bf16
// values; gb: the thread's 8 gamma values. Every thread calls it (a barrier)
template <int K, int EPS_BITS>
__device__ __forceinline__ void
    cmp_rms_norm8(float const (&x)[8], bool on, uint4 gb, float (&y)[8]) {
  constexpr int NW = K / 256;
  static_assert(K % 256 == 0 && NW >= 1 && NW <= 8,
                "cmp_rms_norm8: 8 columns per thread, whole warps");
  int const w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __shared__ float red_c[8];
  if (on) {
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      ss = __fadd_rn(ss, __fmul_rn(x[k], x[k])); // no fma: the squares rounded
    }
    for (int d = 16; d > 0; d >>= 1) {
      ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, d));
    }
    if (l == 0) {
      red_c[w] = ss;
    }
  }
  __syncthreads();
  if (on) {
    float tot = red_c[0];
#pragma unroll
    for (int i = 1; i < NW; i++) {
      tot = __fadd_rn(tot, red_c[i]);
    }
    float const r =
        rsqrtf(__fadd_rn(__fmul_rn(tot, 1.0f / K), __int_as_float(EPS_BITS)));
    __nv_bfloat162 const *g2 = reinterpret_cast<__nv_bfloat162 const *>(&gb);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      float2 const gm = __bfloat1622float2(g2[k]);
      y[2 * k] = bf16r(__fmul_rn(__fmul_rn(x[2 * k], r), gm.x));
      y[2 * k + 1] = bf16r(__fmul_rn(__fmul_rn(x[2 * k + 1], r), gm.y));
    }
  }
}

template <int D, int R, int NW, int P, int EPS_BITS>
__device__ __forceinline__ void compressor_task(float const *part,
                                                __nv_bfloat16 const *gamma,
                                                int const *pos,
                                                float *state,
                                                long long const *req,
                                                __nv_bfloat16 *out,
                                                uint32_t *done_cnt,
                                                int t) {
  static_assert(D == 512 && (R == 1 || R == 2) && NW == (R == 1 ? D : 2 * D),
                "compressor: 512 columns, 8 per thread (threads 0..63); ratio "
                "1 (wkv) or 2 ([wkv; wgate])");
  int const c = 8 * threadIdx.x, p = pos[t];
  bool const on = threadIdx.x < D / 8;
  bool const fires = (p + 1) % R == 0; // the same for the whole CTA
  float x[8] = {}; // the latent before the norm (bf16 values)
  uint4 gb = make_uint4(0, 0, 0, 0);
  if (on) {
    gb = *reinterpret_cast<uint4 const *>(gamma + c);
    if constexpr (R == 1) {
      add_parts8<NW, P>(part, t, c, x);
      rearm_parts8<NW, P>(part, t, c);
#pragma unroll
      for (int k = 0; k < 8; k++) {
        x[k] = bf16r(x[k]); // wkv's bf16 output
      }
    } else {
      float kv[8], sc[8];
      add_parts8<NW, P>(part, t, c, kv);     // wkv(x), fp32
      add_parts8<NW, P>(part, t, D + c, sc); // wgate(x), fp32
      rearm_parts8<NW, P>(part, t, c);
      rearm_parts8<NW, P>(part, t, D + c);
      int const slot = p % R;
      // the framework's ring: row (request slot) R + p % R holds kv | score
      float *const st0 = state + (size_t)req[t] * R * 2 * D;
      float kv_o[8], sc_o[8]; // the other slot (an earlier launch wrote it)
      if (fires) {
        float4 const *a =
            reinterpret_cast<float4 const *>(st0 + (1 - slot) * 2 * D + c);
        float4 const *b =
            reinterpret_cast<float4 const *>(st0 + (1 - slot) * 2 * D + D + c);
#pragma unroll
        for (int h = 0; h < 2; h++) {
          float4 const u = a[h], v = b[h];
          kv_o[4 * h] = u.x;
          kv_o[4 * h + 1] = u.y;
          kv_o[4 * h + 2] = u.z;
          kv_o[4 * h + 3] = u.w;
          sc_o[4 * h] = v.x;
          sc_o[4 * h + 1] = v.y;
          sc_o[4 * h + 2] = v.z;
          sc_o[4 * h + 3] = v.w;
        }
      }
      float4 *const ka = reinterpret_cast<float4 *>(st0 + slot * 2 * D + c);
      float4 *const sa = reinterpret_cast<float4 *>(st0 + slot * 2 * D + D + c);
      ka[0] = make_float4(kv[0], kv[1], kv[2], kv[3]);
      ka[1] = make_float4(kv[4], kv[5], kv[6], kv[7]);
      sa[0] = make_float4(sc[0], sc[1], sc[2], sc[3]);
      sa[1] = make_float4(sc[4], sc[5], sc[6], sc[7]);
      if (fires) { // slot 1 is this step's (p odd), slot 0 the step before's
#pragma unroll
        for (int k = 0; k < 8; k++) {
          float const k0 = kv_o[k], k1 = kv[k], s0 = sc_o[k], s1 = sc[k];
          float const m = fmaxf(s0, s1);
          float const e0 = expf(__fsub_rn(s0, m)), e1 = expf(__fsub_rn(s1, m));
          float const sum = __fadd_rn(e0, e1);
          x[k] = bf16r(__fadd_rn(__fmul_rn(k0, __fdiv_rn(e0, sum)),
                                 __fmul_rn(k1, __fdiv_rn(e1, sum))));
        }
      }
    }
  }
  if (fires) { // uniform over the CTA: the norm's barrier
    float y[8];
    cmp_rms_norm8<D, EPS_BITS>(x, on, gb, y);
    if (on) {
      *reinterpret_cast<uint4 *>(out + (size_t)t * D + c) =
          pack_bf16x8(make_float4(y[0], y[1], y[2], y[3]),
                      make_float4(y[4], y[5], y[6], y[7]));
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(done_cnt, 1); // release: the CTA's stores before (the barrier)
  }
}

// compressor task t = token t, grid (T, 1, 1); CW: the compressor's GEMM (a
// gemm_tile in slots: CW::y K parts, CW::v[2] = NW rows). PARAMS {D, R, eps
// (fp32 bits)}; BUF_SLOTS {gamma (the compressor's norm, [D] bf16), positions
// ([T] int32), with R = 2 the state ([slots][R][2][D] fp32) and the request
// slots ([T] int64)}, graph tensors; the
// node's buffer: the latent [T][D] bf16; one counter of its tasks
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class CW,
          class... REST>
__device__ __forceinline__ void run_compressor(Maps const &,
                                               G const &g,
                                               KernelLocals &,
                                               StaticTask const &tk) {
  constexpr int D = PARAMS::v[0], R = PARAMS::v[1];
  static_assert(SELF::x == T && SELF::y == 1 && SELF::z == 1,
                "compressor: one task per token");
  static_assert(SELF::buf >= 0 && SELF::counter >= 0,
                "compressor: its output in its buffer, a counter of its tasks");
  static_assert(CW::buf >= 0 && CW::v[0] == COMBINE_SLOTS &&
                    CW::v[2] == (R == 1 ? D : 2 * D) && CW::y >= 1 &&
                    CW::y <= 8,
                "compressor: its GEMM's K parts in slots (at most 8)");
  float *state = nullptr;
  long long const *req = nullptr;
  if constexpr (R > 1) {
    state = buf_at<float>(g, slot_at<BUF_SLOTS, 2>());
    req = buf_at<long long const>(g, slot_at<BUF_SLOTS, 3>());
  }
  compressor_task<D, R, CW::v[2], CW::y, PARAMS::v[2]>(
      buf_at<float const>(g, CW::buf),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 1>()),
      state,
      req,
      buf_at<__nv_bfloat16>(g, SELF::buf),
      g.cnt + SELF::counter,
      tk.x);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_compressor() {
  return 0;
}

} // namespace static_mk
