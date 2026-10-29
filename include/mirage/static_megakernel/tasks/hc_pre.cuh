// tasks/hc_pre.cuh -- the second half of mHC's pre step (SGLang's
// mhc_pre_torch), for the next sublayer, which runs after the layer (a serving
// framework's: SGLang's boundary). Grid (T, 2, 1): both tasks of token t add
// hc_post_mix's KS partial products of row t (a tree over the threads), scale
// by 1 / rms(R[t]) -> z = m s + b; then task (t, 0)
//   pre  = sigmoid(z[0:HC]) + hc_eps
//   x    = RMSNorm(sum_i pre_i R_i) gamma
// with the norm as SGLang's fused mhc_pre (mhc.py mhc_pre_big_fuse_with_norm:
// the sum of squares of the fp32 weighted sum, x = bf16(bf16(sum) rsqrt w), one
// rounding) -> x [T][H] bf16, the node's buffer; and task (t, 1) finishes post
// and comb (hc_comb_post: 20 dependent Sinkhorn steps, ~3 us) into graph
// tensors h_post [T][HC], h_res [T][HC HC] fp32 (SGLang's MHCState), at the
// same time. Then the node's counter += 1.
// R[t], gamma, base and scale are loaded before the mixes are polled (R from a
// node of this layer: after its count, hc_post_mix's one); warp 0 has one
// coefficient per lane (comb on lanes 0 .. HC HC - 1, so a row or a column is
// HC lanes a fixed stride apart).
#pragma once
#include "../core.cuh"

namespace static_mk {

constexpr int HC_MIX_STRIDE = 32; // floats per (part, token) of the partial
                                  // mixes hc_pre adds (hc_post_mix's output)

__device__ __forceinline__ float4 f4_add(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__device__ __forceinline__ float4 f4_shfl_xor(float4 a, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, a.x, o),
                     __shfl_xor_sync(0xffffffffu, a.y, o),
                     __shfl_xor_sync(0xffffffffu, a.z, o),
                     __shfl_xor_sync(0xffffffffu, a.w, o));
}

constexpr int HC_COEF = 32; // floats of a token's coefficients: pre [HC], then
                            // (z of) post [HC] and comb [HC][HC] (row i = from
                            // stream i), 0s

__device__ __forceinline__ float hc_sigmoid(float z) {
  return 1.0f / (1.0f + expf(-z));
}

// the sum of v over the HC lanes first, first + stride, ... (in that order)
template <int HC>
__device__ __forceinline__ float hc_lane_sum(float v, int first, int stride) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < HC; i++) {
    s += __shfl_sync(0xffffffffu, v, first + i * stride);
  }
  return s;
}

// lane l's coefficient in the HC_COEF floats of a token: comb (lanes 0 .. HC
// HC - 1) at 2 HC + l, pre (the next HC lanes) at 0 .., post (the next HC) at
// HC .., the other lanes at l (0s)
template <int HC>
__device__ __forceinline__ int hc_coef_index(int lane) {
  constexpr int CB = HC * HC;
  return lane < CB ? 2 * HC + lane : (lane < CB + 2 * HC ? lane - CB : lane);
}

// the sum of v over the HC lanes of a row of comb (lanes i HC .. i HC + HC - 1)
// or of a column (lanes j, j + HC, ...): a butterfly (every lane of the row or
// column gets the same sum)
template <int HC>
__device__ __forceinline__ float hc_row_sum(float v) {
#pragma unroll
  for (int o = 1; o < HC; o <<= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
  }
  return v;
}
template <int HC>
__device__ __forceinline__ float hc_col_sum(float v) {
#pragma unroll
  for (int o = HC; o < HC * HC; o <<= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, o);
  }
  return v;
}

// comb and post from z (lane l: z of coefficient hc_coef_index(l)); the whole
// warp calls it. Lane l < HC HC gets comb[l / HC][l % HC]: softmax over the
// row,
// + eps, then Sinkhorn: column, ITERS - 1 x (row, column) normalisations, each
// element divided by its row's or column's sum (+ eps) as vLLM's / SGLang's
// mhc_pre_torch, one element per lane, the sums butterflies, the fast division
// (__fdividef, 2 ulp; the coefficients move within 1e-7: the 2 ITERS steps are
// one dependent chain, an IEEE division in it ~100 cycles). A post lane gets
// post_mult sigmoid(z), the others 0.
template <int HC, int ITERS>
__device__ __forceinline__ float
    hc_comb_post(float z, int lane, float hc_eps, float post_mult) {
  constexpr int CB = HC * HC;
  static_assert((HC & (HC - 1)) == 0 && CB <= 16,
                "hc_comb_post: HC a power of 2, comb on lanes 0 .. 15");
  float mx = z;
#pragma unroll
  for (int o = 1; o < HC; o <<= 1) {
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  }
  float c = expf(z - mx);
  c = __fdividef(c, hc_row_sum<HC>(c)) + hc_eps;
  c = __fdividef(c, hc_col_sum<HC>(c) + hc_eps);
#pragma unroll 1
  for (int it = 1; it < ITERS; it++) {
    c = __fdividef(c, hc_row_sum<HC>(c) + hc_eps);
    c = __fdividef(c, hc_col_sum<HC>(c) + hc_eps);
  }
  return lane < CB ? c
                   : (lane >= CB + HC && lane < CB + 2 * HC
                          ? hc_sigmoid(z) * post_mult
                          : 0.f);
}

template <int KS, int HC, int H, int NMIX, int ITERS>
__device__ __forceinline__ void hc_pre_task(float const *mix,
                                            __nv_bfloat16 const *res,
                                            float const *hc_small,
                                            __nv_bfloat16 const *gamma,
                                            __nv_bfloat16 *x,
                                            float *h_post_out,
                                            float *h_res_out,
                                            uint32_t *done_cnt,
                                            uint32_t const *res_cnt,
                                            uint32_t res_need,
                                            int t,
                                            float rms_eps,
                                            float hc_eps,
                                            float norm_eps,
                                            float post_mult,
                                            int part,
                                            char *scratch STAGE_STAMP_PARAM) {
  constexpr int CB = HC * HC; // comb lanes; pre: CB .. CB + HC - 1, post next
  static_assert(NMIX == CB + 2 * HC && NMIX + 1 <= HC_MIX_STRIDE &&
                    HC_COEF == 32,
                "hc_pre: the coefficients are one warp's lanes");
  constexpr int PER = H / 256; // columns per thread
  static_assert(PER * 256 == H && PER % 8 == 0,
                "hc_pre: H a multiple of 2048 (8 columns per load)");
  constexpr int Q = HC_MIX_STRIDE / 4; // float4s per (part, token)
  constexpr int NP = (KS * Q + 255) / 256;
  static_assert(32 % Q == 0, "hc_pre: a warp holds whole rows of the mixes");
  float *mix_s = reinterpret_cast<float *>(scratch); // [8][HC_MIX_STRIDE]
  float *coef_s = mix_s + 8 * HC_MIX_STRIDE;         // [HC_COEF]
  float *red_s = coef_s + HC_COEF;                   // [8]
  int const lane = threadIdx.x & 31;
  int const c0 = threadIdx.x * PER;
  // 0. what does not wait for the mixes: gamma (this thread's columns), base
  // and scale (warp 0: lane's coefficient ci): graph inputs, loaded first (from
  // HBM, in flight during the waits); then R[t] (this thread's columns; from a
  // node of this layer: after token t's count, a tight poll)
  uint4 r[HC][PER / 8], gm[PER / 8];
  if (part == 0) { // the export part needs no R (the mixes hold its rms)
#pragma unroll
    for (int k8 = 0; k8 < PER / 8; k8++) {
      gm[k8] = __ldg(reinterpret_cast<uint4 const *>(gamma + c0 + 8 * k8));
    }
  }
  int const ci = hc_coef_index<HC>(lane);
  float base = 0.f, scale = 0.f;
  if (threadIdx.x < NMIX) {
    base = hc_small[ci];
    scale = hc_small[NMIX + (lane < CB ? 2 : (lane < CB + HC ? 0 : 1))];
  }
  if (part == 0 && res_cnt != nullptr) {
    if (threadIdx.x == 0) {
      while (cnt_ld(res_cnt) < res_need) {
      } // acquire: token t's R (its count)
    }
    __syncthreads();
  }
  if (part == 0) {
#pragma unroll
    for (int k8 = 0; k8 < PER / 8; k8++) {
#pragma unroll
      for (int i = 0; i < HC; i++) {
        r[i][k8] = *reinterpret_cast<uint4 const *>(
            res + ((size_t)t * HC + i) * H + c0 + 8 * k8);
      }
    }
  }
  // 1. the mixes: every (part, float4) of row t polled at once (0xFF prefill)
  float4 v[NP];
#pragma unroll
  for (int k = 0; k < NP; k++) {
    int const p = threadIdx.x + 256 * k;
    if (p < KS * Q) {
      v[k] = ldf4_poll(mix + ((size_t)(p / Q) * T + t) * HC_MIX_STRIDE +
                       4 * (p % Q));
    }
  }
#pragma unroll
  for (int k = 0; k < NP; k++) {
    int const p = threadIdx.x + 256 * k;
    if (p < KS * Q) {
      while (!validf4(v[k])) {
        v[k] = ldf4_poll(mix + ((size_t)(p / Q) * T + t) * HC_MIX_STRIDE +
                         4 * (p % Q));
      }
    }
  }
  // the KS parts added: a thread's NP (float4 threadIdx.x % Q of each), then
  // over the lanes with the same float4 (a butterfly), then over the warps
  float4 a = v[0];
#pragma unroll
  for (int k = 1; k < NP; k++) {
    if (threadIdx.x + 256 * k < KS * Q) {
      a = f4_add(a, v[k]);
    }
  }
#pragma unroll
  for (int o = Q; o < 32; o <<= 1) {
    a = f4_add(a, f4_shfl_xor(a, o));
  }
  __syncthreads(); // the ring area is free once every warp has left the
                   // previous task (a GEMM's MMAs read it asynchronously)
  if (lane < Q) {
    reinterpret_cast<float4 *>(mix_s)[(threadIdx.x >> 5) * Q + lane] = a;
  }
  __syncthreads();
  STAGE_STAMP_T0(0) // timing build: the mixes landed
  // 2. warp 0: the 8 warps' sums added, / rms(R[t]), z = m s + b of
  // coefficient ci; pre finished here, post and comb by task (t, 1)
  if (threadIdx.x < 32) {
    float m = 0.f;
#pragma unroll
    for (int w = 0; w < 8; w++) {
      m += mix_s[w * HC_MIX_STRIDE + lane];
    }
    float const rr =
        rsqrtf(__shfl_sync(0xffffffffu, m, NMIX) / (float)(HC * H) + rms_eps);
    float const z = __shfl_sync(0xffffffffu, m, ci) * rr * scale + base;
    float const val = lane >= CB && lane < CB + HC ? hc_sigmoid(z) + hc_eps
                                                   : (lane < NMIX ? z : 0.f);
    coef_s[ci] = val;
  }
  __syncthreads();
  if (part == 1) { // post and comb finished (warp 0), no x
    if (threadIdx.x < 32) {
      float const c =
          hc_comb_post<HC, ITERS>(coef_s[ci], lane, hc_eps, post_mult);
      if (lane < CB) {
        h_res_out[(size_t)t * CB + lane] = c;
      } else if (lane >= CB + HC && lane < CB + 2 * HC) {
        h_post_out[(size_t)t * HC + lane - CB - HC] = c;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      STAGE_STAMP_AT(1)
      cnt_add(done_cnt, 1); // release: the coefficients
    }
    return;
  }
  STAGE_STAMP_T0(1) // timing build: pre
  // 3. collapse the streams with pre, then RMSNorm: thread j owns columns
  // [PER j, PER j + PER)
  float y[PER];
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < PER; k++) {
    y[k] = 0.f;
  }
#pragma unroll
  for (int i = 0; i < HC; i++) {
    float const pi = coef_s[i];
#pragma unroll
    for (int k8 = 0; k8 < PER / 8; k8++) {
      __nv_bfloat16 const *b =
          reinterpret_cast<__nv_bfloat16 const *>(&r[i][k8]);
#pragma unroll
      for (int k = 0; k < 8; k++) {
        y[8 * k8 + k] += pi * __bfloat162float(b[k]);
      }
    }
  }
#pragma unroll
  for (int k = 0; k < PER; k++) { // SGLang's fused norm: the squares of the
                                  // fp32 sum
    ss += y[k] * y[k];
    y[k] = __bfloat162float(__float2bfloat16(y[k]));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, o);
  }
  if (lane == 0) {
    red_s[threadIdx.x >> 5] = ss;
  }
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int w = 0; w < 8; w++) {
    tot += red_s[w];
  }
  float const rn = rsqrtf(tot / (float)H + norm_eps);
#pragma unroll
  for (int k8 = 0; k8 < PER / 8; k8++) {
    __nv_bfloat16 const *gb = reinterpret_cast<__nv_bfloat16 const *>(&gm[k8]);
    uint4 ou;
    __nv_bfloat16 *ob = reinterpret_cast<__nv_bfloat16 *>(&ou);
#pragma unroll
    for (int k = 0; k < 8; k++) { // SGLang's fused norm: one rounding
      ob[k] = __float2bfloat16(y[8 * k8 + k] * rn * __bfloat162float(gb[k]));
    }
    *reinterpret_cast<uint4 *>(x + (size_t)t * H + c0 + 8 * k8) = ou;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(done_cnt, 1); // release: x
  }
}

// hc_pre task (token x, part y). MIX: the hc_post_mix node (its K parts MIX::x,
// params {HC, H, NMIX}); RES: the streams' producer (a graph input, or
// hc_post_mix: one counter, x tasks); PARAMS {H, rms eps, hc eps, norm eps,
// post multiplier (fp32 bits), Sinkhorn iterations}; BUF_SLOTS {the residual
// streams, hc_small (base [NMIX], scale [3] fp32), gamma [H] bf16, h_post,
// h_res}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class MIX,
          class RES,
          class... IN>
__device__ __forceinline__ void run_hc_pre(Maps const &,
                                           G const &g,
                                           KernelLocals &L,
                                           StaticTask const &tk) {
  constexpr int H = PARAMS::v[0];
  constexpr int HC = MIX::v[0], NMIX = MIX::v[2];
  static_assert(
      MIX::buf >= 0 && MIX::v[1] == H,
      "hc_pre: its input is an hc_post_mix node's output of the same H");
  static_assert(SELF::x == T && SELF::y == 2 && SELF::buf >= 0 &&
                    SELF::counter >= 0,
                "hc_pre: two tasks per token, its output x and its counter");
  static_assert(RES::counter < 0 || RES::y == 1,
                "hc_pre: R's producer counts in one counter (hc_post_mix: grid "
                "(parts, 1))");
  hc_pre_task<MIX::x, HC, H, NMIX, PARAMS::v[5]>(
      buf_at<float const>(g, MIX::buf),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<__nv_bfloat16>(g, SELF::buf),
      buf_at<float>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<float>(g, slot_at<BUF_SLOTS, 4>()),
      cnt_now(g) + SELF::counter,
      RES::counter >= 0 ? cnt_now(g) + RES::counter : nullptr,
      RES::x,
      tk.x,
      __int_as_float(PARAMS::v[1]),
      __int_as_float(PARAMS::v[2]),
      __int_as_float(PARAMS::v[3]),
      __int_as_float(PARAMS::v[4]),
      tk.y,
      L.sm STAGE_STAMP_ARG(L.stage_stamps));
}

// dynamic shared memory: the scratch, inside the ring area
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class MIX,
          class... IN>
constexpr int smem_hc_pre() {
  static_assert((8 * HC_MIX_STRIDE + HC_COEF + 8) * 4 <= RING_BYTES,
                "hc_pre: its scratch in the ring area");
  return RING_BYTES;
}

} // namespace static_mk
