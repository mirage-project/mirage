// tasks/hc_pre.cuh -- the second half of mHC's pre step (vLLM's
// mhc_pre_torch), task t of grid (T, 1, 1) = token t: add hc_mix's KS partial
// products of row t in order, scale by 1 / rms(R[t]) -> z = m s + b, then
//   pre  = sigmoid(z[0:HC]) + hc_eps
//   x    = RMSNorm(bf16(sum_i pre_i R_i)) gamma
// x [T][H] bf16 is the node's buffer (the next GEMM reads it through its
// tensor map); after it, HC_COEF floats per token: pre and the z of post and
// comb, which hc_post finishes (hc_comb_post below: off the path to x). Then
// the node's counter += 1.
// R[t], gamma, base and scale are loaded before the mixes are polled; warp 0
// has one coefficient per lane (comb on lanes 0 .. HC HC - 1, so a row or a
// column is HC lanes a fixed stride apart).
#pragma once
#include "../core.cuh"
#include "hc_mix.cuh"

namespace static_mk {

constexpr int HC_COEF = 32; // floats per token after x: pre [HC], then z of
                            // post [HC] and comb [HC][HC] (row i = from
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

// comb and post from z (lane l: z of coefficient hc_coef_index(l)); the whole
// warp calls it. Lane l < HC HC gets comb[l / HC][l % HC]: softmax over the row
// (lanes row0 .. row0 + HC - 1), + eps, then Sinkhorn: column, ITERS - 1 x
// (row, column) normalisations. A post lane gets post_mult sigmoid(z), the
// others 0.
template <int HC, int ITERS>
__device__ __forceinline__ float
    hc_comb_post(float z, int lane, float hc_eps, float post_mult) {
  constexpr int CB = HC * HC;
  int const row0 = lane & ~(HC - 1), col0 = lane & (HC - 1);
  float mx = -3.402823466e38f;
#pragma unroll
  for (int j = 0; j < HC; j++) {
    mx = fmaxf(mx, __shfl_sync(0xffffffffu, z, row0 + j));
  }
  float c = expf(z - mx);
  c = c / hc_lane_sum<HC>(c, row0, 1) + hc_eps;
  c = c / (hc_lane_sum<HC>(c, col0, HC) + hc_eps);
#pragma unroll 1
  for (int it = 1; it < ITERS; it++) {
    c = c / (hc_lane_sum<HC>(c, row0, 1) + hc_eps);
    c = c / (hc_lane_sum<HC>(c, col0, HC) + hc_eps);
  }
  return lane < CB ? c
                   : (lane >= CB + HC && lane < CB + 2 * HC
                          ? hc_sigmoid(z) * post_mult
                          : 0.f);
}

template <int KS, int HC, int H, int NMIX>
__device__ __forceinline__ void hc_pre_task(float const *mix,
                                            __nv_bfloat16 const *res,
                                            float const *hc_small,
                                            __nv_bfloat16 const *gamma,
                                            __nv_bfloat16 *x,
                                            float *coef_out,
                                            uint32_t *done_cnt,
                                            int t,
                                            float rms_eps,
                                            float hc_eps,
                                            float norm_eps,
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
  float *mix_s = reinterpret_cast<float *>(scratch); // [KS][HC_MIX_STRIDE]
  float *coef_s = mix_s + KS * HC_MIX_STRIDE;        // [HC_COEF]
  float *red_s = coef_s + HC_COEF;                   // [8]
  int const lane = threadIdx.x & 31;
  int const c0 = threadIdx.x * PER;
  // 0. what does not wait for hc_mix: R[t] and gamma (this thread's columns),
  // base and scale (warp 0: lane's coefficient ci)
  uint4 r[HC][PER / 8], gm[PER / 8];
#pragma unroll
  for (int k8 = 0; k8 < PER / 8; k8++) {
#pragma unroll
    for (int i = 0; i < HC; i++) {
      r[i][k8] = *reinterpret_cast<uint4 const *>(
          res + ((size_t)t * HC + i) * H + c0 + 8 * k8);
    }
    gm[k8] = *reinterpret_cast<uint4 const *>(gamma + c0 + 8 * k8);
  }
  int const ci = hc_coef_index<HC>(lane);
  float base = 0.f, scale = 0.f;
  if (threadIdx.x < NMIX) {
    base = hc_small[ci];
    scale = hc_small[NMIX + (lane < CB ? 2 : (lane < CB + HC ? 0 : 1))];
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
  __syncthreads(); // the ring area is free once every warp has left the
                   // previous task (a GEMM's MMAs read it asynchronously)
#pragma unroll
  for (int k = 0; k < NP; k++) {
    int const p = threadIdx.x + 256 * k;
    if (p < KS * Q) {
      reinterpret_cast<float4 *>(mix_s)[p] = v[k];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    STAGE_STAMP_AT(0)
  } // timing build: the mixes landed
  // 2. warp 0: the KS parts added in order, / rms(R[t]), z = m s + b of
  // coefficient ci; pre finished here, post and comb by hc_post
  if (threadIdx.x < 32) {
    float m = 0.f;
#pragma unroll 8
    for (int s = 0; s < KS; s++) {
      m += mix_s[s * HC_MIX_STRIDE + lane];
    }
    float const rr =
        rsqrtf(__shfl_sync(0xffffffffu, m, NMIX) / (float)(HC * H) + rms_eps);
    float const z = __shfl_sync(0xffffffffu, m, ci) * rr * scale + base;
    float const val = lane >= CB && lane < CB + HC ? hc_sigmoid(z) + hc_eps
                                                   : (lane < NMIX ? z : 0.f);
    coef_s[ci] = val;
    coef_out[(size_t)t * HC_COEF + ci] = val;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    STAGE_STAMP_AT(1)
  } // timing build: pre
  // 3. collapse the streams with pre (bf16, as vLLM's layer input), then
  // RMSNorm: thread j owns columns [PER j, PER j + PER)
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
  for (int k = 0; k < PER; k++) {
    y[k] = __bfloat162float(__float2bfloat16(y[k]));
    ss += y[k] * y[k];
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
    for (int k = 0; k < 8; k++) {
      float const nrm = __bfloat162float(__float2bfloat16(y[8 * k8 + k] * rn));
      ob[k] = __float2bfloat16(nrm * __bfloat162float(gb[k]));
    }
    *reinterpret_cast<uint4 *>(x + (size_t)t * H + c0 + 8 * k8) = ou;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(done_cnt, 1); // release: x and the coefficients
  }
}

// hc_pre task x = token x. MIX: the hc_mix node (its K parts MIX::x, params
// {HC, H, NMIX}); PARAMS {H, rms eps, hc eps, norm eps, post multiplier (fp32
// bits), Sinkhorn iterations (the last two: hc_post's)}; BUF_SLOTS {the
// residual streams, hc_small (base [NMIX], scale [3] fp32), gamma [H] bf16}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class MIX,
          class... IN>
__device__ __forceinline__ void run_hc_pre(Maps const &,
                                           G const &g,
                                           KernelLocals &L,
                                           StaticTask const &tk) {
  constexpr int H = PARAMS::v[0];
  constexpr int HC = MIX::v[0], NMIX = MIX::v[2];
  static_assert(MIX::buf >= 0 && MIX::v[1] == H,
                "hc_pre: its input is an hc_mix node's output of the same H");
  static_assert(SELF::x == T && SELF::buf >= 0 && SELF::counter >= 0,
                "hc_pre: one task per token, its output x and its counter");
  __nv_bfloat16 *x = buf_at<__nv_bfloat16>(g, SELF::buf);
  hc_pre_task<MIX::x, HC, H, NMIX>(
      buf_at<float const>(g, MIX::buf),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>()),
      x,
      reinterpret_cast<float *>(x + (size_t)T * H),
      g.cnt + SELF::counter,
      tk.x,
      __int_as_float(PARAMS::v[1]),
      __int_as_float(PARAMS::v[2]),
      __int_as_float(PARAMS::v[3]),
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
  static_assert((MIX::x * HC_MIX_STRIDE + HC_COEF + 8) * 4 <= RING_BYTES,
                "hc_pre: its scratch in the ring area");
  return RING_BYTES;
}

} // namespace static_mk
