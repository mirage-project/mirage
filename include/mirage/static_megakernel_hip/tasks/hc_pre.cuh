// tasks/hc_pre.cuh (HIP) -- as include/mirage/static_megakernel/tasks/hc_pre.cuh: the second half of mHC's pre step
// (SGLang's mhc_pre) for the next sublayer. Grid (T, 2, 1): both tasks of token t add hc_post_mix's KS partial mixes of
// row t, scale by 1 / rms(R[t]) -> z = m s + b; then task (t, 0)
//   pre = sigmoid(z[CB .. CB + HC)) + hc_eps
//   x   = RMSNorm(sum_i pre_i R_i) gamma        (SGLang's fused norm: squares of the fp32 sum, one rounding)
// -> x [T][H] bf16 (the node's buffer), and task (t, 1) finishes post and comb (softmax, Sinkhorn) into the graph tensors
// h_post [T][HC], h_res [T][HC HC] fp32 (SGLang's MHCState). Then the node's counter += 1.
// HIP layout: 4 waves of 64; wave 0 lanes 0 .. 31 hold one coefficient each (comb on lanes 0 .. CB - 1, pre next, post
// next: hc_coef_index); the mixes ([KS][T][HC_MIX_STRIDE] fp32, 0xFF-prefilled, st16_out by hc_post_mix) polled at
// device scope; R (hc_post_mix's R', same GPU) loaded at device scope after its counter.
#pragma once
#include "../core.cuh"

namespace static_mk {

constexpr int HC_MIX_STRIDE = 32; // floats per (part, token) of the partial mixes (hc_post_mix's output)
constexpr int HC_COEF = 32;       // a token's coefficients: pre [HC], post [HC], comb [HC][HC], 0s

__device__ __forceinline__ float hc_sigmoid(float z) {
  return 1.0f / (1.0f + expf(-z));
}

// lane l's coefficient in a token's HC_COEF floats: comb (lanes 0 .. CB - 1) at 2 HC + l, pre (the next HC lanes) at 0 ..,
// post (the next HC) at HC .., the other lanes at l (0s)
template <int HC>
__device__ __forceinline__ int hc_coef_index(int lane) {
  constexpr int CB = HC * HC;
  return lane < CB ? 2 * HC + lane : (lane < CB + 2 * HC ? lane - CB : lane);
}
// the sum over a row of comb (lanes i HC .. i HC + HC - 1) or over a column (lanes j, j + HC, ..): butterflies (offsets
// below 32: the 32-lane half of the wave)
template <int HC>
__device__ __forceinline__ float hc_row_sum(float v) {
#pragma unroll
  for (int o = 1; o < HC; o <<= 1) {
    v += __shfl_xor(v, o);
  }
  return v;
}
template <int HC>
__device__ __forceinline__ float hc_col_sum(float v) {
#pragma unroll
  for (int o = HC; o < HC * HC; o <<= 1) {
    v += __shfl_xor(v, o);
  }
  return v;
}

// comb and post from z (lane l: z of coefficient hc_coef_index(l)), the whole wave calls it: lane l < CB gets
// comb[l / HC][l % HC] (softmax over the row + eps, then Sinkhorn: column, ITERS - 1 x (row, column), each divided by its
// sum (+ eps) as SGLang's mhc_pre_torch); a post lane gets post_mult sigmoid(z); the others 0
template <int HC, int ITERS>
__device__ __forceinline__ float hc_comb_post(float z, int lane, float hc_eps, float post_mult) {
  constexpr int CB = HC * HC;
  static_assert((HC & (HC - 1)) == 0 && CB <= 16, "hc_comb_post: HC a power of 2, comb on lanes 0 .. 15");
  float mx = z;
#pragma unroll
  for (int o = 1; o < HC; o <<= 1) {
    mx = fmaxf(mx, __shfl_xor(mx, o));
  }
  float c = expf(z - mx);
  c = __fdividef(c, hc_row_sum<HC>(c)) + hc_eps;
  c = __fdividef(c, hc_col_sum<HC>(c) + hc_eps);
#pragma unroll 1
  for (int it = 1; it < ITERS; it++) {
    c = __fdividef(c, hc_row_sum<HC>(c) + hc_eps);
    c = __fdividef(c, hc_col_sum<HC>(c) + hc_eps);
  }
  return lane < CB ? c : (lane >= CB + HC && lane < CB + 2 * HC ? hc_sigmoid(z) * post_mult : 0.f);
}

// (hc_post_mix.cuh: from T STATIC_HCPM_RPOLL_MIN_T hc_pre's buffer slot 5 = hc_post_mix's R' copy; the threshold read here
// before that file)
#ifndef STATIC_HCPM_RPOLL_MIN_T
#define STATIC_HCPM_RPOLL_MIN_T 16
#endif
#define STATIC_HCPM_RPOLL_PRE (T >= STATIC_HCPM_RPOLL_MIN_T)
template <int KS, int HC, int H, int NMIX, int ITERS>
__device__ __forceinline__ void hc_pre_task(G const &g,
                                            __hip_bfloat16 const *rpoll, // (T >= STATIC_HCPM_RPOLL_MIN_T: hc_post_mix's R' copy, or nullptr)
                                            float const *mix,
                                            __hip_bfloat16 const *res,
                                            float const *hc_small,
                                            __hip_bfloat16 const *gamma,
                                            __hip_bfloat16 *x,
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
  constexpr int CB = HC * HC;
  static_assert(NMIX == CB + 2 * HC && NMIX + 1 <= HC_MIX_STRIDE && HC_COEF == 32,
                "hc_pre: the coefficients are 32 lanes of wave 0");
  constexpr int PER = H / 256; // columns per thread
  static_assert(PER * 256 == H && PER % 8 == 0, "hc_pre: H a multiple of 2048 (8 columns per load)");
  constexpr int Q = HC_MIX_STRIDE / 4; // float4s per (part, token): 8
  constexpr int NP = (KS * Q + 255) / 256;
  static_assert(256 % Q == 0, "hc_pre: a thread's float4s are one index of the row");
  float *mix_s = reinterpret_cast<float *>(scratch); // [4 waves][HC_MIX_STRIDE]
  float *coef_s = mix_s + 4 * HC_MIX_STRIDE;         // [HC_COEF]
  float *red_s = coef_s + HC_COEF;                   // [4]
  float *out_s = red_s + 4;                          // [HC + CB] (part 1's results, for 16-B stores)
  int const tid = STATIC_TIDX, lane = tid & 63, wave = tid >> 6;
  int const c0 = tid * PER;
  // 0. what does not wait for the mixes: gamma, base and scale (graph inputs); then R[t] (after its count)
  uint4 r[HC][PER / 8], gm[PER / 8];
  if (part == 0) {
#pragma unroll
    for (int k8 = 0; k8 < PER / 8; k8++) {
      gm[k8] = *reinterpret_cast<uint4 const *>(gamma + c0 + 8 * k8);
    }
  }
  int const ci = hc_coef_index<HC>(lane);
  float base = 0.f, scale = 0.f;
  if (tid < NMIX) {
    base = hc_small[ci];
    scale = hc_small[NMIX + (lane < CB ? 2 : (lane < CB + HC ? 0 : 1))];
  }
  if (part == 0) {
    if (rpoll != nullptr) { // (R' polled in hc_post_mix's 0xFF-prefilled copy, no count wait)
      for (;;) {
        bool all = true;
#pragma unroll
        for (int k8 = 0; k8 < PER / 8; k8++) {
#pragma unroll
          for (int i = 0; i < HC; i++) {
            r[i][k8] = ld16_in_at(rpoll, (uint32_t)((((size_t)t * HC + i) * H + c0 + 8 * k8) * 2));
            all = all && valid16w(r[i][k8]);
          }
        }
        if (all) {
          break;
        }
        poll_pause();
      }
    } else {
      if (res_cnt != nullptr) {
        cnt_wait_in(res_cnt, res_need);
      }
#pragma unroll
      for (int k8 = 0; k8 < PER / 8; k8++) {
#pragma unroll
        for (int i = 0; i < HC; i++) {
          r[i][k8] = ld16_in_at(res, (uint32_t)((((size_t)t * HC + i) * H + c0 + 8 * k8) * 2));
        }
      }
    }
  }
  // 1. the mixes: every (part, float4) of row t polled at once (0xFF prefill)
  float4 v[NP];
#pragma unroll
  for (int k = 0; k < NP; k++) {
    int const p = tid + 256 * k;
    v[k] = make_float4(0.f, 0.f, 0.f, 0.f);
    if (p < KS * Q) {
      v[k] = ldf4_dev(mix + ((size_t)(p / Q) * T + t) * HC_MIX_STRIDE + 4 * (p % Q));
    }
  }
#pragma unroll
  for (int k = 0; k < NP; k++) {
    int const p = tid + 256 * k;
    if (p < KS * Q) {
      while (!validf4(v[k])) {
        poll_pause();
        v[k] = ldf4_dev(mix + ((size_t)(p / Q) * T + t) * HC_MIX_STRIDE + 4 * (p % Q));
      }
    }
  }
  // the KS parts added: a thread's NP (its float4 index tid % Q), then the lanes with the same index (xor Q .. 32), then
  // the 4 waves
  float4 a = v[0];
#pragma unroll
  for (int k = 1; k < NP; k++) {
    a.x += v[k].x;
    a.y += v[k].y;
    a.z += v[k].z;
    a.w += v[k].w;
  }
#pragma unroll
  for (int o = Q; o < 64; o <<= 1) {
    a.x += __shfl_xor(a.x, o);
    a.y += __shfl_xor(a.y, o);
    a.z += __shfl_xor(a.z, o);
    a.w += __shfl_xor(a.w, o);
  }
  __syncthreads(); // the LDS is free once every wave has left the previous task
  if (lane < Q) {
    reinterpret_cast<float4 *>(mix_s)[wave * Q + lane] = a;
  }
  __syncthreads();
  STAGE_STAMP_AT(0)
  // 2. wave 0: the waves' sums, / rms(R[t]), z = m s + b of coefficient ci; pre finished here, post and comb by (t, 1)
  if (tid < 32) {
    float m = 0.f;
#pragma unroll
    for (int w = 0; w < 4; w++) {
      m += mix_s[w * HC_MIX_STRIDE + lane];
    }
    float const rr = rsqrtf(__shfl(m, NMIX) / (float)(HC * H) + rms_eps);
    float const z = __shfl(m, ci) * rr * scale + base;
    float const val = lane >= CB && lane < CB + HC ? hc_sigmoid(z) + hc_eps : (lane < NMIX ? z : 0.f);
    coef_s[ci] = val;
  }
  __syncthreads();
  if (part == 1) { // post and comb (wave 0), no x
    if (tid < 32) {
      float const c = hc_comb_post<HC, ITERS>(coef_s[ci], lane, hc_eps, post_mult);
      if (lane < CB) {
        out_s[HC + lane] = c;
      } else if (lane >= CB + HC && lane < CB + 2 * HC) {
        out_s[lane - CB - HC] = c;
      }
    }
    __syncthreads();
    static_assert(HC == 4, "hc_pre: h_post one 16-B vector, h_res HC of them");
    if (tid < 1 + HC) { // thread 0: h_post [t][0 .. 4); thread 1 + i: h_res [t][4 i .. 4 i + 4)
      float const *src = out_s + 4 * tid;
      uint4 const u = make_uint4(__float_as_uint(src[0]), __float_as_uint(src[1]), __float_as_uint(src[2]),
                                 __float_as_uint(src[3]));
      st16_out(tid == 0 ? (void *)(h_post_out + (size_t)t * HC) : (void *)(h_res_out + (size_t)t * CB + 4 * (tid - 1)),
               u);
    }
    STAGE_STAMP_AT(1)
    task_post(done_cnt, 1); // release: the coefficients
    return;
  }
  // 3. collapse the streams with pre, then RMSNorm: thread j owns columns [PER j, PER j + PER)
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
      uint32_t const w[4] = {r[i][k8].x, r[i][k8].y, r[i][k8].z, r[i][k8].w};
#pragma unroll
      for (int k = 0; k < 4; k++) {
        y[8 * k8 + 2 * k] += pi * bf16_lo(w[k]);
        y[8 * k8 + 2 * k + 1] += pi * bf16_hi(w[k]);
      }
    }
  }
#pragma unroll
  for (int k = 0; k < PER; k++) { // SGLang's fused norm: the squares of the fp32 sum
    ss += y[k] * y[k];
    y[k] = round_bf16(y[k]);
  }
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) {
    ss += __shfl_xor(ss, o);
  }
  if (lane == 0) {
    red_s[wave] = ss;
  }
  __syncthreads();
  float const tot = red_s[0] + red_s[1] + red_s[2] + red_s[3];
  float const rn = rsqrtf(tot / (float)H + norm_eps);
#pragma unroll
  for (int k8 = 0; k8 < PER / 8; k8++) {
    uint32_t const gw[4] = {gm[k8].x, gm[k8].y, gm[k8].z, gm[k8].w};
    uint32_t ow[4];
#pragma unroll
    for (int k = 0; k < 4; k++) { // one rounding
      ow[k] = pack_bf16x2(y[8 * k8 + 2 * k] * rn * bf16_lo(gw[k]), y[8 * k8 + 2 * k + 1] * rn * bf16_hi(gw[k]));
    }
    st16_out(x + (size_t)t * H + c0 + 8 * k8, make_uint4(ow[0], ow[1], ow[2], ow[3]));
  }
  STAGE_STAMP_AT(1)
  task_post(done_cnt, 1); // release: x
}

// hc_pre task (token x, part y). MIX: the hc_post_mix node (its K parts MIX::x, params {HC, H, NMIX[, PUSH]}); RES: the
// streams' producer (a graph input, or hc_post_mix: one counter, x tasks); PARAMS {H, rms eps, hc eps, norm eps, post
// multiplier (fp32 bits), Sinkhorn iterations}; BUF_SLOTS {the residual streams, hc_small (base [NMIX], scale [3] fp32),
// gamma [H] bf16, h_post, h_res}
template <bool ON, class BUF_SLOTS, int K, class P>
__device__ __forceinline__ P const *hc_pre_slot(G const &g) {
  if constexpr (ON) {
    return buf_at<P const>(g, slot_at<BUF_SLOTS, K>());
  } else {
    return nullptr;
  }
}
template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class MIX, class RES, class... IN>
__device__ __forceinline__ void run_hc_pre(Maps const &, G const &g, KernelLocals &L, StaticTask const &tk) {
  constexpr int H = PARAMS::v[0];
  constexpr int HC = MIX::v[0], NMIX = MIX::v[2];
  static_assert(MIX::buf >= 0 && MIX::v[1] == H, "hc_pre: its input is an hc_post_mix node's output of the same H");
  static_assert(SELF::x == T && SELF::y == 2 && SELF::buf >= 0 && SELF::counter >= 0,
                "hc_pre: two tasks per token, its output x and its counter");
  // (RES: hc_post_mix counts its x y tasks in one counter; y > 1: token groups, STATIC_HCPM_TG)
  hc_pre_task<MIX::x, HC, H, NMIX, PARAMS::v[5]>(
      g, hc_pre_slot<STATIC_HCPM_RPOLL_PRE, BUF_SLOTS, 5, __hip_bfloat16>(g), buf_at<float const>(g, MIX::buf),
                                                 buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
                                                 buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()),
                                                 buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>()),
                                                 buf_at<__hip_bfloat16>(g, SELF::buf),
                                                 buf_at<float>(g, slot_at<BUF_SLOTS, 3>()),
                                                 buf_at<float>(g, slot_at<BUF_SLOTS, 4>()),
                                                 cnt_of(g) + SELF::counter,
                                                 RES::counter >= 0 ? cnt_of(g) + RES::counter : nullptr,
                                                 RES::x * RES::y,
                                                 tk.x,
                                                 __int_as_float(PARAMS::v[1]),
                                                 __int_as_float(PARAMS::v[2]),
                                                 __int_as_float(PARAMS::v[3]),
                                                 __int_as_float(PARAMS::v[4]),
                                                 tk.y,
                                                 L.sm STAGE_STAMP_ARG(L.stage_stamps));
}

template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class MIX, class... IN>
constexpr int smem_hc_pre() {
  return (4 * HC_MIX_STRIDE + HC_COEF + 4 + 32) * 4;
}

} // namespace static_mk
