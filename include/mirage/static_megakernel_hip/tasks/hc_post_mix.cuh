// tasks/hc_post_mix.cuh (HIP) -- as include/mirage/static_megakernel/tasks/hc_post_mix.cuh: mHC's post step of a TP
// sublayer and the first half of the next sublayer's pre step in one task. Task s of grid (H / 32, 1, 1) = columns
// [32 s, 32 s + 32) of every stream of every token:
//   1. before the sums land: fn's slice (the next sublayer's hc_*_fn, NMIX rows x the task's HC 32 columns) and the
//      tokens' coefficients (h_post [T][HC], h_res [T][HC HC] fp32: the sublayer's, finished) into LDS; thread (t, v)
//      (v: 8 columns) loads R[t][i] of its columns (HC 16-B vectors)
//   2. thread (t, v) lands the GPUs' bf16 partial sums of its 8 columns (allreduce_land_bf16: fp32, GPU order), rounded to
//      bf16 (the all-reduced value): from an allreduce_send node ([GPUS][T][H]) or a push_all gemm_tile
//      ([GPUS][H / PR][T][PR], PR = its rows per task)
//   3. R'[t][j][c] = bf16(post[j] out[c] + sum_i comb[i][j] R[t][i][c]), rounded as SGLang's _mhc_post_torch (each product,
//      the sum over i in order, then post out added: no FMA) -> R' (a graph tensor, 16-B device-scope stores) and LDS
//   4. the partial mixes of the task's columns: mix[t][r] = sum_k R'[t][k] fn[r][k] (k < HC 32, fp32 FMAs in k order),
//      entry NMIX the sum of squares of R'[t], then 0s up to HC_MIX_STRIDE -> row (s, t) of the output
//      ([KS][T][HC_MIX_STRIDE] fp32, 0xFF-prefilled, which hc_pre polls), 16-B device-scope stores
// Counter: one; every task adds 1 (hc_pre reads R' once it is the task count).
#pragma once
#include "../core.cuh"
#include "hc_pre.cuh"

namespace static_mk {

// After a push_all / finish_all gemm_tile, from T STATIC_HCPM_FEWPOLL_MIN_T: before the landing, thread 0 alone polls one vector of
// every GPU's slot (the task's last token, its last 8 columns) until all are valid; then every thread lands as usual (each
// still polls its own vectors). The polling of 128 CUs x 256 threads (4 system-scope 16-B loads a round) slowed the other
// GPUs' pushes landing: MI355X 11-08 glmkda12, KDA T 64 landed +15.8 vs +27.2 us, e2e 115.6 vs 131.8; T 32 80.1 vs 92.9;
// MoE T 16 (allreduce_send) 170.8 vs 168.8
// (only after a push_all gemm_tile, from T 8: MI355X 11-08 glmkda13, A B A B: KDA T 1 / 8 / 32 / 64 +1.4 / -4 / -12.5 / -15 us, DSA +1.9 / 0 / -4.4 /
// -2, dense +2.1 / +1.6 / 0 / -9.6)
#ifndef STATIC_HCPM_FEWPOLL_MIN_T
#define STATIC_HCPM_FEWPOLL_MIN_T 8
#endif
// STATIC_HCPM_MFMA_MIN_T (default 16): the smallest task token count of the MFMA mixes (16: one 16-token tile; below it a wave
// per token, 25 dependent wave sums each). MI355X 11-08 glmhc4, T 16, 17 vs 16 (A B A B): MoE 157.4 / 160.3 vs 154.9 / 152.8 us,
// KDA 72.1 / 74.2 vs 70.2 / 69.7, DSA 119.8 / 119.4 vs 114.2 / 115.2, dense 63.4 / 63.0 vs 60.5 / 59.9
// From T STATIC_HCPM_RPOLL_MIN_T (16; the host's MPK_HCPM_RPOLL_MIN_T must match): R' also into
// a 0xFF-prefilled scratch (buffer slot 5) that hc_pre polls with the mixes - no wait for hc_post_mix's count before its R'
// loads (one round trip less). MI355X 11-08 glmhc5 (A B A B): MoE T 16 153.6 / 150.4 vs 148.5 / 149.7 us, T 64 257.4 / 258.8 vs
// 254.4 / 254.1, T 1 level; KDA T 64 116.7 / 114.2 vs 113.2 / 112.5; dense T 64 93.6 / 93.0 vs 90.9 / 89.3, T 8 +1.7; DSA level
#ifndef STATIC_HCPM_RPOLL_MIN_T
#define STATIC_HCPM_RPOLL_MIN_T 16
#endif
#ifndef STATIC_HCPM_MFMA_MIN_T
#define STATIC_HCPM_MFMA_MIN_T 16
#endif

template <size_t SOFF, int H, int HC, int NMIX, int PR, int NSEND = 1, int SEND_CNT = -1, int TG = 1, int NPP = 1, bool HAND = false> // SOFF: the sums' offset in the exchange region; PR 0: row-major
__device__ __forceinline__ void hc_post_mix_task(G const &g,
                                                 float const *h_post,
                                                 float const *h_res,
                                                 __hip_bfloat16 const *res,
                                                 __hip_bfloat16 *rout,
                                                 float const *fn,
                                                 float *mix,
                                                 uint32_t *done_cnt,
                                                 int s,
                                                 int tg, // (TG > 1: this task's token group, tokens [tg TL, tg TL + TL))
                                                 __hip_bfloat16 *rpoll, // (T >= STATIC_HCPM_RPOLL_MIN_T: R' again, polled by hc_pre)
                                                 __hip_bfloat16 *contract, // (or null: the mean of R' over the streams, [T][H])
                                                 char *scratch STAGE_STAMP_PARAM) {
#pragma clang fp contract(off)
  constexpr size_t S_RANK = (size_t)T * H * 2; // one GPU's slot
  constexpr int TL = T / TG;                    // (the task's tokens)
  static_assert(TL * TG == T, "hc_post_mix: whole token groups");
  int const t0 = tg * TL;
  // KP, FS: an R' row, an fn row in LDS (+1 float: the rows on other banks; MI355X 11-07: fn rows 128 floats apart read by
  // the lanes of one token, 8-way bank conflicts, the mixes at T 64 ~26 us)
  constexpr int C = 32, CB = HC * HC, KW = HC * C, KP = KW + 1, FS = KW + 1;
  static_assert(4 * TL <= 256 && H % C == 0 && C == HC_MIX_STRIDE && NMIX + 1 <= HC_MIX_STRIDE,
                "hc_post_mix: a thread per (token, 8 columns), 32 columns a task");
  static_assert(PR == 0 || (PR % 8 == 0 && H % PR == 0), "hc_post_mix: tile-major partials in whole 16-B vectors");
  float *fn_s = reinterpret_cast<float *>(scratch); // [NMIX][FS]
  float *rp_s = fn_s + NMIX * FS;                    // [T][KP]
  float *cf_s = rp_s + T * KP;                       // [T][HC + CB]: post, then comb
  int const tid = STATIC_TIDX;
  int const tl = tid >> 2, t = t0 + tl, v = tid & 3, c0 = s * C + 8 * v; // thread (t, v): token t, columns [c0, c0 + 8)
  bool const act = tid < 4 * TL;
  // 1. what does not wait for the sums
  constexpr int FNV = (NMIX * KW + 255) / 256;
  float fv[FNV];
#pragma unroll
  for (int q = 0; q < FNV; q++) {
    int const idx = tid + 256 * q, r = idx / KW, k = idx % KW;
    fv[q] = idx < NMIX * KW ? fn[(size_t)r * HC * H + (size_t)(k / C) * H + s * C + k % C] : 0.f;
  }
  constexpr int CFV = (TL * (HC + CB) + 255) / 256;
  float cv[CFV];
#pragma unroll
  for (int q = 0; q < CFV; q++) {
    int const idx = tid + 256 * q, tt = idx / (HC + CB), k = idx % (HC + CB);
    cv[q] = idx < TL * (HC + CB) ? (k < HC ? h_post[(size_t)(t0 + tt) * HC + k] : h_res[(size_t)(t0 + tt) * CB + k - HC]) : 0.f;
  }
  uint4 rv[HC];
  if (act) {
#pragma unroll
    for (int i = 0; i < HC; i++) {
      rv[i] = ld16_in_at(res, (uint32_t)((((size_t)t * HC + i) * H + c0) * 2));
    }
  }
  __syncthreads(); // the LDS is free once every wave has left the previous task
#pragma unroll
  for (int q = 0; q < FNV; q++) {
    int const idx = tid + 256 * q;
    if (idx < NMIX * KW) {
      fn_s[(idx / KW) * FS + idx % KW] = fv[q];
    }
  }
#pragma unroll
  for (int q = 0; q < CFV; q++) {
    int const idx = tid + 256 * q;
    if (idx < TL * (HC + CB)) {
      cf_s[idx] = cv[q];
    }
  }
  STAGE_STAMP_AT(0)
  // (SLAND: allreduce_send landed and added the sums, core.cuh)
  constexpr bool SLAND = !SOLO && PR == 0 && T >= STATIC_ARS_LAND_MIN_T;
  // (every thread still polls its own vectors below: the results do not depend on it)
  if constexpr (!SOLO && T >= STATIC_HCPM_FEWPOLL_MIN_T && PR > 0) { // (thread 0 alone polls one vector per GPU first)
    if (tid == 0) {
      int const tq = t0 + TL - 1, cl = s * C + 24;
      size_t const ol = PR == 0 ? (size_t)tq * H * 2 + (size_t)cl * 2
                                : (size_t)(cl / PR) * T * PR * 2 + (size_t)tq * PR * 2 + (size_t)(cl % PR) * 2;
      float od[8];
      if constexpr (NPP > 1) {
        allreduce_land_bf16_nr<S_RANK, GPUS * NPP>(g.rv + SOFF + rg_set() + ol, od);
      } else {
        allreduce_land_bf16<S_RANK>(g, g.rv + SOFF + rg_set() + ol, 0, od);
      }
      asm volatile("" ::"v"(od[0]) : "memory");
    }
    __syncthreads();
  }
  // 2. the sums of this thread's 8 columns
  float o[8];
  if (act) {
    size_t const off = PR == 0 ? (size_t)t * H * 2 + (size_t)c0 * 2
                               : (size_t)(c0 / PR) * T * PR * 2 + (size_t)t * PR * 2 + (size_t)(c0 % PR) * 2;
    if constexpr (SLAND) { // (the sums, bf16, in this GPU's own slot)
      uint4 u = make_uint4(0u, 0u, 0u, 0u);
      if constexpr (HAND) { // (the LDS hand-off: the send task before this one on the CU left the sums in LDS)
        u = *reinterpret_cast<uint4 const *>(scratch + ARS_HAND_OFF + ((size_t)tl * 4 + v) * 16);
      } else {
        u = ld16_at<AUX_DEV>(g.rv + SOFF + rg_set() + off, (uint32_t)(g.rank * S_RANK));
      }
      if constexpr (!HAND) { // (0xFF until the sender writes the sum: polled, no count)
        while (!valid16w(u)) {
          tail_pause();
          u = ld16_at<AUX_DEV>(g.rv + SOFF + rg_set() + off, (uint32_t)(g.rank * S_RANK));
        }
      }
      uint32_t const wd[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int k = 0; k < 4; k++) {
        o[2 * k] = bf16_lo(wd[k]);
        o[2 * k + 1] = bf16_hi(wd[k]);
      }
    } else if constexpr (NPP > 1) { // (the push_all gemm's K parts too: GPUS x NPP slots)
      allreduce_land_bf16_nr<S_RANK, GPUS * NPP>(g.rv + SOFF + rg_set() + off, o);
    } else {
      allreduce_land_bf16<S_RANK>(g, g.rv + SOFF + rg_set() + off, 0, o);
    }
#pragma unroll
    for (int k = 0; k < 8; k++) {
      o[k] = round_bf16(o[k]); // the all-reduced bf16
    }
  }
  __syncthreads(); // cf_s, fn_s
  // 3. R'
  if (act) {
    float r[HC][8];
#pragma unroll
    for (int i = 0; i < HC; i++) {
      uint32_t const w[4] = {rv[i].x, rv[i].y, rv[i].z, rv[i].w};
#pragma unroll
      for (int k = 0; k < 4; k++) {
        r[i][2 * k] = bf16_lo(w[k]);
        r[i][2 * k + 1] = bf16_hi(w[k]);
      }
    }
    float const *cf = cf_s + tl * (HC + CB);
    float csum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; // (contract: the streams' bf16 values added in stream order)
#pragma unroll
    for (int j = 0; j < HC; j++) {
      float yv[8];
#pragma unroll
      for (int k = 0; k < 8; k++) {
        float a = __fmul_rn(cf[HC + 0 * HC + j], r[0][k]);
#pragma unroll
        for (int i = 1; i < HC; i++) {
          a = __fadd_rn(a, __fmul_rn(cf[HC + i * HC + j], r[i][k]));
        }
        yv[k] = round_bf16(__fadd_rn(__fmul_rn(cf[j], o[k]), a));
        rp_s[tl * KP + j * C + 8 * v + k] = yv[k];
      }
      st16_out(rout + ((size_t)t * HC + j) * H + c0,
               make_uint4(pack_bf16x2(yv[0], yv[1]), pack_bf16x2(yv[2], yv[3]), pack_bf16x2(yv[4], yv[5]),
                          pack_bf16x2(yv[6], yv[7])));
#pragma unroll
      for (int k = 0; k < 8; k++) {
        csum[k] += yv[k];
      }
      if constexpr (T >= STATIC_HCPM_RPOLL_MIN_T) {
        st16_out(rpoll + ((size_t)t * HC + j) * H + c0,
                 make_uint4(pack_bf16x2(yv[0], yv[1]), pack_bf16x2(yv[2], yv[3]), pack_bf16x2(yv[4], yv[5]),
                            pack_bf16x2(yv[6], yv[7])));
      }
    }
    if (contract != nullptr) { // (torch's bf16 mean: fp32 sum over the streams, / HC, rounded)
      float m8[8];
#pragma unroll
      for (int k = 0; k < 8; k++) {
        m8[k] = csum[k] / (float)HC;
      }
      st16_out(contract + (size_t)t * H + c0, make_uint4(pack_bf16x2(m8[0], m8[1]), pack_bf16x2(m8[2], m8[3]),
                                                         pack_bf16x2(m8[4], m8[5]), pack_bf16x2(m8[6], m8[7])));
    }
  }
  __syncthreads();
  STAGE_STAMP_AT(1)
  // 4. (T > 16) item (t, q) = floats [4 q, 4 q + 4) of row (s, t): q < NMIX / 4 four mixes, then the sum of squares and
  // 0s, then 0s; the lanes of a wave consecutive tokens (MI355X 11-07 glmmhc3: at T 32 / 64 faster than a wave per token,
  // 60.3 / 100.7 vs 66.3 / 114.5 us e2e)
  if constexpr (TL >= STATIC_HCPM_MFMA_MIN_T && TL % 16 == 0) {
    // (TL >= STATIC_HCPM_MFMA_MIN_T) the mixes as fp32 MFMAs (16x16x4): D[token][mix] = sum_k R'[token][k] fn[mix][k], tiles
    // of 16 tokens x 16 mixes (mixes >= NMIX: B = 0, so entries NMIX + 1 .. 31 come out 0), K = KW in steps of 4; entry NMIX =
    // the sum of squares (4 threads per token, then two shuffles). MI355X 11-07 glmmoe46: the per-thread 128-step FMA chains
    // over LDS took ~11 us of a T 64 task
    static_assert(TL % 16 == 0 && KW % 16 == 0 && NMIX < HC_MIX_STRIDE && HC_MIX_STRIDE == 32, "hc_post_mix: MFMA mixes");
    constexpr int NTILE = (TL / 16) * 2;
    int const wv = tid >> 6, ln = tid & 63, r = ln & 15, kq = ln >> 4;
    for (int ti = wv; ti < NTILE; ti += 4) {
      int const tt = ti >> 1, mt = ti & 1, mrow = 16 * mt + r;
      bool const bz = mrow >= NMIX;
      float const *ar = rp_s + (16 * tt + r) * KP;
      float const *br = fn_s + (bz ? 0 : mrow) * FS;
      v4f acc = v4f{0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
      for (int k0 = 0; k0 < KW; k0 += 4) {
        float const a = ar[k0 + kq];
        float const b = bz ? 0.f : br[k0 + kq];
        acc = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, acc, 0, 0, 0);
      }
      if (mrow != NMIX) { // (D: tokens 16 tt + 4 kq + i, mix column mrow)
#pragma unroll
        for (int i = 0; i < 4; i++) {
          __hip_atomic_store(reinterpret_cast<uint32_t *>(mix + ((size_t)s * T + t0 + 16 * tt + 4 * kq + i) * HC_MIX_STRIDE + mrow),
                             __float_as_uint(acc[i]), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
    }
    {
      int const tq = tid >> 2, q = tid & 3;
      float a = 0.f;
      if (tq < TL) {
        float const *rr = rp_s + tq * KP + q * (KW / 4);
#pragma unroll 8
        for (int k = 0; k < KW / 4; k++) {
          a = fmaf(rr[k], rr[k], a);
        }
      }
      a += __shfl_xor(a, 1);
      a += __shfl_xor(a, 2);
      if (tq < TL && q == 0) {
        __hip_atomic_store(reinterpret_cast<uint32_t *>(mix + ((size_t)s * T + t0 + tq) * HC_MIX_STRIDE + NMIX), __float_as_uint(a),
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  } else if constexpr (TL > 16) {
    constexpr int QN = HC_MIX_STRIDE / 4;
    static_assert(NMIX % 4 == 0, "hc_post_mix: whole float4s of mixes");
    for (int it = tid; it < TL * QN; it += 256) {
      int const tt = it % TL, q = it / TL;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      float const *rr = rp_s + tt * KP;
      if (4 * q < NMIX) {
        float const *f0 = fn_s + (4 * q) * FS;
#pragma unroll 8
        for (int k = 0; k < KW; k++) {
          float const xv = rr[k];
#pragma unroll
          for (int e = 0; e < 4; e++) {
            acc[e] = fmaf(xv, f0[e * FS + k], acc[e]);
          }
        }
      } else if (4 * q == NMIX) {
#pragma unroll 8
        for (int k = 0; k < KW; k++) {
          acc[0] = fmaf(rr[k], rr[k], acc[0]);
        }
      }
      st16_out(mix + ((size_t)s * T + t0 + tt) * HC_MIX_STRIDE + 4 * q,
               make_uint4(__float_as_uint(acc[0]), __float_as_uint(acc[1]), __float_as_uint(acc[2]),
                          __float_as_uint(acc[3])));
    }
  } else {
  // 4. (T <= 16) the mixes: per token (a wave each: tokens w, w + 4, ..) its NMIX mixes mix[r] = sum_k R'[k] fn[r][k] and the sum of
  // squares of R' (k < HC 32), the 64 lanes splitting k (lane: k = lane + 64 i), then each sum over the wave; row (s, t)
  // of the output = the NMIX mixes, the sum of squares, 0s up to HC_MIX_STRIDE (lanes 0 .. QN - 1 one float4 each).
  // (MI355X 11-07: a thread per float4 of mixes ran a 128-step dependent FMA chain over LDS; at T 1 only 8 threads, ~7 us)
  constexpr int QN = HC_MIX_STRIDE / 4, NI = KW / 64;
  static_assert(NMIX % 4 == 0 && KW % 64 == 0 && NMIX + 1 <= HC_MIX_STRIDE, "hc_post_mix: whole float4s of mixes");
  int const wv = tid >> 6, ln = tid & 63;
  for (int tt = wv; tt < TL; tt += 4) {
    float const *rr = rp_s + tt * KP;
    float xv[NI];
#pragma unroll
    for (int i = 0; i < NI; i++) {
      xv[i] = rr[ln + 64 * i];
    }
    float res[HC_MIX_STRIDE];
#pragma unroll
    for (int r = 0; r < HC_MIX_STRIDE; r++) {
      res[r] = 0.f;
    }
#pragma unroll
    for (int r = 0; r <= NMIX; r++) {
      float a = 0.f;
#pragma unroll
      for (int i = 0; i < NI; i++) {
        a = fmaf(xv[i], r < NMIX ? fn_s[r * FS + ln + 64 * i] : xv[i], a);
      }
      res[r] = wave_sum64(a);
    }
    if (ln < QN) {
      float o[4];
#pragma unroll
      for (int e = 0; e < 4; e++) {
        o[e] = 0.f;
#pragma unroll
        for (int q = 0; q < QN; q++) {
          o[e] = ln == q ? res[4 * q + e] : o[e];
        }
      }
      st16_out(mix + ((size_t)s * T + t0 + tt) * HC_MIX_STRIDE + 4 * ln,
               make_uint4(__float_as_uint(o[0]), __float_as_uint(o[1]), __float_as_uint(o[2]), __float_as_uint(o[3])));
    }
  }
  }
  task_post(done_cnt, 1); // release: R' of these columns, every token, and the mixes
}

template <class BUF_SLOTS>
__device__ __forceinline__ __hip_bfloat16 *hcpm_contract_ptr(G const &g) {
  if constexpr (sizeof(BUF_SLOTS::v) / sizeof(int) > 7) { // (slot 6 present: v ends with -1)
    if constexpr (slot_at<BUF_SLOTS, 6>() >= 0) {
      return buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 6>());
    }
  }
  return nullptr;
}
template <class BUF_SLOTS>
__device__ __forceinline__ __hip_bfloat16 *hcpm_rpoll_ptr(G const &g) {
  if constexpr (T >= STATIC_HCPM_RPOLL_MIN_T) {
    return buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 5>());
  } else {
    return nullptr;
  }
}
// hc_post_mix task x. SEND: the node whose bf16 partial sums it adds: an allreduce_send node (params {H, 1[, ...]}) or
// (PARAMS[3] = 1) a push_all gemm_tile (params {COMBINE_PUSH_ALL, K, N = H, ..}, its grid x = H / its rows per task);
// PARAMS {HC, H, NMIX[, push_all]} (hc_pre reads the first three); BUF_SLOTS {the residual streams R, the output R'
// ([T][HC H] bf16 graph tensors), h_post, h_res (fp32 graph tensors), fn [NMIX][HC H] fp32}; its buffer: the mixes; its
// counter: the count of its tasks
template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class SEND, class... IN>
__device__ __forceinline__ void run_hc_post_mix(Maps const &, G const &g, KernelLocals &L, StaticTask const &tk) {
  constexpr int HC = PARAMS::v[0], H = PARAMS::v[1], NMIX = PARAMS::v[2];
  constexpr bool PUSH = sizeof(PARAMS::v) / sizeof(int) > 4 && PARAMS::v[3] == 1;
  static_assert(NMIX == HC * (HC + 2), "hc_post_mix: params {HC, H, NMIX = HC (HC + 2)}");
  static_assert(PUSH ? ((SEND::v[0] == COMBINE_PUSH_ALL || SEND::v[0] == COMBINE_FINISH_ALL) && SEND::v[2] == H)
                     : (SEND::v[0] == H && SEND::v[1] >= 1),
                "hc_post_mix: rows of H from an allreduce_send node, or a push_all / finish_all gemm_tile's partials");
  // (the sender's parts per GPU: push_all one slot per K part, finish_all one (its last K part pushed the sums))
  constexpr int NPP_ = PUSH ? (SEND::v[0] == COMBINE_FINISH_ALL ? 1 : SEND::y) : 1;
  static_assert(SEND::buf >= 0 && exchange_bytes[SEND::buf] ==
                                       (size_t)GPUS * NPP_ * T * H * 2,
                "hc_post_mix: the partial sums, [GPUS] x T x H bf16 in the exchange region");
  static_assert(SELF::x * 32 == H && SELF::buf >= 0 && SELF::counter >= 0,
                "hc_post_mix: a task per 32 columns, its mixes and counter");
  hc_post_mix_task<exchange_offset[SEND::buf], H, HC, NMIX, PUSH ? H / SEND::x : 0, SEND::x * SEND::y * SEND::z, SEND::counter,
                   SELF::y, NPP_, !PUSH && ARS_HAND_ON((param_or0<SEND, 2>()), SEND::x, H)>(
      g,
      buf_at<float const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 4>()),
      buf_at<float>(g, SELF::buf),
      cnt_of(g) + SELF::counter,
      tk.x,
      tk.y,
      hcpm_rpoll_ptr<BUF_SLOTS>(g),
      hcpm_contract_ptr<BUF_SLOTS>(g),
      L.sm STAGE_STAMP_ARG(L.stage_stamps));
}

template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class... IN>
constexpr int smem_hc_post_mix() {
  constexpr int HC = PARAMS::v[0], NMIX = PARAMS::v[2], KW = HC * 32;
  return (NMIX * (KW + 1) + T * (KW + 1) + T * (HC + HC * HC)) * 4;
}

} // namespace static_mk
