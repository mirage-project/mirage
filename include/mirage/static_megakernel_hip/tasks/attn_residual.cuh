// tasks/attn_residual.cuh (HIP, gfx950) -- K3's MLP-side attention residual
// after a KDA (or MLA) layer's o_proj, with the TP all-reduce folded in (math:
// mpk-kernels tests/kda_reference.py attn_residual = SGLang ROCm's
// attn_res_hip _agg_kernel + RMSNorm). Task (x, y) of grid (T, NC, 1) = token
// x, columns [y CW, (y + 1) CW) (CW = H / NC = 1024; the NC tasks of a token
// run at the same time and add their partial sums through a scratch buffer):
//   a   = the GPUs' partial sums of the row (a push_all gemm_tile's output in
//         the exchange region, [GPUS][T][H] bf16, 0xFF-prefilled), added in
//         fp32 in GPU order, rounded to bf16 (SGLang ROCm: the all-reduce's
//         bf16 output)
//   u   = bf16(prefix + a)                 -> u_out (the next layer's prefix)
//   c   = q * gain (fp32, not rounded: SGLang ROCm's cw2, attn_residual.py
//         get_cw; B300's CUDA path rounds it to bf16)
//   for src in the n bank rows (info[0]; strides info[1], info[2]) and u:
//         s = (c . src) / sqrt(mean(src^2) + eps)
//   w   = softmax(s);  h = sum w src (fp32)
//   out = bf16(h rsqrt(mean(h^2) + eps) ow)
// Before the partials land (the bank is known at the start): each chunk's
// dots of the bank rows with c and the bank rows' pairwise dots (their Gram
// matrix, the sums of squares on its diagonal) -> scratch, a count on the
// token's first counter (this round overlaps the layer's earlier steps). After
// u: each chunk's sum of squares of u, its dot with c and with every bank row
// -> scratch, a count on the token's second counter. Every chunk adds the NC
// partials (chunk order) -> the scores, the weights and the sum of h^2 =
// sum_s sum_s' w_s w_s' (src_s . src_s') (one round after the partials land).
// 256 threads: thread i = columns 4 i .. 4 i + 3 of the chunk; every load of a
// phase is issued before its first use (one memory round trip per phase).
#pragma once
#include "../core.cuh"

// The chunks' values (scratch, 0xFF before each launch) are their own flags: no counts on counters 1 .. 2 T, the
// readers re-load a value until it is not all-ones (no release wait, no counter round trip; MI355X 10-13 kda16: KDA
// 53.5 / 53.7 vs 56.8 / 57.2 us). The weights and h's sum of squares in every wave at once, no barrier: lane s source
// s's score (the softmax by two wave reductions), lane s1 its row of the pairs s1 <= s2 (the weights by readlane), one
// more wave sum (MI355X 10-15 kda155: ~3.5 us from the exchange to the weights with every thread doing all 17 scores
// and the Gram terms one after the other).
// STATIC_AR_POST_NOWAIT 1: a kernel with no reader of counter 0 (KDA / MLA alone): the count without waiting for the
// output stores; the kernel's end waits for them
#ifndef STATIC_AR_POLL2
#define STATIC_AR_POLL2 0
#endif
#ifndef STATIC_AR_POST_NOWAIT
#define STATIC_AR_POST_NOWAIT 0
#endif

namespace static_mk {

constexpr int ATTN_RES_NBMAX = 16;               // bank rows at most
constexpr int AR_NS = ATTN_RES_NBMAX + 1;        // sources: the bank rows, then u
constexpr int AR_PA = ATTN_RES_NBMAX + ATTN_RES_NBMAX * (ATTN_RES_NBMAX + 1) / 2; // early round: dot_b, Gram (b <= b')
constexpr int AR_PB = 2 + ATTN_RES_NBMAX;        // after u: u . u, c . u, u . bank row b
constexpr int AR_PW = AR_PA + AR_PB;             // scratch floats per chunk
__device__ __forceinline__ constexpr int ar_gidx(int b, int b2) { // b <= b2
  return ATTN_RES_NBMAX + b * ATTN_RES_NBMAX - b * (b - 1) / 2 + (b2 - b);
}
constexpr int AR_CW = 1024;                      // columns per chunk (H / NC: 1024 at NC 7, 1792 at NC 4)
// a thread's CPT columns as bf16 (4: 8 B, 8: 16 B) and their loads / stores
template <int CPT>
struct ArVec;
template <>
struct ArVec<4> {
  typedef uint2 type;
  __device__ static float at(type v, int e) { return (e & 1) ? bf16_hi((&v.x)[e / 2]) : bf16_lo((&v.x)[e / 2]); }
  __device__ static type zero() { return make_uint2(0u, 0u); }
};
template <>
struct ArVec<8> {
  typedef uint4 type;
  __device__ static float at(type v, int e) { return (e & 1) ? bf16_hi((&v.x)[e / 2]) : bf16_lo((&v.x)[e / 2]); }
  __device__ static type zero() { return make_uint4(0u, 0u, 0u, 0u); }
};

__device__ __forceinline__ void ar_st4_dev(void *p, float v) {
  asm volatile("global_store_dword %0, %1, off sc1" ::"v"(p), "v"(__float_as_uint(v)) : "memory");
}
__device__ __forceinline__ void ar_st8_dev(void *p, uint2 v) {
  v2u const w = {v.x, v.y};
  asm volatile("global_store_dwordx2 %0, %1, off sc1" ::"v"(p), "v"(w) : "memory");
}
template <int CPT>
__device__ __forceinline__ void ar_stv_dev(__hip_bfloat16 *p, float const *v) { // CPT bf16 at device scope
  if constexpr (CPT == 4) {
    ar_st8_dev(p, make_uint2(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3])));
  } else {
    st16_dev(p, make_uint4(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]), pack_bf16x2(v[4], v[5]), pack_bf16x2(v[6], v[7])));
  }
}
__device__ __forceinline__ float ar_ld4_dev(void const *base, uint32_t off) {
  return __uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(buf_rsrc(base), off, 0, AUX_DEV));
}

// 8 B by a buffer load with cache policy AUX (MOE: o and S)
template <int AUX>
__device__ __forceinline__ uint2 ar_ld8_at(void const *base, uint32_t off) {
  typedef unsigned ar_v2u __attribute__((ext_vector_type(2)));
  ar_v2u const w = __builtin_amdgcn_raw_buffer_load_b64(buf_rsrc(base), off, 0, AUX);
  return make_uint2(w.x, w.y);
}
// MOE (tasks/moe_attn_residual.cuh, aggregation 1 at the end of a MoE layer): a = bf16(o + S), o the latent_up rows of the
// GPU that owns the columns (sum_send's layout in the exchange region at AOFF), S sum_gpus's buffer (ssum, device scope),
// both polled (0xFF before each launch), as residual_add; u = y; info[3] != 0: u also into bank row info[0] (SGLang's
// fused bank write of a block-write layer)
template <size_t AOFF, int H, int NC, int PR = 0, bool MOE = false> // PR > 0: the partials tile-major ([GPUS][H / PR][T][PR], gemm_tile's push_all)
__device__ __forceinline__ void attn_residual_task(G const &g,
                                                   KernelLocals &L,
                                                   __hip_bfloat16 const *bank,
                                                   __hip_bfloat16 const *prefix,
                                                   __hip_bfloat16 const *q,
                                                   __hip_bfloat16 const *gain,
                                                   __hip_bfloat16 const *ow,
                                                   __hip_bfloat16 *u_out,
                                                   int const *info,
                                                   float eps,
                                                   __hip_bfloat16 *out,
                                                   float *part,     // [T][NC][AR_PW]
                                                   uint32_t *cnt,   // [1 + 2 T]: tasks done; per token the early round, the u round
                                                   int t,
                                                   int ch STAGE_STAMP_PARAM,
                                                   __hip_bfloat16 const *ssum = nullptr,
                                                   uint32_t const *pre_cnt = nullptr,
                                                   uint32_t pre_need = 0) {
  constexpr int CW = H / NC, CPT = CW <= 1024 ? 4 : 8, NTH = CW / CPT; // columns per chunk, per thread; threads with columns
  static_assert(H == NC * CW && CW % (8 * CPT) == 0 && NTH <= 256 && GPUS * (CW / 8) % 256 == 0,
                "attn_residual: chunks of 256 threads x 4 or up to 256 threads x 8 columns");
  using V = ArVec<CPT>;
  constexpr int NVC = CW / 8;                    // 16-B vectors per chunk row
  constexpr int LPT = GPUS * NVC / 256;          // landing loads per thread
  static_assert(GPUS * NVC % 256 == 0 && LPT >= 1, "attn_residual: whole landing rounds");
  constexpr size_t RANK = (size_t)T * H * 2;
  int const tid = task_tid(), lane = tid & 63, wave = tid >> 6;
  int const nb = info[0];
  size_t const bs_tok = (size_t)info[1], bs_row = (size_t)info[2];
  bool const live = tid < NTH;                   // a thread with columns (the others: zeros in the sums)
  int const c0 = ch * CW + CPT * (live ? tid : 0); // this thread's first column
  // LDS: the landed partials [GPUS][CW] bf16, the waves' sums, the chunks' sums
  uint4 *const lnd = reinterpret_cast<uint4 *>(L.sm);
  float *const red = reinterpret_cast<float *>(L.sm + GPUS * CW * 2); // [4 waves][AR_PW]
  float *const tot = red + 4 * AR_PW;                                     // [AR_PW] the chunks' sums
  __syncthreads(); // the LDS may still be read by the previous task
  if (pre_cnt != nullptr) { // (prefix / bank rows written by a node of this kernel, e.g. moe_agg1's second count)
    if (tid == 0) {
      cnt_wait_in(pre_cnt, pre_need);
    }
    __syncthreads();
  }
  // 0. what needs no input: this thread's prefix, c, ow and bank columns
  float pre[CPT], cc[CPT], owv[CPT];
  typename V::type bk[ATTN_RES_NBMAX];
  {
    typedef typename V::type VT;
    VT const p2 = live ? *reinterpret_cast<VT const *>(prefix + (size_t)t * H + c0) : V::zero(),
             q2 = live ? *reinterpret_cast<VT const *>(q + c0) : V::zero(),
             g2 = live ? *reinterpret_cast<VT const *>(gain + c0) : V::zero(),
             o2 = live ? *reinterpret_cast<VT const *>(ow + c0) : V::zero();
#pragma unroll
    for (int b = 0; b < ATTN_RES_NBMAX; b++) {
      bk[b] = (live && b < nb) ? *reinterpret_cast<VT const *>(bank + t * bs_tok + b * bs_row + c0) : V::zero();
    }
#pragma unroll
    for (int e = 0; e < CPT; e++) {
      pre[e] = V::at(p2, e);
      cc[e] = V::at(q2, e) * V::at(g2, e);
      owv[e] = V::at(o2, e);
    }
  }
  // 0b. the early round: this chunk's c . bank row b and the bank rows'
  // pairwise dots (each value summed over the wave, lane 0 into red)
  float *const pa = part + ((size_t)t * NC + ch) * AR_PW;
  for (int i = tid; i < 4 * AR_PW; i += 256) {
    red[i] = 0.f;
  }
  __syncthreads();
  {
#pragma unroll
    for (int b = 0; b < ATTN_RES_NBMAX; b++) {
      if (b < nb) {
        float d = 0.f;
#pragma unroll
        for (int e = 0; e < CPT; e++) {
          d = fmaf(cc[e], V::at(bk[b], e), d);
        }
        d = wave_sum64(d);
        if (lane == 0) {
          red[wave * AR_PW + b] = d;
        }
#pragma unroll
        for (int b2 = b; b2 < ATTN_RES_NBMAX; b2++) {
          if (b2 < nb) {
            float g2 = 0.f;
#pragma unroll
            for (int e = 0; e < CPT; e++) {
              g2 = fmaf(V::at(bk[b], e), V::at(bk[b2], e), g2);
            }
            g2 = wave_sum64(g2);
            if (lane == 0) {
              red[wave * AR_PW + ar_gidx(b, b2)] = g2;
            }
          }
        }
      }
    }
  }
  __syncthreads();
  if (tid < AR_PA) {
    ar_st4_dev(pa + tid, red[tid] + red[AR_PW + tid] + red[2 * AR_PW + tid] + red[3 * AR_PW + tid]);
  }
  // 1. the GPUs' partials of this chunk (all of a thread's loads in flight per
  // round) into LDS
  unsigned char const *const abase = g.rv + AOFF + rg_set();
  auto aoff = [&](int slot, int vec) -> uint32_t { // GPU slot's partial, columns ch CW + 8 vec .. + 7 of token t
    int const n = ch * CW + 8 * vec;
    return (uint32_t)(slot * RANK + (PR == 0 ? ((size_t)t * H + n) * 2 : (((size_t)(n / PR) * T + t) * PR + n % PR) * 2));
  };
  static_assert(PR == 0 || (PR % 8 == 0 && H % PR == 0), "attn_residual: tile-major partials in whole 16-B vectors");
  float u[CPT];
  if constexpr (MOE) {
    static_assert(CPT == 4 && (H / GPUS) % 4 == 0, "attn_residual MOE: 4 columns per thread");
    // this thread's 4 columns: o from the owner GPU r, S; y = bf16(bf16(o + S) + prefix) (residual_add's roundings)
    int const r = c0 / (H / GPUS), cc = c0 - r * (H / GPUS);
    uint32_t const ooff = (uint32_t)(AOFF + (size_t)r * T * (H / GPUS) * 2 + (size_t)t * (H / GPUS) * 2 + cc * 2 + rg_set());
    uint32_t const soff = (uint32_t)(((size_t)t * H + c0) * 2);
    uint2 ou = make_uint2(0u, 0u), su = make_uint2(0u, 0u);
    bool ok_o = !live || (SOLO && r != g.rank), ok_s = !live; // SOLO: another GPU's o: zeros
    for (;;) {
      if (!ok_o) {
        ou = ar_ld8_at<AUX_SYS>(g.rv, ooff);
      }
      if (!ok_s) {
        su = ar_ld8_at<AUX_DEV>(ssum, soff);
      }
      ok_o = ok_o || (ou.x != 0xFFFFFFFFu && ou.y != 0xFFFFFFFFu);
      ok_s = ok_s || (su.x != 0xFFFFFFFFu && su.y != 0xFFFFFFFFu);
      if (__all(ok_o && ok_s)) {
        break;
      }
      tail_pause();
    }
    uint32_t const ow2[2] = {ou.x, ou.y}, sw2[2] = {su.x, su.y};
#pragma unroll
    for (int e = 0; e < CPT; e++) {
      float const ov = (e & 1) ? bf16_hi(ow2[e / 2]) : bf16_lo(ow2[e / 2]),
                  sv = (e & 1) ? bf16_hi(sw2[e / 2]) : bf16_lo(sw2[e / 2]);
      u[e] = live ? round_bf16(pre[e] + round_bf16(ov + sv)) : 0.f;
    }
    if (live) {
      ar_stv_dev<CPT>(u_out + (size_t)t * H + c0, u);
    }
    if (live && info[3] != 0) { // the block-write layer's bank row nb
      ar_stv_dev<CPT>(const_cast<__hip_bfloat16 *>(bank) + t * bs_tok + nb * bs_row + c0, u);
    }
#if STATIC_STAMP_SET == 5 || STATIC_STAMP_SET == 0
    if (tid == 0) {
      STAGE_STAMP_AT(0)
    }
#endif
  } else {
  {
    uint4 v[LPT];
    for (;;) {
      bool all = true;
#pragma unroll
      for (int j = 0; j < LPT; j++) {
        int const k = tid + 256 * j, r = k / NVC, vec = k % NVC;
        if (SOLO && r != 0) {
          v[j] = make_uint4(0u, 0u, 0u, 0u); // (SOLO: the other GPUs' partials are zeros)
        } else {
          int const slot = SOLO ? g.rank : r;
          v[j] = ld16_at<AUX_SYS>(abase, aoff(slot, vec));
          all = all && valid16w(v[j]);
        }
      }
      if (all) {
        break;
      }
      tail_pause();
    }
#pragma unroll
    for (int j = 0; j < LPT; j++) {
      lnd[tid + 256 * j] = v[j];
    }
  }
  __syncthreads();
#if STATIC_STAMP_SET == 5
  if (tid == 0) { // (stamp set 5: stamp 0 = the partials landed (every thread's), 1 = the after-u values stored (issued))
    STAGE_STAMP_AT(0)
  }
#endif
  {
    float s[CPT];
#pragma unroll
    for (int e = 0; e < CPT; e++) {
      s[e] = 0.f;
    }
    __hip_bfloat16 const *const lb = reinterpret_cast<__hip_bfloat16 const *>(lnd);
#pragma unroll
    for (int r = 0; r < GPUS; r++) { // GPU order
      typename V::type const w2 = live ? *reinterpret_cast<typename V::type const *>(lb + r * CW + CPT * tid) : V::zero();
#pragma unroll
      for (int e = 0; e < CPT; e++) {
        s[e] += V::at(w2, e);
      }
    }
#pragma unroll
    for (int e = 0; e < CPT; e++) {
      u[e] = live ? round_bf16(pre[e] + round_bf16(s[e])) : 0.f;
    }
    // (device scope: a consumer in the same kernel reads it, e.g. a fused MoE's
    // residual add)
    if (live) {
      ar_stv_dev<CPT>(u_out + (size_t)t * H + c0, u);
    }
  }
#if STATIC_STAMP_SET == 0
  if (tid == 0) { // (timing build: stage stamp 0 = the GPUs' partials landed and added)
    STAGE_STAMP_AT(0)
  }
#endif
  } // (!MOE)
  // 2. after u: this chunk's u . u, c . u and u . bank row b
  {
    float v0 = 0.f, v1 = 0.f;
#pragma unroll
    for (int e = 0; e < CPT; e++) {
      v0 = fmaf(u[e], u[e], v0);
      v1 = fmaf(cc[e], u[e], v1);
    }
    v0 = wave_sum64(v0);
    v1 = wave_sum64(v1);
    if (lane == 0) {
      red[wave * AR_PW + AR_PA] = v0;
      red[wave * AR_PW + AR_PA + 1] = v1;
    }
#pragma unroll
    for (int b = 0; b < ATTN_RES_NBMAX; b++) {
      if (b < nb) {
        float d = 0.f;
#pragma unroll
        for (int e = 0; e < CPT; e++) {
          d = fmaf(u[e], V::at(bk[b], e), d);
        }
        d = wave_sum64(d);
        if (lane == 0) {
          red[wave * AR_PW + AR_PA + 2 + b] = d;
        }
      }
    }
  }
  __syncthreads();
  if (tid < AR_PB) {
    int const i = AR_PA + tid;
    ar_st4_dev(pa + i, red[i] + red[AR_PW + i] + red[2 * AR_PW + i] + red[3 * AR_PW + i]);
  }
#if STATIC_STAMP_SET == 5
  if (tid == 0) {
    STAGE_STAMP_AT(1)
  }
#endif
#if STATIC_STAMP_SET == 4
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); // (stamp set 4: stamp 0 = this task's stores done, 1 = the exchange done)
  if (tid == 0) {
    STAGE_STAMP_AT(0)
  }
#endif
  // the NC chunks' values added in chunk order (thread i: value i; its NC loads
  // in flight together, re-loaded until none is all-ones)
  for (int i = tid; i < AR_PW; i += 256) {
    float v[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) {
      v[c] = ar_ld4_dev(part, (uint32_t)((((size_t)t * NC + c) * AR_PW + i) * 4));
    }
    for (;;) {
      bool all = true;
#pragma unroll
      for (int c = 0; c < NC; c++) {
        all = all && __float_as_uint(v[c]) != 0xFFFFFFFFu;
      }
      if (all) {
        break;
      }
      tail_pause();
#pragma unroll
      for (int c = 0; c < NC; c++) {
        if (__float_as_uint(v[c]) == 0xFFFFFFFFu) {
          v[c] = ar_ld4_dev(part, (uint32_t)((((size_t)t * NC + c) * AR_PW + i) * 4));
        }
      }
    }
    float sum = 0.f;
#pragma unroll
    for (int c = 0; c < NC; c++) {
      sum += v[c];
    }
    tot[i] = sum;
  }
  __syncthreads();
#if STATIC_STAMP_SET == 4
  if (tid == 0) {
    STAGE_STAMP_AT(1)
  }
#endif
  float wts[AR_NS];
  float rstd_w;
  {
    int const s1 = lane;
    bool const lv = s1 < AR_NS && (s1 == ATTN_RES_NBMAX || s1 < nb);
    float sc = -3.0e38f;
    if (lv) {
      bool const is_u = s1 == ATTN_RES_NBMAX;
      float const sq = is_u ? tot[AR_PA] : tot[ar_gidx(s1, s1)], dt = is_u ? tot[AR_PA + 1] : tot[s1];
      sc = dt * rsqrtf(sq / (float)H + eps);
    }
    float const mxw = wave_max64(sc);
    float const ex = lv ? __expf(sc - mxw) : 0.f;
    float const dn = wave_sum64(ex);
    float const w = ex / dn;
    float hp = 0.f;
#pragma unroll
    for (int s2 = 0; s2 < AR_NS; s2++) {
      float const w2 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(w), s2));
      wts[s2] = w2;
      if (s1 < AR_NS && s2 >= s1) {
        float const gv = s2 == ATTN_RES_NBMAX ? (s1 == ATTN_RES_NBMAX ? tot[AR_PA] : tot[AR_PA + 2 + s1]) : tot[ar_gidx(s1, s2)];
        hp = fmaf((s1 == s2 ? 1.f : 2.f) * w * w2, gv, hp);
      }
    }
    float const hs = wave_sum64(hp);
    rstd_w = rsqrtf(fmaxf(hs, 0.f) / (float)H + eps);
  }
#if STATIC_STAMP_SET == 0
  if (tid == 0) { // (stage stamp 1: the weights and h's RMS known)
    STAGE_STAMP_AT(1)
  }
#endif
  // 3. h = the weighted sources (fp32) of this thread's columns, out
  float h[CPT];
#pragma unroll
  for (int e = 0; e < CPT; e++) {
    h[e] = wts[ATTN_RES_NBMAX] * u[e];
  }
#pragma unroll
  for (int b = 0; b < ATTN_RES_NBMAX; b++) {
    if (b < nb) {
#pragma unroll
      for (int e = 0; e < CPT; e++) {
        h[e] = fmaf(wts[b], V::at(bk[b], e), h[e]);
      }
    }
  }
  float const rstd = rstd_w;
  if (live) {
    float o[CPT];
#pragma unroll
    for (int e = 0; e < CPT; e++) {
      o[e] = h[e] * rstd * owv[e];
    }
    ar_stv_dev<CPT>(out + (size_t)t * H + c0, o);
  }
#if STATIC_AR_POST_NOWAIT
  __syncthreads();
  if (tid == 0) {
    cnt_post(cnt, 1);
  }
#else
  task_post(cnt, 1); // (counter 0: this task's outputs out, its consumers in the kernel wait for T x NC)
#endif
  __syncthreads();
}

// TOK (attn_residual_layer chunks = 1, a push_all o_proj): task (t, 0) = token t, every column on one CU, as moe_agg1: no exchange
// between chunk tasks (the 7 chunks of a token swapped their dots through a scratch: MI355X dec242, T 16, ~6.5 us from the
// partials landed to the weights). Thread i: columns 4 i + 1024 j .. + 3, j < H / 1024; the bank rows in LDS (at most
// ATTN_TOK_NBMAX); the GPUs' partials of the thread's columns in one burst per pass (8-B loads), added in GPU order; u =
// bf16(prefix + bf16(sum)); scores (c . src) / sqrt(mean(src^2) + eps) for the bank rows and u, softmax; h = the weighted sum
// (fp32), its sum of squares from h (as moe_agg1 / SGLang ROCm's attn_res_hip; the chunked task expands it over the sources);
// out = bf16(h rsqrt(mean(h^2) + eps) ow). One count per task.
constexpr int ATTN_TOK_NBMAX = 8;
__device__ __forceinline__ void ar_tok_lds_bar() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}
__device__ __forceinline__ void ar_tok_st8_dev(void *p, uint2 v) {
  typedef unsigned v2ut __attribute__((ext_vector_type(2)));
  v2ut const w = {v.x, v.y};
  asm volatile("global_store_dwordx2 %0, %1, off sc1\n s_nop 1" ::"v"(p), "v"(w) : "memory");
}
template <size_t AOFF, int H, int PR>
__device__ __forceinline__ void attn_residual_tok_task(G const &g,
                                                       KernelLocals &L,
                                                       __hip_bfloat16 const *bank,
                                                       __hip_bfloat16 const *prefix,
                                                       __hip_bfloat16 const *q,
                                                       __hip_bfloat16 const *gain,
                                                       __hip_bfloat16 const *ow,
                                                       __hip_bfloat16 *u_out,
                                                       int const *info,
                                                       float eps,
                                                       __hip_bfloat16 *out,
                                                       uint32_t *cnt,
                                                       int t STAGE_STAMP_PARAM,
                                                       uint32_t const *pre_cnt = nullptr,
                                                       uint32_t pre_need = 0) {
  constexpr int NG = H / 1024, NV = 2 * ATTN_TOK_NBMAX + 2;
  static_assert(H % 1024 == 0 && (PR == 0 || PR % 4 == 0), "attn_residual TOK: 4 columns per thread per 1024");
  constexpr size_t RANK = (size_t)T * H * 2;
  int const tid = task_tid_all(), lane = tid & 63, wave = tid >> 6;
  int const nb = info[0];
  if (nb > ATTN_TOK_NBMAX) {
    __builtin_trap(); // (more bank rows than the LDS holds: build with chunks)
  }
  size_t const bs_tok = (size_t)info[1], bs_row = (size_t)info[2];
  float *const red = reinterpret_cast<float *>(L.sm);             // [4 waves][NV], then [4] h's sums of squares
  uint2 *const bkl = reinterpret_cast<uint2 *>(L.sm + 1024);       // the bank rows [row][j][thread], 8 B each
  __syncthreads(); // the LDS may still be read by the previous task
  if (pre_cnt != nullptr) {
    if (tid == 0) {
      cnt_wait_in(pre_cnt, pre_need);
    }
    __syncthreads();
  }
  auto col = [&](int j) { return 4 * tid + 1024 * j; };
  auto at = [](uint2 v, int e) { return (e & 1) ? bf16_hi((&v.x)[e / 2]) : bf16_lo((&v.x)[e / 2]); };
  // 0. what needs no input: prefix, c, ow; the bank rows (into LDS) and their c . b, b . b
  uint2 pre2[NG], ow2[NG];
  float cc[NG][4];
#pragma unroll
  for (int j = 0; j < NG; j++) {
    pre2[j] = *reinterpret_cast<uint2 const *>(prefix + (size_t)t * H + col(j));
    ow2[j] = *reinterpret_cast<uint2 const *>(ow + col(j));
    uint2 const q2 = *reinterpret_cast<uint2 const *>(q + col(j)), g2 = *reinterpret_cast<uint2 const *>(gain + col(j));
#pragma unroll
    for (int e = 0; e < 4; e++) {
      cc[j][e] = at(q2, e) * at(g2, e);
    }
  }
#pragma unroll 1
  for (int b = 0; b < nb; b++) {
    uint2 bk[NG];
#pragma unroll
    for (int j = 0; j < NG; j++) {
      bk[j] = *reinterpret_cast<uint2 const *>(bank + t * bs_tok + b * bs_row + col(j));
    }
    float d = 0.f, sq = 0.f;
#pragma unroll
    for (int j = 0; j < NG; j++) {
#pragma unroll
      for (int e = 0; e < 4; e++) {
        d = fmaf(cc[j][e], at(bk[j], e), d);
        sq = fmaf(at(bk[j], e), at(bk[j], e), sq);
      }
      bkl[(b * NG + j) * 256 + tid] = bk[j];
    }
    d = wave_sum64(d);
    sq = wave_sum64(sq);
    if (lane == 0) {
      red[wave * NV + 2 * b] = d;
      red[wave * NV + 2 * b + 1] = sq;
    }
  }
  // 1. the GPUs' partials of this thread's columns: every load of a pass in one burst, then checked
  unsigned char const *const abase = g.rv + AOFF + rg_set();
  auto aoff = [&](int slot, int n) -> uint32_t {
    return (uint32_t)(slot * RANK + (PR == 0 ? ((size_t)t * H + n) * 2 : (((size_t)(n / PR) * T + t) * PR + n % PR) * 2));
  };
  // (in groups of NGH column groups: all of a group's loads in one burst; NG x GPUS 8-B values at once spilled the kernel)
  constexpr int NGH = (NG + 1) / 2;
  float u[NG][4];
#pragma unroll
  for (int h0 = 0; h0 < NG; h0 += NGH) {
    uint2 pv[NGH][GPUS];
    for (;;) {
#pragma unroll
      for (int jj = 0; jj < NGH; jj++)
#pragma unroll
        for (int r = 0; r < GPUS; r++) {
          pv[jj][r] = (h0 + jj >= NG || (SOLO && r != 0)) ? make_uint2(0u, 0u)
                                                          : ar_ld8_at<AUX_SYS>(abase, aoff(SOLO ? g.rank : r, col(h0 + jj)));
        }
      bool all = true;
#pragma unroll
      for (int jj = 0; jj < NGH; jj++)
#pragma unroll
        for (int r = 0; r < GPUS; r++) {
          all = all && pv[jj][r].x != 0xFFFFFFFFu && pv[jj][r].y != 0xFFFFFFFFu;
        }
      if (all) {
        break;
      }
      tail_pause();
    }
#pragma unroll
    for (int jj = 0; jj < NGH; jj++) {
      int const j = h0 + jj;
      if (j < NG) {
        float sv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int r = 0; r < GPUS; r++) { // GPU order
#pragma unroll
          for (int e = 0; e < 4; e++) {
            sv[e] += at(pv[jj][r], e);
          }
        }
#pragma unroll
        for (int e = 0; e < 4; e++) {
          u[j][e] = round_bf16(at(pre2[j], e) + round_bf16(sv[e]));
        }
        ar_tok_st8_dev(u_out + (size_t)t * H + col(j), make_uint2(pack_bf16x2(u[j][0], u[j][1]), pack_bf16x2(u[j][2], u[j][3])));
      }
    }
  }
#if STATIC_STAMP_SET == 0
  if (tid == 0) { // (timing build: stage stamp 0 = the GPUs' partials landed and added)
    STAGE_STAMP_AT(0)
  }
#endif
  // 2. c . u and u . u, then every source's sums over the 4 waves
  {
    float d = 0.f, sq = 0.f;
#pragma unroll
    for (int j = 0; j < NG; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        d = fmaf(cc[j][e], u[j][e], d);
        sq = fmaf(u[j][e], u[j][e], sq);
      }
    d = wave_sum64(d);
    sq = wave_sum64(sq);
    if (lane == 0) {
      red[wave * NV + 2 * ATTN_TOK_NBMAX] = d;
      red[wave * NV + 2 * ATTN_TOK_NBMAX + 1] = sq;
    }
  }
  ar_tok_lds_bar();
  float wts[ATTN_TOK_NBMAX + 1];
  {
    float mx = -3.0e38f;
#pragma unroll
    for (int s_ = 0; s_ <= ATTN_TOK_NBMAX; s_++) {
      bool const lv = s_ == ATTN_TOK_NBMAX || s_ < nb;
      float const dt = red[2 * s_] + red[NV + 2 * s_] + red[2 * NV + 2 * s_] + red[3 * NV + 2 * s_],
                  sq = red[2 * s_ + 1] + red[NV + 2 * s_ + 1] + red[2 * NV + 2 * s_ + 1] + red[3 * NV + 2 * s_ + 1];
      wts[s_] = lv ? dt * rsqrtf(sq / (float)H + eps) : -3.0e38f;
      mx = fmaxf(mx, wts[s_]);
    }
    float den = 0.f;
#pragma unroll
    for (int s_ = 0; s_ <= ATTN_TOK_NBMAX; s_++) {
      bool const lv = s_ == ATTN_TOK_NBMAX || s_ < nb;
      wts[s_] = lv ? __expf(wts[s_] - mx) : 0.f;
      den += wts[s_];
    }
#pragma unroll
    for (int s_ = 0; s_ <= ATTN_TOK_NBMAX; s_++) {
      wts[s_] = wts[s_] / den;
    }
  }
  // 3. h, its sum of squares, out
#pragma unroll
  for (int j = 0; j < NG; j++)
#pragma unroll
    for (int e = 0; e < 4; e++) {
      u[j][e] *= wts[ATTN_TOK_NBMAX];
    }
#pragma unroll 1
  for (int b = 0; b < nb; b++) {
    float wb = 0.f;
#pragma unroll
    for (int s_ = 0; s_ < ATTN_TOK_NBMAX; s_++) {
      wb = s_ == b ? wts[s_] : wb;
    }
    uint2 rr[NG];
#pragma unroll
    for (int j = 0; j < NG; j++) {
      rr[j] = bkl[(b * NG + j) * 256 + tid];
    }
#pragma unroll
    for (int j = 0; j < NG; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        u[j][e] = fmaf(wb, at(rr[j], e), u[j][e]);
      }
  }
  float hs = 0.f;
#pragma unroll
  for (int j = 0; j < NG; j++)
#pragma unroll
    for (int e = 0; e < 4; e++) {
      hs = fmaf(u[j][e], u[j][e], hs);
    }
  hs = wave_sum64(hs);
  if (lane == 0) {
    red[4 * NV + wave] = hs;
  }
  ar_tok_lds_bar();
  float const rstd = rsqrtf((red[4 * NV] + red[4 * NV + 1] + red[4 * NV + 2] + red[4 * NV + 3]) / (float)H + eps);
#if STATIC_STAMP_SET == 0
  if (tid == 0) { // (stage stamp 1: the weights and h's RMS known)
    STAGE_STAMP_AT(1)
  }
#endif
#pragma unroll
  for (int j = 0; j < NG; j++) {
    float o4[4];
#pragma unroll
    for (int e = 0; e < 4; e++) {
      o4[e] = u[j][e] * rstd * at(ow2[j], e);
    }
    ar_tok_st8_dev(out + (size_t)t * H + col(j), make_uint2(pack_bf16x2(o4[0], o4[1]), pack_bf16x2(o4[2], o4[3])));
  }
#if STATIC_AR_POST_NOWAIT
  __syncthreads();
  if (tid == 0) {
    cnt_post(cnt, 1);
  }
#else
  task_post(cnt, 1);
#endif
  __syncthreads();
}

// RS (the o_proj a COMBINE_PUSH_RS gemm_tile, T > 16 decode): task (t, 0) on GPU r = token t, this GPU's columns
// [r CW, (r + 1) CW) (CW = H / GPUS, the GPUs are the chunks): the GPUs' partials of those columns (o_proj sent each its
// rows: [GPUS][CW / PR][T][PR] bf16 at AOFF), u and the chunk's dots as attn_residual_task; the dots to every GPU (the
// node's exchange buffer at XOFF: [T][GPUS][AR_PW] fp32), the GPUs' added in GPU order; the weights, out; this GPU's u and
// out columns to every GPU ([T][H] bf16 each, after the dots), then the token's whole rows (every GPU's columns, polled)
// into u_out and out (this GPU's graph tensors). Every value 0xFF before each launch (its own flag). One count per task.
#ifndef STATIC_AR_U_EARLY
#define STATIC_AR_U_EARLY 0
#endif
template <size_t AOFF, size_t XOFF, int H, int PR>
__device__ __forceinline__ void attn_residual_rs_task(G const &g,
                                                      KernelLocals &L,
                                                      __hip_bfloat16 const *bank,
                                                      __hip_bfloat16 const *prefix,
                                                      __hip_bfloat16 const *q,
                                                      __hip_bfloat16 const *gain,
                                                      __hip_bfloat16 const *ow,
                                                      __hip_bfloat16 *u_out,
                                                      int const *info,
                                                      float eps,
                                                      __hip_bfloat16 *out,
                                                      uint32_t *cnt,
                                                      int t STAGE_STAMP_PARAM,
                                                      uint32_t const *pre_cnt = nullptr,
                                                      uint32_t pre_need = 0) {
  constexpr int NC = GPUS, CW = H / GPUS, CPT = 4, NTH = CW / CPT;
  static_assert(H % (GPUS * 32) == 0 && NTH <= 256 && CW % PR == 0 && PR % 8 == 0,
                "attn_residual RS: 4 columns per thread, whole o_proj row tiles per GPU");
  using V = ArVec<CPT>;
  constexpr int NVC = CW / 8, NV = GPUS * NVC, LPT = (NV + 255) / 256; // 16-B vectors per GPU's columns, all GPUs'
  constexpr int NVR = H / 8, LPR = (NVR + 255) / 256;                  // 16-B vectors per whole row
  constexpr size_t RANK = (size_t)T * CW * 2;                          // a source GPU's slot in o_proj's buffer
  constexpr size_t XP = XOFF, XU = XOFF + (size_t)T * GPUS * AR_PW * 4, XO = XU + (size_t)T * H * 2;
  int const ch = g.rank;
  int const tid = task_tid_all(), lane = tid & 63, wave = tid >> 6;
  int const nb = info[0];
  size_t const bs_tok = (size_t)info[1], bs_row = (size_t)info[2];
  bool const live = tid < NTH;
  int const c0 = ch * CW + CPT * (live ? tid : 0);
  uint4 *const lnd = reinterpret_cast<uint4 *>(L.sm);                  // [GPUS][CW] bf16
  float *const red = reinterpret_cast<float *>(L.sm + GPUS * CW * 2); // [4 waves][AR_PW]
  float *const tot = red + 4 * AR_PW;                                  // [AR_PW]
  unsigned char const *const xr = g.rv + rg_set();
  __syncthreads();
  if (pre_cnt != nullptr) {
    if (tid == 0) {
      cnt_wait_in(pre_cnt, pre_need);
    }
    __syncthreads();
  }
  float pre[CPT], cc[CPT], owv[CPT];
  typename V::type bk[ATTN_RES_NBMAX];
  {
    typedef typename V::type VT;
    VT const p2 = live ? *reinterpret_cast<VT const *>(prefix + (size_t)t * H + c0) : V::zero(),
             q2 = live ? *reinterpret_cast<VT const *>(q + c0) : V::zero(),
             g2 = live ? *reinterpret_cast<VT const *>(gain + c0) : V::zero(),
             o2 = live ? *reinterpret_cast<VT const *>(ow + c0) : V::zero();
#pragma unroll
    for (int b = 0; b < ATTN_RES_NBMAX; b++) {
      bk[b] = (live && b < nb) ? *reinterpret_cast<VT const *>(bank + t * bs_tok + b * bs_row + c0) : V::zero();
    }
#pragma unroll
    for (int e = 0; e < CPT; e++) {
      pre[e] = V::at(p2, e);
      cc[e] = V::at(q2, e) * V::at(g2, e);
      owv[e] = V::at(o2, e);
    }
  }
  for (int i = tid; i < 4 * AR_PW; i += 256) {
    red[i] = 0.f;
  }
  __syncthreads();
  // the early round (c . bank row b, the bank rows' pairwise dots) of this GPU's columns, to every GPU
#pragma unroll
  for (int b = 0; b < ATTN_RES_NBMAX; b++) {
    if (b < nb) {
      float d = 0.f;
#pragma unroll
      for (int e = 0; e < CPT; e++) {
        d = fmaf(cc[e], V::at(bk[b], e), d);
      }
      d = wave_sum64(d);
      if (lane == 0) {
        red[wave * AR_PW + b] = d;
      }
#pragma unroll
      for (int b2 = b; b2 < ATTN_RES_NBMAX; b2++) {
        if (b2 < nb) {
          float g2 = 0.f;
#pragma unroll
          for (int e = 0; e < CPT; e++) {
            g2 = fmaf(V::at(bk[b], e), V::at(bk[b2], e), g2);
          }
          g2 = wave_sum64(g2);
          if (lane == 0) {
            red[wave * AR_PW + ar_gidx(b, b2)] = g2;
          }
        }
      }
    }
  }
  __syncthreads();
  if (tid < AR_PA) {
    push4(g, XP + (((size_t)t * NC + ch) * AR_PW + tid) * 4,
          __float_as_uint(red[tid] + red[AR_PW + tid] + red[2 * AR_PW + tid] + red[3 * AR_PW + tid]));
  }
  // the GPUs' partials of this GPU's columns into LDS (all loads of a round in flight)
  {
    uint4 v[LPT];
    bool ok[LPT];
#pragma unroll
    for (int j = 0; j < LPT; j++) {
      int const k = tid + 256 * j;
      ok[j] = k >= NV || (SOLO && k / NVC != g.rank);
      v[j] = make_uint4(0u, 0u, 0u, 0u);
    }
    for (;;) {
      bool all = true;
#if STATIC_AR_POLL2
      // (STATIC_AR_POLL2: every load of a pass issued in one burst, then checked: a load checked inside its own branch was
      // waited for alone - MI355X dec265, latent_up's R fill: first pass 8.8 -> 3.8 us)
#pragma unroll
      for (int j = 0; j < LPT; j++) {
        int const k = tid + 256 * j, r = k / NVC, n = 8 * (k % NVC);
        if (k < NV && !(SOLO && r != g.rank)) {
          v[j] = ld16_at<AUX_SYS>(g.rv, (uint32_t)(AOFF + rg_set() + r * RANK + (((size_t)(n / PR) * T + t) * PR + n % PR) * 2));
        }
      }
#pragma unroll
      for (int j = 0; j < LPT; j++) {
        ok[j] = ok[j] || valid16w(v[j]);
        all = all && ok[j];
      }
#else
#pragma unroll
      for (int j = 0; j < LPT; j++) {
        if (!ok[j]) {
          int const k = tid + 256 * j, r = k / NVC, n = 8 * (k % NVC);
          v[j] = ld16_at<AUX_SYS>(g.rv, (uint32_t)(AOFF + rg_set() + r * RANK + (((size_t)(n / PR) * T + t) * PR + n % PR) * 2));
          ok[j] = valid16w(v[j]);
        }
        all = all && ok[j];
      }
#endif
      if (all) {
        break;
      }
      tail_pause();
    }
#pragma unroll
    for (int j = 0; j < LPT; j++) {
      int const k = tid + 256 * j;
      if (k < NV) {
        lnd[k] = v[j];
      }
    }
  }
  __syncthreads();
  float u[CPT];
  {
    float sv[CPT];
#pragma unroll
    for (int e = 0; e < CPT; e++) {
      sv[e] = 0.f;
    }
    __hip_bfloat16 const *const lb = reinterpret_cast<__hip_bfloat16 const *>(lnd);
#pragma unroll
    for (int r = 0; r < GPUS; r++) { // GPU order
      typename V::type const w2 = live ? *reinterpret_cast<typename V::type const *>(lb + r * CW + CPT * tid) : V::zero();
#pragma unroll
      for (int e = 0; e < CPT; e++) {
        sv[e] += V::at(w2, e);
      }
    }
#pragma unroll
    for (int e = 0; e < CPT; e++) {
      u[e] = live ? round_bf16(pre[e] + round_bf16(sv[e])) : 0.f;
    }
  }
#if STATIC_STAMP_SET == 0
  if (tid == 0) { // (timing build: stage stamp 0 = the GPUs' partials landed and added)
    STAGE_STAMP_AT(0)
  }
#endif
#if STATIC_AR_U_EARLY == 1
  // (STATIC_AR_U_EARLY: this GPU's u columns to every GPU now, not after the weights - their stores go out during the dots' hop
  // instead of before the last one: a wave's remote stores go out ~0.6 us apart, dec399 T 64: weights -> end 7.3 us)
  if (live) {
    push8(g, XU + ((size_t)t * H + c0) * 2, make_uint2(pack_bf16x2(u[0], u[1]), pack_bf16x2(u[2], u[3])));
  }
#endif
  // after u: u . u, c . u, u . bank row b of this GPU's columns, to every GPU
  {
    float v0 = 0.f, v1 = 0.f;
#pragma unroll
    for (int e = 0; e < CPT; e++) {
      v0 = fmaf(u[e], u[e], v0);
      v1 = fmaf(cc[e], u[e], v1);
    }
    v0 = wave_sum64(v0);
    v1 = wave_sum64(v1);
    if (lane == 0) {
      red[wave * AR_PW + AR_PA] = v0;
      red[wave * AR_PW + AR_PA + 1] = v1;
    }
#pragma unroll
    for (int b = 0; b < ATTN_RES_NBMAX; b++) {
      if (b < nb) {
        float d = 0.f;
#pragma unroll
        for (int e = 0; e < CPT; e++) {
          d = fmaf(u[e], V::at(bk[b], e), d);
        }
        d = wave_sum64(d);
        if (lane == 0) {
          red[wave * AR_PW + AR_PA + 2 + b] = d;
        }
      }
    }
  }
  __syncthreads();
  if (tid < AR_PB) {
    int const i = AR_PA + tid;
    push4(g, XP + (((size_t)t * NC + ch) * AR_PW + i) * 4,
          __float_as_uint(red[i] + red[AR_PW + i] + red[2 * AR_PW + i] + red[3 * AR_PW + i]));
  }
#if STATIC_AR_U_EARLY == 2
  // (2: the u columns right after the after-u dots' push, so those stores do not wait behind the u stores - dec470, 1: the dots'
  // hop 4.2 -> 5.3 us, weights -> end 7.5 -> 4.6)
  if (live) {
    push8(g, XU + ((size_t)t * H + c0) * 2, make_uint2(pack_bf16x2(u[0], u[1]), pack_bf16x2(u[2], u[3])));
  }
#endif
  // the GPUs' values added in GPU order (thread i: value i; its NC loads in flight, re-loaded until none is all-ones)
  for (int i = tid; i < AR_PW; i += 256) {
    float v[NC];
    bool ok[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) {
      ok[c] = false;
      v[c] = 0.f;
    }
    for (;;) {
      bool all = true;
#if STATIC_AR_POLL2
#pragma unroll
      for (int c = 0; c < NC; c++) {
        v[c] = __uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(buf_rsrc(xr), (uint32_t)(XP + (((size_t)t * NC + c) * AR_PW + i) * 4), 0, AUX_SYS));
      }
#pragma unroll
      for (int c = 0; c < NC; c++) {
        ok[c] = ok[c] || __float_as_uint(v[c]) != 0xFFFFFFFFu;
        all = all && ok[c];
      }
#else
#pragma unroll
      for (int c = 0; c < NC; c++) {
        if (!ok[c]) {
          v[c] = __uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(buf_rsrc(xr), (uint32_t)(XP + (((size_t)t * NC + c) * AR_PW + i) * 4), 0, AUX_SYS));
          ok[c] = __float_as_uint(v[c]) != 0xFFFFFFFFu;
        }
        all = all && ok[c];
      }
#endif
      if (all) {
        break;
      }
      tail_pause();
    }
    float sum = 0.f;
#pragma unroll
    for (int c = 0; c < NC; c++) {
      sum += v[c];
    }
    tot[i] = sum;
  }
  __syncthreads();
  float wts[AR_NS];
  float rstd_w;
  {
    int const s1 = lane;
    bool const lv = s1 < AR_NS && (s1 == ATTN_RES_NBMAX || s1 < nb);
    float sc = -3.0e38f;
    if (lv) {
      bool const is_u = s1 == ATTN_RES_NBMAX;
      float const sq = is_u ? tot[AR_PA] : tot[ar_gidx(s1, s1)], dt = is_u ? tot[AR_PA + 1] : tot[s1];
      sc = dt * rsqrtf(sq / (float)H + eps);
    }
    float const mxw = wave_max64(sc);
    float const ex = lv ? __expf(sc - mxw) : 0.f;
    float const dn = wave_sum64(ex);
    float const w = ex / dn;
    float hp = 0.f;
#pragma unroll
    for (int s2 = 0; s2 < AR_NS; s2++) {
      float const w2 = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(w), s2));
      wts[s2] = w2;
      if (s1 < AR_NS && s2 >= s1) {
        float const gv = s2 == ATTN_RES_NBMAX ? (s1 == ATTN_RES_NBMAX ? tot[AR_PA] : tot[AR_PA + 2 + s1]) : tot[ar_gidx(s1, s2)];
        hp = fmaf((s1 == s2 ? 1.f : 2.f) * w * w2, gv, hp);
      }
    }
    float const hs = wave_sum64(hp);
    rstd_w = rsqrtf(fmaxf(hs, 0.f) / (float)H + eps);
  }
#if STATIC_STAMP_SET == 0
  if (tid == 0) { // (stage stamp 1: the weights and h's RMS known)
    STAGE_STAMP_AT(1)
  }
#endif
  if (live) { // this GPU's u and out columns to every GPU
    float h[CPT], o[CPT];
#pragma unroll
    for (int e = 0; e < CPT; e++) {
      h[e] = wts[ATTN_RES_NBMAX] * u[e];
    }
#pragma unroll
    for (int b = 0; b < ATTN_RES_NBMAX; b++) {
      if (b < nb) {
#pragma unroll
        for (int e = 0; e < CPT; e++) {
          h[e] = fmaf(wts[b], V::at(bk[b], e), h[e]);
        }
      }
    }
#pragma unroll
    for (int e = 0; e < CPT; e++) {
      o[e] = h[e] * rstd_w * owv[e];
    }
#if !STATIC_AR_U_EARLY
    push8(g, XU + ((size_t)t * H + c0) * 2, make_uint2(pack_bf16x2(u[0], u[1]), pack_bf16x2(u[2], u[3])));
#endif
    push8(g, XO + ((size_t)t * H + c0) * 2, make_uint2(pack_bf16x2(o[0], o[1]), pack_bf16x2(o[2], o[3])));
  }
  // the token's whole rows (every GPU's columns) into this GPU's u_out and out
#if STATIC_AR_POLL2
  { // (STATIC_AR_POLL2: every row's u and out pieces of the thread loaded in one burst per pass, checked after; stores at the end)
    uint4 a[LPR], b[LPR];
    bool oa[LPR], ob[LPR];
#pragma unroll
    for (int j = 0; j < LPR; j++) {
      int const k = tid + 256 * j;
      oa[j] = k >= NVR || (SOLO && (8 * k) / CW != g.rank);
      ob[j] = oa[j];
      a[j] = make_uint4(0u, 0u, 0u, 0u);
      b[j] = a[j];
    }
    for (;;) {
#pragma unroll
      for (int j = 0; j < LPR; j++) {
        int const k = tid + 256 * j;
        if (k < NVR && !(SOLO && (8 * k) / CW != g.rank)) {
          a[j] = ld16_at<AUX_SYS>(xr, (uint32_t)(XU + ((size_t)t * H + 8 * k) * 2));
          b[j] = ld16_at<AUX_SYS>(xr, (uint32_t)(XO + ((size_t)t * H + 8 * k) * 2));
        }
      }
      bool all = true;
#pragma unroll
      for (int j = 0; j < LPR; j++) {
        oa[j] = oa[j] || valid16w(a[j]);
        ob[j] = ob[j] || valid16w(b[j]);
        all = all && oa[j] && ob[j];
      }
      if (all) {
        break;
      }
      tail_pause();
    }
#pragma unroll
    for (int j = 0; j < LPR; j++) {
      int const k = tid + 256 * j;
      if (k < NVR) {
        st16_dev(u_out + (size_t)t * H + 8 * k, a[j]);
        st16_dev(out + (size_t)t * H + 8 * k, b[j]);
      }
    }
  }
#else
#pragma unroll
  for (int j = 0; j < LPR; j++) {
    int const k = tid + 256 * j;
    if (k < NVR) {
      uint4 a, b;
      bool oa = SOLO && (8 * k) / CW != g.rank, ob = oa;
      if (oa) {
        a = make_uint4(0u, 0u, 0u, 0u);
        b = a;
      }
      for (;;) {
        if (!oa) {
          a = ld16_at<AUX_SYS>(xr, (uint32_t)(XU + ((size_t)t * H + 8 * k) * 2));
          oa = valid16w(a);
        }
        if (!ob) {
          b = ld16_at<AUX_SYS>(xr, (uint32_t)(XO + ((size_t)t * H + 8 * k) * 2));
          ob = valid16w(b);
        }
        if (oa && ob) {
          break;
        }
        tail_pause();
      }
      st16_dev(u_out + (size_t)t * H + 8 * k, a);
      st16_dev(out + (size_t)t * H + 8 * k, b);
    }
  }
#endif
#if STATIC_AR_POST_NOWAIT
  __syncthreads();
  if (tid == 0) {
    cnt_post(cnt, 1);
  }
#else
  task_post(cnt, 1);
#endif
  __syncthreads();
}

// task (x, y) = token x, chunk y; A = the push_all gemm_tile node (its exchange
// buffer, H = its params[2]); params = [H, eps bits]; buffer slots = bank,
// prefix, q, gain, ow, u_out, info, the partial sums' scratch; counters: 1 + 2 T
// (counter 0: the tasks done, the node's consumers wait for it)
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class A,
          class... IN>
__device__ __forceinline__ void run_attn_residual(Maps const &,
                                                  G const &g,
                                                  KernelLocals &L,
                                                  StaticTask const &tk) {
  static_assert(A::buf >= 0 && (A::v[0] == COMBINE_PUSH_ALL || A::v[0] == COMBINE_PUSH_RS || A::v[0] == COMBINE_FINISH_RS) &&
                    A::v[2] == PARAMS::v[0],
                "attn_residual: a push_all (or push_rs / finish_rs) gemm_tile's output of H rows");
  if constexpr (A::v[0] == COMBINE_PUSH_RS || A::v[0] == COMBINE_FINISH_RS) { // (RS: the node's buffer the exchange area; out and u_out buffer slots 5, 7)
    static_assert(SELF::y == 1 && SELF::buf >= 0 &&
                      exchange_bytes[SELF::buf] == (size_t)T * GPUS * AR_PW * 4 + 2 * (size_t)T * PARAMS::v[0] * 2,
                  "attn_residual RS: a task per token, the node's exchange buffer [T][GPUS][AR_PW] fp32 + 2 x [T][H] bf16");
    uint32_t const *pc_ = nullptr;
    uint32_t pn_ = 0;
    if constexpr (sizeof(PARAMS::v) / sizeof(int) > 3) { // params [H, eps, counter, need]: wait for the prefix's writer
      pc_ = cnt_of(g) + param_at<PARAMS, 2>();
      pn_ = (uint32_t)param_at<PARAMS, 3>();
    }
    attn_residual_rs_task<exchange_offset[A::buf], exchange_offset[SELF::buf], PARAMS::v[0], PARAMS::v[0] / A::x>(
        g, L,
        buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
        buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>()),
        buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>()),
        buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 3>()),
        buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 4>()),
        buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 5>()),
        buf_at<int const>(g, slot_at<BUF_SLOTS, 6>()),
        __int_as_float(PARAMS::v[1]),
        buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 7>()),
        cnt_of(g) + SELF::counter,
        tk.x STAGE_STAMP_ARG(L.stage_stamps),
        pc_,
        pn_);
    return;
  } else if constexpr (SELF::y == 1) { // TOK (chunks = 1): a task per token, no exchange
    static_assert(SELF::buf >= 0 && SELF::x == T && SELF::counter >= 0, "attn_residual TOK: one task per token, its counter");
    uint32_t const *pc_ = nullptr;
    uint32_t pn_ = 0;
    if constexpr (sizeof(PARAMS::v) / sizeof(int) > 3) { // params [H, eps, counter, need]: wait for the prefix's writer
      pc_ = cnt_of(g) + param_at<PARAMS, 2>();
      pn_ = (uint32_t)param_at<PARAMS, 3>();
    }
    attn_residual_tok_task<exchange_offset[A::buf], PARAMS::v[0], PARAMS::v[0] / A::x>(
        g, L,
        buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
        buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>()),
        buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>()),
        buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 3>()),
        buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 4>()),
        buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 5>()),
        buf_at<int const>(g, slot_at<BUF_SLOTS, 6>()),
        __int_as_float(PARAMS::v[1]),
        buf_at<__hip_bfloat16>(g, SELF::buf),
        cnt_of(g) + SELF::counter,
        tk.x STAGE_STAMP_ARG(L.stage_stamps),
        pc_,
        pn_);
    return;
  } else {
  static_assert(SELF::buf >= 0 && SELF::x == T && SELF::counter >= 0,
                "attn_residual: one task per token and chunk, its counters");
  if constexpr (sizeof(PARAMS::v) / sizeof(int) > 3) { // params [H, eps, counter, need]: wait for the prefix's writer
    attn_residual_task<exchange_offset[A::buf], PARAMS::v[0], SELF::y, PARAMS::v[0] / A::x>(
        g, L,
        buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
        buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>()),
        buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>()),
        buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 3>()),
        buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 4>()),
        buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 5>()),
        buf_at<int const>(g, slot_at<BUF_SLOTS, 6>()),
        __int_as_float(PARAMS::v[1]),
        buf_at<__hip_bfloat16>(g, SELF::buf),
        buf_at<float>(g, slot_at<BUF_SLOTS, 7>()),
        cnt_of(g) + SELF::counter,
        tk.x,
        tk.y STAGE_STAMP_ARG(L.stage_stamps),
        nullptr,
        cnt_of(g) + param_at<PARAMS, 2>(),
        (uint32_t)param_at<PARAMS, 3>());
    return;
  }
  attn_residual_task<exchange_offset[A::buf], PARAMS::v[0], SELF::y, PARAMS::v[0] / A::x>(
      g, L,
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 4>()),
      buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 5>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 6>()),
      __int_as_float(PARAMS::v[1]),
      buf_at<__hip_bfloat16>(g, SELF::buf),
      buf_at<float>(g, slot_at<BUF_SLOTS, 7>()),
      cnt_of(g) + SELF::counter,
      tk.x,
      tk.y STAGE_STAMP_ARG(L.stage_stamps));
  }
}

template <class A, class... R>
constexpr int ar_input_combine() {
  return A::v[0];
}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_attn_residual() {
  // (RS: grid y 1, the chunk = H / GPUS columns; TOK: grid y 1 with a push_all input - the bank rows in LDS)
  if constexpr (SELF::y == 1 && ar_input_combine<IN...>() == COMBINE_PUSH_ALL) {
    return 1024 + ATTN_TOK_NBMAX * (PARAMS::v[0] / 1024) * 256 * 8;
  }
  return GPUS * (PARAMS::v[0] / (SELF::y == 1 ? GPUS : SELF::y)) * 2 + (4 * AR_PW + AR_PW + 32) * 4; // (+ 32: not used now; kept, the LDS size as measured)
}

} // namespace static_mk
