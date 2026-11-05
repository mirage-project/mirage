// tasks/topk_route.cuh -- the routing of one token t: add the router's NSPLIT
// partial sums of row t (lpart = the router node's output, K parts in slots
// (gemm_tile.cuh COMBINE_SLOTS), in a fixed order so that every GPU routes the
// same way), sigmoid, + score correction bias, top K of the NE experts,
// renormalise the K sigmoid scores (and multiply them by the routed scaling
// factor, if there is one) -> K (expert, weight) pairs, written to `pairs`
// (the node's buffer, [T][K]) as expert << 32 | weight bits
// (0xFF-prefilled: moe_experts polls them). Sub-experts (DUP > 1): each expert
// is DUP experts of the expert bank (its intermediate in DUP parts, e.g. one
// GPU's 512 features of GLM-5.3-Flash as 2 x 256), so expert e gives DUP pairs
// (e DUP + j, weight), j < DUP: [T][K DUP] pairs. Shared experts (SHARED > 0,
// GLM-5.3-Flash's shared expert as 2 sub-experts, run in the experts node's
// queue): every token also gets the SHARED bank experts after the routed ones,
// (NE DUP + s, 1.0), s < SHARED: [T][K DUP + SHARED] pairs. warm = false (this
// node's first task on the SM: its code is cold; static_schedule.case_code):
// first a dry pass, the same code on bias-only scores without loading or
// storing anything (it loads the instructions while the router's sums are on
// their way), then the real pass. 256 threads. ROWS < T (params[5]; a serving
// framework's speculative verify step of ROWS live rows, the rest idle): a row
// t >= ROWS is not routed: its K DUP routed pairs are the shared bank experts
// (expert 0 without shared experts) at weight 0, so it streams no expert of its
// own (its logits are not read, not re-armed).
#pragma once
#include "../core.cuh"

namespace static_mk {

// the scratch below is in the ring area (from the dynamic base: no load is in
// flight during topk_route)
template <int NE, int K>
constexpr int topk_scratch_bytes() {
  return 2 * NE * 4 + 2 * (NE + 8 * K) * 4 + 2 * K * 4;
}

// N: the router's rows (the logits' row stride, >= NE: rows past NE are
// padding, not read); SCALE_BITS: the fp32 bits of the routed scaling factor
// (0: none); DUP: sub-experts per expert; SHARED: the shared bank experts
// (above)
template <int NSPLIT,
          int NE,
          int N,
          int K,
          int SCALE_BITS = 0,
          int DUP = 1,
          int SHARED = 0,
          int ROWS = T>
__device__ __forceinline__ void topk_route_task(float const *bias,
                                                float *lpart,
                                                unsigned long long *pairs,
                                                int t,
                                                char *sm,
                                                bool warm STAGE_STAMP_PARAM) {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads(); // the scratch below is in the ring area: the previous tile's
                   // epilogue must have read every landed stage
  static_assert(ROWS >= 1 && ROWS <= T, "topk_route: 1 .. T live rows");
  if constexpr (ROWS < T) {
    if (t >= ROWS) { // an idle row (the header comment): no expert of its own
      if (threadIdx.x < K * DUP + SHARED) {
        int const i = threadIdx.x;
        unsigned const e =
            i < K * DUP
                ? (SHARED > 0 ? NE * DUP + i % (SHARED > 0 ? SHARED : 1) : 0u)
                : NE * DUP + (i - K * DUP);
        pairs[t * (K * DUP + SHARED) + i] =
            ((unsigned long long)e << 32) |
            (unsigned long long)__float_as_uint(i < K * DUP ? 0.f : 1.0f);
      }
      return;
    }
  }
  // stage 1: each warp ranks PER_WARP of the keys and keeps its top K; stage 2
  // ranks those NCAND candidates
  constexpr int PER_WARP = NE / 8, NCAND = 8 * K;
  static_assert(NE % 32 == 0 && PER_WARP <= 128 && PER_WARP >= K &&
                    NCAND <= 256,
                "topk_route: 8 warps of up to 128 experts, top K of each");
  // scratch: sc_s [NE] sigmoid scores, [NE] not used, key_s [NE] + cand_s
  // [NCAND] 64-bit keys, sel_s [K], wsel_s [K]
  float *sc_s = reinterpret_cast<float *>(sm);
  unsigned long long *key_s = reinterpret_cast<unsigned long long *>(
      sc_s + 2 * NE); // key = orderable (score + bias) << 32 | (~expert id)
  unsigned long long *cand_s = key_s + NE;
  int *sel_s = reinterpret_cast<int *>(cand_s + NCAND);
  float *wsel_s = reinterpret_cast<float *>(sel_s + K);
  // pass 0 (only when the code is cold: !warm) is the dry pass: the same code
  // on bias-only scores, no loads, no stores; one copy of the code for both
  // passes (unroll 1), so pass 0 loads the instructions pass 1 runs
#pragma unroll 1
  for (int pass = warm ? 1 : 0; pass < 2; pass++) {
    bool const real = pass == 1;
    // scores: thread q < NE / 4 owns experts 4q .. 4q + 3; the NSPLIT partial
    // sums are loaded together (float4, independent loads)
    if (threadIdx.x < NE / 4) {
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      float4 lp[NSPLIT];
#pragma unroll
      for (int p_ = 0; p_ < NSPLIT; p_++) {
        lp[p_] = make_float4(
            0.f, 0.f, 0.f, 0.f); // dry pass: scores from the bias only
      }
      if (real) { // re-load all parts until every one has landed (0xFF prefill)
        bool all;
        do {
#pragma unroll
          for (int p_ = 0; p_ < NSPLIT; p_++) {
            lp[p_] = ldf4_relaxed(lpart + ((size_t)p_ * T + t) * N +
                                  threadIdx.x * 4);
          }
          all = true;
#pragma unroll
          for (int p_ = 0; p_ < NSPLIT; p_++) {
            all = all && validf4(lp[p_]);
          }
        } while (!all);
      }
#pragma unroll
      for (int p_ = 0; p_ < NSPLIT; p_++) {
        acc.x += lp[p_].x;
        acc.y += lp[p_].y;
        acc.z += lp[p_].z;
        acc.w += lp[p_].w;
      }
      float4 const b4 =
          *reinterpret_cast<float4 const *>(bias + threadIdx.x * 4);
      float const lg4[4] = {acc.x, acc.y, acc.z, acc.w},
                  bs4[4] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
      for (int k = 0; k < 4; k++) {
        int const e = threadIdx.x * 4 + k;
        float const sg = 1.0f / (1.0f + __expf(-lg4[k]));
        sc_s[e] = sg;
        uint32_t u = __float_as_uint(sg + bs4[k]);
        u = (u & 0x80000000u)
                ? ~u
                : (u | 0x80000000u); // float -> unsigned with the same order
        key_s[e] =
            ((unsigned long long)u << 32) |
            (unsigned long long)(0xFFFFFFFFu -
                                 (uint32_t)e); // ties: lower expert id wins
      }
    }
    __syncthreads();
    // stage 1: warp w ranks its PER_WARP keys among themselves; the K with rank
    // < K go to cand_s[w * K + rank]
    {
      unsigned long long mine[4];
      int rk[4];
#pragma unroll
      for (int q = 0; q < 4; q++) {
        int const i = lane + 32 * q;
        mine[q] = (i < PER_WARP) ? key_s[warp * PER_WARP + i] : 0ull;
        rk[q] = 0;
      }
#pragma unroll 16
      for (int j = 0; j < PER_WARP; j++) {
        unsigned long long const o = key_s[warp * PER_WARP + j];
#pragma unroll
        for (int q = 0; q < 4; q++) {
          rk[q] += (o > mine[q]);
        }
      }
#pragma unroll
      for (int q = 0; q < 4; q++) {
        if (lane + 32 * q < PER_WARP && rk[q] < K) {
          cand_s[warp * K + rk[q]] = mine[q];
        }
      }
    }
    __syncthreads();
    // stage 2: NCAND threads rank the NCAND candidates; rank < K -> sel_s[rank]
    // (keys are unique, so ranks are unique)
    if (threadIdx.x < NCAND) {
      unsigned long long const mine = cand_s[threadIdx.x];
      int rk = 0;
#pragma unroll 16
      for (int j = 0; j < NCAND; j++) {
        rk += (cand_s[j] > mine);
      }
      if (rk < K) {
        sel_s[rk] = (int)(0xFFFFFFFFu - (uint32_t)(mine & 0xFFFFFFFFull));
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) { // a guard: the ranks above always fill sel_s
      for (int j = 0; j < K; j++) {
        if (sel_s[j] < 0 || sel_s[j] >= NE) {
          sel_s[j] = 0;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x < K) {
      wsel_s[threadIdx.x] = sc_s[sel_s[threadIdx.x]];
    }
    __syncthreads();
    if (threadIdx.x < K) {
      float sum = 0.f;
      for (int k = 0; k < K; k++) {
        sum += wsel_s[k];
      }
      if (real) {
        float w = wsel_s[threadIdx.x] / sum;
        if constexpr (SCALE_BITS != 0) {
          w *= __int_as_float(SCALE_BITS);
        }
#pragma unroll
        for (int j = 0; j < DUP; j++) {
          pairs[t * (K * DUP + SHARED) + threadIdx.x * DUP + j] =
              ((unsigned long long)(unsigned)(sel_s[threadIdx.x] * DUP + j)
               << 32) |
              (unsigned long long)__float_as_uint(w);
        }
      }
    }
    if constexpr (SHARED > 0) { // the shared bank experts, weight 1
      if (real && threadIdx.x >= K && threadIdx.x < K + SHARED) {
        int const s_ = threadIdx.x - K;
        pairs[t * (K * DUP + SHARED) + K * DUP + s_] =
            ((unsigned long long)(unsigned)(NE * DUP + s_) << 32) |
            (unsigned long long)__float_as_uint(1.0f);
      }
    }
    if constexpr (REARM) { // the one reader of these parts; after the pairs are
                           // written (off the path to moe_experts); not the
                           // padding rows (never read)
      if (real && threadIdx.x < NE / 4)
#pragma unroll
        for (int p_ = 0; p_ < NSPLIT; p_++) {
          rearm16(lpart + ((size_t)p_ * T + t) * N + threadIdx.x * 4);
        }
    }
    __syncthreads();
    if (pass == 0 && threadIdx.x == 0) {
      STAGE_STAMP_AT(0)
    } // timing build: the dry pass's end
  }
#ifdef STATIC_TIMING_BUILD
  if (warm && threadIdx.x == 0 && stamp) {
    stamp[0] = 0; // timing build: no dry pass
  }
#endif
}

// the sub-experts per expert of a topk_route node P (its params[3]; 1 without)
template <class P>
__host__ __device__ constexpr int topk_dup() {
  return param_or<P, 3, 1>() > 1 ? param_or<P, 3, 1>() : 1;
}
// its shared bank experts (params[4]; 0 without)
template <class P>
__host__ __device__ constexpr int topk_shared() {
  return param_or<P, 4, 0>();
}
// the experts of the bank it routes over (routed sub-experts, then the shared
// ones) and its pairs per token
template <class P>
__host__ __device__ constexpr int topk_bank_experts() {
  return P::v[0] * topk_dup<P>() + topk_shared<P>();
}
template <class P>
__host__ __device__ constexpr int topk_pairs_per_token() {
  return P::v[1] * topk_dup<P>() + topk_shared<P>();
}

// topk_route task x = token x; adds the router's LOGITS::y partial sums (K
// parts in slots; its N rows: NE experts, then padding); PARAMS {NE (experts),
// K (experts per token)[, the routed scaling factor as fp32 bits (0: none)[,
// DUP (sub-experts per expert)[, SHARED (shared bank experts)[, ROWS (live
// rows, the header comment; T)]]]]}; BUF_SLOTS {the score correction bias (a
// graph tensor, [NE] fp32)}; warm: above
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class LOGITS,
          class... REST>
__device__ __forceinline__ void run_topk_route(Maps const &,
                                               G const &g,
                                               KernelLocals &L,
                                               StaticTask const &tk,
                                               bool warm) {
  constexpr int NE = PARAMS::v[0], K = PARAMS::v[1];
  // v[2]: the scaling factor's bits, or v's closing 0 when there are 2 params
  // (param_at requires the param)
  constexpr int SCALE_BITS = PARAMS::v[2], N = LOGITS::v[2];
  static_assert(
      LOGITS::v[0] == COMBINE_SLOTS && N >= NE && N % 4 == 0,
      "topk_route: the router's N >= NE logits, its K parts in slots");
  static_assert(SELF::buf >= 0, "topk_route: its output pairs");
  static_assert(K + topk_shared<PARAMS>() <= 256,
                "topk_route: a thread for each routed and shared pair");
  constexpr int ROWS =
      param_or<PARAMS, 5, 0>() > 0 ? param_or<PARAMS, 5, 0>() : T;
  topk_route_task<LOGITS::y,
                  NE,
                  N,
                  K,
                  SCALE_BITS,
                  topk_dup<PARAMS>(),
                  topk_shared<PARAMS>(),
                  ROWS>(buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()),
                        buf_at<float>(g, LOGITS::buf),
                        buf_at<unsigned long long>(g, SELF::buf),
                        tk.x,
                        L.sm,
                        warm STAGE_STAMP_ARG(L.stage_stamps));
}
// dynamic shared memory: the scratch, inside the ring area
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_topk_route() {
  static_assert(topk_scratch_bytes<PARAMS::v[0], PARAMS::v[1]>() <= RING_BYTES,
                "topk_route: its scratch in the ring area");
  return RING_BYTES;
}

} // namespace static_mk
