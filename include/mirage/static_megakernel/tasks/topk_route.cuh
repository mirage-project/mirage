// tasks/topk_route.cuh -- the routing of one token t: add the router's NSPLIT
// partial sums of row t (lpart = the router node's output, K parts in slots
// (gemm_tile.cuh COMBINE_SLOTS), in a fixed order so that every GPU routes the
// same way), sigmoid, + score correction bias, top K of the NE experts,
// renormalise the K sigmoid scores -> K (expert, weight) pairs, written to
// `pairs` (the node's buffer, [T][K]) as expert << 32 | weight bits
// (0xFF-prefilled: moe_experts polls them). warm = false (this node's first
// task on the SM: its code is cold; static_schedule.case_code): first a dry
// pass, the same code on bias-only scores without loading or storing anything
// (it loads the instructions while the router's sums are on their way), then
// the real pass. 256 threads.
#pragma once
#include "../core.cuh"

namespace static_mk {

// the scratch below is in the ring area (from the dynamic base: no load is in
// flight during topk_route)
template <int NE, int K>
constexpr int topk_scratch_bytes() {
  return 2 * NE * 4 + 2 * (NE + 8 * K) * 4 + 2 * K * 4;
}

template <int NSPLIT, int NE, int K>
__device__ __forceinline__ void topk_route_task(float const *bias,
                                                float *lpart,
                                                unsigned long long *pairs,
                                                int t,
                                                char *sm,
                                                bool warm STAGE_STAMP_PARAM) {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads(); // the scratch below is in the ring area: the previous tile's
                   // epilogue must have read every landed stage
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
            lp[p_] = ldf4_relaxed(lpart + ((size_t)p_ * T + t) * NE +
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
        float const w = wsel_s[threadIdx.x] / sum;
        pairs[t * K + threadIdx.x] =
            ((unsigned long long)(unsigned)sel_s[threadIdx.x] << 32) |
            (unsigned long long)__float_as_uint(w);
      }
    }
    if constexpr (REARM) { // the one reader of these parts; after the pairs are
                           // written (off the path to moe_experts)
      if (real && threadIdx.x < NE / 4)
#pragma unroll
        for (int p_ = 0; p_ < NSPLIT; p_++) {
          rearm16(lpart + ((size_t)p_ * T + t) * NE + threadIdx.x * 4);
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

// topk_route task x = token x; adds the router's LOGITS::y partial sums (K
// parts in slots); PARAMS {NE (experts), K (experts per token)}; BUF_SLOTS {the
// score correction bias (a graph tensor, [NE] fp32)}; warm: above
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
  static_assert(LOGITS::v[0] == COMBINE_SLOTS && LOGITS::v[2] == NE,
                "topk_route: the router's NE logits, its K parts in slots");
  static_assert(SELF::buf >= 0, "topk_route: its output pairs");
  topk_route_task<LOGITS::y, NE, K>(
      buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()),
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
