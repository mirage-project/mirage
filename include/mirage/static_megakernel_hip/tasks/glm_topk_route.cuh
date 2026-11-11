// tasks/glm_topk_route.cuh (HIP) -- GLM-5.3-Flash's routing of one token t: K3's topk_route (sigmoid, + score correction
// bias, top K by a radix select, renormalised) with: the router's logits row NL wide (a router padded to whole 128-row
// tiles; experts NE .. NL - 1 never chosen: their bias is -inf), K <= 16, the routed scaling factor (SGLang's
// routed_scaling_factor, fp32 bits; 0: none), and SH shared experts per token after the routed pairs (bank experts SH0 ..
// SH0 + SH - 1, weight 1). Pairs [T][K + SH] (expert << 32 | weight bits; 0xFF-prefilled, polled by the experts).
#pragma once
#include "topk_route.cuh" // topk_pick_radix, the scratch layout

namespace static_mk {

template <int NSPLIT, int NE, int NL, int K, int SCALE_BITS, int SH, int SH0>
__device__ __forceinline__ void glm_topk_route_task(float const *bias,
                                                              float *lpart,
                                                              unsigned long long *pairs,
                                                              int t,
                                                              char *sm,
                                                              bool warm STAGE_STAMP_PARAM) {
  constexpr bool RET = false;
  unsigned long long pair = ~0ull;
  (void)pair;
  __syncthreads(); // the scratch below is LDS a previous task may still read
  constexpr int NCAND = 8 * K; // cand_s's size (the layout of topk_scratch_bytes)
  // (the pick's 300 words: in the [NE] floats after the scores when NE >= 300, else its own region after the rest -
  // topk_pick_own_region, counted in smem_glm_topk_route; GLM-5.3: NE 256)
  static_assert(NE % 64 == 0 && NE / 4 <= 256 && K <= NE && NCAND <= 256 && NL >= NE && NL % 4 == 0,
                "topk_route: NE / 64 experts per lane of wave 0");
  // scratch: sc_s [NE] sigmoid scores, [NE] the pick's (topk_pick_radix, 300
  // words), key_s [NE] + cand_s [NCAND] 64-bit keys, sel_s [K] (+ [K] unused)
  float *sc_s = reinterpret_cast<float *>(sm);
  unsigned long long *key_s =
      reinterpret_cast<unsigned long long *>(sc_s + 2 * NE);
  unsigned long long *cand_s = key_s + NE;
  int *sel_s = reinterpret_cast<int *>(cand_s + NCAND);
#ifdef STATIC_TIMING_BUILD
  // the timing build: thread 0's times in the real pass, from the K parts landing: keys made, top K picked, weights stored
  long long t_land = 0, t_keys = 0, t_sel = 0;
#endif
  // no dry pass on gfx950 (MI355X 10-10: it ended 3-5 us after the router's K
  // parts had landed, and the real pass after it was no faster)
  (void)warm;
#pragma unroll 1
  for (int pass = 1; pass < 2; pass++) {
    bool const real = pass == 1;
    if (STATIC_TIDX < NE / 4) {
      // the bias first: its load is in flight while the K parts are polled
      float4 const b4 =
          *reinterpret_cast<float4 const *>(bias + STATIC_TIDX * 4);
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      float4 lp[NSPLIT];
#pragma unroll
      for (int p_ = 0; p_ < NSPLIT; p_++) {
        lp[p_] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      if (real) { // re-load all parts until every one has landed
        bool all;
        do {
#pragma unroll
          for (int p_ = 0; p_ < NSPLIT; p_++) {
            lp[p_] = as_float4(ld16_at<AUX_DEV>(
                lpart,
                (uint32_t)((((size_t)p_ * T + t) * NL + STATIC_TIDX * 4) * 4)));
          }
          all = true;
#pragma unroll
          for (int p_ = 0; p_ < NSPLIT; p_++) {
            all = all && validf4(lp[p_]);
          }
          if (!all) {
            poll_pause();
          }
        } while (!all);
#ifdef STATIC_TIMING_BUILD
        if (STATIC_TIDX == 0 && stamp) {
          stamp[1] = t_land = gtime(); // the timing build: the K parts landed
        }
#endif
      }
#pragma unroll
      for (int p_ = 0; p_ < NSPLIT; p_++) {
        acc.x += lp[p_].x;
        acc.y += lp[p_].y;
        acc.z += lp[p_].z;
        acc.w += lp[p_].w;
      }
      float const lg4[4] = {acc.x, acc.y, acc.z, acc.w},
                  bs4[4] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
      for (int k = 0; k < 4; k++) {
        int const e = STATIC_TIDX * 4 + k;
        float sg = 1.0f / (1.0f + __expf(-lg4[k]));
        sg = sg == sg ? sg : 0.f; // (a NaN logit: score 0; MI355X 11-07 glmsrv3: the graph capture's warm-up run fed
                                  // uninitialised states, NaN keys outranked every score and padded experts got picked)
        sc_s[e] = sg;
        uint32_t u = __float_as_uint(sg + bs4[k]);
        u = (u & 0x80000000u) ? ~u : (u | 0x80000000u); // orderable
        if (bs4[k] == -INFINITY) {
          u = 0u; // a padded expert: below every real one, never picked
        }
        key_s[e] = ((unsigned long long)u << 32) |
                   (unsigned long long)(0xFFFFFFFFu - (uint32_t)e);
      }
    }
    __syncthreads();
#ifdef STATIC_TIMING_BUILD
    if (real && STATIC_TIDX == 0) {
      t_keys = gtime();
    }
#endif
    // the top K (all 256 threads; topk_pick_radix), into sel_s in expert order;
    // its scratch: the NE unused floats after sc_s
#ifdef STATIC_TIMING_BUILD
    long long const sel_c0 = clock64(), sel_t0 = gtime(); // the timing build: the shader clock's rate over the pick
#endif
    topk_pick_radix<NE, K>(
        key_s,
        sel_s,
        topk_pick_own_region<NE>()
            ? reinterpret_cast<uint32_t *>(sm + topk_scratch_bytes_base<NE, K>())
            : reinterpret_cast<uint32_t *>(sc_s + NE));
    __syncthreads();
#ifdef STATIC_TIMING_BUILD
    if (real && STATIC_TIDX == 0) {
      t_sel = gtime();
    }
    if (real && STATIC_TIDX == 0 && stamp) {
      // stage stamp 1 (replacing the K parts' landing): shader cycles | ns << 32 of the pick
      long long const c1 = clock64(), t1 = gtime();
      stamp[1] = (unsigned long long)(c1 - sel_c0) | ((unsigned long long)(t1 - sel_t0) << 32);
    }
#endif
    // lane k < K of wave 0: pick k, its score, the sum of the K scores by DPP
    // within the row of 16 (MI355X 10-11: thread 0's guard loop over sel_s and
    // a 16-read loop per thread for the sum, one dependent LDS read after the
    // other with two barriers, were ~2000 cycles after the pick)
    static_assert(K <= 16, "glm_topk_route: the weights' sum is one DPP row of 16 (lanes >= K add 0)");
    if (STATIC_TIDX < 64) {
      int e = STATIC_TIDX < K ? sel_s[STATIC_TIDX] : 0;
      e = e < 0 || e >= NE ? 0 : e; // a guard: the pick always fills sel_s
      float const sc = STATIC_TIDX < K ? sc_s[e] : 0.f;
      float sum = sc;
      sum += __int_as_float(__builtin_amdgcn_update_dpp(
          0, __float_as_int(sum), 0x128, 0xF, 0xF, false)); // row_ror:8
      sum += __int_as_float(__builtin_amdgcn_update_dpp(
          0, __float_as_int(sum), 0x124, 0xF, 0xF, false)); // row_ror:4
      sum += __int_as_float(__builtin_amdgcn_update_dpp(
          0, __float_as_int(sum), 0x4E, 0xF, 0xF, false)); // quad [2,3,0,1]
      sum += __int_as_float(__builtin_amdgcn_update_dpp(
          0, __float_as_int(sum), 0xB1, 0xF, 0xF, false)); // quad [1,0,3,2]
      // lane 0's: each lane added the 16 in its own order
      sum = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(sum), 0));
      if (real && STATIC_TIDX < K) {
        float const w = SCALE_BITS ? sc / sum * __int_as_float(SCALE_BITS) : sc / sum;
        if constexpr (RET) {
          pair = ((unsigned long long)(unsigned)e << 32) | (unsigned long long)__float_as_uint(w);
        } else {
          __hip_atomic_store(
              pairs + t * (K + SH) + STATIC_TIDX,
              ((unsigned long long)(unsigned)e << 32) |
                  (unsigned long long)__float_as_uint(w),
              __ATOMIC_RELAXED,
              __HIP_MEMORY_SCOPE_AGENT);
        }
      }
      if (real && STATIC_TIDX >= 16 && STATIC_TIDX < 16 + SH) { // the shared experts: bank experts SH0 + s, weight 1
        __hip_atomic_store(pairs + t * (K + SH) + K + STATIC_TIDX - 16,
                           ((unsigned long long)(unsigned)(SH0 + STATIC_TIDX - 16) << 32) |
                               (unsigned long long)__float_as_uint(1.0f),
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    __syncthreads();
  }
#ifdef STATIC_TIMING_BUILD
  // stage stamp 0: (keys - landed) | (picked - landed) << 16 | (end - landed) << 32, in 10 ns units
  if (STATIC_TIDX == 0 && stamp && t_land) {
    long long const t_end = gtime();
    stamp[0] = (unsigned long long)((t_keys - t_land) / 10) |
               ((unsigned long long)((t_sel - t_land) / 10) << 16) |
               ((unsigned long long)((t_end - t_land) / 10) << 32);
  }
#endif
}


// glm_topk_route task x = token x; adds the router's LOGITS::y partial sums; PARAMS {NE (padded to 64), K, scale bits, SH,
// SH0}; BUF_SLOTS {the score correction bias ([NE] fp32, -inf past the real experts)}
template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class LOGITS, class... REST>
__device__ __forceinline__ void run_glm_topk_route(Maps const &, G const &g, KernelLocals &L, StaticTask const &tk) {
  constexpr int NE = PARAMS::v[0], K = PARAMS::v[1];
  static_assert(LOGITS::v[0] == COMBINE_SLOTS && LOGITS::v[2] >= NE,
                "glm_topk_route: the router's logits (at least NE), its K parts in slots");
  static_assert(SELF::buf >= 0, "glm_topk_route: its output pairs");
  glm_topk_route_task<LOGITS::y, NE, LOGITS::v[2], K, PARAMS::v[2], PARAMS::v[3], PARAMS::v[4]>(
      buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<float>(g, LOGITS::buf),
      buf_at<unsigned long long>(g, SELF::buf),
      tk.x,
      L.sm,
      false STAGE_STAMP_ARG(L.stage_stamps));
}
template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class... IN>
constexpr int smem_glm_topk_route() {
  return topk_scratch_bytes<PARAMS::v[0], PARAMS::v[1]>();
}

} // namespace static_mk
