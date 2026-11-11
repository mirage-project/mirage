// tasks/topk_route.cuh (HIP) -- as include/mirage/static_megakernel/tasks/
// topk_route.cuh: the routing of one token t: add the router's NSPLIT partial
// sums of row t (K parts in slots, COMBINE_SLOTS, polled: device-scope loads),
// sigmoid, + score correction bias, top K of the NE experts, renormalise the K
// sigmoid scores -> K (expert, weight) pairs, written to `pairs` ([T][K], expert
// << 32 | weight bits; 0xFF-prefilled, polled by moe_experts: each pair one
// 8-byte device-scope store). No dry pass on gfx950 (`warm` unused). 256
// threads make the scores, then pick the top K by a radix select over the score
// bits (topk_pick_radix); CUDA's 8 warps rank each key through shared memory
// instead.
#pragma once
#include "../core.cuh"

namespace static_mk {

// the radix select's digit: 8 bits per round (MI355X 10-11, solo: 11 bits, 2048
// bins -> topk ends 17.4-18.2 us vs 14.8-16.1, e2e +6-10 us: wave 0's scan and
// search of 32 bins per lane cost more than the rounds saved)
constexpr int TOPK_DIGIT = 8;
// the pick's scratch words: the histogram, then st [4], cnt [2][16], ao [2][4]
constexpr int topk_pick_words() {
  return (1 << TOPK_DIGIT) + 44;
}
template <int NE, int K>
constexpr int topk_scratch_bytes_base() {
  return 2 * NE * 4 + 2 * (NE + 8 * K) * 4 + 2 * K * 4;
}
// the pick's scratch: the NE floats after the scores when they hold it, else
// its own region after the rest
template <int NE>
constexpr bool topk_pick_own_region() {
  return topk_pick_words() > NE;
}
template <int NE, int K>
constexpr int topk_scratch_bytes() {
  return topk_scratch_bytes_base<NE, K>() +
         (topk_pick_own_region<NE>() ? topk_pick_words() * 4 : 0);
}

// wave-wide inclusive prefix sum (lane order) in registers: DPP row shifts
// within each row of 16 lanes, then row broadcasts of lane 15 / 31 to the rows
// above (gfx9 DPP; disabled rows and lanes shifted in from outside a row add 0)
__device__ __forceinline__ int wave_incl_scan(int x) {
  x += __builtin_amdgcn_update_dpp(0, x, 0x111, 0xF, 0xF, true); // row_shr:1
  x += __builtin_amdgcn_update_dpp(0, x, 0x112, 0xF, 0xF, true); // row_shr:2
  x += __builtin_amdgcn_update_dpp(0, x, 0x114, 0xF, 0xF, true); // row_shr:4
  x += __builtin_amdgcn_update_dpp(0, x, 0x118, 0xF, 0xF, true); // row_shr:8
  x += __builtin_amdgcn_update_dpp(
      0, x, 0x142, 0xA, 0xF, false); // row_bcast:15 -> rows 1, 3
  x += __builtin_amdgcn_update_dpp(
      0, x, 0x143, 0xC, 0xF, false); // row_bcast:31 -> rows 2, 3
  return x;
}
// wave-wide AND and OR: within each row of 16 by DPP (quad_perm [1,0,3,2],
// [2,3,0,1], row_ror:4, row_ror:8), then the 4 rows' lanes 0 by readlane
__device__ __forceinline__ void wave_and_or(uint32_t &a, uint32_t &o) {
  int ai = (int)a, oi = (int)o;
  ai &= __builtin_amdgcn_update_dpp(ai, ai, 0xB1, 0xF, 0xF, false);
  oi |= __builtin_amdgcn_update_dpp(oi, oi, 0xB1, 0xF, 0xF, false);
  ai &= __builtin_amdgcn_update_dpp(ai, ai, 0x4E, 0xF, 0xF, false);
  oi |= __builtin_amdgcn_update_dpp(oi, oi, 0x4E, 0xF, 0xF, false);
  ai &= __builtin_amdgcn_update_dpp(ai, ai, 0x124, 0xF, 0xF, false);
  oi |= __builtin_amdgcn_update_dpp(oi, oi, 0x124, 0xF, 0xF, false);
  ai &= __builtin_amdgcn_update_dpp(ai, ai, 0x128, 0xF, 0xF, false);
  oi |= __builtin_amdgcn_update_dpp(oi, oi, 0x128, 0xF, 0xF, false);
  a = (uint32_t)(__builtin_amdgcn_readlane(ai, 0) &
                 __builtin_amdgcn_readlane(ai, 16) &
                 __builtin_amdgcn_readlane(ai, 32) &
                 __builtin_amdgcn_readlane(ai, 48));
  o = (uint32_t)(__builtin_amdgcn_readlane(oi, 0) |
                 __builtin_amdgcn_readlane(oi, 16) |
                 __builtin_amdgcn_readlane(oi, 32) |
                 __builtin_amdgcn_readlane(oi, 48));
}

// the top K of the NE orderable scores (the keys' high words) -> sel_s, in
// expert order; all 256 threads, expert e = STATIC_TIDX + 256 m. A radix select:
// first the bits every score shares (AND == OR over the block) are skipped (hb
// = the highest bit where scores differ). Round d takes the digit of bits lo ..
// hi (hi = hb - 8 d, lo = max(hi - 7, 0)): a 256-bin LDS histogram of it over
// the scores whose bits above hi equal P's; wave 0 (lane l: bins 255 - 4 l down
// to 252 - 4 l, an inclusive scan over the lanes in registers) finds the digit
// b where the scores above (a higher digit) stay below `need` and with bin b
// reach it; P takes b, need drops by the scores above. Bin b holding exactly
// `need` scores ends it (all of them picked); after the round with lo = 0, P is
// the K-th score itself: scores > P, then the lowest expert ids of those == P
// (the keys' order). Scratch (LDS): hist [256], st [4], cnt [2][16], ao [2][4].
// (MI355X 10-11, amd/bench/topk_select.hip, one workgroup per CU: 3.5 us on
// sigmoid + bias scores, 2.7 random; wave 3's 32-round binary search over the
// score bits, used before: 6.2 us, in the layer ~6.9; per instruction a wave
// waits ~9 cycles for a dependent add, ~64 for a dependent LDS read.)

__device__ __forceinline__ uint32_t topk_wave_max_u32(uint32_t v) {
  auto mx = [](uint32_t a, uint32_t b) { return a > b ? a : b; };
  v = mx(v, (uint32_t)__builtin_amdgcn_mov_dpp((int)v, 0xB1, 0xF, 0xF, false));  // quad_perm [1,0,3,2]
  v = mx(v, (uint32_t)__builtin_amdgcn_mov_dpp((int)v, 0x4E, 0xF, 0xF, false));  // quad_perm [2,3,0,1]
  v = mx(v, (uint32_t)__builtin_amdgcn_mov_dpp((int)v, 0x141, 0xF, 0xF, false)); // row_half_mirror
  v = mx(v, (uint32_t)__builtin_amdgcn_mov_dpp((int)v, 0x140, 0xF, 0xF, false)); // row_mirror
  uint32_t const r0 = (uint32_t)__builtin_amdgcn_readlane((int)v, 0), r1 = (uint32_t)__builtin_amdgcn_readlane((int)v, 16),
                 r2 = (uint32_t)__builtin_amdgcn_readlane((int)v, 32), r3 = (uint32_t)__builtin_amdgcn_readlane((int)v, 48);
  return mx(mx(r0, r1), mx(r2, r3));
}
template <int NE, int K>
__device__ __forceinline__ void topk_pick_wave(unsigned long long const *key_s, int *sel_s) {
  static_assert(NE == 896 && K == 16, "topk_pick_wave: 896 experts (14 per lane), K 16");
  if (STATIC_TIDX >= 64) {
    return;
  }
  int const lane = STATIC_TIDX;
  uint32_t hi[16], lo[16];
#pragma unroll
  for (int j = 0; j < 14; j++) {
    unsigned long long const k = key_s[lane + 64 * j];
    hi[j] = (uint32_t)(k >> 32);
    lo[j] = (uint32_t)k;
  }
  hi[14] = hi[15] = lo[14] = lo[15] = 0u;
  constexpr int NET[] = {0, 1, 2, 3, 0, 2, 1, 3, 1, 2, 4, 5, 6, 7, 4, 6, 5, 7, 5, 6, 0, 4, 2, 6, 2, 4, 1, 5, 3, 7, 3, 5, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 8, 10, 9, 11, 9, 10, 12, 13, 14, 15, 12, 14, 13, 15, 13, 14, 8, 12, 10, 14, 10, 12, 9, 13, 11, 15, 11, 13, 9, 10, 11, 12, 13, 14, 0, 8, 4, 12, 4, 8, 2, 10, 6, 14, 6, 10, 2, 4, 6, 8, 10, 12, 1, 9, 5, 13, 5, 9, 3, 11, 7, 15, 7, 11, 3, 5, 7, 9, 11, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
#pragma unroll
  for (int c = 0; c < (int)(sizeof(NET) / sizeof(int)); c += 2) {
    int const a = NET[c], b = NET[c + 1];
    bool const sw = hi[b] > hi[a] || (hi[b] == hi[a] && lo[b] > lo[a]);
    uint32_t const ha = hi[a], la = lo[a];
    hi[a] = sw ? hi[b] : ha;
    lo[a] = sw ? lo[b] : la;
    hi[b] = sw ? ha : hi[b];
    lo[b] = sw ? la : lo[b];
  }
  int my_e = 0;
#pragma unroll 1
  for (int r = 0; r < K; r++) {
    uint32_t const mh = topk_wave_max_u32(hi[0]);
    bool c = hi[0] == mh;
    unsigned long long bal = __ballot(c);
    if (__popcll(bal) > 1) { // (equal scores: the lower expert id, the larger low word)
      uint32_t const ml = topk_wave_max_u32(c ? lo[0] : 0u);
      c = c && lo[0] == ml;
      bal = __ballot(c);
    }
    int const w = __ffsll((long long)bal) - 1;
    uint32_t const wl = (uint32_t)__builtin_amdgcn_readlane((int)lo[0], w);
    if (lane == r) {
      my_e = (int)(0xFFFFFFFFu - wl);
    }
    bool const win = lane == w;
#pragma unroll
    for (int i = 0; i < 15; i++) {
      hi[i] = win ? hi[i + 1] : hi[i];
      lo[i] = win ? lo[i + 1] : lo[i];
    }
    hi[15] = win ? 0u : hi[15];
    lo[15] = win ? 0u : lo[15];
  }
  // expert order: lane r < K stores its expert at its rank among the K
  int rank = 0;
#pragma unroll
  for (int j = 0; j < K; j++) {
    int const ej = __builtin_amdgcn_readlane(my_e, j);
    rank += ej < my_e;
  }
  if (lane < K) {
    sel_s[rank] = my_e;
  }
}
template <int NE, int K>
__device__ __forceinline__ void topk_pick_radix(
    unsigned long long const *key_s, int *sel_s, uint32_t *scr) {
  constexpr int M = (NE + 255) / 256;
  static_assert(M * 4 <= 16 && K <= NE, "topk_route: at most 1024 experts");
  constexpr int DB = TOPK_DIGIT, NBIN = 1 << DB, PER = NBIN / 64;
  uint32_t *hist = scr, *st = scr + NBIN, *ao = scr + NBIN + 36;
  int *cnt = reinterpret_cast<int *>(scr + NBIN + 4);
  int const tid = task_tid_all(), ln = tid & 63, w = tid >> 6;
  uint32_t u[M];
  bool ok[M];
  uint32_t ka = 0xFFFFFFFFu, ko = 0;
#pragma unroll
  for (int m = 0; m < M; m++) {
    int const e = tid + 256 * m;
    ok[m] = e < NE;
    u[m] = ok[m] ? (uint32_t)(key_s[e] >> 32) : 0u;
    if (ok[m]) {
      ka &= u[m];
      ko |= u[m];
    }
  }
  wave_and_or(ka, ko);
  if (ln == 0) {
    ao[w] = ka;
    ao[4 + w] = ko;
  }
  uint32_t P = 0;
  int need = K, lo = 0, hb = -1;
  bool done = false;
#pragma unroll 1
  for (int d = 0;; d++) {
#pragma unroll
    for (int i = 0; i < NBIN / 256; i++) {
      hist[tid + 256 * i] = 0;
    }
    __syncthreads();
    if (d == 0) {
      uint4 const a4 = *reinterpret_cast<uint4 const *>(ao),
                  o4 = *reinterpret_cast<uint4 const *>(ao + 4);
      uint32_t const all_a = a4.x & a4.y & a4.z & a4.w,
                     diff = all_a ^ (o4.x | o4.y | o4.z | o4.w);
      if (!diff) { // every score the same: the lowest K experts
        P = all_a;
        break;
      }
      hb = 31 - __clz(diff);
      P = all_a & (hb >= 31 ? 0u : ~((2u << hb) - 1u));
    }
    int const hi = hb - DB * d;
    lo = hi - (DB - 1) > 0 ? hi - (DB - 1) : 0;
    uint32_t const dmask = (2u << (hi - lo)) - 1u;
#pragma unroll
    for (int m = 0; m < M; m++) {
      if (ok[m] && (d == 0 || (u[m] >> (hi + 1)) == (P >> (hi + 1)))) {
        atomicAdd(&hist[(u[m] >> lo) & dmask], 1u);
      }
    }
    __syncthreads();
    if (w == 0) { // lane l: bins NBIN - PER (l + 1) .. NBIN - PER l - 1
      uint32_t hv[PER];
#pragma unroll
      for (int q = 0; q < PER / 4; q++) {
        uint4 const h4 = *reinterpret_cast<uint4 const *>(
            hist + NBIN - PER * (ln + 1) + 4 * q);
        hv[4 * q] = h4.x;
        hv[4 * q + 1] = h4.y;
        hv[4 * q + 2] = h4.z;
        hv[4 * q + 3] = h4.w;
      }
      int c = 0;
#pragma unroll
      for (int k = 0; k < PER; k++) {
        c += (int)hv[k];
      }
      int const inc = wave_incl_scan(c); // scores in bins >= NBIN - PER (l + 1)
      int run = inc - c;                 // scores in bins >= NBIN - PER l
      if (run < need && inc >= need) {
        int b = 0, cb = 0;
        bool found = false;
#pragma unroll
        for (int k = 0; k < PER; k++) { // the lane's bins, highest first
          if (!found) {
            uint32_t const hk = hv[PER - 1 - k];
            if (run + (int)hk >= need) {
              b = NBIN - 1 - PER * ln - k;
              cb = (int)hk;
              found = true;
            } else {
              run += (int)hk;
            }
          }
        }
        st[0] = P | ((uint32_t)b << lo);
        st[1] = (uint32_t)(need - run);
        st[2] = cb == need - run;
      }
    }
    __syncthreads();
    P = st[0];
    need = (int)st[1];
    done = st[2] != 0;
    if (done || lo == 0) {
      break;
    }
  }
  // picks: gt (above P in the decided bits; with `done`, bin b too), eq (== P,
  // not done: the lowest `need` experts); positions in expert order from the
  // picks per group (m, wave) and the ballots in the wave
  bool gt[M], eq[M];
  unsigned long long bg[M], be[M];
#pragma unroll
  for (int m = 0; m < M; m++) {
    gt[m] = ok[m] && (done ? (u[m] >> lo) >= (P >> lo) : u[m] > P);
    eq[m] = ok[m] && !done && u[m] == P;
    bg[m] = __ballot(gt[m]);
    be[m] = __ballot(eq[m]);
  }
  if (ln == 0) {
#pragma unroll
    for (int m = 0; m < M; m++) {
      cnt[m * 4 + w] = __popcll(bg[m]);
      cnt[16 + m * 4 + w] = __popcll(be[m]);
    }
  }
  __syncthreads();
  unsigned long long const below = (1ull << ln) - 1ull;
  int gpre = 0, epre = 0; // the picks of the groups (m' < m, any wave)
#pragma unroll
  for (int m = 0; m < M; m++) {
    int4 const cg = *reinterpret_cast<int4 const *>(cnt + 4 * m),
               ce = *reinterpret_cast<int4 const *>(cnt + 16 + 4 * m);
    int const g4[4] = {cg.x, cg.y, cg.z, cg.w}, e4[4] = {ce.x, ce.y, ce.z, ce.w};
    int gb = gpre + __popcll(bg[m] & below), eb = epre + __popcll(be[m] & below);
#pragma unroll
    for (int ww = 0; ww < 3; ww++) {
      if (ww < w) {
        gb += g4[ww];
        eb += e4[ww];
      }
    }
    gpre += g4[0] + g4[1] + g4[2] + g4[3];
    epre += e4[0] + e4[1] + e4[2] + e4[3];
    if (gt[m] || (eq[m] && eb < need)) {
      sel_s[gb + (eb < need ? eb : need)] = tid + 256 * m;
    }
  }
}

// RET: the pairs not stored but returned (wave 0's lanes 0 .. K - 1: pair k = lane; every other thread ~0): moe_experts with a
// local route (tasks/moe_experts.cuh RSPLIT) makes the same pairs itself
template <int NSPLIT, int NE, int K, bool RET = false>
__device__ __forceinline__ unsigned long long topk_route_task(float const *bias,
                                                              float *lpart,
                                                              unsigned long long *pairs,
                                                              int t,
                                                              char *sm,
                                                              bool warm STAGE_STAMP_PARAM) {
  unsigned long long pair = ~0ull;
  __syncthreads(); // the scratch below is LDS a previous task may still read
  constexpr int NCAND = 8 * K; // cand_s's size (the layout of topk_scratch_bytes)
  static_assert(NE >= 300, "topk_route: the pick's scratch in [NE] floats");
  static_assert(NE % 64 == 0 && NE / 4 <= 256 && K <= NE && NCAND <= 256,
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
                (uint32_t)((((size_t)p_ * T + t) * NE + STATIC_TIDX * 4) * 4)));
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
        float const sg = 1.0f / (1.0f + __expf(-lg4[k]));
        sc_s[e] = sg;
        uint32_t u = __float_as_uint(sg + bs4[k]);
        u = (u & 0x80000000u) ? ~u : (u | 0x80000000u); // orderable
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
    static_assert(K == 16, "topk_route: the weights' sum is one DPP row of 16");
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
        float const w = sc / sum;
        if constexpr (RET) {
          pair = ((unsigned long long)(unsigned)e << 32) | (unsigned long long)__float_as_uint(w);
        } else {
          __hip_atomic_store(
              pairs + t * K + STATIC_TIDX,
              ((unsigned long long)(unsigned)e << 32) |
                  (unsigned long long)__float_as_uint(w),
              __ATOMIC_RELAXED,
              __HIP_MEMORY_SCOPE_AGENT);
        }
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
  return pair;
}

// topk_route task x = token x; adds the router's LOGITS::y partial sums;
// PARAMS {NE, K}; BUF_SLOTS {the score correction bias ([NE] fp32)}
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
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_topk_route() {
  return topk_scratch_bytes<PARAMS::v[0], PARAMS::v[1]>();
}

} // namespace static_mk
