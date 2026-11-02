// Microbenchmark: topk_route's top-16-of-896 selection (tasks/topk_route.cuh) alone, in a small kernel, to separate the loop's
// own cost from the layer kernel's (MI355X 10-10: in the layer, wave 3's 32-round binary search took ~6.9 us at 2.3 GHz, ~440
// cycles per count of 14 compares).
//
// One workgroup of 256 threads per CU (96 KB LDS: one per CU), each picks the top 16 of its own 896 random orderable keys with
// the same code as the layer (per-lane compare counts + 4 bit-ballots, wave 3, 32 rounds), stamps the pick (s_memrealtime and
// the shader clock), and checks the 16 against a CPU sort.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 topk_select.hip -o topk_select
// Run:   ./topk_select   -> per-variant pick time (us, median over workgroups) and shader GHz, then a JSON line
#include <cmath>
#include <cstring>
#include <random>
#include <string>

#include "common.h"

constexpr int NE = 896, K = 16, Q = NE / 64;

// the binary search on a lane's 14 scores (expert = 14 lane + q), for the candidates' fallback
__device__ inline void pick_binary_lanes(const uint32_t *u, int *sel_s, int ln) {
  auto count_ge = [&](uint32_t v) {
    int c = 0;
#pragma unroll
    for (int q = 0; q < Q; q++) c += u[q] >= v ? 1 : 0;
    int tot = 0;
#pragma unroll
    for (int b = 0; b < 4; b++) tot += __popcll(__ballot((c >> b) & 1)) << b;
    return tot;
  };
  uint32_t th = 0;
#pragma unroll 1
  for (int bit = 31; bit >= 0; bit--) {
    uint32_t const cand = th | (1u << bit);
    if (count_ge(cand) >= K) th = cand;
  }
  int need = K - (th == 0xFFFFFFFFu ? 0 : count_ge(th + 1));
  unsigned long long const below = (1ull << ln) - 1ull;
  // expert order = lane-major: this lane's picks after every lower lane's; per lane: > th first in q order, then == th
  int gt = 0, eq = 0;
#pragma unroll
  for (int q = 0; q < Q; q++) {
    gt += u[q] > th;
    eq += u[q] == th;
  }
  int eq_before = 0, gt_before = 0;
#pragma unroll
  for (int bt = 0; bt < 4; bt++) {
    eq_before += __popcll(__ballot((eq >> bt) & 1) & below) << bt;
    gt_before += __popcll(__ballot((gt >> bt) & 1) & below) << bt;
  }
  int take_eq = need - eq_before;   // this lane's equal scores taken: the lowest expert ids first
  take_eq = take_eq < 0 ? 0 : (take_eq > eq ? eq : take_eq);
  int eq_taken_before = eq_before < need ? eq_before : need;
  int pos = gt_before + eq_taken_before;
#pragma unroll
  for (int q = 0; q < Q; q++) {
    bool pick = u[q] > th;
    if (u[q] == th && take_eq > 0) {
      pick = true;
      take_eq--;
    }
    if (pick) sel_s[pos++] = Q * ln + q;
  }
}

// variant 0: one wave's 32-round binary search over the score bits (the layer's code)
__device__ inline void pick_binary(const uint32_t *key_s, int *sel_s) {
  int const ln = threadIdx.x - 192;
  uint32_t u[Q];
#pragma unroll
  for (int q = 0; q < Q; q++) u[q] = key_s[ln + 64 * q];
  auto count_ge = [&](uint32_t v) {
    int c = 0;
#pragma unroll
    for (int q = 0; q < Q; q++) c += u[q] >= v ? 1 : 0;
    int tot = 0;
#pragma unroll
    for (int b = 0; b < 4; b++) tot += __popcll(__ballot((c >> b) & 1)) << b;
    return tot;
  };
  uint32_t th = 0;
#pragma unroll 1
  for (int bit = 31; bit >= 0; bit--) {
    uint32_t const cand = th | (1u << bit);
    if (count_ge(cand) >= K) th = cand;
  }
  int need = K - (th == 0xFFFFFFFFu ? 0 : count_ge(th + 1));
  int pos = 0;
  unsigned long long const below = (1ull << ln) - 1ull;
#pragma unroll
  for (int q = 0; q < Q; q++) {
    unsigned long long const eq = __ballot(u[q] == th);
    bool const pick = u[q] > th || (u[q] == th && __popcll(eq & below) < need);
    int const ne = __popcll(eq);
    need -= ne < need ? ne : need;
    unsigned long long const pm = __ballot(pick);
    if (pick) sel_s[pos + __popcll(pm & below)] = ln + 64 * q;
    pos += __popcll(pm);
  }
}

// variant 1: candidates. Lane l holds experts 14 l .. 14 l + 13 (7 8-byte LDS reads). The K-th largest lane maximum is a lower
// bound lo of the K-th largest score (K lanes have a score >= it). The scores >= lo (the candidates: at least K, few) are
// ranked against each other (score, then the lower expert id first); rank < K -> sel_s[rank]. Every step is parallel over the
// lanes with fixed-length loops of 16-byte LDS reads (no chain of dependent reads); more than 64 candidates: variant 0.
// Scratch: mx [64] u32, rk [64] u32, cu [64] u32, ce [64] int.
__device__ inline void pick_candidates(const uint32_t *key_s, int *sel_s, uint32_t *scr) {
  int const ln = threadIdx.x - 192;
  uint32_t *mx = scr, *rk = scr + 64, *cu = scr + 128;
  int *ce = reinterpret_cast<int *>(scr + 192);
  uint32_t u[Q];
#pragma unroll
  for (int q = 0; q < Q; q += 2) {
    uint2 const v = *reinterpret_cast<const uint2 *>(key_s + Q * ln + q);
    u[q] = v.x;
    u[q + 1] = v.y;
  }
  uint32_t m = 0;
#pragma unroll
  for (int q = 0; q < Q; q++) m = u[q] > m ? u[q] : m;
  mx[ln] = m;
  cu[ln] = 0;                 // the candidates' unused slots: never ranked above anything
  ce[ln] = 0x7FFFFFFF;
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");   // this wave's LDS writes done (LDS keeps a wave's order)
  // rank of this lane's max among the 64 (ties: the lower lane first)
  int r = 0;
#pragma unroll
  for (int j = 0; j < 64; j += 4) {
    uint4 const o = *reinterpret_cast<const uint4 *>(mx + j);
    r += (o.x > m || (o.x == m && j + 0 < ln)) + (o.y > m || (o.y == m && j + 1 < ln)) +
         (o.z > m || (o.z == m && j + 2 < ln)) + (o.w > m || (o.w == m && j + 3 < ln));
  }
  rk[r] = m;
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  uint32_t const lo = rk[K - 1];
  // the candidates, compacted in expert order: this lane's count, then an exclusive prefix over the lanes by ballots of its
  // bits (a lane holds at most 14)
  int c = 0;
#pragma unroll
  for (int q = 0; q < Q; q++) c += u[q] >= lo;
  unsigned long long const below = (1ull << ln) - 1ull;
  int at = 0, nc = 0;
#pragma unroll
  for (int bt = 0; bt < 4; bt++) {
    unsigned long long const bm = __ballot((c >> bt) & 1);
    at += __popcll(bm & below) << bt;
    nc += __popcll(bm) << bt;
  }
  if (nc > 64) {   // rare: fall back
    pick_binary_lanes(u, sel_s, ln);
    return;
  }
#pragma unroll
  for (int q = 0; q < Q; q++) {
    if (u[q] >= lo) {
      cu[at] = u[q];
      ce[at] = Q * ln + q;
      at++;
    }
  }
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  uint32_t const mu = cu[ln];
  int const me = ce[ln];
  int rr = 0;
#pragma unroll
  for (int j = 0; j < 64; j += 4) {
    uint4 const ou = *reinterpret_cast<const uint4 *>(cu + j);
    int4 const oe = *reinterpret_cast<const int4 *>(ce + j);
    rr += (ou.x > mu || (ou.x == mu && oe.x < me)) + (ou.y > mu || (ou.y == mu && oe.y < me)) +
          (ou.z > mu || (ou.z == mu && oe.z < me)) + (ou.w > mu || (ou.w == mu && oe.w < me));
  }
  if (ln < nc && rr < K) sel_s[rr] = me;
}

// wave-wide inclusive prefix sum (lane order) in registers: DPP row shifts within each row of 16 lanes, then row broadcasts
// of lane 15 / 31 to the rows above (gfx9 DPP; disabled rows and lanes shifted in from outside a row add 0)
__device__ inline int wave_incl_scan(int x) {
  x += __builtin_amdgcn_update_dpp(0, x, 0x111, 0xF, 0xF, true);   // row_shr:1
  x += __builtin_amdgcn_update_dpp(0, x, 0x112, 0xF, 0xF, true);   // row_shr:2
  x += __builtin_amdgcn_update_dpp(0, x, 0x114, 0xF, 0xF, true);   // row_shr:4
  x += __builtin_amdgcn_update_dpp(0, x, 0x118, 0xF, 0xF, true);   // row_shr:8
  x += __builtin_amdgcn_update_dpp(0, x, 0x142, 0xA, 0xF, false);  // row_bcast:15 -> rows 1, 3
  x += __builtin_amdgcn_update_dpp(0, x, 0x143, 0xC, 0xF, false);  // row_bcast:31 -> rows 2, 3
  return x;
}
// wave-wide AND / OR: within each row of 16 by DPP, then the 4 rows' lanes 0 by readlane
__device__ inline void wave_and_or(uint32_t &a, uint32_t &o) {
  int ai = (int)a, oi = (int)o;
  // quad_perm [1,0,3,2], [2,3,0,1], row_ror:4, row_ror:8
  ai &= __builtin_amdgcn_update_dpp(ai, ai, 0xB1, 0xF, 0xF, false);
  oi |= __builtin_amdgcn_update_dpp(oi, oi, 0xB1, 0xF, 0xF, false);
  ai &= __builtin_amdgcn_update_dpp(ai, ai, 0x4E, 0xF, 0xF, false);
  oi |= __builtin_amdgcn_update_dpp(oi, oi, 0x4E, 0xF, 0xF, false);
  ai &= __builtin_amdgcn_update_dpp(ai, ai, 0x124, 0xF, 0xF, false);
  oi |= __builtin_amdgcn_update_dpp(oi, oi, 0x124, 0xF, 0xF, false);
  ai &= __builtin_amdgcn_update_dpp(ai, ai, 0x128, 0xF, 0xF, false);
  oi |= __builtin_amdgcn_update_dpp(oi, oi, 0x128, 0xF, 0xF, false);
  a = (uint32_t)(__builtin_amdgcn_readlane(ai, 0) & __builtin_amdgcn_readlane(ai, 16) & __builtin_amdgcn_readlane(ai, 32) &
                 __builtin_amdgcn_readlane(ai, 48));
  o = (uint32_t)(__builtin_amdgcn_readlane(oi, 0) | __builtin_amdgcn_readlane(oi, 16) | __builtin_amdgcn_readlane(oi, 32) |
                 __builtin_amdgcn_readlane(oi, 48));
}

// variant 2: radix select on all 4 waves. Expert e = tid + 256 m (m < 4). First the bits every key shares (AND == OR over
// the block) are skipped: hb = the highest bit where keys differ. Round d takes the digit of bits lo .. hi (hi = hb - 8 d,
// lo = max(hi - 7, 0)): a 256-bin LDS histogram of it over the keys whose bits above hi equal P's; wave 0 (lane l: bins
// 255 - 4 l down to 252 - 4 l, an inclusive scan over the lanes in registers) finds the digit b where the keys above (a
// higher digit) stay below `need` and with bin b reach it; P takes b, need drops by the keys above. Bin b holding exactly
// `need` keys ends it (all of them picked); after the round with lo = 0, P is the K-th key itself: keys > P, then the
// lowest experts of those == P. Picks go out in expert order (counts per (m, wave) through LDS, ballots in the wave), as
// variant 0. Scratch: hist [256], st [4], cnt [2][16], ao [2][4].
// FAST (variant 4): the same rounds; the picks then go out as: every pick above P appended to sel_s by an LDS counter, the
// ties at P (== P, not done) to a list of at most 64 (more: the ballot path below), wave 0 takes the `need` lowest ids of
// the ties, and lanes 0..15 rank the 16 by expert id within their DPP row (15 rotations) and store sel_s in expert order.
// Scratch words 300, 301: the two counters; 304 ..: the ties.
template <bool FAST>
__device__ inline void pick_radix(const uint32_t *key_s, int *sel_s, uint32_t *scr, long long *ph) {
  int nph = 0;
  auto stamp = [&]() {
    if (threadIdx.x == 0 && nph < 15) ph[nph] = clock64();
    nph++;
  };
  constexpr int M = (NE + 255) / 256;
  uint32_t *hist = scr, *st = scr + 256, *ao = scr + 292;
  int *cnt = reinterpret_cast<int *>(scr + 260);
  int const tid = threadIdx.x, ln = tid & 63, w = tid >> 6;
  uint32_t u[M];
  bool ok[M];
  uint32_t ka = 0xFFFFFFFFu, ko = 0;
#pragma unroll
  for (int m = 0; m < M; m++) {
    int const e = tid + 256 * m;
    ok[m] = e < NE;
    u[m] = ok[m] ? key_s[e] : 0u;
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
  stamp();   // 0: keys in registers, the wave's AND / OR
  uint32_t P = 0;
  int need = K, lo = 0, hb = -1;
  bool done = false;
  if (FAST && tid == 0) {
    scr[300] = 0;
    scr[301] = 0;
  }
#pragma unroll 1
  for (int d = 0;; d++) {
    hist[tid] = 0;
    __syncthreads();
    if (d == 0) {
      uint4 const a4 = *reinterpret_cast<const uint4 *>(ao), o4 = *reinterpret_cast<const uint4 *>(ao + 4);
      uint32_t const diff = (a4.x & a4.y & a4.z & a4.w) ^ (o4.x | o4.y | o4.z | o4.w);
      hb = diff ? 31 - __clz(diff) : -1;
      P = (a4.x & a4.y & a4.z & a4.w) & (hb >= 31 ? 0u : ~((2u << hb) - 1u));
      if (hb < 0) {   // every key the same: the lowest K experts
        P = a4.x & a4.y & a4.z & a4.w;
        break;
      }
    }
    int const hi = hb - 8 * d;
    lo = hi - 7 > 0 ? hi - 7 : 0;
    uint32_t const dmask = (2u << (hi - lo)) - 1u;
    stamp();   // round start + 1: bins zeroed
#pragma unroll
    for (int m = 0; m < M; m++) {
      if (ok[m] && (d == 0 || (u[m] >> (hi + 1)) == (P >> (hi + 1)))) {
        atomicAdd(&hist[(u[m] >> lo) & dmask], 1u);
      }
    }
    __syncthreads();
    stamp();   // + 2: histogram
    if (w == 0) {
      uint4 const h = *reinterpret_cast<const uint4 *>(hist + 252 - 4 * ln);
      int const c = (int)(h.x + h.y + h.z + h.w);
      int const inc = wave_incl_scan(c);   // keys in bins >= 252 - 4 ln
      int run = inc - c;                    // keys in bins >= 256 - 4 ln
      if (run < need && inc >= need) {
        uint32_t const hh[4] = {h.w, h.z, h.y, h.x};   // bins 255 - 4 ln down to 252 - 4 ln
        int b = 0, cb = 0;
        bool found = false;
#pragma unroll
        for (int k = 0; k < 4; k++) {
          if (!found) {
            if (run + (int)hh[k] >= need) {
              b = 255 - 4 * ln - k;
              cb = (int)hh[k];
              found = true;
            } else {
              run += (int)hh[k];
            }
          }
        }
        st[0] = P | ((uint32_t)b << lo);
        st[1] = (uint32_t)(need - run);
        st[2] = cb == need - run;
      }
    }
    __syncthreads();
    stamp();   // + 3: digit found
    P = st[0];
    need = (int)st[1];
    done = st[2] != 0;
    if (done || lo == 0) break;
  }
  // picks: gt (above P in the decided bits; with `done`, bin b too), eq (== P, not done: the lowest `need` experts)
  bool gt[M], eq[M];
  unsigned long long bg[M], be[M];
  if constexpr (FAST) {
    int *const ties = reinterpret_cast<int *>(scr + 304);
#pragma unroll
    for (int m = 0; m < M; m++) {
      bool const g_ = ok[m] && (done ? (u[m] >> lo) >= (P >> lo) : u[m] > P);
      bool const e_ = ok[m] && !done && u[m] == P;
      if (g_) {
        sel_s[atomicAdd(&scr[300], 1u)] = tid + 256 * m;
      }
      if (e_) {
        uint32_t const q = atomicAdd(&scr[301], 1u);
        if (q < 64) ties[q] = tid + 256 * m;
      }
    }
    __syncthreads();
    if (w == 0) {
      int const ng = (int)scr[300], nq = (int)scr[301];
      if (nq <= 64) {   // the `need` lowest ids of the ties after the ng picks above P
        if (ln < nq) {
          int const me = ties[ln];
          int r = 0;
          for (int j = 0; j < nq; j++) r += ties[j] < me;
          if (r < need) sel_s[ng + r] = me;
        }
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        // lanes 0..15 (DPP row 0): rank by expert id, store in that order
        int const e = ln < K ? sel_s[ln] : 0x7FFFFFFF;
        int r = 0;
#define RANK_ROT(k) r += __builtin_amdgcn_update_dpp(0, e, 0x120 + (k), 0xF, 0xF, false) < e ? 1 : 0;
        RANK_ROT(1) RANK_ROT(2) RANK_ROT(3) RANK_ROT(4) RANK_ROT(5) RANK_ROT(6) RANK_ROT(7) RANK_ROT(8)
        RANK_ROT(9) RANK_ROT(10) RANK_ROT(11) RANK_ROT(12) RANK_ROT(13) RANK_ROT(14) RANK_ROT(15)
#undef RANK_ROT
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        if (ln < K) sel_s[r] = e;
      }
    }
    if (scr[301] <= 64) {   // uniform: every wave read the same count after the barrier
      stamp();
      if (threadIdx.x == 0) ph[15] = nph;
      return;
    }
  }
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
  int4 cg[M], ce[M];   // group gi = 4 m + wave: its picks, read once
#pragma unroll
  for (int m = 0; m < M; m++) {
    cg[m] = *reinterpret_cast<const int4 *>(cnt + 4 * m);
    ce[m] = *reinterpret_cast<const int4 *>(cnt + 16 + 4 * m);
  }
  unsigned long long const below = (1ull << ln) - 1ull;
  int gpre = 0, epre = 0;   // the picks of the groups 4 m' + * for m' < m
#pragma unroll
  for (int m = 0; m < M; m++) {
    int const g4[4] = {cg[m].x, cg[m].y, cg[m].z, cg[m].w}, e4[4] = {ce[m].x, ce[m].y, ce[m].z, ce[m].w};
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
    if (gt[m] || (eq[m] && eb < need)) sel_s[gb + (eb < need ? eb : need)] = tid + 256 * m;
  }
  stamp();   // last: picks written
  if (threadIdx.x == 0) ph[15] = nph;
}

// variant 5: per-wave top K by K rounds of a wave max, then wave 0 ranks the 4 K candidates. Wave w's lanes hold experts
// e = tid + 256 m (m < 4, e < NE). Round r: each lane's head (its largest key not taken yet; equal keys in a lane: the lower
// e, i.e. the lower m), the wave's largest head by DPP max within the rows of 16 and readlane over the 4 rows, then (ties
// on the key) the lowest e among those lanes by DPP min; that lane takes it; lane 0 writes (key, e) as candidate w K + r.
// Wave 0: lane l < 4 K holds candidate l, its rank = the candidates ahead of it (larger key, or the same key and a lower e;
// readlane loop), the K with rank < K placed in expert order (a second readlane loop). No LDS atomics, two barriers.
__device__ inline uint32_t wave_max_u32(uint32_t v) {
  int x = (int)v;
  x = (int)max((uint32_t)x, (uint32_t)__builtin_amdgcn_update_dpp(x, x, 0xB1, 0xF, 0xF, false));
  x = (int)max((uint32_t)x, (uint32_t)__builtin_amdgcn_update_dpp(x, x, 0x4E, 0xF, 0xF, false));
  x = (int)max((uint32_t)x, (uint32_t)__builtin_amdgcn_update_dpp(x, x, 0x124, 0xF, 0xF, false));
  x = (int)max((uint32_t)x, (uint32_t)__builtin_amdgcn_update_dpp(x, x, 0x128, 0xF, 0xF, false));
  uint32_t const a = (uint32_t)__builtin_amdgcn_readlane(x, 0), b = (uint32_t)__builtin_amdgcn_readlane(x, 16),
                 c = (uint32_t)__builtin_amdgcn_readlane(x, 32), d = (uint32_t)__builtin_amdgcn_readlane(x, 48);
  return max(max(a, b), max(c, d));
}
__device__ inline uint32_t wave_min_u32(uint32_t v) {
  int x = (int)v;
  x = (int)min((uint32_t)x, (uint32_t)__builtin_amdgcn_update_dpp(x, x, 0xB1, 0xF, 0xF, false));
  x = (int)min((uint32_t)x, (uint32_t)__builtin_amdgcn_update_dpp(x, x, 0x4E, 0xF, 0xF, false));
  x = (int)min((uint32_t)x, (uint32_t)__builtin_amdgcn_update_dpp(x, x, 0x124, 0xF, 0xF, false));
  x = (int)min((uint32_t)x, (uint32_t)__builtin_amdgcn_update_dpp(x, x, 0x128, 0xF, 0xF, false));
  uint32_t const a = (uint32_t)__builtin_amdgcn_readlane(x, 0), b = (uint32_t)__builtin_amdgcn_readlane(x, 16),
                 c = (uint32_t)__builtin_amdgcn_readlane(x, 32), d = (uint32_t)__builtin_amdgcn_readlane(x, 48);
  return min(min(a, b), min(c, d));
}
__device__ inline void pick_argmax(const uint32_t *key_s, int *sel_s, uint32_t *scr) {
  constexpr int M = (NE + 255) / 256;
  int const tid = threadIdx.x, ln = tid & 63, w = tid >> 6;
  uint32_t u[M];
  bool live[M];
#pragma unroll
  for (int m = 0; m < M; m++) {
    int const e = tid + 256 * m;
    live[m] = e < NE;
    u[m] = live[m] ? key_s[e] : 0u;
  }
  uint32_t *cand = scr; // [4 K][2]: key, e
#pragma unroll 1
  for (int r = 0; r < K; r++) {
    // the lane's head: the largest live key (the first m among equal ones)
    uint32_t hk = 0;
    int hm = -1;
#pragma unroll
    for (int m = 0; m < M; m++) {
      if (live[m] && (hm < 0 || u[m] > hk)) {
        hk = u[m];
        hm = m;
      }
    }
    uint32_t const he = hm >= 0 ? (uint32_t)(tid + 256 * hm) : 0xFFFFFFFFu;
    // (a lane with no live key: key 0 and e 0xFFFFFFFF, never the pick while the wave has 64 x M >= K live keys)
    uint32_t const mk = wave_max_u32(hk);
    uint32_t const me = wave_min_u32(hm >= 0 && hk == mk ? he : 0xFFFFFFFFu);
    if (hm >= 0 && hk == mk && he == me) {
#pragma unroll
      for (int m = 0; m < M; m++) {
        if (m == hm) live[m] = false;
      }
    }
    if (ln == 0) {
      cand[2 * (w * K + r)] = mk;
      cand[2 * (w * K + r) + 1] = me;
    }
  }
  __syncthreads();
  if (w == 0) {
    uint32_t const ck = cand[2 * ln], ce = cand[2 * ln + 1]; // (4 K = 64 candidates: one per lane)
    int rank = 0;
#pragma unroll
    for (int j = 0; j < 4 * K; j++) {
      uint32_t const kj = (uint32_t)__builtin_amdgcn_readlane((int)ck, j), ej = (uint32_t)__builtin_amdgcn_readlane((int)ce, j);
      rank += (kj > ck || (kj == ck && ej < ce)) ? 1 : 0;
    }
    bool const sel = rank < K;
    int pos = 0;
#pragma unroll
    for (int j = 0; j < 4 * K; j++) {
      bool const sj = __builtin_amdgcn_readlane((int)sel, j) != 0;
      uint32_t const ej = (uint32_t)__builtin_amdgcn_readlane((int)ce, j);
      pos += (sj && ej < ce) ? 1 : 0;
    }
    if (sel) sel_s[pos] = (int)ce;
  }
  __syncthreads();
}

template <int V>
__global__ void __launch_bounds__(256, 1) pick_kernel(const uint32_t *keys_in, int *sel_out, unsigned long long *t_out) {
  extern __shared__ char lds[];
  uint32_t *key_s = reinterpret_cast<uint32_t *>(lds);
  int *sel_s = reinterpret_cast<int *>(key_s + NE);
  uint32_t *scr = reinterpret_cast<uint32_t *>(sel_s + 64);
  for (int i = threadIdx.x; i < NE; i += 256) key_s[i] = keys_in[(size_t)blockIdx.x * NE + i];
  __syncthreads();
  __shared__ long long ph[16];
  if (V == 3) {   // variant 3: the radix pick once untimed, so the timed run below finds its code in the instruction cache
    pick_radix<false>(key_s, sel_s, scr, ph);
    __syncthreads();
  }
  long long const c0 = clock64(), t0 = wall_clock64();
  if (V == 5) {
    pick_argmax(key_s, sel_s, scr);
  } else if (V == 2 || V == 3 || V == 4) {
    if (V == 4) {
      pick_radix<true>(key_s, sel_s, scr, ph);
    } else {
      pick_radix<false>(key_s, sel_s, scr, ph);
    }
  } else if (threadIdx.x >= 192) {
    if (V == 0) {
      pick_binary(key_s, sel_s);
    } else {
      pick_candidates(key_s, sel_s, scr);
    }
  }
  __syncthreads();
  long long const c1 = clock64(), t1 = wall_clock64();
  if (threadIdx.x < K) sel_out[blockIdx.x * K + threadIdx.x] = sel_s[threadIdx.x];
  if (threadIdx.x == 0) {
    t_out[2 * blockIdx.x] = (unsigned long long)(t1 - t0);
    t_out[2 * blockIdx.x + 1] = (unsigned long long)(c1 - c0);
  }
  if ((V >= 2) && threadIdx.x < 16) {   // phase clocks after the start (cycles), [15]: their count
    t_out[2 * gridDim.x + 16 * blockIdx.x + threadIdx.x] =
        threadIdx.x == 15 ? (unsigned long long)ph[15] : (unsigned long long)(ph[threadIdx.x] - c0);
  }
}

int main() {
  CHECK(hipSetDevice(0));
  print_device(0);
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int const nwg = prop.multiProcessorCount;
  double const tpn = wall_ticks_per_ns(0);
  std::mt19937 rng(7);
  std::vector<uint32_t> keys((size_t)nwg * NE);
  // orderable keys; workgroup w % 3: 0 random 30-bit, 1 only 200 values (many ties, the candidates' fallback), 2 as
  // topk_route's: sigmoid(logit) + bias, logit ~ N(0, 1.5), bias ~ N(0, 0.05), made orderable the same way
  std::normal_distribution<float> nlog(0.f, 1.5f), nbias(0.f, 0.05f);
  for (size_t i = 0; i < keys.size(); i++) {
    int const kind = (int)((i / NE) % 3);
    if (kind == 2) {
      float const sc = 1.f / (1.f + std::exp(-nlog(rng))) + nbias(rng);
      uint32_t uu;
      memcpy(&uu, &sc, 4);
      keys[i] = (uu & 0x80000000u) ? ~uu : (uu | 0x80000000u);
    } else {
      keys[i] = 0x80000000u | (kind == 1 ? (uint32_t)(rng() % 200) : (rng() & 0x3FFFFFFFu));
    }
  }
  uint32_t *dk;
  int *ds;
  unsigned long long *dt;
  CHECK(hipMalloc(&dk, keys.size() * 4));
  CHECK(hipMalloc(&ds, (size_t)nwg * K * 4));
  CHECK(hipMalloc(&dt, (size_t)nwg * 16 + (size_t)nwg * 128));
  CHECK(hipMemcpy(dk, keys.data(), keys.size() * 4, hipMemcpyHostToDevice));
  using Kernel = void (*)(const uint32_t *, int *, unsigned long long *);
  Kernel const variants[6] = {pick_kernel<0>, pick_kernel<1>, pick_kernel<2>, pick_kernel<3>, pick_kernel<4>, pick_kernel<5>};
  const char *names[6] = {"binary search (one wave, 32 rounds)", "candidates (lane maxima, then rank)",
                          "radix select (4 waves, 8-bit digits, LDS histogram)",
                          "radix select, its code run once before (instruction cache warm)",
                          "radix select, picks by LDS counter + DPP rank of the 16 (FAST)",
                          "per-wave top 16 by wave max, then rank of the 64 (argmax)"};
  std::string json;
  for (int v = 0; v < 6; v++) {
    set_max_lds((const void *)variants[v], ONE_PER_CU_LDS);
    std::vector<double> us, ghz, us_rand, us_ties, us_real;
    int wrong = 0;
    for (int rep = 0; rep < 6; rep++) {
      hipLaunchKernelGGL(variants[v], dim3(nwg), dim3(256), ONE_PER_CU_LDS, 0, dk, ds, dt);
      CHECK(hipGetLastError());
      CHECK(hipDeviceSynchronize());
      if (rep == 0) continue;   // the first launch: code fetch
      std::vector<unsigned long long> t((size_t)nwg * 2);
      std::vector<int> sel((size_t)nwg * K);
      CHECK(hipMemcpy(t.data(), dt, t.size() * 8, hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(sel.data(), ds, sel.size() * 4, hipMemcpyDeviceToHost));
      for (int w = 0; w < nwg; w++) {
        us.push_back(t[2 * w] / tpn / 1e3);
        (w % 3 == 0 ? us_rand : w % 3 == 1 ? us_ties : us_real).push_back(t[2 * w] / tpn / 1e3);
        ghz.push_back((double)t[2 * w + 1] / (t[2 * w] / tpn));
        // the reference: the 16 largest keys (ties: lower expert id first)
        std::vector<int> idx(NE);
        for (int e = 0; e < NE; e++) idx[e] = e;
        const uint32_t *kw = &keys[(size_t)w * NE];
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                          [&](int a, int b) { return kw[a] != kw[b] ? kw[a] > kw[b] : a < b; });
        std::vector<int> a(sel.begin() + w * K, sel.begin() + (w + 1) * K), b(idx.begin(), idx.begin() + K);
        if (v >= 2) wrong += !std::is_sorted(a.begin(), a.end());   // the radix variants: sel_s in expert order
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        wrong += a != b;
      }
    }
    printf("pick, %s, small kernel: median %.2f us (random scores %.2f, many ties %.2f, sigmoid + bias %.2f) at %.2f GHz; "
           "wrong picks %d of %zu\n", names[v], median(us), median(us_rand), median(us_ties), median(us_real), median(ghz), wrong,
           us.size());
    if (v >= 2 && v <= 4) {   // the last launch's phase clocks: medians over the workgroups, random and tie-heavy apart
      std::vector<unsigned long long> ph((size_t)nwg * 16);
      CHECK(hipMemcpy(ph.data(), dt + 2 * nwg, ph.size() * 8, hipMemcpyDeviceToHost));
      for (int kind = 0; kind < 3; kind++) {
        int const n = (int)ph[16 * kind + 15];
        printf("  radix phases (%s, %d stamps; cycles after the start): ", kind == 0 ? "random" : kind == 1 ? "many ties" :
               "sigmoid + bias", n);
        for (int i = 0; i < n && i < 15; i++) {
          std::vector<double> c;
          for (int w = kind; w < nwg; w += 3) c.push_back((double)ph[16 * w + i]);
          printf("%.0f ", median(c));
        }
        printf("\n");
      }
    }
    char row[160];
    snprintf(row, sizeof(row), "%s{\"variant\": %d, \"pick_us\": %.3f, \"ghz\": %.3f, \"wrong\": %d}", v ? ", " : "", v, median(us),
             median(ghz), wrong);
    json += row;
  }
  printf("{\"bench\": \"topk_select\", \"rows\": [%s]}\n", json.c_str());
  return 0;
}
